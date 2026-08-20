use anyhow::{bail, Result};

use crate::accelerate::sgemm_nt;
use crate::config::TrainConfig;
use crate::device::SovereignDevice;
use crate::kan::{KanEvalMode, NamedBlob, TernaryKanLinear};
use crate::mixers::{
    causal_shift, causal_shift_backward, embed_lookup, embed_scatter, randn, rmsnorm,
    rmsnorm_backward, CausalAttention,
};
use crate::optim::masked_ce_entropy;
use crate::quant::TernaryHist;
use crate::tensor::SovereignTensor;
use crate::tokenizer::{BpeTokenizer, StreamDecoder};

pub struct RmsNorm {
    pub weight: SovereignTensor,
    pub grad: Vec<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(d: usize) -> Result<Self> {
        Ok(Self {
            weight: SovereignTensor::fill(vec![d], 1.0)?,
            grad: vec![0.0; d],
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &[f32], n: usize, d: usize) -> Result<Vec<f32>> {
        rmsnorm(x, n, d, self.weight.as_slice(), self.eps)
    }
}

pub enum Mixer {
    Shift,
    Attn(CausalAttention),
}

impl Mixer {
    pub fn forward(&self, x: &[f32], b: usize, t: usize, d: usize) -> Result<Vec<f32>> {
        match self {
            Mixer::Shift => causal_shift(x, b, t, d),
            Mixer::Attn(a) => a.forward(x, b, t),
        }
    }
}

struct BlockCache {
    n1_in: Vec<f32>,
    n2_in: Vec<f32>,
    n2_out: Vec<f32>,
    ff_out: Vec<f32>,
    res_n2: Vec<Vec<f32>>,
    res_ff: Vec<Vec<f32>>,
}

pub struct KanBlock {
    pub n1: RmsNorm,
    pub n2: RmsNorm,
    pub mixer: Mixer,
    pub ff: TernaryKanLinear,
}

impl KanBlock {
    pub fn new(cfg: &TrainConfig, rng: &mut impl rand::Rng) -> Result<Self> {
        let mixer = if cfg.mixer == "attn" {
            Mixer::Attn(CausalAttention::new(cfg.d_model, cfg.n_heads, rng)?)
        } else {
            Mixer::Shift
        };
        let mut ff = TernaryKanLinear::new(
            cfg.d_model,
            cfg.d_model,
            cfg.n_basis,
            cfg.moe,
            cfg.n_experts,
            cfg.ternary_delta,
            rng,
        )?;
        ff.router_entropy_coef = cfg.router_entropy_coef as f32;
        ff.knot_ema = cfg.knot_ema as f32;
        Ok(Self {
            n1: RmsNorm::new(cfg.d_model)?,
            n2: RmsNorm::new(cfg.d_model)?,
            mixer,
            ff,
        })
    }

    fn forward_mode(
        &mut self,
        gpu: &SovereignDevice,
        x: &[f32],
        b: usize,
        t: usize,
        d: usize,
        mode: KanEvalMode,
        tape: bool,
    ) -> Result<(Vec<f32>, Option<BlockCache>)> {
        let n = b * t;
        let mix_mode = if mode.mask_thinking() {
            KanEvalMode::Coarse
        } else {
            KanEvalMode::Full
        };
        let n1 = self.n1.forward(x, n, d)?;
        let mix = self.mixer.forward(&n1, b, t, d)?;
        let mut h = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            h[i] = x[i] + mix[i];
        }
        let n2 = self.n2.forward(&h, n, d)?;
        let ff = self.ff.forward_mode(gpu, &n2, n, mix_mode)?;
        let mut y = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            y[i] = h[i] + ff[i];
        }
        let mut res_n2 = Vec::new();
        let mut res_ff = Vec::new();
        for _ in 1..mode.resonance_loops() {
            let rnorm = self.n2.forward(&y, n, d)?;
            let rff = self.ff.forward_mode(gpu, &rnorm, n, KanEvalMode::Full)?;
            if tape {
                res_n2.push(rnorm.clone());
                res_ff.push(rff.clone());
            }
            for i in 0..y.len() {
                y[i] += rff[i];
            }
        }
        let cache = if tape {
            Some(BlockCache {
                n1_in: x.to_vec(),
                n2_in: h,
                n2_out: n2,
                ff_out: ff,
                res_n2,
                res_ff,
            })
        } else {
            None
        };
        Ok((y, cache))
    }

    fn backward(
        &mut self,
        dy: &[f32],
        b: usize,
        t: usize,
        d: usize,
        mode: KanEvalMode,
        cache: &BlockCache,
    ) -> Result<Vec<f32>> {
        let n = b * t;
        let mut gy = dy.to_vec();
        // resonance loops in reverse
        for i in (0..cache.res_n2.len()).rev() {
            let dff = gy.clone();
            let dx_ff = self
                .ff
                .backward(&cache.res_n2[i], &dff, n, KanEvalMode::Full)?;
            let (dn2, dw) = rmsnorm_backward(
                cache_n2_parent(cache, i),
                &dx_ff,
                n,
                d,
                self.n2.weight.as_slice(),
                self.n2.eps,
            )?;
            add_assign(&mut self.n2.grad, &dw);
            add_assign(&mut gy, &dn2);
        }
        let mix_mode = if mode.mask_thinking() {
            KanEvalMode::Coarse
        } else {
            KanEvalMode::Full
        };
        let dx_ff = self
            .ff
            .backward(&cache.n2_out, gy.as_slice(), n, mix_mode)?;
        let (dn2, dw) = rmsnorm_backward(
            &cache.n2_in,
            &dx_ff,
            n,
            d,
            self.n2.weight.as_slice(),
            self.n2.eps,
        )?;
        add_assign(&mut self.n2.grad, &dw);
        let mut dh = gy;
        add_assign(&mut dh, &dn2);
        // h = x + mix(n1(x)); dh flows to mix and x
        let dmix = dh.clone();
        let dn1 = match self.mixer {
            Mixer::Shift => causal_shift_backward(&dmix, b, t, d)?,
            Mixer::Attn(_) => dmix,
        };
        let (dx_n1, dw1) = rmsnorm_backward(
            &cache.n1_in,
            &dn1,
            n,
            d,
            self.n1.weight.as_slice(),
            self.n1.eps,
        )?;
        add_assign(&mut self.n1.grad, &dw1);
        let mut dx = dh;
        add_assign(&mut dx, &dx_n1);
        Ok(dx)
    }
}

fn cache_n2_parent(cache: &BlockCache, res_i: usize) -> &[f32] {
    if res_i == 0 {
        &cache.ff_out
    } else {
        &cache.res_ff[res_i - 1]
    }
}

fn add_assign(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += *y;
    }
}

struct ModelCache {
    ids: Vec<u32>,
    b: usize,
    t: usize,
    blocks: Vec<BlockCache>,
    pre_norm: Vec<f32>,
    hidden: Vec<f32>,
}

pub struct UllisKan {
    pub cfg: TrainConfig,
    pub embed: SovereignTensor,
    pub embed_grad: Vec<f32>,
    pub blocks: Vec<KanBlock>,
    pub norm: RmsNorm,
    pub device: SovereignDevice,
    tape: Option<ModelCache>,
    pub last_ce: f32,
    pub last_entropy: f32,
    pub last_router_entropy: f32,
}

impl UllisKan {
    pub fn new(cfg: TrainConfig, device: SovereignDevice) -> Result<Self> {
        let mut rng = crate::device::rng_from_seed(cfg.seed);
        let embed = SovereignTensor::from_vec(
            vec![cfg.vocab_size, cfg.d_model],
            randn(cfg.vocab_size * cfg.d_model, 1.0, &mut rng),
        )?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let mut b = KanBlock::new(&cfg, &mut rng)?;
            b.ff.bind(&device)?;
            blocks.push(b);
        }
        let v = cfg.vocab_size;
        let d = cfg.d_model;
        Ok(Self {
            norm: RmsNorm::new(cfg.d_model)?,
            embed_grad: vec![0.0; v * d],
            cfg,
            embed,
            blocks,
            device,
            tape: None,
            last_ce: 0.0,
            last_entropy: 0.0,
            last_router_entropy: 0.0,
        })
    }

    pub fn set_phase(&mut self, phase: u8) -> Result<()> {
        for b in &mut self.blocks {
            b.ff.set_phase(phase)?;
        }
        Ok(())
    }

    pub fn extend_grid(&mut self, n_basis: usize) -> Result<()> {
        for b in &mut self.blocks {
            b.ff.extend_grid(n_basis)?;
            b.ff.bind(&self.device)?;
        }
        self.cfg.n_basis = n_basis;
        Ok(())
    }

    /// Grow every block by one adaptively placed knot. Gauss–Jordan lift.
    pub fn insert_knot(&mut self) -> Result<usize> {
        let mut g = self.cfg.n_basis;
        for b in &mut self.blocks {
            g = b.ff.insert_knot()?;
            b.ff.bind(&self.device)?;
        }
        self.cfg.n_basis = g;
        Ok(g)
    }

    pub fn sync_grids(&mut self) {
        for b in &mut self.blocks {
            b.ff.refresh_geometry();
        }
    }

    pub fn pack(&mut self) -> Result<()> {
        for b in &mut self.blocks {
            b.ff.pack()?;
            b.ff.bind(&self.device)?;
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) {
        self.embed_grad.fill(0.0);
        self.norm.grad.fill(0.0);
        for b in &mut self.blocks {
            b.n1.grad.fill(0.0);
            b.n2.grad.fill(0.0);
            b.ff.zero_grad();
        }
    }

    pub fn forward(&mut self, token_ids: &[u32], b: usize, t: usize) -> Result<Vec<f32>> {
        self.forward_mode(token_ids, b, t, KanEvalMode::Full, false)
    }

    pub fn forward_mode(
        &mut self,
        token_ids: &[u32],
        b: usize,
        t: usize,
        mode: KanEvalMode,
        tape: bool,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let n = b * t;
        let mut x = embed_lookup(self.embed.as_slice(), self.cfg.vocab_size, d, token_ids)?;
        let mut block_tapes = Vec::new();
        let gpu = &self.device;
        for blk in &mut self.blocks {
            let (y, c) = blk.forward_mode(gpu, &x, b, t, d, mode, tape)?;
            if let Some(c) = c {
                block_tapes.push(c);
            }
            x = y;
        }
        let pre_norm = if tape { x.clone() } else { Vec::new() };
        let hidden = self.norm.forward(&x, n, d)?;
        let mut logits = vec![0.0f32; n * self.cfg.vocab_size];
        sgemm_nt(
            n,
            self.cfg.vocab_size,
            d,
            1.0,
            &hidden,
            self.embed.as_slice(),
            0.0,
            &mut logits,
        )?;
        if tape {
            self.tape = Some(ModelCache {
                ids: token_ids.to_vec(),
                b,
                t,
                blocks: block_tapes,
                pre_norm,
                hidden,
            });
        }
        Ok(logits)
    }

    pub fn train_step(
        &mut self,
        ids: &[u32],
        targets: &[u32],
        mask: &[u8],
        b: usize,
        t: usize,
        l1: f32,
    ) -> Result<f32> {
        self.zero_grad();
        let v = self.cfg.vocab_size;
        let d = self.cfg.d_model;
        let n = b * t;
        let logits = self.forward_mode(ids, b, t, KanEvalMode::Full, true)?;
        let entropy_coef = self.cfg.entropy_coef as f32;
        let (mut loss, mean_h, dlogits) =
            masked_ce_entropy(&logits, n, v, targets, mask, entropy_coef)?;
        self.last_ce = loss - entropy_coef.max(0.0) * mean_h;
        self.last_entropy = mean_h;
        if l1 > 0.0 {
            loss += l1 * self.l1_penalty();
        }
        // tied logits: logits = hidden @ embed.T
        // dH = dlogits @ embed; dEmbed += hidden.T @ dlogits  (and gather path)
        let tape = self
            .tape
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing tape"))?;
        let mut dhidden = vec![0.0f32; n * d];
        // dhidden = dlogits @ embed  (dlogits [n,v], embed [v,d] → [n,d])
        crate::accelerate::sgemm(
            n,
            d,
            v,
            1.0,
            &dlogits,
            self.embed.as_slice(),
            0.0,
            &mut dhidden,
        )?;
        // dEmbed from tied head: embed_grad += dlogits.T @ hidden → [v,d]
        // For each row of dlogits [v] and hidden [d]: outer product. Use sgemm:
        // C[v,d] += A[v,n] @ B[n,d] where A = dlogits^T
        let dlt = transpose(&dlogits, n, v);
        crate::accelerate::sgemm(v, d, n, 1.0, &dlt, &tape.hidden, 1.0, &mut self.embed_grad)?;

        let (dpre, dw) = rmsnorm_backward(
            &tape.pre_norm,
            &dhidden,
            n,
            d,
            self.norm.weight.as_slice(),
            self.norm.eps,
        )?;
        add_assign(&mut self.norm.grad, &dw);
        let mut dh = dpre;
        for (blk, cache) in self.blocks.iter_mut().zip(tape.blocks.iter()).rev() {
            dh = blk.backward(&dh, tape.b, tape.t, d, KanEvalMode::Full, cache)?;
        }
        let de = embed_scatter(v, d, &tape.ids, &dh)?;
        add_assign(&mut self.embed_grad, &de);
        let mut rh = 0.0f32;
        let mut rn = 0u32;
        for blk in &self.blocks {
            if blk.ff.router_entropy_coef > 0.0 {
                rh += blk.ff.last_router_entropy;
                rn += 1;
            }
        }
        self.last_router_entropy = if rn == 0 { 0.0 } else { rh / rn as f32 };
        if rn > 0 {
            loss += self.cfg.router_entropy_coef as f32 * self.last_router_entropy;
        }
        Ok(loss)
    }

    pub fn l1_penalty(&self) -> f32 {
        let mut acc = 0.0f32;
        let mut n = 0usize;
        for b in &self.blocks {
            if b.ff.packed {
                continue;
            }
            acc += b.ff.l1_penalty();
            n += 1;
        }
        if n == 0 {
            0.0
        } else {
            acc / n as f32
        }
    }

    pub fn ternary_stats(&self) -> Result<TernaryHist> {
        let mut acc = TernaryHist::default();
        let mut n = 0u32;
        for b in &self.blocks {
            if !b.ff.packed && b.ff.phase < 3 {
                continue;
            }
            let h = b.ff.histogram()?;
            acc.merge(&h, n, 1);
            n += 1;
        }
        Ok(acc)
    }

    pub fn generate_tokens(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        temperature: f32,
        eos_id: Option<u32>,
        rng: &mut impl rand::Rng,
    ) -> Result<Vec<u32>> {
        self.generate_tokens_mode(
            prompt_ids,
            max_new,
            temperature,
            eos_id,
            KanEvalMode::Full,
            rng,
        )
    }

    pub fn generate_tokens_mode(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        temperature: f32,
        eos_id: Option<u32>,
        mode: KanEvalMode,
        rng: &mut impl rand::Rng,
    ) -> Result<Vec<u32>> {
        let mut ids = prompt_ids.to_vec();
        let mut out = Vec::new();
        for _ in 0..max_new {
            let nxt = self.next_token(&ids, temperature, mode, rng)?;
            ids.push(nxt);
            out.push(nxt);
            if eos_id == Some(nxt) {
                break;
            }
        }
        Ok(out)
    }

    pub fn next_token(
        &mut self,
        ids: &[u32],
        temperature: f32,
        mode: KanEvalMode,
        rng: &mut impl rand::Rng,
    ) -> Result<u32> {
        let ctx = if ids.len() > self.cfg.seq_len {
            &ids[ids.len() - self.cfg.seq_len..]
        } else {
            ids
        };
        let t = ctx.len().max(1);
        let logits = self.forward_mode(ctx, 1, t, mode, false)?;
        let v = self.cfg.vocab_size;
        let last = &logits[(t - 1) * v..t * v];
        Ok(sample_logits(last, temperature, rng))
    }

    pub fn generate_stream_pieces(
        &mut self,
        prompt: &str,
        tokenizer: &mut BpeTokenizer,
        max_new: usize,
        temperature: f32,
        rng: &mut impl rand::Rng,
    ) -> Result<Vec<String>> {
        let ids = tokenizer.encode(prompt, false, false);
        let eos = tokenizer.eos_id;
        let toks = self.generate_tokens(&ids, max_new, temperature, Some(eos), rng)?;
        let mut dec = StreamDecoder::new(tokenizer);
        let mut pieces = Vec::new();
        for tok in toks {
            let p = dec.push(tok);
            if !p.is_empty() {
                pieces.push(p);
            }
        }
        let tail = dec.flush();
        if !tail.is_empty() {
            pieces.push(tail);
        }
        Ok(pieces)
    }

    pub fn param_report(&self) -> String {
        let mut total = self.embed.numel();
        total += self.norm.weight.numel();
        let mut packed = false;
        let mut g = self.cfg.n_basis;
        for b in &self.blocks {
            total += b.n1.weight.numel();
            total += b.n2.weight.numel();
            if let Mixer::Attn(a) = &b.mixer {
                total += a.qkv.len() + a.proj.len();
            }
            packed |= b.ff.packed;
            g = b.ff.n_basis;
            if let Some(w) = &b.ff.weight_base {
                total += w.numel();
            }
            if let Some(w) = &b.ff.weight_shared {
                total += w.numel();
            }
            if let Some(w) = &b.ff.weight_routed {
                total += w.numel();
            }
            if let Some(w) = &b.ff.router {
                total += w.numel();
            }
            total += b.ff.scale_base.numel();
            total += b.ff.scale_shared.numel();
            total += b.ff.scale_routed.numel();
            total += b.ff.centers.numel();
            total += b.ff.inv_widths.numel();
        }
        format!(
            "params={total} packed={packed} d={} L={} G={g} V={} moe={}",
            self.cfg.d_model, self.cfg.n_layers, self.cfg.vocab_size, self.cfg.moe
        )
    }

    pub fn collect_blobs(&self) -> Result<Vec<(String, NamedBlob)>> {
        let mut out = Vec::new();
        push_named(&mut out, "embed", &self.embed);
        push_named(&mut out, "norm.weight", &self.norm.weight);
        for (i, b) in self.blocks.iter().enumerate() {
            let pfx = format!("blocks.{i}");
            push_named(&mut out, &format!("{pfx}.n1.weight"), &b.n1.weight);
            push_named(&mut out, &format!("{pfx}.n2.weight"), &b.n2.weight);
            if let Mixer::Attn(a) = &b.mixer {
                out.push((
                    format!("{pfx}.attn.qkv"),
                    NamedBlob::F32 {
                        data: a.qkv.clone(),
                        shape: vec![3 * a.d_model, a.d_model],
                    },
                ));
                out.push((
                    format!("{pfx}.attn.proj"),
                    NamedBlob::F32 {
                        data: a.proj.clone(),
                        shape: vec![a.d_model, a.d_model],
                    },
                ));
            }
            for (name, blob) in b.ff.named_tensors()? {
                out.push((format!("{pfx}.ff.{name}"), blob));
            }
        }
        Ok(out)
    }

    pub fn load_blob(&mut self, name: &str, data: &[f32], shape: &[usize]) -> Result<()> {
        if name == "embed" {
            self.embed = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
            return Ok(());
        }
        if name == "norm.weight" {
            self.norm.weight = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
            return Ok(());
        }
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let pfx = format!("blocks.{i}.");
            if let Some(rest) = name.strip_prefix(&pfx) {
                match rest {
                    "n1.weight" => {
                        b.n1.weight = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
                    }
                    "n2.weight" => {
                        b.n2.weight = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
                    }
                    "attn.qkv" => {
                        if let Mixer::Attn(a) = &mut b.mixer {
                            a.qkv = data.to_vec();
                        }
                    }
                    "attn.proj" => {
                        if let Mixer::Attn(a) = &mut b.mixer {
                            a.proj = data.to_vec();
                        }
                    }
                    other if other.starts_with("ff.") => {
                        let kn = other.trim_start_matches("ff.");
                        if kn.starts_with("packed_") {
                            return Ok(());
                        }
                        b.ff.load_f32(kn, data, shape)?;
                    }
                    other => bail!("unknown tensor {name} ({other})"),
                }
                return Ok(());
            }
        }
        bail!("unknown tensor {name}")
    }

    pub fn trainable_snapshot(&self, phase: u8) -> Vec<(String, Vec<f32>, Vec<f32>)> {
        let mut out = Vec::new();
        out.push((
            "embed".into(),
            self.embed.as_slice().to_vec(),
            self.embed_grad.clone(),
        ));
        out.push((
            "norm.weight".into(),
            self.norm.weight.as_slice().to_vec(),
            self.norm.grad.clone(),
        ));
        for (i, b) in self.blocks.iter().enumerate() {
            let pfx = format!("blocks.{i}");
            out.push((
                format!("{pfx}.n1.weight"),
                b.n1.weight.as_slice().to_vec(),
                b.n1.grad.clone(),
            ));
            out.push((
                format!("{pfx}.n2.weight"),
                b.n2.weight.as_slice().to_vec(),
                b.n2.grad.clone(),
            ));
            if !b.ff.packed && phase < 4 {
                if let Some(w) = &b.ff.weight_base {
                    out.push((
                        format!("{pfx}.ff.weight_base"),
                        w.as_slice().to_vec(),
                        b.ff.grad_base.clone(),
                    ));
                }
                if let Some(w) = &b.ff.weight_shared {
                    out.push((
                        format!("{pfx}.ff.weight_shared"),
                        w.as_slice().to_vec(),
                        b.ff.grad_shared.clone(),
                    ));
                }
                if let Some(w) = &b.ff.weight_routed {
                    out.push((
                        format!("{pfx}.ff.weight_routed"),
                        w.as_slice().to_vec(),
                        b.ff.grad_routed.clone(),
                    ));
                }
                if let Some(w) = &b.ff.router {
                    out.push((
                        format!("{pfx}.ff.router"),
                        w.as_slice().to_vec(),
                        b.ff.grad_router.clone(),
                    ));
                }
                if phase < 3 {
                    out.push((
                        format!("{pfx}.ff.centers"),
                        b.ff.centers.as_slice().to_vec(),
                        b.ff.grad_centers.clone(),
                    ));
                }
            }
            out.push((
                format!("{pfx}.ff.scale_base"),
                b.ff.scale_base.as_slice().to_vec(),
                b.ff.grad_scale_base.clone(),
            ));
            out.push((
                format!("{pfx}.ff.scale_shared"),
                b.ff.scale_shared.as_slice().to_vec(),
                b.ff.grad_scale_shared.clone(),
            ));
            out.push((
                format!("{pfx}.ff.scale_routed"),
                b.ff.scale_routed.as_slice().to_vec(),
                b.ff.grad_scale_routed.clone(),
            ));
        }
        out
    }

    pub fn write_param(&mut self, name: &str, data: &[f32]) -> Result<()> {
        if name == "embed" {
            self.embed.as_mut_slice().copy_from_slice(data);
            return Ok(());
        }
        if name == "norm.weight" {
            self.norm.weight.as_mut_slice().copy_from_slice(data);
            return Ok(());
        }
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let pfx = format!("blocks.{i}.");
            let Some(rest) = name.strip_prefix(&pfx) else {
                continue;
            };
            match rest {
                "n1.weight" => b.n1.weight.as_mut_slice().copy_from_slice(data),
                "n2.weight" => b.n2.weight.as_mut_slice().copy_from_slice(data),
                "ff.weight_base" => {
                    if let Some(w) = &mut b.ff.weight_base {
                        w.as_mut_slice().copy_from_slice(data);
                    }
                }
                "ff.weight_shared" => {
                    if let Some(w) = &mut b.ff.weight_shared {
                        w.as_mut_slice().copy_from_slice(data);
                    }
                }
                "ff.weight_routed" => {
                    if let Some(w) = &mut b.ff.weight_routed {
                        w.as_mut_slice().copy_from_slice(data);
                    }
                }
                "ff.router" => {
                    if let Some(w) = &mut b.ff.router {
                        w.as_mut_slice().copy_from_slice(data);
                    }
                }
                "ff.centers" => {
                    b.ff.centers.as_mut_slice().copy_from_slice(data);
                    b.ff.refresh_geometry();
                }
                "ff.inv_widths" => b.ff.inv_widths.as_mut_slice().copy_from_slice(data),
                "ff.scale_base" => b.ff.scale_base.as_mut_slice().copy_from_slice(data),
                "ff.scale_shared" => b.ff.scale_shared.as_mut_slice().copy_from_slice(data),
                "ff.scale_routed" => b.ff.scale_routed.as_mut_slice().copy_from_slice(data),
                _ => {}
            }
            return Ok(());
        }
        Ok(())
    }
}

fn push_named(out: &mut Vec<(String, NamedBlob)>, name: &str, t: &SovereignTensor) {
    out.push((
        name.into(),
        NamedBlob::F32 {
            data: t.as_slice().to_vec(),
            shape: t.shape().to_vec(),
        },
    ));
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

fn sample_logits(logits: &[f32], temperature: f32, rng: &mut impl rand::Rng) -> u32 {
    if temperature <= 0.0 {
        let (i, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        return i as u32;
    }
    let t = temperature.max(1e-5);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let v: Vec<f32> = logits.iter().map(|x| ((x - max) / t).exp()).collect();
    let sum: f32 = v.iter().sum();
    let r: f32 = rng.random::<f32>() * sum;
    let mut acc = 0.0;
    for (i, p) in v.iter().enumerate() {
        acc += *p;
        if r <= acc {
            return i as u32;
        }
    }
    (v.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::SovereignDevice;

    #[test]
    fn forward_shapes() {
        let gpu = SovereignDevice::open(false).unwrap();
        let cfg = TrainConfig {
            d_model: 8,
            n_layers: 2,
            n_basis: 4,
            vocab_size: 32,
            seq_len: 6,
            mixer: "shift".into(),
            moe: false,
            ..TrainConfig::default()
        };
        let mut model = UllisKan::new(cfg, gpu).unwrap();
        let ids: Vec<u32> = (0..12).map(|i| i % 32).collect();
        let y = model.forward(&ids, 2, 6).unwrap();
        assert_eq!(y.len(), 2 * 6 * 32);
    }
}
