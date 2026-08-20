use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};

use crate::config::TrainConfig;
use crate::kan::{KanEvalMode, NamedBlob, TernaryKanLinear};
use crate::mixers::{causal_shift, randn, CausalAttention};
use crate::quant::TernaryHist;
use crate::tokenizer::{BpeTokenizer, StreamDecoder};

pub struct RmsNorm {
    pub weight: Var,
    pub eps: f64,
}

impl RmsNorm {
    pub fn new(d: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: Var::from_tensor(&Tensor::ones(d, DType::F32, device)?)?,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let rms = (xf.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        let y = xf.broadcast_div(&rms)?;
        Ok(y.broadcast_mul(self.weight.as_tensor())?
            .to_dtype(x.dtype())?)
    }
}

pub enum Mixer {
    Shift,
    Attn(CausalAttention),
}

impl Mixer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Mixer::Shift => causal_shift(x),
            Mixer::Attn(a) => a.forward(x),
        }
    }

    pub fn vars(&self) -> Vec<Var> {
        match self {
            Mixer::Shift => Vec::new(),
            Mixer::Attn(a) => a.vars(),
        }
    }
}

pub struct KanBlock {
    pub n1: RmsNorm,
    pub n2: RmsNorm,
    pub mixer: Mixer,
    pub ff: TernaryKanLinear,
}

impl KanBlock {
    pub fn new(cfg: &TrainConfig, device: &Device, rng: &mut impl rand::Rng) -> Result<Self> {
        let mixer = if cfg.mixer == "attn" {
            Mixer::Attn(CausalAttention::new(cfg.d_model, cfg.n_heads, device, rng)?)
        } else {
            Mixer::Shift
        };
        Ok(Self {
            n1: RmsNorm::new(cfg.d_model, device)?,
            n2: RmsNorm::new(cfg.d_model, device)?,
            mixer,
            ff: TernaryKanLinear::new(
                cfg.d_model,
                cfg.d_model,
                cfg.n_basis,
                cfg.moe,
                cfg.n_experts,
                cfg.ternary_delta,
                device,
                rng,
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_mode(x, KanEvalMode::Full)
    }

    pub fn forward_mode(&self, x: &Tensor, mode: KanEvalMode) -> Result<Tensor> {
        let mix_mode = if mode.mask_thinking() {
            KanEvalMode::Coarse
        } else {
            KanEvalMode::Full
        };
        let h = (x + self.mixer.forward(&self.n1.forward(x)?)?)?;
        let mut y = (&h + self.ff.forward_mode(&self.n2.forward(&h)?, mix_mode)?)?;
        // Resonance is KAN-only: mixer already imposed causality once.
        for _ in 1..mode.resonance_loops() {
            y = (&y
                + self
                    .ff
                    .forward_mode(&self.n2.forward(&y)?, KanEvalMode::Full)?)?;
        }
        Ok(y)
    }
}

pub struct UllisKan {
    pub cfg: TrainConfig,
    pub embed: Var, // [V, D]
    pub blocks: Vec<KanBlock>,
    pub norm: RmsNorm,
    pub device: Device,
}

impl UllisKan {
    pub fn new(cfg: TrainConfig, device: &Device) -> Result<Self> {
        let mut rng = crate::device::rng_from_seed(cfg.seed);
        let embed = Var::from_tensor(&randn(
            &[cfg.vocab_size, cfg.d_model],
            1.0,
            device,
            &mut rng,
        )?)?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            blocks.push(KanBlock::new(&cfg, device, &mut rng)?);
        }
        Ok(Self {
            norm: RmsNorm::new(cfg.d_model, device)?,
            cfg,
            embed,
            blocks,
            device: device.clone(),
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
        }
        self.cfg.n_basis = n_basis;
        Ok(())
    }

    pub fn pack(&mut self) -> Result<()> {
        for b in &mut self.blocks {
            b.ff.pack()?;
        }
        Ok(())
    }

    pub fn trainable_vars(&self, phase: u8) -> Vec<Var> {
        let mut v = vec![self.embed.clone(), self.norm.weight.clone()];
        for b in &self.blocks {
            v.push(b.n1.weight.clone());
            v.push(b.n2.weight.clone());
            v.extend(b.mixer.vars());
            v.extend(b.ff.trainable_vars(phase));
        }
        v
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward_mode(token_ids, KanEvalMode::Full)
    }

    pub fn forward_mode(&self, token_ids: &Tensor, mode: KanEvalMode) -> Result<Tensor> {
        let mut x = self.embed_tokens(token_ids)?;
        for b in &self.blocks {
            x = b.forward_mode(&x, mode)?;
        }
        x = self.norm.forward(&x)?;
        let (batch, time, dim) = x.dims3()?;
        let flat = x.reshape((batch * time, dim))?;
        let logits = flat.matmul(&self.embed.as_tensor().t()?)?;
        Ok(logits.reshape((batch, time, self.cfg.vocab_size))?)
    }

    fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (b, t) = token_ids.dims2()?;
        let flat = token_ids.flatten_all()?;
        let e = self.embed.as_tensor().index_select(&flat, 0)?;
        Ok(e.reshape((b, t, self.cfg.d_model))?)
    }

    pub fn l1_penalty(&self) -> Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        let mut n = 0usize;
        for b in &self.blocks {
            if b.ff.packed {
                continue;
            }
            let p = b.ff.l1_penalty()?;
            acc = Some(match acc {
                None => p,
                Some(a) => (a + p)?,
            });
            n += 1;
        }
        match acc {
            Some(a) if n > 0 => Ok((a / n as f64)?),
            _ => Ok(Tensor::zeros((), DType::F32, &self.device)?),
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
        &self,
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
        &self,
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
        &self,
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
        let x = Tensor::from_vec(ctx.to_vec(), (1, t), &self.device)?;
        let logits = self.forward_mode(&x, mode)?;
        let last = logits.i((0, t - 1))?;
        sample_logits(&last, temperature, rng)
    }

    pub fn generate_stream_pieces(
        &self,
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
        let mut total = self.embed.as_tensor().elem_count();
        total += self.norm.weight.as_tensor().elem_count();
        let mut packed = false;
        let mut g = self.cfg.n_basis;
        for b in &self.blocks {
            total += b.n1.weight.as_tensor().elem_count();
            total += b.n2.weight.as_tensor().elem_count();
            if let Mixer::Attn(a) = &b.mixer {
                total += a.qkv.as_tensor().elem_count();
                total += a.proj.as_tensor().elem_count();
            }
            packed |= b.ff.packed;
            g = b.ff.n_basis;
            if let Some(w) = &b.ff.weight_base {
                total += w.as_tensor().elem_count();
            }
            if let Some(w) = &b.ff.weight_shared {
                total += w.as_tensor().elem_count();
            }
            if let Some(w) = &b.ff.weight_routed {
                total += w.as_tensor().elem_count();
            }
            if let Some(w) = &b.ff.router {
                total += w.as_tensor().elem_count();
            }
            total += b.ff.scale_base.as_tensor().elem_count();
            total += b.ff.scale_shared.as_tensor().elem_count();
            total += b.ff.scale_routed.as_tensor().elem_count();
            total += b.ff.centers.as_tensor().elem_count();
        }
        format!(
            "params={total} packed={packed} d={} L={} G={g} V={} moe={}",
            self.cfg.d_model, self.cfg.n_layers, self.cfg.vocab_size, self.cfg.moe
        )
    }

    pub fn collect_blobs(&self) -> Result<Vec<(String, NamedBlob)>> {
        let mut out = Vec::new();
        out.push((
            "embed".into(),
            NamedBlob::F32(self.embed.as_tensor().clone()),
        ));
        out.push((
            "norm.weight".into(),
            NamedBlob::F32(self.norm.weight.as_tensor().clone()),
        ));
        for (i, b) in self.blocks.iter().enumerate() {
            let pfx = format!("blocks.{i}");
            out.push((
                format!("{pfx}.n1.weight"),
                NamedBlob::F32(b.n1.weight.as_tensor().clone()),
            ));
            out.push((
                format!("{pfx}.n2.weight"),
                NamedBlob::F32(b.n2.weight.as_tensor().clone()),
            ));
            if let Mixer::Attn(a) = &b.mixer {
                out.push((
                    format!("{pfx}.attn.qkv"),
                    NamedBlob::F32(a.qkv.as_tensor().clone()),
                ));
                out.push((
                    format!("{pfx}.attn.proj"),
                    NamedBlob::F32(a.proj.as_tensor().clone()),
                ));
            }
            for (name, blob) in b.ff.named_tensors()? {
                out.push((format!("{pfx}.ff.{name}"), blob));
            }
        }
        Ok(out)
    }

    pub fn load_blob(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        if name == "embed" {
            self.embed.set(&tensor.to_device(&self.device)?)?;
            return Ok(());
        }
        if name == "norm.weight" {
            self.norm.weight.set(&tensor.to_device(&self.device)?)?;
            return Ok(());
        }
        // block fields
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let pfx = format!("blocks.{i}.");
            if let Some(rest) = name.strip_prefix(&pfx) {
                match rest {
                    "n1.weight" => b.n1.weight.set(&tensor.to_device(&self.device)?)?,
                    "n2.weight" => b.n2.weight.set(&tensor.to_device(&self.device)?)?,
                    "attn.qkv" => {
                        if let Mixer::Attn(a) = &mut b.mixer {
                            a.qkv.set(&tensor.to_device(&self.device)?)?;
                        }
                    }
                    "attn.proj" => {
                        if let Mixer::Attn(a) = &mut b.mixer {
                            a.proj.set(&tensor.to_device(&self.device)?)?;
                        }
                    }
                    "ff.centers" => b.ff.centers.set(&tensor.to_device(&self.device)?)?,
                    "ff.scale_base" => b.ff.scale_base.set(&tensor.to_device(&self.device)?)?,
                    "ff.scale_shared" => b.ff.scale_shared.set(&tensor.to_device(&self.device)?)?,
                    "ff.scale_routed" => b.ff.scale_routed.set(&tensor.to_device(&self.device)?)?,
                    "ff.router" => {
                        if let Some(r) = &mut b.ff.router {
                            r.set(&tensor.to_device(&self.device)?)?;
                        } else {
                            b.ff.router = Some(Var::from_tensor(&tensor.to_device(&self.device)?)?);
                        }
                    }
                    "ff.weight_base" => {
                        if let Some(w) = &mut b.ff.weight_base {
                            w.set(&tensor.to_device(&self.device)?)?;
                        }
                    }
                    "ff.weight_shared" => {
                        if let Some(w) = &mut b.ff.weight_shared {
                            w.set(&tensor.to_device(&self.device)?)?;
                        }
                    }
                    "ff.weight_routed" => {
                        if let Some(w) = &mut b.ff.weight_routed {
                            w.set(&tensor.to_device(&self.device)?)?;
                        } else {
                            b.ff.weight_routed =
                                Some(Var::from_tensor(&tensor.to_device(&self.device)?)?);
                        }
                    }
                    other => {
                        if other.starts_with("ff.packed_") {
                            return Ok(());
                        }
                        bail!("unknown tensor {name}");
                    }
                }
                return Ok(());
            }
        }
        bail!("unknown tensor {name}")
    }
}

fn sample_logits(logits: &Tensor, temperature: f32, rng: &mut impl rand::Rng) -> Result<u32> {
    use crate::device::tensor_to_vec1_f32;
    let mut v = tensor_to_vec1_f32(logits)?;
    if temperature <= 0.0 {
        let (i, _) = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        return Ok(i as u32);
    }
    let t = temperature.max(1e-5);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in &mut v {
        *x = ((*x - max) / t).exp();
        sum += *x;
    }
    let r: f32 = rng.random::<f32>() * sum;
    let mut acc = 0.0;
    for (i, p) in v.iter().enumerate() {
        acc += *p;
        if r <= acc {
            return Ok(i as u32);
        }
    }
    Ok((v.len() - 1) as u32)
}
