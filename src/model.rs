use std::mem::size_of_val;

use anyhow::{bail, Result};

use crate::accelerate::sgemm_nt;
use crate::config::{KanFactor, MasterDtype, TrainConfig};
use crate::device::SovereignDevice;
use crate::kan::stored_f32;
use crate::kan::{KanEvalMode, NamedBlob, TernaryKanLinear};
use crate::mixers::{
    causal_shift_backward_into, causal_shift_into, embed_lookup_into, embed_scatter_acc, randn,
    rmsnorm, rmsnorm_backward_into, rmsnorm_into, streamed_tied_ce_acc, CausalAttention,
};
use crate::quant::{pack_f16, unpack_f16};
use crate::quant::{tied_logits_i8, PackedI8Matrix, TernaryHist, TrainStepGuard};
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

    pub fn forward_into(&self, x: &[f32], n: usize, d: usize, y: &mut [f32]) -> Result<()> {
        rmsnorm_into(x, n, d, self.weight.as_slice(), self.eps, y)
    }
}

pub enum Mixer {
    Shift,
    Attn(CausalAttention),
}

impl Mixer {
    pub fn forward(&self, x: &[f32], b: usize, t: usize, d: usize) -> Result<Vec<f32>> {
        let mut y = vec![0.0f32; x.len()];
        self.forward_into(x, b, t, d, &mut y)?;
        Ok(y)
    }

    pub fn forward_into(
        &self,
        x: &[f32],
        b: usize,
        t: usize,
        d: usize,
        y: &mut [f32],
    ) -> Result<()> {
        match self {
            Mixer::Shift => causal_shift_into(x, b, t, d, y),
            Mixer::Attn(a) => a.forward_into(x, b, t, y),
        }
    }
}

/// Grow-only `[n, d]` (or `[n]` when `d=1`) scratch. Never shrinks on the hot path.
pub fn ensure_nd(v: &mut Vec<f32>, n: usize, d: usize) {
    let need = n.saturating_mul(d);
    if v.len() < need {
        v.resize(need, 0.0);
    }
}

/// Named activation slots. Train is single-threaded; split-borrow fields, no `checkout`.
#[derive(Debug, Default)]
pub struct TrainWorkspace {
    pub x: Vec<f32>,
    pub n1: Vec<f32>,
    pub mix: Vec<f32>,
    pub h: Vec<f32>,
    pub n2: Vec<f32>,
    pub ff: Vec<f32>,
    pub y: Vec<f32>,
    pub dx: Vec<f32>,
    pub gy: Vec<f32>,
    pub dh: Vec<f32>,
    pub hidden: Vec<f32>,
    pub layer_x: Vec<Vec<f32>>,
    pub xt: Option<SovereignTensor>,
    pub yt: Option<SovereignTensor>,
    pub dyt: Option<SovereignTensor>,
    pub bwd_partial: Option<SovereignTensor>,
    pub bumps: Vec<f32>,
    pub vocab_row: Vec<f32>,
    pub q_row: Vec<f32>,
}

impl TrainWorkspace {
    pub fn prepare(
        &mut self,
        n: usize,
        d: usize,
        layers: usize,
        vocab: usize,
        in_f: usize,
        g: usize,
    ) {
        ensure_nd(&mut self.x, n, d);
        ensure_nd(&mut self.n1, n, d);
        ensure_nd(&mut self.mix, n, d);
        ensure_nd(&mut self.h, n, d);
        ensure_nd(&mut self.n2, n, d);
        ensure_nd(&mut self.ff, n, d);
        ensure_nd(&mut self.y, n, d);
        ensure_nd(&mut self.dx, n, d);
        ensure_nd(&mut self.gy, n, d);
        ensure_nd(&mut self.dh, n, d);
        ensure_nd(&mut self.hidden, n, d);
        self.layer_x.resize(layers, Vec::new());
        for lx in &mut self.layer_x {
            ensure_nd(lx, n, d);
        }
        ensure_nd(&mut self.vocab_row, vocab, 1);
        let row = in_f.saturating_mul(g).max(1);
        let max_floats = 1_000_000 / 4;
        let tile_n = (max_floats / row).max(1).min(n.max(1));
        ensure_nd(&mut self.bumps, tile_n, row);
    }

    pub fn bytes(&self) -> u64 {
        let mut n = 0u64;
        for v in [
            &self.x,
            &self.n1,
            &self.mix,
            &self.h,
            &self.n2,
            &self.ff,
            &self.y,
            &self.dx,
            &self.gy,
            &self.dh,
            &self.hidden,
            &self.bumps,
            &self.vocab_row,
            &self.q_row,
        ] {
            n += size_of_val(v.as_slice()) as u64;
        }
        for lx in &self.layer_x {
            n += size_of_val(lx.as_slice()) as u64;
        }
        if let Some(t) = &self.xt {
            n += size_of_val(t.as_slice()) as u64;
        }
        if let Some(t) = &self.yt {
            n += size_of_val(t.as_slice()) as u64;
        }
        if let Some(t) = &self.dyt {
            n += size_of_val(t.as_slice()) as u64;
        }
        if let Some(t) = &self.bwd_partial {
            n += size_of_val(t.as_slice()) as u64;
        }
        n
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
        ff.moe_topk = cfg.moe_topk;
        ff.moe_aux = cfg.moe_aux as f32;
        if cfg.kan_factor != KanFactor::SharedEdge {
            bail!("unfactored KAN was removed; shared-edge is the only layout");
        }
        Ok(Self {
            n1: RmsNorm::new(cfg.d_model)?,
            n2: RmsNorm::new(cfg.d_model)?,
            mixer,
            ff,
        })
    }

    fn forward_ws(
        &mut self,
        gpu: &SovereignDevice,
        ws: &mut TrainWorkspace,
        b: usize,
        t: usize,
        d: usize,
        mode: KanEvalMode,
        tape: bool,
    ) -> Result<Option<BlockCache>> {
        let n = b * t;
        let nd = n * d;
        let mix_mode = if mode.mask_thinking() {
            KanEvalMode::Coarse
        } else {
            KanEvalMode::Full
        };
        self.n1.forward_into(&ws.x[..nd], n, d, &mut ws.n1[..nd])?;
        self.mixer
            .forward_into(&ws.n1[..nd], b, t, d, &mut ws.mix[..nd])?;
        for i in 0..nd {
            ws.h[i] = ws.x[i] + ws.mix[i];
        }
        self.n2.forward_into(&ws.h[..nd], n, d, &mut ws.n2[..nd])?;
        self.ff.forward_into(
            gpu,
            &ws.n2[..nd],
            n,
            mix_mode,
            &mut ws.ff[..nd],
            &mut ws.xt,
            &mut ws.yt,
        )?;
        for i in 0..nd {
            ws.y[i] = ws.h[i] + ws.ff[i];
        }
        let mut cache = if tape {
            Some(BlockCache {
                n1_in: ws.x[..nd].to_vec(),
                n2_in: ws.h[..nd].to_vec(),
                n2_out: ws.n2[..nd].to_vec(),
                ff_out: ws.ff[..nd].to_vec(),
                res_n2: Vec::new(),
                res_ff: Vec::new(),
            })
        } else {
            None
        };
        for _ in 1..mode.resonance_loops() {
            self.n2.forward_into(&ws.y[..nd], n, d, &mut ws.n2[..nd])?;
            self.ff.forward_into(
                gpu,
                &ws.n2[..nd],
                n,
                KanEvalMode::Full,
                &mut ws.ff[..nd],
                &mut ws.xt,
                &mut ws.yt,
            )?;
            if let Some(c) = cache.as_mut() {
                c.res_n2.push(ws.n2[..nd].to_vec());
                c.res_ff.push(ws.ff[..nd].to_vec());
            }
            for i in 0..nd {
                ws.y[i] += ws.ff[i];
            }
        }
        Ok(cache)
    }

    /// RMS + mixer only. Fused bwd rematerializes ψ in-kernel from `n2_out`.
    fn rematerialize_pre_ff(
        &self,
        ws: &mut TrainWorkspace,
        x: &[f32],
        b: usize,
        t: usize,
        d: usize,
    ) -> Result<BlockCache> {
        let n = b * t;
        let nd = n * d;
        ws.x[..nd].copy_from_slice(&x[..nd]);
        self.n1.forward_into(&ws.x[..nd], n, d, &mut ws.n1[..nd])?;
        self.mixer
            .forward_into(&ws.n1[..nd], b, t, d, &mut ws.mix[..nd])?;
        for i in 0..nd {
            ws.h[i] = ws.x[i] + ws.mix[i];
        }
        self.n2.forward_into(&ws.h[..nd], n, d, &mut ws.n2[..nd])?;
        Ok(BlockCache {
            n1_in: ws.x[..nd].to_vec(),
            n2_in: ws.h[..nd].to_vec(),
            n2_out: ws.n2[..nd].to_vec(),
            ff_out: Vec::new(),
            res_n2: Vec::new(),
            res_ff: Vec::new(),
        })
    }

    fn kan_bwd(
        &mut self,
        gpu: &SovereignDevice,
        x: &[f32],
        n: usize,
        mode: KanEvalMode,
        ws: &mut TrainWorkspace,
        nd: usize,
    ) -> Result<()> {
        self.ff.backward_fused(
            gpu,
            x,
            &ws.gy[..nd],
            n,
            mode,
            &mut ws.dx[..nd],
            &mut ws.xt,
            &mut ws.dyt,
            &mut ws.bwd_partial,
        )
    }

    /// `ws.gy[..n*d]` is ∂L/∂y in and ∂L/∂x out. No `dy.to_vec` / `gy.clone`.
    fn backward(
        &mut self,
        gpu: &SovereignDevice,
        b: usize,
        t: usize,
        d: usize,
        mode: KanEvalMode,
        cache: &BlockCache,
        ws: &mut TrainWorkspace,
    ) -> Result<()> {
        let n = b * t;
        let nd = n * d;
        let span = self.ff.in_features.saturating_mul(self.ff.n_basis).max(1);
        if ws.bumps.len() < span {
            ws.bumps.resize(span, 0.0);
        }
        let mut dw = vec![0.0f32; d];
        for i in (0..cache.res_n2.len()).rev() {
            self.kan_bwd(gpu, &cache.res_n2[i], n, KanEvalMode::Full, ws, nd)?;
            rmsnorm_backward_into(
                cache_n2_parent(cache, i),
                &ws.dx[..nd],
                n,
                d,
                self.n2.weight.as_slice(),
                self.n2.eps,
                &mut ws.dh[..nd],
                &mut dw,
            )?;
            add_assign(&mut self.n2.grad, &dw);
            add_assign(&mut ws.gy[..nd], &ws.dh[..nd]);
        }
        let mix_mode = if mode.mask_thinking() {
            KanEvalMode::Coarse
        } else {
            KanEvalMode::Full
        };
        self.kan_bwd(gpu, &cache.n2_out, n, mix_mode, ws, nd)?;
        rmsnorm_backward_into(
            &cache.n2_in,
            &ws.dx[..nd],
            n,
            d,
            self.n2.weight.as_slice(),
            self.n2.eps,
            &mut ws.dh[..nd],
            &mut dw,
        )?;
        add_assign(&mut self.n2.grad, &dw);
        add_assign(&mut ws.gy[..nd], &ws.dh[..nd]);
        match self.mixer {
            Mixer::Shift => {
                causal_shift_backward_into(&ws.gy[..nd], b, t, d, &mut ws.mix[..nd])?;
            }
            Mixer::Attn(_) => {
                ws.mix[..nd].copy_from_slice(&ws.gy[..nd]);
            }
        }
        rmsnorm_backward_into(
            &cache.n1_in,
            &ws.mix[..nd],
            n,
            d,
            self.n1.weight.as_slice(),
            self.n1.eps,
            &mut ws.dx[..nd],
            &mut dw,
        )?;
        add_assign(&mut self.n1.grad, &dw);
        add_assign(&mut ws.gy[..nd], &ws.dx[..nd]);
        Ok(())
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

fn kan_train_weights(packed: bool, phase: u8) -> bool {
    !packed && phase < 4
}

fn kan_train_centers(packed: bool, phase: u8) -> bool {
    !packed && phase < 3
}

struct ModelCache {
    ids: Vec<u32>,
    b: usize,
    t: usize,
    blocks: Vec<BlockCache>,
    /// Layer-boundary activations `x^{(ℓ)}`. Used when fused gradient
    /// checkpointing is on: interiors are dropped and recomputed.
    layer_inputs: Vec<Vec<f32>>,
    pre_norm: Vec<f32>,
    hidden: Vec<f32>,
    checkpointed: bool,
}

pub struct UllisKan {
    pub cfg: TrainConfig,
    pub embed: SovereignTensor,
    pub embed_i8: PackedI8Matrix,
    pub embed_grad: Vec<f32>,
    pub blocks: Vec<KanBlock>,
    pub norm: RmsNorm,
    pub device: SovereignDevice,
    ws: TrainWorkspace,
    tape: Option<ModelCache>,
    pub last_ce: f32,
    pub last_entropy: f32,
    pub last_router_entropy: f32,
    pub last_aux: f32,
    pub last_mask: f32,
    pub last_fwd_ms: f32,
    pub last_ce_ms: f32,
    pub last_bwd_ms: f32,
}

impl UllisKan {
    pub fn new(cfg: TrainConfig, device: SovereignDevice) -> Result<Self> {
        let mut rng = crate::device::rng_from_seed(cfg.seed);
        let embed_std = (cfg.d_model as f32).sqrt().recip();
        let mut embed = SovereignTensor::from_vec(
            vec![cfg.vocab_size, cfg.d_model],
            randn(cfg.vocab_size * cfg.d_model, embed_std, &mut rng),
        )?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let mut b = KanBlock::new(&cfg, &mut rng)?;
            b.ff.bind(&device)?;
            if cfg.master == MasterDtype::Fp16 {
                b.ff.enable_fp16_master();
            }
            blocks.push(b);
        }
        let v = cfg.vocab_size;
        let d = cfg.d_model;
        let embed_i8 = PackedI8Matrix::quantize(embed.as_slice(), v, d)?;
        embed.attach(&device)?;
        Ok(Self {
            norm: RmsNorm::new(cfg.d_model)?,
            embed_grad: vec![0.0; v * d],
            cfg,
            embed,
            embed_i8,
            blocks,
            device,
            ws: TrainWorkspace::default(),
            tape: None,
            last_ce: 0.0,
            last_entropy: 0.0,
            last_router_entropy: 0.0,
            last_aux: 0.0,
            last_mask: 0.0,
            last_fwd_ms: 0.0,
            last_ce_ms: 0.0,
            last_bwd_ms: 0.0,
        })
    }

    pub fn refresh_embed_i8(&mut self) -> Result<()> {
        let v = self.cfg.vocab_size;
        let d = self.cfg.d_model;
        self.embed_i8 = PackedI8Matrix::quantize(self.embed.as_slice(), v, d)?;
        Ok(())
    }

    /// Quantize only if the packed plane is empty (packed infer after a
    /// code-less grow). Train CE uses FP32 `embed` and must not requantize.
    pub fn ensure_embed_i8(&mut self) -> Result<()> {
        if self.embed_i8.codes.is_empty() {
            self.refresh_embed_i8()?;
        }
        Ok(())
    }

    /// Grow the tied embedding plane to `new_v` without rewriting live rows.
    /// New ids occupy unmapped i8 blocks (zeros) until they receive mass.
    pub fn expand_vocab(&mut self, new_v: usize) -> Result<()> {
        let min_v = crate::tokenizer::MIN_VOCAB as usize;
        if new_v < min_v {
            bail!("expand_vocab {new_v} < hard minimum {min_v}");
        }
        if new_v < self.cfg.vocab_size {
            bail!("cannot shrink embedding {} -> {new_v}", self.cfg.vocab_size);
        }
        if new_v == self.cfg.vocab_size {
            return Ok(());
        }
        let d = self.cfg.d_model;
        let old_v = self.cfg.vocab_size;
        let mut data = vec![0.0f32; new_v * d];
        let old = self.embed.as_slice();
        let n = old_v.saturating_mul(d).min(old.len()).min(data.len());
        data[..n].copy_from_slice(&old[..n]);
        self.embed.detach_gpu();
        self.embed = SovereignTensor::from_vec(vec![new_v, d], data)?;
        self.embed.attach(&self.device)?;
        self.embed_grad.resize(new_v * d, 0.0);
        self.embed_i8.grow_rows(new_v)?;
        self.cfg.vocab_size = new_v;
        Ok(())
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
            let _ = b.ff.bind(&self.device);
        }
    }

    pub fn pack(&mut self) -> Result<()> {
        for b in &mut self.blocks {
            b.ff.pack()?;
            b.ff.bind(&self.device)?;
        }
        self.refresh_embed_i8()?;
        let deq = self.embed_i8.dequantize();
        self.embed.as_mut_slice().copy_from_slice(&deq);
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
        let hidden = self.forward_hidden(token_ids, b, t, mode, tape)?;
        self.project_logits(&hidden, b * t)
    }

    fn forward_hidden(
        &mut self,
        token_ids: &[u32],
        b: usize,
        t: usize,
        mode: KanEvalMode,
        tape: bool,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let n = b * t;
        let nd = n * d;
        let g = self
            .blocks
            .first()
            .map_or(self.cfg.n_basis, |blk| blk.ff.n_basis);
        self.ws
            .prepare(n, d, self.blocks.len(), self.cfg.vocab_size, d, g);
        if self.blocks.iter().any(|blk| blk.ff.packed) {
            self.ensure_embed_i8()?;
            self.embed_i8.lookup_into(token_ids, &mut self.ws.x[..nd]);
        } else {
            embed_lookup_into(
                self.embed.as_slice(),
                self.cfg.vocab_size,
                d,
                token_ids,
                &mut self.ws.x[..nd],
            )?;
        }
        let mut block_tapes = Vec::new();
        let mut layer_inputs = Vec::new();
        let checkpointed = tape && self.cfg.fused_grad_ckpt;
        let gpu = &self.device;
        for (li, blk) in self.blocks.iter_mut().enumerate() {
            if checkpointed {
                self.ws.layer_x[li][..nd].copy_from_slice(&self.ws.x[..nd]);
                layer_inputs.push(self.ws.layer_x[li][..nd].to_vec());
            }
            let c = blk.forward_ws(gpu, &mut self.ws, b, t, d, mode, tape && !checkpointed)?;
            if let Some(c) = c {
                block_tapes.push(c);
            }
            self.ws.x[..nd].copy_from_slice(&self.ws.y[..nd]);
        }
        let pre_norm = if tape {
            self.ws.x[..nd].to_vec()
        } else {
            Vec::new()
        };
        self.norm
            .forward_into(&self.ws.x[..nd], n, d, &mut self.ws.hidden[..nd])?;
        let hidden = self.ws.hidden[..nd].to_vec();
        if tape {
            self.tape = Some(ModelCache {
                ids: token_ids.to_vec(),
                b,
                t,
                blocks: block_tapes,
                layer_inputs,
                pre_norm,
                hidden: hidden.clone(),
                checkpointed,
            });
        }
        Ok(hidden)
    }

    fn project_logits(&self, hidden: &[f32], n: usize) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        let mut logits = vec![0.0f32; n * v];
        if self.blocks.iter().any(|blk| blk.ff.packed) {
            tied_logits_i8(hidden, n, d, &self.embed_i8, &mut logits);
        } else {
            sgemm_nt(
                n,
                v,
                d,
                1.0,
                hidden,
                self.embed.as_slice(),
                0.0,
                &mut logits,
            )?;
        }
        Ok(logits)
    }

    fn project_logits_last(&self, hidden: &[f32], n: usize) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let last = &hidden[(n - 1) * d..n * d];
        self.project_logits(last, 1)
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
        let _guard = TrainStepGuard::enter();
        self.zero_grad();
        let metal_fp16 = self.device.is_metal() && self.cfg.master == MasterDtype::Fp16;
        if !metal_fp16 {
            for blk in &mut self.blocks {
                blk.ff.set_hot_fp32(true)?;
            }
        }
        let v = self.cfg.vocab_size;
        let d = self.cfg.d_model;
        let n = b * t;
        let t_fwd = std::time::Instant::now();
        let _hidden = self.forward_hidden(ids, b, t, KanEvalMode::Full, true)?;
        self.last_fwd_ms = t_fwd.elapsed().as_secs_f32() * 1e3;
        let entropy_coef = self.cfg.entropy_coef as f32;
        let tape = self
            .tape
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing tape"))?;
        ensure_nd(&mut self.ws.dh, n, d);
        ensure_nd(&mut self.ws.vocab_row, v, 1);
        let t_ce = std::time::Instant::now();
        let (mut loss, mean_h) = streamed_tied_ce_acc(
            &tape.hidden,
            self.embed.as_slice(),
            n,
            d,
            v,
            targets,
            mask,
            entropy_coef,
            &mut self.ws.dh[..n * d],
            &mut self.embed_grad,
            &mut self.ws.vocab_row,
        )?;
        self.last_ce_ms = t_ce.elapsed().as_secs_f32() * 1e3;
        self.last_ce = loss - entropy_coef.max(0.0) * mean_h;
        self.last_entropy = mean_h;
        let n_sup = mask.iter().filter(|&&m| m != 0).count();
        self.last_mask = n_sup as f32 / n.max(1) as f32;
        if l1 > 0.0 {
            loss += l1 * self.l1_penalty();
        }

        let nd = n * d;
        ensure_nd(&mut self.ws.gy, n, d);
        ensure_nd(&mut self.ws.dx, n, d);
        ensure_nd(&mut self.ws.mix, n, d);
        let mut dw = vec![0.0f32; d];
        rmsnorm_backward_into(
            &tape.pre_norm,
            &self.ws.dh[..nd],
            n,
            d,
            self.norm.weight.as_slice(),
            self.norm.eps,
            &mut self.ws.gy[..nd],
            &mut dw,
        )?;
        add_assign(&mut self.norm.grad, &dw);
        let gpu = &self.device;
        let t_bwd = std::time::Instant::now();
        if tape.checkpointed {
            // Cheap rematerialize of RMS+mixer; fused bwd rematerializes ψ in TG
            // scratch. Resonance loops > 1 still re-run the full fused forward.
            for (blk, xin) in self.blocks.iter_mut().zip(tape.layer_inputs.iter()).rev() {
                let cache = blk.rematerialize_pre_ff(&mut self.ws, xin, tape.b, tape.t, d)?;
                blk.backward(
                    gpu,
                    tape.b,
                    tape.t,
                    d,
                    KanEvalMode::Full,
                    &cache,
                    &mut self.ws,
                )?;
            }
        } else {
            for (blk, cache) in self.blocks.iter_mut().zip(tape.blocks.iter()).rev() {
                blk.backward(
                    gpu,
                    tape.b,
                    tape.t,
                    d,
                    KanEvalMode::Full,
                    cache,
                    &mut self.ws,
                )?;
            }
        }
        embed_scatter_acc(v, d, &tape.ids, &self.ws.gy[..nd], &mut self.embed_grad)?;
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
        let mut aux = 0.0f32;
        let mut an = 0u32;
        for blk in &self.blocks {
            aux += blk.ff.last_aux;
            an += 1;
        }
        self.last_aux = if an == 0 { 0.0 } else { aux / an as f32 };
        loss += self.last_aux;
        self.last_bwd_ms = t_bwd.elapsed().as_secs_f32() * 1e3;
        if !metal_fp16 {
            for blk in &mut self.blocks {
                blk.ff.set_hot_fp32(false)?;
            }
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
        let hidden = self.forward_hidden(ctx, 1, t, mode, false)?;
        let mut last = self.project_logits_last(&hidden, t)?;
        if self.cfg.d_model <= 64 {
            ban_unigram_run(&mut last, ctx, 8);
        }
        Ok(sample_logits(&last, temperature, rng))
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
            total += stored_numel(&b.ff.weight_base, &b.ff.f16_base);
            total += stored_numel(&b.ff.weight_shared, &b.ff.f16_shared);
            total += stored_numel(&b.ff.weight_routed, &b.ff.f16_routed);
            total += stored_numel(&b.ff.router, &b.ff.f16_router);
            total += b.ff.scale_base.numel();
            total += b.ff.scale_shared.numel();
            total += b.ff.scale_routed.numel();
            total += b.ff.centers.numel();
            total += b.ff.inv_widths.numel();
        }
        format!(
            "params={total} packed={packed} d={} L={} G={g} V={} moe={} kan_factor={:?}",
            self.cfg.d_model,
            self.cfg.n_layers,
            self.cfg.vocab_size,
            self.cfg.moe,
            self.cfg.kan_factor
        )
    }

    pub fn collect_blobs(&self) -> Result<Vec<(String, NamedBlob)>> {
        let mut out = Vec::new();
        out.push((
            "embed".into(),
            NamedBlob::I8 {
                codes: self.embed_i8.codes_u8(),
                scale: self.embed_i8.scale.clone(),
                shape: vec![self.embed_i8.rows, self.embed_i8.cols],
            },
        ));
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

    pub fn load_i8_embed(&mut self, codes: &[u8], scale: &[f32], shape: &[usize]) -> Result<()> {
        if shape.len() != 2 {
            bail!("embed i8 shape {shape:?}");
        }
        self.embed_i8 = PackedI8Matrix::from_u8(shape[0], shape[1], codes, scale)?;
        let deq = self.embed_i8.dequantize();
        self.embed.detach_gpu();
        self.embed = SovereignTensor::from_vec(shape.to_vec(), deq)?;
        self.embed.attach(&self.device)?;
        self.cfg.vocab_size = shape[0];
        Ok(())
    }

    pub fn load_blob(&mut self, name: &str, data: &[f32], shape: &[usize]) -> Result<()> {
        if name == "embed" {
            self.embed.detach_gpu();
            self.embed = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
            self.embed.attach(&self.device)?;
            self.refresh_embed_i8()?;
            return Ok(());
        }
        if name == "norm.weight" {
            self.norm.weight.detach_gpu();
            self.norm.weight = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
            return Ok(());
        }
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let pfx = format!("blocks.{i}.");
            if let Some(rest) = name.strip_prefix(&pfx) {
                match rest {
                    "n1.weight" => {
                        b.n1.weight.detach_gpu();
                        b.n1.weight = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
                    }
                    "n2.weight" => {
                        b.n2.weight.detach_gpu();
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

    /// Oracle snapshot (tests). Hot path uses [`Self::for_each_trainable`].
    pub fn trainable_snapshot(&self, phase: u8) -> Vec<(String, Vec<f32>, Vec<f32>)> {
        let mut out = Vec::new();
        self.for_each_trainable(phase, |name, w, g| {
            out.push((name.to_string(), w.to_vec(), g.to_vec()));
        });
        out
    }

    pub fn trainable_param_bytes(&self, phase: u8) -> u64 {
        let fp16 = self.cfg.master == MasterDtype::Fp16;
        let mut n = 0u64;
        self.for_each_trainable(phase, |name, w, _| {
            let kan_w = name.contains(".ff.weight_") || name.ends_with(".ff.router");
            n += if fp16 && kan_w {
                (w.len() * 2) as u64
            } else {
                size_of_val(w) as u64
            };
        });
        n
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.ws.bytes()
    }

    pub fn embed_i8_bytes(&self) -> u64 {
        (size_of_val(self.embed_i8.codes.as_slice()) + size_of_val(self.embed_i8.scale.as_slice()))
            as u64
    }

    /// Same slot order as the historical snapshot: embed, norm, per-block
    /// n1/n2, KAN weights if `!packed && phase < 4`, centers if `phase < 3`,
    /// then scales.
    pub fn for_each_trainable<F>(&self, phase: u8, mut f: F)
    where
        F: FnMut(&str, &[f32], &[f32]),
    {
        f("embed", self.embed.as_slice(), &self.embed_grad);
        f("norm.weight", self.norm.weight.as_slice(), &self.norm.grad);
        for (i, b) in self.blocks.iter().enumerate() {
            let n1 = format!("blocks.{i}.n1.weight");
            f(&n1, b.n1.weight.as_slice(), &b.n1.grad);
            let n2 = format!("blocks.{i}.n2.weight");
            f(&n2, b.n2.weight.as_slice(), &b.n2.grad);
            if kan_train_weights(b.ff.packed, phase) {
                let nm = format!("blocks.{i}.ff.weight_base");
                visit_stored(
                    &nm,
                    &b.ff.weight_base,
                    &b.ff.f16_base,
                    &b.ff.grad_base,
                    &mut f,
                );
                let nm = format!("blocks.{i}.ff.weight_shared");
                visit_stored(
                    &nm,
                    &b.ff.weight_shared,
                    &b.ff.f16_shared,
                    &b.ff.grad_shared,
                    &mut f,
                );
                let nm = format!("blocks.{i}.ff.weight_routed");
                visit_stored(
                    &nm,
                    &b.ff.weight_routed,
                    &b.ff.f16_routed,
                    &b.ff.grad_routed,
                    &mut f,
                );
                let nm = format!("blocks.{i}.ff.router");
                visit_stored(
                    &nm,
                    &b.ff.router,
                    &b.ff.f16_router,
                    &b.ff.grad_router,
                    &mut f,
                );
                if kan_train_centers(b.ff.packed, phase) {
                    let nm = format!("blocks.{i}.ff.centers");
                    f(&nm, b.ff.centers.as_slice(), &b.ff.grad_centers);
                }
            }
            let sb = format!("blocks.{i}.ff.scale_base");
            f(&sb, b.ff.scale_base.as_slice(), &b.ff.grad_scale_base);
            let ss = format!("blocks.{i}.ff.scale_shared");
            f(&ss, b.ff.scale_shared.as_slice(), &b.ff.grad_scale_shared);
            let sr = format!("blocks.{i}.ff.scale_routed");
            f(&sr, b.ff.scale_routed.as_slice(), &b.ff.grad_scale_routed);
        }
    }

    pub fn for_each_grad<F>(&self, phase: u8, mut f: F)
    where
        F: FnMut(&str, &[f32]),
    {
        self.for_each_trainable(phase, |name, _, g| f(name, g));
    }

    pub fn for_each_param_mut<F>(&mut self, phase: u8, mut f: F)
    where
        F: FnMut(&str, &mut [f32], &[f32]),
    {
        f("embed", self.embed.as_mut_slice(), &self.embed_grad);
        f(
            "norm.weight",
            self.norm.weight.as_mut_slice(),
            &self.norm.grad,
        );
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let n1 = format!("blocks.{i}.n1.weight");
            f(&n1, b.n1.weight.as_mut_slice(), &b.n1.grad);
            let n2 = format!("blocks.{i}.n2.weight");
            f(&n2, b.n2.weight.as_mut_slice(), &b.n2.grad);
            if kan_train_weights(b.ff.packed, phase) {
                let nm = format!("blocks.{i}.ff.weight_base");
                visit_stored_mut(
                    &nm,
                    &mut b.ff.weight_base,
                    &mut b.ff.f16_base,
                    &b.ff.grad_base,
                    &mut f,
                );
                let nm = format!("blocks.{i}.ff.weight_shared");
                visit_stored_mut(
                    &nm,
                    &mut b.ff.weight_shared,
                    &mut b.ff.f16_shared,
                    &b.ff.grad_shared,
                    &mut f,
                );
                let nm = format!("blocks.{i}.ff.weight_routed");
                visit_stored_mut(
                    &nm,
                    &mut b.ff.weight_routed,
                    &mut b.ff.f16_routed,
                    &b.ff.grad_routed,
                    &mut f,
                );
                let nm = format!("blocks.{i}.ff.router");
                visit_stored_mut(
                    &nm,
                    &mut b.ff.router,
                    &mut b.ff.f16_router,
                    &b.ff.grad_router,
                    &mut f,
                );
                if kan_train_centers(b.ff.packed, phase) {
                    let nm = format!("blocks.{i}.ff.centers");
                    f(&nm, b.ff.centers.as_mut_slice(), &b.ff.grad_centers);
                }
            }
            let sb = format!("blocks.{i}.ff.scale_base");
            f(&sb, b.ff.scale_base.as_mut_slice(), &b.ff.grad_scale_base);
            let ss = format!("blocks.{i}.ff.scale_shared");
            f(
                &ss,
                b.ff.scale_shared.as_mut_slice(),
                &b.ff.grad_scale_shared,
            );
            let sr = format!("blocks.{i}.ff.scale_routed");
            f(
                &sr,
                b.ff.scale_routed.as_mut_slice(),
                &b.ff.grad_scale_routed,
            );
        }
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
                "ff.weight_base" => write_or_pack(&mut b.ff.weight_base, &mut b.ff.f16_base, data),
                "ff.weight_shared" => {
                    write_or_pack(&mut b.ff.weight_shared, &mut b.ff.f16_shared, data);
                }
                "ff.weight_routed" => {
                    write_or_pack(&mut b.ff.weight_routed, &mut b.ff.f16_routed, data);
                }
                "ff.router" => write_or_pack(&mut b.ff.router, &mut b.ff.f16_router, data),
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

fn stored_numel(live: &Option<SovereignTensor>, bits: &Option<Vec<u16>>) -> usize {
    if let Some(w) = live {
        w.numel()
    } else {
        bits.as_ref().map_or(0, Vec::len)
    }
}

fn write_or_pack(live: &mut Option<SovereignTensor>, bits: &mut Option<Vec<u16>>, data: &[f32]) {
    if let Some(w) = live.as_mut() {
        w.as_mut_slice().copy_from_slice(data);
    } else if bits.is_some() {
        *bits = Some(pack_f16(data));
    }
}

fn visit_stored(
    name: &str,
    live: &Option<SovereignTensor>,
    bits: &Option<Vec<u16>>,
    grad: &[f32],
    f: &mut impl FnMut(&str, &[f32], &[f32]),
) {
    if let Some(w) = stored_f32(live, bits) {
        f(name, w.as_ref(), grad);
    }
}

fn visit_stored_mut(
    name: &str,
    live: &mut Option<SovereignTensor>,
    bits: &mut Option<Vec<u16>>,
    grad: &[f32],
    f: &mut impl FnMut(&str, &mut [f32], &[f32]),
) {
    if let Some(w) = live.as_mut() {
        f(name, w.as_mut_slice(), grad);
        return;
    }
    if let Some(b) = bits.as_mut() {
        let mut tmp = unpack_f16(b);
        f(name, &mut tmp, grad);
        *b = pack_f16(&tmp);
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

fn ban_unigram_run(logits: &mut [f32], ctx: &[u32], run: usize) {
    if run == 0 || ctx.len() < run {
        return;
    }
    let last = ctx[ctx.len() - 1];
    if !ctx[ctx.len() - run..].iter().all(|&t| t == last) {
        return;
    }
    let i = last as usize;
    if i < logits.len() {
        logits[i] = f32::NEG_INFINITY;
    }
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
    fn unigram_run_ban_blocks_repeated_id() {
        let mut logits = vec![0.0f32; 6];
        logits[4] = 12.0;
        ban_unigram_run(&mut logits, &[4, 4, 4, 4, 4, 4, 4, 4], 8);
        assert!(logits[4].is_infinite() && logits[4] < 0.0);
        let mut logits = vec![0.0f32; 6];
        logits[4] = 12.0;
        ban_unigram_run(&mut logits, &[4, 4, 1, 4, 4, 4, 4, 4], 8);
        assert_eq!(logits[4], 12.0);
    }

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

    #[test]
    fn fused_ckpt_matches_full_tape() {
        let make = |ckpt: bool| {
            let gpu = SovereignDevice::open(false).unwrap();
            let cfg = TrainConfig {
                d_model: 8,
                n_layers: 2,
                n_basis: 4,
                vocab_size: 32,
                seq_len: 6,
                mixer: "shift".into(),
                moe: false,
                fused_grad_ckpt: ckpt,
                ..TrainConfig::default()
            };
            UllisKan::new(cfg, gpu).unwrap()
        };
        let mut full = make(false);
        let mut ckpt = make(true);
        full.set_phase(1).unwrap();
        ckpt.set_phase(1).unwrap();
        let ids: Vec<u32> = (0..12).map(|i| i % 32).collect();
        let y: Vec<u32> = (1..13).map(|i| i % 32).collect();
        let mask = vec![1u8; 12];
        let la = full.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
        let lb = ckpt.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
        assert!((la - lb).abs() < 1e-4, "loss {la} vs {lb}");
        assert!(
            max_abs_all_grads(&full, &ckpt) < 1e-4,
            "ckpt vs full tape max|Δgrad|"
        );
    }

    fn max_abs_all_grads(a: &UllisKan, b: &UllisKan) -> f32 {
        let mut m = 0.0f32;
        let mut pair = |x: &[f32], y: &[f32]| {
            for (u, v) in x.iter().zip(y.iter()) {
                m = m.max((u - v).abs());
            }
        };
        pair(&a.embed_grad, &b.embed_grad);
        pair(&a.norm.grad, &b.norm.grad);
        for (ba, bb) in a.blocks.iter().zip(b.blocks.iter()) {
            pair(&ba.n1.grad, &bb.n1.grad);
            pair(&ba.n2.grad, &bb.n2.grad);
            pair(&ba.ff.grad_base, &bb.ff.grad_base);
            pair(&ba.ff.grad_shared, &bb.ff.grad_shared);
            pair(&ba.ff.grad_routed, &bb.ff.grad_routed);
            pair(&ba.ff.grad_router, &bb.ff.grad_router);
            pair(&ba.ff.grad_centers, &bb.ff.grad_centers);
            pair(&ba.ff.grad_scale_base, &bb.ff.grad_scale_base);
            pair(&ba.ff.grad_scale_shared, &bb.ff.grad_scale_shared);
            pair(&ba.ff.grad_scale_routed, &bb.ff.grad_scale_routed);
        }
        m
    }
}
