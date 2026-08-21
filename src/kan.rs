//! ReLU-bump Ternary KAN: fused Metal/Accelerate forward and backward.

use std::borrow::Cow;

use anyhow::{bail, Result};

use crate::accelerate::{
    apply_topk_gates, bump_grads, bump_inv_widths, mob_kan_fused_cpu, relu_bumps, sgemm_nt,
    softmax_rows, switch_aux, ternarize_row, FusedBwdGrads, MobKanSpec,
};
use crate::config::{split_basis, MasterDtype};
use crate::device::{prefer_host_bwd, SovereignDevice};
use crate::gauss::project_spline_coeffs;
use crate::mixers::{rand_kaiming, rand_uniform};
use crate::quant::{
    codes_to_i8, fit_scale, pack_f16, pack_ternary, ste_gate, ternarize_hard, unpack_f16,
    TernaryHist,
};
use crate::tensor::{
    fused_mob_kan_bwd, fused_mob_kan_step, FusedKanBwdTensors, FusedKanTensors, SovereignTensor,
};

/// Dynamic grid evaluation budget. 2-bit codes and MoE routing are unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KanEvalMode {
    Coarse,
    #[default]
    Full,
    Resonant {
        loops: u8,
    },
}

impl KanEvalMode {
    pub const COARSE_BASIS: usize = 4;

    pub fn mask_thinking(self) -> bool {
        matches!(self, Self::Coarse)
    }

    pub fn resonance_loops(self) -> usize {
        match self {
            Self::Resonant { loops } => (loops as usize).max(1),
            _ => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PackedCodes {
    pub code_base: SovereignTensor,
    pub code_shared: SovereignTensor,
    pub code_routed: Option<SovereignTensor>,
    pub packed_base: Vec<u8>,
    pub packed_shared: Vec<u8>,
    pub packed_routed: Vec<u8>,
}

pub enum NamedBlob {
    F32 {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    Packed {
        bytes: Vec<u8>,
        shape: Vec<usize>,
    },
    I8 {
        codes: Vec<u8>,
        scale: Vec<f32>,
        shape: Vec<usize>,
    },
}

pub struct TernaryKanLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub n_basis: usize,
    pub n_shared: usize,
    pub n_routed: usize,
    pub n_experts: usize,
    pub moe: bool,
    pub delta_ratio: f64,
    pub phase: u8,
    pub packed: bool,
    /// Storage master. Live `weight_*` tensors are the FP32 working copy.
    pub master: MasterDtype,
    pub x_min: f32,
    pub x_max: f32,
    pub inv_width: f32,
    pub centers: SovereignTensor,
    pub inv_widths: SovereignTensor,
    pub weight_base: Option<SovereignTensor>,
    pub weight_shared: Option<SovereignTensor>,
    pub weight_routed: Option<SovereignTensor>,
    pub router: Option<SovereignTensor>,
    pub scale_base: SovereignTensor,
    pub scale_shared: SovereignTensor,
    pub scale_routed: SovereignTensor,
    pub packed_codes: Option<PackedCodes>,
    pub(crate) f16_base: Option<Vec<u16>>,
    pub(crate) f16_shared: Option<Vec<u16>>,
    pub(crate) f16_routed: Option<Vec<u16>>,
    pub(crate) f16_router: Option<Vec<u16>>,
    pub grad_base: Vec<f32>,
    pub grad_shared: Vec<f32>,
    pub grad_routed: Vec<f32>,
    pub grad_router: Vec<f32>,
    pub grad_centers: Vec<f32>,
    pub grad_scale_base: Vec<f32>,
    pub grad_scale_shared: Vec<f32>,
    pub grad_scale_routed: Vec<f32>,
    /// EMA of `|∂L/∂c_g|` — local residual energy on the knot axis.
    pub knot_energy: Vec<f32>,
    /// EMA of mean-square spline grads per incoming edge.
    pub edge_var: Vec<f32>,
    pub knot_ema: f32,
    pub router_entropy_coef: f32,
    pub last_router_entropy: f32,
    pub moe_topk: u32,
    pub moe_aux: f32,
    pub last_aux: f32,
    pub last_route_hits: Vec<u32>,
    pub last_route_tokens: u32,
}

impl TernaryKanLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        n_basis: usize,
        moe: bool,
        n_experts: usize,
        delta_ratio: f64,
        rng: &mut impl rand::Rng,
    ) -> Result<Self> {
        if n_basis < 2 {
            bail!("n_basis must be >= 2");
        }
        let (n_shared, n_routed) = split_basis(n_basis, moe);
        let n_shared = n_shared.max(1);
        let x_min = -2.0f32;
        let x_max = 2.0f32;
        let centers_v = linspace(x_min, x_max, n_basis);
        let inv_w = bump_inv_widths(&centers_v);
        let inv_width = inv_w.iter().copied().sum::<f32>() / inv_w.len() as f32;
        let centers = SovereignTensor::from_vec(vec![n_basis], centers_v)?;
        let inv_widths = SovereignTensor::from_vec(vec![n_basis], inv_w)?;
        let weight_base = SovereignTensor::from_vec(
            vec![out_features, in_features],
            rand_kaiming(out_features, in_features, rng),
        )?;
        let weight_shared = SovereignTensor::from_vec(
            vec![out_features, in_features * n_shared],
            rand_uniform(out_features * in_features * n_shared, -0.05, 0.05, rng),
        )?;
        let (weight_routed, router, scale_routed) = if n_routed > 0 && n_experts > 0 {
            let wr = SovereignTensor::from_vec(
                vec![n_experts, out_features, in_features * n_routed],
                rand_uniform(
                    n_experts * out_features * in_features * n_routed,
                    -0.05,
                    0.05,
                    rng,
                ),
            )?;
            let rt = SovereignTensor::from_vec(
                vec![n_experts, in_features],
                rand_uniform(n_experts * in_features, -0.02, 0.02, rng),
            )?;
            let sr = SovereignTensor::fill(vec![n_experts, out_features], 1.0)?;
            (Some(wr), Some(rt), sr)
        } else {
            (None, None, SovereignTensor::fill(vec![out_features], 1.0)?)
        };
        let mut layer = Self {
            in_features,
            out_features,
            n_basis,
            n_shared,
            n_routed,
            n_experts,
            moe,
            delta_ratio,
            phase: 1,
            packed: false,
            master: MasterDtype::Fp32,
            x_min,
            x_max,
            inv_width,
            centers,
            inv_widths,
            weight_base: Some(weight_base),
            weight_shared: Some(weight_shared),
            weight_routed,
            router,
            scale_base: SovereignTensor::fill(vec![out_features], 1.0)?,
            scale_shared: SovereignTensor::fill(vec![out_features], 1.0)?,
            scale_routed,
            packed_codes: None,
            f16_base: None,
            f16_shared: None,
            f16_routed: None,
            f16_router: None,
            grad_base: Vec::new(),
            grad_shared: Vec::new(),
            grad_routed: Vec::new(),
            grad_router: Vec::new(),
            grad_centers: Vec::new(),
            grad_scale_base: vec![0.0; out_features],
            grad_scale_shared: vec![0.0; out_features],
            grad_scale_routed: Vec::new(),
            knot_energy: vec![0.0; n_basis],
            edge_var: vec![0.0; in_features],
            knot_ema: 0.9,
            router_entropy_coef: 0.0,
            last_router_entropy: 0.0,
            moe_topk: 0,
            moe_aux: 0.01,
            last_aux: 0.0,
            last_route_hits: vec![0; n_experts.max(1)],
            last_route_tokens: 0,
        };
        layer.reset_grads();
        Ok(layer)
    }

    fn reset_grads(&mut self) {
        self.grad_base = vec![0.0; self.out_features * self.in_features];
        self.grad_shared = vec![0.0; self.out_features * self.in_features * self.n_shared.max(1)];
        let rt = self.n_experts.max(1) * self.out_features * self.in_features * self.n_routed;
        self.grad_routed = vec![0.0; rt];
        self.grad_router = vec![0.0; self.n_experts.max(1) * self.in_features];
        self.grad_centers = vec![0.0; self.n_basis];
        self.grad_scale_base = vec![0.0; self.out_features];
        self.grad_scale_shared = vec![0.0; self.out_features];
        let sr = if self.n_routed > 0 {
            self.n_experts.max(1) * self.out_features
        } else {
            self.out_features
        };
        self.grad_scale_routed = vec![0.0; sr];
    }

    pub fn zero_grad(&mut self) {
        for g in [
            &mut self.grad_base,
            &mut self.grad_shared,
            &mut self.grad_routed,
            &mut self.grad_router,
            &mut self.grad_centers,
            &mut self.grad_scale_base,
            &mut self.grad_scale_shared,
            &mut self.grad_scale_routed,
        ] {
            g.fill(0.0);
        }
    }

    pub fn bind(&mut self, gpu: &SovereignDevice) -> Result<()> {
        self.hydrate(Some(gpu))?;
        self.centers.attach(gpu)?;
        self.inv_widths.attach(gpu)?;
        self.scale_base.attach(gpu)?;
        self.scale_shared.attach(gpu)?;
        self.scale_routed.attach(gpu)?;
        if let Some(w) = &mut self.weight_base {
            w.attach(gpu)?;
        }
        if let Some(w) = &mut self.weight_shared {
            w.attach(gpu)?;
        }
        if let Some(w) = &mut self.weight_routed {
            w.attach(gpu)?;
        }
        if let Some(w) = &mut self.router {
            w.attach(gpu)?;
        }
        if let Some(p) = &mut self.packed_codes {
            p.code_base.attach(gpu)?;
            p.code_shared.attach(gpu)?;
            if let Some(r) = &mut p.code_routed {
                r.attach(gpu)?;
            }
        }
        Ok(())
    }

    /// Switch storage master to fp16 and drop live FP32 copies (centers stay FP32).
    pub fn enable_fp16_master(&mut self) {
        self.master = MasterDtype::Fp16;
        self.stash_master();
    }

    pub fn hydrate(&mut self, gpu: Option<&SovereignDevice>) -> Result<()> {
        if self.master != MasterDtype::Fp16 || self.packed {
            return Ok(());
        }
        hydrate_tensor(
            &mut self.weight_base,
            &self.f16_base,
            vec![self.out_features, self.in_features],
            gpu,
        )?;
        hydrate_tensor(
            &mut self.weight_shared,
            &self.f16_shared,
            vec![self.out_features, self.in_features * self.n_shared.max(1)],
            gpu,
        )?;
        if self.n_routed > 0 {
            hydrate_tensor(
                &mut self.weight_routed,
                &self.f16_routed,
                vec![
                    self.n_experts.max(1),
                    self.out_features,
                    self.in_features * self.n_routed,
                ],
                gpu,
            )?;
            hydrate_tensor(
                &mut self.router,
                &self.f16_router,
                vec![self.n_experts.max(1), self.in_features],
                gpu,
            )?;
        }
        Ok(())
    }

    pub fn stash_master(&mut self) {
        if self.master != MasterDtype::Fp16 || self.packed {
            return;
        }
        stash_tensor(&mut self.weight_base, &mut self.f16_base);
        stash_tensor(&mut self.weight_shared, &mut self.f16_shared);
        stash_tensor(&mut self.weight_routed, &mut self.f16_routed);
        stash_tensor(&mut self.router, &mut self.f16_router);
    }

    fn observe_routing(&mut self, x: &[f32], n: usize) -> Result<()> {
        if self.n_routed == 0 || n == 0 || self.router.is_none() {
            return Ok(());
        }
        let rt = self.router.as_ref().map(SovereignTensor::as_slice).unwrap_or(&[]).to_vec();
        let k = self.n_experts.max(1);
        let mut z = vec![0.0f32; n * k];
        sgemm_nt(n, k, self.in_features, 1.0, x, &rt, 0.0, &mut z)?;
        softmax_rows(&mut z, n, k)?;
        apply_topk_gates(&mut z, n, k, self.moe_topk.min(k as u32));
        self.record_routing(&z, n, k);
        Ok(())
    }

    fn record_routing(&mut self, mix_gates: &[f32], n: usize, k: usize) {
        if self.last_route_hits.len() != k {
            self.last_route_hits.resize(k, 0);
        }
        self.last_route_hits.fill(0);
        self.last_route_tokens = n as u32;
        for t in 0..n {
            for e in 0..k {
                if mix_gates[t * k + e] > 0.0 {
                    self.last_route_hits[e] += 1;
                }
            }
        }
    }

    pub fn route_fractions(&self) -> Vec<f32> {
        let t = self.last_route_tokens.max(1) as f32;
        self.last_route_hits.iter().map(|&c| c as f32 / t).collect()
    }

    pub fn master_storage_bytes(&self) -> u64 {
        let n = |v: &Option<Vec<u16>>| v.as_ref().map_or(0, |b| b.len() * 2) as u64;
        n(&self.f16_base) + n(&self.f16_shared) + n(&self.f16_routed) + n(&self.f16_router)
    }

    pub fn set_phase(&mut self, phase: u8) -> Result<()> {
        let prev = self.phase;
        self.phase = phase;
        if prev < 3 && phase >= 3 && !self.packed {
            self.hydrate(None)?;
            self.fit_scales()?;
            self.stash_master();
        }
        Ok(())
    }

    fn fit_scales(&mut self) -> Result<()> {
        let Some(base) = &self.weight_base else {
            return Ok(());
        };
        let Some(shared) = &self.weight_shared else {
            return Ok(());
        };
        let sb = fit_scale(
            base.as_slice(),
            self.out_features,
            self.in_features,
            self.delta_ratio,
        )?;
        self.scale_base.as_mut_slice().copy_from_slice(&sb);
        let ss = fit_scale(
            shared.as_slice(),
            self.out_features,
            self.in_features * self.n_shared.max(1),
            self.delta_ratio,
        )?;
        self.scale_shared.as_mut_slice().copy_from_slice(&ss);
        if let Some(routed) = &self.weight_routed {
            let k = self.n_experts.max(1);
            let rows = k * self.out_features;
            let cols = self.in_features * self.n_routed;
            let s = fit_scale(routed.as_slice(), rows, cols, self.delta_ratio)?;
            self.scale_routed.as_mut_slice().copy_from_slice(&s);
        }
        Ok(())
    }

    fn spec(&self, n: usize, mode: KanEvalMode) -> Result<MobKanSpec> {
        let gs = self.n_shared.min(self.n_basis).max(1);
        let g_use = if mode.mask_thinking() {
            gs.clamp(1, KanEvalMode::COARSE_BASIS)
        } else {
            gs
        };
        let k = if self.n_routed > 0 {
            self.n_experts.max(1)
        } else {
            0
        };
        let spec = MobKanSpec::new(
            n,
            self.in_features,
            self.out_features,
            self.n_basis,
            gs,
            self.n_routed,
            k,
            g_use,
            self.phase,
            mode.mask_thinking(),
            self.packed,
            self.inv_width,
            self.delta_ratio as f32,
        )?;
        let tk = if k == 0 {
            0
        } else {
            self.moe_topk.min(k as u32)
        };
        spec.with_topk(tk)
    }

    pub fn forward(&mut self, gpu: &SovereignDevice, x: &[f32], n: usize) -> Result<Vec<f32>> {
        self.forward_mode(gpu, x, n, KanEvalMode::Full)
    }

    pub fn forward_mode(
        &mut self,
        gpu: &SovereignDevice,
        x: &[f32],
        n: usize,
        mode: KanEvalMode,
    ) -> Result<Vec<f32>> {
        let spec = self.spec(n, mode)?;
        let mut y = vec![0.0f32; spec.y_len()];
        let mut xt = None;
        let mut yt = None;
        self.forward_into(gpu, x, n, mode, &mut y, &mut xt, &mut yt)?;
        Ok(y)
    }

    pub fn forward_into(
        &mut self,
        gpu: &SovereignDevice,
        x: &[f32],
        n: usize,
        mode: KanEvalMode,
        y: &mut [f32],
        xt: &mut Option<SovereignTensor>,
        yt: &mut Option<SovereignTensor>,
    ) -> Result<()> {
        let spec = self.spec(n, mode)?;
        if x.len() != spec.x_len() {
            bail!("kan x len {} != {}", x.len(), spec.x_len());
        }
        if y.len() < spec.y_len() {
            bail!("kan y len {} < {}", y.len(), spec.y_len());
        }
        self.hydrate(Some(gpu))?;
        self.observe_routing(x, n)?;
        let r = if gpu.is_metal() {
            self.forward_metal(gpu, &spec, x, n, y, xt, yt)
        } else {
            self.forward_cpu(&spec, x, y)
        };
        self.stash_master();
        r
    }

    fn weight_views(&self) -> Result<WeightViews<'_>> {
        if self.packed {
            let p = self
                .packed_codes
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("packed codes missing"))?;
            Ok(WeightViews {
                base: p.code_base.as_slice(),
                shared: p.code_shared.as_slice(),
                routed: p.code_routed.as_ref().map(SovereignTensor::as_slice),
                router: self.router.as_ref().map(SovereignTensor::as_slice),
            })
        } else {
            Ok(WeightViews {
                base: self
                    .weight_base
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("weight_base missing"))?
                    .as_slice(),
                shared: self
                    .weight_shared
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("weight_shared missing"))?
                    .as_slice(),
                routed: self.weight_routed.as_ref().map(SovereignTensor::as_slice),
                router: self.router.as_ref().map(SovereignTensor::as_slice),
            })
        }
    }

    fn forward_cpu(&self, spec: &MobKanSpec, x: &[f32], y: &mut [f32]) -> Result<()> {
        let w = self.weight_views()?;
        let y = &mut y[..spec.y_len()];
        y.fill(0.0);
        mob_kan_fused_cpu(
            spec,
            x,
            w.base,
            w.shared,
            w.routed.unwrap_or(&[]),
            w.router.unwrap_or(&[]),
            self.centers.as_slice(),
            self.inv_widths.as_slice(),
            self.scale_base.as_slice(),
            self.scale_shared.as_slice(),
            self.scale_routed.as_slice(),
            y,
        )
    }

    fn forward_metal(
        &mut self,
        gpu: &SovereignDevice,
        spec: &MobKanSpec,
        x: &[f32],
        n: usize,
        y_out: &mut [f32],
        xt: &mut Option<SovereignTensor>,
        yt: &mut Option<SovereignTensor>,
    ) -> Result<()> {
        self.bind(gpu)?;
        let xt = SovereignTensor::reuse_for(xt, vec![n, self.in_features], gpu)?;
        xt.as_mut_slice().copy_from_slice(x);
        let yt = SovereignTensor::reuse_for(yt, vec![n, self.out_features], gpu)?;
        yt.as_mut_slice().fill(0.0);
        if self.packed {
            let p = self
                .packed_codes
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("packed codes missing"))?;
            fused_mob_kan_step(
                gpu,
                spec,
                FusedKanTensors {
                    x: xt,
                    y: yt,
                    w_base: &p.code_base,
                    w_shared: &p.code_shared,
                    w_routed: p.code_routed.as_ref(),
                    router: self.router.as_ref(),
                    centers: &self.centers,
                    inv_widths: &self.inv_widths,
                    scale_base: &self.scale_base,
                    scale_shared: &self.scale_shared,
                    scale_routed: &self.scale_routed,
                },
            )?;
        } else {
            let base = self
                .weight_base
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("weight_base missing"))?;
            let shared = self
                .weight_shared
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("weight_shared missing"))?;
            fused_mob_kan_step(
                gpu,
                spec,
                FusedKanTensors {
                    x: xt,
                    y: yt,
                    w_base: base,
                    w_shared: shared,
                    w_routed: self.weight_routed.as_ref(),
                    router: self.router.as_ref(),
                    centers: &self.centers,
                    inv_widths: &self.inv_widths,
                    scale_base: &self.scale_base,
                    scale_shared: &self.scale_shared,
                    scale_routed: &self.scale_routed,
                },
            )?;
        }
        y_out[..spec.y_len()].copy_from_slice(yt.as_slice());
        Ok(())
    }

    /// Host backward. `dy` is `[n, out]`. Accumulates into `grad_*`.
    pub fn backward(
        &mut self,
        x: &[f32],
        dy: &[f32],
        n: usize,
        mode: KanEvalMode,
    ) -> Result<Vec<f32>> {
        let spec = self.spec(n, mode)?;
        let mut dx = vec![0.0f32; spec.x_len()];
        let span = self.in_features.saturating_mul(self.n_basis).max(1);
        let mut bumps = vec![0.0f32; span.max(1)];
        let mut q_row = Vec::new();
        self.backward_into(x, dy, n, mode, &mut dx, &mut bumps, &mut q_row)?;
        Ok(dx)
    }

    /// Clone-free STE backward. `bumps` holds a token-tile of `ψ` (`tile_n × in × G`).
    /// `q_row` holds TWN rows `[base | shared | routed]` when `phase ≥ 3`.
    pub fn backward_into(
        &mut self,
        x: &[f32],
        dy: &[f32],
        n: usize,
        mode: KanEvalMode,
        dx: &mut [f32],
        bumps: &mut [f32],
        q_row: &mut Vec<f32>,
    ) -> Result<()> {
        let spec = self.spec(n, mode)?;
        if dx.len() < spec.x_len() {
            bail!("kan dx short");
        }
        let dx = &mut dx[..spec.x_len()];
        dx.fill(0.0);
        if self.packed {
            return Ok(());
        }
        self.hydrate(None)?;
        let in_f = self.in_features;
        let out_f = self.out_features;
        let g = self.n_basis;
        let gs = spec.gs_us();
        let gr = spec.gr_us();
        let g_use = spec.g_use_us();
        let k = spec.k_us();
        let qat = spec.quantize();
        let ratio = spec.delta_ratio;
        let scale_on = qat;

        let w_base = self
            .weight_base
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_base"))?
            .as_slice();
        let w_shared = self
            .weight_shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_shared"))?
            .as_slice();
        let w_routed = self
            .weight_routed
            .as_ref()
            .map(SovereignTensor::as_slice)
            .unwrap_or(&[]);
        let router = self
            .router
            .as_ref()
            .map(SovereignTensor::as_slice)
            .unwrap_or(&[]);
        let centers = self.centers.as_slice();
        let inv_w = self.inv_widths.as_slice();
        let sb = self.scale_base.as_slice();
        let ss = self.scale_shared.as_slice();
        let sr = self.scale_routed.as_slice();

        let cols_b = in_f;
        let cols_s = in_f * gs.max(1);
        let cols_r = in_f * gr.max(1);
        let need_q = cols_b + cols_s + cols_r;
        if q_row.len() < need_q {
            q_row.resize(need_q, 0.0);
        }
        let span = in_f.saturating_mul(g).max(1);
        let tile_n = (bumps.len() / span).max(1).min(n.max(1));
        if bumps.len() < span {
            bail!("bump tile {span} does not fit scratch {}", bumps.len());
        }

        let mut gates = vec![0.0f32; n * k.max(1)];
        if !spec.mask_routed() {
            sgemm_nt(n, k, in_f, 1.0, x, router, 0.0, &mut gates)?;
            softmax_rows(&mut gates, n, k)?;
        }
        let mut mix_gates = gates.clone();
        apply_topk_gates(&mut mix_gates, n, k, spec.topk);

        self.last_router_entropy = 0.0;
        self.last_aux = 0.0;
        let ste = |w: f32| -> f32 {
            if qat {
                ste_gate(w)
            } else {
                1.0
            }
        };

        let mut t0 = 0usize;
        while t0 < n {
            let t1 = (t0 + tile_n).min(n);
            let nt = t1 - t0;
            relu_bumps(
                &x[t0 * in_f..t1 * in_f],
                nt,
                in_f,
                centers,
                inv_w,
                &mut bumps[..nt * span],
            )?;
            for o in 0..out_f {
                if qat {
                    ternarize_row(
                        &w_base[o * cols_b..(o + 1) * cols_b],
                        ratio,
                        &mut q_row[..cols_b],
                    );
                    ternarize_row(
                        &w_shared[o * cols_s..(o + 1) * cols_s],
                        ratio,
                        &mut q_row[cols_b..cols_b + cols_s],
                    );
                }
                let sbo = if scale_on { sb[o] } else { 1.0 };
                let sso = if scale_on { ss[o] } else { 1.0 };
                for lt in 0..nt {
                    let t = t0 + lt;
                    let go = dy[t * out_f + o];
                    for i in 0..in_f {
                        let xv = x[t * in_f + i];
                        let qb = if qat {
                            q_row[i]
                        } else {
                            w_base[o * cols_b + i]
                        };
                        dx[t * in_f + i] += go * qb * sbo;
                        self.grad_base[o * cols_b + i] +=
                            go * xv * sbo * ste(w_base[o * cols_b + i]);
                        if scale_on {
                            self.grad_scale_base[o] += go * xv * qb;
                        }
                        for gi in 0..g_use {
                            let b = bumps[(lt * in_f + i) * g + gi];
                            let idx = o * cols_s + i * gs + gi;
                            let qs = if qat {
                                q_row[cols_b + i * gs + gi]
                            } else {
                                w_shared[idx]
                            };
                            self.grad_shared[idx] += go * b * sso * ste(w_shared[idx]);
                            if scale_on {
                                self.grad_scale_shared[o] += go * b * qs;
                            }
                            bump_grads(
                                xv,
                                centers[gi],
                                inv_w[gi],
                                go * qs * sso,
                                &mut dx[t * in_f + i],
                                &mut self.grad_centers[gi],
                            );
                        }
                    }
                    if spec.mask_routed() {
                        continue;
                    }
                    for e in 0..k {
                        if qat && gr > 0 {
                            let row = e * out_f + o;
                            ternarize_row(
                                &w_routed[row * cols_r..(row + 1) * cols_r],
                                ratio,
                                &mut q_row[cols_b + cols_s..cols_b + cols_s + cols_r],
                            );
                        }
                        let gate = mix_gates[t * k + e];
                        if gate == 0.0 {
                            continue;
                        }
                        let sre = if scale_on { sr[e * out_f + o] } else { 1.0 };
                        for i in 0..in_f {
                            let xv = x[t * in_f + i];
                            for gi in 0..gr {
                                let b = bumps[(lt * in_f + i) * g + gs + gi];
                                let idx = (e * out_f + o) * cols_r + i * gr + gi;
                                let qr = if qat {
                                    q_row[cols_b + cols_s + i * gr + gi]
                                } else {
                                    w_routed[idx]
                                };
                                self.grad_routed[idx] += go * gate * b * sre * ste(w_routed[idx]);
                                if scale_on {
                                    self.grad_scale_routed[e * out_f + o] += go * gate * b * qr;
                                }
                                bump_grads(
                                    xv,
                                    centers[gs + gi],
                                    inv_w[gs + gi],
                                    go * gate * qr * sre,
                                    &mut dx[t * in_f + i],
                                    &mut self.grad_centers[gs + gi],
                                );
                            }
                        }
                    }
                }
            }
            t0 = t1;
        }

        if !spec.mask_routed() {
            let mut dg = vec![0.0f32; n * k];
            t0 = 0;
            while t0 < n {
                let t1 = (t0 + tile_n).min(n);
                let nt = t1 - t0;
                relu_bumps(
                    &x[t0 * in_f..t1 * in_f],
                    nt,
                    in_f,
                    centers,
                    inv_w,
                    &mut bumps[..nt * span],
                )?;
                for o in 0..out_f {
                    for e in 0..k {
                        if qat && gr > 0 {
                            let row = e * out_f + o;
                            ternarize_row(
                                &w_routed[row * cols_r..(row + 1) * cols_r],
                                ratio,
                                &mut q_row[cols_b + cols_s..cols_b + cols_s + cols_r],
                            );
                        }
                        let sre = if scale_on { sr[e * out_f + o] } else { 1.0 };
                        for lt in 0..nt {
                            let t = t0 + lt;
                            if mix_gates[t * k + e] == 0.0 {
                                continue;
                            }
                            let go = dy[t * out_f + o];
                            let mut mix = 0.0f32;
                            for i in 0..in_f {
                                for gi in 0..gr {
                                    let b = bumps[(lt * in_f + i) * g + gs + gi];
                                    let idx = (e * out_f + o) * cols_r + i * gr + gi;
                                    let qr = if qat {
                                        q_row[cols_b + cols_s + i * gr + gi]
                                    } else {
                                        w_routed[idx]
                                    };
                                    mix += b * qr * sre;
                                }
                            }
                            dg[t * k + e] += go * mix;
                        }
                    }
                }
                t0 = t1;
            }
            let aux_coef = if spec.dense_router() { 0.0 } else { self.moe_aux };
            let (aux, dp) = if aux_coef > 0.0 {
                switch_aux(&gates, &mix_gates, n, k, aux_coef)
            } else {
                (0.0, [0.0f32; 4])
            };
            let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f32 };
            let mut h_sum = 0.0f32;
            for t in 0..n {
                let gg = &gates[t * k..t * k + k];
                let d = &dg[t * k..t * k + k];
                let dot: f32 = gg.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
                let mut dlogit = [0.0f32; 4];
                for e in 0..k {
                    dlogit[e] = gg[e] * (d[e] - dot);
                }
                if self.router_entropy_coef > 0.0 && k > 1 {
                    let mut h = 0.0f32;
                    for e in 0..k {
                        let p = gg[e].max(1e-12);
                        h -= p * p.ln();
                    }
                    h_sum += h;
                    let lam = self.router_entropy_coef;
                    for e in 0..k {
                        let p = gg[e].max(1e-12);
                        dlogit[e] += lam * (-p * (p.ln() + h));
                    }
                }
                if aux_coef > 0.0 {
                    let mut daux = [0.0f32; 4];
                    for e in 0..k {
                        daux[e] = dp[e] * inv_n;
                    }
                    let da_dot: f32 = gg.iter().zip(daux.iter()).map(|(a, b)| a * b).sum();
                    for e in 0..k {
                        dlogit[e] += gg[e] * (daux[e] - da_dot);
                    }
                }
                for e in 0..k {
                    for i in 0..in_f {
                        self.grad_router[e * in_f + i] += dlogit[e] * x[t * in_f + i];
                        dx[t * in_f + i] += dlogit[e] * router[e * in_f + i];
                    }
                }
            }
            if self.router_entropy_coef > 0.0 && k > 1 {
                self.last_router_entropy = h_sum / n as f32;
            } else {
                self.last_router_entropy = 0.0;
            }
            self.last_aux = aux;
        }

        self.observe_residuals();
        if self.phase >= 3 {
            self.grad_centers.fill(0.0);
        }
        self.stash_master();
        Ok(())
    }

    /// Fused Metal/CPU backward. `ULLIS_HOST_BWD=1` keeps the host STE tape.
    pub fn backward_fused(
        &mut self,
        gpu: &SovereignDevice,
        x: &[f32],
        dy: &[f32],
        n: usize,
        mode: KanEvalMode,
        dx: &mut [f32],
        xt: &mut Option<SovereignTensor>,
        dyt: &mut Option<SovereignTensor>,
        part: &mut Option<SovereignTensor>,
    ) -> Result<()> {
        if prefer_host_bwd() {
            let span = self.in_features.saturating_mul(self.n_basis).max(1);
            let mut bumps = vec![0.0f32; span.max(1)];
            let mut q_row = Vec::new();
            return self.backward_into(x, dy, n, mode, dx, &mut bumps, &mut q_row);
        }
        let spec = self.spec(n, mode)?;
        if dx.len() < spec.x_len() {
            bail!("kan dx short");
        }
        if self.packed {
            dx[..spec.x_len()].fill(0.0);
            return Ok(());
        }
        self.hydrate(Some(gpu))?;
        self.bind(gpu)?;
        let xt_t = SovereignTensor::reuse_for(xt, vec![n, self.in_features], gpu)?;
        xt_t.as_mut_slice().copy_from_slice(&x[..spec.x_len()]);
        let dy_t = SovereignTensor::reuse_for(dyt, vec![n, self.out_features], gpu)?;
        dy_t.as_mut_slice().copy_from_slice(&dy[..spec.y_len()]);
        let w_base = self
            .weight_base
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_base"))?;
        let w_shared = self
            .weight_shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_shared"))?;
        let (entropy, aux) = fused_mob_kan_bwd(
            gpu,
            &spec,
            FusedKanBwdTensors {
                x: xt_t,
                dy: dy_t,
                w_base,
                w_shared,
                w_routed: self.weight_routed.as_ref(),
                router: self.router.as_ref(),
                centers: &self.centers,
                inv_widths: &self.inv_widths,
                scale_base: &self.scale_base,
                scale_shared: &self.scale_shared,
                scale_routed: &self.scale_routed,
                grads: FusedBwdGrads {
                    dx: &mut dx[..spec.x_len()],
                    grad_base: &mut self.grad_base,
                    grad_shared: &mut self.grad_shared,
                    grad_routed: &mut self.grad_routed,
                    grad_router: &mut self.grad_router,
                    grad_centers: &mut self.grad_centers,
                    grad_scale_base: &mut self.grad_scale_base,
                    grad_scale_shared: &mut self.grad_scale_shared,
                    grad_scale_routed: &mut self.grad_scale_routed,
                },
                lambda_r: self.router_entropy_coef,
                aux_coef: if spec.dense_router() { 0.0 } else { self.moe_aux },
            },
            part,
        )?;
        self.last_router_entropy = entropy;
        self.last_aux = aux;
        self.observe_residuals();
        if self.phase >= 3 {
            self.grad_centers.fill(0.0);
        }
        self.stash_master();
        Ok(())
    }

    pub fn extend_grid(&mut self, n_basis: usize) -> Result<()> {
        if n_basis <= self.n_basis {
            return Ok(());
        }
        let new_centers = linspace(self.x_min, self.x_max, n_basis);
        let new_inv = bump_inv_widths(&new_centers);
        self.regrid(new_centers, new_inv)
    }

    /// Insert one knot in the highest residual-energy gap (non-uniform grow).
    pub fn insert_knot(&mut self) -> Result<usize> {
        if self.n_basis + 1 > MobKanSpec::MAX_G as usize {
            bail!("grid already at Metal cap {}", MobKanSpec::MAX_G);
        }
        let old = self.centers.as_slice();
        let g = old.len();
        if g < 2 {
            bail!("need at least 2 knots to insert");
        }
        let gap = self.hottest_gap();
        let c_new = 0.5 * (old[gap] + old[gap + 1]);
        let mut new_centers = Vec::with_capacity(g + 1);
        new_centers.extend_from_slice(&old[..=gap]);
        new_centers.push(c_new);
        new_centers.extend_from_slice(&old[gap + 1..]);
        let new_inv = bump_inv_widths(&new_centers);
        self.regrid(new_centers, new_inv)?;
        Ok(self.n_basis)
    }

    /// Keep knot order after SGD; refresh local bump widths from spacing.
    pub fn refresh_geometry(&mut self) {
        let g = self.n_basis;
        if g == 0 {
            return;
        }
        let span = (self.x_max - self.x_min).max(1e-3);
        let min_gap = (span / (g as f32 * 8.0)).max(1e-4);
        {
            let c = self.centers.as_mut_slice();
            if c.len() != g {
                return;
            }
            c[0] = c[0].clamp(self.x_min, self.x_max);
            for i in 1..g {
                let lo = c[i - 1] + min_gap;
                c[i] = c[i].clamp(lo.min(self.x_max), self.x_max);
            }
            if g >= 2 {
                c[g - 1] = c[g - 1].clamp(self.x_min, self.x_max);
                for i in (0..g - 1).rev() {
                    let hi = c[i + 1] - min_gap;
                    c[i] = c[i].min(hi).max(self.x_min);
                }
            }
        }
        let iw = bump_inv_widths(self.centers.as_slice());
        self.inv_width = if iw.is_empty() {
            1.0
        } else {
            iw.iter().copied().sum::<f32>() / iw.len() as f32
        };
        if self.inv_widths.numel() == iw.len() {
            self.inv_widths.as_mut_slice().copy_from_slice(&iw);
        } else if let Ok(t) = SovereignTensor::from_vec(vec![iw.len()], iw) {
            self.inv_widths.detach_gpu();
            self.inv_widths = t;
        }
    }

    fn hottest_gap(&self) -> usize {
        let c = self.centers.as_slice();
        let g = c.len();
        if g < 2 {
            return 0;
        }
        let mut best = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for i in 0..g - 1 {
            let gap = (c[i + 1] - c[i]).abs().max(1e-8);
            let e0 = self.knot_energy.get(i).copied().unwrap_or(0.0);
            let e1 = self.knot_energy.get(i + 1).copied().unwrap_or(0.0);
            let energy = (e0 + e1).max(1e-8);
            let edge_w = if self.edge_var.is_empty() {
                1.0f32
            } else {
                let mean: f32 =
                    self.edge_var.iter().copied().sum::<f32>() / self.edge_var.len() as f32;
                mean.max(1e-8)
            };
            let s = energy * gap * edge_w;
            if s > best_s {
                best_s = s;
                best = i;
            }
        }
        best
    }

    fn observe_residuals(&mut self) {
        let a = self.knot_ema.clamp(0.0, 0.999);
        let b = 1.0 - a;
        if self.knot_energy.len() != self.n_basis {
            self.knot_energy.resize(self.n_basis, 0.0);
        }
        for i in 0..self.n_basis {
            let v = self.grad_centers.get(i).copied().unwrap_or(0.0).abs();
            self.knot_energy[i] = a * self.knot_energy[i] + b * v;
        }
        if self.edge_var.len() != self.in_features {
            self.edge_var.resize(self.in_features, 0.0);
        }
        let gs = self.n_shared.max(1);
        let cols = self.in_features * gs;
        for i in 0..self.in_features {
            let mut acc = 0.0f32;
            let mut n = 0u32;
            if self.grad_shared.len() == self.out_features * cols {
                for o in 0..self.out_features {
                    let base = o * cols + i * gs;
                    for gi in 0..gs {
                        let g = self.grad_shared[base + gi];
                        acc += g * g;
                        n += 1;
                    }
                }
            }
            let mean = if n == 0 { 0.0 } else { acc / n as f32 };
            self.edge_var[i] = a * self.edge_var[i] + b * mean;
        }
    }

    fn regrid(&mut self, new_centers: Vec<f32>, new_inv: Vec<f32>) -> Result<()> {
        if self.packed {
            bail!("cannot extend grid of a packed layer");
        }
        self.hydrate(None)?;
        if self.phase >= 3 {
            bail!("grid is frozen from QAT (phase 3) onward");
        }
        let n_basis = new_centers.len();
        if n_basis < 2 {
            bail!("n_basis must be >= 2");
        }
        if n_basis > MobKanSpec::MAX_G as usize {
            bail!("n_basis {n_basis} exceeds Metal cap {}", MobKanSpec::MAX_G);
        }
        if new_inv.len() != n_basis {
            bail!("inv_widths len {} != G {n_basis}", new_inv.len());
        }
        let old_g = self.n_basis;
        let old_shared = self.n_shared;
        let old_routed = self.n_routed;
        let old_centers = self.centers.as_slice().to_vec();
        let old_inv = self.inv_widths.as_slice().to_vec();
        let (new_shared, new_routed) = split_basis(n_basis, self.moe);
        let ns = new_shared.max(1);
        let nr = new_routed;

        let base_shared = self
            .weight_shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_shared missing"))?
            .as_slice()
            .to_vec();
        let k = self.n_experts.max(1);
        let mut projected: Vec<Vec<f32>> = Vec::with_capacity(k);
        for e in 0..k {
            let full = concat_spline(
                &base_shared,
                self.weight_routed.as_ref().map(SovereignTensor::as_slice),
                e,
                old_shared,
                old_routed,
                self.out_features,
                self.in_features,
                old_g,
            );
            let lifted = project_spline_coeffs(
                &old_centers,
                &old_inv,
                &new_centers,
                &new_inv,
                &full,
                self.out_features,
                self.in_features,
            )?;
            projected.push(lifted);
        }

        let mut shared_sum = vec![0.0f32; self.out_features * self.in_features * ns];
        let mut routed_slices: Vec<Vec<f32>> = Vec::new();
        for p in &projected {
            for o in 0..self.out_features {
                for i in 0..self.in_features {
                    let cube = o * (self.in_features * n_basis) + i * n_basis;
                    let sh = o * (self.in_features * ns) + i * ns;
                    for gi in 0..ns {
                        shared_sum[sh + gi] += p[cube + gi];
                    }
                }
            }
            if nr > 0 {
                let mut rt = vec![0.0f32; self.out_features * self.in_features * nr];
                for o in 0..self.out_features {
                    for i in 0..self.in_features {
                        let cube = o * (self.in_features * n_basis) + i * n_basis + ns;
                        let dst = o * (self.in_features * nr) + i * nr;
                        rt[dst..dst + nr].copy_from_slice(&p[cube..cube + nr]);
                    }
                }
                routed_slices.push(rt);
            }
        }
        let inv_k = 1.0 / k as f32;
        for v in &mut shared_sum {
            *v *= inv_k;
        }

        self.centers.detach_gpu();
        self.inv_widths.detach_gpu();
        if let Some(w) = self.weight_shared.as_mut() {
            w.detach_gpu();
        }
        if let Some(w) = self.weight_routed.as_mut() {
            w.detach_gpu();
        }
        if let Some(w) = self.router.as_mut() {
            w.detach_gpu();
        }
        self.scale_routed.detach_gpu();

        self.centers = SovereignTensor::from_vec(vec![n_basis], new_centers)?;
        self.inv_widths = SovereignTensor::from_vec(vec![n_basis], new_inv)?;
        self.inv_width = self.inv_widths.as_slice().iter().copied().sum::<f32>() / n_basis as f32;
        self.n_basis = n_basis;
        self.n_shared = ns;
        self.n_routed = nr;
        self.knot_energy = vec![0.0; n_basis];
        self.weight_shared = Some(SovereignTensor::from_vec(
            vec![self.out_features, self.in_features * ns],
            shared_sum,
        )?);
        if nr > 0 && !routed_slices.is_empty() {
            let mut stacked = vec![0.0f32; k * self.out_features * self.in_features * nr];
            let row = self.out_features * self.in_features * nr;
            for (e, sl) in routed_slices.iter().enumerate() {
                stacked[e * row..(e + 1) * row].copy_from_slice(sl);
            }
            self.weight_routed = Some(SovereignTensor::from_vec(
                vec![k, self.out_features, self.in_features * nr],
                stacked,
            )?);
            if self.router.is_none() {
                let mut rng = crate::device::rng_from_seed(0);
                self.router = Some(SovereignTensor::from_vec(
                    vec![k, self.in_features],
                    rand_uniform(k * self.in_features, -0.02, 0.02, &mut rng),
                )?);
            }
            if self.scale_routed.shape().len() == 1 {
                self.scale_routed = SovereignTensor::fill(vec![k, self.out_features], 1.0)?;
            }
        } else {
            self.weight_routed = None;
        }
        self.reset_grads();
        self.stash_master();
        Ok(())
    }

    pub fn l1_penalty(&self) -> f32 {
        let mut parts = Vec::new();
        if let Some(w) = stored_f32(&self.weight_base, &self.f16_base) {
            parts.push(mean_abs(w.as_ref()));
        }
        if let Some(w) = stored_f32(&self.weight_shared, &self.f16_shared) {
            parts.push(mean_abs(w.as_ref()));
        }
        if let Some(w) = stored_f32(&self.weight_routed, &self.f16_routed) {
            parts.push(mean_abs(w.as_ref()));
        }
        if parts.is_empty() {
            0.0
        } else {
            parts.iter().sum::<f32>() / parts.len() as f32
        }
    }

    pub fn snapshot_codes(&self) -> Result<(Vec<i8>, Vec<i8>, Vec<i8>)> {
        if let Some(p) = &self.packed_codes {
            return Ok((
                codes_to_i8(p.code_base.as_slice()),
                codes_to_i8(p.code_shared.as_slice()),
                p.code_routed
                    .as_ref()
                    .map(|t| codes_to_i8(t.as_slice()))
                    .unwrap_or_default(),
            ));
        }
        let base_w = stored_f32(&self.weight_base, &self.f16_base)
            .ok_or_else(|| anyhow::anyhow!("no base"))?;
        let shared_w = stored_f32(&self.weight_shared, &self.f16_shared)
            .ok_or_else(|| anyhow::anyhow!("no shared"))?;
        let base = ternarize_hard(
            base_w.as_ref(),
            self.out_features,
            self.in_features,
            self.delta_ratio,
        )?;
        let shared = ternarize_hard(
            shared_w.as_ref(),
            self.out_features,
            self.in_features * self.n_shared.max(1),
            self.delta_ratio,
        )?;
        let routed = if let Some(w) = stored_f32(&self.weight_routed, &self.f16_routed) {
            codes_to_i8(&ternarize_hard(
                w.as_ref(),
                self.n_experts.max(1) * self.out_features,
                self.in_features * self.n_routed,
                self.delta_ratio,
            )?)
        } else {
            Vec::new()
        };
        Ok((codes_to_i8(&base), codes_to_i8(&shared), routed))
    }

    pub fn histogram(&self) -> Result<TernaryHist> {
        if self.packed {
            if let Some(p) = &self.packed_codes {
                let mut h = TernaryHist::from_f32(p.code_base.as_slice());
                h.merge(&TernaryHist::from_f32(p.code_shared.as_slice()), 1, 1);
                if let Some(r) = &p.code_routed {
                    h.merge(&TernaryHist::from_f32(r.as_slice()), 2, 1);
                }
                return Ok(h);
            }
        }
        if self.phase < 3 {
            return Ok(TernaryHist::default());
        }
        let (b, s, r) = self.snapshot_codes()?;
        let mut h = TernaryHist::from_codes(&b);
        h.merge(&TernaryHist::from_codes(&s), 1, 1);
        if !r.is_empty() {
            h.merge(&TernaryHist::from_codes(&r), 2, 1);
        }
        Ok(h)
    }

    pub fn pack(&mut self) -> Result<()> {
        if self.packed {
            return Ok(());
        }
        self.hydrate(None)?;
        let (b, s, r) = self.snapshot_codes()?;
        let bf: Vec<f32> = b.iter().map(|&c| c as f32).collect();
        let sf: Vec<f32> = s.iter().map(|&c| c as f32).collect();
        let rf: Vec<f32> = r.iter().map(|&c| c as f32).collect();
        let code_base = SovereignTensor::from_vec(vec![self.out_features, self.in_features], bf)?;
        let code_shared = SovereignTensor::from_vec(
            vec![self.out_features, self.in_features * self.n_shared.max(1)],
            sf,
        )?;
        let code_routed = if !r.is_empty() && self.n_routed > 0 {
            Some(SovereignTensor::from_vec(
                vec![
                    self.n_experts.max(1),
                    self.out_features,
                    self.in_features * self.n_routed,
                ],
                rf,
            )?)
        } else {
            None
        };
        self.packed_codes = Some(PackedCodes {
            code_base,
            code_shared,
            code_routed,
            packed_base: pack_ternary(&b),
            packed_shared: pack_ternary(&s),
            packed_routed: pack_ternary(&r),
        });
        if let Some(w) = self.weight_base.as_mut() {
            w.detach_gpu();
        }
        if let Some(w) = self.weight_shared.as_mut() {
            w.detach_gpu();
        }
        if let Some(w) = self.weight_routed.as_mut() {
            w.detach_gpu();
        }
        self.weight_base = None;
        self.weight_shared = None;
        self.weight_routed = None;
        self.f16_base = None;
        self.f16_shared = None;
        self.f16_routed = None;
        self.f16_router = None;
        self.packed = true;
        self.phase = 4;
        Ok(())
    }

    pub fn named_tensors(&self) -> Result<Vec<(String, NamedBlob)>> {
        let mut out = Vec::new();
        push_f32(&mut out, "centers", &self.centers);
        push_f32(&mut out, "inv_widths", &self.inv_widths);
        push_f32(&mut out, "scale_base", &self.scale_base);
        push_f32(&mut out, "scale_shared", &self.scale_shared);
        push_f32(&mut out, "scale_routed", &self.scale_routed);
        push_stored(
            &mut out,
            "router",
            &self.router,
            &self.f16_router,
            vec![self.n_experts.max(1), self.in_features],
        );
        if self.packed {
            if let Some(p) = &self.packed_codes {
                out.push((
                    "packed_base".into(),
                    NamedBlob::Packed {
                        bytes: p.packed_base.clone(),
                        shape: vec![self.out_features, self.in_features],
                    },
                ));
                out.push((
                    "packed_shared".into(),
                    NamedBlob::Packed {
                        bytes: p.packed_shared.clone(),
                        shape: vec![self.out_features, self.in_features * self.n_shared.max(1)],
                    },
                ));
                if !p.packed_routed.is_empty() && self.n_routed > 0 {
                    out.push((
                        "packed_routed".into(),
                        NamedBlob::Packed {
                            bytes: p.packed_routed.clone(),
                            shape: vec![
                                self.n_experts.max(1),
                                self.out_features,
                                self.in_features * self.n_routed,
                            ],
                        },
                    ));
                }
            }
        } else {
            push_stored(
                &mut out,
                "weight_base",
                &self.weight_base,
                &self.f16_base,
                vec![self.out_features, self.in_features],
            );
            push_stored(
                &mut out,
                "weight_shared",
                &self.weight_shared,
                &self.f16_shared,
                vec![self.out_features, self.in_features * self.n_shared.max(1)],
            );
            push_stored(
                &mut out,
                "weight_routed",
                &self.weight_routed,
                &self.f16_routed,
                vec![
                    self.n_experts.max(1),
                    self.out_features,
                    self.in_features * self.n_routed.max(1),
                ],
            );
        }
        Ok(out)
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32], shape: &[usize]) -> Result<()> {
        let t = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
        match name {
            "centers" => {
                self.centers.detach_gpu();
                self.centers = t;
                if self.inv_widths.numel() != self.centers.numel() {
                    let iw = bump_inv_widths(self.centers.as_slice());
                    self.inv_widths.detach_gpu();
                    self.inv_widths = SovereignTensor::from_vec(vec![iw.len()], iw)?;
                }
            }
            "inv_widths" => {
                self.inv_widths.detach_gpu();
                self.inv_widths = t;
            }
            "scale_base" => {
                self.scale_base.detach_gpu();
                self.scale_base = t;
            }
            "scale_shared" => {
                self.scale_shared.detach_gpu();
                self.scale_shared = t;
            }
            "scale_routed" => {
                self.scale_routed.detach_gpu();
                self.scale_routed = t;
            }
            "router" => {
                if let Some(w) = self.router.as_mut() {
                    w.detach_gpu();
                }
                self.router = Some(t);
            }
            "weight_base" => {
                if let Some(w) = self.weight_base.as_mut() {
                    w.detach_gpu();
                }
                self.weight_base = Some(t);
            }
            "weight_shared" => {
                if let Some(w) = self.weight_shared.as_mut() {
                    w.detach_gpu();
                }
                self.weight_shared = Some(t);
            }
            "weight_routed" => {
                if let Some(w) = self.weight_routed.as_mut() {
                    w.detach_gpu();
                }
                self.weight_routed = Some(t);
            }
            _ => bail!("unknown kan tensor {name}"),
        }
        if self.master == MasterDtype::Fp16 {
            self.stash_master();
        }
        Ok(())
    }
}

struct WeightViews<'a> {
    base: &'a [f32],
    shared: &'a [f32],
    routed: Option<&'a [f32]>,
    router: Option<&'a [f32]>,
}

fn push_f32(out: &mut Vec<(String, NamedBlob)>, name: &str, t: &SovereignTensor) {
    out.push((
        name.into(),
        NamedBlob::F32 {
            data: t.as_slice().to_vec(),
            shape: t.shape().to_vec(),
        },
    ));
}

pub(crate) fn stored_f32<'a>(
    live: &'a Option<SovereignTensor>,
    bits: &'a Option<Vec<u16>>,
) -> Option<Cow<'a, [f32]>> {
    if let Some(w) = live {
        return Some(Cow::Borrowed(w.as_slice()));
    }
    bits.as_ref().map(|b| Cow::Owned(unpack_f16(b)))
}

fn push_stored(
    out: &mut Vec<(String, NamedBlob)>,
    name: &str,
    live: &Option<SovereignTensor>,
    bits: &Option<Vec<u16>>,
    shape: Vec<usize>,
) {
    if let Some(t) = live {
        push_f32(out, name, t);
        return;
    }
    if let Some(b) = bits {
        out.push((
            name.into(),
            NamedBlob::F32 {
                data: unpack_f16(b),
                shape,
            },
        ));
    }
}

fn stash_tensor(live: &mut Option<SovereignTensor>, bits: &mut Option<Vec<u16>>) {
    if let Some(t) = live.as_mut() {
        *bits = Some(pack_f16(t.as_slice()));
        t.detach_gpu();
    }
    *live = None;
}

fn hydrate_tensor(
    live: &mut Option<SovereignTensor>,
    bits: &Option<Vec<u16>>,
    shape: Vec<usize>,
    gpu: Option<&SovereignDevice>,
) -> Result<()> {
    if live.is_some() {
        if let (Some(t), Some(g)) = (live.as_mut(), gpu) {
            t.attach(g)?;
        }
        return Ok(());
    }
    let Some(bits) = bits else {
        return Ok(());
    };
    let mut t = SovereignTensor::from_vec(shape, unpack_f16(bits))?;
    if let Some(g) = gpu {
        t.attach(g)?;
    }
    *live = Some(t);
    Ok(())
}

fn mean_abs(t: &[f32]) -> f32 {
    if t.is_empty() {
        0.0
    } else {
        t.iter().map(|v| v.abs()).sum::<f32>() / t.len() as f32
    }
}

fn linspace(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![lo];
    }
    let step = (hi - lo) / (n as f32 - 1.0);
    (0..n).map(|i| lo + step * i as f32).collect()
}

fn concat_spline(
    shared: &[f32],
    routed: Option<&[f32]>,
    expert: usize,
    n_shared: usize,
    n_routed: usize,
    out: usize,
    in_f: usize,
    g: usize,
) -> Vec<f32> {
    let ns = n_shared.max(1);
    let mut full = vec![0.0f32; out * in_f * g];
    for o in 0..out {
        for i in 0..in_f {
            let dst = o * (in_f * g) + i * g;
            let sh = o * (in_f * ns) + i * ns;
            let take = ns.min(g);
            full[dst..dst + take].copy_from_slice(&shared[sh..sh + take]);
            if n_routed > 0 {
                if let Some(rt) = routed {
                    let row = in_f * n_routed;
                    let src = (expert * out + o) * row + i * n_routed;
                    let d = dst + ns;
                    let n = n_routed.min(g.saturating_sub(ns));
                    full[d..d + n].copy_from_slice(&rt[src..src + n]);
                }
            }
        }
    }
    full
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_shapes_cpu() {
        let gpu = SovereignDevice::open(false).unwrap();
        let mut rng = crate::device::rng_from_seed(0);
        let mut layer = TernaryKanLinear::new(6, 5, 4, false, 1, 0.7, &mut rng).unwrap();
        let x = crate::mixers::randn(2 * 3 * 6, 1.0, &mut rng);
        let y = layer.forward(&gpu, &x, 6).unwrap();
        assert_eq!(y.len(), 6 * 5);
    }

    #[test]
    fn coarse_masks_thinking_and_preserves_shape() {
        let gpu = SovereignDevice::open(false).unwrap();
        let mut rng = crate::device::rng_from_seed(1);
        let mut layer = TernaryKanLinear::new(8, 8, 12, true, 3, 0.7, &mut rng).unwrap();
        let x = crate::mixers::randn(2 * 8, 1.0, &mut rng);
        let full = layer.forward_mode(&gpu, &x, 2, KanEvalMode::Full).unwrap();
        let coarse = layer
            .forward_mode(&gpu, &x, 2, KanEvalMode::Coarse)
            .unwrap();
        assert_eq!(full.len(), coarse.len());
        let delta: f32 = full
            .iter()
            .zip(coarse.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / full.len() as f32;
        assert!(delta > 0.0, "routed thinking path must contribute in Full");
    }

    #[test]
    fn insert_knot_grows_and_stays_ordered() {
        let gpu = SovereignDevice::open(false).unwrap();
        let mut rng = crate::device::rng_from_seed(2);
        let mut layer = TernaryKanLinear::new(4, 4, 4, false, 1, 0.7, &mut rng).unwrap();
        let x = crate::mixers::randn(2 * 4, 1.0, &mut rng);
        let y0 = layer.forward(&gpu, &x, 2).unwrap();
        layer.knot_energy = vec![0.1, 2.0, 2.5, 0.1];
        let g = layer.insert_knot().unwrap();
        assert_eq!(g, 5);
        let c = layer.centers.as_slice();
        for i in 1..c.len() {
            assert!(c[i] > c[i - 1], "knots must stay ordered");
        }
        assert_eq!(layer.inv_widths.numel(), 5);
        let y1 = layer.forward(&gpu, &x, 2).unwrap();
        let err: f32 = y0
            .iter()
            .zip(y1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / y0.len() as f32;
        assert!(err < 0.12, "adaptive insert drifted by {err}");
    }
}
