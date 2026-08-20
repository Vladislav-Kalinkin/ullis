//! ReLU-bump Ternary KAN: fused Metal/Accelerate forward, host STE backward.

use anyhow::{bail, Result};

use crate::accelerate::{
    bump_inv_widths, mob_kan_fused_cpu, relu_bumps, sgemm_nt, softmax_rows, MobKanSpec,
};
use crate::config::split_basis;
use crate::device::SovereignDevice;
use crate::gauss::project_spline_coeffs;
use crate::mixers::{rand_kaiming, rand_uniform};
use crate::quant::{codes_to_i8, fit_scale, pack_ternary, ste_gate, ternarize_hard, TernaryHist};
use crate::tensor::{fused_mob_kan_step, FusedKanTensors, SovereignTensor};

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
    F32 { data: Vec<f32>, shape: Vec<usize> },
    Packed { bytes: Vec<u8>, shape: Vec<usize> },
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

    pub fn set_phase(&mut self, phase: u8) -> Result<()> {
        let prev = self.phase;
        self.phase = phase;
        if prev < 3 && phase >= 3 && !self.packed {
            self.fit_scales()?;
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
        MobKanSpec::new(
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
        )
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
        if x.len() != spec.x_len() {
            bail!("kan x len {} != {}", x.len(), spec.x_len());
        }
        if gpu.is_metal() {
            self.forward_metal(gpu, &spec, x, n)
        } else {
            self.forward_cpu(&spec, x)
        }
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

    fn forward_cpu(&self, spec: &MobKanSpec, x: &[f32]) -> Result<Vec<f32>> {
        let w = self.weight_views()?;
        let mut y = vec![0.0f32; spec.y_len()];
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
            &mut y,
        )?;
        Ok(y)
    }

    fn forward_metal(
        &mut self,
        gpu: &SovereignDevice,
        spec: &MobKanSpec,
        x: &[f32],
        n: usize,
    ) -> Result<Vec<f32>> {
        self.bind(gpu)?;
        let mut xt = SovereignTensor::from_vec(vec![n, self.in_features], x.to_vec())?;
        xt.attach(gpu)?;
        let mut yt = SovereignTensor::zeros(vec![n, self.out_features])?;
        yt.attach(gpu)?;
        let dummy = SovereignTensor::zeros(vec![1])?;
        if self.packed {
            let p = self
                .packed_codes
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("packed codes missing"))?;
            fused_mob_kan_step(
                gpu,
                spec,
                FusedKanTensors {
                    x: &xt,
                    y: &mut yt,
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
            let _ = dummy;
            fused_mob_kan_step(
                gpu,
                spec,
                FusedKanTensors {
                    x: &xt,
                    y: &mut yt,
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
        Ok(yt.as_slice().to_vec())
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
        if self.packed {
            return Ok(vec![0.0; spec.x_len()]);
        }
        let in_f = self.in_features;
        let out_f = self.out_features;
        let g = self.n_basis;
        let gs = spec.gs_us();
        let gr = spec.gr_us();
        let g_use = spec.g_use_us();
        let k = spec.k_us();
        let qat = spec.quantize();
        let ratio = spec.delta_ratio;

        let w_base = self
            .weight_base
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_base"))?
            .as_slice()
            .to_vec();
        let w_shared = self
            .weight_shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_shared"))?
            .as_slice()
            .to_vec();
        let w_routed = self
            .weight_routed
            .as_ref()
            .map(|t| t.as_slice().to_vec())
            .unwrap_or_default();
        let router = self
            .router
            .as_ref()
            .map(|t| t.as_slice().to_vec())
            .unwrap_or_default();
        let centers = self.centers.as_slice().to_vec();
        let inv_w = self.inv_widths.as_slice().to_vec();
        let sb = self.scale_base.as_slice().to_vec();
        let ss = self.scale_shared.as_slice().to_vec();
        let sr = self.scale_routed.as_slice().to_vec();

        let q_base = if qat {
            ternarize_hard(&w_base, out_f, in_f, f64::from(ratio))?
        } else {
            w_base.clone()
        };
        let q_shared = if qat {
            ternarize_hard(&w_shared, out_f, in_f * gs, f64::from(ratio))?
        } else {
            w_shared.clone()
        };
        let q_routed = if qat && gr > 0 && k > 0 {
            ternarize_hard(&w_routed, k * out_f, in_f * gr, f64::from(ratio))?
        } else {
            w_routed.clone()
        };

        let mut bumps = vec![0.0f32; n * in_f * g];
        relu_bumps(x, n, in_f, &centers, &inv_w, &mut bumps)?;

        let mut gates = vec![0.0f32; n * k.max(1)];
        if !spec.mask_routed() {
            sgemm_nt(n, k, in_f, 1.0, x, &router, 0.0, &mut gates)?;
            softmax_rows(&mut gates, n, k)?;
        }

        self.last_router_entropy = 0.0;
        let mut dx = vec![0.0f32; n * in_f];
        let ste = |w: f32| -> f32 {
            if qat {
                ste_gate(w)
            } else {
                1.0
            }
        };
        let scale_on = qat;

        for t in 0..n {
            for o in 0..out_f {
                let go = dy[t * out_f + o];
                let sbo = if scale_on { sb[o] } else { 1.0 };
                let sso = if scale_on { ss[o] } else { 1.0 };
                for i in 0..in_f {
                    let xv = x[t * in_f + i];
                    let qb = q_base[o * in_f + i];
                    dx[t * in_f + i] += go * qb * sbo;
                    self.grad_base[o * in_f + i] += go * xv * sbo * ste(w_base[o * in_f + i]);
                    if scale_on {
                        self.grad_scale_base[o] += go * xv * qb;
                    }
                    for gi in 0..g_use {
                        let b = bumps[(t * in_f + i) * g + gi];
                        let idx = o * (in_f * gs) + i * gs + gi;
                        let qs = q_shared[idx];
                        self.grad_shared[idx] += go * b * sso * ste(w_shared[idx]);
                        if scale_on {
                            self.grad_scale_shared[o] += go * b * qs;
                        }
                        // dψ/dx and dψ/dc
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
                    let gate = gates[t * k + e];
                    let sre = if scale_on { sr[e * out_f + o] } else { 1.0 };
                    for i in 0..in_f {
                        let xv = x[t * in_f + i];
                        for gi in 0..gr {
                            let b = bumps[(t * in_f + i) * g + gs + gi];
                            let idx = (e * out_f + o) * (in_f * gr) + i * gr + gi;
                            let qr = q_routed[idx];
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

        if !spec.mask_routed() {
            // dL/dg and softmax + router
            let mut dg = vec![0.0f32; n * k];
            for t in 0..n {
                for e in 0..k {
                    let mut s = 0.0f32;
                    for o in 0..out_f {
                        let go = dy[t * out_f + o];
                        let sre = if scale_on { sr[e * out_f + o] } else { 1.0 };
                        let mut mix = 0.0f32;
                        for i in 0..in_f {
                            for gi in 0..gr {
                                let b = bumps[(t * in_f + i) * g + gs + gi];
                                let idx = (e * out_f + o) * (in_f * gr) + i * gr + gi;
                                mix += b * q_routed[idx] * sre;
                            }
                        }
                        s += go * mix;
                    }
                    dg[t * k + e] = s;
                }
            }
            // softmax backward, then dX/dR
            let mut h_sum = 0.0f32;
            for t in 0..n {
                let g = &gates[t * k..t * k + k];
                let d = &dg[t * k..t * k + k];
                let dot: f32 = g.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
                let mut dlogit = vec![0.0f32; k];
                for e in 0..k {
                    dlogit[e] = g[e] * (d[e] - dot);
                }
                if self.router_entropy_coef > 0.0 && k > 1 {
                    let mut h = 0.0f32;
                    for e in 0..k {
                        let p = g[e].max(1e-12);
                        h -= p * p.ln();
                    }
                    h_sum += h;
                    let lam = self.router_entropy_coef;
                    for e in 0..k {
                        let p = g[e].max(1e-12);
                        dlogit[e] += lam * (-p * (p.ln() + h));
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
        }

        self.observe_residuals();
        if self.phase >= 3 {
            self.grad_centers.fill(0.0);
        }
        Ok(dx)
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
        Ok(())
    }

    pub fn l1_penalty(&self) -> f32 {
        let mut parts = Vec::new();
        if let Some(w) = &self.weight_base {
            parts.push(mean_abs(w.as_slice()));
        }
        if let Some(w) = &self.weight_shared {
            parts.push(mean_abs(w.as_slice()));
        }
        if let Some(w) = &self.weight_routed {
            parts.push(mean_abs(w.as_slice()));
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
        let base = ternarize_hard(
            self.weight_base
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no base"))?
                .as_slice(),
            self.out_features,
            self.in_features,
            self.delta_ratio,
        )?;
        let shared = ternarize_hard(
            self.weight_shared
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no shared"))?
                .as_slice(),
            self.out_features,
            self.in_features * self.n_shared.max(1),
            self.delta_ratio,
        )?;
        let routed = if let Some(w) = &self.weight_routed {
            codes_to_i8(&ternarize_hard(
                w.as_slice(),
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
        self.weight_base = None;
        self.weight_shared = None;
        self.weight_routed = None;
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
        if let Some(r) = &self.router {
            push_f32(&mut out, "router", r);
        }
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
            if let Some(w) = &self.weight_base {
                push_f32(&mut out, "weight_base", w);
            }
            if let Some(w) = &self.weight_shared {
                push_f32(&mut out, "weight_shared", w);
            }
            if let Some(w) = &self.weight_routed {
                push_f32(&mut out, "weight_routed", w);
            }
        }
        Ok(out)
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32], shape: &[usize]) -> Result<()> {
        let t = SovereignTensor::from_vec(shape.to_vec(), data.to_vec())?;
        match name {
            "centers" => {
                self.centers = t;
                if self.inv_widths.numel() != self.centers.numel() {
                    let iw = bump_inv_widths(self.centers.as_slice());
                    self.inv_widths = SovereignTensor::from_vec(vec![iw.len()], iw)?;
                }
            }
            "inv_widths" => self.inv_widths = t,
            "scale_base" => self.scale_base = t,
            "scale_shared" => self.scale_shared = t,
            "scale_routed" => self.scale_routed = t,
            "router" => self.router = Some(t),
            "weight_base" => self.weight_base = Some(t),
            "weight_shared" => self.weight_shared = Some(t),
            "weight_routed" => self.weight_routed = Some(t),
            _ => bail!("unknown kan tensor {name}"),
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

fn bump_grads(x: f32, c: f32, inv: f32, dpsi: f32, dx: &mut f32, dc: &mut f32) {
    let z = (x - c) * inv;
    let u = 1.0 - z.abs();
    if u <= 0.0 {
        return;
    }
    let du = 2.0 * u * dpsi;
    let sgn = if x >= c { 1.0 } else { -1.0 };
    *dx += du * (-inv * sgn);
    *dc += du * (inv * sgn);
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
