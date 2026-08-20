//! ReLU-bump Ternary KAN linear layer: dynamic grid, STE QAT, Mixture-of-Bumps.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use rand::Rng;

use crate::config::split_basis;
use crate::gauss::project_spline_coeffs;
use crate::mixers::{rand_kaiming, rand_uniform};
use crate::quant::{
    codes_to_i8, fit_scale, histogram_tensor, pack_ternary, ternarize_hard, ternarize_ste,
    TernaryHist,
};

/// Dynamic grid evaluation budget. 2-bit codes and MoE routing are unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KanEvalMode {
    /// Linear `a_ji x_i` plus the coarse G=4 shared knots. Routed (thinking)
    /// expert weights are force-masked to zero.
    Coarse,
    /// Full shared + routed Mixture-of-Bumps at the native grid.
    #[default]
    Full,
    /// Full G=12 MoE-KAN; `loops` residual FF passes per block (see `model`).
    Resonant { loops: u8 },
}

impl KanEvalMode {
    /// Coarse knot count — matches training `grid_start`.
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
    pub code_base: Tensor,
    pub code_shared: Tensor,
    pub code_routed: Option<Tensor>,
    pub packed_base: Vec<u8>,
    pub packed_shared: Vec<u8>,
    pub packed_routed: Vec<u8>,
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
    pub inv_width: f64,
    pub centers: Var,
    pub weight_base: Option<Var>,
    pub weight_shared: Option<Var>,
    pub weight_routed: Option<Var>, // [K, out, in*Gr]
    pub router: Option<Var>,        // [K, in]
    pub scale_base: Var,
    pub scale_shared: Var,
    pub scale_routed: Var, // [K, out] or [out]
    pub packed_codes: Option<PackedCodes>,
}

impl TernaryKanLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        n_basis: usize,
        moe: bool,
        n_experts: usize,
        delta_ratio: f64,
        device: &Device,
        rng: &mut impl Rng,
    ) -> Result<Self> {
        if n_basis < 2 {
            bail!("n_basis must be >= 2");
        }
        let (n_shared, n_routed) = split_basis(n_basis, moe);
        let x_min = -2.0f32;
        let x_max = 2.0f32;
        let width = (x_max - x_min) / (n_basis as f32 - 1.0);
        let inv_width = 1.0 / width as f64;
        let centers = linspace_var(x_min, x_max, n_basis, device)?;
        let weight_base = Var::from_tensor(&rand_kaiming(out_features, in_features, device, rng)?)?;
        let weight_shared = Var::from_tensor(&rand_uniform(
            &[out_features, in_features * n_shared.max(1)],
            -0.05,
            0.05,
            device,
            rng,
        )?)?;
        let weight_routed = if n_routed > 0 && n_experts > 0 {
            Some(Var::from_tensor(&rand_uniform(
                &[n_experts, out_features, in_features * n_routed],
                -0.05,
                0.05,
                device,
                rng,
            )?)?)
        } else {
            None
        };
        let router = if n_routed > 0 && n_experts > 0 {
            Some(Var::from_tensor(&rand_uniform(
                &[n_experts, in_features],
                -0.02,
                0.02,
                device,
                rng,
            )?)?)
        } else {
            None
        };
        let ones_out = Tensor::ones(out_features, DType::F32, device)?;
        let scale_routed = if n_routed > 0 && n_experts > 0 {
            Tensor::ones((n_experts, out_features), DType::F32, device)?
        } else {
            ones_out.clone()
        };
        Ok(Self {
            in_features,
            out_features,
            n_basis,
            n_shared: n_shared.max(1),
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
            weight_base: Some(weight_base),
            weight_shared: Some(weight_shared),
            weight_routed,
            router,
            scale_base: Var::from_tensor(&ones_out)?,
            scale_shared: Var::from_tensor(&Tensor::ones(out_features, DType::F32, device)?)?,
            scale_routed: Var::from_tensor(&scale_routed)?,
            packed_codes: None,
        })
    }

    pub fn trainable_vars(&self, phase: u8) -> Vec<Var> {
        let mut v = Vec::new();
        if self.packed {
            v.push(self.scale_base.clone());
            v.push(self.scale_shared.clone());
            v.push(self.scale_routed.clone());
            return v;
        }
        if phase < 4 {
            if let Some(w) = &self.weight_base {
                v.push(w.clone());
            }
            if let Some(w) = &self.weight_shared {
                v.push(w.clone());
            }
            if let Some(w) = &self.weight_routed {
                v.push(w.clone());
            }
            if let Some(w) = &self.router {
                v.push(w.clone());
            }
            if phase < 3 {
                v.push(self.centers.clone());
            }
        }
        v.push(self.scale_base.clone());
        v.push(self.scale_shared.clone());
        v.push(self.scale_routed.clone());
        v
    }

    pub fn set_phase(&mut self, phase: u8) -> Result<()> {
        let prev = self.phase;
        self.phase = phase;
        if prev < 3 && phase >= 3 && !self.packed {
            self.fit_scales()?;
        }
        Ok(())
    }

    fn fit_scales(&self) -> Result<()> {
        let Some(base) = &self.weight_base else {
            return Ok(());
        };
        let Some(shared) = &self.weight_shared else {
            return Ok(());
        };
        self.scale_base
            .set(&fit_scale(base.as_tensor(), self.delta_ratio)?)?;
        self.scale_shared
            .set(&fit_scale(shared.as_tensor(), self.delta_ratio)?)?;
        if let Some(routed) = &self.weight_routed {
            // Fit per-expert: reshape [K*out, in*Gr]
            let t = routed.as_tensor();
            let (k, out, rest) = t.dims3()?;
            let flat = t.reshape((k * out, rest))?;
            let s = fit_scale(&flat, self.delta_ratio)?.reshape((k, out))?;
            self.scale_routed.set(&s)?;
        }
        Ok(())
    }

    fn quant_weight(&self, weight: &Tensor, scale: &Tensor) -> Result<Tensor> {
        if self.phase >= 3 {
            let codes = ternarize_ste(weight, self.delta_ratio)?;
            // scale: [out] or [K, out] — broadcast over the last dim
            broadcast_scale(&codes, scale)
        } else {
            Ok(weight.clone())
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_mode(x, KanEvalMode::Full)
    }

    pub fn forward_mode(&self, x: &Tensor, mode: KanEvalMode) -> Result<Tensor> {
        if self.packed {
            return self.packed_forward_mode(x, mode);
        }
        let orig = x.dims().to_vec();
        let in_f = self.in_features;
        let flat = flatten_leading(x, in_f)?;
        let y = self.dense_mix(&flat, mode)?;
        unflatten_leading(y, &orig, self.out_features)
    }

    fn dense_mix(&self, flat: &Tensor, mode: KanEvalMode) -> Result<Tensor> {
        let n = flat.dim(0)?;
        let in_f = self.in_features;
        let coarse = mode.mask_thinking();

        let base = self
            .weight_base
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_base missing"))?;
        let shared = self
            .weight_shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_shared missing"))?;

        let w_base = self.quant_weight(base.as_tensor(), self.scale_base.as_tensor())?;
        let mut y = flat.matmul(&w_base.t()?)?;

        let bumps = relu_bumps_nd(flat, self.centers.as_tensor(), self.inv_width)?; // [N, in, G]
        let gs = self.n_shared.min(self.n_basis);
        let g_use = shared_basis_used(gs, coarse);
        let shared_b = bumps
            .narrow(D::Minus1, 0, g_use)?
            .reshape((n, in_f * g_use))?;
        let w_shared = self.quant_weight(shared.as_tensor(), self.scale_shared.as_tensor())?;
        let w_shared = slice_shared_weight(&w_shared, in_f, gs, g_use)?;
        y = (y + shared_b.matmul(&w_shared.t()?)?)?;

        let gr = self.n_routed;
        if !coarse && gr > 0 {
            if let (Some(w_r), Some(router)) = (&self.weight_routed, &self.router) {
                let routed_b = bumps.narrow(D::Minus1, gs, gr)?.reshape((n, in_f * gr))?;
                let logits = flat.matmul(&router.as_tensor().t()?)?; // [N, K]
                let gates = candle_nn::ops::softmax(&logits, D::Minus1)?; // [N, K]
                let wq = self.quant_weight(w_r.as_tensor(), self.scale_routed.as_tensor())?;
                let (k, out, rest) = wq.dims3()?;
                let w_flat = wq.reshape((k * out, rest))?;
                let stacked = routed_b.matmul(&w_flat.t()?)?; // [N, K*out]
                let stacked = stacked.reshape((n, k, out))?;
                let mixed = stacked.broadcast_mul(&gates.unsqueeze(D::Minus1)?)?;
                let routed_y = mixed.sum(1)?;
                y = (y + routed_y)?;
            }
        }

        Ok(y)
    }

    fn packed_forward_mode(&self, x: &Tensor, mode: KanEvalMode) -> Result<Tensor> {
        let codes = self
            .packed_codes
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("packed codes missing"))?;
        let orig = x.dims().to_vec();
        let in_f = self.in_features;
        let flat = flatten_leading(x, in_f)?;
        let n = flat.dim(0)?;
        let dtype = flat.dtype();
        let coarse = mode.mask_thinking();

        let w_base = codes.code_base.to_dtype(dtype)?;
        let mut y = flat.matmul(&w_base.t()?)?;
        y = y.broadcast_mul(&self.scale_base.as_tensor().unsqueeze(0)?)?;

        let bumps = relu_bumps_nd(&flat, self.centers.as_tensor(), self.inv_width)?;
        let gs = self.n_shared.min(self.n_basis);
        let g_use = shared_basis_used(gs, coarse);
        let shared_b = bumps
            .narrow(D::Minus1, 0, g_use)?
            .reshape((n, in_f * g_use))?;
        let w_s = slice_shared_weight(&codes.code_shared.to_dtype(dtype)?, in_f, gs, g_use)?;
        let ys = shared_b.matmul(&w_s.t()?)?;
        y = (y + ys.broadcast_mul(&self.scale_shared.as_tensor().unsqueeze(0)?)?)?;

        let gr = self.n_routed;
        if !coarse && gr > 0 {
            if let (Some(code_r), Some(router)) = (&codes.code_routed, &self.router) {
                let routed_b = bumps.narrow(D::Minus1, gs, gr)?.reshape((n, in_f * gr))?;
                let logits = flat.matmul(&router.as_tensor().t()?)?;
                let gates = candle_nn::ops::softmax(&logits, D::Minus1)?;
                let wq = code_r.to_dtype(dtype)?;
                let (k, out, rest) = wq.dims3()?;
                let stacked = routed_b.matmul(&wq.reshape((k * out, rest))?.t()?)?;
                let stacked = stacked.reshape((n, k, out))?;
                let scaled = stacked.broadcast_mul(&self.scale_routed.as_tensor().unsqueeze(0)?)?;
                let gated = scaled.broadcast_mul(&gates.unsqueeze(D::Minus1)?)?;
                y = (y + gated.sum(1)?)?;
            }
        }
        unflatten_leading(y, &orig, self.out_features)
    }

    pub fn extend_grid(&mut self, n_basis: usize) -> Result<()> {
        if self.packed {
            bail!("cannot extend grid of a packed layer");
        }
        if self.phase >= 3 {
            bail!("grid is frozen from QAT (phase 3) onward");
        }
        if n_basis < 2 {
            bail!("n_basis must be >= 2");
        }
        if n_basis <= self.n_basis {
            return Ok(());
        }
        let old_g = self.n_basis;
        let old_shared = self.n_shared;
        let old_routed = self.n_routed;
        let old_centers = self.centers.as_tensor().detach();
        let old_inv = self.inv_width;
        let device = old_centers.device();

        let lo = self.x_min;
        let hi = self.x_max;
        let new_centers_t = linspace_tensor(lo, hi, n_basis, device)?;
        let width = (hi - lo) / (n_basis as f32 - 1.0);
        let new_inv = 1.0 / width as f64;
        let (new_shared, new_routed) = split_basis(n_basis, self.moe);

        // Reconstruct a full [out, in*G] spline per expert (shared || routed[k]),
        // project, then re-split. Shared is the mean of expert projections.
        let base_shared = self
            .weight_shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weight_shared missing"))?
            .as_tensor()
            .detach();

        let k = self.n_experts.max(1);
        let mut projected: Vec<Tensor> = Vec::with_capacity(k);
        for e in 0..k {
            let full = concat_spline(
                &base_shared,
                self.weight_routed.as_ref().map(|v| v.as_tensor()),
                e,
                old_shared,
                old_routed,
                self.out_features,
                self.in_features,
                old_g,
            )?;
            let lifted = project_spline_coeffs(
                &old_centers,
                old_inv,
                old_g,
                &new_centers_t,
                new_inv,
                n_basis,
                &full,
                self.out_features,
                self.in_features,
            )?;
            projected.push(lifted);
        }

        let ns = new_shared.max(1);
        let nr = new_routed;
        let mut shared_sum: Option<Tensor> = None;
        let mut routed_slices: Vec<Tensor> = Vec::new();
        for p in &projected {
            let cube = p.reshape((self.out_features, self.in_features, n_basis))?;
            let sh = cube
                .narrow(D::Minus1, 0, ns)?
                .reshape((self.out_features, self.in_features * ns))?;
            shared_sum = Some(match shared_sum {
                None => sh,
                Some(acc) => (acc + sh)?,
            });
            if nr > 0 {
                let rt = cube
                    .narrow(D::Minus1, ns, nr)?
                    .reshape((self.out_features, self.in_features * nr))?;
                routed_slices.push(rt);
            }
        }
        let shared_new = (shared_sum.unwrap() / k as f64)?;

        self.centers = Var::from_tensor(&new_centers_t)?;
        self.inv_width = new_inv;
        self.n_basis = n_basis;
        self.n_shared = ns;
        self.n_routed = nr;
        self.weight_shared = Some(Var::from_tensor(&shared_new)?);

        if nr > 0 && !routed_slices.is_empty() {
            let stacked = Tensor::stack(&routed_slices, 0)?;
            self.weight_routed = Some(Var::from_tensor(&stacked)?);
            if self.router.is_none() {
                let mut rng = crate::device::rng_from_seed(0);
                self.router = Some(Var::from_tensor(&rand_uniform(
                    &[self.n_experts.max(1), self.in_features],
                    -0.02,
                    0.02,
                    device,
                    &mut rng,
                )?)?);
            }
            if self.scale_routed.as_tensor().rank() == 1 {
                let ones = Tensor::ones(
                    (self.n_experts.max(1), self.out_features),
                    DType::F32,
                    device,
                )?;
                self.scale_routed = Var::from_tensor(&ones)?;
            }
        } else {
            self.weight_routed = None;
        }
        Ok(())
    }

    pub fn l1_penalty(&self) -> Result<Tensor> {
        let mut parts = Vec::new();
        if let Some(w) = &self.weight_base {
            parts.push(w.as_tensor().abs()?.mean_all()?);
        }
        if let Some(w) = &self.weight_shared {
            parts.push(w.as_tensor().abs()?.mean_all()?);
        }
        if let Some(w) = &self.weight_routed {
            parts.push(w.as_tensor().abs()?.mean_all()?);
        }
        if parts.is_empty() {
            return Ok(Tensor::zeros(
                (),
                DType::F32,
                self.scale_base.as_tensor().device(),
            )?);
        }
        let mut acc = parts[0].clone();
        for p in parts.iter().skip(1) {
            acc = (acc + p)?;
        }
        Ok((acc / parts.len() as f64)?)
    }

    pub fn snapshot_codes(&self) -> Result<(Vec<i8>, Vec<i8>, Vec<i8>)> {
        if let Some(p) = &self.packed_codes {
            return Ok((
                codes_to_i8(&p.code_base)?,
                codes_to_i8(&p.code_shared)?,
                match &p.code_routed {
                    Some(t) => codes_to_i8(t)?,
                    None => Vec::new(),
                },
            ));
        }
        let base = ternarize_hard(
            self.weight_base
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no base"))?
                .as_tensor(),
            self.delta_ratio,
        )?;
        let shared = ternarize_hard(
            self.weight_shared
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no shared"))?
                .as_tensor(),
            self.delta_ratio,
        )?;
        let routed = if let Some(w) = &self.weight_routed {
            codes_to_i8(&ternarize_hard(w.as_tensor(), self.delta_ratio)?)?
        } else {
            Vec::new()
        };
        Ok((codes_to_i8(&base)?, codes_to_i8(&shared)?, routed))
    }

    pub fn histogram(&self) -> Result<TernaryHist> {
        if self.packed {
            if let Some(p) = &self.packed_codes {
                let mut h = histogram_tensor(&p.code_base)?;
                let h2 = histogram_tensor(&p.code_shared)?;
                h.merge(&h2, 1, 1);
                if let Some(r) = &p.code_routed {
                    let h3 = histogram_tensor(r)?;
                    h.merge(&h3, 2, 1);
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
        let device = self.scale_base.as_tensor().device();
        let code_base = i8_tensor(&b, &[self.out_features, self.in_features], device)?;
        let code_shared = i8_tensor(
            &s,
            &[self.out_features, self.in_features * self.n_shared.max(1)],
            device,
        )?;
        let code_routed = if !r.is_empty() && self.n_routed > 0 {
            Some(i8_tensor(
                &r,
                &[
                    self.n_experts.max(1),
                    self.out_features,
                    self.in_features * self.n_routed,
                ],
                device,
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
        out.push((
            "centers".into(),
            NamedBlob::F32(self.centers.as_tensor().clone()),
        ));
        out.push((
            "scale_base".into(),
            NamedBlob::F32(self.scale_base.as_tensor().clone()),
        ));
        out.push((
            "scale_shared".into(),
            NamedBlob::F32(self.scale_shared.as_tensor().clone()),
        ));
        out.push((
            "scale_routed".into(),
            NamedBlob::F32(self.scale_routed.as_tensor().clone()),
        ));
        if let Some(r) = &self.router {
            out.push(("router".into(), NamedBlob::F32(r.as_tensor().clone())));
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
                out.push(("weight_base".into(), NamedBlob::F32(w.as_tensor().clone())));
            }
            if let Some(w) = &self.weight_shared {
                out.push((
                    "weight_shared".into(),
                    NamedBlob::F32(w.as_tensor().clone()),
                ));
            }
            if let Some(w) = &self.weight_routed {
                out.push((
                    "weight_routed".into(),
                    NamedBlob::F32(w.as_tensor().clone()),
                ));
            }
        }
        Ok(out)
    }
}

pub enum NamedBlob {
    F32(Tensor),
    Packed { bytes: Vec<u8>, shape: Vec<usize> },
}

fn i8_tensor(codes: &[i8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    if codes.len() != n {
        bail!("code len {} != shape product {}", codes.len(), n);
    }
    let f: Vec<f32> = codes.iter().map(|&c| c as f32).collect();
    Ok(Tensor::from_vec(f, shape, device)?)
}

fn concat_spline(
    shared: &Tensor,
    routed: Option<&Tensor>,
    expert: usize,
    n_shared: usize,
    n_routed: usize,
    out: usize,
    in_f: usize,
    g: usize,
) -> Result<Tensor> {
    let sh = shared.reshape((out, in_f, n_shared.max(1)))?;
    if n_routed == 0 || routed.is_none() {
        // pad routed dim with zeros if G > n_shared (shouldn't happen)
        if n_shared.max(1) == g {
            return Ok(shared.reshape((out, in_f * g))?);
        }
        let z = Tensor::zeros(
            (out, in_f, g - n_shared.max(1)),
            shared.dtype(),
            shared.device(),
        )?;
        let cube = Tensor::cat(&[&sh, &z], 2)?;
        return Ok(cube.reshape((out, in_f * g))?);
    }
    let rt = routed
        .unwrap()
        .narrow(0, expert, 1)?
        .squeeze(0)?
        .reshape((out, in_f, n_routed))?;
    let cube = Tensor::cat(&[&sh, &rt], 2)?;
    Ok(cube.reshape((out, in_f * g))?)
}

fn linspace_tensor(lo: f32, hi: f32, n: usize, device: &Device) -> Result<Tensor> {
    let step = (hi - lo) / (n as f32 - 1.0);
    let v: Vec<f32> = (0..n).map(|i| lo + step * i as f32).collect();
    Ok(Tensor::from_vec(v, n, device)?)
}

fn linspace_var(lo: f32, hi: f32, n: usize, device: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&linspace_tensor(lo, hi, n, device)?)?)
}

/// `x`: [N, in] → bumps [N, in, G]
fn relu_bumps_nd(x: &Tensor, centers: &Tensor, inv_width: f64) -> Result<Tensor> {
    // (x.unsqueeze(-1) - centers) * inv_width
    let z = (x.unsqueeze(D::Minus1)?.broadcast_sub(centers)? * inv_width)?;
    let t = (1.0 - z.abs()?)?;
    Ok(t.relu()?.sqr()?)
}

fn shared_basis_used(n_shared: usize, coarse: bool) -> usize {
    if coarse {
        n_shared.clamp(1, KanEvalMode::COARSE_BASIS)
    } else {
        n_shared.max(1)
    }
}

fn slice_shared_weight(w: &Tensor, in_f: usize, gs: usize, g_use: usize) -> Result<Tensor> {
    if g_use >= gs {
        return Ok(w.clone());
    }
    let out = w.dim(0)?;
    let cube = w.reshape((out, in_f, gs))?;
    Ok(cube
        .narrow(D::Minus1, 0, g_use)?
        .reshape((out, in_f * g_use))?)
}

fn flatten_leading(x: &Tensor, in_f: usize) -> Result<Tensor> {
    let dims = x.dims();
    let last = *dims.last().unwrap_or(&in_f);
    if last != in_f {
        bail!("expected last dim {in_f}, got {last}");
    }
    let n: usize = dims.iter().rev().skip(1).product();
    Ok(x.reshape((n, in_f))?)
}

fn unflatten_leading(y: Tensor, orig: &[usize], out_f: usize) -> Result<Tensor> {
    let mut shape = orig.to_vec();
    if let Some(last) = shape.last_mut() {
        *last = out_f;
    }
    Ok(y.reshape(shape)?)
}

fn broadcast_scale(codes: &Tensor, scale: &Tensor) -> Result<Tensor> {
    // codes [out, ...] or [K, out, ...]; scale [out] or [K, out]
    match (codes.rank(), scale.rank()) {
        (2, 1) => Ok(codes.broadcast_mul(&scale.unsqueeze(1)?)?),
        (3, 2) => Ok(codes.broadcast_mul(&scale.unsqueeze(D::Minus1)?)?),
        (3, 1) => Ok(codes.broadcast_mul(&scale.reshape((1, scale.dim(0)?, 1))?)?),
        _ => {
            // fall back: try unsqueeze last
            let mut s = scale.clone();
            while s.rank() < codes.rank() {
                s = s.unsqueeze(D::Minus1)?;
            }
            Ok(codes.broadcast_mul(&s)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn layer_shapes_cpu() {
        let device = Device::Cpu;
        let mut rng = crate::device::rng_from_seed(0);
        let layer = TernaryKanLinear::new(6, 5, 4, false, 1, 0.7, &device, &mut rng).unwrap();
        let x = crate::mixers::randn(&[2, 3, 6], 1.0, &device, &mut rng).unwrap();
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 3, 5]);
    }

    #[test]
    fn coarse_masks_thinking_and_preserves_shape() {
        let device = Device::Cpu;
        let mut rng = crate::device::rng_from_seed(1);
        let layer = TernaryKanLinear::new(8, 8, 12, true, 3, 0.7, &device, &mut rng).unwrap();
        let x = crate::mixers::randn(&[2, 8], 1.0, &device, &mut rng).unwrap();
        let full = layer.forward_mode(&x, KanEvalMode::Full).unwrap();
        let coarse = layer.forward_mode(&x, KanEvalMode::Coarse).unwrap();
        assert_eq!(full.dims(), coarse.dims());
        let delta = (full - coarse)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(delta > 0.0, "routed thinking path must contribute in Full");
    }
}
