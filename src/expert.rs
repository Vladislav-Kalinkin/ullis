//! Ternary top-k experts with grouped GEMM. Bumps live in W-space after `W_up`.

use anyhow::{bail, Result};

use crate::accelerate::{
    bump_grads, bump_inv_widths, relu_bumps, sgemm, sgemm_nt, sgemm_tn, softmax_rows, ternarize_row,
};
use crate::device::SovereignDevice;
use crate::mixers::rand_kaiming;
use crate::quant::ste_gate;
use crate::scan::sigmoid;

pub const BUMP_G: usize = 4;

fn gemm_nt(
    gpu: Option<&SovereignDevice>,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    if let Some(g) = gpu {
        g.gemm_nt(m, n, k, a, b, c)
    } else {
        sgemm_nt(m, n, k, 1.0, a, b, 0.0, c)
    }
}

#[derive(Clone, Debug)]
pub struct TernaryExpert {
    pub d: usize,
    pub w: usize,
    pub w_up: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_down: Vec<f32>,
    pub bumps: Vec<f32>,
    pub scale_up: Vec<f32>,
    pub scale_gate: Vec<f32>,
    pub scale_down: Vec<f32>,
    pub centers: Vec<f32>,
    pub inv_widths: Vec<f32>,
    pub grad_up: Vec<f32>,
    pub grad_gate: Vec<f32>,
    pub grad_down: Vec<f32>,
    pub grad_bumps: Vec<f32>,
    pub grad_scale_up: Vec<f32>,
    pub grad_scale_gate: Vec<f32>,
    pub grad_scale_down: Vec<f32>,
    pub phase: u8,
    pub packed: bool,
    pub delta_ratio: f32,
    pub codes_up: Option<Vec<f32>>,
    pub codes_gate: Option<Vec<f32>>,
    pub codes_down: Option<Vec<f32>>,
}

impl TernaryExpert {
    pub fn new(d: usize, w: usize, rng: &mut impl rand::Rng) -> Self {
        let centers: Vec<f32> = (0..BUMP_G)
            .map(|g| -2.0 + 4.0 * g as f32 / (BUMP_G - 1) as f32)
            .collect();
        let inv_widths = bump_inv_widths(&centers);
        Self {
            d,
            w,
            w_up: rand_kaiming(w, d, rng),
            w_gate: rand_kaiming(w, d, rng),
            w_down: rand_kaiming(d, w, rng),
            bumps: vec![0.02; w * BUMP_G],
            scale_up: vec![1.0; w],
            scale_gate: vec![1.0; w],
            scale_down: vec![1.0; d],
            centers,
            inv_widths,
            grad_up: vec![0.0; w * d],
            grad_gate: vec![0.0; w * d],
            grad_down: vec![0.0; d * w],
            grad_bumps: vec![0.0; w * BUMP_G],
            grad_scale_up: vec![0.0; w],
            grad_scale_gate: vec![0.0; w],
            grad_scale_down: vec![0.0; d],
            phase: 1,
            packed: false,
            delta_ratio: 0.7,
            codes_up: None,
            codes_gate: None,
            codes_down: None,
        }
    }

    fn tern_mat(&self, w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w.len()];
        for r in 0..rows {
            let mut row = vec![0.0f32; cols];
            ternarize_row(&w[r * cols..(r + 1) * cols], self.delta_ratio, &mut row);
            out[r * cols..(r + 1) * cols].copy_from_slice(&row);
        }
        out
    }

    /// Freeze ternary codes. Phase 4 trains scales only.
    pub fn pack(&mut self) {
        self.codes_up = Some(self.tern_mat(&self.w_up, self.w, self.d));
        self.codes_gate = Some(self.tern_mat(&self.w_gate, self.w, self.d));
        self.codes_down = Some(self.tern_mat(&self.w_down, self.d, self.w));
        self.packed = true;
        self.phase = 4;
    }

    pub fn zero_grad(&mut self) {
        self.grad_up.fill(0.0);
        self.grad_gate.fill(0.0);
        self.grad_down.fill(0.0);
        self.grad_bumps.fill(0.0);
        self.grad_scale_up.fill(0.0);
        self.grad_scale_gate.fill(0.0);
        self.grad_scale_down.fill(0.0);
    }

    fn qat(&self) -> bool {
        self.phase >= 3 && !self.packed
    }

    fn scale_codes(codes: &[f32], rows: usize, cols: usize, scale: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; codes.len()];
        for r in 0..rows {
            let s = scale[r];
            for c in 0..cols {
                out[r * cols + c] = codes[r * cols + c] * s;
            }
        }
        out
    }

    fn quant_mat(
        &self,
        w: &[f32],
        rows: usize,
        cols: usize,
        scale: &[f32],
        frozen: Option<&Vec<f32>>,
    ) -> Result<Vec<f32>> {
        if let Some(codes) = frozen {
            return Ok(Self::scale_codes(codes, rows, cols, scale));
        }
        if !self.qat() && !self.packed {
            return Ok(w.to_vec());
        }
        let codes = self.tern_mat(w, rows, cols);
        Ok(Self::scale_codes(&codes, rows, cols, scale))
    }

    pub fn forward(
        &self,
        x: &[f32],
        n: usize,
        gpu: Option<&SovereignDevice>,
    ) -> Result<(Vec<f32>, ExpertCache)> {
        let d = self.d;
        let w = self.w;
        if x.len() != n * d {
            bail!("expert x len");
        }
        let up = self.quant_mat(
            &self.w_up,
            w,
            d,
            &self.scale_up,
            self.codes_up.as_ref(),
        )?;
        let gate = self.quant_mat(
            &self.w_gate,
            w,
            d,
            &self.scale_gate,
            self.codes_gate.as_ref(),
        )?;
        let down = self.quant_mat(
            &self.w_down,
            d,
            w,
            &self.scale_down,
            self.codes_down.as_ref(),
        )?;
        let mut p = vec![0.0f32; n * w];
        gemm_nt(gpu, n, w, d, x, &up, &mut p)?;
        let mut gpre = vec![0.0f32; n * w];
        gemm_nt(gpu, n, w, d, x, &gate, &mut gpre)?;
        let mut psi = vec![0.0f32; n * w * BUMP_G];
        relu_bumps(&p, n, w, &self.centers, &self.inv_widths, &mut psi)?;
        let mut h = vec![0.0f32; n * w];
        for t in 0..n {
            for j in 0..w {
                let mut bump = 0.0f32;
                let base = (t * w + j) * BUMP_G;
                for g in 0..BUMP_G {
                    bump += self.bumps[j * BUMP_G + g] * psi[base + g];
                }
                let gate_v = sigmoid(gpre[t * w + j]);
                h[t * w + j] = (p[t * w + j] + bump) * gate_v;
            }
        }
        let mut y = vec![0.0f32; n * d];
        gemm_nt(gpu, n, d, w, &h, &down, &mut y)?;
        Ok((
            y,
            ExpertCache {
                x: x.to_vec(),
                p,
                gpre,
                h,
            },
        ))
    }

    pub fn backward(&mut self, cache: &ExpertCache, dy: &[f32], n: usize) -> Result<Vec<f32>> {
        let d = self.d;
        let w = self.w;
        if dy.len() != n * d {
            bail!("expert dy");
        }
        let down = self.quant_mat(
            &self.w_down,
            d,
            w,
            &self.scale_down,
            self.codes_down.as_ref(),
        )?;
        let mut dh = vec![0.0f32; n * w];
        sgemm(n, w, d, 1.0, dy, &down, 0.0, &mut dh)?;
        sgemm_tn(d, w, n, 1.0, dy, &cache.h, 1.0, &mut self.grad_down)?;
        if self.qat() {
            for r in 0..d {
                let row = &self.w_down[r * w..(r + 1) * w];
                for c in 0..w {
                    self.grad_down[r * w + c] *= ste_gate(row[c]);
                }
            }
        }

        let mut psi = vec![0.0f32; n * w * BUMP_G];
        relu_bumps(
            &cache.p,
            n,
            w,
            &self.centers,
            &self.inv_widths,
            &mut psi,
        )?;
        let up = self.quant_mat(
            &self.w_up,
            w,
            d,
            &self.scale_up,
            self.codes_up.as_ref(),
        )?;
        let gate = self.quant_mat(
            &self.w_gate,
            w,
            d,
            &self.scale_gate,
            self.codes_gate.as_ref(),
        )?;

        let mut dp = vec![0.0f32; n * w];
        let mut dgate_pre = vec![0.0f32; n * w];
        for t in 0..n {
            for j in 0..w {
                let mut bump = 0.0f32;
                let base = (t * w + j) * BUMP_G;
                for g in 0..BUMP_G {
                    bump += self.bumps[j * BUMP_G + g] * psi[base + g];
                }
                let pre = cache.gpre[t * w + j];
                let gv = sigmoid(pre);
                let inner = cache.p[t * w + j] + bump;
                let gdh = dh[t * w + j];
                dp[t * w + j] += gdh * gv;
                dgate_pre[t * w + j] += gdh * inner * gv * (1.0 - gv);
                for g in 0..BUMP_G {
                    let dpsi = gdh * gv * self.bumps[j * BUMP_G + g];
                    self.grad_bumps[j * BUMP_G + g] += gdh * gv * psi[base + g];
                    let mut dc = 0.0f32;
                    bump_grads(
                        cache.p[t * w + j],
                        self.centers[g],
                        self.inv_widths[g],
                        dpsi,
                        &mut dp[t * w + j],
                        &mut dc,
                    );
                }
            }
        }

        let mut dx = vec![0.0f32; n * d];
        sgemm(n, d, w, 1.0, &dp, &up, 0.0, &mut dx)?;
        sgemm_tn(w, d, n, 1.0, &dp, &cache.x, 1.0, &mut self.grad_up)?;
        sgemm(n, d, w, 1.0, &dgate_pre, &gate, 1.0, &mut dx)?;
        sgemm_tn(w, d, n, 1.0, &dgate_pre, &cache.x, 1.0, &mut self.grad_gate)?;
        if self.qat() {
            for r in 0..w {
                for c in 0..d {
                    self.grad_up[r * d + c] *= ste_gate(self.w_up[r * d + c]);
                    self.grad_gate[r * d + c] *= ste_gate(self.w_gate[r * d + c]);
                }
            }
        }
        Ok(dx)
    }
}

#[derive(Clone, Debug)]
pub struct MoeCache {
    pub x: Vec<f32>,
    pub full: Vec<f32>,
    pub sparse: Vec<f32>,
    pub caches: Vec<Option<ExpertCache>>,
    pub groups: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
pub struct ExpertCache {
    x: Vec<f32>,
    p: Vec<f32>,
    gpre: Vec<f32>,
    h: Vec<f32>,
}

/// Top-k over `E` experts. Unlike `apply_topk_gates`, `E` is not capped at 4.
pub fn topk_rows(gates: &mut [f32], n: usize, e: usize, k: usize) {
    if k == 0 || k >= e {
        return;
    }
    for t in 0..n {
        let row = &mut gates[t * e..t * e + e];
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, &i) in idx.iter().enumerate() {
            if rank >= k {
                row[i] = 0.0;
            }
        }
        let z: f32 = row.iter().sum();
        if z > 1e-12 {
            let inv = 1.0 / z;
            for g in row.iter_mut() {
                *g *= inv;
            }
        }
    }
}

pub fn switch_aux_vec(full: &[f32], sparse: &[f32], n: usize, e: usize, alpha: f32) -> (f32, Vec<f32>) {
    let mut p = vec![0.0f32; e];
    let mut f = vec![0.0f32; e];
    if n == 0 || e == 0 || alpha == 0.0 {
        return (0.0, p);
    }
    let inv = 1.0 / n as f32;
    for t in 0..n {
        for i in 0..e {
            p[i] += full[t * e + i] * inv;
            if sparse[t * e + i] > 0.0 {
                f[i] += inv;
            }
        }
    }
    let nk = e as f32;
    let mut aux = 0.0f32;
    let mut dp = vec![0.0f32; e];
    for i in 0..e {
        aux += f[i] * p[i];
        dp[i] = alpha * nk * f[i];
    }
    (aux * alpha * nk, dp)
}

/// Route, group tokens per expert, skip empty experts.
pub fn moe_forward(
    experts: &[TernaryExpert],
    router: &[f32],
    x: &[f32],
    n: usize,
    d: usize,
    k: usize,
    gpu: Option<&SovereignDevice>,
) -> Result<(Vec<f32>, MoeCache)> {
    let e = experts.len();
    if e == 0 {
        return Ok((
            vec![0.0; n * d],
            MoeCache {
                x: x.to_vec(),
                full: Vec::new(),
                sparse: Vec::new(),
                caches: Vec::new(),
                groups: Vec::new(),
            },
        ));
    }
    if router.len() != e * d {
        bail!("router len");
    }
    let mut logits = vec![0.0f32; n * e];
    gemm_nt(gpu, n, e, d, x, router, &mut logits)?;
    softmax_rows(&mut logits, n, e)?;
    let full = logits.clone();
    let mut sparse = logits;
    topk_rows(&mut sparse, n, e, k);
    let mut y = vec![0.0f32; n * d];
    let mut caches = Vec::with_capacity(e);
    let mut groups = Vec::with_capacity(e);
    for ei in 0..e {
        let mut idx = Vec::new();
        for t in 0..n {
            if sparse[t * e + ei] > 0.0 {
                idx.push(t);
            }
        }
        if idx.is_empty() {
            caches.push(None);
            groups.push(idx);
            continue;
        }
        let mut xe = vec![0.0f32; idx.len() * d];
        for (row, &t) in idx.iter().enumerate() {
            xe[row * d..(row + 1) * d].copy_from_slice(&x[t * d..(t + 1) * d]);
        }
        let (ye, cache) = experts[ei].forward(&xe, idx.len(), gpu)?;
        for (row, &t) in idx.iter().enumerate() {
            let g = sparse[t * e + ei];
            for c in 0..d {
                y[t * d + c] += g * ye[row * d + c];
            }
        }
        caches.push(Some(cache));
        groups.push(idx);
    }
    Ok((
        y,
        MoeCache {
            x: x.to_vec(),
            full,
            sparse,
            caches,
            groups,
        },
    ))
}

pub fn moe_backward(
    experts: &mut [TernaryExpert],
    router: &[f32],
    grad_router: &mut [f32],
    cache: &MoeCache,
    dy: &[f32],
    n: usize,
    d: usize,
    aux_alpha: f32,
) -> Result<Vec<f32>> {
    let e = experts.len();
    let mut dx = vec![0.0f32; n * d];
    if e == 0 {
        return Ok(dx);
    }
    let mut dlogits = vec![0.0f32; n * e];
    for ei in 0..e {
        let idx = &cache.groups[ei];
        if idx.is_empty() {
            continue;
        }
        let Some(ec) = cache.caches[ei].as_ref() else {
            continue;
        };
        let mut dye = vec![0.0f32; idx.len() * d];
        for (row, &t) in idx.iter().enumerate() {
            let g = cache.sparse[t * e + ei];
            for c in 0..d {
                dye[row * d + c] = g * dy[t * d + c];
            }
        }
        let dxe = experts[ei].backward(ec, &dye, idx.len())?;
        for (row, &t) in idx.iter().enumerate() {
            let g = cache.sparse[t * e + ei];
            for c in 0..d {
                dx[t * d + c] += g * dxe[row * d + c];
            }
        }
        let ye = {
            let mut ye = vec![0.0f32; idx.len() * d];
            let down = experts[ei].quant_mat(
                &experts[ei].w_down,
                d,
                experts[ei].w,
                &experts[ei].scale_down,
                experts[ei].codes_down.as_ref(),
            )?;
            sgemm_nt(idx.len(), d, experts[ei].w, 1.0, &ec.h, &down, 0.0, &mut ye)?;
            ye
        };
        for (row, &t) in idx.iter().enumerate() {
            let mut dot = 0.0f32;
            for c in 0..d {
                dot += dy[t * d + c] * ye[row * d + c];
            }
            dlogits[t * e + ei] += dot;
        }
    }
    let (_, dp) = switch_aux_vec(&cache.full, &cache.sparse, n, e, aux_alpha);
    for t in 0..n {
        for ei in 0..e {
            dlogits[t * e + ei] += dp[ei] / n.max(1) as f32;
        }
    }
    // sparse top-k is a hard mask: backprop through the full softmax only
    // on selected experts (STE on the mask).
    let mut dfull = dlogits.clone();
    for t in 0..n {
        for ei in 0..e {
            if cache.sparse[t * e + ei] == 0.0 {
                dfull[t * e + ei] = 0.0;
            }
        }
        let row = &cache.full[t * e..t * e + e];
        let dy_row = dfull[t * e..t * e + e].to_vec();
        let mut dx_row = vec![0.0f32; e];
        softmax_bwd_row(row, &dy_row, &mut dx_row);
        dfull[t * e..t * e + e].copy_from_slice(&dx_row);
    }
    sgemm_tn(e, d, n, 1.0, &dfull, &cache.x, 1.0, grad_router)?;
    sgemm(n, d, e, 1.0, &dfull, router, 1.0, &mut dx)?;
    Ok(dx)
}

fn softmax_bwd_row(y: &[f32], dy: &[f32], dx: &mut [f32]) {
    let mut acc = 0.0f32;
    for i in 0..y.len() {
        acc += y[i] * dy[i];
    }
    for i in 0..y.len() {
        dx[i] = y[i] * (dy[i] - acc);
    }
}

fn expert_weight_count(experts: &[TernaryExpert]) -> usize {
    experts
        .iter()
        .map(|e| e.w_up.len() + e.w_gate.len() + e.w_down.len() + e.bumps.len())
        .sum()
}

/// Mean |w| over expert maps, matching KAN `l1_penalty`. The previous sum
/// made `l1=1e-3` a ~200 nats term on a 10M model and dominated the clip.
pub fn l1_experts(experts: &[TernaryExpert]) -> f32 {
    let n = expert_weight_count(experts);
    if n == 0 {
        return 0.0;
    }
    let mut s = 0.0f32;
    for e in experts {
        for w in [&e.w_up, &e.w_gate, &e.w_down, &e.bumps] {
            for &v in w {
                s += v.abs();
            }
        }
    }
    s / n as f32
}

pub fn l1_experts_grad(experts: &mut [TernaryExpert], coef: f32) {
    if coef == 0.0 {
        return;
    }
    let n = expert_weight_count(experts);
    if n == 0 {
        return;
    }
    let gscale = coef / n as f32;
    for e in experts {
        for (w, g) in [
            (&e.w_up, &mut e.grad_up),
            (&e.w_gate, &mut e.grad_gate),
            (&e.w_down, &mut e.grad_down),
            (&e.bumps, &mut e.grad_bumps),
        ] {
            for i in 0..w.len() {
                g[i] += gscale * w[i].signum();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::rng_from_seed;

    #[test]
    fn l1_is_mean_not_sum() {
        let mut rng = rng_from_seed(0);
        let e = TernaryExpert::new(8, 4, &mut rng);
        let n = e.w_up.len() + e.w_gate.len() + e.w_down.len() + e.bumps.len();
        let mean = l1_experts(std::slice::from_ref(&e));
        let mut sum = 0.0f32;
        for w in [&e.w_up, &e.w_gate, &e.w_down, &e.bumps] {
            for &v in w {
                sum += v.abs();
            }
        }
        assert!((mean - sum / n as f32).abs() < 1e-6);
        assert!(mean < 2.0, "mean |w| should be O(1), got {mean}");
    }
}
