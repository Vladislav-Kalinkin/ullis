//! Gated diagonal scan + group delay. Control plane, FP32, `[B, T, D]`.

use anyhow::{bail, Result};

const ALPHA_EPS: f32 = 1e-3;
/// `logit(0.95)` so `α ≈ 0.95` at `u = 0`.
pub const ALPHA_BIAS_INIT: f32 = 2.944_439;

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn silu_grad(x: f32, dy: f32) -> f32 {
    let s = sigmoid(x);
    dy * (s + x * s * (1.0 - s))
}

#[inline]
fn sigmoid_grad(s: f32, dy: f32) -> f32 {
    dy * s * (1.0 - s)
}

fn clamp_alpha(a: f32) -> f32 {
    a.clamp(ALPHA_EPS, 1.0 - ALPHA_EPS)
}

/// Delay the last `g` channels by one token. `g = min(32, d/2)`.
pub fn group_width(d: usize) -> usize {
    if d < 2 {
        return 0;
    }
    if d >= 64 {
        32
    } else {
        d / 2
    }
}

pub fn group_delay_into(x: &[f32], b: usize, t: usize, d: usize, y: &mut [f32]) -> Result<()> {
    if x.len() != b * t * d {
        bail!("group_delay len {} != b*t*d {}", x.len(), b * t * d);
    }
    if y.len() < x.len() {
        bail!("group_delay y short");
    }
    let g = group_width(d);
    let keep = d - g;
    let y = &mut y[..x.len()];
    for bi in 0..b {
        for ti in 0..t {
            let src = (bi * t + ti) * d;
            y[src..src + keep].copy_from_slice(&x[src..src + keep]);
            if g == 0 {
                continue;
            }
            if ti == 0 {
                y[src + keep..src + d].fill(0.0);
            } else {
                let prev = (bi * t + (ti - 1)) * d + keep;
                y[src + keep..src + d].copy_from_slice(&x[prev..prev + g]);
            }
        }
    }
    Ok(())
}

pub fn group_delay_bwd_into(
    dy: &[f32],
    b: usize,
    t: usize,
    d: usize,
    dx: &mut [f32],
) -> Result<()> {
    if dy.len() != b * t * d {
        bail!("group_delay_bwd len mismatch");
    }
    if dx.len() < dy.len() {
        bail!("group_delay_bwd dx short");
    }
    let g = group_width(d);
    let keep = d - g;
    let dx = &mut dx[..dy.len()];
    dx.fill(0.0);
    for bi in 0..b {
        for ti in 0..t {
            let i = (bi * t + ti) * d;
            dx[i..i + keep].copy_from_slice(&dy[i..i + keep]);
            if g == 0 {
                continue;
            }
            if ti + 1 < t {
                let nxt = (bi * t + (ti + 1)) * d + keep;
                for k in 0..g {
                    dx[i + keep + k] += dy[nxt + k];
                }
            }
        }
    }
    Ok(())
}

/// Per-channel scan parameters.
#[derive(Clone, Debug)]
pub struct ScanParams {
    pub w_alpha: Vec<f32>,
    pub b_alpha: Vec<f32>,
    pub w_i: Vec<f32>,
    pub b_i: Vec<f32>,
    pub grad_w_alpha: Vec<f32>,
    pub grad_b_alpha: Vec<f32>,
    pub grad_w_i: Vec<f32>,
    pub grad_b_i: Vec<f32>,
    pub d: usize,
}

impl ScanParams {
    pub fn new(d: usize) -> Self {
        Self {
            w_alpha: vec![0.0; d],
            b_alpha: vec![ALPHA_BIAS_INIT; d],
            w_i: vec![0.0; d],
            b_i: vec![1.0; d],
            grad_w_alpha: vec![0.0; d],
            grad_b_alpha: vec![0.0; d],
            grad_w_i: vec![0.0; d],
            grad_b_i: vec![0.0; d],
            d,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad_w_alpha.fill(0.0);
        self.grad_b_alpha.fill(0.0);
        self.grad_w_i.fill(0.0);
        self.grad_b_i.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScanTape {
    pub u: Vec<f32>,
    pub alpha: Vec<f32>,
    pub i_gate: Vec<f32>,
    pub pre_i: Vec<f32>,
    pub h: Vec<f32>,
    pub h0: Vec<f32>,
}

/// Forward. `u` is `[B,T,D]`. `h0` is `[B,D]` or zeros.
pub fn scan_forward(
    params: &ScanParams,
    u_raw: &[f32],
    b: usize,
    t: usize,
    h0: Option<&[f32]>,
) -> Result<(Vec<f32>, ScanTape, Vec<f32>)> {
    let d = params.d;
    if u_raw.len() != b * t * d {
        bail!("scan u len");
    }
    let mut u = vec![0.0f32; u_raw.len()];
    group_delay_into(u_raw, b, t, d, &mut u)?;
    let mut h0v = vec![0.0f32; b * d];
    if let Some(h) = h0 {
        if h.len() != b * d {
            bail!("scan h0 len");
        }
        h0v.copy_from_slice(h);
    }
    let mut alpha = vec![0.0f32; b * t * d];
    let mut i_gate = vec![0.0f32; b * t * d];
    let mut pre_i = vec![0.0f32; b * t * d];
    let mut h = vec![0.0f32; b * t * d];
    let mut h_last = h0v.clone();
    for bi in 0..b {
        for ti in 0..t {
            let row = (bi * t + ti) * d;
            for c in 0..d {
                let uv = u[row + c];
                let pa = params.w_alpha[c] * uv + params.b_alpha[c];
                let pi = params.w_i[c] * uv + params.b_i[c];
                pre_i[row + c] = pi;
                let a = clamp_alpha(sigmoid(pa));
                let ig = silu(pi);
                alpha[row + c] = a;
                i_gate[row + c] = ig;
                let prev = h_last[bi * d + c];
                // Leaky sum (integrator): α ⊙ h + i ⊙ u. Convex EMA cannot count.
                let ht = a * prev + ig * uv;
                h[row + c] = ht;
                h_last[bi * d + c] = ht;
            }
        }
    }
    let tape = ScanTape {
        u,
        alpha,
        i_gate,
        pre_i,
        h: h.clone(),
        h0: h0v,
    };
    Ok((h, tape, h_last))
}

/// Incremental one-token step. `u_tok` is `[B, D]` after RMS/FWHT (no delay).
/// `prev_u` is the previous raw token (`[B,D]`) for group delay, or zeros.
pub fn scan_step(
    params: &ScanParams,
    u_tok: &[f32],
    prev_u: &[f32],
    h_prev: &[f32],
    b: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let d = params.d;
    if u_tok.len() != b * d || h_prev.len() != b * d || prev_u.len() != b * d {
        bail!("scan_step shape");
    }
    let g = group_width(d);
    let keep = d - g;
    let mut u = vec![0.0f32; b * d];
    for bi in 0..b {
        let s = bi * d;
        u[s..s + keep].copy_from_slice(&u_tok[s..s + keep]);
        if g > 0 {
            u[s + keep..s + d].copy_from_slice(&prev_u[s + keep..s + d]);
        }
    }
    let mut h = vec![0.0f32; b * d];
    for bi in 0..b {
        for c in 0..d {
            let uv = u[bi * d + c];
            let a = clamp_alpha(sigmoid(params.w_alpha[c] * uv + params.b_alpha[c]));
            let ig = silu(params.w_i[c] * uv + params.b_i[c]);
            h[bi * d + c] = a * h_prev[bi * d + c] + ig * uv;
        }
    }
    Ok((h, u_tok.to_vec()))
}

/// Backward through scan + group delay. Accumulates into `params` grads.
/// Returns `dx` w.r.t. the pre-delay input (`u_raw`).
pub fn scan_backward(
    params: &mut ScanParams,
    tape: &ScanTape,
    dh: &[f32],
    b: usize,
    t: usize,
) -> Result<Vec<f32>> {
    let d = params.d;
    let n = b * t * d;
    if dh.len() != n {
        bail!("scan_backward dh len");
    }
    let mut du = vec![0.0f32; n];
    let mut dh_carry = vec![0.0f32; b * d];
    for bi in 0..b {
        for ti in (0..t).rev() {
            let row = (bi * t + ti) * d;
            for c in 0..d {
                let ght = dh[row + c] + dh_carry[bi * d + c];
                let a = tape.alpha[row + c];
                let ig = tape.i_gate[row + c];
                let uv = tape.u[row + c];
                let prev = if ti == 0 {
                    tape.h0[bi * d + c]
                } else {
                    tape.h[(bi * t + (ti - 1)) * d + c]
                };
                let da = ght * prev;
                dh_carry[bi * d + c] = ght * a;
                let d_ig = ght * uv;
                let d_uv_z = ght * ig;
                let dpre_a = sigmoid_grad(a, da);
                let dpre_i = silu_grad(tape.pre_i[row + c], d_ig);
                params.grad_w_alpha[c] += dpre_a * uv;
                params.grad_b_alpha[c] += dpre_a;
                params.grad_w_i[c] += dpre_i * uv;
                params.grad_b_i[c] += dpre_i;
                du[row + c] += d_uv_z + dpre_a * params.w_alpha[c] + dpre_i * params.w_i[c];
            }
        }
    }
    let mut dx = vec![0.0f32; n];
    group_delay_bwd_into(&du, b, t, d, &mut dx)?;
    Ok(dx)
}

/// Mean `|∂h_T / ∂h_0|` with `u = 0` (tests long-range memory).
pub fn mean_state_gain(params: &ScanParams, t: usize) -> f32 {
    let d = params.d;
    let mut acc = 0.0f32;
    for c in 0..d {
        let a = clamp_alpha(sigmoid(params.b_alpha[c]));
        acc += a.powi(t as i32).abs();
    }
    acc / d.max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_init_near_095() {
        let p = ScanParams::new(8);
        for &b in &p.b_alpha {
            let a = clamp_alpha(sigmoid(b));
            assert!((a - 0.95).abs() < 1e-3, "alpha {a}");
        }
    }

    #[test]
    fn long_range_gain() {
        let p = ScanParams::new(8);
        let g = mean_state_gain(&p, 64);
        assert!(g > 1e-3, "gain {g}");
    }

    #[test]
    fn batch_isolation() {
        let p = ScanParams::new(4);
        let b = 2usize;
        let t = 3usize;
        let d = 4usize;
        let mut u = vec![0.0f32; b * t * d];
        for i in 0..t * d {
            u[i] = 1.0;
        }
        let (h, _, _) = scan_forward(&p, &u, b, t, None).unwrap();
        for i in 0..t * d {
            assert!(h[t * d + i].abs() < 1e-6, "batch leak {}", h[t * d + i]);
        }
    }

    #[test]
    fn leaky_sum_grows_with_opens() {
        let p = ScanParams::new(4);
        let ones = vec![1.0f32; 8 * 4];
        let (h8, _, _) = scan_forward(&p, &ones, 1, 8, None).unwrap();
        let ones3 = vec![1.0f32; 3 * 4];
        let (h3, _, _) = scan_forward(&p, &ones3, 1, 3, None).unwrap();
        let s8 = h8[7 * 4];
        let s3 = h3[2 * 4];
        assert!(s8 > s3 + 0.5, "integrator 8 opens {s8} vs 3 {s3}");
    }

    #[test]
    fn jacobian_t8() {
        let mut p = ScanParams::new(4);
        let b = 1usize;
        let t = 8usize;
        let d = 4usize;
        let mut u = vec![0.0f32; b * t * d];
        for (i, v) in u.iter_mut().enumerate() {
            *v = (i as f32) * 0.05 - 0.3;
        }
        let (h, tape, _) = scan_forward(&p, &u, b, t, None).unwrap();
        let mut dh = vec![0.0f32; h.len()];
        dh[h.len() - 1] = 1.0;
        p.zero_grad();
        let dx = scan_backward(&mut p, &tape, &dh, b, t).unwrap();
        let eps = 1e-3f32;
        let mut u2 = u.clone();
        u2[0] += eps;
        let (h2, _, _) = scan_forward(&p, &u2, b, t, None).unwrap();
        let num = (h2[h.len() - 1] - h[h.len() - 1]) / eps;
        assert!((dx[0] - num).abs() < 0.05, "dx0 {} num {num}", dx[0]);
    }
}
