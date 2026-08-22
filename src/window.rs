//! Causal local mix. Content scores over a **fixed** lag window, never `T×T`.
//!
//! For each token `t`, keys/values are `x_{t-1} … x_{t-W}` only.
//! Compute `Θ(B T W D)`, tape `Θ(B T W)`. `W` is a constant cap
//! ([`MAX_WINDOW`]), independent of `seq_len` / `context_len`.

use anyhow::{bail, Result};

/// Hard cap so `--window` cannot silently become `T` as sequences grow.
pub const MAX_WINDOW: usize = 64;
/// Content term is a perturbation on lag bias. Unscaled `q·k/√D` at RMSNorm
/// residuals is O(√D) and wipes the copy-lag kernel.
const CONTENT_SCALE: f32 = 0.05;

pub fn clamp_width(w: usize) -> usize {
    w.min(MAX_WINDOW)
}

#[derive(Clone, Debug)]
pub struct WindowParams {
    pub width: usize,
    pub d: usize,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub b_lag: Vec<f32>,
    pub gamma: f32,
    pub grad_w_q: Vec<f32>,
    pub grad_w_k: Vec<f32>,
    pub grad_b_lag: Vec<f32>,
    pub grad_gamma: f32,
}

impl WindowParams {
    pub fn new(d: usize, width: usize) -> Self {
        let width = clamp_width(width);
        Self {
            width,
            d,
            w_q: vec![1.0; d],
            w_k: vec![1.0; d],
            b_lag: vec![0.0; width],
            gamma: 0.1,
            grad_w_q: vec![0.0; d],
            grad_w_k: vec![0.0; d],
            grad_b_lag: vec![0.0; width],
            grad_gamma: 0.0,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad_w_q.fill(0.0);
        self.grad_w_k.fill(0.0);
        self.grad_b_lag.fill(0.0);
        self.grad_gamma = 0.0;
    }
}

#[derive(Clone, Debug, Default)]
pub struct WindowTape {
    pub x: Vec<f32>,
    pub attn: Vec<f32>,
    pub y: Vec<f32>,
    pub b: usize,
    pub t: usize,
    pub d: usize,
    pub w: usize,
}

/// Ring of the last `W` residual rows for incremental decode. `Θ(W D)`.
#[derive(Clone, Debug)]
pub struct WindowRing {
    buf: Vec<f32>,
    d: usize,
    cap: usize,
    len: usize,
    pos: usize,
}

impl WindowRing {
    pub fn new(d: usize, width: usize) -> Self {
        let cap = clamp_width(width);
        Self {
            buf: vec![0.0; cap.saturating_mul(d)],
            d,
            cap,
            len: 0,
            pos: 0,
        }
    }

    fn lag_row(&self, lag: usize) -> Option<&[f32]> {
        if lag >= self.len || self.cap == 0 {
            return None;
        }
        let i = (self.pos + self.cap - 1 - lag) % self.cap;
        Some(&self.buf[i * self.d..(i + 1) * self.d])
    }

    fn push(&mut self, x: &[f32]) {
        if self.cap == 0 || x.len() < self.d {
            return;
        }
        let off = self.pos * self.d;
        self.buf[off..off + self.d].copy_from_slice(&x[..self.d]);
        self.pos = (self.pos + 1) % self.cap;
        if self.len < self.cap {
            self.len += 1;
        }
    }
}

fn inv_sqrt_d(d: usize) -> f32 {
    (d.max(1) as f32).sqrt().recip()
}

fn softmax_n(z: &mut [f32], n: usize) {
    if n == 0 {
        return;
    }
    let m = z[..n].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in z.iter_mut().take(n) {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s.max(1e-20);
    for v in z.iter_mut().take(n) {
        *v *= inv;
    }
}

/// `x` is `[B, T, D]`. Output same shape. Attends only to previous `W` tokens.
pub fn window_forward(
    params: &WindowParams,
    x: &[f32],
    b: usize,
    t: usize,
) -> Result<(Vec<f32>, WindowTape)> {
    let d = params.d;
    let w = params.width;
    let n = b.saturating_mul(t);
    if w == 0 {
        return Ok((
            vec![0.0; n.saturating_mul(d)],
            WindowTape {
                x: x.to_vec(),
                attn: Vec::new(),
                y: vec![0.0; n.saturating_mul(d)],
                b,
                t,
                d,
                w: 0,
            },
        ));
    }
    if x.len() != n.saturating_mul(d) {
        bail!("window x len {} != b*t*d {}", x.len(), n * d);
    }
    let scale = inv_sqrt_d(d);
    let mut y = vec![0.0f32; n * d];
    let mut attn = vec![0.0f32; n * w];
    let mut q = vec![0.0f32; d];
    let mut k = vec![0.0f32; d];
    let mut z = vec![0.0f32; w];
    for bi in 0..b {
        for ti in 0..t {
            let row = (bi * t + ti) * d;
            let xt = &x[row..row + d];
            for c in 0..d {
                q[c] = params.w_q[c] * xt[c];
            }
            let nctx = ti.min(w);
            if nctx == 0 {
                continue;
            }
            for lag in 0..nctx {
                let j = ti - 1 - lag;
                let xj = &x[(bi * t + j) * d..(bi * t + j) * d + d];
                for c in 0..d {
                    k[c] = params.w_k[c] * xj[c];
                }
                let mut dot = 0.0f32;
                for c in 0..d {
                    dot += q[c] * k[c];
                }
                z[lag] = dot * scale * CONTENT_SCALE + params.b_lag[lag];
            }
            softmax_n(&mut z, nctx);
            let arow = (bi * t + ti) * w;
            attn[arow..arow + nctx].copy_from_slice(&z[..nctx]);
            let yrow = &mut y[row..row + d];
            for lag in 0..nctx {
                let j = ti - 1 - lag;
                let xj = &x[(bi * t + j) * d..(bi * t + j) * d + d];
                let p = z[lag] * params.gamma;
                for c in 0..d {
                    yrow[c] += p * xj[c];
                }
            }
        }
    }
    let tape = WindowTape {
        x: x.to_vec(),
        attn,
        y: y.clone(),
        b,
        t,
        d,
        w,
    };
    Ok((y, tape))
}

pub fn window_backward(
    params: &mut WindowParams,
    tape: &WindowTape,
    dy: &[f32],
) -> Result<Vec<f32>> {
    let d = params.d;
    let w = params.width;
    let b = tape.b;
    let t = tape.t;
    let n = b.saturating_mul(t);
    if dy.len() != n.saturating_mul(d) {
        bail!("window dy len");
    }
    let mut dx = vec![0.0f32; n * d];
    if w == 0 {
        return Ok(dx);
    }
    let scale = inv_sqrt_d(d);
    let mut q = vec![0.0f32; d];
    let mut k = vec![0.0f32; d];
    let mut dp = vec![0.0f32; w];
    let mut dz = vec![0.0f32; w];
    for bi in 0..b {
        for ti in 0..t {
            let row = (bi * t + ti) * d;
            let dyt = &dy[row..row + d];
            let xt = &tape.x[row..row + d];
            let nctx = ti.min(w);
            if nctx == 0 {
                continue;
            }
            for c in 0..d {
                q[c] = params.w_q[c] * xt[c];
            }
            let arow = (bi * t + ti) * w;
            let mut ydot = 0.0f32;
            let yhat = &tape.y[row..row + d];
            // y = gamma * sum p v, so d_gamma += dy · (y / gamma) but y already includes gamma.
            if params.gamma.abs() > 1e-12 {
                for c in 0..d {
                    ydot += dyt[c] * yhat[c];
                }
                params.grad_gamma += ydot / params.gamma;
            }
            for lag in 0..nctx {
                let j = ti - 1 - lag;
                let xj = &tape.x[(bi * t + j) * d..(bi * t + j) * d + d];
                let p = tape.attn[arow + lag];
                let mut dpv = 0.0f32;
                let g = p * params.gamma;
                for c in 0..d {
                    dx[(bi * t + j) * d + c] += g * dyt[c];
                    dpv += dyt[c] * xj[c];
                }
                dp[lag] = dpv * params.gamma;
            }
            let mut pdot = 0.0f32;
            for lag in 0..nctx {
                pdot += tape.attn[arow + lag] * dp[lag];
            }
            for lag in 0..nctx {
                dz[lag] = tape.attn[arow + lag] * (dp[lag] - pdot);
            }
            for lag in 0..nctx {
                params.grad_b_lag[lag] += dz[lag];
                let j = ti - 1 - lag;
                let xj = &tape.x[(bi * t + j) * d..(bi * t + j) * d + d];
                for c in 0..d {
                    k[c] = params.w_k[c] * xj[c];
                }
                let g = dz[lag] * scale * CONTENT_SCALE;
                for c in 0..d {
                    let dq = g * k[c];
                    let dk = g * q[c];
                    params.grad_w_q[c] += dq * xt[c];
                    dx[row + c] += dq * params.w_q[c];
                    params.grad_w_k[c] += dk * xj[c];
                    dx[(bi * t + j) * d + c] += dk * params.w_k[c];
                }
            }
        }
    }
    Ok(dx)
}

/// One-token step. Attends to the ring (previous tokens), then pushes `x`.
pub fn window_step(params: &WindowParams, x: &[f32], ring: &mut WindowRing) -> Result<Vec<f32>> {
    let d = params.d;
    if x.len() != d {
        bail!("window_step x len");
    }
    let mut y = vec![0.0f32; d];
    let w = params.width;
    if w == 0 {
        ring.push(x);
        return Ok(y);
    }
    let nctx = ring.len.min(w);
    if nctx == 0 {
        ring.push(x);
        return Ok(y);
    }
    let scale = inv_sqrt_d(d);
    let mut q = vec![0.0f32; d];
    for c in 0..d {
        q[c] = params.w_q[c] * x[c];
    }
    let mut z = vec![0.0f32; nctx];
    let mut ks: Vec<Vec<f32>> = Vec::with_capacity(nctx);
    for lag in 0..nctx {
        let xj = ring.lag_row(lag).expect("ring lag");
        let mut k = vec![0.0f32; d];
        let mut dot = 0.0f32;
        for c in 0..d {
            k[c] = params.w_k[c] * xj[c];
            dot += q[c] * k[c];
        }
        z[lag] = dot * scale * CONTENT_SCALE + params.b_lag[lag];
        ks.push(xj.to_vec());
    }
    softmax_n(&mut z, nctx);
    for lag in 0..nctx {
        let p = z[lag] * params.gamma;
        let xj = &ks[lag];
        for c in 0..d {
            y[c] += p * xj[c];
        }
    }
    ring.push(x);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tape_is_btw_not_tt() {
        let p = WindowParams::new(8, 16);
        let b = 2usize;
        let t = 32usize;
        let x = vec![0.1f32; b * t * 8];
        let (_, tape) = window_forward(&p, &x, b, t).unwrap();
        assert_eq!(tape.attn.len(), b * t * 16);
        assert_ne!(tape.attn.len(), b * t * t);
    }

    #[test]
    fn width_never_tracks_t() {
        assert_eq!(clamp_width(32_768), MAX_WINDOW);
        assert_eq!(clamp_width(8), 8);
        let p = WindowParams::new(4, 4096);
        assert_eq!(p.width, MAX_WINDOW);
        assert_eq!(p.b_lag.len(), MAX_WINDOW);
    }

    #[test]
    fn t0_is_zero_and_causal() {
        let p = WindowParams::new(4, 8);
        let t = 6usize;
        let mut x = vec![0.0f32; t * 4];
        for i in 0..t {
            x[i * 4] = (i + 1) as f32;
        }
        let (y, _) = window_forward(&p, &x, 1, t).unwrap();
        assert!(y[..4].iter().all(|v| *v == 0.0));
        let y_base = y[2 * 4];
        x[5 * 4] = 99.0;
        let (y2, _) = window_forward(&p, &x, 1, t).unwrap();
        assert!((y2[2 * 4] - y_base).abs() < 1e-6);
    }

    #[test]
    fn step_matches_forward_last() {
        let p = WindowParams::new(4, 8);
        let t = 5usize;
        let mut x = vec![0.0f32; t * 4];
        for i in 0..t {
            x[i * 4] = (i as f32) * 0.3 + 0.2;
            x[i * 4 + 1] = 0.1 * i as f32;
        }
        let (y, _) = window_forward(&p, &x, 1, t).unwrap();
        let mut ring = WindowRing::new(4, 8);
        let mut last = vec![0.0f32; 4];
        for i in 0..t {
            last = window_step(&p, &x[i * 4..(i + 1) * 4], &mut ring).unwrap();
        }
        for c in 0..4 {
            assert!((last[c] - y[(t - 1) * 4 + c]).abs() < 1e-5);
        }
    }
}
