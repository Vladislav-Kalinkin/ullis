//! Causal mixers and host RNG init (no Candle).

use anyhow::Result;

use crate::accelerate::{sgemm, sgemm_nt, softmax_rows};

/// Parameter-free token mix: delay half the channels by one step.
/// `x` is `[b, t, c]` row-major.
pub fn causal_shift(x: &[f32], b: usize, t: usize, c: usize) -> Result<Vec<f32>> {
    if x.len() != b * t * c {
        anyhow::bail!("causal_shift len {} != b*t*c {}", x.len(), b * t * c);
    }
    let split = c / 2;
    let mut y = vec![0.0f32; x.len()];
    for bi in 0..b {
        for ti in 0..t {
            let src = (bi * t + ti) * c;
            let dst = src;
            y[dst..dst + split].copy_from_slice(&x[src..src + split]);
            if ti == 0 {
                // delayed half is zeros (already)
            } else {
                let prev = (bi * t + (ti - 1)) * c + split;
                y[dst + split..dst + c].copy_from_slice(&x[prev..prev + (c - split)]);
            }
        }
    }
    Ok(y)
}

/// `dx` for `y = causal_shift(x)`. `dy` is `[b,t,c]`.
pub fn causal_shift_backward(dy: &[f32], b: usize, t: usize, c: usize) -> Result<Vec<f32>> {
    if dy.len() != b * t * c {
        anyhow::bail!("causal_shift_backward len mismatch");
    }
    let split = c / 2;
    let mut dx = vec![0.0f32; dy.len()];
    for bi in 0..b {
        for ti in 0..t {
            let i = (bi * t + ti) * c;
            for k in 0..split {
                dx[i + k] += dy[i + k];
            }
            if ti + 1 < t {
                let nxt = (bi * t + (ti + 1)) * c + split;
                for k in 0..(c - split) {
                    dx[i + split + k] += dy[nxt + k];
                }
            }
        }
    }
    Ok(dx)
}

pub struct CausalAttention {
    pub n_heads: usize,
    pub head_dim: usize,
    pub qkv: Vec<f32>,  // [3d, d]
    pub proj: Vec<f32>, // [d, d]
    pub d_model: usize,
}

impl CausalAttention {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut impl rand::Rng) -> Result<Self> {
        if d_model % n_heads != 0 {
            anyhow::bail!("d_model must be divisible by n_heads");
        }
        let head_dim = d_model / n_heads;
        Ok(Self {
            n_heads,
            head_dim,
            qkv: rand_kaiming(3 * d_model, d_model, rng),
            proj: rand_kaiming(d_model, d_model, rng),
            d_model,
        })
    }

    pub fn forward(&self, x: &[f32], b: usize, t: usize) -> Result<Vec<f32>> {
        let d = self.d_model;
        if x.len() != b * t * d {
            anyhow::bail!("attn x len");
        }
        let mut qkv = vec![0.0f32; b * t * 3 * d];
        sgemm_nt(b * t, 3 * d, d, 1.0, x, &self.qkv, 0.0, &mut qkv)?;
        let hd = self.head_dim;
        let h = self.n_heads;
        let scale = (hd as f32).sqrt().recip();
        let mut out = vec![0.0f32; b * t * d];
        for bi in 0..b {
            for hi in 0..h {
                let mut q = vec![0.0f32; t * hd];
                let mut k = vec![0.0f32; t * hd];
                let mut v = vec![0.0f32; t * hd];
                for ti in 0..t {
                    let row = (bi * t + ti) * 3 * d;
                    q[ti * hd..(ti + 1) * hd]
                        .copy_from_slice(&qkv[row + hi * hd..row + hi * hd + hd]);
                    let kb = row + d + hi * hd;
                    k[ti * hd..(ti + 1) * hd].copy_from_slice(&qkv[kb..kb + hd]);
                    let vb = row + 2 * d + hi * hd;
                    v[ti * hd..(ti + 1) * hd].copy_from_slice(&qkv[vb..vb + hd]);
                }
                let mut attn = vec![0.0f32; t * t];
                sgemm_nt(t, t, hd, scale, &q, &k, 0.0, &mut attn)?;
                for i in 0..t {
                    for j in (i + 1)..t {
                        attn[i * t + j] = f32::NEG_INFINITY;
                    }
                }
                softmax_rows(&mut attn, t, t)?;
                let mut head = vec![0.0f32; t * hd];
                sgemm(t, hd, t, 1.0, &attn, &v, 0.0, &mut head)?;
                for ti in 0..t {
                    let dst = (bi * t + ti) * d + hi * hd;
                    out[dst..dst + hd].copy_from_slice(&head[ti * hd..(ti + 1) * hd]);
                }
            }
        }
        let mut y = vec![0.0f32; b * t * d];
        sgemm_nt(b * t, d, d, 1.0, &out, &self.proj, 0.0, &mut y)?;
        Ok(y)
    }
}

pub fn rand_kaiming(out: usize, in_f: usize, rng: &mut impl rand::Rng) -> Vec<f32> {
    let bound = (in_f as f32).sqrt().recip();
    (0..out * in_f)
        .map(|_| rng.random_range(-bound..bound))
        .collect()
}

pub fn rand_uniform(n: usize, lo: f32, hi: f32, rng: &mut impl rand::Rng) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(lo..hi)).collect()
}

pub fn randn(n: usize, std: f32, rng: &mut impl rand::Rng) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let u: f32 = rng.random::<f32>().max(1e-7);
            let v: f32 = rng.random::<f32>();
            std * (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
        })
        .collect()
}

/// RMSNorm last dim. `x` `[n, d]`, `weight` `[d]`.
pub fn rmsnorm(x: &[f32], n: usize, d: usize, weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.len() != n * d || weight.len() != d {
        anyhow::bail!("rmsnorm shape");
    }
    let mut y = vec![0.0f32; x.len()];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mut ms = 0.0f32;
        for &v in row {
            ms += v * v;
        }
        let inv = (ms / d as f32 + eps).sqrt().recip();
        let dst = &mut y[i * d..(i + 1) * d];
        for j in 0..d {
            dst[j] = row[j] * inv * weight[j];
        }
    }
    Ok(y)
}

/// Backward of RMSNorm. Returns `(dx, dw)`.
pub fn rmsnorm_backward(
    x: &[f32],
    dy: &[f32],
    n: usize,
    d: usize,
    weight: &[f32],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw = vec![0.0f32; d];
    for i in 0..n {
        let xr = &x[i * d..(i + 1) * d];
        let g = &dy[i * d..(i + 1) * d];
        let mut ms = 0.0f32;
        for &v in xr {
            ms += v * v;
        }
        let rms = (ms / d as f32 + eps).sqrt();
        let inv = rms.recip();
        let mut dinv = 0.0f32;
        for j in 0..d {
            dw[j] += g[j] * xr[j] * inv;
            dx[i * d + j] += g[j] * weight[j] * inv;
            dinv += g[j] * xr[j] * weight[j];
        }
        // rms = sqrt(mean sq + eps); d(inv)/d(ms) = -0.5 * inv^3 / d
        let dms = dinv * (-0.5) * inv * inv * inv / d as f32;
        for j in 0..d {
            dx[i * d + j] += 2.0 * xr[j] * dms;
        }
    }
    Ok((dx, dw))
}

pub fn embed_lookup(table: &[f32], vocab: usize, d: usize, ids: &[u32]) -> Result<Vec<f32>> {
    if table.len() != vocab * d {
        anyhow::bail!("embed table len");
    }
    let mut y = vec![0.0f32; ids.len() * d];
    for (t, &id) in ids.iter().enumerate() {
        let i = (id as usize).min(vocab.saturating_sub(1));
        y[t * d..(t + 1) * d].copy_from_slice(&table[i * d..(i + 1) * d]);
    }
    Ok(y)
}

pub fn embed_scatter(vocab: usize, d: usize, ids: &[u32], dy: &[f32]) -> Result<Vec<f32>> {
    let mut dw = vec![0.0f32; vocab * d];
    for (t, &id) in ids.iter().enumerate() {
        let i = (id as usize).min(vocab.saturating_sub(1));
        let src = &dy[t * d..(t + 1) * d];
        let dst = &mut dw[i * d..(i + 1) * d];
        for j in 0..d {
            dst[j] += src[j];
        }
    }
    Ok(dw)
}

/// Masked CE. `logits` `[n, v]`, `targets`/`mask` length `n`. Returns `(loss, dlogits)`.
pub fn masked_cross_entropy(
    logits: &[f32],
    n: usize,
    v: usize,
    targets: &[u32],
    mask: &[u8],
) -> Result<(f32, Vec<f32>)> {
    let (loss, _h, dlogits) = masked_cross_entropy_entropy(logits, n, v, targets, mask, 0.0)?;
    Ok((loss, dlogits))
}

/// Masked CE plus language-agnostic Shannon entropy penalty on the softmax.
///
/// `L = CE + λ H[p]`, `H = −Σ_j p_j log p_j`,
/// `∂H/∂z_k = −p_k (log p_k + H)`.
/// High-entropy (blended) token states are penalized; one-hot assignments
/// are left untouched. Returns `(total_loss, mean_entropy, dlogits)`.
pub fn masked_cross_entropy_entropy(
    logits: &[f32],
    n: usize,
    v: usize,
    targets: &[u32],
    mask: &[u8],
    entropy_coef: f32,
) -> Result<(f32, f32, Vec<f32>)> {
    if logits.len() != n * v || targets.len() != n || mask.len() != n {
        anyhow::bail!("ce shape");
    }
    let mut dlogits = vec![0.0f32; logits.len()];
    let mut loss = 0.0f32;
    let mut h_sum = 0.0f32;
    let mut den = 0.0f32;
    let lam = entropy_coef.max(0.0);
    for i in 0..n {
        if mask[i] == 0 {
            continue;
        }
        den += 1.0;
        let row = &logits[i * v..(i + 1) * v];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.0f32;
        for &zlogit in row {
            z += (zlogit - m).exp();
        }
        let invz = 1.0 / z.max(1e-20);
        let tgt = (targets[i] as usize).min(v.saturating_sub(1));
        loss += -(row[tgt] - m) + z.ln();
        let dst = &mut dlogits[i * v..(i + 1) * v];
        let mut h = 0.0f32;
        for j in 0..v {
            let p = (row[j] - m).exp() * invz;
            dst[j] = p;
            if p > 0.0 {
                h -= p * p.max(1e-12).ln();
            }
        }
        h_sum += h;
        if lam > 0.0 {
            for j in 0..v {
                let p = dst[j].max(1e-12);
                dst[j] += lam * (-p * (p.ln() + h));
            }
        }
        dst[tgt] -= 1.0;
    }
    let inv_den = if den > 0.0 { 1.0 / den } else { 0.0 };
    loss *= inv_den;
    let mean_h = h_sum * inv_den;
    loss += lam * mean_h;
    for g in &mut dlogits {
        *g *= inv_den;
    }
    Ok((loss, mean_h, dlogits))
}
