//! Causal mixers and host RNG init (no Candle).

use anyhow::Result;

use crate::accelerate::{sgemm, sgemm_nt, sgemm_tn, softmax_rows};

/// Parameter-free token mix: delay half the channels by one step.
/// `x` is `[b, t, c]` row-major.
pub fn causal_shift(x: &[f32], b: usize, t: usize, c: usize) -> Result<Vec<f32>> {
    let mut y = vec![0.0f32; b.saturating_mul(t).saturating_mul(c)];
    causal_shift_into(x, b, t, c, &mut y)?;
    Ok(y)
}

/// Write `causal_shift` into `y`. Delayed channels at `t=0` are zeroed.
pub fn causal_shift_into(x: &[f32], b: usize, t: usize, c: usize, y: &mut [f32]) -> Result<()> {
    if x.len() != b * t * c {
        anyhow::bail!("causal_shift len {} != b*t*c {}", x.len(), b * t * c);
    }
    if y.len() < x.len() {
        anyhow::bail!("causal_shift y len {} < {}", y.len(), x.len());
    }
    let split = c / 2;
    let y = &mut y[..x.len()];
    for bi in 0..b {
        for ti in 0..t {
            let src = (bi * t + ti) * c;
            let dst = src;
            y[dst..dst + split].copy_from_slice(&x[src..src + split]);
            if ti == 0 {
                y[dst + split..dst + c].fill(0.0);
            } else {
                let prev = (bi * t + (ti - 1)) * c + split;
                y[dst + split..dst + c].copy_from_slice(&x[prev..prev + (c - split)]);
            }
        }
    }
    Ok(())
}

/// `dx` for `y = causal_shift(x)`. `dy` is `[b,t,c]`.
pub fn causal_shift_backward(dy: &[f32], b: usize, t: usize, c: usize) -> Result<Vec<f32>> {
    let mut dx = vec![0.0f32; b.saturating_mul(t).saturating_mul(c)];
    causal_shift_backward_into(dy, b, t, c, &mut dx)?;
    Ok(dx)
}

pub fn causal_shift_backward_into(
    dy: &[f32],
    b: usize,
    t: usize,
    c: usize,
    dx: &mut [f32],
) -> Result<()> {
    if dy.len() != b * t * c {
        anyhow::bail!("causal_shift_backward len mismatch");
    }
    if dx.len() < dy.len() {
        anyhow::bail!("causal_shift_backward dx short");
    }
    let split = c / 2;
    let dx = &mut dx[..dy.len()];
    dx.fill(0.0);
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
    Ok(())
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
        let mut y = vec![0.0f32; x.len()];
        self.forward_into(x, b, t, &mut y)?;
        Ok(y)
    }

    pub fn forward_into(&self, x: &[f32], b: usize, t: usize, y: &mut [f32]) -> Result<()> {
        let d = self.d_model;
        if x.len() != b * t * d {
            anyhow::bail!("attn x len");
        }
        if y.len() < x.len() {
            anyhow::bail!("attn y len");
        }
        let tmp = self.forward_alloc(x, b, t)?;
        y[..x.len()].copy_from_slice(&tmp);
        Ok(())
    }

    fn forward_alloc(&self, x: &[f32], b: usize, t: usize) -> Result<Vec<f32>> {
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
    let mut y = vec![0.0f32; n.saturating_mul(d)];
    rmsnorm_into(x, n, d, weight, eps, &mut y)?;
    Ok(y)
}

pub fn rmsnorm_into(
    x: &[f32],
    n: usize,
    d: usize,
    weight: &[f32],
    eps: f32,
    y: &mut [f32],
) -> Result<()> {
    if x.len() != n * d || weight.len() != d {
        anyhow::bail!("rmsnorm shape");
    }
    if y.len() < x.len() {
        anyhow::bail!("rmsnorm y len {} < {}", y.len(), x.len());
    }
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
    Ok(())
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
    let mut dx = vec![0.0f32; n.saturating_mul(d)];
    let mut dw = vec![0.0f32; d];
    rmsnorm_backward_into(x, dy, n, d, weight, eps, &mut dx, &mut dw)?;
    Ok((dx, dw))
}

pub fn rmsnorm_backward_into(
    x: &[f32],
    dy: &[f32],
    n: usize,
    d: usize,
    weight: &[f32],
    eps: f32,
    dx: &mut [f32],
    dw: &mut [f32],
) -> Result<()> {
    if x.len() != n * d || dy.len() != n * d || weight.len() != d {
        anyhow::bail!("rmsnorm_backward shape");
    }
    if dx.len() < n * d || dw.len() < d {
        anyhow::bail!("rmsnorm_backward out short");
    }
    dx[..n * d].fill(0.0);
    dw[..d].fill(0.0);
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
    Ok(())
}

pub fn embed_lookup(table: &[f32], vocab: usize, d: usize, ids: &[u32]) -> Result<Vec<f32>> {
    let mut y = vec![0.0f32; ids.len().saturating_mul(d)];
    embed_lookup_into(table, vocab, d, ids, &mut y)?;
    Ok(y)
}

pub fn embed_lookup_into(
    table: &[f32],
    vocab: usize,
    d: usize,
    ids: &[u32],
    y: &mut [f32],
) -> Result<()> {
    if table.len() != vocab * d {
        anyhow::bail!("embed table len");
    }
    if y.len() < ids.len() * d {
        anyhow::bail!("embed lookup y short");
    }
    for (t, &id) in ids.iter().enumerate() {
        let i = (id as usize).min(vocab.saturating_sub(1));
        y[t * d..(t + 1) * d].copy_from_slice(&table[i * d..(i + 1) * d]);
    }
    Ok(())
}

/// Streamed tied-head CE + entropy. Never materializes `[n, V]` logits, so
/// `V=8192` training stays inside the 40 MB envelope.
///
/// Returns `(loss, mean_entropy, dhidden[n,d], dembed[V,d])`.
/// Allocating `dembed` is for tests/oracle only — the train path uses
/// [`streamed_tied_ce_acc`] into the standing `embed_grad` buffer.
pub fn streamed_tied_ce(
    hidden: &[f32],
    embed: &[f32],
    n: usize,
    d: usize,
    v: usize,
    targets: &[u32],
    mask: &[u8],
    entropy_coef: f32,
) -> Result<(f32, f32, Vec<f32>, Vec<f32>)> {
    let mut dhidden = vec![0.0f32; n.saturating_mul(d)];
    let mut dembed = vec![0.0f32; v.saturating_mul(d)];
    let mut row = Vec::new();
    let (loss, mean_h) = streamed_tied_ce_acc(
        hidden,
        embed,
        n,
        d,
        v,
        targets,
        mask,
        entropy_coef,
        &mut dhidden,
        &mut dembed,
        &mut row,
    )?;
    Ok((loss, mean_h, dhidden, dembed))
}

/// Two-pass CE: count `den`, then accumulate `g/den` **directly** into
/// `dhidden[n,d]` (overwritten) and `embed_grad[V,d]` (added).
///
/// Does **not** allocate a `[V,d]` increment and does **not** scale any
/// historical values already in `embed_grad`.
pub fn streamed_tied_ce_acc(
    hidden: &[f32],
    embed: &[f32],
    n: usize,
    d: usize,
    v: usize,
    targets: &[u32],
    mask: &[u8],
    entropy_coef: f32,
    dhidden: &mut [f32],
    embed_grad: &mut [f32],
    logits_row: &mut Vec<f32>,
) -> Result<(f32, f32)> {
    if hidden.len() != n * d || embed.len() != v * d || targets.len() != n || mask.len() != n {
        anyhow::bail!("streamed tied-ce shape");
    }
    if dhidden.len() != n * d {
        anyhow::bail!("dhidden len {} != n*d {}", dhidden.len(), n * d);
    }
    if embed_grad.len() != v * d {
        anyhow::bail!("embed_grad len {} != v*d {}", embed_grad.len(), v * d);
    }
    dhidden.fill(0.0);
    let live: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
    if live.is_empty() {
        return Ok((0.0, 0.0));
    }
    let n_live = live.len();
    let inv_den = 1.0 / n_live as f32;
    let lam = entropy_coef.max(0.0);
    let mut h_live = vec![0.0f32; n_live.saturating_mul(d)];
    for (r, &i) in live.iter().enumerate() {
        h_live[r * d..(r + 1) * d].copy_from_slice(&hidden[i * d..(i + 1) * d]);
    }
    // Keep the logits tile ≲ 2 MB. Never materialize `[n, V]`.
    let max_floats = (2 * 1024 * 1024) / 4;
    let chunk = (max_floats / n_live.max(1)).clamp(128, 8192).min(v.max(1));
    let need = n_live.saturating_mul(chunk);
    if logits_row.len() < need {
        logits_row.resize(need, 0.0);
    }
    let mut m = vec![f32::NEG_INFINITY; n_live];
    let mut z = vec![0.0f32; n_live];
    let mut entropy_row = vec![0.0f32; n_live];
    let mut logit_y = vec![0.0f32; n_live];
    let mut dh_live = vec![0.0f32; n_live.saturating_mul(d)];
    let mut g = vec![0.0f32; need];

    let gemm_chunk = |v0: usize, loc: &mut [f32]| -> Result<usize> {
        let vc = chunk.min(v - v0);
        sgemm_nt(
            n_live,
            vc,
            d,
            1.0,
            &h_live,
            &embed[v0 * d..(v0 + vc) * d],
            0.0,
            &mut loc[..n_live * vc],
        )?;
        Ok(vc)
    };

    let mut v0 = 0usize;
    while v0 < v {
        let loc = &mut logits_row[..need];
        let vc = gemm_chunk(v0, loc)?;
        for r in 0..n_live {
            let i = live[r];
            let y = (targets[i] as usize).min(v.saturating_sub(1));
            let row = &loc[r * vc..(r + 1) * vc];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if mx > m[r] {
                m[r] = mx;
            }
            if y >= v0 && y < v0 + vc {
                logit_y[r] = row[y - v0];
            }
        }
        v0 += vc;
    }

    v0 = 0;
    while v0 < v {
        let loc = &mut logits_row[..need];
        let vc = gemm_chunk(v0, loc)?;
        for r in 0..n_live {
            let row = &loc[r * vc..(r + 1) * vc];
            let mr = m[r];
            let mut s = 0.0f32;
            for &x in row {
                s += (x - mr).exp();
            }
            z[r] += s;
        }
        v0 += vc;
    }

    if lam > 0.0 {
        v0 = 0;
        while v0 < v {
            let loc = &mut logits_row[..need];
            let vc = gemm_chunk(v0, loc)?;
            for r in 0..n_live {
                let row = &loc[r * vc..(r + 1) * vc];
                let inv_z = 1.0 / z[r].max(1e-20);
                let mr = m[r];
                let mut h = 0.0f32;
                for &x in row {
                    let p = ((x - mr).exp() * inv_z).max(1e-12);
                    h -= p * p.ln();
                }
                entropy_row[r] += h;
            }
            v0 += vc;
        }
    }

    let mut loss = 0.0f32;
    for r in 0..n_live {
        // Stable CE: −(z_y − m) + log Σ e^{z−m}. Never floors at −ln(1e-12)=27.63.
        loss += -(logit_y[r] - m[r]) + z[r].max(1e-20).ln();
    }
    v0 = 0;
    while v0 < v {
        let loc = &mut logits_row[..need];
        let vc = gemm_chunk(v0, loc)?;
        for r in 0..n_live {
            let i = live[r];
            let y = (targets[i] as usize).min(v.saturating_sub(1));
            let inv_z = 1.0 / z[r].max(1e-20);
            let mr = m[r];
            let ent = entropy_row[r];
            let grow = &mut g[r * vc..(r + 1) * vc];
            let row = &loc[r * vc..(r + 1) * vc];
            for t in 0..vc {
                let tok = v0 + t;
                let p = (row[t] - mr).exp() * inv_z;
                let mut gk = p;
                if tok == y {
                    gk -= 1.0;
                }
                if lam > 0.0 {
                    gk += lam * (-p.max(1e-12) * (p.max(1e-12).ln() + ent));
                }
                grow[t] = gk * inv_den;
            }
        }
        sgemm(
            n_live,
            d,
            vc,
            1.0,
            &g[..n_live * vc],
            &embed[v0 * d..(v0 + vc) * d],
            1.0,
            &mut dh_live,
        )?;
        sgemm_tn(
            vc,
            d,
            n_live,
            1.0,
            &g[..n_live * vc],
            &h_live,
            1.0,
            &mut embed_grad[v0 * d..(v0 + vc) * d],
        )?;
        v0 += vc;
    }

    for (r, &i) in live.iter().enumerate() {
        dhidden[i * d..(i + 1) * d].copy_from_slice(&dh_live[r * d..(r + 1) * d]);
    }
    let mut h_sum = 0.0f32;
    for e in &entropy_row {
        h_sum += *e;
    }
    loss *= inv_den;
    h_sum *= inv_den;
    loss += lam * h_sum;
    Ok((loss, h_sum))
}

pub fn embed_scatter(vocab: usize, d: usize, ids: &[u32], dy: &[f32]) -> Result<Vec<f32>> {
    let mut dw = vec![0.0f32; vocab * d];
    embed_scatter_acc(vocab, d, ids, dy, &mut dw)?;
    Ok(dw)
}

/// Add `dy[t]` into `embed_grad[id[t]]`. Does not allocate `[V,d]`.
pub fn embed_scatter_acc(
    vocab: usize,
    d: usize,
    ids: &[u32],
    dy: &[f32],
    embed_grad: &mut [f32],
) -> Result<()> {
    if vocab == 0 || d == 0 {
        anyhow::bail!("embed scatter empty vocab/d");
    }
    if dy.len() != ids.len() * d {
        anyhow::bail!(
            "embed scatter dy len {} != ids*d {}",
            dy.len(),
            ids.len() * d
        );
    }
    if embed_grad.len() != vocab * d {
        anyhow::bail!(
            "embed_grad len {} != vocab*d {}",
            embed_grad.len(),
            vocab * d
        );
    }
    for (t, &id) in ids.iter().enumerate() {
        let raw = id as usize;
        debug_assert!(raw < vocab, "embed scatter id {id} >= vocab {vocab}");
        let i = raw.min(vocab.saturating_sub(1));
        let src = &dy[t * d..(t + 1) * d];
        let dst = &mut embed_grad[i * d..i * d + d];
        for j in 0..d {
            dst[j] += src[j];
        }
    }
    Ok(())
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
