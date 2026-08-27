//! CPU FP32 transcription of `RWKV-v8/cuda/wkv7_cuda.cu`.
//!
//! Layout matches the CUDA kernels: `w,q,k,v,a,b,y,sa` are `[B,T,H,N]` and
//! the chunk tape `s` is `[B,H,T/CHUNK,N,N]` with `s[b,h,chunk,j,i] = state_i[j]`.
//! `N = HEAD_SIZE = 16`, `CHUNK_LEN = 16`. Inputs `a,b` are the CUDA `z,a`
//! pair (`RUN_CUDA_RWKV7g(r,w,k,v,-kk,kk*a)`).

use anyhow::{Result, bail};

pub const HEAD_SIZE: usize = 16;
pub const CHUNK_LEN: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub struct Wkv7Forward {
    pub y: Vec<f32>,
    pub s: Vec<f32>,
    pub sa: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Wkv7Backward {
    pub dw: Vec<f32>,
    pub dq: Vec<f32>,
    pub dk: Vec<f32>,
    pub dv: Vec<f32>,
    pub da: Vec<f32>,
    pub db: Vec<f32>,
}

pub(crate) fn require_shape(
    w: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    batch: usize,
    time: usize,
    heads: usize,
) -> Result<usize> {
    if batch == 0 || time == 0 || heads == 0 {
        bail!("WKV7 dimensions must be non-zero");
    }
    if !time.is_multiple_of(CHUNK_LEN) {
        bail!("WKV7 time must be a multiple of {CHUNK_LEN}");
    }
    let len = batch
        .checked_mul(time)
        .and_then(|bt| bt.checked_mul(heads))
        .and_then(|bth| bth.checked_mul(HEAD_SIZE))
        .ok_or_else(|| anyhow::anyhow!("WKV7 shape overflow"))?;
    if [w, q, k, v, a, b].iter().any(|t| t.len() != len) {
        bail!("WKV7 tensor length mismatch");
    }
    Ok(len)
}

fn bt_index(batch: usize, time: usize, heads: usize, t: usize, h: usize, i: usize) -> usize {
    ((batch * time + t) * heads + h) * HEAD_SIZE + i
}

fn s_index(
    batch: usize,
    heads: usize,
    nchunks: usize,
    h: usize,
    chunk: usize,
    j: usize,
    i: usize,
) -> usize {
    ((((batch * heads + h) * nchunks + chunk) * HEAD_SIZE) + j) * HEAD_SIZE + i
}

/// CUDA `forward_kernel`, one thread per head channel.
pub fn wkv7_forward(
    w: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    batch: usize,
    time: usize,
    heads: usize,
) -> Result<Wkv7Forward> {
    let len = require_shape(w, q, k, v, a, b, batch, time, heads)?;
    let nchunks = time / CHUNK_LEN;
    let s_len = batch
        .saturating_mul(heads)
        .saturating_mul(nchunks)
        .saturating_mul(HEAD_SIZE)
        .saturating_mul(HEAD_SIZE);
    let mut y = vec![0.0; len];
    let mut sa_out = vec![0.0; len];
    let mut s = vec![0.0; s_len];
    let mut qh = [0.0; HEAD_SIZE];
    let mut wh = [0.0; HEAD_SIZE];
    let mut kh = [0.0; HEAD_SIZE];
    let mut ah = [0.0; HEAD_SIZE];
    let mut bh = [0.0; HEAD_SIZE];
    for bb in 0..batch {
        for hh in 0..heads {
            let mut state = [[0.0; HEAD_SIZE]; HEAD_SIZE];
            for t in 0..time {
                for i in 0..HEAD_SIZE {
                    let ind = bt_index(bb, time, heads, t, hh, i);
                    qh[i] = q[ind];
                    wh[i] = (-w[ind].exp()).exp();
                    kh[i] = k[ind];
                    ah[i] = a[ind];
                    bh[i] = b[ind];
                }
                for i in 0..HEAD_SIZE {
                    let ind = bt_index(bb, time, heads, t, hh, i);
                    let mut sa = 0.0;
                    for j in 0..HEAD_SIZE {
                        sa += ah[j] * state[i][j];
                    }
                    sa_out[ind] = sa;
                    let vv = v[ind];
                    let mut yy = 0.0;
                    for j in 0..HEAD_SIZE {
                        state[i][j] = state[i][j] * wh[j] + sa * bh[j] + kh[j] * vv;
                        yy += state[i][j] * qh[j];
                    }
                    y[ind] = yy;
                }
                if (t + 1).is_multiple_of(CHUNK_LEN) {
                    let chunk = t / CHUNK_LEN;
                    for i in 0..HEAD_SIZE {
                        for j in 0..HEAD_SIZE {
                            s[s_index(bb, heads, nchunks, hh, chunk, j, i)] = state[i][j];
                        }
                    }
                }
            }
        }
    }
    Ok(Wkv7Forward { y, s, sa: sa_out })
}

pub(crate) fn require_backward_shape(
    w: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    dy: &[f32],
    s: &[f32],
    sa: &[f32],
    batch: usize,
    time: usize,
    heads: usize,
) -> Result<(usize, usize)> {
    let len = require_shape(w, q, k, v, a, b, batch, time, heads)?;
    if dy.len() != len || sa.len() != len {
        bail!("WKV7 backward dy/sa length mismatch");
    }
    let nchunks = time / CHUNK_LEN;
    let s_len = batch
        .saturating_mul(heads)
        .saturating_mul(nchunks)
        .saturating_mul(HEAD_SIZE)
        .saturating_mul(HEAD_SIZE);
    if s.len() != s_len {
        bail!("WKV7 backward state tape length mismatch");
    }
    Ok((len, s_len))
}

/// CUDA `backward_kernel`. `s`/`sa` must come from the matching forward.
pub fn wkv7_backward(
    w: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    dy: &[f32],
    s: &[f32],
    sa: &[f32],
    batch: usize,
    time: usize,
    heads: usize,
) -> Result<Wkv7Backward> {
    let (len, _s_len) = require_backward_shape(w, q, k, v, a, b, dy, s, sa, batch, time, heads)?;
    let nchunks = time / CHUNK_LEN;
    let mut dw = vec![0.0; len];
    let mut dq = vec![0.0; len];
    let mut dk = vec![0.0; len];
    let mut dv = vec![0.0; len];
    let mut da = vec![0.0; len];
    let mut db = vec![0.0; len];
    let mut qh = [0.0; HEAD_SIZE];
    let mut wh = [0.0; HEAD_SIZE];
    let mut kh = [0.0; HEAD_SIZE];
    let mut vh = [0.0; HEAD_SIZE];
    let mut ah = [0.0; HEAD_SIZE];
    let mut bh = [0.0; HEAD_SIZE];
    let mut dyh = [0.0; HEAD_SIZE];
    let mut sah = [0.0; HEAD_SIZE];
    for bb in 0..batch {
        for hh in 0..heads {
            let mut state_t = [[0.0; HEAD_SIZE]; HEAD_SIZE];
            let mut dstate = [[0.0; HEAD_SIZE]; HEAD_SIZE];
            let mut dstate_t = [[0.0; HEAD_SIZE]; HEAD_SIZE];
            for t in (0..time).rev() {
                let mut wi = [0.0; HEAD_SIZE];
                let mut wi_fac = [0.0; HEAD_SIZE];
                for i in 0..HEAD_SIZE {
                    let ind = bt_index(bb, time, heads, t, hh, i);
                    qh[i] = q[ind];
                    wi_fac[i] = -w[ind].exp();
                    wi[i] = wi_fac[i].exp();
                    wh[i] = wi[i];
                    kh[i] = k[ind];
                    ah[i] = a[ind];
                    bh[i] = b[ind];
                    vh[i] = v[ind];
                    dyh[i] = dy[ind];
                    sah[i] = sa[ind];
                }
                if (t + 1).is_multiple_of(CHUNK_LEN) {
                    let chunk = t / CHUNK_LEN;
                    for i in 0..HEAD_SIZE {
                        for j in 0..HEAD_SIZE {
                            // CUDA loads the transpose of the forward store.
                            state_t[i][j] = s[s_index(bb, heads, nchunks, hh, chunk, i, j)];
                        }
                    }
                }
                let mut dsb = [0.0; HEAD_SIZE];
                for i in 0..HEAD_SIZE {
                    let ind = bt_index(bb, time, heads, t, hh, i);
                    let mut dqi = 0.0;
                    for j in 0..HEAD_SIZE {
                        dqi += state_t[i][j] * dyh[j];
                    }
                    dq[ind] = dqi;
                    let iwi = 1.0 / wi[i];
                    for j in 0..HEAD_SIZE {
                        state_t[i][j] = (state_t[i][j] - kh[i] * vh[j] - bh[i] * sah[j]) * iwi;
                        dstate[i][j] += dyh[i] * qh[j];
                        dstate_t[i][j] += qh[i] * dyh[j];
                    }
                    let mut dwi = 0.0;
                    let mut dki = 0.0;
                    let mut dvi = 0.0;
                    let mut dbi = 0.0;
                    let mut dsbi = 0.0;
                    for j in 0..HEAD_SIZE {
                        dwi += dstate_t[i][j] * state_t[i][j];
                        dki += dstate_t[i][j] * vh[j];
                        dvi += dstate[i][j] * kh[j];
                        dsbi += dstate[i][j] * bh[j];
                        dbi += dstate_t[i][j] * sah[j];
                    }
                    dw[ind] = dwi * wi[i] * wi_fac[i];
                    dk[ind] = dki;
                    dv[ind] = dvi;
                    db[ind] = dbi;
                    dsb[i] = dsbi;
                }
                for i in 0..HEAD_SIZE {
                    let ind = bt_index(bb, time, heads, t, hh, i);
                    let mut dai = 0.0;
                    for j in 0..HEAD_SIZE {
                        dai += state_t[i][j] * dsb[j];
                    }
                    da[ind] = dai;
                    for j in 0..HEAD_SIZE {
                        dstate[i][j] = dstate[i][j] * wh[j] + dsb[i] * ah[j];
                        dstate_t[i][j] = dstate_t[i][j] * wi[i] + ah[i] * dsb[j];
                    }
                }
            }
        }
    }
    Ok(Wkv7Backward {
        dw,
        dq,
        dk,
        dv,
        da,
        db,
    })
}

/// Recurrent one-token update. `state` is `[H, i, j]` = thread `i` `state[j]`.
pub fn wkv7_step(
    w: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
    state: &mut [f32],
    heads: usize,
) -> Result<Vec<f32>> {
    if heads == 0 {
        bail!("WKV7 step needs at least one head");
    }
    let len = heads.saturating_mul(HEAD_SIZE);
    if [w, q, k, v, a, b].iter().any(|t| t.len() != len) {
        bail!("WKV7 step tensor length mismatch");
    }
    if state.len() != heads.saturating_mul(HEAD_SIZE).saturating_mul(HEAD_SIZE) {
        bail!("WKV7 step state length mismatch");
    }
    let mut y = vec![0.0; len];
    let mut qh = [0.0; HEAD_SIZE];
    let mut wh = [0.0; HEAD_SIZE];
    let mut kh = [0.0; HEAD_SIZE];
    let mut ah = [0.0; HEAD_SIZE];
    let mut bh = [0.0; HEAD_SIZE];
    for hh in 0..heads {
        let base = hh * HEAD_SIZE;
        for i in 0..HEAD_SIZE {
            qh[i] = q[base + i];
            wh[i] = (-w[base + i].exp()).exp();
            kh[i] = k[base + i];
            ah[i] = a[base + i];
            bh[i] = b[base + i];
        }
        for i in 0..HEAD_SIZE {
            let sbase = (hh * HEAD_SIZE + i) * HEAD_SIZE;
            let mut sa = 0.0;
            for j in 0..HEAD_SIZE {
                sa += ah[j] * state[sbase + j];
            }
            let vv = v[base + i];
            let mut yy = 0.0;
            for j in 0..HEAD_SIZE {
                let s = state[sbase + j] * wh[j] + sa * bh[j] + kh[j] * vv;
                state[sbase + j] = s;
                yy += s * qh[j];
            }
            y[base + i] = yy;
        }
    }
    Ok(y)
}

/// Deterministic T=16 H=1 chunk used by `tests/wkv7.rs`.
pub fn fixture_t16_h1() -> ([Vec<f32>; 6], usize, usize, usize) {
    let batch = 1;
    let time = CHUNK_LEN;
    let heads = 1;
    let n = time * HEAD_SIZE;
    let mut w = vec![0.0; n];
    let mut q = vec![0.0; n];
    let mut k = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    for i in 0..n {
        let x = i as f32;
        w[i] = 0.10 * (x * 0.17).sin();
        q[i] = 0.20 * (x * 0.13).cos();
        k[i] = 0.15 * (x * 0.11 + 0.4).sin();
        v[i] = 0.25 * (x * 0.07 + 0.2).cos();
        a[i] = 0.05 * (x * 0.19 + 1.1).sin();
        b[i] = 0.08 * (x * 0.23 + 0.6).cos();
    }
    ([w, q, k, v, a, b], batch, time, heads)
}
