//! 1-bit ROSA: QKV SAM (`rosa_qkv_ref`) and single-stream `rosa()`.
//!
//! The QKV automaton is a line-by-line transcription of
//! `260212_rosa1bitLM_L12.py::rosa_qkv_ref`. Missing children are `-1`; node 0
//! is the root. `rosa()` is the distinct single-stream routine from
//! `251014_rosa_1bit_layer.py` and is not interchangeable with QKV.

use anyhow::{Result, bail};

/// Finite-difference temperature from `ROSA_1bit` (`τ = 1e-3`).
pub const BITFLIP_TAU: f32 = 1e-3;

/// Online suffix automaton over a binary alphabet, matching `rosa_qkv_ref`.
#[derive(Clone, Debug)]
pub struct RosaSam {
    trans0: Vec<i32>,
    trans1: Vec<i32>,
    fail: Vec<i32>,
    maxlen: Vec<i32>,
    last: Vec<i32>,
    v_hist: Vec<u8>,
    u: i32,
    g: i32,
    w: i32,
    h: i32,
    i: i32,
}

impl RosaSam {
    /// Allocates `2 * max_time + 1` nodes, the same bound as Python `s=2*n+1`.
    pub fn with_max_time(max_time: usize) -> Self {
        let nodes = sam_node_count(max_time);
        Self {
            trans0: vec![-1; nodes],
            trans1: vec![-1; nodes],
            fail: vec![-1; nodes],
            maxlen: vec![0; nodes],
            last: vec![-1; nodes],
            v_hist: Vec::with_capacity(max_time),
            u: 1,
            g: 0,
            w: 0,
            h: 0,
            i: 0,
        }
    }

    pub fn child(&self, p: i32, bit: u8) -> i32 {
        if p < 0 {
            return -1;
        }
        let index = p as usize;
        if bit == 0 {
            self.trans0[index]
        } else {
            self.trans1[index]
        }
    }

    pub fn set_child(&mut self, p: i32, bit: u8, to: i32) {
        if p < 0 {
            return;
        }
        let index = p as usize;
        if bit == 0 {
            self.trans0[index] = to;
        } else {
            self.trans1[index] = to;
        }
    }

    fn grow_to(&mut self, len: usize) {
        if len <= self.trans0.len() {
            return;
        }
        self.trans0.resize(len, -1);
        self.trans1.resize(len, -1);
        self.fail.resize(len, -1);
        self.maxlen.resize(len, 0);
        self.last.resize(len, -1);
    }

    /// One step of `rosa_qkv_ref`. Returns collapsed idx ∈ {0, 1}.
    pub fn push(&mut self, q_bit: u8, k_bit: u8, v_bit: u8) -> u8 {
        debug_assert!(q_bit <= 1 && k_bit <= 1 && v_bit <= 1);
        self.v_hist.push(v_bit);
        let i = self.i;
        let mut p = self.w;
        let mut x = self.h;
        while p != -1 && self.child(p, q_bit) < 0 {
            let mp = self.maxlen[p as usize];
            if x > mp {
                x = mp;
            }
            p = self.fail[p as usize];
        }
        if p == -1 {
            p = 0;
            x = 0;
        } else {
            p = self.child(p, q_bit);
            x += 1;
        }
        let mut v = p;
        while self.fail[v as usize] != -1 && self.maxlen[self.fail[v as usize] as usize] >= x {
            v = self.fail[v as usize];
        }
        while v != -1 && (self.maxlen[v as usize] <= 0 || self.last[v as usize] < 0) {
            v = self.fail[v as usize];
        }
        let y = if v == -1 {
            -1
        } else {
            let pos = (self.last[v as usize] + 1) as usize;
            i32::from(self.v_hist[pos])
        };
        let idx = y.max(0) as u8;
        self.w = p;
        self.h = x;

        let j = self.u;
        self.u += 1;
        self.grow_to(self.u as usize + 2);
        self.maxlen[j as usize] = self.maxlen[self.g as usize] + 1;
        p = self.g;
        while p != -1 && self.child(p, k_bit) < 0 {
            self.set_child(p, k_bit, j);
            p = self.fail[p as usize];
        }
        if p == -1 {
            self.fail[j as usize] = 0;
        } else {
            let d = self.child(p, k_bit);
            if self.maxlen[p as usize] + 1 == self.maxlen[d as usize] {
                self.fail[j as usize] = d;
            } else {
                let b = self.u;
                self.u += 1;
                self.grow_to(self.u as usize);
                self.trans0[b as usize] = self.trans0[d as usize];
                self.trans1[b as usize] = self.trans1[d as usize];
                self.maxlen[b as usize] = self.maxlen[p as usize] + 1;
                self.fail[b as usize] = self.fail[d as usize];
                self.last[b as usize] = self.last[d as usize];
                self.fail[d as usize] = b;
                self.fail[j as usize] = b;
                while p != -1 && self.child(p, k_bit) == d {
                    self.set_child(p, k_bit, b);
                    p = self.fail[p as usize];
                }
            }
        }
        v = j;
        self.g = j;
        while v != -1 && self.last[v as usize] < i {
            self.last[v as usize] = i;
            v = self.fail[v as usize];
        }
        self.i += 1;
        idx
    }
}

/// Full-sequence QKV idx, identical to `RosaSam::push` over `T` steps.
pub fn rosa_qkv_ref(q: &[u8], k: &[u8], v: &[u8]) -> Result<Vec<u8>> {
    if q.len() != k.len() || q.len() != v.len() {
        bail!("rosa_qkv_ref requires q, k, v of equal length");
    }
    if q.iter().chain(k).chain(v).any(|&bit| bit > 1) {
        bail!("rosa_qkv_ref bits must be 0 or 1");
    }
    let mut sam = RosaSam::with_max_time(q.len());
    Ok(q.iter()
        .zip(k)
        .zip(v)
        .map(|((&q_bit, &k_bit), &v_bit)| sam.push(q_bit, k_bit, v_bit))
        .collect())
}

/// Float ROSA-QKV output: `out = (2 * idx - 1) * e`. idx 0 → −e, idx 1 → +e.
pub fn rosa_qkv_out(idx: &[u8], e: f32) -> Vec<f32> {
    idx.iter()
        .map(|&bit| (2.0 * f32::from(bit) - 1.0) * e)
        .collect()
}

/// Single-stream `rosa()` from `251014_rosa_1bit_layer.py`. Returns raw y,
/// where unmatched is `-1` (not collapsed).
pub fn rosa(bits: &[u8]) -> Result<Vec<i32>> {
    if bits.iter().any(|&bit| bit > 1) {
        bail!("rosa bits must be 0 or 1");
    }
    let n = bits.len();
    let nodes = n.saturating_mul(2).saturating_add(1).max(1);
    let mut trans0 = vec![-1i32; nodes];
    let mut trans1 = vec![-1i32; nodes];
    let mut fail = vec![-1i32; nodes];
    let mut maxlen = vec![0i32; nodes];
    let mut last = vec![-1i32; nodes];
    let child = |trans0: &[i32], trans1: &[i32], p: i32, bit: u8| -> i32 {
        if p < 0 {
            -1
        } else if bit == 0 {
            trans0[p as usize]
        } else {
            trans1[p as usize]
        }
    };
    let mut g = 0i32;
    let mut z = 1i32;
    let mut y = vec![-1i32; n];
    for (i, &token) in bits.iter().enumerate() {
        let r = z;
        z += 1;
        maxlen[r as usize] = maxlen[g as usize] + 1;
        let mut p = g;
        while p != -1 && child(&trans0, &trans1, p, token) < 0 {
            if token == 0 {
                trans0[p as usize] = r;
            } else {
                trans1[p as usize] = r;
            }
            p = fail[p as usize];
        }
        if p == -1 {
            fail[r as usize] = 0;
        } else {
            let q = child(&trans0, &trans1, p, token);
            if maxlen[p as usize] + 1 == maxlen[q as usize] {
                fail[r as usize] = q;
            } else {
                let u = z;
                z += 1;
                trans0[u as usize] = trans0[q as usize];
                trans1[u as usize] = trans1[q as usize];
                maxlen[u as usize] = maxlen[p as usize] + 1;
                fail[u as usize] = fail[q as usize];
                last[u as usize] = last[q as usize];
                while p != -1 && child(&trans0, &trans1, p, token) == q {
                    if token == 0 {
                        trans0[p as usize] = u;
                    } else {
                        trans1[p as usize] = u;
                    }
                    p = fail[p as usize];
                }
                fail[q as usize] = u;
                fail[r as usize] = u;
            }
        }
        g = r;
        let mut v = g;
        let mut a = -1i32;
        while v != -1 {
            if maxlen[v as usize] > 0 && last[v as usize] >= 0 {
                a = i32::from(bits[(last[v as usize] + 1) as usize]);
                break;
            }
            v = fail[v as usize];
        }
        y[i] = a;
        v = g;
        let time = i as i32;
        while v != -1 && last[v as usize] < time {
            last[v as usize] = time;
            v = fail[v as usize];
        }
    }
    Ok(y)
}

fn phi(idx: &[u8], gy: &[f32], e: f32) -> f32 {
    idx.iter()
        .zip(gy)
        .map(|(&bit, &g)| g * (2.0 * f32::from(bit) - 1.0) * e)
        .sum()
}

fn flip_stream(bits: &[u8], t: usize, value: u8) -> Vec<u8> {
    let mut copy = bits.to_vec();
    copy[t] = value;
    copy
}

/// CPU `exact_bitflip` for Q, K, V on a single channel. `T` must be `<= 32`.
pub fn exact_bitflip_qkv(
    q_bits: &[u8],
    k_bits: &[u8],
    v_bits: &[u8],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gy: &[f32],
    e: f32,
    tau: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let t = q_bits.len();
    if t == 0 || t > 32 {
        bail!("exact_bitflip requires 1 <= T <= 32");
    }
    if k_bits.len() != t
        || v_bits.len() != t
        || q.len() != t
        || k.len() != t
        || v.len() != t
        || gy.len() != t
    {
        bail!("exact_bitflip shape mismatch");
    }
    if !tau.is_finite() || tau <= 0.0 {
        bail!("exact_bitflip tau must be finite and positive");
    }
    let mut gq = vec![0.0; t];
    let mut gk = vec![0.0; t];
    let mut gv = vec![0.0; t];
    for time in 0..t {
        let mag_q = q[time].abs().max(tau);
        let mag_k = k[time].abs().max(tau);
        let mag_v = v[time].abs().max(tau);
        let phi1_q = phi(
            &rosa_qkv_ref(&flip_stream(q_bits, time, 1), k_bits, v_bits)?,
            gy,
            e,
        );
        let phi0_q = phi(
            &rosa_qkv_ref(&flip_stream(q_bits, time, 0), k_bits, v_bits)?,
            gy,
            e,
        );
        let phi1_k = phi(
            &rosa_qkv_ref(q_bits, &flip_stream(k_bits, time, 1), v_bits)?,
            gy,
            e,
        );
        let phi0_k = phi(
            &rosa_qkv_ref(q_bits, &flip_stream(k_bits, time, 0), v_bits)?,
            gy,
            e,
        );
        let phi1_v = phi(
            &rosa_qkv_ref(q_bits, k_bits, &flip_stream(v_bits, time, 1))?,
            gy,
            e,
        );
        let phi0_v = phi(
            &rosa_qkv_ref(q_bits, k_bits, &flip_stream(v_bits, time, 0))?,
            gy,
            e,
        );
        gq[time] = (phi1_q - phi0_q) / (2.0 * mag_q);
        gk[time] = (phi1_k - phi0_k) / (2.0 * mag_k);
        gv[time] = (phi1_v - phi0_v) / (2.0 * mag_v);
    }
    Ok((gq, gk, gv))
}

pub fn bit_from_activation(value: f32) -> u8 {
    u8::from(value > 0.0)
}

/// Node budget of `rosa_qkv_ref`: `s = 2 * T + 1`.
pub fn sam_node_count(time: usize) -> usize {
    time.saturating_mul(2).saturating_add(1).max(1)
}

/// Five i32 SAM arrays (`trans0/1`, `fail`, `maxlen`, `last`) for one layer.
pub fn sam_workspace_bytes(batch: usize, time: usize, channels: usize) -> Option<usize> {
    batch
        .checked_mul(channels)?
        .checked_mul(sam_node_count(time))?
        .checked_mul(5)?
        .checked_mul(size_of::<i32>())
}

/// Packed Q/K/V bitplanes: `3 * B * T * D / 8` when the product is a multiple of 8.
pub fn qkv_bitplane_bytes(batch: usize, time: usize, channels: usize) -> Option<usize> {
    batch
        .checked_mul(time)?
        .checked_mul(channels)?
        .checked_mul(3)?
        .checked_div(8)
}

/// Packs 0/1 bits into 32-bit words, LSB = lowest index (same as `ullis_sign_pack_bits`).
pub fn pack_bitplane(bits: &[u8]) -> Result<Vec<u32>> {
    if bits.iter().any(|&bit| bit > 1) {
        bail!("pack_bitplane bits must be 0 or 1");
    }
    let mut words = vec![0_u32; bits.len().div_ceil(32)];
    for (index, &bit) in bits.iter().enumerate() {
        if bit != 0 {
            words[index / 32] |= 1 << (index % 32);
        }
    }
    Ok(words)
}

/// Batched `rosa_qkv_ref` over `[B, T, D]` bit tensors stored row-major.
pub fn rosa_qkv_batch(
    q: &[u8],
    k: &[u8],
    v: &[u8],
    batch: usize,
    time: usize,
    channels: usize,
) -> Result<Vec<u8>> {
    let elements = batch
        .checked_mul(time)
        .and_then(|rows| rows.checked_mul(channels))
        .ok_or_else(|| anyhow::anyhow!("rosa_qkv_batch shape overflow"))?;
    if q.len() != elements || k.len() != elements || v.len() != elements {
        bail!("rosa_qkv_batch requires q, k, v of length batch*time*channels");
    }
    if batch == 0 || time == 0 || channels == 0 {
        bail!("rosa_qkv_batch dimensions must be non-zero");
    }
    let mut idx = vec![0_u8; elements];
    for b in 0..batch {
        for c in 0..channels {
            let mut q_ch = Vec::with_capacity(time);
            let mut k_ch = Vec::with_capacity(time);
            let mut v_ch = Vec::with_capacity(time);
            for t in 0..time {
                let index = (b * time + t) * channels + c;
                q_ch.push(q[index]);
                k_ch.push(k[index]);
                v_ch.push(v[index]);
            }
            let channel_idx = rosa_qkv_ref(&q_ch, &k_ch, &v_ch)?;
            for t in 0..time {
                idx[(b * time + t) * channels + c] = channel_idx[t];
            }
        }
    }
    Ok(idx)
}

/// `out[b,t,c] = (2 * idx[b,t,c] - 1) * e[c]`.
pub fn rosa_qkv_out_batched(idx: &[u8], e: &[f32], batch: usize, time: usize, channels: usize) -> Result<Vec<f32>> {
    let elements = batch
        .checked_mul(time)
        .and_then(|rows| rows.checked_mul(channels))
        .ok_or_else(|| anyhow::anyhow!("rosa_qkv_out_batched shape overflow"))?;
    if idx.len() != elements || e.len() != channels {
        bail!("rosa_qkv_out_batched shape mismatch");
    }
    let mut out = vec![0.0; elements];
    for b in 0..batch {
        for t in 0..time {
            for c in 0..channels {
                let index = (b * time + t) * channels + c;
                out[index] = (2.0 * f32::from(idx[index]) - 1.0) * e[c];
            }
        }
    }
    Ok(out)
}
