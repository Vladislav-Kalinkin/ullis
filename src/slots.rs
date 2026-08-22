//! Content-based key/value slot memory. Control plane, FP32, `[B, T, D]`.
//!
//! Sequence mix is causal in `T`, so tokens cannot be independent GPU
//! threads. Each token is a handful of `S×D` maps: `cblas_sgemv` launch
//! loses to a fused host loop at every size in the design envelope, and
//! the GPU has nothing to fill. Slots stay on the host.
//!
//! The GRU write `M ← M + β k (v − kᵀM)` is inverted from the final
//! tables: tape the `kᵀM` addresses (Θ(BTD)), undo in reverse. No
//! rematerialize, no `M_t` tape, algebra unchanged.
//!
//! Read: cosine content on `M_key`, `out = γ sᵀ M_val`.
//! Write: content scores plus temporal link, then the GRU update.
//! Keys and values are separate tables so a value write does not erase
//! the name.

use anyhow::{bail, Result};

use crate::scan::sigmoid;

const ALLOC_TEMP: f32 = 8.0;
/// Cosine temperature for content scores. `softmax(M q / √D)` is dead at
/// language embed scale (`N(0, 0.02²)`, `||v||² ≈ 0.2`): matching-slot
/// logits stay `≪ log S`. Cosine is scale-free; 8 nats peaks a match
/// against hundreds of empty slots.
const CONTENT_TEMP: f32 = 8.0;

#[derive(Clone, Debug)]
pub struct SlotParams {
    pub d: usize,
    pub n_slots: usize,
    pub w_q: Vec<f32>,
    pub w_w: Vec<f32>,
    pub w_link: Vec<f32>,
    pub b_link: f32,
    pub w_bk: f32,
    pub b_bk: f32,
    pub w_bv: f32,
    pub b_bv: f32,
    pub gamma: f32,
    pub b_alloc: f32,
    pub grad_w_q: Vec<f32>,
    pub grad_w_w: Vec<f32>,
    pub grad_w_link: Vec<f32>,
    pub grad_b_link: f32,
    pub grad_w_bk: f32,
    pub grad_b_bk: f32,
    pub grad_w_bv: f32,
    pub grad_b_bv: f32,
    pub grad_gamma: f32,
    pub grad_b_alloc: f32,
}

impl SlotParams {
    pub fn new(d: usize, n_slots: usize) -> Self {
        Self {
            d,
            n_slots,
            w_q: vec![1.0; d],
            w_w: vec![1.0; d],
            w_link: vec![0.0; d],
            b_link: 0.0,
            w_bk: 0.0,
            b_bk: 1.0,
            w_bv: 0.0,
            b_bv: 1.0,
            gamma: 1.0,
            b_alloc: 3.0,
            grad_w_q: vec![0.0; d],
            grad_w_w: vec![0.0; d],
            grad_w_link: vec![0.0; d],
            grad_b_link: 0.0,
            grad_w_bk: 0.0,
            grad_b_bk: 0.0,
            grad_w_bv: 0.0,
            grad_b_bv: 0.0,
            grad_gamma: 0.0,
            grad_b_alloc: 0.0,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad_w_q.fill(0.0);
        self.grad_w_w.fill(0.0);
        self.grad_w_link.fill(0.0);
        self.grad_b_link = 0.0;
        self.grad_w_bk = 0.0;
        self.grad_b_bk = 0.0;
        self.grad_w_bv = 0.0;
        self.grad_b_bv = 0.0;
        self.grad_gamma = 0.0;
        self.grad_b_alloc = 0.0;
    }

    fn add_grads(&mut self, other: &Self) {
        for (a, b) in self.grad_w_q.iter_mut().zip(&other.grad_w_q) {
            *a += *b;
        }
        for (a, b) in self.grad_w_w.iter_mut().zip(&other.grad_w_w) {
            *a += *b;
        }
        for (a, b) in self.grad_w_link.iter_mut().zip(&other.grad_w_link) {
            *a += *b;
        }
        self.grad_b_link += other.grad_b_link;
        self.grad_w_bk += other.grad_w_bk;
        self.grad_b_bk += other.grad_b_bk;
        self.grad_w_bv += other.grad_w_bv;
        self.grad_b_bv += other.grad_b_bv;
        self.grad_gamma += other.grad_gamma;
        self.grad_b_alloc += other.grad_b_alloc;
    }
}

#[derive(Clone, Debug)]
pub struct SlotState {
    pub key: Vec<f32>,
    pub val: Vec<f32>,
    pub usage: Vec<f32>,
    pub p_prev: Vec<f32>,
    batch: usize,
    n_slots: usize,
    d: usize,
}

impl SlotState {
    pub fn new(batch: usize, n_slots: usize, d: usize) -> Self {
        let mut st = Self {
            key: vec![0.0; batch * n_slots * d],
            val: vec![0.0; batch * n_slots * d],
            usage: vec![0.0; batch * n_slots],
            p_prev: vec![0.0; batch * n_slots],
            batch,
            n_slots,
            d,
        };
        st.reset();
        st
    }

    pub fn reset(&mut self) {
        self.val.fill(0.0);
        self.usage.fill(0.0);
        self.p_prev.fill(0.0);
        self.key.fill(0.0);
    }
}

const STAPE_F: usize = 5;
const F_S: usize = 0;
const F_KC: usize = 1;
const F_ALLOC: usize = 2;
const F_KW: usize = 3;
const F_PP: usize = 4;
const SCAL_N: usize = 8;

struct StepView<'a> {
    v: &'a [f32],
    s: &'a [f32],
    k_c: &'a [f32],
    k_alloc: &'a [f32],
    k_w: &'a [f32],
    p_prev: &'a [f32],
    addr_k: &'a [f32],
    addr_v: &'a [f32],
    link: f32,
    g_alloc: f32,
    beta_k: f32,
    beta_v: f32,
    mean_v: f32,
    sig_k: f32,
    sig_v: f32,
}

#[derive(Clone, Debug, Default)]
pub struct SlotTape {
    b: usize,
    t: usize,
    s: usize,
    d: usize,
    v: Vec<f32>,
    stape: Vec<f32>,
    scal: Vec<f32>,
    /// `[B, T, 2, D]` — `kᵀ M_key` / `kᵀ M_val` before each write.
    addr: Vec<f32>,
    final_key: Vec<f32>,
    final_val: Vec<f32>,
}

impl SlotTape {
    fn alloc(b: usize, t: usize, s: usize, d: usize) -> Self {
        let bt = b.saturating_mul(t);
        let sd = s.saturating_mul(d);
        Self {
            b,
            t,
            s,
            d,
            v: vec![0.0; bt.saturating_mul(d)],
            stape: vec![0.0; bt.saturating_mul(STAPE_F).saturating_mul(s)],
            scal: vec![0.0; bt.saturating_mul(SCAL_N)],
            addr: vec![0.0; bt.saturating_mul(2).saturating_mul(d)],
            final_key: vec![0.0; b.saturating_mul(sd)],
            final_val: vec![0.0; b.saturating_mul(sd)],
        }
    }

    fn st_off(&self, bi: usize, ti: usize, field: usize) -> usize {
        ((bi * self.t + ti) * STAPE_F + field) * self.s
    }

    fn addr_off(&self, bi: usize, ti: usize, which: usize) -> usize {
        ((bi * self.t + ti) * 2 + which) * self.d
    }
}

/// Outer-parallelize independent batch rows once `B·T·S·D` is large enough
/// that thread spawn is cheaper than the GEMVs.
const PARALLEL_BTSD: usize = 65_536;

struct SlotScratch {
    q: Vec<f32>,
    y: Vec<f32>,
    addr: Vec<f32>,
    delta: Vec<f32>,
    s_read: Vec<f32>,
    k_c: Vec<f32>,
    k_alloc: Vec<f32>,
    k_w: Vec<f32>,
    d_s: Vec<f32>,
    d_s_sm: Vec<f32>,
    d_qr: Vec<f32>,
    mk: Vec<f32>,
    mv: Vec<f32>,
    d_kw: Vec<f32>,
    d_mk: Vec<f32>,
    d_mv: Vec<f32>,
    d_kc: Vec<f32>,
    d_kc_sm: Vec<f32>,
    d_qw: Vec<f32>,
    wv: Vec<f32>,
    gkey: Vec<f32>,
    gval: Vec<f32>,
    read: Vec<f32>,
}

impl SlotScratch {
    fn new(d: usize, s: usize) -> Self {
        Self {
            q: vec![0.0; d],
            y: vec![0.0; d],
            addr: vec![0.0; d],
            delta: vec![0.0; d],
            s_read: vec![0.0; s],
            k_c: vec![0.0; s],
            k_alloc: vec![0.0; s],
            k_w: vec![0.0; s],
            d_s: vec![0.0; s],
            d_s_sm: vec![0.0; s],
            d_qr: vec![0.0; d],
            mk: vec![0.0; d],
            mv: vec![0.0; d],
            d_kw: vec![0.0; s],
            d_mk: vec![0.0; d],
            d_mv: vec![0.0; d],
            d_kc: vec![0.0; s],
            d_kc_sm: vec![0.0; s],
            d_qw: vec![0.0; d],
            wv: vec![0.0; d],
            gkey: vec![0.0; d],
            gval: vec![0.0; d],
            read: vec![0.0; d],
        }
    }
}

/// `cblas_sgemv` on `S×D` (tens of kFLOP) is launch-bound. A fused loop
/// wins for every `S, D` the slot tables actually use; the threshold is
/// in elements so a future wide table can still fall through to BLAS.
const MICRO_ELEMS: usize = 512 * 1024;

fn gemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    if m.saturating_mul(n) > MICRO_ELEMS {
        crate::accelerate::sgemv(trans, m, n, alpha, a, x, beta, y).expect("slot sgemv");
        return;
    }
    if trans {
        gemv_t(m, n, alpha, a, x, beta, y);
    } else {
        gemv_n(m, n, alpha, a, x, beta, y);
    }
}

fn ger(m: usize, n: usize, alpha: f32, x: &[f32], y: &[f32], a: &mut [f32]) {
    if m.saturating_mul(n) > MICRO_ELEMS {
        crate::accelerate::sger(m, n, alpha, x, y, a).expect("slot sger");
        return;
    }
    ger_micro(m, n, alpha, x, y, a);
}

fn dot_n(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut i = 0;
    while i + 4 <= n {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut acc = acc0 + acc1 + acc2 + acc3;
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

fn axpy_n(alpha: f32, x: &[f32], y: &mut [f32], n: usize) {
    if alpha.abs() < 1e-20 {
        return;
    }
    for i in 0..n {
        y[i] += alpha * x[i];
    }
}

fn gemv_n(m: usize, n: usize, alpha: f32, a: &[f32], x: &[f32], beta: f32, y: &mut [f32]) {
    for i in 0..m {
        let acc = dot_n(&a[i * n..], x, n);
        y[i] = if beta == 0.0 {
            alpha * acc
        } else {
            alpha * acc + beta * y[i]
        };
    }
}

fn gemv_t(m: usize, n: usize, alpha: f32, a: &[f32], x: &[f32], beta: f32, y: &mut [f32]) {
    if beta == 0.0 {
        y[..n].fill(0.0);
    } else if beta != 1.0 {
        for yi in &mut y[..n] {
            *yi *= beta;
        }
    }
    for i in 0..m {
        axpy_n(alpha * x[i], &a[i * n..], y, n);
    }
}

fn ger_micro(m: usize, n: usize, alpha: f32, x: &[f32], y: &[f32], a: &mut [f32]) {
    for i in 0..m {
        axpy_n(alpha * x[i], y, &mut a[i * n..], n);
    }
}

fn softmax_inplace(scores: &mut [f32]) {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        z += *s;
    }
    let inv = 1.0 / z.max(1e-20);
    for s in scores.iter_mut() {
        *s *= inv;
    }
}

fn softmax_bwd(y: &[f32], dy: &[f32], dx: &mut [f32]) {
    let mut acc = 0.0f32;
    for i in 0..y.len() {
        acc += y[i] * dy[i];
    }
    for i in 0..y.len() {
        dx[i] = y[i] * (dy[i] - acc);
    }
}

fn l2_norm(x: &[f32]) -> f32 {
    dot_n(x, x, x.len()).sqrt()
}

fn cosine_from_dot(dot: f32, kn: f32, qn: f32) -> f32 {
    if kn < 1e-8 {
        0.0
    } else {
        CONTENT_TEMP * dot / (kn * qn)
    }
}

/// Cosine scores for two queries in one pass over `K`. Empty keys stay 0.
fn content_pair(
    key: &[f32],
    q1: &[f32],
    q2: &[f32],
    n_slots: usize,
    d: usize,
    out1: &mut [f32],
    out2: &mut [f32],
) {
    let qn1 = l2_norm(&q1[..d]).max(1e-6);
    let qn2 = l2_norm(&q2[..d]).max(1e-6);
    for s in 0..n_slots {
        let row = &key[s * d..];
        let mut n2 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut d2 = 0.0f32;
        let mut c = 0;
        while c + 4 <= d {
            let k0 = row[c];
            let k1 = row[c + 1];
            let k2 = row[c + 2];
            let k3 = row[c + 3];
            n2 += k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
            d1 += k0 * q1[c] + k1 * q1[c + 1] + k2 * q1[c + 2] + k3 * q1[c + 3];
            d2 += k0 * q2[c] + k1 * q2[c + 1] + k2 * q2[c + 2] + k3 * q2[c + 3];
            c += 4;
        }
        while c < d {
            let k = row[c];
            n2 += k * k;
            d1 += k * q1[c];
            d2 += k * q2[c];
            c += 1;
        }
        let kn = n2.sqrt();
        out1[s] = cosine_from_dot(d1, kn, qn1);
        out2[s] = cosine_from_dot(d2, kn, qn2);
    }
}

fn content_scores_bwd(
    key: &[f32],
    q: &[f32],
    d_score: &[f32],
    n_slots: usize,
    d: usize,
    d_key: &mut [f32],
    d_q: &mut [f32],
) {
    let qn = l2_norm(&q[..d]).max(1e-6);
    let mut d_qn = 0.0f32;
    for s in 0..n_slots {
        let row = &key[s * d..];
        let mut n2 = 0.0f32;
        let mut dot = 0.0f32;
        for c in 0..d {
            let k = row[c];
            n2 += k * k;
            dot += k * q[c];
        }
        let kn = n2.sqrt();
        if kn < 1e-8 {
            continue;
        }
        let ds = d_score[s];
        let inv = 1.0 / (kn * qn);
        let y = CONTENT_TEMP * dot * inv;
        let d_dot = CONTENT_TEMP * inv * ds;
        let d_kn = -y / kn * ds;
        d_qn += -y / qn * ds;
        let coef_k = d_kn / kn;
        let dk = &mut d_key[s * d..];
        for c in 0..d {
            dk[c] += coef_k * row[c] + d_dot * q[c];
            d_q[c] += d_dot * row[c];
        }
    }
    let coef_q = d_qn / qn;
    for c in 0..d {
        d_q[c] += coef_q * q[c];
    }
}

fn alloc_weights(usage: &[f32], out: &mut [f32]) {
    for (s, &u) in usage.iter().enumerate() {
        out[s] = -ALLOC_TEMP * u - 4.0 * s as f32;
    }
    softmax_inplace(out);
}

fn mix_addr(
    k_c: &[f32],
    k_alloc: &[f32],
    p_prev: &[f32],
    link: f32,
    g_alloc: f32,
    k_w: &mut [f32],
) {
    let psum: f32 = p_prev.iter().sum();
    for i in 0..k_c.len() {
        let base = (1.0 - g_alloc) * k_c[i] + g_alloc * k_alloc[i];
        k_w[i] = if psum < 1e-8 {
            base
        } else {
            link * p_prev[i] + (1.0 - link) * base
        };
    }
}

fn apply_write(
    tab: &mut [f32],
    k_w: &[f32],
    v: &[f32],
    beta: f32,
    n_slots: usize,
    d: usize,
    addr: &mut [f32],
    delta: &mut [f32],
    addr_out: Option<&mut [f32]>,
) {
    gemv(true, n_slots, d, 1.0, tab, k_w, 0.0, addr);
    if let Some(out) = addr_out {
        out[..d].copy_from_slice(&addr[..d]);
    }
    if !beta.is_finite() || beta.abs() < 1e-20 {
        return;
    }
    for c in 0..d {
        delta[c] = v[c] - addr[c];
    }
    ger(n_slots, d, beta, k_w, delta, tab);
}

fn apply_undo(
    tab: &mut [f32],
    k_w: &[f32],
    v: &[f32],
    beta: f32,
    addr: &[f32],
    n_slots: usize,
    d: usize,
    delta: &mut [f32],
) {
    if !beta.is_finite() || beta.abs() < 1e-20 {
        return;
    }
    for c in 0..d {
        delta[c] = v[c] - addr[c];
    }
    ger(n_slots, d, -beta, k_w, delta, tab);
}

fn batch_span(n_slots: usize, d: usize, b: usize) -> (usize, usize, usize) {
    let n = n_slots * d;
    (b * n, (b + 1) * n, b * n_slots)
}

fn one_token(
    params: &SlotParams,
    vt: &[f32],
    key: &mut [f32],
    val: &mut [f32],
    usage: &mut [f32],
    p_prev: &mut [f32],
    scratch: &mut SlotScratch,
    y_out: &mut [f32],
    tape: Option<(&mut [f32], &mut [f32], &mut [f32], &mut [f32])>,
) {
    let d = params.d;
    let s_n = params.n_slots;
    for c in 0..d {
        scratch.q[c] = params.w_q[c] * vt[c];
        scratch.wv[c] = params.w_w[c] * vt[c];
    }
    content_pair(
        key,
        &scratch.q,
        &scratch.wv,
        s_n,
        d,
        &mut scratch.s_read,
        &mut scratch.k_c,
    );
    softmax_inplace(&mut scratch.s_read);
    gemv(
        true,
        s_n,
        d,
        params.gamma,
        val,
        &scratch.s_read,
        0.0,
        &mut scratch.y,
    );
    softmax_inplace(&mut scratch.k_c);
    alloc_weights(usage, &mut scratch.k_alloc);
    let mut link_logit = params.b_link;
    for c in 0..d {
        link_logit += params.w_link[c] * vt[c];
    }
    let link = sigmoid(link_logit);
    let g_alloc = sigmoid(params.b_alloc);
    mix_addr(
        &scratch.k_c,
        &scratch.k_alloc,
        p_prev,
        link,
        g_alloc,
        &mut scratch.k_w,
    );
    let mean_v = vt.iter().sum::<f32>() / d as f32;
    let psum: f32 = p_prev.iter().sum();
    let follow = if psum < 1e-8 { 0.0 } else { link };
    let sig_k = sigmoid(params.b_bk + params.w_bk * mean_v);
    let sig_v = sigmoid(params.b_bv + params.w_bv * mean_v);
    let beta_k = sig_k * (1.0 - follow);
    let beta_v = sig_v * follow;
    if let Some((st, sc, addr_k, addr_v)) = tape {
        st[F_S * s_n..(F_S + 1) * s_n].copy_from_slice(&scratch.s_read[..s_n]);
        st[F_KC * s_n..(F_KC + 1) * s_n].copy_from_slice(&scratch.k_c[..s_n]);
        st[F_ALLOC * s_n..(F_ALLOC + 1) * s_n].copy_from_slice(&scratch.k_alloc[..s_n]);
        st[F_KW * s_n..(F_KW + 1) * s_n].copy_from_slice(&scratch.k_w[..s_n]);
        st[F_PP * s_n..(F_PP + 1) * s_n].copy_from_slice(&p_prev[..s_n]);
        sc[0] = link;
        sc[1] = g_alloc;
        sc[2] = beta_k;
        sc[3] = beta_v;
        sc[4] = mean_v;
        sc[5] = sig_k;
        sc[6] = sig_v;
        sc[7] = 0.0;
        apply_write(
            key,
            &scratch.k_w,
            vt,
            beta_k,
            s_n,
            d,
            &mut scratch.addr,
            &mut scratch.delta,
            Some(addr_k),
        );
        apply_write(
            val,
            &scratch.k_w,
            vt,
            beta_v,
            s_n,
            d,
            &mut scratch.addr,
            &mut scratch.delta,
            Some(addr_v),
        );
    } else {
        apply_write(
            key,
            &scratch.k_w,
            vt,
            beta_k,
            s_n,
            d,
            &mut scratch.addr,
            &mut scratch.delta,
            None,
        );
        apply_write(
            val,
            &scratch.k_w,
            vt,
            beta_v,
            s_n,
            d,
            &mut scratch.addr,
            &mut scratch.delta,
            None,
        );
    }
    for s in 0..s_n {
        usage[s] += scratch.k_w[s];
    }
    p_prev[..s_n].copy_from_slice(&scratch.k_w[..s_n]);
    y_out.copy_from_slice(&scratch.y);
}

fn fwd_batch(
    params: &SlotParams,
    v: &[f32],
    t: usize,
    key: &mut [f32],
    val: &mut [f32],
    usage: &mut [f32],
    p_prev: &mut [f32],
    y: &mut [f32],
    stape: &mut [f32],
    scal: &mut [f32],
    addr: &mut [f32],
    final_key: &mut [f32],
    final_val: &mut [f32],
) {
    let d = params.d;
    let s_n = params.n_slots;
    let sd = s_n * d;
    let mut scratch = SlotScratch::new(d, s_n);
    for ti in 0..t {
        let row = ti * d;
        let st0 = ti * STAPE_F * s_n;
        let sc0 = ti * SCAL_N;
        let a0 = ti * 2 * d;
        let (addr_k, addr_v) = addr[a0..a0 + 2 * d].split_at_mut(d);
        one_token(
            params,
            &v[row..row + d],
            key,
            val,
            usage,
            p_prev,
            &mut scratch,
            &mut y[row..row + d],
            Some((
                &mut stape[st0..st0 + STAPE_F * s_n],
                &mut scal[sc0..sc0 + SCAL_N],
                addr_k,
                addr_v,
            )),
        );
    }
    final_key[..sd].copy_from_slice(&key[..sd]);
    final_val[..sd].copy_from_slice(&val[..sd]);
}

pub fn slots_forward(
    params: &SlotParams,
    v: &[f32],
    b: usize,
    t: usize,
    state: &mut SlotState,
) -> Result<(Vec<f32>, SlotTape)> {
    let d = params.d;
    let s_n = params.n_slots;
    if v.len() != b * t * d {
        bail!("slots v len");
    }
    if state.batch != b || state.n_slots != s_n || state.d != d {
        *state = SlotState::new(b, s_n, d);
    } else {
        state.reset();
    }
    let mut y = vec![0.0f32; v.len()];
    let mut tape = SlotTape::alloc(b, t, s_n, d);
    tape.v.copy_from_slice(v);
    let sd = s_n * d;
    let parallel =
        b > 1 && b.saturating_mul(t).saturating_mul(s_n).saturating_mul(d) >= PARALLEL_BTSD;
    let key_ch = state.key.chunks_exact_mut(sd);
    let val_ch = state.val.chunks_exact_mut(sd);
    let use_ch = state.usage.chunks_exact_mut(s_n);
    let prev_ch = state.p_prev.chunks_exact_mut(s_n);
    let y_ch = y.chunks_exact_mut(t * d);
    let v_ch = v.chunks_exact(t * d);
    let st_ch = tape.stape.chunks_exact_mut(t * STAPE_F * s_n);
    let sc_ch = tape.scal.chunks_exact_mut(t * SCAL_N);
    let ad_ch = tape.addr.chunks_exact_mut(t * 2 * d);
    let fk_ch = tape.final_key.chunks_exact_mut(sd);
    let fv_ch = tape.final_val.chunks_exact_mut(sd);
    std::thread::scope(|scope| {
        for ((((((((((key, val), usage), p_prev), y_b), v_b), st), sc), ad), fk), fv) in key_ch
            .zip(val_ch)
            .zip(use_ch)
            .zip(prev_ch)
            .zip(y_ch)
            .zip(v_ch)
            .zip(st_ch)
            .zip(sc_ch)
            .zip(ad_ch)
            .zip(fk_ch)
            .zip(fv_ch)
        {
            if parallel {
                scope.spawn(move || {
                    fwd_batch(
                        params, v_b, t, key, val, usage, p_prev, y_b, st, sc, ad, fk, fv,
                    );
                });
            } else {
                fwd_batch(
                    params, v_b, t, key, val, usage, p_prev, y_b, st, sc, ad, fk, fv,
                );
            }
        }
    });
    Ok((y, tape))
}

pub fn slots_step(
    params: &SlotParams,
    v_tok: &[f32],
    b: usize,
    state: &mut SlotState,
) -> Result<Vec<f32>> {
    let d = params.d;
    let s_n = params.n_slots;
    if v_tok.len() != b * d {
        bail!("slots_step v");
    }
    if state.batch != b || state.n_slots != s_n || state.d != d {
        *state = SlotState::new(b, s_n, d);
    }
    let mut y = vec![0.0f32; b * d];
    let mut scratch = SlotScratch::new(d, s_n);
    for bi in 0..b {
        let (k0, k1, p_off) = batch_span(s_n, d, bi);
        one_token(
            params,
            &v_tok[bi * d..(bi + 1) * d],
            &mut state.key[k0..k1],
            &mut state.val[k0..k1],
            &mut state.usage[p_off..p_off + s_n],
            &mut state.p_prev[p_off..p_off + s_n],
            &mut scratch,
            &mut y[bi * d..(bi + 1) * d],
            None,
        );
    }
    Ok(y)
}

fn bwd_batch(params: &mut SlotParams, tape: &SlotTape, bi: usize, dy: &[f32], dv: &mut [f32]) {
    let d = params.d;
    let s_n = params.n_slots;
    let t = tape.t;
    let sd = s_n * d;
    let mut scratch = SlotScratch::new(d, s_n);
    let mut work_key = tape.final_key[bi * sd..bi * sd + sd].to_vec();
    let mut work_val = tape.final_val[bi * sd..bi * sd + sd].to_vec();
    let mut d_key = vec![0.0f32; sd];
    let mut d_val = vec![0.0f32; sd];
    let mut d_p = vec![0.0f32; s_n];
    for ti in (0..t).rev() {
        let st = step_view(tape, bi, ti);
        apply_undo(
            &mut work_key,
            st.k_w,
            st.v,
            st.beta_k,
            st.addr_k,
            s_n,
            d,
            &mut scratch.delta,
        );
        apply_undo(
            &mut work_val,
            st.k_w,
            st.v,
            st.beta_v,
            st.addr_v,
            s_n,
            d,
            &mut scratch.delta,
        );
        bwd_one(
            params,
            &st,
            &work_key,
            &work_val,
            &dy[ti * d..ti * d + d],
            &mut d_key,
            &mut d_val,
            &mut d_p,
            &mut dv[ti * d..ti * d + d],
            &mut scratch,
        );
    }
}

fn step_view(tape: &SlotTape, bi: usize, ti: usize) -> StepView<'_> {
    let s_n = tape.s;
    let d = tape.d;
    let st0 = tape.st_off(bi, ti, 0);
    let sc0 = (bi * tape.t + ti) * SCAL_N;
    let sc = &tape.scal[sc0..sc0 + SCAL_N];
    let v0 = (bi * tape.t + ti) * d;
    let ak = tape.addr_off(bi, ti, 0);
    StepView {
        v: &tape.v[v0..v0 + d],
        s: &tape.stape[st0 + F_S * s_n..st0 + (F_S + 1) * s_n],
        k_c: &tape.stape[st0 + F_KC * s_n..st0 + (F_KC + 1) * s_n],
        k_alloc: &tape.stape[st0 + F_ALLOC * s_n..st0 + (F_ALLOC + 1) * s_n],
        k_w: &tape.stape[st0 + F_KW * s_n..st0 + (F_KW + 1) * s_n],
        p_prev: &tape.stape[st0 + F_PP * s_n..st0 + (F_PP + 1) * s_n],
        addr_k: &tape.addr[ak..ak + d],
        addr_v: &tape.addr[ak + d..ak + 2 * d],
        link: sc[0],
        g_alloc: sc[1],
        beta_k: sc[2],
        beta_v: sc[3],
        mean_v: sc[4],
        sig_k: sc[5],
        sig_v: sc[6],
    }
}

pub fn slots_backward(
    params: &mut SlotParams,
    tape: &SlotTape,
    dy: &[f32],
    b: usize,
    t: usize,
) -> Result<Vec<f32>> {
    let d = params.d;
    let s_n = params.n_slots;
    if dy.len() != b * t * d {
        bail!("slots_backward dy");
    }
    if tape.b != b || tape.t != t || tape.s != s_n || tape.d != d {
        bail!("slots tape shape");
    }
    let mut dv = vec![0.0f32; b * t * d];
    let parallel =
        b > 1 && b.saturating_mul(t).saturating_mul(s_n).saturating_mul(d) >= PARALLEL_BTSD;
    if parallel {
        let mut locals: Vec<SlotParams> = (0..b)
            .map(|_| {
                let mut p = params.clone();
                p.zero_grad();
                p
            })
            .collect();
        let dy_ch = dy.chunks_exact(t * d);
        let dv_ch = dv.chunks_exact_mut(t * d);
        std::thread::scope(|scope| {
            for (bi, ((lp, dy_b), dv_b)) in locals.iter_mut().zip(dy_ch).zip(dv_ch).enumerate() {
                scope.spawn(move || {
                    bwd_batch(lp, tape, bi, dy_b, dv_b);
                });
            }
        });
        for lp in &locals {
            params.add_grads(lp);
        }
    } else {
        for bi in 0..b {
            bwd_batch(
                params,
                tape,
                bi,
                &dy[bi * t * d..(bi + 1) * t * d],
                &mut dv[bi * t * d..(bi + 1) * t * d],
            );
        }
    }
    Ok(dv)
}

fn bwd_one(
    params: &mut SlotParams,
    step: &StepView<'_>,
    key_before: &[f32],
    val_before: &[f32],
    dy: &[f32],
    d_key: &mut [f32],
    d_val: &mut [f32],
    d_p: &mut [f32],
    dv: &mut [f32],
    scratch: &mut SlotScratch,
) {
    let d = params.d;
    let s_n = params.n_slots;

    gemv(
        true,
        s_n,
        d,
        1.0,
        val_before,
        step.s,
        0.0,
        &mut scratch.read,
    );
    params.grad_gamma += dot_n(dy, &scratch.read, d);

    ger(s_n, d, params.gamma, step.s, dy, d_val);
    gemv(
        false,
        s_n,
        d,
        params.gamma,
        val_before,
        dy,
        0.0,
        &mut scratch.d_s,
    );
    softmax_bwd(step.s, &scratch.d_s, &mut scratch.d_s_sm);
    for c in 0..d {
        scratch.wv[c] = params.w_q[c] * step.v[c];
    }
    scratch.d_qr.fill(0.0);
    content_scores_bwd(
        key_before,
        &scratch.wv,
        &scratch.d_s_sm,
        s_n,
        d,
        d_key,
        &mut scratch.d_qr,
    );
    for c in 0..d {
        params.grad_w_q[c] += scratch.d_qr[c] * step.v[c];
        dv[c] += scratch.d_qr[c] * params.w_q[c];
    }

    gemv(
        true,
        s_n,
        d,
        1.0,
        key_before,
        step.k_w,
        0.0,
        &mut scratch.mk,
    );
    gemv(
        true,
        s_n,
        d,
        1.0,
        val_before,
        step.k_w,
        0.0,
        &mut scratch.mv,
    );
    scratch.d_kw[..s_n].copy_from_slice(&d_p[..s_n]);
    d_p.fill(0.0);
    for c in 0..d {
        scratch.delta[c] = step.v[c] - scratch.mk[c];
        scratch.addr[c] = step.v[c] - scratch.mv[c];
    }
    gemv(true, s_n, d, 1.0, d_key, step.k_w, 0.0, &mut scratch.gkey);
    gemv(true, s_n, d, 1.0, d_val, step.k_w, 0.0, &mut scratch.gval);
    let d_betak = dot_n(&scratch.gkey, &scratch.delta, d);
    let d_betav = dot_n(&scratch.gval, &scratch.addr, d);
    gemv(
        false,
        s_n,
        d,
        step.beta_k,
        d_key,
        &scratch.delta,
        1.0,
        &mut scratch.d_kw,
    );
    gemv(
        false,
        s_n,
        d,
        step.beta_v,
        d_val,
        &scratch.addr,
        1.0,
        &mut scratch.d_kw,
    );
    for c in 0..d {
        dv[c] += step.beta_k * scratch.gkey[c] + step.beta_v * scratch.gval[c];
        scratch.d_mk[c] = -step.beta_k * scratch.gkey[c];
        scratch.d_mv[c] = -step.beta_v * scratch.gval[c];
    }
    ger(s_n, d, 1.0, step.k_w, &scratch.d_mk, d_key);
    ger(s_n, d, 1.0, step.k_w, &scratch.d_mv, d_val);
    gemv(
        false,
        s_n,
        d,
        1.0,
        key_before,
        &scratch.d_mk,
        1.0,
        &mut scratch.d_kw,
    );
    gemv(
        false,
        s_n,
        d,
        1.0,
        val_before,
        &scratch.d_mv,
        1.0,
        &mut scratch.d_kw,
    );

    let psum: f32 = step.p_prev.iter().sum();
    let follow = if psum < 1e-8 { 0.0 } else { step.link };
    let dpre_bk = step.sig_k * (1.0 - step.sig_k) * (1.0 - follow) * d_betak;
    let dpre_bv = step.sig_v * (1.0 - step.sig_v) * follow * d_betav;
    params.grad_b_bk += dpre_bk;
    params.grad_w_bk += dpre_bk * step.mean_v;
    params.grad_b_bv += dpre_bv;
    params.grad_w_bv += dpre_bv * step.mean_v;
    let dmean = dpre_bk * params.w_bk + dpre_bv * params.w_bv;
    let inv_d = 1.0 / d as f32;
    for c in 0..d {
        dv[c] += dmean * inv_d;
    }
    let d_follow = -step.sig_k * d_betak + step.sig_v * d_betav;

    let mut d_link = 0.0f32;
    let mut d_galloc = 0.0f32;
    scratch.d_kc.fill(0.0);
    for s in 0..s_n {
        let base = (1.0 - step.g_alloc) * step.k_c[s] + step.g_alloc * step.k_alloc[s];
        if psum < 1e-8 {
            scratch.d_kc[s] += scratch.d_kw[s] * (1.0 - step.g_alloc);
            d_galloc += scratch.d_kw[s] * (step.k_alloc[s] - step.k_c[s]);
        } else {
            d_p[s] += scratch.d_kw[s] * step.link;
            d_link += scratch.d_kw[s] * (step.p_prev[s] - base);
            let db = scratch.d_kw[s] * (1.0 - step.link);
            scratch.d_kc[s] += db * (1.0 - step.g_alloc);
            d_galloc += db * (step.k_alloc[s] - step.k_c[s]);
        }
    }
    if psum >= 1e-8 {
        d_link += d_follow;
    }
    params.grad_b_alloc += step.g_alloc * (1.0 - step.g_alloc) * d_galloc;
    let dpre_l = step.link * (1.0 - step.link) * d_link;
    params.grad_b_link += dpre_l;
    for c in 0..d {
        params.grad_w_link[c] += dpre_l * step.v[c];
        dv[c] += dpre_l * params.w_link[c];
    }
    softmax_bwd(step.k_c, &scratch.d_kc, &mut scratch.d_kc_sm);
    for c in 0..d {
        scratch.wv[c] = params.w_w[c] * step.v[c];
    }
    scratch.d_qw.fill(0.0);
    content_scores_bwd(
        key_before,
        &scratch.wv,
        &scratch.d_kc_sm,
        s_n,
        d,
        d_key,
        &mut scratch.d_qw,
    );
    for c in 0..d {
        params.grad_w_w[c] += scratch.d_qw[c] * step.v[c];
        dv[c] += scratch.d_qw[c] * params.w_w[c];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_value_write_then_read() {
        let d = 8usize;
        let s = 4usize;
        let mut p = SlotParams::new(d, s);
        p.w_link = vec![0.0; d];
        p.w_link[0] = 8.0;
        p.b_link = 0.0;
        p.b_bk = 4.0;
        p.b_bv = 4.0;
        p.gamma = 1.0;
        let mut name = vec![0.0f32; d];
        let mut val = vec![0.0f32; d];
        name[0] = -1.0;
        name[1] = 1.0;
        val[0] = 1.0;
        val[2] = 1.0;
        let mut seq = vec![0.0f32; 3 * d];
        seq[..d].copy_from_slice(&name);
        seq[d..2 * d].copy_from_slice(&val);
        seq[2 * d..].copy_from_slice(&name);
        let mut st = SlotState::new(1, s, d);
        let (y, _) = slots_forward(&p, &seq, 1, 3, &mut st).unwrap();
        let read = &y[2 * d..];
        let mut dot_v = 0.0f32;
        let mut dot_n = 0.0f32;
        for c in 0..d {
            dot_v += read[c] * val[c];
            dot_n += read[c] * name[c];
        }
        assert!(
            dot_v > dot_n,
            "read should match value more than name (v={dot_v} n={dot_n})"
        );
    }

    #[test]
    fn cosine_read_finds_written_slot_at_language_embed_scale() {
        let d = 32usize;
        let s = 16usize;
        let mut p = SlotParams::new(d, s);
        p.w_link = vec![0.0; d];
        p.b_link = 8.0;
        p.b_bk = 4.0;
        p.b_bv = 4.0;
        p.gamma = 1.0;
        p.b_alloc = 4.0;
        let mut name = vec![0.0f32; d];
        name[0] = 0.02;
        name[1] = 0.03;
        let mut val = vec![0.0f32; d];
        val[4] = 0.04;
        val[7] = -0.03;
        let mut seq = vec![0.0f32; 3 * d];
        seq[..d].copy_from_slice(&name);
        seq[d..2 * d].copy_from_slice(&val);
        seq[2 * d..].copy_from_slice(&name);
        let mut st = SlotState::new(1, s, d);
        let (y, _) = slots_forward(&p, &seq, 1, 3, &mut st).unwrap();
        let read = &y[2 * d..];
        let mut dot_v = 0.0f32;
        let mut dot_n = 0.0f32;
        for c in 0..d {
            dot_v += read[c] * val[c];
            dot_n += read[c] * name[c];
        }
        assert!(
            dot_v > dot_n,
            "cosine content must bind at N(0,0.02)-scale embeds (v={dot_v} n={dot_n})"
        );
    }

    #[test]
    fn write_undo_roundtrip() {
        let d = 8usize;
        let s = 5usize;
        let mut tab: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let orig = tab.clone();
        let k_w = [0.1f32, 0.4, 0.2, 0.2, 0.1];
        let v: Vec<f32> = (0..d).map(|i| (i as f32) * 0.05).collect();
        let beta = 0.7f32;
        let mut addr = vec![0.0f32; d];
        let mut delta = vec![0.0f32; d];
        let mut saved = vec![0.0f32; d];
        apply_write(
            &mut tab,
            &k_w,
            &v,
            beta,
            s,
            d,
            &mut addr,
            &mut delta,
            Some(&mut saved),
        );
        assert!(tab.iter().zip(&orig).any(|(a, b)| (a - b).abs() > 1e-6));
        apply_undo(&mut tab, &k_w, &v, beta, &saved, s, d, &mut delta);
        for i in 0..tab.len() {
            assert!(
                (tab[i] - orig[i]).abs() < 1e-5,
                "undo[{i}] {} vs {}",
                tab[i],
                orig[i]
            );
        }
    }

    #[test]
    fn large_step_not_pathological() {
        let d = 64usize;
        let s = 128usize;
        let t = 64usize;
        let b = 4usize;
        let p = SlotParams::new(d, s);
        let v = vec![0.02f32; b * t * d];
        let dy = vec![0.01f32; b * t * d];
        let mut st = SlotState::new(b, s, d);
        let t0 = std::time::Instant::now();
        let (_, tape) = slots_forward(&p, &v, b, t, &mut st).unwrap();
        let mut p = p;
        let _ = slots_backward(&mut p, &tape, &dy, b, t).unwrap();
        let ms = t0.elapsed().as_secs_f32() * 1e3;
        assert!(
            ms < 2_500.0,
            "one slot layer B={b} T={t} S={s} D={d} took {ms:.1}ms"
        );
    }

    #[test]
    #[ignore = "release microbench; run with --ignored --release"]
    fn bench_user_shape_slot_layer() {
        let d = 512usize;
        let s = 124usize;
        let t = 256usize;
        let b = 4usize;
        let p = SlotParams::new(d, s);
        let v = vec![0.02f32; b * t * d];
        let dy = vec![0.01f32; b * t * d];
        let mut st = SlotState::new(b, s, d);
        let t0 = std::time::Instant::now();
        let (_, tape) = slots_forward(&p, &v, b, t, &mut st).unwrap();
        let fwd = t0.elapsed().as_secs_f32() * 1e3;
        let mut p = p;
        let t1 = std::time::Instant::now();
        let _ = slots_backward(&mut p, &tape, &dy, b, t).unwrap();
        let bwd = t1.elapsed().as_secs_f32() * 1e3;
        eprintln!("slot layer D={d} S={s} T={t} B={b} fwd={fwd:.1}ms bwd={bwd:.1}ms");
        assert!(
            fwd + bwd < 800.0,
            "one slot layer D={d} S={s} T={t} B={b} fwd={fwd:.1}ms bwd={bwd:.1}ms"
        );
    }

    #[test]
    fn backward_matches_central_diff() {
        let d = 4usize;
        let s = 3usize;
        let t = 3usize;
        let mut p = SlotParams::new(d, s);
        p.w_q = vec![0.7, -0.2, 0.4, 0.1];
        p.w_w = vec![0.3, 0.5, -0.1, 0.8];
        p.w_link = vec![0.2, -0.4, 0.1, 0.0];
        p.b_link = 0.3;
        p.b_bk = 0.8;
        p.w_bk = 0.4;
        p.b_bv = -0.2;
        p.w_bv = 0.6;
        p.gamma = 1.3;
        p.b_alloc = 1.1;
        let v: Vec<f32> = (0..t * d).map(|i| ((i % 7) as f32) * 0.15 - 0.4).collect();
        let dy: Vec<f32> = (0..t * d).map(|i| ((i % 5) as f32) * 0.1 - 0.2).collect();
        let mut st = SlotState::new(1, s, d);
        let (_, tape) = slots_forward(&p, &v, 1, t, &mut st).unwrap();
        let dv = slots_backward(&mut p, &tape, &dy, 1, t).unwrap();
        let eps = 2e-3f32;
        for i in 0..v.len() {
            let mut vp = v.clone();
            vp[i] += eps;
            let mut st = SlotState::new(1, s, d);
            let (yp, _) = slots_forward(&p, &vp, 1, t, &mut st).unwrap();
            let mut vm = v.clone();
            vm[i] -= eps;
            let mut st = SlotState::new(1, s, d);
            let (ym, _) = slots_forward(&p, &vm, 1, t, &mut st).unwrap();
            let mut lp = 0.0f32;
            let mut lm = 0.0f32;
            for j in 0..dy.len() {
                lp += yp[j] * dy[j];
                lm += ym[j] * dy[j];
            }
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (fd - dv[i]).abs() < 0.08,
                "dv[{i}] analytic={:.5} fd={:.5}",
                dv[i],
                fd
            );
        }
    }
}
