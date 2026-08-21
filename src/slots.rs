//! Content-based key/value slot memory. Control plane, FP32, `[B, T, D]`.
//!
//! Read: `s = softmax(M_key q / √D)`, `out = γ sᵀ M_val`.
//! Write: content scores on `M_key` plus temporal link to the previous
//! write, then a GRU-style update on the addressed slots. Keys and values
//! are separate tables so a value write does not erase the name.

use anyhow::{bail, Result};

use crate::scan::sigmoid;

const ALLOC_TEMP: f32 = 8.0;

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

#[derive(Clone, Debug)]
struct SlotStep {
    v: Vec<f32>,
    s: Vec<f32>,
    k_c: Vec<f32>,
    k_alloc: Vec<f32>,
    k_w: Vec<f32>,
    p_prev: Vec<f32>,
    link: f32,
    g_alloc: f32,
    beta_k: f32,
    beta_v: f32,
    mean_v: f32,
}

#[derive(Clone, Debug)]
struct SlotCkpt {
    key: Vec<f32>,
    val: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct SlotTape {
    steps: Vec<Vec<SlotStep>>,
    ckpts: Vec<Vec<SlotCkpt>>,
    chunk: usize,
}

const SLOT_CHUNK: usize = 16;
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
    crate::accelerate::sgemv(trans, m, n, alpha, a, x, beta, y).expect("slot sgemv");
}

fn ger(m: usize, n: usize, alpha: f32, x: &[f32], y: &[f32], a: &mut [f32]) {
    crate::accelerate::sger(m, n, alpha, x, y, a).expect("slot sger");
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

fn content_scores(key: &[f32], q: &[f32], n_slots: usize, d: usize, out: &mut [f32]) {
    let scale = 24.0 / (d as f32).sqrt();
    gemv(false, n_slots, d, scale, key, q, 0.0, out);
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
) {
    if !beta.is_finite() || beta.abs() < 1e-20 {
        return;
    }
    gemv(true, n_slots, d, 1.0, tab, k_w, 0.0, addr);
    for c in 0..d {
        delta[c] = v[c] - addr[c];
    }
    ger(n_slots, d, beta, k_w, delta, tab);
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
) -> (Vec<f32>, SlotStep) {
    let d = params.d;
    let s_n = params.n_slots;
    for c in 0..d {
        scratch.q[c] = params.w_q[c] * vt[c];
    }
    content_scores(key, &scratch.q, s_n, d, &mut scratch.s_read);
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
    for c in 0..d {
        scratch.q[c] = params.w_w[c] * vt[c];
    }
    content_scores(key, &scratch.q, s_n, d, &mut scratch.k_c);
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
    let beta_k = sigmoid(params.b_bk + params.w_bk * mean_v) * (1.0 - follow);
    let beta_v = sigmoid(params.b_bv + params.w_bv * mean_v) * follow;
    let step = SlotStep {
        v: vt.to_vec(),
        s: scratch.s_read.clone(),
        k_c: scratch.k_c.clone(),
        k_alloc: scratch.k_alloc.clone(),
        k_w: scratch.k_w.clone(),
        p_prev: p_prev.to_vec(),
        link,
        g_alloc,
        beta_k,
        beta_v,
        mean_v,
    };
    apply_write(
        key,
        &scratch.k_w,
        vt,
        beta_k,
        s_n,
        d,
        &mut scratch.addr,
        &mut scratch.delta,
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
    );
    for s in 0..s_n {
        usage[s] += scratch.k_w[s];
    }
    p_prev[..s_n].copy_from_slice(&scratch.k_w[..s_n]);
    (scratch.y.clone(), step)
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
    steps: &mut Vec<SlotStep>,
    ckpts: &mut Vec<SlotCkpt>,
) {
    let d = params.d;
    let s_n = params.n_slots;
    let sd = s_n * d;
    let mut scratch = SlotScratch::new(d, s_n);
    steps.clear();
    steps.reserve(t);
    ckpts.clear();
    for ti in 0..t {
        if ti % SLOT_CHUNK == 0 {
            ckpts.push(SlotCkpt {
                key: key[..sd].to_vec(),
                val: val[..sd].to_vec(),
            });
        }
        let row = ti * d;
        let (yt, step) = one_token(
            params,
            &v[row..row + d],
            key,
            val,
            usage,
            p_prev,
            &mut scratch,
        );
        y[row..row + d].copy_from_slice(&yt);
        steps.push(step);
    }
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
    let mut tape = SlotTape {
        steps: (0..b).map(|_| Vec::with_capacity(t)).collect(),
        ckpts: (0..b).map(|_| Vec::new()).collect(),
        chunk: SLOT_CHUNK,
    };
    let sd = s_n * d;
    let parallel =
        b > 1 && b.saturating_mul(t).saturating_mul(s_n).saturating_mul(d) >= PARALLEL_BTSD;
    let key_ch = state.key.chunks_exact_mut(sd);
    let val_ch = state.val.chunks_exact_mut(sd);
    let use_ch = state.usage.chunks_exact_mut(s_n);
    let prev_ch = state.p_prev.chunks_exact_mut(s_n);
    let y_ch = y.chunks_exact_mut(t * d);
    let v_ch = v.chunks_exact(t * d);
    let step_ch = tape.steps.iter_mut();
    let ckpt_ch = tape.ckpts.iter_mut();
    std::thread::scope(|scope| {
        for (((((((key, val), usage), p_prev), y_b), v_b), steps), ckpts) in key_ch
            .zip(val_ch)
            .zip(use_ch)
            .zip(prev_ch)
            .zip(y_ch)
            .zip(v_ch)
            .zip(step_ch)
            .zip(ckpt_ch)
        {
            if parallel {
                scope.spawn(move || {
                    fwd_batch(params, v_b, t, key, val, usage, p_prev, y_b, steps, ckpts);
                });
            } else {
                fwd_batch(params, v_b, t, key, val, usage, p_prev, y_b, steps, ckpts);
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
        let (yt, _) = one_token(
            params,
            &v_tok[bi * d..(bi + 1) * d],
            &mut state.key[k0..k1],
            &mut state.val[k0..k1],
            &mut state.usage[p_off..p_off + s_n],
            &mut state.p_prev[p_off..p_off + s_n],
            &mut scratch,
        );
        y[bi * d..(bi + 1) * d].copy_from_slice(&yt);
    }
    Ok(y)
}

fn bwd_batch(
    params: &mut SlotParams,
    steps: &[SlotStep],
    ckpts: &[SlotCkpt],
    dy: &[f32],
    t: usize,
    chunk: usize,
    dv: &mut [f32],
) {
    let d = params.d;
    let s_n = params.n_slots;
    let sd = s_n * d;
    let chunk = chunk.max(1);
    let mut scratch = SlotScratch::new(d, s_n);
    let mut work_key = vec![0.0f32; sd];
    let mut work_val = vec![0.0f32; sd];
    let mut seg_key = vec![0.0f32; chunk * sd];
    let mut seg_val = vec![0.0f32; chunk * sd];
    let mut d_key = vec![0.0f32; sd];
    let mut d_val = vec![0.0f32; sd];
    let mut d_p = vec![0.0f32; s_n];
    for seg in (0..ckpts.len()).rev() {
        let t0 = seg * chunk;
        let t1 = (t0 + chunk).min(t);
        let n = t1 - t0;
        work_key.copy_from_slice(&ckpts[seg].key);
        work_val.copy_from_slice(&ckpts[seg].val);
        for i in 0..n {
            seg_key[i * sd..(i + 1) * sd].copy_from_slice(&work_key);
            seg_val[i * sd..(i + 1) * sd].copy_from_slice(&work_val);
            let st = &steps[t0 + i];
            apply_write(
                &mut work_key,
                &st.k_w,
                &st.v,
                st.beta_k,
                s_n,
                d,
                &mut scratch.addr,
                &mut scratch.delta,
            );
            apply_write(
                &mut work_val,
                &st.k_w,
                &st.v,
                st.beta_v,
                s_n,
                d,
                &mut scratch.addr,
                &mut scratch.delta,
            );
        }
        for i in (0..n).rev() {
            let ti = t0 + i;
            bwd_one(
                params,
                &steps[ti],
                &seg_key[i * sd..(i + 1) * sd],
                &seg_val[i * sd..(i + 1) * sd],
                &dy[ti * d..ti * d + d],
                &mut d_key,
                &mut d_val,
                &mut d_p,
                &mut dv[ti * d..ti * d + d],
                &mut scratch,
            );
        }
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
            for ((((lp, steps), ckpts), dy_b), dv_b) in locals
                .iter_mut()
                .zip(tape.steps.iter())
                .zip(tape.ckpts.iter())
                .zip(dy_ch)
                .zip(dv_ch)
            {
                let chunk = tape.chunk;
                scope.spawn(move || {
                    bwd_batch(lp, steps, ckpts, dy_b, t, chunk, dv_b);
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
                &tape.steps[bi],
                &tape.ckpts[bi],
                &dy[bi * t * d..(bi + 1) * t * d],
                t,
                tape.chunk,
                &mut dv[bi * t * d..(bi + 1) * t * d],
            );
        }
    }
    Ok(dv)
}

fn bwd_one(
    params: &mut SlotParams,
    step: &SlotStep,
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
    let scale = 24.0 / (d as f32).sqrt();

    gemv(
        true,
        s_n,
        d,
        1.0,
        val_before,
        &step.s,
        0.0,
        &mut scratch.read,
    );
    params.grad_gamma += crate::accelerate::dot(dy, &scratch.read).unwrap_or(0.0);

    ger(s_n, d, params.gamma, &step.s, dy, d_val);
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
    softmax_bwd(&step.s, &scratch.d_s, &mut scratch.d_s_sm);
    gemv(
        true,
        s_n,
        d,
        scale,
        key_before,
        &scratch.d_s_sm,
        0.0,
        &mut scratch.d_qr,
    );
    for c in 0..d {
        scratch.wv[c] = params.w_q[c] * step.v[c];
    }
    ger(s_n, d, scale, &scratch.d_s_sm, &scratch.wv, d_key);
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
        &step.k_w,
        0.0,
        &mut scratch.mk,
    );
    gemv(
        true,
        s_n,
        d,
        1.0,
        val_before,
        &step.k_w,
        0.0,
        &mut scratch.mv,
    );
    scratch.d_kw[..s_n].copy_from_slice(&d_p[..s_n]);
    d_p.fill(0.0);
    for c in 0..d {
        scratch.delta[c] = step.v[c] - scratch.mk[c];
        scratch.addr[c] = step.v[c] - scratch.mv[c];
    }
    gemv(true, s_n, d, 1.0, d_key, &step.k_w, 0.0, &mut scratch.gkey);
    gemv(true, s_n, d, 1.0, d_val, &step.k_w, 0.0, &mut scratch.gval);
    let d_betak = crate::accelerate::dot(&scratch.gkey, &scratch.delta).unwrap_or(0.0);
    let d_betav = crate::accelerate::dot(&scratch.gval, &scratch.addr).unwrap_or(0.0);
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
    ger(s_n, d, 1.0, &step.k_w, &scratch.d_mk, d_key);
    ger(s_n, d, 1.0, &step.k_w, &scratch.d_mv, d_val);
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

    let dpre_bk = step.beta_k * (1.0 - step.beta_k) * d_betak;
    let dpre_bv = step.beta_v * (1.0 - step.beta_v) * d_betav;
    params.grad_b_bk += dpre_bk;
    params.grad_w_bk += dpre_bk * step.mean_v;
    params.grad_b_bv += dpre_bv;
    params.grad_w_bv += dpre_bv * step.mean_v;
    let dmean = dpre_bk * params.w_bk + dpre_bv * params.w_bv;
    let inv_d = 1.0 / d as f32;
    for c in 0..d {
        dv[c] += dmean * inv_d;
    }

    let psum: f32 = step.p_prev.iter().sum();
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
    params.grad_b_alloc += step.g_alloc * (1.0 - step.g_alloc) * d_galloc;
    let dpre_l = step.link * (1.0 - step.link) * d_link;
    params.grad_b_link += dpre_l;
    for c in 0..d {
        params.grad_w_link[c] += dpre_l * step.v[c];
        dv[c] += dpre_l * params.w_link[c];
    }
    softmax_bwd(&step.k_c, &scratch.d_kc, &mut scratch.d_kc_sm);
    gemv(
        true,
        s_n,
        d,
        scale,
        key_before,
        &scratch.d_kc_sm,
        0.0,
        &mut scratch.d_qw,
    );
    for c in 0..d {
        scratch.wv[c] = params.w_w[c] * step.v[c];
    }
    ger(s_n, d, scale, &scratch.d_kc_sm, &scratch.wv, d_key);
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
