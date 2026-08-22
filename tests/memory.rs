//! Memory-arch capability suite: circuits before corpus.

use rand::Rng;
use ullis::accelerate::ternarize_row;
use ullis::config::{ModelArch, TrainConfig};
use ullis::device::rng_from_seed;
use ullis::memory::{memory_sgd_step, UllisMemory};
use ullis::mixers::{embed_lookup_into, embed_scatter_acc, streamed_tied_ce_acc};
use ullis::optim::DenseSgd;
use ullis::scan::{scan_forward, ScanParams};
use ullis::slots::{slots_backward, slots_forward, SlotParams, SlotState};

const RECALL: u32 = 1;
const NAME0: u32 = 2;
const VAL0: u32 = 10;
const N_NAMES: u32 = 8;
const N_VALS: u32 = 8;
const V_TASK: usize = 18;

fn name_id(i: u32) -> u32 {
    NAME0 + i
}
fn val_id(i: u32) -> u32 {
    VAL0 + i
}

fn bind_seq(pairs: &[(u32, u32)], query: u32) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let mut ids = Vec::new();
    for &(n, v) in pairs {
        ids.push(name_id(n));
        ids.push(val_id(v));
    }
    ids.push(RECALL);
    ids.push(name_id(query));
    let t = ids.len();
    let mut targets = vec![0u32; t];
    let mut mask = vec![0u8; t];
    if t >= 2 {
        targets[t - 1] = val_id(
            pairs
                .iter()
                .find(|p| p.0 == query)
                .map(|p| p.1)
                .unwrap_or(0),
        );
        mask[t - 1] = 1;
    }
    (ids, targets, mask)
}

struct BindNet {
    embed: Vec<f32>,
    embed_grad: Vec<f32>,
    slots: SlotParams,
    d: usize,
}

impl BindNet {
    fn new(d: usize, s: usize, rng: &mut impl Rng) -> Self {
        let mut embed = ullis::mixers::randn(V_TASK * d, 0.4, rng);
        for id in 0..V_TASK {
            let mark = if id as u32 >= VAL0 { 0.35 } else { -0.35 };
            embed[id * d] = mark;
        }
        Self {
            embed,
            embed_grad: vec![0.0; V_TASK * d],
            slots: SlotParams::new(d, s),
            d,
        }
    }

    fn train_step(&mut self, ids: &[u32], targets: &[u32], mask: &[u8]) -> anyhow::Result<f32> {
        let d = self.d;
        let t = ids.len();
        self.embed_grad.fill(0.0);
        self.slots.zero_grad();
        let mut x = vec![0.0f32; t * d];
        embed_lookup_into(&self.embed, V_TASK, d, ids, &mut x)?;
        let mut st = SlotState::new(1, self.slots.n_slots, d);
        let (sv, tape) = slots_forward(&self.slots, &x, 1, t, &mut st)?;
        let mut dh = vec![0.0f32; t * d];
        let mut row = Vec::new();
        let (loss, _) = streamed_tied_ce_acc(
            &sv,
            &self.embed,
            t,
            d,
            V_TASK,
            targets,
            mask,
            0.0,
            &mut dh,
            &mut self.embed_grad,
            &mut row,
        )?;
        let dv = slots_backward(&mut self.slots, &tape, &dh, 1, t)?;
        embed_scatter_acc(V_TASK, d, ids, &dv, &mut self.embed_grad)?;
        Ok(loss)
    }

    fn sgd(&mut self, opt: &mut DenseSgd) {
        let mut sq = 0.0f32;
        for &g in &self.embed_grad {
            sq += g * g;
        }
        sq += self.slots.grad_gamma.powi(2);
        let scale = opt.clip_scale(sq);
        opt.update_slice(0, &mut self.embed, &self.embed_grad, scale);
        opt.update_slice(
            1,
            std::slice::from_mut(&mut self.slots.gamma),
            std::slice::from_ref(&self.slots.grad_gamma),
            scale,
        );
    }

    fn predict(&self, ids: &[u32]) -> anyhow::Result<u32> {
        let d = self.d;
        let t = ids.len();
        let mut x = vec![0.0f32; t * d];
        embed_lookup_into(&self.embed, V_TASK, d, ids, &mut x)?;
        let mut st = SlotState::new(1, self.slots.n_slots, d);
        let (sv, _) = slots_forward(&self.slots, &x, 1, t, &mut st)?;
        let last = &sv[(t - 1) * d..t * d];
        let mut logits = vec![0.0f32; V_TASK];
        ullis::accelerate::sgemm_nt(1, V_TASK, d, 1.0, last, &self.embed, 0.0, &mut logits)?;
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &z) in logits.iter().enumerate() {
            if z > bv {
                bv = z;
                best = i;
            }
        }
        Ok(best as u32)
    }
}

fn bind_lens(d: usize) -> Vec<usize> {
    vec![V_TASK * d, 1]
}

#[test]
fn c3_bind_untrained_orthogonal() {
    let mut rng = rng_from_seed(0);
    let d = 32usize;
    let mut net = BindNet::new(d, 8, &mut rng);
    net.slots.w_link[0] = 20.0;
    net.slots.b_bk = 3.0;
    net.slots.b_bv = 3.0;
    net.slots.b_alloc = 4.0;
    net.embed.fill(0.0);
    for id in 0..V_TASK {
        net.embed[id * d] = if id as u32 >= VAL0 { 0.35 } else { -0.35 };
        let axis = 1 + (id % (d - 1));
        net.embed[id * d + axis] = 1.0;
    }
    let pairs = [(0u32, 0u32), (1, 1), (2, 2), (3, 3)];
    let mut ok = 0u32;
    for q in 0..4u32 {
        let (ids, targets, _) = bind_seq(&pairs, q);
        let pred = net.predict(&ids).unwrap();
        let want = targets[targets.len() - 1];
        if pred == want {
            ok += 1;
        }
    }
    assert_eq!(ok, 4, "untrained orthogonal bind {ok}/4");
}

#[test]
fn c3_bind_one_layer_slots() {
    let mut rng = rng_from_seed(7);
    let d = 32usize;
    let s = 8usize;
    let mut net = BindNet::new(d, s, &mut rng);
    let mut opt = DenseSgd::new(&bind_lens(d), 5e-2, 0.9, 1.0);
    net.slots.b_link = 0.0;
    net.slots.w_link[0] = 20.0;
    net.slots.b_bk = 2.0;
    net.slots.b_bv = 2.0;
    net.slots.b_alloc = 4.0;
    for step in 0..600 {
        let mut used_n = [false; 8];
        let mut used_v = [false; 8];
        let mut pairs = Vec::new();
        for _ in 0..4 {
            let mut n = rng.random_range(0..N_NAMES);
            while used_n[n as usize] {
                n = rng.random_range(0..N_NAMES);
            }
            used_n[n as usize] = true;
            let mut v = rng.random_range(0..N_VALS);
            while used_v[v as usize] {
                v = rng.random_range(0..N_VALS);
            }
            used_v[v as usize] = true;
            pairs.push((n, v));
        }
        let q = pairs[rng.random_range(0..pairs.len())].0;
        let (ids, targets, mask) = bind_seq(&pairs, q);
        let _ = net.train_step(&ids, &targets, &mask).unwrap();
        net.sgd(&mut opt);
        let _ = step;
    }

    let mut ok = 0u32;
    let mut n = 0u32;
    for _ in 0..40 {
        let mut used = [false; 8];
        let mut pairs = Vec::new();
        let mut names_used = [false; 8];
        for _ in 0..4 {
            let mut nm = rng.random_range(0..N_NAMES);
            while names_used[nm as usize] {
                nm = rng.random_range(0..N_NAMES);
            }
            names_used[nm as usize] = true;
            let mut v = rng.random_range(0..N_VALS);
            while used[v as usize] {
                v = rng.random_range(0..N_VALS);
            }
            used[v as usize] = true;
            pairs.push((nm, v));
        }
        let q = pairs[0].0;
        let (ids, targets, _mask) = bind_seq(&pairs, q);
        let pred = net.predict(&ids[..ids.len()]).unwrap();
        if pred == targets[targets.len() - 1] {
            ok += 1;
        }
        n += 1;
    }
    let acc = ok as f32 / n as f32;
    assert!(
        acc >= 0.90,
        "C3 bind AR exact {acc:.2} ({ok}/{n}) — slot algebra failed"
    );
}

#[allow(clippy::field_reassign_with_default)]
fn memory_cfg(d: usize, layers: usize, e: usize, w: usize, slots: usize, v: usize) -> TrainConfig {
    let mut cfg = TrainConfig::default();
    cfg.arch = ModelArch::Memory;
    cfg.d_model = d;
    cfg.n_layers = layers;
    cfg.mem_experts = e;
    cfg.expert_width = w;
    cfg.n_slots = slots;
    cfg.vocab_size = v;
    cfg.moe_topk = 2;
    cfg.seq_len = 16;
    cfg.batch_size = 2;
    cfg
}

#[test]
fn memory_train_step_finite() {
    let mut rng = rng_from_seed(1);
    let cfg = memory_cfg(32, 1, 2, 16, 8, 32);
    let mut model = UllisMemory::new(cfg.clone(), &mut rng).unwrap();
    let t = cfg.seq_len;
    let b = cfg.batch_size;
    let n = b * t;
    let ids: Vec<u32> = (0..n).map(|i| (i as u32) % 31 + 1).collect();
    let targets: Vec<u32> = (0..n).map(|i| ((i + 3) as u32) % 31 + 1).collect();
    let mask = vec![1u8; n];
    let loss = model.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
    assert!(loss.is_finite(), "loss {loss}");
    let mut opt = DenseSgd::new(&model.param_lens(), 1e-3, 0.9, 1.0);
    memory_sgd_step(&mut model, &mut opt).unwrap();
    let tok = model.generate_last(&ids[..8]).unwrap();
    assert!((tok as usize) < 32);
}

#[test]
fn c1_single_bind_copy() {
    let mut rng = rng_from_seed(3);
    let d = 32usize;
    let mut net = BindNet::new(d, 4, &mut rng);
    net.slots.w_link[0] = 20.0;
    net.slots.b_bk = 2.0;
    net.slots.b_bv = 2.0;
    let mut opt = DenseSgd::new(&bind_lens(d), 5e-2, 0.9, 1.0);
    for _ in 0..300 {
        let n = rng.random_range(0..N_NAMES);
        let v = rng.random_range(0..N_VALS);
        let (ids, targets, mask) = bind_seq(&[(n, v)], n);
        let _ = net.train_step(&ids, &targets, &mask).unwrap();
        net.sgd(&mut opt);
    }
    let mut ok = 0u32;
    let trials = 30u32;
    for _ in 0..trials {
        let n = rng.random_range(0..N_NAMES);
        let v = rng.random_range(0..N_VALS);
        let (ids, targets, _) = bind_seq(&[(n, v)], n);
        let pred = net.predict(&ids).unwrap();
        if pred == targets[targets.len() - 1] {
            ok += 1;
        }
    }
    let acc = ok as f32 / trials as f32;
    assert!(acc >= 0.90, "C1 single-bind copy {acc:.2} ({ok}/{trials})");
}

#[test]
fn c7_expert_count_does_not_explode_layer_ms() {
    let mut rng = rng_from_seed(2);
    let t = 32usize;
    let b = 4usize;
    let n = b * t;
    let ids: Vec<u32> = (0..n).map(|i| (i as u32) % 30 + 1).collect();
    let targets: Vec<u32> = ids.iter().map(|x| (*x + 1) % 31 + 1).collect();
    let mask = vec![1u8; n];

    let mut cfg4 = memory_cfg(32, 1, 4, 16, 0, 32);
    cfg4.seq_len = t;
    cfg4.batch_size = b;
    cfg4.n_slots = 0;
    let mut m4 = UllisMemory::new(cfg4, &mut rng).unwrap();
    let _ = m4.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();

    let mut cfg16 = memory_cfg(32, 1, 16, 16, 0, 32);
    cfg16.seq_len = t;
    cfg16.batch_size = b;
    cfg16.n_slots = 0;
    let mut m16 = UllisMemory::new(cfg16, &mut rng).unwrap();
    let _ = m16.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
    let _ = m16.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
    let ms16 = m16.last_fwd_ms + m16.last_bwd_ms;
    let _ = m4.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
    let ms4 = m4.last_fwd_ms + m4.last_bwd_ms;

    let ratio = ms16 / ms4.max(0.05);
    assert!(
        ratio <= 4.0,
        "C7 layer ms E=4 {ms4:.2} vs E=16 {ms16:.2} ratio {ratio:.2} (expect ~1, hard fail >4)"
    );
}

const OPEN: u32 = 1;
const CLOSE: u32 = 2;
const ASK: u32 = 3;
const MAX_DEPTH: i32 = 8;
const V_BRACK: usize = 13;

fn walk_to_depth(depth: i32, extra_pairs: usize) -> (Vec<u32>, i32) {
    let mut ids = vec![OPEN; depth.max(0) as usize];
    for _ in 0..extra_pairs {
        ids.push(OPEN);
        ids.push(CLOSE);
    }
    ids.push(ASK);
    (ids, depth)
}

struct ScanNet {
    embed: Vec<f32>,
    scan: ScanParams,
    d: usize,
}

impl ScanNet {
    fn new(d: usize) -> Self {
        let mut embed = vec![0.0f32; V_BRACK * d];
        embed[OPEN as usize * d] = 1.0;
        embed[CLOSE as usize * d] = -1.0;
        let mut scan = ScanParams::new(d);
        scan.b_alpha.fill(6.9);
        scan.b_i.fill(1.4);
        Self { embed, scan, d }
    }
}

#[test]
fn c4_integrator_h0_tracks_depth() {
    let net = ScanNet::new(32);
    let mut prev = f32::NEG_INFINITY;
    for depth in 0..=MAX_DEPTH {
        let (ids, _) = walk_to_depth(depth, 0);
        let t = ids.len();
        let mut x = vec![0.0f32; t * net.d];
        embed_lookup_into(&net.embed, V_BRACK, net.d, &ids, &mut x).unwrap();
        let (h, _, _) = scan_forward(&net.scan, &x, 1, t, None).unwrap();
        let h0 = h[(t - 1) * net.d];
        assert!(
            h0 > prev,
            "h[0] should increase with depth (depth {depth} h0={h0} prev={prev})"
        );
        prev = h0;
    }
}

fn scan_h0(net: &ScanNet, ids: &[u32]) -> f32 {
    let t = ids.len();
    let mut x = vec![0.0f32; t * net.d];
    embed_lookup_into(&net.embed, V_BRACK, net.d, ids, &mut x).unwrap();
    let (h, _, _) = scan_forward(&net.scan, &x, 1, t, None).unwrap();
    h[(t - 1) * net.d]
}

#[test]
fn c4_bracket_depth_scan() {
    let mut rng = rng_from_seed(11);
    let net = ScanNet::new(32);
    let n_cls = (MAX_DEPTH + 1) as usize;
    let mut cents = vec![0.0f32; n_cls];
    let mut cnt = vec![0u32; n_cls];
    for depth in 0..=MAX_DEPTH {
        for extra in 0..4 {
            let (ids, _) = walk_to_depth(depth, extra);
            cents[depth as usize] += scan_h0(&net, &ids);
            cnt[depth as usize] += 1;
        }
    }
    for k in 0..n_cls {
        cents[k] /= cnt[k].max(1) as f32;
    }
    let mut ok = 0u32;
    let trials = 45u32;
    for i in 0..trials {
        let depth = (i % (MAX_DEPTH as u32 + 1)) as i32;
        let extra = rng.random_range(0..4);
        let (ids, _) = walk_to_depth(depth, extra);
        let h0 = scan_h0(&net, &ids);
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (k, &c) in cents.iter().enumerate() {
            let d = (h0 - c).abs();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        if best as i32 == depth {
            ok += 1;
        }
    }
    let acc = ok as f32 / trials as f32;
    assert!(
        acc >= 0.90,
        "C4 bracket depth nearest-centroid {acc:.2} ({ok}/{trials})"
    );
}

#[test]
fn c8_width_layer_ms_not_quadratic() {
    let mut rng = rng_from_seed(5);
    let t = 32usize;
    let b = 4usize;
    let n = b * t;
    let v = 32usize;
    let ids: Vec<u32> = (0..n).map(|i| (i as u32) % 30 + 1).collect();
    let targets: Vec<u32> = ids.iter().map(|x| (*x + 1) % 31 + 1).collect();
    let mask = vec![1u8; n];

    let mut time_at = |d: usize| -> f32 {
        let mut cfg = memory_cfg(d, 1, 4, 32, 0, v);
        cfg.seq_len = t;
        cfg.batch_size = b;
        cfg.n_slots = 0;
        let mut m = UllisMemory::new(cfg, &mut rng).unwrap();
        let _ = m.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
        let _ = m.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
        m.last_fwd_ms + m.last_bwd_ms
    };
    let ms64 = time_at(64);
    let ms256 = time_at(256);
    let ratio = ms256 / ms64.max(0.05);
    assert!(
        ratio <= 6.0,
        "C8 D=64 {ms64:.2}ms vs D=256 {ms256:.2}ms ratio {ratio:.2} (fail if >6, quadratic ~16)"
    );
}

#[test]
fn c5_real_beats_shuffled_labels() {
    let mut rng = rng_from_seed(9);
    let v = 8usize;
    let mut cfg = memory_cfg(32, 1, 0, 16, 0, v);
    cfg.seq_len = 12;
    cfg.batch_size = 4;
    cfg.entropy_coef = 0.0;
    let mut model = UllisMemory::new(cfg.clone(), &mut rng).unwrap();
    let mut opt = DenseSgd::new(&model.param_lens(), 2e-2, 0.9, 1.0);
    let t = cfg.seq_len;
    let b = cfg.batch_size;
    let n = b * t;
    let cycle = [1u32, 2, 3, 4];
    for _ in 0..200 {
        let ids: Vec<u32> = (0..n).map(|i| cycle[i % cycle.len()]).collect();
        let targets: Vec<u32> = (0..n).map(|i| cycle[(i + 1) % cycle.len()]).collect();
        let mask = vec![1u8; n];
        let _ = model.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
        memory_sgd_step(&mut model, &mut opt).unwrap();
    }
    let ids: Vec<u32> = (0..n).map(|i| cycle[i % cycle.len()]).collect();
    let real_tgt: Vec<u32> = (0..n).map(|i| cycle[(i + 1) % cycle.len()]).collect();
    let mask = vec![1u8; n];
    let real = model.train_step(&ids, &real_tgt, &mask, b, t, 0.0).unwrap();
    let mut shuf = real_tgt.clone();
    for i in 0..n {
        let j = rng.random_range(0..n);
        shuf.swap(i, j);
    }
    let fake = model.train_step(&ids, &shuf, &mask, b, t, 0.0).unwrap();
    assert!(
        real + 0.30 < fake,
        "C5 real CE {real:.3} should beat shuffle {fake:.3} by ≥ 0.3 nats"
    );
}

#[test]
fn c6_ternary_histogram_not_collapsed() {
    let mut rng = rng_from_seed(6);
    let cfg = memory_cfg(32, 1, 2, 16, 0, 32);
    let model = UllisMemory::new(cfg, &mut rng).unwrap();
    let w = &model.blocks[0].experts[0].w_up;
    let cols = 32usize;
    let rows = w.len() / cols;
    let mut z = 0u32;
    let mut p = 0u32;
    let mut n = 0u32;
    let mut codes = vec![0.0f32; cols];
    for r in 0..rows {
        ternarize_row(&w[r * cols..(r + 1) * cols], 0.7, &mut codes);
        for &c in &codes {
            if c > 0.5 {
                p += 1;
            } else if c < -0.5 {
                n += 1;
            } else {
                z += 1;
            }
        }
    }
    let tot = (z + p + n).max(1) as f32;
    let pz = z as f32 / tot;
    let pp = p as f32 / tot;
    let pn = n as f32 / tot;
    assert!(pz < 0.99, "C6 all-zero collapse {pz:.2}");
    assert!(pp + pn > 0.05, "C6 no ±1 mass p={pp:.2} n={pn:.2}");
}

fn cycle_batch(b: usize, t: usize) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let n = b * t;
    let cycle = [1u32, 2, 3, 4];
    let ids: Vec<u32> = (0..n).map(|i| cycle[i % cycle.len()]).collect();
    let targets: Vec<u32> = (0..n).map(|i| cycle[(i + 1) % cycle.len()]).collect();
    (ids, targets, vec![1u8; n])
}

#[test]
fn qat_then_pack_keeps_c5() {
    let mut rng = rng_from_seed(9);
    let v = 8usize;
    let mut cfg = memory_cfg(32, 1, 2, 16, 0, v);
    cfg.seq_len = 12;
    cfg.batch_size = 4;
    cfg.entropy_coef = 0.0;
    let mut model = UllisMemory::new(cfg.clone(), &mut rng).unwrap();
    let mut opt = DenseSgd::new(&model.param_lens(), 2e-2, 0.9, 1.0);
    let b = cfg.batch_size;
    let t = cfg.seq_len;
    let (ids, targets, mask) = cycle_batch(b, t);
    for _ in 0..120 {
        let _ = model.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
        memory_sgd_step(&mut model, &mut opt).unwrap();
    }
    let real_fp = model.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
    model.set_phase(3);
    opt = DenseSgd::new(&model.param_lens(), 1e-2, 0.9, 1.0);
    for _ in 0..40 {
        let _ = model.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
        memory_sgd_step(&mut model, &mut opt).unwrap();
    }
    let real_qat = model.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
    model.pack();
    let real_pk = model.train_step(&ids, &targets, &mask, b, t, 0.0).unwrap();
    let mut shuf = targets.clone();
    for i in 0..shuf.len() {
        let j = rng.random_range(0..shuf.len());
        shuf.swap(i, j);
    }
    let fake = model.train_step(&ids, &shuf, &mask, b, t, 0.0).unwrap();
    assert!(
        real_pk + 0.20 < fake,
        "QAT+pack C5 real {real_pk:.3} (fp {real_fp:.3} qat {real_qat:.3}) vs shuffle {fake:.3}"
    );
    let codes = model.blocks[0].experts[0]
        .codes_up
        .as_ref()
        .expect("packed codes");
    let n = codes.len() as f32;
    let zeros = codes.iter().filter(|c| c.abs() < 0.5).count() as f32 / n;
    assert!(zeros < 0.99, "packed all-zero {zeros:.2}");
}

fn load_tiny_overfit() -> Vec<ullis::data::ChatRecord> {
    let raw = std::fs::read_to_string("data/tiny-overfit.jsonl").expect("data/tiny-overfit.jsonl");
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            ullis::data::parse_jsonl_line(l).unwrap_or_else(|| panic!("bad jsonl line: {l}"))
        })
        .collect()
}

fn pad_lm(
    ids: &[u32],
    mask: &[u8],
    seq_len: usize,
    eos: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let need = seq_len + 1;
    let mut seq = ids.to_vec();
    let mut m = mask.to_vec();
    while seq.len() < need {
        seq.push(eos);
        m.push(0);
    }
    seq.truncate(need);
    m.truncate(need);
    (
        seq[..seq_len].to_vec(),
        seq[1..].to_vec(),
        m[1..].to_vec(),
    )
}

fn greedy_from_prefix(model: &UllisMemory, tok: &ullis::tokenizer::BpeTokenizer, prefix: &[u32]) -> String {
    let mut caches = model.new_cache();
    let mut hidden = Vec::new();
    for (pos, &id) in prefix.iter().enumerate() {
        hidden = model.feed_token(id, pos, &mut caches).expect("feed");
    }
    let mut out = Vec::new();
    for _ in 0..12 {
        let z = model.logits(&hidden).expect("logits");
        let id = UllisMemory::argmax(&z);
        if id == tok.eos_id || id <= 2 {
            break;
        }
        out.push(id);
        let piece = tok.decode(&[id]);
        if piece.contains('\n') {
            break;
        }
        hidden = model
            .feed_token(id, prefix.len() + out.len() - 1, &mut caches)
            .expect("feed gen");
    }
    tok.decode(&out)
}

fn encode_output_supervised(
    tok: &mut ullis::tokenizer::BpeTokenizer,
    rec: &ullis::data::ChatRecord,
) -> (Vec<u32>, Vec<u8>, usize) {
    use ullis::data::{pack_record, TAG_OUTPUT, TAG_THINKING, TAG_THINK_END};
    let prefix = tok.encode(&pack_record(&rec.system, &rec.user, None, None), false, false);
    let think = tok.encode(
        &format!(
            "{TAG_THINKING}\n{}\n{TAG_THINK_END}\n",
            rec.thinking.trim()
        ),
        false,
        false,
    );
    let mut output = tok.encode(
        &format!("{TAG_OUTPUT}\n{}\n", rec.output.trim()),
        false,
        false,
    );
    output.push(tok.eos_id);
    let tag = tok.encode(&format!("{TAG_OUTPUT}\n"), false, false);
    let tag_len = if output.starts_with(&tag) {
        tag.len()
    } else {
        let mut n = 1usize;
        for i in 1..output.len() {
            let d = tok.decode(&output[..i]);
            let after = d.split(TAG_OUTPUT).nth(1).unwrap_or("");
            if after.chars().all(char::is_whitespace) {
                n = i;
            } else {
                break;
            }
        }
        n
    };
    let n_ctx = prefix.len() + think.len();
    let mut ids = prefix;
    ids.extend(think);
    ids.extend(output);
    let mut mask = vec![0u8; ids.len()];
    for m in mask.iter_mut().skip(n_ctx) {
        *m = 1;
    }
    (ids, mask, n_ctx + tag_len)
}

/// Closed 16-line JSONL must overfit: greedy after `<|output|>` copies the answer.
/// Loss is on the output span only (thinking is context). CE-only is not enough.
#[test]
fn c9_tiny_jsonl_overfit_greedy() {
    use ullis::tokenizer::train_wordpiece;

    let recs = load_tiny_overfit();
    assert!(recs.len() >= 12, "tiny-overfit too small: {}", recs.len());
    let texts: Vec<String> = recs.iter().map(|r| r.pack()).collect();
    let mut tok = train_wordpiece(&texts, 512, 7).expect("tiny tokenizer");
    let v = tok.vocab_size as usize;
    let mut max_len = 0usize;
    let mut packed = Vec::new();
    for rec in &recs {
        let (ids, mask, body_at) = encode_output_supervised(&mut tok, rec);
        max_len = max_len.max(ids.len());
        packed.push((ids, mask, body_at));
    }
    let seq_len = max_len.saturating_sub(1).max(8);
    let b = 4usize;
    let mut rng = rng_from_seed(42);
    let mut cfg = memory_cfg(64, 2, 4, 32, 8, v);
    cfg.seq_len = seq_len;
    cfg.batch_size = b;
    cfg.entropy_coef = 0.0;
    cfg.moe_topk = 2;
    let mut model = UllisMemory::new(cfg, &mut rng).expect("tiny memory model");
    let mut opt = DenseSgd::new(&model.param_lens(), 2e-2, 0.9, 1.0);
    let eos = tok.eos_id;
    let mut last_ce = f32::INFINITY;
    for step in 0..1200 {
        let mut xs = Vec::with_capacity(b * seq_len);
        let mut ys = Vec::with_capacity(b * seq_len);
        let mut ms = Vec::with_capacity(b * seq_len);
        for k in 0..b {
            let (ids, mask, _) = &packed[(step + k) % packed.len()];
            let (x, y, m) = pad_lm(ids, mask, seq_len, eos);
            xs.extend(x);
            ys.extend(y);
            ms.extend(m);
        }
        last_ce = model
            .train_step(&xs, &ys, &ms, b, seq_len, 0.0)
            .expect("train_step");
        memory_sgd_step(&mut model, &mut opt).expect("sgd");
    }
    assert!(
        last_ce < 1.0,
        "tiny overfit CE {last_ce:.3} should drop below 1.0 on the output span"
    );

    let mut tf_ok = 0u32;
    for (ids, _, body_at) in &packed {
        let at = (*body_at).min(ids.len().saturating_sub(1));
        let prefix = &ids[..at];
        let want = ids[at];
        let got = {
            let mut caches = model.new_cache();
            let mut hidden = Vec::new();
            for (pos, &id) in prefix.iter().enumerate() {
                hidden = model.feed_token(id, pos, &mut caches).expect("tf feed");
            }
            UllisMemory::argmax(&model.logits(&hidden).expect("tf logits"))
        };
        if got == want {
            tf_ok += 1;
        }
    }
    assert!(
        tf_ok as usize * 5 >= packed.len() * 4,
        "teacher-forced first output token {tf_ok}/{} (need ≥80%)",
        packed.len()
    );

    let probes = [
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
        ("What is 1+1?", "2"),
        ("What is 7+2?", "9"),
        ("What is 2+3?", "5"),
    ];
    let mut ok = 0u32;
    let mut report = String::new();
    for (user, want) in probes {
        let rec = recs
            .iter()
            .find(|r| r.user == user)
            .unwrap_or_else(|| panic!("missing {user}"));
        let (ids, _mask, body_at) = encode_output_supervised(&mut tok, rec);
        let prefix = &ids[..body_at.min(ids.len())];
        let got = greedy_from_prefix(&model, &tok, prefix);
        let first = got
            .trim()
            .lines()
            .next()
            .unwrap_or("")
            .trim();
        let hit = first == want;
        if hit {
            ok += 1;
        }
        let tail = &prefix[prefix.len().saturating_sub(6)..];
        report.push_str(&format!(
            "  user={user:?} want={want:?} got={got:?} tail={:?} hit={hit}\n",
            tok.decode(tail)
        ));
    }
    assert!(
        ok >= 4,
        "C9 greedy overfit {ok}/{} (CE={last_ce:.3} tf={tf_ok}/{})\n{report}",
        probes.len(),
        packed.len()
    );
}

const COPY_MARK: u32 = 1;
const COPY_ASK: u32 = 2;
const COPY_VAL0: u32 = 3;
const COPY_NVAL: u32 = 12;
const COPY_LEN: usize = 4;
const COPY_V: usize = 16;

fn random_copy_vals(rng: &mut impl Rng) -> Vec<u32> {
    (0..COPY_LEN)
        .map(|_| COPY_VAL0 + rng.random_range(0..COPY_NVAL))
        .collect()
}

fn copy_xy(vals: &[u32], seq_len: usize) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let mut full = Vec::with_capacity(2 + 2 * vals.len() + seq_len);
    full.push(COPY_MARK);
    full.extend_from_slice(vals);
    full.push(COPY_ASK);
    full.extend_from_slice(vals);
    let ask_at = 1 + vals.len();
    let need = seq_len + 1;
    while full.len() < need {
        full.push(0);
    }
    let mut x = vec![0u32; seq_len];
    let mut y = vec![0u32; seq_len];
    let mut m = vec![0u8; seq_len];
    for t in 0..seq_len {
        x[t] = full[t];
        y[t] = full[t + 1];
        if t >= ask_at && t < ask_at + vals.len() {
            m[t] = 1;
        }
    }
    (x, y, m)
}

fn greedy_copy(model: &UllisMemory, vals: &[u32]) -> Vec<u32> {
    let mut ctx = vec![COPY_MARK];
    ctx.extend_from_slice(vals);
    ctx.push(COPY_ASK);
    let mut caches = model.new_cache();
    let mut hidden = Vec::new();
    for (pos, &id) in ctx.iter().enumerate() {
        hidden = model.feed_token(id, pos, &mut caches).expect("copy feed");
    }
    let mut out = Vec::with_capacity(vals.len());
    for _ in 0..vals.len() {
        let z = model.logits(&hidden).expect("copy logits");
        let id = UllisMemory::argmax(&z);
        out.push(id);
        hidden = model
            .feed_token(id, ctx.len() + out.len() - 1, &mut caches)
            .expect("copy gen");
    }
    out
}

/// Multi-token AR copy on a closed alphabet. Single-slot bind (C1) is not
/// this: `hello`→`helle` is a 5-byte sequence copy. Overfit must work before
/// held-out induction is a claim.
#[test]
fn c10_multitoken_copy_overfit() {
    let mut rng = rng_from_seed(13);
    let seq_len = 2 + 2 * COPY_LEN;
    let b = 8usize;
    let mut cfg = memory_cfg(64, 2, 4, 32, 16, COPY_V);
    cfg.seq_len = seq_len;
    cfg.batch_size = b;
    cfg.entropy_coef = 0.0;
    cfg.moe_topk = 2;
    let mut model = UllisMemory::new(cfg, &mut rng).expect("copy model");
    let mut opt = DenseSgd::new(&model.param_lens(), 2e-2, 0.9, 1.0);
    let mut train_set = Vec::new();
    for _ in 0..48 {
        train_set.push(random_copy_vals(&mut rng));
    }
    for step in 0..800 {
        let mut xs = Vec::with_capacity(b * seq_len);
        let mut ys = Vec::with_capacity(b * seq_len);
        let mut ms = Vec::with_capacity(b * seq_len);
        for k in 0..b {
            let vals = &train_set[(step + k) % train_set.len()];
            let (x, y, m) = copy_xy(vals, seq_len);
            xs.extend(x);
            ys.extend(y);
            ms.extend(m);
        }
        let _ = model
            .train_step(&xs, &ys, &ms, b, seq_len, 0.0)
            .expect("copy step");
        memory_sgd_step(&mut model, &mut opt).expect("copy sgd");
    }
    let mut ok = 0u32;
    for vals in &train_set {
        if greedy_copy(&model, vals) == *vals {
            ok += 1;
        }
    }
    let acc = ok as f32 / train_set.len() as f32;
    assert!(
        acc >= 0.70,
        "C10 multi-token copy overfit {acc:.2} ({ok}/{}) — sequence copy failed",
        train_set.len()
    );

    let mut held_ok = 0u32;
    let mut held_n = 0u32;
    let mut pos_ok = [0u32; COPY_LEN];
    for _ in 0..40 {
        let vals = random_copy_vals(&mut rng);
        if train_set.iter().any(|t| t == &vals) {
            continue;
        }
        held_n += 1;
        let got = greedy_copy(&model, &vals);
        if got == vals {
            held_ok += 1;
        }
        for i in 0..COPY_LEN.min(got.len()) {
            if got[i] == vals[i] {
                pos_ok[i] += 1;
            }
        }
    }
    let held_acc = if held_n == 0 {
        0.0
    } else {
        held_ok as f32 / held_n as f32
    };
    // Exact unseen 4-grams are above chance (1/12^4) but this mixer is not
    // an induction head. 10% exact is the current architecture ceiling.
    assert!(
        held_n > 0 && held_ok * 10 >= held_n,
        "C10 held-out copy {held_acc:.2} ({held_ok}/{held_n}) pos={pos_ok:?} after overfit {acc:.2}"
    );
}

/// Local mix is Θ(T W D) with W capped. Doubling T must not look like T×T.
#[test]
fn c11_window_t_not_quadratic() {
    let mut rng = rng_from_seed(4);
    let mut time_at = |t: usize| -> f32 {
        let mut cfg = memory_cfg(32, 1, 2, 16, 0, 32);
        cfg.seq_len = t;
        cfg.batch_size = 2;
        cfg.window = 16;
        cfg.n_slots = 0;
        let n = cfg.batch_size * t;
        let ids: Vec<u32> = (0..n).map(|i| (i as u32) % 30 + 1).collect();
        let targets: Vec<u32> = ids.iter().map(|x| (*x + 1) % 31 + 1).collect();
        let mask = vec![1u8; n];
        let mut m = UllisMemory::new(cfg, &mut rng).unwrap();
        let _ = m.train_step(&ids, &targets, &mask, 2, t, 0.0).unwrap();
        let _ = m.train_step(&ids, &targets, &mask, 2, t, 0.0).unwrap();
        m.last_fwd_ms + m.last_bwd_ms
    };
    let ms32 = time_at(32);
    let ms128 = time_at(128);
    let ratio = ms128 / ms32.max(0.05);
    assert!(
        ratio <= 8.0,
        "C11 T=32 {ms32:.2}ms vs T=128 {ms128:.2}ms ratio {ratio:.2} (linear ~4, quadratic ~16)"
    );
}

#[test]
fn ullis04_roundtrip_generate() {
    use ullis::checkpoint::{load_memory, peek_magic, save_memory, MAGIC_MEM};
    use ullis::tokenizer::train_bpe;
    let mut rng = rng_from_seed(2);
    let mut cfg = memory_cfg(32, 1, 2, 16, 8, 260);
    cfg.seq_len = 8;
    let mut model = UllisMemory::new(cfg, &mut rng).unwrap();
    model.pack();
    let tok = train_bpe(&["fn add(a: i32, b: i32) -> i32 { a + b }".into()], 260, 1).unwrap();
    let dir = std::env::temp_dir().join(format!("ullis04-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("packed.bin");
    save_memory(&path, &model, &tok, 4).unwrap();
    let magic = peek_magic(&path).unwrap();
    assert_eq!(&magic, MAGIC_MEM);
    let loaded = load_memory(&path).unwrap();
    let ids: Vec<u32> = (1..8).collect();
    let a = model.generate_last(&ids).unwrap();
    let mut m2 = loaded.model;
    let b = m2.generate_last(&ids).unwrap();
    assert_eq!(a, b, "roundtrip generate {a} vs {b}");
    let _ = std::fs::remove_dir_all(&dir);
}
