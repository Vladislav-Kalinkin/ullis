use ullis::device::SovereignDevice;
use ullis::gauss::solve_square;
use ullis::kan::TernaryKanLinear;
use ullis::quant::{pack_i8_rows, pack_ternary, ste_gate, unpack_i8_rows, unpack_ternary};

#[test]
fn ste_grad_hardtanh_gate() {
    let w = [-1.5f32, -0.1, 0.0, 0.2, 1.2];
    let g: Vec<f32> = w.iter().copied().map(ste_gate).collect();
    assert_eq!(g[0], 0.0, "|w|>1 should be gated off");
    assert!((g[1] - 1.0).abs() < 1e-5);
    assert_eq!(g[4], 0.0);
}

#[test]
fn pack_roundtrip_tensor() {
    let t = vec![-1i8, 0, 1, 0, 1, 1, -1, 0];
    let p = pack_ternary(&t);
    let u = unpack_ternary(&p, t.len());
    assert_eq!(u, t);
}

#[test]
fn i8_embed_roundtrip() {
    let w: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
    let (codes, scale) = pack_i8_rows(&w, 8, 8);
    let back = unpack_i8_rows(&codes, &scale, 8, 8);
    let err: f32 = w
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(err < 1.0 / 64.0, "max abs err {err}");
}

#[test]
fn layer_shapes_qat_pack() {
    let gpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(0);
    let mut layer = TernaryKanLinear::new(6, 5, 4, false, 1, 0.7, &mut rng).unwrap();
    let x = ullis::mixers::randn(2 * 3 * 6, 1.0, &mut rng);
    let y = layer.forward(&gpu, &x, 6).unwrap();
    assert_eq!(y.len(), 6 * 5);
    layer.set_phase(3).unwrap();
    let y2 = layer.forward(&gpu, &x, 6).unwrap();
    assert_eq!(y2.len(), 6 * 5);
    layer.pack().unwrap();
    let y3 = layer.forward(&gpu, &x, 6).unwrap();
    assert_eq!(y3.len(), 6 * 5);
}

#[test]
fn mps_safe_solve_eye() {
    let a = vec![2.0f32, 0.5, 0.5, 3.0];
    let b = vec![1.0f32, 0.0, 2.0, 0.0, 1.0, 3.0];
    let x = solve_square(&a, 2, &b, 3).unwrap();
    let mut recon = vec![0.0f32; 6];
    ullis::accelerate::sgemm(2, 3, 2, 1.0, &a, &x, 0.0, &mut recon).unwrap();
    let err: f32 = recon
        .iter()
        .zip(b.iter())
        .map(|(u, v)| (u - v).abs())
        .sum::<f32>()
        / 6.0;
    assert!(err < 1e-3, "residual {err}");
}

#[test]
fn extend_grid_preserves_forward() {
    let gpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(0);
    let mut layer = TernaryKanLinear::new(8, 6, 4, false, 1, 0.7, &mut rng).unwrap();
    let xs: Vec<f32> = (0..24).map(|i| -1.5 + i as f32 * (3.0 / 23.0)).collect();
    let y0 = layer.forward(&gpu, &xs, 3).unwrap();
    layer.extend_grid(8).unwrap();
    let y1 = layer.forward(&gpu, &xs, 3).unwrap();
    let e1: f32 = y0
        .iter()
        .zip(y1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / y0.len() as f32;
    assert!(e1 < 0.08, "4->8 drift {e1}");
    layer.extend_grid(12).unwrap();
    let y2 = layer.forward(&gpu, &xs, 3).unwrap();
    let e2: f32 = y1
        .iter()
        .zip(y2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / y1.len() as f32;
    assert!(e2 < 0.08, "8->12 drift {e2}");
    assert_eq!(layer.n_basis, 12);
}

#[test]
fn extend_grid_frozen_after_qat() {
    let mut rng = ullis::device::rng_from_seed(1);
    let mut layer = TernaryKanLinear::new(4, 4, 4, false, 1, 0.7, &mut rng).unwrap();
    layer.set_phase(3).unwrap();
    assert!(layer.extend_grid(8).is_err());
}

#[test]
fn entropy_penalty_polarizes_blended_logits() {
    let v = 4usize;
    let logits = vec![0.4f32, 0.3, 0.2, 0.1];
    let targets = [0u32];
    let mask = [1u8];
    let (ce, d_ce) = ullis::mixers::masked_cross_entropy(&logits, 1, v, &targets, &mask).unwrap();
    let (l_h, h, d_h) =
        ullis::mixers::masked_cross_entropy_entropy(&logits, 1, v, &targets, &mask, 0.5).unwrap();
    assert!(h > 0.0, "blended row must have positive entropy");
    assert!(l_h > ce, "entropy penalty must raise the scalar loss");
    let peak = d_h[0] - d_ce[0];
    let rest: f32 = (1..v).map(|j| d_h[j] - d_ce[j]).sum();
    assert!(
        peak < 0.0,
        "entropy grad should boost the already-largest logit, got {peak}"
    );
    assert!(
        rest > -1e-5,
        "mass on the tail should be pushed down, rest={rest}"
    );
}

#[test]
fn adaptive_insert_is_frozen_after_qat() {
    let mut rng = ullis::device::rng_from_seed(3);
    let mut layer = TernaryKanLinear::new(4, 4, 4, false, 1, 0.7, &mut rng).unwrap();
    layer.set_phase(3).unwrap();
    assert!(layer.insert_knot().is_err());
}

#[test]
fn moe_sgd_steps_do_not_retain_graphs() {
    use ullis::config::TrainConfig;
    use ullis::device::synchronize;
    use ullis::model::UllisKan;
    use ullis::optim::SgdMomentum;
    use ullis::telemetry::process_memory_bytes;

    let gpu = SovereignDevice::open(false).unwrap();
    let cfg = TrainConfig {
        d_model: 16,
        n_layers: 2,
        n_basis: 12,
        grid_start: 12,
        seq_len: 16,
        batch_size: 2,
        vocab_size: 128,
        moe: true,
        n_experts: 3,
        mixer: "shift".into(),
        ..TrainConfig::default()
    };
    let mut model = UllisKan::new(cfg, gpu).unwrap();
    model.set_phase(2).unwrap();
    let mut opt = SgdMomentum::new(&model, 2, 1e-3, 0.9, 1.0).unwrap();
    let ids: Vec<u32> = (0..32).map(|i| i % 128).collect();

    let step = |model: &mut UllisKan, opt: &mut SgdMomentum| {
        let mask = [1u8; 32];
        let _ = model.train_step(&ids, &ids, &mask, 2, 16, 1e-3).unwrap();
        opt.step(model, 2).unwrap();
        synchronize(&model.device).unwrap();
    };

    for _ in 0..3 {
        step(&mut model, &mut opt);
    }
    let rss0 = process_memory_bytes();
    let ws0 = model.workspace_bytes();
    let pb0 = model.trainable_param_bytes(2);
    for _ in 0..48 {
        step(&mut model, &mut opt);
    }
    assert_eq!(
        model.workspace_bytes(),
        ws0,
        "TrainWorkspace must be grow-only and stable after warmup"
    );
    assert_eq!(
        model.trainable_param_bytes(2),
        pb0,
        "trainable bytes must not grow across SGD steps"
    );
    let rss1 = process_memory_bytes();
    if rss0 > 0 {
        let growth = rss1.saturating_sub(rss0);
        // Process RSS is shared with parallel tests in this binary (Metal
        // PSOs). The workspace/param checks above are the leak gate;
        // RSS is a coarse backstop.
        assert!(
            growth < 32 * 1024 * 1024,
            "optimizer retained graphs: rss grew by {} MB ({} -> {})",
            growth / (1024 * 1024),
            rss0 / (1024 * 1024),
            rss1 / (1024 * 1024)
        );
    }
}

fn tiny_train_cfg() -> ullis::config::TrainConfig {
    ullis::config::TrainConfig {
        d_model: 8,
        n_layers: 2,
        n_basis: 4,
        grid_start: 4,
        grid_mid: 4,
        grid_final: 8,
        vocab_size: 32,
        seq_len: 6,
        batch_size: 2,
        mixer: "shift".into(),
        moe: true,
        n_experts: 3,
        seed: 11,
        fused_grad_ckpt: true,
        ..ullis::config::TrainConfig::default()
    }
}

fn snapshot_sgd_oracle(
    model: &mut ullis::model::UllisKan,
    phase: u8,
    lr: f32,
    mu: f32,
    max_norm: f32,
) {
    let snap = model.trainable_snapshot(phase);
    let mut sq = 0.0f32;
    for (_, _, g) in &snap {
        for &v in g {
            sq += v * v;
        }
    }
    let scale = if max_norm > 0.0 {
        let n = sq.sqrt();
        if n > max_norm {
            max_norm / n
        } else {
            1.0
        }
    } else {
        1.0
    };
    for (name, mut data, grad) in snap {
        for j in 0..data.len() {
            let vel = scale * grad[j];
            data[j] -= lr * (mu * 0.0 + vel);
        }
        model.write_param(&name, &data).unwrap();
    }
    model.sync_grids();
}

fn max_abs_params(a: &ullis::model::UllisKan, b: &ullis::model::UllisKan, phase: u8) -> f32 {
    let sa = a.trainable_snapshot(phase);
    let sb = b.trainable_snapshot(phase);
    assert_eq!(sa.len(), sb.len());
    let mut m = 0.0f32;
    for ((na, da, _), (nb, db, _)) in sa.iter().zip(sb.iter()) {
        assert_eq!(na, nb);
        assert_eq!(da.len(), db.len());
        for (x, y) in da.iter().zip(db.iter()) {
            m = m.max((x - y).abs());
        }
    }
    m
}

#[test]
fn inplace_sgd_matches_snapshot_oracle() {
    use ullis::model::UllisKan;
    use ullis::optim::SgdMomentum;

    let cfg = tiny_train_cfg();
    let mut a = UllisKan::new(cfg.clone(), SovereignDevice::open(false).unwrap()).unwrap();
    let mut b = UllisKan::new(cfg, SovereignDevice::open(false).unwrap()).unwrap();
    a.set_phase(1).unwrap();
    b.set_phase(1).unwrap();
    let ids: Vec<u32> = (0..12).map(|i| i % 32).collect();
    let y: Vec<u32> = (1..13).map(|i| i % 32).collect();
    let mask = vec![1u8; 12];
    let la = a.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
    let lb = b.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
    assert!((la - lb).abs() < 1e-6, "loss {la} vs {lb}");

    let i8_before = a.embed_i8.codes.clone();
    let mut opt = SgdMomentum::new(&a, 1, 3e-3, 0.9, 1.0).unwrap();
    opt.step(&mut a, 1).unwrap();
    snapshot_sgd_oracle(&mut b, 1, 3e-3, 0.9, 1.0);
    let err = max_abs_params(&a, &b, 1);
    assert!(err < 1e-6, "in-place vs snapshot max-abs {err}");
    assert_eq!(
        a.embed_i8.codes, i8_before,
        "train SGD must not requantize embed i8"
    );
}

#[test]
fn insert_knot_zeros_vel_then_sgd_matches_oracle() {
    use ullis::model::UllisKan;
    use ullis::optim::SgdMomentum;

    let cfg = tiny_train_cfg();
    let mut a = UllisKan::new(cfg.clone(), SovereignDevice::open(false).unwrap()).unwrap();
    let mut b = UllisKan::new(cfg, SovereignDevice::open(false).unwrap()).unwrap();
    a.set_phase(1).unwrap();
    b.set_phase(1).unwrap();
    let ga = a.insert_knot().unwrap();
    let gb = b.insert_knot().unwrap();
    assert_eq!(ga, gb);
    let mut opt = SgdMomentum::new(&a, 1, 3e-3, 0.9, 1.0).unwrap();
    assert!(opt.vel_bytes() > 0);
    let ids: Vec<u32> = (0..12).map(|i| i % 32).collect();
    let y: Vec<u32> = (1..13).map(|i| i % 32).collect();
    let mask = vec![1u8; 12];
    a.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
    b.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
    opt.step(&mut a, 1).unwrap();
    snapshot_sgd_oracle(&mut b, 1, 3e-3, 0.9, 1.0);
    let err = max_abs_params(&a, &b, 1);
    assert!(err < 1e-6, "post-insert in-place vs oracle {err}");
}

#[test]
fn streamed_tied_ce_acc_matches_allocating_oracle() {
    let n = 3usize;
    let d = 4usize;
    let v = 8usize;
    let hidden: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let embed: Vec<f32> = (0..v * d).map(|i| ((i % 7) as f32) * 0.05 - 0.2).collect();
    let targets = [1u32, 3, 0];
    let mask = [1u8, 0, 1];
    let (loss, h, dh, de) =
        ullis::mixers::streamed_tied_ce(&hidden, &embed, n, d, v, &targets, &mask, 0.03).unwrap();
    let mut dh2 = vec![9.0f32; n * d];
    let mut prior = vec![0.25f32; v * d];
    let mut row = Vec::new();
    let (loss2, h2) = ullis::mixers::streamed_tied_ce_acc(
        &hidden, &embed, n, d, v, &targets, &mask, 0.03, &mut dh2, &mut prior, &mut row,
    )
    .unwrap();
    assert!((loss - loss2).abs() < 1e-6);
    assert!((h - h2).abs() < 1e-6);
    for (a, b) in dh.iter().zip(dh2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    for (g, p) in de.iter().zip(prior.iter()) {
        assert!(
            (0.25 + *g - *p).abs() < 1e-5,
            "must not scale historical embed_grad"
        );
    }
}

#[test]
fn teacher_forced_ce_finite_on_thinking_train_anchor() {
    use ullis::config::TrainConfig;
    use ullis::data::{encode_supervised, parse_jsonl_line};
    use ullis::model::UllisKan;
    use ullis::tokenizer::train_wordpiece;

    let line = std::fs::read_to_string("data/thinking-train.jsonl")
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    let rec = parse_jsonl_line(&line).expect("anchor");
    let mut tok = train_wordpiece(&[], 1024, 7).unwrap();
    let (ids, mask) = encode_supervised(&mut tok, &rec, 32);
    let t = 32.min(ids.len().saturating_sub(1)).max(4);
    let x: Vec<u32> = ids[..t].iter().copied().map(|id| id % 1024).collect();
    let y: Vec<u32> = ids[1..t + 1]
        .iter()
        .copied()
        .map(|id| id % 1024)
        .collect();
    let m = mask[1..t + 1].to_vec();
    let gpu = SovereignDevice::open(false).unwrap();
    let cfg = TrainConfig {
        d_model: 16,
        n_layers: 2,
        n_basis: 4,
        vocab_size: 1024,
        seq_len: t,
        mixer: "shift".into(),
        moe: true,
        moe_topk: 0,
        ..TrainConfig::default()
    };
    let mut model = UllisKan::new(cfg, gpu).unwrap();
    let loss = model.train_step(&x, &y, &m, 1, t, 0.0).unwrap();
    assert!(loss.is_finite(), "teacher-forced CE {loss}");
    assert!(model.last_ce.is_finite());
}

#[test]
fn streamed_tied_ce_chunked_stable_at_wide_v() {
    let n = 4usize;
    let d = 8usize;
    let v = 128usize;
    let hidden: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.02 - 0.3).collect();
    let embed: Vec<f32> = (0..v * d).map(|i| ((i % 11) as f32) * 0.03 - 0.1).collect();
    let targets = [3u32, 17, 0, 90];
    let mask = [1u8, 1, 0, 1];
    let mut dh = vec![0.0f32; n * d];
    let mut de = vec![0.0f32; v * d];
    let mut row = Vec::new();
    let (loss, h) = ullis::mixers::streamed_tied_ce_acc(
        &hidden, &embed, n, d, v, &targets, &mask, 0.03, &mut dh, &mut de, &mut row,
    )
    .unwrap();
    assert!(loss.is_finite() && h.is_finite());
    assert!(dh.iter().any(|x| *x != 0.0));
    let mut dh2 = vec![0.0f32; n * d];
    let mut de2 = vec![0.0f32; v * d];
    let mut row2 = Vec::new();
    let (loss2, h2) = ullis::mixers::streamed_tied_ce_acc(
        &hidden, &embed, n, d, v, &targets, &mask, 0.03, &mut dh2, &mut de2, &mut row2,
    )
    .unwrap();
    assert!((loss - loss2).abs() < 1e-6);
    assert!((h - h2).abs() < 1e-6);
}

#[test]
fn streamed_tied_ce_does_not_floor_at_ln_eps() {
    let n = 1usize;
    let d = 2usize;
    let v = 4usize;
    // Peaked logits: hidden · embed[0] >> others, target is token 3.
    let hidden = vec![50.0f32, 0.0];
    let embed = vec![
        50.0, 0.0, // tok 0
        0.0, 0.0, // tok 1
        0.0, 0.0, // tok 2
        0.0, 0.0, // tok 3 (target)
    ];
    let targets = [3u32];
    let mask = [1u8];
    let mut dh = vec![0.0f32; n * d];
    let mut de = vec![0.0f32; v * d];
    let mut row = Vec::new();
    let (loss, _) = ullis::mixers::streamed_tied_ce_acc(
        &hidden, &embed, n, d, v, &targets, &mask, 0.0, &mut dh, &mut de, &mut row,
    )
    .unwrap();
    let floor = -1e-12f32.ln();
    assert!(
        loss > floor + 1.0,
        "stable CE must exceed −ln(1e-12)={floor}, got {loss}"
    );
    assert!(loss.is_finite());
}

#[test]
fn moe_topk_dense_matches_full_softmax() {
    let gpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(3);
    let mut dense = TernaryKanLinear::new(8, 8, 4, true, 3, 0.7, &mut rng).unwrap();
    let mut rng = ullis::device::rng_from_seed(3);
    let mut tagged = TernaryKanLinear::new(8, 8, 4, true, 3, 0.7, &mut rng).unwrap();
    tagged.moe_topk = 0;
    let x = ullis::mixers::randn(4 * 8, 1.0, &mut ullis::device::rng_from_seed(4));
    let y0 = dense.forward(&gpu, &x, 4).unwrap();
    let y1 = tagged.forward(&gpu, &x, 4).unwrap();
    let err = y0
        .iter()
        .zip(y1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(err == 0.0, "topk=0 must be bit-identical, max|Δ|={err}");
}

#[test]
fn moe_topk_routing_histogram_not_collapsed() {
    use ullis::config::TrainConfig;
    use ullis::model::UllisKan;

    let gpu = SovereignDevice::open(false).unwrap();
    let cfg = TrainConfig {
        d_model: 16,
        n_layers: 2,
        n_basis: 4,
        vocab_size: 64,
        seq_len: 16,
        batch_size: 4,
        mixer: "shift".into(),
        moe: true,
        n_experts: 3,
        moe_topk: 1,
        seed: 11,
        fused_grad_ckpt: true,
        ..TrainConfig::default()
    };
    let mut model = UllisKan::new(cfg, gpu).unwrap();
    model.set_phase(1).unwrap();
    let ids: Vec<u32> = (0..64).map(|i| i % 64).collect();
    let _ = model.forward(&ids, 4, 16).unwrap();
    for (li, blk) in model.blocks.iter().enumerate() {
        let fr = blk.ff.route_fractions();
        assert_eq!(fr.len(), 3);
        for (e, f) in fr.iter().enumerate() {
            assert!(*f <= 0.95, "layer {li} expert {e} collapsed: fraction {f}");
        }
        let hits: u32 = blk.ff.last_route_hits.iter().sum();
        assert_eq!(hits, blk.ff.last_route_tokens);
    }
}

#[test]
fn moe_topk1_forward_finite() {
    let gpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(8);
    let mut layer = TernaryKanLinear::new(8, 8, 4, true, 3, 0.7, &mut rng).unwrap();
    layer.moe_topk = 1;
    let x = ullis::mixers::randn(6 * 8, 1.0, &mut rng);
    let y = layer.forward(&gpu, &x, 6).unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
    let fr = layer.route_fractions();
    for f in &fr {
        assert!(*f <= 1.0);
    }
    let hits: u32 = layer.last_route_hits.iter().sum();
    assert_eq!(hits, layer.last_route_tokens);
}

fn snap_kan_to_f16(model: &mut ullis::model::UllisKan) {
    for b in &mut model.blocks {
        for t in [
            &mut b.ff.weight_base,
            &mut b.ff.weight_shared,
            &mut b.ff.weight_routed,
            &mut b.ff.router,
        ]
        .into_iter()
        .flatten()
        {
            ullis::quant::quantize_f16_in_place(t.as_mut_slice());
        }
    }
}

#[test]
fn fp16_master_matches_snapped_fp32_tape() {
    use ullis::config::{MasterDtype, TrainConfig};
    use ullis::model::UllisKan;

    let make = |master: MasterDtype| {
        let gpu = SovereignDevice::open(false).unwrap();
        let cfg = TrainConfig {
            d_model: 8,
            n_layers: 2,
            n_basis: 4,
            vocab_size: 32,
            seq_len: 6,
            mixer: "shift".into(),
            moe: true,
            n_experts: 3,
            seed: 9,
            fused_grad_ckpt: true,
            master,
            ..TrainConfig::default()
        };
        UllisKan::new(cfg, gpu).unwrap()
    };
    let mut fp32 = make(MasterDtype::Fp32);
    let mut fp16 = make(MasterDtype::Fp16);
    snap_kan_to_f16(&mut fp32);
    fp32.set_phase(1).unwrap();
    fp16.set_phase(1).unwrap();
    let ids: Vec<u32> = (0..12).map(|i| i % 32).collect();
    let y: Vec<u32> = (1..13).map(|i| i % 32).collect();
    let mask = vec![1u8; 12];
    let la = fp32.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
    let lb = fp16.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
    assert!((la - lb).abs() < 1e-4, "loss {la} vs {lb}");
    let sa = fp32.trainable_snapshot(1);
    let sb = fp16.trainable_snapshot(1);
    assert_eq!(sa.len(), sb.len());
    let mut m = 0.0f32;
    for ((na, _, ga), (nb, _, gb)) in sa.iter().zip(sb.iter()) {
        assert_eq!(na, nb);
        for (x, y) in ga.iter().zip(gb.iter()) {
            m = m.max((x - y).abs());
        }
    }
    assert!(m < 1e-4, "fp16 master vs snapped fp32 max|Δgrad|={m}");
}

#[test]
fn metal_fp16_master_matches_cpu_fp16() {
    let metal = SovereignDevice::open(true).unwrap();
    if !metal.is_metal() {
        return;
    }
    let cpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(11);
    let mut layer_cpu = TernaryKanLinear::new(16, 16, 4, true, 3, 0.7, &mut rng).unwrap();
    let mut rng = ullis::device::rng_from_seed(11);
    let mut layer_gpu = TernaryKanLinear::new(16, 16, 4, true, 3, 0.7, &mut rng).unwrap();
    layer_cpu.enable_fp16_master();
    layer_gpu.enable_fp16_master();
    let x = ullis::mixers::randn(4 * 16, 1.0, &mut ullis::device::rng_from_seed(12));
    let y_cpu = layer_cpu.forward(&cpu, &x, 4).unwrap();
    let y_gpu = layer_gpu.forward(&metal, &x, 4).unwrap();
    let max = y_cpu
        .iter()
        .zip(y_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 2e-2, "metal half vs cpu fp16 max|Δ|={max}");
}

#[test]
fn q8_mom_updates_params() {
    use ullis::config::{MomDtype, TrainConfig};
    use ullis::model::UllisKan;
    use ullis::optim::SgdMomentum;

    let gpu = SovereignDevice::open(false).unwrap();
    let cfg = TrainConfig {
        d_model: 8,
        n_layers: 2,
        n_basis: 4,
        vocab_size: 32,
        seq_len: 6,
        mixer: "shift".into(),
        moe: true,
        n_experts: 3,
        seed: 4,
        mom: MomDtype::Q8,
        ..TrainConfig::default()
    };
    let mut model = UllisKan::new(cfg, gpu).unwrap();
    model.set_phase(1).unwrap();
    let before = model.trainable_snapshot(1);
    let ids: Vec<u32> = (0..12).map(|i| i % 32).collect();
    let y: Vec<u32> = (1..13).map(|i| i % 32).collect();
    let mask = vec![1u8; 12];
    model.train_step(&ids, &y, &mask, 2, 6, 0.0).unwrap();
    let mut opt = SgdMomentum::new(&model, 1, 3e-3, 0.9, 1.0).unwrap();
    opt.step(&mut model, 1).unwrap();
    let after = model.trainable_snapshot(1);
    let mut changed = false;
    for ((_, da, _), (_, db, _)) in before.iter().zip(after.iter()) {
        for (x, y) in da.iter().zip(db.iter()) {
            if (x - y).abs() > 1e-12 {
                changed = true;
            }
        }
    }
    assert!(changed, "q8 momentum must update weights");
}

#[test]
fn fused_forward_d512_cpu() {
    let gpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(5);
    let mut layer = TernaryKanLinear::new(512, 64, 4, true, 3, 0.7, &mut rng).unwrap();
    let x = ullis::mixers::randn(2 * 512, 1.0, &mut rng);
    let y = layer.forward(&gpu, &x, 2).unwrap();
    assert_eq!(y.len(), 2 * 64);
    assert!(y.iter().all(|v| v.is_finite()));
    assert!(y.iter().any(|v| *v != 0.0));
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn layer_bwd_pair(
    gpu_host: &SovereignDevice,
    gpu_fused: &SovereignDevice,
    in_f: usize,
    out_f: usize,
    n: usize,
    moe: bool,
    phase: u8,
) -> f32 {
    layer_bwd_pair_factor(
        gpu_host,
        gpu_fused,
        in_f,
        out_f,
        n,
        moe,
        phase,
        ullis::KanFactor::None,
    )
}

fn layer_bwd_pair_factor(
    gpu_host: &SovereignDevice,
    gpu_fused: &SovereignDevice,
    in_f: usize,
    out_f: usize,
    n: usize,
    moe: bool,
    phase: u8,
    factor: ullis::KanFactor,
) -> f32 {
    let mut rng = ullis::device::rng_from_seed(21);
    let mut host = TernaryKanLinear::new(in_f, out_f, 4, moe, 3, 0.7, &mut rng).unwrap();
    let mut rng = ullis::device::rng_from_seed(21);
    let mut fused = TernaryKanLinear::new(in_f, out_f, 4, moe, 3, 0.7, &mut rng).unwrap();
    if factor != ullis::KanFactor::None {
        let mut rng = ullis::device::rng_from_seed(23);
        host.apply_kan_factor(factor, &mut rng).unwrap();
        let mut rng = ullis::device::rng_from_seed(23);
        fused.apply_kan_factor(factor, &mut rng).unwrap();
    }
    host.set_phase(phase).unwrap();
    fused.set_phase(phase).unwrap();
    let x = ullis::mixers::randn(n * in_f, 1.0, &mut ullis::device::rng_from_seed(22));
    let _ = host.forward(gpu_host, &x, n).unwrap();
    let _ = fused.forward(gpu_fused, &x, n).unwrap();
    let dy: Vec<f32> = (0..n * out_f)
        .map(|i| 0.01 * ((i % 5) as f32 - 2.0))
        .collect();
    let dx_h = host
        .backward(&x, &dy, n, ullis::kan::KanEvalMode::Full)
        .unwrap();
    let mut dx_f = vec![0.0f32; n * in_f];
    let mut xt = None;
    let mut dyt = None;
    let mut part = None;
    fused
        .backward_fused(
            gpu_fused,
            &x,
            &dy,
            n,
            ullis::kan::KanEvalMode::Full,
            &mut dx_f,
            &mut xt,
            &mut dyt,
            &mut part,
        )
        .unwrap();
    let mut m = max_abs(&dx_h, &dx_f);
    m = m.max(max_abs(&host.grad_base, &fused.grad_base));
    m = m.max(max_abs(&host.grad_shared, &fused.grad_shared));
    m = m.max(max_abs(&host.grad_routed, &fused.grad_routed));
    m = m.max(max_abs(&host.grad_router, &fused.grad_router));
    m = m.max(max_abs(&host.grad_centers, &fused.grad_centers));
    m = m.max(max_abs(&host.grad_scale_base, &fused.grad_scale_base));
    m = m.max(max_abs(&host.grad_scale_shared, &fused.grad_scale_shared));
    m = m.max(max_abs(&host.grad_scale_routed, &fused.grad_scale_routed));
    m
}

#[test]
fn fused_bwd_cpu_matches_host_d32() {
    let cpu = SovereignDevice::open(false).unwrap();
    let err = layer_bwd_pair(&cpu, &cpu, 32, 32, 8, true, 1);
    assert!(err < 1e-4, "cpu fused vs host d=32 phase1 max|Δ|={err}");
    let err = layer_bwd_pair(&cpu, &cpu, 32, 32, 8, true, 3);
    assert!(err < 1e-4, "cpu fused vs host d=32 qat max|Δ|={err}");
}

#[test]
fn fused_bwd_cpu_matches_host_d512() {
    let cpu = SovereignDevice::open(false).unwrap();
    let err = layer_bwd_pair(&cpu, &cpu, 512, 64, 2, true, 1);
    assert!(err < 1e-4, "cpu fused vs host d=512 max|Δ|={err}");
}

#[test]
fn fused_bwd_metal_matches_cpu_d32() {
    let metal = SovereignDevice::open(true).unwrap();
    if !metal.is_metal() {
        return;
    }
    let cpu = SovereignDevice::open(false).unwrap();
    let err = layer_bwd_pair(&cpu, &metal, 32, 32, 8, true, 1);
    assert!(err < 1e-4, "metal fused vs host d=32 max|Δ|={err}");
    let err = layer_bwd_pair(&cpu, &metal, 32, 32, 8, true, 3);
    assert!(err < 1e-4, "metal fused vs host d=32 qat max|Δ|={err}");
}

#[test]
fn fused_bwd_metal_matches_cpu_d512() {
    let metal = SovereignDevice::open(true).unwrap();
    if !metal.is_metal() {
        return;
    }
    let cpu = SovereignDevice::open(false).unwrap();
    let err = layer_bwd_pair(&cpu, &metal, 512, 64, 2, true, 1);
    assert!(err < 1e-4, "metal fused vs host d=512 max|Δ|={err}");
}

#[test]
fn fused_forward_d512_metal_matches_cpu() {
    let metal = SovereignDevice::open(true).unwrap();
    if !metal.is_metal() {
        return;
    }
    let cpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(6);
    let mut layer = TernaryKanLinear::new(512, 64, 4, true, 3, 0.7, &mut rng).unwrap();
    let x = ullis::mixers::randn(2 * 512, 1.0, &mut rng);
    let y_cpu = layer.forward(&cpu, &x, 2).unwrap();
    let y_gpu = layer.forward(&metal, &x, 2).unwrap();
    let max = y_cpu
        .iter()
        .zip(y_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-4, "layer d=512 metal vs cpu max|Δ|={max}");
}

#[test]
fn shared_edge_has_fewer_spline_params() {
    let mut rng = ullis::device::rng_from_seed(1);
    let layer = TernaryKanLinear::new(8, 8, 4, true, 3, 0.7, &mut rng).unwrap();
    let es = layer.weight_shared.as_ref().unwrap().numel();
    let er = layer.weight_routed.as_ref().unwrap().numel();
    assert_eq!(es, 8 * layer.n_shared);
    assert_eq!(er, 3 * 8 * layer.n_routed);
    assert!(es < 8 * 8 * layer.n_shared);
    assert!(er < 3 * 8 * 8 * layer.n_routed);
}

#[test]
fn shared_edge_forward_finite_and_pack() {
    let gpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(2);
    let mut layer = TernaryKanLinear::new(8, 8, 4, true, 3, 0.7, &mut rng).unwrap();
    layer
        .apply_kan_factor(ullis::KanFactor::SharedEdge, &mut rng)
        .unwrap();
    let x = ullis::mixers::randn(4 * 8, 1.0, &mut rng);
    let y = layer.forward(&gpu, &x, 4).unwrap();
    assert_eq!(y.len(), 4 * 8);
    assert!(y.iter().all(|v| v.is_finite()));
    layer.set_phase(3).unwrap();
    let y2 = layer.forward(&gpu, &x, 4).unwrap();
    assert!(y2.iter().all(|v| v.is_finite()));
    layer.pack().unwrap();
    let y3 = layer.forward(&gpu, &x, 4).unwrap();
    assert_eq!(y3.len(), y2.len());
    assert!(y3.iter().all(|v| v.is_finite()));
}

#[test]
fn shared_edge_insert_knot_preserves_shape() {
    let gpu = SovereignDevice::open(false).unwrap();
    let mut rng = ullis::device::rng_from_seed(3);
    let mut layer = TernaryKanLinear::new(6, 6, 4, true, 3, 0.7, &mut rng).unwrap();
    layer
        .apply_kan_factor(ullis::KanFactor::SharedEdge, &mut rng)
        .unwrap();
    let x = ullis::mixers::randn(2 * 6, 1.0, &mut rng);
    let y0 = layer.forward(&gpu, &x, 2).unwrap();
    layer.knot_energy = vec![0.1, 2.0, 2.5, 0.1];
    let g = layer.insert_knot().unwrap();
    assert_eq!(g, 5);
    let y1 = layer.forward(&gpu, &x, 2).unwrap();
    assert_eq!(y0.len(), y1.len());
    assert!(y1.iter().all(|v| v.is_finite()));
    assert_eq!(
        layer.weight_shared.as_ref().unwrap().numel(),
        6 * layer.n_shared
    );
}

#[test]
fn shared_edge_fused_bwd_matches_host() {
    let cpu = SovereignDevice::open(false).unwrap();
    let err = layer_bwd_pair_factor(&cpu, &cpu, 16, 16, 8, true, 1, ullis::KanFactor::SharedEdge);
    assert!(
        err < 1e-4,
        "shared-edge cpu fused vs host phase1 max|Δ|={err}"
    );
    let err = layer_bwd_pair_factor(&cpu, &cpu, 16, 16, 8, true, 3, ullis::KanFactor::SharedEdge);
    assert!(err < 1e-4, "shared-edge cpu fused vs host qat max|Δ|={err}");
}

#[test]
fn shared_edge_metal_matches_cpu() {
    let metal = SovereignDevice::open(true).unwrap();
    if !metal.is_metal() {
        return;
    }
    let cpu = SovereignDevice::open(false).unwrap();
    let err = layer_bwd_pair_factor(
        &cpu,
        &metal,
        16,
        16,
        8,
        true,
        1,
        ullis::KanFactor::SharedEdge,
    );
    assert!(err < 1e-4, "shared-edge metal vs host max|Δ|={err}");
    let mut rng = ullis::device::rng_from_seed(9);
    let mut layer = TernaryKanLinear::new(16, 16, 4, true, 3, 0.7, &mut rng).unwrap();
    layer
        .apply_kan_factor(ullis::KanFactor::SharedEdge, &mut rng)
        .unwrap();
    let x = ullis::mixers::randn(4 * 16, 1.0, &mut rng);
    let y_cpu = layer.forward(&cpu, &x, 4).unwrap();
    let y_gpu = layer.forward(&metal, &x, 4).unwrap();
    let max = y_cpu
        .iter()
        .zip(y_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-4, "shared-edge fwd metal vs cpu max|Δ|={max}");
}

#[test]
fn shared_edge_is_default_layout() {
    let spec = ullis::MobKanSpec::new(2, 4, 3, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
    assert_eq!(spec.kan_factor, 1);
    assert_eq!(spec.w_shared_len(), 4 * 3);
    assert_eq!(spec.scale_shared_len(), 4);
}
