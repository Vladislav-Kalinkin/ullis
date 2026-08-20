use ullis::device::SovereignDevice;
use ullis::gauss::solve_square;
use ullis::kan::TernaryKanLinear;
use ullis::quant::{pack_ternary, ste_gate, unpack_ternary};

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
    for _ in 0..48 {
        step(&mut model, &mut opt);
    }
    let rss1 = process_memory_bytes();
    if rss0 > 0 {
        let growth = rss1.saturating_sub(rss0);
        assert!(
            growth < 80 * 1024 * 1024,
            "optimizer retained graphs: rss grew by {} MB ({} -> {})",
            growth / (1024 * 1024),
            rss0 / (1024 * 1024),
            rss1 / (1024 * 1024)
        );
    }
}
