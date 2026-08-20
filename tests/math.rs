use candle_core::{Device, Tensor, Var};
use ullis::gauss::mps_safe_solve;
use ullis::kan::TernaryKanLinear;
use ullis::quant::{pack_ternary, ternarize_ste, unpack_ternary};

#[test]
fn ste_grad_hardtanh_gate() {
    let device = Device::Cpu;
    let w = Var::from_tensor(
        &Tensor::from_vec(vec![-1.5f32, -0.1, 0.0, 0.2, 1.2], 5, &device).unwrap(),
    )
    .unwrap();
    let q = ternarize_ste(w.as_tensor(), 0.5).unwrap();
    let loss = q.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let g = grads.get(w.as_tensor()).expect("grad");
    let gv = g.to_vec1::<f32>().unwrap();
    assert_eq!(gv[0], 0.0, "|w|>1 should be gated off");
    assert!((gv[1] - 1.0).abs() < 1e-5);
    assert_eq!(gv[4], 0.0);
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
    let device = Device::Cpu;
    let mut rng = ullis::device::rng_from_seed(0);
    let mut layer = TernaryKanLinear::new(6, 5, 4, false, 1, 0.7, &device, &mut rng).unwrap();
    let x = Tensor::randn(0f32, 1f32, (2, 3, 6), &device).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 5]);
    layer.set_phase(3).unwrap();
    let y2 = layer.forward(&x).unwrap();
    assert_eq!(y2.dims(), &[2, 3, 5]);
    layer.pack().unwrap();
    let y3 = layer.forward(&x).unwrap();
    assert_eq!(y3.dims(), &[2, 3, 5]);
}

#[test]
fn mps_safe_solve_eye() {
    let device = Device::Cpu;
    let a = Tensor::from_vec(vec![2.0f32, 0.5, 0.5, 3.0], (2, 2), &device).unwrap();
    let b = Tensor::from_vec(vec![1.0f32, 0.0, 2.0, 0.0, 1.0, 3.0], (2, 3), &device).unwrap();
    let x = mps_safe_solve(&a, &b).unwrap();
    let recon = a.matmul(&x).unwrap();
    let err = (recon - b)
        .unwrap()
        .abs()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(err < 1e-3, "residual {err}");
}

#[test]
fn extend_grid_preserves_forward() {
    let device = Device::Cpu;
    let mut rng = ullis::device::rng_from_seed(0);
    let mut layer = TernaryKanLinear::new(8, 6, 4, false, 1, 0.7, &device, &mut rng).unwrap();
    let xs: Vec<f32> = (0..24).map(|i| -1.5 + i as f32 * (3.0 / 23.0)).collect();
    let x = Tensor::from_vec(xs, (3, 8), &device).unwrap();
    let y0 = layer.forward(&x).unwrap().detach();
    layer.extend_grid(8).unwrap();
    let y1 = layer.forward(&x).unwrap();
    let e1 = (y0 - y1.detach())
        .unwrap()
        .abs()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(e1 < 0.08, "4->8 drift {e1}");
    layer.extend_grid(12).unwrap();
    let y2 = layer.forward(&x).unwrap();
    let e2 = (y1.detach() - y2.detach())
        .unwrap()
        .abs()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(e2 < 0.08, "8->12 drift {e2}");
    assert_eq!(layer.n_basis, 12);
}

#[test]
fn extend_grid_frozen_after_qat() {
    let device = Device::Cpu;
    let mut rng = ullis::device::rng_from_seed(1);
    let mut layer = TernaryKanLinear::new(4, 4, 4, false, 1, 0.7, &device, &mut rng).unwrap();
    layer.set_phase(3).unwrap();
    assert!(layer.extend_grid(8).is_err());
}

/// Regression: SGD velocity used to keep the full backprop tape (`vel = μ·vel + g`
/// without detach), so RSS grew ~1 GB / 20 steps under G=12 MoE.
#[test]
fn moe_sgd_steps_do_not_retain_graphs() {
    use ullis::config::TrainConfig;
    use ullis::device::synchronize;
    use ullis::model::UllisKan;
    use ullis::optim::SgdMomentum;
    use ullis::telemetry::process_memory_bytes;

    let device = Device::Cpu;
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
    let mut model = UllisKan::new(cfg, &device).unwrap();
    model.set_phase(2).unwrap();
    let mut opt = SgdMomentum::new(model.trainable_vars(2), 1e-3, 0.9, 1.0).unwrap();
    let ids: Vec<u32> = (0..32).map(|i| i % 128).collect();
    let x = Tensor::from_vec(ids.clone(), (2, 16), &device).unwrap();
    let y = Tensor::from_vec(ids, (2, 16), &device).unwrap();

    let step = |opt: &mut SgdMomentum| {
        let logits = model.forward(&x).unwrap();
        let (b, t, v) = logits.dims3().unwrap();
        let loss = candle_nn::loss::cross_entropy(
            &logits.reshape((b * t, v)).unwrap(),
            &y.flatten_all().unwrap(),
        )
        .unwrap();
        let loss = (loss + (model.l1_penalty().unwrap() * 1e-3).unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
        drop(loss);
        drop(grads);
        drop(logits);
        synchronize(&device).unwrap();
    };

    for _ in 0..3 {
        step(&mut opt);
    }
    let rss0 = process_memory_bytes();
    for _ in 0..48 {
        step(&mut opt);
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
