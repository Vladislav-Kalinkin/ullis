use ullis::wkv7::{CHUNK_LEN, HEAD_SIZE, fixture_t16_h1, wkv7_backward, wkv7_forward, wkv7_step};

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn fixture_t16_h1_is_checked_in() {
    let ([w, q, k, v, a, b], batch, time, heads) = fixture_t16_h1();
    assert_eq!(time, CHUNK_LEN);
    assert_eq!(heads, 1);
    let fwd = wkv7_forward(&w, &q, &k, &v, &a, &b, batch, time, heads).unwrap();
    assert_eq!(fwd.y.len(), time * HEAD_SIZE);
    assert_eq!(fwd.s.len(), HEAD_SIZE * HEAD_SIZE);
    assert!(
        fwd.y
            .iter()
            .chain(&fwd.s)
            .chain(&fwd.sa)
            .all(|x| x.is_finite())
    );
    // Snapshot taken from this CPU transcription (not live CUDA).
    assert!((fwd.y[0] - 3.780_991_6e-2).abs() < 1e-5);
    assert!((fwd.y[15] - 1.216_479_8e-2).abs() < 1e-5);
    assert!((fwd.y[255] - 4.352_594_6e-2).abs() < 1e-5);
    let y_sum: f32 = fwd.y.iter().sum();
    assert!((y_sum + 0.350_610_94).abs() < 1e-4);
}

#[test]
fn step_matches_chunk_forward() {
    let ([w, q, k, v, a, b], _, time, heads) = fixture_t16_h1();
    let fwd = wkv7_forward(&w, &q, &k, &v, &a, &b, 1, time, heads).unwrap();
    let mut state = vec![0.0; HEAD_SIZE * HEAD_SIZE];
    let mut y = Vec::new();
    for t in 0..time {
        let range = t * HEAD_SIZE..(t + 1) * HEAD_SIZE;
        let yt = wkv7_step(
            &w[range.clone()],
            &q[range.clone()],
            &k[range.clone()],
            &v[range.clone()],
            &a[range.clone()],
            &b[range.clone()],
            &mut state,
            heads,
        )
        .unwrap();
        y.extend_from_slice(&yt);
    }
    assert!(max_abs(&y, &fwd.y) < 1e-5);
}

#[test]
fn backward_runs_and_stays_finite() {
    let ([w, q, k, v, a, b], batch, time, heads) = fixture_t16_h1();
    let fwd = wkv7_forward(&w, &q, &k, &v, &a, &b, batch, time, heads).unwrap();
    let dy: Vec<f32> = (0..fwd.y.len()).map(|i| (i as f32 * 0.01).sin()).collect();
    let bwd = wkv7_backward(
        &w, &q, &k, &v, &a, &b, &dy, &fwd.s, &fwd.sa, batch, time, heads,
    )
    .unwrap();
    assert!(
        bwd.dw
            .iter()
            .chain(&bwd.dq)
            .chain(&bwd.dk)
            .chain(&bwd.dv)
            .chain(&bwd.da)
            .chain(&bwd.db)
            .all(|x| x.is_finite())
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_matches_cpu_on_fixture() {
    let Ok(runtime) = ullis::metal::MetalRuntime::new() else {
        return;
    };
    let ([w, q, k, v, a, b], batch, time, heads) = fixture_t16_h1();
    let cpu = wkv7_forward(&w, &q, &k, &v, &a, &b, batch, time, heads).unwrap();
    let gpu = runtime
        .wkv7_forward(&w, &q, &k, &v, &a, &b, batch, time, heads)
        .unwrap();
    assert!(
        max_abs(&cpu.y, &gpu.y) < 2e-4,
        "y {}",
        max_abs(&cpu.y, &gpu.y)
    );
    assert!(
        max_abs(&cpu.s, &gpu.s) < 2e-4,
        "s {}",
        max_abs(&cpu.s, &gpu.s)
    );
    assert!(max_abs(&cpu.sa, &gpu.sa) < 2e-4);
    let dy: Vec<f32> = (0..cpu.y.len()).map(|i| (i as f32 * 0.01).sin()).collect();
    let cpu_b = wkv7_backward(
        &w, &q, &k, &v, &a, &b, &dy, &cpu.s, &cpu.sa, batch, time, heads,
    )
    .unwrap();
    let gpu_b = runtime
        .wkv7_backward(
            &w, &q, &k, &v, &a, &b, &dy, &cpu.s, &cpu.sa, batch, time, heads,
        )
        .unwrap();
    // dq is a rewind of the transposed state; FP32 accumulates ~1e-3 over the chunk.
    assert!(
        max_abs(&cpu_b.dq, &gpu_b.dq) < 5e-3,
        "dq {}",
        max_abs(&cpu_b.dq, &gpu_b.dq)
    );
    assert!(max_abs(&cpu_b.dv, &gpu_b.dv) < 2e-3);
    assert!(max_abs(&cpu_b.dw, &gpu_b.dw) < 2e-3);
    assert!(max_abs(&cpu_b.da, &gpu_b.da) < 2e-3);
}
