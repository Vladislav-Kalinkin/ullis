use ullis::model::{Fp16Linear, LAYER_NORM_EPS, LayerNorm, PackedBinaryLinear, RwkvCMixX070};

#[test]
fn seeded_cmix_value_is_kaiming_not_zero() {
    let ffn = RwkvCMixX070::seeded(8, 32, 7).unwrap();
    let n = 8 * 32;
    let mut sum_sq = 0.0_f32;
    for i in 0..n {
        let v = ffn.value.weight().get(i);
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32).sqrt();
    assert!(
        rms > 0.05,
        "CMix value was zero-initialized; key STE cannot train, rms={rms}"
    );
}

#[test]
fn layer_norm_matches_population_moments() {
    let ln = LayerNorm::new(3);
    let x = [1.0_f32, 2.0, 3.0];
    let y = ln.forward(&x, 1).unwrap();
    let mean = 2.0;
    let var = 2.0 / 3.0;
    let inv = (var + LAYER_NORM_EPS).sqrt().recip();
    for (actual, src) in y.iter().zip(x) {
        let expected = (src - mean) * inv;
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn cmix_with_zero_shift_is_relu2_then_value() {
    let ffn = RwkvCMixX070::from_parts_for_test(
        [0.0, 0.0],
        PackedBinaryLinear::from_signs(2, 2, &[1, -1, -1, 1], 1.0, false).unwrap(),
        Fp16Linear::from_f32(2, 2, &[1.0, 0.0, 0.0, 1.0]).unwrap(),
    );
    // x: t0=(1, 0), t1=(0, 1)
    let x = [1.0_f32, 0.0, 0.0, 1.0];
    // x_k = 0 ⇒ k_in = x
    // key signs [[+1,-1],[-1,+1]] scale 1
    // t0: [1*1 + (-1)*0, (-1)*1 + 1*0] = [1, -1] → relu² [1, 0]
    // t1: [1*0 + (-1)*1, (-1)*0 + 1*1] = [-1, 1] → relu² [0, 1]
    // value = I ⇒ y t0=(1,0) t1=(0,1)
    let y = ffn.forward(&x, 1, 2).unwrap();
    assert!((y[0] - 1.0).abs() < 1e-5);
    assert!(y[1].abs() < 1e-5);
    assert!(y[2].abs() < 1e-5);
    assert!((y[3] - 1.0).abs() < 1e-5);
}

#[test]
fn cmix_time_shift_uses_previous_token() {
    let ffn = RwkvCMixX070::from_parts_for_test(
        [1.0, 0.0],
        PackedBinaryLinear::from_signs(1, 2, &[1, 1], 1.0, false).unwrap(),
        Fp16Linear::from_f32(2, 1, &[1.0, 0.0]).unwrap(),
    );
    // D=2, dim_ffn=1. x t0=(2,0) t1=(0,0)
    // xx t0 = -x0 = (-2, 0); k_in0 = (2,0)+(-2,0)*(1,0) wait x_k=(1,0)
    // k_in0 = (2,0) + (-2,0)*(1,0) = (2,0)+(-2,0)=(0,0)
    // xx t1 = x0-x1 = (2,0); k_in1 = (0,0)+(2,0)*(1,0)=(2,0)
    // key all +1: k0=0, k1=2; relu² = 0, 4
    // value W = [1, 0]^T shape [out=2, in=1] = [1, 0] so y = k * first column
    let x = [2.0_f32, 0.0, 0.0, 0.0];
    let y = ffn.forward(&x, 1, 2).unwrap();
    assert!(y[0].abs() < 1e-5 && y[1].abs() < 1e-5);
    assert!((y[2] - 4.0).abs() < 1e-4);
    assert!(y[3].abs() < 1e-5);
}
