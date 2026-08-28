use ullis::model::PackedBinaryLinear;

#[test]
fn forward_uses_signs_not_latents() {
    let linear = PackedBinaryLinear::from_signs(1, 2, &[1, -1], 0.5, true).unwrap();
    let y = linear.forward(&[2.0, 3.0], 1).unwrap();
    // y = 0 + 0.5 * (1*2 + (-1)*3) = -0.5
    assert!((y[0] + 0.5).abs() < 1e-6);
    assert!((linear.latent().get(0) - 0.5).abs() < 1e-5);
    assert!((linear.latent().get(1) + 0.5).abs() < 1e-5);
}

#[test]
fn ste_gradients_match_the_binaryconnect_contract() {
    let linear = PackedBinaryLinear::from_signs(1, 2, &[1, -1], 0.5, true).unwrap();
    let x = [2.0_f32, 3.0];
    let gy = [1.0_f32];
    let mut g_w = [0.0_f32; 2];
    let mut g_x = [0.0_f32; 2];
    let mut g_scale = [0.0_f32; 1];
    let mut g_bias = [0.0_f32; 1];
    linear
        .backward_ste(&x, &gy, 1, &mut g_w, Some(&mut g_x), &mut g_scale, Some(&mut g_bias))
        .unwrap();
    assert!((g_w[0] - 1.0).abs() < 1e-6);
    assert!((g_w[1] - 1.5).abs() < 1e-6);
    assert!((g_x[0] - 0.5).abs() < 1e-6);
    assert!((g_x[1] + 0.5).abs() < 1e-6);
    assert!((g_scale[0] + 1.0).abs() < 1e-6);
    assert!((g_bias[0] - 1.0).abs() < 1e-6);
}

#[test]
fn latent_persists_across_steps_instead_of_reconstructing() {
    let mut linear = PackedBinaryLinear::from_signs(1, 1, &[1], 0.5, false).unwrap();
    let x = [1.0_f32];
    let gy = [1.0_f32];
    let mut g_w = [0.0_f32; 1];
    let mut g_scale = [0.0_f32; 1];
    linear
        .backward_ste(&x, &gy, 1, &mut g_w, None, &mut g_scale, None)
        .unwrap();
    // Hold the row scale fixed so the assertion isolates latent persistence
    // from scale SGD. Magnitude STE on the proxy must not snap back to scale*sign.
    linear
        .apply_clipped_sgd(&g_w, &[0.0], None, 0.1)
        .unwrap();
    let latent_after = linear.latent().get(0);
    let reconstructed = linear.scale().get(0) * linear.sign_at(0);
    assert!(
        (latent_after - reconstructed).abs() > 1e-3,
        "latent {latent_after} snapped back to scale*sign {reconstructed}"
    );

    g_w[0] = 0.0;
    g_scale[0] = 0.0;
    linear
        .backward_ste(&x, &gy, 1, &mut g_w, None, &mut g_scale, None)
        .unwrap();
    linear
        .apply_clipped_sgd(&g_w, &[0.0], None, 0.1)
        .unwrap();
    assert_ne!(linear.latent().get(0), latent_after);
}

#[test]
fn scale_is_learned_not_mean_abs_latent() {
    let mut linear = PackedBinaryLinear::from_signs(1, 2, &[1, -1], 0.5, false).unwrap();
    let x = [1.0_f32, 0.0];
    let gy = [1.0_f32];
    let mut g_w = [0.0_f32; 2];
    let mut g_scale = [0.0_f32; 1];
    linear
        .backward_ste(&x, &gy, 1, &mut g_w, None, &mut g_scale, None)
        .unwrap();
    linear
        .apply_clipped_sgd(&g_w, &g_scale, None, 0.05)
        .unwrap();
    let mean_abs = 0.5
        * (linear.latent().get(0).abs() + linear.latent().get(1).abs());
    assert!(
        (linear.scale().get(0) - mean_abs).abs() > 1e-4,
        "scale collapsed to mean|latent|"
    );
}
