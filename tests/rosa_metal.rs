#![cfg(target_os = "macos")]

use std::time::Instant;

use ullis::config::TrainConfig;
use ullis::metal::{
    MetalRuntime, ROSA_QKV_1BIT_BWD_E_KERNEL_NAME, ROSA_QKV_1BIT_FWD_KERNEL_NAME,
    validate_metal_kernel,
};
use ullis::metal::MetalDispatchShape;
use ullis::rosa::{
    bit_from_activation, pack_bitplane, qkv_bitplane_bytes, rosa_qkv_batch, rosa_qkv_out_batched,
    rosa_qkv_ref, sam_workspace_bytes,
};

const QKV_Q: [u8; 8] = [0, 1, 1, 0, 1, 1, 0, 0];
const QKV_K: [u8; 8] = [1, 1, 0, 1, 0, 1, 1, 0];
const QKV_V: [u8; 8] = [0, 1, 0, 1, 1, 0, 1, 0];
const QKV_IDX: [u8; 8] = [0, 1, 0, 1, 1, 0, 1, 0];

const ROSA_T5_C3: [[f32; 3]; 5] = [
    [-0.1, 2.0, 0.0],
    [0.4, -4.2, -1.5],
    [1.1, 1.2, 2.5],
    [-3.1, -2.2, 1.5],
    [2.1, -3.2, -2.5],
];

fn runtime() -> Option<MetalRuntime> {
    MetalRuntime::new().ok()
}

fn flatten_t5() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut bits = Vec::with_capacity(15);
    for row in &ROSA_T5_C3 {
        bits.extend(row.iter().copied().map(bit_from_activation));
    }
    (bits.clone(), bits.clone(), bits)
}

fn patterned_bits(batch: usize, time: usize, channels: usize, salt: u8) -> Vec<u8> {
    let mut bits = vec![0_u8; batch * time * channels];
    for b in 0..batch {
        for t in 0..time {
            for c in 0..channels {
                let index = (b * time + t) * channels + c;
                bits[index] = ((t * 13 + c * 7 + b * 3 + usize::from(salt)) >> 2) as u8 & 1;
            }
        }
    }
    bits
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= atol,
            "index {index}: {a} vs {e} (atol {atol})"
        );
    }
}

#[test]
fn metal_rosa_kernel_is_present_and_compiles() {
    let Some(_) = runtime() else {
        return;
    };
    let shape = MetalDispatchShape::new(1, 32, 8).unwrap();
    let width = validate_metal_kernel(ROSA_QKV_1BIT_FWD_KERNEL_NAME, shape).unwrap();
    assert!(width > 0);
    let width = validate_metal_kernel(ROSA_QKV_1BIT_BWD_E_KERNEL_NAME, shape).unwrap();
    assert!(width > 0);
}

#[test]
fn metal_g_e_matches_cpu_including_unmatched() {
    let Some(runtime) = runtime() else {
        return;
    };
    let gy = [0.5_f32, -1.0, 0.25, 2.0, -0.5, 0.0, 1.25, -0.75];
    let idx = [0_u8, 1, 0, 1, 1, 0, 1, 0];
    let gpu = runtime.rosa_qkv_1bit_bwd_e(&gy, &idx, 1, 8, 1).unwrap();
    let expected: f32 = gy
        .iter()
        .zip(idx)
        .map(|(&g, bit)| g * (2.0 * f32::from(bit) - 1.0))
        .sum();
    assert!((gpu[0] - expected).abs() < 1e-5);
}

#[test]
fn metal_qkv_matches_python_fixture() {
    let Some(runtime) = runtime() else {
        return;
    };
    let gpu = runtime
        .rosa_qkv_1bit_fwd(
            &pack_bitplane(&QKV_Q).unwrap(),
            &pack_bitplane(&QKV_K).unwrap(),
            &pack_bitplane(&QKV_V).unwrap(),
            &[0.25],
            1,
            8,
            1,
        )
        .unwrap();
    assert_eq!(gpu.idx, QKV_IDX);
    assert_eq!(gpu.idx, rosa_qkv_ref(&QKV_Q, &QKV_K, &QKV_V).unwrap());
    assert_close(&gpu.out, &[-0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25, -0.25], 2e-3);
}

#[test]
fn metal_qkv_matches_t5_c3_and_collapses_unmatched_to_minus_e() {
    let Some(runtime) = runtime() else {
        return;
    };
    let (q, k, v) = flatten_t5();
    let e = [0.5_f32, 0.25, 1.0];
    let gpu = runtime
        .rosa_qkv_1bit_fwd(
            &pack_bitplane(&q).unwrap(),
            &pack_bitplane(&k).unwrap(),
            &pack_bitplane(&v).unwrap(),
            &e,
            1,
            5,
            3,
        )
        .unwrap();
    let cpu_idx = rosa_qkv_batch(&q, &k, &v, 1, 5, 3).unwrap();
    let cpu_out = rosa_qkv_out_batched(&cpu_idx, &e, 1, 5, 3).unwrap();
    assert_eq!(gpu.idx, cpu_idx);
    assert_close(&gpu.out, &cpu_out, 2e-3);
    assert!(gpu.out.iter().all(|v| v.abs() > 0.1), "1-bit QKV never emits 0");
}

#[test]
fn metal_qkv_is_bit_exact_on_t32_two_batch() {
    let Some(runtime) = runtime() else {
        return;
    };
    let batch = 2;
    let time = 32;
    let channels = 8;
    let q = patterned_bits(batch, time, channels, 1);
    let k = patterned_bits(batch, time, channels, 2);
    let v = patterned_bits(batch, time, channels, 3);
    let e: Vec<f32> = (0..channels).map(|c| 0.125 * (c as f32 + 1.0)).collect();
    let gpu = runtime
        .rosa_qkv_1bit_fwd(
            &pack_bitplane(&q).unwrap(),
            &pack_bitplane(&k).unwrap(),
            &pack_bitplane(&v).unwrap(),
            &e,
            batch,
            time,
            channels,
        )
        .unwrap();
    let cpu_idx = rosa_qkv_batch(&q, &k, &v, batch, time, channels).unwrap();
    let cpu_out = rosa_qkv_out_batched(&cpu_idx, &e, batch, time, channels).unwrap();
    assert_eq!(gpu.idx, cpu_idx);
    assert_close(&gpu.out, &cpu_out, 3e-3);
}

#[test]
fn metal_sam_fwd_t32_d256_under_five_milliseconds() {
    let Some(runtime) = runtime() else {
        return;
    };
    let batch = 1;
    let time = 32;
    let channels = 256;
    let q = patterned_bits(batch, time, channels, 4);
    let k = patterned_bits(batch, time, channels, 5);
    let v = patterned_bits(batch, time, channels, 6);
    let e = vec![0.25_f32; channels];
    let q_bits = pack_bitplane(&q).unwrap();
    let k_bits = pack_bitplane(&k).unwrap();
    let v_bits = pack_bitplane(&v).unwrap();
    let warmup = runtime
        .rosa_qkv_1bit_fwd(&q_bits, &k_bits, &v_bits, &e, batch, time, channels)
        .unwrap();
    let cpu_idx = rosa_qkv_batch(&q, &k, &v, batch, time, channels).unwrap();
    assert_eq!(warmup.idx, cpu_idx);

    let started = Instant::now();
    let gpu = runtime
        .rosa_qkv_1bit_fwd(&q_bits, &k_bits, &v_bits, &e, batch, time, channels)
        .unwrap();
    let elapsed = started.elapsed();
    assert_eq!(gpu.idx, cpu_idx);
    assert!(
        elapsed.as_secs_f64() * 1_000.0 < 5.0,
        "SAM fwd T=32 D=256 took {elapsed:?}, budget is 5 ms"
    );
}

#[test]
fn metal_train_step_stop_grad_runs_and_finite_loss() {
    let Some(runtime) = runtime() else {
        return;
    };
    use ullis::tokenizer::MIN_VOCAB;
    use ullis::{TrainConfig, UllisHeron};
    let cfg = TrainConfig {
        vocab_size: MIN_VOCAB as usize,
        d_model: 16,
        n_layers: 1,
        dim_ffn: 64,
        context_len: 32,
        tmix_lora_rank: 8,
        ..Default::default()
    };
    let mut model = UllisHeron::new(cfg).unwrap();
    let tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 8) as u32).collect();
    let loss = model
        .train_step_metal(&runtime, &tokens, 1, 32, 1e-3)
        .unwrap();
    assert!(loss.next_token.is_finite());
    assert_eq!(loss.next_token_count, 31);
}

#[test]
fn memory_estimate_rosa_fields_match_named_formulas() {
    let cfg = TrainConfig::default();
    let estimate = cfg.memory_estimate().unwrap();
    assert_eq!(estimate.rosa_sam_peak, 40 * cfg.context_len * cfg.d_model);
    assert_eq!(
        estimate.qkv_bitplanes,
        qkv_bitplane_bytes(cfg.batch_size, cfg.context_len, cfg.d_model).unwrap()
    );
    let exact = sam_workspace_bytes(cfg.batch_size, cfg.context_len, cfg.d_model).unwrap();
    assert!(
        exact.abs_diff(estimate.rosa_sam_peak) <= 20 * cfg.d_model,
        "estimate 40TD approximates 20*(2T+1)*D; peak {} exact {exact}",
        estimate.rosa_sam_peak
    );
}
