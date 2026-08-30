use ullis::rosa::{
    BITFLIP_TAU, RosaSam, bit_from_activation, exact_bitflip_qkv, pack_bitplane,
    qkv_bitplane_bytes, rosa, rosa_qkv_batch, rosa_qkv_batch_packed, rosa_qkv_out,
    rosa_qkv_out_batched, rosa_qkv_ref, sam_node_count, sam_workspace_bytes,
};

/// T=5 C=3 fixture from `251014_rosa_1bit_layer.py`.
const ROSA_T5_C3: [[f32; 3]; 5] = [
    [-0.1, 2.0, 0.0],
    [0.4, -4.2, -1.5],
    [1.1, 1.2, 2.5],
    [-3.1, -2.2, 1.5],
    [2.1, -3.2, -2.5],
];

const ROSA_Y: [[i32; 5]; 3] = [[-1, -1, 1, 1, 1], [-1, -1, 0, 1, 0], [-1, 0, -1, 1, 1]];

const QKV_Q: [u8; 8] = [0, 1, 1, 0, 1, 1, 0, 0];
const QKV_K: [u8; 8] = [1, 1, 0, 1, 0, 1, 1, 0];
const QKV_V: [u8; 8] = [0, 1, 0, 1, 1, 0, 1, 0];
const QKV_IDX: [u8; 8] = [0, 1, 0, 1, 1, 0, 1, 0];

fn channel_bits(channel: usize) -> Vec<u8> {
    ROSA_T5_C3
        .iter()
        .map(|row| bit_from_activation(row[channel]))
        .collect()
}

#[test]
fn rosa_matches_t5_c3_fixture() {
    for channel in 0..3 {
        let bits = channel_bits(channel);
        assert_eq!(rosa(&bits).unwrap(), ROSA_Y[channel]);
    }
}

#[test]
fn zero_activation_is_bit_zero() {
    assert_eq!(bit_from_activation(0.0), 0);
    assert_eq!(bit_from_activation(-0.0), 0);
    assert_eq!(bit_from_activation(1e-8), 1);
}

#[test]
fn qkv_matches_python_rosa_qkv_ref_fixture() {
    assert_eq!(rosa_qkv_ref(&QKV_Q, &QKV_K, &QKV_V).unwrap(), QKV_IDX);
}

#[test]
fn qkv_output_is_plus_or_minus_e() {
    let e = 0.25;
    let out = rosa_qkv_out(&QKV_IDX, e);
    assert_eq!(out, vec![-e, e, -e, e, e, -e, e, -e]);
}

#[test]
fn incremental_push_matches_one_shot() {
    let one_shot = rosa_qkv_ref(&QKV_Q, &QKV_K, &QKV_V).unwrap();
    let mut sam = RosaSam::with_max_time(QKV_Q.len());
    let incremental: Vec<u8> = QKV_Q
        .iter()
        .zip(QKV_K)
        .zip(QKV_V)
        .map(|((&q, k), v)| sam.push(q, k, v))
        .collect();
    assert_eq!(incremental, one_shot);
}

#[test]
fn reset_reuses_workspace_without_leaking_stream_state() {
    let mut sam = RosaSam::with_max_time(QKV_Q.len());
    let first: Vec<u8> = QKV_Q
        .iter()
        .zip(QKV_K)
        .zip(QKV_V)
        .map(|((&q, k), v)| sam.push(q, k, v))
        .collect();
    sam.reset();
    let second: Vec<u8> = QKV_Q
        .iter()
        .zip(QKV_K)
        .zip(QKV_V)
        .map(|((&q, k), v)| sam.push(q, k, v))
        .collect();
    assert_eq!(first, QKV_IDX);
    assert_eq!(second, QKV_IDX);
}

#[test]
fn qkv_same_streams_on_t5_c3_collapse_unmatched_to_zero() {
    for channel in 0..3 {
        let bits = channel_bits(channel);
        let idx = rosa_qkv_ref(&bits, &bits, &bits).unwrap();
        let expected: Vec<u8> = ROSA_Y[channel].iter().map(|&y| y.max(0) as u8).collect();
        assert_eq!(idx, expected, "channel {channel}");
    }
}

#[test]
fn exact_bitflip_matches_independent_phi() {
    let q = [0.4_f32, -0.2, 1.5, -3.0];
    let k = [0.1_f32, 0.0, -0.5, 2.0];
    let v = [-1.0_f32, 0.3, 0.0, 0.8];
    let gy = [0.5_f32, -1.0, 0.25, 2.0];
    let e = 0.75;
    let q_bits: Vec<u8> = q.iter().copied().map(bit_from_activation).collect();
    let k_bits: Vec<u8> = k.iter().copied().map(bit_from_activation).collect();
    let v_bits: Vec<u8> = v.iter().copied().map(bit_from_activation).collect();
    let (gq, gk, gv) =
        exact_bitflip_qkv(&q_bits, &k_bits, &v_bits, &q, &k, &v, &gy, e, BITFLIP_TAU).unwrap();

    let phi = |idx: &[u8]| -> f32 {
        idx.iter()
            .zip(&gy)
            .map(|(&bit, &g)| g * (2.0 * f32::from(bit) - 1.0) * e)
            .sum()
    };
    for t in 0..4 {
        let mut q1 = q_bits.clone();
        q1[t] = 1;
        let mut q0 = q_bits.clone();
        q0[t] = 0;
        let mag = q[t].abs().max(BITFLIP_TAU);
        let expected = (phi(&rosa_qkv_ref(&q1, &k_bits, &v_bits).unwrap())
            - phi(&rosa_qkv_ref(&q0, &k_bits, &v_bits).unwrap()))
            / (2.0 * mag);
        assert!((gq[t] - expected).abs() < 1e-6, "gq[{t}]");
        let mut k1 = k_bits.clone();
        k1[t] = 1;
        let mut k0 = k_bits.clone();
        k0[t] = 0;
        let mag = k[t].abs().max(BITFLIP_TAU);
        let expected = (phi(&rosa_qkv_ref(&q_bits, &k1, &v_bits).unwrap())
            - phi(&rosa_qkv_ref(&q_bits, &k0, &v_bits).unwrap()))
            / (2.0 * mag);
        assert!((gk[t] - expected).abs() < 1e-6, "gk[{t}]");
        let mut v1 = v_bits.clone();
        v1[t] = 1;
        let mut v0 = v_bits.clone();
        v0[t] = 0;
        let mag = v[t].abs().max(BITFLIP_TAU);
        let expected = (phi(&rosa_qkv_ref(&q_bits, &k_bits, &v1).unwrap())
            - phi(&rosa_qkv_ref(&q_bits, &k_bits, &v0).unwrap()))
            / (2.0 * mag);
        assert!((gv[t] - expected).abs() < 1e-6, "gv[{t}]");
    }
}

#[test]
fn pack_bitplane_matches_lsb_word_layout() {
    let bits = [1_u8, 0, 1, 1, 0, 0, 0, 1];
    assert_eq!(
        pack_bitplane(&bits).unwrap(),
        vec![1 | (1 << 2) | (1 << 3) | (1 << 7)]
    );
}

#[test]
fn batched_qkv_matches_per_channel_ref_and_pm_e() {
    let mut q = Vec::new();
    let mut k = Vec::new();
    let mut v = Vec::new();
    for t in 0..5 {
        q.extend(ROSA_T5_C3[t].iter().map(|&x| bit_from_activation(x)));
        k.extend(ROSA_T5_C3[t].iter().map(|&x| bit_from_activation(x)));
        v.extend(ROSA_T5_C3[t].iter().map(|&x| bit_from_activation(x)));
    }
    let idx = rosa_qkv_batch(&q, &k, &v, 1, 5, 3).unwrap();
    let packed = rosa_qkv_batch_packed(
        &pack_bitplane(&q).unwrap(),
        &pack_bitplane(&k).unwrap(),
        &pack_bitplane(&v).unwrap(),
        1,
        5,
        3,
    )
    .unwrap();
    assert_eq!(idx, packed);
    for channel in 0..3 {
        let expected: Vec<u8> = ROSA_Y[channel].iter().map(|&y| y.max(0) as u8).collect();
        let got: Vec<u8> = (0..5).map(|t| idx[t * 3 + channel]).collect();
        assert_eq!(got, expected, "channel {channel}");
    }
    let e = [0.5_f32, 0.25, 1.0];
    let out = rosa_qkv_out_batched(&idx, &e, 1, 5, 3).unwrap();
    for channel in 0..3 {
        for t in 0..5 {
            let expected = (2.0 * f32::from(idx[t * 3 + channel]) - 1.0) * e[channel];
            assert!((out[t * 3 + channel] - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn sam_workspace_and_qkv_bitplanes_match_memory_estimate_formula() {
    assert_eq!(sam_node_count(2048), 2 * 2048 + 1);
    assert_eq!(
        sam_workspace_bytes(1, 2048, 256).unwrap(),
        5 * 4 * 256 * (2 * 2048 + 1)
    );
    assert_eq!(
        qkv_bitplane_bytes(1, 2048, 256).unwrap(),
        3 * 2048 * 256 / 8
    );
}

#[test]
fn exact_bitflip_rejects_long_sequences() {
    let bits = vec![0_u8; 33];
    let acts = vec![0.0_f32; 33];
    assert!(
        exact_bitflip_qkv(
            &bits,
            &bits,
            &bits,
            &acts,
            &acts,
            &acts,
            &acts,
            1.0,
            BITFLIP_TAU
        )
        .is_err()
    );
}
