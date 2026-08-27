use ullis::config::{Architecture, TrainConfig};
use ullis::tokenizer::MIN_VOCAB;
use ullis::UllisHeron;

fn reverse_cfg() -> TrainConfig {
    TrainConfig {
        architecture: Architecture::RosaRwkv7,
        d_model: 32,
        n_layers: 2,
        vocab_size: 12,
        context_len: 144,
        batch_size: 1,
        dim_ffn: 128,
        tmix_lora_rank: 8,
        ..Default::default()
    }
}

fn get_randint(digits: usize, rng: &mut u64) -> u64 {
    *rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *rng;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let lo = if digits <= 1 {
        0
    } else {
        10_u64.pow((digits - 1) as u32)
    };
    let hi = 10_u64.pow(digits as u32) - 1;
    lo + z % (hi - lo + 1)
}

fn encode_reverse(body: &str) -> Vec<u32> {
    const ALPHA: &str = "0123456789,#";
    let mut ids: Vec<u32> = body
        .chars()
        .map(|ch| ALPHA.find(ch).unwrap_or(11) as u32)
        .collect();
    ids.resize(144, 11);
    ids
}

/// CPU FP16 hybrid L2-D32 reverse 1–8 digits. Diagnostic: not a 90% gate.
#[test]
fn reverse_l2_d32_smoke_1_to_8_digits_runs() {
    let model = UllisHeron::new(reverse_cfg()).unwrap();
    assert_eq!(model.cfg.architecture, Architecture::RosaRwkv7);
    let mut rng = 42_u64;
    let mut n_all = 0_usize;
    let mut n_good = 0_usize;
    for digits in 1..=8 {
        for _ in 0..4 {
            let raw = get_randint(digits, &mut rng).to_string();
            let body = format!("{raw},{}", raw.chars().rev().collect::<String>());
            let src = encode_reverse(&body);
            let logits = model.logits(&src, 1, 144).unwrap();
            let xx: String = src
                .iter()
                .map(|&id| b"0123456789,#"[id as usize] as char)
                .collect();
            let p1 = xx.find(',').unwrap();
            let p2 = xx.find('#').unwrap();
            n_all += p2 - p1;
            for offset in 0..(p2 - p1) {
                let row = &logits[(p1 + offset) * 12..(p1 + offset + 1) * 12];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0 as u32;
                if pred == src[p1 + 1 + offset] {
                    n_good += 1;
                }
            }
        }
    }
    assert!(n_all > 0);
    assert!(n_good <= n_all);
}

#[test]
fn hybrid_hidden_is_finite_on_padded_t() {
    let model = UllisHeron::new(reverse_cfg()).unwrap();
    let tokens: Vec<u32> = (0..16).map(|i| i % 12).collect();
    let hidden = model.hidden(&tokens, 1, 16).unwrap();
    assert_eq!(hidden.len(), 16 * 32);
    assert!(hidden.iter().all(|v| v.is_finite()));
}

#[test]
fn hybrid_generate_step_matches_one_shot_logits() {
    let model = UllisHeron::new(reverse_cfg()).unwrap();
    let tokens: Vec<u32> = (0..16).map(|i| (i * 3) % 12).collect();
    let logits = model.logits(&tokens, 1, 16).unwrap();
    let mut state = model.generate_state().unwrap();
    for (t, &id) in tokens.iter().enumerate() {
        let step = model.generate_step(&mut state, id).unwrap();
        let expected = &logits[t * 12..(t + 1) * 12];
        for (a, b) in step.iter().zip(expected) {
            assert!((a - b).abs() < 2e-4, "t={t} {a} vs {b}");
        }
    }
}

#[test]
fn plusminus_pad_is_144_not_129() {
    let cfg = TrainConfig {
        architecture: Architecture::RosaRwkv7,
        d_model: 32,
        n_layers: 2,
        vocab_size: 13,
        context_len: 144,
        dim_ffn: 128,
        tmix_lora_rank: 8,
        ..Default::default()
    };
    cfg.validate().unwrap();
    assert!(
        TrainConfig {
            context_len: 129,
            vocab_size: MIN_VOCAB as usize,
            ..cfg.clone()
        }
        .validate()
        .is_err()
    );
}
