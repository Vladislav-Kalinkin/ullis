use ullis::{MtpBatcher, TrainConfig, UllisHyena};

#[test]
fn model_exposes_two_mtp_heads() {
    let cfg = TrainConfig {
        vocab_size: 320,
        d_model: 8,
        n_layers: 1,
        context_len: 16,
        ..Default::default()
    };
    let model = UllisHyena::new(cfg.clone()).unwrap();
    let (one, two) = model.mtp_logits(&[4, 5, 6, 7], 1, 4).unwrap();
    assert_eq!(one.len(), 4 * cfg.vocab_size);
    assert_eq!(two.len(), 4 * cfg.vocab_size);
    assert_ne!(one, two);
}

#[test]
fn model_rejects_context_beyond_budget() {
    let cfg = TrainConfig {
        vocab_size: 320,
        d_model: 8,
        n_layers: 1,
        context_len: 2,
        ..Default::default()
    };
    assert!(UllisHyena::new(cfg)
        .unwrap()
        .hidden(&[4, 5, 6], 1, 3)
        .is_err());
}

#[test]
fn materialized_mtp_is_rejected_before_large_allocation() {
    let cfg = TrainConfig {
        vocab_size: 8192,
        d_model: 8,
        n_layers: 1,
        context_len: 32_768,
        ..Default::default()
    };
    let model = UllisHyena::new(cfg).unwrap();
    let ids = vec![4; 32_768];
    let error = model.mtp_logits(&ids, 1, ids.len()).unwrap_err();
    assert!(error.to_string().contains("streamed MTP loss"));
}

#[test]
fn streamed_mtp_loss_matches_materialized_logits() {
    let cfg = TrainConfig {
        vocab_size: 320,
        d_model: 8,
        n_layers: 1,
        context_len: 8,
        ..Default::default()
    };
    let model = UllisHyena::new(cfg.clone()).unwrap();
    let ids = [4, 5, 6, 7];
    let (one, two) = model.mtp_logits(&ids, 1, ids.len()).unwrap();
    let expected_one = mean_cross_entropy(&one, cfg.vocab_size, &[5, 6, 7]);
    let expected_two = mean_cross_entropy(&two, cfg.vocab_size, &[6, 7]);
    let loss = model.streamed_mtp_loss(&ids, 1, ids.len()).unwrap();
    assert!((loss.next_token - expected_one).abs() < 1e-5);
    assert!((loss.second_token - expected_two).abs() < 1e-5);
    assert_eq!((loss.next_token_count, loss.second_token_count), (3, 2));
}

#[test]
fn model_accepts_a_zero_copy_mtp_batch() {
    let cfg = TrainConfig {
        vocab_size: 320,
        d_model: 8,
        n_layers: 1,
        context_len: 8,
        ..Default::default()
    };
    let model = UllisHyena::new(cfg).unwrap();
    let tokens = [4, 5, 6, 7];
    let batch = MtpBatcher::new(&tokens, 1, 4).unwrap().next().unwrap();
    assert_eq!(
        model.streamed_batch_loss(batch).unwrap().next_token_count,
        3
    );
}

fn mean_cross_entropy(logits: &[f32], vocab_size: usize, targets: &[u32]) -> f32 {
    targets
        .iter()
        .enumerate()
        .map(|(row, &target)| {
            let row_logits = &logits[row * vocab_size..(row + 1) * vocab_size];
            let max = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max + row_logits
                .iter()
                .map(|value| (value - max).exp())
                .sum::<f32>()
                .ln()
                - row_logits[target as usize]
        })
        .sum::<f32>()
        / targets.len() as f32
}
