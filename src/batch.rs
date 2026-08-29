//! Zero-copy fixed-shape batches for causal next-token training.
//!
//! A caller owns the token corpus and this module only borrows windows from it.
//! That keeps dataset size out of the trainer's working set and avoids padded
//! tensors whose size grows with the longest document in a batch.

use crate::config::TrainConfig;
use anyhow::{Result, bail};

/// One contiguous `[batch, time]` token view.
#[derive(Clone, Copy, Debug)]
pub struct CausalBatch<'a> {
    tokens: &'a [u32],
    labels: &'a [u32],
    batch_size: usize,
    time: usize,
}

impl<'a> CausalBatch<'a> {
    pub const fn tokens(self) -> &'a [u32] {
        self.tokens
    }

    /// Next-token targets aligned with [`Self::tokens`]. Unsupervised positions
    /// are `pad` when the caller built an SFT stream.
    pub const fn labels(self) -> &'a [u32] {
        self.labels
    }

    pub const fn batch_size(self) -> usize {
        self.batch_size
    }

    pub const fn time(self) -> usize {
        self.time
    }
}

/// Iterates non-overlapping, fixed-size causal batches over a token corpus.
#[derive(Clone, Debug)]
pub struct CausalBatcher<'a> {
    tokens: &'a [u32],
    labels: &'a [u32],
    batch_size: usize,
    time: usize,
    batch_tokens: usize,
    offset: usize,
}

impl<'a> CausalBatcher<'a> {
    pub fn new(tokens: &'a [u32], batch_size: usize, time: usize) -> Result<Self> {
        Self::new_with_labels(tokens, tokens, batch_size, time)
    }

    pub fn new_with_labels(
        tokens: &'a [u32],
        labels: &'a [u32],
        batch_size: usize,
        time: usize,
    ) -> Result<Self> {
        if batch_size == 0 || time < 2 {
            bail!("causal batches require a non-zero batch size and time >= 2");
        }
        if tokens.len() != labels.len() {
            bail!("causal label stream length must match tokens");
        }
        let batch_tokens = batch_size
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("causal batch shape overflow"))?;
        Ok(Self {
            tokens,
            labels,
            batch_size,
            time,
            batch_tokens,
            offset: 0,
        })
    }

    pub fn from_config(tokens: &'a [u32], cfg: &TrainConfig, time: usize) -> Result<Self> {
        cfg.validate()?;
        if time > cfg.context_len {
            bail!("batch time exceeds configured context length");
        }
        Self::new(tokens, cfg.batch_size, time)
    }

    pub fn from_config_with_labels(
        tokens: &'a [u32],
        labels: &'a [u32],
        cfg: &TrainConfig,
        time: usize,
    ) -> Result<Self> {
        cfg.validate()?;
        if time > cfg.context_len {
            bail!("batch time exceeds configured context length");
        }
        Self::new_with_labels(tokens, labels, cfg.batch_size, time)
    }

    pub fn remaining_batches(&self) -> usize {
        self.tokens.len().saturating_sub(self.offset) / self.batch_tokens
    }

    /// Advance past `n` already-completed train steps so resume does not
    /// replay the prefix of the packed stream.
    pub fn skip_batches(&mut self, n: usize) -> Result<()> {
        let skip = n
            .checked_mul(self.batch_tokens)
            .ok_or_else(|| anyhow::anyhow!("resume skip overflow"))?;
        let end = self
            .offset
            .checked_add(skip)
            .ok_or_else(|| anyhow::anyhow!("resume skip overflow"))?;
        if end > self.tokens.len() {
            bail!(
                "resume step {n} needs {} packed tokens but the stream only has {}",
                end,
                self.tokens.len()
            );
        }
        self.offset = end;
        Ok(())
    }
}

impl<'a> Iterator for CausalBatcher<'a> {
    type Item = CausalBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.offset.checked_add(self.batch_tokens)?;
        if end > self.tokens.len() {
            return None;
        }
        let batch = CausalBatch {
            tokens: &self.tokens[self.offset..end],
            labels: &self.labels[self.offset..end],
            batch_size: self.batch_size,
            time: self.time,
        };
        self.offset = end;
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_borrow_fixed_windows_and_skip_only_the_tail() {
        let tokens: Vec<u32> = (0..17).collect();
        let mut batches = CausalBatcher::new(&tokens, 2, 4).unwrap();
        assert_eq!(batches.remaining_batches(), 2);
        let first = batches.next().unwrap();
        assert_eq!(first.tokens(), &tokens[..8]);
        assert_eq!(first.labels(), &tokens[..8]);
        assert_eq!((first.batch_size(), first.time()), (2, 4));
        let second = batches.next().unwrap();
        assert_eq!(second.tokens(), &tokens[8..16]);
        assert!(batches.next().is_none());
    }

    #[test]
    fn accepts_time_two_for_next_token_loss() {
        let tokens: Vec<u32> = (0..4).collect();
        let mut batches = CausalBatcher::new(&tokens, 1, 2).unwrap();
        assert_eq!(batches.next().unwrap().tokens(), &tokens[..2]);
    }

    #[test]
    fn rejects_shapes_that_cannot_support_next_token_loss() {
        assert!(CausalBatcher::new(&[1, 2, 3], 0, 2).is_err());
        assert!(CausalBatcher::new(&[1, 2, 3], 1, 1).is_err());
    }

    #[test]
    fn supervised_batcher_keeps_labels_aligned() {
        let tokens: Vec<u32> = (0..8).collect();
        let labels = [0_u32, 1, 0, 3, 0, 5, 0, 7];
        let mut batches = CausalBatcher::new_with_labels(&tokens, &labels, 1, 4).unwrap();
        let first = batches.next().unwrap();
        assert_eq!(first.tokens(), &tokens[..4]);
        assert_eq!(first.labels(), &labels[..4]);
        assert!(CausalBatcher::new_with_labels(&[1, 2], &[1], 1, 2).is_err());
    }

    #[test]
    fn skip_batches_starts_at_the_resume_window() {
        let tokens: Vec<u32> = (0..16).collect();
        let mut batches = CausalBatcher::new(&tokens, 1, 4).unwrap();
        batches.skip_batches(2).unwrap();
        assert_eq!(batches.next().unwrap().tokens(), &tokens[8..12]);
        assert!(
            CausalBatcher::new(&tokens, 1, 4)
                .unwrap()
                .skip_batches(5)
                .is_err()
        );
    }

    #[test]
    fn config_batcher_rejects_invalid_configuration_before_borrowing_data() {
        let cfg = TrainConfig {
            vocab_size: 1,
            ..Default::default()
        };
        assert!(CausalBatcher::from_config(&[1, 2, 3], &cfg, 2).is_err());
    }
}
