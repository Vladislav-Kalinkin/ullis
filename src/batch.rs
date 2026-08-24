//! Zero-copy fixed-shape batches for causal MTP training.
//!
//! A caller owns the token corpus and this module only borrows windows from it.
//! That keeps dataset size out of the trainer's working set and avoids padded
//! tensors whose size grows with the longest document in a batch.

use crate::config::TrainConfig;
use anyhow::{Result, bail};

/// One contiguous `[batch, time]` token view.
#[derive(Clone, Copy, Debug)]
pub struct MtpBatch<'a> {
    tokens: &'a [u32],
    batch_size: usize,
    time: usize,
}

impl<'a> MtpBatch<'a> {
    pub const fn tokens(self) -> &'a [u32] {
        self.tokens
    }

    pub const fn batch_size(self) -> usize {
        self.batch_size
    }

    pub const fn time(self) -> usize {
        self.time
    }
}

/// Iterates non-overlapping, fixed-size MTP batches over a token corpus.
#[derive(Clone, Debug)]
pub struct MtpBatcher<'a> {
    tokens: &'a [u32],
    batch_size: usize,
    time: usize,
    batch_tokens: usize,
    offset: usize,
}

impl<'a> MtpBatcher<'a> {
    pub fn new(tokens: &'a [u32], batch_size: usize, time: usize) -> Result<Self> {
        if batch_size == 0 || time < 3 {
            bail!("MTP batches require a non-zero batch size and time >= 3");
        }
        let batch_tokens = batch_size
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP batch shape overflow"))?;
        Ok(Self {
            tokens,
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

    pub fn remaining_batches(&self) -> usize {
        self.tokens.len().saturating_sub(self.offset) / self.batch_tokens
    }
}

impl<'a> Iterator for MtpBatcher<'a> {
    type Item = MtpBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.offset.checked_add(self.batch_tokens)?;
        if end > self.tokens.len() {
            return None;
        }
        let batch = MtpBatch {
            tokens: &self.tokens[self.offset..end],
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
        let mut batches = MtpBatcher::new(&tokens, 2, 4).unwrap();
        assert_eq!(batches.remaining_batches(), 2);
        let first = batches.next().unwrap();
        assert_eq!(first.tokens(), &tokens[..8]);
        assert_eq!((first.batch_size(), first.time()), (2, 4));
        let second = batches.next().unwrap();
        assert_eq!(second.tokens(), &tokens[8..16]);
        assert!(batches.next().is_none());
    }

    #[test]
    fn rejects_shapes_that_cannot_support_two_mtp_horizons() {
        assert!(MtpBatcher::new(&[1, 2, 3], 0, 3).is_err());
        assert!(MtpBatcher::new(&[1, 2, 3], 1, 2).is_err());
    }

    #[test]
    fn config_batcher_rejects_invalid_configuration_before_borrowing_data() {
        let cfg = TrainConfig {
            vocab_size: 1,
            ..Default::default()
        };
        assert!(MtpBatcher::from_config(&[1, 2, 3], &cfg, 3).is_err());
    }
}
