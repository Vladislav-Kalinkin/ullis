//! Configuration for the single supported architecture: dense ternary Hyena.
use crate::optimizer::LionConfig;
use crate::tokenizer::{BpeTokenizer, MIN_VOCAB};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
pub const MAX_CONTEXT_LEN: usize = 32_768;
/// Default process budget.  Leaving headroom is essential on unified memory:
/// macOS needs room for the window server, Metal driver, and file cache.
pub const DEFAULT_MEMORY_BUDGET_BYTES: usize = 1_073_741_824;

/// Conservative, overflow-checked upper bounds used before allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// FP32 trainable state before optimiser copies.
    pub parameters: usize,
    /// Two packed bitplanes for each ternary projection.
    pub ternary_codes: usize,
    /// Per-output dequantisation scales for ternary projections.
    pub ternary_scales: usize,
    pub forward_working_set: usize,
    /// Reused real filter and two complex FFT work buffers for one channel.
    pub hyena_workspace: usize,
    pub materialized_mtp_logits: usize,
}

impl MemoryEstimate {
    pub fn inference_peak(self) -> Option<usize> {
        self.parameters
            .checked_add(self.ternary_codes)
            .and_then(|total| total.checked_add(self.ternary_scales))
            .and_then(|total| total.checked_add(self.forward_working_set))
            .and_then(|total| total.checked_add(self.hyena_workspace))
    }

    pub fn training_peak(self) -> Option<usize> {
        // FP32 master weights, gradient, and Lion's one momentum vector.
        // Packed codes are resident independently and remain needed for the
        // forward pass.
        self.parameters
            .checked_mul(12)
            .and_then(|weights| weights.checked_add(self.ternary_codes))
            .and_then(|total| total.checked_add(self.ternary_scales))
            .and_then(|total| total.checked_add(self.forward_working_set))
            .and_then(|total| total.checked_add(self.hyena_workspace))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub context_len: usize,
    pub batch_size: usize,
    pub filter_order: usize,
    pub ternary_delta: f32,
    pub seed: u64,
    #[serde(default)]
    pub lion: LionConfig,
    /// Hard allocation budget.  Runtime paths reject oversized requests before
    /// building vectors; callers may set a lower value for constrained Macs.
    #[serde(default = "default_memory_budget_bytes")]
    pub memory_budget_bytes: usize,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_layers: 6,
            vocab_size: 8192,
            context_len: MAX_CONTEXT_LEN,
            batch_size: 1,
            filter_order: 8,
            ternary_delta: 0.7,
            seed: 7,
            lion: LionConfig::default(),
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}
impl TrainConfig {
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.n_layers == 0 || self.vocab_size < MIN_VOCAB as usize {
            bail!("d_model, n_layers, and vocab_size must be non-zero (vocab >= {MIN_VOCAB})");
        }
        if self.context_len == 0 || self.context_len > MAX_CONTEXT_LEN {
            bail!("context_len must be in 1..={MAX_CONTEXT_LEN}");
        }
        if self.filter_order == 0 || !self.ternary_delta.is_finite() || self.ternary_delta <= 0.0 {
            bail!("filter_order and ternary_delta must be positive");
        }
        self.lion.validate()?;
        let estimate = self.memory_estimate()?;
        if !matches!(estimate.training_peak(), Some(n) if n <= self.memory_budget_bytes) {
            bail!(
                "configuration needs more than the {} MiB memory budget; reduce d_model, layers, batch_size, or context_len",
                self.memory_budget_bytes / (1024 * 1024)
            );
        }
        Ok(())
    }

    pub fn memory_estimate(&self) -> Result<MemoryEstimate> {
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| anyhow::anyhow!("model size overflow"))
        };
        let add = |a: usize, b: usize| {
            a.checked_add(b)
                .ok_or_else(|| anyhow::anyhow!("model size overflow"))
        };
        let d2 = mul(self.d_model, self.d_model)?;
        let embedding = mul(self.vocab_size, self.d_model)?;
        // Every layer has a D→2D and D→D ternary projection, plus three
        // FP32 vectors for the implicit filter's compact generator.
        let layer_projection = mul(3, d2)?;
        let layer_filter = mul(3, mul(self.d_model, self.filter_order)?)?;
        let per_layer = add(layer_projection, layer_filter)?;
        let block_parameters = mul(self.n_layers, per_layer)?;
        let mtp_parameters = mul(2, d2)?;
        let parameter_floats = add(add(embedding, block_parameters)?, mtp_parameters)?;
        let parameters = mul(parameter_floats, size_of::<f32>())?;
        // Codes are stored in separate positive/negative 64-bit bitplanes.
        // Each projection is rounded independently, matching the model layout.
        let packed = |weights: usize| {
            weights
                .checked_add(63)
                .and_then(|n| n.checked_div(64))
                .and_then(|words| words.checked_mul(2 * size_of::<u64>()))
                .ok_or_else(|| anyhow::anyhow!("ternary code size overflow"))
        };
        let input_codes = packed(mul(2, d2)?)?;
        let output_codes = packed(d2)?;
        let layer_codes = add(input_codes, output_codes)?;
        let mtp_codes = add(packed(d2)?, packed(d2)?)?;
        let ternary_codes = add(mul(self.n_layers, layer_codes)?, mtp_codes)?;
        let layer_scales = mul(3, self.d_model)?;
        let ternary_scales = mul(
            add(mul(self.n_layers, layer_scales)?, mul(2, self.d_model)?)?,
            size_of::<f32>(),
        )?;
        let rows = mul(self.batch_size, self.context_len)?;
        let activations = mul(rows, self.d_model)?;
        // Seven [B,T,D] FP32 buffers safely cover the residual input, 2D
        // projection (which also holds the gate), convolution output, update,
        // replacement residual, MTP head overlap, and allocator headroom.
        let forward_working_set = mul(activations, 7 * size_of::<f32>())?;
        let convolution_len = self
            .context_len
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| anyhow::anyhow!("FFT workspace size overflow"))?;
        let fft_len = convolution_len
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("FFT workspace size overflow"))?;
        let complex_work = mul(2, mul(fft_len, 2 * size_of::<f32>())?)?;
        let hyena_workspace = add(complex_work, mul(self.context_len, size_of::<f32>())?)?;
        let materialized_mtp_logits = mul(mul(rows, self.vocab_size)?, 2 * size_of::<f32>())?;
        Ok(MemoryEstimate {
            parameters,
            ternary_codes,
            ternary_scales,
            forward_working_set,
            hyena_workspace,
            materialized_mtp_logits,
        })
    }

    /// Binds model output rows to the tokenizer that will actually produce
    /// training ids. A BPE ceiling must not reserve embedding rows for merges
    /// which were never learned.
    pub fn with_tokenizer(mut self, tokenizer: &BpeTokenizer) -> Result<Self> {
        self.vocab_size = tokenizer.vocab_size() as usize;
        self.validate()?;
        Ok(self)
    }

    /// Full vocab logits are useful for tiny tests and generation, but must
    /// never be allocated for 32k pretraining.  The trainer will use streamed
    /// cross-entropy instead.
    pub fn validate_materialized_mtp(&self, time: usize) -> Result<()> {
        if time == 0 || time > self.context_len {
            bail!("time must be in 1..=context_len");
        }
        let rows = self
            .batch_size
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        let bytes = rows
            .checked_mul(self.vocab_size)
            .and_then(|n| n.checked_mul(2 * size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("MTP logits size overflow"))?;
        if bytes > self.memory_budget_bytes / 4 {
            bail!(
                "materialized MTP logits require {} MiB; use streamed MTP loss instead",
                bytes / (1024 * 1024)
            );
        }
        Ok(())
    }
}

const fn default_memory_budget_bytes() -> usize {
    DEFAULT_MEMORY_BUDGET_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::train_bpe;

    #[test]
    fn estimate_accounts_for_two_bit_ternary_planes() {
        let cfg = TrainConfig {
            vocab_size: 320,
            d_model: 8,
            n_layers: 1,
            context_len: 8,
            ..Default::default()
        };
        // D→2D: 128 weights -> 32 bytes; D→D: 64 -> 16; two MTP D→D
        // heads add 32 bytes, for 80 bytes total.
        assert_eq!(cfg.memory_estimate().unwrap().ternary_codes, 80);
        assert_eq!(cfg.memory_estimate().unwrap().ternary_scales, 160);
    }

    #[test]
    fn estimate_includes_reused_fft_workspace() {
        let cfg = TrainConfig {
            context_len: 32,
            ..Default::default()
        };
        // FFT length is 64. Two complex FP32 buffers use 1024 bytes; the
        // real filter channel uses another 128 bytes.
        assert_eq!(cfg.memory_estimate().unwrap().hyena_workspace, 1_152);
    }

    #[test]
    fn tokenizer_binding_uses_compact_vocab_and_revalidates_budget() {
        let tokenizer = train_bpe(&["tiny tiny corpus".into()], 512, 1).unwrap();
        let config = TrainConfig::default().with_tokenizer(&tokenizer).unwrap();
        assert_eq!(config.vocab_size, tokenizer.vocab_size() as usize);
    }
}
