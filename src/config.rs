//! Configuration for the single supported architecture: dense ternary Hyena.
use crate::optimizer::{LionConfig, OptimizerKind};
use crate::tokenizer::{BpeTokenizer, MIN_VOCAB};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
pub const MAX_CONTEXT_LEN: usize = 32_768;
/// Default process budget.  Leaving headroom is essential on unified memory:
/// macOS needs room for the window server, Metal driver, and file cache.
pub const DEFAULT_MEMORY_BUDGET_BYTES: usize = 1_073_741_824;

/// Describes the resident representation targeted by the Metal trainer.
///
/// The current CPU numerical oracle remains FP32. This profile is deliberately
/// separate from that oracle: configuration must be able to budget the final
/// trainer without pretending that its FP16 buffers already exist today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LowMemoryTrainingProfile {
    /// FP16 latent weights are required to move ternary thresholds without a
    /// permanent FP32 master copy.
    pub latent_weight_bytes: usize,
    /// Tied embeddings and compact implicit-filter parameters.
    pub parameter_bytes: usize,
    /// Row dequantisation scales for packed ternary projections.
    pub scale_bytes: usize,
    /// Resident activations and their checkpoint boundaries.
    pub activation_bytes: usize,
    /// Per-layer gradient workspace. It is reused after a fused update rather
    /// than allocated for every parameter in the model.
    pub gradient_bytes: usize,
    /// Complex FFT precision. Keeping this at four bytes per component is the
    /// conservative starting point for long contexts on M1.
    pub fft_component_bytes: usize,
    /// Number of `[B,T,D]` activation checkpoints retained by backward.
    pub activation_checkpoints: usize,
}

impl Default for LowMemoryTrainingProfile {
    fn default() -> Self {
        Self {
            latent_weight_bytes: 2,
            parameter_bytes: 2,
            scale_bytes: 2,
            activation_bytes: 2,
            gradient_bytes: 2,
            fft_component_bytes: 4,
            activation_checkpoints: 2,
        }
    }
}

impl LowMemoryTrainingProfile {
    fn validate(self) -> Result<()> {
        if self.latent_weight_bytes != 2
            || self.parameter_bytes != 2
            || self.scale_bytes != 2
            || self.activation_bytes != 2
            || self.gradient_bytes != 2
            || self.fft_component_bytes != 4
            || self.activation_checkpoints == 0
        {
            bail!(
                "the only supported low-memory profile is FP16 storage with FP32 FFT components and at least one activation checkpoint"
            );
        }
        Ok(())
    }
}

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
    /// Cached shared Metal buffers for one dense Hyena convolution. This
    /// includes two signal FFT buffers, two filter FFT buffers, and both the
    /// shared and returned causal output, but not a host staging spectrum.
    pub metal_hyena_workspace: usize,
    pub materialized_mtp_logits: usize,
    /// Detailed budget for the planned FP16 Metal trainer. This is not used to
    /// admit the current FP32 reference implementation.
    pub low_memory_training: LowMemoryTrainingEstimate,
}

/// Peak components of the planned low-memory training path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LowMemoryTrainingEstimate {
    pub latent_ternary_weights: usize,
    pub dense_parameters: usize,
    pub ternary_codes: usize,
    pub ternary_scales: usize,
    pub optimizer_state: usize,
    pub reusable_gradient_workspace: usize,
    pub checkpoint_activations: usize,
    pub fft_workspace: usize,
}

impl LowMemoryTrainingEstimate {
    pub fn peak(self) -> Option<usize> {
        self.latent_ternary_weights
            .checked_add(self.dense_parameters)
            .and_then(|total| total.checked_add(self.ternary_codes))
            .and_then(|total| total.checked_add(self.ternary_scales))
            .and_then(|total| total.checked_add(self.optimizer_state))
            .and_then(|total| total.checked_add(self.reusable_gradient_workspace))
            .and_then(|total| total.checked_add(self.checkpoint_activations))
            .and_then(|total| total.checked_add(self.fft_workspace))
    }
}

impl MemoryEstimate {
    pub fn inference_peak(self) -> Option<usize> {
        self.parameters
            .checked_add(self.ternary_codes)
            .and_then(|total| total.checked_add(self.ternary_scales))
            .and_then(|total| total.checked_add(self.forward_working_set))
            .and_then(|total| {
                total.checked_add(self.hyena_workspace.max(self.metal_hyena_workspace))
            })
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
            .and_then(|total| {
                total.checked_add(self.hyena_workspace.max(self.metal_hyena_workspace))
            })
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
    /// Maximum causal receptive field of each Hyena filter.  Longer contexts
    /// are processed as a stream, so this—not `context_len`—sets FFT scratch.
    #[serde(default = "default_hyena_kernel_len")]
    pub hyena_kernel_len: usize,
    /// Tokens processed per overlap-save FFT block. Must be at least the
    /// bounded receptive field.
    #[serde(default = "default_hyena_chunk_len")]
    pub hyena_chunk_len: usize,
    pub ternary_delta: f32,
    pub seed: u64,
    #[serde(default)]
    pub lion: LionConfig,
    /// Selects the optimizer-state budget for the forthcoming GPU trainer.
    /// Lion's numerical hyperparameters remain in `lion` for compatibility.
    #[serde(default)]
    pub optimizer: OptimizerKind,
    #[serde(default)]
    pub low_memory: LowMemoryTrainingProfile,
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
            // 32k remains supported with an explicitly larger unified-memory
            // budget; 8k is the safe out-of-the-box setting under 1 GiB once
            // resident Metal FFT and gate buffers are reserved.
            context_len: 8_192,
            batch_size: 1,
            filter_order: 8,
            hyena_kernel_len: default_hyena_kernel_len(),
            hyena_chunk_len: default_hyena_chunk_len(),
            ternary_delta: 0.7,
            seed: 7,
            lion: LionConfig::default(),
            optimizer: OptimizerKind::LionFp16,
            low_memory: LowMemoryTrainingProfile::default(),
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
        if self.filter_order == 0
            || self.hyena_kernel_len == 0
            || self.hyena_chunk_len < self.hyena_kernel_len
            || !self.ternary_delta.is_finite()
            || self.ternary_delta <= 0.0
        {
            bail!("filter_order and ternary_delta must be positive");
        }
        self.lion.validate()?;
        self.low_memory.validate()?;
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
        let bounded_chunk_len = self.hyena_chunk_len.min(self.context_len);
        let bounded_kernel_len = self.hyena_kernel_len.min(self.context_len);
        let convolution_len = bounded_chunk_len
            .checked_add(bounded_kernel_len)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| anyhow::anyhow!("FFT workspace size overflow"))?;
        let fft_len = convolution_len
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("FFT workspace size overflow"))?;
        let complex_work = mul(2, mul(fft_len, 2 * size_of::<f32>())?)?;
        let hyena_workspace = add(complex_work, mul(bounded_kernel_len, size_of::<f32>())?)?;
        let signal_transforms = mul(self.batch_size, self.d_model)?;
        let chunk_count = self.context_len.div_ceil(bounded_chunk_len);
        let chunked_signal_transforms = mul(signal_transforms, chunk_count)?;
        let signal_fft = mul(
            chunked_signal_transforms,
            mul(fft_len, 2 * size_of::<f32>())?,
        )?;
        let filter_fft = mul(self.d_model, mul(fft_len, 2 * size_of::<f32>())?)?;
        let metal_fft_workspace = add(
            add(mul(2, signal_fft)?, mul(2, filter_fft)?)?,
            // The final shared output and its CPU return value coexist until
            // the caller takes ownership of the Vec.
            mul(2, mul(activations, size_of::<f32>())?)?,
        )?;
        // A resident block also keeps its input, `[B,T,2D]` projection, and
        // two `[B,T,2D]` gate buffers alive while the mixer runs. Reserving
        // these now prevents a later readback-free forward path from making a
        // previously accepted 32k configuration enter swap.
        let metal_projection_gate = mul(7, mul(activations, size_of::<f32>())?)?;
        let metal_hyena_workspace = add(metal_fft_workspace, metal_projection_gate)?;
        let materialized_mtp_logits = mul(mul(rows, self.vocab_size)?, 2 * size_of::<f32>())?;
        let ternary_weights = add(mul(self.n_layers, layer_projection)?, mtp_parameters)?;
        let dense_parameters = add(embedding, mul(self.n_layers, layer_filter)?)?;
        let largest_trainable_tensor = embedding.max(mul(2, d2)?);
        let profile = self.low_memory;
        let low_memory_fft_workspace = add(
            add(
                mul(
                    2,
                    mul(
                        signal_transforms,
                        mul(fft_len, 2 * profile.fft_component_bytes)?,
                    )?,
                )?,
                mul(
                    2,
                    mul(self.d_model, mul(fft_len, 2 * profile.fft_component_bytes)?)?,
                )?,
            )?,
            mul(2, mul(activations, profile.activation_bytes)?)?,
        )?;
        let optimizer_state = self
            .optimizer
            .state_bytes(ternary_weights, profile.latent_weight_bytes)?;
        let low_memory_training = LowMemoryTrainingEstimate {
            latent_ternary_weights: mul(ternary_weights, profile.latent_weight_bytes)?,
            dense_parameters: mul(dense_parameters, profile.parameter_bytes)?,
            ternary_codes,
            ternary_scales: ternary_scales / size_of::<f32>() * profile.scale_bytes,
            optimizer_state,
            reusable_gradient_workspace: mul(largest_trainable_tensor, profile.gradient_bytes)?,
            checkpoint_activations: mul(
                profile.activation_checkpoints,
                mul(activations, profile.activation_bytes)?,
            )?,
            fft_workspace: low_memory_fft_workspace,
        };
        Ok(MemoryEstimate {
            parameters,
            ternary_codes,
            ternary_scales,
            forward_working_set,
            hyena_workspace,
            metal_hyena_workspace,
            materialized_mtp_logits,
            low_memory_training,
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

const fn default_hyena_kernel_len() -> usize {
    1_024
}

const fn default_hyena_chunk_len() -> usize {
    2_048
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
    fn estimate_reserves_cached_metal_hyena_buffers_without_host_staging() {
        let cfg = TrainConfig {
            d_model: 2,
            batch_size: 1,
            context_len: 4,
            ..Default::default()
        };
        // FFT buffers and outputs consume 576 bytes. Resident input,
        // projection, and two gate buffers add seven [B,T,D] FP32 tensors.
        assert_eq!(cfg.memory_estimate().unwrap().metal_hyena_workspace, 800);
    }

    #[test]
    fn tokenizer_binding_uses_compact_vocab_and_revalidates_budget() {
        let tokenizer = train_bpe(&["tiny tiny corpus".into()], 512, 1).unwrap();
        let config = TrainConfig::default().with_tokenizer(&tokenizer).unwrap();
        assert_eq!(config.vocab_size, tokenizer.vocab_size() as usize);
    }

    #[test]
    fn low_memory_ledger_separates_reusable_gradient_and_optimizer_state() {
        let cfg = TrainConfig {
            vocab_size: 320,
            d_model: 8,
            n_layers: 1,
            context_len: 8,
            ..Default::default()
        };
        let estimate = cfg.memory_estimate().unwrap().low_memory_training;
        // Ternary masters: D→2D + D→D + two MTP heads = 320 values.
        assert_eq!(estimate.latent_ternary_weights, 640);
        assert_eq!(estimate.optimizer_state, 640);
        // The tied embedding is the largest single trainable tensor, not a
        // second full-model gradient allocation.
        assert_eq!(estimate.reusable_gradient_workspace, 320 * 8 * 2);
        assert!(estimate.peak().is_some());
    }

    #[test]
    fn stateless_sgd_has_no_persistent_optimizer_budget() {
        let cfg = TrainConfig {
            optimizer: OptimizerKind::StatelessSgd,
            ..Default::default()
        };
        assert_eq!(
            cfg.memory_estimate()
                .unwrap()
                .low_memory_training
                .optimizer_state,
            0
        );
    }
}
