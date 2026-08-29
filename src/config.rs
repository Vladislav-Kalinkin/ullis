//! Configuration for the Heron / ROSA-RWKV7 architectures.
use crate::tokenizer::{BpeTokenizer, MIN_VOCAB};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub const MAX_CONTEXT_LEN: usize = 32_768;
/// Default process budget. Leaving headroom is essential on unified memory:
/// macOS needs room for the window server, Metal driver, and file cache.
pub const DEFAULT_MEMORY_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;
pub const DEFAULT_HEAD_SIZE: usize = 16;
const COMMAND_SLACK_BYTES: usize = 32 * 1024 * 1024;
/// Conservative activation-checkpoint reuse (four named layer snapshots).
const CHECKPOINT_LAYERS: usize = 4;
const FP16_BYTES: usize = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    #[default]
    Heron,
    RosaRwkv7,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RosaGradMode {
    #[default]
    StopGradBits,
    ExactBitflip,
    SteSign,
}

/// Overflow-checked upper bounds used before allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub embedding: usize,
    pub packed_bits_and_scales: usize,
    pub fp16_matrices: usize,
    pub ln_and_vec: usize,
    pub act_checkpoints: usize,
    pub qkv_bitplanes: usize,
    pub rosa_sam_peak: usize,
    pub packed_latents: usize,
    pub binaryconnect_workspace: usize,
    pub ce_scratch: usize,
    pub wkv_tape: usize,
    pub bwd_rosa_scratch: usize,
    pub command_slack: usize,
    /// FP32 error-diffusion carry for every resident FP16 parameter, including BinaryConnect latents.
    pub sgd_residual: usize,
}

impl MemoryEstimate {
    pub fn peak(self) -> Option<usize> {
        let add = |a: Option<usize>, b: usize| a.and_then(|total| total.checked_add(b));
        [
            self.embedding,
            self.packed_bits_and_scales,
            self.fp16_matrices,
            self.ln_and_vec,
            self.act_checkpoints,
            self.qkv_bitplanes,
            self.rosa_sam_peak,
            self.packed_latents,
            self.binaryconnect_workspace,
            self.ce_scratch,
            self.wkv_tape,
            self.bwd_rosa_scratch,
            self.command_slack,
            self.sgd_residual,
        ]
        .into_iter()
        .fold(Some(0), add)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    #[serde(default)]
    pub architecture: Architecture,
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub context_len: usize,
    pub batch_size: usize,
    /// Hidden width of CMix. Zero in JSON means `4 * d_model`.
    #[serde(default)]
    pub dim_ffn: usize,
    #[serde(default = "default_rosa_bits")]
    pub rosa_bits: u8,
    #[serde(default)]
    pub rosa_grad: RosaGradMode,
    #[serde(default = "default_head_size")]
    pub head_size: usize,
    #[serde(default)]
    pub tmix_lora_rank: usize,
    pub seed: u64,
    #[serde(default = "default_memory_budget_bytes")]
    pub memory_budget_bytes: usize,
}

const fn default_memory_budget_bytes() -> usize {
    DEFAULT_MEMORY_BUDGET_BYTES
}

const fn default_rosa_bits() -> u8 {
    1
}

const fn default_head_size() -> usize {
    DEFAULT_HEAD_SIZE
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            architecture: Architecture::Heron,
            d_model: 256,
            n_layers: 6,
            vocab_size: 8192,
            context_len: 2_048,
            batch_size: 1,
            dim_ffn: 1_024,
            rosa_bits: 1,
            rosa_grad: RosaGradMode::StopGradBits,
            head_size: DEFAULT_HEAD_SIZE,
            tmix_lora_rank: 16,
            seed: 7,
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

impl TrainConfig {
    pub fn resolved_dim_ffn(&self) -> usize {
        if self.dim_ffn == 0 {
            self.d_model.saturating_mul(4)
        } else {
            self.dim_ffn
        }
    }

    pub fn resolved_tmix_lora_rank(&self) -> usize {
        if self.tmix_lora_rank == 0 {
            if self.d_model <= 64 { 8 } else { 16 }
        } else {
            self.tmix_lora_rank
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.n_layers == 0 {
            bail!("d_model and n_layers must be non-zero");
        }
        let min_vocab = if matches!(self.architecture, Architecture::RosaRwkv7) {
            12
        } else {
            MIN_VOCAB as usize
        };
        if self.vocab_size < min_vocab {
            bail!("vocab_size must be at least {min_vocab}");
        }
        if self.context_len == 0 || self.context_len > MAX_CONTEXT_LEN {
            bail!("context_len must be in 1..={MAX_CONTEXT_LEN}");
        }
        if self.batch_size == 0 {
            bail!("batch_size must be positive");
        }
        if self.rosa_bits != 1 {
            bail!("rosa_bits must be 1 in Ullis 0.10 (4-bit ROSA is post-0.10)");
        }
        if self.head_size != DEFAULT_HEAD_SIZE {
            bail!("head_size must be {DEFAULT_HEAD_SIZE}");
        }
        let dim_ffn = self.resolved_dim_ffn();
        if dim_ffn == 0 {
            bail!("dim_ffn must be positive");
        }
        let rank = self.resolved_tmix_lora_rank();
        if rank != 8 && rank != 16 {
            bail!("tmix_lora_rank must be 8 or 16");
        }
        if matches!(self.architecture, Architecture::RosaRwkv7) {
            if !self.d_model.is_multiple_of(self.head_size) {
                bail!("rosa_rwkv7 requires d_model to be a multiple of head_size");
            }
            if !self.context_len.is_multiple_of(16) {
                bail!("rosa_rwkv7 requires context_len to be a multiple of 16");
            }
        }
        let estimate = self.memory_estimate()?;
        if !matches!(estimate.peak(), Some(n) if n <= self.memory_budget_bytes) {
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
        let packed_bytes = |weights: usize| {
            weights
                .checked_add(31)
                .and_then(|n| n.checked_div(32))
                .and_then(|words| words.checked_mul(size_of::<u32>()))
                .ok_or_else(|| anyhow::anyhow!("packed bitplane size overflow"))
        };

        let d = self.d_model;
        let v = self.vocab_size;
        let layers = self.n_layers;
        let dim_ffn = self.resolved_dim_ffn();
        let rows = mul(self.batch_size, self.context_len)?;
        let d2 = mul(d, d)?;
        let ffn_mat = mul(dim_ffn, d)?;
        let head_mat = mul(v, d)?;

        let embedding = mul(head_mat, FP16_BYTES)?;

        let layer_packed_weights = add(mul(4, d2)?, ffn_mat)?;
        let packed_weights = add(mul(layers, layer_packed_weights)?, head_mat)?;
        let packed_bits = packed_bytes(packed_weights)?;
        // Q/K/V/O scales + CMix-key scales + head scales, plus QKVO bias.
        let scale_rows = add(add(mul(layers, mul(4, d)?)?, mul(layers, dim_ffn)?)?, v)?;
        let scales = mul(scale_rows, FP16_BYTES)?;
        let rosa_bias = mul(mul(layers, mul(4, d)?)?, FP16_BYTES)?;
        let packed_bits_and_scales = add(add(packed_bits, scales)?, rosa_bias)?;

        let mut fp16_matrices = mul(mul(layers, ffn_mat)?, FP16_BYTES)?;
        if matches!(self.architecture, Architecture::RosaRwkv7) {
            let rank = self.resolved_tmix_lora_rank();
            let heads = d / self.head_size;
            let lora = mul(d, rank)?;
            let tmix = add(
                add(mul(6, d)?, mul(8, lora)?)?,
                add(add(mul(5, d)?, mul(heads, self.head_size)?)?, mul(4, d2)?)?,
            )?;
            fp16_matrices = add(fp16_matrices, mul(mul(layers, tmix)?, FP16_BYTES)?)?;
        }

        // ln0 (layer 0) + ln2/ln3 per layer + ln_out, plus x_qkv/e/x_k.
        let ln_vecs = add(mul(2, d)?, mul(layers, mul(4, d)?)?)?;
        let rosa_vecs = mul(layers, mul(4, d)?)?;
        let cmix_shift = mul(layers, d)?;
        let ln_and_vec = mul(add(add(ln_vecs, rosa_vecs)?, cmix_shift)?, FP16_BYTES)?;

        let per_layer_acts = mul(12, mul(rows, d)?)?;
        let act_checkpoints = mul(CHECKPOINT_LAYERS, mul(per_layer_acts, FP16_BYTES)?)?;

        let qkv_bitplanes = mul(3, rows)?
            .checked_mul(d)
            .and_then(|bits| bits.checked_div(8))
            .ok_or_else(|| anyhow::anyhow!("QKV bitplane size overflow"))?;
        let rosa_sam_peak = mul(40, mul(self.context_len, d)?)?;
        let packed_latents = mul(packed_weights, FP16_BYTES)?;
        let largest_matrix = d2.max(ffn_mat).max(head_mat);
        let binaryconnect_workspace = mul(largest_matrix, size_of::<f32>())?;
        let ce_scratch = mul(add(d, v)?, 8)?;

        let wkv_tape = if matches!(self.architecture, Architecture::RosaRwkv7) {
            let heads = d / self.head_size;
            let chunks = self.context_len.div_ceil(16);
            let state = mul(
                mul(mul(self.batch_size, heads)?, chunks)?,
                mul(self.head_size, self.head_size)?,
            )?;
            let sa = mul(
                mul(mul(self.batch_size, self.context_len)?, heads)?,
                self.head_size,
            )?;
            mul(add(state, sa)?, size_of::<f32>())?
        } else {
            0
        };
        let bwd_rosa_scratch = 0;
        let command_slack = COMMAND_SLACK_BYTES;
        let fp16_elements = add(
            add(embedding / FP16_BYTES, fp16_matrices / FP16_BYTES)?,
            add(ln_and_vec / FP16_BYTES, packed_latents / FP16_BYTES)?,
        )?;
        let sgd_residual = mul(fp16_elements, size_of::<f32>())?;

        Ok(MemoryEstimate {
            embedding,
            packed_bits_and_scales,
            fp16_matrices,
            ln_and_vec,
            act_checkpoints,
            qkv_bitplanes,
            rosa_sam_peak,
            packed_latents,
            binaryconnect_workspace,
            ce_scratch,
            wkv_tape,
            bwd_rosa_scratch,
            command_slack,
            sgd_residual,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::train_bpe;

    #[test]
    fn default_profile_fits_the_four_gib_budget() {
        let cfg = TrainConfig::default();
        cfg.validate().unwrap();
        let estimate = cfg.memory_estimate().unwrap();
        let peak = estimate.peak().unwrap();
        assert!(peak <= DEFAULT_MEMORY_BUDGET_BYTES);
        assert!(
            peak < 200 * 1024 * 1024,
            "default peak {peak} should stay well under 200 MiB"
        );
        assert_eq!(estimate.embedding, 8192 * 256 * 2);
        assert_eq!(estimate.rosa_sam_peak, 40 * 2048 * 256);
        assert_eq!(estimate.qkv_bitplanes, 3 * 2048 * 256 / 8);
        assert_eq!(estimate.binaryconnect_workspace, 8192 * 256 * 4);
        assert_eq!(estimate.wkv_tape, 0);
        assert_eq!(cfg.context_len, 2048);
    }

    #[test]
    fn wide_heron_profile_is_admitted() {
        let cfg = TrainConfig {
            n_layers: 12,
            d_model: 768,
            dim_ffn: 768 * 4,
            context_len: 512,
            tmix_lora_rank: 16,
            ..Default::default()
        };
        cfg.validate().unwrap();
        let peak = cfg.memory_estimate().unwrap().peak().unwrap();
        assert!(
            peak < 1024 * 1024 * 1024,
            "wide heron peak {peak} should stay under 1 GiB after FP32 SGD residual"
        );
    }

    #[test]
    fn hybrid_rejects_unpadded_digit_lengths() {
        let cfg = TrainConfig {
            architecture: Architecture::RosaRwkv7,
            d_model: 32,
            n_layers: 2,
            dim_ffn: 128,
            context_len: 129,
            tmix_lora_rank: 8,
            vocab_size: MIN_VOCAB as usize,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn tokenizer_binding_uses_compact_vocab_and_revalidates_budget() {
        let tokenizer = train_bpe(&["tiny tiny corpus".into()], 512, 1).unwrap();
        let config = TrainConfig::default().with_tokenizer(&tokenizer).unwrap();
        assert_eq!(config.vocab_size, tokenizer.vocab_size() as usize);
    }

    #[test]
    fn four_bit_rosa_is_rejected() {
        let cfg = TrainConfig {
            rosa_bits: 4,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }
}
