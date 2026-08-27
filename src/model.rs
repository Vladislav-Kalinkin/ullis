//! Heron / ROSA model skeleton and version-2 checkpoint schema.
//!
//! Trainable forward and BinaryConnect live in later PRs. This module owns the
//! public types, admission through `TrainConfig`, and the irreversible cut of
//! Hyena `format_version: 1` checkpoints.

use crate::config::{Architecture, TrainConfig};
use crate::precision::Fp16;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;

/// Next-token cross-entropy statistics. Values are means over valid positions;
/// no `[batch, time, vocab]` logits tensor is retained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausalLoss {
    pub next_token: f32,
    pub next_token_count: usize,
}

impl CausalLoss {
    pub fn mean(self) -> f32 {
        self.next_token
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PackedBinaryCheckpoint {
    /// Little-endian u32 words, row-major `[out, in]`. Bit 0 of word 0 is
    /// weight `[0, 0]`; `1` means `+1`, `0` means `-1`.
    bits: Vec<u32>,
    scale_bits: Vec<u16>,
    bias_bits: Option<Vec<u16>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LayerNormBits {
    weight: Vec<u16>,
    bias: Vec<u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Fp16Vec(Vec<u16>);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RosaCheckpoint {
    x_q: Fp16Vec,
    x_k: Fp16Vec,
    x_v: Fp16Vec,
    e: Fp16Vec,
    q: PackedBinaryCheckpoint,
    k: PackedBinaryCheckpoint,
    v: PackedBinaryCheckpoint,
    o: PackedBinaryCheckpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CmixCheckpoint {
    x_k: Fp16Vec,
    key: PackedBinaryCheckpoint,
    value_bits: Fp16Vec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HeronBlockCheckpoint {
    ln0: Option<LayerNormBits>,
    ln2: LayerNormBits,
    ln3: LayerNormBits,
    rosa: RosaCheckpoint,
    ffn: CmixCheckpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TmixCheckpoint {
    x_r: Fp16Vec,
    x_w: Fp16Vec,
    x_k: Fp16Vec,
    x_v: Fp16Vec,
    x_a: Fp16Vec,
    x_g: Fp16Vec,
    w1: Fp16Vec,
    a1: Fp16Vec,
    v1: Fp16Vec,
    g1: Fp16Vec,
    w2: Fp16Vec,
    a2: Fp16Vec,
    v2: Fp16Vec,
    g2: Fp16Vec,
    w0: Fp16Vec,
    a0: Fp16Vec,
    v0: Fp16Vec,
    k_k: Fp16Vec,
    k_a: Fp16Vec,
    r_k: Fp16Vec,
    receptance: Fp16Vec,
    key: Fp16Vec,
    value: Fp16Vec,
    output: Fp16Vec,
    ln_x_weight: Fp16Vec,
    ln_x_bias: Fp16Vec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HybridBlockCheckpoint {
    ln_a: LayerNormBits,
    ln_b: LayerNormBits,
    ln_c: LayerNormBits,
    tmix: TmixCheckpoint,
    rosa: RosaCheckpoint,
    ffn: CmixCheckpoint,
}

/// Versioned snapshot of persistent Heron state.
///
/// Packed ±1 matrices store bits, learned scales, and official bias. FP16
/// masters for those matrices are process-local BinaryConnect latents and are
/// not written here. Hyena `format_version: 1` files are intentionally
/// unloadable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub format_version: u32,
    pub config: TrainConfig,
    embedding_bits: Vec<u16>,
    ln_out: LayerNormBits,
    head: PackedBinaryCheckpoint,
    #[serde(default)]
    blocks: Vec<HeronBlockCheckpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    hybrid_blocks: Vec<HybridBlockCheckpoint>,
}

/// Product model. Weights are a correctly shaped zero/init skeleton until the
/// CPU Heron PR fills BinaryConnect and ROSA.
#[derive(Clone, Debug)]
pub struct UllisHeron {
    pub cfg: TrainConfig,
    checkpoint: ModelCheckpoint,
}

impl UllisHeron {
    pub fn new(cfg: TrainConfig) -> Result<Self> {
        cfg.validate()?;
        let checkpoint = skeleton_checkpoint(&cfg)?;
        Ok(Self { cfg, checkpoint })
    }

    pub fn checkpoint(&self) -> ModelCheckpoint {
        self.checkpoint.clone()
    }

    pub fn from_checkpoint(checkpoint: ModelCheckpoint) -> Result<Self> {
        if checkpoint.format_version != CHECKPOINT_FORMAT_VERSION {
            bail!(
                "Hyena checkpoints (v1) are intentionally unloadable after the RWKV-8 cut (got format_version {})",
                checkpoint.format_version
            );
        }
        checkpoint.config.validate()?;
        validate_checkpoint_shapes(&checkpoint)?;
        Ok(Self {
            cfg: checkpoint.config.clone(),
            checkpoint,
        })
    }

    pub fn train_step(
        &mut self,
        _tokens: &[u32],
        _batch: usize,
        _time: usize,
        _learning_rate: f32,
    ) -> Result<CausalLoss> {
        bail!("Heron train not wired")
    }
}

fn ones(len: usize) -> Vec<u16> {
    vec![Fp16::from_f32(1.0).to_bits(); len]
}

fn zeros(len: usize) -> Vec<u16> {
    vec![0; len]
}

fn packed_linear(out: usize, in_features: usize, bias: bool, scale: f32) -> PackedBinaryCheckpoint {
    let weights = out.saturating_mul(in_features);
    let words = weights.div_ceil(32);
    PackedBinaryCheckpoint {
        bits: vec![0; words],
        scale_bits: vec![Fp16::from_f32(scale).to_bits(); out],
        bias_bits: bias.then(|| zeros(out)),
    }
}

fn layer_norm(d: usize) -> LayerNormBits {
    LayerNormBits {
        weight: ones(d),
        bias: zeros(d),
    }
}

fn fp16_zeros(len: usize) -> Fp16Vec {
    Fp16Vec(zeros(len))
}

fn fp16_ones(len: usize) -> Fp16Vec {
    Fp16Vec(ones(len))
}

fn rosa_checkpoint(d: usize) -> RosaCheckpoint {
    let scale = (d as f32).sqrt().recip();
    RosaCheckpoint {
        x_q: fp16_zeros(d),
        x_k: fp16_zeros(d),
        x_v: fp16_zeros(d),
        e: fp16_ones(d),
        q: packed_linear(d, d, true, scale),
        k: packed_linear(d, d, true, scale),
        v: packed_linear(d, d, true, scale),
        o: packed_linear(d, d, true, scale),
    }
}

fn cmix_checkpoint(d: usize, dim_ffn: usize) -> CmixCheckpoint {
    let key_scale = (d as f32).sqrt().recip();
    CmixCheckpoint {
        x_k: fp16_zeros(d),
        key: packed_linear(dim_ffn, d, false, key_scale),
        value_bits: fp16_zeros(dim_ffn.saturating_mul(d)),
    }
}

fn tmix_checkpoint(cfg: &TrainConfig) -> TmixCheckpoint {
    let d = cfg.d_model;
    let rank = cfg.resolved_tmix_lora_rank();
    let heads = d / cfg.head_size.max(1);
    TmixCheckpoint {
        x_r: fp16_zeros(d),
        x_w: fp16_zeros(d),
        x_k: fp16_zeros(d),
        x_v: fp16_zeros(d),
        x_a: fp16_zeros(d),
        x_g: fp16_zeros(d),
        w1: fp16_zeros(d.saturating_mul(rank)),
        a1: fp16_zeros(d.saturating_mul(rank)),
        v1: fp16_zeros(d.saturating_mul(rank)),
        g1: fp16_zeros(d.saturating_mul(rank)),
        w2: fp16_zeros(rank.saturating_mul(d)),
        a2: fp16_zeros(rank.saturating_mul(d)),
        v2: fp16_zeros(rank.saturating_mul(d)),
        g2: fp16_zeros(rank.saturating_mul(d)),
        w0: fp16_zeros(d),
        a0: fp16_zeros(d),
        v0: fp16_zeros(d),
        k_k: fp16_zeros(d),
        k_a: fp16_zeros(d),
        r_k: fp16_zeros(heads.saturating_mul(cfg.head_size)),
        receptance: fp16_zeros(d.saturating_mul(d)),
        key: fp16_zeros(d.saturating_mul(d)),
        value: fp16_zeros(d.saturating_mul(d)),
        output: fp16_zeros(d.saturating_mul(d)),
        ln_x_weight: ones_vec(d),
        ln_x_bias: fp16_zeros(d),
    }
}

fn ones_vec(len: usize) -> Fp16Vec {
    Fp16Vec(ones(len))
}

fn packed_words(out: usize, in_features: usize) -> usize {
    out.saturating_mul(in_features).div_ceil(32)
}

fn check_packed(matrix: &PackedBinaryCheckpoint, out: usize, in_features: usize, bias: bool) -> Result<()> {
    if matrix.bits.len() != packed_words(out, in_features) || matrix.scale_bits.len() != out {
        bail!("checkpoint packed-linear shape mismatch");
    }
    match (&matrix.bias_bits, bias) {
        (Some(bias_bits), true) if bias_bits.len() == out => Ok(()),
        (None, false) => Ok(()),
        _ => bail!("checkpoint packed-linear bias mismatch"),
    }
}

fn check_ln(ln: &LayerNormBits, d: usize) -> Result<()> {
    if ln.weight.len() != d || ln.bias.len() != d {
        bail!("checkpoint LayerNorm shape mismatch");
    }
    Ok(())
}

fn check_vec(vec: &Fp16Vec, len: usize, name: &str) -> Result<()> {
    if vec.0.len() != len {
        bail!("checkpoint {name} length mismatch");
    }
    Ok(())
}

fn check_rosa(rosa: &RosaCheckpoint, d: usize) -> Result<()> {
    check_vec(&rosa.x_q, d, "x_q")?;
    check_vec(&rosa.x_k, d, "x_k")?;
    check_vec(&rosa.x_v, d, "x_v")?;
    check_vec(&rosa.e, d, "e")?;
    check_packed(&rosa.q, d, d, true)?;
    check_packed(&rosa.k, d, d, true)?;
    check_packed(&rosa.v, d, d, true)?;
    check_packed(&rosa.o, d, d, true)?;
    Ok(())
}

fn check_cmix(ffn: &CmixCheckpoint, d: usize, dim_ffn: usize) -> Result<()> {
    check_vec(&ffn.x_k, d, "cmix.x_k")?;
    check_packed(&ffn.key, dim_ffn, d, false)?;
    check_vec(&ffn.value_bits, dim_ffn.saturating_mul(d), "cmix.value")?;
    Ok(())
}

fn validate_checkpoint_shapes(checkpoint: &ModelCheckpoint) -> Result<()> {
    let cfg = &checkpoint.config;
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let dim_ffn = cfg.resolved_dim_ffn();
    if checkpoint.embedding_bits.len() != v.saturating_mul(d) {
        bail!("checkpoint embedding shape does not match its configuration");
    }
    check_ln(&checkpoint.ln_out, d)?;
    check_packed(&checkpoint.head, v, d, false)?;
    match cfg.architecture {
        Architecture::Heron => {
            if checkpoint.blocks.len() != cfg.n_layers || !checkpoint.hybrid_blocks.is_empty() {
                bail!("heron checkpoint must store exactly n_layers Heron blocks");
            }
            for (index, block) in checkpoint.blocks.iter().enumerate() {
                match (&block.ln0, index) {
                    (Some(ln0), 0) => check_ln(ln0, d)?,
                    (None, _) if index > 0 => {}
                    _ => bail!("ln0 must be present only on Heron layer 0"),
                }
                check_ln(&block.ln2, d)?;
                check_ln(&block.ln3, d)?;
                check_rosa(&block.rosa, d)?;
                check_cmix(&block.ffn, d, dim_ffn)?;
            }
        }
        Architecture::RosaRwkv7 => {
            if checkpoint.hybrid_blocks.len() != cfg.n_layers || !checkpoint.blocks.is_empty() {
                bail!("rosa_rwkv7 checkpoint must store exactly n_layers hybrid blocks");
            }
            let rank = cfg.resolved_tmix_lora_rank();
            let heads = d / cfg.head_size;
            for block in &checkpoint.hybrid_blocks {
                check_ln(&block.ln_a, d)?;
                check_ln(&block.ln_b, d)?;
                check_ln(&block.ln_c, d)?;
                check_rosa(&block.rosa, d)?;
                check_cmix(&block.ffn, d, dim_ffn)?;
                let tmix = &block.tmix;
                check_vec(&tmix.w1, d.saturating_mul(rank), "tmix.w1")?;
                check_vec(&tmix.w2, rank.saturating_mul(d), "tmix.w2")?;
                check_vec(&tmix.receptance, d.saturating_mul(d), "tmix.receptance")?;
                check_vec(&tmix.r_k, heads.saturating_mul(cfg.head_size), "tmix.r_k")?;
                check_vec(&tmix.ln_x_weight, d, "tmix.ln_x")?;
            }
        }
    }
    Ok(())
}

fn skeleton_checkpoint(cfg: &TrainConfig) -> Result<ModelCheckpoint> {
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let dim_ffn = cfg.resolved_dim_ffn();
    let head_scale = (d as f32).sqrt().recip();
    let (blocks, hybrid_blocks) = match cfg.architecture {
        Architecture::Heron => {
            let blocks = (0..cfg.n_layers)
                .map(|layer| HeronBlockCheckpoint {
                    ln0: (layer == 0).then(|| layer_norm(d)),
                    ln2: layer_norm(d),
                    ln3: layer_norm(d),
                    rosa: rosa_checkpoint(d),
                    ffn: cmix_checkpoint(d, dim_ffn),
                })
                .collect();
            (blocks, Vec::new())
        }
        Architecture::RosaRwkv7 => {
            let hybrid = (0..cfg.n_layers)
                .map(|_| HybridBlockCheckpoint {
                    ln_a: layer_norm(d),
                    ln_b: layer_norm(d),
                    ln_c: layer_norm(d),
                    tmix: tmix_checkpoint(cfg),
                    rosa: rosa_checkpoint(d),
                    ffn: cmix_checkpoint(d, dim_ffn),
                })
                .collect();
            (Vec::new(), hybrid)
        }
    };
    let checkpoint = ModelCheckpoint {
        format_version: CHECKPOINT_FORMAT_VERSION,
        config: cfg.clone(),
        embedding_bits: zeros(v.saturating_mul(d)),
        ln_out: layer_norm(d),
        head: packed_linear(v, d, false, head_scale),
        blocks,
        hybrid_blocks,
    };
    validate_checkpoint_shapes(&checkpoint)?;
    Ok(checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Architecture;
    use crate::tokenizer::MIN_VOCAB;

    #[test]
    fn new_heron_emits_version_two_skeleton() {
        let model = UllisHeron::new(TrainConfig {
            vocab_size: MIN_VOCAB as usize,
            d_model: 16,
            n_layers: 1,
            dim_ffn: 64,
            context_len: 32,
            tmix_lora_rank: 8,
            ..Default::default()
        })
        .unwrap();
        let checkpoint = model.checkpoint();
        assert_eq!(checkpoint.format_version, 2);
        assert_eq!(checkpoint.blocks.len(), 1);
        assert!(checkpoint.blocks[0].ln0.is_some());
        assert!(checkpoint.head.bias_bits.is_none());
        assert!(checkpoint.blocks[0].rosa.q.bias_bits.is_some());
        let restored = UllisHeron::from_checkpoint(checkpoint).unwrap();
        assert_eq!(restored.cfg.d_model, 16);
    }

    #[test]
    fn hyena_v1_checkpoints_are_rejected() {
        let checkpoint = ModelCheckpoint {
            format_version: 1,
            config: TrainConfig::default(),
            embedding_bits: Vec::new(),
            ln_out: layer_norm(1),
            head: packed_linear(1, 1, false, 1.0),
            blocks: Vec::new(),
            hybrid_blocks: Vec::new(),
        };
        let error = UllisHeron::from_checkpoint(checkpoint).unwrap_err().to_string();
        assert!(error.contains("Hyena checkpoints (v1)"));
    }

    #[test]
    fn train_step_is_not_wired() {
        let mut model = UllisHeron::new(TrainConfig {
            vocab_size: MIN_VOCAB as usize,
            d_model: 16,
            n_layers: 1,
            dim_ffn: 64,
            context_len: 32,
            tmix_lora_rank: 8,
            ..Default::default()
        })
        .unwrap();
        let error = model.train_step(&[1, 2], 1, 2, 1e-3).unwrap_err().to_string();
        assert_eq!(error, "Heron train not wired");
    }

    #[test]
    fn hybrid_skeleton_round_trips() {
        let model = UllisHeron::new(TrainConfig {
            architecture: Architecture::RosaRwkv7,
            vocab_size: MIN_VOCAB as usize,
            d_model: 32,
            n_layers: 2,
            dim_ffn: 128,
            context_len: 144,
            tmix_lora_rank: 8,
            ..Default::default()
        })
        .unwrap();
        let checkpoint = model.checkpoint();
        assert!(checkpoint.blocks.is_empty());
        assert_eq!(checkpoint.hybrid_blocks.len(), 2);
        UllisHeron::from_checkpoint(checkpoint).unwrap();
    }
}
