use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const N_EXPERTS: usize = 3;
pub const EXPERT_NAMES: [&str; N_EXPERTS] = ["python", "rust", "bash"];

/// Defaults fit an M1 8 GB unified-memory budget (Ullis v0.5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub d_model: usize,
    pub n_layers: usize,
    /// Current / start grid; updated as the grid grows.
    pub n_basis: usize,
    pub grid_start: usize,
    pub grid_mid: usize,
    pub grid_final: usize,
    pub seq_len: usize,
    /// Continuous `SovereignFlashBuffer` cap (token ids). Independent of `seq_len`.
    #[serde(default = "default_context_len")]
    pub context_len: usize,
    pub batch_size: usize,
    /// `"shift"` (0 extra params) or `"attn"`.
    pub mixer: String,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub lr: f64,
    pub lr_qat: f64,
    pub lr_harden: f64,
    pub momentum: f64,
    pub l1: f64,
    pub ternary_delta: f64,
    pub steps_per_epoch: usize,
    pub epochs_warmup: usize,
    pub epochs_sparsify: usize,
    pub epochs_qat: usize,
    pub epochs_harden: usize,
    pub max_norm: f64,
    pub seed: u64,
    pub ckpt_dir: String,
    pub log_every: usize,
    pub tokenizer_path: String,
    pub data_path: String,
    /// Mixture-of-Bumps: split G into shared + routed experts.
    pub moe: bool,
    pub n_experts: usize,
    /// Shannon entropy penalty on vocab softmax (`λ_H`). Language-agnostic.
    #[serde(default = "default_entropy_coef")]
    pub entropy_coef: f64,
    /// Shannon entropy penalty on the K-expert router (`λ_R`).
    #[serde(default = "default_router_entropy_coef")]
    pub router_entropy_coef: f64,
    /// Insert one non-uniform knot every N steps during phases 1–2.
    #[serde(default = "default_knot_insert_every")]
    pub knot_insert_every: usize,
    /// EMA decay for per-knot residual energy / per-edge grad variance.
    #[serde(default = "default_knot_ema")]
    pub knot_ema: f64,
    /// Recompute layer activations on the backward pass (fused Metal / host).
    #[serde(default = "default_fused_grad_ckpt")]
    pub fused_grad_ckpt: bool,
    /// Storage master for KAN weights. Compute is always FP32 (hot unpack).
    #[serde(default)]
    pub master: MasterDtype,
    /// Momentum storage. Compute is always FP32.
    #[serde(default)]
    pub mom: MomDtype,
    /// 0 = dense MoB (bit-identical). 1|2 = per-token top-k after full softmax.
    #[serde(default)]
    pub moe_topk: u32,
    /// Switch load-balance `α · K · Σ f_i P_i`. Used only when `moe_topk > 0`.
    #[serde(default = "default_moe_aux")]
    pub moe_aux: f64,
    /// KAN factorization. Shared-edge is the production layout; `None` is rejected.
    #[serde(default)]
    pub kan_factor: KanFactor,
    /// `kan` (production) or `memory` (experimental).
    #[serde(default)]
    pub arch: ModelArch,
    /// Memory-arch expert inner width `W`.
    #[serde(default = "default_expert_width")]
    pub expert_width: usize,
    /// Memory-arch slot count `S`.
    #[serde(default = "default_n_slots")]
    pub n_slots: usize,
    /// Memory-arch expert count `E`. Independent of MoB `n_experts`.
    #[serde(default = "default_mem_experts")]
    pub mem_experts: usize,
    /// Causal local mix width. 0 disables. Clamped to 64 — never `T`.
    #[serde(default = "default_window")]
    pub window: usize,
    /// If true, CE is only on the output span (thinking is context).
    #[serde(default)]
    pub mask_output: bool,
}

/// Trainable model class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArch {
    #[default]
    Kan,
    Memory,
}

impl ModelArch {
    pub fn parse_name(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "kan" | "ulliskan" | "" => Ok(Self::Kan),
            "memory" | "mem" => Ok(Self::Memory),
            other => bail!("--arch {other}: expected kan|memory"),
        }
    }
}

/// Spline coefficient layout. Shared-edge is the only production path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum KanFactor {
    None,
    #[default]
    SharedEdge,
}

impl KanFactor {
    pub fn parse_name(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "off" | "unfactored" => Ok(Self::None),
            "shared-edge" | "shared_edge" | "edge" => Ok(Self::SharedEdge),
            other => bail!("--kan-factor {other}: expected none|shared-edge"),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::None => 0,
            Self::SharedEdge => 1,
        }
    }

    pub fn shared_edge(self) -> bool {
        matches!(self, Self::SharedEdge)
    }
}

/// KAN weight storage dtype. Centers / Gauss / knots stay FP32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MasterDtype {
    #[default]
    Fp32,
    Fp16,
}

impl MasterDtype {
    pub fn parse_name(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fp32" | "f32" => Ok(Self::Fp32),
            "fp16" | "f16" => Ok(Self::Fp16),
            other => bail!("--master {other}: expected fp32|fp16"),
        }
    }
}

/// SGD velocity storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MomDtype {
    #[default]
    Fp32,
    Q8,
}

impl MomDtype {
    pub fn parse_name(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fp32" | "f32" => Ok(Self::Fp32),
            "q8" | "int8" | "i8" => Ok(Self::Q8),
            other => bail!("--mom {other}: expected fp32|q8"),
        }
    }
}

fn default_entropy_coef() -> f64 {
    0.0
}
fn default_router_entropy_coef() -> f64 {
    0.05
}
fn default_knot_insert_every() -> usize {
    50
}
fn default_knot_ema() -> f64 {
    0.9
}
fn default_context_len() -> usize {
    crate::data::MAX_TOKEN_BUF
}
fn default_fused_grad_ckpt() -> bool {
    true
}
fn default_moe_aux() -> f64 {
    0.01
}
fn default_expert_width() -> usize {
    64
}
fn default_n_slots() -> usize {
    32
}
fn default_mem_experts() -> usize {
    4
}
fn default_window() -> usize {
    16
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            n_layers: 3,
            n_basis: 4,
            grid_start: 4,
            grid_mid: 8,
            grid_final: 12,
            seq_len: 96,
            context_len: default_context_len(),
            batch_size: 4,
            mixer: "shift".into(),
            n_heads: 1,
            vocab_size: 8192,
            lr: 3e-3,
            lr_qat: 1e-3,
            lr_harden: 3e-4,
            momentum: 0.9,
            l1: 1e-3,
            ternary_delta: 0.7,
            steps_per_epoch: 200,
            epochs_warmup: 3,
            epochs_sparsify: 2,
            epochs_qat: 4,
            epochs_harden: 2,
            max_norm: 1.0,
            seed: 7,
            ckpt_dir: "checkpoints".into(),
            log_every: 20,
            tokenizer_path: String::new(),
            data_path: "data/thinking-train.jsonl".into(),
            moe: true,
            n_experts: N_EXPERTS,
            entropy_coef: default_entropy_coef(),
            router_entropy_coef: default_router_entropy_coef(),
            knot_insert_every: default_knot_insert_every(),
            knot_ema: default_knot_ema(),
            fused_grad_ckpt: default_fused_grad_ckpt(),
            master: MasterDtype::Fp32,
            mom: MomDtype::Fp32,
            moe_topk: 0,
            moe_aux: default_moe_aux(),
            kan_factor: KanFactor::SharedEdge,
            arch: ModelArch::Kan,
            expert_width: default_expert_width(),
            n_slots: default_n_slots(),
            mem_experts: default_mem_experts(),
            window: default_window(),
            mask_output: false,
        }
    }
}

impl TrainConfig {
    /// Shared / routed split of the bump grid.
    ///
    /// G=4 → (3,1), G=8 → (6,2), G=12 → (8,4). All-shared when MoB is off.
    pub fn split_basis(&self) -> (usize, usize) {
        split_basis(self.n_basis, self.moe)
    }
}

pub fn split_basis(g: usize, moe: bool) -> (usize, usize) {
    if !moe || g < 4 {
        return (g, 0);
    }
    let routed = (g / 3).max(1);
    let routed = routed.min(g.saturating_sub(1));
    (g - routed, routed)
}

pub fn grid_target(cfg: &TrainConfig, phase: u8, epoch: usize) -> usize {
    if phase >= 3 {
        return cfg.n_basis;
    }
    if phase >= 2 {
        return cfg.grid_final;
    }
    let mid = cfg.epochs_warmup / 2;
    if epoch >= mid || epoch + 1 == cfg.epochs_warmup {
        cfg.grid_mid
    } else {
        cfg.grid_start
    }
}

/// Continuous knot scheduler: grow `G` by one every `knot_insert_every` steps
/// until `grid_final`, frozen from QAT onward. Placement is non-uniform.
pub fn next_grid_size(cfg: &TrainConfig, phase: u8, global_step: usize, current: usize) -> usize {
    if phase >= 3 || current >= cfg.grid_final {
        return current;
    }
    if cfg.knot_insert_every == 0 {
        return current;
    }
    if global_step > 0 && global_step % cfg.knot_insert_every == 0 {
        (current + 1)
            .min(cfg.grid_final)
            .min(crate::accelerate::MobKanSpec::MAX_G as usize)
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_schedule() {
        let cfg = TrainConfig {
            epochs_warmup: 3,
            grid_start: 4,
            grid_mid: 8,
            grid_final: 12,
            knot_insert_every: 50,
            n_basis: 4,
            ..TrainConfig::default()
        };
        assert_eq!(grid_target(&cfg, 1, 0), 4);
        assert_eq!(grid_target(&cfg, 1, 1), 8);
        assert_eq!(grid_target(&cfg, 1, 2), 8);
        assert_eq!(grid_target(&cfg, 2, 0), 12);
        assert_eq!(grid_target(&cfg, 3, 0), cfg.n_basis);
        assert_eq!(next_grid_size(&cfg, 1, 0, 4), 4);
        assert_eq!(next_grid_size(&cfg, 1, 50, 4), 5);
        assert_eq!(next_grid_size(&cfg, 1, 49, 4), 4);
        assert_eq!(next_grid_size(&cfg, 3, 50, 12), 12);
        assert_eq!(next_grid_size(&cfg, 1, 50, 12), 12);
    }

    #[test]
    fn split_matches_spec() {
        assert_eq!(split_basis(4, true), (3, 1));
        assert_eq!(split_basis(8, true), (6, 2));
        assert_eq!(split_basis(12, true), (8, 4));
        assert_eq!(split_basis(12, false), (12, 0));
    }
}
