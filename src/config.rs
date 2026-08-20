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
            batch_size: 4,
            mixer: "shift".into(),
            n_heads: 1,
            vocab_size: 4096,
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
            data_path: "data/train.jsonl".into(),
            moe: true,
            n_experts: N_EXPERTS,
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
            ..TrainConfig::default()
        };
        assert_eq!(grid_target(&cfg, 1, 0), 4);
        assert_eq!(grid_target(&cfg, 1, 1), 8);
        assert_eq!(grid_target(&cfg, 1, 2), 8);
        assert_eq!(grid_target(&cfg, 2, 0), 12);
        assert_eq!(grid_target(&cfg, 3, 0), cfg.n_basis);
    }

    #[test]
    fn split_matches_spec() {
        assert_eq!(split_basis(4, true), (3, 1));
        assert_eq!(split_basis(8, true), (6, 2));
        assert_eq!(split_basis(12, true), (8, 4));
        assert_eq!(split_basis(12, false), (12, 0));
    }
}
