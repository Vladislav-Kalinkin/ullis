//! Memory-conscious optimisers for Heron training.
//!
//! Default training is stateless clipped SGD. Lion keeps one momentum vector
//! per FP16 tensor when explicitly selected; it does not allocate a second
//! variance vector.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Optimizer choices supported by Ullis's memory contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    /// Fused clipped SGD update; no persistent optimizer allocation.
    #[default]
    StatelessSgd,
    /// One FP16 momentum value per FP16 tensor (emb, LN, bias, CMix value, Tmix).
    LionFp16,
}

impl OptimizerKind {
    /// Heron 0.10 train is clipped SGD on FP16 tensors and BinaryConnect latents.
    pub fn require_train_step(self) -> Result<()> {
        match self {
            Self::StatelessSgd => Ok(()),
            Self::LionFp16 => {
                bail!("Heron train uses stateless clipped SGD; lion_fp16 is not wired")
            }
        }
    }

    pub(crate) fn state_bytes(
        self,
        parameter_count: usize,
        latent_weight_bytes: usize,
    ) -> Result<usize> {
        match self {
            Self::LionFp16 => parameter_count
                .checked_mul(latent_weight_bytes)
                .ok_or_else(|| anyhow::anyhow!("Lion FP16 state size overflow")),
            Self::StatelessSgd => Ok(0),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LionConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

impl LionConfig {
    pub fn validate(self) -> Result<()> {
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.beta1.is_finite()
            || !self.beta2.is_finite()
            || !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
        {
            bail!("invalid Lion hyperparameters");
        }
        Ok(())
    }
}

/// Mutable Lion state. Its sole allocation is one FP32 momentum vector.
#[derive(Clone, Debug)]
pub struct Lion {
    cfg: LionConfig,
    momentum: Vec<f32>,
}

impl Lion {
    pub fn new(parameter_count: usize, cfg: LionConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            momentum: vec![0.0; parameter_count],
        })
    }

    pub fn state_bytes(&self) -> usize {
        self.momentum.len() * size_of::<f32>()
    }

    /// Applies decoupled weight decay and a sign update, then advances the
    /// single momentum vector. Inputs are checked before mutation.
    pub fn step(&mut self, parameters: &mut [f32], gradient: &[f32]) -> Result<()> {
        if parameters.len() != self.momentum.len()
            || gradient.len() != self.momentum.len()
            || parameters.iter().any(|value| !value.is_finite())
            || gradient.iter().any(|value| !value.is_finite())
        {
            bail!("Lion parameter or gradient shape/value is invalid");
        }
        for ((parameter, momentum), &gradient) in parameters
            .iter_mut()
            .zip(self.momentum.iter_mut())
            .zip(gradient)
        {
            let update = self.cfg.beta1 * *momentum + (1.0 - self.cfg.beta1) * gradient;
            *parameter *= 1.0 - self.cfg.learning_rate * self.cfg.weight_decay;
            *parameter -= self.cfg.learning_rate * update.signum();
            *momentum = self.cfg.beta2 * *momentum + (1.0 - self.cfg.beta2) * gradient;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lion_has_one_f32_state_per_parameter() {
        let lion = Lion::new(3, LionConfig::default()).unwrap();
        assert_eq!(lion.state_bytes(), 3 * size_of::<f32>());
    }

    #[test]
    fn lion_uses_sign_update_and_rejects_nan() {
        let cfg = LionConfig {
            learning_rate: 0.1,
            beta1: 0.0,
            beta2: 0.0,
            weight_decay: 0.0,
        };
        let mut lion = Lion::new(2, cfg).unwrap();
        let mut parameters = [1.0, -1.0];
        lion.step(&mut parameters, &[2.0, -3.0]).unwrap();
        assert_eq!(parameters, [0.9, -0.9]);
        assert!(lion.step(&mut parameters, &[f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn train_step_rejects_lion_until_wired() {
        assert!(OptimizerKind::StatelessSgd.require_train_step().is_ok());
        assert!(OptimizerKind::LionFp16.require_train_step().is_err());
    }

    #[test]
    fn planned_optimizer_state_is_explicit_about_its_memory_tradeoff() {
        assert_eq!(OptimizerKind::LionFp16.state_bytes(256, 2).unwrap(), 512);
        assert_eq!(
            OptimizerKind::StatelessSgd.state_bytes(1_000, 2).unwrap(),
            0
        );
    }
}
