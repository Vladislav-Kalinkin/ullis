//! SGD with momentum and global-norm clipping. Host `Vec<f32>` velocities;
//! no autograd tape.
//!
//! Training objective (language-agnostic):
//! `L = CE_mask + λ_H H[softmax(z)] + λ_R H[softmax(r)] + λ_1 ‖w‖_1`
//! where `H[p] = −Σ p log p` is Shannon entropy. No language tags.

use anyhow::Result;

use crate::mixers::masked_cross_entropy_entropy;
use crate::model::UllisKan;

/// Masked CE + logit entropy penalty. See `mixers::masked_cross_entropy_entropy`.
pub fn masked_ce_entropy(
    logits: &[f32],
    n: usize,
    v: usize,
    targets: &[u32],
    mask: &[u8],
    entropy_coef: f32,
) -> Result<(f32, f32, Vec<f32>)> {
    masked_cross_entropy_entropy(logits, n, v, targets, mask, entropy_coef)
}

pub struct SgdMomentum {
    pub lr: f32,
    pub momentum: f32,
    pub max_norm: f32,
    vel: Vec<Vec<f32>>,
    names: Vec<String>,
}

impl SgdMomentum {
    pub fn new(model: &UllisKan, phase: u8, lr: f64, momentum: f64, max_norm: f64) -> Result<Self> {
        let snap = model.trainable_snapshot(phase);
        let names = snap.iter().map(|(n, _, _)| n.clone()).collect();
        let vel = snap.iter().map(|(_, d, _)| vec![0.0f32; d.len()]).collect();
        Ok(Self {
            lr: lr as f32,
            momentum: momentum as f32,
            max_norm: max_norm as f32,
            vel,
            names,
        })
    }

    pub fn step(&mut self, model: &mut UllisKan, phase: u8) -> Result<()> {
        let snap = model.trainable_snapshot(phase);
        if snap.len() != self.names.len() {
            *self = Self::new(model, phase, f64::from(self.lr), f64::from(self.momentum), f64::from(self.max_norm))?;
            return self.step(model, phase);
        }
        let mut sq = 0.0f32;
        for (_, _, g) in &snap {
            for &v in g {
                sq += v * v;
            }
        }
        let scale = if self.max_norm > 0.0 {
            let n = sq.sqrt();
            if n > self.max_norm {
                self.max_norm / n
            } else {
                1.0
            }
        } else {
            1.0
        };
        for (i, (name, mut data, grad)) in snap.into_iter().enumerate() {
            if i >= self.vel.len() {
                self.vel.push(vec![0.0; data.len()]);
            }
            let vel = &mut self.vel[i];
            if vel.len() != data.len() {
                vel.resize(data.len(), 0.0);
            }
            for j in 0..data.len() {
                vel[j] = vel[j] * self.momentum + grad[j] * scale;
                data[j] -= self.lr * vel[j];
            }
            model.write_param(&name, &data)?;
        }
        model.sync_grids();
        Ok(())
    }
}
