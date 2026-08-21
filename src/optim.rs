//! SGD with momentum and global-norm clipping. Host `Vec<f32>` velocities;
//! no autograd tape.
//!
//! Training objective (language-agnostic):
//! `L = CE_mask + λ_H H[softmax(z)] + λ_R H[softmax(r)] + λ_1 ‖w‖_1`
//! where `H[p] = −Σ p log p` is Shannon entropy. No language tags.

use std::mem::size_of_val;

use anyhow::Result;

use crate::config::MomDtype;
use crate::mixers::masked_cross_entropy_entropy;
use crate::model::UllisKan;
use crate::quant::{dequant_q8, quant_q8};

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
    vel_q8: Option<Vec<(Vec<i8>, f32)>>,
    names: Vec<String>,
}

impl SgdMomentum {
    pub fn new(model: &UllisKan, phase: u8, lr: f64, momentum: f64, max_norm: f64) -> Result<Self> {
        let mut names = Vec::new();
        let mut vel = Vec::new();
        model.for_each_trainable(phase, |name, w, _g| {
            names.push(name.to_string());
            vel.push(vec![0.0f32; w.len()]);
        });
        let vel_q8 = if model.cfg.mom == MomDtype::Q8 {
            Some(
                vel.iter()
                    .map(|v| (vec![0i8; v.len()], 1e-12f32))
                    .collect(),
            )
        } else {
            None
        };
        Ok(Self {
            lr: lr as f32,
            momentum: momentum as f32,
            max_norm: max_norm as f32,
            vel,
            vel_q8,
            names,
        })
    }

    pub fn vel_bytes(&self) -> u64 {
        if let Some(q8) = &self.vel_q8 {
            q8.iter()
                .map(|(c, _)| size_of_val(c.as_slice()) as u64 + 4)
                .sum()
        } else {
            self.vel
                .iter()
                .map(|v| size_of_val(v.as_slice()) as u64)
                .sum()
        }
    }

    fn slots_match(&self, model: &UllisKan, phase: u8) -> bool {
        let mut i = 0usize;
        let mut ok = true;
        model.for_each_grad(phase, |name, g| {
            if i >= self.names.len()
                || self.names[i] != name
                || self.vel.get(i).map(Vec::len) != Some(g.len())
            {
                ok = false;
            }
            i += 1;
        });
        ok && i == self.names.len() && i == self.vel.len()
    }

    /// In-place two-pass SGD. Length/name mismatch (knot insert without
    /// `SgdMomentum::new`) rebuilds **zero** velocity — never pad tails.
    pub fn step(&mut self, model: &mut UllisKan, phase: u8) -> Result<()> {
        if !self.slots_match(model, phase) {
            *self = Self::new(
                model,
                phase,
                f64::from(self.lr),
                f64::from(self.momentum),
                f64::from(self.max_norm),
            )?;
        }
        let mut sq = 0.0f32;
        model.for_each_grad(phase, |_name, g| {
            for &v in g {
                sq += v * v;
            }
        });
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
        let mut i = 0usize;
        let lr = self.lr;
        let mu = self.momentum;
        let q8 = self.vel_q8.is_some();
        model.for_each_param_mut(phase, |name, w, g| {
            debug_assert_eq!(name, self.names[i].as_str());
            debug_assert_eq!(w.len(), g.len());
            let packed = {
                let vel = &mut self.vel[i];
                if vel.len() != w.len() {
                    vel.resize(w.len(), 0.0);
                }
                if q8 {
                    let (codes, vs) = &self.vel_q8.as_ref().expect("q8 slot")[i];
                    dequant_q8(codes, *vs, vel);
                }
                for j in 0..w.len() {
                    vel[j] = vel[j] * mu + g[j] * scale;
                    w[j] -= lr * vel[j];
                }
                let p = if q8 { Some(quant_q8(vel)) } else { None };
                if q8 {
                    vel.clear();
                }
                p
            };
            if let Some(p) = packed {
                self.vel_q8.as_mut().expect("q8 slot")[i] = p;
            }
            i += 1;
        });
        model.sync_grids();
        Ok(())
    }
}
