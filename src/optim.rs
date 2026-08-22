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
            Some(vel.iter().map(|v| (vec![0i8; v.len()], 1e-12f32)).collect())
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

/// SGD+momentum over a list of `(weight, grad)` slices. Used by `--arch memory`.
#[derive(Debug)]
pub struct DenseSgd {
    pub lr: f32,
    pub momentum: f32,
    pub max_norm: f32,
    vel: Vec<Vec<f32>>,
    vel_q8: Option<Vec<(Vec<i8>, f32)>>,
}

impl DenseSgd {
    pub fn new(lens: &[usize], lr: f64, momentum: f64, max_norm: f64) -> Self {
        Self::new_dtype(lens, lr, momentum, max_norm, MomDtype::Fp32)
    }

    pub fn new_dtype(
        lens: &[usize],
        lr: f64,
        momentum: f64,
        max_norm: f64,
        mom: MomDtype,
    ) -> Self {
        let vel: Vec<Vec<f32>> = lens.iter().map(|&n| vec![0.0f32; n]).collect();
        let vel_q8 = if mom == MomDtype::Q8 {
            Some(
                vel.iter()
                    .map(|v| (vec![0i8; v.len()], 1e-12f32))
                    .collect(),
            )
        } else {
            None
        };
        Self {
            lr: lr as f32,
            momentum: momentum as f32,
            max_norm: max_norm as f32,
            vel,
            vel_q8,
        }
    }

    pub fn clip_scale(&self, sq: f32) -> f32 {
        if self.max_norm > 0.0 {
            let n = sq.sqrt();
            if n > self.max_norm {
                self.max_norm / n
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    pub fn update_slice(&mut self, i: usize, w: &mut [f32], g: &[f32], scale: f32) {
        self.update_slice_kind(i, w, g, scale, true);
    }

    /// Control-plane / embedding update. Never Q8-packs this slot's velocity
    /// (per-tensor Q8 on the V×D table zeros rare-token momentum).
    pub fn update_slice_fp32(&mut self, i: usize, w: &mut [f32], g: &[f32], scale: f32) {
        self.update_slice_kind(i, w, g, scale, false);
    }

    fn update_slice_kind(
        &mut self,
        i: usize,
        w: &mut [f32],
        g: &[f32],
        scale: f32,
        allow_q8: bool,
    ) {
        if i >= self.vel.len() {
            self.vel.resize(i + 1, Vec::new());
        }
        if self.vel[i].len() != w.len() {
            self.vel[i] = vec![0.0f32; w.len()];
        }
        let lr = self.lr;
        let mu = self.momentum;
        let vel = &mut self.vel[i];
        let n = w.len().min(g.len());
        if vel.len() != w.len() {
            vel.resize(w.len(), 0.0);
        }
        let q8_on = allow_q8 && self.vel_q8.is_some();
        if q8_on {
            let q8 = self.vel_q8.as_mut().expect("q8");
            if i >= q8.len() {
                q8.resize(i + 1, (Vec::new(), 1e-12));
            }
            if q8[i].0.len() != vel.len() {
                q8[i] = (vec![0i8; vel.len()], 1e-12);
            }
            dequant_q8(&q8[i].0, q8[i].1, vel);
        }
        for j in 0..n {
            vel[j] = vel[j] * mu + g[j] * scale;
            w[j] -= lr * vel[j];
        }
        if q8_on {
            self.vel_q8.as_mut().expect("q8")[i] = quant_q8(vel);
            vel.clear();
        }
    }

    pub fn step(&mut self, params: &mut [(&mut [f32], &[f32])]) -> Result<()> {
        if params.len() != self.vel.len() {
            self.vel = params.iter().map(|(w, _)| vec![0.0f32; w.len()]).collect();
        }
        let mut sq = 0.0f32;
        for (_, g) in params.iter() {
            for &v in *g {
                sq += v * v;
            }
        }
        let scale = self.clip_scale(sq);
        for (i, (w, g)) in params.iter_mut().enumerate() {
            self.update_slice(i, w, g, scale);
        }
        Ok(())
    }
}
