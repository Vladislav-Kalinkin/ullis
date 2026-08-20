//! SGD with momentum and global-norm clipping.
//! Candle's stock SGD has no momentum.
//!
//! Optimizer state **must** be graph-detached. Candle tensors carry a backprop
//! tape: `vel = μ·vel + g` without `detach()` keeps every previous step's
//! forward/backward graph (and every Metal buffer) reachable from `vel`.
//! That is the G=12 MoE RSS explosion (~1 GB / 20 steps).

use anyhow::Result;
use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor, Var};

pub struct SgdMomentum {
    pub lr: f64,
    pub momentum: f64,
    pub max_norm: f64,
    slots: Vec<Slot>,
}

struct Slot {
    var: Var,
    vel: Tensor,
}

impl SgdMomentum {
    pub fn new(vars: Vec<Var>, lr: f64, momentum: f64, max_norm: f64) -> Result<Self> {
        let mut slots = Vec::with_capacity(vars.len());
        for var in vars {
            let t = var.as_tensor();
            // Fresh zeros: no op, no tape.
            let vel = Tensor::zeros(t.shape(), t.dtype(), t.device())?.detach();
            slots.push(Slot { var, vel });
        }
        Ok(Self {
            lr,
            momentum,
            max_norm,
            slots,
        })
    }

    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        let mut detached: Vec<Option<Tensor>> = Vec::with_capacity(self.slots.len());
        let mut sq = 0.0f64;
        for slot in &self.slots {
            match grads.get(slot.var.as_tensor()) {
                Some(g) => {
                    let s = g
                        .sqr()?
                        .sum_all()?
                        .to_dtype(DType::F32)?
                        .to_scalar::<f32>()?;
                    sq += f64::from(s);
                    // Break the tape *before* folding g into velocity.
                    detached.push(Some(g.detach()));
                }
                None => detached.push(None),
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

        for (slot, g) in self.slots.iter_mut().zip(detached) {
            let Some(g) = g else { continue };
            let g = (g * scale)?;
            let vel = ((&slot.vel * self.momentum)? + g)?;
            slot.vel = vel.detach();
            let updated = (slot.var.as_tensor() - (&slot.vel * self.lr)?)?;
            // Var::set copies bytes in-place; the source tensor can then drop.
            slot.var.set(&updated)?;
        }
        Ok(())
    }
}
