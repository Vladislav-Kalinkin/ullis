//! Ternary codes {-1, 0, +1} via STE, plus 2-bit pack/unpack.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};

/// Per-output-row TWN threshold: δ = ratio · mean(|w_row|).
pub fn row_delta(weight: &Tensor, ratio: f64) -> Result<Tensor> {
    let dims = weight.dims();
    if dims.len() < 2 {
        let mean = weight.abs()?.mean_all()?;
        return Ok((mean * ratio)?);
    }
    let out = dims[0];
    let rest: usize = dims[1..].iter().product();
    let row = weight.reshape((out, rest))?;
    let delta = (row.abs()?.mean_keepdim(1)? * ratio)?; // [out, 1]
    let mut view = vec![out];
    view.extend(std::iter::repeat(1).take(dims.len() - 1));
    Ok(delta.reshape(view)?)
}

/// MPS-safe ternary via comparisons (no sign()*mask, which yields -0).
pub fn ternary_from_threshold(weight: &Tensor, delta: &Tensor) -> Result<Tensor> {
    let delta = delta.broadcast_as(weight.shape())?;
    let pos = weight.gt(&delta)?.to_dtype(weight.dtype())?;
    let neg_thr = delta.neg()?;
    let neg = weight.lt(&neg_thr)?.to_dtype(weight.dtype())?;
    Ok((pos - neg)?)
}

pub fn ternarize_hard(weight: &Tensor, delta_ratio: f64) -> Result<Tensor> {
    let delta = row_delta(weight, delta_ratio)?;
    ternary_from_threshold(weight, &delta)
}

/// Forward: discrete ternary. Backward: hardtanh STE (`|w| ≤ 1`).
///
/// `q = clamp(w,-1,1) - detach(clamp(w,-1,1)) + detach(ternary(w))`
pub fn ternarize_ste(weight: &Tensor, delta_ratio: f64) -> Result<Tensor> {
    let codes = ternarize_hard(weight, delta_ratio)?.detach();
    let clamped = weight.clamp(-1.0, 1.0)?;
    let identity_on_gate = (&clamped - clamped.detach())?;
    Ok((identity_on_gate + codes)?)
}

pub fn codes_to_i8(codes: &Tensor) -> Result<Vec<i8>> {
    let cpu = if codes.device().is_cpu() {
        codes.clone()
    } else {
        codes.to_device(&candle_core::Device::Cpu)?
    };
    let flat = cpu.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(flat
        .into_iter()
        .map(|v| {
            if v > 0.5 {
                1
            } else if v < -0.5 {
                -1
            } else {
                0
            }
        })
        .collect())
}

/// Pack values in {-1,0,+1} into u8, 4 codes per byte (2 bits: 0, 1, 2=-1).
pub fn pack_ternary(codes: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.len().div_ceil(4));
    let mut i = 0;
    while i < codes.len() {
        let mut byte = 0u8;
        for lane in 0..4 {
            let enc = if i + lane < codes.len() {
                let c = codes[i + lane];
                if c < 0 {
                    2u8
                } else {
                    c as u8
                }
            } else {
                0u8
            };
            byte |= (enc & 3) << (2 * lane);
        }
        out.push(byte);
        i += 4;
    }
    out
}

/// Inverse of [`pack_ternary`]. `n` is the logical number of codes (padding dropped).
pub fn unpack_ternary(packed: &[u8], n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n);
    for &byte in packed {
        for lane in 0..4 {
            if out.len() >= n {
                break;
            }
            let bits = (byte >> (2 * lane)) & 3;
            out.push(if bits == 2 { -1 } else { bits as i8 });
        }
        if out.len() >= n {
            break;
        }
    }
    out.truncate(n);
    out
}

pub fn pack_codes_tensor(codes: &Tensor) -> Result<Vec<u8>> {
    Ok(pack_ternary(&codes_to_i8(codes)?))
}

pub fn unpack_to_tensor(
    packed: &[u8],
    shape: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let codes = unpack_ternary(packed, n);
    if codes.len() != n {
        bail!("unpack length {} != {}", codes.len(), n);
    }
    let f: Vec<f32> = codes.iter().map(|&c| c as f32).collect();
    Ok(Tensor::from_vec(f, shape, device)?)
}

#[derive(Clone, Debug, Default)]
pub struct TernaryHist {
    pub frac_neg: f32,
    pub frac_zero: f32,
    pub frac_pos: f32,
}

impl TernaryHist {
    pub fn from_codes(codes: &[i8]) -> Self {
        let n = codes.len().max(1) as f32;
        let mut neg = 0u32;
        let mut zero = 0u32;
        let mut pos = 0u32;
        for &c in codes {
            match c.cmp(&0) {
                std::cmp::Ordering::Less => neg += 1,
                std::cmp::Ordering::Greater => pos += 1,
                std::cmp::Ordering::Equal => zero += 1,
            }
        }
        Self {
            frac_neg: neg as f32 / n,
            frac_zero: zero as f32 / n,
            frac_pos: pos as f32 / n,
        }
    }

    pub fn merge(&mut self, other: &Self, n: u32, add: u32) {
        let tot = (n + add).max(1) as f32;
        self.frac_neg = (self.frac_neg * n as f32 + other.frac_neg * add as f32) / tot;
        self.frac_zero = (self.frac_zero * n as f32 + other.frac_zero * add as f32) / tot;
        self.frac_pos = (self.frac_pos * n as f32 + other.frac_pos * add as f32) / tot;
    }
}

pub fn histogram_tensor(codes: &Tensor) -> Result<TernaryHist> {
    Ok(TernaryHist::from_codes(&codes_to_i8(codes)?))
}

/// TWN-optimal per-out scale: argmin_α ||w − α · ternary(w)||.
pub fn fit_scale(weight: &Tensor, delta_ratio: f64) -> Result<Tensor> {
    let codes = ternarize_hard(weight, delta_ratio)?;
    let out = weight.dim(0)?;
    let rest: usize = weight.dims()[1..].iter().product();
    let w = weight.reshape((out, rest))?;
    let c = codes.reshape((out, rest))?;
    let denom = c.abs()?.sum_keepdim(1)?.clamp(1.0, f64::MAX)?;
    let num = (w.broadcast_mul(&c))?.sum_keepdim(1)?;
    Ok(num.broadcast_div(&denom)?.reshape(out)?)
}

pub fn mean_abs(t: &Tensor) -> Result<Tensor> {
    Ok(t.abs()?.mean_all()?)
}

/// Convenience: flatten last two dims product for L1.
pub fn mean_abs_all(tensors: &[&Tensor]) -> Result<Tensor> {
    let mut acc: Option<Tensor> = None;
    let mut n = 0usize;
    for t in tensors {
        let m = t.abs()?.mean_all()?;
        acc = Some(match acc {
            None => m,
            Some(a) => (a + m)?,
        });
        n += 1;
    }
    match acc {
        Some(a) if n > 0 => Ok((a / n as f64)?),
        _ => {
            let dev = tensors
                .first()
                .map(|t| t.device().clone())
                .unwrap_or(candle_core::Device::Cpu);
            Ok(Tensor::zeros((), DType::F32, &dev)?)
        }
    }
}

pub fn last_dim_product(t: &Tensor) -> Result<usize> {
    Ok(t.dims().iter().skip(1).product())
}

#[allow(dead_code)]
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax(x, D::Minus1).map_err(|e| anyhow::anyhow!(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let t = vec![-1, 0, 1, 0, 1, 1, -1, 0];
        let p = pack_ternary(&t);
        let u = unpack_ternary(&p, t.len());
        assert_eq!(u, t);
    }

    #[test]
    fn pack_odd_length_pads() {
        let t = vec![-1i8, 0, 1];
        let p = pack_ternary(&t);
        assert_eq!(p.len(), 1);
        let u = unpack_ternary(&p, t.len());
        assert_eq!(u, t);
    }
}
