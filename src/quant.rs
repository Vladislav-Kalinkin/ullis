//! Ternary codes {-1, 0, +1} via STE, plus 2-bit pack/unpack.

use anyhow::Result;

use crate::accelerate::ternarize_rows;

/// Per-output-row TWN: `δ = ratio · mean(|w_row|)`, codes in `{-1,0,+1}`.
pub fn ternarize_hard(
    weight: &[f32],
    rows: usize,
    cols: usize,
    delta_ratio: f64,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; weight.len()];
    ternarize_rows(weight, rows, cols, delta_ratio as f32, &mut out)?;
    Ok(out)
}

/// STE gate: identity on `|w| ≤ 1`, zero outside (hardtanh).
pub fn ste_gate(w: f32) -> f32 {
    if w.abs() <= 1.0 {
        1.0
    } else {
        0.0
    }
}

pub fn codes_to_i8(codes: &[f32]) -> Vec<i8> {
    codes
        .iter()
        .map(|&v| {
            if v > 0.5 {
                1
            } else if v < -0.5 {
                -1
            } else {
                0
            }
        })
        .collect()
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

    pub fn from_f32(codes: &[f32]) -> Self {
        Self::from_codes(&codes_to_i8(codes))
    }

    pub fn merge(&mut self, other: &Self, n: u32, add: u32) {
        let tot = (n + add).max(1) as f32;
        self.frac_neg = (self.frac_neg * n as f32 + other.frac_neg * add as f32) / tot;
        self.frac_zero = (self.frac_zero * n as f32 + other.frac_zero * add as f32) / tot;
        self.frac_pos = (self.frac_pos * n as f32 + other.frac_pos * add as f32) / tot;
    }
}

/// TWN-optimal per-row scale: `α = Σ w·q / Σ |q|`.
pub fn fit_scale(weight: &[f32], rows: usize, cols: usize, delta_ratio: f64) -> Result<Vec<f32>> {
    let codes = ternarize_hard(weight, rows, cols, delta_ratio)?;
    let mut scales = vec![1.0f32; rows];
    for r in 0..rows {
        let s = r * cols;
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for j in 0..cols {
            let q = codes[s + j];
            num += weight[s + j] * q;
            den += q.abs();
        }
        scales[r] = if den > 0.0 { num / den } else { 1.0 };
    }
    Ok(scales)
}

pub fn mean_abs(t: &[f32]) -> f32 {
    if t.is_empty() {
        return 0.0;
    }
    t.iter().map(|v| v.abs()).sum::<f32>() / t.len() as f32
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

    #[test]
    fn ste_gate_hardtanh() {
        assert_eq!(ste_gate(-1.5), 0.0);
        assert_eq!(ste_gate(-0.1), 1.0);
        assert_eq!(ste_gate(1.2), 0.0);
    }
}
