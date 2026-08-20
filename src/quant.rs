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

/// Symmetric per-row int8: `q = round(w / scale)`, `scale = max(|w|) / 127`.
#[derive(Clone, Debug)]
pub struct PackedI8Matrix {
    pub rows: usize,
    pub cols: usize,
    pub codes: Vec<i8>,
    pub scale: Vec<f32>,
}

impl PackedI8Matrix {
    pub fn quantize(weight: &[f32], rows: usize, cols: usize) -> Result<Self> {
        if weight.len() != rows.saturating_mul(cols) {
            anyhow::bail!(
                "i8 quant len {} != {rows}*{cols}",
                weight.len()
            );
        }
        let (codes, scale) = pack_i8_rows(weight, rows, cols);
        Ok(Self {
            rows,
            cols,
            codes,
            scale,
        })
    }

    pub fn dequantize(&self) -> Vec<f32> {
        unpack_i8_rows(&self.codes, &self.scale, self.rows, self.cols)
    }

    pub fn lookup(&self, ids: &[u32]) -> Vec<f32> {
        let d = self.cols;
        let v = self.rows.max(1);
        let mut y = vec![0.0f32; ids.len() * d];
        for (t, &id) in ids.iter().enumerate() {
            let row = (id as usize).min(v - 1);
            let s = self.scale[row];
            let src = &self.codes[row * d..(row + 1) * d];
            let dst = &mut y[t * d..(t + 1) * d];
            for j in 0..d {
                dst[j] = f32::from(src[j]) * s;
            }
        }
        y
    }

    pub fn codes_u8(&self) -> Vec<u8> {
        self.codes.iter().map(|&c| c as u8).collect()
    }

    pub fn from_u8(rows: usize, cols: usize, codes: &[u8], scale: &[f32]) -> Result<Self> {
        if codes.len() != rows.saturating_mul(cols) {
            anyhow::bail!("i8 codes len {} != {rows}*{cols}", codes.len());
        }
        if scale.len() != rows {
            anyhow::bail!("i8 scale len {} != {rows}", scale.len());
        }
        Ok(Self {
            rows,
            cols,
            codes: codes.iter().map(|&b| b as i8).collect(),
            scale: scale.to_vec(),
        })
    }
}

/// Pack a row-major `[rows, cols]` matrix into int8 + per-row scale.
pub fn pack_i8_rows(weight: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut codes = vec![0i8; rows * cols];
    let mut scale = vec![1.0f32; rows];
    if cols == 0 {
        return (codes, scale);
    }
    for r in 0..rows {
        let s = r * cols;
        let mut amax = 0.0f32;
        for j in 0..cols {
            amax = amax.max(weight[s + j].abs());
        }
        let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scale[r] = sc;
        let inv = 1.0 / sc;
        for j in 0..cols {
            let q = (weight[s + j] * inv).round().clamp(-127.0, 127.0);
            codes[s + j] = q as i8;
        }
    }
    (codes, scale)
}

pub fn unpack_i8_rows(codes: &[i8], scale: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = r * cols;
        let sc = scale.get(r).copied().unwrap_or(1.0);
        for j in 0..cols {
            out[s + j] = f32::from(codes[s + j]) * sc;
        }
    }
    out
}

/// `logits[n, V] = hidden[n, D] @ (codes[V, D] * scale[V])ᵀ` on the host.
pub fn tied_logits_i8(hidden: &[f32], n: usize, d: usize, q: &PackedI8Matrix, out: &mut [f32]) {
    let v = q.rows;
    for i in 0..n {
        let h = &hidden[i * d..(i + 1) * d];
        for row in 0..v {
            let codes = &q.codes[row * d..(row + 1) * d];
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += h[j] * f32::from(codes[j]);
            }
            out[i * v + row] = acc * q.scale[row];
        }
    }
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

    #[test]
    fn i8_pack_roundtrip_close() {
        let w = vec![0.0f32, 0.5, -0.25, 1.0, -1.0, 0.1];
        let q = PackedI8Matrix::quantize(&w, 2, 3).unwrap();
        let back = q.dequantize();
        for (a, b) in w.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0 / 64.0, "{a} vs {b}");
        }
        let u8s = q.codes_u8();
        let q2 = PackedI8Matrix::from_u8(2, 3, &u8s, &q.scale).unwrap();
        assert_eq!(q2.codes, q.codes);
    }
}
