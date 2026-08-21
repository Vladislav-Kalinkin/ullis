//! Ternary codes {-1, 0, +1} via STE, plus 2-bit pack/unpack.

use anyhow::Result;
#[cfg(debug_assertions)]
use std::cell::Cell;

use crate::accelerate::ternarize_rows;

#[cfg(debug_assertions)]
thread_local! {
    static TRAIN_STEP_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// RAII: `PackedI8Matrix::quantize` must not run during `UllisKan::train_step`.
pub struct TrainStepGuard {
    _priv: (),
}

impl TrainStepGuard {
    pub fn enter() -> Self {
        #[cfg(debug_assertions)]
        TRAIN_STEP_ACTIVE.with(|c| c.set(true));
        Self { _priv: () }
    }
}

impl Drop for TrainStepGuard {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        TRAIN_STEP_ACTIVE.with(|c| c.set(false));
    }
}

#[cfg_attr(not(debug_assertions), allow(dead_code))]
fn train_step_active() -> bool {
    #[cfg(debug_assertions)]
    {
        TRAIN_STEP_ACTIVE.with(Cell::get)
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

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

/// IEEE-754 binary16 bits. Round-to-nearest-even; used as the fp16 **storage** master.
pub fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }
    let exp16 = exp - 127 + 15;
    if exp16 >= 31 {
        return sign | 0x7c00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mant32 = mant | 0x80_0000;
        let shift = 14 - exp16;
        let mut m = mant32 >> shift;
        let round_bit = (mant32 >> (shift - 1)) & 1;
        let sticky = mant32 & ((1 << (shift - 1)) - 1);
        if round_bit == 1 && (sticky != 0 || (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let mut m = mant >> 13;
    let round_bit = (mant >> 12) & 1;
    let sticky = mant & 0x0fff;
    if round_bit == 1 && (sticky != 0 || (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            let e = exp16 + 1;
            if e >= 31 {
                return sign | 0x7c00;
            }
            return sign | ((e as u16) << 10);
        }
    }
    sign | ((exp16 as u16) << 10) | (m as u16)
}

/// Inverse of [`f32_to_f16_bits`].
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e: i32 = 127 - 15 + 1;
            while m < 0x400 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn pack_f16(src: &[f32]) -> Vec<u16> {
    src.iter().copied().map(f32_to_f16_bits).collect()
}

pub fn unpack_f16(src: &[u16]) -> Vec<f32> {
    src.iter().copied().map(f16_bits_to_f32).collect()
}

/// Snap `dst` to the fp16 lattice in place (identity after pack→unpack).
pub fn quantize_f16_in_place(dst: &mut [f32]) {
    for v in dst {
        *v = f16_bits_to_f32(f32_to_f16_bits(*v));
    }
}

/// Symmetric int8 velocity: `scale = max|v| / 127`.
pub fn quant_q8(v: &[f32]) -> (Vec<i8>, f32) {
    let mut max = 0.0f32;
    for &x in v {
        max = max.max(x.abs());
    }
    let scale = (max / 127.0).max(1e-12);
    let codes = v
        .iter()
        .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (codes, scale)
}

pub fn dequant_q8(codes: &[i8], scale: f32, out: &mut [f32]) {
    let n = codes.len().min(out.len());
    for i in 0..n {
        out[i] = f32::from(codes[i]) * scale;
    }
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

/// Rows per sparse block. Unused blocks are never allocated.
pub const I8_BLOCK_ROWS: usize = 64;
/// SIMD lane padding for packed-i8 row stride (elements).
///
/// Scaled vocabularies do not share a packed scale with a neighbouring row.
/// Default `D=32` keeps stride=32 (no extra bytes vs v0.8 dense).
pub const I8_STRIDE_ALIGN: usize = 16;
const UNMAPPED: usize = usize::MAX;

/// Symmetric per-row int8 stored as a **block-sparse strided** plane.
///
/// Logical shape `[rows, cols]`. Physical layout is live blocks of
/// [`I8_BLOCK_ROWS`] rows, each row occupying `stride ≥ cols` bytes aligned to
/// [`I8_STRIDE_ALIGN`]. Empty blocks (all-zero rows) are omitted so growing
/// `V` to 131 072+ does not densify the embedding slab.
#[derive(Clone, Debug)]
pub struct PackedI8Matrix {
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub block_rows: usize,
    /// Concatenated live blocks: each is `block_rows * stride` i8 codes.
    pub codes: Vec<i8>,
    pub scale: Vec<f32>,
    /// Logical block → byte offset in `codes`, or [`UNMAPPED`].
    map: Vec<usize>,
}

pub fn i8_row_stride(cols: usize) -> usize {
    let cols = cols.max(1);
    cols.div_ceil(I8_STRIDE_ALIGN) * I8_STRIDE_ALIGN
}

fn block_bytes(block_rows: usize, stride: usize) -> usize {
    block_rows.saturating_mul(stride)
}

fn n_blocks(rows: usize, block_rows: usize) -> usize {
    rows.div_ceil(block_rows.max(1))
}

fn block_is_live(weight: &[f32], rows: usize, cols: usize, row0: usize, block_rows: usize) -> bool {
    let end = (row0 + block_rows).min(rows);
    for r in row0..end {
        let s = r * cols;
        for j in 0..cols {
            if weight[s + j] != 0.0 {
                return true;
            }
        }
    }
    false
}

impl PackedI8Matrix {
    pub fn quantize(weight: &[f32], rows: usize, cols: usize) -> Result<Self> {
        debug_assert!(
            !train_step_active(),
            "PackedI8Matrix::quantize must not run inside UllisKan::train_step"
        );
        if weight.len() != rows.saturating_mul(cols) {
            anyhow::bail!("i8 quant len {} != {rows}*{cols}", weight.len());
        }
        let stride = i8_row_stride(cols);
        let block_rows = I8_BLOCK_ROWS;
        let nb = n_blocks(rows, block_rows);
        let mut scale = vec![1.0f32; rows];
        let mut map = vec![UNMAPPED; nb];
        let mut codes = Vec::new();
        let bb = block_bytes(block_rows, stride);
        for b in 0..nb {
            let row0 = b * block_rows;
            if !block_is_live(weight, rows, cols, row0, block_rows) {
                continue;
            }
            let off = codes.len();
            codes.resize(off + bb, 0);
            map[b] = off;
            let end = (row0 + block_rows).min(rows);
            for r in row0..end {
                let s = r * cols;
                let mut amax = 0.0f32;
                for j in 0..cols {
                    amax = amax.max(weight[s + j].abs());
                }
                let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                scale[r] = sc;
                let inv = 1.0 / sc;
                let dst = off + (r - row0) * stride;
                for j in 0..cols {
                    let q = (weight[s + j] * inv).round().clamp(-127.0, 127.0);
                    codes[dst + j] = q as i8;
                }
            }
        }
        Ok(Self {
            rows,
            cols,
            stride,
            block_rows,
            codes,
            scale,
            map,
        })
    }

    /// Allocate empty tail blocks so the logical row count can grow without
    /// rewriting live embedding rows (no semantic drift on the old lexicon).
    pub fn grow_rows(&mut self, new_rows: usize) -> Result<()> {
        if new_rows < self.rows {
            anyhow::bail!("cannot shrink i8 rows {} -> {new_rows}", self.rows);
        }
        if new_rows == self.rows {
            return Ok(());
        }
        self.scale.resize(new_rows, 1.0);
        self.rows = new_rows;
        let nb = n_blocks(self.rows, self.block_rows);
        self.map.resize(nb, UNMAPPED);
        Ok(())
    }

    pub fn is_dense(&self) -> bool {
        !self.map.is_empty() && self.map.iter().all(|&o| o != UNMAPPED)
    }

    pub fn live_blocks(&self) -> usize {
        self.map.iter().filter(|&&o| o != UNMAPPED).count()
    }

    fn row_codes(&self, row: usize) -> Option<&[i8]> {
        if row >= self.rows {
            return None;
        }
        let b = row / self.block_rows;
        let off = *self.map.get(b)?;
        if off == UNMAPPED {
            return None;
        }
        let local = row % self.block_rows;
        let start = off + local * self.stride;
        let end = start + self.cols;
        if end > self.codes.len() {
            return None;
        }
        Some(&self.codes[start..end])
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let Some(src) = self.row_codes(r) else {
                continue;
            };
            let sc = self.scale.get(r).copied().unwrap_or(1.0);
            let dst = r * self.cols;
            for j in 0..self.cols {
                out[dst + j] = f32::from(src[j]) * sc;
            }
        }
        out
    }

    pub fn lookup(&self, ids: &[u32]) -> Vec<f32> {
        let d = self.cols;
        let mut y = vec![0.0f32; ids.len() * d];
        self.lookup_into(ids, &mut y);
        y
    }

    pub fn lookup_into(&self, ids: &[u32], y: &mut [f32]) {
        let d = self.cols;
        let v = self.rows.max(1);
        let n = ids.len() * d;
        if y.len() < n {
            return;
        }
        y[..n].fill(0.0);
        for (t, &id) in ids.iter().enumerate() {
            let row = (id as usize).min(v - 1);
            let dst = &mut y[t * d..(t + 1) * d];
            let Some(src) = self.row_codes(row) else {
                continue;
            };
            let s = self.scale.get(row).copied().unwrap_or(1.0);
            for j in 0..d {
                dst[j] = f32::from(src[j]) * s;
            }
        }
    }

    /// Dense `[V, D]` u8 payload (checkpoint / Metal). Unmapped rows are zero.
    pub fn codes_u8(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rows.saturating_mul(self.cols)];
        for r in 0..self.rows {
            let Some(src) = self.row_codes(r) else {
                continue;
            };
            let dst = r * self.cols;
            for j in 0..self.cols {
                out[dst + j] = src[j] as u8;
            }
        }
        out
    }

    pub fn from_u8(rows: usize, cols: usize, codes: &[u8], scale: &[f32]) -> Result<Self> {
        if codes.len() != rows.saturating_mul(cols) {
            anyhow::bail!("i8 codes len {} != {rows}*{cols}", codes.len());
        }
        if scale.len() != rows {
            anyhow::bail!("i8 scale len {} != {rows}", scale.len());
        }
        let weight = unpack_i8_rows(
            &codes.iter().map(|&b| b as i8).collect::<Vec<_>>(),
            scale,
            rows,
            cols,
        );
        Self::quantize(&weight, rows, cols)
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
/// Unmapped blocks contribute exact zeros (empty vocab tail).
pub fn tied_logits_i8(hidden: &[f32], n: usize, d: usize, q: &PackedI8Matrix, out: &mut [f32]) {
    let v = q.rows;
    for i in 0..n {
        let h = &hidden[i * d..(i + 1) * d];
        for row in 0..v {
            let Some(codes) = q.row_codes(row) else {
                out[i * v + row] = 0.0;
                continue;
            };
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
    fn f16_roundtrip_is_lattice_idempotent() {
        for &x in &[-2.0f32, -0.5, 0.0, 0.1, 1.0, 3.5, 1e-4, 12.5] {
            let y = f16_bits_to_f32(f32_to_f16_bits(x));
            let z = f16_bits_to_f32(f32_to_f16_bits(y));
            assert_eq!(y.to_bits(), z.to_bits(), "lattice {x} -> {y} -> {z}");
        }
    }

    #[test]
    fn q8_roundtrip_scale() {
        let v = vec![0.0f32, 0.5, -0.25, 1.0, -1.0];
        let (c, s) = quant_q8(&v);
        let mut back = vec![0.0f32; v.len()];
        dequant_q8(&c, s, &mut back);
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0 / 64.0, "{a} vs {b}");
        }
    }

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
        let a = q.dequantize();
        let b = q2.dequantize();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn block_sparse_skips_empty_tail() {
        let mut w = vec![0.0f32; 192 * 4];
        for i in 0..32 {
            w[i] = 0.5;
        }
        let q = PackedI8Matrix::quantize(&w, 192, 4).unwrap();
        assert_eq!(q.rows, 192);
        assert!(q.stride >= 4);
        assert_eq!(q.live_blocks(), 1);
        assert!(!q.is_dense());
        let ids = [0u32, 100, 191];
        let y = q.lookup(&ids);
        assert!(y[..4].iter().any(|v| *v != 0.0));
        assert!(y[4..].iter().all(|v| *v == 0.0));
        let mut grown = q.clone();
        grown.grow_rows(131_072).unwrap();
        assert_eq!(grown.rows, 131_072);
        assert_eq!(grown.live_blocks(), 1);
        assert_eq!(grown.codes.len(), q.codes.len());
    }
}
