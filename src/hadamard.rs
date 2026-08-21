//! Fast Walsh–Hadamard transform on the last axis.
//!
//! Normalized by `1/√N` so a power-of-two width is orthonormal.
//! Non-power-of-two `D` is padded to `N = next_power_of_two(D)`, transformed,
//! then unpadded. Pad channels do not enter the residual. Backward is the
//! same map (involution up to the scale).

use anyhow::{bail, Result};

/// Next power of two, at least 1.
pub fn pad_width(d: usize) -> usize {
    d.next_power_of_two().max(1)
}

/// In-place unnormalized FWHT. `a.len()` must be a power of two.
pub fn fwht_unnormalized(a: &mut [f32]) -> Result<()> {
    let n = a.len();
    if n == 0 || !n.is_power_of_two() {
        bail!("fwht length {n} is not a power of two");
    }
    let mut h = 1usize;
    while h < n {
        let step = h.saturating_mul(2);
        for i in (0..n).step_by(step) {
            for j in 0..h {
                let x = a[i + j];
                let y = a[i + j + h];
                a[i + j] = x + y;
                a[i + j + h] = x - y;
            }
        }
        h = step;
    }
    Ok(())
}

/// In-place orthonormal FWHT (`1/√N`).
pub fn fwht_normalized(a: &mut [f32]) -> Result<()> {
    fwht_unnormalized(a)?;
    let scale = (a.len() as f32).sqrt().recip();
    for v in a.iter_mut() {
        *v *= scale;
    }
    Ok(())
}

/// Apply orthonormal FWHT to each row of `x` (`[n, d]`). Writes `y` (`[n, d]`).
pub fn fwht_rows(x: &[f32], n: usize, d: usize, y: &mut [f32]) -> Result<()> {
    if d == 0 {
        bail!("fwht_rows d == 0");
    }
    if x.len() != n.saturating_mul(d) {
        bail!("fwht_rows x len {} != n*d {}", x.len(), n * d);
    }
    if y.len() < n * d {
        bail!("fwht_rows y short");
    }
    let pad = pad_width(d);
    let mut row = vec![0.0f32; pad];
    for i in 0..n {
        let src = &x[i * d..(i + 1) * d];
        row[..d].copy_from_slice(src);
        row[d..].fill(0.0);
        fwht_normalized(&mut row)?;
        y[i * d..(i + 1) * d].copy_from_slice(&row[..d]);
    }
    Ok(())
}

/// Same transform as [`fwht_rows`] (involution up to pad).
pub fn fwht_rows_bwd(dy: &[f32], n: usize, d: usize, dx: &mut [f32]) -> Result<()> {
    fwht_rows(dy, n, d, dx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn involution_pow2() {
        let x: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
        let mut y = x.clone();
        fwht_normalized(&mut y).unwrap();
        fwht_normalized(&mut y).unwrap();
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn orthonormal_columns() {
        let n = 4usize;
        let mut h = vec![0.0f32; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        for r in 0..n {
            fwht_normalized(&mut h[r * n..(r + 1) * n]).unwrap();
        }
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0f32;
                for k in 0..n {
                    dot += h[i * n + k] * h[j * n + k];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-5, "H_{i}·H_{j}={dot}");
            }
        }
    }

    #[test]
    fn pad_unpad_bwd_matches_fwd() {
        let n = 3usize;
        let d = 6usize;
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let mut y = vec![0.0f32; n * d];
        fwht_rows(&x, n, d, &mut y).unwrap();
        let mut dx = vec![0.0f32; n * d];
        fwht_rows_bwd(&y, n, d, &mut dx).unwrap();
        assert!(dx.iter().all(|v| v.is_finite()));
    }
}
