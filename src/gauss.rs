//! Vectorized G×G Gauss–Jordan for spline coefficient projection.

use anyhow::{anyhow, Result};

use crate::accelerate::{bump_inv_widths, relu_bumps, sgemm};

/// Solve `gram @ X = rhs` for small SPD `gram` [G, G] row-major, `rhs` [G, E].
pub fn solve_square(gram: &[f32], n: usize, rhs: &[f32], e: usize) -> Result<Vec<f32>> {
    if gram.len() != n * n {
        return Err(anyhow!("gram len {} != n*n {}", gram.len(), n * n));
    }
    if rhs.len() != n * e {
        return Err(anyhow!("rhs len {} != n*e {}", rhs.len(), n * e));
    }
    let mut g = vec![vec![0.0f32; n]; n];
    let mut b = vec![vec![0.0f32; e]; n];
    for i in 0..n {
        g[i].copy_from_slice(&gram[i * n..(i + 1) * n]);
        b[i].copy_from_slice(&rhs[i * e..(i + 1) * e]);
    }
    let x = gauss_jordan_f32(&g, &b)?;
    let mut flat = Vec::with_capacity(n * e);
    for row in &x {
        flat.extend_from_slice(row);
    }
    Ok(flat)
}

pub fn gauss_jordan_f32(gram: &[Vec<f32>], rhs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    let n = gram.len();
    if n == 0 || gram.iter().any(|r| r.len() != n) {
        return Err(anyhow!("gram must be square"));
    }
    let e = rhs[0].len();
    let mut diag_mean = 0.0f32;
    for (i, row) in gram.iter().enumerate() {
        diag_mean += row[i];
    }
    diag_mean = (diag_mean / n as f32).max(1e-8);
    let ridge = 1e-6 * diag_mean;

    let mut aug = vec![vec![0.0f32; n + e]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = gram[i][j];
            if i == j {
                aug[i][j] += ridge;
            }
        }
        for j in 0..e {
            aug[i][n + j] = rhs[i][j];
        }
    }

    for i in 0..n {
        let mut pivot = aug[i][i];
        if pivot.abs() < 1e-8 {
            pivot = 1e-8;
        }
        for j in 0..(n + e) {
            aug[i][j] /= pivot;
        }
        let row = aug[i].clone();
        for r in 0..n {
            if r == i {
                continue;
            }
            let c = aug[r][i];
            for j in 0..(n + e) {
                aug[r][j] -= c * row[j];
            }
        }
        aug[i] = row;
    }

    let mut x = vec![vec![0.0f32; e]; n];
    for i in 0..n {
        x[i].copy_from_slice(&aug[i][n..]);
    }
    Ok(x)
}

/// Least-squares lift of spline weights from `G_old` to `G_new`.
///
/// `weight_spline`: [out, in * G_old] → [out, in * G_new].
/// `old_inv_widths` / `new_inv_widths` may be length `G` or `1` (broadcast).
pub fn project_spline_coeffs(
    old_centers: &[f32],
    old_inv_widths: &[f32],
    new_centers: &[f32],
    new_inv_widths: &[f32],
    weight_spline: &[f32],
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let old_g = old_centers.len();
    let new_g = new_centers.len();
    if old_g == 0 || new_g == 0 {
        return Err(anyhow!("empty grid"));
    }
    if weight_spline.len() != out_features * in_features * old_g {
        return Err(anyhow!("spline weight len mismatch"));
    }
    let m = (new_g * 16).max(64);
    let lo = old_centers
        .iter()
        .chain(new_centers.iter())
        .copied()
        .fold(f32::INFINITY, f32::min);
    let hi = old_centers
        .iter()
        .chain(new_centers.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let pad = ((hi - lo) * 0.15).max(0.25);
    let xs: Vec<f32> = linspace(lo - pad, hi + pad, m);

    let old_iw;
    let new_iw;
    let old_iw_s: &[f32] = if old_inv_widths.is_empty() {
        old_iw = bump_inv_widths(old_centers);
        &old_iw
    } else {
        old_inv_widths
    };
    let new_iw_s: &[f32] = if new_inv_widths.is_empty() {
        new_iw = bump_inv_widths(new_centers);
        &new_iw
    } else {
        new_inv_widths
    };

    let mut psi_old = vec![0.0f32; m * old_g];
    let mut psi_new = vec![0.0f32; m * new_g];
    relu_bumps(&xs, m, 1, old_centers, old_iw_s, &mut psi_old)?;
    relu_bumps(&xs, m, 1, new_centers, new_iw_s, &mut psi_new)?;

    // b_old: [E, old_g] with E = out*in, row-major from [out, in, old_g]
    let e = out_features * in_features;
    let mut b_old = vec![0.0f32; e * old_g];
    for o in 0..out_features {
        for i in 0..in_features {
            let src = (o * in_features + i) * old_g;
            // weight is [out, in*old_g] with inner G
            let wsrc = o * (in_features * old_g) + i * old_g;
            b_old[src..src + old_g].copy_from_slice(&weight_spline[wsrc..wsrc + old_g]);
        }
    }

    // target = psi_old [m, old_g] @ b_old^T [old_g, E] → [m, E]
    let mut target = vec![0.0f32; m * e];
    sgemm(
        m,
        e,
        old_g,
        1.0,
        &psi_old,
        &transpose(&b_old, e, old_g),
        0.0,
        &mut target,
    )?;

    // gram = psi_new^T @ psi_new → [new_g, new_g]
    let psi_new_t = transpose(&psi_new, m, new_g);
    let mut gram = vec![0.0f32; new_g * new_g];
    sgemm(new_g, new_g, m, 1.0, &psi_new_t, &psi_new, 0.0, &mut gram)?;
    // rhs = psi_new^T @ target → [new_g, E]
    let mut rhs = vec![0.0f32; new_g * e];
    sgemm(new_g, e, m, 1.0, &psi_new_t, &target, 0.0, &mut rhs)?;

    let b_new_ge = solve_square(&gram, new_g, &rhs, e)?; // [new_g, E]
    let b_new = transpose(&b_new_ge, new_g, e); // [E, new_g]

    let mut out = vec![0.0f32; out_features * in_features * new_g];
    for o in 0..out_features {
        for i in 0..in_features {
            let src = (o * in_features + i) * new_g;
            let dst = o * (in_features * new_g) + i * new_g;
            out[dst..dst + new_g].copy_from_slice(&b_new[src..src + new_g]);
        }
    }
    Ok(out)
}

fn linspace(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![lo];
    }
    let step = (hi - lo) / (n as f32 - 1.0);
    (0..n).map(|i| lo + step * i as f32).collect()
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_small_spd() {
        let a = vec![vec![2.0f32, 0.5], vec![0.5, 3.0]];
        let b = vec![vec![1.0, 0.0, 2.0], vec![0.0, 1.0, 3.0]];
        let x = gauss_jordan_f32(&a, &b).unwrap();
        for col in 0..3 {
            let y0 = a[0][0] * x[0][col] + a[0][1] * x[1][col];
            let y1 = a[1][0] * x[0][col] + a[1][1] * x[1][col];
            assert!((y0 - b[0][col]).abs() < 1e-3);
            assert!((y1 - b[1][col]).abs() < 1e-3);
        }
    }
}
