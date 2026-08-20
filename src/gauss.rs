//! Vectorized G×G Gauss–Jordan for spline coefficient projection.
//!
//! Matches the Python `mps_safe_solve`: only elementwise / broadcast / cat,
//! so the 8–12 knot projection can stay on Metal (no linalg CPU fallback).

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};

/// Replace row `i` of `mat` by `row` without in-place views.
fn replace_row(mat: &Tensor, i: usize, row: &Tensor) -> Result<Tensor> {
    let n = mat.dim(0)?;
    let row = if row.rank() == 1 {
        row.unsqueeze(0)?
    } else {
        row.clone()
    };
    if n == 1 {
        return Ok(row);
    }
    if i == 0 {
        Ok(Tensor::cat(&[&row, &mat.narrow(0, 1, n - 1)?], 0)?)
    } else if i + 1 == n {
        Ok(Tensor::cat(&[&mat.narrow(0, 0, i)?, &row], 0)?)
    } else {
        Ok(Tensor::cat(
            &[
                &mat.narrow(0, 0, i)?,
                &row,
                &mat.narrow(0, i + 1, n - i - 1)?,
            ],
            0,
        )?)
    }
}

fn eye(n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    Ok(Tensor::eye(n, dtype, device)?)
}

/// Solve `gram @ X = rhs` for small SPD `gram` [G, G], `rhs` [G, E].
pub fn mps_safe_solve(gram: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    match tensor_gauss_jordan(gram, rhs) {
        Ok(x) => Ok(x),
        Err(_) => cpu_gauss_jordan(gram, rhs),
    }
}

fn tensor_gauss_jordan(gram: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let n = gram.dim(0)?;
    if gram.dim(1)? != n {
        return Err(anyhow!("gram must be square"));
    }
    let dtype = gram.dtype();
    let device = gram.device();
    let mut diag_sum = Tensor::zeros((), dtype, device)?;
    for i in 0..n {
        diag_sum = (diag_sum + gram.i((i, i))?)?;
    }
    let diag_mean = (diag_sum / n as f64)?;
    let floor = Tensor::new(1e-8f32, device)?.to_dtype(dtype)?;
    let diag_mean = diag_mean.maximum(&floor)?;
    let ridge = (diag_mean * 1e-6)?;
    let i_n = eye(n, dtype, device)?;
    let aug_l = (gram + i_n.broadcast_mul(&ridge)?)?;
    let mut aug = Tensor::cat(&[&aug_l, rhs], 1)?;
    let width = aug.dim(1)?;

    for i in 0..n {
        let pivot = aug.i((i, i))?;
        let pivot_abs = pivot.abs()?.maximum(&floor)?;
        let row = (aug.narrow(0, i, 1)? / pivot_abs.reshape(())?)?;
        // Zero the i-th elimination coefficient so the pivot row is restored.
        let col = aug.narrow(1, i, 1)?; // [n, 1]
        let mut mask = vec![1f32; n];
        mask[i] = 0.0;
        let mask_t = Tensor::from_vec(mask, (n, 1), device)?.to_dtype(dtype)?;
        let coeffs = col.broadcast_mul(&mask_t)?;
        aug = replace_row(&aug, i, &row.reshape(width)?)?;
        let delta = coeffs.broadcast_mul(&row)?;
        aug = (aug - delta)?;
        aug = replace_row(&aug, i, &row.reshape(width)?)?;
    }
    Ok(aug.narrow(1, n, aug.dim(1)? - n)?)
}

/// Host f32 Gauss–Jordan (G ≤ 16). Used when a Metal op is missing.
fn cpu_gauss_jordan(gram: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let n = gram.dim(0)?;
    let e = rhs.dim(1)?;
    let gram_c = gram.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let rhs_c = rhs.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let g = gram_c.to_vec2::<f32>()?;
    let b = rhs_c.to_vec2::<f32>()?;
    let x = gauss_jordan_f32(&g, &b)?;
    let mut flat = Vec::with_capacity(n * e);
    for row in &x {
        flat.extend_from_slice(row);
    }
    let out = Tensor::from_vec(flat, (n, e), &Device::Cpu)?;
    Ok(out.to_device(gram.device())?.to_dtype(gram.dtype())?)
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
        // Capture pivot row, then eliminate.
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

/// `ψ_g(x) = relu(1 - |x - c_g| / w)^2`. `x`: [M], `centers`: [G] → [M, G].
pub fn relu_bumps(x: &Tensor, centers: &Tensor, inv_width: f64) -> Result<Tensor> {
    // x.unsqueeze(-1) - centers  →  [M, G]
    let x_c = x.unsqueeze(D::Minus1)?;
    let z = (x_c.broadcast_sub(centers)? * inv_width)?;
    let t = (1.0 - z.abs()?)?;
    Ok(t.relu()?.sqr()?)
}

fn linspace(lo: f32, hi: f32, m: usize, device: &Device) -> Result<Tensor> {
    if m == 1 {
        return Ok(Tensor::new(&[lo][..], device)?);
    }
    let step = (hi - lo) / (m as f32 - 1.0);
    let v: Vec<f32> = (0..m).map(|i| lo + step * i as f32).collect();
    Ok(Tensor::from_vec(v, m, device)?)
}

/// Least-squares lift of `b` from G_old to G_new.
///
/// `weight_spline`: [out, in * G_old] → [out, in * G_new].
pub fn project_spline_coeffs(
    old_centers: &Tensor,
    old_inv_width: f64,
    old_g: usize,
    new_centers: &Tensor,
    new_inv_width: f64,
    new_g: usize,
    weight_spline: &Tensor,
    out_features: usize,
    in_features: usize,
) -> Result<Tensor> {
    let device = weight_spline.device();
    let dtype = DType::F32;
    let m = (new_g * 16).max(64);
    let old_c = old_centers
        .to_dtype(dtype)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let new_c = new_centers
        .to_dtype(dtype)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let lo = old_c
        .iter()
        .chain(new_c.iter())
        .copied()
        .fold(f32::INFINITY, f32::min);
    let hi = old_c
        .iter()
        .chain(new_c.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let pad = ((hi - lo) * 0.15).max(0.25);
    let xs = linspace(lo - pad, hi + pad, m, device)?;

    let oc = old_centers.to_device(device)?.to_dtype(dtype)?;
    let nc = new_centers.to_device(device)?.to_dtype(dtype)?;
    let psi_old = relu_bumps(&xs, &oc, old_inv_width)?; // [M, G_old]
    let psi_new = relu_bumps(&xs, &nc, new_inv_width)?; // [M, G_new]

    let b_old = weight_spline
        .to_dtype(dtype)?
        .reshape((out_features * in_features, old_g))?;
    let target = psi_old.matmul(&b_old.t()?)?; // [M, E]
    let gram = psi_new.t()?.matmul(&psi_new)?; // [G_new, G_new]
    let rhs = psi_new.t()?.matmul(&target)?; // [G_new, E]
    let b_new = mps_safe_solve(&gram, &rhs)?.t()?; // [E, G_new]
    Ok(b_new
        .reshape((out_features, in_features * new_g))?
        .to_dtype(weight_spline.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_small_spd() {
        let a = vec![vec![2.0f32, 0.5], vec![0.5, 3.0]];
        let b = vec![vec![1.0, 0.0, 2.0], vec![0.0, 1.0, 3.0]];
        let x = gauss_jordan_f32(&a, &b).unwrap();
        // a @ x ≈ b
        for col in 0..3 {
            let y0 = a[0][0] * x[0][col] + a[0][1] * x[1][col];
            let y1 = a[1][0] * x[0][col] + a[1][1] * x[1][col];
            assert!((y0 - b[0][col]).abs() < 1e-3);
            assert!((y1 - b[1][col]).abs() < 1e-3);
        }
    }
}
