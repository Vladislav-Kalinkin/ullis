//! Host SIMD / SME vector path via macOS Accelerate (BLAS + vDSP).
//!
//! Apple's runtime maps these calls onto NEON (M1–M3) and SME (M4/M5). No AMX
//! assembly is emitted from Ullis; the layout stays hardware-agnostic.
//!
//! Raw `extern "C"` lives here. Metal buffer mapping lives in `device`.

#![allow(unsafe_code)]

use anyhow::{bail, Result};

/// Launch / math descriptor for `ullis_mob_kan_fused_step`.
///
/// `#[repr(C)]` layout is shared with the MSL `constant` buffer. Integers are
/// `u32` because Metal `uint` is 32-bit. Keep this struct 16-byte friendly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MobKanSpec {
    pub n: u32,
    pub in_f: u32,
    pub out_f: u32,
    pub g: u32,
    pub gs: u32,
    pub gr: u32,
    pub k: u32,
    pub g_use: u32,
    pub phase: u32,
    pub coarse: u32,
    pub packed: u32,
    pub pad: u32,
    pub inv_width: f32,
    pub delta_ratio: f32,
}

impl MobKanSpec {
    pub const MAX_IN: u32 = 256;
    pub const MAX_G: u32 = 16;
    pub const MAX_K: u32 = 4;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        in_f: usize,
        out_f: usize,
        g: usize,
        gs: usize,
        gr: usize,
        k: usize,
        g_use: usize,
        phase: u8,
        coarse: bool,
        packed: bool,
        inv_width: f32,
        delta_ratio: f32,
    ) -> Result<Self> {
        let spec = Self {
            n: u32_dim(n, "n")?,
            in_f: u32_dim(in_f, "in_f")?,
            out_f: u32_dim(out_f, "out_f")?,
            g: u32_dim(g, "g")?,
            gs: u32_dim(gs, "gs")?,
            gr: u32_dim(gr, "gr")?,
            k: u32_dim(k, "k")?,
            g_use: u32_dim(g_use, "g_use")?,
            phase: u32::from(phase),
            coarse: u32::from(coarse),
            packed: u32::from(packed),
            pad: 0,
            inv_width,
            delta_ratio,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        if self.n == 0 {
            bail!("MobKanSpec.n == 0");
        }
        if self.in_f == 0 || self.out_f == 0 {
            bail!("MobKanSpec requires in_f, out_f > 0");
        }
        if self.in_f > Self::MAX_IN {
            bail!(
                "in_f {} exceeds threadgroup cap {}",
                self.in_f,
                Self::MAX_IN
            );
        }
        if self.g == 0 || self.g > Self::MAX_G {
            bail!("g {} out of (0, {}]", self.g, Self::MAX_G);
        }
        if self.k > Self::MAX_K {
            bail!("k {} exceeds {}", self.k, Self::MAX_K);
        }
        if self.gs == 0 {
            bail!("gs must be >= 1");
        }
        if self.gs + self.gr != self.g {
            bail!("gs + gr = {} + {} != g {}", self.gs, self.gr, self.g);
        }
        if self.g_use == 0 || self.g_use > self.gs {
            bail!("g_use {} not in 1..=gs {}", self.g_use, self.gs);
        }
        if self.gr > 0 && self.k == 0 {
            bail!("routed bumps require k > 0");
        }
        if !self.inv_width.is_finite() || self.inv_width <= 0.0 {
            bail!("inv_width must be finite and > 0");
        }
        Ok(())
    }

    pub fn n_us(&self) -> usize {
        self.n as usize
    }
    pub fn in_us(&self) -> usize {
        self.in_f as usize
    }
    pub fn out_us(&self) -> usize {
        self.out_f as usize
    }
    pub fn g_us(&self) -> usize {
        self.g as usize
    }
    pub fn gs_us(&self) -> usize {
        self.gs as usize
    }
    pub fn gr_us(&self) -> usize {
        self.gr as usize
    }
    pub fn k_us(&self) -> usize {
        self.k as usize
    }
    pub fn g_use_us(&self) -> usize {
        self.g_use as usize
    }

    pub fn x_len(&self) -> usize {
        self.n_us() * self.in_us()
    }
    pub fn y_len(&self) -> usize {
        self.n_us() * self.out_us()
    }
    pub fn w_base_len(&self) -> usize {
        self.out_us() * self.in_us()
    }
    pub fn w_shared_len(&self) -> usize {
        self.out_us() * self.in_us() * self.gs_us()
    }
    pub fn w_routed_len(&self) -> usize {
        self.k_us() * self.out_us() * self.in_us() * self.gr_us()
    }
    pub fn router_len(&self) -> usize {
        self.k_us() * self.in_us()
    }
    pub fn centers_len(&self) -> usize {
        self.g_us()
    }
    pub fn scale_vec_len(&self) -> usize {
        self.out_us()
    }
    pub fn scale_routed_len(&self) -> usize {
        let k = self.k_us().max(1);
        k * self.out_us()
    }

    pub fn quantize(&self) -> bool {
        self.phase >= 3 && self.packed == 0
    }
    pub fn use_codes(&self) -> bool {
        self.packed != 0
    }
    pub fn mask_routed(&self) -> bool {
        self.coarse != 0 || self.gr == 0 || self.k == 0
    }

    pub fn scratch_floats(&self) -> usize {
        self.in_us() + self.in_us() * self.g_us() + self.k_us().max(1)
    }
}

fn u32_dim(v: usize, name: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| anyhow::anyhow!("{name} {v} does not fit u32"))
}

// ---------------------------------------------------------------------------
// Accelerate FFI (macOS). Portable scalar fallbacks elsewhere.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[allow(non_camel_case_types)]
type vDSP_Length = usize;
#[cfg(target_os = "macos")]
#[allow(non_camel_case_types)]
type vDSP_Stride = isize;

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);
    fn vDSP_vadd(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vsub(
        b: *const f32,
        ib: vDSP_Stride,
        a: *const f32,
        ia: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vmul(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vsmul(
        a: *const f32,
        ia: vDSP_Stride,
        s: *const f32,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vsadd(
        a: *const f32,
        ia: vDSP_Stride,
        s: *const f32,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vabs(a: *const f32, ia: vDSP_Stride, c: *mut f32, ic: vDSP_Stride, n: vDSP_Length);
    fn vDSP_sve(a: *const f32, ia: vDSP_Stride, c: *mut f32, n: vDSP_Length);
    fn vDSP_maxv(a: *const f32, ia: vDSP_Stride, c: *mut f32, n: vDSP_Length);
    fn vDSP_vfill(s: *const f32, c: *mut f32, ic: vDSP_Stride, n: vDSP_Length);
    fn vDSP_dotpr(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        n: vDSP_Length,
    );
    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
}

fn i32_dim(n: usize, name: &str) -> Result<i32> {
    i32::try_from(n).map_err(|_| anyhow::anyhow!("{name} {n} does not fit i32"))
}

/// Row-major `C[m, n] = alpha * A[m, k] @ B[k, n] + beta * C`.
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    if a.len() < m.saturating_mul(k) {
        bail!("sgemm A len {} < m*k {}", a.len(), m * k);
    }
    if b.len() < k.saturating_mul(n) {
        bail!("sgemm B len {} < k*n {}", b.len(), k * n);
    }
    if c.len() < m.saturating_mul(n) {
        bail!("sgemm C len {} < m*n {}", c.len(), m * n);
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    sgemm_inner(m, n, k, alpha, a, b, beta, c)
}

/// Row-major `C[m, n] = alpha * A[m, k] @ B[n, k]^T + beta * C`.
pub fn sgemm_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    if a.len() < m.saturating_mul(k) {
        bail!("sgemm_nt A len {} < m*k {}", a.len(), m * k);
    }
    if b.len() < n.saturating_mul(k) {
        bail!("sgemm_nt B len {} < n*k {}", b.len(), n * k);
    }
    if c.len() < m.saturating_mul(n) {
        bail!("sgemm_nt C len {} < m*n {}", c.len(), m * n);
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    sgemm_nt_inner(m, n, k, alpha, a, b, beta, c)
}

#[cfg(target_os = "macos")]
fn sgemm_inner(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    let m_i = i32_dim(m, "m")?;
    let n_i = i32_dim(n, "n")?;
    let k_i = i32_dim(k, "k")?;
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m_i,
            n_i,
            k_i,
            alpha,
            a.as_ptr(),
            k_i,
            b.as_ptr(),
            n_i,
            beta,
            c.as_mut_ptr(),
            n_i,
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn sgemm_nt_inner(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    let m_i = i32_dim(m, "m")?;
    let n_i = i32_dim(n, "n")?;
    let k_i = i32_dim(k, "k")?;
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m_i,
            n_i,
            k_i,
            alpha,
            a.as_ptr(),
            k_i,
            b.as_ptr(),
            k_i,
            beta,
            c.as_mut_ptr(),
            n_i,
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn sgemm_inner(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            let idx = i * n + j;
            c[idx] = alpha * acc + beta * c[idx];
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn sgemm_nt_inner(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            let brow = j * k;
            let arow = i * k;
            for p in 0..k {
                acc += a[arow + p] * b[brow + p];
            }
            let idx = i * n + j;
            c[idx] = alpha * acc + beta * c[idx];
        }
    }
    Ok(())
}

/// `y += alpha * x`.
pub fn saxpy(alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
    if x.len() != y.len() {
        bail!("saxpy len mismatch {} vs {}", x.len(), y.len());
    }
    if x.is_empty() {
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        let n = i32_dim(x.len(), "saxpy")?;
        unsafe {
            cblas_saxpy(n, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
        Ok(())
    }
}

/// `out[i] = a[i] + b[i]`.
pub fn vadd(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    same3(a, b, out, "vadd")?;
    #[cfg(target_os = "macos")]
    unsafe {
        vDSP_vadd(a.as_ptr(), 1, b.as_ptr(), 1, out.as_mut_ptr(), 1, a.len());
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }
    Ok(())
}

/// `out[i] = a[i] - b[i]`.
pub fn vsub(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    same3(a, b, out, "vsub")?;
    #[cfg(target_os = "macos")]
    unsafe {
        // vDSP_vsub is C = A - B with argument order (B, A, C).
        vDSP_vsub(b.as_ptr(), 1, a.as_ptr(), 1, out.as_mut_ptr(), 1, a.len());
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..a.len() {
            out[i] = a[i] - b[i];
        }
    }
    Ok(())
}

/// `out[i] = a[i] * b[i]`.
pub fn vmul(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    same3(a, b, out, "vmul")?;
    #[cfg(target_os = "macos")]
    unsafe {
        vDSP_vmul(a.as_ptr(), 1, b.as_ptr(), 1, out.as_mut_ptr(), 1, a.len());
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
    }
    Ok(())
}

/// `out[i] = a[i] * s`.
pub fn vsmul(a: &[f32], s: f32, out: &mut [f32]) -> Result<()> {
    same2(a, out, "vsmul")?;
    #[cfg(target_os = "macos")]
    unsafe {
        vDSP_vsmul(a.as_ptr(), 1, &s, out.as_mut_ptr(), 1, a.len());
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..a.len() {
            out[i] = a[i] * s;
        }
    }
    Ok(())
}

/// `out[i] = a[i] + s`.
pub fn vsadd(a: &[f32], s: f32, out: &mut [f32]) -> Result<()> {
    same2(a, out, "vsadd")?;
    #[cfg(target_os = "macos")]
    unsafe {
        vDSP_vsadd(a.as_ptr(), 1, &s, out.as_mut_ptr(), 1, a.len());
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..a.len() {
            out[i] = a[i] + s;
        }
    }
    Ok(())
}

/// `out[i] = |a[i]|`.
pub fn vabs(a: &[f32], out: &mut [f32]) -> Result<()> {
    same2(a, out, "vabs")?;
    #[cfg(target_os = "macos")]
    unsafe {
        vDSP_vabs(a.as_ptr(), 1, out.as_mut_ptr(), 1, a.len());
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..a.len() {
            out[i] = a[i].abs();
        }
    }
    Ok(())
}

pub fn sum(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    #[cfg(target_os = "macos")]
    {
        let mut c = 0.0f32;
        unsafe {
            vDSP_sve(a.as_ptr(), 1, &mut c, a.len());
        }
        c
    }
    #[cfg(not(target_os = "macos"))]
    {
        a.iter().copied().sum()
    }
}

pub fn maxv(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::NEG_INFINITY;
    }
    #[cfg(target_os = "macos")]
    {
        let mut c = 0.0f32;
        unsafe {
            vDSP_maxv(a.as_ptr(), 1, &mut c, a.len());
        }
        c
    }
    #[cfg(not(target_os = "macos"))]
    {
        a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

pub fn fill(out: &mut [f32], s: f32) {
    if out.is_empty() {
        return;
    }
    #[cfg(target_os = "macos")]
    unsafe {
        vDSP_vfill(&s, out.as_mut_ptr(), 1, out.len());
    }
    #[cfg(not(target_os = "macos"))]
    {
        out.fill(s);
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        bail!("dot len mismatch {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    #[cfg(target_os = "macos")]
    {
        let mut c = 0.0f32;
        unsafe {
            vDSP_dotpr(a.as_ptr(), 1, b.as_ptr(), 1, &mut c, a.len());
        }
        Ok(c)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }
}

pub fn vs_exp(x: &[f32], y: &mut [f32]) -> Result<()> {
    same2(x, y, "exp")?;
    if x.is_empty() {
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        let n = i32_dim(x.len(), "exp")?;
        unsafe {
            vvexpf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..x.len() {
            y[i] = x[i].exp();
        }
        Ok(())
    }
}

fn same2(a: &[f32], b: &[f32], op: &str) -> Result<()> {
    if a.len() != b.len() {
        bail!("{op} len mismatch {} vs {}", a.len(), b.len());
    }
    Ok(())
}

fn same3(a: &[f32], b: &[f32], c: &[f32], op: &str) -> Result<()> {
    if a.len() != b.len() || a.len() != c.len() {
        bail!("{op} len mismatch {} / {} / {}", a.len(), b.len(), c.len());
    }
    Ok(())
}

/// In-place row softmax over the last dimension `k`. `logits` is `[n, k]`.
pub fn softmax_rows(logits: &mut [f32], n: usize, k: usize) -> Result<()> {
    if n.saturating_mul(k) != logits.len() {
        bail!("softmax_rows len {} != n*k {}", logits.len(), n * k);
    }
    if k == 0 {
        bail!("softmax_rows k == 0");
    }
    let mut shifted = vec![0.0f32; k];
    for row in 0..n {
        let s = row * k;
        let row_s = &mut logits[s..s + k];
        let m = maxv(row_s);
        for i in 0..k {
            shifted[i] = row_s[i] - m;
        }
        vs_exp(&shifted, row_s)?;
        let z = sum(row_s).max(1e-20);
        let inv = 1.0 / z;
        for v in row_s.iter_mut() {
            *v *= inv;
        }
    }
    Ok(())
}

/// Inverse bump widths from a (possibly non-uniform) ordered knot vector.
///
/// Interior knot `g` gets support `½(c_{g+1} − c_{g−1})`; endpoints use the
/// adjacent gap. Uniform linspace recovers a constant `inv_width`.
pub fn bump_inv_widths(centers: &[f32]) -> Vec<f32> {
    let g = centers.len();
    if g == 0 {
        return Vec::new();
    }
    if g == 1 {
        return vec![1.0];
    }
    let mut w = vec![0.0f32; g];
    w[0] = (centers[1] - centers[0]).abs().max(1e-4);
    w[g - 1] = (centers[g - 1] - centers[g - 2]).abs().max(1e-4);
    for i in 1..g - 1 {
        w[i] = (0.5 * (centers[i + 1] - centers[i - 1]).abs()).max(1e-4);
    }
    for v in &mut w {
        *v = 1.0 / *v;
    }
    w
}

/// Quadratic ReLU bump: `ψ_g(x) = relu(1 − |x − c_g| · inv_width_g)²`.
///
/// `x`: `[n, in]`, `centers`: `[g]`, `inv_widths`: `[g]` or `[1]` (broadcast),
/// `out`: `[n, in, g]`.
pub fn relu_bumps(
    x: &[f32],
    n: usize,
    in_f: usize,
    centers: &[f32],
    inv_widths: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let g = centers.len();
    if x.len() != n * in_f {
        bail!("relu_bumps x len {} != n*in {}", x.len(), n * in_f);
    }
    if out.len() != n * in_f * g {
        bail!(
            "relu_bumps out len {} != n*in*g {}",
            out.len(),
            n * in_f * g
        );
    }
    if inv_widths.is_empty() {
        bail!("relu_bumps inv_widths empty");
    }
    if inv_widths.len() != 1 && inv_widths.len() != g {
        bail!(
            "relu_bumps inv_widths len {} != 1 or g={g}",
            inv_widths.len()
        );
    }
    let broadcast = inv_widths.len() == 1;
    for t in 0..n {
        for i in 0..in_f {
            let xv = x[t * in_f + i];
            let base = (t * in_f + i) * g;
            for (gi, &c) in centers.iter().enumerate() {
                let inv = if broadcast {
                    inv_widths[0]
                } else {
                    inv_widths[gi]
                };
                let z = (xv - c) * inv;
                let rel = (1.0 - z.abs()).max(0.0);
                out[base + gi] = rel * rel;
            }
        }
    }
    Ok(())
}

/// TWN: `δ = ratio · mean(|row|)`, codes in `{-1,0,+1}`.
pub fn ternarize_rows(
    weight: &[f32],
    rows: usize,
    cols: usize,
    ratio: f32,
    out: &mut [f32],
) -> Result<()> {
    if weight.len() != rows * cols {
        bail!(
            "ternarize weight len {} != rows*cols {}",
            weight.len(),
            rows * cols
        );
    }
    if out.len() != weight.len() {
        bail!("ternarize out len mismatch");
    }
    if cols == 0 {
        bail!("ternarize cols == 0");
    }
    let inv = 1.0 / cols as f32;
    for r in 0..rows {
        let s = r * cols;
        let row = &weight[s..s + cols];
        let mut mean_abs = 0.0f32;
        for &w in row {
            mean_abs += w.abs();
        }
        let delta = ratio * mean_abs * inv;
        let dst = &mut out[s..s + cols];
        for (d, &w) in dst.iter_mut().zip(row.iter()) {
            *d = if w > delta {
                1.0
            } else if w < -delta {
                -1.0
            } else {
                0.0
            };
        }
    }
    Ok(())
}

fn scale_rows(mat: &mut [f32], rows: usize, cols: usize, scale: &[f32]) -> Result<()> {
    if scale.len() != rows {
        bail!("scale len {} != rows {}", scale.len(), rows);
    }
    for r in 0..rows {
        let s = scale[r];
        let row = &mut mat[r * cols..(r + 1) * cols];
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    Ok(())
}

fn prepare_weights(
    spec: &MobKanSpec,
    raw: &[f32],
    rows: usize,
    cols: usize,
    scale: &[f32],
    scratch: &mut Vec<f32>,
) -> Result<()> {
    scratch.clear();
    scratch.extend_from_slice(raw);
    if spec.quantize() {
        let mut codes = vec![0.0f32; raw.len()];
        ternarize_rows(raw, rows, cols, spec.delta_ratio, &mut codes)?;
        scratch.clear();
        scratch.extend_from_slice(&codes);
        scale_rows(scratch, rows, cols, scale)?;
    } else if spec.use_codes() {
        scale_rows(scratch, rows, cols, scale)?;
    }
    Ok(())
}

/// CPU fused MoB-KAN forward matching `kernel void ullis_mob_kan_fused_step`.
///
/// GEMM legs go through Accelerate (`cblas_sgemm` → NEON / SME). Bump
/// evaluation stays in a tight scalar loop (G ≤ 16); no extra activation
/// tensors escape this stack frame besides the caller-provided `y`.
pub fn mob_kan_fused_cpu(
    spec: &MobKanSpec,
    x: &[f32],
    w_base: &[f32],
    w_shared: &[f32],
    w_routed: &[f32],
    router: &[f32],
    centers: &[f32],
    inv_widths: &[f32],
    scale_base: &[f32],
    scale_shared: &[f32],
    scale_routed: &[f32],
    y: &mut [f32],
) -> Result<()> {
    spec.validate()?;
    if x.len() != spec.x_len() {
        bail!("x len {} != {}", x.len(), spec.x_len());
    }
    if y.len() != spec.y_len() {
        bail!("y len {} != {}", y.len(), spec.y_len());
    }
    if w_base.len() != spec.w_base_len() {
        bail!("w_base len");
    }
    if w_shared.len() != spec.w_shared_len() {
        bail!("w_shared len");
    }
    if centers.len() != spec.centers_len() {
        bail!("centers len");
    }
    if !inv_widths.is_empty()
        && inv_widths.len() != 1
        && inv_widths.len() != spec.centers_len()
    {
        bail!(
            "inv_widths len {} != 1 or g={}",
            inv_widths.len(),
            spec.centers_len()
        );
    }
    if scale_base.len() != spec.scale_vec_len() || scale_shared.len() != spec.scale_vec_len() {
        bail!("scale len");
    }

    let n = spec.n_us();
    let in_f = spec.in_us();
    let out_f = spec.out_us();
    let g = spec.g_us();
    let gs = spec.gs_us();
    let gr = spec.gr_us();
    let k = spec.k_us();
    let g_use = spec.g_use_us();

    let mut wb = Vec::new();
    let mut ws = Vec::new();
    prepare_weights(spec, w_base, out_f, in_f, scale_base, &mut wb)?;
    prepare_weights(spec, w_shared, out_f, in_f * gs, scale_shared, &mut ws)?;

    fill(y, 0.0);
    sgemm_nt(n, out_f, in_f, 1.0, x, &wb, 0.0, y)?;

    let mut bumps = vec![0.0f32; n * in_f * g];
    let iw_fallback = [spec.inv_width];
    let iw = if inv_widths.is_empty() {
        iw_fallback.as_slice()
    } else {
        inv_widths
    };
    relu_bumps(x, n, in_f, centers, iw, &mut bumps)?;

    let mut shared_b = vec![0.0f32; n * in_f * g_use];
    for t in 0..n {
        for i in 0..in_f {
            let src = (t * in_f + i) * g;
            let dst = (t * in_f + i) * g_use;
            shared_b[dst..dst + g_use].copy_from_slice(&bumps[src..src + g_use]);
        }
    }
    let mut w_shared_use = vec![0.0f32; out_f * in_f * g_use];
    for o in 0..out_f {
        for i in 0..in_f {
            let src = o * (in_f * gs) + i * gs;
            let dst = o * (in_f * g_use) + i * g_use;
            w_shared_use[dst..dst + g_use].copy_from_slice(&ws[src..src + g_use]);
        }
    }
    let mut y_s = vec![0.0f32; n * out_f];
    sgemm_nt(
        n,
        out_f,
        in_f * g_use,
        1.0,
        &shared_b,
        &w_shared_use,
        0.0,
        &mut y_s,
    )?;
    saxpy(1.0, &y_s, y)?;

    if spec.mask_routed() {
        return Ok(());
    }
    if w_routed.len() != spec.w_routed_len() {
        bail!("w_routed len");
    }
    if router.len() != spec.router_len() {
        bail!("router len");
    }
    if scale_routed.len() < spec.scale_routed_len() {
        bail!("scale_routed len");
    }

    let mut logits = vec![0.0f32; n * k];
    sgemm_nt(n, k, in_f, 1.0, x, router, 0.0, &mut logits)?;
    softmax_rows(&mut logits, n, k)?;

    let mut wr = Vec::new();
    prepare_weights(spec, w_routed, k * out_f, in_f * gr, scale_routed, &mut wr)?;

    let mut routed_b = vec![0.0f32; n * in_f * gr];
    for t in 0..n {
        for i in 0..in_f {
            let src = (t * in_f + i) * g + gs;
            let dst = (t * in_f + i) * gr;
            routed_b[dst..dst + gr].copy_from_slice(&bumps[src..src + gr]);
        }
    }

    let mut stacked = vec![0.0f32; n * k * out_f];
    sgemm_nt(
        n,
        k * out_f,
        in_f * gr,
        1.0,
        &routed_b,
        &wr,
        0.0,
        &mut stacked,
    )?;

    for t in 0..n {
        for e in 0..k {
            let gate = logits[t * k + e];
            let src = (t * k + e) * out_f;
            let dst = t * out_f;
            for o in 0..out_f {
                y[dst + o] += gate * stacked[src + o];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_nt_matches_naive() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2, we want A @ B^T
        let mut c = vec![0.0f32; 4];
        sgemm_nt(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        // row0 · row0 of B = 1*5+2*6 = 17
        // row0 · row1 of B = 1*7+2*8 = 23
        // row1 · row0 = 3*5+4*6 = 39
        // row1 · row1 = 3*7+4*8 = 53
        assert!((c[0] - 17.0).abs() < 1e-5);
        assert!((c[1] - 23.0).abs() < 1e-5);
        assert!((c[2] - 39.0).abs() < 1e-5);
        assert!((c[3] - 53.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut z = vec![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0];
        softmax_rows(&mut z, 2, 3).unwrap();
        let s0 = z[0] + z[1] + z[2];
        let s1 = z[3] + z[4] + z[5];
        assert!((s0 - 1.0).abs() < 1e-5);
        assert!((s1 - 1.0).abs() < 1e-5);
        assert!((z[3] - z[4]).abs() < 1e-5);
    }

    #[test]
    fn fused_cpu_shapes_and_finite() {
        let spec = MobKanSpec::new(2, 4, 3, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        let x = vec![0.1f32; spec.x_len()];
        let w_base = vec![0.05f32; spec.w_base_len()];
        let w_shared = vec![0.02f32; spec.w_shared_len()];
        let w_routed = vec![0.01f32; spec.w_routed_len()];
        let router = vec![0.0f32; spec.router_len()];
        let centers: Vec<f32> = (0..spec.g_us())
            .map(|i| -2.0 + 4.0 * i as f32 / (spec.g_us() as f32 - 1.0))
            .collect();
        let inv_widths = bump_inv_widths(&centers);
        let scale_base = vec![1.0f32; spec.scale_vec_len()];
        let scale_shared = vec![1.0f32; spec.scale_vec_len()];
        let scale_routed = vec![1.0f32; spec.scale_routed_len()];
        let mut y = vec![0.0f32; spec.y_len()];
        mob_kan_fused_cpu(
            &spec,
            &x,
            &w_base,
            &w_shared,
            &w_routed,
            &router,
            &centers,
            &inv_widths,
            &scale_base,
            &scale_shared,
            &scale_routed,
            &mut y,
        )
        .unwrap();
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(y.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn coarse_masks_routed() {
        // G=4 over [-2, 2] → width = 4/3, inv_width = 0.75. Routed knot sits at c=2.
        let inv = 0.75f32;
        let spec_full = MobKanSpec::new(1, 4, 2, 4, 3, 1, 3, 3, 1, false, false, inv, 0.7).unwrap();
        let spec_coarse = MobKanSpec::new(1, 4, 2, 4, 3, 1, 3, 3, 1, true, false, inv, 0.7).unwrap();
        let x = vec![1.6f32, 1.8, 2.0, 1.2];
        let w_base = vec![0.3f32; spec_full.w_base_len()];
        let w_shared = vec![0.1f32; spec_full.w_shared_len()];
        let mut w_routed = vec![0.0f32; spec_full.w_routed_len()];
        w_routed.fill(2.0);
        let router = vec![1.0f32; spec_full.router_len()];
        let centers = vec![-2.0f32, -0.66, 0.66, 2.0];
        let inv_widths = bump_inv_widths(&centers);
        let ones_o = vec![1.0f32; 2];
        let ones_r = vec![1.0f32; spec_full.scale_routed_len()];
        let mut y_full = vec![0.0f32; 2];
        let mut y_coarse = vec![0.0f32; 2];
        mob_kan_fused_cpu(
            &spec_full,
            &x,
            &w_base,
            &w_shared,
            &w_routed,
            &router,
            &centers,
            &inv_widths,
            &ones_o,
            &ones_o,
            &ones_r,
            &mut y_full,
        )
        .unwrap();
        mob_kan_fused_cpu(
            &spec_coarse,
            &x,
            &w_base,
            &w_shared,
            &w_routed,
            &router,
            &centers,
            &inv_widths,
            &ones_o,
            &ones_o,
            &ones_r,
            &mut y_coarse,
        )
        .unwrap();
        let delta = (y_full[0] - y_coarse[0]).abs() + (y_full[1] - y_coarse[1]).abs();
        assert!(delta > 1e-4, "routed path must contribute, delta={delta}");
    }
}
