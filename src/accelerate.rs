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
    /// In-tile width (≤ `TILE_IN`). Model `in_f` may exceed this.
    pub tile_in: u32,
    /// Out-tile width. Required whenever `out_f` exceeds threadgroup width.
    pub out_tile: u32,
    /// Token-tile (CPU fwd / fused bwd occupancy). Forward Metal stays 1 token/TG.
    pub n_tile: u32,
    /// 0 = dense (all K). 1|2 = per-token top-k after full softmax.
    pub topk: u32,
    /// 0 = unfactored `[out, in·G]`. 1 = shared-edge `[in, G]` (PR8).
    pub kan_factor: u32,
    /// Layout pad so `MobKanSpec` stays 16-byte-sized (80 bytes).
    pub pad: [u32; 2],
    pub inv_width: f32,
    pub delta_ratio: f32,
}

impl MobKanSpec {
    /// Practical in-tile cap. Threadgroup scratch allows ~480 at G=16; 256 is conservative.
    pub const TILE_IN: u32 = 256;
    pub const OUT_TILE: u32 = 32;
    /// Token-tile width. Production fused-bwd launches `ceil(n / N_TILE)` TGs.
    pub const N_TILE: u32 = 16;
    /// 32 KiB threadgroup / 4 bytes. Scratch is `TIN + TIN·G + K`.
    pub const TG_SCRATCH_FLOATS: u32 = 8192;
    /// Historical name: **in-tile** cap, not model width.
    pub const MAX_IN: u32 = Self::TILE_IN;
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
        let in_f_u = u32_dim(in_f, "in_f")?;
        let out_f_u = u32_dim(out_f, "out_f")?;
        let n_u = u32_dim(n, "n")?;
        let g_u = u32_dim(g, "g")?;
        let k_u = u32_dim(k, "k")?;
        let spec = Self {
            n: n_u,
            in_f: in_f_u,
            out_f: out_f_u,
            g: g_u,
            gs: u32_dim(gs, "gs")?,
            gr: u32_dim(gr, "gr")?,
            k: k_u,
            g_use: u32_dim(g_use, "g_use")?,
            phase: u32::from(phase),
            coarse: u32::from(coarse),
            packed: u32::from(packed),
            tile_in: Self::choose_tile_in(in_f_u, g_u, k_u),
            out_tile: out_f_u.clamp(1, Self::OUT_TILE),
            n_tile: n_u.clamp(1, Self::N_TILE),
            topk: 0,
            kan_factor: 1,
            pad: [0; 2],
            inv_width,
            delta_ratio,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn choose_tile_in(in_f: u32, g: u32, k: u32) -> u32 {
        let k = k.max(1);
        let denom = 1u32.saturating_add(g).max(1);
        // Reserve TILE_IN floats for the QAT `d_base[out_f]` cache.
        let max_tin = Self::TG_SCRATCH_FLOATS
            .saturating_sub(k)
            .saturating_sub(Self::TILE_IN)
            / denom;
        in_f.min(Self::TILE_IN).min(max_tin.max(1)).max(1)
    }

    /// Test-only: force awkward tiles. Production callers use [`Self::new`].
    pub fn force_tiles(mut self, tile_in: u32, out_tile: u32, n_tile: u32) -> Result<Self> {
        self.tile_in = tile_in;
        self.out_tile = out_tile;
        self.n_tile = n_tile;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.n == 0 {
            bail!("MobKanSpec.n == 0");
        }
        if self.in_f == 0 || self.out_f == 0 {
            bail!("MobKanSpec requires in_f, out_f > 0");
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
        if self.tile_in == 0 || self.tile_in > self.in_f {
            bail!("tile_in {} not in 1..=in_f {}", self.tile_in, self.in_f);
        }
        if self.out_tile == 0 || self.out_tile > self.out_f {
            bail!("out_tile {} not in 1..=out_f {}", self.out_tile, self.out_f);
        }
        if self.n_tile == 0 || self.n_tile > self.n {
            bail!("n_tile {} not in 1..=n {}", self.n_tile, self.n);
        }
        if self.topk > 2 {
            bail!("topk {} not in 0..=2", self.topk);
        }
        if self.topk > self.k && self.k > 0 {
            bail!("topk {} exceeds k {}", self.topk, self.k);
        }
        if self.kan_factor != 1 {
            bail!(
                "kan_factor {} invalid: shared-edge (1) is required",
                self.kan_factor
            );
        }
        let _ = self.pad;
        if self.scratch_floats() > Self::TG_SCRATCH_FLOATS as usize {
            bail!(
                "threadgroup scratch {} exceeds {}",
                self.scratch_floats(),
                Self::TG_SCRATCH_FLOATS
            );
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
    pub fn tile_in_us(&self) -> usize {
        self.tile_in as usize
    }
    pub fn out_tile_us(&self) -> usize {
        self.out_tile as usize
    }
    pub fn n_tile_us(&self) -> usize {
        self.n_tile as usize
    }
    pub fn topk_us(&self) -> usize {
        self.topk as usize
    }

    /// Dense (all experts) iff `topk == 0` or `topk >= k`.
    pub fn dense_router(&self) -> bool {
        self.topk == 0 || self.k == 0 || self.topk >= self.k
    }

    pub fn shared_edge(&self) -> bool {
        self.kan_factor == 1
    }

    pub fn with_topk(mut self, topk: u32) -> Result<Self> {
        self.topk = topk;
        self.validate()?;
        Ok(self)
    }

    pub fn with_kan_factor(mut self, kan_factor: u32) -> Result<Self> {
        if kan_factor != 1 {
            bail!("unfactored KAN was removed");
        }
        self.kan_factor = 1;
        self.validate()?;
        Ok(self)
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
        if self.shared_edge() {
            self.in_us() * self.gs_us()
        } else {
            self.out_us() * self.in_us() * self.gs_us()
        }
    }
    pub fn w_routed_len(&self) -> usize {
        if self.shared_edge() {
            self.k_us() * self.in_us() * self.gr_us()
        } else {
            self.k_us() * self.out_us() * self.in_us() * self.gr_us()
        }
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
    pub fn scale_shared_len(&self) -> usize {
        if self.shared_edge() {
            self.in_us()
        } else {
            self.out_us()
        }
    }
    pub fn scale_routed_len(&self) -> usize {
        let k = self.k_us().max(1);
        if self.shared_edge() {
            k * self.in_us()
        } else {
            k * self.out_us()
        }
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

    /// Threadgroup floats for one rematerialized token (no `d_base`).
    pub fn scratch_token_floats(&self) -> usize {
        let tin = self.tile_in_us();
        let k = self.k_us().max(1);
        // x[TIN] + ψ[TIN·G] + gates[K] + φ[TIN] + ρ[K·TIN]
        tin + tin * self.g_us() + k + tin + k * tin
    }

    /// How many tokens a fused-bwd TG can rematerialize at once (2D threads).
    /// Slot 0 keeps `d_base[out]`; extra slots pack behind it. Cap 4.
    pub fn bwd_tok_par(&self) -> u32 {
        let one = self.scratch_token_floats().saturating_add(self.out_us());
        let extra = self.scratch_token_floats().max(1);
        let budget = Self::TG_SCRATCH_FLOATS as usize;
        if one > budget {
            return 1;
        }
        let par = 1 + (budget - one) / extra;
        (par.max(1).min(self.n_tile_us()).min(4)) as u32
    }

    /// Forward TG scratch: one token + `d_base[out]`. Must stay small so
    /// 384 token-TGs keep occupancy. Bwd extra slots live in [`Self::scratch_floats`].
    pub fn scratch_floats_fwd(&self) -> usize {
        self.scratch_token_floats().saturating_add(self.out_us())
    }

    pub fn scratch_floats(&self) -> usize {
        let extra = self.scratch_token_floats();
        let par = self.bwd_tok_par() as usize;
        extra
            .saturating_add(self.out_us())
            .saturating_add(extra.saturating_mul(par.saturating_sub(1)))
    }

    pub fn n_tiles(&self) -> usize {
        self.n_us().div_ceil(self.n_tile_us().max(1))
    }

    pub fn out_tiles(&self) -> usize {
        self.out_us().div_ceil(self.out_tile_us().max(1))
    }
}

/// Host/device layout of one in-tile fused-bwd partial slab.
///
/// Grid is `(n_tiles, out_tiles)`. Base / dx / dg stay unique per output or
/// token. Shared / routed / dss / dsr / dc are **one copy per TG** (threads
/// atomic-add). Reduce is a host sum.
#[derive(Clone, Copy, Debug)]
pub struct BwdPartialLayout {
    pub n_tiles: usize,
    pub out_tiles: usize,
    pub tin: usize,
    pub n_tile: usize,
    pub out_tile: usize,
    pub gs: usize,
    pub gr: usize,
    pub k: usize,
    pub g: usize,
    pub off_base: usize,
    pub off_shared: usize,
    pub off_routed: usize,
    pub off_dx: usize,
    pub off_dsb: usize,
    pub off_dss: usize,
    pub off_dsr: usize,
    pub off_dc: usize,
    pub off_dg: usize,
    pub floats: usize,
}

impl BwdPartialLayout {
    pub fn from_spec(spec: &MobKanSpec) -> Self {
        let n_tiles = spec.n_tiles();
        let out_tiles = spec.out_tiles();
        let tg = n_tiles.saturating_mul(out_tiles);
        let tin = spec.tile_in_us();
        let ot = spec.out_tile_us();
        let nt = spec.n_tile_us();
        let gs = spec.gs_us().max(1);
        let gr = spec.gr_us().max(1);
        let k = spec.k_us().max(1);
        let g = spec.g_us();
        let mut off = 0usize;
        let off_base = off;
        // Unique per output: threads do not share a base row.
        off += tg.saturating_mul(ot).saturating_mul(tin);
        let off_shared = off;
        // Compact: one [TIN, gs] per TG. Threads atomic-add (not ot copies).
        off += tg.saturating_mul(tin).saturating_mul(gs);
        let off_routed = off;
        off += tg.saturating_mul(k).saturating_mul(tin).saturating_mul(gr);
        let off_dx = off;
        off += tg.saturating_mul(nt).saturating_mul(tin);
        let off_dsb = off;
        off += tg.saturating_mul(ot);
        let off_dss = off;
        off += tg.saturating_mul(tin);
        let off_dsr = off;
        off += tg.saturating_mul(k).saturating_mul(tin);
        let off_dc = off;
        // Compact: one [G] per TG.
        off += tg.saturating_mul(g);
        let off_dg = off;
        // Unique slot per output in the tile so TG threads do not race on dg.
        off += tg.saturating_mul(nt).saturating_mul(ot).saturating_mul(k);
        Self {
            n_tiles,
            out_tiles,
            tin,
            n_tile: nt,
            out_tile: ot,
            gs,
            gr,
            k,
            g,
            off_base,
            off_shared,
            off_routed,
            off_dx,
            off_dsb,
            off_dss,
            off_dsr,
            off_dc,
            off_dg,
            floats: off,
        }
    }

    pub fn tg_index(&self, nt: usize, ot: usize) -> usize {
        nt.saturating_mul(self.out_tiles).saturating_add(ot)
    }
}

/// Mutable grad views for fused MoB-KAN backward. Lengths match `MobKanSpec`.
pub struct FusedBwdGrads<'a> {
    pub dx: &'a mut [f32],
    pub grad_base: &'a mut [f32],
    pub grad_shared: &'a mut [f32],
    pub grad_routed: &'a mut [f32],
    pub grad_router: &'a mut [f32],
    pub grad_centers: &'a mut [f32],
    pub grad_scale_base: &'a mut [f32],
    pub grad_scale_shared: &'a mut [f32],
    pub grad_scale_routed: &'a mut [f32],
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

/// Row-major `C[m, n] = alpha * A[k, m]^T @ B[k, n] + beta * C`.
/// `A` is stored `[k, m]` (e.g. `g[n_live, Vc]` with `k=n_live`, `m=Vc`).
pub fn sgemm_tn(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    if a.len() < k.saturating_mul(m) {
        bail!("sgemm_tn A len {} < k*m {}", a.len(), k * m);
    }
    if b.len() < k.saturating_mul(n) {
        bail!("sgemm_tn B len {} < k*n {}", b.len(), k * n);
    }
    if c.len() < m.saturating_mul(n) {
        bail!("sgemm_tn C len {} < m*n {}", c.len(), m * n);
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    sgemm_tn_inner(m, n, k, alpha, a, b, beta, c)
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

#[cfg(target_os = "macos")]
fn sgemm_tn_inner(
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
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            m_i,
            n_i,
            k_i,
            alpha,
            a.as_ptr(),
            m_i,
            b.as_ptr(),
            n_i,
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

#[cfg(not(target_os = "macos"))]
fn sgemm_tn_inner(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<()> {
    // A is [k, m], B is [k, n], C is [m, n]; C += A^T @ B
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[p * m + i] * b[p * n + j];
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

/// Zero non-selected experts per token. `topk == 0` or `topk >= k` is a no-op
/// (dense, bit-identical). Ties keep the lowest index.
pub fn apply_topk_gates(gates: &mut [f32], n: usize, k: usize, topk: u32) {
    if k == 0 || k > 4 {
        return;
    }
    let tk = topk as usize;
    if tk == 0 || tk >= k {
        return;
    }
    if gates.len() < n.saturating_mul(k) {
        return;
    }
    for t in 0..n {
        let row = &mut gates[t * k..t * k + k];
        let mut used = 0u32;
        for _ in 0..tk {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for e in 0..k {
                if (used >> e) & 1 == 1 {
                    continue;
                }
                if row[e] > bv {
                    bv = row[e];
                    best = e;
                }
            }
            used |= 1 << best;
        }
        for e in 0..k {
            if (used >> e) & 1 == 0 {
                row[e] = 0.0;
            }
        }
    }
}

/// Switch load-balance: `α · K · Σ_e f_e P_e`. `P` is mean full-softmax;
/// `f` is the fraction of tokens that selected expert `e`. Returns `(aux, dP)`.
pub fn switch_aux(full: &[f32], sparse: &[f32], n: usize, k: usize, alpha: f32) -> (f32, [f32; 4]) {
    let mut p = [0.0f32; 4];
    let mut f = [0.0f32; 4];
    if n == 0 || k == 0 || alpha == 0.0 {
        return (0.0, p);
    }
    let inv = 1.0 / n as f32;
    for t in 0..n {
        for e in 0..k {
            p[e] += full[t * k + e] * inv;
            if sparse[t * k + e] > 0.0 {
                f[e] += inv;
            }
        }
    }
    let nk = k as f32;
    let mut aux = 0.0f32;
    let mut dp = [0.0f32; 4];
    for e in 0..k {
        aux += f[e] * p[e];
        dp[e] = alpha * nk * f[e];
    }
    (aux * alpha * nk, dp)
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

/// `∂ψ/∂x` and `∂ψ/∂c` for `ψ = relu(1 − |x−c|·inv)²`.
pub fn bump_grads(x: f32, c: f32, inv: f32, dpsi: f32, dx: &mut f32, dc: &mut f32) {
    let z = (x - c) * inv;
    let u = 1.0 - z.abs();
    if u <= 0.0 {
        return;
    }
    let du = 2.0 * u * dpsi;
    let sgn = if x >= c { 1.0 } else { -1.0 };
    *dx += du * (-inv * sgn);
    *dc += du * (inv * sgn);
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

/// One TWN row; identical to a single iteration of [`ternarize_rows`].
pub fn ternarize_row(row: &[f32], ratio: f32, out: &mut [f32]) {
    let cols = row.len();
    if cols == 0 || out.len() < cols {
        return;
    }
    let mut mean_abs = 0.0f32;
    for &w in row {
        mean_abs += w.abs();
    }
    let delta = ratio * mean_abs / cols as f32;
    for (d, &w) in out[..cols].iter_mut().zip(row.iter()) {
        *d = if w > delta {
            1.0
        } else if w < -delta {
            -1.0
        } else {
            0.0
        };
    }
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

fn pack_x_tile(
    x: &[f32],
    in_f: usize,
    t0: usize,
    tn: usize,
    in0: usize,
    tin: usize,
    dst: &mut [f32],
) {
    for t in 0..tn {
        let src = (t0 + t) * in_f + in0;
        dst[t * tin..t * tin + tin].copy_from_slice(&x[src..src + tin]);
    }
}

fn pack_base_tile(
    w: &[f32],
    in_f: usize,
    o0: usize,
    ot: usize,
    in0: usize,
    tin: usize,
    dst: &mut [f32],
) {
    for o in 0..ot {
        let src = (o0 + o) * in_f + in0;
        dst[o * tin..o * tin + tin].copy_from_slice(&w[src..src + tin]);
    }
}

fn add_y_tile(
    y: &mut [f32],
    out_f: usize,
    t0: usize,
    tn: usize,
    o0: usize,
    ot: usize,
    tile: &[f32],
) {
    for t in 0..tn {
        let dst = (t0 + t) * out_f + o0;
        let src = t * ot;
        for o in 0..ot {
            y[dst + o] += tile[src + o];
        }
    }
}

/// CPU fused MoB-KAN forward matching `kernel void ullis_mob_kan_fused_step`.
///
/// GEMM legs go through Accelerate (`cblas_sgemm` → NEON / SME) over `TIN`
/// (and `N_TILE` / `OUT_TILE`). Bump eval is `n_tile × TIN × G`, never
/// `n × in × G`.
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
    if !inv_widths.is_empty() && inv_widths.len() != 1 && inv_widths.len() != spec.centers_len() {
        bail!(
            "inv_widths len {} != 1 or g={}",
            inv_widths.len(),
            spec.centers_len()
        );
    }
    if scale_base.len() != spec.scale_vec_len() || scale_shared.len() != spec.scale_shared_len() {
        bail!("scale len");
    }

    mob_kan_fused_cpu_shared_edge(
        spec,
        x,
        w_base,
        w_shared,
        w_routed,
        router,
        centers,
        inv_widths,
        scale_base,
        scale_shared,
        scale_routed,
        y,
    )
}

/// Shared-edge forward: one univariate spline per incoming edge, then `W_base`.
///
/// `φ_i = Σ_g Q(W_shared[i,g]) ψ_g(x_i)`
/// `ρ_{k,i} = Σ_g Q(W_routed[k,i,g]) ψ_{gs+g}(x_i)`
/// `y = Q(W_base) (x + φ + Σ_k g_k ρ_k)`
fn mob_kan_fused_cpu_shared_edge(
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
    let n = spec.n_us();
    let in_f = spec.in_us();
    let out_f = spec.out_us();
    let g = spec.g_us();
    let gs = spec.gs_us();
    let gr = spec.gr_us();
    let k = spec.k_us();
    let g_use = spec.g_use_us();
    let tin_max = spec.tile_in_us();
    let ot_max = spec.out_tile_us();
    let nt_max = spec.n_tile_us();

    let mut wb = Vec::new();
    let mut ws = Vec::new();
    prepare_weights(spec, w_base, out_f, in_f, scale_base, &mut wb)?;
    prepare_weights(spec, w_shared, in_f, gs.max(1), scale_shared, &mut ws)?;

    fill(y, 0.0);

    let routed = !spec.mask_routed();
    let mut logits = Vec::new();
    let mut wr = Vec::new();
    if routed {
        if w_routed.len() != spec.w_routed_len() {
            bail!("w_routed len");
        }
        if router.len() != spec.router_len() {
            bail!("router len");
        }
        if scale_routed.len() < spec.scale_routed_len() {
            bail!("scale_routed len");
        }
        logits.resize(n * k.max(1), 0.0);
        sgemm_nt(n, k, in_f, 1.0, x, router, 0.0, &mut logits)?;
        softmax_rows(&mut logits, n, k)?;
        apply_topk_gates(&mut logits, n, k, spec.topk);
        prepare_weights(
            spec,
            w_routed,
            k.max(1) * in_f,
            gr.max(1),
            scale_routed,
            &mut wr,
        )?;
    }

    let iw_fallback = [spec.inv_width];
    let iw = if inv_widths.is_empty() {
        iw_fallback.as_slice()
    } else {
        inv_widths
    };

    let mut x_tile = vec![0.0f32; nt_max * tin_max];
    let mut bumps = vec![0.0f32; nt_max * tin_max * g];
    let mut u_tile = vec![0.0f32; nt_max * tin_max];
    let mut wb_tile = vec![0.0f32; ot_max * tin_max];
    let mut y_tile = vec![0.0f32; nt_max * ot_max];

    let mut t0 = 0usize;
    while t0 < n {
        let tn = nt_max.min(n - t0);
        let mut in0 = 0usize;
        while in0 < in_f {
            let tin = tin_max.min(in_f - in0);
            pack_x_tile(x, in_f, t0, tn, in0, tin, &mut x_tile);
            relu_bumps(
                &x_tile[..tn * tin],
                tn,
                tin,
                centers,
                iw,
                &mut bumps[..tn * tin * g],
            )?;
            for t in 0..tn {
                for i in 0..tin {
                    let src = (t * tin + i) * g;
                    let mut phi = 0.0f32;
                    let gi_in = in0 + i;
                    for gi in 0..g_use {
                        phi += bumps[src + gi] * ws[gi_in * gs + gi];
                    }
                    let mut rho = 0.0f32;
                    if routed && gr > 0 {
                        for e in 0..k {
                            let gate = logits[(t0 + t) * k + e];
                            if gate == 0.0 {
                                continue;
                            }
                            let mut mix = 0.0f32;
                            let row = (e * in_f + gi_in) * gr;
                            for gi in 0..gr {
                                mix += bumps[src + gs + gi] * wr[row + gi];
                            }
                            rho += gate * mix;
                        }
                    }
                    u_tile[t * tin + i] = x_tile[t * tin + i] + phi + rho;
                }
            }

            let mut o0 = 0usize;
            while o0 < out_f {
                let ot = ot_max.min(out_f - o0);
                pack_base_tile(&wb, in_f, o0, ot, in0, tin, &mut wb_tile);
                y_tile[..tn * ot].fill(0.0);
                sgemm_nt(tn, ot, tin, 1.0, &u_tile, &wb_tile, 0.0, &mut y_tile)?;
                add_y_tile(y, out_f, t0, tn, o0, ot, &y_tile);
                o0 += ot;
            }
            in0 += tin;
        }
        t0 += tn;
    }
    Ok(())
}

fn qat_and_ste(w: f32, delta: f32, scale: f32, qat: bool, packed: bool) -> (f32, f32) {
    if packed {
        return (w * scale, 1.0);
    }
    if qat {
        let code = if w > delta {
            1.0
        } else if w < -delta {
            -1.0
        } else {
            0.0
        };
        let ste = if w.abs() <= 1.0 { 1.0 } else { 0.0 };
        (code * scale, ste)
    } else {
        (w, 1.0)
    }
}

fn row_delta(row: &[f32], ratio: f32) -> f32 {
    if row.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f32;
    for &w in row {
        s += w.abs();
    }
    ratio * s / row.len() as f32
}

fn inv_at(inv_widths: &[f32], gi: usize, fallback: f32) -> f32 {
    let v = if inv_widths.len() == 1 {
        inv_widths[0]
    } else {
        inv_widths.get(gi).copied().unwrap_or(fallback)
    };
    if v.is_finite() && v > 0.0 {
        v
    } else {
        fallback
    }
}

/// Tiled CPU fused backward. Oracle is host `TernaryKanLinear::backward_into`.
///
/// Token / in / out tiles match Metal (`N_TILE`, `TIN`, `OUT_TILE`). Router
/// Jacobian + `λ_R` run on the host after the KAN dW/dX/dCenters pass.
pub fn mob_kan_fused_bwd_cpu(
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
    dy: &[f32],
    lambda_r: f32,
    aux_coef: f32,
    grads: FusedBwdGrads<'_>,
) -> Result<(f32, f32)> {
    spec.validate()?;
    if spec.packed != 0 {
        grads.dx[..spec.x_len()].fill(0.0);
        return Ok((0.0, 0.0));
    }
    if x.len() != spec.x_len() || dy.len() != spec.y_len() {
        bail!("fused bwd x/dy rank");
    }
    if grads.dx.len() < spec.x_len() {
        bail!("fused bwd dx short");
    }
    grads.dx[..spec.x_len()].fill(0.0);
    mob_kan_fused_bwd_cpu_shared_edge(
        spec,
        x,
        w_base,
        w_shared,
        w_routed,
        router,
        centers,
        inv_widths,
        scale_base,
        scale_shared,
        scale_routed,
        dy,
        lambda_r,
        aux_coef,
        grads,
    )
}

fn mob_kan_fused_bwd_cpu_shared_edge(
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
    dy: &[f32],
    lambda_r: f32,
    aux_coef: f32,
    grads: FusedBwdGrads<'_>,
) -> Result<(f32, f32)> {
    let n = spec.n_us();
    let in_f = spec.in_us();
    let out_f = spec.out_us();
    let g = spec.g_us();
    let gs = spec.gs_us();
    let gr = spec.gr_us();
    let k = spec.k_us();
    let g_use = spec.g_use_us();
    let qat = spec.quantize();
    let packed = spec.use_codes();
    let scale_on = qat || packed;
    let ratio = spec.delta_ratio;
    let routed = !spec.mask_routed();
    let dx = &mut grads.dx[..spec.x_len()];

    let mut gates = vec![0.0f32; n * k.max(1)];
    if routed {
        if w_routed.len() != spec.w_routed_len() || router.len() != spec.router_len() {
            bail!("fused bwd routed weights");
        }
        sgemm_nt(n, k, in_f, 1.0, x, router, 0.0, &mut gates)?;
        softmax_rows(&mut gates, n, k)?;
    }
    let mut mix_gates = gates.clone();
    apply_topk_gates(&mut mix_gates, n, k, spec.topk);

    let mut d_base = vec![0.0f32; out_f];
    let mut d_sh = vec![0.0f32; in_f];
    let mut d_rt = vec![0.0f32; k.max(1) * in_f];
    if qat {
        for o in 0..out_f {
            d_base[o] = row_delta(&w_base[o * in_f..(o + 1) * in_f], ratio);
        }
        for i in 0..in_f {
            d_sh[i] = row_delta(&w_shared[i * gs..(i + 1) * gs], ratio);
            if routed && gr > 0 {
                for e in 0..k {
                    let row = (e * in_f + i) * gr;
                    d_rt[e * in_f + i] = row_delta(&w_routed[row..row + gr], ratio);
                }
            }
        }
    }

    let tin_max = spec.tile_in_us();
    let nt_max = spec.n_tile_us();
    let mut bumps = vec![0.0f32; nt_max * tin_max * g];
    let mut x_tile = vec![0.0f32; nt_max * tin_max];
    let mut u_tile = vec![0.0f32; nt_max * tin_max];
    let mut q_s = vec![0.0f32; tin_max * gs.max(1)];
    let mut q_r = vec![0.0f32; k.max(1) * tin_max * gr.max(1)];
    let mut rho = vec![0.0f32; k.max(1) * tin_max];
    let mut d_u = vec![0.0f32; tin_max];
    let mut dg = vec![0.0f32; n * k.max(1)];

    let mut t0 = 0usize;
    while t0 < n {
        let tn = nt_max.min(n - t0);
        let mut in0 = 0usize;
        while in0 < in_f {
            let tin = tin_max.min(in_f - in0);
            pack_x_tile(x, in_f, t0, tn, in0, tin, &mut x_tile);
            relu_bumps(
                &x_tile[..tn * tin],
                tn,
                tin,
                centers,
                inv_widths,
                &mut bumps[..tn * tin * g],
            )?;
            for lt in 0..tn {
                let t = t0 + lt;
                for i in 0..tin {
                    let gi_in = in0 + i;
                    let src = (lt * tin + i) * g;
                    let ssi = if scale_on { scale_shared[gi_in] } else { 1.0 };
                    let mut phi = 0.0f32;
                    for gi in 0..g_use {
                        let (qs, _) =
                            qat_and_ste(w_shared[gi_in * gs + gi], d_sh[gi_in], ssi, qat, packed);
                        q_s[i * gs + gi] = qs;
                        phi += bumps[src + gi] * qs;
                    }
                    let mut rsum = 0.0f32;
                    if routed && gr > 0 {
                        for e in 0..k {
                            let gate = mix_gates[t * k + e];
                            let sri = if scale_on {
                                scale_routed[e * in_f + gi_in]
                            } else {
                                1.0
                            };
                            let mut mix = 0.0f32;
                            let row = (e * in_f + gi_in) * gr;
                            for gi in 0..gr {
                                let (qr, _) = qat_and_ste(
                                    w_routed[row + gi],
                                    d_rt[e * in_f + gi_in],
                                    sri,
                                    qat,
                                    packed,
                                );
                                q_r[(e * tin + i) * gr + gi] = qr;
                                mix += bumps[src + gs + gi] * qr;
                            }
                            rho[e * tin + i] = mix;
                            rsum += gate * mix;
                        }
                    }
                    u_tile[lt * tin + i] = x_tile[lt * tin + i] + phi + rsum;
                }

                d_u[..tin].fill(0.0);
                for o in 0..out_f {
                    let go = dy[t * out_f + o];
                    let sbo = if scale_on { scale_base[o] } else { 1.0 };
                    let db = d_base[o];
                    for i in 0..tin {
                        let gi_in = in0 + i;
                        let w = w_base[o * in_f + gi_in];
                        let (qw, ste) = qat_and_ste(w, db, sbo, qat, packed);
                        dx[t * in_f + gi_in] += go * qw;
                        grads.grad_base[o * in_f + gi_in] += go * u_tile[lt * tin + i] * sbo * ste;
                        if scale_on {
                            let code = if sbo.abs() > 0.0 { qw / sbo } else { 0.0 };
                            grads.grad_scale_base[o] += go * u_tile[lt * tin + i] * code;
                        }
                        d_u[i] += go * qw;
                    }
                }

                for i in 0..tin {
                    let gi_in = in0 + i;
                    let xv = x_tile[lt * tin + i];
                    let src = (lt * tin + i) * g;
                    let ssi = if scale_on { scale_shared[gi_in] } else { 1.0 };
                    let du = d_u[i];
                    for gi in 0..g_use {
                        let idx = gi_in * gs + gi;
                        let b = bumps[src + gi];
                        let (_, ste_s) = qat_and_ste(w_shared[idx], d_sh[gi_in], ssi, qat, packed);
                        grads.grad_shared[idx] += du * b * ssi * ste_s;
                        if scale_on {
                            let qs = q_s[i * gs + gi];
                            let code = if ssi.abs() > 0.0 { qs / ssi } else { 0.0 };
                            grads.grad_scale_shared[gi_in] += du * b * code;
                        }
                        let inv = inv_at(inv_widths, gi, spec.inv_width);
                        bump_grads(
                            xv,
                            centers[gi],
                            inv,
                            du * q_s[i * gs + gi],
                            &mut dx[t * in_f + gi_in],
                            &mut grads.grad_centers[gi],
                        );
                    }
                    if !routed || gr == 0 {
                        continue;
                    }
                    for e in 0..k {
                        let gate = mix_gates[t * k + e];
                        dg[t * k + e] += du * rho[e * tin + i];
                        if gate == 0.0 {
                            continue;
                        }
                        let sri = if scale_on {
                            scale_routed[e * in_f + gi_in]
                        } else {
                            1.0
                        };
                        let dr = d_rt[e * in_f + gi_in];
                        for gi in 0..gr {
                            let idx = (e * in_f + gi_in) * gr + gi;
                            let b = bumps[src + gs + gi];
                            let qr = q_r[(e * tin + i) * gr + gi];
                            let (_, ste_r) = qat_and_ste(w_routed[idx], dr, sri, qat, packed);
                            grads.grad_routed[idx] += du * gate * b * sri * ste_r;
                            if scale_on {
                                let code = if sri.abs() > 0.0 { qr / sri } else { 0.0 };
                                grads.grad_scale_routed[e * in_f + gi_in] += du * gate * b * code;
                            }
                            let inv = inv_at(inv_widths, gs + gi, spec.inv_width);
                            bump_grads(
                                xv,
                                centers[gs + gi],
                                inv,
                                du * gate * qr,
                                &mut dx[t * in_f + gi_in],
                                &mut grads.grad_centers[gs + gi],
                            );
                        }
                    }
                }
            }
            in0 += tin;
        }
        t0 += tn;
    }

    let mut h_sum = 0.0f32;
    let (aux, dp) = if routed && aux_coef > 0.0 && !spec.dense_router() {
        switch_aux(&gates, &mix_gates, n, k, aux_coef)
    } else {
        (0.0, [0.0f32; 4])
    };
    let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f32 };
    if routed && k > 0 {
        for t in 0..n {
            let gg = &gates[t * k..t * k + k];
            let d = &dg[t * k..t * k + k];
            let dot: f32 = gg.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
            let mut dlogit = [0.0f32; 4];
            for e in 0..k {
                dlogit[e] = gg[e] * (d[e] - dot);
            }
            if lambda_r > 0.0 && k > 1 {
                let mut h = 0.0f32;
                for e in 0..k {
                    let p = gg[e].max(1e-12);
                    h -= p * p.ln();
                }
                h_sum += h;
                for e in 0..k {
                    let p = gg[e].max(1e-12);
                    dlogit[e] += lambda_r * (-p * (p.ln() + h));
                }
            }
            if aux_coef > 0.0 && !spec.dense_router() {
                let mut daux = [0.0f32; 4];
                for e in 0..k {
                    daux[e] = dp[e] * inv_n;
                }
                let da_dot: f32 = gg.iter().zip(daux.iter()).map(|(a, b)| a * b).sum();
                for e in 0..k {
                    dlogit[e] += gg[e] * (daux[e] - da_dot);
                }
            }
            for e in 0..k {
                for i in 0..in_f {
                    grads.grad_router[e * in_f + i] += dlogit[e] * x[t * in_f + i];
                    dx[t * in_f + i] += dlogit[e] * router[e * in_f + i];
                }
            }
        }
    }
    let ent = if lambda_r > 0.0 && k > 1 && routed {
        h_sum / n as f32
    } else {
        0.0
    };
    Ok((ent, aux))
}

/// Fold one in-tile of fused-bwd partials into dense grads / `dx`.
pub fn reduce_bwd_partials(
    spec: &MobKanSpec,
    layout: &BwdPartialLayout,
    in0: usize,
    tin: usize,
    part: &[f32],
    grads: &mut FusedBwdGrads<'_>,
) -> Result<()> {
    if part.len() < layout.floats {
        bail!("bwd partials {} < {}", part.len(), layout.floats);
    }
    let n = spec.n_us();
    let in_f = spec.in_us();
    let out_f = spec.out_us();
    let gs = spec.gs_us();
    let gr = spec.gr_us();
    let k = spec.k_us();
    let g = spec.g_us();
    let routed = !spec.mask_routed();
    let nt_max = layout.n_tile;
    let ot_max = layout.out_tile;
    let tin_cap = layout.tin;

    for nt in 0..layout.n_tiles {
        let t0 = nt * nt_max;
        let tn = nt_max.min(n.saturating_sub(t0));
        for ot in 0..layout.out_tiles {
            let o0 = ot * ot_max;
            let otn = ot_max.min(out_f.saturating_sub(o0));
            let tg = layout.tg_index(nt, ot);
            for lo in 0..otn {
                let o = o0 + lo;
                let base_row = layout.off_base + (tg * ot_max + lo) * tin_cap;
                saxpy(
                    1.0,
                    &part[base_row..base_row + tin],
                    &mut grads.grad_base[o * in_f + in0..o * in_f + in0 + tin],
                )?;
                grads.grad_scale_base[o] += part[layout.off_dsb + tg * ot_max + lo];
            }
            if gs > 0 {
                let sh_row = layout.off_shared + tg * tin_cap * layout.gs;
                saxpy(
                    1.0,
                    &part[sh_row..sh_row + tin * layout.gs],
                    &mut grads.grad_shared[(in0 * gs)..(in0 + tin) * gs],
                )?;
            }
            let dss = layout.off_dss + tg * tin_cap;
            saxpy(
                1.0,
                &part[dss..dss + tin],
                &mut grads.grad_scale_shared[in0..in0 + tin],
            )?;
            if routed {
                for e in 0..k {
                    let rr = layout.off_routed + ((tg * layout.k + e) * tin_cap) * layout.gr;
                    let gr_a = gr.max(1);
                    saxpy(
                        1.0,
                        &part[rr..rr + tin * layout.gr],
                        &mut grads.grad_routed
                            [((e * in_f + in0) * gr_a)..(e * in_f + in0 + tin) * gr_a],
                    )?;
                    let dsr = layout.off_dsr + (tg * layout.k + e) * tin_cap;
                    saxpy(
                        1.0,
                        &part[dsr..dsr + tin],
                        &mut grads.grad_scale_routed[e * in_f + in0..e * in_f + in0 + tin],
                    )?;
                }
            }
            for lt in 0..tn {
                let t = t0 + lt;
                let dx_row = layout.off_dx + (tg * nt_max + lt) * tin_cap;
                saxpy(
                    1.0,
                    &part[dx_row..dx_row + tin],
                    &mut grads.dx[t * in_f + in0..t * in_f + in0 + tin],
                )?;
            }
            let dc_row = layout.off_dc + tg * g;
            for gi in 0..g {
                grads.grad_centers[gi] += part[dc_row + gi];
            }
        }
    }
    Ok(())
}

/// Read `dg` partials `[n, K]` (summed over out-tiles and in-tiles happens across launches).
pub fn acc_dg_partials(
    spec: &MobKanSpec,
    layout: &BwdPartialLayout,
    part: &[f32],
    dg: &mut [f32],
) -> Result<()> {
    if !spec.mask_routed() {
        let n = spec.n_us();
        let k = spec.k_us();
        let nt_max = layout.n_tile;
        for nt in 0..layout.n_tiles {
            let t0 = nt * nt_max;
            let tn = nt_max.min(n.saturating_sub(t0));
            for ot in 0..layout.out_tiles {
                let tg = layout.tg_index(nt, ot);
                for lt in 0..tn {
                    let dst = (t0 + lt) * k;
                    for lo in 0..layout.out_tile {
                        let src = layout.off_dg
                            + (((tg * nt_max + lt) * layout.out_tile + lo) * layout.k);
                        for e in 0..k {
                            dg[dst + e] += part[src + e];
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Softmax Jacobian + `λ_R` entropy on full-K router logits. Updates `dRouter` and `dx`.
pub fn router_bwd_cpu(
    spec: &MobKanSpec,
    x: &[f32],
    router: &[f32],
    gates: &[f32],
    mix_gates: &[f32],
    dg: &[f32],
    lambda_r: f32,
    aux_coef: f32,
    grad_router: &mut [f32],
    dx: &mut [f32],
) -> Result<(f32, f32)> {
    if spec.mask_routed() {
        return Ok((0.0, 0.0));
    }
    let n = spec.n_us();
    let in_f = spec.in_us();
    let k = spec.k_us();
    let (aux, dp) = if aux_coef > 0.0 && !spec.dense_router() {
        switch_aux(gates, mix_gates, n, k, aux_coef)
    } else {
        (0.0, [0.0f32; 4])
    };
    let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f32 };
    let mut h_sum = 0.0f32;
    for t in 0..n {
        let gg = &gates[t * k..t * k + k];
        let d = &dg[t * k..t * k + k];
        let dot: f32 = gg.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
        let mut dlogit = [0.0f32; 4];
        for e in 0..k {
            dlogit[e] = gg[e] * (d[e] - dot);
        }
        if lambda_r > 0.0 && k > 1 {
            let mut h = 0.0f32;
            for e in 0..k {
                let p = gg[e].max(1e-12);
                h -= p * p.ln();
            }
            h_sum += h;
            for e in 0..k {
                let p = gg[e].max(1e-12);
                dlogit[e] += lambda_r * (-p * (p.ln() + h));
            }
        }
        if aux_coef > 0.0 && !spec.dense_router() {
            let mut daux = [0.0f32; 4];
            for e in 0..k {
                daux[e] = dp[e] * inv_n;
            }
            let da_dot: f32 = gg.iter().zip(daux.iter()).map(|(a, b)| a * b).sum();
            for e in 0..k {
                dlogit[e] += gg[e] * (daux[e] - da_dot);
            }
        }
        for e in 0..k {
            for i in 0..in_f {
                grad_router[e * in_f + i] += dlogit[e] * x[t * in_f + i];
                dx[t * in_f + i] += dlogit[e] * router[e * in_f + i];
            }
        }
    }
    let ent = if lambda_r > 0.0 && k > 1 {
        h_sum / n as f32
    } else {
        0.0
    };
    Ok((ent, aux))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

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
    fn apply_topk_keeps_k_and_zeros_rest() {
        let mut g = vec![0.1f32, 0.7, 0.2, 0.5, 0.1, 0.4];
        apply_topk_gates(&mut g, 2, 3, 1);
        assert!((g[1] - 0.7).abs() < 1e-6);
        assert_eq!(g[0], 0.0);
        assert_eq!(g[2], 0.0);
        assert!((g[3] - 0.5).abs() < 1e-6);
        assert_eq!(g[4], 0.0);
        assert_eq!(g[5], 0.0);
        let mut dense = vec![0.1f32, 0.7, 0.2];
        let orig = dense.clone();
        apply_topk_gates(&mut dense, 1, 3, 0);
        assert_eq!(dense, orig);
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
        let scale_shared = vec![1.0f32; spec.scale_shared_len()];
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
        let spec_coarse =
            MobKanSpec::new(1, 4, 2, 4, 3, 1, 3, 3, 1, true, false, inv, 0.7).unwrap();
        let x = vec![1.6f32, 1.8, 2.0, 1.2];
        let w_base = vec![0.3f32; spec_full.w_base_len()];
        let w_shared = vec![0.1f32; spec_full.w_shared_len()];
        let mut w_routed = vec![0.0f32; spec_full.w_routed_len()];
        w_routed.fill(2.0);
        let router = vec![1.0f32; spec_full.router_len()];
        let centers = vec![-2.0f32, -0.66, 0.66, 2.0];
        let inv_widths = bump_inv_widths(&centers);
        let ones_o = vec![1.0f32; spec_full.scale_vec_len()];
        let ones_s = vec![1.0f32; spec_full.scale_shared_len()];
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
            &ones_s,
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
            &ones_s,
            &ones_r,
            &mut y_coarse,
        )
        .unwrap();
        let delta = (y_full[0] - y_coarse[0]).abs() + (y_full[1] - y_coarse[1]).abs();
        assert!(delta > 1e-4, "routed path must contribute, delta={delta}");
    }

    #[test]
    fn spec_layout_is_16_byte_friendly() {
        assert_eq!(size_of::<MobKanSpec>(), 80);
        assert_eq!(size_of::<MobKanSpec>() % 16, 0);
    }

    #[test]
    fn bwd_n_tile_keeps_occupancy_at_d256_train_shape() {
        let spec =
            MobKanSpec::new(384, 256, 256, 12, 8, 4, 3, 8, 1, false, false, 1.5, 0.7).unwrap();
        assert_eq!(spec.n_tile, MobKanSpec::N_TILE);
        assert!(
            spec.n_tiles() > 1,
            "n_tiles={} (need occupancy)",
            spec.n_tiles()
        );
    }

    #[test]
    fn spec_lifts_in_f_above_tile_cap() {
        let spec = MobKanSpec::new(2, 512, 64, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        assert_eq!(spec.in_f, 512);
        assert_eq!(spec.tile_in, MobKanSpec::TILE_IN);
        assert!(spec.in_f > MobKanSpec::MAX_IN);
        assert_eq!(spec.out_tile, MobKanSpec::OUT_TILE);
        assert!(spec.out_f > spec.out_tile);
        let extra = 256 + 256 * 4 + 3 + 256 + 3 * 256;
        let par = spec.bwd_tok_par() as usize;
        assert_eq!(spec.scratch_floats_fwd(), extra + spec.out_us());
        assert_eq!(
            spec.scratch_floats(),
            extra + spec.out_us() + extra * par.saturating_sub(1)
        );
        assert!(spec.scratch_floats() <= MobKanSpec::TG_SCRATCH_FLOATS as usize);
    }

    #[test]
    fn bwd_tok_par_packs_multiple_tokens_at_d256_g4() {
        let spec = MobKanSpec::new(384, 256, 256, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        assert!(
            spec.bwd_tok_par() >= 2,
            "tok_par={} (want ≥2 at G=4 so tokens are not fully serial)",
            spec.bwd_tok_par()
        );
        assert!(
            spec.scratch_floats_fwd() < spec.scratch_floats(),
            "fwd scratch must stay 1-token so occupancy does not collapse"
        );
    }

    #[test]
    fn bwd_partial_compact_stays_small_at_d256_g12() {
        let spec =
            MobKanSpec::new(384, 256, 256, 12, 8, 4, 3, 8, 1, false, false, 1.5, 0.7).unwrap();
        let layout = BwdPartialLayout::from_spec(&spec);
        let mb = (layout.floats * 4) as f64 / (1024.0 * 1024.0);
        assert!(
            mb < 24.0,
            "compact fused-bwd slab {mb:.1} MB (want < 24 at d=256 G=12 n=384)"
        );
    }

    fn run_fused(spec: &MobKanSpec, x: &[f32], w: &FusedW) -> Vec<f32> {
        let mut y = vec![0.0f32; spec.y_len()];
        mob_kan_fused_cpu(
            spec, x, &w.base, &w.shared, &w.routed, &w.router, &w.centers, &w.inv, &w.sb, &w.ss,
            &w.sr, &mut y,
        )
        .unwrap();
        y
    }

    struct FusedW {
        base: Vec<f32>,
        shared: Vec<f32>,
        routed: Vec<f32>,
        router: Vec<f32>,
        centers: Vec<f32>,
        inv: Vec<f32>,
        sb: Vec<f32>,
        ss: Vec<f32>,
        sr: Vec<f32>,
    }

    fn fused_weights(spec: &MobKanSpec) -> (Vec<f32>, FusedW) {
        let x: Vec<f32> = (0..spec.x_len())
            .map(|i| (i as f32) * 0.017 - 0.4)
            .collect();
        let centers: Vec<f32> = (0..spec.g_us())
            .map(|i| -2.0 + 4.0 * i as f32 / (spec.g_us() as f32 - 1.0).max(1.0))
            .collect();
        let w = FusedW {
            base: (0..spec.w_base_len())
                .map(|i| (i as f32).sin() * 0.2)
                .collect(),
            shared: (0..spec.w_shared_len())
                .map(|i| (i as f32).cos() * 0.05)
                .collect(),
            routed: (0..spec.w_routed_len())
                .map(|i| ((i % 7) as f32) * 0.03 - 0.1)
                .collect(),
            router: (0..spec.router_len())
                .map(|i| (i as f32) * 0.01 - 0.02)
                .collect(),
            inv: bump_inv_widths(&centers),
            centers,
            sb: vec![1.0f32; spec.scale_vec_len()],
            ss: vec![1.0f32; spec.scale_shared_len()],
            sr: vec![1.0f32; spec.scale_routed_len()],
        };
        (x, w)
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(u, v)| (u - v).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn tiled_cpu_matches_default_d32() {
        let spec = MobKanSpec::new(5, 32, 24, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        let (x, w) = fused_weights(&spec);
        let y0 = run_fused(&spec, &x, &w);
        let spec_t = spec.force_tiles(7, 5, 3).unwrap();
        let y1 = run_fused(&spec_t, &x, &w);
        let err = max_abs(&y0, &y1);
        assert!(err < 1e-4, "d32 tile vs default max|Δ|={err}");
        assert!(y0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tiled_cpu_matches_forced_d512() {
        let spec = MobKanSpec::new(3, 512, 64, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        assert!(spec.tile_in < spec.in_f);
        assert!(spec.out_tile < spec.out_f);
        let (x, w) = fused_weights(&spec);
        let y0 = run_fused(&spec, &x, &w);
        let spec_t = spec.force_tiles(64, 16, 2).unwrap();
        let y1 = run_fused(&spec_t, &x, &w);
        let err = max_abs(&y0, &y1);
        assert!(err < 1e-4, "d512 tile vs default max|Δ|={err}");
        assert!(y0.iter().any(|v| *v != 0.0));
    }
}
