//! `SovereignTensor` — page-aligned host slab aliased as a Metal Shared buffer.
//!
//! No Candle `Tensor`. The `PageSlab` is the CPU/Accelerate source of truth.
//! On Metal, `wrap_shared_bytes_no_copy` aliases the **whole** 16 KiB-aligned
//! slab; there is no host↔device memcpy and no `host_gen`/`device_gen`.
//!
//! Drop order is declaration order (first field first). `gpu` is declared
//! before `slab` so the `MTLBuffer` dies before `PageSlab::dealloc`. Call
//! `detach_gpu` before replacing a tensor whose numel changes (`regrid`,
//! `refresh_geometry`, `expand_vocab`). Never wrap a mid-slab interior.

use anyhow::{bail, Result};

use crate::accelerate::{
    acc_dg_partials, apply_topk_gates, mob_kan_fused_bwd_cpu, mob_kan_fused_cpu,
    reduce_bwd_partials, router_bwd_cpu, sgemm_nt, softmax_rows, BwdPartialLayout, FusedBwdGrads,
    MobKanSpec,
};
use crate::device::{self, Backend, PageSlab, SovereignDevice};

/// Lightweight f32 tensor with a page-aligned host slab and optional Metal wrap.
pub struct SovereignTensor {
    // Drop = declaration order. gpu MUST be first so the wrap dies before dealloc.
    #[cfg(target_os = "macos")]
    gpu: Option<GpuSlot>,
    slab: PageSlab,
    shape: Vec<usize>,
    numel: usize,
}

#[cfg(target_os = "macos")]
struct GpuSlot {
    buffer: metal::Buffer,
}

impl std::fmt::Debug for SovereignTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovereignTensor")
            .field("shape", &self.shape)
            .field("numel", &self.numel)
            .field("gpu", &self.has_gpu())
            .finish()
    }
}

impl Clone for SovereignTensor {
    fn clone(&self) -> Self {
        let mut t = Self::zeros(self.shape.clone()).expect("clone of a valid SovereignTensor");
        if self.numel > 0 {
            t.as_mut_slice().copy_from_slice(self.as_slice());
        }
        t
    }
}

impl SovereignTensor {
    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let n = numel_shape(&shape)?;
        if data.len() != n {
            bail!(
                "SovereignTensor data len {} != shape product {n}",
                data.len()
            );
        }
        let mut t = Self::zeros(shape)?;
        if n > 0 {
            t.as_mut_slice().copy_from_slice(&data);
        }
        Ok(t)
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        let n = numel_shape(&shape)?;
        let slab = PageSlab::new(n.max(1).saturating_mul(4))?;
        Ok(Self {
            #[cfg(target_os = "macos")]
            gpu: None,
            slab,
            shape,
            numel: n,
        })
    }

    pub fn fill(shape: Vec<usize>, value: f32) -> Result<Self> {
        let mut t = Self::zeros(shape)?;
        t.as_mut_slice().fill(value);
        Ok(t)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    pub fn has_gpu(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        self.slab.f32_at(self.numel).expect("tensor slab")
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.slab.f32_at_mut(self.numel).expect("tensor slab")
    }

    /// Bind a Shared Metal wrap of the whole slab. Idempotent while attached.
    ///
    /// Exclusive CPU/GPU epochs: do not keep `&mut [f32]` live across
    /// `dispatch_fused_mob_kan`. After a GPU write, `wait_until_completed`
    /// (inside dispatch) is the host-visibility fence.
    pub fn attach(&mut self, gpu: &SovereignDevice) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            let Some(mtl) = gpu.mtl_device() else {
                return Ok(());
            };
            if self.gpu.is_some() {
                return Ok(());
            }
            let bytes = self.slab.as_bytes();
            let buffer = device::wrap_shared_bytes_no_copy(mtl, bytes)?;
            self.gpu = Some(GpuSlot { buffer });
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = gpu;
            Ok(())
        }
    }

    /// Drop the Metal wrap, keeping slab data. Call before realloc / replace.
    pub fn detach_gpu(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.gpu = None;
        }
    }

    #[cfg(target_os = "macos")]
    pub fn metal_buffer(&self) -> Option<&metal::Buffer> {
        self.gpu.as_ref().map(|s| &s.buffer)
    }

    /// Grow-or-reuse a scratch tensor of exact `numel`. Used for pooled Metal x/y.
    pub fn reuse_for<'a>(
        slot: &'a mut Option<Self>,
        shape: Vec<usize>,
        gpu: &SovereignDevice,
    ) -> Result<&'a mut Self> {
        let n = numel_shape(&shape)?;
        let reuse = slot.as_ref().is_some_and(|t| t.numel() == n);
        if reuse {
            let t = slot
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("reuse slot empty"))?;
            t.reshape(shape)?;
            t.attach(gpu)?;
            Ok(t)
        } else {
            if let Some(old) = slot.as_mut() {
                old.detach_gpu();
            }
            let mut t = Self::zeros(shape)?;
            t.attach(gpu)?;
            *slot = Some(t);
            slot.as_mut()
                .ok_or_else(|| anyhow::anyhow!("reuse slot set"))
        }
    }

    pub fn reshape(&mut self, shape: Vec<usize>) -> Result<()> {
        let n = numel_shape(&shape)?;
        if n != self.numel {
            bail!("reshape {} -> {shape:?} changes numel {}", self.numel, n);
        }
        self.shape = shape;
        Ok(())
    }
}

fn numel_shape(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut n = 1usize;
    for &d in shape {
        n = n
            .checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("shape overflow {shape:?}"))?;
    }
    Ok(n)
}

/// Bindings for one fused MoB-KAN launch. Callers own the tensors; this
/// struct only borrows them through the pipeline.
pub struct FusedKanTensors<'a> {
    pub x: &'a SovereignTensor,
    pub y: &'a mut SovereignTensor,
    pub w_base: &'a SovereignTensor,
    pub w_shared: &'a SovereignTensor,
    pub w_routed: Option<&'a SovereignTensor>,
    pub router: Option<&'a SovereignTensor>,
    pub centers: &'a SovereignTensor,
    pub inv_widths: &'a SovereignTensor,
    pub scale_base: &'a SovereignTensor,
    pub scale_shared: &'a SovereignTensor,
    pub scale_routed: &'a SovereignTensor,
    pub weight_half: bool,
    pub half: Option<HalfWeightBufs<'a>>,
}

/// Bindings for one fused MoB-KAN backward launch.
pub struct FusedKanBwdTensors<'a> {
    pub x: &'a SovereignTensor,
    pub dy: &'a SovereignTensor,
    pub w_base: &'a SovereignTensor,
    pub w_shared: &'a SovereignTensor,
    pub w_routed: Option<&'a SovereignTensor>,
    pub router: Option<&'a SovereignTensor>,
    pub centers: &'a SovereignTensor,
    pub inv_widths: &'a SovereignTensor,
    pub scale_base: &'a SovereignTensor,
    pub scale_shared: &'a SovereignTensor,
    pub scale_routed: &'a SovereignTensor,
    pub grads: FusedBwdGrads<'a>,
    pub lambda_r: f32,
    pub aux_coef: f32,
    pub weight_half: bool,
    pub half: Option<HalfWeightBufs<'a>>,
}

/// FP16 master weights aliased as Shared `half` buffers. Compute promotes to float.
pub struct HalfWeightBufs<'a> {
    pub base: &'a HalfWire,
    pub shared: &'a HalfWire,
    pub routed: Option<&'a HalfWire>,
    pub router: Option<&'a HalfWire>,
}

/// Page-aligned `u16` slab wrapped as a Metal Shared buffer of `half`.
pub struct HalfWire {
    #[cfg(target_os = "macos")]
    gpu: Option<GpuSlot>,
    slab: PageSlab,
    n: usize,
}

impl std::fmt::Debug for HalfWire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HalfWire").field("n", &self.n).finish()
    }
}

impl HalfWire {
    fn new(n: usize, gpu: &SovereignDevice) -> Result<Self> {
        let bytes = n.saturating_mul(2).max(16);
        let mut t = Self {
            #[cfg(target_os = "macos")]
            gpu: None,
            slab: PageSlab::new(bytes)?,
            n,
        };
        t.attach(gpu)?;
        Ok(t)
    }

    pub fn reuse_u16<'a>(
        slot: &'a mut Option<Self>,
        data: &[u16],
        gpu: &SovereignDevice,
    ) -> Result<&'a mut Self> {
        let n = data.len();
        let ok = slot.as_ref().is_some_and(|t| t.n == n);
        if !ok {
            if let Some(old) = slot.as_mut() {
                old.detach_gpu();
            }
            *slot = Some(Self::new(n, gpu)?);
        }
        let t = slot
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("half wire slot"))?;
        t.as_u16_mut()?.copy_from_slice(data);
        t.attach(gpu)?;
        Ok(t)
    }

    fn as_u16_mut(&mut self) -> Result<&mut [u16]> {
        self.slab.u16_at_mut(self.n)
    }

    fn attach(&mut self, gpu: &SovereignDevice) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            let Some(mtl) = gpu.mtl_device() else {
                return Ok(());
            };
            if self.gpu.is_some() {
                return Ok(());
            }
            let buffer = device::wrap_shared_bytes_no_copy(mtl, self.slab.as_bytes())?;
            self.gpu = Some(GpuSlot { buffer });
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = gpu;
            Ok(())
        }
    }

    fn detach_gpu(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.gpu = None;
        }
    }

    #[cfg(target_os = "macos")]
    pub fn metal_buffer(&self) -> Option<&metal::Buffer> {
        self.gpu.as_ref().map(|s| &s.buffer)
    }
}

/// Run fused backward. Metal writes TG-private partials and the host reduces;
/// CPU is the tiled Accelerate path. Returns `(router entropy, aux)`.
pub fn fused_mob_kan_bwd(
    gpu: &SovereignDevice,
    spec: &MobKanSpec,
    tensors: FusedKanBwdTensors<'_>,
    part: &mut Option<SovereignTensor>,
) -> Result<(f32, f32)> {
    spec.validate()?;
    check_len(tensors.x, spec.x_len(), "x")?;
    check_len(tensors.dy, spec.y_len(), "dy")?;
    if tensors.weight_half {
        let h = tensors
            .half
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("half weights missing"))?;
        if h.base.n != spec.w_base_len() {
            bail!("half w_base {} != {}", h.base.n, spec.w_base_len());
        }
        if h.shared.n != spec.w_shared_len() {
            bail!("half w_shared {} != {}", h.shared.n, spec.w_shared_len());
        }
    } else {
        check_len(tensors.w_base, spec.w_base_len(), "w_base")?;
        check_len(tensors.w_shared, spec.w_shared_len(), "w_shared")?;
    }
    check_len(tensors.centers, spec.centers_len(), "centers")?;
    check_len(tensors.inv_widths, spec.centers_len(), "inv_widths")?;
    match gpu.backend() {
        Backend::Metal => fused_bwd_metal(gpu, spec, tensors, part),
        Backend::Cpu => fused_bwd_cpu(spec, tensors),
    }
}

fn fused_bwd_cpu(spec: &MobKanSpec, tensors: FusedKanBwdTensors<'_>) -> Result<(f32, f32)> {
    if tensors.weight_half {
        bail!("fp16 master compute is Metal-only");
    }
    let FusedKanBwdTensors {
        x,
        dy,
        w_base,
        w_shared,
        w_routed,
        router,
        centers,
        inv_widths,
        scale_base,
        scale_shared,
        scale_routed,
        grads,
        lambda_r,
        aux_coef,
        weight_half: _,
        half: _,
    } = tensors;
    let empty: &[f32] = &[];
    let wr = w_routed.map_or(empty, SovereignTensor::as_slice);
    let rt = router.map_or(empty, SovereignTensor::as_slice);
    mob_kan_fused_bwd_cpu(
        spec,
        x.as_slice(),
        w_base.as_slice(),
        w_shared.as_slice(),
        wr,
        rt,
        centers.as_slice(),
        inv_widths.as_slice(),
        scale_base.as_slice(),
        scale_shared.as_slice(),
        scale_routed.as_slice(),
        dy.as_slice(),
        lambda_r,
        aux_coef,
        grads,
    )
}

#[cfg(target_os = "macos")]
fn fused_bwd_metal(
    gpu: &SovereignDevice,
    spec: &MobKanSpec,
    tensors: FusedKanBwdTensors<'_>,
    part_slot: &mut Option<SovereignTensor>,
) -> Result<(f32, f32)> {
    let FusedKanBwdTensors {
        x,
        dy,
        w_base,
        w_shared,
        w_routed,
        router,
        centers,
        inv_widths,
        scale_base,
        scale_shared,
        scale_routed,
        mut grads,
        lambda_r,
        aux_coef,
        weight_half,
        half,
    } = tensors;
    if spec.packed != 0 {
        grads.dx[..spec.x_len()].fill(0.0);
        return Ok((0.0, 0.0));
    }
    let layout = BwdPartialLayout::from_spec(spec);
    let part = SovereignTensor::reuse_for(part_slot, vec![layout.floats.max(1)], gpu)?;
    grads.dx[..spec.x_len()].fill(0.0);

    let empty: &[f32] = &[];
    let router_s = router.map_or(empty, SovereignTensor::as_slice);
    let n = spec.n_us();
    let k = spec.k_us();
    let in_f = spec.in_us();
    let routed = !spec.mask_routed();
    let mut gates = vec![0.0f32; n * k.max(1)];
    let mut dg = vec![0.0f32; n * k.max(1)];
    if routed {
        sgemm_nt(n, k, in_f, 1.0, x.as_slice(), router_s, 0.0, &mut gates)?;
        softmax_rows(&mut gates, n, k)?;
    }
    let mut mix_gates = gates.clone();
    apply_topk_gates(&mut mix_gates, n, k, spec.topk);

    let dummy = gpu.dummy_buffer();
    let tin_max = spec.tile_in_us();
    let mut in0 = 0usize;
    while in0 < spec.in_us() {
        let tin = tin_max.min(spec.in_us() - in0);
        part.as_mut_slice().fill(0.0);
        {
            let xb = x
                .metal_buffer()
                .ok_or_else(|| anyhow::anyhow!("x has no Metal buffer"))?;
            let dyb = dy
                .metal_buffer()
                .ok_or_else(|| anyhow::anyhow!("dy has no Metal buffer"))?;
            let w_base_b = if weight_half {
                half.as_ref()
                    .and_then(|h| h.base.metal_buffer())
                    .ok_or_else(|| anyhow::anyhow!("half w_base has no Metal buffer"))?
            } else {
                w_base
                    .metal_buffer()
                    .ok_or_else(|| anyhow::anyhow!("w_base has no Metal buffer"))?
            };
            let w_shared_b = if weight_half {
                half.as_ref()
                    .and_then(|h| h.shared.metal_buffer())
                    .ok_or_else(|| anyhow::anyhow!("half w_shared has no Metal buffer"))?
            } else {
                w_shared
                    .metal_buffer()
                    .ok_or_else(|| anyhow::anyhow!("w_shared has no Metal buffer"))?
            };
            let centers_b = centers
                .metal_buffer()
                .ok_or_else(|| anyhow::anyhow!("centers has no Metal buffer"))?;
            let inv_b = inv_widths
                .metal_buffer()
                .ok_or_else(|| anyhow::anyhow!("inv_widths has no Metal buffer"))?;
            let sb = scale_base
                .metal_buffer()
                .ok_or_else(|| anyhow::anyhow!("scale_base has no Metal buffer"))?;
            let ss = scale_shared
                .metal_buffer()
                .ok_or_else(|| anyhow::anyhow!("scale_shared has no Metal buffer"))?;
            let sr = scale_routed
                .metal_buffer()
                .ok_or_else(|| anyhow::anyhow!("scale_routed has no Metal buffer"))?;
            let w_routed_b = if weight_half {
                half.as_ref()
                    .and_then(|h| h.routed.and_then(HalfWire::metal_buffer))
                    .unwrap_or(dummy)
            } else {
                w_routed
                    .and_then(SovereignTensor::metal_buffer)
                    .unwrap_or(dummy)
            };
            let router_b = if weight_half {
                half.as_ref()
                    .and_then(|h| h.router.and_then(HalfWire::metal_buffer))
                    .unwrap_or(dummy)
            } else {
                router
                    .and_then(SovereignTensor::metal_buffer)
                    .unwrap_or(dummy)
            };
            let part_b = part
                .metal_buffer()
                .ok_or_else(|| anyhow::anyhow!("part has no Metal buffer"))?;
            gpu.dispatch_fused_mob_kan_bwd(
                spec,
                in0 as u32,
                tin as u32,
                xb,
                dyb,
                w_base_b,
                w_shared_b,
                w_routed_b,
                router_b,
                centers_b,
                inv_b,
                sb,
                ss,
                sr,
                part_b,
                weight_half,
            )?;
        }
        reduce_bwd_partials(spec, &layout, in0, tin, part.as_slice(), &mut grads)?;
        acc_dg_partials(spec, &layout, part.as_slice(), &mut dg)?;
        in0 += tin;
    }
    router_bwd_cpu(
        spec,
        x.as_slice(),
        router_s,
        &gates,
        &mix_gates,
        &dg,
        lambda_r,
        aux_coef,
        grads.grad_router,
        grads.dx,
    )
}

#[cfg(not(target_os = "macos"))]
fn fused_bwd_metal(
    _gpu: &SovereignDevice,
    spec: &MobKanSpec,
    tensors: FusedKanBwdTensors<'_>,
    _part: &mut Option<SovereignTensor>,
) -> Result<(f32, f32)> {
    fused_bwd_cpu(spec, tensors)
}

/// Run the fused step on `gpu`. Metal path is a single compute encoder;
/// CPU path is Accelerate GEMM + bump eval.
pub fn fused_mob_kan_step(
    gpu: &SovereignDevice,
    spec: &MobKanSpec,
    tensors: FusedKanTensors<'_>,
) -> Result<()> {
    spec.validate()?;
    check_len(tensors.x, spec.x_len(), "x")?;
    check_len(tensors.y, spec.y_len(), "y")?;
    check_len(tensors.centers, spec.centers_len(), "centers")?;
    check_len(tensors.inv_widths, spec.centers_len(), "inv_widths")?;
    check_len(tensors.scale_base, spec.scale_vec_len(), "scale_base")?;
    check_len(
        tensors.scale_shared,
        spec.scale_shared_len(),
        "scale_shared",
    )?;
    if tensors.weight_half {
        let h = tensors
            .half
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("half weights missing"))?;
        if h.base.n != spec.w_base_len() {
            bail!("half w_base {} != {}", h.base.n, spec.w_base_len());
        }
        if h.shared.n != spec.w_shared_len() {
            bail!("half w_shared {} != {}", h.shared.n, spec.w_shared_len());
        }
    } else {
        check_len(tensors.w_base, spec.w_base_len(), "w_base")?;
        check_len(tensors.w_shared, spec.w_shared_len(), "w_shared")?;
    }
    if !spec.mask_routed() {
        check_len(
            tensors.scale_routed,
            spec.scale_routed_len(),
            "scale_routed",
        )?;
        if tensors.weight_half {
            let h = tensors
                .half
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("half weights missing"))?;
            let wr = h
                .routed
                .ok_or_else(|| anyhow::anyhow!("half w_routed required"))?;
            let rt = h
                .router
                .ok_or_else(|| anyhow::anyhow!("half router required"))?;
            if wr.n != spec.w_routed_len() {
                bail!("half w_routed {} != {}", wr.n, spec.w_routed_len());
            }
            if rt.n != spec.router_len() {
                bail!("half router {} != {}", rt.n, spec.router_len());
            }
        } else {
            let wr = tensors
                .w_routed
                .ok_or_else(|| anyhow::anyhow!("w_routed required"))?;
            let rt = tensors
                .router
                .ok_or_else(|| anyhow::anyhow!("router required"))?;
            check_len(wr, spec.w_routed_len(), "w_routed")?;
            check_len(rt, spec.router_len(), "router")?;
        }
    }

    match gpu.backend() {
        Backend::Metal => fused_metal(gpu, spec, tensors),
        Backend::Cpu => fused_cpu(spec, tensors),
    }
}

fn check_len(t: &SovereignTensor, n: usize, name: &str) -> Result<()> {
    if t.numel() != n {
        bail!("{name} numel {} != {n}", t.numel());
    }
    Ok(())
}

fn fused_cpu(spec: &MobKanSpec, tensors: FusedKanTensors<'_>) -> Result<()> {
    if tensors.weight_half {
        bail!("fp16 master compute is Metal-only");
    }
    let empty: &[f32] = &[];
    let wr = tensors.w_routed.map_or(empty, SovereignTensor::as_slice);
    let rt = tensors.router.map_or(empty, SovereignTensor::as_slice);
    mob_kan_fused_cpu(
        spec,
        tensors.x.as_slice(),
        tensors.w_base.as_slice(),
        tensors.w_shared.as_slice(),
        wr,
        rt,
        tensors.centers.as_slice(),
        tensors.inv_widths.as_slice(),
        tensors.scale_base.as_slice(),
        tensors.scale_shared.as_slice(),
        tensors.scale_routed.as_slice(),
        tensors.y.as_mut_slice(),
    )
}

#[cfg(target_os = "macos")]
fn fused_metal(
    gpu: &SovereignDevice,
    spec: &MobKanSpec,
    tensors: FusedKanTensors<'_>,
) -> Result<()> {
    // Alias: no memcpy. Dispatch `wait_until_completed` is the host fence.
    {
        let x = tensors
            .x
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("x has no Metal buffer; call attach()"))?;
        let y = tensors
            .y
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("y has no Metal buffer; call attach()"))?;
        let dummy = gpu.dummy_buffer();
        let (w_base, w_shared, w_routed, router) = if tensors.weight_half {
            let h = tensors
                .half
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("half weights missing"))?;
            (
                h.base
                    .metal_buffer()
                    .ok_or_else(|| anyhow::anyhow!("half w_base has no Metal buffer"))?,
                h.shared
                    .metal_buffer()
                    .ok_or_else(|| anyhow::anyhow!("half w_shared has no Metal buffer"))?,
                h.routed.and_then(HalfWire::metal_buffer).unwrap_or(dummy),
                h.router.and_then(HalfWire::metal_buffer).unwrap_or(dummy),
            )
        } else {
            (
                tensors
                    .w_base
                    .metal_buffer()
                    .ok_or_else(|| anyhow::anyhow!("w_base has no Metal buffer"))?,
                tensors
                    .w_shared
                    .metal_buffer()
                    .ok_or_else(|| anyhow::anyhow!("w_shared has no Metal buffer"))?,
                tensors
                    .w_routed
                    .and_then(SovereignTensor::metal_buffer)
                    .unwrap_or(dummy),
                tensors
                    .router
                    .and_then(SovereignTensor::metal_buffer)
                    .unwrap_or(dummy),
            )
        };
        let centers = tensors
            .centers
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("centers has no Metal buffer"))?;
        let inv_widths = tensors
            .inv_widths
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("inv_widths has no Metal buffer"))?;
        let scale_base = tensors
            .scale_base
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("scale_base has no Metal buffer"))?;
        let scale_shared = tensors
            .scale_shared
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("scale_shared has no Metal buffer"))?;
        let scale_routed = tensors
            .scale_routed
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("scale_routed has no Metal buffer"))?;

        gpu.dispatch_fused_mob_kan(
            spec,
            x,
            y,
            w_base,
            w_shared,
            w_routed,
            router,
            centers,
            inv_widths,
            scale_base,
            scale_shared,
            scale_routed,
            tensors.weight_half,
        )?;
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn fused_metal(
    _gpu: &SovereignDevice,
    spec: &MobKanSpec,
    tensors: FusedKanTensors<'_>,
) -> Result<()> {
    fused_cpu(spec, tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_roundtrip_and_reshape() {
        let mut t =
            SovereignTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.numel(), 6);
        t.as_mut_slice()[0] = 9.0;
        t.reshape(vec![3, 2]).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.as_slice()[0], 9.0);
        let p = t.as_slice().as_ptr() as usize;
        assert_eq!(p % 16_384, 0, "slab must be 16 KiB-aligned for DMA wrap");
    }

    #[test]
    fn clone_is_independent_host_copy() {
        let mut t = SovereignTensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
        let mut u = t.clone();
        u.as_mut_slice()[0] = 7.0;
        t.as_mut_slice()[0] = 5.0;
        assert_eq!(t.as_slice()[0], 5.0);
        assert_eq!(u.as_slice()[0], 7.0);
        assert!(!u.has_gpu());
    }

    #[test]
    fn cpu_fused_via_tensors() {
        let gpu = SovereignDevice::open(false).unwrap();
        let spec = MobKanSpec::new(2, 4, 3, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        let x = SovereignTensor::fill(vec![2, 4], 0.2).unwrap();
        let mut y = SovereignTensor::zeros(vec![2, 3]).unwrap();
        let w_base = SovereignTensor::fill(vec![3, 4], 0.05).unwrap();
        let w_shared = SovereignTensor::fill(vec![4, 3], 0.02).unwrap();
        let w_routed = SovereignTensor::fill(vec![3, 4, 1], 0.01).unwrap();
        let router = SovereignTensor::zeros(vec![3, 4]).unwrap();
        let centers = SovereignTensor::from_vec(vec![4], vec![-2.0, -0.66, 0.66, 2.0]).unwrap();
        let iw = crate::accelerate::bump_inv_widths(centers.as_slice());
        let inv_widths = SovereignTensor::from_vec(vec![4], iw).unwrap();
        let scale_base = SovereignTensor::fill(vec![3], 1.0).unwrap();
        let scale_shared = SovereignTensor::fill(vec![4], 1.0).unwrap();
        let scale_routed = SovereignTensor::fill(vec![3, 4], 1.0).unwrap();
        fused_mob_kan_step(
            &gpu,
            &spec,
            FusedKanTensors {
                x: &x,
                y: &mut y,
                w_base: &w_base,
                w_shared: &w_shared,
                w_routed: Some(&w_routed),
                router: Some(&router),
                centers: &centers,
                inv_widths: &inv_widths,
                scale_base: &scale_base,
                scale_shared: &scale_shared,
                scale_routed: &scale_routed,
                weight_half: false,
                half: None,
            },
        )
        .unwrap();
        assert!(y.as_slice().iter().all(|v| v.is_finite()));
    }

    /// Field order: `gpu` is declared before `slab`. Drop of an attached
    /// tensor must `gpu.take()` (MTLBuffer) before `PageSlab::dealloc`.
    /// miri does not cover Metal; this is the debug-glue stand-in.
    #[cfg(target_os = "macos")]
    #[test]
    fn detach_before_drop_and_alias_host() {
        let gpu = SovereignDevice::open(true).unwrap();
        if !gpu.is_metal() {
            return;
        }
        let mut t = SovereignTensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        t.attach(&gpu).unwrap();
        assert!(t.has_gpu());
        t.as_mut_slice()[0] = 9.0;
        assert_eq!(t.as_slice()[0], 9.0);
        t.detach_gpu();
        assert!(!t.has_gpu());
        assert_eq!(t.as_slice()[0], 9.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_alias_matches_cpu() {
        let gpu = SovereignDevice::open(true).unwrap();
        if !gpu.is_metal() {
            return;
        }
        let cpu = SovereignDevice::open(false).unwrap();
        let spec = MobKanSpec::new(2, 4, 3, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        let mut x = SovereignTensor::fill(vec![2, 4], 0.2).unwrap();
        let mut y_gpu = SovereignTensor::zeros(vec![2, 3]).unwrap();
        let mut w_base = SovereignTensor::fill(vec![3, 4], 0.05).unwrap();
        let mut w_shared = SovereignTensor::fill(vec![4, 3], 0.02).unwrap();
        let mut w_routed = SovereignTensor::fill(vec![3, 4, 1], 0.01).unwrap();
        let mut router = SovereignTensor::zeros(vec![3, 4]).unwrap();
        let mut centers = SovereignTensor::from_vec(vec![4], vec![-2.0, -0.66, 0.66, 2.0]).unwrap();
        let iw = crate::accelerate::bump_inv_widths(centers.as_slice());
        let mut inv_widths = SovereignTensor::from_vec(vec![4], iw).unwrap();
        let mut scale_base = SovereignTensor::fill(vec![3], 1.0).unwrap();
        let mut scale_shared = SovereignTensor::fill(vec![4], 1.0).unwrap();
        let mut scale_routed = SovereignTensor::fill(vec![3, 4], 1.0).unwrap();
        for t in [
            &mut x,
            &mut y_gpu,
            &mut w_base,
            &mut w_shared,
            &mut w_routed,
            &mut router,
            &mut centers,
            &mut inv_widths,
            &mut scale_base,
            &mut scale_shared,
            &mut scale_routed,
        ] {
            t.attach(&gpu).unwrap();
        }
        fused_mob_kan_step(
            &gpu,
            &spec,
            FusedKanTensors {
                x: &x,
                y: &mut y_gpu,
                w_base: &w_base,
                w_shared: &w_shared,
                w_routed: Some(&w_routed),
                router: Some(&router),
                centers: &centers,
                inv_widths: &inv_widths,
                scale_base: &scale_base,
                scale_shared: &scale_shared,
                scale_routed: &scale_routed,
                weight_half: false,
                half: None,
            },
        )
        .unwrap();
        let mut y_cpu = SovereignTensor::zeros(vec![2, 3]).unwrap();
        fused_mob_kan_step(
            &cpu,
            &spec,
            FusedKanTensors {
                x: &x,
                y: &mut y_cpu,
                w_base: &w_base,
                w_shared: &w_shared,
                w_routed: Some(&w_routed),
                router: Some(&router),
                centers: &centers,
                inv_widths: &inv_widths,
                scale_base: &scale_base,
                scale_shared: &scale_shared,
                scale_routed: &scale_routed,
                weight_half: false,
                half: None,
            },
        )
        .unwrap();
        let max = y_gpu
            .as_slice()
            .iter()
            .zip(y_cpu.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "metal vs cpu max|Δ|={max}");
    }
}
