//! Bare-metal Apple Silicon device: `MTLDevice`, `MTLCommandQueue`, fused MSL.
//!
//! This is the only module that maps Metal buffer pointers. Accelerate C
//! symbols live in `accelerate`. Candle helpers at the bottom are a
//! transitional bridge and will vanish with the KAN/train port.

#![allow(unsafe_code)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::mem::size_of;
use std::ptr::{self, NonNull};

use anyhow::{bail, Context, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::accelerate::MobKanSpec;
#[cfg(target_os = "macos")]
use crate::telemetry::record_gpu_wait;

pub fn rng_from_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Host STE tape (`TernaryKanLinear::backward_into`) instead of fused bwd.
pub fn prefer_host_bwd() -> bool {
    std::env::var_os("ULLIS_HOST_BWD").is_some_and(|v| v == "1")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
}

/// Compile-time Metal sub-allocation flags (preprocessor macros).
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceFlags {
    /// `ULLIS_FUSED_GRAD_CKPT`: rematerialize layer interiors on the backward
    /// dispatch instead of keeping a device-side activation tape.
    pub fused_grad_ckpt: bool,
}

/// Runtime GPU / host device owning the fused MoB-KAN pipeline.
pub struct SovereignDevice {
    backend: Backend,
    name: String,
    flags: DeviceFlags,
    #[cfg(target_os = "macos")]
    metal: Option<MetalInner>,
}

#[cfg(target_os = "macos")]
struct MetalInner {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    pipeline_bwd: metal::ComputePipelineState,
    pipeline_half: metal::ComputePipelineState,
    pipeline_bwd_half: metal::ComputePipelineState,
    embed_i8: metal::ComputePipelineState,
    logits_i8: metal::ComputePipelineState,
    dummy: metal::Buffer,
}

/// Page-aligned host slab used as a no-copy Metal Shared backing store.
#[derive(Debug)]
pub struct PageSlab {
    ptr: NonNull<u8>,
    layout: Layout,
    bytes: usize,
}

// Exclusive owner of the allocation.
unsafe impl Send for PageSlab {}

impl PageSlab {
    /// `bytes` is rounded up to a 16 KiB Apple-Silicon page.
    pub fn new(bytes: usize) -> Result<Self> {
        let align = 16_384usize;
        let size = bytes.max(align);
        let size = size.div_ceil(align) * align;
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| anyhow::anyhow!("page layout: {e}"))?;
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or_else(|| anyhow::anyhow!("page alloc failed"))?;
        Ok(Self {
            ptr,
            layout,
            bytes: size,
        })
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.bytes) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.bytes) }
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn u32_at(&self, byte_off: usize, n: usize) -> Result<&[u32]> {
        let need = byte_off
            .checked_add(n.saturating_mul(4))
            .ok_or_else(|| anyhow::anyhow!("u32 slice overflow"))?;
        if need > self.bytes {
            bail!("u32 slice {byte_off}+{n} overruns {} byte slab", self.bytes);
        }
        if byte_off % 4 != 0 {
            bail!("u32 slice offset {byte_off} is not 4-byte aligned");
        }
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(byte_off).cast::<u32>(), n) })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn u32_at_mut(&mut self, byte_off: usize, n: usize) -> Result<&mut [u32]> {
        let need = byte_off
            .checked_add(n.saturating_mul(4))
            .ok_or_else(|| anyhow::anyhow!("u32 slice overflow"))?;
        if need > self.bytes {
            bail!("u32 slice {byte_off}+{n} overruns {} byte slab", self.bytes);
        }
        if byte_off % 4 != 0 {
            bail!("u32 slice offset {byte_off} is not 4-byte aligned");
        }
        Ok(unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(byte_off).cast::<u32>(), n)
        })
    }

    /// First `n` f32s. The slab pointer is 16 KiB-aligned, so this is always
    /// a valid f32 view of the prefix (never a mid-slab Metal wrap).
    #[allow(clippy::cast_ptr_alignment)]
    pub fn u16_at(&self, n: usize) -> Result<&[u16]> {
        let need = n
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("u16 slice overflow"))?;
        if need > self.bytes {
            bail!("u16 slice {n} overruns {} byte slab", self.bytes);
        }
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast::<u16>(), n) })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn u16_at_mut(&mut self, n: usize) -> Result<&mut [u16]> {
        let need = n
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("u16 slice overflow"))?;
        if need > self.bytes {
            bail!("u16 slice {n} overruns {} byte slab", self.bytes);
        }
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<u16>(), n) })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn f32_at(&self, n: usize) -> Result<&[f32]> {
        let need = n
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("f32 slice overflow"))?;
        if need > self.bytes {
            bail!("f32 slice {n} overruns {} byte slab", self.bytes);
        }
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast::<f32>(), n) })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn f32_at_mut(&mut self, n: usize) -> Result<&mut [f32]> {
        let need = n
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("f32 slice overflow"))?;
        if need > self.bytes {
            bail!("f32 slice {n} overruns {} byte slab", self.bytes);
        }
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<f32>(), n) })
    }
}

impl Drop for PageSlab {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct I8GemmSpec {
    n: u32,
    d: u32,
    v: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct MobKanBwdLaunch {
    spec: MobKanSpec,
    in0: u32,
    n_tiles: u32,
    out_tiles: u32,
    tin: u32,
    tok_par: u32,
    pad: [u32; 3],
}

impl std::fmt::Debug for SovereignDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovereignDevice")
            .field("backend", &self.backend)
            .field("name", &self.name)
            .field("fused_grad_ckpt", &self.flags.fused_grad_ckpt)
            .finish()
    }
}

impl SovereignDevice {
    /// Open Metal if requested and available, otherwise the Accelerate CPU path.
    pub fn open(prefer_metal: bool) -> Result<Self> {
        Self::open_with(prefer_metal, DeviceFlags::default())
    }

    pub fn open_with(prefer_metal: bool, flags: DeviceFlags) -> Result<Self> {
        if prefer_metal {
            #[cfg(target_os = "macos")]
            {
                match compile_metal(flags) {
                    Ok(metal) => {
                        let name = metal.device.name().to_string();
                        return Ok(Self {
                            backend: Backend::Metal,
                            name,
                            flags,
                            metal: Some(metal),
                        });
                    }
                    Err(e) => {
                        eprintln!("ullis: Metal fused pipeline unavailable ({e}); CPU/Accelerate fallback");
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = prefer_metal;
            }
        }
        Ok(Self {
            backend: Backend::Cpu,
            name: "cpu+accelerate".into(),
            flags,
            #[cfg(target_os = "macos")]
            metal: None,
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_metal(&self) -> bool {
        self.backend == Backend::Metal
    }

    pub fn fused_grad_ckpt(&self) -> bool {
        self.flags.fused_grad_ckpt
    }

    #[cfg(target_os = "macos")]
    pub fn mtl_device(&self) -> Option<&metal::Device> {
        self.metal.as_ref().map(|m| &m.device)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn mtl_device(&self) -> Option<()> {
        None
    }

    #[cfg(target_os = "macos")]
    pub fn dummy_buffer(&self) -> &metal::Buffer {
        &self
            .metal
            .as_ref()
            .expect("dummy_buffer on Metal device")
            .dummy
    }

    /// Encode `ullis_mob_kan_fused_step` and wait. Buffers must be Shared.
    #[cfg(target_os = "macos")]
    pub fn dispatch_fused_mob_kan(
        &self,
        spec: &MobKanSpec,
        x: &metal::Buffer,
        y: &metal::Buffer,
        w_base: &metal::Buffer,
        w_shared: &metal::Buffer,
        w_routed: &metal::Buffer,
        router: &metal::Buffer,
        centers: &metal::Buffer,
        inv_widths: &metal::Buffer,
        scale_base: &metal::Buffer,
        scale_shared: &metal::Buffer,
        scale_routed: &metal::Buffer,
        weight_half: bool,
    ) -> Result<()> {
        spec.validate()?;
        let inner = self
            .metal
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("dispatch on CPU device"))?;

        let scratch_bytes = (spec.scratch_floats_fwd().max(4) * 4) as u64;
        let pso = if weight_half {
            &inner.pipeline_half
        } else {
            &inner.pipeline
        };
        let tpg = threadgroup_width(pso, spec.out_f as u64, spec.out_tile as u64);
        let groups = metal::MTLSize::new(u64::from(spec.n), 1, 1);
        let threads = metal::MTLSize::new(tpg, 1, 1);

        with_autorelease(|| {
            let cmd = inner.queue.new_command_buffer();
            cmd.set_label("ullis.mob_kan.fused");
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pso);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(y), 0);
            enc.set_buffer(2, Some(w_base), 0);
            enc.set_buffer(3, Some(w_shared), 0);
            enc.set_buffer(4, Some(w_routed), 0);
            enc.set_buffer(5, Some(router), 0);
            enc.set_buffer(6, Some(centers), 0);
            enc.set_buffer(7, Some(scale_base), 0);
            enc.set_buffer(8, Some(scale_shared), 0);
            enc.set_buffer(9, Some(scale_routed), 0);
            enc.set_bytes(
                10,
                size_of::<MobKanSpec>() as u64,
                ptr::from_ref(spec).cast(),
            );
            enc.set_buffer(11, Some(inv_widths), 0);
            enc.set_threadgroup_memory_length(0, scratch_bytes);
            enc.dispatch_thread_groups(groups, threads);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            record_gpu_wait();
            let status = cmd.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                bail!("fused command buffer status {status:?}");
            }
            Ok(())
        })
    }

    /// Encode `ullis_mob_kan_fused_bwd` for one in-tile. `part` is the TG-private
    /// slab (`BwdPartialLayout`). Wait is the host fence before reduce.
    #[cfg(target_os = "macos")]
    pub fn dispatch_fused_mob_kan_bwd(
        &self,
        spec: &MobKanSpec,
        in0: u32,
        tin: u32,
        x: &metal::Buffer,
        dy: &metal::Buffer,
        w_base: &metal::Buffer,
        w_shared: &metal::Buffer,
        w_routed: &metal::Buffer,
        router: &metal::Buffer,
        centers: &metal::Buffer,
        inv_widths: &metal::Buffer,
        scale_base: &metal::Buffer,
        scale_shared: &metal::Buffer,
        scale_routed: &metal::Buffer,
        part: &metal::Buffer,
        weight_half: bool,
    ) -> Result<()> {
        spec.validate()?;
        let inner = self
            .metal
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("dispatch on CPU device"))?;
        let tok_par = spec.bwd_tok_par().max(1);
        let launch = MobKanBwdLaunch {
            spec: *spec,
            in0,
            n_tiles: spec.n_tiles() as u32,
            out_tiles: spec.out_tiles() as u32,
            tin,
            tok_par,
            pad: [0; 3],
        };
        let scratch_bytes = (spec.scratch_floats().max(4) * 4) as u64;
        let pso = if weight_half {
            &inner.pipeline_bwd_half
        } else {
            &inner.pipeline_bwd
        };
        let tpg_y = u64::from(tok_par).max(1);
        let cap = pso.max_total_threads_per_threadgroup().max(1);
        let tpg_x = threadgroup_width(pso, spec.out_f as u64, spec.out_tile as u64)
            .min(cap / tpg_y)
            .max(1);
        let groups = metal::MTLSize::new(u64::from(launch.n_tiles), u64::from(launch.out_tiles), 1);
        let threads = metal::MTLSize::new(tpg_x, tpg_y, 1);
        with_autorelease(|| {
            let cmd = inner.queue.new_command_buffer();
            cmd.set_label("ullis.mob_kan.fused_bwd");
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pso);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(dy), 0);
            enc.set_buffer(2, Some(w_base), 0);
            enc.set_buffer(3, Some(w_shared), 0);
            enc.set_buffer(4, Some(w_routed), 0);
            enc.set_buffer(5, Some(router), 0);
            enc.set_buffer(6, Some(centers), 0);
            enc.set_buffer(7, Some(scale_base), 0);
            enc.set_buffer(8, Some(scale_shared), 0);
            enc.set_buffer(9, Some(scale_routed), 0);
            enc.set_buffer(10, Some(inv_widths), 0);
            enc.set_bytes(
                11,
                size_of::<MobKanBwdLaunch>() as u64,
                ptr::from_ref(&launch).cast(),
            );
            enc.set_buffer(12, Some(part), 0);
            enc.set_threadgroup_memory_length(0, scratch_bytes);
            enc.dispatch_thread_groups(groups, threads);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            record_gpu_wait();
            let status = cmd.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                bail!("fused bwd command buffer status {status:?}");
            }
            Ok(())
        })
    }

    /// Unpack packed-i8 embedding rows for `ids` into `y` (`[n, d]`).
    #[cfg(target_os = "macos")]
    pub fn dispatch_i8_embed_lookup(
        &self,
        codes: &metal::Buffer,
        scale: &metal::Buffer,
        ids: &metal::Buffer,
        y: &metal::Buffer,
        n: u32,
        d: u32,
        v: u32,
    ) -> Result<()> {
        let inner = self
            .metal
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("dispatch on CPU device"))?;
        let spec = I8GemmSpec {
            n,
            d,
            v,
            stride: d.max(1),
        };
        with_autorelease(|| {
            let cmd = inner.queue.new_command_buffer();
            cmd.set_label("ullis.embed.i8");
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&inner.embed_i8);
            enc.set_buffer(0, Some(codes), 0);
            enc.set_buffer(1, Some(scale), 0);
            enc.set_buffer(2, Some(ids), 0);
            enc.set_buffer(3, Some(y), 0);
            enc.set_bytes(
                4,
                size_of::<I8GemmSpec>() as u64,
                ptr::from_ref(&spec).cast(),
            );
            let tpg = inner
                .embed_i8
                .thread_execution_width()
                .max(1)
                .min(n.max(1) as u64);
            enc.dispatch_threads(
                metal::MTLSize::new(u64::from(n.max(1)), 1, 1),
                metal::MTLSize::new(tpg, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            record_gpu_wait();
            if cmd.status() != metal::MTLCommandBufferStatus::Completed {
                bail!("i8 embed command status {:?}", cmd.status());
            }
            Ok(())
        })
    }

    /// Tied logits: `y[n, V] = hidden[n, D] @ (i8_codes[V, D] * scale[V])ᵀ`.
    #[cfg(target_os = "macos")]
    pub fn dispatch_i8_tied_logits(
        &self,
        hidden: &metal::Buffer,
        codes: &metal::Buffer,
        scale: &metal::Buffer,
        logits: &metal::Buffer,
        n: u32,
        d: u32,
        v: u32,
    ) -> Result<()> {
        let inner = self
            .metal
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("dispatch on CPU device"))?;
        let spec = I8GemmSpec {
            n,
            d,
            v,
            stride: d.max(1),
        };
        let tpg = inner
            .logits_i8
            .thread_execution_width()
            .max(1)
            .min(u64::from(v.max(1)));
        with_autorelease(|| {
            let cmd = inner.queue.new_command_buffer();
            cmd.set_label("ullis.logits.i8");
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&inner.logits_i8);
            enc.set_buffer(0, Some(hidden), 0);
            enc.set_buffer(1, Some(codes), 0);
            enc.set_buffer(2, Some(scale), 0);
            enc.set_buffer(3, Some(logits), 0);
            enc.set_bytes(
                4,
                size_of::<I8GemmSpec>() as u64,
                ptr::from_ref(&spec).cast(),
            );
            // All thread-index args are uint2 (`thread_position_in_grid`).
            enc.dispatch_threads(
                metal::MTLSize::new(u64::from(v.max(1)), u64::from(n.max(1)), 1),
                metal::MTLSize::new(tpg, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            record_gpu_wait();
            if cmd.status() != metal::MTLCommandBufferStatus::Completed {
                bail!("i8 logits command status {:?}", cmd.status());
            }
            Ok(())
        })
    }
}

#[cfg(target_os = "macos")]
fn threadgroup_width(pso: &metal::ComputePipelineState, out_f: u64, out_tile: u64) -> u64 {
    let simd = pso.thread_execution_width().max(1);
    let cap = pso.max_total_threads_per_threadgroup().max(1);
    out_f.min(out_tile).min(simd).min(cap).max(1)
}

#[cfg(target_os = "macos")]
fn compile_metal(flags: DeviceFlags) -> Result<MetalInner> {
    with_autorelease(|| {
        let device = metal::Device::system_default().context("MTLCreateSystemDefaultDevice")?;
        let queue = device.new_command_queue();
        queue.set_label("ullis.sovereign.queue");
        let opts = metal::CompileOptions::new();
        opts.set_fast_math_enabled(true);
        opts.set_language_version(metal::MTLLanguageVersion::V2_3);
        // Explicit sub-allocation compiler flag: fused gradient checkpointing.
        // Injected as an MSL preprocessor define so metal-rs 0.29 does not need
        // an NSDictionary for `setPreprocessorMacros`.
        let lib = compile_fused_library(&device, &opts, flags, false)?;
        let lib_h = compile_fused_library(&device, &opts, flags, true)?;
        let pipeline = pso_fn(&device, &lib, "ullis_mob_kan_fused_step")?;
        let pipeline_bwd = pso_fn(&device, &lib, "ullis_mob_kan_fused_bwd")?;
        let pipeline_half = pso_fn(&device, &lib_h, "ullis_mob_kan_fused_step")?;
        let pipeline_bwd_half = pso_fn(&device, &lib_h, "ullis_mob_kan_fused_bwd")?;
        let embed_i8 = pso_fn(&device, &lib, "ullis_i8_embed_lookup")?;
        let logits_i8 = pso_fn(&device, &lib, "ullis_i8_tied_logits")?;
        let dummy = alloc_shared_f32_buffer(&device, 8)?;
        Ok(MetalInner {
            device,
            queue,
            pipeline,
            pipeline_bwd,
            pipeline_half,
            pipeline_bwd_half,
            embed_i8,
            logits_i8,
            dummy,
        })
    })
}

#[cfg(target_os = "macos")]
fn compile_fused_library(
    device: &metal::Device,
    opts: &metal::CompileOptions,
    flags: DeviceFlags,
    w_half: bool,
) -> Result<metal::Library> {
    let src = format!(
        "#define ULLIS_FUSED_GRAD_CKPT {}\n#define ULLIS_W_HALF {}\n{}",
        u32::from(flags.fused_grad_ckpt),
        u32::from(w_half),
        FUSED_MSL
    );
    device
        .new_library_with_source(&src, opts)
        .map_err(|e| anyhow::anyhow!("MSL compile half={w_half}: {e}"))
}

#[cfg(target_os = "macos")]
fn pso_fn(
    device: &metal::Device,
    lib: &metal::Library,
    name: &str,
) -> Result<metal::ComputePipelineState> {
    let function = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("MSL function {name}: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("PSO {name}: {e}"))
}

#[cfg(target_os = "macos")]
fn with_autorelease<T>(f: impl FnOnce() -> T) -> T {
    metal::objc::rc::autoreleasepool(f)
}

/// Shared unified-memory `MTLBuffer` for `n` floats (minimum 16 bytes).
#[cfg(target_os = "macos")]
pub fn alloc_shared_f32_buffer(device: &metal::Device, n_floats: usize) -> Result<metal::Buffer> {
    let bytes = n_floats.max(1).saturating_mul(4).max(16) as u64;
    let opts = metal::MTLResourceOptions::StorageModeShared
        | metal::MTLResourceOptions::CPUCacheModeDefaultCache
        | metal::MTLResourceOptions::HazardTrackingModeTracked;
    let buf = device.new_buffer(bytes, opts);
    if buf.length() < bytes {
        bail!("MTLBuffer length {} < {bytes}", buf.length());
    }
    Ok(buf)
}

#[cfg(target_os = "macos")]
pub fn alloc_shared_bytes(device: &metal::Device, n_bytes: usize) -> Result<metal::Buffer> {
    let bytes = n_bytes.max(16) as u64;
    let opts = metal::MTLResourceOptions::StorageModeShared
        | metal::MTLResourceOptions::CPUCacheModeDefaultCache
        | metal::MTLResourceOptions::HazardTrackingModeTracked;
    let buf = device.new_buffer(bytes, opts);
    if buf.length() < bytes {
        bail!("MTLBuffer length {} < {bytes}", buf.length());
    }
    Ok(buf)
}

/// Zero-copy Shared buffer wrapping a page-aligned host pointer. Caller retains
/// ownership of `bytes` for the lifetime of the returned buffer.
#[cfg(target_os = "macos")]
pub fn wrap_shared_bytes_no_copy(device: &metal::Device, bytes: &[u8]) -> Result<metal::Buffer> {
    if bytes.as_ptr() as usize % 16_384 != 0 {
        bail!("DMA wrap requires 16 KiB alignment");
    }
    let opts = metal::MTLResourceOptions::StorageModeShared
        | metal::MTLResourceOptions::CPUCacheModeDefaultCache
        | metal::MTLResourceOptions::HazardTrackingModeTracked;
    let buf =
        device.new_buffer_with_bytes_no_copy(bytes.as_ptr().cast(), bytes.len() as u64, opts, None);
    Ok(buf)
}

#[cfg(target_os = "macos")]
pub fn write_shared_bytes(buffer: &metal::Buffer, src: &[u8]) -> Result<()> {
    let need = src.len() as u64;
    if need > buffer.length() {
        bail!("write {} bytes into MTLBuffer of {}", need, buffer.length());
    }
    if src.is_empty() {
        return Ok(());
    }
    unsafe {
        let dst = buffer.contents().cast::<u8>();
        if dst.is_null() {
            bail!("MTLBuffer.contents is null");
        }
        ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn write_shared_u32_buffer(buffer: &metal::Buffer, src: &[u32]) -> Result<()> {
    let need = (src.len() * 4) as u64;
    if need > buffer.length() {
        bail!("write {} bytes into MTLBuffer of {}", need, buffer.length());
    }
    if src.is_empty() {
        return Ok(());
    }
    unsafe {
        let dst = buffer.contents().cast::<u32>();
        if dst.is_null() {
            bail!("MTLBuffer.contents is null");
        }
        ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn write_shared_f32_buffer(buffer: &metal::Buffer, src: &[f32]) -> Result<()> {
    let need = (src.len() * 4) as u64;
    if need > buffer.length() {
        bail!("write {} bytes into MTLBuffer of {}", need, buffer.length());
    }
    if src.is_empty() {
        return Ok(());
    }
    unsafe {
        let dst = buffer.contents().cast::<f32>();
        if dst.is_null() {
            bail!("MTLBuffer.contents is null");
        }
        ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn read_shared_f32_buffer(buffer: &metal::Buffer, dst: &mut [f32]) -> Result<()> {
    let need = (dst.len() * 4) as u64;
    if need > buffer.length() {
        bail!("read {} bytes from MTLBuffer of {}", need, buffer.length());
    }
    if dst.is_empty() {
        return Ok(());
    }
    unsafe {
        let src = buffer.contents().cast::<f32>();
        if src.is_null() {
            bail!("MTLBuffer.contents is null");
        }
        ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
    }
    Ok(())
}

/// Fused Mixture-of-Bumps KAN: softmax router + quadratic ReLU bumps + TWN STE.
///
/// One threadgroup per token. Activations live in threadgroup memory; no
/// intermediate bump / gate buffers are allocated in device RAM.
const FUSED_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#ifndef ULLIS_FUSED_GRAD_CKPT
#define ULLIS_FUSED_GRAD_CKPT 0
#endif
#ifndef ULLIS_W_HALF
#define ULLIS_W_HALF 0
#endif
#if ULLIS_W_HALF
typedef half wgt_t;
#else
typedef float wgt_t;
#endif
inline float ld_w(wgt_t w) { return float(w); }

struct MobKanSpec {
    uint n;
    uint in_f;
    uint out_f;
    uint g;
    uint gs;
    uint gr;
    uint k;
    uint g_use;
    uint phase;
    uint coarse;
    uint packed;
    uint tile_in;
    uint out_tile;
    uint n_tile;
    uint topk;
    uint kan_factor;
    uint pad1;
    uint pad2;
    float inv_width;
    float delta_ratio;
};

inline void apply_topk_tg(threadgroup float* gate_s, uint k, uint topk) {
    if (topk == 0u || topk >= k) {
        return;
    }
    uint used = 0u;
    for (uint s = 0u; s < topk; ++s) {
        uint best = 0u;
        float bv = -INFINITY;
        for (uint e = 0u; e < k; ++e) {
            if ((used & (1u << e)) != 0u) {
                continue;
            }
            if (gate_s[e] > bv) {
                bv = gate_s[e];
                best = e;
            }
        }
        used |= 1u << best;
    }
    for (uint e = 0u; e < k; ++e) {
        if ((used & (1u << e)) == 0u) {
            gate_s[e] = 0.0f;
        }
    }
}

inline float relu_sq(float t) {
    float r = max(t, 0.0f);
    return r * r;
}

inline float apply_w(float w, float delta, float scale, uint qat, uint packed) {
    if (packed != 0u) {
        return w * scale;
    }
    if (qat != 0u) {
        float q = 0.0f;
        if (w > delta) {
            q = 1.0f;
        } else if (w < -delta) {
            q = -1.0f;
        }
        return q * scale;
    }
    return w;
}

inline float row_delta(device const wgt_t* row, uint cols, float ratio) {
    float s = 0.0f;
    for (uint i = 0; i < cols; ++i) {
        s += fabs(ld_w(row[i]));
    }
    float inv = 1.0f / float(max(cols, 1u));
    return ratio * s * inv;
}

inline float twn_code(float w, float delta) {
    if (w > delta) {
        return 1.0f;
    }
    if (w < -delta) {
        return -1.0f;
    }
    return 0.0f;
}

inline float ste_w(float w) {
    return fabs(w) <= 1.0f ? 1.0f : 0.0f;
}

inline float qw_of(float w, float delta, float scale, uint qat, uint packed) {
    if (packed != 0u) {
        return w * scale;
    }
    if (qat != 0u) {
        return twn_code(w, delta) * scale;
    }
    return w;
}

inline void bump_pair(float x, float c, float inv, float dpsi, thread float* dx, thread float* dc) {
    float z = (x - c) * inv;
    float u = 1.0f - fabs(z);
    if (u <= 0.0f) {
        return;
    }
    float du = 2.0f * u * dpsi;
    float sgn = (x >= c) ? 1.0f : -1.0f;
    *dx += du * (-inv * sgn);
    *dc += du * (inv * sgn);
}

// tok_par>1: several tokens in one TG accumulate into the same dW slot.
inline void atomic_add_f(device float* p, float v) {
    if (v == 0.0f) {
        return;
    }
    device atomic_uint* a = (device atomic_uint*)p;
    uint old = atomic_load_explicit(a, memory_order_relaxed);
    while (true) {
        uint neu = as_type<uint>(as_type<float>(old) + v);
        if (atomic_compare_exchange_weak_explicit(
                a, &old, neu, memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

struct MobKanBwdLaunch {
    MobKanSpec p;
    uint in0;
    uint n_tiles;
    uint out_tiles;
    uint tin;
    uint tok_par;
    uint pad1;
    uint pad2;
    uint pad3;
};

kernel void ullis_mob_kan_fused_step(
    device const float* x            [[buffer(0)]],
    device       float* y            [[buffer(1)]],
    device const wgt_t* w_base       [[buffer(2)]],
    device const wgt_t* w_shared     [[buffer(3)]],
    device const wgt_t* w_routed     [[buffer(4)]],
    device const wgt_t* router       [[buffer(5)]],
    device const float* centers      [[buffer(6)]],
    device const float* scale_base   [[buffer(7)]],
    device const float* scale_shared [[buffer(8)]],
    device const float* scale_routed [[buffer(9)]],
    constant MobKanSpec& p           [[buffer(10)]],
    device const float* inv_widths   [[buffer(11)]],
    threadgroup float* scratch       [[threadgroup(0)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tpg                         [[threads_per_threadgroup]],
    uint gid                         [[threadgroup_position_in_grid]]
) {
    if (gid >= p.n) {
        return;
    }

    const uint in_f = p.in_f;
    const uint g = p.g;
    const uint tin_cap = p.tile_in;
    const uint otile = p.out_tile;
    // Threadgroup sub-allocation: x[TIN], ψ[TIN·G], gates[K]. Model in_f
    // may exceed TIN; the in0 loop streams tiles. When
    // ULLIS_FUSED_GRAD_CKPT=1 these are the only activation bytes; the host
    // tape keeps layer-boundary x^{(ℓ)} and re-dispatches this kernel to
    // rematerialize ψ / gates on the backward pass.
    threadgroup float* x_s = scratch;
    threadgroup float* bump_s = scratch + tin_cap;
    threadgroup float* gate_s = scratch + tin_cap + tin_cap * g;
    threadgroup float* phi_s = gate_s + max(p.k, 1u);
    threadgroup float* rho_s = phi_s + tin_cap;
    threadgroup float* dbase_s = rho_s + max(p.k, 1u) * tin_cap;
#if ULLIS_FUSED_GRAD_CKPT
    (void)0;
#endif

    device const float* x_n = x + gid * in_f;

    const uint k = p.k;
    const uint gr = p.gr;
    const bool routed = (p.coarse == 0u) && (gr > 0u) && (k > 0u);
    // Router needs the full x (not a TIN slice).
    if (routed && tid == 0u) {
        float logits[4];
        float m = -INFINITY;
        for (uint e = 0; e < k; ++e) {
            float s = 0.0f;
            device const wgt_t* wr = router + e * in_f;
            for (uint i = 0; i < in_f; ++i) {
                s += x_n[i] * ld_w(wr[i]);
            }
            logits[e] = s;
            m = max(m, s);
        }
        float zsum = 0.0f;
        for (uint e = 0; e < k; ++e) {
            float v = exp(logits[e] - m);
            gate_s[e] = v;
            zsum += v;
        }
        float inv = 1.0f / max(zsum, 1e-20f);
        for (uint e = 0; e < k; ++e) {
            gate_s[e] *= inv;
        }
        apply_topk_tg(gate_s, k, p.topk);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_f = p.out_f;
    const uint gs = p.gs;
    const uint g_use = p.g_use;
    const uint qat = (p.phase >= 3u && p.packed == 0u) ? 1u : 0u;
    const uint packed = p.packed;
    const uint sh_len = in_f * gs;
    const uint rt_len = in_f * gr;
    (void)sh_len;
    (void)rt_len;

    if (qat != 0u) {
        for (uint o = tid; o < out_f; o += tpg) {
            dbase_s[o] = row_delta(w_base + o * in_f, in_f, p.delta_ratio);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint in0 = 0u; in0 < in_f; in0 += tin_cap) {
        uint tin = min(tin_cap, in_f - in0);
        for (uint i = tid; i < tin; i += tpg) {
            x_s[i] = x_n[in0 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint n_bumps = tin * g;
        for (uint idx = tid; idx < n_bumps; idx += tpg) {
            uint i = idx / g;
            uint gi = idx - i * g;
            float inv = inv_widths[gi];
            if (inv <= 0.0f || !isfinite(inv)) {
                inv = p.inv_width;
            }
            float z = (x_s[i] - centers[gi]) * inv;
            bump_s[idx] = relu_sq(1.0f - fabs(z));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < tin; i += tpg) {
            float phi = 0.0f;
            device const wgt_t* ws_i = w_shared + (in0 + i) * gs;
            float d_sh = (qat != 0u) ? row_delta(ws_i, gs, p.delta_ratio) : 0.0f;
            float ss = (qat != 0u || packed != 0u) ? scale_shared[in0 + i] : 1.0f;
            for (uint gi = 0; gi < g_use; ++gi) {
                phi += bump_s[i * g + gi] * apply_w(ld_w(ws_i[gi]), d_sh, ss, qat, packed);
            }
            phi_s[i] = phi;
            if (routed) {
                for (uint e = 0; e < k; ++e) {
                    device const wgt_t* wr = w_routed + (e * in_f + in0 + i) * max(gr, 1u);
                    float d_rt = (qat != 0u) ? row_delta(wr, gr, p.delta_ratio) : 0.0f;
                    float sr = (qat != 0u || packed != 0u) ? scale_routed[e * in_f + in0 + i] : 1.0f;
                    float mix = 0.0f;
                    for (uint gi = 0; gi < gr; ++gi) {
                        mix += bump_s[i * g + gs + gi] * apply_w(ld_w(wr[gi]), d_rt, sr, qat, packed);
                    }
                    rho_s[e * tin_cap + i] = mix;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint o0 = 0u; o0 < out_f; o0 += otile) {
            uint o_end = min(o0 + otile, out_f);
            for (uint o = o0 + tid; o < o_end; o += tpg) {
                float acc = (in0 == 0u) ? 0.0f : y[gid * out_f + o];
                device const wgt_t* wb = w_base + o * in_f;
                float d_base = (qat != 0u) ? dbase_s[o] : 0.0f;
                float sb = (qat != 0u || packed != 0u) ? scale_base[o] : 1.0f;
                for (uint i = 0; i < tin; ++i) {
                    float u = x_s[i] + phi_s[i];
                    if (routed) {
                        for (uint e = 0; e < k; ++e) {
                            u += gate_s[e] * rho_s[e * tin_cap + i];
                        }
                    }
                    acc += u * apply_w(ld_w(wb[in0 + i]), d_base, sb, qat, packed);
                }
                y[gid * out_f + o] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void ullis_mob_kan_fused_bwd(
    device const float* x            [[buffer(0)]],
    device const float* dy           [[buffer(1)]],
    device const wgt_t* w_base       [[buffer(2)]],
    device const wgt_t* w_shared     [[buffer(3)]],
    device const wgt_t* w_routed     [[buffer(4)]],
    device const wgt_t* router       [[buffer(5)]],
    device const float* centers      [[buffer(6)]],
    device const float* scale_base   [[buffer(7)]],
    device const float* scale_shared [[buffer(8)]],
    device const float* scale_routed [[buffer(9)]],
    device const float* inv_widths   [[buffer(10)]],
    constant MobKanBwdLaunch& L      [[buffer(11)]],
    device float* part               [[buffer(12)]],
    threadgroup float* scratch       [[threadgroup(0)]],
    uint2 tid                        [[thread_position_in_threadgroup]],
    uint2 tpg                        [[threads_per_threadgroup]],
    uint2 gid                        [[threadgroup_position_in_grid]]
) {
    const MobKanSpec p = L.p;
    const uint nt = gid.x;
    const uint ot = gid.y;
    if (nt >= L.n_tiles || ot >= L.out_tiles) {
        return;
    }
    const uint tid0 = tid.x;
    const uint tpgx = max(tpg.x, 1u);
    const uint py = tid.y;
    const uint tok_par = max(L.tok_par, 1u);
    const uint n = p.n;
    const uint in_f = p.in_f;
    const uint out_f = p.out_f;
    const uint g = p.g;
    const uint gs = p.gs;
    const uint gr = p.gr;
    const uint k = p.k;
    const uint g_use = p.g_use;
    const uint tin_cap = p.tile_in;
    const uint ot_cap = p.out_tile;
    const uint nt_cap = p.n_tile;
    const uint in0 = L.in0;
    const uint tin = L.tin;
    const uint t0 = nt * nt_cap;
    const uint o0 = ot * ot_cap;
    if (t0 >= n || o0 >= out_f) {
        return;
    }
    const uint tn = min(nt_cap, n - t0);
    const uint otn = min(ot_cap, out_f - o0);

    const uint per = tin_cap + tin_cap * g + max(p.k, 1u) + tin_cap + max(p.k, 1u) * tin_cap;
    threadgroup float* slot = (py == 0u)
        ? scratch
        : scratch + per + p.out_f + (py - 1u) * per;
    threadgroup float* x_s = slot;
    threadgroup float* bump_s = slot + tin_cap;
    threadgroup float* gate_s = slot + tin_cap + tin_cap * g;
    threadgroup float* phi_s = gate_s + max(p.k, 1u);
    threadgroup float* rho_s = phi_s + tin_cap;
    threadgroup float* dbase_s = scratch + per;

    const uint qat = (p.phase >= 3u && p.packed == 0u) ? 1u : 0u;
    const uint packed = p.packed;
    const uint scale_on = (qat != 0u || packed != 0u) ? 1u : 0u;
    const uint edge = 1u;
    const bool routed = (p.coarse == 0u) && (gr > 0u) && (k > 0u);
    const uint sh_len = in_f * gs;
    const uint rt_len = in_f * max(gr, 1u);
    const uint gs_a = max(gs, 1u);
    const uint gr_a = max(gr, 1u);
    const uint k_a = max(k, 1u);
    const uint ntg = L.n_tiles * L.out_tiles;
    const uint tg = nt * L.out_tiles + ot;

    uint off = 0u;
    const uint off_base = off;
    off += ntg * ot_cap * tin_cap;
    const uint off_shared = off;
    off += ntg * tin_cap * gs_a;
    const uint off_routed = off;
    off += ntg * k_a * tin_cap * gr_a;
    const uint off_dx = off;
    off += ntg * nt_cap * tin_cap;
    const uint off_dsb = off;
    off += ntg * ot_cap;
    const uint off_dss = off;
    off += ntg * tin_cap;
    const uint off_dsr = off;
    off += ntg * k_a * tin_cap;
    const uint off_dc = off;
    off += ntg * g;
    const uint off_dg = off;

    if (qat != 0u && py == 0u) {
        for (uint lo = tid0; lo < otn; lo += tpgx) {
            uint o = o0 + lo;
            dbase_s[lo] = row_delta(w_base + o * in_f, in_f, p.delta_ratio);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint niter = (tn + tok_par - 1u) / tok_par;
    for (uint it = 0u; it < niter; ++it) {
        uint lt = it * tok_par + py;
        const bool live = lt < tn;
        uint t = t0 + (live ? lt : 0u);
        device const float* x_n = x + t * in_f;
        if (live && routed && tid0 == 0u) {
            float logits[4];
            float m = -INFINITY;
            for (uint e = 0u; e < k; ++e) {
                float s = 0.0f;
                device const wgt_t* wr = router + e * in_f;
                for (uint i = 0u; i < in_f; ++i) {
                    s += x_n[i] * ld_w(wr[i]);
                }
                logits[e] = s;
                m = max(m, s);
            }
            float zsum = 0.0f;
            for (uint e = 0u; e < k; ++e) {
                float v = exp(logits[e] - m);
                gate_s[e] = v;
                zsum += v;
            }
            float inv = 1.0f / max(zsum, 1e-20f);
            for (uint e = 0u; e < k; ++e) {
                gate_s[e] *= inv;
            }
            apply_topk_tg(gate_s, k, p.topk);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (live) {
        for (uint i = tid0; i < tin; i += tpgx) {
            x_s[i] = x_n[in0 + i];
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint n_bumps = tin * g;
        if (live) {
        for (uint idx = tid0; idx < n_bumps; idx += tpgx) {
            uint i = idx / g;
            uint gi = idx - i * g;
            float inv = inv_widths[gi];
            if (inv <= 0.0f || !isfinite(inv)) {
                inv = p.inv_width;
            }
            float z = (x_s[i] - centers[gi]) * inv;
            bump_s[idx] = relu_sq(1.0f - fabs(z));
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (live && edge != 0u) {
            for (uint i = tid0; i < tin; i += tpgx) {
                float phi = 0.0f;
                device const wgt_t* ws_i = w_shared + (in0 + i) * gs;
                float d_sh_i = (qat != 0u) ? row_delta(ws_i, gs, p.delta_ratio) : 0.0f;
                float ssi = (scale_on != 0u) ? scale_shared[in0 + i] : 1.0f;
                for (uint gi = 0u; gi < g_use; ++gi) {
                    phi += bump_s[i * g + gi] * qw_of(ld_w(ws_i[gi]), d_sh_i, ssi, qat, packed);
                }
                phi_s[i] = phi;
                if (routed) {
                    for (uint e = 0u; e < k; ++e) {
                        device const wgt_t* wr = w_routed + (e * in_f + in0 + i) * gr_a;
                        float d_rt_i = (qat != 0u) ? row_delta(wr, gr, p.delta_ratio) : 0.0f;
                        float sri = (scale_on != 0u) ? scale_routed[e * in_f + in0 + i] : 1.0f;
                        float mix = 0.0f;
                        for (uint gi = 0u; gi < gr; ++gi) {
                            mix += bump_s[i * g + gs + gi] * qw_of(ld_w(wr[gi]), d_rt_i, sri, qat, packed);
                        }
                        rho_s[e * tin_cap + i] = mix;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (live) {
        for (uint lo = tid0; lo < otn; lo += tpgx) {
            uint o = o0 + lo;
            float go = dy[t * out_f + o];
            float sbo = (scale_on != 0u) ? scale_base[o] : 1.0f;
            device const wgt_t* wb = w_base + o * in_f;
            float d_base = (qat != 0u) ? dbase_s[lo] : 0.0f;
            float dsb_acc = 0.0f;
            device float* pb = part + off_base + (tg * ot_cap + lo) * tin_cap;
            device float* ps = part + off_shared + tg * tin_cap * gs_a;
            if (edge != 0u) {
                device float* dss_row = part + off_dss + tg * tin_cap;
                for (uint i = 0u; i < tin; ++i) {
                    float u = x_s[i] + phi_s[i];
                    if (routed) {
                        for (uint e = 0u; e < k; ++e) {
                            u += gate_s[e] * rho_s[e * tin_cap + i];
                        }
                    }
                    float w = ld_w(wb[in0 + i]);
                    float qw = qw_of(w, d_base, sbo, qat, packed);
                    float ste = (qat != 0u) ? ste_w(w) : 1.0f;
                    atomic_add_f(pb + i, go * u * sbo * ste);
                    if (scale_on != 0u) {
                        float code = (sbo != 0.0f) ? (qw / sbo) : 0.0f;
                        dsb_acc += go * u * code;
                    }
                    float du = go * qw;
                    device const wgt_t* ws_i = w_shared + (in0 + i) * gs;
                    float d_sh_i = (qat != 0u) ? row_delta(ws_i, gs, p.delta_ratio) : 0.0f;
                    float ssi = (scale_on != 0u) ? scale_shared[in0 + i] : 1.0f;
                    for (uint gi = 0u; gi < g_use; ++gi) {
                        float b = bump_s[i * g + gi];
                        float ww = ld_w(ws_i[gi]);
                        float qs = qw_of(ww, d_sh_i, ssi, qat, packed);
                        float ste_s = (qat != 0u) ? ste_w(ww) : 1.0f;
                        atomic_add_f(ps + i * gs_a + gi, du * b * ssi * ste_s);
                        if (scale_on != 0u) {
                            float code = (ssi != 0.0f) ? (qs / ssi) : 0.0f;
                            atomic_add_f(dss_row + i, du * b * code);
                        }
                    }
                }
                atomic_add_f(part + off_dsb + tg * ot_cap + lo, dsb_acc);
                if (routed) {
                    for (uint e = 0u; e < k; ++e) {
                        float gate = gate_s[e];
                        device float* pr = part + off_routed
                            + ((tg * k_a + e) * tin_cap) * gr_a;
                        device float* dsr_row = part + off_dsr
                            + (tg * k_a + e) * tin_cap;
                        float mix_dg = 0.0f;
                        for (uint i = 0u; i < tin; ++i) {
                            float qw = qw_of(ld_w(wb[in0 + i]), d_base, sbo, qat, packed);
                            float du = go * qw;
                            mix_dg += du * rho_s[e * tin_cap + i];
                            if (gate <= 0.0f) {
                                continue;
                            }
                            device const wgt_t* wr = w_routed + (e * in_f + in0 + i) * gr_a;
                            float d_rt_i = (qat != 0u) ? row_delta(wr, gr, p.delta_ratio) : 0.0f;
                            float sri = (scale_on != 0u) ? scale_routed[e * in_f + in0 + i] : 1.0f;
                            for (uint gi = 0u; gi < gr; ++gi) {
                                float b = bump_s[i * g + gs + gi];
                                float ww = ld_w(wr[gi]);
                                float qr = qw_of(ww, d_rt_i, sri, qat, packed);
                                float ste_r = (qat != 0u) ? ste_w(ww) : 1.0f;
                                atomic_add_f(pr + i * gr_a + gi, du * gate * b * sri * ste_r);
                                if (scale_on != 0u) {
                                    float code = (sri != 0.0f) ? (qr / sri) : 0.0f;
                                    atomic_add_f(dsr_row + i, du * gate * b * code);
                                }
                            }
                        }
                        part[off_dg + (((tg * nt_cap + lt) * ot_cap + lo) * k_a) + e] += mix_dg;
                    }
                }
            } else {
                float sso = (scale_on != 0u) ? scale_shared[o] : 1.0f;
                device const wgt_t* ws = w_shared + o * sh_len;
                float d_sh = (qat != 0u) ? row_delta(ws, sh_len, p.delta_ratio) : 0.0f;
                float dss_acc = 0.0f;
                for (uint i = 0u; i < tin; ++i) {
                    float xv = x_s[i];
                    float w = ld_w(wb[in0 + i]);
                    float qw = qw_of(w, d_base, sbo, qat, packed);
                    float ste = (qat != 0u) ? ste_w(w) : 1.0f;
                    atomic_add_f(pb + i, go * xv * sbo * ste);
                    if (scale_on != 0u) {
                        float code = (sbo != 0.0f) ? (qw / sbo) : 0.0f;
                        dsb_acc += go * xv * code;
                    }
                    for (uint gi = 0u; gi < g_use; ++gi) {
                        float b = bump_s[i * g + gi];
                        float ww = ld_w(ws[(in0 + i) * gs + gi]);
                        float qs = qw_of(ww, d_sh, sso, qat, packed);
                        float ste_s = (qat != 0u) ? ste_w(ww) : 1.0f;
                        atomic_add_f(ps + i * gs_a + gi, go * b * sso * ste_s);
                        if (scale_on != 0u) {
                            float code = (sso != 0.0f) ? (qs / sso) : 0.0f;
                            dss_acc += go * b * code;
                        }
                    }
                }
                atomic_add_f(part + off_dsb + tg * ot_cap + lo, dsb_acc);
                atomic_add_f(part + off_dss + tg * tin_cap, dss_acc);

                if (routed) {
                    for (uint e = 0u; e < k; ++e) {
                        float gate = gate_s[e];
                        if (gate <= 0.0f) {
                            continue;
                        }
                        float sre = (scale_on != 0u) ? scale_routed[e * out_f + o] : 1.0f;
                        device const wgt_t* wr = w_routed + (e * out_f + o) * rt_len;
                        float d_rt = (qat != 0u) ? row_delta(wr, rt_len, p.delta_ratio) : 0.0f;
                        float mix = 0.0f;
                        float dsr_acc = 0.0f;
                        device float* pr = part + off_routed
                            + ((tg * k_a + e) * tin_cap) * gr_a;
                        for (uint i = 0u; i < tin; ++i) {
                            for (uint gi = 0u; gi < gr; ++gi) {
                                float b = bump_s[i * g + gs + gi];
                                float ww = ld_w(wr[(in0 + i) * gr + gi]);
                                float qr = qw_of(ww, d_rt, sre, qat, packed);
                                float ste_r = (qat != 0u) ? ste_w(ww) : 1.0f;
                                atomic_add_f(pr + i * gr_a + gi, go * gate * b * sre * ste_r);
                                if (scale_on != 0u) {
                                    float code = (sre != 0.0f) ? (qr / sre) : 0.0f;
                                    dsr_acc += go * gate * b * code;
                                }
                                mix += b * qr;
                            }
                        }
                        atomic_add_f(part + off_dsr + (tg * k_a + e) * tin_cap, dsr_acc);
                        part[off_dg + (((tg * nt_cap + lt) * ot_cap + lo) * k_a) + e] += go * mix;
                    }
                }
            }
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (live) {
        for (uint i = tid0; i < tin; i += tpgx) {
            float xv = x_s[i];
            float acc = 0.0f;
            float dc_add[16];
            for (uint gi = 0u; gi < g; ++gi) {
                dc_add[gi] = 0.0f;
            }
            if (edge != 0u) {
                for (uint lo = 0u; lo < otn; ++lo) {
                    uint o = o0 + lo;
                    float go = dy[t * out_f + o];
                    float sbo = (scale_on != 0u) ? scale_base[o] : 1.0f;
                    device const wgt_t* wb = w_base + o * in_f;
                    float d_base = (qat != 0u) ? dbase_s[lo] : 0.0f;
                    acc += go * qw_of(ld_w(wb[in0 + i]), d_base, sbo, qat, packed);
                }
                float du = acc;
                device const wgt_t* ws_i = w_shared + (in0 + i) * gs;
                float d_sh_i = (qat != 0u) ? row_delta(ws_i, gs, p.delta_ratio) : 0.0f;
                float ssi = (scale_on != 0u) ? scale_shared[in0 + i] : 1.0f;
                for (uint gi = 0u; gi < g_use; ++gi) {
                    float qs = qw_of(ld_w(ws_i[gi]), d_sh_i, ssi, qat, packed);
                    float inv = inv_widths[gi];
                    if (inv <= 0.0f || !isfinite(inv)) {
                        inv = p.inv_width;
                    }
                    bump_pair(xv, centers[gi], inv, du * qs, &acc, &dc_add[gi]);
                }
                if (routed) {
                    for (uint e = 0u; e < k; ++e) {
                        float gate = gate_s[e];
                        if (gate <= 0.0f) {
                            continue;
                        }
                        device const wgt_t* wr = w_routed + (e * in_f + in0 + i) * gr_a;
                        float d_rt_i = (qat != 0u) ? row_delta(wr, gr, p.delta_ratio) : 0.0f;
                        float sri = (scale_on != 0u) ? scale_routed[e * in_f + in0 + i] : 1.0f;
                        for (uint gi = 0u; gi < gr; ++gi) {
                            float qr = qw_of(ld_w(wr[gi]), d_rt_i, sri, qat, packed);
                            float inv = inv_widths[gs + gi];
                            if (inv <= 0.0f || !isfinite(inv)) {
                                inv = p.inv_width;
                            }
                            bump_pair(
                                xv,
                                centers[gs + gi],
                                inv,
                                du * gate * qr,
                                &acc,
                                &dc_add[gs + gi]
                            );
                        }
                    }
                }
            } else {
                for (uint lo = 0u; lo < otn; ++lo) {
                    uint o = o0 + lo;
                    float go = dy[t * out_f + o];
                    float sbo = (scale_on != 0u) ? scale_base[o] : 1.0f;
                    float sso = (scale_on != 0u) ? scale_shared[o] : 1.0f;
                    device const wgt_t* wb = w_base + o * in_f;
                    device const wgt_t* ws = w_shared + o * sh_len;
                    float d_base = (qat != 0u) ? dbase_s[lo] : 0.0f;
                    float d_sh = (qat != 0u) ? row_delta(ws, sh_len, p.delta_ratio) : 0.0f;
                    float qw = qw_of(ld_w(wb[in0 + i]), d_base, sbo, qat, packed);
                    acc += go * qw;
                    for (uint gi = 0u; gi < g_use; ++gi) {
                        float b = bump_s[i * g + gi];
                        (void)b;
                        float ww = ld_w(ws[(in0 + i) * gs + gi]);
                        float qs = qw_of(ww, d_sh, sso, qat, packed);
                        float inv = inv_widths[gi];
                        if (inv <= 0.0f || !isfinite(inv)) {
                            inv = p.inv_width;
                        }
                        bump_pair(xv, centers[gi], inv, go * qs, &acc, &dc_add[gi]);
                    }
                    if (routed) {
                        for (uint e = 0u; e < k; ++e) {
                            float gate = gate_s[e];
                            if (gate <= 0.0f) {
                                continue;
                            }
                            float sre = (scale_on != 0u) ? scale_routed[e * out_f + o] : 1.0f;
                            device const wgt_t* wr = w_routed + (e * out_f + o) * rt_len;
                            float d_rt = (qat != 0u) ? row_delta(wr, rt_len, p.delta_ratio) : 0.0f;
                            for (uint gi = 0u; gi < gr; ++gi) {
                                float ww = ld_w(wr[(in0 + i) * gr + gi]);
                                float qr = qw_of(ww, d_rt, sre, qat, packed);
                                float inv = inv_widths[gs + gi];
                                if (inv <= 0.0f || !isfinite(inv)) {
                                    inv = p.inv_width;
                                }
                                bump_pair(
                                    xv,
                                    centers[gs + gi],
                                    inv,
                                    go * gate * qr,
                                    &acc,
                                    &dc_add[gs + gi]
                                );
                            }
                        }
                    }
                }
            }
            part[off_dx + (tg * nt_cap + lt) * tin_cap + i] += acc;
            for (uint gi = 0u; gi < g; ++gi) {
                atomic_add_f(part + off_dc + tg * g + gi, dc_add[gi]);
            }
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct I8GemmSpec {
    uint n;
    uint d;
    uint v;
    uint stride;
};

kernel void ullis_i8_embed_lookup(
    device const char*  codes [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const uint*  ids   [[buffer(2)]],
    device       float* y     [[buffer(3)]],
    constant I8GemmSpec& p    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.n) {
        return;
    }
    uint id = ids[gid];
    if (id >= p.v) {
        id = p.v - 1u;
    }
    float s = scale[id];
    uint row_stride = p.stride == 0u ? p.d : p.stride;
    device const char* row = codes + id * row_stride;
    device float* out = y + gid * p.d;
    for (uint j = 0; j < p.d; ++j) {
        out[j] = float(row[j]) * s;
    }
}

kernel void ullis_i8_tied_logits(
    device const float* hidden [[buffer(0)]],
    device const char*  codes  [[buffer(1)]],
    device const float* scale  [[buffer(2)]],
    device       float* logits [[buffer(3)]],
    constant I8GemmSpec& p     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tok = gid.x;
    uint i = gid.y;
    if (i >= p.n || tok >= p.v) {
        return;
    }
    device const float* h = hidden + i * p.d;
    uint row_stride = p.stride == 0u ? p.d : p.stride;
    device const char* row = codes + tok * row_stride;
    float acc = 0.0f;
    for (uint j = 0; j < p.d; ++j) {
        acc += h[j] * float(row[j]);
    }
    logits[i * p.v + tok] = acc * scale[tok];
}
"#;

pub fn setup_device(prefer_metal: bool) -> Result<SovereignDevice> {
    SovereignDevice::open(prefer_metal)
}

pub fn setup_device_with(prefer_metal: bool, flags: DeviceFlags) -> Result<SovereignDevice> {
    SovereignDevice::open_with(prefer_metal, flags)
}

pub fn device_name(device: &SovereignDevice) -> &str {
    if device.is_metal() {
        "metal"
    } else {
        device.name()
    }
}

pub fn synchronize(_device: &SovereignDevice) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerate::mob_kan_fused_cpu;

    fn tiny_spec() -> MobKanSpec {
        MobKanSpec::new(3, 4, 2, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap()
    }

    #[test]
    fn cpu_device_opens() {
        let d = SovereignDevice::open(false).unwrap();
        assert_eq!(d.backend(), Backend::Cpu);
        assert!(!d.is_metal());
    }

    #[test]
    fn fused_msl_compiles_on_apple_silicon() {
        let d = SovereignDevice::open(true).unwrap();
        #[cfg(target_os = "macos")]
        {
            assert!(
                d.is_metal(),
                "Metal fused pipeline must compile on macOS; got {}",
                d.name()
            );
            assert!(!d.name().is_empty());
        }
        let _ = d;
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_fused_matches_accelerate() {
        let gpu = SovereignDevice::open(true).unwrap();
        if !gpu.is_metal() {
            return;
        }
        let spec = tiny_spec();
        let x: Vec<f32> = (0..spec.x_len()).map(|i| (i as f32) * 0.07 - 0.4).collect();
        let w_base: Vec<f32> = (0..spec.w_base_len())
            .map(|i| (i as f32).sin() * 0.2)
            .collect();
        let w_shared: Vec<f32> = (0..spec.w_shared_len())
            .map(|i| (i as f32).cos() * 0.05)
            .collect();
        let w_routed: Vec<f32> = (0..spec.w_routed_len())
            .map(|i| ((i % 7) as f32) * 0.03 - 0.1)
            .collect();
        let router: Vec<f32> = (0..spec.router_len())
            .map(|i| (i as f32) * 0.01 - 0.02)
            .collect();
        let centers = vec![-2.0f32, -0.66, 0.66, 2.0];
        let inv_widths = crate::accelerate::bump_inv_widths(&centers);
        let scale_base = vec![1.0f32; spec.scale_vec_len()];
        let scale_shared = vec![1.0f32; spec.scale_shared_len()];
        let scale_routed = vec![1.0f32; spec.scale_routed_len()];
        let mut y_cpu = vec![0.0f32; spec.y_len()];
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
            &mut y_cpu,
        )
        .unwrap();

        let mtl = gpu.mtl_device().unwrap();
        let bx = alloc_shared_f32_buffer(mtl, x.len()).unwrap();
        let by = alloc_shared_f32_buffer(mtl, y_cpu.len()).unwrap();
        let bwb = alloc_shared_f32_buffer(mtl, w_base.len()).unwrap();
        let bws = alloc_shared_f32_buffer(mtl, w_shared.len()).unwrap();
        let bwr = alloc_shared_f32_buffer(mtl, w_routed.len()).unwrap();
        let brt = alloc_shared_f32_buffer(mtl, router.len()).unwrap();
        let bc = alloc_shared_f32_buffer(mtl, centers.len()).unwrap();
        let biw = alloc_shared_f32_buffer(mtl, inv_widths.len()).unwrap();
        let bsb = alloc_shared_f32_buffer(mtl, scale_base.len()).unwrap();
        let bss = alloc_shared_f32_buffer(mtl, scale_shared.len()).unwrap();
        let bsr = alloc_shared_f32_buffer(mtl, scale_routed.len()).unwrap();
        write_shared_f32_buffer(&bx, &x).unwrap();
        write_shared_f32_buffer(&bwb, &w_base).unwrap();
        write_shared_f32_buffer(&bws, &w_shared).unwrap();
        write_shared_f32_buffer(&bwr, &w_routed).unwrap();
        write_shared_f32_buffer(&brt, &router).unwrap();
        write_shared_f32_buffer(&bc, &centers).unwrap();
        write_shared_f32_buffer(&biw, &inv_widths).unwrap();
        write_shared_f32_buffer(&bsb, &scale_base).unwrap();
        write_shared_f32_buffer(&bss, &scale_shared).unwrap();
        write_shared_f32_buffer(&bsr, &scale_routed).unwrap();

        gpu.dispatch_fused_mob_kan(
            &spec, &bx, &by, &bwb, &bws, &bwr, &brt, &bc, &biw, &bsb, &bss, &bsr, false,
        )
        .unwrap();
        let mut y_gpu = vec![0.0f32; spec.y_len()];
        read_shared_f32_buffer(&by, &mut y_gpu).unwrap();
        for (a, b) in y_cpu.iter().zip(y_gpu.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "cpu {a} vs metal {b} delta {}",
                (a - b).abs()
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_fused_d512_matches_accelerate() {
        let gpu = SovereignDevice::open(true).unwrap();
        if !gpu.is_metal() {
            return;
        }
        let spec = MobKanSpec::new(2, 512, 64, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        assert!(spec.tile_in < spec.in_f);
        assert!(spec.out_tile < spec.out_f);
        let x: Vec<f32> = (0..spec.x_len()).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let w_base: Vec<f32> = (0..spec.w_base_len())
            .map(|i| (i as f32).sin() * 0.05)
            .collect();
        let w_shared: Vec<f32> = (0..spec.w_shared_len())
            .map(|i| (i as f32).cos() * 0.02)
            .collect();
        let w_routed: Vec<f32> = (0..spec.w_routed_len())
            .map(|i| ((i % 11) as f32) * 0.01 - 0.05)
            .collect();
        let router: Vec<f32> = (0..spec.router_len())
            .map(|i| (i as f32) * 0.002 - 0.01)
            .collect();
        let centers = vec![-2.0f32, -0.66, 0.66, 2.0];
        let inv_widths = crate::accelerate::bump_inv_widths(&centers);
        let scale_base = vec![1.0f32; spec.scale_vec_len()];
        let scale_shared = vec![1.0f32; spec.scale_shared_len()];
        let scale_routed = vec![1.0f32; spec.scale_routed_len()];
        let mut y_cpu = vec![0.0f32; spec.y_len()];
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
            &mut y_cpu,
        )
        .unwrap();

        let mtl = gpu.mtl_device().unwrap();
        let bx = alloc_shared_f32_buffer(mtl, x.len()).unwrap();
        let by = alloc_shared_f32_buffer(mtl, y_cpu.len()).unwrap();
        let bwb = alloc_shared_f32_buffer(mtl, w_base.len()).unwrap();
        let bws = alloc_shared_f32_buffer(mtl, w_shared.len()).unwrap();
        let bwr = alloc_shared_f32_buffer(mtl, w_routed.len()).unwrap();
        let brt = alloc_shared_f32_buffer(mtl, router.len()).unwrap();
        let bc = alloc_shared_f32_buffer(mtl, centers.len()).unwrap();
        let biw = alloc_shared_f32_buffer(mtl, inv_widths.len()).unwrap();
        let bsb = alloc_shared_f32_buffer(mtl, scale_base.len()).unwrap();
        let bss = alloc_shared_f32_buffer(mtl, scale_shared.len()).unwrap();
        let bsr = alloc_shared_f32_buffer(mtl, scale_routed.len()).unwrap();
        write_shared_f32_buffer(&bx, &x).unwrap();
        write_shared_f32_buffer(&bwb, &w_base).unwrap();
        write_shared_f32_buffer(&bws, &w_shared).unwrap();
        write_shared_f32_buffer(&bwr, &w_routed).unwrap();
        write_shared_f32_buffer(&brt, &router).unwrap();
        write_shared_f32_buffer(&bc, &centers).unwrap();
        write_shared_f32_buffer(&biw, &inv_widths).unwrap();
        write_shared_f32_buffer(&bsb, &scale_base).unwrap();
        write_shared_f32_buffer(&bss, &scale_shared).unwrap();
        write_shared_f32_buffer(&bsr, &scale_routed).unwrap();

        gpu.dispatch_fused_mob_kan(
            &spec, &bx, &by, &bwb, &bws, &bwr, &brt, &bc, &biw, &bsb, &bss, &bsr, false,
        )
        .unwrap();
        let mut y_gpu = vec![0.0f32; spec.y_len()];
        read_shared_f32_buffer(&by, &mut y_gpu).unwrap();
        let max = y_cpu
            .iter()
            .zip(y_gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "d512 metal vs cpu max|Δ|={max}");
    }
}
