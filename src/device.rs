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

pub fn rng_from_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
}

/// Runtime GPU / host device owning the fused MoB-KAN pipeline.
pub struct SovereignDevice {
    backend: Backend,
    name: String,
    #[cfg(target_os = "macos")]
    metal: Option<MetalInner>,
}

#[cfg(target_os = "macos")]
struct MetalInner {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
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
    pad: u32,
}

impl std::fmt::Debug for SovereignDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovereignDevice")
            .field("backend", &self.backend)
            .field("name", &self.name)
            .finish()
    }
}

impl SovereignDevice {
    /// Open Metal if requested and available, otherwise the Accelerate CPU path.
    pub fn open(prefer_metal: bool) -> Result<Self> {
        if prefer_metal {
            #[cfg(target_os = "macos")]
            {
                match compile_metal() {
                    Ok(metal) => {
                        let name = metal.device.name().to_string();
                        return Ok(Self {
                            backend: Backend::Metal,
                            name,
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
    ) -> Result<()> {
        spec.validate()?;
        let inner = self
            .metal
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("dispatch on CPU device"))?;

        let scratch_bytes = (spec.scratch_floats().max(4) * 4) as u64;
        let tpg = threadgroup_width(&inner.pipeline, spec.out_f as u64);
        let groups = metal::MTLSize::new(u64::from(spec.n), 1, 1);
        let threads = metal::MTLSize::new(tpg, 1, 1);

        with_autorelease(|| {
            let cmd = inner.queue.new_command_buffer();
            cmd.set_label("ullis.mob_kan.fused");
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&inner.pipeline);
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
            let status = cmd.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                bail!("fused command buffer status {status:?}");
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
        let spec = I8GemmSpec { n, d, v, pad: 0 };
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
        let spec = I8GemmSpec { n, d, v, pad: 0 };
        let tpg = inner
            .logits_i8
            .thread_execution_width()
            .max(1)
            .min(u64::from(v.max(1)));
        let groups_x = u64::from(v).div_ceil(tpg).max(1);
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
            enc.dispatch_thread_groups(
                metal::MTLSize::new(groups_x, u64::from(n.max(1)), 1),
                metal::MTLSize::new(tpg, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if cmd.status() != metal::MTLCommandBufferStatus::Completed {
                bail!("i8 logits command status {:?}", cmd.status());
            }
            Ok(())
        })
    }
}

#[cfg(target_os = "macos")]
fn threadgroup_width(pso: &metal::ComputePipelineState, out_f: u64) -> u64 {
    let simd = pso.thread_execution_width().max(1);
    let cap = pso.max_total_threads_per_threadgroup().max(1);
    out_f.min(simd).min(cap).max(1)
}

#[cfg(target_os = "macos")]
fn compile_metal() -> Result<MetalInner> {
    with_autorelease(|| {
        let device = metal::Device::system_default().context("MTLCreateSystemDefaultDevice")?;
        let queue = device.new_command_queue();
        queue.set_label("ullis.sovereign.queue");
        let opts = metal::CompileOptions::new();
        opts.set_fast_math_enabled(true);
        opts.set_language_version(metal::MTLLanguageVersion::V2_3);
        let library = device
            .new_library_with_source(FUSED_MSL, &opts)
            .map_err(|e| anyhow::anyhow!("MSL compile: {e}"))?;
        let function = library
            .get_function("ullis_mob_kan_fused_step", None)
            .map_err(|e| anyhow::anyhow!("MSL function: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("PSO: {e}"))?;
        let embed_fn = library
            .get_function("ullis_i8_embed_lookup", None)
            .map_err(|e| anyhow::anyhow!("MSL embed i8: {e}"))?;
        let embed_i8 = device
            .new_compute_pipeline_state_with_function(&embed_fn)
            .map_err(|e| anyhow::anyhow!("PSO embed i8: {e}"))?;
        let logits_fn = library
            .get_function("ullis_i8_tied_logits", None)
            .map_err(|e| anyhow::anyhow!("MSL logits i8: {e}"))?;
        let logits_i8 = device
            .new_compute_pipeline_state_with_function(&logits_fn)
            .map_err(|e| anyhow::anyhow!("PSO logits i8: {e}"))?;
        let dummy = alloc_shared_f32_buffer(&device, 8)?;
        Ok(MetalInner {
            device,
            queue,
            pipeline,
            embed_i8,
            logits_i8,
            dummy,
        })
    })
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
    uint pad;
    float inv_width;
    float delta_ratio;
};

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

inline float row_delta(device const float* row, uint cols, float ratio) {
    float s = 0.0f;
    for (uint i = 0; i < cols; ++i) {
        s += fabs(row[i]);
    }
    float inv = 1.0f / float(max(cols, 1u));
    return ratio * s * inv;
}

kernel void ullis_mob_kan_fused_step(
    device const float* x            [[buffer(0)]],
    device       float* y            [[buffer(1)]],
    device const float* w_base       [[buffer(2)]],
    device const float* w_shared     [[buffer(3)]],
    device const float* w_routed     [[buffer(4)]],
    device const float* router       [[buffer(5)]],
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
    threadgroup float* x_s = scratch;
    threadgroup float* bump_s = scratch + in_f;
    threadgroup float* gate_s = scratch + in_f + in_f * g;

    device const float* x_n = x + gid * in_f;
    for (uint i = tid; i < in_f; i += tpg) {
        x_s[i] = x_n[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_bumps = in_f * g;
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

    const uint k = p.k;
    const uint gr = p.gr;
    const bool routed = (p.coarse == 0u) && (gr > 0u) && (k > 0u);
    if (routed && tid == 0u) {
        float logits[4];
        float m = -INFINITY;
        for (uint e = 0; e < k; ++e) {
            float s = 0.0f;
            device const float* wr = router + e * in_f;
            for (uint i = 0; i < in_f; ++i) {
                s += x_s[i] * wr[i];
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_f = p.out_f;
    const uint gs = p.gs;
    const uint g_use = p.g_use;
    const uint qat = (p.phase >= 3u && p.packed == 0u) ? 1u : 0u;
    const uint packed = p.packed;
    const uint sh_len = in_f * gs;
    const uint rt_len = in_f * gr;

    for (uint o = tid; o < out_f; o += tpg) {
        float acc = 0.0f;

        device const float* wb = w_base + o * in_f;
        float d_base = (qat != 0u) ? row_delta(wb, in_f, p.delta_ratio) : 0.0f;
        float sb = (qat != 0u || packed != 0u) ? scale_base[o] : 1.0f;
        for (uint i = 0; i < in_f; ++i) {
            acc += x_s[i] * apply_w(wb[i], d_base, sb, qat, packed);
        }

        device const float* ws = w_shared + o * sh_len;
        float d_sh = (qat != 0u) ? row_delta(ws, sh_len, p.delta_ratio) : 0.0f;
        float ss = (qat != 0u || packed != 0u) ? scale_shared[o] : 1.0f;
        for (uint i = 0; i < in_f; ++i) {
            for (uint gi = 0; gi < g_use; ++gi) {
                float b = bump_s[i * g + gi];
                float w = ws[i * gs + gi];
                acc += b * apply_w(w, d_sh, ss, qat, packed);
            }
        }

        if (routed) {
            for (uint e = 0; e < k; ++e) {
                device const float* wr = w_routed + (e * out_f + o) * rt_len;
                float d_rt = (qat != 0u) ? row_delta(wr, rt_len, p.delta_ratio) : 0.0f;
                float sr = (qat != 0u || packed != 0u) ? scale_routed[e * out_f + o] : 1.0f;
                float mix = 0.0f;
                for (uint i = 0; i < in_f; ++i) {
                    for (uint gi = 0; gi < gr; ++gi) {
                        float b = bump_s[i * g + gs + gi];
                        float w = wr[i * gr + gi];
                        mix += b * apply_w(w, d_rt, sr, qat, packed);
                    }
                }
                acc += gate_s[e] * mix;
            }
        }

        y[gid * out_f + o] = acc;
    }
}

struct I8GemmSpec {
    uint n;
    uint d;
    uint v;
    uint pad;
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
    device const char* row = codes + id * p.d;
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
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint i = gid.y;
    uint tok = gid.x * tpg + tid;
    if (i >= p.n || tok >= p.v) {
        return;
    }
    device const float* h = hidden + i * p.d;
    device const char* row = codes + tok * p.d;
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
        if d.is_metal() {
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
        let scale_shared = vec![1.0f32; spec.scale_vec_len()];
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
            &spec, &bx, &by, &bwb, &bws, &bwr, &brt, &bc, &biw, &bsb, &bss, &bsr,
        )
        .unwrap();
        let mut y_gpu = vec![0.0f32; spec.y_len()];
        read_shared_f32_buffer(&by, &mut y_gpu).unwrap();
        for (a, b) in y_cpu.iter().zip(y_gpu.iter()) {
            assert!(
                (a - b).abs() < 2e-4,
                "cpu {a} vs metal {b} delta {}",
                (a - b).abs()
            );
        }
    }
}
