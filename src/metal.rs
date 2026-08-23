//! Metal forward-path admission and shader-pipeline validation.
//!
//! This module intentionally starts with pipeline construction only. Buffer
//! mapping is the one place where Metal requires raw pointers; it will live in
//! a small, audited follow-up boundary rather than weakening the safe CPU model
//! API throughout the crate.

use anyhow::{bail, Result};

#[cfg(target_os = "macos")]
use crate::hyena::HyenaFftPlan;

/// A validated one-dimensional dispatch.
///
/// The first GPU kernels operate on flattened `[batch, time, channels]`
/// tensors, so every index is representable by Metal's 32-bit `uint` without
/// relying on truncating casts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalDispatchShape {
    pub batch: usize,
    pub time: usize,
    pub channels: usize,
}

impl MetalDispatchShape {
    pub fn new(batch: usize, time: usize, channels: usize) -> Result<Self> {
        if batch == 0 || time == 0 || channels == 0 {
            bail!("Metal dispatch dimensions must be non-zero");
        }
        let elements = batch
            .checked_mul(time)
            .and_then(|rows| rows.checked_mul(channels))
            .ok_or_else(|| anyhow::anyhow!("Metal dispatch shape overflow"))?;
        if u32::try_from(elements).is_err() {
            bail!("Metal dispatch has more than u32::MAX elements");
        }
        Ok(Self {
            batch,
            time,
            channels,
        })
    }

    pub fn elements(self) -> usize {
        self.batch * self.time * self.channels
    }
}

/// The first compiled pipeline. It is an elementwise identity kernel used to
/// prove buffer layout and dispatch mechanics before replacing it with fused
/// RMSNorm/ternary and FFT stages.
pub const IDENTITY_KERNEL_NAME: &str = "ullis_identity";
pub const RMS_NORM_KERNEL_NAME: &str = "ullis_rms_norm";
pub const TERNARY_LINEAR_KERNEL_NAME: &str = "ullis_ternary_linear";
pub const RMS_NORM_TERNARY_LINEAR_KERNEL_NAME: &str = "ullis_rms_norm_ternary_linear";
pub const FFT_BITREVERSE_KERNEL_NAME: &str = "ullis_fft_bitreverse";
pub const FFT_STAGE_KERNEL_NAME: &str = "ullis_fft_stage";
pub const FFT_COMPLEX_MULTIPLY_KERNEL_NAME: &str = "ullis_fft_complex_multiply";
pub const FFT_EXTRACT_CAUSAL_KERNEL_NAME: &str = "ullis_fft_extract_causal";
pub const IMPLICIT_FILTER_KERNEL_NAME: &str = "ullis_generate_implicit_filter";
pub const TANH_GATE_KERNEL_NAME: &str = "ullis_tanh_gate_in_place";
pub const PACK_STRIDED_REAL_KERNEL_NAME: &str = "ullis_pack_strided_real_to_complex";
pub const APPLY_GATE_KERNEL_NAME: &str = "ullis_apply_gate";
pub const RESIDUAL_ADD_KERNEL_NAME: &str = "ullis_residual_add";
pub const HYENA_METAL_SOURCE: &str = include_str!("metal/hyena.metal");

/// Checked dimensions for one packed-ternary projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TernaryLinearShape {
    pub rows: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl TernaryLinearShape {
    pub fn new(rows: usize, in_features: usize, out_features: usize) -> Result<Self> {
        MetalDispatchShape::new(rows, 1, out_features)?;
        if in_features == 0 || u32::try_from(in_features).is_err() {
            bail!("Metal ternary input width is invalid");
        }
        Ok(Self {
            rows,
            in_features,
            out_features,
        })
    }

    pub fn packed_words(self) -> Result<usize> {
        self.in_features
            .checked_mul(self.out_features)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary weight shape overflow"))
            .map(|weights| weights.div_ceil(64))
    }
}

/// CPU reference for the exact packed-bitplane convention used by the Metal
/// ternary shader. It is kept public for GPU equivalence tests and contains no
/// model state or allocation beyond its output.
pub fn ternary_reference(
    input: &[f32],
    positive: &[u64],
    negative: &[u64],
    scales: &[f32],
    shape: TernaryLinearShape,
) -> Result<Vec<f32>> {
    let input_len = shape
        .rows
        .checked_mul(shape.in_features)
        .ok_or_else(|| anyhow::anyhow!("ternary input shape overflow"))?;
    let output_len = shape
        .rows
        .checked_mul(shape.out_features)
        .ok_or_else(|| anyhow::anyhow!("ternary output shape overflow"))?;
    let weights = shape
        .in_features
        .checked_mul(shape.out_features)
        .ok_or_else(|| anyhow::anyhow!("ternary weight shape overflow"))?;
    if input.len() != input_len
        || positive.len() != shape.packed_words()?
        || negative.len() != shape.packed_words()?
        || scales.len() != shape.out_features
    {
        bail!("ternary reference shape mismatch");
    }
    let mut output = vec![0.0; output_len];
    for row in 0..shape.rows {
        for out in 0..shape.out_features {
            let mut sum = 0.0;
            for i in 0..shape.in_features {
                let w = out * shape.in_features + i;
                let bit = 1_u64 << (w % 64);
                let code = if positive[w / 64] & bit != 0 {
                    1.0
                } else if negative[w / 64] & bit != 0 {
                    -1.0
                } else {
                    0.0
                };
                sum += input[row * shape.in_features + i] * code;
            }
            output[row * shape.out_features + out] = sum * scales[out];
        }
    }
    debug_assert_eq!(weights.div_ceil(64), positive.len());
    Ok(output)
}

/// CPU reference for fused RMSNorm and packed ternary projection. The
/// normalized row is never materialized, mirroring the GPU kernel's memory
/// contract.
pub fn rms_norm_ternary_reference(
    input: &[f32],
    positive: &[u64],
    negative: &[u64],
    scales: &[f32],
    shape: TernaryLinearShape,
) -> Result<Vec<f32>> {
    let input_len = shape
        .rows
        .checked_mul(shape.in_features)
        .ok_or_else(|| anyhow::anyhow!("fused ternary input shape overflow"))?;
    if input.len() != input_len
        || positive.len() != shape.packed_words()?
        || negative.len() != shape.packed_words()?
        || scales.len() != shape.out_features
    {
        bail!("fused ternary reference shape mismatch");
    }
    let output_len = shape
        .rows
        .checked_mul(shape.out_features)
        .ok_or_else(|| anyhow::anyhow!("fused ternary output shape overflow"))?;
    let mut output = vec![0.0; output_len];
    for row in 0..shape.rows {
        let source = &input[row * shape.in_features..(row + 1) * shape.in_features];
        let inverse_rms = (source.iter().map(|value| value * value).sum::<f32>()
            / shape.in_features as f32
            + 1e-5)
            .sqrt()
            .recip();
        for out in 0..shape.out_features {
            let mut sum = 0.0;
            for (i, value) in source.iter().enumerate() {
                let weight = out * shape.in_features + i;
                let bit = 1_u64 << (weight % 64);
                let code = if positive[weight / 64] & bit != 0 {
                    1.0
                } else if negative[weight / 64] & bit != 0 {
                    -1.0
                } else {
                    0.0
                };
                sum += value * code;
            }
            output[row * shape.out_features + out] = sum * inverse_rms * scales[out];
        }
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
pub fn validate_metal_pipeline(shape: MetalDispatchShape) -> Result<usize> {
    validate_metal_kernel(IDENTITY_KERNEL_NAME, shape)
}

/// Compiles a named Ullis MSL entry point and checks its dispatch capacity.
/// This admits a kernel before it is allowed into the model execution path.
#[cfg(target_os = "macos")]
pub fn validate_metal_kernel(kernel_name: &str, shape: MetalDispatchShape) -> Result<usize> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary,
    };

    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
    let source = NSString::from_str(HYENA_METAL_SOURCE);
    let options = MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| anyhow::anyhow!("Metal shader compilation failed: {error}"))?;
    let name = NSString::from_str(kernel_name);
    let function = library
        .newFunctionWithName(&name)
        .ok_or_else(|| anyhow::anyhow!("Metal function {kernel_name:?} is missing"))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| anyhow::anyhow!("Metal pipeline creation failed: {error}"))?;
    let width = pipeline.maxTotalThreadsPerThreadgroup();
    if width == 0 {
        bail!("Metal pipeline reported zero threads per threadgroup");
    }
    // The checked shape is deliberately consumed here: later dispatch code can
    // use the same type without a second unchecked `usize -> uint` conversion.
    let _ = shape.elements();
    Ok(width)
}

#[cfg(not(target_os = "macos"))]
pub fn validate_metal_pipeline(_shape: MetalDispatchShape) -> Result<usize> {
    bail!("Ullis Metal backend requires macOS on Apple Silicon")
}

#[cfg(not(target_os = "macos"))]
pub fn validate_metal_kernel(_kernel_name: &str, _shape: MetalDispatchShape) -> Result<usize> {
    bail!("Ullis Metal backend requires macOS on Apple Silicon")
}

/// Executes the stage-zero Metal kernel and returns a fresh output vector.
///
/// This is intentionally a correctness harness, not the final tensor runtime:
/// it proves the owned-buffer, command-buffer, and `[B,T,D]` dispatch contract
/// against the CPU before more complex kernels are introduced.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn execute_reference_kernel(
    input: &[f32],
    kernel_name: &str,
    grid_width: usize,
    scalars: &[u32],
) -> Result<Vec<f32>> {
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLResourceOptions, MTLSize,
    };

    MetalDispatchShape::new(1, input.len(), 1)?;
    if grid_width == 0 || u32::try_from(grid_width).is_err() {
        bail!("Metal grid width is invalid");
    }
    let bytes = input
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("Metal buffer byte size overflow"))?;
    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
    let source = NSString::from_str(HYENA_METAL_SOURCE);
    let options = MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| anyhow::anyhow!("Metal shader compilation failed: {error}"))?;
    let name = NSString::from_str(kernel_name);
    let function = library
        .newFunctionWithName(&name)
        .ok_or_else(|| anyhow::anyhow!("Metal identity function is missing"))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| anyhow::anyhow!("Metal pipeline creation failed: {error}"))?;
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| anyhow::anyhow!("Metal command queue is unavailable"))?;
    let input_buffer = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| anyhow::anyhow!("Metal input buffer allocation failed"))?;
    let output_buffer = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| anyhow::anyhow!("Metal output buffer allocation failed"))?;

    // SAFETY: Shared buffers are allocated for exactly `bytes`, which was
    // checked from `input.len() * size_of::<f32>()`; both pointers stay valid
    // while their retained buffers are in scope and Metal has not been sent a
    // command buffer yet.
    unsafe {
        input_buffer
            .contents()
            .cast::<f32>()
            .as_ptr()
            .copy_from_nonoverlapping(input.as_ptr(), input.len());
    }

    let command = queue
        .commandBuffer()
        .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
    encoder.setComputePipelineState(&pipeline);
    // SAFETY: indices 0 and 1 are the shared buffers in every reference MSL
    // kernel. Scalar words are copied synchronously into consecutive slots.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
        for (offset, scalar) in scalars.iter().enumerate() {
            encoder.setBytes_length_atIndex(
                NonNull::from(scalar).cast::<c_void>(),
                size_of::<u32>(),
                offset + 2,
            );
        }
    }
    let thread_width = pipeline.maxTotalThreadsPerThreadgroup().min(grid_width);
    if thread_width == 0 {
        bail!("Metal pipeline reported zero threads per threadgroup");
    }
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: grid_width,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    if let Some(error) = command.error() {
        bail!("Metal identity command failed: {error}");
    }

    let mut output = vec![0.0; input.len()];
    // SAFETY: GPU work is complete, `output_buffer` remains retained, and the
    // destination vector has exactly the number of initialized `f32` slots
    // represented by the source buffer.
    unsafe {
        output.as_mut_ptr().copy_from_nonoverlapping(
            output_buffer.contents().cast::<f32>().as_ptr(),
            output.len(),
        );
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
pub fn identity_forward(input: &[f32]) -> Result<Vec<f32>> {
    let elements = u32::try_from(input.len())
        .map_err(|_| anyhow::anyhow!("Metal element count exceeds u32"))?;
    execute_reference_kernel(input, IDENTITY_KERNEL_NAME, input.len(), &[elements])
}

/// GPU numerical-reference RMSNorm over contiguous rows.
#[cfg(target_os = "macos")]
pub fn rms_norm_forward(input: &[f32], rows: usize, channels: usize) -> Result<Vec<f32>> {
    let shape = MetalDispatchShape::new(rows, 1, channels)?;
    if input.len() != shape.elements() {
        bail!("RMSNorm input shape mismatch");
    }
    let rows = u32::try_from(rows).map_err(|_| anyhow::anyhow!("RMSNorm rows exceed u32"))?;
    let channels =
        u32::try_from(channels).map_err(|_| anyhow::anyhow!("RMSNorm channels exceed u32"))?;
    execute_reference_kernel(
        input,
        RMS_NORM_KERNEL_NAME,
        rows as usize,
        &[rows, channels],
    )
}

/// Reusable Metal objects for the hot ternary projection path.
///
/// The runtime is intentionally single-threaded (`RefCell` protects its
/// scratch buffers). A trainer should own one runtime on its dispatch thread;
/// this prevents hidden locks and makes resource lifetime explicit.
#[cfg(target_os = "macos")]
pub struct MetalRuntime {
    device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    queue: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    ternary_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    fused_rms_norm_ternary_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    fft_bitreverse_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    fft_stage_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    fft_multiply_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    fft_extract_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    implicit_filter_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    tanh_gate_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pack_strided_real_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    apply_gate_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    residual_add_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    ternary_buffers: std::cell::RefCell<TernaryBuffers>,
    fft_buffers: std::cell::RefCell<FftBuffers>,
    filter_fft_buffers: std::cell::RefCell<FftBuffers>,
    hyena_output_buffer: std::cell::RefCell<OutputBuffer>,
    implicit_filter_parameters: std::cell::RefCell<ImplicitFilterParameters>,
    gate_buffers: std::cell::RefCell<GateBuffers>,
    activations: std::cell::RefCell<ActivationBuffers>,
}

/// Which resident activation slot owns the current residual stream.  A block
/// always writes the next stream to the other slot, preventing accidental
/// in-place residual aliasing at the Rust API boundary.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentActivationSlot {
    First,
    Second,
}

#[cfg(target_os = "macos")]
impl ResidentActivationSlot {
    pub const fn other(self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::First,
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct TernaryBuffers {
    input: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    positive:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    negative:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    scales: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    output: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    input_capacity: usize,
    packed_capacity: usize,
    scale_capacity: usize,
    output_capacity: usize,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct FftBuffers {
    first: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    second: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    capacity: usize,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct OutputBuffer {
    buffer: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    capacity: usize,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct ImplicitFilterParameters {
    freq: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    phase: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    decay: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    capacity: usize,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct GateBuffers {
    input: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    output: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    capacity: usize,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct ActivationBuffers {
    first: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    second: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    capacity: usize,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Complex32 {
    real: f32,
    imaginary: f32,
}

#[cfg(target_os = "macos")]
enum HyenaFilterSource<'a> {
    Dense(&'a [f32]),
    Implicit(&'a crate::hyena::ImplicitFilter),
}

#[cfg(target_os = "macos")]
impl FftBuffers {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        bytes: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};

        if self.capacity < bytes {
            let shared = MTLResourceOptions::StorageModeShared;
            self.first = Some(
                device
                    .newBufferWithLength_options(bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal FFT source allocation failed"))?,
            );
            self.second = Some(
                device
                    .newBufferWithLength_options(bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal FFT scratch allocation failed"))?,
            );
            self.capacity = bytes;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl OutputBuffer {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        bytes: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};

        if self.capacity < bytes {
            self.buffer = Some(
                device
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| anyhow::anyhow!("Metal Hyena output allocation failed"))?,
            );
            self.capacity = bytes;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl ImplicitFilterParameters {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        bytes: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};
        if self.capacity < bytes {
            let options = MTLResourceOptions::StorageModeShared;
            self.freq = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal implicit-filter frequency allocation failed")
                    })?,
            );
            self.phase = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal implicit-filter phase allocation failed")
                    })?,
            );
            self.decay = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal implicit-filter decay allocation failed")
                    })?,
            );
            self.capacity = bytes;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl GateBuffers {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        bytes: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};
        if self.capacity < bytes {
            let options = MTLResourceOptions::StorageModeShared;
            self.input = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| anyhow::anyhow!("Metal gate input allocation failed"))?,
            );
            self.output = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| anyhow::anyhow!("Metal gate output allocation failed"))?,
            );
            self.capacity = bytes;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl ActivationBuffers {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        bytes: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};
        if self.capacity < bytes {
            let options = MTLResourceOptions::StorageModeShared;
            self.first = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| anyhow::anyhow!("Metal activation buffer allocation failed"))?,
            );
            self.second = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| anyhow::anyhow!("Metal activation scratch allocation failed"))?,
            );
            self.capacity = bytes;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl TernaryBuffers {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        input_bytes: usize,
        packed_bytes: usize,
        scale_bytes: usize,
        output_bytes: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};

        let shared = MTLResourceOptions::StorageModeShared;
        if self.input_capacity < input_bytes {
            self.input = Some(
                device
                    .newBufferWithLength_options(input_bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal input buffer allocation failed"))?,
            );
            self.input_capacity = input_bytes;
        }
        if self.packed_capacity < packed_bytes {
            self.positive = Some(
                device
                    .newBufferWithLength_options(packed_bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal positive bitplane allocation failed"))?,
            );
            self.negative = Some(
                device
                    .newBufferWithLength_options(packed_bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal negative bitplane allocation failed"))?,
            );
            self.packed_capacity = packed_bytes;
        }
        if self.scale_capacity < scale_bytes {
            self.scales = Some(
                device
                    .newBufferWithLength_options(scale_bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal scale buffer allocation failed"))?,
            );
            self.scale_capacity = scale_bytes;
        }
        if self.output_capacity < output_bytes {
            self.output = Some(
                device
                    .newBufferWithLength_options(output_bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal output buffer allocation failed"))?,
            );
            self.output_capacity = output_bytes;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    /// Compiles the ternary pipeline once and creates its command queue.
    pub fn new() -> Result<Self> {
        use objc2_foundation::NSString;
        use objc2_metal::{MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
        let source = NSString::from_str(HYENA_METAL_SOURCE);
        let options = MTLCompileOptions::new();
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|error| anyhow::anyhow!("Metal shader compilation failed: {error}"))?;
        let name = NSString::from_str(TERNARY_LINEAR_KERNEL_NAME);
        let function = library
            .newFunctionWithName(&name)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary function is missing"))?;
        let ternary_pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| anyhow::anyhow!("Metal pipeline creation failed: {error}"))?;
        let fused_name = NSString::from_str(RMS_NORM_TERNARY_LINEAR_KERNEL_NAME);
        let fused_function = library
            .newFunctionWithName(&fused_name)
            .ok_or_else(|| anyhow::anyhow!("Metal fused ternary function is missing"))?;
        let fused_rms_norm_ternary_pipeline = device
            .newComputePipelineStateWithFunction_error(&fused_function)
            .map_err(|error| anyhow::anyhow!("Metal fused pipeline creation failed: {error}"))?;
        let bitreverse_name = NSString::from_str(FFT_BITREVERSE_KERNEL_NAME);
        let bitreverse_function = library
            .newFunctionWithName(&bitreverse_name)
            .ok_or_else(|| anyhow::anyhow!("Metal FFT bitreverse function is missing"))?;
        let fft_bitreverse_pipeline = device
            .newComputePipelineStateWithFunction_error(&bitreverse_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal FFT bitreverse pipeline creation failed: {error}")
            })?;
        let stage_name = NSString::from_str(FFT_STAGE_KERNEL_NAME);
        let stage_function = library
            .newFunctionWithName(&stage_name)
            .ok_or_else(|| anyhow::anyhow!("Metal FFT stage function is missing"))?;
        let fft_stage_pipeline = device
            .newComputePipelineStateWithFunction_error(&stage_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal FFT stage pipeline creation failed: {error}")
            })?;
        let multiply_name = NSString::from_str(FFT_COMPLEX_MULTIPLY_KERNEL_NAME);
        let multiply_function = library
            .newFunctionWithName(&multiply_name)
            .ok_or_else(|| anyhow::anyhow!("Metal FFT multiply function is missing"))?;
        let fft_multiply_pipeline = device
            .newComputePipelineStateWithFunction_error(&multiply_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal FFT multiply pipeline creation failed: {error}")
            })?;
        let extract_name = NSString::from_str(FFT_EXTRACT_CAUSAL_KERNEL_NAME);
        let extract_function = library
            .newFunctionWithName(&extract_name)
            .ok_or_else(|| anyhow::anyhow!("Metal FFT extract function is missing"))?;
        let fft_extract_pipeline = device
            .newComputePipelineStateWithFunction_error(&extract_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal FFT extract pipeline creation failed: {error}")
            })?;
        let implicit_name = NSString::from_str(IMPLICIT_FILTER_KERNEL_NAME);
        let implicit_function = library
            .newFunctionWithName(&implicit_name)
            .ok_or_else(|| anyhow::anyhow!("Metal implicit-filter function is missing"))?;
        let implicit_filter_pipeline = device
            .newComputePipelineStateWithFunction_error(&implicit_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal implicit-filter pipeline creation failed: {error}")
            })?;
        let gate_name = NSString::from_str(TANH_GATE_KERNEL_NAME);
        let gate_function = library
            .newFunctionWithName(&gate_name)
            .ok_or_else(|| anyhow::anyhow!("Metal tanh-gate function is missing"))?;
        let tanh_gate_pipeline = device
            .newComputePipelineStateWithFunction_error(&gate_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal tanh-gate pipeline creation failed: {error}")
            })?;
        let make_pipeline = |kernel_name: &str| -> Result<_> {
            let name = NSString::from_str(kernel_name);
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| anyhow::anyhow!("Metal function {kernel_name} is missing"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| {
                    anyhow::anyhow!("Metal pipeline {kernel_name} creation failed: {error}")
                })
        };
        let pack_strided_real_pipeline = make_pipeline(PACK_STRIDED_REAL_KERNEL_NAME)?;
        let apply_gate_pipeline = make_pipeline(APPLY_GATE_KERNEL_NAME)?;
        let residual_add_pipeline = make_pipeline(RESIDUAL_ADD_KERNEL_NAME)?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow::anyhow!("Metal command queue is unavailable"))?;
        Ok(Self {
            device,
            queue,
            ternary_pipeline,
            fused_rms_norm_ternary_pipeline,
            fft_bitreverse_pipeline,
            fft_stage_pipeline,
            fft_multiply_pipeline,
            fft_extract_pipeline,
            implicit_filter_pipeline,
            tanh_gate_pipeline,
            pack_strided_real_pipeline,
            apply_gate_pipeline,
            residual_add_pipeline,
            ternary_buffers: std::cell::RefCell::new(TernaryBuffers::default()),
            fft_buffers: std::cell::RefCell::new(FftBuffers::default()),
            filter_fft_buffers: std::cell::RefCell::new(FftBuffers::default()),
            hyena_output_buffer: std::cell::RefCell::new(OutputBuffer::default()),
            implicit_filter_parameters: std::cell::RefCell::new(ImplicitFilterParameters::default()),
            gate_buffers: std::cell::RefCell::new(GateBuffers::default()),
            activations: std::cell::RefCell::new(ActivationBuffers::default()),
        })
    }

    /// Reserves the two resident activation slots used to ping-pong residual
    /// state between Hyena blocks. Allocation is grow-only and checked.
    pub fn reserve_activations(&self, rows: usize, width: usize) -> Result<()> {
        if rows == 0 || width == 0 {
            bail!("Metal activation dimensions must be non-zero");
        }
        let bytes = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal activation size overflow"))?;
        self.activations.borrow_mut().ensure(&self.device, bytes)
    }

    /// Uploads the initial embedding stream once.  Subsequent resident blocks
    /// exchange only GPU buffers; callers receive the slot token explicitly.
    #[allow(unsafe_code)]
    pub fn upload_resident_activations(
        &self,
        values: &[f32],
        rows: usize,
        width: usize,
    ) -> Result<ResidentActivationSlot> {
        use objc2_metal::MTLBuffer;
        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Metal resident activation shape overflow"))?;
        if rows == 0 || width == 0 || values.len() != elements {
            bail!("Metal resident activation shape mismatch");
        }
        self.reserve_activations(rows, width)?;
        let activations = self.activations.borrow();
        let first = activations
            .first
            .as_ref()
            .expect("checked Metal activation buffer");
        // SAFETY: `reserve_activations` admitted exactly this many FP32 values
        // and no command can use the new input before this method returns.
        unsafe {
            first
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(values.as_ptr(), elements);
        }
        Ok(ResidentActivationSlot::First)
    }

    /// Reads a resident hidden stream only after the complete model forward.
    #[allow(unsafe_code)]
    pub fn download_resident_activations(
        &self,
        slot: ResidentActivationSlot,
        rows: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        use objc2_metal::MTLBuffer;
        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Metal resident activation shape overflow"))?;
        let activations = self.activations.borrow();
        if activations.capacity
            < elements
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| anyhow::anyhow!("Metal resident activation size overflow"))?
        {
            bail!("Metal resident activations were not initialized");
        }
        let buffer = match slot {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked Metal activation buffer");
        let mut result = vec![0.0; elements];
        // SAFETY: resident commands synchronously complete before this public
        // extraction point, and the selected shared buffer has checked size.
        unsafe {
            result
                .as_mut_ptr()
                .copy_from_nonoverlapping(buffer.contents().cast::<f32>().as_ptr(), elements);
        }
        Ok(result)
    }

    /// Second submission of a Hyena block: ternary output projection followed
    /// by residual addition.  Both its input and result stay in the resident
    /// ping-pong slots; only immutable packed weights are copied from host.
    #[allow(unsafe_code)]
    pub fn resident_output_projection(
        &self,
        residual_slot: ResidentActivationSlot,
        rows: usize,
        width: usize,
        positive: &[u64],
        negative: &[u64],
        scales: &[f32],
    ) -> Result<ResidentActivationSlot> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let shape = TernaryLinearShape::new(rows, width, width)?;
        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Metal resident output shape overflow"))?;
        if positive.len() != shape.packed_words()?
            || negative.len() != positive.len()
            || scales.len() != width
        {
            bail!("Metal resident output weight shape mismatch");
        }
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident output size overflow"))?;
        let packed_bytes = positive
            .len()
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident packed size overflow"))?;
        self.reserve_activations(rows, width)?;
        let mut ternary = self.ternary_buffers.borrow_mut();
        ternary.ensure(
            &self.device,
            bytes,
            packed_bytes,
            size_of_val(scales),
            bytes,
        )?;
        let activations = self.activations.borrow();
        let residual = match residual_slot {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked residual activation");
        let next = match residual_slot.other() {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked next activation");
        let positive_buffer = ternary.positive.as_ref().expect("checked positive weights");
        let negative_buffer = ternary.negative.as_ref().expect("checked negative weights");
        let scale_buffer = ternary.scales.as_ref().expect("checked scales");
        let projected = ternary.output.as_ref().expect("checked ternary output");
        // SAFETY: all shared buffers were grown to the checked dimensions.
        unsafe {
            positive_buffer
                .contents()
                .cast::<u64>()
                .as_ptr()
                .copy_from_nonoverlapping(positive.as_ptr(), positive.len());
            negative_buffer
                .contents()
                .cast::<u64>()
                .as_ptr()
                .copy_from_nonoverlapping(negative.as_ptr(), negative.len());
            scale_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(scales.as_ptr(), scales.len());
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_ternary(
            encoder.as_ref(),
            &self.ternary_pipeline,
            next,
            positive_buffer,
            negative_buffer,
            scale_buffer,
            projected,
            shape,
            false,
        )?;
        self.encode_residual_add(
            encoder.as_ref(),
            residual,
            projected,
            next,
            u32::try_from(elements)
                .map_err(|_| anyhow::anyhow!("Metal resident elements exceed u32"))?,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident output command failed: {error}");
        }
        Ok(residual_slot.other())
    }

    /// Starts the first block submission with fused RMSNorm/ternary input
    /// projection and the in-place Hyena gate.  The projection is retained in
    /// `gate_buffers.output` for the following FFT mixer; no activation leaves
    /// Metal at this boundary.
    #[allow(unsafe_code)]
    pub fn resident_input_projection(
        &self,
        slot: ResidentActivationSlot,
        rows: usize,
        width: usize,
        positive: &[u64],
        negative: &[u64],
        scales: &[f32],
    ) -> Result<()> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let out_width = width
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal resident input width overflow"))?;
        let shape = TernaryLinearShape::new(rows, width, out_width)?;
        if positive.len() != shape.packed_words()?
            || negative.len() != positive.len()
            || scales.len() != out_width
        {
            bail!("Metal resident input weight shape mismatch");
        }
        let input_bytes = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal resident input size overflow"))?;
        let output_bytes = rows
            .checked_mul(out_width)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal resident projection size overflow"))?;
        let packed_bytes = size_of_val(positive);
        self.reserve_activations(rows, width)?;
        self.ternary_buffers.borrow_mut().ensure(
            &self.device,
            input_bytes,
            packed_bytes,
            size_of_val(scales),
            output_bytes,
        )?;
        self.gate_buffers
            .borrow_mut()
            .ensure(&self.device, output_bytes)?;
        let activations = self.activations.borrow();
        let ternary = self.ternary_buffers.borrow();
        let gates = self.gate_buffers.borrow();
        let input = match slot {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked resident input");
        let positive_buffer = ternary.positive.as_ref().expect("checked positive weights");
        let negative_buffer = ternary.negative.as_ref().expect("checked negative weights");
        let scale_buffer = ternary.scales.as_ref().expect("checked scales");
        let projected = ternary.output.as_ref().expect("checked projection output");
        let gated = gates.output.as_ref().expect("checked gate output");
        // SAFETY: immutable packed weights fit the persistent shared buffers.
        unsafe {
            positive_buffer
                .contents()
                .cast::<u64>()
                .as_ptr()
                .copy_from_nonoverlapping(positive.as_ptr(), positive.len());
            negative_buffer
                .contents()
                .cast::<u64>()
                .as_ptr()
                .copy_from_nonoverlapping(negative.as_ptr(), negative.len());
            scale_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(scales.as_ptr(), scales.len());
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_ternary(
            encoder.as_ref(),
            &self.fused_rms_norm_ternary_pipeline,
            input,
            positive_buffer,
            negative_buffer,
            scale_buffer,
            projected,
            shape,
            true,
        )?;
        self.encode_tanh_gate(encoder.as_ref(), projected, gated, rows, width)?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident input command failed: {error}");
        }
        Ok(())
    }

    /// Completes the resident Hyena mixer after `resident_input_projection`.
    /// The gated `[B*T, 2D]` projection is packed directly into the signal
    /// FFT buffer, convolved with an on-device implicit filter, and written
    /// into the opposite activation slot. No `[B, T, D]` value is materialized
    /// on the host along this path.
    #[allow(unsafe_code)]
    pub fn resident_hyena_mixer(
        &self,
        slot: ResidentActivationSlot,
        batch: usize,
        time: usize,
        channels: usize,
        filter: &crate::hyena::ImplicitFilter,
    ) -> Result<()> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputePipelineState,
        };

        let shape = MetalDispatchShape::new(batch, time, channels)?;
        let plan = HyenaFftPlan::new(time)?;
        let transforms = batch
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal resident transform shape overflow"))?;
        let signal_elements = transforms
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal resident signal FFT shape overflow"))?;
        let filter_elements = channels
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal resident filter shape overflow"))?;
        let filter_fft_elements = channels
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal resident filter FFT shape overflow"))?;
        let signal_bytes = signal_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident signal size overflow"))?;
        let filter_bytes = filter_fft_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident filter size overflow"))?;
        let output_bytes = shape
            .elements()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident mixer output size overflow"))?;
        let (freq, phase, decay, order) = filter.parameter_slices(channels)?;
        let parameter_bytes = freq
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident filter parameter size overflow"))?;
        self.reserve_activations(batch * time, channels)?;
        self.fft_buffers
            .borrow_mut()
            .ensure(&self.device, signal_bytes)?;
        self.filter_fft_buffers
            .borrow_mut()
            .ensure(&self.device, filter_bytes)?;
        self.hyena_output_buffer
            .borrow_mut()
            .ensure(&self.device, output_bytes)?;
        self.implicit_filter_parameters
            .borrow_mut()
            .ensure(&self.device, parameter_bytes)?;

        let fft_len = u32::try_from(plan.fft_len)
            .map_err(|_| anyhow::anyhow!("Metal resident FFT length exceeds u32"))?;
        let transforms_u32 = u32::try_from(transforms)
            .map_err(|_| anyhow::anyhow!("Metal resident transform count exceeds u32"))?;
        let channels_u32 = u32::try_from(channels)
            .map_err(|_| anyhow::anyhow!("Metal resident channel count exceeds u32"))?;
        let time_u32 =
            u32::try_from(time).map_err(|_| anyhow::anyhow!("Metal resident time exceeds u32"))?;
        let elements_u32 = u32::try_from(shape.elements())
            .map_err(|_| anyhow::anyhow!("Metal resident element count exceeds u32"))?;
        let order_u32 = u32::try_from(order)
            .map_err(|_| anyhow::anyhow!("Metal resident filter order exceeds u32"))?;
        let filter_elements_u32 = u32::try_from(filter_elements)
            .map_err(|_| anyhow::anyhow!("Metal resident filter elements exceed u32"))?;

        let activations = self.activations.borrow();
        let gates = self.gate_buffers.borrow();
        let signal_buffers = self.fft_buffers.borrow();
        let filter_buffers = self.filter_fft_buffers.borrow();
        let mixer_output = self.hyena_output_buffer.borrow();
        let parameters = self.implicit_filter_parameters.borrow();
        let gated = gates
            .output
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Metal resident gate is not initialized"))?;
        let next = match slot.other() {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked next resident activation");
        let signal_first = signal_buffers
            .first
            .as_ref()
            .expect("checked resident signal buffer");
        let signal_second = signal_buffers
            .second
            .as_ref()
            .expect("checked resident signal scratch buffer");
        let filter_first = filter_buffers
            .first
            .as_ref()
            .expect("checked resident filter buffer");
        let filter_second = filter_buffers
            .second
            .as_ref()
            .expect("checked resident filter scratch buffer");
        let mixed = mixer_output
            .buffer
            .as_ref()
            .expect("checked resident mixer output");
        let freq_buffer = parameters
            .freq
            .as_ref()
            .expect("checked resident frequency buffer");
        let phase_buffer = parameters
            .phase
            .as_ref()
            .expect("checked resident phase buffer");
        let decay_buffer = parameters
            .decay
            .as_ref()
            .expect("checked resident decay buffer");
        // SAFETY: the capacity checks above cover every copied parameter and
        // complex padding lane. These buffers stay borrowed through completion.
        unsafe {
            signal_first
                .contents()
                .cast::<Complex32>()
                .as_ptr()
                .write_bytes(0, signal_elements);
            filter_first
                .contents()
                .cast::<Complex32>()
                .as_ptr()
                .write_bytes(0, filter_fft_elements);
            freq_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(freq.as_ptr(), freq.len());
            phase_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(phase.as_ptr(), phase.len());
            decay_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(decay.as_ptr(), decay.len());
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_pack_strided_real(
            encoder.as_ref(),
            gated,
            signal_first,
            time_u32,
            channels_u32,
            channels_u32
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("Metal resident stride exceeds u32"))?,
            0,
            fft_len,
            elements_u32,
        )?;
        self.encode_implicit_filter(
            encoder.as_ref(),
            freq_buffer,
            phase_buffer,
            decay_buffer,
            filter_first,
            time_u32,
            order_u32,
            fft_len,
            filter_elements_u32,
        )?;
        let dispatch_two = |pipeline: &objc2::runtime::ProtocolObject<
            dyn MTLComputePipelineState,
        >,
                            input,
                            output,
                            total,
                            scalars: &[u32]| {
            self.encode_fft_two_buffer(encoder.as_ref(), pipeline, input, output, total, scalars)
        };
        let run_fft = |first, second, transform_count, total, inverse| -> Result<bool> {
            dispatch_two(
                &self.fft_bitreverse_pipeline,
                first,
                second,
                total,
                &[fft_len, transform_count],
            )?;
            let mut source_is_first = false;
            for stage in 1..=plan.stages {
                let (source, destination) = if source_is_first {
                    (first, second)
                } else {
                    (second, first)
                };
                dispatch_two(
                    &self.fft_stage_pipeline,
                    source,
                    destination,
                    total,
                    &[fft_len, transform_count, stage, u32::from(inverse)],
                )?;
                source_is_first = !source_is_first;
            }
            Ok(source_is_first)
        };
        let signal_source_is_first = run_fft(
            signal_first,
            signal_second,
            transforms_u32,
            signal_elements,
            false,
        )?;
        let filter_source_is_first = run_fft(
            filter_first,
            filter_second,
            channels_u32,
            filter_fft_elements,
            false,
        )?;
        let signal_spectrum = if signal_source_is_first {
            signal_first
        } else {
            signal_second
        };
        let filter_spectrum = if filter_source_is_first {
            filter_first
        } else {
            filter_second
        };
        let product_is_first = !signal_source_is_first;
        let product = if product_is_first {
            signal_first
        } else {
            signal_second
        };
        self.encode_fft_multiply(
            encoder.as_ref(),
            signal_spectrum,
            filter_spectrum,
            product,
            fft_len,
            channels_u32,
            transforms_u32,
            signal_elements,
        )?;
        let inverse_bitreversed_is_first = !product_is_first;
        let inverse_bitreversed = if inverse_bitreversed_is_first {
            signal_first
        } else {
            signal_second
        };
        dispatch_two(
            &self.fft_bitreverse_pipeline,
            product,
            inverse_bitreversed,
            signal_elements,
            &[fft_len, transforms_u32],
        )?;
        let mut inverse_source_is_first = inverse_bitreversed_is_first;
        for stage in 1..=plan.stages {
            let (source, destination) = if inverse_source_is_first {
                (signal_first, signal_second)
            } else {
                (signal_second, signal_first)
            };
            dispatch_two(
                &self.fft_stage_pipeline,
                source,
                destination,
                signal_elements,
                &[fft_len, transforms_u32, stage, 1],
            )?;
            inverse_source_is_first = !inverse_source_is_first;
        }
        let inverse = if inverse_source_is_first {
            signal_first
        } else {
            signal_second
        };
        self.encode_causal_extract(
            encoder.as_ref(),
            inverse,
            mixed,
            time_u32,
            channels_u32,
            fft_len,
            elements_u32,
        )?;
        self.encode_apply_gate(
            encoder.as_ref(),
            mixed,
            gated,
            next,
            channels_u32,
            channels_u32
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("Metal resident stride exceeds u32"))?,
            channels_u32,
            elements_u32,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident Hyena mixer failed: {error}");
        }
        Ok(())
    }

    /// Runs a projection without recompiling MSL or reallocating buffers when
    /// the existing capacities already fit the requested shape.
    #[allow(unsafe_code)]
    fn dispatch_ternary(
        &self,
        input: &[f32],
        positive: &[u64],
        negative: &[u64],
        scales: &[f32],
        shape: TernaryLinearShape,
        pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        fused_rms_norm: bool,
    ) -> Result<Vec<f32>> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let input_len = shape
            .rows
            .checked_mul(shape.in_features)
            .ok_or_else(|| anyhow::anyhow!("ternary input shape overflow"))?;
        let output_len = shape
            .rows
            .checked_mul(shape.out_features)
            .ok_or_else(|| anyhow::anyhow!("ternary output shape overflow"))?;
        let packed_words = shape.packed_words()?;
        if input.len() != input_len
            || positive.len() != packed_words
            || negative.len() != packed_words
            || scales.len() != shape.out_features
        {
            bail!("Metal ternary projection shape mismatch");
        }
        let input_bytes = input_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal input buffer byte size overflow"))?;
        let packed_bytes = packed_words
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("Metal ternary buffer byte size overflow"))?;
        let scale_bytes = scales
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal scale buffer byte size overflow"))?;
        let output_bytes = output_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal output buffer byte size overflow"))?;
        let mut buffers = self.ternary_buffers.borrow_mut();
        buffers.ensure(
            &self.device,
            input_bytes,
            packed_bytes,
            scale_bytes,
            output_bytes,
        )?;
        let input_buffer = buffers.input.as_ref().expect("checked Metal input buffer");
        let positive_buffer = buffers
            .positive
            .as_ref()
            .expect("checked Metal positive buffer");
        let negative_buffer = buffers
            .negative
            .as_ref()
            .expect("checked Metal negative buffer");
        let scale_buffer = buffers.scales.as_ref().expect("checked Metal scale buffer");
        let output_buffer = buffers
            .output
            .as_ref()
            .expect("checked Metal output buffer");
        // SAFETY: Capacities were checked above; host writes complete before
        // submission and the mutable scratch borrow lasts through completion.
        unsafe {
            input_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(input.as_ptr(), input_len);
            positive_buffer
                .contents()
                .cast::<u64>()
                .as_ptr()
                .copy_from_nonoverlapping(positive.as_ptr(), packed_words);
            negative_buffer
                .contents()
                .cast::<u64>()
                .as_ptr()
                .copy_from_nonoverlapping(negative.as_ptr(), packed_words);
            scale_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(scales.as_ptr(), scales.len());
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_ternary(
            encoder.as_ref(),
            pipeline,
            input_buffer,
            positive_buffer,
            negative_buffer,
            scale_buffer,
            output_buffer,
            shape,
            fused_rms_norm,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal ternary command failed: {error}");
        }
        let mut output = vec![0.0; output_len];
        // SAFETY: Command completion makes the output shared buffer readable;
        // its initialized elements exactly match the destination allocation.
        unsafe {
            output.as_mut_ptr().copy_from_nonoverlapping(
                output_buffer.contents().cast::<f32>().as_ptr(),
                output_len,
            );
        }
        Ok(output)
    }

    #[allow(unsafe_code)]
    fn encode_ternary(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        positive: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        negative: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        scales: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        shape: TernaryLinearShape,
        fused_rms_norm: bool,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        let rows =
            u32::try_from(shape.rows).map_err(|_| anyhow::anyhow!("Metal rows exceed u32"))?;
        let input_width = u32::try_from(shape.in_features)
            .map_err(|_| anyhow::anyhow!("Metal input width exceeds u32"))?;
        let output_width = u32::try_from(shape.out_features)
            .map_err(|_| anyhow::anyhow!("Metal output width exceeds u32"))?;
        let output_elements = shape
            .rows
            .checked_mul(shape.out_features)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary output shape overflow"))?;
        encoder.setComputePipelineState(pipeline);
        // SAFETY: slots and scalar offsets exactly match both ternary MSL declarations.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(positive), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(negative), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(scales), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 4);
            for (slot, scalar) in [rows, input_width, output_width].iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    slot + 5,
                );
            }
        }
        let width = if fused_rms_norm {
            pipeline.maxTotalThreadsPerThreadgroup().min(256)
        } else {
            pipeline
                .maxTotalThreadsPerThreadgroup()
                .min(output_elements)
        };
        if width == 0 {
            bail!("Metal ternary pipeline reported zero threads per threadgroup");
        }
        if fused_rms_norm {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: shape.rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: output_elements,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok(())
    }

    /// Runs the unfused packed ternary projection through cached Metal state.
    #[allow(unsafe_code)]
    pub fn ternary_linear_forward(
        &self,
        input: &[f32],
        positive: &[u64],
        negative: &[u64],
        scales: &[f32],
        shape: TernaryLinearShape,
    ) -> Result<Vec<f32>> {
        self.dispatch_ternary(
            input,
            positive,
            negative,
            scales,
            shape,
            &self.ternary_pipeline,
            false,
        )
    }

    /// Fuses RMSNorm with ternary projection, keeping normalized activations
    /// virtual and reusing the runtime's packed-weight buffers.
    #[allow(unsafe_code)]
    pub fn rms_norm_ternary_linear_forward(
        &self,
        input: &[f32],
        positive: &[u64],
        negative: &[u64],
        scales: &[f32],
        shape: TernaryLinearShape,
    ) -> Result<Vec<f32>> {
        self.dispatch_ternary(
            input,
            positive,
            negative,
            scales,
            shape,
            &self.fused_rms_norm_ternary_pipeline,
            true,
        )
    }

    /// GPU reference for the in-place gate layout used by a Hyena input
    /// projection: `[rows, 2 * channels] -> [rows, 2 * channels]`.
    #[allow(unsafe_code)]
    pub fn tanh_gate_forward(
        &self,
        input: &[f32],
        rows: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let elements = rows
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("Metal gate shape overflow"))?;
        if rows == 0 || channels == 0 || input.len() != elements {
            bail!("Metal gate shape mismatch");
        }
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal gate size overflow"))?;
        let mut buffers = self.gate_buffers.borrow_mut();
        buffers.ensure(&self.device, bytes)?;
        let source = buffers
            .input
            .as_ref()
            .expect("checked Metal gate input buffer");
        let output = buffers
            .output
            .as_ref()
            .expect("checked Metal gate output buffer");
        // SAFETY: the shared input buffer has the checked capacity and remains borrowed until completion.
        unsafe {
            source
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(input.as_ptr(), elements);
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_tanh_gate(encoder.as_ref(), source, output, rows, channels)?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal gate command failed: {error}");
        }
        let mut result = vec![0.0; elements];
        // SAFETY: completion makes every output element readable from the shared buffer.
        unsafe {
            result
                .as_mut_ptr()
                .copy_from_nonoverlapping(output.contents().cast::<f32>().as_ptr(), elements);
        }
        Ok(result)
    }

    #[allow(unsafe_code)]
    fn encode_tanh_gate(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        rows: usize,
        channels: usize,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        let elements = rows
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("Metal gate shape overflow"))?;
        let elements_u32 = u32::try_from(elements)
            .map_err(|_| anyhow::anyhow!("Metal gate elements exceed u32"))?;
        let channels_u32 = u32::try_from(channels)
            .map_err(|_| anyhow::anyhow!("Metal gate channels exceed u32"))?;
        self.tanh_gate_pipeline.maxTotalThreadsPerThreadgroup();
        encoder.setComputePipelineState(&self.tanh_gate_pipeline);
        // SAFETY: slots 0..3 exactly match ullis_tanh_gate_in_place.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::from(&elements_u32).cast::<c_void>(),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&channels_u32).cast::<c_void>(),
                size_of::<u32>(),
                3,
            );
        }
        let width = self
            .tanh_gate_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(elements);
        if width == 0 {
            bail!("Metal gate pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Encodes the three layout-only Hyena operations used by the resident
    /// path.  Keeping them here makes their buffer-slot contracts auditable
    /// and lets one command buffer chain them without a host readback.
    #[allow(unsafe_code)]
    fn encode_pack_strided_real(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        time: u32,
        channels: u32,
        stride: u32,
        offset: u32,
        fft_len: u32,
        elements: u32,
    ) -> Result<()> {
        self.encode_elementwise_buffers(
            encoder,
            &self.pack_strided_real_pipeline,
            &[input, output],
            &[time, channels, stride, offset, fft_len, elements],
            elements as usize,
        )
    }

    #[allow(unsafe_code)]
    fn encode_apply_gate(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        mixed: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        projection: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        channels: u32,
        stride: u32,
        gate_offset: u32,
        elements: u32,
    ) -> Result<()> {
        self.encode_elementwise_buffers(
            encoder,
            &self.apply_gate_pipeline,
            &[mixed, projection, output],
            &[channels, stride, gate_offset, elements],
            elements as usize,
        )
    }

    #[allow(unsafe_code)]
    fn encode_residual_add(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        residual: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        update: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        elements: u32,
    ) -> Result<()> {
        self.encode_elementwise_buffers(
            encoder,
            &self.residual_add_pipeline,
            &[residual, update, output],
            &[elements],
            elements as usize,
        )
    }

    #[allow(unsafe_code)]
    fn encode_elementwise_buffers(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        buffers: &[&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>],
        scalars: &[u32],
        elements: usize,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        if elements == 0 {
            bail!("Metal elementwise dispatch cannot be empty");
        }
        encoder.setComputePipelineState(pipeline);
        // SAFETY: callers pair buffer/scalar lists with the exact MSL ABI.
        unsafe {
            for (slot, buffer) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(*buffer), 0, slot);
            }
            for (offset, scalar) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    buffers.len() + offset,
                );
            }
        }
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(elements);
        if width == 0 {
            bail!("Metal elementwise pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Executes the layout-only tail of a Hyena block in one submission.  It
    /// is intentionally public as a small equivalence boundary while the
    /// model-level resident API is wired: only the final residual stream is
    /// read back, never the packed or gated intermediates.
    #[allow(unsafe_code)]
    pub fn resident_gate_residual_reference(
        &self,
        residual: &[f32],
        mixed: &[f32],
        projection: &[f32],
        rows: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal resident element shape overflow"))?;
        let projected = elements
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal resident projection shape overflow"))?;
        if rows == 0
            || channels == 0
            || residual.len() != elements
            || mixed.len() != elements
            || projection.len() != projected
        {
            bail!("Metal resident gate/residual shape mismatch");
        }
        let activation_bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident activation size overflow"))?;
        let projection_bytes = projected
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident projection size overflow"))?;
        self.reserve_activations(rows, channels)?;
        self.gate_buffers
            .borrow_mut()
            .ensure(&self.device, projection_bytes)?;
        self.hyena_output_buffer
            .borrow_mut()
            .ensure(&self.device, activation_bytes)?;
        let plan = HyenaFftPlan::new(rows)?;
        let transform_elements = elements
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal resident FFT shape overflow"))?;
        let transform_bytes = transform_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident FFT size overflow"))?;
        self.fft_buffers
            .borrow_mut()
            .ensure(&self.device, transform_bytes)?;
        let activations = self.activations.borrow();
        let gates = self.gate_buffers.borrow();
        let output = self.hyena_output_buffer.borrow();
        let fft = self.fft_buffers.borrow();
        let first = activations
            .first
            .as_ref()
            .expect("checked Metal activation buffer");
        let second = activations
            .second
            .as_ref()
            .expect("checked Metal activation scratch");
        let gate = gates.output.as_ref().expect("checked Metal gate buffer");
        let mixed_buffer = output.buffer.as_ref().expect("checked Metal Hyena output");
        let fft_first = fft.first.as_ref().expect("checked Metal FFT source buffer");
        // SAFETY: each persistent shared allocation was checked above and no
        // GPU command observes it until these complete host writes finish.
        unsafe {
            first
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(residual.as_ptr(), elements);
            mixed_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(mixed.as_ptr(), elements);
            gate.contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(projection.as_ptr(), projected);
            fft_first
                .contents()
                .cast::<Complex32>()
                .as_ptr()
                .write_bytes(0, transform_elements);
        }
        let channels_u32 =
            u32::try_from(channels).map_err(|_| anyhow::anyhow!("Metal channels exceed u32"))?;
        let projection_stride = channels_u32
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal projection stride overflow"))?;
        let elements_u32 =
            u32::try_from(elements).map_err(|_| anyhow::anyhow!("Metal elements exceed u32"))?;
        let rows_u32 = u32::try_from(rows).map_err(|_| anyhow::anyhow!("Metal rows exceed u32"))?;
        let fft_len_u32 = u32::try_from(plan.fft_len)
            .map_err(|_| anyhow::anyhow!("Metal FFT length exceeds u32"))?;
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_pack_strided_real(
            encoder.as_ref(),
            gate,
            fft_first,
            rows_u32,
            channels_u32,
            projection_stride,
            0,
            fft_len_u32,
            elements_u32,
        )?;
        self.encode_apply_gate(
            encoder.as_ref(),
            mixed_buffer,
            gate,
            second,
            channels_u32,
            projection_stride,
            channels_u32,
            elements_u32,
        )?;
        self.encode_residual_add(encoder.as_ref(), first, second, second, elements_u32)?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident gate/residual command failed: {error}");
        }
        let mut result = vec![0.0; elements];
        // SAFETY: completion makes the shared final activation readable.
        unsafe {
            result
                .as_mut_ptr()
                .copy_from_nonoverlapping(second.contents().cast::<f32>().as_ptr(), elements);
        }
        Ok(result)
    }

    #[allow(unsafe_code)]
    fn encode_fft_two_buffer(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        total: usize,
        scalars: &[u32],
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        encoder.setComputePipelineState(pipeline);
        // SAFETY: slots 0/1 and scalar slots from 2 match the FFT MSL kernels.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
            for (offset, scalar) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    offset + 2,
                );
            }
        }
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(total);
        if width == 0 {
            bail!("Metal FFT pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: total,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[allow(unsafe_code)]
    fn encode_implicit_filter(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        freq: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        phase: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        decay: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        time: u32,
        order: u32,
        fft_len: u32,
        elements: u32,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        encoder.setComputePipelineState(&self.implicit_filter_pipeline);
        // SAFETY: slots 0..7 exactly match ullis_generate_implicit_filter.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(freq), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(phase), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(decay), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            for (slot, scalar) in [time, order, fft_len, elements].iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    slot + 4,
                );
            }
        }
        let width = self
            .implicit_filter_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(elements as usize);
        if width == 0 {
            bail!("Metal implicit-filter pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: elements as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[allow(unsafe_code)]
    fn encode_fft_multiply(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        signal: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        filter: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        fft_len: u32,
        channels: u32,
        transforms: u32,
        total: usize,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        encoder.setComputePipelineState(&self.fft_multiply_pipeline);
        // SAFETY: slots 0..5 exactly match ullis_fft_complex_multiply.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(signal), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(filter), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
            for (slot, scalar) in [fft_len, channels, transforms].iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    slot + 3,
                );
            }
        }
        let width = self
            .fft_multiply_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(total);
        if width == 0 {
            bail!("Metal FFT multiply pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: total,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[allow(unsafe_code)]
    fn encode_causal_extract(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        time: u32,
        channels: u32,
        fft_len: u32,
        elements: u32,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        encoder.setComputePipelineState(&self.fft_extract_pipeline);
        // SAFETY: slots 0..5 exactly match ullis_fft_extract_causal.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
            for (slot, scalar) in [time, channels, fft_len, elements].iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    slot + 2,
                );
            }
        }
        let width = self
            .fft_extract_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(elements as usize);
        if width == 0 {
            bail!("Metal FFT extract pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: elements as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Runs batched complex radix-2 FFTs through cached ping-pong buffers.
    /// Values are interleaved as `(real, imaginary)` and each transform must
    /// have the same power-of-two width.
    #[allow(unsafe_code)]
    pub fn fft_reference(
        &self,
        values: &[(f32, f32)],
        transforms: usize,
        inverse: bool,
    ) -> Result<Vec<(f32, f32)>> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
        };

        if transforms == 0 || values.is_empty() || !values.len().is_multiple_of(transforms) {
            bail!("Metal FFT shape mismatch");
        }
        let fft_len = values.len() / transforms;
        if !fft_len.is_power_of_two() || u32::try_from(values.len()).is_err() {
            bail!("Metal FFT length must be a non-zero u32 power of two");
        }
        let fft_len_u32 =
            u32::try_from(fft_len).map_err(|_| anyhow::anyhow!("Metal FFT length exceeds u32"))?;
        let transforms_u32 = u32::try_from(transforms)
            .map_err(|_| anyhow::anyhow!("Metal FFT transform count exceeds u32"))?;
        let bytes = values
            .len()
            .checked_mul(size_of::<(f32, f32)>())
            .ok_or_else(|| anyhow::anyhow!("Metal FFT byte size overflow"))?;
        let mut buffers = self.fft_buffers.borrow_mut();
        buffers.ensure(&self.device, bytes)?;
        let first = buffers
            .first
            .as_ref()
            .expect("checked Metal FFT source buffer");
        let second = buffers
            .second
            .as_ref()
            .expect("checked Metal FFT scratch buffer");
        // SAFETY: The shared source capacity was checked from `values`, and
        // the borrow keeps both ping-pong buffers stable until GPU completion.
        unsafe {
            first
                .contents()
                .cast::<(f32, f32)>()
                .as_ptr()
                .copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        let dispatch = |encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
                        pipeline: &objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>,
                        input: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                        output: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                        scalars: &[u32]|
         -> Result<()> {
            encoder.setComputePipelineState(pipeline);
            // SAFETY: Slots 0/1 and scalar slots beginning at 2 match both
            // FFT MSL signatures; `setBytes` copies each scalar immediately.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
                for (offset, scalar) in scalars.iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        NonNull::from(scalar).cast::<c_void>(),
                        size_of::<u32>(),
                        offset + 2,
                    );
                }
            }
            let width = pipeline.maxTotalThreadsPerThreadgroup().min(values.len());
            if width == 0 {
                bail!("Metal FFT pipeline reported zero threads per threadgroup");
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: values.len(),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        };
        dispatch(
            encoder.as_ref(),
            &self.fft_bitreverse_pipeline,
            first.as_ref(),
            second.as_ref(),
            &[fft_len_u32, transforms_u32],
        )?;
        let mut source_is_first = false;
        for stage in 1..=fft_len.ilog2() {
            let (input, output) = if source_is_first {
                (first.as_ref(), second.as_ref())
            } else {
                (second.as_ref(), first.as_ref())
            };
            dispatch(
                encoder.as_ref(),
                &self.fft_stage_pipeline,
                input,
                output,
                &[fft_len_u32, transforms_u32, stage, u32::from(inverse)],
            )?;
            source_is_first = !source_is_first;
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal FFT command failed: {error}");
        }
        let result = if source_is_first { first } else { second };
        let mut output = vec![(0.0, 0.0); values.len()];
        // SAFETY: Command completion makes the selected result buffer readable
        // and it has exactly the requested complex element capacity.
        unsafe {
            output.as_mut_ptr().copy_from_nonoverlapping(
                result.contents().cast::<(f32, f32)>().as_ptr(),
                values.len(),
            );
        }
        if inverse {
            let scale = fft_len as f32;
            for value in &mut output {
                value.0 /= scale;
                value.1 /= scale;
            }
        }
        Ok(output)
    }

    /// Generates an implicit filter on Metal into the cached FFT filter buffer.
    /// The returned Vec exists only for numerical verification; the following
    /// integration step will consume that buffer directly in the FFT chain.
    #[allow(unsafe_code)]
    pub fn implicit_filter_forward(
        &self,
        filter: &crate::hyena::ImplicitFilter,
        channels: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
        };

        let (freq, phase, decay, order) = filter.parameter_slices(channels)?;
        let plan = HyenaFftPlan::new(time)?;
        let elements = channels
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal implicit-filter shape overflow"))?;
        let fft_elements = channels
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal implicit-filter FFT shape overflow"))?;
        let parameter_bytes = freq
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal implicit-filter parameter size overflow"))?;
        let fft_bytes = fft_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal implicit-filter output size overflow"))?;
        let time_u32 = u32::try_from(time)
            .map_err(|_| anyhow::anyhow!("Metal implicit-filter time exceeds u32"))?;
        let order_u32 = u32::try_from(order)
            .map_err(|_| anyhow::anyhow!("Metal implicit-filter order exceeds u32"))?;
        let fft_len_u32 = u32::try_from(plan.fft_len)
            .map_err(|_| anyhow::anyhow!("Metal implicit-filter FFT length exceeds u32"))?;
        let elements_u32 = u32::try_from(elements)
            .map_err(|_| anyhow::anyhow!("Metal implicit-filter element count exceeds u32"))?;
        let mut parameters = self.implicit_filter_parameters.borrow_mut();
        let mut buffers = self.filter_fft_buffers.borrow_mut();
        parameters.ensure(&self.device, parameter_bytes)?;
        buffers.ensure(&self.device, fft_bytes)?;
        let freq_buffer = parameters
            .freq
            .as_ref()
            .expect("checked Metal implicit frequency buffer");
        let phase_buffer = parameters
            .phase
            .as_ref()
            .expect("checked Metal implicit phase buffer");
        let decay_buffer = parameters
            .decay
            .as_ref()
            .expect("checked Metal implicit decay buffer");
        let output = buffers
            .first
            .as_ref()
            .expect("checked Metal implicit output buffer");
        // SAFETY: all shared buffers have checked capacities and stay borrowed until completion.
        unsafe {
            freq_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(freq.as_ptr(), freq.len());
            phase_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(phase.as_ptr(), phase.len());
            decay_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(decay.as_ptr(), decay.len());
            output
                .contents()
                .cast::<Complex32>()
                .as_ptr()
                .write_bytes(0, fft_elements);
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        encoder.setComputePipelineState(&self.implicit_filter_pipeline);
        // SAFETY: slots 0..7 exactly match ullis_generate_implicit_filter.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(freq_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(phase_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(decay_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            for (slot, scalar) in [time_u32, order_u32, fft_len_u32, elements_u32]
                .iter()
                .enumerate()
            {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    slot + 4,
                );
            }
        }
        let width = self
            .implicit_filter_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(elements);
        if width == 0 {
            bail!("Metal implicit-filter pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal implicit-filter command failed: {error}");
        }
        let mut result = vec![0.0; elements];
        // SAFETY: command completion makes generated complex values readable; only real lanes are returned.
        unsafe {
            let generated = output.contents().cast::<Complex32>().as_ptr();
            for channel in 0..channels {
                for position in 0..time {
                    result[channel * time + position] =
                        (*generated.add(channel * plan.fft_len + position)).real;
                }
            }
        }
        Ok(result)
    }

    /// Executes dense causal Hyena convolution in one Metal command buffer.
    ///
    /// `input` has `[batch, time, channels]` layout and `filter` has
    /// `[channels, time]` layout. This is intentionally a dense-filter
    /// reference boundary: the production implicit/strided mixer remains on
    /// the CPU until its filter generator can write directly into GPU storage.
    /// No FFT spectrum crosses the CPU/GPU boundary.
    #[allow(unsafe_code)]
    pub fn causal_long_conv_forward(
        &self,
        input: &[f32],
        filter: &[f32],
        batch: usize,
        time: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        self.causal_long_conv_with_filter(
            input,
            HyenaFilterSource::Dense(filter),
            batch,
            time,
            channels,
            channels,
            0,
        )
    }

    /// Runs the complete implicit-filter Hyena mixer in one command buffer.
    /// It writes the compact filter directly to the FFT buffer before the
    /// filter transform, so no `[channels, time]` host filter exists.
    #[allow(unsafe_code)]
    pub fn causal_long_conv_implicit_forward(
        &self,
        input: &[f32],
        filter: &crate::hyena::ImplicitFilter,
        batch: usize,
        time: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        self.causal_long_conv_with_filter(
            input,
            HyenaFilterSource::Implicit(filter),
            batch,
            time,
            channels,
            channels,
            0,
        )
    }

    /// Strided variant for the signal half of a `[B,T,2D]` projection.
    #[allow(unsafe_code)]
    pub fn causal_long_conv_implicit_strided_forward(
        &self,
        input: &[f32],
        filter: &crate::hyena::ImplicitFilter,
        batch: usize,
        time: usize,
        channels: usize,
        input_stride: usize,
        input_offset: usize,
    ) -> Result<Vec<f32>> {
        self.causal_long_conv_with_filter(
            input,
            HyenaFilterSource::Implicit(filter),
            batch,
            time,
            channels,
            input_stride,
            input_offset,
        )
    }

    #[allow(unsafe_code)]
    fn causal_long_conv_with_filter(
        &self,
        input: &[f32],
        filter_source: HyenaFilterSource<'_>,
        batch: usize,
        time: usize,
        channels: usize,
        input_stride: usize,
        input_offset: usize,
    ) -> Result<Vec<f32>> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputePipelineState,
        };

        let shape = MetalDispatchShape::new(batch, time, channels)?;
        let plan = HyenaFftPlan::new(time)?;
        let transforms = batch
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena transform shape overflow"))?;
        let filter_elements = channels
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena filter shape overflow"))?;
        let input_rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena input shape overflow"))?;
        let minimum_input = input_rows
            .checked_sub(1)
            .and_then(|row| row.checked_mul(input_stride))
            .and_then(|start| start.checked_add(input_offset))
            .and_then(|start| start.checked_add(channels))
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena input shape overflow"))?;
        if input_stride < channels || input.len() < minimum_input {
            bail!("Metal Hyena causal convolution shape mismatch");
        }
        let implicit_parameters = match &filter_source {
            HyenaFilterSource::Dense(filter) => {
                if filter.len() != filter_elements {
                    bail!("Metal Hyena causal convolution shape mismatch");
                }
                None
            }
            HyenaFilterSource::Implicit(filter) => Some(filter.parameter_slices(channels)?),
        };
        let signal_elements = transforms
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena signal FFT shape overflow"))?;
        let filter_fft_elements = channels
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena filter FFT shape overflow"))?;
        let signal_bytes = signal_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena signal buffer size overflow"))?;
        let filter_bytes = filter_fft_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena filter buffer size overflow"))?;
        let output_bytes = shape
            .elements()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena output buffer size overflow"))?;
        let fft_len = u32::try_from(plan.fft_len)
            .map_err(|_| anyhow::anyhow!("Metal Hyena FFT length exceeds u32"))?;
        let transforms_u32 = u32::try_from(transforms)
            .map_err(|_| anyhow::anyhow!("Metal Hyena transform count exceeds u32"))?;
        let channels_u32 = u32::try_from(channels)
            .map_err(|_| anyhow::anyhow!("Metal Hyena channel count exceeds u32"))?;
        let time_u32 =
            u32::try_from(time).map_err(|_| anyhow::anyhow!("Metal Hyena time exceeds u32"))?;
        let elements_u32 = u32::try_from(shape.elements())
            .map_err(|_| anyhow::anyhow!("Metal Hyena element count exceeds u32"))?;

        let mut signal_buffers = self.fft_buffers.borrow_mut();
        let mut filter_buffers = self.filter_fft_buffers.borrow_mut();
        let mut output_buffer = self.hyena_output_buffer.borrow_mut();
        let mut parameter_buffers = self.implicit_filter_parameters.borrow_mut();
        signal_buffers.ensure(&self.device, signal_bytes)?;
        filter_buffers.ensure(&self.device, filter_bytes)?;
        output_buffer.ensure(&self.device, output_bytes)?;
        if let Some((freq, _, _, _)) = implicit_parameters {
            parameter_buffers.ensure(
                &self.device,
                freq.len().checked_mul(size_of::<f32>()).ok_or_else(|| {
                    anyhow::anyhow!("Metal implicit-filter parameter size overflow")
                })?,
            )?;
        }
        let signal_first = signal_buffers
            .first
            .as_ref()
            .expect("checked Metal Hyena signal buffer");
        let signal_second = signal_buffers
            .second
            .as_ref()
            .expect("checked Metal Hyena signal scratch buffer");
        let filter_first = filter_buffers
            .first
            .as_ref()
            .expect("checked Metal Hyena filter buffer");
        let filter_second = filter_buffers
            .second
            .as_ref()
            .expect("checked Metal Hyena filter scratch buffer");
        let output = output_buffer
            .buffer
            .as_ref()
            .expect("checked Metal Hyena output buffer");
        // SAFETY: each persistent shared buffer was grown to the exact checked
        // count. Zeroing its complex elements provides FFT padding without a
        // host-sized staging Vec; every later indexed write is bounded by the
        // validated tensor dimensions, and the borrows keep buffers exclusive
        // until GPU completion.
        unsafe {
            let signal_ptr = signal_first.contents().cast::<Complex32>().as_ptr();
            signal_ptr.write_bytes(0, signal_elements);
            for sequence in 0..batch {
                for position in 0..time {
                    let source_offset = (sequence * time + position) * input_stride + input_offset;
                    for channel in 0..channels {
                        let destination = (sequence * channels + channel) * plan.fft_len + position;
                        (*signal_ptr.add(destination)).real = input[source_offset + channel];
                    }
                }
            }
            let filter_ptr = filter_first.contents().cast::<Complex32>().as_ptr();
            filter_ptr.write_bytes(0, filter_fft_elements);
            if let HyenaFilterSource::Dense(filter) = &filter_source {
                for channel in 0..channels {
                    for position in 0..time {
                        (*filter_ptr.add(channel * plan.fft_len + position)).real =
                            filter[channel * time + position];
                    }
                }
            }
            if let Some((freq, phase, decay, _)) = implicit_parameters {
                parameter_buffers
                    .freq
                    .as_ref()
                    .expect("checked Metal implicit frequency buffer")
                    .contents()
                    .cast::<f32>()
                    .as_ptr()
                    .copy_from_nonoverlapping(freq.as_ptr(), freq.len());
                parameter_buffers
                    .phase
                    .as_ref()
                    .expect("checked Metal implicit phase buffer")
                    .contents()
                    .cast::<f32>()
                    .as_ptr()
                    .copy_from_nonoverlapping(phase.as_ptr(), phase.len());
                parameter_buffers
                    .decay
                    .as_ref()
                    .expect("checked Metal implicit decay buffer")
                    .contents()
                    .cast::<f32>()
                    .as_ptr()
                    .copy_from_nonoverlapping(decay.as_ptr(), decay.len());
            }
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        if let Some((_, _, _, order)) = implicit_parameters {
            let order_u32 = u32::try_from(order)
                .map_err(|_| anyhow::anyhow!("Metal implicit-filter order exceeds u32"))?;
            let filter_elements_u32 = u32::try_from(filter_elements)
                .map_err(|_| anyhow::anyhow!("Metal implicit-filter element count exceeds u32"))?;
            let freq = parameter_buffers
                .freq
                .as_ref()
                .expect("checked Metal implicit frequency buffer");
            let phase = parameter_buffers
                .phase
                .as_ref()
                .expect("checked Metal implicit phase buffer");
            let decay = parameter_buffers
                .decay
                .as_ref()
                .expect("checked Metal implicit decay buffer");
            self.encode_implicit_filter(
                encoder.as_ref(),
                freq,
                phase,
                decay,
                filter_first,
                time_u32,
                order_u32,
                fft_len,
                filter_elements_u32,
            )?;
        }
        let dispatch_two = |pipeline: &objc2::runtime::ProtocolObject<
            dyn MTLComputePipelineState,
        >,
                            input: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                            output: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                            total: usize,
                            scalars: &[u32]|
         -> Result<()> {
            self.encode_fft_two_buffer(encoder.as_ref(), pipeline, input, output, total, scalars)
        };
        let run_fft = |first: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       second: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       transforms: u32,
                       total: usize,
                       inverse: bool|
         -> Result<bool> {
            dispatch_two(
                &self.fft_bitreverse_pipeline,
                first,
                second,
                total,
                &[fft_len, transforms],
            )?;
            let mut source_is_first = false;
            for stage in 1..=plan.stages {
                let (source, destination) = if source_is_first {
                    (first, second)
                } else {
                    (second, first)
                };
                dispatch_two(
                    &self.fft_stage_pipeline,
                    source,
                    destination,
                    total,
                    &[fft_len, transforms, stage, u32::from(inverse)],
                )?;
                source_is_first = !source_is_first;
            }
            Ok(source_is_first)
        };
        let signal_source_is_first = run_fft(
            signal_first.as_ref(),
            signal_second.as_ref(),
            transforms_u32,
            signal_elements,
            false,
        )?;
        let filter_source_is_first = run_fft(
            filter_first.as_ref(),
            filter_second.as_ref(),
            channels_u32,
            filter_fft_elements,
            false,
        )?;
        let signal_spectrum = if signal_source_is_first {
            signal_first.as_ref()
        } else {
            signal_second.as_ref()
        };
        let filter_spectrum = if filter_source_is_first {
            filter_first.as_ref()
        } else {
            filter_second.as_ref()
        };
        let multiply_output_is_first = !signal_source_is_first;
        let product = if multiply_output_is_first {
            signal_first.as_ref()
        } else {
            signal_second.as_ref()
        };
        self.encode_fft_multiply(
            encoder.as_ref(),
            signal_spectrum,
            filter_spectrum,
            product,
            fft_len,
            channels_u32,
            transforms_u32,
            signal_elements,
        )?;
        let inverse_bitreversed_is_first = !multiply_output_is_first;
        let inverse_bitreversed = if inverse_bitreversed_is_first {
            signal_first.as_ref()
        } else {
            signal_second.as_ref()
        };
        dispatch_two(
            &self.fft_bitreverse_pipeline,
            product,
            inverse_bitreversed,
            signal_elements,
            &[fft_len, transforms_u32],
        )?;
        let mut inverse_source_is_first = inverse_bitreversed_is_first;
        for stage in 1..=plan.stages {
            let (source, destination) = if inverse_source_is_first {
                (signal_first.as_ref(), signal_second.as_ref())
            } else {
                (signal_second.as_ref(), signal_first.as_ref())
            };
            dispatch_two(
                &self.fft_stage_pipeline,
                source,
                destination,
                signal_elements,
                &[fft_len, transforms_u32, stage, 1],
            )?;
            inverse_source_is_first = !inverse_source_is_first;
        }
        let inverse = if inverse_source_is_first {
            signal_first.as_ref()
        } else {
            signal_second.as_ref()
        };
        self.encode_causal_extract(
            encoder.as_ref(),
            inverse,
            output,
            time_u32,
            channels_u32,
            fft_len,
            elements_u32,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal Hyena causal convolution failed: {error}");
        }
        let mut result = vec![0.0; shape.elements()];
        // SAFETY: completion makes every written output element readable from the shared buffer.
        unsafe {
            result
                .as_mut_ptr()
                .copy_from_nonoverlapping(output.contents().cast::<f32>().as_ptr(), result.len());
        }
        Ok(result)
    }
}

/// Executes one packed ternary projection on Metal.
///
/// This is still a numerical-reference path: buffers and pipeline are created
/// per call so the result can be compared directly with the safe CPU model.
/// A later persistent runtime will own and reuse these Metal objects.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub fn ternary_linear_forward(
    input: &[f32],
    positive: &[u64],
    negative: &[u64],
    scales: &[f32],
    shape: TernaryLinearShape,
) -> Result<Vec<f32>> {
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLResourceOptions, MTLSize,
    };

    let input_len = shape
        .rows
        .checked_mul(shape.in_features)
        .ok_or_else(|| anyhow::anyhow!("ternary input shape overflow"))?;
    let output_len = shape
        .rows
        .checked_mul(shape.out_features)
        .ok_or_else(|| anyhow::anyhow!("ternary output shape overflow"))?;
    let packed_words = shape.packed_words()?;
    if input.len() != input_len
        || positive.len() != packed_words
        || negative.len() != packed_words
        || scales.len() != shape.out_features
    {
        bail!("Metal ternary projection shape mismatch");
    }
    let input_bytes = input_len
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("Metal input buffer byte size overflow"))?;
    let packed_bytes = packed_words
        .checked_mul(size_of::<u64>())
        .ok_or_else(|| anyhow::anyhow!("Metal ternary buffer byte size overflow"))?;
    let scale_bytes = scales
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("Metal scale buffer byte size overflow"))?;
    let output_bytes = output_len
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("Metal output buffer byte size overflow"))?;
    let rows = u32::try_from(shape.rows).map_err(|_| anyhow::anyhow!("Metal rows exceed u32"))?;
    let in_features = u32::try_from(shape.in_features)
        .map_err(|_| anyhow::anyhow!("Metal input width exceeds u32"))?;
    let out_features = u32::try_from(shape.out_features)
        .map_err(|_| anyhow::anyhow!("Metal output width exceeds u32"))?;

    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
    let source = NSString::from_str(HYENA_METAL_SOURCE);
    let options = MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| anyhow::anyhow!("Metal shader compilation failed: {error}"))?;
    let name = NSString::from_str(TERNARY_LINEAR_KERNEL_NAME);
    let function = library
        .newFunctionWithName(&name)
        .ok_or_else(|| anyhow::anyhow!("Metal ternary function is missing"))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| anyhow::anyhow!("Metal pipeline creation failed: {error}"))?;
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| anyhow::anyhow!("Metal command queue is unavailable"))?;
    let shared = MTLResourceOptions::StorageModeShared;
    let input_buffer = device
        .newBufferWithLength_options(input_bytes, shared)
        .ok_or_else(|| anyhow::anyhow!("Metal input buffer allocation failed"))?;
    let positive_buffer = device
        .newBufferWithLength_options(packed_bytes, shared)
        .ok_or_else(|| anyhow::anyhow!("Metal positive bitplane allocation failed"))?;
    let negative_buffer = device
        .newBufferWithLength_options(packed_bytes, shared)
        .ok_or_else(|| anyhow::anyhow!("Metal negative bitplane allocation failed"))?;
    let scale_buffer = device
        .newBufferWithLength_options(scale_bytes, shared)
        .ok_or_else(|| anyhow::anyhow!("Metal scale buffer allocation failed"))?;
    let output_buffer = device
        .newBufferWithLength_options(output_bytes, shared)
        .ok_or_else(|| anyhow::anyhow!("Metal output buffer allocation failed"))?;

    // SAFETY: Each shared buffer was allocated from the exact checked byte
    // count of its source slice. The command is only submitted after all host
    // copies complete, and retained buffers outlive the command.
    unsafe {
        input_buffer
            .contents()
            .cast::<f32>()
            .as_ptr()
            .copy_from_nonoverlapping(input.as_ptr(), input_len);
        positive_buffer
            .contents()
            .cast::<u64>()
            .as_ptr()
            .copy_from_nonoverlapping(positive.as_ptr(), packed_words);
        negative_buffer
            .contents()
            .cast::<u64>()
            .as_ptr()
            .copy_from_nonoverlapping(negative.as_ptr(), packed_words);
        scale_buffer
            .contents()
            .cast::<f32>()
            .as_ptr()
            .copy_from_nonoverlapping(scales.as_ptr(), scales.len());
    }

    let command = queue
        .commandBuffer()
        .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
    encoder.setComputePipelineState(&pipeline);
    // SAFETY: Buffer slots and scalar slots exactly mirror the MSL signature
    // for `ullis_ternary_linear`; scalar values are copied synchronously.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&positive_buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&negative_buffer), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&scale_buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&rows).cast::<c_void>(), size_of::<u32>(), 5);
        encoder.setBytes_length_atIndex(
            NonNull::from(&in_features).cast::<c_void>(),
            size_of::<u32>(),
            6,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&out_features).cast::<c_void>(),
            size_of::<u32>(),
            7,
        );
    }
    let thread_width = pipeline.maxTotalThreadsPerThreadgroup().min(output_len);
    if thread_width == 0 {
        bail!("Metal ternary pipeline reported zero threads per threadgroup");
    }
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: output_len,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    if let Some(error) = command.error() {
        bail!("Metal ternary command failed: {error}");
    }
    let mut output = vec![0.0; output_len];
    // SAFETY: The command completed successfully, so Metal no longer writes
    // the retained shared buffer, whose initialized byte count matches output.
    unsafe {
        output
            .as_mut_ptr()
            .copy_from_nonoverlapping(output_buffer.contents().cast::<f32>().as_ptr(), output_len);
    }
    Ok(output)
}

#[cfg(not(target_os = "macos"))]
pub fn identity_forward(_input: &[f32]) -> Result<Vec<f32>> {
    bail!("Ullis Metal backend requires macOS on Apple Silicon")
}

#[cfg(not(target_os = "macos"))]
pub fn rms_norm_forward(_input: &[f32], _rows: usize, _channels: usize) -> Result<Vec<f32>> {
    bail!("Ullis Metal backend requires macOS on Apple Silicon")
}

#[cfg(not(target_os = "macos"))]
pub fn ternary_linear_forward(
    _input: &[f32],
    _positive: &[u64],
    _negative: &[u64],
    _scales: &[f32],
    _shape: TernaryLinearShape,
) -> Result<Vec<f32>> {
    bail!("Ullis Metal backend requires macOS on Apple Silicon")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_shape_rejects_zero_and_32_bit_overflow() {
        assert!(MetalDispatchShape::new(0, 1, 1).is_err());
        assert!(MetalDispatchShape::new(1, 1, 0).is_err());
        assert!(MetalDispatchShape::new(65_536, 65_536, 1).is_err());
        assert_eq!(MetalDispatchShape::new(2, 3, 4).unwrap().elements(), 24);
    }

    #[test]
    fn ternary_shape_uses_two_packed_bitplanes() {
        assert_eq!(
            TernaryLinearShape::new(2, 65, 3)
                .unwrap()
                .packed_words()
                .unwrap(),
            4
        );
    }

    #[test]
    fn ternary_reference_decodes_both_bitplanes() {
        let shape = TernaryLinearShape::new(1, 3, 2).unwrap();
        let output = ternary_reference(
            &[2.0, 3.0, 5.0],
            &[0b011001],
            &[0b000010],
            &[2.0, 0.5],
            shape,
        )
        .unwrap();
        assert_eq!(output, vec![-2.0, 2.5]);
    }

    #[test]
    fn fused_reference_keeps_normalized_rows_virtual() {
        let shape = TernaryLinearShape::new(1, 3, 2).unwrap();
        let output = rms_norm_ternary_reference(
            &[3.0, 4.0, 0.0],
            &[0b011001],
            &[0b000010],
            &[2.0, 0.5],
            shape,
        )
        .unwrap();
        let inverse_rms = (25.0_f32 / 3.0 + 1e-5).sqrt().recip();
        assert!((output[0] + 2.0 * inverse_rms).abs() < 1e-6);
        assert!((output[1] - 3.5 * inverse_rms).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_gate_and_residual_match_cpu_when_metal_is_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let residual = [1.0, -2.0, 0.5, 3.0];
        let mixed = [2.0, 4.0, -1.0, 0.25];
        // Signal | gate for each row.
        let projection = [9.0, 8.0, 0.5, -0.25, 7.0, 6.0, -0.75, 0.4];
        let actual = runtime
            .resident_gate_residual_reference(&residual, &mixed, &projection, 2, 2)
            .unwrap();
        let expected = [2.0, -3.0, 1.25, 3.1];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_projection_ping_pongs_without_host_activation_copy() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let source = [1.0, -2.0, 0.5, 3.0];
        let slot = runtime.upload_resident_activations(&source, 2, 2).unwrap();
        // Identity ternary matrix, encoded row-major by output feature.
        let slot = runtime
            .resident_output_projection(slot, 2, 2, &[0b1001], &[0], &[1.0, 1.0])
            .unwrap();
        let actual = runtime.download_resident_activations(slot, 2, 2).unwrap();
        assert_eq!(actual, [2.0, -4.0, 1.0, 6.0]);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn fft_stage_kernels_compile_on_the_local_metal_device() {
        let shape = MetalDispatchShape::new(1, 8, 1).unwrap();
        for kernel in [
            FFT_BITREVERSE_KERNEL_NAME,
            FFT_STAGE_KERNEL_NAME,
            FFT_COMPLEX_MULTIPLY_KERNEL_NAME,
            FFT_EXTRACT_CAUSAL_KERNEL_NAME,
            IMPLICIT_FILTER_KERNEL_NAME,
            TANH_GATE_KERNEL_NAME,
            PACK_STRIDED_REAL_KERNEL_NAME,
            APPLY_GATE_KERNEL_NAME,
            RESIDUAL_ADD_KERNEL_NAME,
        ] {
            if let Ok(width) = validate_metal_kernel(kernel, shape) {
                assert!(width > 0);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn identity_shader_compiles_on_the_local_metal_device() {
        // CI and sandboxed shells may deliberately expose no GPU. The public
        // constructor reports that condition as a recoverable error; a local
        // Metal-enabled run still compiles the shader and checks its pipeline.
        if let Ok(width) = validate_metal_pipeline(MetalDispatchShape::new(1, 8, 16).unwrap()) {
            assert!(width > 0);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn rms_norm_shader_compiles_on_the_local_metal_device() {
        if let Ok(width) = validate_metal_kernel(
            RMS_NORM_KERNEL_NAME,
            MetalDispatchShape::new(1, 8, 16).unwrap(),
        ) {
            assert!(width > 0);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn ternary_shader_compiles_on_the_local_metal_device() {
        if let Ok(width) = validate_metal_kernel(
            TERNARY_LINEAR_KERNEL_NAME,
            MetalDispatchShape::new(1, 8, 16).unwrap(),
        ) {
            assert!(width > 0);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn identity_kernel_round_trips_fp32_data_when_metal_is_available() {
        let input = [-1.0, 0.0, 0.5, 3.25];
        if let Ok(output) = identity_forward(&input) {
            assert_eq!(output, input);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn rms_norm_matches_cpu_reference_when_metal_is_available() {
        let input = [3.0, 4.0, 0.0, 2.0, -2.0, 1.0];
        if let Ok(output) = rms_norm_forward(&input, 2, 3) {
            for (row, actual) in output.chunks_exact(3).enumerate() {
                let source = &input[row * 3..row * 3 + 3];
                let inv = (source.iter().map(|x| x * x).sum::<f32>() / 3.0 + 1e-5)
                    .sqrt()
                    .recip();
                for (&value, expected) in actual.iter().zip(source.iter().map(|x| x * inv)) {
                    assert!((value - expected).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn ternary_kernel_matches_cpu_reference_when_metal_is_available() {
        let shape = TernaryLinearShape::new(2, 3, 2).unwrap();
        let input = [2.0, 3.0, 5.0, -1.0, 4.0, 2.0];
        let positive = [0b011001];
        let negative = [0b000010];
        let scales = [2.0, 0.5];
        let expected = ternary_reference(&input, &positive, &negative, &scales, shape).unwrap();
        if let Ok(actual) = ternary_linear_forward(&input, &positive, &negative, &scales, shape) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn runtime_reuses_ternary_pipeline_and_buffers_when_metal_is_available() {
        let shape = TernaryLinearShape::new(2, 3, 2).unwrap();
        let input = [2.0, 3.0, 5.0, -1.0, 4.0, 2.0];
        let positive = [0b011001];
        let negative = [0b000010];
        let scales = [2.0, 0.5];
        let expected = ternary_reference(&input, &positive, &negative, &scales, shape).unwrap();
        if let Ok(runtime) = MetalRuntime::new() {
            let first = runtime
                .ternary_linear_forward(&input, &positive, &negative, &scales, shape)
                .unwrap();
            let second = runtime
                .ternary_linear_forward(&input, &positive, &negative, &scales, shape)
                .unwrap();
            assert_eq!(first, expected);
            assert_eq!(second, expected);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn fused_runtime_matches_cpu_reference_when_metal_is_available() {
        let shape = TernaryLinearShape::new(2, 3, 2).unwrap();
        let input = [3.0, 4.0, 0.0, -1.0, 4.0, 2.0];
        let positive = [0b011001];
        let negative = [0b000010];
        let scales = [2.0, 0.5];
        let expected =
            rms_norm_ternary_reference(&input, &positive, &negative, &scales, shape).unwrap();
        if let Ok(runtime) = MetalRuntime::new() {
            let actual = runtime
                .rms_norm_ternary_linear_forward(&input, &positive, &negative, &scales, shape)
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn cached_fft_round_trips_complex_data_when_metal_is_available() {
        let input = [(1.0, 0.0), (2.0, -1.0), (0.5, 3.0), (-2.0, 0.25)];
        if let Ok(runtime) = MetalRuntime::new() {
            let spectrum = runtime.fft_reference(&input, 1, false).unwrap();
            let output = runtime.fft_reference(&spectrum, 1, true).unwrap();
            for (actual, expected) in output.iter().zip(input) {
                assert!((actual.0 - expected.0).abs() < 1e-5);
                assert!((actual.1 - expected.1).abs() < 1e-5);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn cached_fft_matches_known_forward_spectrum_when_metal_is_available() {
        let input = [(1.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let expected = [(3.0, 0.0), (1.0, -2.0), (-1.0, 0.0), (1.0, 2.0)];
        if let Ok(runtime) = MetalRuntime::new() {
            let actual = runtime.fft_reference(&input, 1, false).unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual.0 - expected.0).abs() < 1e-5);
                assert!((actual.1 - expected.1).abs() < 1e-5);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn causal_hyena_chain_matches_cpu_reference_when_metal_is_available() {
        let input = [0.5, -1.0, 1.5, 2.0, -0.5, 3.0, 4.0, -2.0];
        let filter = [0.5, 0.25, -0.5, 0.0, 1.0, -1.0, 0.5, 0.0];
        let expected = crate::hyena::causal_long_conv(&input, &filter, 1, 4, 2).unwrap();
        if let Ok(runtime) = MetalRuntime::new() {
            let actual = runtime
                .causal_long_conv_forward(&input, &filter, 1, 4, 2)
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn implicit_filter_matches_cpu_reference_when_metal_is_available() {
        let filter = crate::hyena::ImplicitFilter::new(2, 3, 7);
        let expected = filter.generate(2, 4).unwrap();
        if let Ok(runtime) = MetalRuntime::new() {
            let actual = runtime.implicit_filter_forward(&filter, 2, 4).unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn implicit_hyena_chain_matches_cpu_reference_when_metal_is_available() {
        let filter = crate::hyena::ImplicitFilter::new(2, 3, 7);
        let input = [0.5, -1.0, 1.5, 2.0, -0.5, 3.0, 4.0, -2.0];
        let expected = crate::hyena::causal_long_conv_implicit(&input, &filter, 1, 4, 2).unwrap();
        if let Ok(runtime) = MetalRuntime::new() {
            let actual = runtime
                .causal_long_conv_implicit_forward(&input, &filter, 1, 4, 2)
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn tanh_gate_keeps_signal_and_transforms_gate_when_metal_is_available() {
        let input = [1.0, -2.0, 0.0, 1.0, 3.0, -0.5, -1.0, 2.0];
        if let Ok(runtime) = MetalRuntime::new() {
            let actual = runtime.tanh_gate_forward(&input, 2, 2).unwrap();
            for row in 0..2 {
                assert_eq!(&actual[row * 4..row * 4 + 2], &input[row * 4..row * 4 + 2]);
                for column in 0..2 {
                    let index = row * 4 + 2 + column;
                    assert!((actual[index] - input[index].tanh()).abs() < 1e-6);
                }
            }
        }
    }
}
