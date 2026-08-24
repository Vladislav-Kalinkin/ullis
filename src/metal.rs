//! Metal forward-path admission and shader-pipeline validation.
//!
//! This module intentionally starts with pipeline construction only. Buffer
//! mapping is the one place where Metal requires raw pointers; it will live in
//! a small, audited follow-up boundary rather than weakening the safe CPU model
//! API throughout the crate.

use anyhow::{Result, bail};

#[cfg(target_os = "macos")]
use crate::hyena::{HyenaChunkPlan, HyenaFftPlan};

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
pub const CLIPPED_SGD_FP16_KERNEL_NAME: &str = "ullis_clipped_sgd_fp16";
pub const STREAMED_CROSS_ENTROPY_FP16_KERNEL_NAME: &str = "ullis_streamed_cross_entropy_fp16";
pub const TERNARY_ROW_SCALES_FP16_KERNEL_NAME: &str = "ullis_ternary_row_scales_fp16";
pub const REFRESH_TERNARY_CODES_FP16_KERNEL_NAME: &str = "ullis_refresh_ternary_codes_fp16";
pub const RMS_NORM_KERNEL_NAME: &str = "ullis_rms_norm";
pub const RMS_NORM_BACKWARD_KERNEL_NAME: &str = "ullis_rms_norm_backward";
pub const TERNARY_LINEAR_KERNEL_NAME: &str = "ullis_ternary_linear";
pub const TERNARY_LINEAR_FP16_KERNEL_NAME: &str = "ullis_ternary_linear_fp16";
pub const TERNARY_INPUT_BACKWARD_KERNEL_NAME: &str = "ullis_ternary_linear_input_backward";
pub const TERNARY_STE_WEIGHT_BACKWARD_KERNEL_NAME: &str =
    "ullis_ternary_linear_ste_weight_backward";
pub const CAUSAL_CONV_INPUT_BACKWARD_KERNEL_NAME: &str = "ullis_causal_conv_input_backward";
pub const CAUSAL_CONV_FILTER_BACKWARD_KERNEL_NAME: &str = "ullis_causal_conv_filter_backward";
pub const EXTRACT_PROJECTION_SIGNAL_KERNEL_NAME: &str = "ullis_extract_projection_signal";
pub const ADD_PROJECTION_SIGNAL_GRADIENT_KERNEL_NAME: &str = "ullis_add_projection_signal_gradient";
pub const RMS_NORM_TERNARY_LINEAR_KERNEL_NAME: &str = "ullis_rms_norm_ternary_linear";
pub const FFT_BITREVERSE_KERNEL_NAME: &str = "ullis_fft_bitreverse";
pub const FFT_STAGE_KERNEL_NAME: &str = "ullis_fft_stage";
pub const FFT_COMPLEX_MULTIPLY_KERNEL_NAME: &str = "ullis_fft_complex_multiply";
pub const PACK_REVERSE_GRADIENT_KERNEL_NAME: &str = "ullis_pack_reverse_gradient_to_complex";
pub const PACK_FILTER_KERNEL_NAME: &str = "ullis_pack_filter_to_complex";
pub const FFT_EXTRACT_INPUT_BACKWARD_KERNEL_NAME: &str = "ullis_fft_extract_input_backward";
pub const FFT_EXTRACT_CAUSAL_KERNEL_NAME: &str = "ullis_fft_extract_causal";
pub const IMPLICIT_FILTER_KERNEL_NAME: &str = "ullis_generate_implicit_filter";
pub const IMPLICIT_FILTER_FP16_KERNEL_NAME: &str = "ullis_generate_implicit_filter_fp16";
pub const TANH_GATE_KERNEL_NAME: &str = "ullis_tanh_gate_in_place";
pub const TANH_GATE_FP16_KERNEL_NAME: &str = "ullis_tanh_gate_fp16";
pub const PACK_STRIDED_REAL_KERNEL_NAME: &str = "ullis_pack_strided_real_to_complex";
pub const PACK_OVERLAP_SAVE_KERNEL_NAME: &str = "ullis_pack_overlap_save_to_complex";
pub const EXTRACT_OVERLAP_SAVE_KERNEL_NAME: &str = "ullis_extract_overlap_save";
pub const APPLY_GATE_KERNEL_NAME: &str = "ullis_apply_gate";
pub const APPLY_GATE_FP16_KERNEL_NAME: &str = "ullis_apply_gate_fp16";
pub const HYENA_GATE_BACKWARD_KERNEL_NAME: &str = "ullis_hyena_gate_backward";
pub const RESIDUAL_ADD_KERNEL_NAME: &str = "ullis_residual_add";
pub const RESIDUAL_ADD_FP16_KERNEL_NAME: &str = "ullis_residual_add_fp16";
pub const HYENA_METAL_SOURCE: &str = include_str!("metal/hyena.metal");

/// Checked dimensions for one packed-ternary projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TernaryLinearShape {
    pub rows: usize,
    pub in_features: usize,
    pub out_features: usize,
}

/// GPU result for the exact local derivative of `mixed * tanh(gate)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MetalHyenaGateBackward {
    pub mixed_gradient: Vec<f32>,
    pub projection_gradient: Vec<f32>,
}

/// GPU result for the packed-forward input derivative and ternary STE weight
/// surrogate. Both vectors are FP32 local workspaces, not persistent state.
#[derive(Clone, Debug, PartialEq)]
pub struct MetalTernaryLinearBackward {
    pub input_gradient: Vec<f32>,
    pub latent_weight_gradient: Vec<f32>,
}

/// GPU result for exact bounded causal-convolution derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct MetalCausalConvBackward {
    pub input_gradient: Vec<f32>,
    pub filter_gradient: Vec<f32>,
}

/// Final tensors read back from one resident Hyena block backward command.
#[derive(Clone, Debug, PartialEq)]
pub struct MetalHyenaBlockBackward {
    pub input_gradient: Vec<f32>,
    pub input_projection_weight_gradient: Vec<f32>,
    pub output_projection_weight_gradient: Vec<f32>,
    pub filter_gradient: Vec<f32>,
}

/// Readbacks needed after a cached block backward pass that updates resident
/// projection weights in place. Projection gradients never cross the host
/// boundary in this path.
#[derive(Clone, Debug, PartialEq)]
pub struct MetalHyenaBlockUpdatedBackward {
    pub input_gradient: Vec<f32>,
    pub filter_gradient: Vec<f32>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct ResidentTernaryUpdates<'a> {
    input: &'a ResidentTrainableFp16TernaryWeights,
    output: &'a ResidentTrainableFp16TernaryWeights,
    learning_rate: f32,
}

#[cfg(target_os = "macos")]
enum CachedBlockBackwardResult {
    Reference(MetalHyenaBlockBackward),
    Updated(MetalHyenaBlockUpdatedBackward),
}

/// Forward values retained on Metal for one Hyena block training pass.
///
/// These are deliberately independent of the runtime's ping-pong scratch
/// slots: a later block is free to reuse those slots while this cache remains
/// valid for reverse traversal.  No tensor is copied through the CPU.
#[cfg(target_os = "macos")]
pub struct ResidentHyenaBlockCache {
    input: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    normalized_input:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    gated_projection:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    mixed: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    rows: usize,
    channels: usize,
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
    ternary_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    identity_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    clipped_sgd_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    streamed_cross_entropy_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    ternary_row_scales_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    refresh_ternary_codes_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    rms_norm_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    ternary_input_backward_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    ternary_ste_weight_backward_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    causal_conv_input_backward_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    causal_conv_filter_backward_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    extract_projection_signal_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    add_projection_signal_gradient_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    rms_norm_backward_pipeline: objc2::rc::Retained<
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
    pack_reverse_gradient_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pack_filter_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    fft_extract_input_backward_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    fft_extract_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    implicit_filter_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    implicit_filter_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    tanh_gate_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    tanh_gate_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pack_strided_real_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pack_overlap_save_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    extract_overlap_save_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    apply_gate_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    apply_gate_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    hyena_gate_backward_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    residual_add_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    residual_add_fp16_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    ternary_buffers: std::cell::RefCell<TernaryBuffers>,
    fft_buffers: std::cell::RefCell<FftBuffers>,
    filter_fft_buffers: std::cell::RefCell<FftBuffers>,
    hyena_output_buffer: std::cell::RefCell<OutputBuffer>,
    implicit_filter_parameters: std::cell::RefCell<ImplicitFilterParameters>,
    gate_buffers: std::cell::RefCell<GateBuffers>,
    activations: std::cell::RefCell<ActivationBuffers>,
    gradient_activations: std::cell::RefCell<ActivationBuffers>,
    fp16_activations: std::cell::RefCell<Fp16ActivationBuffers>,
    streamed_cross_entropy: std::cell::RefCell<StreamedCrossEntropyBuffers>,
    backward_buffers: std::cell::RefCell<BackwardBuffers>,
    block_backward_buffers: std::cell::RefCell<BlockBackwardBuffers>,
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

/// Which resident slot owns the current reverse-mode gradient. It is separate
/// from forward activation ping-pong so all block caches remain valid during
/// the complete reverse traversal.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentGradientSlot {
    First,
    Second,
}

#[cfg(target_os = "macos")]
impl ResidentGradientSlot {
    pub const fn other(self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::First,
        }
    }
}

/// Slot token for the FP16 resident training/inference stream.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentFp16ActivationSlot {
    First,
    Second,
    Third,
}

#[cfg(target_os = "macos")]
impl ResidentFp16ActivationSlot {
    pub const fn other(self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::Third,
            Self::Third => Self::First,
        }
    }
}

/// Immutable packed ternary weights retained by Metal across FP16 projection
/// dispatches. The caller owns this object, making upload lifetime explicit.
#[cfg(target_os = "macos")]
pub struct ResidentFp16TernaryWeights {
    positive: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    negative: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    scales: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    shape: TernaryLinearShape,
}

/// Persistent FP16 master parameters for a stateless Metal optimizer step.
///
/// The object contains parameters only: gradients are transient and the
/// optimizer retains neither momentum nor variance. It is intentionally
/// separate from packed ternary inference weights while GPU code-refresh is
/// brought online.
#[cfg(target_os = "macos")]
pub struct ResidentFp16Parameters {
    parameters: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    len: usize,
}

/// Exact streamed cross-entropy result at the explicit CPU graph boundary.
/// The GPU never materializes per-vocabulary logits or probabilities.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq)]
pub struct MetalStreamedCrossEntropy {
    pub loss_sum: f32,
    pub token_count: usize,
    pub head_gradient: Vec<f32>,
}

/// Compact loss statistics returned by the all-resident cross-entropy path.
/// Its `D`-wide derivative remains in a [`ResidentGradientSlot`].
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq)]
pub struct MetalResidentCrossEntropy {
    pub loss_sum: f32,
    pub token_count: usize,
    pub gradient_slot: ResidentGradientSlot,
}

/// Fully resident, trainable ternary projection state.
///
/// FP16 masters update in place; Metal then rebuilds scales and packed
/// bitplanes in the same command submission. No optimizer state is retained.
#[cfg(target_os = "macos")]
pub struct ResidentTrainableFp16TernaryWeights {
    master: ResidentFp16Parameters,
    positive: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    negative: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    scales: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    in_features: usize,
    out_features: usize,
    threshold_ratio: f32,
}

/// Compact, persistent FP16 state for one implicit Hyena filter. It contains
/// only the three generator vectors; no optimiser moments are retained.
#[cfg(target_os = "macos")]
pub struct ResidentImplicitFilterParameters {
    freq: ResidentFp16Parameters,
    phase: ResidentFp16Parameters,
    decay: ResidentFp16Parameters,
    channels: usize,
    order: usize,
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

/// Two FP16 activation buffers ping-ponged across resident operations.
#[cfg(target_os = "macos")]
#[derive(Default)]
struct Fp16ActivationBuffers {
    first: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    second: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    third: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    capacity: usize,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct StreamedCrossEntropyBuffers {
    head: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    tokens: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    gradient:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    loss: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    head_capacity: usize,
    tokens_capacity: usize,
    gradient_capacity: usize,
    loss_capacity: usize,
}

/// Five independent, grow-only FP32 buffers shared by the local backward
/// reference kernels. They are deliberately named by data-flow role rather
/// than operation: the same allocation serves ternary, convolution, and
/// RMSNorm without a per-dispatch Metal allocation.
#[cfg(target_os = "macos")]
#[derive(Default)]
struct BackwardBuffers {
    source: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    auxiliary:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    output_gradient:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    input_gradient:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    parameter_gradient:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    source_capacity: usize,
    auxiliary_capacity: usize,
    output_gradient_capacity: usize,
    input_gradient_capacity: usize,
    parameter_gradient_capacity: usize,
}

/// Tensor slots retained for one complete FP32 Hyena block backward pass.
/// The slots are indexed by the graph encoder rather than exposed as a public
/// tensor API: this keeps ownership local and makes every capacity grow-only.
#[cfg(target_os = "macos")]
#[derive(Default)]
struct BlockBackwardBuffers {
    buffers: Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    capacities: Vec<usize>,
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
enum ResidentImplicitFilterSource<'a> {
    Host(&'a crate::hyena::ImplicitFilter),
    Trainable(&'a ResidentImplicitFilterParameters),
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
impl Fp16ActivationBuffers {
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
                    .ok_or_else(|| anyhow::anyhow!("Metal FP16 activation allocation failed"))?,
            );
            self.second = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal FP16 activation scratch allocation failed")
                    })?,
            );
            self.third = Some(
                device
                    .newBufferWithLength_options(bytes, options)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal FP16 activation work allocation failed")
                    })?,
            );
            self.capacity = bytes;
        }
        Ok(())
    }

    fn buffer(
        &self,
        slot: ResidentFp16ActivationSlot,
    ) -> Result<&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>> {
        let buffer = match slot {
            ResidentFp16ActivationSlot::First => self.first.as_deref(),
            ResidentFp16ActivationSlot::Second => self.second.as_deref(),
            ResidentFp16ActivationSlot::Third => self.third.as_deref(),
        };
        buffer.ok_or_else(|| anyhow::anyhow!("Metal FP16 activations are not allocated"))
    }
}

#[cfg(target_os = "macos")]
impl StreamedCrossEntropyBuffers {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        head_bytes: usize,
        token_bytes: usize,
        gradient_bytes: usize,
        loss_bytes: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};
        let shared = MTLResourceOptions::StorageModeShared;
        let grow = |slot: &mut Option<_>, capacity: &mut usize, bytes, name| -> Result<()> {
            if *capacity < bytes {
                *slot = Some(
                    device
                        .newBufferWithLength_options(bytes, shared)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Metal streamed cross-entropy {name} allocation failed")
                        })?,
                );
                *capacity = bytes;
            }
            Ok(())
        };
        grow(&mut self.head, &mut self.head_capacity, head_bytes, "head")?;
        grow(
            &mut self.tokens,
            &mut self.tokens_capacity,
            token_bytes,
            "tokens",
        )?;
        grow(
            &mut self.gradient,
            &mut self.gradient_capacity,
            gradient_bytes,
            "gradient",
        )?;
        grow(&mut self.loss, &mut self.loss_capacity, loss_bytes, "loss")
    }
}

#[cfg(target_os = "macos")]
impl BackwardBuffers {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        source_bytes: usize,
        auxiliary_bytes: usize,
        output_gradient_bytes: usize,
        input_gradient_bytes: usize,
        parameter_gradient_bytes: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};

        let shared = MTLResourceOptions::StorageModeShared;
        if self.source_capacity < source_bytes {
            self.source = Some(
                device
                    .newBufferWithLength_options(source_bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal backward source allocation failed"))?,
            );
            self.source_capacity = source_bytes;
        }
        if self.auxiliary_capacity < auxiliary_bytes {
            self.auxiliary = Some(
                device
                    .newBufferWithLength_options(auxiliary_bytes, shared)
                    .ok_or_else(|| anyhow::anyhow!("Metal backward auxiliary allocation failed"))?,
            );
            self.auxiliary_capacity = auxiliary_bytes;
        }
        if self.output_gradient_capacity < output_gradient_bytes {
            self.output_gradient = Some(
                device
                    .newBufferWithLength_options(output_gradient_bytes, shared)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal backward output-gradient allocation failed")
                    })?,
            );
            self.output_gradient_capacity = output_gradient_bytes;
        }
        if self.input_gradient_capacity < input_gradient_bytes {
            self.input_gradient = Some(
                device
                    .newBufferWithLength_options(input_gradient_bytes, shared)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal backward input-gradient allocation failed")
                    })?,
            );
            self.input_gradient_capacity = input_gradient_bytes;
        }
        if self.parameter_gradient_capacity < parameter_gradient_bytes {
            self.parameter_gradient = Some(
                device
                    .newBufferWithLength_options(parameter_gradient_bytes, shared)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal backward parameter-gradient allocation failed")
                    })?,
            );
            self.parameter_gradient_capacity = parameter_gradient_bytes;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl BlockBackwardBuffers {
    fn ensure(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        byte_requirements: &[usize],
    ) -> Result<()> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};

        let shared = MTLResourceOptions::StorageModeShared;
        for (index, &bytes) in byte_requirements.iter().enumerate() {
            if self.capacities.get(index).copied().unwrap_or_default() < bytes {
                let buffer = device
                    .newBufferWithLength_options(bytes, shared)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Metal block-backward slot {index} allocation failed")
                    })?;
                if index == self.buffers.len() {
                    self.buffers.push(buffer);
                    self.capacities.push(bytes);
                } else {
                    self.buffers[index] = buffer;
                    self.capacities[index] = bytes;
                }
            }
        }
        Ok(())
    }

    fn buffer(
        &self,
        index: usize,
    ) -> Result<&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>> {
        self.buffers
            .get(index)
            .map(objc2::rc::Retained::as_ref)
            .ok_or_else(|| anyhow::anyhow!("Metal block-backward slot {index} is not allocated"))
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
    /// Establishes an explicit host-visible boundary after queued resident
    /// work. Training uses this only before checkpoint readback.
    pub fn synchronize(&self) -> Result<()> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal synchronization command buffer allocation failed")
        })?;
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal synchronization failed: {error}");
        }
        Ok(())
    }

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
        let identity_name = NSString::from_str(IDENTITY_KERNEL_NAME);
        let identity_function = library
            .newFunctionWithName(&identity_name)
            .ok_or_else(|| anyhow::anyhow!("Metal identity function is missing"))?;
        let identity_pipeline = device
            .newComputePipelineStateWithFunction_error(&identity_function)
            .map_err(|error| anyhow::anyhow!("Metal identity pipeline failed: {error}"))?;
        let clipped_sgd_fp16_name = NSString::from_str(CLIPPED_SGD_FP16_KERNEL_NAME);
        let clipped_sgd_fp16_function = library
            .newFunctionWithName(&clipped_sgd_fp16_name)
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 SGD function is missing"))?;
        let clipped_sgd_fp16_pipeline = device
            .newComputePipelineStateWithFunction_error(&clipped_sgd_fp16_function)
            .map_err(|error| anyhow::anyhow!("Metal FP16 SGD pipeline failed: {error}"))?;
        let streamed_cross_entropy_name =
            NSString::from_str(STREAMED_CROSS_ENTROPY_FP16_KERNEL_NAME);
        let streamed_cross_entropy_function = library
            .newFunctionWithName(&streamed_cross_entropy_name)
            .ok_or_else(|| anyhow::anyhow!("Metal streamed cross-entropy function is missing"))?;
        let streamed_cross_entropy_fp16_pipeline = device
            .newComputePipelineStateWithFunction_error(&streamed_cross_entropy_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal streamed cross-entropy pipeline failed: {error}")
            })?;
        let rms_name = NSString::from_str(RMS_NORM_KERNEL_NAME);
        let rms_function = library
            .newFunctionWithName(&rms_name)
            .ok_or_else(|| anyhow::anyhow!("Metal RMSNorm function is missing"))?;
        let rms_norm_pipeline = device
            .newComputePipelineStateWithFunction_error(&rms_function)
            .map_err(|error| anyhow::anyhow!("Metal RMSNorm pipeline failed: {error}"))?;
        let name = NSString::from_str(TERNARY_LINEAR_KERNEL_NAME);
        let function = library
            .newFunctionWithName(&name)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary function is missing"))?;
        let ternary_pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| anyhow::anyhow!("Metal pipeline creation failed: {error}"))?;
        let fp16_name = NSString::from_str(TERNARY_LINEAR_FP16_KERNEL_NAME);
        let fp16_function = library
            .newFunctionWithName(&fp16_name)
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 ternary function is missing"))?;
        let ternary_fp16_pipeline = device
            .newComputePipelineStateWithFunction_error(&fp16_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal FP16 ternary pipeline creation failed: {error}")
            })?;
        let input_backward_name = NSString::from_str(TERNARY_INPUT_BACKWARD_KERNEL_NAME);
        let input_backward_function = library
            .newFunctionWithName(&input_backward_name)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary input-backward function is missing"))?;
        let ternary_input_backward_pipeline = device
            .newComputePipelineStateWithFunction_error(&input_backward_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal ternary input-backward pipeline failed: {error}")
            })?;
        let weight_backward_name = NSString::from_str(TERNARY_STE_WEIGHT_BACKWARD_KERNEL_NAME);
        let weight_backward_function = library
            .newFunctionWithName(&weight_backward_name)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary weight-backward function is missing"))?;
        let ternary_ste_weight_backward_pipeline = device
            .newComputePipelineStateWithFunction_error(&weight_backward_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal ternary weight-backward pipeline failed: {error}")
            })?;
        let causal_input_backward_name = NSString::from_str(CAUSAL_CONV_INPUT_BACKWARD_KERNEL_NAME);
        let causal_input_backward_function = library
            .newFunctionWithName(&causal_input_backward_name)
            .ok_or_else(|| anyhow::anyhow!("Metal causal input-backward function is missing"))?;
        let causal_conv_input_backward_pipeline = device
            .newComputePipelineStateWithFunction_error(&causal_input_backward_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal causal input-backward pipeline failed: {error}")
            })?;
        let causal_filter_backward_name =
            NSString::from_str(CAUSAL_CONV_FILTER_BACKWARD_KERNEL_NAME);
        let causal_filter_backward_function = library
            .newFunctionWithName(&causal_filter_backward_name)
            .ok_or_else(|| anyhow::anyhow!("Metal causal filter-backward function is missing"))?;
        let causal_conv_filter_backward_pipeline = device
            .newComputePipelineStateWithFunction_error(&causal_filter_backward_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal causal filter-backward pipeline failed: {error}")
            })?;
        let extract_projection_signal_name =
            NSString::from_str(EXTRACT_PROJECTION_SIGNAL_KERNEL_NAME);
        let extract_projection_signal_function = library
            .newFunctionWithName(&extract_projection_signal_name)
            .ok_or_else(|| anyhow::anyhow!("Metal signal-extract function is missing"))?;
        let extract_projection_signal_pipeline = device
            .newComputePipelineStateWithFunction_error(&extract_projection_signal_function)
            .map_err(|error| anyhow::anyhow!("Metal signal-extract pipeline failed: {error}"))?;
        let add_projection_signal_gradient_name =
            NSString::from_str(ADD_PROJECTION_SIGNAL_GRADIENT_KERNEL_NAME);
        let add_projection_signal_gradient_function = library
            .newFunctionWithName(&add_projection_signal_gradient_name)
            .ok_or_else(|| anyhow::anyhow!("Metal projection-gradient-add function is missing"))?;
        let add_projection_signal_gradient_pipeline = device
            .newComputePipelineStateWithFunction_error(&add_projection_signal_gradient_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal projection-gradient-add pipeline failed: {error}")
            })?;
        let rms_backward_name = NSString::from_str(RMS_NORM_BACKWARD_KERNEL_NAME);
        let rms_backward_function = library
            .newFunctionWithName(&rms_backward_name)
            .ok_or_else(|| anyhow::anyhow!("Metal RMSNorm backward function is missing"))?;
        let rms_norm_backward_pipeline = device
            .newComputePipelineStateWithFunction_error(&rms_backward_function)
            .map_err(|error| anyhow::anyhow!("Metal RMSNorm backward pipeline failed: {error}"))?;
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
        let implicit_fp16_name = NSString::from_str(IMPLICIT_FILTER_FP16_KERNEL_NAME);
        let implicit_fp16_function = library
            .newFunctionWithName(&implicit_fp16_name)
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 implicit-filter function is missing"))?;
        let implicit_filter_fp16_pipeline = device
            .newComputePipelineStateWithFunction_error(&implicit_fp16_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal FP16 implicit-filter pipeline creation failed: {error}")
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
        let fp16_gate_name = NSString::from_str(TANH_GATE_FP16_KERNEL_NAME);
        let fp16_gate_function = library
            .newFunctionWithName(&fp16_gate_name)
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 tanh-gate function is missing"))?;
        let tanh_gate_fp16_pipeline = device
            .newComputePipelineStateWithFunction_error(&fp16_gate_function)
            .map_err(|error| {
                anyhow::anyhow!("Metal FP16 tanh-gate pipeline creation failed: {error}")
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
        let ternary_row_scales_fp16_pipeline = make_pipeline(TERNARY_ROW_SCALES_FP16_KERNEL_NAME)?;
        let refresh_ternary_codes_fp16_pipeline =
            make_pipeline(REFRESH_TERNARY_CODES_FP16_KERNEL_NAME)?;
        let pack_strided_real_pipeline = make_pipeline(PACK_STRIDED_REAL_KERNEL_NAME)?;
        let pack_reverse_gradient_pipeline = make_pipeline(PACK_REVERSE_GRADIENT_KERNEL_NAME)?;
        let pack_filter_pipeline = make_pipeline(PACK_FILTER_KERNEL_NAME)?;
        let fft_extract_input_backward_pipeline =
            make_pipeline(FFT_EXTRACT_INPUT_BACKWARD_KERNEL_NAME)?;
        let pack_overlap_save_pipeline = make_pipeline(PACK_OVERLAP_SAVE_KERNEL_NAME)?;
        let extract_overlap_save_pipeline = make_pipeline(EXTRACT_OVERLAP_SAVE_KERNEL_NAME)?;
        let apply_gate_pipeline = make_pipeline(APPLY_GATE_KERNEL_NAME)?;
        let apply_gate_fp16_pipeline = make_pipeline(APPLY_GATE_FP16_KERNEL_NAME)?;
        let hyena_gate_backward_pipeline = make_pipeline(HYENA_GATE_BACKWARD_KERNEL_NAME)?;
        let residual_add_pipeline = make_pipeline(RESIDUAL_ADD_KERNEL_NAME)?;
        let residual_add_fp16_pipeline = make_pipeline(RESIDUAL_ADD_FP16_KERNEL_NAME)?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow::anyhow!("Metal command queue is unavailable"))?;
        Ok(Self {
            device,
            queue,
            identity_pipeline,
            clipped_sgd_fp16_pipeline,
            streamed_cross_entropy_fp16_pipeline,
            ternary_row_scales_fp16_pipeline,
            refresh_ternary_codes_fp16_pipeline,
            rms_norm_pipeline,
            ternary_pipeline,
            ternary_fp16_pipeline,
            ternary_input_backward_pipeline,
            ternary_ste_weight_backward_pipeline,
            causal_conv_input_backward_pipeline,
            causal_conv_filter_backward_pipeline,
            extract_projection_signal_pipeline,
            add_projection_signal_gradient_pipeline,
            rms_norm_backward_pipeline,
            fused_rms_norm_ternary_pipeline,
            fft_bitreverse_pipeline,
            fft_stage_pipeline,
            fft_multiply_pipeline,
            pack_reverse_gradient_pipeline,
            pack_filter_pipeline,
            fft_extract_input_backward_pipeline,
            fft_extract_pipeline,
            implicit_filter_pipeline,
            implicit_filter_fp16_pipeline,
            tanh_gate_pipeline,
            tanh_gate_fp16_pipeline,
            pack_strided_real_pipeline,
            pack_overlap_save_pipeline,
            extract_overlap_save_pipeline,
            apply_gate_pipeline,
            apply_gate_fp16_pipeline,
            hyena_gate_backward_pipeline,
            residual_add_pipeline,
            residual_add_fp16_pipeline,
            ternary_buffers: std::cell::RefCell::new(TernaryBuffers::default()),
            fft_buffers: std::cell::RefCell::new(FftBuffers::default()),
            filter_fft_buffers: std::cell::RefCell::new(FftBuffers::default()),
            hyena_output_buffer: std::cell::RefCell::new(OutputBuffer::default()),
            implicit_filter_parameters: std::cell::RefCell::new(ImplicitFilterParameters::default()),
            gate_buffers: std::cell::RefCell::new(GateBuffers::default()),
            activations: std::cell::RefCell::new(ActivationBuffers::default()),
            gradient_activations: std::cell::RefCell::new(ActivationBuffers::default()),
            fp16_activations: std::cell::RefCell::new(Fp16ActivationBuffers::default()),
            streamed_cross_entropy: std::cell::RefCell::new(StreamedCrossEntropyBuffers::default()),
            backward_buffers: std::cell::RefCell::new(BackwardBuffers::default()),
            block_backward_buffers: std::cell::RefCell::new(BlockBackwardBuffers::default()),
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

    /// Reserves the independent FP32 reverse-mode ping-pong pair.
    pub fn reserve_gradients(&self, rows: usize, width: usize) -> Result<()> {
        if rows == 0 || width == 0 {
            bail!("Metal gradient dimensions must be non-zero");
        }
        let bytes = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal gradient size overflow"))?;
        self.gradient_activations
            .borrow_mut()
            .ensure(&self.device, bytes)
    }

    /// Reserves two FP16 resident activation slots. The slots are deliberately
    /// independent from the legacy FP32 inference buffers.
    pub fn reserve_fp16_activations(&self, rows: usize, width: usize) -> Result<()> {
        if rows == 0 || width == 0 {
            bail!("Metal FP16 activation dimensions must be non-zero");
        }
        let bytes = rows
            .checked_mul(width)
            .and_then(|elements| elements.checked_mul(size_of::<u16>()))
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 activation size overflow"))?;
        self.fp16_activations
            .borrow_mut()
            .ensure(&self.device, bytes)
    }

    /// Reserves the complete FP32 cache and gradient set needed by one Hyena
    /// block backward graph. No allocation occurs while the graph is encoded
    /// when this is called at the training-shape boundary.
    pub fn reserve_block_backward(
        &self,
        rows: usize,
        channels: usize,
        kernel_len: usize,
    ) -> Result<()> {
        if rows == 0 || channels == 0 || kernel_len == 0 {
            bail!("Metal block-backward dimensions must be non-zero");
        }
        let activation = rows
            .checked_mul(channels)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal block-backward activation overflow"))?;
        let projection = activation
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal block-backward projection overflow"))?;
        let filter = channels
            .checked_mul(kernel_len)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal block-backward filter overflow"))?;
        // input, normalized, gated projection, mixed, upstream, gated mixed,
        // output-projection gradient, gate gradient, signal, signal gradient,
        // normalized-input gradient, final input gradient, and three parameter
        // gradients (output projection, input projection, filter), followed
        // by the materialized bounded filter used by the direct reference
        // adjoint.
        let mut buffers = self.block_backward_buffers.borrow_mut();
        buffers.ensure(
            &self.device,
            &[
                activation,
                activation,
                projection,
                activation,
                activation,
                activation,
                activation,
                projection,
                activation,
                activation,
                activation,
                activation,
                channels
                    .checked_mul(channels)
                    .and_then(|elements| elements.checked_mul(size_of::<f32>()))
                    .ok_or_else(|| anyhow::anyhow!("Metal output-weight gradient overflow"))?,
                channels
                    .checked_mul(channels)
                    .and_then(|elements| elements.checked_mul(2))
                    .and_then(|elements| elements.checked_mul(size_of::<f32>()))
                    .ok_or_else(|| anyhow::anyhow!("Metal input-weight gradient overflow"))?,
                filter,
                filter,
            ],
        )?;
        // Check that the first graph slot is immediately addressable; all
        // remaining slots were allocated by the same checked requirements.
        let _ = buffers.buffer(0)?;
        Ok(())
    }

    /// Allocates the retained forward tape for one block.  This happens once
    /// per live training microbatch; all capture dispatches merely write these
    /// buffers, so subsequent blocks cannot invalidate earlier caches.
    pub fn new_hyena_block_cache(
        &self,
        rows: usize,
        channels: usize,
    ) -> Result<ResidentHyenaBlockCache> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};

        if rows == 0 || channels == 0 {
            bail!("Metal Hyena cache dimensions must be non-zero");
        }
        let activation_bytes = rows
            .checked_mul(channels)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena cache activation overflow"))?;
        let projection_bytes = activation_bytes
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena cache projection overflow"))?;
        let shared = MTLResourceOptions::StorageModeShared;
        let allocate = |bytes, name: &str| {
            self.device
                .newBufferWithLength_options(bytes, shared)
                .ok_or_else(|| anyhow::anyhow!("Metal Hyena cache {name} allocation failed"))
        };
        Ok(ResidentHyenaBlockCache {
            input: allocate(activation_bytes, "input")?,
            normalized_input: allocate(activation_bytes, "normalized input")?,
            gated_projection: allocate(projection_bytes, "gated projection")?,
            mixed: allocate(activation_bytes, "mixed")?,
            rows,
            channels,
        })
    }

    /// Encodes the complete exact-reference backward graph for one cached
    /// Hyena block. The public compatibility boundary uploads and downloads
    /// gradients once; stack training should use
    /// [`Self::hyena_block_backward_cached_from_resident_gradient`] to keep
    /// inter-block gradients on Metal.
    #[allow(unsafe_code)]
    pub fn hyena_block_backward_cached_reference(
        &self,
        cache: &ResidentHyenaBlockCache,
        upstream: &[f32],
        input_positive: &[u64],
        input_negative: &[u64],
        input_scales: &[f32],
        output_positive: &[u64],
        output_negative: &[u64],
        output_scales: &[f32],
        filter: &[f32],
        batch: usize,
        time: usize,
        plan: HyenaChunkPlan,
    ) -> Result<MetalHyenaBlockBackward> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal block backward row overflow"))?;
        let upstream_slot = self.upload_resident_gradient(upstream, rows, cache.channels)?;
        self.hyena_block_backward_cached_from_resident_gradient(
            cache,
            upstream_slot,
            upstream_slot.other(),
            input_positive,
            input_negative,
            input_scales,
            output_positive,
            output_negative,
            output_scales,
            filter,
            batch,
            time,
            plan,
        )
    }

    /// Encodes one cached Hyena block backward pass with resident gradient
    /// ping-pong. `upstream_slot` is consumed as GPU input and the residual
    /// predecessor is written to `destination_slot`; no host gradient is used
    /// to feed the preceding block. Parameter gradients remain explicit CPU
    /// readbacks for the reference updater.
    #[allow(unsafe_code)]
    pub fn hyena_block_backward_cached_from_resident_gradient(
        &self,
        cache: &ResidentHyenaBlockCache,
        upstream_slot: ResidentGradientSlot,
        destination_slot: ResidentGradientSlot,
        input_positive: &[u64],
        input_negative: &[u64],
        input_scales: &[f32],
        output_positive: &[u64],
        output_negative: &[u64],
        output_scales: &[f32],
        filter: &[f32],
        batch: usize,
        time: usize,
        plan: HyenaChunkPlan,
    ) -> Result<MetalHyenaBlockBackward> {
        match self.hyena_block_backward_cached_impl(
            cache,
            upstream_slot,
            destination_slot,
            input_positive,
            input_negative,
            input_scales,
            output_positive,
            output_negative,
            output_scales,
            filter,
            batch,
            time,
            plan,
            None,
            true,
            true,
        )? {
            CachedBlockBackwardResult::Reference(result) => Ok(result),
            CachedBlockBackwardResult::Updated(_) => {
                unreachable!("reference backward has no updates")
            }
        }
    }

    /// Runs cached block backward and applies its two ternary projection
    /// gradients directly to resident FP16 masters.  Derived row scales and
    /// packed codes are rebuilt in the same second command buffer, so neither
    /// projection gradient is copied back to the CPU.
    #[allow(unsafe_code)]
    pub fn hyena_block_backward_cached_and_update_resident(
        &self,
        cache: &ResidentHyenaBlockCache,
        upstream_slot: ResidentGradientSlot,
        destination_slot: ResidentGradientSlot,
        input_positive: &[u64],
        input_negative: &[u64],
        input_scales: &[f32],
        output_positive: &[u64],
        output_negative: &[u64],
        output_scales: &[f32],
        input_weights: &ResidentTrainableFp16TernaryWeights,
        output_weights: &ResidentTrainableFp16TernaryWeights,
        filter: &[f32],
        batch: usize,
        time: usize,
        plan: HyenaChunkPlan,
        learning_rate: f32,
        compute_filter_gradient: bool,
        readback: bool,
    ) -> Result<MetalHyenaBlockUpdatedBackward> {
        match self.hyena_block_backward_cached_impl(
            cache,
            upstream_slot,
            destination_slot,
            input_positive,
            input_negative,
            input_scales,
            output_positive,
            output_negative,
            output_scales,
            filter,
            batch,
            time,
            plan,
            Some(ResidentTernaryUpdates {
                input: input_weights,
                output: output_weights,
                learning_rate,
            }),
            compute_filter_gradient,
            readback,
        )? {
            CachedBlockBackwardResult::Updated(result) => Ok(result),
            CachedBlockBackwardResult::Reference(_) => unreachable!("resident update requested"),
        }
    }

    #[allow(unsafe_code)]
    fn hyena_block_backward_cached_impl(
        &self,
        cache: &ResidentHyenaBlockCache,
        upstream_slot: ResidentGradientSlot,
        destination_slot: ResidentGradientSlot,
        input_positive: &[u64],
        input_negative: &[u64],
        input_scales: &[f32],
        output_positive: &[u64],
        output_negative: &[u64],
        output_scales: &[f32],
        filter: &[f32],
        batch: usize,
        time: usize,
        plan: HyenaChunkPlan,
        updates: Option<ResidentTernaryUpdates<'_>>,
        compute_filter_gradient: bool,
        readback: bool,
    ) -> Result<CachedBlockBackwardResult> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputePipelineState,
        };

        let plan = plan.for_sequence(time)?;
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal block backward row overflow"))?;
        let channels = cache.channels;
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal block backward activation overflow"))?;
        let projected = elements
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal block backward projection overflow"))?;
        let filter_elements = channels
            .checked_mul(plan.kernel_len)
            .ok_or_else(|| anyhow::anyhow!("Metal block backward filter overflow"))?;
        let input_shape = TernaryLinearShape::new(rows, channels, 2 * channels)?;
        let output_shape = TernaryLinearShape::new(rows, channels, channels)?;
        let host_weights = updates.is_none();
        if cache.rows != rows
            || filter.len() != filter_elements
            || (host_weights
                && (input_positive.len() != input_shape.packed_words()?
                    || input_negative.len() != input_positive.len()
                    || input_scales.len() != 2 * channels
                    || output_positive.len() != output_shape.packed_words()?
                    || output_negative.len() != output_positive.len()
                    || output_scales.len() != channels))
            || filter.iter().any(|value| !value.is_finite())
        {
            bail!("Metal cached block backward shape/value mismatch");
        }
        if let Some(updates) = updates
            && (!updates.learning_rate.is_finite()
                || updates.learning_rate <= 0.0
                || updates.input.in_features != channels
                || updates.input.out_features != 2 * channels
                || updates.output.in_features != channels
                || updates.output.out_features != channels)
        {
            bail!("Metal cached block resident update shape/value mismatch");
        }
        self.reserve_gradients(rows, channels)?;
        self.reserve_block_backward(rows, channels, plan.kernel_len)?;
        // The adjoint of a causal convolution is a convolution of the
        // time-reversed output gradient with the original filter. Keep that
        // transform resident and reuse the grow-only forward FFT scratch.
        let fft_plan = HyenaFftPlan::new(time)?;
        let transforms = batch
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal block backward transform overflow"))?;
        let signal_fft_elements = transforms
            .checked_mul(fft_plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal block backward signal FFT overflow"))?;
        let filter_fft_elements = channels
            .checked_mul(fft_plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal block backward filter FFT overflow"))?;
        let signal_fft_bytes = signal_fft_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal block backward signal FFT size overflow"))?;
        let filter_fft_bytes = filter_fft_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal block backward filter FFT size overflow"))?;
        self.fft_buffers
            .borrow_mut()
            .ensure(&self.device, signal_fft_bytes)?;
        self.filter_fft_buffers
            .borrow_mut()
            .ensure(&self.device, filter_fft_bytes)?;
        let input_packed_bytes = input_positive
            .len()
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("Metal block backward input code overflow"))?;
        let output_packed_bytes = output_positive
            .len()
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("Metal block backward output code overflow"))?;
        let packed_bytes = input_packed_bytes.max(output_packed_bytes);
        let scale_bytes = size_of_val(input_scales).max(size_of_val(output_scales));
        let mut ternary = self.ternary_buffers.borrow_mut();
        ternary.ensure(&self.device, 0, packed_bytes, scale_bytes, 0)?;
        let positive = ternary.positive.as_ref();
        let negative = ternary.negative.as_ref();
        let scales = ternary.scales.as_ref();
        let buffers = self.block_backward_buffers.borrow();
        let upstream_buffer = buffers.buffer(4)?;
        let gated_mixed = buffers.buffer(5)?;
        let output_input_gradient = buffers.buffer(6)?;
        let projection_gradient = buffers.buffer(7)?;
        let signal = buffers.buffer(8)?;
        let signal_gradient = buffers.buffer(9)?;
        let normalized_gradient = buffers.buffer(10)?;
        let input_gradient = buffers.buffer(11)?;
        let output_weight_gradient = buffers.buffer(12)?;
        let input_weight_gradient = buffers.buffer(13)?;
        let filter_gradient = buffers.buffer(14)?;
        let filter_buffer = buffers.buffer(15)?;
        let gradients = self.gradient_activations.borrow();
        let resident_upstream = match upstream_slot {
            ResidentGradientSlot::First => gradients.first.as_ref(),
            ResidentGradientSlot::Second => gradients.second.as_ref(),
        }
        .expect("checked Metal resident gradient source");
        let resident_destination = match destination_slot {
            ResidentGradientSlot::First => gradients.first.as_ref(),
            ResidentGradientSlot::Second => gradients.second.as_ref(),
        }
        .expect("checked Metal resident gradient destination");
        if std::ptr::eq(resident_upstream, resident_destination) {
            bail!("Metal cached block backward gradient slots must differ");
        }
        // SAFETY: validated exact-size shared buffers are written before the
        // command begins, and immutable filter values are uploaded once.
        unsafe {
            filter_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(filter.as_ptr(), filter_elements);
        }
        let rows_u32 = u32::try_from(rows).map_err(|_| anyhow::anyhow!("Metal rows exceed u32"))?;
        let channels_u32 =
            u32::try_from(channels).map_err(|_| anyhow::anyhow!("Metal channels exceed u32"))?;
        let elements_u32 =
            u32::try_from(elements).map_err(|_| anyhow::anyhow!("Metal elements exceed u32"))?;
        let kernel_u32 = u32::try_from(plan.kernel_len)
            .map_err(|_| anyhow::anyhow!("Metal kernel length exceeds u32"))?;
        let batch_u32 =
            u32::try_from(batch).map_err(|_| anyhow::anyhow!("Metal batch exceeds u32"))?;
        let time_u32 =
            u32::try_from(time).map_err(|_| anyhow::anyhow!("Metal time exceeds u32"))?;
        let fft_len_u32 = u32::try_from(fft_plan.fft_len)
            .map_err(|_| anyhow::anyhow!("Metal backward FFT length exceeds u32"))?;
        let transforms_u32 = u32::try_from(transforms)
            .map_err(|_| anyhow::anyhow!("Metal backward transform count exceeds u32"))?;
        let signal_fft_elements_u32 = u32::try_from(signal_fft_elements)
            .map_err(|_| anyhow::anyhow!("Metal backward FFT elements exceed u32"))?;
        let filter_fft_elements_u32 = u32::try_from(filter_fft_elements)
            .map_err(|_| anyhow::anyhow!("Metal backward filter FFT elements exceed u32"))?;
        let signal_fft = self.fft_buffers.borrow();
        let filter_fft = self.filter_fft_buffers.borrow();
        let signal_first = signal_fft
            .first
            .as_ref()
            .expect("checked Metal backward signal FFT buffer");
        let signal_second = signal_fft
            .second
            .as_ref()
            .expect("checked Metal backward signal FFT scratch buffer");
        let filter_first = filter_fft
            .first
            .as_ref()
            .expect("checked Metal backward filter FFT buffer");
        let filter_second = filter_fft
            .second
            .as_ref()
            .expect("checked Metal backward filter FFT scratch buffer");
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_identity(
            encoder.as_ref(),
            resident_upstream,
            upstream_buffer,
            elements,
        )?;
        self.encode_apply_gate(
            encoder.as_ref(),
            &cache.mixed,
            &cache.gated_projection,
            gated_mixed,
            channels_u32,
            channels_u32 * 2,
            channels_u32,
            elements_u32,
        )?;
        let (output_positive_buffer, output_negative_buffer, output_scale_buffer) =
            if let Some(updates) = updates {
                (
                    &*updates.output.positive,
                    &*updates.output.negative,
                    &*updates.output.scales,
                )
            } else {
                let positive = positive.expect("checked Metal positive codes");
                let negative = negative.expect("checked Metal negative codes");
                let scales = scales.expect("checked Metal scales");
                // SAFETY: immutable host codes fit the checked shared buffers.
                unsafe {
                    positive
                        .contents()
                        .cast::<u64>()
                        .as_ptr()
                        .copy_from_nonoverlapping(output_positive.as_ptr(), output_positive.len());
                    negative
                        .contents()
                        .cast::<u64>()
                        .as_ptr()
                        .copy_from_nonoverlapping(output_negative.as_ptr(), output_negative.len());
                    scales
                        .contents()
                        .cast::<f32>()
                        .as_ptr()
                        .copy_from_nonoverlapping(output_scales.as_ptr(), output_scales.len());
                }
                (&**positive, &**negative, &**scales)
            };
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.ternary_ste_weight_backward_pipeline,
            &[
                gated_mixed,
                upstream_buffer,
                output_scale_buffer,
                output_weight_gradient,
            ],
            &[rows_u32, channels_u32, channels_u32],
            channels * channels,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.ternary_input_backward_pipeline,
            &[
                upstream_buffer,
                output_positive_buffer,
                output_negative_buffer,
                output_scale_buffer,
                output_input_gradient,
            ],
            &[rows_u32, channels_u32, channels_u32],
            elements,
        )?;
        self.encode_hyena_gate_backward(
            encoder.as_ref(),
            &cache.mixed,
            &cache.gated_projection,
            output_input_gradient,
            signal_gradient,
            projection_gradient,
            channels_u32,
            elements_u32,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.extract_projection_signal_pipeline,
            &[&cache.gated_projection, signal],
            &[channels_u32, elements_u32],
            elements,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.pack_reverse_gradient_pipeline,
            &[signal_gradient, signal_first],
            &[time_u32, channels_u32, fft_len_u32, signal_fft_elements_u32],
            signal_fft_elements,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.pack_filter_pipeline,
            &[filter_buffer, filter_first],
            &[
                channels_u32,
                kernel_u32,
                fft_len_u32,
                filter_fft_elements_u32,
            ],
            filter_fft_elements,
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
                &[fft_len_u32, transform_count],
            )?;
            let mut source_is_first = false;
            for stage in 1..=fft_plan.stages {
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
                    &[fft_len_u32, transform_count, stage, u32::from(inverse)],
                )?;
                source_is_first = !source_is_first;
            }
            Ok(source_is_first)
        };
        let signal_source_is_first = run_fft(
            signal_first,
            signal_second,
            transforms_u32,
            signal_fft_elements,
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
            fft_len_u32,
            channels_u32,
            transforms_u32,
            signal_fft_elements,
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
            signal_fft_elements,
            &[fft_len_u32, transforms_u32],
        )?;
        let mut inverse_source_is_first = inverse_bitreversed_is_first;
        for stage in 1..=fft_plan.stages {
            let (source, destination) = if inverse_source_is_first {
                (signal_first, signal_second)
            } else {
                (signal_second, signal_first)
            };
            dispatch_two(
                &self.fft_stage_pipeline,
                source,
                destination,
                signal_fft_elements,
                &[fft_len_u32, transforms_u32, stage, 1],
            )?;
            inverse_source_is_first = !inverse_source_is_first;
        }
        let inverse = if inverse_source_is_first {
            signal_first
        } else {
            signal_second
        };
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.fft_extract_input_backward_pipeline,
            &[inverse, normalized_gradient],
            &[time_u32, channels_u32, fft_len_u32, elements_u32],
            elements,
        )?;
        if compute_filter_gradient {
            self.encode_elementwise_buffers(
                encoder.as_ref(),
                &self.causal_conv_filter_backward_pipeline,
                &[signal, signal_gradient, filter_gradient],
                &[batch_u32, time_u32, channels_u32, kernel_u32],
                filter_elements,
            )?;
        }
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.add_projection_signal_gradient_pipeline,
            &[projection_gradient, normalized_gradient],
            &[channels_u32, elements_u32],
            projected,
        )?;
        encoder.endEncoding();
        command.commit();
        if readback {
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                bail!("Metal cached block backward first command failed: {error}");
            }
        }
        let (input_positive_buffer, input_negative_buffer, input_scale_buffer) =
            if let Some(updates) = updates {
                (
                    &*updates.input.positive,
                    &*updates.input.negative,
                    &*updates.input.scales,
                )
            } else {
                let positive = positive.expect("checked Metal positive codes");
                let negative = negative.expect("checked Metal negative codes");
                let scales = scales.expect("checked Metal scales");
                // SAFETY: immutable host codes fit the checked shared buffers.
                unsafe {
                    positive
                        .contents()
                        .cast::<u64>()
                        .as_ptr()
                        .copy_from_nonoverlapping(input_positive.as_ptr(), input_positive.len());
                    negative
                        .contents()
                        .cast::<u64>()
                        .as_ptr()
                        .copy_from_nonoverlapping(input_negative.as_ptr(), input_negative.len());
                    scales
                        .contents()
                        .cast::<f32>()
                        .as_ptr()
                        .copy_from_nonoverlapping(input_scales.as_ptr(), input_scales.len());
                }
                (&**positive, &**negative, &**scales)
            };
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.ternary_ste_weight_backward_pipeline,
            &[
                &cache.normalized_input,
                projection_gradient,
                input_scale_buffer,
                input_weight_gradient,
            ],
            &[rows_u32, channels_u32, channels_u32 * 2],
            channels * channels * 2,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.ternary_input_backward_pipeline,
            &[
                projection_gradient,
                input_positive_buffer,
                input_negative_buffer,
                input_scale_buffer,
                normalized_gradient,
            ],
            &[rows_u32, channels_u32, channels_u32 * 2],
            elements,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.rms_norm_backward_pipeline,
            &[
                &cache.input,
                &cache.normalized_input,
                normalized_gradient,
                input_gradient,
            ],
            &[rows_u32, channels_u32],
            rows,
        )?;
        self.encode_residual_add(
            encoder.as_ref(),
            upstream_buffer,
            input_gradient,
            input_gradient,
            elements_u32,
        )?;
        self.encode_identity(
            encoder.as_ref(),
            input_gradient,
            resident_destination,
            elements,
        )?;
        if let Some(updates) = updates {
            self.encode_trainable_fp16_ternary_stateless_sgd(
                encoder.as_ref(),
                updates.output,
                output_weight_gradient,
                updates.learning_rate,
            )?;
            self.encode_trainable_fp16_ternary_stateless_sgd(
                encoder.as_ref(),
                updates.input,
                input_weight_gradient,
                updates.learning_rate,
            )?;
        }
        encoder.endEncoding();
        command.commit();
        if readback {
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                bail!("Metal cached block backward command failed: {error}");
            }
        }
        let read = |buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>, len: usize| {
            let mut values = vec![0.0; len];
            unsafe {
                values
                    .as_mut_ptr()
                    .copy_from_nonoverlapping(buffer.contents().cast::<f32>().as_ptr(), len);
            }
            values
        };
        let input_gradient = if readback {
            read(resident_destination, elements)
        } else {
            Vec::new()
        };
        let filter_gradient = if readback && compute_filter_gradient {
            read(filter_gradient, filter_elements)
        } else {
            Vec::new()
        };
        Ok(if updates.is_some() {
            CachedBlockBackwardResult::Updated(MetalHyenaBlockUpdatedBackward {
                input_gradient,
                filter_gradient,
            })
        } else {
            CachedBlockBackwardResult::Reference(MetalHyenaBlockBackward {
                input_gradient,
                input_projection_weight_gradient: read(
                    input_weight_gradient,
                    channels * channels * 2,
                ),
                output_projection_weight_gradient: read(
                    output_weight_gradient,
                    channels * channels,
                ),
                filter_gradient,
            })
        })
    }

    /// Uploads one FP16 stream to the first resident slot.
    #[allow(unsafe_code)]
    pub fn upload_resident_fp16_activations(
        &self,
        values: &crate::precision::Fp16Storage,
        rows: usize,
        width: usize,
    ) -> Result<ResidentFp16ActivationSlot> {
        use objc2_metal::MTLBuffer;

        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 activation shape overflow"))?;
        if rows == 0 || width == 0 || values.len() != elements {
            bail!("Metal resident FP16 activation shape mismatch");
        }
        self.reserve_fp16_activations(rows, width)?;
        let activations = self.fp16_activations.borrow();
        let destination = activations.buffer(ResidentFp16ActivationSlot::First)?;
        // SAFETY: `reserve_fp16_activations` established exact capacity.
        unsafe {
            destination
                .contents()
                .cast::<u16>()
                .as_ptr()
                .copy_from_nonoverlapping(values.as_bits().as_ptr(), elements);
        }
        Ok(ResidentFp16ActivationSlot::First)
    }

    /// Downloads a resident FP16 stream only at an explicit graph boundary.
    #[allow(unsafe_code)]
    pub fn download_resident_fp16_activations(
        &self,
        slot: ResidentFp16ActivationSlot,
        rows: usize,
        width: usize,
    ) -> Result<crate::precision::Fp16Storage> {
        use objc2_metal::MTLBuffer;

        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 activation shape overflow"))?;
        let activations = self.fp16_activations.borrow();
        if activations.capacity < elements * size_of::<u16>() {
            bail!("Metal resident FP16 activation download exceeds allocation");
        }
        let source = activations.buffer(slot)?;
        let mut values = vec![0_u16; elements];
        // SAFETY: allocation capacity and destination length were checked.
        unsafe {
            values
                .as_mut_ptr()
                .copy_from_nonoverlapping(source.contents().cast::<u16>().as_ptr(), elements);
        }
        Ok(crate::precision::Fp16Storage::from_bits(values))
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

    /// Uploads the terminal reverse-mode gradient to the independent resident
    /// pair. Later block-backward fusion writes each predecessor to `other()`.
    #[allow(unsafe_code)]
    pub fn upload_resident_gradient(
        &self,
        values: &[f32],
        rows: usize,
        width: usize,
    ) -> Result<ResidentGradientSlot> {
        use objc2_metal::MTLBuffer;

        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Metal resident gradient shape overflow"))?;
        if values.len() != elements || values.iter().any(|value| !value.is_finite()) {
            bail!("Metal resident gradient shape/value mismatch");
        }
        self.reserve_gradients(rows, width)?;
        let gradients = self.gradient_activations.borrow();
        let first = gradients
            .first
            .as_ref()
            .expect("checked Metal resident gradient buffer");
        // SAFETY: capacity and input length were validated before this copy.
        unsafe {
            first
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(values.as_ptr(), elements);
        }
        Ok(ResidentGradientSlot::First)
    }

    /// Downloads a reverse-mode gradient only at an explicit graph boundary.
    #[allow(unsafe_code)]
    pub fn download_resident_gradient(
        &self,
        slot: ResidentGradientSlot,
        rows: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        use objc2_metal::MTLBuffer;

        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Metal resident gradient shape overflow"))?;
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident gradient size overflow"))?;
        let gradients = self.gradient_activations.borrow();
        if gradients.capacity < bytes {
            bail!("Metal resident gradients were not initialized");
        }
        let buffer = match slot {
            ResidentGradientSlot::First => gradients.first.as_ref(),
            ResidentGradientSlot::Second => gradients.second.as_ref(),
        }
        .expect("checked Metal resident gradient buffer");
        let mut result = vec![0.0; elements];
        // SAFETY: allocation capacity and destination length were checked.
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

    /// Resident output projection using trainable packed weights directly.
    /// Unlike [`Self::resident_output_projection`], this path neither uploads
    /// codes nor scales from the CPU, so it remains valid after GPU updates.
    #[allow(unsafe_code)]
    pub fn resident_output_projection_trainable(
        &self,
        residual_slot: ResidentActivationSlot,
        rows: usize,
        width: usize,
        weights: &ResidentTrainableFp16TernaryWeights,
    ) -> Result<ResidentActivationSlot> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        if weights.in_features != width || weights.out_features != width {
            bail!("Metal resident trainable output weight shape mismatch");
        }
        let shape = TernaryLinearShape::new(rows, width, width)?;
        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Metal resident output shape overflow"))?;
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident output size overflow"))?;
        self.reserve_activations(rows, width)?;
        self.ternary_buffers
            .borrow_mut()
            .ensure(&self.device, bytes, 0, 0, bytes)?;
        let activations = self.activations.borrow();
        let ternary = self.ternary_buffers.borrow();
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
        let projected = ternary.output.as_ref().expect("checked ternary output");
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
            weights.positive.as_ref(),
            weights.negative.as_ref(),
            weights.scales.as_ref(),
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
        // The training queue is in-order. Its next consumer establishes the
        // completion boundary, so waiting here only serializes CPU submission.
        Ok(residual_slot.other())
    }

    /// Applies a square resident ternary projection without a residual add.
    /// This is the MTP-head forward primitive: the hidden stream stays in its
    /// source slot and the head output occupies the other activation slot.
    #[allow(unsafe_code)]
    pub fn resident_ternary_head_forward_trainable(
        &self,
        input_slot: ResidentActivationSlot,
        rows: usize,
        width: usize,
        weights: &ResidentTrainableFp16TernaryWeights,
    ) -> Result<ResidentActivationSlot> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        if weights.in_features != width || weights.out_features != width {
            bail!("Metal resident MTP-head weight shape mismatch");
        }
        let shape = TernaryLinearShape::new(rows, width, width)?;
        self.reserve_activations(rows, width)?;
        let activations = self.activations.borrow();
        let input = match input_slot {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked resident MTP-head input");
        let output = match input_slot.other() {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked resident MTP-head output");
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal resident MTP-head command buffer allocation failed")
        })?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal resident MTP-head encoder allocation failed"))?;
        self.encode_ternary(
            encoder.as_ref(),
            &self.ternary_pipeline,
            input,
            weights.positive.as_ref(),
            weights.negative.as_ref(),
            weights.scales.as_ref(),
            output,
            shape,
            false,
        )?;
        encoder.endEncoding();
        command.commit();
        // Streamed cross-entropy consumes this slot on the same in-order
        // queue and waits before its scalar loss is read back.
        Ok(input_slot.other())
    }

    /// Backpropagates one resident MTP head and refreshes its FP16 ternary
    /// master in the same submission. With `accumulate` the predecessor from
    /// the first head is added in-place, yielding the terminal Hyena gradient
    /// without a CPU gradient tensor or an extra D-wide workspace.
    #[allow(unsafe_code)]
    pub fn resident_ternary_head_backward_update(
        &self,
        input_slot: ResidentActivationSlot,
        output_gradient: ResidentGradientSlot,
        destination: ResidentGradientSlot,
        rows: usize,
        width: usize,
        weights: &ResidentTrainableFp16TernaryWeights,
        learning_rate: f32,
        accumulate: bool,
    ) -> Result<ResidentGradientSlot> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        if weights.in_features != width
            || weights.out_features != width
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
        {
            bail!("Metal resident MTP-head backward shape/value mismatch");
        }
        let elements = rows.checked_mul(width).ok_or_else(|| {
            anyhow::anyhow!("Metal resident MTP-head backward activation overflow")
        })?;
        let parameter_bytes = width
            .checked_mul(width)
            .and_then(|count| count.checked_mul(size_of::<f32>()))
            .ok_or_else(|| {
                anyhow::anyhow!("Metal resident MTP-head backward parameter overflow")
            })?;
        self.reserve_activations(rows, width)?;
        self.reserve_gradients(rows, width)?;
        self.backward_buffers
            .borrow_mut()
            .ensure(&self.device, 0, 0, 0, 0, parameter_bytes)?;
        let activations = self.activations.borrow();
        let input = match input_slot {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked resident MTP-head backward input");
        let gradients = self.gradient_activations.borrow();
        let output = match output_gradient {
            ResidentGradientSlot::First => gradients.first.as_ref(),
            ResidentGradientSlot::Second => gradients.second.as_ref(),
        }
        .expect("checked resident MTP-head output gradient");
        let destination_buffer = match destination {
            ResidentGradientSlot::First => gradients.first.as_ref(),
            ResidentGradientSlot::Second => gradients.second.as_ref(),
        }
        .expect("checked resident MTP-head input gradient");
        let previous = match destination.other() {
            ResidentGradientSlot::First => gradients.first.as_ref(),
            ResidentGradientSlot::Second => gradients.second.as_ref(),
        }
        .expect("checked resident MTP-head accumulated gradient");
        let backward = self.backward_buffers.borrow();
        let parameter_gradient = backward
            .parameter_gradient
            .as_ref()
            .expect("checked resident MTP-head parameter gradient");
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| anyhow::anyhow!("Metal resident MTP-head rows exceed u32"))?;
        let width_u32 = u32::try_from(width)
            .map_err(|_| anyhow::anyhow!("Metal resident MTP-head width exceeds u32"))?;
        let elements_u32 = u32::try_from(elements)
            .map_err(|_| anyhow::anyhow!("Metal resident MTP-head elements exceed u32"))?;
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal resident MTP-head backward command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal resident MTP-head backward encoder allocation failed")
        })?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.ternary_ste_weight_backward_pipeline,
            &[input, output, weights.scales.as_ref(), parameter_gradient],
            &[rows_u32, width_u32, width_u32],
            elements,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.ternary_input_backward_pipeline,
            &[
                output,
                weights.positive.as_ref(),
                weights.negative.as_ref(),
                weights.scales.as_ref(),
                destination_buffer,
            ],
            &[rows_u32, width_u32, width_u32],
            elements,
        )?;
        if accumulate {
            self.encode_residual_add(
                encoder.as_ref(),
                destination_buffer,
                previous,
                destination_buffer,
                elements_u32,
            )?;
        }
        self.encode_trainable_fp16_ternary_stateless_sgd(
            encoder.as_ref(),
            weights,
            parameter_gradient,
            learning_rate,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident MTP-head backward command failed: {error}");
        }
        Ok(destination)
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
        cache: Option<&ResidentHyenaBlockCache>,
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
        if let Some(cache) = cache
            && (cache.rows != rows || cache.channels != width)
        {
            bail!("Metal Hyena cache shape mismatch");
        }
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
        if let Some(cache) = cache {
            self.encode_identity(encoder.as_ref(), input, &cache.input, rows * width)?;
            self.encode_rms_norm(
                encoder.as_ref(),
                input,
                &cache.normalized_input,
                rows,
                width,
            )?;
            self.encode_ternary(
                encoder.as_ref(),
                &self.ternary_pipeline,
                &cache.normalized_input,
                positive_buffer,
                negative_buffer,
                scale_buffer,
                projected,
                shape,
                false,
            )?;
            self.encode_tanh_gate(
                encoder.as_ref(),
                &self.tanh_gate_pipeline,
                projected,
                &cache.gated_projection,
                rows,
                width,
            )?;
        } else {
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
            self.encode_tanh_gate(
                encoder.as_ref(),
                &self.tanh_gate_pipeline,
                projected,
                gated,
                rows,
                width,
            )?;
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident input command failed: {error}");
        }
        Ok(())
    }

    /// Resident fused RMSNorm/input projection using persistent trainable
    /// ternary state. Codes and scales are consumed directly from Metal.
    #[allow(unsafe_code)]
    pub fn resident_input_projection_trainable(
        &self,
        slot: ResidentActivationSlot,
        rows: usize,
        width: usize,
        weights: &ResidentTrainableFp16TernaryWeights,
        cache: Option<&ResidentHyenaBlockCache>,
    ) -> Result<()> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let out_width = width
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal resident input width overflow"))?;
        if weights.in_features != width || weights.out_features != out_width {
            bail!("Metal resident trainable input weight shape mismatch");
        }
        let shape = TernaryLinearShape::new(rows, width, out_width)?;
        let input_bytes = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal resident input size overflow"))?;
        let output_bytes = rows
            .checked_mul(out_width)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("Metal resident projection size overflow"))?;
        self.reserve_activations(rows, width)?;
        self.ternary_buffers
            .borrow_mut()
            .ensure(&self.device, input_bytes, 0, 0, output_bytes)?;
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
        let projected = ternary.output.as_ref().expect("checked projection output");
        let gated = gates.output.as_ref().expect("checked gate output");
        if let Some(cache) = cache
            && (cache.rows != rows || cache.channels != width)
        {
            bail!("Metal Hyena cache shape mismatch");
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        if let Some(cache) = cache {
            self.encode_identity(encoder.as_ref(), input, &cache.input, rows * width)?;
            self.encode_rms_norm(
                encoder.as_ref(),
                input,
                &cache.normalized_input,
                rows,
                width,
            )?;
            self.encode_ternary(
                encoder.as_ref(),
                &self.ternary_pipeline,
                &cache.normalized_input,
                weights.positive.as_ref(),
                weights.negative.as_ref(),
                weights.scales.as_ref(),
                projected,
                shape,
                false,
            )?;
            self.encode_tanh_gate(
                encoder.as_ref(),
                &self.tanh_gate_pipeline,
                projected,
                &cache.gated_projection,
                rows,
                width,
            )?;
        } else {
            self.encode_ternary(
                encoder.as_ref(),
                &self.fused_rms_norm_ternary_pipeline,
                input,
                weights.positive.as_ref(),
                weights.negative.as_ref(),
                weights.scales.as_ref(),
                projected,
                shape,
                true,
            )?;
            self.encode_tanh_gate(
                encoder.as_ref(),
                &self.tanh_gate_pipeline,
                projected,
                gated,
                rows,
                width,
            )?;
        }
        encoder.endEncoding();
        command.commit();
        // The following resident mixer is ordered after this submission.
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
        plan: HyenaChunkPlan,
        cache: Option<&ResidentHyenaBlockCache>,
    ) -> Result<()> {
        self.resident_hyena_mixer_impl(
            slot,
            batch,
            time,
            channels,
            ResidentImplicitFilterSource::Host(filter),
            plan,
            cache,
            true,
        )
    }

    /// Same resident mixer, but consumes compact FP16 parameters owned by the
    /// persistent training state instead of uploading a CPU filter each pass.
    pub fn resident_hyena_mixer_trainable(
        &self,
        slot: ResidentActivationSlot,
        batch: usize,
        time: usize,
        channels: usize,
        filter: &ResidentImplicitFilterParameters,
        plan: HyenaChunkPlan,
        cache: Option<&ResidentHyenaBlockCache>,
    ) -> Result<()> {
        self.resident_hyena_mixer_impl(
            slot,
            batch,
            time,
            channels,
            ResidentImplicitFilterSource::Trainable(filter),
            plan,
            cache,
            false,
        )
    }

    #[allow(unsafe_code)]
    fn resident_hyena_mixer_impl(
        &self,
        slot: ResidentActivationSlot,
        batch: usize,
        time: usize,
        channels: usize,
        filter_source: ResidentImplicitFilterSource<'_>,
        plan: HyenaChunkPlan,
        cache: Option<&ResidentHyenaBlockCache>,
        wait_for_completion: bool,
    ) -> Result<()> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputePipelineState,
        };

        let shape = MetalDispatchShape::new(batch, time, channels)?;
        let plan = plan.for_sequence(time)?;
        let chunks = time.div_ceil(plan.chunk_len);
        let transforms = batch
            .checked_mul(chunks)
            .and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| anyhow::anyhow!("Metal resident transform shape overflow"))?;
        let signal_elements = transforms
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal resident signal FFT shape overflow"))?;
        let filter_elements = channels
            .checked_mul(plan.kernel_len)
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
        let (host_parameters, order) = match filter_source {
            ResidentImplicitFilterSource::Host(filter) => {
                let (freq, phase, decay, order) = filter.parameter_slices(channels)?;
                (Some((freq, phase, decay)), order)
            }
            ResidentImplicitFilterSource::Trainable(parameters) => {
                if parameters.channels != channels {
                    bail!("Metal resident implicit-filter channel mismatch");
                }
                (None, parameters.order)
            }
        };
        let parameter_bytes = host_parameters
            .as_ref()
            .and_then(|(freq, _, _)| freq.len().checked_mul(size_of::<f32>()))
            .unwrap_or(0);
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
        if parameter_bytes != 0 {
            self.implicit_filter_parameters
                .borrow_mut()
                .ensure(&self.device, parameter_bytes)?;
        }

        let fft_len = u32::try_from(plan.fft_len)
            .map_err(|_| anyhow::anyhow!("Metal resident FFT length exceeds u32"))?;
        let transforms_u32 = u32::try_from(transforms)
            .map_err(|_| anyhow::anyhow!("Metal resident transform count exceeds u32"))?;
        let channels_u32 = u32::try_from(channels)
            .map_err(|_| anyhow::anyhow!("Metal resident channel count exceeds u32"))?;
        let time_u32 =
            u32::try_from(time).map_err(|_| anyhow::anyhow!("Metal resident time exceeds u32"))?;
        let chunk_u32 = u32::try_from(plan.chunk_len)
            .map_err(|_| anyhow::anyhow!("Metal resident chunk length exceeds u32"))?;
        let kernel_u32 = u32::try_from(plan.kernel_len)
            .map_err(|_| anyhow::anyhow!("Metal resident kernel length exceeds u32"))?;
        let chunks_u32 = u32::try_from(chunks)
            .map_err(|_| anyhow::anyhow!("Metal resident chunk count exceeds u32"))?;
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
        let scratch_gated = gates
            .output
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Metal resident gate is not initialized"))?;
        let gated: &objc2::runtime::ProtocolObject<dyn MTLBuffer> = if let Some(cache) = cache {
            if cache.rows != batch * time || cache.channels != channels {
                bail!("Metal Hyena cache shape mismatch");
            }
            &cache.gated_projection
        } else {
            scratch_gated
        };
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
        let (freq_buffer, phase_buffer, decay_buffer) = match filter_source {
            ResidentImplicitFilterSource::Host(_) => (
                &**parameters
                    .freq
                    .as_ref()
                    .expect("checked resident frequency buffer"),
                &**parameters
                    .phase
                    .as_ref()
                    .expect("checked resident phase buffer"),
                &**parameters
                    .decay
                    .as_ref()
                    .expect("checked resident decay buffer"),
            ),
            ResidentImplicitFilterSource::Trainable(parameters) => (
                &*parameters.freq.parameters,
                &*parameters.phase.parameters,
                &*parameters.decay.parameters,
            ),
        };
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
            if let Some((freq, phase, decay)) = host_parameters {
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
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_pack_overlap_save(
            encoder.as_ref(),
            gated,
            signal_first,
            [
                time_u32,
                channels_u32,
                channels_u32
                    .checked_mul(2)
                    .ok_or_else(|| anyhow::anyhow!("Metal resident stride exceeds u32"))?,
                0,
                chunk_u32,
                kernel_u32,
                fft_len,
                chunks_u32,
                u32::try_from(signal_elements)
                    .map_err(|_| anyhow::anyhow!("Metal resident FFT elements exceed u32"))?,
            ],
            signal_elements,
        )?;
        match filter_source {
            ResidentImplicitFilterSource::Host(_) => self.encode_implicit_filter(
                encoder.as_ref(),
                freq_buffer,
                phase_buffer,
                decay_buffer,
                filter_first,
                kernel_u32,
                time_u32,
                order_u32,
                fft_len,
                filter_elements_u32,
            )?,
            ResidentImplicitFilterSource::Trainable(_) => self.encode_implicit_filter_fp16(
                encoder.as_ref(),
                freq_buffer,
                phase_buffer,
                decay_buffer,
                filter_first,
                kernel_u32,
                time_u32,
                order_u32,
                fft_len,
                filter_elements_u32,
            )?,
        }
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
        self.encode_extract_overlap_save(
            encoder.as_ref(),
            inverse,
            mixed,
            [
                time_u32,
                channels_u32,
                chunk_u32,
                kernel_u32,
                fft_len,
                chunks_u32,
                elements_u32,
            ],
            shape.elements(),
        )?;
        if let Some(cache) = cache {
            self.encode_identity(encoder.as_ref(), mixed, &cache.mixed, shape.elements())?;
        }
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
        if wait_for_completion {
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                bail!("Metal resident Hyena mixer failed: {error}");
            }
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

    /// Runs both packed ternary backward kernels using the runtime's grow-only
    /// forward and backward workspaces. Results cross the host boundary only
    /// because this is still a numerical-reference API.
    #[allow(unsafe_code)]
    pub fn ternary_linear_backward_reference(
        &self,
        input: &[f32],
        output_gradient: &[f32],
        positive: &[u64],
        negative: &[u64],
        scales: &[f32],
        shape: TernaryLinearShape,
    ) -> Result<MetalTernaryLinearBackward> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let input_len = shape
            .rows
            .checked_mul(shape.in_features)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary backward input shape overflow"))?;
        let output_len = shape
            .rows
            .checked_mul(shape.out_features)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary backward output shape overflow"))?;
        let weight_len = shape
            .in_features
            .checked_mul(shape.out_features)
            .ok_or_else(|| anyhow::anyhow!("Metal ternary backward weight shape overflow"))?;
        let packed_words = shape.packed_words()?;
        if input.len() != input_len
            || output_gradient.len() != output_len
            || positive.len() != packed_words
            || negative.len() != packed_words
            || scales.len() != shape.out_features
            || input
                .iter()
                .chain(output_gradient)
                .chain(scales)
                .any(|value| !value.is_finite())
        {
            bail!("Metal ternary backward shape/value mismatch");
        }
        let input_bytes = input_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal ternary backward input bytes overflow"))?;
        let output_bytes = output_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal ternary backward output bytes overflow"))?;
        let packed_bytes = packed_words
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("Metal ternary backward code bytes overflow"))?;
        let scale_bytes = scales
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal ternary backward scale bytes overflow"))?;
        let weight_bytes = weight_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal ternary backward weight bytes overflow"))?;
        let mut buffers = self.ternary_buffers.borrow_mut();
        buffers.ensure(&self.device, input_bytes, packed_bytes, scale_bytes, 0)?;
        let input_buffer = buffers.input.as_ref().expect("checked Metal ternary input");
        let positive_buffer = buffers
            .positive
            .as_ref()
            .expect("checked Metal positive codes");
        let negative_buffer = buffers
            .negative
            .as_ref()
            .expect("checked Metal negative codes");
        let scale_buffer = buffers
            .scales
            .as_ref()
            .expect("checked Metal ternary scales");
        let mut backward = self.backward_buffers.borrow_mut();
        backward.ensure(&self.device, 0, 0, output_bytes, input_bytes, weight_bytes)?;
        let output_gradient_buffer = backward
            .output_gradient
            .as_ref()
            .expect("checked Metal ternary output gradient");
        let input_gradient = backward
            .input_gradient
            .as_ref()
            .expect("checked Metal ternary input gradient");
        let weight_gradient = backward
            .parameter_gradient
            .as_ref()
            .expect("checked Metal ternary weight gradient");
        // SAFETY: every shared allocation has exactly the validated source
        // capacity and no command is submitted until these copies complete.
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
            output_gradient_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(output_gradient.as_ptr(), output_len);
        }
        let rows =
            u32::try_from(shape.rows).map_err(|_| anyhow::anyhow!("Metal rows exceed u32"))?;
        let in_features = u32::try_from(shape.in_features)
            .map_err(|_| anyhow::anyhow!("Metal input width exceeds u32"))?;
        let out_features = u32::try_from(shape.out_features)
            .map_err(|_| anyhow::anyhow!("Metal output width exceeds u32"))?;
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.ternary_input_backward_pipeline,
            &[
                output_gradient_buffer,
                positive_buffer,
                negative_buffer,
                scale_buffer,
                input_gradient,
            ],
            &[rows, in_features, out_features],
            input_len,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.ternary_ste_weight_backward_pipeline,
            &[
                input_buffer,
                output_gradient_buffer,
                scale_buffer,
                weight_gradient,
            ],
            &[rows, in_features, out_features],
            weight_len,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal ternary backward command failed: {error}");
        }
        let mut input_gradient_result = vec![0.0; input_len];
        let mut weight_gradient_result = vec![0.0; weight_len];
        // SAFETY: command completion makes both shared result buffers readable.
        unsafe {
            input_gradient_result.as_mut_ptr().copy_from_nonoverlapping(
                input_gradient.contents().cast::<f32>().as_ptr(),
                input_len,
            );
            weight_gradient_result
                .as_mut_ptr()
                .copy_from_nonoverlapping(
                    weight_gradient.contents().cast::<f32>().as_ptr(),
                    weight_len,
                );
        }
        Ok(MetalTernaryLinearBackward {
            input_gradient: input_gradient_result,
            latent_weight_gradient: weight_gradient_result,
        })
    }

    /// Exact direct bounded-convolution backward on Metal. This intentionally
    /// mirrors the CPU reference before replacing its reductions with FFT
    /// adjoints, so the training graph has a stable causal contract.
    #[allow(unsafe_code)]
    pub fn causal_chunked_conv_backward_reference(
        &self,
        input: &[f32],
        filter: &[f32],
        output_gradient: &[f32],
        batch: usize,
        time: usize,
        channels: usize,
        plan: HyenaChunkPlan,
    ) -> Result<MetalCausalConvBackward> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let plan = plan.for_sequence(time)?;
        let elements = batch
            .checked_mul(time)
            .and_then(|rows| rows.checked_mul(channels))
            .ok_or_else(|| anyhow::anyhow!("Metal causal backward shape overflow"))?;
        let filter_elements = channels
            .checked_mul(plan.kernel_len)
            .ok_or_else(|| anyhow::anyhow!("Metal causal backward filter overflow"))?;
        if batch == 0
            || channels == 0
            || channels > 256
            || input.len() != elements
            || filter.len() != filter_elements
            || output_gradient.len() != elements
            || input
                .iter()
                .chain(filter)
                .chain(output_gradient)
                .any(|value| !value.is_finite())
        {
            bail!("Metal causal backward shape/value mismatch");
        }
        let element_bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal causal backward activation bytes overflow"))?;
        let filter_bytes = filter_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal causal backward filter bytes overflow"))?;
        let mut buffers = self.backward_buffers.borrow_mut();
        buffers.ensure(
            &self.device,
            element_bytes,
            filter_bytes,
            element_bytes,
            element_bytes,
            filter_bytes,
        )?;
        let input_buffer = buffers.source.as_ref().expect("checked Metal causal input");
        let filter_buffer = buffers
            .auxiliary
            .as_ref()
            .expect("checked Metal causal filter");
        let output_gradient_buffer = buffers
            .output_gradient
            .as_ref()
            .expect("checked Metal causal output gradient");
        let input_gradient_buffer = buffers
            .input_gradient
            .as_ref()
            .expect("checked Metal causal input gradient");
        let filter_gradient_buffer = buffers
            .parameter_gradient
            .as_ref()
            .expect("checked Metal causal filter gradient");
        // SAFETY: each shared allocation has exactly its validated slice size.
        unsafe {
            input_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(input.as_ptr(), elements);
            filter_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(filter.as_ptr(), filter_elements);
            output_gradient_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(output_gradient.as_ptr(), elements);
        }
        let batch = u32::try_from(batch).map_err(|_| anyhow::anyhow!("Metal batch exceeds u32"))?;
        let time = u32::try_from(time).map_err(|_| anyhow::anyhow!("Metal time exceeds u32"))?;
        let channels =
            u32::try_from(channels).map_err(|_| anyhow::anyhow!("Metal channels exceed u32"))?;
        let kernel_len = u32::try_from(plan.kernel_len)
            .map_err(|_| anyhow::anyhow!("Metal kernel length exceeds u32"))?;
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.causal_conv_input_backward_pipeline,
            &[filter_buffer, output_gradient_buffer, input_gradient_buffer],
            &[batch, time, channels, kernel_len],
            elements,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.causal_conv_filter_backward_pipeline,
            &[input_buffer, output_gradient_buffer, filter_gradient_buffer],
            &[batch, time, channels, kernel_len],
            filter_elements,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal causal backward command failed: {error}");
        }
        let mut input_gradient = vec![0.0; elements];
        let mut filter_gradient = vec![0.0; filter_elements];
        // SAFETY: completion makes both exact-size shared outputs readable.
        unsafe {
            input_gradient.as_mut_ptr().copy_from_nonoverlapping(
                input_gradient_buffer.contents().cast::<f32>().as_ptr(),
                elements,
            );
            filter_gradient.as_mut_ptr().copy_from_nonoverlapping(
                filter_gradient_buffer.contents().cast::<f32>().as_ptr(),
                filter_elements,
            );
        }
        Ok(MetalCausalConvBackward {
            input_gradient,
            filter_gradient,
        })
    }

    /// Exact row-wise RMSNorm backward reference on Metal. The normalized
    /// input is caller-owned cache from the matching forward block.
    #[allow(unsafe_code)]
    pub fn rms_norm_backward_reference(
        &self,
        input: &[f32],
        normalized: &[f32],
        output_gradient: &[f32],
        rows: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal RMSNorm backward shape overflow"))?;
        if rows == 0
            || channels == 0
            || input.len() != elements
            || normalized.len() != elements
            || output_gradient.len() != elements
            || input
                .iter()
                .chain(normalized)
                .chain(output_gradient)
                .any(|value| !value.is_finite())
        {
            bail!("Metal RMSNorm backward shape/value mismatch");
        }
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal RMSNorm backward byte size overflow"))?;
        let mut buffers = self.backward_buffers.borrow_mut();
        buffers.ensure(&self.device, bytes, bytes, bytes, bytes, 0)?;
        let input_buffer = buffers
            .source
            .as_ref()
            .expect("checked Metal RMSNorm input");
        let normalized_buffer = buffers
            .auxiliary
            .as_ref()
            .expect("checked Metal RMSNorm normalized input");
        let output_gradient_buffer = buffers
            .output_gradient
            .as_ref()
            .expect("checked Metal RMSNorm output gradient");
        let input_gradient_buffer = buffers
            .input_gradient
            .as_ref()
            .expect("checked Metal RMSNorm input gradient");
        // SAFETY: each shared allocation has exactly the checked source size.
        unsafe {
            input_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(input.as_ptr(), elements);
            normalized_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(normalized.as_ptr(), elements);
            output_gradient_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(output_gradient.as_ptr(), elements);
        }
        let rows =
            u32::try_from(rows).map_err(|_| anyhow::anyhow!("Metal RMSNorm rows exceed u32"))?;
        let channels = u32::try_from(channels)
            .map_err(|_| anyhow::anyhow!("Metal RMSNorm channels exceed u32"))?;
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.rms_norm_backward_pipeline,
            &[
                input_buffer,
                normalized_buffer,
                output_gradient_buffer,
                input_gradient_buffer,
            ],
            &[rows, channels],
            rows as usize,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal RMSNorm backward command failed: {error}");
        }
        let mut input_gradient = vec![0.0; elements];
        // SAFETY: command completion makes the exact-size shared output readable.
        unsafe {
            input_gradient.as_mut_ptr().copy_from_nonoverlapping(
                input_gradient_buffer.contents().cast::<f32>().as_ptr(),
                elements,
            );
        }
        Ok(input_gradient)
    }

    /// Keeps the convolution-to-projection join on Metal: extracts the
    /// signal half of a gated projection and adds its derivative to a gate
    /// derivative in one command buffer. This is the layout boundary used by
    /// the resident block-backward graph.
    #[allow(unsafe_code)]
    pub fn projection_signal_backward_reference(
        &self,
        gated_projection: &[f32],
        gate_gradient: &[f32],
        signal_gradient: &[f32],
        rows: usize,
        channels: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal projection join shape overflow"))?;
        let projected = elements
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal projection join width overflow"))?;
        if rows == 0
            || channels == 0
            || gated_projection.len() != projected
            || gate_gradient.len() != projected
            || signal_gradient.len() != elements
            || gated_projection
                .iter()
                .chain(gate_gradient)
                .chain(signal_gradient)
                .any(|value| !value.is_finite())
        {
            bail!("Metal projection join shape/value mismatch");
        }
        let activation_bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal projection join activation overflow"))?;
        let projected_bytes = projected
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal projection join bytes overflow"))?;
        let mut buffers = self.backward_buffers.borrow_mut();
        buffers.ensure(
            &self.device,
            projected_bytes,
            activation_bytes,
            projected_bytes,
            activation_bytes,
            0,
        )?;
        let projection = buffers.source.as_ref().expect("checked Metal projection");
        let signal_gradient_buffer = buffers
            .auxiliary
            .as_ref()
            .expect("checked Metal signal gradient");
        let projection_gradient = buffers
            .output_gradient
            .as_ref()
            .expect("checked Metal projection gradient");
        let signal = buffers
            .input_gradient
            .as_ref()
            .expect("checked Metal signal");
        // SAFETY: grow-only capacities exactly cover the checked source slices.
        unsafe {
            projection
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(gated_projection.as_ptr(), projected);
            signal_gradient_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(signal_gradient.as_ptr(), elements);
            projection_gradient
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(gate_gradient.as_ptr(), projected);
        }
        let channels =
            u32::try_from(channels).map_err(|_| anyhow::anyhow!("Metal channels exceed u32"))?;
        let elements =
            u32::try_from(elements).map_err(|_| anyhow::anyhow!("Metal elements exceed u32"))?;
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.extract_projection_signal_pipeline,
            &[projection, signal],
            &[channels, elements],
            elements as usize,
        )?;
        self.encode_elementwise_buffers(
            encoder.as_ref(),
            &self.add_projection_signal_gradient_pipeline,
            &[projection_gradient, signal_gradient_buffer],
            &[channels, elements],
            elements as usize,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal projection join command failed: {error}");
        }
        let mut signal_result = vec![0.0; elements as usize];
        let mut projection_gradient_result = vec![0.0; projected];
        // SAFETY: command completion makes both shared results readable.
        unsafe {
            signal_result.as_mut_ptr().copy_from_nonoverlapping(
                signal.contents().cast::<f32>().as_ptr(),
                signal_result.len(),
            );
            projection_gradient_result
                .as_mut_ptr()
                .copy_from_nonoverlapping(
                    projection_gradient.contents().cast::<f32>().as_ptr(),
                    projection_gradient_result.len(),
                );
        }
        Ok((signal_result, projection_gradient_result))
    }

    /// Executes the packed ternary projection with FP16 input, scales, and
    /// output buffers. Dot products accumulate in FP32 inside the shader.
    #[allow(unsafe_code)]
    pub fn ternary_linear_forward_fp16(
        &self,
        input: &crate::precision::Fp16Storage,
        positive: &[u64],
        negative: &[u64],
        scales: &crate::precision::Fp16Storage,
        shape: TernaryLinearShape,
    ) -> Result<crate::precision::Fp16Storage> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let input_len = shape
            .rows
            .checked_mul(shape.in_features)
            .ok_or_else(|| anyhow::anyhow!("FP16 ternary input shape overflow"))?;
        let output_len = shape
            .rows
            .checked_mul(shape.out_features)
            .ok_or_else(|| anyhow::anyhow!("FP16 ternary output shape overflow"))?;
        let packed_words = shape.packed_words()?;
        if input.len() != input_len
            || positive.len() != packed_words
            || negative.len() != packed_words
            || scales.len() != shape.out_features
        {
            bail!("Metal FP16 ternary projection shape mismatch");
        }
        let input_bytes = input.bytes();
        let packed_bytes = packed_words
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 ternary buffer byte size overflow"))?;
        let scale_bytes = scales.bytes();
        let output_bytes = output_len
            .checked_mul(size_of::<u16>())
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 output buffer byte size overflow"))?;
        let mut buffers = self.ternary_buffers.borrow_mut();
        buffers.ensure(
            &self.device,
            input_bytes,
            packed_bytes,
            scale_bytes,
            output_bytes,
        )?;
        let input_buffer = buffers
            .input
            .as_ref()
            .expect("checked Metal FP16 input buffer");
        let positive_buffer = buffers
            .positive
            .as_ref()
            .expect("checked Metal positive buffer");
        let negative_buffer = buffers
            .negative
            .as_ref()
            .expect("checked Metal negative buffer");
        let scale_buffer = buffers
            .scales
            .as_ref()
            .expect("checked Metal FP16 scale buffer");
        let output_buffer = buffers
            .output
            .as_ref()
            .expect("checked Metal FP16 output buffer");
        // SAFETY: each destination has at least the checked byte capacity and
        // host writes finish before GPU submission.
        unsafe {
            input_buffer
                .contents()
                .cast::<u16>()
                .as_ptr()
                .copy_from_nonoverlapping(input.as_bits().as_ptr(), input_len);
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
                .cast::<u16>()
                .as_ptr()
                .copy_from_nonoverlapping(scales.as_bits().as_ptr(), scales.len());
        }
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal FP16 ternary command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal FP16 ternary compute encoder allocation failed")
        })?;
        self.encode_ternary(
            encoder.as_ref(),
            &self.ternary_fp16_pipeline,
            input_buffer,
            positive_buffer,
            negative_buffer,
            scale_buffer,
            output_buffer,
            shape,
            false,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal FP16 ternary command failed: {error}");
        }
        let mut output = vec![0_u16; output_len];
        // SAFETY: command completion makes exactly `output_len` initialized
        // half values visible in the shared output buffer.
        unsafe {
            output.as_mut_ptr().copy_from_nonoverlapping(
                output_buffer.contents().cast::<u16>().as_ptr(),
                output_len,
            );
        }
        Ok(crate::precision::Fp16Storage::from_bits(output))
    }

    /// Uploads immutable packed ternary codes and FP16 scales once. The result
    /// can be reused for every microbatch until the optimizer refreshes codes.
    #[allow(unsafe_code)]
    pub fn upload_fp16_ternary_weights(
        &self,
        positive: &[u64],
        negative: &[u64],
        scales: &crate::precision::Fp16Storage,
        shape: TernaryLinearShape,
    ) -> Result<ResidentFp16TernaryWeights> {
        use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

        let packed_words = shape.packed_words()?;
        if positive.len() != packed_words
            || negative.len() != packed_words
            || scales.len() != shape.out_features
        {
            bail!("Metal resident FP16 ternary weight shape mismatch");
        }
        let packed_bytes = packed_words
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident ternary code size overflow"))?;
        let scale_bytes = scales.bytes();
        let options = MTLResourceOptions::StorageModeShared;
        let positive_buffer = self
            .device
            .newBufferWithLength_options(packed_bytes, options)
            .ok_or_else(|| anyhow::anyhow!("Metal resident positive code allocation failed"))?;
        let negative_buffer = self
            .device
            .newBufferWithLength_options(packed_bytes, options)
            .ok_or_else(|| anyhow::anyhow!("Metal resident negative code allocation failed"))?;
        let scale_buffer = self
            .device
            .newBufferWithLength_options(scale_bytes, options)
            .ok_or_else(|| anyhow::anyhow!("Metal resident FP16 scale allocation failed"))?;
        // SAFETY: buffer lengths exactly match validated source slice lengths.
        unsafe {
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
                .cast::<u16>()
                .as_ptr()
                .copy_from_nonoverlapping(scales.as_bits().as_ptr(), scales.len());
        }
        Ok(ResidentFp16TernaryWeights {
            positive: positive_buffer,
            negative: negative_buffer,
            scales: scale_buffer,
            shape,
        })
    }

    /// Uploads FP16 master parameters once for repeated stateless optimizer
    /// steps. Unlike Lion, this object carries no optimizer state beyond the
    /// model parameters themselves.
    #[allow(unsafe_code)]
    pub fn upload_resident_fp16_parameters(
        &self,
        values: &crate::precision::Fp16Storage,
    ) -> Result<ResidentFp16Parameters> {
        use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

        if values.is_empty() || (0..values.len()).any(|index| !values.get(index).is_finite()) {
            bail!("Metal resident FP16 parameter values are invalid");
        }
        let buffer = self
            .device
            .newBufferWithLength_options(values.bytes(), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Metal resident FP16 parameter allocation failed"))?;
        // SAFETY: allocation is exactly `values.bytes()` and the source has
        // `values.len()` initialized half values.
        unsafe {
            buffer
                .contents()
                .cast::<u16>()
                .as_ptr()
                .copy_from_nonoverlapping(values.as_bits().as_ptr(), values.len());
        }
        Ok(ResidentFp16Parameters {
            parameters: buffer,
            len: values.len(),
        })
    }

    /// Applies clipped stateless SGD to FP16 master parameters on Metal.
    /// Gradients are temporary FP32 values in a grow-only shared workspace;
    /// no momentum or variance allocation is created.
    #[allow(unsafe_code)]
    pub fn resident_fp16_stateless_sgd(
        &self,
        parameters: &ResidentFp16Parameters,
        gradient: &[f32],
        learning_rate: f32,
    ) -> Result<()> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        if gradient.len() != parameters.len
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || gradient.iter().any(|value| !value.is_finite())
        {
            bail!("Metal resident FP16 SGD shape/value mismatch");
        }
        let gradient_bytes = gradient
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident FP16 SGD gradient overflow"))?;
        let mut buffers = self.backward_buffers.borrow_mut();
        buffers.ensure(&self.device, 0, gradient_bytes, 0, 0, 0)?;
        let gradient_buffer = buffers
            .auxiliary
            .as_ref()
            .expect("checked Metal SGD gradient workspace");
        // SAFETY: the grow-only workspace has at least `gradient_bytes`.
        unsafe {
            gradient_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(gradient.as_ptr(), gradient.len());
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 SGD command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal FP16 SGD compute encoder allocation failed"))?;
        self.encode_clipped_sgd_fp16(
            encoder.as_ref(),
            parameters.parameters.as_ref(),
            gradient_buffer,
            learning_rate,
            parameters.len,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal FP16 SGD command failed: {error}");
        }
        Ok(())
    }

    /// Reads resident master parameters at an explicit validation or
    /// checkpoint boundary.
    #[allow(unsafe_code)]
    pub fn download_resident_fp16_parameters(
        &self,
        parameters: &ResidentFp16Parameters,
    ) -> Result<crate::precision::Fp16Storage> {
        use objc2_metal::MTLBuffer;

        let mut bits = vec![0_u16; parameters.len];
        // SAFETY: the retained allocation has exactly `len` half elements and
        // public optimizer commands wait for completion before returning.
        unsafe {
            bits.as_mut_ptr().copy_from_nonoverlapping(
                parameters.parameters.contents().cast::<u16>().as_ptr(),
                parameters.len,
            );
        }
        Ok(crate::precision::Fp16Storage::from_bits(bits))
    }

    /// Computes exact streamed cross-entropy on Metal using a resident FP16
    /// tied embedding. Each row scans vocabulary twice and emits only its
    /// D-wide gradient and scalar loss; `[rows, vocab]` logits never exist.
    #[allow(unsafe_code)]
    pub fn streamed_cross_entropy_fp16_resident(
        &self,
        head: &[f32],
        embedding: &ResidentFp16Parameters,
        tokens: &[u32],
        batch: usize,
        time: usize,
        channels: usize,
        vocab: usize,
        horizon: usize,
    ) -> Result<MetalStreamedCrossEntropy> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLSize,
        };

        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal cross-entropy row overflow"))?;
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal cross-entropy activation overflow"))?;
        let embedding_elements = vocab
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal cross-entropy embedding overflow"))?;
        if rows == 0
            || channels == 0
            || channels > 256
            || vocab == 0
            || horizon == 0
            || horizon >= time
            || head.len() != elements
            || tokens.len() != rows
            || embedding.len != embedding_elements
            || head.iter().any(|value| !value.is_finite())
            || tokens.iter().any(|&token| token as usize >= vocab)
        {
            bail!("Metal tiled streamed cross-entropy requires valid shapes and d_model <= 256");
        }
        let head_bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal cross-entropy head size overflow"))?;
        let token_bytes = rows
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("Metal cross-entropy token size overflow"))?;
        let loss_bytes = rows
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal cross-entropy loss size overflow"))?;
        let mut buffers = self.streamed_cross_entropy.borrow_mut();
        buffers.ensure(
            &self.device,
            head_bytes,
            token_bytes,
            head_bytes,
            loss_bytes,
        )?;
        let head_buffer = buffers
            .head
            .as_ref()
            .expect("checked Metal streamed head buffer");
        let token_buffer = buffers
            .tokens
            .as_ref()
            .expect("checked Metal streamed token buffer");
        let gradient_buffer = buffers
            .gradient
            .as_ref()
            .expect("checked Metal streamed gradient buffer");
        let loss_buffer = buffers
            .loss
            .as_ref()
            .expect("checked Metal streamed loss buffer");
        unsafe {
            head_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(head.as_ptr(), elements);
            token_buffer
                .contents()
                .cast::<u32>()
                .as_ptr()
                .copy_from_nonoverlapping(tokens.as_ptr(), rows);
        }
        let scalars = [
            u32::try_from(rows)
                .map_err(|_| anyhow::anyhow!("Metal cross-entropy rows exceed u32"))?,
            u32::try_from(time)
                .map_err(|_| anyhow::anyhow!("Metal cross-entropy time exceeds u32"))?,
            u32::try_from(channels)
                .map_err(|_| anyhow::anyhow!("Metal cross-entropy channels exceed u32"))?,
            u32::try_from(vocab)
                .map_err(|_| anyhow::anyhow!("Metal cross-entropy vocabulary exceeds u32"))?,
            u32::try_from(horizon)
                .map_err(|_| anyhow::anyhow!("Metal cross-entropy horizon exceeds u32"))?,
        ];
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal streamed cross-entropy command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal streamed cross-entropy encoder allocation failed")
        })?;
        encoder.setComputePipelineState(&self.streamed_cross_entropy_fp16_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(head_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(embedding.parameters.as_ref()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(token_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(gradient_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(loss_buffer), 0, 4);
            for (slot, scalar) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    slot + 5,
                );
            }
            let gradient_scale = 1.0f32;
            encoder.setBytes_length_atIndex(
                NonNull::from(&gradient_scale).cast::<c_void>(),
                size_of::<f32>(),
                10,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal streamed cross-entropy command failed: {error}");
        }
        let mut gradient = vec![0.0; elements];
        let mut losses = vec![0.0; rows];
        unsafe {
            gradient.as_mut_ptr().copy_from_nonoverlapping(
                gradient_buffer.contents().cast::<f32>().as_ptr(),
                elements,
            );
            losses
                .as_mut_ptr()
                .copy_from_nonoverlapping(loss_buffer.contents().cast::<f32>().as_ptr(), rows);
        }
        Ok(MetalStreamedCrossEntropy {
            loss_sum: losses.into_iter().sum(),
            token_count: batch * (time - horizon),
            head_gradient: gradient,
        })
    }

    /// Resident form of streamed tied-embedding cross-entropy. The head is
    /// read directly from the forward ping-pong stream and its exact `D`-wide
    /// derivative is written to the reverse ping-pong stream. Thus the graph
    /// never round-trips a terminal `[B*T,D]` gradient through CPU memory.
    #[allow(unsafe_code)]
    pub fn streamed_cross_entropy_fp16_from_activation(
        &self,
        head_slot: ResidentActivationSlot,
        embedding: &ResidentFp16Parameters,
        tokens: &[u32],
        batch: usize,
        time: usize,
        channels: usize,
        vocab: usize,
        horizon: usize,
    ) -> Result<MetalResidentCrossEntropy> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLSize,
        };

        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal resident cross-entropy row overflow"))?;
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal resident cross-entropy activation overflow"))?;
        let embedding_elements = vocab
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal resident cross-entropy embedding overflow"))?;
        if rows == 0
            || channels == 0
            || channels > 256
            || vocab == 0
            || horizon == 0
            || horizon >= time
            || tokens.len() != rows
            || embedding.len != embedding_elements
            || tokens.iter().any(|&token| token as usize >= vocab)
        {
            bail!("Metal tiled streamed cross-entropy requires valid shapes and d_model <= 256");
        }
        let activation_bytes = elements.checked_mul(size_of::<f32>()).ok_or_else(|| {
            anyhow::anyhow!("Metal resident cross-entropy activation size overflow")
        })?;
        let token_bytes = rows
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident cross-entropy token size overflow"))?;
        let loss_bytes = rows
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident cross-entropy loss size overflow"))?;
        self.reserve_activations(rows, channels)?;
        self.reserve_gradients(rows, channels)?;
        let mut ce_buffers = self.streamed_cross_entropy.borrow_mut();
        ce_buffers.ensure(
            &self.device,
            activation_bytes,
            token_bytes,
            activation_bytes,
            loss_bytes,
        )?;
        let tokens_buffer = ce_buffers.tokens.as_ref().expect("checked CE token buffer");
        let loss_buffer = ce_buffers.loss.as_ref().expect("checked CE loss buffer");
        unsafe {
            tokens_buffer
                .contents()
                .cast::<u32>()
                .as_ptr()
                .copy_from_nonoverlapping(tokens.as_ptr(), rows);
        }
        let activations = self.activations.borrow();
        let head = match head_slot {
            ResidentActivationSlot::First => activations.first.as_ref(),
            ResidentActivationSlot::Second => activations.second.as_ref(),
        }
        .expect("checked resident CE head activation");
        let gradients = self.gradient_activations.borrow();
        let gradient = gradients
            .first
            .as_ref()
            .expect("checked resident CE gradient");
        let scalars = [
            u32::try_from(rows).map_err(|_| anyhow::anyhow!("Metal CE rows exceed u32"))?,
            u32::try_from(time).map_err(|_| anyhow::anyhow!("Metal CE time exceeds u32"))?,
            u32::try_from(channels).map_err(|_| anyhow::anyhow!("Metal CE channels exceed u32"))?,
            u32::try_from(vocab).map_err(|_| anyhow::anyhow!("Metal CE vocabulary exceeds u32"))?,
            u32::try_from(horizon).map_err(|_| anyhow::anyhow!("Metal CE horizon exceeds u32"))?,
        ];
        let gradient_scale = 1.0f32 / (batch * (time - horizon)) as f32;
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!(
                "Metal resident streamed cross-entropy command buffer allocation failed"
            )
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal resident streamed cross-entropy encoder allocation failed")
        })?;
        encoder.setComputePipelineState(&self.streamed_cross_entropy_fp16_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(head), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(embedding.parameters.as_ref()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(tokens_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(gradient), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(loss_buffer), 0, 4);
            for (slot, scalar) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::from(scalar).cast::<c_void>(),
                    size_of::<u32>(),
                    slot + 5,
                );
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(&gradient_scale).cast::<c_void>(),
                size_of::<f32>(),
                10,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident streamed cross-entropy command failed: {error}");
        }
        let mut losses = vec![0.0; rows];
        unsafe {
            losses
                .as_mut_ptr()
                .copy_from_nonoverlapping(loss_buffer.contents().cast::<f32>().as_ptr(), rows);
        }
        Ok(MetalResidentCrossEntropy {
            loss_sum: losses.into_iter().sum(),
            token_count: batch * (time - horizon),
            gradient_slot: ResidentGradientSlot::First,
        })
    }

    /// Uploads the three compact implicit-filter vectors once for a resident
    /// training run. Their FP16 representation is the sole persistent master
    /// state; gradients are supplied only for the duration of an update.
    pub fn upload_resident_implicit_filter_parameters(
        &self,
        filter: &crate::hyena::ImplicitFilter,
        channels: usize,
    ) -> Result<ResidentImplicitFilterParameters> {
        let (freq, phase, decay, order) = filter.parameter_slices(channels)?;
        Ok(ResidentImplicitFilterParameters {
            freq: self.upload_resident_fp16_parameters(
                &crate::precision::Fp16Storage::from_f32(freq.iter().copied()),
            )?,
            phase: self.upload_resident_fp16_parameters(
                &crate::precision::Fp16Storage::from_f32(phase.iter().copied()),
            )?,
            decay: self.upload_resident_fp16_parameters(
                &crate::precision::Fp16Storage::from_f32(decay.iter().copied()),
            )?,
            channels,
            order,
        })
    }

    /// Stateless update for resident compact filter parameters. No momentum,
    /// variance, or CPU master copy is retained.
    pub fn resident_implicit_filter_stateless_sgd(
        &self,
        parameters: &ResidentImplicitFilterParameters,
        gradient: &crate::hyena::ImplicitFilterBackward,
        learning_rate: f32,
    ) -> Result<()> {
        let len = parameters
            .channels
            .checked_mul(parameters.order)
            .ok_or_else(|| anyhow::anyhow!("Metal resident implicit-filter parameter overflow"))?;
        if gradient.freq_gradient.len() != len
            || gradient.phase_gradient.len() != len
            || gradient.decay_gradient.len() != len
        {
            bail!("Metal resident implicit-filter gradient shape mismatch");
        }
        self.resident_fp16_stateless_sgd(&parameters.freq, &gradient.freq_gradient, learning_rate)?;
        self.resident_fp16_stateless_sgd(
            &parameters.phase,
            &gradient.phase_gradient,
            learning_rate,
        )?;
        self.resident_fp16_stateless_sgd(&parameters.decay, &gradient.decay_gradient, learning_rate)
    }

    /// Explicit checkpoint/validation readback for compact resident filter
    /// parameters. Normal training never calls this method.
    pub fn download_resident_implicit_filter_parameters(
        &self,
        parameters: &ResidentImplicitFilterParameters,
    ) -> Result<(
        crate::precision::Fp16Storage,
        crate::precision::Fp16Storage,
        crate::precision::Fp16Storage,
    )> {
        Ok((
            self.download_resident_fp16_parameters(&parameters.freq)?,
            self.download_resident_fp16_parameters(&parameters.phase)?,
            self.download_resident_fp16_parameters(&parameters.decay)?,
        ))
    }

    /// Uploads one trainable ternary projection and immediately builds its
    /// packed inference representation on Metal.
    #[allow(unsafe_code)]
    pub fn upload_trainable_fp16_ternary_weights(
        &self,
        master: &crate::precision::Fp16Storage,
        in_features: usize,
        out_features: usize,
        threshold_ratio: f32,
    ) -> Result<ResidentTrainableFp16TernaryWeights> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};

        let parameter_count = in_features
            .checked_mul(out_features)
            .ok_or_else(|| anyhow::anyhow!("Metal trainable ternary parameter overflow"))?;
        if in_features == 0
            || out_features == 0
            || master.len() != parameter_count
            || !threshold_ratio.is_finite()
            || threshold_ratio < 0.0
        {
            bail!("Metal trainable ternary weight shape/value mismatch");
        }
        let packed_bytes = parameter_count
            .div_ceil(64)
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("Metal trainable ternary code overflow"))?;
        let scale_bytes = out_features
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal trainable ternary scale overflow"))?;
        let shared = MTLResourceOptions::StorageModeShared;
        let allocate = |bytes, name: &str| {
            self.device
                .newBufferWithLength_options(bytes, shared)
                .ok_or_else(|| anyhow::anyhow!("Metal trainable ternary {name} allocation failed"))
        };
        let weights = ResidentTrainableFp16TernaryWeights {
            master: self.upload_resident_fp16_parameters(master)?,
            positive: allocate(packed_bytes, "positive codes")?,
            negative: allocate(packed_bytes, "negative codes")?,
            scales: allocate(scale_bytes, "scales")?,
            in_features,
            out_features,
            threshold_ratio,
        };
        self.refresh_trainable_fp16_ternary_weights(&weights)?;
        Ok(weights)
    }

    /// Updates a resident ternary master tensor and refreshes all derived
    /// inference state before returning. The gradient workspace is transient.
    pub fn resident_trainable_fp16_ternary_stateless_sgd(
        &self,
        weights: &ResidentTrainableFp16TernaryWeights,
        gradient: &[f32],
        learning_rate: f32,
    ) -> Result<()> {
        self.resident_fp16_stateless_sgd(&weights.master, gradient, learning_rate)?;
        self.refresh_trainable_fp16_ternary_weights(weights)
    }

    /// Reads trainable projection state only for validation/checkpointing.
    #[allow(unsafe_code)]
    pub fn download_trainable_fp16_ternary_weights(
        &self,
        weights: &ResidentTrainableFp16TernaryWeights,
    ) -> Result<(crate::precision::Fp16Storage, Vec<u64>, Vec<u64>, Vec<f32>)> {
        use objc2_metal::MTLBuffer;

        let parameter_count = weights
            .in_features
            .checked_mul(weights.out_features)
            .ok_or_else(|| anyhow::anyhow!("Metal trainable ternary parameter overflow"))?;
        let packed_words = parameter_count.div_ceil(64);
        let mut positive = vec![0_u64; packed_words];
        let mut negative = vec![0_u64; packed_words];
        let mut scales = vec![0.0; weights.out_features];
        // SAFETY: every retained buffer was allocated from these exact lengths
        // and all public update commands complete before returning.
        unsafe {
            positive.as_mut_ptr().copy_from_nonoverlapping(
                weights.positive.contents().cast::<u64>().as_ptr(),
                packed_words,
            );
            negative.as_mut_ptr().copy_from_nonoverlapping(
                weights.negative.contents().cast::<u64>().as_ptr(),
                packed_words,
            );
            scales.as_mut_ptr().copy_from_nonoverlapping(
                weights.scales.contents().cast::<f32>().as_ptr(),
                weights.out_features,
            );
        }
        Ok((
            self.download_resident_fp16_parameters(&weights.master)?,
            positive,
            negative,
            scales,
        ))
    }

    #[allow(unsafe_code)]
    fn refresh_trainable_fp16_ternary_weights(
        &self,
        weights: &ResidentTrainableFp16TernaryWeights,
    ) -> Result<()> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal trainable ternary refresh command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal trainable ternary refresh compute encoder allocation failed")
        })?;
        self.encode_refresh_trainable_fp16_ternary_weights(encoder.as_ref(), weights)?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal trainable ternary refresh command failed: {error}");
        }
        Ok(())
    }

    #[allow(unsafe_code)]
    fn encode_refresh_trainable_fp16_ternary_weights(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        weights: &ResidentTrainableFp16TernaryWeights,
    ) -> Result<()> {
        let parameter_count = weights
            .in_features
            .checked_mul(weights.out_features)
            .ok_or_else(|| anyhow::anyhow!("Metal trainable ternary parameter overflow"))?;
        let in_features = u32::try_from(weights.in_features)
            .map_err(|_| anyhow::anyhow!("Metal trainable ternary input width exceeds u32"))?;
        let out_features = u32::try_from(weights.out_features)
            .map_err(|_| anyhow::anyhow!("Metal trainable ternary output width exceeds u32"))?;
        self.encode_elementwise_buffers(
            encoder,
            &self.ternary_row_scales_fp16_pipeline,
            &[weights.master.parameters.as_ref(), weights.scales.as_ref()],
            &[in_features, out_features],
            weights.out_features,
        )?;
        self.encode_refresh_ternary_codes_fp16(encoder, weights, parameter_count)
    }

    #[allow(unsafe_code)]
    fn encode_trainable_fp16_ternary_stateless_sgd(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        weights: &ResidentTrainableFp16TernaryWeights,
        gradient: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        learning_rate: f32,
    ) -> Result<()> {
        let parameter_count = weights
            .in_features
            .checked_mul(weights.out_features)
            .ok_or_else(|| anyhow::anyhow!("Metal trainable ternary parameter overflow"))?;
        self.encode_clipped_sgd_fp16(
            encoder,
            weights.master.parameters.as_ref(),
            gradient,
            learning_rate,
            parameter_count,
        )?;
        self.encode_refresh_trainable_fp16_ternary_weights(encoder, weights)
    }

    /// FP16 projection using a previously uploaded immutable weight object.
    #[allow(unsafe_code)]
    pub fn ternary_linear_forward_fp16_resident(
        &self,
        input: &crate::precision::Fp16Storage,
        weights: &ResidentFp16TernaryWeights,
    ) -> Result<crate::precision::Fp16Storage> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let shape = weights.shape;
        let input_len = shape
            .rows
            .checked_mul(shape.in_features)
            .ok_or_else(|| anyhow::anyhow!("resident FP16 ternary input shape overflow"))?;
        let output_len = shape
            .rows
            .checked_mul(shape.out_features)
            .ok_or_else(|| anyhow::anyhow!("resident FP16 ternary output shape overflow"))?;
        if input.len() != input_len {
            bail!("Metal resident FP16 ternary input shape mismatch");
        }
        let input_bytes = input.bytes();
        let output_bytes = output_len
            .checked_mul(size_of::<u16>())
            .ok_or_else(|| anyhow::anyhow!("Metal resident FP16 output size overflow"))?;
        let mut buffers = self.ternary_buffers.borrow_mut();
        buffers.ensure(&self.device, input_bytes, 0, 0, output_bytes)?;
        let input_buffer = buffers
            .input
            .as_ref()
            .expect("checked resident FP16 input buffer");
        let output_buffer = buffers
            .output
            .as_ref()
            .expect("checked resident FP16 output buffer");
        // SAFETY: capacity was ensured and host write precedes command submit.
        unsafe {
            input_buffer
                .contents()
                .cast::<u16>()
                .as_ptr()
                .copy_from_nonoverlapping(input.as_bits().as_ptr(), input_len);
        }
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 ternary command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 ternary compute encoder allocation failed")
        })?;
        self.encode_ternary(
            encoder.as_ref(),
            &self.ternary_fp16_pipeline,
            input_buffer,
            weights.positive.as_ref(),
            weights.negative.as_ref(),
            weights.scales.as_ref(),
            output_buffer,
            shape,
            false,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident FP16 ternary command failed: {error}");
        }
        let mut output = vec![0_u16; output_len];
        // SAFETY: command completion makes the initialized shared output readable.
        unsafe {
            output.as_mut_ptr().copy_from_nonoverlapping(
                output_buffer.contents().cast::<u16>().as_ptr(),
                output_len,
            );
        }
        Ok(crate::precision::Fp16Storage::from_bits(output))
    }

    /// Applies a resident FP16 ternary projection between activation slots.
    /// Call `reserve_fp16_activations(rows, max(input_width, output_width))`
    /// before the initial upload, so capacity growth never invalidates a live
    /// source slot.
    pub fn resident_ternary_linear_fp16(
        &self,
        slot: ResidentFp16ActivationSlot,
        weights: &ResidentFp16TernaryWeights,
    ) -> Result<ResidentFp16ActivationSlot> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let output_bytes = weights
            .shape
            .rows
            .checked_mul(weights.shape.out_features)
            .and_then(|elements| elements.checked_mul(size_of::<u16>()))
            .ok_or_else(|| anyhow::anyhow!("resident FP16 projection output size overflow"))?;
        let activations = self.fp16_activations.borrow();
        if activations.capacity < output_bytes {
            bail!(
                "resident FP16 activation capacity is too small; reserve the maximum projection width before upload"
            );
        }
        let source = activations.buffer(slot)?;
        let destination = activations.buffer(slot.other())?;
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 projection command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 projection compute encoder allocation failed")
        })?;
        self.encode_ternary(
            encoder.as_ref(),
            &self.ternary_fp16_pipeline,
            source,
            weights.positive.as_ref(),
            weights.negative.as_ref(),
            weights.scales.as_ref(),
            destination,
            weights.shape,
            false,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident FP16 projection failed: {error}");
        }
        Ok(slot.other())
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
        self.encode_tanh_gate(
            encoder.as_ref(),
            &self.tanh_gate_pipeline,
            source,
            output,
            rows,
            channels,
        )?;
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

    /// Applies the FP16 Hyena gate while retaining both projection layouts on
    /// the GPU: `[rows, 2D]` in one slot becomes `[rows, 2D]` in the other.
    pub fn resident_tanh_gate_fp16(
        &self,
        slot: ResidentFp16ActivationSlot,
        rows: usize,
        channels: usize,
    ) -> Result<ResidentFp16ActivationSlot> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let bytes = rows
            .checked_mul(channels)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| elements.checked_mul(size_of::<u16>()))
            .ok_or_else(|| anyhow::anyhow!("resident FP16 gate shape overflow"))?;
        let activations = self.fp16_activations.borrow();
        if rows == 0 || channels == 0 || activations.capacity < bytes {
            bail!("resident FP16 gate activation capacity is too small");
        }
        let source = activations.buffer(slot)?;
        let destination = activations.buffer(slot.other())?;
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 gate command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 gate compute encoder allocation failed")
        })?;
        self.encode_tanh_gate(
            encoder.as_ref(),
            &self.tanh_gate_fp16_pipeline,
            source,
            destination,
            rows,
            channels,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident FP16 gate failed: {error}");
        }
        Ok(slot.other())
    }

    /// Multiplies a mixed `[rows,D]` stream by the gate half of a resident
    /// `[rows,2D]` projection, writing FP16 output to an explicit work slot.
    pub fn resident_apply_gate_fp16(
        &self,
        mixed_slot: ResidentFp16ActivationSlot,
        gated_projection_slot: ResidentFp16ActivationSlot,
        output_slot: ResidentFp16ActivationSlot,
        rows: usize,
        channels: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        if gated_projection_slot == output_slot {
            bail!("resident FP16 gate output may not overwrite its projection input");
        }
        let projection_bytes = rows
            .checked_mul(channels)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| elements.checked_mul(size_of::<u16>()))
            .ok_or_else(|| anyhow::anyhow!("resident FP16 apply-gate shape overflow"))?;
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("resident FP16 apply-gate shape overflow"))?;
        let activations = self.fp16_activations.borrow();
        if rows == 0 || channels == 0 || activations.capacity < projection_bytes {
            bail!("resident FP16 apply-gate activation capacity is too small");
        }
        let mixed = activations.buffer(mixed_slot)?;
        let projection = activations.buffer(gated_projection_slot)?;
        let output = activations.buffer(output_slot)?;
        let channels_u32 = u32::try_from(channels)
            .map_err(|_| anyhow::anyhow!("resident FP16 apply-gate channels exceed u32"))?;
        let elements_u32 = u32::try_from(elements)
            .map_err(|_| anyhow::anyhow!("resident FP16 apply-gate elements exceed u32"))?;
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 apply-gate command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 apply-gate compute encoder allocation failed")
        })?;
        self.encode_apply_gate_fp16(
            encoder.as_ref(),
            mixed,
            projection,
            output,
            channels_u32,
            channels_u32
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("resident FP16 stride exceeds u32"))?,
            channels_u32,
            elements_u32,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident FP16 apply-gate failed: {error}");
        }
        Ok(())
    }

    /// Adds two resident `[rows,D]` FP16 streams into an explicit output slot.
    pub fn resident_residual_add_fp16(
        &self,
        residual_slot: ResidentFp16ActivationSlot,
        update_slot: ResidentFp16ActivationSlot,
        output_slot: ResidentFp16ActivationSlot,
        rows: usize,
        channels: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("resident FP16 residual shape overflow"))?;
        let bytes = elements
            .checked_mul(size_of::<u16>())
            .ok_or_else(|| anyhow::anyhow!("resident FP16 residual size overflow"))?;
        let activations = self.fp16_activations.borrow();
        if rows == 0 || channels == 0 || activations.capacity < bytes {
            bail!("resident FP16 residual activation capacity is too small");
        }
        let residual = activations.buffer(residual_slot)?;
        let update = activations.buffer(update_slot)?;
        let output = activations.buffer(output_slot)?;
        let elements_u32 = u32::try_from(elements)
            .map_err(|_| anyhow::anyhow!("resident FP16 residual elements exceed u32"))?;
        let command = self.queue.commandBuffer().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 residual command buffer allocation failed")
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            anyhow::anyhow!("Metal resident FP16 residual compute encoder allocation failed")
        })?;
        self.encode_residual_add_fp16(encoder.as_ref(), residual, update, output, elements_u32)?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal resident FP16 residual add failed: {error}");
        }
        Ok(())
    }

    #[allow(unsafe_code)]
    fn encode_tanh_gate(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
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
        encoder.setComputePipelineState(pipeline);
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
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(elements);
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
    fn encode_pack_overlap_save(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        scalars: [u32; 9],
        elements: usize,
    ) -> Result<()> {
        self.encode_elementwise_buffers(
            encoder,
            &self.pack_overlap_save_pipeline,
            &[input, output],
            &scalars,
            elements,
        )
    }

    #[allow(unsafe_code)]
    fn encode_extract_overlap_save(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        scalars: [u32; 7],
        elements: usize,
    ) -> Result<()> {
        self.encode_elementwise_buffers(
            encoder,
            &self.extract_overlap_save_pipeline,
            &[input, output],
            &scalars,
            elements,
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
    fn encode_apply_gate_fp16(
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
            &self.apply_gate_fp16_pipeline,
            &[mixed, projection, output],
            &[channels, stride, gate_offset, elements],
            elements as usize,
        )
    }

    #[allow(unsafe_code)]
    fn encode_hyena_gate_backward(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        mixed: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        projection: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output_gradient: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        mixed_gradient: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        projection_gradient: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        channels: u32,
        elements: u32,
    ) -> Result<()> {
        self.encode_elementwise_buffers(
            encoder,
            &self.hyena_gate_backward_pipeline,
            &[
                mixed,
                projection,
                output_gradient,
                mixed_gradient,
                projection_gradient,
            ],
            &[channels, elements],
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
    fn encode_identity(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        elements: usize,
    ) -> Result<()> {
        let elements = u32::try_from(elements)
            .map_err(|_| anyhow::anyhow!("Metal identity elements exceed u32"))?;
        self.encode_elementwise_buffers(
            encoder,
            &self.identity_pipeline,
            &[input, output],
            &[elements],
            elements as usize,
        )
    }

    #[allow(unsafe_code)]
    fn encode_clipped_sgd_fp16(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        parameters: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        gradient: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        learning_rate: f32,
        elements: usize,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

        let elements = u32::try_from(elements)
            .map_err(|_| anyhow::anyhow!("Metal FP16 SGD elements exceed u32"))?;
        encoder.setComputePipelineState(&self.clipped_sgd_fp16_pipeline);
        // SAFETY: the buffers and scalar constants match the MSL ABI exactly.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(parameters), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gradient), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::from(&learning_rate).cast::<c_void>(),
                size_of::<f32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&elements).cast::<c_void>(),
                size_of::<u32>(),
                3,
            );
        }
        let width = self
            .clipped_sgd_fp16_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(elements as usize);
        if width == 0 {
            bail!("Metal FP16 SGD pipeline reported zero threads per threadgroup");
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
    fn encode_refresh_ternary_codes_fp16(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        weights: &ResidentTrainableFp16TernaryWeights,
        parameter_count: usize,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

        let in_features = u32::try_from(weights.in_features)
            .map_err(|_| anyhow::anyhow!("Metal ternary code-refresh input width exceeds u32"))?;
        let parameter_count = u32::try_from(parameter_count).map_err(|_| {
            anyhow::anyhow!("Metal ternary code-refresh parameter count exceeds u32")
        })?;
        let words = (parameter_count as usize).div_ceil(64);
        encoder.setComputePipelineState(&self.refresh_ternary_codes_fp16_pipeline);
        // SAFETY: buffer order and constants exactly match the code-refresh
        // MSL signature; `setBytes` copies scalar values into the encoder.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.master.parameters.as_ref()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.scales.as_ref()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weights.positive.as_ref()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(weights.negative.as_ref()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::from(&weights.threshold_ratio).cast::<c_void>(),
                size_of::<f32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&in_features).cast::<c_void>(),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameter_count).cast::<c_void>(),
                size_of::<u32>(),
                6,
            );
        }
        let width = self
            .refresh_ternary_codes_fp16_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(words);
        if width == 0 {
            bail!("Metal ternary code-refresh pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: words,
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
    fn encode_rms_norm(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        input: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        rows: usize,
        channels: usize,
    ) -> Result<()> {
        let rows =
            u32::try_from(rows).map_err(|_| anyhow::anyhow!("Metal RMSNorm rows exceed u32"))?;
        let channels = u32::try_from(channels)
            .map_err(|_| anyhow::anyhow!("Metal RMSNorm channels exceed u32"))?;
        self.encode_elementwise_buffers(
            encoder,
            &self.rms_norm_pipeline,
            &[input, output],
            &[rows, channels],
            rows as usize,
        )
    }

    #[allow(unsafe_code)]
    fn encode_residual_add_fp16(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        residual: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        update: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        elements: u32,
    ) -> Result<()> {
        self.encode_elementwise_buffers(
            encoder,
            &self.residual_add_fp16_pipeline,
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

    /// Runs the exact local Hyena gate backward kernel and reads back only its
    /// two gradients. It is a numerical-reference boundary for the upcoming
    /// resident training graph; all intermediate buffers stay on the runtime.
    #[allow(unsafe_code)]
    pub fn hyena_gate_backward_reference(
        &self,
        mixed: &[f32],
        gated_projection: &[f32],
        output_gradient: &[f32],
        rows: usize,
        channels: usize,
    ) -> Result<MetalHyenaGateBackward> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("Metal gate backward shape overflow"))?;
        let projected = elements
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Metal gate backward projection overflow"))?;
        if rows == 0
            || channels == 0
            || mixed.len() != elements
            || output_gradient.len() != elements
            || gated_projection.len() != projected
            || mixed
                .iter()
                .chain(gated_projection)
                .chain(output_gradient)
                .any(|value| !value.is_finite())
        {
            bail!("Metal gate backward shape/value mismatch");
        }
        let activation_bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal gate backward activation overflow"))?;
        let projection_bytes = projected
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal gate backward projection overflow"))?;
        self.reserve_activations(rows, channels)?;
        self.gate_buffers
            .borrow_mut()
            .ensure(&self.device, projection_bytes)?;
        self.hyena_output_buffer
            .borrow_mut()
            .ensure(&self.device, activation_bytes)?;
        let activations = self.activations.borrow();
        let gates = self.gate_buffers.borrow();
        let output = self.hyena_output_buffer.borrow();
        let mixed_buffer = activations
            .first
            .as_ref()
            .expect("checked Metal activation buffer");
        let mixed_gradient = activations
            .second
            .as_ref()
            .expect("checked Metal activation scratch");
        let projection_buffer = gates.input.as_ref().expect("checked Metal gate input");
        let projection_gradient = gates.output.as_ref().expect("checked Metal gate output");
        let upstream = output.buffer.as_ref().expect("checked Metal Hyena output");
        // SAFETY: allocations match the checked slices and commands start only
        // after these shared-buffer writes are complete.
        unsafe {
            mixed_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(mixed.as_ptr(), elements);
            projection_buffer
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(gated_projection.as_ptr(), projected);
            upstream
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(output_gradient.as_ptr(), elements);
            projection_gradient
                .contents()
                .cast::<f32>()
                .as_ptr()
                .write_bytes(0, projected);
        }
        let channels =
            u32::try_from(channels).map_err(|_| anyhow::anyhow!("Metal channels exceed u32"))?;
        let elements =
            u32::try_from(elements).map_err(|_| anyhow::anyhow!("Metal elements exceed u32"))?;
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        self.encode_hyena_gate_backward(
            encoder.as_ref(),
            mixed_buffer,
            projection_buffer,
            upstream,
            mixed_gradient,
            projection_gradient,
            channels,
            elements,
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal gate backward command failed: {error}");
        }
        let mut mixed_gradient_result = vec![0.0; elements as usize];
        let mut projection_gradient_result = vec![0.0; projected];
        // SAFETY: completion makes the shared outputs readable at their exact
        // checked element counts.
        unsafe {
            mixed_gradient_result.as_mut_ptr().copy_from_nonoverlapping(
                mixed_gradient.contents().cast::<f32>().as_ptr(),
                elements as usize,
            );
            projection_gradient_result
                .as_mut_ptr()
                .copy_from_nonoverlapping(
                    projection_gradient.contents().cast::<f32>().as_ptr(),
                    projected,
                );
        }
        Ok(MetalHyenaGateBackward {
            mixed_gradient: mixed_gradient_result,
            projection_gradient: projection_gradient_result,
        })
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
        sequence_len: u32,
        order: u32,
        fft_len: u32,
        elements: u32,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        encoder.setComputePipelineState(&self.implicit_filter_pipeline);
        // SAFETY: slots 0..8 exactly match ullis_generate_implicit_filter.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(freq), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(phase), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(decay), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            for (slot, scalar) in [time, sequence_len, order, fft_len, elements]
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
    fn encode_implicit_filter_fp16(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        freq: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        phase: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        decay: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        output: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        time: u32,
        sequence_len: u32,
        order: u32,
        fft_len: u32,
        elements: u32,
    ) -> Result<()> {
        use core::ffi::c_void;
        use core::ptr::NonNull;
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
        encoder.setComputePipelineState(&self.implicit_filter_fp16_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(freq), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(phase), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(decay), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            for (slot, scalar) in [time, sequence_len, order, fft_len, elements]
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
            .implicit_filter_fp16_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(elements as usize);
        if width == 0 {
            bail!("Metal FP16 implicit-filter pipeline reported zero threads per threadgroup");
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
        // SAFETY: slots 0..8 exactly match ullis_generate_implicit_filter.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(freq_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(phase_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(decay_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
            for (slot, scalar) in [time_u32, time_u32, order_u32, fft_len_u32, elements_u32]
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

    /// Runs exact bounded-receptive-field overlap-save convolution on Metal.
    /// The filter is generated only for `plan.kernel_len` positions, normalized
    /// against the full sequence length, and every FFT buffer is sized by the
    /// chunk plan rather than `time`.
    #[allow(unsafe_code)]
    pub fn causal_chunked_conv_implicit_strided_forward(
        &self,
        input: &[f32],
        filter: &crate::hyena::ImplicitFilter,
        batch: usize,
        time: usize,
        channels: usize,
        input_stride: usize,
        input_offset: usize,
        plan: HyenaChunkPlan,
    ) -> Result<Vec<f32>> {
        use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let plan = plan.for_sequence(time)?;
        let shape = MetalDispatchShape::new(batch, time, channels)?;
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena row overflow"))?;
        let minimum_input = rows
            .checked_sub(1)
            .and_then(|row| row.checked_mul(input_stride))
            .and_then(|start| start.checked_add(input_offset))
            .and_then(|start| start.checked_add(channels))
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena input overflow"))?;
        if input_stride < channels || input.len() < minimum_input {
            bail!("Metal chunked Hyena convolution shape mismatch");
        }
        let chunks = time.div_ceil(plan.chunk_len);
        let transforms = batch
            .checked_mul(chunks)
            .and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena transform overflow"))?;
        let signal_elements = transforms
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena FFT overflow"))?;
        let filter_elements = channels
            .checked_mul(plan.kernel_len)
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena filter overflow"))?;
        let filter_fft_elements = channels
            .checked_mul(plan.fft_len)
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena filter FFT overflow"))?;
        let input_bytes = input
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena input size overflow"))?;
        let signal_bytes = signal_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena signal size overflow"))?;
        let filter_bytes = filter_fft_elements
            .checked_mul(size_of::<Complex32>())
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena filter size overflow"))?;
        let output_bytes = shape
            .elements()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena output size overflow"))?;
        let (freq, phase, decay, order) = filter.parameter_slices(channels)?;
        let parameter_bytes = freq
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal chunked Hyena parameter overflow"))?;
        self.gate_buffers
            .borrow_mut()
            .ensure(&self.device, input_bytes)?;
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
        let as_u32 = |value: usize, name: &str| {
            u32::try_from(value)
                .map_err(|_| anyhow::anyhow!("Metal chunked Hyena {name} exceeds u32"))
        };
        let time_u32 = as_u32(time, "time")?;
        let channels_u32 = as_u32(channels, "channels")?;
        let stride_u32 = as_u32(input_stride, "stride")?;
        let offset_u32 = as_u32(input_offset, "offset")?;
        let chunk_u32 = as_u32(plan.chunk_len, "chunk length")?;
        let kernel_u32 = as_u32(plan.kernel_len, "kernel length")?;
        let fft_u32 = as_u32(plan.fft_len, "FFT length")?;
        let chunks_u32 = as_u32(chunks, "chunk count")?;
        let transforms_u32 = as_u32(transforms, "transform count")?;
        let signal_u32 = as_u32(signal_elements, "FFT elements")?;
        let filter_u32 = as_u32(filter_elements, "filter elements")?;
        let output_u32 = as_u32(shape.elements(), "output elements")?;
        let order_u32 = as_u32(order, "filter order")?;

        let gates = self.gate_buffers.borrow();
        let signal = self.fft_buffers.borrow();
        let filter_fft = self.filter_fft_buffers.borrow();
        let output = self.hyena_output_buffer.borrow();
        let parameters = self.implicit_filter_parameters.borrow();
        let source = gates.input.as_ref().expect("checked Metal chunked input");
        let signal_first = signal.first.as_ref().expect("checked Metal chunked signal");
        let signal_second = signal
            .second
            .as_ref()
            .expect("checked Metal chunked signal scratch");
        let filter_first = filter_fft
            .first
            .as_ref()
            .expect("checked Metal chunked filter");
        let filter_second = filter_fft
            .second
            .as_ref()
            .expect("checked Metal chunked filter scratch");
        let destination = output
            .buffer
            .as_ref()
            .expect("checked Metal chunked output");
        let freq_buffer = parameters
            .freq
            .as_ref()
            .expect("checked Metal chunked freq");
        let phase_buffer = parameters
            .phase
            .as_ref()
            .expect("checked Metal chunked phase");
        let decay_buffer = parameters
            .decay
            .as_ref()
            .expect("checked Metal chunked decay");
        unsafe {
            source
                .contents()
                .cast::<f32>()
                .as_ptr()
                .copy_from_nonoverlapping(input.as_ptr(), input.len());
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
        self.encode_pack_overlap_save(
            encoder.as_ref(),
            source,
            signal_first,
            [
                time_u32,
                channels_u32,
                stride_u32,
                offset_u32,
                chunk_u32,
                kernel_u32,
                fft_u32,
                chunks_u32,
                signal_u32,
            ],
            signal_elements,
        )?;
        self.encode_implicit_filter(
            encoder.as_ref(),
            freq_buffer,
            phase_buffer,
            decay_buffer,
            filter_first,
            kernel_u32,
            time_u32,
            order_u32,
            fft_u32,
            filter_u32,
        )?;
        let dispatch = |pipeline, input, output, total, scalars: &[u32]| {
            self.encode_fft_two_buffer(encoder.as_ref(), pipeline, input, output, total, scalars)
        };
        let run_fft = |first, second, transform_count, total, inverse| -> Result<bool> {
            dispatch(
                &self.fft_bitreverse_pipeline,
                first,
                second,
                total,
                &[fft_u32, transform_count],
            )?;
            let mut source_is_first = false;
            for stage in 1..=plan.stages {
                let (source, destination) = if source_is_first {
                    (first, second)
                } else {
                    (second, first)
                };
                dispatch(
                    &self.fft_stage_pipeline,
                    source,
                    destination,
                    total,
                    &[fft_u32, transform_count, stage, u32::from(inverse)],
                )?;
                source_is_first = !source_is_first;
            }
            Ok(source_is_first)
        };
        let signal_is_first = run_fft(
            signal_first,
            signal_second,
            transforms_u32,
            signal_elements,
            false,
        )?;
        let filter_is_first = run_fft(
            filter_first,
            filter_second,
            channels_u32,
            filter_fft_elements,
            false,
        )?;
        let signal_spectrum = if signal_is_first {
            signal_first
        } else {
            signal_second
        };
        let filter_spectrum = if filter_is_first {
            filter_first
        } else {
            filter_second
        };
        let product_is_first = !signal_is_first;
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
            fft_u32,
            channels_u32,
            transforms_u32,
            signal_elements,
        )?;
        let inverse_bitrev_is_first = !product_is_first;
        let inverse_bitrev = if inverse_bitrev_is_first {
            signal_first
        } else {
            signal_second
        };
        dispatch(
            &self.fft_bitreverse_pipeline,
            product,
            inverse_bitrev,
            signal_elements,
            &[fft_u32, transforms_u32],
        )?;
        let mut inverse_is_first = inverse_bitrev_is_first;
        for stage in 1..=plan.stages {
            let (source, destination) = if inverse_is_first {
                (signal_first, signal_second)
            } else {
                (signal_second, signal_first)
            };
            dispatch(
                &self.fft_stage_pipeline,
                source,
                destination,
                signal_elements,
                &[fft_u32, transforms_u32, stage, 1],
            )?;
            inverse_is_first = !inverse_is_first;
        }
        let inverse = if inverse_is_first {
            signal_first
        } else {
            signal_second
        };
        self.encode_extract_overlap_save(
            encoder.as_ref(),
            inverse,
            destination,
            [
                time_u32,
                channels_u32,
                chunk_u32,
                kernel_u32,
                fft_u32,
                chunks_u32,
                output_u32,
            ],
            shape.elements(),
        )?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal chunked Hyena convolution failed: {error}");
        }
        let mut result = vec![0.0; shape.elements()];
        unsafe {
            result.as_mut_ptr().copy_from_nonoverlapping(
                destination.contents().cast::<f32>().as_ptr(),
                result.len(),
            );
        }
        Ok(result)
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
    fn metal_gate_backward_matches_cpu_when_metal_is_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let mixed = [0.5, -1.5, 2.0, 0.25];
        let gated_projection = [
            9.0,
            8.0,
            0.25_f32.tanh(),
            (-0.75_f32).tanh(),
            7.0,
            6.0,
            0.5_f32.tanh(),
            1.25_f32.tanh(),
        ];
        let output_gradient = [0.75, -0.5, 1.0, -1.5];
        let expected =
            crate::model::hyena_gate_backward(&mixed, &gated_projection, &output_gradient, 2)
                .unwrap();
        let actual = runtime
            .hyena_gate_backward_reference(&mixed, &gated_projection, &output_gradient, 2, 2)
            .unwrap();
        assert_eq!(actual.mixed_gradient, expected.mixed_gradient);
        assert_eq!(actual.projection_gradient, expected.projection_gradient);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_ternary_backward_matches_cpu_contract_when_metal_is_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let shape = TernaryLinearShape::new(2, 3, 2).unwrap();
        let input = [0.25, -1.5, 2.0, 0.5, 1.0, -0.75];
        let output_gradient = [0.75, -0.5, 1.0, 0.25];
        // Output rows: [+1, -1, 0] and [+1, 0, -1].
        let positive = [0b010_001_u64];
        let negative = [0b100_010_u64];
        let scales = [0.5, 0.75];
        let code = |weight: usize| {
            let bit = 1_u64 << weight;
            if positive[0] & bit != 0 {
                1.0
            } else if negative[0] & bit != 0 {
                -1.0
            } else {
                0.0
            }
        };
        let mut expected_input = vec![0.0; input.len()];
        let mut expected_weight = vec![0.0; 6];
        for row in 0..2 {
            for feature in 0..3 {
                for output in 0..2 {
                    expected_input[row * 3 + feature] += output_gradient[row * 2 + output]
                        * scales[output]
                        * code(output * 3 + feature);
                    expected_weight[output * 3 + feature] += output_gradient[row * 2 + output]
                        * scales[output]
                        * input[row * 3 + feature];
                }
            }
        }
        let actual = runtime
            .ternary_linear_backward_reference(
                &input,
                &output_gradient,
                &positive,
                &negative,
                &scales,
                shape,
            )
            .unwrap();
        for (actual, expected) in actual.input_gradient.iter().zip(expected_input) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        for (actual, expected) in actual.latent_weight_gradient.iter().zip(expected_weight) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_fp16_stateless_sgd_matches_clipped_cpu_contract_when_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let initial = crate::precision::Fp16Storage::from_f32([0.5, -1.0, 0.25, -0.75]);
        let gradient = [2.0, -3.0, 0.125, -0.5];
        let parameters = runtime.upload_resident_fp16_parameters(&initial).unwrap();
        runtime
            .resident_fp16_stateless_sgd(&parameters, &gradient, 0.2)
            .unwrap();
        let actual = runtime
            .download_resident_fp16_parameters(&parameters)
            .unwrap();
        let mut expected = initial;
        for (index, &value) in gradient.iter().enumerate() {
            expected.apply_clipped_sgd(index, value, 0.2);
        }
        assert_eq!(actual, expected);
        assert!(
            runtime
                .resident_fp16_stateless_sgd(&parameters, &[f32::NAN; 4], 0.2)
                .is_err()
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_trainable_ternary_refreshes_scales_and_codes_after_sgd_when_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let master = crate::precision::Fp16Storage::from_f32([1.0, -0.5, 0.1, -0.8, 0.4, 0.2]);
        let weights = runtime
            .upload_trainable_fp16_ternary_weights(&master, 3, 2, 0.7)
            .unwrap();
        runtime
            .resident_trainable_fp16_ternary_stateless_sgd(
                &weights,
                &[2.0, -3.0, 0.0, 0.25, -0.5, 1.0],
                0.2,
            )
            .unwrap();
        let (actual_master, positive, negative, scales) = runtime
            .download_trainable_fp16_ternary_weights(&weights)
            .unwrap();
        let mut expected_master = master;
        for (index, &gradient) in [2.0_f32, -3.0, 0.0, 0.25, -0.5, 1.0].iter().enumerate() {
            expected_master.apply_clipped_sgd(index, gradient, 0.2);
        }
        assert_eq!(actual_master, expected_master);
        let mut expected_positive = 0_u64;
        let mut expected_negative = 0_u64;
        for row in 0..2 {
            let start = row * 3;
            let scale = (0..3)
                .map(|feature| expected_master.get(start + feature).abs())
                .sum::<f32>()
                / 3.0;
            assert!((scales[row] - scale).abs() < 1e-6);
            for feature in 0..3 {
                let bit = 1_u64 << (start + feature);
                let value = expected_master.get(start + feature);
                if value > 0.7 * scale {
                    expected_positive |= bit;
                } else if value < -0.7 * scale {
                    expected_negative |= bit;
                }
            }
        }
        assert_eq!(positive, vec![expected_positive]);
        assert_eq!(negative, vec![expected_negative]);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_backward_workspace_is_grow_only_when_metal_is_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let large = TernaryLinearShape::new(2, 3, 2).unwrap();
        runtime
            .ternary_linear_backward_reference(
                &[1.0, -1.0, 0.5, 0.25, 2.0, -0.5],
                &[0.5, -0.25, 1.0, 0.75],
                &[0b010_001_u64],
                &[0b100_010_u64],
                &[0.5, 0.75],
                large,
            )
            .unwrap();
        let capacity_after_large = {
            let buffers = runtime.backward_buffers.borrow();
            (
                buffers.output_gradient_capacity,
                buffers.input_gradient_capacity,
                buffers.parameter_gradient_capacity,
            )
        };
        let small = TernaryLinearShape::new(1, 2, 1).unwrap();
        runtime
            .ternary_linear_backward_reference(
                &[1.0, -1.0],
                &[0.5],
                &[0b01_u64],
                &[0],
                &[1.0],
                small,
            )
            .unwrap();
        let buffers = runtime.backward_buffers.borrow();
        assert_eq!(
            (
                buffers.output_gradient_capacity,
                buffers.input_gradient_capacity,
                buffers.parameter_gradient_capacity,
            ),
            capacity_after_large
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_bounded_convolution_backward_matches_cpu_when_metal_is_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let plan = HyenaChunkPlan::new(4, 3).unwrap();
        let input = [0.5, -1.0, 1.5, 2.0, -0.5, 3.0, 4.0, -2.0];
        let filter = [0.5, -0.25, 0.125, 1.0, 0.0, -0.5];
        let output_gradient = [0.25, -0.5, 1.0, 0.75, -1.5, 0.5, 0.25, -0.75];
        let expected = crate::hyena::causal_chunked_conv_backward(
            &input,
            &filter,
            &output_gradient,
            1,
            4,
            2,
            plan,
        )
        .unwrap();
        let actual = runtime
            .causal_chunked_conv_backward_reference(
                &input,
                &filter,
                &output_gradient,
                1,
                4,
                2,
                plan,
            )
            .unwrap();
        for (actual, expected) in actual.input_gradient.iter().zip(expected.input_gradient) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        for (actual, expected) in actual.filter_gradient.iter().zip(expected.filter_gradient) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_rms_norm_backward_matches_cpu_contract_when_metal_is_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let input = [3.0, 4.0, 0.0, 2.0, -2.0, 1.0];
        let output_gradient = [0.25, -0.5, 1.0, -1.5, 0.5, 0.75];
        let mut normalized = vec![0.0; input.len()];
        let mut expected = vec![0.0; input.len()];
        for row in 0..2 {
            let offset = row * 3;
            let inverse_rms = (input[offset..offset + 3]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / 3.0
                + 1e-5)
                .sqrt()
                .recip();
            for channel in 0..3 {
                normalized[offset + channel] = input[offset + channel] * inverse_rms;
            }
            let projection = normalized[offset..offset + 3]
                .iter()
                .zip(&output_gradient[offset..offset + 3])
                .map(|(value, gradient)| value * gradient)
                .sum::<f32>()
                / 3.0;
            for channel in 0..3 {
                expected[offset + channel] = inverse_rms
                    * (output_gradient[offset + channel]
                        - normalized[offset + channel] * projection);
            }
        }
        let actual = runtime
            .rms_norm_backward_reference(&input, &normalized, &output_gradient, 2, 3)
            .unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_projection_signal_join_matches_cpu_when_metal_is_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let projection = [1.0, 2.0, 9.0, 8.0, -3.0, 4.0, 7.0, 6.0];
        let gate_gradient = [0.5, -0.25, 1.0, 2.0, -1.5, 0.75, 0.25, -0.5];
        let signal_gradient = [0.125, -0.5, 1.25, 0.25];
        let (signal, joined) = runtime
            .projection_signal_backward_reference(
                &projection,
                &gate_gradient,
                &signal_gradient,
                2,
                2,
            )
            .unwrap();
        assert_eq!(signal, [1.0, 2.0, -3.0, 4.0]);
        assert_eq!(joined, [0.625, -0.75, 1.0, 2.0, -0.25, 1.0, 0.25, -0.5]);
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
    fn resident_gradient_slots_are_independent_and_grow_only_when_available() {
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        let source = [1.0, -2.0, 0.5, 3.0];
        let slot = runtime.upload_resident_gradient(&source, 2, 2).unwrap();
        assert_eq!(slot, ResidentGradientSlot::First);
        assert_eq!(slot.other(), ResidentGradientSlot::Second);
        assert_eq!(
            runtime.download_resident_gradient(slot, 2, 2).unwrap(),
            source
        );
        runtime.reserve_gradients(1, 2).unwrap();
        assert_eq!(
            runtime.download_resident_gradient(slot, 2, 2).unwrap(),
            source
        );
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
        let expected = crate::hyena::causal_chunked_conv(
            &input,
            &filter,
            1,
            4,
            2,
            HyenaChunkPlan::new(4, 4).unwrap(),
        )
        .unwrap();
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
        let expected = crate::hyena::causal_chunked_conv_implicit_strided(
            &input,
            &filter,
            1,
            4,
            2,
            2,
            0,
            HyenaChunkPlan::new(4, 4).unwrap(),
        )
        .unwrap();
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
    fn chunked_implicit_hyena_chain_matches_cpu_across_boundaries_when_available() {
        let filter = crate::hyena::ImplicitFilter::new(2, 3, 7);
        let plan = HyenaChunkPlan::new(4, 3).unwrap();
        let input = [
            0.5, -1.0, 1.5, 2.0, -0.5, 3.0, 4.0, -2.0, 1.25, 0.75, -3.0, 2.5, 0.0, -1.5,
        ];
        let expected = crate::hyena::causal_chunked_conv_implicit_strided(
            &input, &filter, 1, 7, 2, 2, 0, plan,
        )
        .unwrap();
        if let Ok(runtime) = MetalRuntime::new() {
            let actual = runtime
                .causal_chunked_conv_implicit_strided_forward(&input, &filter, 1, 7, 2, 2, 0, plan)
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
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

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_fp16_gate_matches_quantized_contract_when_metal_is_available() {
        let values =
            crate::precision::Fp16Storage::from_f32([1.0, -2.0, 0.5, -1.5, -0.25, 0.75, 2.0, -3.0]);
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        runtime.reserve_fp16_activations(2, 4).unwrap();
        let slot = runtime
            .upload_resident_fp16_activations(&values, 2, 4)
            .unwrap();
        let slot = runtime.resident_tanh_gate_fp16(slot, 2, 2).unwrap();
        let actual = runtime
            .download_resident_fp16_activations(slot, 2, 4)
            .unwrap();
        for row in 0..2 {
            for channel in 0..2 {
                assert_eq!(actual.get(row * 4 + channel), values.get(row * 4 + channel));
                let gate = values.get(row * 4 + 2 + channel).tanh();
                assert_eq!(
                    actual.get(row * 4 + 2 + channel),
                    crate::precision::Fp16::from_f32(gate).to_f32()
                );
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_fp16_apply_gate_and_residual_stay_on_slots_when_metal_is_available() {
        let values = crate::precision::Fp16Storage::from_f32([1.0, 2.0, 0.5, -1.0]);
        let Ok(runtime) = MetalRuntime::new() else {
            return;
        };
        runtime.reserve_fp16_activations(1, 4).unwrap();
        let residual = runtime
            .upload_resident_fp16_activations(&values, 1, 4)
            .unwrap();
        let gated = runtime.resident_tanh_gate_fp16(residual, 1, 2).unwrap();
        runtime
            .resident_apply_gate_fp16(residual, gated, ResidentFp16ActivationSlot::Third, 1, 2)
            .unwrap();
        runtime
            .resident_residual_add_fp16(residual, ResidentFp16ActivationSlot::Third, residual, 1, 2)
            .unwrap();
        let actual = runtime
            .download_resident_fp16_activations(residual, 1, 2)
            .unwrap();
        for channel in 0..2 {
            let mixed = values.get(channel);
            let gate = crate::precision::Fp16::from_f32(values.get(2 + channel).tanh()).to_f32();
            let update = crate::precision::Fp16::from_f32(mixed * gate).to_f32();
            assert_eq!(
                actual.get(channel),
                crate::precision::Fp16::from_f32(mixed + update).to_f32()
            );
        }
    }
}
