//! Metal runtime for Heron: LayerNorm, BinaryConnect, FP16 linear, streamed CE,
//! 1-bit QKV ROSA SAM, and WKV7.
//!
//! Buffer mapping lives in [`ffi`]. There is no MPS path. Identity remains a
//! pipeline-smoke entry point.

use anyhow::{Result, bail};
use core::cell::RefCell;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::model::{CE_NO_IGNORE, causal_ce_gradient_scale, causal_ce_row_valid};
use crate::precision::Fp16;

pub mod ffi;

use self::ffi::{MetalBuffer, set_buffer, set_bytes_f32, set_bytes_u32};

type Device = Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

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

/// Checked dimensions for one packed or FP16 projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearDispatchShape {
    pub rows: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearDispatchShape {
    pub fn new(rows: usize, in_features: usize, out_features: usize) -> Result<Self> {
        MetalDispatchShape::new(rows, 1, out_features)?;
        if in_features == 0 || u32::try_from(in_features).is_err() {
            bail!("Metal linear input width is invalid");
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
            .ok_or_else(|| anyhow::anyhow!("Metal linear weight shape overflow"))
            .map(|weights| weights.div_ceil(32))
    }
}

pub const IDENTITY_KERNEL_NAME: &str = "ullis_identity";
pub const LAYER_NORM_KERNEL_NAME: &str = "ullis_layer_norm";
pub const LAYER_NORM_BACKWARD_KERNEL_NAME: &str = "ullis_layer_norm_backward";
pub const LAYER_NORM_PARAM_BWD_KERNEL_NAME: &str = "ullis_layer_norm_param_bwd";
pub const TIME_SHIFT_DELTA_KERNEL_NAME: &str = "ullis_time_shift_delta";
pub const TIME_SHIFT_MIX_KERNEL_NAME: &str = "ullis_time_shift_mix";
pub const TIME_SHIFT_MIX3_KERNEL_NAME: &str = "ullis_time_shift_mix3";
pub const BINARY_LINEAR_KERNEL_NAME: &str = "ullis_binary_linear";
pub const BINARY_LINEAR_INPUT_BWD_KERNEL_NAME: &str = "ullis_binary_linear_input_bwd";
pub const BINARY_LINEAR_SCALE_BWD_KERNEL_NAME: &str = "ullis_binary_linear_scale_bwd";
pub const BINARY_LINEAR_WEIGHT_BWD_KERNEL_NAME: &str = "ullis_binary_linear_weight_bwd";
pub const BINARY_LINEAR_SCALE_BWD_FROM_OUTPUT_KERNEL_NAME: &str =
    "ullis_binary_linear_scale_bwd_from_output";
pub const BINARY_LINEAR_LATENT_SGD_KERNEL_NAME: &str = "ullis_binary_linear_latent_sgd";
pub const LINEAR_TILE: usize = 16;
pub const SCALE_BWD_THREADS: usize = 256;
pub const FP16_LINEAR_KERNEL_NAME: &str = "ullis_fp16_linear";
pub const FP16_LINEAR_BWD_KERNEL_NAME: &str = "ullis_fp16_linear_bwd";
pub const SIGN_PACK_BITS_KERNEL_NAME: &str = "ullis_sign_pack_bits";
pub const PACK_LATENT_BITS_KERNEL_NAME: &str = "ullis_pack_latent_bits";
pub const ROSA_SAM_RESET_KERNEL_NAME: &str = "ullis_rosa_sam_reset";
pub const ROSA_QKV_1BIT_FWD_KERNEL_NAME: &str = "ullis_rosa_qkv_1bit_fwd";
pub const ROSA_QKV_1BIT_BWD_E_KERNEL_NAME: &str = "ullis_rosa_qkv_1bit_bwd_e";
pub const CMIX_RELU2_KERNEL_NAME: &str = "ullis_cmix_relu2";
pub const CMIX_RELU2_BACKWARD_KERNEL_NAME: &str = "ullis_cmix_relu2_backward";
pub const RESIDUAL_ADD_KERNEL_NAME: &str = "ullis_residual_add";
pub const STREAMED_CROSS_ENTROPY_FP16_KERNEL_NAME: &str = "ullis_streamed_cross_entropy_fp16";
pub const SOFTMAX_CROSS_ENTROPY_KERNEL_NAME: &str = "ullis_softmax_cross_entropy";
pub const CLIPPED_SGD_FP16_KERNEL_NAME: &str = "ullis_clipped_sgd_fp16";
pub const BINARYCONNECT_SGD_FP16_KERNEL_NAME: &str = "ullis_binaryconnect_sgd_fp16";
pub const WKV7_FORWARD_KERNEL_NAME: &str = "ullis_wkv7_forward";
pub const WKV7_BACKWARD_KERNEL_NAME: &str = "ullis_wkv7_backward";

pub const RWKV8_METAL_SOURCE: &str = include_str!("metal/rwkv8.metal");

pub const PR3_KERNEL_NAMES: &[&str] = &[
    IDENTITY_KERNEL_NAME,
    LAYER_NORM_KERNEL_NAME,
    LAYER_NORM_BACKWARD_KERNEL_NAME,
    LAYER_NORM_PARAM_BWD_KERNEL_NAME,
    TIME_SHIFT_DELTA_KERNEL_NAME,
    TIME_SHIFT_MIX_KERNEL_NAME,
    TIME_SHIFT_MIX3_KERNEL_NAME,
    BINARY_LINEAR_KERNEL_NAME,
    BINARY_LINEAR_INPUT_BWD_KERNEL_NAME,
    BINARY_LINEAR_SCALE_BWD_KERNEL_NAME,
    BINARY_LINEAR_WEIGHT_BWD_KERNEL_NAME,
    BINARY_LINEAR_SCALE_BWD_FROM_OUTPUT_KERNEL_NAME,
    BINARY_LINEAR_LATENT_SGD_KERNEL_NAME,
    FP16_LINEAR_KERNEL_NAME,
    FP16_LINEAR_BWD_KERNEL_NAME,
    SIGN_PACK_BITS_KERNEL_NAME,
    PACK_LATENT_BITS_KERNEL_NAME,
    CMIX_RELU2_KERNEL_NAME,
    CMIX_RELU2_BACKWARD_KERNEL_NAME,
    RESIDUAL_ADD_KERNEL_NAME,
    STREAMED_CROSS_ENTROPY_FP16_KERNEL_NAME,
    SOFTMAX_CROSS_ENTROPY_KERNEL_NAME,
    CLIPPED_SGD_FP16_KERNEL_NAME,
    BINARYCONNECT_SGD_FP16_KERNEL_NAME,
];

pub const PR4_KERNEL_NAMES: &[&str] = &[ROSA_SAM_RESET_KERNEL_NAME, ROSA_QKV_1BIT_FWD_KERNEL_NAME];
pub const PR5_KERNEL_NAMES: &[&str] = &[ROSA_QKV_1BIT_BWD_E_KERNEL_NAME];
pub const PR8_KERNEL_NAMES: &[&str] = &[WKV7_FORWARD_KERNEL_NAME, WKV7_BACKWARD_KERNEL_NAME];

/// Compiles the identity entry point and checks its dispatch capacity.
pub fn validate_metal_pipeline(shape: MetalDispatchShape) -> Result<usize> {
    validate_metal_kernel(IDENTITY_KERNEL_NAME, shape)
}

/// Compiles a named Ullis MSL entry point and checks its dispatch capacity.
pub fn validate_metal_kernel(kernel_name: &str, shape: MetalDispatchShape) -> Result<usize> {
    let pipeline = compile_named_pipeline(kernel_name)?;
    let width = pipeline.maxTotalThreadsPerThreadgroup();
    if width == 0 {
        bail!("Metal pipeline reported zero threads per threadgroup");
    }
    let _ = shape.elements();
    Ok(width)
}

fn compile_options() -> Retained<objc2_metal::MTLCompileOptions> {
    use objc2_metal::{MTLCompileOptions, MTLMathMode};

    let options = MTLCompileOptions::new();
    // Default fast-math relaxes exp/log enough to miss the WKV7 CPU oracle.
    options.setMathMode(MTLMathMode::Safe);
    options
}

fn compile_named_pipeline(kernel_name: &str) -> Result<Pipeline> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
    let source = NSString::from_str(RWKV8_METAL_SOURCE);
    let options = compile_options();
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| anyhow::anyhow!("Metal shader compilation failed: {error}"))?;
    pipeline_from_library(&device, &library, kernel_name)
}

fn pipeline_from_library(
    device: &Device,
    library: &Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>,
    kernel_name: &str,
) -> Result<Pipeline> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLDevice, MTLLibrary};

    let name = NSString::from_str(kernel_name);
    let function = library
        .newFunctionWithName(&name)
        .ok_or_else(|| anyhow::anyhow!("Metal function {kernel_name:?} is missing"))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| anyhow::anyhow!("Metal pipeline {kernel_name:?} failed: {error}"))
}

fn fp16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .copied()
        .map(|v| Fp16::from_f32(v).to_bits())
        .collect()
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("Metal {label} exceeds u32"))
}

/// Local derivatives of affine LayerNorm.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerNormBackward {
    pub input_gradient: Vec<f32>,
    pub weight_gradient: Vec<f32>,
    pub bias_gradient: Vec<f32>,
}

/// STE gradients for a packed ±1 linear (no `g_w`; that lives in latent SGD).
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryLinearBackward {
    pub input_gradient: Vec<f32>,
    pub scale_gradient: Vec<f32>,
    pub bias_gradient: Option<Vec<f32>>,
    pub weight_gradient: Vec<f32>,
}

/// Dense FP16 linear derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct Fp16LinearBackward {
    pub input_gradient: Vec<f32>,
    pub weight_gradient: Vec<f32>,
}

/// Resident 1-bit QKV SAM forward: collapsed idx and `out = (2·idx − 1)·e`.
#[derive(Clone, Debug, PartialEq)]
pub struct RosaQkvForward {
    pub idx: Vec<u8>,
    pub out: Vec<f32>,
}

/// Fused ROSA-QKV block: time-mix, QKV, SAM, and output projection.
#[derive(Clone, Debug, PartialEq)]
pub struct RosaQkvBlockForward {
    pub idx: Vec<u8>,
    pub y: Vec<f32>,
    pub out: Vec<f32>,
}

/// Fused CMix block activations kept for STE backward.
#[derive(Clone, Debug, PartialEq)]
pub struct CmixBlockForward {
    pub xx: Vec<f32>,
    pub shifted: Vec<f32>,
    pub key: Vec<f32>,
    pub relu2: Vec<f32>,
    pub out: Vec<f32>,
}

/// Packed head train with BinaryConnect SGD applied on GPU. `g_w` stays resident.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedHeadTrainSgd {
    pub mean_loss: f32,
    pub hidden_gradient: Vec<f32>,
    pub scale_gradient: Vec<f32>,
    pub next_latent: Vec<u16>,
    pub next_residual: Vec<f32>,
    pub next_bits: Vec<u32>,
    pub row_loss: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CmixBlockBackwardSgd {
    pub g_shifted: Vec<f32>,
    pub g_key_scale: Vec<f32>,
    pub next_key_latent: Vec<u16>,
    pub next_key_residual: Vec<f32>,
    pub next_key_bits: Vec<u32>,
    pub next_value_weight: Vec<u16>,
    pub next_value_residual: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RosaOStopGradSgd {
    pub e_gradient: Vec<f32>,
    pub scale_gradient: Vec<f32>,
    pub bias_gradient: Vec<f32>,
    pub next_latent: Vec<u16>,
    pub next_residual: Vec<f32>,
    pub next_bits: Vec<u32>,
}

/// Next-token streamed CE without a `[rows, vocab]` logit tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamedCrossEntropy {
    pub mean_loss: f32,
    pub hidden_gradient: Vec<f32>,
    pub scale_gradient: Vec<f32>,
    pub row_loss: Vec<f32>,
    pub logit_gradient: Vec<f32>,
}

/// Head train: packed linear + softmax CE + STE `g_w`, logits stay on GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedHeadTrain {
    pub mean_loss: f32,
    pub hidden_gradient: Vec<f32>,
    pub scale_gradient: Vec<f32>,
    pub weight_gradient: Vec<f32>,
}

struct RecycledBuffer<'a> {
    inner: Option<MetalBuffer>,
    scratch: &'a RefCell<Vec<MetalBuffer>>,
}

impl RecycledBuffer<'_> {
    fn live(&self) -> &MetalBuffer {
        self.inner
            .as_ref()
            .expect("scratch buffer is returned to the pool")
    }
}

impl Drop for RecycledBuffer<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.inner.take() {
            self.scratch.borrow_mut().push(buffer);
        }
    }
}

impl core::ops::Deref for RecycledBuffer<'_> {
    type Target = MetalBuffer;

    fn deref(&self) -> &MetalBuffer {
        self.live()
    }
}

impl AsRef<MetalBuffer> for RecycledBuffer<'_> {
    fn as_ref(&self) -> &MetalBuffer {
        self.live()
    }
}

impl AsRef<MetalBuffer> for MetalBuffer {
    fn as_ref(&self) -> &MetalBuffer {
        self
    }
}

struct Pipelines {
    identity: Pipeline,
    layer_norm: Pipeline,
    layer_norm_backward: Pipeline,
    layer_norm_param_bwd: Pipeline,
    time_shift_delta: Pipeline,
    time_shift_mix: Pipeline,
    time_shift_mix3: Pipeline,
    binary_linear: Pipeline,
    binary_linear_input_bwd: Pipeline,
    binary_linear_scale_bwd: Pipeline,
    binary_linear_weight_bwd: Pipeline,
    binary_linear_scale_bwd_from_output: Pipeline,
    binary_linear_latent_sgd: Pipeline,
    fp16_linear: Pipeline,
    fp16_linear_bwd: Pipeline,
    sign_pack_bits: Pipeline,
    pack_latent_bits: Pipeline,
    rosa_sam_reset: Pipeline,
    rosa_qkv_1bit_fwd: Pipeline,
    rosa_qkv_1bit_bwd_e: Pipeline,
    cmix_relu2: Pipeline,
    cmix_relu2_backward: Pipeline,
    residual_add: Pipeline,
    streamed_cross_entropy_fp16: Pipeline,
    softmax_cross_entropy: Pipeline,
    clipped_sgd_fp16: Pipeline,
    binaryconnect_sgd_fp16: Pipeline,
    wkv7_forward: Pipeline,
    wkv7_backward: Pipeline,
}

/// Reusable Metal objects for resident Heron kernels. No MPS GEMM.
pub struct MetalRuntime {
    device: Device,
    queue: Queue,
    pipelines: Pipelines,
    scratch: RefCell<Vec<MetalBuffer>>,
}

impl std::fmt::Debug for MetalRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalRuntime").finish_non_exhaustive()
    }
}

impl MetalRuntime {
    pub fn new() -> Result<Self> {
        use objc2_foundation::NSString;
        use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
        let source = NSString::from_str(RWKV8_METAL_SOURCE);
        let options = compile_options();
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|error| anyhow::anyhow!("Metal shader compilation failed: {error}"))?;
        let pipelines = Pipelines {
            identity: pipeline_from_library(&device, &library, IDENTITY_KERNEL_NAME)?,
            layer_norm: pipeline_from_library(&device, &library, LAYER_NORM_KERNEL_NAME)?,
            layer_norm_backward: pipeline_from_library(
                &device,
                &library,
                LAYER_NORM_BACKWARD_KERNEL_NAME,
            )?,
            layer_norm_param_bwd: pipeline_from_library(
                &device,
                &library,
                LAYER_NORM_PARAM_BWD_KERNEL_NAME,
            )?,
            time_shift_delta: pipeline_from_library(
                &device,
                &library,
                TIME_SHIFT_DELTA_KERNEL_NAME,
            )?,
            time_shift_mix: pipeline_from_library(&device, &library, TIME_SHIFT_MIX_KERNEL_NAME)?,
            time_shift_mix3: pipeline_from_library(&device, &library, TIME_SHIFT_MIX3_KERNEL_NAME)?,
            binary_linear: pipeline_from_library(&device, &library, BINARY_LINEAR_KERNEL_NAME)?,
            binary_linear_input_bwd: pipeline_from_library(
                &device,
                &library,
                BINARY_LINEAR_INPUT_BWD_KERNEL_NAME,
            )?,
            binary_linear_scale_bwd: pipeline_from_library(
                &device,
                &library,
                BINARY_LINEAR_SCALE_BWD_KERNEL_NAME,
            )?,
            binary_linear_weight_bwd: pipeline_from_library(
                &device,
                &library,
                BINARY_LINEAR_WEIGHT_BWD_KERNEL_NAME,
            )?,
            binary_linear_scale_bwd_from_output: pipeline_from_library(
                &device,
                &library,
                BINARY_LINEAR_SCALE_BWD_FROM_OUTPUT_KERNEL_NAME,
            )?,
            binary_linear_latent_sgd: pipeline_from_library(
                &device,
                &library,
                BINARY_LINEAR_LATENT_SGD_KERNEL_NAME,
            )?,
            fp16_linear: pipeline_from_library(&device, &library, FP16_LINEAR_KERNEL_NAME)?,
            fp16_linear_bwd: pipeline_from_library(&device, &library, FP16_LINEAR_BWD_KERNEL_NAME)?,
            sign_pack_bits: pipeline_from_library(&device, &library, SIGN_PACK_BITS_KERNEL_NAME)?,
            pack_latent_bits: pipeline_from_library(
                &device,
                &library,
                PACK_LATENT_BITS_KERNEL_NAME,
            )?,
            rosa_sam_reset: pipeline_from_library(&device, &library, ROSA_SAM_RESET_KERNEL_NAME)?,
            rosa_qkv_1bit_fwd: pipeline_from_library(
                &device,
                &library,
                ROSA_QKV_1BIT_FWD_KERNEL_NAME,
            )?,
            rosa_qkv_1bit_bwd_e: pipeline_from_library(
                &device,
                &library,
                ROSA_QKV_1BIT_BWD_E_KERNEL_NAME,
            )?,
            cmix_relu2: pipeline_from_library(&device, &library, CMIX_RELU2_KERNEL_NAME)?,
            cmix_relu2_backward: pipeline_from_library(
                &device,
                &library,
                CMIX_RELU2_BACKWARD_KERNEL_NAME,
            )?,
            residual_add: pipeline_from_library(&device, &library, RESIDUAL_ADD_KERNEL_NAME)?,
            streamed_cross_entropy_fp16: pipeline_from_library(
                &device,
                &library,
                STREAMED_CROSS_ENTROPY_FP16_KERNEL_NAME,
            )?,
            softmax_cross_entropy: pipeline_from_library(
                &device,
                &library,
                SOFTMAX_CROSS_ENTROPY_KERNEL_NAME,
            )?,
            clipped_sgd_fp16: pipeline_from_library(
                &device,
                &library,
                CLIPPED_SGD_FP16_KERNEL_NAME,
            )?,
            binaryconnect_sgd_fp16: pipeline_from_library(
                &device,
                &library,
                BINARYCONNECT_SGD_FP16_KERNEL_NAME,
            )?,
            wkv7_forward: pipeline_from_library(&device, &library, WKV7_FORWARD_KERNEL_NAME)?,
            wkv7_backward: pipeline_from_library(&device, &library, WKV7_BACKWARD_KERNEL_NAME)?,
        };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow::anyhow!("Metal command queue is unavailable"))?;
        Ok(Self {
            device,
            queue,
            pipelines,
            scratch: RefCell::new(Vec::new()),
        })
    }

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

    fn alloc_shared(&self, bytes: usize) -> Result<MetalBuffer> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};

        if bytes == 0 {
            bail!("Metal buffer length must be positive");
        }
        let inner = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Metal buffer allocation failed"))?;
        Ok(MetalBuffer::from_retained(inner, bytes))
    }

    fn shared_buffer(&self, bytes: usize) -> Result<RecycledBuffer<'_>> {
        let recycled = {
            let mut free = self.scratch.borrow_mut();
            free.iter()
                .position(|buffer| buffer.len() >= bytes)
                .map(|index| free.swap_remove(index))
        };
        let inner = match recycled {
            Some(buffer) => buffer,
            None => self.alloc_shared(bytes.next_power_of_two().max(bytes))?,
        };
        Ok(RecycledBuffer {
            inner: Some(inner),
            scratch: &self.scratch,
        })
    }

    fn buffer_f32(&self, values: &[f32]) -> Result<RecycledBuffer<'_>> {
        let bytes = values
            .len()
            .checked_mul(size_of::<f32>())
            .unwrap_or(size_of::<f32>())
            .max(size_of::<f32>());
        let buffer = self.shared_buffer(bytes)?;
        if values.is_empty() {
            buffer.zero()?;
        } else {
            buffer.write_f32(values)?;
        }
        Ok(buffer)
    }

    fn buffer_u16(&self, values: &[u16]) -> Result<RecycledBuffer<'_>> {
        let bytes = values
            .len()
            .checked_mul(size_of::<u16>())
            .unwrap_or(size_of::<u16>())
            .max(size_of::<u16>());
        let buffer = self.shared_buffer(bytes)?;
        if values.is_empty() {
            buffer.zero()?;
        } else {
            buffer.write_u16(values)?;
        }
        Ok(buffer)
    }

    fn buffer_u32(&self, values: &[u32]) -> Result<RecycledBuffer<'_>> {
        let bytes = values
            .len()
            .checked_mul(size_of::<u32>())
            .unwrap_or(size_of::<u32>())
            .max(size_of::<u32>());
        let buffer = self.shared_buffer(bytes)?;
        if values.is_empty() {
            buffer.zero()?;
        } else {
            buffer.write_u32(values)?;
        }
        Ok(buffer)
    }

    fn zeros_f32(&self, len: usize) -> Result<RecycledBuffer<'_>> {
        let bytes = len.saturating_mul(size_of::<f32>()).max(size_of::<f32>());
        let buffer = self.shared_buffer(bytes)?;
        buffer.zero()?;
        Ok(buffer)
    }

    fn alloc_f32(&self, len: usize) -> Result<RecycledBuffer<'_>> {
        let bytes = len.saturating_mul(size_of::<f32>()).max(size_of::<f32>());
        self.shared_buffer(bytes)
    }

    fn alloc_bytes(&self, bytes: usize) -> Result<RecycledBuffer<'_>> {
        self.shared_buffer(bytes.max(1))
    }

    fn submit(&self, encode: impl FnOnce(&ffi::ComputeEncoder) -> Result<()>) -> Result<()> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        encode(encoder.as_ref())?;
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("Metal command failed: {error}");
        }
        Ok(())
    }

    fn set_u32s(encoder: &ffi::ComputeEncoder, start: usize, values: &[u32]) -> Result<()> {
        for (offset, value) in values.iter().enumerate() {
            set_bytes_u32(encoder, start + offset, &[*value])?;
        }
        Ok(())
    }

    fn encode_1d<B: AsRef<MetalBuffer>>(
        encoder: &ffi::ComputeEncoder,
        pipeline: &Pipeline,
        buffers: &[B],
        constants: impl FnOnce(&ffi::ComputeEncoder) -> Result<()>,
        threads: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

        if threads == 0 {
            bail!("Metal dispatch cannot be empty");
        }
        encoder.setComputePipelineState(pipeline);
        for (slot, buffer) in buffers.iter().enumerate() {
            set_buffer(encoder, buffer.as_ref(), slot);
        }
        constants(encoder)?;
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(threads);
        if width == 0 {
            bail!("Metal pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads,
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

    fn encode_clipped_sgd_fp16<B: AsRef<MetalBuffer>>(
        encoder: &ffi::ComputeEncoder,
        pipeline: &Pipeline,
        parameters: &B,
        residual: &B,
        gradient: &B,
        learning_rate: f32,
        elements: usize,
    ) -> Result<()> {
        Self::encode_1d(
            encoder,
            pipeline,
            &[parameters, residual, gradient],
            |encoder| {
                set_bytes_f32(encoder, 3, &[learning_rate])?;
                set_bytes_u32(encoder, 4, &[as_u32(elements, "elements")?])
            },
            elements,
        )
    }

    fn encode_binaryconnect_sgd_fp16<B: AsRef<MetalBuffer>>(
        encoder: &ffi::ComputeEncoder,
        pipeline: &Pipeline,
        parameters: &B,
        residual: &B,
        gradient: &B,
        learning_rate: f32,
        ste_scale: f32,
        elements: usize,
    ) -> Result<()> {
        Self::encode_1d(
            encoder,
            pipeline,
            &[parameters, residual, gradient],
            |encoder| {
                set_bytes_f32(encoder, 3, &[learning_rate])?;
                set_bytes_u32(encoder, 4, &[as_u32(elements, "elements")?])?;
                set_bytes_f32(encoder, 5, &[ste_scale])
            },
            elements,
        )
    }

    fn encode_tiled<B: AsRef<MetalBuffer>>(
        encoder: &ffi::ComputeEncoder,
        pipeline: &Pipeline,
        buffers: &[B],
        constants: impl FnOnce(&ffi::ComputeEncoder) -> Result<()>,
        width: usize,
        height: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

        if width == 0 || height == 0 {
            bail!("Metal tiled dispatch cannot be empty");
        }
        encoder.setComputePipelineState(pipeline);
        for (slot, buffer) in buffers.iter().enumerate() {
            set_buffer(encoder, buffer.as_ref(), slot);
        }
        constants(encoder)?;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: width.div_ceil(LINEAR_TILE),
                height: height.div_ceil(LINEAR_TILE),
                depth: 1,
            },
            MTLSize {
                width: LINEAR_TILE,
                height: LINEAR_TILE,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_scale_groups<B: AsRef<MetalBuffer>>(
        encoder: &ffi::ComputeEncoder,
        pipeline: &Pipeline,
        buffers: &[B],
        constants: impl FnOnce(&ffi::ComputeEncoder) -> Result<()>,
        out_features: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

        if out_features == 0 {
            bail!("Metal scale-bwd dispatch cannot be empty");
        }
        encoder.setComputePipelineState(pipeline);
        for (slot, buffer) in buffers.iter().enumerate() {
            set_buffer(encoder, buffer.as_ref(), slot);
        }
        constants(encoder)?;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: out_features,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: SCALE_BWD_THREADS,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_wkv7<B: AsRef<MetalBuffer>>(
        encoder: &ffi::ComputeEncoder,
        pipeline: &Pipeline,
        buffers: &[B],
        time: u32,
        heads: u32,
        batch: u32,
        constant_start: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

        encoder.setComputePipelineState(pipeline);
        for (slot, buffer) in buffers.iter().enumerate() {
            set_buffer(encoder, buffer.as_ref(), slot);
        }
        set_bytes_u32(encoder, constant_start, &[time])?;
        set_bytes_u32(encoder, constant_start + 1, &[heads])?;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: usize::try_from(heads).unwrap_or(0),
                height: usize::try_from(batch).unwrap_or(0),
                depth: 1,
            },
            MTLSize {
                width: crate::wkv7::HEAD_SIZE,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Metal-resident CUDA `forward_kernel`. `T` must be a multiple of 16.
    pub fn wkv7_forward(
        &self,
        w: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        a: &[f32],
        b: &[f32],
        batch: usize,
        time: usize,
        heads: usize,
    ) -> Result<crate::wkv7::Wkv7Forward> {
        crate::wkv7::require_shape(w, q, k, v, a, b, batch, time, heads)?;
        let len = batch
            .saturating_mul(time)
            .saturating_mul(heads)
            .saturating_mul(crate::wkv7::HEAD_SIZE);
        let s_len = batch
            .saturating_mul(heads)
            .saturating_mul(time / crate::wkv7::CHUNK_LEN)
            .saturating_mul(crate::wkv7::HEAD_SIZE)
            .saturating_mul(crate::wkv7::HEAD_SIZE);
        let w_b = self.buffer_f32(w)?;
        let q_b = self.buffer_f32(q)?;
        let k_b = self.buffer_f32(k)?;
        let v_b = self.buffer_f32(v)?;
        let a_b = self.buffer_f32(a)?;
        let b_b = self.buffer_f32(b)?;
        let y_b = self.zeros_f32(len)?;
        let s_b = self.zeros_f32(s_len.max(1))?;
        let sa_b = self.zeros_f32(len)?;
        let t = as_u32(time, "WKV7 time")?;
        let h = as_u32(heads, "WKV7 heads")?;
        let bb = as_u32(batch, "WKV7 batch")?;
        self.submit(|encoder| {
            Self::encode_wkv7(
                encoder,
                &self.pipelines.wkv7_forward,
                &[&w_b, &q_b, &k_b, &v_b, &a_b, &b_b, &y_b, &s_b, &sa_b],
                t,
                h,
                bb,
                9,
            )
        })?;
        let mut y = vec![0.0; len];
        let mut s = vec![0.0; s_len];
        let mut sa = vec![0.0; len];
        y_b.read_f32(&mut y)?;
        s_b.read_f32(&mut s)?;
        sa_b.read_f32(&mut sa)?;
        Ok(crate::wkv7::Wkv7Forward { y, s, sa })
    }

    /// Metal-resident CUDA `backward_kernel`.
    pub fn wkv7_backward(
        &self,
        w: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        a: &[f32],
        b: &[f32],
        dy: &[f32],
        s: &[f32],
        sa: &[f32],
        batch: usize,
        time: usize,
        heads: usize,
    ) -> Result<crate::wkv7::Wkv7Backward> {
        crate::wkv7::require_backward_shape(w, q, k, v, a, b, dy, s, sa, batch, time, heads)?;
        let len = batch
            .saturating_mul(time)
            .saturating_mul(heads)
            .saturating_mul(crate::wkv7::HEAD_SIZE);
        let w_b = self.buffer_f32(w)?;
        let q_b = self.buffer_f32(q)?;
        let k_b = self.buffer_f32(k)?;
        let v_b = self.buffer_f32(v)?;
        let a_b = self.buffer_f32(a)?;
        let b_b = self.buffer_f32(b)?;
        let dy_b = self.buffer_f32(dy)?;
        let s_b = self.buffer_f32(s)?;
        let sa_b = self.buffer_f32(sa)?;
        let dw_b = self.zeros_f32(len)?;
        let dq_b = self.zeros_f32(len)?;
        let dk_b = self.zeros_f32(len)?;
        let dv_b = self.zeros_f32(len)?;
        let da_b = self.zeros_f32(len)?;
        let db_b = self.zeros_f32(len)?;
        let t = as_u32(time, "WKV7 time")?;
        let h = as_u32(heads, "WKV7 heads")?;
        let bb = as_u32(batch, "WKV7 batch")?;
        self.submit(|encoder| {
            Self::encode_wkv7(
                encoder,
                &self.pipelines.wkv7_backward,
                &[
                    &w_b, &q_b, &k_b, &v_b, &a_b, &b_b, &dy_b, &s_b, &sa_b, &dw_b, &dq_b, &dk_b,
                    &dv_b, &da_b, &db_b,
                ],
                t,
                h,
                bb,
                15,
            )
        })?;
        let mut dw = vec![0.0; len];
        let mut dq = vec![0.0; len];
        let mut dk = vec![0.0; len];
        let mut dv = vec![0.0; len];
        let mut da = vec![0.0; len];
        let mut db = vec![0.0; len];
        dw_b.read_f32(&mut dw)?;
        dq_b.read_f32(&mut dq)?;
        dk_b.read_f32(&mut dk)?;
        dv_b.read_f32(&mut dv)?;
        da_b.read_f32(&mut da)?;
        db_b.read_f32(&mut db)?;
        Ok(crate::wkv7::Wkv7Backward {
            dw,
            dq,
            dk,
            dv,
            da,
            db,
        })
    }

    pub fn identity(&self, input: &[f32]) -> Result<Vec<f32>> {
        MetalDispatchShape::new(1, input.len().max(1), 1)?;
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let elements = as_u32(input.len(), "element count")?;
        let input_buffer = self.buffer_f32(input)?;
        let output_buffer = self.zeros_f32(input.len())?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.identity,
                &[&input_buffer, &output_buffer],
                |encoder| Self::set_u32s(encoder, 2, &[elements]),
                input.len(),
            )
        })?;
        let mut output = vec![0.0_f32; input.len()];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
    }

    pub fn residual_add(&self, residual: &[f32], update: &[f32]) -> Result<Vec<f32>> {
        if residual.len() != update.len() {
            bail!("residual add length mismatch");
        }
        if residual.is_empty() {
            return Ok(Vec::new());
        }
        let elements = as_u32(residual.len(), "residual elements")?;
        let residual_buffer = self.buffer_f32(residual)?;
        let update_buffer = self.buffer_f32(update)?;
        let output_buffer = self.zeros_f32(residual.len())?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.residual_add,
                &[&residual_buffer, &update_buffer, &output_buffer],
                |encoder| Self::set_u32s(encoder, 3, &[elements]),
                residual.len(),
            )
        })?;
        let mut output = vec![0.0; residual.len()];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
    }

    pub fn time_shift_delta(
        &self,
        input: &[f32],
        rows: usize,
        time: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        if rows.checked_mul(channels) != Some(input.len())
            || time == 0
            || !rows.is_multiple_of(time)
        {
            bail!("time-shift input shape mismatch");
        }
        let input_buffer = self.buffer_f32(input)?;
        let output_buffer = self.zeros_f32(input.len())?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.time_shift_delta,
                &[&input_buffer, &output_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        2,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(time, "time")?,
                            as_u32(channels, "channels")?,
                        ],
                    )
                },
                input.len(),
            )
        })?;
        let mut output = vec![0.0; input.len()];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
    }

    pub fn sign_pack_bits(&self, input: &[f32]) -> Result<Vec<u32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let words = input.len().div_ceil(32);
        let input_buffer = self.buffer_f32(input)?;
        let bits_buffer = self.buffer_u32(&vec![0_u32; words])?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.sign_pack_bits,
                &[&input_buffer, &bits_buffer],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(input.len(), "elements")?]),
                words,
            )
        })?;
        let mut bits = vec![0_u32; words];
        bits_buffer.read_u32(&mut bits)?;
        Ok(bits)
    }

    /// Metal-resident `rosa_qkv_ref` over packed `[B, T, D]` bitplanes.
    ///
    /// One thread owns `(batch, channel)`. SAM arrays live in global i32
    /// buffers of shape `[B, D, 2T+1]`. `idx` is bit-exact with
    /// [`crate::rosa::rosa_qkv_batch`]; `out = (2·idx − 1)·e` uses FP16 `e`.
    pub fn rosa_qkv_1bit_fwd(
        &self,
        q_bits: &[u32],
        k_bits: &[u32],
        v_bits: &[u32],
        e: &[f32],
        batch: usize,
        time: usize,
        channels: usize,
    ) -> Result<RosaQkvForward> {
        let shape = MetalDispatchShape::new(batch, time, channels)?;
        if e.len() != channels {
            bail!("ROSA e must have one scale per channel");
        }
        let bit_count = shape.elements();
        let words = bit_count.div_ceil(32);
        if q_bits.len() != words || k_bits.len() != words || v_bits.len() != words {
            bail!("ROSA QKV bitplane word count mismatch");
        }
        let nodes = crate::rosa::sam_node_count(time);
        let sam_len = batch
            .checked_mul(channels)
            .and_then(|rows| rows.checked_mul(nodes))
            .ok_or_else(|| anyhow::anyhow!("ROSA SAM workspace overflow"))?;
        let sam_bytes = sam_len
            .checked_mul(size_of::<i32>())
            .ok_or_else(|| anyhow::anyhow!("ROSA SAM byte count overflow"))?;

        let q_buffer = self.buffer_u32(q_bits)?;
        let k_buffer = self.buffer_u32(k_bits)?;
        let v_buffer = self.buffer_u32(v_bits)?;
        let e_buffer = self.buffer_u16(&fp16_bits(e))?;
        let trans0 = self.alloc_bytes(sam_bytes)?;
        let trans1 = self.alloc_bytes(sam_bytes)?;
        let fail = self.alloc_bytes(sam_bytes)?;
        let maxlen = self.alloc_bytes(sam_bytes)?;
        let last = self.alloc_bytes(sam_bytes)?;
        let idx_buffer = self.alloc_bytes(bit_count)?;
        let out_buffer = self.alloc_f32(bit_count)?;
        let threads = batch
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("ROSA dispatch overflow"))?;

        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.rosa_sam_reset,
                &[&trans0, &trans1, &fail, &maxlen, &last],
                |encoder| set_bytes_u32(encoder, 5, &[as_u32(sam_len, "SAM nodes")?]),
                sam_len,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.rosa_qkv_1bit_fwd,
                &[
                    &q_buffer,
                    &k_buffer,
                    &v_buffer,
                    &e_buffer,
                    &trans0,
                    &trans1,
                    &fail,
                    &maxlen,
                    &last,
                    &idx_buffer,
                    &out_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        11,
                        &[
                            as_u32(batch, "batch")?,
                            as_u32(time, "time")?,
                            as_u32(channels, "channels")?,
                        ],
                    )
                },
                threads,
            )
        })?;

        let mut idx = vec![0_u8; bit_count];
        idx_buffer.read_bytes(&mut idx)?;
        let mut out = vec![0.0_f32; bit_count];
        out_buffer.read_f32(&mut out)?;
        Ok(RosaQkvForward { idx, out })
    }

    /// Exact `g_e[c] = Σ gy · (2·idx − 1)` including unmatched positions.
    pub fn rosa_qkv_1bit_bwd_e(
        &self,
        output_gradient: &[f32],
        idx: &[u8],
        batch: usize,
        time: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        let shape = MetalDispatchShape::new(batch, time, channels)?;
        if output_gradient.len() != shape.elements() || idx.len() != shape.elements() {
            bail!("ROSA g_e shape mismatch");
        }
        let gy_buffer = self.buffer_f32(output_gradient)?;
        let idx_buffer = self.shared_buffer(idx.len())?;
        idx_buffer.write_bytes(idx)?;
        let ge_buffer = self.zeros_f32(channels)?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.rosa_qkv_1bit_bwd_e,
                &[&gy_buffer, &idx_buffer, &ge_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        3,
                        &[
                            as_u32(batch, "batch")?,
                            as_u32(time, "time")?,
                            as_u32(channels, "channels")?,
                        ],
                    )
                },
                channels,
            )
        })?;
        let mut e_gradient = vec![0.0_f32; channels];
        ge_buffer.read_f32(&mut e_gradient)?;
        Ok(e_gradient)
    }

    fn encode_binary_linear<B: AsRef<MetalBuffer>>(
        encoder: &ffi::ComputeEncoder,
        pipeline: &Pipeline,
        input: B,
        bits: B,
        scale: B,
        bias: B,
        output: B,
        rows: usize,
        in_features: usize,
        out_features: usize,
        has_bias: bool,
    ) -> Result<()> {
        Self::encode_tiled(
            encoder,
            pipeline,
            &[input, bits, scale, bias, output],
            |encoder| {
                Self::set_u32s(
                    encoder,
                    5,
                    &[
                        as_u32(rows, "rows")?,
                        as_u32(in_features, "in")?,
                        as_u32(out_features, "out")?,
                        u32::from(has_bias),
                    ],
                )
            },
            out_features,
            rows,
        )
    }

    /// Time-mix + QKV packed linears + SAM + output projection in one command buffer.
    /// Activations never leave the GPU between those stages; only `idx`, `y`, and `out` are read.
    pub fn rosa_block_forward(
        &self,
        x: &[f32],
        mix_q: &[u16],
        mix_k: &[u16],
        mix_v: &[u16],
        q_bits: &[u32],
        q_scale: &[u16],
        q_bias: &[u16],
        k_bits: &[u32],
        k_scale: &[u16],
        k_bias: &[u16],
        v_bits: &[u32],
        v_scale: &[u16],
        v_bias: &[u16],
        e: &[u16],
        o_bits: &[u32],
        o_scale: &[u16],
        o_bias: &[u16],
        batch: usize,
        time: usize,
        channels: usize,
    ) -> Result<RosaQkvBlockForward> {
        let shape = MetalDispatchShape::new(batch, time, channels)?;
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("ROSA block shape overflow"))?;
        let linear = LinearDispatchShape::new(rows, channels, channels)?;
        if x.len() != shape.elements()
            || mix_q.len() != channels
            || mix_k.len() != channels
            || mix_v.len() != channels
            || e.len() != channels
            || q_scale.len() != channels
            || k_scale.len() != channels
            || v_scale.len() != channels
            || o_scale.len() != channels
            || q_bias.len() != channels
            || k_bias.len() != channels
            || v_bias.len() != channels
            || o_bias.len() != channels
        {
            bail!("ROSA block mix/scale/bias length mismatch");
        }
        self.check_binary_shape_u16(x, q_bits, q_scale, Some(q_bias), linear)?;
        self.check_binary_shape_u16(x, k_bits, k_scale, Some(k_bias), linear)?;
        self.check_binary_shape_u16(x, v_bits, v_scale, Some(v_bias), linear)?;
        self.check_binary_shape_u16(x, o_bits, o_scale, Some(o_bias), linear)?;
        let bit_count = shape.elements();
        let words = bit_count.div_ceil(32);

        let x_b = self.buffer_f32(x)?;
        let mix_q_b = self.buffer_u16(mix_q)?;
        let mix_k_b = self.buffer_u16(mix_k)?;
        let mix_v_b = self.buffer_u16(mix_v)?;
        let q_in = self.alloc_f32(bit_count)?;
        let k_in = self.alloc_f32(bit_count)?;
        let v_in = self.alloc_f32(bit_count)?;
        let q_bits_b = self.buffer_u32(q_bits)?;
        let k_bits_b = self.buffer_u32(k_bits)?;
        let v_bits_b = self.buffer_u32(v_bits)?;
        let o_bits_b = self.buffer_u32(o_bits)?;
        let q_scale_b = self.buffer_u16(q_scale)?;
        let k_scale_b = self.buffer_u16(k_scale)?;
        let v_scale_b = self.buffer_u16(v_scale)?;
        let o_scale_b = self.buffer_u16(o_scale)?;
        let q_bias_b = self.buffer_u16(q_bias)?;
        let k_bias_b = self.buffer_u16(k_bias)?;
        let v_bias_b = self.buffer_u16(v_bias)?;
        let o_bias_b = self.buffer_u16(o_bias)?;
        let q_act = self.alloc_f32(bit_count)?;
        let k_act = self.alloc_f32(bit_count)?;
        let v_act = self.alloc_f32(bit_count)?;
        let q_packed = self.alloc_bytes(words.saturating_mul(size_of::<u32>()).max(4))?;
        let k_packed = self.alloc_bytes(words.saturating_mul(size_of::<u32>()).max(4))?;
        let v_packed = self.alloc_bytes(words.saturating_mul(size_of::<u32>()).max(4))?;

        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.time_shift_mix3,
                &[&x_b, &mix_q_b, &mix_k_b, &mix_v_b, &q_in, &k_in, &v_in],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        7,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(time, "time")?,
                            as_u32(channels, "channels")?,
                        ],
                    )
                },
                bit_count,
            )?;
            Self::encode_binary_linear(
                encoder,
                &self.pipelines.binary_linear,
                &q_in,
                &q_bits_b,
                &q_scale_b,
                &q_bias_b,
                &q_act,
                rows,
                channels,
                channels,
                true,
            )?;
            Self::encode_binary_linear(
                encoder,
                &self.pipelines.binary_linear,
                &k_in,
                &k_bits_b,
                &k_scale_b,
                &k_bias_b,
                &k_act,
                rows,
                channels,
                channels,
                true,
            )?;
            Self::encode_binary_linear(
                encoder,
                &self.pipelines.binary_linear,
                &v_in,
                &v_bits_b,
                &v_scale_b,
                &v_bias_b,
                &v_act,
                rows,
                channels,
                channels,
                true,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.sign_pack_bits,
                &[&q_act, &q_packed],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(bit_count, "elements")?]),
                words,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.sign_pack_bits,
                &[&k_act, &k_packed],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(bit_count, "elements")?]),
                words,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.sign_pack_bits,
                &[&v_act, &v_packed],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(bit_count, "elements")?]),
                words,
            )
        })?;

        let mut q_plane = vec![0_u32; words];
        let mut k_plane = vec![0_u32; words];
        let mut v_plane = vec![0_u32; words];
        q_packed.read_u32(&mut q_plane)?;
        k_packed.read_u32(&mut k_plane)?;
        v_packed.read_u32(&mut v_plane)?;
        drop((
            x_b, mix_q_b, mix_k_b, mix_v_b, q_in, k_in, v_in, q_bits_b, k_bits_b, v_bits_b,
            q_scale_b, k_scale_b, v_scale_b, q_bias_b, k_bias_b, v_bias_b, q_act, k_act, v_act,
            q_packed, k_packed, v_packed,
        ));
        let idx = crate::rosa::rosa_qkv_batch_packed(
            &q_plane, &k_plane, &v_plane, batch, time, channels,
        )?;
        let e_f32: Vec<f32> = e
            .iter()
            .copied()
            .map(|bits| Fp16::from_bits(bits).to_f32())
            .collect();
        let y = crate::rosa::rosa_qkv_out_batched(&idx, &e_f32, batch, time, channels)?;
        let y_buffer = self.buffer_f32(&y)?;
        let out_buffer = self.alloc_f32(bit_count)?;
        self.submit(|encoder| {
            Self::encode_binary_linear(
                encoder,
                &self.pipelines.binary_linear,
                &y_buffer,
                &o_bits_b,
                &o_scale_b,
                &o_bias_b,
                &out_buffer,
                rows,
                channels,
                channels,
                true,
            )
        })?;
        let mut out = vec![0.0_f32; bit_count];
        out_buffer.read_f32(&mut out)?;
        Ok(RosaQkvBlockForward { idx, y, out })
    }

    pub fn cmix_relu2(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.elementwise_unary(&self.pipelines.cmix_relu2, input)
    }

    pub fn cmix_relu2_backward(&self, input: &[f32], output_gradient: &[f32]) -> Result<Vec<f32>> {
        if input.len() != output_gradient.len() {
            bail!("relu2 backward length mismatch");
        }
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let input_buffer = self.buffer_f32(input)?;
        let gy_buffer = self.buffer_f32(output_gradient)?;
        let gx_buffer = self.zeros_f32(input.len())?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.cmix_relu2_backward,
                &[&input_buffer, &gy_buffer, &gx_buffer],
                |encoder| set_bytes_u32(encoder, 3, &[as_u32(input.len(), "elements")?]),
                input.len(),
            )
        })?;
        let mut gx = vec![0.0; input.len()];
        gx_buffer.read_f32(&mut gx)?;
        Ok(gx)
    }

    fn elementwise_unary(&self, pipeline: &Pipeline, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let input_buffer = self.buffer_f32(input)?;
        let output_buffer = self.zeros_f32(input.len())?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                pipeline,
                &[&input_buffer, &output_buffer],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(input.len(), "elements")?]),
                input.len(),
            )
        })?;
        let mut output = vec![0.0; input.len()];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
    }

    pub fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        if rows.checked_mul(channels) != Some(input.len())
            || weight.len() != channels
            || bias.len() != channels
        {
            bail!("LayerNorm input shape mismatch");
        }
        let input_buffer = self.buffer_f32(input)?;
        let weight_buffer = self.buffer_u16(&fp16_bits(weight))?;
        let bias_buffer = self.buffer_u16(&fp16_bits(bias))?;
        let output_buffer = self.zeros_f32(input.len())?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.layer_norm,
                &[&input_buffer, &weight_buffer, &bias_buffer, &output_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[as_u32(rows, "rows")?, as_u32(channels, "channels")?],
                    )
                },
                rows,
            )
        })?;
        let mut output = vec![0.0; input.len()];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
    }

    pub fn layer_norm_backward(
        &self,
        input: &[f32],
        output_gradient: &[f32],
        weight: &[f32],
        rows: usize,
        channels: usize,
    ) -> Result<LayerNormBackward> {
        if rows.checked_mul(channels) != Some(input.len())
            || output_gradient.len() != input.len()
            || weight.len() != channels
        {
            bail!("LayerNorm backward shape mismatch");
        }
        let input_buffer = self.buffer_f32(input)?;
        let gy_buffer = self.buffer_f32(output_gradient)?;
        let weight_buffer = self.buffer_u16(&fp16_bits(weight))?;
        let gx_buffer = self.alloc_f32(input.len())?;
        let mean_buffer = self.alloc_f32(rows)?;
        let inv_buffer = self.alloc_f32(rows)?;
        let gw_buffer = self.alloc_f32(channels)?;
        let gb_buffer = self.alloc_f32(channels)?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.layer_norm_backward,
                &[
                    &input_buffer,
                    &gy_buffer,
                    &weight_buffer,
                    &gx_buffer,
                    &mean_buffer,
                    &inv_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        6,
                        &[as_u32(rows, "rows")?, as_u32(channels, "channels")?],
                    )
                },
                rows,
            )?;
            Self::encode_scale_groups(
                encoder,
                &self.pipelines.layer_norm_param_bwd,
                &[
                    &input_buffer,
                    &gy_buffer,
                    &mean_buffer,
                    &inv_buffer,
                    &gw_buffer,
                    &gb_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        6,
                        &[as_u32(rows, "rows")?, as_u32(channels, "channels")?],
                    )
                },
                channels,
            )
        })?;
        let mut input_gradient = vec![0.0; input.len()];
        let mut weight_gradient = vec![0.0; channels];
        let mut bias_gradient = vec![0.0; channels];
        gx_buffer.read_f32(&mut input_gradient)?;
        gw_buffer.read_f32(&mut weight_gradient)?;
        gb_buffer.read_f32(&mut bias_gradient)?;
        Ok(LayerNormBackward {
            input_gradient,
            weight_gradient,
            bias_gradient,
        })
    }

    pub fn binary_linear(
        &self,
        input: &[f32],
        bits: &[u32],
        scale: &[f32],
        bias: Option<&[f32]>,
        shape: LinearDispatchShape,
    ) -> Result<Vec<f32>> {
        self.check_binary_shape(input, bits, scale, bias, shape)?;
        let out_len = shape.rows.saturating_mul(shape.out_features);
        let input_buffer = self.buffer_f32(input)?;
        let bits_buffer = self.buffer_u32(bits)?;
        let scale_buffer = self.buffer_u16(&fp16_bits(scale))?;
        let bias_buffer = self.buffer_u16(&fp16_bits(bias.unwrap_or(&[])))?;
        let output_buffer = self.zeros_f32(out_len)?;
        let has_bias = u32::from(bias.is_some());
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear,
                &[
                    &input_buffer,
                    &bits_buffer,
                    &scale_buffer,
                    &bias_buffer,
                    &output_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                            has_bias,
                        ],
                    )
                },
                shape.out_features,
                shape.rows,
            )
        })?;
        let mut output = vec![0.0; out_len];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
    }

    pub fn binary_linear_pack3(
        &self,
        inputs: [&[f32]; 3],
        bits: [&[u32]; 3],
        scales: [&[f32]; 3],
        biases: [Option<&[f32]>; 3],
        shape: LinearDispatchShape,
    ) -> Result<[Vec<f32>; 3]> {
        let out_len = shape.rows.saturating_mul(shape.out_features);
        let mut in_bufs = Vec::with_capacity(3);
        let mut bit_bufs = Vec::with_capacity(3);
        let mut scale_bufs = Vec::with_capacity(3);
        let mut bias_bufs = Vec::with_capacity(3);
        let mut out_bufs = Vec::with_capacity(3);
        for i in 0..3 {
            self.check_binary_shape(inputs[i], bits[i], scales[i], biases[i], shape)?;
            in_bufs.push(self.buffer_f32(inputs[i])?);
            bit_bufs.push(self.buffer_u32(bits[i])?);
            scale_bufs.push(self.buffer_u16(&fp16_bits(scales[i]))?);
            bias_bufs.push(self.buffer_u16(&fp16_bits(biases[i].unwrap_or(&[])))?);
            out_bufs.push(self.alloc_f32(out_len)?);
        }
        self.submit(|encoder| {
            for i in 0..3 {
                Self::encode_tiled(
                    encoder,
                    &self.pipelines.binary_linear,
                    &[
                        &in_bufs[i],
                        &bit_bufs[i],
                        &scale_bufs[i],
                        &bias_bufs[i],
                        &out_bufs[i],
                    ],
                    |encoder| {
                        Self::set_u32s(
                            encoder,
                            5,
                            &[
                                as_u32(shape.rows, "rows")?,
                                as_u32(shape.in_features, "in")?,
                                as_u32(shape.out_features, "out")?,
                                u32::from(biases[i].is_some()),
                            ],
                        )
                    },
                    shape.out_features,
                    shape.rows,
                )?;
            }
            Ok(())
        })?;
        let mut outs = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0..3 {
            outs[i] = vec![0.0; out_len];
            out_bufs[i].read_f32(&mut outs[i])?;
        }
        Ok(outs)
    }

    pub fn cmix_ffn_forward(
        &self,
        shifted: &[f32],
        key_bits: &[u32],
        key_scale: &[f32],
        value_weight: &[f32],
        rows: usize,
        d_model: usize,
        dim_ffn: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let key_shape = LinearDispatchShape::new(rows, d_model, dim_ffn)?;
        let value_shape = LinearDispatchShape::new(rows, dim_ffn, d_model)?;
        self.check_binary_shape(shifted, key_bits, key_scale, None, key_shape)?;
        let key_len = rows.saturating_mul(dim_ffn);
        let out_len = rows.saturating_mul(d_model);
        let shifted_b = self.buffer_f32(shifted)?;
        let bits_b = self.buffer_u32(key_bits)?;
        let scale_b = self.buffer_u16(&fp16_bits(key_scale))?;
        let bias_b = self.buffer_u16(&fp16_bits(&[]))?;
        let value_w_b = self.buffer_u16(&fp16_bits(value_weight))?;
        let key_b = self.alloc_f32(key_len)?;
        let relu_b = self.alloc_f32(key_len)?;
        let out_b = self.alloc_f32(out_len)?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear,
                &[&shifted_b, &bits_b, &scale_b, &bias_b, &key_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(d_model, "in")?,
                            as_u32(dim_ffn, "out")?,
                            0,
                        ],
                    )
                },
                dim_ffn,
                rows,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.cmix_relu2,
                &[&key_b, &relu_b],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(key_len, "elements")?]),
                key_len,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear,
                &[&relu_b, &value_w_b, &out_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        3,
                        &[
                            as_u32(value_shape.rows, "rows")?,
                            as_u32(value_shape.in_features, "in")?,
                            as_u32(value_shape.out_features, "out")?,
                        ],
                    )
                },
                value_shape.out_features,
                value_shape.rows,
            )
        })?;
        let mut key = vec![0.0; key_len];
        let mut relu2 = vec![0.0; key_len];
        let mut out = vec![0.0; out_len];
        key_b.read_f32(&mut key)?;
        relu_b.read_f32(&mut relu2)?;
        out_b.read_f32(&mut out)?;
        Ok((key, relu2, out))
    }

    pub fn cmix_block_forward(
        &self,
        x: &[f32],
        mix: &[u16],
        key_bits: &[u32],
        key_scale: &[u16],
        value_weight: &[u16],
        batch: usize,
        time: usize,
        d_model: usize,
        dim_ffn: usize,
    ) -> Result<CmixBlockForward> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("CMix shape overflow"))?;
        let key_shape = LinearDispatchShape::new(rows, d_model, dim_ffn)?;
        let value_shape = LinearDispatchShape::new(rows, dim_ffn, d_model)?;
        if x.len() != rows.saturating_mul(d_model)
            || mix.len() != d_model
            || key_scale.len() != dim_ffn
            || value_weight.len() != d_model.saturating_mul(dim_ffn)
        {
            bail!("CMix block shape mismatch");
        }
        self.check_binary_shape_u16(x, key_bits, key_scale, None, key_shape)?;
        let key_len = rows.saturating_mul(dim_ffn);
        let out_len = rows.saturating_mul(d_model);
        let x_b = self.buffer_f32(x)?;
        let mix_b = self.buffer_u16(mix)?;
        let xx_b = self.alloc_f32(x.len())?;
        let shifted_b = self.alloc_f32(x.len())?;
        let bits_b = self.buffer_u32(key_bits)?;
        let scale_b = self.buffer_u16(key_scale)?;
        let bias_b = self.buffer_u16(&[])?;
        let value_w_b = self.buffer_u16(value_weight)?;
        let key_b = self.alloc_f32(key_len)?;
        let relu_b = self.alloc_f32(key_len)?;
        let out_b = self.alloc_f32(out_len)?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.time_shift_mix,
                &[&x_b, &mix_b, &xx_b, &shifted_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(time, "time")?,
                            as_u32(d_model, "channels")?,
                        ],
                    )
                },
                x.len(),
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear,
                &[&shifted_b, &bits_b, &scale_b, &bias_b, &key_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(d_model, "in")?,
                            as_u32(dim_ffn, "out")?,
                            0,
                        ],
                    )
                },
                dim_ffn,
                rows,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.cmix_relu2,
                &[&key_b, &relu_b],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(key_len, "elements")?]),
                key_len,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear,
                &[&relu_b, &value_w_b, &out_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        3,
                        &[
                            as_u32(value_shape.rows, "rows")?,
                            as_u32(value_shape.in_features, "in")?,
                            as_u32(value_shape.out_features, "out")?,
                        ],
                    )
                },
                value_shape.out_features,
                value_shape.rows,
            )
        })?;
        let mut xx = vec![0.0; x.len()];
        let mut shifted = vec![0.0; x.len()];
        let mut key = vec![0.0; key_len];
        let mut relu2 = vec![0.0; key_len];
        let mut out = vec![0.0; out_len];
        xx_b.read_f32(&mut xx)?;
        shifted_b.read_f32(&mut shifted)?;
        key_b.read_f32(&mut key)?;
        relu_b.read_f32(&mut relu2)?;
        out_b.read_f32(&mut out)?;
        Ok(CmixBlockForward {
            xx,
            shifted,
            key,
            relu2,
            out,
        })
    }

    pub fn cmix_ffn_backward(
        &self,
        shifted: &[f32],
        key: &[f32],
        relu2: &[f32],
        gy: &[f32],
        key_bits: &[u32],
        key_scale: &[f32],
        value_weight: &[f32],
        rows: usize,
        d_model: usize,
        dim_ffn: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let key_shape = LinearDispatchShape::new(rows, d_model, dim_ffn)?;
        let value_shape = LinearDispatchShape::new(rows, dim_ffn, d_model)?;
        self.check_binary_shape(shifted, key_bits, key_scale, None, key_shape)?;
        let key_len = rows.saturating_mul(dim_ffn);
        let value_weights = d_model.saturating_mul(dim_ffn);
        let key_weights = dim_ffn.saturating_mul(d_model);
        if relu2.len() != key_len
            || key.len() != key_len
            || gy.len() != rows.saturating_mul(d_model)
        {
            bail!("CMix backward shape mismatch");
        }
        let shifted_b = self.buffer_f32(shifted)?;
        let key_b = self.buffer_f32(key)?;
        let relu_b = self.buffer_f32(relu2)?;
        let gy_b = self.buffer_f32(gy)?;
        let bits_b = self.buffer_u32(key_bits)?;
        let scale_b = self.buffer_u16(&fp16_bits(key_scale))?;
        let value_w_b = self.buffer_u16(&fp16_bits(value_weight))?;
        let g_relu_b = self.alloc_f32(key_len)?;
        let g_value_b = self.alloc_f32(value_weights)?;
        let g_key_b = self.alloc_f32(key_len)?;
        let g_shifted_b = self.alloc_f32(shifted.len())?;
        let g_scale_b = self.alloc_f32(dim_ffn)?;
        let g_bias_b = self.alloc_f32(dim_ffn)?;
        let g_key_w_b = self.alloc_f32(key_weights)?;
        let empty_bias_b = self.buffer_u16(&fp16_bits(&[]))?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear_bwd,
                &[&relu_b, &gy_b, &value_w_b, &g_relu_b, &g_value_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(value_shape.rows, "rows")?,
                            as_u32(value_shape.in_features, "in")?,
                            as_u32(value_shape.out_features, "out")?,
                            0,
                        ],
                    )
                },
                value_shape.in_features,
                value_shape.rows,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear_bwd,
                &[&relu_b, &gy_b, &value_w_b, &g_relu_b, &g_value_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(value_shape.rows, "rows")?,
                            as_u32(value_shape.in_features, "in")?,
                            as_u32(value_shape.out_features, "out")?,
                            1,
                        ],
                    )
                },
                value_shape.in_features,
                value_shape.out_features,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.cmix_relu2_backward,
                &[&key_b, &g_relu_b, &g_key_b],
                |encoder| set_bytes_u32(encoder, 3, &[as_u32(key_len, "elements")?]),
                key_len,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_input_bwd,
                &[&g_key_b, &bits_b, &scale_b, &g_shifted_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(key_shape.rows, "rows")?,
                            as_u32(key_shape.in_features, "in")?,
                            as_u32(key_shape.out_features, "out")?,
                        ],
                    )
                },
                key_shape.in_features,
                key_shape.rows,
            )?;
            Self::encode_scale_groups(
                encoder,
                &self.pipelines.binary_linear_scale_bwd_from_output,
                &[
                    &key_b,
                    &g_key_b,
                    &scale_b,
                    &empty_bias_b,
                    &g_scale_b,
                    &g_bias_b,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        6,
                        &[
                            as_u32(key_shape.rows, "rows")?,
                            as_u32(key_shape.out_features, "out")?,
                            0,
                        ],
                    )
                },
                dim_ffn,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_weight_bwd,
                &[&shifted_b, &g_key_b, &scale_b, &g_key_w_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(key_shape.rows, "rows")?,
                            as_u32(key_shape.in_features, "in")?,
                            as_u32(key_shape.out_features, "out")?,
                        ],
                    )
                },
                key_shape.in_features,
                key_shape.out_features,
            )
        })?;
        let mut g_shifted = vec![0.0; shifted.len()];
        let mut g_key_w = vec![0.0; key_weights];
        let mut g_key_scale = vec![0.0; dim_ffn];
        let mut g_value_w = vec![0.0; value_weights];
        g_shifted_b.read_f32(&mut g_shifted)?;
        g_key_w_b.read_f32(&mut g_key_w)?;
        g_scale_b.read_f32(&mut g_key_scale)?;
        g_value_b.read_f32(&mut g_value_w)?;
        Ok((g_shifted, g_key_w, g_key_scale, g_value_w))
    }

    pub fn cmix_block_backward_sgd(
        &self,
        shifted: &[f32],
        key: &[f32],
        relu2: &[f32],
        gy: &[f32],
        key_bits: &[u32],
        key_scale: &[u16],
        key_latent: &[u16],
        key_residual: &[f32],
        value_weight: &[u16],
        value_residual: &[f32],
        rows: usize,
        d_model: usize,
        dim_ffn: usize,
        learning_rate: f32,
        ste_scale: f32,
    ) -> Result<CmixBlockBackwardSgd> {
        let key_shape = LinearDispatchShape::new(rows, d_model, dim_ffn)?;
        let value_shape = LinearDispatchShape::new(rows, dim_ffn, d_model)?;
        self.check_binary_shape_u16(shifted, key_bits, key_scale, None, key_shape)?;
        let key_len = rows.saturating_mul(dim_ffn);
        let value_weights = d_model.saturating_mul(dim_ffn);
        let key_weights = dim_ffn.saturating_mul(d_model);
        if relu2.len() != key_len
            || key.len() != key_len
            || gy.len() != rows.saturating_mul(d_model)
            || key_latent.len() != key_weights
            || key_residual.len() != key_weights
            || value_weight.len() != value_weights
            || value_residual.len() != value_weights
        {
            bail!("CMix backward SGD shape mismatch");
        }
        let key_words = key_weights.div_ceil(32);
        let shifted_b = self.buffer_f32(shifted)?;
        let key_b = self.buffer_f32(key)?;
        let relu_b = self.buffer_f32(relu2)?;
        let gy_b = self.buffer_f32(gy)?;
        let bits_b = self.buffer_u32(key_bits)?;
        let scale_b = self.buffer_u16(key_scale)?;
        let value_w_b = self.buffer_u16(value_weight)?;
        let value_residual_b = self.buffer_f32(value_residual)?;
        let key_latent_b = self.buffer_u16(key_latent)?;
        let key_residual_b = self.buffer_f32(key_residual)?;
        let g_relu_b = self.alloc_f32(key_len)?;
        let g_value_b = self.alloc_f32(value_weights)?;
        let g_key_b = self.alloc_f32(key_len)?;
        let g_shifted_b = self.alloc_f32(shifted.len())?;
        let g_scale_b = self.alloc_f32(dim_ffn)?;
        let g_bias_b = self.alloc_f32(dim_ffn)?;
        let g_key_w_b = self.alloc_f32(key_weights)?;
        let empty_bias_b = self.buffer_u16(&[])?;
        let next_key_bits = self.alloc_bytes(key_words.saturating_mul(size_of::<u32>()).max(4))?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear_bwd,
                &[&relu_b, &gy_b, &value_w_b, &g_relu_b, &g_value_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(value_shape.rows, "rows")?,
                            as_u32(value_shape.in_features, "in")?,
                            as_u32(value_shape.out_features, "out")?,
                            0,
                        ],
                    )
                },
                value_shape.in_features,
                value_shape.rows,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear_bwd,
                &[&relu_b, &gy_b, &value_w_b, &g_relu_b, &g_value_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(value_shape.rows, "rows")?,
                            as_u32(value_shape.in_features, "in")?,
                            as_u32(value_shape.out_features, "out")?,
                            1,
                        ],
                    )
                },
                value_shape.in_features,
                value_shape.out_features,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.cmix_relu2_backward,
                &[&key_b, &g_relu_b, &g_key_b],
                |encoder| set_bytes_u32(encoder, 3, &[as_u32(key_len, "elements")?]),
                key_len,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_input_bwd,
                &[&g_key_b, &bits_b, &scale_b, &g_shifted_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(key_shape.rows, "rows")?,
                            as_u32(key_shape.in_features, "in")?,
                            as_u32(key_shape.out_features, "out")?,
                        ],
                    )
                },
                key_shape.in_features,
                key_shape.rows,
            )?;
            Self::encode_scale_groups(
                encoder,
                &self.pipelines.binary_linear_scale_bwd_from_output,
                &[
                    &key_b,
                    &g_key_b,
                    &scale_b,
                    &empty_bias_b,
                    &g_scale_b,
                    &g_bias_b,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        6,
                        &[
                            as_u32(key_shape.rows, "rows")?,
                            as_u32(key_shape.out_features, "out")?,
                            0,
                        ],
                    )
                },
                dim_ffn,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_weight_bwd,
                &[&shifted_b, &g_key_b, &scale_b, &g_key_w_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(key_shape.rows, "rows")?,
                            as_u32(key_shape.in_features, "in")?,
                            as_u32(key_shape.out_features, "out")?,
                        ],
                    )
                },
                key_shape.in_features,
                key_shape.out_features,
            )?;
            Self::encode_clipped_sgd_fp16(
                encoder,
                &self.pipelines.clipped_sgd_fp16,
                &value_w_b,
                &value_residual_b,
                &g_value_b,
                learning_rate,
                value_weights,
            )?;
            Self::encode_binaryconnect_sgd_fp16(
                encoder,
                &self.pipelines.binaryconnect_sgd_fp16,
                &key_latent_b,
                &key_residual_b,
                &g_key_w_b,
                learning_rate,
                ste_scale,
                key_weights,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.pack_latent_bits,
                &[&key_latent_b, &next_key_bits],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(key_weights, "elements")?]),
                key_words,
            )
        })?;
        let mut g_shifted = vec![0.0; shifted.len()];
        let mut g_key_scale = vec![0.0; dim_ffn];
        let mut next_value = vec![0_u16; value_weights];
        let mut next_value_residual = vec![0.0; value_weights];
        let mut next_key_latent = vec![0_u16; key_weights];
        let mut next_key_residual = vec![0.0; key_weights];
        let mut packed = vec![0_u32; key_words];
        g_shifted_b.read_f32(&mut g_shifted)?;
        g_scale_b.read_f32(&mut g_key_scale)?;
        value_w_b.read_u16(&mut next_value)?;
        value_residual_b.read_f32(&mut next_value_residual)?;
        key_latent_b.read_u16(&mut next_key_latent)?;
        key_residual_b.read_f32(&mut next_key_residual)?;
        next_key_bits.read_u32(&mut packed)?;
        Ok(CmixBlockBackwardSgd {
            g_shifted,
            g_key_scale,
            next_key_latent,
            next_key_residual,
            next_key_bits: packed,
            next_value_weight: next_value,
            next_value_residual,
        })
    }

    pub fn rosa_o_stop_grad_sgd(
        &self,
        y: &[f32],
        out: &[f32],
        gy: &[f32],
        idx: &[u8],
        o_bits: &[u32],
        o_scale: &[u16],
        o_bias: &[u16],
        latent: &[u16],
        residual: &[f32],
        rows: usize,
        channels: usize,
        learning_rate: f32,
        ste_scale: f32,
    ) -> Result<RosaOStopGradSgd> {
        let shape = LinearDispatchShape::new(rows, channels, channels)?;
        self.check_binary_shape_u16(y, o_bits, o_scale, Some(o_bias), shape)?;
        if out.len() != y.len()
            || gy.len() != y.len()
            || idx.len() != y.len()
            || latent.len() != channels.saturating_mul(channels)
            || residual.len() != latent.len()
        {
            bail!("ROSA O stop-grad shape mismatch");
        }
        let weights = channels.saturating_mul(channels);
        let words = weights.div_ceil(32);
        let y_b = self.buffer_f32(y)?;
        let out_b = self.buffer_f32(out)?;
        let gy_b = self.buffer_f32(gy)?;
        let idx_b = self.alloc_bytes(idx.len().max(1))?;
        idx_b.write_bytes(idx)?;
        let bits_b = self.buffer_u32(o_bits)?;
        let scale_b = self.buffer_u16(o_scale)?;
        let bias_b = self.buffer_u16(o_bias)?;
        let latent_b = self.buffer_u16(latent)?;
        let residual_b = self.buffer_f32(residual)?;
        let g_y_b = self.alloc_f32(y.len())?;
        let g_scale_b = self.alloc_f32(channels)?;
        let g_bias_b = self.alloc_f32(channels)?;
        let gw_b = self.alloc_f32(weights)?;
        let ge_b = self.alloc_f32(channels)?;
        let next_bits = self.alloc_bytes(words.saturating_mul(size_of::<u32>()).max(4))?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_input_bwd,
                &[&gy_b, &bits_b, &scale_b, &g_y_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(channels, "in")?,
                            as_u32(channels, "out")?,
                        ],
                    )
                },
                channels,
                rows,
            )?;
            Self::encode_scale_groups(
                encoder,
                &self.pipelines.binary_linear_scale_bwd_from_output,
                &[&out_b, &gy_b, &scale_b, &bias_b, &g_scale_b, &g_bias_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        6,
                        &[as_u32(rows, "rows")?, as_u32(channels, "out")?, 1],
                    )
                },
                channels,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_weight_bwd,
                &[&y_b, &gy_b, &scale_b, &gw_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(channels, "in")?,
                            as_u32(channels, "out")?,
                        ],
                    )
                },
                channels,
                channels,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.rosa_qkv_1bit_bwd_e,
                &[&g_y_b, &idx_b, &ge_b],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        3,
                        &[1, as_u32(rows, "time")?, as_u32(channels, "channels")?],
                    )
                },
                channels,
            )?;
            Self::encode_binaryconnect_sgd_fp16(
                encoder,
                &self.pipelines.binaryconnect_sgd_fp16,
                &latent_b,
                &residual_b,
                &gw_b,
                learning_rate,
                ste_scale,
                weights,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.pack_latent_bits,
                &[&latent_b, &next_bits],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(weights, "elements")?]),
                words,
            )
        })?;
        let mut g_e = vec![0.0; channels];
        let mut g_scale = vec![0.0; channels];
        let mut g_bias = vec![0.0; channels];
        let mut next_latent = vec![0_u16; weights];
        let mut next_residual = vec![0.0; weights];
        let mut packed = vec![0_u32; words];
        ge_b.read_f32(&mut g_e)?;
        g_scale_b.read_f32(&mut g_scale)?;
        g_bias_b.read_f32(&mut g_bias)?;
        latent_b.read_u16(&mut next_latent)?;
        residual_b.read_f32(&mut next_residual)?;
        next_bits.read_u32(&mut packed)?;
        Ok(RosaOStopGradSgd {
            e_gradient: g_e,
            scale_gradient: g_scale,
            bias_gradient: g_bias,
            next_latent,
            next_residual,
            next_bits: packed,
        })
    }

    pub fn binary_linear_backward(
        &self,
        input: &[f32],
        output_gradient: &[f32],
        bits: &[u32],
        scale: &[f32],
        has_bias: bool,
        shape: LinearDispatchShape,
    ) -> Result<BinaryLinearBackward> {
        self.check_binary_shape(input, bits, scale, None, shape)?;
        let out_len = shape.rows.saturating_mul(shape.out_features);
        if output_gradient.len() != out_len {
            bail!("binary linear output-gradient shape mismatch");
        }
        let input_buffer = self.buffer_f32(input)?;
        let gy_buffer = self.buffer_f32(output_gradient)?;
        let bits_buffer = self.buffer_u32(bits)?;
        let scale_buffer = self.buffer_u16(&fp16_bits(scale))?;
        let gx_buffer = self.alloc_f32(input.len())?;
        let g_scale_buffer = self.alloc_f32(shape.out_features)?;
        let g_bias_buffer = self.alloc_f32(shape.out_features)?;
        let weights = shape.out_features.saturating_mul(shape.in_features);
        let gw_buffer = self.alloc_f32(weights)?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_input_bwd,
                &[&gy_buffer, &bits_buffer, &scale_buffer, &gx_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                        ],
                    )
                },
                shape.in_features,
                shape.rows,
            )?;
            Self::encode_scale_groups(
                encoder,
                &self.pipelines.binary_linear_scale_bwd,
                &[
                    &input_buffer,
                    &gy_buffer,
                    &bits_buffer,
                    &g_scale_buffer,
                    &g_bias_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                            u32::from(has_bias),
                        ],
                    )
                },
                shape.out_features,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_weight_bwd,
                &[&input_buffer, &gy_buffer, &scale_buffer, &gw_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                        ],
                    )
                },
                shape.in_features,
                shape.out_features,
            )
        })?;
        let mut input_gradient = vec![0.0; input.len()];
        let mut scale_gradient = vec![0.0; shape.out_features];
        let mut weight_gradient = vec![0.0; weights];
        gx_buffer.read_f32(&mut input_gradient)?;
        g_scale_buffer.read_f32(&mut scale_gradient)?;
        gw_buffer.read_f32(&mut weight_gradient)?;
        let bias_gradient = if has_bias {
            let mut values = vec![0.0; shape.out_features];
            g_bias_buffer.read_f32(&mut values)?;
            Some(values)
        } else {
            None
        };
        Ok(BinaryLinearBackward {
            input_gradient,
            scale_gradient,
            bias_gradient,
            weight_gradient,
        })
    }

    pub fn binary_linear_weight_bwd(
        &self,
        input: &[f32],
        output_gradient: &[f32],
        scale: &[f32],
        shape: LinearDispatchShape,
    ) -> Result<Vec<f32>> {
        let weights = shape.out_features.saturating_mul(shape.in_features);
        if input.len() != shape.rows.saturating_mul(shape.in_features)
            || output_gradient.len() != shape.rows.saturating_mul(shape.out_features)
            || scale.len() != shape.out_features
        {
            bail!("binary linear weight-gradient shape mismatch");
        }
        let input_buffer = self.buffer_f32(input)?;
        let gy_buffer = self.buffer_f32(output_gradient)?;
        let scale_buffer = self.buffer_u16(&fp16_bits(scale))?;
        let gw_buffer = self.alloc_f32(weights)?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_weight_bwd,
                &[&input_buffer, &gy_buffer, &scale_buffer, &gw_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                        ],
                    )
                },
                shape.in_features,
                shape.out_features,
            )
        })?;
        let mut weight_gradient = vec![0.0; weights];
        gw_buffer.read_f32(&mut weight_gradient)?;
        Ok(weight_gradient)
    }

    pub fn binary_linear_latent_sgd(
        &self,
        latent: &[u16],
        input: &[f32],
        output_gradient: &[f32],
        scale: &[f32],
        learning_rate: f32,
        shape: LinearDispatchShape,
    ) -> Result<(Vec<u16>, Vec<u32>, Vec<f32>)> {
        let weights = shape
            .out_features
            .checked_mul(shape.in_features)
            .ok_or_else(|| anyhow::anyhow!("binary latent shape overflow"))?;
        if latent.len() != weights
            || input.len() != shape.rows.saturating_mul(shape.in_features)
            || output_gradient.len() != shape.rows.saturating_mul(shape.out_features)
            || scale.len() != shape.out_features
        {
            bail!("binary latent SGD shape mismatch");
        }
        let words = weights.div_ceil(32);
        let latent_buffer = self.buffer_u16(latent)?;
        let bits_buffer = self.buffer_u32(&vec![0_u32; words])?;
        let input_buffer = self.buffer_f32(input)?;
        let gy_buffer = self.buffer_f32(output_gradient)?;
        let scale_buffer = self.buffer_u16(&fp16_bits(scale))?;
        let gw_buffer = self.zeros_f32(weights)?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.binary_linear_latent_sgd,
                &[
                    &latent_buffer,
                    &bits_buffer,
                    &input_buffer,
                    &gy_buffer,
                    &scale_buffer,
                    &gw_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        6,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                        ],
                    )?;
                    set_bytes_f32(encoder, 9, &[learning_rate])
                },
                words,
            )
        })?;
        let mut next_latent = vec![0_u16; weights];
        let mut bits = vec![0_u32; words];
        let mut g_w = vec![0.0; weights];
        latent_buffer.read_u16(&mut next_latent)?;
        bits_buffer.read_u32(&mut bits)?;
        gw_buffer.read_f32(&mut g_w)?;
        Ok((next_latent, bits, g_w))
    }

    pub fn fp16_linear(
        &self,
        input: &[f32],
        weight: &[f32],
        shape: LinearDispatchShape,
    ) -> Result<Vec<f32>> {
        let weights = shape.out_features.saturating_mul(shape.in_features);
        if input.len() != shape.rows.saturating_mul(shape.in_features) || weight.len() != weights {
            bail!("FP16 linear shape mismatch");
        }
        let out_len = shape.rows.saturating_mul(shape.out_features);
        let input_buffer = self.buffer_f32(input)?;
        let weight_buffer = self.buffer_u16(&fp16_bits(weight))?;
        let output_buffer = self.zeros_f32(out_len)?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear,
                &[&input_buffer, &weight_buffer, &output_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        3,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                        ],
                    )
                },
                shape.out_features,
                shape.rows,
            )
        })?;
        let mut output = vec![0.0; out_len];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
    }

    pub fn fp16_linear_backward(
        &self,
        input: &[f32],
        output_gradient: &[f32],
        weight: &[f32],
        shape: LinearDispatchShape,
    ) -> Result<Fp16LinearBackward> {
        let weights = shape.out_features.saturating_mul(shape.in_features);
        if input.len() != shape.rows.saturating_mul(shape.in_features)
            || output_gradient.len() != shape.rows.saturating_mul(shape.out_features)
            || weight.len() != weights
        {
            bail!("FP16 linear backward shape mismatch");
        }
        let input_buffer = self.buffer_f32(input)?;
        let gy_buffer = self.buffer_f32(output_gradient)?;
        let weight_buffer = self.buffer_u16(&fp16_bits(weight))?;
        let gx_buffer = self.zeros_f32(input.len())?;
        let gw_buffer = self.zeros_f32(weights)?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear_bwd,
                &[
                    &input_buffer,
                    &gy_buffer,
                    &weight_buffer,
                    &gx_buffer,
                    &gw_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                            0,
                        ],
                    )
                },
                shape.in_features,
                shape.rows,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.fp16_linear_bwd,
                &[
                    &input_buffer,
                    &gy_buffer,
                    &weight_buffer,
                    &gx_buffer,
                    &gw_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(shape.rows, "rows")?,
                            as_u32(shape.in_features, "in")?,
                            as_u32(shape.out_features, "out")?,
                            1,
                        ],
                    )
                },
                shape.in_features,
                shape.out_features,
            )
        })?;
        let mut input_gradient = vec![0.0; input.len()];
        let mut weight_gradient = vec![0.0; weights];
        gx_buffer.read_f32(&mut input_gradient)?;
        gw_buffer.read_f32(&mut weight_gradient)?;
        Ok(Fp16LinearBackward {
            input_gradient,
            weight_gradient,
        })
    }

    fn causal_supervised_count(
        tokens: &[u32],
        time: usize,
        horizon: usize,
        ignore_id: u32,
    ) -> usize {
        (0..tokens.len())
            .filter(|&row| causal_ce_row_valid(row, time, horizon, tokens, ignore_id))
            .count()
    }

    pub fn streamed_cross_entropy_fp16(
        &self,
        hidden: &[f32],
        bits: &[u32],
        scale: &[f32],
        tokens: &[u32],
        rows: usize,
        time: usize,
        channels: usize,
        vocab: usize,
        horizon: usize,
    ) -> Result<StreamedCrossEntropy> {
        if rows.checked_mul(channels) != Some(hidden.len())
            || tokens.len() != rows
            || scale.len() != vocab
            || time == 0
            || !rows.is_multiple_of(time)
            || bits.len() != (vocab.saturating_mul(channels)).div_ceil(32)
        {
            bail!("streamed CE shape mismatch");
        }
        let n_valid = Self::causal_supervised_count(tokens, time, horizon, CE_NO_IGNORE);
        let gradient_scale = causal_ce_gradient_scale(n_valid, time);
        let hidden_buffer = self.buffer_f32(hidden)?;
        let bits_buffer = self.buffer_u32(bits)?;
        let scale_buffer = self.buffer_u16(&fp16_bits(scale))?;
        let tokens_buffer = self.buffer_u32(tokens)?;
        let gx_buffer = self.alloc_f32(hidden.len())?;
        let loss_buffer = self.alloc_f32(rows)?;
        let g_scale_buffer = self.zeros_f32(vocab)?;
        let gy_buffer = self.alloc_f32(rows.saturating_mul(vocab))?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.streamed_cross_entropy_fp16,
                &[
                    &hidden_buffer,
                    &bits_buffer,
                    &scale_buffer,
                    &tokens_buffer,
                    &gx_buffer,
                    &loss_buffer,
                    &g_scale_buffer,
                    &gy_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        8,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(time, "time")?,
                            as_u32(channels, "channels")?,
                            as_u32(vocab, "vocab")?,
                            as_u32(horizon, "horizon")?,
                        ],
                    )?;
                    set_bytes_f32(encoder, 13, &[gradient_scale])?;
                    set_bytes_u32(encoder, 14, &[CE_NO_IGNORE])
                },
                rows,
            )
        })?;
        let mut hidden_gradient = vec![0.0; hidden.len()];
        let mut row_loss = vec![0.0; rows];
        let mut scale_gradient = vec![0.0; vocab];
        let mut logit_gradient = vec![0.0; rows.saturating_mul(vocab)];
        gx_buffer.read_f32(&mut hidden_gradient)?;
        loss_buffer.read_f32(&mut row_loss)?;
        g_scale_buffer.read_f32(&mut scale_gradient)?;
        gy_buffer.read_f32(&mut logit_gradient)?;
        let mean_loss = if n_valid == 0 {
            0.0
        } else {
            row_loss.iter().sum::<f32>() / n_valid as f32
        };
        Ok(StreamedCrossEntropy {
            mean_loss,
            hidden_gradient,
            scale_gradient,
            row_loss,
            logit_gradient,
        })
    }

    pub fn apply_latent_sgd(
        &self,
        latent: &[u16],
        residual: &[f32],
        gradient: &[f32],
        learning_rate: f32,
        ste_scale: f32,
    ) -> Result<(Vec<u16>, Vec<f32>, Vec<u32>)> {
        if latent.len() != gradient.len() || residual.len() != latent.len() {
            bail!("latent SGD length mismatch");
        }
        if latent.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let words = latent.len().div_ceil(32);
        let latent_buffer = self.buffer_u16(latent)?;
        let residual_buffer = self.buffer_f32(residual)?;
        let grad_buffer = self.buffer_f32(gradient)?;
        let bits_buffer = self.alloc_bytes(words.saturating_mul(size_of::<u32>()).max(4))?;
        self.submit(|encoder| {
            Self::encode_binaryconnect_sgd_fp16(
                encoder,
                &self.pipelines.binaryconnect_sgd_fp16,
                &latent_buffer,
                &residual_buffer,
                &grad_buffer,
                learning_rate,
                ste_scale,
                latent.len(),
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.pack_latent_bits,
                &[&latent_buffer, &bits_buffer],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(latent.len(), "elements")?]),
                words,
            )
        })?;
        let mut next = vec![0_u16; latent.len()];
        let mut next_residual = vec![0.0; latent.len()];
        let mut bits = vec![0_u32; words];
        latent_buffer.read_u16(&mut next)?;
        residual_buffer.read_f32(&mut next_residual)?;
        bits_buffer.read_u32(&mut bits)?;
        Ok((next, next_residual, bits))
    }

    pub fn clipped_sgd_fp16(
        &self,
        parameters: &[u16],
        residual: &[f32],
        gradient: &[f32],
        learning_rate: f32,
    ) -> Result<(Vec<u16>, Vec<f32>)> {
        self.fp16_sgd(
            &self.pipelines.clipped_sgd_fp16,
            parameters,
            residual,
            gradient,
            learning_rate,
        )
    }

    pub fn binaryconnect_sgd_fp16(
        &self,
        parameters: &[u16],
        residual: &[f32],
        gradient: &[f32],
        learning_rate: f32,
    ) -> Result<(Vec<u16>, Vec<f32>)> {
        if parameters.len() != gradient.len() || residual.len() != parameters.len() {
            bail!("BinaryConnect SGD length mismatch");
        }
        if parameters.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let param_buffer = self.buffer_u16(parameters)?;
        let residual_buffer = self.buffer_f32(residual)?;
        let grad_buffer = self.buffer_f32(gradient)?;
        self.submit(|encoder| {
            Self::encode_binaryconnect_sgd_fp16(
                encoder,
                &self.pipelines.binaryconnect_sgd_fp16,
                &param_buffer,
                &residual_buffer,
                &grad_buffer,
                learning_rate,
                1.0,
                parameters.len(),
            )
        })?;
        let mut updated = vec![0_u16; parameters.len()];
        let mut next_residual = vec![0.0; parameters.len()];
        param_buffer.read_u16(&mut updated)?;
        residual_buffer.read_f32(&mut next_residual)?;
        Ok((updated, next_residual))
    }

    fn fp16_sgd(
        &self,
        pipeline: &Pipeline,
        parameters: &[u16],
        residual: &[f32],
        gradient: &[f32],
        learning_rate: f32,
    ) -> Result<(Vec<u16>, Vec<f32>)> {
        if parameters.len() != gradient.len() || residual.len() != parameters.len() {
            bail!("clipped SGD length mismatch");
        }
        if parameters.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let param_buffer = self.buffer_u16(parameters)?;
        let residual_buffer = self.buffer_f32(residual)?;
        let grad_buffer = self.buffer_f32(gradient)?;
        self.submit(|encoder| {
            Self::encode_clipped_sgd_fp16(
                encoder,
                pipeline,
                &param_buffer,
                &residual_buffer,
                &grad_buffer,
                learning_rate,
                parameters.len(),
            )
        })?;
        let mut updated = vec![0_u16; parameters.len()];
        let mut next_residual = vec![0.0; parameters.len()];
        param_buffer.read_u16(&mut updated)?;
        residual_buffer.read_f32(&mut next_residual)?;
        Ok((updated, next_residual))
    }

    /// Packed ±1 head: logits, softmax CE, and STE gradients in one command buffer.
    /// Logits and `gy` stay resident; only `gx`, `g_scale`, `g_w`, and row loss are read.
    pub fn packed_head_train(
        &self,
        hidden: &[f32],
        bits: &[u32],
        scale: &[f32],
        tokens: &[u32],
        rows: usize,
        time: usize,
        channels: usize,
        vocab: usize,
        horizon: usize,
    ) -> Result<PackedHeadTrain> {
        let shape = LinearDispatchShape::new(rows, channels, vocab)?;
        self.check_binary_shape(hidden, bits, scale, None, shape)?;
        if tokens.len() != rows || time == 0 || !rows.is_multiple_of(time) {
            bail!("packed head CE shape mismatch");
        }
        let ignore_id = CE_NO_IGNORE;
        let n_valid = Self::causal_supervised_count(tokens, time, horizon, ignore_id);
        let gradient_scale = causal_ce_gradient_scale(n_valid, time);
        let logits_len = rows.saturating_mul(vocab);
        let weights = vocab.saturating_mul(channels);
        let hidden_buffer = self.buffer_f32(hidden)?;
        let bits_buffer = self.buffer_u32(bits)?;
        let scale_buffer = self.buffer_u16(&fp16_bits(scale))?;
        let bias_buffer = self.buffer_u16(&fp16_bits(&[]))?;
        let tokens_buffer = self.buffer_u32(tokens)?;
        let logits_buffer = self.alloc_f32(logits_len)?;
        let gy_buffer = self.alloc_f32(logits_len)?;
        let loss_buffer = self.alloc_f32(rows)?;
        let gx_buffer = self.alloc_f32(hidden.len())?;
        let g_scale_buffer = self.alloc_f32(vocab)?;
        let g_bias_buffer = self.alloc_f32(vocab)?;
        let gw_buffer = self.alloc_f32(weights)?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear,
                &[
                    &hidden_buffer,
                    &bits_buffer,
                    &scale_buffer,
                    &bias_buffer,
                    &logits_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(channels, "in")?,
                            as_u32(vocab, "out")?,
                            0,
                        ],
                    )
                },
                vocab,
                rows,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.softmax_cross_entropy,
                &[&logits_buffer, &tokens_buffer, &loss_buffer, &gy_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(time, "time")?,
                            as_u32(vocab, "vocab")?,
                            as_u32(horizon, "horizon")?,
                        ],
                    )?;
                    set_bytes_f32(encoder, 8, &[gradient_scale])?;
                    set_bytes_u32(encoder, 9, &[ignore_id])
                },
                rows,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_input_bwd,
                &[&gy_buffer, &bits_buffer, &scale_buffer, &gx_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(channels, "in")?,
                            as_u32(vocab, "out")?,
                        ],
                    )
                },
                channels,
                rows,
            )?;
            Self::encode_scale_groups(
                encoder,
                &self.pipelines.binary_linear_scale_bwd_from_output,
                &[
                    &logits_buffer,
                    &gy_buffer,
                    &scale_buffer,
                    &bias_buffer,
                    &g_scale_buffer,
                    &g_bias_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        6,
                        &[as_u32(rows, "rows")?, as_u32(vocab, "out")?, 0],
                    )
                },
                vocab,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_weight_bwd,
                &[&hidden_buffer, &gy_buffer, &scale_buffer, &gw_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(channels, "in")?,
                            as_u32(vocab, "out")?,
                        ],
                    )
                },
                channels,
                vocab,
            )
        })?;
        let mut hidden_gradient = vec![0.0; hidden.len()];
        let mut scale_gradient = vec![0.0; vocab];
        let mut weight_gradient = vec![0.0; weights];
        let mut row_loss = vec![0.0; rows];
        gx_buffer.read_f32(&mut hidden_gradient)?;
        g_scale_buffer.read_f32(&mut scale_gradient)?;
        gw_buffer.read_f32(&mut weight_gradient)?;
        loss_buffer.read_f32(&mut row_loss)?;
        let mean_loss = if n_valid == 0 {
            0.0
        } else {
            row_loss.iter().sum::<f32>() / n_valid as f32
        };
        Ok(PackedHeadTrain {
            mean_loss,
            hidden_gradient,
            scale_gradient,
            weight_gradient,
        })
    }

    /// Packed head + STE latent SGD in one command buffer. Does not read `g_w`.
    pub fn packed_head_train_sgd(
        &self,
        hidden: &[f32],
        bits: &[u32],
        scale: &[u16],
        latent: &[u16],
        residual: &[f32],
        tokens: &[u32],
        rows: usize,
        time: usize,
        channels: usize,
        vocab: usize,
        horizon: usize,
        ignore_id: u32,
        learning_rate: f32,
        ste_scale: f32,
    ) -> Result<PackedHeadTrainSgd> {
        let shape = LinearDispatchShape::new(rows, channels, vocab)?;
        let weights = vocab.saturating_mul(channels);
        if hidden.len() != rows.saturating_mul(channels)
            || bits.len() != shape.packed_words()?
            || scale.len() != vocab
            || latent.len() != weights
            || residual.len() != weights
            || tokens.len() != rows
            || time == 0
            || !rows.is_multiple_of(time)
        {
            bail!("packed head SGD shape mismatch");
        }
        let n_valid = Self::causal_supervised_count(tokens, time, horizon, ignore_id);
        let gradient_scale = causal_ce_gradient_scale(n_valid, time);
        let logits_len = rows.saturating_mul(vocab);
        let words = weights.div_ceil(32);
        let hidden_buffer = self.buffer_f32(hidden)?;
        let bits_buffer = self.buffer_u32(bits)?;
        let scale_buffer = self.buffer_u16(scale)?;
        let bias_buffer = self.buffer_u16(&[])?;
        let tokens_buffer = self.buffer_u32(tokens)?;
        let latent_buffer = self.buffer_u16(latent)?;
        let residual_buffer = self.buffer_f32(residual)?;
        let next_bits = self.alloc_bytes(words.saturating_mul(size_of::<u32>()).max(4))?;
        let logits_buffer = self.alloc_f32(logits_len)?;
        let gy_buffer = self.alloc_f32(logits_len)?;
        let loss_buffer = self.alloc_f32(rows)?;
        let gx_buffer = self.alloc_f32(hidden.len())?;
        let g_scale_buffer = self.alloc_f32(vocab)?;
        let g_bias_buffer = self.alloc_f32(vocab)?;
        let gw_buffer = self.alloc_f32(weights)?;
        self.submit(|encoder| {
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear,
                &[
                    &hidden_buffer,
                    &bits_buffer,
                    &scale_buffer,
                    &bias_buffer,
                    &logits_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        5,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(channels, "in")?,
                            as_u32(vocab, "out")?,
                            0,
                        ],
                    )
                },
                vocab,
                rows,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.softmax_cross_entropy,
                &[&logits_buffer, &tokens_buffer, &loss_buffer, &gy_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(time, "time")?,
                            as_u32(vocab, "vocab")?,
                            as_u32(horizon, "horizon")?,
                        ],
                    )?;
                    set_bytes_f32(encoder, 8, &[gradient_scale])?;
                    set_bytes_u32(encoder, 9, &[ignore_id])
                },
                rows,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_input_bwd,
                &[&gy_buffer, &bits_buffer, &scale_buffer, &gx_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(channels, "in")?,
                            as_u32(vocab, "out")?,
                        ],
                    )
                },
                channels,
                rows,
            )?;
            Self::encode_scale_groups(
                encoder,
                &self.pipelines.binary_linear_scale_bwd_from_output,
                &[
                    &logits_buffer,
                    &gy_buffer,
                    &scale_buffer,
                    &bias_buffer,
                    &g_scale_buffer,
                    &g_bias_buffer,
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        6,
                        &[as_u32(rows, "rows")?, as_u32(vocab, "out")?, 0],
                    )
                },
                vocab,
            )?;
            Self::encode_tiled(
                encoder,
                &self.pipelines.binary_linear_weight_bwd,
                &[&hidden_buffer, &gy_buffer, &scale_buffer, &gw_buffer],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        4,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(channels, "in")?,
                            as_u32(vocab, "out")?,
                        ],
                    )
                },
                channels,
                vocab,
            )?;
            Self::encode_binaryconnect_sgd_fp16(
                encoder,
                &self.pipelines.binaryconnect_sgd_fp16,
                &latent_buffer,
                &residual_buffer,
                &gw_buffer,
                learning_rate,
                ste_scale,
                weights,
            )?;
            Self::encode_1d(
                encoder,
                &self.pipelines.pack_latent_bits,
                &[&latent_buffer, &next_bits],
                |encoder| set_bytes_u32(encoder, 2, &[as_u32(weights, "elements")?]),
                words,
            )
        })?;
        let mut hidden_gradient = vec![0.0; hidden.len()];
        let mut scale_gradient = vec![0.0; vocab];
        let mut row_loss = vec![0.0; rows];
        let mut next_latent = vec![0_u16; weights];
        let mut next_residual = vec![0.0; weights];
        let mut packed = vec![0_u32; words];
        gx_buffer.read_f32(&mut hidden_gradient)?;
        g_scale_buffer.read_f32(&mut scale_gradient)?;
        loss_buffer.read_f32(&mut row_loss)?;
        latent_buffer.read_u16(&mut next_latent)?;
        residual_buffer.read_f32(&mut next_residual)?;
        next_bits.read_u32(&mut packed)?;
        let mean_loss = if n_valid == 0 {
            0.0
        } else {
            row_loss.iter().sum::<f32>() / n_valid as f32
        };
        Ok(PackedHeadTrainSgd {
            mean_loss,
            hidden_gradient,
            scale_gradient,
            next_latent,
            next_residual,
            next_bits: packed,
            row_loss,
        })
    }

    fn check_binary_shape_u16(
        &self,
        input: &[f32],
        bits: &[u32],
        scale: &[u16],
        bias: Option<&[u16]>,
        shape: LinearDispatchShape,
    ) -> Result<()> {
        if input.len() != shape.rows.saturating_mul(shape.in_features)
            || bits.len() != shape.packed_words()?
            || scale.len() != shape.out_features
            || bias.is_some_and(|bias| bias.len() != shape.out_features)
        {
            bail!("binary linear shape mismatch");
        }
        Ok(())
    }

    fn check_binary_shape(
        &self,
        input: &[f32],
        bits: &[u32],
        scale: &[f32],
        bias: Option<&[f32]>,
        shape: LinearDispatchShape,
    ) -> Result<()> {
        if input.len() != shape.rows.saturating_mul(shape.in_features)
            || bits.len() != shape.packed_words()?
            || scale.len() != shape.out_features
            || bias.is_some_and(|bias| bias.len() != shape.out_features)
        {
            bail!("binary linear shape mismatch");
        }
        Ok(())
    }
}

/// Executes the identity smoke kernel and returns a fresh output vector.
pub fn identity_forward(input: &[f32]) -> Result<Vec<f32>> {
    MetalRuntime::new()?.identity(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Fp16Linear, LAYER_NORM_EPS, LayerNorm, PackedBinaryLinear};
    use crate::precision::Fp16Storage;

    fn runtime() -> Option<MetalRuntime> {
        MetalRuntime::new().ok()
    }

    fn pack_plus_bits(plus: impl Iterator<Item = bool>, weights: usize) -> Vec<u32> {
        let mut bits = vec![0_u32; weights.div_ceil(32)];
        for (index, is_plus) in plus.enumerate() {
            if is_plus {
                bits[index / 32] |= 1_u32 << (index % 32);
            }
        }
        bits
    }

    fn assert_close(actual: &[f32], expected: &[f32], atol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= atol,
                "index {index}: {a} vs {e} (atol {atol})"
            );
        }
    }

    fn cpu_layer_norm_backward(
        x: &[f32],
        gy: &[f32],
        weight: &[f32],
        rows: usize,
        channels: usize,
    ) -> LayerNormBackward {
        let mut gx = vec![0.0; x.len()];
        let mut gw = vec![0.0; channels];
        let mut gb = vec![0.0; channels];
        for row in 0..rows {
            let start = row * channels;
            let src = &x[start..start + channels];
            let mean = src.iter().sum::<f32>() / channels as f32;
            let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / channels as f32;
            let inv = (var + LAYER_NORM_EPS).sqrt().recip();
            let mut sum_dxhat = 0.0;
            let mut sum_dxhat_xhat = 0.0;
            let mut xhat = vec![0.0; channels];
            let mut dxhat = vec![0.0; channels];
            for c in 0..channels {
                xhat[c] = (src[c] - mean) * inv;
                dxhat[c] = gy[start + c] * weight[c];
                sum_dxhat += dxhat[c];
                sum_dxhat_xhat += dxhat[c] * xhat[c];
                gw[c] += gy[start + c] * xhat[c];
                gb[c] += gy[start + c];
            }
            let inv_n = inv / channels as f32;
            for c in 0..channels {
                gx[start + c] =
                    inv_n * (channels as f32 * dxhat[c] - sum_dxhat - xhat[c] * sum_dxhat_xhat);
            }
        }
        LayerNormBackward {
            input_gradient: gx,
            weight_gradient: gw,
            bias_gradient: gb,
        }
    }

    fn cpu_time_shift(x: &[f32], time: usize, channels: usize) -> Vec<f32> {
        let mut xx = vec![0.0; x.len()];
        let rows = x.len() / channels;
        for row in 0..rows {
            let t = row % time;
            for c in 0..channels {
                let index = row * channels + c;
                xx[index] = if t == 0 {
                    -x[index]
                } else {
                    x[index - channels] - x[index]
                };
            }
        }
        xx
    }

    fn cpu_streamed_ce(
        hidden: &[f32],
        linear: &PackedBinaryLinear,
        tokens: &[u32],
        time: usize,
        horizon: usize,
    ) -> StreamedCrossEntropy {
        let rows = tokens.len();
        let channels = linear.in_features();
        let vocab = linear.out_features();
        let logits = linear.forward(hidden, rows).unwrap();
        let mut n_valid = 0_usize;
        for row in 0..rows {
            if row % time + horizon < time {
                n_valid += 1;
            }
        }
        let scale = causal_ce_gradient_scale(n_valid, time);
        let mut row_loss = vec![0.0; rows];
        let mut gy = vec![0.0; rows * vocab];
        for row in 0..rows {
            if row % time + horizon >= time {
                continue;
            }
            let target = tokens[row + horizon] as usize;
            let start = row * vocab;
            let row_logits = &logits[start..start + vocab];
            let max = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row_logits.iter().map(|v| (v - max).exp()).sum();
            row_loss[row] = max + exp_sum.ln() - row_logits[target];
            for v in 0..vocab {
                let p = (row_logits[v] - max).exp() / exp_sum;
                gy[start + v] = scale * (p - f32::from(v == target));
            }
        }
        let mut g_w = vec![0.0; vocab * channels];
        let mut gx = vec![0.0; hidden.len()];
        let mut g_scale = vec![0.0; vocab];
        linear
            .backward_ste(
                hidden,
                &gy,
                rows,
                &mut g_w,
                Some(&mut gx),
                &mut g_scale,
                None,
            )
            .unwrap();
        let mean_loss = if n_valid == 0 {
            0.0
        } else {
            row_loss.iter().sum::<f32>() / n_valid as f32
        };
        StreamedCrossEntropy {
            mean_loss,
            hidden_gradient: gx,
            scale_gradient: g_scale,
            row_loss,
            logit_gradient: gy,
        }
    }

    #[test]
    fn dispatch_shape_rejects_zero_and_32_bit_overflow() {
        assert!(MetalDispatchShape::new(0, 1, 1).is_err());
        assert!(MetalDispatchShape::new(1, 1, 0).is_err());
        assert!(MetalDispatchShape::new(65_536, 65_536, 1).is_err());
        assert_eq!(MetalDispatchShape::new(2, 3, 4).unwrap().elements(), 24);
    }

    #[test]
    fn shader_source_has_no_mps_or_hyena_aliases() {
        let source = RWKV8_METAL_SOURCE.to_ascii_lowercase();
        assert!(!source.contains("mps"));
        assert!(!source.contains("hyena"));
        assert!(!source.contains("fft"));
        assert!(!source.contains("rms_norm"));
        for name in PR3_KERNEL_NAMES {
            assert!(
                RWKV8_METAL_SOURCE.contains(&format!("kernel void {name}(")),
                "missing kernel {name}"
            );
        }
        for name in PR4_KERNEL_NAMES {
            assert!(
                RWKV8_METAL_SOURCE.contains(&format!("kernel void {name}(")),
                "missing kernel {name}"
            );
        }
        for name in PR5_KERNEL_NAMES {
            assert!(
                RWKV8_METAL_SOURCE.contains(&format!("kernel void {name}(")),
                "missing kernel {name}"
            );
        }
        for name in PR8_KERNEL_NAMES {
            assert!(
                RWKV8_METAL_SOURCE.contains(&format!("kernel void {name}(")),
                "missing kernel {name}"
            );
        }
        assert!(!RWKV8_METAL_SOURCE.contains("ullis_rosa_qkv_1bit_bwd_bits"));
    }

    #[test]
    fn pr3_kernels_compile_on_the_local_metal_device() {
        let Ok(_) = MetalRuntime::new() else {
            return;
        };
        let shape = MetalDispatchShape::new(1, 8, 16).unwrap();
        for name in PR3_KERNEL_NAMES
            .iter()
            .chain(PR4_KERNEL_NAMES)
            .chain(PR5_KERNEL_NAMES)
            .chain(PR8_KERNEL_NAMES)
        {
            let width = validate_metal_kernel(name, shape).unwrap();
            assert!(width > 0, "{name}");
        }
    }

    #[test]
    fn identity_kernel_round_trips_fp32_data_when_metal_is_available() {
        let input = [-1.0, 0.0, 0.5, 3.25];
        if let Ok(output) = identity_forward(&input) {
            assert_eq!(output, input);
        }
    }

    #[test]
    fn layer_norm_matches_cpu_affine_reference() {
        let Some(runtime) = runtime() else {
            return;
        };
        let ln = LayerNorm::from_bits(
            fp16_bits(&[1.25, 0.75, 1.0, 0.5]),
            fp16_bits(&[0.1, -0.2, 0.0, 0.05]),
        )
        .unwrap();
        let x = [1.0_f32, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0, 2.5];
        let cpu = ln.forward(&x, 2).unwrap();
        let gpu = runtime
            .layer_norm(&x, &[1.25, 0.75, 1.0, 0.5], &[0.1, -0.2, 0.0, 0.05], 2, 4)
            .unwrap();
        assert_close(&gpu, &cpu, 2e-3);
    }

    #[test]
    fn layer_norm_backward_matches_cpu() {
        let Some(runtime) = runtime() else {
            return;
        };
        let x = [1.0_f32, 2.0, 0.0, -1.0, 4.0, 1.0, -2.0, 0.5];
        let gy = [0.2_f32, -0.1, 0.3, 0.0, -0.4, 0.15, 0.05, -0.2];
        let weight = [1.0_f32, 0.5, 1.5, 0.75];
        let cpu = cpu_layer_norm_backward(&x, &gy, &weight, 2, 4);
        let gpu = runtime.layer_norm_backward(&x, &gy, &weight, 2, 4).unwrap();
        assert_close(&gpu.input_gradient, &cpu.input_gradient, 3e-3);
        assert_close(&gpu.weight_gradient, &cpu.weight_gradient, 3e-3);
        assert_close(&gpu.bias_gradient, &cpu.bias_gradient, 3e-3);
    }

    #[test]
    fn binary_linear_matches_packed_cpu_forward_and_ste() {
        let Some(runtime) = runtime() else {
            return;
        };
        let signs = [1_i8, -1, -1, 1, 1, 1];
        let linear = PackedBinaryLinear::from_signs(2, 3, &signs, 0.5, true).unwrap();
        let x = [0.5_f32, -1.0, 2.0, 1.0, 0.0, -0.5];
        let cpu = linear.forward(&x, 2).unwrap();
        let bits = pack_plus_bits(signs.iter().map(|&s| s >= 0), 6);
        let shape = LinearDispatchShape::new(2, 3, 2).unwrap();
        let gpu = runtime
            .binary_linear(&x, &bits, &[0.5, 0.5], Some(&[0.0, 0.0]), shape)
            .unwrap();
        assert_close(&gpu, &cpu, 2e-3);

        let gy = [1.0_f32, -0.5, 0.25, 0.75];
        let mut g_w = [0.0_f32; 6];
        let mut g_x = [0.0_f32; 6];
        let mut g_scale = [0.0_f32; 2];
        let mut g_bias = [0.0_f32; 2];
        linear
            .backward_ste(
                &x,
                &gy,
                2,
                &mut g_w,
                Some(&mut g_x),
                &mut g_scale,
                Some(&mut g_bias),
            )
            .unwrap();
        let gpu_bwd = runtime
            .binary_linear_backward(&x, &gy, &bits, &[0.5, 0.5], true, shape)
            .unwrap();
        assert_close(&gpu_bwd.input_gradient, &g_x, 2e-3);
        assert_close(&gpu_bwd.scale_gradient, &g_scale, 2e-3);
        assert_close(gpu_bwd.bias_gradient.as_ref().unwrap(), &g_bias, 2e-3);
        assert_close(&gpu_bwd.weight_gradient, &g_w, 2e-3);
    }

    #[test]
    fn binary_linear_latent_sgd_matches_cpu_updater() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut linear = PackedBinaryLinear::from_signs(1, 2, &[1, -1], 0.5, false).unwrap();
        let x = [1.0_f32, 2.0];
        let gy = [1.0_f32];
        let mut g_w = [0.0_f32; 2];
        let mut g_scale = [0.0_f32; 1];
        linear
            .backward_ste(&x, &gy, 1, &mut g_w, None, &mut g_scale, None)
            .unwrap();
        linear.apply_clipped_sgd(&g_w, &[0.0], None, 0.1).unwrap();
        let shape = LinearDispatchShape::new(1, 2, 1).unwrap();
        let latent = [
            Fp16::from_f32(0.5).to_bits(),
            Fp16::from_f32(-0.5).to_bits(),
        ];
        let (next, bits, gpu_gw) = runtime
            .binary_linear_latent_sgd(&latent, &x, &gy, &[0.5], 0.1, shape)
            .unwrap();
        assert_close(&gpu_gw, &g_w, 2e-4);
        assert_eq!(next[0], Fp16::from_f32(linear.latent().get(0)).to_bits());
        assert_eq!(next[1], Fp16::from_f32(linear.latent().get(1)).to_bits());
        assert_eq!(bits[0] & 1, u32::from(linear.sign_at(0) > 0.0));
        assert_eq!((bits[0] >> 1) & 1, u32::from(linear.sign_at(1) > 0.0));
    }

    #[test]
    fn fp16_linear_matches_cpu_forward_and_backward() {
        let Some(runtime) = runtime() else {
            return;
        };
        let weight = [1.0_f32, -0.5, 0.25, 2.0, 0.0, -1.0];
        let linear = Fp16Linear::from_f32(2, 3, &weight).unwrap();
        let x = [0.5_f32, 1.0, -1.0, 2.0, 0.0, 0.25];
        let cpu = linear.forward(&x, 2).unwrap();
        let shape = LinearDispatchShape::new(2, 3, 2).unwrap();
        let gpu = runtime.fp16_linear(&x, &weight, shape).unwrap();
        assert_close(&gpu, &cpu, 2e-3);

        let gy = [1.0_f32, -1.0, 0.5, 0.25];
        let mut gx = vec![0.0; 6];
        let mut gw = vec![0.0; 6];
        for row in 0..2 {
            for o in 0..2 {
                for i in 0..3 {
                    let w = Fp16::from_f32(weight[o * 3 + i]).to_f32();
                    gx[row * 3 + i] += gy[row * 2 + o] * w;
                    gw[o * 3 + i] += gy[row * 2 + o] * x[row * 3 + i];
                }
            }
        }
        let gpu_bwd = runtime
            .fp16_linear_backward(&x, &gy, &weight, shape)
            .unwrap();
        assert_close(&gpu_bwd.input_gradient, &gx, 3e-3);
        assert_close(&gpu_bwd.weight_gradient, &gw, 3e-3);
    }

    #[test]
    fn tiled_linears_match_cpu_past_tile_boundary() {
        let Some(runtime) = runtime() else {
            return;
        };
        let rows = 17;
        let inn = 19;
        let out = 18;
        let mut signs = Vec::with_capacity(out * inn);
        let mut weight = Vec::with_capacity(out * inn);
        let mut x = Vec::with_capacity(rows * inn);
        let mut gy = Vec::with_capacity(rows * out);
        for o in 0..out {
            for i in 0..inn {
                signs.push(if (o + i) % 2 == 0 { 1_i8 } else { -1 });
                weight.push(((o * 3 + i) % 7) as f32 * 0.25 - 0.5);
            }
        }
        for row in 0..rows {
            for i in 0..inn {
                x.push(((row * 5 + i) % 11) as f32 * 0.125 - 0.5);
            }
            for o in 0..out {
                gy.push(((row + o) % 5) as f32 * 0.2 - 0.4);
            }
        }
        let linear = PackedBinaryLinear::from_signs(out, inn, &signs, 0.5, true).unwrap();
        let cpu = linear.forward(&x, rows).unwrap();
        let bits = pack_plus_bits(signs.iter().map(|&s| s >= 0), out * inn);
        let shape = LinearDispatchShape::new(rows, inn, out).unwrap();
        let gpu = runtime
            .binary_linear(&x, &bits, &[0.5; 18], Some(&[0.0; 18]), shape)
            .unwrap();
        assert_close(&gpu, &cpu, 3e-3);
        let mut g_w = vec![0.0; out * inn];
        let mut g_x = vec![0.0; rows * inn];
        let mut g_scale = vec![0.0; out];
        let mut g_bias = vec![0.0; out];
        linear
            .backward_ste(
                &x,
                &gy,
                rows,
                &mut g_w,
                Some(&mut g_x),
                &mut g_scale,
                Some(&mut g_bias),
            )
            .unwrap();
        let gpu_bwd = runtime
            .binary_linear_backward(&x, &gy, &bits, &[0.5; 18], true, shape)
            .unwrap();
        assert_close(&gpu_bwd.input_gradient, &g_x, 4e-3);
        assert_close(&gpu_bwd.scale_gradient, &g_scale, 4e-3);
        assert_close(gpu_bwd.bias_gradient.as_ref().unwrap(), &g_bias, 4e-3);
        assert_close(&gpu_bwd.weight_gradient, &g_w, 4e-3);

        let fp = Fp16Linear::from_f32(out, inn, &weight).unwrap();
        let cpu_y = fp.forward(&x, rows).unwrap();
        let gpu_y = runtime.fp16_linear(&x, &weight, shape).unwrap();
        assert_close(&gpu_y, &cpu_y, 3e-3);
        let (cpu_gx, cpu_gw) = fp.backward(&x, &gy, rows).unwrap();
        let gpu_fp = runtime
            .fp16_linear_backward(&x, &gy, &weight, shape)
            .unwrap();
        assert_close(&gpu_fp.input_gradient, &cpu_gx, 4e-3);
        assert_close(&gpu_fp.weight_gradient, &cpu_gw, 4e-3);
    }

    #[test]
    fn time_shift_relu2_residual_and_sign_pack_match_cpu() {
        let Some(runtime) = runtime() else {
            return;
        };
        let x = [1.0_f32, -2.0, 0.5, 0.0, 3.0, -0.25];
        let gpu_xx = runtime.time_shift_delta(&x, 3, 3, 2).unwrap();
        assert_close(&gpu_xx, &cpu_time_shift(&x, 3, 2), 1e-6);

        let relu = runtime.cmix_relu2(&x).unwrap();
        let cpu_relu: Vec<f32> = x.iter().map(|v| v.max(0.0) * v.max(0.0)).collect();
        assert_close(&relu, &cpu_relu, 1e-6);
        let gy = [1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let gx = runtime.cmix_relu2_backward(&x, &gy).unwrap();
        let cpu_gx: Vec<f32> = x
            .iter()
            .zip(&gy)
            .map(|(v, g)| if *v > 0.0 { 2.0 * v * g } else { 0.0 })
            .collect();
        assert_close(&gx, &cpu_gx, 1e-6);

        let residual = runtime.residual_add(&x, &[0.5; 6]).unwrap();
        let cpu_res: Vec<f32> = x.iter().map(|v| v + 0.5).collect();
        assert_close(&residual, &cpu_res, 1e-6);

        let bits = runtime.sign_pack_bits(&x).unwrap();
        let expected = pack_plus_bits(x.iter().map(|v| *v > 0.0), x.len());
        assert_eq!(bits, expected);
    }

    #[test]
    fn clipped_sgd_fp16_matches_cpu_residual() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut cpu = Fp16Storage::from_f32([0.25, -1.0, 2.0]);
        let gradient = [0.001_f32, 0.5, -0.25];
        for (index, g) in gradient.iter().enumerate() {
            cpu.apply_clipped_sgd(index, *g, 0.01);
        }
        let original = fp16_bits(&[0.25, -1.0, 2.0]);
        let residual = [0.0_f32; 3];
        let (gpu, gpu_residual) = runtime
            .clipped_sgd_fp16(&original, &residual, &gradient, 0.01)
            .unwrap();
        assert_eq!(gpu, cpu.as_bits());
        assert_eq!(gpu_residual.as_slice(), cpu.residual());
    }

    #[test]
    fn binaryconnect_sgd_fp16_matches_cpu_magnitude_clip() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut cpu = Fp16Storage::from_f32([0.0625, -0.0625, 0.9]);
        let gradient = [1e-5_f32, -1e-6, 2.0];
        for (index, g) in gradient.iter().enumerate() {
            cpu.apply_binaryconnect_sgd(index, *g, 0.1);
        }
        let original = fp16_bits(&[0.0625, -0.0625, 0.9]);
        let residual = [0.0_f32; 3];
        let (gpu, gpu_residual) = runtime
            .binaryconnect_sgd_fp16(&original, &residual, &gradient, 0.1)
            .unwrap();
        assert_eq!(gpu, cpu.as_bits());
        assert_eq!(gpu_residual.as_slice(), cpu.residual());
    }

    #[test]
    fn streamed_cross_entropy_matches_cpu_oracle_without_materializing_on_gpu() {
        let Some(runtime) = runtime() else {
            return;
        };
        let signs = [1_i8, -1, 1, -1, -1, 1, 1, 1];
        let linear = PackedBinaryLinear::from_signs(4, 2, &signs, 0.5, false).unwrap();
        let hidden = [0.5_f32, -0.25, 1.0, 0.0, -0.5, 0.75];
        let tokens = [1_u32, 3, 0];
        let cpu = cpu_streamed_ce(&hidden, &linear, &tokens, 3, 1);
        let bits = pack_plus_bits(signs.iter().map(|&s| s >= 0), 8);
        let gpu = runtime
            .streamed_cross_entropy_fp16(&hidden, &bits, &[0.5; 4], &tokens, 3, 3, 2, 4, 1)
            .unwrap();
        assert!((gpu.mean_loss - cpu.mean_loss).abs() < 3e-3);
        assert_close(&gpu.hidden_gradient, &cpu.hidden_gradient, 5e-3);
        assert_close(&gpu.scale_gradient, &cpu.scale_gradient, 5e-3);
        assert_close(&gpu.logit_gradient, &cpu.logit_gradient, 5e-3);
        assert_eq!(gpu.row_loss[2], 0.0);

        let fused = runtime
            .packed_head_train(&hidden, &bits, &[0.5; 4], &tokens, 3, 3, 2, 4, 1)
            .unwrap();
        assert!((fused.mean_loss - cpu.mean_loss).abs() < 3e-3);
        assert_close(&fused.hidden_gradient, &cpu.hidden_gradient, 5e-3);
        assert_close(&fused.scale_gradient, &cpu.scale_gradient, 5e-3);
        let mut g_w = vec![0.0; 8];
        let mut g_x = vec![0.0; 6];
        let mut g_scale = vec![0.0; 4];
        linear
            .backward_ste(
                &hidden,
                &cpu.logit_gradient,
                3,
                &mut g_w,
                Some(&mut g_x),
                &mut g_scale,
                None,
            )
            .unwrap();
        assert_close(&fused.weight_gradient, &g_w, 5e-3);
    }
}
