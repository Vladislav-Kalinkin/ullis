//! Metal runtime for Heron: LayerNorm, BinaryConnect, FP16 linear, and streamed CE.
//!
//! Buffer mapping lives in [`ffi`]. There is no MPS path. ROSA SAM and WKV7
//! kernels arrive in later PRs. Identity remains a pipeline-smoke entry point.

use anyhow::{Result, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

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
pub const TIME_SHIFT_DELTA_KERNEL_NAME: &str = "ullis_time_shift_delta";
pub const BINARY_LINEAR_KERNEL_NAME: &str = "ullis_binary_linear";
pub const BINARY_LINEAR_INPUT_BWD_KERNEL_NAME: &str = "ullis_binary_linear_input_bwd";
pub const BINARY_LINEAR_SCALE_BWD_KERNEL_NAME: &str = "ullis_binary_linear_scale_bwd";
pub const BINARY_LINEAR_LATENT_SGD_KERNEL_NAME: &str = "ullis_binary_linear_latent_sgd";
pub const FP16_LINEAR_KERNEL_NAME: &str = "ullis_fp16_linear";
pub const FP16_LINEAR_BWD_KERNEL_NAME: &str = "ullis_fp16_linear_bwd";
pub const SIGN_PACK_BITS_KERNEL_NAME: &str = "ullis_sign_pack_bits";
pub const CMIX_RELU2_KERNEL_NAME: &str = "ullis_cmix_relu2";
pub const CMIX_RELU2_BACKWARD_KERNEL_NAME: &str = "ullis_cmix_relu2_backward";
pub const RESIDUAL_ADD_KERNEL_NAME: &str = "ullis_residual_add";
pub const STREAMED_CROSS_ENTROPY_FP16_KERNEL_NAME: &str = "ullis_streamed_cross_entropy_fp16";
pub const CLIPPED_SGD_FP16_KERNEL_NAME: &str = "ullis_clipped_sgd_fp16";

pub const RWKV8_METAL_SOURCE: &str = include_str!("metal/rwkv8.metal");

pub const PR3_KERNEL_NAMES: &[&str] = &[
    IDENTITY_KERNEL_NAME,
    LAYER_NORM_KERNEL_NAME,
    LAYER_NORM_BACKWARD_KERNEL_NAME,
    TIME_SHIFT_DELTA_KERNEL_NAME,
    BINARY_LINEAR_KERNEL_NAME,
    BINARY_LINEAR_INPUT_BWD_KERNEL_NAME,
    BINARY_LINEAR_SCALE_BWD_KERNEL_NAME,
    BINARY_LINEAR_LATENT_SGD_KERNEL_NAME,
    FP16_LINEAR_KERNEL_NAME,
    FP16_LINEAR_BWD_KERNEL_NAME,
    SIGN_PACK_BITS_KERNEL_NAME,
    CMIX_RELU2_KERNEL_NAME,
    CMIX_RELU2_BACKWARD_KERNEL_NAME,
    RESIDUAL_ADD_KERNEL_NAME,
    STREAMED_CROSS_ENTROPY_FP16_KERNEL_NAME,
    CLIPPED_SGD_FP16_KERNEL_NAME,
];

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

fn compile_named_pipeline(kernel_name: &str) -> Result<Pipeline> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice};

    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
    let source = NSString::from_str(RWKV8_METAL_SOURCE);
    let options = MTLCompileOptions::new();
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
}

/// Dense FP16 linear derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct Fp16LinearBackward {
    pub input_gradient: Vec<f32>,
    pub weight_gradient: Vec<f32>,
}

/// Next-token streamed CE without a `[rows, vocab]` logit tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamedCrossEntropy {
    pub mean_loss: f32,
    pub hidden_gradient: Vec<f32>,
    pub scale_gradient: Vec<f32>,
    pub row_loss: Vec<f32>,
}

struct Pipelines {
    identity: Pipeline,
    layer_norm: Pipeline,
    layer_norm_backward: Pipeline,
    time_shift_delta: Pipeline,
    binary_linear: Pipeline,
    binary_linear_input_bwd: Pipeline,
    binary_linear_scale_bwd: Pipeline,
    binary_linear_latent_sgd: Pipeline,
    fp16_linear: Pipeline,
    fp16_linear_bwd: Pipeline,
    sign_pack_bits: Pipeline,
    cmix_relu2: Pipeline,
    cmix_relu2_backward: Pipeline,
    residual_add: Pipeline,
    streamed_cross_entropy_fp16: Pipeline,
    clipped_sgd_fp16: Pipeline,
}

/// Reusable Metal objects for resident Heron kernels. No MPS GEMM.
pub struct MetalRuntime {
    device: Device,
    queue: Queue,
    pipelines: Pipelines,
}

impl std::fmt::Debug for MetalRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalRuntime").finish_non_exhaustive()
    }
}

impl MetalRuntime {
    pub fn new() -> Result<Self> {
        use objc2_foundation::NSString;
        use objc2_metal::{MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice};

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
        let source = NSString::from_str(RWKV8_METAL_SOURCE);
        let options = MTLCompileOptions::new();
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
            time_shift_delta: pipeline_from_library(
                &device,
                &library,
                TIME_SHIFT_DELTA_KERNEL_NAME,
            )?,
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
            binary_linear_latent_sgd: pipeline_from_library(
                &device,
                &library,
                BINARY_LINEAR_LATENT_SGD_KERNEL_NAME,
            )?,
            fp16_linear: pipeline_from_library(&device, &library, FP16_LINEAR_KERNEL_NAME)?,
            fp16_linear_bwd: pipeline_from_library(&device, &library, FP16_LINEAR_BWD_KERNEL_NAME)?,
            sign_pack_bits: pipeline_from_library(&device, &library, SIGN_PACK_BITS_KERNEL_NAME)?,
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
            clipped_sgd_fp16: pipeline_from_library(
                &device,
                &library,
                CLIPPED_SGD_FP16_KERNEL_NAME,
            )?,
        };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow::anyhow!("Metal command queue is unavailable"))?;
        Ok(Self {
            device,
            queue,
            pipelines,
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

    fn shared_buffer(&self, bytes: usize) -> Result<MetalBuffer> {
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

    fn buffer_f32(&self, values: &[f32]) -> Result<MetalBuffer> {
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

    fn buffer_u16(&self, values: &[u16]) -> Result<MetalBuffer> {
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

    fn buffer_u32(&self, values: &[u32]) -> Result<MetalBuffer> {
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

    fn zeros_f32(&self, len: usize) -> Result<MetalBuffer> {
        let bytes = len.saturating_mul(size_of::<f32>()).max(size_of::<f32>());
        let buffer = self.shared_buffer(bytes)?;
        buffer.zero()?;
        Ok(buffer)
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

    fn encode_1d(
        encoder: &ffi::ComputeEncoder,
        pipeline: &Pipeline,
        buffers: &[&MetalBuffer],
        constants: impl FnOnce(&ffi::ComputeEncoder) -> Result<()>,
        threads: usize,
    ) -> Result<()> {
        use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

        if threads == 0 {
            bail!("Metal dispatch cannot be empty");
        }
        encoder.setComputePipelineState(pipeline);
        for (slot, buffer) in buffers.iter().enumerate() {
            set_buffer(encoder, buffer, slot);
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
        let gx_buffer = self.zeros_f32(input.len())?;
        let gw_buffer = self.zeros_f32(channels)?;
        let gb_buffer = self.zeros_f32(channels)?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.layer_norm_backward,
                &[
                    &input_buffer,
                    &gy_buffer,
                    &weight_buffer,
                    &gx_buffer,
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
                rows,
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
            Self::encode_1d(
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
                out_len,
            )
        })?;
        let mut output = vec![0.0; out_len];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
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
        let gx_buffer = self.zeros_f32(input.len())?;
        let g_scale_buffer = self.zeros_f32(shape.out_features)?;
        let g_bias_buffer = self.zeros_f32(shape.out_features)?;
        self.submit(|encoder| {
            Self::encode_1d(
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
                input.len(),
            )?;
            Self::encode_1d(
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
            )
        })?;
        let mut input_gradient = vec![0.0; input.len()];
        let mut scale_gradient = vec![0.0; shape.out_features];
        gx_buffer.read_f32(&mut input_gradient)?;
        g_scale_buffer.read_f32(&mut scale_gradient)?;
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
        })
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
            Self::encode_1d(
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
                out_len,
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
            Self::encode_1d(
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
                input.len(),
            )?;
            Self::encode_1d(
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
                weights,
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
        let mut n_valid = 0_usize;
        for row in 0..rows {
            if row % time + horizon < time {
                n_valid += 1;
            }
        }
        let gradient_scale = if n_valid == 0 {
            0.0
        } else {
            1.0 / n_valid as f32
        };
        let hidden_buffer = self.buffer_f32(hidden)?;
        let bits_buffer = self.buffer_u32(bits)?;
        let scale_buffer = self.buffer_u16(&fp16_bits(scale))?;
        let tokens_buffer = self.buffer_u32(tokens)?;
        let gx_buffer = self.zeros_f32(hidden.len())?;
        let loss_buffer = self.zeros_f32(rows)?;
        let g_scale_buffer = self.zeros_f32(vocab)?;
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
                ],
                |encoder| {
                    Self::set_u32s(
                        encoder,
                        7,
                        &[
                            as_u32(rows, "rows")?,
                            as_u32(time, "time")?,
                            as_u32(channels, "channels")?,
                            as_u32(vocab, "vocab")?,
                            as_u32(horizon, "horizon")?,
                        ],
                    )?;
                    set_bytes_f32(encoder, 12, &[gradient_scale])
                },
                rows,
            )
        })?;
        let mut hidden_gradient = vec![0.0; hidden.len()];
        let mut row_loss = vec![0.0; rows];
        let mut scale_gradient = vec![0.0; vocab];
        gx_buffer.read_f32(&mut hidden_gradient)?;
        loss_buffer.read_f32(&mut row_loss)?;
        g_scale_buffer.read_f32(&mut scale_gradient)?;
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
        })
    }

    pub fn clipped_sgd_fp16(
        &self,
        parameters: &[u16],
        gradient: &[f32],
        learning_rate: f32,
    ) -> Result<Vec<u16>> {
        if parameters.len() != gradient.len() {
            bail!("clipped SGD length mismatch");
        }
        if parameters.is_empty() {
            return Ok(Vec::new());
        }
        let param_buffer = self.buffer_u16(parameters)?;
        let grad_buffer = self.buffer_f32(gradient)?;
        self.submit(|encoder| {
            Self::encode_1d(
                encoder,
                &self.pipelines.clipped_sgd_fp16,
                &[&param_buffer, &grad_buffer],
                |encoder| {
                    set_bytes_f32(encoder, 2, &[learning_rate])?;
                    set_bytes_u32(encoder, 3, &[as_u32(parameters.len(), "elements")?])
                },
                parameters.len(),
            )
        })?;
        let mut updated = vec![0_u16; parameters.len()];
        param_buffer.read_u16(&mut updated)?;
        Ok(updated)
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
        let scale = if n_valid == 0 {
            0.0
        } else {
            1.0 / n_valid as f32
        };
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
        assert!(!RWKV8_METAL_SOURCE.contains("ullis_rosa_qkv"));
        assert!(!RWKV8_METAL_SOURCE.contains("ullis_wkv7"));
    }

    #[test]
    fn pr3_kernels_compile_on_the_local_metal_device() {
        let Ok(_) = MetalRuntime::new() else {
            return;
        };
        let shape = MetalDispatchShape::new(1, 8, 16).unwrap();
        for name in PR3_KERNEL_NAMES {
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
    fn clipped_sgd_fp16_matches_cpu_ulp_floor() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut cpu = Fp16Storage::from_f32([0.25, -1.0, 2.0]);
        let gradient = [0.001_f32, 0.5, -0.25];
        for (index, g) in gradient.iter().enumerate() {
            cpu.apply_clipped_sgd(index, *g, 0.01);
        }
        let original = fp16_bits(&[0.25, -1.0, 2.0]);
        let gpu = runtime
            .clipped_sgd_fp16(&original, &gradient, 0.01)
            .unwrap();
        assert_eq!(gpu, cpu.as_bits());
        let _ = gpu;
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
        assert_eq!(gpu.row_loss[2], 0.0);
    }
}
