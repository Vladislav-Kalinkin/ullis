//! Thin Metal runtime for Heron: device, queue, and the identity smoke kernel.
//!
//! Buffer mapping lives in [`ffi`]. Hyena FFT, RMSNorm, ternary, and MPS GEMM
//! paths are gone. Compute kernels other than identity arrive in later PRs.

use anyhow::{Result, bail};

pub mod ffi;

use self::ffi::{MetalBuffer, set_buffer, set_bytes_u32};

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

/// Pipeline smoke kernel. It is not a model operator.
pub const IDENTITY_KERNEL_NAME: &str = "ullis_identity";
pub const RWKV8_METAL_SOURCE: &str = include_str!("metal/rwkv8.metal");

/// Compiles the identity entry point and checks its dispatch capacity.
pub fn validate_metal_pipeline(shape: MetalDispatchShape) -> Result<usize> {
    validate_metal_kernel(IDENTITY_KERNEL_NAME, shape)
}

/// Compiles a named Ullis MSL entry point and checks its dispatch capacity.
pub fn validate_metal_kernel(kernel_name: &str, shape: MetalDispatchShape) -> Result<usize> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary,
    };

    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
    let source = NSString::from_str(RWKV8_METAL_SOURCE);
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
    let _ = shape.elements();
    Ok(width)
}

/// Reusable Metal objects for later resident kernels. PR 1 only compiles
/// identity so the device/queue/pipeline contract is proven.
pub struct MetalRuntime {
    device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    queue: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    identity_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
}

impl std::fmt::Debug for MetalRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalRuntime").finish_non_exhaustive()
    }
}

impl MetalRuntime {
    pub fn new() -> Result<Self> {
        use objc2_foundation::NSString;
        use objc2_metal::{MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};

        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow::anyhow!("Metal device is unavailable"))?;
        let source = NSString::from_str(RWKV8_METAL_SOURCE);
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
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow::anyhow!("Metal command queue is unavailable"))?;
        Ok(Self {
            device,
            queue,
            identity_pipeline,
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

    pub fn identity(&self, input: &[f32]) -> Result<Vec<f32>> {
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
            MTLComputePipelineState, MTLSize,
        };

        MetalDispatchShape::new(1, input.len().max(1), 1)?;
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let elements = u32::try_from(input.len())
            .map_err(|_| anyhow::anyhow!("Metal element count exceeds u32"))?;
        let bytes = input
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal buffer byte size overflow"))?;
        let input_buffer = self.shared_buffer(bytes)?;
        let output_buffer = self.shared_buffer(bytes)?;
        input_buffer.write_f32(input)?;

        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| anyhow::anyhow!("Metal compute encoder allocation failed"))?;
        encoder.setComputePipelineState(&self.identity_pipeline);
        set_buffer(encoder.as_ref(), &input_buffer, 0);
        set_buffer(encoder.as_ref(), &output_buffer, 1);
        set_bytes_u32(encoder.as_ref(), 2, &[elements])?;
        let thread_width = self
            .identity_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(input.len());
        if thread_width == 0 {
            bail!("Metal pipeline reported zero threads per threadgroup");
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: input.len(),
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

        let mut output = vec![0.0_f32; input.len()];
        output_buffer.read_f32(&mut output)?;
        Ok(output)
    }
}

/// Executes the identity smoke kernel and returns a fresh output vector.
pub fn identity_forward(input: &[f32]) -> Result<Vec<f32>> {
    MetalRuntime::new()?.identity(input)
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
    fn identity_shader_compiles_on_the_local_metal_device() {
        if let Ok(width) = validate_metal_pipeline(MetalDispatchShape::new(1, 8, 16).unwrap()) {
            assert!(width > 0);
        }
    }

    #[test]
    fn identity_kernel_round_trips_fp32_data_when_metal_is_available() {
        let input = [-1.0, 0.0, 0.5, 3.25];
        if let Ok(output) = identity_forward(&input) {
            assert_eq!(output, input);
        }
    }
}
