use anyhow::{Context, Result};
use candle_core::{DType, Device};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn rng_from_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

pub fn setup_device(prefer_metal: bool) -> Result<Device> {
    if prefer_metal {
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(d) => return Ok(d),
                Err(e) => {
                    eprintln!("ullis: Metal unavailable ({e}); falling back to CPU");
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = prefer_metal;
        }
    }
    Ok(Device::Cpu)
}

pub fn amp_dtype(_device: &Device) -> DType {
    DType::F32
}

pub fn device_name(device: &Device) -> &'static str {
    if device.is_metal() {
        "metal"
    } else if device.is_cuda() {
        "cuda"
    } else {
        "cpu"
    }
}

pub fn tensor_to_vec1_f32(t: &candle_core::Tensor) -> Result<Vec<f32>> {
    let cpu = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    cpu.flatten_all()?
        .to_vec1::<f32>()
        .map_err(|e| anyhow::anyhow!(e))
}

pub fn scalar_f32(t: &candle_core::Tensor) -> Result<f32> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()
        .context("scalar_f32")
}

/// Wait for in-flight Metal work and drop unused MTLBuffers.
///
/// Candle's Metal backend only returns command-buffer scratch to the pool
/// from `wait_until_completed` → `drop_unused_buffers`. Skipping this in a
/// training loop leaks device memory even after Rust drops the tensors.
pub fn synchronize(device: &Device) -> Result<()> {
    device.synchronize().map_err(|e| anyhow::anyhow!(e))
}
