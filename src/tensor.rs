//! `SovereignTensor` — host `Vec<f32>` plus an isolated Metal buffer descriptor.
//!
//! No Candle `Tensor`. The host vector is the CPU/Accelerate source of truth.
//! The Metal buffer is a Shared unified-memory allocation owned by this value
//! and released on drop. All pointer casts go through `device` (the only
//! `unsafe` island for Metal).

use anyhow::{bail, Result};

use crate::accelerate::{mob_kan_fused_cpu, MobKanSpec};
use crate::device::{self, Backend, SovereignDevice};

/// Lightweight f32 tensor with manual host/device pipeline ownership.
pub struct SovereignTensor {
    shape: Vec<usize>,
    host: Vec<f32>,
    host_gen: u64,
    device_gen: u64,
    #[cfg(target_os = "macos")]
    gpu: Option<GpuSlot>,
}

#[cfg(target_os = "macos")]
struct GpuSlot {
    buffer: metal::Buffer,
    floats: usize,
}

impl std::fmt::Debug for SovereignTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovereignTensor")
            .field("shape", &self.shape)
            .field("numel", &self.numel())
            .field("host_gen", &self.host_gen)
            .field("device_gen", &self.device_gen)
            .field("gpu", &self.has_gpu())
            .finish()
    }
}

impl Clone for SovereignTensor {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            host: self.host.clone(),
            host_gen: 1,
            device_gen: 0,
            #[cfg(target_os = "macos")]
            gpu: None,
        }
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
        Ok(Self {
            shape,
            host: data,
            host_gen: 1,
            device_gen: 0,
            #[cfg(target_os = "macos")]
            gpu: None,
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        let n = numel_shape(&shape)?;
        Self::from_vec(shape, vec![0.0; n])
    }

    pub fn fill(shape: Vec<usize>, value: f32) -> Result<Self> {
        let n = numel_shape(&shape)?;
        Self::from_vec(shape, vec![value; n])
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.host.len()
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

    pub fn host_dirty(&self) -> bool {
        self.host_gen != self.device_gen && self.has_gpu()
    }

    pub fn device_dirty(&self) -> bool {
        self.device_gen > self.host_gen
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.host
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.host_gen = self
            .host_gen
            .saturating_add(1)
            .max(self.device_gen.saturating_add(1));
        &mut self.host
    }

    /// Bind a Shared Metal buffer. Idempotent if the existing buffer is sized.
    pub fn attach(&mut self, gpu: &SovereignDevice) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            let Some(mtl) = gpu.mtl_device() else {
                return Ok(());
            };
            let n = self.numel().max(1);
            let reuse = self
                .gpu
                .as_ref()
                .is_some_and(|slot| slot.floats >= self.numel().max(1));
            if !reuse {
                let buffer = device::alloc_shared_f32_buffer(mtl, n)?;
                self.gpu = Some(GpuSlot { buffer, floats: n });
                self.device_gen = 0;
            }
            self.upload()?;
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = gpu;
            Ok(())
        }
    }

    /// Host → Shared buffer. No-op when already in sync.
    pub fn upload(&mut self) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            if self.host_gen == self.device_gen {
                return Ok(());
            }
            let Some(slot) = self.gpu.as_ref() else {
                return Ok(());
            };
            device::write_shared_f32_buffer(&slot.buffer, &self.host)?;
            self.device_gen = self.host_gen;
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(())
        }
    }

    /// Shared buffer → host. Call after a GPU kernel writes this tensor.
    pub fn download(&mut self) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            let Some(slot) = self.gpu.as_ref() else {
                return Ok(());
            };
            device::read_shared_f32_buffer(&slot.buffer, &mut self.host)?;
            self.host_gen = self.host_gen.saturating_add(1);
            self.device_gen = self.host_gen;
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(())
        }
    }

    /// Drop the Metal buffer, keeping host data. Releases unified pages.
    pub fn detach_gpu(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.gpu = None;
            self.device_gen = 0;
        }
    }

    #[cfg(target_os = "macos")]
    pub fn metal_buffer(&self) -> Option<&metal::Buffer> {
        self.gpu.as_ref().map(|s| &s.buffer)
    }

    pub fn reshape(&mut self, shape: Vec<usize>) -> Result<()> {
        let n = numel_shape(&shape)?;
        if n != self.numel() {
            bail!("reshape {} -> {shape:?} changes numel {}", self.numel(), n);
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
    check_len(tensors.w_base, spec.w_base_len(), "w_base")?;
    check_len(tensors.w_shared, spec.w_shared_len(), "w_shared")?;
    check_len(tensors.centers, spec.centers_len(), "centers")?;
    check_len(tensors.inv_widths, spec.centers_len(), "inv_widths")?;
    check_len(tensors.scale_base, spec.scale_vec_len(), "scale_base")?;
    check_len(tensors.scale_shared, spec.scale_vec_len(), "scale_shared")?;
    if !spec.mask_routed() {
        let wr = tensors
            .w_routed
            .ok_or_else(|| anyhow::anyhow!("w_routed required"))?;
        let rt = tensors
            .router
            .ok_or_else(|| anyhow::anyhow!("router required"))?;
        check_len(wr, spec.w_routed_len(), "w_routed")?;
        check_len(rt, spec.router_len(), "router")?;
        check_len(
            tensors.scale_routed,
            spec.scale_routed_len(),
            "scale_routed",
        )?;
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
    // Uploads must happen on &mut tensors. We cannot mut-borrow all inputs
    // through FusedKanTensors (they are shared refs). Callers attach+upload
    // weights; we upload y's output buffer and require inputs already synced.
    tensors.y.upload()?;
    {
        let x = tensors
            .x
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("x has no Metal buffer; call attach()"))?;
        let y = tensors
            .y
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("y has no Metal buffer; call attach()"))?;
        let w_base = tensors
            .w_base
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("w_base has no Metal buffer"))?;
        let w_shared = tensors
            .w_shared
            .metal_buffer()
            .ok_or_else(|| anyhow::anyhow!("w_shared has no Metal buffer"))?;
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
        let dummy = gpu.dummy_buffer();
        let w_routed = tensors
            .w_routed
            .and_then(SovereignTensor::metal_buffer)
            .unwrap_or(dummy);
        let router = tensors
            .router
            .and_then(SovereignTensor::metal_buffer)
            .unwrap_or(dummy);

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
        )?;
    }
    tensors.y.download()?;
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
    }

    #[test]
    fn cpu_fused_via_tensors() {
        let gpu = SovereignDevice::open(false).unwrap();
        let spec = MobKanSpec::new(2, 4, 3, 4, 3, 1, 3, 3, 1, false, false, 1.5, 0.7).unwrap();
        let x = SovereignTensor::fill(vec![2, 4], 0.2).unwrap();
        let mut y = SovereignTensor::zeros(vec![2, 3]).unwrap();
        let w_base = SovereignTensor::fill(vec![3, 4], 0.05).unwrap();
        let w_shared = SovereignTensor::fill(vec![3, 12], 0.02).unwrap();
        let w_routed = SovereignTensor::fill(vec![3, 3, 4], 0.01).unwrap();
        let router = SovereignTensor::zeros(vec![3, 4]).unwrap();
        let centers = SovereignTensor::from_vec(vec![4], vec![-2.0, -0.66, 0.66, 2.0]).unwrap();
        let iw = crate::accelerate::bump_inv_widths(centers.as_slice());
        let inv_widths = SovereignTensor::from_vec(vec![4], iw).unwrap();
        let scale_base = SovereignTensor::fill(vec![3], 1.0).unwrap();
        let scale_shared = SovereignTensor::fill(vec![3], 1.0).unwrap();
        let scale_routed = SovereignTensor::fill(vec![3, 3], 1.0).unwrap();
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
            },
        )
        .unwrap();
        assert!(y.as_slice().iter().all(|v| v.is_finite()));
    }
}
