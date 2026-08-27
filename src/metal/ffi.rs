//! Audited Metal FFI: shared-buffer mapping and encoder scalar writes.
//!
//! Crate-level `unsafe_code = "deny"` stays in force. This is the only module
//! allowed to talk to `MTLBuffer::contents` and `setBytes`.

#![allow(unsafe_code)]

use anyhow::{Result, bail};
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

/// Reads larger than this from `map` are traced. Checkpoint snapshots and
/// tests may exceed it; the train hot path must not.
pub const TRACE_COPY_THRESHOLD: usize = 4096;

pub type ComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

#[derive(Debug)]
pub struct MetalBuffer {
    inner: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

pub struct MappedBytes<'a> {
    ptr: *mut u8,
    len: usize,
    _owner: &'a MetalBuffer,
}

impl std::fmt::Debug for MappedBytes<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedBytes")
            .field("len", &self.len)
            .finish()
    }
}

impl MetalBuffer {
    pub(crate) fn from_retained(
        inner: Retained<ProtocolObject<dyn MTLBuffer>>,
        len: usize,
    ) -> Self {
        Self { inner, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn as_mtl(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.inner
    }

    /// Maps `byte_len` bytes of shared storage for writing.
    pub fn map_mut(&self, byte_len: usize) -> Result<MappedBytes<'_>> {
        self.map_inner(byte_len, false)
    }

    /// Maps `byte_len` bytes of shared storage for reading. Reads above
    /// [`TRACE_COPY_THRESHOLD`] are logged so residency tests can catch them.
    pub fn map(&self, byte_len: usize) -> Result<MappedBytes<'_>> {
        self.map_inner(byte_len, true)
    }

    pub fn write_f32(&self, values: &[f32]) -> Result<()> {
        let byte_len = values
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal f32 write size overflow"))?;
        let mut mapped = self.map_mut(byte_len)?;
        mapped.as_mut_slice().copy_from_slice(f32_as_bytes(values));
        Ok(())
    }

    pub fn read_f32(&self, values: &mut [f32]) -> Result<()> {
        let byte_len = values
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Metal f32 read size overflow"))?;
        let mapped = self.map(byte_len)?;
        f32_as_bytes_mut(values).copy_from_slice(mapped.as_slice());
        Ok(())
    }

    fn map_inner(&self, byte_len: usize, is_read: bool) -> Result<MappedBytes<'_>> {
        if byte_len > self.len {
            bail!("Metal map length {byte_len} exceeds buffer {}", self.len);
        }
        if is_read && byte_len > TRACE_COPY_THRESHOLD {
            eprintln!("trace_metal_copies: map read {byte_len} bytes");
        }
        // SAFETY: the buffer is StorageModeShared, retained for `'self`, and
        // `byte_len` was checked against the allocation. Unified-memory
        // `contents()` is valid for CPU access while the buffer lives.
        let ptr = { self.inner.contents().as_ptr().cast::<u8>() };
        Ok(MappedBytes {
            ptr,
            len: byte_len,
            _owner: self,
        })
    }
}

impl MappedBytes<'_> {
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` came from a retained shared MTLBuffer and `len` was
        // checked against that allocation in `map_inner`.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: same as `as_slice`; the map is exclusive for this owner.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

pub fn set_bytes_u32(encoder: &ComputeEncoder, index: usize, values: &[u32]) -> Result<()> {
    if values.is_empty() {
        bail!("set_bytes_u32 requires at least one value");
    }
    let bytes = values
        .len()
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| anyhow::anyhow!("set_bytes_u32 length overflow"))?;
    // SAFETY: `setBytes` copies `bytes` synchronously from `values`, which
    // stays live for the duration of the call. `index` is the MSL buffer slot.
    unsafe {
        encoder.setBytes_length_atIndex(NonNull::from(&values[0]).cast::<c_void>(), bytes, index);
    }
    Ok(())
}

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    let len = values.len().saturating_mul(size_of::<f32>());
    // SAFETY: `values` is a live `[f32]` occupying `len` bytes.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), len) }
}

fn f32_as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    let len = values.len().saturating_mul(size_of::<f32>());
    // SAFETY: `values` is a live exclusive `[f32]` occupying `len` bytes.
    unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len) }
}

pub fn set_buffer(encoder: &ComputeEncoder, buffer: &MetalBuffer, index: usize) {
    // SAFETY: `buffer` is retained by the caller for the lifetime of the
    // command encoder / command buffer that consumes this binding.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer.as_mtl()), 0, index);
    }
}
