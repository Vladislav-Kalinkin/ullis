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
        self.write_bytes(f32_as_bytes(values))
    }

    pub fn read_f32(&self, values: &mut [f32]) -> Result<()> {
        self.read_bytes(f32_as_bytes_mut(values))
    }

    pub fn write_u16(&self, values: &[u16]) -> Result<()> {
        self.write_bytes(u16_as_bytes(values))
    }

    pub fn read_u16(&self, values: &mut [u16]) -> Result<()> {
        self.read_bytes(u16_as_bytes_mut(values))
    }

    pub fn write_u32(&self, values: &[u32]) -> Result<()> {
        self.write_bytes(u32_as_bytes(values))
    }

    pub fn read_u32(&self, values: &mut [u32]) -> Result<()> {
        self.read_bytes(u32_as_bytes_mut(values))
    }

    pub fn write_bytes(&self, values: &[u8]) -> Result<()> {
        let mut mapped = self.map_mut(values.len())?;
        mapped.as_mut_slice().copy_from_slice(values);
        Ok(())
    }

    pub fn read_bytes(&self, values: &mut [u8]) -> Result<()> {
        let mapped = self.map(values.len())?;
        values.copy_from_slice(mapped.as_slice());
        Ok(())
    }

    pub fn zero(&self) -> Result<()> {
        let mut mapped = self.map_mut(self.len)?;
        mapped.as_mut_slice().fill(0);
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
    set_bytes(encoder, index, u32_as_bytes(values), "set_bytes_u32")
}

pub fn set_bytes_f32(encoder: &ComputeEncoder, index: usize, values: &[f32]) -> Result<()> {
    set_bytes(encoder, index, f32_as_bytes(values), "set_bytes_f32")
}

fn set_bytes(encoder: &ComputeEncoder, index: usize, bytes: &[u8], name: &str) -> Result<()> {
    if bytes.is_empty() {
        bail!("{name} requires at least one value");
    }
    // SAFETY: `setBytes` copies `bytes` synchronously from `bytes`, which
    // stays live for the duration of the call. `index` is the MSL buffer slot.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(&bytes[0]).cast::<c_void>(),
            bytes.len(),
            index,
        );
    }
    Ok(())
}

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    as_bytes(values)
}

fn f32_as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    as_bytes_mut(values)
}

fn u16_as_bytes(values: &[u16]) -> &[u8] {
    as_bytes(values)
}

fn u16_as_bytes_mut(values: &mut [u16]) -> &mut [u8] {
    as_bytes_mut(values)
}

fn u32_as_bytes(values: &[u32]) -> &[u8] {
    as_bytes(values)
}

fn u32_as_bytes_mut(values: &mut [u32]) -> &mut [u8] {
    as_bytes_mut(values)
}

fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    let len = values.len().saturating_mul(size_of::<T>());
    // SAFETY: `values` is a live `[T]` occupying `len` bytes.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), len) }
}

fn as_bytes_mut<T: Copy>(values: &mut [T]) -> &mut [u8] {
    let len = values.len().saturating_mul(size_of::<T>());
    // SAFETY: `values` is a live exclusive `[T]` occupying `len` bytes.
    unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len) }
}

pub fn set_buffer(encoder: &ComputeEncoder, buffer: &MetalBuffer, index: usize) {
    // SAFETY: `buffer` is retained by the caller for the lifetime of the
    // command encoder / command buffer that consumes this binding.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer.as_mtl()), 0, index);
    }
}
