//! Real-time RAM / tok/s / ternary-entropy HUD.
//!
//! The macOS probe is the only `unsafe` in the crate: a single `task_info`
//! syscall behind a safe wrapper.

#![allow(unsafe_code)]

use std::mem::{size_of, MaybeUninit};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static GPU_WAITS: AtomicU64 = AtomicU64::new(0);

/// Count a Metal `wait_until_completed` on the fused/i8 path.
pub fn record_gpu_wait() {
    GPU_WAITS.fetch_add(1, Ordering::Relaxed);
}

/// Swap-out the wait counter (HUD).
pub fn take_gpu_waits() -> u64 {
    GPU_WAITS.swap(0, Ordering::Relaxed)
}

use crate::quant::TernaryHist;

static METAL_HELLO_MB: OnceLock<f64> = OnceLock::new();

/// Cache RSS after `SovereignDevice::open` (empty fused pipeline, no model).
pub fn cache_metal_hello_mb(mb: f64) {
    let _ = METAL_HELLO_MB.set(mb);
}

pub fn metal_hello_mb() -> f64 {
    METAL_HELLO_MB.get().copied().unwrap_or(0.0)
}

/// Split train-memory HUD. `gpu_alias` is 1 when Metal wraps PageSlab tensors.
#[derive(Clone, Debug, Default)]
pub struct TrainFootprint {
    pub rss_mb: f64,
    pub baseline_metal_mb: f64,
    pub net_mb: f64,
    pub params_bytes: u64,
    pub grad_bytes: u64,
    pub opt_bytes: u64,
    pub workspace_bytes: u64,
    pub gpu_alias: u8,
    pub embed_i8_bytes: u64,
    pub scratch_bumps: u64,
}

impl TrainFootprint {
    pub fn format_fields(&self) -> String {
        format!(
            " net={:.1}MB params={:.1}kB grad={:.1}kB opt={:.1}kB ws={:.1}kB i8={:.1}kB alias={}",
            self.net_mb,
            self.params_bytes as f64 / 1024.0,
            self.grad_bytes as f64 / 1024.0,
            self.opt_bytes as f64 / 1024.0,
            self.workspace_bytes as f64 / 1024.0,
            self.embed_i8_bytes as f64 / 1024.0,
            self.gpu_alias
        )
    }
}

#[derive(Clone, Debug)]
pub struct Hud {
    pub rss_mb: f64,
    pub tok_s: f64,
    pub hist: TernaryHist,
    pub extra: String,
}

impl Hud {
    pub fn format_line(&self, prefix: &str) -> String {
        format!(
            "{prefix} rss={:.1}MB tok/s={:.1} zero={:.2} +={:.2} -={:.2}{}",
            self.rss_mb,
            self.tok_s,
            self.hist.frac_zero,
            self.hist.frac_pos,
            self.hist.frac_neg,
            self.extra
        )
    }
}

pub struct Throughput {
    t0: Instant,
    tokens: u64,
}

impl Throughput {
    pub fn new() -> Self {
        Self {
            t0: Instant::now(),
            tokens: 0,
        }
    }

    pub fn add(&mut self, n: u64) {
        self.tokens += n;
    }

    pub fn tok_s(&self) -> f64 {
        let dt = self.t0.elapsed().as_secs_f64().max(1e-9);
        self.tokens as f64 / dt
    }

    pub fn reset(&mut self) {
        self.t0 = Instant::now();
        self.tokens = 0;
    }
}

impl Default for Throughput {
    fn default() -> Self {
        Self::new()
    }
}

/// Active process memory in bytes (unified footprint on Apple Silicon).
pub fn process_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        macos_memory_bytes().unwrap_or(0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

pub fn process_memory_mb() -> f64 {
    process_memory_bytes() as f64 / (1024.0 * 1024.0)
}

#[cfg(target_os = "macos")]
fn macos_memory_bytes() -> Option<u64> {
    macos_phys_footprint().or_else(macos_resident_size)
}

/// `task_vm_info.phys_footprint` — compressed-memory-aware unique footprint.
#[cfg(target_os = "macos")]
#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct MachTaskBasicInfo {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time_seconds: i32,
    user_time_microseconds: i32,
    system_time_seconds: i32,
    system_time_microseconds: i32,
    policy: i32,
    suspend_count: i32,
}

#[cfg(target_os = "macos")]
fn macos_phys_footprint() -> Option<u64> {
    unsafe {
        task_info_struct::<TaskVmInfo>(mach2::task_info::TASK_VM_INFO).map(|i| i.phys_footprint)
    }
}

#[cfg(target_os = "macos")]
fn macos_resident_size() -> Option<u64> {
    unsafe {
        task_info_struct::<MachTaskBasicInfo>(mach2::task_info::MACH_TASK_BASIC_INFO)
            .map(|i| i.resident_size)
    }
}

/// Isolated `task_info(mach_task_self(), flavor)` read. One unsafe block.
#[cfg(target_os = "macos")]
unsafe fn task_info_struct<T>(flavor: u32) -> Option<T> {
    use mach2::kern_return::KERN_SUCCESS;
    use mach2::message::mach_msg_type_number_t;
    use mach2::task::task_info;
    use mach2::traps::mach_task_self;

    let mut info = MaybeUninit::<T>::uninit();
    let mut count: mach_msg_type_number_t =
        (size_of::<T>() / size_of::<mach2::vm_types::natural_t>()) as mach_msg_type_number_t;
    let kr = unsafe {
        task_info(
            mach_task_self(),
            flavor,
            info.as_mut_ptr().cast(),
            &raw mut count,
        )
    };
    if kr == KERN_SUCCESS {
        Some(unsafe { info.assume_init() })
    } else {
        None
    }
}

pub fn print_hud(line: &str) {
    println!("{line}");
}
