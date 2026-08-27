//! Ullis: RWKV-8 Heron / ROSA language-model core for Apple Silicon.
pub mod batch;
pub mod config;
pub mod model;
pub mod optimizer;
pub mod precision;
pub mod rosa;
pub mod tokenizer;
#[cfg(target_os = "macos")]
pub mod metal;
pub use batch::{CausalBatch, CausalBatcher};
pub use config::{Architecture, MemoryEstimate, RosaGradMode, TrainConfig};
pub use model::{CausalLoss, ModelCheckpoint, UllisHeron};
pub use optimizer::{Lion, LionConfig, OptimizerKind};
pub use precision::{Fp16, Fp16Storage};
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
