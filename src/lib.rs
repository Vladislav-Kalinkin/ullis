//! Ullis: RWKV-8 Heron / ROSA language-model core for Apple Silicon.
pub mod batch;
pub mod config;
pub mod conversation;
pub mod decode;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod model;
pub mod optimizer;
pub mod precision;
pub mod rosa;
pub mod tokenizer;
pub mod wkv7;
pub use batch::{CausalBatch, CausalBatcher};
pub use config::{Architecture, MemoryEstimate, RosaGradMode, TrainConfig};
pub use conversation::{ChatMessage, generation_prefix, render_messages};
pub use decode::{DecodeConfig, apply_openai_penalties, select_token};
pub use model::{
    CE_NO_IGNORE, CausalLoss, CheckpointInspect, HEAD_BIAS_LR_MULT, HeadUnigramInstall,
    HeronGenerateState, InferenceStateBytes, ModelCheckpoint, ParamCounts, TrainStepProfile,
    UllisHeron, causal_ce_gradient_scale,
};
pub use optimizer::{Lion, LionConfig, OptimizerKind};
pub use precision::{Fp16, Fp16Storage};
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
