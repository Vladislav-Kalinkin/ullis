//! Ullis: a dense ternary Hyena language-model core for Apple Silicon.
//! KAN, recurrent slot memory, routing, and experts are intentionally absent.
pub mod batch;
pub mod config;
pub mod hyena;
pub mod metal;
pub mod model;
pub mod optimizer;
pub mod precision;
pub mod tokenizer;
pub use batch::{MtpBatch, MtpBatcher};
pub use config::{
    LowMemoryTrainingEstimate, LowMemoryTrainingProfile, MemoryEstimate, TrainConfig,
};
pub use hyena::{CausalConvBackward, HyenaChunkPlan, ImplicitFilterBackward};
#[cfg(target_os = "macos")]
pub use metal::{
    MetalHyenaBlockUpdatedBackward, MetalResidentCrossEntropy, MetalStreamedCrossEntropy,
    ResidentFp16Parameters, ResidentGradientSlot, ResidentImplicitFilterParameters,
    ResidentTrainableFp16TernaryWeights,
};
pub use model::{
    HyenaGateBackward, ModelCheckpoint, MtpHeadBackward, MtpLoss, MtpProjectionBackward,
    TernaryLinearBackward, UllisHyena, hyena_gate_backward,
};
#[cfg(target_os = "macos")]
pub use model::{
    MetalHyenaStackBackward, MetalResidentHyenaProjectionStep, MetalResidentHyenaTrainingState,
};
pub use optimizer::{Lion, LionConfig, OptimizerKind};
pub use precision::{Fp16, Fp16Storage};
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
