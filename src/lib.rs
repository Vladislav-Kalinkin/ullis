//! Ullis AI Engine v0.7 Cognitive Core — fused Metal MoB-KAN + Accelerate SME.

pub mod accelerate;
pub mod chat;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod device;
pub mod gauss;
pub mod kan;
pub mod mixers;
pub mod model;
pub mod optim;
pub mod quant;
pub mod seed;
pub mod telemetry;
pub mod tensor;
pub mod think;
pub mod tokenizer;
pub mod train;

pub use accelerate::MobKanSpec;
pub use config::TrainConfig;
pub use device::SovereignDevice;
pub use kan::TernaryKanLinear;
pub use model::UllisKan;
pub use quant::{pack_ternary, unpack_ternary};
pub use tensor::SovereignTensor;
pub use think::ThinkingMode;
pub use tokenizer::BpeTokenizer;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
