//! Ullis AI Engine v0.8 Deep Context — byte-fallback WordPiece, packed-i8 embed.

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
pub use data::SovereignFlashBuffer;
pub use device::SovereignDevice;
pub use kan::TernaryKanLinear;
pub use model::UllisKan;
pub use quant::{pack_i8_rows, pack_ternary, unpack_i8_rows, unpack_ternary, PackedI8Matrix};
pub use tensor::SovereignTensor;
pub use think::ThinkingMode;
pub use tokenizer::BpeTokenizer;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
