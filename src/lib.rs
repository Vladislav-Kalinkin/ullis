//! Ullis AI Engine v0.9 Infinite Lexicon — expandable WordPiece, packed-i8 embed.

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
pub mod telemetry;
pub mod tensor;
pub mod think;
pub mod tokenizer;
pub mod train;

pub use accelerate::{FusedBwdGrads, MobKanSpec};
pub use device::prefer_host_bwd;
pub use config::{MasterDtype, MomDtype, TrainConfig};
pub use data::SovereignFlashBuffer;
pub use device::SovereignDevice;
pub use kan::TernaryKanLinear;
pub use model::UllisKan;
pub use quant::{
    f16_bits_to_f32, f32_to_f16_bits, pack_f16, pack_i8_rows, pack_ternary, unpack_f16,
    unpack_i8_rows, unpack_ternary, PackedI8Matrix,
};
pub use tensor::SovereignTensor;
pub use think::ThinkingMode;
pub use tokenizer::{validate_vocab_size, BpeTokenizer, DEFAULT_VOCAB, MAX_VOCAB, MIN_VOCAB};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
