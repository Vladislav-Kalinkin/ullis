//! Ullis AI Engine v0.5 — ternary Kolmogorov–Arnold visual reasoning stream in Rust.

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
pub mod think;
pub mod tokenizer;
pub mod train;

pub use config::TrainConfig;
pub use kan::TernaryKanLinear;
pub use model::UllisKan;
pub use quant::{pack_ternary, unpack_ternary};
pub use think::ThinkingMode;
pub use tokenizer::BpeTokenizer;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
