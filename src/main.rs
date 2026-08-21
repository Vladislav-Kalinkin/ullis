//! Ullis v0.9 Infinite Lexicon — WordPiece V≥8192, packed-i8 embed, flash buffer.
//!
//! # Compilation graph
//! ```text
//! tokens → embed[V,D]
//!        → L × KanBlock{
//!             RMSNorm → causal mixer (shift | 1-head attn)
//!             RMSNorm → TernaryKanLinear  φ_ji = a_ji x_i + Σ_g b_jig ψ_g(x_i)
//!                       ψ_g = relu(1 − |x − c_g|/w)²
//!                       G split shared/routed (MoB); non-uniform knot insert G=4…12
//!                       router softmax(x W_rᵀ), K=3 (python|rust|bash)
//!                       --thinking xhigh: +3 residual KAN loops (mixer once)
//!          }
//!        → RMSNorm → tied logits = h @ embedᵀ
//! ```
//! STE QAT: forward `{-1,0,+1}`, backward hardtanh `|w|≤1`. Packed 2-bit on disk,
//! int8 codes in RAM (Metal has no 2-bit GEMM). Tied embed stored once.
//!
//! # Memory vectors (flat inference envelope, target < 35 MB RSS)
//! - `DialogueCache` — persistent `system` + `(user, output)` ring
//!   (`DIALOGUE_TURN_CAP=6`, `DIALOGUE_CHAR_CAP=3072`). Never stores thinking.
//! - `ReasoningScratch` — ephemeral think tokens/text. `clear()`/`wipe()` zeros,
//!   drops, and `shrink_to_fit`s the instant the output stream ends.
//! - `JsonlStream` `SovereignFlashBuffer` token ring, cap 32 768 ids (~128 KB), O(seq_len)
//!   independent of corpus size. JSONL via `BufReader` + `serde_json`.
//! - Working set: `d=32`, `L=3`, `G=12`, `V=8192`, `T=96`, `B=4`, SGD (no Adam).
//!
//! # Gauss–Jordan GPU solver (`gauss::mps_safe_solve`)
//! Grid lift G=4→8→12 samples `M=max(64,16G)` points and solves
//! `Ψ_newᵀ Ψ_new b' = Ψ_newᵀ Ψ_old b` on-device. Elementwise / broadcast / `cat`
//! only (no `linalg` CPU fallback on the happy path). Ridge `1e-6 · mean(diag)`,
//! pivot floor `1e-8`. Host `gauss_jordan_f32` is the G≤16 fallback.
//!
//! Binary: `ullis train | chat | smoke`. License: AGPL-3.0.

use anyhow::Result;
use clap::{Parser, Subcommand};

use ullis::chat::{run_chat, ChatArgs};
use ullis::train::{run_smoke, train, TrainArgs};

#[derive(Parser)]
#[command(
    name = "ullis",
    version,
    about = "Ullis AI Engine v0.9 Infinite Lexicon — WordPiece V≥8192 / packed-i8 embed"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// 4-phase ternary QAT from a streaming JSONL corpus.
    Train(TrainArgs),
    /// Interactive streaming REPL (or --prompt for one-shot). `--thinking` sets the budget.
    Chat(ChatArgs),
    /// Numerical smoke: STE, Gauss–Jordan, pack, generate.
    Smoke {
        #[arg(long)]
        cpu: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train(args) => {
            train(args)?;
        }
        Command::Chat(args) => {
            run_chat(args)?;
        }
        Command::Smoke { cpu } => {
            run_smoke(cpu)?;
        }
    }
    Ok(())
}
