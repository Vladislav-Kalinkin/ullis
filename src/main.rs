use anyhow::Result;
use clap::Parser;

/// Prints the architecture selected by this refactor. Training and Metal
/// dispatch are intentionally added only after the CPU reference path is
/// numerically validated.
#[derive(Debug, Parser)]
#[command(name = "ullis", version, about = "Dense ternary Hyena core")]
struct Cli {
    /// Validate a minimal causal forward pass.
    #[arg(long)]
    smoke: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.smoke {
        let cfg = ullis::TrainConfig {
            vocab_size: 512,
            d_model: 16,
            n_layers: 1,
            context_len: 32,
            ..Default::default()
        };
        let model = ullis::UllisHyena::new(cfg)?;
        let loss = model.streamed_mtp_loss(&[4, 5, 6, 7], 1, 4)?;
        println!(
            "hyena smoke: streamed MTP loss {:.4} ({}/{})",
            loss.mean(),
            loss.next_token_count,
            loss.second_token_count
        );
    } else {
        println!("Ullis is now a dense ternary Hyena core. Run --smoke to validate it.");
    }
    Ok(())
}
