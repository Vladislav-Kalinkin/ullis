use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullis::tokenizer::{train_bpe, BpeTokenizer, DEFAULT_VOCAB, MIN_VOCAB};
use ullis::{Architecture, CausalBatcher, ModelCheckpoint, OptimizerKind, TrainConfig, UllisHeron};

#[derive(Debug, Parser)]
#[command(name = "ullis", version, about = "RWKV-8 Heron / ROSA training tools")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(long)]
    smoke: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[command(flatten)]
        overrides: Box<TrainOverrides>,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f32,
        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,
        #[arg(long, value_enum, default_value_t = Backend::Metal)]
        backend: Backend,
        #[arg(long, default_value_t = 16)]
        bpe_train_mib: usize,
    },
    Tokenize {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Inspect {
        #[arg(long)]
        run: PathBuf,
    },
    Chat {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value = "sessions/default.jsonl")]
        session: PathBuf,
        #[arg(long, value_enum, default_value_t = ThinkingLevel::Medium)]
        thinking: ThinkingLevel,
    },
    Generate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
    },
}

#[derive(Clone, Debug, Args, Default)]
struct TrainOverrides {
    #[arg(long, value_enum)]
    architecture: Option<ArchitectureArg>,
    #[arg(long)]
    d_model: Option<usize>,
    #[arg(long = "layers")]
    n_layers: Option<usize>,
    #[arg(long)]
    context_len: Option<usize>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    vocab_size: Option<usize>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    memory_budget_mib: Option<usize>,
    #[arg(long, value_enum)]
    optimizer: Option<OptimizerArg>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ArchitectureArg {
    Heron,
    RosaRwkv7,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OptimizerArg {
    StatelessSgd,
    LionFp16,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ThinkingLevel {
    Low,
    Medium,
    High,
    Xhigh,
    Off,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Metal,
    Cpu,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DatasetRecord {
    id: String,
    messages: Vec<DatasetMessage>,
    #[serde(default)]
    metadata: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DatasetMessage {
    role: String,
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ToolCall {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

fn validate_record(record: &DatasetRecord) -> Result<()> {
    if record.id.trim().is_empty() || record.messages.is_empty() {
        bail!("dataset record needs a non-empty id and messages");
    }
    for message in &record.messages {
        match message.role.as_str() {
            "system" | "user" | "assistant" | "tool" => {}
            role => bail!("record {} has unsupported role {role:?}", record.id),
        }
        if message.role == "assistant"
            && message
                .thinking
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            bail!(
                "assistant message in record {} must contain non-empty thinking",
                record.id
            );
        }
        if message.role == "tool" && message.tool_call_id.as_deref().is_none_or(str::is_empty) {
            bail!("tool message in record {} needs tool_call_id", record.id);
        }
    }
    Ok(())
}

fn load_dataset(path: &Path) -> Result<Vec<DatasetRecord>> {
    let file = File::open(path).with_context(|| format!("open dataset {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(Ok((index + 1, line))),
            Err(error) => Some(Err(error.into())),
        })
        .map(|entry: Result<(usize, String)>| {
            let (index, line) = entry?;
            let record: DatasetRecord = serde_json::from_str(&line)
                .with_context(|| format!("parse dataset line {index}"))?;
            validate_record(&record).with_context(|| format!("validate dataset line {index}"))?;
            Ok(record)
        })
        .collect()
}

fn default_train_config(vocab_size: usize) -> TrainConfig {
    TrainConfig {
        vocab_size,
        ..Default::default()
    }
}

fn smoke_config(vocab_size: usize) -> TrainConfig {
    TrainConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size,
        context_len: 32,
        dim_ffn: 64,
        tmix_lora_rank: 8,
        ..Default::default()
    }
}

fn apply_overrides(cfg: &mut TrainConfig, overrides: &TrainOverrides) -> Result<()> {
    if let Some(architecture) = overrides.architecture {
        cfg.architecture = match architecture {
            ArchitectureArg::Heron => Architecture::Heron,
            ArchitectureArg::RosaRwkv7 => Architecture::RosaRwkv7,
        };
    }
    if let Some(value) = overrides.d_model {
        cfg.d_model = value;
        if cfg.dim_ffn == 0 || cfg.dim_ffn == 4 * 256 {
            cfg.dim_ffn = value.saturating_mul(4);
        }
        if overrides.architecture.is_none() {
            cfg.tmix_lora_rank = if value <= 64 { 8 } else { 16 };
        }
    }
    if let Some(value) = overrides.n_layers {
        cfg.n_layers = value;
    }
    if let Some(value) = overrides.context_len {
        cfg.context_len = value;
    }
    if let Some(value) = overrides.batch_size {
        cfg.batch_size = value;
    }
    if let Some(value) = overrides.vocab_size {
        cfg.vocab_size = value;
    }
    if let Some(value) = overrides.seed {
        cfg.seed = value;
    }
    if let Some(mib) = overrides.memory_budget_mib {
        cfg.memory_budget_bytes = mib
            .checked_mul(1024 * 1024)
            .ok_or_else(|| anyhow::anyhow!("memory-budget-mib overflows bytes"))?;
    }
    match overrides.optimizer {
        Some(OptimizerArg::LionFp16) => cfg.optimizer = OptimizerKind::LionFp16,
        Some(OptimizerArg::StatelessSgd) | None => cfg.optimizer = OptimizerKind::StatelessSgd,
    }
    Ok(())
}

fn load_config(path: Option<PathBuf>, overrides: &TrainOverrides) -> Result<TrainConfig> {
    let mut cfg = match path {
        Some(path) => {
            let source = fs::read_to_string(&path)?;
            toml::from_str(&source)
                .or_else(|_| serde_json::from_str(&source))
                .with_context(|| format!("parse TOML or JSON config {}", path.display()))?
        }
        None => default_train_config(DEFAULT_VOCAB as usize),
    };
    apply_overrides(&mut cfg, overrides)?;
    Ok(cfg)
}

fn training_text(record: &DatasetRecord) -> String {
    let mut out = String::new();
    for message in &record.messages {
        match message.role.as_str() {
            "system" => {
                out.push_str("<system>");
                out.push_str(&message.content);
                out.push_str("</system>");
            }
            "user" => {
                out.push_str("<user>");
                out.push_str(&message.content);
                out.push_str("</user>");
            }
            "assistant" => {
                out.push_str("<assistant><thinking>");
                out.push_str(message.thinking.as_deref().unwrap_or(""));
                out.push_str("</thinking>");
                out.push_str(&message.content);
                out.push_str("</assistant>");
            }
            "tool" => {
                out.push_str("<tool>");
                out.push_str(&message.content);
                out.push_str("</tool>");
            }
            _ => {}
        }
    }
    out
}

fn cycle_corpus(ids: &[u32], needed: usize) -> Result<Vec<u32>> {
    if ids.is_empty() {
        bail!("encoded corpus is empty");
    }
    Ok(ids.iter().copied().cycle().take(needed).collect())
}

fn write_metrics(path: &Path, row: &serde_json::Value) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{row}")?;
    Ok(())
}

fn train(
    data: PathBuf,
    run: PathBuf,
    config_path: Option<PathBuf>,
    overrides: TrainOverrides,
    resume: Option<PathBuf>,
    steps: usize,
    learning_rate: f32,
    checkpoint_every: usize,
    backend: Backend,
    _bpe_train_mib: usize,
) -> Result<()> {
    if steps == 0 || checkpoint_every == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
        bail!("steps, checkpoint-every, and learning-rate must be positive");
    }
    let mut cfg = load_config(config_path, &overrides)?;
    cfg.validate()?;
    if matches!(cfg.architecture, Architecture::RosaRwkv7) {
        bail!("rosa_rwkv7 train is not wired");
    }
    cfg.optimizer.require_train_step()?;
    let records = load_dataset(&data)?;
    let texts: Vec<String> = records.iter().map(training_text).collect();
    fs::create_dir_all(&run)?;
    let (mut model, mut tokenizer) = if let Some(path) = resume {
        load_model(&path)?
    } else {
        let tokenizer = train_bpe(&texts, cfg.vocab_size as u32, cfg.seed)?;
        cfg = cfg.with_tokenizer(&tokenizer)?;
        (UllisHeron::new(cfg)?, tokenizer)
    };
    tokenizer.save(run.join("tokenizer.json"))?;
    fs::write(
        run.join("config.json"),
        serde_json::to_string_pretty(&model.cfg)?,
    )?;

    let mut corpus = Vec::new();
    for text in &texts {
        corpus.extend(tokenizer.encode(text, true, true));
    }
    let needed = steps
        .saturating_mul(model.cfg.batch_size)
        .saturating_mul(model.cfg.context_len);
    let stream = cycle_corpus(&corpus, needed)?;
    let batcher = CausalBatcher::from_config(&stream, &model.cfg, model.cfg.context_len)?;

    #[cfg(target_os = "macos")]
    let metal = match backend {
        Backend::Metal => Some(ullis::metal::MetalRuntime::new()?),
        Backend::Cpu => None,
    };
    #[cfg(not(target_os = "macos"))]
    if matches!(backend, Backend::Metal) {
        bail!("Metal backend requires macOS on Apple Silicon");
    }

    let metrics_path = run.join("metrics.jsonl");
    let mut loss_ema = None;
    let ln_v = (model.cfg.vocab_size as f32).ln();
    let mut still_random = true;
    for (step_index, batch) in batcher.enumerate() {
        if step_index >= steps {
            break;
        }
        let started = Instant::now();
        #[cfg(target_os = "macos")]
        let loss = if let Some(runtime) = &metal {
            model.train_step_metal(
                runtime,
                batch.tokens(),
                batch.batch_size(),
                batch.time(),
                learning_rate,
            )?
        } else {
            model.train_step(
                batch.tokens(),
                batch.batch_size(),
                batch.time(),
                learning_rate,
            )?
        };
        #[cfg(not(target_os = "macos"))]
        let loss = model.train_step(
            batch.tokens(),
            batch.batch_size(),
            batch.time(),
            learning_rate,
        )?;
        let elapsed = started.elapsed();
        let step = step_index + 1;
        let ema = match loss_ema {
            None => loss.next_token,
            Some(prev) => 0.9 * prev + 0.1 * loss.next_token,
        };
        let delta = loss_ema.map(|prev| loss.next_token - prev).unwrap_or(0.0);
        loss_ema = Some(ema);
        if (ema - ln_v).abs() > 0.2 {
            still_random = false;
        }
        let millis = elapsed.as_secs_f64() * 1_000.0;
        let tokens = batch.tokens().len();
        let tps = if elapsed.as_secs_f64() > 0.0 {
            tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let row = serde_json::json!({
            "step": step,
            "tokens": tokens,
            "batch_tokens": tokens,
            "supervised_tokens": loss.next_token_count,
            "step_millis": millis,
            "step_tokens_per_second": tps,
            "tokens_per_second": tps,
            "loss": loss.next_token,
            "loss_ema": ema,
            "loss_delta": delta,
            "learning_rate": learning_rate,
            "architecture": "heron",
            "rosa_bits": model.cfg.rosa_bits,
            "rosa_grad": "stop_grad_bits",
            "binary_flip_count": loss.binary_flip_count,
        });
        write_metrics(&metrics_path, &row)?;
        println!(
            "step {step}/{steps} loss={:.4} ema={:.4} rosa_grad=stop_grad_bits flips={}",
            loss.next_token, ema, loss.binary_flip_count
        );
        if step.is_multiple_of(checkpoint_every) || step == steps {
            fs::write(
                run.join("checkpoint.json"),
                serde_json::to_string(&model.checkpoint())?,
            )?;
        }
        if step == 100 && still_random {
            eprintln!(
                "hint: not learning; check rosa_grad and lr (loss_ema stayed near ln(V)={ln_v:.3})"
            );
        }
    }
    Ok(())
}

fn load_model(checkpoint: &Path) -> Result<(UllisHeron, BpeTokenizer)> {
    let checkpoint_data: ModelCheckpoint = serde_json::from_slice(&fs::read(checkpoint)?)
        .with_context(|| format!("parse checkpoint {}", checkpoint.display()))?;
    let model = UllisHeron::from_checkpoint(checkpoint_data)?;
    let tokenizer_path = checkpoint
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer.json");
    let tokenizer = BpeTokenizer::load(tokenizer_path)?;
    if tokenizer.vocab_size() as usize != model.cfg.vocab_size {
        bail!("checkpoint vocabulary does not match its adjacent tokenizer.json");
    }
    Ok((model, tokenizer))
}

fn chat(checkpoint: PathBuf, _session: PathBuf, _thinking: ThinkingLevel) -> Result<()> {
    let _ = load_model(&checkpoint)?;
    bail!("Heron chat not wired")
}

fn generate(checkpoint: PathBuf, _prompt: String, max_tokens: usize) -> Result<()> {
    if max_tokens == 0 {
        bail!("max-tokens must be positive");
    }
    let _ = load_model(&checkpoint)?;
    bail!("Heron generate not wired")
}

fn smoke() -> Result<()> {
    let cfg = smoke_config(MIN_VOCAB as usize);
    let mut model = UllisHeron::new(cfg.clone())?;
    let tokens: Vec<u32> = (0..cfg.context_len)
        .map(|i| 4 + (i as u32 % 8))
        .collect();
    let ln_v = (cfg.vocab_size as f32).ln();
    println!(
        "heron smoke | architecture {:?} | d {} | layers {} | context {} | vocab {} | rosa_grad {:?}",
        model.cfg.architecture,
        model.cfg.d_model,
        model.cfg.n_layers,
        model.cfg.context_len,
        model.cfg.vocab_size,
        model.cfg.rosa_grad,
    );
    #[cfg(target_os = "macos")]
    let loss = match ullis::metal::MetalRuntime::new() {
        Ok(runtime) => {
            println!("metal runtime ready; running stop_grad_bits train_step");
            model.train_step_metal(&runtime, &tokens, 1, cfg.context_len, 1e-3)?
        }
        Err(error) => {
            println!("metal unavailable ({error}); CPU train_step");
            model.train_step(&tokens, 1, cfg.context_len, 1e-3)?
        }
    };
    #[cfg(not(target_os = "macos"))]
    let loss = model.train_step(&tokens, 1, cfg.context_len, 1e-3)?;
    println!(
        "smoke loss={:.4} ln(V)={:.4} supervised={} flips={} rosa_grad=stop_grad_bits",
        loss.next_token, ln_v, loss.next_token_count, loss.binary_flip_count
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.smoke {
        return smoke();
    }
    match cli.command {
        Some(Command::Train {
            data,
            run,
            config,
            overrides,
            resume,
            steps,
            learning_rate,
            checkpoint_every,
            backend,
            bpe_train_mib,
        }) => train(
            data,
            run,
            config,
            *overrides,
            resume,
            steps,
            learning_rate,
            checkpoint_every,
            backend,
            bpe_train_mib,
        ),
        Some(Command::Tokenize { data, output }) => {
            let records = load_dataset(&data)?;
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(output)?;
            for record in records {
                writeln!(file, "{}", serde_json::to_string(&record)?)?;
            }
            Ok(())
        }
        Some(Command::Inspect { run }) => {
            println!("{}", fs::read_to_string(run.join("config.json"))?);
            Ok(())
        }
        Some(Command::Chat {
            checkpoint,
            session,
            thinking,
        }) => chat(checkpoint, session, thinking),
        Some(Command::Generate {
            checkpoint,
            prompt,
            max_tokens,
        }) => generate(checkpoint, prompt, max_tokens),
        None => {
            println!("Run `ullis --help` for training, dataset, and chat commands.");
            Ok(())
        }
    }
}
