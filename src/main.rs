use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullis::{
    ModelCheckpoint, MtpBatcher, OptimizerKind, TrainConfig, UllisHyena,
    tokenizer::{BpeTokenizer, DEFAULT_VOCAB, train_bpe},
};

#[derive(Debug, Parser)]
#[command(name = "ullis", version, about = "Dense ternary Hyena training tools")]
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
        /// Continue from a previously saved checkpoint instead of initializing.
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f32,
        /// Write a checkpoint after this many steps (and always at the end).
        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,
        /// CPU is a diagnostic fallback. Metal is the normal Ullis trainer.
        #[arg(long, value_enum, default_value_t = Backend::Metal)]
        backend: Backend,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ToolCall {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct TrainMetric {
    step: usize,
    tokens: usize,
    tokens_per_second: f64,
    loss: f32,
    loss_ema: f32,
    mtp_next: f32,
    mtp_second: f32,
    learning_rate: f32,
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

fn training_text(record: &DatasetRecord) -> String {
    record
        .messages
        .iter()
        .map(|message| {
            let thinking = message
                .thinking
                .as_deref()
                .map(|value| format!("<thinking>{value}</thinking>"))
                .unwrap_or_default();
            format!("<{}>{thinking}{}", message.role, message.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn small_config(vocab_size: usize) -> TrainConfig {
    TrainConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size,
        context_len: 32,
        batch_size: 1,
        hyena_kernel_len: 32,
        hyena_chunk_len: 32,
        ..Default::default()
    }
}

fn train(
    data: PathBuf,
    run: PathBuf,
    config_path: Option<PathBuf>,
    resume: Option<PathBuf>,
    steps: usize,
    learning_rate: f32,
    checkpoint_every: usize,
    backend: Backend,
) -> Result<()> {
    if steps == 0 || checkpoint_every == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
        bail!("steps, checkpoint-every, and learning-rate must be positive");
    }
    let records = load_dataset(&data)?;
    if records.is_empty() {
        bail!("dataset has no records");
    }
    let mut cfg = match config_path {
        Some(path) => serde_json::from_str(&fs::read_to_string(&path)?)
            .with_context(|| format!("parse config {}", path.display()))?,
        None => small_config(DEFAULT_VOCAB as usize),
    };
    let training_texts = records.iter().map(training_text).collect::<Vec<_>>();
    let mut tokenizer = train_bpe(&training_texts, cfg.vocab_size as u32, cfg.seed)?;
    let mut tokens = training_texts
        .iter()
        .flat_map(|text| tokenizer.encode(text, true, true))
        .collect::<Vec<_>>();
    cfg.vocab_size = tokenizer.vocab_size() as usize;
    // The CLI's actual updater is stateless SGD. Keep the persisted memory
    // ledger truthful even when an imported configuration chose Lion.
    cfg.optimizer = OptimizerKind::StatelessSgd;
    cfg.validate()?;
    let time = cfg.context_len.min(32);
    if tokens.len() < time {
        tokens.resize(time, tokenizer.eos_id);
    }
    fs::create_dir_all(&run)?;
    fs::write(run.join("config.json"), serde_json::to_vec_pretty(&cfg)?)?;
    tokenizer.save(run.join("tokenizer.json"))?;
    let mut metrics = OpenOptions::new()
        .create(true)
        .append(true)
        .open(run.join("metrics.jsonl"))?;
    let mut model = match resume {
        Some(path) => {
            let checkpoint: ModelCheckpoint = serde_json::from_slice(&fs::read(&path)?)
                .with_context(|| format!("parse checkpoint {}", path.display()))?;
            let mut model = UllisHyena::from_checkpoint(checkpoint)?;
            if model.cfg.vocab_size != tokenizer.vocab_size() as usize {
                bail!("checkpoint vocabulary does not match tokenizer");
            }
            model.cfg.optimizer = OptimizerKind::StatelessSgd;
            cfg = model.cfg.clone();
            model
        }
        None => UllisHyena::new(cfg.clone())?,
    };
    let mut batcher = MtpBatcher::from_config(&tokens, &cfg, time)?;
    let started = Instant::now();
    let mut loss_ema = None;
    #[cfg(target_os = "macos")]
    let metal_runtime = matches!(backend, Backend::Metal)
        .then(ullis::metal::MetalRuntime::new)
        .transpose()?;
    #[cfg(target_os = "macos")]
    let metal_state = metal_runtime
        .as_ref()
        .map(|runtime| model.new_metal_resident_training_state(runtime))
        .transpose()?;
    #[cfg(not(target_os = "macos"))]
    if matches!(backend, Backend::Metal) {
        bail!(
            "Metal is Ullis's default trainer and requires macOS Apple Silicon; use --backend cpu only for the reference fallback"
        );
    }
    for step in 1..=steps {
        let batch = batcher.next().unwrap_or_else(|| {
            batcher = MtpBatcher::from_config(&tokens, &cfg, time).expect("validated batcher");
            batcher
                .next()
                .expect("padded token corpus yields one batch")
        });
        let loss = match backend {
            Backend::Cpu => model.train_step_stateless_sgd(
                batch.tokens(),
                batch.batch_size(),
                batch.time(),
                learning_rate,
            )?,
            #[cfg(target_os = "macos")]
            Backend::Metal => {
                let runtime = metal_runtime
                    .as_ref()
                    .expect("Metal runtime is initialized");
                let state = metal_state.as_ref().expect("Metal state is initialized");
                model.train_step_metal_resident_stateless_sgd(
                    runtime,
                    state,
                    batch.tokens(),
                    batch.batch_size(),
                    batch.time(),
                    learning_rate,
                )?
            }
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => unreachable!("rejected before training"),
        };
        let processed = step * batch.tokens().len();
        let loss_ema_value = match loss_ema {
            Some(previous) => 0.95 * previous + 0.05 * loss.mean(),
            None => loss.mean(),
        };
        loss_ema = Some(loss_ema_value);
        let metric = TrainMetric {
            step,
            tokens: processed,
            tokens_per_second: processed as f64
                / started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE),
            loss: loss.mean(),
            loss_ema: loss_ema_value,
            mtp_next: loss.next_token,
            mtp_second: loss.second_token,
            learning_rate,
        };
        writeln!(metrics, "{}", serde_json::to_string(&metric)?)?;
        println!(
            "step {step} | tok {processed} | {:.1} tok/s | loss {:.4} | ema {:.4} | mtp+1 {:.4} | mtp+2 {:.4} | lr {learning_rate:.2e}",
            metric.tokens_per_second,
            metric.loss,
            metric.loss_ema,
            metric.mtp_next,
            metric.mtp_second
        );
        if step % checkpoint_every == 0 || step == steps {
            #[cfg(target_os = "macos")]
            let checkpoint = if let (Some(runtime), Some(state)) =
                (metal_runtime.as_ref(), metal_state.as_ref())
            {
                model.checkpoint_metal_resident(runtime, state)?
            } else {
                model.checkpoint()
            };
            #[cfg(not(target_os = "macos"))]
            let checkpoint = model.checkpoint();
            fs::write(
                run.join("checkpoint.json"),
                serde_json::to_vec(&checkpoint)?,
            )?;
        }
    }
    Ok(())
}

fn load_model(checkpoint: &Path) -> Result<(UllisHyena, BpeTokenizer)> {
    let checkpoint_data: ModelCheckpoint = serde_json::from_slice(&fs::read(checkpoint)?)
        .with_context(|| format!("parse checkpoint {}", checkpoint.display()))?;
    let model = UllisHyena::from_checkpoint(checkpoint_data)?;
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

fn greedy_generate(
    model: &UllisHyena,
    tokenizer: &mut BpeTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> Result<String> {
    if max_tokens == 0 {
        bail!("max-tokens must be positive");
    }
    let mut ids = tokenizer.encode(prompt, true, false);
    let mut produced = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let start = ids.len().saturating_sub(model.cfg.context_len);
        let mut context = ids[start..].to_vec();
        while context.len() < 3 {
            context.insert(0, tokenizer.bos_id);
        }
        let time = context.len();
        let (logits, _) = model.mtp_logits(&context, 1, time)?;
        let row = &logits[(time - 1) * model.cfg.vocab_size..time * model.cfg.vocab_size];
        let next = row
            .iter()
            .enumerate()
            .filter(|(id, _)| *id as u32 != tokenizer.pad_id && *id as u32 != tokenizer.bos_id)
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id as u32)
            .expect("non-empty vocabulary");
        if next == tokenizer.eos_id {
            break;
        }
        ids.push(next);
        produced.push(next);
    }
    Ok(tokenizer.decode(&produced))
}

fn chat(checkpoint: PathBuf, session: PathBuf, mut thinking: ThinkingLevel) -> Result<()> {
    let (model, mut tokenizer) = load_model(&checkpoint)?;
    if let Some(parent) = session.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut history = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&session)?;
    println!("Ullis chat. /think low|medium|high|xhigh|off, /save, /quit");
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line == "/quit" {
            break;
        }
        if line == "/save" {
            history.flush()?;
            println!("saved {}", session.display());
            continue;
        }
        if let Some(level) = line.strip_prefix("/think ") {
            thinking = match level {
                "low" => ThinkingLevel::Low,
                "medium" => ThinkingLevel::Medium,
                "high" => ThinkingLevel::High,
                "xhigh" => ThinkingLevel::Xhigh,
                "off" => ThinkingLevel::Off,
                _ => {
                    println!("unknown thinking level");
                    continue;
                }
            };
            println!("thinking: {thinking:?}");
            continue;
        }
        let user = DatasetMessage {
            role: "user".into(),
            content: line.clone(),
            thinking: None,
            tool_calls: None,
            tool_call_id: None,
        };
        writeln!(history, "{}", serde_json::to_string(&user)?)?;
        let prompt = format!("<user>{line}\n<assistant><thinking level=\"{thinking:?}\">");
        let reply = greedy_generate(&model, &mut tokenizer, &prompt, 64)?;
        let assistant = DatasetMessage {
            role: "assistant".into(),
            content: reply.clone(),
            thinking: Some(format!("requested:{thinking:?}")),
            tool_calls: None,
            tool_call_id: None,
        };
        writeln!(history, "{}", serde_json::to_string(&assistant)?)?;
        println!("{reply}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.smoke {
        let model = UllisHyena::new(small_config(512))?;
        println!(
            "hyena smoke: streamed MTP loss {:.4}",
            model.streamed_mtp_loss(&[4, 5, 6, 7], 1, 4)?.mean()
        );
        return Ok(());
    }
    match cli.command {
        Some(Command::Train {
            data,
            run,
            config,
            resume,
            steps,
            learning_rate,
            checkpoint_every,
            backend,
        }) => train(
            data,
            run,
            config,
            resume,
            steps,
            learning_rate,
            checkpoint_every,
            backend,
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
        }) => {
            let (model, mut tokenizer) = load_model(&checkpoint)?;
            println!(
                "{}",
                greedy_generate(&model, &mut tokenizer, &prompt, max_tokens)?
            );
            Ok(())
        }
        None => {
            println!("Run `ullis --help` for training, dataset, and chat commands.");
            Ok(())
        }
    }
}
