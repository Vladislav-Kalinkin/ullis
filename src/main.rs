use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullis::conversation::{
    ByteSpan, ChatMessage, generation_prefix, pack_document_windows, render_messages,
    supervised_labels, truncate_at_assistant_end,
};
use ullis::decode::{DecodeConfig, apply_openai_penalties, bump_count, select_token};
use ullis::tokenizer::{BpeTokenizer, DEFAULT_VOCAB, MIN_VOCAB, train_bpe};
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
        /// Cap loaded JSONL text in MiB. 0 (default) reads the whole file.
        /// Independent of `--bpe-train-mib`, which only sizes the BPE fit.
        #[arg(long, default_value_t = 0)]
        data_mib: usize,
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
        #[command(flatten)]
        decode: DecodeArgs,
    },
    Generate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[command(flatten)]
        decode: DecodeArgs,
    },
    EvalDigits {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, value_enum)]
        task: DigitTask,
        #[arg(long, default_value_t = 8)]
        max_digits: usize,
    },
}

#[derive(Clone, Debug, Args)]
struct DecodeArgs {
    /// 0 is greedy argmax (Holtzman loops). Default matches Completions-style sampling.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    /// Nucleus mass in (0, 1]. Ignored when temperature is 0.
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    /// OpenAI frequency_penalty in [-2, 2]: subtract count·α from each generated id.
    #[arg(long, default_value_t = 0.5)]
    frequency_penalty: f32,
    /// OpenAI presence_penalty in [-2, 2]: subtract α once a generated id has appeared.
    #[arg(long, default_value_t = 0.0)]
    presence_penalty: f32,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

impl DecodeArgs {
    fn config(&self) -> Result<DecodeConfig> {
        let cfg = DecodeConfig {
            temperature: self.temperature,
            top_p: self.top_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            min_new_tokens: 1,
            seed: self.seed,
        };
        cfg.validate()?;
        Ok(cfg)
    }
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DigitTask {
    Reverse,
    Plusminus,
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

fn log_status(message: impl Display) {
    eprintln!("ullis: {message}");
    let _ = io::stderr().flush();
}

struct TrainingDocument {
    text: String,
    spans: Vec<ByteSpan>,
}

fn training_document(record: &DatasetRecord) -> TrainingDocument {
    let messages: Vec<ChatMessage<'_>> = record
        .messages
        .iter()
        .map(|message| ChatMessage {
            role: &message.role,
            content: &message.content,
            thinking: message.thinking.as_deref(),
        })
        .collect();
    let (text, spans) = render_messages(&messages);
    TrainingDocument { text, spans }
}

/// Stream JSONL into training strings, stopping once `max_bytes` of text is in hand.
fn load_training_texts(
    path: &Path,
    max_bytes: usize,
) -> Result<(Vec<TrainingDocument>, usize, usize)> {
    let file = File::open(path).with_context(|| format!("open dataset {}", path.display()))?;
    let mut texts = Vec::new();
    let mut records = 0_usize;
    let mut text_bytes = 0_usize;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("read dataset line {}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: DatasetRecord = serde_json::from_str(&line)
            .with_context(|| format!("parse dataset line {}", index + 1))?;
        validate_record(&record).with_context(|| format!("validate dataset line {}", index + 1))?;
        let document = training_document(&record);
        text_bytes = text_bytes.saturating_add(document.text.len());
        texts.push(document);
        records += 1;
        if text_bytes >= max_bytes {
            break;
        }
    }
    if texts.is_empty() {
        bail!("dataset {} has no training records", path.display());
    }
    Ok((texts, records, text_bytes))
}

fn cap_texts_bytes(texts: &[TrainingDocument], max_bytes: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut used = 0_usize;
    for text in texts {
        if max_bytes != usize::MAX && used >= max_bytes && !out.is_empty() {
            break;
        }
        used = used.saturating_add(text.text.len());
        out.push(text.text.clone());
    }
    out
}

fn encode_training_stream(
    documents: &[TrainingDocument],
    tokenizer: &mut BpeTokenizer,
    context_len: usize,
    needed: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    if needed == 0 {
        bail!("encoded corpus is empty");
    }
    let mut packed = Vec::new();
    for document in documents {
        let ids = tokenizer.encode(&document.text, true, true);
        let labels = supervised_labels(tokenizer, &ids, &document.text, &document.spans);
        packed.push((ids, labels));
    }
    pack_document_windows(&packed, context_len, tokenizer.pad_id, needed)
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

fn write_metrics(path: &Path, row: &serde_json::Value) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
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
    bpe_train_mib: usize,
    data_mib: usize,
) -> Result<()> {
    if steps == 0 || checkpoint_every == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
        bail!("steps, checkpoint-every, and learning-rate must be positive");
    }
    let started_all = Instant::now();
    log_status(format!(
        "train start steps={steps} data={} run={} backend={backend:?}",
        data.display(),
        run.display()
    ));

    let cfg_for_budget = if resume.is_some() {
        None
    } else {
        Some(load_config(config_path.clone(), &overrides)?)
    };
    if let Some(cfg) = &cfg_for_budget {
        cfg.validate()?;
        if matches!(cfg.architecture, Architecture::RosaRwkv7) {
            bail!("rosa_rwkv7 train is not wired");
        }
        cfg.optimizer.require_train_step()?;
    }
    let bpe_bytes = bpe_train_mib.saturating_mul(1024 * 1024);
    let text_budget = if data_mib == 0 {
        usize::MAX
    } else {
        data_mib.saturating_mul(1024 * 1024)
    };

    let load_started = Instant::now();
    log_status(format!(
        "loading dataset {} (data-mib={}, text budget {} bytes, bpe-train-mib={bpe_train_mib})",
        data.display(),
        if data_mib == 0 {
            "all".into()
        } else {
            data_mib.to_string()
        },
        if text_budget == usize::MAX {
            "unlimited".into()
        } else {
            text_budget.to_string()
        }
    ));
    let (texts, records, text_bytes) = load_training_texts(&data, text_budget)?;
    log_status(format!(
        "loaded {records} records, {text_bytes} bytes of training text in {:.1}s",
        load_started.elapsed().as_secs_f64()
    ));

    let (mut model, mut tokenizer) = if let Some(path) = resume {
        log_status(format!("resuming {}", path.display()));
        let (model, tokenizer) = load_model(&path)?;
        reject_resume_overrides(&model.cfg, &overrides)?;
        if let Some(path) = config_path {
            let file_cfg = load_config(Some(path), &TrainOverrides::default())?;
            reject_resume_file_config(&model.cfg, &file_cfg)?;
        }
        (model, tokenizer)
    } else {
        let mut cfg = cfg_for_budget
            .ok_or_else(|| anyhow::anyhow!("fresh train is missing a resolved config"))?;
        fs::create_dir_all(&run)?;
        let tokenizer_path = run.join("tokenizer.json");
        let tokenizer = if tokenizer_path.is_file() {
            log_status(format!(
                "reusing {} (delete it to retrain BPE)",
                tokenizer_path.display()
            ));
            BpeTokenizer::load(&tokenizer_path)?
        } else {
            let bpe_started = Instant::now();
            let bpe_texts = cap_texts_bytes(
                &texts,
                if bpe_bytes == 0 {
                    usize::MAX
                } else {
                    bpe_bytes
                },
            );
            let bpe_text_bytes: usize = bpe_texts.iter().map(String::len).sum();
            log_status(format!(
                "training BPE vocab_ceiling={} on {bpe_text_bytes} bytes",
                cfg.vocab_size
            ));
            let tokenizer = train_bpe(&bpe_texts, cfg.vocab_size as u32, cfg.seed)?;
            log_status(format!(
                "BPE finished vocab={} merges={} in {:.1}s",
                tokenizer.vocab_size(),
                tokenizer.merges.len(),
                bpe_started.elapsed().as_secs_f64()
            ));
            tokenizer
        };
        cfg = cfg.with_tokenizer(&tokenizer)?;
        let model_started = Instant::now();
        log_status(format!(
            "initializing heron d={} layers={} vocab={} context={}",
            cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.context_len
        ));
        let model = UllisHeron::new(cfg)?;
        log_status(format!(
            "model ready in {:.1}s",
            model_started.elapsed().as_secs_f64()
        ));
        (model, tokenizer)
    };
    if matches!(model.cfg.architecture, Architecture::RosaRwkv7) {
        bail!("rosa_rwkv7 train is not wired");
    }
    model.cfg.optimizer.require_train_step()?;
    fs::create_dir_all(&run)?;
    tokenizer.save(run.join("tokenizer.json"))?;
    fs::write(
        run.join("config.json"),
        serde_json::to_string_pretty(&model.cfg)?,
    )?;

    let needed = steps
        .saturating_mul(model.cfg.batch_size)
        .saturating_mul(model.cfg.context_len);
    let encode_started = Instant::now();
    log_status(format!("encoding {needed} training tokens"));
    let pad_id = tokenizer.pad_id;
    let (stream, labels) =
        encode_training_stream(&texts, &mut tokenizer, model.cfg.context_len, needed)?;
    log_status(format!(
        "encoded {} tokens (document windows, assistant+EOS CE) in {:.1}s",
        stream.len(),
        encode_started.elapsed().as_secs_f64()
    ));
    let prior = model.install_head_unigram_prior(&labels, pad_id)?;
    if prior.applied {
        log_status(format!(
            "head unigram prior n={} bias_rms={:.3} scale_rms={:.4} kaiming_shrink={}",
            prior.n_targets, prior.bias_rms, prior.scale_rms, prior.shrunk_kaiming_scale
        ));
    } else {
        log_status(format!(
            "head unigram prior skipped (bias already fitted, rms={:.3})",
            prior.bias_rms
        ));
    }
    let batcher = CausalBatcher::from_config_with_labels(
        &stream,
        &labels,
        &model.cfg,
        model.cfg.context_len,
    )?;

    #[cfg(target_os = "macos")]
    let metal = match backend {
        Backend::Metal => {
            let metal_started = Instant::now();
            log_status("compiling Metal shaders");
            let runtime = ullis::metal::MetalRuntime::new()?;
            log_status(format!(
                "Metal ready in {:.1}s",
                metal_started.elapsed().as_secs_f64()
            ));
            let rows = model.cfg.batch_size.saturating_mul(model.cfg.context_len);
            let rosa_readback = rows
                .saturating_mul(model.cfg.d_model)
                .saturating_mul(model.cfg.n_layers)
                .saturating_mul(size_of::<f32>().saturating_mul(2).saturating_add(1));
            log_status(format!(
                "Metal train: LN/QKV/CMix/head on GPU; ROSA SAM on CPU (~{rosa_readback} bytes idx+y+out/step)"
            ));
            Some(runtime)
        }
        Backend::Cpu => {
            log_status("CPU backend; no Metal kernels");
            None
        }
    };
    #[cfg(not(target_os = "macos"))]
    if matches!(backend, Backend::Metal) {
        bail!("Metal backend requires macOS on Apple Silicon");
    }

    log_status(format!(
        "starting loop after {:.1}s of setup",
        started_all.elapsed().as_secs_f64()
    ));
    log_status(format!(
        "optimizer={:?} lr={learning_rate} rosa_grad={:?} (QKV frozen; window-mean CE on FP16, token-sum STE on BinaryConnect; |w0|={})",
        model.cfg.optimizer,
        model.cfg.rosa_grad,
        ullis::model::BINARYCONNECT_INIT_ABS
    ));
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
            model.train_step_metal_with_labels(
                runtime,
                batch.tokens(),
                batch.labels(),
                pad_id,
                batch.batch_size(),
                batch.time(),
                learning_rate,
            )?
        } else {
            model.train_step_with_labels(
                batch.tokens(),
                batch.labels(),
                pad_id,
                batch.batch_size(),
                batch.time(),
                learning_rate,
            )?
        };
        #[cfg(not(target_os = "macos"))]
        let loss = model.train_step_with_labels(
            batch.tokens(),
            batch.labels(),
            pad_id,
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
        let phases = model.last_step_profile().map(|profile| {
            profile
                .phases_ms
                .iter()
                .map(|(name, ms)| (name.clone(), serde_json::json!(ms)))
                .collect::<serde_json::Map<_, _>>()
        });
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
            "loss_p10": loss.loss_p10,
            "loss_p50": loss.loss_p50,
            "loss_p90": loss.loss_p90,
            "unigram_ce": loss.unigram_ce,
            "unique_targets": loss.unique_targets,
            "learning_rate": learning_rate,
            "architecture": "heron",
            "rosa_bits": model.cfg.rosa_bits,
            "rosa_grad": "stop_grad_bits",
            "binary_flip_count": loss.binary_flip_count,
            "flips_head": loss.flips_head,
            "flips_cmix": loss.flips_cmix,
            "flips_rosa_o": loss.flips_rosa_o,
            "embed_grad_rms": loss.embed_grad_rms,
            "head_scale_grad_rms": loss.head_scale_grad_rms,
            "head_scale_rms": loss.head_scale_rms,
            "residual_abs_mean": loss.residual_abs_mean,
            "cmix_value_rms": loss.cmix_value_rms,
            "head_latent_abs_mean": loss.head_latent_abs_mean,
            "head_latent_step_abs": loss.head_latent_step_abs,
            "head_bias_rms": loss.head_bias_rms,
            "phases_ms": phases,
        });
        write_metrics(&metrics_path, &row)?;
        println!(
            "step {step}/{steps} loss={:.4} ema={:.4} p10={:.3} p50={:.3} p90={:.3} unigram={:.3} unique={} n={} flips={}/{}/{} (head/cmix/o) embed_grms={:.2e} scale_grms={:.2e} cmix_vrms={:.3} |w|={:.3} dw={:.2e} bias_rms={:.3} resid={:.2e} {millis:.0}ms {tps:.0} tok/s",
            loss.next_token,
            ema,
            loss.loss_p10,
            loss.loss_p50,
            loss.loss_p90,
            loss.unigram_ce,
            loss.unique_targets,
            loss.next_token_count,
            loss.flips_head,
            loss.flips_cmix,
            loss.flips_rosa_o,
            loss.embed_grad_rms,
            loss.head_scale_grad_rms,
            loss.cmix_value_rms,
            loss.head_latent_abs_mean,
            loss.head_latent_step_abs,
            loss.head_bias_rms,
            loss.residual_abs_mean
        );
        if let Some(profile) = model.last_step_profile() {
            log_status(format!("step {step} phases {}", profile.line()));
        }
        let _ = io::stdout().flush();
        if step.is_multiple_of(checkpoint_every) || step == steps {
            // Snapshot only on a checkpoint boundary: bits+scale+bias, never RAM latents.
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

fn load_checkpoint(path: &Path) -> Result<ModelCheckpoint> {
    let bytes = fs::read(path).with_context(|| format!("read checkpoint {}", path.display()))?;
    ModelCheckpoint::from_json_bytes(&bytes)
        .with_context(|| format!("parse checkpoint {}", path.display()))
}

fn load_model(checkpoint: &Path) -> Result<(UllisHeron, BpeTokenizer)> {
    let checkpoint_data = load_checkpoint(checkpoint)?;
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

fn reject_resume_overrides(cfg: &TrainConfig, overrides: &TrainOverrides) -> Result<()> {
    if let Some(architecture) = overrides.architecture {
        let architecture = match architecture {
            ArchitectureArg::Heron => Architecture::Heron,
            ArchitectureArg::RosaRwkv7 => Architecture::RosaRwkv7,
        };
        if architecture != cfg.architecture {
            bail!(
                "--resume cannot change architecture (checkpoint {:?}, override {architecture:?})",
                cfg.architecture
            );
        }
    }
    reject_resume_field("d-model", overrides.d_model, cfg.d_model)?;
    reject_resume_field("layers", overrides.n_layers, cfg.n_layers)?;
    reject_resume_field("context-len", overrides.context_len, cfg.context_len)?;
    reject_resume_field("batch-size", overrides.batch_size, cfg.batch_size)?;
    reject_resume_field("vocab-size", overrides.vocab_size, cfg.vocab_size)?;
    reject_resume_field("seed", overrides.seed, cfg.seed)?;
    if let Some(mib) = overrides.memory_budget_mib {
        let bytes = mib
            .checked_mul(1024 * 1024)
            .ok_or_else(|| anyhow::anyhow!("memory-budget-mib overflows bytes"))?;
        if bytes != cfg.memory_budget_bytes {
            bail!(
                "--resume cannot change memory-budget-mib (checkpoint {}, override {mib})",
                cfg.memory_budget_bytes / (1024 * 1024)
            );
        }
    }
    if let Some(optimizer) = overrides.optimizer {
        let optimizer = match optimizer {
            OptimizerArg::StatelessSgd => OptimizerKind::StatelessSgd,
            OptimizerArg::LionFp16 => OptimizerKind::LionFp16,
        };
        if optimizer != cfg.optimizer {
            bail!(
                "--resume cannot change optimizer (checkpoint {:?}, override {optimizer:?})",
                cfg.optimizer
            );
        }
    }
    Ok(())
}

fn reject_resume_field<T: Copy + PartialEq + Display>(
    name: &str,
    requested: Option<T>,
    current: T,
) -> Result<()> {
    if let Some(value) = requested
        && value != current
    {
        bail!("--resume cannot change {name} (checkpoint {current}, override {value})");
    }
    Ok(())
}

fn reject_resume_file_config(checkpoint: &TrainConfig, file: &TrainConfig) -> Result<()> {
    if checkpoint.architecture != file.architecture
        || checkpoint.d_model != file.d_model
        || checkpoint.n_layers != file.n_layers
        || checkpoint.vocab_size != file.vocab_size
        || checkpoint.context_len != file.context_len
        || checkpoint.batch_size != file.batch_size
        || checkpoint.resolved_dim_ffn() != file.resolved_dim_ffn()
        || checkpoint.rosa_bits != file.rosa_bits
    {
        bail!("--config does not match the resumed checkpoint; omit --config when using --resume");
    }
    Ok(())
}

fn inspect_run(run: &Path) -> Result<()> {
    let checkpoint = load_checkpoint(&run.join("checkpoint.json"))?;
    let report = checkpoint.inspect()?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn generate_completion(
    model: &UllisHeron,
    tokenizer: &mut BpeTokenizer,
    prompt: &str,
    max_tokens: usize,
    chat_template: bool,
    decode: DecodeConfig,
) -> Result<String> {
    if max_tokens == 0 {
        bail!("max-tokens must be positive");
    }
    decode.validate()?;
    // Train CE is next-token on assistant spans of `render_messages`. A raw
    // prompt is OOD unless it is wrapped with the same open assistant turn.
    let prompt = if chat_template {
        generation_prefix(prompt)
    } else {
        prompt.to_string()
    };
    let mut ids = tokenizer.encode(&prompt, true, false);
    if ids.is_empty() {
        ids.push(tokenizer.bos_id);
    }
    if ids.len() > model.cfg.context_len {
        bail!(
            "prompt is longer than context_len ({})",
            model.cfg.context_len
        );
    }
    let mut state = model.generate_state()?;
    let mut logits = Vec::new();
    for &id in &ids {
        logits = model.generate_step(&mut state, id)?;
    }
    let mut produced = Vec::with_capacity(max_tokens);
    let mut counts = vec![0_u32; logits.len()];
    let mut rng = decode.seed | 1;
    for _ in 0..max_tokens {
        apply_openai_penalties(
            &mut logits,
            &counts,
            decode.presence_penalty,
            decode.frequency_penalty,
        );
        let suppress_eos = produced.len() < decode.min_new_tokens;
        let next = select_token(
            &logits,
            decode,
            &mut rng,
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            suppress_eos,
        );
        if next == tokenizer.eos_id {
            break;
        }
        if state.time() >= model.cfg.context_len {
            break;
        }
        produced.push(next);
        bump_count(&mut counts, next);
        if chat_template {
            let decoded = tokenizer.decode(&produced);
            if let Some(truncated) = truncate_at_assistant_end(&decoded) {
                return Ok(truncated.to_string());
            }
        }
        logits = model.generate_step(&mut state, next)?;
        if counts.len() != logits.len() {
            counts.resize(logits.len(), 0);
        }
    }
    Ok(tokenizer.decode(&produced))
}

fn chat(
    checkpoint: PathBuf,
    session: PathBuf,
    mut thinking: ThinkingLevel,
    decode: DecodeConfig,
) -> Result<()> {
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
        let reply = generate_completion(&model, &mut tokenizer, &line, 64, true, decode)?;
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

fn generate(
    checkpoint: PathBuf,
    prompt: String,
    max_tokens: usize,
    decode: DecodeConfig,
) -> Result<()> {
    if max_tokens == 0 {
        bail!("max-tokens must be positive");
    }
    let (model, mut tokenizer) = load_model(&checkpoint)?;
    println!(
        "{}",
        generate_completion(&model, &mut tokenizer, &prompt, max_tokens, true, decode)?
    );
    Ok(())
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn get_randint(digits: usize, rng: &mut u64) -> u64 {
    let lo = if digits <= 1 {
        0
    } else {
        10_u64.pow((digits - 1) as u32)
    };
    let hi = 10_u64.pow(digits as u32) - 1;
    lo + splitmix(rng) % (hi - lo + 1)
}

fn wkv_pad_len(script_t: usize) -> usize {
    script_t.div_ceil(16) * 16
}

fn eval_digits(checkpoint: PathBuf, task: DigitTask, max_digits: usize) -> Result<()> {
    if max_digits == 0 {
        bail!("max-digits must be positive");
    }
    let model = UllisHeron::from_checkpoint(load_checkpoint(&checkpoint)?)?;
    if !matches!(model.cfg.architecture, Architecture::RosaRwkv7) {
        bail!("eval-digits requires a rosa_rwkv7 checkpoint");
    }
    let (script_t, pad, alphabet, vocab) = match task {
        DigitTask::Reverse => (129_usize, b'#' as u32, "0123456789,#", 12_usize),
        DigitTask::Plusminus => (129_usize, b'=' as u32, "0123456789+-=", 13_usize),
    };
    if model.cfg.vocab_size < vocab {
        bail!("checkpoint vocab is smaller than the {task:?} alphabet");
    }
    let t_wkv = wkv_pad_len(script_t);
    if model.cfg.context_len < t_wkv {
        bail!(
            "checkpoint context_len {} cannot hold padded T_wkv {t_wkv}",
            model.cfg.context_len
        );
    }
    let mut rng = model.cfg.seed | 1;
    let mut n_good = 0_usize;
    let mut n_all = 0_usize;
    for digits in 1..=max_digits {
        let sequences: Vec<Vec<u32>> = match task {
            DigitTask::Reverse => (0..10)
                .map(|_| {
                    let raw = get_randint(digits, &mut rng).to_string();
                    let body = format!("{raw},{}", raw.chars().rev().collect::<String>());
                    encode_digit_line(&body, alphabet, pad, t_wkv)
                })
                .collect(),
            DigitTask::Plusminus => {
                let mut out = Vec::new();
                for ii in 1..2 * digits {
                    let (aa, bb) = if ii <= digits {
                        (ii, digits)
                    } else {
                        (digits, 2 * digits - ii)
                    };
                    let a = get_randint(aa, &mut rng) as i64;
                    let b = get_randint(bb, &mut rng) as i64;
                    let plus = splitmix(&mut rng) & 1 == 0;
                    let result = if plus { a + b } else { a - b };
                    let op = if plus { '+' } else { '-' };
                    let body = format!("{a}{op}{b}={result}");
                    out.push(encode_digit_line(&body, alphabet, pad, t_wkv));
                }
                out
            }
        };
        for src in sequences {
            let input = &src[..t_wkv];
            let logits = model.logits(input, 1, t_wkv)?;
            let vocab_size = model.cfg.vocab_size;
            let predicted: Vec<u32> = (0..t_wkv)
                .map(|t| {
                    let row = &logits[t * vocab_size..(t + 1) * vocab_size];
                    row.iter()
                        .take(vocab)
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(id, _)| id as u32)
                        .unwrap_or(0)
                })
                .collect();
            let xx: String = input
                .iter()
                .map(|&id| {
                    alphabet
                        .as_bytes()
                        .get(id as usize)
                        .copied()
                        .unwrap_or(b'?') as char
                })
                .collect();
            let (p1, p2) = match task {
                DigitTask::Reverse => {
                    let p1 = xx.find(',').unwrap_or(0);
                    let p2 = xx.find('#').unwrap_or_else(|| xx.len().saturating_sub(1));
                    (p1, p2)
                }
                DigitTask::Plusminus => {
                    let p1 = xx.find('=').unwrap_or(0);
                    let rest = &xx[p1 + 1..];
                    let p2 = p1 + 1 + rest.find('=').unwrap_or(0);
                    (p1, p2)
                }
            };
            if p2 <= p1 {
                continue;
            }
            n_all += p2 - p1;
            for offset in 0..(p2 - p1) {
                if predicted[p1 + offset] == src[p1 + 1 + offset] {
                    n_good += 1;
                }
            }
        }
        println!("digits {digits} running {n_good}/{n_all}");
    }
    println!("eval-digits {task:?} correct {n_good} / {n_all} (unpadded span, T_wkv={t_wkv})");
    Ok(())
}

fn encode_digit_line(body: &str, alphabet: &str, pad: u32, t_wkv: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = body
        .chars()
        .map(|ch| alphabet.find(ch).map(|i| i as u32).unwrap_or(pad))
        .collect();
    ids.resize(t_wkv, pad);
    ids
}

fn smoke() -> Result<()> {
    let cfg = smoke_config(MIN_VOCAB as usize);
    let mut model = UllisHeron::new(cfg.clone())?;
    let tokens: Vec<u32> = (0..cfg.context_len).map(|i| 4 + (i as u32 % 8)).collect();
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
            data_mib,
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
            data_mib,
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
        Some(Command::Inspect { run }) => inspect_run(&run),
        Some(Command::Chat {
            checkpoint,
            session,
            thinking,
            decode,
        }) => chat(checkpoint, session, thinking, decode.config()?),
        Some(Command::Generate {
            checkpoint,
            prompt,
            max_tokens,
            decode,
        }) => generate(checkpoint, prompt, max_tokens, decode.config()?),
        Some(Command::EvalDigits {
            checkpoint,
            task,
            max_digits,
        }) => eval_digits(checkpoint, task, max_digits),
        None => {
            println!("Run `ullis --help` for training, dataset, and chat commands.");
            Ok(())
        }
    }
}
