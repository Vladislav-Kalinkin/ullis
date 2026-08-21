use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use crate::checkpoint;
use crate::config::{next_grid_size, TrainConfig};
use crate::data::{jsonl_corpus_texts, warn_corpus_homogeneity, JsonlStream};
use crate::device::{
    device_name, setup_device, setup_device_with, synchronize, Backend, DeviceFlags,
};
use crate::model::UllisKan;
use crate::optim::SgdMomentum;
use crate::telemetry::{
    cache_metal_hello_mb, metal_hello_mb, process_memory_mb, Throughput, TrainFootprint,
};
use crate::tokenizer::{load_or_train, BpeTokenizer};

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// JSONL corpus (`{"system","user","thinking","output"}` per line). Required; never synthesized.
    #[arg(long, default_value = "data/thinking-train.jsonl")]
    pub data: PathBuf,
    /// Steps per epoch (each of the 4 phases).
    #[arg(long, default_value_t = 200)]
    pub steps: usize,
    /// Checkpoint directory (`packed.bin` is written here).
    #[arg(long, default_value = "checkpoints")]
    pub out: PathBuf,
    #[arg(long, default_value_t = 32)]
    pub d_model: usize,
    #[arg(long, default_value_t = 3)]
    pub layers: usize,
    #[arg(long, default_value_t = 4)]
    pub basis: usize,
    #[arg(long, default_value_t = 8)]
    pub grid_mid: usize,
    #[arg(long, default_value_t = 12)]
    pub grid_final: usize,
    #[arg(long, default_value_t = 96)]
    pub seq_len: usize,
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,
    /// Vocabulary capacity. Hard minimum 8192; scales to 131072+.
    #[arg(long = "vocab-size", visible_alias = "vocab", default_value_t = 8192)]
    pub vocab_size: u32,
    /// Continuous `SovereignFlashBuffer` token-ring cap.
    #[arg(long = "context-len", default_value_t = 32_768)]
    pub context_len: usize,
    /// Kept for old scripts; fused checkpointing is always on.
    #[arg(long = "fused-grad-ckpt", default_value_t = true, action = clap::ArgAction::Set, hide = true)]
    pub fused_grad_ckpt: bool,
    #[arg(long, default_value = "shift")]
    pub mixer: String,
    #[arg(long, default_value_t = 3)]
    pub epochs_warmup: usize,
    #[arg(long, default_value_t = 2)]
    pub epochs_sparsify: usize,
    #[arg(long, default_value_t = 4)]
    pub epochs_qat: usize,
    #[arg(long, default_value_t = 2)]
    pub epochs_harden: usize,
    #[arg(long, default_value_t = 3e-3)]
    pub lr: f64,
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
    #[arg(long, default_value_t = 20)]
    pub log_every: usize,
    #[arg(long, default_value = "")]
    pub tokenizer: String,
    /// Disable Mixture-of-Bumps (recover the v2 all-shared edge function).
    #[arg(long, default_value_t = false)]
    pub no_moe: bool,
    /// Vocab-softmax Shannon entropy penalty (language-agnostic).
    #[arg(long, default_value_t = 0.03)]
    pub entropy_coef: f64,
    /// MoB router Shannon entropy penalty.
    #[arg(long, default_value_t = 0.05)]
    pub router_entropy_coef: f64,
    /// Insert one non-uniform knot every N steps (phases 1–2). 0 disables.
    #[arg(long, default_value_t = 50)]
    pub knot_every: usize,
    #[arg(long)]
    pub cpu: bool,
    /// KAN weight storage master. Compute stays FP32. `fp32` (default) or `fp16`.
    #[arg(long, default_value = "fp32")]
    pub master: String,
    /// Momentum storage. `fp32` (default) or `q8`.
    #[arg(long, default_value = "fp32")]
    pub mom: String,
    /// MoE top-k. Default `0` = dense (bit-identical). `1|2` = per-token top-k.
    #[arg(long = "moe-topk", default_value_t = 0)]
    pub moe_topk: u32,
    /// Switch load-balance coefficient (only when `--moe-topk` > 0).
    #[arg(long = "moe-aux", default_value_t = 0.01)]
    pub moe_aux: f64,
    /// `kan` (production) or `memory` (experimental FWHT+scan+slots+experts).
    #[arg(long, default_value = "kan")]
    pub arch: String,
    /// Memory-arch expert count `E`. Ignored for `kan`.
    #[arg(long = "experts", default_value_t = 4)]
    pub experts: usize,
    /// Memory-arch expert inner width `W`.
    #[arg(long = "expert-width", default_value_t = 64)]
    pub expert_width: usize,
    /// Memory-arch slot count `S`.
    #[arg(long = "slots", default_value_t = 32)]
    pub slots: usize,
}

impl TrainArgs {
    pub fn to_config(&self) -> TrainConfig {
        TrainConfig {
            d_model: self.d_model,
            n_layers: self.layers,
            n_basis: self.basis,
            grid_start: self.basis,
            grid_mid: self.grid_mid,
            grid_final: self.grid_final,
            seq_len: self.seq_len,
            context_len: self.context_len.max(self.seq_len),
            batch_size: self.batch_size,
            mixer: self.mixer.clone(),
            vocab_size: self.vocab_size as usize,
            fused_grad_ckpt: true,
            steps_per_epoch: self.steps,
            epochs_warmup: self.epochs_warmup,
            epochs_sparsify: self.epochs_sparsify,
            epochs_qat: self.epochs_qat,
            epochs_harden: self.epochs_harden,
            lr: self.lr,
            ckpt_dir: self.out.to_string_lossy().into_owned(),
            seed: self.seed,
            tokenizer_path: self.tokenizer.clone(),
            log_every: self.log_every,
            data_path: self.data.to_string_lossy().into_owned(),
            moe: !self.no_moe,
            entropy_coef: self.entropy_coef,
            router_entropy_coef: self.router_entropy_coef,
            knot_insert_every: self.knot_every,
            expert_width: self.expert_width,
            n_slots: self.slots,
            mem_experts: self.experts,
            ..TrainConfig::default()
        }
    }
}

const PHASES: [(u8, &str, fn(&TrainConfig) -> usize, fn(&TrainConfig) -> f64); 4] = [
    (1, "warmup", |c| c.epochs_warmup, |c| c.lr),
    (2, "sparsify", |c| c.epochs_sparsify, |c| c.lr),
    (3, "qat", |c| c.epochs_qat, |c| c.lr_qat),
    (4, "harden", |c| c.epochs_harden, |c| c.lr_harden),
];

/// Fused gradient checkpointing (v0.9).
///
/// Let `F_ℓ` be the fused MoB-KAN block (RMSNorm → mixer → RMSNorm → KAN)
/// compiled as `ullis_mob_kan_fused_step` with `ULLIS_FUSED_GRAD_CKPT=1`.
/// The full tape stores interiors
/// `{ n1, h, n2, ff, res_* }` per layer. Checkpointing stores only the
/// boundary `x^{(ℓ)}` and rematerializes interiors on the backward pass:
///
/// ```text
/// Forward:
///   x^{(0)} = Embed(ids)
///   for ℓ = 0 .. L-1:
///       save x^{(ℓ)}
///       x^{(ℓ+1)} = F_ℓ(x^{(ℓ)})          // fused GPU, drop interiors
///   ĥ = RMSNorm(x^{(L)})
///
/// Backward (identical to full tape):
///   g = ∂L/∂ĥ
///   g ← ∂ RMSNorm*(x^{(L)}, g)
///   for ℓ = L-1 .. 0:
///       (n1,h,n2,ff) = F_ℓ(x^{(ℓ)})      // recompute, same Metal kernel
///       g ← B_ℓ(g; n1,h,n2,ff)           // exact STE / bump backward
///   ∂L/∂E ← scatter(g, ids)
/// ```
///
/// Because `F_ℓ` is deterministic given weights, `B_ℓ(F_ℓ(x), g) = B_ℓ(tape, g)`.
/// Peak activation RAM is `Θ(L · n · d)` instead of `Θ(L · 4 · n · d)`
/// (plus resonance buffers), i.e. up to ~50% of the pre-training working set.
pub fn train(args: TrainArgs) -> Result<PathBuf> {
    let mut cfg = args.to_config();
    cfg.master = crate::config::MasterDtype::parse_name(&args.master)?;
    cfg.mom = crate::config::MomDtype::parse_name(&args.mom)?;
    if args.moe_topk > 2 {
        anyhow::bail!("--moe-topk {} not in 0|1|2", args.moe_topk);
    }
    cfg.moe_topk = args.moe_topk;
    cfg.moe_aux = args.moe_aux;
    cfg.kan_factor = crate::config::KanFactor::SharedEdge;
    cfg.fused_grad_ckpt = true;
    cfg.n_basis = cfg.grid_start;
    cfg.arch = crate::config::ModelArch::parse_name(&args.arch)?;
    cfg.expert_width = args.expert_width.max(1);
    cfg.n_slots = args.slots;
    cfg.mem_experts = args.experts;
    if cfg.arch == crate::config::ModelArch::Memory && cfg.moe_topk == 0 {
        cfg.moe_topk = 2;
    }
    crate::tokenizer::validate_vocab_size(cfg.vocab_size as u32)?;
    if cfg.arch == crate::config::ModelArch::Memory {
        return train_memory(args, cfg);
    }
    let device = setup_device_with(
        !args.cpu,
        DeviceFlags {
            fused_grad_ckpt: cfg.fused_grad_ckpt,
        },
    )?;
    cache_metal_hello_mb(process_memory_mb());
    println!("metal_hello={:.1}MB (rss gate: hello+12)", metal_hello_mb());
    let tok_path = if cfg.tokenizer_path.is_empty() {
        None
    } else {
        Some(Path::new(&cfg.tokenizer_path))
    };
    let data_path = PathBuf::from(&cfg.data_path);
    if !data_path.exists() {
        anyhow::bail!(
            "JSONL corpus missing: {} (pass --data; Ullis does not synthesize training text)",
            data_path.display()
        );
    }
    println!("corpus: loading {}", data_path.display());
    warn_corpus_homogeneity(&data_path)?;
    if cfg.vocab_size > 8192 {
        let embed_mb = (cfg.vocab_size * cfg.d_model * 4) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "warn: V={} embed FP32 ≈ {:.1} MB (×3 with grad+vel). Default quality path is V=8192.",
            cfg.vocab_size, embed_mb
        );
    }
    let texts = jsonl_corpus_texts(&data_path, 8_192)?;
    if texts.is_empty() {
        anyhow::bail!("JSONL corpus is empty: {}", data_path.display());
    }
    let nchars: usize = texts.iter().map(String::len).sum();
    println!(
        "corpus: {} records, {:.1} MB packed text; tokenizer V={}",
        texts.len(),
        nchars as f64 / (1024.0 * 1024.0),
        cfg.vocab_size
    );
    let tokenizer = load_or_train(cfg.vocab_size as u32, &texts, tok_path, cfg.seed)?;
    println!(
        "tokenizer ready: V={} pieces={} merges={}",
        tokenizer.vocab_size,
        tokenizer.populated(),
        tokenizer.merges.len()
    );
    cfg.vocab_size = tokenizer.vocab_size as usize;

    let mut stream = JsonlStream::open_with_cap(
        &data_path,
        tokenizer.clone(),
        cfg.seq_len,
        cfg.context_len,
        cfg.seed,
    )?;

    println!(
        "device={} vocab={} moe={} topk={} aux={} kan_factor={:?} master={:?} mom={:?} data={} jsonl=v4 {}",
        device_name(&device),
        tokenizer.vocab_size,
        cfg.moe,
        cfg.moe_topk,
        cfg.moe_aux,
        cfg.kan_factor,
        cfg.master,
        cfg.mom,
        data_path.display(),
        format_cfg(&cfg)
    );
    let n = cfg.batch_size * cfg.seq_len;
    println!(
        "ce_gemm n·V·d = {}·{}·{} (chunked sgemm, no [n,V] logits)",
        n, cfg.vocab_size, cfg.d_model
    );

    let mut model = UllisKan::new(cfg.clone(), device)?;
    println!("{}", model.param_report());

    let ckpt_dir = PathBuf::from(&cfg.ckpt_dir);
    std::fs::create_dir_all(&ckpt_dir)?;
    tokenizer.save(ckpt_dir.join("tokenizer.json"))?;

    for (phase, name, epochs_of, lr_of) in PHASES {
        let epochs = epochs_of(&cfg);
        let lr = lr_of(&cfg);
        model.set_phase(phase)?;
        let mut opt = SgdMomentum::new(&model, phase, lr, cfg.momentum, cfg.max_norm)?;
        println!(
            "\n== phase {phase} {name} epochs={epochs} lr={lr} G={} ==",
            model.cfg.n_basis
        );
        let t0 = Instant::now();
        let mut thru = Throughput::new();
        let mut global_step = 0usize;

        for epoch in 0..epochs {
            let mut running = 0.0f32;
            let mut n_seen = 0u32;
            thru.reset();

            for step in 0..cfg.steps_per_epoch {
                let lv = {
                    let (x, y, mask) = stream.next_batch(cfg.batch_size)?;
                    let l1 = if phase == 2 { cfg.l1 as f32 } else { 0.0 };
                    let lv = model.train_step(&x, &y, &mask, cfg.batch_size, cfg.seq_len, l1)?;
                    opt.step(&mut model, phase)?;
                    lv
                };
                synchronize(&model.device)?;
                global_step += 1;
                if let Some(g) = maybe_insert(&mut model, &mut cfg, phase, global_step) {
                    println!(
                        "  knot insert G -> {g} (non-uniform Gauss–Jordan, residual-energy site)"
                    );
                    opt = SgdMomentum::new(&model, phase, lr, cfg.momentum, cfg.max_norm)?;
                    println!("  {}", model.param_report());
                }

                running += lv;
                n_seen += 1;
                thru.add((cfg.batch_size * cfg.seq_len) as u64);

                if step % cfg.log_every == 0 {
                    let stats = model.ternary_stats()?;
                    let avg = running / n_seen.max(1) as f32;
                    let fp = train_footprint(&model, &opt, phase);
                    println!(
                        "  {name} e{epoch} s{step:04} loss={avg:.4} ce={:.4} H={:.3} Hr={:.3} mask={:.2} rss={:.1}MB{} tok/s={:.0} G={} zero={:.2} +={:.2} -={:.2} fwd={:.1}ms ce={:.1}ms bwd={:.1}ms waits={}",
                        model.last_ce,
                        model.last_entropy,
                        model.last_router_entropy,
                        model.last_mask,
                        fp.rss_mb,
                        fp.format_fields(),
                        thru.tok_s(),
                        model.cfg.n_basis,
                        stats.frac_zero,
                        stats.frac_pos,
                        stats.frac_neg,
                        model.last_fwd_ms,
                        model.last_ce_ms,
                        model.last_bwd_ms,
                        crate::telemetry::take_gpu_waits()
                    );
                    running = 0.0;
                    n_seen = 0;
                    thru.reset();
                }
            }
            checkpoint::save(
                ckpt_dir.join(format!("phase{phase}.bin")),
                &model,
                stream.tokenizer(),
                phase,
            )?;
            checkpoint::save(ckpt_dir.join("last.bin"), &model, stream.tokenizer(), phase)?;
        }
        println!("phase {phase} done in {:.1}s", t0.elapsed().as_secs_f32());
    }

    model.pack()?;
    let packed_path = ckpt_dir.join("packed.bin");
    checkpoint::save(&packed_path, &model, stream.tokenizer(), 4)?;
    write_card(
        &ckpt_dir.join("model_card.json"),
        &model,
        stream.tokenizer(),
    )?;
    println!("packed inference checkpoint -> {}", packed_path.display());
    println!("{}", model.param_report());
    Ok(packed_path)
}

fn train_memory(args: TrainArgs, mut cfg: TrainConfig) -> Result<PathBuf> {
    let data_path = PathBuf::from(&cfg.data_path);
    if !data_path.exists() {
        anyhow::bail!(
            "JSONL corpus missing: {} (pass --data; Ullis does not synthesize training text)",
            data_path.display()
        );
    }
    warn_corpus_homogeneity(&data_path)?;
    let texts = jsonl_corpus_texts(&data_path, 8_192)?;
    if texts.is_empty() {
        anyhow::bail!("JSONL corpus is empty: {}", data_path.display());
    }
    let tok_path = if cfg.tokenizer_path.is_empty() {
        None
    } else {
        Some(Path::new(&cfg.tokenizer_path))
    };
    let tokenizer = load_or_train(cfg.vocab_size as u32, &texts, tok_path, cfg.seed)?;
    cfg.vocab_size = tokenizer.vocab_size as usize;
    let mut stream = JsonlStream::open_with_cap(
        &data_path,
        tokenizer.clone(),
        cfg.seq_len,
        cfg.context_len,
        cfg.seed,
    )?;
    let mut rng = crate::device::rng_from_seed(cfg.seed);
    let device = setup_device(!args.cpu)?;
    let mut model = crate::memory::UllisMemory::with_device(cfg.clone(), &mut rng, device)?;
    println!(
        "arch=memory device={} {}",
        device_name(&model.device),
        model.param_report()
    );
    println!(
        "corpus {} V={} E={} W={} S={} k={} T={} B={}",
        data_path.display(),
        cfg.vocab_size,
        cfg.mem_experts,
        cfg.expert_width,
        cfg.n_slots,
        cfg.moe_topk,
        cfg.seq_len,
        cfg.batch_size
    );
    let ckpt_dir = PathBuf::from(&cfg.ckpt_dir);
    std::fs::create_dir_all(&ckpt_dir)?;
    tokenizer.save(ckpt_dir.join("tokenizer.json"))?;

    for (phase, name, epochs_of, lr_of) in PHASES {
        let epochs = epochs_of(&cfg);
        let lr = lr_of(&cfg);
        model.set_phase(phase);
        let mut opt = crate::optim::DenseSgd::new(&model.param_lens(), lr, cfg.momentum, cfg.max_norm);
        println!("\n== phase {phase} {name} epochs={epochs} lr={lr} memory ==");
        let t0 = Instant::now();
        let mut thru = Throughput::new();
        for epoch in 0..epochs {
            let mut running = 0.0f32;
            let mut n_seen = 0u32;
            thru.reset();
            for step in 0..cfg.steps_per_epoch {
                let (x, y, mask) = stream.next_batch(cfg.batch_size)?;
                let l1 = if phase == 2 { cfg.l1 as f32 } else { 0.0 };
                let lv = model.train_step(&x, &y, &mask, cfg.batch_size, cfg.seq_len, l1)?;
                crate::memory::memory_sgd_step(&mut model, &mut opt)?;
                running += lv;
                n_seen += 1;
                thru.add((cfg.batch_size * cfg.seq_len) as u64);
                if step % cfg.log_every == 0 {
                    let avg = running / n_seen.max(1) as f32;
                    println!(
                        "  {name} e{epoch} s{step:04} loss={avg:.4} ce={:.4} H={:.3} mask={:.2} rss={:.1}MB tok/s={:.0} fwd={:.1}ms ce={:.1}ms bwd={:.1}ms",
                        model.last_ce,
                        model.last_entropy,
                        model.last_mask,
                        process_memory_mb(),
                        thru.tok_s(),
                        model.last_fwd_ms,
                        model.last_ce_ms,
                        model.last_bwd_ms
                    );
                    running = 0.0;
                    n_seen = 0;
                    thru.reset();
                }
            }
        }
        checkpoint::save_memory(
            ckpt_dir.join(format!("phase{phase}.bin")),
            &model,
            stream.tokenizer(),
            phase,
        )?;
        checkpoint::save_memory(
            ckpt_dir.join("last.bin"),
            &model,
            stream.tokenizer(),
            phase,
        )?;
        println!(
            "phase {phase} done in {:.1}s  wrote {}/last.bin",
            t0.elapsed().as_secs_f32(),
            ckpt_dir.display()
        );
    }

    checkpoint::save_memory(
        ckpt_dir.join("last.bin"),
        &model,
        stream.tokenizer(),
        model.phase,
    )?;
    model.pack();
    let packed_path = ckpt_dir.join("packed.bin");
    checkpoint::save_memory(&packed_path, &model, stream.tokenizer(), 4)?;
    let card = serde_json::json!({
        "engine": "Ullis Memory",
        "arch": "memory",
        "config": model.cfg,
        "vocab_size": tokenizer.vocab_size,
        "report": model.param_report(),
        "cpu": args.cpu,
        "files": ["packed.bin", "last.bin", "tokenizer.json"],
    });
    std::fs::write(
        ckpt_dir.join("model_card.json"),
        serde_json::to_string_pretty(&card)?,
    )?;
    println!("memory packed checkpoint -> {}", packed_path.display());
    println!("memory last.bin (fp32)    -> {}", ckpt_dir.join("last.bin").display());
    println!("{}", model.param_report());
    Ok(packed_path)
}

fn maybe_insert(
    model: &mut UllisKan,
    cfg: &mut TrainConfig,
    phase: u8,
    global_step: usize,
) -> Option<usize> {
    let target = next_grid_size(cfg, phase, global_step, cfg.n_basis);
    if target <= cfg.n_basis {
        return None;
    }
    match model.insert_knot() {
        Ok(g) => {
            cfg.n_basis = g;
            Some(g)
        }
        Err(_) => None,
    }
}

fn train_footprint(model: &UllisKan, opt: &SgdMomentum, phase: u8) -> TrainFootprint {
    let rss_mb = process_memory_mb();
    let baseline_metal_mb = metal_hello_mb();
    let params_bytes = model.trainable_param_bytes(phase);
    TrainFootprint {
        rss_mb,
        baseline_metal_mb,
        net_mb: (rss_mb - baseline_metal_mb).max(0.0),
        params_bytes,
        grad_bytes: params_bytes,
        opt_bytes: opt.vel_bytes(),
        workspace_bytes: model.workspace_bytes(),
        gpu_alias: u8::from(model.device.backend() == Backend::Metal),
        embed_i8_bytes: model.embed_i8_bytes(),
        scratch_bumps: 0,
    }
}

fn format_cfg(cfg: &TrainConfig) -> String {
    format!(
        "d={} L={} G0={} T={} B={}",
        cfg.d_model, cfg.n_layers, cfg.grid_start, cfg.seq_len, cfg.batch_size
    )
}

fn write_card(path: &Path, model: &UllisKan, tokenizer: &BpeTokenizer) -> Result<()> {
    let mut packed_bytes = 0usize;
    for b in &model.blocks {
        if let Some(p) = &b.ff.packed_codes {
            packed_bytes += p.packed_base.len() + p.packed_shared.len() + p.packed_routed.len();
        }
    }
    let card = serde_json::json!({
        "engine": "Ullis AI Engine v0.9",
        "config": model.cfg,
        "vocab_size": tokenizer.vocab_size,
        "n_merges": tokenizer.merges.len(),
        "packed_bytes": packed_bytes,
        "embed_params": model.embed.numel(),
        "report": model.param_report(),
    });
    std::fs::write(path, serde_json::to_string_pretty(&card)?)?;
    Ok(())
}

pub fn run_smoke(cpu: bool) -> Result<()> {
    let device = setup_device(!cpu)?;
    println!("smoke device={}", device_name(&device));

    let mut rng = crate::device::rng_from_seed(0);
    let mut layer = crate::kan::TernaryKanLinear::new(16, 8, 4, false, 1, 0.7, &mut rng)?;
    let x = crate::mixers::randn(4 * 16, 1.0, &mut rng);
    layer.set_phase(1)?;
    let y1 = layer.forward(&device, &x, 4)?;
    assert_eq!(y1.len(), 4 * 8);

    let mut wide = crate::kan::TernaryKanLinear::new(512, 64, 4, false, 1, 0.7, &mut rng)?;
    let xw = crate::mixers::randn(2 * 512, 1.0, &mut rng);
    let yw = wide.forward(&device, &xw, 2)?;
    assert_eq!(yw.len(), 2 * 64);
    assert!(yw.iter().all(|v| v.is_finite()));

    let y_coarse = y1.clone();
    layer.extend_grid(8)?;
    assert_eq!(layer.n_basis, 8);
    let y_mid = layer.forward(&device, &x, 4)?;
    let err_grid: f32 = y_coarse
        .iter()
        .zip(y_mid.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / y_coarse.len() as f32;
    assert!(err_grid < 0.15, "grid 4->8 drifted by {err_grid}");

    layer.extend_grid(12)?;
    let y_fine = layer.forward(&device, &x, 4)?;
    let n = y_fine.len() as f32;
    let mut dy: Vec<f32> = y_fine.iter().map(|v| 2.0 * v / n).collect();
    let _dx = layer.backward(&x, &dy, 4, crate::kan::KanEvalMode::Full)?;
    assert!(layer.weight_shared.as_ref().unwrap().numel() > 0);

    layer.set_phase(3)?;
    layer.zero_grad();
    let y3 = layer.forward(&device, &x, 4)?;
    let n = y3.len() as f32;
    dy = y3.iter().map(|v| 2.0 * v / n).collect();
    let _ = layer.backward(&x, &dy, 4, crate::kan::KanEvalMode::Full)?;
    assert!(layer.grad_base.iter().any(|g| *g != 0.0));

    let (cb, _, _) = layer.snapshot_codes()?;
    let packed = crate::quant::pack_ternary(&cb);
    let rt = crate::quant::unpack_ternary(&packed, cb.len());
    assert_eq!(rt, cb, "2-bit pack/unpack must be lossless");

    let y_qat = layer.forward(&device, &x, 4)?;
    layer.pack()?;
    let y_pk = layer.forward(&device, &x, 4)?;
    let err: f32 = y_qat
        .iter()
        .zip(y_pk.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / y_qat.len() as f32;

    let mut gram = vec![0.0f32; 25];
    for i in 0..5 {
        gram[i * 5 + i] = 2.0;
    }
    let rhs: Vec<f32> = (0..15).map(|i| i as f32).collect();
    let solved = crate::gauss::solve_square(&gram, 5, &rhs, 3)?;
    let mut recon = vec![0.0f32; 15];
    crate::accelerate::sgemm(5, 3, 5, 1.0, &gram, &solved, 0.0, &mut recon)?;
    let rec_err: f32 = recon
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 15.0;
    assert!(rec_err < 1e-3, "gauss-jordan residual {rec_err}");

    let texts = vec![
        "def load(path):\n    return path\n".into(),
        "fn main() {\n    match x {\n        Ok(s) => s,\n    }\n}\n".into(),
        "#!/usr/bin/env bash\nset -euo pipefail\n".into(),
    ];
    let mut tok = crate::tokenizer::train_bpe(&texts, 512, 0)?;
    let sample = "def load(path):\n    return path\n";
    let ids = tok.encode(sample, false, false);
    assert_eq!(tok.decode(&ids), sample);

    let mut cfg = TrainConfig {
        d_model: 16,
        n_layers: 2,
        n_basis: 4,
        grid_start: 4,
        seq_len: 24,
        batch_size: 2,
        mixer: "shift".into(),
        vocab_size: tok.vocab_size as usize,
        moe: false,
        ..TrainConfig::default()
    };
    let mut model = UllisKan::new(cfg.clone(), device)?;
    model.set_phase(1)?;
    let ids_t: Vec<u32> = (0u32..48).map(|i| i % tok.vocab_size).collect();
    let logits = model.forward(&ids_t, 2, 24)?;
    assert_eq!(logits.len(), 2 * 24 * tok.vocab_size as usize);
    model.extend_grid(8)?;
    cfg.n_basis = 8;
    model.set_phase(4)?;
    model.pack()?;
    let mut rng = crate::device::rng_from_seed(0);
    let _ = model.generate_stream_pieces("def run(", &mut tok, 8, 0.0, &mut rng)?;
    let y_full = model.forward(&ids_t, 2, 24)?;
    let y_coarse = model.forward_mode(&ids_t, 2, 24, crate::kan::KanEvalMode::Coarse, false)?;
    assert_eq!(y_full.len(), y_coarse.len());

    println!(
        "smoke ok device={} packed_err={err:.2e} grid_err={err_grid:.2e} {}",
        device_name(&model.device),
        model.param_report()
    );
    Ok(())
}
