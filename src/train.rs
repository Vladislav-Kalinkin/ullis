use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use candle_core::DType;
use clap::Args;

use crate::checkpoint;
use crate::config::{grid_target, TrainConfig};
use crate::data::JsonlStream;
use crate::device::{device_name, scalar_f32, setup_device, synchronize};
use crate::model::UllisKan;
use crate::optim::SgdMomentum;
use crate::seed::{corpus_texts, ensure_jsonl};
use crate::telemetry::{process_memory_mb, Throughput};
use crate::tokenizer::{load_or_train, BpeTokenizer};

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// JSONL corpus (`{"system","user","thinking","output"}` per line; legacy `text`/`lang` lifted).
    #[arg(long, default_value = "data/train.jsonl")]
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
    #[arg(long, default_value_t = 4096)]
    pub vocab: u32,
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
    #[arg(long)]
    pub cpu: bool,
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
            batch_size: self.batch_size,
            mixer: self.mixer.clone(),
            vocab_size: self.vocab as usize,
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

pub fn train(args: TrainArgs) -> Result<PathBuf> {
    let mut cfg = args.to_config();
    cfg.n_basis = cfg.grid_start;
    let device = setup_device(!args.cpu)?;
    let tok_path = if cfg.tokenizer_path.is_empty() {
        None
    } else {
        Some(Path::new(&cfg.tokenizer_path))
    };
    let texts = corpus_texts(240, cfg.seed);
    let tokenizer = load_or_train(cfg.vocab_size as u32, &texts, tok_path, cfg.seed)?;
    cfg.vocab_size = tokenizer.vocab_size as usize;

    let data_path = ensure_jsonl(&cfg.data_path, cfg.seed)?;
    let mut stream = JsonlStream::open(&data_path, tokenizer.clone(), cfg.seq_len, cfg.seed)?;

    println!(
        "device={} vocab={} moe={} data={} jsonl=v4 {}",
        device_name(&device),
        tokenizer.vocab_size,
        cfg.moe,
        data_path.display(),
        format_cfg(&cfg)
    );

    let mut model = UllisKan::new(cfg.clone(), &device)?;
    println!("{}", model.param_report());

    let ckpt_dir = PathBuf::from(&cfg.ckpt_dir);
    std::fs::create_dir_all(&ckpt_dir)?;
    tokenizer.save(ckpt_dir.join("tokenizer.json"))?;

    for (phase, name, epochs_of, lr_of) in PHASES {
        let epochs = epochs_of(&cfg);
        let lr = lr_of(&cfg);
        model.set_phase(phase)?;
        let mut opt =
            SgdMomentum::new(model.trainable_vars(phase), lr, cfg.momentum, cfg.max_norm)?;
        println!(
            "\n== phase {phase} {name} epochs={epochs} lr={lr} G={} ==",
            model.cfg.n_basis
        );
        let t0 = Instant::now();
        let mut thru = Throughput::new();

        for epoch in 0..epochs {
            if let Some(target) = maybe_grid(&mut model, &mut cfg, phase, epoch) {
                println!("  grid G -> {target} (vectorized Gauss–Jordan projection)");
                opt =
                    SgdMomentum::new(model.trainable_vars(phase), lr, cfg.momentum, cfg.max_norm)?;
                println!("  {}", model.param_report());
            }
            let mut running = 0.0f32;
            let mut n_seen = 0u32;
            thru.reset();

            for step in 0..cfg.steps_per_epoch {
                let lv = {
                    let (x, y, mask) = stream.next_batch(cfg.batch_size, &device)?;
                    let logits = model.forward(&x)?;
                    let (b, t, v) = logits.dims3()?;
                    let mut loss = crate::data::masked_cross_entropy(
                        &logits.reshape((b * t, v))?,
                        &y.flatten_all()?,
                        &mask.flatten_all()?,
                    )?;
                    // Drop logits before L1 / backward so the CE graph is the only tape.
                    drop(logits);
                    drop(mask);
                    if phase == 2 {
                        loss = (loss + (model.l1_penalty()? * cfg.l1)?)?;
                    }
                    let grads = loss.backward()?;
                    let lv = scalar_f32(&loss)?;
                    drop(loss);
                    opt.step(&grads)?;
                    drop(grads);
                    drop(x);
                    drop(y);
                    lv
                };
                // Return Metal command-buffer scratch to the pool this step.
                synchronize(&device)?;

                running += lv;
                n_seen += 1;
                thru.add((cfg.batch_size * cfg.seq_len) as u64);

                if step % cfg.log_every == 0 {
                    let stats = model.ternary_stats()?;
                    let avg = running / n_seen.max(1) as f32;
                    println!(
                        "  {name} e{epoch} s{step:04} loss={avg:.4} rss={:.1}MB tok/s={:.0} G={} zero={:.2} +={:.2} -={:.2}",
                        process_memory_mb(),
                        thru.tok_s(),
                        model.cfg.n_basis,
                        stats.frac_zero,
                        stats.frac_pos,
                        stats.frac_neg
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
    let _ = DType::F32;
    Ok(packed_path)
}

fn maybe_grid(
    model: &mut UllisKan,
    cfg: &mut TrainConfig,
    phase: u8,
    epoch: usize,
) -> Option<usize> {
    if phase >= 3 {
        return None;
    }
    let target = grid_target(cfg, phase, epoch);
    if target <= cfg.n_basis {
        return None;
    }
    if model.extend_grid(target).is_err() {
        return None;
    }
    cfg.n_basis = target;
    Some(target)
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
        "engine": "Ullis AI Engine v0.5",
        "config": model.cfg,
        "vocab_size": tokenizer.vocab_size,
        "n_merges": tokenizer.merges.len(),
        "packed_bytes": packed_bytes,
        "embed_params": model.embed.as_tensor().elem_count(),
        "report": model.param_report(),
    });
    std::fs::write(path, serde_json::to_string_pretty(&card)?)?;
    Ok(())
}

pub fn run_smoke(cpu: bool) -> Result<()> {
    use candle_core::{Device, Tensor};

    let device = setup_device(!cpu)?;
    println!("smoke device={}", device_name(&device));

    let mut rng = crate::device::rng_from_seed(0);
    let mut layer = crate::kan::TernaryKanLinear::new(16, 8, 4, false, 1, 0.7, &device, &mut rng)?;
    let x = crate::mixers::randn(&[4, 16], 1.0, &device, &mut rng)?;
    layer.set_phase(1)?;
    let y1 = layer.forward(&x)?;
    assert_eq!(y1.dims(), &[4, 8]);

    let y_coarse = y1.detach();
    layer.extend_grid(8)?;
    assert_eq!(layer.n_basis, 8);
    let y_mid = layer.forward(&x)?;
    let err_grid = {
        let d = (y_coarse - y_mid.detach())?.abs()?.mean_all()?;
        scalar_f32(&d)?
    };
    assert!(err_grid < 0.15, "grid 4->8 drifted by {err_grid}");

    layer.extend_grid(12)?;
    let y_fine = layer.forward(&x)?;
    let loss = y_fine.sqr()?.mean_all()?;
    let _grads = loss.backward()?;
    assert!(
        layer
            .weight_shared
            .as_ref()
            .unwrap()
            .as_tensor()
            .elem_count()
            > 0
    );

    layer.set_phase(3)?;
    let y3 = layer.forward(&x)?;
    let loss = y3.sqr()?.mean_all()?;
    let grads = loss.backward()?;
    assert!(grads
        .get(layer.weight_base.as_ref().unwrap().as_tensor())
        .is_some());

    let (cb, _, _) = layer.snapshot_codes()?;
    let packed = crate::quant::pack_ternary(&cb);
    let rt = crate::quant::unpack_ternary(&packed, cb.len());
    assert_eq!(rt, cb, "2-bit pack/unpack must be lossless");

    let y_qat = layer.forward(&x)?.detach();
    layer.pack()?;
    let y_pk = layer.forward(&x)?;
    let err = scalar_f32(&(y_qat - y_pk)?.abs()?.mean_all()?)?;

    let gram = Tensor::eye(5, DType::F32, &device)?.affine(2.0, 0.0)?;
    let rhs = Tensor::arange(0f32, 15f32, &device)?.reshape((5, 3))?;
    let solved = crate::gauss::mps_safe_solve(&gram, &rhs)?;
    let recon = gram.matmul(&solved)?;
    let rec_err = scalar_f32(&(recon - rhs)?.abs()?.mean_all()?)?;
    assert!(rec_err < 1e-3, "gauss-jordan residual {rec_err}");

    let texts = corpus_texts(80, 0);
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
    let mut model = UllisKan::new(cfg.clone(), &device)?;
    model.set_phase(1)?;
    let ids_t = Tensor::from_vec(
        (0u32..48).map(|i| i % tok.vocab_size).collect::<Vec<_>>(),
        (2, 24),
        &device,
    )?;
    let logits = model.forward(&ids_t)?;
    assert_eq!(logits.dims(), &[2, 24, tok.vocab_size as usize]);
    model.extend_grid(8)?;
    cfg.n_basis = 8;
    model.set_phase(4)?;
    model.pack()?;
    let mut rng = crate::device::rng_from_seed(0);
    let _ = model.generate_stream_pieces("def run(", &mut tok, 8, 0.0, &mut rng)?;
    let y_full = model.forward(&ids_t)?;
    let y_coarse = model.forward_mode(&ids_t, crate::kan::KanEvalMode::Coarse)?;
    assert_eq!(y_full.dims(), y_coarse.dims());
    drop(y_full);
    drop(y_coarse);

    println!(
        "smoke ok device={} packed_err={err:.2e} grid_err={err_grid:.2e} {}",
        device_name(&device),
        model.param_report()
    );
    let _ = Device::Cpu;
    Ok(())
}
