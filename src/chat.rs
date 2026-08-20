//! Streaming REPL with multi-tier `--thinking` budgets and ANSI visual reasoning.
//!
//! Thinking tokens flush immediately in dim/italic gray. The first
//! `<|/thinking|>` (or `</thinking>`) prints `└──` and the `output` lane
//! streams in bold green. `ReasoningScratch::clear` runs the instant the
//! visible stream ends — thinking never enters `DialogueCache`.

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use rand::SeedableRng;

use crate::checkpoint;
use crate::data::{pack_record, TAG_OUTPUT, TAG_SYSTEM, TAG_THINKING, TAG_THINK_END, TAG_USER};
use crate::device::{device_name, setup_device, synchronize};
use crate::kan::KanEvalMode;
use crate::model::UllisKan;
use crate::telemetry::process_memory_mb;
use crate::think::{strip_tags, thinking_closed, DialogueCache, ReasoningScratch, ThinkingMode};
use crate::tokenizer::{BpeTokenizer, StreamDecoder};

const BANNER: &str = r"
  _   _ _ _ _
 | | | | | (_)___
 | | | | | | / __|
 | |_| | | | \__ \
  \___/  |_|_|___/

 Ullis AI Engine v0.6 Sovereign | fused Metal MoB-KAN
 type a prefix  ·  language is inferred  ·  /help  /exit
";

const HELP: &str = "\
commands
  /help                 this text
  /exit /quit           leave the stream
  /clear                reset conversation context
  /temp <f>             sampling temperature (0 = greedy)
  /max  <n>             max new tokens (visible output)
  /thinking <tier>      low | medium | high | xhigh
  /system <text>        replace the system prompt
  /stats                model report + rss
thinking streams dim/italic; output streams bold green after └──
anything else is a prompt — `def ` steers Python, `fn ` steers Rust
";

const DEFAULT_SYSTEM: &str = "You are a compact ternary KAN code engine. Infer the language from the prompt tokens and emit well-formed source.";

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_THINK: &str = "\x1b[2;3m";
const ANSI_OUTPUT: &str = "\x1b[1;32m";
const THINK_BANNER: &str = "[Ullis is thinking...]";
const DIVIDER: &str = "└──";

const STRUCTURAL_TAGS: &[&str] = &[
    TAG_THINK_END,
    TAG_THINKING,
    TAG_OUTPUT,
    TAG_SYSTEM,
    TAG_USER,
    "</thinking>",
];

#[derive(Debug, Args)]
pub struct ChatArgs {
    /// Packed checkpoint (`packed.bin`) or a directory containing it.
    #[arg(long, default_value = "checkpoints/packed.bin")]
    pub model: PathBuf,
    #[arg(long)]
    pub prompt: Option<String>,
    #[arg(long, default_value_t = 120)]
    pub max_new: usize,
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
    /// Reasoning budget: low (bypass) / medium / high / xhigh (resonance).
    #[arg(long, value_enum, default_value_t = ThinkingMode::Medium)]
    pub thinking: ThinkingMode,
    /// Seeds the early hidden state (`system` key).
    #[arg(long, default_value = DEFAULT_SYSTEM)]
    pub system: String,
    #[arg(long)]
    pub cpu: bool,
}

pub fn run_chat(args: ChatArgs) -> Result<()> {
    let device = setup_device(!args.cpu)?;
    let path = resolve_model(&args.model)?;
    let loaded =
        checkpoint::load(&path, device).with_context(|| format!("load {}", path.display()))?;
    let mut model = loaded.model;
    let mut tokenizer = loaded.tokenizer;
    println!(
        "loaded {} on {}  {}  thinking={}",
        path.display(),
        device_name(&model.device),
        model.param_report(),
        args.thinking.as_str()
    );
    if let Some(p) = args.prompt {
        print_banner();
        let mut cache = DialogueCache::new(args.system.clone());
        stream_turn(
            &mut model,
            &mut tokenizer,
            &mut cache,
            &p,
            args.max_new,
            args.temperature,
            args.thinking,
        )?;
        return Ok(());
    }
    repl(
        &mut model,
        &mut tokenizer,
        args.max_new,
        args.temperature,
        args.thinking,
        args.system,
    )
}

fn resolve_model(p: &Path) -> Result<PathBuf> {
    if p.is_dir() {
        let cand = p.join("packed.bin");
        if cand.exists() {
            return Ok(cand);
        }
        let cand = p.join("last.bin");
        if cand.exists() {
            return Ok(cand);
        }
    }
    Ok(p.to_path_buf())
}

fn print_banner() {
    print!("{BANNER}");
    let _ = io::stdout().flush();
}

fn use_color() -> bool {
    io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn stream_turn(
    model: &mut UllisKan,
    tokenizer: &mut BpeTokenizer,
    cache: &mut DialogueCache,
    user: &str,
    max_new: usize,
    temperature: f32,
    thinking: ThinkingMode,
) -> Result<String> {
    print!("ullis▸ ");
    let _ = io::stdout().flush();

    let (system, user_block) = cache.pack_user(user);
    let kan_mode = thinking.kan_mode();
    let think_budget = thinking.think_budget(model.cfg.seq_len);
    let mut rng = rand::rngs::StdRng::from_os_rng();
    let t0 = Instant::now();
    let mut scratch = ReasoningScratch::new();
    let eos = tokenizer.eos_id;
    let color = use_color();
    let mut paint = PaintScan::new(think_budget > 0);
    let mut stdout = io::stdout();

    let seed = if think_budget == 0 {
        pack_record(&system, &user_block, None, Some(""))
    } else {
        let mut s = pack_record(&system, &user_block, None, None);
        s.push_str(TAG_THINKING);
        s.push('\n');
        s
    };
    let mut ctx = tokenizer.encode(&seed, false, false);
    trim_ctx(&mut ctx, model.cfg.seq_len);

    let mut emitted = 0usize;
    if think_budget > 0 {
        let mut dec = StreamDecoder::new(tokenizer);
        for _ in 0..think_budget {
            let nxt = model.next_token(&ctx, temperature, kan_mode, &mut rng)?;
            ctx.push(nxt);
            scratch.push_token(nxt);
            emitted += 1;
            let piece = dec.push(nxt);
            if !piece.is_empty() {
                scratch.push_text(&piece);
                emit_ops(&mut stdout, color, &paint.feed(&piece))?;
            }
            if nxt == eos || thinking_closed(scratch.text()) {
                break;
            }
        }
        let tail = dec.flush();
        if !tail.is_empty() {
            scratch.push_text(&tail);
            emit_ops(&mut stdout, color, &paint.feed(&tail))?;
        }
        if paint.lane == Lane::Think {
            emit_ops(&mut stdout, color, &paint.feed(TAG_THINK_END))?;
        }
        if !scratch.text().contains(TAG_THINK_END) {
            let closer = format!("{TAG_THINK_END}\n");
            ctx.extend(tokenizer.encode(&closer, false, false));
        }
        let out_tag = format!("{TAG_OUTPUT}\n");
        ctx.extend(tokenizer.encode(&out_tag, false, false));
        trim_ctx(&mut ctx, model.cfg.seq_len);
    } else if color {
        write!(stdout, "{ANSI_OUTPUT}")?;
        stdout.flush()?;
    }

    let (visible, n_out) = color_stream(
        model,
        tokenizer,
        &mut ctx,
        &mut paint,
        max_new,
        temperature,
        kan_mode,
        eos,
        color,
        &mut rng,
        &mut stdout,
    )?;
    emitted += n_out;
    emit_ops(&mut stdout, color, &[PaintOp::Reset])?;
    stdout.flush()?;
    println!();

    // Ephemeral GC: thinking activations/tokens leave the ring before persist.
    scratch.clear();
    ctx.clear();
    ctx.shrink_to_fit();
    synchronize(&model.device)?;

    cache.persist_turn(user.trim().to_string(), visible.clone());

    let dt = t0.elapsed().as_secs_f64().max(1e-9);
    let stats = model.ternary_stats().unwrap_or_default();
    println!(
        "  └ {emitted} tok  {:.1} tok/s  rss={:.1}MB  think={}  zero={:.2} +={:.2} -={:.2}",
        emitted as f64 / dt,
        process_memory_mb(),
        thinking.as_str(),
        stats.frac_zero,
        stats.frac_pos,
        stats.frac_neg
    );
    Ok(visible)
}

fn trim_ctx(ctx: &mut Vec<u32>, seq_len: usize) {
    if ctx.len() > seq_len {
        let drain = ctx.len() - seq_len;
        ctx.drain(..drain);
    }
}

fn color_stream(
    model: &mut UllisKan,
    tokenizer: &BpeTokenizer,
    ctx: &mut Vec<u32>,
    paint: &mut PaintScan,
    max_new: usize,
    temperature: f32,
    mode: KanEvalMode,
    eos: u32,
    color: bool,
    rng: &mut impl rand::Rng,
    stdout: &mut impl Write,
) -> Result<(String, usize)> {
    let mut dec = StreamDecoder::new(tokenizer);
    let mut emitted = 0usize;
    for _ in 0..max_new {
        let nxt = model.next_token(ctx, temperature, mode, rng)?;
        ctx.push(nxt);
        emitted += 1;
        let piece = dec.push(nxt);
        if !piece.is_empty() {
            emit_ops(stdout, color, &paint.feed(&piece))?;
        }
        if nxt == eos {
            break;
        }
        if paint.output_text.ends_with("\n\n") && emitted > 4 {
            break;
        }
    }
    let tail = dec.flush();
    if !tail.is_empty() {
        emit_ops(stdout, color, &paint.feed(&tail))?;
    }
    emit_ops(stdout, color, &paint.finish())?;
    Ok((strip_tags(&paint.output_text).trim().to_string(), emitted))
}

fn emit_ops(w: &mut impl Write, color: bool, ops: &[PaintOp]) -> io::Result<()> {
    for op in ops {
        match op {
            PaintOp::Banner => {
                if color {
                    writeln!(w, "{ANSI_THINK}{THINK_BANNER}")?;
                } else {
                    writeln!(w, "{THINK_BANNER}")?;
                }
            }
            PaintOp::Think(s) | PaintOp::Output(s) => write!(w, "{s}")?,
            PaintOp::Divider => {
                if color {
                    write!(w, "{ANSI_RESET}\n{DIVIDER}\n{ANSI_OUTPUT}")?;
                } else {
                    write!(w, "\n{DIVIDER}\n")?;
                }
            }
            PaintOp::Reset => {
                if color {
                    write!(w, "{ANSI_RESET}")?;
                }
            }
        }
        w.flush()?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Think,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PaintOp {
    Banner,
    Think(String),
    Divider,
    Output(String),
    Reset,
}

#[derive(Debug)]
struct PaintScan {
    pending: String,
    lane: Lane,
    bannered: bool,
    divided: bool,
    output_text: String,
}

impl PaintScan {
    fn new(start_think: bool) -> Self {
        Self {
            pending: String::new(),
            lane: if start_think {
                Lane::Think
            } else {
                Lane::Output
            },
            bannered: false,
            divided: false,
            output_text: String::new(),
        }
    }

    fn feed(&mut self, piece: &str) -> Vec<PaintOp> {
        self.pending.push_str(piece);
        self.drain(false)
    }

    fn finish(&mut self) -> Vec<PaintOp> {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> Vec<PaintOp> {
        let mut ops = Vec::new();
        loop {
            if self.pending.is_empty() {
                break;
            }
            if let Some((i, len, kind)) = earliest_tag(&self.pending) {
                let prefix = self.pending[..i].to_string();
                self.pending.drain(..i + len);
                self.emit_text(&prefix, &mut ops);
                match kind {
                    TagKind::ThinkEnd | TagKind::Output if self.lane == Lane::Think => {
                        self.close_think(&mut ops);
                    }
                    TagKind::ThinkStart if self.lane == Lane::Output => {
                        self.lane = Lane::Think;
                    }
                    _ => {}
                }
                continue;
            }
            if flush {
                let rest = std::mem::take(&mut self.pending);
                self.emit_text(&rest, &mut ops);
            } else {
                let hold = hold_len(&self.pending);
                let emit_n = self.pending.len() - hold;
                if emit_n == 0 {
                    break;
                }
                let text = self.pending[..emit_n].to_string();
                self.pending.drain(..emit_n);
                self.emit_text(&text, &mut ops);
            }
        }
        ops
    }

    fn close_think(&mut self, ops: &mut Vec<PaintOp>) {
        self.ensure_banner(ops);
        if !self.divided {
            ops.push(PaintOp::Divider);
            self.divided = true;
        }
        self.lane = Lane::Output;
    }

    fn ensure_banner(&mut self, ops: &mut Vec<PaintOp>) {
        if self.lane == Lane::Think && !self.bannered {
            ops.push(PaintOp::Banner);
            self.bannered = true;
        }
    }

    fn emit_text(&mut self, text: &str, ops: &mut Vec<PaintOp>) {
        if text.is_empty() {
            return;
        }
        match self.lane {
            Lane::Think => {
                self.ensure_banner(ops);
                ops.push(PaintOp::Think(text.to_string()));
            }
            Lane::Output => {
                if !self.divided && self.bannered {
                    ops.push(PaintOp::Divider);
                    self.divided = true;
                }
                self.output_text.push_str(text);
                ops.push(PaintOp::Output(text.to_string()));
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TagKind {
    ThinkStart,
    ThinkEnd,
    Output,
    Other,
}

fn earliest_tag(s: &str) -> Option<(usize, usize, TagKind)> {
    let mut best: Option<(usize, usize, TagKind)> = None;
    for (tag, kind) in [
        (TAG_THINK_END, TagKind::ThinkEnd),
        ("</thinking>", TagKind::ThinkEnd),
        (TAG_THINKING, TagKind::ThinkStart),
        (TAG_OUTPUT, TagKind::Output),
        (TAG_SYSTEM, TagKind::Other),
        (TAG_USER, TagKind::Other),
    ] {
        if let Some(i) = s.find(tag) {
            let better = match best {
                None => true,
                Some((bi, blen, _)) => i < bi || (i == bi && tag.len() > blen),
            };
            if better {
                best = Some((i, tag.len(), kind));
            }
        }
    }
    best
}

fn hold_len(s: &str) -> usize {
    for (i, _) in s.char_indices() {
        let suffix = &s[i..];
        if STRUCTURAL_TAGS
            .iter()
            .any(|t| t.starts_with(suffix) && t.len() > suffix.len())
        {
            return s.len() - i;
        }
    }
    0
}

fn repl(
    model: &mut UllisKan,
    tokenizer: &mut BpeTokenizer,
    mut max_new: usize,
    mut temperature: f32,
    mut thinking: ThinkingMode,
    system: String,
) -> Result<()> {
    print_banner();
    println!("{}", model.param_report());
    let mut cache = DialogueCache::new(system);
    let stdin = io::stdin();
    loop {
        print!("you▸ ");
        let _ = io::stdout().flush();
        let mut raw = String::new();
        let n = stdin.read_line(&mut raw)?;
        if n == 0 {
            println!();
            break;
        }
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "/exit" | "/quit" | "/q") {
            break;
        }
        if matches!(line, "/help" | "/?") {
            print!("{HELP}");
            continue;
        }
        if line == "/clear" {
            cache.clear();
            println!("context cleared");
            continue;
        }
        if line == "/stats" {
            let stats = model.ternary_stats().unwrap_or_default();
            println!(
                "{}  rss={:.1}MB  thinking={}  turns={}  zero={:.2} +={:.2} -={:.2}",
                model.param_report(),
                process_memory_mb(),
                thinking.as_str(),
                cache.turn_count(),
                stats.frac_zero,
                stats.frac_pos,
                stats.frac_neg
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("/thinking") {
            let rest = rest.trim();
            if rest.is_empty() {
                println!("thinking={}", thinking.as_str());
            } else if let Ok(v) = rest.parse::<ThinkingMode>() {
                thinking = v;
                println!("thinking={}", thinking.as_str());
            } else {
                println!("usage: /thinking low|medium|high|xhigh");
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("/system") {
            let rest = rest.trim();
            if rest.is_empty() {
                println!("system={}", cache.system());
            } else {
                cache.set_system(rest.to_string());
                println!("system updated");
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("/temp") {
            let rest = rest.trim();
            if rest.is_empty() {
                println!("temperature={temperature}");
            } else if let Ok(v) = rest.parse::<f32>() {
                temperature = v;
                println!("temperature={temperature}");
            } else {
                println!("usage: /temp 0.7");
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("/max") {
            let rest = rest.trim();
            if rest.is_empty() {
                println!("max_new={max_new}");
            } else if let Ok(v) = rest.parse::<usize>() {
                max_new = v.max(1);
                println!("max_new={max_new}");
            } else {
                println!("usage: /max 120");
            }
            continue;
        }

        let user = raw.trim_end_matches('\n');
        stream_turn(
            model,
            tokenizer,
            &mut cache,
            user,
            max_new,
            temperature,
            thinking,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_tag_across_tokens_closes_think() {
        let mut p = PaintScan::new(true);
        let a = p.feed("check braces { } <|/");
        assert!(a.iter().any(|op| matches!(op, PaintOp::Banner)));
        assert!(a.iter().any(|op| matches!(op, PaintOp::Think(_))));
        assert!(!a.iter().any(|op| matches!(op, PaintOp::Divider)));
        let b = p.feed("thinking|>\nfn add() {}");
        assert!(b.iter().any(|op| matches!(op, PaintOp::Divider)));
        assert!(b
            .iter()
            .any(|op| matches!(op, PaintOp::Output(s) if s.contains("fn add"))));
        assert_eq!(p.lane, Lane::Output);
        assert!(p.output_text.contains("fn add() {}"));
        assert!(!a
            .iter()
            .any(|op| matches!(op, PaintOp::Think(s) if s.contains("fn add"))));
    }

    #[test]
    fn html_think_end_also_divides() {
        let mut p = PaintScan::new(true);
        let _ = p.feed("lifetime of s: &str");
        let ops = p.feed("</thinking>\npub fn f(s: &str) -> usize { s.len() }");
        assert!(ops.iter().any(|op| matches!(op, PaintOp::Divider)));
        assert!(p.output_text.contains("pub fn f"));
    }

    #[test]
    fn rust_generic_lt_is_not_a_tag() {
        let mut p = PaintScan::new(true);
        let ops = p.feed("Vec<i32> map");
        assert!(ops
            .iter()
            .any(|op| matches!(op, PaintOp::Think(s) if s.contains("Vec<i32>"))));
        assert!(!ops.iter().any(|op| matches!(op, PaintOp::Divider)));
        assert_eq!(p.lane, Lane::Think);
    }

    #[test]
    fn low_mode_skips_banner() {
        let mut p = PaintScan::new(false);
        let ops = p.feed("print(1)\n");
        assert!(!ops.iter().any(|op| matches!(op, PaintOp::Banner)));
        assert!(ops.iter().any(|op| matches!(op, PaintOp::Output(_))));
    }
}
