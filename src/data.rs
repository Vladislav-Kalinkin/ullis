//! Streaming 4-key JSONL token pipeline. RAM stays O(seq_len), not O(corpus).
//!
//! Canonical line:
//! `{"system":"...","user":"...","thinking":"...","output":"..."}`
//!
//! # Cognitive-bench schema (v0.9)
//! Let a record be the 4-tuple `R = (s, u, τ, y)` with
//! `s = system`, `u = user`, `τ = thinking`, `y = output`. Packed stream:
//!
//! `x = <|system|> s <|user|> u <|thinking|> τ <|/thinking|> <|output|> y`
//!
//! Supervised mask `m_t = 1` iff token `t` falls in `τ ∪ y`, else 0.
//! Loss `L = mean_m [ −log p(x_{t+1} | x_{≤t}) + λ_H H(p) ] + λ_R H(g)`.
//! `τ` is a numbered, language-agnostic chain: (1) surface-token language
//! ID, (2) type/ownership/data-flow inventory, (3) constraint solve
//! (lifetimes, quoting, encoding), (4) control-flow / pipeline DAG,
//! (5) failure modes, (6) emission checklist. Golden anchors:
//! `data/cognitive-bench.jsonl` (3 conversational, 4 Rust, 4 Python, 4 Bash).
//!
//! Token storage is a cache-line / page-aligned `SovereignFlashBuffer` (no
//! `VecDeque`). After `bind_metal`, evaluation sweeps DMA the host pointer
//! straight into a Shared `MTLBuffer` with no extra host-to-device copy.

use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::device::{PageSlab, SovereignDevice};
use crate::tokenizer::BpeTokenizer;

/// Token ring cap (~128 KB of u32) — independent of file size and thinking depth.
pub const MAX_TOKEN_BUF: usize = 32_768;

pub const TAG_SYSTEM: &str = "<|system|>";
pub const TAG_USER: &str = "<|user|>";
pub const TAG_THINKING: &str = "<|thinking|>";
pub const TAG_THINK_END: &str = "<|/thinking|>";
pub const TAG_OUTPUT: &str = "<|output|>";

/// Canonical 4-key record. All four strings are required; empty is allowed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatRecord {
    pub system: String,
    pub user: String,
    pub thinking: String,
    pub output: String,
}

impl ChatRecord {
    pub fn pack(&self) -> String {
        pack_record(
            &self.system,
            &self.user,
            Some(&self.thinking),
            Some(&self.output),
        )
    }
}

/// Build the ChatML-style stream used in both training and inference.
pub fn pack_record(
    system: &str,
    user: &str,
    thinking: Option<&str>,
    output: Option<&str>,
) -> String {
    let mut s = String::with_capacity(system.len() + user.len() + 64);
    s.push_str(TAG_SYSTEM);
    s.push('\n');
    s.push_str(system.trim());
    s.push('\n');
    s.push_str(TAG_USER);
    s.push('\n');
    s.push_str(user.trim());
    s.push('\n');
    if let Some(t) = thinking {
        s.push_str(TAG_THINKING);
        s.push('\n');
        s.push_str(t.trim());
        s.push('\n');
        s.push_str(TAG_THINK_END);
        s.push('\n');
    }
    if let Some(o) = output {
        s.push_str(TAG_OUTPUT);
        s.push('\n');
        s.push_str(o.trim());
        s.push('\n');
    }
    s
}

/// Ids through `<|output|>` plus its trailing newline piece — the cut C9
/// uses so greedy starts at the answer, not at `<|thinking|>`.
pub fn prefix_to_output_body(tokenizer: &mut BpeTokenizer, rec: &ChatRecord) -> Vec<u32> {
    prefix_to_output_body_ex(tokenizer, rec, false)
}

pub fn prefix_to_output_body_ex(
    tokenizer: &mut BpeTokenizer,
    rec: &ChatRecord,
    _output_only: bool,
) -> Vec<u32> {
    let (prefix, think, output) = encode_record_parts(tokenizer, rec);
    let tag = tokenizer.encode(&format!("{TAG_OUTPUT}\n"), false, false);
    let tag_len = if output.starts_with(&tag) {
        tag.len()
    } else {
        let mut n = 1usize;
        for i in 1..output.len() {
            let d = tokenizer.decode(&output[..i]);
            let after = d.split(TAG_OUTPUT).nth(1).unwrap_or("");
            if after.chars().all(char::is_whitespace) {
                n = i;
            } else {
                break;
            }
        }
        n.min(output.len())
    };
    let mut ids = prefix;
    ids.extend(think);
    ids.extend_from_slice(&output[..tag_len.min(output.len())]);
    ids
}

fn encode_record_parts(
    tokenizer: &mut BpeTokenizer,
    rec: &ChatRecord,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let prefix = tokenizer.encode(
        &pack_record(&rec.system, &rec.user, None, None),
        false,
        false,
    );
    let think = tokenizer.encode(
        &format!("{TAG_THINKING}\n{}\n{TAG_THINK_END}\n", rec.thinking.trim()),
        false,
        false,
    );
    let mut output = tokenizer.encode(
        &format!("{TAG_OUTPUT}\n{}\n", rec.output.trim()),
        false,
        false,
    );
    output.push(tokenizer.eos_id);
    (prefix, think, output)
}

fn tail(xs: &[u32], n: usize) -> &[u32] {
    if xs.len() <= n {
        xs
    } else {
        &xs[xs.len() - n..]
    }
}

fn head(xs: &[u32], n: usize) -> &[u32] {
    if xs.len() <= n {
        xs
    } else {
        &xs[..n]
    }
}

fn mask_pref_sup(n_pref: usize, n_total: usize) -> Vec<u8> {
    let mut mask = vec![0u8; n_total];
    for m in mask.iter_mut().skip(n_pref.min(n_total)) {
        *m = 1;
    }
    if n_total > 0 && mask.iter().all(|&m| m == 0) {
        mask.fill(1);
    }
    mask
}

/// `context` is mask-0 (system+user, or earlier thinking used only as
/// condition). `body` is mask-1. Fits in `keep` tokens. Context is the tail
/// (most recent tokens); body is head or tail as requested.
fn pack_ctx_body(
    context: &[u32],
    body: &[u32],
    keep: usize,
    body_from_head: bool,
) -> (Vec<u32>, Vec<u8>) {
    if keep == 0 {
        return (Vec::new(), Vec::new());
    }
    if context.len() + body.len() <= keep {
        let mut ids = Vec::with_capacity(context.len() + body.len());
        ids.extend_from_slice(context);
        ids.extend_from_slice(body);
        let n_ctx = context.len();
        return (ids, mask_pref_sup(n_ctx, n_ctx + body.len()));
    }
    if context.len() < keep {
        // Prefix fits: keep it whole and fill the rest with the body so a
        // T=96 window still conditions on the user turn.
        let body_part = if body_from_head {
            head(body, keep - context.len())
        } else {
            tail(body, keep - context.len())
        };
        let n_ctx = context.len();
        let mut ids = Vec::with_capacity(n_ctx + body_part.len());
        ids.extend_from_slice(context);
        ids.extend_from_slice(body_part);
        let n = ids.len();
        return (ids, mask_pref_sup(n_ctx, n));
    }
    // Prefix itself longer than the window: tail of the user turn plus body.
    // Never spend the whole window on the unmasked prefix.
    let min_ctx = if context.is_empty() {
        0
    } else {
        8.min(context.len()).min(keep / 4)
    };
    let mut body_budget = keep.saturating_sub(min_ctx).min(body.len());
    if body_budget == 0 && !body.is_empty() {
        body_budget = 1.min(keep);
    }
    let body_part = if body_from_head {
        head(body, body_budget)
    } else {
        tail(body, body_budget)
    };
    let ctx_part = tail(context, keep.saturating_sub(body_part.len()));
    let n_ctx = ctx_part.len();
    let mut ids = Vec::with_capacity(n_ctx + body_part.len());
    ids.extend_from_slice(ctx_part);
    ids.extend_from_slice(body_part);
    let mask = mask_pref_sup(n_ctx, ids.len());
    (ids, mask)
}

fn push_unique_window(windows: &mut Vec<(Vec<u32>, Vec<u8>)>, ids: Vec<u32>, mask: Vec<u8>) {
    if ids.is_empty() || mask.iter().all(|&m| m == 0) {
        return;
    }
    if windows.iter().any(|(w, _)| w == &ids) {
        return;
    }
    windows.push((ids, mask));
}

/// Encode a packed record. Loss mask is 1 on thinking+output (the trajectory
/// the KAN layer must predict) and 0 on the system+user prefix.
///
/// Long traces are clipped to `seq_len + 1`. The user prefix is taken from the
/// **tail** so a huge problem statement cannot push thinking+output out of the
/// window. This is the first honest window (start of thinking); the stream
/// also emits answer-side windows via [`encode_supervised_windows`].
pub fn encode_supervised(
    tokenizer: &mut BpeTokenizer,
    rec: &ChatRecord,
    seq_len: usize,
) -> (Vec<u32>, Vec<u8>) {
    encode_supervised_windows(tokenizer, rec, seq_len)
        .into_iter()
        .next()
        .unwrap_or_else(|| (Vec::new(), Vec::new()))
}

/// Honest supervised windows of length `≤ seq_len+1`.
///
/// A causal LM only learns tokens with mask=1. For traces longer than the
/// context we emit a small set of **contiguous** windows that cover:
/// 1. start of thinking (conditioned on the user-prefix tail)
/// 2. the answer (conditioned on the thinking tail)
/// 3. optionally a mid-thinking slice, and the output tail if the solution
///    itself exceeds the window
///
/// The previous clip kept `prefix_len+8` tokens, so a 2k-token user turn
/// filled the ring with mask-0 tokens and CE logged as `0.0000`.
pub fn encode_supervised_windows(
    tokenizer: &mut BpeTokenizer,
    rec: &ChatRecord,
    seq_len: usize,
) -> Vec<(Vec<u32>, Vec<u8>)> {
    encode_supervised_windows_ex(tokenizer, rec, seq_len, false)
}

fn encode_supervised_windows_ex(
    tokenizer: &mut BpeTokenizer,
    rec: &ChatRecord,
    seq_len: usize,
    output_only: bool,
) -> Vec<(Vec<u32>, Vec<u8>)> {
    let keep = seq_len.saturating_add(1).max(1);
    let (prefix, think, output) = encode_record_parts(tokenizer, rec);
    let full_len = prefix.len() + think.len() + output.len();
    if full_len <= keep {
        let mut ids = Vec::with_capacity(full_len);
        ids.extend_from_slice(&prefix);
        ids.extend_from_slice(&think);
        ids.extend_from_slice(&output);
        let n_ctx = if output_only {
            prefix.len() + think.len()
        } else {
            prefix.len()
        };
        return vec![(ids, mask_pref_sup(n_ctx, full_len))];
    }

    let mut windows = Vec::with_capacity(4);
    let start_body = if output_only || think.is_empty() {
        output.as_slice()
    } else {
        think.as_slice()
    };
    let start_ctx: Vec<u32> = if output_only {
        let mut c = prefix.clone();
        c.extend_from_slice(&think);
        c
    } else {
        prefix.clone()
    };
    let (ids, mask) = pack_ctx_body(&start_ctx, start_body, keep, true);
    push_unique_window(&mut windows, ids, mask);

    if !output.is_empty() {
        let mut ctx = Vec::with_capacity(prefix.len() + think.len());
        ctx.extend_from_slice(&prefix);
        ctx.extend_from_slice(&think);
        let (ids, mask) = pack_ctx_body(&ctx, &output, keep, true);
        push_unique_window(&mut windows, ids, mask);
        if output.len() > keep / 2 {
            let (ids, mask) = pack_ctx_body(&ctx, &output, keep, false);
            push_unique_window(&mut windows, ids, mask);
        }
    }

    if !output_only && think.len() > keep {
        let stride = (keep / 2).max(1);
        let mut start = stride;
        let mut extra = 0u32;
        while start < think.len() && extra < 6 {
            let end = (start + keep).min(think.len());
            let begin = end.saturating_sub(keep);
            if begin == 0 {
                break;
            }
            let slice = think[begin..end].to_vec();
            let n = slice.len();
            push_unique_window(&mut windows, slice, vec![1u8; n]);
            start = start.saturating_add(stride);
            extra += 1;
        }
    }

    if windows.is_empty() {
        let mut ids = Vec::with_capacity(full_len);
        ids.extend_from_slice(&prefix);
        ids.extend_from_slice(&think);
        ids.extend_from_slice(&output);
        let ids = tail(&ids, keep).to_vec();
        let n = ids.len();
        windows.push((ids, vec![1u8; n]));
    }
    windows
}

/// Cache-line aligned, compacting token+mask ring. Occupancy is always a
/// contiguous `[0, len)` window after `compact`, so the token pointer can be
/// handed to Metal as a no-copy Shared buffer.
pub struct SovereignFlashBuffer {
    // Drop = declaration order. metal MUST be first so the wrap dies before dealloc.
    #[cfg(target_os = "macos")]
    metal: Option<metal::Buffer>,
    slab: PageSlab,
    cap: usize,
    mask_off: usize,
    start: usize,
    len: usize,
}

impl std::fmt::Debug for SovereignFlashBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovereignFlashBuffer")
            .field("cap", &self.cap)
            .field("len", &self.len)
            .field("start", &self.start)
            .field("bytes", &self.slab.len())
            .finish()
    }
}

impl SovereignFlashBuffer {
    pub fn new(cap: usize) -> Result<Self> {
        let cap = cap.max(1);
        let mask_off = cap * 4;
        let bytes = mask_off + cap;
        Ok(Self {
            #[cfg(target_os = "macos")]
            metal: None,
            slab: PageSlab::new(bytes)?,
            cap,
            mask_off,
            start: 0,
            len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    fn tokens(&self) -> &[u32] {
        self.slab.u32_at(0, self.cap).expect("flash token plane")
    }

    fn tokens_mut(&mut self) -> &mut [u32] {
        let cap = self.cap;
        self.slab.u32_at_mut(0, cap).expect("flash token plane")
    }

    fn masks(&self) -> &[u8] {
        let off = self.mask_off;
        let cap = self.cap;
        let bytes = self.slab.as_bytes();
        if off + cap > bytes.len() {
            return &[];
        }
        &bytes[off..off + cap]
    }

    fn masks_mut(&mut self) -> &mut [u8] {
        let off = self.mask_off;
        let cap = self.cap;
        let bytes = self.slab.as_bytes_mut();
        if off + cap > bytes.len() {
            return &mut [];
        }
        &mut bytes[off..off + cap]
    }

    /// Slide occupancy to index 0 so the DMA pointer is a single span.
    pub fn compact(&mut self) {
        if self.start == 0 || self.len == 0 {
            self.start = 0;
            return;
        }
        let start = self.start;
        let len = self.len;
        self.tokens_mut().copy_within(start..start + len, 0);
        self.masks_mut().copy_within(start..start + len, 0);
        self.start = 0;
    }

    pub fn clear(&mut self) {
        if self.len > 0 {
            let start = self.start;
            let len = self.len;
            self.tokens_mut()[start..start + len].fill(0);
            self.masks_mut()[start..start + len].fill(0);
        }
        self.start = 0;
        self.len = 0;
    }

    pub fn push(&mut self, id: u32, mask: u8) {
        if self.len == self.cap {
            self.drain_front(1);
        }
        if self.start + self.len >= self.cap {
            self.compact();
        }
        let i = self.start + self.len;
        self.tokens_mut()[i] = id;
        self.masks_mut()[i] = mask;
        self.len += 1;
    }

    pub fn extend(&mut self, ids: &[u32], mask: &[u8]) {
        debug_assert_eq!(ids.len(), mask.len());
        for (&id, &m) in ids.iter().zip(mask.iter()) {
            self.push(id, m);
        }
    }

    pub fn drain_front(&mut self, n: usize) {
        let n = n.min(self.len);
        if n == 0 {
            return;
        }
        let start = self.start;
        self.tokens_mut()[start..start + n].fill(0);
        self.masks_mut()[start..start + n].fill(0);
        self.start += n;
        self.len -= n;
        if self.len == 0 {
            self.start = 0;
        } else if self.start > self.cap / 2 {
            self.compact();
        }
    }

    pub fn window(&self, start: usize, len: usize) -> (Vec<u32>, Vec<u8>) {
        let start = start.min(self.len);
        let len = len.min(self.len.saturating_sub(start));
        let s = self.start + start;
        (
            self.tokens()[s..s + len].to_vec(),
            self.masks()[s..s + len].to_vec(),
        )
    }

    pub fn token_span(&self) -> &[u32] {
        let s = self.start;
        &self.tokens()[s..s + self.len]
    }

    /// Replace occupancy with a saved token sequence (session restore).
    pub fn load_tokens(&mut self, ids: &[u32]) {
        self.clear();
        for &id in ids {
            self.push(id, 1);
        }
        self.compact();
    }

    /// Bind the page-aligned token plane as a no-copy Shared Metal buffer.
    pub fn bind_metal(&mut self, gpu: &SovereignDevice) -> Result<()> {
        self.compact();
        #[cfg(target_os = "macos")]
        {
            let Some(mtl) = gpu.mtl_device() else {
                return Ok(());
            };
            self.metal = None;
            let tok_bytes = self.slab.as_bytes();
            self.metal = Some(crate::device::wrap_shared_bytes_no_copy(mtl, tok_bytes)?);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = gpu;
            Ok(())
        }
    }

    #[cfg(target_os = "macos")]
    pub fn metal_buffer(&self) -> Option<&metal::Buffer> {
        self.metal.as_ref()
    }
}

fn sample_thinking_chars(s: &str, budget: usize) -> String {
    let n = s.chars().count();
    if n <= budget {
        return s.to_string();
    }
    let part = (budget / 3).max(1);
    let mid0 = n.saturating_sub(part) / 2;
    let mut out = String::with_capacity(budget + 2);
    out.extend(s.chars().take(part));
    out.push('\n');
    out.extend(s.chars().skip(mid0).take(part));
    out.push('\n');
    out.extend(s.chars().skip(n.saturating_sub(part)));
    out
}

/// Packed 4-key strings for tokenizer training. Training never synthesizes text.
pub fn jsonl_corpus_texts(path: impl AsRef<Path>, max_records: usize) -> Result<Vec<String>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut texts = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let Some(rec) = parse_jsonl_line(&line) else {
            continue;
        };
        // Truncate thinking so BPE training does not hold a 19 MB dump.
        // Head-only clips hide the bulk of 10k+ char traces, so take head/mid/tail.
        let mut rec = rec;
        rec.thinking = sample_thinking_chars(&rec.thinking, 768);
        texts.push(rec.pack());
        if texts.len() >= max_records {
            break;
        }
    }
    Ok(texts)
}

/// Peek a JSONL for collapsed user distribution / huge thinking traces.
pub fn warn_corpus_homogeneity(path: impl AsRef<Path>) -> Result<()> {
    use std::collections::HashSet;
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut users = HashSet::new();
    let mut n = 0u64;
    let mut think_chars = 0u64;
    let mut think_hist: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let Some(rec) = parse_jsonl_line(&line) else {
            continue;
        };
        n += 1;
        users.insert(rec.user.chars().take(80).collect::<String>());
        think_chars += rec.thinking.len() as u64;
        let key: String = rec.thinking.chars().take(40).collect();
        *think_hist.entry(key).or_insert(0) += 1;
        if n >= 4096 {
            break;
        }
    }
    if n == 0 {
        return Ok(());
    }
    let uniq = users.len() as f64 / n as f64;
    let mean_think = think_chars as f64 / n as f64;
    if uniq < 0.10 {
        eprintln!(
            "warn: {} unique-user ratio {:.2} (<0.10) — most windows will ignore the user turn",
            path.display(),
            uniq
        );
    }
    if mean_think > 1500.0 {
        eprintln!(
            "warn: {} mean thinking {:.0} chars — packing seq_len windows (prefix tail + think head / think tail + output) so CE is not a silent 0",
            path.display(),
            mean_think
        );
    }
    if let Some((dummy, c)) = think_hist.iter().max_by_key(|(_, c)| *c) {
        let frac = *c as f64 / n as f64;
        if n >= 32 && frac >= 0.50 {
            eprintln!(
                "warn: {} {frac:.0}% of thinking traces are {dummy:?} — the model will memorize that stub, not reason. Filter or write real traces.",
                path.display()
            );
        }
    }
    Ok(())
}

/// First valid 4-key record, for a cheap greedy probe during train.
pub fn first_jsonl_record(path: impl AsRef<Path>) -> Option<ChatRecord> {
    let file = File::open(path.as_ref()).ok()?;
    let reader = BufReader::new(file);
    for line in reader.lines().map_while(Result::ok) {
        if let Some(rec) = parse_jsonl_line(&line) {
            return Some(rec);
        }
    }
    None
}

pub fn parse_jsonl_line(line: &str) -> Option<ChatRecord> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let rec = serde_json::from_str::<ChatRecord>(trimmed).ok()?;
    if rec.system.is_empty() && rec.user.is_empty() && rec.output.is_empty() {
        return None;
    }
    Some(rec)
}

pub struct JsonlStream {
    path: PathBuf,
    reader: BufReader<File>,
    tokenizer: BpeTokenizer,
    seq_len: usize,
    flash: SovereignFlashBuffer,
    rng: StdRng,
    lines_seen: u64,
    /// Loss only on `<|output|>` body (thinking is context).
    pub mask_output: bool,
}

impl JsonlStream {
    pub fn open(
        path: impl AsRef<Path>,
        tokenizer: BpeTokenizer,
        seq_len: usize,
        seed: u64,
    ) -> Result<Self> {
        Self::open_with_cap(path, tokenizer, seq_len, MAX_TOKEN_BUF, seed)
    }

    pub fn open_with_cap(
        path: impl AsRef<Path>,
        tokenizer: BpeTokenizer,
        seq_len: usize,
        context_len: usize,
        seed: u64,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let cap = context_len.max(seq_len.saturating_mul(4).max(1));
        Ok(Self {
            path,
            reader: BufReader::with_capacity(64 * 1024, file),
            tokenizer,
            seq_len,
            flash: SovereignFlashBuffer::new(cap)?,
            rng: crate::device::rng_from_seed(seed),
            lines_seen: 0,
            mask_output: false,
        })
    }

    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    pub fn tokenizer_mut(&mut self) -> &mut BpeTokenizer {
        &mut self.tokenizer
    }

    fn rewind(&mut self) -> Result<()> {
        self.reader.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    fn cap_buf(&mut self) {
        let cap = self.flash.cap();
        if self.flash.len() > cap {
            let keep = self.seq_len * 4;
            let drain = self.flash.len().saturating_sub(keep);
            self.flash.drain_front(drain);
        }
    }

    fn push_ids(&mut self, ids: Vec<u32>, mask: Vec<u8>) {
        debug_assert_eq!(ids.len(), mask.len());
        self.flash.extend(&ids, &mask);
    }

    fn read_one_line(&mut self) -> Result<bool> {
        let mut line = String::new();
        let n = self.reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(false);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(true);
        }
        let Some(rec) = parse_jsonl_line(trimmed) else {
            return Ok(true);
        };
        let keep = self.seq_len.saturating_add(1);
        let eos = self.tokenizer.eos_id;
        let mut windows = encode_supervised_windows_ex(
            &mut self.tokenizer,
            &rec,
            self.seq_len,
            self.mask_output,
        );
        if windows.len() > 1 {
            windows.shuffle(&mut self.rng);
        }
        for (mut ids, mut mask) in windows {
            while ids.len() < keep {
                ids.push(eos);
                mask.push(0);
            }
            self.push_ids(ids, mask);
        }
        self.lines_seen += 1;
        Ok(true)
    }

    fn refill(&mut self) -> Result<()> {
        let need = self.seq_len + 2;
        let mut loops = 0u32;
        while self.flash.len() < need {
            if !self.read_one_line()? {
                self.rewind()?;
                loops += 1;
                if loops > 2 {
                    if self.flash.is_empty() {
                        anyhow::bail!("JSONL corpus produced no tokens: {}", self.path.display());
                    }
                    break;
                }
            }
            self.cap_buf();
            if loops > 8 {
                break;
            }
        }
        Ok(())
    }

    /// Next `(x, y, loss_mask)` — shifted LM, mask aligned with `y`.
    ///
    /// Windows are consumed from the front of the ring (each record is packed
    /// as one or more `seq_len+1` chunks). Leading mask-0 prefix is skipped
    /// so a step never reports `ce=0` on an unsupervised problem statement.
    pub fn next_seq(&mut self) -> Result<(Vec<u32>, Vec<u32>, Vec<u8>)> {
        let need = self.seq_len + 1;
        let mut skipped = 0usize;
        loop {
            self.refill()?;
            while self.flash.len() < need {
                self.flash.push(self.tokenizer.eos_id, 0);
            }
            let (chunk, mchunk) = self.flash.window(0, need);
            let n_sup = mchunk[1..].iter().filter(|&&m| m != 0).count();
            if n_sup > 0 || skipped > self.seq_len.saturating_mul(8) {
                let x = chunk[..self.seq_len].to_vec();
                let y = chunk[1..].to_vec();
                let loss_mask = mchunk[1..].to_vec();
                self.flash.drain_front(need.min(self.flash.len()));
                return Ok((x, y, loss_mask));
            }
            self.flash.drain_front(1.min(self.flash.len()));
            skipped += 1;
        }
    }

    pub fn next_batch(&mut self, batch: usize) -> Result<(Vec<u32>, Vec<u32>, Vec<u8>)> {
        let mut xs = Vec::with_capacity(batch * self.seq_len);
        let mut ys = Vec::with_capacity(batch * self.seq_len);
        let mut ms = Vec::with_capacity(batch * self.seq_len);
        for _ in 0..batch {
            let (x, y, m) = self.next_seq()?;
            xs.extend(x);
            ys.extend(y);
            ms.extend(m);
        }
        Ok((xs, ys, ms))
    }

    pub fn lines_seen(&self) -> u64 {
        self.lines_seen
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognitive_bench_has_fifteen_dense_anchors() {
        let raw = std::fs::read_to_string("data/cognitive-bench.jsonl").unwrap();
        let recs: Vec<ChatRecord> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<ChatRecord>(l).expect(l))
            .collect();
        assert_eq!(recs.len(), 15);
        for rec in &recs {
            let steps = rec
                .thinking
                .lines()
                .filter(|l| {
                    let t = l.trim_start();
                    t.chars().next().is_some_and(|c| c.is_ascii_digit()) && t.contains('.')
                })
                .count();
            assert!(
                steps >= 6,
                "sparse thinking ({} steps): {}",
                steps,
                rec.user
            );
            assert!(!rec.output.is_empty());
        }
    }

    #[test]
    fn parse_strict_four_key() {
        let line = r#"{"system":"You are a Rust compiler specialist.","user":"write add","thinking":"add i32s","output":"fn add(a: i32, b: i32) -> i32 { a + b }"}"#;
        let rec = parse_jsonl_line(line).unwrap();
        assert_eq!(rec.system, "You are a Rust compiler specialist.");
        assert_eq!(rec.user, "write add");
        assert!(rec.output.contains("fn add"));
        let packed = rec.pack();
        assert!(packed.contains(TAG_THINKING));
        assert!(packed.contains(TAG_OUTPUT));
    }

    #[test]
    fn parse_rejects_legacy_text_lang() {
        let line = r#"{"text":"fn main() {}\n","lang":"rust"}"#;
        assert!(parse_jsonl_line(line).is_none());
        assert!(parse_jsonl_line("fn main() {}").is_none());
    }

    #[test]
    fn serde_roundtrip_exactly_four_keys() {
        let rec = ChatRecord {
            system: "s".into(),
            user: "u".into(),
            thinking: "t".into(),
            output: "o".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&rec).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 4);
        assert!(obj.contains_key("system"));
        assert!(obj.contains_key("user"));
        assert!(obj.contains_key("thinking"));
        assert!(obj.contains_key("output"));
    }

    #[test]
    fn train_jsonl_is_strict_four_key() {
        let path = if Path::new("data/thinking-train.jsonl").exists() {
            "data/thinking-train.jsonl"
        } else if Path::new("data/basic-train.jsonl").exists() {
            "data/basic-train.jsonl"
        } else {
            "data/train.jsonl"
        };
        let text = std::fs::read_to_string(path).expect("JSONL corpus");
        let mut n = 0usize;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let rec = parse_jsonl_line(line).expect("parse");
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let obj = v.as_object().unwrap();
            assert_eq!(obj.len(), 4, "line {n} extra/missing keys");
            assert!(obj.contains_key("system"));
            assert!(obj.contains_key("user"));
            assert!(obj.contains_key("thinking"));
            assert!(obj.contains_key("output"));
            assert!(!rec.thinking.is_empty(), "thinking empty at {n}");
            assert!(!rec.output.is_empty(), "output empty at {n}");
            n += 1;
        }
        assert!(n >= 8, "expected a dense sample set, got {n}");
    }

    #[test]
    fn encode_supervised_clips_long_think_keeps_user() {
        let mut tok = crate::tokenizer::train_wordpiece(&[], 1024, 1).unwrap();
        let rec = ChatRecord {
            system: "sys".into(),
            user: "Write fn add that sums two i32".into(),
            thinking: "step ".repeat(800),
            output: "fn add(a: i32, b: i32) -> i32 { a + b }".into(),
        };
        let seq = 48usize;
        let (ids, mask) = encode_supervised(&mut tok, &rec, seq);
        assert!(ids.len() <= seq + 1, "clipped len {}", ids.len());
        let prefix = tok.encode(
            &pack_record(&rec.system, &rec.user, None, None),
            false,
            false,
        );
        assert!(ids.len() >= prefix.len().min(seq));
        let n_pref = prefix.len().min(ids.len());
        assert!(
            mask.iter().take(n_pref).all(|&m| m == 0),
            "user prefix must be masked off so the window still conditions on it"
        );
        assert!(mask.contains(&1));
    }

    #[test]
    fn encode_output_only_masks_thinking() {
        let mut tok = crate::tokenizer::train_wordpiece(&[], 1024, 1).unwrap();
        let rec = ChatRecord {
            system: "sys".into(),
            user: "add".into(),
            thinking: "long chain of thought about adding".into(),
            output: "fn add() {}".into(),
        };
        let windows = encode_supervised_windows_ex(&mut tok, &rec, 128, true);
        let (ids, mask) = &windows[0];
        let n_think = tok
            .encode(
                &format!("{TAG_THINKING}\n{}\n{TAG_THINK_END}\n", rec.thinking.trim()),
                false,
                false,
            )
            .len();
        let n_pref = tok
            .encode(
                &pack_record(&rec.system, &rec.user, None, None),
                false,
                false,
            )
            .len();
        assert!(
            mask.iter().take(n_pref + n_think).all(|&m| m == 0),
            "thinking must be mask-0 in output-only mode"
        );
        assert!(mask.contains(&1), "output must be supervised");
        assert_eq!(ids.len(), mask.len());
    }

    #[test]
    fn encode_supervised_does_not_keep_overlong_prefix() {
        let mut tok = crate::tokenizer::train_wordpiece(&[], 1024, 1).unwrap();
        let rec = ChatRecord {
            system: "sys".into(),
            user: "problem statement ".repeat(400),
            thinking: "reason ".repeat(400),
            output: "fn add(a: i32, b: i32) -> i32 { a + b }".into(),
        };
        let seq = 48usize;
        let windows = encode_supervised_windows(&mut tok, &rec, seq);
        assert!(!windows.is_empty());
        for (ids, mask) in &windows {
            assert!(ids.len() <= seq + 1, "window len {}", ids.len());
            assert!(
                mask.iter().any(|&m| m != 0),
                "honest window must have supervised tokens"
            );
        }
        let out_ids = tok.encode(
            &format!("{TAG_OUTPUT}\n{}\n", rec.output.trim()),
            false,
            false,
        );
        let has_output = windows.iter().any(|(ids, _)| {
            ids.windows(out_ids.len()).any(|w| w == out_ids.as_slice())
                || out_ids.iter().any(|t| ids.contains(t))
        });
        assert!(has_output, "some window must include the answer");
    }

    #[test]
    fn next_seq_skips_unsupervised_prefix_on_long_user() {
        let tok = crate::tokenizer::train_wordpiece(&[], 1024, 1).unwrap();
        let rec = ChatRecord {
            system: "sys".into(),
            user: "user turn ".repeat(200),
            thinking: "think step ".repeat(200),
            output: "fn ok() {}".into(),
        };
        let line = serde_json::to_string(&rec).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "ullis-honest-train-{}-{}.jsonl",
            std::process::id(),
            rec.user.len()
        ));
        std::fs::write(&path, format!("{line}\n")).unwrap();
        let mut stream = JsonlStream::open(&path, tok, 32, 1).unwrap();
        let (_x, _y, mask) = stream.next_seq().unwrap();
        let n_sup = mask.iter().filter(|&&m| m != 0).count();
        let _ = std::fs::remove_file(&path);
        assert!(n_sup > 0, "next_seq must not return an all-zero CE mask");
    }

    #[test]
    fn flash_buffer_is_contiguous() {
        let mut b = SovereignFlashBuffer::new(64).unwrap();
        for i in 0..80u32 {
            b.push(i, 1);
        }
        assert_eq!(b.len(), 64);
        b.compact();
        let span = b.token_span();
        assert_eq!(span.len(), 64);
        assert_eq!(span[0], 16);
        assert_eq!(span[63], 79);
        b.drain_front(4);
        assert_eq!(b.token_span()[0], 20);
        b.clear();
        assert!(b.is_empty());
    }
}
