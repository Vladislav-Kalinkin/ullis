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

/// Encode a packed record. Loss mask is 1 on thinking+output (the trajectory
/// the KAN layer must predict) and 0 on the system+user prefix.
///
/// Long traces are clipped to `seq_len + 1` keeping the user→thinking boundary
/// (head of thinking + output), so a `T=96` window still sees the user turn.
pub fn encode_supervised(
    tokenizer: &mut BpeTokenizer,
    rec: &ChatRecord,
    seq_len: usize,
) -> (Vec<u32>, Vec<u8>) {
    let think_span = format!("{TAG_THINKING}\n{}\n{TAG_THINK_END}\n", rec.thinking.trim());
    let out_span = format!("{TAG_OUTPUT}\n{}\n", rec.output.trim());

    let mut ids = tokenizer.encode(
        &pack_record(&rec.system, &rec.user, None, None),
        false,
        false,
    );
    let prefix_len = ids.len();
    ids.extend(tokenizer.encode(&think_span, false, false));
    ids.extend(tokenizer.encode(&out_span, false, false));
    ids.push(tokenizer.eos_id);
    let keep = seq_len.saturating_add(1).max(prefix_len.saturating_add(8));
    if ids.len() > keep {
        let mut clipped = ids[..prefix_len.min(keep)].to_vec();
        let budget = keep.saturating_sub(clipped.len());
        let rest = &ids[prefix_len..];
        if rest.len() > budget {
            clipped.extend_from_slice(&rest[..budget]);
        } else {
            clipped.extend_from_slice(rest);
        }
        ids = clipped;
    }
    let mut mask = vec![0u8; ids.len()];
    for m in mask.iter_mut().skip(prefix_len.min(ids.len())) {
        *m = 1;
    }
    if mask.iter().all(|&m| m == 0) {
        mask.fill(1);
    }
    (ids, mask)
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
        let think = rec.thinking.chars().take(512).collect::<String>();
        texts.push(format!("{}\n{think}\n{}", rec.user, rec.output));
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
    for line in reader.lines() {
        let line = line?;
        let Some(rec) = parse_jsonl_line(&line) else {
            continue;
        };
        n += 1;
        users.insert(rec.user.chars().take(80).collect::<String>());
        think_chars += rec.thinking.len() as u64;
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
            "warn: {} mean thinking {:.0} chars — records are clipped to seq_len so the user prefix stays in-window",
            path.display(),
            mean_think
        );
    }
    Ok(())
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
        let (ids, mask) = encode_supervised(&mut self.tokenizer, &rec, self.seq_len);
        self.push_ids(ids, mask);
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
    pub fn next_seq(&mut self) -> Result<(Vec<u32>, Vec<u32>, Vec<u8>)> {
        self.refill()?;
        while self.flash.len() < self.seq_len + 1 {
            self.flash.push(self.tokenizer.eos_id, 0);
        }
        let max_start = self.flash.len() - self.seq_len - 1;
        // Prefer the left of the buffer (user→thinking boundary after clip).
        let start = if max_start == 0 {
            0
        } else if self.rng.random::<f32>() < 0.55 {
            0
        } else if self.rng.random::<f32>() < 0.5 {
            max_start
        } else {
            self.rng.random_range(0..=max_start)
        };
        let (chunk, mchunk) = self.flash.window(start, self.seq_len + 1);
        let x = chunk[..self.seq_len].to_vec();
        let y = chunk[1..].to_vec();
        let loss_mask = mchunk[1..].to_vec();
        if self.rng.random::<f32>() < 0.05 {
            let drain = (start + 16).min(self.flash.len());
            self.flash.drain_front(drain);
        }
        Ok((x, y, loss_mask))
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
