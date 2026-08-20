//! Streaming 4-key JSONL token pipeline. RAM stays O(seq_len), not O(corpus).
//!
//! Canonical line:
//! `{"system":"...","user":"...","thinking":"...","output":"..."}`
//!
//! Legacy `{"text":"...","lang":"..."}` and raw-text lines are lifted in-stream
//! so existing corpora keep training.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::tokenizer::BpeTokenizer;

/// Token ring cap (~128 KB of u32) — independent of file size and thinking depth.
const MAX_TOKEN_BUF: usize = 32_768;

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

#[derive(Debug, Deserialize)]
struct LegacyRecord {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    lang: Option<String>,
}

impl LegacyRecord {
    fn text(&self) -> Option<&str> {
        self.text
            .as_deref()
            .or(self.content.as_deref())
            .filter(|s| !s.is_empty())
    }
}

impl ChatRecord {
    pub fn system_for_lang(lang: &str) -> &'static str {
        match lang {
            "rust" => "You are a Rust compiler specialist.",
            "python" => "You are a Python interpreter specialist.",
            "bash" => "You are a POSIX shell specialist.",
            _ => "You are a compact ternary KAN code engine.",
        }
    }

    pub fn from_legacy(text: &str, lang: Option<&str>) -> Self {
        let lang = lang.unwrap_or_else(|| infer_lang(text));
        let system = Self::system_for_lang(lang).to_string();
        let thinking = format!(
            "1. Language signal is {lang}.\n2. Read the requested snippet.\n3. Emit complete, well-formed source that matches the surrounding tokens."
        );
        Self {
            system,
            user: "Write the following program.".into(),
            thinking,
            output: text.trim_end().to_string(),
        }
    }

    pub fn pack(&self) -> String {
        pack_record(
            &self.system,
            &self.user,
            Some(&self.thinking),
            Some(&self.output),
        )
    }
}

pub fn infer_lang(text: &str) -> &'static str {
    if text.contains("fn ") || text.contains("impl ") || text.contains("pub ") {
        "rust"
    } else if text.contains("#!/usr/bin/env bash") || text.contains("set -euo") {
        "bash"
    } else {
        "python"
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
pub fn encode_supervised(tokenizer: &mut BpeTokenizer, rec: &ChatRecord) -> (Vec<u32>, Vec<u8>) {
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
    let mut mask = vec![0u8; ids.len()];
    for m in mask.iter_mut().skip(prefix_len) {
        *m = 1;
    }
    if mask.iter().all(|&m| m == 0) {
        mask.fill(1);
    }
    (ids, mask)
}

pub fn parse_jsonl_line(line: &str) -> Option<ChatRecord> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(rec) = serde_json::from_str::<ChatRecord>(trimmed) {
        if rec.system.is_empty() && rec.user.is_empty() && rec.output.is_empty() {
            return None;
        }
        return Some(rec);
    }
    if let Ok(leg) = serde_json::from_str::<LegacyRecord>(trimmed) {
        if let Some(text) = leg.text() {
            return Some(ChatRecord::from_legacy(text, leg.lang.as_deref()));
        }
    }
    Some(ChatRecord::from_legacy(trimmed, None))
}

pub struct JsonlStream {
    path: PathBuf,
    reader: BufReader<File>,
    tokenizer: BpeTokenizer,
    seq_len: usize,
    buf: VecDeque<u32>,
    mask: VecDeque<u8>,
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
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        Ok(Self {
            path,
            reader: BufReader::with_capacity(64 * 1024, file),
            tokenizer,
            seq_len,
            buf: VecDeque::with_capacity(seq_len * 4),
            mask: VecDeque::with_capacity(seq_len * 4),
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
        if self.buf.len() > MAX_TOKEN_BUF {
            let keep = self.seq_len * 4;
            let drain = self.buf.len() - keep;
            self.buf.drain(..drain);
            if self.mask.len() >= drain {
                self.mask.drain(..drain);
            }
        }
    }

    fn push_ids(&mut self, ids: Vec<u32>, mask: Vec<u8>) {
        debug_assert_eq!(ids.len(), mask.len());
        self.buf.extend(ids);
        self.mask.extend(mask);
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
        let (ids, mask) = encode_supervised(&mut self.tokenizer, &rec);
        self.push_ids(ids, mask);
        self.lines_seen += 1;
        Ok(true)
    }

    fn refill(&mut self) -> Result<()> {
        let need = self.seq_len + 2;
        let mut loops = 0u32;
        while self.buf.len() < need {
            if !self.read_one_line()? {
                self.rewind()?;
                loops += 1;
                if loops > 2 && self.buf.len() < need {
                    let rec = ChatRecord::from_legacy("fn main() {}\n", Some("rust"));
                    let (ids, mask) = encode_supervised(&mut self.tokenizer, &rec);
                    self.push_ids(ids, mask);
                }
            }
            self.cap_buf();
            if loops > 8 {
                break;
            }
        }
        Ok(())
    }

    fn window(&self, start: usize, len: usize) -> (Vec<u32>, Vec<u8>) {
        let ids: Vec<u32> = self.buf.iter().skip(start).take(len).copied().collect();
        let mask: Vec<u8> = self.mask.iter().skip(start).take(len).copied().collect();
        (ids, mask)
    }

    /// Next `(x, y, loss_mask)` — shifted LM, mask aligned with `y`.
    pub fn next_seq(&mut self) -> Result<(Vec<u32>, Vec<u32>, Vec<u8>)> {
        self.refill()?;
        while self.buf.len() < self.seq_len + 1 {
            self.buf.push_back(self.tokenizer.eos_id);
            self.mask.push_back(0);
        }
        while self.mask.len() < self.buf.len() {
            self.mask.push_back(1);
        }
        let max_start = self.buf.len() - self.seq_len - 1;
        // Bias toward the tail so thinking+output stay in-window on long records.
        let start = if max_start == 0 {
            0
        } else if self.rng.random::<f32>() < 0.65 {
            max_start
        } else {
            self.rng.random_range(0..=max_start)
        };
        let (chunk, mchunk) = self.window(start, self.seq_len + 1);
        let x = chunk[..self.seq_len].to_vec();
        let y = chunk[1..].to_vec();
        let loss_mask = mchunk[1..].to_vec();
        if self.rng.random::<f32>() < 0.05 {
            let drain = (start + 16).min(self.buf.len());
            self.buf.drain(..drain);
            let md = drain.min(self.mask.len());
            self.mask.drain(..md);
        }
        Ok((x, y, loss_mask))
    }

    pub fn next_batch(
        &mut self,
        batch: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut xs = Vec::with_capacity(batch * self.seq_len);
        let mut ys = Vec::with_capacity(batch * self.seq_len);
        let mut ms = Vec::with_capacity(batch * self.seq_len);
        for _ in 0..batch {
            let (x, y, m) = self.next_seq()?;
            xs.extend(x);
            ys.extend(y);
            ms.extend(m.into_iter().map(u32::from));
        }
        let x = Tensor::from_vec(xs, (batch, self.seq_len), device)?;
        let y = Tensor::from_vec(ys, (batch, self.seq_len), device)?;
        let m = Tensor::from_vec(ms, (batch, self.seq_len), device)?;
        Ok((x, y, m))
    }

    pub fn lines_seen(&self) -> u64 {
        self.lines_seen
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Mean token NLL over positions where `mask != 0`. Stays on-device.
pub fn masked_cross_entropy(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (n, _v) = logits.dims2()?;
    let log_sm = candle_nn::ops::log_softmax(logits, D::Minus1)?;
    let idx = targets.to_dtype(DType::U32)?.reshape((n, 1))?;
    let picked = log_sm.gather(&idx, 1)?.reshape(n)?;
    let m = mask.to_dtype(DType::F32)?.reshape(n)?;
    let weighted = picked.neg()?.mul(&m)?;
    let den = m.sum_all()?.clamp(1.0f32, f32::MAX)?;
    Ok((weighted.sum_all()? / den)?)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn parse_legacy_lifts() {
        let line = r#"{"text":"fn main() {}\n","lang":"rust"}"#;
        let rec = parse_jsonl_line(line).unwrap();
        assert!(rec.system.contains("Rust"));
        assert_eq!(rec.output.trim(), "fn main() {}");
        assert!(!rec.thinking.is_empty());
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
        let path = "data/train.jsonl";
        let text = std::fs::read_to_string(path).expect("data/train.jsonl");
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
}
