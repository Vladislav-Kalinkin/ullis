//! Byte-level BPE with code-seeded merges. No tiktoken / HuggingFace dependency.

use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

pub const PAD: &str = "<pad>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const UNK: &str = "<unk>";
pub const N_SPECIAL: u32 = 4;
pub const BYTE_OFFSET: u32 = N_SPECIAL; // ids 4..259 are raw UTF-8 bytes

pub const CODE_SEEDS: &[&str] = &[
    "def ",
    "fn ",
    "impl ",
    "impl",
    "match ",
    "match",
    "return ",
    "return",
    "class ",
    "import ",
    "from ",
    "pub ",
    "let ",
    "mut ",
    "const ",
    "async ",
    "await ",
    "struct ",
    "enum ",
    "trait ",
    "where ",
    "if ",
    "elif ",
    "else",
    "for ",
    "while ",
    "loop ",
    "try:",
    "except ",
    "raise ",
    "self",
    "Self",
    "True",
    "False",
    "None",
    "Ok(",
    "Err(",
    "print",
    "println",
    "    ",
    "\t",
    "        ",
    "->",
    "::",
    "=>",
    "()",
    "{}",
    "[]",
    "==",
    "!=",
    "&&",
    "||",
    "+=",
    "-=",
    "-> ",
    ":\n",
    "{\n",
    "\n    ",
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "use std::",
    "std::io::",
    "__name__",
    "Path",
    "<|system|>",
    "<|user|>",
    "<|thinking|>",
    "<|/thinking|>",
    "<|output|>",
];

static CODE_SPLIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|[ \t]+|\r?\n|."#,
    )
    .expect("code-split regex")
});

static SEEDS_SORTED: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut v = CODE_SEEDS.to_vec();
    v.sort_by_key(|s| std::cmp::Reverse(s.len()));
    v.dedup();
    v
});

pub fn byte_id(b: u8) -> u32 {
    BYTE_OFFSET + b as u32
}

pub fn apply_merge(seq: &[u32], left: u32, right: u32, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
            out.push(new_id);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenizerJson {
    pub vocab_size: u32,
    pub merges: Vec<[u32; 2]>,
    pub specials: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    pub vocab_size: u32,
    pub specials: Vec<String>,
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
    pub merges: Vec<(u32, u32)>,
    pair_to_id: HashMap<(u32, u32), u32>,
    pair_rank: HashMap<(u32, u32), u32>,
    id_to_bytes: Vec<Vec<u8>>,
    encode_cache: HashMap<Vec<u8>, Vec<u32>>,
}

impl BpeTokenizer {
    pub fn new(
        vocab_size: u32,
        merges: Vec<(u32, u32)>,
        specials: Option<Vec<String>>,
    ) -> Result<Self> {
        if vocab_size < BYTE_OFFSET + 256 {
            bail!("vocab_size must be >= {}", BYTE_OFFSET + 256);
        }
        let specials = specials.unwrap_or_else(|| {
            vec![
                PAD.to_string(),
                BOS.to_string(),
                EOS.to_string(),
                UNK.to_string(),
            ]
        });
        let mut tok = Self {
            vocab_size,
            specials,
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            unk_id: 3,
            merges,
            pair_to_id: HashMap::new(),
            pair_rank: HashMap::new(),
            id_to_bytes: vec![Vec::new(); vocab_size as usize],
            encode_cache: HashMap::new(),
        };
        tok.rebuild();
        Ok(tok)
    }

    fn rebuild(&mut self) {
        self.pair_to_id.clear();
        self.pair_rank.clear();
        self.encode_cache.clear();
        self.id_to_bytes = vec![Vec::new(); self.vocab_size as usize];
        for b in 0..=255u8 {
            self.id_to_bytes[byte_id(b) as usize] = vec![b];
        }
        let mut kept = Vec::new();
        for (rank, &(left, right)) in self.merges.iter().enumerate() {
            let next_id = BYTE_OFFSET + 256 + rank as u32;
            if next_id >= self.vocab_size {
                break;
            }
            self.pair_to_id.insert((left, right), next_id);
            self.pair_rank.insert((left, right), rank as u32);
            let mut bytes = self.id_to_bytes[left as usize].clone();
            bytes.extend_from_slice(&self.id_to_bytes[right as usize]);
            self.id_to_bytes[next_id as usize] = bytes;
            kept.push((left, right));
        }
        self.merges = kept;
    }

    pub fn encode_bytes(&mut self, data: &[u8]) -> Vec<u32> {
        if let Some(cached) = self.encode_cache.get(data) {
            return cached.clone();
        }
        let mut ids: Vec<u32> = data.iter().copied().map(byte_id).collect();
        while ids.len() > 1 {
            let mut min_rank: Option<u32> = None;
            let mut min_i = 0usize;
            for i in 0..ids.len() - 1 {
                if let Some(&r) = self.pair_rank.get(&(ids[i], ids[i + 1])) {
                    if min_rank.map(|m| r < m).unwrap_or(true) {
                        min_rank = Some(r);
                        min_i = i;
                    }
                }
            }
            let Some(_) = min_rank else { break };
            let nid = self.pair_to_id[&(ids[min_i], ids[min_i + 1])];
            ids.splice(min_i..min_i + 2, [nid]);
        }
        if self.encode_cache.len() < 8192 {
            self.encode_cache.insert(data.to_vec(), ids.clone());
        }
        ids
    }

    pub fn encode(&mut self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        ids.extend(self.encode_text(text));
        if add_eos {
            ids.push(self.eos_id);
        }
        ids
    }

    fn encode_text(&mut self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut i = 0;
        let n = text.len();
        while i < n {
            let mut matched = false;
            for seed in SEEDS_SORTED.iter() {
                if text[i..].starts_with(*seed) {
                    let ids = self.encode_bytes(seed.as_bytes());
                    out.extend(ids);
                    i += seed.len();
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
            if let Some(m) = CODE_SPLIT.find_at(text, i) {
                if m.start() == i {
                    let ids = self.encode_bytes(m.as_str().as_bytes());
                    out.extend(ids);
                    i = m.end();
                    continue;
                }
            }
            let ch = text[i..].chars().next().unwrap();
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            out.extend(self.encode_bytes(s.as_bytes()));
            i += ch.len_utf8();
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut buf = Vec::new();
        for &i in ids {
            if i >= self.vocab_size {
                continue;
            }
            if i == self.pad_id || i == self.bos_id || i == self.eos_id || i == self.unk_id {
                continue;
            }
            buf.extend_from_slice(&self.id_to_bytes[i as usize]);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    pub fn token_bytes(&self, token_id: u32) -> &[u8] {
        if token_id >= self.vocab_size {
            return b"";
        }
        &self.id_to_bytes[token_id as usize]
    }

    pub fn to_json(&self) -> TokenizerJson {
        TokenizerJson {
            vocab_size: self.vocab_size,
            merges: self
                .merges
                .iter()
                .map(|&(left, right)| [left, right])
                .collect(),
            specials: self.specials.clone(),
        }
    }

    pub fn from_json(data: &TokenizerJson) -> Result<Self> {
        let merges = data.merges.iter().map(|p| (p[0], p[1])).collect();
        Self::new(data.vocab_size, merges, Some(data.specials.clone()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string(&self.to_json())?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("read tokenizer {}", path.as_ref().display()))?;
        let data: TokenizerJson = serde_json::from_str(&raw)?;
        Self::from_json(&data)
    }

    pub fn load_default() -> Result<Self> {
        let data: TokenizerJson = serde_json::from_str(DEFAULT_TOKENIZER_JSON)?;
        Self::from_json(&data)
    }
}

pub const DEFAULT_TOKENIZER_JSON: &str = include_str!("../assets/tokenizer-4096.json");

pub struct StreamDecoder<'a> {
    tokenizer: &'a BpeTokenizer,
    buf: Vec<u8>,
}

impl<'a> StreamDecoder<'a> {
    pub fn new(tokenizer: &'a BpeTokenizer) -> Self {
        Self {
            tokenizer,
            buf: Vec::new(),
        }
    }

    pub fn push(&mut self, token_id: u32) -> String {
        if token_id == self.tokenizer.eos_id {
            return self.flush();
        }
        self.buf
            .extend_from_slice(self.tokenizer.token_bytes(token_id));
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                let out = s.to_string();
                self.buf.clear();
                out
            }
            Err(err) => {
                let valid = err.valid_up_to();
                if valid == 0 {
                    if err.error_len().is_some() {
                        // invalid sequence — skip one byte
                        let out = String::from_utf8_lossy(&self.buf[..1]).into_owned();
                        self.buf.drain(..1);
                        return out;
                    }
                    String::new()
                } else {
                    let out = String::from_utf8_lossy(&self.buf[..valid]).into_owned();
                    self.buf.drain(..valid);
                    out
                }
            }
        }
    }

    pub fn flush(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        out
    }
}

fn left_fold_seed(
    ids: &[u32],
    pair_to_id: &mut HashMap<(u32, u32), u32>,
    mut next_id: u32,
    vocab_size: u32,
) -> (Vec<(u32, u32)>, u32) {
    let mut merges = Vec::new();
    let mut seq = ids.to_vec();
    while seq.len() > 1 && next_id < vocab_size {
        let a = seq[0];
        let b = seq[1];
        let nid = if let Some(&id) = pair_to_id.get(&(a, b)) {
            id
        } else {
            pair_to_id.insert((a, b), next_id);
            merges.push((a, b));
            let id = next_id;
            next_id += 1;
            id
        };
        seq = std::iter::once(nid)
            .chain(seq.iter().copied().skip(2))
            .collect();
    }
    (merges, next_id)
}

pub fn train_bpe(texts: &[String], vocab_size: u32, seed: u64) -> Result<BpeTokenizer> {
    use rand::seq::SliceRandom;

    let mut rng = crate::device::rng_from_seed(seed);
    let mut corpus: Vec<&str> = texts
        .iter()
        .map(|s| s.as_str())
        .filter(|t| !t.is_empty())
        .collect();
    if corpus.is_empty() {
        corpus.push("def main():\n    return 0\n");
    }
    corpus.shuffle(&mut rng);

    let mut pair_to_id: HashMap<(u32, u32), u32> = HashMap::new();
    let mut merges: Vec<(u32, u32)> = Vec::new();
    let mut next_id = BYTE_OFFSET + 256;

    for s in CODE_SEEDS {
        let ids: Vec<u32> = s.bytes().map(byte_id).collect();
        let (new_merges, n) = left_fold_seed(&ids, &mut pair_to_id, next_id, vocab_size);
        merges.extend(new_merges);
        next_id = n;
        if next_id >= vocab_size {
            break;
        }
    }

    let proto = BpeTokenizer::new(vocab_size, merges.clone(), None)?;
    let mut seqs: Vec<Vec<u32>> = Vec::new();
    {
        let mut p = proto;
        for t in &corpus {
            seqs.push(p.encode_bytes(t.as_bytes()));
        }
    }

    while next_id < vocab_size {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for seq in &seqs {
            if seq.len() < 2 {
                continue;
            }
            for w in seq.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += 1;
            }
        }
        if counts.is_empty() {
            break;
        }
        let ((a, b), _) = counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .expect("non-empty");
        if let Some(&nid) = pair_to_id.get(&(a, b)) {
            seqs = seqs.iter().map(|s| apply_merge(s, a, b, nid)).collect();
            continue;
        }
        pair_to_id.insert((a, b), next_id);
        merges.push((a, b));
        seqs = seqs.iter().map(|s| apply_merge(s, a, b, next_id)).collect();
        next_id += 1;
    }

    BpeTokenizer::new(vocab_size, merges, None)
}

pub fn load_or_train(
    vocab_size: u32,
    texts: &[String],
    path: Option<&Path>,
    seed: u64,
) -> Result<BpeTokenizer> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = path {
        if !p.as_os_str().is_empty() {
            candidates.push(p.to_path_buf());
        }
    }
    candidates.push(Path::new("ullis/assets/tokenizer-4096.json").to_path_buf());
    candidates.push(Path::new("ullis-core/assets/tokenizer-4096.json").to_path_buf());
    for cand in candidates {
        if cand.exists() {
            let tok = BpeTokenizer::load(&cand)?;
            if tok.vocab_size == vocab_size {
                return Ok(tok);
            }
        }
    }
    if vocab_size == 4096 {
        if let Ok(tok) = BpeTokenizer::load_default() {
            return Ok(tok);
        }
    }
    train_bpe(texts, vocab_size, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_vocab() {
        let tok = BpeTokenizer::load_default().unwrap();
        assert_eq!(tok.vocab_size, 4096);
        assert_eq!(tok.merges.len(), 1772);
    }

    #[test]
    fn roundtrip_ascii() {
        let mut tok = BpeTokenizer::load_default().unwrap();
        let s = "def load(path):\n    return path\n";
        let ids = tok.encode(s, false, false);
        assert_eq!(tok.decode(&ids), s);
    }
}
