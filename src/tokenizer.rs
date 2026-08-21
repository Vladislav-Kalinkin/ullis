//! Byte-fallback WordPiece (language-agnostic). No tiktoken / HuggingFace.
//!
//! Layout:
//! - ids `0..3`     specials `<pad> <bos> <eos> <unk>`
//! - ids `4..259`   raw UTF-8 bytes (`BYTE_OFFSET + b`)
//! - ids `260..V-1` WordPiece atoms (whole words / syntax) then trained pieces
//!
//! Encode is greedy longest-match over the piece table. Unmapped UTF-8, including
//! incomplete multi-byte sequences, falls back to raw byte ids in-stream and
//! never panics.
//!
//! Production scale is `V ≥ 8192` ([`MIN_VOCAB`]), selectable at runtime via
//! `--vocab-size` up to [`MAX_VOCAB`] (1 048 576). Tables below [`MIN_VOCAB`] are rejected.
//! Empty tail ids (when pair-merges exhaust before `V`) occupy no piece bytes and
//! decode as empty, so the lexicon can grow without a dense rewrite.

use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const PAD: &str = "<pad>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const UNK: &str = "<unk>";
pub const N_SPECIAL: u32 = 4;
/// ids `4..259` are raw UTF-8 bytes. Byte *values* are still `0..=255`.
pub const BYTE_OFFSET: u32 = N_SPECIAL;
/// Hard minimum production vocabulary.
pub const MIN_VOCAB: u32 = 8192;
/// Default production scale (equals [`MIN_VOCAB`]).
pub const DEFAULT_VOCAB: u32 = MIN_VOCAB;
/// Absolute ceiling for `--vocab-size` (131 072+; Metal buffer / i8-block cap).
pub const MAX_VOCAB: u32 = 1_048_576;

/// Runtime `--vocab-size` gate. Unit tests may construct smaller tables via
/// [`BpeTokenizer::new`] so encode/decode math stays cheap.
pub fn validate_vocab_size(vocab_size: u32) -> Result<u32> {
    if vocab_size < MIN_VOCAB {
        bail!("vocab-size {vocab_size} is below the hard minimum {MIN_VOCAB}");
    }
    if vocab_size > MAX_VOCAB {
        bail!("vocab-size {vocab_size} exceeds the absolute ceiling {MAX_VOCAB}");
    }
    Ok(vocab_size)
}

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

/// High-frequency English / Russian function words and syntax identifiers.
/// Each occupies a single token id when `V` has room (WordPiece atoms).
pub const LANG_SEEDS: &[&str] = &[
    "the ",
    "of ",
    "and ",
    "to ",
    "in ",
    "is ",
    "you ",
    "that ",
    "it ",
    "he ",
    "was ",
    "for ",
    "on ",
    "are ",
    "as ",
    "with ",
    "his ",
    "they ",
    "at ",
    "be ",
    "this ",
    "have ",
    "from ",
    "or ",
    "one ",
    "had ",
    "by ",
    "but ",
    "not ",
    "what ",
    "all ",
    "were ",
    "we ",
    "when ",
    "your ",
    "can ",
    "said ",
    "there ",
    "use ",
    "an ",
    "each ",
    "which ",
    "she ",
    "do ",
    "how ",
    "their ",
    "if ",
    "will ",
    "up ",
    "other ",
    "about ",
    "out ",
    "many ",
    "then ",
    "them ",
    "these ",
    "so ",
    "some ",
    "her ",
    "would ",
    "make ",
    "like ",
    "him ",
    "into ",
    "time ",
    "has ",
    "look ",
    "two ",
    "more ",
    "write ",
    "go ",
    "see ",
    "no ",
    "way ",
    "could ",
    "people ",
    "my ",
    "than ",
    "first ",
    "been ",
    "who ",
    "its ",
    "now ",
    "find ",
    "long ",
    "down ",
    "day ",
    "did ",
    "get ",
    "come ",
    "made ",
    "may ",
    "part ",
    "function ",
    "class",
    "import",
    "struct",
    "enum",
    "trait",
    "const",
    "async",
    "await",
    "while",
    "break",
    "continue",
    "yield",
    "where",
    "type",
    "mod ",
    "true",
    "false",
    "null",
    "error",
    "result",
    "option",
    "string",
    "vector",
    "list",
    "dict",
    "file",
    "open",
    "read",
    "write",
    "load",
    "save",
    "train",
    "model",
    "token",
    "layer",
    "weight",
    "tensor",
    "kernel",
    "buffer",
    "device",
    "metal",
    "rust",
    "python",
    "bash",
    "json",
    "data",
    "code",
    "main",
    "test",
    "assert",
    "debug",
    "info",
    "warn",
    "panic",
    "unwrap",
    "clone",
    "copy",
    "drop",
    "default",
    "spawn",
    "thread",
    "process",
    "config",
    "vocab",
    "embed",
    "logit",
    "softmax",
    "loss",
    "grad",
    "scale",
    "pack",
    "quant",
    "ternary",
    "spline",
    "knot",
    "grid",
    "expert",
    "router",
    "think",
    "output",
    "system",
    "user",
    "the",
    "and",
    "that",
    "with",
    "from",
    "this",
    "function",
    "return",
    "и ",
    "в ",
    "не ",
    "на ",
    "что ",
    "я ",
    "он ",
    "с ",
    "как ",
    "а ",
    "то ",
    "все ",
    "она ",
    "так ",
    "его ",
    "но ",
    "да ",
    "ты ",
    "к ",
    "у ",
    "же ",
    "вы ",
    "за ",
    "бы ",
    "по ",
    "только ",
    "мне ",
    "было ",
    "вот ",
    "от ",
    "меня ",
    "ещё ",
    "нет ",
    "о ",
    "из ",
    "ему ",
    "теперь ",
    "когда ",
    "даже ",
    "ну ",
    "вдруг ",
    "ли ",
    "если ",
    "уже ",
    "или ",
    "ни ",
    "быть ",
    "был ",
    "до ",
    "вас ",
    "опять ",
    "вам ",
    "ведь ",
    "там ",
    "потом ",
    "себя ",
    "ничего ",
    "ей ",
    "может ",
    "они ",
    "тут ",
    "где ",
    "есть ",
    "надо ",
    "для ",
    "мы ",
    "тебя ",
    "их ",
    "чем ",
    "была ",
    "сам ",
    "чтоб ",
    "без ",
    "будто ",
    "чего ",
    "раз ",
    "тоже ",
    "себе ",
    "под ",
    "будет ",
    "тогда ",
    "кто ",
    "этот ",
    "того ",
    "потому ",
    "этого ",
    "какой ",
    "совсем ",
    "ним ",
    "здесь ",
    "этом ",
    "один ",
    "почти ",
    "мой ",
    "тем ",
    "чтобы ",
    "кажется ",
    "сейчас ",
    "были ",
    "куда ",
    "зачем ",
    "сказать ",
    "никогда ",
    "можно ",
    "при ",
    "наконец ",
    "два ",
    "об ",
    "другой ",
    "хоть ",
    "после ",
    "над ",
    "больше ",
    "тот ",
    "через ",
    "эти ",
    "нас ",
    "про ",
    "всего ",
    "них ",
    "какая ",
    "много ",
    "разве ",
    "три ",
    "эту ",
    "моя ",
    "впрочем ",
    "хорошо ",
    "свою ",
    "этой ",
    "перед ",
    "иногда ",
    "лучше ",
    "чуть ",
    "том ",
    "нельзя ",
    "такой ",
    "им ",
    "более ",
    "всегда ",
    "конечно ",
    "всю ",
    "между ",
    "функция ",
    "вернуть ",
    "результат ",
    "модель ",
    "токен ",
    "слой ",
    "вес ",
    "ошибка ",
    "файл ",
    "данные ",
    "код ",
    "привет ",
    "мир ",
    "и",
    "в",
    "не",
    "на",
    "что",
    "я",
    "он",
    "как",
    "это",
    "для",
    "или",
    "если",
    "при",
    "self.",
    "Self::",
    "std::",
    "let mut ",
    "pub fn ",
    "pub struct ",
    "pub enum ",
    "pub trait ",
    "impl ",
    "async fn ",
    "-> Result",
    "unwrap()",
    "to_string()",
    "into()",
    "as_str()",
    "as_slice()",
    "to_vec()",
    "len()",
    "is_empty()",
    "push(",
    "pop()",
    "insert(",
    "remove(",
    "HashMap",
    "VecDeque",
    "Option<",
    "Result<",
    "String",
    "usize",
    "i32",
    "i64",
    "u32",
    "u64",
    "f32",
    "f64",
    "bool",
    "def ",
    "lambda ",
    "None",
    "True",
    "False",
    "print(",
    "len(",
    "range(",
    "enumerate(",
    "zip(",
    "open(",
    "read()",
    "write(",
    "os.path",
    "pathlib",
    "numpy",
    "torch",
    "return",
    "#!/bin/bash",
    "set -euo",
    "pipefail",
    "local ",
    "then",
    "fi",
    "done",
    "esac",
    "echo ",
    "export ",
    "source ",
    "[[ ",
    "]]",
];

static ATOMS_SORTED: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut v: Vec<&'static str> = CODE_SEEDS
        .iter()
        .chain(LANG_SEEDS.iter())
        .copied()
        .collect();
    v.sort_by_key(|s| std::cmp::Reverse(s.len()));
    v.dedup();
    v
});

pub fn byte_id(b: u8) -> u32 {
    BYTE_OFFSET + u32::from(b)
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
    #[serde(default)]
    pub atoms: Vec<String>,
    #[serde(default = "wordpiece_model")]
    pub model: String,
}

fn wordpiece_model() -> String {
    "wordpiece".into()
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
    pub atoms: Vec<String>,
    id_to_bytes: Vec<Vec<u8>>,
    piece_to_id: HashMap<Vec<u8>, u32>,
    max_piece_len: usize,
    encode_cache: HashMap<Vec<u8>, Vec<u32>>,
}

impl BpeTokenizer {
    pub fn new(
        vocab_size: u32,
        merges: Vec<(u32, u32)>,
        specials: Option<Vec<String>>,
    ) -> Result<Self> {
        Self::new_with_atoms(vocab_size, merges, Vec::new(), specials)
    }

    pub fn new_with_atoms(
        vocab_size: u32,
        merges: Vec<(u32, u32)>,
        atoms: Vec<String>,
        specials: Option<Vec<String>>,
    ) -> Result<Self> {
        if vocab_size < BYTE_OFFSET + 256 {
            bail!(
                "vocab_size must be >= {} (byte-fallback floor)",
                BYTE_OFFSET + 256
            );
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
            atoms,
            id_to_bytes: vec![Vec::new(); vocab_size as usize],
            piece_to_id: HashMap::new(),
            max_piece_len: 1,
            encode_cache: HashMap::new(),
        };
        tok.rebuild();
        Ok(tok)
    }

    /// Grow the id plane in place. Existing pieces stay put; new slots are empty
    /// until later merges or [`crate::quant::PackedI8Matrix`] block allocation.
    pub fn expand_to(&mut self, vocab_size: u32) -> Result<()> {
        if vocab_size < self.vocab_size {
            bail!(
                "cannot shrink tokenizer vocab {} -> {vocab_size}",
                self.vocab_size
            );
        }
        if vocab_size == self.vocab_size {
            return Ok(());
        }
        self.vocab_size = vocab_size;
        self.id_to_bytes.resize(vocab_size as usize, Vec::new());
        self.encode_cache.clear();
        Ok(())
    }

    pub fn populated(&self) -> u32 {
        let mut n = BYTE_OFFSET + 256;
        n += self.atoms.len() as u32;
        n += self.merges.len() as u32;
        n.min(self.vocab_size)
    }

    fn rebuild(&mut self) {
        self.piece_to_id.clear();
        self.encode_cache.clear();
        self.id_to_bytes = vec![Vec::new(); self.vocab_size as usize];
        self.max_piece_len = 1;
        for b in 0..=255u8 {
            let id = byte_id(b);
            let bytes = vec![b];
            self.id_to_bytes[id as usize] = bytes.clone();
            self.piece_to_id.insert(bytes, id);
        }
        let mut next_id = BYTE_OFFSET + 256;
        let mut kept_atoms = Vec::new();
        for atom in &self.atoms {
            if next_id >= self.vocab_size {
                break;
            }
            let bytes = atom.as_bytes().to_vec();
            if bytes.is_empty() {
                continue;
            }
            if self.piece_to_id.contains_key(&bytes) {
                continue;
            }
            self.id_to_bytes[next_id as usize] = bytes.clone();
            self.max_piece_len = self.max_piece_len.max(bytes.len());
            self.piece_to_id.insert(bytes, next_id);
            kept_atoms.push(atom.clone());
            next_id += 1;
        }
        self.atoms = kept_atoms;
        let mut kept = Vec::new();
        for &(left, right) in &self.merges {
            if next_id >= self.vocab_size {
                break;
            }
            let lu = left as usize;
            let ru = right as usize;
            if lu >= self.id_to_bytes.len() || ru >= self.id_to_bytes.len() {
                continue;
            }
            let mut bytes = self.id_to_bytes[lu].clone();
            bytes.extend_from_slice(&self.id_to_bytes[ru]);
            if !bytes.is_empty() {
                self.piece_to_id.entry(bytes.clone()).or_insert(next_id);
                self.max_piece_len = self.max_piece_len.max(bytes.len());
            }
            self.id_to_bytes[next_id as usize] = bytes;
            kept.push((left, right));
            next_id += 1;
        }
        self.merges = kept;
    }

    /// Greedy longest-match WordPiece. Unknown bytes become `byte_id(b)`.
    pub fn encode_bytes(&mut self, data: &[u8]) -> Vec<u32> {
        if let Some(cached) = self.encode_cache.get(data) {
            return cached.clone();
        }
        let ids = self.encode_bytes_uncached(data);
        if self.encode_cache.len() < 4096 {
            self.encode_cache.insert(data.to_vec(), ids.clone());
        }
        ids
    }

    fn encode_bytes_uncached(&self, data: &[u8]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(data.len());
        let mut i = 0;
        while i < data.len() {
            let max_l = self.max_piece_len.min(data.len() - i);
            let mut matched = false;
            for len in (1..=max_l).rev() {
                if let Some(&id) = self.piece_to_id.get(&data[i..i + len]) {
                    ids.push(id);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if !matched {
                ids.push(byte_id(data[i]));
                i += 1;
            }
        }
        ids
    }

    pub fn encode(&mut self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        ids.extend(self.encode_bytes(text.as_bytes()));
        if add_eos {
            ids.push(self.eos_id);
        }
        ids
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
            atoms: self.atoms.clone(),
            model: "wordpiece".into(),
        }
    }

    pub fn from_json(data: &TokenizerJson) -> Result<Self> {
        let merges = data.merges.iter().map(|p| (p[0], p[1])).collect();
        Self::new_with_atoms(
            data.vocab_size,
            merges,
            data.atoms.clone(),
            Some(data.specials.clone()),
        )
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
        train_wordpiece(&[], DEFAULT_VOCAB, 7)
    }
}

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

fn collect_atoms(vocab_size: u32) -> Vec<String> {
    let mut atoms = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let budget = vocab_size.saturating_sub(BYTE_OFFSET + 256) as usize;
    for s in ATOMS_SORTED.iter() {
        if atoms.len() >= budget {
            break;
        }
        let b = s.as_bytes();
        if b.is_empty() || !seen.insert(b.to_vec()) {
            continue;
        }
        atoms.push((*s).to_string());
    }
    atoms
}

/// Byte-fallback WordPiece trainer. Seeds occupy single ids; remaining slots
/// are filled by frequency pair-merges on unique pretokenized chunks.
///
/// The previous trainer rescanned the whole corpus (byte ids) on every merge.
/// At `V=65536` and a ~20 MB JSONL that is tens of thousands of O(N) passes
/// and looks like a hang after `metal_hello`. This path BPE-s unique
/// whitespace chunks with incremental pair counts.
pub fn train_wordpiece(texts: &[String], vocab_size: u32, seed: u64) -> Result<BpeTokenizer> {
    use std::collections::BinaryHeap;
    use std::io::Write;

    let _ = seed;
    let atoms = collect_atoms(vocab_size);
    let mut proto = BpeTokenizer::new_with_atoms(vocab_size, Vec::new(), atoms.clone(), None)?;
    let mut next_id = BYTE_OFFSET + 256 + proto.atoms.len() as u32;
    if next_id >= vocab_size {
        return BpeTokenizer::new_with_atoms(vocab_size, Vec::new(), atoms, None);
    }

    let mut chunk_freq: HashMap<Vec<u8>, u32> = HashMap::new();
    if texts.iter().all(|t| t.is_empty()) {
        *chunk_freq
            .entry(ATOMS_SORTED.join("").into_bytes())
            .or_insert(0) += 1;
        for atom in ATOMS_SORTED.iter() {
            *chunk_freq.entry(atom.as_bytes().to_vec()).or_insert(0) += 1;
        }
    } else {
        for t in texts {
            for chunk in t.split_inclusive(char::is_whitespace) {
                if chunk.is_empty() {
                    continue;
                }
                *chunk_freq.entry(chunk.as_bytes().to_vec()).or_insert(0) += 1;
            }
        }
    }

    let mut words: Vec<Vec<u32>> = Vec::with_capacity(chunk_freq.len());
    let mut freqs: Vec<u32> = Vec::with_capacity(chunk_freq.len());
    for (chunk, f) in chunk_freq {
        if f == 0 {
            continue;
        }
        words.push(proto.encode_bytes(&chunk));
        freqs.push(f);
    }

    let mut pair_count: HashMap<(u32, u32), u64> = HashMap::new();
    let mut pair_occ: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    let mut heap: BinaryHeap<(u64, u32, u32)> = BinaryHeap::new();
    for (wi, (sym, &freq)) in words.iter().zip(freqs.iter()).enumerate() {
        let f = u64::from(freq);
        for w in sym.windows(2) {
            let p = (w[0], w[1]);
            *pair_count.entry(p).or_insert(0) += f;
            pair_occ.entry(p).or_default().push(wi as u32);
        }
    }
    for (&(a, b), &c) in &pair_count {
        if c >= 2 {
            heap.push((c, a, b));
        }
    }

    let mut merges: Vec<(u32, u32)> = Vec::new();
    let start_id = next_id;
    let mut seen_words = vec![0u32; words.len()];
    let mut stamp = 1u32;
    while next_id < vocab_size {
        let Some((cnt, a, b)) = heap.pop() else {
            break;
        };
        if pair_count.get(&(a, b)).copied().unwrap_or(0) != cnt {
            continue;
        }
        if cnt < 2 {
            break;
        }
        let nid = next_id;
        merges.push((a, b));
        next_id += 1;

        let mut affected = pair_occ.remove(&(a, b)).unwrap_or_default();
        affected.sort_unstable();
        affected.dedup();
        stamp = stamp.wrapping_add(1);
        if stamp == 0 {
            seen_words.fill(0);
            stamp = 1;
        }
        for wi in affected {
            let i = wi as usize;
            if i >= words.len() || seen_words[i] == stamp {
                continue;
            }
            seen_words[i] = stamp;
            let old = &words[i];
            if !old.windows(2).any(|w| w[0] == a && w[1] == b) {
                continue;
            }
            let freq = u64::from(freqs[i]);
            for w in old.windows(2) {
                let p = (w[0], w[1]);
                if let Some(c) = pair_count.get_mut(&p) {
                    *c = c.saturating_sub(freq);
                }
            }
            let new_sym = apply_merge(old, a, b, nid);
            for w in new_sym.windows(2) {
                let p = (w[0], w[1]);
                let c = pair_count.entry(p).or_insert(0);
                *c = c.saturating_add(freq);
                pair_occ.entry(p).or_default().push(wi);
                if *c >= 2 {
                    heap.push((*c, p.0, p.1));
                }
            }
            words[i] = new_sym;
        }
        pair_count.remove(&(a, b));

        let done = next_id - start_id;
        if done == 1 || done % 2048 == 0 {
            println!(
                "  tokenizer merge {done} / {}  chunks={} heap={}",
                vocab_size.saturating_sub(start_id),
                words.len(),
                heap.len()
            );
            let _ = std::io::stdout().flush();
        }
    }

    BpeTokenizer::new_with_atoms(vocab_size, merges, atoms, None)
}

pub fn train_bpe(texts: &[String], vocab_size: u32, seed: u64) -> Result<BpeTokenizer> {
    train_wordpiece(texts, vocab_size, seed)
}

pub fn load_or_train(
    vocab_size: u32,
    texts: &[String],
    path: Option<&Path>,
    seed: u64,
) -> Result<BpeTokenizer> {
    let vocab_size = validate_vocab_size(vocab_size)?;
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = path {
        if !p.as_os_str().is_empty() {
            candidates.push(p.to_path_buf());
        }
    }
    candidates.push(Path::new("assets/tokenizer-8192.json").to_path_buf());
    candidates.push(Path::new("ullis/assets/tokenizer-8192.json").to_path_buf());
    candidates.push(Path::new("checkpoints/tokenizer.json").to_path_buf());
    for cand in candidates {
        if cand.exists() {
            let mut tok = BpeTokenizer::load(&cand)?;
            if tok.vocab_size == vocab_size {
                return Ok(tok);
            }
            if tok.vocab_size < vocab_size && tok.vocab_size >= MIN_VOCAB {
                tok.expand_to(vocab_size)?;
                return Ok(tok);
            }
        }
    }
    train_wordpiece(texts, vocab_size, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_vocab() {
        let tok = train_wordpiece(&[], DEFAULT_VOCAB, 7).unwrap();
        assert_eq!(tok.vocab_size, 8192);
        assert!(!tok.atoms.is_empty());
        assert!(tok.merges.len() + tok.atoms.len() >= 256);
    }

    #[test]
    fn roundtrip_ascii() {
        let mut tok = train_wordpiece(&[], 1024, 1).unwrap();
        let s = "def load(path):\n    return path\n";
        let ids = tok.encode(s, false, false);
        assert_eq!(tok.decode(&ids), s);
    }

    #[test]
    fn seeds_are_single_tokens() {
        let mut tok = train_wordpiece(&[], 2048, 1).unwrap();
        for atom in ["def ", "fn ", "return", "<|system|>", "the ", "и "] {
            let ids = tok.encode(atom, false, false);
            assert_eq!(ids.len(), 1, "{atom:?} -> {ids:?}");
        }
    }

    #[test]
    fn byte_fallback_never_panics() {
        let mut tok = train_wordpiece(&[], 320, 0).unwrap();
        let junk: Vec<u8> = (0u8..=255).chain([0xFF, 0xFE, 0x80, 0xC0, 0xC1]).collect();
        let ids = tok.encode_bytes(&junk);
        assert!(!ids.is_empty());
        for &id in &ids {
            assert!(id < tok.vocab_size);
        }
        let _ = tok.decode(&ids);
    }

    #[test]
    fn bilingual_roundtrip() {
        let mut tok = train_wordpiece(&[], 2048, 2).unwrap();
        let s = "Hello мир fn main()";
        let ids = tok.encode(s, false, false);
        assert_eq!(tok.decode(&ids), s);
    }

    #[test]
    fn rejects_below_min_vocab() {
        assert!(validate_vocab_size(4096).is_err());
        assert!(validate_vocab_size(MIN_VOCAB - 1).is_err());
        assert_eq!(validate_vocab_size(MIN_VOCAB).unwrap(), MIN_VOCAB);
        assert_eq!(validate_vocab_size(131_072).unwrap(), 131_072);
        assert!(validate_vocab_size(MAX_VOCAB + 1).is_err());
    }

    #[test]
    fn expand_keeps_pieces() {
        let mut tok = train_wordpiece(&[], 1024, 1).unwrap();
        let s = "fn main()";
        let ids = tok.encode(s, false, false);
        tok.expand_to(16_384).unwrap();
        assert_eq!(tok.vocab_size, 16_384);
        assert_eq!(tok.decode(&ids), s);
        assert_eq!(tok.encode(s, false, false), ids);
    }
}
