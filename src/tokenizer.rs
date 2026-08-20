//! Byte-fallback WordPiece (language-agnostic). No tiktoken / HuggingFace.
//!
//! Layout:
//! - ids `0..3`     specials `<pad> <bos> <eos> <unk>`
//! - ids `4..259`   raw UTF-8 bytes (`BYTE_OFFSET + b`)
//! - ids `260..V-1` WordPiece atoms (whole words / syntax) then trained pieces
//!
//! Encode is greedy longest-match over the piece table. Unmapped UTF-8, including
//! incomplete multi-byte sequences, falls back to raw byte ids in-stream and
//! never panics. Default scale is `V = 8192`.

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
pub const DEFAULT_VOCAB: u32 = 8192;

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
            atoms,
            id_to_bytes: vec![Vec::new(); vocab_size as usize],
            piece_to_id: HashMap::new(),
            max_piece_len: 1,
            encode_cache: HashMap::new(),
        };
        tok.rebuild();
        Ok(tok)
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
        if self.encode_cache.len() < 8192 {
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
        let data: TokenizerJson = serde_json::from_str(DEFAULT_TOKENIZER_JSON)?;
        let tok = Self::from_json(&data)?;
        if tok.vocab_size == DEFAULT_VOCAB {
            return Ok(tok);
        }
        train_wordpiece(&[], DEFAULT_VOCAB, 7)
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
/// are filled by frequency pair-merges on the corpus.
pub fn train_wordpiece(texts: &[String], vocab_size: u32, seed: u64) -> Result<BpeTokenizer> {
    use rand::seq::SliceRandom;

    let mut rng = crate::device::rng_from_seed(seed);
    let atoms = collect_atoms(vocab_size);
    let mut corpus: Vec<String> = texts.iter().filter(|t| !t.is_empty()).cloned().collect();
    if corpus.is_empty() {
        corpus.push(ATOMS_SORTED.join(""));
        corpus.push(
            "The function returns a result from the model. Hello world.\n\
             Привет мир. Это тестовый текст для словаря. Функция возвращает результат.\n\
             def load(path):\n    return path\nfn main() {}\n"
                .into(),
        );
    }
    corpus.shuffle(&mut rng);

    let proto = BpeTokenizer::new_with_atoms(vocab_size, Vec::new(), atoms.clone(), None)?;
    let mut next_id = BYTE_OFFSET + 256 + proto.atoms.len() as u32;
    let mut pair_to_id: HashMap<(u32, u32), u32> = HashMap::new();
    let mut merges: Vec<(u32, u32)> = Vec::new();
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
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = path {
        if !p.as_os_str().is_empty() {
            candidates.push(p.to_path_buf());
        }
    }
    candidates.push(Path::new("assets/tokenizer-8192.json").to_path_buf());
    candidates.push(Path::new("ullis/assets/tokenizer-8192.json").to_path_buf());
    candidates.push(Path::new("assets/tokenizer-4096.json").to_path_buf());
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
    if vocab_size == DEFAULT_VOCAB {
        if let Ok(tok) = train_wordpiece(texts, vocab_size, seed) {
            return Ok(tok);
        }
    }
    if vocab_size == 4096 {
        if let Ok(tok) = BpeTokenizer::load_default() {
            if tok.vocab_size == 4096 {
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
}
