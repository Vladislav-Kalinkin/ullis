//! Dataset-trained byte-level BPE (language-agnostic). No tiktoken / HuggingFace.
//!
//! Layout:
//! - ids `0..3`     specials `<pad> <bos> <eos> <unk>`
//! - ids `4..259`   raw UTF-8 bytes (`BYTE_OFFSET + b`)
//! - ids `260..V-1` merges learned only from the supplied corpus
//!
//! Encode is greedy longest-match over the learned piece table. Unmapped UTF-8, including
//! incomplete multi-byte sequences, falls back to raw byte ids in-stream and
//! never panics.
//!
//! The tokenizer has no language or code seed list: reserving vocabulary slots for
//! guessed words made small, domain-specific training corpora less efficient. Its
//! vocabulary is reproducible from the corpus and configuration instead.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const PAD: &str = "<pad>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const UNK: &str = "<unk>";
pub const N_SPECIAL: u32 = 4;
/// ids `4..259` are raw UTF-8 bytes. Byte *values* are still `0..=255`.
pub const BYTE_OFFSET: u32 = N_SPECIAL;
/// Default size for general-purpose micro-model corpora. Smaller vocabularies are
/// valid and often preferable for small specialised datasets.
pub const DEFAULT_VOCAB: u32 = 8192;
/// The byte alphabet plus the four special tokens.
pub const MIN_VOCAB: u32 = BYTE_OFFSET + 256;
/// A safety limit; the model configuration performs the RAM admission check.
pub const MAX_VOCAB: u32 = 1_048_576;
/// Encoding is often fed line-sized chunks. Do not retain a whole corpus line
/// merely because it was encoded once.
const ENCODE_CACHE_MAX_ENTRY_BYTES: usize = 4 * 1024;
/// A fixed byte budget keeps the convenience cache from competing with the
/// training batch and activation buffers on small unified-memory Macs.
const ENCODE_CACHE_MAX_BYTES: usize = 1024 * 1024;

/// Runtime `--vocab-size` gate.
pub fn validate_vocab_size(vocab_size: u32) -> Result<u32> {
    if vocab_size < MIN_VOCAB {
        bail!("vocab-size {vocab_size} is below the hard minimum {MIN_VOCAB}");
    }
    if vocab_size > MAX_VOCAB {
        bail!("vocab-size {vocab_size} exceeds the absolute ceiling {MAX_VOCAB}");
    }
    Ok(vocab_size)
}

pub const fn byte_id(byte: u8) -> u32 {
    BYTE_OFFSET + byte as u32
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
    #[serde(default = "byte_bpe_model")]
    pub model: String,
}

fn byte_bpe_model() -> String {
    "byte_bpe_v1".into()
}

#[derive(Clone, Debug, Default)]
struct PieceTrieNode {
    token_id: Option<u32>,
    children: HashMap<u8, usize>,
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    vocab_size: u32,
    pub specials: Vec<String>,
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
    pub merges: Vec<(u32, u32)>,
    id_to_bytes: Vec<Vec<u8>>,
    piece_trie: Vec<PieceTrieNode>,
    encode_cache: HashMap<Vec<u8>, Vec<u32>>,
    encode_cache_bytes: usize,
}

impl BpeTokenizer {
    pub fn new(
        vocab_size: u32,
        merges: Vec<(u32, u32)>,
        specials: Option<Vec<String>>,
    ) -> Result<Self> {
        validate_vocab_size(vocab_size)?;
        let expected_vocab = MIN_VOCAB
            .checked_add(u32::try_from(merges.len()).context("tokenizer merge count overflow")?)
            .ok_or_else(|| anyhow::anyhow!("tokenizer merge count overflow"))?;
        if vocab_size != expected_vocab {
            bail!(
                "vocab_size {vocab_size} is not compact for {} merges (expected {expected_vocab})",
                merges.len()
            );
        }
        for (index, &(left, right)) in merges.iter().enumerate() {
            let new_id = MIN_VOCAB + index as u32;
            if left >= new_id || right >= new_id {
                bail!("merge {index} references a token not defined before it");
            }
        }
        let specials = specials.unwrap_or_else(|| {
            vec![
                PAD.to_string(),
                BOS.to_string(),
                EOS.to_string(),
                UNK.to_string(),
            ]
        });
        if specials.as_slice() != [PAD, BOS, EOS, UNK] {
            bail!("tokenizer specials must be exactly <pad>, <bos>, <eos>, <unk>");
        }
        let mut tok = Self {
            vocab_size,
            specials,
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            unk_id: 3,
            merges,
            id_to_bytes: vec![Vec::new(); vocab_size as usize],
            piece_trie: vec![PieceTrieNode::default()],
            encode_cache: HashMap::new(),
            encode_cache_bytes: 0,
        };
        tok.rebuild();
        Ok(tok)
    }

    pub fn populated(&self) -> u32 {
        self.vocab_size
    }

    /// Exact, compact number of token ids emitted by this tokenizer.
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    fn rebuild(&mut self) {
        self.encode_cache.clear();
        self.encode_cache_bytes = 0;
        self.id_to_bytes = vec![Vec::new(); self.vocab_size as usize];
        self.piece_trie.clear();
        self.piece_trie.push(PieceTrieNode::default());
        for b in 0..=255u8 {
            let id = byte_id(b);
            let bytes = vec![b];
            self.id_to_bytes[id as usize] = bytes.clone();
            self.insert_piece(&bytes, id);
        }
        let mut next_id = BYTE_OFFSET + 256;
        for index in 0..self.merges.len() {
            let (left, right) = self.merges[index];
            let lu = left as usize;
            let ru = right as usize;
            let mut bytes = self.id_to_bytes[lu].clone();
            bytes.extend_from_slice(&self.id_to_bytes[ru]);
            debug_assert!(
                !bytes.is_empty(),
                "validated merge inputs are defined pieces"
            );
            self.insert_piece(&bytes, next_id);
            self.id_to_bytes[next_id as usize] = bytes;
            next_id += 1;
        }
    }

    fn insert_piece(&mut self, bytes: &[u8], token_id: u32) {
        let mut node = 0;
        for &byte in bytes {
            let child = match self.piece_trie[node].children.get(&byte) {
                Some(&child) => child,
                None => {
                    let child = self.piece_trie.len();
                    self.piece_trie.push(PieceTrieNode::default());
                    self.piece_trie[node].children.insert(byte, child);
                    child
                }
            };
            node = child;
        }
        self.piece_trie[node].token_id = Some(token_id);
    }

    /// Greedy longest-match WordPiece. Unknown bytes become `byte_id(b)`.
    pub fn encode_bytes(&mut self, data: &[u8]) -> Vec<u32> {
        if let Some(cached) = self.encode_cache.get(data) {
            return cached.clone();
        }
        let ids = self.encode_bytes_uncached(data);
        let cache_bytes = data
            .len()
            .saturating_add(ids.len().saturating_mul(size_of::<u32>()));
        if data.len() <= ENCODE_CACHE_MAX_ENTRY_BYTES
            && cache_bytes <= ENCODE_CACHE_MAX_BYTES.saturating_sub(self.encode_cache_bytes)
        {
            self.encode_cache.insert(data.to_vec(), ids.clone());
            self.encode_cache_bytes += cache_bytes;
        }
        ids
    }

    fn encode_bytes_uncached(&self, data: &[u8]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(data.len());
        let mut i = 0;
        while i < data.len() {
            let mut node = 0;
            let mut matched = None;
            for (offset, &byte) in data[i..].iter().enumerate() {
                let Some(&child) = self.piece_trie[node].children.get(&byte) else {
                    break;
                };
                node = child;
                if let Some(token_id) = self.piece_trie[node].token_id {
                    matched = Some((token_id, offset + 1));
                }
            }
            if let Some((token_id, length)) = matched {
                ids.push(token_id);
                i += length;
            } else {
                // Every byte is inserted above, but keep this fallback so an
                // invalid future vocabulary can never make encoding loop.
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
            model: byte_bpe_model(),
        }
    }

    pub fn from_json(data: &TokenizerJson) -> Result<Self> {
        if data.model != byte_bpe_model() {
            bail!(
                "unsupported tokenizer model {:?}; retrain this tokenizer with the current byte_bpe_v1 trainer",
                data.model
            );
        }
        let merges: Vec<_> = data.merges.iter().map(|p| (p[0], p[1])).collect();
        let compact_vocab = MIN_VOCAB
            .checked_add(u32::try_from(merges.len()).context("tokenizer merge count overflow")?)
            .ok_or_else(|| anyhow::anyhow!("tokenizer merge count overflow"))?;
        if data.vocab_size < compact_vocab {
            bail!("tokenizer merge table exceeds its declared vocabulary");
        }
        Self::new(compact_vocab, merges, Some(data.specials.clone()))
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
        train_bpe(&[], DEFAULT_VOCAB, 7)
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

/// Byte-level BPE trainer. `max_vocab_size` is an upper limit, not a promise:
/// the returned vocabulary is compact and contains no unused token ids.
pub fn train_bpe(texts: &[String], max_vocab_size: u32, seed: u64) -> Result<BpeTokenizer> {
    let mut chunk_freq = HashMap::new();
    for text in texts {
        add_chunks(&mut chunk_freq, text);
    }
    train_bpe_from_chunks(chunk_freq, max_vocab_size, seed)
}

/// Trains BPE while reading text line-by-line. The full source corpus never
/// needs to be collected as `Vec<String>`; only unique chunk frequencies and
/// the merge working set are retained.
pub fn train_bpe_from_reader<R: BufRead>(
    mut reader: R,
    max_vocab_size: u32,
    seed: u64,
) -> Result<BpeTokenizer> {
    let mut chunk_freq = HashMap::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        add_chunks(&mut chunk_freq, &line);
    }
    train_bpe_from_chunks(chunk_freq, max_vocab_size, seed)
}

fn add_chunks(chunk_freq: &mut HashMap<Vec<u8>, u32>, text: &str) {
    for chunk in text.split_inclusive(char::is_whitespace) {
        if !chunk.is_empty() {
            let frequency = chunk_freq.entry(chunk.as_bytes().to_vec()).or_insert(0);
            *frequency = frequency.saturating_add(1);
        }
    }
}

/// All non-special vocabulary slots are filled by frequency pair-merges on
/// unique whitespace chunks from the supplied corpus.
///
/// The previous trainer rescanned the whole corpus (byte ids) on every merge.
/// At `V=65536` and a ~20 MB JSONL that is tens of thousands of O(N) passes
/// and looks like a hang after `metal_hello`. This path BPE-s unique whitespace
/// chunks with incremental pair counts.
fn train_bpe_from_chunks(
    chunk_freq: HashMap<Vec<u8>, u32>,
    max_vocab_size: u32,
    seed: u64,
) -> Result<BpeTokenizer> {
    use std::collections::BinaryHeap;

    let _ = seed;
    validate_vocab_size(max_vocab_size)?;
    let mut proto = BpeTokenizer::new(MIN_VOCAB, Vec::new(), None)?;
    let mut next_id = BYTE_OFFSET + 256;

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
    let mut seen_words = vec![0u32; words.len()];
    let mut stamp = 1u32;
    while next_id < max_vocab_size {
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
    }

    BpeTokenizer::new(next_id, merges, None)
}

pub fn load_or_train(
    max_vocab_size: u32,
    texts: &[String],
    path: Option<&Path>,
    seed: u64,
) -> Result<BpeTokenizer> {
    validate_vocab_size(max_vocab_size)?;
    if let Some(p) = path {
        if p.as_os_str().is_empty() {
            return train_bpe(texts, max_vocab_size, seed);
        }
        let tok = BpeTokenizer::load(p)?;
        if tok.vocab_size > max_vocab_size {
            bail!(
                "tokenizer vocabulary {} exceeds requested maximum {max_vocab_size}",
                tok.vocab_size,
            );
        }
        return Ok(tok);
    }
    train_bpe(texts, max_vocab_size, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_vocab() {
        let tok = train_bpe(&[], DEFAULT_VOCAB, 7).unwrap();
        assert_eq!(tok.vocab_size(), MIN_VOCAB);
        assert!(tok.merges.is_empty());
    }

    #[test]
    fn roundtrip_ascii() {
        let mut tok = train_bpe(&[], 1024, 1).unwrap();
        let s = "def load(path):\n    return path\n";
        let ids = tok.encode(s, false, false);
        assert_eq!(tok.decode(&ids), s);
    }

    #[test]
    fn corpus_drives_merges() {
        let mut tok512 = train_bpe(&["hello hello hello hello hello".into()], 512, 1).unwrap();
        assert!(
            !tok512.merges.is_empty(),
            "repeated corpus text should create merges"
        );
        let hello = tok512.encode("hello", false, false);
        assert!(
            hello.len() <= 2,
            "hello should merge at V=512, got {} pieces {:?}",
            hello.len(),
            hello
                .iter()
                .map(|&i| tok512.decode(&[i]))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn byte_fallback_never_panics() {
        let mut tok = train_bpe(&[], 320, 0).unwrap();
        let junk: Vec<u8> = (0u8..=255).chain([0xFF, 0xFE, 0x80, 0xC0, 0xC1]).collect();
        let ids = tok.encode_bytes(&junk);
        assert!(!ids.is_empty());
        for &id in &ids {
            assert!(id < tok.vocab_size());
        }
        let _ = tok.decode(&ids);
    }

    #[test]
    fn bilingual_roundtrip() {
        let mut tok = train_bpe(&[], 2048, 2).unwrap();
        let s = "Hello мир fn main()";
        let ids = tok.encode(s, false, false);
        assert_eq!(tok.decode(&ids), s);
    }

    #[test]
    fn rejects_below_min_vocab() {
        assert!(validate_vocab_size(MIN_VOCAB - 1).is_err());
        assert_eq!(validate_vocab_size(320).unwrap(), 320);
        assert_eq!(validate_vocab_size(MIN_VOCAB).unwrap(), MIN_VOCAB);
        assert_eq!(validate_vocab_size(131_072).unwrap(), 131_072);
        assert!(validate_vocab_size(MAX_VOCAB + 1).is_err());
        let tok = load_or_train(512, &["fn add(a: i32, b: i32)".into()], None, 1).unwrap();
        assert!(tok.vocab_size() <= 512);
    }

    #[test]
    fn reader_training_matches_slice_training() {
        let texts = ["alpha beta\n".to_string(), "alpha beta\n".to_string()];
        let from_slice = train_bpe(&texts, 512, 1).unwrap();
        let from_reader = train_bpe_from_reader(&b"alpha beta\nalpha beta\n"[..], 512, 1).unwrap();
        assert_eq!(from_reader.vocab_size(), from_slice.vocab_size());
        assert_eq!(from_reader.merges, from_slice.merges);
    }

    #[test]
    fn json_load_compacts_legacy_empty_ids() {
        let tokenizer = train_bpe(&["alpha alpha alpha".into()], 512, 1).unwrap();
        let mut json = tokenizer.to_json();
        json.vocab_size = 512;
        let loaded = BpeTokenizer::from_json(&json).unwrap();
        assert_eq!(loaded.vocab_size(), tokenizer.vocab_size());
    }

    #[test]
    fn constructor_rejects_unused_vocab_ids_and_invalid_merges() {
        assert!(BpeTokenizer::new(MIN_VOCAB + 1, vec![], None).is_err());
        assert!(BpeTokenizer::new(MIN_VOCAB + 1, vec![(MIN_VOCAB, byte_id(b'a'))], None).is_err());
    }

    #[test]
    fn encode_cache_has_a_strict_byte_budget() {
        let mut tokenizer = train_bpe(&[], MIN_VOCAB, 1).unwrap();
        let oversized = vec![b'x'; ENCODE_CACHE_MAX_ENTRY_BYTES + 1];
        let expected = tokenizer.encode_bytes(&oversized);
        assert_eq!(tokenizer.encode_cache_bytes, 0);
        assert_eq!(tokenizer.encode_bytes(&oversized), expected);

        let chunk = vec![b'y'; ENCODE_CACHE_MAX_ENTRY_BYTES];
        for suffix in 0..300u16 {
            let mut unique = chunk.clone();
            unique[0] = (suffix & 0xff) as u8;
            unique[1] = (suffix >> 8) as u8;
            tokenizer.encode_bytes(&unique);
        }
        assert!(tokenizer.encode_cache_bytes <= ENCODE_CACHE_MAX_BYTES);
    }
}
