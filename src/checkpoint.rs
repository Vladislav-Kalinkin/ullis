//! Self-contained `packed.bin` (magic ULLIS03).

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::TrainConfig;
use crate::device::SovereignDevice;
use crate::kan::NamedBlob;
use crate::model::UllisKan;
use crate::quant::{pack_ternary, unpack_ternary};
use crate::tensor::SovereignTensor;
use crate::tokenizer::{BpeTokenizer, TokenizerJson};

pub const MAGIC: &[u8; 8] = b"ULLIS03\n";

#[derive(Serialize, Deserialize)]
pub struct Header {
    pub engine: String,
    pub config: TrainConfig,
    pub tokenizer: TokenizerJson,
    pub phase: u8,
    pub packed: bool,
    pub tensors: Vec<TensorMeta>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub nbytes: u64,
    pub packed: bool,
}

pub fn save(
    path: impl AsRef<Path>,
    model: &UllisKan,
    tokenizer: &BpeTokenizer,
    phase: u8,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let blobs = model.collect_blobs()?;
    let mut payload = Vec::new();
    let mut metas = Vec::new();
    for (name, blob) in blobs {
        match blob {
            NamedBlob::F32 { data, shape } => {
                let bytes = f32_to_le(&data);
                metas.push(TensorMeta {
                    name,
                    dtype: "f32".into(),
                    shape,
                    offset: payload.len() as u64,
                    nbytes: bytes.len() as u64,
                    packed: false,
                });
                payload.extend_from_slice(&bytes);
            }
            NamedBlob::Packed { bytes, shape } => {
                metas.push(TensorMeta {
                    name,
                    dtype: "u8".into(),
                    shape,
                    offset: payload.len() as u64,
                    nbytes: bytes.len() as u64,
                    packed: true,
                });
                payload.extend_from_slice(&bytes);
            }
            NamedBlob::I8 {
                codes,
                scale,
                shape,
            } => {
                let start = payload.len() as u64;
                payload.extend_from_slice(&codes);
                let scale_bytes = f32_to_le(&scale);
                payload.extend_from_slice(&scale_bytes);
                metas.push(TensorMeta {
                    name,
                    dtype: "i8".into(),
                    shape,
                    offset: start,
                    nbytes: (codes.len() + scale_bytes.len()) as u64,
                    packed: false,
                });
            }
        }
    }
    let packed = model.blocks.iter().any(|b| b.ff.packed);
    let header = Header {
        engine: "Ullis AI Engine v0.8".into(),
        config: model.cfg.clone(),
        tokenizer: tokenizer.to_json(),
        phase,
        packed,
        tensors: metas,
    };
    let header_bytes = serde_json::to_vec(&header)?;
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(MAGIC)?;
    f.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
    f.write_all(&header_bytes)?;
    let written = 8 + 4 + header_bytes.len();
    let pad = (64 - (written % 64)) % 64;
    f.write_all(&vec![0u8; pad])?;
    f.write_all(&payload)?;
    Ok(())
}

pub struct Loaded {
    pub model: UllisKan,
    pub tokenizer: BpeTokenizer,
    pub phase: u8,
}

pub fn load(path: impl AsRef<Path>, device: SovereignDevice) -> Result<Loaded> {
    let path = path.as_ref();
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!(
            "bad magic in {}: {:x?} (want ULLIS03)",
            path.display(),
            magic
        );
    }
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)?;
    let hlen = u32::from_le_bytes(len_buf) as usize;
    let mut header_bytes = vec![0u8; hlen];
    f.read_exact(&mut header_bytes)?;
    let header: Header = serde_json::from_slice(&header_bytes)?;
    let written = 8 + 4 + hlen;
    let pad = (64 - (written % 64)) % 64;
    let mut pad_buf = vec![0u8; pad];
    f.read_exact(&mut pad_buf)?;
    let mut payload = Vec::new();
    f.read_to_end(&mut payload)?;

    let tokenizer = BpeTokenizer::from_json(&header.tokenizer)?;
    let mut cfg = header.config.clone();
    cfg.vocab_size = tokenizer.vocab_size as usize;
    let mut model = UllisKan::new(cfg, device)?;
    if header.packed {
        model.pack()?;
    }

    for meta in &header.tensors {
        let start = meta.offset as usize;
        let end = start + meta.nbytes as usize;
        if end > payload.len() {
            bail!("tensor {} overruns payload", meta.name);
        }
        let bytes = &payload[start..end];
        if meta.packed {
            let n: usize = meta.shape.iter().product();
            let codes = unpack_ternary(bytes, n);
            apply_packed(&mut model, &meta.name, &codes, &meta.shape)?;
        } else if meta.dtype == "i8" && meta.name == "embed" {
            let n: usize = meta.shape.iter().product();
            if bytes.len() < n {
                bail!("embed i8 overruns payload");
            }
            let codes = &bytes[..n];
            let scale = le_to_f32(&bytes[n..]);
            model.load_i8_embed(codes, &scale, &meta.shape)?;
        } else {
            let v = le_to_f32(bytes);
            if let Err(e) = model.load_blob(&meta.name, &v, &meta.shape) {
                eprintln!("ullis: skip {}: {e}", meta.name);
            }
        }
    }
    if !header.tensors.iter().any(|t| t.name.contains("inv_widths")) {
        model.sync_grids();
    }
    Ok(Loaded {
        model,
        tokenizer,
        phase: header.phase,
    })
}

fn apply_packed(model: &mut UllisKan, name: &str, codes: &[i8], shape: &[usize]) -> Result<()> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 4 || parts[0] != "blocks" {
        return Ok(());
    }
    let idx: usize = parts[1].parse().unwrap_or(0);
    if idx >= model.blocks.len() {
        return Ok(());
    }
    let f: Vec<f32> = codes.iter().map(|&c| c as f32).collect();
    let t = SovereignTensor::from_vec(shape.to_vec(), f)?;
    let packed = pack_ternary(codes);
    let ff = &mut model.blocks[idx].ff;
    if ff.packed_codes.is_none() {
        ff.pack()?;
    }
    if let Some(pc) = &mut ff.packed_codes {
        match parts[3] {
            "packed_base" => {
                pc.code_base = t;
                pc.packed_base = packed;
            }
            "packed_shared" => {
                pc.code_shared = t;
                pc.packed_shared = packed;
            }
            "packed_routed" => {
                pc.code_routed = Some(t);
                pc.packed_routed = packed;
            }
            _ => {}
        }
    }
    Ok(())
}

fn f32_to_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
