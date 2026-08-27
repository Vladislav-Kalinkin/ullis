//! CPU Heron: LayerNorm, BinaryConnect linears, CMix x070, and checkpoint v2.
//!
//! Packed ±1 matrices keep FP16 latents in RAM for the life of the process.
//! Checkpoints store bits, learned scales, and bias only.

use crate::config::{Architecture, TrainConfig};
use crate::precision::{Fp16, Fp16Storage};
use crate::rosa::{RosaSam, bit_from_activation};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;

/// Next-token cross-entropy statistics. Values are means over valid positions;
/// no `[batch, time, vocab]` logits tensor is retained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausalLoss {
    pub next_token: f32,
    pub next_token_count: usize,
}

impl CausalLoss {
    pub fn mean(self) -> f32 {
        self.next_token
    }
}

pub const LAYER_NORM_EPS: f32 = 1e-5;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn packed_word_count(weights: usize) -> usize {
    weights.div_ceil(32)
}

fn bit_is_plus(bits: &[u32], index: usize) -> bool {
    bits[index / 32] & (1_u32 << (index % 32)) != 0
}

fn pack_plus_bits(plus: impl Iterator<Item = bool>, weights: usize) -> Vec<u32> {
    let mut bits = vec![0_u32; packed_word_count(weights)];
    for (index, is_plus) in plus.enumerate() {
        if is_plus {
            bits[index / 32] |= 1_u32 << (index % 32);
        }
    }
    bits
}

fn time_shift_delta(x: &[f32], batch: usize, time: usize, channels: usize) -> Result<Vec<f32>> {
    let rows = batch
        .checked_mul(time)
        .ok_or_else(|| anyhow::anyhow!("time-shift shape overflow"))?;
    if x.len() != rows.saturating_mul(channels) || time == 0 || channels == 0 {
        bail!("time-shift input shape mismatch");
    }
    let mut xx = vec![0.0; x.len()];
    for b in 0..batch {
        for t in 0..time {
            let row = (b * time + t) * channels;
            let prev = if t == 0 {
                None
            } else {
                Some((b * time + t - 1) * channels)
            };
            for c in 0..channels {
                let xt = x[row + c];
                xx[row + c] = match prev {
                    None => -xt,
                    Some(prev_row) => x[prev_row + c] - xt,
                };
            }
        }
    }
    Ok(xx)
}

/// Affine LayerNorm over the last dimension, `eps = 1e-5`, matching `nn.LayerNorm`.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Fp16Storage,
    bias: Fp16Storage,
}

impl LayerNorm {
    pub fn new(channels: usize) -> Self {
        Self {
            weight: Fp16Storage::from_f32((0..channels).map(|_| 1.0)),
            bias: Fp16Storage::zeros(channels),
        }
    }

    pub fn from_bits(weight: Vec<u16>, bias: Vec<u16>) -> Result<Self> {
        if weight.len() != bias.len() || weight.is_empty() {
            bail!("LayerNorm checkpoint length mismatch");
        }
        Ok(Self {
            weight: Fp16Storage::from_bits(weight),
            bias: Fp16Storage::from_bits(bias),
        })
    }

    pub fn channels(&self) -> usize {
        self.weight.len()
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let channels = self.channels();
        if rows.checked_mul(channels) != Some(x.len()) {
            bail!("LayerNorm input shape mismatch");
        }
        let mut y = vec![0.0; x.len()];
        for row in 0..rows {
            let start = row * channels;
            let src = &x[start..start + channels];
            let mean = src.iter().sum::<f32>() / channels as f32;
            let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / channels as f32;
            let inv = (var + LAYER_NORM_EPS).sqrt().recip();
            for c in 0..channels {
                y[start + c] = (src[c] - mean) * inv * self.weight.get(c) + self.bias.get(c);
            }
        }
        Ok(y)
    }
}

/// Dense FP16 matrix without bias (CMix value, later Tmix).
#[derive(Clone, Debug)]
pub struct Fp16Linear {
    out_features: usize,
    in_features: usize,
    weight: Fp16Storage,
}

impl Fp16Linear {
    pub fn zeros(out_features: usize, in_features: usize) -> Result<Self> {
        let weights = out_features
            .checked_mul(in_features)
            .ok_or_else(|| anyhow::anyhow!("FP16 linear shape overflow"))?;
        Ok(Self {
            out_features,
            in_features,
            weight: Fp16Storage::zeros(weights),
        })
    }

    pub fn from_f32(out_features: usize, in_features: usize, values: &[f32]) -> Result<Self> {
        if values.len() != out_features.saturating_mul(in_features) {
            bail!("FP16 linear value length mismatch");
        }
        Ok(Self {
            out_features,
            in_features,
            weight: Fp16Storage::from_f32(values.iter().copied()),
        })
    }

    pub fn from_bits(out_features: usize, in_features: usize, bits: Vec<u16>) -> Result<Self> {
        if bits.len() != out_features.saturating_mul(in_features) {
            bail!("FP16 linear checkpoint length mismatch");
        }
        Ok(Self {
            out_features,
            in_features,
            weight: Fp16Storage::from_bits(bits),
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        if rows.checked_mul(self.in_features) != Some(x.len()) {
            bail!("FP16 linear input shape mismatch");
        }
        let mut y = vec![0.0; rows.saturating_mul(self.out_features)];
        for row in 0..rows {
            let x_row = &x[row * self.in_features..(row + 1) * self.in_features];
            for o in 0..self.out_features {
                let mut sum = 0.0;
                let w_row = o * self.in_features;
                for i in 0..self.in_features {
                    sum += self.weight.get(w_row + i) * x_row[i];
                }
                y[row * self.out_features + o] = sum;
            }
        }
        Ok(y)
    }
}

/// Packed ±1 linear with a persistent FP16 BinaryConnect latent.
#[derive(Clone, Debug)]
pub struct PackedBinaryLinear {
    out_features: usize,
    in_features: usize,
    bits: Vec<u32>,
    scale: Fp16Storage,
    bias: Option<Fp16Storage>,
    latent: Fp16Storage,
}

impl PackedBinaryLinear {
    pub fn seeded(out_features: usize, in_features: usize, bias: bool, seed: u64) -> Result<Self> {
        let weights = out_features
            .checked_mul(in_features)
            .ok_or_else(|| anyhow::anyhow!("packed linear shape overflow"))?;
        let scale_value = (in_features as f32).sqrt().recip();
        let scale = Fp16Storage::from_f32((0..out_features).map(|_| scale_value));
        let mut state = seed | 1;
        let plus = (0..weights).map(|_| splitmix64(&mut state) & 1 == 1);
        let bits = pack_plus_bits(plus, weights);
        let mut latent = Fp16Storage::zeros(weights);
        for o in 0..out_features {
            let s = scale.get(o);
            for i in 0..in_features {
                let index = o * in_features + i;
                let sign = if bit_is_plus(&bits, index) { 1.0 } else { -1.0 };
                latent.set(index, s * sign);
            }
        }
        Ok(Self {
            out_features,
            in_features,
            bits,
            scale,
            bias: bias.then(|| Fp16Storage::zeros(out_features)),
            latent,
        })
    }

    pub fn from_signs(
        out_features: usize,
        in_features: usize,
        signs: &[i8],
        scale: f32,
        bias: bool,
    ) -> Result<Self> {
        let weights = out_features
            .checked_mul(in_features)
            .ok_or_else(|| anyhow::anyhow!("packed linear shape overflow"))?;
        if signs.len() != weights {
            bail!("packed linear sign length mismatch");
        }
        let bits = pack_plus_bits(signs.iter().map(|&s| s >= 0), weights);
        let scale_store = Fp16Storage::from_f32((0..out_features).map(|_| scale));
        let mut latent = Fp16Storage::zeros(weights);
        for (index, &sign) in signs.iter().enumerate() {
            let s = if sign >= 0 { 1.0 } else { -1.0 };
            latent.set(index, scale * s);
        }
        Ok(Self {
            out_features,
            in_features,
            bits,
            scale: scale_store,
            bias: bias.then(|| Fp16Storage::zeros(out_features)),
            latent,
        })
    }

    fn from_packed(
        out_features: usize,
        in_features: usize,
        bits: Vec<u32>,
        scale_bits: Vec<u16>,
        bias_bits: Option<Vec<u16>>,
    ) -> Result<Self> {
        let weights = out_features
            .checked_mul(in_features)
            .ok_or_else(|| anyhow::anyhow!("packed linear shape overflow"))?;
        if bits.len() != packed_word_count(weights) || scale_bits.len() != out_features {
            bail!("packed linear checkpoint shape mismatch");
        }
        let scale = Fp16Storage::from_bits(scale_bits);
        let bias = match bias_bits {
            Some(values) if values.len() == out_features => Some(Fp16Storage::from_bits(values)),
            None => None,
            Some(_) => bail!("packed linear bias length mismatch"),
        };
        let mut latent = Fp16Storage::zeros(weights);
        for o in 0..out_features {
            let s = scale.get(o);
            for i in 0..in_features {
                let index = o * in_features + i;
                let sign = if bit_is_plus(&bits, index) { 1.0 } else { -1.0 };
                latent.set(index, s * sign);
            }
        }
        Ok(Self {
            out_features,
            in_features,
            bits,
            scale,
            bias,
            latent,
        })
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn latent(&self) -> &Fp16Storage {
        &self.latent
    }

    pub fn scale(&self) -> &Fp16Storage {
        &self.scale
    }

    pub fn sign_at(&self, index: usize) -> f32 {
        if bit_is_plus(&self.bits, index) {
            1.0
        } else {
            -1.0
        }
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        if rows.checked_mul(self.in_features) != Some(x.len()) {
            bail!("packed linear input shape mismatch");
        }
        let mut y = vec![0.0; rows.saturating_mul(self.out_features)];
        for row in 0..rows {
            let x_row = &x[row * self.in_features..(row + 1) * self.in_features];
            for o in 0..self.out_features {
                let mut sum = 0.0;
                let base = o * self.in_features;
                for i in 0..self.in_features {
                    sum += self.sign_at(base + i) * x_row[i];
                }
                let bias = self.bias.as_ref().map_or(0.0, |bias| bias.get(o));
                y[row * self.out_features + o] = bias + self.scale.get(o) * sum;
            }
        }
        Ok(y)
    }

    /// STE through `sign`. `g_w` is `[out, in]` and is accumulated, not zeroed.
    pub fn backward_ste(
        &self,
        x: &[f32],
        gy: &[f32],
        rows: usize,
        g_w: &mut [f32],
        mut g_x: Option<&mut [f32]>,
        g_scale: &mut [f32],
        mut g_bias: Option<&mut [f32]>,
    ) -> Result<()> {
        let weights = self.out_features.saturating_mul(self.in_features);
        if rows.checked_mul(self.in_features) != Some(x.len())
            || rows.checked_mul(self.out_features) != Some(gy.len())
            || g_w.len() != weights
            || g_scale.len() != self.out_features
        {
            bail!("packed linear backward shape mismatch");
        }
        if let Some(ref gx) = g_x
            && gx.len() != x.len()
        {
            bail!("packed linear input-gradient shape mismatch");
        }
        if let Some(ref gb) = g_bias
            && gb.len() != self.out_features
        {
            bail!("packed linear bias-gradient shape mismatch");
        }
        for row in 0..rows {
            let x_row = &x[row * self.in_features..(row + 1) * self.in_features];
            let gy_row = &gy[row * self.out_features..(row + 1) * self.out_features];
            for o in 0..self.out_features {
                let scale = self.scale.get(o);
                let gy_o = gy_row[o];
                let mut signed_dot = 0.0;
                let base = o * self.in_features;
                for i in 0..self.in_features {
                    let s = self.sign_at(base + i);
                    g_w[base + i] += gy_o * scale * x_row[i];
                    if let Some(ref mut gx) = g_x {
                        gx[row * self.in_features + i] += gy_o * scale * s;
                    }
                    signed_dot += s * x_row[i];
                }
                g_scale[o] += gy_o * signed_dot;
                if let Some(ref mut gb) = g_bias {
                    gb[o] += gy_o;
                }
            }
        }
        Ok(())
    }

    pub fn apply_clipped_sgd(
        &mut self,
        g_w: &[f32],
        g_scale: &[f32],
        g_bias: Option<&[f32]>,
        learning_rate: f32,
    ) -> Result<()> {
        let weights = self.out_features.saturating_mul(self.in_features);
        if g_w.len() != weights || g_scale.len() != self.out_features {
            bail!("packed linear SGD shape mismatch");
        }
        for i in 0..weights {
            self.latent.apply_clipped_sgd(i, g_w[i], learning_rate);
        }
        for o in 0..self.out_features {
            self.scale.apply_clipped_sgd(o, g_scale[o], learning_rate);
        }
        if let (Some(bias), Some(g_bias)) = (self.bias.as_mut(), g_bias) {
            if g_bias.len() != self.out_features {
                bail!("packed linear bias SGD shape mismatch");
            }
            for o in 0..self.out_features {
                bias.apply_clipped_sgd(o, g_bias[o], learning_rate);
            }
        }
        self.rebinarize();
        Ok(())
    }

    pub fn rebinarize(&mut self) {
        let weights = self.out_features.saturating_mul(self.in_features);
        self.bits = pack_plus_bits(
            (0..weights).map(|index| self.latent.get(index) >= 0.0),
            weights,
        );
    }
}

/// CMix x070: `k = relu(key(x + xx * x_k))^2`, then FP16 `value(k)`.
#[derive(Clone, Debug)]
pub struct RwkvCMixX070 {
    x_k: Fp16Storage,
    pub key: PackedBinaryLinear,
    pub value: Fp16Linear,
}

impl RwkvCMixX070 {
    pub fn seeded(d_model: usize, dim_ffn: usize, seed: u64) -> Result<Self> {
        Ok(Self {
            x_k: Fp16Storage::zeros(d_model),
            key: PackedBinaryLinear::seeded(dim_ffn, d_model, false, seed)?,
            value: Fp16Linear::zeros(d_model, dim_ffn)?,
        })
    }

    pub fn from_parts_for_test(
        x_k: impl IntoIterator<Item = f32>,
        key: PackedBinaryLinear,
        value: Fp16Linear,
    ) -> Self {
        Self {
            x_k: Fp16Storage::from_f32(x_k),
            key,
            value,
        }
    }

    pub fn forward(&self, x: &[f32], batch: usize, time: usize) -> Result<Vec<f32>> {
        let d = self.x_k.len();
        let xx = time_shift_delta(x, batch, time, d)?;
        let rows = batch.saturating_mul(time);
        let mut shifted = vec![0.0; x.len()];
        for i in 0..rows {
            for c in 0..d {
                let index = i * d + c;
                shifted[index] = x[index] + xx[index] * self.x_k.get(c);
            }
        }
        let key = self.key.forward(&shifted, rows)?;
        let relu2: Vec<f32> = key.iter().map(|v| v.max(0.0) * v.max(0.0)).collect();
        self.value.forward(&relu2, rows)
    }
}

#[derive(Clone, Debug)]
pub struct RwkvRosaQkv1Bit {
    x_q: Fp16Storage,
    x_k: Fp16Storage,
    x_v: Fp16Storage,
    e: Fp16Storage,
    q: PackedBinaryLinear,
    k: PackedBinaryLinear,
    v: PackedBinaryLinear,
    o: PackedBinaryLinear,
}

impl RwkvRosaQkv1Bit {
    pub fn seeded(d_model: usize, seed: u64) -> Result<Self> {
        Ok(Self {
            x_q: Fp16Storage::zeros(d_model),
            x_k: Fp16Storage::zeros(d_model),
            x_v: Fp16Storage::zeros(d_model),
            e: Fp16Storage::from_f32((0..d_model).map(|_| 1.0)),
            q: PackedBinaryLinear::seeded(d_model, d_model, true, seed)?,
            k: PackedBinaryLinear::seeded(d_model, d_model, true, seed ^ 1)?,
            v: PackedBinaryLinear::seeded(d_model, d_model, true, seed ^ 2)?,
            o: PackedBinaryLinear::seeded(d_model, d_model, true, seed ^ 3)?,
        })
    }

    pub fn forward(&self, x: &[f32], batch: usize, time: usize) -> Result<Vec<f32>> {
        let d = self.e.len();
        let xx = time_shift_delta(x, batch, time, d)?;
        let rows = batch.saturating_mul(time);
        let mut q_in = vec![0.0; x.len()];
        let mut k_in = vec![0.0; x.len()];
        let mut v_in = vec![0.0; x.len()];
        for i in 0..rows {
            for c in 0..d {
                let index = i * d + c;
                q_in[index] = x[index] + xx[index] * self.x_q.get(c);
                k_in[index] = x[index] + xx[index] * self.x_k.get(c);
                v_in[index] = x[index] + xx[index] * self.x_v.get(c);
            }
        }
        let q = self.q.forward(&q_in, rows)?;
        let k = self.k.forward(&k_in, rows)?;
        let v = self.v.forward(&v_in, rows)?;
        let mut y = vec![0.0; rows.saturating_mul(d)];
        for b in 0..batch {
            for c in 0..d {
                let mut sam = RosaSam::with_max_time(time);
                for t in 0..time {
                    let index = (b * time + t) * d + c;
                    let idx = sam.push(
                        bit_from_activation(q[index]),
                        bit_from_activation(k[index]),
                        bit_from_activation(v[index]),
                    );
                    y[index] = (2.0 * f32::from(idx) - 1.0) * self.e.get(c);
                }
            }
        }
        self.o.forward(&y, rows)
    }
}

#[derive(Clone, Debug)]
struct HeronBlock {
    ln0: Option<LayerNorm>,
    ln2: LayerNorm,
    ln3: LayerNorm,
    rosa: RwkvRosaQkv1Bit,
    ffn: RwkvCMixX070,
}

impl HeronBlock {
    fn forward(&self, x: &[f32], batch: usize, time: usize) -> Result<Vec<f32>> {
        let mut h = if let Some(ln0) = &self.ln0 {
            ln0.forward(x, batch.saturating_mul(time))?
        } else {
            x.to_vec()
        };
        let rosa_in = self.ln3.forward(&h, batch.saturating_mul(time))?;
        let rosa_out = self.rosa.forward(&rosa_in, batch, time)?;
        for (dst, src) in h.iter_mut().zip(&rosa_out) {
            *dst += *src;
        }
        let ffn_in = self.ln2.forward(&h, batch.saturating_mul(time))?;
        let ffn_out = self.ffn.forward(&ffn_in, batch, time)?;
        for (dst, src) in h.iter_mut().zip(&ffn_out) {
            *dst += *src;
        }
        Ok(h)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PackedBinaryCheckpoint {
    /// Little-endian u32 words, row-major `[out, in]`. Bit 0 of word 0 is
    /// weight `[0, 0]`; `1` means `+1`, `0` means `-1`.
    bits: Vec<u32>,
    scale_bits: Vec<u16>,
    bias_bits: Option<Vec<u16>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LayerNormBits {
    weight: Vec<u16>,
    bias: Vec<u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Fp16Vec(Vec<u16>);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RosaCheckpoint {
    x_q: Fp16Vec,
    x_k: Fp16Vec,
    x_v: Fp16Vec,
    e: Fp16Vec,
    q: PackedBinaryCheckpoint,
    k: PackedBinaryCheckpoint,
    v: PackedBinaryCheckpoint,
    o: PackedBinaryCheckpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CmixCheckpoint {
    x_k: Fp16Vec,
    key: PackedBinaryCheckpoint,
    value_bits: Fp16Vec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HeronBlockCheckpoint {
    ln0: Option<LayerNormBits>,
    ln2: LayerNormBits,
    ln3: LayerNormBits,
    rosa: RosaCheckpoint,
    ffn: CmixCheckpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TmixCheckpoint {
    x_r: Fp16Vec,
    x_w: Fp16Vec,
    x_k: Fp16Vec,
    x_v: Fp16Vec,
    x_a: Fp16Vec,
    x_g: Fp16Vec,
    w1: Fp16Vec,
    a1: Fp16Vec,
    v1: Fp16Vec,
    g1: Fp16Vec,
    w2: Fp16Vec,
    a2: Fp16Vec,
    v2: Fp16Vec,
    g2: Fp16Vec,
    w0: Fp16Vec,
    a0: Fp16Vec,
    v0: Fp16Vec,
    k_k: Fp16Vec,
    k_a: Fp16Vec,
    r_k: Fp16Vec,
    receptance: Fp16Vec,
    key: Fp16Vec,
    value: Fp16Vec,
    output: Fp16Vec,
    ln_x_weight: Fp16Vec,
    ln_x_bias: Fp16Vec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HybridBlockCheckpoint {
    ln_a: LayerNormBits,
    ln_b: LayerNormBits,
    ln_c: LayerNormBits,
    tmix: TmixCheckpoint,
    rosa: RosaCheckpoint,
    ffn: CmixCheckpoint,
}

/// Versioned snapshot of persistent Heron state.
///
/// Packed ±1 matrices store bits, learned scales, and official bias. FP16
/// masters for those matrices are process-local BinaryConnect latents and are
/// not written here. Hyena `format_version: 1` files are intentionally
/// unloadable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub format_version: u32,
    pub config: TrainConfig,
    embedding_bits: Vec<u16>,
    ln_out: LayerNormBits,
    head: PackedBinaryCheckpoint,
    #[serde(default)]
    blocks: Vec<HeronBlockCheckpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    hybrid_blocks: Vec<HybridBlockCheckpoint>,
}

/// Product model. Packed BinaryConnect latents live in RAM; checkpoints omit them.
#[derive(Clone, Debug)]
pub struct UllisHeron {
    pub cfg: TrainConfig,
    embedding: Fp16Storage,
    blocks: Vec<HeronBlock>,
    ln_out: LayerNorm,
    head: PackedBinaryLinear,
    g_w: Vec<f32>,
    hybrid_checkpoint: Option<ModelCheckpoint>,
}

impl UllisHeron {
    pub fn new(cfg: TrainConfig) -> Result<Self> {
        cfg.validate()?;
        match cfg.architecture {
            Architecture::Heron => Self::new_heron(cfg),
            Architecture::RosaRwkv7 => {
                let checkpoint = skeleton_checkpoint(&cfg)?;
                Ok(Self {
                    cfg,
                    embedding: Fp16Storage::zeros(1),
                    blocks: Vec::new(),
                    ln_out: LayerNorm::new(1),
                    head: PackedBinaryLinear::seeded(1, 1, false, 1)?,
                    g_w: vec![0.0; 1],
                    hybrid_checkpoint: Some(checkpoint),
                })
            }
        }
    }

    fn new_heron(cfg: TrainConfig) -> Result<Self> {
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let dim_ffn = cfg.resolved_dim_ffn();
        let embedding = seeded_embedding(v.saturating_mul(d), d, cfg.seed);
        let blocks = (0..cfg.n_layers)
            .map(|layer| {
                Ok(HeronBlock {
                    ln0: (layer == 0).then(|| LayerNorm::new(d)),
                    ln2: LayerNorm::new(d),
                    ln3: LayerNorm::new(d),
                    rosa: RwkvRosaQkv1Bit::seeded(d, cfg.seed.wrapping_add(layer as u64 + 1))?,
                    ffn: RwkvCMixX070::seeded(
                        d,
                        dim_ffn,
                        cfg.seed.wrapping_add(layer as u64 + 17),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let head = PackedBinaryLinear::seeded(v, d, false, cfg.seed ^ 0x9E37)?;
        let max_matrix = d
            .saturating_mul(d)
            .max(d.saturating_mul(dim_ffn))
            .max(v.saturating_mul(d));
        Ok(Self {
            cfg,
            embedding,
            blocks,
            ln_out: LayerNorm::new(d),
            head,
            g_w: vec![0.0; max_matrix],
            hybrid_checkpoint: None,
        })
    }

    pub fn gradient_workspace(&self) -> &[f32] {
        &self.g_w
    }

    pub fn checkpoint(&self) -> ModelCheckpoint {
        if let Some(hybrid) = &self.hybrid_checkpoint {
            return hybrid.clone();
        }
        ModelCheckpoint {
            format_version: CHECKPOINT_FORMAT_VERSION,
            config: self.cfg.clone(),
            embedding_bits: self.embedding.as_bits().to_vec(),
            ln_out: LayerNormBits {
                weight: self.ln_out.weight.as_bits().to_vec(),
                bias: self.ln_out.bias.as_bits().to_vec(),
            },
            head: packed_to_checkpoint(&self.head),
            blocks: self
                .blocks
                .iter()
                .map(|block| HeronBlockCheckpoint {
                    ln0: block.ln0.as_ref().map(|ln| LayerNormBits {
                        weight: ln.weight.as_bits().to_vec(),
                        bias: ln.bias.as_bits().to_vec(),
                    }),
                    ln2: LayerNormBits {
                        weight: block.ln2.weight.as_bits().to_vec(),
                        bias: block.ln2.bias.as_bits().to_vec(),
                    },
                    ln3: LayerNormBits {
                        weight: block.ln3.weight.as_bits().to_vec(),
                        bias: block.ln3.bias.as_bits().to_vec(),
                    },
                    rosa: RosaCheckpoint {
                        x_q: Fp16Vec(block.rosa.x_q.as_bits().to_vec()),
                        x_k: Fp16Vec(block.rosa.x_k.as_bits().to_vec()),
                        x_v: Fp16Vec(block.rosa.x_v.as_bits().to_vec()),
                        e: Fp16Vec(block.rosa.e.as_bits().to_vec()),
                        q: packed_to_checkpoint(&block.rosa.q),
                        k: packed_to_checkpoint(&block.rosa.k),
                        v: packed_to_checkpoint(&block.rosa.v),
                        o: packed_to_checkpoint(&block.rosa.o),
                    },
                    ffn: CmixCheckpoint {
                        x_k: Fp16Vec(block.ffn.x_k.as_bits().to_vec()),
                        key: packed_to_checkpoint(&block.ffn.key),
                        value_bits: Fp16Vec(block.ffn.value.weight.as_bits().to_vec()),
                    },
                })
                .collect(),
            hybrid_blocks: Vec::new(),
        }
    }

    pub fn from_checkpoint(checkpoint: ModelCheckpoint) -> Result<Self> {
        if checkpoint.format_version != CHECKPOINT_FORMAT_VERSION {
            bail!(
                "Hyena checkpoints (v1) are intentionally unloadable after the RWKV-8 cut (got format_version {})",
                checkpoint.format_version
            );
        }
        checkpoint.config.validate()?;
        validate_checkpoint_shapes(&checkpoint)?;
        match checkpoint.config.architecture {
            Architecture::RosaRwkv7 => {
                let cfg = checkpoint.config.clone();
                Ok(Self {
                    cfg,
                    embedding: Fp16Storage::zeros(1),
                    blocks: Vec::new(),
                    ln_out: LayerNorm::new(1),
                    head: PackedBinaryLinear::seeded(1, 1, false, 1)?,
                    g_w: vec![0.0; 1],
                    hybrid_checkpoint: Some(checkpoint),
                })
            }
            Architecture::Heron => heron_from_checkpoint(checkpoint),
        }
    }

    pub fn hidden(&self, ids: &[u32], batch: usize, time: usize) -> Result<Vec<f32>> {
        if self.hybrid_checkpoint.is_some() {
            bail!("rosa_rwkv7 CPU hidden is not wired in this PR");
        }
        let d = self.cfg.d_model;
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("token shape overflow"))?;
        if batch == 0
            || batch > self.cfg.batch_size
            || ids.len() != rows
            || time == 0
            || time > self.cfg.context_len
        {
            bail!("token shape or context length is invalid");
        }
        let mut x = vec![0.0; rows.saturating_mul(d)];
        for (row, &id) in ids.iter().enumerate() {
            if id as usize >= self.cfg.vocab_size {
                bail!("token id exceeds vocabulary");
            }
            let offset = id as usize * d;
            for c in 0..d {
                x[row * d + c] = self.embedding.get(offset + c);
            }
        }
        for block in &self.blocks {
            x = block.forward(&x, batch, time)?;
        }
        self.ln_out.forward(&x, rows)
    }

    pub fn train_step(
        &mut self,
        _tokens: &[u32],
        _batch: usize,
        _time: usize,
        _learning_rate: f32,
    ) -> Result<CausalLoss> {
        bail!("Heron train not wired")
    }
}

fn packed_to_checkpoint(linear: &PackedBinaryLinear) -> PackedBinaryCheckpoint {
    PackedBinaryCheckpoint {
        bits: linear.bits.clone(),
        scale_bits: linear.scale.as_bits().to_vec(),
        bias_bits: linear.bias.as_ref().map(|bias| bias.as_bits().to_vec()),
    }
}

fn packed_from_checkpoint(
    packed: PackedBinaryCheckpoint,
    out_features: usize,
    in_features: usize,
    bias: bool,
) -> Result<PackedBinaryLinear> {
    if packed.bias_bits.is_some() != bias {
        bail!("packed linear bias flag mismatch");
    }
    PackedBinaryLinear::from_packed(
        out_features,
        in_features,
        packed.bits,
        packed.scale_bits,
        packed.bias_bits,
    )
}

fn seeded_embedding(len: usize, d_model: usize, seed: u64) -> Fp16Storage {
    let scale = (d_model as f32).sqrt().recip();
    let mut state = seed | 1;
    Fp16Storage::from_f32((0..len).map(|_| {
        let word = splitmix64(&mut state);
        let unit = (word >> 11) as f32 / ((1_u64 << 53) as f32);
        (unit * 2.0 - 1.0) * scale
    }))
}

fn heron_from_checkpoint(checkpoint: ModelCheckpoint) -> Result<UllisHeron> {
    let cfg = checkpoint.config.clone();
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let dim_ffn = cfg.resolved_dim_ffn();
    let embedding = Fp16Storage::from_bits(checkpoint.embedding_bits);
    let ln_out = LayerNorm::from_bits(checkpoint.ln_out.weight, checkpoint.ln_out.bias)?;
    let head = packed_from_checkpoint(checkpoint.head, v, d, false)?;
    let blocks = checkpoint
        .blocks
        .into_iter()
        .map(|block| {
            Ok(HeronBlock {
                ln0: block
                    .ln0
                    .map(|ln| LayerNorm::from_bits(ln.weight, ln.bias))
                    .transpose()?,
                ln2: LayerNorm::from_bits(block.ln2.weight, block.ln2.bias)?,
                ln3: LayerNorm::from_bits(block.ln3.weight, block.ln3.bias)?,
                rosa: RwkvRosaQkv1Bit {
                    x_q: Fp16Storage::from_bits(block.rosa.x_q.0),
                    x_k: Fp16Storage::from_bits(block.rosa.x_k.0),
                    x_v: Fp16Storage::from_bits(block.rosa.x_v.0),
                    e: Fp16Storage::from_bits(block.rosa.e.0),
                    q: packed_from_checkpoint(block.rosa.q, d, d, true)?,
                    k: packed_from_checkpoint(block.rosa.k, d, d, true)?,
                    v: packed_from_checkpoint(block.rosa.v, d, d, true)?,
                    o: packed_from_checkpoint(block.rosa.o, d, d, true)?,
                },
                ffn: RwkvCMixX070 {
                    x_k: Fp16Storage::from_bits(block.ffn.x_k.0),
                    key: packed_from_checkpoint(block.ffn.key, dim_ffn, d, false)?,
                    value: Fp16Linear::from_bits(d, dim_ffn, block.ffn.value_bits.0)?,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let max_matrix = d
        .saturating_mul(d)
        .max(d.saturating_mul(dim_ffn))
        .max(v.saturating_mul(d));
    Ok(UllisHeron {
        cfg,
        embedding,
        blocks,
        ln_out,
        head,
        g_w: vec![0.0; max_matrix],
        hybrid_checkpoint: None,
    })
}

fn ones(len: usize) -> Vec<u16> {
    vec![Fp16::from_f32(1.0).to_bits(); len]
}

fn zeros(len: usize) -> Vec<u16> {
    vec![0; len]
}

fn packed_linear(out: usize, in_features: usize, bias: bool, scale: f32) -> PackedBinaryCheckpoint {
    let weights = out.saturating_mul(in_features);
    let words = weights.div_ceil(32);
    PackedBinaryCheckpoint {
        bits: vec![0; words],
        scale_bits: vec![Fp16::from_f32(scale).to_bits(); out],
        bias_bits: bias.then(|| zeros(out)),
    }
}

fn layer_norm(d: usize) -> LayerNormBits {
    LayerNormBits {
        weight: ones(d),
        bias: zeros(d),
    }
}

fn fp16_zeros(len: usize) -> Fp16Vec {
    Fp16Vec(zeros(len))
}

fn fp16_ones(len: usize) -> Fp16Vec {
    Fp16Vec(ones(len))
}

fn rosa_checkpoint(d: usize) -> RosaCheckpoint {
    let scale = (d as f32).sqrt().recip();
    RosaCheckpoint {
        x_q: fp16_zeros(d),
        x_k: fp16_zeros(d),
        x_v: fp16_zeros(d),
        e: fp16_ones(d),
        q: packed_linear(d, d, true, scale),
        k: packed_linear(d, d, true, scale),
        v: packed_linear(d, d, true, scale),
        o: packed_linear(d, d, true, scale),
    }
}

fn cmix_checkpoint(d: usize, dim_ffn: usize) -> CmixCheckpoint {
    let key_scale = (d as f32).sqrt().recip();
    CmixCheckpoint {
        x_k: fp16_zeros(d),
        key: packed_linear(dim_ffn, d, false, key_scale),
        value_bits: fp16_zeros(dim_ffn.saturating_mul(d)),
    }
}

fn tmix_checkpoint(cfg: &TrainConfig) -> TmixCheckpoint {
    let d = cfg.d_model;
    let rank = cfg.resolved_tmix_lora_rank();
    let heads = d / cfg.head_size.max(1);
    TmixCheckpoint {
        x_r: fp16_zeros(d),
        x_w: fp16_zeros(d),
        x_k: fp16_zeros(d),
        x_v: fp16_zeros(d),
        x_a: fp16_zeros(d),
        x_g: fp16_zeros(d),
        w1: fp16_zeros(d.saturating_mul(rank)),
        a1: fp16_zeros(d.saturating_mul(rank)),
        v1: fp16_zeros(d.saturating_mul(rank)),
        g1: fp16_zeros(d.saturating_mul(rank)),
        w2: fp16_zeros(rank.saturating_mul(d)),
        a2: fp16_zeros(rank.saturating_mul(d)),
        v2: fp16_zeros(rank.saturating_mul(d)),
        g2: fp16_zeros(rank.saturating_mul(d)),
        w0: fp16_zeros(d),
        a0: fp16_zeros(d),
        v0: fp16_zeros(d),
        k_k: fp16_zeros(d),
        k_a: fp16_zeros(d),
        r_k: fp16_zeros(heads.saturating_mul(cfg.head_size)),
        receptance: fp16_zeros(d.saturating_mul(d)),
        key: fp16_zeros(d.saturating_mul(d)),
        value: fp16_zeros(d.saturating_mul(d)),
        output: fp16_zeros(d.saturating_mul(d)),
        ln_x_weight: ones_vec(d),
        ln_x_bias: fp16_zeros(d),
    }
}

fn ones_vec(len: usize) -> Fp16Vec {
    Fp16Vec(ones(len))
}

fn packed_words(out: usize, in_features: usize) -> usize {
    out.saturating_mul(in_features).div_ceil(32)
}

fn check_packed(
    matrix: &PackedBinaryCheckpoint,
    out: usize,
    in_features: usize,
    bias: bool,
) -> Result<()> {
    if matrix.bits.len() != packed_words(out, in_features) || matrix.scale_bits.len() != out {
        bail!("checkpoint packed-linear shape mismatch");
    }
    match (&matrix.bias_bits, bias) {
        (Some(bias_bits), true) if bias_bits.len() == out => Ok(()),
        (None, false) => Ok(()),
        _ => bail!("checkpoint packed-linear bias mismatch"),
    }
}

fn check_ln(ln: &LayerNormBits, d: usize) -> Result<()> {
    if ln.weight.len() != d || ln.bias.len() != d {
        bail!("checkpoint LayerNorm shape mismatch");
    }
    Ok(())
}

fn check_vec(vec: &Fp16Vec, len: usize, name: &str) -> Result<()> {
    if vec.0.len() != len {
        bail!("checkpoint {name} length mismatch");
    }
    Ok(())
}

fn check_rosa(rosa: &RosaCheckpoint, d: usize) -> Result<()> {
    check_vec(&rosa.x_q, d, "x_q")?;
    check_vec(&rosa.x_k, d, "x_k")?;
    check_vec(&rosa.x_v, d, "x_v")?;
    check_vec(&rosa.e, d, "e")?;
    check_packed(&rosa.q, d, d, true)?;
    check_packed(&rosa.k, d, d, true)?;
    check_packed(&rosa.v, d, d, true)?;
    check_packed(&rosa.o, d, d, true)?;
    Ok(())
}

fn check_cmix(ffn: &CmixCheckpoint, d: usize, dim_ffn: usize) -> Result<()> {
    check_vec(&ffn.x_k, d, "cmix.x_k")?;
    check_packed(&ffn.key, dim_ffn, d, false)?;
    check_vec(&ffn.value_bits, dim_ffn.saturating_mul(d), "cmix.value")?;
    Ok(())
}

fn validate_checkpoint_shapes(checkpoint: &ModelCheckpoint) -> Result<()> {
    let cfg = &checkpoint.config;
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let dim_ffn = cfg.resolved_dim_ffn();
    if checkpoint.embedding_bits.len() != v.saturating_mul(d) {
        bail!("checkpoint embedding shape does not match its configuration");
    }
    check_ln(&checkpoint.ln_out, d)?;
    check_packed(&checkpoint.head, v, d, false)?;
    match cfg.architecture {
        Architecture::Heron => {
            if checkpoint.blocks.len() != cfg.n_layers || !checkpoint.hybrid_blocks.is_empty() {
                bail!("heron checkpoint must store exactly n_layers Heron blocks");
            }
            for (index, block) in checkpoint.blocks.iter().enumerate() {
                match (&block.ln0, index) {
                    (Some(ln0), 0) => check_ln(ln0, d)?,
                    (None, _) if index > 0 => {}
                    _ => bail!("ln0 must be present only on Heron layer 0"),
                }
                check_ln(&block.ln2, d)?;
                check_ln(&block.ln3, d)?;
                check_rosa(&block.rosa, d)?;
                check_cmix(&block.ffn, d, dim_ffn)?;
            }
        }
        Architecture::RosaRwkv7 => {
            if checkpoint.hybrid_blocks.len() != cfg.n_layers || !checkpoint.blocks.is_empty() {
                bail!("rosa_rwkv7 checkpoint must store exactly n_layers hybrid blocks");
            }
            let rank = cfg.resolved_tmix_lora_rank();
            let heads = d / cfg.head_size;
            for block in &checkpoint.hybrid_blocks {
                check_ln(&block.ln_a, d)?;
                check_ln(&block.ln_b, d)?;
                check_ln(&block.ln_c, d)?;
                check_rosa(&block.rosa, d)?;
                check_cmix(&block.ffn, d, dim_ffn)?;
                let tmix = &block.tmix;
                check_vec(&tmix.w1, d.saturating_mul(rank), "tmix.w1")?;
                check_vec(&tmix.w2, rank.saturating_mul(d), "tmix.w2")?;
                check_vec(&tmix.receptance, d.saturating_mul(d), "tmix.receptance")?;
                check_vec(&tmix.r_k, heads.saturating_mul(cfg.head_size), "tmix.r_k")?;
                check_vec(&tmix.ln_x_weight, d, "tmix.ln_x")?;
            }
        }
    }
    Ok(())
}

fn skeleton_checkpoint(cfg: &TrainConfig) -> Result<ModelCheckpoint> {
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let dim_ffn = cfg.resolved_dim_ffn();
    let head_scale = (d as f32).sqrt().recip();
    let (blocks, hybrid_blocks) = match cfg.architecture {
        Architecture::Heron => {
            let blocks = (0..cfg.n_layers)
                .map(|layer| HeronBlockCheckpoint {
                    ln0: (layer == 0).then(|| layer_norm(d)),
                    ln2: layer_norm(d),
                    ln3: layer_norm(d),
                    rosa: rosa_checkpoint(d),
                    ffn: cmix_checkpoint(d, dim_ffn),
                })
                .collect();
            (blocks, Vec::new())
        }
        Architecture::RosaRwkv7 => {
            let hybrid = (0..cfg.n_layers)
                .map(|_| HybridBlockCheckpoint {
                    ln_a: layer_norm(d),
                    ln_b: layer_norm(d),
                    ln_c: layer_norm(d),
                    tmix: tmix_checkpoint(cfg),
                    rosa: rosa_checkpoint(d),
                    ffn: cmix_checkpoint(d, dim_ffn),
                })
                .collect();
            (Vec::new(), hybrid)
        }
    };
    let checkpoint = ModelCheckpoint {
        format_version: CHECKPOINT_FORMAT_VERSION,
        config: cfg.clone(),
        embedding_bits: zeros(v.saturating_mul(d)),
        ln_out: layer_norm(d),
        head: packed_linear(v, d, false, head_scale),
        blocks,
        hybrid_blocks,
    };
    validate_checkpoint_shapes(&checkpoint)?;
    Ok(checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Architecture;
    use crate::tokenizer::MIN_VOCAB;

    #[test]
    fn new_heron_emits_version_two_skeleton() {
        let model = UllisHeron::new(TrainConfig {
            vocab_size: MIN_VOCAB as usize,
            d_model: 16,
            n_layers: 1,
            dim_ffn: 64,
            context_len: 32,
            tmix_lora_rank: 8,
            ..Default::default()
        })
        .unwrap();
        let checkpoint = model.checkpoint();
        assert_eq!(checkpoint.format_version, 2);
        assert_eq!(checkpoint.blocks.len(), 1);
        assert!(checkpoint.blocks[0].ln0.is_some());
        assert!(checkpoint.head.bias_bits.is_none());
        assert!(checkpoint.blocks[0].rosa.q.bias_bits.is_some());
        let restored = UllisHeron::from_checkpoint(checkpoint).unwrap();
        assert_eq!(restored.cfg.d_model, 16);
        let hidden = restored.hidden(&[4, 5, 6, 7], 1, 4).unwrap();
        assert_eq!(hidden.len(), 4 * 16);
        assert!(hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hyena_v1_checkpoints_are_rejected() {
        let checkpoint = ModelCheckpoint {
            format_version: 1,
            config: TrainConfig::default(),
            embedding_bits: Vec::new(),
            ln_out: layer_norm(1),
            head: packed_linear(1, 1, false, 1.0),
            blocks: Vec::new(),
            hybrid_blocks: Vec::new(),
        };
        let error = UllisHeron::from_checkpoint(checkpoint)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Hyena checkpoints (v1)"));
    }

    #[test]
    fn train_step_is_not_wired() {
        let mut model = UllisHeron::new(TrainConfig {
            vocab_size: MIN_VOCAB as usize,
            d_model: 16,
            n_layers: 1,
            dim_ffn: 64,
            context_len: 32,
            tmix_lora_rank: 8,
            ..Default::default()
        })
        .unwrap();
        let error = model
            .train_step(&[1, 2], 1, 2, 1e-3)
            .unwrap_err()
            .to_string();
        assert_eq!(error, "Heron train not wired");
    }

    #[test]
    fn hybrid_skeleton_round_trips() {
        let model = UllisHeron::new(TrainConfig {
            architecture: Architecture::RosaRwkv7,
            vocab_size: MIN_VOCAB as usize,
            d_model: 32,
            n_layers: 2,
            dim_ffn: 128,
            context_len: 144,
            tmix_lora_rank: 8,
            ..Default::default()
        })
        .unwrap();
        let checkpoint = model.checkpoint();
        assert!(checkpoint.blocks.is_empty());
        assert_eq!(checkpoint.hybrid_blocks.len(), 2);
        UllisHeron::from_checkpoint(checkpoint).unwrap();
    }
}
