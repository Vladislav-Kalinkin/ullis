//! CPU Heron: LayerNorm, BinaryConnect linears, CMix x070, and checkpoint v2.
//!
//! Packed ±1 matrices keep FP16 latents in RAM for the life of the process.
//! Checkpoints store bits, learned scales, and bias only.

use crate::config::{Architecture, RosaGradMode, TrainConfig};
use crate::precision::{Fp16, Fp16Storage};
#[cfg(target_os = "macos")]
use crate::rosa::pack_bitplane;
use crate::rosa::{RosaSam, bit_from_activation, sam_workspace_bytes};
use crate::wkv7::{self, CHUNK_LEN, HEAD_SIZE};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;

/// Sentinel for next-token CE: every in-range target is supervised.
pub const CE_NO_IGNORE: u32 = u32::MAX;

/// Positions whose *target* (`row + horizon`) is `ignore_id` contribute no CE.
pub fn causal_ce_row_valid(
    row: usize,
    time: usize,
    horizon: usize,
    targets: &[u32],
    ignore_id: u32,
) -> bool {
    if time == 0 || row % time + horizon >= time {
        return false;
    }
    let target_row = row + horizon;
    target_row < targets.len() && targets[target_row] != ignore_id
}

/// Softmax `gy = p − y` scale for FP16 tensors (embeddings, LN, scales, CMix value).
///
/// Denominator is the window length `T−1`, not `n_valid`. Mean-over-valid
/// (`1/N`) makes a singleton EOS step as large as a 2000-token assistant span
/// and, with `lr=0.01` plus per-element clip `[-1,1]`, turns head scales into
/// sign-SGD. Token-sum (`1`) does the same with `scale_grms` in the thousands
/// and randomizes embeddings until CPU ROSA SAM falls off a cliff. `N = 0`
/// zeroes `gy`. Reported loss is still the mean over valid tokens.
pub fn causal_ce_gradient_scale(n_valid: usize, time: usize) -> f32 {
    if n_valid == 0 || time < 2 {
        0.0
    } else {
        1.0 / (time - 1) as f32
    }
}

/// Next-token cross-entropy statistics. Values are means over valid positions;
/// no `[batch, time, vocab]` logits tensor is retained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausalLoss {
    pub next_token: f32,
    pub next_token_count: usize,
    pub binary_flip_count: usize,
    pub loss_p10: f32,
    pub loss_p50: f32,
    pub loss_p90: f32,
    pub unigram_ce: f32,
    pub unique_targets: usize,
    pub flips_head: usize,
    pub flips_cmix: usize,
    pub flips_rosa_o: usize,
    pub embed_grad_rms: f32,
    pub head_scale_grad_rms: f32,
    pub head_scale_rms: f32,
    pub residual_abs_mean: f32,
    pub cmix_value_rms: f32,
    pub head_latent_abs_mean: f32,
    pub head_latent_step_abs: f32,
}

/// Wall-clock breakdown of one `train_step`. Phases with the same name are summed.
#[derive(Clone, Debug, Default)]
pub struct TrainStepProfile {
    pub phases_ms: Vec<(String, f64)>,
}

impl TrainStepProfile {
    fn add(&mut self, name: &str, started: Instant) {
        let ms = started.elapsed().as_secs_f64() * 1_000.0;
        if let Some((_, total)) = self.phases_ms.iter_mut().find(|(key, _)| key == name) {
            *total += ms;
        } else {
            self.phases_ms.push((name.to_string(), ms));
        }
    }

    pub fn line(&self) -> String {
        self.phases_ms
            .iter()
            .map(|(name, ms)| format!("{name}={ms:.0}ms"))
            .collect::<Vec<_>>()
            .join(" ")
    }
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

fn bit_flip_count(before: &[u32], after: &[u32]) -> usize {
    before
        .iter()
        .zip(after)
        .map(|(a, b)| (a ^ b).count_ones() as usize)
        .sum()
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

fn time_shift_on(
    x: &[f32],
    batch: usize,
    time: usize,
    channels: usize,
    metal: Option<&MetalTrainRuntime<'_>>,
) -> Result<Vec<f32>> {
    #[cfg(target_os = "macos")]
    if let Some(device) = metal {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("time-shift shape overflow"))?;
        return device.runtime.time_shift_delta(x, rows, time, channels);
    }
    let _ = metal;
    time_shift_delta(x, batch, time, channels)
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

fn time_shift_one(x: &[f32], prev: Option<&[f32]>) -> Result<Vec<f32>> {
    if let Some(prev) = prev
        && prev.len() != x.len()
    {
        bail!("time-shift prev length mismatch");
    }
    let mut xx = vec![0.0; x.len()];
    for c in 0..x.len() {
        xx[c] = match prev {
            None => -x[c],
            Some(prev) => prev[c] - x[c],
        };
    }
    Ok(xx)
}

/// Affine LayerNorm over the last dimension, `eps = 1e-5`, matching `nn.LayerNorm`.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub(crate) weight: Fp16Storage,
    pub(crate) bias: Fp16Storage,
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

    pub fn backward(
        &self,
        x: &[f32],
        gy: &[f32],
        rows: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let channels = self.channels();
        if rows.checked_mul(channels) != Some(x.len()) || gy.len() != x.len() {
            bail!("LayerNorm backward shape mismatch");
        }
        let mut gx = vec![0.0; x.len()];
        let mut gw = vec![0.0; channels];
        let mut gb = vec![0.0; channels];
        for row in 0..rows {
            let start = row * channels;
            let src = &x[start..start + channels];
            let mean = src.iter().sum::<f32>() / channels as f32;
            let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / channels as f32;
            let inv = (var + LAYER_NORM_EPS).sqrt().recip();
            let mut sum_dxhat = 0.0;
            let mut sum_dxhat_xhat = 0.0;
            let mut xhat = vec![0.0; channels];
            let mut dxhat = vec![0.0; channels];
            for c in 0..channels {
                xhat[c] = (src[c] - mean) * inv;
                dxhat[c] = gy[start + c] * self.weight.get(c);
                sum_dxhat += dxhat[c];
                sum_dxhat_xhat += dxhat[c] * xhat[c];
                gw[c] += gy[start + c] * xhat[c];
                gb[c] += gy[start + c];
            }
            let inv_n = inv / channels as f32;
            for c in 0..channels {
                gx[start + c] =
                    inv_n * (channels as f32 * dxhat[c] - sum_dxhat - xhat[c] * sum_dxhat_xhat);
            }
        }
        Ok((gx, gw, gb))
    }

    pub fn apply_clipped_sgd(
        &mut self,
        g_weight: &[f32],
        g_bias: &[f32],
        learning_rate: f32,
    ) -> Result<()> {
        if g_weight.len() != self.channels() || g_bias.len() != self.channels() {
            bail!("LayerNorm SGD shape mismatch");
        }
        for c in 0..self.channels() {
            self.weight.apply_clipped_sgd(c, g_weight[c], learning_rate);
            self.bias.apply_clipped_sgd(c, g_bias[c], learning_rate);
        }
        Ok(())
    }

    fn forward_on(
        &self,
        x: &[f32],
        rows: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            return device.runtime.layer_norm(
                x,
                &fp16_to_f32(&self.weight),
                &fp16_to_f32(&self.bias),
                rows,
                self.channels(),
            );
        }
        let _ = metal;
        self.forward(x, rows)
    }

    fn backward_on(
        &self,
        x: &[f32],
        gy: &[f32],
        rows: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let bwd = device.runtime.layer_norm_backward(
                x,
                gy,
                &fp16_to_f32(&self.weight),
                rows,
                self.channels(),
            )?;
            return Ok((bwd.input_gradient, bwd.weight_gradient, bwd.bias_gradient));
        }
        let _ = metal;
        self.backward(x, gy, rows)
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

    pub fn backward(&self, x: &[f32], gy: &[f32], rows: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let weights = self.out_features.saturating_mul(self.in_features);
        if rows.checked_mul(self.in_features) != Some(x.len())
            || rows.checked_mul(self.out_features) != Some(gy.len())
        {
            bail!("FP16 linear backward shape mismatch");
        }
        let mut gx = vec![0.0; x.len()];
        let mut gw = vec![0.0; weights];
        for row in 0..rows {
            let x_row = &x[row * self.in_features..(row + 1) * self.in_features];
            let gy_row = &gy[row * self.out_features..(row + 1) * self.out_features];
            for o in 0..self.out_features {
                let base = o * self.in_features;
                let g = gy_row[o];
                for i in 0..self.in_features {
                    gw[base + i] += g * x_row[i];
                    gx[row * self.in_features + i] += g * self.weight.get(base + i);
                }
            }
        }
        Ok((gx, gw))
    }

    pub fn apply_clipped_sgd(&mut self, g_weight: &[f32], learning_rate: f32) -> Result<()> {
        if g_weight.len() != self.weight.len() {
            bail!("FP16 linear SGD shape mismatch");
        }
        for i in 0..self.weight.len() {
            self.weight.apply_clipped_sgd(i, g_weight[i], learning_rate);
        }
        Ok(())
    }

    /// Kaiming-uniform-style init matching `nn.Linear` (`bound = 1/sqrt(fan_in)`).
    pub fn seeded(out_features: usize, in_features: usize, seed: u64) -> Result<Self> {
        let weights = out_features
            .checked_mul(in_features)
            .ok_or_else(|| anyhow::anyhow!("FP16 linear shape overflow"))?;
        let scale = (in_features as f32).sqrt().recip();
        let mut state = seed | 1;
        Ok(Self {
            out_features,
            in_features,
            weight: Fp16Storage::from_f32((0..weights).map(|_| {
                let word = splitmix64(&mut state);
                let unit = (word >> 11) as f32 / ((1_u64 << 53) as f32);
                (unit * 2.0 - 1.0) * scale
            })),
        })
    }

    pub fn weight(&self) -> &Fp16Storage {
        &self.weight
    }

    fn weight_bits(&self) -> &[u16] {
        self.weight.as_bits()
    }

    fn replace_weight(&mut self, bits: Vec<u16>, residual: Vec<f32>) -> Result<()> {
        self.weight.install(bits, residual)
    }

    fn forward_on(
        &self,
        x: &[f32],
        rows: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let shape =
                crate::metal::LinearDispatchShape::new(rows, self.in_features, self.out_features)?;
            return device
                .runtime
                .fp16_linear(x, &fp16_to_f32(&self.weight), shape);
        }
        let _ = metal;
        self.forward(x, rows)
    }

    fn backward_on(
        &self,
        x: &[f32],
        gy: &[f32],
        rows: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let shape =
                crate::metal::LinearDispatchShape::new(rows, self.in_features, self.out_features)?;
            let bwd =
                device
                    .runtime
                    .fp16_linear_backward(x, gy, &fp16_to_f32(&self.weight), shape)?;
            return Ok((bwd.input_gradient, bwd.weight_gradient));
        }
        let _ = metal;
        self.backward(x, gy, rows)
    }
}

/// Initial |latent| for a freshly packed ±1 matrix.
///
/// The forward row scale is `1/sqrt(fan_in)` and must stay that way so logit
/// variance matches a dense linear. The BinaryConnect proxy is a different
/// quantity: only `sign(latent)` is used in the forward pass. Starting near
/// zero lets a token-sum STE gradient cross the decision boundary in a
/// short run; reconstructing a checkpoint at `±1` keeps a trained sign sticky.
pub const BINARYCONNECT_INIT_ABS: f32 = 0.01;

/// Undo the window-length mean on a BinaryConnect STE so the ±0.01 proxy sees a
/// token sum. Pass `T−1` in train (same denominator as [`causal_ce_gradient_scale`]).
pub fn binaryconnect_ste_scale(slots: usize) -> f32 {
    slots as f32
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
        for index in 0..weights {
            let sign = if bit_is_plus(&bits, index) { 1.0 } else { -1.0 };
            latent.set(index, BINARYCONNECT_INIT_ABS * sign);
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
        // Checkpoints omit the proxy. Reconstruct at the BinaryConnect rails so
        // a trained sign is sticky (needs 1/lr agreeing steps to flip). Fresh
        // `seeded` matrices use BINARYCONNECT_INIT_ABS instead.
        for index in 0..weights {
            let sign = if bit_is_plus(&bits, index) { 1.0 } else { -1.0 };
            latent.set(index, sign);
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

    pub fn bits(&self) -> &[u32] {
        &self.bits
    }

    fn scale_bits(&self) -> &[u16] {
        self.scale.as_bits()
    }

    fn bias_bits(&self) -> Option<&[u16]> {
        self.bias.as_ref().map(Fp16Storage::as_bits)
    }

    fn latent_bits(&self) -> &[u16] {
        self.latent.as_bits()
    }

    fn replace_packed(
        &mut self,
        latent: Vec<u16>,
        residual: Vec<f32>,
        bits: Vec<u32>,
    ) -> Result<()> {
        if bits.len() != packed_word_count(self.latent.len()) {
            bail!("packed linear install length mismatch");
        }
        self.latent.install(latent, residual)?;
        self.bits = bits;
        Ok(())
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
        self.apply_packed_sgd(g_w, g_scale, g_bias, learning_rate, 1.0)
    }

    fn apply_packed_sgd(
        &mut self,
        g_w: &[f32],
        g_scale: &[f32],
        g_bias: Option<&[f32]>,
        learning_rate: f32,
        ste_scale: f32,
    ) -> Result<()> {
        let weights = self.out_features.saturating_mul(self.in_features);
        if g_w.len() != weights || g_scale.len() != self.out_features {
            bail!("packed linear SGD shape mismatch");
        }
        for i in 0..weights {
            self.latent
                .apply_binaryconnect_sgd(i, g_w[i] * ste_scale, learning_rate);
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

    fn flip_after_gradients(
        &mut self,
        g_w: &[f32],
        g_scale: &[f32],
        g_bias: Option<&[f32]>,
        learning_rate: f32,
        ste_scale: f32,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<usize> {
        let before = self.bits.clone();
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let (next_latent, next_residual, next_bits) = device.runtime.apply_latent_sgd(
                self.latent_bits(),
                self.latent.residual(),
                g_w,
                learning_rate,
                ste_scale,
            )?;
            self.replace_packed(next_latent, next_residual, next_bits)?;
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
            return Ok(bit_flip_count(&before, &self.bits));
        }
        #[cfg(not(target_os = "macos"))]
        let _ = metal;
        self.apply_packed_sgd(g_w, g_scale, g_bias, learning_rate, ste_scale)?;
        Ok(bit_flip_count(&before, &self.bits))
    }

    pub fn rebinarize(&mut self) {
        let weights = self.out_features.saturating_mul(self.in_features);
        self.bits = pack_plus_bits(
            (0..weights).map(|index| self.latent.get(index) >= 0.0),
            weights,
        );
    }

    fn scale_f32(&self) -> Vec<f32> {
        fp16_to_f32(&self.scale)
    }

    fn bias_f32(&self) -> Option<Vec<f32>> {
        self.bias.as_ref().map(fp16_to_f32)
    }

    fn forward_on(
        &self,
        x: &[f32],
        rows: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let shape =
                crate::metal::LinearDispatchShape::new(rows, self.in_features, self.out_features)?;
            let scale = self.scale_f32();
            let bias = self.bias_f32();
            return device
                .runtime
                .binary_linear(x, &self.bits, &scale, bias.as_deref(), shape);
        }
        let _ = metal;
        self.forward(x, rows)
    }

    fn backward_ste_on(
        &self,
        x: &[f32],
        gy: &[f32],
        rows: usize,
        g_w: &mut [f32],
        mut g_x: Option<&mut [f32]>,
        g_scale: &mut [f32],
        mut g_bias: Option<&mut [f32]>,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let shape =
                crate::metal::LinearDispatchShape::new(rows, self.in_features, self.out_features)?;
            let scale = self.scale_f32();
            let bwd = device.runtime.binary_linear_backward(
                x,
                gy,
                &self.bits,
                &scale,
                self.bias.is_some(),
                shape,
            )?;
            if g_w.len() != bwd.weight_gradient.len() || g_scale.len() != bwd.scale_gradient.len() {
                bail!("packed linear metal gradient length mismatch");
            }
            g_w.copy_from_slice(&bwd.weight_gradient);
            g_scale.copy_from_slice(&bwd.scale_gradient);
            if let Some(dst) = g_x.as_mut() {
                dst.copy_from_slice(&bwd.input_gradient);
            }
            if let (Some(dst), Some(src)) = (g_bias.as_mut(), bwd.bias_gradient.as_ref()) {
                dst.copy_from_slice(src);
            }
            return Ok(());
        }
        let _ = metal;
        self.backward_ste(x, gy, rows, g_w, g_x, g_scale, g_bias)
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
            // Official CMix is nn.Linear, not a zero matrix. A zero value blocks
            // STE into the packed key (g_key = W_value^T g_y) for the whole run.
            value: Fp16Linear::seeded(d_model, dim_ffn, seed ^ 0xC0FF_EE11)?,
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

    fn forward_one(&self, x: &[f32], prev: Option<&[f32]>) -> Result<Vec<f32>> {
        let d = self.x_k.len();
        if x.len() != d {
            bail!("CMix generate shape mismatch");
        }
        let xx = time_shift_one(x, prev)?;
        let mut shifted = vec![0.0; d];
        for c in 0..d {
            shifted[c] = x[c] + xx[c] * self.x_k.get(c);
        }
        let key = self.key.forward(&shifted, 1)?;
        let relu2: Vec<f32> = key.iter().map(|v| v.max(0.0) * v.max(0.0)).collect();
        self.value.forward(&relu2, 1)
    }

    fn forward_tape(
        &self,
        x: &[f32],
        batch: usize,
        time: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<CmixTape> {
        let d = self.x_k.len();
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let fwd = device.runtime.cmix_block_forward(
                x,
                self.x_k.as_bits(),
                self.key.bits(),
                self.key.scale_bits(),
                self.value.weight_bits(),
                batch,
                time,
                d,
                self.key.out_features,
            )?;
            return Ok(CmixTape {
                xx: fwd.xx,
                shifted: fwd.shifted,
                key: fwd.key,
                relu2: fwd.relu2,
                out: fwd.out,
            });
        }
        let xx = time_shift_on(x, batch, time, d, metal)?;
        let rows = batch.saturating_mul(time);
        let mut shifted = vec![0.0; x.len()];
        for i in 0..rows {
            for c in 0..d {
                let index = i * d + c;
                shifted[index] = x[index] + xx[index] * self.x_k.get(c);
            }
        }
        let key = self.key.forward_on(&shifted, rows, metal)?;
        let relu2 = relu2_on(&key, metal)?;
        let out = self.value.forward_on(&relu2, rows, metal)?;
        Ok(CmixTape {
            xx,
            shifted,
            key,
            relu2,
            out,
        })
    }

    fn backward_update(
        &mut self,
        tape: &CmixTape,
        gy: &[f32],
        g_x: &mut [f32],
        g_w: &mut [f32],
        learning_rate: f32,
        ste_scale: f32,
        batch: usize,
        time: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<usize> {
        let d = self.x_k.len();
        let rows = batch.saturating_mul(time);
        let key_weights = self.key.out_features.saturating_mul(self.key.in_features);
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let before = self.key.bits.clone();
            let bwd = device.runtime.cmix_block_backward_sgd(
                &tape.shifted,
                &tape.key,
                &tape.relu2,
                gy,
                self.key.bits(),
                self.key.scale_bits(),
                self.key.latent_bits(),
                self.key.latent.residual(),
                self.value.weight_bits(),
                self.value.weight.residual(),
                rows,
                d,
                self.key.out_features,
                learning_rate,
                ste_scale,
            )?;
            self.value
                .replace_weight(bwd.next_value_weight, bwd.next_value_residual)?;
            self.key.replace_packed(
                bwd.next_key_latent,
                bwd.next_key_residual,
                bwd.next_key_bits,
            )?;
            for o in 0..self.key.out_features {
                self.key
                    .scale
                    .apply_clipped_sgd(o, bwd.g_key_scale[o], learning_rate);
            }
            let flips = bit_flip_count(&before, &self.key.bits);
            let _ = g_w;
            let _ = key_weights;
            let g_shifted = bwd.g_shifted;
            let mut g_mix = vec![0.0; d];
            lerp_shift_backward(
                &tape.xx, &self.x_k, &g_shifted, batch, time, d, g_x, &mut g_mix,
            );
            for c in 0..d {
                self.x_k.apply_clipped_sgd(c, g_mix[c], learning_rate);
            }
            return Ok(flips);
        }
        let (g_relu2, g_value) = self.value.backward_on(&tape.relu2, gy, rows, metal)?;
        self.value.apply_clipped_sgd(&g_value, learning_rate)?;
        let g_key = relu2_bwd_on(&tape.key, &g_relu2, metal)?;
        g_w[..key_weights].fill(0.0);
        let mut g_scale = vec![0.0; self.key.out_features];
        let mut g_shifted = vec![0.0; tape.shifted.len()];
        self.key.backward_ste_on(
            &tape.shifted,
            &g_key,
            rows,
            &mut g_w[..key_weights],
            Some(&mut g_shifted),
            &mut g_scale,
            None,
            metal,
        )?;
        let flips = self.key.flip_after_gradients(
            &g_w[..key_weights],
            &g_scale,
            None,
            learning_rate,
            ste_scale,
            metal,
        )?;
        let mut g_mix = vec![0.0; d];
        lerp_shift_backward(
            &tape.xx, &self.x_k, &g_shifted, batch, time, d, g_x, &mut g_mix,
        );
        for c in 0..d {
            self.x_k.apply_clipped_sgd(c, g_mix[c], learning_rate);
        }
        Ok(flips)
    }
}

fn relu2_on(input: &[f32], metal: Option<&MetalTrainRuntime<'_>>) -> Result<Vec<f32>> {
    #[cfg(target_os = "macos")]
    if let Some(device) = metal {
        return device.runtime.cmix_relu2(input);
    }
    let _ = metal;
    Ok(input.iter().map(|v| v.max(0.0) * v.max(0.0)).collect())
}

fn relu2_bwd_on(
    input: &[f32],
    output_gradient: &[f32],
    metal: Option<&MetalTrainRuntime<'_>>,
) -> Result<Vec<f32>> {
    #[cfg(target_os = "macos")]
    if let Some(device) = metal {
        return device.runtime.cmix_relu2_backward(input, output_gradient);
    }
    let _ = metal;
    Ok(input
        .iter()
        .zip(output_gradient)
        .map(|(v, g)| if *v > 0.0 { 2.0 * *v * *g } else { 0.0 })
        .collect())
}

fn lerp_shift_backward(
    xx: &[f32],
    mix: &Fp16Storage,
    g_shifted: &[f32],
    batch: usize,
    time: usize,
    channels: usize,
    g_x: &mut [f32],
    g_mix: &mut [f32],
) {
    let mut g_xx = vec![0.0; g_shifted.len()];
    for b in 0..batch {
        for t in 0..time {
            for c in 0..channels {
                let index = (b * time + t) * channels + c;
                g_mix[c] += g_shifted[index] * xx[index];
                g_x[index] += g_shifted[index];
                g_xx[index] = g_shifted[index] * mix.get(c);
            }
        }
    }
    for b in 0..batch {
        for t in 0..time {
            for c in 0..channels {
                let index = (b * time + t) * channels + c;
                if t > 0 {
                    let prev = (b * time + t - 1) * channels + c;
                    g_x[prev] += g_xx[index];
                }
                g_x[index] -= g_xx[index];
            }
        }
    }
}

struct CmixTape {
    xx: Vec<f32>,
    shifted: Vec<f32>,
    key: Vec<f32>,
    relu2: Vec<f32>,
    out: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct RwkvRosaQkv1Bit {
    pub(crate) x_q: Fp16Storage,
    pub(crate) x_k: Fp16Storage,
    pub(crate) x_v: Fp16Storage,
    pub(crate) e: Fp16Storage,
    pub(crate) q: PackedBinaryLinear,
    pub(crate) k: PackedBinaryLinear,
    pub(crate) v: PackedBinaryLinear,
    pub(crate) o: PackedBinaryLinear,
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

    fn forward_one(
        &self,
        x: &[f32],
        prev: Option<&[f32]>,
        sams: &mut [RosaSam],
    ) -> Result<Vec<f32>> {
        let d = self.e.len();
        if x.len() != d || sams.len() != d {
            bail!("ROSA generate shape mismatch");
        }
        let xx = time_shift_one(x, prev)?;
        let mut q_in = vec![0.0; d];
        let mut k_in = vec![0.0; d];
        let mut v_in = vec![0.0; d];
        for c in 0..d {
            q_in[c] = x[c] + xx[c] * self.x_q.get(c);
            k_in[c] = x[c] + xx[c] * self.x_k.get(c);
            v_in[c] = x[c] + xx[c] * self.x_v.get(c);
        }
        let q = self.q.forward(&q_in, 1)?;
        let k = self.k.forward(&k_in, 1)?;
        let v = self.v.forward(&v_in, 1)?;
        let mut y = vec![0.0; d];
        for (c, sam) in sams.iter_mut().enumerate() {
            let idx = sam.push(
                bit_from_activation(q[c]),
                bit_from_activation(k[c]),
                bit_from_activation(v[c]),
            );
            y[c] = (2.0 * f32::from(idx) - 1.0) * self.e.get(c);
        }
        self.o.forward(&y, 1)
    }

    fn forward_tape(
        &self,
        x: &[f32],
        batch: usize,
        time: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<RosaTape> {
        let d = self.e.len();
        let rows = batch.saturating_mul(time);
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let q_bias = self
                .q
                .bias_bits()
                .ok_or_else(|| anyhow::anyhow!("ROSA Q bias missing"))?;
            let k_bias = self
                .k
                .bias_bits()
                .ok_or_else(|| anyhow::anyhow!("ROSA K bias missing"))?;
            let v_bias = self
                .v
                .bias_bits()
                .ok_or_else(|| anyhow::anyhow!("ROSA V bias missing"))?;
            let o_bias = self
                .o
                .bias_bits()
                .ok_or_else(|| anyhow::anyhow!("ROSA O bias missing"))?;
            let fwd = device.runtime.rosa_block_forward(
                x,
                self.x_q.as_bits(),
                self.x_k.as_bits(),
                self.x_v.as_bits(),
                self.q.bits(),
                self.q.scale_bits(),
                q_bias,
                self.k.bits(),
                self.k.scale_bits(),
                k_bias,
                self.v.bits(),
                self.v.scale_bits(),
                v_bias,
                self.e.as_bits(),
                self.o.bits(),
                self.o.scale_bits(),
                o_bias,
                batch,
                time,
                d,
            )?;
            return Ok(RosaTape {
                y: fwd.y,
                idx: fwd.idx,
                out: fwd.out,
            });
        }
        let xx = time_shift_on(x, batch, time, d, metal)?;
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
        let q = self.q.forward_on(&q_in, rows, metal)?;
        let k = self.k.forward_on(&k_in, rows, metal)?;
        let v = self.v.forward_on(&v_in, rows, metal)?;
        let (idx, y) = rosa_qkv_y(&q, &k, &v, &self.e, batch, time, d, metal)?;
        let out = self.o.forward_on(&y, rows, metal)?;
        Ok(RosaTape { y, idx, out })
    }

    fn backward_stop_grad(
        &mut self,
        tape: &RosaTape,
        gy: &[f32],
        g_w: &mut [f32],
        learning_rate: f32,
        ste_scale: f32,
        rows: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<usize> {
        let d = self.e.len();
        let weights = self.o.out_features.saturating_mul(self.o.in_features);
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let o_bias = self
                .o
                .bias_bits()
                .ok_or_else(|| anyhow::anyhow!("ROSA O bias missing"))?;
            let before = self.o.bits.clone();
            let bwd = device.runtime.rosa_o_stop_grad_sgd(
                &tape.y,
                &tape.out,
                gy,
                &tape.idx,
                self.o.bits(),
                self.o.scale_bits(),
                o_bias,
                self.o.latent_bits(),
                self.o.latent.residual(),
                rows,
                d,
                learning_rate,
                ste_scale,
            )?;
            self.o
                .replace_packed(bwd.next_latent, bwd.next_residual, bwd.next_bits)?;
            for c in 0..d {
                self.o
                    .scale
                    .apply_clipped_sgd(c, bwd.scale_gradient[c], learning_rate);
                if let Some(bias) = self.o.bias.as_mut() {
                    bias.apply_clipped_sgd(c, bwd.bias_gradient[c], learning_rate);
                }
                self.e
                    .apply_clipped_sgd(c, bwd.e_gradient[c], learning_rate);
            }
            let _ = g_w;
            let _ = weights;
            return Ok(bit_flip_count(&before, &self.o.bits));
        }
        g_w[..weights].fill(0.0);
        let mut g_scale = vec![0.0; self.o.out_features];
        let mut g_bias = vec![0.0; self.o.out_features];
        let mut g_y = vec![0.0; tape.y.len()];
        self.o.backward_ste_on(
            &tape.y,
            gy,
            rows,
            &mut g_w[..weights],
            Some(&mut g_y),
            &mut g_scale,
            Some(&mut g_bias),
            metal,
        )?;
        let flips = self.o.flip_after_gradients(
            &g_w[..weights],
            &g_scale,
            Some(&g_bias),
            learning_rate,
            ste_scale,
            metal,
        )?;
        let g_e = rosa_e_grad(&g_y, &tape.idx, d, metal)?;
        for c in 0..d {
            self.e.apply_clipped_sgd(c, g_e[c], learning_rate);
        }
        Ok(flips)
    }
}

struct RosaTape {
    y: Vec<f32>,
    idx: Vec<u8>,
    out: Vec<f32>,
}

struct MetalTrainRuntime<'a> {
    #[cfg(target_os = "macos")]
    runtime: &'a crate::metal::MetalRuntime,
    #[cfg(not(target_os = "macos"))]
    _unused: core::marker::PhantomData<&'a ()>,
}

fn rosa_qkv_y(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    e: &Fp16Storage,
    batch: usize,
    time: usize,
    d: usize,
    metal: Option<&MetalTrainRuntime<'_>>,
) -> Result<(Vec<u8>, Vec<f32>)> {
    #[cfg(target_os = "macos")]
    if let Some(device) = metal {
        let q_bits = q
            .iter()
            .copied()
            .map(bit_from_activation)
            .collect::<Vec<_>>();
        let k_bits = k
            .iter()
            .copied()
            .map(bit_from_activation)
            .collect::<Vec<_>>();
        let v_bits = v
            .iter()
            .copied()
            .map(bit_from_activation)
            .collect::<Vec<_>>();
        let e_vec: Vec<f32> = (0..d).map(|c| e.get(c)).collect();
        let fwd = device.runtime.rosa_qkv_1bit_fwd(
            &pack_bitplane(&q_bits)?,
            &pack_bitplane(&k_bits)?,
            &pack_bitplane(&v_bits)?,
            &e_vec,
            batch,
            time,
            d,
        )?;
        return Ok((fwd.idx, fwd.out));
    }
    let mut idx = vec![0_u8; batch.saturating_mul(time).saturating_mul(d)];
    let mut y = vec![0.0; idx.len()];
    for b in 0..batch {
        for c in 0..d {
            let mut sam = RosaSam::with_max_time(time);
            for t in 0..time {
                let index = (b * time + t) * d + c;
                let bit = sam.push(
                    bit_from_activation(q[index]),
                    bit_from_activation(k[index]),
                    bit_from_activation(v[index]),
                );
                idx[index] = bit;
                y[index] = (2.0 * f32::from(bit) - 1.0) * e.get(c);
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = metal;
    Ok((idx, y))
}

fn rosa_e_grad(
    gy: &[f32],
    idx: &[u8],
    channels: usize,
    metal: Option<&MetalTrainRuntime<'_>>,
) -> Result<Vec<f32>> {
    if channels == 0 || gy.len() != idx.len() || !gy.len().is_multiple_of(channels) {
        bail!("ROSA g_e shape mismatch");
    }
    let rows = gy.len() / channels;
    #[cfg(target_os = "macos")]
    if let Some(device) = metal {
        return device
            .runtime
            .rosa_qkv_1bit_bwd_e(gy, idx, 1, rows, channels);
    }
    let mut g_e = vec![0.0; channels];
    for row in 0..rows {
        for c in 0..channels {
            let index = row * channels + c;
            g_e[c] += gy[index] * (2.0 * f32::from(idx[index]) - 1.0);
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = metal;
    Ok(g_e)
}

const GROUP_NORM_EPS: f32 = 64e-5;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn gemm_right(
    x: &[f32],
    w: &Fp16Storage,
    rows: usize,
    inner: usize,
    out: usize,
) -> Result<Vec<f32>> {
    if rows.checked_mul(inner) != Some(x.len()) || w.len() != inner.saturating_mul(out) {
        bail!("LoRA gemm shape mismatch");
    }
    let mut y = vec![0.0; rows.saturating_mul(out)];
    for row in 0..rows {
        let x_row = &x[row * inner..(row + 1) * inner];
        for o in 0..out {
            let mut sum = 0.0;
            for i in 0..inner {
                sum += x_row[i] * w.get(i * out + o);
            }
            y[row * out + o] = sum;
        }
    }
    Ok(y)
}

fn group_norm(
    x: &[f32],
    weight: &Fp16Storage,
    bias: &Fp16Storage,
    rows: usize,
    channels: usize,
    groups: usize,
) -> Result<Vec<f32>> {
    if rows.checked_mul(channels) != Some(x.len())
        || weight.len() != channels
        || bias.len() != channels
        || groups == 0
        || !channels.is_multiple_of(groups)
    {
        bail!("GroupNorm shape mismatch");
    }
    let group_size = channels / groups;
    let mut y = vec![0.0; x.len()];
    for row in 0..rows {
        for g in 0..groups {
            let start = row * channels + g * group_size;
            let slice = &x[start..start + group_size];
            let mean = slice.iter().sum::<f32>() / group_size as f32;
            let var =
                slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / group_size as f32;
            let inv = (var + GROUP_NORM_EPS).sqrt().recip();
            for c in 0..group_size {
                let index = start + c;
                let channel = g * group_size + c;
                y[index] = (x[index] - mean) * inv * weight.get(channel) + bias.get(channel);
            }
        }
    }
    Ok(y)
}

fn uniform_fp16(len: usize, lo: f32, hi: f32, seed: u64) -> Fp16Storage {
    let mut state = seed | 1;
    let span = hi - lo;
    Fp16Storage::from_f32((0..len).map(|_| {
        let unit = (splitmix64(&mut state) >> 11) as f32 / ((1_u64 << 53) as f32);
        lo + span * unit
    }))
}

fn fp16_to_f32(values: &Fp16Storage) -> Vec<f32> {
    (0..values.len()).map(|i| values.get(i)).collect()
}

fn ortho_fp16(rows: usize, cols: usize, scale: f32, seed: u64) -> Fp16Storage {
    let gain = if rows > cols {
        (rows as f32 / cols as f32).sqrt()
    } else {
        1.0
    } * scale;
    let mut state = seed | 1;
    let mut m = vec![0.0; rows.saturating_mul(cols)];
    for value in &mut m {
        let unit = (splitmix64(&mut state) >> 11) as f32 / ((1_u64 << 53) as f32);
        *value = unit * 2.0 - 1.0;
    }
    let k = cols.min(rows);
    for c in 0..k {
        let mut n2 = 0.0;
        for r in 0..rows {
            n2 += m[r * cols + c] * m[r * cols + c];
        }
        let n = n2.sqrt().max(1e-8);
        for r in 0..rows {
            m[r * cols + c] /= n;
        }
        for c2 in (c + 1)..cols {
            let mut dot = 0.0;
            for r in 0..rows {
                dot += m[r * cols + c] * m[r * cols + c2];
            }
            for r in 0..rows {
                m[r * cols + c2] -= dot * m[r * cols + c];
            }
        }
    }
    for value in &mut m {
        *value *= gain;
    }
    Fp16Storage::from_f32(m)
}

/// Full `RWKV_Tmix_x070` wrapper. Weights are FP16; WKV7 itself is FP32.
#[derive(Clone, Debug)]
pub(crate) struct RwkvTmixX070 {
    layer_id: usize,
    n_head: usize,
    x_r: Fp16Storage,
    x_w: Fp16Storage,
    x_k: Fp16Storage,
    x_v: Fp16Storage,
    x_a: Fp16Storage,
    x_g: Fp16Storage,
    w1: Fp16Storage,
    a1: Fp16Storage,
    v1: Fp16Storage,
    g1: Fp16Storage,
    w2: Fp16Storage,
    a2: Fp16Storage,
    v2: Fp16Storage,
    g2: Fp16Storage,
    w0: Fp16Storage,
    a0: Fp16Storage,
    v0: Fp16Storage,
    k_k: Fp16Storage,
    k_a: Fp16Storage,
    r_k: Fp16Storage,
    receptance: Fp16Linear,
    key: Fp16Linear,
    value: Fp16Linear,
    output: Fp16Linear,
    ln_x_weight: Fp16Storage,
    ln_x_bias: Fp16Storage,
}

impl RwkvTmixX070 {
    fn seeded(
        d_model: usize,
        n_layers: usize,
        layer_id: usize,
        rank: usize,
        seed: u64,
    ) -> Result<Self> {
        if d_model == 0 || !d_model.is_multiple_of(HEAD_SIZE) {
            bail!("Tmix d_model must be a multiple of {HEAD_SIZE}");
        }
        let n_head = d_model / HEAD_SIZE;
        let n = HEAD_SIZE as f32;
        let c = d_model as f32;
        let ratio_0_to_1 = if n_layers <= 1 {
            0.0
        } else {
            layer_id as f32 / (n_layers - 1) as f32
        };
        let ratio_1_to_almost0 = 1.0 - (layer_id as f32 / n_layers.max(1) as f32);
        let mut x_r = Vec::with_capacity(d_model);
        let mut x_w = Vec::with_capacity(d_model);
        let mut x_k = Vec::with_capacity(d_model);
        let mut x_v = Vec::with_capacity(d_model);
        let mut x_a = Vec::with_capacity(d_model);
        let mut x_g = Vec::with_capacity(d_model);
        let mut w0 = Vec::with_capacity(d_model);
        let mut a0 = Vec::with_capacity(d_model);
        let mut v0 = Vec::with_capacity(d_model);
        let mut k_k = Vec::with_capacity(d_model);
        let mut k_a = Vec::with_capacity(d_model);
        for i in 0..d_model {
            let ddd = i as f32 / c;
            x_r.push(1.0 - ddd.powf(0.2 * ratio_1_to_almost0));
            x_w.push(1.0 - ddd.powf(0.9 * ratio_1_to_almost0));
            x_k.push(1.0 - ddd.powf(0.7 * ratio_1_to_almost0));
            x_v.push(1.0 - ddd.powf(0.7 * ratio_1_to_almost0));
            x_a.push(1.0 - ddd.powf(0.9 * ratio_1_to_almost0));
            x_g.push(1.0 - ddd.powf(0.2 * ratio_1_to_almost0));
            let linear = i as f32 / (c - 1.0).max(1.0) - 0.5;
            let mut zigzag = ((i as f32 % n) - ((n - 1.0) / 2.0)) / ((n - 1.0) / 2.0);
            zigzag *= zigzag.abs();
            let www =
                -6.0 + 6.0 * (i as f32 / (c - 1.0).max(1.0)).powf(1.0 + ratio_0_to_1.powf(0.3));
            w0.push(www + 0.5 + zigzag * 2.5);
            a0.push(-0.19 + zigzag * 0.3 + linear * 0.4);
            v0.push(0.73 - linear * 0.4);
            k_k.push(0.71 - linear * 0.1);
            k_a.push(1.02);
        }
        let scale = (d_model as f32).sqrt().recip();
        Ok(Self {
            layer_id,
            n_head,
            x_r: Fp16Storage::from_f32(x_r),
            x_w: Fp16Storage::from_f32(x_w),
            x_k: Fp16Storage::from_f32(x_k),
            x_v: Fp16Storage::from_f32(x_v),
            x_a: Fp16Storage::from_f32(x_a),
            x_g: Fp16Storage::from_f32(x_g),
            w1: Fp16Storage::zeros(d_model.saturating_mul(rank)),
            a1: Fp16Storage::zeros(d_model.saturating_mul(rank)),
            v1: Fp16Storage::zeros(d_model.saturating_mul(rank)),
            g1: Fp16Storage::zeros(d_model.saturating_mul(rank)),
            w2: ortho_fp16(rank, d_model, 0.1, seed ^ 0xA1),
            a2: ortho_fp16(rank, d_model, 0.1, seed ^ 0xA2),
            v2: ortho_fp16(rank, d_model, 0.1, seed ^ 0xA3),
            g2: ortho_fp16(rank, d_model, 0.1, seed ^ 0xA4),
            w0: Fp16Storage::from_f32(w0),
            a0: Fp16Storage::from_f32(a0),
            v0: Fp16Storage::from_f32(v0),
            k_k: Fp16Storage::from_f32(k_k),
            k_a: Fp16Storage::from_f32(k_a),
            r_k: Fp16Storage::from_f32((0..d_model).map(|_| -0.04)),
            receptance: Fp16Linear::from_f32(
                d_model,
                d_model,
                &fp16_to_f32(&uniform_fp16(
                    d_model * d_model,
                    -0.5 * scale,
                    0.5 * scale,
                    seed ^ 0xB1,
                )),
            )?,
            key: Fp16Linear::from_f32(
                d_model,
                d_model,
                &fp16_to_f32(&uniform_fp16(
                    d_model * d_model,
                    -0.05 * scale,
                    0.05 * scale,
                    seed ^ 0xB2,
                )),
            )?,
            value: Fp16Linear::from_f32(
                d_model,
                d_model,
                &fp16_to_f32(&uniform_fp16(
                    d_model * d_model,
                    -0.5 * scale,
                    0.5 * scale,
                    seed ^ 0xB3,
                )),
            )?,
            output: Fp16Linear::zeros(d_model, d_model)?,
            ln_x_weight: Fp16Storage::from_f32((0..d_model).map(|_| 1.0)),
            ln_x_bias: Fp16Storage::zeros(d_model),
        })
    }

    fn mix_decay(
        &self,
        xw: &[f32],
        xa: &[f32],
        xv: &[f32],
        xg: &[f32],
        rows: usize,
        d: usize,
        rank: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let w_lora = gemm_right(xw, &self.w1, rows, d, rank)?;
        let w_lora: Vec<f32> = w_lora.iter().map(|v| v.tanh()).collect();
        let w_lora = gemm_right(&w_lora, &self.w2, rows, rank, d)?;
        let a_lora = gemm_right(xa, &self.a1, rows, d, rank)?;
        let a_lora = gemm_right(&a_lora, &self.a2, rows, rank, d)?;
        let v_lora = gemm_right(xv, &self.v1, rows, d, rank)?;
        let v_lora = gemm_right(&v_lora, &self.v2, rows, rank, d)?;
        let g_lora = gemm_right(xg, &self.g1, rows, d, rank)?;
        let g_sig: Vec<f32> = g_lora.iter().copied().map(sigmoid).collect();
        let g = gemm_right(&g_sig, &self.g2, rows, rank, d)?;
        let mut w = vec![0.0; rows * d];
        let mut a = vec![0.0; rows * d];
        let mut vmix = vec![0.0; rows * d];
        for row in 0..rows {
            for c in 0..d {
                let index = row * d + c;
                w[index] = -softplus(-(self.w0.get(c) + w_lora[index])) - 0.5;
                a[index] = sigmoid(self.a0.get(c) + a_lora[index]);
                vmix[index] = sigmoid(self.v0.get(c) + v_lora[index]);
            }
        }
        Ok((w, a, vmix, g))
    }

    fn apply_value_residual(
        &self,
        v: &mut [f32],
        v_mix: &[f32],
        v_first: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        if self.layer_id == 0 {
            return Ok(v.to_vec());
        }
        let first = v_first.ok_or_else(|| anyhow::anyhow!("Tmix v_first missing after layer 0"))?;
        if first.len() != v.len() || v_mix.len() != v.len() {
            bail!("Tmix value residual shape mismatch");
        }
        for i in 0..v.len() {
            v[i] += (first[i] - v[i]) * v_mix[i];
        }
        Ok(first.to_vec())
    }

    fn run_wkv(
        &self,
        r: &[f32],
        w: &[f32],
        k: &[f32],
        v: &[f32],
        a: &[f32],
        batch: usize,
        time: usize,
        d: usize,
    ) -> Result<Vec<f32>> {
        let heads = self.n_head;
        let mut kk = vec![0.0; r.len()];
        for b in 0..batch {
            for t in 0..time {
                for h in 0..heads {
                    let mut n2 = 0.0;
                    let mut buf = [0.0; HEAD_SIZE];
                    for n in 0..HEAD_SIZE {
                        let index = ((b * time + t) * d) + h * HEAD_SIZE + n;
                        buf[n] = k[index] * self.k_k.get(h * HEAD_SIZE + n);
                        n2 += buf[n] * buf[n];
                    }
                    let inv = n2.max(1e-12).sqrt().recip();
                    for n in 0..HEAD_SIZE {
                        let index = ((b * time + t) * d) + h * HEAD_SIZE + n;
                        kk[index] = buf[n] * inv;
                    }
                }
            }
        }
        let mut k_scaled = k.to_vec();
        for i in 0..k.len() {
            let c = i % d;
            k_scaled[i] *= 1.0 + (a[i] - 1.0) * self.k_a.get(c);
        }
        let mut wkv_a = vec![0.0; r.len()];
        let mut wkv_b = vec![0.0; r.len()];
        for i in 0..r.len() {
            wkv_a[i] = -kk[i];
            wkv_b[i] = kk[i] * a[i];
        }
        let fwd = wkv7::wkv7_forward(w, r, &k_scaled, v, &wkv_a, &wkv_b, batch, time, heads)?;
        let x = group_norm(
            &fwd.y,
            &self.ln_x_weight,
            &self.ln_x_bias,
            batch.saturating_mul(time),
            d,
            heads,
        )?;
        let mut extra = vec![0.0; x.len()];
        for b in 0..batch {
            for t in 0..time {
                for h in 0..heads {
                    let mut rk = 0.0;
                    for n in 0..HEAD_SIZE {
                        let index = ((b * time + t) * d) + h * HEAD_SIZE + n;
                        rk += r[index] * k_scaled[index] * self.r_k.get(h * HEAD_SIZE + n);
                    }
                    for n in 0..HEAD_SIZE {
                        let index = ((b * time + t) * d) + h * HEAD_SIZE + n;
                        extra[index] = rk * v[index];
                    }
                }
            }
        }
        add_inplace(&mut extra, &x);
        Ok(extra)
    }

    fn lerp_streams(
        &self,
        x: &[f32],
        batch: usize,
        time: usize,
        prev: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let d = self.x_r.len();
        let rows = batch.saturating_mul(time);
        let xx = if time == 1 {
            time_shift_one(x, prev)?
        } else {
            time_shift_delta(x, batch, time, d)?
        };
        let mut xr = vec![0.0; x.len()];
        let mut xw = vec![0.0; x.len()];
        let mut xk = vec![0.0; x.len()];
        let mut xv = vec![0.0; x.len()];
        let mut xa = vec![0.0; x.len()];
        let mut xg = vec![0.0; x.len()];
        for i in 0..rows {
            for c in 0..d {
                let index = i * d + c;
                xr[index] = x[index] + xx[index] * self.x_r.get(c);
                xw[index] = x[index] + xx[index] * self.x_w.get(c);
                xk[index] = x[index] + xx[index] * self.x_k.get(c);
                xv[index] = x[index] + xx[index] * self.x_v.get(c);
                xa[index] = x[index] + xx[index] * self.x_a.get(c);
                xg[index] = x[index] + xx[index] * self.x_g.get(c);
            }
        }
        Ok((xr, xw, xk, xv, xa, xg))
    }

    fn finish(&self, x: &[f32], g: &[f32], rows: usize) -> Result<Vec<f32>> {
        let mut gated = vec![0.0; x.len()];
        for (dst, (xx, gg)) in gated.iter_mut().zip(x.iter().zip(g)) {
            *dst = *xx * *gg;
        }
        self.output.forward(&gated, rows)
    }

    fn forward(
        &self,
        x: &[f32],
        v_first: Option<&[f32]>,
        batch: usize,
        time: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.x_r.len();
        let rows = batch.saturating_mul(time);
        let rank = self.w1.len().checked_div(d).unwrap_or(8);
        let (xr, xw, xk, xv, xa, xg) = self.lerp_streams(x, batch, time, None)?;
        let r = self.receptance.forward(&xr, rows)?;
        let (w, a, v_mix, g) = self.mix_decay(&xw, &xa, &xv, &xg, rows, d, rank)?;
        let k = self.key.forward(&xk, rows)?;
        let mut v = self.value.forward(&xv, rows)?;
        let new_v_first = self.apply_value_residual(&mut v, &v_mix, v_first)?;
        let mut x_wkv = self.run_wkv(&r, &w, &k, &v, &a, batch, time, d)?;
        x_wkv = self.finish(&x_wkv, &g, rows)?;
        Ok((x_wkv, new_v_first))
    }

    fn forward_one(
        &self,
        x: &[f32],
        prev: Option<&[f32]>,
        v_first: Option<&[f32]>,
        wkv_state: &mut [f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.x_r.len();
        let rank = self.w1.len().checked_div(d).unwrap_or(8);
        let (xr, xw, xk, xv, xa, xg) = self.lerp_streams(x, 1, 1, prev)?;
        let r = self.receptance.forward(&xr, 1)?;
        let (w, a, v_mix, g) = self.mix_decay(&xw, &xa, &xv, &xg, 1, d, rank)?;
        let mut k = self.key.forward(&xk, 1)?;
        let mut v = self.value.forward(&xv, 1)?;
        let new_v_first = self.apply_value_residual(&mut v, &v_mix, v_first)?;
        let heads = self.n_head;
        let mut kk = vec![0.0; d];
        for h in 0..heads {
            let mut n2 = 0.0;
            let mut buf = [0.0; HEAD_SIZE];
            for n in 0..HEAD_SIZE {
                let index = h * HEAD_SIZE + n;
                buf[n] = k[index] * self.k_k.get(index);
                n2 += buf[n] * buf[n];
            }
            let inv = n2.max(1e-12).sqrt().recip();
            for n in 0..HEAD_SIZE {
                kk[h * HEAD_SIZE + n] = buf[n] * inv;
            }
        }
        for i in 0..d {
            k[i] *= 1.0 + (a[i] - 1.0) * self.k_a.get(i);
        }
        let mut wkv_a = vec![0.0; d];
        let mut wkv_b = vec![0.0; d];
        for i in 0..d {
            wkv_a[i] = -kk[i];
            wkv_b[i] = kk[i] * a[i];
        }
        let y = wkv7::wkv7_step(&w, &r, &k, &v, &wkv_a, &wkv_b, wkv_state, heads)?;
        let x_n = group_norm(&y, &self.ln_x_weight, &self.ln_x_bias, 1, d, heads)?;
        let mut extra = vec![0.0; d];
        for h in 0..heads {
            let mut rk = 0.0;
            for n in 0..HEAD_SIZE {
                let index = h * HEAD_SIZE + n;
                rk += r[index] * k[index] * self.r_k.get(index);
            }
            for n in 0..HEAD_SIZE {
                let index = h * HEAD_SIZE + n;
                extra[index] = rk * v[index];
            }
        }
        add_inplace(&mut extra, &x_n);
        let out = self.finish(&extra, &g, 1)?;
        Ok((out, new_v_first))
    }
}

#[derive(Clone, Debug)]
struct HybridBlock {
    ln_a: LayerNorm,
    ln_b: LayerNorm,
    ln_c: LayerNorm,
    tmix: RwkvTmixX070,
    rosa: RwkvRosaQkv1Bit,
    ffn: RwkvCMixX070,
}

impl HybridBlock {
    fn forward(
        &self,
        x: &[f32],
        v_first: Option<&[f32]>,
        batch: usize,
        time: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let rows = batch.saturating_mul(time);
        let xr = self
            .rosa
            .forward(&self.ln_c.forward(x, rows)?, batch, time)?;
        let (xx, v_first) =
            self.tmix
                .forward(&self.ln_a.forward(x, rows)?, v_first, batch, time)?;
        let mut h = x.to_vec();
        add_inplace(&mut h, &xx);
        add_inplace(&mut h, &xr);
        let ffn = self
            .ffn
            .forward(&self.ln_b.forward(&h, rows)?, batch, time)?;
        add_inplace(&mut h, &ffn);
        Ok((h, v_first))
    }

    fn forward_one(
        &self,
        x: &[f32],
        layer: &mut HeronLayerGenerateState,
        v_first: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let rosa_in = self.ln_c.forward(x, 1)?;
        let xr = self
            .rosa
            .forward_one(&rosa_in, layer.rosa_x_prev.as_deref(), &mut layer.sams)?;
        layer.rosa_x_prev = Some(rosa_in);
        let tmix_in = self.ln_a.forward(x, 1)?;
        let state = layer
            .wkv_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("hybrid generate is missing WKV state"))?;
        let (xx, v_first) =
            self.tmix
                .forward_one(&tmix_in, layer.tmix_x_prev.as_deref(), v_first, state)?;
        layer.tmix_x_prev = Some(tmix_in);
        let mut h = x.to_vec();
        add_inplace(&mut h, &xx);
        add_inplace(&mut h, &xr);
        let ffn_in = self.ln_b.forward(&h, 1)?;
        let ffn = self
            .ffn
            .forward_one(&ffn_in, layer.cmix_x_prev.as_deref())?;
        layer.cmix_x_prev = Some(ffn_in);
        add_inplace(&mut h, &ffn);
        Ok((h, v_first))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HeronBlock {
    ln0: Option<LayerNorm>,
    ln2: LayerNorm,
    pub(crate) ln3: LayerNorm,
    pub(crate) rosa: RwkvRosaQkv1Bit,
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

    fn forward_one(&self, x: &[f32], layer: &mut HeronLayerGenerateState) -> Result<Vec<f32>> {
        let mut h = if let Some(ln0) = &self.ln0 {
            ln0.forward(x, 1)?
        } else {
            x.to_vec()
        };
        let rosa_in = self.ln3.forward(&h, 1)?;
        let rosa_out =
            self.rosa
                .forward_one(&rosa_in, layer.rosa_x_prev.as_deref(), &mut layer.sams)?;
        layer.rosa_x_prev = Some(rosa_in);
        add_inplace(&mut h, &rosa_out);
        let ffn_in = self.ln2.forward(&h, 1)?;
        let ffn_out = self
            .ffn
            .forward_one(&ffn_in, layer.cmix_x_prev.as_deref())?;
        layer.cmix_x_prev = Some(ffn_in);
        add_inplace(&mut h, &ffn_out);
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    blocks: Vec<HeronBlockCheckpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    hybrid_blocks: Vec<HybridBlockCheckpoint>,
}

/// Persistent-parameter census for `ullis inspect`. Counts are elements, not bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParamCounts {
    pub embedding: usize,
    pub packed_bits: usize,
    pub fp16_matrices: usize,
    pub layer_norm: usize,
    pub rosa_e: usize,
    /// Time-shift / LoRA-side vectors that are FP16 but not dense matrices.
    pub fp16_vectors: usize,
}

/// Generate-time working set besides weights: one SAM per layer plus last-token shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceStateBytes {
    pub sam_bytes: usize,
    pub time_shift_bytes: usize,
}

/// Structured `ullis inspect` payload: config, param split, checkpoint version, SAM+shift.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointInspect {
    pub format_version: u32,
    pub architecture: Architecture,
    pub config: TrainConfig,
    pub param_counts: ParamCounts,
    pub inference_state: InferenceStateBytes,
}

fn hyena_unloadable(version: u64) -> anyhow::Error {
    anyhow::anyhow!(
        "Hyena checkpoints (v1) are intentionally unloadable after the RWKV-8 cut (got format_version {version})"
    )
}

fn require_v2(version: u64) -> Result<()> {
    if version != u64::from(CHECKPOINT_FORMAT_VERSION) {
        return Err(hyena_unloadable(version));
    }
    Ok(())
}

/// Online generate/chat state: one SAM per channel per layer, plus time-shift
/// inputs. Hybrid layers also keep WKV state `[H,N,N]` FP32.
#[derive(Clone, Debug)]
pub struct HeronGenerateState {
    layers: Vec<HeronLayerGenerateState>,
    time: usize,
}

#[derive(Clone, Debug)]
struct HeronLayerGenerateState {
    sams: Vec<RosaSam>,
    rosa_x_prev: Option<Vec<f32>>,
    cmix_x_prev: Option<Vec<f32>>,
    tmix_x_prev: Option<Vec<f32>>,
    wkv_state: Option<Vec<f32>>,
}

impl HeronGenerateState {
    pub fn time(&self) -> usize {
        self.time
    }
}

/// Product model. Packed BinaryConnect latents live in RAM; checkpoints omit them.
#[derive(Clone, Debug)]
pub struct UllisHeron {
    pub cfg: TrainConfig,
    embedding: Fp16Storage,
    pub(crate) blocks: Vec<HeronBlock>,
    hybrid_blocks: Vec<HybridBlock>,
    ln_out: LayerNorm,
    pub(crate) head: PackedBinaryLinear,
    g_w: Vec<f32>,
    last_profile: Option<TrainStepProfile>,
}

impl UllisHeron {
    pub fn new(cfg: TrainConfig) -> Result<Self> {
        cfg.validate()?;
        match cfg.architecture {
            Architecture::Heron => Self::new_heron(cfg),
            Architecture::RosaRwkv7 => Self::new_hybrid(cfg),
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
            hybrid_blocks: Vec::new(),
            ln_out: LayerNorm::new(d),
            head,
            g_w: vec![0.0; max_matrix],
            last_profile: None,
        })
    }

    fn new_hybrid(cfg: TrainConfig) -> Result<Self> {
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let dim_ffn = cfg.resolved_dim_ffn();
        let rank = cfg.resolved_tmix_lora_rank();
        let embedding = seeded_embedding(v.saturating_mul(d), d, cfg.seed);
        let hybrid_blocks = (0..cfg.n_layers)
            .map(|layer| {
                Ok(HybridBlock {
                    ln_a: LayerNorm::new(d),
                    ln_b: LayerNorm::new(d),
                    ln_c: LayerNorm::new(d),
                    tmix: RwkvTmixX070::seeded(
                        d,
                        cfg.n_layers,
                        layer,
                        rank,
                        cfg.seed.wrapping_add(layer as u64 + 3),
                    )?,
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
            blocks: Vec::new(),
            hybrid_blocks,
            ln_out: LayerNorm::new(d),
            head,
            g_w: vec![0.0; max_matrix],
            last_profile: None,
        })
    }

    pub fn last_step_profile(&self) -> Option<&TrainStepProfile> {
        self.last_profile.as_ref()
    }

    pub fn gradient_workspace(&self) -> &[f32] {
        &self.g_w
    }

    pub fn checkpoint(&self) -> ModelCheckpoint {
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
                    rosa: rosa_live_checkpoint(&block.rosa),
                    ffn: cmix_live_checkpoint(&block.ffn),
                })
                .collect(),
            hybrid_blocks: self
                .hybrid_blocks
                .iter()
                .map(|block| HybridBlockCheckpoint {
                    ln_a: LayerNormBits {
                        weight: block.ln_a.weight.as_bits().to_vec(),
                        bias: block.ln_a.bias.as_bits().to_vec(),
                    },
                    ln_b: LayerNormBits {
                        weight: block.ln_b.weight.as_bits().to_vec(),
                        bias: block.ln_b.bias.as_bits().to_vec(),
                    },
                    ln_c: LayerNormBits {
                        weight: block.ln_c.weight.as_bits().to_vec(),
                        bias: block.ln_c.bias.as_bits().to_vec(),
                    },
                    tmix: tmix_live_checkpoint(&block.tmix),
                    rosa: rosa_live_checkpoint(&block.rosa),
                    ffn: cmix_live_checkpoint(&block.ffn),
                })
                .collect(),
        }
    }

    pub fn from_checkpoint(checkpoint: ModelCheckpoint) -> Result<Self> {
        require_v2(u64::from(checkpoint.format_version))?;
        checkpoint.config.validate()?;
        validate_checkpoint_shapes(&checkpoint)?;
        match checkpoint.config.architecture {
            Architecture::RosaRwkv7 => hybrid_from_checkpoint(checkpoint),
            Architecture::Heron => heron_from_checkpoint(checkpoint),
        }
    }

    pub fn hidden(&self, ids: &[u32], batch: usize, time: usize) -> Result<Vec<f32>> {
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
        if !self.hybrid_blocks.is_empty() && !time.is_multiple_of(CHUNK_LEN) {
            bail!("rosa_rwkv7 hidden requires time multiple of {CHUNK_LEN}");
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
        if self.hybrid_blocks.is_empty() {
            for block in &self.blocks {
                x = block.forward(&x, batch, time)?;
            }
        } else {
            let mut v_first: Option<Vec<f32>> = None;
            for block in &self.hybrid_blocks {
                let (h, vf) = block.forward(&x, v_first.as_deref(), batch, time)?;
                x = h;
                v_first = Some(vf);
            }
        }
        self.ln_out.forward(&x, rows)
    }

    pub fn logits(&self, ids: &[u32], batch: usize, time: usize) -> Result<Vec<f32>> {
        let hidden = self.hidden(ids, batch, time)?;
        self.head.forward(&hidden, batch.saturating_mul(time))
    }

    /// Empty SAM (`trans*=-1`, `fail[0]=-1`) and no time-shift history.
    pub fn generate_state(&self) -> Result<HeronGenerateState> {
        let d = self.cfg.d_model;
        let max_time = self.cfg.context_len;
        let n_layers = if self.hybrid_blocks.is_empty() {
            self.blocks.len()
        } else {
            self.hybrid_blocks.len()
        };
        let heads = d / self.cfg.head_size.max(1);
        Ok(HeronGenerateState {
            layers: (0..n_layers)
                .map(|_| HeronLayerGenerateState {
                    sams: (0..d).map(|_| RosaSam::with_max_time(max_time)).collect(),
                    rosa_x_prev: None,
                    cmix_x_prev: None,
                    tmix_x_prev: None,
                    wkv_state: (!self.hybrid_blocks.is_empty()).then(|| {
                        vec![0.0; heads.saturating_mul(HEAD_SIZE).saturating_mul(HEAD_SIZE)]
                    }),
                })
                .collect(),
            time: 0,
        })
    }

    /// One token: same `RosaSam::push` as train, greedy-ready last-token logits.
    pub fn generate_step(&self, state: &mut HeronGenerateState, token: u32) -> Result<Vec<f32>> {
        if token as usize >= self.cfg.vocab_size {
            bail!("token id exceeds vocabulary");
        }
        if state.time >= self.cfg.context_len {
            bail!("generate exceeded context_len ({})", self.cfg.context_len);
        }
        let n_layers = if self.hybrid_blocks.is_empty() {
            self.blocks.len()
        } else {
            self.hybrid_blocks.len()
        };
        if state.layers.len() != n_layers {
            bail!("generate state does not match this model");
        }
        let d = self.cfg.d_model;
        let mut x = vec![0.0; d];
        let offset = token as usize * d;
        for c in 0..d {
            x[c] = self.embedding.get(offset + c);
        }
        if self.hybrid_blocks.is_empty() {
            for (block, layer) in self.blocks.iter().zip(&mut state.layers) {
                x = block.forward_one(&x, layer)?;
            }
        } else {
            let mut v_first: Option<Vec<f32>> = None;
            for (block, layer) in self.hybrid_blocks.iter().zip(&mut state.layers) {
                let (h, vf) = block.forward_one(&x, layer, v_first.as_deref())?;
                x = h;
                v_first = Some(vf);
            }
        }
        let hidden = self.ln_out.forward(&x, 1)?;
        state.time += 1;
        self.head.forward(&hidden, 1)
    }

    pub fn train_step(
        &mut self,
        tokens: &[u32],
        batch: usize,
        time: usize,
        learning_rate: f32,
    ) -> Result<CausalLoss> {
        self.train_step_on(
            tokens,
            tokens,
            CE_NO_IGNORE,
            batch,
            time,
            learning_rate,
            None,
        )
    }

    pub fn train_step_with_labels(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        ignore_id: u32,
        batch: usize,
        time: usize,
        learning_rate: f32,
    ) -> Result<CausalLoss> {
        self.train_step_on(tokens, labels, ignore_id, batch, time, learning_rate, None)
    }

    #[cfg(target_os = "macos")]
    pub fn train_step_metal(
        &mut self,
        runtime: &crate::metal::MetalRuntime,
        tokens: &[u32],
        batch: usize,
        time: usize,
        learning_rate: f32,
    ) -> Result<CausalLoss> {
        self.train_step_metal_with_labels(
            runtime,
            tokens,
            tokens,
            CE_NO_IGNORE,
            batch,
            time,
            learning_rate,
        )
    }

    #[cfg(target_os = "macos")]
    pub fn train_step_metal_with_labels(
        &mut self,
        runtime: &crate::metal::MetalRuntime,
        tokens: &[u32],
        labels: &[u32],
        ignore_id: u32,
        batch: usize,
        time: usize,
        learning_rate: f32,
    ) -> Result<CausalLoss> {
        self.train_step_on(
            tokens,
            labels,
            ignore_id,
            batch,
            time,
            learning_rate,
            Some(MetalTrainRuntime { runtime }),
        )
    }

    fn train_step_on(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        ignore_id: u32,
        batch: usize,
        time: usize,
        learning_rate: f32,
        metal: Option<MetalTrainRuntime<'_>>,
    ) -> Result<CausalLoss> {
        self.cfg.optimizer.require_train_step()?;
        if !matches!(self.cfg.rosa_grad, RosaGradMode::StopGradBits) {
            bail!("only rosa_grad=stop_grad_bits is wired");
        }
        if !self.hybrid_blocks.is_empty() {
            bail!("rosa_rwkv7 train is not wired");
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 || time < 2 {
            bail!("train step requires a positive learning rate and time >= 2");
        }
        let d = self.cfg.d_model;
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("token shape overflow"))?;
        if batch == 0
            || batch > self.cfg.batch_size
            || tokens.len() != rows
            || labels.len() != rows
            || time > self.cfg.context_len
            || tokens.iter().any(|&id| id as usize >= self.cfg.vocab_size)
            || labels
                .iter()
                .any(|&id| id != ignore_id && id as usize >= self.cfg.vocab_size)
        {
            bail!("token shape or context length is invalid");
        }

        let metal_ref = metal.as_ref();
        let mut profile = TrainStepProfile::default();
        let mut x = vec![0.0; rows.saturating_mul(d)];
        let started = Instant::now();
        for (row, &id) in tokens.iter().enumerate() {
            let offset = id as usize * d;
            for c in 0..d {
                x[row * d + c] = self.embedding.get(offset + c);
            }
        }
        profile.add("embed", started);

        let mut tapes = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let ln0_in = x.clone();
            let started = Instant::now();
            if let Some(ln0) = &block.ln0 {
                x = ln0.forward_on(&x, rows, metal_ref)?;
            }
            let rosa_in = block.ln3.forward_on(&x, rows, metal_ref)?;
            profile.add("fwd_ln", started);
            let started = Instant::now();
            let rosa = block.rosa.forward_tape(&rosa_in, batch, time, metal_ref)?;
            add_inplace(&mut x, &rosa.out);
            profile.add("fwd_rosa", started);
            let after_rosa = x.clone();
            let started = Instant::now();
            let ln2_out = block.ln2.forward_on(&x, rows, metal_ref)?;
            let cmix = block.ffn.forward_tape(&ln2_out, batch, time, metal_ref)?;
            add_inplace(&mut x, &cmix.out);
            profile.add("fwd_cmix", started);
            tapes.push(BlockTape {
                ln0_in,
                rosa,
                after_rosa,
                cmix,
            });
        }
        let pre_ln_out = x.clone();
        let started = Instant::now();
        let hidden = self.ln_out.forward_on(&x, rows, metal_ref)?;
        profile.add("fwd_ln", started);

        let started = Instant::now();
        let (loss, mut g_x, n_valid, flips_head, head_diag) = self.apply_head_ce(
            &hidden,
            labels,
            ignore_id,
            time,
            learning_rate,
            rows,
            metal_ref,
        )?;
        // FP16 tensors keep the window-length mean. Packed latents undo it so
        // the ±0.01 BinaryConnect proxy can cross zero in a short run.
        let ste_scale = binaryconnect_ste_scale(time.saturating_sub(1));
        profile.add("head", started);
        let started = Instant::now();
        let (g_ln_in, g_w_ln, g_b_ln) =
            self.ln_out
                .backward_on(&pre_ln_out, &g_x, rows, metal_ref)?;
        self.ln_out
            .apply_clipped_sgd(&g_w_ln, &g_b_ln, learning_rate)?;
        g_x = g_ln_in;
        profile.add("bwd_ln", started);

        let mut flips_cmix = 0_usize;
        let mut flips_rosa_o = 0_usize;
        for (block, tape) in self.blocks.iter_mut().zip(tapes).rev() {
            let mut g_after_rosa = g_x.clone();
            let mut g_ln2_out = vec![0.0; rows.saturating_mul(d)];
            let started = Instant::now();
            flips_cmix += block.ffn.backward_update(
                &tape.cmix,
                &g_x,
                &mut g_ln2_out,
                &mut self.g_w,
                learning_rate,
                ste_scale,
                batch,
                time,
                metal_ref,
            )?;
            profile.add("bwd_cmix", started);
            let started = Instant::now();
            let (g_ln2_in, g_w2, g_b2) =
                block
                    .ln2
                    .backward_on(&tape.after_rosa, &g_ln2_out, rows, metal_ref)?;
            block.ln2.apply_clipped_sgd(&g_w2, &g_b2, learning_rate)?;
            add_inplace(&mut g_after_rosa, &g_ln2_in);
            profile.add("bwd_ln", started);

            let g_residual = g_after_rosa.clone();
            let started = Instant::now();
            flips_rosa_o += block.rosa.backward_stop_grad(
                &tape.rosa,
                &g_after_rosa,
                &mut self.g_w,
                learning_rate,
                ste_scale,
                rows,
                metal_ref,
            )?;
            profile.add("bwd_rosa", started);
            let started = Instant::now();
            if let Some(ln0) = &mut block.ln0 {
                let (g_in, gw, gb) = ln0.backward_on(&tape.ln0_in, &g_residual, rows, metal_ref)?;
                ln0.apply_clipped_sgd(&gw, &gb, learning_rate)?;
                g_x = g_in;
            } else {
                g_x = g_residual;
            }
            profile.add("bwd_ln", started);
        }

        let started = Instant::now();
        for (row, &id) in tokens.iter().enumerate() {
            let offset = id as usize * d;
            for c in 0..d {
                self.embedding
                    .apply_clipped_sgd(offset + c, g_x[row * d + c], learning_rate);
            }
        }
        profile.add("embed_sgd", started);
        self.last_profile = Some(profile);

        let flips = flips_head
            .saturating_add(flips_cmix)
            .saturating_add(flips_rosa_o);
        Ok(CausalLoss {
            next_token: loss,
            next_token_count: n_valid,
            binary_flip_count: flips,
            loss_p10: head_diag.loss_p10,
            loss_p50: head_diag.loss_p50,
            loss_p90: head_diag.loss_p90,
            unigram_ce: head_diag.unigram_ce,
            unique_targets: head_diag.unique_targets,
            flips_head,
            flips_cmix,
            flips_rosa_o,
            embed_grad_rms: rms(&g_x),
            head_scale_grad_rms: head_diag.head_scale_grad_rms,
            head_scale_rms: rms_fp16(&self.head.scale),
            residual_abs_mean: mean_abs(self.embedding.residual()),
            cmix_value_rms: self
                .blocks
                .first()
                .map(|block| rms_fp16(block.ffn.value.weight()))
                .unwrap_or(0.0),
            head_latent_abs_mean: mean_abs_fp16(&self.head.latent),
            head_latent_step_abs: head_diag.head_latent_step_abs,
        })
    }

    fn apply_head_ce(
        &mut self,
        hidden: &[f32],
        targets: &[u32],
        ignore_id: u32,
        time: usize,
        learning_rate: f32,
        rows: usize,
        metal: Option<&MetalTrainRuntime<'_>>,
    ) -> Result<(f32, Vec<f32>, usize, usize, HeadTrainDiag)> {
        let d = self.cfg.d_model;
        let vocab = self.head.out_features;
        let mut n_valid = 0_usize;
        for row in 0..rows {
            if causal_ce_row_valid(row, time, 1, targets, ignore_id) {
                n_valid += 1;
            }
        }
        let (unigram_ce, unique_targets) = unigram_cross_entropy(targets, time, ignore_id);
        let weights = vocab.saturating_mul(d);
        #[cfg(target_os = "macos")]
        if let Some(device) = metal {
            let before = self.head.bits.clone();
            let before_latent = self.head.latent_bits().to_vec();
            let head = device.runtime.packed_head_train_sgd(
                hidden,
                self.head.bits(),
                self.head.scale_bits(),
                self.head.latent_bits(),
                self.head.latent.residual(),
                targets,
                rows,
                time,
                d,
                vocab,
                1,
                ignore_id,
                learning_rate,
                binaryconnect_ste_scale(time.saturating_sub(1)),
            )?;
            self.head
                .replace_packed(head.next_latent, head.next_residual, head.next_bits)?;
            for o in 0..vocab {
                self.head
                    .scale
                    .apply_clipped_sgd(o, head.scale_gradient[o], learning_rate);
            }
            let flips = bit_flip_count(&before, &self.head.bits);
            let (loss_p10, loss_p50, loss_p90) =
                loss_percentiles(&head.row_loss, time, targets, ignore_id);
            let _ = weights;
            return Ok((
                head.mean_loss,
                head.hidden_gradient,
                n_valid,
                flips,
                HeadTrainDiag {
                    loss_p10,
                    loss_p50,
                    loss_p90,
                    unigram_ce,
                    unique_targets,
                    head_scale_grad_rms: rms(&head.scale_gradient),
                    head_latent_step_abs: mean_abs_delta_fp16(
                        &before_latent,
                        self.head.latent_bits(),
                    ),
                },
            ));
        }
        let _ = metal;
        let loss_scale = causal_ce_gradient_scale(n_valid, time);
        self.g_w[..weights].fill(0.0);
        let mut g_scale = vec![0.0; vocab];
        let mut gx = vec![0.0; hidden.len()];
        let mut loss_sum = 0.0;
        let mut row_loss = vec![0.0; rows];
        for row in 0..rows {
            if !causal_ce_row_valid(row, time, 1, targets, ignore_id) {
                continue;
            }
            let target = targets[row + 1] as usize;
            let h = &hidden[row * d..(row + 1) * d];
            let logits = self.head.forward(h, 1)?;
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0;
            let mut exps = vec![0.0; vocab];
            for v in 0..vocab {
                exps[v] = (logits[v] - max).exp();
                exp_sum += exps[v];
            }
            let ce = max + exp_sum.ln() - logits[target];
            row_loss[row] = ce;
            loss_sum += ce;
            let mut gy = vec![0.0; vocab];
            for v in 0..vocab {
                gy[v] = loss_scale * (exps[v] / exp_sum - f32::from(v == target));
            }
            self.head.backward_ste(
                h,
                &gy,
                1,
                &mut self.g_w[..weights],
                Some(&mut gx[row * d..(row + 1) * d]),
                &mut g_scale,
                None,
            )?;
        }
        let before_latent = self.head.latent_bits().to_vec();
        let flips = self.head.flip_after_gradients(
            &self.g_w[..weights],
            &g_scale,
            None,
            learning_rate,
            binaryconnect_ste_scale(time.saturating_sub(1)),
            metal,
        )?;
        let mean = if n_valid == 0 {
            0.0
        } else {
            loss_sum / n_valid as f32
        };
        let (loss_p10, loss_p50, loss_p90) = loss_percentiles(&row_loss, time, targets, ignore_id);
        Ok((
            mean,
            gx,
            n_valid,
            flips,
            HeadTrainDiag {
                loss_p10,
                loss_p50,
                loss_p90,
                unigram_ce,
                unique_targets,
                head_scale_grad_rms: rms(&g_scale),
                head_latent_step_abs: mean_abs_delta_fp16(&before_latent, self.head.latent_bits()),
            },
        ))
    }
}

struct HeadTrainDiag {
    loss_p10: f32,
    loss_p50: f32,
    loss_p90: f32,
    unigram_ce: f32,
    unique_targets: usize,
    head_scale_grad_rms: f32,
    head_latent_step_abs: f32,
}

fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

fn rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = values.iter().map(|v| v * v).sum();
    (sum_sq / values.len() as f32).sqrt()
}

fn rms_fp16(values: &Fp16Storage) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = (0..values.len())
        .map(|i| {
            let v = values.get(i);
            v * v
        })
        .sum();
    (sum_sq / values.len() as f32).sqrt()
}

fn mean_abs(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32
}

fn mean_abs_fp16(values: &Fp16Storage) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f32 = (0..values.len()).map(|i| values.get(i).abs()).sum();
    sum / values.len() as f32
}

fn mean_abs_delta_fp16(before: &[u16], after: &[u16]) -> f32 {
    if before.is_empty() || before.len() != after.len() {
        return 0.0;
    }
    let sum: f32 = before
        .iter()
        .zip(after)
        .map(|(a, b)| (Fp16::from_bits(*a).to_f32() - Fp16::from_bits(*b).to_f32()).abs())
        .sum();
    sum / before.len() as f32
}

fn loss_percentiles(
    row_loss: &[f32],
    time: usize,
    targets: &[u32],
    ignore_id: u32,
) -> (f32, f32, f32) {
    let mut valid: Vec<f32> = row_loss
        .iter()
        .enumerate()
        .filter(|(row, _)| causal_ce_row_valid(*row, time, 1, targets, ignore_id))
        .map(|(_, loss)| *loss)
        .filter(|loss| loss.is_finite())
        .collect();
    if valid.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    valid.sort_by(|a, b| a.total_cmp(b));
    let at = |p: f32| {
        let index = ((p * (valid.len() - 1) as f32).round() as usize).min(valid.len() - 1);
        valid[index]
    };
    (at(0.10), at(0.50), at(0.90))
}

fn unigram_cross_entropy(targets: &[u32], time: usize, ignore_id: u32) -> (f32, usize) {
    let mut counts = HashMap::new();
    let mut n_valid = 0_usize;
    for row in 0..targets.len() {
        if !causal_ce_row_valid(row, time, 1, targets, ignore_id) {
            continue;
        }
        let target = targets[row + 1];
        *counts.entry(target).or_insert(0_usize) += 1;
        n_valid += 1;
    }
    if n_valid == 0 {
        return (0.0, 0);
    }
    let n = n_valid as f32;
    let mut ce = 0.0;
    for count in counts.values() {
        let p = *count as f32 / n;
        ce -= *count as f32 * p.ln();
    }
    (ce / n, counts.len())
}

struct BlockTape {
    ln0_in: Vec<f32>,
    rosa: RosaTape,
    after_rosa: Vec<f32>,
    cmix: CmixTape,
}

impl ModelCheckpoint {
    /// Parse a JSON snapshot, hard-failing any `format_version` other than 2
    /// before the v2 payload schema is applied.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).context("checkpoint is not JSON")?;
        let version = value
            .get("format_version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        require_v2(version)?;
        let checkpoint: Self =
            serde_json::from_value(value).context("parse checkpoint v2 payload")?;
        validate_checkpoint_shapes(&checkpoint)?;
        Ok(checkpoint)
    }

    pub fn inspect(&self) -> Result<CheckpointInspect> {
        validate_checkpoint_shapes(self)?;
        let cfg = &self.config;
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let layers = cfg.n_layers;
        let dim_ffn = cfg.resolved_dim_ffn();
        let rank = cfg.resolved_tmix_lora_rank();
        let embedding = mul_count(v, d)?;
        let d2 = mul_count(d, d)?;
        let qkvo = mul_count(4, d2)?;
        let cmix_key = mul_count(dim_ffn, d)?;
        let cmix_value = mul_count(d, dim_ffn)?;
        let packed_layer = add_count(qkvo, cmix_key)?;
        let packed_bits = add_count(mul_count(v, d)?, mul_count(layers, packed_layer)?)?;
        let rosa_e = mul_count(layers, d)?;
        let (fp16_matrices, layer_norm, fp16_vectors) = match cfg.architecture {
            Architecture::Heron => {
                let mut layer_norm =
                    add_count(mul_count(2, d)?, mul_count(layers, mul_count(4, d)?)?)?;
                if layers > 0 {
                    layer_norm = add_count(layer_norm, mul_count(2, d)?)?;
                }
                (
                    mul_count(layers, cmix_value)?,
                    layer_norm,
                    mul_count(layers, mul_count(4, d)?)?,
                )
            }
            Architecture::RosaRwkv7 => {
                let lora = mul_count(8, mul_count(d, rank)?)?;
                let tmix_dense = mul_count(4, d2)?;
                (
                    mul_count(layers, add_count(cmix_value, add_count(tmix_dense, lora)?)?)?,
                    add_count(mul_count(2, d)?, mul_count(layers, mul_count(8, d)?)?)?,
                    mul_count(layers, mul_count(16, d)?)?,
                )
            }
        };
        let sam_one = sam_workspace_bytes(1, cfg.context_len, d)
            .ok_or_else(|| anyhow::anyhow!("SAM inference state overflow"))?;
        let sam_bytes = mul_count(layers, sam_one)?;
        let time_shift_bytes = mul_count(mul_count(layers, d)?, size_of::<f32>())?;
        Ok(CheckpointInspect {
            format_version: self.format_version,
            architecture: cfg.architecture,
            config: cfg.clone(),
            param_counts: ParamCounts {
                embedding,
                packed_bits,
                fp16_matrices,
                layer_norm,
                rosa_e,
                fp16_vectors,
            },
            inference_state: InferenceStateBytes {
                sam_bytes,
                time_shift_bytes,
            },
        })
    }
}

fn mul_count(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow::anyhow!("parameter count overflow"))
}

fn add_count(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| anyhow::anyhow!("parameter count overflow"))
}

fn rosa_live_checkpoint(rosa: &RwkvRosaQkv1Bit) -> RosaCheckpoint {
    RosaCheckpoint {
        x_q: Fp16Vec(rosa.x_q.as_bits().to_vec()),
        x_k: Fp16Vec(rosa.x_k.as_bits().to_vec()),
        x_v: Fp16Vec(rosa.x_v.as_bits().to_vec()),
        e: Fp16Vec(rosa.e.as_bits().to_vec()),
        q: packed_to_checkpoint(&rosa.q),
        k: packed_to_checkpoint(&rosa.k),
        v: packed_to_checkpoint(&rosa.v),
        o: packed_to_checkpoint(&rosa.o),
    }
}

fn cmix_live_checkpoint(ffn: &RwkvCMixX070) -> CmixCheckpoint {
    CmixCheckpoint {
        x_k: Fp16Vec(ffn.x_k.as_bits().to_vec()),
        key: packed_to_checkpoint(&ffn.key),
        value_bits: Fp16Vec(ffn.value.weight.as_bits().to_vec()),
    }
}

fn tmix_live_checkpoint(tmix: &RwkvTmixX070) -> TmixCheckpoint {
    TmixCheckpoint {
        x_r: Fp16Vec(tmix.x_r.as_bits().to_vec()),
        x_w: Fp16Vec(tmix.x_w.as_bits().to_vec()),
        x_k: Fp16Vec(tmix.x_k.as_bits().to_vec()),
        x_v: Fp16Vec(tmix.x_v.as_bits().to_vec()),
        x_a: Fp16Vec(tmix.x_a.as_bits().to_vec()),
        x_g: Fp16Vec(tmix.x_g.as_bits().to_vec()),
        w1: Fp16Vec(tmix.w1.as_bits().to_vec()),
        a1: Fp16Vec(tmix.a1.as_bits().to_vec()),
        v1: Fp16Vec(tmix.v1.as_bits().to_vec()),
        g1: Fp16Vec(tmix.g1.as_bits().to_vec()),
        w2: Fp16Vec(tmix.w2.as_bits().to_vec()),
        a2: Fp16Vec(tmix.a2.as_bits().to_vec()),
        v2: Fp16Vec(tmix.v2.as_bits().to_vec()),
        g2: Fp16Vec(tmix.g2.as_bits().to_vec()),
        w0: Fp16Vec(tmix.w0.as_bits().to_vec()),
        a0: Fp16Vec(tmix.a0.as_bits().to_vec()),
        v0: Fp16Vec(tmix.v0.as_bits().to_vec()),
        k_k: Fp16Vec(tmix.k_k.as_bits().to_vec()),
        k_a: Fp16Vec(tmix.k_a.as_bits().to_vec()),
        r_k: Fp16Vec(tmix.r_k.as_bits().to_vec()),
        receptance: Fp16Vec(tmix.receptance.weight.as_bits().to_vec()),
        key: Fp16Vec(tmix.key.weight.as_bits().to_vec()),
        value: Fp16Vec(tmix.value.weight.as_bits().to_vec()),
        output: Fp16Vec(tmix.output.weight.as_bits().to_vec()),
        ln_x_weight: Fp16Vec(tmix.ln_x_weight.as_bits().to_vec()),
        ln_x_bias: Fp16Vec(tmix.ln_x_bias.as_bits().to_vec()),
    }
}

fn tmix_from_checkpoint(
    tmix: TmixCheckpoint,
    d: usize,
    _rank: usize,
    layer_id: usize,
) -> Result<RwkvTmixX070> {
    Ok(RwkvTmixX070 {
        layer_id,
        n_head: d / HEAD_SIZE,
        x_r: Fp16Storage::from_bits(tmix.x_r.0),
        x_w: Fp16Storage::from_bits(tmix.x_w.0),
        x_k: Fp16Storage::from_bits(tmix.x_k.0),
        x_v: Fp16Storage::from_bits(tmix.x_v.0),
        x_a: Fp16Storage::from_bits(tmix.x_a.0),
        x_g: Fp16Storage::from_bits(tmix.x_g.0),
        w1: Fp16Storage::from_bits(tmix.w1.0),
        a1: Fp16Storage::from_bits(tmix.a1.0),
        v1: Fp16Storage::from_bits(tmix.v1.0),
        g1: Fp16Storage::from_bits(tmix.g1.0),
        w2: Fp16Storage::from_bits(tmix.w2.0),
        a2: Fp16Storage::from_bits(tmix.a2.0),
        v2: Fp16Storage::from_bits(tmix.v2.0),
        g2: Fp16Storage::from_bits(tmix.g2.0),
        w0: Fp16Storage::from_bits(tmix.w0.0),
        a0: Fp16Storage::from_bits(tmix.a0.0),
        v0: Fp16Storage::from_bits(tmix.v0.0),
        k_k: Fp16Storage::from_bits(tmix.k_k.0),
        k_a: Fp16Storage::from_bits(tmix.k_a.0),
        r_k: Fp16Storage::from_bits(tmix.r_k.0),
        receptance: Fp16Linear::from_bits(d, d, tmix.receptance.0)?,
        key: Fp16Linear::from_bits(d, d, tmix.key.0)?,
        value: Fp16Linear::from_bits(d, d, tmix.value.0)?,
        output: Fp16Linear::from_bits(d, d, tmix.output.0)?,
        ln_x_weight: Fp16Storage::from_bits(tmix.ln_x_weight.0),
        ln_x_bias: Fp16Storage::from_bits(tmix.ln_x_bias.0),
    })
}

fn hybrid_from_checkpoint(checkpoint: ModelCheckpoint) -> Result<UllisHeron> {
    let cfg = checkpoint.config.clone();
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let dim_ffn = cfg.resolved_dim_ffn();
    let rank = cfg.resolved_tmix_lora_rank();
    let embedding = Fp16Storage::from_bits(checkpoint.embedding_bits);
    let ln_out = LayerNorm::from_bits(checkpoint.ln_out.weight, checkpoint.ln_out.bias)?;
    let head = packed_from_checkpoint(checkpoint.head, v, d, false)?;
    let hybrid_blocks = checkpoint
        .hybrid_blocks
        .into_iter()
        .enumerate()
        .map(|(layer, block)| {
            Ok(HybridBlock {
                ln_a: LayerNorm::from_bits(block.ln_a.weight, block.ln_a.bias)?,
                ln_b: LayerNorm::from_bits(block.ln_b.weight, block.ln_b.bias)?,
                ln_c: LayerNorm::from_bits(block.ln_c.weight, block.ln_c.bias)?,
                tmix: tmix_from_checkpoint(block.tmix, d, rank, layer)?,
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
        blocks: Vec::new(),
        hybrid_blocks,
        ln_out,
        head,
        g_w: vec![0.0; max_matrix],
        last_profile: None,
    })
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
        hybrid_blocks: Vec::new(),
        last_profile: None,
    })
}

#[cfg(test)]
fn ones(len: usize) -> Vec<u16> {
    vec![Fp16::from_f32(1.0).to_bits(); len]
}

#[cfg(test)]
fn zeros(len: usize) -> Vec<u16> {
    vec![0; len]
}

#[cfg(test)]
fn packed_linear(out: usize, in_features: usize, bias: bool, scale: f32) -> PackedBinaryCheckpoint {
    let weights = out.saturating_mul(in_features);
    let words = weights.div_ceil(32);
    PackedBinaryCheckpoint {
        bits: vec![0; words],
        scale_bits: vec![Fp16::from_f32(scale).to_bits(); out],
        bias_bits: bias.then(|| zeros(out)),
    }
}

#[cfg(test)]
fn layer_norm(d: usize) -> LayerNormBits {
    LayerNormBits {
        weight: ones(d),
        bias: zeros(d),
    }
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
    fn causal_ce_gradient_scale_is_window_length_not_token_sum() {
        assert_eq!(causal_ce_gradient_scale(0, 2048), 0.0);
        assert_eq!(causal_ce_gradient_scale(1403, 1), 0.0);
        let scale = causal_ce_gradient_scale(1, 2048);
        assert!((scale - 1.0 / 2047.0).abs() < 1e-8);
        assert!((causal_ce_gradient_scale(2047, 2048) - scale).abs() < 1e-8);
    }

    fn tiny_train_cfg() -> TrainConfig {
        TrainConfig {
            vocab_size: MIN_VOCAB as usize,
            d_model: 16,
            n_layers: 1,
            dim_ffn: 64,
            context_len: 32,
            tmix_lora_rank: 8,
            ..Default::default()
        }
    }

    #[test]
    fn ignored_pad_targets_are_excluded_from_ce_count() {
        let mut model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let mut tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 8) as u32).collect();
        let mut labels = tokens.clone();
        for slot in 16..32 {
            tokens[slot] = 0;
            labels[slot] = 0;
        }
        let loss = model
            .train_step_with_labels(&tokens, &labels, 0, 1, 32, 0.05)
            .unwrap();
        assert_eq!(loss.next_token_count, 15);
        assert!(loss.next_token.is_finite());
    }

    #[test]
    fn window_length_ce_does_not_let_a_singleton_window_dominate() {
        let mut dense = UllisHeron::new(tiny_train_cfg()).unwrap();
        let mut sparse = dense.clone();
        let tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 8) as u32).collect();
        let dense_loss = dense.train_step(&tokens, 1, 32, 0.05).unwrap();
        let mut labels = vec![0_u32; 32];
        labels[16] = tokens[16];
        let sparse_loss = sparse
            .train_step_with_labels(&tokens, &labels, 0, 1, 32, 0.05)
            .unwrap();
        assert_eq!(dense_loss.next_token_count, 31);
        assert_eq!(sparse_loss.next_token_count, 1);
        assert!(
            dense_loss.embed_grad_rms > sparse_loss.embed_grad_rms,
            "dense embed_grms={} must exceed singleton embed_grms={}",
            dense_loss.embed_grad_rms,
            sparse_loss.embed_grad_rms
        );
        assert!(
            dense_loss.head_scale_grad_rms < 50.0,
            "window-length CE must not turn head scales into sign-SGD, scale_grms={}",
            dense_loss.head_scale_grad_rms
        );
    }

    #[test]
    fn train_diagnostics_report_row_loss_spread_and_unigram_baseline() {
        let mut model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 8) as u32).collect();
        let loss = model.train_step(&tokens, 1, 32, 0.05).unwrap();
        assert!(loss.next_token.is_finite());
        assert!(loss.loss_p10 <= loss.loss_p50);
        assert!(loss.loss_p50 <= loss.loss_p90);
        assert!(loss.unigram_ce.is_finite() && loss.unigram_ce > 0.0);
        assert!(loss.unique_targets >= 2);
        assert_eq!(
            loss.binary_flip_count,
            loss.flips_head + loss.flips_cmix + loss.flips_rosa_o
        );
        assert!(
            loss.cmix_value_rms > 0.01,
            "CMix value must be kaiming-initialized, rms={}",
            loss.cmix_value_rms
        );
    }

    #[test]
    fn token_sum_ste_flips_mean_reduced_target_row() {
        let mut linear = PackedBinaryLinear::seeded(1, 256, false, 1).unwrap();
        let n_valid = 1024_usize;
        let g_w = vec![1.0 / n_valid as f32 / 16.0; 256];
        let g_scale = [0.0_f32];
        let before = linear.sign_at(0);
        for _ in 0..20 {
            linear
                .apply_packed_sgd(&g_w, &g_scale, None, 1e-2, binaryconnect_ste_scale(n_valid))
                .unwrap();
        }
        assert_ne!(
            linear.sign_at(0),
            before,
            "token-sum STE must cross ±0.01 within 20 steps, latent={}",
            linear.latent().get(0)
        );
    }

    #[test]
    fn binaryconnect_magnitude_ste_preserves_softmax_class_ratio() {
        let mut linear = PackedBinaryLinear::from_signs(1, 1, &[1], 0.0625, false).unwrap();
        let tiny = [1e-5_f32];
        let g_scale = [0.0];
        for _ in 0..100 {
            linear
                .apply_clipped_sgd(&tiny, &g_scale, None, 1e-3)
                .unwrap();
        }
        assert!(
            linear.sign_at(0) > 0.0,
            "a 1e-5 wrong-class STE must not flip the bit in 100 lr=1e-3 steps, latent={}",
            linear.latent().get(0)
        );
        let large = [1.0_f32];
        for _ in 0..20 {
            linear
                .apply_clipped_sgd(&large, &g_scale, None, 1e-2)
                .unwrap();
        }
        assert!(
            linear.sign_at(0) < 0.0,
            "O(1) STE must flip the BinaryConnect proxy, latent={}",
            linear.latent().get(0)
        );
    }

    fn greedy_argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .filter(|(id, _)| *id >= 4)
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id as u32)
            .unwrap()
    }

    #[test]
    fn greedy_decode_uses_prefix_not_unigram() {
        let mut model = UllisHeron::new(tiny_train_cfg()).unwrap();
        // Bigram 4↔5. A 1-token CMix shift can represent it; unigram is a coin flip.
        let tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 2) as u32).collect();
        for _ in 0..80 {
            model.train_step(&tokens, 1, 32, 0.08).unwrap();
        }
        let mut state = model.generate_state().unwrap();
        let logits_after_4 = model.generate_step(&mut state, 4).unwrap();
        assert_eq!(
            greedy_argmax(&logits_after_4),
            5,
            "after fitting 4,5,4,5, greedy(4) must emit 5"
        );
        let logits_after_45 = model.generate_step(&mut state, 5).unwrap();
        assert_eq!(
            greedy_argmax(&logits_after_45),
            4,
            "after fitting 4,5,4,5, greedy(4,5) must emit 4"
        );
    }

    #[test]
    fn seeded_binary_latent_is_independent_of_row_scale() {
        let linear = PackedBinaryLinear::seeded(4, 256, false, 1).unwrap();
        let scale = (256.0_f32).sqrt().recip();
        assert!((linear.scale().get(0) - scale).abs() < 1e-3);
        for i in 0..linear.latent().len() {
            assert!(
                (linear.latent().get(i).abs() - BINARYCONNECT_INIT_ABS).abs() < 1e-4,
                "latent {} should be ±{}, not ±scale {}",
                linear.latent().get(i),
                BINARYCONNECT_INIT_ABS,
                scale
            );
        }
    }

    #[test]
    fn train_step_stop_grad_freezes_qkv_and_updates_e_o_head() {
        let mut model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let q_bits = model.blocks[0].rosa.q.bits().to_vec();
        let k_bits = model.blocks[0].rosa.k.bits().to_vec();
        let v_bits = model.blocks[0].rosa.v.bits().to_vec();
        let ln3_w = model.blocks[0].ln3.weight.as_bits().to_vec();
        let x_q = model.blocks[0].rosa.x_q.as_bits().to_vec();
        let e_before = model.blocks[0].rosa.e.as_bits().to_vec();
        let o_bits = model.blocks[0].rosa.o.bits().to_vec();
        let o_scale = model.blocks[0].rosa.o.scale().as_bits().to_vec();
        let head_scale = model.head.scale().as_bits().to_vec();
        let tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 8) as u32).collect();
        let loss = model.train_step(&tokens, 1, 32, 0.05).unwrap();
        assert!(loss.next_token.is_finite());
        assert_eq!(loss.next_token_count, 31);
        assert_eq!(model.blocks[0].rosa.q.bits(), q_bits.as_slice());
        assert_eq!(model.blocks[0].rosa.k.bits(), k_bits.as_slice());
        assert_eq!(model.blocks[0].rosa.v.bits(), v_bits.as_slice());
        assert_eq!(model.blocks[0].ln3.weight.as_bits(), ln3_w.as_slice());
        assert_eq!(model.blocks[0].rosa.x_q.as_bits(), x_q.as_slice());
        assert_ne!(model.blocks[0].rosa.e.as_bits(), e_before.as_slice());
        assert!(
            model.blocks[0].rosa.o.bits() != o_bits.as_slice()
                || model.blocks[0].rosa.o.scale().as_bits() != o_scale.as_slice()
        );
        assert_ne!(model.head.scale().as_bits(), head_scale.as_slice());
    }

    #[test]
    fn train_step_loss_drops_below_ln_v_without_qkv_bit_grad() {
        let mut model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 4) as u32).collect();
        let ln_v = (MIN_VOCAB as f32).ln();
        let mut last = f32::INFINITY;
        for _ in 0..80 {
            last = model.train_step(&tokens, 1, 32, 0.08).unwrap().next_token;
        }
        assert!(
            last < ln_v - 0.2,
            "loss {last} should beat ln(V)={ln_v} via CMix/head/e, not QKV bits"
        );
    }

    #[test]
    fn exact_bitflip_mode_is_rejected_until_pr5b() {
        let mut cfg = tiny_train_cfg();
        cfg.rosa_grad = RosaGradMode::ExactBitflip;
        let mut model = UllisHeron::new(cfg).unwrap();
        let err = model
            .train_step(&[4, 5], 1, 2, 1e-3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("stop_grad_bits"));
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

    #[test]
    fn hybrid_roundtrip_preserves_hidden() {
        let model = UllisHeron::new(TrainConfig {
            architecture: Architecture::RosaRwkv7,
            vocab_size: 12,
            d_model: 32,
            n_layers: 2,
            dim_ffn: 128,
            context_len: 144,
            tmix_lora_rank: 8,
            ..Default::default()
        })
        .unwrap();
        let tokens: Vec<u32> = (0..16).map(|i| i % 12).collect();
        let hidden = model.hidden(&tokens, 1, 16).unwrap();
        let restored = UllisHeron::from_checkpoint(model.checkpoint()).unwrap();
        let hidden_restored = restored.hidden(&tokens, 1, 16).unwrap();
        assert_eq!(hidden, hidden_restored);
    }

    #[test]
    fn checkpoint_json_matches_v2_schema_and_omits_latents() {
        let model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let json = serde_json::to_value(model.checkpoint()).unwrap();
        assert_eq!(json["format_version"], 2);
        assert_eq!(json["config"]["architecture"], "heron");
        assert_eq!(json["config"]["d_model"], 16);
        assert_eq!(json["config"]["n_layers"], 1);
        assert_eq!(json["config"]["vocab_size"], MIN_VOCAB);
        assert_eq!(json["config"]["context_len"], 32);
        assert_eq!(json["config"]["dim_ffn"], 64);
        assert_eq!(json["config"]["rosa_bits"], 1);
        assert_eq!(json["config"]["rosa_grad"], "stop_grad_bits");
        assert_eq!(json["config"]["optimizer"], "stateless_sgd");
        assert!(json["head"]["bias_bits"].is_null());
        assert!(json["head"]["bits"].is_array());
        assert!(json["head"]["scale_bits"].is_array());
        assert!(json["ln_out"]["weight"].is_array());
        let block = &json["blocks"][0];
        assert!(block["ln0"]["weight"].is_array());
        assert!(block["rosa"]["q"]["bias_bits"].is_array());
        assert!(block["rosa"]["e"].is_array());
        assert!(block["ffn"]["key"]["bias_bits"].is_null());
        assert!(block["ffn"]["value_bits"].is_array());
        let dump = json.to_string();
        assert!(!dump.contains("latent"));
        assert!(!dump.contains("master_bits"));
        assert!(!dump.contains("hyena"));
        assert!(!json.as_object().unwrap().contains_key("hybrid_blocks"));
    }

    #[test]
    fn checkpoint_roundtrip_after_train_preserves_forward() {
        let mut model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 8) as u32).collect();
        model.train_step(&tokens, 1, 32, 0.05).unwrap();
        let hidden = model.hidden(&tokens, 1, 32).unwrap();
        let bytes = serde_json::to_vec(&model.checkpoint()).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("latent"));
        let restored =
            UllisHeron::from_checkpoint(ModelCheckpoint::from_json_bytes(&bytes).unwrap()).unwrap();
        let hidden_restored = restored.hidden(&tokens, 1, 32).unwrap();
        assert_eq!(hidden, hidden_restored);
        assert_eq!(
            restored.blocks[0].rosa.e.as_bits(),
            model.blocks[0].rosa.e.as_bits()
        );
        assert_eq!(restored.head.bits(), model.head.bits());
        assert_eq!(
            restored.head.scale().as_bits(),
            model.head.scale().as_bits()
        );
    }

    #[test]
    fn json_v1_payload_is_rejected_before_v2_schema() {
        let error = ModelCheckpoint::from_json_bytes(
            br#"{"format_version":1,"config":{"d_model":256},"embedding_bits":[]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("Hyena checkpoints (v1)"));
        assert!(error.contains("format_version 1"));
    }

    #[test]
    fn json_without_version_is_rejected_as_hyena() {
        let error = ModelCheckpoint::from_json_bytes(br#"{"embedding_bits":[]}"#)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Hyena checkpoints (v1)"));
        assert!(error.contains("format_version 0"));
    }

    #[test]
    fn inspect_splits_param_counts_and_inference_state() {
        let model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let report = model.checkpoint().inspect().unwrap();
        assert_eq!(report.format_version, 2);
        assert_eq!(report.architecture, Architecture::Heron);
        let d = 16_usize;
        let v = MIN_VOCAB as usize;
        let ffn = 64_usize;
        assert_eq!(report.param_counts.embedding, v * d);
        assert_eq!(report.param_counts.packed_bits, v * d + 4 * d * d + ffn * d);
        assert_eq!(report.param_counts.fp16_matrices, d * ffn);
        assert_eq!(report.param_counts.layer_norm, 8 * d);
        assert_eq!(report.param_counts.rosa_e, d);
        assert_eq!(report.param_counts.fp16_vectors, 4 * d);
        assert_eq!(
            report.inference_state.sam_bytes,
            sam_workspace_bytes(1, 32, d).unwrap()
        );
        assert_eq!(
            report.inference_state.time_shift_bytes,
            d * size_of::<f32>()
        );
    }

    #[test]
    fn generate_step_logits_match_one_shot_hidden() {
        let model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let tokens: Vec<u32> = (0..8).map(|i| 4 + (i % 8) as u32).collect();
        let d = model.cfg.d_model;
        let hidden = model.hidden(&tokens, 1, tokens.len()).unwrap();
        let mut state = model.generate_state().unwrap();
        assert_eq!(state.time(), 0);
        for (t, &id) in tokens.iter().enumerate() {
            let logits = model.generate_step(&mut state, id).unwrap();
            let expected = model.head.forward(&hidden[t * d..(t + 1) * d], 1).unwrap();
            assert_eq!(logits, expected);
            assert_eq!(state.time(), t + 1);
        }
    }

    #[test]
    fn rosa_forward_one_matches_oneshot_and_reuses_push() {
        let model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let tokens: Vec<u32> = (0..8).map(|i| 4 + i as u32).collect();
        let d = model.cfg.d_model;
        let mut x = vec![0.0; tokens.len() * d];
        for (row, &id) in tokens.iter().enumerate() {
            let offset = id as usize * d;
            for c in 0..d {
                x[row * d + c] = model.embedding.get(offset + c);
            }
        }
        let block = &model.blocks[0];
        let h = block
            .ln0
            .as_ref()
            .unwrap()
            .forward(&x, tokens.len())
            .unwrap();
        let rosa_in = block.ln3.forward(&h, tokens.len()).unwrap();
        let one_shot = block.rosa.forward(&rosa_in, 1, tokens.len()).unwrap();
        let mut sams: Vec<RosaSam> = (0..d)
            .map(|_| RosaSam::with_max_time(tokens.len()))
            .collect();
        let mut prev = None;
        let mut incremental = Vec::new();
        for t in 0..tokens.len() {
            let row = &rosa_in[t * d..(t + 1) * d];
            let out = block
                .rosa
                .forward_one(row, prev.as_deref(), &mut sams)
                .unwrap();
            incremental.extend_from_slice(&out);
            prev = Some(row.to_vec());
        }
        assert_eq!(incremental, one_shot);
    }

    #[test]
    fn generate_step_rejects_past_context_len() {
        let model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let mut state = model.generate_state().unwrap();
        for i in 0..model.cfg.context_len {
            model.generate_step(&mut state, 4 + (i as u32 % 8)).unwrap();
        }
        let err = model.generate_step(&mut state, 4).unwrap_err().to_string();
        assert!(err.contains("context_len"));
    }

    #[test]
    fn generate_step_matches_one_shot_after_train() {
        let mut model = UllisHeron::new(tiny_train_cfg()).unwrap();
        let tokens: Vec<u32> = (0..32).map(|i| 4 + (i % 8) as u32).collect();
        model.train_step(&tokens, 1, 32, 0.05).unwrap();
        let prefix = &tokens[..8];
        let d = model.cfg.d_model;
        let hidden = model.hidden(prefix, 1, prefix.len()).unwrap();
        let mut state = model.generate_state().unwrap();
        for (t, &id) in prefix.iter().enumerate() {
            let logits = model.generate_step(&mut state, id).unwrap();
            let expected = model.head.forward(&hidden[t * d..(t + 1) * d], 1).unwrap();
            assert_eq!(logits, expected);
        }
    }
}
