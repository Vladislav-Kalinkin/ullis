//! Dense ternary Hyena model and multi-token prediction heads.
use crate::batch::MtpBatch;
use crate::config::TrainConfig;
use crate::hyena::{causal_long_conv_implicit_strided, ImplicitFilter};
use crate::optimizer::Lion;
use anyhow::{bail, Result};

/// Cross-entropy statistics for the two MTP horizons. Values are means over
/// valid positions; no `[batch, time, vocab]` logits tensor is retained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpLoss {
    pub next_token: f32,
    pub second_token: f32,
    pub next_token_count: usize,
    pub second_token_count: usize,
}

impl MtpLoss {
    pub fn mean(self) -> f32 {
        (self.next_token + self.second_token) * 0.5
    }
}

/// Two bitplanes encode `{-1, 0, +1}` without an i8 value per weight.
#[derive(Clone, Debug)]
struct PackedTernary {
    positive: Vec<u64>,
    negative: Vec<u64>,
    len: usize,
}

impl PackedTernary {
    fn zeros(len: usize) -> Self {
        let words = len.div_ceil(64);
        Self {
            positive: vec![0; words],
            negative: vec![0; words],
            len,
        }
    }

    fn set(&mut self, index: usize, code: i8) {
        debug_assert!(index < self.len);
        let word = index / 64;
        let bit = 1_u64 << (index % 64);
        self.positive[word] &= !bit;
        self.negative[word] &= !bit;
        if code > 0 {
            self.positive[word] |= bit;
        } else if code < 0 {
            self.negative[word] |= bit;
        }
    }

    fn get(&self, index: usize) -> f32 {
        debug_assert!(index < self.len);
        let word = index / 64;
        let bit = 1_u64 << (index % 64);
        if self.positive[word] & bit != 0 {
            1.0
        } else if self.negative[word] & bit != 0 {
            -1.0
        } else {
            0.0
        }
    }

    fn storage_bytes(&self) -> usize {
        (self.positive.len() + self.negative.len()) * size_of::<u64>()
    }
}

#[derive(Clone, Debug)]
pub struct TernaryLinear {
    master: Vec<f32>,
    codes: PackedTernary,
    row_scales: Vec<f32>,
    in_features: usize,
    out_features: usize,
    threshold_ratio: f32,
}
impl TernaryLinear {
    pub fn seeded(
        in_features: usize,
        out_features: usize,
        threshold_ratio: f32,
        seed: u64,
    ) -> Self {
        let scale = (in_features as f32).sqrt().recip();
        let master = seeded_values(in_features * out_features, scale, seed);
        let mut result = Self {
            master,
            codes: PackedTernary::zeros(in_features * out_features),
            row_scales: vec![0.0; out_features],
            in_features,
            out_features,
            threshold_ratio,
        };
        result.refresh_codes();
        result
    }
    pub fn refresh_codes(&mut self) {
        for row in 0..self.out_features {
            let start = row * self.in_features;
            let mean = self.master[start..start + self.in_features]
                .iter()
                .map(|v| v.abs())
                .sum::<f32>()
                / self.in_features as f32;
            let threshold = self.threshold_ratio * mean;
            self.row_scales[row] = mean;
            for (offset, &weight) in self.master[start..start + self.in_features]
                .iter()
                .enumerate()
            {
                let code = if weight > threshold {
                    1
                } else if weight < -threshold {
                    -1
                } else {
                    0
                };
                self.codes.set(start + offset, code);
            }
        }
    }

    /// Applies a clipped straight-through gradient to FP32 master weights and
    /// immediately refreshes the packed ternary inference plane.
    pub fn apply_ste_gradient(&mut self, gradient: &[f32], learning_rate: f32) -> Result<()> {
        if gradient.len() != self.master.len()
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || gradient.iter().any(|value| !value.is_finite())
        {
            bail!("invalid ternary STE gradient or learning rate");
        }
        for (weight, &gradient) in self.master.iter_mut().zip(gradient) {
            *weight -= learning_rate * gradient.clamp(-1.0, 1.0);
        }
        self.refresh_codes();
        Ok(())
    }

    /// Applies Lion to master weights, then refreshes the packed ternary plane.
    /// The optimiser owns exactly one FP32 momentum value per master weight.
    pub fn apply_lion_gradient(&mut self, optimizer: &mut Lion, gradient: &[f32]) -> Result<()> {
        optimizer.step(&mut self.master, gradient)?;
        self.refresh_codes();
        Ok(())
    }

    pub fn packed_code_bytes(&self) -> usize {
        self.codes.storage_bytes()
    }
    pub fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        if x.len() != rows * self.in_features {
            bail!("ternary linear input shape mismatch");
        }
        let mut out = vec![0.0; rows * self.out_features];
        for r in 0..rows {
            for o in 0..self.out_features {
                for i in 0..self.in_features {
                    out[r * self.out_features + o] +=
                        x[r * self.in_features + i] * self.codes.get(o * self.in_features + i);
                }
                out[r * self.out_features + o] *= self.row_scales[o];
            }
        }
        Ok(out)
    }

    /// Numerical-reference Metal projection using this layer's immutable
    /// packed weights. It deliberately does not silently fall back to CPU:
    /// callers can distinguish an unavailable device from a verified GPU run.
    pub fn forward_metal_reference(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        crate::metal::ternary_linear_forward(
            x,
            &self.codes.positive,
            &self.codes.negative,
            &self.row_scales,
            crate::metal::TernaryLinearShape::new(rows, self.in_features, self.out_features)?,
        )
    }

    /// Dispatches through a caller-owned Metal runtime, retaining pipelines
    /// and scratch buffers across layer calls.
    #[cfg(target_os = "macos")]
    pub fn forward_with_metal_runtime(
        &self,
        runtime: &crate::metal::MetalRuntime,
        x: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        runtime.ternary_linear_forward(
            x,
            &self.codes.positive,
            &self.codes.negative,
            &self.row_scales,
            crate::metal::TernaryLinearShape::new(rows, self.in_features, self.out_features)?,
        )
    }

    /// GPU equivalent of `forward_rms_norm`; the normalized intermediate is
    /// retained only inside the Metal threadgroup, never as a tensor buffer.
    #[cfg(target_os = "macos")]
    pub fn forward_rms_norm_with_metal_runtime(
        &self,
        runtime: &crate::metal::MetalRuntime,
        x: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        runtime.rms_norm_ternary_linear_forward(
            x,
            &self.codes.positive,
            &self.codes.negative,
            &self.row_scales,
            crate::metal::TernaryLinearShape::new(rows, self.in_features, self.out_features)?,
        )
    }

    /// Fused parameter-free RMSNorm plus ternary projection. This gives each
    /// residual stream a stable scale without allocating a normalized tensor.
    pub fn forward_rms_norm(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        if x.len() != rows * self.in_features {
            bail!("ternary RMSNorm input shape mismatch");
        }
        let mut out = vec![0.0; rows * self.out_features];
        for row in 0..rows {
            let input = &x[row * self.in_features..(row + 1) * self.in_features];
            let mean_square =
                input.iter().map(|value| value * value).sum::<f32>() / self.in_features as f32;
            let inverse_rms = (mean_square + 1e-5).sqrt().recip();
            for output in 0..self.out_features {
                let mut sum = 0.0;
                for input_index in 0..self.in_features {
                    sum += input[input_index]
                        * self.codes.get(output * self.in_features + input_index);
                }
                out[row * self.out_features + output] = sum * inverse_rms * self.row_scales[output];
            }
        }
        Ok(out)
    }

    #[cfg(target_os = "macos")]
    fn metal_parts(&self) -> (&[u64], &[u64], &[f32]) {
        (&self.codes.positive, &self.codes.negative, &self.row_scales)
    }
}

/// Deterministic initializer shared by FP32 embeddings and ternary master
/// weights. Keeping embeddings out of `TernaryLinear` avoids allocating a
/// temporary packed-code plane while constructing the model.
fn seeded_values(len: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

#[derive(Clone, Debug)]
struct HyenaBlock {
    input: TernaryLinear,
    output: TernaryLinear,
    filter: ImplicitFilter,
    d_model: usize,
}
impl HyenaBlock {
    fn seeded(cfg: &TrainConfig, seed: u64) -> Self {
        Self {
            input: TernaryLinear::seeded(cfg.d_model, 2 * cfg.d_model, cfg.ternary_delta, seed),
            output: TernaryLinear::seeded(
                cfg.d_model,
                cfg.d_model,
                cfg.ternary_delta,
                seed ^ 0xa5a5,
            ),
            filter: ImplicitFilter::new(cfg.d_model, cfg.filter_order, seed ^ 0x5a5a),
            d_model: cfg.d_model,
        }
    }
    fn forward(&self, x: &[f32], batch: usize, time: usize) -> Result<Vec<f32>> {
        let mut projected = self.input.forward_rms_norm(x, batch * time)?;
        for n in 0..batch * time {
            for c in 0..self.d_model {
                let gate_index = n * 2 * self.d_model + self.d_model + c;
                projected[gate_index] = projected[gate_index].tanh();
            }
        }
        let mut mixed = causal_long_conv_implicit_strided(
            &projected,
            &self.filter,
            batch,
            time,
            self.d_model,
            2 * self.d_model,
            0,
        )?;
        for n in 0..batch * time {
            for c in 0..self.d_model {
                mixed[n * self.d_model + c] *= projected[n * 2 * self.d_model + self.d_model + c];
            }
        }
        let update = self.output.forward(&mixed, batch * time)?;
        Ok(x.iter().zip(update).map(|(a, b)| a + b).collect())
    }

    #[cfg(target_os = "macos")]
    fn forward_metal_reference(
        &self,
        runtime: &crate::metal::MetalRuntime,
        x: &[f32],
        batch: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena block row overflow"))?;
        let projected = self
            .input
            .forward_rms_norm_with_metal_runtime(runtime, x, rows)?;
        let gated = runtime.tanh_gate_forward(&projected, rows, self.d_model)?;
        let mut mixed = runtime.causal_long_conv_implicit_strided_forward(
            &gated,
            &self.filter,
            batch,
            time,
            self.d_model,
            2 * self.d_model,
            0,
        )?;
        for row in 0..rows {
            for channel in 0..self.d_model {
                mixed[row * self.d_model + channel] *=
                    gated[row * 2 * self.d_model + self.d_model + channel];
            }
        }
        let update = self
            .output
            .forward_with_metal_runtime(runtime, &mixed, rows)?;
        Ok(x.iter()
            .zip(update)
            .map(|(left, right)| left + right)
            .collect())
    }

    /// Executes one complete block with both residual states resident in the
    /// runtime. The only host copies are the packed immutable weights.
    #[cfg(target_os = "macos")]
    fn forward_metal_resident(
        &self,
        runtime: &crate::metal::MetalRuntime,
        slot: crate::metal::ResidentActivationSlot,
        batch: usize,
        time: usize,
    ) -> Result<crate::metal::ResidentActivationSlot> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena block row overflow"))?;
        let (positive, negative, scales) = self.input.metal_parts();
        runtime.resident_input_projection(slot, rows, self.d_model, positive, negative, scales)?;
        runtime.resident_hyena_mixer(slot, batch, time, self.d_model, &self.filter)?;
        let (positive, negative, scales) = self.output.metal_parts();
        runtime.resident_output_projection(slot, rows, self.d_model, positive, negative, scales)
    }
}

/// Inference core. `mtp_logits` exposes separate t+1 and t+2 pretraining heads.
#[derive(Clone, Debug)]
pub struct UllisHyena {
    pub cfg: TrainConfig,
    embedding: Vec<f32>,
    blocks: Vec<HyenaBlock>,
    mtp_one: TernaryLinear,
    mtp_two: TernaryLinear,
}
impl UllisHyena {
    pub fn new(cfg: TrainConfig) -> Result<Self> {
        cfg.validate()?;
        let embedding = seeded_values(
            cfg.vocab_size * cfg.d_model,
            (cfg.d_model as f32).sqrt().recip(),
            cfg.seed,
        );
        let blocks = (0..cfg.n_layers)
            .map(|i| HyenaBlock::seeded(&cfg, cfg.seed.wrapping_add(i as u64 + 1)))
            .collect();
        Ok(Self {
            mtp_one: TernaryLinear::seeded(
                cfg.d_model,
                cfg.d_model,
                cfg.ternary_delta,
                cfg.seed ^ 1,
            ),
            mtp_two: TernaryLinear::seeded(
                cfg.d_model,
                cfg.d_model,
                cfg.ternary_delta,
                cfg.seed ^ 2,
            ),
            cfg,
            embedding,
            blocks,
        })
    }
    pub fn hidden(&self, ids: &[u32], batch: usize, time: usize) -> Result<Vec<f32>> {
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
        let d = self.cfg.d_model;
        let values = rows
            .checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("hidden-state shape overflow"))?;
        let mut x = vec![0.0; values];
        for (row, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.cfg.vocab_size {
                bail!("token id {id} out of vocabulary");
            }
            x[row * d..(row + 1) * d].copy_from_slice(&self.embedding[id * d..(id + 1) * d]);
        }
        for block in &self.blocks {
            x = block.forward(&x, batch, time)?;
        }
        Ok(x)
    }

    /// Complete Metal numerical-reference forward. It deliberately exposes a
    /// distinct API until residual and projection buffers remain resident
    /// across blocks; callers never silently receive a mixed CPU/GPU path.
    #[cfg(target_os = "macos")]
    pub fn hidden_metal_reference(
        &self,
        runtime: &crate::metal::MetalRuntime,
        ids: &[u32],
        batch: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
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
        let mut x = vec![
            0.0;
            rows.checked_mul(self.cfg.d_model)
                .ok_or_else(|| anyhow::anyhow!("hidden-state shape overflow"))?
        ];
        for (row, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.cfg.vocab_size {
                bail!("token id {id} out of vocabulary");
            }
            x[row * self.cfg.d_model..(row + 1) * self.cfg.d_model].copy_from_slice(
                &self.embedding[id * self.cfg.d_model..(id + 1) * self.cfg.d_model],
            );
        }
        for block in &self.blocks {
            x = block.forward_metal_reference(runtime, &x, batch, time)?;
        }
        Ok(x)
    }

    /// Complete Metal forward whose inter-block hidden state never crosses
    /// the CPU/GPU boundary. This is the production inference shape; the
    /// separate `hidden_metal_reference` remains useful for diagnostics.
    #[cfg(target_os = "macos")]
    pub fn hidden_metal_resident(
        &self,
        runtime: &crate::metal::MetalRuntime,
        ids: &[u32],
        batch: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
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
        let d = self.cfg.d_model;
        let mut embedding_stream = vec![
            0.0;
            rows.checked_mul(d).ok_or_else(|| anyhow::anyhow!(
                "hidden-state shape overflow"
            ))?
        ];
        for (row, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.cfg.vocab_size {
                bail!("token id {id} out of vocabulary");
            }
            embedding_stream[row * d..(row + 1) * d]
                .copy_from_slice(&self.embedding[id * d..(id + 1) * d]);
        }
        let mut slot = runtime.upload_resident_activations(&embedding_stream, rows, d)?;
        for block in &self.blocks {
            slot = block.forward_metal_resident(runtime, slot, batch, time)?;
        }
        runtime.download_resident_activations(slot, rows, d)
    }

    /// Computes t+1 and t+2 cross-entropy without allocating vocab logits for
    /// every position. The only per-row storage is the existing `D`-wide MTP
    /// head output, so its memory stays `O(B*T*D)` as context grows.
    pub fn streamed_mtp_loss(&self, ids: &[u32], batch: usize, time: usize) -> Result<MtpLoss> {
        if time < 3 {
            bail!("MTP loss needs at least three tokens per sequence");
        }
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        let hidden = self.hidden(ids, batch, time)?;
        let one = self.mtp_one.forward(&hidden, rows)?;
        let (next_sum, next_count) = self.cross_entropy_horizon(&one, ids, batch, time, 1);
        drop(one);
        let two = self.mtp_two.forward(&hidden, rows)?;
        let (second_sum, second_count) = self.cross_entropy_horizon(&two, ids, batch, time, 2);
        Ok(MtpLoss {
            next_token: next_sum / next_count as f32,
            second_token: second_sum / second_count as f32,
            next_token_count: next_count,
            second_token_count: second_count,
        })
    }

    /// Convenience entry point for the zero-copy training batcher.
    pub fn streamed_batch_loss(&self, batch: MtpBatch<'_>) -> Result<MtpLoss> {
        self.streamed_mtp_loss(batch.tokens(), batch.batch_size(), batch.time())
    }
    pub fn mtp_logits(
        &self,
        ids: &[u32],
        batch: usize,
        time: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.cfg.validate_materialized_mtp(time)?;
        let hidden = self.hidden(ids, batch, time)?;
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        let one = self.mtp_one.forward(&hidden, rows)?;
        let two = self.mtp_two.forward(&hidden, rows)?;
        Ok((self.project(&one), self.project(&two)))
    }
    fn project(&self, hidden: &[f32]) -> Vec<f32> {
        let rows = hidden.len() / self.cfg.d_model;
        let mut logits = vec![0.0; rows * self.cfg.vocab_size];
        for r in 0..rows {
            for v in 0..self.cfg.vocab_size {
                logits[r * self.cfg.vocab_size + v] = hidden
                    [r * self.cfg.d_model..(r + 1) * self.cfg.d_model]
                    .iter()
                    .zip(&self.embedding[v * self.cfg.d_model..(v + 1) * self.cfg.d_model])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        logits
    }

    fn row_cross_entropy(&self, hidden: &[f32], row: usize, target: u32) -> f32 {
        let target = target as usize;
        debug_assert!(target < self.cfg.vocab_size);
        let state = &hidden[row * self.cfg.d_model..(row + 1) * self.cfg.d_model];
        let mut max_logit = f32::NEG_INFINITY;
        let mut target_logit = 0.0;
        for token in 0..self.cfg.vocab_size {
            let logit = dot(
                state,
                &self.embedding[token * self.cfg.d_model..(token + 1) * self.cfg.d_model],
            );
            max_logit = max_logit.max(logit);
            if token == target {
                target_logit = logit;
            }
        }
        let mut exp_sum = 0.0;
        for token in 0..self.cfg.vocab_size {
            exp_sum += (dot(
                state,
                &self.embedding[token * self.cfg.d_model..(token + 1) * self.cfg.d_model],
            ) - max_logit)
                .exp();
        }
        max_logit + exp_sum.ln() - target_logit
    }

    fn cross_entropy_horizon(
        &self,
        head: &[f32],
        ids: &[u32],
        batch: usize,
        time: usize,
        horizon: usize,
    ) -> (f32, usize) {
        let mut sum = 0.0;
        let mut count = 0;
        for sequence in 0..batch {
            let base = sequence * time;
            for position in 0..time - horizon {
                sum +=
                    self.row_cross_entropy(head, base + position, ids[base + position + horizon]);
                count += 1;
            }
        }
        (sum, count)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_codes_use_two_bits_per_weight() {
        let layer = TernaryLinear::seeded(128, 4, 0.7, 7);
        assert_eq!(layer.packed_code_bytes(), 128);
        assert_eq!(layer.packed_code_bytes() * 4, 128 * 4);
    }

    #[test]
    fn embedding_initializer_is_deterministic_and_width_scaled() {
        assert_eq!(seeded_values(4, 0.25, 7), seeded_values(4, 0.25, 7));
        assert!(seeded_values(64, 0.25, 7)
            .iter()
            .all(|value| value.abs() <= 0.25));
    }

    #[test]
    fn ste_update_is_clipped_and_rejects_invalid_gradients() {
        let mut layer = TernaryLinear::seeded(2, 1, 0.7, 7);
        let before = layer.master.clone();
        layer.apply_ste_gradient(&[100.0, -100.0], 0.5).unwrap();
        assert_eq!(layer.master[0], before[0] - 0.5);
        assert_eq!(layer.master[1], before[1] + 0.5);
        assert!(layer.apply_ste_gradient(&[f32::NAN, 0.0], 0.1).is_err());
    }

    #[test]
    fn lion_update_refreshes_ternary_master_weights() {
        let mut layer = TernaryLinear::seeded(2, 1, 0.7, 7);
        let mut lion = Lion::new(2, crate::optimizer::LionConfig::default()).unwrap();
        let before = layer.master.clone();
        layer.apply_lion_gradient(&mut lion, &[1.0, -1.0]).unwrap();
        assert_ne!(layer.master, before);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn ternary_layer_metal_reference_matches_cpu_when_available() {
        let layer = TernaryLinear::seeded(5, 3, 0.7, 42);
        let input = [0.25, -1.0, 2.5, 0.0, 1.0, -0.75, 0.5, 1.5, -2.0, 0.25];
        let expected = layer.forward(&input, 2).unwrap();
        if let Ok(actual) = layer.forward_metal_reference(&input, 2) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn ternary_layer_uses_caller_owned_metal_runtime_when_available() {
        let layer = TernaryLinear::seeded(5, 3, 0.7, 42);
        let input = [0.25, -1.0, 2.5, 0.0, 1.0, -0.75, 0.5, 1.5, -2.0, 0.25];
        let expected = layer.forward(&input, 2).unwrap();
        if let Ok(runtime) = crate::metal::MetalRuntime::new() {
            let actual = layer
                .forward_with_metal_runtime(&runtime, &input, 2)
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn fused_rms_norm_ternary_layer_matches_cpu_when_available() {
        let layer = TernaryLinear::seeded(5, 3, 0.7, 42);
        let input = [0.25, -1.0, 2.5, 0.0, 1.0, -0.75, 0.5, 1.5, -2.0, 0.25];
        let expected = layer.forward_rms_norm(&input, 2).unwrap();
        if let Ok(runtime) = crate::metal::MetalRuntime::new() {
            let actual = layer
                .forward_rms_norm_with_metal_runtime(&runtime, &input, 2)
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn complete_metal_forward_matches_cpu_when_available() {
        let cfg = TrainConfig {
            d_model: 4,
            n_layers: 1,
            vocab_size: 320,
            context_len: 4,
            batch_size: 1,
            ..Default::default()
        };
        let model = UllisHyena::new(cfg).unwrap();
        let ids = [1, 2, 3, 4];
        let expected = model.hidden(&ids, 1, 4).unwrap();
        if let Ok(runtime) = crate::metal::MetalRuntime::new() {
            let actual = model.hidden_metal_reference(&runtime, &ids, 1, 4).unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_metal_forward_matches_cpu_across_two_blocks_when_available() {
        let cfg = TrainConfig {
            d_model: 4,
            n_layers: 2,
            vocab_size: 320,
            context_len: 4,
            batch_size: 1,
            ..Default::default()
        };
        let model = UllisHyena::new(cfg).unwrap();
        let ids = [1, 2, 3, 4];
        let expected = model.hidden(&ids, 1, 4).unwrap();
        let Ok(runtime) = crate::metal::MetalRuntime::new() else {
            return;
        };
        let actual = model.hidden_metal_resident(&runtime, &ids, 1, 4).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
    }
}
