//! Dense ternary Hyena model and multi-token prediction heads.
use crate::batch::MtpBatch;
use crate::config::TrainConfig;
use crate::hyena::{
    CausalConvBackward, HyenaChunkPlan, ImplicitFilter, ImplicitFilterBackward,
    causal_chunked_conv_implicit_strided,
};
use crate::optimizer::Lion;
use crate::precision::Fp16Storage;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Cross-entropy statistics for the two MTP horizons. Values are means over
/// valid positions; no `[batch, time, vocab]` logits tensor is retained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpLoss {
    pub next_token: f32,
    pub second_token: f32,
    pub next_token_count: usize,
    pub second_token_count: usize,
}

/// Versioned, lossless snapshot of the persistent FP16 model state.
///
/// The checkpoint deliberately stores binary16 bit patterns rather than
/// widened floats.  It therefore cannot reintroduce a hidden FP32 master
/// model while moving a run between the CPU reference and Metal backends.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub format_version: u32,
    pub config: TrainConfig,
    embedding_bits: Vec<u16>,
    blocks: Vec<HyenaBlockCheckpoint>,
    mtp_one: TernaryLinearCheckpoint,
    mtp_two: TernaryLinearCheckpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TernaryLinearCheckpoint {
    master_bits: Vec<u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HyenaBlockCheckpoint {
    input: TernaryLinearCheckpoint,
    output: TernaryLinearCheckpoint,
    freq: Vec<f32>,
    phase: Vec<f32>,
    decay: Vec<f32>,
}

/// Streamed cross-entropy derivatives for both MTP heads.
///
/// Each gradient has `[batch * time, d_model]` layout. Terminal positions
/// without a target are zero. There is deliberately no dense embedding/output
/// gradient: tied embeddings are reduced one vocabulary row at a time by the
/// eventual updater instead of allocating `[vocab, d_model]` FP32 storage.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpHeadBackward {
    pub loss: MtpLoss,
    pub next_head_gradient: Vec<f32>,
    pub second_head_gradient: Vec<f32>,
}

/// Backward result after both MTP projections have been reduced into their
/// shared hidden-state input.
///
/// Projection weight gradients use the ternary STE contract and can be
/// applied independently by a streaming optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpProjectionBackward {
    pub hidden_gradient: Vec<f32>,
    pub next_projection: TernaryLinearBackward,
    pub second_projection: TernaryLinearBackward,
}

/// Reverse-pass result for a complete resident Hyena stack.
///
/// The block order matches the forward model order. This is intentionally a
/// numerical-reference handoff: the per-block gradients are downloaded so the
/// CPU updater can validate and consume them while the all-resident optimizer
/// graph is still being built.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq)]
pub struct MetalHyenaStackBackward {
    pub input_gradient: Vec<f32>,
    pub blocks: Vec<crate::metal::MetalHyenaBlockBackward>,
}

/// Persistent Metal projection state for a Hyena training run.
///
/// It owns the trainable ternary projections and compact implicit filters.
/// Activations and gradients remain grow-only runtime workspaces; neither
/// optimiser moments nor a second FP32 model copy are retained.
#[cfg(target_os = "macos")]
pub struct MetalResidentHyenaTrainingState {
    blocks: Vec<MetalResidentHyenaBlockWeights>,
    mtp_one: crate::metal::ResidentTrainableFp16TernaryWeights,
    mtp_two: crate::metal::ResidentTrainableFp16TernaryWeights,
    embedding: crate::metal::ResidentFp16Parameters,
}

#[cfg(target_os = "macos")]
struct MetalResidentHyenaBlockWeights {
    input: crate::metal::ResidentTrainableFp16TernaryWeights,
    output: crate::metal::ResidentTrainableFp16TernaryWeights,
    filter: crate::metal::ResidentImplicitFilterParameters,
}

/// Readbacks from a projection-only resident training step. Projection
/// gradients are consumed on Metal; raw filter gradients remain available for
/// validation of the compact implicit-filter updater.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq)]
pub struct MetalResidentHyenaProjectionStep {
    pub input_gradient: Vec<f32>,
    pub filter_gradients: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct CrossEntropyNormalizer {
    max_logits: Vec<f32>,
    exp_sums: Vec<f32>,
}

/// Surrogate derivatives for one packed ternary projection.
///
/// `latent_weight_gradient` uses the straight-through estimator: it treats the
/// quantizer and per-row scale as identity during backward. This is not the
/// mathematical derivative of thresholding; it is the explicit training
/// contract that lets FP16 latent weights cross ternary thresholds.
#[derive(Clone, Debug, PartialEq)]
pub struct TernaryLinearBackward {
    pub input_gradient: Vec<f32>,
    pub latent_weight_gradient: Vec<f32>,
}

/// Backward result for `mixed * tanh(gate_projection)`. The signal half of
/// the projection has no direct gate-path derivative and is therefore zero.
#[derive(Clone, Debug, PartialEq)]
pub struct HyenaGateBackward {
    pub mixed_gradient: Vec<f32>,
    pub projection_gradient: Vec<f32>,
}

/// Complete local derivatives for one Hyena block. Ternary projection weights
/// use the documented STE surrogate; filter parameters use their analytic
/// derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct HyenaBlockBackward {
    pub input_gradient: Vec<f32>,
    pub input_projection: TernaryLinearBackward,
    pub output_projection: TernaryLinearBackward,
    pub filter: ImplicitFilterBackward,
}

/// Exact derivative of the elementwise Hyena gate. `gated_projection` has
/// `[rows, 2 * channels]` layout with its second half already passed through
/// `tanh`, exactly as the forward mixer stores it.
pub fn hyena_gate_backward(
    mixed: &[f32],
    gated_projection: &[f32],
    output_gradient: &[f32],
    channels: usize,
) -> Result<HyenaGateBackward> {
    if channels == 0
        || mixed.is_empty()
        || mixed.len() != output_gradient.len()
        || gated_projection.len() != mixed.len().saturating_mul(2)
        || mixed
            .iter()
            .chain(gated_projection)
            .chain(output_gradient)
            .any(|value| !value.is_finite())
    {
        bail!("Hyena gate backward shape/value mismatch");
    }
    let mut mixed_gradient = vec![0.0; mixed.len()];
    let mut projection_gradient = vec![0.0; gated_projection.len()];
    for index in 0..mixed.len() {
        let row = index / channels;
        let channel = index % channels;
        let gate_index = row * 2 * channels + channels + channel;
        let gate = gated_projection[gate_index];
        mixed_gradient[index] = output_gradient[index] * gate;
        projection_gradient[gate_index] =
            output_gradient[index] * mixed[index] * (1.0 - gate * gate);
    }
    Ok(HyenaGateBackward {
        mixed_gradient,
        projection_gradient,
    })
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
    /// FP16 latent weights determine ternary thresholds. Forward uses only
    /// packed codes and scales, so no FP32 master copy is resident.
    master: Fp16Storage,
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
        let master = Fp16Storage::from_f32(seeded_values(in_features * out_features, scale, seed));
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
            let mean = (start..start + self.in_features)
                .map(|index| self.master.get(index).abs())
                .sum::<f32>()
                / self.in_features as f32;
            let threshold = self.threshold_ratio * mean;
            self.row_scales[row] = mean;
            for offset in 0..self.in_features {
                let weight = self.master.get(start + offset);
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
        for (index, &gradient) in gradient.iter().enumerate() {
            self.master
                .apply_clipped_sgd(index, gradient, learning_rate);
        }
        self.refresh_codes();
        Ok(())
    }

    /// Applies Lion to master weights, then refreshes the packed ternary plane.
    /// The optimiser owns exactly one FP32 momentum value per master weight.
    pub fn apply_lion_gradient(&mut self, optimizer: &mut Lion, gradient: &[f32]) -> Result<()> {
        let mut parameters: Vec<f32> = (0..self.master.len())
            .map(|index| self.master.get(index))
            .collect();
        optimizer.step(&mut parameters, gradient)?;
        for (index, value) in parameters.into_iter().enumerate() {
            self.master.set(index, value);
        }
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

    /// Computes the exact input derivative of the packed forward projection
    /// and the clipped-STE surrogate derivative for its FP16 latent weights.
    /// All buffers are caller-sized by the local layer only; no model-wide
    /// gradient tensor is retained.
    pub fn backward_ste(
        &self,
        input: &[f32],
        output_gradient: &[f32],
        rows: usize,
    ) -> Result<TernaryLinearBackward> {
        if rows == 0
            || input.len() != rows * self.in_features
            || output_gradient.len() != rows * self.out_features
            || input
                .iter()
                .chain(output_gradient)
                .any(|value| !value.is_finite())
        {
            bail!("ternary linear backward shape/value mismatch");
        }
        let mut input_gradient = vec![0.0; input.len()];
        let mut latent_weight_gradient = vec![0.0; self.master.len()];
        for row in 0..rows {
            for output in 0..self.out_features {
                let gradient = output_gradient[row * self.out_features + output];
                let scale = self.row_scales[output];
                for feature in 0..self.in_features {
                    let weight = output * self.in_features + feature;
                    input_gradient[row * self.in_features + feature] +=
                        gradient * scale * self.codes.get(weight);
                    latent_weight_gradient[weight] +=
                        gradient * scale * input[row * self.in_features + feature];
                }
            }
        }
        Ok(TernaryLinearBackward {
            input_gradient,
            latent_weight_gradient,
        })
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

    /// Low-memory Metal projection. Input, scales, and output are FP16 in
    /// shared buffers; the shader widens each dot product to FP32 locally.
    #[cfg(target_os = "macos")]
    pub fn forward_metal_fp16_reference(
        &self,
        runtime: &crate::metal::MetalRuntime,
        x: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let input = Fp16Storage::from_f32(x.iter().copied());
        let scales = Fp16Storage::from_f32(self.row_scales.iter().copied());
        let output = runtime.ternary_linear_forward_fp16(
            &input,
            &self.codes.positive,
            &self.codes.negative,
            &scales,
            crate::metal::TernaryLinearShape::new(rows, self.in_features, self.out_features)?,
        )?;
        Ok((0..output.len()).map(|index| output.get(index)).collect())
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

fn restore_ternary_linear(
    layer: &mut TernaryLinear,
    checkpoint: TernaryLinearCheckpoint,
) -> Result<()> {
    if checkpoint.master_bits.len() != layer.master.len() {
        bail!("checkpoint ternary projection shape mismatch");
    }
    layer.master = Fp16Storage::from_bits(checkpoint.master_bits);
    layer.refresh_codes();
    Ok(())
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
    chunk_plan: HyenaChunkPlan,
    d_model: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
struct HyenaBlockCache {
    input: Vec<f32>,
    normalized_input: Vec<f32>,
    gated_projection: Vec<f32>,
    mixed: Vec<f32>,
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
            chunk_plan: HyenaChunkPlan::new(cfg.hyena_chunk_len, cfg.hyena_kernel_len)
                .expect("TrainConfig validates the Hyena chunk plan"),
            d_model: cfg.d_model,
        }
    }
    fn forward(&self, x: &[f32], batch: usize, time: usize) -> Result<Vec<f32>> {
        Ok(self.forward_with_cache(x, batch, time)?.0)
    }

    fn forward_with_cache(
        &self,
        x: &[f32],
        batch: usize,
        time: usize,
    ) -> Result<(Vec<f32>, HyenaBlockCache)> {
        let normalized_input = rms_norm(x, batch * time, self.d_model)?;
        let mut projected = self.input.forward(&normalized_input, batch * time)?;
        for n in 0..batch * time {
            for c in 0..self.d_model {
                let gate_index = n * 2 * self.d_model + self.d_model + c;
                projected[gate_index] = projected[gate_index].tanh();
            }
        }
        let mut mixed = causal_chunked_conv_implicit_strided(
            &projected,
            &self.filter,
            batch,
            time,
            self.d_model,
            2 * self.d_model,
            0,
            self.chunk_plan,
        )?;
        let pre_gate_mixed = mixed.clone();
        for n in 0..batch * time {
            for c in 0..self.d_model {
                mixed[n * self.d_model + c] *= projected[n * 2 * self.d_model + self.d_model + c];
            }
        }
        let update = self.output.forward(&mixed, batch * time)?;
        let output = x.iter().zip(update).map(|(a, b)| a + b).collect();
        Ok((
            output,
            HyenaBlockCache {
                input: x.to_vec(),
                normalized_input,
                gated_projection: projected,
                mixed: pre_gate_mixed,
            },
        ))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn backward(
        &self,
        cache: &HyenaBlockCache,
        output_gradient: &[f32],
        batch: usize,
        time: usize,
    ) -> Result<HyenaBlockBackward> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Hyena backward row overflow"))?;
        if output_gradient.len() != rows * self.d_model {
            bail!("Hyena block backward output-gradient shape mismatch");
        }
        let mut gated_mixed = cache.mixed.clone();
        for row in 0..rows {
            for channel in 0..self.d_model {
                gated_mixed[row * self.d_model + channel] *=
                    cache.gated_projection[row * 2 * self.d_model + self.d_model + channel];
            }
        }
        let output_projection = self
            .output
            .backward_ste(&gated_mixed, output_gradient, rows)?;
        let gate = hyena_gate_backward(
            &cache.mixed,
            &cache.gated_projection,
            &output_projection.input_gradient,
            self.d_model,
        )?;
        let plan = self.chunk_plan.for_sequence(time)?;
        let mut signal = vec![0.0; rows * self.d_model];
        let mut filter_values = vec![0.0; self.d_model * plan.kernel_len];
        for channel in 0..self.d_model {
            self.filter.generate_channel_prefix(
                channel,
                &mut filter_values[channel * plan.kernel_len..(channel + 1) * plan.kernel_len],
                time,
            )?;
            for row in 0..rows {
                signal[row * self.d_model + channel] =
                    cache.gated_projection[row * 2 * self.d_model + channel];
            }
        }
        let CausalConvBackward {
            input_gradient: signal_gradient,
            filter_gradient,
        } = crate::hyena::causal_chunked_conv_backward(
            &signal,
            &filter_values,
            &gate.mixed_gradient,
            batch,
            time,
            self.d_model,
            plan,
        )?;
        let filter =
            self.filter
                .backward_prefix(self.d_model, &filter_gradient, plan.kernel_len, time)?;
        let mut projection_gradient = gate.projection_gradient;
        for (destination, source) in projection_gradient
            .chunks_exact_mut(2 * self.d_model)
            .zip(signal_gradient.chunks_exact(self.d_model))
        {
            for channel in 0..self.d_model {
                destination[channel] += source[channel];
            }
        }
        let input_projection =
            self.input
                .backward_ste(&cache.normalized_input, &projection_gradient, rows)?;
        let normalized_input_gradient = rms_norm_backward(
            &cache.input,
            &cache.normalized_input,
            &input_projection.input_gradient,
            rows,
            self.d_model,
        )?;
        let input_gradient = output_gradient
            .iter()
            .zip(&normalized_input_gradient)
            .map(|(residual, update)| residual + update)
            .collect();
        Ok(HyenaBlockBackward {
            input_gradient,
            input_projection,
            output_projection,
            filter,
        })
    }

    fn apply_stateless_gradient(
        &mut self,
        gradient: &HyenaBlockBackward,
        learning_rate: f32,
    ) -> Result<()> {
        self.input.apply_ste_gradient(
            &gradient.input_projection.latent_weight_gradient,
            learning_rate,
        )?;
        self.output.apply_ste_gradient(
            &gradient.output_projection.latent_weight_gradient,
            learning_rate,
        )?;
        self.filter
            .apply_stateless_gradient(&gradient.filter, learning_rate)
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
        let mut mixed = runtime.causal_chunked_conv_implicit_strided_forward(
            &gated,
            &self.filter,
            batch,
            time,
            self.d_model,
            2 * self.d_model,
            0,
            self.chunk_plan,
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
        if self.chunk_plan.for_sequence(time)?.kernel_len != time {
            bail!(
                "resident Metal long-convolution reference does not yet implement the configured bounded Hyena receptive field"
            );
        }
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena block row overflow"))?;
        let (positive, negative, scales) = self.input.metal_parts();
        runtime.resident_input_projection(
            slot,
            rows,
            self.d_model,
            positive,
            negative,
            scales,
            None,
        )?;
        runtime.resident_hyena_mixer(
            slot,
            batch,
            time,
            self.d_model,
            &self.filter,
            self.chunk_plan,
            None,
        )?;
        let (positive, negative, scales) = self.output.metal_parts();
        runtime.resident_output_projection(slot, rows, self.d_model, positive, negative, scales)
    }

    /// Training variant of the resident forward path. Its cache owns the
    /// values needed for this block's reverse pass, so the next block may
    /// immediately reuse all runtime scratch buffers.
    #[cfg(target_os = "macos")]
    fn forward_metal_resident_with_cache(
        &self,
        runtime: &crate::metal::MetalRuntime,
        slot: crate::metal::ResidentActivationSlot,
        batch: usize,
        time: usize,
    ) -> Result<(
        crate::metal::ResidentActivationSlot,
        crate::metal::ResidentHyenaBlockCache,
    )> {
        if self.chunk_plan.for_sequence(time)?.kernel_len != time {
            bail!(
                "resident Metal long-convolution reference does not yet implement the configured bounded Hyena receptive field"
            );
        }
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena block row overflow"))?;
        let cache = runtime.new_hyena_block_cache(rows, self.d_model)?;
        let (positive, negative, scales) = self.input.metal_parts();
        runtime.resident_input_projection(
            slot,
            rows,
            self.d_model,
            positive,
            negative,
            scales,
            Some(&cache),
        )?;
        runtime.resident_hyena_mixer(
            slot,
            batch,
            time,
            self.d_model,
            &self.filter,
            self.chunk_plan,
            Some(&cache),
        )?;
        let (positive, negative, scales) = self.output.metal_parts();
        let next = runtime.resident_output_projection(
            slot,
            rows,
            self.d_model,
            positive,
            negative,
            scales,
        )?;
        Ok((next, cache))
    }

    /// Training forward using persistent Metal-owned ternary projections.
    #[cfg(target_os = "macos")]
    fn forward_metal_resident_with_trainable_cache(
        &self,
        runtime: &crate::metal::MetalRuntime,
        slot: crate::metal::ResidentActivationSlot,
        weights: &MetalResidentHyenaBlockWeights,
        batch: usize,
        time: usize,
    ) -> Result<(
        crate::metal::ResidentActivationSlot,
        crate::metal::ResidentHyenaBlockCache,
    )> {
        if self.chunk_plan.for_sequence(time)?.kernel_len != time {
            bail!(
                "resident Metal long-convolution reference does not yet implement the configured bounded Hyena receptive field"
            );
        }
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal Hyena block row overflow"))?;
        let cache = runtime.new_hyena_block_cache(rows, self.d_model)?;
        runtime.resident_input_projection_trainable(
            slot,
            rows,
            self.d_model,
            &weights.input,
            Some(&cache),
        )?;
        runtime.resident_hyena_mixer_trainable(
            slot,
            batch,
            time,
            self.d_model,
            &weights.filter,
            self.chunk_plan,
            Some(&cache),
        )?;
        let next = runtime.resident_output_projection_trainable(
            slot,
            rows,
            self.d_model,
            &weights.output,
        )?;
        Ok((next, cache))
    }
}

const RMS_EPSILON: f32 = 1e-5;

fn rms_norm(input: &[f32], rows: usize, channels: usize) -> Result<Vec<f32>> {
    if rows == 0 || channels == 0 || input.len() != rows * channels {
        bail!("RMSNorm shape mismatch");
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let values = &input[row * channels..(row + 1) * channels];
        let inverse_rms = (values.iter().map(|value| value * value).sum::<f32>() / channels as f32
            + RMS_EPSILON)
            .sqrt()
            .recip();
        for (destination, source) in output[row * channels..(row + 1) * channels]
            .iter_mut()
            .zip(values)
        {
            *destination = source * inverse_rms;
        }
    }
    Ok(output)
}

#[cfg_attr(not(test), allow(dead_code))]
fn rms_norm_backward(
    input: &[f32],
    normalized: &[f32],
    output_gradient: &[f32],
    rows: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    if rows == 0
        || channels == 0
        || input.len() != rows * channels
        || normalized.len() != input.len()
        || output_gradient.len() != input.len()
    {
        bail!("RMSNorm backward shape mismatch");
    }
    let mut input_gradient = vec![0.0; input.len()];
    for row in 0..rows {
        let input_row = &input[row * channels..(row + 1) * channels];
        let normalized_row = &normalized[row * channels..(row + 1) * channels];
        let gradient_row = &output_gradient[row * channels..(row + 1) * channels];
        let inverse_rms = (input_row.iter().map(|value| value * value).sum::<f32>()
            / channels as f32
            + RMS_EPSILON)
            .sqrt()
            .recip();
        let projection = normalized_row
            .iter()
            .zip(gradient_row)
            .map(|(value, gradient)| value * gradient)
            .sum::<f32>()
            / channels as f32;
        for channel in 0..channels {
            input_gradient[row * channels + channel] =
                inverse_rms * (gradient_row[channel] - normalized_row[channel] * projection);
        }
    }
    Ok(input_gradient)
}

/// Inference core. `mtp_logits` exposes separate t+1 and t+2 pretraining heads.
#[derive(Clone, Debug)]
pub struct UllisHyena {
    pub cfg: TrainConfig,
    /// Tied input/output table is stored in FP16; dot products still widen to
    /// FP32 in the numerical reference and future Metal reduction kernels.
    embedding: Fp16Storage,
    blocks: Vec<HyenaBlock>,
    mtp_one: TernaryLinear,
    mtp_two: TernaryLinear,
}
impl UllisHyena {
    pub fn new(cfg: TrainConfig) -> Result<Self> {
        cfg.validate()?;
        let embedding = Fp16Storage::from_f32(seeded_values(
            cfg.vocab_size * cfg.d_model,
            (cfg.d_model as f32).sqrt().recip(),
            cfg.seed,
        ));
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

    /// Captures all persistent model parameters without any optimiser state.
    pub fn checkpoint(&self) -> ModelCheckpoint {
        ModelCheckpoint {
            format_version: 1,
            config: self.cfg.clone(),
            embedding_bits: self.embedding.as_bits().to_vec(),
            blocks: self
                .blocks
                .iter()
                .map(|block| HyenaBlockCheckpoint {
                    input: TernaryLinearCheckpoint {
                        master_bits: block.input.master.as_bits().to_vec(),
                    },
                    output: TernaryLinearCheckpoint {
                        master_bits: block.output.master.as_bits().to_vec(),
                    },
                    freq: block.filter.freq.clone(),
                    phase: block.filter.phase.clone(),
                    decay: block.filter.decay.clone(),
                })
                .collect(),
            mtp_one: TernaryLinearCheckpoint {
                master_bits: self.mtp_one.master.as_bits().to_vec(),
            },
            mtp_two: TernaryLinearCheckpoint {
                master_bits: self.mtp_two.master.as_bits().to_vec(),
            },
        }
    }

    /// Restores a checkpoint and rebuilds the derived packed ternary plane.
    pub fn from_checkpoint(checkpoint: ModelCheckpoint) -> Result<Self> {
        if checkpoint.format_version != 1 {
            bail!(
                "unsupported Ullis checkpoint version {}",
                checkpoint.format_version
            );
        }
        checkpoint.config.validate()?;
        if checkpoint.embedding_bits.len()
            != checkpoint.config.vocab_size * checkpoint.config.d_model
            || checkpoint.blocks.len() != checkpoint.config.n_layers
        {
            bail!("checkpoint model shapes do not match its configuration");
        }
        let mut model = Self::new(checkpoint.config)?;
        model.embedding = Fp16Storage::from_bits(checkpoint.embedding_bits);
        for (block, saved) in model.blocks.iter_mut().zip(checkpoint.blocks) {
            restore_ternary_linear(&mut block.input, saved.input)?;
            restore_ternary_linear(&mut block.output, saved.output)?;
            let expected = block.filter.freq.len();
            if saved.freq.len() != expected
                || saved.phase.len() != expected
                || saved.decay.len() != expected
                || saved
                    .freq
                    .iter()
                    .chain(&saved.phase)
                    .chain(&saved.decay)
                    .any(|v| !v.is_finite())
            {
                bail!("checkpoint implicit-filter shape/value mismatch");
            }
            block.filter.freq = saved.freq;
            block.filter.phase = saved.phase;
            block.filter.decay = saved.decay;
        }
        restore_ternary_linear(&mut model.mtp_one, checkpoint.mtp_one)?;
        restore_ternary_linear(&mut model.mtp_two, checkpoint.mtp_two)?;
        Ok(model)
    }
    pub fn hidden(&self, ids: &[u32], batch: usize, time: usize) -> Result<Vec<f32>> {
        let mut x = self.input_embeddings(ids, batch, time)?;
        for block in &self.blocks {
            x = block.forward(&x, batch, time)?;
        }
        Ok(x)
    }

    fn input_embeddings(&self, ids: &[u32], batch: usize, time: usize) -> Result<Vec<f32>> {
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
            for channel in 0..d {
                x[row * d + channel] = self.embedding.get(id * d + channel);
            }
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
            for channel in 0..self.cfg.d_model {
                x[row * self.cfg.d_model + channel] =
                    self.embedding.get(id * self.cfg.d_model + channel);
            }
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
            for channel in 0..d {
                embedding_stream[row * d + channel] = self.embedding.get(id * d + channel);
            }
        }
        let mut slot = runtime.upload_resident_activations(&embedding_stream, rows, d)?;
        for block in &self.blocks {
            slot = block.forward_metal_resident(runtime, slot, batch, time)?;
        }
        runtime.download_resident_activations(slot, rows, d)
    }

    /// Runs the resident forward graph while retaining one GPU-only forward
    /// cache per block for the upcoming reverse traversal.  The returned
    /// caches intentionally remain opaque: their only supported consumer is
    /// the Metal backward graph, not host-side activation inspection.
    #[cfg(target_os = "macos")]
    pub fn hidden_metal_resident_for_backward(
        &self,
        runtime: &crate::metal::MetalRuntime,
        ids: &[u32],
        batch: usize,
        time: usize,
    ) -> Result<(Vec<f32>, Vec<crate::metal::ResidentHyenaBlockCache>)> {
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
            for channel in 0..d {
                embedding_stream[row * d + channel] = self.embedding.get(id * d + channel);
            }
        }
        let mut slot = runtime.upload_resident_activations(&embedding_stream, rows, d)?;
        let mut caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next, cache) =
                block.forward_metal_resident_with_cache(runtime, slot, batch, time)?;
            slot = next;
            caches.push(cache);
        }
        Ok((
            runtime.download_resident_activations(slot, rows, d)?,
            caches,
        ))
    }

    /// Uploads each Hyena projection once for a persistent Metal training run.
    #[cfg(target_os = "macos")]
    pub fn new_metal_resident_training_state(
        &self,
        runtime: &crate::metal::MetalRuntime,
    ) -> Result<MetalResidentHyenaTrainingState> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            blocks.push(MetalResidentHyenaBlockWeights {
                input: runtime.upload_trainable_fp16_ternary_weights(
                    &block.input.master,
                    block.input.in_features,
                    block.input.out_features,
                    block.input.threshold_ratio,
                )?,
                output: runtime.upload_trainable_fp16_ternary_weights(
                    &block.output.master,
                    block.output.in_features,
                    block.output.out_features,
                    block.output.threshold_ratio,
                )?,
                filter: runtime
                    .upload_resident_implicit_filter_parameters(&block.filter, self.cfg.d_model)?,
            });
        }
        Ok(MetalResidentHyenaTrainingState {
            blocks,
            mtp_one: runtime.upload_trainable_fp16_ternary_weights(
                &self.mtp_one.master,
                self.mtp_one.in_features,
                self.mtp_one.out_features,
                self.mtp_one.threshold_ratio,
            )?,
            mtp_two: runtime.upload_trainable_fp16_ternary_weights(
                &self.mtp_two.master,
                self.mtp_two.in_features,
                self.mtp_two.out_features,
                self.mtp_two.threshold_ratio,
            )?,
            embedding: runtime.upload_resident_fp16_parameters(&self.embedding)?,
        })
    }

    /// Takes a lossless checkpoint of the Metal-owned training weights.
    ///
    /// This is intentionally an explicit synchronization boundary: normal
    /// training retains no CPU mirror of these mutable tensors.
    #[cfg(target_os = "macos")]
    pub fn checkpoint_metal_resident(
        &self,
        runtime: &crate::metal::MetalRuntime,
        state: &MetalResidentHyenaTrainingState,
    ) -> Result<ModelCheckpoint> {
        if state.blocks.len() != self.blocks.len() {
            bail!("Metal resident checkpoint state does not match model");
        }
        let mut checkpoint = self.checkpoint();
        checkpoint.embedding_bits = runtime
            .download_resident_fp16_parameters(&state.embedding)?
            .as_bits()
            .to_vec();
        for (saved, weights) in checkpoint.blocks.iter_mut().zip(&state.blocks) {
            saved.input.master_bits = runtime
                .download_trainable_fp16_ternary_weights(&weights.input)?
                .0
                .as_bits()
                .to_vec();
            saved.output.master_bits = runtime
                .download_trainable_fp16_ternary_weights(&weights.output)?
                .0
                .as_bits()
                .to_vec();
            let (freq, phase, decay) =
                runtime.download_resident_implicit_filter_parameters(&weights.filter)?;
            saved.freq = freq
                .as_bits()
                .iter()
                .map(|&bits| crate::precision::Fp16::from_bits(bits).to_f32())
                .collect();
            saved.phase = phase
                .as_bits()
                .iter()
                .map(|&bits| crate::precision::Fp16::from_bits(bits).to_f32())
                .collect();
            saved.decay = decay
                .as_bits()
                .iter()
                .map(|&bits| crate::precision::Fp16::from_bits(bits).to_f32())
                .collect();
        }
        checkpoint.mtp_one.master_bits = runtime
            .download_trainable_fp16_ternary_weights(&state.mtp_one)?
            .0
            .as_bits()
            .to_vec();
        checkpoint.mtp_two.master_bits = runtime
            .download_trainable_fp16_ternary_weights(&state.mtp_two)?
            .0
            .as_bits()
            .to_vec();
        Ok(checkpoint)
    }

    /// Runs a cached resident forward pass using Metal-owned projection state.
    #[cfg(target_os = "macos")]
    pub fn hidden_metal_resident_for_training(
        &self,
        runtime: &crate::metal::MetalRuntime,
        state: &MetalResidentHyenaTrainingState,
        ids: &[u32],
        batch: usize,
        time: usize,
    ) -> Result<(Vec<f32>, Vec<crate::metal::ResidentHyenaBlockCache>)> {
        let (slot, caches) =
            self.forward_metal_resident_training_cached(runtime, state, ids, batch, time)?;
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("token shape overflow"))?;
        Ok((
            runtime.download_resident_activations(slot, rows, self.cfg.d_model)?,
            caches,
        ))
    }

    #[cfg(target_os = "macos")]
    fn forward_metal_resident_training_cached(
        &self,
        runtime: &crate::metal::MetalRuntime,
        state: &MetalResidentHyenaTrainingState,
        ids: &[u32],
        batch: usize,
        time: usize,
    ) -> Result<(
        crate::metal::ResidentActivationSlot,
        Vec<crate::metal::ResidentHyenaBlockCache>,
    )> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("token shape overflow"))?;
        if batch == 0
            || batch > self.cfg.batch_size
            || ids.len() != rows
            || time == 0
            || time > self.cfg.context_len
            || state.blocks.len() != self.blocks.len()
        {
            bail!("Metal resident training state or token shape is invalid");
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
            for channel in 0..d {
                embedding_stream[row * d + channel] = self.embedding.get(id * d + channel);
            }
        }
        let mut slot = runtime.upload_resident_activations(&embedding_stream, rows, d)?;
        let mut caches = Vec::with_capacity(self.blocks.len());
        for (block, weights) in self.blocks.iter().zip(&state.blocks) {
            let (next, cache) = block
                .forward_metal_resident_with_trainable_cache(runtime, slot, weights, batch, time)?;
            slot = next;
            caches.push(cache);
        }
        Ok((slot, caches))
    }

    /// Consumes a terminal gradient and updates every resident projection in
    /// reverse order. Only stack-input and filter gradients are read back.
    #[cfg(target_os = "macos")]
    pub fn hidden_metal_backward_update_resident(
        &self,
        runtime: &crate::metal::MetalRuntime,
        state: &MetalResidentHyenaTrainingState,
        ids: &[u32],
        output_gradient: &[f32],
        batch: usize,
        time: usize,
        learning_rate: f32,
        train_filters: bool,
    ) -> Result<MetalResidentHyenaProjectionStep> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal stack backward row overflow"))?;
        let elements = rows
            .checked_mul(self.cfg.d_model)
            .ok_or_else(|| anyhow::anyhow!("Metal stack backward activation overflow"))?;
        if output_gradient.len() != elements
            || output_gradient.iter().any(|value| !value.is_finite())
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
        {
            bail!("Metal resident training backward shape/value mismatch");
        }
        let (_, caches) =
            self.hidden_metal_resident_for_training(runtime, state, ids, batch, time)?;
        let mut gradient_slot =
            runtime.upload_resident_gradient(output_gradient, rows, self.cfg.d_model)?;
        let mut reverse_filter_gradients = Vec::with_capacity(self.blocks.len());
        for ((block, cache), weights) in self
            .blocks
            .iter()
            .zip(caches.iter())
            .zip(state.blocks.iter())
            .rev()
        {
            let plan = block.chunk_plan.for_sequence(time)?;
            // The forward cache was produced from the FP16 resident master.
            // Recreate that same compact state for the exact-reference
            // convolution/filter backward; this is a tiny O(D*order)
            // validation bridge, never an activation or optimiser tensor.
            let mut resident_filter = block.filter.clone();
            if train_filters {
                let (freq, phase, decay) =
                    runtime.download_resident_implicit_filter_parameters(&weights.filter)?;
                resident_filter.freq = (0..freq.len()).map(|index| freq.get(index)).collect();
                resident_filter.phase = (0..phase.len()).map(|index| phase.get(index)).collect();
                resident_filter.decay = (0..decay.len()).map(|index| decay.get(index)).collect();
            }
            let filter = resident_filter.generate(self.cfg.d_model, plan.kernel_len)?;
            let destination_slot = gradient_slot.other();
            let backward = runtime.hyena_block_backward_cached_and_update_resident(
                cache,
                gradient_slot,
                destination_slot,
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                &weights.input,
                &weights.output,
                &filter,
                batch,
                time,
                block.chunk_plan,
                learning_rate,
                train_filters,
            )?;
            gradient_slot = destination_slot;
            reverse_filter_gradients.push(backward.filter_gradient);
            let filter_gradient = reverse_filter_gradients
                .last()
                .expect("just pushed filter gradient");
            let filter_backward = resident_filter.backward_prefix(
                self.cfg.d_model,
                filter_gradient,
                plan.kernel_len,
                time,
            )?;
            runtime.resident_implicit_filter_stateless_sgd(
                &weights.filter,
                &filter_backward,
                learning_rate,
            )?;
        }
        reverse_filter_gradients.reverse();
        Ok(MetalResidentHyenaProjectionStep {
            input_gradient: runtime.download_resident_gradient(
                gradient_slot,
                rows,
                self.cfg.d_model,
            )?,
            filter_gradients: reverse_filter_gradients,
        })
    }

    /// Continues resident reverse mode from a gradient already produced by a
    /// Metal loss/head graph. This is the no-terminal-readback training path.
    #[cfg(target_os = "macos")]
    fn hidden_metal_backward_update_from_resident_gradient(
        &self,
        runtime: &crate::metal::MetalRuntime,
        state: &MetalResidentHyenaTrainingState,
        caches: &[crate::metal::ResidentHyenaBlockCache],
        mut gradient_slot: crate::metal::ResidentGradientSlot,
        batch: usize,
        time: usize,
        learning_rate: f32,
        train_filters: bool,
    ) -> Result<MetalResidentHyenaProjectionStep> {
        if caches.len() != self.blocks.len() || !learning_rate.is_finite() || learning_rate <= 0.0 {
            bail!("Metal resident cached backward state/value mismatch");
        }
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal stack backward row overflow"))?;
        let mut reverse_filter_gradients = Vec::with_capacity(self.blocks.len());
        for ((block, cache), weights) in self
            .blocks
            .iter()
            .zip(caches.iter())
            .zip(state.blocks.iter())
            .rev()
        {
            let plan = block.chunk_plan.for_sequence(time)?;
            let (freq, phase, decay) =
                runtime.download_resident_implicit_filter_parameters(&weights.filter)?;
            let mut resident_filter = block.filter.clone();
            resident_filter.freq = (0..freq.len()).map(|index| freq.get(index)).collect();
            resident_filter.phase = (0..phase.len()).map(|index| phase.get(index)).collect();
            resident_filter.decay = (0..decay.len()).map(|index| decay.get(index)).collect();
            let filter = resident_filter.generate(self.cfg.d_model, plan.kernel_len)?;
            let destination_slot = gradient_slot.other();
            let backward = runtime.hyena_block_backward_cached_and_update_resident(
                cache,
                gradient_slot,
                destination_slot,
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                &weights.input,
                &weights.output,
                &filter,
                batch,
                time,
                block.chunk_plan,
                learning_rate,
                train_filters,
            )?;
            gradient_slot = destination_slot;
            if train_filters {
                let filter_backward = resident_filter.backward_prefix(
                    self.cfg.d_model,
                    &backward.filter_gradient,
                    plan.kernel_len,
                    time,
                )?;
                reverse_filter_gradients.push(backward.filter_gradient);
                runtime.resident_implicit_filter_stateless_sgd(
                    &weights.filter,
                    &filter_backward,
                    learning_rate,
                )?;
            }
        }
        reverse_filter_gradients.reverse();
        Ok(MetalResidentHyenaProjectionStep {
            input_gradient: runtime.download_resident_gradient(
                gradient_slot,
                rows,
                self.cfg.d_model,
            )?,
            filter_gradients: reverse_filter_gradients,
        })
    }

    /// One complete resident Metal MTP step. MTP heads, streamed loss, their
    /// summed terminal gradient, and all Hyena projection updates stay on GPU;
    /// only compact loss/filter statistics cross the explicit validation edge.
    #[cfg(target_os = "macos")]
    pub fn train_step_metal_resident_stateless_sgd(
        &self,
        runtime: &crate::metal::MetalRuntime,
        state: &MetalResidentHyenaTrainingState,
        ids: &[u32],
        batch: usize,
        time: usize,
        learning_rate: f32,
        train_filters: bool,
    ) -> Result<MtpLoss> {
        if time < 3 || !learning_rate.is_finite() || learning_rate <= 0.0 {
            bail!("Metal resident MTP training shape/value mismatch");
        }
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal resident MTP rows overflow"))?;
        let (hidden_slot, caches) =
            self.forward_metal_resident_training_cached(runtime, state, ids, batch, time)?;
        let first_head = runtime.resident_ternary_head_forward_trainable(
            hidden_slot,
            rows,
            self.cfg.d_model,
            &state.mtp_one,
        )?;
        let first_loss = runtime.streamed_cross_entropy_fp16_from_activation(
            first_head,
            &state.embedding,
            ids,
            batch,
            time,
            self.cfg.d_model,
            self.cfg.vocab_size,
            1,
        )?;
        runtime.resident_ternary_head_backward_update(
            hidden_slot,
            first_loss.gradient_slot,
            first_loss.gradient_slot.other(),
            rows,
            self.cfg.d_model,
            &state.mtp_one,
            learning_rate,
            false,
        )?;
        let second_head = runtime.resident_ternary_head_forward_trainable(
            hidden_slot,
            rows,
            self.cfg.d_model,
            &state.mtp_two,
        )?;
        let second_loss = runtime.streamed_cross_entropy_fp16_from_activation(
            second_head,
            &state.embedding,
            ids,
            batch,
            time,
            self.cfg.d_model,
            self.cfg.vocab_size,
            2,
        )?;
        let terminal_gradient = runtime.resident_ternary_head_backward_update(
            hidden_slot,
            second_loss.gradient_slot,
            second_loss.gradient_slot,
            rows,
            self.cfg.d_model,
            &state.mtp_two,
            learning_rate,
            true,
        )?;
        self.hidden_metal_backward_update_from_resident_gradient(
            runtime,
            state,
            &caches,
            terminal_gradient,
            batch,
            time,
            learning_rate,
            train_filters,
        )?;
        Ok(MtpLoss {
            next_token: first_loss.loss_sum / first_loss.token_count as f32,
            second_token: second_loss.loss_sum / second_loss.token_count as f32,
            next_token_count: first_loss.token_count,
            second_token_count: second_loss.token_count,
        })
    }

    /// Runs a complete cached Metal reverse pass through every Hyena block.
    ///
    /// Forward activations and the reverse stream stay resident in Metal
    /// buffers for the complete block traversal. The reference API still
    /// publishes each block's parameter gradients to the CPU updater, but no
    /// input gradient is round-tripped between blocks.
    #[cfg(target_os = "macos")]
    pub fn hidden_metal_backward_reference(
        &self,
        runtime: &crate::metal::MetalRuntime,
        ids: &[u32],
        output_gradient: &[f32],
        batch: usize,
        time: usize,
    ) -> Result<MetalHyenaStackBackward> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("Metal stack backward row overflow"))?;
        let elements = rows
            .checked_mul(self.cfg.d_model)
            .ok_or_else(|| anyhow::anyhow!("Metal stack backward activation overflow"))?;
        if output_gradient.len() != elements
            || output_gradient.iter().any(|value| !value.is_finite())
        {
            bail!("Metal stack backward output-gradient shape/value mismatch");
        }
        let (_, caches) = self.hidden_metal_resident_for_backward(runtime, ids, batch, time)?;
        let mut gradient_slot =
            runtime.upload_resident_gradient(output_gradient, rows, self.cfg.d_model)?;
        let mut reverse_blocks = Vec::with_capacity(self.blocks.len());
        for (block, cache) in self.blocks.iter().zip(caches.iter()).rev() {
            let plan = block.chunk_plan.for_sequence(time)?;
            let filter = block.filter.generate(self.cfg.d_model, plan.kernel_len)?;
            let (input_positive, input_negative, input_scales) = block.input.metal_parts();
            let (output_positive, output_negative, output_scales) = block.output.metal_parts();
            let destination_slot = gradient_slot.other();
            let backward = runtime.hyena_block_backward_cached_from_resident_gradient(
                cache,
                gradient_slot,
                destination_slot,
                input_positive,
                input_negative,
                input_scales,
                output_positive,
                output_negative,
                output_scales,
                &filter,
                batch,
                time,
                block.chunk_plan,
            )?;
            gradient_slot = destination_slot;
            reverse_blocks.push(backward);
        }
        reverse_blocks.reverse();
        Ok(MetalHyenaStackBackward {
            input_gradient: runtime.download_resident_gradient(
                gradient_slot,
                rows,
                self.cfg.d_model,
            )?,
            blocks: reverse_blocks,
        })
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

    /// Computes the exact cross-entropy derivative at each MTP projection
    /// output without materializing logits. The vocabulary is scanned twice
    /// per supervised row (normalizer, then expectation), retaining only the
    /// `[B*T,D]` head gradients required by the next backward stage.
    pub fn streamed_mtp_head_backward(
        &self,
        ids: &[u32],
        batch: usize,
        time: usize,
    ) -> Result<MtpHeadBackward> {
        if time < 3 {
            bail!("MTP backward needs at least three tokens per sequence");
        }
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        let hidden = self.hidden(ids, batch, time)?;
        let one = self.mtp_one.forward(&hidden, rows)?;
        let (next_sum, next_count, next_head_gradient) =
            self.streamed_cross_entropy_horizon_backward(&one, ids, batch, time, 1)?;
        drop(one);
        let two = self.mtp_two.forward(&hidden, rows)?;
        let (second_sum, second_count, second_head_gradient) =
            self.streamed_cross_entropy_horizon_backward(&two, ids, batch, time, 2)?;
        Ok(MtpHeadBackward {
            loss: MtpLoss {
                next_token: next_sum / next_count as f32,
                second_token: second_sum / second_count as f32,
                next_token_count: next_count,
                second_token_count: second_count,
            },
            next_head_gradient,
            second_head_gradient,
        })
    }

    /// Backpropagates both normalized MTP losses through their ternary heads.
    /// The returned hidden gradient is their sum and is ready to enter the
    /// final Hyena block. No activation beyond the caller-owned hidden state
    /// and local projection gradients is retained.
    pub fn backward_mtp_heads(
        &self,
        hidden: &[f32],
        batch: usize,
        time: usize,
        head_backward: &MtpHeadBackward,
    ) -> Result<MtpProjectionBackward> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        if rows == 0
            || hidden.len() != rows * self.cfg.d_model
            || head_backward.next_head_gradient.len() != hidden.len()
            || head_backward.second_head_gradient.len() != hidden.len()
        {
            bail!("MTP projection backward shape is invalid");
        }
        let next_projection =
            self.mtp_one
                .backward_ste(hidden, &head_backward.next_head_gradient, rows)?;
        let second_projection =
            self.mtp_two
                .backward_ste(hidden, &head_backward.second_head_gradient, rows)?;
        let hidden_gradient = next_projection
            .input_gradient
            .iter()
            .zip(&second_projection.input_gradient)
            .map(|(next, second)| next + second)
            .collect();
        Ok(MtpProjectionBackward {
            hidden_gradient,
            next_projection,
            second_projection,
        })
    }

    /// Runs one reference training step with no persistent optimizer state.
    ///
    /// A single forward pass retains one compact cache per block, eliminating
    /// the former quadratic recomputation of all earlier layers during each
    /// reverse step. This deliberately trades `O(L*B*T*D)` activation RAM for
    /// linear layer-dispatch time. Ternary projections and compact filter
    /// parameters are updated immediately after their local gradients are
    /// formed. The tied embedding table is frozen for now: an exact
    /// output-softmax update is dense over the vocabulary and belongs in the
    /// dedicated streaming embedding updater.
    pub fn train_step_stateless_sgd(
        &mut self,
        ids: &[u32],
        batch: usize,
        time: usize,
        learning_rate: f32,
    ) -> Result<MtpLoss> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            bail!("training learning rate must be finite and positive");
        }
        if time < 3 {
            bail!("MTP training needs at least three tokens per sequence");
        }
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        let mut hidden = self.input_embeddings(ids, batch, time)?;
        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next, cache) = block.forward_with_cache(&hidden, batch, time)?;
            hidden = next;
            block_caches.push(cache);
        }
        let one = self.mtp_one.forward(&hidden, rows)?;
        let next_normalizer = self.cross_entropy_normalizer(&one, ids, batch, time, 1)?;
        let (next_sum, next_count, next_head_gradient) =
            self.streamed_cross_entropy_horizon_backward(&one, ids, batch, time, 1)?;
        let two = self.mtp_two.forward(&hidden, rows)?;
        let second_normalizer = self.cross_entropy_normalizer(&two, ids, batch, time, 2)?;
        let (second_sum, second_count, second_head_gradient) =
            self.streamed_cross_entropy_horizon_backward(&two, ids, batch, time, 2)?;
        let head_backward = MtpHeadBackward {
            loss: MtpLoss {
                next_token: next_sum / next_count as f32,
                second_token: second_sum / second_count as f32,
                next_token_count: next_count,
                second_token_count: second_count,
            },
            next_head_gradient,
            second_head_gradient,
        };
        let mtp_backward = self.backward_mtp_heads(&hidden, batch, time, &head_backward)?;
        self.mtp_one.apply_ste_gradient(
            &mtp_backward.next_projection.latent_weight_gradient,
            learning_rate,
        )?;
        self.mtp_two.apply_ste_gradient(
            &mtp_backward.second_projection.latent_weight_gradient,
            learning_rate,
        )?;
        let mut gradient = mtp_backward.hidden_gradient;
        for (block, cache) in self.blocks.iter_mut().zip(block_caches.iter()).rev() {
            let block_backward = block.backward(cache, &gradient, batch, time)?;
            block.apply_stateless_gradient(&block_backward, learning_rate)?;
            gradient = block_backward.input_gradient;
        }
        self.apply_tied_embedding_gradient(
            &one,
            &two,
            &next_normalizer,
            &second_normalizer,
            ids,
            batch,
            time,
            &gradient,
            learning_rate,
        )?;
        Ok(head_backward.loss)
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
                    .enumerate()
                    .map(|(channel, a)| a * self.embedding.get(v * self.cfg.d_model + channel))
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
            let logit = self.dot_embedding(state, token);
            max_logit = max_logit.max(logit);
            if token == target {
                target_logit = logit;
            }
        }
        let mut exp_sum = 0.0;
        for token in 0..self.cfg.vocab_size {
            exp_sum += (self.dot_embedding(state, token) - max_logit).exp();
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

    fn streamed_cross_entropy_horizon_backward(
        &self,
        head: &[f32],
        ids: &[u32],
        batch: usize,
        time: usize,
        horizon: usize,
    ) -> Result<(f32, usize, Vec<f32>)> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        if horizon == 0
            || horizon >= time
            || head.len() != rows * self.cfg.d_model
            || ids.len() != rows
        {
            bail!("MTP cross-entropy backward shape or horizon is invalid");
        }
        let d = self.cfg.d_model;
        let vocab = self.cfg.vocab_size;
        let mut loss_sum = 0.0;
        let mut count = 0;
        let mut gradient = vec![0.0; head.len()];
        for sequence in 0..batch {
            let base = sequence * time;
            for position in 0..time - horizon {
                let row = base + position;
                let target = ids[row + horizon] as usize;
                if target >= vocab {
                    bail!("token id {target} out of vocabulary");
                }
                let state = &head[row * d..(row + 1) * d];
                let mut max_logit = f32::NEG_INFINITY;
                for token in 0..vocab {
                    max_logit = max_logit.max(self.dot_embedding(state, token));
                }
                let mut exp_sum = 0.0;
                for token in 0..vocab {
                    exp_sum += (self.dot_embedding(state, token) - max_logit).exp();
                }
                let target_logit = self.dot_embedding(state, target);
                loss_sum += max_logit + exp_sum.ln() - target_logit;
                for token in 0..vocab {
                    let probability =
                        (self.dot_embedding(state, token) - max_logit).exp() / exp_sum;
                    for channel in 0..d {
                        gradient[row * d + channel] +=
                            probability * self.embedding.get(token * d + channel);
                    }
                }
                for channel in 0..d {
                    gradient[row * d + channel] -= self.embedding.get(target * d + channel);
                }
                count += 1;
            }
        }
        debug_assert!(count > 0);
        let inverse_count = (count as f32).recip();
        for value in &mut gradient {
            *value *= inverse_count;
        }
        Ok((loss_sum, count, gradient))
    }

    fn cross_entropy_normalizer(
        &self,
        head: &[f32],
        ids: &[u32],
        batch: usize,
        time: usize,
        horizon: usize,
    ) -> Result<CrossEntropyNormalizer> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        if horizon == 0
            || horizon >= time
            || head.len() != rows * self.cfg.d_model
            || ids.len() != rows
        {
            bail!("MTP normalizer shape or horizon is invalid");
        }
        let mut max_logits = vec![0.0; rows];
        let mut exp_sums = vec![0.0; rows];
        for sequence in 0..batch {
            let base = sequence * time;
            for position in 0..time - horizon {
                let row = base + position;
                let state = &head[row * self.cfg.d_model..(row + 1) * self.cfg.d_model];
                let mut max_logit = f32::NEG_INFINITY;
                for token in 0..self.cfg.vocab_size {
                    max_logit = max_logit.max(self.dot_embedding(state, token));
                }
                let mut exp_sum = 0.0;
                for token in 0..self.cfg.vocab_size {
                    exp_sum += (self.dot_embedding(state, token) - max_logit).exp();
                }
                max_logits[row] = max_logit;
                exp_sums[row] = exp_sum;
            }
        }
        Ok(CrossEntropyNormalizer {
            max_logits,
            exp_sums,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_tied_embedding_gradient(
        &mut self,
        next_head: &[f32],
        second_head: &[f32],
        next_normalizer: &CrossEntropyNormalizer,
        second_normalizer: &CrossEntropyNormalizer,
        ids: &[u32],
        batch: usize,
        time: usize,
        input_gradient: &[f32],
        learning_rate: f32,
    ) -> Result<()> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| anyhow::anyhow!("MTP rows overflow"))?;
        let d = self.cfg.d_model;
        if next_head.len() != rows * d
            || second_head.len() != rows * d
            || input_gradient.len() != rows * d
            || ids.len() != rows
            || next_normalizer.max_logits.len() != rows
            || next_normalizer.exp_sums.len() != rows
            || second_normalizer.max_logits.len() != rows
            || second_normalizer.exp_sums.len() != rows
        {
            bail!("tied embedding gradient shape is invalid");
        }
        for token in 0..self.cfg.vocab_size {
            let mut gradient = vec![0.0; d];
            for (head, normalizer, horizon) in [
                (next_head, next_normalizer, 1_usize),
                (second_head, second_normalizer, 2_usize),
            ] {
                let count = batch * (time - horizon);
                for sequence in 0..batch {
                    let base = sequence * time;
                    for position in 0..time - horizon {
                        let row = base + position;
                        let state = &head[row * d..(row + 1) * d];
                        let probability =
                            (self.dot_embedding(state, token) - normalizer.max_logits[row]).exp()
                                / normalizer.exp_sums[row];
                        let target = ids[row + horizon] as usize;
                        let target_indicator = if token == target { 1.0 } else { 0.0 };
                        let coefficient = (probability - target_indicator) / count as f32;
                        for channel in 0..d {
                            gradient[channel] += coefficient * state[channel];
                        }
                    }
                }
            }
            for row in 0..rows {
                if ids[row] as usize == token {
                    for channel in 0..d {
                        gradient[channel] += input_gradient[row * d + channel];
                    }
                }
            }
            for channel in 0..d {
                let index = token * d + channel;
                self.embedding
                    .apply_clipped_sgd(index, gradient[channel], learning_rate);
            }
        }
        Ok(())
    }

    fn dot_embedding(&self, state: &[f32], token: usize) -> f32 {
        state
            .iter()
            .enumerate()
            .map(|(channel, value)| value * self.embedding.get(token * self.cfg.d_model + channel))
            .sum()
    }
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
        assert!(
            seeded_values(64, 0.25, 7)
                .iter()
                .all(|value| value.abs() <= 0.25)
        );
    }

    #[test]
    fn ste_update_is_clipped_and_rejects_invalid_gradients() {
        let mut layer = TernaryLinear::seeded(2, 1, 0.7, 7);
        let before = (0..layer.master.len())
            .map(|index| layer.master.get(index))
            .collect::<Vec<_>>();
        layer.apply_ste_gradient(&[100.0, -100.0], 0.5).unwrap();
        assert_eq!(layer.master.get(0), before[0] - 0.5);
        assert_eq!(layer.master.get(1), before[1] + 0.5);
        assert!(layer.apply_ste_gradient(&[f32::NAN, 0.0], 0.1).is_err());
    }

    #[test]
    fn ternary_backward_matches_input_finite_difference_and_ste_contract() {
        let layer = TernaryLinear::seeded(3, 2, 0.7, 19);
        let input = [0.25, -1.5, 2.0];
        let output_gradient = [0.75, -0.5];
        let backward = layer.backward_ste(&input, &output_gradient, 1).unwrap();
        let epsilon = 1e-3;
        for feature in 0..3 {
            let mut plus = input;
            let mut minus = input;
            plus[feature] += epsilon;
            minus[feature] -= epsilon;
            let plus_loss = layer
                .forward(&plus, 1)
                .unwrap()
                .iter()
                .zip(output_gradient)
                .map(|(value, gradient)| value * gradient)
                .sum::<f32>();
            let minus_loss = layer
                .forward(&minus, 1)
                .unwrap()
                .iter()
                .zip(output_gradient)
                .map(|(value, gradient)| value * gradient)
                .sum::<f32>();
            let numerical = (plus_loss - minus_loss) / (2.0 * epsilon);
            assert!((backward.input_gradient[feature] - numerical).abs() < 1e-3);
        }
        for output in 0..2 {
            for feature in 0..3 {
                let weight = output * 3 + feature;
                assert_eq!(
                    backward.latent_weight_gradient[weight],
                    output_gradient[output] * layer.row_scales[output] * input[feature]
                );
            }
        }
    }

    #[test]
    fn streamed_cross_entropy_backward_matches_finite_difference() {
        let cfg = TrainConfig {
            d_model: 2,
            n_layers: 1,
            vocab_size: 320,
            context_len: 3,
            batch_size: 1,
            hyena_kernel_len: 3,
            hyena_chunk_len: 3,
            ..Default::default()
        };
        let model = UllisHyena::new(cfg).unwrap();
        let ids = [7, 19, 31];
        let head = [0.25, -0.5, -0.75, 1.0, 2.0, -1.5];
        let (_, count, backward) = model
            .streamed_cross_entropy_horizon_backward(&head, &ids, 1, 3, 1)
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(&backward[4..], &[0.0, 0.0]);
        let loss = |head: &[f32]| {
            let (sum, count) = model.cross_entropy_horizon(head, &ids, 1, 3, 1);
            sum / count as f32
        };
        let epsilon = 1e-3;
        for index in 0..head.len() {
            let mut plus = head;
            let mut minus = head;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical = (loss(&plus) - loss(&minus)) / (2.0 * epsilon);
            assert!((backward[index] - numerical).abs() < 2e-3);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_streamed_cross_entropy_matches_cpu_reference_when_available() {
        let Ok(runtime) = crate::metal::MetalRuntime::new() else {
            return;
        };
        let cfg = TrainConfig {
            d_model: 2,
            n_layers: 1,
            vocab_size: 17,
            context_len: 3,
            batch_size: 1,
            hyena_kernel_len: 3,
            hyena_chunk_len: 3,
            ..Default::default()
        };
        let model = UllisHyena::new(cfg).unwrap();
        let ids = [7, 9, 11];
        let head = [0.25, -0.5, -0.75, 1.0, 2.0, -1.5];
        let embedding = runtime
            .upload_resident_fp16_parameters(&model.embedding)
            .unwrap();
        let actual = runtime
            .streamed_cross_entropy_fp16_resident(&head, &embedding, &ids, 1, 3, 2, 17, 1)
            .unwrap();
        let head_slot = runtime.upload_resident_activations(&head, 3, 2).unwrap();
        let resident = runtime
            .streamed_cross_entropy_fp16_from_activation(
                head_slot, &embedding, &ids, 1, 3, 2, 17, 1,
            )
            .unwrap();
        let resident_gradient = runtime
            .download_resident_gradient(resident.gradient_slot, 3, 2)
            .unwrap();
        let (expected_loss, expected_count, expected_gradient) = model
            .streamed_cross_entropy_horizon_backward(&head, &ids, 1, 3, 1)
            .unwrap();
        assert_eq!(actual.token_count, expected_count);
        assert_eq!(resident.token_count, expected_count);
        assert!((actual.loss_sum - expected_loss).abs() < 1e-5);
        assert!((resident.loss_sum - expected_loss).abs() < 1e-5);
        for (actual, expected) in actual.head_gradient.iter().zip(&expected_gradient) {
            assert!((actual / expected_count as f32 - expected).abs() < 1e-5);
        }
        for (actual, expected) in resident_gradient.iter().zip(&expected_gradient) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn mtp_projection_backward_matches_hidden_finite_difference() {
        let cfg = TrainConfig {
            d_model: 2,
            n_layers: 1,
            vocab_size: 320,
            context_len: 3,
            batch_size: 1,
            hyena_kernel_len: 3,
            hyena_chunk_len: 3,
            ..Default::default()
        };
        let model = UllisHyena::new(cfg).unwrap();
        let ids = [7, 19, 31];
        let hidden = [0.25, -0.5, -0.75, 1.0, 2.0, -1.5];
        let one = model.mtp_one.forward(&hidden, 3).unwrap();
        let (_, _, next_head_gradient) = model
            .streamed_cross_entropy_horizon_backward(&one, &ids, 1, 3, 1)
            .unwrap();
        let two = model.mtp_two.forward(&hidden, 3).unwrap();
        let (_, _, second_head_gradient) = model
            .streamed_cross_entropy_horizon_backward(&two, &ids, 1, 3, 2)
            .unwrap();
        let backward = model
            .backward_mtp_heads(
                &hidden,
                1,
                3,
                &MtpHeadBackward {
                    loss: MtpLoss {
                        next_token: 0.0,
                        second_token: 0.0,
                        next_token_count: 2,
                        second_token_count: 1,
                    },
                    next_head_gradient,
                    second_head_gradient,
                },
            )
            .unwrap();
        let loss = |hidden: &[f32]| {
            let one = model.mtp_one.forward(hidden, 3).unwrap();
            let two = model.mtp_two.forward(hidden, 3).unwrap();
            let (one_sum, one_count) = model.cross_entropy_horizon(&one, &ids, 1, 3, 1);
            let (two_sum, two_count) = model.cross_entropy_horizon(&two, &ids, 1, 3, 2);
            one_sum / one_count as f32 + two_sum / two_count as f32
        };
        let epsilon = 1e-3;
        for index in 0..hidden.len() {
            let mut plus = hidden;
            let mut minus = hidden;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical = (loss(&plus) - loss(&minus)) / (2.0 * epsilon);
            assert!((backward.hidden_gradient[index] - numerical).abs() < 2e-3);
        }
    }

    #[test]
    fn stateless_train_step_updates_heads_and_hyena_block_without_optimizer_state() {
        let cfg = TrainConfig {
            d_model: 2,
            n_layers: 1,
            vocab_size: 320,
            context_len: 3,
            batch_size: 1,
            hyena_kernel_len: 3,
            hyena_chunk_len: 3,
            ..Default::default()
        };
        let mut model = UllisHyena::new(cfg).unwrap();
        let embedding_before = model.embedding.clone();
        let mtp_before = model.mtp_one.master.clone();
        let block_before = model.blocks[0].input.master.clone();
        let filter_before = model.blocks[0].filter.freq.clone();
        let loss = model
            .train_step_stateless_sgd(&[7, 19, 31], 1, 3, 0.01)
            .unwrap();
        assert_eq!(loss.next_token_count, 2);
        assert_eq!(loss.second_token_count, 1);
        assert_ne!(model.embedding, embedding_before);
        assert_ne!(model.mtp_one.master, mtp_before);
        assert_ne!(model.blocks[0].input.master, block_before);
        assert_ne!(model.blocks[0].filter.freq, filter_before);
    }

    #[test]
    fn hyena_gate_backward_matches_finite_differences() {
        let mixed = [0.5, -1.5];
        let pre_gate = [0.25_f32, -0.75];
        let gated = [9.0, 9.0, pre_gate[0].tanh(), pre_gate[1].tanh()];
        let output_gradient = [0.75, -0.5];
        let backward = hyena_gate_backward(&mixed, &gated, &output_gradient, 2).unwrap();
        let epsilon = 1e-3;
        let loss = |mixed: &[f32], pre_gate: &[f32]| {
            mixed
                .iter()
                .zip(pre_gate)
                .zip(output_gradient)
                .map(|((mixed, gate), gradient)| mixed * gate.tanh() * gradient)
                .sum::<f32>()
        };
        for index in 0..2 {
            let mut plus = mixed;
            let mut minus = mixed;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            assert!(
                (backward.mixed_gradient[index]
                    - (loss(&plus, &pre_gate) - loss(&minus, &pre_gate)) / (2.0 * epsilon))
                    .abs()
                    < 1e-3
            );
            let mut gate_plus = pre_gate;
            let mut gate_minus = pre_gate;
            gate_plus[index] += epsilon;
            gate_minus[index] -= epsilon;
            assert!(
                (backward.projection_gradient[2 + index]
                    - (loss(&mixed, &gate_plus) - loss(&mixed, &gate_minus)) / (2.0 * epsilon))
                    .abs()
                    < 1e-3
            );
            assert_eq!(backward.projection_gradient[index], 0.0);
        }
    }

    #[test]
    fn hyena_block_backward_matches_input_finite_difference() {
        let cfg = TrainConfig {
            d_model: 2,
            n_layers: 1,
            vocab_size: 320,
            context_len: 4,
            batch_size: 1,
            hyena_kernel_len: 3,
            hyena_chunk_len: 4,
            ..Default::default()
        };
        let block = HyenaBlock::seeded(&cfg, 17);
        let input = [0.5, -1.0, 1.5, 2.0, -0.5, 3.0, 4.0, -2.0];
        let output_gradient = [0.25, -0.5, 1.0, 0.75, -1.5, 0.5, 0.25, -0.75];
        let (_, cache) = block.forward_with_cache(&input, 1, 4).unwrap();
        let backward = block.backward(&cache, &output_gradient, 1, 4).unwrap();
        let loss = |input: &[f32]| {
            block
                .forward(input, 1, 4)
                .unwrap()
                .iter()
                .zip(output_gradient)
                .map(|(value, gradient)| value * gradient)
                .sum::<f32>()
        };
        let epsilon = 1e-3;
        for index in 0..input.len() {
            let mut plus = input;
            let mut minus = input;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical = (loss(&plus) - loss(&minus)) / (2.0 * epsilon);
            assert!((backward.input_gradient[index] - numerical).abs() < 2e-3);
        }
        assert_eq!(backward.input_projection.latent_weight_gradient.len(), 8);
        assert_eq!(backward.output_projection.latent_weight_gradient.len(), 4);
        assert_eq!(backward.filter.freq_gradient.len(), 2 * cfg.filter_order);
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
    fn fp16_ternary_layer_matches_quantized_cpu_contract_when_available() {
        let layer = TernaryLinear::seeded(5, 3, 0.7, 42);
        let input = [0.25, -1.0, 2.5, 0.0, 1.0, -0.75, 0.5, 1.5, -2.0, 0.25];
        let Ok(runtime) = crate::metal::MetalRuntime::new() else {
            return;
        };
        let actual = layer
            .forward_metal_fp16_reference(&runtime, &input, 2)
            .unwrap();
        let quantized_input = Fp16Storage::from_f32(input);
        let quantized_scales = Fp16Storage::from_f32(layer.row_scales.iter().copied());
        let weights = runtime
            .upload_fp16_ternary_weights(
                &layer.codes.positive,
                &layer.codes.negative,
                &quantized_scales,
                crate::metal::TernaryLinearShape::new(2, 5, 3).unwrap(),
            )
            .unwrap();
        let resident = runtime
            .ternary_linear_forward_fp16_resident(&quantized_input, &weights)
            .unwrap();
        let resident = (0..resident.len())
            .map(|index| resident.get(index))
            .collect::<Vec<_>>();
        runtime.reserve_fp16_activations(2, 5).unwrap();
        let slot = runtime
            .upload_resident_fp16_activations(&quantized_input, 2, 5)
            .unwrap();
        let slot = runtime
            .resident_ternary_linear_fp16(slot, &weights)
            .unwrap();
        let activation_resident = runtime
            .download_resident_fp16_activations(slot, 2, 3)
            .unwrap();
        let activation_resident = (0..activation_resident.len())
            .map(|index| activation_resident.get(index))
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; 6];
        for row in 0..2 {
            for output in 0..3 {
                let sum = (0..5)
                    .map(|channel| {
                        quantized_input.get(row * 5 + channel)
                            * layer.codes.get(output * 5 + channel)
                    })
                    .sum::<f32>();
                expected[row * 3 + output] =
                    crate::precision::Fp16::from_f32(sum * quantized_scales.get(output)).to_f32();
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(resident, expected);
        assert_eq!(activation_resident, expected);
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

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_metal_forward_cache_matches_normal_resident_path_when_available() {
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
        let Ok(runtime) = crate::metal::MetalRuntime::new() else {
            return;
        };
        let expected = model.hidden_metal_resident(&runtime, &ids, 1, 4).unwrap();
        let (actual, caches) = model
            .hidden_metal_resident_for_backward(&runtime, &ids, 1, 4)
            .unwrap();
        assert_eq!(caches.len(), 2);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_metal_stack_backward_matches_cpu_across_two_blocks_when_available() {
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
        let upstream = [
            0.7, -0.2, 0.4, -0.6, -0.1, 0.5, -0.3, 0.8, 0.6, -0.7, 0.2, 0.1, -0.5, 0.3, 0.9, -0.4,
        ];
        let mut activation = Vec::with_capacity(ids.len() * model.cfg.d_model);
        for &id in &ids {
            for channel in 0..model.cfg.d_model {
                activation.push(
                    model
                        .embedding
                        .get(id as usize * model.cfg.d_model + channel),
                );
            }
        }
        let mut cpu_caches = Vec::new();
        for block in &model.blocks {
            let (next, cache) = block.forward_with_cache(&activation, 1, 4).unwrap();
            activation = next;
            cpu_caches.push(cache);
        }
        let mut cpu_gradient = upstream.to_vec();
        let mut cpu_reverse = Vec::new();
        for (block, cache) in model.blocks.iter().zip(cpu_caches.iter()).rev() {
            let backward = block.backward(cache, &cpu_gradient, 1, 4).unwrap();
            cpu_gradient = backward.input_gradient.clone();
            cpu_reverse.push(backward);
        }
        cpu_reverse.reverse();
        let Ok(runtime) = crate::metal::MetalRuntime::new() else {
            return;
        };
        let actual = model
            .hidden_metal_backward_reference(&runtime, &ids, &upstream, 1, 4)
            .unwrap();
        assert_eq!(actual.blocks.len(), 2);
        for (actual, expected) in actual.input_gradient.iter().zip(&cpu_gradient) {
            assert!((actual - expected).abs() < 6e-4, "{actual} != {expected}");
        }
        for (actual, expected) in actual.blocks.iter().zip(&cpu_reverse) {
            for (actual, expected) in actual
                .input_projection_weight_gradient
                .iter()
                .zip(&expected.input_projection.latent_weight_gradient)
            {
                assert!((actual - expected).abs() < 6e-4, "{actual} != {expected}");
            }
            for (actual, expected) in actual
                .output_projection_weight_gradient
                .iter()
                .zip(&expected.output_projection.latent_weight_gradient)
            {
                assert!((actual - expected).abs() < 6e-4, "{actual} != {expected}");
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn cached_metal_block_backward_matches_cpu_when_available() {
        let cfg = TrainConfig {
            d_model: 4,
            n_layers: 1,
            vocab_size: 320,
            context_len: 4,
            batch_size: 1,
            ..Default::default()
        };
        let model = UllisHyena::new(cfg).unwrap();
        let block = &model.blocks[0];
        let input = vec![
            0.2, -0.4, 0.6, -0.8, 0.1, 0.3, -0.5, 0.7, -0.9, 0.2, 0.4, -0.6, 0.8, -0.1, 0.5, -0.3,
        ];
        let upstream = vec![
            0.7, -0.2, 0.4, -0.6, -0.1, 0.5, -0.3, 0.8, 0.6, -0.7, 0.2, 0.1, -0.5, 0.3, 0.9, -0.4,
        ];
        let (_, cpu_cache) = block.forward_with_cache(&input, 1, 4).unwrap();
        let expected = block.backward(&cpu_cache, &upstream, 1, 4).unwrap();
        let Ok(runtime) = crate::metal::MetalRuntime::new() else {
            return;
        };
        let slot = runtime.upload_resident_activations(&input, 4, 4).unwrap();
        let (_, cache) = block
            .forward_metal_resident_with_cache(&runtime, slot, 1, 4)
            .unwrap();
        let plan = block.chunk_plan.for_sequence(4).unwrap();
        let filter = block.filter.generate(4, plan.kernel_len).unwrap();
        let (input_positive, input_negative, input_scales) = block.input.metal_parts();
        let (output_positive, output_negative, output_scales) = block.output.metal_parts();
        let actual = runtime
            .hyena_block_backward_cached_reference(
                &cache,
                &upstream,
                input_positive,
                input_negative,
                input_scales,
                output_positive,
                output_negative,
                output_scales,
                &filter,
                1,
                4,
                block.chunk_plan,
            )
            .unwrap();
        for (actual, expected) in actual.input_gradient.iter().zip(&expected.input_gradient) {
            assert!((actual - expected).abs() < 3e-4, "{actual} != {expected}");
        }
        for (actual, expected) in actual
            .input_projection_weight_gradient
            .iter()
            .zip(&expected.input_projection.latent_weight_gradient)
        {
            assert!((actual - expected).abs() < 3e-4, "{actual} != {expected}");
        }
        for (actual, expected) in actual
            .output_projection_weight_gradient
            .iter()
            .zip(&expected.output_projection.latent_weight_gradient)
        {
            assert!((actual - expected).abs() < 3e-4, "{actual} != {expected}");
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn cached_metal_block_backward_updates_resident_projections_without_gradient_readback() {
        let cfg = TrainConfig {
            d_model: 4,
            n_layers: 1,
            vocab_size: 320,
            context_len: 4,
            batch_size: 1,
            ..Default::default()
        };
        let model = UllisHyena::new(cfg).unwrap();
        let block = &model.blocks[0];
        let input = vec![
            0.2, -0.4, 0.6, -0.8, 0.1, 0.3, -0.5, 0.7, -0.9, 0.2, 0.4, -0.6, 0.8, -0.1, 0.5, -0.3,
        ];
        let upstream = vec![
            0.7, -0.2, 0.4, -0.6, -0.1, 0.5, -0.3, 0.8, 0.6, -0.7, 0.2, 0.1, -0.5, 0.3, 0.9, -0.4,
        ];
        let (_, cpu_cache) = block.forward_with_cache(&input, 1, 4).unwrap();
        let expected = block.backward(&cpu_cache, &upstream, 1, 4).unwrap();
        let learning_rate = 0.03;
        let mut expected_input = block.input.clone();
        let mut expected_output = block.output.clone();
        expected_input
            .apply_ste_gradient(
                &expected.input_projection.latent_weight_gradient,
                learning_rate,
            )
            .unwrap();
        expected_output
            .apply_ste_gradient(
                &expected.output_projection.latent_weight_gradient,
                learning_rate,
            )
            .unwrap();
        let Ok(runtime) = crate::metal::MetalRuntime::new() else {
            return;
        };
        let slot = runtime.upload_resident_activations(&input, 4, 4).unwrap();
        let (_, cache) = block
            .forward_metal_resident_with_cache(&runtime, slot, 1, 4)
            .unwrap();
        let input_weights = runtime
            .upload_trainable_fp16_ternary_weights(
                &block.input.master,
                block.input.in_features,
                block.input.out_features,
                block.input.threshold_ratio,
            )
            .unwrap();
        let output_weights = runtime
            .upload_trainable_fp16_ternary_weights(
                &block.output.master,
                block.output.in_features,
                block.output.out_features,
                block.output.threshold_ratio,
            )
            .unwrap();
        let gradient_slot = runtime.upload_resident_gradient(&upstream, 4, 4).unwrap();
        let plan = block.chunk_plan.for_sequence(4).unwrap();
        let filter = block.filter.generate(4, plan.kernel_len).unwrap();
        let (input_positive, input_negative, input_scales) = block.input.metal_parts();
        let (output_positive, output_negative, output_scales) = block.output.metal_parts();
        let actual = runtime
            .hyena_block_backward_cached_and_update_resident(
                &cache,
                gradient_slot,
                gradient_slot.other(),
                input_positive,
                input_negative,
                input_scales,
                output_positive,
                output_negative,
                output_scales,
                &input_weights,
                &output_weights,
                &filter,
                1,
                4,
                block.chunk_plan,
                learning_rate,
                true,
            )
            .unwrap();
        for (actual, expected) in actual.input_gradient.iter().zip(&expected.input_gradient) {
            assert!((actual - expected).abs() < 3e-4, "{actual} != {expected}");
        }
        assert_eq!(actual.filter_gradient.len(), filter.len());
        let (input_master, input_positive, input_negative, input_scales) = runtime
            .download_trainable_fp16_ternary_weights(&input_weights)
            .unwrap();
        assert_eq!(input_master, expected_input.master);
        assert_eq!(input_positive, expected_input.codes.positive);
        assert_eq!(input_negative, expected_input.codes.negative);
        assert_eq!(input_scales, expected_input.row_scales);
        let (output_master, output_positive, output_negative, output_scales) = runtime
            .download_trainable_fp16_ternary_weights(&output_weights)
            .unwrap();
        assert_eq!(output_master, expected_output.master);
        assert_eq!(output_positive, expected_output.codes.positive);
        assert_eq!(output_negative, expected_output.codes.negative);
        assert_eq!(output_scales, expected_output.row_scales);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resident_training_state_reuses_gpu_projection_updates_across_forwards() {
        let cfg = TrainConfig {
            d_model: 4,
            n_layers: 1,
            vocab_size: 320,
            context_len: 4,
            batch_size: 1,
            ..Default::default()
        };
        let model = UllisHyena::new(cfg).unwrap();
        let Ok(runtime) = crate::metal::MetalRuntime::new() else {
            return;
        };
        let state = model.new_metal_resident_training_state(&runtime).unwrap();
        let ids = [1, 2, 3, 4];
        let (before, _) = model
            .hidden_metal_resident_for_training(&runtime, &state, &ids, 1, 4)
            .unwrap();
        let reference = model.hidden_metal_resident(&runtime, &ids, 1, 4).unwrap();
        for (actual, expected) in before.iter().zip(&reference) {
            assert!((actual - expected).abs() < 3e-4, "{actual} != {expected}");
        }
        let upstream = [
            0.7, -0.2, 0.4, -0.6, -0.1, 0.5, -0.3, 0.8, 0.6, -0.7, 0.2, 0.1, -0.5, 0.3, 0.9, -0.4,
        ];
        let step = model
            .hidden_metal_backward_update_resident(
                &runtime, &state, &ids, &upstream, 1, 4, 0.03, true,
            )
            .unwrap();
        assert_eq!(step.input_gradient.len(), upstream.len());
        assert_eq!(step.filter_gradients.len(), 1);
        let (after, _) = model
            .hidden_metal_resident_for_training(&runtime, &state, &ids, 1, 4)
            .unwrap();
        assert!(
            before
                .iter()
                .zip(&after)
                .any(|(before, after)| (before - after).abs() > 1e-6)
        );
    }

    #[test]
    fn checkpoint_round_trip_preserves_fp16_model_output() {
        let cfg = TrainConfig {
            d_model: 4,
            n_layers: 1,
            vocab_size: 320,
            context_len: 4,
            batch_size: 1,
            ..Default::default()
        };
        let mut model = UllisHyena::new(cfg).unwrap();
        model
            .train_step_stateless_sgd(&[1, 2, 3, 4], 1, 4, 0.01)
            .unwrap();
        let checkpoint = model.checkpoint();
        let restored = UllisHyena::from_checkpoint(checkpoint).unwrap();
        assert_eq!(
            model.hidden(&[1, 2, 3, 4], 1, 4).unwrap(),
            restored.hidden(&[1, 2, 3, 4], 1, 4).unwrap()
        );
    }

    #[test]
    fn stateless_sgd_overfits_a_repeated_fixed_batch() {
        let cfg = TrainConfig {
            d_model: 4,
            n_layers: 1,
            vocab_size: 320,
            context_len: 8,
            batch_size: 1,
            ..Default::default()
        };
        let mut model = UllisHyena::new(cfg).unwrap();
        let ids = [7; 8];
        let initial = model.streamed_mtp_loss(&ids, 1, 8).unwrap().mean();
        for _ in 0..256 {
            model.train_step_stateless_sgd(&ids, 1, 8, 0.01).unwrap();
        }
        let final_loss = model.streamed_mtp_loss(&ids, 1, 8).unwrap().mean();
        assert!(final_loss < initial * 0.95, "{initial} -> {final_loss}");
    }
}
