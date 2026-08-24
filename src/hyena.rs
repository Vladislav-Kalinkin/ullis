//! Causal Hyena sequence mixing.
//!
//! This module is deliberately backend-neutral.  Its radix-2 FFT is the
//! reference implementation for the Metal kernel; keeping one definition of
//! causal convolution prevents the usual CPU/GPU padding drift.

use anyhow::{Result, bail};

/// Validated zero-padded radix-2 FFT geometry for causal convolution.
///
/// Keeping this plan backend-neutral gives CPU and Metal exactly the same
/// padding convention and lets the memory admission layer reason about the
/// two complex work buffers before allocating them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HyenaFftPlan {
    pub time: usize,
    pub fft_len: usize,
    pub stages: u32,
}

impl HyenaFftPlan {
    pub fn new(time: usize) -> Result<Self> {
        if time == 0 {
            bail!("Hyena FFT time must be non-zero");
        }
        let convolution_len = time
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| anyhow::anyhow!("causal_long_conv FFT length overflow"))?;
        let fft_len = convolution_len
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("causal_long_conv FFT length overflow"))?;
        Ok(Self {
            time,
            fft_len,
            stages: fft_len.ilog2(),
        })
    }
}

/// Bounded-workspace overlap-save geometry for a causal Hyena mixer.
///
/// The receptive field is intentionally explicit: an exact convolution with a
/// `T`-tap filter needs `O(T)` spectral workspace, regardless of how input is
/// chunked.  This plan instead makes the model use a `kernel_len`-tap causal
/// filter and evaluates it in `chunk_len`-token blocks.  It is exact for that
/// bounded filter while its reusable FFT workspace is independent of sequence
/// length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HyenaChunkPlan {
    pub chunk_len: usize,
    pub kernel_len: usize,
    pub fft_len: usize,
    pub stages: u32,
}

impl HyenaChunkPlan {
    pub fn new(chunk_len: usize, kernel_len: usize) -> Result<Self> {
        if chunk_len == 0 || kernel_len == 0 || kernel_len > chunk_len {
            bail!("Hyena chunk plan requires 0 < kernel_len <= chunk_len");
        }
        let convolution_len = chunk_len
            .checked_add(kernel_len)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| anyhow::anyhow!("Hyena chunk FFT length overflow"))?;
        let fft_len = convolution_len
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("Hyena chunk FFT length overflow"))?;
        Ok(Self {
            chunk_len,
            kernel_len,
            fft_len,
            stages: fft_len.ilog2(),
        })
    }

    /// Adapts a configured plan to a shorter sequence without changing its
    /// causal semantics or allocating a needlessly large scratch buffer.
    pub fn for_sequence(self, time: usize) -> Result<Self> {
        if time == 0 {
            bail!("Hyena chunk sequence length must be non-zero");
        }
        Self::new(self.chunk_len.min(time), self.kernel_len.min(time))
    }
}

/// A compact positional filter generator.  The generated filter is real and
/// deterministic, so it is safe to cache per `(length, channel)` at inference.
#[derive(Clone, Debug)]
pub struct ImplicitFilter {
    channels: usize,
    pub freq: Vec<f32>,
    pub phase: Vec<f32>,
    pub decay: Vec<f32>,
}

impl ImplicitFilter {
    pub fn new(channels: usize, order: usize, seed: u64) -> Self {
        let width = order.max(1);
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) as f32
        };
        let n = channels.saturating_mul(width);
        Self {
            channels,
            freq: (0..n).map(|_| 0.01 + 1.5 * sample()).collect(),
            phase: (0..n).map(|_| 6.283_185_5 * sample()).collect(),
            decay: (0..n).map(|_| 0.01 + 0.2 * sample()).collect(),
        }
    }

    pub fn generate(&self, channels: usize, len: usize) -> Result<Vec<f32>> {
        self.validate_channels(channels)?;
        let mut out = vec![0.0; channels * len];
        for c in 0..channels {
            self.generate_channel(c, &mut out[c * len..(c + 1) * len])?;
        }
        Ok(out)
    }

    /// Generates one channel directly into caller-owned workspace.
    pub fn generate_channel(&self, channel: usize, out: &mut [f32]) -> Result<()> {
        self.generate_channel_prefix(channel, out, out.len())
    }

    /// Generates a filter prefix while retaining positions relative to the
    /// full sequence.  This is what makes a bounded receptive field exactly
    /// equal to truncating the ordinary implicit filter at `out.len()`.
    pub fn generate_channel_prefix(
        &self,
        channel: usize,
        out: &mut [f32],
        sequence_len: usize,
    ) -> Result<()> {
        let channels = self.channels()?;
        if channel >= channels || out.is_empty() || sequence_len == 0 {
            bail!("invalid implicit-filter channel or output shape");
        }
        let order = self.freq.len() / channels;
        for (time, value) in out.iter_mut().enumerate() {
            let pos = time as f32 / sequence_len as f32;
            let mut sum = 0.0;
            for k in 0..order {
                let index = channel * order + k;
                sum += (-self.decay[index] * time as f32).exp()
                    * (self.freq[index] * pos + self.phase[index]).cos();
            }
            // Causal filters are normalized so sequence length does not
            // silently increase activation scale.
            *value = sum / order as f32;
        }
        Ok(())
    }

    fn channels(&self) -> Result<usize> {
        if self.channels == 0
            || self.freq.is_empty()
            || self.freq.len() != self.phase.len()
            || self.freq.len() != self.decay.len()
            || !self.freq.len().is_multiple_of(self.channels)
        {
            bail!("invalid implicit-filter parameters");
        }
        Ok(self.channels)
    }

    fn validate_channels(&self, channels: usize) -> Result<()> {
        if channels == 0 || self.channels()? != channels {
            bail!("invalid implicit-filter shape");
        }
        Ok(())
    }

    /// Validated compact parameters for a backend that generates the filter
    /// directly into its own workspace.
    pub(crate) fn parameter_slices(
        &self,
        channels: usize,
    ) -> Result<(&[f32], &[f32], &[f32], usize)> {
        self.validate_channels(channels)?;
        Ok((
            &self.freq,
            &self.phase,
            &self.decay,
            self.freq.len() / channels,
        ))
    }
}

/// Exact overlap-save causal convolution for an explicit bounded filter.
///
/// `filter` is `[channels, plan.kernel_len]`, while `x` remains
/// `[batch, time, channels]`.  The returned values are identical to direct
/// causal convolution with that finite filter, but scratch space is
/// `O(plan.fft_len)` rather than `O(time)`.
pub fn causal_chunked_conv(
    x: &[f32],
    filter: &[f32],
    batch: usize,
    time: usize,
    channels: usize,
    plan: HyenaChunkPlan,
) -> Result<Vec<f32>> {
    let plan = plan.for_sequence(time)?;
    let filter_values = channels
        .checked_mul(plan.kernel_len)
        .ok_or_else(|| anyhow::anyhow!("causal_chunked_conv filter shape overflow"))?;
    if filter.len() != filter_values {
        bail!("causal_chunked_conv filter shape mismatch");
    }
    causal_chunked_conv_with(
        x,
        batch,
        time,
        channels,
        channels,
        0,
        plan,
        |channel, kernel| {
            kernel.copy_from_slice(
                &filter[channel * plan.kernel_len..(channel + 1) * plan.kernel_len],
            );
            Ok(())
        },
    )
}

/// Bounded-receptive-field implicit convolution over a channel range inside a
/// wider row layout.  It never materialises `[channels, time]` filters or a
/// full-context FFT workspace.
pub fn causal_chunked_conv_implicit_strided(
    x: &[f32],
    filter: &ImplicitFilter,
    batch: usize,
    time: usize,
    channels: usize,
    row_width: usize,
    channel_offset: usize,
    plan: HyenaChunkPlan,
) -> Result<Vec<f32>> {
    filter.validate_channels(channels)?;
    let plan = plan.for_sequence(time)?;
    causal_chunked_conv_with(
        x,
        batch,
        time,
        channels,
        row_width,
        channel_offset,
        plan,
        |channel, kernel| filter.generate_channel_prefix(channel, kernel, time),
    )
}

fn causal_chunked_conv_with(
    x: &[f32],
    batch: usize,
    time: usize,
    channels: usize,
    input_width: usize,
    input_offset: usize,
    plan: HyenaChunkPlan,
    mut generate_kernel: impl FnMut(usize, &mut [f32]) -> Result<()>,
) -> Result<Vec<f32>> {
    let values = batch
        .checked_mul(time)
        .and_then(|rows| rows.checked_mul(channels))
        .ok_or_else(|| anyhow::anyhow!("causal_chunked_conv shape overflow"))?;
    let input_values = batch
        .checked_mul(time)
        .and_then(|rows| rows.checked_mul(input_width))
        .ok_or_else(|| anyhow::anyhow!("causal_chunked_conv input shape overflow"))?;
    if batch == 0
        || time == 0
        || channels == 0
        || input_width < channels
        || input_offset > input_width - channels
        || x.len() != input_values
    {
        bail!("causal_chunked_conv shape mismatch");
    }
    let mut out = vec![0.0; values];
    let mut kernel_values = vec![0.0; plan.kernel_len];
    let mut signal = vec![(0.0, 0.0); plan.fft_len];
    let mut kernel = vec![(0.0, 0.0); plan.fft_len];
    for channel in 0..channels {
        generate_kernel(channel, &mut kernel_values)?;
        kernel.fill((0.0, 0.0));
        for (slot, &value) in kernel.iter_mut().zip(&kernel_values) {
            slot.0 = value;
        }
        fft(&mut kernel, false);
        for sequence in 0..batch {
            for start in (0..time).step_by(plan.chunk_len) {
                let count = (time - start).min(plan.chunk_len);
                signal.fill((0.0, 0.0));
                let history = plan.kernel_len - 1;
                for slot in 0..history + count {
                    let source_time = start as isize + slot as isize - history as isize;
                    if source_time >= 0 {
                        signal[slot].0 = x[(sequence * time + source_time as usize) * input_width
                            + input_offset
                            + channel];
                    }
                }
                fft(&mut signal, false);
                for (value, kernel_value) in signal.iter_mut().zip(&kernel) {
                    *value = complex_mul(*value, *kernel_value);
                }
                fft(&mut signal, true);
                for offset in 0..count {
                    out[(sequence * time + start + offset) * channels + channel] =
                        signal[history + offset].0;
                }
            }
        }
    }
    Ok(out)
}

fn complex_mul(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// In-place, unnormalised forward / normalised inverse radix-2 FFT.
fn fft(values: &mut [(f32, f32)], inverse: bool) {
    let n = values.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut width = 2;
    while width <= n {
        let theta = sign * 2.0 * std::f32::consts::PI / width as f32;
        let root = (theta.cos(), theta.sin());
        for base in (0..n).step_by(width) {
            let mut w = (1.0, 0.0);
            for offset in 0..width / 2 {
                let even = values[base + offset];
                let odd = complex_mul(values[base + offset + width / 2], w);
                values[base + offset] = (even.0 + odd.0, even.1 + odd.1);
                values[base + offset + width / 2] = (even.0 - odd.0, even.1 - odd.1);
                w = complex_mul(w, root);
            }
        }
        width *= 2;
    }
    if inverse {
        for value in values {
            value.0 /= n as f32;
            value.1 /= n as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_is_causal_and_matches_direct_form() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let h = [0.5, 0.25, -0.5, 0.0];
        let y = causal_chunked_conv(&x, &h, 1, 4, 1, HyenaChunkPlan::new(4, 4).unwrap()).unwrap();
        let expected = [0.5, 1.25, 1.5, 1.75];
        for (actual, want) in y.iter().zip(expected) {
            assert!((actual - want).abs() < 1e-5, "{actual} != {want}");
        }
    }

    #[test]
    fn implicit_path_matches_materialized_filter() {
        let filter = ImplicitFilter::new(2, 3, 7);
        let x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let materialized = filter.generate(2, 4).unwrap();
        let plan = HyenaChunkPlan::new(4, 4).unwrap();
        let expected = causal_chunked_conv(&x, &materialized, 1, 4, 2, plan).unwrap();
        let actual =
            causal_chunked_conv_implicit_strided(&x, &filter, 1, 4, 2, 2, 0, plan).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn strided_implicit_path_matches_dense_layout() {
        let filter = ImplicitFilter::new(2, 3, 7);
        let dense = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let mut interleaved = Vec::new();
        for row in dense.chunks_exact(2) {
            interleaved.extend_from_slice(row);
            interleaved.extend_from_slice(&[9.0, 9.0]);
        }
        let plan = HyenaChunkPlan::new(4, 4).unwrap();
        let expected =
            causal_chunked_conv_implicit_strided(&dense, &filter, 1, 4, 2, 2, 0, plan).unwrap();
        let actual =
            causal_chunked_conv_implicit_strided(&interleaved, &filter, 1, 4, 2, 4, 0, plan)
                .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn filter_rejects_channel_counts_that_would_drop_parameters() {
        let filter = ImplicitFilter::new(4, 3, 7);
        assert!(
            causal_chunked_conv_implicit_strided(
                &[0.0; 4],
                &filter,
                1,
                2,
                2,
                2,
                0,
                HyenaChunkPlan::new(2, 2).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn chunked_overlap_save_matches_direct_bounded_convolution_across_chunks() {
        let x = [
            1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0,
        ];
        let filter = [0.5, -0.25, 0.125, 1.0, 0.0, -0.5];
        let plan = HyenaChunkPlan::new(4, 3).unwrap();
        let actual = causal_chunked_conv(&x, &filter, 1, 7, 2, plan).unwrap();
        let mut expected = vec![0.0; x.len()];
        for time in 0..7 {
            for channel in 0..2 {
                for tap in 0..3 {
                    if tap <= time {
                        expected[time * 2 + channel] +=
                            x[(time - tap) * 2 + channel] * filter[channel * 3 + tap];
                    }
                }
            }
        }
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn implicit_chunked_path_is_exact_for_the_full_sequence_filter_prefix() {
        let filter = ImplicitFilter::new(2, 3, 7);
        let plan = HyenaChunkPlan::new(4, 3).unwrap();
        let x = [
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
        ];
        let mut prefix = vec![0.0; 2 * 3];
        for channel in 0..2 {
            filter
                .generate_channel_prefix(channel, &mut prefix[channel * 3..(channel + 1) * 3], 7)
                .unwrap();
        }
        let expected = causal_chunked_conv(&x, &prefix, 1, 7, 2, plan).unwrap();
        let actual =
            causal_chunked_conv_implicit_strided(&x, &filter, 1, 7, 2, 2, 0, plan).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn chunk_plan_rejects_an_unbounded_or_inverted_window() {
        assert!(HyenaChunkPlan::new(0, 1).is_err());
        assert!(HyenaChunkPlan::new(2, 3).is_err());
        assert_eq!(HyenaChunkPlan::new(4, 3).unwrap().fft_len, 8);
    }

    #[test]
    fn fft_plan_uses_causal_zero_padding_geometry() {
        assert_eq!(HyenaFftPlan::new(1).unwrap().fft_len, 1);
        assert_eq!(
            HyenaFftPlan::new(4).unwrap(),
            HyenaFftPlan {
                time: 4,
                fft_len: 8,
                stages: 3
            }
        );
        assert_eq!(HyenaFftPlan::new(32_000).unwrap().fft_len, 65_536);
        assert!(HyenaFftPlan::new(0).is_err());
    }
}
