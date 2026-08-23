//! Causal Hyena sequence mixing.
//!
//! This module is deliberately backend-neutral.  Its radix-2 FFT is the
//! reference implementation for the Metal kernel; keeping one definition of
//! causal convolution prevents the usual CPU/GPU padding drift.

use anyhow::{bail, Result};

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
        let channels = self.channels()?;
        if channel >= channels || out.is_empty() {
            bail!("invalid implicit-filter channel or output shape");
        }
        let order = self.freq.len() / channels;
        let len = out.len() as f32;
        for (time, value) in out.iter_mut().enumerate() {
            let pos = time as f32 / len;
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
}

/// Per-channel causal long convolution using zero-padded radix-2 FFT.
/// `x` and `filter` are `[batch, time, channels]` and `[channels, time]`.
pub fn causal_long_conv(
    x: &[f32],
    filter: &[f32],
    batch: usize,
    time: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    let filter_len = channels
        .checked_mul(time)
        .ok_or_else(|| anyhow::anyhow!("causal_long_conv filter shape overflow"))?;
    if filter.len() != filter_len {
        bail!("causal_long_conv shape mismatch");
    }
    causal_long_conv_with(x, batch, time, channels, channels, 0, |channel, kernel| {
        kernel.copy_from_slice(&filter[channel * time..(channel + 1) * time]);
        Ok(())
    })
}

/// Causal long convolution that generates one implicit-filter channel at a
/// time. This avoids materialising a `[channels, time]` FP32 filter tensor.
pub fn causal_long_conv_implicit(
    x: &[f32],
    filter: &ImplicitFilter,
    batch: usize,
    time: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    filter.validate_channels(channels)?;
    causal_long_conv_with(x, batch, time, channels, channels, 0, |channel, kernel| {
        filter.generate_channel(channel, kernel)
    })
}

/// Implicit causal convolution over a channel range inside a wider row layout.
/// `row_width` and `channel_offset` let callers reuse projection storage.
pub fn causal_long_conv_implicit_strided(
    x: &[f32],
    filter: &ImplicitFilter,
    batch: usize,
    time: usize,
    channels: usize,
    row_width: usize,
    channel_offset: usize,
) -> Result<Vec<f32>> {
    filter.validate_channels(channels)?;
    causal_long_conv_with(
        x,
        batch,
        time,
        channels,
        row_width,
        channel_offset,
        |channel, kernel| filter.generate_channel(channel, kernel),
    )
}

fn causal_long_conv_with(
    x: &[f32],
    batch: usize,
    time: usize,
    channels: usize,
    input_width: usize,
    input_offset: usize,
    mut generate_kernel: impl FnMut(usize, &mut [f32]) -> Result<()>,
) -> Result<Vec<f32>> {
    let values = batch
        .checked_mul(time)
        .and_then(|rows| rows.checked_mul(channels))
        .ok_or_else(|| anyhow::anyhow!("causal_long_conv shape overflow"))?;
    let input_values = batch
        .checked_mul(time)
        .and_then(|rows| rows.checked_mul(input_width))
        .ok_or_else(|| anyhow::anyhow!("causal_long_conv input shape overflow"))?;
    if batch == 0
        || time == 0
        || channels == 0
        || input_width < channels
        || input_offset > input_width - channels
        || x.len() != input_values
    {
        bail!("causal_long_conv shape mismatch");
    }
    let fft_len = HyenaFftPlan::new(time)?.fft_len;
    let mut out = vec![0.0; values];
    let mut kernel_values = vec![0.0; time];
    let mut signal = vec![(0.0, 0.0); fft_len];
    let mut kernel = vec![(0.0, 0.0); fft_len];
    for channel in 0..channels {
        generate_kernel(channel, &mut kernel_values)?;
        kernel.fill((0.0, 0.0));
        for (slot, &value) in kernel.iter_mut().zip(&kernel_values) {
            slot.0 = value;
        }
        fft(&mut kernel, false);
        for sequence in 0..batch {
            signal.fill((0.0, 0.0));
            for time_index in 0..time {
                signal[time_index].0 =
                    x[(sequence * time + time_index) * input_width + input_offset + channel];
            }
            fft(&mut signal, false);
            for (value, kernel_value) in signal.iter_mut().zip(&kernel) {
                *value = complex_mul(*value, *kernel_value);
            }
            fft(&mut signal, true);
            for time_index in 0..time {
                out[(sequence * time + time_index) * channels + channel] = signal[time_index].0;
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
        let y = causal_long_conv(&x, &h, 1, 4, 1).unwrap();
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
        let expected = causal_long_conv(&x, &materialized, 1, 4, 2).unwrap();
        let actual = causal_long_conv_implicit(&x, &filter, 1, 4, 2).unwrap();
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
        let expected = causal_long_conv_implicit(&dense, &filter, 1, 4, 2).unwrap();
        let actual =
            causal_long_conv_implicit_strided(&interleaved, &filter, 1, 4, 2, 4, 0).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn filter_rejects_channel_counts_that_would_drop_parameters() {
        let filter = ImplicitFilter::new(4, 3, 7);
        assert!(causal_long_conv_implicit(&[0.0; 4], &filter, 1, 2, 2).is_err());
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
