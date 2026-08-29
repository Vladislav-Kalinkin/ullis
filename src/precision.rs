//! Compact IEEE-754 binary16 storage used by trainable Ullis parameters.
//!
//! Values are widened only at the arithmetic boundary. This keeps the model's
//! persistent latent weights out of FP32 while allowing the CPU reference path
//! to retain FP32 accumulation and deterministic numerical tests.

use anyhow::{Result, bail};

/// One IEEE-754 binary16 value stored without a dependency on a host-specific
/// floating-point type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp16(u16);

impl Fp16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }

    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x007f_ffff;
        if exponent == 0xff {
            return Self(sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 });
        }
        let half_exponent = exponent - 127 + 15;
        if half_exponent >= 31 {
            return Self(sign | 0x7c00);
        }
        if half_exponent <= 0 {
            if half_exponent < -10 {
                return Self(sign);
            }
            let significand = mantissa | 0x0080_0000;
            let shift = (14 - half_exponent) as u32;
            let mut rounded = significand >> shift;
            let remainder = significand & ((1_u32 << shift) - 1);
            let halfway = 1_u32 << (shift - 1);
            if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
                rounded += 1;
            }
            return Self(sign | rounded as u16);
        }
        let mut rounded = sign | ((half_exponent as u16) << 10) | (mantissa >> 13) as u16;
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
            rounded += 1;
        }
        Self(rounded)
    }

    pub fn to_f32(self) -> f32 {
        let sign = (u32::from(self.0 & 0x8000)) << 16;
        let exponent = u32::from((self.0 >> 10) & 0x1f);
        let mantissa = u32::from(self.0 & 0x03ff);
        let bits = match exponent {
            0 if mantissa == 0 => sign,
            0 => {
                let mut significand = mantissa;
                let mut exponent = -14_i32;
                while significand & 0x0400 == 0 {
                    significand <<= 1;
                    exponent -= 1;
                }
                sign | (((exponent + 127) as u32) << 23) | ((significand & 0x03ff) << 13)
            }
            31 => sign | 0x7f80_0000 | (mantissa << 13),
            _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
        };
        f32::from_bits(bits)
    }
}

/// Growable, contiguous FP16 parameter storage with an FP32 error-diffusion carry.
///
/// Mean-reduced language-model gradients are routinely smaller than one FP16 ULP.
/// A zero-state ULP floor either drops those updates or inflates them to a full
/// ULP, so packed BinaryConnect latents and embeddings never actually integrate
/// the STE gradient. Safetensors snapshots store the carry; JSON v2 files still
/// reconstruct it as zeros.
#[derive(Clone, Debug)]
pub struct Fp16Storage {
    values: Vec<u16>,
    residual: Vec<f32>,
}

impl PartialEq for Fp16Storage {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl Eq for Fp16Storage {}

impl Fp16Storage {
    /// Applies clipped SGD through an FP32 residual so sub-ULP steps accumulate.
    ///
    /// `gradient` is clipped to `[-1, 1]` before the learning-rate scale, matching
    /// the previous optimizer contract. The carry is what replaces the biased ULP floor.
    /// Use this for continuous FP16 tensors (embeddings, LayerNorm, scales, CMix value).
    pub fn apply_clipped_sgd(&mut self, index: usize, gradient: f32, learning_rate: f32) {
        let current = Fp16(self.values[index]).to_f32();
        let update = learning_rate * gradient.clamp(-1.0, 1.0);
        let desired = current - update + self.residual[index];
        let rounded = Fp16::from_f32(desired);
        self.residual[index] = desired - rounded.to_f32();
        self.values[index] = rounded.0;
    }

    /// BinaryConnect proxy step (Courbariaux et al.): accumulate the STE
    /// gradient, clip the proxy to `[-1, 1]`, and let the caller binarize
    /// `sign(proxy)` for the forward pass.
    ///
    /// Sign-SGD on this tensor is *not* BinaryConnect. Softmax CE has one
    /// target class and `|V|-1` others; `sign(g)` treats a `1e-7` wrong-class
    /// row the same as the target, so packed head bits become noise and greedy
    /// decode collapses. Magnitude STE keeps the `|V|:1` ratio. Sub-ULP steps
    /// still integrate in the FP32 carry.
    pub fn apply_binaryconnect_sgd(&mut self, index: usize, gradient: f32, learning_rate: f32) {
        if !gradient.is_finite() || gradient == 0.0 {
            return;
        }
        let current = Fp16(self.values[index]).to_f32();
        let update = learning_rate * gradient.clamp(-1.0, 1.0);
        let desired = (current - update + self.residual[index]).clamp(-1.0, 1.0);
        let rounded = Fp16::from_f32(desired);
        self.residual[index] = desired - rounded.to_f32();
        self.values[index] = rounded.0;
    }

    pub fn from_f32(values: impl IntoIterator<Item = f32>) -> Self {
        let values: Vec<u16> = values
            .into_iter()
            .map(|value| Fp16::from_f32(value).0)
            .collect();
        let residual = vec![0.0; values.len()];
        Self { values, residual }
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            values: vec![Fp16::default().0; len],
            residual: vec![0.0; len],
        }
    }

    pub fn from_bits(values: Vec<u16>) -> Self {
        let residual = vec![0.0; values.len()];
        Self { values, residual }
    }

    /// Restore FP16 values and their FP32 error-diffusion carry.
    pub fn from_bits_and_residual(values: Vec<u16>, residual: Vec<f32>) -> Result<Self> {
        if residual.len() != values.len() {
            bail!("FP16 residual length mismatch");
        }
        Ok(Self { values, residual })
    }

    /// Replace the carry in place, keeping the current FP16 payload.
    pub fn install_residual(&mut self, residual: Vec<f32>) -> Result<()> {
        if residual.len() != self.residual.len() {
            bail!("FP16 residual length mismatch");
        }
        self.residual = residual;
        Ok(())
    }

    pub fn residual(&self) -> &[f32] {
        &self.residual
    }

    pub fn install(&mut self, values: Vec<u16>, residual: Vec<f32>) -> Result<()> {
        if values.len() != self.values.len() || residual.len() != self.residual.len() {
            bail!("FP16 storage install length mismatch");
        }
        self.values = values;
        self.residual = residual;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.values.len() * size_of::<Fp16>()
    }

    /// Native-endian IEEE-754 binary16 payload suitable for an MTLBuffer.
    pub fn as_bits(&self) -> &[u16] {
        &self.values
    }

    pub fn get(&self, index: usize) -> f32 {
        Fp16(self.values[index]).to_f32()
    }

    pub fn set(&mut self, index: usize, value: f32) {
        self.values[index] = Fp16::from_f32(value).0;
        self.residual[index] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_preserves_representable_values_and_handles_specials() {
        for value in [0.0, -0.0, 1.0, -2.0, 0.5, 65_504.0] {
            assert_eq!(Fp16::from_f32(value).to_f32(), value);
        }
        assert!(Fp16::from_f32(f32::NAN).to_f32().is_nan());
        assert!(Fp16::from_f32(f32::INFINITY).to_f32().is_infinite());
    }

    #[test]
    fn fp16_storage_uses_two_bytes_per_resident_parameter() {
        let mut storage = Fp16Storage::from_f32([0.25, -1.5, 2.0]);
        assert_eq!(storage.bytes(), 6);
        assert_eq!(storage.get(1), -1.5);
        storage.set(0, 0.75);
        assert_eq!(storage.get(0), 0.75);
    }

    #[test]
    fn clipped_sgd_applies_an_update_larger_than_one_ulp() {
        let mut storage = Fp16Storage::from_f32([0.25]);
        storage.apply_clipped_sgd(0, 1.0, 0.01);
        assert!(storage.get(0) < 0.25);
    }

    #[test]
    fn clipped_sgd_accumulates_sub_ulp_updates_in_the_fp32_residual() {
        let mut storage = Fp16Storage::from_f32([0.25]);
        let before = storage.get(0);
        storage.apply_clipped_sgd(0, 0.001, 0.01);
        assert_eq!(storage.get(0), before);
        assert!(storage.residual()[0] < 0.0);
        for _ in 0..64 {
            storage.apply_clipped_sgd(0, 0.001, 0.01);
        }
        assert!(
            storage.get(0) < before,
            "residual must eventually cross an FP16 ULP, got {}",
            storage.get(0)
        );
    }

    #[test]
    fn binaryconnect_sgd_follows_ste_magnitude_and_clips_the_proxy() {
        let mut storage = Fp16Storage::from_f32([0.5, 0.5]);
        storage.apply_binaryconnect_sgd(0, 1.0, 0.1);
        storage.apply_binaryconnect_sgd(1, 0.1, 0.1);
        assert!(
            (storage.get(0) - 0.4).abs() < 1e-3,
            "unit STE step must move by lr, got {}",
            storage.get(0)
        );
        assert!(
            (storage.get(1) - 0.49).abs() < 1e-3,
            "0.1 STE step must move by 0.1·lr, got {}",
            storage.get(1)
        );
        assert!(
            (0.5 - storage.get(0)).abs() > (0.5 - storage.get(1)).abs(),
            "larger |g| must produce a larger latent step (softmax ratio)"
        );
        storage.apply_binaryconnect_sgd(0, -1.0, 4.0);
        assert_eq!(storage.get(0), 1.0);
        storage.apply_binaryconnect_sgd(0, 1.0, 4.0);
        assert_eq!(storage.get(0), -1.0);
    }

    #[test]
    fn from_bits_and_residual_roundtrips() {
        let mut storage = Fp16Storage::from_f32([0.25]);
        storage.apply_clipped_sgd(0, 0.001, 0.01);
        let restored = Fp16Storage::from_bits_and_residual(
            storage.as_bits().to_vec(),
            storage.residual().to_vec(),
        )
        .unwrap();
        assert_eq!(restored.get(0), storage.get(0));
        assert_eq!(restored.residual(), storage.residual());
    }

    #[test]
    fn binaryconnect_sgd_accumulates_sub_ulp_ste_in_the_residual() {
        let mut storage = Fp16Storage::from_f32([0.25]);
        let before = storage.get(0);
        storage.apply_binaryconnect_sgd(0, 0.001, 0.01);
        assert_eq!(storage.get(0), before);
        assert!(storage.residual()[0] < 0.0);
        for _ in 0..64 {
            storage.apply_binaryconnect_sgd(0, 0.001, 0.01);
        }
        assert!(
            storage.get(0) < before,
            "BinaryConnect residual must eventually cross an FP16 ULP, got {}",
            storage.get(0)
        );
    }
}
