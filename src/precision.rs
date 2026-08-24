//! Compact IEEE-754 binary16 storage used by trainable Ullis parameters.
//!
//! Values are widened only at the arithmetic boundary. This keeps the model's
//! persistent latent weights out of FP32 while allowing the CPU reference path
//! to retain FP32 accumulation and deterministic numerical tests.

/// One IEEE-754 binary16 value stored without a dependency on a host-specific
/// floating-point type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp16(u16);

impl Fp16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
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

/// Growable, contiguous FP16 parameter storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp16Storage {
    values: Vec<u16>,
}

impl Fp16Storage {
    /// Applies clipped SGD directly to the FP16 persistent value.
    ///
    /// A usual round-to-nearest update can silently become a no-op when a
    /// normalized language-model gradient is smaller than one half ULP.  If a
    /// meaningful fraction of the next FP16 step is present, this method
    /// advances one representable value in the gradient direction.  It is a
    /// deterministic, zero-state alternative to an FP32 residual accumulator.
    pub fn apply_clipped_sgd(&mut self, index: usize, gradient: f32, learning_rate: f32) {
        const MIN_ULP_FRACTION: f32 = 1.0 / 32.0;
        let current = Fp16(self.values[index]);
        let update = learning_rate * gradient.clamp(-1.0, 1.0);
        let desired = current.to_f32() - update;
        let rounded = Fp16::from_f32(desired);
        if rounded.0 != current.0 || update == 0.0 {
            self.values[index] = rounded.0;
            return;
        }
        let neighbor = if update.is_sign_positive() {
            current.next_down()
        } else {
            current.next_up()
        };
        let ulp = (neighbor.to_f32() - current.to_f32()).abs();
        self.values[index] = if ulp.is_finite() && update.abs() >= ulp * MIN_ULP_FRACTION {
            neighbor.0
        } else {
            current.0
        };
    }
    pub fn from_f32(values: impl IntoIterator<Item = f32>) -> Self {
        Self {
            values: values
                .into_iter()
                .map(|value| Fp16::from_f32(value).0)
                .collect(),
        }
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            values: vec![Fp16::default().0; len],
        }
    }

    pub fn from_bits(values: Vec<u16>) -> Self {
        Self { values }
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
    }
}

impl Fp16 {
    fn next_up(self) -> Self {
        if self.0 == 0 || self.0 == 0x8000 {
            return Self(1);
        }
        if self.0 & 0x8000 == 0 {
            Self(self.0.saturating_add(1))
        } else {
            Self(self.0.saturating_sub(1))
        }
    }

    fn next_down(self) -> Self {
        if self.0 == 0 || self.0 == 0x8000 {
            return Self(0x8001);
        }
        if self.0 & 0x8000 == 0 {
            Self(self.0.saturating_sub(1))
        } else {
            Self(self.0.saturating_add(1))
        }
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
    fn clipped_sgd_preserves_a_meaningful_sub_ulp_update_without_state() {
        let mut storage = Fp16Storage::from_f32([0.25]);
        storage.apply_clipped_sgd(0, 0.001, 0.01);
        assert!(storage.get(0) < 0.25);
    }
}
