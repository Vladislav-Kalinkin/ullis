//! Open-ended decode: nucleus sampling and OpenAI-style logit penalties.
//!
//! Greedy argmax is the Holtzman degeneration case (loops, bland n-grams).
//! Commercial APIs keep temperature/top-p plus additive presence/frequency
//! penalties on the generated prefix. Unlikelihood belongs in the trainer
//! once CE is below the batch unigram; it is not a decode substitute.

use anyhow::{Result, bail};

/// OpenAI Completions penalty range.
pub const PENALTY_MIN: f32 = -2.0;
pub const PENALTY_MAX: f32 = 2.0;

/// Defaults match the public Completions knobs, not greedy.
/// `frequency_penalty` is non-zero because a 256-d 1-bit Heron at CE ≈ unigram
/// otherwise copies a single corpus unigram under argmax.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub min_new_tokens: usize,
    pub seed: u64,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.9,
            presence_penalty: 0.0,
            frequency_penalty: 0.5,
            min_new_tokens: 1,
            seed: 7,
        }
    }
}

impl DecodeConfig {
    pub fn validate(self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            bail!("temperature must be finite and >= 0 (0 is greedy)");
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            bail!("top-p must be in (0, 1]");
        }
        for (name, value) in [
            ("presence-penalty", self.presence_penalty),
            ("frequency-penalty", self.frequency_penalty),
        ] {
            if !value.is_finite() || !(PENALTY_MIN..=PENALTY_MAX).contains(&value) {
                bail!("{name} must be finite and in [{PENALTY_MIN}, {PENALTY_MAX}]");
            }
        }
        Ok(())
    }

    pub fn is_greedy(self) -> bool {
        self.temperature == 0.0
    }
}

/// OpenAI Completions: `μ[j] -= c[j]·α_freq + 1[c[j]>0]·α_pres`.
///
/// `counts` are **generated** ids only. Penalizing the prompt would suppress
/// words the user just typed.
pub fn apply_openai_penalties(
    logits: &mut [f32],
    counts: &[u32],
    presence_penalty: f32,
    frequency_penalty: f32,
) {
    let n = logits.len().min(counts.len());
    for j in 0..n {
        let count = counts[j];
        if count == 0 {
            continue;
        }
        logits[j] -= frequency_penalty * count as f32;
        logits[j] -= presence_penalty;
    }
}

pub fn bump_count(counts: &mut [u32], token: u32) {
    let index = token as usize;
    if let Some(slot) = counts.get_mut(index) {
        *slot = slot.saturating_add(1);
    }
}

/// SplitMix64; same mixer as model init, so a decode seed is reproducible.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit_interval(state: &mut u64) -> f32 {
    let word = splitmix64(state);
    (word >> 11) as f32 / ((1_u64 << 53) as f32)
}

fn masked(id: u32, pad_id: u32, bos_id: u32, eos_id: u32, suppress_eos: bool) -> bool {
    id == pad_id || id == bos_id || (suppress_eos && id == eos_id)
}

/// Next id after penalties. Temperature 0 is argmax on the penalized logits.
pub fn select_token(
    logits: &[f32],
    cfg: DecodeConfig,
    rng: &mut u64,
    pad_id: u32,
    bos_id: u32,
    eos_id: u32,
    suppress_eos: bool,
) -> u32 {
    if cfg.is_greedy() {
        return greedy_token(logits, pad_id, bos_id, eos_id, suppress_eos);
    }
    nucleus_token(
        logits,
        cfg.temperature,
        cfg.top_p,
        rng,
        pad_id,
        bos_id,
        eos_id,
        suppress_eos,
    )
}

fn greedy_token(logits: &[f32], pad_id: u32, bos_id: u32, eos_id: u32, suppress_eos: bool) -> u32 {
    logits
        .iter()
        .enumerate()
        .filter(|(id, logit)| {
            logit.is_finite() && !masked(*id as u32, pad_id, bos_id, eos_id, suppress_eos)
        })
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(id, _)| id as u32)
        .expect("non-empty vocabulary")
}

fn nucleus_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    rng: &mut u64,
    pad_id: u32,
    bos_id: u32,
    eos_id: u32,
    suppress_eos: bool,
) -> u32 {
    let temp = temperature.max(1e-5);
    let mut max = f32::NEG_INFINITY;
    for (id, logit) in logits.iter().enumerate() {
        if masked(id as u32, pad_id, bos_id, eos_id, suppress_eos) || !logit.is_finite() {
            continue;
        }
        if *logit > max {
            max = *logit;
        }
    }
    if !max.is_finite() {
        return greedy_token(logits, pad_id, bos_id, eos_id, false);
    }
    let mut mass = Vec::new();
    let mut z = 0.0_f32;
    for (id, logit) in logits.iter().enumerate() {
        let id = id as u32;
        if masked(id, pad_id, bos_id, eos_id, suppress_eos) || !logit.is_finite() {
            continue;
        }
        let weight = ((logit - max) / temp).exp();
        z += weight;
        mass.push((id, weight));
    }
    if mass.is_empty() || z <= 0.0 || !z.is_finite() {
        return greedy_token(logits, pad_id, bos_id, eos_id, false);
    }
    for item in &mut mass {
        item.1 /= z;
    }
    mass.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut cumulative = 0.0;
    let mut last = 0;
    for (index, (_, p)) in mass.iter().enumerate() {
        cumulative += *p;
        last = index;
        if cumulative >= top_p {
            break;
        }
    }
    let nucleus = &mass[..=last];
    let z_n: f32 = nucleus.iter().map(|(_, p)| *p).sum();
    if z_n <= 0.0 {
        return nucleus[0].0;
    }
    let mut draw = unit_interval(rng) * z_n;
    for (id, p) in nucleus {
        if draw <= *p {
            return *id;
        }
        draw -= *p;
    }
    nucleus[nucleus.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_frequency_penalty_is_count_times_alpha() {
        let mut logits = [1.0_f32, 1.0, 1.0];
        let counts = [0_u32, 3, 1];
        apply_openai_penalties(&mut logits, &counts, 0.0, 0.5);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - (1.0 - 1.5)).abs() < 1e-6);
        assert!((logits[2] - (1.0 - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn openai_presence_penalty_is_once_per_seen_token() {
        let mut logits = [2.0_f32, 2.0, 2.0];
        let counts = [0_u32, 9, 1];
        apply_openai_penalties(&mut logits, &counts, 0.4, 0.0);
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 1.6).abs() < 1e-6);
        assert!((logits[2] - 1.6).abs() < 1e-6);
    }

    #[test]
    fn frequency_penalty_breaks_greedy_unigram_loop() {
        let mut logits = [0.0_f32, 5.0, 4.5];
        let mut counts = [0_u32; 3];
        let mut last = 1_u32;
        for _ in 0..8 {
            apply_openai_penalties(&mut logits, &counts, 0.0, 0.6);
            last = greedy_token(&logits, 99, 99, 99, false);
            bump_count(&mut counts, last);
            logits = [0.0, 5.0, 4.5];
        }
        assert_eq!(
            last, 2,
            "after enough frequency penalty the runner-up must win, counts={counts:?}"
        );
    }

    #[test]
    fn temperature_zero_is_argmax() {
        let cfg = DecodeConfig {
            temperature: 0.0,
            ..DecodeConfig::default()
        };
        let mut rng = 1_u64;
        let logits = [0.1_f32, 3.0, 2.9];
        assert_eq!(select_token(&logits, cfg, &mut rng, 9, 9, 9, false), 1);
        assert_eq!(select_token(&logits, cfg, &mut rng, 9, 9, 9, false), 1);
    }

    #[test]
    fn min_new_tokens_suppresses_eos() {
        let logits = [0.0_f32, 0.0, 9.0, 1.0];
        let eos = 2;
        assert_eq!(greedy_token(&logits, 0, 1, eos, false), eos);
        assert_eq!(greedy_token(&logits, 0, 1, eos, true), 3);
    }

    #[test]
    fn nucleus_p_one_stays_inside_the_support() {
        let logits = [0.0_f32, 8.0, -20.0, 1.0];
        let mut rng = 11_u64;
        for _ in 0..32 {
            let id = nucleus_token(&logits, 0.7, 1.0, &mut rng, 9, 9, 9, false);
            assert!(id == 1 || id == 3 || id == 0, "sampled {id}");
        }
    }

    #[test]
    fn decode_config_rejects_out_of_range_penalties() {
        let bad_freq = DecodeConfig {
            frequency_penalty: 3.0,
            ..DecodeConfig::default()
        };
        assert!(bad_freq.validate().is_err());
        let bad_p = DecodeConfig {
            top_p: 0.0,
            ..DecodeConfig::default()
        };
        assert!(bad_p.validate().is_err());
    }
}
