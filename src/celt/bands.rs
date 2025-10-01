#![allow(dead_code)]

//! Helper routines from `celt/bands.c` that are self-contained enough to port
//! ahead of the rest of the band analysis logic.
//!
//! The goal is to translate building blocks that have little coupling with the
//! more complex pieces of the encoder so that future ports can focus on the
//! higher-level control flow.

use crate::celt::types::OpusVal16;

/// Applies a hysteresis decision to a scalar value.
///
/// Mirrors `hysteresis_decision()` from `celt/bands.c`. The helper walks the
/// provided threshold table and returns the first band whose threshold exceeds
/// the current value. Hysteresis offsets are used to avoid flapping between
/// adjacent bands when the value is close to the threshold shared by two
/// regions. The `prev` argument supplies the previously selected band.
#[must_use]
pub(crate) fn hysteresis_decision(
    value: OpusVal16,
    thresholds: &[OpusVal16],
    hysteresis: &[OpusVal16],
    prev: usize,
) -> usize {
    debug_assert_eq!(thresholds.len(), hysteresis.len());
    let count = thresholds.len();
    debug_assert!(prev <= count, "prev index must be within the table bounds");

    let mut index = 0;
    while index < count {
        if value < thresholds[index] {
            break;
        }
        index += 1;
    }

    if prev < count && index > prev
        && value < thresholds[prev] + hysteresis[prev] {
            index = prev;
        }

    if prev > 0 && index < prev
        && value > thresholds[prev - 1] - hysteresis[prev - 1] {
            index = prev;
        }

    index
}

/// Linear congruential pseudo-random number generator used by the band tools.
///
/// Mirrors `celt_lcg_rand()` from `celt/bands.c`. The generator matches the
/// parameters from Numerical Recipes and returns a new 32-bit seed value.
#[must_use]
#[inline]
pub(crate) fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

#[cfg(test)]
mod tests {
    use super::{celt_lcg_rand, hysteresis_decision};

    #[test]
    fn hysteresis_matches_reference_logic() {
        // Synthetic thresholds with simple hysteresis offsets.
        let thresholds = [0.2_f32, 0.4, 0.6, 0.8];
        let hysteresis = [0.05_f32; 4];

        fn reference(value: f32, thresholds: &[f32], hysteresis: &[f32], prev: usize) -> usize {
            let count = thresholds.len();
            let mut i = 0;
            while i < count {
                if value < thresholds[i] {
                    break;
                }
                i += 1;
            }

            if i > prev && prev < count && value < thresholds[prev] + hysteresis[prev] {
                i = prev;
            }
            if i < prev && prev > 0 && value > thresholds[prev - 1] - hysteresis[prev - 1] {
                i = prev;
            }
            i
        }

        let values = [0.0, 0.15, 0.25, 0.39, 0.41, 0.59, 0.61, 0.79, 0.81, 0.95];

        for prev in 0..=thresholds.len() {
            for &value in &values {
                let expected = reference(value, &thresholds, &hysteresis, prev);
                assert_eq!(
                    hysteresis_decision(value, &thresholds, &hysteresis, prev),
                    expected,
                    "value {value}, prev {prev}",
                );
            }
        }
    }

    #[test]
    fn lcg_rand_produces_expected_sequence() {
        let mut seed = 0xDEAD_BEEF_u32;
        let mut expected = [0_u32; 5];
        for slot in &mut expected {
            let next = ((1_664_525_u64 * u64::from(seed)) + 1_013_904_223_u64) & 0xFFFF_FFFF;
            *slot = next as u32;
            seed = next as u32;
        }

        seed = 0xDEAD_BEEF_u32;
        for &value in &expected {
            seed = celt_lcg_rand(seed);
            assert_eq!(seed, value);
        }
    }
}
