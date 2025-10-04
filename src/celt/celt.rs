#![allow(dead_code)]

//! Helpers ported from `celt/celt.c`.
//!
//! This module starts translating small pieces of the CELT top-level glue. The
//! helpers exposed here have no dependencies on the rest of the encoder or
//! decoder state so they can be exercised in isolation while larger control
//! flow is still being translated.

use crate::celt::types::OpusInt32;

/// Returns the downsampling factor that maps the 48 kHz reference rate to the
/// provided sampling rate.
///
/// Mirrors the behaviour of `resampling_factor()` from `celt/celt.c`, which is
/// used to derive the coarse pitch analysis stride. Unsupported sampling rates
/// fall back to zero just like the reference implementation when custom modes
/// are enabled.
#[must_use]
pub(crate) fn resampling_factor(rate: OpusInt32) -> u32 {
    match rate {
        48_000 => 1,
        24_000 => 2,
        16_000 => 3,
        12_000 => 4,
        8_000 => 6,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::resampling_factor;

    #[test]
    fn matches_reference_mapping() {
        assert_eq!(resampling_factor(48_000), 1);
        assert_eq!(resampling_factor(24_000), 2);
        assert_eq!(resampling_factor(16_000), 3);
        assert_eq!(resampling_factor(12_000), 4);
        assert_eq!(resampling_factor(8_000), 6);
    }

    #[test]
    fn returns_zero_for_unsupported_rates() {
        assert_eq!(resampling_factor(44_100), 0);
        assert_eq!(resampling_factor(96_000), 0);
    }
}
