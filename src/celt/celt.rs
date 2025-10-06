#![allow(dead_code)]

//! Helpers ported from `celt/celt.c`.
//!
//! This module starts translating small pieces of the CELT top-level glue. The
//! helpers exposed here have no dependencies on the rest of the encoder or
//! decoder state so they can be exercised in isolation while larger control
//! flow is still being translated.

use crate::celt::types::{OpusCustomMode, OpusInt32};

/// TF change table mirroring `tf_select_table` from `celt/celt.c`.
///
/// Positive values indicate better frequency resolution (longer effective
/// windows) whereas negative values favour time resolution. The second index is
/// computed as `4 * is_transient + 2 * tf_select + per_band_flag`.
pub(crate) const TF_SELECT_TABLE: [[i8; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];

/// Returns the canonical error string associated with an Opus/CELT error code.
///
/// Mirrors `opus_strerror()` from `celt/celt.c` for the subset of error codes
/// used by the reference implementation. Unrecognised codes fall back to the
/// "unknown error" string just like the C helper.
#[must_use]
pub(crate) fn opus_strerror(error: i32) -> &'static str {
    match error {
        0 => "success",
        -1 => "invalid argument",
        -2 => "buffer too small",
        -3 => "internal error",
        -4 => "corrupted stream",
        -5 => "request not implemented",
        -6 => "invalid state",
        -7 => "memory allocation failed",
        _ => "unknown error",
    }
}

/// Compile-time version string matching the format returned by the reference
/// implementation's `opus_get_version_string()` helper.
pub(crate) const OPUS_VERSION_STRING: &str = concat!("libopus ", env!("CARGO_PKG_VERSION"));

/// Returns the textual version identifier for the library.
#[must_use]
pub(crate) fn opus_get_version_string() -> &'static str {
    OPUS_VERSION_STRING
}

/// Fills `cap` with the per-band dynamic allocation caps for the provided mode.
///
/// Mirrors the behaviour of `init_caps()` from `celt/celt.c`, scaling the
/// cached limits by the number of channels and the effective band size derived
/// from `LM`. The caller must provide a `cap` slice whose length matches the
/// number of energy bands in the mode.
pub(crate) fn init_caps(mode: &OpusCustomMode<'_>, cap: &mut [i32], lm: usize, channels: usize) {
    let nb_ebands = mode.num_ebands;
    assert_eq!(cap.len(), nb_ebands, "cap slice must cover every band");
    assert!(channels > 0, "channel count must be positive");
    assert!(
        mode.e_bands.len() >= nb_ebands + 1,
        "mode does not expose the terminating band edge"
    );

    let stride = 2 * lm + (channels - 1);
    let base_offset = nb_ebands * stride;
    let caps_table = &mode.cache.caps;
    assert!(
        base_offset + nb_ebands <= caps_table.len(),
        "pulse cache caps table is too small"
    );

    for (band_index, cap_value) in cap.iter_mut().enumerate() {
        let band_width = i32::from(mode.e_bands[band_index + 1] - mode.e_bands[band_index]);
        let n = band_width << lm;
        let cached_cap = i32::from(caps_table[base_offset + band_index]) + 64;
        let scaled = cached_cap * (channels as i32) * n;
        *cap_value = scaled >> 2;
    }
}

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
    use super::{
        OPUS_VERSION_STRING, TF_SELECT_TABLE, init_caps, opus_get_version_string, opus_strerror,
        resampling_factor,
    };
    use crate::celt::types::{MdctLookup, OpusCustomMode, PulseCacheData};
    use alloc::{format, vec};

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

    #[test]
    fn tf_select_table_matches_reference_layout() {
        let expected = [
            [0, -1, 0, -1, 0, -1, 0, -1],
            [0, -1, 0, -2, 1, 0, 1, -1],
            [0, -2, 0, -3, 2, 0, 1, -1],
            [0, -2, 0, -3, 3, 0, 1, -1],
        ];
        assert_eq!(TF_SELECT_TABLE, expected);
    }

    #[test]
    fn init_caps_scales_cached_limits() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![10, 20, 30, 40, 50, 60]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut caps = vec![0; mode.num_ebands];
        init_caps(&mode, &mut caps, 1, 1);

        assert_eq!(caps, [114, 186]);
    }

    #[test]
    fn opus_strerror_matches_reference_strings() {
        assert_eq!(opus_strerror(0), "success");
        assert_eq!(opus_strerror(-1), "invalid argument");
        assert_eq!(opus_strerror(-7), "memory allocation failed");
        assert_eq!(opus_strerror(1), "unknown error");
        assert_eq!(opus_strerror(-42), "unknown error");
    }

    #[test]
    fn opus_get_version_string_matches_constant() {
        assert_eq!(opus_get_version_string(), OPUS_VERSION_STRING);
        let expected = format!("libopus {}", env!("CARGO_PKG_VERSION"));
        assert_eq!(opus_get_version_string(), expected);
    }
}
