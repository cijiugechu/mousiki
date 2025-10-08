#![allow(dead_code)]

//! Helpers ported from `celt/celt.c`.
//!
//! This module starts translating small pieces of the CELT top-level glue. The
//! helpers exposed here have no dependencies on the rest of the encoder or
//! decoder state so they can be exercised in isolation while larger control
//! flow is still being translated.

use crate::celt::types::{CeltCoef, OpusCustomMode, OpusInt32, OpusVal16, OpusVal32};

/// Minimum comb-filter period supported by the scalar implementation.
const COMBFILTER_MINPERIOD: usize = 15;

/// Tapset gains mirroring the tables embedded in the reference implementation.
const TAPSET_GAINS: [[OpusVal16; 3]; 3] = [
    [0.306_640_62, 0.217_041_02, 0.129_638_67],
    [0.463_867_2, 0.268_066_4, 0.0],
    [0.799_804_7, 0.100_097_656, 0.0],
];

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

/// Applies the constant-coefficient comb filter used by the encoder/decoder.
///
/// Mirrors `comb_filter_const_c()` from `celt/celt.c` for the float build. The
/// `x` slice must expose at least `t + 2` samples of history before
/// `x_start` alongside `y.len()` samples starting at `x_start`, allowing the
/// routine to mirror the negative pointer indexing present in the C
/// implementation.
pub(crate) fn comb_filter_const(
    y: &mut [OpusVal32],
    x: &[OpusVal32],
    x_start: usize,
    t: usize,
    g10: CeltCoef,
    g11: CeltCoef,
    g12: CeltCoef,
) {
    let n = y.len();
    if n == 0 {
        return;
    }

    assert!(t >= COMBFILTER_MINPERIOD, "comb filter period too small");
    assert!(
        x_start >= t + 2,
        "input slice does not provide enough history for the comb filter",
    );
    assert!(
        x.len() >= x_start + n,
        "input slice must provide x_start + n samples",
    );

    let mut x4 = x[x_start - t - 2];
    let mut x3 = x[x_start - t - 1];
    let mut x2 = x[x_start - t];
    let mut x1 = x[x_start - t + 1];

    for (i, sample) in y.iter_mut().enumerate() {
        let current = x[x_start + i];
        let x0 = x[x_start + i - t + 2];
        let acc = current + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
        *sample = acc;

        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

/// Applies the variable tapset comb filter with optional overlap ramping.
///
/// Mirrors the scalar implementation of `comb_filter()` from `celt/celt.c`.
/// The caller must provide the `x` buffer with enough history before
/// `x_start` (at least `max(T0, T1) + 2` samples) in addition to the `n`
/// samples of the current frame. `y` must provide room for `n` output samples.
#[allow(clippy::too_many_arguments)]
pub(crate) fn comb_filter(
    y: &mut [OpusVal32],
    x: &[OpusVal32],
    x_start: usize,
    n: usize,
    mut t0: i32,
    mut t1: i32,
    g0: OpusVal16,
    g1: OpusVal16,
    tapset0: usize,
    tapset1: usize,
    window: &[CeltCoef],
    overlap: usize,
    _arch: i32,
) {
    if n == 0 {
        return;
    }

    assert!(n <= y.len(), "output slice must hold n samples");
    assert!(x.len() >= x_start + n, "input slice must expose n samples");
    assert!(tapset0 < TAPSET_GAINS.len(), "invalid tapset index");
    assert!(tapset1 < TAPSET_GAINS.len(), "invalid tapset index");

    if g0 == 0.0 && g1 == 0.0 {
        let src = &x[x_start..x_start + n];
        y[..n].copy_from_slice(src);
        return;
    }

    t0 = t0.max(COMBFILTER_MINPERIOD as i32);
    t1 = t1.max(COMBFILTER_MINPERIOD as i32);
    let t0 = t0 as usize;
    let t1 = t1 as usize;

    assert!(
        x_start >= t0 + 2 && x_start >= t1 + 2,
        "input slice lacks the required comb filter history",
    );

    let tap0 = TAPSET_GAINS[tapset0];
    let tap1 = TAPSET_GAINS[tapset1];
    let g00 = g0 * tap0[0];
    let g01 = g0 * tap0[1];
    let g02 = g0 * tap0[2];
    let g10 = g1 * tap1[0];
    let g11 = g1 * tap1[1];
    let g12 = g1 * tap1[2];

    let mut x1 = x[x_start - t1 + 1];
    let mut x2 = x[x_start - t1];
    let mut x3 = x[x_start - t1 - 1];
    let mut x4 = x[x_start - t1 - 2];

    let mut overlap = overlap.min(n);
    if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        overlap = 0;
    } else if overlap > 0 {
        assert!(
            window.len() >= overlap,
            "window must expose at least overlap samples",
        );
    }

    for i in 0..overlap {
        let x0 = x[x_start + i - t1 + 2];
        let f = window[i] * window[i];
        let one_minus_f = 1.0 - f;

        let current = x[x_start + i];
        let past0 = x[x_start + i - t0];
        let past1 = x[x_start + i - t0 + 1];
        let pastm1 = x[x_start + i - t0 - 1];
        let past2 = x[x_start + i - t0 + 2];
        let pastm2 = x[x_start + i - t0 - 2];

        let blended = current
            + one_minus_f * g00 * past0
            + one_minus_f * g01 * (past1 + pastm1)
            + one_minus_f * g02 * (past2 + pastm2)
            + f * g10 * x2
            + f * g11 * (x1 + x3)
            + f * g12 * (x0 + x4);

        y[i] = blended;

        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }

    if g1 == 0.0 {
        if overlap < n {
            let src = &x[x_start + overlap..x_start + n];
            y[overlap..n].copy_from_slice(src);
        }
        return;
    }

    if overlap < n {
        comb_filter_const(&mut y[overlap..n], x, x_start + overlap, t1, g10, g11, g12);
    }
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
        mode.e_bands.len() > nb_ebands,
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
        OPUS_VERSION_STRING, TF_SELECT_TABLE, comb_filter, comb_filter_const, init_caps,
        opus_get_version_string, opus_strerror, resampling_factor,
    };
    use crate::celt::types::{MdctLookup, OpusCustomMode, PulseCacheData};
    use alloc::vec::Vec;
    use alloc::{format, vec};

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPSILON
    }

    fn comb_filter_const_reference(
        x: &[f32],
        x_start: usize,
        t: usize,
        n: usize,
        g10: f32,
        g11: f32,
        g12: f32,
    ) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }

        let mut out = vec![0.0; n];
        let mut x4 = x[x_start - t - 2];
        let mut x3 = x[x_start - t - 1];
        let mut x2 = x[x_start - t];
        let mut x1 = x[x_start - t + 1];

        for i in 0..n {
            let current = x[x_start + i];
            let x0 = x[x_start + i - t + 2];
            out[i] = current + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
            x4 = x3;
            x3 = x2;
            x2 = x1;
            x1 = x0;
        }

        out
    }

    fn comb_filter_reference(
        x: &[f32],
        x_start: usize,
        n: usize,
        t0: i32,
        t1: i32,
        g0: f32,
        g1: f32,
        tapset0: usize,
        tapset1: usize,
        window: &[f32],
        overlap: usize,
    ) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }

        if g0 == 0.0 && g1 == 0.0 {
            return x[x_start..x_start + n].to_vec();
        }

        let t0 = t0.max(super::COMBFILTER_MINPERIOD as i32) as usize;
        let t1 = t1.max(super::COMBFILTER_MINPERIOD as i32) as usize;

        let tap0 = super::TAPSET_GAINS[tapset0];
        let tap1 = super::TAPSET_GAINS[tapset1];
        let g00 = g0 * tap0[0];
        let g01 = g0 * tap0[1];
        let g02 = g0 * tap0[2];
        let g10 = g1 * tap1[0];
        let g11 = g1 * tap1[1];
        let g12 = g1 * tap1[2];

        let mut out = vec![0.0; n];

        let mut x1 = x[x_start - t1 + 1];
        let mut x2 = x[x_start - t1];
        let mut x3 = x[x_start - t1 - 1];
        let mut x4 = x[x_start - t1 - 2];

        let mut overlap = overlap.min(n);
        if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
            overlap = 0;
        }

        for i in 0..overlap {
            let x0 = x[x_start + i - t1 + 2];
            let f = window[i] * window[i];
            let one_minus_f = 1.0 - f;

            let current = x[x_start + i];
            let past0 = x[x_start + i - t0];
            let past1 = x[x_start + i - t0 + 1];
            let pastm1 = x[x_start + i - t0 - 1];
            let past2 = x[x_start + i - t0 + 2];
            let pastm2 = x[x_start + i - t0 - 2];

            out[i] = current
                + one_minus_f * g00 * past0
                + one_minus_f * g01 * (past1 + pastm1)
                + one_minus_f * g02 * (past2 + pastm2)
                + f * g10 * x2
                + f * g11 * (x1 + x3)
                + f * g12 * (x0 + x4);

            x4 = x3;
            x3 = x2;
            x2 = x1;
            x1 = x0;
        }

        if g1 == 0.0 {
            if overlap < n {
                out[overlap..].copy_from_slice(&x[x_start + overlap..x_start + n]);
            }
            return out;
        }

        if overlap < n {
            let tail =
                comb_filter_const_reference(x, x_start + overlap, t1, n - overlap, g10, g11, g12);
            out[overlap..].copy_from_slice(&tail);
        }

        out
    }

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

    #[test]
    fn comb_filter_const_matches_reference() {
        let history = super::COMBFILTER_MINPERIOD + 2;
        let n = 6;
        let mut x = Vec::new();
        for i in 0..(history + n + 4) {
            x.push(i as f32 * 0.25);
        }

        let mut output = vec![0.0; n];
        let g10 = 0.45;
        let g11 = -0.2;
        let g12 = 0.05;

        comb_filter_const(
            &mut output,
            &x,
            history,
            super::COMBFILTER_MINPERIOD,
            g10,
            g11,
            g12,
        );

        let expected =
            comb_filter_const_reference(&x, history, super::COMBFILTER_MINPERIOD, n, g10, g11, g12);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn comb_filter_matches_reference() {
        let t0 = 21;
        let t1 = 27;
        let history = (t1 as usize) + 3;
        let n = 12;
        let mut x = Vec::new();
        for i in 0..(history + n + 6) {
            x.push((i as f32 * 0.13).sin());
        }

        let mut y = vec![0.0; n];
        let g0 = 0.6;
        let g1 = 0.35;
        let tapset0 = 0;
        let tapset1 = 2;
        let window = [0.1, 0.4, 0.6, 0.7, 0.5, 0.2];
        let overlap = window.len();

        comb_filter(
            &mut y, &x, history, n, t0, t1, g0, g1, tapset0, tapset1, &window, overlap, 0,
        );

        let expected = comb_filter_reference(
            &x, history, n, t0, t1, g0, g1, tapset0, tapset1, &window, overlap,
        );

        for (a, b) in y.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn comb_filter_zero_gains_copy_input() {
        let history = super::COMBFILTER_MINPERIOD + 5;
        let n = 8;
        let mut x = Vec::new();
        for i in 0..(history + n) {
            x.push(i as f32 * 0.5);
        }

        let mut y = vec![0.0; n];
        comb_filter(&mut y, &x, history, n, 10, 12, 0.0, 0.0, 0, 1, &[], 0, 0);

        let expected = x[history..history + n].to_vec();
        assert_eq!(y, expected);
    }
}
