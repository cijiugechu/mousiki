#![allow(dead_code)]

//! Helper routines from `celt/bands.c` that are self-contained enough to port
//! ahead of the rest of the band analysis logic.
//!
//! The goal is to translate building blocks that have little coupling with the
//! more complex pieces of the encoder so that future ports can focus on the
//! higher-level control flow.

use core::f32::consts::FRAC_1_SQRT_2;

use crate::celt::{
    celt_inner_prod, celt_sqrt, ec_ilog,
    types::{CeltGlog, CeltSig, OpusCustomMode, OpusVal16, OpusVal32},
};

/// Fixed-point fractional multiply mirroring the `FRAC_MUL16` macro from the C
/// sources.
#[inline]
fn frac_mul16(a: i32, b: i32) -> i32 {
    let a = a as i16;
    let b = b as i16;
    ((16_384 + i32::from(a) * i32::from(b)) >> 15) as i32
}

/// Bit-exact cosine approximation used by the band analysis heuristics.
///
/// Mirrors `bitexact_cos()` from `celt/bands.c`. The helper operates entirely
/// in 16-bit fixed-point arithmetic so that it matches the reference
/// implementation across platforms.
#[must_use]
pub(crate) fn bitexact_cos(x: i16) -> i16 {
    let tmp = (4_096 + i32::from(x) * i32::from(x)) >> 13;
    let mut x2 = tmp;
    x2 = (32_767 - x2) + frac_mul16(x2, -7_651 + frac_mul16(x2, 8_277 + frac_mul16(-626, x2)));
    (1 + x2) as i16
}

/// Bit-exact logarithmic tangent helper used by the stereo analysis logic.
///
/// Mirrors `bitexact_log2tan()` from `celt/bands.c`, relying on the shared
/// range coder log helper to normalise the sine and cosine magnitudes before
/// evaluating the polynomial approximation.
#[must_use]
pub(crate) fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32) as i32;
    let ls = ec_ilog(isin as u32) as i32;

    let shift_cos = 15 - lc;
    debug_assert!(shift_cos >= 0);
    let icos = icos << shift_cos;

    let shift_sin = 15 - ls;
    debug_assert!(shift_sin >= 0);
    let isin = isin << shift_sin;

    ((ls - lc) << 11) + frac_mul16(isin, frac_mul16(isin, -2_597) + 7_932)
        - frac_mul16(icos, frac_mul16(icos, -2_597) + 7_932)
}

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

    if prev < count && index > prev && value < thresholds[prev] + hysteresis[prev] {
        index = prev;
    }

    if prev > 0 && index < prev && value > thresholds[prev - 1] - hysteresis[prev - 1] {
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

/// Computes stereo weighting factors used when balancing channel distortion.
///
/// Mirrors `compute_channel_weights()` from `celt/bands.c`. The helper adjusts
/// the per-channel energies by a fraction of the smaller energy so that the
/// stereo weighting is slightly more conservative than a pure proportional
/// split.
#[must_use]
pub(crate) fn compute_channel_weights(ex: OpusVal32, ey: OpusVal32) -> [OpusVal16; 2] {
    let min_energy = ex.min(ey);
    let adjusted_ex = ex + min_energy / 3.0;
    let adjusted_ey = ey + min_energy / 3.0;
    [adjusted_ex, adjusted_ey]
}

/// Converts a mid/side representation into left/right stereo samples.
///
/// Mirrors `stereo_split()` from `celt/bands.c`. The helper applies the
/// orthonormal transform that maps a mid (sum) signal and a side (difference)
/// signal back to the left/right domain while preserving energy. CELT encodes
/// mid/side pairs using Q15 fixed-point arithmetic; the float build operates on
/// `f32`, so the Rust port multiplies by `FRAC_1_SQRT_2` instead of the
/// `QCONST16(0.70710678f, 15)` constant used in the original source.
pub(crate) fn stereo_split(x: &mut [f32], y: &mut [f32]) {
    assert_eq!(
        x.len(),
        y.len(),
        "stereo_split expects slices of equal length",
    );

    for (left, right) in x.iter_mut().zip(y.iter_mut()) {
        let mid = FRAC_1_SQRT_2 * *left;
        let side = FRAC_1_SQRT_2 * *right;
        *left = mid + side;
        *right = side - mid;
    }
}

/// Computes the per-band energy for the supplied channels.
///
/// Ports the float build of `compute_band_energies()` from `celt/bands.c`. The
/// helper sums the squared magnitudes within each critical band and stores the
/// square-rooted result in `band_e`. A small bias of `1e-27` mirrors the
/// reference implementation and keeps the normalisation stable even for silent
/// bands.
pub(crate) fn compute_band_energies(
    mode: &OpusCustomMode<'_>,
    x: &[CeltSig],
    band_e: &mut [CeltGlog],
    end: usize,
    channels: usize,
    lm: usize,
    arch: i32,
) {
    let _ = arch;

    assert!(
        end <= mode.num_ebands,
        "end band must not exceed mode bands"
    );
    assert!(
        mode.e_bands.len() >= end + 1,
        "eBands must contain end + 1 entries"
    );

    let n = mode.short_mdct_size << lm;
    assert!(
        x.len() >= channels * n,
        "input spectrum is too short for the mode"
    );

    let stride = mode.num_ebands;
    assert!(
        band_e.len() >= channels * stride,
        "band energy buffer too small"
    );

    for c in 0..channels {
        let signal_base = c * n;
        let energy_base = c * stride;

        for band in 0..end {
            let band_start = (mode.e_bands[band] as usize) << lm;
            let band_end = (mode.e_bands[band + 1] as usize) << lm;
            assert!(band_end <= n, "band end exceeds MDCT length");

            let slice = &x[signal_base + band_start..signal_base + band_end];
            let sum = 1e-27_f32 + celt_inner_prod(slice, slice);
            band_e[energy_base + band] = celt_sqrt(sum);
        }
    }
}

/// Normalises each band to unit energy.
///
/// Mirrors the float implementation of `normalise_bands()` from `celt/bands.c`
/// by scaling the MDCT spectrum in-place. The gain for each band is computed
/// from the `band_e` table produced by [`compute_band_energies`], with the same
/// `1e-27` bias to guard against division by zero.
pub(crate) fn normalise_bands(
    mode: &OpusCustomMode<'_>,
    freq: &[CeltSig],
    x: &mut [OpusVal16],
    band_e: &[CeltGlog],
    end: usize,
    channels: usize,
    m: usize,
) {
    assert!(
        end <= mode.num_ebands,
        "end band must not exceed mode bands"
    );
    assert!(
        mode.e_bands.len() >= end + 1,
        "eBands must contain end + 1 entries"
    );

    let n = m * mode.short_mdct_size;
    assert!(freq.len() >= channels * n, "frequency buffer too small");
    assert!(x.len() >= channels * n, "normalisation buffer too small");

    let stride = mode.num_ebands;
    assert!(
        band_e.len() >= channels * stride,
        "band energy buffer too small"
    );

    for c in 0..channels {
        let freq_base = c * n;
        let energy_base = c * stride;

        for band in 0..end {
            let start = m * (mode.e_bands[band] as usize);
            let stop = m * (mode.e_bands[band + 1] as usize);
            assert!(stop <= n, "band end exceeds MDCT length");

            let gain = 1.0 / (1e-27_f32 + band_e[energy_base + band]);
            for idx in start..stop {
                x[freq_base + idx] = freq[freq_base + idx] * gain;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bitexact_cos, bitexact_log2tan, celt_lcg_rand, compute_band_energies,
        compute_channel_weights, frac_mul16, hysteresis_decision, normalise_bands, stereo_split,
    };
    use crate::celt::types::{CeltSig, MdctLookup, OpusCustomMode, PulseCache};
    use alloc::vec;
    use alloc::vec::Vec;

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
    fn channel_weights_match_reference_formula() {
        let cases = [
            (0.0, 0.0),
            (1.0, 4.0),
            (4.0, 1.0),
            (10.0, 10.0),
            (3.75, 0.25),
        ];

        for &(ex, ey) in &cases {
            let weights = compute_channel_weights(ex, ey);
            let min_energy = ex.min(ey);
            let reference_ex = ex + min_energy / 3.0;
            let reference_ey = ey + min_energy / 3.0;

            assert!((weights[0] - reference_ex).abs() <= f32::EPSILON * 4.0);
            assert!((weights[1] - reference_ey).abs() <= f32::EPSILON * 4.0);
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

    #[test]
    fn stereo_split_matches_reference_transform() {
        let mut mid = [1.0_f32, -1.5, 0.25, 0.0];
        let mut side = [0.5_f32, 2.0, -0.75, 1.25];

        let mut expected_mid = mid;
        let mut expected_side = side;
        for idx in 0..mid.len() {
            let m = core::f32::consts::FRAC_1_SQRT_2 * expected_mid[idx];
            let s = core::f32::consts::FRAC_1_SQRT_2 * expected_side[idx];
            expected_mid[idx] = m + s;
            expected_side[idx] = s - m;
        }

        stereo_split(&mut mid, &mut side);

        for (observed, reference) in mid.iter().zip(expected_mid.iter()) {
            assert!((observed - reference).abs() <= f32::EPSILON * 16.0);
        }
        for (observed, reference) in side.iter().zip(expected_side.iter()) {
            assert!((observed - reference).abs() <= f32::EPSILON * 16.0);
        }
    }

    #[test]
    fn frac_mul16_matches_c_macro() {
        // Compare a handful of values against a direct evaluation of the C
        // macro written in Rust.
        fn reference(a: i32, b: i32) -> i32 {
            let a = a as i16;
            let b = b as i16;
            (16_384 + i32::from(a) * i32::from(b)) >> 15
        }

        let samples = [
            (-32_768, -32_768),
            (-32_768, 32_767),
            (-20_000, 16_000),
            (-626, 8_000),
            (8_277, -5_000),
            (7_932, 2_000),
            (32_767, 32_767),
        ];

        for &(a, b) in &samples {
            assert_eq!(frac_mul16(a, b), reference(a, b));
        }
    }

    #[test]
    fn bitexact_cos_matches_reference_samples() {
        let inputs = [-16_383, -12_000, -6_000, -1, 0, 1, 6_000, 12_000, 16_383];
        let expected = [
            3, 13_371, 27_494, -32_768, -32_768, -32_768, 27_494, 13_371, 3,
        ];

        for (&input, &value) in inputs.iter().zip(expected.iter()) {
            assert_eq!(bitexact_cos(input), value);
        }
    }

    #[test]
    fn bitexact_log2tan_matches_reference_samples() {
        let inputs = [
            (23_170, 32_767),
            (11_585, 32_767),
            (16_384, 23_170),
            (30_000, 12_345),
            (12_345, 30_000),
            (1, 32_767),
            (32_767, 1),
        ];
        let expected = [-1_025, -3_073, -993, 2_631, -2_631, -30_690, 30_690];

        for ((isin, icos), &value) in inputs.iter().zip(expected.iter()) {
            assert_eq!(bitexact_log2tan(*isin, *icos), value);
        }
    }

    fn dummy_mode<'a>(e_bands: &'a [i16], short_mdct_size: usize) -> OpusCustomMode<'a> {
        let mdct = MdctLookup::new(short_mdct_size, 0, &[]);
        OpusCustomMode {
            sample_rate: 48_000,
            overlap: 0,
            num_ebands: e_bands.len() - 1,
            effective_ebands: e_bands.len() - 1,
            pre_emphasis: [0.0; 4],
            e_bands,
            max_lm: 0,
            num_short_mdcts: 1,
            short_mdct_size,
            num_alloc_vectors: 0,
            alloc_vectors: &[],
            log_n: &[],
            window: &[],
            mdct,
            cache: PulseCache {
                size: 0,
                index: &[],
                bits: &[],
                caps: &[],
            },
        }
    }

    #[test]
    fn compute_band_energies_matches_manual_sum() {
        let e_bands = [0i16, 2, 4];
        let mode = dummy_mode(&e_bands, 4);
        let channels = 2;
        let lm = 1usize;
        let n = mode.short_mdct_size << lm;

        let mut spectrum = Vec::with_capacity(channels * n);
        for idx in 0..channels * n {
            spectrum.push((idx as f32 * 0.13 - 0.5).sin());
        }

        let mut band_e = vec![0.0; mode.num_ebands * channels];
        compute_band_energies(
            &mode,
            &spectrum,
            &mut band_e,
            mode.num_ebands,
            channels,
            lm,
            0,
        );

        for c in 0..channels {
            for b in 0..mode.num_ebands {
                let start = ((mode.e_bands[b] as usize) << lm) + c * n;
                let stop = ((mode.e_bands[b + 1] as usize) << lm) + c * n;
                let sum: f32 = spectrum[start..stop].iter().map(|v| v * v).sum();
                let expected = (1e-27_f32 + sum).sqrt();
                let idx = b + c * mode.num_ebands;
                assert!(
                    (band_e[idx] - expected).abs() <= 1e-6,
                    "channel {c}, band {b}"
                );
            }
        }
    }

    #[test]
    fn normalise_bands_scales_by_inverse_energy() {
        let e_bands = [0i16, 2, 4];
        let mode = dummy_mode(&e_bands, 4);
        let channels = 1usize;
        let m = 2usize;
        let n = mode.short_mdct_size * m;

        let freq: Vec<CeltSig> = (0..n).map(|i| (i as f32 * 0.21 - 0.4).cos()).collect();
        let mut norm = vec![0.0f32; freq.len()];

        let mut band_e = vec![0.0f32; mode.num_ebands * channels];
        for b in 0..mode.num_ebands {
            let start = m * (mode.e_bands[b] as usize);
            let stop = m * (mode.e_bands[b + 1] as usize);
            let sum: f32 = freq[start..stop].iter().map(|v| v * v).sum();
            band_e[b] = (1e-27_f32 + sum).sqrt();
        }

        normalise_bands(
            &mode,
            &freq,
            &mut norm,
            &band_e,
            mode.num_ebands,
            channels,
            m,
        );

        for b in 0..mode.num_ebands {
            let start = m * (mode.e_bands[b] as usize);
            let stop = m * (mode.e_bands[b + 1] as usize);
            let gain = 1.0 / (1e-27_f32 + band_e[b]);
            for j in start..stop {
                assert!(
                    (norm[j] - freq[j] * gain).abs() <= 1e-6,
                    "band {b}, index {j}"
                );
            }
        }
    }
}
