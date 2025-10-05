#![allow(dead_code)]

//! Helper routines from `celt/bands.c` that are self-contained enough to port
//! ahead of the rest of the band analysis logic.
//!
//! The goal is to translate building blocks that have little coupling with the
//! more complex pieces of the encoder so that future ports can focus on the
//! higher-level control flow.

use alloc::vec;

use core::f32::consts::FRAC_1_SQRT_2;

use crate::celt::{
    SPREAD_AGGRESSIVE, SPREAD_LIGHT, SPREAD_NONE, SPREAD_NORMAL, celt_exp2, celt_inner_prod,
    celt_rsqrt, celt_rsqrt_norm, celt_sqrt, celt_udiv, dual_inner_prod, ec_ilog,
    renormalise_vector,
    types::{CeltGlog, CeltSig, OpusCustomMode, OpusVal16, OpusVal32},
};

/// Small positive constant used throughout the CELT band tools to avoid divisions by zero.
const EPSILON: f32 = 1e-15;

/// Indexing table for converting natural-order Hadamard coefficients into the
/// "ordery" permutation used by CELT's spreading analysis.
///
/// The layout mirrors the compact array embedded in `celt/bands.c`, grouping
/// permutations for strides of 2, 4, 8, and 16. The Hadamard interleaving logic
/// selects the slice corresponding to the current stride when the `hadamard`
/// flag is active.
const ORDERY_TABLES: [&[usize]; 4] = [
    &[1, 0],
    &[3, 0, 2, 1],
    &[7, 0, 4, 3, 6, 1, 5, 2],
    &[15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5],
];

fn hadamard_ordery(stride: usize) -> Option<&'static [usize]> {
    match stride {
        2 => Some(ORDERY_TABLES[0]),
        4 => Some(ORDERY_TABLES[1]),
        8 => Some(ORDERY_TABLES[2]),
        16 => Some(ORDERY_TABLES[3]),
        _ => None,
    }
}

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

/// Collapses an intensity-coded stereo band back into the mid channel.
///
/// Mirrors the float configuration of `intensity_stereo()` from `celt/bands.c`.
/// The helper derives linear weights from the per-channel band energies and
/// mixes the encoded side channel into the mid channel while preserving the
/// overall energy of the pair.
pub(crate) fn intensity_stereo(
    mode: &OpusCustomMode<'_>,
    x: &mut [OpusVal16],
    y: &[OpusVal16],
    band_e: &[OpusVal32],
    band_id: usize,
    n: usize,
) {
    assert!(
        band_id < mode.num_ebands,
        "band index must be within the mode range"
    );
    assert!(x.len() >= n, "output band must contain at least n samples");
    assert!(y.len() >= n, "side band must contain at least n samples");

    let stride = mode.num_ebands;
    assert!(
        band_e.len() >= stride * 2,
        "band energy buffer must store both channel energies",
    );
    assert!(
        band_id + stride < band_e.len(),
        "band energy buffer too small for right channel",
    );

    let left = band_e[band_id];
    let right = band_e[band_id + stride];
    let norm = EPSILON + celt_sqrt(EPSILON + left * left + right * right);
    let a1 = left / norm;
    let a2 = right / norm;

    for idx in 0..n {
        let l = x[idx];
        let r = y[idx];
        x[idx] = a1 * l + a2 * r;
    }
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

/// Restores energy to bands that collapsed during transient coding.
///
/// Mirrors the float build of `anti_collapse()` from `celt/bands.c`. When a
/// short MDCT band loses all pulses the decoder injects shaped noise with a
/// gain derived from recent band energies. The helper mirrors the reference
/// pseudo-random sequence, energy guards, and subsequent renormalisation so the
/// decoder matches the C implementation bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn anti_collapse(
    mode: &OpusCustomMode<'_>,
    x: &mut [OpusVal16],
    collapse_masks: &[u8],
    lm: usize,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[CeltGlog],
    prev1_log_e: &[CeltGlog],
    prev2_log_e: &[CeltGlog],
    pulses: &[i32],
    mut seed: u32,
    encode: bool,
    arch: i32,
) {
    assert!(channels > 0, "anti_collapse requires at least one channel");
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(
        collapse_masks.len() >= channels * end,
        "collapse masks too short"
    );
    assert!(
        log_e.len() >= channels * mode.num_ebands,
        "logE buffer too small"
    );
    assert!(
        prev1_log_e.len() >= channels * mode.num_ebands,
        "prev1 buffer too small"
    );
    assert!(
        prev2_log_e.len() >= channels * mode.num_ebands,
        "prev2 buffer too small"
    );
    assert!(
        pulses.len() >= end,
        "pulse buffer too small for requested bands"
    );

    let expected_stride = mode.short_mdct_size << lm;
    assert_eq!(
        size, expected_stride,
        "channel stride must match the MDCT length for the block size",
    );
    assert!(
        x.len() >= channels * size,
        "spectrum buffer shorter than the requested channel span",
    );

    let block_count = 1usize << lm;
    let band_stride = mode.num_ebands;

    for band in start..end {
        let band_begin =
            usize::try_from(mode.e_bands[band]).expect("band index must be non-negative");
        let band_end =
            usize::try_from(mode.e_bands[band + 1]).expect("band index must be non-negative");
        let width = band_end.saturating_sub(band_begin);
        if width == 0 {
            continue;
        }

        let pulses_for_band = pulses[band];
        assert!(pulses_for_band >= 0, "pulse counts must be non-negative");
        let numerator = u32::try_from(pulses_for_band)
            .expect("pulse count fits in u32")
            .wrapping_add(1);
        let denom = u32::try_from(width).expect("band width fits in u32");
        debug_assert!(denom > 0, "band width must be positive");
        let depth = (celt_udiv(numerator, denom) >> lm) as i32;

        let thresh = 0.5 * celt_exp2(-0.125 * depth as f32);
        let sqrt_1 = celt_rsqrt((width << lm) as f32);

        for channel in 0..channels {
            let mask = collapse_masks[band * channels + channel] as u32;
            let channel_base = channel * size;
            let band_base = channel_base + (band_begin << lm);
            let band_len = width << lm;
            assert!(
                band_base + band_len <= x.len(),
                "band slice exceeds spectrum length"
            );

            let mut prev1 = prev1_log_e[channel * band_stride + band];
            let mut prev2 = prev2_log_e[channel * band_stride + band];

            if !encode && channels == 1 {
                let alt = band_stride + band;
                if alt < prev1_log_e.len() {
                    prev1 = prev1.max(prev1_log_e[alt]);
                }
                if alt < prev2_log_e.len() {
                    prev2 = prev2.max(prev2_log_e[alt]);
                }
            }

            let mut ediff = log_e[channel * band_stride + band] - prev1.min(prev2);
            if ediff < 0.0 {
                ediff = 0.0;
            }

            let mut r = 2.0 * celt_exp2(-ediff);
            if lm == 3 {
                r *= 1.414_213_56;
            }
            r = r.min(thresh);
            r *= sqrt_1;

            let mut needs_renorm = false;

            for k in 0..block_count {
                if mask & (1u32 << k) == 0 {
                    for j in 0..width {
                        seed = celt_lcg_rand(seed);
                        let idx = band_base + (j << lm) + k;
                        x[idx] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    needs_renorm = true;
                }
            }

            if needs_renorm {
                let end_idx = band_base + band_len;
                renormalise_vector(&mut x[band_base..end_idx], band_len, 1.0, arch);
            }
        }
    }
}

/// Reconstructs left/right stereo samples from a mid/side representation.
///
/// Mirrors the float configuration of `stereo_merge()` from `celt/bands.c`.
/// The helper evaluates the energies of the `X ± Y` combinations to derive
/// normalisation gains, then applies the inverse transform to recover the
/// left and right channels. If either energy falls below the conservative
/// threshold used by the reference implementation, the side channel is
/// replaced by the mid channel to avoid amplifying near-silent noise.
pub(crate) fn stereo_merge(x: &mut [OpusVal16], y: &mut [OpusVal16], mid: OpusVal32) {
    assert_eq!(
        x.len(),
        y.len(),
        "stereo_merge expects slices of equal length",
    );

    if x.is_empty() {
        return;
    }

    let (mut cross, side_energy) = dual_inner_prod(y, x, y);
    cross *= mid;
    let mid_energy = mid * mid;
    let el = mid_energy + side_energy - 2.0 * cross;
    let er = mid_energy + side_energy + 2.0 * cross;

    if er < 6e-4 || el < 6e-4 {
        y.copy_from_slice(x);
        return;
    }

    let lgain = celt_rsqrt_norm(el);
    let rgain = celt_rsqrt_norm(er);

    for (left, right) in x.iter_mut().zip(y.iter_mut()) {
        let mid_scaled = mid * *left;
        let side_val = *right;
        *left = lgain * (mid_scaled - side_val);
        *right = rgain * (mid_scaled + side_val);
    }
}

/// Decides how aggressively PVQ pulses should be spread in the current frame.
///
/// Mirrors the float configuration of `spreading_decision()` from
/// `celt/bands.c`. The helper analyses the normalised spectrum stored in `x`
/// and classifies each band based on the proportion of low-energy coefficients.
/// The resulting score is filtered through a simple recursive average and a
/// small hysteresis term controlled by the previous decision. High-frequency
/// statistics optionally update the pitch tapset selector when `update_hf` is
/// `true`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spreading_decision(
    mode: &OpusCustomMode,
    x: &[OpusVal16],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: usize,
    channels: usize,
    m: usize,
    spread_weight: &[i32],
) -> i32 {
    assert!(end > 0, "band range must contain at least one band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(spread_weight.len() >= end, "insufficient spread weights");

    let n0 = m * mode.short_mdct_size;
    assert!(x.len() >= channels * n0, "spectrum buffer too small");

    let last_band_width =
        m * (mode.e_bands[end] as usize).saturating_sub(mode.e_bands[end - 1] as usize);
    if last_band_width <= 8 {
        return SPREAD_NONE;
    }

    let mut sum = 0i32;
    let mut nb_bands = 0i32;
    let mut hf_sum = 0i32;

    for c in 0..channels {
        let channel_base = c * n0;
        for band in 0..end {
            let start = m * (mode.e_bands[band] as usize);
            let stop = m * (mode.e_bands[band + 1] as usize);
            let n = stop - start;
            if n <= 8 {
                continue;
            }

            let slice = &x[channel_base + start..channel_base + stop];
            let mut tcount = [0i32; 3];
            for &value in slice {
                let x2n = value * value * n as OpusVal16;
                if x2n < 0.25 {
                    tcount[0] += 1;
                }
                if x2n < 0.0625 {
                    tcount[1] += 1;
                }
                if x2n < 0.015625 {
                    tcount[2] += 1;
                }
            }

            if band + 4 > mode.num_ebands {
                let numerator = 32 * (tcount[1] + tcount[0]);
                hf_sum += celt_udiv(numerator as u32, n as u32) as i32;
            }

            let mut tmp = 0i32;
            if 2 * tcount[2] >= n as i32 {
                tmp += 1;
            }
            if 2 * tcount[1] >= n as i32 {
                tmp += 1;
            }
            if 2 * tcount[0] >= n as i32 {
                tmp += 1;
            }

            let weight = spread_weight[band];
            sum += tmp * weight;
            nb_bands += weight;
        }
    }

    if update_hf {
        if hf_sum != 0 {
            let denom = (channels as i32) * (4 - mode.num_ebands as i32 + end as i32);
            if denom > 0 {
                hf_sum = celt_udiv(hf_sum as u32, denom as u32) as i32;
            } else {
                hf_sum = 0;
            }
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        match *tapset_decision {
            2 => hf_sum += 4,
            0 => hf_sum -= 4,
            _ => {}
        }
        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }

    assert!(
        nb_bands > 0,
        "spreading analysis requires at least one band"
    );
    let scaled = ((sum as i64) << 8) as u32;
    let denom = nb_bands as u32;
    let mut sum = celt_udiv(scaled, denom) as i32;
    sum = (sum + *average) >> 1;
    *average = sum;

    let hysteresis = ((3 - last_decision) << 7) + 64;
    sum = (3 * sum + hysteresis + 2) >> 2;

    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

/// Restores the natural band ordering after a Hadamard transform.
///
/// Mirrors `deinterleave_hadamard()` from `celt/bands.c`. The routine copies the
/// interleaved coefficients into a temporary buffer before writing them back in
/// natural band order. When `hadamard` is `true`, the function applies the
/// "ordery" permutation so that the Hadamard DC term appears at the end of the
/// output sequence, matching the reference implementation.
pub(crate) fn deinterleave_hadamard(x: &mut [OpusVal16], n0: usize, stride: usize, hadamard: bool) {
    if stride == 0 {
        return;
    }

    let n = n0.checked_mul(stride).expect("stride * n0 overflowed");
    assert!(x.len() >= n, "input buffer too small for deinterleave");

    if n == 0 {
        return;
    }

    let mut tmp = vec![0.0f32; n];

    if hadamard {
        let ordery = hadamard_ordery(stride)
            .expect("hadamard interleave only defined for strides of 2, 4, 8, or 16");
        assert_eq!(ordery.len(), stride);
        for (i, &ord) in ordery.iter().enumerate() {
            for j in 0..n0 {
                tmp[ord * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }

    x[..n].copy_from_slice(&tmp);
}

/// Applies the Hadamard interleaving used by CELT's spreading decisions.
///
/// Mirrors `interleave_hadamard()` from `celt/bands.c`. The helper stores the
/// natural-order coefficients into a temporary buffer, optionally applying the
/// "ordery" permutation when `hadamard` is `true`. The resulting layout matches
/// the reference code, ensuring that deinterleaving reverses the transform
/// exactly.
pub(crate) fn interleave_hadamard(x: &mut [OpusVal16], n0: usize, stride: usize, hadamard: bool) {
    if stride == 0 {
        return;
    }

    let n = n0.checked_mul(stride).expect("stride * n0 overflowed");
    assert!(x.len() >= n, "input buffer too small for interleave");

    if n == 0 {
        return;
    }

    let mut tmp = vec![0.0f32; n];

    if hadamard {
        let ordery = hadamard_ordery(stride)
            .expect("hadamard interleave only defined for strides of 2, 4, 8, or 16");
        assert_eq!(ordery.len(), stride);
        for (i, &ord) in ordery.iter().enumerate() {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ord * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }

    x[..n].copy_from_slice(&tmp);
}

/// Applies a single-level Haar transform across interleaved coefficients.
///
/// Mirrors `haar1()` from `celt/bands.c`, scaling each pair of samples by
/// `1/sqrt(2)` before computing their sum and difference. The coefficients are
/// laid out in `stride` interleaved groups, matching the memory layout used by
/// CELT's band folding routines.
pub(crate) fn haar1(x: &mut [OpusVal16], n0: usize, stride: usize) {
    if stride == 0 || n0 < 2 {
        return;
    }

    let half = n0 / 2;
    if half == 0 {
        return;
    }

    let required = stride * n0;
    assert!(
        x.len() >= required,
        "haar1 expects at least stride * n0 coefficients"
    );

    let scale = FRAC_1_SQRT_2 as OpusVal16;

    for i in 0..stride {
        for j in 0..half {
            let idx0 = stride * (2 * j) + i;
            let idx1 = idx0 + stride;
            debug_assert!(idx1 < x.len());

            let tmp1 = scale * x[idx0];
            let tmp2 = scale * x[idx1];
            x[idx0] = tmp1 + tmp2;
            x[idx1] = tmp1 - tmp2;
        }
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
        EPSILON, anti_collapse, bitexact_cos, bitexact_log2tan, celt_lcg_rand,
        compute_band_energies, compute_channel_weights, deinterleave_hadamard, frac_mul16, haar1,
        hysteresis_decision, intensity_stereo, interleave_hadamard, normalise_bands,
        spreading_decision, stereo_merge, stereo_split,
    };
    use crate::celt::types::{CeltSig, MdctLookup, OpusCustomMode, PulseCacheData};
    use crate::celt::{
        SPREAD_AGGRESSIVE, SPREAD_NONE, SPREAD_NORMAL, celt_rsqrt_norm, dual_inner_prod,
    };
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
    fn haar1_preserves_signal_when_applied_twice() {
        let mut data = vec![
            0.25_f32, -1.5, 3.5, 0.75, -2.25, 1.0, 0.5, -0.125, 2.0, -3.0, 1.5, 0.25,
        ];
        let original = data.clone();

        // Apply the transform twice; the Haar matrix is orthonormal so the
        // second application inverts the first.
        haar1(&mut data, 12, 1);
        haar1(&mut data, 12, 1);

        for (expected, observed) in original.iter().zip(data.iter()) {
            assert!(
                (expected - observed).abs() <= 1e-6,
                "expected {expected}, got {observed}"
            );
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
    fn intensity_stereo_matches_reference_weights() {
        let e_bands = [0i16, 2, 4, 6, 8];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 4];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(4, 0);
        let mode = OpusCustomMode::new(
            48_000,
            0,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            PulseCacheData::default(),
        );

        let mut x = vec![0.5, -0.75, 0.25, -0.125];
        let y = vec![0.25, 0.5, -0.5, 0.75];
        let mut band_e = vec![0.0f32; mode.num_ebands * 2];
        band_e[2] = 1.8;
        band_e[2 + mode.num_ebands] = 0.9;

        let left = band_e[2];
        let right = band_e[2 + mode.num_ebands];
        let norm = EPSILON + (EPSILON + left * left + right * right).sqrt();
        let a1 = left / norm;
        let a2 = right / norm;
        let mut expected = x.clone();
        for (idx, value) in expected.iter_mut().enumerate() {
            *value = a1 * *value + a2 * y[idx];
        }

        intensity_stereo(&mode, &mut x, &y, &band_e, 2, y.len());

        for (idx, (&observed, &reference)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (observed - reference).abs() <= 1e-6,
                "sample {idx}: observed={observed}, expected={reference}"
            );
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
    fn spreading_returns_aggressive_for_concentrated_energy() {
        let e_bands = [0i16, 16, 32];
        let mode = dummy_mode(&e_bands, 32);
        let channels = 1;
        let m = 1;
        let end = 2;
        let spread_weight = [1, 1];
        let spectrum = vec![1.0f32; channels * m * mode.short_mdct_size];
        let mut average = 0;
        let mut hf_average = 0;
        let mut tapset = 1;

        let decision = spreading_decision(
            &mode,
            &spectrum,
            &mut average,
            SPREAD_NORMAL,
            &mut hf_average,
            &mut tapset,
            false,
            end,
            channels,
            m,
            &spread_weight,
        );

        assert_eq!(decision, SPREAD_AGGRESSIVE);
        assert_eq!(tapset, 1);
        assert_eq!(hf_average, 0);
        assert_eq!(average, 0);
    }

    #[test]
    fn spreading_returns_normal_when_single_threshold_met() {
        let e_bands = [0i16, 16, 32];
        let mode = dummy_mode(&e_bands, 32);
        let channels = 1;
        let m = 1;
        let end = 2;
        let spread_weight = [1, 1];
        let mut spectrum = vec![1.0f32; channels * m * mode.short_mdct_size];
        for idx in 0..8 {
            spectrum[idx] = 0.1;
        }
        let mut average = 0;
        let mut hf_average = 0;
        let mut tapset = 1;

        let decision = spreading_decision(
            &mode,
            &spectrum,
            &mut average,
            SPREAD_NORMAL,
            &mut hf_average,
            &mut tapset,
            false,
            end,
            channels,
            m,
            &spread_weight,
        );

        assert_eq!(decision, SPREAD_NORMAL);
        assert_eq!(tapset, 1);
        assert_eq!(hf_average, 0);
        assert_eq!(average, 64);
    }

    #[test]
    fn spreading_returns_none_when_all_thresholds_met() {
        let e_bands = [0i16, 16, 32];
        let mode = dummy_mode(&e_bands, 32);
        let channels = 1;
        let m = 1;
        let end = 1;
        let spread_weight = [1];
        let mut spectrum = vec![1.0f32; channels * m * mode.short_mdct_size];
        for idx in 0..8 {
            spectrum[idx] = 0.01;
        }
        let mut average = 0;
        let mut hf_average = 0;
        let mut tapset = 1;

        let decision = spreading_decision(
            &mode,
            &spectrum,
            &mut average,
            SPREAD_NONE,
            &mut hf_average,
            &mut tapset,
            false,
            end,
            channels,
            m,
            &spread_weight,
        );

        assert_eq!(decision, SPREAD_NONE);
        assert_eq!(tapset, 1);
        assert_eq!(hf_average, 0);
        assert_eq!(average, 384);
    }

    #[test]
    fn spreading_updates_hf_tracking() {
        let e_bands: Vec<i16> = (0..=10).map(|i| (i * 24) as i16).collect();
        let mode = dummy_mode(&e_bands, 256);
        let channels = 1;
        let m = 1;
        let end = mode.num_ebands;
        let spread_weight = vec![1; end];
        let spectrum = vec![0.0f32; channels * m * mode.short_mdct_size];
        let mut average = 0;
        let mut hf_average = 0;
        let mut tapset = 1;

        let decision = spreading_decision(
            &mode,
            &spectrum,
            &mut average,
            SPREAD_NONE,
            &mut hf_average,
            &mut tapset,
            true,
            end,
            channels,
            m,
            &spread_weight,
        );

        assert_eq!(decision, SPREAD_NONE);
        assert_eq!(tapset, 2);
        assert_eq!(hf_average, 24);
        assert_eq!(average, 384);
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

    fn reference_stereo_merge(x: &[f32], y: &[f32], mid: f32) -> (Vec<f32>, Vec<f32>) {
        let mut left = x.to_vec();
        let mut right = y.to_vec();

        let (mut cross, side_energy) = dual_inner_prod(&right, &left, &right);
        cross *= mid;
        let mid_energy = mid * mid;
        let el = mid_energy + side_energy - 2.0 * cross;
        let er = mid_energy + side_energy + 2.0 * cross;

        if er < 6e-4 || el < 6e-4 {
            right.copy_from_slice(&left);
            return (left, right);
        }

        let lgain = celt_rsqrt_norm(el);
        let rgain = celt_rsqrt_norm(er);

        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mid_scaled = mid * *l;
            let side_val = *r;
            *l = lgain * (mid_scaled - side_val);
            *r = rgain * (mid_scaled + side_val);
        }

        (left, right)
    }

    #[test]
    fn stereo_merge_matches_reference_transform() {
        let mut left = [0.8, -0.25, 0.5, -0.75, 0.1, 0.3];
        let mut right = [-0.2, 0.4, -0.6, 0.3, -0.1, 0.2];
        let mid = 0.9;

        let (expected_left, expected_right) = reference_stereo_merge(&left, &right, mid);
        stereo_merge(&mut left, &mut right, mid);

        for (idx, (&value, &expected)) in left.iter().zip(expected_left.iter()).enumerate() {
            assert!(
                (value - expected).abs() <= 1e-6,
                "left[{idx}] mismatch: value={value}, expected={expected}"
            );
        }

        for (idx, (&value, &expected)) in right.iter().zip(expected_right.iter()).enumerate() {
            assert!(
                (value - expected).abs() <= 1e-6,
                "right[{idx}] mismatch: value={value}, expected={expected}"
            );
        }
    }

    #[test]
    fn stereo_merge_copies_mid_for_low_energy() {
        let mut left = [0.0f32; 4];
        let mut right = [1e-3f32, -1e-3, 2e-3, -2e-3];
        stereo_merge(&mut left, &mut right, 0.0);
        assert_eq!(right, left);
    }

    fn reference_ordery(stride: usize) -> &'static [usize] {
        match stride {
            2 => &[1, 0],
            4 => &[3, 0, 2, 1],
            8 => &[7, 0, 4, 3, 6, 1, 5, 2],
            16 => &[15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5],
            _ => panic!("unsupported stride"),
        }
    }

    #[test]
    fn hadamard_interleave_matches_reference_layout() {
        for &stride in &[2, 4, 8, 16] {
            let n0 = 4usize;
            let n = n0 * stride;
            let mut data: Vec<f32> = (0..n).map(|v| v as f32).collect();
            let mut expected = data.clone();

            let ordery = reference_ordery(stride);
            for (i, &ord) in ordery.iter().enumerate() {
                for j in 0..n0 {
                    expected[j * stride + i] = data[ord * n0 + j];
                }
            }

            interleave_hadamard(&mut data, n0, stride, true);
            assert_eq!(data[..n], expected[..n]);
        }
    }

    #[test]
    fn interleave_and_deinterleave_round_trip() {
        let cases = [
            (4usize, 2usize, false),
            (4, 2, true),
            (8, 4, false),
            (8, 4, true),
        ];

        for &(n0, stride, hadamard) in &cases {
            let n = n0 * stride;
            let data: Vec<f32> = (0..n).map(|v| (v as f32) * 0.5 - 3.0).collect();
            let mut transformed = data.clone();
            interleave_hadamard(&mut transformed, n0, stride, hadamard);
            deinterleave_hadamard(&mut transformed, n0, stride, hadamard);
            assert_eq!(transformed[..n], data[..n]);
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
        let mdct = MdctLookup::new(short_mdct_size, 0);
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
            cache: PulseCacheData::default(),
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

    #[test]
    fn anti_collapse_fills_collapsed_band_with_noise() {
        let e_bands = [0i16, 2];
        let mode = dummy_mode(&e_bands, 4);
        let lm = 1usize;
        let channels = 1usize;
        let size = mode.short_mdct_size << lm;
        let mut spectrum = vec![0.0f32; channels * size];
        let collapse_masks = vec![0u8; mode.num_ebands * channels];
        let log_e = vec![5.0f32; mode.num_ebands * channels];
        let prev1 = vec![0.0f32; mode.num_ebands * channels];
        let prev2 = vec![0.0f32; mode.num_ebands * channels];
        let pulses = vec![0i32; mode.num_ebands];

        anti_collapse(
            &mode,
            &mut spectrum,
            &collapse_masks,
            lm,
            channels,
            size,
            0,
            mode.num_ebands,
            &log_e,
            &prev1,
            &prev2,
            &pulses,
            0xDEAD_BEEF,
            false,
            0,
        );

        let band_width = usize::try_from(e_bands[1] - e_bands[0]).unwrap();
        let samples = band_width << lm;
        let energy: f32 = spectrum[..samples].iter().map(|v| v * v).sum();

        assert!(spectrum[..samples].iter().any(|v| *v != 0.0));
        assert!(energy > 0.0);
        assert!((energy - 1.0).abs() <= 1e-3, "renormalised energy {energy}");
    }
}
