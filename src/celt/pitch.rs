#![allow(dead_code)]

//! Pitch analysis helpers translated from `celt/pitch.c`.
//!
//! The original implementation provides a collection of small math routines
//! that can be ported in isolation before the full pitch search is
//! reimplemented.  These helpers expose the same behaviour for the float build
//! of CELT while leveraging Rust's slice-based APIs for memory safety.

use crate::celt::math::{celt_sqrt, frac_div32};
use crate::celt::types::{CeltSig, OpusVal16, OpusVal32};
use crate::celt::{celt_autocorr, celt_lpc, celt_udiv};
use alloc::vec;
use core::cmp::min;

/// Selects the two most promising pitch lags based on normalised correlation.
///
/// Ports the float variant of `find_best_pitch()` from `celt/pitch.c`.  The
/// routine scans the coarse cross-correlation vector produced by
/// [`celt_pitch_xcorr`] and maintains the two candidates with the largest
/// energy-normalised scores.  The function mirrors the C implementation by
/// comparing cross-multiplied numerators and denominators instead of dividing
/// the correlation energy directly, which preserves the ordering even when the
/// intermediate values grow large.
pub(crate) fn find_best_pitch(
    xcorr: &[OpusVal32],
    y: &[OpusVal16],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [i32; 2],
) {
    assert!(
        xcorr.len() >= max_pitch,
        "xcorr must contain max_pitch elements"
    );
    assert!(
        y.len() >= len + max_pitch,
        "y must contain len + max_pitch samples to slide the energy window",
    );

    let mut syy: OpusVal32 = 1.0;
    for &sample in &y[..len] {
        syy += sample * sample;
    }

    let mut best_num = [-1.0, -1.0];
    let mut best_den = [0.0, 0.0];
    best_pitch[0] = 0;
    best_pitch[1] = if max_pitch > 1 { 1 } else { 0 };

    for (i, &corr) in xcorr.iter().enumerate().take(max_pitch) {
        if corr > 0.0 {
            let mut corr16 = corr;
            // Matches the float implementation, which rescales the correlation
            // before squaring to avoid intermediate infinities.  The constant
            // factor cancels out when comparing the normalised scores.
            corr16 *= 1e-12;
            let num = corr16 * corr16;

            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i as i32;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i as i32;
                }
            }
        }

        let entering = y[i + len];
        let leaving = y[i];
        syy += entering * entering - leaving * leaving;
        if syy < 1.0 {
            syy = 1.0;
        }
    }
}

/// Computes the inner product between two input vectors.
///
/// Mirrors the behaviour of `celt_inner_prod_c()` from `celt/pitch.c` when the
/// codec is compiled in float mode.  The function asserts that the inputs share
/// the same length and returns the accumulated dot product as a 32-bit float.
pub(crate) fn celt_inner_prod(x: &[OpusVal16], y: &[OpusVal16]) -> OpusVal32 {
    assert_eq!(
        x.len(),
        y.len(),
        "vectors provided to celt_inner_prod must have the same length",
    );

    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| a * b)
        .sum::<OpusVal32>()
}

/// Computes two inner products between the same `x` vector and two targets.
///
/// Ports the scalar `dual_inner_prod_c()` helper from `celt/pitch.c` for the
/// float configuration.  The function evaluates the dot products `(x · y0)` and
/// `(x · y1)` in a single pass over the data, returning the pair as a tuple.
///
/// Callers must supply slices of identical length; this mirrors the original C
/// signature where the routine expects `N` samples for each input.
pub(crate) fn dual_inner_prod(
    x: &[OpusVal16],
    y0: &[OpusVal16],
    y1: &[OpusVal16],
) -> (OpusVal32, OpusVal32) {
    assert!(
        x.len() == y0.len() && x.len() == y1.len(),
        "dual_inner_prod inputs must have the same length"
    );

    let mut xy0 = 0.0;
    let mut xy1 = 0.0;

    for ((&a, &b0), &b1) in x.iter().zip(y0.iter()).zip(y1.iter()) {
        xy0 += a * b0;
        xy1 += a * b1;
    }

    (xy0, xy1)
}

/// Computes the normalised open-loop pitch gain.
///
/// Mirrors the float version of `compute_pitch_gain()` in `celt/pitch.c`, which
/// scales the correlation `xy` by the geometric mean of `xx` and `yy`.  The C
/// routine adds a bias of `1` under the square root to avoid division by zero;
/// the Rust port retains this behaviour to match the reference implementation.
#[inline]
pub(crate) fn compute_pitch_gain(xy: OpusVal32, xx: OpusVal32, yy: OpusVal32) -> OpusVal16 {
    // The float build uses `xy / celt_sqrt(1 + xx * yy)`.
    (xy / celt_sqrt(1.0 + xx * yy)) as OpusVal16
}

/// Computes the cross-correlation between the target vector and delayed copies.
///
/// Mirrors the scalar `celt_pitch_xcorr_c()` helper from `celt/pitch.c` when the
/// codec is built for floating-point targets. The routine fills `xcorr` with the
/// inner products between `x` and each `len`-sample window of `y`, starting at
/// delays `0..max_pitch-1`.
pub(crate) fn celt_pitch_xcorr(
    x: &[OpusVal16],
    y: &[OpusVal16],
    len: usize,
    max_pitch: usize,
    xcorr: &mut [OpusVal32],
) {
    assert!(x.len() >= len, "input x must provide at least len samples");
    assert!(
        y.len() >= len + max_pitch.saturating_sub(1),
        "input y must provide len + max_pitch - 1 samples"
    );
    assert!(
        xcorr.len() >= max_pitch,
        "output buffer must store max_pitch correlation values"
    );

    let x_head = &x[..len];
    for (delay, slot) in xcorr.iter_mut().enumerate().take(max_pitch) {
        let y_window = &y[delay..delay + len];
        *slot = x_head
            .iter()
            .zip(y_window.iter())
            .map(|(&a, &b)| a * b)
            .sum();
    }
}

/// Performs the coarse-to-fine pitch search used by the encoder analysis paths.
///
/// This mirrors the float implementation of `pitch_search()` from
/// `celt/pitch.c`. The routine operates on downsampled input buffers,
/// performing a decimated sweep followed by a refined search around the best
/// candidates. The final pitch lag is pseudo-interpolated using the
/// neighbouring correlations to match the C reference behaviour.
pub(crate) fn pitch_search(
    x_lp: &[OpusVal16],
    y: &[OpusVal16],
    len: usize,
    max_pitch: usize,
    _arch: i32,
) -> i32 {
    assert!(len > 0, "pitch_search requires a non-empty target length");
    assert!(
        max_pitch > 0,
        "pitch_search requires a positive search span"
    );
    assert!(x_lp.len() >= len, "x_lp must provide len samples");

    let lag = len + max_pitch;
    assert!(
        y.len() >= lag,
        "y must contain len + max_pitch samples for the search window",
    );

    let len_quarter = len >> 2;
    let lag_quarter = lag >> 2;
    let max_pitch_half = max_pitch >> 1;
    let max_pitch_quarter = max_pitch >> 2;

    let mut best_pitch = [0i32, 0i32];

    if len_quarter > 0 && max_pitch_quarter > 0 {
        let mut x_lp4 = vec![0.0; len_quarter];
        for (j, slot) in x_lp4.iter_mut().enumerate() {
            *slot = x_lp[2 * j];
        }

        let mut y_lp4 = vec![0.0; lag_quarter];
        for (j, slot) in y_lp4.iter_mut().enumerate() {
            *slot = y[2 * j];
        }

        let mut xcorr = vec![0.0; max_pitch_quarter];
        celt_pitch_xcorr(&x_lp4, &y_lp4, len_quarter, max_pitch_quarter, &mut xcorr);

        let y_needed = min(y_lp4.len(), len_quarter + max_pitch_quarter);
        find_best_pitch(
            &xcorr,
            &y_lp4[..y_needed],
            len_quarter,
            max_pitch_quarter,
            &mut best_pitch,
        );
    }

    let mut xcorr = vec![0.0; max_pitch_half.max(1)];

    if max_pitch_half > 0 {
        let len_half = len >> 1;
        if len_half > 0 {
            for i in 0..max_pitch_half {
                if (i as i32 - 2 * best_pitch[0]).abs() > 2
                    && (i as i32 - 2 * best_pitch[1]).abs() > 2
                {
                    continue;
                }
                let start = i;
                let end = start + len_half;
                if end > y.len() {
                    break;
                }
                let sum: OpusVal32 = x_lp[..len_half]
                    .iter()
                    .zip(&y[start..end])
                    .map(|(&a, &b)| a * b)
                    .sum();
                xcorr[i] = sum.max(-1.0);
            }

            let y_needed = min(y.len(), len_half + max_pitch_half);
            find_best_pitch(
                &xcorr[..max_pitch_half],
                &y[..y_needed],
                len_half,
                max_pitch_half,
                &mut best_pitch,
            );

            if best_pitch[0] > 0 && (best_pitch[0] as usize) < max_pitch_half - 1 {
                let a = xcorr[(best_pitch[0] - 1) as usize];
                let b = xcorr[best_pitch[0] as usize];
                let c = xcorr[(best_pitch[0] + 1) as usize];
                let mut offset = 0;
                if (c - a) > 0.7 * (b - a) {
                    offset = 1;
                } else if (a - c) > 0.7 * (b - c) {
                    offset = -1;
                }
                return 2 * best_pitch[0] - offset;
            }
        }
    }

    2 * best_pitch[0]
}

const SECOND_CHECK: [i32; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

fn window<'a>(data: &'a [OpusVal16], center: usize, offset: isize, len: usize) -> &'a [OpusVal16] {
    let start = center as isize + offset;
    assert!(start >= 0, "window would start before the buffer");
    let start = start as usize;
    let end = start + len;
    assert!(end <= data.len(), "window extends beyond the buffer");
    &data[start..end]
}

/// Suppresses spurious pitch-doubling detections.
///
/// Mirrors the float variant of `remove_doubling()` from `celt/pitch.c`. The
/// routine evaluates nearby subharmonics of the detected pitch and returns an
/// adjusted lag alongside the updated harmonic gain.
pub(crate) fn remove_doubling(
    x: &[OpusVal16],
    maxperiod: usize,
    minperiod: usize,
    n: usize,
    t0: &mut i32,
    prev_period: i32,
    prev_gain: OpusVal16,
    _arch: i32,
) -> OpusVal16 {
    assert!(maxperiod > 0, "maxperiod must be positive");
    assert!(minperiod > 0, "minperiod must be positive");
    assert!(n > 0, "window size must be positive");
    assert!(
        x.len() >= maxperiod + n,
        "x must contain maxperiod + n samples",
    );

    let minperiod0 = minperiod as i32;
    let maxperiod_half = maxperiod >> 1;
    let minperiod_half = minperiod >> 1;
    let t0_half = (*t0 >> 1).clamp(0, maxperiod_half.saturating_sub(1) as i32);
    let prev_period_half = prev_period >> 1;
    let n_half = n >> 1;

    if maxperiod_half <= 1 || n_half == 0 {
        *t0 = (*t0).max(minperiod0);
        return prev_gain;
    }

    let center = maxperiod_half;
    assert!(
        center + n_half <= x.len(),
        "insufficient samples for windowed view"
    );

    let x_center = window(x, center, 0, n_half);
    let x_t0 = window(x, center, -(t0_half as isize), n_half);
    let (xx, xy) = dual_inner_prod(x_center, x_center, x_t0);

    let mut yy_lookup = vec![0.0; maxperiod_half + 1];
    yy_lookup[0] = xx;
    let mut yy = xx;

    for i in 1..=maxperiod_half {
        let prev_sample = x[center - i];
        let enter_sample = x[center + n_half - i];
        yy += prev_sample * prev_sample - enter_sample * enter_sample;
        yy_lookup[i] = yy.max(0.0);
    }

    yy = yy_lookup[t0_half as usize];
    let mut best_xy = xy;
    let mut best_yy = yy;
    let mut g = compute_pitch_gain(xy, xx, yy);
    let g0 = g;
    let max_allowed = maxperiod_half.saturating_sub(1) as i32;
    let mut t = if max_allowed >= 1 {
        t0_half.clamp(1, max_allowed)
    } else {
        0
    };

    for k in 2..=15 {
        let t1 = celt_udiv((2 * t0_half + k) as u32, (2 * k) as u32) as i32;
        if t1 < minperiod_half as i32 {
            break;
        }
        if t1 as usize > maxperiod_half {
            continue;
        }
        let t1b = if k == 2 {
            if t1 + t0_half > maxperiod_half as i32 {
                t0_half
            } else {
                t0_half + t1
            }
        } else {
            let check = SECOND_CHECK[k as usize];
            celt_udiv((2 * check * t0_half + k) as u32, (2 * k) as u32) as i32
        };
        if t1b as usize > maxperiod_half {
            continue;
        }

        let x_t1 = window(x, center, -(t1 as isize), n_half);
        let x_t1b = window(x, center, -(t1b as isize), n_half);
        let (mut xy1, xy2) = dual_inner_prod(x_center, x_t1, x_t1b);
        xy1 = 0.5 * (xy1 + xy2);
        let yy1 = 0.5 * (yy_lookup[t1 as usize] + yy_lookup[t1b as usize]);
        let g1 = compute_pitch_gain(xy1, xx, yy1);

        let diff = (t1 - prev_period_half).abs();
        let cont = if diff <= 1 {
            prev_gain
        } else if diff <= 2 && 5 * ((k * k) as i32) < t0_half {
            0.5 * prev_gain
        } else {
            0.0
        };

        let mut thresh = (0.7 * g0 - cont).max(0.3);
        if t1 < 3 * minperiod_half as i32 {
            thresh = (0.85 * g0 - cont).max(0.4);
        } else if t1 < 2 * minperiod_half as i32 {
            thresh = (0.9 * g0 - cont).max(0.5);
        }

        if g1 > thresh {
            best_xy = xy1;
            best_yy = yy1;
            if max_allowed >= 1 {
                t = t1.clamp(1, max_allowed);
            } else {
                t = 0;
            }
            g = g1;
        }
    }

    best_xy = best_xy.max(0.0);
    let mut pg = if best_yy <= best_xy {
        1.0
    } else {
        frac_div32(best_xy, best_yy + 1.0)
    };

    let mut xcorr = [0.0; 3];
    for (k, slot) in xcorr.iter_mut().enumerate() {
        let lag = t + k as i32 - 1;
        let windowed = window(x, center, -(lag as isize), n_half);
        *slot = celt_inner_prod(x_center, windowed);
    }

    let mut offset = 0;
    if (xcorr[2] - xcorr[0]) > 0.7 * (xcorr[1] - xcorr[0]) {
        offset = 1;
    } else if (xcorr[0] - xcorr[2]) > 0.7 * (xcorr[1] - xcorr[2]) {
        offset = -1;
    }

    if pg > g {
        pg = g;
    }

    let updated = 2 * t - offset;
    *t0 = updated.max(minperiod0);

    pg
}

fn celt_fir5(x: &mut [OpusVal16], num: &[OpusVal16; 5]) {
    let [num0, num1, num2, num3, num4] = *num;
    let mut mem0 = 0.0;
    let mut mem1 = 0.0;
    let mut mem2 = 0.0;
    let mut mem3 = 0.0;
    let mut mem4 = 0.0;

    for sample in x.iter_mut() {
        let current = *sample;
        let sum = current + num0 * mem0 + num1 * mem1 + num2 * mem2 + num3 * mem3 + num4 * mem4;

        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = current;

        *sample = sum;
    }
}

/// Downsamples the input channels to a mono low-pass signal used by the pitch search.
///
/// Mirrors the float build of `pitch_downsample()` in `celt/pitch.c`. The routine
/// averages pairs of input samples across one or two channels, applies LPC-based
/// noise shaping, and stores the downsampled result in `x_lp`.
pub(crate) fn pitch_downsample(x: &[&[CeltSig]], x_lp: &mut [OpusVal16], len: usize, arch: i32) {
    assert!(!x.is_empty(), "at least one channel is required");
    assert!(
        x.len() <= 2,
        "pitch_downsample supports at most two channels"
    );
    assert!(
        len >= 2,
        "pitch_downsample requires at least two input samples"
    );

    for (idx, channel) in x.iter().enumerate() {
        assert!(
            channel.len() >= len,
            "channel {idx} must provide at least len samples",
        );
    }

    let half_len = len / 2;
    assert!(
        x_lp.len() >= half_len,
        "output buffer must contain len / 2 samples"
    );

    if half_len == 0 {
        return;
    }

    let x_lp = &mut x_lp[..half_len];
    x_lp.fill(0.0);

    for channel in x {
        x_lp[0] += 0.25 * channel[1] + 0.5 * channel[0];
        for i in 1..half_len {
            let base = 2 * i;
            x_lp[i] += 0.25 * channel[base - 1] + 0.5 * channel[base] + 0.25 * channel[base + 1];
        }
    }

    let mut ac = [0.0; 5];
    celt_autocorr(x_lp, &mut ac, None, 0, 4, arch);

    ac[0] *= 1.0001;
    for i in 1..=4 {
        let coeff = 0.008 * i as f32;
        ac[i] -= ac[i] * coeff * coeff;
    }

    let mut lpc = [0.0; 4];
    celt_lpc(&mut lpc, &ac);

    let mut tmp = 1.0;
    for coeff in &mut lpc {
        tmp *= 0.9;
        *coeff *= tmp;
    }

    let c1 = 0.8;
    let lpc2 = [
        lpc[0] + 0.8,
        lpc[1] + c1 * lpc[0],
        lpc[2] + c1 * lpc[1],
        lpc[3] + c1 * lpc[2],
        c1 * lpc[3],
    ];

    celt_fir5(x_lp, &lpc2);
}

#[cfg(test)]
mod tests {
    use super::{
        SECOND_CHECK, celt_fir5, celt_inner_prod, celt_pitch_xcorr, compute_pitch_gain,
        dual_inner_prod, find_best_pitch, pitch_downsample, pitch_search, remove_doubling,
    };
    use crate::celt::math::celt_sqrt;
    use crate::celt::types::{CeltSig, OpusVal16, OpusVal32};
    use crate::celt::{celt_autocorr, celt_lpc, celt_udiv, frac_div32};
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::PI;

    fn generate_sequence(len: usize, seed: u32) -> Vec<OpusVal16> {
        let mut state = seed;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let val = ((state >> 8) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            data.push(val as OpusVal16);
        }
        data
    }

    #[test]
    fn inner_product_matches_reference() {
        let x = generate_sequence(64, 0x1234_5678);
        let y = generate_sequence(64, 0x8765_4321);

        let expected: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        let result = celt_inner_prod(&x, &y);

        assert!((expected - result).abs() < 1e-6);
    }

    #[test]
    fn dual_inner_product_matches_individual_computations() {
        let x = generate_sequence(48, 0x4242_4242);
        let y0 = generate_sequence(48, 0x1357_9bdf);
        let y1 = generate_sequence(48, 0x0246_8ace);

        let (dot0, dot1) = dual_inner_prod(&x, &y0, &y1);
        let expected0: f32 = x.iter().zip(y0.iter()).map(|(&a, &b)| a * b).sum();
        let expected1: f32 = x.iter().zip(y1.iter()).map(|(&a, &b)| a * b).sum();

        assert!((dot0 - expected0).abs() < 1e-6);
        assert!((dot1 - expected1).abs() < 1e-6);
    }

    #[test]
    fn pitch_gain_matches_reference_formula() {
        let xy = 0.75f32;
        let xx = 0.5f32;
        let yy = 1.25f32;

        let expected = (xy / celt_sqrt(1.0 + xx * yy)) as OpusVal16;
        let gain = compute_pitch_gain(xy, xx, yy);

        assert!((expected - gain).abs() < 1e-6);
    }

    #[test]
    fn pitch_xcorr_matches_naive_cross_correlation() {
        let len = 16usize;
        let max_pitch = 8usize;

        let x = generate_sequence(len, 0x0f0f_0f0f);
        let y = generate_sequence(len + max_pitch, 0x1337_4242);

        let mut xcorr = vec![0.0f32; max_pitch];
        celt_pitch_xcorr(&x, &y, len, max_pitch, &mut xcorr);

        for delay in 0..max_pitch {
            let expected: f32 = x
                .iter()
                .zip(y[delay..delay + len].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            assert!(
                (expected - xcorr[delay]).abs() < 1e-6,
                "mismatch at delay {delay}: expected {expected}, got {}",
                xcorr[delay]
            );
        }
    }

    fn reference_find_best_pitch(
        xcorr: &[OpusVal32],
        y: &[OpusVal16],
        len: usize,
        max_pitch: usize,
    ) -> [i32; 2] {
        let mut syy = 1.0;
        for &sample in &y[..len] {
            syy += sample * sample;
        }

        let mut best_num = [-1.0, -1.0];
        let mut best_den = [0.0, 0.0];
        let mut best_pitch = [0, if max_pitch > 1 { 1 } else { 0 }];

        for i in 0..max_pitch {
            let corr = xcorr[i];
            if corr > 0.0 {
                let num = corr * corr;
                if num * best_den[1] > best_num[1] * syy {
                    if num * best_den[0] > best_num[0] * syy {
                        best_num[1] = best_num[0];
                        best_den[1] = best_den[0];
                        best_pitch[1] = best_pitch[0];
                        best_num[0] = num;
                        best_den[0] = syy;
                        best_pitch[0] = i as i32;
                    } else {
                        best_num[1] = num;
                        best_den[1] = syy;
                        best_pitch[1] = i as i32;
                    }
                }
            }

            let entering = y[i + len];
            let leaving = y[i];
            syy += entering * entering - leaving * leaving;
            if syy < 1.0 {
                syy = 1.0;
            }
        }

        best_pitch
    }

    #[test]
    fn find_best_pitch_matches_reference() {
        let len = 48usize;
        let max_pitch = 24usize;

        let x = generate_sequence(len, 0x1111_2222);
        let mut y = vec![0.0; len + max_pitch];
        let primary_lag = 7usize;
        let secondary_lag = 15usize;

        for i in 0..len {
            y[i + primary_lag] += x[i];
            y[i + secondary_lag] += 0.6 * x[i];
        }

        let mut xcorr = vec![0.0; max_pitch];
        celt_pitch_xcorr(&x, &y, len, max_pitch, &mut xcorr);

        let mut best = [0i32; 2];
        find_best_pitch(&xcorr, &y, len, max_pitch, &mut best);

        let expected = reference_find_best_pitch(&xcorr, &y, len, max_pitch);
        assert_eq!(best, expected);
    }

    fn reference_pitch_downsample(x: &[&[CeltSig]], len: usize, arch: i32) -> Vec<OpusVal16> {
        let half_len = len / 2;
        let mut downsampled = vec![0.0; half_len];
        if half_len == 0 {
            return downsampled;
        }

        for channel in x {
            downsampled[0] += 0.25 * channel[1] + 0.5 * channel[0];
            for i in 1..half_len {
                let base = 2 * i;
                downsampled[i] +=
                    0.25 * channel[base - 1] + 0.5 * channel[base] + 0.25 * channel[base + 1];
            }
        }

        let mut ac = [0.0; 5];
        celt_autocorr(&downsampled, &mut ac, None, 0, 4, arch);

        ac[0] *= 1.0001;
        for i in 1..=4 {
            let coeff = 0.008 * i as f32;
            ac[i] -= ac[i] * coeff * coeff;
        }

        let mut lpc = [0.0; 4];
        celt_lpc(&mut lpc, &ac);

        let mut tmp = 1.0;
        for coeff in &mut lpc {
            tmp *= 0.9;
            *coeff *= tmp;
        }

        let c1 = 0.8;
        let lpc2 = [
            lpc[0] + 0.8,
            lpc[1] + c1 * lpc[0],
            lpc[2] + c1 * lpc[1],
            lpc[3] + c1 * lpc[2],
            c1 * lpc[3],
        ];

        let mut filtered = downsampled;
        celt_fir5(&mut filtered, &lpc2);
        filtered
    }

    #[test]
    fn pitch_downsample_matches_reference_for_mono() {
        let len = 64;
        let input = generate_sequence(len, 0x5555_aaaa);
        let channels: [&[CeltSig]; 1] = [&input];

        let mut output = vec![0.0; len / 2];
        pitch_downsample(&channels, &mut output, len, 0);

        let expected = reference_pitch_downsample(&channels, len, 0);

        for (result, reference) in output.iter().zip(expected.iter()) {
            assert!((result - reference).abs() < 1e-6);
        }
    }

    #[test]
    fn pitch_downsample_matches_reference_for_stereo() {
        let len = 48;
        let left = generate_sequence(len, 0x1234_ffff);
        let right = generate_sequence(len, 0xabcd_0001);
        let channels: [&[CeltSig]; 2] = [&left, &right];

        let mut output = vec![0.0; len / 2];
        pitch_downsample(&channels, &mut output, len, 0);

        let expected = reference_pitch_downsample(&channels, len, 0);

        for (result, reference) in output.iter().zip(expected.iter()) {
            assert!((result - reference).abs() < 1e-6);
        }
    }

    fn reference_pitch_search(
        x_lp: &[OpusVal16],
        y: &[OpusVal16],
        len: usize,
        max_pitch: usize,
    ) -> i32 {
        let lag = len + max_pitch;
        let len_quarter = len >> 2;
        let lag_quarter = lag >> 2;
        let max_pitch_quarter = max_pitch >> 2;

        let mut best = [0, if max_pitch_quarter > 1 { 1 } else { 0 }];
        if len_quarter > 0 && max_pitch_quarter > 0 {
            let mut x_lp4 = vec![0.0; len_quarter];
            for (j, slot) in x_lp4.iter_mut().enumerate() {
                *slot = x_lp[2 * j];
            }

            let mut y_lp4 = vec![0.0; lag_quarter];
            for (j, slot) in y_lp4.iter_mut().enumerate() {
                *slot = y[2 * j];
            }

            let mut xcorr = vec![0.0; max_pitch_quarter];
            for i in 0..max_pitch_quarter {
                let mut sum = 0.0;
                for j in 0..len_quarter {
                    sum += x_lp4[j] * y_lp4[i + j];
                }
                xcorr[i] = sum;
            }

            find_best_pitch(
                &xcorr,
                &y_lp4[..len_quarter + max_pitch_quarter],
                len_quarter,
                max_pitch_quarter,
                &mut best,
            );
        }

        let max_pitch_half = max_pitch >> 1;
        if max_pitch_half == 0 {
            return 2 * best[0];
        }

        let len_half = len >> 1;
        let mut xcorr = vec![0.0; max_pitch_half];
        if len_half > 0 {
            for i in 0..max_pitch_half {
                if (i as i32 - 2 * best[0]).abs() > 2 && (i as i32 - 2 * best[1]).abs() > 2 {
                    continue;
                }
                let mut sum = 0.0;
                for j in 0..len_half {
                    sum += x_lp[j] * y[i + j];
                }
                xcorr[i] = sum.max(-1.0);
            }

            find_best_pitch(
                &xcorr,
                &y[..len_half + max_pitch_half],
                len_half,
                max_pitch_half,
                &mut best,
            );
        }

        let mut offset = 0;
        if best[0] > 0 && (best[0] as usize) < max_pitch_half - 1 {
            let a = xcorr[(best[0] - 1) as usize];
            let b = xcorr[best[0] as usize];
            let c = xcorr[(best[0] + 1) as usize];
            if (c - a) > 0.7 * (b - a) {
                offset = 1;
            } else if (a - c) > 0.7 * (b - c) {
                offset = -1;
            }
        }

        2 * best[0] - offset
    }

    fn reference_remove_doubling(
        x: &[OpusVal16],
        maxperiod: usize,
        minperiod: usize,
        n: usize,
        t0: &mut i32,
        prev_period: i32,
        prev_gain: OpusVal16,
    ) -> OpusVal16 {
        let minperiod0 = minperiod as i32;
        let maxperiod_half = maxperiod >> 1;
        let minperiod_half = minperiod >> 1;
        let t0_half = (*t0 >> 1).clamp(0, maxperiod_half.saturating_sub(1) as i32);
        let prev_period_half = prev_period >> 1;
        let n_half = n >> 1;

        if maxperiod_half <= 1 || n_half == 0 {
            *t0 = (*t0).max(minperiod0);
            return prev_gain;
        }

        let center = maxperiod_half;
        let x_center = &x[center..center + n_half];
        let x_t0 = &x[center - t0_half as usize..center - t0_half as usize + n_half];

        let mut xx = 0.0;
        let mut xy = 0.0;
        for j in 0..n_half {
            xx += x_center[j] * x_center[j];
            xy += x_center[j] * x_t0[j];
        }

        let mut yy_lookup = vec![0.0; maxperiod_half + 1];
        yy_lookup[0] = xx;
        let mut yy = xx;
        for i in 1..=maxperiod_half {
            let prev_sample = x[center - i];
            let enter_sample = x[center + n_half - i];
            yy += prev_sample * prev_sample - enter_sample * enter_sample;
            yy_lookup[i] = yy.max(0.0);
        }

        yy = yy_lookup[t0_half as usize];
        let mut best_xy = xy;
        let mut best_yy = yy;
        let g0 = compute_pitch_gain(xy, xx, yy);
        let mut g = g0;
        let max_allowed = maxperiod_half.saturating_sub(1) as i32;
        let mut t = if max_allowed >= 1 {
            t0_half.clamp(1, max_allowed)
        } else {
            0
        };

        for k in 2..=15 {
            let t1 = celt_udiv((2 * t0_half + k) as u32, (2 * k) as u32) as i32;
            if t1 < minperiod_half as i32 {
                break;
            }
            if t1 as usize > maxperiod_half {
                continue;
            }
            let t1b = if k == 2 {
                if t1 + t0_half > maxperiod_half as i32 {
                    t0_half
                } else {
                    t0_half + t1
                }
            } else {
                let check = SECOND_CHECK[k as usize];
                celt_udiv((2 * check * t0_half + k) as u32, (2 * k) as u32) as i32
            };
            if t1b as usize > maxperiod_half {
                continue;
            }

            let x_t1 = &x[center - t1 as usize..center - t1 as usize + n_half];
            let x_t1b = &x[center - t1b as usize..center - t1b as usize + n_half];
            let mut xy1 = 0.0;
            let mut xy2 = 0.0;
            for j in 0..n_half {
                xy1 += x_center[j] * x_t1[j];
                xy2 += x_center[j] * x_t1b[j];
            }
            xy1 = 0.5 * (xy1 + xy2);
            let yy1 = 0.5 * (yy_lookup[t1 as usize] + yy_lookup[t1b as usize]);
            let g1 = compute_pitch_gain(xy1, xx, yy1);

            let diff = (t1 - prev_period_half).abs();
            let cont = if diff <= 1 {
                prev_gain
            } else if diff <= 2 && 5 * ((k * k) as i32) < t0_half {
                0.5 * prev_gain
            } else {
                0.0
            };

            let mut thresh = (0.7 * g0 - cont).max(0.3);
            if t1 < 3 * minperiod_half as i32 {
                thresh = (0.85 * g0 - cont).max(0.4);
            } else if t1 < 2 * minperiod_half as i32 {
                thresh = (0.9 * g0 - cont).max(0.5);
            }

            if g1 > thresh {
                best_xy = xy1;
                best_yy = yy1;
                if max_allowed >= 1 {
                    t = t1.clamp(1, max_allowed);
                } else {
                    t = 0;
                }
                g = g1;
            }
        }

        best_xy = best_xy.max(0.0);
        let mut pg = if best_yy <= best_xy {
            1.0
        } else {
            frac_div32(best_xy, best_yy + 1.0)
        };

        let mut xcorr = [0.0; 3];
        for (k, slot) in xcorr.iter_mut().enumerate() {
            let lag = t + k as i32 - 1;
            let lag_usize = lag as usize;
            let x_lag = &x[center - lag_usize..center - lag_usize + n_half];
            let mut sum = 0.0;
            for j in 0..n_half {
                sum += x_center[j] * x_lag[j];
            }
            *slot = sum;
        }

        let mut offset = 0;
        if (xcorr[2] - xcorr[0]) > 0.7 * (xcorr[1] - xcorr[0]) {
            offset = 1;
        } else if (xcorr[0] - xcorr[2]) > 0.7 * (xcorr[1] - xcorr[2]) {
            offset = -1;
        }

        if pg > g {
            pg = g;
        }

        let updated = 2 * t - offset;
        *t0 = updated.max(minperiod0);
        pg
    }

    #[test]
    fn pitch_search_matches_reference() {
        let len = 96usize;
        let max_pitch = 48usize;
        let fundamental = 34usize;

        let mut x_lp = vec![0.0; len];
        for (i, sample) in x_lp.iter_mut().enumerate() {
            let phase = 2.0 * PI * (i as f32) / fundamental as f32;
            *sample = phase.sin();
        }

        let mut y = vec![0.0; len + max_pitch];
        for i in 0..len {
            y[i + fundamental] += x_lp[i];
        }
        for i in 0..len {
            y[i + 20] += 0.4 * x_lp[i];
        }

        let reference = reference_pitch_search(&x_lp, &y, len, max_pitch);
        let result = pitch_search(&x_lp, &y, len, max_pitch, 0);
        assert_eq!(result, reference);
    }

    #[test]
    fn remove_doubling_matches_reference() {
        let maxperiod = 120usize;
        let minperiod = 40usize;
        let n = 80usize;
        let fundamental = 60usize;

        let mut x = vec![0.0; maxperiod + n];
        for (i, sample) in x.iter_mut().enumerate() {
            let phase = 2.0 * PI * (i as f32) / fundamental as f32;
            *sample = phase.sin();
        }

        let mut t0_reference = (2 * fundamental) as i32;
        let mut t0_result = t0_reference;

        let expected = reference_remove_doubling(
            &x,
            maxperiod,
            minperiod,
            n,
            &mut t0_reference,
            fundamental as i32,
            0.8,
        );
        let gain = remove_doubling(
            &x,
            maxperiod,
            minperiod,
            n,
            &mut t0_result,
            fundamental as i32,
            0.8,
            0,
        );

        assert_eq!(t0_result, t0_reference);
        assert!((gain - expected).abs() < 1e-6);
    }
}
