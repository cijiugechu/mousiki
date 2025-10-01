#![allow(dead_code)]

//! Pitch analysis helpers translated from `celt/pitch.c`.
//!
//! The original implementation provides a collection of small math routines
//! that can be ported in isolation before the full pitch search is
//! reimplemented.  These helpers expose the same behaviour for the float build
//! of CELT while leveraging Rust's slice-based APIs for memory safety.

use crate::celt::math::celt_sqrt;
use crate::celt::types::{CeltSig, OpusVal16, OpusVal32};
use crate::celt::{celt_autocorr, celt_lpc};

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
        celt_fir5, celt_inner_prod, celt_pitch_xcorr, compute_pitch_gain, dual_inner_prod,
        pitch_downsample,
    };
    use crate::celt::math::celt_sqrt;
    use crate::celt::types::{CeltSig, OpusVal16};
    use crate::celt::{celt_autocorr, celt_lpc};
    use alloc::vec;
    use alloc::vec::Vec;

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
}
