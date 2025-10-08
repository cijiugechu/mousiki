#![allow(dead_code)]

//! Linear prediction helpers mirrored from `celt/celt_lpc.c`.
//!
//! The Levinson-Durbin recursion exposed as `_celt_lpc()` in the C
//! implementation has minimal dependencies on the rest of the encoder.  This
//! makes it a convenient building block to port early, as later modules such as
//! the pitch analysis and postfilter reuse it.

use alloc::borrow::Cow;

use crate::celt::math::frac_div32;
use crate::celt::pitch::celt_pitch_xcorr;
use crate::celt::types::{CeltCoef, OpusVal16, OpusVal32};

/// Computes LPC coefficients from the autocorrelation sequence using the
/// Levinson-Durbin recursion.
///
/// Mirrors the float build of `_celt_lpc()` from `celt/celt_lpc.c`. The caller
/// provides the first `order + 1` autocorrelation entries in `ac`, with
/// `ac[0]` containing the zero-lag value. The resulting predictor coefficients
/// are written to `lpc`.
///
/// The routine aborts early when the prediction error falls below
/// `0.001 * ac[0]`, matching the conservative bailout used by the reference
/// implementation to avoid unstable filters when the signal energy becomes
/// negligible.
pub(crate) fn celt_lpc(lpc: &mut [OpusVal16], ac: &[OpusVal32]) {
    let order = lpc.len();
    assert!(
        ac.len() > order,
        "autocorrelation must provide order + 1 samples"
    );

    for coeff in lpc.iter_mut() {
        *coeff = 0.0;
    }

    if order == 0 {
        return;
    }

    let ac0 = ac[0];
    if ac0 <= 1e-10 {
        return;
    }

    let mut error = ac0;

    for i in 0..order {
        let mut rr = 0.0;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];

        let reflection = -frac_div32(rr, error);
        lpc[i] = reflection;

        let half = (i + 1) >> 1;
        for j in 0..half {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + reflection * tmp2;
            lpc[i - 1 - j] = tmp2 + reflection * tmp1;
        }

        error -= (reflection * reflection) * error;
        if error <= 0.001 * ac0 {
            break;
        }
    }
}

/// Applies a causal FIR filter to the input sequence.
///
/// Mirrors the behaviour of `celt_fir_c()` from `celt/celt_lpc.c` for the float
/// build where the optimisation-specific helpers collapse to straightforward
/// scalar operations. The `x` slice must contain the `ord` samples of history
/// followed by `N` new samples, where `N` matches the length of `y`.
///
/// The implementation intentionally keeps the history layout identical to the C
/// code: `x[ord + i]` corresponds to the current sample while `x[ord + i - 1 - j]`
/// exposes the `j`-th past value.
pub(crate) fn celt_fir(x: &[OpusVal16], num: &[OpusVal16], y: &mut [OpusVal16]) {
    debug_assert!(!core::ptr::eq(x.as_ptr(), y.as_ptr()));

    let ord = num.len();
    let n = y.len();
    assert!(x.len() >= ord + n, "input must provide ord history samples");

    for i in 0..n {
        let mut acc = x[ord + i];
        for (tap, coeff) in num.iter().enumerate() {
            acc += coeff * x[ord + i - 1 - tap];
        }
        y[i] = acc;
    }
}

/// Applies an all-pole IIR filter and updates the provided memory buffer.
///
/// Mirrors the small-footprint implementation of `celt_iir()` in
/// `celt/celt_lpc.c` for the float build. The denominator coefficients in
/// `den` encode the autoregressive part of the filter and the `mem` slice stores
/// the past outputs (`y[n-1]`, `y[n-2]`, ...).
pub(crate) fn celt_iir(
    x: &[OpusVal32],
    den: &[OpusVal16],
    y: &mut [OpusVal32],
    mem: &mut [OpusVal16],
) {
    let ord = den.len();
    assert_eq!(mem.len(), ord, "IIR memory must match denominator order");
    assert_eq!(
        x.len(),
        y.len(),
        "input and output must have the same length"
    );

    if ord == 0 {
        y.copy_from_slice(x);
        return;
    }

    for (input, output) in x.iter().zip(y.iter_mut()) {
        let mut acc = *input;
        for (coeff, state) in den.iter().zip(mem.iter()) {
            acc -= coeff * *state;
        }

        *output = acc;

        for idx in (1..ord).rev() {
            mem[idx] = mem[idx - 1];
        }
        mem[0] = acc as OpusVal16;
    }
}

/// Computes the autocorrelation sequence of the input signal.
///
/// Mirrors the float configuration of `_celt_autocorr()` from `celt/celt_lpc.c`.
/// The routine optionally applies a symmetric analysis window spanning
/// `overlap` samples at the start and end of `x` before evaluating the
/// autocorrelation up to `lag`.
///
/// The returned shift value is specific to the fixed-point build in the
/// reference implementation; for the float configuration it always resolves to
/// zero and is provided only for API compatibility with future ports that may
/// consume the output.
pub(crate) fn celt_autocorr(
    x: &[OpusVal16],
    ac: &mut [OpusVal32],
    window: Option<&[CeltCoef]>,
    overlap: usize,
    lag: usize,
    arch: i32,
) -> i32 {
    assert!(
        !x.is_empty(),
        "input signal must contain at least one sample"
    );
    assert!(
        ac.len() > lag,
        "autocorrelation buffer must hold lag + 1 values"
    );
    assert!(lag <= x.len(), "lag must not exceed the input length");
    assert!(
        overlap <= x.len(),
        "window overlap cannot exceed the input length"
    );

    let n = x.len();
    let fast_n = n - lag;

    let xptr_cow = if overlap == 0 {
        Cow::Borrowed(x)
    } else {
        let window = window.expect("window coefficients required when overlap > 0");
        assert!(
            window.len() >= overlap,
            "window must provide at least overlap coefficients"
        );

        let mut buffer = x.to_vec();
        for i in 0..overlap {
            let w = window[i];
            buffer[i] *= w;
            let tail = n - i - 1;
            buffer[tail] *= w;
        }

        Cow::Owned(buffer)
    };
    let xptr = xptr_cow.as_ref();

    let _ = arch;

    celt_pitch_xcorr(xptr, xptr, fast_n, lag + 1, ac);

    for (k, slot) in ac.iter_mut().enumerate().take(lag + 1) {
        let mut d = 0.0;
        for i in k + fast_n..n {
            d += xptr[i] * xptr[i - k];
        }
        *slot += d;
    }

    0
}

#[cfg(test)]
mod tests {
    use super::{celt_autocorr, celt_fir, celt_iir, celt_lpc};
    use crate::celt::types::{OpusVal16, OpusVal32};
    use alloc::vec;
    use alloc::vec::Vec;

    fn reference_lpc(ac: &[f64], order: usize) -> Vec<f64> {
        let mut lpc = vec![0.0f64; order];
        if order == 0 {
            return lpc;
        }

        let ac0 = ac[0];
        if ac0 <= 1e-10 {
            return lpc;
        }

        let mut error = ac0;
        for i in 0..order {
            let mut rr = 0.0f64;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];

            let reflection = -rr / error;
            lpc[i] = reflection;

            let half = (i + 1) >> 1;
            for j in 0..half {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + reflection * tmp2;
                lpc[i - 1 - j] = tmp2 + reflection * tmp1;
            }

            error -= (reflection * reflection) * error;
            if error <= 0.001 * ac0 {
                break;
            }
        }

        lpc
    }

    fn generate_autocorrelation(order: usize, len: usize) -> Vec<f64> {
        let mut seed = 0x1234_5678u32;
        let mut signal = vec![0.0f64; len + order];

        for n in order..signal.len() {
            let noise = {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let val = f64::from(seed >> 1) / f64::from(u32::MAX >> 1);
                (val - 0.5) * 0.1
            };

            let mut value = noise;
            for k in 1..=order {
                value += 0.3f64.powi(k as i32) * signal[n - k];
            }
            signal[n] = value;
        }

        let mut ac = vec![0.0f64; order + 1];
        for lag in 0..=order {
            let mut sum = 0.0f64;
            for n in lag..signal.len() {
                sum += signal[n] * signal[n - lag];
            }
            ac[lag] = sum;
        }
        ac
    }

    fn reference_autocorr(signal: &[OpusVal16], lag: usize) -> Vec<OpusVal32> {
        let n = signal.len();
        (0..=lag)
            .map(|k| {
                let mut acc = 0.0f32;
                for i in 0..n.saturating_sub(k) {
                    acc += signal[i] * signal[i + k];
                }
                acc
            })
            .collect()
    }

    #[test]
    fn autocorr_matches_reference_without_window() {
        let n = 32;
        let lag = 6;
        let mut seed = 0x2468_acdfu32;
        let mut signal = vec![0.0f32; n];
        for sample in &mut signal {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = ((seed >> 9) as f32 / (u32::MAX >> 1) as f32) - 1.0;
        }

        let mut ac = vec![0.0f32; lag + 1];
        let shift = celt_autocorr(&signal, &mut ac, None, 0, lag, 0);

        let expected = reference_autocorr(&signal, lag);
        for (lhs, rhs) in ac.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
        assert_eq!(shift, 0);
    }

    #[test]
    fn autocorr_applies_window_symmetrically() {
        let n = 24;
        let lag = 4;
        let overlap = 6;
        let mut signal = vec![0.0f32; n];
        for (idx, sample) in signal.iter_mut().enumerate() {
            *sample = (idx as f32 * 0.1).sin();
        }

        let mut window = vec![0.0f32; overlap];
        for (i, slot) in window.iter_mut().enumerate() {
            let phase = core::f32::consts::PI * (i as f32 + 0.5) / overlap as f32;
            *slot = phase.sin();
        }

        let mut windowed = signal.clone();
        for i in 0..overlap {
            windowed[i] *= window[i];
            let tail = n - i - 1;
            windowed[tail] *= window[i];
        }

        let mut ac = vec![0.0f32; lag + 1];
        celt_autocorr(&signal, &mut ac, Some(window.as_slice()), overlap, lag, 0);

        let expected = reference_autocorr(&windowed, lag);
        for (lhs, rhs) in ac.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn lpc_matches_reference_for_randomish_signal() {
        let order = 8;
        let ac = generate_autocorrelation(order, 128);
        let expected = reference_lpc(&ac, order);

        let mut coeffs = vec![0.0f32; order];
        let ac_f32: Vec<OpusVal32> = ac.iter().map(|&v| v as OpusVal32).collect();
        celt_lpc(&mut coeffs, &ac_f32);

        for (got, want) in coeffs.iter().zip(expected.iter()) {
            assert!(
                (f64::from(*got) - *want).abs() <= 1e-5,
                "got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lpc_leaves_coefficients_zero_for_low_energy() {
        let mut coeffs = [1.0f32, -2.0, 3.0];
        let ac: [OpusVal32; 4] = [1e-12, 0.0, 0.0, 0.0];
        celt_lpc(&mut coeffs, &ac);
        assert_eq!(coeffs, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn lpc_handles_zero_order() {
        let mut coeffs: [OpusVal16; 0] = [];
        let ac: [OpusVal32; 1] = [1.0];
        celt_lpc(&mut coeffs, &ac);
    }

    #[test]
    fn fir_matches_reference_response() {
        let ord = 4;
        let taps = [0.2f32, -0.15, 0.05, 0.1];
        let history = [0.5f32, -0.25, 0.1, 0.0];
        let input = [0.3f32, -0.4, 0.2, -0.1, 0.05, 0.6];

        let mut buffer = history.to_vec();
        buffer.extend_from_slice(&input);

        let mut output = vec![0.0f32; input.len()];
        celt_fir(&buffer, &taps, &mut output);

        for (i, got) in output.iter().enumerate() {
            let mut expected = buffer[ord + i];
            for k in 0..ord {
                expected += taps[k] * buffer[ord + i - 1 - k];
            }
            assert!(
                (expected - *got).abs() <= 1e-6,
                "idx {i}: got {got}, want {expected}"
            );
        }
    }

    #[test]
    fn iir_matches_reference_response() {
        let den = [0.4f32, -0.2, 0.1];
        let input = [0.5f32, 0.1, -0.3, 0.2, 0.0, -0.1, 0.4];

        let mut mem = vec![0.0f32; den.len()];
        let mut output = vec![0.0f32; input.len()];
        celt_iir(&input, &den, &mut output, &mut mem);

        let mut ref_mem = vec![0.0f32; den.len()];
        let mut expected = vec![0.0f32; input.len()];
        for (idx, (&x, y)) in input.iter().zip(expected.iter_mut()).enumerate() {
            let mut acc = x;
            for (coeff, state) in den.iter().zip(ref_mem.iter()) {
                acc -= coeff * *state;
            }
            *y = acc;
            for j in (1..den.len()).rev() {
                ref_mem[j] = ref_mem[j - 1];
            }
            if !den.is_empty() {
                ref_mem[0] = acc;
            }
            assert!(
                (output[idx] - acc).abs() <= 1e-6,
                "idx {idx}: got {}, want {acc}",
                output[idx]
            );
        }

        assert_eq!(mem.len(), ref_mem.len());
        for (idx, (got, want)) in mem.iter().zip(ref_mem.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6,
                "mem {idx}: got {got}, want {want}"
            );
        }
    }
}
