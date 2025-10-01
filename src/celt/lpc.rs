#![allow(dead_code)]

//! Linear prediction helpers mirrored from `celt/celt_lpc.c`.
//!
//! The Levinson-Durbin recursion exposed as `_celt_lpc()` in the C
//! implementation has minimal dependencies on the rest of the encoder.  This
//! makes it a convenient building block to port early, as later modules such as
//! the pitch analysis and postfilter reuse it.

use crate::celt::math::frac_div32;
use crate::celt::types::{OpusVal16, OpusVal32};

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
        ac.len() >= order + 1,
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

#[cfg(test)]
mod tests {
    use super::celt_lpc;
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
                let val = (seed >> 1) as f64 / (u32::MAX >> 1) as f64;
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
}
