//! Fixed-point autocorrelation helper used by the SILK analysis pipeline.
//!
//! Ports `silk_autocorr` from `silk/fixed/autocorr_FIX.c`, providing the
//! dynamically scaled energy measurements consumed by the LPC and noise-shaping
//! stages. The routine mirrors the reference implementation's headroom
//! tracking, returning the applied shift so downstream code can interpret the
//! 32-bit accumulators safely.

use alloc::vec::Vec;

use crate::celt::{celt_ilog2, ec_ilog};

/// Computes the autocorrelation vector of `input` up to `correlation_count`
/// taps.
///
/// The function returns the scaling exponent applied to the `results` slice,
/// matching the behaviour of the C implementation's `scale` out-parameter. The
/// caller must provide a `results` buffer whose length is at least
/// `min(input.len(), correlation_count)`.
pub fn autocorr(
    results: &mut [i32],
    input: &[i16],
    correlation_count: usize,
    arch: i32,
) -> i32 {
    let corr_count = correlation_count.min(input.len());
    assert!(
        results.len() >= corr_count,
        "results buffer must hold at least correlation_count elements"
    );

    for slot in &mut results[..corr_count] {
        *slot = 0;
    }

    if corr_count == 0 {
        return 0;
    }

    let n = input.len();
    let mut ac0 = 1 + ((n as i32) << 7);
    let mut idx = 0usize;

    if (n & 1) != 0 {
        ac0 += energy_term(input[0]);
        idx = 1;
    }

    while idx + 1 < n {
        ac0 += energy_term(input[idx]);
        ac0 += energy_term(input[idx + 1]);
        idx += 2;
    }

    let mut shift = celt_ilog2(ac0) - 30 + 10;
    shift /= 2;

    let scaled_storage = if shift > 0 {
        Some(
            input
                .iter()
                .map(|&sample| pshr32(i32::from(sample), shift) as i16)
                .collect::<Vec<_>>(),
        )
    } else {
        shift = 0;
        None
    };
    let signal = scaled_storage
        .as_deref()
        .unwrap_or(input);

    compute_autocorrelation(&mut results[..corr_count], signal);

    let mut total_shift = shift * 2;

    if total_shift <= 0 {
        let adjustment = 1i32
            .checked_shl((-total_shift) as u32)
            .expect("shift overflow when adjusting autocorrelation DC term");
        results[0] = results[0].wrapping_add(adjustment);
    }

    normalize_autocorrelation(&mut results[..corr_count], &mut total_shift);

    let _ = arch;

    total_shift
}

fn compute_autocorrelation(output: &mut [i32], signal: &[i16]) {
    for (lag, slot) in output.iter_mut().enumerate() {
        let mut acc = 0i64;
        for idx in lag..signal.len() {
            acc += i64::from(signal[idx]) * i64::from(signal[idx - lag]);
        }
        debug_assert!(
            (i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&acc),
            "autocorrelation overflowed 32-bit range"
        );
        *slot = acc as i32;
    }
}

fn normalize_autocorrelation(ac: &mut [i32], total_shift: &mut i32) {
    if ac.is_empty() {
        return;
    }

    debug_assert!(
        ac[0] > 0,
        "autocorrelation DC term must be strictly positive before normalisation"
    );

    if ac[0] < 268_435_456 {
        let shift2 = 29 - ec_ilog(ac[0] as u32);
        for value in ac.iter_mut() {
            *value <<= shift2;
        }
        *total_shift -= shift2;
    } else if ac[0] >= 536_870_912 {
        let mut shift2 = 1;
        if ac[0] >= 1_073_741_824 {
            shift2 += 1;
        }
        for value in ac.iter_mut() {
            *value >>= shift2;
        }
        *total_shift += shift2;
    }
}

#[inline]
fn energy_term(sample: i16) -> i32 {
    (i32::from(sample) * i32::from(sample)) >> 9
}

#[inline]
fn pshr32(value: i32, shift: i32) -> i32 {
    debug_assert!(shift > 0);
    let rounding = 1 << (shift - 1);
    (value + rounding) >> shift
}

#[cfg(test)]
mod tests {
    use super::autocorr;

    #[test]
    fn matches_reference_for_small_positive_sequence() {
        let input = [1, 2, 3, 4];
        let mut output = [0i32; 3];
        let taps = output.len();

        let scale = autocorr(&mut output, &input, taps, 0);

        assert_eq!(output, [520_093_696, 335_544_320, 184_549_376]);
        assert_eq!(scale, -24);
    }

    #[test]
    fn matches_reference_for_high_energy_signal() {
        let input = [
            30_000, -20_000, 15_000, -10_000, 5_000, -2_500, 1_250, -625,
        ];
        let mut output = [0i32; 5];
        let taps = output.len();

        let scale = autocorr(&mut output, &input, taps, 0);

        assert_eq!(
            output,
            [
                414_550_781,
                -279_101_563,
                189_453_125,
                -113_281_250,
                56_250_000
            ]
        );
        assert_eq!(scale, 2);
    }

    #[test]
    fn matches_reference_for_mixed_polarity_samples() {
        let input = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
        let mut output = [0i32; 4];
        let taps = output.len();

        let scale = autocorr(&mut output, &input, taps, 0);

        assert_eq!(
            output,
            [
                462_422_016,
                387_973_120,
                315_621_376,
                245_366_784
            ]
        );
        assert_eq!(scale, -20);
    }
}
