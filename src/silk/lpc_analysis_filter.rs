//! Port of `silk/LPC_analysis_filter.c`.
//!
//! Applies the SILK MA prediction filter to an input signal, producing a
//! residual stream in Q0 domain while mirroring the overflow behaviour of the
//! reference fixed-point implementation.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

/// Apply the LPC analysis filter with Q12 prediction coefficients.
///
/// The function writes `len` samples into `output`, zeroing the first `order`
/// entries to match the C implementation's implicit zero state. The caller must
/// ensure that `order` is even, at least six taps, and no larger than `len`.
pub fn lpc_analysis_filter(
    output: &mut [i16],
    input: &[i16],
    coeffs_q12: &[i16],
    len: usize,
    order: usize,
) {
    assert!(len <= output.len(), "output buffer too small");
    assert!(len <= input.len(), "input buffer too small");
    assert!(order <= coeffs_q12.len(), "coefficient slice too short");
    assert!(order >= 6, "filter order must be at least six taps");
    assert!(order % 2 == 0, "filter order must be even");
    assert!(order <= len, "filter order cannot exceed len");

    for ix in order..len {
        let mut acc = i32::from(input[ix - 1]) * i32::from(coeffs_q12[0]);
        for k in 1..order {
            let sample = i32::from(input[ix - 1 - k]);
            let coeff = i32::from(coeffs_q12[k]);
            acc = acc.wrapping_add(sample * coeff);
        }

        let predicted = i32::from(input[ix]) << 12;
        let residual_q12 = predicted.wrapping_sub(acc);
        let residual = rshift_round(residual_q12, 12);
        output[ix] = sat16(residual);
    }

    for out in &mut output[..order.min(len)] {
        *out = 0;
    }
}

#[inline]
fn rshift_round(value: i32, shift: u32) -> i32 {
    debug_assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

#[inline]
fn sat16(value: i32) -> i16 {
    if value > i32::from(i16::MAX) {
        i16::MAX
    } else if value < i32::from(i16::MIN) {
        i16::MIN
    } else {
        value as i16
    }
}

#[cfg(test)]
mod tests {
    use super::lpc_analysis_filter;

    #[test]
    fn zeros_initial_state() {
        let mut output = [1i16; 12];
        let input = [0i16; 12];
        let coeffs = [0i16; 8];
        lpc_analysis_filter(&mut output, &input, &coeffs, input.len(), coeffs.len());
        assert_eq!(&output[..coeffs.len()], &[0; 8]);
    }

    #[test]
    fn matches_reference_sequence() {
        let input = [
            1345, -2123, 543, 1200, -876, 2222, -3000, 4096, -1024, 512, 256, -128, 64, -32, 16,
            -8,
        ];
        let coeffs = [4096, -2048, 1024, -512, 256, -128, 64, -32];
        let mut output = [0i16; 16];
        lpc_analysis_filter(&mut output, &input, &coeffs, input.len(), coeffs.len());
        assert_eq!(
            output,
            [
                0, 0, 0, 0, 0, 0, 0, 0, -7299, 4679, -2348, 920, -327, 96, -7, -24,
            ]
        );
    }
}
