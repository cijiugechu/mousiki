//! Port of `silk/fixed/find_LTP_FIX.c`.
//!
//! Builds the long-term prediction (LTP) correlation matrices and vectors that
//! feed the LTP gain quantiser. The routine mirrors the reference fixed-point
//! implementation, including the dynamic right-shift alignment and the final
//! Q17 normalisation to keep the subsequent solver numerically stable.

use crate::silk::MAX_NB_SUBFR;
use crate::silk::corr_matrix::{corr_matrix_fix, corr_vector_fix};
use crate::silk::sum_sqr_shift::sum_sqr_shift;
use crate::silk::tuning_parameters::LTP_CORR_INV_MAX;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use core::cmp::Ordering;

const LTP_CORR_INV_MAX_Q16: i32 = ((LTP_CORR_INV_MAX as f64) * ((1 << 16) as f64) + 0.5) as i32;

/// Mirrors `silk_find_LTP_FIX`.
///
/// * `xx_ltp_q17` — destination for the per-subframe correlation matrices.
/// * `x_ltp_q17` — destination for the correlation vectors.
/// * `residual` — pitch residual with the LTP history prepended.
/// * `residual_offset` — index where the current frame begins inside `residual`.
/// * `lag` — per-subframe pitch lags.
/// * `subfr_length` — number of samples per subframe.
/// * `nb_subfr` — number of subframes in the frame (2 or 4).
/// * `arch` — run-time architecture flag preserved for DSP parity.
#[allow(clippy::too_many_arguments)]
pub fn find_ltp_fix(
    xx_ltp_q17: &mut [i32],
    x_ltp_q17: &mut [i32],
    residual: &[i16],
    residual_offset: usize,
    lag: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    arch: i32,
) {
    assert!(nb_subfr > 0, "nb_subfr must be positive");
    assert!(nb_subfr <= MAX_NB_SUBFR, "nb_subfr exceeds MAX_NB_SUBFR");
    assert!(lag.len() >= nb_subfr, "lag slice shorter than nb_subfr");
    assert!(subfr_length > 0, "subframe length must be positive");

    let matrix_stride = LTP_ORDER * LTP_ORDER;
    assert!(
        xx_ltp_q17.len() >= nb_subfr * matrix_stride,
        "XX buffer must hold nb_subfr × LTP_ORDER² entries"
    );
    assert!(
        x_ltp_q17.len() >= nb_subfr * LTP_ORDER,
        "xX buffer must hold nb_subfr × LTP_ORDER entries"
    );

    let mut r_ptr_idx = residual_offset;

    for (subfr, &lag_value) in lag.iter().take(nb_subfr).enumerate() {
        assert!(lag_value > 0, "LTP lag must be positive");
        let lag_usize = usize::try_from(lag_value).expect("lag must be non-negative");
        let history = lag_usize + LTP_ORDER / 2;
        assert!(
            r_ptr_idx >= history,
            "insufficient pitch history before subframe {subfr}"
        );

        let lag_ptr_idx = r_ptr_idx - history;
        let corr_len = subfr_length + LTP_ORDER - 1;
        let energy_len = subfr_length + LTP_ORDER;
        let lag_end = lag_ptr_idx + corr_len;
        let energy_end = r_ptr_idx + energy_len;
        assert!(
            lag_end <= residual.len(),
            "lag window exceeds residual slice"
        );
        assert!(
            energy_end <= residual.len(),
            "energy window exceeds residual slice"
        );

        let energy_slice = &residual[r_ptr_idx..energy_end];
        let (mut xx_energy, xx_shifts) = sum_sqr_shift(energy_slice);

        let matrix_offset = subfr * matrix_stride;
        let matrix_block = &mut xx_ltp_q17[matrix_offset..matrix_offset + matrix_stride];
        let (mut nrg, xx_matrix_shifts) = corr_matrix_fix(
            matrix_block,
            &residual[lag_ptr_idx..lag_end],
            subfr_length,
            LTP_ORDER,
            arch,
        );

        let extra_shifts = xx_shifts - xx_matrix_shifts;
        let x_x_shifts = match extra_shifts.cmp(&0) {
            Ordering::Greater => {
                shift_block(matrix_block, extra_shifts);
                nrg = rshift32(nrg, extra_shifts);
                xx_shifts
            }
            Ordering::Less => {
                xx_energy = rshift32(xx_energy, -extra_shifts);
                xx_matrix_shifts
            }
            Ordering::Equal => xx_shifts,
        };

        let vector_offset = subfr * LTP_ORDER;
        let vector_block = &mut x_ltp_q17[vector_offset..vector_offset + LTP_ORDER];
        corr_vector_fix(
            vector_block,
            &residual[lag_ptr_idx..lag_end],
            &residual[r_ptr_idx..r_ptr_idx + subfr_length],
            subfr_length,
            LTP_ORDER,
            x_x_shifts,
            arch,
        );

        let mut temp = smlawb(1, nrg, LTP_CORR_INV_MAX_Q16);
        temp = temp.max(xx_energy);
        debug_assert!(temp > 0);

        scale_block(matrix_block, temp);
        scale_block(vector_block, temp);

        r_ptr_idx += subfr_length;
    }
}

fn shift_block(values: &mut [i32], shift: i32) {
    for value in values {
        *value = rshift32(*value, shift);
    }
}

fn scale_block(values: &mut [i32], denom: i32) {
    for value in values {
        let scaled = ((i64::from(*value) << 17) / i64::from(denom)) as i32;
        *value = scaled;
    }
}

fn smlawb(acc: i32, x: i32, y_q16: i32) -> i32 {
    let product = (i64::from(x) * i64::from(i32::from(y_q16 as i16))) >> 16;
    acc.wrapping_add(product as i32)
}

fn rshift32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else if shift >= 31 {
        if value >= 0 { 0 } else { -1 }
    } else {
        value >> shift
    }
}

#[cfg(test)]
mod tests {
    use super::LTP_ORDER;
    use super::find_ltp_fix;
    use crate::silk::MAX_NB_SUBFR;
    use alloc::vec;

    #[test]
    fn zero_residual_produces_zero_correlations() {
        let nb_subfr = 2;
        let subfr_length = 4;
        let mut xx = vec![0; nb_subfr * LTP_ORDER * LTP_ORDER];
        let mut x = vec![0; nb_subfr * LTP_ORDER];
        let residual = vec![0i16; 64];
        let lag = [10, 12, 0, 0];

        find_ltp_fix(
            &mut xx,
            &mut x,
            &residual,
            20,
            &lag,
            subfr_length,
            nb_subfr,
            0,
        );

        assert!(xx.iter().all(|&v| v == 0));
        assert!(x.iter().all(|&v| v == 0));
    }

    #[test]
    fn matches_reference_for_small_vector() {
        let nb_subfr = 2;
        let subfr_length = 4;
        let mut xx = vec![0; nb_subfr * LTP_ORDER * LTP_ORDER];
        let mut x = vec![0; nb_subfr * LTP_ORDER];
        let mut residual = vec![0i16; 64];
        for (idx, sample) in residual.iter_mut().enumerate() {
            *sample = (idx as i32 * 13 - 40) as i16;
        }
        let lag = [6, 8, 0, 0];

        find_ltp_fix(
            &mut xx,
            &mut x,
            &residual,
            12,
            &lag,
            subfr_length,
            nb_subfr,
            0,
        );

        let expected_xx = [
            14257, 12103, 9948, 7794, 5639, 12103, 10284, 8465, 6645, 4826, 9948, 8465, 6981, 5497,
            4014, 7794, 6645, 5497, 4349, 3201, 5639, 4826, 4014, 3201, 2388, 14351, 12677, 11002,
            9328, 7654, 12677, 11201, 9726, 8250, 6775, 11002, 9726, 8449, 7172, 5895, 9328, 8250,
            7172, 6094, 5016, 7654, 6775, 5895, 5016, 4137,
        ];
        let expected_x = [
            22875, 19379, 15883, 12386, 8890, 24397, 21530, 18663, 15796, 12929,
        ];

        assert_eq!(xx.as_slice(), expected_xx);
        assert_eq!(x.as_slice(), expected_x);
    }

    #[test]
    #[should_panic]
    fn panics_when_nb_subfr_exceeds_limit() {
        let mut xx = vec![0; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
        let mut x = vec![0; MAX_NB_SUBFR * LTP_ORDER];
        let residual = vec![0i16; 128];
        let lag = [4; MAX_NB_SUBFR];
        find_ltp_fix(&mut xx, &mut x, &residual, 20, &lag, 4, MAX_NB_SUBFR + 1, 0);
    }
}
