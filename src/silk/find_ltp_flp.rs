//! Port of `silk/float/find_LTP_FLP.c`.
//!
//! Builds the floating-point long-term prediction (LTP) correlation matrices
//! and vectors that feed the LTP gain quantiser, mirroring the reference C
//! implementation’s scaling and normalisation.

use crate::silk::MAX_NB_SUBFR;
use crate::silk::corr_matrix_flp::{corr_matrix_flp, corr_vector_flp};
use crate::silk::energy_flp::energy;
use crate::silk::scale_vector::scale_vector;
use crate::silk::tuning_parameters::LTP_CORR_INV_MAX;
use crate::silk::vq_wmat_ec::LTP_ORDER;

/// Mirrors `silk_find_LTP_FLP`.
///
/// * `xx_ltp` — destination for the per-subframe correlation matrices.
/// * `x_ltp` — destination for the correlation vectors.
/// * `residual` — pitch residual with the LTP history prepended.
/// * `residual_offset` — index where the current frame begins inside `residual`.
/// * `lag` — per-subframe pitch lags.
/// * `subfr_length` — number of samples per subframe.
/// * `nb_subfr` — number of subframes in the frame (2 or 4).
/// * `arch` — run-time architecture flag preserved for DSP parity.
#[allow(clippy::too_many_arguments)]
pub fn find_ltp_flp(
    xx_ltp: &mut [f32],
    x_ltp: &mut [f32],
    residual: &[f32],
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
        xx_ltp.len() >= nb_subfr * matrix_stride,
        "XX buffer must hold nb_subfr × LTP_ORDER² entries"
    );
    assert!(
        x_ltp.len() >= nb_subfr * LTP_ORDER,
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

        let matrix_offset = subfr * matrix_stride;
        let vector_offset = subfr * LTP_ORDER;
        let matrix_block = &mut xx_ltp[matrix_offset..matrix_offset + matrix_stride];
        let vector_block = &mut x_ltp[vector_offset..vector_offset + LTP_ORDER];

        corr_matrix_flp(
            matrix_block,
            &residual[lag_ptr_idx..lag_end],
            subfr_length,
            LTP_ORDER,
            arch,
        );

        corr_vector_flp(
            vector_block,
            &residual[lag_ptr_idx..lag_end],
            &residual[r_ptr_idx..r_ptr_idx + subfr_length],
            subfr_length,
            LTP_ORDER,
            arch,
        );

        let xx_energy = energy(&residual[r_ptr_idx..energy_end]) as f32;
        let last_diag_idx = matrix_stride - 1;
        let denom = xx_energy
            .max(LTP_CORR_INV_MAX * 0.5 * (matrix_block[0] + matrix_block[last_diag_idx]) + 1.0);
        debug_assert!(denom.is_sign_positive());
        let inv_denom = 1.0 / denom;

        scale_vector(matrix_block, inv_denom);
        scale_vector(vector_block, inv_denom);

        r_ptr_idx += subfr_length;
    }
}

#[cfg(test)]
mod tests {
    use super::{LTP_ORDER, find_ltp_flp};
    use crate::silk::MAX_NB_SUBFR;
    use crate::silk::tuning_parameters::LTP_CORR_INV_MAX;
    use alloc::vec;
    use alloc::vec::Vec;

    fn reference_corr_vector(x: &[f32], t: &[f32], l: usize, order: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; order];
        for lag in 0..order {
            let start = order - 1 - lag;
            let mut dot = 0.0f32;
            for n in 0..l {
                dot += x[start + n] * t[n];
            }
            result[lag] = dot;
        }
        result
    }

    fn reference_corr_matrix(x: &[f32], l: usize, order: usize) -> Vec<f32> {
        let mut matrix = vec![0.0f32; order * order];
        for row in 0..order {
            let row_ptr = order - 1 - row;
            for col in 0..order {
                let col_ptr = order - 1 - col;
                let mut dot = 0.0f32;
                for n in 0..l {
                    dot += x[row_ptr + n] * x[col_ptr + n];
                }
                matrix[row * order + col] = dot;
            }
        }
        matrix
    }

    fn reference_find_ltp_flp(
        residual: &[f32],
        residual_offset: usize,
        lag: &[i32],
        subfr_length: usize,
        nb_subfr: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let matrix_stride = LTP_ORDER * LTP_ORDER;
        let mut xx = vec![0.0f32; nb_subfr * matrix_stride];
        let mut x = vec![0.0f32; nb_subfr * LTP_ORDER];
        let mut r_ptr_idx = residual_offset;

        for (subfr, &lag_value) in lag.iter().take(nb_subfr).enumerate() {
            let lag_usize = usize::try_from(lag_value).expect("lag must be non-negative");
            let history = lag_usize + LTP_ORDER / 2;
            let lag_ptr_idx = r_ptr_idx - history;
            let corr_len = subfr_length + LTP_ORDER - 1;

            let lag_slice = &residual[lag_ptr_idx..lag_ptr_idx + corr_len];
            let target = &residual[r_ptr_idx..r_ptr_idx + subfr_length];

            let mut matrix_block = reference_corr_matrix(lag_slice, subfr_length, LTP_ORDER);
            let mut vector_block =
                reference_corr_vector(lag_slice, target, subfr_length, LTP_ORDER);

            let xx_energy =
                super::energy(&residual[r_ptr_idx..r_ptr_idx + subfr_length + LTP_ORDER]) as f32;
            let last_diag_idx = matrix_stride - 1;
            let denom = xx_energy.max(
                LTP_CORR_INV_MAX * 0.5 * (matrix_block[0] + matrix_block[last_diag_idx]) + 1.0,
            );
            let scale = 1.0 / denom;
            for value in matrix_block.iter_mut() {
                *value *= scale;
            }
            for value in vector_block.iter_mut() {
                *value *= scale;
            }

            let matrix_offset = subfr * matrix_stride;
            xx[matrix_offset..matrix_offset + matrix_stride].copy_from_slice(&matrix_block);

            let vector_offset = subfr * LTP_ORDER;
            x[vector_offset..vector_offset + LTP_ORDER].copy_from_slice(&vector_block);

            r_ptr_idx += subfr_length;
        }

        (xx, x)
    }

    #[test]
    fn zero_residual_produces_zero_correlations() {
        let nb_subfr = 2;
        let subfr_length = 4;
        let mut xx = vec![1.0f32; nb_subfr * LTP_ORDER * LTP_ORDER];
        let mut x = vec![1.0f32; nb_subfr * LTP_ORDER];
        let residual = vec![0.0f32; 48];
        let lag = [8, 8, 0, 0];

        find_ltp_flp(
            &mut xx,
            &mut x,
            &residual,
            20,
            &lag,
            subfr_length,
            nb_subfr,
            0,
        );

        assert!(xx.iter().all(|&v| v == 0.0));
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn matches_reference_computation() {
        let nb_subfr = 2;
        let subfr_length = 6;
        let mut xx = vec![0.0f32; nb_subfr * LTP_ORDER * LTP_ORDER];
        let mut x = vec![0.0f32; nb_subfr * LTP_ORDER];
        let mut residual = vec![0.0f32; 48];
        for (idx, sample) in residual.iter_mut().enumerate() {
            *sample = (idx as f32 * 0.15) - 2.0;
        }
        let lag = [7, 10, 0, 0];
        let residual_offset = 18;

        find_ltp_flp(
            &mut xx,
            &mut x,
            &residual,
            residual_offset,
            &lag,
            subfr_length,
            nb_subfr,
            0,
        );

        let (expected_xx, expected_x) =
            reference_find_ltp_flp(&residual, residual_offset, &lag, subfr_length, nb_subfr);

        assert_eq!(xx.len(), expected_xx.len());
        assert_eq!(x.len(), expected_x.len());

        for (idx, (&actual, &expected)) in xx.iter().zip(expected_xx.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-4,
                "matrix difference at index {idx} exceeds tolerance: {diff}"
            );
        }

        for (idx, (&actual, &expected)) in x.iter().zip(expected_x.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-4,
                "vector difference at index {idx} exceeds tolerance: {diff}"
            );
        }
    }

    #[test]
    #[should_panic]
    fn panics_when_nb_subfr_exceeds_limit() {
        let mut xx = vec![0.0f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
        let mut x = vec![0.0f32; MAX_NB_SUBFR * LTP_ORDER];
        let residual = vec![0.0f32; 64];
        let lag = [4; MAX_NB_SUBFR];

        find_ltp_flp(&mut xx, &mut x, &residual, 20, &lag, 4, MAX_NB_SUBFR + 1, 0);
    }
}
