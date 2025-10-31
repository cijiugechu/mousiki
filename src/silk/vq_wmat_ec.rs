//! Port of the entropy-constrained, matrix-weighted vector quantiser used
//! by the SILK encoder when searching the long-term prediction (LTP)
//! codebook.
//!
//! The routine mirrors `silk_VQ_WMat_EC_c` from `silk/VQ_WMat_EC.c`,
//! evaluating each codebook row against the correlation matrix and vector
//! to minimise the weighted rate/distortion score while respecting the
//! configured gain bound.

use crate::silk::lin2log::lin2log;

const LTP_ORDER: usize = 5;
const SILK_FIX_CONST_1_001_Q15: i32 = 32_801;

/// Result produced by [`vq_wmat_ec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VqWMatEcResult {
    /// Index of the best codebook entry.
    pub index: i8,
    /// Residual energy in Q15 for the selected entry, including the penalty term.
    pub residual_energy_q15: i32,
    /// Weighted rate/distortion score in Q8 for the selected entry.
    pub rate_dist_q8: i32,
    /// Sum of absolute LTP coefficients for the winning row, expressed in Q7.
    pub gain_q7: i32,
}

impl Default for VqWMatEcResult {
    fn default() -> Self {
        Self {
            index: 0,
            residual_energy_q15: i32::MAX,
            rate_dist_q8: i32::MAX,
            gain_q7: 0,
        }
    }
}

/// Entropy-constrained matrix-weighted VQ for 5-element LTP vectors.
///
/// * `xx_q17` — flattened 5×5 correlation matrix stored in row-major order.
/// * `x_x_q17` — correlation vector.
/// * `cb_q7` — concatenated codebook rows in Q7, with each row containing five entries.
/// * `cb_gain_q7` — per-row effective gains in Q7.
/// * `cl_q5` — per-row code lengths in Q5.
/// * `subfr_len` — number of time-domain samples represented by the correlations.
/// * `max_gain_q7` — limit on the sum of absolute LTP coefficients.
///
/// The returned [`VqWMatEcResult`] mirrors the structure updated by the C implementation.
#[allow(clippy::too_many_arguments)]
pub fn vq_wmat_ec(
    xx_q17: &[i32; LTP_ORDER * LTP_ORDER],
    x_x_q17: &[i32; LTP_ORDER],
    cb_q7: &[i8],
    cb_gain_q7: &[u8],
    cl_q5: &[u8],
    subfr_len: i32,
    max_gain_q7: i32,
) -> VqWMatEcResult {
    let l = cb_gain_q7.len();
    assert_eq!(
        cb_q7.len(),
        l * LTP_ORDER,
        "codebook rows must be contiguous"
    );
    assert_eq!(cl_q5.len(), l, "code lengths must match the codebook size");

    let mut neg_xx_q24 = [0i32; LTP_ORDER];
    for (dst, &value) in neg_xx_q24.iter_mut().zip(x_x_q17.iter()) {
        *dst = value.wrapping_shl(7).wrapping_neg();
    }

    let mut best = VqWMatEcResult::default();

    for (row_index, row) in cb_q7.chunks_exact(LTP_ORDER).enumerate() {
        let gain_tmp_q7 = i32::from(cb_gain_q7[row_index]);
        let penalty = (gain_tmp_q7 - max_gain_q7).max(0).wrapping_shl(11);

        let mut sum1_q15 = SILK_FIX_CONST_1_001_Q15;

        let mut sum2_q24 = mla(neg_xx_q24[0], xx_q17[1], i32::from(row[1]));
        sum2_q24 = mla(sum2_q24, xx_q17[2], i32::from(row[2]));
        sum2_q24 = mla(sum2_q24, xx_q17[3], i32::from(row[3]));
        sum2_q24 = mla(sum2_q24, xx_q17[4], i32::from(row[4]));
        sum2_q24 = sum2_q24.wrapping_shl(1);
        sum2_q24 = mla(sum2_q24, xx_q17[0], i32::from(row[0]));
        sum1_q15 = smlawb(sum1_q15, sum2_q24, i32::from(row[0]));

        sum2_q24 = mla(neg_xx_q24[1], xx_q17[7], i32::from(row[2]));
        sum2_q24 = mla(sum2_q24, xx_q17[8], i32::from(row[3]));
        sum2_q24 = mla(sum2_q24, xx_q17[9], i32::from(row[4]));
        sum2_q24 = sum2_q24.wrapping_shl(1);
        sum2_q24 = mla(sum2_q24, xx_q17[6], i32::from(row[1]));
        sum1_q15 = smlawb(sum1_q15, sum2_q24, i32::from(row[1]));

        sum2_q24 = mla(neg_xx_q24[2], xx_q17[13], i32::from(row[3]));
        sum2_q24 = mla(sum2_q24, xx_q17[14], i32::from(row[4]));
        sum2_q24 = sum2_q24.wrapping_shl(1);
        sum2_q24 = mla(sum2_q24, xx_q17[12], i32::from(row[2]));
        sum1_q15 = smlawb(sum1_q15, sum2_q24, i32::from(row[2]));

        sum2_q24 = mla(neg_xx_q24[3], xx_q17[19], i32::from(row[4]));
        sum2_q24 = sum2_q24.wrapping_shl(1);
        sum2_q24 = mla(sum2_q24, xx_q17[18], i32::from(row[3]));
        sum1_q15 = smlawb(sum1_q15, sum2_q24, i32::from(row[3]));

        sum2_q24 = neg_xx_q24[4].wrapping_shl(1);
        sum2_q24 = mla(sum2_q24, xx_q17[24], i32::from(row[4]));
        sum1_q15 = smlawb(sum1_q15, sum2_q24, i32::from(row[4]));

        if sum1_q15 >= 0 {
            let sum_with_penalty = sum1_q15.wrapping_add(penalty);
            let bits_res_q8 = smulbb(subfr_len, lin2log(sum_with_penalty) - (15 << 7));
            let bits_tot_q8 = add_lshift32(bits_res_q8, i32::from(cl_q5[row_index]), 2);

            if bits_tot_q8 <= best.rate_dist_q8 {
                best.index = row_index as i8;
                best.residual_energy_q15 = sum_with_penalty;
                best.rate_dist_q8 = bits_tot_q8;
                best.gain_q7 = gain_tmp_q7;
            }
        }
    }

    best
}

fn mla(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(b.wrapping_mul(c))
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let product = (i64::from(b) * i64::from(i32::from(c as i16))) >> 16;
    a.wrapping_add(product as i32)
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from((a as i16).wrapping_mul(b as i16))
}

fn add_lshift32(a: i32, b: i32, shift: u32) -> i32 {
    a.wrapping_add(b.wrapping_shl(shift))
}

#[cfg(test)]
mod tests {
    use super::{LTP_ORDER, VqWMatEcResult, vq_wmat_ec};
    use alloc::vec::Vec;

    #[test]
    fn selects_single_codebook_entry() {
        let xx_q17: [i32; 25] = [
            20_000, 1_000, 2_000, 3_000, 4_000, 1_000, 19_000, 1_100, 1_200, 1_300, 2_000, 1_100,
            18_000, 1_400, 1_500, 3_000, 1_200, 1_400, 17_000, 1_600, 4_000, 1_300, 1_500, 1_600,
            16_000,
        ];
        let x_x_q17: [i32; 5] = [300, 400, 500, 600, 700];
        let cb_q7 = [-3, -2, -1, 0, 1];
        let cb_gain_q7 = [12u8];
        let cl_q5 = [5u8];

        let result = vq_wmat_ec(&xx_q17, &x_x_q17, &cb_q7, &cb_gain_q7, &cl_q5, 60, 20);

        assert_eq!(
            result,
            VqWMatEcResult {
                index: 0,
                residual_energy_q15: 32_810,
                rate_dist_q8: 20,
                gain_q7: 12,
            },
        );
    }

    #[test]
    fn prefers_lower_rate_distortion() {
        let xx_q17: [i32; 25] = [
            21_000, 900, 1_100, 1_200, 1_400, 900, 21_000, 1_000, 1_100, 1_200, 1_100, 1_000,
            21_000, 1_050, 1_100, 1_200, 1_100, 1_050, 21_000, 1_000, 1_400, 1_200, 1_100, 1_000,
            21_000,
        ];
        let x_x_q17: [i32; 5] = [250, 260, 270, 280, 290];
        let cb_q7 = [
            3, 2, 1, 0, -1, // row 0
            -2, -2, -2, -2, -2, // row 1
        ];
        let cb_gain_q7 = [18u8, 10u8];
        let cl_q5 = [10u8, 6u8];

        let result = vq_wmat_ec(&xx_q17, &x_x_q17, &cb_q7, &cb_gain_q7, &cl_q5, 80, 25);

        assert_eq!(
            result,
            VqWMatEcResult {
                index: 1,
                residual_energy_q15: 32_816,
                rate_dist_q8: 24,
                gain_q7: 10,
            },
        );
    }

    #[test]
    fn returns_default_when_all_candidates_negative() {
        let xx_q17: [i32; 25] = [
            -1_000_000, 0, 0, 0, 0, 0, -1_000_000, 0, 0, 0, 0, 0, -1_000_000, 0, 0, 0, 0, 0,
            -1_000_000, 0, 0, 0, 0, 0, -1_000_000,
        ];
        let x_x_q17: [i32; 5] = [0, 0, 0, 0, 0];
        let cb_q7 = [127, 127, 127, 127, 127];
        let cb_gain_q7 = [127u8];
        let cl_q5 = [0u8];

        let result = vq_wmat_ec(&xx_q17, &x_x_q17, &cb_q7, &cb_gain_q7, &cl_q5, 40, 200);

        assert_eq!(result, VqWMatEcResult::default());
    }

    #[test]
    fn codebook_layout_matches_rows() {
        let mut cb = Vec::new();
        for row in 0..3 {
            for col in 0..LTP_ORDER {
                cb.push((row * 10 + col) as i8);
            }
        }
        let xx_q17: [i32; 25] = [
            25_000, 0, 0, 0, 0, 0, 25_000, 0, 0, 0, 0, 0, 25_000, 0, 0, 0, 0, 0, 25_000, 0, 0, 0,
            0, 0, 25_000,
        ];
        let x_x_q17: [i32; 5] = [0, 0, 0, 0, 0];
        let cb_gain_q7 = [0u8, 0u8, 0u8];
        let cl_q5 = [0u8, 0u8, 0u8];

        let result = vq_wmat_ec(&xx_q17, &x_x_q17, &cb, &cb_gain_q7, &cl_q5, 20, 100);

        assert_eq!(result.index, 0);
    }
}
