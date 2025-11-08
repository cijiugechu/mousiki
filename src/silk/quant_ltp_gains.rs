//! Ports the long-term prediction (LTP) gain quantiser from `silk/quant_LTP_gains.c`.
//!
//! The routine scans the three SILK LTP codebooks, runs the entropy-constrained
//! VQ search for each, and selects the configuration with the lowest rate/distortion
//! cost. It mirrors the fixed-point implementation, including the cumulative
//! `sum_log_gain` bookkeeping that guards against unstable gain trajectories.

use crate::silk::lin2log::lin2log;
use crate::silk::log2lin::log2lin;
use crate::silk::tables_ltp::{
    NB_LTP_CBKS, SILK_LTP_GAIN_BITS_Q5, SILK_LTP_GAIN_VQ_GAIN_Q7, SILK_LTP_GAIN_VQ_Q7,
    SILK_LTP_VQ_SIZES,
};
use crate::silk::tuning_parameters::MAX_SUM_LOG_GAIN_DB;
use crate::silk::vq_wmat_ec::{vq_wmat_ec, LTP_ORDER};
use crate::silk::MAX_NB_SUBFR;

const LOG_SCALE_OFFSET_Q7: i32 = 7 << 7;
const GAIN_SAFETY_Q7: i32 = ((0.4 * (1 << 7) as f32) + 0.5) as i32;
const MAX_SUM_LOG_GAIN_DB_Q7: i32 =
    (((MAX_SUM_LOG_GAIN_DB as f64) / 6.0) * (1 << 7) as f64 + 0.5) as i32;

/// Quantise LTP gains by evaluating all SILK codebooks.
#[allow(clippy::too_many_arguments)]
pub fn silk_quant_ltp_gains(
    b_q14: &mut [i16],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_db_q7: &mut i32,
    xx_q17: &[i32],
    x_x_q17: &[i32],
    subfr_len: i32,
    nb_subfr: usize,
) {
    debug_assert!(matches!(nb_subfr, 2 | 4));
    assert_eq!(
        b_q14.len(),
        nb_subfr * LTP_ORDER,
        "B_Q14 must provide nb_subfr × LTP_ORDER taps"
    );
    assert_eq!(
        cbk_index.len(),
        nb_subfr,
        "codebook index slice must match nb_subfr"
    );
    assert_eq!(
        xx_q17.len(),
        nb_subfr * LTP_ORDER * LTP_ORDER,
        "XX_Q17 must contain nb_subfr correlation matrices"
    );
    assert_eq!(
        x_x_q17.len(),
        nb_subfr * LTP_ORDER,
        "xX_Q17 must contain nb_subfr correlation vectors"
    );

    let mut temp_idx = [0i8; MAX_NB_SUBFR];
    let mut min_rate_dist_q8 = i32::MAX;
    let mut best_sum_log_gain_q7 = *sum_log_gain_q7;
    let mut res_nrg_q15 = 0;

    for k in 0..NB_LTP_CBKS {
        let cb_rows = SILK_LTP_GAIN_VQ_Q7[k];
        let gains_q7 = SILK_LTP_GAIN_VQ_GAIN_Q7[k];
        let code_lengths_q5 = SILK_LTP_GAIN_BITS_Q5[k];

        debug_assert_eq!(cb_rows.len(), SILK_LTP_VQ_SIZES[k] as usize);
        debug_assert_eq!(cb_rows.len(), gains_q7.len());
        debug_assert_eq!(cb_rows.len(), code_lengths_q5.len());

        res_nrg_q15 = 0;
        let mut rate_dist_q8 = 0;
        let mut sum_log_gain_tmp_q7 = *sum_log_gain_q7;

        let mut xx_offset = 0;
        let mut x_offset = 0;

        for temp in temp_idx.iter_mut().take(nb_subfr) {
            let log_target = (MAX_SUM_LOG_GAIN_DB_Q7 - sum_log_gain_tmp_q7)
                .saturating_add(LOG_SCALE_OFFSET_Q7);
            let mut max_gain_q7 = log2lin(log_target).saturating_sub(GAIN_SAFETY_Q7);
            if max_gain_q7 < 0 {
                max_gain_q7 = 0;
            }

            let mut xx_block = [0i32; LTP_ORDER * LTP_ORDER];
            xx_block.copy_from_slice(
                &xx_q17[xx_offset..xx_offset + LTP_ORDER * LTP_ORDER],
            );
            let mut x_block = [0i32; LTP_ORDER];
            x_block.copy_from_slice(&x_x_q17[x_offset..x_offset + LTP_ORDER]);

            let result = vq_wmat_ec(
                &xx_block,
                &x_block,
                cb_rows,
                gains_q7,
                code_lengths_q5,
                subfr_len,
                max_gain_q7,
            );

            *temp = result.index;
            res_nrg_q15 = add_pos_sat32(res_nrg_q15, result.residual_energy_q15);
            rate_dist_q8 = add_pos_sat32(rate_dist_q8, result.rate_dist_q8);

            let gain_log_delta =
                lin2log(GAIN_SAFETY_Q7 + result.gain_q7) - LOG_SCALE_OFFSET_Q7;
            sum_log_gain_tmp_q7 = (sum_log_gain_tmp_q7 + gain_log_delta).max(0);

            xx_offset += LTP_ORDER * LTP_ORDER;
            x_offset += LTP_ORDER;
        }

        if rate_dist_q8 <= min_rate_dist_q8 {
            min_rate_dist_q8 = rate_dist_q8;
            *periodicity_index = k as i8;
            cbk_index[..nb_subfr].copy_from_slice(&temp_idx[..nb_subfr]);
            best_sum_log_gain_q7 = sum_log_gain_tmp_q7;
        }
    }

    let selected_cb = SILK_LTP_GAIN_VQ_Q7[*periodicity_index as usize];
    for (subfr, &row_idx) in cbk_index.iter().enumerate() {
        let idx = usize::try_from(row_idx).expect("LTP codebook index must be non-negative");
        let taps = selected_cb
            .get(idx)
            .expect("index must fall within the selected LTP codebook");
        for (tap, &value) in taps.iter().enumerate() {
            b_q14[subfr * LTP_ORDER + tap] = (i32::from(value) << 7) as i16;
        }
    }

    if nb_subfr == 2 {
        res_nrg_q15 >>= 1;
    } else {
        res_nrg_q15 >>= 2;
    }

    *sum_log_gain_q7 = best_sum_log_gain_q7;
    *pred_gain_db_q7 = -3 * (lin2log(res_nrg_q15) - (15 << 7));
}

fn add_pos_sat32(a: i32, b: i32) -> i32 {
    let sum = i64::from(a) + i64::from(b);
    if sum > i64::from(i32::MAX) {
        i32::MAX
    } else {
        sum as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_two_subframes() {
        let mut b_q14 = [0i16; 2 * LTP_ORDER];
        let mut cbk_index = [0i8; 2];
        let mut periodicity = -1i8;
        let mut sum_log_gain = 1200;
        let mut pred_gain = 123;

        let xx_q17 = [
            21000, 1200, 800, 400, 200, 1200, 20500, 1100, 600, 300, 800, 1100, 20000, 900, 500,
            400, 600, 900, 19500, 700, 200, 300, 500, 700, 19000, 22000, 1500, 900, 500, 300, 1500,
            21000, 1200, 800, 400, 900, 1200, 20500, 700, 350, 500, 800, 700, 19800, 600, 300, 400,
            350, 600, 19200,
        ];
        let x_x_q17 = [1000, 800, 600, 400, 200, 900, 700, 500, 300, 100];

        silk_quant_ltp_gains(
            &mut b_q14,
            &mut cbk_index,
            &mut periodicity,
            &mut sum_log_gain,
            &mut pred_gain,
            &xx_q17,
            &x_x_q17,
            60,
            2,
        );

        assert_eq!(periodicity, 0);
        assert_eq!(pred_gain, 0);
        assert_eq!(sum_log_gain, 1098);
        assert_eq!(cbk_index, [0, 0]);
        assert_eq!(
            b_q14,
            [512, 768, 3072, 896, 640, 512, 768, 3072, 896, 640]
        );
    }

    #[test]
    fn matches_reference_four_subframes() {
        let mut b_q14 = [0i16; 4 * LTP_ORDER];
        let mut cbk_index = [0i8; 4];
        let mut periodicity = -1i8;
        let mut sum_log_gain = 2500;
        let mut pred_gain = 456;

        let xx_q17 = [
            21000, 1200, 800, 400, 200, 1200, 20500, 1100, 600, 300, 800, 1100, 20000, 900, 500,
            400, 600, 900, 19500, 700, 200, 300, 500, 700, 19000, 22500, 1600, 1000, 600, 250,
            1600, 21500, 1300, 900, 450, 1000, 1300, 20700, 750, 380, 600, 900, 750, 19900, 650,
            250, 450, 380, 650, 19350, 20000, 1100, 700, 300, 100, 1100, 19800, 800, 400, 200, 700,
            800, 19400, 600, 250, 300, 400, 600, 19000, 500, 100, 200, 250, 500, 18800, 21500, 1700,
            1300, 900, 500, 1700, 21200, 1400, 1100, 600, 1300, 1400, 20800, 1000, 550, 900, 1100,
            1000, 20000, 700, 500, 600, 550, 700, 19500,
        ];
        let x_x_q17 = [
            800, 600, 400, 200, 0, 750, 550, 350, 150, 50, 700, 500, 300, 100, -50, 650, 450, 250,
            50, -100,
        ];

        silk_quant_ltp_gains(
            &mut b_q14,
            &mut cbk_index,
            &mut periodicity,
            &mut sum_log_gain,
            &mut pred_gain,
            &xx_q17,
            &x_x_q17,
            120,
            4,
        );

        assert_eq!(periodicity, 0);
        assert_eq!(pred_gain, 0);
        assert_eq!(sum_log_gain, 2296);
        assert_eq!(cbk_index, [0, 0, 0, 0]);
        assert_eq!(
            b_q14,
            [
                512, 768, 3072, 896, 640, 512, 768, 3072, 896, 640, 512, 768, 3072, 896, 640, 512,
                768, 3072, 896, 640
            ]
        );
    }
}
