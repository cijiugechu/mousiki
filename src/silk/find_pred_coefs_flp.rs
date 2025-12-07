//! Port of `silk/float/find_pred_coefs_FLP.c`.
//!
//! Computes the floating-point LPC and LTP predictors for the encoder analysis
//! path, mirroring the voiced/unvoiced branches from the reference SILK
//! implementation.

#![allow(clippy::too_many_arguments)]
use core::convert::TryFrom;

use crate::silk::decode_indices::ConditionalCoding;
use crate::silk::encoder::control_flp::EncoderControlFlp;
use crate::silk::encoder::state::MAX_FRAME_LENGTH;
use crate::silk::encoder::state_flp::EncoderStateFlp;
use crate::silk::find_lpc_flp::find_lpc_flp;
use crate::silk::find_ltp_flp::find_ltp_flp;
use crate::silk::ltp_analysis_filter_flp::ltp_analysis_filter_flp;
use crate::silk::ltp_scale_ctrl_flp::ltp_scale_ctrl_flp;
use crate::silk::ltp_scale_ctrl::LtpScaleCtrlParams;
use crate::silk::process_nlsfs::{ProcessNlsfConfig, process_nlsfs};
use crate::silk::quant_ltp_gains::silk_quant_ltp_gains;
use crate::silk::residual_energy_flp::residual_energy_flp;
use crate::silk::scale_vector::scale_copy_vector;
use crate::silk::sigproc_flp::silk_float2int;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR};
use libm::powf;

const MAX_PREDICTION_POWER_GAIN: f32 = 1.0e4;
const MAX_PREDICTION_POWER_GAIN_AFTER_RESET: f32 = 1.0e2;
const Q17_SCALE: f32 = 131_072.0;
const INV_Q14_SCALE: f32 = 1.0 / 16_384.0;
const INV_Q12_SCALE: f32 = 1.0 / 4096.0;
const INV_LOG_GAIN_SCALE: f32 = 1.0 / 128.0;
const MAX_LPC_IN_PRE: usize = MAX_NB_SUBFR * MAX_LPC_ORDER + MAX_FRAME_LENGTH;

/// Mirrors `silk_find_pred_coefs_FLP`.
///
/// * `res_pitch` — pitch residual with `ltp_mem_length` samples of history and
///   at least `la_pitch` samples of look-ahead.
/// * `x` — input signal preceded by `ltp_mem_length` history samples.
pub fn find_pred_coefs_flp(
    encoder: &mut EncoderStateFlp,
    control: &mut EncoderControlFlp,
    res_pitch: &[f32],
    x: &[f32],
    cond_coding: ConditionalCoding,
) {
    let nb_subfr = encoder.common.nb_subfr;
    assert!(
        nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2,
        "nb_subfr must be 2 or 4"
    );

    let order = encoder.common.predict_lpc_order;
    assert!(
        matches!(order, 6 | 8 | 10 | 12 | 16),
        "unsupported LPC order: {order}"
    );
    let subfr_length = encoder.common.subfr_length;
    assert!(subfr_length > 0, "subframe length must be positive");

    let frame_length = encoder.common.frame_length;
    assert_eq!(
        frame_length,
        nb_subfr * subfr_length,
        "frame length must match nb_subfr × subfr_length"
    );

    assert!(
        encoder.common.ltp_mem_length >= order,
        "ltp_mem_length must be at least the LPC order"
    );
    let la_pitch = usize::try_from(encoder.common.la_pitch).expect("la_pitch must be non-negative");

    let required_res_len = encoder.common.ltp_mem_length + frame_length + la_pitch;
    assert!(
        res_pitch.len() >= required_res_len,
        "res_pitch slice too short for history + frame"
    );

    let total_input = encoder.common.ltp_mem_length + frame_length;
    assert!(
        x.len() >= total_input,
        "input slice too short for history + frame"
    );

    let mut inv_gains = [0.0f32; MAX_NB_SUBFR];
    for (inv_gain, &gain) in inv_gains
        .iter_mut()
        .zip(control.gains.iter().take(nb_subfr))
    {
        assert!(gain > 0.0, "subframe gains must stay positive");
        *inv_gain = 1.0 / gain;
    }

    let chunk = subfr_length + order;
    let lpc_in_pre_len = nb_subfr
        .checked_mul(chunk)
        .expect("nb_subfr × chunk overflow");
    debug_assert!(
        lpc_in_pre_len <= MAX_LPC_IN_PRE,
        "LPC_in_pre exceeds stack buffer"
    );
    let mut lpc_in_pre_buf = [0.0f32; MAX_LPC_IN_PRE];
    let lpc_in_pre = &mut lpc_in_pre_buf[..lpc_in_pre_len];
    let taps_len = nb_subfr * LTP_ORDER;

    if matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced) {
        ensure_pitch_history(encoder.common.ltp_mem_length, order, control.pitch_l[0]);

        let mut xx_ltp = [0.0f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
        let mut x_ltp = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];

        find_ltp_flp(
            &mut xx_ltp[..taps_len * LTP_ORDER],
            &mut x_ltp[..taps_len],
            res_pitch,
            encoder.common.ltp_mem_length,
            &control.pitch_l[..nb_subfr],
            subfr_length,
            nb_subfr,
            encoder.common.arch,
        );

        quant_ltp_gains_flp(
            &mut control.ltp_coef[..taps_len],
            &mut encoder.common.indices.ltp_index[..nb_subfr],
            &mut encoder.common.indices.per_index,
            &mut encoder.common.sum_log_gain_q7,
            &mut control.lt_pred_cod_gain,
            &xx_ltp[..taps_len * LTP_ORDER],
            &x_ltp[..taps_len],
            subfr_length,
            nb_subfr,
            encoder.common.arch,
        );

        // Copy the LTP-scale inputs so we can borrow `indices` mutably without aliasing `common`.
        let ltp_params = LtpScaleCtrlParams {
            packet_loss_perc: encoder.common.packet_loss_perc,
            frames_per_packet: encoder.common.n_frames_per_packet,
            lbrr_enabled: encoder.common.lbrr_enabled,
            snr_db_q7: encoder.common.snr_db_q7,
        };
        let scale = ltp_scale_ctrl_flp(
            &ltp_params,
            &mut encoder.common.indices,
            cond_coding,
            control.lt_pred_cod_gain,
        );
        control.ltp_scale = scale;

        let x_ptr_offset = encoder.common.ltp_mem_length - order;
        ltp_analysis_filter_flp(
            lpc_in_pre,
            x,
            x_ptr_offset,
            &control.ltp_coef[..taps_len],
            &control.pitch_l[..nb_subfr],
            &inv_gains[..nb_subfr],
            subfr_length,
            nb_subfr,
            order,
        );
    } else {
        let mut x_ptr_idx = encoder.common.ltp_mem_length - order;
        for (subfr, dest) in lpc_in_pre.chunks_mut(chunk).take(nb_subfr).enumerate() {
            let src_end = x_ptr_idx + chunk;
            assert!(
                src_end <= x.len(),
                "input slice too short for subframe {subfr}"
            );
            scale_copy_vector(dest, &x[x_ptr_idx..src_end], inv_gains[subfr]);
            x_ptr_idx += subfr_length;
        }

        control.ltp_coef[..taps_len].fill(0.0);
        encoder.common.indices.ltp_index[..nb_subfr].fill(0);
        encoder.common.indices.per_index = 0;
        encoder.common.indices.ltp_scale_index = 0;
        control.lt_pred_cod_gain = 0.0;
        encoder.common.sum_log_gain_q7 = 0;
        control.ltp_scale = 0.0;
    }

    let min_inv_gain = if encoder.common.first_frame_after_reset {
        1.0 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET
    } else {
        let base = powf(2.0, control.lt_pred_cod_gain / 3.0) / MAX_PREDICTION_POWER_GAIN;
        base / (0.25 + 0.75 * control.coding_quality)
    };

    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    find_lpc_flp(
        &mut encoder.common,
        &mut nlsf_q15[..order],
        lpc_in_pre,
        min_inv_gain,
    );

    process_nlsfs_flp(&mut encoder.common, control, &mut nlsf_q15[..order]);

    residual_energy_flp(
        &mut control.res_nrg,
        lpc_in_pre,
        &control.pred_coef,
        &control.gains,
        subfr_length,
        nb_subfr,
        order,
    );

    encoder.common.prev_nlsf_q15[..order].copy_from_slice(&nlsf_q15[..order]);
}

fn ensure_pitch_history(ltp_mem_length: usize, order: usize, pitch: i32) {
    assert!(pitch > 0, "pitch lag must be positive");
    let pitch_usize = usize::try_from(pitch).expect("pitch lag must fit in usize");
    let required = pitch_usize + (LTP_ORDER / 2);
    assert!(
        ltp_mem_length.saturating_sub(order) >= required,
        "insufficient LTP history for pitch analysis"
    );
}

fn quant_ltp_gains_flp(
    b: &mut [f32],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_db: &mut f32,
    xx: &[f32],
    x_x: &[f32],
    subfr_len: usize,
    nb_subfr: usize,
    _arch: i32,
) {
    assert!(matches!(nb_subfr, 2 | 4));
    let taps_len = nb_subfr * LTP_ORDER;
    assert_eq!(
        b.len(),
        taps_len,
        "LTP coefficient buffer must hold nb_subfr × LTP_ORDER"
    );
    assert_eq!(
        cbk_index.len(),
        nb_subfr,
        "codebook index slice must match nb_subfr"
    );
    assert_eq!(
        xx.len(),
        taps_len * LTP_ORDER,
        "XX slice must contain nb_subfr correlation matrices"
    );
    assert_eq!(
        x_x.len(),
        taps_len,
        "xX slice must contain nb_subfr correlation vectors"
    );

    let mut xx_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
    for (dst, &src) in xx_q17.iter_mut().zip(xx.iter()) {
        *dst = silk_float2int(src * Q17_SCALE);
    }

    let mut x_x_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER];
    for (dst, &src) in x_x_q17.iter_mut().zip(x_x.iter()) {
        *dst = silk_float2int(src * Q17_SCALE);
    }

    let mut b_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    let mut pred_gain_db_q7 = 0;
    silk_quant_ltp_gains(
        &mut b_q14[..taps_len],
        cbk_index,
        periodicity_index,
        sum_log_gain_q7,
        &mut pred_gain_db_q7,
        &xx_q17[..taps_len * LTP_ORDER],
        &x_x_q17[..taps_len],
        subfr_len as i32,
        nb_subfr,
    );

    for (dst, &src) in b.iter_mut().zip(b_q14.iter().take(taps_len)) {
        *dst = f32::from(src) * INV_Q14_SCALE;
    }
    *pred_gain_db = (pred_gain_db_q7 as f32) * INV_LOG_GAIN_SCALE;
}

fn process_nlsfs_flp(
    encoder: &mut crate::silk::encoder::state::EncoderStateCommon,
    control: &mut EncoderControlFlp,
    nlsf_q15: &mut [i16],
) {
    let order = encoder.predict_lpc_order;
    assert!(
        nlsf_q15.len() >= order,
        "NLSF buffer shorter than LPC order"
    );

    let survivors =
        usize::try_from(encoder.nlsf_msvq_survivors).expect("survivor count must be non-negative");
    let cfg = ProcessNlsfConfig {
        speech_activity_q8: encoder.speech_activity_q8,
        nb_subframes: encoder.nb_subfr,
        predict_lpc_order: order,
        use_interpolated_nlsfs: encoder.use_interpolated_nlsfs,
        nlsf_msvq_survivors: survivors,
        codebook: encoder.ps_nlsf_cb,
        arch: encoder.arch,
    };

    let mut pred_coef_q12 = [[0i16; MAX_LPC_ORDER]; 2];
    process_nlsfs(
        &cfg,
        &mut encoder.indices,
        &mut pred_coef_q12,
        nlsf_q15,
        &encoder.prev_nlsf_q15[..order],
    );

    for (dst_row, src_row) in control.pred_coef.iter_mut().zip(pred_coef_q12.iter()) {
        for (dst, &src) in dst_row.iter_mut().zip(src_row.iter().take(order)) {
            *dst = f32::from(src) * INV_Q12_SCALE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn unvoiced_path_resets_ltp_state() {
        let mut encoder = EncoderStateFlp::default();
        encoder.common.indices.signal_type = FrameSignalType::Unvoiced;
        encoder.common.nlsf_msvq_survivors = 1;

        let nb_subfr = encoder.common.nb_subfr;
        let order = encoder.common.predict_lpc_order;
        let total = encoder.common.ltp_mem_length + encoder.common.frame_length;
        let la_pitch = usize::try_from(encoder.common.la_pitch).unwrap();

        let res_pitch = vec![0.0f32; total + la_pitch];
        let mut x = vec![0.0f32; total];
        // Seed the buffer to avoid all-zero predictors.
        for sample in x.iter_mut().take(total) {
            *sample = 0.001;
        }

        let mut control = EncoderControlFlp::default();
        for gain in control.gains.iter_mut().take(nb_subfr) {
            *gain = 1.0;
        }

        find_pred_coefs_flp(
            &mut encoder,
            &mut control,
            &res_pitch,
            &x,
            ConditionalCoding::Independent,
        );

        assert!(
            control.ltp_coef[..nb_subfr * LTP_ORDER]
                .iter()
                .all(|&c| c == 0.0)
        );
        assert_eq!(control.lt_pred_cod_gain, 0.0);
        assert_eq!(encoder.common.sum_log_gain_q7, 0);
        assert_eq!(encoder.common.indices.ltp_scale_index, 0);
        assert_eq!(control.ltp_scale, 0.0);
        assert!(control.res_nrg[..nb_subfr].iter().all(|v| v.is_finite()));
        assert!(
            control
                .pred_coef
                .iter()
                .all(|row| row[..order].iter().all(|v| v.is_finite()))
        );
    }
}
