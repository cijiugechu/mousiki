//! Port of `silk/fixed/find_pred_coefs_FIX.c`.
//!
//! This helper estimates the long-term predictor (LTP) gains for voiced
//! frames, quantises and scales them, rebuilds the LPC predictors from the
//! LTP-filtered signal, and accumulates the residual energy that downstream
//! noise-shaping stages rely on. Unvoiced frames skip the LTP analysis and
//! reuse the scaled input directly, mirroring the reference fixed-point SILK
//! encoder.

use alloc::vec;
use core::convert::TryFrom;
use core::ptr;

use crate::silk::decode_indices::ConditionalCoding;
use crate::silk::encoder::control::EncoderControl;
use crate::silk::encoder::state::EncoderChannelState;
use crate::silk::find_lpc::find_lpc;
use crate::silk::find_ltp::find_ltp_fix;
use crate::silk::log2lin::log2lin;
use crate::silk::ltp_analysis_filter::ltp_analysis_filter;
use crate::silk::ltp_scale_ctrl::ltp_scale_ctrl;
use crate::silk::process_nlsfs::{ProcessNlsfConfig, process_nlsfs};
use crate::silk::quant_ltp_gains::silk_quant_ltp_gains;
use crate::silk::residual_energy::residual_energy;
use crate::silk::stereo_find_predictor::div32_varq;
use crate::silk::vector_ops::scale_copy_vector16;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR};

const INV_MAX_PRED_GAIN_AFTER_RESET_Q30: i32 = ((1.0 / 100.0) * ((1 << 30) as f64) + 0.5) as i32;
const ONE_THIRD_Q16: i32 = ((1.0 / 3.0) * ((1 << 16) as f64) + 0.5) as i32;
const QUARTER_Q18: i32 = ((0.25) * ((1 << 18) as f64) + 0.5) as i32;
const THREE_QUARTERS_Q18: i32 = ((0.75) * ((1 << 18) as f64) + 0.5) as i32;
const MAX_PREDICTION_POWER_GAIN_Q0: i32 = 10_000;

/// Mirrors `silk_find_pred_coefs_FIX`.
///
/// * `res_pitch` — pitch residual with `ltp_mem_length` samples of history and
///   at least `encoder.common.la_pitch` samples of look-ahead.
/// * `x` — input signal preceded by `predict_lpc_order` history samples.
#[allow(clippy::too_many_arguments)]
pub fn find_pred_coefs(
    encoder: &mut EncoderChannelState,
    control: &mut EncoderControl,
    res_pitch: &[i16],
    x: &[i16],
    cond_coding: ConditionalCoding,
) {
    let nb_subfr = encoder.common.nb_subfr;
    assert!(nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2);

    let order = encoder.common.predict_lpc_order;
    assert!(matches!(order, 10 | 16));
    assert!(order <= MAX_LPC_ORDER);

    let subfr_length = encoder.common.subfr_length;
    assert!(subfr_length > 0);

    let frame_length = encoder.common.frame_length;
    assert_eq!(frame_length, nb_subfr * subfr_length);

    let ltp_mem_length = encoder.common.ltp_mem_length;
    assert!(ltp_mem_length >= order);
    let la_pitch = usize::try_from(encoder.common.la_pitch).expect("la_pitch must be non-negative");

    assert!(
        res_pitch.len() >= ltp_mem_length + frame_length + la_pitch,
        "pitch residual slice too short"
    );
    let total_input = ltp_mem_length + frame_length;
    assert!(x.len() >= total_input, "input slice too short");

    let mut inv_gains_q16 = [0i32; MAX_NB_SUBFR];
    let mut local_gains = [0i32; MAX_NB_SUBFR];
    let mut min_gain_q16 = i32::MAX >> 6;

    for &gain in control.gains_q16.iter().take(nb_subfr) {
        min_gain_q16 = min_gain_q16.min(gain);
    }

    for (idx, inv_gain) in inv_gains_q16.iter_mut().take(nb_subfr).enumerate() {
        let gain = control.gains_q16[idx];
        assert!(gain > 0, "subframe gains must stay positive");
        let ratio = div32_varq(min_gain_q16, gain, 14);
        let limited = ratio.max(100);
        let clamped = limited.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
        *inv_gain = clamped;
        local_gains[idx] = div32(1 << 16, clamped.max(1));
    }

    let chunk = subfr_length + order;
    let mut lpc_in_pre = vec![0i16; nb_subfr * chunk];

    if matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced) {
        ensure_pitch_history(encoder.common.ltp_mem_length, order, control.pitch_l[0]);

        let taps_len = nb_subfr * LTP_ORDER;
        let mut x_ltp_q17 = vec![0i32; taps_len];
        let mut xx_ltp_q17 = vec![0i32; taps_len * LTP_ORDER];

        find_ltp_fix(
            &mut xx_ltp_q17,
            &mut x_ltp_q17,
            res_pitch,
            encoder.common.ltp_mem_length,
            &control.pitch_l[..nb_subfr],
            subfr_length,
            nb_subfr,
            encoder.common.arch,
        );

        silk_quant_ltp_gains(
            &mut control.ltp_coef_q14[..taps_len],
            &mut encoder.common.indices.ltp_index[..nb_subfr],
            &mut encoder.common.indices.per_index,
            &mut encoder.common.sum_log_gain_q7,
            &mut control.lt_pred_cod_gain_q7,
            &xx_ltp_q17,
            &x_ltp_q17,
            subfr_length as i32,
            nb_subfr,
        );

        let scale = unsafe {
            // SAFETY: We create disjoint references to the encoder common state and
            // its side-information indices for the duration of the call. The helper
            // only reads from `common` and mutates `indices`, matching the C layout.
            let common_ptr = ptr::addr_of!(encoder.common);
            let indices_ptr = ptr::addr_of_mut!(encoder.common.indices);
            ltp_scale_ctrl(
                &*common_ptr,
                &mut *indices_ptr,
                cond_coding,
                control.lt_pred_cod_gain_q7,
            )
        };
        control.ltp_scale_q14 = scale;

        let x_ptr_offset = encoder.common.ltp_mem_length - order;
        ltp_analysis_filter(
            &mut lpc_in_pre,
            x,
            x_ptr_offset,
            &control.ltp_coef_q14[..taps_len],
            &control.pitch_l[..nb_subfr],
            &inv_gains_q16[..nb_subfr],
            subfr_length,
            nb_subfr,
            order,
        );
    } else {
        let mut x_ptr_idx = encoder.common.ltp_mem_length - order;
        for (inv_gain, dest) in inv_gains_q16
            .iter()
            .zip(lpc_in_pre.chunks_mut(chunk))
            .take(nb_subfr)
        {
            let src_end = x_ptr_idx + chunk;
            assert!(src_end <= x.len(), "input slice too short for subframe");
            scale_copy_vector16(dest, &x[x_ptr_idx..src_end], *inv_gain);
            x_ptr_idx += subfr_length;
        }

        control.ltp_coef_q14[..nb_subfr * LTP_ORDER].fill(0);
        encoder.common.indices.ltp_index[..nb_subfr].fill(0);
        encoder.common.indices.per_index = 0;
        encoder.common.indices.ltp_scale_index = 0;
        control.lt_pred_cod_gain_q7 = 0;
        encoder.common.sum_log_gain_q7 = 0;
        control.ltp_scale_q14 = 0;
    }

    let min_inv_gain_q30 = if encoder.common.first_frame_after_reset {
        INV_MAX_PRED_GAIN_AFTER_RESET_Q30
    } else {
        let base_q7 = smlawb(16 << 7, control.lt_pred_cod_gain_q7, ONE_THIRD_Q16);
        let lin_q16 = log2lin(base_q7);
        let quality_q18 = smlawb(QUARTER_Q18, THREE_QUARTERS_Q18, control.coding_quality_q14);
        let denom = smulww(MAX_PREDICTION_POWER_GAIN_Q0, quality_q18).max(1);
        div32_varq(lin_q16, denom, 14)
    };

    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    find_lpc(
        &mut encoder.common,
        &mut nlsf_q15[..order],
        &lpc_in_pre,
        min_inv_gain_q30,
    );

    let survivors = usize::try_from(encoder.common.nlsf_msvq_survivors)
        .expect("survivor count must be non-negative");
    let cfg = ProcessNlsfConfig {
        speech_activity_q8: encoder.common.speech_activity_q8,
        nb_subframes: nb_subfr,
        predict_lpc_order: order,
        use_interpolated_nlsfs: encoder.common.use_interpolated_nlsfs,
        nlsf_msvq_survivors: survivors,
        codebook: encoder.common.ps_nlsf_cb,
        arch: encoder.common.arch,
    };

    process_nlsfs(
        &cfg,
        &mut encoder.common.indices,
        &mut control.pred_coef_q12,
        &mut nlsf_q15[..order],
        &encoder.common.prev_nlsf_q15[..order],
    );

    encoder.common.prev_nlsf_q15[..order].copy_from_slice(&nlsf_q15[..order]);

    residual_energy(
        &mut control.res_nrg,
        &mut control.res_nrg_q,
        &lpc_in_pre,
        &control.pred_coef_q12,
        &local_gains,
        subfr_length,
        nb_subfr,
        order,
        encoder.common.arch,
    );
}

fn div32(num: i32, denom: i32) -> i32 {
    debug_assert!(denom > 0);
    num / denom
}

fn ensure_pitch_history(ltp_mem_length: usize, order: usize, pitch: i32) {
    assert!(pitch > 0, "pitch lag must be positive");
    let pitch_usize = usize::try_from(pitch).expect("pitch lag must fit in usize");
    let required = pitch_usize + (LTP_ORDER / 2);
    assert!(ltp_mem_length.saturating_sub(order) >= required);
}

fn smlawb(acc: i32, x: i32, y_q16: i32) -> i32 {
    acc.wrapping_add(((i64::from(x) * i64::from(y_q16 as i16)) >> 16) as i32)
}

fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::TryFrom;
    use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;
    use alloc::vec::Vec;

    fn mock_encoder(signal_type: FrameSignalType, first_after_reset: bool) -> EncoderChannelState {
        let mut encoder = EncoderChannelState::default();
        encoder.common.indices.signal_type = signal_type;
        encoder.common.first_frame_after_reset = first_after_reset;
        encoder.common.nlsf_msvq_survivors = 4;
        encoder.common.packet_loss_perc = 5;
        encoder.common.n_frames_per_packet = 1;
        encoder.common.snr_db_q7 = 1500;
        encoder
    }

    fn mock_residual(len: usize) -> Vec<i16> {
        (0..len)
            .map(|idx| ((idx as i32 * 13 + 7) & 0xFFFF) as i16)
            .collect()
    }

    fn mock_input(len: usize) -> Vec<i16> {
        (0..len)
            .map(|idx| ((idx as i32 * 23).wrapping_sub(10_000)) as i16)
            .collect()
    }

    #[test]
    fn unvoiced_frame_skips_ltp() {
        let mut encoder = mock_encoder(FrameSignalType::Unvoiced, true);
        let nb_subfr = encoder.common.nb_subfr;
        let frame_length = encoder.common.frame_length;
        let ltp_history = encoder.common.ltp_mem_length;
        let la_pitch = usize::try_from(encoder.common.la_pitch).expect("la_pitch must be non-negative");
        let order = encoder.common.predict_lpc_order;

        let res_pitch = mock_residual(ltp_history + frame_length + la_pitch);
        let x = mock_input(ltp_history + frame_length);
        for sample in encoder.common.prev_nlsf_q15.iter_mut().take(order) {
            *sample = 500;
        }

        let mut control = EncoderControl::default();
        control.gains_q16[..nb_subfr].fill(1 << 16);
        control.pitch_l[..nb_subfr].fill(20);

        find_pred_coefs(
            &mut encoder,
            &mut control,
            &res_pitch,
            &x,
            ConditionalCoding::Conditional,
        );

        assert_eq!(control.lt_pred_cod_gain_q7, 0);
        assert_eq!(encoder.common.sum_log_gain_q7, 0);
        assert_eq!(encoder.common.indices.ltp_scale_index, 0);
        assert_eq!(control.ltp_scale_q14, 0);
        assert!(control.ltp_coef_q14.iter().all(|&tap| tap == 0));
        assert!(
            encoder
                .common
                .prev_nlsf_q15
                .iter()
                .take(order)
                .any(|&value| value != 0)
        );
    }

    #[test]
    fn voiced_path_populates_ltp_state() {
        let mut encoder = mock_encoder(FrameSignalType::Voiced, false);
        let nb_subfr = encoder.common.nb_subfr;
        let frame_length = encoder.common.frame_length;
        let ltp_history = encoder.common.ltp_mem_length;
        let la_pitch = usize::try_from(encoder.common.la_pitch).expect("la_pitch must be non-negative");
        let order = encoder.common.predict_lpc_order;

        let res_pitch = mock_residual(ltp_history + frame_length + la_pitch);
        let x = mock_input(ltp_history + frame_length);

        for (idx, sample) in encoder
            .common
            .prev_nlsf_q15
            .iter_mut()
            .enumerate()
            .take(order)
        {
            *sample = (idx as i16) * 200;
        }

        let mut control = EncoderControl::default();
        control.gains_q16[..nb_subfr].fill(90_000);
        control.pitch_l[..nb_subfr].copy_from_slice(&[40, 42, 44, 46]);
        control.coding_quality_q14 = 8192;

        find_pred_coefs(
            &mut encoder,
            &mut control,
            &res_pitch,
            &x,
            ConditionalCoding::Independent,
        );

        assert!(control.lt_pred_cod_gain_q7 > 0);
        assert!(encoder.common.sum_log_gain_q7 > 0);
        assert!(
            control
                .ltp_coef_q14
                .iter()
                .take(nb_subfr * LTP_ORDER)
                .any(|&tap| tap != 0)
        );
        let index = encoder.common.indices.ltp_scale_index as usize;
        assert_eq!(
            control.ltp_scale_q14,
            i32::from(SILK_LTPSCALES_TABLE_Q14[index])
        );
        assert!(
            encoder
                .common
                .prev_nlsf_q15
                .iter()
                .take(order)
                .any(|&v| v != 0)
        );
    }
}
