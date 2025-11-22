//! Port of `silk/float/find_pitch_lags_FLP.c`.
//!
//! Mirrors the floating-point pitch-analysis helper that windows the input,
//! derives LPC coefficients, filters the signal to obtain the residual, and
//! invokes the FLP pitch estimator to populate the lag indices.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::too_many_arguments,
    clippy::arithmetic_side_effects
)]

use crate::silk::apply_sine_window_flp::apply_sine_window_flp;
use crate::silk::autocorrelation_flp::autocorrelation;
use crate::silk::bwexpander_flp::bwexpander;
use crate::silk::encoder::control_flp::EncoderControlFlp;
use crate::silk::encoder::state::{FIND_PITCH_LPC_WIN_MS, MAX_FIND_PITCH_LPC_ORDER, MAX_FS_KHZ};
use crate::silk::encoder::state_flp::EncoderStateFlp;
use crate::silk::k2a_flp::k2a_flp;
use crate::silk::lpc_analysis_filter_flp::lpc_analysis_filter_flp;
use crate::silk::pitch_analysis_core_flp::pitch_analysis_core_flp;
use crate::silk::schur_flp::silk_schur_flp;
use crate::silk::tuning_parameters::{
    FIND_PITCH_BANDWIDTH_EXPANSION, FIND_PITCH_WHITE_NOISE_FRACTION,
};
use crate::silk::{FrameSignalType, MAX_NB_SUBFR};
use core::convert::TryFrom;

const MAX_PITCH_LPC_WIN_LENGTH: usize = FIND_PITCH_LPC_WIN_MS * MAX_FS_KHZ;

/// Mirrors `silk_find_pitch_lags_FLP`.
///
/// * `encoder` — floating-point per-channel encoder state.
/// * `control` — floating-point per-frame encoder control working area.
/// * `res` — destination for the LPC residual, including the LTP history and
///   pitch look-ahead samples.
/// * `x` — input signal aligned so that the first `encoder.common.ltp_mem_length`
///   samples contain the previous frame history.
pub fn find_pitch_lags_flp(
    encoder: &mut EncoderStateFlp,
    control: &mut EncoderControlFlp,
    res: &mut [f32],
    x: &[f32],
    arch: i32,
) {
    let nb_subfr = encoder.common.nb_subfr;
    assert!(
        nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2,
        "nb_subfr must be 2 or 4"
    );

    let la_pitch = usize::try_from(encoder.common.la_pitch).expect("la_pitch must fit in usize");
    let buf_len = la_pitch + encoder.common.frame_length + encoder.common.ltp_mem_length;
    assert!(res.len() >= buf_len, "residual buffer too short");
    assert!(x.len() >= buf_len, "input buffer too short");

    let pitch_lpc_win_length = encoder.common.pitch_lpc_win_length;
    assert!(
        pitch_lpc_win_length <= MAX_PITCH_LPC_WIN_LENGTH,
        "pitch LPC window exceeds static scratch space"
    );
    assert!(
        pitch_lpc_win_length >= la_pitch * 2,
        "pitch LPC window shorter than the required taper"
    );
    assert!(
        buf_len >= pitch_lpc_win_length,
        "window length exceeds available samples"
    );

    let order = usize::try_from(encoder.common.pitch_estimation_lpc_order)
        .expect("pitch_estimation_lpc_order must be non-negative");
    assert!(
        (6..=MAX_FIND_PITCH_LPC_ORDER as usize).contains(&order),
        "pitch_estimation_lpc_order must be in [6, MAX_FIND_PITCH_LPC_ORDER]"
    );
    assert!(
        order.is_multiple_of(2),
        "pitch_estimation_lpc_order must be even"
    );
    assert!(
        matches!(encoder.common.fs_khz, 8 | 12 | 16),
        "unsupported sampling rate"
    );

    let x_ptr = &x[buf_len - pitch_lpc_win_length..buf_len];
    let mut wsig = [0f32; MAX_PITCH_LPC_WIN_LENGTH];
    let window = &mut wsig[..pitch_lpc_win_length];
    let (head, rest) = window.split_at_mut(la_pitch);
    apply_sine_window_flp(head, &x_ptr[..la_pitch], 1);

    let (middle, tail) = rest.split_at_mut(pitch_lpc_win_length - la_pitch * 2);
    middle.copy_from_slice(&x_ptr[la_pitch..pitch_lpc_win_length - la_pitch]);
    apply_sine_window_flp(
        tail,
        &x_ptr[pitch_lpc_win_length - la_pitch..pitch_lpc_win_length],
        2,
    );

    let corr_count = order + 1;
    let mut auto_corr = [0f32; MAX_FIND_PITCH_LPC_ORDER as usize + 1];
    autocorrelation(&mut auto_corr[..corr_count], window, corr_count);
    auto_corr[0] += auto_corr[0] * FIND_PITCH_WHITE_NOISE_FRACTION + 1.0;

    let mut refl_coef = [0f32; MAX_FIND_PITCH_LPC_ORDER as usize];
    let res_nrg = silk_schur_flp(&mut refl_coef[..order], &auto_corr[..corr_count], order);
    control.pred_gain = auto_corr[0] / res_nrg.max(1.0);

    let mut a = [0f32; MAX_FIND_PITCH_LPC_ORDER as usize];
    k2a_flp(&mut a[..order], &refl_coef[..order]);

    bwexpander(&mut a[..order], FIND_PITCH_BANDWIDTH_EXPANSION);
    lpc_analysis_filter_flp(
        &mut res[..buf_len],
        &a[..order],
        &x[..buf_len],
        buf_len,
        order,
    );

    if encoder.common.indices.signal_type != FrameSignalType::Inactive
        && !encoder.common.first_frame_after_reset
    {
        let mut thrhld = 0.6f32;
        thrhld -= 0.004f32 * encoder.common.pitch_estimation_lpc_order as f32;
        thrhld -= 0.1f32 * encoder.common.speech_activity_q8 as f32 * (1.0 / 256.0);
        thrhld -= 0.15f32 * (i32::from(encoder.common.prev_signal_type) >> 1) as f32;
        thrhld -= 0.1f32 * encoder.common.input_tilt_q15 as f32 * (1.0 / 32768.0);

        let voiced = pitch_analysis_core_flp(
            &res[..buf_len],
            &mut control.pitch_l[..nb_subfr],
            &mut encoder.common.indices.lag_index,
            &mut encoder.common.indices.contour_index,
            &mut encoder.ltp_corr,
            encoder.common.prev_lag,
            encoder.common.pitch_estimation_threshold_q16 as f32 * (1.0 / 65536.0),
            thrhld,
            encoder.common.fs_khz,
            encoder.common.pitch_estimation_complexity,
            nb_subfr as i32,
            arch,
        ) == 0;

        encoder.common.indices.signal_type = if voiced {
            FrameSignalType::Voiced
        } else {
            FrameSignalType::Unvoiced
        };
    } else {
        control.pitch_l = [0; MAX_NB_SUBFR];
        encoder.common.indices.lag_index = 0;
        encoder.common.indices.contour_index = 0;
        encoder.ltp_corr = 0.0;
        encoder.common.indices.signal_type = FrameSignalType::Unvoiced;
    }
}
