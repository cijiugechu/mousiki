//! Port of `silk/fixed/find_pitch_lags_FIX.c`.
//!
//! This helper performs the initial LPC analysis used by the SILK encoder to
//! drive pitch estimation. It windows the input with rising and falling sine
//! tapers, computes the short-term prediction coefficients, filters the input
//! to obtain the LPC residual, and then invokes the core pitch estimator to
//! populate the lag indices for the current frame.

use crate::silk::apply_sine_window::apply_sine_window;
use crate::silk::autocorr::autocorr;
use crate::silk::bwexpander::bwexpander;
use crate::silk::encoder::control::EncoderControl;
use crate::silk::encoder::state::{
    EncoderChannelState, FIND_PITCH_LPC_WIN_MS, MAX_FIND_PITCH_LPC_ORDER, MAX_FS_KHZ,
};
use crate::silk::k2a::k2a;
use crate::silk::lpc_analysis_filter::lpc_analysis_filter;
use crate::silk::pitch_analysis_core::pitch_analysis_core;
use crate::silk::schur::silk_schur;
use crate::silk::stereo_find_predictor::div32_varq;
use crate::silk::tuning_parameters::{
    FIND_PITCH_BANDWIDTH_EXPANSION, FIND_PITCH_WHITE_NOISE_FRACTION,
};
use crate::silk::{FrameSignalType, MAX_NB_SUBFR};
use core::convert::TryFrom;

const FIND_PITCH_WHITE_NOISE_FRACTION_Q16: i32 =
    ((FIND_PITCH_WHITE_NOISE_FRACTION * (1 << 16) as f32) + 0.5) as i32;
const FIND_PITCH_BANDWIDTH_EXPANSION_Q16: i32 =
    ((FIND_PITCH_BANDWIDTH_EXPANSION * (1 << 16) as f32) + 0.5) as i32;
const THRESH_BASE_Q13: i32 = ((0.6 * (1 << 13) as f64) + 0.5) as i32;
const THRESH_LPC_ORDER_Q13: i32 = ((-0.004 * (1 << 13) as f64) + 0.5) as i32;
const THRESH_SPEECH_ACTIVITY_Q21: i32 = ((-0.1 * (1 << 21) as f64) + 0.5) as i32;
const THRESH_PREV_SIGNAL_Q13: i32 = ((-0.15 * (1 << 13) as f64) + 0.5) as i32;
const THRESH_INPUT_TILT_Q14: i32 = ((-0.1 * (1 << 14) as f64) + 0.5) as i32;
const MAX_PITCH_LPC_WIN_LENGTH: usize = FIND_PITCH_LPC_WIN_MS * MAX_FS_KHZ;

/// Mirrors `silk_find_pitch_lags_FIX`.
///
/// * `encoder` — per-channel encoder state (providing the pitch-analysis
///   buffers and side-information indices).
/// * `control` — per-frame encoder control working area.
/// * `res` — destination for the LPC residual, including the LTP history and
///   pitch look-ahead samples.
/// * `x` — input signal aligned so that the first `encoder.common.ltp_mem_length`
///   samples contain the previous frame history.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::arithmetic_side_effects
)]
pub fn find_pitch_lags(
    encoder: &mut EncoderChannelState,
    control: &mut EncoderControl,
    res: &mut [i16],
    x: &[i16],
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
    let mut wsig = [0i16; MAX_PITCH_LPC_WIN_LENGTH];
    let window = &mut wsig[..pitch_lpc_win_length];
    let (head, rest) = window.split_at_mut(la_pitch);
    apply_sine_window(head, &x_ptr[..la_pitch], 1);

    let (middle, tail) = rest.split_at_mut(pitch_lpc_win_length - la_pitch * 2);
    middle.copy_from_slice(&x_ptr[la_pitch..pitch_lpc_win_length - la_pitch]);
    apply_sine_window(
        tail,
        &x_ptr[pitch_lpc_win_length - la_pitch..pitch_lpc_win_length],
        2,
    );

    let corr_count = order + 1;
    let mut auto_corr = [0i32; MAX_FIND_PITCH_LPC_ORDER as usize + 1];
    let mut autocorr_scratch = [0i16; MAX_PITCH_LPC_WIN_LENGTH];
    autocorr(
        &mut auto_corr[..corr_count],
        window,
        corr_count,
        encoder.common.arch,
        &mut autocorr_scratch[..pitch_lpc_win_length],
    );
    auto_corr[0] = smlawb(
        auto_corr[0],
        auto_corr[0],
        FIND_PITCH_WHITE_NOISE_FRACTION_Q16,
    ) + 1;

    let mut rc_q15 = [0i16; MAX_FIND_PITCH_LPC_ORDER as usize];
    let res_nrg = silk_schur(&mut rc_q15[..order], &auto_corr[..corr_count], order);
    control.pred_gain_q16 = div32_varq(auto_corr[0], res_nrg.max(1), 16);

    let mut a_q24 = [0i32; MAX_FIND_PITCH_LPC_ORDER as usize];
    k2a(&mut a_q24[..order], &rc_q15[..order]);

    let mut a_q12 = [0i16; MAX_FIND_PITCH_LPC_ORDER as usize];
    for (dst, &src) in a_q12.iter_mut().zip(a_q24.iter()).take(order) {
        *dst = sat16(rshift(src, 12));
    }

    bwexpander(&mut a_q12[..order], FIND_PITCH_BANDWIDTH_EXPANSION_Q16);
    lpc_analysis_filter(
        &mut res[..buf_len],
        &x[..buf_len],
        &a_q12[..order],
        buf_len,
        order,
    );

    if encoder.common.indices.signal_type != FrameSignalType::Inactive
        && !encoder.common.first_frame_after_reset
    {
        let mut thr_q13 = THRESH_BASE_Q13;
        thr_q13 = smlabb(
            thr_q13,
            THRESH_LPC_ORDER_Q13,
            encoder.common.pitch_estimation_lpc_order,
        );
        thr_q13 = smlawb(
            thr_q13,
            THRESH_SPEECH_ACTIVITY_Q21,
            encoder.common.speech_activity_q8,
        );
        let prev_signal = i32::from(encoder.common.prev_signal_type) >> 1;
        thr_q13 = smlabb(thr_q13, THRESH_PREV_SIGNAL_Q13, prev_signal);
        thr_q13 = smlawb(
            thr_q13,
            THRESH_INPUT_TILT_Q14,
            encoder.common.input_tilt_q15,
        );
        thr_q13 = i32::from(sat16(thr_q13));

        let voiced = pitch_analysis_core(
            &res[..buf_len],
            &mut control.pitch_l[..nb_subfr],
            &mut encoder.common.indices.lag_index,
            &mut encoder.common.indices.contour_index,
            &mut encoder.ltp_corr_q15,
            encoder.common.prev_lag,
            encoder.common.pitch_estimation_threshold_q16,
            thr_q13,
            encoder.common.fs_khz,
            encoder.common.pitch_estimation_complexity,
            nb_subfr as i32,
            encoder.common.arch,
        ) == 0;

        encoder.common.indices.signal_type = if voiced {
            FrameSignalType::Voiced
        } else {
            FrameSignalType::Unvoiced
        };
    } else {
        for lag in control.pitch_l.iter_mut().take(nb_subfr) {
            *lag = 0;
        }
        encoder.common.indices.lag_index = 0;
        encoder.common.indices.contour_index = 0;
        encoder.ltp_corr_q15 = 0;
    }
}

#[inline]
fn rshift(value: i32, shift: u32) -> i32 {
    value >> shift
}

#[inline]
fn sat16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

#[inline]
fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(b.wrapping_mul(c))
}

#[inline]
fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let c_low = i32::from(c as i16);
    let product = (i64::from(b) * i64::from(c_low)) >> 16;
    a.wrapping_add(product as i32)
}

#[cfg(test)]
mod tests {
    use super::MAX_PITCH_LPC_WIN_LENGTH;
    use super::find_pitch_lags;
    use crate::silk::FrameSignalType;
    use crate::silk::encoder::control::EncoderControl;
    use crate::silk::encoder::state::{EncoderChannelState, FIND_PITCH_LPC_WIN_MS, MAX_FS_KHZ};
    use alloc::vec;

    fn buffer_lengths(encoder: &EncoderChannelState) -> (usize, usize) {
        let la_pitch = encoder.common.la_pitch as usize;
        let buf_len = la_pitch + encoder.common.frame_length + encoder.common.ltp_mem_length;
        (la_pitch, buf_len)
    }

    #[test]
    fn voiced_frame_updates_pitch_lags() {
        let mut encoder = EncoderChannelState::default();
        encoder.common.pitch_estimation_lpc_order = 12;
        encoder.common.pitch_estimation_threshold_q16 = (0.25f32 * 65_536.0) as i32;
        encoder.common.speech_activity_q8 = 128;
        encoder.common.indices.signal_type = FrameSignalType::Unvoiced;
        encoder.common.prev_signal_type = FrameSignalType::Unvoiced;
        encoder.common.first_frame_after_reset = false;

        let (la_pitch, buf_len) = buffer_lengths(&encoder);
        let mut x = vec![0i16; buf_len];
        let period = 80;
        for (idx, sample) in x.iter_mut().enumerate() {
            *sample = if (idx % period) < period / 2 {
                800
            } else {
                -800
            };
        }

        let mut res = vec![0i16; buf_len];
        let mut control = EncoderControl::default();

        find_pitch_lags(&mut encoder, &mut control, &mut res, &x);

        assert_eq!(encoder.common.indices.signal_type, FrameSignalType::Voiced);
        assert!(encoder.ltp_corr_q15 > 0);
        assert!(
            control.pitch_l[..encoder.common.nb_subfr]
                .iter()
                .all(|&lag| lag > 0)
        );
        assert!(control.pred_gain_q16 > 0);
        assert!(res.iter().take(buf_len - la_pitch).any(|&v| v != 0));
    }

    #[test]
    fn inactive_frames_reset_pitch_state() {
        let mut encoder = EncoderChannelState::default();
        encoder.common.pitch_estimation_lpc_order = 10;
        encoder.common.pitch_estimation_threshold_q16 = (0.2f32 * 65_536.0) as i32;
        encoder.common.indices.signal_type = FrameSignalType::Inactive;
        encoder.common.first_frame_after_reset = true;

        let (_, buf_len) = buffer_lengths(&encoder);
        let x = vec![0i16; buf_len];
        let mut res = vec![0i16; buf_len];
        let mut control = EncoderControl::default();

        find_pitch_lags(&mut encoder, &mut control, &mut res, &x);

        assert_eq!(
            encoder.common.indices.signal_type,
            FrameSignalType::Inactive
        );
        assert!(
            control.pitch_l[..encoder.common.nb_subfr]
                .iter()
                .all(|&lag| lag == 0)
        );
        assert_eq!(encoder.common.indices.lag_index, 0);
        assert_eq!(encoder.common.indices.contour_index, 0);
        assert_eq!(encoder.ltp_corr_q15, 0);
    }

    #[test]
    fn static_window_scratch_is_long_enough() {
        let max_needed = FIND_PITCH_LPC_WIN_MS * MAX_FS_KHZ;
        assert!(MAX_PITCH_LPC_WIN_LENGTH >= max_needed);
    }
}
