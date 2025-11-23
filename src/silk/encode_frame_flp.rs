//! Port of `silk/float/encode_frame_FLP.c`.
//!
//! Mirrors the floating-point encoder frame driver, including the VAD helper
//! and optional in-band low-bitrate redundancy (LBRR) path.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use crate::range::RangeEncoder;
use crate::silk::decode_indices::{ConditionalCoding, SideInfoIndices};
use crate::silk::encode_indices::EncoderIndicesState;
use crate::silk::encode_pulses::silk_encode_pulses;
use crate::silk::encoder::control_flp::EncoderControlFlp;
use crate::silk::encoder::state::{
    EncoderStateCommon, LA_PITCH_MS, LA_SHAPE_MS, MAX_FRAME_LENGTH, MAX_FS_KHZ,
    NoiseShapingQuantizerState,
};
use crate::silk::encoder::state_flp::EncoderStateFlp;
use crate::silk::errors::SilkError;
use crate::silk::find_pitch_lags_flp::find_pitch_lags_flp;
use crate::silk::find_pred_coefs_flp::find_pred_coefs_flp;
use crate::silk::gain_quant::{silk_gains_dequant, silk_gains_id, silk_gains_quant};
use crate::silk::noise_shape_analysis_flp::noise_shape_analysis_flp;
use crate::silk::process_gains_flp::process_gains_flp;
use crate::silk::sigproc_flp::silk_short2float_array;
use crate::silk::tuning_parameters::{
    LBRR_SPEECH_ACTIVITY_THRES, MAX_CONSECUTIVE_DTX, NB_SPEECH_FRAMES_BEFORE_DTX,
    SPEECH_ACTIVITY_DTX_THRES, VAD_NO_ACTIVITY,
};
use crate::silk::vad::compute_speech_activity_q8_common;
use crate::silk::wrappers_flp::silk_nsq_wrapper_flp;
use crate::silk::{
    FrameQuantizationOffsetType, FrameSignalType, MAX_FRAMES_PER_PACKET, MAX_NB_SUBFR,
};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp;

const LA_PITCH_MAX: usize = LA_PITCH_MS * MAX_FS_KHZ;

#[inline]
fn q8_from_float(value: f32) -> i32 {
    ((value * 256.0) + 0.5) as i32
}

fn quant_offset_to_int(offset: FrameQuantizationOffsetType) -> i32 {
    match offset {
        FrameQuantizationOffsetType::Low => 0,
        FrameQuantizationOffsetType::High => 1,
    }
}

fn encode_indices_with_state(
    common: &EncoderStateCommon,
    range_encoder: &mut RangeEncoder,
    indices: &SideInfoIndices,
    coding: ConditionalCoding,
    encode_lbrr: bool,
) -> (FrameSignalType, i16) {
    let mut indices_state = EncoderIndicesState {
        nb_subfr: common.nb_subfr,
        fs_khz: common.fs_khz,
        predict_lpc_order: common.predict_lpc_order,
        nlsf_codebook: common.ps_nlsf_cb,
        pitch_lag_low_bits_icdf: common.pitch_lag_low_bits_icdf,
        pitch_contour_icdf: common.pitch_contour_icdf,
        prev_signal_type: common.ec_prev_signal_type,
        prev_lag_index: common.ec_prev_lag_index,
    };
    indices_state.encode_indices(range_encoder, indices, coding, encode_lbrr);
    (indices_state.prev_signal_type, indices_state.prev_lag_index)
}

/// Voice activity detection helper used by the floating-point encoder path.
pub fn silk_encode_do_vad_flp(encoder: &mut EncoderStateFlp, activity: i32) {
    let frame = encoder.common.frame_length;
    assert!(
        encoder.common.n_frames_encoded < MAX_FRAMES_PER_PACKET,
        "n_frames_encoded exceeds MAX_FRAMES_PER_PACKET"
    );
    assert!(
        frame < encoder.common.input_buf.len(),
        "input buffer shorter than expected frame"
    );

    let threshold_q8 = q8_from_float(SPEECH_ACTIVITY_DTX_THRES);
    let input: Vec<i16> = encoder.common.input_buf[1..frame + 1].to_vec();
    let speech_activity = compute_speech_activity_q8_common(&mut encoder.common, &input);
    encoder.common.speech_activity_q8 = i32::from(speech_activity);
    if activity == VAD_NO_ACTIVITY && encoder.common.speech_activity_q8 >= threshold_q8 {
        encoder.common.speech_activity_q8 = threshold_q8 - 1;
    }

    if encoder.common.speech_activity_q8 < threshold_q8 {
        encoder.common.indices.signal_type = FrameSignalType::Inactive;
        encoder.common.no_speech_counter += 1;
        if encoder.common.no_speech_counter <= NB_SPEECH_FRAMES_BEFORE_DTX {
            encoder.common.in_dtx = false;
        } else if encoder.common.no_speech_counter
            > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX
        {
            encoder.common.no_speech_counter = NB_SPEECH_FRAMES_BEFORE_DTX;
            encoder.common.in_dtx = false;
        }
        encoder.common.vad_flags[encoder.common.n_frames_encoded] = false;
    } else {
        encoder.common.no_speech_counter = 0;
        encoder.common.in_dtx = false;
        encoder.common.indices.signal_type = FrameSignalType::Unvoiced;
        encoder.common.vad_flags[encoder.common.n_frames_encoded] = true;
    }
}

/// Low-bitrate redundancy (LBRR) side-channel encoding.
fn silk_lbrr_encode_flp(
    encoder: &mut EncoderStateFlp,
    control: &mut EncoderControlFlp,
    xfw: &[f32],
    cond_coding: ConditionalCoding,
) {
    let frame_idx = encoder.common.n_frames_encoded;
    assert!(
        frame_idx < MAX_FRAMES_PER_PACKET,
        "frame index exceeds MAX_FRAMES_PER_PACKET"
    );
    if !(encoder.common.lbrr_enabled
        && encoder.common.speech_activity_q8 > q8_from_float(LBRR_SPEECH_ACTIVITY_THRES))
    {
        return;
    }

    encoder.common.lbrr_flags[frame_idx] = true;

    let mut nsq_lbrr = encoder.common.nsq_state.clone();
    let mut indices_lbrr = encoder.common.indices.clone();

    let temp_gains = control.gains;

    if frame_idx == 0 || !encoder.common.lbrr_flags[frame_idx - 1] {
        encoder.common.lbrr_prev_last_gain_index = encoder.shape_state.last_gain_index;
        let updated = i32::from(indices_lbrr.gains_indices[0]) + encoder.common.lbrr_gain_increases;
        indices_lbrr.gains_indices[0] =
            cmp::min(updated, crate::silk::tables_gain::N_LEVELS_QGAIN as i32 - 1) as i8;
    }

    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    let conditional = matches!(cond_coding, ConditionalCoding::Conditional);
    silk_gains_dequant(
        &mut gains_q16,
        &indices_lbrr.gains_indices,
        &mut encoder.common.lbrr_prev_last_gain_index,
        conditional,
    );

    for (gain, &quant_q16) in control.gains.iter_mut().zip(gains_q16.iter()) {
        *gain = quant_q16 as f32 * (1.0 / 65_536.0);
    }

    let mut pulses_lbrr = vec![0i8; encoder.common.frame_length];
    let common_snapshot = encoder.common.clone();
    silk_nsq_wrapper_flp(
        &common_snapshot,
        control,
        &mut indices_lbrr,
        &mut nsq_lbrr,
        &mut pulses_lbrr,
        xfw,
    );

    encoder.common.indices_lbrr[frame_idx] = indices_lbrr;
    encoder.common.pulses_lbrr[frame_idx][..encoder.common.frame_length]
        .copy_from_slice(&pulses_lbrr);

    control.gains = temp_gains;
}

/// Floating-point frame encoder (mirror of `silk_encode_frame_FLP`).
#[allow(clippy::too_many_lines)]
pub fn silk_encode_frame_flp(
    encoder: &mut EncoderStateFlp,
    bytes_out: &mut i32,
    range_encoder: &mut RangeEncoder,
    cond_coding: ConditionalCoding,
    max_bits: i32,
    use_cbr: bool,
) -> SilkError {
    let mut enc_ctrl = EncoderControlFlp::default();
    let nb_subfr = encoder.common.nb_subfr;
    let frame_length = encoder.common.frame_length;

    let mut res_pitch = [0f32; 2 * MAX_FRAME_LENGTH + LA_PITCH_MAX];
    let mut gain_lock = [false; MAX_NB_SUBFR];
    let mut best_gain_mult = [0i32; MAX_NB_SUBFR];
    let mut best_sum = [0i32; MAX_NB_SUBFR];

    let bits_margin = if use_cbr { 5 } else { max_bits / 4 };

    encoder.common.indices.seed = (encoder.common.frame_counter & 3) as i8;
    encoder.common.frame_counter = encoder.common.frame_counter.wrapping_add(1);

    let la_shape_samples = LA_SHAPE_MS * encoder.common.fs_khz as usize;
    let ltp_offset = encoder.common.ltp_mem_length;
    {
        let x_frame = &mut encoder.x_buf[ltp_offset..];
        encoder
            .common
            .lp_state
            .lp_variable_cutoff(&mut encoder.common.input_buf[1..frame_length + 1]);

        silk_short2float_array(
            &mut x_frame[la_shape_samples..la_shape_samples + frame_length],
            &encoder.common.input_buf[1..frame_length + 1],
        );

        for i in 0..8 {
            let jitter = (1 - (i & 2)) as f32 * 1e-6;
            let offset = la_shape_samples + i * (frame_length >> 3);
            x_frame[offset] += jitter;
        }
    }

    if !encoder.common.prefill_flag {
        let buf_len = ltp_offset
            + frame_length
            + usize::try_from(encoder.common.la_pitch).expect("la_pitch must be non-negative");
        debug_assert!(buf_len <= res_pitch.len());
        debug_assert!(ltp_offset + buf_len <= encoder.x_buf.len());
        let x_pitch: Vec<f32> = encoder.x_buf[ltp_offset..ltp_offset + buf_len].to_vec();
        find_pitch_lags_flp(
            encoder,
            &mut enc_ctrl,
            &mut res_pitch[..buf_len],
            &x_pitch,
            encoder.common.arch,
        );
        let res_pitch_frame = &res_pitch[ltp_offset..];
        let x_frame_full: Vec<f32> = encoder.x_buf[ltp_offset..].to_vec();
        noise_shape_analysis_flp(encoder, &mut enc_ctrl, res_pitch_frame, &x_frame_full);
        find_pred_coefs_flp(
            encoder,
            &mut enc_ctrl,
            res_pitch_frame,
            &x_frame_full,
            cond_coding,
        );
        process_gains_flp(encoder, &mut enc_ctrl, cond_coding);
        let frame_slice: Vec<f32> = encoder.x_buf
            [ltp_offset + la_shape_samples..ltp_offset + la_shape_samples + frame_length]
            .to_vec();
        silk_lbrr_encode_flp(encoder, &mut enc_ctrl, &frame_slice, cond_coding);

        let max_iter = 6;
        let mut found_upper = false;
        let mut found_lower = false;
        let mut n_bits_lower = 0;
        let mut n_bits_upper = 0;
        let mut gain_mult_lower = 0;
        let mut gain_mult_upper = 0;
        let mut gains_id_lower = -1;
        let mut gains_id_upper = -1;
        let mut gain_mult_q8: i32 = 1 << 8;
        let mut last_gain_index_copy2 = 0i8;
        let mut gains_id = silk_gains_id(&encoder.common.indices.gains_indices[..nb_subfr]);

        let initial_range = range_encoder.clone();
        let initial_nsq = encoder.common.nsq_state.clone();
        let seed_copy = encoder.common.indices.seed;
        let ec_prev_lag_index_copy = encoder.common.ec_prev_lag_index;
        let ec_prev_signal_type_copy = encoder.common.ec_prev_signal_type;

        let mut best_range_state: Option<RangeEncoder> = None;
        let mut best_nsq_state: Option<NoiseShapingQuantizerState> = None;
        let mut best_indices: Option<SideInfoIndices> = None;
        let mut best_pulses: Option<[i8; MAX_FRAME_LENGTH]> = None;

        let mut iter = 0;
        loop {
            let mut n_bits;
            if gains_id == gains_id_lower {
                n_bits = n_bits_lower;
            } else if gains_id == gains_id_upper {
                n_bits = n_bits_upper;
            } else {
                if iter > 0 {
                    *range_encoder = initial_range.clone();
                    encoder.common.nsq_state = initial_nsq.clone();
                    encoder.common.indices.seed = seed_copy;
                    encoder.common.ec_prev_lag_index = ec_prev_lag_index_copy;
                    encoder.common.ec_prev_signal_type = ec_prev_signal_type_copy;
                }

                let backup_range = if iter == max_iter && !found_lower {
                    Some(range_encoder.clone())
                } else {
                    None
                };

                let common_snapshot = encoder.common.clone();
                silk_nsq_wrapper_flp(
                    &common_snapshot,
                    &enc_ctrl,
                    &mut encoder.common.indices,
                    &mut encoder.common.nsq_state,
                    &mut encoder.common.pulses[..frame_length],
                    frame_slice.as_slice(),
                );

                let (prev_sig_type, prev_lag_index) = encode_indices_with_state(
                    &encoder.common,
                    range_encoder,
                    &encoder.common.indices,
                    cond_coding,
                    false,
                );
                encoder.common.ec_prev_signal_type = prev_sig_type;
                encoder.common.ec_prev_lag_index = prev_lag_index;
                silk_encode_pulses(
                    range_encoder,
                    i32::from(encoder.common.indices.signal_type),
                    quant_offset_to_int(encoder.common.indices.quant_offset_type),
                    &mut encoder.common.pulses[..frame_length],
                    frame_length,
                );

                n_bits = range_encoder.tell();

                if iter == max_iter
                    && !found_lower
                    && n_bits > max_bits
                    && let Some(backup) = backup_range
                {
                    *range_encoder = backup;
                    encoder.shape_state.last_gain_index = enc_ctrl.last_gain_index_prev;
                    for idx in encoder
                        .common
                        .indices
                        .gains_indices
                        .iter_mut()
                        .take(nb_subfr)
                    {
                        *idx = 4;
                    }
                    if !matches!(cond_coding, ConditionalCoding::Conditional) {
                        encoder.common.indices.gains_indices[0] = enc_ctrl.last_gain_index_prev;
                    }
                    encoder.common.ec_prev_lag_index = ec_prev_lag_index_copy;
                    encoder.common.ec_prev_signal_type = ec_prev_signal_type_copy;
                    encoder.common.pulses[..frame_length].fill(0);

                    let (prev_sig_type, prev_lag_index) = encode_indices_with_state(
                        &encoder.common,
                        range_encoder,
                        &encoder.common.indices,
                        cond_coding,
                        false,
                    );
                    encoder.common.ec_prev_signal_type = prev_sig_type;
                    encoder.common.ec_prev_lag_index = prev_lag_index;
                    silk_encode_pulses(
                        range_encoder,
                        i32::from(encoder.common.indices.signal_type),
                        quant_offset_to_int(encoder.common.indices.quant_offset_type),
                        &mut encoder.common.pulses[..frame_length],
                        frame_length,
                    );
                    n_bits = range_encoder.tell();
                }

                if !use_cbr && iter == 0 && n_bits <= max_bits {
                    break;
                }
            }

            if iter == max_iter {
                if found_lower
                    && (gains_id == gains_id_lower || n_bits > max_bits)
                    && let (Some(saved_range), Some(saved_nsq), Some(saved_indices), Some(pulses)) = (
                        best_range_state.take(),
                        best_nsq_state.take(),
                        best_indices.take(),
                        best_pulses.take(),
                    )
                {
                    *range_encoder = saved_range;
                    encoder.common.nsq_state = saved_nsq;
                    encoder.common.indices = saved_indices;
                    encoder.common.pulses = pulses;
                    encoder.shape_state.last_gain_index = last_gain_index_copy2;
                }
                break;
            }

            if n_bits > max_bits {
                if !found_lower && iter >= 2 {
                    enc_ctrl.lambda = (enc_ctrl.lambda * 1.5).max(1.5);
                    encoder.common.indices.quant_offset_type = FrameQuantizationOffsetType::Low;
                    found_upper = false;
                    gains_id_upper = -1;
                } else {
                    found_upper = true;
                    n_bits_upper = n_bits;
                    gain_mult_upper = gain_mult_q8;
                    gains_id_upper = gains_id;
                }
            } else if n_bits < max_bits - bits_margin {
                found_lower = true;
                n_bits_lower = n_bits;
                gain_mult_lower = gain_mult_q8;
                if gains_id != gains_id_lower {
                    gains_id_lower = gains_id;
                    best_range_state = Some(range_encoder.clone());
                    best_nsq_state = Some(encoder.common.nsq_state.clone());
                    best_indices = Some(encoder.common.indices.clone());
                    best_pulses = Some(encoder.common.pulses);
                    last_gain_index_copy2 = encoder.shape_state.last_gain_index;
                }
            } else {
                break;
            }

            if !found_lower && n_bits > max_bits {
                for i in 0..nb_subfr {
                    let mut sum = 0;
                    let start = i * encoder.common.subfr_length;
                    let end = start + encoder.common.subfr_length;
                    for &pulse in encoder.common.pulses[start..end].iter() {
                        sum += i32::from(pulse.abs());
                    }
                    if iter == 0 || (sum < best_sum[i] && !gain_lock[i]) {
                        best_sum[i] = sum;
                        best_gain_mult[i] = gain_mult_q8;
                    } else {
                        gain_lock[i] = true;
                    }
                }
            }

            if found_lower && found_upper {
                if n_bits_upper == n_bits_lower {
                    break;
                }
                let diff = gain_mult_upper - gain_mult_lower;
                gain_mult_q8 = gain_mult_lower
                    + diff * (max_bits - n_bits_lower) / (n_bits_upper - n_bits_lower);
                let upper_limit = gain_mult_lower + (diff >> 2);
                if gain_mult_q8 > upper_limit {
                    gain_mult_q8 = upper_limit;
                } else {
                    let lower_limit = gain_mult_upper - (diff >> 2);
                    if gain_mult_q8 < lower_limit {
                        gain_mult_q8 = lower_limit;
                    }
                }
            } else if n_bits > max_bits {
                gain_mult_q8 = i32::min(1024, gain_mult_q8 * 3 / 2);
            } else {
                gain_mult_q8 = i32::max(64, gain_mult_q8 * 4 / 5);
            }

            let mut p_gains_q16 = [0i32; MAX_NB_SUBFR];
            for i in 0..nb_subfr {
                let tmp = if gain_lock[i] {
                    best_gain_mult[i]
                } else {
                    gain_mult_q8
                };
                let prod = ((i64::from(enc_ctrl.gains_unq_q16[i]) * i64::from(tmp) + (1 << 15))
                    >> 16) as i32;
                p_gains_q16[i] = prod.saturating_mul(1 << 8);
            }

            encoder.shape_state.last_gain_index = enc_ctrl.last_gain_index_prev;
            let conditional = matches!(cond_coding, ConditionalCoding::Conditional);
            silk_gains_quant(
                &mut encoder.common.indices.gains_indices[..nb_subfr],
                &mut p_gains_q16[..nb_subfr],
                &mut encoder.shape_state.last_gain_index,
                conditional,
            );
            gains_id = silk_gains_id(&encoder.common.indices.gains_indices[..nb_subfr]);
            for (gain, &quant_q16) in enc_ctrl
                .gains
                .iter_mut()
                .zip(p_gains_q16.iter())
                .take(nb_subfr)
            {
                *gain = quant_q16 as f32 * (1.0 / 65_536.0);
            }

            iter += 1;
        }
    }

    let tail = encoder.common.ltp_mem_length + LA_SHAPE_MS * encoder.common.fs_khz as usize;
    encoder
        .x_buf
        .copy_within(frame_length..frame_length + tail, 0);

    if encoder.common.prefill_flag {
        *bytes_out = 0;
        return SilkError::NoError;
    }

    encoder.common.prev_lag = enc_ctrl.pitch_l[nb_subfr - 1];
    encoder.common.prev_signal_type = encoder.common.indices.signal_type;
    encoder.common.first_frame_after_reset = false;
    *bytes_out = (range_encoder.tell() + 7) >> 3;

    SilkError::NoError
}
