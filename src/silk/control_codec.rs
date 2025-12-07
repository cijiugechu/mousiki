//! Port of `silk/control_codec.c`.
//!
//! Wires together the various encoder-side control helpers. This mirrors
//! `silk_control_encoder`, refreshing the resampler, internal sampling rate,
//! complexity, packet-loss tuning, and low-bit-rate redundancy state before a
//! new packet is encoded.

use alloc::vec;
use core::cmp;

use crate::silk::control_audio_bandwidth::control_audio_bandwidth;
use crate::silk::encoder::state::{
    EncoderChannelState, EncoderShapeState, EncoderStateCommon, FIND_PITCH_LPC_WIN_MS,
    FIND_PITCH_LPC_WIN_MS_2_SF, LA_PITCH_MS, LA_SHAPE_MAX, LA_SHAPE_MS, LTP_MEM_LENGTH_MS,
    MAX_DEL_DEC_STATES, MAX_FIND_PITCH_LPC_ORDER, MAX_FRAME_LENGTH_MS, NoiseShapingQuantizerState,
    SHAPE_LPC_WIN_MAX, SUB_FRAME_LENGTH_MS,
};
use crate::silk::errors::SilkError;
use crate::silk::lp_variable_cutoff::LpState;
use crate::silk::pitch_est_tables::{
    SILK_PE_MAX_COMPLEX, SILK_PE_MID_COMPLEX, SILK_PE_MIN_COMPLEX,
};
use crate::silk::resampler::{Resampler, ResamplerInitError, silk_resampler};
use crate::silk::tables_nlsf_cb_nb_mb::SILK_NLSF_CB_NB_MB;
use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;
use crate::silk::tables_other::{SILK_UNIFORM4_ICDF, SILK_UNIFORM6_ICDF, SILK_UNIFORM8_ICDF};
use crate::silk::tables_pitch_lag::{
    PITCH_CONTOUR_10_MS_ICDF, PITCH_CONTOUR_10_MS_NB_ICDF, PITCH_CONTOUR_ICDF,
    PITCH_CONTOUR_NB_ICDF,
};
use crate::silk::tuning_parameters::WARPING_MULTIPLIER;
use crate::silk::{EncControl, FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR, MIN_LPC_ORDER};

/// Controls the SILK encoder signal path.
pub fn control_encoder(
    encoder: &mut EncoderChannelState,
    enc_control: &mut EncControl,
    allow_bw_switch: bool,
    channel_nb: i32,
    force_fs_khz: Option<i32>,
) -> Result<(), SilkError> {
    {
        let common = encoder.common_mut();
        common.use_dtx = enc_control.use_dtx != 0;
        common.use_cbr = enc_control.use_cbr != 0;
        common.api_sample_rate_hz = enc_control.api_sample_rate;
        common.max_internal_sample_rate_hz = enc_control.max_internal_sample_rate;
        common.min_internal_sample_rate_hz = enc_control.min_internal_sample_rate;
        common.desired_internal_sample_rate_hz = enc_control.desired_internal_sample_rate;
        common.use_in_band_fec = enc_control.use_in_band_fec != 0;
        common.n_channels_api = enc_control.n_channels_api;
        common.n_channels_internal = enc_control.n_channels_internal;
        common.allow_bandwidth_switch = allow_bw_switch;
        common.channel_nb = channel_nb;
    }

    if encoder.common().controlled_since_last_payload && !encoder.common().prefill_flag {
        if encoder.common().api_sample_rate_hz != encoder.common().prev_api_sample_rate_hz
            && encoder.common().fs_khz > 0
        {
            setup_resamplers(encoder, encoder.common().fs_khz)?;
        }
        return Ok(());
    }

    let mut fs_khz = control_audio_bandwidth(encoder, enc_control);
    if let Some(force) = force_fs_khz.filter(|&rate| rate != 0) {
        if !matches!(force, 8 | 12 | 16) {
            return Err(SilkError::EncFsNotSupported);
        }
        fs_khz = force;
    }

    setup_resamplers(encoder, fs_khz)?;
    setup_fs(encoder, fs_khz, enc_control.payload_size_ms)?;
    setup_complexity(encoder.common_mut(), enc_control.complexity);
    encoder.common_mut().packet_loss_perc = enc_control.packet_loss_percentage;
    setup_lbrr(encoder.common_mut(), enc_control);
    encoder.common_mut().controlled_since_last_payload = true;

    Ok(())
}

fn setup_resamplers(encoder: &mut EncoderChannelState, fs_khz: i32) -> Result<(), SilkError> {
    let (current_fs_khz, prev_api_fs_hz, api_fs_hz, nb_subfr) = {
        let common = encoder.common();
        (
            common.fs_khz,
            common.prev_api_sample_rate_hz,
            common.api_sample_rate_hz,
            common.nb_subfr,
        )
    };

    if current_fs_khz == fs_khz && prev_api_fs_hz == api_fs_hz {
        return Ok(());
    }

    if current_fs_khz == 0 {
        encoder
            .resampler_state
            .silk_resampler_init(api_fs_hz, fs_khz * 1000, true)
            .map_err(map_resampler_error)?;
    } else {
        let buf_length_ms = ((nb_subfr * SUB_FRAME_LENGTH_MS) << 1) + LA_SHAPE_MS;
        let old_buf_samples = buf_length_ms * current_fs_khz as usize;
        debug_assert!(
            old_buf_samples <= encoder.x_buf.len(),
            "pitch-analysis buffer too small"
        );

        let mut temp_resampler = Resampler::default();
        temp_resampler
            .silk_resampler_init(current_fs_khz * 1000, api_fs_hz, false)
            .map_err(map_resampler_error)?;

        let api_buf_samples = buf_length_ms * (api_fs_hz / 1000) as usize;
        let mut api_buffer = vec![0i16; api_buf_samples];
        let produced = silk_resampler(
            &mut temp_resampler,
            &mut api_buffer,
            &encoder.x_buf[..old_buf_samples],
        );
        debug_assert_eq!(produced, api_buf_samples);

        let new_buf_samples = buf_length_ms * fs_khz as usize;
        encoder
            .resampler_state
            .silk_resampler_init(api_fs_hz, fs_khz * 1000, true)
            .map_err(map_resampler_error)?;
        if new_buf_samples > encoder.x_buf.len() {
            return Err(SilkError::EncInternalError);
        }
        let produced = silk_resampler(
            &mut encoder.resampler_state,
            &mut encoder.x_buf[..new_buf_samples],
            &api_buffer,
        );
        debug_assert_eq!(produced, new_buf_samples);
    }

    encoder.common_mut().prev_api_sample_rate_hz = api_fs_hz;
    Ok(())
}

fn setup_fs(
    encoder: &mut EncoderChannelState,
    fs_khz: i32,
    packet_size_ms: i32,
) -> Result<(), SilkError> {
    {
        let common = encoder.common_mut();
        if packet_size_ms != common.packet_size_ms {
            match packet_size_ms {
                10 => {
                    common.n_frames_per_packet = 1;
                    common.nb_subfr = 2;
                    common.frame_length = (packet_size_ms * fs_khz) as usize;
                    common.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz as usize;
                    common.pitch_contour_icdf = if common.fs_khz == 8 {
                        &PITCH_CONTOUR_10_MS_NB_ICDF
                    } else {
                        &PITCH_CONTOUR_10_MS_ICDF
                    };
                }
                20 | 40 | 60 => {
                    common.n_frames_per_packet =
                        (packet_size_ms / MAX_FRAME_LENGTH_MS as i32) as usize;
                    common.nb_subfr = MAX_NB_SUBFR;
                    common.frame_length = (20 * fs_khz) as usize;
                    common.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz as usize;
                    common.pitch_contour_icdf = if common.fs_khz == 8 {
                        &PITCH_CONTOUR_NB_ICDF
                    } else {
                        &PITCH_CONTOUR_ICDF
                    };
                }
                _ => return Err(SilkError::EncPacketSizeNotSupported),
            }
            common.packet_size_ms = packet_size_ms;
            common.target_rate_bps = 0;
        }
    }

    debug_assert!(matches!(fs_khz, 8 | 12 | 16));
    if encoder.common().fs_khz != fs_khz {
        encoder.shape_state = EncoderShapeState::default();
        encoder.shape_state.last_gain_index = 10;
        encoder.common.nsq_state = NoiseShapingQuantizerState::default();
        encoder.common.nsq_state.lag_prev = 100;
        encoder.common.nsq_state.prev_gain_q16 = 65_536;
        encoder.lp_state = LpState::default();

        let nb_subfr = encoder.common().nb_subfr;
        {
            let common = encoder.common_mut();
            common.prev_nlsf_q15 = [0; MAX_LPC_ORDER];
            common.input_buf_ix = 0;
            common.n_frames_encoded = 0;
            common.target_rate_bps = 0;
            common.prev_lag = 100;
            common.first_frame_after_reset = true;
            common.prev_signal_type = FrameSignalType::Inactive;
            common.fs_khz = fs_khz;
            common.pitch_contour_icdf = match (fs_khz, nb_subfr == MAX_NB_SUBFR) {
                (8, true) => &PITCH_CONTOUR_NB_ICDF,
                (8, false) => &PITCH_CONTOUR_10_MS_NB_ICDF,
                (_, true) => &PITCH_CONTOUR_ICDF,
                (_, false) => &PITCH_CONTOUR_10_MS_ICDF,
            };
            if fs_khz == 8 || fs_khz == 12 {
                common.predict_lpc_order = MIN_LPC_ORDER;
                common.ps_nlsf_cb = &SILK_NLSF_CB_NB_MB;
            } else {
                common.predict_lpc_order = MAX_LPC_ORDER;
                common.ps_nlsf_cb = &SILK_NLSF_CB_WB;
            }
            common.subfr_length = SUB_FRAME_LENGTH_MS * fs_khz as usize;
            common.frame_length = common.subfr_length * nb_subfr;
            common.ltp_mem_length = LTP_MEM_LENGTH_MS * fs_khz as usize;
            common.la_pitch = LA_PITCH_MS as i32 * fs_khz;
            common.max_pitch_lag = 18 * fs_khz;
            common.pitch_lpc_win_length = if nb_subfr == MAX_NB_SUBFR {
                FIND_PITCH_LPC_WIN_MS * fs_khz as usize
            } else {
                FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz as usize
            };
            common.pitch_lag_low_bits_icdf = match fs_khz {
                16 => &SILK_UNIFORM8_ICDF,
                12 => &SILK_UNIFORM6_ICDF,
                _ => &SILK_UNIFORM4_ICDF,
            };
        }
    }

    debug_assert_eq!(
        encoder.common().subfr_length * encoder.common().nb_subfr,
        encoder.common().frame_length
    );
    Ok(())
}

fn setup_complexity(common: &mut EncoderStateCommon, complexity: i32) {
    debug_assert!((0..=10).contains(&complexity));
    let coeff = |val: f32| ((val * ((1 << 16) as f32)) + 0.5) as i32;
    if complexity < 1 {
        assign_complexity(
            common,
            SILK_PE_MIN_COMPLEX,
            coeff(0.8),
            6,
            12,
            3,
            1,
            false,
            2,
            0,
        );
    } else if complexity < 2 {
        assign_complexity(
            common,
            SILK_PE_MID_COMPLEX,
            coeff(0.76),
            8,
            14,
            5,
            1,
            false,
            3,
            0,
        );
    } else if complexity < 3 {
        assign_complexity(
            common,
            SILK_PE_MIN_COMPLEX,
            coeff(0.8),
            6,
            12,
            3,
            2,
            false,
            2,
            0,
        );
    } else if complexity < 4 {
        assign_complexity(
            common,
            SILK_PE_MID_COMPLEX,
            coeff(0.76),
            8,
            14,
            5,
            2,
            false,
            4,
            0,
        );
    } else if complexity < 6 {
        let warp = coeff(WARPING_MULTIPLIER) * common.fs_khz;
        assign_complexity(
            common,
            SILK_PE_MID_COMPLEX,
            coeff(0.74),
            10,
            16,
            5,
            2,
            true,
            6,
            warp,
        );
    } else if complexity < 8 {
        let warp = coeff(WARPING_MULTIPLIER) * common.fs_khz;
        assign_complexity(
            common,
            SILK_PE_MID_COMPLEX,
            coeff(0.72),
            12,
            20,
            5,
            3,
            true,
            8,
            warp,
        );
    } else {
        let warp = coeff(WARPING_MULTIPLIER) * common.fs_khz;
        assign_complexity(
            common,
            SILK_PE_MAX_COMPLEX,
            coeff(0.7),
            16,
            24,
            5,
            MAX_DEL_DEC_STATES,
            true,
            16,
            warp,
        );
    }

    if common.pitch_estimation_lpc_order > common.predict_lpc_order as i32 {
        common.pitch_estimation_lpc_order = common.predict_lpc_order as i32;
    }
    common.shape_win_length = (SUB_FRAME_LENGTH_MS as i32 * common.fs_khz) + 2 * common.la_shape;
    common.complexity = complexity;

    debug_assert!(common.pitch_estimation_lpc_order <= MAX_FIND_PITCH_LPC_ORDER);
    debug_assert!(common.shaping_lpc_order <= crate::silk::MAX_SHAPE_LPC_ORDER as i32);
    debug_assert!(common.n_states_delayed_decision <= MAX_DEL_DEC_STATES);
    debug_assert!(common.warping_q16 <= 32_767);
    debug_assert!(common.la_shape <= LA_SHAPE_MAX as i32);
    debug_assert!(common.shape_win_length <= SHAPE_LPC_WIN_MAX);
}

#[allow(clippy::too_many_arguments)]
fn assign_complexity(
    common: &mut EncoderStateCommon,
    pitch_complexity: usize,
    threshold_q16: i32,
    pitch_lpc_order: i32,
    shaping_lpc_order: i32,
    la_shape_mult: i32,
    n_states: i32,
    use_interpolation: bool,
    survivors: i32,
    warping_q16: i32,
) {
    common.pitch_estimation_complexity = pitch_complexity as i32;
    common.pitch_estimation_threshold_q16 = threshold_q16;
    common.pitch_estimation_lpc_order = pitch_lpc_order;
    common.shaping_lpc_order = shaping_lpc_order;
    common.la_shape = la_shape_mult * common.fs_khz;
    common.n_states_delayed_decision = n_states;
    common.use_interpolated_nlsfs = use_interpolation;
    common.nlsf_msvq_survivors = survivors;
    common.warping_q16 = warping_q16;
}

fn setup_lbrr(common: &mut EncoderStateCommon, enc_control: &EncControl) {
    let previous = common.lbrr_enabled;
    common.lbrr_enabled = enc_control.lbrr_coded != 0;
    if common.lbrr_enabled {
        if previous {
            let reduction = ((i64::from(common.packet_loss_perc) * 13_107) >> 16) as i32;
            common.lbrr_gain_increases = cmp::max(7 - reduction, 3);
        } else {
            common.lbrr_gain_increases = 7;
        }
    }
}

fn map_resampler_error(_: ResamplerInitError) -> SilkError {
    SilkError::EncInternalError
}
