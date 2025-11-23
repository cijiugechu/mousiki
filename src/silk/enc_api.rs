#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

//! Port of `silk/enc_API.c`.
//!
//! Exposes the top-level SILK encoder entry points that wrap the per-channel
//! encoder state, resamplers, stereo helpers, and range encoder so that callers
//! can drive the Rust port using the same control structure as the reference
//! implementation.

use alloc::vec;
use core::cmp::{max, min};

use crate::range::RangeEncoder;
use crate::silk::check_control_input::EncControl;
use crate::silk::control_codec::control_encoder;
use crate::silk::control_snr::control_snr;
use crate::silk::decode_indices::ConditionalCoding;
use crate::silk::encode_frame::{silk_encode_do_vad, silk_encode_frame};
use crate::silk::encode_indices::EncoderIndicesState;
use crate::silk::encode_pulses::silk_encode_pulses;
use crate::silk::encoder::state::{
    Encoder, EncoderChannelState, EncoderShapeState, NoiseShapingQuantizerState,
    ENCODER_NUM_CHANNELS,
};
use crate::silk::errors::SilkError;
use crate::silk::hp_variable_cutoff::hp_variable_cutoff;
use crate::silk::init_encoder::init_encoder as init_channel;
use crate::silk::resampler::silk_resampler;
use crate::silk::stereo_encode_pred::{stereo_encode_mid_only, stereo_encode_pred};
use crate::silk::stereo_lr_to_ms::{StereoConversionResult, StereoEncState};
use crate::silk::tables_other::{SILK_LBRR_FLAGS_ICDF_PTR, SILK_QUANTIZATION_OFFSETS_Q10};
use crate::silk::tuning_parameters::{
    BITRESERVOIR_DECAY_TIME_MS, MAX_BANDWIDTH_SWITCH_DELAY_MS, SPEECH_ACTIVITY_DTX_THRES,
};
use crate::silk::{FrameQuantizationOffsetType, FrameSignalType, MAX_FRAMES_PER_PACKET, MAX_LPC_ORDER};

/// Prefill behaviour used by [`silk_encode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    /// Regular encode path.
    None,
    /// Prefill the encoder without emitting a payload.
    Prefill,
    /// Prefill while preserving the variable LP state across the reset.
    PrefillWithState,
}

impl PrefillMode {
    const fn is_active(self) -> bool {
        !matches!(self, Self::None)
    }

    const fn keep_low_pass_state(self) -> bool {
        matches!(self, Self::PrefillWithState)
    }
}

/// Mirrors `silk_InitEncoder`.
pub fn silk_init_encoder(
    encoder: &mut Encoder,
    arch: i32,
    status: &mut EncControl,
) -> Result<(), SilkError> {
    *encoder = Encoder::default();
    for channel in encoder.state_fxx.iter_mut().take(ENCODER_NUM_CHANNELS) {
        init_channel(channel, arch)?;
    }
    encoder.n_channels_api = 1;
    encoder.n_channels_internal = 1;
    query_encoder(encoder, status);
    Ok(())
}

/// Populates the encoder status structure from the current channel state.
fn query_encoder(encoder: &Encoder, status: &mut EncControl) {
    let common = &encoder.state_fxx[0].common;
    status.n_channels_api = encoder.n_channels_api;
    status.n_channels_internal = encoder.n_channels_internal;
    status.api_sample_rate = common.api_sample_rate_hz;
    status.max_internal_sample_rate = common.max_internal_sample_rate_hz;
    status.min_internal_sample_rate = common.min_internal_sample_rate_hz;
    status.desired_internal_sample_rate = common.desired_internal_sample_rate_hz;
    status.payload_size_ms = common.packet_size_ms;
    status.bit_rate = common.target_rate_bps;
    status.packet_loss_percentage = common.packet_loss_perc;
    status.complexity = common.complexity;
    status.use_in_band_fec = i32::from(common.use_in_band_fec);
    status.use_dtx = i32::from(common.use_dtx);
    status.use_cbr = i32::from(common.use_cbr);
    status.internal_sample_rate = common.fs_khz * 1000;
    status.allow_bandwidth_switch = common.allow_bandwidth_switch;
    status.in_wb_mode_without_variable_lp =
        common.fs_khz == 16 && common.lp_state.mode == 0;
    status.stereo_width_q14 = i32::from(encoder.stereo_state.smth_width_q14);
    status.signal_type = common.indices.signal_type;
    status.offset = i32::from(
        SILK_QUANTIZATION_OFFSETS_Q10
            [common.indices.signal_type as usize >> 1]
            [quant_offset_as_i32(common.indices.quant_offset_type) as usize],
    );
}

/// Rust translation of `silk_Encode`.
#[allow(clippy::too_many_arguments)]
pub fn silk_encode(
    encoder: &mut Encoder,
    control: &mut EncControl,
    samples_in: &[i16],
    range_encoder: &mut RangeEncoder,
    n_bytes_out: &mut i32,
    prefill: PrefillMode,
    activity: i32,
) -> Result<(), SilkError> {
    *n_bytes_out = 0;

    let mut remaining_samples = i32::try_from(samples_in.len())
        .map_err(|_| SilkError::EncInputInvalidNoOfSamples)?;

    if control.reduced_dependency {
        for channel in encoder.state_fxx.iter_mut() {
            channel.common_mut().first_frame_after_reset = true;
        }
    }
    for channel in encoder.state_fxx.iter_mut() {
        channel.common_mut().n_frames_encoded = 0;
    }

    control.check_control_input()?;
    control.switch_ready = false;

    let internal_channels = usize::try_from(control.n_channels_internal)
        .unwrap_or(ENCODER_NUM_CHANNELS)
        .min(ENCODER_NUM_CHANNELS);
    let api_channels = usize::try_from(control.n_channels_api)
        .unwrap_or(ENCODER_NUM_CHANNELS)
        .min(ENCODER_NUM_CHANNELS);

    if control.n_channels_internal > encoder.n_channels_internal {
        for n in encoder.n_channels_internal as usize..internal_channels {
            let arch = encoder.state_fxx[0].common.arch;
            init_channel(&mut encoder.state_fxx[n], arch)?;
        }
        encoder.stereo_state = StereoEncState::default();
        encoder.prev_decode_only_middle = false;
    }

    let transition = control.payload_size_ms != encoder.state_fxx[0].common.packet_size_ms
        || encoder.n_channels_internal != control.n_channels_internal;

    encoder.n_channels_api = control.n_channels_api;
    encoder.n_channels_internal = control.n_channels_internal;

    let api_sample_rate = control.api_sample_rate;
    let n_blocks_of_10ms = (100 * remaining_samples) / api_sample_rate;
    let tot_blocks = if n_blocks_of_10ms > 1 {
        n_blocks_of_10ms >> 1
    } else {
        1
    };
    let mut curr_block = 0;

    let mut tmp_payload_size_ms = 0;
    let mut tmp_complexity = 0;

    if prefill.is_active() {
        if n_blocks_of_10ms != 1 {
            return Err(SilkError::EncInputInvalidNoOfSamples);
        }

        let mut saved_lp = encoder.state_fxx[0].common.lp_state.clone();
        if prefill.keep_low_pass_state() {
            saved_lp.saved_fs_khz = encoder.state_fxx[0].common.fs_khz;
        }

        for channel in encoder.state_fxx.iter_mut().take(internal_channels) {
            let arch = channel.common.arch;
            init_channel(channel, arch)?;
            if prefill.keep_low_pass_state() {
                channel.common.lp_state = saved_lp.clone();
            }
            channel.common.controlled_since_last_payload = false;
            channel.common.prefill_flag = true;
        }
        tmp_payload_size_ms = control.payload_size_ms;
        control.payload_size_ms = 10;
        tmp_complexity = control.complexity;
        control.complexity = 0;
    } else {
        if n_blocks_of_10ms * api_sample_rate != 100 * remaining_samples || remaining_samples < 0 {
            return Err(SilkError::EncInputInvalidNoOfSamples);
        }
        if 1000 * remaining_samples > control.payload_size_ms * api_sample_rate {
            return Err(SilkError::EncInputInvalidNoOfSamples);
        }
    }

    for n in 0..internal_channels {
        let force_fs_khz = if n == 1 {
            Some(encoder.state_fxx[0].common.fs_khz)
        } else {
            None
        };
        control_encoder(
            &mut encoder.state_fxx[n],
            control,
            encoder.allow_bandwidth_switch,
            n as i32,
            force_fs_khz,
        )?;

        if encoder.state_fxx[n].common.first_frame_after_reset || transition {
            encoder.state_fxx[n].common.lbrr_flags = [false; MAX_FRAMES_PER_PACKET];
        }
        encoder.state_fxx[n].common.in_dtx = encoder.state_fxx[n].common.use_dtx;
    }

    debug_assert!(
        control.n_channels_internal == 1
            || encoder.state_fxx[0].common.fs_khz == encoder.state_fxx[1].common.fs_khz
    );

    let fs_khz = encoder.state_fxx[0].common.fs_khz;
    let frame_length = encoder.state_fxx[0].common.frame_length;

    let n_samples_to_buffer_max = (10 * n_blocks_of_10ms * fs_khz) as usize;
    let n_samples_from_input_max = (n_samples_to_buffer_max * encoder.state_fxx[0].common.api_sample_rate_hz as usize)
        / ((fs_khz * 1000) as usize);
    let mut buf = vec![0i16; n_samples_from_input_max];
    let mut input_offset = 0usize;
    let n_channels_api = api_channels;

    loop {
        let mut curr_n_bits_used_lbrr = 0;
        let samples_to_buffer = encoder.state_fxx[0]
            .common
            .frame_length
            .saturating_sub(encoder.state_fxx[0].common.input_buf_ix)
            .min(n_samples_to_buffer_max);
        let samples_from_input = (samples_to_buffer * encoder.state_fxx[0].common.api_sample_rate_hz as usize)
            / ((fs_khz * 1000) as usize);

        if control.n_channels_api == 2 && control.n_channels_internal == 2 {
            let frame_id = encoder.state_fxx[0].common.n_frames_encoded;
            for n in 0..samples_from_input {
                buf[n] = samples_in[input_offset + 2 * n];
            }
            if encoder.n_prev_channels_internal == 1 && frame_id == 0 {
                encoder.state_fxx[1].resampler_state = encoder.state_fxx[0].resampler_state.clone();
            }
            let out_idx = encoder.state_fxx[0].common.input_buf_ix + 2;
            let produced = silk_resampler(
                &mut encoder.state_fxx[0].resampler_state,
                &mut encoder.state_fxx[0].common.input_buf[out_idx..out_idx + samples_to_buffer],
                &buf[..samples_from_input],
            );
            debug_assert_eq!(produced, samples_to_buffer);
            encoder.state_fxx[0].common.input_buf_ix += samples_to_buffer;

            let mut samples_to_buffer_ch1 = encoder.state_fxx[1]
                .common
                .frame_length
                .saturating_sub(encoder.state_fxx[1].common.input_buf_ix)
                .min((10 * n_blocks_of_10ms * encoder.state_fxx[1].common.fs_khz) as usize);
            for n in 0..samples_from_input {
                buf[n] = samples_in[input_offset + 2 * n + 1];
            }
            let out_idx = encoder.state_fxx[1].common.input_buf_ix + 2;
            if samples_to_buffer_ch1 == 0 {
                samples_to_buffer_ch1 = samples_to_buffer;
            }
            let produced = silk_resampler(
                &mut encoder.state_fxx[1].resampler_state,
                &mut encoder.state_fxx[1].common.input_buf
                    [out_idx..out_idx + samples_to_buffer_ch1],
                &buf[..samples_from_input],
            );
            debug_assert_eq!(produced, samples_to_buffer_ch1);
            encoder.state_fxx[1].common.input_buf_ix += samples_to_buffer_ch1;
        } else if control.n_channels_api == 2 && control.n_channels_internal == 1 {
            for n in 0..samples_from_input {
                let sum = i32::from(samples_in[input_offset + 2 * n])
                    + i32::from(samples_in[input_offset + 2 * n + 1]);
                buf[n] = sat16(rshift_round(sum, 1));
            }
            let out_idx = encoder.state_fxx[0].common.input_buf_ix + 2;
            let produced = silk_resampler(
                &mut encoder.state_fxx[0].resampler_state,
                &mut encoder.state_fxx[0].common.input_buf[out_idx..out_idx + samples_to_buffer],
                &buf[..samples_from_input],
            );
            debug_assert_eq!(produced, samples_to_buffer);
            if encoder.n_prev_channels_internal == 2
                && encoder.state_fxx[0].common.n_frames_encoded == 0
            {
                let produced = silk_resampler(
                    &mut encoder.state_fxx[1].resampler_state,
                    &mut encoder.state_fxx[1].common.input_buf
                        [encoder.state_fxx[1].common.input_buf_ix + 2
                            ..encoder.state_fxx[1].common.input_buf_ix + 2 + samples_to_buffer],
                    &buf[..samples_from_input],
                );
                debug_assert_eq!(produced, samples_to_buffer);
                for n in 0..encoder.state_fxx[0].common.frame_length {
                    let lhs = encoder.state_fxx[0].common.input_buf
                        [encoder.state_fxx[0].common.input_buf_ix + n + 2];
                    let rhs = encoder.state_fxx[1].common.input_buf
                        [encoder.state_fxx[1].common.input_buf_ix + n + 2];
                    encoder.state_fxx[0].common.input_buf
                        [encoder.state_fxx[0].common.input_buf_ix + n + 2] =
                        sat16((i32::from(lhs) + i32::from(rhs)) >> 1);
                }
            }
            encoder.state_fxx[0].common.input_buf_ix += samples_to_buffer;
        } else {
            buf[..samples_from_input]
                .copy_from_slice(&samples_in[input_offset..input_offset + samples_from_input]);
            let out_idx = encoder.state_fxx[0].common.input_buf_ix + 2;
            let produced = silk_resampler(
                &mut encoder.state_fxx[0].resampler_state,
                &mut encoder.state_fxx[0].common.input_buf[out_idx..out_idx + samples_to_buffer],
                &buf[..samples_from_input],
            );
            debug_assert_eq!(produced, samples_to_buffer);
            encoder.state_fxx[0].common.input_buf_ix += samples_to_buffer;
        }

        input_offset += samples_from_input * n_channels_api;
        remaining_samples -= i32::try_from(samples_from_input)
            .expect("input chunk size must fit in i32")
            * control.n_channels_api;

        encoder.allow_bandwidth_switch = false;

        if encoder.state_fxx[0].common.input_buf_ix >= frame_length {
            debug_assert_eq!(encoder.state_fxx[0].common.input_buf_ix, frame_length);
            if control.n_channels_internal == 2 {
                debug_assert_eq!(encoder.state_fxx[1].common.input_buf_ix, frame_length);
            }

            let frames_per_packet = encoder.state_fxx[0].common.n_frames_per_packet;
            let frames_per_packet_i32 =
                i32::try_from(frames_per_packet).expect("frames_per_packet fits in i32");
            debug_assert!(frames_per_packet_i32 > 0);
            let frames_encoded = encoder.state_fxx[0].common.n_frames_encoded;
            let frames_encoded_i32 =
                i32::try_from(frames_encoded).expect("frame counter fits in i32");
            let frame_idx = frames_encoded;

            if encoder.state_fxx[0].common.n_frames_encoded == 0 && !prefill.is_active() {
                let mut icdf = [0u8; 2];
                let header_bits = (frames_per_packet + 1) * internal_channels;
                let initial_icdf = 256u16 - (256u16 >> header_bits);
                icdf[0] = initial_icdf as u8;
                range_encoder.encode_icdf(0, &icdf, 8);
                curr_n_bits_used_lbrr = range_encoder.tell();

                for n in 0..internal_channels {
                    let mut lbrr_symbol = 0;
                    for i in 0..frames_per_packet {
                        if encoder.state_fxx[n].common.lbrr_flags[i] {
                            lbrr_symbol |= 1 << i;
                        }
                    }
                    encoder.state_fxx[n].common.lbrr_flag = lbrr_symbol > 0;
                    if lbrr_symbol > 0 && frames_per_packet > 1 {
                        let table = SILK_LBRR_FLAGS_ICDF_PTR[frames_per_packet - 2];
                        range_encoder.encode_icdf((lbrr_symbol - 1) as usize, table, 8);
                    }
                }

                for i in 0..frames_per_packet {
                    for n in 0..internal_channels {
                        if encoder.state_fxx[n].common.lbrr_flags[i] {
                            let cond_coding = if i > 0
                                && encoder.state_fxx[n].common.lbrr_flags[i - 1]
                            {
                                ConditionalCoding::Conditional
                            } else {
                                ConditionalCoding::Independent
                            };
                            let mut indices_state =
                                encoder_indices_state_from_common(&encoder.state_fxx[n]);
                            indices_state.encode_indices(
                                range_encoder,
                                &encoder.state_fxx[n].common.indices_lbrr[i],
                                cond_coding,
                                true,
                            );
                            encoder.state_fxx[n].common.ec_prev_signal_type =
                                indices_state.prev_signal_type;
                            encoder.state_fxx[n].common.ec_prev_lag_index =
                                indices_state.prev_lag_index;

                            silk_encode_pulses(
                                range_encoder,
                                i32::from(
                                    encoder.state_fxx[n].common.indices_lbrr[i].signal_type,
                                ),
                                quant_offset_as_i32(
                                    encoder.state_fxx[n].common.indices_lbrr[i].quant_offset_type,
                                ),
                                &mut encoder.state_fxx[n].common.pulses_lbrr[i][..frame_length],
                                frame_length,
                            );
                        }
                    }
                }

                for n in 0..internal_channels {
                    encoder.state_fxx[n].common.lbrr_flags = [false; MAX_FRAMES_PER_PACKET];
                }
                curr_n_bits_used_lbrr = range_encoder.tell() - curr_n_bits_used_lbrr;
            }

            for channel in encoder.state_fxx.iter_mut().take(internal_channels) {
                hp_variable_cutoff(channel);
            }

            let mut n_bits = (control.bit_rate * control.payload_size_ms) / 1000;
            if !prefill.is_active() {
                if curr_n_bits_used_lbrr < 10 {
                    encoder.n_bits_used_lbrr = 0;
                } else if encoder.n_bits_used_lbrr < 10 {
                    encoder.n_bits_used_lbrr = curr_n_bits_used_lbrr;
                } else {
                    encoder.n_bits_used_lbrr =
                        i32::midpoint(encoder.n_bits_used_lbrr, curr_n_bits_used_lbrr);
                }
                n_bits -= encoder.n_bits_used_lbrr;
            }
            n_bits /= frames_per_packet_i32;

            let mut target_rate_bps = if control.payload_size_ms == 10 {
                n_bits * 100
            } else {
                n_bits * 50
            };
            target_rate_bps -=
                (encoder.n_bits_exceeded * 1000) / BITRESERVOIR_DECAY_TIME_MS;
            if !prefill.is_active() && frames_encoded > 0 {
                let bits_balance = range_encoder.tell()
                    - encoder.n_bits_used_lbrr
                    - n_bits * frames_encoded_i32;
                target_rate_bps -= (bits_balance * 1000) / BITRESERVOIR_DECAY_TIME_MS;
            }

            let min_rate = min(control.bit_rate, 5000);
            let max_rate = max(control.bit_rate, 5000);
            target_rate_bps = target_rate_bps.clamp(min_rate, max_rate);

            let mut ms_target_rates_bps = [0; 2];
            if control.n_channels_internal == 2 {
                let speech_activity_q8 = encoder.state_fxx[0].common.speech_activity_q8;
                let fs_khz_internal = encoder.state_fxx[0].common.fs_khz;
                let result = {
                    let (left_state, right_state) = encoder.state_fxx.split_at_mut(1);
                    let left_buf = &mut left_state[0].common.input_buf[2..frame_length + 2];
                    let right_buf = &mut right_state[0].common.input_buf[2..frame_length + 2];
                    encoder
                        .stereo_state
                        .lr_to_ms(left_buf, right_buf, target_rate_bps, speech_activity_q8, control.to_mono, fs_khz_internal)
                };
                store_stereo_result(&mut encoder.stereo_state, frame_idx, &result);
                ms_target_rates_bps = result.mid_side_rates_bps;

                if !result.mid_only_flag && encoder.prev_decode_only_middle {
                    reset_side_channel(&mut encoder.state_fxx[1]);
                }
                if result.mid_only_flag {
                    encoder.state_fxx[1].common.vad_flags[frame_idx] = false;
                } else {
                    silk_encode_do_vad(&mut encoder.state_fxx[1], activity);
                }
                if !prefill.is_active() {
                    stereo_encode_pred(
                        range_encoder,
                        &encoder.stereo_state.pred_ix[frame_idx],
                    );
                    if !encoder.state_fxx[1].common.vad_flags[frame_idx] {
                        stereo_encode_mid_only(
                            range_encoder,
                            encoder.stereo_state.mid_only_flags[frame_idx] != 0,
                        );
                    }
                }
            } else {
                encoder.state_fxx[0]
                    .common
                    .input_buf[..2]
                    .copy_from_slice(&encoder.stereo_state.s_mid);
                encoder
                    .stereo_state
                    .s_mid
                    .copy_from_slice(
                        &encoder.state_fxx[0].common.input_buf
                            [frame_length..frame_length + 2],
                    );
            }

            silk_encode_do_vad(&mut encoder.state_fxx[0], activity);

            for n in 0..internal_channels {
                let mut max_bits = control.max_bits;
                if tot_blocks == 2 && curr_block == 0 {
                    max_bits = (max_bits * 3) / 5;
                } else if tot_blocks == 3 {
                    if curr_block == 0 {
                        max_bits = (max_bits * 2) / 5;
                    } else if curr_block == 1 {
                        max_bits = (max_bits * 3) / 4;
                    }
                }
                let mut use_cbr = control.use_cbr != 0 && curr_block == tot_blocks - 1;

                let channel_rate_bps = if control.n_channels_internal == 1 {
                    target_rate_bps
                } else {
                    ms_target_rates_bps[n]
                };

                if n == 0 && control.n_channels_internal == 2 && ms_target_rates_bps[1] > 0 {
                    use_cbr = false;
                    max_bits -= control.max_bits / (tot_blocks * 2);
                }

                if channel_rate_bps > 0 {
                    control_snr(&mut encoder.state_fxx[n], channel_rate_bps)?;

                    let cond_coding = if frames_encoded_i32 - n as i32 <= 0 {
                        ConditionalCoding::Independent
                    } else if n > 0 && encoder.prev_decode_only_middle {
                        ConditionalCoding::IndependentNoLtpScaling
                    } else {
                        ConditionalCoding::Conditional
                    };

                    let err = silk_encode_frame(
                        &mut encoder.state_fxx[n],
                        n_bytes_out,
                        range_encoder,
                        cond_coding,
                        max_bits,
                        use_cbr,
                    );
                    if err != SilkError::NoError {
                        return Err(err);
                    }
                }

                encoder.state_fxx[n].common.controlled_since_last_payload = false;
                encoder.state_fxx[n].common.input_buf_ix = 0;
                encoder.state_fxx[n].common.n_frames_encoded += 1;
            }

            let current_flag_idx = frame_idx.min(MAX_FRAMES_PER_PACKET - 1);
            encoder.prev_decode_only_middle =
                encoder.stereo_state.mid_only_flags[current_flag_idx] != 0;

            if *n_bytes_out > 0 && encoder.state_fxx[0].common.n_frames_encoded == frames_per_packet {
                let mut flags = 0u32;
                for n in 0..internal_channels {
                    for i in 0..frames_per_packet {
                        flags <<= 1;
                        flags |= u32::from(encoder.state_fxx[n].common.vad_flags[i]);
                    }
                    flags <<= 1;
                    flags |= u32::from(encoder.state_fxx[n].common.lbrr_flag);
                }
                if !prefill.is_active() {
                    let nbits = (frames_per_packet_i32 + 1) * control.n_channels_internal;
                    range_encoder.patch_initial_bits(flags, nbits as u32);
                }

                if encoder.state_fxx[0].common.in_dtx
                    && (control.n_channels_internal == 1
                        || encoder.state_fxx[1].common.in_dtx)
                {
                    *n_bytes_out = 0;
                }

                encoder.n_bits_exceeded += *n_bytes_out * 8;
                encoder.n_bits_exceeded -=
                    (control.bit_rate * control.payload_size_ms) / 1000;
                encoder.n_bits_exceeded = encoder.n_bits_exceeded.clamp(0, 10_000);

                let base_q8 = ((SPEECH_ACTIVITY_DTX_THRES * 256.0) + 0.5) as i32;
                let coef_q24 = (((1.0 - SPEECH_ACTIVITY_DTX_THRES)
                    / (MAX_BANDWIDTH_SWITCH_DELAY_MS as f32))
                    * (1u32 << 24) as f32
                    + 0.5) as i32;
                let speech_thr_q8 = base_q8
                    + (((i64::from(coef_q24)
                        * i64::from(encoder.time_since_switch_allowed_ms))
                        + (1 << 15))
                        >> 16) as i32;
                if encoder.state_fxx[0].common.speech_activity_q8 < speech_thr_q8 {
                    encoder.allow_bandwidth_switch = true;
                    encoder.time_since_switch_allowed_ms = 0;
                } else {
                    encoder.allow_bandwidth_switch = false;
                    encoder.time_since_switch_allowed_ms += control.payload_size_ms;
                }

                for channel in encoder.state_fxx.iter_mut().take(internal_channels) {
                    channel.common.allow_bandwidth_switch = encoder.allow_bandwidth_switch;
                }
            }

            if remaining_samples == 0 {
                break;
            }
        } else {
            break;
        }
        curr_block += 1;
    }

    encoder.n_prev_channels_internal = control.n_channels_internal;

    control.allow_bandwidth_switch = encoder.allow_bandwidth_switch;
    control.in_wb_mode_without_variable_lp =
        encoder.state_fxx[0].common.fs_khz == 16 && encoder.state_fxx[0].common.lp_state.mode == 0;
    control.internal_sample_rate = encoder.state_fxx[0].common.fs_khz * 1000;
    control.stereo_width_q14 = if control.to_mono {
        0
    } else {
        i32::from(encoder.stereo_state.smth_width_q14)
    };

    if prefill.is_active() {
        control.payload_size_ms = tmp_payload_size_ms;
        control.complexity = tmp_complexity;
        for channel in encoder.state_fxx.iter_mut().take(internal_channels) {
            channel.common.controlled_since_last_payload = false;
            channel.common.prefill_flag = false;
        }
    }

    control.signal_type = encoder.state_fxx[0].common.indices.signal_type;
    control.offset = i32::from(
        SILK_QUANTIZATION_OFFSETS_Q10
            [encoder.state_fxx[0].common.indices.signal_type as usize >> 1]
            [quant_offset_as_i32(encoder.state_fxx[0].common.indices.quant_offset_type) as usize],
    );

    Ok(())
}

fn encoder_indices_state_from_common(channel: &EncoderChannelState) -> EncoderIndicesState {
    EncoderIndicesState {
        nb_subfr: channel.common.nb_subfr,
        fs_khz: channel.common.fs_khz,
        predict_lpc_order: channel.common.predict_lpc_order,
        nlsf_codebook: channel.common.ps_nlsf_cb,
        pitch_lag_low_bits_icdf: channel.common.pitch_lag_low_bits_icdf,
        pitch_contour_icdf: channel.common.pitch_contour_icdf,
        prev_signal_type: channel.common.ec_prev_signal_type,
        prev_lag_index: channel.common.ec_prev_lag_index,
    }
}

fn quant_offset_as_i32(offset: FrameQuantizationOffsetType) -> i32 {
    match offset {
        FrameQuantizationOffsetType::Low => 0,
        FrameQuantizationOffsetType::High => 1,
    }
}

fn reset_side_channel(channel: &mut EncoderChannelState) {
    channel.shape_state = EncoderShapeState::default();
    channel.common.nsq_state = NoiseShapingQuantizerState::default();
    channel.common.prev_nlsf_q15 = [0; MAX_LPC_ORDER];
    channel.common.lp_state.in_lp_state = [0; 2];
    channel.common.prev_lag = 100;
    channel.common.nsq_state.lag_prev = 100;
    channel.shape_state.last_gain_index = 10;
    channel.common.prev_signal_type = FrameSignalType::Inactive;
    channel.common.nsq_state.prev_gain_q16 = 65_536;
    channel.common.first_frame_after_reset = true;
}

fn store_stereo_result(state: &mut StereoEncState, frame_idx: usize, result: &StereoConversionResult) {
    let idx = frame_idx.min(MAX_FRAMES_PER_PACKET - 1);
    state.pred_ix[idx] = result.indices;
    state.mid_only_flags[idx] = if result.mid_only_flag { 1 } else { 0 };
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else {
        (value + (1 << (shift - 1))) >> shift
    }
}

fn sat16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}
