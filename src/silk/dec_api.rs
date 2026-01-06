//! Port of `silk/dec_API.c`.
//!
//! This module mirrors the public SILK decoder entry points that operate on
//! the multi-channel `silk_decoder` super-structure.  It wires together the
//! per-channel [`DecoderState`] instances, stereo mid/side helper, range
//! decoder, PLC/CNG glue, and API-facing resampler so that callers can invoke
//! the same lifecycle (`silk_InitDecoder`, `silk_ResetDecoder`) and per-frame
//! decode workflow (`silk_Decode`) exposed by the reference C sources.

use alloc::vec;
use alloc::vec::Vec;
use core::array::from_fn;
use core::cmp::min;

use crate::silk::SilkRangeDecoder;
use crate::silk::decode_frame::{DecodeFlag, silk_decode_frame};
use crate::silk::decode_indices::{ConditionalCoding, DecoderIndicesState, SideInfoIndices};
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_set_fs::{DecoderSetFsError, MAX_DECODER_BUFFER, MAX_FRAME_LENGTH};
use crate::silk::decoder_state::DecoderState;
use crate::silk::errors::SilkError;
use crate::silk::init_decoder::{init_decoder as init_channel, reset_decoder as reset_channel};
use crate::silk::resampler::silk_resampler;
use crate::silk::stereo_decode_pred::{stereo_decode_mid_only, stereo_decode_pred};
use crate::silk::stereo_ms_to_lr::StereoDecState;
use crate::silk::tables_other::SILK_LBRR_FLAGS_ICDF_PTR;
use crate::silk::{FrameQuantizationOffsetType, FrameSignalType, MAX_FRAMES_PER_PACKET};

/// Maximum number of decoder channels handled by the public API.
pub const DECODER_NUM_CHANNELS: usize = 2;
const MAX_API_FS_KHZ: i32 = 48;

/// Top-level SILK decoder super-structure mirroring `silk_decoder`.
#[derive(Debug)]
pub struct Decoder {
    /// Per-channel decoder working state (`silk_decoder_state`).
    pub channel_states: [DecoderState; DECODER_NUM_CHANNELS],
    /// Stereo predictor/history shared across channels.
    pub stereo_state: StereoDecState,
    /// Number of channels exposed via the API.
    pub n_channels_api: i32,
    /// Number of internally decoded channels.
    pub n_channels_internal: i32,
    /// Tracks whether the previous call decoded only the mid channel.
    pub prev_decode_only_middle: bool,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            channel_states: from_fn(|_| DecoderState::default()),
            stereo_state: StereoDecState::default(),
            n_channels_api: 1,
            n_channels_internal: 1,
            prev_decode_only_middle: false,
        }
    }
}

impl Decoder {
    /// Returns an immutable view of a channel state.
    pub fn channel_state(&self, index: usize) -> &DecoderState {
        &self.channel_states[index]
    }

    /// Returns a mutable view of a channel state.
    pub fn channel_state_mut(&mut self, index: usize) -> &mut DecoderState {
        &mut self.channel_states[index]
    }
}

/// Decoder control parameters mirrored from `silk_DecControlStruct`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecControl {
    /// Number of channels advertised through the public API (1 or 2).
    pub n_channels_api: i32,
    /// Number of internally decoded channels (1 or 2).
    pub n_channels_internal: i32,
    /// Output sample rate in Hertz.
    pub api_sample_rate: i32,
    /// Internal decoder sample rate in Hertz.
    pub internal_sample_rate: i32,
    /// Packet duration in milliseconds (10/20/40/60 or 0 for PLC).
    pub payload_size_ms: i32,
    /// Previous frame pitch lag reported at 48 kHz.
    pub prev_pitch_lag: i32,
    /// Enables the optional deep-PLC path.
    pub enable_deep_plc: bool,
}

impl Default for DecControl {
    fn default() -> Self {
        Self {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 16_000,
            internal_sample_rate: 16_000,
            payload_size_ms: 20,
            prev_pitch_lag: 0,
            enable_deep_plc: false,
        }
    }
}

/// Mirrors `silk_ResetDecoder` by clearing the per-channel state and stereo memory.
pub fn reset_decoder(decoder: &mut Decoder) -> Result<(), SilkError> {
    for state in decoder.channel_states.iter_mut() {
        reset_channel(state)?;
    }
    decoder.stereo_state = StereoDecState::default();
    decoder.prev_decode_only_middle = false;
    Ok(())
}

/// Mirrors `silk_InitDecoder` by reinitialising each channel.
pub fn init_decoder(decoder: &mut Decoder) -> Result<(), SilkError> {
    for state in decoder.channel_states.iter_mut() {
        init_channel(state)?;
    }
    decoder.stereo_state = StereoDecState::default();
    decoder.prev_decode_only_middle = false;
    decoder.n_channels_api = 1;
    decoder.n_channels_internal = 1;
    Ok(())
}

/// Rust translation of `silk_Decode`.
///
/// Returns the number of samples written to `samples_out` (per channel) after
/// resampling to `control.api_sample_rate`.
#[allow(clippy::too_many_arguments)]
pub fn silk_decode(
    decoder: &mut Decoder,
    control: &mut DecControl,
    lost_flag: DecodeFlag,
    new_packet: bool,
    range_decoder: &mut impl SilkRangeDecoder,
    samples_out: &mut [i16],
    arch: i32,
) -> Result<usize, SilkError> {
    let internal_channels = usize::try_from(control.n_channels_internal)
        .unwrap_or(DECODER_NUM_CHANNELS)
        .min(DECODER_NUM_CHANNELS);
    debug_assert!((1..=DECODER_NUM_CHANNELS).contains(&internal_channels));

    if new_packet {
        for state in decoder.channel_states.iter_mut().take(internal_channels) {
            state.n_frames_decoded = 0;
        }
    }

    if internal_channels > usize::try_from(decoder.n_channels_internal).unwrap_or(1) {
        for state in decoder.channel_states.iter_mut().take(internal_channels) {
            init_channel(state)?;
        }
    }

    let stereo_to_mono = control.n_channels_internal == 1
        && decoder.n_channels_internal == 2
        && control.internal_sample_rate == decoder.channel_states[0].sample_rate.fs_khz * 1000;

    if decoder.channel_states[0].n_frames_decoded == 0 {
        for state in decoder.channel_states.iter_mut().take(internal_channels) {
            let (frames_per_packet, nb_subframes) =
                frames_per_packet(control.payload_size_ms).ok_or(SilkError::DecInvalidFrameSize)?;
            state.n_frames_per_packet = frames_per_packet;
            state.sample_rate.nb_subfr = nb_subframes;
        }
    }

    for state in decoder.channel_states.iter_mut().take(internal_channels) {
        let fs_khz = to_internal_fs_khz(control.internal_sample_rate)
            .ok_or(SilkError::DecInvalidSamplingFrequency)?;
        state
            .sample_rate
            .set_sample_rates(fs_khz, control.api_sample_rate)
            .map_err(|err| map_set_fs_error(&err))?;
    }

    if control.n_channels_api == 2
        && control.n_channels_internal == 2
        && (decoder.n_channels_api == 1 || decoder.n_channels_internal == 1)
    {
        decoder.stereo_state.pred_prev_q13 = [0; 2];
        decoder.stereo_state.s_side = [0; 2];
        decoder.channel_states[1].sample_rate.resampler_state = decoder.channel_states[0]
            .sample_rate
            .resampler_state
            .clone();
    }

    decoder.n_channels_api = control.n_channels_api;
    decoder.n_channels_internal = control.n_channels_internal;

    if control.api_sample_rate < 8_000 || control.api_sample_rate > MAX_API_FS_KHZ * 1_000 {
        return Err(SilkError::DecInvalidSamplingFrequency);
    }

    let mut decode_only_middle = decoder.prev_decode_only_middle;

    if lost_flag != DecodeFlag::PacketLoss && decoder.channel_states[0].n_frames_decoded == 0 {
        decode_vad_and_lbrr(
            decoder,
            control,
            lost_flag,
            range_decoder,
            &mut decode_only_middle,
        );
    }
    let mut ms_pred_q13 = [0i32; 2];
    if control.n_channels_internal == 2 {
        let frame_idx = usize::try_from(decoder.channel_states[0].n_frames_decoded).unwrap_or(0);
        let decode_mid_side = if lost_flag == DecodeFlag::Normal {
            true
        } else if lost_flag == DecodeFlag::Lbrr {
            decoder.channel_states[0]
                .lbrr_flags
                .get(frame_idx)
                .copied()
                .unwrap_or(0)
                == 1
        } else {
            false
        };
        if decode_mid_side {
            stereo_decode_pred(range_decoder, &mut ms_pred_q13);
            let need_mid_only = match lost_flag {
                DecodeFlag::Normal => {
                    decoder.channel_states[1]
                        .vad_flags
                        .get(frame_idx)
                        .copied()
                        .unwrap_or(0)
                        == 0
                }
                DecodeFlag::Lbrr => {
                    decoder.channel_states[1]
                        .lbrr_flags
                        .get(frame_idx)
                        .copied()
                        .unwrap_or(0)
                        == 0
                }
                DecodeFlag::PacketLoss => false,
            };
            decode_only_middle = if need_mid_only {
                stereo_decode_mid_only(range_decoder)
            } else {
                false
            };
        } else {
            for (dst, &src) in ms_pred_q13
                .iter_mut()
                .zip(decoder.stereo_state.pred_prev_q13.iter())
            {
                *dst = i32::from(src);
            }
        }
    } else {
        decode_only_middle = false;
    }

    if control.n_channels_internal == 2 && !decode_only_middle && decoder.prev_decode_only_middle {
        let side = &mut decoder.channel_states[1];
        side.sample_rate.out_buf = [0; MAX_DECODER_BUFFER];
        side.sample_rate.s_lpc_q14_buf.fill(0);
        side.sample_rate.lag_prev = 100;
        side.sample_rate.last_gain_index = 10;
        side.sample_rate.prev_signal_type = FrameSignalType::Inactive;
        side.sample_rate.first_frame_after_reset = true;
    }

    let frame_length = decoder.channel_states[0].sample_rate.frame_length;
    let mut channel_buffers: Vec<Vec<i16>> = (0..internal_channels)
        .map(|_| vec![0; frame_length + 2])
        .collect();
    let mut n_samples_out_dec = 0usize;
    let has_side = match lost_flag {
        DecodeFlag::Normal => !decode_only_middle,
        DecodeFlag::PacketLoss => !decoder.prev_decode_only_middle,
        DecodeFlag::Lbrr => {
            !decoder.prev_decode_only_middle
                || (control.n_channels_internal == 2
                    && decoder.channel_states[1]
                        .lbrr_flags
                        .get(decoder.channel_states[1].n_frames_decoded as usize)
                        .copied()
                        .unwrap_or(0)
                        == 1)
        }
    };

    decoder.channel_states[0].plc_state.enable_deep_plc = control.enable_deep_plc;

    for (channel_idx, buffer) in channel_buffers
        .iter_mut()
        .enumerate()
        .take(internal_channels)
    {
        if channel_idx == 0 || has_side {
            let frame_index = decoder.channel_states[0].n_frames_decoded - channel_idx as i32;
            let coding = {
                let state_snapshot = &decoder.channel_states[channel_idx];
                conditional_coding_for_channel(
                    state_snapshot,
                    frame_index,
                    channel_idx,
                    decoder,
                    lost_flag,
                )
            };
            let state = &mut decoder.channel_states[channel_idx];
            n_samples_out_dec = silk_decode_frame(
                state,
                range_decoder,
                &mut buffer[2..2 + frame_length],
                lost_flag,
                coding,
                arch,
            );
            state.n_frames_decoded += 1;
        } else {
            buffer[2..2 + frame_length].fill(0);
            decoder.channel_states[channel_idx].n_frames_decoded += 1;
        }
    }

    if control.n_channels_api == 2 && control.n_channels_internal == 2 {
        let (mid_buf, side_buf) = channel_buffers.split_at_mut(1);
        decoder.stereo_state.ms_to_lr(
            &mut mid_buf[0],
            &mut side_buf[0],
            &ms_pred_q13,
            decoder.channel_states[0].sample_rate.fs_khz,
            n_samples_out_dec,
        );
    } else {
        channel_buffers[0][..2].copy_from_slice(&decoder.stereo_state.s_mid);
        decoder
            .stereo_state
            .s_mid
            .copy_from_slice(&channel_buffers[0][n_samples_out_dec..n_samples_out_dec + 2]);
    }

    let fs_khz = decoder.channel_states[0].sample_rate.fs_khz;
    let n_samples_api = ((n_samples_out_dec as i64) * i64::from(control.api_sample_rate)
        / (i64::from(fs_khz) * 1000)) as usize;
    if samples_out.len() < n_samples_api * usize::try_from(control.n_channels_api).unwrap_or(1) {
        return Err(SilkError::DecPayloadTooLarge);
    }

    let mut resampled = vec![0i16; n_samples_api];
    let active_channels = min(control.n_channels_api, control.n_channels_internal) as usize;

    for channel_idx in 0..active_channels {
        let input = &channel_buffers[channel_idx][1..1 + n_samples_out_dec];
        let produced = silk_resampler(
            &mut decoder.channel_states[channel_idx]
                .sample_rate
                .resampler_state,
            &mut resampled,
            input,
        );
        debug_assert_eq!(produced, n_samples_api);
        if control.n_channels_api == 2 {
            for (frame_idx, &sample) in resampled.iter().enumerate() {
                samples_out[channel_idx + 2 * frame_idx] = sample;
            }
        } else {
            samples_out[..n_samples_api].copy_from_slice(&resampled);
        }
    }

    if control.n_channels_api == 2 && control.n_channels_internal == 1 {
        if stereo_to_mono {
            let produced = silk_resampler(
                &mut decoder.channel_states[1].sample_rate.resampler_state,
                &mut resampled,
                &channel_buffers[0][1..1 + n_samples_out_dec],
            );
            debug_assert_eq!(produced, n_samples_api);
            for (frame_idx, &sample) in resampled.iter().enumerate() {
                samples_out[1 + 2 * frame_idx] = sample;
            }
        } else {
            for frame_idx in 0..n_samples_api {
                samples_out[1 + 2 * frame_idx] = samples_out[2 * frame_idx];
            }
        }
    }

    if decoder.channel_states[0].sample_rate.prev_signal_type == FrameSignalType::Voiced {
        let mult_tab = [6, 4, 3];
        let idx = ((fs_khz - 8) / 4) as usize;
        control.prev_pitch_lag = decoder.channel_states[0].sample_rate.lag_prev * mult_tab[idx];
    } else {
        control.prev_pitch_lag = 0;
    }

    if lost_flag == DecodeFlag::PacketLoss {
        for state in decoder.channel_states.iter_mut().take(internal_channels) {
            state.sample_rate.last_gain_index = 10;
        }
    } else {
        decoder.prev_decode_only_middle = decode_only_middle;
    }

    Ok(n_samples_api)
}

fn frames_per_packet(payload_ms: i32) -> Option<(i32, usize)> {
    match payload_ms {
        0 | 10 => Some((1, 2)),
        20 => Some((1, 4)),
        40 => Some((2, 4)),
        60 => Some((3, 4)),
        _ => None,
    }
}

fn to_internal_fs_khz(fs_hz: i32) -> Option<i32> {
    if fs_hz <= 0 {
        return None;
    }
    let fs_khz = (fs_hz >> 10) + 1;
    match fs_khz {
        8 | 12 | 16 => Some(fs_khz),
        _ => None,
    }
}

fn map_set_fs_error(err: &DecoderSetFsError) -> SilkError {
    match err {
        DecoderSetFsError::UnsupportedInternalSampleRate(_)
        | DecoderSetFsError::InvalidSubframeCount(_)
        | DecoderSetFsError::Resampler(_) => SilkError::DecInvalidSamplingFrequency,
    }
}

fn quant_offset_type(offset: FrameQuantizationOffsetType) -> i32 {
    match offset {
        FrameQuantizationOffsetType::Low => 0,
        FrameQuantizationOffsetType::High => 1,
    }
}

fn decode_vad_and_lbrr(
    decoder: &mut Decoder,
    control: &DecControl,
    lost_flag: DecodeFlag,
    range_decoder: &mut impl SilkRangeDecoder,
    decode_only_middle: &mut bool,
) {
    let channels = usize::try_from(control.n_channels_internal).unwrap_or(1);
    for state in decoder.channel_states.iter_mut().take(channels) {
        for frame in 0..state.n_frames_per_packet as usize {
            state.vad_flags[frame] = range_decoder.decode_symbol_logp(1) as i32;
        }
        state.lbrr_flag = range_decoder.decode_symbol_logp(1) as i32;
    }

    for state in decoder.channel_states.iter_mut().take(channels) {
        state.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        if state.lbrr_flag != 0 {
            if state.n_frames_per_packet == 1 {
                state.lbrr_flags[0] = 1;
            } else {
                let idx = (state.n_frames_per_packet - 2) as usize;
                let symbol = range_decoder.decode_icdf(SILK_LBRR_FLAGS_ICDF_PTR[idx], 8) + 1;
                for frame in 0..state.n_frames_per_packet as usize {
                    state.lbrr_flags[frame] = (symbol >> frame) as i32 & 1;
                }
            }
        }
    }

    if lost_flag != DecodeFlag::Normal {
        return;
    }

    let mut skip_pred = [0i32; 2];
    let mut temp_pulses = vec![0i16; MAX_FRAME_LENGTH];

    for frame in 0..decoder.channel_states[0].n_frames_per_packet as usize {
        for ch in 0..channels {
            let side_has_lbrr = control.n_channels_internal == 2
                && ch == 0
                && decoder.channel_states[1]
                    .lbrr_flags
                    .get(frame)
                    .copied()
                    .unwrap_or(0)
                    != 0;
            let state = &mut decoder.channel_states[ch];
            if state.lbrr_flags[frame] == 0 {
                continue;
            }

            if control.n_channels_internal == 2 && ch == 0 {
                stereo_decode_pred(range_decoder, &mut skip_pred);
                if !side_has_lbrr {
                    *decode_only_middle = stereo_decode_mid_only(range_decoder);
                }
            }

            let cond = if frame > 0 && state.lbrr_flags[frame - 1] != 0 {
                ConditionalCoding::Conditional
            } else {
                ConditionalCoding::Independent
            };
            let indices = decode_side_info(state, range_decoder, frame, true, cond);
            let frame_len = state.sample_rate.frame_length;
            silk_decode_pulses(
                range_decoder,
                &mut temp_pulses[..frame_len],
                i32::from(indices.signal_type),
                quant_offset_type(indices.quant_offset_type),
                frame_len,
            );
        }
    }
}

fn decode_side_info(
    state: &mut DecoderState,
    range_decoder: &mut impl SilkRangeDecoder,
    frame_index: usize,
    decode_lbrr: bool,
    coding: ConditionalCoding,
) -> SideInfoIndices {
    let sr = &state.sample_rate;
    let mut vad_flags = [false; MAX_FRAMES_PER_PACKET];
    for (dst, &src) in vad_flags.iter_mut().zip(state.vad_flags.iter()) {
        *dst = src != 0;
    }

    let mut indices_state = DecoderIndicesState {
        vad_flags,
        nb_subfr: sr.nb_subfr,
        fs_khz: sr.fs_khz,
        lpc_order: sr.lpc_order,
        pitch_lag_low_bits_icdf: sr.pitch_lag_low_bits_icdf,
        pitch_contour_icdf: sr.pitch_contour_icdf,
        nlsf_codebook: sr.ps_nlsf_cb,
        prev_signal_type: state.ec_prev_signal_type,
        prev_lag_index: state.ec_prev_lag_index,
    };

    let indices = indices_state.decode_indices(range_decoder, frame_index, decode_lbrr, coding);
    state.ec_prev_signal_type = indices_state.prev_signal_type;
    state.ec_prev_lag_index = indices_state.prev_lag_index;
    state.indices = indices.clone();
    indices
}

fn conditional_coding_for_channel(
    state: &DecoderState,
    frame_index: i32,
    channel_idx: usize,
    decoder: &Decoder,
    lost_flag: DecodeFlag,
) -> ConditionalCoding {
    if frame_index <= 0 {
        return ConditionalCoding::Independent;
    }

    match lost_flag {
        DecodeFlag::Lbrr => {
            let idx = (frame_index - 1) as usize;
            if state.lbrr_flags.get(idx).copied().unwrap_or(0) != 0 {
                ConditionalCoding::Conditional
            } else {
                ConditionalCoding::Independent
            }
        }
        _ if channel_idx > 0 && decoder.prev_decode_only_middle => {
            ConditionalCoding::IndependentNoLtpScaling
        }
        _ => ConditionalCoding::Conditional,
    }
}

#[cfg(test)]
mod tests {
    use super::{DecControl, Decoder, frames_per_packet};

    #[test]
    fn frames_per_packet_matches_reference() {
        assert_eq!(frames_per_packet(0), Some((1, 2)));
        assert_eq!(frames_per_packet(10), Some((1, 2)));
        assert_eq!(frames_per_packet(20), Some((1, 4)));
        assert_eq!(frames_per_packet(40), Some((2, 4)));
        assert_eq!(frames_per_packet(60), Some((3, 4)));
        assert_eq!(frames_per_packet(15), None);
    }

    #[test]
    fn decoder_default_initialises_channel_counts() {
        let decoder = Decoder::default();
        assert_eq!(decoder.n_channels_api, 1);
        assert_eq!(decoder.n_channels_internal, 1);
        assert!(!decoder.prev_decode_only_middle);
    }

    #[test]
    fn dec_control_defaults_to_wideband_mono() {
        let control = DecControl::default();
        assert_eq!(control.n_channels_api, 1);
        assert_eq!(control.api_sample_rate, 16_000);
        assert_eq!(control.payload_size_ms, 20);
        assert!(!control.enable_deep_plc);
    }
}
