//! Port of `silk/decode_frame.c`.
//!
//! This module stitches together the range-decoder side-information helpers,
//! pulse decoder, parameter reconstruction, inverse NSQ, PLC, and CNG stages
//! so that a single SILK frame can be recovered exactly like the reference C
//! implementation.

use alloc::vec;

use crate::range::RangeDecoder;
use crate::silk::cng::{apply_cng, ComfortNoiseInputs, DecoderControl, PlcState};
use crate::silk::decode_core::silk_decode_core;
use crate::silk::decode_indices::{ConditionalCoding, DecoderIndicesState, SideInfoIndices};
use crate::silk::decode_parameters::{silk_decode_parameters, DecoderParametersState};
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_set_fs::MAX_FRAME_LENGTH;
use crate::silk::decoder_state::DecoderState;
use crate::silk::plc::{silk_plc, silk_plc_glue_frames};
use crate::silk::{FrameQuantizationOffsetType, MAX_FRAMES_PER_PACKET};

const SHELL_CODEC_FRAME_LENGTH: usize = 16;

/// Mirrors the `lostFlag` values consumed by `silk_decode_frame`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeFlag {
    /// Regular frame decode.
    Normal,
    /// Packet was lost; synthesize audio via PLC/CNG only.
    PacketLoss,
    /// Decode the low-bit-rate redundancy (LBRR) data instead of the primary frame.
    Lbrr,
}

impl DecodeFlag {
    fn should_decode_payload(self, lbrr_active: bool) -> bool {
        match self {
            Self::Normal => true,
            Self::PacketLoss => false,
            Self::Lbrr => lbrr_active,
        }
    }

    fn is_lbrr(self) -> bool {
        matches!(self, Self::Lbrr)
    }
}

/// Mirrors `silk_decode_frame`: decodes (or conceals) a single SILK frame.
///
/// # Returns
/// The number of samples written to `output`, which always matches the decoder
/// state's frame length.
pub fn silk_decode_frame(
    state: &mut DecoderState,
    range_decoder: &mut RangeDecoder<'_>,
    output: &mut [i16],
    lost_flag: DecodeFlag,
    cond_coding: ConditionalCoding,
    arch: i32,
) -> usize {
    let frame_length = state.sample_rate.frame_length;
    assert!(
        frame_length > 0 && frame_length <= MAX_FRAME_LENGTH,
        "frame length {} out of range",
        frame_length
    );
    assert!(
        output.len() >= frame_length,
        "output buffer shorter than frame"
    );

    let frame_index = usize::try_from(state.n_frames_decoded)
        .expect("frame counter must not be negative");
    assert!(
        frame_index < MAX_FRAMES_PER_PACKET,
        "frame index {} exceeds MAX_FRAMES_PER_PACKET",
        frame_index
    );

    let mut control = DecoderControl {
        ltp_scale_q14: 0,
        ..DecoderControl::default()
    };

    if lost_flag.should_decode_payload(state.lbrr_flags[frame_index] != 0) {
        let indices = decode_side_information(
            state,
            range_decoder,
            frame_index,
            lost_flag.is_lbrr(),
            cond_coding,
        );
        state.indices = indices.clone();

        let padded_length = align_shell_frame_length(frame_length);
        let mut pulses = vec![0i16; padded_length];
        silk_decode_pulses(
            range_decoder,
            &mut pulses,
            i32::from(indices.signal_type),
            quant_offset_as_i32(indices.quant_offset_type),
            frame_length,
        );

        let mut params = build_parameters_state(state);
        silk_decode_parameters(&mut params, &mut control, cond_coding);
        commit_parameters(state, params);

        silk_decode_core(
            state,
            &mut control,
            &mut output[..frame_length],
            &pulses[..frame_length],
            arch,
        );
        silk_plc(state, &mut control, &mut output[..frame_length], false, arch);
        state.loss_count = 0;
        state.sample_rate.first_frame_after_reset = false;
    } else {
        silk_plc(state, &mut control, &mut output[..frame_length], true, arch);
    }

    refresh_output_buffer(state, &output[..frame_length]);

    let sr = &state.sample_rate;
    let cng_inputs = ComfortNoiseInputs {
        fs_khz: sr.fs_khz,
        lpc_order: sr.lpc_order,
        nb_subfr: sr.nb_subfr,
        subfr_length: sr.subfr_length,
        prev_signal_type: sr.prev_signal_type,
        loss_count: state.loss_count,
        prev_nlsf_q15: &state.prev_nlsf_q15,
        exc_q14: &state.exc_q14,
    };
    let plc_snapshot = plc_summary(state);
    apply_cng(
        &mut state.cng_state,
        &plc_snapshot,
        &control,
        &cng_inputs,
        &mut output[..frame_length],
    );
    silk_plc_glue_frames(state, &mut output[..frame_length]);

    let nb_subfr = state.sample_rate.nb_subfr;
    state.sample_rate.lag_prev = i32::from(control.pitch_l[nb_subfr - 1]);

    frame_length
}

fn decode_side_information(
    state: &mut DecoderState,
    range_decoder: &mut RangeDecoder<'_>,
    frame_index: usize,
    decode_lbrr: bool,
    cond_coding: ConditionalCoding,
) -> SideInfoIndices {
    let sr = &state.sample_rate;
    let mut vad_flags = [false; MAX_FRAMES_PER_PACKET];
    for (dst, &flag) in vad_flags.iter_mut().zip(state.vad_flags.iter()) {
        *dst = flag != 0;
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
    let indices =
        indices_state.decode_indices(range_decoder, frame_index, decode_lbrr, cond_coding);
    state.ec_prev_signal_type = indices_state.prev_signal_type;
    state.ec_prev_lag_index = indices_state.prev_lag_index;
    indices
}

fn build_parameters_state(state: &DecoderState) -> DecoderParametersState {
    let sr = &state.sample_rate;
    DecoderParametersState {
        indices: state.indices.clone(),
        prev_nlsf_q15: state.prev_nlsf_q15,
        nlsf_codebook: sr.ps_nlsf_cb,
        lpc_order: sr.lpc_order,
        nb_subfr: sr.nb_subfr,
        fs_khz: sr.fs_khz,
        loss_count: state.loss_count,
        first_frame_after_reset: sr.first_frame_after_reset,
        last_gain_index: i8::try_from(sr.last_gain_index)
            .expect("gain index must fit in i8 for parameter decoding"),
        arch: state.arch,
    }
}

fn commit_parameters(state: &mut DecoderState, params: DecoderParametersState) {
    state.prev_nlsf_q15 = params.prev_nlsf_q15;
    state.indices = params.indices;
    state.sample_rate.last_gain_index = i32::from(params.last_gain_index);
}

fn refresh_output_buffer(state: &mut DecoderState, frame: &[i16]) {
    let sr = &mut state.sample_rate;
    let frame_length = frame.len();
    assert_eq!(
        frame_length, sr.frame_length,
        "frame slice must match decoder frame length"
    );
    assert!(
        sr.ltp_mem_length >= frame_length,
        "ltp_mem_length shorter than frame length"
    );
    let mv_len = sr.ltp_mem_length - frame_length;
    assert!(
        sr.out_buf.len() >= sr.ltp_mem_length,
        "out_buf shorter than ltp_mem_length"
    );

    if mv_len > 0 {
        let src_start = frame_length;
        let src_end = frame_length + mv_len;
        sr.out_buf
            .copy_within(src_start..src_end, 0);
    }

    sr.out_buf[mv_len..mv_len + frame_length].copy_from_slice(frame);
}

fn align_shell_frame_length(frame_length: usize) -> usize {
    if frame_length.is_multiple_of(SHELL_CODEC_FRAME_LENGTH) {
        frame_length
    } else {
        ((frame_length / SHELL_CODEC_FRAME_LENGTH) + 1) * SHELL_CODEC_FRAME_LENGTH
    }
}

fn quant_offset_as_i32(offset: FrameQuantizationOffsetType) -> i32 {
    match offset {
        FrameQuantizationOffsetType::Low => 0,
        FrameQuantizationOffsetType::High => 1,
    }
}

fn plc_summary(state: &DecoderState) -> PlcState {
    PlcState {
        rand_scale_q14: i32::from(state.plc_state.rand_scale_q14),
        prev_gain_q16: state.plc_state.prev_gain_q16,
    }
}

#[cfg(test)]
mod tests {
    use super::{silk_decode_frame, DecodeFlag};
    use crate::range::{RangeDecoder, RangeEncoder};
    use crate::silk::decode_indices::{ConditionalCoding, SideInfoIndices};
    use crate::silk::encode_indices::EncoderIndicesState;
    use crate::silk::encode_pulses::silk_encode_pulses;
    use crate::silk::decoder_state::DecoderState;
    use crate::silk::{FrameQuantizationOffsetType, FrameSignalType};
    use alloc::vec;

    fn configured_decoder_state() -> DecoderState {
        let mut state = DecoderState::default();
        state
            .sample_rate
            .set_sample_rates(16, 16_000)
            .expect("16 kHz configuration must succeed");
        state
    }

    #[test]
    fn packet_loss_path_updates_loss_counter() {
        let mut state = configured_decoder_state();
        let frame_length = state.sample_rate.frame_length;
        let mut decoder = RangeDecoder::init(&[]);
        let mut output = vec![0i16; frame_length];

        let produced = silk_decode_frame(
            &mut state,
            &mut decoder,
            &mut output,
            DecodeFlag::PacketLoss,
            ConditionalCoding::Independent,
            0,
        );

        assert_eq!(produced, frame_length);
        assert_eq!(state.loss_count, 1);
    }

    #[test]
    fn decodes_simple_unvoiced_frame() {
        let mut state = configured_decoder_state();
        let frame_length = state.sample_rate.frame_length;

        let mut encoder = RangeEncoder::new();
        let mut indices_state = EncoderIndicesState::default();
        let mut indices = SideInfoIndices::default();
        indices.signal_type = FrameSignalType::Unvoiced;
        indices.quant_offset_type = FrameQuantizationOffsetType::Low;
        indices.nlsf_interp_coef_q2 = 4;
        indices.gains_indices = [10, 6, 6, 6];
        indices.seed = 3;
        indices_state.encode_indices(&mut encoder, &indices, ConditionalCoding::Independent, false);

        let mut pulses = vec![0i8; frame_length];
        for (i, sample) in pulses.iter_mut().enumerate() {
            *sample = match i % 5 {
                0 => 3,
                1 => -2,
                2 => 0,
                3 => 1,
                _ => -1,
            };
        }
        silk_encode_pulses(
            &mut encoder,
            i32::from(indices.signal_type),
            0,
            &mut pulses,
            frame_length,
        );
        let payload = encoder.finish();

        let mut decoder = RangeDecoder::init(&payload);
        state.vad_flags[0] = 1;
        let mut output = vec![0i16; frame_length];
        let produced = silk_decode_frame(
            &mut state,
            &mut decoder,
            &mut output,
            DecodeFlag::Normal,
            ConditionalCoding::Independent,
            0,
        );

        assert_eq!(produced, frame_length);
        assert_eq!(state.loss_count, 0);
        assert!(!state.sample_rate.first_frame_after_reset);
        assert_eq!(state.sample_rate.prev_signal_type, FrameSignalType::Unvoiced);
        assert!(output.iter().any(|&sample| sample != 0));
    }
}
