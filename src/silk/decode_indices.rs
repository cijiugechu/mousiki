//! Port of `silk/decode_indices.c`.
//!
//! This module mirrors the SILK decoder helper that extracts the frame signal
//! type, quantisation gains, NLSF codebook indices, pitch lags, and LTP
//! metadata from the range-coded bitstream.  The routine sits between the
//! top-level `silk_decode_frame` driver and the parameter reconstruction
//! helpers, so it only touches the compact side information needed later in
//! the pipeline.

use crate::silk::SilkRangeDecoder;
use crate::silk::icdf::{
    DELTA_QUANTIZATION_GAIN, INDEPENDENT_QUANTIZATION_GAIN_LSB,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE, INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED,
};
use crate::silk::nlsf_unpack::nlsf_unpack;
use crate::silk::tables_ltp::{SILK_LTP_GAIN_ICDF, SILK_LTP_PER_INDEX_ICDF};
use crate::silk::tables_other::{
    SILK_LTPSCALE_ICDF, SILK_NLSF_EXT_ICDF, SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
    SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF, SILK_UNIFORM4_ICDF,
    SILK_UNIFORM8_ICDF,
};
use crate::silk::tables_pitch_lag::{PITCH_CONTOUR_ICDF, PITCH_DELTA_ICDF, PITCH_LAG_ICDF};
use crate::silk::{
    FrameQuantizationOffsetType, FrameSignalType, MAX_FRAMES_PER_PACKET, MAX_LPC_ORDER,
    MAX_NB_SUBFR, SilkNlsfCb,
};

const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
const NLSF_STAGE2_SYMBOLS: usize = (NLSF_QUANT_MAX_AMPLITUDE as usize * 2) + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionalCoding {
    Independent,
    IndependentNoLtpScaling,
    Conditional,
}

impl ConditionalCoding {
    fn is_independent(self) -> bool {
        matches!(
            self,
            ConditionalCoding::Independent | ConditionalCoding::IndependentNoLtpScaling
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideInfoIndices {
    pub gains_indices: [i8; MAX_NB_SUBFR],
    pub ltp_index: [i8; MAX_NB_SUBFR],
    pub nlsf_indices: [i8; MAX_LPC_ORDER + 1],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: FrameSignalType,
    pub quant_offset_type: FrameQuantizationOffsetType,
    pub nlsf_interp_coef_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

impl Default for SideInfoIndices {
    fn default() -> Self {
        Self {
            gains_indices: [0; MAX_NB_SUBFR],
            ltp_index: [0; MAX_NB_SUBFR],
            nlsf_indices: [0; MAX_LPC_ORDER + 1],
            lag_index: 0,
            contour_index: 0,
            signal_type: FrameSignalType::Inactive,
            quant_offset_type: FrameQuantizationOffsetType::Low,
            nlsf_interp_coef_q2: 4,
            per_index: 0,
            ltp_scale_index: 0,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecoderIndicesState {
    pub vad_flags: [bool; MAX_FRAMES_PER_PACKET],
    pub nb_subfr: usize,
    pub fs_khz: i32,
    pub lpc_order: usize,
    pub pitch_lag_low_bits_icdf: &'static [u8],
    pub pitch_contour_icdf: &'static [u8],
    pub nlsf_codebook: &'static SilkNlsfCb,
    pub prev_signal_type: FrameSignalType,
    pub prev_lag_index: i16,
}

impl DecoderIndicesState {
    pub fn new(nlsf_codebook: &'static SilkNlsfCb) -> Self {
        Self {
            vad_flags: [false; MAX_FRAMES_PER_PACKET],
            nb_subfr: MAX_NB_SUBFR,
            fs_khz: 16,
            lpc_order: MAX_LPC_ORDER,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM8_ICDF,
            pitch_contour_icdf: &PITCH_CONTOUR_ICDF,
            nlsf_codebook,
            prev_signal_type: FrameSignalType::Inactive,
            prev_lag_index: 0,
        }
    }

    pub fn decode_indices(
        &mut self,
        range_decoder: &mut impl SilkRangeDecoder,
        frame_index: usize,
        decode_lbrr: bool,
        coding: ConditionalCoding,
    ) -> SideInfoIndices {
        assert!(frame_index < MAX_FRAMES_PER_PACKET);
        assert!(self.nb_subfr == MAX_NB_SUBFR || self.nb_subfr == MAX_NB_SUBFR / 2);
        assert!(self.lpc_order <= MAX_LPC_ORDER);

        let mut indices = SideInfoIndices::default();
        let raw_type = if decode_lbrr || self.vad_flags[frame_index] {
            range_decoder.decode_icdf(&SILK_TYPE_OFFSET_VAD_ICDF, 8) as i32 + 2
        } else {
            range_decoder.decode_icdf(&SILK_TYPE_OFFSET_NO_VAD_ICDF, 8) as i32
        };

        indices.signal_type = match raw_type >> 1 {
            0 => FrameSignalType::Inactive,
            1 => FrameSignalType::Unvoiced,
            _ => FrameSignalType::Voiced,
        };
        indices.quant_offset_type = if raw_type & 1 == 0 {
            FrameQuantizationOffsetType::Low
        } else {
            FrameQuantizationOffsetType::High
        };

        self.decode_gains(range_decoder, coding, &mut indices);
        self.decode_nlsf(range_decoder, &mut indices);
        self.decode_pitch_and_ltp(range_decoder, coding, &mut indices);

        indices.seed = range_decoder.decode_icdf(&SILK_UNIFORM4_ICDF, 8) as i8;
        self.prev_signal_type = indices.signal_type;

        indices
    }

    fn decode_gains(
        &self,
        range_decoder: &mut impl SilkRangeDecoder,
        coding: ConditionalCoding,
        indices: &mut SideInfoIndices,
    ) {
        if coding.is_independent() {
            let msb_ctx = match indices.signal_type {
                FrameSignalType::Inactive => INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE,
                FrameSignalType::Unvoiced => INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED,
                FrameSignalType::Voiced => INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED,
            };
            let msb = range_decoder.decode_symbol_with_icdf(msb_ctx) as i32;
            let lsb =
                range_decoder.decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_LSB) as i32;
            indices.gains_indices[0] = ((msb << 3) | lsb) as i8;
        } else {
            indices.gains_indices[0] =
                range_decoder.decode_symbol_with_icdf(DELTA_QUANTIZATION_GAIN) as i8;
        }

        for subframe in 1..self.nb_subfr {
            indices.gains_indices[subframe] =
                range_decoder.decode_symbol_with_icdf(DELTA_QUANTIZATION_GAIN) as i8;
        }
    }

    fn decode_nlsf(
        &self,
        range_decoder: &mut impl SilkRangeDecoder,
        indices: &mut SideInfoIndices,
    ) {
        let stage_class = match indices.signal_type {
            FrameSignalType::Voiced => 1,
            _ => 0,
        };
        let vectors = self.nlsf_codebook.n_vectors as usize;
        let start = stage_class * vectors;
        debug_assert!(start + vectors <= self.nlsf_codebook.cb1_icdf.len());
        let stage1_index =
            range_decoder.decode_icdf(&self.nlsf_codebook.cb1_icdf[start..start + vectors], 8);
        indices.nlsf_indices[0] = stage1_index as i8;

        let order = self.lpc_order;
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred_q8 = [0u8; MAX_LPC_ORDER];
        nlsf_unpack(
            &mut ec_ix[..order],
            &mut pred_q8[..order],
            self.nlsf_codebook,
            stage1_index,
        );

        for (i, &entropy_offset) in ec_ix.iter().take(order).enumerate() {
            let offset = entropy_offset as usize;
            debug_assert!(offset + NLSF_STAGE2_SYMBOLS <= self.nlsf_codebook.ec_icdf.len());
            let mut symbol = range_decoder.decode_icdf(
                &self.nlsf_codebook.ec_icdf[offset..offset + NLSF_STAGE2_SYMBOLS],
                8,
            ) as i32;
            if symbol == 0 {
                symbol -= range_decoder.decode_icdf(&SILK_NLSF_EXT_ICDF, 8) as i32;
            } else if symbol == 2 * NLSF_QUANT_MAX_AMPLITUDE {
                symbol += range_decoder.decode_icdf(&SILK_NLSF_EXT_ICDF, 8) as i32;
            }
            indices.nlsf_indices[i + 1] = (symbol - NLSF_QUANT_MAX_AMPLITUDE) as i8;
        }

        indices.nlsf_interp_coef_q2 = if self.nb_subfr == MAX_NB_SUBFR {
            range_decoder.decode_icdf(&SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8
        } else {
            4
        };
    }

    fn decode_pitch_and_ltp(
        &mut self,
        range_decoder: &mut impl SilkRangeDecoder,
        coding: ConditionalCoding,
        indices: &mut SideInfoIndices,
    ) {
        if indices.signal_type != FrameSignalType::Voiced {
            return;
        }

        let mut decode_absolute = true;
        if matches!(coding, ConditionalCoding::Conditional)
            && self.prev_signal_type == FrameSignalType::Voiced
        {
            let delta = range_decoder.decode_icdf(&PITCH_DELTA_ICDF, 8) as i16;
            if delta > 0 {
                let adjusted = delta - 9;
                indices.lag_index = self.prev_lag_index.saturating_add(adjusted);
                decode_absolute = false;
            }
        }

        if decode_absolute {
            let lag_mult = self.fs_khz >> 1;
            debug_assert!(lag_mult > 0);
            let high = range_decoder.decode_icdf(&PITCH_LAG_ICDF, 8) as i32;
            let low = range_decoder.decode_icdf(self.pitch_lag_low_bits_icdf, 8) as i32;
            indices.lag_index = (high * lag_mult + low) as i16;
        }
        self.prev_lag_index = indices.lag_index;

        indices.contour_index = range_decoder.decode_icdf(self.pitch_contour_icdf, 8) as i8;

        let per_index = range_decoder.decode_icdf(&SILK_LTP_PER_INDEX_ICDF, 8);
        indices.per_index = per_index as i8;
        let gain_icdf = SILK_LTP_GAIN_ICDF[per_index];
        for k in 0..self.nb_subfr {
            indices.ltp_index[k] = range_decoder.decode_icdf(gain_icdf, 8) as i8;
        }

        indices.ltp_scale_index = if matches!(coding, ConditionalCoding::Independent) {
            range_decoder.decode_icdf(&SILK_LTPSCALE_ICDF, 8) as i8
        } else {
            0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt::EcDec;
    use crate::range::RangeEncoder;
    use crate::silk::nlsf_unpack::nlsf_unpack;
    use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;

    fn encode_stage_one(
        encoder: &mut RangeEncoder,
        signal_type: FrameSignalType,
        stage1_index: usize,
    ) {
        let vectors = SILK_NLSF_CB_WB.n_vectors as usize;
        let class = if matches!(signal_type, FrameSignalType::Voiced) {
            1
        } else {
            0
        };
        let start = class * vectors;
        encoder.encode_icdf(
            stage1_index,
            &SILK_NLSF_CB_WB.cb1_icdf[start..start + vectors],
            8,
        );
    }

    fn encode_stage_two(
        encoder: &mut RangeEncoder,
        stage1_index: usize,
        order: usize,
        symbol: usize,
    ) {
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred = [0u8; MAX_LPC_ORDER];
        nlsf_unpack(
            &mut ec_ix[..order],
            &mut pred[..order],
            &SILK_NLSF_CB_WB,
            stage1_index,
        );
        for i in 0..order {
            let offset = ec_ix[i] as usize;
            encoder.encode_icdf(
                symbol,
                &SILK_NLSF_CB_WB.ec_icdf[offset..offset + NLSF_STAGE2_SYMBOLS],
                8,
            );
        }
    }

    #[test]
    fn decodes_unvoiced_frame_indices() {
        let mut encoder = RangeEncoder::new();
        encoder.encode_icdf(1, &SILK_TYPE_OFFSET_VAD_ICDF, 8);
        encoder.encode_symbol_with_icdf(2, INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED);
        encoder.encode_symbol_with_icdf(5, INDEPENDENT_QUANTIZATION_GAIN_LSB);
        for &delta in &[0usize, 1, 2] {
            encoder.encode_symbol_with_icdf(delta, DELTA_QUANTIZATION_GAIN);
        }
        encode_stage_one(&mut encoder, FrameSignalType::Unvoiced, 3);
        encode_stage_two(&mut encoder, 3, MAX_LPC_ORDER, 4);
        encoder.encode_icdf(2, &SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8);
        encoder.encode_icdf(2, &SILK_UNIFORM4_ICDF, 8);
        let mut packet = encoder.finish();

        let mut decoder = EcDec::new(packet.as_mut_slice());
        let mut state = DecoderIndicesState::new(&SILK_NLSF_CB_WB);
        state.vad_flags[0] = true;
        let indices = state.decode_indices(&mut decoder, 0, false, ConditionalCoding::Independent);

        assert_eq!(indices.signal_type, FrameSignalType::Unvoiced);
        assert_eq!(indices.quant_offset_type, FrameQuantizationOffsetType::High);
        assert_eq!(indices.gains_indices, [21, 0, 1, 2]);
        assert_eq!(indices.nlsf_indices[0], 3);
        assert!(
            indices.nlsf_indices[1..=MAX_LPC_ORDER]
                .iter()
                .all(|value| *value == 0)
        );
        assert_eq!(indices.nlsf_interp_coef_q2, 2);
        assert_eq!(indices.seed, 2);
        assert_eq!(state.prev_signal_type, FrameSignalType::Unvoiced);
    }

    #[test]
    fn decodes_voiced_frame_with_delta_pitch() {
        let mut encoder = RangeEncoder::new();
        encoder.encode_icdf(3, &SILK_TYPE_OFFSET_VAD_ICDF, 8);
        encoder.encode_symbol_with_icdf(4, DELTA_QUANTIZATION_GAIN);
        for &delta in &[2usize, 3, 4] {
            encoder.encode_symbol_with_icdf(delta, DELTA_QUANTIZATION_GAIN);
        }
        encode_stage_one(&mut encoder, FrameSignalType::Voiced, 4);
        encode_stage_two(&mut encoder, 4, MAX_LPC_ORDER, 4);
        encoder.encode_icdf(1, &SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8);
        encoder.encode_icdf(10, &PITCH_DELTA_ICDF, 8);
        encoder.encode_icdf(2, &PITCH_CONTOUR_ICDF, 8);
        encoder.encode_icdf(1, &SILK_LTP_PER_INDEX_ICDF, 8);
        let gain_icdf = SILK_LTP_GAIN_ICDF[1];
        for symbol in 0..MAX_NB_SUBFR {
            encoder.encode_icdf(symbol, gain_icdf, 8);
        }
        encoder.encode_icdf(1, &SILK_UNIFORM4_ICDF, 8);
        let mut packet = encoder.finish();

        let mut decoder = EcDec::new(packet.as_mut_slice());
        let mut state = DecoderIndicesState::new(&SILK_NLSF_CB_WB);
        state.vad_flags[0] = true;
        state.prev_signal_type = FrameSignalType::Voiced;
        state.prev_lag_index = 100;
        let indices = state.decode_indices(&mut decoder, 0, false, ConditionalCoding::Conditional);

        assert_eq!(indices.signal_type, FrameSignalType::Voiced);
        assert_eq!(indices.gains_indices, [4, 2, 3, 4]);
        assert_eq!(indices.lag_index, 101);
        assert_eq!(indices.contour_index, 2);
        assert_eq!(indices.per_index, 1);
        assert_eq!(&indices.ltp_index[..MAX_NB_SUBFR], &[0, 1, 2, 3]);
        assert_eq!(indices.ltp_scale_index, 0);
        assert_eq!(indices.seed, 1);
        assert_eq!(state.prev_lag_index, 101);
        assert_eq!(state.prev_signal_type, FrameSignalType::Voiced);
    }
}
