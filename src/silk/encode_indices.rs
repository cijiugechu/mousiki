//! Port of `silk/encode_indices.c`.
//!
//! This module mirrors the SILK encoder helper that range-encodes the compact
//! side-information indices (signal type, gain steps, NLSF codebook entries,
//! pitch metadata, seed, etc.) before the excitation pulses are written.  It
//! complements `decode_indices.rs`, sharing the same [`SideInfoIndices`]
//! struct so encode/decode paths stay bit-exact.

use crate::range::RangeEncoder;
use crate::silk::decode_indices::{ConditionalCoding, SideInfoIndices};
use crate::silk::icdf::{
    DELTA_QUANTIZATION_GAIN, INDEPENDENT_QUANTIZATION_GAIN_LSB,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE, INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED,
};
use crate::silk::nlsf_unpack::nlsf_unpack;
use crate::silk::tables_gain::{DELTA_GAIN_QUANT_LEVELS, N_LEVELS_QGAIN};
use crate::silk::tables_ltp::{SILK_LTP_GAIN_ICDF, SILK_LTP_PER_INDEX_ICDF};
use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;
use crate::silk::tables_other::{
    SILK_LTPSCALE_ICDF, SILK_NLSF_EXT_ICDF, SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
    SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF, SILK_UNIFORM4_ICDF,
    SILK_UNIFORM8_ICDF,
};
use crate::silk::tables_pitch_lag::{PITCH_CONTOUR_ICDF, PITCH_DELTA_ICDF, PITCH_LAG_ICDF};
use crate::silk::{
    FrameQuantizationOffsetType, FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR, SilkNlsfCb,
};

const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
const NLSF_STAGE2_SYMBOLS: usize = (NLSF_QUANT_MAX_AMPLITUDE as usize * 2) + 1;

/// Minimal subset of the encoder state required by [`EncoderIndicesState::encode_indices`].
#[derive(Clone, Debug)]
pub struct EncoderIndicesState {
    /// Number of subframes per frame (2 or 4).
    pub nb_subfr: usize,
    /// Internal sampling rate (kHz).
    pub fs_khz: i32,
    /// Prediction filter order (10 or 16 in the reference implementation).
    pub predict_lpc_order: usize,
    /// Active NLSF codebook.
    pub nlsf_codebook: &'static SilkNlsfCb,
    /// iCDF table used for the pitch-lag low bits.
    pub pitch_lag_low_bits_icdf: &'static [u8],
    /// iCDF table used for the pitch contour indices.
    pub pitch_contour_icdf: &'static [u8],
    /// Previous encoded signal classification (tracks conditional coding history).
    pub prev_signal_type: FrameSignalType,
    /// Previous encoded lag index (Q0 samples).
    pub prev_lag_index: i16,
}

impl Default for EncoderIndicesState {
    fn default() -> Self {
        Self {
            nb_subfr: MAX_NB_SUBFR,
            fs_khz: 16,
            predict_lpc_order: MAX_LPC_ORDER,
            nlsf_codebook: &SILK_NLSF_CB_WB,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM8_ICDF,
            pitch_contour_icdf: &PITCH_CONTOUR_ICDF,
            prev_signal_type: FrameSignalType::Inactive,
            prev_lag_index: 0,
        }
    }
}

impl EncoderIndicesState {
    /// Construct an indices state backed by the provided NLSF codebook.
    pub fn new(nlsf_codebook: &'static SilkNlsfCb) -> Self {
        Self {
            nlsf_codebook,
            ..Self::default()
        }
    }

    /// Mirrors `silk_encode_indices`, range-encoding the per-frame metadata.
    pub fn encode_indices(
        &mut self,
        range_encoder: &mut RangeEncoder,
        indices: &SideInfoIndices,
        coding: ConditionalCoding,
        encode_lbrr: bool,
    ) {
        assert!(
            self.nb_subfr == MAX_NB_SUBFR || self.nb_subfr == MAX_NB_SUBFR / 2,
            "nb_subfr must be 2 or 4"
        );
        assert!(
            self.predict_lpc_order <= MAX_LPC_ORDER,
            "predict_lpc_order exceeds MAX_LPC_ORDER"
        );

        self.encode_signal_type(range_encoder, indices, encode_lbrr);
        self.encode_gains(range_encoder, indices, coding);
        self.encode_nlsf(range_encoder, indices);

        if self.nb_subfr == MAX_NB_SUBFR {
            let coef = indices.nlsf_interp_coef_q2 as usize;
            range_encoder.encode_icdf(coef, &SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8);
        }

        self.encode_pitch_and_ltp(range_encoder, indices, coding);

        range_encoder.encode_icdf(indices.seed as usize, &SILK_UNIFORM4_ICDF, 8);

        self.prev_signal_type = indices.signal_type;
        self.prev_lag_index = indices.lag_index;
    }

    fn encode_signal_type(
        &self,
        encoder: &mut RangeEncoder,
        indices: &SideInfoIndices,
        encode_lbrr: bool,
    ) {
        let quant_offset = match indices.quant_offset_type {
            FrameQuantizationOffsetType::Low => 0,
            FrameQuantizationOffsetType::High => 1,
        };
        let type_offset = 2 * i32::from(indices.signal_type) + quant_offset;
        if encode_lbrr || type_offset >= 2 {
            encoder.encode_icdf((type_offset - 2) as usize, &SILK_TYPE_OFFSET_VAD_ICDF, 8);
        } else {
            encoder.encode_icdf(type_offset as usize, &SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
        }
    }

    fn encode_gains(
        &self,
        encoder: &mut RangeEncoder,
        indices: &SideInfoIndices,
        coding: ConditionalCoding,
    ) {
        if matches!(
            coding,
            ConditionalCoding::Independent | ConditionalCoding::IndependentNoLtpScaling
        ) {
            let gain_index = i32::from(indices.gains_indices[0]);
            assert!(
                (0..N_LEVELS_QGAIN as i32).contains(&gain_index),
                "gain index exceeds N_LEVELS_QGAIN"
            );
            let msb_ctx = match indices.signal_type {
                FrameSignalType::Inactive => INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE,
                FrameSignalType::Unvoiced => INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED,
                FrameSignalType::Voiced => INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED,
            };
            let msb = (gain_index >> 3) as usize;
            let lsb = (gain_index & 7) as usize;
            encoder.encode_symbol_with_icdf(msb, msb_ctx);
            encoder.encode_symbol_with_icdf(lsb, INDEPENDENT_QUANTIZATION_GAIN_LSB);
        } else {
            let gain_index = i32::from(indices.gains_indices[0]);
            assert!(
                (0..DELTA_GAIN_QUANT_LEVELS as i32).contains(&gain_index),
                "delta gain index out of range"
            );
            encoder.encode_symbol_with_icdf(gain_index as usize, DELTA_QUANTIZATION_GAIN);
        }

        for subframe in 1..self.nb_subfr {
            let gain_index = i32::from(indices.gains_indices[subframe]);
            assert!(
                (0..DELTA_GAIN_QUANT_LEVELS as i32).contains(&gain_index),
                "delta gain index out of range"
            );
            encoder.encode_symbol_with_icdf(gain_index as usize, DELTA_QUANTIZATION_GAIN);
        }
    }

    fn encode_nlsf(&self, encoder: &mut RangeEncoder, indices: &SideInfoIndices) {
        let order = self.predict_lpc_order;
        assert!(
            order <= self.nlsf_codebook.order as usize,
            "predict_lpc_order exceeds codebook order"
        );
        let vectors = self.nlsf_codebook.n_vectors as usize;
        let stage_class = match indices.signal_type {
            FrameSignalType::Voiced => 1,
            _ => 0,
        };
        let stage1_index = indices.nlsf_indices[0] as usize;
        assert!(
            stage1_index < vectors,
            "stage-one NLSF index exceeds codebook entries"
        );
        let offset = stage_class * vectors;
        encoder.encode_icdf(
            stage1_index,
            &self.nlsf_codebook.cb1_icdf[offset..offset + vectors],
            8,
        );

        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred_q8 = [0u8; MAX_LPC_ORDER];
        nlsf_unpack(
            &mut ec_ix[..order],
            &mut pred_q8[..order],
            self.nlsf_codebook,
            stage1_index,
        );

        for (i, &entropy_offset) in ec_ix.iter().take(order).enumerate() {
            let symbol = i32::from(indices.nlsf_indices[i + 1]);
            let table_start = entropy_offset as usize;
            let table = &self.nlsf_codebook.ec_icdf[table_start..table_start + NLSF_STAGE2_SYMBOLS];
            if symbol >= NLSF_QUANT_MAX_AMPLITUDE {
                encoder.encode_icdf((2 * NLSF_QUANT_MAX_AMPLITUDE) as usize, table, 8);
                encoder.encode_icdf(
                    (symbol - NLSF_QUANT_MAX_AMPLITUDE) as usize,
                    &SILK_NLSF_EXT_ICDF,
                    8,
                );
            } else if symbol <= -NLSF_QUANT_MAX_AMPLITUDE {
                encoder.encode_icdf(0, table, 8);
                encoder.encode_icdf(
                    (-symbol - NLSF_QUANT_MAX_AMPLITUDE) as usize,
                    &SILK_NLSF_EXT_ICDF,
                    8,
                );
            } else {
                encoder.encode_icdf((symbol + NLSF_QUANT_MAX_AMPLITUDE) as usize, table, 8);
            }
        }
    }

    fn encode_pitch_and_ltp(
        &mut self,
        encoder: &mut RangeEncoder,
        indices: &SideInfoIndices,
        coding: ConditionalCoding,
    ) {
        if indices.signal_type != FrameSignalType::Voiced {
            return;
        }

        let mut encode_absolute_lag = true;
        if matches!(coding, ConditionalCoding::Conditional)
            && self.prev_signal_type == FrameSignalType::Voiced
        {
            let delta = i32::from(indices.lag_index) - i32::from(self.prev_lag_index);
            let mut symbol = 0;
            if (-8..=11).contains(&delta) {
                encode_absolute_lag = false;
                symbol = (delta + 9) as usize;
            }
            encoder.encode_icdf(symbol, &PITCH_DELTA_ICDF, 8);
        }

        if encode_absolute_lag {
            let fs_div_2 = self.fs_khz >> 1;
            assert!(fs_div_2 > 0, "fs_khz must be positive");
            let lag = i32::from(indices.lag_index);
            let pitch_high_bits = (lag / fs_div_2) as usize;
            let pitch_low_bits = (lag - (pitch_high_bits as i32 * fs_div_2)) as usize;
            encoder.encode_icdf(pitch_high_bits, &PITCH_LAG_ICDF, 8);
            encoder.encode_icdf(pitch_low_bits, self.pitch_lag_low_bits_icdf, 8);
        }
        self.prev_lag_index = indices.lag_index;

        encoder.encode_icdf(indices.contour_index as usize, self.pitch_contour_icdf, 8);

        let per_index = indices.per_index as usize;
        encoder.encode_icdf(per_index, &SILK_LTP_PER_INDEX_ICDF, 8);
        let gain_table = SILK_LTP_GAIN_ICDF[per_index];
        for subframe in 0..self.nb_subfr {
            encoder.encode_icdf(indices.ltp_index[subframe] as usize, gain_table, 8);
        }

        if matches!(coding, ConditionalCoding::Independent) {
            encoder.encode_icdf(indices.ltp_scale_index as usize, &SILK_LTPSCALE_ICDF, 8);
        } else {
            debug_assert_eq!(
                indices.ltp_scale_index, 0,
                "conditional coding expects zero LTP scale index"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::RangeDecoder;
    use crate::silk::decode_indices::DecoderIndicesState;
    use crate::silk::tables_pitch_lag::PITCH_CONTOUR_ICDF;

    #[test]
    fn encode_decode_unvoiced_independent_frame() {
        let mut encoder_state = EncoderIndicesState::default();
        let mut indices = SideInfoIndices::default();
        indices.signal_type = FrameSignalType::Unvoiced;
        indices.quant_offset_type = FrameQuantizationOffsetType::High;
        indices.gains_indices = [21, 0, 1, 2];
        indices.nlsf_indices[0] = 3;
        indices.nlsf_interp_coef_q2 = 2;
        indices.seed = 2;

        let mut encoder = RangeEncoder::new();
        encoder_state.encode_indices(
            &mut encoder,
            &indices,
            ConditionalCoding::Independent,
            false,
        );
        let payload = encoder.finish();

        let mut decoder = RangeDecoder::init(&payload);
        let mut decoder_state = DecoderIndicesState::new(&SILK_NLSF_CB_WB);
        decoder_state.vad_flags[0] = true;
        let decoded =
            decoder_state.decode_indices(&mut decoder, 0, false, ConditionalCoding::Independent);

        assert_eq!(decoded.signal_type, indices.signal_type);
        assert_eq!(decoded.quant_offset_type, indices.quant_offset_type);
        assert_eq!(decoded.gains_indices, indices.gains_indices);
        assert_eq!(decoded.nlsf_indices, indices.nlsf_indices);
        assert_eq!(decoded.nlsf_interp_coef_q2, indices.nlsf_interp_coef_q2);
        assert_eq!(decoded.seed, indices.seed);
    }

    #[test]
    fn encode_decode_voiced_conditional_frame_with_delta_pitch() {
        let mut encoder_state = EncoderIndicesState {
            prev_signal_type: FrameSignalType::Voiced,
            prev_lag_index: 120,
            pitch_contour_icdf: &PITCH_CONTOUR_ICDF,
            ..EncoderIndicesState::default()
        };
        let mut indices = SideInfoIndices::default();
        indices.signal_type = FrameSignalType::Voiced;
        indices.quant_offset_type = FrameQuantizationOffsetType::Low;
        indices.gains_indices = [4, 3, 2, 1];
        indices.nlsf_indices[0] = 4;
        indices.nlsf_indices[1] = 5;
        indices.nlsf_indices[2] = -5;
        indices.lag_index = 125;
        indices.contour_index = 10;
        indices.per_index = 1;
        indices.ltp_index = [3, 5, 7, 9];
        indices.ltp_scale_index = 0;
        indices.seed = 1;

        let mut encoder = RangeEncoder::new();
        encoder_state.encode_indices(
            &mut encoder,
            &indices,
            ConditionalCoding::Conditional,
            false,
        );
        assert_eq!(encoder_state.prev_signal_type, indices.signal_type);
        assert_eq!(encoder_state.prev_lag_index, indices.lag_index);
        let payload = encoder.finish();

        let mut decoder = RangeDecoder::init(&payload);
        let mut decoder_state = DecoderIndicesState::new(&SILK_NLSF_CB_WB);
        decoder_state.vad_flags[0] = true;
        decoder_state.prev_signal_type = FrameSignalType::Voiced;
        decoder_state.prev_lag_index = 120;
        let decoded =
            decoder_state.decode_indices(&mut decoder, 0, false, ConditionalCoding::Conditional);

        assert_eq!(decoded.signal_type, indices.signal_type);
        assert_eq!(decoded.quant_offset_type, indices.quant_offset_type);
        assert_eq!(decoded.gains_indices, indices.gains_indices);
        assert_eq!(decoded.nlsf_indices, indices.nlsf_indices);
        assert_eq!(decoded.lag_index, indices.lag_index);
        assert_eq!(decoded.contour_index, indices.contour_index);
        assert_eq!(decoded.per_index, indices.per_index);
        assert_eq!(decoded.ltp_index, indices.ltp_index);
        assert_eq!(decoded.ltp_scale_index, indices.ltp_scale_index);
        assert_eq!(decoded.seed, indices.seed);
    }
}
