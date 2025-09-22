mod nomarlize;

use super::icdf::{FRAME_TYPE_VAD_ACTIVE, FRAME_TYPE_VAD_INACTIVE};
use super::{FrameQuantizationOffsetType, FrameSignalType};
use crate::math::ilog;
use crate::packet::Bandwidth;
use crate::range::RangeDecoder;
use crate::silk::codebook::{
    LSF_ORDERING_FOR_POLYNOMIAL_EVALUATION_NARROWBAND_AND_MEDIUMBAND,
    LSF_ORDERING_FOR_POLYNOMIAL_EVALUATION_WIDEBAND,
    MINIMUM_SPACING_FOR_NORMALIZED_LSCOEFFICIENTS_NARROWBAND_AND_MEDIUMBAND,
    MINIMUM_SPACING_FOR_NORMALIZED_LSCOEFFICIENTS_WIDEBAND,
    NORMALIZED_LSF_STAGE_ONE_NARROWBAND_OR_MEDIUMBAND, NORMALIZED_LSF_STAGE_ONE_WIDEBAND,
    NORMALIZED_LSF_STAGE_TWO_INDEX_NARROWBAND_OR_MEDIUMBAND,
    NORMALIZED_LSF_STAGE_TWO_INDEX_WIDEBAND,
    PREDICTION_WEIGHT_FOR_NARROWBAND_AND_MEDIUMBAND_NORMALIZED_LSF,
    PREDICTION_WEIGHT_FOR_WIDEBAND_NORMALIZED_LSF,
    PREDICTION_WEIGHT_SELECTION_FOR_NARROWBAND_AND_MEDIUMBAND_NORMALIZED_LSF,
    PREDICTION_WEIGHT_SELECTION_FOR_WIDEBAND_NORMALIZED_LSF, Q12_COSINE_TABLE_FOR_LSFCONVERION,
    SUBFRAME_PITCH_COUNTER_MEDIUMBAND_OR_WIDEBAND20_MS, SUBFRAME_PITCH_COUNTER_NARROWBAND20_MS,
};
use crate::silk::icdf::{
    DELTA_QUANTIZATION_GAIN, INDEPENDENT_QUANTIZATION_GAIN_LSB,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE, INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED, NORMALIZED_LSF_INTERPOLATION_INDEX,
    NORMALIZED_LSF_STAGE_1_INDEX_NARROWBAND_OR_MEDIUMBAND_UNVOICED,
    NORMALIZED_LSF_STAGE_1_INDEX_NARROWBAND_OR_MEDIUMBAND_VOICED,
    NORMALIZED_LSF_STAGE_1_INDEX_WIDEBAND_UNVOICED, NORMALIZED_LSF_STAGE_1_INDEX_WIDEBAND_VOICED,
    NORMALIZED_LSF_STAGE_2_INDEX, NORMALIZED_LSF_STAGE_2_INDEX_EXTENSION,
    PRIMARY_PITCH_LAG_HIGH_PART, PRIMARY_PITCH_LAG_LOW_PART_MEDIUMBAND,
    PRIMARY_PITCH_LAG_LOW_PART_NARROWBAND, PRIMARY_PITCH_LAG_LOW_PART_WIDEBAND,
    SUBFRAME_PITCH_CONTOUR_MEDIUMBAND_OR_WIDEBAND20_MS, SUBFRAME_PITCH_CONTOUR_NARROWBAND20_MS,
};
use core::fmt;
use nomarlize::{A32Q17, Aq12Coefficients, Aq12List, MAX_D_LPC, MAX_D2_LPC, NlsfQ15, ResQ10};

#[derive(Debug)]
pub struct DecoderBuilder {
    final_out_values: [f32; 306],
}

const INIT_OUT_VALUES: [f32; 306] = [0.; 306];

impl DecoderBuilder {
    pub const fn new() -> Self {
        Self {
            final_out_values: INIT_OUT_VALUES,
        }
    }

    pub fn build<'a>(self, buf: &'a [u8]) -> Decoder<'a> {
        Decoder {
            range_decoder: RangeDecoder::init(buf),
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: self.final_out_values,
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        }
    }
}

impl core::default::Default for DecoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Decoder maintains the state needed to decode a stream
/// of Silk frames.
#[derive(Debug)]
pub struct Decoder<'a> {
    range_decoder: RangeDecoder<'a>,
    // Have we decoded a frame yet?
    have_decoded: bool,
    #[allow(dead_code)]
    is_previous_frame_voiced: bool,
    previous_log_gain: i32,
    #[allow(dead_code)]
    final_out_values: [f32; 306],
    n0_q15: [i16; MAX_D_LPC],
    n0_q15_len: usize,
}

const SUBFRAME_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PitchLagInfo {
    pub lag_max: u32,
    pub pitch_lags: [i16; SUBFRAME_COUNT],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePitchLagsError {
    NonAbsoluteLagsUnsupported,
    UnsupportedBandwidth,
}

impl fmt::Display for DecodePitchLagsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonAbsoluteLagsUnsupported => {
                f.write_str("silk decoder does not support non-absolute lags")
            }
            Self::UnsupportedBandwidth => {
                f.write_str("unsupported bandwidth for pitch lag decoding")
            }
        }
    }
}

impl<'a> Decoder<'a> {
    /// Each SILK frame contains a single "frame type" symbol that jointly
    /// codes the signal type and quantization offset type of the
    /// corresponding frame.
    ///
    /// See [section-4.2.7.3](https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.3)
    pub fn determine_frame_type(
        &mut self,
        voice_activity_detected: bool,
    ) -> (FrameSignalType, FrameQuantizationOffsetType) {
        let frame_type_symbol = if voice_activity_detected {
            self.range_decoder
                .decode_symbol_with_icdf(FRAME_TYPE_VAD_ACTIVE)
        } else {
            self.range_decoder
                .decode_symbol_with_icdf(FRAME_TYPE_VAD_INACTIVE)
        };

        // +------------+-------------+--------------------------+
        // | Frame Type | Signal Type | Quantization Offset Type |
        // +------------+-------------+--------------------------+
        // | 0          | Inactive    |                      Low |
        // |            |             |                          |
        // | 1          | Inactive    |                     High |
        // |            |             |                          |
        // | 2          | Unvoiced    |                      Low |
        // |            |             |                          |
        // | 3          | Unvoiced    |                     High |
        // |            |             |                          |
        // | 4          | Voiced      |                      Low |
        // |            |             |                          |
        // | 5          | Voiced      |                     High |
        // +------------+-------------+--------------------------+
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.3

        match (voice_activity_detected, frame_type_symbol) {
            (false, 0) => (FrameSignalType::Inactive, FrameQuantizationOffsetType::Low),
            (false, _) => (FrameSignalType::Inactive, FrameQuantizationOffsetType::High),
            (true, 0) => (FrameSignalType::Unvoiced, FrameQuantizationOffsetType::Low),
            (true, 1) => (FrameSignalType::Unvoiced, FrameQuantizationOffsetType::High),
            (true, 2) => (FrameSignalType::Voiced, FrameQuantizationOffsetType::Low),
            (true, 3) => (FrameSignalType::Voiced, FrameQuantizationOffsetType::High),
            _ => unreachable!(),
        }
    }

    /// A separate quantization gain is coded for each 5 ms subframe
    ///
    /// See [section-4.2.7.4](https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.4)
    #[allow(unused_assignments)]
    pub fn decode_subframe_quantizations(
        &mut self,
        signal_type: FrameSignalType,
    ) -> [f32; SUBFRAME_COUNT] {
        let mut log_gain = 0;
        let mut delta_gain_index = 0;
        let mut gain_index: i32 = 0;
        let mut gain_q_16 = [0f32; SUBFRAME_COUNT];

        for (subframe_index, gain_value) in gain_q_16.iter_mut().enumerate() {
            // The subframe gains are either coded independently, or relative to the
            // gain from the most recent coded subframe in the same channel.
            //
            // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.4
            if subframe_index == 0 {
                // In an independently coded subframe gain, the 3 most significant bits
                // of the quantization gain are decoded using a PDF selected from
                // Table 11 based on the decoded signal type
                gain_index = match signal_type {
                    FrameSignalType::Inactive => self
                        .range_decoder
                        .decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE)
                        as i32,
                    FrameSignalType::Voiced => self
                        .range_decoder
                        .decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED)
                        as i32,
                    FrameSignalType::Unvoiced => self
                        .range_decoder
                        .decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED)
                        as i32,
                };

                // The 3 least significant bits are decoded using a uniform PDF:
                // These 6 bits are combined to form a value, gain_index, between 0 and 63.
                gain_index = (gain_index << 3)
                    | ((self
                        .range_decoder
                        .decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_LSB))
                        as i32);

                // When the gain for the previous subframe is available, then the
                // current gain is limited as follows:
                //     log_gain = max(gain_index, previous_log_gain - 16)
                if self.have_decoded {
                    log_gain = gain_index.max(self.previous_log_gain - 16)
                } else {
                    log_gain = gain_index
                }
            } else {
                // For subframes that do not have an independent gain (including the
                // first subframe of frames not listed as using independent coding
                // above), the quantization gain is coded relative to the gain from the
                // previous subframe
                delta_gain_index = self
                    .range_decoder
                    .decode_symbol_with_icdf(DELTA_QUANTIZATION_GAIN)
                    as i32;

                // The following formula translates this index into a quantization gain
                // for the current subframe using the gain from the previous subframe:
                //      log_gain = clamp(0, max(2*delta_gain_index - 16, previous_log_gain + delta_gain_index - 4), 63)
                log_gain = (2 * delta_gain_index - 16)
                    .max(self.previous_log_gain + delta_gain_index - 4)
                    .clamp(0, 63);
            }

            self.previous_log_gain = log_gain;

            // silk_gains_dequant() (gain_quant.c) dequantizes log_gain for the k'th
            // subframe and converts it into a linear Q16 scale factor via
            //
            //       gain_Q16[k] = silk_log2lin((0x1D1C71*log_gain>>16) + 2090)
            //
            let in_log_q7 = ((0x1D1C71 * log_gain) >> 16) + 2090;
            let i = in_log_q7 >> 7; // integer exponent
            let f = in_log_q7 & 127; // fractional exponent

            // The function silk_log2lin() (log2lin.c) computes an approximation of
            // 2**(inLog_Q7/128.0), where inLog_Q7 is its Q7 input.  Let i =
            // inLog_Q7>>7 be the integer part of inLogQ7 and f = inLog_Q7&127 be
            // the fractional part.  Then,
            //
            //             (1<<i) + ((-174*f*(128-f)>>16)+f)*((1<<i)>>7)
            //
            // yields the approximate exponential.  The final Q16 gain values lies
            // between 81920 and 1686110208, inclusive (representing scale factors
            // of 1.25 to 25728, respectively).

            *gain_value =
                ((1 << i) + (((-174 * f * (128 - f)) >> 16) + f) * ((1 << i) >> 7)) as f32;
        }

        gain_q_16
    }

    /// A set of normalized Line Spectral Frequency (LSF) coefficients follow
    /// the quantization gains in the bitstream and represent the Linear
    /// Predictive Coding (LPC) coefficients for the current SILK frame.
    ///
    /// See [Section-4.2.7.5.1](https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.1).
    pub fn normalize_line_spectral_frequency_stage_one(
        &mut self,
        voice_activity_detected: bool,
        bandwidth: Bandwidth,
    ) -> u32 {
        // The first VQ stage uses a 32-element codebook, coded with one of the
        // PDFs in Table 14, depending on the audio bandwidth and the signal
        // type of the current SILK frame.  This yields a single index, I1, for
        // the entire frame, which
        //
        // 1.  Indexes an element in a coarse codebook,
        // 2.  Selects the PDFs for the second stage of the VQ, and
        // 3.  Selects the prediction weights used to remove intra-frame
        //     redundancy from the second stage.
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.1
        use Bandwidth::*;

        match (voice_activity_detected, bandwidth) {
            (false, Narrow | Medium) => self.range_decoder.decode_symbol_with_icdf(
                NORMALIZED_LSF_STAGE_1_INDEX_NARROWBAND_OR_MEDIUMBAND_UNVOICED,
            ),
            (true, Narrow | Medium) => self.range_decoder.decode_symbol_with_icdf(
                NORMALIZED_LSF_STAGE_1_INDEX_NARROWBAND_OR_MEDIUMBAND_VOICED,
            ),
            (false, Wide) => self
                .range_decoder
                .decode_symbol_with_icdf(NORMALIZED_LSF_STAGE_1_INDEX_WIDEBAND_UNVOICED),
            (true, Wide) => self
                .range_decoder
                .decode_symbol_with_icdf(NORMALIZED_LSF_STAGE_1_INDEX_WIDEBAND_VOICED),
            (_, _) => unimplemented!(),
        }
    }

    /// A set of normalized Line Spectral Frequency (LSF) coefficients follow
    /// the quantization gains in the bitstream and represent the Linear
    /// Predictive Coding (LPC) coefficients for the current SILK frame.
    ///
    /// see [section-4.2.7.5.2](https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.2).
    pub fn normalize_line_spectral_frequency_stage_two(
        &mut self,
        bandwidth: Bandwidth,
        i1: u32,
    ) -> ResQ10 {
        // Decoding the second stage residual proceeds as follows.  For each
        // coefficient, the decoder reads a symbol using the PDF corresponding
        // to I1 from either Table 17 or Table 18,
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.2
        let codebook = if bandwidth == Bandwidth::Wide {
            // codebookNormalizedLSFStageTwoIndexWideband
            NORMALIZED_LSF_STAGE_TWO_INDEX_WIDEBAND
        } else {
            // codebookNormalizedLSFStageTwoIndexNarrowbandOrMediumband
            NORMALIZED_LSF_STAGE_TWO_INDEX_NARROWBAND_OR_MEDIUMBAND
        };

        let mut i2 = [0i8; MAX_D_LPC];
        let actual_i2_len = codebook[0].len();
        for i in 0..actual_i2_len {
            // the decoder reads a symbol using the PDF corresponding
            // to I1 from either Table 17 or Table 18 and subtracts 4 from the
            // result to give an index in the range -4 to 4, inclusive.
            //
            // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.2
            i2[i] = (self.range_decoder.decode_symbol_with_icdf(
                NORMALIZED_LSF_STAGE_2_INDEX[codebook[i1 as usize][i] as usize],
            )) as i8
                - 4;

            // If the index is either -4 or 4, it reads a second symbol using the PDF in
            // Table 19, and adds the value of this second symbol to the index,
            // using the same sign.  This gives the index, I2[k], a total range of
            // -10 to 10, inclusive.
            //
            // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.2
            if i2[i] == -4 {
                i2[i] -= (self
                    .range_decoder
                    .decode_symbol_with_icdf(NORMALIZED_LSF_STAGE_2_INDEX_EXTENSION))
                    as i8;
            } else if i2[i] == 4 {
                i2[i] += (self
                    .range_decoder
                    .decode_symbol_with_icdf(NORMALIZED_LSF_STAGE_2_INDEX_EXTENSION))
                    as i8;
            }
        }

        // The decoded indices from both stages are translated back into
        // normalized LSF coefficients. The stage-2 indices represent residuals
        // after both the first stage of the VQ and a separate backwards-prediction
        // step. The backwards prediction process in the encoder subtracts a prediction
        // from each residual formed by a multiple of the coefficient that follows it.
        // The decoder must undo this process.
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.2

        // qstep is the Q16 quantization step size, which is 11796 for NB and MB and 9830
        // for WB (representing step sizes of approximately 0.18 and 0.15, respectively).
        let qstep = if bandwidth == Bandwidth::Wide {
            9830
        } else {
            11796
        };

        // stage-2 residual
        let mut res_q10 = [0i16; 16];

        // Let d_LPC be the order of the codebook, i.e., 10 for NB and MB, and 16 for WB
        let d_lpc = actual_i2_len;

        // for 0 <= k < d_LPC-1
        for k in (0..=(d_lpc - 1)).rev() {
            // The stage-2 residual for each coefficient is computed via
            //
            //     res_Q10[k] = (k+1 < d_LPC ? (res_Q10[k+1]*pred_Q8[k])>>8 : 0) + ((((I2[k]<<10) - sign(I2[k])*102)*qstep)>>16) ,
            //

            // The following computes
            //
            // (k+1 < d_LPC ? (res_Q10[k+1]*pred_Q8[k])>>8 : 0)
            //
            let mut first_operand = 0;
            if k + 1 < d_lpc {
                // Each coefficient selects its prediction weight from one of the two lists based on the stage-1 index, I1.
                // let pred_Q8[k] be the weight for the k'th coefficient selected by this process for 0 <= k < d_LPC-1
                let pred_q8 = if bandwidth == Bandwidth::Wide {
                    PREDICTION_WEIGHT_FOR_WIDEBAND_NORMALIZED_LSF
                        [PREDICTION_WEIGHT_SELECTION_FOR_WIDEBAND_NORMALIZED_LSF[i1 as usize][k]
                            as usize][k] as isize
                } else {
                    PREDICTION_WEIGHT_FOR_NARROWBAND_AND_MEDIUMBAND_NORMALIZED_LSF
                        [PREDICTION_WEIGHT_SELECTION_FOR_NARROWBAND_AND_MEDIUMBAND_NORMALIZED_LSF
                            [i1 as usize][k] as usize][k] as isize
                };

                first_operand = ((res_q10[k + 1] as isize) * pred_q8) >> 8;
            }

            // The following computes
            //
            // (((I2[k]<<10) - sign(I2[k])*102)*qstep)>>16
            //.
            let i2k = i2[k] as isize;
            let second_operand = (((i2k << 10) - (i2k.signum() * 102)) * (qstep as isize)) >> 16;

            res_q10[k] = (first_operand + second_operand) as i16;
        }

        if actual_i2_len == 10 {
            ResQ10::NarrowOrMedium(res_q10[0..10].try_into().unwrap())
        } else {
            ResQ10::Wide(res_q10)
        }
    }
    /// https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.5
    pub fn normalize_lsf_interpolation(&mut self, n2_q15: &[i16]) -> (Option<NlsfQ15>, i16) {
        let w_q2 = self
            .range_decoder
            .decode_symbol_with_icdf(NORMALIZED_LSF_INTERPOLATION_INDEX) as i16;

        if w_q2 == 4 || !self.have_decoded {
            return (None, w_q2);
        }

        debug_assert!(n2_q15.len() <= MAX_D_LPC);
        debug_assert!(self.n0_q15_len >= n2_q15.len());

        let mut n1_q15 = NlsfQ15::new(n2_q15.len());
        let w_q2_i32 = i32::from(w_q2);

        for (idx, n1_value) in n1_q15.as_mut_slice().iter_mut().enumerate() {
            let prev = self.n0_q15[idx];
            let diff = i32::from(n2_q15[idx]) - i32::from(prev);
            let interpolated = i32::from(prev) + ((w_q2_i32 * diff) >> 2);
            *n1_value = interpolated as i16;
        }

        (Some(n1_q15), w_q2)
    }

    #[allow(dead_code)]
    fn generate_a_q12(
        &mut self,
        q15: Option<&NlsfQ15>,
        bandwidth: Bandwidth,
        a_q12: &mut Aq12List,
    ) {
        if let Some(q15_values) = q15 {
            let mut a32_q17 =
                self.convert_normalized_lsfs_to_lpc_coefficients(q15_values, bandwidth);
            self.limit_lpc_coefficients_range(&mut a32_q17);
            let aq12_coeffs = self.limit_lpc_filter_prediction_gain(&a32_q17);
            a_q12.push(&aq12_coeffs);
        }
    }

    /// https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.6.1
    pub fn decode_pitch_lags(
        &mut self,
        signal_type: FrameSignalType,
        bandwidth: Bandwidth,
    ) -> Result<Option<PitchLagInfo>, DecodePitchLagsError> {
        if signal_type != FrameSignalType::Voiced {
            return Ok(None);
        }

        let lag_absolute = true;
        let (lag, lag_min, lag_max) = if lag_absolute {
            let (low_part_icdf, lag_scale, lag_min, lag_max) = match bandwidth {
                Bandwidth::Narrow => (PRIMARY_PITCH_LAG_LOW_PART_NARROWBAND, 4, 16, 144),
                Bandwidth::Medium => (PRIMARY_PITCH_LAG_LOW_PART_MEDIUMBAND, 6, 24, 216),
                Bandwidth::Wide => (PRIMARY_PITCH_LAG_LOW_PART_WIDEBAND, 8, 32, 288),
                _ => return Err(DecodePitchLagsError::UnsupportedBandwidth),
            };

            let lag_high = self
                .range_decoder
                .decode_symbol_with_icdf(PRIMARY_PITCH_LAG_HIGH_PART);
            let lag_low = self.range_decoder.decode_symbol_with_icdf(low_part_icdf);

            (lag_high * lag_scale + lag_low + lag_min, lag_min, lag_max)
        } else {
            return Err(DecodePitchLagsError::NonAbsoluteLagsUnsupported);
        };

        let (lag_cb, lag_icdf) = match bandwidth {
            Bandwidth::Narrow => (
                SUBFRAME_PITCH_COUNTER_NARROWBAND20_MS,
                SUBFRAME_PITCH_CONTOUR_NARROWBAND20_MS,
            ),
            Bandwidth::Medium | Bandwidth::Wide => (
                SUBFRAME_PITCH_COUNTER_MEDIUMBAND_OR_WIDEBAND20_MS,
                SUBFRAME_PITCH_CONTOUR_MEDIUMBAND_OR_WIDEBAND20_MS,
            ),
            _ => return Err(DecodePitchLagsError::UnsupportedBandwidth),
        };

        let contour_index = self.range_decoder.decode_symbol_with_icdf(lag_icdf) as usize;

        let mut pitch_lags = [0i16; SUBFRAME_COUNT];
        for (idx, value) in pitch_lags.iter_mut().enumerate() {
            let offset = lag_cb[contour_index][idx] as i32;
            let candidate = lag as i32 + offset;
            *value = candidate.clamp(lag_min as i32, lag_max as i32) as i16;
        }

        Ok(Some(PitchLagInfo {
            lag_max,
            pitch_lags,
        }))
    }

    #[allow(dead_code)]
    fn convert_normalized_lsfs_to_lpc_coefficients(
        &self,
        n1_q15: &NlsfQ15,
        bandwidth: Bandwidth,
    ) -> A32Q17 {
        let mut c_q17 = [0i32; MAX_D_LPC];
        let ordering = if bandwidth == Bandwidth::Wide {
            LSF_ORDERING_FOR_POLYNOMIAL_EVALUATION_WIDEBAND
        } else {
            LSF_ORDERING_FOR_POLYNOMIAL_EVALUATION_NARROWBAND_AND_MEDIUMBAND
        };

        for (k, &value) in n1_q15.as_slice().iter().enumerate() {
            let i = (value >> 8) as usize;
            let f = (value & 255) as i32;
            let cos_val = Q12_COSINE_TABLE_FOR_LSFCONVERION[i];
            let cos_next = Q12_COSINE_TABLE_FOR_LSFCONVERION[i + 1];

            c_q17[ordering[k] as usize] = (cos_val * 256 + (cos_next - cos_val) * f + 4) >> 3;
        }

        let d_lpc = n1_q15.len();
        let d2 = d_lpc / 2;
        let mut p_q16 = [0i32; MAX_D2_LPC];
        let mut q_q16 = [0i32; MAX_D2_LPC];

        p_q16[0] = 1 << 16;
        q_q16[0] = 1 << 16;
        p_q16[1] = -c_q17[0];
        q_q16[1] = -c_q17[1];

        for k in 1..d2 {
            let coeff_even = c_q17[2 * k] as i64;
            let coeff_odd = c_q17[2 * k + 1] as i64;

            p_q16[k + 1] =
                p_q16[k - 1] * 2 - ((coeff_even * i64::from(p_q16[k]) + 32_768) >> 16) as i32;
            q_q16[k + 1] =
                q_q16[k - 1] * 2 - ((coeff_odd * i64::from(q_q16[k]) + 32_768) >> 16) as i32;

            for j in (2..=k).rev() {
                p_q16[j] +=
                    p_q16[j - 2] - ((coeff_even * i64::from(p_q16[j - 1]) + 32_768) >> 16) as i32;
                q_q16[j] +=
                    q_q16[j - 2] - ((coeff_odd * i64::from(q_q16[j - 1]) + 32_768) >> 16) as i32;
            }

            p_q16[1] -= c_q17[2 * k];
            q_q16[1] -= c_q17[2 * k + 1];
        }

        let mut result = A32Q17::new(d_lpc);
        let result_slice = result.as_mut_slice();
        for k in 0..d2 {
            let diff_q = q_q16[k + 1] - q_q16[k];
            let sum_p = p_q16[k + 1] + p_q16[k];
            result_slice[k] = -diff_q - sum_p;
            result_slice[d_lpc - k - 1] = diff_q - sum_p;
        }

        result
    }

    #[allow(dead_code)]
    fn limit_lpc_coefficients_range(&self, a32_q17: &mut A32Q17) {
        let len = a32_q17.len();
        let mut bandwidth_expansion_round = 0;

        while bandwidth_expansion_round < 10 {
            let mut maxabs_q17 = 0u32;
            let mut maxabs_index = 0usize;

            for (idx, &value) in a32_q17.as_slice().iter().enumerate() {
                let abs_value = value.unsigned_abs();
                if abs_value > maxabs_q17 {
                    maxabs_q17 = abs_value;
                    maxabs_index = idx;
                }
            }

            let maxabs_q12 = ((maxabs_q17 + 16) >> 5).min(163_838);

            if maxabs_q12 > 32_767 {
                let mut sc_q16 = [0u32; MAX_D_LPC];
                let numerator = (maxabs_q12 - 32_767) << 14;
                let denom = ((maxabs_q12 * ((maxabs_index + 1) as u32)) >> 2).max(1);

                sc_q16[0] = 65_470 - numerator / denom;

                for k in 0..len {
                    let scaled = (i64::from(a32_q17.as_slice()[k]) * i64::from(sc_q16[k])) >> 16;
                    a32_q17.as_mut_slice()[k] = scaled as i32;

                    if k + 1 < len {
                        sc_q16[k + 1] = (sc_q16[0] * sc_q16[k] + 32_768) >> 16;
                    }
                }
            } else {
                break;
            }

            bandwidth_expansion_round += 1;
        }

        if bandwidth_expansion_round == 9 {
            for value in a32_q17.as_mut_slice().iter_mut() {
                let q12 = ((*value + 16) >> 5).clamp(-32_768, 32_767);
                *value = q12 << 5;
            }
        }
    }

    #[allow(dead_code)]
    fn limit_lpc_filter_prediction_gain(&self, a32_q17: &A32Q17) -> Aq12Coefficients {
        let mut coeffs = Aq12Coefficients::new(a32_q17.len());
        for (dst, &src) in coeffs
            .as_mut_slice()
            .iter_mut()
            .zip(a32_q17.as_slice().iter())
        {
            *dst = ((src + 16) >> 5) as f32;
        }

        coeffs
    }
}

/// The normalized LSF stabilization procedure ensures that
/// consecutive values of the normalized LSF coefficients, NLSF_Q15[],
/// are spaced some minimum distance apart (predetermined to be the 0.01
/// percentile of a large training set).
///
/// see [section-4.2.7.5](https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.4)
fn normalize_lsf_stabilization(nlsf_q15: &mut [i16], d_lpc: isize, bandwidth: Bandwidth) {
    // Let NDeltaMin_Q15[k] be the minimum required spacing for the current
    // audio bandwidth from Table 25.
    //
    // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.4
    let ndelta_min_q15 = if bandwidth == Bandwidth::Wide {
        // codebookMinimumSpacingForNormalizedLSCoefficientsWideband
        MINIMUM_SPACING_FOR_NORMALIZED_LSCOEFFICIENTS_WIDEBAND
    } else {
        MINIMUM_SPACING_FOR_NORMALIZED_LSCOEFFICIENTS_NARROWBAND_AND_MEDIUMBAND
    };

    // The procedure starts off by trying to make small adjustments that
    // attempt to minimize the amount of distortion introduced.  After 20
    // such adjustments, it falls back to a more direct method that
    // guarantees the constraints are enforced but may require large
    // adjustments.
    //
    // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.4
    for _adjustment in 0..=19 {
        // First, the procedure finds the index
        // i where NLSF_Q15[i] - NLSF_Q15[i-1] - NDeltaMin_Q15[i] is the
        // smallest, breaking ties by using the lower value of i.
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.4
        let mut i: isize = 0;
        let mut i_value = isize::MAX;

        for nlsf_index in 0..=(nlsf_q15.len()) {
            // For the purposes of computing this spacing for the first and last coefficient,
            // NLSF_Q15[-1] is taken to be 0 and NLSF_Q15[d_LPC] is taken to be 32768
            //
            // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.4
            let previous_nlsf = if nlsf_index != 0 {
                nlsf_q15[nlsf_index - 1] as isize
            } else {
                0
            };
            let current_nlsf = if nlsf_index != nlsf_q15.len() {
                nlsf_q15[nlsf_index] as isize
            } else {
                32768
            };

            let spacing_value: isize =
                current_nlsf - previous_nlsf - (ndelta_min_q15[nlsf_index] as isize);
            if spacing_value < i_value {
                i = nlsf_index as isize;
                i_value = spacing_value;
            }
        }

        // If this value is non-negative, then the stabilization stops; the coefficients
        // satisfy all the constraints.
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.4
        if i_value >= 0 {
            return;
        }
        // if i == 0, it sets NLSF_Q15[0] to NDeltaMin_Q15[0]
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.4
        if i == 0 {
            nlsf_q15[0] = (ndelta_min_q15[0]) as i16;

            continue;
        }
        // if i == d_LPC, it sets
        //  NLSF_Q15[d_LPC-1] to (32768 - NDeltaMin_Q15[d_LPC])
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.4
        if i == d_lpc {
            nlsf_q15[d_lpc as usize - 1] = (32768 - ndelta_min_q15[d_lpc as usize]) as i16;

            continue;
        }

        // 	For all other values of i, both NLSF_Q15[i-1] and NLSF_Q15[i] are updated as
        // follows:
        //                                              i-1
        //                                              __
        //     min_center_Q15 = (NDeltaMin_Q15[i]>>1) + \  NDeltaMin_Q15[k]
        //                                              /_
        //                                              k=0
        //
        let mut min_center_q15 = ndelta_min_q15[i as usize] >> 1;
        for k in 0..=(i - 1) {
            min_center_q15 += ndelta_min_q15[k as usize];
        }

        // 		                                                d_LPC
        //                                                      __
        //     max_center_Q15 = 32768 - (NDeltaMin_Q15[i]>>1) - \  NDeltaMin_Q15[k]
        //                                                      /_
        //                                                     k=i+1
        let mut max_center_q15 = 32768 - (ndelta_min_q15[i as usize] >> 1);
        for k in (i + 1)..=d_lpc {
            max_center_q15 -= ndelta_min_q15[k as usize];
        }

        //     center_freq_Q15 = clamp(min_center_Q15[i],
        //                     (NLSF_Q15[i-1] + NLSF_Q15[i] + 1)>>1
        //                     max_center_Q15[i])
        let center_freq_q15 =
            ((((nlsf_q15[i as usize - 1] as isize) + (nlsf_q15[i as usize] as isize) + 1) >> 1)
                as i32)
                .clamp(min_center_q15 as i32, max_center_q15 as i32) as isize;

        //    NLSF_Q15[i-1] = center_freq_Q15 - (NDeltaMin_Q15[i]>>1)
        //    NLSF_Q15[i] = NLSF_Q15[i-1] + NDeltaMin_Q15[i]
        nlsf_q15[i as usize - 1] =
            (center_freq_q15 - (ndelta_min_q15[i as usize] >> 1) as isize) as i16;
        nlsf_q15[i as usize] = nlsf_q15[i as usize - 1] + (ndelta_min_q15[i as usize] as i16);
    }

    // After the 20th repetition of the above procedure, the following
    // fallback procedure executes once.  First, the values of NLSF_Q15[k]
    // for 0 <= k < d_LPC are sorted in ascending order.  Then, for each
    // value of k from 0 to d_LPC-1, NLSF_Q15[k] is set to
    // sort.Slice(nlsfQ15, func(i, j int) bool {
    // 	return nlsfQ15[i] < nlsfQ15[j]
    // })
    // The slice length is bounded by the LPC order, so insertion sort is fine here.
    for i in 1..nlsf_q15.len() {
        let mut j = i;
        let current = nlsf_q15[i];
        while j > 0 && nlsf_q15[j - 1] > current {
            nlsf_q15[j] = nlsf_q15[j - 1];
            j -= 1;
        }
        nlsf_q15[j] = current;
    }

    // Then, for each value of k from 0 to d_LPC-1, NLSF_Q15[k] is set to
    //
    //   max(NLSF_Q15[k], NLSF_Q15[k-1] + NDeltaMin_Q15[k])
    for k in 0..=(d_lpc as usize - 1) {
        let prev_nlsf = if k != 0 { nlsf_q15[k - 1] } else { 0 };

        nlsf_q15[k] = nlsf_q15[k].max(prev_nlsf + (ndelta_min_q15[k] as i16));
    }

    // Next, for each value of k from d_LPC-1 down to 0, NLSF_Q15[k] is set
    // to
    //
    //   min(NLSF_Q15[k], NLSF_Q15[k+1] - NDeltaMin_Q15[k+1])
    for k in (0..=(d_lpc as usize - 1)).rev() {
        let next_nlsf = if k != (d_lpc as usize) - 1 {
            nlsf_q15[k + 1] as isize
        } else {
            32768
        };

        nlsf_q15[k] = nlsf_q15[k].min((next_nlsf - (ndelta_min_q15[k + 1] as isize)) as i16);
    }
}

/// Once the stage-1 index I1 and the stage-2 residual res_Q10[] have
/// been decoded, the final normalized LSF coefficients can be
/// reconstructed.
///
/// see [section-4.2.7.5.3](https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.3)
fn normalize_line_spectral_frequency_coefficients(
    d_lpc: usize,
    nlsf_q15: &mut [i16],
    bandwidth: Bandwidth,
    res_q10: &[i16],
    i1: u32,
) {
    let mut w2_q18 = [0usize; MAX_D_LPC];
    let mut w_q9 = [0i16; MAX_D_LPC];

    let cb1_q8 = if bandwidth == Bandwidth::Wide {
        NORMALIZED_LSF_STAGE_ONE_WIDEBAND
    } else {
        NORMALIZED_LSF_STAGE_ONE_NARROWBAND_OR_MEDIUMBAND
    };

    // Let cb1_Q8[k] be the k'th entry of the stage-1 codebook vector from Table 23 or Table 24.
    // Then, for 0 <= k < d_LPC, the following expression computes the
    // square of the weight as a Q18 value:
    //
    //          w2_Q18[k] = (1024/(cb1_Q8[k] - cb1_Q8[k-1])
    //                       + 1024/(cb1_Q8[k+1] - cb1_Q8[k])) << 16
    //
    // where cb1_Q8[-1] = 0 and cb1_Q8[d_LPC] = 256, and the division is
    // integer division.  This is reduced to an unsquared, Q9 value using
    // the following square-root approximation:
    //
    // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.3
    for k in 0..d_lpc {
        let mut k_minus_one = 0usize;
        let mut k_plus_one = 256usize;
        if k != 0 {
            k_minus_one = cb1_q8[i1 as usize][k - 1] as usize;
        }

        if k + 1 != d_lpc {
            k_plus_one = cb1_q8[i1 as usize][k + 1] as usize;
        }

        w2_q18[k] = (1024 / (cb1_q8[i1 as usize][k] as usize - k_minus_one)
            + 1024 / (k_plus_one - cb1_q8[i1 as usize][k] as usize))
            << 16;

        // This is reduced to an unsquared, Q9 value using
        // the following square-root approximation:
        //
        //     i = ilog(w2_Q18[k])
        //     f = (w2_Q18[k]>>(i-8)) & 127
        //     y = ((i&1) ? 32768 : 46214) >> ((32-i)>>1)
        //     w_Q9[k] = y + ((213*f*y)>>16)
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.3
        let i = ilog((w2_q18[k]) as isize);
        let f = ((w2_q18[k] >> (i - 8)) & 127) as isize;

        let mut y = 46214;
        if (i & 1) != 0 {
            y = 32768;
        }

        y >>= (32 - i) >> 1;
        w_q9[k] = (y + ((213 * f * y) >> 16)) as i16;

        // Given the stage-1 codebook entry cb1_Q8[], the stage-2 residual
        // res_Q10[], and their corresponding weights, w_Q9[], the reconstructed
        // normalized LSF coefficients are
        //
        //    NLSF_Q15[k] = clamp(0,
        //               (cb1_Q8[k]<<7) + (res_Q10[k]<<14)/w_Q9[k], 32767)
        //
        // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.5.3
        let cb1_val = (cb1_q8[i1 as usize][k] as i32) << 7;
        let res_val = (res_q10[k] as i32) << 14;
        let w_val = w_q9[k] as i32;
        let result = cb1_val + res_val / w_val;

        nlsf_q15[k] = result.clamp(0, 32767) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SILK_FRAME: &[u8] = &[0x0B, 0xE4, 0xC1, 0x36, 0xEC, 0xC5, 0x80];
    const TEST_RES_Q_10: [i16; 16] = [138, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const TEST_NLSF_Q_15: [i16; 16] = [
        2132, 3584, 5504, 7424, 9472, 11392, 13440, 15360, 17280, 19200, 21120, 23040, 25088,
        27008, 28928, 30848,
    ];
    const TEST_CONVERT_NLSF_Q15: [i16; 16] = [
        0x0854, 0x0E00, 0x1580, 0x1D00, 0x2500, 0x2C80, 0x3480, 0x3C00, 0x4380, 0x4B00, 0x5280,
        0x5A00, 0x6200, 0x6980, 0x7100, 0x7880,
    ];
    const EXPECTED_A32_Q17: [i32; 16] = [
        12_974, 9_765, 4_176, 3_646, -3_766, -4_429, -2_292, -4_663, -3_441, -3_848, -4_493,
        -1_614, -1_960, -3_112, -2_153, -2_898,
    ];
    const EXPECTED_AQ12: [f32; 16] = [
        405.0, 305.0, 131.0, 114.0, -118.0, -138.0, -72.0, -146.0, -108.0, -120.0, -140.0, -50.0,
        -61.0, -97.0, -67.0, -91.0,
    ];
    const TEST_PITCH_LAG_FRAME: &[u8] = &[
        0xb4, 0xe2, 0x2c, 0x0e, 0x10, 0x65, 0x1d, 0xa9, 0x07, 0x5c, 0x36, 0x8f, 0x96, 0x7b, 0xf4,
        0x89, 0x41, 0x55, 0x98, 0x7a, 0x39, 0x2e, 0x6b, 0x71, 0xa4, 0x03, 0x70, 0xbf,
    ];

    #[test]
    fn determine_frame_type() {
        let mut decoder = Decoder {
            range_decoder: RangeDecoder {
                buf: TEST_SILK_FRAME,
                bits_read: 31,
                range_size: 536870912,
                high_and_coded_difference: 437100388,
            },
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let (signal_type, quantization_offset_type) = decoder.determine_frame_type(false);
        assert_eq!(signal_type, FrameSignalType::Inactive);
        assert_eq!(quantization_offset_type, FrameQuantizationOffsetType::High);
    }

    #[test]
    fn decode_subframe_quantizations() {
        let mut decoder = Decoder {
            range_decoder: RangeDecoder {
                buf: TEST_SILK_FRAME,
                bits_read: 31,
                range_size: 482344960,
                high_and_coded_difference: 437100388,
            },
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let quantizations = decoder.decode_subframe_quantizations(FrameSignalType::Inactive);
        assert_eq!(quantizations, [210944., 112640., 96256., 96256.]);
    }

    #[test]
    fn normalize_line_spectral_frequency_stage_one() {
        let mut decoder = Decoder {
            range_decoder: RangeDecoder {
                buf: TEST_SILK_FRAME,
                bits_read: 47,
                range_size: 722810880,
                high_and_coded_difference: 387065757,
            },
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        assert_eq!(
            9,
            decoder.normalize_line_spectral_frequency_stage_one(false, Bandwidth::Wide)
        );
    }

    #[test]
    fn test_normalize_lsf_stabilization() {
        let mut input = [
            856, 2310, 3452, 4865, 4852, 7547, 9662, 11512, 13884, 15919, 18467, 20487, 23559,
            25900, 28222, 30700,
        ];

        let expected_out = [
            856, 2310, 3452, 4858, 4861, 7547, 9662, 11512, 13884, 15919, 18467, 20487, 23559,
            25900, 28222, 30700,
        ];

        normalize_lsf_stabilization(&mut input, 16, Bandwidth::Wide);
        assert_eq!(&input, &expected_out);

        let mut input2 = [
            1533, 1674, 2506, 4374, 6630, 9867, 10260, 10691, 14397, 16969, 19355, 21645, 25228,
            26972, 30514, 30208,
        ];

        let expected_out2 = [
            1533, 1674, 2506, 4374, 6630, 9867, 10260, 10691, 14397, 16969, 19355, 21645, 25228,
            26972, 30360, 30363,
        ];

        normalize_lsf_stabilization(&mut input2, 16, Bandwidth::Wide);
        assert_eq!(&input2, &expected_out2);
    }

    #[test]
    fn normalize_line_spectral_frequency_stage_two() {
        let mut decoder = Decoder {
            range_decoder: RangeDecoder {
                buf: TEST_SILK_FRAME,
                bits_read: 47,
                range_size: 50822640,
                high_and_coded_difference: 5895957,
            },
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let res_q10 = decoder.normalize_line_spectral_frequency_stage_two(Bandwidth::Wide, 9);

        assert_eq!(res_q10, ResQ10::Wide(TEST_RES_Q_10));
    }

    #[test]
    fn decode_pitch_lags() {
        let mut decoder = Decoder {
            range_decoder: RangeDecoder {
                buf: TEST_PITCH_LAG_FRAME,
                bits_read: 73,
                range_size: 30_770_362,
                high_and_coded_difference: 1_380_489,
            },
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let result = decoder
            .decode_pitch_lags(FrameSignalType::Voiced, Bandwidth::Wide)
            .expect("pitch lag decoding should succeed");

        let info = result.expect("expected voiced frame pitch lags");
        assert_eq!(info.lag_max, 288);
        assert_eq!(info.pitch_lags, [206, 206, 206, 206]);
    }

    #[test]
    fn decode_pitch_lags_returns_none_for_unvoiced_frames() {
        let mut decoder = Decoder {
            range_decoder: RangeDecoder::init(&[]),
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let result = decoder
            .decode_pitch_lags(FrameSignalType::Unvoiced, Bandwidth::Wide)
            .expect("pitch lag decoding should not fail for unvoiced frames");
        assert!(result.is_none());
    }

    #[test]
    fn normalize_lsf_interpolation_wq2_equals_four() {
        let mut decoder = Decoder {
            range_decoder: RangeDecoder {
                buf: TEST_SILK_FRAME,
                bits_read: 55,
                range_size: 493_249_168,
                high_and_coded_difference: 174_371_199,
            },
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let (n1_q15, w_q2) = decoder.normalize_lsf_interpolation(&[]);
        assert!(n1_q15.is_none());
        assert_eq!(w_q2, 4);
    }

    #[test]
    fn normalize_lsf_interpolation_wq2_equals_one() {
        let frame: &[u8] = &[
            0xac, 0xbd, 0xa9, 0xf7, 0x26, 0x24, 0x5a, 0xa4, 0x00, 0x37, 0xbf, 0x9c, 0xde, 0x0e,
            0xcf, 0x94, 0x64, 0xaa, 0xf9, 0x87, 0xd0, 0x79, 0x19, 0xa8, 0x21, 0xc0,
        ];
        let mut decoder = Decoder {
            range_decoder: RangeDecoder {
                buf: frame,
                bits_read: 65,
                range_size: 1_231_761_776,
                high_and_coded_difference: 1_068_195_183,
            },
            have_decoded: true,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 16,
        };
        decoder.n0_q15[..16].copy_from_slice(&[
            518, 380, 4444, 6982, 8752, 10510, 12381, 14102, 15892, 17651, 19340, 21888, 23936,
            25984, 28160, 30208,
        ]);

        let n2_q15 = [
            215, 1447, 3712, 5120, 7168, 9088, 11264, 13184, 15232, 17536, 19712, 21888, 24192,
            26240, 28416, 30336,
        ];
        let expected = NlsfQ15::from_slice(&[
            442, 646, 4261, 6516, 8356, 10154, 12101, 13872, 15727, 17622, 19433, 21888, 24000,
            26048, 28224, 30240,
        ]);

        let (actual, w_q2) = decoder.normalize_lsf_interpolation(&n2_q15);
        assert_eq!(w_q2, 1);
        assert_eq!(actual, Some(expected));
    }

    #[test]
    fn test_normalize_line_spectral_frequency_coefficients() {
        let mut input_nlsf_q15 = [0i16; 16];
        normalize_line_spectral_frequency_coefficients(
            16,
            &mut input_nlsf_q15,
            Bandwidth::Wide,
            &TEST_RES_Q_10,
            9,
        );
        assert_eq!(&input_nlsf_q15, &TEST_NLSF_Q_15);
    }

    #[test]
    fn convert_normalized_lsfs_to_lpc_coefficients_matches_reference() {
        let decoder = Decoder {
            range_decoder: RangeDecoder::init(&[]),
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let nlsf = NlsfQ15::from_slice(&TEST_CONVERT_NLSF_Q15);
        let actual = decoder.convert_normalized_lsfs_to_lpc_coefficients(&nlsf, Bandwidth::Wide);

        assert_eq!(actual.as_slice(), &EXPECTED_A32_Q17);
    }

    #[test]
    fn limit_lpc_coefficients_range_preserves_reference_values() {
        let decoder = Decoder {
            range_decoder: RangeDecoder::init(&[]),
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let mut a32 = A32Q17::new(EXPECTED_A32_Q17.len());
        a32.as_mut_slice().copy_from_slice(&EXPECTED_A32_Q17);

        decoder.limit_lpc_coefficients_range(&mut a32);

        assert_eq!(a32.as_slice(), &EXPECTED_A32_Q17);
    }

    #[test]
    fn limit_lpc_filter_prediction_gain_matches_reference() {
        let decoder = Decoder {
            range_decoder: RangeDecoder::init(&[]),
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let mut a32 = A32Q17::new(EXPECTED_A32_Q17.len());
        a32.as_mut_slice().copy_from_slice(&EXPECTED_A32_Q17);

        let a_q12 = decoder.limit_lpc_filter_prediction_gain(&a32);

        assert_eq!(a_q12.as_slice(), &EXPECTED_AQ12);
    }

    #[test]
    fn generate_a_q12_appends_coefficients_when_available() {
        let mut decoder = Decoder {
            range_decoder: RangeDecoder::init(&[]),
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
        };

        let nlsf = NlsfQ15::from_slice(&TEST_CONVERT_NLSF_Q15);
        let mut a_q12 = Aq12List::new();

        decoder.generate_a_q12(None, Bandwidth::Wide, &mut a_q12);
        assert!(a_q12.is_empty());

        decoder.generate_a_q12(Some(&nlsf), Bandwidth::Wide, &mut a_q12);
        assert_eq!(a_q12.len(), 1);
        assert_eq!(a_q12.get(0), &EXPECTED_AQ12);
    }
}
