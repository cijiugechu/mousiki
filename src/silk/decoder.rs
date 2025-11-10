mod nomarlize;

use super::icdf::{FRAME_TYPE_VAD_ACTIVE, FRAME_TYPE_VAD_INACTIVE};
use super::{FrameQuantizationOffsetType, FrameSignalType};
use crate::math::{ilog, sign};
use crate::packet::Bandwidth;
use crate::range::RangeDecoder;
use crate::silk::code_signs;
use crate::silk::decode_pitch::silk_decode_pitch;
use crate::silk::codebook::{
    CODEBOOK_LTP_FILTER_PERIODICITY_INDEX_0, CODEBOOK_LTP_FILTER_PERIODICITY_INDEX_1,
    CODEBOOK_LTP_FILTER_PERIODICITY_INDEX_2,
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
};
use crate::silk::icdf::{
    self, DELTA_QUANTIZATION_GAIN, INDEPENDENT_QUANTIZATION_GAIN_LSB,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE, INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED, LINEAR_CONGRUENTIAL_GENERATOR_SEED,
    LTP_FILTER_INDEX0, LTP_FILTER_INDEX1, LTP_FILTER_INDEX2, LTP_SCALING_PARAMETER,
    NORMALIZED_LSF_INTERPOLATION_INDEX,
    NORMALIZED_LSF_STAGE_1_INDEX_NARROWBAND_OR_MEDIUMBAND_UNVOICED,
    NORMALIZED_LSF_STAGE_1_INDEX_NARROWBAND_OR_MEDIUMBAND_VOICED,
    NORMALIZED_LSF_STAGE_1_INDEX_WIDEBAND_UNVOICED, NORMALIZED_LSF_STAGE_1_INDEX_WIDEBAND_VOICED,
    NORMALIZED_LSF_STAGE_2_INDEX, NORMALIZED_LSF_STAGE_2_INDEX_EXTENSION, PERIODICITY_INDEX,
    PRIMARY_PITCH_LAG_HIGH_PART, PRIMARY_PITCH_LAG_LOW_PART_MEDIUMBAND,
    PRIMARY_PITCH_LAG_LOW_PART_NARROWBAND, PRIMARY_PITCH_LAG_LOW_PART_WIDEBAND,
    SUBFRAME_PITCH_CONTOUR_MEDIUMBAND_OR_WIDEBAND20_MS, SUBFRAME_PITCH_CONTOUR_NARROWBAND20_MS,
};
use core::convert::TryFrom;
use core::fmt;
use log::{debug, trace};
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

    pub fn build(self) -> Decoder {
        Decoder {
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: self.final_out_values,
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
            previous_frame_lpc_values: [0.0; MAX_D_LPC],
            previous_frame_lpc_values_len: 0,
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
pub struct Decoder {
    // Have we decoded a frame yet?
    have_decoded: bool,
    is_previous_frame_voiced: bool,
    previous_log_gain: i32,
    final_out_values: [f32; 306],
    n0_q15: [i16; MAX_D_LPC],
    n0_q15_len: usize,
    previous_frame_lpc_values: [f32; MAX_D_LPC],
    previous_frame_lpc_values_len: usize,
}

const SUBFRAME_COUNT: usize = 4;
const MAX_SHELL_BLOCKS: usize = 20;
const PULSECOUNT_LARGEST_PARTITION_SIZE: usize = 16;
const MAX_EXCITATION_SAMPLES: usize = MAX_SHELL_BLOCKS * PULSECOUNT_LARGEST_PARTITION_SIZE;
const MAX_LSB_COUNT: u8 = 10;
const MAX_SUBFRAME_SAMPLES: usize = SUBFRAME_COUNT * 80;
const MAX_PITCH_LAG: usize = 288;
const LTP_FILTER_TAP_COUNT: usize = 5;
const MAX_RES_LAG: usize = MAX_PITCH_LAG + 2;
const INV_Q23: f32 = 1.0 / 8_388_608.0;
const NANOSECONDS_20_MS: u32 = 20_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PitchLagInfo {
    pub lag_max: u32,
    pub pitch_lags: [i16; SUBFRAME_COUNT],
}

#[derive(Debug, Clone)]
pub struct ShellBlockCounts {
    pub block_count: usize,
    pub pulse_counts: [u8; MAX_SHELL_BLOCKS],
    pub lsb_counts: [u8; MAX_SHELL_BLOCKS],
}

impl ShellBlockCounts {
    pub fn new(block_count: usize) -> Self {
        debug_assert!(block_count <= MAX_SHELL_BLOCKS);
        Self {
            block_count,
            pulse_counts: [0; MAX_SHELL_BLOCKS],
            lsb_counts: [0; MAX_SHELL_BLOCKS],
        }
    }

    /// Returns the per-block pulse counts encoded with their LSB extension flags,
    /// matching the layout expected by the SILK sign decoder.
    pub fn sign_sums(&self) -> [i32; MAX_SHELL_BLOCKS] {
        let mut sums = [0i32; MAX_SHELL_BLOCKS];
        for (block_idx, sum) in sums.iter_mut().enumerate().take(self.block_count) {
            let base = i32::from(self.pulse_counts[block_idx]);
            let lsb = i32::from(self.lsb_counts[block_idx]);
            *sum = base | (lsb << 5);
        }
        sums
    }
}

#[derive(Debug, Clone)]
pub struct ExcitationQ23 {
    pub len: usize,
    pub values: [i32; MAX_EXCITATION_SAMPLES],
}

impl ExcitationQ23 {
    pub fn new(len: usize) -> Self {
        debug_assert!(len <= MAX_EXCITATION_SAMPLES);
        Self {
            len,
            values: [0; MAX_EXCITATION_SAMPLES],
        }
    }
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

impl core::error::Error for DecodePitchLagsError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    UnsupportedFrameDuration,
    StereoUnsupported,
    OutBufferTooSmall,
    UnsupportedLowBitrateRedundancy,
    PitchLags(DecodePitchLagsError),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFrameDuration => f.write_str("only 20 ms SILK frames are supported"),
            Self::StereoUnsupported => f.write_str("stereo SILK decoding is unsupported"),
            Self::OutBufferTooSmall => f.write_str("output buffer is too small for decoded frame"),
            Self::UnsupportedLowBitrateRedundancy => {
                f.write_str("low bit-rate redundancy is unsupported")
            }
            Self::PitchLags(err) => write!(f, "pitch lag decoding failed: {err}"),
        }
    }
}

impl core::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::PitchLags(err) => Some(err),
            _ => None,
        }
    }
}

impl Decoder {
    fn decode_header_bits(&mut self, range_decoder: &mut RangeDecoder<'_>) -> (bool, bool) {
        let voice_activity_detected = range_decoder.decode_symbol_logp(1) == 1;
        let low_bit_rate_redundancy = range_decoder.decode_symbol_logp(1) == 1;
        (voice_activity_detected, low_bit_rate_redundancy)
    }

    /// Each SILK frame contains a single "frame type" symbol that jointly
    /// codes the signal type and quantization offset type of the
    /// corresponding frame.
    ///
    /// See [section-4.2.7.3](https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.3)
    pub fn determine_frame_type(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        voice_activity_detected: bool,
    ) -> (FrameSignalType, FrameQuantizationOffsetType) {
        let frame_type_symbol = if voice_activity_detected {
            range_decoder.decode_symbol_with_icdf(FRAME_TYPE_VAD_ACTIVE)
        } else {
            range_decoder.decode_symbol_with_icdf(FRAME_TYPE_VAD_INACTIVE)
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
        range_decoder: &mut RangeDecoder<'_>,
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
                    FrameSignalType::Inactive => range_decoder
                        .decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE)
                        as i32,
                    FrameSignalType::Voiced => range_decoder
                        .decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED)
                        as i32,
                    FrameSignalType::Unvoiced => range_decoder
                        .decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED)
                        as i32,
                };

                // The 3 least significant bits are decoded using a uniform PDF:
                // These 6 bits are combined to form a value, gain_index, between 0 and 63.
                gain_index = (gain_index << 3)
                    | (range_decoder.decode_symbol_with_icdf(INDEPENDENT_QUANTIZATION_GAIN_LSB)
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
                delta_gain_index =
                    range_decoder.decode_symbol_with_icdf(DELTA_QUANTIZATION_GAIN) as i32;

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
        range_decoder: &mut RangeDecoder<'_>,
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
            (false, Narrow | Medium) => range_decoder.decode_symbol_with_icdf(
                NORMALIZED_LSF_STAGE_1_INDEX_NARROWBAND_OR_MEDIUMBAND_UNVOICED,
            ),
            (true, Narrow | Medium) => range_decoder.decode_symbol_with_icdf(
                NORMALIZED_LSF_STAGE_1_INDEX_NARROWBAND_OR_MEDIUMBAND_VOICED,
            ),
            (false, Wide) => range_decoder
                .decode_symbol_with_icdf(NORMALIZED_LSF_STAGE_1_INDEX_WIDEBAND_UNVOICED),
            (true, Wide) => {
                range_decoder.decode_symbol_with_icdf(NORMALIZED_LSF_STAGE_1_INDEX_WIDEBAND_VOICED)
            }
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
        range_decoder: &mut RangeDecoder<'_>,
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
            i2[i] = (range_decoder.decode_symbol_with_icdf(
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
                i2[i] -= (range_decoder
                    .decode_symbol_with_icdf(NORMALIZED_LSF_STAGE_2_INDEX_EXTENSION))
                    as i8;
            } else if i2[i] == 4 {
                i2[i] += (range_decoder
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
    pub fn normalize_lsf_interpolation(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        n2_q15: &[i16],
    ) -> (Option<NlsfQ15>, i16) {
        let w_q2 = range_decoder.decode_symbol_with_icdf(NORMALIZED_LSF_INTERPOLATION_INDEX) as i16;

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
        range_decoder: &mut RangeDecoder<'_>,
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

            let lag_high = range_decoder.decode_symbol_with_icdf(PRIMARY_PITCH_LAG_HIGH_PART);
            let lag_low = range_decoder.decode_symbol_with_icdf(low_part_icdf);

            (lag_high * lag_scale + lag_low + lag_min, lag_min, lag_max)
        } else {
            return Err(DecodePitchLagsError::NonAbsoluteLagsUnsupported);
        };

        let lag_icdf = match bandwidth {
            Bandwidth::Narrow => SUBFRAME_PITCH_CONTOUR_NARROWBAND20_MS,
            Bandwidth::Medium | Bandwidth::Wide => {
                SUBFRAME_PITCH_CONTOUR_MEDIUMBAND_OR_WIDEBAND20_MS
            }
            _ => return Err(DecodePitchLagsError::UnsupportedBandwidth),
        };

        let contour_index = range_decoder.decode_symbol_with_icdf(lag_icdf) as i8;

        let mut pitch_lags = [0i16; SUBFRAME_COUNT];
        let lag_index = (lag - lag_min) as i16;
        let fs_khz = match bandwidth {
            Bandwidth::Narrow => 8,
            Bandwidth::Medium => 12,
            Bandwidth::Wide => 16,
            _ => return Err(DecodePitchLagsError::UnsupportedBandwidth),
        };
        silk_decode_pitch(
            lag_index,
            contour_index,
            &mut pitch_lags,
            fs_khz,
            SUBFRAME_COUNT,
        );

        Ok(Some(PitchLagInfo {
            lag_max,
            pitch_lags,
        }))
    }

    /// See [section-4.2.7.6.2](https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.6.2)
    pub fn decode_ltp_filter_coefficients(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        signal_type: FrameSignalType,
    ) -> Option<[[i8; 5]; SUBFRAME_COUNT]> {
        if signal_type != FrameSignalType::Voiced {
            return None;
        }

        let periodicity_index = range_decoder.decode_symbol_with_icdf(PERIODICITY_INDEX) as usize;
        debug_assert!(periodicity_index < 3);
        if periodicity_index > 2 {
            return None;
        }

        let filter_icdf = match periodicity_index {
            0 => LTP_FILTER_INDEX0,
            1 => LTP_FILTER_INDEX1,
            _ => LTP_FILTER_INDEX2,
        };

        let codebook: &'static [[i8; 5]] = match periodicity_index {
            0 => &CODEBOOK_LTP_FILTER_PERIODICITY_INDEX_0,
            1 => &CODEBOOK_LTP_FILTER_PERIODICITY_INDEX_1,
            _ => &CODEBOOK_LTP_FILTER_PERIODICITY_INDEX_2,
        };

        let mut coefficients = [[0i8; 5]; SUBFRAME_COUNT];
        for coeff in coefficients.iter_mut() {
            let filter_index = range_decoder.decode_symbol_with_icdf(filter_icdf) as usize;
            *coeff = codebook[filter_index];
        }

        Some(coefficients)
    }

    /// See [section-4.2.7.6.3](https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.6.3)
    pub fn decode_ltp_scaling_parameter(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        signal_type: FrameSignalType,
    ) -> f32 {
        const SCALE_FACTORS_Q14: [f32; 3] = [15_565.0, 12_288.0, 8_192.0];

        if signal_type != FrameSignalType::Voiced {
            return SCALE_FACTORS_Q14[0];
        }

        let index = range_decoder.decode_symbol_with_icdf(LTP_SCALING_PARAMETER) as usize;

        SCALE_FACTORS_Q14.get(index).copied().unwrap_or(0.0)
    }

    /// See [section-4.2.7.7](https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.7)
    pub fn decode_linear_congruential_generator_seed(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
    ) -> u32 {
        range_decoder.decode_symbol_with_icdf(LINEAR_CONGRUENTIAL_GENERATOR_SEED)
    }

    fn samples_in_subframe(&self, bandwidth: Bandwidth) -> usize {
        bandwidth.samples_in_subframe() as usize
    }

    #[allow(clippy::too_many_arguments)]
    fn ltp_synthesis(
        &mut self,
        out: &mut [f32],
        b_q7: &[[i8; LTP_FILTER_TAP_COUNT]; SUBFRAME_COUNT],
        pitch_lags: &[i16; SUBFRAME_COUNT],
        n: usize,
        j: usize,
        subframe_index: usize,
        d_lpc: usize,
        mut ltp_scale_q14: f32,
        w_q2: i16,
        a_q12: &[f32],
        gain_q16: &[f32; SUBFRAME_COUNT],
        res: &mut [f32],
        res_lag: &mut [f32],
    ) {
        let n_isize = n as isize;
        let out_end = if subframe_index < 2 || w_q2 == 4 {
            -(subframe_index as isize) * n_isize
        } else {
            ltp_scale_q14 = 16_384.0;
            -((subframe_index as isize) - 2) * n_isize
        };

        let pitch = pitch_lags[subframe_index] as isize;
        let res_len = res.len() as isize;
        let res_lag_len = res_lag.len() as isize;

        let mut i = -pitch - 2;
        while i < out_end {
            let index = i + j as isize;

            if index >= res_len {
                i += 1;
                continue;
            }

            let mut write_to_lag = false;
            let (mut res_val, res_index) = if index >= 0 {
                let index_usize = index as usize;
                if index_usize >= out.len() {
                    i += 1;
                    continue;
                }
                (out[index_usize], index_usize)
            } else {
                let lag_index = res_lag_len + index;
                if lag_index < 0 || lag_index >= res_lag_len {
                    i += 1;
                    continue;
                }
                let final_len = self.final_out_values.len() as isize;
                let final_index = final_len + index;
                let value = if final_index < 0 || final_index >= final_len {
                    0.0
                } else {
                    self.final_out_values[final_index as usize]
                };
                write_to_lag = true;
                (value, lag_index as usize)
            };

            for (k, &coeff) in a_q12.iter().take(d_lpc).enumerate() {
                let out_index = index - (k as isize) - 1;
                let out_value = if out_index >= 0 {
                    out[out_index as usize]
                } else {
                    let final_index = self.final_out_values.len() as isize + out_index;
                    if final_index < 0 || final_index >= self.final_out_values.len() as isize {
                        0.0
                    } else {
                        self.final_out_values[final_index as usize]
                    }
                };

                res_val -= out_value * (coeff / 4096.0);
            }

            res_val = clamp_negative_one_to_one(res_val);
            let gain = gain_q16[subframe_index];
            if gain != 0.0 {
                res_val *= (4.0 * ltp_scale_q14) / gain;

                if !write_to_lag {
                    res[res_index] = res_val;
                } else if res_index < res_lag.len() {
                    res_lag[res_index] = res_val;
                }
            }

            i += 1;
        }

        if subframe_index > 0 {
            let current_gain = gain_q16[subframe_index];
            if current_gain != 0.0 {
                let scaled_gain = gain_q16[subframe_index - 1] / current_gain;
                let mut idx = out_end;
                while idx < 0 {
                    let res_index = j as isize + idx;
                    if res_index < 0 {
                        let lag_index = res_lag_len + res_index;
                        if lag_index >= 0 && lag_index < res_lag_len {
                            res_lag[lag_index as usize] *= scaled_gain;
                        }
                    } else if res_index < res_len {
                        res[res_index as usize] *= scaled_gain;
                    }

                    idx += 1;
                }
            }
        }

        for sample_index in j..(j + n) {
            let mut res_sum = res[sample_index];

            for tap in 0..LTP_FILTER_TAP_COUNT {
                let res_index = sample_index as isize - pitch + 2 - tap as isize;
                let value = if res_index < 0 {
                    let lag_index = res_lag_len + res_index;
                    if lag_index >= 0 && lag_index < res_lag_len {
                        res_lag[lag_index as usize]
                    } else {
                        0.0
                    }
                } else if res_index < res_len {
                    res[res_index as usize]
                } else {
                    0.0
                };

                res_sum += value * (f32::from(b_q7[subframe_index][tap]) / 128.0);
            }

            res[sample_index] = res_sum;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lpc_synthesis(
        &mut self,
        out: &mut [f32],
        n: usize,
        subframe_index: usize,
        d_lpc: usize,
        a_q12: &[f32],
        res: &[f32],
        gain_q16: &[f32; SUBFRAME_COUNT],
        lpc: &mut [f32],
        frame_samples: usize,
    ) {
        let j = n * subframe_index;
        let gain_factor = gain_q16[subframe_index] / 65_536.0;

        for i in 0..n {
            let sample_index = j + i;
            if sample_index >= res.len() || sample_index >= frame_samples {
                break;
            }

            let mut lpc_val = gain_factor * res[sample_index];

            for (k, &coeff) in a_q12.iter().take(d_lpc).enumerate() {
                let lpc_index = sample_index as isize - k as isize - 1;
                let current = if lpc_index >= 0 {
                    lpc[lpc_index as usize]
                } else if subframe_index == 0 && i < self.previous_frame_lpc_values_len {
                    let prev_len = self.previous_frame_lpc_values_len as isize;
                    let idx = prev_len - 1 + i as isize - k as isize;
                    if idx >= 0 && idx < prev_len {
                        self.previous_frame_lpc_values[idx as usize]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                lpc_val += current * (coeff / 4096.0);
            }

            lpc[sample_index] = lpc_val;
            if sample_index < out.len() {
                out[sample_index] = clamp_negative_one_to_one(lpc_val);
            }

            if subframe_index == SUBFRAME_COUNT - 1
                && self.have_decoded
                && sample_index == out.len().saturating_sub(1)
                && d_lpc <= frame_samples
            {
                let start = frame_samples - d_lpc;
                self.previous_frame_lpc_values[..d_lpc].copy_from_slice(&lpc[start..frame_samples]);
                self.previous_frame_lpc_values_len = d_lpc;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn silk_frame_reconstruction(
        &mut self,
        signal_type: FrameSignalType,
        bandwidth: Bandwidth,
        d_lpc: usize,
        lag_max: u32,
        b_q7: Option<&[[i8; LTP_FILTER_TAP_COUNT]; SUBFRAME_COUNT]>,
        pitch_lags: Option<&[i16; SUBFRAME_COUNT]>,
        excitation: &ExcitationQ23,
        ltp_scale_q14: f32,
        w_q2: i16,
        a_q12: &Aq12List,
        gain_q16: &[f32; SUBFRAME_COUNT],
        out: &mut [f32],
    ) {
        let n = self.samples_in_subframe(bandwidth);
        let frame_samples = n * SUBFRAME_COUNT;

        let mut lpc = [0.0f32; MAX_SUBFRAME_SAMPLES];
        let mut res = [0.0f32; MAX_EXCITATION_SAMPLES];
        let mut res_lag = [0.0f32; MAX_RES_LAG];

        let res_len = excitation.len.min(res.len());
        for (dst, &value) in res.iter_mut().zip(excitation.values.iter()).take(res_len) {
            *dst = (value as f32) * INV_Q23;
        }

        let mut res_lag_len = lag_max as usize + 2;
        if res_lag_len > res_lag.len() {
            res_lag_len = res_lag.len();
        }

        for subframe_index in 0..SUBFRAME_COUNT {
            let aq_index = if subframe_index > 1 && a_q12.len() > 1 {
                1
            } else {
                0
            };

            let aq_slice: &[f32] = if a_q12.is_empty() {
                &[]
            } else {
                a_q12.get(aq_index.min(a_q12.len() - 1))
            };

            let j = n * subframe_index;

            if signal_type == FrameSignalType::Voiced
                && let (Some(b_q7_values), Some(pitch_values)) = (b_q7, pitch_lags)
            {
                self.ltp_synthesis(
                    out,
                    b_q7_values,
                    pitch_values,
                    n,
                    j,
                    subframe_index,
                    d_lpc,
                    ltp_scale_q14,
                    w_q2,
                    aq_slice,
                    gain_q16,
                    &mut res[..res_len],
                    &mut res_lag[..res_lag_len],
                );
            }

            self.lpc_synthesis(
                out,
                n,
                subframe_index,
                d_lpc,
                aq_slice,
                &res[..res_len],
                gain_q16,
                &mut lpc[..frame_samples],
                frame_samples,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &mut self,
        input: &[u8],
        out: &mut [f32],
        is_stereo: bool,
        nanoseconds: u32,
        bandwidth: Bandwidth,
    ) -> Result<(), DecodeError> {
        debug!(
            "silk::Decoder::decode: payload={} stereo={} ns={} bandwidth={:?}",
            input.len(),
            is_stereo,
            nanoseconds,
            bandwidth
        );
        let subframe_size = self.samples_in_subframe(bandwidth);
        if nanoseconds != NANOSECONDS_20_MS {
            debug!(
                "silk::Decoder::decode: unsupported frame duration {}ns",
                nanoseconds
            );
            return Err(DecodeError::UnsupportedFrameDuration);
        }
        if is_stereo {
            debug!("silk::Decoder::decode: stereo frames are not supported");
            return Err(DecodeError::StereoUnsupported);
        }

        let total_samples = subframe_size * SUBFRAME_COUNT;
        if total_samples > out.len() {
            debug!(
                "silk::Decoder::decode: output buffer too small (needed {}, available {})",
                total_samples,
                out.len()
            );
            return Err(DecodeError::OutBufferTooSmall);
        }

        let mut range_decoder = RangeDecoder::init(input);
        trace!(
            "silk::Decoder::decode: range decoder initialized (subframe_size={})",
            subframe_size
        );

        let (voice_activity_detected, low_bit_rate_redundancy) =
            self.decode_header_bits(&mut range_decoder);
        trace!(
            "silk::Decoder::decode: header bits vad={} lbr={}",
            voice_activity_detected, low_bit_rate_redundancy
        );
        if low_bit_rate_redundancy {
            debug!("silk::Decoder::decode: low bitrate redundancy not supported");
            return Err(DecodeError::UnsupportedLowBitrateRedundancy);
        }

        let (signal_type, quantization_offset_type) =
            self.determine_frame_type(&mut range_decoder, voice_activity_detected);
        trace!(
            "silk::Decoder::decode: frame type signal={:?} q_offset={:?}",
            signal_type, quantization_offset_type
        );

        let gain_q16 = self.decode_subframe_quantizations(&mut range_decoder, signal_type);
        trace!("silk::Decoder::decode: subframe gains {:?}", &gain_q16);

        let i1 = self.normalize_line_spectral_frequency_stage_one(
            &mut range_decoder,
            signal_type == FrameSignalType::Voiced,
            bandwidth,
        );
        trace!(
            "silk::Decoder::decode: normalized lsf stage one index {}",
            i1
        );

        let res_q10 =
            self.normalize_line_spectral_frequency_stage_two(&mut range_decoder, bandwidth, i1);
        let d_lpc = res_q10.d_lpc();
        trace!("silk::Decoder::decode: stage two residual length {}", d_lpc);
        let mut nlsf_q15 = [0i16; MAX_D_LPC];
        normalize_line_spectral_frequency_coefficients(
            d_lpc,
            &mut nlsf_q15[..d_lpc],
            bandwidth,
            &res_q10,
            i1,
        );
        normalize_lsf_stabilization(&mut nlsf_q15[..d_lpc], d_lpc as isize, bandwidth);

        let (n1_q15, w_q2) =
            self.normalize_lsf_interpolation(&mut range_decoder, &nlsf_q15[..d_lpc]);
        let reuse_previous = n1_q15.is_some();
        trace!(
            "silk::Decoder::decode: interpolation weight {} reuse_previous={}",
            w_q2, reuse_previous
        );

        let mut a_q12 = Aq12List::new();
        if let Some(ref n1_values) = n1_q15 {
            self.generate_a_q12(Some(n1_values), bandwidth, &mut a_q12);
        }
        let nlsf_current = NlsfQ15::from_slice(&nlsf_q15[..d_lpc]);
        self.generate_a_q12(Some(&nlsf_current), bandwidth, &mut a_q12);

        let pitch_info = self
            .decode_pitch_lags(&mut range_decoder, signal_type, bandwidth)
            .map_err(DecodeError::PitchLags)?;
        let has_pitch_info = pitch_info.is_some();
        let lag_max = pitch_info.as_ref().map(|info| info.lag_max).unwrap_or(0);
        let pitch_lags_ref: Option<&[i16; SUBFRAME_COUNT]> =
            pitch_info.as_ref().map(|info| &info.pitch_lags);
        trace!(
            "silk::Decoder::decode: pitch info present={} lag_max={}",
            has_pitch_info, lag_max
        );

        let ltp_coefficients = self.decode_ltp_filter_coefficients(&mut range_decoder, signal_type);
        let ltp_scale_q14 = self.decode_ltp_scaling_parameter(&mut range_decoder, signal_type);
        let lcg_seed = self.decode_linear_congruential_generator_seed(&mut range_decoder);
        trace!(
            "silk::Decoder::decode: ltp present={} scale_q14={} seed={}",
            ltp_coefficients.is_some(),
            ltp_scale_q14,
            lcg_seed
        );
        let shell_blocks = self.decode_shell_blocks(nanoseconds, bandwidth);
        trace!(
            "silk::Decoder::decode: shell blocks {} (bandwidth {:?})",
            shell_blocks, bandwidth
        );
        let rate_level =
            self.decode_rate_level(&mut range_decoder, signal_type == FrameSignalType::Voiced);
        trace!(
            "silk::Decoder::decode: rate level {} (voiced={})",
            rate_level,
            signal_type == FrameSignalType::Voiced
        );
        let counts = self.decode_pulse_and_lsb_counts(&mut range_decoder, shell_blocks, rate_level);
        trace!(
            "silk::Decoder::decode: pulse counts blocks={} first_block={} first_lsb={}",
            counts.block_count,
            counts.pulse_counts.first().copied().unwrap_or(0),
            counts.lsb_counts.first().copied().unwrap_or(0)
        );
        let excitation = self.decode_excitation(
            &mut range_decoder,
            signal_type,
            quantization_offset_type,
            lcg_seed,
            &counts,
        );
        trace!(
            "silk::Decoder::decode: excitation length {}",
            excitation.len
        );

        self.silk_frame_reconstruction(
            signal_type,
            bandwidth,
            d_lpc,
            lag_max,
            ltp_coefficients.as_ref(),
            pitch_lags_ref,
            &excitation,
            ltp_scale_q14,
            w_q2,
            &a_q12,
            &gain_q16,
            out,
        );

        self.is_previous_frame_voiced = signal_type == FrameSignalType::Voiced;

        self.n0_q15[..d_lpc].copy_from_slice(&nlsf_q15[..d_lpc]);
        self.n0_q15_len = d_lpc;

        if out.len() >= self.final_out_values.len() {
            let start = out.len() - self.final_out_values.len();
            self.final_out_values
                .copy_from_slice(&out[start..out.len()]);
        } else {
            self.final_out_values[..out.len()].copy_from_slice(out);
        }

        self.have_decoded = true;
        trace!(
            "silk::Decoder::decode: frame complete (out_len={} samples)",
            out.len()
        );

        Ok(())
    }

    /// https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.8
    pub fn decode_shell_blocks(&self, nanoseconds: u32, bandwidth: Bandwidth) -> usize {
        match (bandwidth, nanoseconds) {
            (Bandwidth::Narrow, 10_000_000) => 5,
            (Bandwidth::Medium, 10_000_000) => 8,
            (Bandwidth::Wide, 10_000_000) | (Bandwidth::Narrow, 20_000_000) => 10,
            (Bandwidth::Medium, 20_000_000) => 15,
            (Bandwidth::Wide, 20_000_000) => 20,
            _ => 0,
        }
    }

    /// https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.8.1
    pub fn decode_rate_level(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        voice_activity_detected: bool,
    ) -> u32 {
        if voice_activity_detected {
            range_decoder.decode_symbol_with_icdf(icdf::RATE_LEVEL_VOICED)
        } else {
            range_decoder.decode_symbol_with_icdf(icdf::RATE_LEVEL_UNVOICED)
        }
    }

    /// https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.8.2
    pub fn decode_pulse_and_lsb_counts(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        shell_blocks: usize,
        rate_level: u32,
    ) -> ShellBlockCounts {
        debug_assert!(shell_blocks <= MAX_SHELL_BLOCKS);
        let mut counts = ShellBlockCounts::new(shell_blocks);

        let rate_index = rate_level as usize;
        debug_assert!(rate_index < icdf::PULSE_COUNT.len());
        let rate_index = rate_index.min(icdf::PULSE_COUNT.len() - 1);

        for block_idx in 0..shell_blocks {
            let mut pulse_count =
                range_decoder.decode_symbol_with_icdf(icdf::PULSE_COUNT[rate_index]) as u8;

            if pulse_count == 17 {
                let mut lsb_count = 0u8;
                while pulse_count == 17 && lsb_count < MAX_LSB_COUNT {
                    pulse_count = range_decoder.decode_symbol_with_icdf(icdf::PULSE_COUNT[9]) as u8;
                    lsb_count += 1;
                }
                counts.lsb_counts[block_idx] = lsb_count;

                if lsb_count == MAX_LSB_COUNT {
                    pulse_count =
                        range_decoder.decode_symbol_with_icdf(icdf::PULSE_COUNT[10]) as u8;
                }
            }

            counts.pulse_counts[block_idx] = pulse_count;
        }

        counts
    }

    /// https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.8
    pub fn decode_excitation(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        signal_type: FrameSignalType,
        quantization_offset_type: FrameQuantizationOffsetType,
        mut seed: u32,
        counts: &ShellBlockCounts,
    ) -> ExcitationQ23 {
        let len = counts.block_count * PULSECOUNT_LARGEST_PARTITION_SIZE;

        let offset_q23 = match (signal_type, quantization_offset_type) {
            (FrameSignalType::Inactive, FrameQuantizationOffsetType::Low)
            | (FrameSignalType::Unvoiced, FrameQuantizationOffsetType::Low) => 25,
            (FrameSignalType::Inactive, FrameQuantizationOffsetType::High)
            | (FrameSignalType::Unvoiced, FrameQuantizationOffsetType::High) => 60,
            (FrameSignalType::Voiced, FrameQuantizationOffsetType::Low) => 8,
            (FrameSignalType::Voiced, FrameQuantizationOffsetType::High) => 25,
        };

        let mut e_raw = [0i32; MAX_EXCITATION_SAMPLES];
        self.decode_pulse_location_into(range_decoder, counts, &mut e_raw, len);
        self.decode_excitation_lsb_into(range_decoder, &mut e_raw, counts, len);
        self.decode_excitation_sign_into(
            range_decoder,
            &mut e_raw,
            signal_type,
            quantization_offset_type,
            counts,
            len,
        );

        let mut excitation = ExcitationQ23::new(len);
        for (raw, slot) in e_raw.iter().take(len).zip(excitation.values.iter_mut()) {
            let mut value = (raw << 8) - sign(*raw) * 20 + offset_q23;
            seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
            if seed & 0x8000_0000 != 0 {
                value = -value;
            }
            seed = seed.wrapping_add(*raw as u32);
            *slot = value;
        }

        excitation
    }

    fn decode_pulse_location_into(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        counts: &ShellBlockCounts,
        e_raw: &mut [i32; MAX_EXCITATION_SAMPLES],
        len: usize,
    ) {
        for sample in e_raw.iter_mut().take(len) {
            *sample = 0;
        }

        for block_idx in 0..counts.block_count {
            let pulses = counts.pulse_counts[block_idx];
            if pulses == 0 {
                continue;
            }

            let base_index = block_idx * PULSECOUNT_LARGEST_PARTITION_SIZE;
            let mut e_index = base_index;

            let mut partition16 = [0u8; 2];
            self.partition_pulse_count(
                range_decoder,
                &icdf::PULSE_COUNT_SPLIT16_SAMPLE_PARTITIONS,
                pulses,
                &mut partition16,
            );

            for &count8 in &partition16 {
                let mut partition8 = [0u8; 2];
                self.partition_pulse_count(
                    range_decoder,
                    &icdf::PULSE_COUNT_SPLIT8_SAMPLE_PARTITIONS,
                    count8,
                    &mut partition8,
                );

                for &count4 in &partition8 {
                    let mut partition4 = [0u8; 2];
                    self.partition_pulse_count(
                        range_decoder,
                        &icdf::PULSE_COUNT_SPLIT4_SAMPLE_PARTITIONS,
                        count4,
                        &mut partition4,
                    );

                    for &count2 in &partition4 {
                        let mut partition2 = [0u8; 2];
                        self.partition_pulse_count(
                            range_decoder,
                            &icdf::PULSE_COUNT_SPLIT2_SAMPLE_PARTITIONS,
                            count2,
                            &mut partition2,
                        );

                        e_raw[e_index] = i32::from(partition2[0]);
                        e_index += 1;
                        e_raw[e_index] = i32::from(partition2[1]);
                        e_index += 1;
                    }
                }
            }
        }
    }

    fn decode_excitation_lsb_into(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        e_raw: &mut [i32; MAX_EXCITATION_SAMPLES],
        counts: &ShellBlockCounts,
        len: usize,
    ) {
        for (sample_idx, sample) in e_raw.iter_mut().take(len).enumerate() {
            let block_idx = sample_idx / PULSECOUNT_LARGEST_PARTITION_SIZE;
            let lsb_count = counts.lsb_counts[block_idx];
            for _ in 0..lsb_count {
                let bit = range_decoder.decode_symbol_with_icdf(icdf::EXCITATION_LSB);
                *sample = (*sample << 1) | bit as i32;
            }
        }
    }

    fn decode_excitation_sign_into(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        e_raw: &mut [i32; MAX_EXCITATION_SAMPLES],
        signal_type: FrameSignalType,
        quantization_offset_type: FrameQuantizationOffsetType,
        counts: &ShellBlockCounts,
        len: usize,
    ) {
        if len == 0 {
            return;
        }

        debug_assert_eq!(
            len,
            counts.block_count * PULSECOUNT_LARGEST_PARTITION_SIZE,
            "excitation length must match shell block allocation"
        );

        let signal_type_index = match signal_type {
            FrameSignalType::Inactive => 0,
            FrameSignalType::Unvoiced => 1,
            FrameSignalType::Voiced => 2,
        };
        let quant_offset_index = match quantization_offset_type {
            FrameQuantizationOffsetType::Low => 0,
            FrameQuantizationOffsetType::High => 1,
        };

        let mut pulses = [0i16; MAX_EXCITATION_SAMPLES];
        for (dst, &value) in pulses.iter_mut().take(len).zip(e_raw.iter().take(len)) {
            *dst = i16::try_from(value).expect("shell pulse magnitude exceeds i16 range");
        }

        let sign_sums = counts.sign_sums();
        code_signs::silk_decode_signs(
            range_decoder,
            &mut pulses[..len],
            len,
            signal_type_index,
            quant_offset_index,
            &sign_sums[..counts.block_count],
        );

        for (src, dst) in pulses.iter().take(len).zip(e_raw.iter_mut()) {
            *dst = i32::from(*src);
        }
    }

    fn partition_pulse_count(
        &mut self,
        range_decoder: &mut RangeDecoder<'_>,
        contexts: &[icdf::ICDFContext; 16],
        block: u8,
        halves: &mut [u8; 2],
    ) {
        if block == 0 {
            halves[0] = 0;
            halves[1] = 0;
            return;
        }

        let index = (block as usize).saturating_sub(1).min(contexts.len() - 1);
        let left = range_decoder.decode_symbol_with_icdf(contexts[index]) as u8;
        halves[0] = left;
        halves[1] = block.saturating_sub(left);
    }

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
            let f = i32::from(value & 255);
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
            let coeff_even = i64::from(c_q17[2 * k]);
            let coeff_odd = i64::from(c_q17[2 * k + 1]);

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
            let current_nlsf = if nlsf_index == nlsf_q15.len() {
                32768
            } else {
                nlsf_q15[nlsf_index] as isize
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
                .clamp(i32::from(min_center_q15), i32::from(max_center_q15)) as isize;

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
        let next_nlsf = if k == (d_lpc as usize) - 1 {
            32768
        } else {
            nlsf_q15[k + 1] as isize
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
        let cb1_val = i32::from(cb1_q8[i1 as usize][k]) << 7;
        let res_val = i32::from(res_q10[k]) << 14;
        let w_val = i32::from(w_q9[k]);
        let result = cb1_val + res_val / w_val;

        nlsf_q15[k] = result.clamp(0, 32767) as i16;
    }
}

fn clamp_negative_one_to_one(value: f32) -> f32 {
    value.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SILK_FRAME: &[u8] = &[0x0B, 0xE4, 0xC1, 0x36, 0xEC, 0xC5, 0x80];
    const TEST_SILK_FRAME_SECOND: &[u8] = &[0x07, 0xC9, 0x72, 0x27, 0xE1, 0x44, 0xEA, 0x50];
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
    const TEST_LCG_FRAME: &[u8] = &[
        0x84, 0x2e, 0x67, 0xd3, 0x85, 0x65, 0x54, 0xe3, 0x9d, 0x90, 0x0a, 0xfa, 0x98, 0xea, 0xfd,
        0x98, 0x94, 0x41, 0xf9, 0x6d, 0x1d, 0xa0,
    ];
    const EXPECTED_DECODE_OUT_0: [f32; 320] = [
        0.000023f32,
        0.000025f32,
        0.000027f32,
        -0.000018f32,
        0.000025f32,
        -0.000021f32,
        0.000021f32,
        -0.000024f32,
        0.000021f32,
        0.000021f32,
        -0.000022f32,
        -0.000026f32,
        0.000018f32,
        0.000022f32,
        -0.000023f32,
        -0.000025f32,
        -0.000027f32,
        0.000017f32,
        0.000020f32,
        -0.000021f32,
        0.000023f32,
        0.000027f32,
        -0.000018f32,
        -0.000023f32,
        -0.000024f32,
        0.000020f32,
        -0.000024f32,
        0.000021f32,
        0.000023f32,
        0.000027f32,
        0.000029f32,
        -0.000016f32,
        -0.000020f32,
        -0.000025f32,
        0.000018f32,
        -0.000026f32,
        -0.000028f32,
        -0.000028f32,
        -0.000028f32,
        0.000016f32,
        -0.000025f32,
        -0.000025f32,
        0.000021f32,
        0.000025f32,
        0.000027f32,
        -0.000016f32,
        0.000030f32,
        -0.000016f32,
        -0.000020f32,
        -0.000024f32,
        -0.000026f32,
        0.000019f32,
        0.000022f32,
        0.000025f32,
        -0.000019f32,
        -0.000021f32,
        -0.000024f32,
        -0.000027f32,
        -0.000029f32,
        -0.000030f32,
        0.000017f32,
        0.000022f32,
        0.000026f32,
        0.000030f32,
        0.000033f32,
        -0.000012f32,
        -0.000018f32,
        -0.000023f32,
        -0.000026f32,
        -0.000029f32,
        -0.000029f32,
        0.000016f32,
        -0.000025f32,
        0.000021f32,
        0.000024f32,
        0.000028f32,
        -0.000017f32,
        0.000027f32,
        0.000028f32,
        0.000029f32,
        -0.000006f32,
        0.000017f32,
        0.000015f32,
        0.000015f32,
        -0.000011f32,
        0.000011f32,
        0.000011f32,
        -0.000014f32,
        0.000008f32,
        -0.000016f32,
        0.000008f32,
        -0.000016f32,
        -0.000016f32,
        -0.000018f32,
        -0.000017f32,
        -0.000017f32,
        0.000008f32,
        -0.000014f32,
        -0.000013f32,
        -0.000013f32,
        -0.000012f32,
        0.000011f32,
        -0.000010f32,
        0.000015f32,
        0.000016f32,
        -0.000006f32,
        0.000015f32,
        -0.000008f32,
        -0.000009f32,
        -0.000012f32,
        0.000012f32,
        0.000012f32,
        0.000013f32,
        -0.000009f32,
        -0.000011f32,
        0.000011f32,
        0.000012f32,
        -0.000012f32,
        0.000012f32,
        0.000013f32,
        0.000014f32,
        -0.000011f32,
        0.000013f32,
        -0.000011f32,
        -0.000013f32,
        -0.000016f32,
        0.000008f32,
        -0.000015f32,
        0.000010f32,
        -0.000013f32,
        -0.000013f32,
        -0.000015f32,
        0.000010f32,
        -0.000013f32,
        0.000011f32,
        -0.000011f32,
        -0.000011f32,
        -0.000013f32,
        0.000012f32,
        -0.000011f32,
        0.000013f32,
        0.000015f32,
        0.000016f32,
        0.000016f32,
        0.000017f32,
        -0.000007f32,
        -0.000010f32,
        -0.000013f32,
        -0.000015f32,
        -0.000017f32,
        0.000007f32,
        -0.000015f32,
        -0.000015f32,
        0.000009f32,
        0.000012f32,
        -0.000011f32,
        0.000012f32,
        -0.000010f32,
        0.000013f32,
        -0.000011f32,
        0.000012f32,
        0.000012f32,
        0.000014f32,
        0.000014f32,
        -0.000007f32,
        0.000012f32,
        -0.000010f32,
        0.000010f32,
        0.000010f32,
        0.000011f32,
        -0.000010f32,
        0.000009f32,
        -0.000011f32,
        0.000008f32,
        0.000009f32,
        -0.000010f32,
        -0.000013f32,
        -0.000013f32,
        -0.000014f32,
        0.000006f32,
        0.000009f32,
        -0.000010f32,
        -0.000011f32,
        -0.000011f32,
        -0.000012f32,
        0.000008f32,
        0.000011f32,
        0.000013f32,
        -0.000007f32,
        -0.000008f32,
        -0.000010f32,
        -0.000011f32,
        0.000009f32,
        -0.000010f32,
        -0.000011f32,
        0.000009f32,
        -0.000010f32,
        -0.000011f32,
        0.000010f32,
        0.000012f32,
        -0.000009f32,
        -0.000010f32,
        -0.000010f32,
        -0.000012f32,
        0.000009f32,
        0.000011f32,
        0.000012f32,
        0.000014f32,
        -0.000007f32,
        0.000012f32,
        -0.000009f32,
        0.000011f32,
        -0.000010f32,
        0.000010f32,
        -0.000011f32,
        -0.000012f32,
        -0.000013f32,
        -0.000013f32,
        -0.000014f32,
        0.000007f32,
        -0.000012f32,
        0.000009f32,
        -0.000010f32,
        -0.000010f32,
        -0.000011f32,
        0.000010f32,
        0.000012f32,
        0.000013f32,
        -0.000006f32,
        0.000013f32,
        -0.000007f32,
        -0.000009f32,
        0.000010f32,
        -0.000010f32,
        -0.000011f32,
        0.000008f32,
        -0.000010f32,
        -0.000012f32,
        -0.000012f32,
        0.000009f32,
        0.000009f32,
        0.000011f32,
        0.000013f32,
        0.000014f32,
        0.000015f32,
        0.000014f32,
        -0.000007f32,
        0.000012f32,
        0.000011f32,
        0.000012f32,
        -0.000010f32,
        -0.000012f32,
        0.000008f32,
        0.000008f32,
        0.000009f32,
        0.000009f32,
        -0.000010f32,
        -0.000012f32,
        -0.000014f32,
        -0.000014f32,
        0.000006f32,
        0.000008f32,
        -0.000010f32,
        -0.000012f32,
        0.000010f32,
        -0.000010f32,
        0.000010f32,
        0.000012f32,
        0.000013f32,
        -0.000008f32,
        -0.000009f32,
        -0.000010f32,
        0.000009f32,
        -0.000010f32,
        -0.000011f32,
        0.000008f32,
        -0.000011f32,
        -0.000012f32,
        -0.000012f32,
        -0.000012f32,
        -0.000013f32,
        0.000008f32,
        -0.000011f32,
        -0.000011f32,
        0.000010f32,
        0.000013f32,
        -0.000007f32,
        -0.000008f32,
        -0.000009f32,
        -0.000010f32,
        0.000009f32,
        0.000011f32,
        0.000013f32,
        -0.000007f32,
        0.000013f32,
        -0.000008f32,
        0.000011f32,
        -0.000010f32,
        0.000011f32,
        0.000011f32,
        0.000012f32,
        0.000012f32,
        0.000013f32,
        -0.000008f32,
        0.000010f32,
        -0.000011f32,
        0.000009f32,
        -0.000012f32,
        -0.000013f32,
        -0.000014f32,
        0.000006f32,
        -0.000013f32,
        -0.000013f32,
        0.000008f32,
        -0.000011f32,
        -0.000012f32,
        -0.000012f32,
        0.000010f32,
        0.000011f32,
        0.000013f32,
    ];
    const EXPECTED_DECODE_OUT_1: [f32; 320] = [
        0.000011f32,
        -0.000009f32,
        -0.000011f32,
        -0.000012f32,
        0.000009f32,
        0.000010f32,
        -0.000010f32,
        0.000011f32,
        0.000012f32,
        -0.000008f32,
        0.000011f32,
        -0.000009f32,
        -0.000010f32,
        -0.000012f32,
        0.000008f32,
        0.000009f32,
        -0.000010f32,
        0.000011f32,
        0.000012f32,
        0.000013f32,
        0.000012f32,
        0.000013f32,
        -0.000007f32,
        0.000011f32,
        0.000011f32,
        0.000011f32,
        0.000011f32,
        0.000012f32,
        -0.000009f32,
        0.000009f32,
        -0.000012f32,
        -0.000013f32,
        0.000006f32,
        0.000008f32,
        0.000009f32,
        0.000010f32,
        0.000012f32,
        0.000012f32,
        0.000012f32,
        -0.000009f32,
        -0.000011f32,
        -0.000013f32,
        0.000007f32,
        -0.000013f32,
        0.000008f32,
        0.000009f32,
        0.000011f32,
        -0.000009f32,
        -0.000011f32,
        0.000009f32,
        -0.000011f32,
        -0.000012f32,
        -0.000013f32,
        0.000008f32,
        0.000010f32,
        -0.000009f32,
        0.000011f32,
        -0.000008f32,
        -0.000010f32,
        0.000009f32,
        -0.000010f32,
        0.000010f32,
        0.000011f32,
        -0.000008f32,
        0.000011f32,
        -0.000009f32,
        -0.000010f32,
        0.000029f32,
        -0.000008f32,
        -0.000010f32,
        0.000009f32,
        0.000012f32,
        -0.000010f32,
        -0.000011f32,
        0.000010f32,
        0.000010f32,
        -0.000010f32,
        -0.000011f32,
        0.000009f32,
        0.000011f32,
        0.000011f32,
        0.000012f32,
        -0.000008f32,
        0.000011f32,
        -0.000009f32,
        -0.000011f32,
        0.000008f32,
        -0.000011f32,
        -0.000012f32,
        0.000007f32,
        -0.000011f32,
        -0.000012f32,
        -0.000013f32,
        0.000009f32,
        0.000009f32,
        0.000012f32,
        -0.000008f32,
        -0.000009f32,
        0.000011f32,
        -0.000009f32,
        -0.000010f32,
        -0.000011f32,
        -0.000012f32,
        -0.000013f32,
        0.000008f32,
        -0.000011f32,
        0.000010f32,
        -0.000009f32,
        -0.000009f32,
        -0.000012f32,
        0.000010f32,
        -0.000010f32,
        -0.000010f32,
        0.000011f32,
        0.000012f32,
        -0.000008f32,
        0.000012f32,
        -0.000007f32,
        0.000012f32,
        -0.000009f32,
        0.000011f32,
        0.000011f32,
        0.000012f32,
        -0.000008f32,
        0.000011f32,
        0.000012f32,
        0.000012f32,
        0.000012f32,
        0.000012f32,
        0.000012f32,
        0.000012f32,
        -0.000009f32,
        -0.000012f32,
        -0.000014f32,
        -0.000015f32,
        0.000005f32,
        0.000007f32,
        0.000009f32,
        -0.000011f32,
        -0.000011f32,
        0.000009f32,
        -0.000011f32,
        0.000009f32,
        -0.000010f32,
        -0.000010f32,
        -0.000012f32,
        -0.000012f32,
        0.000009f32,
        0.000011f32,
        -0.000008f32,
        0.000012f32,
        -0.000008f32,
        0.000012f32,
        -0.000009f32,
        -0.000009f32,
        -0.000011f32,
        0.000009f32,
        -0.000010f32,
        0.000009f32,
        0.000012f32,
        0.000013f32,
        -0.000008f32,
        -0.000009f32,
        -0.000011f32,
        0.000009f32,
        0.000010f32,
        0.000011f32,
        -0.000009f32,
        -0.000010f32,
        0.000010f32,
        0.000010f32,
        0.000011f32,
        -0.000009f32,
        -0.000010f32,
        0.000029f32,
        -0.000009f32,
        0.000010f32,
        -0.000010f32,
        -0.000010f32,
        0.000008f32,
        -0.000012f32,
        0.000009f32,
        0.000009f32,
        -0.000009f32,
        0.000010f32,
        -0.000010f32,
        0.000010f32,
        -0.000011f32,
        -0.000011f32,
        0.000009f32,
        -0.000011f32,
        0.000010f32,
        -0.000011f32,
        0.000011f32,
        0.000011f32,
        0.000012f32,
        -0.000008f32,
        0.000011f32,
        -0.000009f32,
        0.000010f32,
        0.000010f32,
        -0.000009f32,
        -0.000011f32,
        0.000009f32,
        -0.000011f32,
        0.000008f32,
        0.000010f32,
        -0.000009f32,
        -0.000012f32,
        0.000009f32,
        0.000010f32,
        0.000011f32,
        0.000013f32,
        0.000013f32,
        -0.000008f32,
        -0.000010f32,
        -0.000012f32,
        -0.000013f32,
        -0.000014f32,
        0.000006f32,
        0.000008f32,
        -0.000011f32,
        0.000010f32,
        0.000012f32,
        0.000013f32,
        -0.000008f32,
        0.000012f32,
        -0.000009f32,
        0.000010f32,
        0.000011f32,
        0.000012f32,
        0.000013f32,
        -0.000008f32,
        -0.000010f32,
        -0.000013f32,
        0.000007f32,
        0.000008f32,
        0.000010f32,
        -0.000010f32,
        0.000010f32,
        0.000010f32,
        0.000012f32,
        -0.000009f32,
        0.000011f32,
        -0.000010f32,
        -0.000012f32,
        0.000007f32,
        0.000010f32,
        0.000011f32,
        -0.000009f32,
        -0.000010f32,
        -0.000013f32,
        -0.000013f32,
        0.000007f32,
        0.000009f32,
        0.000011f32,
        -0.000009f32,
        0.000011f32,
        -0.000009f32,
        0.000011f32,
        0.000012f32,
        0.000013f32,
        -0.000008f32,
        -0.000010f32,
        0.000009f32,
        -0.000011f32,
        0.000029f32,
        -0.000009f32,
        -0.000010f32,
        -0.000013f32,
        0.000008f32,
        -0.000012f32,
        0.000008f32,
        -0.000011f32,
        0.000010f32,
        0.000010f32,
        0.000012f32,
        0.000013f32,
        -0.000007f32,
        -0.000009f32,
        -0.000012f32,
        -0.000013f32,
        0.000007f32,
        0.000009f32,
        -0.000010f32,
        -0.000011f32,
        0.000009f32,
        0.000011f32,
        -0.000010f32,
        -0.000010f32,
        -0.000011f32,
        -0.000012f32,
        0.000008f32,
        -0.000011f32,
        -0.000011f32,
        -0.000011f32,
        -0.000011f32,
        0.000009f32,
        -0.000010f32,
        0.000011f32,
        0.000013f32,
        -0.000007f32,
        -0.000009f32,
        0.000011f32,
        -0.000008f32,
        -0.000010f32,
        -0.000011f32,
        0.000010f32,
        -0.000011f32,
        -0.000011f32,
        -0.000012f32,
        0.000009f32,
        -0.000010f32,
        0.000010f32,
        -0.000009f32,
        -0.000010f32,
        -0.000011f32,
        0.000010f32,
        -0.000010f32,
        0.000011f32,
    ];
    const FLOAT_EQUALITY_THRESHOLD: f32 = 0.000001f32;

    #[test]
    fn determine_frame_type() {
        let mut decoder = Decoder {
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
            previous_frame_lpc_values: [0.0; MAX_D_LPC],
            previous_frame_lpc_values_len: 0,
        };
        let mut range_decoder = RangeDecoder {
            buf: TEST_SILK_FRAME,
            bits_read: 31,
            range_size: 536_870_912,
            high_and_coded_difference: 437_100_388,
        };

        let (signal_type, quantization_offset_type) =
            decoder.determine_frame_type(&mut range_decoder, false);
        assert_eq!(signal_type, FrameSignalType::Inactive);
        assert_eq!(quantization_offset_type, FrameQuantizationOffsetType::High);
    }

    #[test]
    fn decode_subframe_quantizations() {
        let mut decoder = Decoder {
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
            previous_frame_lpc_values: [0.0; MAX_D_LPC],
            previous_frame_lpc_values_len: 0,
        };
        let mut range_decoder = RangeDecoder {
            buf: TEST_SILK_FRAME,
            bits_read: 31,
            range_size: 482_344_960,
            high_and_coded_difference: 437_100_388,
        };

        let quantizations =
            decoder.decode_subframe_quantizations(&mut range_decoder, FrameSignalType::Inactive);
        assert_eq!(quantizations, [210944., 112640., 96256., 96256.]);
    }

    #[test]
    fn normalize_line_spectral_frequency_stage_one() {
        let mut decoder = Decoder {
            have_decoded: false,
            is_previous_frame_voiced: false,
            previous_log_gain: 0,
            final_out_values: [0.; 306],
            n0_q15: [0; MAX_D_LPC],
            n0_q15_len: 0,
            previous_frame_lpc_values: [0.0; MAX_D_LPC],
            previous_frame_lpc_values_len: 0,
        };
        let mut range_decoder = RangeDecoder {
            buf: TEST_SILK_FRAME,
            bits_read: 47,
            range_size: 722_810_880,
            high_and_coded_difference: 387_065_757,
        };

        assert_eq!(
            9,
            decoder.normalize_line_spectral_frequency_stage_one(
                &mut range_decoder,
                false,
                Bandwidth::Wide,
            )
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
        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder {
            buf: TEST_SILK_FRAME,
            bits_read: 47,
            range_size: 50_822_640,
            high_and_coded_difference: 5_895_957,
        };

        let res_q10 = decoder.normalize_line_spectral_frequency_stage_two(
            &mut range_decoder,
            Bandwidth::Wide,
            9,
        );

        assert_eq!(res_q10, ResQ10::Wide(TEST_RES_Q_10));
    }

    #[test]
    fn decode_pitch_lags() {
        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder {
            buf: TEST_PITCH_LAG_FRAME,
            bits_read: 73,
            range_size: 30_770_362,
            high_and_coded_difference: 1_380_489,
        };

        let result = decoder
            .decode_pitch_lags(&mut range_decoder, FrameSignalType::Voiced, Bandwidth::Wide)
            .expect("pitch lag decoding should succeed");

        let info = result.expect("expected voiced frame pitch lags");
        assert_eq!(info.lag_max, 288);
        assert_eq!(info.pitch_lags, [206, 206, 206, 206]);
    }

    #[test]
    fn decode_ltp_filter_coefficients_returns_expected_taps() {
        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder {
            buf: TEST_PITCH_LAG_FRAME,
            bits_read: 89,
            range_size: 253_853_952,
            high_and_coded_difference: 138_203_876,
        };

        let coeffs = decoder
            .decode_ltp_filter_coefficients(&mut range_decoder, FrameSignalType::Voiced)
            .expect("voiced frame should decode LTP filter coefficients");

        assert_eq!(
            coeffs,
            [
                [1, 1, 8, 1, 1],
                [2, 0, 77, 11, 9],
                [1, 1, 8, 1, 1],
                [-1, 36, 64, 27, -6],
            ]
        );
    }

    #[test]
    fn decode_ltp_scaling_parameter_returns_default_for_unvoiced_frames() {
        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder::init(&[]);

        let scale =
            decoder.decode_ltp_scaling_parameter(&mut range_decoder, FrameSignalType::Unvoiced);
        assert_eq!(scale, 15_565.0);
    }

    #[test]
    fn decode_ltp_scaling_parameter_decodes_voiced_frames() {
        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder {
            buf: TEST_PITCH_LAG_FRAME,
            bits_read: 105,
            range_size: 160_412_192,
            high_and_coded_difference: 164_623_240,
        };

        let scale =
            decoder.decode_ltp_scaling_parameter(&mut range_decoder, FrameSignalType::Voiced);
        assert_eq!(scale, 15_565.0);
    }

    #[test]
    fn decode_linear_congruential_generator_seed_reads_expected_value() {
        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder {
            buf: TEST_LCG_FRAME,
            bits_read: 71,
            range_size: 851_775_140,
            high_and_coded_difference: 846_837_397,
        };

        let seed = decoder.decode_linear_congruential_generator_seed(&mut range_decoder);
        assert_eq!(seed, 0);
    }

    #[test]
    fn decode_returns_error_for_non_20_ms_frames() {
        let mut decoder = DecoderBuilder::new().build();
        let mut out = [0.0f32; 320];
        let result = decoder.decode(TEST_SILK_FRAME, &mut out, false, 1, Bandwidth::Wide);
        assert!(matches!(result, Err(DecodeError::UnsupportedFrameDuration)));
    }

    #[test]
    fn decode_returns_error_for_stereo_frames() {
        let mut decoder = DecoderBuilder::new().build();
        let mut out = [0.0f32; 320];
        let result = decoder.decode(
            TEST_SILK_FRAME,
            &mut out,
            true,
            NANOSECONDS_20_MS,
            Bandwidth::Wide,
        );
        assert!(matches!(result, Err(DecodeError::StereoUnsupported)));
    }

    #[test]
    fn decode_returns_error_when_output_buffer_too_small() {
        let mut decoder = DecoderBuilder::new().build();
        let mut out = [0.0f32; 50];
        let result = decoder.decode(
            TEST_SILK_FRAME,
            &mut out,
            false,
            NANOSECONDS_20_MS,
            Bandwidth::Wide,
        );
        assert!(matches!(result, Err(DecodeError::OutBufferTooSmall)));
    }

    #[test]
    fn decode_matches_go_fixture_for_unvoiced_frame() {
        let mut decoder = DecoderBuilder::new().build();
        let mut out = [0.0f32; 320];

        decoder
            .decode(
                TEST_SILK_FRAME,
                &mut out,
                false,
                NANOSECONDS_20_MS,
                Bandwidth::Wide,
            )
            .expect("decode should succeed");

        for (actual, expected) in out.iter().zip(EXPECTED_DECODE_OUT_0.iter()) {
            assert!((actual - expected).abs() < FLOAT_EQUALITY_THRESHOLD);
        }
    }

    #[test]
    fn decode_matches_go_fixture_for_subsequent_unvoiced_frame() {
        let mut decoder = DecoderBuilder::new().build();
        let mut out = [0.0f32; 320];

        decoder
            .decode(
                TEST_SILK_FRAME,
                &mut out,
                false,
                NANOSECONDS_20_MS,
                Bandwidth::Wide,
            )
            .expect("initial decode should succeed");

        decoder
            .decode(
                TEST_SILK_FRAME_SECOND,
                &mut out,
                false,
                NANOSECONDS_20_MS,
                Bandwidth::Wide,
            )
            .expect("subsequent decode should succeed");

        for (actual, expected) in out.iter().zip(EXPECTED_DECODE_OUT_1.iter()) {
            assert!((actual - expected).abs() < FLOAT_EQUALITY_THRESHOLD);
        }
    }

    #[test]
    fn decode_shell_blocks_matches_reference() {
        let decoder = DecoderBuilder::new().build();

        assert_eq!(
            decoder.decode_shell_blocks(10_000_000, Bandwidth::Narrow),
            5
        );
        assert_eq!(
            decoder.decode_shell_blocks(10_000_000, Bandwidth::Medium),
            8
        );
        assert_eq!(decoder.decode_shell_blocks(10_000_000, Bandwidth::Wide), 10);
        assert_eq!(
            decoder.decode_shell_blocks(20_000_000, Bandwidth::Narrow),
            10
        );
        assert_eq!(
            decoder.decode_shell_blocks(20_000_000, Bandwidth::Medium),
            15
        );
        assert_eq!(decoder.decode_shell_blocks(20_000_000, Bandwidth::Wide), 20);
    }

    #[test]
    fn decode_excitation_matches_go_fixture() {
        const EXPECTED: &[i32] = &[
            25, -25, -25, -25, 25, 25, -25, 25, 25, -25, 25, -25, -25, -25, 25, 25, -25, 25, 25,
            25, 25, -211, -25, -25, 25, -25, 25, -25, 25, -25, -25, -25, 25, 25, -25, -25, 261,
            517, -25, 25, -25, -25, -25, -25, -25, -25, 25, -25, -25, 25, -25, 25, -25, 25, 25, 25,
            25, -25, 25, -25, 25, 25, 25, 25, -25, 25, 25, 25, 25, -25, -25, -25, -25, -25, -25,
            -25, 25, 25, -25, 25, 211, 25, -25, -25, 25, 211, 25, 25, 25, -25, 25, 25, -25, -25,
            -25, 25, 25, 25, 25, -25, 25, 25, -25, 25, 25, 25, 25, 25, -25, -25, 25, -25, -25, 25,
            25, -25, 25, 25, 25, -25, -25, -25, -25, -25, -25, 25, 25, 25, 25, 25, -25, 25, -25,
            -25, 25, 25, 25, 25, 25, 25, 25, -25, 25, -211, 25, -25, -25, 25, 25, -25, -25, -25,
            -25, -25, -25, -25, 25, 25, -25, -25, 25, 25, -25, 25, -25, -25, -25, 25, 25, -25, 25,
            -25, -211, -25, 25, 25, 25, -25, -25, -25, -25, 25, 25, -25, -25, 25, -25, -25, 25, 25,
            25, -25, -25, -25, -25, -25, 25, 25, -25, -211, 25, -25, 25, 25, -25, -25, 25, -25, 25,
            -25, 25, 25, -25, -211, -25, 25, 25, -25, 25, 25, -25, -211, -25, 25, 25, 25, -25, -25,
            -25, -25, 25, -211, 25, 25, 25, 25, 25, 25, -25, -25, 25, -25, 517, 517, -467, -25, 25,
            25, -25, -25, 25, -25, 25, 25, 25, -25, -25, -25, 25, 25, -25, -25, 25, -25, 25, -25,
            25, -25, 25, -25, -25, -25, 25, 25, -25, -25, 211, 25, 25, 25, 25, -25, -25, 25, -25,
            -25, -25, -25, 211, -25, 25, -25, -25, 25, -25, -25, 25, -25, 25, -25, 25, 25, -25, 25,
            -25, 25, 25, 25, 25, -25, -25, -25, 25, -25, 25, 25, -25, -25, -25, 25,
        ];

        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder {
            buf: TEST_LCG_FRAME,
            bits_read: 71,
            range_size: 851_775_140,
            high_and_coded_difference: 846_837_397,
        };

        let seed = decoder.decode_linear_congruential_generator_seed(&mut range_decoder);
        let shell_blocks = decoder.decode_shell_blocks(20_000_000, Bandwidth::Wide);
        assert_eq!(shell_blocks, 20);

        let rate_level = decoder.decode_rate_level(&mut range_decoder, false);
        let counts =
            decoder.decode_pulse_and_lsb_counts(&mut range_decoder, shell_blocks, rate_level);
        assert_eq!(counts.block_count, shell_blocks);

        let excitation = decoder.decode_excitation(
            &mut range_decoder,
            FrameSignalType::Unvoiced,
            FrameQuantizationOffsetType::Low,
            seed,
            &counts,
        );

        assert_eq!(EXPECTED.len(), 320);
        assert_eq!(excitation.len, EXPECTED.len());
        assert_eq!(&excitation.values[..excitation.len], EXPECTED);
    }

    #[test]
    fn decode_pitch_lags_returns_none_for_unvoiced_frames() {
        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder::init(&[]);

        let result = decoder
            .decode_pitch_lags(
                &mut range_decoder,
                FrameSignalType::Unvoiced,
                Bandwidth::Wide,
            )
            .expect("pitch lag decoding should not fail for unvoiced frames");
        assert!(result.is_none());
    }

    #[test]
    fn normalize_lsf_interpolation_wq2_equals_four() {
        let mut decoder = DecoderBuilder::new().build();
        let mut range_decoder = RangeDecoder {
            buf: TEST_SILK_FRAME,
            bits_read: 55,
            range_size: 493_249_168,
            high_and_coded_difference: 174_371_199,
        };

        let (n1_q15, w_q2) = decoder.normalize_lsf_interpolation(&mut range_decoder, &[]);
        assert!(n1_q15.is_none());
        assert_eq!(w_q2, 4);
    }

    #[test]
    fn normalize_lsf_interpolation_wq2_equals_one() {
        let frame: &[u8] = &[
            0xac, 0xbd, 0xa9, 0xf7, 0x26, 0x24, 0x5a, 0xa4, 0x00, 0x37, 0xbf, 0x9c, 0xde, 0x0e,
            0xcf, 0x94, 0x64, 0xaa, 0xf9, 0x87, 0xd0, 0x79, 0x19, 0xa8, 0x21, 0xc0,
        ];
        let mut decoder = DecoderBuilder::new().build();
        decoder.have_decoded = true;
        decoder.n0_q15_len = 16;
        let mut range_decoder = RangeDecoder {
            buf: frame,
            bits_read: 65,
            range_size: 1_231_761_776,
            high_and_coded_difference: 1_068_195_183,
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

        let (actual, w_q2) = decoder.normalize_lsf_interpolation(&mut range_decoder, &n2_q15);
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
        let decoder = DecoderBuilder::new().build();

        let nlsf = NlsfQ15::from_slice(&TEST_CONVERT_NLSF_Q15);
        let actual = decoder.convert_normalized_lsfs_to_lpc_coefficients(&nlsf, Bandwidth::Wide);

        assert_eq!(actual.as_slice(), &EXPECTED_A32_Q17);
    }

    #[test]
    fn limit_lpc_coefficients_range_preserves_reference_values() {
        let decoder = DecoderBuilder::new().build();

        let mut a32 = A32Q17::new(EXPECTED_A32_Q17.len());
        a32.as_mut_slice().copy_from_slice(&EXPECTED_A32_Q17);

        decoder.limit_lpc_coefficients_range(&mut a32);

        assert_eq!(a32.as_slice(), &EXPECTED_A32_Q17);
    }

    #[test]
    fn limit_lpc_filter_prediction_gain_matches_reference() {
        let decoder = DecoderBuilder::new().build();

        let mut a32 = A32Q17::new(EXPECTED_A32_Q17.len());
        a32.as_mut_slice().copy_from_slice(&EXPECTED_A32_Q17);

        let a_q12 = decoder.limit_lpc_filter_prediction_gain(&a32);

        assert_eq!(a_q12.as_slice(), &EXPECTED_AQ12);
    }

    #[test]
    fn generate_a_q12_appends_coefficients_when_available() {
        let mut decoder = DecoderBuilder::new().build();

        let nlsf = NlsfQ15::from_slice(&TEST_CONVERT_NLSF_Q15);
        let mut a_q12 = Aq12List::new();

        decoder.generate_a_q12(None, Bandwidth::Wide, &mut a_q12);
        assert!(a_q12.is_empty());

        decoder.generate_a_q12(Some(&nlsf), Bandwidth::Wide, &mut a_q12);
        assert_eq!(a_q12.len(), 1);
        assert_eq!(a_q12.get(0), &EXPECTED_AQ12);
    }
}
