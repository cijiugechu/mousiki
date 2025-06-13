use super::icdf::{FRAME_TYPE_VAD_ACTIVE, FRAME_TYPE_VAD_INACTIVE};
use super::{FrameQuantizationOffsetType, FrameSignalType};
use crate::range::RangeDecoder;
use crate::silk::icdf::{
    DELTA_QUANTIZATION_GAIN, INDEPENDENT_QUANTIZATION_GAIN_LSB,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED,
    INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED,
};

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
            previous_log_gain: 0,
            final_out_values: self.final_out_values,
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
    previous_log_gain: i32,
    final_out_values: [f32; 306],
}

const SUBFRAME_COUNT: usize = 4;

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
            (false, 0) => {
                (FrameSignalType::Inactive, FrameQuantizationOffsetType::Low)
            }
            (false, _) => {
                (FrameSignalType::Inactive, FrameQuantizationOffsetType::High)
            }
            (true, 0) => {
                (FrameSignalType::Unvoiced, FrameQuantizationOffsetType::Low)
            }
            (true, 1) => {
                (FrameSignalType::Unvoiced, FrameQuantizationOffsetType::High)
            }
            (true, 2) => {
                (FrameSignalType::Voiced, FrameQuantizationOffsetType::Low)
            }
            (true, 3) => {
                (FrameSignalType::Voiced, FrameQuantizationOffsetType::High)
            }
            _ => unreachable!(),
        }
    }

    #[allow(unused_assignments)]
    pub fn decode_subframe_quantizations(
        &mut self,
        signal_type: FrameSignalType,
    ) -> [f32; SUBFRAME_COUNT] {
        let mut log_gain = 0;
        let mut delta_gain_index = 0;
        let mut gain_index: i32 = 0;
        let mut gain_q_16 = [0f32; SUBFRAME_COUNT];

        for subframe_index in 0..SUBFRAME_COUNT {
            // The subframe gains are either coded independently, or relative to the
            // gain from the most recent coded subframe in the same channel.
            //
            // https://datatracker.ietf.org/doc/html/rfc6716#section-4.2.7.4
            if subframe_index == 0 {
                // In an independently coded subframe gain, the 3 most significant bits
                // of the quantization gain are decoded using a PDF selected from
                // Table 11 based on the decoded signal type
                gain_index = match signal_type {
                    FrameSignalType::Inactive => {
                        self.range_decoder.decode_symbol_with_icdf(
                            INDEPENDENT_QUANTIZATION_GAIN_MSB_INACTIVE,
                        ) as i32
                    }
                    FrameSignalType::Voiced => {
                        self.range_decoder.decode_symbol_with_icdf(
                            INDEPENDENT_QUANTIZATION_GAIN_MSB_VOICED,
                        ) as i32
                    }
                    FrameSignalType::Unvoiced => {
                        self.range_decoder.decode_symbol_with_icdf(
                            INDEPENDENT_QUANTIZATION_GAIN_MSB_UNVOICED,
                        ) as i32
                    }
                };

                // The 3 least significant bits are decoded using a uniform PDF:
                // These 6 bits are combined to form a value, gain_index, between 0 and 63.
                gain_index = (gain_index << 3)
                    | ((self.range_decoder.decode_symbol_with_icdf(
                        INDEPENDENT_QUANTIZATION_GAIN_LSB,
                    )) as i32);

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

            gain_q_16[subframe_index] = ((1 << i)
                + (((-174 * f * (128 - f)) >> 16) + f) * ((1 << i) >> 7))
                as f32;
        }

        gain_q_16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SILK_FRAME: &[u8] = &[0x0B, 0xE4, 0xC1, 0x36, 0xEC, 0xC5, 0x80];

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
            previous_log_gain: 0,
            final_out_values: [0.; 306],
        };

        let (signal_type, quantization_offset_type) =
            decoder.determine_frame_type(false);
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
            previous_log_gain: 0,
            final_out_values: [0.; 306],
        };

        let quantizations =
            decoder.decode_subframe_quantizations(FrameSignalType::Inactive);
        assert_eq!(quantizations, [210944., 112640., 96256., 96256.]);
    }
}
