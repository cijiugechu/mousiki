use super::icdf::{FRAME_TYPE_VAD_ACTIVE, FRAME_TYPE_VAD_INACTIVE};
use super::{FrameQuantizationOffsetType, FrameSignalType};
use crate::range::RangeDecoder;

#[derive(Debug)]
pub struct DecoderBuilder {
    final_out_values: [f32; 306],
}

impl DecoderBuilder {
    pub fn new() -> Self {
        Self {
            final_out_values: [0.; 306],
        }
    }

    pub fn build<'a>(self, buf: &'a [u8]) -> Decoder<'a> {
        Decoder {
            range_decoder: RangeDecoder::init(buf),
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
    final_out_values: [f32; 306],
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
            final_out_values: [0.; 306],
        };

        let (signal_type, quantization_offset_type) = decoder.determine_frame_type(false);
        assert_eq!(signal_type, FrameSignalType::Inactive);
        assert_eq!(quantization_offset_type, FrameQuantizationOffsetType::High);
    }
}
