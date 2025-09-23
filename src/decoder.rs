use crate::bitdepth::{BitDepthError, convert_float32_le_to_signed16_le};
use crate::packet::{Bandwidth, Mode};
use crate::resample;
use crate::silk::decoder as silk_decoder;
use log::{debug, trace};

/// Number of float samples produced by the SILK decoder for a single 20 ms frame.
const SILK_FRAME_SAMPLES: usize = 320;
/// Opus SILK frames are upsampled by a factor of three to reach 48 kHz output.
const UPSAMPLE_FACTOR: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderError {
    TooShortForTableOfContents,
    UnsupportedFrameCode(u8),
    UnsupportedConfigurationMode,
    Silk(silk_decoder::DecodeError),
    BitDepth(BitDepthError),
}

impl core::fmt::Display for DecoderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShortForTableOfContents => {
                f.write_str("packet is too short to contain table of contents header")
            }
            Self::UnsupportedFrameCode(code) => {
                write!(f, "unsupported frame code: {code}")
            }
            Self::UnsupportedConfigurationMode => f.write_str("unsupported configuration mode"),
            Self::Silk(err) => write!(f, "silk decode error: {err}"),
            Self::BitDepth(err) => write!(f, "bit depth conversion failed: {err:?}"),
        }
    }
}

impl core::error::Error for DecoderError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Silk(err) => Some(err),
            Self::BitDepth(_err) => None,
            _ => None,
        }
    }
}

impl From<silk_decoder::DecodeError> for DecoderError {
    fn from(value: silk_decoder::DecodeError) -> Self {
        Self::Silk(value)
    }
}

impl From<BitDepthError> for DecoderError {
    fn from(value: BitDepthError) -> Self {
        Self::BitDepth(value)
    }
}

#[derive(Debug, Clone, Copy)]
struct TableOfContentsHeader(u8);

impl TableOfContentsHeader {
    fn new(byte: u8) -> Self {
        Self(byte)
    }

    fn configuration(self) -> Configuration {
        Configuration(self.0 >> 3)
    }

    fn is_stereo(self) -> bool {
        (self.0 & 0b0000_0100) != 0
    }

    fn frame_code(self) -> FrameCountCodeDiscriminant {
        match self.0 & 0b0000_0011 {
            0 => FrameCountCodeDiscriminant::Single,
            1 => FrameCountCodeDiscriminant::DoubleEqual,
            2 => FrameCountCodeDiscriminant::DoubleDifferent,
            _ => FrameCountCodeDiscriminant::Arbitrary,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameCountCodeDiscriminant {
    Single,
    DoubleEqual,
    DoubleDifferent,
    Arbitrary,
}

#[derive(Debug, Clone, Copy)]
struct Configuration(u8);

impl Configuration {
    fn mode(self) -> Option<Mode> {
        let value = self.0;
        match value {
            0..=11 => Some(Mode::SILK),
            12..=15 => Some(Mode::HYBRID),
            16..=31 => Some(Mode::CELT),
            _ => None,
        }
    }

    fn frame_duration_nanoseconds(self) -> Option<u32> {
        match self.0 {
            16 | 20 | 24 | 28 => Some(2_500_000),
            17 | 21 | 25 | 29 => Some(5_000_000),
            0 | 4 | 8 | 12 | 14 | 18 | 22 | 26 | 30 => Some(10_000_000),
            1 | 5 | 9 | 13 | 15 | 19 | 23 | 27 | 31 => Some(20_000_000),
            2 | 6 => Some(40_000_000),
            3 | 7 | 11 => Some(60_000_000),
            _ => None,
        }
    }

    fn bandwidth(self) -> Option<Bandwidth> {
        match self.0 {
            0..=3 => Some(Bandwidth::Narrow),
            4..=7 => Some(Bandwidth::Medium),
            8..=11 => Some(Bandwidth::Wide),
            12..=13 => Some(Bandwidth::SuperWide),
            14..=15 => Some(Bandwidth::Full),
            16..=19 => Some(Bandwidth::Narrow),
            20..=23 => Some(Bandwidth::Wide),
            24..=27 => Some(Bandwidth::SuperWide),
            28..=31 => Some(Bandwidth::Full),
            _ => None,
        }
    }
}

pub struct Decoder {
    silk_decoder: silk_decoder::Decoder,
    silk_buffer: [f32; SILK_FRAME_SAMPLES],
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            silk_decoder: silk_decoder::DecoderBuilder::new().build(),
            silk_buffer: [0.0; SILK_FRAME_SAMPLES],
        }
    }

    fn decode_internal(&mut self, input: &[u8]) -> Result<(Bandwidth, bool), DecoderError> {
        if input.is_empty() {
            return Err(DecoderError::TooShortForTableOfContents);
        }

        let toc_header = TableOfContentsHeader::new(input[0]);
        let configuration = toc_header.configuration();
        let frame_code = toc_header.frame_code();

        trace!(
            "decode_internal: toc=0x{:02x}, frame_code={:?}, stereo={} configuration={:?}",
            input[0],
            frame_code,
            toc_header.is_stereo(),
            configuration
        );

        if frame_code != FrameCountCodeDiscriminant::Single {
            debug!(
                "decode_internal: unsupported frame code {:?} in header byte 0x{:02x}",
                frame_code, input[0]
            );
            return Err(DecoderError::UnsupportedFrameCode(input[0] & 0b11));
        }

        let mode = configuration
            .mode()
            .ok_or(DecoderError::UnsupportedConfigurationMode)?;
        if mode != Mode::SILK {
            debug!(
                "decode_internal: rejecting non-SILK mode {:?} from configuration {:?}",
                mode, configuration
            );
            return Err(DecoderError::UnsupportedConfigurationMode);
        }

        let nanoseconds = configuration
            .frame_duration_nanoseconds()
            .ok_or(DecoderError::UnsupportedConfigurationMode)?;
        let bandwidth = configuration
            .bandwidth()
            .ok_or(DecoderError::UnsupportedConfigurationMode)?;

        let encoded_frame = &input[1..];

        trace!(
            "decode_internal: dispatching SILK frame with bandwidth {:?}, frame_ns={}, stereo={} (payload {} bytes)",
            bandwidth,
            nanoseconds,
            toc_header.is_stereo(),
            encoded_frame.len()
        );

        self.silk_decoder.decode(
            encoded_frame,
            &mut self.silk_buffer,
            toc_header.is_stereo(),
            nanoseconds,
            bandwidth,
        )?;

        trace!(
            "decode_internal: SILK decode complete (bandwidth {:?}, stereo={})",
            bandwidth,
            toc_header.is_stereo()
        );

        Ok((bandwidth, toc_header.is_stereo()))
    }

    pub fn decode(
        &mut self,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<(Bandwidth, bool), DecoderError> {
        let (bandwidth, stereo) = self.decode_internal(input)?;
        trace!(
            "decode: converting frame to i16 PCM (bandwidth {:?}, stereo={}, out_len={})",
            bandwidth,
            stereo,
            out.len()
        );
        convert_float32_le_to_signed16_le(&self.silk_buffer, out, UPSAMPLE_FACTOR)?;
        Ok((bandwidth, stereo))
    }

    pub fn decode_float32(
        &mut self,
        input: &[u8],
        out: &mut [f32],
    ) -> Result<(Bandwidth, bool), DecoderError> {
        let (bandwidth, stereo) = self.decode_internal(input)?;
        resample::up(&self.silk_buffer, out, UPSAMPLE_FACTOR);
        trace!(
            "decode_float32: upsampled frame (bandwidth {:?}, stereo={}, out_len={})",
            bandwidth,
            stereo,
            out.len()
        );
        Ok((bandwidth, stereo))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PACKET: &[u8] = &[0x48, 0x0B, 0xE4, 0xC1, 0x36, 0xEC, 0xC5, 0x80];
    const STEREO_PACKET: &[u8] = &[0x4C, 0x0B, 0xE4, 0xC1, 0x36, 0xEC, 0xC5, 0x80];
    const EXPECTED_FLOATS_PREFIX: &[f32] = &[
        0.000023, 0.000025, 0.000027, -0.000018, 0.000025, -0.000021, 0.000021, -0.000024,
        0.000021, 0.000021, -0.000022, -0.000026,
    ];

    #[test]
    fn decode_errors_when_packet_too_short() {
        let mut decoder = Decoder::new();
        let mut out = [0u8; SILK_FRAME_SAMPLES * UPSAMPLE_FACTOR * 2];
        let err = decoder.decode(&[], &mut out).unwrap_err();
        assert!(matches!(err, DecoderError::TooShortForTableOfContents));
    }

    #[test]
    fn decode_errors_when_frame_code_unsupported() {
        let mut decoder = Decoder::new();
        let mut out = [0u8; SILK_FRAME_SAMPLES * UPSAMPLE_FACTOR * 2];
        let packet = &[0x49, 0x00];
        let err = decoder.decode(packet, &mut out).unwrap_err();
        assert!(matches!(err, DecoderError::UnsupportedFrameCode(1)));
    }

    #[test]
    fn decode_errors_when_configuration_not_silk() {
        let mut decoder = Decoder::new();
        let mut out = [0u8; SILK_FRAME_SAMPLES * UPSAMPLE_FACTOR * 2];
        let packet = &[0x80, 0x00];
        let err = decoder.decode(packet, &mut out).unwrap_err();
        assert!(matches!(err, DecoderError::UnsupportedConfigurationMode));
    }

    #[test]
    fn decode_float32_matches_expected_prefix() {
        let mut decoder = Decoder::new();
        let mut out = [0.0f32; SILK_FRAME_SAMPLES * UPSAMPLE_FACTOR];
        let (bandwidth, stereo) = decoder
            .decode_float32(TEST_PACKET, &mut out)
            .expect("decoding should succeed");

        assert_eq!(bandwidth, Bandwidth::Wide);
        assert!(!stereo);

        for (chunk, &expected) in out
            .chunks_exact(UPSAMPLE_FACTOR)
            .zip(EXPECTED_FLOATS_PREFIX.iter())
        {
            assert!((chunk[0] - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn decode_propagates_silk_errors() {
        let mut decoder = Decoder::new();
        let mut out = [0.0f32; SILK_FRAME_SAMPLES * UPSAMPLE_FACTOR];
        let err = decoder.decode_float32(STEREO_PACKET, &mut out).unwrap_err();
        assert!(matches!(
            err,
            DecoderError::Silk(silk_decoder::DecodeError::StereoUnsupported)
        ));
    }

    #[test]
    fn decode_returns_bandwidth_and_stereo_flag() {
        let mut decoder = Decoder::new();
        let mut out = [0u8; SILK_FRAME_SAMPLES * UPSAMPLE_FACTOR * 2];
        let (bandwidth, stereo) = decoder
            .decode(TEST_PACKET, &mut out)
            .expect("decoding should succeed");
        assert_eq!(bandwidth, Bandwidth::Wide);
        assert!(!stereo);
        assert!(out.iter().any(|&byte| byte != 0));
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}
