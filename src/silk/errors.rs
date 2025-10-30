//! Port of `silk/errors.h`.
//!
//! Exposes the encoder and decoder error codes used throughout the SILK
//! implementation. These mirror the C definitions exactly so that Rust callers
//! can perform the same error classification as the reference implementation.

use core::fmt;

/// Error codes produced by the SILK encoder and decoder.
///
/// The discriminant values mirror the constants defined in
/// `silk/errors.h`, preserving the original numeric codes.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SilkError {
    /// No error occurred.
    NoError = 0,

    /// Input length is not a multiple of 10 ms, or the length exceeds the packet size.
    EncInputInvalidNoOfSamples = -101,

    /// Sampling frequency not 8000, 12000 or 16000 Hertz.
    EncFsNotSupported = -102,

    /// Packet size not 10, 20, 40, or 60 ms.
    EncPacketSizeNotSupported = -103,

    /// Allocated payload buffer too short.
    EncPayloadBufTooShort = -104,

    /// Loss rate not between 0 and 100 percent.
    EncInvalidLossRate = -105,

    /// Complexity setting not valid, must be within 0..=10.
    EncInvalidComplexitySetting = -106,

    /// In-band FEC setting not valid, must be 0 or 1.
    EncInvalidInbandFecSetting = -107,

    /// DTX setting not valid, must be 0 or 1.
    EncInvalidDtxSetting = -108,

    /// Constant-bit-rate setting not valid, must be 0 or 1.
    EncInvalidCbrSetting = -109,

    /// Internal encoder error.
    EncInternalError = -110,

    /// Number of channels setting invalid.
    EncInvalidNumberOfChannelsError = -111,

    /// Target bitrate is outside the supported range.
    EncInvalidBitrate = -112,

    /// Output sampling frequency lower than the internal decoded sampling frequency.
    DecInvalidSamplingFrequency = -200,

    /// Payload size exceeded the maximum allowed 1024 bytes.
    DecPayloadTooLarge = -201,

    /// Payload has bit errors.
    DecPayloadError = -202,

    /// Frame size is invalid.
    DecInvalidFrameSize = -203,
}

impl SilkError {
    /// Returns the numeric error code corresponding to this enum variant.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Converts a raw SILK error code into the corresponding [`SilkError`] value.
    pub const fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::NoError),
            -101 => Some(Self::EncInputInvalidNoOfSamples),
            -102 => Some(Self::EncFsNotSupported),
            -103 => Some(Self::EncPacketSizeNotSupported),
            -104 => Some(Self::EncPayloadBufTooShort),
            -105 => Some(Self::EncInvalidLossRate),
            -106 => Some(Self::EncInvalidComplexitySetting),
            -107 => Some(Self::EncInvalidInbandFecSetting),
            -108 => Some(Self::EncInvalidDtxSetting),
            -109 => Some(Self::EncInvalidCbrSetting),
            -110 => Some(Self::EncInternalError),
            -111 => Some(Self::EncInvalidNumberOfChannelsError),
            -112 => Some(Self::EncInvalidBitrate),
            -200 => Some(Self::DecInvalidSamplingFrequency),
            -201 => Some(Self::DecPayloadTooLarge),
            -202 => Some(Self::DecPayloadError),
            -203 => Some(Self::DecInvalidFrameSize),
            _ => None,
        }
    }
}

impl fmt::Display for SilkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

#[cfg(test)]
mod tests {
    use super::SilkError;

    #[test]
    fn discriminants_match_reference_values() {
        assert_eq!(SilkError::NoError.code(), 0);
        assert_eq!(SilkError::EncInvalidLossRate as i32, -105);
        assert_eq!(SilkError::EncInvalidBitrate as i32, -112);
        assert_eq!(SilkError::DecInvalidFrameSize as i32, -203);
    }

    #[test]
    fn round_trips_from_raw_codes() {
        for (code, expected) in [
            (0, SilkError::NoError),
            (-101, SilkError::EncInputInvalidNoOfSamples),
            (-110, SilkError::EncInternalError),
            (-112, SilkError::EncInvalidBitrate),
            (-201, SilkError::DecPayloadTooLarge),
        ] {
            assert_eq!(SilkError::from_code(code), Some(expected));
        }

        assert_eq!(SilkError::from_code(-999), None);
    }
}
