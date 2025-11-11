//! Port of `silk/check_control_input.c`.
//!
//! Validates encoder control parameters before they are applied to the SILK
//! encoder. The checks mirror the reference C implementation so that invalid
//! configurations yield the same error codes as the original library.

use crate::silk::errors::SilkError;

/// Maximum number of channels supported by the SILK encoder.
const ENCODER_NUM_CHANNELS: i32 = 2;

/// Minimum target bitrate supported by the SILK encoder, in bits per second.
///
/// Mirrors `MIN_TARGET_RATE_BPS` from `silk/define.h`.
const MIN_TARGET_RATE_BPS: i32 = 5_000;

/// Maximum target bitrate supported by the SILK encoder, in bits per second.
///
/// Mirrors `MAX_TARGET_RATE_BPS` from `silk/define.h`.
const MAX_TARGET_RATE_BPS: i32 = 80_000;

/// Encoder control parameters mirrored from `silk_EncControlStruct`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncControl {
    /// Number of channels for the API contract; either 1 or 2.
    pub n_channels_api: i32,
    /// Number of internal channels used by the encoder; either 1 or 2.
    pub n_channels_internal: i32,
    /// Input sample rate in Hertz.
    pub api_sample_rate: i32,
    /// Maximum internal sampling rate in Hertz.
    pub max_internal_sample_rate: i32,
    /// Minimum internal sampling rate in Hertz.
    pub min_internal_sample_rate: i32,
    /// Requested internal sampling rate in Hertz.
    pub desired_internal_sample_rate: i32,
    /// Number of milliseconds per packet.
    pub payload_size_ms: i32,
    /// Target bitrate during active speech in bits per second.
    pub bit_rate: i32,
    /// Reported uplink packet loss percentage (0–100).
    pub packet_loss_percentage: i32,
    /// Complexity setting in the range 0–10.
    pub complexity: i32,
    /// Enables in-band forward error correction when set to 1.
    pub use_in_band_fec: i32,
    /// Forces the encoder to emit low-bit-rate redundancy for the current packet.
    pub lbrr_coded: i32,
    /// Enables discontinuous transmission when set to 1.
    pub use_dtx: i32,
    /// Enables constant-bitrate mode when set to 1.
    pub use_cbr: i32,
    /// Maximum number of bits the encoder may spend on the current frame.
    pub max_bits: i32,
    /// Flag indicating that the Opus wrapper allows SILK to switch bandwidth this frame.
    pub opus_can_switch: bool,
    /// Flag set by SILK when it is ready to switch bandwidth.
    pub switch_ready: bool,
}

impl Default for EncControl {
    fn default() -> Self {
        Self {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 16_000,
            max_internal_sample_rate: 16_000,
            min_internal_sample_rate: 16_000,
            desired_internal_sample_rate: 16_000,
            payload_size_ms: 20,
            bit_rate: 32_000,
            packet_loss_percentage: 0,
            complexity: 10,
            use_in_band_fec: 0,
            lbrr_coded: 0,
            use_dtx: 0,
            use_cbr: 0,
            max_bits: 0,
            opus_can_switch: false,
            switch_ready: false,
        }
    }
}

impl EncControl {
    /// Validate the encoder control parameters.
    ///
    /// Returns [`SilkError::NoError`] on success or the matching error code when the
    /// configuration is invalid.
    pub fn check_control_input(&self) -> Result<(), SilkError> {
        const API_SAMPLE_RATES: [i32; 7] = [8000, 12_000, 16_000, 24_000, 32_000, 44_100, 48_000];
        const INTERNAL_SAMPLE_RATES: [i32; 3] = [8000, 12_000, 16_000];
        const PAYLOAD_SIZES_MS: [i32; 4] = [10, 20, 40, 60];

        if !API_SAMPLE_RATES.contains(&self.api_sample_rate)
            || !INTERNAL_SAMPLE_RATES.contains(&self.desired_internal_sample_rate)
            || !INTERNAL_SAMPLE_RATES.contains(&self.max_internal_sample_rate)
            || !INTERNAL_SAMPLE_RATES.contains(&self.min_internal_sample_rate)
            || self.min_internal_sample_rate > self.desired_internal_sample_rate
            || self.max_internal_sample_rate < self.desired_internal_sample_rate
            || self.min_internal_sample_rate > self.max_internal_sample_rate
        {
            return Err(SilkError::EncFsNotSupported);
        }

        if self.bit_rate < MIN_TARGET_RATE_BPS || self.bit_rate > MAX_TARGET_RATE_BPS {
            return Err(SilkError::EncInvalidBitrate);
        }

        if !PAYLOAD_SIZES_MS.contains(&self.payload_size_ms) {
            return Err(SilkError::EncPacketSizeNotSupported);
        }

        if !(0..=100).contains(&self.packet_loss_percentage) {
            return Err(SilkError::EncInvalidLossRate);
        }

        if !matches!(self.use_dtx, 0 | 1) {
            return Err(SilkError::EncInvalidDtxSetting);
        }

        if !matches!(self.use_cbr, 0 | 1) {
            return Err(SilkError::EncInvalidCbrSetting);
        }

        if !matches!(self.use_in_band_fec, 0 | 1) {
            return Err(SilkError::EncInvalidInbandFecSetting);
        }

        if !matches!(self.lbrr_coded, 0 | 1) {
            return Err(SilkError::EncInternalError);
        }

        if self.n_channels_api < 1
            || self.n_channels_api > ENCODER_NUM_CHANNELS
            || self.n_channels_internal < 1
            || self.n_channels_internal > ENCODER_NUM_CHANNELS
            || self.n_channels_internal > self.n_channels_api
        {
            return Err(SilkError::EncInvalidNumberOfChannelsError);
        }

        if !(0..=10).contains(&self.complexity) {
            return Err(SilkError::EncInvalidComplexitySetting);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_configuration() {
        let control = EncControl::default();
        assert_eq!(control.check_control_input(), Ok(()));
    }

    #[test]
    fn rejects_invalid_sample_rates() {
        let mut control = EncControl::default();
        control.api_sample_rate = 11_000;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncFsNotSupported)
        );

        control.api_sample_rate = 16_000;
        control.desired_internal_sample_rate = 20_000;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncFsNotSupported)
        );

        control.desired_internal_sample_rate = 12_000;
        control.max_internal_sample_rate = 8_000;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncFsNotSupported)
        );
    }

    #[test]
    fn validates_payload_size() {
        let mut control = EncControl::default();
        control.payload_size_ms = 15;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncPacketSizeNotSupported)
        );
    }

    #[test]
    fn validates_bit_rate_bounds() {
        let mut control = EncControl::default();
        control.bit_rate = 4_000;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidBitrate)
        );

        control.bit_rate = 90_000;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidBitrate)
        );
    }

    #[test]
    fn checks_boolean_flags() {
        let mut control = EncControl::default();
        control.use_dtx = 2;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidDtxSetting)
        );

        control.use_dtx = 0;
        control.use_cbr = -1;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidCbrSetting)
        );

        control.use_cbr = 0;
        control.use_in_band_fec = 5;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidInbandFecSetting)
        );
    }

    #[test]
    fn validates_channel_configuration() {
        let mut control = EncControl::default();
        control.n_channels_api = 0;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidNumberOfChannelsError)
        );

        control.n_channels_api = 2;
        control.n_channels_internal = 3;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidNumberOfChannelsError)
        );

        control.n_channels_internal = 2;
        control.n_channels_api = 1;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidNumberOfChannelsError)
        );
    }

    #[test]
    fn enforces_complexity_bounds() {
        let mut control = EncControl::default();
        control.complexity = 11;
        assert_eq!(
            control.check_control_input(),
            Err(SilkError::EncInvalidComplexitySetting)
        );
    }
}
