//! Encoder-side state representations.
//!
//! The C implementation layers multiple encoder structs.  Each channel stores a
//! `silk_encoder_state` (`sCmn` in the fixed-point build) that in turn feeds the
//! adaptive high-pass controller and many other helpers.  This module starts
//! porting that hierarchy by exposing the common fields required by the Rust
//! translation of `silk_HP_variable_cutoff`.

use crate::silk::FrameSignalType;
use crate::silk::lin2log::lin2log;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;

/// Number of VAD bands tracked per SILK channel.
pub const VAD_N_BANDS: usize = 4;

/// Minimal subset of the encoder common state needed by the Rust ports.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderStateCommon {
    /// Previously decoded signal classification.
    pub prev_signal_type: FrameSignalType,
    /// Internal sampling rate in kHz.
    pub fs_khz: i32,
    /// Previous frame pitch lag (in samples).
    pub prev_lag: i32,
    /// Per-band input quality metrics in Q15.
    pub input_quality_bands_q15: [i32; VAD_N_BANDS],
    /// Smoothed speech-activity estimate in Q8.
    pub speech_activity_q8: i32,
    /// Smoothed logarithmic cut-off frequency in Q15.
    pub variable_hp_smth1_q15: i32,
}

impl Default for EncoderStateCommon {
    fn default() -> Self {
        Self {
            prev_signal_type: FrameSignalType::Inactive,
            fs_khz: 16,
            prev_lag: 0,
            input_quality_bands_q15: [0; VAD_N_BANDS],
            speech_activity_q8: 0,
            variable_hp_smth1_q15: lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8,
        }
    }
}

/// Encoder channel state (Rust mirror of `silk_encoder_state` minus unported fields).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct EncoderChannelState {
    /// Common fields shared with the floating-point build.
    pub common: EncoderStateCommon,
}

impl EncoderChannelState {
    /// Creates a new channel state with default-initialised members.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a channel state around an existing common state snapshot.
    #[must_use]
    pub fn with_common(common: EncoderStateCommon) -> Self {
        Self { common }
    }

    /// Borrow the common encoder fields.
    #[must_use]
    pub fn common(&self) -> &EncoderStateCommon {
        &self.common
    }

    /// Mutably borrow the common encoder fields.
    #[must_use]
    pub fn common_mut(&mut self) -> &mut EncoderStateCommon {
        &mut self.common
    }

    /// Update the adaptive high-pass smoother using the current channel statistics.
    pub fn update_variable_high_pass(&mut self) {
        crate::silk::hp_variable_cutoff::hp_variable_cutoff(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_state_common_defaults_match_reference() {
        let common = EncoderStateCommon::default();
        assert_eq!(common.prev_signal_type, FrameSignalType::Inactive);
        assert_eq!(common.fs_khz, 16);
        assert_eq!(common.prev_lag, 0);
        assert_eq!(common.input_quality_bands_q15, [0; VAD_N_BANDS]);
        assert_eq!(common.speech_activity_q8, 0);
        assert_eq!(
            common.variable_hp_smth1_q15,
            lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8
        );
    }

    #[test]
    fn encoder_channel_state_default_wraps_common() {
        let channel = EncoderChannelState::default();
        assert_eq!(*channel.common(), EncoderStateCommon::default());
    }

    #[test]
    fn encoder_channel_state_with_common_preserves_input() {
        let mut custom = EncoderStateCommon::default();
        custom.fs_khz = 24;
        let channel = EncoderChannelState::with_common(custom.clone());
        assert_eq!(channel.common(), &custom);
    }
}
