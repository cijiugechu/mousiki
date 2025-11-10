//! Encoder-side state representations.
//!
//! The C implementation layers multiple encoder structs.  Each channel stores a
//! `silk_encoder_state` (`sCmn` in the fixed-point build) that in turn feeds the
//! adaptive high-pass controller and many other helpers.  This module starts
//! porting that hierarchy by exposing the common fields required by the Rust
//! translation of `silk_HP_variable_cutoff`.

use crate::silk::FrameSignalType;
use crate::silk::MAX_NB_SUBFR;
use crate::silk::lin2log::lin2log;
use crate::silk::lp_variable_cutoff::LpState;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;

/// Number of VAD bands tracked per SILK channel.
pub const VAD_N_BANDS: usize = 4;
/// Internal SILK maximum sampling rate (kHz).
pub(crate) const MAX_FS_KHZ: usize = 16;
/// Sub-frame duration in milliseconds.
pub(crate) const SUB_FRAME_LENGTH_MS: usize = 5;
/// Default number of milliseconds per frame.
pub(crate) const MAX_FRAME_LENGTH_MS: usize = SUB_FRAME_LENGTH_MS * MAX_NB_SUBFR;
/// Default internal sampling rate in kHz used when initialising the encoder state.
const DEFAULT_INTERNAL_FS_KHZ: i32 = 16;
/// Default frame length in samples (20 ms @ 16 kHz).
pub(crate) const DEFAULT_FRAME_LENGTH: usize =
    MAX_FRAME_LENGTH_MS * DEFAULT_INTERNAL_FS_KHZ as usize;
/// Bias used by the VAD noise estimator.
pub(crate) const VAD_NOISE_LEVELS_BIAS: i32 = 50;
/// Initial smoothed SNR per VAD band (100 * 256 -> 20 dB).
const INITIAL_NRG_RATIO_Q8: i32 = 100 * 256;
/// Number of frames used for the initial fast noise update phase.
const INITIAL_VAD_COUNTER: i32 = 15;

/// Fixed-point voice activity detector state (mirror of `silk_VAD_state`).
#[derive(Clone, Debug, PartialEq)]
pub struct VadState {
    pub ana_state: [i32; 2],
    pub ana_state1: [i32; 2],
    pub ana_state2: [i32; 2],
    pub xnrg_subfr: [i32; VAD_N_BANDS],
    pub nrg_ratio_smth_q8: [i32; VAD_N_BANDS],
    pub hp_state: i16,
    pub nl: [i32; VAD_N_BANDS],
    pub inv_nl: [i32; VAD_N_BANDS],
    pub noise_level_bias: [i32; VAD_N_BANDS],
    pub counter: i32,
}

impl Default for VadState {
    fn default() -> Self {
        let mut state = Self {
            ana_state: [0; 2],
            ana_state1: [0; 2],
            ana_state2: [0; 2],
            xnrg_subfr: [0; VAD_N_BANDS],
            nrg_ratio_smth_q8: [INITIAL_NRG_RATIO_Q8; VAD_N_BANDS],
            hp_state: 0,
            nl: [0; VAD_N_BANDS],
            inv_nl: [0; VAD_N_BANDS],
            noise_level_bias: [0; VAD_N_BANDS],
            counter: INITIAL_VAD_COUNTER,
        };
        state.reset();
        state
    }
}

impl VadState {
    /// Mirrors `silk_VAD_Init` by reinitialising the noise estimator members.
    pub fn reset(&mut self) {
        for (band, bias) in self.noise_level_bias.iter_mut().enumerate() {
            *bias = (VAD_NOISE_LEVELS_BIAS / (band as i32 + 1)).max(1);
        }
        for (nl, bias) in self.nl.iter_mut().zip(self.noise_level_bias.iter()) {
            *nl = 100 * *bias;
        }
        for (inv, nl) in self.inv_nl.iter_mut().zip(self.nl.iter()) {
            *inv = if *nl != 0 { i32::MAX / *nl } else { 0 };
        }
        self.nrg_ratio_smth_q8 = [INITIAL_NRG_RATIO_Q8; VAD_N_BANDS];
        self.xnrg_subfr = [0; VAD_N_BANDS];
        self.hp_state = 0;
        self.counter = INITIAL_VAD_COUNTER;
    }
}

/// Minimal subset of the encoder common state needed by the Rust ports.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderStateCommon {
    /// Previously decoded signal classification.
    pub prev_signal_type: FrameSignalType,
    /// Internal sampling rate in kHz.
    pub fs_khz: i32,
    /// Number of 5 ms subframes tracked per frame (2 or 4).
    pub nb_subfr: usize,
    /// Active frame length in samples.
    pub frame_length: usize,
    /// External API sample rate in Hz.
    pub api_sample_rate_hz: i32,
    /// Maximum internal sampling rate allowed in Hz.
    pub max_internal_sample_rate_hz: i32,
    /// Minimum internal sampling rate allowed in Hz.
    pub min_internal_sample_rate_hz: i32,
    /// Requested internal sampling rate in Hz.
    pub desired_internal_sample_rate_hz: i32,
    /// Whether the encoder may change its internal bandwidth this frame.
    pub allow_bandwidth_switch: bool,
    /// Previous frame pitch lag (in samples).
    pub prev_lag: i32,
    /// Target bitrate expressed in bits per second.
    pub target_rate_bps: i32,
    /// Encoder-side SNR tuning value in Q7.
    pub snr_db_q7: i32,
    /// Per-band input quality metrics in Q15.
    pub input_quality_bands_q15: [i32; VAD_N_BANDS],
    /// Smoothed tilt estimate in Q15.
    pub input_tilt_q15: i32,
    /// Smoothed speech-activity estimate in Q8.
    pub speech_activity_q8: i32,
    /// Smoothed logarithmic cut-off frequency in Q15.
    pub variable_hp_smth1_q15: i32,
}

impl Default for EncoderStateCommon {
    fn default() -> Self {
        Self {
            prev_signal_type: FrameSignalType::Inactive,
            fs_khz: DEFAULT_INTERNAL_FS_KHZ,
            nb_subfr: MAX_NB_SUBFR,
            frame_length: DEFAULT_FRAME_LENGTH,
            api_sample_rate_hz: DEFAULT_INTERNAL_FS_KHZ * 1000,
            max_internal_sample_rate_hz: DEFAULT_INTERNAL_FS_KHZ * 1000,
            min_internal_sample_rate_hz: DEFAULT_INTERNAL_FS_KHZ * 1000,
            desired_internal_sample_rate_hz: DEFAULT_INTERNAL_FS_KHZ * 1000,
            allow_bandwidth_switch: false,
            prev_lag: 0,
            target_rate_bps: 0,
            snr_db_q7: 0,
            input_quality_bands_q15: [0; VAD_N_BANDS],
            input_tilt_q15: 0,
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
    /// Voice activity detector state.
    pub vad: VadState,
    /// Variable low-pass filter state used during bandwidth transitions.
    pub lp_state: LpState,
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
        Self {
            common,
            vad: VadState::default(),
            lp_state: LpState::default(),
        }
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

    /// Borrow the VAD state.
    #[must_use]
    pub fn vad(&self) -> &VadState {
        &self.vad
    }

    /// Mutably borrow the VAD state.
    #[must_use]
    pub fn vad_mut(&mut self) -> &mut VadState {
        &mut self.vad
    }

    /// Simultaneously borrow the common encoder fields and VAD state.
    pub(crate) fn parts_mut(&mut self) -> (&mut EncoderStateCommon, &mut VadState) {
        let ptr = self as *mut Self;
        unsafe { (&mut (*ptr).common, &mut (*ptr).vad) }
    }

    /// Simultaneously borrow the common encoder fields and low-pass transition state.
    pub(crate) fn common_and_lp_mut(&mut self) -> (&mut EncoderStateCommon, &mut LpState) {
        let ptr = self as *mut Self;
        unsafe { (&mut (*ptr).common, &mut (*ptr).lp_state) }
    }

    /// Borrow the bandwidth-transition low-pass state.
    #[must_use]
    pub fn low_pass_state(&self) -> &LpState {
        &self.lp_state
    }

    /// Mutably borrow the bandwidth-transition low-pass state.
    #[must_use]
    pub fn low_pass_state_mut(&mut self) -> &mut LpState {
        &mut self.lp_state
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
        assert_eq!(common.fs_khz, DEFAULT_INTERNAL_FS_KHZ);
        assert_eq!(common.nb_subfr, MAX_NB_SUBFR);
        assert_eq!(common.frame_length, DEFAULT_FRAME_LENGTH);
        assert_eq!(common.api_sample_rate_hz, DEFAULT_INTERNAL_FS_KHZ * 1000);
        assert_eq!(
            common.max_internal_sample_rate_hz,
            DEFAULT_INTERNAL_FS_KHZ * 1000
        );
        assert_eq!(
            common.min_internal_sample_rate_hz,
            DEFAULT_INTERNAL_FS_KHZ * 1000
        );
        assert_eq!(
            common.desired_internal_sample_rate_hz,
            DEFAULT_INTERNAL_FS_KHZ * 1000
        );
        assert!(!common.allow_bandwidth_switch);
        assert_eq!(common.prev_lag, 0);
        assert_eq!(common.target_rate_bps, 0);
        assert_eq!(common.snr_db_q7, 0);
        assert_eq!(common.input_quality_bands_q15, [0; VAD_N_BANDS]);
        assert_eq!(common.input_tilt_q15, 0);
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
        assert_eq!(channel.vad(), &VadState::default());
        assert_eq!(channel.low_pass_state(), &LpState::default());
    }

    #[test]
    fn encoder_channel_state_with_common_preserves_input() {
        let mut custom = EncoderStateCommon::default();
        custom.fs_khz = 24;
        let channel = EncoderChannelState::with_common(custom.clone());
        assert_eq!(channel.common(), &custom);
    }

    #[test]
    fn vad_state_reset_matches_reference_bias() {
        let mut vad = VadState::default();
        vad.noise_level_bias = [0; VAD_N_BANDS];
        vad.reset();
        assert_eq!(vad.noise_level_bias[0], VAD_NOISE_LEVELS_BIAS);
        assert_eq!(vad.noise_level_bias[1], VAD_NOISE_LEVELS_BIAS / 2);
        assert_eq!(vad.noise_level_bias[2], VAD_NOISE_LEVELS_BIAS / 3);
        assert_eq!(vad.noise_level_bias[3], VAD_NOISE_LEVELS_BIAS / 4);
        assert!(vad.nl.iter().all(|&nl| nl > 0));
        assert!(vad.inv_nl.iter().all(|&inv| inv > 0));
        assert_eq!(vad.nrg_ratio_smth_q8, [INITIAL_NRG_RATIO_Q8; VAD_N_BANDS]);
    }
}
