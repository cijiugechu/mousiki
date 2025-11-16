//! Encoder-side state representations.
//!
//! The C implementation layers multiple encoder structs.  Each channel stores a
//! `silk_encoder_state` (`sCmn` in the fixed-point build) that in turn feeds the
//! adaptive high-pass controller and many other helpers.  This module starts
//! porting that hierarchy by exposing the common fields required by the Rust
//! translation of `silk_HP_variable_cutoff`.

use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::FrameSignalType;
use crate::silk::SilkNlsfCb;
use crate::silk::StereoEncState;
use crate::silk::lin2log::lin2log;
use crate::silk::lp_variable_cutoff::LpState;
use crate::silk::resampler::Resampler;
use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;
use crate::silk::tables_other::SILK_UNIFORM8_ICDF;
use crate::silk::tables_pitch_lag::PITCH_CONTOUR_ICDF;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR};
use core::array::from_fn;

/// Number of VAD bands tracked per SILK channel.
pub const VAD_N_BANDS: usize = 4;
/// Number of encoder channels managed by the top-level super-structure.
pub const ENCODER_NUM_CHANNELS: usize = 2;
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
/// Look-ahead applied during pitch analysis (ms).
pub(crate) const LA_PITCH_MS: usize = 2;
/// Look-ahead applied during noise-shaping analysis (ms).
pub(crate) const LA_SHAPE_MS: usize = 5;
/// Maximum look-ahead in samples for noise shaping.
pub(crate) const LA_SHAPE_MAX: usize = LA_SHAPE_MS * MAX_FS_KHZ;
/// LPC window (ms) used during 20 ms (4 subframe) pitch estimation.
pub(crate) const FIND_PITCH_LPC_WIN_MS: usize = 20 + (LA_PITCH_MS << 1);
/// LPC window (ms) used during 10 ms (2 subframe) pitch estimation.
pub(crate) const FIND_PITCH_LPC_WIN_MS_2_SF: usize = 10 + (LA_PITCH_MS << 1);
/// Number of milliseconds retained in the pitch-analysis buffer.
pub(crate) const LTP_MEM_LENGTH_MS: usize = 20;
/// Maximum frame length in samples given the supported subframes and sampling rates.
pub(crate) const MAX_FRAME_LENGTH: usize = MAX_FRAME_LENGTH_MS * MAX_FS_KHZ;
/// Size of the encoder input buffer.
pub(crate) const INPUT_BUFFER_LENGTH: usize = MAX_FRAME_LENGTH + 2;
/// Number of samples stored in the pitch-analysis scratch buffer.
pub(crate) const X_BUFFER_LENGTH: usize = 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX;
/// Maximum number of delayed-decision states.
pub(crate) const MAX_DEL_DEC_STATES: i32 = 4;
/// Maximum LPC order used during pitch estimation.
pub(crate) const MAX_FIND_PITCH_LPC_ORDER: i32 = 16;
/// Upper bound on the noise-shaping analysis window (samples).
pub(crate) const SHAPE_LPC_WIN_MAX: i32 = 15 * MAX_FS_KHZ as i32;
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

/// Fixed-point noise-shaping analysis state (`silk_shape_state_FIX`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct EncoderShapeState {
    pub last_gain_index: i32,
    pub harm_boost_smth_q16: i32,
    pub harm_shape_gain_smth_q16: i32,
    pub tilt_smth_q16: i32,
}

/// Minimal mirror of `silk_nsq_state` used by the Rust ports so far.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NoiseShapingQuantizerState {
    pub lag_prev: i32,
    pub prev_gain_q16: i32,
}

/// Minimal subset of the encoder common state needed by the Rust ports.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderStateCommon {
    /// Enables discontinuous transmission.
    pub use_dtx: bool,
    /// Enables constant-bit-rate mode.
    pub use_cbr: bool,
    /// Enables in-band forward error correction.
    pub use_in_band_fec: bool,
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
    /// API sample rate used during the previous packet.
    pub prev_api_sample_rate_hz: i32,
    /// Maximum internal sampling rate allowed in Hz.
    pub max_internal_sample_rate_hz: i32,
    /// Minimum internal sampling rate allowed in Hz.
    pub min_internal_sample_rate_hz: i32,
    /// Requested internal sampling rate in Hz.
    pub desired_internal_sample_rate_hz: i32,
    /// Whether the encoder may change its internal bandwidth this frame.
    pub allow_bandwidth_switch: bool,
    /// Number of channels exposed to the API.
    pub n_channels_api: i32,
    /// Number of internal encoder channels.
    pub n_channels_internal: i32,
    /// Channel index within a stereo encoder.
    pub channel_nb: i32,
    /// Previous frame pitch lag (in samples).
    pub prev_lag: i32,
    /// Samples tracked per subframe.
    pub subfr_length: usize,
    /// Samples stored in the LTP state.
    pub ltp_mem_length: usize,
    /// Look-ahead for pitch analysis (samples).
    pub la_pitch: i32,
    /// Look-ahead for noise-shaping analysis (samples).
    pub la_shape: i32,
    /// Window length for noise-shaping analysis (samples).
    pub shape_win_length: i32,
    /// Maximum supported pitch lag in samples.
    pub max_pitch_lag: i32,
    /// Pitch-analysis LPC window length (samples).
    pub pitch_lpc_win_length: usize,
    /// Active LPC order used for prediction.
    pub predict_lpc_order: usize,
    /// Active NLSF codebook.
    pub ps_nlsf_cb: &'static SilkNlsfCb,
    /// Pointer to the active pitch-contour iCDF.
    pub pitch_contour_icdf: &'static [u8],
    /// Pointer to the active low-bit pitch-lag iCDF.
    pub pitch_lag_low_bits_icdf: &'static [u8],
    /// Cached quantised NLSF vector from the previous frame.
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
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
    /// Secondary smoother used by the adaptive high-pass controller.
    pub variable_hp_smth2_q15: i32,
    /// Packet size in milliseconds.
    pub packet_size_ms: i32,
    /// Downlink packet loss percentage.
    pub packet_loss_perc: i32,
    /// Number of frames stored per packet.
    pub n_frames_per_packet: usize,
    /// Number of frames encoded so far in the current packet.
    pub n_frames_encoded: usize,
    /// Quantisation indices for the current frame.
    pub indices: SideInfoIndices,
    /// Write index into the input buffer.
    pub input_buf_ix: usize,
    /// Flag indicating the first frame after a reset.
    pub first_frame_after_reset: bool,
    /// Ensures codec control only runs once per packet.
    pub controlled_since_last_payload: bool,
    /// Indicates that only buffers were prefilled (no coding).
    pub prefill_flag: bool,
    /// Pitch-estimator complexity level.
    pub pitch_estimation_complexity: i32,
    /// Pitch-estimator threshold in Q16.
    pub pitch_estimation_threshold_q16: i32,
    /// Pitch-estimator LPC order.
    pub pitch_estimation_lpc_order: i32,
    /// LPC order used for noise-shaping filters.
    pub shaping_lpc_order: i32,
    /// Number of delayed-decision states.
    pub n_states_delayed_decision: i32,
    /// Enables NLSF interpolation.
    pub use_interpolated_nlsfs: bool,
    /// Number of survivors in the NLSF MSVQ search.
    pub nlsf_msvq_survivors: i32,
    /// Warping control parameter in Q16.
    pub warping_q16: i32,
    /// Complexity setting (0-10).
    pub complexity: i32,
    /// Tracks whether low-bit-rate redundancy is enabled.
    pub lbrr_enabled: bool,
    /// Gain increase applied when coding LBRR frames.
    pub lbrr_gain_increases: i32,
    /// Architecture flag used to select specialised kernels.
    pub arch: i32,
}

impl Default for EncoderStateCommon {
    fn default() -> Self {
        let api_fs_hz = DEFAULT_INTERNAL_FS_KHZ * 1000;
        let subfr_length = SUB_FRAME_LENGTH_MS * DEFAULT_INTERNAL_FS_KHZ as usize;
        let ltp_mem_length = LTP_MEM_LENGTH_MS * DEFAULT_INTERNAL_FS_KHZ as usize;
        let la_pitch = (LA_PITCH_MS as i32) * DEFAULT_INTERNAL_FS_KHZ;
        let la_shape = (LA_SHAPE_MS as i32) * DEFAULT_INTERNAL_FS_KHZ;
        let shape_win_length =
            (SUB_FRAME_LENGTH_MS as i32 * DEFAULT_INTERNAL_FS_KHZ) + 2 * la_shape;
        let max_pitch_lag = 18 * DEFAULT_INTERNAL_FS_KHZ;
        let pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * DEFAULT_INTERNAL_FS_KHZ as usize;
        Self {
            use_dtx: false,
            use_cbr: false,
            use_in_band_fec: false,
            prev_signal_type: FrameSignalType::Inactive,
            fs_khz: DEFAULT_INTERNAL_FS_KHZ,
            nb_subfr: MAX_NB_SUBFR,
            frame_length: DEFAULT_FRAME_LENGTH,
            api_sample_rate_hz: api_fs_hz,
            prev_api_sample_rate_hz: api_fs_hz,
            max_internal_sample_rate_hz: api_fs_hz,
            min_internal_sample_rate_hz: api_fs_hz,
            desired_internal_sample_rate_hz: api_fs_hz,
            allow_bandwidth_switch: false,
            n_channels_api: 1,
            n_channels_internal: 1,
            channel_nb: 0,
            prev_lag: 0,
            subfr_length,
            ltp_mem_length,
            la_pitch,
            la_shape,
            shape_win_length,
            max_pitch_lag,
            pitch_lpc_win_length,
            predict_lpc_order: MAX_LPC_ORDER,
            ps_nlsf_cb: &SILK_NLSF_CB_WB,
            pitch_contour_icdf: &PITCH_CONTOUR_ICDF,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM8_ICDF,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            target_rate_bps: 0,
            snr_db_q7: 0,
            input_quality_bands_q15: [0; VAD_N_BANDS],
            input_tilt_q15: 0,
            speech_activity_q8: 0,
            variable_hp_smth1_q15: lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8,
            variable_hp_smth2_q15: lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8,
            packet_size_ms: MAX_FRAME_LENGTH_MS as i32,
            packet_loss_perc: 0,
            n_frames_per_packet: 1,
            n_frames_encoded: 0,
            indices: SideInfoIndices::default(),
            input_buf_ix: 0,
            first_frame_after_reset: true,
            controlled_since_last_payload: false,
            prefill_flag: false,
            pitch_estimation_complexity: 0,
            pitch_estimation_threshold_q16: 0,
            pitch_estimation_lpc_order: 0,
            shaping_lpc_order: 0,
            n_states_delayed_decision: 0,
            use_interpolated_nlsfs: false,
            nlsf_msvq_survivors: 0,
            warping_q16: 0,
            complexity: 0,
            lbrr_enabled: false,
            lbrr_gain_increases: 0,
            arch: 0,
        }
    }
}

/// Encoder channel state (Rust mirror of `silk_encoder_state` minus unported fields).
#[derive(Clone, Debug)]
pub struct EncoderChannelState {
    /// Common fields shared with the floating-point build.
    pub common: EncoderStateCommon,
    /// Voice activity detector state.
    pub vad: VadState,
    /// Variable low-pass filter state used during bandwidth transitions.
    pub lp_state: LpState,
    /// Noise-shaping analysis state.
    pub shape_state: EncoderShapeState,
    /// Noise-shaping quantiser state.
    pub nsq_state: NoiseShapingQuantizerState,
    /// High-level SILK resampler state used by the API wrapper.
    pub resampler_state: Resampler,
    /// Buffered input samples preserved across frames.
    pub input_buf: [i16; INPUT_BUFFER_LENGTH],
    /// Pitch-analysis buffer mirrored from `x_buf`.
    pub x_buf: [i16; X_BUFFER_LENGTH],
}

impl Default for EncoderChannelState {
    fn default() -> Self {
        Self {
            common: EncoderStateCommon::default(),
            vad: VadState::default(),
            lp_state: LpState::default(),
            shape_state: EncoderShapeState::default(),
            nsq_state: NoiseShapingQuantizerState::default(),
            resampler_state: Resampler::default(),
            input_buf: [0; INPUT_BUFFER_LENGTH],
            x_buf: [0; X_BUFFER_LENGTH],
        }
    }
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
            shape_state: EncoderShapeState::default(),
            nsq_state: NoiseShapingQuantizerState::default(),
            resampler_state: Resampler::default(),
            input_buf: [0; INPUT_BUFFER_LENGTH],
            x_buf: [0; X_BUFFER_LENGTH],
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

/// Top-level encoder super-structure mirroring `silk_encoder`.
#[derive(Clone, Debug)]
pub struct Encoder {
    /// Per-channel encoder states (`state_Fxx` in the reference sources).
    pub state_fxx: [EncoderChannelState; ENCODER_NUM_CHANNELS],
    /// Stereo mid/side predictor state shared across channels.
    pub stereo_state: StereoEncState,
    /// Number of bits consumed by the LBRR side channel in the current packet.
    pub n_bits_used_lbrr: i32,
    /// Tracks bit-budget overflows reported by the encoder control path.
    pub n_bits_exceeded: i32,
    /// Number of active channels exposed through the public API.
    pub n_channels_api: i32,
    /// Number of internally active encoder channels.
    pub n_channels_internal: i32,
    /// Previous number of internal channels (used during transitions).
    pub n_prev_channels_internal: i32,
    /// Cooldown timer before another bandwidth switch is permitted.
    pub time_since_switch_allowed_ms: i32,
    /// Indicates whether bandwidth switching is currently allowed.
    pub allow_bandwidth_switch: bool,
    /// Remembers whether the decoder was forced to mid-only last frame.
    pub prev_decode_only_middle: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            state_fxx: from_fn(|_| EncoderChannelState::default()),
            stereo_state: StereoEncState::default(),
            n_bits_used_lbrr: 0,
            n_bits_exceeded: 0,
            n_channels_api: 1,
            n_channels_internal: 1,
            n_prev_channels_internal: 1,
            time_since_switch_allowed_ms: 0,
            allow_bandwidth_switch: false,
            prev_decode_only_middle: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_state_common_defaults_match_reference() {
        let common = EncoderStateCommon::default();
        let api_fs = DEFAULT_INTERNAL_FS_KHZ * 1000;
        let subfr_len = SUB_FRAME_LENGTH_MS * DEFAULT_INTERNAL_FS_KHZ as usize;
        let ltp_mem_len = LTP_MEM_LENGTH_MS * DEFAULT_INTERNAL_FS_KHZ as usize;
        assert_eq!(common.prev_signal_type, FrameSignalType::Inactive);
        assert!(!common.use_dtx);
        assert!(!common.use_cbr);
        assert!(!common.use_in_band_fec);
        assert_eq!(common.fs_khz, DEFAULT_INTERNAL_FS_KHZ);
        assert_eq!(common.nb_subfr, MAX_NB_SUBFR);
        assert_eq!(common.frame_length, DEFAULT_FRAME_LENGTH);
        assert_eq!(common.api_sample_rate_hz, api_fs);
        assert_eq!(common.prev_api_sample_rate_hz, api_fs);
        assert_eq!(common.max_internal_sample_rate_hz, api_fs);
        assert_eq!(common.min_internal_sample_rate_hz, api_fs);
        assert_eq!(common.desired_internal_sample_rate_hz, api_fs);
        assert!(!common.allow_bandwidth_switch);
        assert_eq!(common.n_channels_api, 1);
        assert_eq!(common.n_channels_internal, 1);
        assert_eq!(common.channel_nb, 0);
        assert_eq!(common.prev_lag, 0);
        assert_eq!(common.subfr_length, subfr_len);
        assert_eq!(common.ltp_mem_length, ltp_mem_len);
        assert_eq!(
            common.la_pitch,
            LA_PITCH_MS as i32 * DEFAULT_INTERNAL_FS_KHZ
        );
        assert_eq!(
            common.la_shape,
            LA_SHAPE_MS as i32 * DEFAULT_INTERNAL_FS_KHZ
        );
        assert_eq!(
            common.pitch_lpc_win_length,
            FIND_PITCH_LPC_WIN_MS * DEFAULT_INTERNAL_FS_KHZ as usize
        );
        assert_eq!(common.predict_lpc_order, MAX_LPC_ORDER);
        assert_eq!(common.ps_nlsf_cb, &SILK_NLSF_CB_WB);
        assert_eq!(common.pitch_contour_icdf, &PITCH_CONTOUR_ICDF);
        assert_eq!(common.pitch_lag_low_bits_icdf, &SILK_UNIFORM8_ICDF);
        assert_eq!(common.prev_nlsf_q15, [0; MAX_LPC_ORDER]);
        assert_eq!(common.target_rate_bps, 0);
        assert_eq!(common.snr_db_q7, 0);
        assert_eq!(common.input_quality_bands_q15, [0; VAD_N_BANDS]);
        assert_eq!(common.input_tilt_q15, 0);
        assert_eq!(common.speech_activity_q8, 0);
        assert_eq!(
            common.variable_hp_smth1_q15,
            lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8
        );
        assert_eq!(
            common.variable_hp_smth2_q15,
            lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8
        );
        assert_eq!(common.packet_size_ms, MAX_FRAME_LENGTH_MS as i32);
        assert_eq!(common.packet_loss_perc, 0);
        assert_eq!(common.n_frames_per_packet, 1);
        assert_eq!(common.n_frames_encoded, 0);
        assert_eq!(common.indices, SideInfoIndices::default());
        assert_eq!(common.input_buf_ix, 0);
        assert!(common.first_frame_after_reset);
        assert!(!common.controlled_since_last_payload);
        assert!(!common.prefill_flag);
        assert!(!common.lbrr_enabled);
        assert_eq!(common.lbrr_gain_increases, 0);
        assert_eq!(common.arch, 0);
    }

    #[test]
    fn encoder_channel_state_default_wraps_common() {
        let channel = EncoderChannelState::default();
        assert_eq!(*channel.common(), EncoderStateCommon::default());
        assert_eq!(channel.vad(), &VadState::default());
        assert_eq!(channel.low_pass_state(), &LpState::default());
        assert_eq!(channel.shape_state, EncoderShapeState::default());
        assert_eq!(channel.nsq_state, NoiseShapingQuantizerState::default());
        assert_eq!(channel.resampler_state.fs_in_khz(), 0);
        assert_eq!(channel.resampler_state.fs_out_khz(), 0);
        assert_eq!(channel.input_buf, [0; INPUT_BUFFER_LENGTH]);
        assert_eq!(channel.x_buf, [0; X_BUFFER_LENGTH]);
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

    #[test]
    fn encoder_super_state_defaults_cover_channels() {
        let encoder = Encoder::default();
        assert_eq!(encoder.state_fxx.len(), ENCODER_NUM_CHANNELS);
        assert!(
            encoder
                .state_fxx
                .iter()
                .all(|channel| *channel.common() == EncoderStateCommon::default())
        );
        assert_eq!(encoder.n_channels_api, 1);
        assert_eq!(encoder.n_channels_internal, 1);
        assert_eq!(encoder.n_prev_channels_internal, 1);
        assert_eq!(encoder.n_bits_used_lbrr, 0);
        assert_eq!(encoder.n_bits_exceeded, 0);
        assert_eq!(encoder.time_since_switch_allowed_ms, 0);
        assert!(!encoder.allow_bandwidth_switch);
        assert!(!encoder.prev_decode_only_middle);
    }
}
