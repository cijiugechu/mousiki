//! Encoder tuning parameters shared across SILK signal-processing helpers.
//!
//! Ported from `silk/tuning_parameters.h` in the reference Opus implementation.

/// Decay time for the encoder bit-reservoir, in milliseconds.
pub const BITRESERVOIR_DECAY_TIME_MS: i32 = 500;

/// Level of the noise floor for the whitening filter LPC analysis in pitch analysis.
pub const FIND_PITCH_WHITE_NOISE_FRACTION: f32 = 1e-3;

/// Bandwidth expansion for the whitening filter in pitch analysis.
pub const FIND_PITCH_BANDWIDTH_EXPANSION: f32 = 0.99;

/// LPC analysis regularisation factor.
pub const FIND_LPC_COND_FAC: f32 = 1e-5;

/// Maximum cumulative long-term prediction gain in dB.
pub const MAX_SUM_LOG_GAIN_DB: f32 = 250.0;

/// Reciprocal of the maximum correlation used in the LTP analysis.
pub const LTP_CORR_INV_MAX: f32 = 0.03;

/// First-order smoothing coefficient for low-end pitch frequency estimation.
pub const VARIABLE_HP_SMTH_COEF1: f32 = 0.1;

/// Second-order smoothing coefficient for low-end pitch frequency estimation.
pub const VARIABLE_HP_SMTH_COEF2: f32 = 0.015;

/// Maximum allowed delta in the log-domain pitch frequency smoother.
pub const VARIABLE_HP_MAX_DELTA_FREQ: f32 = 0.4;

/// Minimum cutoff frequency for the adaptive high-pass filter in Hz.
pub const VARIABLE_HP_MIN_CUTOFF_HZ: i32 = 60;

/// Maximum cutoff frequency for the adaptive high-pass filter in Hz.
pub const VARIABLE_HP_MAX_CUTOFF_HZ: i32 = 100;

/// Voice activity detection threshold.
pub const SPEECH_ACTIVITY_DTX_THRES: f32 = 0.05;
/// Number of speech frames required before entering discontinuous transmission.
pub const NB_SPEECH_FRAMES_BEFORE_DTX: i32 = 10;
/// Maximum number of consecutive DTX frames.
pub const MAX_CONSECUTIVE_DTX: i32 = 20;
/// External VAD flag indicating inactivity.
pub const VAD_NO_ACTIVITY: i32 = 0;

/// Speech activity threshold for enabling low bit-rate redundancy (LBRR).
pub const LBRR_SPEECH_ACTIVITY_THRES: f32 = 0.3;

/// Reduction in coding SNR during low speech activity, in dB.
pub const BG_SNR_DECR_DB: f32 = 2.0;

/// Factor for reducing quantisation noise during voiced speech, in dB.
pub const HARM_SNR_INCR_DB: f32 = 2.0;

/// Factor for reducing quantisation noise for unvoiced sparse signals, in dB.
pub const SPARSE_SNR_INCR_DB: f32 = 2.0;

/// Threshold for sparseness measurement controlling quantisation offset during unvoiced frames.
pub const ENERGY_VARIATION_THRESHOLD_QNT_OFFSET: f32 = 0.6;

/// Warping control factor.
pub const WARPING_MULTIPLIER: f32 = 0.015;

/// Fraction added to the first autocorrelation value in noise-shaping.
pub const SHAPE_WHITE_NOISE_FRACTION: f32 = 3e-5;

/// Noise-shaping filter chirp factor.
pub const BANDWIDTH_EXPANSION: f32 = 0.94;

/// Harmonic noise-shaping contribution.
pub const HARMONIC_SHAPING: f32 = 0.3;

/// Additional harmonic noise-shaping for high bit-rates or noisy input.
pub const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING: f32 = 0.2;

/// Parameter for shaping noise towards higher frequencies.
pub const HP_NOISE_COEF: f32 = 0.25;

/// Parameter for emphasising high-frequency noise during voiced speech.
pub const HARM_HP_NOISE_COEF: f32 = 0.35;

/// Parameter for applying a high-pass tilt to the input signal.
pub const INPUT_TILT: f32 = 0.05;

/// Parameter for extra high-pass tilt to the input signal at high bit-rates.
pub const HIGH_RATE_INPUT_TILT: f32 = 0.1;

/// Parameter for reducing noise at very low frequencies.
pub const LOW_FREQ_SHAPING: f32 = 4.0;

/// Reduction amount applied to low-frequency shaping for low-quality, low-frequency signals.
pub const LOW_QUALITY_LOW_FREQ_SHAPING_DECR: f32 = 0.5;

/// Subframe smoothing coefficient for HarmBoost, HarmShapeGain, and Tilt controls.
pub const SUBFR_SMTH_COEF: f32 = 0.4;

/// Base offset for the residual quantiser rate/distortion trade-off.
pub const LAMBDA_OFFSET: f32 = 1.2;

/// Speech-activity component of the residual quantiser rate/distortion trade-off.
pub const LAMBDA_SPEECH_ACT: f32 = -0.2;

/// Penalty for delayed decisions in the residual quantiser rate/distortion trade-off.
pub const LAMBDA_DELAYED_DECISIONS: f32 = -0.05;

/// Input quality component of the residual quantiser rate/distortion trade-off.
pub const LAMBDA_INPUT_QUALITY: f32 = -0.1;

/// Coding quality component of the residual quantiser rate/distortion trade-off.
pub const LAMBDA_CODING_QUALITY: f32 = -0.2;

/// Quantisation offset component of the residual quantiser rate/distortion trade-off.
pub const LAMBDA_QUANT_OFFSET: f32 = 0.8;

/// Compensation factor in bitrate calculations for 10 ms modes (in bits per second).
pub const REDUCE_BITRATE_10_MS_BPS: i32 = 2200;

/// Maximum time before allowing a bandwidth transition, in milliseconds.
pub const MAX_BANDWIDTH_SWITCH_DELAY_MS: i32 = 5000;
