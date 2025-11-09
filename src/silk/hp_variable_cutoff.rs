//! Adaptive high-pass cut-off helper from `silk/HP_variable_cutoff.c`.
//!
//! The original routine updates the smoothed logarithmic cut-off frequency that
//! feeds SILK's high-pass filter based on voiced pitch statistics. Only a tiny
//! subset of the encoder state is required, so this module focuses on the fields
//! that impact the cut-off estimation while keeping the API ergonomic for the
//! still-incomplete encoder port.

use crate::silk::FrameSignalType;
use crate::silk::encoder::state::EncoderChannelState;
use crate::silk::lin2log::lin2log;
use crate::silk::tuning_parameters::{
    VARIABLE_HP_MAX_CUTOFF_HZ, VARIABLE_HP_MAX_DELTA_FREQ, VARIABLE_HP_MIN_CUTOFF_HZ,
    VARIABLE_HP_SMTH_COEF1,
};

/// Q16 representation of [`VARIABLE_HP_MIN_CUTOFF_HZ`].
const VARIABLE_HP_MIN_CUTOFF_HZ_Q16: i32 = VARIABLE_HP_MIN_CUTOFF_HZ << 16;
/// Q7 representation of [`VARIABLE_HP_MAX_DELTA_FREQ`].
const VARIABLE_HP_MAX_DELTA_FREQ_Q7: i32 =
    (VARIABLE_HP_MAX_DELTA_FREQ * ((1 << 7) as f32) + 0.5) as i32;
/// Q16 representation of [`VARIABLE_HP_SMTH_COEF1`].
const VARIABLE_HP_SMTH_COEF1_Q16: i32 = (VARIABLE_HP_SMTH_COEF1 * ((1 << 16) as f32) + 0.5) as i32;

/// Update the variable high-pass cut-off state based on the previous voiced frame.
///
/// This mirrors the logic in `silk_HP_variable_cutoff` but focuses on the core
/// state used to derive the smoothed logarithmic cut-off bound. Only the first
/// channel's state is updated; multi-channel handling can be layered on top
/// later once the full encoder state is available.
pub fn hp_variable_cutoff(state: &mut EncoderChannelState) {
    let common = state.common_mut();
    if common.prev_signal_type != FrameSignalType::Voiced || common.prev_lag <= 0 {
        return;
    }

    let pitch_freq_hz_q16 =
        ((i64::from(common.fs_khz) * 1000_i64) << 16) / i64::from(common.prev_lag);
    let mut pitch_freq_log_q7 = lin2log(pitch_freq_hz_q16 as i32) - (16 << 7);

    let quality_q15 = common.input_quality_bands_q15[0];
    let min_cutoff_log_q7 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ_Q16) - (16 << 7);
    let quality_term = smulwb(-(quality_q15 << 2), quality_q15);
    pitch_freq_log_q7 = smlawb(
        pitch_freq_log_q7,
        quality_term,
        pitch_freq_log_q7 - min_cutoff_log_q7,
    );

    let mut delta_freq_q7 = pitch_freq_log_q7 - (common.variable_hp_smth1_q15 >> 8);
    if delta_freq_q7 < 0 {
        delta_freq_q7 = delta_freq_q7.wrapping_mul(3);
    }
    delta_freq_q7 = limit(
        delta_freq_q7,
        -VARIABLE_HP_MAX_DELTA_FREQ_Q7,
        VARIABLE_HP_MAX_DELTA_FREQ_Q7,
    );

    let speech_weight = smulbb(common.speech_activity_q8, delta_freq_q7);
    common.variable_hp_smth1_q15 = smlawb(
        common.variable_hp_smth1_q15,
        speech_weight,
        VARIABLE_HP_SMTH_COEF1_Q16,
    );

    let min_log_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
    let max_log_q15 = lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) << 8;
    common.variable_hp_smth1_q15 = limit(common.variable_hp_smth1_q15, min_log_q15, max_log_q15);
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwb(b, c))
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

fn limit(value: i32, min: i32, max: i32) -> i32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_non_voiced_frames() {
        let mut state = EncoderChannelState::default();
        let common = state.common_mut();
        common.prev_signal_type = FrameSignalType::Unvoiced;
        common.variable_hp_smth1_q15 = 1234;
        hp_variable_cutoff(&mut state);
        assert_eq!(state.common().variable_hp_smth1_q15, 1234);
    }

    #[test]
    fn tightens_cutoff_for_high_quality_voiced_frame() {
        let mut state = EncoderChannelState::default();
        {
            let common = state.common_mut();
            common.prev_signal_type = FrameSignalType::Voiced;
            common.prev_lag = 80;
            common.input_quality_bands_q15 = [28_000, 0, 0, 0];
            common.speech_activity_q8 = 180;
        }
        let before = state.common().variable_hp_smth1_q15;
        hp_variable_cutoff(&mut state);
        assert!(state.common().variable_hp_smth1_q15 > before);
        assert_eq!(state.common().variable_hp_smth1_q15, 194_454);
    }

    #[test]
    fn clamps_delta_when_pitch_jumps_down() {
        let mut state = EncoderChannelState::default();
        {
            let common = state.common_mut();
            common.prev_signal_type = FrameSignalType::Voiced;
            common.prev_lag = 400;
            common.variable_hp_smth1_q15 = lin2log(120) << 8;
            common.input_quality_bands_q15 = [31_000, 0, 0, 0];
            common.speech_activity_q8 = 200;
        }
        hp_variable_cutoff(&mut state);
        assert_eq!(
            state.common().variable_hp_smth1_q15,
            lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) << 8
        );
    }
}
