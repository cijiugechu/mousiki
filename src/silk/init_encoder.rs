//! Port of `silk/init_encoder.c`.
//!
//! The reference helper clears the per-channel encoder state, re-primes the
//! adaptive high-pass smoother, and reinitialises the fixed-point VAD before
//! encoding resumes. This translation mirrors that behaviour for the current
//! Rust `EncoderChannelState`.

use crate::silk::encoder::state::EncoderChannelState;
use crate::silk::errors::SilkError;
use crate::silk::lin2log::lin2log;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;

/// Mirrors `silk_init_encoder`.
pub fn init_encoder(state: &mut EncoderChannelState, arch: i32) -> Result<(), SilkError> {
    *state = EncoderChannelState::default();

    let hp_log_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
    {
        let common = state.common_mut();
        common.arch = arch;
        common.variable_hp_smth1_q15 = hp_log_q15;
        common.variable_hp_smth2_q15 = hp_log_q15;
        common.first_frame_after_reset = true;
    }
    state.vad_mut().reset();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::init_encoder;
    use crate::silk::encoder::state::{EncoderChannelState, EncoderStateCommon, VadState};
    use crate::silk::lin2log::lin2log;
    use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;

    #[test]
    fn init_encoder_resets_channel_state() {
        let mut state = EncoderChannelState::default();
        {
            let common = state.common_mut();
            common.fs_khz = 24;
            common.variable_hp_smth1_q15 = 0;
            common.variable_hp_smth2_q15 = 0;
            common.first_frame_after_reset = false;
        }
        state.input_buf[0] = 123;
        state.vad.counter = -5;

        init_encoder(&mut state, 7).unwrap();

        let mut expected_common = EncoderStateCommon::default();
        expected_common.arch = 7;
        expected_common.variable_hp_smth1_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        expected_common.variable_hp_smth2_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        assert_eq!(state.common(), &expected_common);
        assert_eq!(state.vad(), &VadState::default());
        assert!(state.input_buf.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn init_encoder_updates_architecture_flag() {
        let mut state = EncoderChannelState::default();
        init_encoder(&mut state, 3).unwrap();
        assert_eq!(state.common().arch, 3);

        init_encoder(&mut state, 5).unwrap();
        assert_eq!(state.common().arch, 5);
    }
}
