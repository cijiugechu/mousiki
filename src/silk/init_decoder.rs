//! Port of `silk/init_decoder.c`.
//!
//! The reference implementation exposes two helpers:
//! - `silk_reset_decoder` clears the per-channel decoder state while preserving
//!   the architecture-specific capability flags.
//! - `silk_init_decoder` fully reinitialises a channel by zeroing the struct
//!   before delegating to `silk_reset_decoder`.
//!
//! These wrappers keep the Rust decoder state in sync with the behaviour
//! expected by the C API entry points.

use crate::celt::opus_select_arch;
use crate::silk::decoder_state::DecoderState;
use crate::silk::errors::SilkError;

/// Mirrors `silk_reset_decoder`.
pub fn reset_decoder(state: &mut DecoderState) -> Result<(), SilkError> {
    *state = DecoderState::default();

    let lpc_order = state.sample_rate.lpc_order;
    state.cng_state.reset(lpc_order);
    state.plc_state.reset(state.sample_rate.frame_length);
    state.arch = opus_select_arch();

    Ok(())
}

/// Mirrors `silk_init_decoder`.
pub fn init_decoder(state: &mut DecoderState) -> Result<(), SilkError> {
    reset_decoder(state)
}

#[cfg(test)]
mod tests {
    use super::{init_decoder, reset_decoder};
    use crate::celt::opus_select_arch;
    use crate::silk::decoder_state::DecoderState;

    #[test]
    fn reset_decoder_initialises_architecture() {
        let mut state = DecoderState::default();
        reset_decoder(&mut state).unwrap();
        assert_eq!(state.arch, opus_select_arch());
    }

    #[test]
    fn init_decoder_is_alias_for_reset() {
        let mut state = DecoderState::default();
        init_decoder(&mut state).unwrap();
        assert!(
            state.cng_state.smoothed_nlsf_q15()[0] > 0,
            "CNG reset should prime the smoothed NLSF grid"
        );
        assert_eq!(state.plc_state.prev_gain_q16, [1 << 16; 2]);
    }
}
