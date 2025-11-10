//! Mirrors `silk/control_audio_bandwidth.c`.
//!
//! Determines the encoder's internal sampling rate (in kHz) while handling
//! transitions between narrowband, wideband, and super-wideband modes. The
//! routine enforces the same state machine as the C implementation so that the
//! bandwidth-switch permission handshake with the surrounding Opus encoder
//! behaves identically.

use crate::silk::EncControl;
use crate::silk::encoder::state::EncoderChannelState;
use crate::silk::lp_variable_cutoff::TRANSITION_FRAMES;

/// Mirrors `silk_control_audio_bandwidth`.
#[allow(clippy::arithmetic_side_effects)]
pub fn control_audio_bandwidth(
    encoder: &mut EncoderChannelState,
    enc_control: &mut EncControl,
) -> i32 {
    let (common, lp_state) = encoder.common_and_lp_mut();
    let mut orig_khz = common.fs_khz;
    if orig_khz == 0 {
        orig_khz = lp_state.saved_fs_khz;
    }
    let mut fs_khz = orig_khz;
    let mut fs_hz = fs_khz * 1000;

    if fs_hz == 0 {
        fs_hz = common
            .desired_internal_sample_rate_hz
            .min(common.api_sample_rate_hz);
        fs_khz = fs_hz / 1000;
    } else if fs_hz > common.api_sample_rate_hz
        || fs_hz > common.max_internal_sample_rate_hz
        || fs_hz < common.min_internal_sample_rate_hz
    {
        fs_hz = common
            .api_sample_rate_hz
            .min(common.max_internal_sample_rate_hz);
        fs_hz = fs_hz.max(common.min_internal_sample_rate_hz);
        fs_khz = fs_hz / 1000;
    } else {
        if lp_state.transition_frame_no >= TRANSITION_FRAMES {
            lp_state.mode = 0;
        }
        if common.allow_bandwidth_switch || enc_control.opus_can_switch {
            let orig_hz = orig_khz * 1000;
            if orig_hz > common.desired_internal_sample_rate_hz {
                if enc_control.opus_can_switch {
                    if lp_state.mode == 0 {
                        lp_state.transition_frame_no = TRANSITION_FRAMES;
                        lp_state.in_lp_state = [0; 2];
                    }
                    lp_state.mode = 0;
                    fs_khz = if orig_khz == 16 { 12 } else { 8 };
                } else if lp_state.transition_frame_no <= 0 {
                    enc_control.switch_ready = true;
                    reserve_frame_bits(enc_control);
                } else {
                    if lp_state.mode == 0 {
                        lp_state.transition_frame_no = TRANSITION_FRAMES;
                        lp_state.in_lp_state = [0; 2];
                    }
                    lp_state.mode = -2;
                }
            } else if orig_hz < common.desired_internal_sample_rate_hz {
                if enc_control.opus_can_switch {
                    fs_khz = if orig_khz == 8 { 12 } else { 16 };
                    lp_state.transition_frame_no = 0;
                    lp_state.in_lp_state = [0; 2];
                    lp_state.mode = 1;
                } else if lp_state.mode == 0 {
                    enc_control.switch_ready = true;
                    reserve_frame_bits(enc_control);
                } else {
                    lp_state.mode = 1;
                }
            } else if lp_state.mode < 0 {
                lp_state.mode = 1;
            }
        }
    }

    fs_khz
}

#[allow(clippy::arithmetic_side_effects)]
fn reserve_frame_bits(enc_control: &mut EncControl) {
    let denom = enc_control.payload_size_ms + 5;
    debug_assert!(denom != 0, "payload_size_ms + 5 must stay non-zero");
    if denom != 0 {
        enc_control.max_bits -= enc_control.max_bits * 5 / denom;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::encoder::state::EncoderChannelState;

    #[test]
    fn selects_initial_rate_when_uninitialized() {
        let mut encoder = EncoderChannelState::default();
        {
            let common = encoder.common_mut();
            common.fs_khz = 0;
            common.desired_internal_sample_rate_hz = 16_000;
            common.api_sample_rate_hz = 48_000;
        }
        let mut control = EncControl::default();
        let fs_khz = control_audio_bandwidth(&mut encoder, &mut control);
        assert_eq!(fs_khz, 16);
    }

    #[test]
    fn switches_down_immediately_when_allowed() {
        let mut encoder = EncoderChannelState::default();
        {
            let common = encoder.common_mut();
            common.fs_khz = 16;
            common.desired_internal_sample_rate_hz = 8_000;
            common.allow_bandwidth_switch = true;
        }
        let mut control = EncControl::default();
        control.opus_can_switch = true;

        let fs_khz = control_audio_bandwidth(&mut encoder, &mut control);
        assert_eq!(fs_khz, 12);
        let lp_state = encoder.low_pass_state();
        assert_eq!(lp_state.transition_frame_no, TRANSITION_FRAMES);
        assert_eq!(lp_state.mode, 0);
        assert_eq!(lp_state.in_lp_state, [0; 2]);
    }

    #[test]
    fn requests_down_switch_when_parent_cannot_switch() {
        let mut encoder = EncoderChannelState::default();
        {
            let common = encoder.common_mut();
            common.fs_khz = 16;
            common.desired_internal_sample_rate_hz = 8_000;
            common.allow_bandwidth_switch = true;
        }
        {
            let lp_state = encoder.low_pass_state_mut();
            lp_state.transition_frame_no = 0;
            lp_state.mode = 0;
        }
        let mut control = EncControl::default();
        control.opus_can_switch = false;
        control.max_bits = 1000;

        let fs_khz = control_audio_bandwidth(&mut encoder, &mut control);
        assert_eq!(fs_khz, 16);
        assert!(control.switch_ready);
        assert_eq!(control.max_bits, 800);
    }

    #[test]
    fn switches_up_when_allowed() {
        let mut encoder = EncoderChannelState::default();
        {
            let common = encoder.common_mut();
            common.fs_khz = 8;
            common.desired_internal_sample_rate_hz = 16_000;
            common.allow_bandwidth_switch = true;
        }
        let mut control = EncControl::default();
        control.opus_can_switch = true;

        let fs_khz = control_audio_bandwidth(&mut encoder, &mut control);
        assert_eq!(fs_khz, 16);
        let lp_state = encoder.low_pass_state();
        assert_eq!(lp_state.transition_frame_no, 0);
        assert_eq!(lp_state.mode, 0);
    }

    #[test]
    fn clamps_internal_rate_when_outside_limits() {
        let mut encoder = EncoderChannelState::default();
        {
            let common = encoder.common_mut();
            common.fs_khz = 16;
            common.api_sample_rate_hz = 12_000;
            common.max_internal_sample_rate_hz = 12_000;
            common.min_internal_sample_rate_hz = 8_000;
        }
        let mut control = EncControl::default();
        let fs_khz = control_audio_bandwidth(&mut encoder, &mut control);
        assert_eq!(fs_khz, 12);
    }
}
