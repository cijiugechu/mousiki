//! SSE4.1 voice-activity detector entry point mirroring `silk/x86/VAD_sse4_1.c`.
//!
//! Runtime CPU detection is not wired up yet, so this shim delegates to the
//! scalar VAD helper until x86 dispatch selects the SIMD fast path.

use crate::silk::encoder::state::EncoderStateCommon;
use crate::silk::vad::compute_speech_activity_q8_common;

/// Mirrors `silk_VAD_GetSA_Q8_sse4_1`.
#[inline]
pub fn silk_vad_get_sa_q8_sse4_1(common: &mut EncoderStateCommon, input: &[i16]) -> u8 {
    compute_speech_activity_q8_common(common, input)
}

#[cfg(test)]
mod tests {
    use super::silk_vad_get_sa_q8_sse4_1;
    use crate::silk::encoder::state::EncoderStateCommon;
    use crate::silk::vad::compute_speech_activity_q8_common;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn matches_scalar_on_silence() {
        let mut common = EncoderStateCommon::default();
        let input = vec![0i16; common.frame_length];
        let mut common_clone = common.clone();

        assert_eq!(
            silk_vad_get_sa_q8_sse4_1(&mut common, &input),
            compute_speech_activity_q8_common(&mut common_clone, &input)
        );
    }

    #[test]
    fn matches_scalar_on_ramp() {
        let mut common = EncoderStateCommon::default();
        let input: Vec<i16> = (0..common.frame_length)
            .map(|i| (i as i16).wrapping_mul(3))
            .collect();
        let mut common_clone = common.clone();

        assert_eq!(
            silk_vad_get_sa_q8_sse4_1(&mut common, &input),
            compute_speech_activity_q8_common(&mut common_clone, &input)
        );
    }
}
