//! SSE4.1 entry point for the fixed-point noise-shaping quantiser.
//!
//! The C implementation in `silk/x86/NSQ_sse4_1.c` wires this fast path into
//! the x86 runtime dispatch table. Runtime CPU detection is still stubbed in
//! the Rust port, so this shim preserves the entry point while delegating to
//! the safe scalar [`silk_nsq`] implementation.

use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::encoder::state::{EncoderStateCommon, NoiseShapingQuantizerState};
use crate::silk::nsq::silk_nsq;

/// Mirrors `silk_NSQ_sse4_1` while deferring to the scalar implementation
/// until SIMD dispatch is enabled.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn silk_nsq_sse4_1(
    encoder: &EncoderStateCommon,
    nsq: &mut NoiseShapingQuantizerState,
    indices: &SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    silk_nsq(
        encoder,
        nsq,
        indices,
        x16,
        pulses,
        pred_coef_q12,
        ltp_coef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
    )
}

#[cfg(test)]
mod tests {
    use super::silk_nsq_sse4_1;
    use crate::silk::decode_indices::SideInfoIndices;
    use crate::silk::encoder::state::{EncoderStateCommon, NoiseShapingQuantizerState};
    use crate::silk::nsq::silk_nsq;
    use crate::silk::vq_wmat_ec::LTP_ORDER;
    use crate::silk::{
        FrameQuantizationOffsetType, FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR,
        MAX_SHAPE_LPC_ORDER,
    };
    use alloc::vec;

    #[test]
    fn forwards_to_scalar_path() {
        let mut encoder = EncoderStateCommon::default();
        encoder.shaping_lpc_order = MAX_SHAPE_LPC_ORDER as i32;

        let frame_len = encoder.frame_length;
        let mut x16 = vec![0i16; frame_len];
        for (i, sample) in x16.iter_mut().enumerate() {
            *sample = (i as i16).wrapping_mul(5).wrapping_sub(75);
        }

        let mut pulses_scalar = vec![0i8; frame_len];
        let mut pulses_sse = vec![0i8; frame_len];

        let mut nsq_scalar = NoiseShapingQuantizerState::default();
        let mut nsq_sse = nsq_scalar.clone();

        let mut indices = SideInfoIndices::default();
        indices.signal_type = FrameSignalType::Voiced;
        indices.quant_offset_type = FrameQuantizationOffsetType::High;
        indices.seed = 11;

        let pred_coef_q12 = vec![0i16; MAX_NB_SUBFR * MAX_LPC_ORDER];
        let ltp_coef_q14 = vec![0i16; LTP_ORDER * MAX_NB_SUBFR];
        let ar_q13 = vec![0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
        let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
        let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
        let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
        let gains_q16 = vec![1 << 16; MAX_NB_SUBFR];
        let pitch_l = vec![60; MAX_NB_SUBFR];
        let lambda_q10 = 1024;
        let ltp_scale_q14 = 1 << 14;

        silk_nsq(
            &encoder,
            &mut nsq_scalar,
            &indices,
            &x16,
            &mut pulses_scalar,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );

        silk_nsq_sse4_1(
            &encoder,
            &mut nsq_sse,
            &indices,
            &x16,
            &mut pulses_sse,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );

        assert_eq!(pulses_scalar, pulses_sse);
        assert_eq!(nsq_scalar, nsq_sse);
    }
}
