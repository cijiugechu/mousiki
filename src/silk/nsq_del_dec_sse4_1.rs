//! SSE4.1 entry point for the delayed-decision noise-shaping quantiser.
//!
//! The C implementation in `silk/x86/NSQ_del_dec_sse4_1.c` wires an SSE4.1
//! fast path into the x86 runtime dispatch table. Runtime CPU detection is
//! still stubbed (`OPUS_ARCHMASK` is zero), so this Rust port preserves the
//! entry point while delegating to the safe scalar [`silk_nsq_del_dec`]
//! helper.

use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::encoder::state::{EncoderStateCommon, NoiseShapingQuantizerState};
use crate::silk::nsq_del_dec::silk_nsq_del_dec;

/// Mirrors `silk_NSQ_del_dec_sse4_1` while deferring to the scalar
/// implementation until SIMD dispatch is enabled.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn silk_nsq_del_dec_sse4_1(
    encoder: &EncoderStateCommon,
    nsq: &mut NoiseShapingQuantizerState,
    indices: &mut SideInfoIndices,
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
    silk_nsq_del_dec(
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
    use super::silk_nsq_del_dec_sse4_1;
    use crate::silk::decode_indices::SideInfoIndices;
    use crate::silk::encoder::state::{EncoderStateCommon, NoiseShapingQuantizerState};
    use crate::silk::nsq_del_dec::silk_nsq_del_dec;
    use crate::silk::vq_wmat_ec::LTP_ORDER;
    use crate::silk::{
        FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER,
    };
    use alloc::vec;

    #[test]
    fn forwards_to_scalar_path() {
        let mut encoder = EncoderStateCommon::default();
        encoder.n_states_delayed_decision = 1;
        encoder.shaping_lpc_order = MAX_SHAPE_LPC_ORDER as i32;

        let frame_len = encoder.frame_length;
        let mut x16 = vec![0i16; frame_len];
        for (i, sample) in x16.iter_mut().enumerate() {
            *sample = (i as i16).wrapping_mul(3).wrapping_sub(100);
        }

        let mut pulses_scalar = vec![0i8; frame_len];
        let mut pulses_sse = vec![0i8; frame_len];

        let mut nsq_scalar = NoiseShapingQuantizerState::default();
        let mut nsq_sse = nsq_scalar.clone();

        let mut indices_scalar = SideInfoIndices::default();
        indices_scalar.signal_type = FrameSignalType::Unvoiced;
        let mut indices_sse = SideInfoIndices::default();
        indices_sse.signal_type = FrameSignalType::Unvoiced;

        let pred_coef_q12 = vec![0i16; MAX_NB_SUBFR * MAX_LPC_ORDER];
        let ltp_coef_q14 = vec![0i16; LTP_ORDER * MAX_NB_SUBFR];
        let ar_q13 = vec![0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
        let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
        let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
        let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
        let gains_q16 = vec![1 << 16; MAX_NB_SUBFR];
        let pitch_l = vec![0i32; MAX_NB_SUBFR];
        let lambda_q10 = 42;
        let ltp_scale_q14 = 1 << 14;

        silk_nsq_del_dec(
            &encoder,
            &mut nsq_scalar,
            &mut indices_scalar,
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

        silk_nsq_del_dec_sse4_1(
            &encoder,
            &mut nsq_sse,
            &mut indices_sse,
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
        assert_eq!(indices_scalar.gains_indices, indices_sse.gains_indices);
        assert_eq!(indices_scalar.ltp_index, indices_sse.ltp_index);
        assert_eq!(indices_scalar.nlsf_indices, indices_sse.nlsf_indices);
        assert_eq!(indices_scalar.lag_index, indices_sse.lag_index);
        assert_eq!(indices_scalar.contour_index, indices_sse.contour_index);
        assert_eq!(indices_scalar.signal_type, indices_sse.signal_type);
        assert_eq!(indices_scalar.quant_offset_type, indices_sse.quant_offset_type);
        assert_eq!(indices_scalar.nlsf_interp_coef_q2, indices_sse.nlsf_interp_coef_q2);
        assert_eq!(indices_scalar.per_index, indices_sse.per_index);
        assert_eq!(indices_scalar.ltp_scale_index, indices_sse.ltp_scale_index);
        assert_eq!(indices_scalar.seed, indices_sse.seed);
    }
}
