//! Runtime dispatch table mirroring `silk/arm/arm_silk_map.c`.
//!
//! The C implementation switches between scalar and NEON/dotprod encoder kernels at
//! runtime. CPU detection is not wired up in the Rust port yet, so each slot in these
//! tables points to the safe Rust implementation while preserving the
//! `arch & OPUS_ARCHMASK` indexing scheme.

use crate::celt::OPUS_ARCHMASK;
#[cfg(test)]
use crate::silk::biquad_alt::biquad_alt_stride2;
use crate::silk::biquad_alt_neon_intr::biquad_alt_stride2_neon;
use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::encoder::state::{EncoderStateCommon, NoiseShapingQuantizerState};
use crate::silk::lpc_inv_pred_gain::lpc_inverse_pred_gain;
use crate::silk::nsq::nsq_noise_shape_feedback_loop;
use crate::silk::nsq_del_dec::silk_nsq_del_dec;
use crate::silk::warped_autocorrelation::warped_autocorrelation;

const ARCH_IMPL_COUNT: usize = (OPUS_ARCHMASK as usize) + 1;

pub type SilkBiquadAltStride2Impl = fn(&[i16], &[i32; 3], &[i32; 2], &mut [i32; 4], &mut [i16]);
pub const SILK_BIQUAD_ALT_STRIDE2_IMPL: [SilkBiquadAltStride2Impl; ARCH_IMPL_COUNT] =
    [biquad_alt_stride2_neon; ARCH_IMPL_COUNT];

pub type SilkLpcInversePredGainImpl = fn(&[i16]) -> i32;
pub const SILK_LPC_INVERSE_PRED_GAIN_IMPL: [SilkLpcInversePredGainImpl; ARCH_IMPL_COUNT] =
    [lpc_inverse_pred_gain; ARCH_IMPL_COUNT];

#[allow(clippy::too_many_arguments)]
pub type SilkNsqDelDecImpl = fn(
    &EncoderStateCommon,
    &mut NoiseShapingQuantizerState,
    &mut SideInfoIndices,
    &[i16],
    &mut [i8],
    &[i16],
    &[i16],
    &[i16],
    &[i32],
    &[i32],
    &[i32],
    &[i32],
    &[i32],
    i32,
    i32,
);
pub const SILK_NSQ_DEL_DEC_IMPL: [SilkNsqDelDecImpl; ARCH_IMPL_COUNT] =
    [silk_nsq_del_dec; ARCH_IMPL_COUNT];

pub type SilkNsqNoiseShapeFeedbackLoopImpl = fn(i32, &mut [i32], &[i16], usize) -> i32;
pub const SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL: [SilkNsqNoiseShapeFeedbackLoopImpl;
    ARCH_IMPL_COUNT] = [nsq_noise_shape_feedback_loop; ARCH_IMPL_COUNT];

pub type SilkWarpedAutocorrelationFixImpl = fn(&mut [i32], &[i16], i32, usize) -> i32;
pub const SILK_WARPED_AUTOCORRELATION_FIX_IMPL: [SilkWarpedAutocorrelationFixImpl;
    ARCH_IMPL_COUNT] = [warped_autocorrelation; ARCH_IMPL_COUNT];

#[inline]
fn dispatch_index(arch: i32) -> usize {
    assert!(arch >= 0, "arch must be non-negative");
    assert_eq!(
        arch & OPUS_ARCHMASK,
        arch,
        "arch {arch} exceeds OPUS_ARCHMASK {OPUS_ARCHMASK}"
    );
    arch as usize
}

#[inline]
pub fn select_biquad_alt_stride2_impl(arch: i32) -> SilkBiquadAltStride2Impl {
    SILK_BIQUAD_ALT_STRIDE2_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_lpc_inverse_pred_gain_impl(arch: i32) -> SilkLpcInversePredGainImpl {
    SILK_LPC_INVERSE_PRED_GAIN_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_silk_nsq_del_dec_impl(arch: i32) -> SilkNsqDelDecImpl {
    SILK_NSQ_DEL_DEC_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_nsq_noise_shape_feedback_loop_impl(arch: i32) -> SilkNsqNoiseShapeFeedbackLoopImpl {
    SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_warped_autocorrelation_fix_impl(arch: i32) -> SilkWarpedAutocorrelationFixImpl {
    SILK_WARPED_AUTOCORRELATION_FIX_IMPL[dispatch_index(arch)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_index_matches_arch_mask() {
        assert_eq!(dispatch_index(0), 0);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn dispatch_index_rejects_invalid_arch() {
        let _ = select_biquad_alt_stride2_impl(OPUS_ARCHMASK + 1);
    }

    #[test]
    fn biquad_dispatch_matches_scalar() {
        let mut state = [2_345_678, -3_456_789, 4_567_890, -5_678_901];
        let mut expected_state = state;
        let b = [1_145_324_612, -229_064_922, 1_145_324_612];
        let a = [-1_010_580_540, 505_290_270];
        let input = [1357, -2468, 3579, -4680, 5791, -6802, 7913, -8024];
        let mut output = [0i16; 8];
        let mut expected_output = output;

        let fn_ptr = select_biquad_alt_stride2_impl(0);
        fn_ptr(&input, &b, &a, &mut state, &mut output);

        biquad_alt_stride2(&input, &b, &a, &mut expected_state, &mut expected_output);

        assert_eq!(output, expected_output);
        assert_eq!(state, expected_state);
    }

    #[test]
    fn lpc_inverse_pred_gain_dispatch_matches_scalar() {
        let coeffs = [
            290, -1100, 845, -451, 712, -301, 210, -77, 33, -12, 6, -3, 1, -1, 0, 0,
        ];
        let fn_ptr = select_lpc_inverse_pred_gain_impl(0);
        assert_eq!(fn_ptr(&coeffs), lpc_inverse_pred_gain(&coeffs));
    }

    #[test]
    fn nsq_feedback_dispatch_matches_scalar() {
        let mut ar2_q14 = [0i32; 10];
        let mut expected_ar2 = ar2_q14;
        let coef_q13 = [12i16, -34, 56, -78, 90, -12, 34, -56, 78, -90];
        let fn_ptr = select_nsq_noise_shape_feedback_loop_impl(0);
        let arch_value = fn_ptr(1234, &mut ar2_q14, &coef_q13, coef_q13.len());
        let scalar_value =
            nsq_noise_shape_feedback_loop(1234, &mut expected_ar2, &coef_q13, coef_q13.len());

        assert_eq!(arch_value, scalar_value);
        assert_eq!(ar2_q14, expected_ar2);
    }

    #[test]
    fn warped_autocorr_dispatch_matches_scalar() {
        let mut corr = [0i32; 5];
        let mut expected_corr = corr;
        let input = [1i16, -2, 3, -4, 5, -6];
        let warping_q16 = 18_000;

        let fn_ptr = select_warped_autocorrelation_fix_impl(0);
        let scale = fn_ptr(&mut corr, &input, warping_q16, 4);
        let expected_scale = warped_autocorrelation(&mut expected_corr, &input, warping_q16, 4);

        assert_eq!(scale, expected_scale);
        assert_eq!(corr, expected_corr);
    }
}
