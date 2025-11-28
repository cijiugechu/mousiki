//! Runtime dispatch table mirroring `silk/x86/x86_silk_map.c`.
//!
//! The C implementation uses this table to switch between the scalar and
//! architecture-optimised encoder kernels. Runtime CPU detection is not wired
//! up in the Rust port yet, so each entry maps to the safe Rust implementation
//! while still honouring the `arch & OPUS_ARCHMASK` indexing scheme.

use crate::celt::OPUS_ARCHMASK;
use crate::silk::burg_modified::silk_burg_modified;
use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::encoder::state::{EncoderStateCommon, NoiseShapingQuantizerState};
use crate::silk::inner_product_flp_avx2::inner_product_flp_avx2;
use crate::silk::nsq::silk_nsq;
use crate::silk::nsq_del_dec::silk_nsq_del_dec;
use crate::silk::vad_sse4_1::silk_vad_get_sa_q8_sse4_1;
use crate::silk::vector_ops_fix_sse4_1::inner_prod16_sse4_1;
use crate::silk::vq_wmat_ec::{VqWMatEcResult, LTP_ORDER};
use crate::silk::vq_wmat_ec_sse4_1::vq_wmat_ec_sse4_1;
#[cfg(test)]
use crate::silk::vq_wmat_ec::vq_wmat_ec;

const ARCH_IMPL_COUNT: usize = (OPUS_ARCHMASK as usize) + 1;

pub type SilkInnerProd16Impl = fn(&[i16], &[i16]) -> i64;
pub const SILK_INNER_PROD16_IMPL: [SilkInnerProd16Impl; ARCH_IMPL_COUNT] =
    [inner_prod16_sse4_1; ARCH_IMPL_COUNT];

pub type SilkVadGetSaQ8Impl = fn(&mut EncoderStateCommon, &[i16]) -> u8;
pub const SILK_VAD_GETSA_Q8_IMPL: [SilkVadGetSaQ8Impl; ARCH_IMPL_COUNT] =
    [silk_vad_get_sa_q8_sse4_1; ARCH_IMPL_COUNT];

pub type SilkNsqImpl = fn(
    &EncoderStateCommon,
    &mut NoiseShapingQuantizerState,
    &SideInfoIndices,
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
pub const SILK_NSQ_IMPL: [SilkNsqImpl; ARCH_IMPL_COUNT] = [silk_nsq; ARCH_IMPL_COUNT];

pub type SilkVqWMatEcImpl = fn(
    &[i32; LTP_ORDER * LTP_ORDER],
    &[i32; LTP_ORDER],
    &[[i8; LTP_ORDER]],
    &[u8],
    &[u8],
    i32,
    i32,
) -> VqWMatEcResult;
pub const SILK_VQ_WMAT_EC_IMPL: [SilkVqWMatEcImpl; ARCH_IMPL_COUNT] =
    [vq_wmat_ec_sse4_1; ARCH_IMPL_COUNT];

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

pub type SilkBurgModifiedImpl = fn(
    &mut i32,
    &mut i32,
    &mut [i32],
    &[i16],
    i32,
    usize,
    usize,
    usize,
    i32,
);
pub const SILK_BURG_MODIFIED_IMPL: [SilkBurgModifiedImpl; ARCH_IMPL_COUNT] =
    [silk_burg_modified; ARCH_IMPL_COUNT];

pub type SilkInnerProductFlpImpl = fn(&[f32], &[f32]) -> f64;
pub const SILK_INNER_PRODUCT_FLP_IMPL: [SilkInnerProductFlpImpl; ARCH_IMPL_COUNT] =
    [inner_product_flp_avx2; ARCH_IMPL_COUNT];

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
pub fn select_inner_prod16_impl(arch: i32) -> SilkInnerProd16Impl {
    SILK_INNER_PROD16_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_vad_get_sa_q8_impl(arch: i32) -> SilkVadGetSaQ8Impl {
    SILK_VAD_GETSA_Q8_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_silk_nsq_impl(arch: i32) -> SilkNsqImpl {
    SILK_NSQ_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_vq_wmat_ec_impl(arch: i32) -> SilkVqWMatEcImpl {
    SILK_VQ_WMAT_EC_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_silk_nsq_del_dec_impl(arch: i32) -> SilkNsqDelDecImpl {
    SILK_NSQ_DEL_DEC_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_silk_burg_modified_impl(arch: i32) -> SilkBurgModifiedImpl {
    SILK_BURG_MODIFIED_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_inner_product_flp_impl(arch: i32) -> SilkInnerProductFlpImpl {
    SILK_INNER_PRODUCT_FLP_IMPL[dispatch_index(arch)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::encoder::state::EncoderStateCommon;
    use crate::silk::inner_product_flp::inner_product_flp;
    use crate::silk::vector_ops::inner_prod16;
    use crate::silk::vad::compute_speech_activity_q8_common;
    use alloc::vec;

    #[test]
    fn dispatch_index_matches_arch_mask() {
        assert_eq!(dispatch_index(0), 0);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn dispatch_index_rejects_invalid_arch() {
        let _ = select_inner_prod16_impl(OPUS_ARCHMASK + 1);
    }

    #[test]
    fn inner_prod16_dispatch_matches_scalar() {
        let fn_ptr = select_inner_prod16_impl(0);
        let a = [1, -2, 3, -4];
        let b = [4, 3, -2, -1];
        assert_eq!(fn_ptr(&a, &b), inner_prod16(&a, &b));
    }

    #[test]
    fn vad_dispatch_matches_scalar() {
        let mut common = EncoderStateCommon::default();
        let input = vec![0i16; common.frame_length];
        let fn_ptr = select_vad_get_sa_q8_impl(0);
        assert_eq!(
            fn_ptr(&mut common, &input),
            compute_speech_activity_q8_common(&mut common, &input)
        );
    }

    #[test]
    fn vq_dispatch_matches_scalar() {
        let xx_q17 = [0; LTP_ORDER * LTP_ORDER];
        let x_x_q17 = [0; LTP_ORDER];
        let cb_q7 = vec![[0; LTP_ORDER]; 1];
        let cb_gain_q7 = [0u8; 1];
        let cl_q5 = [0u8; 1];
        let subfr_len = 20;
        let max_gain_q7 = 0;
        let fn_ptr = select_vq_wmat_ec_impl(0);
        assert_eq!(
            fn_ptr(
                &xx_q17,
                &x_x_q17,
                &cb_q7,
                &cb_gain_q7,
                &cl_q5,
                subfr_len,
                max_gain_q7
            ),
            vq_wmat_ec(
                &xx_q17,
                &x_x_q17,
                &cb_q7,
                &cb_gain_q7,
                &cl_q5,
                subfr_len,
                max_gain_q7
            )
        );
    }

    #[test]
    fn flp_dispatch_matches_scalar() {
        let fn_ptr = select_inner_product_flp_impl(0);
        let a = [0.5f32, -1.0, 2.0];
        let b = [1.0f32, 0.25, -0.5];
        assert_eq!(fn_ptr(&a, &b), inner_product_flp(&a, &b));
    }

    #[test]
    fn burg_dispatch_matches_scalar() {
        let order = 2;
        let subfr_length = 4;
        let nb_subfr = 1;
        let min_inv_gain_q30 = 1 << 26;
        let x = [100, -200, 50, -25];
        let mut res_direct = 0;
        let mut res_direct_q = 0;
        let mut res_dispatch = 0;
        let mut res_dispatch_q = 0;
        let mut a_direct = [0i32; 2];
        let mut a_dispatch = [0i32; 2];

        silk_burg_modified(
            &mut res_direct,
            &mut res_direct_q,
            &mut a_direct,
            &x,
            min_inv_gain_q30,
            subfr_length,
            nb_subfr,
            order,
            0,
        );
        select_silk_burg_modified_impl(0)(
            &mut res_dispatch,
            &mut res_dispatch_q,
            &mut a_dispatch,
            &x,
            min_inv_gain_q30,
            subfr_length,
            nb_subfr,
            order,
            0,
        );

        assert_eq!(res_direct, res_dispatch);
        assert_eq!(res_direct_q, res_dispatch_q);
        assert_eq!(a_direct, a_dispatch);
    }

    #[test]
    fn tables_have_expected_size() {
        assert_eq!(SILK_INNER_PROD16_IMPL.len(), ARCH_IMPL_COUNT);
        assert_eq!(SILK_VAD_GETSA_Q8_IMPL.len(), ARCH_IMPL_COUNT);
        assert_eq!(SILK_NSQ_IMPL.len(), ARCH_IMPL_COUNT);
        assert_eq!(SILK_VQ_WMAT_EC_IMPL.len(), ARCH_IMPL_COUNT);
        assert_eq!(SILK_NSQ_DEL_DEC_IMPL.len(), ARCH_IMPL_COUNT);
        assert_eq!(SILK_BURG_MODIFIED_IMPL.len(), ARCH_IMPL_COUNT);
        assert_eq!(SILK_INNER_PRODUCT_FLP_IMPL.len(), ARCH_IMPL_COUNT);
    }
}
