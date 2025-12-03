#![allow(dead_code)]

//! Runtime dispatch table mirroring `celt/arm/arm_celt_map.c`.
//!
//! The reference ARM map wires NEON/NE10 FFT and MDCT implementations alongside
//! the scalar fallbacks. Runtime dispatch is not enabled in the Rust port yet,
//! so these tables all resolve to the scalar helpers while keeping the
//! architecture-indexed layout intact.

use crate::celt::cpu_support::OPUS_ARCHMASK;
use crate::celt::{opus_fft, opus_ifft, KissFftCpx, KissFftState};
use crate::celt::math::{celt_float2int16, opus_limit2_checkwithin1};
use crate::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
use crate::celt::pitch::{celt_inner_prod, celt_pitch_xcorr, dual_inner_prod, xcorr_kernel};
use crate::celt::types::{CeltCoef, MdctLookup, OpusVal16, OpusVal32};

const ARCH_IMPL_COUNT: usize = (OPUS_ARCHMASK as usize) + 1;

pub type CeltFloat2Int16Impl = fn(&[f32], &mut [i16]);
pub const CELT_FLOAT2INT16_IMPL: [CeltFloat2Int16Impl; ARCH_IMPL_COUNT] =
    [celt_float2int16; ARCH_IMPL_COUNT];

pub type OpusLimit2CheckWithin1Impl = fn(&mut [f32]) -> bool;
pub const OPUS_LIMIT2_CHECKWITHIN1_IMPL: [OpusLimit2CheckWithin1Impl; ARCH_IMPL_COUNT] =
    [opus_limit2_checkwithin1; ARCH_IMPL_COUNT];

pub type CeltInnerProdImpl = fn(&[OpusVal16], &[OpusVal16]) -> OpusVal32;
pub const CELT_INNER_PROD_IMPL: [CeltInnerProdImpl; ARCH_IMPL_COUNT] =
    [celt_inner_prod; ARCH_IMPL_COUNT];

pub type DualInnerProdImpl =
    fn(&[OpusVal16], &[OpusVal16], &[OpusVal16]) -> (OpusVal32, OpusVal32);
pub const DUAL_INNER_PROD_IMPL: [DualInnerProdImpl; ARCH_IMPL_COUNT] =
    [dual_inner_prod; ARCH_IMPL_COUNT];

pub type CeltPitchXcorrImpl =
    fn(&[OpusVal16], &[OpusVal16], usize, usize, &mut [OpusVal32]);
pub const CELT_PITCH_XCORR_IMPL: [CeltPitchXcorrImpl; ARCH_IMPL_COUNT] =
    [celt_pitch_xcorr; ARCH_IMPL_COUNT];

pub type XcorrKernelImpl = fn(&[OpusVal16], &[OpusVal16], &mut [OpusVal32; 4], usize);
pub const XCORR_KERNEL_IMPL: [XcorrKernelImpl; ARCH_IMPL_COUNT] =
    [xcorr_kernel; ARCH_IMPL_COUNT];

fn opus_fft_alloc_arch_stub(_state: &mut KissFftState) -> i32 {
    0
}

fn opus_fft_free_arch_stub(_state: &mut KissFftState) {}

pub type OpusFftAllocArchImpl = fn(&mut KissFftState) -> i32;
pub const OPUS_FFT_ALLOC_ARCH_IMPL: [OpusFftAllocArchImpl; ARCH_IMPL_COUNT] =
    [opus_fft_alloc_arch_stub; ARCH_IMPL_COUNT];

pub type OpusFftFreeArchImpl = fn(&mut KissFftState);
pub const OPUS_FFT_FREE_ARCH_IMPL: [OpusFftFreeArchImpl; ARCH_IMPL_COUNT] =
    [opus_fft_free_arch_stub; ARCH_IMPL_COUNT];

pub type OpusFftImpl = fn(&KissFftState, &[KissFftCpx], &mut [KissFftCpx]);
pub const OPUS_FFT: [OpusFftImpl; ARCH_IMPL_COUNT] = [opus_fft; ARCH_IMPL_COUNT];
pub const OPUS_IFFT: [OpusFftImpl; ARCH_IMPL_COUNT] = [opus_ifft; ARCH_IMPL_COUNT];

pub type CltMdctForwardImpl = fn(
    &MdctLookup,
    &[f32],
    &mut [f32],
    &[CeltCoef],
    usize,
    usize,
    usize,
);
pub const CLT_MDCT_FORWARD_IMPL: [CltMdctForwardImpl; ARCH_IMPL_COUNT] =
    [clt_mdct_forward; ARCH_IMPL_COUNT];

pub type CltMdctBackwardImpl = fn(
    &MdctLookup,
    &[f32],
    &mut [f32],
    &[CeltCoef],
    usize,
    usize,
    usize,
);
pub const CLT_MDCT_BACKWARD_IMPL: [CltMdctBackwardImpl; ARCH_IMPL_COUNT] =
    [clt_mdct_backward; ARCH_IMPL_COUNT];

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
pub fn select_celt_float2int16_impl(arch: i32) -> CeltFloat2Int16Impl {
    CELT_FLOAT2INT16_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_opus_limit2_checkwithin1_impl(arch: i32) -> OpusLimit2CheckWithin1Impl {
    OPUS_LIMIT2_CHECKWITHIN1_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_celt_inner_prod_impl(arch: i32) -> CeltInnerProdImpl {
    CELT_INNER_PROD_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_dual_inner_prod_impl(arch: i32) -> DualInnerProdImpl {
    DUAL_INNER_PROD_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_celt_pitch_xcorr_impl(arch: i32) -> CeltPitchXcorrImpl {
    CELT_PITCH_XCORR_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_xcorr_kernel_impl(arch: i32) -> XcorrKernelImpl {
    XCORR_KERNEL_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_opus_fft_alloc_arch_impl(arch: i32) -> OpusFftAllocArchImpl {
    OPUS_FFT_ALLOC_ARCH_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_opus_fft_free_arch_impl(arch: i32) -> OpusFftFreeArchImpl {
    OPUS_FFT_FREE_ARCH_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_opus_fft_impl(arch: i32) -> OpusFftImpl {
    OPUS_FFT[dispatch_index(arch)]
}

#[inline]
pub fn select_opus_ifft_impl(arch: i32) -> OpusFftImpl {
    OPUS_IFFT[dispatch_index(arch)]
}

#[inline]
pub fn select_clt_mdct_forward_impl(arch: i32) -> CltMdctForwardImpl {
    CLT_MDCT_FORWARD_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_clt_mdct_backward_impl(arch: i32) -> CltMdctBackwardImpl {
    CLT_MDCT_BACKWARD_IMPL[dispatch_index(arch)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn float2int16_dispatch_matches_scalar() {
        let input = vec![0.9, -1.25, 0.5, 2.5];
        let mut via_dispatch = vec![0i16; input.len()];
        let mut expected = vec![0i16; input.len()];

        select_celt_float2int16_impl(0)(&input, &mut via_dispatch);
        celt_float2int16(&input, &mut expected);
        assert_eq!(via_dispatch, expected);
    }

    #[test]
    fn mdct_dispatch_matches_scalar_forward() {
        let lookup = MdctLookup::new(32, 0);
        let n = lookup.len();
        let overlap = n >> 1;
        let input = vec![0.0f32; overlap + n];
        let window = vec![0.5f32; lookup.len() >> 1];
        let mut via_dispatch = vec![0.0f32; n >> 1];
        let mut expected = vec![0.0f32; n >> 1];

        select_clt_mdct_forward_impl(0)(
            &lookup,
            &input,
            &mut via_dispatch,
            &window,
            overlap,
            0,
            1,
        );
        clt_mdct_forward(
            &lookup,
            &input,
            &mut expected,
            &window,
            overlap,
            0,
            1,
        );
        assert_eq!(via_dispatch, expected);
    }

    #[test]
    fn fft_dispatch_matches_scalar() {
        let state = KissFftState::new(8);
        let input = vec![KissFftCpx::default(); state.nfft()];
        let mut via_dispatch = vec![KissFftCpx::default(); state.nfft()];
        let mut expected = vec![KissFftCpx::default(); state.nfft()];

        select_opus_fft_impl(0)(&state, &input, &mut via_dispatch);
        opus_fft(&state, &input, &mut expected);
        assert_eq!(via_dispatch, expected);
    }
}
