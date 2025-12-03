#![allow(dead_code)]

//! Runtime dispatch table mirroring `celt/x86/x86_celt_map.c`.
//!
//! The reference implementation uses this map to select SIMD fast paths based
//! on the detected x86 architecture flags. The Rust port keeps the same table
//! layout but routes every entry to the scalar helpers because
//! `OPUS_ARCHMASK` is stubbed to zero until runtime dispatch is implemented.

use crate::celt::cpu_support::OPUS_ARCHMASK;
use crate::celt::lpc::celt_fir;
use crate::celt::pitch::{celt_inner_prod, celt_pitch_xcorr, dual_inner_prod, xcorr_kernel};
use crate::celt::types::{CeltCoef, OpusInt32, OpusVal16, OpusVal32};
use crate::celt::vq::op_pvq_search;
use crate::celt::comb_filter_const;

const ARCH_IMPL_COUNT: usize = (OPUS_ARCHMASK as usize) + 1;

pub type CeltFirImpl = fn(&[OpusVal16], &[OpusVal16], &mut [OpusVal16]);
pub const CELT_FIR_IMPL: [CeltFirImpl; ARCH_IMPL_COUNT] = [celt_fir; ARCH_IMPL_COUNT];

pub type XcorrKernelImpl = fn(&[OpusVal16], &[OpusVal16], &mut [OpusVal32; 4], usize);
pub const XCORR_KERNEL_IMPL: [XcorrKernelImpl; ARCH_IMPL_COUNT] =
    [xcorr_kernel; ARCH_IMPL_COUNT];

pub type CeltInnerProdImpl = fn(&[OpusVal16], &[OpusVal16]) -> OpusVal32;
pub const CELT_INNER_PROD_IMPL: [CeltInnerProdImpl; ARCH_IMPL_COUNT] =
    [celt_inner_prod; ARCH_IMPL_COUNT];

pub type CeltPitchXcorrImpl =
    fn(&[OpusVal16], &[OpusVal16], usize, usize, &mut [OpusVal32]);
pub const CELT_PITCH_XCORR_IMPL: [CeltPitchXcorrImpl; ARCH_IMPL_COUNT] =
    [celt_pitch_xcorr; ARCH_IMPL_COUNT];

pub type DualInnerProdImpl =
    fn(&[OpusVal16], &[OpusVal16], &[OpusVal16]) -> (OpusVal32, OpusVal32);
pub const DUAL_INNER_PROD_IMPL: [DualInnerProdImpl; ARCH_IMPL_COUNT] =
    [dual_inner_prod; ARCH_IMPL_COUNT];

pub type CombFilterConstImpl = fn(
    &mut [OpusVal32],
    &[OpusVal32],
    usize,
    usize,
    CeltCoef,
    CeltCoef,
    CeltCoef,
);
pub const COMB_FILTER_CONST_IMPL: [CombFilterConstImpl; ARCH_IMPL_COUNT] =
    [comb_filter_const; ARCH_IMPL_COUNT];

pub type OpPvqSearchImpl =
    fn(&mut [OpusVal16], &mut [OpusInt32], usize, i32, i32) -> OpusVal32;
pub const OP_PVQ_SEARCH_IMPL: [OpPvqSearchImpl; ARCH_IMPL_COUNT] =
    [op_pvq_search; ARCH_IMPL_COUNT];

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
pub fn select_celt_fir_impl(arch: i32) -> CeltFirImpl {
    CELT_FIR_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_xcorr_kernel_impl(arch: i32) -> XcorrKernelImpl {
    XCORR_KERNEL_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_celt_inner_prod_impl(arch: i32) -> CeltInnerProdImpl {
    CELT_INNER_PROD_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_celt_pitch_xcorr_impl(arch: i32) -> CeltPitchXcorrImpl {
    CELT_PITCH_XCORR_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_dual_inner_prod_impl(arch: i32) -> DualInnerProdImpl {
    DUAL_INNER_PROD_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_comb_filter_const_impl(arch: i32) -> CombFilterConstImpl {
    COMB_FILTER_CONST_IMPL[dispatch_index(arch)]
}

#[inline]
pub fn select_op_pvq_search_impl(arch: i32) -> OpPvqSearchImpl {
    OP_PVQ_SEARCH_IMPL[dispatch_index(arch)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn xcorr_kernel_dispatch_matches_scalar() {
        let len = 10usize;
        let x = (0..len).map(|v| v as OpusVal16 * 0.1).collect::<Vec<_>>();
        let y = (0..len + 3)
            .map(|v| -(v as OpusVal16) * 0.2)
            .collect::<Vec<_>>();

        let mut sums = [0.0f32; 4];
        let mut expected = [0.0f32; 4];
        select_xcorr_kernel_impl(0)(&x, &y, &mut sums, len);
        xcorr_kernel(&x, &y, &mut expected, len);
        assert_eq!(sums, expected);
    }

    #[test]
    fn pitch_xcorr_dispatch_matches_scalar() {
        let len = 16usize;
        let max_pitch = 6usize;
        let x = (0..len).map(|v| v as OpusVal16 * 0.01).collect::<Vec<_>>();
        let y = (0..len + max_pitch)
            .map(|v| (v as OpusVal16 * 0.02) - 0.5)
            .collect::<Vec<_>>();

        let mut via_dispatch = vec![0.0f32; max_pitch];
        let mut expected = vec![0.0f32; max_pitch];
        select_celt_pitch_xcorr_impl(0)(&x, &y, len, max_pitch, &mut via_dispatch);
        celt_pitch_xcorr(&x, &y, len, max_pitch, &mut expected);
        assert_eq!(via_dispatch, expected);
    }

    #[test]
    fn op_pvq_search_dispatch_matches_scalar() {
        let n = 5usize;
        let mut x = vec![0.5, -0.25, 0.75, -0.5, 0.1];
        let mut pulses = vec![0i32; n];

        let via_dispatch =
            select_op_pvq_search_impl(0)(&mut x.clone(), &mut pulses.clone(), n, 4, 0);
        let expected = op_pvq_search(&mut x, &mut pulses, n, 4, 0);
        assert_eq!(via_dispatch, expected);
    }
}
