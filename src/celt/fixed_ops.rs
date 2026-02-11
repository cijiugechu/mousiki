#![allow(dead_code)]

use core::cmp::Ordering;

use super::types::{FixedCeltSig, FixedOpusVal16, FixedOpusVal32};
use libm::trunc;

#[inline]
pub(crate) fn qconst16(value: f64, bits: u32) -> FixedOpusVal16 {
    let scale = (1i64 << bits) as f64;
    trunc(value * scale + 0.5) as FixedOpusVal16
}

#[inline]
pub(crate) fn qconst32(value: f64, bits: u32) -> FixedOpusVal32 {
    let scale = (1i64 << bits) as f64;
    trunc(value * scale + 0.5) as FixedOpusVal32
}

#[inline]
pub(crate) fn add32(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    a.wrapping_add(b)
}

#[inline]
pub(crate) fn sub32(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    a.wrapping_sub(b)
}

#[inline]
pub(crate) fn add32_ovflw(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    a.wrapping_add(b)
}

#[inline]
pub(crate) fn sub32_ovflw(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    a.wrapping_sub(b)
}

#[inline]
pub(crate) fn neg32_ovflw(a: FixedOpusVal32) -> FixedOpusVal32 {
    a.wrapping_neg()
}

#[inline]
pub(crate) fn abs32(a: FixedOpusVal32) -> FixedOpusVal32 {
    if a < 0 {
        a.wrapping_neg()
    } else {
        a
    }
}

#[inline]
pub(crate) fn extract16(a: FixedOpusVal32) -> FixedOpusVal16 {
    a as FixedOpusVal16
}

#[inline]
pub(crate) fn extend32(a: FixedOpusVal16) -> FixedCeltSig {
    FixedCeltSig::from(a)
}

#[inline]
pub(crate) fn shl32(a: FixedOpusVal32, shift: u32) -> FixedOpusVal32 {
    a.wrapping_shl(shift)
}

#[inline]
pub(crate) fn shr32(a: FixedOpusVal32, shift: u32) -> FixedOpusVal32 {
    a >> shift
}

#[inline]
pub(crate) fn shr16(a: FixedOpusVal16, shift: u32) -> FixedOpusVal16 {
    a >> shift
}

#[inline]
pub(crate) fn vshr32(a: FixedOpusVal32, shift: i32) -> FixedOpusVal32 {
    match shift.cmp(&0) {
        Ordering::Greater => shr32(a, shift as u32),
        Ordering::Less => shl32(a, (-shift) as u32),
        Ordering::Equal => a,
    }
}

#[inline]
pub(crate) fn pshr32(a: FixedOpusVal32, shift: u32) -> FixedOpusVal32 {
    if shift == 0 {
        return a;
    }
    let bias = 1i32.wrapping_shl(shift - 1);
    shr32(a.wrapping_add(bias), shift)
}

#[inline]
pub(crate) fn pshr32_ovflw(a: FixedOpusVal32, shift: u32) -> FixedOpusVal32 {
    if shift == 0 {
        return a;
    }
    let bias = 1i32.wrapping_shl(shift - 1);
    shr32(a.wrapping_add(bias), shift)
}

#[inline]
pub(crate) fn mult16_16(a: FixedOpusVal16, b: FixedOpusVal16) -> FixedOpusVal32 {
    FixedOpusVal32::from(a) * FixedOpusVal32::from(b)
}

#[inline]
pub(crate) fn mult16_16_q15(a: FixedOpusVal16, b: FixedOpusVal16) -> FixedOpusVal16 {
    (mult16_16(a, b) >> 15) as FixedOpusVal16
}

#[inline]
pub(crate) fn mult16_16_p15(a: FixedOpusVal16, b: FixedOpusVal16) -> FixedOpusVal16 {
    ((mult16_16(a, b) + 16_384) >> 15) as FixedOpusVal16
}

#[inline]
pub(crate) fn mult16_32_q15(a: FixedOpusVal16, b: FixedOpusVal32) -> FixedOpusVal32 {
    ((i64::from(a) * i64::from(b)) >> 15) as FixedOpusVal32
}

#[inline]
pub(crate) fn mult16_32_q16(a: FixedOpusVal16, b: FixedOpusVal32) -> FixedOpusVal32 {
    ((i64::from(a) * i64::from(b)) >> 16) as FixedOpusVal32
}

#[inline]
pub(crate) fn mult32_32_q31(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    ((i64::from(a) * i64::from(b)) >> 31) as FixedOpusVal32
}

#[inline]
pub(crate) fn mult32_32_q16(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    ((i64::from(a) * i64::from(b)) >> 16) as FixedOpusVal32
}

#[inline]
pub(crate) fn mult32_32_p31(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    ((i64::from(a) * i64::from(b) + (1i64 << 30)) >> 31) as FixedOpusVal32
}

#[inline]
pub(crate) fn mult32_32_32(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    a.wrapping_mul(b)
}

#[inline]
pub(crate) fn div32(a: FixedOpusVal32, b: FixedOpusVal32) -> FixedOpusVal32 {
    debug_assert!(b != 0);
    a / b
}

#[inline]
pub(crate) fn mac16_16(c: FixedOpusVal32, a: FixedOpusVal16, b: FixedOpusVal16) -> FixedOpusVal32 {
    add32(c, mult16_16(a, b))
}
