#![allow(dead_code)]

//! Fixed-point helpers from `celt/mathops.c`.
//!
//! The reference CELT implementation provides a number of specialised
//! fixed-point math routines that are used when the codec is built without
//! floating-point support. The existing Rust port already covered the
//! floating-point variants in [`math`]; this module mirrors the integer
//! counterparts so that future translations that depend on the fixed-point
//! helpers can reuse them directly.

use crate::celt::math::celt_ilog2;

pub(crate) fn celt_maxabs16(samples: &[i16]) -> i32 {
    let mut max_abs = 0i32;
    for &sample in samples {
        let abs = i32::from(sample).abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    max_abs
}

pub(crate) fn celt_maxabs32(samples: &[i32]) -> i32 {
    let mut max_abs = 0i32;
    for &sample in samples {
        let abs = sample.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    max_abs
}

fn vshr32(a: i32, shift: i32) -> i32 {
    match shift.cmp(&0) {
        core::cmp::Ordering::Greater => a >> shift,
        core::cmp::Ordering::Less => a.wrapping_shl((-shift) as u32),
        core::cmp::Ordering::Equal => a,
    }
}

fn pshr32(a: i32, shift: u32) -> i32 {
    if shift == 0 {
        a
    } else {
        let bias = 1i64 << (shift - 1);
        ((i64::from(a) + bias) >> shift) as i32
    }
}

fn round16(value: i32, bits: u32) -> i16 {
    pshr32(value, bits) as i16
}

fn mult16_16(a: i16, b: i16) -> i32 {
    i32::from(a) * i32::from(b)
}

fn mult16_16_q15(a: i16, b: i16) -> i16 {
    (mult16_16(a, b) >> 15) as i16
}

fn mult16_16_p15(a: i16, b: i16) -> i16 {
    ((mult16_16(a, b) + 16_384) >> 15) as i16
}

fn mult16_32_q15(a: i16, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 15) as i32
}

fn mult32_32_q31(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 31) as i32
}

fn shl32(value: i32, shift: u32) -> i32 {
    value.wrapping_shl(shift)
}

fn shr16(value: i16, shift: u32) -> i16 {
    value >> shift
}

fn shl16(value: i16, shift: u32) -> i16 {
    i32::from(value).wrapping_shl(shift) as i16
}

fn extract16(value: i32) -> i16 {
    value as i16
}

fn add16(a: i16, b: i16) -> i16 {
    a.wrapping_add(b)
}

fn sub16(a: i16, b: i16) -> i16 {
    a.wrapping_sub(b)
}

fn add32(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

fn sub32(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

fn min16(a: i16, b: i16) -> i16 {
    if a < b { a } else { b }
}

/// Fixed-point reciprocal square root in the range `[0.25, 1)`.
///
/// Mirrors `celt_rsqrt_norm()` from the reference implementation. Inputs are
/// Q16 fixed-point values and the output is returned in Q14 precision.
pub(crate) fn celt_rsqrt_norm(x: i32) -> i16 {
    let n = (x - 32_768) as i16; // Q15 offset
    let r = add16(
        23_557,
        mult16_16_q15(n, add16(-13_490, mult16_16_q15(n, 6_713))),
    );
    let r2 = mult16_16_q15(r, r);
    let y = shl16(sub16(add16(mult16_16_q15(r2, n), r2), 16_384), 1);

    add16(
        r,
        mult16_16_q15(r, mult16_16_q15(y, sub16(mult16_16_q15(y, 12_288), 16_384))),
    )
}

/// Fixed-point square root approximation.
///
/// This mirrors the `_celt_sqrt()` helper from `mathops.c`, operating on QX
/// inputs and returning a QX/2 result.
pub(crate) fn celt_sqrt(mut x: i32) -> i32 {
    if x == 0 {
        return 0;
    }
    if x >= 1_073_741_824 {
        return 32_767;
    }

    let k = (celt_ilog2(x) >> 1) - 7;
    x = vshr32(x, 2 * k);
    let n = (x - 32_768) as i16;
    let coeffs = [23_171, 11_574, -2_901, 1_592, -1_002, 336];

    let mut acc = coeffs[5];
    acc = add16(coeffs[4], mult16_16_q15(n, acc));
    acc = add16(coeffs[3], mult16_16_q15(n, acc));
    acc = add16(coeffs[2], mult16_16_q15(n, acc));
    acc = add16(coeffs[1], mult16_16_q15(n, acc));
    let result = add32(i32::from(coeffs[0]), i32::from(mult16_16_q15(n, acc)));
    vshr32(result, 7 - k)
}

fn celt_cos_pi_2(x: i16) -> i16 {
    let x2 = mult16_16_p15(x, x);
    let mut term = mult16_16_p15(-626, x2);
    term = add16(8_277, term);
    term = mult16_16_p15(x2, term);
    term = add16(-7_651, term);
    let acc = add32(i32::from(sub16(32_767, x2)), i32::from(term));
    add16(1, min16(32_766, acc as i16))
}

/// Fixed-point cosine helper used by the MDCT window generation code.
pub(crate) fn celt_cos_norm(mut x: i32) -> i16 {
    x &= 0x0001_FFFF;
    if x > (1 << 16) {
        x = (1 << 17) - x;
    }

    if x & 0x0000_7FFF != 0 {
        if x < (1 << 15) {
            celt_cos_pi_2(x as i16)
        } else {
            -celt_cos_pi_2((65_536 - x) as i16)
        }
    } else if x & 0x0000_FFFF != 0 {
        0
    } else if x & 0x0001_FFFF != 0 {
        -32_767
    } else {
        32_767
    }
}

/// Fixed-point reciprocal approximation.
pub(crate) fn celt_rcp(x: i32) -> i32 {
    debug_assert!(x > 0);

    let i = celt_ilog2(x);
    let n = (vshr32(x, i - 15) - 32_768) as i16;
    let mut r = add16(30_840, mult16_16_q15(-15_420, n));

    let term = add16(mult16_16_q15(r, n), add16(r, -32_768i16));
    r = sub16(r, mult16_16_q15(r, term));
    let term = add16(mult16_16_q15(r, n), add16(r, -32_768i16));
    r = sub16(r, add16(1, mult16_16_q15(r, term)));

    vshr32(i32::from(r), i - 16)
}

/// Divides two Q32 values returning a Q32/Q29 quotient.
pub(crate) fn frac_div32_q29(a: i32, b: i32) -> i32 {
    debug_assert!(b != 0);

    let shift = celt_ilog2(b) - 29;
    let a = vshr32(a, shift);
    let b = vshr32(b, shift);
    let rcp = round16(celt_rcp(i32::from(round16(b, 16))), 3);
    let mut result = mult16_32_q15(rcp, a);
    let rem = pshr32(a, 2) - mult32_32_q31(result, b);
    result = add32(result, shl32(mult16_32_q15(rcp, rem), 2));
    result
}

/// Saturated fractional division helper.
pub(crate) fn frac_div32(a: i32, b: i32) -> i32 {
    let result = frac_div32_q29(a, b);
    if result >= 536_870_912 {
        2_147_483_647
    } else if result <= -536_870_912 {
        -2_147_483_647
    } else {
        shl32(result, 2)
    }
}

/// Fixed-point division using reciprocal.
/// Mirrors `celt_div` macro in C fixed-point build.
pub(crate) fn celt_div(a: i32, b: i32) -> i32 {
    mult32_32_q31(a, celt_rcp(b))
}

/// 4th order polynomial approximation of atan.
/// Input is in Q15 format and normalized by pi/4. Output is in Q15 format.
/// Mirrors `celt_atan01()` from `celt/mathops.h`.
fn celt_atan01(x: i16) -> i16 {
    const M1: i32 = 32767;
    const M2: i32 = -21;
    const M3: i32 = -11943;
    const M4: i32 = 4936;
    
    // Nested polynomial evaluation: x * (M1 + x * (M2 + x * (M3 + x * M4)))
    let term4 = mult16_16_p15(M4 as i16, x);
    let term3 = add32(M3, i32::from(term4));
    let term3 = mult16_16_p15(x, term3 as i16);
    let term2 = add32(M2, i32::from(term3));
    let term2 = mult16_16_p15(x, term2 as i16);
    let term1 = add32(M1, i32::from(term2));
    mult16_16_p15(x, term1 as i16)
}

/// atan2() approximation valid for positive input values.
/// Mirrors `celt_atan2p()` from `celt/mathops.h`.
pub(crate) fn celt_atan2p(y: i16, x: i16) -> i16 {
    if y < x {
        let mut arg = celt_div(shl32(i32::from(y), 15), i32::from(x));
        if arg >= 32767 {
            arg = 32767;
        }
        shr16(celt_atan01(extract16(arg) as i16), 1)
    } else {
        let mut arg = celt_div(shl32(i32::from(x), 15), i32::from(y));
        if arg >= 32767 {
            arg = 32767;
        }
        25736 - shr16(celt_atan01(extract16(arg) as i16), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        celt_cos_norm, celt_maxabs16, celt_maxabs32, celt_rcp, celt_rsqrt_norm, celt_sqrt,
        frac_div32, frac_div32_q29, vshr32,
    };

    #[test]
    fn vshr32_matches_expected_behaviour() {
        assert_eq!(vshr32(1 << 16, 1), 1 << 15);
        assert_eq!(vshr32(1 << 10, -2), 1 << 12);
        assert_eq!(vshr32(-1 << 10, 2), -1 << 8);
    }

    #[test]
    fn frac_division_maintains_scaling() {
        let num = 1 << 20;
        let den = 1 << 15;
        let q29 = frac_div32_q29(num, den);
        let q = frac_div32(num, den);
        assert_eq!(q, q29 << 2);
    }

    #[test]
    fn reciprocal_returns_positive_values() {
        for x in (1 << 15)..(1 << 18) {
            assert!(celt_rcp(x) > 0);
        }
    }

    #[test]
    fn rsqrt_norm_stays_positive() {
        for value in (1 << 15)..(1 << 16) {
            let r = celt_rsqrt_norm(value);
            assert!(r > 0);
        }
    }

    #[test]
    fn cos_norm_returns_bounded_values() {
        assert_eq!(celt_cos_norm(0), 32_767);

        for raw in (0..=1 << 16).step_by(1 << 12) {
            let value = i32::from(celt_cos_norm(raw));
            assert!((-32_767..=32_767).contains(&value));
        }
    }

    #[test]
    fn sqrt_monotonic() {
        let mut prev = celt_sqrt(1 << 16);
        for x in ((1 << 16) + 1)..((1 << 16) + 1_000) {
            let current = celt_sqrt(x);
            assert!(current >= prev);
            prev = current;
        }
    }

    #[test]
    fn maxabs_helpers_match_expected_values() {
        let samples_16 = [0i16, -12, 9, 32, -31];
        let samples_32 = [0i32, -12_000, 9_000, 32_000, -31_999];

        assert_eq!(celt_maxabs16(&samples_16), 32);
        assert_eq!(celt_maxabs32(&samples_32), 32_000);
    }
}
