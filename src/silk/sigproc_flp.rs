//! Port of the lightweight helpers from `silk/float/SigProc_FLP.h`.
//!
//! The floating-point encoder path relies on a handful of inline utilities for
//! sigmoid evaluation, float-to-int conversion, saturating array casts, and a
//! base-2 logarithm helper.  The original C header exposes these as static
//! inline functions; this module mirrors that behaviour with safe Rust
//! equivalents so the remaining FLP routines can reuse the same building
//! blocks.

use crate::celt::float2int;
use libm::{expf, log10};

/// Pi constant used by the SILK floating-point helpers.
pub const PI: f32 = core::f32::consts::PI;

/// Returns the smaller of two floating-point values.
#[inline]
pub fn silk_min_float(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Returns the larger of two floating-point values.
#[inline]
pub fn silk_max_float(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Absolute value helper mirroring the C macro.
#[inline]
pub fn silk_abs_float(a: f32) -> f32 {
    a.abs()
}

/// Logistic sigmoid helper used by several FLP encoder routines.
#[inline]
pub fn silk_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf(-x))
}

/// Floating-point to integer conversion that matches the C `float2int` macro.
///
/// The conversion rounds to the nearest even integer just like the C
/// implementation backed by `lrintf`.
#[inline]
pub fn silk_float2int(x: f32) -> i32 {
    float2int(x)
}

/// Saturating float-to-i16 conversion for entire slices.
///
/// Mirrors `silk_float2short_array` by rounding each element to the nearest
/// even integer and clamping to the i16 range.
pub fn silk_float2short_array(out: &mut [i16], input: &[f32]) {
    assert_eq!(
        out.len(),
        input.len(),
        "output and input slices must have matching lengths"
    );

    for (dst, &src) in out.iter_mut().zip(input.iter()) {
        let rounded = silk_float2int(src);
        *dst = rounded
            .clamp(i32::from(i16::MIN), i32::from(i16::MAX))
            as i16;
    }
}

/// Lossless i16-to-f32 conversion for entire slices.
pub fn silk_short2float_array(out: &mut [f32], input: &[i16]) {
    assert_eq!(
        out.len(),
        input.len(),
        "output and input slices must have matching lengths"
    );

    for (dst, &src) in out.iter_mut().zip(input.iter()) {
        *dst = f32::from(src);
    }
}

/// Base-2 logarithm helper.
///
/// The C macro expresses `log2(x)` using `log10` to avoid pulling in an extra
/// dependency; we keep the same formulation for bit-for-bit parity.
#[inline]
pub fn silk_log2(x: f64) -> f32 {
    (core::f64::consts::LOG2_10 * log10(x)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_matches_reference_points() {
        let expected = [0.5f32, 0.880_797_1, 0.119_202_92];
        let actual = [
            silk_sigmoid(0.0),
            silk_sigmoid(2.0),
            silk_sigmoid(-2.0),
        ];
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn float2int_keeps_even_rounding() {
        assert_eq!(silk_float2int(1.5), 2);
        assert_eq!(silk_float2int(2.5), 2);
        assert_eq!(silk_float2int(-1.5), -2);
        assert_eq!(silk_float2int(-2.5), -2);
    }

    #[test]
    fn array_conversions_round_and_saturate() {
        let input = [0.4f32, 1.6, 32_800.3, -40_000.9];
        let mut out_i16 = [0i16; 4];
        silk_float2short_array(&mut out_i16, &input);
        assert_eq!(out_i16, [0, 2, i16::MAX, i16::MIN]);

        let mut out_f32 = [0.0f32; 4];
        silk_short2float_array(&mut out_f32, &out_i16);
        assert_eq!(out_f32, [0.0, 2.0, 32_767.0, -32_768.0]);
    }

    #[test]
    fn log2_matches_simple_values() {
        assert_eq!(silk_log2(1.0), 0.0);
        assert!((silk_log2(2.0) - 1.0).abs() < 1e-6);
        assert!((silk_log2(8.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn min_max_abs_helpers_match_std() {
        assert_eq!(silk_min_float(1.0, -2.0), -2.0);
        assert_eq!(silk_max_float(1.0, -2.0), 1.0);
        assert_eq!(silk_abs_float(-3.5), 3.5);
    }
}
