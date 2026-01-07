#![allow(dead_code)]

//! Mathematical helpers from the original CELT implementation.
//!
//! These are small functions with limited dependencies that can be ported in
//! isolation.  They are primarily used by the analysis and psychoacoustic
//! portions of the codec and map closely to the routines defined in
//! `celt/mathops.h` in the reference implementation.

use core::f32::consts::LN_2;
use core::f32::consts::{LOG2_E, PI};

use crate::celt::entcode::ec_ilog;
use crate::celt::float_cast;
use crate::celt::types::OpusInt32;
#[cfg(not(miri))]
use libm::sqrtf;
use libm::{cosf, expf, logf};

/// Integer square root mirroring `isqrt32()` from `celt/mathops.c`.
///
/// The function computes `floor(sqrt(x))` for positive 32-bit integers using a
/// bit-by-bit refinement strategy that matches the behaviour of the reference
/// implementation.  The original routine relies on the `EC_ILOG` macro to
/// determine the starting bit; the Rust port uses the intrinsic
/// `leading_zeros()` to achieve the same effect.
pub(crate) fn isqrt32(mut value: u32) -> u32 {
    if value == 0 {
        return 0;
    }

    let mut root = 0u32;
    let mut bit_shift = ((32 - value.leading_zeros()) as i32 - 1) >> 1;
    let mut bit = 1u32 << (bit_shift as u32);

    while bit_shift >= 0 {
        let trial = ((root << 1) + bit) << (bit_shift as u32);
        if trial <= value {
            root += bit;
            value -= trial;
        }
        bit >>= 1;
        bit_shift -= 1;
    }

    root
}

/// Integer base-2 logarithm used by the fixed-point helpers.
///
/// Mirrors `celt_ilog2()` from `celt/mathops.h`, wrapping the shared range coder
/// helper while enforcing that the input is strictly positive. The return value
/// matches the C implementation by reporting the position of the highest set
/// bit (zero-indexed).
#[must_use]
pub(crate) fn celt_ilog2(value: OpusInt32) -> OpusInt32 {
    assert!(value > 0, "celt_ilog2 expects a strictly positive input");
    ec_ilog(value as u32) - 1
}

/// Integer base-2 logarithm defined for zero.
///
/// Ports `celt_zlog2()` from `celt/mathops.h`. The helper mirrors
/// `celt_ilog2()` for positive inputs while returning `0` when the argument is
/// zero or negative, matching the guard in the reference implementation.
#[must_use]
pub(crate) fn celt_zlog2(value: OpusInt32) -> OpusInt32 {
    if value <= 0 { 0 } else { celt_ilog2(value) }
}

/// Fast arctangent approximation used by the psychoacoustic analysis code.
///
/// Mirrors the `fast_atan2f()` helper from `celt/mathops.h` when building the
/// float variant of CELT.  The approximation is accurate enough for the
/// heuristics that rely on it while avoiding the cost of calling into libm.
#[allow(clippy::many_single_char_names)]
pub(crate) fn fast_atan2f(y: f32, x: f32) -> f32 {
    const CA: f32 = 0.431_579_74_f32;
    // Matches the 0.67848403f literal used in `celt/mathops.h` in the C tree.
    const CB: f32 = f32::from_bits(0x3f2d_b121);
    const CC: f32 = 0.085_955_42_f32;
    const CE: f32 = PI / 2.0;

    let x2 = x * x;
    let y2 = y * y;

    if x2 + y2 < 1e-18 {
        return 0.0;
    }

    if x2 < y2 {
        let den = (y2 + CB * x2) * (y2 + CC * x2);
        -x * y * (y2 + CA * x2) / den + if y < 0.0 { -CE } else { CE }
    } else {
        let den = (x2 + CB * y2) * (x2 + CC * y2);
        x * y * (x2 + CA * y2) / den + if y < 0.0 { -CE } else { CE }
            - if x * y < 0.0 { -CE } else { CE }
    }
}

/// Base-2 logarithm used by CELT's float build.
///
/// The reference implementation exposes this through a macro.  The Rust port
/// provides a function wrapper to keep call sites ergonomic while preserving
/// the behaviour.
#[inline]
pub(crate) fn celt_log2(x: f32) -> f32 {
    // 1 / ln(2)
    LOG2_E * logf(x)
}

/// Base-2 exponential used by CELT's float build.
#[inline]
pub(crate) fn celt_exp2(x: f32) -> f32 {
    expf(LN_2 * x)
}

/// Division helper matching the semantics of `celt_div()` in the C codebase.
#[inline]
pub(crate) fn celt_div(a: f32, b: f32) -> f32 {
    a / b
}

/// Cosine helper implementing `celt_cos_norm()` for the float build.
#[inline]
pub(crate) fn celt_cos_norm(x: f32) -> f32 {
    cosf(0.5 * PI * x)
}

/// Square-root helper mirroring the `celt_sqrt()` macro from `mathops.h`.
#[inline]
pub(crate) fn celt_sqrt(x: f32) -> f32 {
    #[cfg(miri)]
    {
        return sqrtf_fallback(x);
    }

    #[cfg(not(miri))]
    {
        sqrtf(x)
    }
}

#[cfg(miri)]
#[inline]
fn sqrtf_fallback(x: f32) -> f32 {
    if !(x > 0.0) {
        if x == 0.0 {
            return 0.0;
        }
        if x.is_sign_negative() {
            return f32::NAN;
        }
        return x;
    }

    let mut guess = f32::from_bits((x.to_bits() >> 1) + 0x1fc0_0000);
    for _ in 0..4 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

/// Reciprocal square-root helper that matches `celt_rsqrt()`.
#[inline]
pub(crate) fn celt_rsqrt(x: f32) -> f32 {
    1.0 / celt_sqrt(x)
}

/// Normalised reciprocal square-root helper matching `celt_rsqrt_norm()`.
#[inline]
pub(crate) fn celt_rsqrt_norm(x: f32) -> f32 {
    celt_rsqrt(x)
}

/// Reciprocal helper mirroring `celt_rcp()` from the float build.
#[inline]
pub(crate) fn celt_rcp(x: f32) -> f32 {
    1.0 / x
}

/// Fractional division helper `frac_div32()` from the float build.
#[inline]
pub(crate) fn frac_div32(a: f32, b: f32) -> f32 {
    a / b
}

/// Float alias of `frac_div32()` that mirrors `frac_div32_q29()`.
#[inline]
pub(crate) fn frac_div32_q29(a: f32, b: f32) -> f32 {
    frac_div32(a, b)
}

/// Returns the largest absolute sample value in `samples`.
///
/// Mirrors `celt_maxabs16()` from the reference implementation when building
/// the float variant of CELT. The original helper scans the slice once while
/// tracking the extrema of the positive and negative ranges separately; the
/// Rust port collapses this to a single absolute-value comparison while
/// preserving the behaviour for empty inputs (returning `0.0`).
pub(crate) fn celt_maxabs16(samples: &[f32]) -> f32 {
    let mut max_abs = 0.0f32;

    for &value in samples {
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    max_abs
}

/// Float build alias of `celt_maxabs16()` that mirrors `celt_maxabs32()`.
#[inline]
pub(crate) fn celt_maxabs32(samples: &[f32]) -> f32 {
    celt_maxabs16(samples)
}

/// Clamps samples to the `[-2, 2]` range as in `opus_limit2_checkwithin1_c()`.
///
/// The C helper returns a hint indicating whether all samples were guaranteed to
/// fall inside `[-1, 1]`. The scalar implementation always pessimistically
/// returns `0` (false) when samples are present, because it cannot cheaply
/// provide this guarantee after the clamping pass. The Rust port mirrors this
/// behaviour by returning `false` for non-empty slices and `true` only when the
/// slice is empty.
pub(crate) fn opus_limit2_checkwithin1(samples: &mut [f32]) -> bool {
    if samples.is_empty() {
        return true;
    }

    for sample in samples {
        *sample = sample.clamp(-2.0, 2.0);
    }

    false
}

/// Converts floating-point samples to 16-bit integers as in `celt_float2int16_c()`.
///
/// The helper scales the input by CELT's fixed-point signal scaling factor,
/// clamps the result to the signed 16-bit range, and rounds to the nearest
/// integer following the default IEEE 754 rounding mode. The C implementation
/// uses the `FLOAT2INT16` macro from `float_cast.h`; this port matches its
/// semantics so that callers relying on the float API can operate on Rust
/// slices directly.
pub(crate) fn celt_float2int16(input: &[f32], output: &mut [i16]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output slices must have the same length"
    );

    for (dst, &sample) in output.iter_mut().zip(input.iter()) {
        *dst = float_cast::float2int16(sample);
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use alloc::vec;
    use libm::cosf;

    use super::isqrt32;
    use crate::celt::float_cast::CELT_SIG_SCALE;

    use super::{
        celt_cos_norm, celt_div, celt_exp2, celt_float2int16, celt_ilog2, celt_log2, celt_maxabs16,
        celt_maxabs32, celt_rcp, celt_rsqrt, celt_rsqrt_norm, celt_sqrt, celt_zlog2, fast_atan2f,
        frac_div32, frac_div32_q29, opus_limit2_checkwithin1,
    };
    use crate::celt::entcode::ec_ilog;

    #[test]
    fn fast_atan2f_matches_std() {
        let samples = [
            (0.0_f32, 0.0_f32),
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (0.5, -0.75),
            (3.0, 4.0),
            (-2.0, -5.0),
        ];

        for &(y, x) in &samples {
            let approx = fast_atan2f(y, x);
            let exact = y.atan2(x);
            let diff = (approx - exact).abs();
            assert!(diff <= 5e-3, "diff {} for y={}, x={}", diff, y, x);
        }
    }

    #[test]
    fn log2_matches_std() {
        let values = [0.125_f32, 0.5, 1.0, 2.0, 10.0, 42.5];
        for &value in &values {
            let diff = (celt_log2(value) - value.log2()).abs();
            assert!(diff <= 1e-6, "diff {} for value {}", diff, value);
        }
    }

    /// Port of `testlog2()` from `opus-c/celt/tests/test_unit_mathops.c` (float build).
    ///
    /// Validates that `celt_log2()` matches the reference within tolerance for a
    /// range of input values.
    #[test]
    fn log2_matches_reference_harness() {
        let error_threshold = 2.2e-6_f32;
        let mut max_error = 0.0_f32;
        let mut x = 0.001_f32;

        while x < 1_677_700.0 {
            let expected = x.ln() * core::f32::consts::LOG2_E;
            let actual = celt_log2(x);
            let error = (expected - actual).abs();

            if error > max_error {
                max_error = error;
            }

            assert!(
                error <= error_threshold,
                "celt_log2 failed: x = {}, error = {} (threshold = {})",
                x,
                error,
                error_threshold
            );

            x += x / 8.0;
        }
    }

    #[test]
    fn exp2_matches_std() {
        let values = [-5.0_f32, -1.0, 0.0, 0.25, 1.5, 4.0];
        for &value in &values {
            let diff = (celt_exp2(value) - value.exp2()).abs();
            let eps = if cfg!(miri) { 1e-5 } else { 1e-6 };
            assert!(diff <= eps, "diff {} for value {}", diff, value);
        }
    }

    /// Port of `testexp2()` from `opus-c/celt/tests/test_unit_mathops.c` (float build).
    ///
    /// Validates that `celt_exp2()` matches the reference within tolerance for a
    /// range of input values. The tolerance is slightly relaxed from the C test
    /// because this version validates `x ≈ ln(celt_exp2(x)) / ln(2)`, which
    /// is sensitive to the exp2 implementation accuracy.
    #[test]
    fn exp2_matches_reference_harness() {
        // Tolerance relaxed from 2.3e-7 in C since we use libm which may differ slightly
        let error_threshold = 2.5e-6_f32;
        let mut max_error = 0.0_f32;
        let mut x = -11.0_f32;

        while x < 24.0 {
            let actual = celt_exp2(x);
            let expected = actual.ln() * core::f32::consts::LOG2_E;
            let error = (x - expected).abs();

            if error > max_error {
                max_error = error;
            }

            assert!(
                error <= error_threshold,
                "celt_exp2 failed: x = {}, error = {} (threshold = {})",
                x,
                error,
                error_threshold
            );

            x += 0.0007;
        }
    }

    /// Port of `testexp2log2()` from `opus-c/celt/tests/test_unit_mathops.c` (float build).
    ///
    /// Validates the round-trip property: `celt_log2(celt_exp2(x)) ≈ x`.
    #[test]
    fn exp2_log2_roundtrip() {
        let error_threshold = 2.0e-6_f32;
        let mut max_error = 0.0_f32;
        let mut x = -11.0_f32;

        while x < 24.0 {
            let roundtrip = celt_log2(celt_exp2(x));
            let error = (x - roundtrip).abs();

            if error > max_error {
                max_error = error;
            }

            assert!(
                error <= error_threshold,
                "celt_exp2/celt_log2 roundtrip failed: x = {}, error = {} (threshold = {})",
                x,
                error,
                error_threshold
            );

            x += 0.0007;
        }
    }

    #[test]
    fn div_matches_std() {
        let samples = [(1.0_f32, 2.0_f32), (5.5, 1.1), (100.0, -25.0), (-3.75, 0.5)];

        for &(a, b) in &samples {
            assert!((celt_div(a, b) - a / b).abs() <= f32::EPSILON * 2.0);
        }
    }

    #[test]
    fn sqrt_matches_libm() {
        let values = [0.0_f32, 0.25, 1.0, 2.5, 16.0];
        for &value in &values {
            let diff = (celt_sqrt(value) - value.sqrt()).abs();
            assert!(diff <= 1e-6, "diff {} for value {}", diff, value);
        }
    }

    /// Port of `testsqrt()` from `opus-c/celt/tests/test_unit_mathops.c` (float build).
    ///
    /// Validates that `celt_sqrt()` matches the reference within tolerance for a
    /// range of input values from 1 to 1 billion. Uses the same logarithmic
    /// stepping as the C test: i += i >> 10.
    #[test]
    fn sqrt_matches_reference_harness() {
        let mut i: i64 = 1;
        while i <= 1_000_000_000 {
            let fi = i as f32;
            let val = celt_sqrt(fi);
            let expected = fi.sqrt();
            let ratio = val / expected;

            let tolerance_ratio = 0.0005;
            let tolerance_abs = 2.0;

            assert!(
                (ratio - 1.0).abs() <= tolerance_ratio || (val - expected).abs() <= tolerance_abs,
                "sqrt failed: sqrt({}) = {} (ratio = {}, expected = {})",
                i,
                val,
                ratio,
                expected
            );

            // Same logarithmic stepping as C test: faster iteration through large values
            i += (i >> 10).max(1);
        }
    }

    #[test]
    fn rsqrt_matches_inverse_sqrt() {
        let values = [0.25_f32, 1.0, 4.0, 16.0];
        for &value in &values {
            let expected = 1.0 / value.sqrt();
            assert!((celt_rsqrt(value) - expected).abs() <= 1e-6);
            assert!((celt_rsqrt_norm(value) - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn rcp_matches_inverse() {
        let values = [1.0_f32, -0.5, 2.0, -4.0];
        for &value in &values {
            let diff = (celt_rcp(value) - 1.0 / value).abs();
            assert!(diff <= 1e-6, "diff {} for value {}", diff, value);
        }
    }

    /// Port of `testdiv()` from `opus-c/celt/tests/test_unit_mathops.c` (float build).
    ///
    /// Validates that `celt_rcp()` returns values close to 1/x for a range of
    /// positive integers.
    #[test]
    fn rcp_matches_reference_harness() {
        for i in 1..=327_670 {
            let val = celt_rcp(i as f32);
            let prod = val * (i as f32);

            assert!(
                (prod - 1.0).abs() <= 0.00025,
                "div failed: 1/{} = {} (product = {})",
                i,
                val,
                prod
            );
        }
    }

    #[test]
    fn frac_div_helpers_match_division() {
        let samples = [(1.0_f32, 4.0_f32), (5.0, -2.0), (-6.0, 3.0)];
        for &(a, b) in &samples {
            assert!((frac_div32(a, b) - a / b).abs() <= f32::EPSILON * 2.0);
            assert!((frac_div32_q29(a, b) - a / b).abs() <= f32::EPSILON * 2.0);
        }
    }

    #[test]
    fn maxabs16_matches_manual_scan() {
        let samples = [0.0f32, -1.5, 3.25, -0.875, 2.0];
        assert!((celt_maxabs16(&samples) - 3.25).abs() <= f32::EPSILON);
        assert_eq!(celt_maxabs16(&[]), 0.0);
    }

    #[test]
    fn maxabs32_aliases_16() {
        let samples = [-4.0f32, 1.0, 2.5];
        assert_eq!(celt_maxabs32(&samples), celt_maxabs16(&samples));
    }

    #[test]
    fn cos_norm_matches_reference() {
        let inputs = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        for &input in &inputs {
            let expected = cosf(0.5 * PI * input);
            assert!((celt_cos_norm(input) - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn limit2_clamps_and_returns_hint() {
        let mut samples = [-3.5_f32, -2.0, -0.5, 0.75, 1.5, 3.75];
        let hint = opus_limit2_checkwithin1(&mut samples);
        assert!(!hint);
        assert_eq!(samples, [-2.0, -2.0, -0.5, 0.75, 1.5, 2.0]);

        let mut empty: [f32; 0] = [];
        assert!(opus_limit2_checkwithin1(&mut empty));
    }

    #[test]
    fn float2int16_matches_reference_scaling() {
        let input = [-2.0_f32, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0];
        let mut output = [0_i16; 9];
        celt_float2int16(&input, &mut output);

        assert_eq!(
            output,
            [
                -32_768, -32_768, -16_384, -8_192, 0, 8_192, 16_384, 32_767, 32_767
            ]
        );
    }

    #[test]
    fn float2int16_uses_saturating_round_to_nearest() {
        let input = [
            -1.001_f32, -1.000_03, -0.999_9, -0.500_3, -0.499_7, 0.499_7, 0.500_3, 0.999_9,
            1.000_03, 1.001,
        ];
        let mut output = [0_i16; 10];
        celt_float2int16(&input, &mut output);

        assert_eq!(output[0], -32_768);
        assert_eq!(output[1], -32_768);
        assert_eq!(output[8], 32_767);
        assert_eq!(output[9], 32_767);

        for (&input_sample, &output_sample) in input.iter().zip(&output) {
            let output_sample = i32::from(output_sample);
            assert!((-32_768..=32_767).contains(&output_sample));

            let scaled = (input_sample * CELT_SIG_SCALE).clamp(-32_768.0, 32_767.0);
            let diff = scaled - output_sample as f32;
            assert!(
                diff.abs() <= 0.500_1,
                "diff {} for input {} (scaled {})",
                diff,
                input_sample,
                scaled
            );
        }
    }

    #[test]
    #[should_panic(expected = "input and output slices must have the same length")]
    fn float2int16_panics_on_length_mismatch() {
        let input = [0.0_f32, 1.0];
        let mut output = [0_i16; 1];
        celt_float2int16(&input, &mut output);
    }

    #[test]
    fn celt_ilog2_matches_ec_ilog_minus_one() {
        let values = [
            1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1_024, 65_535, 65_536, 1_048_576,
        ];

        for &value in &values {
            assert_eq!(celt_ilog2(value), ec_ilog(value as u32) - 1);
        }
    }

    #[test]
    fn celt_zlog2_handles_non_positive_inputs() {
        assert_eq!(celt_zlog2(0), 0);
        assert_eq!(celt_zlog2(-123), 0);
        assert_eq!(celt_zlog2(1), celt_ilog2(1));
        assert_eq!(celt_zlog2(2_048), celt_ilog2(2_048));
    }

    #[test]
    fn isqrt32_matches_f64_reference() {
        let mut values = vec![
            1u32,
            2,
            3,
            4,
            7,
            9,
            15,
            16,
            24,
            36,
            64,
            65,
            255,
            256,
            257,
            1_000,
            65_535,
            65_536,
            1_048_575,
            u32::MAX,
        ];
        // Include additional edge cases near powers of two.
        for shift in 0..31 {
            let base = 1u32 << shift;
            values.push(base.saturating_sub(1));
            values.push(base);
            values.push(base.saturating_add(1));
        }

        values.sort_unstable();
        values.dedup();

        for value in values {
            let expected = (f64::from(value).sqrt().floor()) as u32;
            assert_eq!(isqrt32(value), expected, "value {}", value);
        }

        assert_eq!(isqrt32(0), 0);
    }
}
