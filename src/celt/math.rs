#![allow(dead_code)]

//! Mathematical helpers from the original CELT implementation.
//!
//! These are small functions with limited dependencies that can be ported in
//! isolation.  They are primarily used by the analysis and psychoacoustic
//! portions of the codec and map closely to the routines defined in
//! `celt/mathops.h` in the reference implementation.

use core::f32::consts::PI;

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

/// Fast arctangent approximation used by the psychoacoustic analysis code.
///
/// Mirrors the `fast_atan2f()` helper from `celt/mathops.h` when building the
/// float variant of CELT.  The approximation is accurate enough for the
/// heuristics that rely on it while avoiding the cost of calling into libm.
#[allow(clippy::many_single_char_names)]
pub(crate) fn fast_atan2f(y: f32, x: f32) -> f32 {
    const CA: f32 = 0.431_579_74;
    const CB: f32 = 0.678_484_03;
    const CC: f32 = 0.085_955_42;
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
    const INV_LN_2: f32 = 1.442_695_040_888_963_4;
    INV_LN_2 * logf(x)
}

/// Base-2 exponential used by CELT's float build.
#[inline]
pub(crate) fn celt_exp2(x: f32) -> f32 {
    const LN_2: f32 = 0.693_147_180_559_945_3;
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

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use libm::cosf;

    use super::isqrt32;

    use super::{celt_cos_norm, celt_div, celt_exp2, celt_log2, fast_atan2f};

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

    #[test]
    fn exp2_matches_std() {
        let values = [-5.0_f32, -1.0, 0.0, 0.25, 1.5, 4.0];
        for &value in &values {
            let diff = (celt_exp2(value) - value.exp2()).abs();
            assert!(diff <= 1e-6, "diff {} for value {}", diff, value);
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
    fn cos_norm_matches_reference() {
        let inputs = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        for &input in &inputs {
            let expected = cosf(0.5 * PI * input);
            assert!((celt_cos_norm(input) - expected).abs() <= 1e-6);
        }
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
