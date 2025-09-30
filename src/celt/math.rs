#![allow(dead_code)]

//! Mathematical helpers from the original CELT implementation.
//!
//! These are small functions with limited dependencies that can be ported in
//! isolation.  They are primarily used by the analysis and psychoacoustic
//! portions of the codec and map closely to the routines defined in
//! `celt/mathops.h` in the reference implementation.

use core::f32::consts::PI;

use libm::{expf, logf};

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

#[cfg(test)]
mod tests {
    use super::{celt_exp2, celt_log2, fast_atan2f};

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
}
