#![allow(dead_code)]

//! Floating-point to integer conversion helpers from `celt/float_cast.h`.
//!
//! The original header provides a family of macros that round floating-point
//! samples to integral types using the rounding behaviour guaranteed by C99's
//! `lrintf()`/`lrint()` functions.  CELT relies on these helpers when bridging
//! between the float API and the fixed-point internals.  The Rust port exposes
//! equivalent functions so that other translated modules can depend on the same
//! rounding semantics without reimplementing the details.

use libm::rintf;

/// Scaling factor used by CELT to map floating-point samples to its internal
/// fixed-point representation.
pub(crate) const CELT_SIG_SCALE: f32 = 32_768.0;

/// Rounds a `f32` to the nearest `i32`, matching the behaviour of the
/// `float2int()` helper from the C implementation.
///
/// The reference code delegates to `lrintf()` when it is available, which
/// rounds to the nearest integer using the current floating-point rounding
/// mode (round-to-nearest-even in practice).  Rust's `as` conversion from
/// `f32` to `i32` already saturates on overflow, so the implementation simply
/// applies `rintf()` before casting.
#[must_use]
pub(crate) fn float2int(value: f32) -> i32 {
    rintf(value) as i32
}

/// Converts a floating-point sample to a signed 16-bit integer using CELT's
/// canonical scaling and rounding behaviour.
///
/// Mirrors the `FLOAT2INT16()` macro in `float_cast.h` by scaling the input,
/// clamping it to the representable range, and delegating to [`float2int`] for
/// the final rounding step.
#[must_use]
pub(crate) fn float2int16(value: f32) -> i16 {
    let scaled = (value * CELT_SIG_SCALE).clamp(-32_768.0, 32_767.0);
    float2int(scaled) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float2int_rounds_to_nearest_even() {
        // Half-way cases should round to the nearest even integer, matching the
        // default IEEE 754 rounding mode used by the C implementation.
        assert_eq!(float2int(1.5), 2);
        assert_eq!(float2int(2.5), 2);
        assert_eq!(float2int(-1.5), -2);
        assert_eq!(float2int(-2.5), -2);
    }

    #[test]
    fn float2int16_clamps_to_i16_range() {
        // Values outside the 16-bit range are clamped before rounding.
        assert_eq!(float2int16(2.0), 32_767);
        assert_eq!(float2int16(-2.0), -32_768);
        // In-range values follow the same rounding mode as float2int().
        assert_eq!(float2int16(0.500_1 / CELT_SIG_SCALE), 1);
        assert_eq!(float2int16(-0.500_1 / CELT_SIG_SCALE), -1);
    }
}
