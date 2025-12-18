#![allow(dead_code)]

//! Fixed-point architecture helpers derived from CELT's `arch.h`.
//!
//! The upstream C implementation uses a set of macros to define the Q formats
//! and conversions between:
//! - `celt_sig` (internal CELT signal, Q27 in fixed builds),
//! - `opus_res` (Opus "resolution" samples, either 16-bit or 24-bit integers),
//! - public PCM integer formats.
//!
//! The Rust port still uses the floating-point signal graph even when the
//! `fixed_point` feature is enabled, but future fixed-point DSP ports will
//! reuse these constants and integer conversion helpers to stay aligned with
//! the reference semantics.

/// Number of fractional bits in the fixed-point `celt_sig` representation.
///
/// Mirrors `SIG_SHIFT` in `opus-c/celt/arch.h` when `FIXED_POINT` is enabled.
pub(crate) const SIG_SHIFT: u32 = 12;

/// Safe saturation limit for 32-bit signals.
///
/// Mirrors `SIG_SAT` in `opus-c/celt/arch.h`.
pub(crate) const SIG_SAT: i32 = 536_870_911;

/// Scaling applied to unit-norm MDCT vectors in fixed-point builds.
///
/// Mirrors `NORM_SCALING` in `opus-c/celt/arch.h`.
pub(crate) const NORM_SCALING: i16 = 16_384;

/// Bit shift used for CELT gain values.
///
/// Mirrors `DB_SHIFT` in `opus-c/celt/arch.h` when `FIXED_POINT` is enabled.
pub(crate) const DB_SHIFT: u32 = 24;

/// Q15 representation of 1.0.
pub(crate) const Q15_ONE: i16 = 32_767;

/// Q31 representation of 1.0.
pub(crate) const Q31_ONE: i32 = 2_147_483_647;

/// `opus-c`'s non-`ENABLE_RES24` fixed-point build uses 16-bit `opus_res`.
pub(crate) const RES_SHIFT: u32 = 0;

/// Maximum bit depth allowed by the `opus_res` representation.
///
/// Mirrors `MAX_ENCODING_DEPTH` from `opus-c/celt/arch.h` for the RES16 build.
pub(crate) const MAX_ENCODING_DEPTH: u32 = 16;

#[inline]
pub(crate) fn sat16(x: i32) -> i16 {
    if x > 32_767 {
        32_767
    } else if x < -32_768 {
        -32_768
    } else {
        x as i16
    }
}

#[inline]
fn extend32(x: i16) -> i32 {
    i32::from(x)
}

#[inline]
fn shl32(x: i32, shift: u32) -> i32 {
    debug_assert!(shift < 32);
    ((x as u32) << shift) as i32
}

#[inline]
fn shr32(x: i32, shift: u32) -> i32 {
    debug_assert!(shift < 32);
    x >> shift
}

/// 32-bit arithmetic right shift with round-to-nearest behaviour.
///
/// Mirrors `PSHR32()` from `opus-c/celt/fixed_generic.h`.
#[inline]
pub(crate) fn pshr32(x: i32, shift: u32) -> i32 {
    if shift == 0 {
        return x;
    }
    let bias = shl32(1, shift - 1);
    shr32(x.wrapping_add(bias), shift)
}

/// Convert a fixed-point `celt_sig` sample to a 16-bit PCM sample.
///
/// Mirrors `SIG2WORD16()` from `opus-c/celt/fixed_generic.h` (and consequently
/// the `SIG2RES()` macro for the RES16 build).
#[inline]
pub(crate) fn sig2word16(sig: i32) -> i16 {
    sat16(pshr32(sig, SIG_SHIFT))
}

/// Convert a fixed-point `celt_sig` sample to a fixed-point `opus_res` sample
/// (`i16` for the RES16 build).
#[inline]
pub(crate) fn sig2res(sig: i32) -> i16 {
    sig2word16(sig)
}

/// Convert a fixed-point `opus_res` sample to a fixed-point `celt_sig` sample.
#[inline]
pub(crate) fn res2sig(res: i16) -> i32 {
    shl32(extend32(res), SIG_SHIFT - RES_SHIFT)
}

/// Convert a fixed-point `opus_res` sample (RES16) to 16-bit PCM.
#[inline]
pub(crate) fn res2int16(res: i16) -> i16 {
    res
}

/// Convert a fixed-point `opus_res` sample (RES16) to 24-bit PCM stored in an
/// `i32`.
#[inline]
pub(crate) fn res2int24(res: i16) -> i32 {
    shl32(extend32(res), 8)
}

/// Convert a 16-bit PCM sample to a fixed-point `opus_res` sample (RES16).
#[inline]
pub(crate) fn int16tores(sample: i16) -> i16 {
    sample
}

/// Convert a 24-bit PCM sample stored in an `i32` to a fixed-point `opus_res`
/// sample (RES16), using the same rounding and saturation semantics as the C
/// macros.
#[inline]
pub(crate) fn int24tores(sample: i32) -> i16 {
    sat16(pshr32(sample, 8))
}

/// Saturating addition of two RES16 samples.
#[inline]
pub(crate) fn add_res(a: i16, b: i16) -> i16 {
    sat16(i32::from(a) + i32::from(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pshr32_matches_reference_biasing() {
        assert_eq!(pshr32(0, 0), 0);
        assert_eq!(pshr32(1, 0), 1);

        // Positive values: round-to-nearest (ties up because of the +bias).
        assert_eq!(pshr32(3, 1), 2);
        assert_eq!(pshr32(2, 1), 1);
        assert_eq!(pshr32(1, 1), 1);

        // Negative values: the reference macro adds a positive bias before the
        // arithmetic shift, which rounds toward zero in half-way cases.
        assert_eq!(pshr32(-3, 1), -1);
        assert_eq!(pshr32(-2, 1), -1);
        assert_eq!(pshr32(-1, 1), 0);
    }

    #[test]
    fn sig2word16_saturates_after_scaling() {
        assert_eq!(sig2word16(0), 0);
        assert_eq!(sig2word16(shl32(32_767, SIG_SHIFT)), 32_767);
        assert_eq!(sig2word16(shl32(-32_768, SIG_SHIFT)), -32_768);

        // One past full scale must saturate.
        assert_eq!(sig2word16(shl32(32_768, SIG_SHIFT)), 32_767);
        assert_eq!(sig2word16(shl32(-32_769, SIG_SHIFT)), -32_768);
    }

    #[test]
    fn res_sig_roundtrip_is_exact_for_res16() {
        for &value in &[-32_768i16, -12_345, -1, 0, 1, 12_345, 32_767] {
            assert_eq!(sig2res(res2sig(value)), value);
        }
    }

    #[test]
    fn int24_conversions_roundtrip_for_byte_aligned_values() {
        for &value in &[
            -8_388_608i32,
            -1_234_432,
            -256,
            0,
            256,
            1_234_432,
            8_388_352,
        ] {
            let res = int24tores(value);
            let back = res2int24(res);
            assert_eq!(back, value);
        }
    }

    #[test]
    fn int24_to_res_saturates() {
        assert_eq!(int24tores(8_388_607), 32_767);
        assert_eq!(int24tores(-8_388_608), -32_768);
        assert_eq!(int24tores(9_000_000), 32_767);
        assert_eq!(int24tores(-9_000_000), -32_768);
    }

    #[test]
    fn add_res_saturates_like_sat16() {
        assert_eq!(add_res(30_000, 10_000), 32_767);
        assert_eq!(add_res(-30_000, -10_000), -32_768);
        assert_eq!(add_res(10_000, -3_000), 7_000);
    }
}
