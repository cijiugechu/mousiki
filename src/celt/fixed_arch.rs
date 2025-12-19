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

use super::float_cast::{float2int, CELT_SIG_SCALE};
use super::types::{FixedCeltSig, FixedOpusRes, FixedOpusVal16, FixedOpusVal32};
#[cfg(not(feature = "enable_res24"))]
use super::float_cast::float2int16;

/// Number of fractional bits in the fixed-point `celt_sig` representation.
///
/// Mirrors `SIG_SHIFT` in `opus-c/celt/arch.h` when `FIXED_POINT` is enabled.
pub(crate) const SIG_SHIFT: u32 = 12;

/// Safe saturation limit for 32-bit signals.
///
/// Mirrors `SIG_SAT` in `opus-c/celt/arch.h`.
pub(crate) const SIG_SAT: FixedCeltSig = 536_870_911;

/// Scaling applied to unit-norm MDCT vectors in fixed-point builds.
///
/// Mirrors `NORM_SCALING` in `opus-c/celt/arch.h`.
pub(crate) const NORM_SCALING: FixedOpusVal16 = 16_384;

/// Bit shift used for CELT gain values.
///
/// Mirrors `DB_SHIFT` in `opus-c/celt/arch.h` when `FIXED_POINT` is enabled.
pub(crate) const DB_SHIFT: u32 = 24;

/// Q15 representation of 1.0.
pub(crate) const Q15_ONE: FixedOpusVal16 = i16::MAX;

/// Q31 representation of 1.0.
pub(crate) const Q31_ONE: FixedOpusVal32 = i32::MAX;

/// Smallest non-zero value in fixed-point builds.
pub(crate) const EPSILON: FixedOpusVal16 = 1;

/// Placeholder for "very small" fixed-point values.
pub(crate) const VERY_SMALL: FixedOpusVal16 = 0;

/// Largest 16-bit fixed-point value.
pub(crate) const VERY_LARGE16: FixedOpusVal16 = i16::MAX;

#[cfg(feature = "enable_res24")]
pub(crate) const RES_SHIFT: u32 = 8;
#[cfg(not(feature = "enable_res24"))]
pub(crate) const RES_SHIFT: u32 = 0;

/// Maximum bit depth allowed by the `opus_res` representation.
///
/// Mirrors `MAX_ENCODING_DEPTH` from `opus-c/celt/arch.h` for the RES16 build.
#[cfg(feature = "enable_res24")]
pub(crate) const MAX_ENCODING_DEPTH: u32 = 24;
#[cfg(not(feature = "enable_res24"))]
pub(crate) const MAX_ENCODING_DEPTH: u32 = 16;

/// Converts a fixed-point `opus_res` sample to a floating-point sample in `[-1, 1)`.
///
/// Mirrors the `RES2FLOAT()` macro for the selected fixed-point build.
#[inline]
pub(crate) fn res2float(res: FixedOpusRes) -> f32 {
    #[cfg(feature = "enable_res24")]
    {
        (res as f32) * (1.0 / (CELT_SIG_SCALE * 256.0))
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        f32::from(res) * (1.0 / CELT_SIG_SCALE)
    }
}

/// Converts a floating-point sample in `[-1, 1]` to a fixed-point `opus_res` sample.
///
/// Mirrors the `FLOAT2RES()` macro for the selected fixed-point build.
#[inline]
pub(crate) fn float2res(sample: f32) -> FixedOpusRes {
    #[cfg(feature = "enable_res24")]
    {
        float2int(CELT_SIG_SCALE * 256.0 * sample)
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        float2int16(sample)
    }
}

/// Converts a floating-point sample in `[-1, 1]` to a fixed-point `celt_sig` sample.
///
/// Mirrors the `FLOAT2SIG()` macro in `opus-c/celt/arch.h`.
#[inline]
pub(crate) fn float2sig(sample: f32) -> FixedCeltSig {
    let scale = CELT_SIG_SCALE * (1_u32 << SIG_SHIFT) as f32;
    float2int(scale * sample)
}

#[inline]
pub(crate) fn sat16(x: i32) -> FixedOpusVal16 {
    if x > 32_767 {
        32_767
    } else if x < -32_768 {
        -32_768
    } else {
        x as FixedOpusVal16
    }
}

#[inline]
fn extend32(x: FixedOpusVal16) -> FixedCeltSig {
    FixedCeltSig::from(x)
}

#[inline]
fn shl32(x: FixedCeltSig, shift: u32) -> FixedCeltSig {
    debug_assert!(shift < 32);
    ((x as u32) << shift) as FixedCeltSig
}

#[inline]
fn shr32(x: FixedCeltSig, shift: u32) -> FixedCeltSig {
    debug_assert!(shift < 32);
    x >> shift
}

/// 32-bit arithmetic right shift with round-to-nearest behaviour.
///
/// Mirrors `PSHR32()` from `opus-c/celt/fixed_generic.h`.
#[inline]
pub(crate) fn pshr32(x: FixedCeltSig, shift: u32) -> FixedCeltSig {
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
pub(crate) fn sig2word16(sig: FixedCeltSig) -> FixedOpusVal16 {
    sat16(pshr32(sig, SIG_SHIFT))
}

/// Convert a fixed-point `celt_sig` sample to a fixed-point `opus_res` sample
/// (`i16` for the RES16 build, `i32` for the RES24 build).
#[inline]
pub(crate) fn sig2res(sig: FixedCeltSig) -> FixedOpusRes {
    #[cfg(feature = "enable_res24")]
    {
        pshr32(sig, SIG_SHIFT - RES_SHIFT)
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        sig2word16(sig)
    }
}

/// Convert a fixed-point `opus_res` sample to a fixed-point `celt_sig` sample.
#[inline]
pub(crate) fn res2sig(res: FixedOpusRes) -> FixedCeltSig {
    #[cfg(feature = "enable_res24")]
    {
        shl32(res, SIG_SHIFT - RES_SHIFT)
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        shl32(extend32(res), SIG_SHIFT - RES_SHIFT)
    }
}

/// Convert a 16-bit PCM sample to a fixed-point `celt_sig` sample.
#[inline]
pub(crate) fn int16tosig(sample: FixedOpusVal16) -> FixedCeltSig {
    shl32(extend32(sample), SIG_SHIFT)
}

/// Convert a 24-bit PCM sample to a fixed-point `celt_sig` sample.
#[inline]
pub(crate) fn int24tosig(sample: i32) -> FixedCeltSig {
    shl32(sample, SIG_SHIFT - 8)
}

/// Convert a fixed-point `opus_res` sample to 16-bit PCM.
#[inline]
pub(crate) fn res2int16(res: FixedOpusRes) -> FixedOpusVal16 {
    #[cfg(feature = "enable_res24")]
    {
        sat16(pshr32(res, RES_SHIFT))
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        res
    }
}

/// Convert a fixed-point `opus_res` sample to a fixed-point `opus_val16`.
#[inline]
pub(crate) fn res2val16(res: FixedOpusRes) -> FixedOpusVal16 {
    res2int16(res)
}

/// Convert a fixed-point `opus_res` sample to 24-bit PCM stored in an `i32`.
#[inline]
pub(crate) fn res2int24(res: FixedOpusRes) -> i32 {
    #[cfg(feature = "enable_res24")]
    {
        res
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        shl32(extend32(res), 8)
    }
}

/// Convert a 16-bit PCM sample to a fixed-point `opus_res` sample.
#[inline]
pub(crate) fn int16tores(sample: FixedOpusVal16) -> FixedOpusRes {
    #[cfg(feature = "enable_res24")]
    {
        shl32(extend32(sample), RES_SHIFT)
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        sample
    }
}

/// Convert a 24-bit PCM sample stored in an `i32` to a fixed-point `opus_res`
/// sample, using the same rounding and saturation semantics as the C macros.
#[inline]
pub(crate) fn int24tores(sample: i32) -> FixedOpusRes {
    #[cfg(feature = "enable_res24")]
    {
        sample
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        sat16(pshr32(sample, 8))
    }
}

/// Multiply a Q15 value by a fixed-point `opus_res` sample.
#[inline]
pub(crate) fn mult16_res_q15(a: FixedOpusVal16, b: FixedOpusRes) -> FixedCeltSig {
    let product = i64::from(a) * i64::from(b);
    (product >> 15) as FixedCeltSig
}

/// Addition of two `opus_res` samples.
#[inline]
pub(crate) fn add_res(a: FixedOpusRes, b: FixedOpusRes) -> FixedOpusRes {
    #[cfg(feature = "enable_res24")]
    {
        let sum = i64::from(a) + i64::from(b);
        debug_assert!(
            (i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&sum),
            "opus_res addition overflow: {a} + {b}"
        );
        sum as i32
    }
    #[cfg(not(feature = "enable_res24"))]
    {
        sat16(i32::from(a) + i32::from(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "enable_res24")]
    const TEST_VALUES: &[FixedOpusRes] = &[
        -8_388_608i32,
        -1_234_432,
        -256,
        -1,
        0,
        1,
        256,
        1_234_432,
        8_388_352,
        8_388_607,
    ];

    #[cfg(not(feature = "enable_res24"))]
    const TEST_VALUES: &[FixedOpusRes] = &[-32_768i16, -12_345, -1, 0, 1, 12_345, 32_767];

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
    fn res_sig_roundtrip_is_exact_for_selected_res() {
        for &value in TEST_VALUES {
            assert_eq!(sig2res(res2sig(value)), value);
        }
    }

    #[test]
    fn pcm_to_sig_conversions_share_the_same_scale() {
        let unit = 1.0 / CELT_SIG_SCALE;
        assert_eq!(float2sig(unit), int16tosig(1));
        assert_eq!(float2sig(-unit), int16tosig(-1));

        for &value in &[-8_388_608i32, -1_234_432, -256, 0, 256, 1_234_432, 8_388_352] {
            assert_eq!(int24tosig(value), res2sig(int24tores(value)));
        }
    }

    #[test]
    fn int24_conversions_roundtrip_for_byte_aligned_values() {
        for &value in &[-8_388_608i32, -1_234_432, -256, 0, 256, 1_234_432, 8_388_352] {
            let res = int24tores(value);
            let back = res2int24(res);
            assert_eq!(back, value);
        }
    }

    #[test]
    #[cfg(not(feature = "enable_res24"))]
    fn int24_to_res_saturates() {
        assert_eq!(int24tores(8_388_607), 32_767);
        assert_eq!(int24tores(-8_388_608), -32_768);
        assert_eq!(int24tores(9_000_000), 32_767);
        assert_eq!(int24tores(-9_000_000), -32_768);
    }

    #[test]
    #[cfg(not(feature = "enable_res24"))]
    fn add_res_saturates_like_sat16() {
        assert_eq!(add_res(30_000, 10_000), 32_767);
        assert_eq!(add_res(-30_000, -10_000), -32_768);
        assert_eq!(add_res(10_000, -3_000), 7_000);
    }

    #[test]
    #[cfg(feature = "enable_res24")]
    fn add_res_behaves_like_add32_for_res24() {
        assert_eq!(add_res(1_000_000, 2_000_000), 3_000_000);
        assert_eq!(add_res(-1_000_000, 500_000), -500_000);
    }

    #[test]
    fn float_res_round_trips_on_exact_grid_points() {
        for &value in TEST_VALUES {
            let sample = res2float(value);
            assert_eq!(float2res(sample), value);
        }
    }

    #[test]
    fn mult16_res_q15_matches_scaled_product() {
        let a: FixedOpusVal16 = 16_384;
        let b = int16tores(10_000);
        let expected = (i64::from(a) * i64::from(b)) >> 15;
        assert_eq!(mult16_res_q15(a, b), expected as FixedCeltSig);
    }

    #[test]
    #[cfg(feature = "enable_res24")]
    fn res24_to_int16_rounds_and_saturates_like_reference() {
        assert_eq!(res2int16(0), 0);
        assert_eq!(res2int16(32_767i32 << RES_SHIFT), 32_767);
        assert_eq!(res2int16(-32_768i32 << RES_SHIFT), -32_768);

        // One past full scale must saturate after shifting down to i16.
        assert_eq!(res2int16(32_768i32 << RES_SHIFT), 32_767);
        assert_eq!(res2int16((-32_769i32) << RES_SHIFT), -32_768);
    }
}
