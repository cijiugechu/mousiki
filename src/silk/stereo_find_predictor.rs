//! Port of `silk/stereo_find_predictor.c`.
//!
//! Computes the least-squares predictor that maps the low- and high-pass
//! filtered mid channel onto the side channel while updating the smoothed mid
//! and residual magnitudes tracked by the encoder's stereo state. The routine
//! mirrors the fixed-point arithmetic and Q-domain bookkeeping of the
//! reference C implementation.

use crate::silk::inner_prod_aligned::inner_prod_aligned_scale;
use crate::silk::sum_sqr_shift::sum_sqr_shift;

/// Find the mid/side stereo predictor and update the smoothed mid and residual
/// amplitudes.
///
/// Returns a tuple containing the predictor in Q13 and the ratio between the
/// smoothed residual and mid magnitudes in Q14, matching the behaviour of the
/// reference `silk_stereo_find_predictor` routine.
pub fn stereo_find_predictor(
    x: &[i16],
    y: &[i16],
    mid_res_amp_q0: &mut [i32; 2],
    // Expected range: 0..=32767; used by SMLAWB as a Q16-like smoothing factor.
    smooth_coef_q16: i32,
) -> (i32, i32) {
    debug_assert_eq!(x.len(), y.len(), "input vectors must have equal lengths");
    if x.len() != y.len() {
        return (0, 0);
    }

    if x.is_empty() {
        return (0, 0);
    }

    let (mut nrgx, scale1) = sum_sqr_shift(x);
    let (mut nrgy, scale2) = sum_sqr_shift(y);
    let mut scale = scale1.max(scale2);
    if scale & 1 != 0 {
        scale += 1;
    }

    nrgy = rshift32(nrgy, scale - scale2);
    nrgx = rshift32(nrgx, scale - scale1);
    if nrgx < 1 {
        nrgx = 1;
    }

    let corr = inner_prod_aligned_scale(x, y, scale);
    let mut pred_q13 = div32_varq(corr, nrgx, 13);
    pred_q13 = pred_q13.clamp(-(1 << 14), 1 << 14);
    let pred2_q10 = smulwb(pred_q13, pred_q13);

    let smooth_coef_q16 = smooth_coef_q16.max(pred2_q10.abs());
    debug_assert!(smooth_coef_q16 < 32768);

    scale >>= 1;

    let target_mid = lshift32(sqrt_approx(nrgx), scale);
    mid_res_amp_q0[0] = smlawb(
        mid_res_amp_q0[0],
        target_mid.wrapping_sub(mid_res_amp_q0[0]),
        smooth_coef_q16,
    );

    nrgy = sub_lshift32(nrgy, smulwb(corr, pred_q13), 4);
    nrgy = add_lshift32(nrgy, smulwb(nrgx, pred2_q10), 6);

    let target_res = lshift32(sqrt_approx(nrgy), scale);
    mid_res_amp_q0[1] = smlawb(
        mid_res_amp_q0[1],
        target_res.wrapping_sub(mid_res_amp_q0[1]),
        smooth_coef_q16,
    );

    let mut ratio_q14 = div32_varq(mid_res_amp_q0[1], mid_res_amp_q0[0].max(1), 14);
    ratio_q14 = ratio_q14.clamp(0, 32767);

    (pred_q13, ratio_q14)
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let product = (i64::from(b) * i64::from(c as i16)) >> 16;
    a.wrapping_add(product as i32)
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from((a as i16).wrapping_mul(b as i16))
}

fn lshift32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else {
        value.wrapping_shl(shift as u32)
    }
}

fn rshift32(value: i32, shift: i32) -> i32 {
    if shift <= 0 { value } else { value >> shift }
}

fn add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b.wrapping_shl(shift as u32))
}

fn sub_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_sub(b.wrapping_shl(shift as u32))
}

const SQRT_COEF_Q7: i32 = 213; // Matches SILK polynomial coefficient.

fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }

    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32768 } else { 46214 };
    y >>= lz >> 1;
    smlawb(y, y, smulbb(SQRT_COEF_Q7, frac_q7))
}

fn clz_frac(x: i32) -> (i32, i32) {
    let ux = x as u32;
    let lz = ux.leading_zeros() as i32;
    let rotate = ((24 - lz) & 31) as u32;
    let frac = (ux.rotate_right(rotate) & 0x7f) as i32;
    (lz, frac)
}

fn div32_varq(a32: i32, b32: i32, q_res: i32) -> i32 {
    assert!(b32 != 0, "denominator must be non-zero");
    assert!(q_res >= 0, "result Q-domain must be non-negative");

    let abs_a = if a32 == i32::MIN { i32::MAX } else { a32.abs() };
    let abs_b = if b32 == i32::MIN { i32::MAX } else { b32.abs() };
    let a_headroom = clz32(abs_a) - 1;
    let mut a_norm = lshift32(a32, a_headroom);
    let b_headroom = clz32(abs_b) - 1;
    let b_norm = lshift32(b32, b_headroom);

    let denom16 = rshift32(b_norm, 16);
    debug_assert!(denom16 != 0, "normalized denominator high word must be non-zero");
    let b_inv = div32_16(i32::MAX >> 2, denom16);

    let mut result = smulwb(a_norm, b_inv);

    let correction = lshift_ovflw(smmul(b_norm, result), 3);
    a_norm = sub32_ovflw(a_norm, correction);
    result = smlawb(result, a_norm, b_inv);

    let lshift = 29 + a_headroom - b_headroom - q_res;
    if lshift < 0 {
        lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        rshift32(result, lshift)
    } else {
        0
    }
}

fn clz32(value: i32) -> i32 {
    if value == 0 {
        32
    } else {
        (value as u32).leading_zeros() as i32
    }
}

fn div32_16(a: i32, b: i32) -> i32 {
    a / b
}

fn sub32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

fn lshift_ovflw(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else {
        value.wrapping_shl(shift as u32)
    }
}

fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

fn lshift_sat32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return value;
    }

    let shifted = (i64::from(value)) << shift;
    if shifted > i64::from(i32::MAX) {
        i32::MAX
    } else if shifted < i64::from(i32::MIN) {
        i32::MIN
    } else {
        shifted as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_predictor_for_known_vectors() {
        let x = [1000, -2000, 3000, -4000];
        let y = [500, -600, 700, -800];
        let mut mid_res = [100_000, 90_000];
        let (pred, ratio) = stereo_find_predictor(&x, &y, &mut mid_res, 5_000);
        assert_eq!(pred, 1911);
        assert_eq!(ratio, 14_683);
        assert_eq!(mid_res, [92_784, 83_155]);
    }

    #[test]
    fn handles_large_mid_side_amplitudes() {
        let x = [15_000, -16_000, 17_000, -18_000, 19_000, -20_000];
        let y = [8_000, -9_000, 10_000, -11_000, 12_000, -13_000];
        let mut mid_res = [80_000, 85_000];
        let (pred, ratio) = stereo_find_predictor(&x, &y, &mut mid_res, 3_000);
        assert_eq!(pred, 4_946);
        assert_eq!(ratio, 16_987);
        assert_eq!(mid_res, [78_291, 81_177]);
    }

    #[test]
    fn clamps_ratio_when_mid_energy_is_small() {
        let x = [32_760, -32_760, 32_760, -32_760];
        let y = [1_000, 2_000, -3_000, -4_000];
        let mut mid_res = [120_000, 130_000];
        let (pred, ratio) = stereo_find_predictor(&x, &y, &mut mid_res, 1_000);
        assert_eq!(pred, 0);
        assert_eq!(ratio, 17_612);
        assert_eq!(mid_res, [119_165, 128_099]);
    }
}
