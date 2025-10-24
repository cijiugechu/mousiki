//! Port of the `silk_LPC_inverse_pred_gain` helper from the SILK reference
//! implementation.
//!
//! The routines in this module verify the stability of an LPC predictor and
//! compute the inverse prediction gain using fixed-point arithmetic. They mirror
//! the logic found in `silk/LPC_inv_pred_gain.c` and are used by both encoder
//! and decoder paths when validating LPC coefficient sets.

use core::cmp::max;

const QA: i32 = 24;
// 0.99975 == 3999 / 4000 with rounding.
const A_LIMIT: i32 = (((1i64 << QA) * 3999 + 2000) / 4000) as i32;
// 0.0001 == 1 / 10_000 with rounding.
const MIN_INV_GAIN_Q30: i32 = (((1i64 << 30) + 5_000) / 10_000) as i32;

/// Maximum LPC order handled by the fixed-point helpers.
pub const SILK_MAX_ORDER_LPC: usize = 24;

/// Computes the inverse prediction gain of LPC coefficients in Q12 precision.
///
/// The function returns the gain in Q30 format. A return value of zero indicates
/// that the coefficient set is unstable or violates the power-gain constraint
/// from the reference implementation.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
pub fn lpc_inverse_pred_gain(a_q12: &[i16]) -> i32 {
    let order = a_q12.len();
    if order == 0 {
        // Unity gain for zero-tap predictor.
        return 1 << 30;
    }
    debug_assert!(order > 0, "LPC order must be strictly positive");
    assert!(
        order <= SILK_MAX_ORDER_LPC,
        "order {} exceeds {}",
        order,
        SILK_MAX_ORDER_LPC
    );

    let mut a_tmp = [0i32; SILK_MAX_ORDER_LPC];
    let mut dc_resp = 0i32;

    for (idx, &coeff) in a_q12.iter().enumerate().take(order) {
        dc_resp += i32::from(coeff);
        a_tmp[idx] = i32::from(coeff) << (QA - 12);
    }

    if dc_resp >= 4096 {
        return 0;
    }

    lpc_inverse_pred_gain_qa(&mut a_tmp[..order])
}

fn lpc_inverse_pred_gain_qa(a_qa: &mut [i32]) -> i32 {
    let order = a_qa.len();
    debug_assert!(!a_qa.is_empty());

    let mut inv_gain_q30 = 1 << 30;

    for k in (1..order).rev() {
        let a_k = a_qa[k];
        if !(-A_LIMIT..=A_LIMIT).contains(&a_k) {
            return 0;
        }

        let rc_q31 = -shift_left(a_k, 31 - QA);
        let rc_mult1_q30 = (1 << 30) - smmul(rc_q31, rc_q31);

        inv_gain_q30 = shift_left(smmul(inv_gain_q30, rc_mult1_q30), 2);
        if inv_gain_q30 < MIN_INV_GAIN_Q30 {
            return 0;
        }

        let mult2q = 32 - leading_zeros_i32(rc_mult1_q30.abs());
        let rc_mult2 = inverse32_varq(rc_mult1_q30, mult2q + 30);

        for n in 0..((k + 1) >> 1) {
            let tmp1 = a_qa[n];
            let tmp2 = a_qa[k - n - 1];

            let updated_1 = update_coefficient(tmp1, tmp2, rc_q31, rc_mult2, mult2q);
            if let Some(value) = updated_1 {
                a_qa[n] = value;
            } else {
                return 0;
            }

            let updated_2 = update_coefficient(tmp2, tmp1, rc_q31, rc_mult2, mult2q);
            if let Some(value) = updated_2 {
                a_qa[k - n - 1] = value;
            } else {
                return 0;
            }
        }
    }

    let a0 = a_qa[0];
    if !(-A_LIMIT..=A_LIMIT).contains(&a0) {
        return 0;
    }

    let rc_q31 = -shift_left(a0, 31 - QA);
    let rc_mult1_q30 = (1 << 30) - smmul(rc_q31, rc_q31);

    inv_gain_q30 = shift_left(smmul(inv_gain_q30, rc_mult1_q30), 2);
    if inv_gain_q30 < MIN_INV_GAIN_Q30 {
        return 0;
    }

    inv_gain_q30
}

fn update_coefficient(
    original: i32,
    paired: i32,
    rc_q31: i32,
    rc_mult2: i32,
    mult2q: i32,
) -> Option<i32> {
    let adjustment = mul32_frac_q(paired, rc_q31, 31);
    let diff = sub_sat32(original, adjustment);
    let product = i64::from(diff) * i64::from(rc_mult2);
    let updated = rshift_round64(product, mult2q);
    if updated > i64::from(i32::MAX) || updated < i64::from(i32::MIN) {
        None
    } else {
        Some(updated as i32)
    }
}

fn inverse32_varq(b32: i32, qres: i32) -> i32 {
    debug_assert!(b32 != 0);
    debug_assert!(qres > 0);
    if b32 == 0 || qres <= 0 {
        return 0;
    }

    let abs_b32 = max(b32.abs(), 1);
    let b_headroom = leading_zeros_i32(abs_b32) - 1;
    let b32_nrm = shift_left(b32, b_headroom);

    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = b32_inv << 16;

    let err_q32 = ((1 << 29) - smulwb(b32_nrm, b32_inv)) << 3;
    result = smlaww(result, err_q32, b32_inv);

    let lshift = 61 - b_headroom - qres;
    if lshift <= 0 {
        saturating_lshift(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

fn smlaww(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((i64::from(b) * i64::from(c)) >> 16) as i32)
}

fn shift_left(value: i32, shift: i32) -> i32 {
    debug_assert!((0..32).contains(&shift));
    value.wrapping_shl(shift as u32)
}

fn saturating_lshift(value: i32, shift: i32) -> i32 {
    debug_assert!(shift >= 0);
    let shifted = i64::from(value) << shift;
    shifted.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn sub_sat32(a: i32, b: i32) -> i32 {
    let diff = i64::from(a) - i64::from(b);
    if diff > i64::from(i32::MAX) {
        i32::MAX
    } else if diff < i64::from(i32::MIN) {
        i32::MIN
    } else {
        diff as i32
    }
}

fn mul32_frac_q(a: i32, b: i32, q: i32) -> i32 {
    debug_assert!(q > 0);
    rshift_round64(i64::from(a) * i64::from(b), q) as i32
}

fn rshift_round64(value: i64, shift: i32) -> i64 {
    debug_assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn leading_zeros_i32(value: i32) -> i32 {
    if value == 0 {
        32
    } else {
        value.leading_zeros() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::{
        inverse32_varq,
        lpc_inverse_pred_gain,
        mul32_frac_q,
        smmul,
        sub_sat32,
        QA,
        SILK_MAX_ORDER_LPC,
    };

    #[test]
    fn computes_gain_for_simple_predictor() {
        let coeffs = [2048i16];
        assert_eq!(lpc_inverse_pred_gain(&coeffs), 805_306_368);
    }

    #[test]
    fn computes_gain_for_four_tap_predictor() {
        let coeffs = [1024, -512, 256, -128];
        assert_eq!(lpc_inverse_pred_gain(&coeffs), 1_006_430_076);
    }

    #[test]
    fn detects_unstable_dc_response() {
        let coeffs = [4096i16];
        assert_eq!(lpc_inverse_pred_gain(&coeffs), 0);
    }

    #[test]
    fn zero_tap_predictor_returns_unity_gain() {
        let coeffs: [i16; 0] = [];
        assert_eq!(lpc_inverse_pred_gain(&coeffs), 1 << 30);
    }

    #[test]
    fn matches_reference_for_asymmetric_predictor() {
        let coeffs = [3000, -2000, 1000, -500];
        assert_eq!(lpc_inverse_pred_gain(&coeffs), 691_862_120);
    }

    #[test]
    fn handles_max_order_predictor() {
        let coeffs = [0i16; SILK_MAX_ORDER_LPC];
        assert_eq!(lpc_inverse_pred_gain(&coeffs), 1 << 30);
    }

    #[test]
    fn helper_macros_match_reference_behaviour() {
        assert_eq!(smmul(1 << 30, 1 << 30), 1 << 28);
        assert_eq!(sub_sat32(i32::MAX, -1), i32::MAX);
        assert_eq!(mul32_frac_q(1 << QA, 1 << (31 - QA), 31), 1);
        assert_eq!(inverse32_varq(1 << 30, 31), 1);
    }
}
