//! Port of `silk/fixed/schur64_FIX.c`.
//!
//! Mirrors the high-precision Schur recursion used by the SILK noise-shaping
//! analysis. Compared to the faster `silk_schur` helper, this variant keeps the
//! full 64-bit intermediate products to improve the accuracy of the reflection
//! coefficients and residual energy.

use crate::silk::lpc_inv_pred_gain::SILK_MAX_ORDER_LPC;
use crate::silk::stereo_find_predictor::div32_varq;

const ALMOST_ONE_Q16: i32 = 64_881; // SILK_FIX_CONST(0.99, 16)

/// High-precision Schur recursion matching `silk_schur64`.
///
/// `rc_q16` stores the reflection coefficients in Q16 while the return value
/// provides the final residual energy clamped to at least one.
pub fn schur64(rc_q16: &mut [i32], c: &[i32], order: usize) -> i32 {
    assert!(
        order <= SILK_MAX_ORDER_LPC,
        "order must not exceed SILK_MAX_ORDER_LPC"
    );
    assert!(
        c.len() > order,
        "correlation buffer must contain order + 1 elements"
    );
    assert!(
        rc_q16.len() >= order,
        "reflection coefficient buffer must match order"
    );

    if c[0] <= 0 {
        for coeff in rc_q16.iter_mut().take(order) {
            *coeff = 0;
        }
        return 0;
    }

    if order == 0 {
        return c[0].max(1);
    }

    let mut corr = [[0i32; 2]; SILK_MAX_ORDER_LPC + 1];
    for (dst, &src) in corr.iter_mut().zip(c.iter()).take(order + 1) {
        dst[0] = src;
        dst[1] = src;
    }

    let mut k = 0;
    while k < order {
        if corr[k + 1][0].abs() >= corr[0][1] {
            rc_q16[k] = if corr[k + 1][0] > 0 {
                -ALMOST_ONE_Q16
            } else {
                ALMOST_ONE_Q16
            };
            k += 1;
            break;
        }

        let rc_tmp_q31 = div32_varq(-corr[k + 1][0], corr[0][1], 31);
        rc_q16[k] = rshift_round(rc_tmp_q31, 15);

        for n in 0..(order - k) {
            let ctmp1_q30 = corr[n + k + 1][0];
            let ctmp2_q30 = corr[n][1];
            corr[n + k + 1][0] = ctmp1_q30.wrapping_add(smmul(lshift(ctmp2_q30, 1), rc_tmp_q31));
            corr[n][1] = ctmp2_q30.wrapping_add(smmul(lshift(ctmp1_q30, 1), rc_tmp_q31));
        }

        k += 1;
    }

    while k < order {
        rc_q16[k] = 0;
        k += 1;
    }

    corr[0][1].max(1)
}

fn lshift(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else {
        value.wrapping_shl(shift as u32)
    }
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else if shift == 1 {
        (value >> 1).wrapping_add(value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_non_positive_energy() {
        let mut rc = [123; 3];
        let energy = schur64(&mut rc, &[0, 1, 2, 3], 3);
        assert_eq!(energy, 0);
        assert!(rc.iter().all(|&value| value == 0));
    }

    #[test]
    fn matches_reference_first_order_case() {
        let mut rc = [0i32; 1];
        let energy = schur64(&mut rc, &[100_000, 5_000], 1);
        assert_eq!(rc[0], -3_277);
        assert_eq!(energy, 99_750);
    }

    #[test]
    fn matches_reference_second_order_case() {
        let mut rc = [0i32; 2];
        let energy = schur64(&mut rc, &[150_000, -20_000, 5_000], 2);
        assert_eq!(rc, [8_738, -1_038]);
        assert_eq!(energy, 147_296);
    }

    #[test]
    fn clamps_unstable_reflection_to_almost_one() {
        let mut rc = [0i32; 2];
        let energy = schur64(&mut rc, &[200_000, 210_000, -50_000], 2);
        assert_eq!(rc, [-ALMOST_ONE_Q16, 0]);
        assert_eq!(energy, 200_000);
    }
}
