//! Port of the fixed-point `silk_k2a_Q16` helper from
//! `silk/fixed/k2a_Q16_FIX.c` in the reference SILK implementation. The
//! routine performs the "step up" transformation that converts Q16
//! reflection coefficients into the forward LPC predictor coefficients used
//! throughout the codec's fixed-point pipeline.

/// Converts Q16 reflection coefficients to Q24 prediction coefficients.
///
/// The caller must supply an output buffer with at least as many elements as
/// there are reflection coefficients. Only the portion of the buffer
/// corresponding to the provided coefficients is updated; any additional
/// capacity is left untouched.
///
/// This translation preserves the original routine's wrapping semantics.
pub fn k2a_q16(a_q24: &mut [i32], rc_q16: &[i32]) {
    let order = rc_q16.len();
    assert!(a_q24.len() >= order, "output buffer is smaller than rc_q16");

    let a_q24 = &mut a_q24[..order];

    for (k, &rc) in rc_q16.iter().enumerate() {
        let half = (k + 1) >> 1;

        for n in 0..half {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smla_ww(tmp1, tmp2, rc);
            a_q24[k - n - 1] = silk_smla_ww(tmp2, tmp1, rc);
        }

        a_q24[k] = rc.wrapping_shl(8).wrapping_neg();
    }
}

#[inline]
fn silk_smla_ww(a32: i32, b32: i32, c32: i32) -> i32 {
    a32.wrapping_add(((i64::from(b32) * i64::from(c32)) >> 16) as i32)
}

#[cfg(test)]
mod tests {
    use super::k2a_q16;

    #[test]
    fn single_reflection_matches_reference() {
        let mut a_q24 = [0i32; 1];
        let rc_q16 = [32_768i32];

        k2a_q16(&mut a_q24, &rc_q16);

        assert_eq!(a_q24, [-8_388_608]);
    }

    #[test]
    fn two_stage_conversion_produces_expected_coefficients() {
        let mut a_q24 = [0i32; 2];
        let rc_q16 = [16_384i32, 8_192i32];

        k2a_q16(&mut a_q24, &rc_q16);

        assert_eq!(a_q24, [-4_718_592, -2_097_152]);
    }

    #[test]
    fn handles_mixed_sign_reflection_coefficients() {
        let mut a_q24 = [0i32; 3];
        let rc_q16 = [30_000i32, -20_000i32, 10_000i32];

        k2a_q16(&mut a_q24, &rc_q16);

        assert_eq!(a_q24, [-4_555_000, 4_305_752, -2_560_000]);
    }

    #[test]
    fn leaves_trailing_capacity_untouched() {
        let mut a_q24 = [0i32; 4];
        let rc_q16 = [1_000i32, 2_000i32];

        k2a_q16(&mut a_q24, &rc_q16);

        assert_eq!(a_q24, [-263_813, -512_000, 0, 0]);
    }

    #[test]
    fn handles_empty_inputs() {
        let mut a_q24 = [0i32; 0];
        let rc_q16: [i32; 0] = [];

        k2a_q16(&mut a_q24, &rc_q16);
    }
}
