//! Port of the fixed-point `silk_k2a` helper from `silk/fixed/k2a_FIX.c` in the
//! reference SILK implementation. The routine performs the "step up"
//! transformation that converts reflection coefficients (PARCOR coefficients)
//! into the forward LPC predictor coefficients used throughout the decoder and
//! encoder signal-processing pipeline.

/// Converts reflection coefficients to prediction coefficients.
///
/// The inputs are expressed with the same Q-format as the C reference: the
/// reflection coefficients are Q15 values and the resulting predictor
/// coefficients are accumulated in Q24. The caller must provide an output
/// buffer whose length is at least the number of supplied reflection
/// coefficients; only the first `rc_q15.len()` entries will be updated.
///
/// This is a direct translation of the reference routine and therefore uses
/// the same wrapping arithmetic semantics.
pub fn k2a(a_q24: &mut [i32], rc_q15: &[i16]) {
    let order = rc_q15.len();
    assert!(a_q24.len() >= order, "output buffer is smaller than rc_q15");

    let a_q24 = &mut a_q24[..order];

    for k in 0..order {
        let rc = i32::from(rc_q15[k]);
        let half = k.div_ceil(2);

        for n in 0..half {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlawb(tmp1, silk_lshift(tmp2, 1), rc);
            a_q24[k - n - 1] = silk_smlawb(tmp2, silk_lshift(tmp1, 1), rc);
        }

        a_q24[k] = -silk_lshift(rc, 9);
    }
}

#[inline]
fn silk_smlawb(a: i32, b: i32, c: i32) -> i32 {
    let c_low = i32::from(c as i16);
    let product = (i64::from(b) * i64::from(c_low)) >> 16;
    a.wrapping_add(product as i32)
}

#[inline]
fn silk_lshift(value: i32, shift: u32) -> i32 {
    debug_assert!(shift < 32, "shift must be less than 32");
    ((value as u32) << shift) as i32
}

#[cfg(test)]
mod tests {
    use super::k2a;

    #[test]
    fn single_reflection_matches_reference() {
        let mut a_q24 = [0i32; 1];
        let rc_q15 = [16_384i16];

        k2a(&mut a_q24, &rc_q15);

        assert_eq!(a_q24, [-8_388_608]);
    }

    #[test]
    fn two_stage_conversion_produces_expected_coefficients() {
        let mut a_q24 = [0i32; 2];
        let rc_q15 = [8_192i16, 4_096i16];

        k2a(&mut a_q24, &rc_q15);

        assert_eq!(a_q24, [-4_718_592, -2_097_152]);
    }

    #[test]
    fn handles_mixed_sign_reflection_coefficients() {
        let mut a_q24 = [0i32; 3];
        let rc_q15 = [3_000i16, -2_000i16, 1_000i16];

        k2a(&mut a_q24, &rc_q15);

        assert_eq!(a_q24, [-1_411_000, 979_986, -512_000]);
    }

    #[test]
    fn leaves_trailing_capacity_untouched() {
        let mut a_q24 = [0i32; 4];
        let rc_q15 = [1_000i16, 2_000i16];

        k2a(&mut a_q24, &rc_q15);

        assert_eq!(a_q24, [-543_250, -1_024_000, 0, 0]);
    }

    #[test]
    fn handles_empty_inputs() {
        let mut a_q24 = [0i32; 0];
        let rc_q15: [i16; 0] = [];

        k2a(&mut a_q24, &rc_q15);
    }
}
