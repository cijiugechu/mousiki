//! Ports the SILK first-order analysis filter bank from the reference C sources.
//!
//! The original implementation lives in `silk/ana_filt_bank_1.c` and splits an input
//! signal into low- and high-frequency components using a pair of first-order all-pass
//! filters. The routine maintains a two-element Q10 state that carries filter history
//! across calls.

/// Q15 coefficients lifted from `silk/ana_filt_bank_1.c`.
const A_FB1_20: i16 = 5394 << 1;
const A_FB1_21: i16 = -24290;

/// Splits `input` into low/high bands using the SILK analysis filter bank.
///
/// The function consumes samples in pairs, updating `state` in-place and writing
/// `input.len() / 2` decimated samples into `low_band` and `high_band`. The caller must
/// ensure that `input.len()` is even.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub fn ana_filt_bank_1(
    state: &mut [i32; 2],
    low_band: &mut [i16],
    high_band: &mut [i16],
    input: &[i16],
) {
    assert!(input.len().is_multiple_of(2), "input length must be even");
    let half_len = input.len() / 2;
    assert!(low_band.len() >= half_len, "low_band buffer too small");
    assert!(high_band.len() >= half_len, "high_band buffer too small");

    for k in 0..half_len {
        let mut in32 = i32::from(input[2 * k]) << 10;

        let mut y = in32 - state[0];
        let mut x = smlawb(y, y, i32::from(A_FB1_21));
        let out_1 = state[0] + x;
        state[0] = in32 + x;

        in32 = i32::from(input[2 * k + 1]) << 10;

        y = in32 - state[1];
        x = smulwb(y, i32::from(A_FB1_20));
        let out_2 = state[1] + x;
        state[1] = in32 + x;

        let sum = out_2 + out_1;
        let diff = out_2 - out_1;

        low_band[k] = sat16(rshift_round(sum, 11));
        high_band[k] = sat16(rshift_round(diff, 11));
    }
}

#[inline]
fn smlawb(acc: i32, value: i32, coef_q15: i32) -> i32 {
    let product = i64::from(value) * i64::from(coef_q15 as i16);
    acc.wrapping_add((product >> 16) as i32)
}

#[inline]
fn smulwb(value: i32, coef_q15: i32) -> i32 {
    let product = i64::from(value) * i64::from(coef_q15 as i16);
    (product >> 16) as i32
}

#[inline]
fn sat16(value: i32) -> i16 {
    if value > i32::from(i16::MAX) {
        i16::MAX
    } else if value < i32::from(i16::MIN) {
        i16::MIN
    } else {
        value as i16
    }
}

#[inline]
fn rshift_round(value: i32, shift: u32) -> i32 {
    assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

#[cfg(test)]
mod tests {
    use super::ana_filt_bank_1;

    #[test]
    fn splits_constant_signal() {
        let mut state = [0i32; 2];
        let input = [1i16; 8];
        let mut low = [0i16; 4];
        let mut high = [0i16; 4];
        ana_filt_bank_1(&mut state, &mut low, &mut high, &input);
        assert_eq!(low, [0, 1, 1, 1]);
        assert_eq!(high, [0, 0, 0, 0]);
        assert_eq!(state, [863, 1_023]);
    }

    #[test]
    fn matches_reference_values() {
        let mut state = [12_345, -54_321];
        let input = [
            25_340, -4_753, 19_673, 28_343, -2_438, -27_347, -13_032, 3_506, 1_845, -3_463, 21_367,
            24_385,
        ];
        let mut low = [0i16; 6];
        let mut high = [0i16; 6];
        ana_filt_bank_1(&mut state, &mut low, &mut high, &input);
        assert_eq!(low, [7_563, 13_865, 12_275, -20_892, 1_549, 8_803]);
        assert_eq!(high, [-8_390, -13_816, 11_558, -9_801, 6_439, -9_567]);
        assert_eq!(state, [27_090_108, 30_044_726]);
    }
}
