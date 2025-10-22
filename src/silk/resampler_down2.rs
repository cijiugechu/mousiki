//! Port of the fixed-point 2× downsampler from the SILK reference implementation.
//!
//! This mirrors the `silk_resampler_down2` helper from `silk/resampler_down2.c`, which
//! runs a pair of first-order all-pass filters in parallel to separate even/odd input
//! samples before combining them into a decimated output stream. The filters operate in
//! Q10 precision and keep a small `[i32; 2]` state that tracks their running sums between
//! calls.

/// Q15 coefficients lifted from `silk/resampler_rom.h`.
const RESAMPLER_DOWN2_COEF0: i32 = 9_872;
const RESAMPLER_DOWN2_COEF1: i32 = 39_809 - 65_536;

/// Downsamples `input` by a factor of two using first-order all-pass sections.
///
/// The function consumes pairs of input samples, updating `state` in-place and writing
/// `input.len() / 2` decimated samples into `output`. Any trailing odd input sample is
/// ignored, matching the behaviour of the C reference implementation.
///
/// # Panics
///
/// * If `output.len()` is smaller than `input.len() / 2`.
/// * If `state` does not hold exactly two elements.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub fn resampler_down2(state: &mut [i32; 2], output: &mut [i16], input: &[i16]) {
    let len2 = input.len() / 2;
    assert!(
        output.len() >= len2,
        "output buffer too small: need {} samples",
        len2
    );

    for k in 0..len2 {
        let mut in32 = i32::from(input[2 * k]) << 10;
        let mut y = in32 - state[0];
        let mut x = smlawb(y, y, RESAMPLER_DOWN2_COEF1);
        let mut out32 = state[0] + x;
        state[0] = in32 + x;

        in32 = i32::from(input[2 * k + 1]) << 10;
        y = in32 - state[1];
        x = smulwb(y, RESAMPLER_DOWN2_COEF0);
        out32 += state[1];
        out32 += x;
        state[1] = in32 + x;

        output[k] = sat16(rshift_round(out32, 11));
    }
}

#[inline]
fn smlawb(a: i32, b: i32, coef_q15: i32) -> i32 {
    let product = i64::from(b) * i64::from(coef_q15 as i16);
    a.wrapping_add((product >> 16) as i32)
}

#[inline]
fn smulwb(a: i32, coef_q15: i32) -> i32 {
    let product = i64::from(a) * i64::from(coef_q15 as i16);
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
    use super::resampler_down2;

    #[test]
    fn handles_zero_input() {
        let mut state = [0i32; 2];
        let input = [0i16; 8];
        let mut output = [0i16; 4];
        resampler_down2(&mut state, &mut output, &input);
        assert_eq!(output, [0, 0, 0, 0]);
        assert_eq!(state, [0, 0]);
    }

    #[test]
    fn matches_reference_sequence() {
        let mut state = [0i32; 2];
        let input = [1000, -1000, 2000, -2000];
        let mut output = [0i16; 2];
        resampler_down2(&mut state, &mut output, &input);
        assert_eq!(output, [228, 284]);
        assert_eq!(state, [2_292_180, -2_179_015]);
    }

    #[test]
    fn propagates_state_between_calls() {
        let mut state = [12_345, -54_321];
        let input = [
            25_340, -4_753, 19_673, 28_343, -2_438, -27_347, -13_032, 3_506, 1_845, -3_463, 21_367,
            24_385,
        ];
        let mut output = [0i16; 6];
        resampler_down2(&mut state, &mut output, &input);
        assert_eq!(output, [7_318, 13_784, 12_751, -20_786, 1_202, 8_517]);
        assert_eq!(state, [27_270_072, 29_567_755]);
    }

    #[test]
    fn allows_odd_length_input() {
        let mut state = [0i32; 2];
        let input = [1i16, 2, 3];
        let mut output = [0i16; 2];
        resampler_down2(&mut state, &mut output, &input);
        let produced = input.len() / 2;
        assert_eq!(&output[..produced], &[0]);
        assert_eq!(state, [1_646, 2_356]);
    }
}
