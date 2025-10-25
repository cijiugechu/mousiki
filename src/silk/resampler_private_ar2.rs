//! Port of the second-order AR filter helper from the SILK resampler.
//!
//! This mirrors `silk_resampler_private_AR2` from `silk/resampler_private_AR2.c`, which
//! feeds incoming 16-bit samples through a pair of single-delay sections using Q14
//! coefficients. The routine updates the two-element state in-place and produces a Q8
//! output stream used by higher level resampler stages.

/// Runs the SILK second-order AR filter on `input`, writing Q8 output samples into
/// `output_q8` and updating `state` in-place.
///
/// # Panics
///
/// * If `output_q8.len()` is smaller than `input.len()`.
pub fn resampler_private_ar2(
    state: &mut [i32; 2],
    output_q8: &mut [i32],
    input: &[i16],
    a_q14: &[i16; 2],
) {
    assert!(
        output_q8.len() >= input.len(),
        "output buffer too small: need {} entries",
        input.len()
    );

    for (k, &sample) in input.iter().enumerate() {
        let out32 = add_lshift32(state[0], i32::from(sample), 8);
        output_q8[k] = out32;

        let out32 = lshift(out32, 2);
        state[0] = smlawb(state[1], out32, i32::from(a_q14[0]));
        state[1] = smulwb(out32, i32::from(a_q14[1]));
    }
}

#[inline]
fn add_lshift32(a: i32, b: i32, shift: u32) -> i32 {
    a.wrapping_add(b.wrapping_shl(shift))
}

#[inline]
fn lshift(value: i32, shift: u32) -> i32 {
    value.wrapping_shl(shift)
}

#[inline]
fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let product = i64::from(b) * i64::from(c as i16);
    a.wrapping_add((product >> 16) as i32)
}

#[inline]
fn smulwb(a: i32, b: i32) -> i32 {
    let product = i64::from(a) * i64::from(b as i16);
    (product >> 16) as i32
}

#[cfg(test)]
mod tests {
    use super::resampler_private_ar2;

    #[test]
    fn matches_reference_case1() {
        let mut state = [0i32; 2];
        let mut output = [0i32; 4];
        let input = [1000i16, -1000, 2000, -2000];
        let coeffs = [17476i16, -8566];

        resampler_private_ar2(&mut state, &mut output, &input, &coeffs);

        assert_eq!(output, [256_000, 17_062, 396_355, -98_149]);
        assert_eq!(state, [-311_917, 51_314]);
    }

    #[test]
    fn handles_nonzero_state_and_coefficients() {
        let mut state = [123_456i32, -654_321];
        let mut output = [0i32; 6];
        let input = [23_123i16, -18_234, 12_763, -28_761, 3_123, -9_631];
        let coeffs = [15_360i16, 8_192];

        resampler_private_ar2(&mut state, &mut output, &input, &coeffs);

        assert_eq!(
            output,
            [6_042_944, 343_035, 6_610_395, -994_054, 3_172_759, 11_898]
        );
        assert_eq!(state, [1_597_533, 5_949]);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn panics_on_small_output_buffer() {
        let mut state = [0i32; 2];
        let mut output = [0i32; 2];
        let input = [1i16, 2, 3];
        let coeffs = [1000i16, -2000];

        resampler_private_ar2(&mut state, &mut output, &input, &coeffs);
    }
}
