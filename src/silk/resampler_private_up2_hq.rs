//! Port of the high-quality 2× upsampler used by SILK's resamplers.
//!
//! This mirrors the fixed-point helper found in `silk/resampler_private_up2_HQ.c`,
//! which runs two parallel cascades of first-order all-pass sections followed by a
//! small notch filter. The routine consumes `len` 16-bit input samples and emits
//! `2 * len` output samples while updating a six-element IIR state in-place.

use super::resampler_rom::{SILK_RESAMPLER_UP2_HQ_0, SILK_RESAMPLER_UP2_HQ_1};

/// Minimal resampler state that exposes the `[i32; 6]` IIR delay elements expected by
/// the high-quality 2× upsampler.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResamplerStateUp2Hq {
    /// Internal IIR delay elements stored in Q10 format.
    pub s_iir: [i32; 6],
}

impl ResamplerStateUp2Hq {
    /// Creates a new state with all delay elements cleared.
    pub const fn new() -> Self {
        Self { s_iir: [0; 6] }
    }

    pub fn resampler_private_up2_hq_wrapper(
        &mut self,
        output: &mut [i16],
        input: &[i16],
    ) {
        resampler_private_up2_hq(&mut self.s_iir, output, input);
    }
}

/// Runs the high-quality 2× upsampler on `input`, writing interleaved even/odd samples
/// into `output` and updating `state` in-place.
///
/// # Panics
///
/// * If `output.len()` is smaller than `2 * input.len()`.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn resampler_private_up2_hq(state: &mut [i32; 6], output: &mut [i16], input: &[i16]) {
    assert!(
        output.len() >= input.len() * 2,
        "output buffer too small: need {} samples",
        input.len() * 2
    );

    for (k, &sample) in input.iter().enumerate() {
        let in32 = i32::from(sample) << 10;

        let mut y = in32 - state[0];
        let mut x = smulwb(y, i32::from(SILK_RESAMPLER_UP2_HQ_0[0]));
        let mut out32_1 = state[0] + x;
        state[0] = in32 + x;

        y = out32_1 - state[1];
        x = smulwb(y, i32::from(SILK_RESAMPLER_UP2_HQ_0[1]));
        let mut out32_2 = state[1] + x;
        state[1] = out32_1 + x;

        y = out32_2 - state[2];
        x = smlawb(y, y, i32::from(SILK_RESAMPLER_UP2_HQ_0[2]));
        out32_1 = state[2] + x;
        state[2] = out32_2 + x;

        output[2 * k] = sat16(rshift_round(out32_1, 10));

        y = in32 - state[3];
        x = smulwb(y, i32::from(SILK_RESAMPLER_UP2_HQ_1[0]));
        out32_1 = state[3] + x;
        state[3] = in32 + x;

        y = out32_1 - state[4];
        x = smulwb(y, i32::from(SILK_RESAMPLER_UP2_HQ_1[1]));
        out32_2 = state[4] + x;
        state[4] = out32_1 + x;

        y = out32_2 - state[5];
        x = smlawb(y, y, i32::from(SILK_RESAMPLER_UP2_HQ_1[2]));
        out32_1 = state[5] + x;
        state[5] = out32_2 + x;

        output[2 * k + 1] = sat16(rshift_round(out32_1, 10));
    }
}

#[inline]
fn smulwb(a: i32, b_q15: i32) -> i32 {
    let product = i64::from(a) * i64::from(b_q15 as i16);
    (product >> 16) as i32
}

#[inline]
fn smlawb(a: i32, b: i32, c_q15: i32) -> i32 {
    let product = i64::from(b) * i64::from(c_q15 as i16);
    a.wrapping_add((product >> 16) as i32)
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
    use super::{ResamplerStateUp2Hq, resampler_private_up2_hq};

    #[test]
    fn produces_zero_output_for_zero_input() {
        let mut state = [0i32; 6];
        let input = [0i16; 8];
        let mut output = [0i16; 16];
        resampler_private_up2_hq(&mut state, &mut output, &input);
        assert!(output.iter().all(|&sample| sample == 0));
        assert_eq!(state, [0; 6]);
    }

    #[test]
    fn matches_expected_sequence() {
        let mut state = [0i32; 6];
        let input = [1000i16, -1000, 2000, -2000];
        let mut output = [0i16; 8];
        resampler_private_up2_hq(&mut state, &mut output, &input);
        assert_eq!(output, [4, 35, 152, 381, 571, 423, -52, -236]);
        assert_eq!(
            state,
            [
                -2_159_345, 2_825_288, -1_646_130, -2_512_442, 3_368_205, -583_113
            ]
        );
    }

    #[test]
    fn wrapper_delegates_to_internal_state() {
        let mut state = ResamplerStateUp2Hq::new();
        let input = [3123i16, -1812, 904, -222];
        let mut output = [0i16; 8];
        state.resampler_private_up2_hq_wrapper(&mut output, &input);
        assert_eq!(output, [11, 109, 478, 1236, 1967, 1680, -29, -1_740]);
        assert_eq!(
            state.s_iir,
            [
                -260_118, 1_931_422, -5_580_563, -384_499, 3_302_757, -4_867_408
            ]
        );
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn panics_if_output_is_too_small() {
        let mut state = [0i32; 6];
        let input = [1i16, 2, 3];
        let mut output = [0i16; 5];
        resampler_private_up2_hq(&mut state, &mut output, &input);
    }
}
