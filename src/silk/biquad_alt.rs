//! Ports the SILK fixed-point second-order ARMA filter helpers from `silk/biquad_alt.c`.
//!
//! The C implementation provides two entry points: one that processes samples with a
//! stride of one and one that handles interleaved stereo buffers (stride two). Both are
//! reproduced here with idiomatic Rust signatures while maintaining the original fixed-
//! point arithmetic and state-update semantics.

/// Applies the alternative biquad filter implementation to a mono input buffer.
///
/// This mirrors `silk_biquad_alt_stride1` from the reference C sources. The filter uses
/// Q28 feed-forward coefficients (`b_q28`), Q28 feedback coefficients (`a_q28`), and a
/// two-element Q12 state vector (`state`) that is updated in place. The caller must
/// provide an output slice that is at least as long as `input`.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::arithmetic_side_effects
)]
pub fn biquad_alt_stride1(
    input: &[i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    state: &mut [i32; 2],
    output: &mut [i16],
) {
    assert_eq!(input.len(), output.len());

    let mut s0 = state[0];
    let mut s1 = state[1];

    let a0_l_q28 = (-a_q28[0]) & 0x0000_3fff;
    let a0_u_q28 = (-a_q28[0]) >> 14;
    let a1_l_q28 = (-a_q28[1]) & 0x0000_3fff;
    let a1_u_q28 = (-a_q28[1]) >> 14;

    for (in_sample, out_sample) in input.iter().zip(output.iter_mut()) {
        let inval = i32::from(*in_sample);
        let mut out32_q14 = smlawb(s0, b_q28[0], inval);
        out32_q14 <<= 2;

        s0 = s1.wrapping_add(rshift_round(smulwb(out32_q14, a0_l_q28), 14));
        s0 = smlawb(s0, out32_q14, a0_u_q28);
        s0 = smlawb(s0, b_q28[1], inval);

        s1 = rshift_round(smulwb(out32_q14, a1_l_q28), 14);
        s1 = smlawb(s1, out32_q14, a1_u_q28);
        s1 = smlawb(s1, b_q28[2], inval);

        let rounded = out32_q14.wrapping_add((1 << 14) - 1) >> 14;
        *out_sample = sat16(rounded);
    }

    state[0] = s0;
    state[1] = s1;
}

/// Applies the alternative biquad filter to interleaved stereo data with stride two.
///
/// This mirrors `silk_biquad_alt_stride2_c` from the reference C implementation and
/// expects `state` to contain four Q12 accumulator values. The input and output buffers
/// must have the same (even) length because each iteration consumes and produces two
/// samples.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::arithmetic_side_effects
)]
pub fn biquad_alt_stride2(
    input: &[i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    state: &mut [i32; 4],
    output: &mut [i16],
) {
    assert_eq!(input.len(), output.len());
    assert!(input.len() % 2 == 0, "signal length must be even");

    let mut s0 = state[0];
    let mut s1 = state[1];
    let mut s2 = state[2];
    let mut s3 = state[3];

    let a0_l_q28 = (-a_q28[0]) & 0x0000_3fff;
    let a0_u_q28 = (-a_q28[0]) >> 14;
    let a1_l_q28 = (-a_q28[1]) & 0x0000_3fff;
    let a1_u_q28 = (-a_q28[1]) >> 14;

    for (chunk_in, chunk_out) in input.chunks_exact(2).zip(output.chunks_exact_mut(2)) {
        let inval0 = i32::from(chunk_in[0]);
        let inval1 = i32::from(chunk_in[1]);

        let out0_q14 = smlawb(s0, b_q28[0], inval0) << 2;
        let out1_q14 = smlawb(s2, b_q28[0], inval1) << 2;

        s0 = s1.wrapping_add(rshift_round(smulwb(out0_q14, a0_l_q28), 14));
        s2 = s3.wrapping_add(rshift_round(smulwb(out1_q14, a0_l_q28), 14));
        s0 = smlawb(s0, out0_q14, a0_u_q28);
        s2 = smlawb(s2, out1_q14, a0_u_q28);
        s0 = smlawb(s0, b_q28[1], inval0);
        s2 = smlawb(s2, b_q28[1], inval1);

        s1 = rshift_round(smulwb(out0_q14, a1_l_q28), 14);
        s3 = rshift_round(smulwb(out1_q14, a1_l_q28), 14);
        s1 = smlawb(s1, out0_q14, a1_u_q28);
        s3 = smlawb(s3, out1_q14, a1_u_q28);
        s1 = smlawb(s1, b_q28[2], inval0);
        s3 = smlawb(s3, b_q28[2], inval1);

        let rounded0 = out0_q14.wrapping_add((1 << 14) - 1) >> 14;
        let rounded1 = out1_q14.wrapping_add((1 << 14) - 1) >> 14;
        chunk_out[0] = sat16(rounded0);
        chunk_out[1] = sat16(rounded1);
    }

    state[0] = s0;
    state[1] = s1;
    state[2] = s2;
    state[3] = s3;
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

#[inline]
fn rshift_round(value: i32, shift: i32) -> i32 {
    debug_assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
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

#[cfg(test)]
mod tests {
    use super::{biquad_alt_stride1, biquad_alt_stride2};

    #[test]
    fn stride1_matches_reference_sequence() {
        let mut state = [123_456, -654_321];
        let b = [1_145_324_612, -229_064_922, 1_145_324_612];
        let a = [-1_010_580_540, 505_290_270];
        let input = [1234, -2345, 3456, -4567, 5678, -6789];
        let mut output = [0i16; 6];

        biquad_alt_stride1(&input, &b, &a, &mut state, &mut output);

        assert_eq!(output, [5296, -12_464, 14_977, -12_502, 17_619, -32_768]);
        assert_eq!(state, [19_797_224, 142_793_818]);
    }

    #[test]
    fn stride2_matches_reference_sequence() {
        let mut state = [2_345_678, -3_456_789, 4_567_890, -5_678_901];
        let b = [1_145_324_612, -229_064_922, 1_145_324_612];
        let a = [-1_010_580_540, 505_290_270];
        let input = [1357, -2468, 3579, -4680, 5791, -6802, 7913, -8024];
        let mut output = [0i16; 8];

        biquad_alt_stride2(&input, &b, &a, &mut state, &mut output);

        assert_eq!(
            output,
            [
                6363, -9414, 11_772, -17_033, 12_698, -13_828, 18_946, -13_083
            ]
        );
        assert_eq!(state, [-42_612_823, -7_780_131, 28_399_995, -39_356_343]);
    }

    #[test]
    fn identity_coefficients_copy_input() {
        let mut state = [0; 2];
        let b = [1 << 28, 0, 0];
        let a = [0, 0];
        let input = [100, -200, 300, -400];
        let mut output = [0i16; 4];

        biquad_alt_stride1(&input, &b, &a, &mut state, &mut output);
        assert_eq!(output, input);
        assert_eq!(state, [0, 0]);
    }
}
