//! Port of `silk/fixed/apply_sine_window_FIX.c`.
//!
//! Applies a sine window to a 16-bit PCM signal using the fixed-point recurrence
//! from the original SILK implementation. The window length must match the C
//! constraints (16–120 samples in multiples of four) and callers choose between
//! the rising (type 1) and falling (type 2) halves of the sine curve.

const FREQ_TABLE_Q16: [i16; 27] = [
    12_111, 9_804, 8_235, 7_100, 6_239, 5_565, 5_022, 4_575, 4_202, 3_885, 3_612, 3_375, 3_167,
    2_984, 2_820, 2_674, 2_542, 2_422, 2_313, 2_214, 2_123, 2_038, 1_961, 1_889, 1_822, 1_760,
    1_702,
];

const UNITY_Q16: i32 = 1 << 16;

/// Apply a fixed-point sine window to `input`, writing the result into `output`.
///
/// `win_type` selects the rising (1) or falling (2) section of the sine window.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::arithmetic_side_effects
)]
pub fn apply_sine_window(output: &mut [i16], input: &[i16], win_type: i32) {
    assert_eq!(
        output.len(),
        input.len(),
        "input and output buffers must have matching lengths"
    );
    assert!(
        matches!(win_type, 1 | 2),
        "window type must be 1 (rising) or 2 (falling)"
    );

    let length = input.len();
    assert!(
        (16..=120).contains(&(length as i32)),
        "window length must be between 16 and 120 samples"
    );
    assert!(
        length.is_multiple_of(4),
        "window length must be a multiple of four samples"
    );

    let table_index = length / 4 - 4;
    assert!(
        table_index < FREQ_TABLE_Q16.len(),
        "unsupported window length: index out of range"
    );

    let freq_q16 = i32::from(FREQ_TABLE_Q16[table_index]);
    let coef_q16 = smulwb(freq_q16, -freq_q16);

    let mut s0_q16: i32;
    let mut s1_q16: i32;

    if win_type == 1 {
        s0_q16 = 0;
        s1_q16 = freq_q16.wrapping_add((length as i32) >> 3);
    } else {
        s0_q16 = UNITY_Q16;
        s1_q16 = UNITY_Q16
            .wrapping_add(coef_q16 >> 1)
            .wrapping_add((length as i32) >> 4);
    }

    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let avg_q16 = (s0_q16.wrapping_add(s1_q16)) >> 1;
        chunk_out[0] = smulwb(avg_q16, i32::from(chunk_in[0])) as i16;
        chunk_out[1] = smulwb(s1_q16, i32::from(chunk_in[1])) as i16;

        s0_q16 = smulwb(s1_q16, coef_q16)
            .wrapping_add(s1_q16 << 1)
            .wrapping_sub(s0_q16)
            .wrapping_add(1);
        if s0_q16 > UNITY_Q16 {
            s0_q16 = UNITY_Q16;
        }

        let avg_q16 = (s0_q16.wrapping_add(s1_q16)) >> 1;
        chunk_out[2] = smulwb(avg_q16, i32::from(chunk_in[2])) as i16;
        chunk_out[3] = smulwb(s0_q16, i32::from(chunk_in[3])) as i16;

        s1_q16 = smulwb(s0_q16, coef_q16)
            .wrapping_add(s0_q16 << 1)
            .wrapping_sub(s1_q16);
        if s1_q16 > UNITY_Q16 {
            s1_q16 = UNITY_Q16;
        }
    }
}

#[inline]
fn smulwb(a: i32, b: i32) -> i32 {
    let product = i64::from(a) * i64::from(i32::from(b as i16));
    (product >> 16) as i32
}

#[cfg(test)]
mod tests {
    use super::apply_sine_window;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn rising_window_matches_reference_sequence() {
        let input: Vec<i16> = (0..16).map(|i| (i as i32 * 100) as i16).collect();
        let mut output = vec![0i16; input.len()];

        apply_sine_window(&mut output, &input, 1);

        assert_eq!(
            output,
            [
                0, 18, 54, 109, 178, 264, 362, 474, 591, 722, 851, 989, 1119, 1256, 1376, 1500
            ]
        );
    }

    #[test]
    fn falling_window_matches_reference_sequence() {
        let input: Vec<i16> = (0..16).map(|i| (1000 - i as i32 * 50) as i16).collect();
        let mut output = vec![0i16; input.len()];

        apply_sine_window(&mut output, &input, 2);

        assert_eq!(
            output,
            [
                991, 933, 861, 792, 712, 637, 555, 479, 401, 330, 261, 199, 143, 95, 54, 22
            ]
        );
    }

    #[test]
    fn longer_window_handles_negative_samples() {
        let length = 32usize;
        let input: Vec<i16> = (0..length)
            .map(|i| ((i as i32 % 8) - 4) * 200)
            .map(|v| v as i16)
            .collect();
        let mut output = vec![0i16; length];

        apply_sine_window(&mut output, &input, 1);

        assert_eq!(
            output,
            [
                -39, -58, -57, -38, 0, 56, 130, 223, -333, -276, -201, -109, 0, 123, 262, 414,
                -580, -455, -315, -164, 0, 173, 355, 546, -744, -569, -385, -195, 0, 198, 398, 600
            ]
        );
    }
}
