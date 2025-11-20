//! Port of `silk/float/apply_sine_window_FLP.c`.
//!
//! Applies a floating-point sine window to an input slice. The window length
//! must be a multiple of four samples and callers select either the rising
//! (type 1) or falling (type 2) half of the sine curve.

use core::f32::consts::PI;

/// Apply a sine window to floating-point input samples.
pub fn apply_sine_window_flp(output: &mut [f32], input: &[f32], win_type: i32) {
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
        length.is_multiple_of(4),
        "window length must be a multiple of four samples"
    );

    let freq = PI / (length as f32 + 1.0);
    let c = 2.0 - freq * freq;

    let (mut s0, mut s1) = if win_type == 1 {
        (0.0f32, freq)
    } else {
        (1.0, 0.5 * c)
    };

    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        chunk_out[0] = chunk_in[0] * 0.5 * (s0 + s1);
        chunk_out[1] = chunk_in[1] * s1;
        s0 = c * s1 - s0;
        chunk_out[2] = chunk_in[2] * 0.5 * (s1 + s0);
        chunk_out[3] = chunk_in[3] * s0;
        s1 = c * s0 - s1;
    }
}

#[cfg(test)]
mod tests {
    use super::apply_sine_window_flp;

    fn assert_close(lhs: &[f32], rhs: &[f32]) {
        const TOLERANCE: f32 = 5e-6;
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "slices must have identical lengths when comparing"
        );
        for (index, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= TOLERANCE,
                "difference at index {index} exceeds tolerance: {diff} > {TOLERANCE}"
            );
        }
    }

    #[test]
    fn rising_window_matches_reference_values() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0f32; 8];

        apply_sine_window_flp(&mut output, &input, 1);

        let expected = [
            0.174_532_92,
            0.698_131_7,
            1.506_997_3,
            2.622_396_2,
            3.844_621_4,
            5.293_497,
            6.592_775_3,
            8.011_205,
        ];

        assert_close(&output, &expected);
    }

    #[test]
    fn falling_window_matches_reference_values() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0f32; 8];

        apply_sine_window_flp(&mut output, &input, 2);

        let expected = [
            0.969_538_27,
            1.878_153_1,
            2.554_209,
            3.054_918,
            3.147_635,
            2.971_946_5,
            2.316_614_6,
            1.332_524_3,
        ];

        assert_close(&output, &expected);
    }

    #[test]
    #[should_panic(expected = "input and output buffers must have matching lengths")]
    fn panics_on_mismatched_lengths() {
        let input = [0.0f32; 4];
        let mut output = [0.0f32; 8];

        apply_sine_window_flp(&mut output, &input, 1);
    }

    #[test]
    #[should_panic(expected = "window type must be 1 (rising) or 2 (falling)")]
    fn panics_on_invalid_window_type() {
        let input = [0.0f32; 4];
        let mut output = [0.0f32; 4];

        apply_sine_window_flp(&mut output, &input, 0);
    }

    #[test]
    #[should_panic(expected = "window length must be a multiple of four samples")]
    fn panics_on_length_not_multiple_of_four() {
        let input = [0.0f32; 6];
        let mut output = [0.0f32; 6];

        apply_sine_window_flp(&mut output, &input, 1);
    }
}
