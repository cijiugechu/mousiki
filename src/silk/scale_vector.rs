//! Floating-point vector scaling helpers from `silk/float/scale_vector_FLP.c`.
//!
//! These utilities are dependency-light and intentionally mirror the unrolled
//! loops from the C implementation so future ports of the FLP analysis path
//! can reuse the same building blocks without touching the C sources.

/// Multiplies each element of `data` by `gain` in-place.
///
/// Mirrors `silk_scale_vector_FLP`, including the 4× unrolled loop that helps
/// the reference implementation pipeline floating-point multiplies.
pub fn scale_vector(data: &mut [f32], gain: f32) {
    let mut i = 0;
    let data_size4 = data.len() & !3;

    while i < data_size4 {
        data[i] *= gain;
        data[i + 1] *= gain;
        data[i + 2] *= gain;
        data[i + 3] *= gain;
        i += 4;
    }

    while i < data.len() {
        data[i] *= gain;
        i += 1;
    }
}

/// Copies `data_in` into `data_out` while applying the floating-point `gain`.
///
/// This mirrors `silk_scale_copy_vector_FLP` and keeps the original loop
/// structure so callers that rely on the exact traversal order see the same
/// observable behaviour as the C version.
///
/// # Panics
/// Panics if the input and output slices have different lengths.
pub fn scale_copy_vector(data_out: &mut [f32], data_in: &[f32], gain: f32) {
    assert_eq!(
        data_out.len(),
        data_in.len(),
        "input and output slices must have identical lengths"
    );

    let mut i = 0;
    let data_size4 = data_in.len() & !3;

    while i < data_size4 {
        data_out[i] = gain * data_in[i];
        data_out[i + 1] = gain * data_in[i + 1];
        data_out[i + 2] = gain * data_in[i + 2];
        data_out[i + 3] = gain * data_in[i + 3];
        i += 4;
    }

    while i < data_in.len() {
        data_out[i] = gain * data_in[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{scale_copy_vector, scale_vector};
    use alloc::vec::Vec;

    #[test]
    fn scale_vector_matches_reference_loop() {
        let mut data = [0.25f32, -0.5, 1.0, -1.75, 0.0];
        let expected = data
            .iter()
            .map(|value| value * 2.0)
            .collect::<Vec<f32>>();

        scale_vector(&mut data, 2.0);

        assert_eq!(data.as_slice(), expected.as_slice());
    }

    #[test]
    fn scale_copy_vector_scales_into_output() {
        let input = [1.0f32, -0.25, 0.5, -2.0, 4.0, -8.0];
        let mut output = [0.0f32; 6];
        let expected = input
            .iter()
            .map(|value| value * -0.5)
            .collect::<Vec<f32>>();

        scale_copy_vector(&mut output, &input, -0.5);

        assert_eq!(output.as_slice(), expected.as_slice());
    }

    #[test]
    #[should_panic(expected = "input and output slices must have identical lengths")]
    fn scale_copy_vector_panics_on_mismatched_lengths() {
        let input = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 2];
        scale_copy_vector(&mut output, &input, 1.0);
    }
}
