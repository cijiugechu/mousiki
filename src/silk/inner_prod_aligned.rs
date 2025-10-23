//! Port of the `silk_inner_prod_aligned_scale` helper from the reference SILK
//! implementation.
//!
//! The routine computes the inner product between two 16-bit vectors while
//! applying an arithmetic right shift to each product term before the
//! accumulation. This mirrors the behaviour of the C function in
//! `silk/inner_prod_aligned.c`, which is used throughout the resampler helpers
//! and other fixed-point kernels where partial scaling is required.

/// Computes the scaled inner product of `in_vec1` and `in_vec2`.
///
/// Each product term is right-shifted by `scale` bits prior to accumulation,
/// matching the behaviour of the reference implementation's
/// `silk_inner_prod_aligned_scale` routine.
///
/// # Panics
///
/// * If the input slices have different lengths.
/// * If `scale` falls outside the range `0..32` (i.e. `0 <= scale < 32`).
pub fn inner_prod_aligned_scale(in_vec1: &[i16], in_vec2: &[i16], scale: i32) -> i32 {
    assert!((0..32).contains(&scale), "scale must be in the range 0..32");
    assert_eq!(
        in_vec1.len(),
        in_vec2.len(),
        "input vectors must have identical lengths"
    );

    let mut sum = 0i32;
    for (&a, &b) in in_vec1.iter().zip(in_vec2.iter()) {
        let product = i32::from(a) * i32::from(b);
        let shifted = product >> scale;
        sum = sum.wrapping_add(shifted);
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::inner_prod_aligned_scale;

    #[test]
    fn matches_unscaled_inner_product() {
        let vec1 = [1, 2, 3, 4];
        let vec2 = [-1, 2, -3, 4];
        let expected: i32 = vec1
            .iter()
            .zip(vec2.iter())
            .map(|(&a, &b)| i32::from(a) * i32::from(b))
            .sum();
        assert_eq!(inner_prod_aligned_scale(&vec1, &vec2, 0), expected);
    }

    #[test]
    fn applies_right_shift_before_accumulation() {
        let vec1 = [300, -400, 500, -600];
        let vec2 = [-7, 9, -11, 13];
        // Reference calculation using 32-bit arithmetic with an arithmetic shift.
        let expected =
            ((300 * -7) >> 2) + ((-400 * 9) >> 2) + ((500 * -11) >> 2) + ((-600 * 13) >> 2);
        assert_eq!(inner_prod_aligned_scale(&vec1, &vec2, 2), expected);
    }

    #[test]
    fn handles_large_scale_values() {
        let vec1 = [12345, -23456];
        let vec2 = [-3210, 4321];
        let expected = ((12345 * -3210) >> 8) + ((-23456 * 4321) >> 8);
        assert_eq!(inner_prod_aligned_scale(&vec1, &vec2, 8), expected);
    }

    #[test]
    fn empty_vectors_return_zero() {
        let vec: [i16; 0] = [];
        assert_eq!(inner_prod_aligned_scale(&vec, &vec, 0), 0);
    }
}
