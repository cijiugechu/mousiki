//! Fixed-point vector helpers from `silk/fixed/vector_ops_FIX.c`.
//!
//! This module provides small, dependency-light utilities that scale 16-bit or
//! 32-bit vectors and compute 64-bit inner products. These helpers are shared
//! by several encoder-side DSP kernels (e.g., `find_pred_coefs_FIX.c`) and are
//! intentionally implemented without heap allocations.

/// Copies `data_in` into `data_out` while applying a Q16 gain factor.
///
/// Mirrors `silk_scale_copy_vector16` by multiplying each input sample with
/// `gain_q16` using the `silk_SMULWB` primitive before truncating back to
/// 16-bit precision.
///
/// # Panics
/// Panics if the slices have different lengths.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn scale_copy_vector16(data_out: &mut [i16], data_in: &[i16], gain_q16: i32) {
    assert_eq!(
        data_out.len(),
        data_in.len(),
        "input and output slices must match in length"
    );

    for (dst, &src) in data_out.iter_mut().zip(data_in.iter()) {
        let scaled = smulwb(gain_q16, i32::from(src));
        debug_assert!(
            (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&scaled),
            "value {scaled} cannot be represented as i16"
        );
        *dst = scaled as i16;
    }
}

/// Multiplies each 32-bit element in-place by a Q26 gain and right shifts by 8.
///
/// This matches the behaviour of `silk_scale_vector32_Q26_lshift_18`, producing
/// a Q18 result (`gain_q26` × `data[i]` >> 8) without allocating intermediate
/// buffers.
pub fn scale_vector32_q26_lshift_18(data: &mut [i32], gain_q26: i32) {
    for value in data.iter_mut() {
        let product = i64::from(*value) * i64::from(gain_q26);
        let shifted = product >> 8;
        debug_assert!(
            (i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&shifted),
            "value {shifted} cannot be represented as i32"
        );
        *value = shifted as i32;
    }
}

/// Computes the 64-bit inner product between two 16-bit vectors.
///
/// This follows `silk_inner_prod16_c`, returning the full-precision sum without
/// scaling. The result uses 64-bit accumulation to match the C implementation.
///
/// # Panics
/// Panics if the slices have different lengths.
pub fn inner_prod16(in_vec1: &[i16], in_vec2: &[i16]) -> i64 {
    assert_eq!(
        in_vec1.len(),
        in_vec2.len(),
        "input vectors must have identical lengths"
    );

    let mut sum = 0i64;
    for (&a, &b) in in_vec1.iter().zip(in_vec2.iter()) {
        sum = sum.wrapping_add(i64::from(a) * i64::from(b));
    }

    sum
}

/// Multiplies a Q16 value by the low 16 bits of another operand and shifts
/// the 48-bit product right by 16, mirroring `silk_SMULWB`.
#[inline]
fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

#[cfg(test)]
mod tests {
    use super::{inner_prod16, scale_copy_vector16, scale_vector32_q26_lshift_18};
    use alloc::vec::Vec;

    #[test]
    fn scale_copy_matches_reference() {
        let input = [1234, -2345, 32767, -32768];
        let gain_q16 = (3 * (1 << 16)) / 2; // 1.5 in Q16
        let mut output = [0i16; 4];
        scale_copy_vector16(&mut output, &input, gain_q16);
        let expected: Vec<i16> = input
            .iter()
            .map(|&x| ((i64::from(gain_q16) * i64::from(x)) >> 16) as i16)
            .collect();
        assert_eq!(output.as_slice(), expected.as_slice());
    }

    #[test]
    fn scale_vector32_matches_reference() {
        let gain_q26 = (3 * (1 << 26)) / 4; // 0.75 in Q26
        let mut data = [1 << 24, -(1 << 23), 123456789, -98765432];
        let expected: Vec<i32> = data
            .iter()
            .map(|&x| ((i64::from(x) * i64::from(gain_q26)) >> 8) as i32)
            .collect();
        scale_vector32_q26_lshift_18(&mut data, gain_q26);
        assert_eq!(data.as_slice(), expected.as_slice());
    }

    #[test]
    fn inner_product_matches_manual_sum() {
        let a = [300, -400, 500, -600, 42];
        let b = [-7, 9, -11, 13, -21];
        let expected = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| i64::from(x) * i64::from(y))
            .sum::<i64>();
        assert_eq!(inner_prod16(&a, &b), expected);
    }

    #[test]
    fn inner_product_empty_slice_is_zero() {
        let empty: [i16; 0] = [];
        assert_eq!(inner_prod16(&empty, &empty), 0);
    }
}
