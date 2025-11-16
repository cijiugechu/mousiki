//! Floating-point inner product helper from `silk/float/inner_product_FLP.c`.
//!
//! The SILK floating-point analysis paths still rely on a few DSP kernels
//! implemented in C.  `silk_inner_product_FLP_c` is a lightweight helper that
//! accumulates the dot product of two `silk_float` buffers using double
//! precision.  Keeping the accumulation in `f64` avoids the precision loss that
//! would occur if we truncated each partial product back to `f32`.

/// Computes the inner product of two floating-point vectors using `f64`
/// accumulation.
///
/// Mirrors `silk_inner_product_FLP_c` by unrolling the main loop four times and
/// expressing the remaining tail with a scalar loop.  The function panics when
/// the input slices differ in length, mirroring the implicit contract from the
/// C implementation.
pub fn inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    assert_eq!(
        data1.len(),
        data2.len(),
        "input vectors must have identical lengths"
    );

    let len = data1.len();
    let mut idx = 0;
    let mut result = 0.0f64;

    while idx + 3 < len {
        result += f64::from(data1[idx]) * f64::from(data2[idx])
            + f64::from(data1[idx + 1]) * f64::from(data2[idx + 1])
            + f64::from(data1[idx + 2]) * f64::from(data2[idx + 2])
            + f64::from(data1[idx + 3]) * f64::from(data2[idx + 3]);
        idx += 4;
    }

    while idx < len {
        result += f64::from(data1[idx]) * f64::from(data2[idx]);
        idx += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::inner_product_flp;

    #[test]
    fn matches_reference_dot_product() {
        let v1 = [0.5, -1.75, 3.25, -4.0, 2.0, -0.25, 0.125];
        let v2 = [1.0, 0.25, -2.0, 4.5, -1.5, 2.25, -0.625];
        let expected: f64 = v1
            .iter()
            .zip(v2.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        assert_eq!(inner_product_flp(&v1, &v2), expected);
    }

    #[test]
    fn handles_non_multiple_of_four_lengths() {
        let v1 = [0.0f32, 1.0, 2.0];
        let v2 = [1.0f32, 0.5, -1.0];
        let expected: f64 = v1
            .iter()
            .zip(v2.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        assert_eq!(inner_product_flp(&v1, &v2), expected);
    }

    #[test]
    fn empty_vectors_return_zero() {
        let empty: [f32; 0] = [];
        assert_eq!(inner_product_flp(&empty, &empty), 0.0);
    }
}
