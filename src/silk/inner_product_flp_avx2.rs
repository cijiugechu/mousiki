//! AVX2 fast path for the floating-point inner product helper.
//!
//! The C implementation in `silk/float/x86/inner_product_FLP_avx2.c` uses AVX2
//! intrinsics to accumulate eight products per iteration. Runtime CPU
//! detection is still stubbed (`OPUS_ARCHMASK` is zero), so this Rust port
//! forwards to the scalar [`inner_product_flp`] helper while keeping the entry
//! point that the x86 dispatch table expects.

use crate::silk::inner_product_flp::inner_product_flp;

/// Mirrors `silk_inner_product_FLP_avx2`.
#[inline]
pub fn inner_product_flp_avx2(data1: &[f32], data2: &[f32]) -> f64 {
    // The SIMD fast path produces the same result as the scalar helper; reuse
    // the safe Rust implementation until runtime dispatch is enabled.
    inner_product_flp(data1, data2)
}

#[cfg(test)]
mod tests {
    use super::inner_product_flp_avx2;
    use crate::silk::inner_product_flp::inner_product_flp;
    use alloc::vec;

    #[test]
    fn matches_scalar_inner_product() {
        let data1 = [0.5f32, -1.75, 3.25, -4.0, 2.0, -0.25, 0.125, 0.375];
        let data2 = [1.0f32, 0.25, -2.0, 4.5, -1.5, 2.25, -0.625, 1.5];

        assert_eq!(
            inner_product_flp_avx2(&data1, &data2),
            inner_product_flp(&data1, &data2)
        );
    }

    #[test]
    fn supports_arbitrary_lengths() {
        let mut data1 = vec![0.0f32; 17];
        let mut data2 = vec![0.0f32; 17];
        for (i, (a, b)) in data1.iter_mut().zip(data2.iter_mut()).enumerate() {
            *a = (i as f32) - 4.0;
            *b = (i as f32) * 0.25 - 1.0;
        }

        assert_eq!(
            inner_product_flp_avx2(&data1, &data2),
            inner_product_flp(&data1, &data2)
        );
    }
}
