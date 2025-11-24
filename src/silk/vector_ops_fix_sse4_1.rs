//! SSE4.1 inner-product fast path mirroring `silk/fixed/x86/vector_ops_FIX_sse4_1.c`.
//!
//! The C variant accumulates the 16-bit dot product with SSE4.1 intrinsics.
//! Runtime CPU dispatch remains disabled through `OPUS_ARCHMASK`, so this
//! entry point reuses the scalar [`inner_prod16`] helper until x86 runtime
//! selection is wired up.

use crate::silk::vector_ops::inner_prod16;

/// Mirrors `silk_inner_prod16_sse4_1`.
#[inline]
pub fn inner_prod16_sse4_1(in_vec1: &[i16], in_vec2: &[i16]) -> i64 {
    // SIMD and scalar paths are bit-identical; delegate until CPU detection is enabled.
    inner_prod16(in_vec1, in_vec2)
}

#[cfg(test)]
mod tests {
    use super::inner_prod16_sse4_1;
    use crate::silk::vector_ops::inner_prod16;

    #[test]
    fn matches_scalar_inner_product() {
        let a = [500, -400, 300, -200, 100, -50, 25];
        let b = [-7, 9, -11, 13, -15, 17, -19];
        assert_eq!(inner_prod16_sse4_1(&a, &b), inner_prod16(&a, &b));
    }

    #[test]
    fn handles_non_multiple_of_four_lengths() {
        let a = [1i16, 2, 3, 4, 5];
        let b = [-5i16, -4, -3, -2, -1];
        assert_eq!(inner_prod16_sse4_1(&a, &b), inner_prod16(&a, &b));
    }

    #[test]
    fn accepts_empty_inputs() {
        let a: [i16; 0] = [];
        assert_eq!(inner_prod16_sse4_1(&a, &a), 0);
    }
}
