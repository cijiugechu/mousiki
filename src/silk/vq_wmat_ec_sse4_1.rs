//! SSE4.1 fast path for the entropy-constrained, matrix-weighted VQ used by
//! the SILK encoder when searching the 5-tap LTP codebook.
//!
//! The C implementation in `silk/x86/VQ_WMat_EC_sse4_1.c` relies on SSE4.1
//! intrinsics to accelerate the same math performed by the scalar
//! [`vq_wmat_ec`] helper. Runtime CPU dispatch is currently disabled via
//! `OPUS_ARCHMASK`, so this Rust version delegates to the safe scalar helper
//! while keeping the dedicated entry point that the x86 dispatch table expects.

use crate::silk::vq_wmat_ec::{vq_wmat_ec, VqWMatEcResult, LTP_ORDER};

/// Mirrors `silk_VQ_WMat_EC_sse4_1`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn vq_wmat_ec_sse4_1(
    xx_q17: &[i32; LTP_ORDER * LTP_ORDER],
    x_x_q17: &[i32; LTP_ORDER],
    cb_q7: &[[i8; LTP_ORDER]],
    cb_gain_q7: &[u8],
    cl_q5: &[u8],
    subfr_len: i32,
    max_gain_q7: i32,
) -> VqWMatEcResult {
    // The SIMD fast path produces identical results to the scalar helper.
    // We reuse the Rust translation until runtime CPU dispatch is enabled.
    vq_wmat_ec(
        xx_q17,
        x_x_q17,
        cb_q7,
        cb_gain_q7,
        cl_q5,
        subfr_len,
        max_gain_q7,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn matches_scalar_vq() {
        let xx_q17 = [
            19_000, 900, 1_100, 1_200, 1_400, 900, 19_000, 1_000, 1_100, 1_200, 1_100, 1_000,
            19_000, 1_050, 1_100, 1_200, 1_100, 1_050, 19_000, 1_000, 1_400, 1_200, 1_100, 1_000,
            19_000,
        ];
        let x_x_q17 = [250, 260, 270, 280, 290];
        let cb_q7 = [[3, 2, 1, 0, -1], [-2, -2, -2, -2, -2]];
        let cb_gain_q7 = [18u8, 10u8];
        let cl_q5 = [10u8, 6u8];
        let subfr_len = 80;
        let max_gain_q7 = 25;

        let scalar = vq_wmat_ec(
            &xx_q17,
            &x_x_q17,
            &cb_q7,
            &cb_gain_q7,
            &cl_q5,
            subfr_len,
            max_gain_q7,
        );
        let simd = vq_wmat_ec_sse4_1(
            &xx_q17,
            &x_x_q17,
            &cb_q7,
            &cb_gain_q7,
            &cl_q5,
            subfr_len,
            max_gain_q7,
        );

        assert_eq!(scalar, simd);
    }

    #[test]
    fn accepts_arbitrary_codebook_layout() {
        let mut cb = vec![[0i8; LTP_ORDER]; 4];
        for (row_idx, row) in cb.iter_mut().enumerate() {
            for (col_idx, value) in row.iter_mut().enumerate() {
                *value = (row_idx as i8 - (2 * col_idx as i8)) as i8;
            }
        }
        let xx_q17 = [
            25_000, 0, 0, 0, 0, 0, 25_000, 0, 0, 0, 0, 0, 25_000, 0, 0, 0, 0, 0, 25_000, 0, 0, 0,
            0, 0, 25_000,
        ];
        let x_x_q17 = [0, 0, 0, 0, 0];
        let cb_gain_q7 = [0u8; 4];
        let cl_q5 = [0u8; 4];
        let subfr_len = 20;
        let max_gain_q7 = 100;

        let simd = vq_wmat_ec_sse4_1(
            &xx_q17,
            &x_x_q17,
            &cb,
            &cb_gain_q7,
            &cl_q5,
            subfr_len,
            max_gain_q7,
        );
        let scalar = vq_wmat_ec(
            &xx_q17,
            &x_x_q17,
            &cb,
            &cb_gain_q7,
            &cl_q5,
            subfr_len,
            max_gain_q7,
        );

        assert_eq!(simd, scalar);
    }
}
