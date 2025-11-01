//! Quantisation-error helper for SILK NLSF vector codebooks.
//!
//! Port of `silk_NLSF_VQ` from the reference SILK implementation.

/// Compute the quantisation error for each codebook entry when encoding an NLSF
/// vector.
///
/// `in_q15` contains the target NLSF vector in Q15, and the `codebook_q8` /
/// `weights_q9` slices describe `K` candidate vectors in Q8 together with their
/// predictive weights in Q9. The output slice must have length `K` and receives
/// the accumulated absolute prediction errors in Q24, mirroring the C helper.
///
/// # Panics
///
/// Panics if the input order is not even or if any of the provided slices have
/// inconsistent lengths.
pub fn nlsf_vq(err_q24: &mut [i32], in_q15: &[i16], codebook_q8: &[u8], weights_q9: &[i16]) {
    debug_assert!(in_q15.len().is_multiple_of(2));

    if err_q24.is_empty() {
        assert!(codebook_q8.is_empty(), "codebook must be empty when K = 0");
        assert!(
            weights_q9.is_empty(),
            "weight table must be empty when K = 0"
        );
        return;
    }

    assert!(
        !in_q15.is_empty() && in_q15.len().is_multiple_of(2),
        "NLSF order must be non-zero and even",
    );

    assert_eq!(
        codebook_q8.len(),
        weights_q9.len(),
        "codebook vectors and weights must have the same length",
    );

    let lpc_order = in_q15.len();
    let expected_len = err_q24
        .len()
        .checked_mul(lpc_order)
        .expect("codebook length overflow");
    assert_eq!(
        codebook_q8.len(),
        expected_len,
        "codebook length must match K * LPC_order",
    );

    for (vec_idx, err_out) in err_q24.iter_mut().enumerate() {
        let base = vec_idx * lpc_order;
        let vector = &codebook_q8[base..base + lpc_order];
        let weights = &weights_q9[base..base + lpc_order];

        let mut sum_error_q24 = 0i32;
        let mut pred_q24 = 0i32;

        let mut m = lpc_order - 2;
        loop {
            let diff_q15 = i32::from(in_q15[m + 1]) - (i32::from(vector[m + 1]) << 7);
            let diffw_q24 = smulbb(diff_q15, weights[m + 1]);
            let delta_q24 = diffw_q24 - (pred_q24 >> 1);
            sum_error_q24 = sum_error_q24.wrapping_add(silk_abs(delta_q24));
            pred_q24 = diffw_q24;

            let diff_q15 = i32::from(in_q15[m]) - (i32::from(vector[m]) << 7);
            let diffw_q24 = smulbb(diff_q15, weights[m]);
            let delta_q24 = diffw_q24 - (pred_q24 >> 1);
            sum_error_q24 = sum_error_q24.wrapping_add(silk_abs(delta_q24));
            pred_q24 = diffw_q24;

            debug_assert!(sum_error_q24 >= 0);

            if m == 0 {
                break;
            }
            m -= 2;
        }

        *err_out = sum_error_q24;
    }
}

#[inline]
fn smulbb(a_q15: i32, weight_q9: i16) -> i32 {
    i32::from(a_q15 as i16) * i32::from(weight_q9)
}

#[inline]
fn silk_abs(value: i32) -> i32 {
    if value >= 0 {
        value
    } else {
        value.wrapping_neg()
    }
}

#[cfg(test)]
mod tests {
    use super::nlsf_vq;

    #[test]
    fn computes_errors_for_multiple_codebook_vectors() {
        let in_q15 = [1000, -5000, 2000, -1000];
        let codebook_q8 = [100u8, 120, 140, 160, 90, 110, 130, 150];
        let weights_q9 = [500i16, 600, 700, 800, 400, 500, 600, 700];

        let mut errors = [0i32; 2];
        nlsf_vq(&mut errors, &in_q15, &codebook_q8, &weights_q9);

        assert_eq!(errors, [26_588_000, 21_564_000]);
    }

    #[test]
    fn handles_different_target_vectors() {
        let in_q15 = [3000, -2000, 1000, -1500];
        let codebook_q8 = [100u8, 120, 140, 160, 90, 110, 130, 150];
        let weights_q9 = [500i16, 600, 700, 800, 400, 500, 600, 700];

        let mut errors = [0i32; 2];
        nlsf_vq(&mut errors, &in_q15, &codebook_q8, &weights_q9);

        assert_eq!(errors, [25_438_000, 20_589_000]);
    }
}
