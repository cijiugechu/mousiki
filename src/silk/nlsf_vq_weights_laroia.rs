//! Laroia low-complexity NLSF weighting helper.
//!
//! Port of `silk_NLSF_VQ_weights_laroia` from the reference SILK implementation.

const NLSF_W_Q: i32 = 2;
const WEIGHT_NUMERATOR_Q17: i32 = 1 << (15 + NLSF_W_Q);

#[inline]
fn clamp_to_positive_i16(value: i32) -> i16 {
    value.clamp(0, i32::from(i16::MAX)) as i16
}

#[inline]
fn weight_from_interval(interval: i32) -> i32 {
    WEIGHT_NUMERATOR_Q17 / interval.max(1)
}

/// Compute the Laroia low-complexity NLSF weights.
///
/// `nlsf_q15` contains the line spectral frequencies expressed in Q15. The
/// output slice is filled with the corresponding weights in Q(`NLSF_W_Q`).
/// Both slices must have the same even length.
///
/// # Panics
///
/// Panics if the input slice is empty, has an odd length, or the output slice
/// has a different length.
pub fn nlsf_vq_weights_laroia(weights_q_out: &mut [i16], nlsf_q15: &[i16]) {
    debug_assert!(!nlsf_q15.is_empty());
    debug_assert!(nlsf_q15.len() % 2 == 0);
    debug_assert_eq!(weights_q_out.len(), nlsf_q15.len());

    assert!(!nlsf_q15.is_empty(), "NLSF vector must not be empty");
    assert!(nlsf_q15.len() % 2 == 0, "NLSF vector length must be even");
    assert_eq!(
        weights_q_out.len(),
        nlsf_q15.len(),
        "Output length mismatch"
    );

    let mut tmp1 = weight_from_interval(i32::from(nlsf_q15[0]));
    let mut tmp2 = weight_from_interval(i32::from(nlsf_q15[1]) - i32::from(nlsf_q15[0]));

    weights_q_out[0] = clamp_to_positive_i16(tmp1 + tmp2);
    debug_assert!(weights_q_out[0] > 0);

    for k in (1..nlsf_q15.len() - 1).step_by(2) {
        tmp1 = weight_from_interval(i32::from(nlsf_q15[k + 1]) - i32::from(nlsf_q15[k]));
        weights_q_out[k] = clamp_to_positive_i16(tmp1 + tmp2);
        debug_assert!(weights_q_out[k] > 0);

        tmp2 = weight_from_interval(i32::from(nlsf_q15[k + 2]) - i32::from(nlsf_q15[k + 1]));
        weights_q_out[k + 1] = clamp_to_positive_i16(tmp1 + tmp2);
        debug_assert!(weights_q_out[k + 1] > 0);
    }

    tmp1 = weight_from_interval((1 << 15) - i32::from(nlsf_q15.last().copied().unwrap()));
    let last = weights_q_out.len() - 1;
    weights_q_out[last] = clamp_to_positive_i16(tmp1 + tmp2);
    debug_assert!(weights_q_out[last] > 0);
}

#[cfg(test)]
mod tests {
    use super::nlsf_vq_weights_laroia;

    #[test]
    fn computes_laroia_weights_for_typical_values() {
        let nlsf = [1000, 5000, 12_000, 20_000];
        let mut weights = [0i16; 4];
        nlsf_vq_weights_laroia(&mut weights, &nlsf);
        assert_eq!(weights, [163, 50, 34, 26]);
    }

    #[test]
    fn saturates_when_intervals_are_extreme() {
        let nlsf = [0, 32_767];
        let mut weights = [0i16; 2];
        nlsf_vq_weights_laroia(&mut weights, &nlsf);
        assert_eq!(weights, [i16::MAX, i16::MAX]);
    }
}
