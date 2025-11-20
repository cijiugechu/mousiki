//! Floating-point warped autocorrelation helper from
//! `silk/float/warped_autocorrelation_FLP.c`.
//!
//! Computes warped autocorrelations for an even LPC order using the same
//! two-stage all-pass structure as the reference implementation.

use crate::silk::MAX_SHAPE_LPC_ORDER;

/// Mirrors `silk_warped_autocorrelation_FLP`.
pub fn warped_autocorrelation_flp(
    corr: &mut [f32],
    input: &[f32],
    warping: f32,
    order: usize,
) {
    assert!(
        order <= MAX_SHAPE_LPC_ORDER,
        "order must be <= {MAX_SHAPE_LPC_ORDER}"
    );
    assert!(order.is_multiple_of(2), "order must be even");
    assert!(
        corr.len() > order,
        "corr must expose at least {} slots",
        order + 1
    );

    let warping = f64::from(warping);

    let mut state = [0f64; MAX_SHAPE_LPC_ORDER + 1];
    let mut accum = [0f64; MAX_SHAPE_LPC_ORDER + 1];

    for &sample in input {
        let mut tmp1 = f64::from(sample);

        let mut section = 0;
        while section < order {
            let tmp2 = state[section] + warping * state[section + 1] - warping * tmp1;
            state[section] = tmp1;
            accum[section] += state[0] * tmp1;

            tmp1 = state[section + 1] + warping * state[section + 2] - warping * tmp2;
            state[section + 1] = tmp2;
            accum[section + 1] += state[0] * tmp2;

            section += 2;
        }

        state[order] = tmp1;
        accum[order] += state[0] * tmp1;
    }

    for (dst, &src) in corr.iter_mut().take(order + 1).zip(accum.iter()) {
        *dst = src as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::warped_autocorrelation_flp;
    use crate::silk::MAX_SHAPE_LPC_ORDER;

    #[test]
    fn matches_reference_output() {
        let input = [0.2f32, -0.4, 0.25, -0.1, 0.05];
        let warping = 0.3;
        let order = 4;
        let mut corr = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];

        warped_autocorrelation_flp(&mut corr[..order + 1], &input, warping, order);

        let expected = [
            0.2750000059604645,
            -0.2486477941274643,
            0.1916804015636444,
            -0.1361631602048874,
            0.09137232601642609,
        ];
        for (got, exp) in corr.iter().take(order + 1).zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-6,
                "expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn zero_warping_reduces_to_simple_products() {
        let input = [0.1f32, 0.2, 0.3];
        let order = 2;
        let mut corr = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];

        warped_autocorrelation_flp(&mut corr[..order + 1], &input, 0.0, order);

        let expected = [0.14, 0.08, 0.03];
        for (got, exp) in corr.iter().take(order + 1).zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-6,
                "expected {exp}, got {got}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "order must be even")]
    fn rejects_odd_order() {
        let mut corr = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];
        warped_autocorrelation_flp(&mut corr[..3], &[0.0; 4], 0.5, 3);
    }
}
