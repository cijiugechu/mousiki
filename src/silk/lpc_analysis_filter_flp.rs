//! Port of `silk/float/LPC_analysis_filter_FLP.c`.
//!
//! Applies an LPC analysis filter to floating-point samples, mirroring the
//! order-specific kernels from the reference implementation.

/// Apply the floating-point LPC analysis filter.
///
/// The routine writes `length` samples into `residual`, zeroing the first
/// `order` entries to mirror the C implementation's implicit zero state.
/// Supported orders match the reference specialisations (6, 8, 10, 12, or 16).
pub fn lpc_analysis_filter_flp(
    residual: &mut [f32],
    pred_coeffs: &[f32],
    input: &[f32],
    length: usize,
    order: usize,
) {
    assert!(length <= residual.len(), "residual buffer too small");
    assert!(length <= input.len(), "input buffer too small");
    assert!(order <= pred_coeffs.len(), "coefficient slice too short");
    assert!(order <= length, "filter order cannot exceed length");
    assert!(
        matches!(order, 6 | 8 | 10 | 12 | 16),
        "unsupported LPC order: {order}"
    );

    match order {
        6 => filter_order::<6>(residual, pred_coeffs, input, length),
        8 => filter_order::<8>(residual, pred_coeffs, input, length),
        10 => filter_order::<10>(residual, pred_coeffs, input, length),
        12 => filter_order::<12>(residual, pred_coeffs, input, length),
        16 => filter_order::<16>(residual, pred_coeffs, input, length),
        _ => unreachable!("unsupported LPC order already filtered by assert"),
    }

    for out in &mut residual[..order] {
        *out = 0.0;
    }
}

fn filter_order<const ORDER: usize>(
    residual: &mut [f32],
    pred_coeffs: &[f32],
    input: &[f32],
    length: usize,
) {
    for ix in ORDER..length {
        let mut lpc_pred = 0.0f32;
        for k in 0..ORDER {
            lpc_pred += input[ix - 1 - k] * pred_coeffs[k];
        }
        residual[ix] = input[ix] - lpc_pred;
    }
}

#[cfg(test)]
mod tests {
    use super::lpc_analysis_filter_flp;

    fn assert_close(lhs: &[f32], rhs: &[f32]) {
        const TOLERANCE: f32 = 1e-6;
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "slices must have identical lengths when comparing"
        );
        for (index, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= TOLERANCE,
                "difference at index {index} exceeds tolerance: {diff} > {TOLERANCE}"
            );
        }
    }

    #[test]
    fn zeroes_initial_state() {
        let input = [0.5f32, -0.25, 0.75, -0.5, 1.25, -1.0, 0.5, 0.25];
        let coeffs = [0.2f32; 6];
        let mut residual = [1.0f32; 8];

        lpc_analysis_filter_flp(&mut residual, &coeffs, &input, input.len(), coeffs.len());

        assert_eq!(&residual[..coeffs.len()], &[0.0; 6]);
    }

    #[test]
    fn computes_residual_for_order_six() {
        let input = [
            0.5f32, -0.25, 0.75, -0.5, 1.25, -1.0, 0.5, 0.25, -0.75, 0.125,
        ];
        let coeffs = [0.2, -0.1, 0.05, -0.025, 0.0125, -0.00625];
        let mut residual = [0.0f32; 10];

        lpc_analysis_filter_flp(&mut residual, &coeffs, &input, input.len(), coeffs.len());

        let expected = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.875,
            -0.035_937_5,
            -0.657_812_5,
            0.231_25,
        ];
        assert_close(&residual, &expected);
    }

    #[test]
    fn computes_residual_for_order_eight() {
        let input = [
            1.0f32, 0.5, -0.25, 0.75, -0.5, 1.25, -1.0, 0.5, 0.25, -0.75, 0.125, -0.625,
        ];
        let coeffs = [
            0.25,
            -0.125,
            0.0625,
            -0.03125,
            0.015_625,
            -0.007_812_5,
            0.003_906_25,
            -0.001_953_125,
        ];
        let mut residual = [0.0f32; 12];

        lpc_analysis_filter_flp(&mut residual, &coeffs, &input, input.len(), coeffs.len());

        let expected = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.107_421_875,
            -0.632_812_5,
            0.254_394_53,
            -0.721_191_4,
        ];
        assert_close(&residual, &expected);
    }

    #[test]
    #[should_panic(expected = "unsupported LPC order: 7")]
    fn rejects_unsupported_order() {
        let mut residual = [0.0f32; 8];
        let coeffs = [0.1f32; 7];
        let input = [0.0f32; 8];

        lpc_analysis_filter_flp(&mut residual, &coeffs, &input, input.len(), coeffs.len());
    }
}
