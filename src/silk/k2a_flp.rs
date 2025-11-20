//! Port of the floating-point `silk_k2a_FLP` helper from
//! `silk/float/k2a_FLP.c`. The routine performs the "step up" transformation,
//! converting reflection coefficients into the forward LPC predictor
//! coefficients used by the FLP analysis pipeline.

/// Converts reflection coefficients to prediction coefficients (floating point).
///
/// The output slice must hold at least `rc.len()` elements; only the first
/// `rc.len()` entries are updated. This is a direct translation of the C
/// reference and therefore uses the same wraparound semantics.
pub fn k2a_flp(a: &mut [f32], rc: &[f32]) {
    let order = rc.len();
    assert!(
        a.len() >= order,
        "output buffer is smaller than the number of reflection coefficients"
    );

    let a = &mut a[..order];

    for (k, &rck) in rc.iter().enumerate() {
        let half = (k + 1) >> 1;
        for n in 0..half {
            let tmp1 = a[n];
            let tmp2 = a[k - n - 1];
            a[n] = tmp1 + tmp2 * rck;
            a[k - n - 1] = tmp2 + tmp1 * rck;
        }
        a[k] = -rck;
    }
}

#[cfg(test)]
mod tests {
    use super::k2a_flp;

    fn assert_close(lhs: &[f32], rhs: &[f32]) {
        const TOLERANCE: f32 = 1e-6;
        assert_eq!(lhs.len(), rhs.len(), "slice lengths must match");
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= TOLERANCE,
                "difference at index {idx} exceeds tolerance: {diff} > {TOLERANCE}"
            );
        }
    }

    #[test]
    fn single_reflection_matches_reference() {
        let mut a = [0.0f32; 1];
        let rc = [0.5f32];

        k2a_flp(&mut a, &rc);

        assert_close(&a, &[-0.5]);
    }

    #[test]
    fn updates_two_stage_coefficients() {
        let mut a = [0.0f32; 2];
        let rc = [0.1f32, -0.3];

        k2a_flp(&mut a, &rc);

        assert_close(&a, &[-0.07, 0.3]);
    }

    #[test]
    fn propagates_three_stage_transformation() {
        let mut a = [0.0f32; 3];
        let rc = [0.25f32, -0.4, 0.1];

        k2a_flp(&mut a, &rc);

        assert_close(&a, &[-0.11, 0.385, -0.1]);
    }

    #[test]
    fn leaves_extra_capacity_untouched() {
        let mut a = [0.0f32, 1.0, 2.0, 3.0];
        let rc = [0.2f32, 0.1];

        k2a_flp(&mut a, &rc);

        assert_close(&a[0..2], &[-0.22, -0.1]);
        assert_eq!(a[2..], [2.0, 3.0]);
    }

    #[test]
    fn handles_empty_inputs() {
        let mut a: [f32; 0] = [];
        let rc: [f32; 0] = [];

        k2a_flp(&mut a, &rc);
    }
}
