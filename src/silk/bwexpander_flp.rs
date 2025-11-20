//! Floating-point bandwidth expander from `silk/float/bwexpander_FLP.c`.
//!
//! This helper chirps LPC predictor coefficients in-place using a floating-point
//! `chirp` factor, shrinking their magnitudes to widen the effective bandwidth.
//! The loop structure mirrors the reference C implementation so downstream FLP
//! analysis code can evolve without depending on the C sources.

/// Applies a floating-point chirp to the LPC predictor coefficients in `ar`.
///
/// Mirrors `silk_bwexpander_FLP` by multiplying each tap by the progressively
/// decaying `chirp` factor. A `chirp` of `1.0` leaves the predictor unchanged,
/// while smaller values gradually damp later coefficients.
pub fn bwexpander(ar: &mut [f32], chirp: f32) {
    if ar.is_empty() {
        return;
    }

    let Some((last, head)) = ar.split_last_mut() else {
        return;
    };

    let mut cfac = chirp;

    for value in head.iter_mut() {
        *value *= cfac;
        cfac *= chirp;
    }

    *last *= cfac;
}

#[cfg(test)]
mod tests {
    use super::bwexpander;

    #[test]
    fn chirps_predictor_coefficients() {
        let mut ar = [1.0f32, 0.5, -0.25];

        bwexpander(&mut ar, 0.9);

        assert!((ar[0] - 0.9).abs() < 1e-6);
        assert!((ar[1] - 0.405).abs() < 1e-6);
        assert!((ar[2] + 0.18225).abs() < 1e-6);
    }

    #[test]
    fn chirp_of_one_leaves_coefficients_unchanged() {
        let mut ar = [0.75f32, -0.5, 0.25, -0.125];
        let original = ar;

        bwexpander(&mut ar, 1.0);

        assert_eq!(ar, original);
    }

    #[test]
    fn handles_empty_input() {
        let mut ar: [f32; 0] = [];
        bwexpander(&mut ar, 0.5);
    }
}
