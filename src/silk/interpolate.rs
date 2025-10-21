//! Utilities for interpolating LPC parameter vectors.
//!
//! Port of `silk_interpolate` from `silk/interpolate.c` in the reference
//! implementation. The routine linearly interpolates between two LPC
//! coefficient vectors using a Q2 interpolation factor.

/// Maximum LPC order supported by the SILK codec.
pub const MAX_LPC_ORDER: usize = 16;

/// Interpolates between two LPC coefficient vectors.
///
/// `ifact_q2` is given in Q2 format and must be in the range `0..=4`. The
/// slices are expected to have the same length and contain at most
/// [`MAX_LPC_ORDER`] elements.
pub fn interpolate(xi: &mut [i16], x0: &[i16], x1: &[i16], ifact_q2: i32) {
    assert_eq!(xi.len(), x0.len());
    assert_eq!(xi.len(), x1.len());
    assert!(xi.len() <= MAX_LPC_ORDER);
    assert!((0..=4).contains(&ifact_q2));

    let factor = i32::from(ifact_q2 as i16);

    for ((xi_i, &x0_i), &x1_i) in xi.iter_mut().zip(x0.iter()).zip(x1.iter()) {
        let diff = (i32::from(x1_i) - i32::from(x0_i)) as i16;
        let product = i32::from(diff) * factor;
        let value = i32::from(x0_i) + (product >> 2);
        *xi_i = value as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_first_vector_when_factor_zero() {
        let x0 = [1, -2, 3000, -16384];
        let x1 = [5, 6, -3000, 16384];
        let mut xi = [0; 4];

        interpolate(&mut xi, &x0, &x1, 0);

        assert_eq!(xi, x0);
    }

    #[test]
    fn returns_second_vector_when_factor_full() {
        let x0 = [-30000, 1234, -200];
        let x1 = [30000, -4321, 200];
        let mut xi = [0; 3];

        interpolate(&mut xi, &x0, &x1, 4);

        assert_eq!(xi, x1);
    }

    #[test]
    fn interpolates_half_way_for_factor_two() {
        let x0 = [1000, -1000, 0, 16384];
        let x1 = [2000, -2000, 4000, -16384];
        let mut xi = [0; 4];

        interpolate(&mut xi, &x0, &x1, 2);

        assert_eq!(xi, [1500, -1500, 2000, 0]);
    }

    #[test]
    fn matches_reference_behavior_for_large_differences() {
        let x0 = [-30000, 32000];
        let x1 = [30000, -32000];
        let mut xi = [0; 2];

        interpolate(&mut xi, &x0, &x1, 3);

        // Expect rounding toward negative infinity on the Q2 shift and
        // wrapping behaviour from the 16-bit cast.
        assert_eq!(xi, [31384, -32384]);
    }
}
