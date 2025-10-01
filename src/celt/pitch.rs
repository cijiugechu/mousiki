#![allow(dead_code)]

//! Pitch analysis helpers translated from `celt/pitch.c`.
//!
//! The original implementation provides a collection of small math routines
//! that can be ported in isolation before the full pitch search is
//! reimplemented.  These helpers expose the same behaviour for the float build
//! of CELT while leveraging Rust's slice-based APIs for memory safety.

use crate::celt::math::celt_sqrt;
use crate::celt::types::{OpusVal16, OpusVal32};

/// Computes the inner product between two input vectors.
///
/// Mirrors the behaviour of `celt_inner_prod_c()` from `celt/pitch.c` when the
/// codec is compiled in float mode.  The function asserts that the inputs share
/// the same length and returns the accumulated dot product as a 32-bit float.
pub(crate) fn celt_inner_prod(x: &[OpusVal16], y: &[OpusVal16]) -> OpusVal32 {
    assert_eq!(
        x.len(),
        y.len(),
        "vectors provided to celt_inner_prod must have the same length",
    );

    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| a * b)
        .sum::<OpusVal32>()
}

/// Computes two inner products between the same `x` vector and two targets.
///
/// Ports the scalar `dual_inner_prod_c()` helper from `celt/pitch.c` for the
/// float configuration.  The function evaluates the dot products `(x · y0)` and
/// `(x · y1)` in a single pass over the data, returning the pair as a tuple.
///
/// Callers must supply slices of identical length; this mirrors the original C
/// signature where the routine expects `N` samples for each input.
pub(crate) fn dual_inner_prod(
    x: &[OpusVal16],
    y0: &[OpusVal16],
    y1: &[OpusVal16],
) -> (OpusVal32, OpusVal32) {
    assert!(
        x.len() == y0.len() && x.len() == y1.len(),
        "dual_inner_prod inputs must have the same length"
    );

    let mut xy0 = 0.0;
    let mut xy1 = 0.0;

    for ((&a, &b0), &b1) in x.iter().zip(y0.iter()).zip(y1.iter()) {
        xy0 += a * b0;
        xy1 += a * b1;
    }

    (xy0, xy1)
}

/// Computes the normalised open-loop pitch gain.
///
/// Mirrors the float version of `compute_pitch_gain()` in `celt/pitch.c`, which
/// scales the correlation `xy` by the geometric mean of `xx` and `yy`.  The C
/// routine adds a bias of `1` under the square root to avoid division by zero;
/// the Rust port retains this behaviour to match the reference implementation.
#[inline]
pub(crate) fn compute_pitch_gain(xy: OpusVal32, xx: OpusVal32, yy: OpusVal32) -> OpusVal16 {
    // The float build uses `xy / celt_sqrt(1 + xx * yy)`.
    (xy / celt_sqrt(1.0 + xx * yy)) as OpusVal16
}

#[cfg(test)]
mod tests {
    use super::{celt_inner_prod, compute_pitch_gain, dual_inner_prod};
    use crate::celt::math::celt_sqrt;
    use crate::celt::types::OpusVal16;
    use alloc::vec::Vec;

    fn generate_sequence(len: usize, seed: u32) -> Vec<OpusVal16> {
        let mut state = seed;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let val = ((state >> 8) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            data.push(val as OpusVal16);
        }
        data
    }

    #[test]
    fn inner_product_matches_reference() {
        let x = generate_sequence(64, 0x1234_5678);
        let y = generate_sequence(64, 0x8765_4321);

        let expected: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        let result = celt_inner_prod(&x, &y);

        assert!((expected - result).abs() < 1e-6);
    }

    #[test]
    fn dual_inner_product_matches_individual_computations() {
        let x = generate_sequence(48, 0x4242_4242);
        let y0 = generate_sequence(48, 0x1357_9bdf);
        let y1 = generate_sequence(48, 0x0246_8ace);

        let (dot0, dot1) = dual_inner_prod(&x, &y0, &y1);
        let expected0: f32 = x.iter().zip(y0.iter()).map(|(&a, &b)| a * b).sum();
        let expected1: f32 = x.iter().zip(y1.iter()).map(|(&a, &b)| a * b).sum();

        assert!((dot0 - expected0).abs() < 1e-6);
        assert!((dot1 - expected1).abs() < 1e-6);
    }

    #[test]
    fn pitch_gain_matches_reference_formula() {
        let xy = 0.75f32;
        let xx = 0.5f32;
        let yy = 1.25f32;

        let expected = (xy / celt_sqrt(1.0 + xx * yy)) as OpusVal16;
        let gain = compute_pitch_gain(xy, xx, yy);

        assert!((expected - gain).abs() < 1e-6);
    }
}
