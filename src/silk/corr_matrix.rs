//! Correlation-matrix helpers from `silk/fixed/corrMatrix_FIX.c`.
//!
//! These routines build the `X' * t` vector and `X' * X` matrix used by the
//! encoder-side least-squares solvers. The implementation mirrors the fixed-
//! point C reference closely, including the incremental updates that avoid
//! recomputing entire dot products when sliding the analysis window.

use crate::silk::inner_prod_aligned::{inner_prod_aligned, inner_prod_aligned_scale};
use crate::silk::sum_sqr_shift::sum_sqr_shift;

/// Computes the correlation vector `X' * t` used by the linear predictor.
///
/// * `xt` — output slice that receives `order` correlation terms.
/// * `x` — source signal with `l + order - 1` samples backing the data matrix.
/// * `t` — target vector with `l` samples.
/// * `l` — column length of the implicit data matrix.
/// * `order` — number of predictor taps (i.e., number of columns in `X`).
/// * `rshifts` — right shift applied to each product when `> 0`.
/// * `arch` — runtime architecture flag retained for API parity (unused for now).
#[allow(clippy::too_many_arguments)]
pub fn corr_vector_fix(
    xt: &mut [i32],
    x: &[i16],
    t: &[i16],
    l: usize,
    order: usize,
    rshifts: i32,
    arch: i32,
) {
    assert!(order > 0, "order must be positive");
    assert!(l > 0, "vector length must be positive");
    assert!(
        xt.len() >= order,
        "output slice must hold at least `order` entries"
    );
    let history = order - 1;
    assert!(
        x.len() >= l + history,
        "`x` must contain at least l + order - 1 samples"
    );
    assert!(t.len() >= l, "`t` must contain at least `l` samples");
    assert!(rshifts >= 0, "rshifts must be non-negative");

    let target = &t[..l];
    let mut start = history;
    for slot in xt.iter_mut().take(order) {
        let column = &x[start..start + l];
        let value = if rshifts > 0 {
            inner_prod_aligned_scale(column, target, rshifts)
        } else {
            debug_assert!(rshifts == 0, "rshifts must be zero for unscaled path");
            inner_prod_aligned(column, target, arch)
        };
        *slot = value;
        start = start.saturating_sub(1);
    }
}

/// Builds the symmetric correlation matrix `X' * X` using incremental updates.
///
/// Returns the total energy and the right-shift count so callers can mirror the
/// reference API signature.
pub fn corr_matrix_fix(xx: &mut [i32], x: &[i16], l: usize, order: usize, arch: i32) -> (i32, i32) {
    assert!(order > 0, "order must be positive");
    assert!(l > 0, "vector length must be positive");
    let history = order - 1;
    assert!(
        x.len() >= l + history,
        "`x` must contain at least l + order - 1 samples"
    );
    assert!(
        xx.len() >= order * order,
        "matrix slice must hold order × order entries"
    );

    let head_len = l + history;
    let (nrg_total, rshifts) = sum_sqr_shift(&x[..head_len]);
    let mut energy = nrg_total;
    let start = history;

    for &sample in x.iter().take(history) {
        energy = sub_rshift32(energy, mul_i16(sample, sample), rshifts);
    }
    set_matrix(xx, order, 0, 0, energy);
    debug_assert!(energy >= 0, "correlation energy must be non-negative");

    let mut diag_energy = energy;
    for j in 1..order {
        let leaving_idx = start + l - j;
        let entering_idx = start - j;
        diag_energy = sub_rshift32(
            diag_energy,
            mul_i16(x[leaving_idx], x[leaving_idx]),
            rshifts,
        );
        diag_energy = add_rshift32(
            diag_energy,
            mul_i16(x[entering_idx], x[entering_idx]),
            rshifts,
        );
        set_matrix(xx, order, j, j, diag_energy);
        debug_assert!(diag_energy >= 0, "correlation energy must be non-negative");
    }

    if order > 1 {
        for lag in 1..order {
            let base_lag_start = start - lag;
            let column_zero = &x[start..start + l];
            let column_lag = &x[base_lag_start..base_lag_start + l];

            let mut cross = if rshifts > 0 {
                inner_prod_aligned_scale(column_zero, column_lag, rshifts)
            } else {
                debug_assert!(rshifts == 0, "rshifts must be zero for unscaled path");
                inner_prod_aligned(column_zero, column_lag, arch)
            };
            set_symmetric(xx, order, lag, 0, cross);

            for j in 1..(order - lag) {
                let leave_a = start + l - j;
                let leave_b = base_lag_start + l - j;
                let enter_a = start - j;
                let enter_b = base_lag_start - j;

                if rshifts > 0 {
                    cross = sub_rshift32(cross, mul_i16(x[leave_a], x[leave_b]), rshifts);
                    cross = add_rshift32(cross, mul_i16(x[enter_a], x[enter_b]), rshifts);
                } else {
                    cross = cross
                        .wrapping_sub(mul_i16(x[leave_a], x[leave_b]))
                        .wrapping_add(mul_i16(x[enter_a], x[enter_b]));
                }

                set_symmetric(xx, order, lag + j, j, cross);
            }
        }
    }

    (nrg_total, rshifts)
}

#[inline]
fn mul_i16(a: i16, b: i16) -> i32 {
    i32::from(a) * i32::from(b)
}

#[inline]
fn add_rshift32(acc: i32, value: i32, shift: i32) -> i32 {
    acc.wrapping_add(rshift32(value, shift))
}

#[inline]
fn sub_rshift32(acc: i32, value: i32, shift: i32) -> i32 {
    acc.wrapping_sub(rshift32(value, shift))
}

#[inline]
fn rshift32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else if shift >= 31 {
        if value >= 0 { 0 } else { -1 }
    } else {
        value >> shift
    }
}

#[inline]
fn set_symmetric(matrix: &mut [i32], order: usize, row: usize, col: usize, value: i32) {
    set_matrix(matrix, order, row, col, value);
    set_matrix(matrix, order, col, row, value);
}

#[inline]
fn set_matrix(matrix: &mut [i32], order: usize, row: usize, col: usize, value: i32) {
    let index = row * order + col;
    matrix[index] = value;
}

#[cfg(test)]
mod tests {
    use super::{corr_matrix_fix, corr_vector_fix};
    use crate::silk::sum_sqr_shift::sum_sqr_shift;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn corr_vector_matches_reference_without_shift() {
        let x = [5, -3, 2, 7, -1, 4];
        let t = [10, -20, 15, -5];
        let l = 4;
        let order = 3;
        let mut actual = [0i32; 3];
        corr_vector_fix(&mut actual, &x, &t, l, order, 0, 0);
        let expected = reference_corr_vector(&x, &t, l, order, 0);
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn corr_vector_matches_reference_with_shift() {
        let x = [1000, -2000, 1500, -500, 200, -100, 50];
        let t = [-7, 13, -9, 5, -3];
        let l = 5;
        let order = 3;
        let mut actual = [0i32; 3];
        corr_vector_fix(&mut actual, &x, &t, l, order, 2, 0);
        let expected = reference_corr_vector(&x, &t, l, order, 2);
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn corr_matrix_matches_reference_for_zero_shift() {
        let x = [3, -1, 4, -2, 5, -3, 6];
        let l = 4;
        let order = 3;
        let mut matrix = [0i32; 9];
        let (nrg, rshift) = corr_matrix_fix(&mut matrix, &x, l, order, 0);
        let (expected_nrg, expected_shift) = sum_sqr_shift(&x[..l + order - 1]);
        assert_eq!(nrg, expected_nrg);
        assert_eq!(rshift, expected_shift);
        let expected = reference_corr_matrix(&x, l, order, rshift);
        assert_eq!(matrix.as_slice(), expected.as_slice());
    }

    #[test]
    fn corr_matrix_matches_reference_with_shift() {
        let x = [4000, -3000, 2000, -1000, 500, -250, 125, -60];
        let l = 5;
        let order = 4;
        let mut matrix = [0i32; 16];
        let (nrg, rshift) = corr_matrix_fix(&mut matrix, &x, l, order, 0);
        let (expected_nrg, expected_shift) = sum_sqr_shift(&x[..l + order - 1]);
        assert_eq!(nrg, expected_nrg);
        assert_eq!(rshift, expected_shift);
        let expected = reference_corr_matrix(&x, l, order, rshift);
        assert_eq!(matrix.as_slice(), expected.as_slice());
    }

    fn reference_corr_vector(x: &[i16], t: &[i16], l: usize, order: usize, shift: i32) -> Vec<i32> {
        let history = order - 1;
        let mut result = Vec::with_capacity(order);
        for lag in 0..order {
            let start = history - lag;
            let mut acc = 0i32;
            for i in 0..l {
                let product = i32::from(x[start + i]) * i32::from(t[i]);
                acc += shift_product(product, shift);
            }
            result.push(acc);
        }
        result
    }

    fn reference_corr_matrix(x: &[i16], l: usize, order: usize, shift: i32) -> Vec<i32> {
        let history = order - 1;
        let mut result = vec![0i32; order * order];
        for row in 0..order {
            for col in row..order {
                let row_start = history - row;
                let col_start = history - col;
                let mut acc = 0i32;
                for i in 0..l {
                    let product = i32::from(x[row_start + i]) * i32::from(x[col_start + i]);
                    acc += shift_product(product, shift);
                }
                result[row * order + col] = acc;
                result[col * order + row] = acc;
            }
        }
        result
    }

    fn shift_product(product: i32, shift: i32) -> i32 {
        if shift > 0 { product >> shift } else { product }
    }
}
