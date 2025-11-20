//! Floating-point correlation helpers from `silk/float/corrMatrix_FLP.c`.
//!
//! These routines build the `X' * t` vector and `X' * X` matrix used by the
//! floating-point predictor analysis paths. They reuse the shared inner-product
//! and energy kernels so callers observe the same accumulation order as the C
//! reference.

use crate::silk::energy_flp::energy;
use crate::silk::inner_product_flp::inner_product_flp;

/// Computes the correlation vector `X' * t` used by the floating-point
/// least-squares solvers.
#[allow(clippy::too_many_arguments)]
pub fn corr_vector_flp(
    xt: &mut [f32],
    x: &[f32],
    t: &[f32],
    l: usize,
    order: usize,
    _arch: i32,
)
{
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

    let target = &t[..l];
    let mut column_start = history;
    for slot in xt.iter_mut().take(order) {
        let column = &x[column_start..column_start + l];
        *slot = inner_product_flp(column, target) as f32;
        column_start = column_start.saturating_sub(1);
    }
}

/// Builds the symmetric correlation matrix `X' * X` using the floating-point
/// analysis buffers.
pub fn corr_matrix_flp(xx: &mut [f32], x: &[f32], l: usize, order: usize, _arch: i32) {
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

    let start = history;
    let column0 = &x[start..start + l];

    let mut diag_energy = energy(column0);
    set_matrix(xx, order, 0, 0, diag_energy as f32);

    for j in 1..order {
        let entering = start - j;
        let leaving = start + l - j;
        diag_energy += f64::from(x[entering]) * f64::from(x[entering])
            - f64::from(x[leaving]) * f64::from(x[leaving]);
        set_matrix(xx, order, j, j, diag_energy as f32);
    }

    if order == 1 {
        return;
    }

    let mut lag_start = history - 1;
    for lag in 1..order {
        let mut cross = inner_product_flp(column0, &x[lag_start..lag_start + l]);
        set_symmetric(xx, order, lag, 0, cross as f32);

        for j in 1..(order - lag) {
            let leave_a = start + l - j;
            let leave_b = lag_start + l - j;
            let enter_a = start - j;
            let enter_b = lag_start - j;

            cross += f64::from(x[enter_a]) * f64::from(x[enter_b])
                - f64::from(x[leave_a]) * f64::from(x[leave_b]);
            set_symmetric(xx, order, lag + j, j, cross as f32);
        }

        if lag_start == 0 {
            break;
        }
        lag_start -= 1;
    }
}

#[inline]
fn set_symmetric(matrix: &mut [f32], order: usize, row: usize, col: usize, value: f32) {
    set_matrix(matrix, order, row, col, value);
    set_matrix(matrix, order, col, row, value);
}

#[inline]
fn set_matrix(matrix: &mut [f32], order: usize, row: usize, col: usize, value: f32) {
    let index = row * order + col;
    matrix[index] = value;
}

#[cfg(test)]
mod tests {
    use super::{corr_matrix_flp, corr_vector_flp};
    use crate::silk::inner_product_flp::inner_product_flp;
    use alloc::vec;
    use alloc::vec::Vec;

    fn reference_corr_vector(x: &[f32], t: &[f32], l: usize, order: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(order);
        for lag in 0..order {
            let start = order - 1 - lag;
            let column = &x[start..start + l];
            result.push(inner_product_flp(column, &t[..l]) as f32);
        }
        result
    }

    fn reference_corr_matrix(x: &[f32], l: usize, order: usize) -> Vec<f32> {
        let mut matrix = vec![0.0f32; order * order];
        for row in 0..order {
            let row_ptr = order - 1 - row;
            let row_column = &x[row_ptr..row_ptr + l];
            for col in 0..order {
                let col_ptr = order - 1 - col;
                let col_column = &x[col_ptr..col_ptr + l];
                let dot = inner_product_flp(row_column, col_column) as f32;
                matrix[row * order + col] = dot;
            }
        }
        matrix
    }

    fn assert_close(lhs: &[f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len(), "slices must match in length");
        for (index, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-5,
                "difference at index {index} exceeds tolerance: {diff}"
            );
        }
    }

    #[test]
    fn corr_vector_matches_reference() {
        let x = [0.5f32, -1.25, 2.0, 3.5, -0.75, 1.25, -2.5];
        let t = [1.0f32, -0.5, 0.25, -0.125, 0.0625];
        let l = 5;
        let order = 3;
        let mut actual = [0.0f32; 3];
        corr_vector_flp(&mut actual, &x, &t, l, order, 0);
        let expected = reference_corr_vector(&x, &t, l, order);
        assert_close(&actual, &expected);
    }

    #[test]
    fn corr_matrix_matches_reference() {
        let x = [0.5f32, -1.0, 2.5, 3.0, -0.75, 0.25, 1.0, -2.0];
        let l = 5;
        let order = 4;
        let mut actual = [0.0f32; 16];
        corr_matrix_flp(&mut actual, &x, l, order, 0);
        let expected = reference_corr_matrix(&x, l, order);
        assert_close(&actual, &expected);
    }

    #[test]
    fn handles_single_tap_order() {
        let x = [1.0f32, 2.0, 3.0];
        let l = 2;
        let order = 1;
        let mut xt = [0.0f32; 1];
        corr_vector_flp(&mut xt, &x, &x[1..], l, order, 0);
        assert!((xt[0] - inner_product_flp(&x[0..2], &x[1..3]) as f32).abs() < 1e-6);

        let mut xx = [0.0f32; 1];
        corr_matrix_flp(&mut xx, &x, l, order, 0);
        assert!((xx[0] - 5.0).abs() < 1e-6);
    }
}
