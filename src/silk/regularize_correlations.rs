//! Port of the fixed-point correlation regularisation helper from
//! `silk/fixed/regularize_correlations_FIX.c`.
//!
//! The routine adds a small noise term to the diagonal of the correlation
//! matrix `XX` and to the first element of the correlation vector `xx`. This
//! mirrors the behaviour in the SILK encoder where the adjustment helps to
//! stabilise subsequent linear solves.

/// Adds `noise` to each diagonal entry of the `dim × dim` matrix `xx_matrix`
/// and to the first element of the correlation vector `xx_vector`.
///
/// Both containers must be large enough to cover the touched elements.
pub fn regularize_correlations(
    xx_matrix: &mut [i32],
    xx_vector: &mut [i32],
    noise: i32,
    dim: usize,
) {
    assert!(
        xx_matrix.len() >= dim * dim,
        "matrix must provide room for a {dim}×{dim} layout"
    );
    assert!(
        !xx_vector.is_empty(),
        "correlation vector must contain at least one element"
    );

    for i in 0..dim {
        let idx = i * dim + i;
        xx_matrix[idx] = xx_matrix[idx].wrapping_add(noise);
    }

    xx_vector[0] = xx_vector[0].wrapping_add(noise);
}

#[cfg(test)]
mod tests {
    use super::regularize_correlations;

    #[test]
    fn adds_noise_to_diagonal_and_vector() {
        let mut matrix = [
            10, 1, 2, //
            3, 20, 4, //
            5, 6, 30,
        ];
        let mut vector = [100, 200, 300];

        regularize_correlations(&mut matrix, &mut vector, 7, 3);

        assert_eq!(
            matrix,
            [
                17, 1, 2, //
                3, 27, 4, //
                5, 6, 37,
            ]
        );
        assert_eq!(vector, [107, 200, 300]);
    }

    #[test]
    fn handles_zero_dimension_by_only_touching_vector() {
        let mut matrix = [1, 2, 3, 4];
        let mut vector = [i32::MAX];

        regularize_correlations(&mut matrix, &mut vector, 1, 0);

        assert_eq!(matrix, [1, 2, 3, 4]);
        assert_eq!(vector, [i32::MIN]); // wrapping add
    }
}
