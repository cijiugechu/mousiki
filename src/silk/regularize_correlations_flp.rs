//! Floating-point correlation regulariser from `silk/float/regularize_correlations_FLP.c`.
//!
//! The helper adds a small noise term to each diagonal entry of the correlation
//! matrix and to the first element of the correlation vector. This mirrors the
//! stabilisation used before solving the normal equations in the SILK FLP
//! analysis path.

/// Adds `noise` to each diagonal entry of the `dim × dim` matrix `xx_matrix`
/// and to the first element of the correlation vector `xx_vector`.
///
/// Both containers must be large enough to cover the touched elements.
pub fn regularize_correlations_flp(
    xx_matrix: &mut [f32],
    xx_vector: &mut [f32],
    noise: f32,
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
        xx_matrix[idx] += noise;
    }

    xx_vector[0] += noise;
}

#[cfg(test)]
mod tests {
    use super::regularize_correlations_flp;

    #[test]
    fn adds_noise_to_diagonal_and_vector() {
        let mut matrix = [
            10.0f32, 1.0, 2.0, //
            3.0, 20.0, 4.0, //
            5.0, 6.0, 30.0,
        ];
        let mut vector = [100.0f32, 200.0, 300.0];

        regularize_correlations_flp(&mut matrix, &mut vector, 0.25, 3);

        assert_eq!(
            matrix,
            [
                10.25, 1.0, 2.0, //
                3.0, 20.25, 4.0, //
                5.0, 6.0, 30.25,
            ]
        );
        assert_eq!(vector, [100.25, 200.0, 300.0]);
    }

    #[test]
    fn handles_zero_dimension_by_only_touching_vector() {
        let mut matrix = [1.0f32, 2.0, 3.0, 4.0];
        let mut vector = [0.0f32];

        regularize_correlations_flp(&mut matrix, &mut vector, 1.5, 0);

        assert_eq!(matrix, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vector, [1.5]);
    }

    #[test]
    #[should_panic(expected = "matrix must provide room for a 4×4 layout")]
    fn panics_when_matrix_too_small() {
        let mut matrix = [0.0f32; 8]; // only 2x4 entries
        let mut vector = [0.0f32; 4];
        regularize_correlations_flp(&mut matrix, &mut vector, 0.1, 4);
    }

    #[test]
    #[should_panic(expected = "correlation vector must contain at least one element")]
    fn panics_when_vector_empty() {
        let mut matrix = [0.0f32; 1];
        let mut vector = [];
        regularize_correlations_flp(&mut matrix, &mut vector, 0.1, 1);
    }
}
