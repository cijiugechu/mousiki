//! Floating-point autocorrelation helper from `silk/float/autocorrelation_FLP.c`.
//!
//! The routine computes `correlation_count` taps of the autocorrelation
//! sequence for `input_data`, truncating the tap count to the input length just
//! like the reference implementation. Results are written into `results`
//! without allocating, keeping the FLP analysis path dependency-light.

use crate::silk::inner_product_flp::inner_product_flp;

/// Computes the autocorrelation sequence for the provided floating-point input.
///
/// Mirrors `silk_autocorrelation_FLP`: if `correlation_count` exceeds
/// `input_data.len()`, the tap count is clamped to the input length. Only the
/// first `min(correlation_count, input_data.len())` entries in `results` are
/// written; the remainder is left untouched.
pub fn autocorrelation(results: &mut [f32], input_data: &[f32], correlation_count: usize) {
    let count = correlation_count.min(input_data.len());
    assert!(
        results.len() >= count,
        "results buffer must hold at least correlation_count entries"
    );

    let input_len = input_data.len();
    for (i, output) in results.iter_mut().take(count).enumerate() {
        let head = &input_data[..input_len - i];
        let tail = &input_data[i..];
        *output = inner_product_flp(head, tail) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::autocorrelation;

    #[test]
    fn computes_autocorrelation_sequence() {
        let input = [1.0f32, 2.0, 3.0];
        let mut results = [0.0f32; 3];

        autocorrelation(&mut results, &input, 3);

        assert!((results[0] - 14.0).abs() < 1e-6);
        assert!((results[1] - 8.0).abs() < 1e-6);
        assert!((results[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn clamps_correlation_count_to_input_length() {
        let input = [0.5f32, -0.5];
        let mut results = [1.0f32, 1.0, 1.0, 1.0];

        autocorrelation(&mut results, &input, 4);

        assert!((results[0] - 0.5).abs() < 1e-6);
        assert!((results[1] + 0.25).abs() < 1e-6);
        // Remaining entries untouched
        assert_eq!(results[2], 1.0);
        assert_eq!(results[3], 1.0);
    }

    #[test]
    #[should_panic(expected = "results buffer must hold at least correlation_count entries")]
    fn panics_when_results_too_small() {
        let input = [1.0f32, 2.0, 3.0];
        let mut results = [0.0f32; 2];
        autocorrelation(&mut results, &input, 3);
    }
}
