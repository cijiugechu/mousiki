//! Floating-point energy helper from `silk/float/energy_FLP.c`.
//!
//! The routine is dependency-light and mirrors the 4× unrolled loops from the
//! reference implementation so future FLP analysis helpers can reuse the exact
//! same accumulation order when computing squared magnitudes.

/// Returns the sum of squares for the provided floating-point samples.
///
/// Mirrors `silk_energy_FLP`, returning an `f64` to match the original double
/// accumulator.
pub fn energy(data: &[f32]) -> f64 {
    let mut result = 0.0f64;
    let mut i = 0;
    let data_size4 = data.len() & !3;

    while i < data_size4 {
        let sample0 = f64::from(data[i]);
        let sample1 = f64::from(data[i + 1]);
        let sample2 = f64::from(data[i + 2]);
        let sample3 = f64::from(data[i + 3]);

        result += sample0 * sample0 + sample1 * sample1 + sample2 * sample2 + sample3 * sample3;
        i += 4;
    }

    while i < data.len() {
        let sample = f64::from(data[i]);
        result += sample * sample;
        i += 1;
    }

    debug_assert!(result >= 0.0);
    result
}

#[cfg(test)]
mod tests {
    use super::energy;

    #[test]
    fn accumulates_sum_of_squares() {
        let data = [1.0f32, -2.0, 3.0, -4.0];
        let result = energy(&data);
        assert!((result - 30.0).abs() < 1e-6);
    }

    #[test]
    fn handles_non_multiple_of_four_lengths() {
        let data = [0.5f32, -1.5, 2.25];
        let result = energy(&data);
        let expected = 0.25 + 2.25 + 5.0625;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn empty_slice_returns_zero() {
        let data: [f32; 0] = [];
        assert_eq!(energy(&data), 0.0);
    }
}
