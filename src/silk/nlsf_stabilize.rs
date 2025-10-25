//! Port of the SILK NLSF stabiliser from `silk/NLSF_stabilize.c`.
//!
//! The routine enforces the minimum delta constraints on an in-place
//! normalised line spectral frequency (NLSF) vector, ensuring that it stays
//! ordered, respects the guard bands, and remains within the representable
//! Q15 domain. The algorithm mirrors the reference implementation's
//! iterative adjustment strategy and fall-back insertion sort.

use super::sort::insertion_sort_increasing_all_values_int16;

const MAX_LOOPS: usize = 20;

/// Stabilises an NLSF vector in-place.
///
/// * `nlsf_q15` - Mutable slice containing the NLSF values in Q15 format.
/// * `n_delta_min_q15` - Slice of length `nlsf_q15.len() + 1` that describes the
///   minimum deltas to the boundaries and neighbouring coefficients.
pub fn nlsf_stabilize(nlsf_q15: &mut [i16], n_delta_min_q15: &[i16]) {
    let l = nlsf_q15.len();
    if l == 0 {
        return;
    }

    debug_assert_eq!(n_delta_min_q15.len(), l + 1);
    debug_assert!(*n_delta_min_q15.last().unwrap() >= 1);

    for _ in 0..MAX_LOOPS {
        let mut min_diff_q15 = i32::from(nlsf_q15[0]) - i32::from(n_delta_min_q15[0]);
        let mut index = 0usize;

        for i in 1..l {
            let diff_q15 = i32::from(nlsf_q15[i])
                - (i32::from(nlsf_q15[i - 1]) + i32::from(n_delta_min_q15[i]));
            if diff_q15 < min_diff_q15 {
                min_diff_q15 = diff_q15;
                index = i;
            }
        }

        let last_diff_q15 =
            (1 << 15) - (i32::from(nlsf_q15[l - 1]) + i32::from(n_delta_min_q15[l]));
        if last_diff_q15 < min_diff_q15 {
            min_diff_q15 = last_diff_q15;
            index = l;
        }

        if min_diff_q15 >= 0 {
            return;
        }

        match index {
            0 => {
                nlsf_q15[0] = n_delta_min_q15[0];
            }
            i if i == l => {
                let upper = (1 << 15) - i32::from(n_delta_min_q15[l]);
                nlsf_q15[l - 1] = clamp_to_i16(upper);
            }
            i => {
                let mut min_center_q15 = 0i32;
                for &delta in &n_delta_min_q15[..i] {
                    min_center_q15 += i32::from(delta);
                }
                min_center_q15 += i32::from(n_delta_min_q15[i]) >> 1;

                let mut max_center_q15 = 1 << 15;
                for &delta in &n_delta_min_q15[i + 1..=l] {
                    max_center_q15 -= i32::from(delta);
                }
                max_center_q15 -= i32::from(n_delta_min_q15[i]) >> 1;

                let sum = i32::from(nlsf_q15[i - 1]) + i32::from(nlsf_q15[i]);
                let mut center_freq_q15 = (sum + 1) >> 1;
                center_freq_q15 = center_freq_q15.clamp(min_center_q15, max_center_q15);

                let half_delta = i32::from(n_delta_min_q15[i]) >> 1;
                let lower = center_freq_q15 - half_delta;
                nlsf_q15[i - 1] = clamp_to_i16(lower);
                let upper = i32::from(nlsf_q15[i - 1]) + i32::from(n_delta_min_q15[i]);
                nlsf_q15[i] = clamp_to_i16(upper);
            }
        }
    }

    insertion_sort_increasing_all_values_int16(nlsf_q15);

    nlsf_q15[0] = nlsf_q15[0].max(n_delta_min_q15[0]);

    for i in 1..l {
        let min_value = nlsf_q15[i - 1].saturating_add(n_delta_min_q15[i]);
        nlsf_q15[i] = nlsf_q15[i].max(min_value);
    }

    let upper_limit = (1 << 15) - i32::from(n_delta_min_q15[l]);
    nlsf_q15[l - 1] = nlsf_q15[l - 1].min(clamp_to_i16(upper_limit));

    for i in (0..l - 1).rev() {
        let max_value = i32::from(nlsf_q15[i + 1]) - i32::from(n_delta_min_q15[i + 1]);
        nlsf_q15[i] = nlsf_q15[i].min(clamp_to_i16(max_value));
    }
}

fn clamp_to_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use super::nlsf_stabilize;

    #[test]
    fn enforces_minimum_deltas() {
        let mut nlsf = [200, 205, 210, 215];
        let deltas = [10, 20, 20, 20, 10];

        nlsf_stabilize(&mut nlsf, &deltas);

        for (pair, &delta) in nlsf.windows(2).zip(deltas.iter().skip(1)) {
            let diff = i32::from(pair[1]) - i32::from(pair[0]);
            assert!(diff >= i32::from(delta));
        }
        assert!(i32::from(nlsf[0]) >= i32::from(deltas[0]));
        let upper_guard = (1 << 15) - i32::from(deltas[deltas.len() - 1]);
        assert!(i32::from(nlsf[nlsf.len() - 1]) <= upper_guard);
    }

    #[test]
    fn fallback_sort_produces_sorted_output() {
        let mut nlsf = [30000, -2000, 15000, 16000, 17000];
        let deltas = [5, 50, 50, 50, 50, 5];

        nlsf_stabilize(&mut nlsf, &deltas);

        for (pair, &delta) in nlsf.windows(2).zip(deltas.iter().skip(1)) {
            let diff = i32::from(pair[1]) - i32::from(pair[0]);
            assert!(diff >= i32::from(delta));
        }
        assert!(i32::from(nlsf[0]) >= i32::from(deltas[0]));
        let upper_guard = (1 << 15) - i32::from(deltas[deltas.len() - 1]);
        assert!(i32::from(nlsf[nlsf.len() - 1]) <= upper_guard);
    }
}
