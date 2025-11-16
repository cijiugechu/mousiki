//! Partial port of the `silk/sort.c` helpers used throughout the SILK fixed-point pipeline.
//!
//! The routines implement insertion-sort variants that operate in-place on
//! fixed-point vectors while optionally tracking the original element indices.
//! They are simple utilities that many of the decoder and encoder building
//! blocks depend on, making them a low-dependency candidate for early
//! translation.

/// Sorts the first `k` elements of `a` in increasing order using insertion sort
/// while tracking the indices of the selected elements.
///
/// This mirrors `silk_insertion_sort_increasing` from the reference
/// implementation. Only the first `k` positions are guaranteed to be sorted;
/// the remainder of the slice is left untouched, just like the C routine that
/// only invests work into the requested number of entries.
pub fn insertion_sort_increasing(a: &mut [i32], idx: &mut [i32], k: usize) {
    debug_assert!(!a.is_empty());
    debug_assert!(!idx.is_empty());
    debug_assert!(k > 0);
    debug_assert!(k <= a.len());
    debug_assert!(k <= idx.len());

    for (i, slot) in idx.iter_mut().enumerate().take(k) {
        *slot = i as i32;
    }

    for i in 1..k {
        let value = a[i];
        let mut j = i;
        while j > 0 && value < a[j - 1] {
            a[j] = a[j - 1];
            idx[j] = idx[j - 1];
            j -= 1;
        }
        a[j] = value;
        idx[j] = i as i32;
    }

    for i in k..a.len() {
        let value = a[i];
        if value < a[k - 1] {
            let mut j = k - 1;
            while j > 0 && value < a[j - 1] {
                a[j] = a[j - 1];
                idx[j] = idx[j - 1];
                j -= 1;
            }
            a[j] = value;
            idx[j] = i as i32;
        }
    }
}

/// Sorts the `i16` values in `a` in decreasing order while keeping track of the
/// original indices for the first `k` slots.
///
/// The logic mirrors `silk_insertion_sort_decreasing_int16` from the C code and
/// is only used by the fixed-point configuration. For parity with the original
/// implementation the sort stops after ensuring the first `k` positions are in
/// order.
pub fn insertion_sort_decreasing_int16(a: &mut [i16], idx: &mut [i32], k: usize) {
    debug_assert!(!a.is_empty());
    debug_assert!(!idx.is_empty());
    debug_assert!(k > 0);
    debug_assert!(k <= a.len());
    debug_assert!(k <= idx.len());

    for (i, slot) in idx.iter_mut().enumerate().take(k) {
        *slot = i as i32;
    }

    for i in 1..k {
        let value = a[i];
        let mut j = i;
        while j > 0 && value > a[j - 1] {
            a[j] = a[j - 1];
            idx[j] = idx[j - 1];
            j -= 1;
        }
        a[j] = value;
        idx[j] = i as i32;
    }

    for i in k..a.len() {
        let value = a[i];
        if value > a[k - 1] {
            let mut j = k - 1;
            while j > 0 && value > a[j - 1] {
                a[j] = a[j - 1];
                idx[j] = idx[j - 1];
                j -= 1;
            }
            a[j] = value;
            idx[j] = i as i32;
        }
    }
}

/// Float equivalent of `insertion_sort_decreasing_int16`.
///
/// Mirrors `silk_insertion_sort_decreasing_FLP` from `silk/float/sort_FLP.c`
/// so the FLP analysis helpers can reuse the same top-`k` selection logic
/// without dipping into the C sources.
pub fn insertion_sort_decreasing_f32(a: &mut [f32], idx: &mut [i32], k: usize) {
    debug_assert!(!a.is_empty());
    debug_assert!(!idx.is_empty());
    debug_assert!(k > 0);
    debug_assert!(k <= a.len());
    debug_assert!(k <= idx.len());

    for (i, slot) in idx.iter_mut().enumerate().take(k) {
        *slot = i as i32;
    }

    for i in 1..k {
        let value = a[i];
        let mut j = i;
        while j > 0 && value > a[j - 1] {
            a[j] = a[j - 1];
            idx[j] = idx[j - 1];
            j -= 1;
        }
        a[j] = value;
        idx[j] = i as i32;
    }

    for i in k..a.len() {
        let value = a[i];
        if value > a[k - 1] {
            let mut j = k - 1;
            while j > 0 && value > a[j - 1] {
                a[j] = a[j - 1];
                idx[j] = idx[j - 1];
                j -= 1;
            }
            a[j] = value;
            idx[j] = i as i32;
        }
    }
}

/// Sorts the entire slice `a` in increasing order.
///
/// This mirrors the behaviour of `silk_insertion_sort_increasing_all_values_int16`
/// and is primarily used for helper table preparation steps.
pub fn insertion_sort_increasing_all_values_int16(a: &mut [i16]) {
    if a.is_empty() {
        return;
    }

    for i in 1..a.len() {
        let value = a[i];
        let mut j = i;
        while j > 0 && value < a[j - 1] {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        insertion_sort_decreasing_f32, insertion_sort_decreasing_int16,
        insertion_sort_increasing, insertion_sort_increasing_all_values_int16,
    };

    #[test]
    fn increasing_sort_tracks_indices() {
        let mut values = [10, 3, 5, 7, 2];
        let original = values;
        let mut idx = [0i32; 3];

        insertion_sort_increasing(&mut values, &mut idx, 3);

        assert_eq!(&values[..3], &[2, 3, 5]);
        assert_eq!(idx, [4, 1, 2]);
        assert_eq!(&values[3..], &original[3..]);
    }

    #[test]
    fn increasing_sort_handles_single_element() {
        let mut values = [42, -7, 13];
        let mut idx = [0i32; 1];

        insertion_sort_increasing(&mut values, &mut idx, 1);

        assert_eq!(values[0], -7);
        assert_eq!(idx[0], 1);
    }

    #[test]
    fn decreasing_sort_prefers_larger_values() {
        let mut values = [4i16, -1, 9, 2, 7];
        let mut idx = [0i32; 4];

        insertion_sort_decreasing_int16(&mut values, &mut idx, 4);

        assert_eq!(&values[..4], &[9, 7, 4, 2]);
        assert_eq!(idx, [2, 4, 0, 3]);
    }

    #[test]
    fn increasing_all_values_sorts_entire_slice() {
        let mut values = [5i16, -3, 9, 0, 1];
        insertion_sort_increasing_all_values_int16(&mut values);
        assert_eq!(values, [-3, 0, 1, 5, 9]);
    }

    #[test]
    fn decreasing_sort_flp_matches_reference() {
        let mut values = [0.25f32, -3.0, 1.5, 7.0, -2.0, 9.5, 0.0];
        let mut idx = [0i32; 4];

        insertion_sort_decreasing_f32(&mut values, &mut idx, 4);

        assert_eq!(&values[..4], &[9.5, 7.0, 1.5, 0.25]);
        assert_eq!(idx, [5, 3, 2, 0]);
    }
}
