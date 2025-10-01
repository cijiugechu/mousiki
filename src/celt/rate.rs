//! Helpers from the CELT rate control module.
//!
//! The reference implementation in `celt/rate.c` exposes a number of
//! lightweight helpers that other translation units depend on.  This module
//! begins porting that surface by translating the constant tables and the
//! inline helpers that describe the pseudo-pulse grid.

#![allow(dead_code)]

/// Maximum pseudo-pulse index described in the C headers.
pub(crate) const MAX_PSEUDO: i32 = 40;
/// Base-2 logarithm of [`MAX_PSEUDO`] used by the search helpers.
pub(crate) const LOG_MAX_PSEUDO: i32 = 6;
/// Maximum pulses tracked by the allocation helpers.
pub(crate) const CELT_MAX_PULSES: usize = 128;
/// Maximum number of fine bits stored per band.
pub(crate) const MAX_FINE_BITS: i32 = 8;
/// Fine energy quantiser offset.
pub(crate) const FINE_OFFSET: i32 = 21;
/// Offset applied to the qtheta bit allocation for the single phase search.
pub(crate) const QTHETA_OFFSET: i32 = 4;
/// Offset applied when performing the two-phase qtheta search.
pub(crate) const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// Returns the number of pulses represented by the pseudo-pulse index `i`.
///
/// This mirrors the inline helper from `celt/rate.h`.  The first eight entries
/// map one-to-one, after which the sequence doubles every eight indices while
/// repeating the base pattern modulo eight.
pub(crate) fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Determines if `V(N, K)` fits inside an unsigned 32-bit integer.
///
/// In the reference C implementation this guard is only compiled for custom
/// modes.  It precomputes the limits for both `N` and `K` and applies the same
/// branching logic, allowing the port to reuse the pulse cache generation logic
/// without pulling in the full rate control module.
pub(crate) fn fits_in32(n: i32, k: i32) -> bool {
    const MAX_N: [i16; 15] = [
        32767, 32767, 32767, 1476, 283, 109, 60, 40, 29, 24, 20, 18, 16, 14, 13,
    ];
    const MAX_K: [i16; 15] = [
        32767, 32767, 32767, 32767, 1172, 238, 95, 53, 36, 27, 22, 18, 16, 15, 13,
    ];

    if n >= 14 {
        if k >= 14 {
            false
        } else {
            n <= MAX_N[k as usize] as i32
        }
    } else {
        k <= MAX_K[n as usize] as i32
    }
}

#[cfg(test)]
mod tests {
    use super::{fits_in32, get_pulses};

    #[test]
    fn get_pulses_matches_reference_pattern() {
        let expected = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30,
        ];
        for (i, &value) in expected.iter().enumerate() {
            assert_eq!(get_pulses(i as i32), value);
        }

        // Spot-check the first few entries of the next doubling interval.
        assert_eq!(get_pulses(24), 32);
        assert_eq!(get_pulses(31), 60);
    }

    #[test]
    fn fits_in32_replicates_thresholds() {
        // For n < 14 the max K threshold is provided by MAX_K.
        assert!(fits_in32(13, 15));
        assert!(!fits_in32(13, 16));

        // For n >= 14 the logic flips to checking max N.
        assert!(fits_in32(14, 13));
        assert!(!fits_in32(14, 14));

        // Boundaries around the large "always fits" region.
        assert!(fits_in32(0, 32767));
        assert!(fits_in32(1, 32767));
        assert!(fits_in32(2, 32767));

        // Large values that violate the final MAX_N entry should fail.
        assert!(!fits_in32(15, 13));
    }
}
