//! Helpers from the CELT rate control module.
//!
//! The reference implementation in `celt/rate.c` exposes a number of
//! lightweight helpers that other translation units depend on.  This module
//! begins porting that surface by translating the constant tables and the
//! inline helpers that describe the pseudo-pulse grid.

#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::convert::TryFrom;

use crate::celt::cwrs::get_required_bits;
use crate::celt::entcode::BITRES;
use crate::celt::types::{OpusInt16, OpusInt32, PulseCacheData};

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

/// Recomputes the pulse cache used by custom modes.
///
/// Mirrors `compute_pulse_cache()` from `celt/rate.c`, porting the allocation of
/// the PVQ lookup tables and the per-band bit caps to safe Rust containers. The
/// helper is written to match the original control flow closely so that the
/// results can be compared against the C reference when validating future
/// translations.
#[allow(clippy::too_many_lines)]
pub(crate) fn compute_pulse_cache(
    e_bands: &[OpusInt16],
    log_n: &[OpusInt16],
    lm: usize,
) -> PulseCacheData {
    let nb_ebands = e_bands.len().saturating_sub(1);
    let rows = nb_ebands * (lm + 2);
    let mut index = vec![-1i32; rows];
    let mut entry_n = Vec::new();
    let mut entry_k = Vec::new();
    let mut entry_offset = Vec::new();
    let mut curr = 0i32;

    for i in 0..=(lm + 1) {
        for j in 0..nb_ebands {
            let mut n = i32::from(e_bands[j + 1] - e_bands[j]);
            n = (n << i) >> 1;
            let row = i * nb_ebands + j;
            index[row] = -1;

            for k in 0..=i {
                for n_idx in 0..nb_ebands {
                    if k == i && n_idx >= j {
                        break;
                    }
                    let mut other = i32::from(e_bands[n_idx + 1] - e_bands[n_idx]);
                    other = (other << k) >> 1;
                    if n == other {
                        index[row] = index[k * nb_ebands + n_idx];
                        break;
                    }
                }
                if index[row] != -1 {
                    break;
                }
            }

            if index[row] == -1 && n != 0 {
                let mut k = 0;
                while k < MAX_PSEUDO && fits_in32(n, get_pulses(k + 1)) {
                    k += 1;
                }
                entry_n.push(n);
                entry_k.push(k);
                entry_offset.push(curr);
                index[row] = curr;
                curr += k + 1;
            }
        }
    }

    let mut bits = vec![0u8; curr.max(0) as usize];
    for idx in 0..entry_n.len() {
        let n = entry_n[idx] as usize;
        let k = entry_k[idx] as usize;
        let offset = entry_offset[idx] as usize;
        let mut scratch = vec![0 as OpusInt16; CELT_MAX_PULSES + 1];
        let max_k = get_pulses(entry_k[idx]) as usize;
        get_required_bits(&mut scratch, n, max_k, BITRES as OpusInt32);

        bits[offset] = k as u8;
        for j in 1..=k {
            let pulses = get_pulses(j as i32) as usize;
            let value = scratch[pulses] - 1;
            debug_assert!((0..=u8::MAX as OpusInt16).contains(&value));
            bits[offset + j] = value as u8;
        }
    }

    let mut caps = vec![0u8; (lm + 1) * 2 * nb_ebands];
    let shift = BITRES as usize;
    for i in 0..=lm {
        for c in 1..=2 {
            let c_i32 = c as i32;
            for j in 0..nb_ebands {
                let mut n0 = i32::from(e_bands[j + 1] - e_bands[j]);
                let max_bits = if (n0 << i) == 1 {
                    (c_i32 * (1 + MAX_FINE_BITS)) << shift
                } else {
                    let mut lm0 = 0i32;
                    if n0 > 2 {
                        n0 >>= 1;
                        lm0 -= 1;
                    } else if n0 <= 1 {
                        lm0 = i32::min(i as i32, 1);
                        n0 <<= lm0 as usize;
                    }

                    let row = ((lm0 + 1) as usize) * nb_ebands + j;
                    let cache_offset = index[row];
                    debug_assert!(cache_offset >= 0, "pulse cache entry should exist");
                    let cache_offset = cache_offset as usize;
                    let entry_k = bits[cache_offset] as i32;
                    let base_idx = cache_offset + entry_k as usize;
                    let mut local_bits = i32::from(bits[base_idx]) + 1;
                    let iterations = (i as i32 - lm0).max(0);
                    let mut n = n0;
                    for k_iter in 0..iterations {
                        local_bits <<= 1;
                        let offset = ((i32::from(log_n[j]) + ((lm0 + k_iter) << shift)) >> 1)
                            - QTHETA_OFFSET;
                        let two_n_minus_one = 2 * n - 1;
                        let num = 459 * (two_n_minus_one * offset + local_bits);
                        let den = (two_n_minus_one << 9) - 459;
                        let mut qb = ((num + (den >> 1)) / den).min(57);
                        debug_assert!(qb >= 0);
                        if qb < 0 {
                            qb = 0;
                        }
                        local_bits += qb;
                        let offset_guard = offset.max(0);
                        local_bits = local_bits.max(c_i32 * (offset_guard + (4 << shift)));
                        n <<= 1;
                    }

                    let ndof = (c_i32 * n0 - (c_i32 - 1)).max(1);
                    local_bits = local_bits.min(c_i32 * n0 << shift);
                    let pulses = get_pulses(entry_k) + 2 * (ndof - 1);
                    let floor = c_i32 * ((pulses / (2 * ndof)) << shift);
                    local_bits = local_bits.max(floor);
                    if (n0 << i) == 2 {
                        let extra = ((c_i32 * (1 + 16)) >> 1) << shift;
                        local_bits = local_bits.max(extra);
                    }
                    if c == 2 && (n0 << i) >= 2 {
                        let extra = ((2 * (1 + 24)) >> 1) << shift;
                        local_bits = local_bits.max(extra);
                    }

                    local_bits
                };

                let cap_idx = i * 2 * nb_ebands + (c - 1) * nb_ebands + j;
                if !caps.is_empty() {
                    caps[cap_idx] = ((max_bits >> shift).min(255)) as u8;
                }
            }
        }
    }

    let index = index
        .into_iter()
        .map(|value| i16::try_from(value).expect("pulse cache index exceeds 16-bit range"))
        .collect();

    PulseCacheData::new(index, bits, caps)
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;

    use super::{compute_pulse_cache, fits_in32, get_pulses};

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

    #[test]
    fn compute_pulse_cache_assigns_shared_offsets() {
        let e_bands = [0i16, 2, 6];
        let log_n = [6i16, 7];
        let lm = 1usize;
        let cache = compute_pulse_cache(&e_bands, &log_n, lm);

        let nb_ebands = e_bands.len() - 1;
        assert_eq!(cache.size, cache.bits.len());
        assert_eq!(cache.index.len(), nb_ebands * (lm + 2));
        assert_eq!(cache.caps.len(), (lm + 1) * 2 * nb_ebands);

        let mut seen = BTreeMap::new();
        for i in 0..=(lm + 1) {
            for j in 0..nb_ebands {
                let n = i32::from(e_bands[j + 1] - e_bands[j]);
                let n = (n << i) >> 1;
                if n == 0 {
                    continue;
                }
                let offset = cache.index[i * nb_ebands + j];
                if let Some(&expected) = seen.get(&n) {
                    assert_eq!(offset, expected);
                } else {
                    seen.insert(n, offset);
                    let offset = offset as usize;
                    let k = cache.bits[offset] as usize;
                    assert!(offset + k < cache.bits.len());
                }
            }
        }
    }
}
