#![allow(dead_code)]

//! Pulse vector combinatorics helpers from the reference CELT implementation.
//!
//! The routines in this module have minimal dependencies on the rest of the
//! encoder/decoder pipeline and can therefore be ported in isolation.  They
//! primarily operate on integer combinatorics used by the codeword
//! enumeration logic in `cwrs.c`.

use crate::celt::entcode::ec_ilog;
use crate::celt::types::{OpusInt32, OpusUint32};

/// Returns a conservatively large estimate of `log2(val)` with `frac` fractional bits.
///
/// Mirrors `log2_frac()` from `celt/cwrs.c`. The routine assumes `val > 0` and that
/// `frac` is non-negative. The result is guaranteed to be greater than or equal to
/// the exact value, matching the behaviour of the C implementation which the
/// bit-allocation heuristics rely on for safety margins.
#[must_use]
pub(crate) fn log2_frac(mut val: OpusUint32, frac: OpusInt32) -> OpusInt32 {
    debug_assert!(val > 0);
    debug_assert!(frac >= 0);

    let l = ec_ilog(val);
    if val & (val - 1) != 0 {
        if l > 16 {
            val = ((val - 1) >> ((l - 16) as u32)) + 1;
        } else {
            val <<= (16 - l) as u32;
        }

        let mut acc = (l - 1) << frac;
        let mut current_frac = frac;

        loop {
            let b = (val >> 16) as OpusInt32;
            let shift = current_frac as u32;
            debug_assert!(current_frac <= 30);
            acc += b << shift;
            val = (val + b as OpusUint32) >> (b as u32);
            val = ((val * val) + 0x7FFF) >> 15;

            if current_frac <= 0 {
                break;
            }
            current_frac -= 1;
        }

        acc + OpusInt32::from(val > 0x8000)
    } else {
        (l - 1) << frac
    }
}

/// Advances a combinatorial row following the `unext()` recurrence from `celt/cwrs.c`.
///
/// The slice mirrors the C pointer passed to `unext`, which requires at least two
/// elements. The `ui0` parameter provides the base case for the new row/column and
/// matches the final argument of the C helper.
pub(crate) fn unext(ui: &mut [OpusUint32], mut ui0: OpusUint32) {
    debug_assert!(ui.len() >= 2);

    for j in 1..ui.len() {
        let ui1 = ui[j]
            .checked_add(ui[j - 1])
            .and_then(|acc| acc.checked_add(ui0))
            .expect("U(n, k) overflowed 32 bits");
        ui[j - 1] = ui0;
        ui0 = ui1;
    }

    if let Some(last) = ui.last_mut() {
        *last = ui0;
    }
}

/// Rewinds a combinatorial row following the `uprev()` recurrence from `celt/cwrs.c`.
///
/// The slice mirrors the pointer passed to the C helper and must contain at least two
/// elements. The `ui0` value supplies the base case for the reconstructed row.
pub(crate) fn uprev(ui: &mut [OpusUint32], mut ui0: OpusUint32) {
    debug_assert!(ui.len() >= 2);

    for j in 1..ui.len() {
        let ui1 = ui[j]
            .checked_sub(ui[j - 1])
            .and_then(|acc| acc.checked_sub(ui0))
            .expect("U(n, k) underflowed 32 bits");
        ui[j - 1] = ui0;
        ui0 = ui1;
    }

    if let Some(last) = ui.last_mut() {
        *last = ui0;
    }
}

/// Computes the `U(n, k)` row used by the PVQ codeword enumeration logic.
///
/// Mirrors `ncwrs_urow()` from the small-footprint path in `celt/cwrs.c`. The
/// provided buffer must have space for `k + 2` entries, mirroring the layout of
/// the C implementation where indices `0..=k+1` are populated. The return value is
/// `V(n, k) = U(n, k) + U(n, k + 1)`.
#[must_use]
pub(crate) fn ncwrs_urow(n: usize, k: usize, u: &mut [OpusUint32]) -> OpusUint32 {
    debug_assert!(n >= 2);
    debug_assert!(k > 0);

    let len = k + 2;
    debug_assert!(u.len() >= len);

    u[0] = 0;
    u[1] = 1;
    for (idx, slot) in u.iter_mut().enumerate().skip(2).take(len - 2) {
        *slot = ((idx as OpusUint32) << 1) - 1;
    }

    if n > 2 {
        for _ in 2..n {
            let slice = &mut u[1..len];
            unext(slice, 1);
        }
    }

    u[k].checked_add(u[k + 1])
        .expect("V(n, k) overflowed 32 bits")
}

#[cfg(test)]
mod tests {
    use super::{log2_frac, ncwrs_urow, unext, uprev};
    use crate::celt::types::OpusUint32;
    use alloc::vec;
    use alloc::vec::Vec;

    fn reference_log2_frac(val: u32, frac: i32) -> i32 {
        let scale = 1 << frac;
        ((val as f64).log2() * f64::from(scale)).ceil() as i32
    }

    #[test]
    fn matches_reference_estimate_for_small_values() {
        for val in 1..=256u32 {
            for frac in 0..=6 {
                let exact = reference_log2_frac(val, frac);
                let estimate = log2_frac(val, frac);
                assert!(
                    estimate >= exact,
                    "estimate {} < exact {} for val={}, frac={}",
                    estimate,
                    exact,
                    val,
                    frac
                );
                assert!(
                    estimate - exact <= 1,
                    "estimate {} too far from exact {} for val={}, frac={}",
                    estimate,
                    exact,
                    val,
                    frac
                );
            }
        }
    }

    #[test]
    fn matches_reference_estimate_for_large_values() {
        let samples = [
            0x0001_FFEE,
            0x00FF_FFFF,
            0x0F00_0001,
            0x8000_0000,
            0xFFFF_FFFE,
        ];
        for &val in &samples {
            for frac in 0..=6 {
                let exact = reference_log2_frac(val, frac);
                let estimate = log2_frac(val, frac);
                assert!(estimate >= exact);
                assert!(estimate - exact <= 2);
            }
        }
    }

    fn reference_u_table(n_max: usize, k_max: usize) -> Vec<Vec<u64>> {
        let mut table = vec![vec![0u64; k_max + 2]; n_max + 1];
        table[0][0] = 1;

        for n in 0..=n_max {
            for k in 0..=k_max + 1 {
                if n == 0 && k == 0 {
                    continue;
                }
                if n == 0 {
                    table[n][k] = 0;
                } else if k == 0 {
                    table[n][k] = 0;
                } else {
                    table[n][k] = table[n - 1][k] + table[n][k - 1] + table[n - 1][k - 1];
                }
            }
        }

        table
    }

    #[test]
    fn ncwrs_urow_matches_reference_values() {
        let n_max = 5;
        let k_max = 5;
        let reference = reference_u_table(n_max, k_max);

        for n in 2..=n_max {
            for k in 1..=k_max {
                let mut u = vec![0u32; k + 2];
                let v = ncwrs_urow(n, k, &mut u);
                for idx in 0..=k + 1 {
                    assert_eq!(u[idx] as u64, reference[n][idx], "U({n}, {idx}) mismatch");
                }
                let expected_v = reference[n][k] + reference[n][k + 1];
                assert_eq!(v as u64, expected_v, "V({n}, {k}) mismatch");
            }
        }
    }

    #[test]
    fn unext_and_uprev_are_inverses_for_small_rows() {
        let n = 4usize;
        let k = 3usize;
        let mut row = vec![0u32; k + 2];
        let mut expected = row.clone();
        // Fill `row` with the values produced by `ncwrs_urow` and keep the
        // initial configuration in `expected` for later comparison.
        let _ = ncwrs_urow(n, k, &mut row);
        expected.copy_from_slice(&row);

        let slice_len = k + 1;
        let (head, tail) = expected.split_at_mut(1);
        let mut working: Vec<OpusUint32> = tail[..slice_len].to_vec();
        unext(&mut working, 1);
        uprev(&mut working, 1);
        head[0] = 0;
        expected[1..1 + slice_len].copy_from_slice(&working);

        assert_eq!(row, expected);
    }
}
