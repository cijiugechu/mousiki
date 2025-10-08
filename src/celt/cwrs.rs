#![allow(dead_code)]

//! Pulse vector combinatorics helpers from the reference CELT implementation.
//!
//! The routines in this module have minimal dependencies on the rest of the
//! encoder/decoder pipeline and can therefore be ported in isolation.  They
//! primarily operate on integer combinatorics used by the codeword
//! enumeration logic in `cwrs.c`.

use alloc::vec;

use crate::celt::entcode::ec_ilog;
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::celt::types::{OpusInt16, OpusInt32, OpusUint32, OpusVal32};

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

fn icwrs1(value: OpusInt32) -> (OpusUint32, usize) {
    let pulses = value.unsigned_abs() as usize;
    let index = OpusUint32::from(value < 0);
    (index, pulses)
}

fn icwrs(
    y: &[OpusInt32],
    n: usize,
    total_pulses: usize,
    u: &mut [OpusUint32],
) -> (OpusUint32, OpusUint32) {
    debug_assert!(n >= 2);
    debug_assert!(total_pulses > 0);
    debug_assert!(y.len() >= n);
    debug_assert!(u.len() >= total_pulses + 2);

    u[0] = 0;
    for (idx, slot) in u.iter_mut().enumerate().skip(1).take(total_pulses + 1) {
        *slot = ((idx as OpusUint32) << 1) - 1;
    }

    let last = y[n - 1];
    let (mut index, pulses_used) = icwrs1(last);

    let mut j = n - 2;
    index = index
        .checked_add(u[pulses_used])
        .expect("icwrs index overflowed 32 bits");

    let mut pulses_acc = pulses_used + y[j].unsigned_abs() as usize;
    if y[j] < 0 {
        index = index
            .checked_add(u[pulses_acc + 1])
            .expect("icwrs index overflowed 32 bits");
    }

    while j > 0 {
        unext(&mut u[..total_pulses + 2], 0);
        j -= 1;
        index = index
            .checked_add(u[pulses_acc])
            .expect("icwrs index overflowed 32 bits");
        pulses_acc += y[j].unsigned_abs() as usize;
        if y[j] < 0 {
            index = index
                .checked_add(u[pulses_acc + 1])
                .expect("icwrs index overflowed 32 bits");
        }
    }

    let nc = u[pulses_acc]
        .checked_add(u[pulses_acc + 1])
        .expect("V(n, k) overflowed 32 bits");
    (index, nc)
}

fn cwrsi(
    n: usize,
    mut k: usize,
    mut index: OpusUint32,
    y: &mut [OpusInt32],
    u: &mut [OpusUint32],
) -> OpusVal32 {
    debug_assert!(n > 0);
    debug_assert!(y.len() >= n);
    debug_assert!(u.len() >= k + 2);

    let mut energy: OpusVal32 = 0.0;

    for value_ref in y.iter_mut().take(n) {
        let sign_threshold = u[k + 1];
        let sign = if index >= sign_threshold {
            index -= sign_threshold;
            -1
        } else {
            0
        };

        let mut pulses_in_dim = k;
        let mut entry = u[k];
        while entry > index {
            k -= 1;
            entry = u[k];
        }

        index -= entry;
        pulses_in_dim -= k;

        let value = if sign == 0 {
            pulses_in_dim as OpusInt32
        } else {
            -(pulses_in_dim as OpusInt32)
        };

        *value_ref = value;
        let val = value as OpusVal32;
        energy += val * val;

        uprev(&mut u[..k + 2], 0);
    }

    energy
}

pub(crate) fn encode_pulses(y: &[OpusInt32], n: usize, k: usize, enc: &mut EcEnc<'_>) {
    debug_assert!(k > 0);
    debug_assert!(n >= 2);
    debug_assert!(y.len() >= n);

    let mut workspace = vec![0u32; k + 2];
    let (index, total) = icwrs(y, n, k, &mut workspace);
    enc.enc_uint(index, total);
}

pub(crate) fn decode_pulses(
    y: &mut [OpusInt32],
    n: usize,
    k: usize,
    dec: &mut EcDec<'_>,
) -> OpusVal32 {
    debug_assert!(k > 0);
    debug_assert!(n >= 2);
    debug_assert!(y.len() >= n);

    let mut workspace = vec![0u32; k + 2];
    let total = ncwrs_urow(n, k, &mut workspace);
    let index = dec.dec_uint(total);
    cwrsi(n, k, index, y, &mut workspace)
}

/// Computes the number of fractional bits required to represent each pulse count.
///
/// Mirrors `get_required_bits()` from `celt/cwrs.c` in the reference
/// implementation. The output slice must have capacity for `max_k + 1` entries,
/// matching the C routine which fills indices `0..=max_k`. The first element is
/// always set to zero. For `n == 1` the bit requirement is constant across all
/// pulse counts; otherwise the helper evaluates the `U(n, k)` recurrence and
/// applies [`log2_frac`] to the resulting `V(n, k)` table entries.
pub(crate) fn get_required_bits(bits: &mut [OpusInt16], n: usize, max_k: usize, frac: OpusInt32) {
    debug_assert!(max_k > 0);
    debug_assert!(bits.len() > max_k);
    debug_assert!(frac >= 0);

    bits[0] = 0;
    if n == 1 {
        let value = 1i32 << frac;
        debug_assert!(value <= i32::from(OpusInt16::MAX));
        for slot in bits.iter_mut().take(max_k + 1).skip(1) {
            *slot = value as OpusInt16;
        }
        return;
    }

    let mut u = vec![0u32; max_k + 2];
    let _ = ncwrs_urow(n, max_k, &mut u);

    for (k, slot) in bits.iter_mut().enumerate().take(max_k + 1).skip(1) {
        let total = u[k]
            .checked_add(u[k + 1])
            .expect("V(n, k) exceeded 32 bits");
        let required = log2_frac(total, frac);
        debug_assert!(required >= 0);
        debug_assert!(required <= OpusInt32::from(OpusInt16::MAX));
        *slot = required as OpusInt16;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decode_pulses, encode_pulses, get_required_bits, log2_frac, ncwrs_urow, unext, uprev,
    };
    use crate::celt::entdec::EcDec;
    use crate::celt::entenc::EcEnc;
    use crate::celt::types::{OpusInt16, OpusInt32, OpusUint32, OpusVal32};
    use alloc::vec;
    use alloc::vec::Vec;

    fn reference_log2_frac(val: u32, frac: i32) -> i32 {
        let scale = 1 << frac;
        (f64::from(val).log2() * f64::from(scale)).ceil() as i32
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

        if n_max >= 1 {
            for k in 1..=k_max + 1 {
                table[1][k] = 1;
            }
        }

        for n in 2..=n_max {
            for k in 1..=k_max + 1 {
                table[n][k] = table[n - 1][k] + table[n][k - 1] + table[n - 1][k - 1];
            }
        }

        table
    }

    #[test]
    fn ncwrs_urow_matches_reference_values() {
        let n_max = 5;
        let k_max = 5;
        let reference = reference_u_table(n_max, k_max);

        for (n, _) in reference.iter().enumerate().take(n_max + 1).skip(2) {
            for k in 1..=k_max {
                let mut u = vec![0u32; k + 2];
                let v = ncwrs_urow(n, k, &mut u);
                for (idx, _) in u.iter().enumerate().take(k + 1 + 1) {
                    assert_eq!(u64::from(u[idx]), reference[n][idx], "U({n}, {idx}) mismatch");
                }
                let expected_v = reference[n][k] + reference[n][k + 1];
                assert_eq!(u64::from(v), expected_v, "V({n}, {k}) mismatch");
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

    fn enumerate_pulses(n: usize, k: usize) -> Vec<Vec<OpusInt32>> {
        let mut current = vec![0; n];
        let mut out = Vec::new();

        fn search(
            idx: usize,
            n: usize,
            k: usize,
            current: &mut [OpusInt32],
            out: &mut Vec<Vec<OpusInt32>>,
        ) {
            if idx == n {
                let total: usize = current.iter().map(|&v| v.unsigned_abs() as usize).sum();
                if total == k {
                    out.push(current.to_vec());
                }
                return;
            }

            let limit = k as OpusInt32;
            for value in -limit..=limit {
                current[idx] = value;
                search(idx + 1, n, k, current, out);
            }
        }

        search(0, n, k, &mut current, &mut out);
        out
    }

    #[test]
    fn encode_and_decode_round_trip_small_vectors() {
        let configs = &[(2usize, 1usize), (2, 2), (3, 2)];

        for &(n, k) in configs {
            for pulses in enumerate_pulses(n, k) {
                if pulses.iter().all(|&v| v == 0) {
                    continue;
                }

                let mut buffer = vec![0u8; 128];
                let mut encoder = EcEnc::new(buffer.as_mut_slice());
                encode_pulses(&pulses, n, k, &mut encoder);
                encoder.enc_done();

                let mut decode_buf = buffer.clone();
                let mut decoder = EcDec::new(decode_buf.as_mut_slice());
                let mut decoded = vec![0; n];
                let energy = decode_pulses(&mut decoded, n, k, &mut decoder);

                assert_eq!(decoded, pulses);

                let expected: OpusVal32 = pulses
                    .iter()
                    .map(|&v| {
                        let val = v as OpusVal32;
                        val * val
                    })
                    .sum();
                assert!((energy - expected).abs() < 1e-6);
            }
        }
    }

    fn reference_required_bits(n: usize, max_k: usize, frac: OpusInt32) -> Vec<OpusInt16> {
        let mut bits = vec![0i16; max_k + 1];
        if n == 1 {
            let value = 1i32 << frac;
            for slot in bits.iter_mut().skip(1) {
                *slot = value as i16;
            }
            return bits;
        }

        let table = reference_u_table(n, max_k);
        for (k, slot) in bits.iter_mut().enumerate().take(max_k + 1).skip(1) {
            let total = table[n][k] + table[n][k + 1];
            let required = reference_log2_frac(total as u32, frac);
            *slot = required as i16;
        }

        bits
    }

    #[test]
    fn get_required_bits_matches_reference() {
        let max_n = 5;
        let max_k = 5;

        for n in 1..=max_n {
            for frac in 0..=6 {
                let mut bits = vec![0i16; max_k + 1];
                get_required_bits(&mut bits, n, max_k, frac);
                let expected = reference_required_bits(n, max_k, frac);
                assert_eq!(bits, expected, "Mismatch for n={n}, frac={frac}");
            }
        }
    }
}
