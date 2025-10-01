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

#[cfg(test)]
mod tests {
    use super::log2_frac;

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
        let samples = [0x0001_FFEE, 0x00FF_FFFF, 0x0F00_0001, 0x8000_0000, 0xFFFF_FFFE];
        for &val in &samples {
            for frac in 0..=6 {
                let exact = reference_log2_frac(val, frac);
                let estimate = log2_frac(val, frac);
                assert!(estimate >= exact);
                assert!(estimate - exact <= 2);
            }
        }
    }
}
