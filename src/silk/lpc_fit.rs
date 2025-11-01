//! Port of the fixed-point `silk_LPC_fit` helper from the reference SILK
//! implementation (`silk/LPC_fit.c`). The routine converts a vector of LPC
//! coefficients stored in a high-precision Q-domain into 16-bit Q12 values while
//! ensuring the conversion cannot overflow. When the coefficients exceed the
//! representable range it gradually applies a chirp (bandwidth expansion) until
//! the values fall inside the 16-bit bounds or, as a last resort, clips them.

use core::cmp::{Ordering, min};

use super::bwexpander_32::bwexpander_32;

const MAX_ABS_CLIP: i32 = 163_838;
const MAX_ITERATIONS: usize = 10;
const FIX_CONST_0_999_Q16: i32 = 65_470; // round(0.999 * 2^16)

/// Convert LPC coefficients from `a_qin` (Q`qin`) into Q`qout` 16-bit values in
/// `a_qout`. The routine mutates `a_qin` when bandwidth expansion or clipping is
/// required so the caller can observe the adjusted coefficients.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn lpc_fit(a_qout: &mut [i16], a_qin: &mut [i32], qout: i32, qin: i32) {
    assert_eq!(a_qout.len(), a_qin.len(), "input/output orders must match");
    let order = a_qout.len();
    if order == 0 {
        return;
    }

    let mut clipped = true;

    for _ in 0..MAX_ITERATIONS {
        let (mut maxabs, mut idx) = (0i32, 0usize);
        for (k, &value) in a_qin.iter().enumerate() {
            let absval = value.abs();
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }

        let mut maxabs_qout = rshift_round(maxabs, qin - qout);
        if maxabs_qout <= i32::from(i16::MAX) {
            clipped = false;
            break;
        }

        maxabs_qout = min(maxabs_qout, MAX_ABS_CLIP);
        let numerator = (maxabs_qout - i32::from(i16::MAX)) << 14;
        let denom = ((maxabs_qout * (idx as i32 + 1)) >> 2).max(1);
        let chirp_q16 = FIX_CONST_0_999_Q16 - numerator / denom;
        bwexpander_32(a_qin, chirp_q16);
    }

    if clipped {
        for (out, value) in a_qout.iter_mut().zip(a_qin.iter_mut()) {
            let scaled = rshift_round(*value, qin - qout);
            let saturated = sat16(scaled);
            *out = saturated;
            *value = rescale_back_to_qin(saturated, qin - qout);
        }
    } else {
        for (out, &value) in a_qout.iter_mut().zip(a_qin.iter()) {
            *out = rshift_round(value, qin - qout) as i16;
        }
    }
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    match shift.cmp(&0) {
        Ordering::Equal => value,
        Ordering::Greater => {
            if shift == 1 {
                (value >> 1) + (value & 1)
            } else {
                ((value >> (shift - 1)) + 1) >> 1
            }
        }
        Ordering::Less => value.wrapping_shl((-shift) as u32),
    }
}

fn rescale_back_to_qin(value: i16, shift: i32) -> i32 {
    if shift >= 0 {
        i32::from(value).wrapping_shl(shift as u32)
    } else {
        rshift_round(i32::from(value), -shift)
    }
}

fn sat16(value: i32) -> i16 {
    if value > i32::from(i16::MAX) {
        i16::MAX
    } else if value < i32::from(i16::MIN) {
        i16::MIN
    } else {
        value as i16
    }
}

#[cfg(test)]
mod tests {
    use super::lpc_fit;

    #[test]
    fn converts_coefficients_that_fit_in_range() {
        let mut input = [4000, -8000, 6000, -2000];
        let mut output = [0i16; 4];

        lpc_fit(&mut output, &mut input, 12, 13);

        assert_eq!(output, [2000, -4000, 3000, -1000]);
        assert_eq!(input, [4000, -8000, 6000, -2000]);
    }

    #[test]
    fn applies_bandwidth_expansion_when_values_exceed_limit() {
        let mut input = [1_500_000, -1_200_000, 800_000, -600_000];
        let mut output = [0i16; 4];

        lpc_fit(&mut output, &mut input, 12, 17);

        assert!(output.iter().all(|&coeff| coeff.abs() <= i16::MAX));
        assert!(input.iter().any(|&value| value != 0));
    }
}
