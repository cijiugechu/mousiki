//! Fixed-point approximation of the base-2 exponential used by the SILK decoder helpers.
//!
//! This module mirrors `silk_log2lin` from `silk/log2lin.c` in the reference C implementation.
//! The function acts as a cheap inverse of [`lin2log`](crate::silk::lin2log::lin2log), mapping a
//! Q7 fixed-point logarithmic value back to the linear domain without resorting to floating-point
//! math.

/// Approximation of `2^(x / 128)` for 32-bit fixed-point arguments in the Q7 domain.
///
/// Negative inputs clamp to zero and sufficiently large inputs saturate at `i32::MAX`, matching
/// the behaviour of the C routine.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    if in_log_q7 >= 3967 {
        return i32::MAX;
    }

    let mut out = 1i32 << ((in_log_q7 >> 7) as u32);
    let frac_q7 = in_log_q7 & 0x7f;
    let product = frac_q7 * (128 - frac_q7);
    // Magic constant derived from Taylor series fitting of 2^(frac / 128)
    let correction = frac_q7 + ((i64::from(product) * -174) >> 16) as i32;

    // Split to multiply before shifting for inputs below 2048 (2^16 output),
    // shifting first afterwards to avoid overflow while retaining precision
    if in_log_q7 < 2048 {
        out += ((i64::from(out) * i64::from(correction)) >> 7) as i32;
    } else {
        out += (i64::from(out >> 7) * i64::from(correction)) as i32;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::log2lin;

    #[test]
    fn matches_reference_values() {
        let cases = [
            (-128, 0),
            (-1, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (127, 1),
            (128, 2),
            (203, 3),
            (256, 4),
            (296, 4),
            (384, 8),
            (512, 16),
            (640, 32),
            (896, 128),
            (1024, 256),
            (1919, 32_512),
            (2048, 65_536),
            (3966, 2_122_317_824),
            (3967, 2_147_483_647),
        ];

        for (input, expected) in cases {
            assert_eq!(log2lin(input), expected, "log2lin({input})");
        }
    }

    #[test]
    fn matches_reference_for_2264() {
        assert_eq!(log2lin(2264), 210_944);
    }
}
