//! Fixed-point approximation of `128 * log2(x)` used throughout the SILK signal-processing helpers.
//!
//! This module mirrors `silk_lin2log` from `silk/lin2log.c` in the reference C implementation. The
//! routine exposes a cheap logarithm that scales its input by 128, delivering results in the Q7
//! domain without resorting to floating-point math. The translation sticks closely to the C code,
//! including the implicit behaviour for zero and negative inputs.

/// Approximation of `128 * log2(x)` for 32-bit integers.
///
/// The return value lives in the Q7 domain – i.e. it is the base-2 logarithm multiplied by 128.
/// The behaviour matches the C implementation: non-positive inputs yield a result of `-128`, and
/// negative values are accepted for completeness even though the original algorithm only relies on
/// non-negative magnitudes.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub fn lin2log(in_lin: i32) -> i32 {
    let in_lin_u32 = in_lin as u32;
    let lz = in_lin_u32.leading_zeros() as i32;

    let rot = 24 - lz;
    let rotated = if rot >= 0 {
        in_lin_u32.rotate_right(rot as u32)
    } else {
        in_lin_u32.rotate_left((-rot) as u32)
    } as i32;
    let frac_q7 = rotated & 0x7f;

    let product = frac_q7 * (128 - frac_q7);
    let correction = frac_q7 + ((i64::from(product) * 179) >> 16) as i32;

    ((31 - lz) * 128) + correction
}

#[cfg(test)]
mod tests {
    use super::lin2log;

    #[test]
    fn matches_reference_values() {
        let cases = [
            (0, -128),
            (1, 0),
            (2, 128),
            (3, 203),
            (4, 256),
            (5, 296),
            (8, 384),
            (16, 512),
            (31, 634),
            (32, 640),
            (63, 765),
            (64, 768),
            (127, 894),
            (128, 896),
            (129, 897),
            (1024, 1280),
            (12_345, 1739),
            (32_767, 1919),
            (65_535, 2047),
            (100_000, 2126),
            (123_456_789, 3441),
            (-1, 4095),
        ];

        for (input, expected) in cases {
            assert_eq!(lin2log(input), expected, "lin2log({input})");
        }
    }
}
