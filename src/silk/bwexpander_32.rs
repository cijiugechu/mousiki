//! Port of the `silk_bwexpander_32` helper from `silk/bwexpander_32.c` in the
//! reference SILK implementation. The routine applies a Q16 chirp factor to a
//! set of 32-bit LPC coefficients, gradually shrinking their magnitudes to
//! expand the filter bandwidth.

/// Chirps (bandwidth expands) a 32-bit autoregressive filter.
///
/// The `chirp_q16` argument uses Q16 fixed-point scaling where `1 << 16`
/// represents a factor of `1.0`.
pub fn bwexpander_32(ar: &mut [i32], chirp_q16: i32) {
    let Some((last, head)) = ar.split_last_mut() else {
        return;
    };

    let mut chirp_q16_inner = chirp_q16;
    let chirp_minus_one_q16 = chirp_q16_inner.wrapping_sub(1 << 16);

    for value in head.iter_mut() {
        *value = smulww(chirp_q16_inner, *value);
        let product = mul(chirp_q16_inner, chirp_minus_one_q16);
        chirp_q16_inner = chirp_q16_inner.wrapping_add(rshift_round(product, 16));
    }

    *last = smulww(chirp_q16_inner, *last);
}

#[inline]
fn mul(a: i32, b: i32) -> i64 {
    i64::from(a) * i64::from(b)
}

#[inline]
fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

#[inline]
fn rshift_round(value: i64, shift: u32) -> i32 {
    if shift == 0 {
        return value as i32;
    }

    let rounded = if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    };

    rounded as i32
}

#[cfg(test)]
mod tests {
    use super::bwexpander_32;

    fn q16_from_ratio(numerator: i32, denominator: i32) -> i32 {
        ((numerator * 65_536) + (denominator / 2)) / denominator
    }

    #[test]
    fn leaves_coefficients_unchanged_for_unity_chirp() {
        let mut ar = [123_456_789, -98_765_432];
        bwexpander_32(&mut ar, 1 << 16);
        assert_eq!(ar, [123_456_789, -98_765_432]);
    }

    #[test]
    fn attenuates_coefficients_for_point_nine_chirp() {
        let chirp_q16 = q16_from_ratio(9, 10);
        let mut ar = [32_000_000, -16_000_000, 8_000_000, -4_000_000];

        bwexpander_32(&mut ar, chirp_q16);

        assert_eq!(ar, [28_799_804, -12_959_717, 5_831_787, -2_624_268]);
    }

    #[test]
    fn handles_empty_input() {
        let mut ar: [i32; 0] = [];
        bwexpander_32(&mut ar, 0);
    }
}
