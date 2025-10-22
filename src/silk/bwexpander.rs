//! Port of the `silk_bwexpander` helper from `silk/bwexpander.c` in the reference
//! SILK implementation. The routine applies a chirp factor to the LPC coefficients
//! stored in `ar`, gradually shrinking their magnitudes to expand the predictor
//! bandwidth.

/// Chirps (bandwidth expands) an autoregressive filter.
///
/// This mirrors the fixed-point logic from the reference C implementation. The
/// `chirp_q16` argument uses Q16 fixed-point scaling where `1 << 16` represents a
/// factor of `1.0`.
pub fn bwexpander(ar: &mut [i16], chirp_q16: i32) {
    let Some((last, head)) = ar.split_last_mut() else {
        return;
    };

    let mut chirp_q16_inner = chirp_q16;
    let chirp_minus_one_q16 = chirp_q16_inner - (1 << 16);

    for value in head.iter_mut() {
        *value = rshift_round(silk_mul(chirp_q16_inner, i32::from(*value)), 16) as i16;
        chirp_q16_inner += rshift_round(silk_mul(chirp_q16_inner, chirp_minus_one_q16), 16);
    }

    *last = rshift_round(silk_mul(chirp_q16_inner, i32::from(*last)), 16) as i16;
}

fn silk_mul(a: i32, b: i32) -> i64 {
    i64::from(a) * i64::from(b)
}

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
    use super::bwexpander;

    fn q16_from_ratio(numerator: i32, denominator: i32) -> i32 {
        ((numerator * 65_536) + (denominator / 2)) / denominator
    }

    #[test]
    fn matches_reference_case_for_point_nine_chirp() {
        let chirp_q16 = q16_from_ratio(9, 10);
        let mut ar = [8192, -4096, 2048, -1024];

        bwexpander(&mut ar, chirp_q16);

        assert_eq!(ar, [7373, -3318, 1493, -672]);
    }

    #[test]
    fn matches_reference_case_for_half_chirp() {
        let chirp_q16 = q16_from_ratio(1, 2);
        let mut ar = [1000, -1000];

        bwexpander(&mut ar, chirp_q16);

        assert_eq!(ar, [500, -250]);
    }

    #[test]
    fn handles_empty_input() {
        let mut ar: [i16; 0] = [];
        bwexpander(&mut ar, 0);
    }
}
