//! Approximate sigmoid helper used throughout the SILK fixed-point routines.
//!
//! This module mirrors the lookup-table based `silk_sigm_Q15` implementation
//! from `silk/sigm_Q15.c` in the reference Opus sources. The function maps a
//! Q5 fixed-point input to a saturated Q15 output while avoiding expensive
//! transcendental operations.

/// Slope values in Q10 used to interpolate between the sigmoid lookup entries.
const SIGM_LUT_SLOPE_Q10: [i32; 6] = [237, 153, 73, 30, 12, 7];

/// Sigmoid lookup table for non-negative inputs expressed in Q15.
const SIGM_LUT_POS_Q15: [i32; 6] = [16384, 23955, 28861, 31213, 32178, 32548];

/// Sigmoid lookup table for non-positive inputs expressed in Q15.
const SIGM_LUT_NEG_Q15: [i32; 6] = [16384, 8812, 3906, 1554, 589, 219];

/// Approximate logistic function working on Q5 fixed-point arguments.
///
/// The routine clamps large magnitudes to the `[0, 32767]` range and mirrors the
/// behaviour of the C implementation used by SILK's predictor tuning helpers.
#[must_use]
pub fn sigm_q15(mut input_q5: i32) -> i32 {
    if input_q5 < 0 {
        input_q5 = -input_q5;
        if input_q5 >= 6 * 32 {
            0
        } else {
            let index = (input_q5 >> 5) as usize;
            let fractional = input_q5 & 0x1f;
            SIGM_LUT_NEG_Q15[index] - SIGM_LUT_SLOPE_Q10[index] * fractional
        }
    } else if input_q5 >= 6 * 32 {
        32_767
    } else {
        let index = (input_q5 >> 5) as usize;
        let fractional = input_q5 & 0x1f;
        SIGM_LUT_POS_Q15[index] + SIGM_LUT_SLOPE_Q10[index] * fractional
    }
}

#[cfg(test)]
mod tests {
    use super::sigm_q15;

    #[test]
    fn clamps_for_large_magnitudes() {
        assert_eq!(sigm_q15(192), 32_767);
        assert_eq!(sigm_q15(256), 32_767);
        assert_eq!(sigm_q15(-192), 0);
        assert_eq!(sigm_q15(-500), 0);
    }

    #[test]
    fn matches_lookup_table_anchors() {
        let anchors = [
            (0, 16_384),
            (32, 23_955),
            (64, 28_861),
            (96, 31_213),
            (128, 32_178),
            (160, 32_548),
        ];

        let negatives = [16_384, 8_812, 3_906, 1_554, 589, 219];

        for ((input, expected_pos), expected_neg) in anchors.into_iter().zip(negatives) {
            assert_eq!(sigm_q15(input), expected_pos, "sigm_q15({input})");
            assert_eq!(
                sigm_q15(-input),
                expected_neg,
                "sigm_q15({input}) negative anchor"
            );
        }
    }

    #[test]
    fn interpolates_between_entries() {
        let cases = [
            (1, 16_621),
            (31, 23_731),
            (33, 24_108),
            (95, 31_124),
            (127, 32_143),
            (159, 32_550),
            (-1, 16_147),
            (-31, 9_037),
            (-33, 8_659),
            (-95, 1_643),
            (-127, 624),
            (-159, 217),
        ];

        for (input, expected) in cases {
            assert_eq!(sigm_q15(input), expected, "sigm_q15({input})");
        }
    }
}
