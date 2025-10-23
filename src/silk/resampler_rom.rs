//! Coefficient tables used by SILK's fixed-point resamplers.
//!
//! Ported from `silk/resampler_rom.c` and `silk/resampler_rom.h` in the reference
//! Opus implementation.
//! Source: https://gitlab.xiph.org/xiph/opus/-/blob/v1.5.2/silk/resampler_rom.c
//! Values copied verbatim; see the upstream license header for provenance.

/// Order of the first fractional downsampler FIR sections.
pub const RESAMPLER_DOWN_ORDER_FIR0: usize = 18;

/// Order of the second fractional downsampler FIR sections.
pub const RESAMPLER_DOWN_ORDER_FIR1: usize = 24;

/// Order of the third fractional downsampler FIR sections.
pub const RESAMPLER_DOWN_ORDER_FIR2: usize = 36;

/// Order of the 1/12 fractional interpolation filter.
pub const RESAMPLER_ORDER_FIR_12: usize = 8;

/// Q15 coefficient for the first all-pass section in the 2× downsampler.
pub const SILK_RESAMPLER_DOWN2_0: i16 = 9_872;

/// Q15 coefficient for the second all-pass section in the 2× downsampler.
pub const SILK_RESAMPLER_DOWN2_1: i16 = -25_727;

/// Q15 coefficients for the first branch of the high-quality 2× upsampler.
pub const SILK_RESAMPLER_UP2_HQ_0: [i16; 3] = [1_746, 14_986, -26_453];

/// Q15 coefficients for the second branch of the high-quality 2× upsampler.
pub const SILK_RESAMPLER_UP2_HQ_1: [i16; 3] = [6_854, 25_769, -9_994];

/// IIR/FIR coefficients for the 3/4 fractional downsampler.
#[rustfmt::skip]
pub static SILK_RESAMPLER_3_4_COEFS: [i16; 2 + 3 * RESAMPLER_DOWN_ORDER_FIR0 / 2] = [
    -20_694, -13_867,
        -49,     64,     17,   -157,    353,   -496,    163,  11_047,  22_205,
        -39,      6,     91,   -170,    186,     23,   -896,   6_336,  19_928,
        -19,    -36,    102,    -89,    -24,    328,   -951,   2_568,  15_909,
];

/// IIR/FIR coefficients for the 2/3 fractional downsampler.
#[rustfmt::skip]
pub static SILK_RESAMPLER_2_3_COEFS: [i16; 2 + 2 * RESAMPLER_DOWN_ORDER_FIR0 / 2] = [
    -14_457, -14_019,
         64,    128,   -122,     36,    310,   -768,    584,   9_267,  17_733,
         12,    128,     18,   -142,    288,   -117,   -865,   4_123,  14_459,
];

/// IIR/FIR coefficients for the 1/2 fractional downsampler.
#[rustfmt::skip]
pub static SILK_RESAMPLER_1_2_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR1 / 2] = [
      616, -14_323,
       -10,     39,     58,    -46,    -84,    120,    184,   -315,   -541,   1_284,
     5_380,   9_024,
];

/// IIR/FIR coefficients for the 1/3 fractional downsampler.
#[rustfmt::skip]
pub static SILK_RESAMPLER_1_3_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2] = [
    16_102, -15_162,
       -13,      0,     20,     26,      5,    -31,    -43,     -4,     65,     90,      7,
      -157,   -248,    -44,    593,   1_583,  2_612,  3_271,
];

/// IIR/FIR coefficients for the 1/4 fractional downsampler.
#[rustfmt::skip]
pub static SILK_RESAMPLER_1_4_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2] = [
    22_500, -15_099,
         3,    -14,    -20,    -15,      2,     25,     37,     25,    -16,    -71,   -107,
       -79,     50,    292,    623,    982,  1_288,  1_464,
];

/// IIR/FIR coefficients for the 1/6 fractional downsampler.
#[rustfmt::skip]
pub static SILK_RESAMPLER_1_6_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2] = [
    27_540, -15_257,
        17,     12,      8,      1,    -10,    -22,    -30,    -32,    -22,      3,     44,
       100,    168,    243,    317,    381,    429,    455,
];

/// Low-quality coefficients for the 2/3 fractional downsampler.
#[rustfmt::skip]
pub static SILK_RESAMPLER_2_3_COEFS_LQ: [i16; 2 + 2 * 2] = [
    -2_797,  -6_507,
     4_697,  10_739,
     1_567,   8_276,
];

/// Interpolation fractions of 1/24, 3/24, ..., 23/24 for the 12-phase FIR.
#[rustfmt::skip]
pub static SILK_RESAMPLER_FRAC_FIR_12: [[i16; RESAMPLER_ORDER_FIR_12 / 2]; 12] = [
    [  189,  -600,   617, 30_567 ],
    [  117,  -159, -1_070, 29_704 ],
    [   52,   221, -2_392, 28_276 ],
    [   -4,   529, -3_350, 26_341 ],
    [  -48,   758, -3_956, 23_973 ],
    [  -80,   905, -4_235, 21_254 ],
    [  -99,   972, -4_222, 18_278 ],
    [ -107,   967, -3_957, 15_143 ],
    [ -103,   896, -3_487, 11_950 ],
    [  -91,   773, -2_865,  8_798 ],
    [  -71,   611, -2_143,  5_784 ],
    [  -46,   425, -1_375,  2_996 ],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_tables_have_expected_shapes() {
        assert_eq!(
            SILK_RESAMPLER_3_4_COEFS.len(),
            2 + 3 * RESAMPLER_DOWN_ORDER_FIR0 / 2
        );
        assert_eq!(
            SILK_RESAMPLER_2_3_COEFS.len(),
            2 + 2 * RESAMPLER_DOWN_ORDER_FIR0 / 2
        );
        assert_eq!(
            SILK_RESAMPLER_1_2_COEFS.len(),
            2 + RESAMPLER_DOWN_ORDER_FIR1 / 2
        );
        assert_eq!(
            SILK_RESAMPLER_1_3_COEFS.len(),
            2 + RESAMPLER_DOWN_ORDER_FIR2 / 2
        );
        assert_eq!(
            SILK_RESAMPLER_1_4_COEFS.len(),
            2 + RESAMPLER_DOWN_ORDER_FIR2 / 2
        );
        assert_eq!(
            SILK_RESAMPLER_1_6_COEFS.len(),
            2 + RESAMPLER_DOWN_ORDER_FIR2 / 2
        );
        assert_eq!(SILK_RESAMPLER_2_3_COEFS_LQ.len(), 6);
        assert_eq!(SILK_RESAMPLER_FRAC_FIR_12.len(), 12);
        assert_eq!(
            SILK_RESAMPLER_FRAC_FIR_12[0].len(),
            RESAMPLER_ORDER_FIR_12 / 2
        );
    }

    #[test]
    fn reference_values_match_c_tables() {
        assert_eq!(SILK_RESAMPLER_DOWN2_0, 9_872);
        assert_eq!(SILK_RESAMPLER_DOWN2_1, -25_727);
        assert_eq!(SILK_RESAMPLER_UP2_HQ_0, [1_746, 14_986, -26_453]);
        assert_eq!(SILK_RESAMPLER_UP2_HQ_1, [6_854, 25_769, -9_994]);
        assert_eq!(SILK_RESAMPLER_3_4_COEFS[0], -20_694);
        assert_eq!(SILK_RESAMPLER_3_4_COEFS[28], 15_909);
        assert_eq!(SILK_RESAMPLER_2_3_COEFS[10], 17_733);
        assert_eq!(SILK_RESAMPLER_1_2_COEFS[11], 1_284);
        assert_eq!(SILK_RESAMPLER_1_3_COEFS[16], 593);
        assert_eq!(SILK_RESAMPLER_1_4_COEFS[13], -79);
        assert_eq!(SILK_RESAMPLER_1_6_COEFS[16], 317);
        assert_eq!(SILK_RESAMPLER_2_3_COEFS_LQ[3], 10_739);
        assert_eq!(SILK_RESAMPLER_FRAC_FIR_12[0], [189, -600, 617, 30_567]);
        assert_eq!(SILK_RESAMPLER_FRAC_FIR_12[11], [-46, 425, -1_375, 2_996]);
    }

    #[test]
    fn tables_have_stable_checksums() {
        fn sum_i32(xs: &[i16]) -> i32 {
            xs.iter().map(|&v| i32::from(v)).sum()
        }

        assert_eq!(sum_i32(&SILK_RESAMPLER_3_4_COEFS), 41_839);
        assert_eq!(sum_i32(&SILK_RESAMPLER_2_3_COEFS), 16_660);
        assert_eq!(sum_i32(&SILK_RESAMPLER_1_2_COEFS), 1_386);
        assert_eq!(sum_i32(&SILK_RESAMPLER_1_3_COEFS), 8_672);
        assert_eq!(sum_i32(&SILK_RESAMPLER_1_4_COEFS), 11_870);
        assert_eq!(sum_i32(&SILK_RESAMPLER_1_6_COEFS), 14_345);
        assert_eq!(sum_i32(&SILK_RESAMPLER_2_3_COEFS_LQ), 15_975);

        let mut frac_sum = 0;
        for row in SILK_RESAMPLER_FRAC_FIR_12.iter() {
            frac_sum += sum_i32(row);
        }
        assert_eq!(frac_sum, 196_636);
    }
}
