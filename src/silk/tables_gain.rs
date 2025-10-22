//! Quantization gain probability tables for SILK decoding.
//!
//! Ported from `silk/tables_gain.c` in the reference Opus implementation.

/// Number of discrete quantization gain levels.
pub const N_LEVELS_QGAIN: usize = 64;

/// Maximum delta gain quantization step.
pub const MAX_DELTA_GAIN_QUANT: i32 = 36;

/// Minimum delta gain quantization step.
pub const MIN_DELTA_GAIN_QUANT: i32 = -4;

/// Total number of delta gain quantization entries.
pub const DELTA_GAIN_QUANT_LEVELS: usize =
    (MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT + 1) as usize;

/// C equivalent: `silk_gain_iCDF`.
pub const SILK_GAIN_ICDF: [[u8; N_LEVELS_QGAIN / 8]; 3] = [
    [224, 112, 44, 15, 3, 2, 1, 0],
    [254, 237, 192, 132, 70, 23, 4, 0],
    [255, 252, 226, 155, 61, 11, 2, 0],
];

/// C equivalent: `silk_delta_gain_iCDF`.
pub const SILK_DELTA_GAIN_ICDF: [u8; DELTA_GAIN_QUANT_LEVELS] = [
    250, 245, 234, 203, 71, 50, 42, 38, 35, 33, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
    17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];
