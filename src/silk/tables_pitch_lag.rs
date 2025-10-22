//! Probability tables for SILK pitch lag decoding.
//!
//! Ported from `silk/tables_pitch_lag.c` in the reference Opus implementation.

/// C equivalent: `silk_pitch_lag_iCDF`.
pub const PITCH_LAG_ICDF: [u8; 32] = [
    253, 250, 244, 233, 212, 182, 150, 131, 120, 110, 98, 85, 72, 60, 49, 40, 32, 25, 19, 15, 13,
    11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];

/// C equivalent: `silk_pitch_delta_iCDF`.
pub const PITCH_DELTA_ICDF: [u8; 21] = [
    210, 208, 206, 203, 199, 193, 183, 168, 142, 104, 74, 52, 37, 27, 20, 14, 10, 6, 4, 2, 0,
];

/// C equivalent: `silk_pitch_contour_iCDF`.
pub const PITCH_CONTOUR_ICDF: [u8; 34] = [
    223, 201, 183, 167, 152, 138, 124, 111, 98, 88, 79, 70, 62, 56, 50, 44, 39, 35, 31, 27, 24, 21,
    18, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1, 0,
];

/// C equivalent: `silk_pitch_contour_NB_iCDF`.
pub const PITCH_CONTOUR_NB_ICDF: [u8; 11] = [188, 176, 155, 138, 119, 97, 67, 43, 26, 10, 0];

/// C equivalent: `silk_pitch_contour_10_ms_iCDF`.
pub const PITCH_CONTOUR_10_MS_ICDF: [u8; 12] = [165, 119, 80, 61, 47, 35, 27, 20, 14, 9, 4, 0];

/// C equivalent: `silk_pitch_contour_10_ms_NB_iCDF`.
pub const PITCH_CONTOUR_10_MS_NB_ICDF: [u8; 3] = [113, 63, 0];
