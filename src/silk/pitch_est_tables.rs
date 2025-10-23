//! Pitch estimator codebook tables and configuration constants for SILK decoding.
//!
//! Ported from `silk/pitch_est_tables.c` and `silk/pitch_est_defines.h` in the
//! reference Opus implementation.

/// Maximum sampling frequency (in kHz) supported by the pitch estimator.
pub const PE_MAX_FS_KHZ: usize = 16;

/// Maximum number of 5 ms subframes within a SILK frame.
pub const PE_MAX_NB_SUBFR: usize = 4;

/// Subframe duration in milliseconds.
pub const PE_SUBFR_LENGTH_MS: usize = 5;

/// Size of the LTP history buffer in milliseconds.
pub const PE_LTP_MEM_LENGTH_MS: usize = 4 * PE_SUBFR_LENGTH_MS;

/// Maximum frame duration in milliseconds.
pub const PE_MAX_FRAME_LENGTH_MS: usize =
    PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS;

/// Maximum frame length when expressed in samples.
pub const PE_MAX_FRAME_LENGTH: usize = PE_MAX_FRAME_LENGTH_MS * PE_MAX_FS_KHZ;

/// Maximum frame length divided by four.
pub const PE_MAX_FRAME_LENGTH_ST_1: usize = PE_MAX_FRAME_LENGTH >> 2;

/// Maximum frame length divided by two.
pub const PE_MAX_FRAME_LENGTH_ST_2: usize = PE_MAX_FRAME_LENGTH >> 1;

/// Upper bound on the pitch lag in milliseconds.
pub const PE_MAX_LAG_MS: usize = 18;

/// Lower bound on the pitch lag in milliseconds.
pub const PE_MIN_LAG_MS: usize = 2;

/// Upper bound on the pitch lag in samples.
pub const PE_MAX_LAG: usize = PE_MAX_LAG_MS * PE_MAX_FS_KHZ;

/// Lower bound on the pitch lag in samples.
pub const PE_MIN_LAG: usize = PE_MIN_LAG_MS * PE_MAX_FS_KHZ;

/// Number of lags searched during the delayed-decision stage.
pub const PE_D_SRCH_LENGTH: usize = 24;

/// Number of codebook entries used during stage three lag refinement.
pub const PE_NB_STAGE3_LAGS: usize = 5;

/// Number of stage-two codebook entries for 20 ms frames.
pub const PE_NB_CBKS_STAGE2: usize = 3;

/// Extended number of stage-two codebook entries for 20 ms frames.
pub const PE_NB_CBKS_STAGE2_EXT: usize = 11;

/// Maximum number of stage-three codebook entries for 20 ms frames.
pub const PE_NB_CBKS_STAGE3_MAX: usize = 34;

/// Middle number of stage-three codebook entries for 20 ms frames.
pub const PE_NB_CBKS_STAGE3_MID: usize = 24;

/// Minimum number of stage-three codebook entries for 20 ms frames.
pub const PE_NB_CBKS_STAGE3_MIN: usize = 16;

/// Number of stage-three codebook entries for 10 ms frames.
pub const PE_NB_CBKS_STAGE3_10_MS: usize = 12;

/// Number of stage-two codebook entries for 10 ms frames.
pub const PE_NB_CBKS_STAGE2_10_MS: usize = 3;

/// Bias applied to favour shorter lags during the search.
pub const PE_SHORTLAG_BIAS: f32 = 0.2;

/// Bias applied to favour previously observed lags.
pub const PE_PREVLAG_BIAS: f32 = 0.2;

/// Bias applied to prefer flatter pitch contours.
pub const PE_FLATCONTOUR_BIAS: f32 = 0.05;

/// Minimum complexity setting for the pitch estimator.
pub const SILK_PE_MIN_COMPLEX: usize = 0;

/// Medium complexity setting for the pitch estimator.
pub const SILK_PE_MID_COMPLEX: usize = 1;

/// Maximum complexity setting for the pitch estimator.
pub const SILK_PE_MAX_COMPLEX: usize = 2;

const PE_MAX_NB_SUBFR_HALF: usize = PE_MAX_NB_SUBFR / 2;

/// C equivalent: `silk_CB_lags_stage2_10_ms`.
pub const SILK_CB_LAGS_STAGE2_10_MS: [[i8; PE_NB_CBKS_STAGE2_10_MS]; PE_MAX_NB_SUBFR_HALF] =
    [[0, 1, 0], [0, 0, 1]];

/// C equivalent: `silk_CB_lags_stage3_10_ms`.
pub const SILK_CB_LAGS_STAGE3_10_MS: [[i8; PE_NB_CBKS_STAGE3_10_MS]; PE_MAX_NB_SUBFR_HALF] = [
    [0, 0, 1, -1, 1, -1, 2, -2, 2, -2, 3, -3],
    [0, 1, 0, 1, -1, 2, -1, 2, -2, 3, -2, 3],
];

/// C equivalent: `silk_Lag_range_stage3_10_ms`.
pub const SILK_LAG_RANGE_STAGE3_10_MS: [[i8; 2]; PE_MAX_NB_SUBFR_HALF] = [[-3, 7], [-2, 7]];

/// C equivalent: `silk_CB_lags_stage2`.
pub const SILK_CB_LAGS_STAGE2: [[i8; PE_NB_CBKS_STAGE2_EXT]; PE_MAX_NB_SUBFR] = [
    [0, 2, -1, -1, -1, 0, 0, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, -1, 2, 1, 0, 1, 1, 0, 0, -1, -1],
];

/// C equivalent: `silk_CB_lags_stage3`.
pub const SILK_CB_LAGS_STAGE3: [[i8; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR] = [
    [
        0, 0, 1, -1, 0, 1, -1, 0, -1, 1, -2, 2, -2, -2, 2, -3, 2, 3, -3, -4, 3, -4, 4, 4, -5, 5,
        -6, -5, 6, -7, 6, 5, 8, -9,
    ],
    [
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 1, -1, -1, 1, -1, 2, 1, -1, 2, -2, -2,
        2, -2, 2, 2, 3, -3,
    ],
    [
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, -1, 1, 0, 0, 2, 1, -1, 2, -1, -1, 2, -1, 2, 2,
        -1, 3, -2, -2, -2, 3,
    ],
    [
        0, 1, 0, 0, 1, 0, 1, -1, 2, -1, 2, -1, 2, 3, -2, 3, -2, -2, 4, 4, -3, 5, -3, -4, 6, -4, 6,
        5, -5, 8, -6, -5, -7, 9,
    ],
];

/// C equivalent: `silk_Lag_range_stage3`.
pub const SILK_LAG_RANGE_STAGE3: [[[i8; 2]; PE_MAX_NB_SUBFR]; SILK_PE_MAX_COMPLEX + 1] = [
    [[-5, 8], [-1, 6], [-1, 6], [-4, 10]],
    [[-6, 10], [-2, 6], [-1, 6], [-5, 10]],
    [[-9, 12], [-3, 7], [-2, 7], [-7, 13]],
];

/// C equivalent: `silk_nb_cbk_searchs_stage3`.
pub const SILK_NB_CBK_SEARCHS_STAGE3: [i8; SILK_PE_MAX_COMPLEX + 1] = [
    PE_NB_CBKS_STAGE3_MIN as i8,
    PE_NB_CBKS_STAGE3_MID as i8,
    PE_NB_CBKS_STAGE3_MAX as i8,
];
