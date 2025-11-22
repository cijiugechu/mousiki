//! Floating-point encoder control structure.
//!
//! Mirrors `silk_encoder_control_FLP` from `silk/float/structs_FLP.h`, exposing
//! the per-frame working buffers produced by the FLP analysis path.

use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER};

/// Working state produced by the floating-point encoder control path each frame.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderControlFlp {
    /// Q0 gains per subframe.
    pub gains: [f32; MAX_NB_SUBFR],
    /// LPC predictor coefficients for the two half frames.
    pub pred_coef: [[f32; MAX_LPC_ORDER]; 2],
    /// LTP predictor coefficients per subframe.
    pub ltp_coef: [f32; MAX_NB_SUBFR * LTP_ORDER],
    /// LTP scaling factor.
    pub ltp_scale: f32,
    /// Pitch lags per subframe (in samples).
    pub pitch_l: [i32; MAX_NB_SUBFR],

    /// Shaping AR coefficients packed across subframes.
    pub ar: [f32; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    /// Low-frequency MA shaping coefficients.
    pub lf_ma_shp: [f32; MAX_NB_SUBFR],
    /// Low-frequency AR shaping coefficients.
    pub lf_ar_shp: [f32; MAX_NB_SUBFR],
    /// Spectral tilt per subframe.
    pub tilt: [f32; MAX_NB_SUBFR],
    /// Harmonic shape gain per subframe.
    pub harm_shape_gain: [f32; MAX_NB_SUBFR],
    /// Rate/distortion trade-off lambda.
    pub lambda: f32,
    /// Input-quality metric.
    pub input_quality: f32,
    /// Coding-quality metric.
    pub coding_quality: f32,

    /// Predicted coding gain.
    pub pred_gain: f32,
    /// Long-term prediction coding gain.
    pub lt_pred_cod_gain: f32,
    /// Residual energy per subframe.
    pub res_nrg: [f32; MAX_NB_SUBFR],

    /// Unquantised gains in Q16 (before scalar quantisation).
    pub gains_unq_q16: [i32; MAX_NB_SUBFR],
    /// Previous frame gain index used for hysteresis.
    pub last_gain_index_prev: i8,
}

impl Default for EncoderControlFlp {
    fn default() -> Self {
        Self {
            gains: [0.0; MAX_NB_SUBFR],
            pred_coef: [[0.0; MAX_LPC_ORDER]; 2],
            ltp_coef: [0.0; MAX_NB_SUBFR * LTP_ORDER],
            ltp_scale: 0.0,
            pitch_l: [0; MAX_NB_SUBFR],
            ar: [0.0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
            lf_ma_shp: [0.0; MAX_NB_SUBFR],
            lf_ar_shp: [0.0; MAX_NB_SUBFR],
            tilt: [0.0; MAX_NB_SUBFR],
            harm_shape_gain: [0.0; MAX_NB_SUBFR],
            lambda: 0.0,
            input_quality: 0.0,
            coding_quality: 0.0,
            pred_gain: 0.0,
            lt_pred_cod_gain: 0.0,
            res_nrg: [0.0; MAX_NB_SUBFR],
            gains_unq_q16: [0; MAX_NB_SUBFR],
            last_gain_index_prev: 0,
        }
    }
}
