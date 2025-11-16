//! Encoder per-frame control structure.
//!
//! Mirrors `silk_encoder_control_FIX` from `silk/fixed/structs_FIX.h`. The struct stores the
//! temporary parameters derived by helpers such as `silk_noise_shape_analysis_FIX`,
//! `silk_process_gains_FIX`, and `silk_quant_ltp_gains`. Keeping the layout close to the
//! reference implementation lets the rest of the encoder port reuse the same storage without
//! bespoke adapters.

use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER};

/// Working state produced by the encoder control path each frame.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderControl {
    /// Q16 gains per subframe.
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// LPC predictor coefficients in Q12 for the two half frames.
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    /// LTP predictor coefficients in Q14 for each subframe.
    pub ltp_coef_q14: [i16; MAX_NB_SUBFR * LTP_ORDER],
    /// Q14 LTP scaling factor.
    pub ltp_scale_q14: i32,
    /// Pitch lags per subframe (in samples).
    pub pitch_l: [i32; MAX_NB_SUBFR],
    /// Shaping AR coefficients packed across subframes (Q13).
    pub ar_q13: [i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    /// Low-frequency shaping coefficients in Q14.
    pub lf_shp_q14: [i32; MAX_NB_SUBFR],
    /// Per-subframe spectral tilt in Q14.
    pub tilt_q14: [i32; MAX_NB_SUBFR],
    /// Per-subframe harmonic shape gain in Q14.
    pub harm_shape_gain_q14: [i32; MAX_NB_SUBFR],
    /// Rate/distortion trade-off lambda in Q10.
    pub lambda_q10: i32,
    /// Input-quality metric in Q14.
    pub input_quality_q14: i32,
    /// Coding-quality metric in Q14.
    pub coding_quality_q14: i32,
    /// Predicted coding gain in Q16.
    pub pred_gain_q16: i32,
    /// Long-term prediction coding gain in Q7.
    pub lt_pred_cod_gain_q7: i32,
    /// Residual energies per subframe.
    pub res_nrg: [i32; MAX_NB_SUBFR],
    /// Q-domain for each residual energy entry.
    pub res_nrg_q: [i32; MAX_NB_SUBFR],
    /// Unquantised gains in Q16 (before scalar quantisation).
    pub gains_unq_q16: [i32; MAX_NB_SUBFR],
    /// Previous frame gain index used for hysteresis.
    pub last_gain_index_prev: i8,
}

impl Default for EncoderControl {
    fn default() -> Self {
        Self {
            gains_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; MAX_NB_SUBFR * LTP_ORDER],
            ltp_scale_q14: 0,
            pitch_l: [0; MAX_NB_SUBFR],
            ar_q13: [0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
            lf_shp_q14: [0; MAX_NB_SUBFR],
            tilt_q14: [0; MAX_NB_SUBFR],
            harm_shape_gain_q14: [0; MAX_NB_SUBFR],
            lambda_q10: 0,
            input_quality_q14: 0,
            coding_quality_q14: 0,
            pred_gain_q16: 0,
            lt_pred_cod_gain_q7: 0,
            res_nrg: [0; MAX_NB_SUBFR],
            res_nrg_q: [0; MAX_NB_SUBFR],
            gains_unq_q16: [0; MAX_NB_SUBFR],
            last_gain_index_prev: 0,
        }
    }
}
