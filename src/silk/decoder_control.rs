//! Rust mirror of `silk_decoder_control` from `silk/structs.h`.
//!
//! The SILK decoder reconstructs predictor gains, pitch lags, and LTP tap
//! metadata via `silk_decode_parameters`. The original C implementation stores
//! those per-frame parameters inside `silk_decoder_control`; this module
//! exposes the same layout so that helpers such as PLC, CNG, and the inverse
//! NSQ path can share a single typed representation.

use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR};

/// Decoder control parameters produced by `silk_decode_parameters`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderControl {
    /// Q0 pitch lags per subframe.
    pub pitch_l: [i32; MAX_NB_SUBFR],
    /// Q16 gains per subframe.
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// LPC predictor coefficients for each half-frame (Q12).
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    /// Long-term prediction taps for each subframe (Q14).
    pub ltp_coef_q14: [i16; MAX_NB_SUBFR * LTP_ORDER],
    /// LTP scaling factor (Q14).
    pub ltp_scale_q14: i32,
}

impl Default for DecoderControl {
    fn default() -> Self {
        Self {
            pitch_l: [0; MAX_NB_SUBFR],
            gains_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; MAX_NB_SUBFR * LTP_ORDER],
            ltp_scale_q14: 0,
        }
    }
}
