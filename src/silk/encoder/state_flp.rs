//! Floating-point encoder state scaffolding.
//!
//! Mirrors `silk_encoder_state_FLP` from `silk/float/structs_FLP.h`, layering
//! a small amount of FLP-specific state on top of the common encoder fields.

use super::state::{EncoderStateCommon, VadState, X_BUFFER_LENGTH};
use crate::silk::lp_variable_cutoff::LpState;

/// Floating-point noise-shaping analysis state (`silk_shape_state_FLP`).
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderShapeStateFlp {
    /// Previous gain index used by the quantiser.
    pub last_gain_index: i8,
    /// Smoothed harmonic shape gain.
    pub harm_shape_gain_smth: f32,
    /// Smoothed spectral tilt.
    pub tilt_smth: f32,
}

impl Default for EncoderShapeStateFlp {
    fn default() -> Self {
        Self {
            last_gain_index: 0,
            harm_shape_gain_smth: 0.0,
            tilt_smth: 0.0,
        }
    }
}

/// Floating-point encoder channel state (mirror of `silk_encoder_state_FLP`).
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderStateFlp {
    /// Common encoder fields shared with the fixed-point build.
    pub common: EncoderStateCommon,
    // Mirror the fixed-point layout: keep VAD/LP outside `common` so split borrows stay safe.
    /// Voice activity detector state.
    pub vad_state: VadState,
    /// Variable low-pass filter state used during bandwidth transitions.
    pub lp_state: LpState,
    /// Floating-point noise-shaping state.
    pub shape_state: EncoderShapeStateFlp,
    /// Pitch/noise-shaping analysis buffer.
    pub x_buf: [f32; X_BUFFER_LENGTH],
    /// Normalised correlation from the pitch-lag estimator.
    pub ltp_corr: f32,
}

impl Default for EncoderStateFlp {
    fn default() -> Self {
        Self {
            common: EncoderStateCommon::default(),
            vad_state: VadState::default(),
            lp_state: LpState::default(),
            shape_state: EncoderShapeStateFlp::default(),
            x_buf: [0.0; X_BUFFER_LENGTH],
            ltp_corr: 0.0,
        }
    }
}

impl EncoderStateFlp {
    /// Creates a new floating-point encoder channel state with default fields.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}
