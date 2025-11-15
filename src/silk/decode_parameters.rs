//! Port of `silk/decode_parameters.c`.
//!
//! This helper reconstructs the decoder-side predictor coefficients, gains,
//! and long-term prediction metadata after the range decoder has emitted the
//! compact side information.  It mirrors the fixed-point reference implementation
//! so higher-level decode drivers can reuse the same LPC and LTP parameters.

use core::convert::TryFrom;

use crate::silk::bwexpander::bwexpander;
use crate::silk::decode_indices::{ConditionalCoding, SideInfoIndices};
use crate::silk::decode_pitch::silk_decode_pitch;
use crate::silk::gain_quant::silk_gains_dequant;
use crate::silk::nlsf_decode::nlsf_decode;
use crate::silk::nlsf2a::nlsf2a;
use crate::silk::decoder_control::DecoderControl;
use crate::silk::tables_ltp::SILK_LTP_GAIN_VQ_Q7;
use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;
use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR, SilkNlsfCb};

const BWE_AFTER_LOSS_Q16: i32 = 63_570;

/// Minimal decoder state needed by [`silk_decode_parameters`].
#[derive(Clone, Debug)]
pub struct DecoderParametersState {
    /// Entropy-decoded side information for the current frame.
    pub indices: SideInfoIndices,
    /// Last decoded NLSF vector expressed in Q15.
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    /// Active SILK NLSF codebook.
    pub nlsf_codebook: &'static SilkNlsfCb,
    /// LPC order (10 for NB/MB, 16 for WB/SWB).
    pub lpc_order: usize,
    /// Number of 5 ms subframes tracked by the decoder.
    pub nb_subfr: usize,
    /// Internal sampling rate in kHz (8/12/16).
    pub fs_khz: i32,
    /// Number of consecutive packet losses observed.
    pub loss_count: i32,
    /// Whether the previous frame reset the decoder state.
    pub first_frame_after_reset: bool,
    /// Previous de-quantised gain index.
    pub last_gain_index: i8,
    /// Architecture selector retained for API parity (unused, but kept for future SIMD hooks).
    pub arch: i32,
}

impl Default for DecoderParametersState {
    fn default() -> Self {
        Self {
            indices: SideInfoIndices::default(),
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            nlsf_codebook: &SILK_NLSF_CB_WB,
            lpc_order: MAX_LPC_ORDER,
            nb_subfr: MAX_NB_SUBFR,
            fs_khz: 16,
            loss_count: 0,
            first_frame_after_reset: true,
            last_gain_index: 0,
            arch: 0,
        }
    }
}

impl DecoderParametersState {
    /// Creates a state that uses the supplied NLSF codebook.
    pub fn with_codebook(nlsf_codebook: &'static SilkNlsfCb) -> Self {
        Self {
            nlsf_codebook,
            ..Self::default()
        }
    }
}

/// Mirrors `silk_decode_parameters`: decodes gains, predictor coefficients,
/// and LTP metadata from the side-information indices.
pub fn silk_decode_parameters(
    state: &mut DecoderParametersState,
    control: &mut DecoderControl,
    cond_coding: ConditionalCoding,
) {
    let nb_subfr = state.nb_subfr;
    assert!(nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2);
    let order = state.lpc_order;
    assert!((1..=MAX_LPC_ORDER).contains(&order));
    assert!(matches!(state.fs_khz, 8 | 12 | 16));

    let codebook_order = usize::try_from(state.nlsf_codebook.order)
        .expect("NLSF codebook order must fit into usize");
    assert_eq!(
        order, codebook_order,
        "LPC order must match NLSF codebook order"
    );

    silk_gains_dequant(
        &mut control.gains_q16[..nb_subfr],
        &state.indices.gains_indices[..nb_subfr],
        &mut state.last_gain_index,
        matches!(cond_coding, ConditionalCoding::Conditional),
    );
    control
        .gains_q16
        .iter_mut()
        .skip(nb_subfr)
        .for_each(|gain| *gain = 0);

    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    nlsf_decode(
        &mut nlsf_q15[..order],
        &state.indices.nlsf_indices[..order + 1],
        state.nlsf_codebook,
    );
    nlsf2a(
        &mut control.pred_coef_q12[1][..order],
        &nlsf_q15[..order],
        state.arch,
    );

    if state.first_frame_after_reset {
        state.indices.nlsf_interp_coef_q2 = 4;
    }

    if state.indices.nlsf_interp_coef_q2 < 4 {
        let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
        let interp_q2 = i32::from(state.indices.nlsf_interp_coef_q2);
        for i in 0..order {
            let prev = i32::from(state.prev_nlsf_q15[i]);
            let curr = i32::from(nlsf_q15[i]);
            let delta = curr - prev;
            let blended = prev + ((interp_q2 * delta) >> 2);
            nlsf0_q15[i] = blended.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
        nlsf2a(
            &mut control.pred_coef_q12[0][..order],
            &nlsf0_q15[..order],
            state.arch,
        );
    } else {
        let row = control.pred_coef_q12[1];
        control.pred_coef_q12[0][..order].copy_from_slice(&row[..order]);
    }

    for row in &mut control.pred_coef_q12 {
        row.iter_mut().skip(order).for_each(|coef| *coef = 0);
    }

    state.prev_nlsf_q15[..order].copy_from_slice(&nlsf_q15[..order]);
    state
        .prev_nlsf_q15
        .iter_mut()
        .skip(order)
        .for_each(|value| *value = 0);

    if state.loss_count > 0 {
        for row in &mut control.pred_coef_q12 {
            bwexpander(&mut row[..order], BWE_AFTER_LOSS_Q16);
        }
    }

    control.pitch_l.fill(0);
    control.ltp_coef_q14.fill(0);
    control.ltp_scale_q14 = 0;

    if matches!(state.indices.signal_type, FrameSignalType::Voiced) {
        silk_decode_pitch(
            state.indices.lag_index,
            state.indices.contour_index,
            &mut control.pitch_l,
            state.fs_khz,
            nb_subfr,
        );

        let per_index =
            usize::try_from(state.indices.per_index).expect("PERIndex must be non-negative");
        let codebook = SILK_LTP_GAIN_VQ_Q7
            .get(per_index)
            .expect("PERIndex out of range");

        for (subframe, &ltp_index) in state.indices.ltp_index.iter().take(nb_subfr).enumerate() {
            let row = usize::try_from(ltp_index).expect("LTPIndex must be non-negative");
            let taps = codebook.get(row).expect("LTPIndex out of range");
            for (tap, &value_q7) in taps.iter().enumerate() {
                control.ltp_coef_q14[subframe * LTP_ORDER + tap] = i16::from(value_q7) << 7;
            }
        }

        let ltp_scale_index = usize::try_from(state.indices.ltp_scale_index)
            .expect("LTP scale index must be non-negative");
        control.ltp_scale_q14 = i32::from(*SILK_LTPSCALES_TABLE_Q14
            .get(ltp_scale_index)
            .expect("LTP scale index out of range"));
    } else {
        state.indices.per_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::decode_pitch::silk_decode_pitch;
    use crate::silk::tables_ltp::SILK_LTP_GAIN_VQ_Q7;
    use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;
    use alloc::vec::Vec;

    fn base_state() -> DecoderParametersState {
        DecoderParametersState {
            first_frame_after_reset: false,
            ..DecoderParametersState::default()
        }
    }

    #[test]
    fn forces_interpolation_skip_on_reset() {
        let mut state = DecoderParametersState::default();
        state.indices.nlsf_interp_coef_q2 = 1;
        state.indices.signal_type = FrameSignalType::Unvoiced;
        let mut control = DecoderControl::default();

        silk_decode_parameters(&mut state, &mut control, ConditionalCoding::Independent);

        assert_eq!(state.indices.nlsf_interp_coef_q2, 4);
        assert!(control.pitch_l.iter().all(|&lag| lag == 0));

        let mut expected_nlsf = [0i16; MAX_LPC_ORDER];
        nlsf_decode(
            &mut expected_nlsf[..state.lpc_order],
            &state.indices.nlsf_indices[..state.lpc_order + 1],
            state.nlsf_codebook,
        );
        assert_eq!(
            &state.prev_nlsf_q15[..state.lpc_order],
            &expected_nlsf[..state.lpc_order]
        );
    }

    #[test]
    fn populates_voiced_ltp_parameters() {
        let mut state = base_state();
        state.indices.signal_type = FrameSignalType::Voiced;
        state.indices.per_index = 0;
        state.indices.ltp_scale_index = 1;
        state.indices.ltp_index = [0; MAX_NB_SUBFR];
        state.indices.lag_index = 5;
        state.indices.contour_index = 0;
        let mut control = DecoderControl::default();

        let mut expected_pitch = [0i32; MAX_NB_SUBFR];
        silk_decode_pitch(
            state.indices.lag_index,
            state.indices.contour_index,
            &mut expected_pitch,
            state.fs_khz,
            state.nb_subfr,
        );

        silk_decode_parameters(&mut state, &mut control, ConditionalCoding::Independent);

        assert_eq!(&control.pitch_l, &expected_pitch);

        let expected_taps: Vec<i16> = SILK_LTP_GAIN_VQ_Q7[0][0]
            .iter()
            .map(|&tap| i16::from(tap) << 7)
            .collect();
        assert_eq!(&control.ltp_coef_q14[..LTP_ORDER], &expected_taps[..]);
        assert_eq!(
            control.ltp_scale_q14,
            i32::from(
                SILK_LTPSCALES_TABLE_Q14
                    [usize::try_from(state.indices.ltp_scale_index).unwrap()]
            )
        );
    }

    #[test]
    fn bandwidth_expansion_reduces_predictor_magnitudes_after_loss() {
        let initial = base_state();
        let mut no_loss_state = initial.clone();
        let mut control_no_loss = DecoderControl::default();
        silk_decode_parameters(
            &mut no_loss_state,
            &mut control_no_loss,
            ConditionalCoding::Independent,
        );

        let mut loss_state = initial;
        loss_state.loss_count = 2;
        let mut control_loss = DecoderControl::default();
        silk_decode_parameters(
            &mut loss_state,
            &mut control_loss,
            ConditionalCoding::Independent,
        );

        let mut any_reduced = false;
        for row in 0..2 {
            for idx in 0..loss_state.lpc_order {
                let base = i32::from(control_no_loss.pred_coef_q12[row][idx]).abs();
                let after = i32::from(control_loss.pred_coef_q12[row][idx]).abs();
                assert!(after <= base);
                if after < base {
                    any_reduced = true;
                }
            }
        }
        assert!(any_reduced);
    }
}
