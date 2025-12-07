//! Port of `silk/fixed/LTP_scale_ctrl_FIX.c`.
//!
//! This helper tunes the long-term prediction (LTP) state scaling used by the
//! encoder when conditional coding is disabled.  The reference implementation
//! derives an LTP scaling index from the predicted coding gain, the expected
//! packet loss across the packet, and the configured SNR target.  The result is
//! stored in the side-information indices and converted to the Q14 scaling used
//! by the rest of the encoder pipeline.

use crate::silk::decode_indices::{ConditionalCoding, SideInfoIndices};
use crate::silk::log2lin::log2lin;
use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;

/// Inputs needed by [`ltp_scale_ctrl`], split from the encoder state to avoid aliasing.
#[derive(Clone, Copy, Debug)]
pub struct LtpScaleCtrlParams {
    pub packet_loss_perc: i32,
    pub frames_per_packet: usize,
    pub lbrr_enabled: bool,
    pub snr_db_q7: i32,
}

/// Mirrors `silk_LTP_scale_ctrl_FIX`.
///
/// Returns the selected LTP scaling factor in Q14 while updating the
/// side-information indices with the matching entropy-coded index.  The
/// conditional coding mode determines whether LTP scaling is evaluated or
/// forced to the default (index zero) value.
pub fn ltp_scale_ctrl(
    params: &LtpScaleCtrlParams,
    indices: &mut SideInfoIndices,
    cond_coding: ConditionalCoding,
    lt_pred_cod_gain_q7: i32,
) -> i32 {
    let mut scale_index = 0;

    if matches!(cond_coding, ConditionalCoding::Independent) {
        let frames_per_packet = i32::try_from(params.frames_per_packet)
            .expect("frames per packet fits in i32");
        debug_assert!(frames_per_packet > 0, "frames per packet must be positive");

        let mut round_loss = params.packet_loss_perc.saturating_mul(frames_per_packet);
        if params.lbrr_enabled {
            // LBRR reduces the effective packet loss by roughly squaring the loss rate.
            let squared = round_loss.saturating_mul(round_loss);
            round_loss = 2 + squared / 100;
        }

        let gain_weight = lt_pred_cod_gain_q7.saturating_mul(round_loss);
        const BASE_Q7: i32 = 128 * 7;
        let threshold0 = log2lin(BASE_Q7 + 2900 - params.snr_db_q7);
        if gain_weight > threshold0 {
            scale_index += 1;
        }
        let threshold1 = log2lin(BASE_Q7 + 3900 - params.snr_db_q7);
        if gain_weight > threshold1 {
            scale_index += 1;
        }
    }

    indices.ltp_scale_index = scale_index as i8;
    i32::from(SILK_LTPSCALES_TABLE_Q14[scale_index as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> LtpScaleCtrlParams {
        LtpScaleCtrlParams {
            packet_loss_perc: 0,
            frames_per_packet: 1,
            lbrr_enabled: false,
            snr_db_q7: 0,
        }
    }

    #[test]
    fn conditional_coding_forces_default_scale() {
        let params = params();
        let mut indices = SideInfoIndices::default();
        indices.ltp_scale_index = 2;
        let scale = ltp_scale_ctrl(&params, &mut indices, ConditionalCoding::Conditional, 250);
        assert_eq!(indices.ltp_scale_index, 0);
        assert_eq!(scale, i32::from(SILK_LTPSCALES_TABLE_Q14[0]));

        let scale = ltp_scale_ctrl(
            &params,
            &mut indices,
            ConditionalCoding::IndependentNoLtpScaling,
            250,
        );
        assert_eq!(indices.ltp_scale_index, 0);
        assert_eq!(scale, i32::from(SILK_LTPSCALES_TABLE_Q14[0]));
    }

    #[test]
    fn low_loss_keeps_max_scale() {
        let mut params = params();
        params.packet_loss_perc = 5;
        params.frames_per_packet = 2;
        params.snr_db_q7 = 0;
        let mut indices = SideInfoIndices::default();

        let scale = ltp_scale_ctrl(&params, &mut indices, ConditionalCoding::Independent, 120);
        assert_eq!(indices.ltp_scale_index, 0);
        assert_eq!(scale, i32::from(SILK_LTPSCALES_TABLE_Q14[0]));
    }

    #[test]
    fn medium_gain_selects_mid_scale() {
        let mut params = params();
        params.packet_loss_perc = 25;
        params.frames_per_packet = 2;
        params.snr_db_q7 = 3000;
        let mut indices = SideInfoIndices::default();

        let scale = ltp_scale_ctrl(&params, &mut indices, ConditionalCoding::Independent, 100);
        assert_eq!(indices.ltp_scale_index, 1);
        assert_eq!(scale, i32::from(SILK_LTPSCALES_TABLE_Q14[1]));
    }

    #[test]
    fn lbrr_high_loss_triggers_min_scale() {
        let mut params = params();
        params.packet_loss_perc = 40;
        params.frames_per_packet = 3;
        params.lbrr_enabled = true;
        params.snr_db_q7 = 6000;
        let mut indices = SideInfoIndices::default();

        let scale = ltp_scale_ctrl(&params, &mut indices, ConditionalCoding::Independent, 500);
        assert_eq!(indices.ltp_scale_index, 2);
        assert_eq!(scale, i32::from(SILK_LTPSCALES_TABLE_Q14[2]));
    }
}
