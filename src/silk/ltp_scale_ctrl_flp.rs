//! Port of `silk/float/LTP_scale_ctrl_FLP.c`.
//!
//! Selects the long-term prediction (LTP) scaling factor for the floating-point
//! encoder path. When conditional coding is disabled, the helper raises the
//! LTP scale index based on the predicted coding gain, packet-loss heuristics,
//! and the current SNR target; otherwise it falls back to the default scale.

use crate::silk::decode_indices::{ConditionalCoding, SideInfoIndices};
use crate::silk::log2lin::log2lin;
use crate::silk::ltp_scale_ctrl::LtpScaleCtrlParams;
use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;

/// Mirrors `silk_LTP_scale_ctrl_FLP`.
///
/// Returns the floating-point LTP scaling factor while updating
/// `indices.ltp_scale_index` with the entropy-coded scale selection.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn ltp_scale_ctrl_flp(
    params: &LtpScaleCtrlParams,
    indices: &mut SideInfoIndices,
    cond_coding: ConditionalCoding,
    lt_pred_cod_gain: f32,
) -> f32 {
    let mut scale_index = 0;

    if matches!(cond_coding, ConditionalCoding::Independent) {
        let frames_per_packet =
            i32::try_from(params.frames_per_packet).expect("frames per packet fits in i32");
        debug_assert!(frames_per_packet > 0, "frames per packet must be positive");

        let mut round_loss = params.packet_loss_perc.saturating_mul(frames_per_packet);
        if params.lbrr_enabled {
            // LBRR reduces the effective loss rate by roughly squaring the percentage.
            let squared = round_loss.saturating_mul(round_loss);
            round_loss = 2 + squared / 100;
        }

        let gain_weight = i32::from(lt_pred_cod_gain as i16).saturating_mul(round_loss);
        let threshold0 = log2lin(2900 - params.snr_db_q7);
        if gain_weight > threshold0 {
            scale_index += 1;
        }
        let threshold1 = log2lin(3900 - params.snr_db_q7);
        if gain_weight > threshold1 {
            scale_index += 1;
        }
    }

    indices.ltp_scale_index = scale_index as i8;
    f32::from(SILK_LTPSCALES_TABLE_Q14[scale_index as usize]) * (1.0 / 16384.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_scale(index: usize) -> f32 {
        f32::from(SILK_LTPSCALES_TABLE_Q14[index]) * (1.0 / 16384.0)
    }

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

        let scale =
            ltp_scale_ctrl_flp(&params, &mut indices, ConditionalCoding::Conditional, 120.0);

        assert_eq!(indices.ltp_scale_index, 0);
        assert!((scale - expected_scale(0)).abs() < 1e-6);
    }

    #[test]
    fn independent_sets_mid_scale_when_only_first_threshold_trips() {
        let mut params = params();
        params.packet_loss_perc = 5;
        params.snr_db_q7 = 3000;
        let mut indices = SideInfoIndices::default();

        let scale = ltp_scale_ctrl_flp(&params, &mut indices, ConditionalCoding::Independent, 10.4);

        assert_eq!(indices.ltp_scale_index, 1);
        assert!((scale - expected_scale(1)).abs() < 1e-6);
    }

    #[test]
    fn high_snr_and_loss_push_lowest_scale() {
        let mut params = params();
        params.packet_loss_perc = 10;
        params.frames_per_packet = 2;
        params.snr_db_q7 = 5000;
        let mut indices = SideInfoIndices::default();

        let scale = ltp_scale_ctrl_flp(&params, &mut indices, ConditionalCoding::Independent, 5.0);

        assert_eq!(indices.ltp_scale_index, 2);
        assert!((scale - expected_scale(2)).abs() < 1e-6);
    }
}
