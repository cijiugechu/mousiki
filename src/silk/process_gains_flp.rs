//! Port of `silk_process_gains_FLP`.
//!
//! Mirrors the floating-point gain processing helper from
//! `silk/float/process_gains_FLP.c`, applying the LTP-dependent gain reduction,
//! limiting the subframe gains, quantising them, and updating the lambda and
//! quantisation-offset bookkeeping for the FLP encoder path.

use crate::silk::decode_indices::ConditionalCoding;
use crate::silk::encoder::control_flp::EncoderControlFlp;
use crate::silk::encoder::state_flp::EncoderStateFlp;
use crate::silk::gain_quant::silk_gains_quant;
use crate::silk::sigproc_flp::{silk_min_float, silk_sigmoid};
use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;
use crate::silk::tuning_parameters::{
    LAMBDA_CODING_QUALITY, LAMBDA_DELAYED_DECISIONS, LAMBDA_INPUT_QUALITY, LAMBDA_OFFSET,
    LAMBDA_QUANT_OFFSET, LAMBDA_SPEECH_ACT,
};
use crate::silk::{FrameQuantizationOffsetType, FrameSignalType, MAX_NB_SUBFR};
use libm::{powf, sqrtf};

const Q16_SCALE: f32 = 65_536.0;

/// Mirrors `silk_process_gains_FLP`.
pub fn process_gains_flp(
    encoder: &mut EncoderStateFlp,
    control: &mut EncoderControlFlp,
    cond_coding: ConditionalCoding,
) {
    let nb_subfr = encoder.common.nb_subfr;
    assert!(
        nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2,
        "encoder supports 2 or 4 subframes"
    );

    if matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced) {
        let reduction =
            1.0_f32 - 0.5_f32 * silk_sigmoid(0.25_f32 * (control.lt_pred_cod_gain - 12.0_f32));
        for gain in control.gains.iter_mut().take(nb_subfr) {
            *gain *= reduction;
        }
    }

    let subfr_length = encoder.common.subfr_length as f32;
    assert!(subfr_length > 0.0, "subframe length must be positive");

    let inv_max_sqr_val = powf(
        2.0,
        0.33 * (21.0 - (encoder.common.snr_db_q7 as f32) * (1.0 / 128.0)),
    ) / subfr_length;

    let res_nrg = control.res_nrg;
    for (k, gain) in control.gains.iter_mut().take(nb_subfr).enumerate() {
        let adjusted = sqrtf(*gain * *gain + res_nrg[k] * inv_max_sqr_val);
        *gain = silk_min_float(adjusted, 32_767.0);
    }

    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    for (dst, &gain) in gains_q16
        .iter_mut()
        .zip(control.gains.iter().take(nb_subfr))
    {
        *dst = (gain * Q16_SCALE) as i32;
    }

    control.gains_unq_q16[..nb_subfr].copy_from_slice(&gains_q16[..nb_subfr]);
    control.last_gain_index_prev = encoder.shape_state.last_gain_index;

    let conditional = matches!(cond_coding, ConditionalCoding::Conditional);
    let mut last_gain_index = control.last_gain_index_prev;
    silk_gains_quant(
        &mut encoder.common.indices.gains_indices[..nb_subfr],
        &mut gains_q16[..nb_subfr],
        &mut last_gain_index,
        conditional,
    );
    encoder.shape_state.last_gain_index = last_gain_index;

    for (gain, &quant_q16) in control
        .gains
        .iter_mut()
        .zip(gains_q16.iter())
        .take(nb_subfr)
    {
        *gain = quant_q16 as f32 / Q16_SCALE;
    }

    if matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced) {
        let combined =
            control.lt_pred_cod_gain + (encoder.common.input_tilt_q15 as f32) * (1.0 / 32_768.0);
        encoder.common.indices.quant_offset_type = if combined > 1.0 {
            FrameQuantizationOffsetType::Low
        } else {
            FrameQuantizationOffsetType::High
        };
    }

    let signal_row = (i32::from(encoder.common.indices.signal_type) >> 1) as usize;
    let quant_col = match encoder.common.indices.quant_offset_type {
        FrameQuantizationOffsetType::Low => 0,
        FrameQuantizationOffsetType::High => 1,
    };
    let quant_offset =
        f32::from(SILK_QUANTIZATION_OFFSETS_Q10[signal_row][quant_col]) * (1.0 / 1024.0);

    control.lambda = LAMBDA_OFFSET
        + LAMBDA_DELAYED_DECISIONS * encoder.common.n_states_delayed_decision as f32
        + LAMBDA_SPEECH_ACT * encoder.common.speech_activity_q8 as f32 * (1.0 / 256.0)
        + LAMBDA_INPUT_QUALITY * control.input_quality
        + LAMBDA_CODING_QUALITY * control.coding_quality
        + LAMBDA_QUANT_OFFSET * quant_offset;

    debug_assert!(
        (0.0..2.0).contains(&control.lambda),
        "lambda must remain inside the reference range"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_gains_flp_updates_indices_and_lambda() {
        let mut encoder = EncoderStateFlp::default();
        encoder.common.nb_subfr = 2;
        encoder.common.indices.signal_type = FrameSignalType::Voiced;
        encoder.common.n_states_delayed_decision = 1;
        encoder.common.speech_activity_q8 = 128;
        encoder.common.input_tilt_q15 = 5_000;

        let mut control = EncoderControlFlp::default();
        control.gains[0] = 2.0;
        control.gains[1] = 3.0;
        control.lt_pred_cod_gain = 0.5;
        control.res_nrg[0] = 0.0;
        control.res_nrg[1] = 0.0;
        control.input_quality = 0.4;
        control.coding_quality = 0.2;

        let nb_subfr = encoder.common.nb_subfr;
        let reduction = 1.0 - 0.5 * silk_sigmoid(0.25 * (control.lt_pred_cod_gain - 12.0));
        let mut expected_gains = control.gains;
        for gain in expected_gains.iter_mut().take(nb_subfr) {
            *gain *= reduction;
        }

        let inv_max_sqr_val = powf(
            2.0,
            0.33 * (21.0 - (encoder.common.snr_db_q7 as f32) * (1.0 / 128.0)),
        ) / encoder.common.subfr_length as f32;
        for (k, gain) in expected_gains.iter_mut().take(nb_subfr).enumerate() {
            let clamped = sqrtf(*gain * *gain + control.res_nrg[k] * inv_max_sqr_val);
            *gain = silk_min_float(clamped, 32_767.0);
        }

        let mut expected_unquant_q16 = [0i32; MAX_NB_SUBFR];
        for (dst, &gain) in expected_unquant_q16
            .iter_mut()
            .zip(expected_gains.iter().take(nb_subfr))
        {
            *dst = (gain * Q16_SCALE) as i32;
        }
        let mut expected_quant_q16 = expected_unquant_q16;
        let mut expected_indices = encoder.common.indices.gains_indices;
        let mut expected_last = encoder.shape_state.last_gain_index;
        silk_gains_quant(
            &mut expected_indices[..nb_subfr],
            &mut expected_quant_q16[..nb_subfr],
            &mut expected_last,
            false,
        );

        let expected_quant_offset = {
            let combined = control.lt_pred_cod_gain
                + (encoder.common.input_tilt_q15 as f32) * (1.0 / 32_768.0);
            let quant_offset_type = if combined > 1.0 {
                FrameQuantizationOffsetType::Low
            } else {
                FrameQuantizationOffsetType::High
            };
            let quant_col = match quant_offset_type {
                FrameQuantizationOffsetType::Low => 0,
                FrameQuantizationOffsetType::High => 1,
            };
            let row = (i32::from(encoder.common.indices.signal_type) >> 1) as usize;
            f32::from(SILK_QUANTIZATION_OFFSETS_Q10[row][quant_col]) * (1.0 / 1024.0)
        };
        let expected_lambda = LAMBDA_OFFSET
            + LAMBDA_DELAYED_DECISIONS * encoder.common.n_states_delayed_decision as f32
            + LAMBDA_SPEECH_ACT * encoder.common.speech_activity_q8 as f32 * (1.0 / 256.0)
            + LAMBDA_INPUT_QUALITY * control.input_quality
            + LAMBDA_CODING_QUALITY * control.coding_quality
            + LAMBDA_QUANT_OFFSET * expected_quant_offset;

        process_gains_flp(&mut encoder, &mut control, ConditionalCoding::Independent);

        assert_eq!(
            control.gains_unq_q16[..nb_subfr],
            expected_unquant_q16[..nb_subfr]
        );
        assert_eq!(
            encoder.common.indices.gains_indices[..nb_subfr],
            expected_indices[..nb_subfr]
        );
        assert_eq!(encoder.shape_state.last_gain_index, expected_last);
        for (actual, expected) in control
            .gains
            .iter()
            .take(nb_subfr)
            .zip(expected_quant_q16.iter())
        {
            assert!((actual - (*expected as f32) / Q16_SCALE).abs() < 1e-6);
        }
        assert_eq!(
            encoder.common.indices.quant_offset_type,
            FrameQuantizationOffsetType::High
        );
        assert!((control.lambda - expected_lambda).abs() < 1e-6);
    }
}
