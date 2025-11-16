//! Port of `silk_process_gains_FIX`.
//!
//! This helper clamps the unquantised subframe gains, applies the LTP gain
//! reduction for voiced frames, encodes the gain indices, and tunes the
//! residual quantiser's rate/distortion lambda. The implementation mirrors the
//! fixed-point arithmetic in `silk/fixed/process_gains_FIX.c`.

use core::cmp::Ordering;

use crate::silk::decode_indices::ConditionalCoding;
use crate::silk::encoder::control::EncoderControl;
use crate::silk::encoder::state::EncoderChannelState;
use crate::silk::gain_quant::silk_gains_quant;
use crate::silk::log2lin::log2lin;
use crate::silk::sigm_q15::sigm_q15;
use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;
use crate::silk::{FrameQuantizationOffsetType, FrameSignalType, MAX_NB_SUBFR};

const LTP_SIGMOID_OFFSET_Q7: i32 = 12 << 7;
const INV_MAX_SQR_BASE_Q7: i32 = 8894; // SILK_FIX_CONST(21 + 16 / 0.33, 7)
const INV_MAX_SQR_EXP_Q16: i32 = 21_627; // SILK_FIX_CONST(0.33, 16)
const ONE_Q7: i32 = 1 << 7;
const LAMBDA_OFFSET_Q10: i32 = 1_229; // SILK_FIX_CONST(LAMBDA_OFFSET, 10)
const LAMBDA_DELAYED_DECISIONS_Q10: i32 = -50; // SILK_FIX_CONST(LAMBDA_DELAYED_DECISIONS, 10)
const LAMBDA_SPEECH_ACT_Q18: i32 = -52_428; // SILK_FIX_CONST(LAMBDA_SPEECH_ACT, 18)
const LAMBDA_INPUT_QUALITY_Q12: i32 = -409; // SILK_FIX_CONST(LAMBDA_INPUT_QUALITY, 12)
const LAMBDA_CODING_QUALITY_Q12: i32 = -818; // SILK_FIX_CONST(LAMBDA_CODING_QUALITY, 12)
const LAMBDA_QUANT_OFFSET_Q16: i32 = 52_429; // SILK_FIX_CONST(LAMBDA_QUANT_OFFSET, 16)

/// Mirrors `silk_process_gains_FIX`.
pub fn process_gains(
    encoder: &mut EncoderChannelState,
    control: &mut EncoderControl,
    cond_coding: ConditionalCoding,
) {
    let nb_subfr = encoder.common.nb_subfr;
    assert!(
        nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2,
        "encoder supports 2 or 4 subframes"
    );

    if matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced) {
        let diff_q7 = control.lt_pred_cod_gain_q7 - LTP_SIGMOID_OFFSET_Q7;
        let scaled_q5 = rshift_round(diff_q7, 4);
        let reduction_q16 = -sigm_q15(scaled_q5);
        for gain in control.gains_q16.iter_mut().take(nb_subfr) {
            *gain = smlawb(*gain, *gain, reduction_q16);
        }
    }

    let subfr_length = encoder.common.subfr_length as i32;
    let log_arg_q7 = smulwb(
        INV_MAX_SQR_BASE_Q7 - encoder.common.snr_db_q7,
        INV_MAX_SQR_EXP_Q16,
    );
    let inv_max_sqr_val_q16 = if subfr_length > 0 {
        div32_16(log2lin(log_arg_q7), subfr_length)
    } else {
        0
    };

    for k in 0..nb_subfr {
        let mut res_nrg_part = smulww(control.res_nrg[k], inv_max_sqr_val_q16);
        let res_q = control.res_nrg_q[k];
        if res_q > 0 {
            res_nrg_part = rshift_round(res_nrg_part, res_q);
        } else if res_q < 0 {
            let shift = -res_q;
            let limit = i32::MAX >> shift;
            if res_nrg_part >= limit {
                res_nrg_part = i32::MAX;
            } else {
                res_nrg_part = res_nrg_part.wrapping_shl(shift as u32);
            }
        }

        let gain = control.gains_q16[k];
        let gain_squared = add_sat32(res_nrg_part, smmul(gain, gain));
        if gain_squared < i32::from(i16::MAX) {
            let precise = smla_ww(res_nrg_part.wrapping_shl(16), gain, gain);
            debug_assert!(precise > 0, "gain clamp expects positive precision");
            let root = sqrt_approx(precise);
            let clamped = root.min(i32::MAX >> 8);
            control.gains_q16[k] = lshift_sat32(clamped, 8);
        } else {
            let root = sqrt_approx(gain_squared);
            let clamped = root.min(i32::MAX >> 16);
            control.gains_q16[k] = lshift_sat32(clamped, 16);
        }
    }

    control.gains_unq_q16[..nb_subfr].copy_from_slice(&control.gains_q16[..nb_subfr]);
    control.last_gain_index_prev = encoder.shape_state.last_gain_index as i8;

    let conditional = matches!(cond_coding, ConditionalCoding::Conditional);
    let mut last_gain_index = control.last_gain_index_prev;
    silk_gains_quant(
        &mut encoder.common.indices.gains_indices[..nb_subfr],
        &mut control.gains_q16[..nb_subfr],
        &mut last_gain_index,
        conditional,
    );
    encoder.shape_state.last_gain_index = i32::from(last_gain_index);

    if matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced) {
        let combined = control.lt_pred_cod_gain_q7 + (encoder.common.input_tilt_q15 >> 8);
        encoder.common.indices.quant_offset_type = if combined > ONE_Q7 {
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
    let quant_offset_q10 = i32::from(SILK_QUANTIZATION_OFFSETS_Q10[signal_row][quant_col]);

    control.lambda_q10 = LAMBDA_OFFSET_Q10
        + smulbb(
            LAMBDA_DELAYED_DECISIONS_Q10,
            encoder.common.n_states_delayed_decision,
        )
        + smulwb(LAMBDA_SPEECH_ACT_Q18, encoder.common.speech_activity_q8)
        + smulwb(LAMBDA_INPUT_QUALITY_Q12, control.input_quality_q14)
        + smulwb(LAMBDA_CODING_QUALITY_Q12, control.coding_quality_q14)
        + smulwb(LAMBDA_QUANT_OFFSET_Q16, quant_offset_q10);

    debug_assert!(
        control.lambda_q10 > 0 && control.lambda_q10 < (2 << 10),
        "lambda must stay within the fixed-point range"
    );
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    debug_assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

fn smla_ww(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulww(b, c))
}

fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

fn div32_16(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}

fn add_sat32(a: i32, b: i32) -> i32 {
    let sum = i64::from(a) + i64::from(b);
    if sum > i64::from(i32::MAX) {
        i32::MAX
    } else if sum < i64::from(i32::MIN) {
        i32::MIN
    } else {
        sum as i32
    }
}

fn lshift_sat32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return value;
    }
    if shift >= 31 {
        return match value.cmp(&0) {
            Ordering::Greater => i32::MAX,
            Ordering::Less => i32::MIN,
            Ordering::Equal => 0,
        };
    }
    let max_val = i32::MAX >> shift;
    let min_val = i32::MIN >> shift;
    if value > max_val {
        i32::MAX
    } else if value < min_val {
        i32::MIN
    } else {
        value << shift
    }
}

fn smulbb(a: i32, b: i32) -> i32 {
    let lhs = i32::from(a as i16);
    let rhs = i32::from(b as i16);
    lhs * rhs
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwb(b, c))
}

fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let leading = x.leading_zeros() as i32;
    let frac = ((x as u32).rotate_right(((24 - leading) & 31) as u32) & 0x7f) as i32;
    let mut y = if leading & 1 != 0 { 32_768 } else { 46_214 };
    y >>= leading >> 1;
    smlawb(y, y, smulbb(213, frac))
}

#[cfg(test)]
mod tests {
    use super::process_gains;
    use crate::silk::decode_indices::{ConditionalCoding, SideInfoIndices};
    use crate::silk::encoder::control::EncoderControl;
    use crate::silk::encoder::state::{EncoderChannelState, EncoderStateCommon};
    use crate::silk::{FrameQuantizationOffsetType, FrameSignalType, MAX_NB_SUBFR};

    fn encoder_state(signal_type: FrameSignalType) -> EncoderChannelState {
        let mut state = EncoderChannelState::default();
        state.common.nb_subfr = MAX_NB_SUBFR;
        state.common.subfr_length = 40;
        state.common.snr_db_q7 = 2_560;
        state.common.indices = SideInfoIndices {
            signal_type,
            quant_offset_type: FrameQuantizationOffsetType::Low,
            ..SideInfoIndices::default()
        };
        state
    }

    #[test]
    fn voiced_frames_reduce_gains_when_ltp_gain_is_high() {
        let mut encoder = encoder_state(FrameSignalType::Voiced);
        encoder.common.input_tilt_q15 = 0;

        let mut control = EncoderControl::default();
        control.gains_q16 = [1 << 16; MAX_NB_SUBFR];
        control.lt_pred_cod_gain_q7 = 3_200;

        process_gains(&mut encoder, &mut control, ConditionalCoding::Independent);
        for gain in control.gains_unq_q16.iter() {
            assert!(
                *gain < 1 << 16,
                "voiced reduction should act on unquantised gains"
            );
        }

        assert_eq!(
            encoder.common.indices.quant_offset_type,
            FrameQuantizationOffsetType::Low
        );
    }

    #[test]
    fn unvoiced_frames_keep_high_quant_offset_when_ltp_gain_is_low() {
        let mut encoder = encoder_state(FrameSignalType::Voiced);
        encoder.common.input_tilt_q15 = 1 << 10;

        let mut control = EncoderControl::default();
        control.gains_q16 = [90_000; MAX_NB_SUBFR];
        control.lt_pred_cod_gain_q7 = 50;

        process_gains(&mut encoder, &mut control, ConditionalCoding::Independent);
        assert_eq!(
            encoder.common.indices.quant_offset_type,
            FrameQuantizationOffsetType::High
        );
    }

    #[test]
    fn lambda_tracks_quality_and_speech_activity() {
        let mut encoder = EncoderChannelState::default();
        encoder.common = EncoderStateCommon {
            snr_db_q7: 0,
            subfr_length: 40,
            nb_subfr: MAX_NB_SUBFR,
            n_states_delayed_decision: 2,
            speech_activity_q8: 128,
            input_tilt_q15: 0,
            indices: SideInfoIndices::default(),
            ..EncoderStateCommon::default()
        };

        let mut control = EncoderControl::default();
        control.gains_q16 = [65_536; MAX_NB_SUBFR];
        control.res_nrg = [10_000; MAX_NB_SUBFR];
        control.res_nrg_q = [0; MAX_NB_SUBFR];
        control.input_quality_q14 = 1 << 13;
        control.coding_quality_q14 = 1 << 13;

        process_gains(&mut encoder, &mut control, ConditionalCoding::Conditional);

        assert!(control.lambda_q10 < 1 << 11);
        assert!(encoder.shape_state.last_gain_index != 0);
        for gain in control.gains_q16.iter() {
            assert!(*gain >= 65_536);
        }
    }
}
