//! Port of `silk/decode_core.c`.
//!
//! Mirrors the inverse noise-shaping quantiser (NSQ) stage that synthesises
//! time-domain LPC output from the decoded excitation pulses, LTP coefficients,
//! and predictor gains. The implementation follows the reference fixed-point
//! arithmetic exactly so downstream PLC and CNG helpers observe the same state.

use core::convert::TryFrom;

use crate::silk::cng::{silk_rand, DecoderControl};
use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::decoder_set_fs::{MAX_FRAME_LENGTH, MAX_SUB_FRAME_LENGTH};
use crate::silk::decoder_state::DecoderState;
use crate::silk::lpc_analysis_filter::lpc_analysis_filter;
use crate::silk::lpc_inv_pred_gain::inverse32_varq;
use crate::silk::stereo_find_predictor::div32_varq;
use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{
    FrameQuantizationOffsetType, FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR,
};

const MAX_LTP_MEM_LENGTH: usize = 4 * MAX_SUB_FRAME_LENGTH;
const UNITY_Q16: i32 = 1 << 16;
const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
const VOICED_TRANSITION_Q14: i16 = 4096; // SILK_FIX_CONST(0.25, 14)

/// Mirrors `silk_decode_core`: reconstructs the LPC output for a single SILK frame.
pub fn silk_decode_core(
    state: &mut DecoderState,
    control: &mut DecoderControl,
    output: &mut [i16],
    pulses: &[i16],
    arch: i32,
) {
    let (frame_length, subfr_length, nb_subfr, ltp_mem_length, lpc_order, prev_signal_type, lag_prev) =
        {
            let sr = &state.sample_rate;
            (
                sr.frame_length,
                sr.subfr_length,
                sr.nb_subfr,
                sr.ltp_mem_length,
                sr.lpc_order,
                sr.prev_signal_type,
                sr.lag_prev,
            )
        };

    assert!(
        frame_length > 0 && frame_length <= MAX_FRAME_LENGTH,
        "decoder frame length {} out of range",
        frame_length
    );
    assert!(
        output.len() >= frame_length,
        "output buffer must hold {} samples",
        frame_length
    );
    assert!(
        pulses.len() >= frame_length,
        "pulse buffer must hold {} samples",
        frame_length
    );
    assert!(
        nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2,
        "unsupported subframe count {nb_subfr}"
    );
    assert!(
        (1..=MAX_LPC_ORDER).contains(&lpc_order),
        "invalid LPC order {lpc_order}"
    );
    assert!(
        subfr_length > 0 && subfr_length <= MAX_SUB_FRAME_LENGTH,
        "invalid subframe length {subfr_length}"
    );
    assert!(
        ltp_mem_length > 0 && ltp_mem_length <= MAX_LTP_MEM_LENGTH,
        "invalid LTP memory length {ltp_mem_length}"
    );
    debug_assert!(state.prev_gain_q16 != 0);
    let _ = arch;

    let mut s_ltp = [0i16; MAX_LTP_MEM_LENGTH];
    let mut s_ltp_q15 = [0i32; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH];
    let mut res_q14 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut s_lpc_q14 = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&state.sample_rate.s_lpc_q14_buf);

    let quant_offset_q10 = quantization_offset_q10(&state.indices);
    let nlsf_interp_flag = state.indices.nlsf_interp_coef_q2 < 4;

    let mut rand_seed = i32::from(state.indices.seed);
    for (i, &pulse_q0) in pulses.iter().enumerate().take(frame_length) {
        rand_seed = silk_rand(rand_seed);
        let pulse = i32::from(pulse_q0);
        let mut sample = pulse << 14;
        if sample > 0 {
            sample -= QUANT_LEVEL_ADJUST_Q10 << 4;
        } else if sample < 0 {
            sample += QUANT_LEVEL_ADJUST_Q10 << 4;
        }
        sample += quant_offset_q10 << 4;
        if rand_seed < 0 {
            sample = -sample;
        }
        state.exc_q14[i] = sample;
        rand_seed = rand_seed.wrapping_add(pulse);
    }

    let mut pexc_idx = 0usize;
    let mut pxq_idx = 0usize;
    let mut s_ltp_buf_idx = ltp_mem_length;

    for k in 0..nb_subfr {
        assert!(
            pexc_idx + subfr_length <= frame_length,
            "excitation index overrun"
        );
        assert!(
            pxq_idx + subfr_length <= frame_length,
            "output index overrun"
        );

        let predictor_row = k >> 1;
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        a_q12[..lpc_order]
            .copy_from_slice(&control.pred_coef_q12[predictor_row][..lpc_order]);

        let ltp_offset = k * LTP_ORDER;
        let (b_start, b_end) = (ltp_offset, ltp_offset + LTP_ORDER);
        let b_q14 = &mut control.ltp_coef_q14[b_start..b_end];

        let gain_q16 = control.gains_q16[k];
        assert!(gain_q16 > 0, "decoder gain must be positive");
        let gain_q10 = gain_q16 >> 6;
        let mut inv_gain_q31 = inverse32_varq(gain_q16, 47);
        assert!(inv_gain_q31 != 0, "inverse gain must be non-zero");

        let same_gain = gain_q16 == state.prev_gain_q16;
        let gain_adj_q16 = if same_gain {
            UNITY_Q16
        } else {
            div32_varq(state.prev_gain_q16, gain_q16, 16)
        };
        if !same_gain {
            for value in &mut s_lpc_q14[..MAX_LPC_ORDER] {
                *value = smulww(gain_adj_q16, *value);
            }
        }
        state.prev_gain_q16 = gain_q16;

        let mut signal_type = state.indices.signal_type;
        if state.loss_count > 0
            && matches!(prev_signal_type, FrameSignalType::Voiced)
            && !matches!(state.indices.signal_type, FrameSignalType::Voiced)
            && k < MAX_NB_SUBFR / 2
        {
            b_q14.fill(0);
            b_q14[LTP_ORDER / 2] = VOICED_TRANSITION_Q14;
            signal_type = FrameSignalType::Voiced;
            control.pitch_l[k] = clamp_to_i16(lag_prev);
        }

        let pres_q14_slice: &[i32];
        if matches!(signal_type, FrameSignalType::Voiced) {
            let lag = i32::from(control.pitch_l[k]);
            assert!(lag > 0, "voiced subframes require a positive pitch lag");
            let lag_usize = usize::try_from(lag).expect("pitch lag must fit usize");

            if k == 0 || (k == 2 && nlsf_interp_flag) {
                let start_idx = ltp_mem_length as isize
                    - lag as isize
                    - lpc_order as isize
                    - (LTP_ORDER / 2) as isize;
                assert!(start_idx > 0, "rewhitening start index must be positive");
                let start = start_idx as usize;

                if k == 2 {
                    let copy_len = 2 * subfr_length;
                    assert!(
                        ltp_mem_length + copy_len <= state.sample_rate.out_buf.len(),
                        "decoder out buffer too small"
                    );
                    state.sample_rate.out_buf[ltp_mem_length..ltp_mem_length + copy_len]
                        .copy_from_slice(&output[..copy_len]);
                }

                let len = ltp_mem_length - start;
                let input_offset = start + k * subfr_length;
                assert!(
                    input_offset + len <= state.sample_rate.out_buf.len(),
                    "decoder history buffer too small"
                );

                lpc_analysis_filter(
                    &mut s_ltp[start..ltp_mem_length],
                    &state.sample_rate.out_buf[input_offset..input_offset + len],
                    &a_q12[..lpc_order],
                    len,
                    lpc_order,
                );

                if k == 0 {
                    inv_gain_q31 =
                        lshift(smulwb(inv_gain_q31, i32::from(control.ltp_scale_q14)), 2);
                }

                let span = lag_usize + LTP_ORDER / 2;
                assert!(
                    span <= ltp_mem_length,
                    "LTP span exceeds history ({span} > {ltp_mem_length})"
                );
                for i in 0..span {
                    let dest_idx = s_ltp_buf_idx - 1 - i;
                    let src_idx = ltp_mem_length - 1 - i;
                    s_ltp_q15[dest_idx] = smulwb(inv_gain_q31, i32::from(s_ltp[src_idx]));
                }
            } else if gain_adj_q16 != UNITY_Q16 {
                let span = lag_usize + LTP_ORDER / 2;
                assert!(
                    span <= s_ltp_buf_idx,
                    "LTP span exceeds current buffer ({span} > {s_ltp_buf_idx})"
                );
                for i in 0..span {
                    let dest_idx = s_ltp_buf_idx - 1 - i;
                    s_ltp_q15[dest_idx] = smulww(gain_adj_q16, s_ltp_q15[dest_idx]);
                }
            }

            let mut pred_lag_index = s_ltp_buf_idx
                .checked_sub(lag_usize)
                .expect("pitch lag must not exceed buffer index")
                + LTP_ORDER / 2;
            assert!(
                pred_lag_index >= LTP_ORDER,
                "insufficient history for LTP prediction"
            );
            assert!(
                s_ltp_buf_idx + subfr_length <= s_ltp_q15.len(),
                "LTP buffer overflow"
            );

            {
                let pres = &mut res_q14[..subfr_length];
                for (i, target) in pres.iter_mut().enumerate().take(subfr_length) {
                    let mut ltp_pred_q13 = 2;
                    ltp_pred_q13 =
                        smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_index], i32::from(b_q14[0]));
                    ltp_pred_q13 =
                        smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_index - 1], i32::from(b_q14[1]));
                    ltp_pred_q13 =
                        smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_index - 2], i32::from(b_q14[2]));
                    ltp_pred_q13 =
                        smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_index - 3], i32::from(b_q14[3]));
                    ltp_pred_q13 =
                        smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_index - 4], i32::from(b_q14[4]));
                    pred_lag_index += 1;

                    let value = add_lshift32(state.exc_q14[pexc_idx + i], ltp_pred_q13, 1);
                    *target = value;
                    s_ltp_q15[s_ltp_buf_idx] = value << 1;
                    s_ltp_buf_idx += 1;
                }
            }

            pres_q14_slice = &res_q14[..subfr_length];
        } else {
            pres_q14_slice = &state.exc_q14[pexc_idx..pexc_idx + subfr_length];
        }

        let pxq = &mut output[pxq_idx..pxq_idx + subfr_length];
        for (i, sample) in pres_q14_slice.iter().enumerate().take(subfr_length) {
            let mut lpc_pred_q10 = (lpc_order as i32) >> 1;
            for (tap, &coef) in a_q12.iter().enumerate().take(lpc_order) {
                let hist_idx = MAX_LPC_ORDER + i - 1 - tap;
                lpc_pred_q10 = smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[hist_idx],
                    i32::from(coef),
                );
            }

            let base = MAX_LPC_ORDER + i;
            s_lpc_q14[base] = add_sat32(*sample, lshift_sat32(lpc_pred_q10, 4));
            let scaled = smulww(s_lpc_q14[base], gain_q10);
            pxq[i] = sat16(rshift_round(scaled, 8));
        }

        s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
        pexc_idx += subfr_length;
        pxq_idx += subfr_length;
    }

    state
        .sample_rate
        .s_lpc_q14_buf
        .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
}

fn quantization_offset_q10(indices: &SideInfoIndices) -> i32 {
    let row = match indices.signal_type {
        FrameSignalType::Voiced => 1,
        _ => 0,
    };
    let col = match indices.quant_offset_type {
        FrameQuantizationOffsetType::Low => 0,
        FrameQuantizationOffsetType::High => 1,
    };
    i32::from(SILK_QUANTIZATION_OFFSETS_Q10[row][col])
}

#[inline]
fn clamp_to_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

#[inline]
fn add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b.wrapping_shl(shift as u32))
}

#[inline]
fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

#[inline]
fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

#[inline]
fn smlawb(acc: i32, x: i32, y: i32) -> i32 {
    acc.wrapping_add(smulwb(x, y))
}

#[inline]
fn add_sat32(a: i32, b: i32) -> i32 {
    (i64::from(a) + i64::from(b))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[inline]
fn lshift(value: i32, shift: i32) -> i32 {
    value.wrapping_shl(shift as u32)
}

#[inline]
fn lshift_sat32(value: i32, shift: i32) -> i32 {
    let shifted = i64::from(value) << shift;
    shifted.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[inline]
fn rshift_round(value: i32, shift: i32) -> i32 {
    debug_assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

#[inline]
fn sat16(value: i32) -> i16 {
    clamp_to_i16(value)
}
