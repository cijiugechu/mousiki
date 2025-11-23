#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use core::cmp::max;

use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::encoder::state::{
    EncoderStateCommon, MAX_FRAME_LENGTH, MAX_LTP_MEM_LENGTH, MAX_SUB_FRAME_LENGTH,
    NSQ_LPC_BUF_LENGTH, NoiseShapingQuantizerState,
};
use crate::silk::lpc_analysis_filter::lpc_analysis_filter;
use crate::silk::lpc_inv_pred_gain::inverse32_varq;
use crate::silk::stereo_find_predictor::div32_varq;
use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{
    FrameQuantizationOffsetType, FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER,
};

pub(crate) const HARM_SHAPE_FIR_TAPS: usize = 3;
pub(crate) const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
pub(crate) const RAND_MULTIPLIER: i32 = 196_314_165;
pub(crate) const RAND_INCREMENT: i32 = 907_633_515;

/// Fixed-point noise-shaping quantiser (`silk_NSQ`).
#[allow(clippy::too_many_lines)]
pub fn silk_nsq(
    encoder: &EncoderStateCommon,
    nsq: &mut NoiseShapingQuantizerState,
    indices: &SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_shp_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    assert_eq!(encoder.nb_subfr, pitch_l.len());
    assert_eq!(encoder.nb_subfr, harm_shape_gain_q14.len());
    assert_eq!(encoder.nb_subfr, tilt_q14.len());
    assert_eq!(encoder.nb_subfr, lf_shp_q14.len());
    assert_eq!(encoder.nb_subfr, gains_q16.len());
    assert!(encoder.nb_subfr == MAX_NB_SUBFR || encoder.nb_subfr == MAX_NB_SUBFR / 2);
    assert_eq!(encoder.frame_length, x16.len());
    assert_eq!(encoder.frame_length, pulses.len());
    assert!(encoder.predict_lpc_order <= MAX_LPC_ORDER);
    assert!(encoder.shaping_lpc_order as usize <= MAX_SHAPE_LPC_ORDER);
    assert!(encoder.subfr_length <= MAX_SUB_FRAME_LENGTH);

    let frame_length = encoder.frame_length;
    let subfr_length = encoder.subfr_length;
    let ltp_mem_length = encoder.ltp_mem_length;
    let shaping_lpc_order = encoder.shaping_lpc_order as usize;
    assert!(shaping_lpc_order.is_multiple_of(2));

    let ltp_buffer_len = ltp_mem_length + frame_length;
    assert!(ltp_buffer_len <= MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH);

    let mut s_ltp_q15 = [0i32; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH];
    let mut s_ltp = [0i16; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH];
    let mut x_sc_q10 = [0i32; MAX_SUB_FRAME_LENGTH];

    nsq.rand_seed = i32::from(indices.seed);

    let mut lag = nsq.lag_prev;
    let offset_q10 = quantization_offset(indices.signal_type, indices.quant_offset_type);
    let lsf_interpolation_flag = if indices.nlsf_interp_coef_q2 == 4 {
        0
    } else {
        1
    };

    nsq.s_ltp_shp_buf_idx = ltp_mem_length;
    nsq.s_ltp_buf_idx = ltp_mem_length;

    for k in 0..encoder.nb_subfr {
        let frame_offset = k * subfr_length;
        let x_subframe = &x16[frame_offset..frame_offset + subfr_length];
        let pulses_sub = &mut pulses[frame_offset..frame_offset + subfr_length];

        let pred_coef_offset = ((k >> 1) | (1 - lsf_interpolation_flag)) * MAX_LPC_ORDER;
        assert!(pred_coef_offset + encoder.predict_lpc_order <= pred_coef_q12.len());
        let a_q12 = &pred_coef_q12[pred_coef_offset..pred_coef_offset + encoder.predict_lpc_order];

        let b_q14_offset = k * LTP_ORDER;
        assert!(b_q14_offset + LTP_ORDER <= ltp_coef_q14.len());
        let b_q14 = &ltp_coef_q14[b_q14_offset..b_q14_offset + LTP_ORDER];

        let ar_shp_offset = k * MAX_SHAPE_LPC_ORDER;
        assert!(ar_shp_offset + shaping_lpc_order <= ar_shp_q13.len());
        let ar_shp = &ar_shp_q13[ar_shp_offset..ar_shp_offset + shaping_lpc_order];

        nsq.rewhite_flag = false;
        if indices.signal_type == FrameSignalType::Voiced {
            lag = pitch_l[k];

            if (k & (3 - (lsf_interpolation_flag << 1))) == 0 {
                let start_idx = ltp_mem_length as i32
                    - lag
                    - encoder.predict_lpc_order as i32
                    - (LTP_ORDER as i32 / 2);
                assert!(start_idx > 0);
                let start = start_idx as usize;
                assert!(start + frame_offset + ltp_mem_length - start <= nsq.xq.len());
                lpc_analysis_filter(
                    &mut s_ltp[start..start + ltp_mem_length - start],
                    &nsq.xq[start + frame_offset..start + frame_offset + ltp_mem_length - start],
                    a_q12,
                    ltp_mem_length - start,
                    encoder.predict_lpc_order,
                );
                nsq.rewhite_flag = true;
                nsq.s_ltp_buf_idx = ltp_mem_length;
            }
        }

        nsq_scale_states(
            encoder,
            nsq,
            x_subframe,
            &mut x_sc_q10[..subfr_length],
            &s_ltp,
            &mut s_ltp_q15,
            k,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            indices.signal_type,
        );

        silk_noise_shape_quantizer(
            nsq,
            indices.signal_type,
            &x_sc_q10[..subfr_length],
            pulses_sub,
            ltp_mem_length + frame_offset,
            &mut s_ltp_q15,
            a_q12,
            b_q14,
            ar_shp,
            lag,
            harm_shape_gain_q14[k],
            tilt_q14[k],
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            subfr_length,
            shaping_lpc_order,
            encoder.predict_lpc_order,
        );
    }

    nsq.lag_prev = pitch_l[encoder.nb_subfr - 1];

    nsq.xq
        .copy_within(frame_length..frame_length + ltp_mem_length, 0);
    nsq.s_ltp_shp_q14
        .copy_within(frame_length..frame_length + ltp_mem_length, 0);
}

fn nsq_scale_states(
    encoder: &EncoderStateCommon,
    nsq: &mut NoiseShapingQuantizerState,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: FrameSignalType,
) {
    let lag = pitch_l[subfr];
    assert!(lag >= 0);
    let lag = lag as usize;

    let mut inv_gain_q31 = inverse32_varq(max(gains_q16[subfr], 1), 47);
    assert_ne!(inv_gain_q31, 0);

    let inv_gain_q26 = rshift_round(inv_gain_q31, 5);
    for (scaled, &sample) in x_sc_q10.iter_mut().zip(x16.iter()) {
        *scaled = smulww(i32::from(sample), inv_gain_q26);
    }

    if nsq.rewhite_flag {
        if subfr == 0 {
            inv_gain_q31 = smulwb(inv_gain_q31, ltp_scale_q14).wrapping_shl(2);
        }
        assert!(nsq.s_ltp_buf_idx >= lag + (LTP_ORDER / 2));
        let start = nsq.s_ltp_buf_idx - lag - (LTP_ORDER / 2);
        assert!(nsq.s_ltp_buf_idx <= s_ltp_q15.len());
        for i in start..nsq.s_ltp_buf_idx {
            assert!(i < MAX_FRAME_LENGTH);
            s_ltp_q15[i] = smulwb(inv_gain_q31, i32::from(s_ltp[i]));
        }
    }

    if gains_q16[subfr] != nsq.prev_gain_q16 {
        let gain_adj_q16 = div32_varq(nsq.prev_gain_q16, gains_q16[subfr], 16);

        assert!(nsq.s_ltp_shp_buf_idx >= encoder.ltp_mem_length);
        let start = nsq.s_ltp_shp_buf_idx - encoder.ltp_mem_length;
        for value in &mut nsq.s_ltp_shp_q14[start..nsq.s_ltp_shp_buf_idx] {
            *value = smulww(gain_adj_q16, *value);
        }

        if signal_type == FrameSignalType::Voiced && !nsq.rewhite_flag {
            assert!(nsq.s_ltp_buf_idx >= lag + (LTP_ORDER / 2));
            let start = nsq.s_ltp_buf_idx - lag - (LTP_ORDER / 2);
            for value in &mut s_ltp_q15[start..nsq.s_ltp_buf_idx] {
                *value = smulww(gain_adj_q16, *value);
            }
        }

        nsq.s_lf_ar_shp_q14 = smulww(gain_adj_q16, nsq.s_lf_ar_shp_q14);
        nsq.s_diff_shp_q14 = smulww(gain_adj_q16, nsq.s_diff_shp_q14);

        for value in &mut nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH] {
            *value = smulww(gain_adj_q16, *value);
        }
        for value in &mut nsq.s_ar2_q14[..MAX_SHAPE_LPC_ORDER] {
            *value = smulww(gain_adj_q16, *value);
        }

        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

#[allow(clippy::too_many_arguments)]
fn silk_noise_shape_quantizer(
    nsq: &mut NoiseShapingQuantizerState,
    signal_type: FrameSignalType,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq_offset: usize,
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_fir_packed_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: usize,
    shaping_lpc_order: usize,
    predict_lpc_order: usize,
) {
    assert_eq!(x_sc_q10.len(), length);
    assert_eq!(pulses.len(), length);
    assert!(xq_offset + length <= nsq.xq.len());
    assert!(lag >= 0);
    let lag = lag as usize;
    let gain_q10 = gain_q16 >> 6;
    assert!(nsq.s_ltp_shp_buf_idx >= lag + (HARM_SHAPE_FIR_TAPS / 2));
    let mut shp_lag_ptr = nsq.s_ltp_shp_buf_idx - lag + (HARM_SHAPE_FIR_TAPS / 2);
    assert!(nsq.s_ltp_buf_idx >= lag + (LTP_ORDER / 2));
    let mut pred_lag_ptr = nsq.s_ltp_buf_idx - lag + (LTP_ORDER / 2);
    let mut lpc_q14_offset = NSQ_LPC_BUF_LENGTH - 1;

    for i in 0..length {
        nsq.rand_seed = rand(nsq.rand_seed);

        let lpc_pred_q10 =
            noise_shape_short_prediction(&nsq.s_lpc_q14, lpc_q14_offset, a_q12, predict_lpc_order);

        let ltp_pred_q13 = if signal_type == FrameSignalType::Voiced {
            let base = pred_lag_ptr;
            assert!(base >= 4);
            let pred0 = s_ltp_q15[base];
            let pred1 = s_ltp_q15[base - 1];
            let pred2 = s_ltp_q15[base - 2];
            let pred3 = s_ltp_q15[base - 3];
            let pred4 = s_ltp_q15[base - 4];

            let mut acc = 2;
            acc = smlawb(acc, pred0, i32::from(b_q14[0]));
            acc = smlawb(acc, pred1, i32::from(b_q14[1]));
            acc = smlawb(acc, pred2, i32::from(b_q14[2]));
            acc = smlawb(acc, pred3, i32::from(b_q14[3]));
            acc = smlawb(acc, pred4, i32::from(b_q14[4]));
            pred_lag_ptr += 1;
            acc
        } else {
            0
        };

        assert_eq!(shaping_lpc_order % 2, 0);
        let n_ar_q12 = nsq_noise_shape_feedback_loop(
            nsq.s_diff_shp_q14,
            &mut nsq.s_ar2_q14[..shaping_lpc_order],
            ar_shp_q13,
            shaping_lpc_order,
        );

        let n_ar_q12 = smlawb(n_ar_q12, nsq.s_lf_ar_shp_q14, tilt_q14);

        let n_lf_q12 = smulwb(nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx - 1], lf_shp_q14);
        let n_lf_q12 = smlawt(n_lf_q12, nsq.s_lf_ar_shp_q14, lf_shp_q14);

        assert!(lag > 0 || signal_type != FrameSignalType::Voiced);

        let mut tmp1 = sub32_ovflw(lshift(lpc_pred_q10, 2), n_ar_q12);
        tmp1 = sub32_ovflw(tmp1, n_lf_q12);

        if lag > 0 {
            assert!(shp_lag_ptr >= 2);
            let lag_base = shp_lag_ptr;
            let mut n_ltp_q13 = smulwb(
                add_sat32(nsq.s_ltp_shp_q14[lag_base], nsq.s_ltp_shp_q14[lag_base - 2]),
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 = smlawt(
                n_ltp_q13,
                nsq.s_ltp_shp_q14[lag_base - 1],
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 = lshift(n_ltp_q13, 1);
            shp_lag_ptr += 1;

            let tmp2 = ltp_pred_q13.wrapping_sub(n_ltp_q13);
            tmp1 = add32_ovflw(tmp2, lshift(tmp1, 1));
            tmp1 = rshift_round(tmp1, 3);
        } else {
            tmp1 = rshift_round(tmp1, 2);
        }

        let mut r_q10 = x_sc_q10[i].wrapping_sub(tmp1);
        if nsq.rand_seed < 0 {
            r_q10 = -r_q10;
        }
        r_q10 = limit_32(r_q10, -(31 << 10), 30 << 10);

        let mut q1_q10 = r_q10.wrapping_sub(offset_q10);
        let mut q1_q0 = q1_q10 >> 10;

        if lambda_q10 > 2048 {
            let rdo_offset = (lambda_q10 >> 1) - 512;
            if q1_q10 > rdo_offset {
                q1_q0 = (q1_q10 - rdo_offset) >> 10;
            } else if q1_q10 < -rdo_offset {
                q1_q0 = (q1_q10 + rdo_offset) >> 10;
            } else if q1_q10 < 0 {
                q1_q0 = -1;
            } else {
                q1_q0 = 0;
            }
        }

        let q2_q10;
        let mut rd1_q20;
        let mut rd2_q20;

        if q1_q0 > 0 {
            q1_q10 = lshift(q1_q0, 10).wrapping_sub(QUANT_LEVEL_ADJUST_Q10);
            q1_q10 = q1_q10.wrapping_add(offset_q10);
            q2_q10 = q1_q10.wrapping_add(1024);
            rd1_q20 = smulbb(q1_q10, lambda_q10);
            rd2_q20 = smulbb(q2_q10, lambda_q10);
        } else if q1_q0 == 0 {
            q1_q10 = offset_q10;
            q2_q10 = q1_q10.wrapping_add(1024 - QUANT_LEVEL_ADJUST_Q10);
            rd1_q20 = smulbb(q1_q10, lambda_q10);
            rd2_q20 = smulbb(q2_q10, lambda_q10);
        } else if q1_q0 == -1 {
            q2_q10 = offset_q10;
            q1_q10 = q2_q10.wrapping_sub(1024 - QUANT_LEVEL_ADJUST_Q10);
            rd1_q20 = smulbb(-q1_q10, lambda_q10);
            rd2_q20 = smulbb(q2_q10, lambda_q10);
        } else {
            q1_q10 = lshift(q1_q0, 10).wrapping_add(QUANT_LEVEL_ADJUST_Q10);
            q1_q10 = q1_q10.wrapping_add(offset_q10);
            q2_q10 = q1_q10.wrapping_add(1024);
            rd1_q20 = smulbb(-q1_q10, lambda_q10);
            rd2_q20 = smulbb(-q2_q10, lambda_q10);
        }

        let mut rr_q10 = r_q10.wrapping_sub(q1_q10);
        rd1_q20 = smlabb(rd1_q20, rr_q10, rr_q10);

        rr_q10 = r_q10.wrapping_sub(q2_q10);
        rd2_q20 = smlabb(rd2_q20, rr_q10, rr_q10);

        if rd2_q20 < rd1_q20 {
            q1_q10 = q2_q10;
        }

        pulses[i] = rshift_round(q1_q10, 10) as i8;

        let mut exc_q14 = lshift(q1_q10, 4);
        if nsq.rand_seed < 0 {
            exc_q14 = -exc_q14;
        }

        let lpc_exc_q14 = add_lshift32(exc_q14, ltp_pred_q13, 1);
        let xq_q14 = add32_ovflw(lpc_exc_q14, lshift(lpc_pred_q10, 4));

        nsq.xq[xq_offset + i] = sat16(rshift_round(smulww(xq_q14, gain_q10), 8));

        lpc_q14_offset += 1;
        nsq.s_lpc_q14[lpc_q14_offset] = xq_q14;
        nsq.s_diff_shp_q14 = sub32_ovflw(xq_q14, lshift(x_sc_q10[i], 4));
        let s_lf_ar_shp_q14 = sub32_ovflw(nsq.s_diff_shp_q14, lshift(n_ar_q12, 2));
        nsq.s_lf_ar_shp_q14 = s_lf_ar_shp_q14;

        nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx] =
            sub32_ovflw(s_lf_ar_shp_q14, lshift(n_lf_q12, 2));
        s_ltp_q15[nsq.s_ltp_buf_idx] = lshift(lpc_exc_q14, 1);
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        nsq.rand_seed = add32_ovflw(nsq.rand_seed, i32::from(pulses[i]));
    }

    nsq.s_lpc_q14
        .copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
}

pub(crate) fn noise_shape_short_prediction(
    s_lpc_q14: &[i32],
    offset: usize,
    coef_q12: &[i16],
    order: usize,
) -> i32 {
    assert!(order == 10 || order == 16);
    assert!(offset + 1 >= order);
    assert!(coef_q12.len() >= order);

    let mut out = (order as i32) >> 1;
    for j in 0..order {
        out = smlawb(out, s_lpc_q14[offset - j], i32::from(coef_q12[j]));
    }
    out
}

pub(crate) fn nsq_noise_shape_feedback_loop(
    s_diff_shp_q14: i32,
    ar2_q14: &mut [i32],
    coef_q13: &[i16],
    order: usize,
) -> i32 {
    assert!(coef_q13.len() >= order);
    let mut tmp2 = s_diff_shp_q14;
    let mut tmp1 = ar2_q14[0];
    ar2_q14[0] = tmp2;

    let mut out = (order as i32) >> 1;
    out = smlawb(out, tmp2, i32::from(coef_q13[0]));

    for j in (2..order).step_by(2) {
        tmp2 = ar2_q14[j - 1];
        ar2_q14[j - 1] = tmp1;
        out = smlawb(out, tmp1, i32::from(coef_q13[j - 1]));
        tmp1 = ar2_q14[j];
        ar2_q14[j] = tmp2;
        out = smlawb(out, tmp2, i32::from(coef_q13[j]));
    }

    ar2_q14[order - 1] = tmp1;
    out = smlawb(out, tmp1, i32::from(coef_q13[order - 1]));
    lshift(out, 1)
}

pub(crate) fn quantization_offset(
    signal_type: FrameSignalType,
    quant_offset_type: FrameQuantizationOffsetType,
) -> i32 {
    let row = match signal_type {
        FrameSignalType::Voiced => 1,
        _ => 0,
    };
    let col = match quant_offset_type {
        FrameQuantizationOffsetType::Low => 0,
        FrameQuantizationOffsetType::High => 1,
    };
    i32::from(SILK_QUANTIZATION_OFFSETS_Q10[row][col])
}

#[inline]
pub(crate) fn rand(seed: i32) -> i32 {
    RAND_INCREMENT.wrapping_add(seed.wrapping_mul(RAND_MULTIPLIER))
}

#[inline]
pub(crate) fn add_sat32(a: i32, b: i32) -> i32 {
    (i64::from(a) + i64::from(b)).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[inline]
pub(crate) fn add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b.wrapping_shl(shift as u32))
}

#[inline]
pub(crate) fn add32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

#[inline]
pub(crate) fn sub32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline]
pub(crate) fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16).wrapping_mul(i32::from(b as i16))
}

#[inline]
pub(crate) fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

#[inline]
pub(crate) fn smulwt(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b >> 16)) >> 16) as i32
}

#[inline]
pub(crate) fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

#[inline]
pub(crate) fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulbb(b, c))
}

#[inline]
pub(crate) fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwb(b, c))
}

#[inline]
pub(crate) fn smlawt(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwt(b, c))
}

#[inline]
pub(crate) fn lshift(value: i32, shift: i32) -> i32 {
    value.wrapping_shl(shift as u32)
}

#[inline]
pub(crate) fn rshift_round(value: i32, shift: i32) -> i32 {
    assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

#[inline]
pub(crate) fn sat16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

#[inline]
pub(crate) fn limit_32(value: i32, min_val: i32, max_val: i32) -> i32 {
    value.clamp(min_val, max_val)
}
