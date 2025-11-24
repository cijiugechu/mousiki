#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop
)]

use core::cmp::{max, min};

use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::encoder::state::{
    EncoderStateCommon, MAX_DEL_DEC_STATES, MAX_FRAME_LENGTH, MAX_LTP_MEM_LENGTH,
    MAX_SUB_FRAME_LENGTH, NSQ_LPC_BUF_LENGTH, NoiseShapingQuantizerState,
};
use crate::silk::lpc_analysis_filter::lpc_analysis_filter;
use crate::silk::lpc_inv_pred_gain::inverse32_varq;
use crate::silk::nsq::{
    HARM_SHAPE_FIR_TAPS, QUANT_LEVEL_ADJUST_Q10, add_sat32, add32_ovflw, limit_32, lshift,
    quantization_offset, rand, rshift_round, sat16, smlabb, smlawb, smlawt, smulbb, smulwb, smulww,
    sub32_ovflw,
};
use crate::silk::stereo_find_predictor::div32_varq;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER};

const DECISION_DELAY: usize = 40;

#[derive(Clone, Debug)]
struct DelayedDecisionState {
    s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    rand_state: [i32; DECISION_DELAY],
    q_q10: [i32; DECISION_DELAY],
    xq_q14: [i32; DECISION_DELAY],
    pred_q15: [i32; DECISION_DELAY],
    shape_q14: [i32; DECISION_DELAY],
    s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    lf_ar_q14: i32,
    diff_q14: i32,
    seed: i32,
    seed_init: i32,
    rd_q10: i32,
}

impl Default for DelayedDecisionState {
    fn default() -> Self {
        Self {
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            rand_state: [0; DECISION_DELAY],
            q_q10: [0; DECISION_DELAY],
            xq_q14: [0; DECISION_DELAY],
            pred_q15: [0; DECISION_DELAY],
            shape_q14: [0; DECISION_DELAY],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            lf_ar_q14: 0,
            diff_q14: 0,
            seed: 0,
            seed_init: 0,
            rd_q10: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SampleState {
    q_q10: i32,
    rd_q10: i32,
    xq_q14: i32,
    lf_ar_q14: i32,
    diff_q14: i32,
    s_ltp_shp_q14: i32,
    lpc_exc_q14: i32,
}

type SamplePair = [SampleState; 2];

/// Delayed-decision noise-shaping quantiser (`silk_NSQ_del_dec`).
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_del_dec(
    encoder: &EncoderStateCommon,
    nsq: &mut NoiseShapingQuantizerState,
    indices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
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
    assert_eq!(x16.len(), encoder.frame_length);
    assert_eq!(pulses.len(), encoder.frame_length);
    assert!(encoder.predict_lpc_order <= MAX_LPC_ORDER);
    assert!(encoder.shaping_lpc_order as usize <= MAX_SHAPE_LPC_ORDER);
    assert!(encoder.subfr_length <= MAX_SUB_FRAME_LENGTH);

    assert!(encoder.n_states_delayed_decision > 0);
    let n_states = encoder.n_states_delayed_decision as usize;

    let mut ps_del_dec_storage: [DelayedDecisionState; MAX_DEL_DEC_STATES as usize] =
        core::array::from_fn(|_| DelayedDecisionState::default());
    let ps_del_dec = &mut ps_del_dec_storage[..n_states];

    let mut lag = nsq.lag_prev;
    assert_ne!(nsq.prev_gain_q16, 0);

    for (k, state) in ps_del_dec.iter_mut().enumerate() {
        state.seed = (k as i32 + i32::from(indices.seed)) & 3;
        state.seed_init = state.seed;
        state.rd_q10 = 0;
        state.lf_ar_q14 = nsq.s_lf_ar_shp_q14;
        state.diff_q14 = nsq.s_diff_shp_q14;
        state.shape_q14[0] = nsq.s_ltp_shp_q14[encoder.ltp_mem_length - 1];
        state.s_lpc_q14[..NSQ_LPC_BUF_LENGTH].copy_from_slice(&nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]);
        state.s_ar2_q14.copy_from_slice(&nsq.s_ar2_q14);
    }

    let offset_q10 = quantization_offset(indices.signal_type, indices.quant_offset_type);
    let mut smpl_buf_idx = 0usize;

    let mut decision_delay = min(DECISION_DELAY, encoder.subfr_length);
    if indices.signal_type == FrameSignalType::Voiced {
        for &lag_k in pitch_l.iter().take(encoder.nb_subfr) {
            decision_delay = min(
                decision_delay,
                max(lag_k - (LTP_ORDER as i32 / 2) - 1, 0) as usize,
            );
        }
    } else if lag > 0 {
        decision_delay = min(
            decision_delay,
            max(lag - (LTP_ORDER as i32 / 2) - 1, 0) as usize,
        );
    }

    let lsf_interpolation_flag = if indices.nlsf_interp_coef_q2 == 4 {
        0
    } else {
        1
    };

    let mut s_ltp_q15 = [0i32; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH];
    let mut s_ltp = [0i16; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH];
    let mut x_sc_q10 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut delayed_gain_q10 = [0i32; DECISION_DELAY];

    nsq.s_ltp_shp_buf_idx = encoder.ltp_mem_length;
    nsq.s_ltp_buf_idx = encoder.ltp_mem_length;

    let mut pulses_offset = 0;
    let mut x16_offset = 0;
    let mut xq_offset = encoder.ltp_mem_length;
    let mut subfr = 0;

    for k in 0..encoder.nb_subfr {
        let a_q12_offset = ((k >> 1) | (1 - lsf_interpolation_flag)) * MAX_LPC_ORDER;
        let a_q12 = &pred_coef_q12[a_q12_offset..a_q12_offset + encoder.predict_lpc_order];
        let b_q14_offset = k * LTP_ORDER;
        let b_q14 = &ltp_coef_q14[b_q14_offset..b_q14_offset + LTP_ORDER];
        let ar_shp_q13_offset = k * MAX_SHAPE_LPC_ORDER;
        let ar_shp_q13 =
            &ar_q13[ar_shp_q13_offset..ar_shp_q13_offset + encoder.shaping_lpc_order as usize];

        let mut harm_shape_fir_packed_q14 = harm_shape_gain_q14[k] >> 2;
        harm_shape_fir_packed_q14 |= (harm_shape_gain_q14[k] >> 1) << 16;

        nsq.rewhite_flag = false;
        if indices.signal_type == FrameSignalType::Voiced {
            lag = pitch_l[k];

            if (k & (3 - (lsf_interpolation_flag << 1))) == 0 {
                if k == 2 {
                    let mut rd_min_q10 = ps_del_dec[0].rd_q10;
                    let mut winner_ind = 0;
                    for i in 1..n_states {
                        if ps_del_dec[i].rd_q10 < rd_min_q10 {
                            rd_min_q10 = ps_del_dec[i].rd_q10;
                            winner_ind = i;
                        }
                    }

                    for i in 0..n_states {
                        if i != winner_ind {
                            ps_del_dec[i].rd_q10 = add_sat32(ps_del_dec[i].rd_q10, i32::MAX >> 4);
                        }
                    }

                    let winner = &ps_del_dec[winner_ind];
                    let mut last_smple_idx = (smpl_buf_idx + decision_delay) % DECISION_DELAY;
                    for i in 0..decision_delay {
                        last_smple_idx = (last_smple_idx + DECISION_DELAY - 1) % DECISION_DELAY;

                        assert!(pulses_offset + i >= decision_delay);
                        let pulse_out_idx = pulses_offset + i - decision_delay;
                        pulses[pulse_out_idx] =
                            rshift_round(winner.q_q10[last_smple_idx], 10) as i8;

                        assert!(xq_offset + i >= decision_delay);
                        let pxq_idx = xq_offset + i - decision_delay;
                        nsq.xq[pxq_idx] = sat16(rshift_round(
                            smulww(winner.xq_q14[last_smple_idx], gains_q16[1] >> 6),
                            8,
                        ));

                        let shp_idx = nsq.s_ltp_shp_buf_idx - decision_delay + i;
                        nsq.s_ltp_shp_q14[shp_idx] = winner.shape_q14[last_smple_idx];
                    }

                    subfr = 0;
                }

                let start_idx = encoder.ltp_mem_length as i32
                    - lag
                    - encoder.predict_lpc_order as i32
                    - (LTP_ORDER as i32 / 2);
                assert!(start_idx > 0);
                let start = start_idx as usize;
                lpc_analysis_filter(
                    &mut s_ltp[start..start + encoder.ltp_mem_length - start],
                    &nsq.xq[start + k * encoder.subfr_length
                        ..start + k * encoder.subfr_length + encoder.ltp_mem_length - start],
                    a_q12,
                    encoder.ltp_mem_length - start,
                    encoder.predict_lpc_order,
                );

                nsq.s_ltp_buf_idx = encoder.ltp_mem_length;
                nsq.rewhite_flag = true;
            }
        }

        nsq_del_dec_scale_states(
            encoder,
            nsq,
            ps_del_dec,
            &x16[x16_offset..x16_offset + encoder.subfr_length],
            &mut x_sc_q10[..encoder.subfr_length],
            &s_ltp,
            &mut s_ltp_q15,
            k,
            n_states,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            indices.signal_type,
            decision_delay,
        );

        noise_shape_quantizer_del_dec(
            encoder,
            nsq,
            ps_del_dec,
            indices.signal_type,
            &x_sc_q10[..encoder.subfr_length],
            pulses,
            pulses_offset,
            xq_offset,
            &mut s_ltp_q15,
            &mut delayed_gain_q10,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_fir_packed_q14,
            tilt_q14[k],
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            subfr,
            encoder.shaping_lpc_order as usize,
            encoder.predict_lpc_order,
            encoder.warping_q16,
            n_states,
            &mut smpl_buf_idx,
            decision_delay,
        );
        subfr += 1;

        x16_offset += encoder.subfr_length;
        pulses_offset += encoder.subfr_length;
        xq_offset += encoder.subfr_length;
    }

    let mut rd_min_q10 = ps_del_dec[0].rd_q10;
    let mut winner_ind = 0;
    for k in 1..n_states {
        if ps_del_dec[k].rd_q10 < rd_min_q10 {
            rd_min_q10 = ps_del_dec[k].rd_q10;
            winner_ind = k;
        }
    }

    let winner = &ps_del_dec[winner_ind];
    indices.seed = winner.seed_init as i8;
    let mut last_smple_idx = (smpl_buf_idx + decision_delay) % DECISION_DELAY;
    let gain_q10 = gains_q16[encoder.nb_subfr - 1] >> 6;
    assert!(pulses_offset >= decision_delay);
    assert!(xq_offset >= decision_delay);
    assert!(nsq.s_ltp_shp_buf_idx >= decision_delay);
    for i in 0..decision_delay {
        last_smple_idx = (last_smple_idx + DECISION_DELAY - 1) % DECISION_DELAY;
        assert!(pulses_offset + i >= decision_delay);
        let pulse_out_idx = pulses_offset + i - decision_delay;
        pulses[pulse_out_idx] = rshift_round(winner.q_q10[last_smple_idx], 10) as i8;
        assert!(xq_offset + i >= decision_delay);
        let pxq_idx = xq_offset + i - decision_delay;
        nsq.xq[pxq_idx] = sat16(rshift_round(
            smulww(winner.xq_q14[last_smple_idx], gain_q10),
            8,
        ));
        let shp_idx = nsq.s_ltp_shp_buf_idx - decision_delay + i;
        nsq.s_ltp_shp_q14[shp_idx] = winner.shape_q14[last_smple_idx];
    }
    nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH].copy_from_slice(
        &winner.s_lpc_q14[encoder.subfr_length..encoder.subfr_length + NSQ_LPC_BUF_LENGTH],
    );
    nsq.s_ar2_q14.copy_from_slice(&winner.s_ar2_q14);

    nsq.s_lf_ar_shp_q14 = winner.lf_ar_q14;
    nsq.s_diff_shp_q14 = winner.diff_q14;
    nsq.lag_prev = pitch_l[encoder.nb_subfr - 1];

    nsq.xq.copy_within(
        encoder.frame_length..encoder.frame_length + encoder.ltp_mem_length,
        0,
    );
    nsq.s_ltp_shp_q14.copy_within(
        encoder.frame_length..encoder.frame_length + encoder.ltp_mem_length,
        0,
    );
}

#[allow(clippy::too_many_arguments)]
fn noise_shape_quantizer_del_dec(
    encoder: &EncoderStateCommon,
    nsq: &mut NoiseShapingQuantizerState,
    states: &mut [DelayedDecisionState],
    signal_type: FrameSignalType,
    x_q10: &[i32],
    pulses: &mut [i8],
    pulses_offset: usize,
    xq_offset: usize,
    s_ltp_q15: &mut [i32],
    delayed_gain_q10: &mut [i32],
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
    subfr: usize,
    shaping_lpc_order: usize,
    predict_lpc_order: usize,
    warping_q16: i32,
    n_states: usize,
    smpl_buf_idx: &mut usize,
    decision_delay: usize,
) {
    assert_eq!(x_q10.len(), encoder.subfr_length);
    assert_eq!(shaping_lpc_order, encoder.shaping_lpc_order as usize);
    assert_eq!(shaping_lpc_order % 2, 0);

    let mut sample_state_storage: [SamplePair; MAX_DEL_DEC_STATES as usize] =
        core::array::from_fn(|_| [SampleState::default(); 2]);
    let sample_state = &mut sample_state_storage[..n_states];
    assert!(lag >= 0);
    let lag_usize = lag as usize;
    assert!(nsq.s_ltp_buf_idx + LTP_ORDER / 2 >= lag_usize);
    assert!(nsq.s_ltp_shp_buf_idx + HARM_SHAPE_FIR_TAPS / 2 >= lag_usize);
    let mut shp_lag_ptr = nsq.s_ltp_shp_buf_idx + HARM_SHAPE_FIR_TAPS / 2 - lag_usize;
    let mut pred_lag_ptr = nsq.s_ltp_buf_idx + LTP_ORDER / 2 - lag_usize;
    let gain_q10 = gain_q16 >> 6;

    for i in 0..encoder.subfr_length {
        if signal_type == FrameSignalType::Voiced {
            assert!(pred_lag_ptr >= 4);
            assert!(pred_lag_ptr < s_ltp_q15.len());
        }
        if lag > 0 {
            assert!(shp_lag_ptr >= 2);
            assert!(shp_lag_ptr < nsq.s_ltp_shp_q14.len());
        }

        let ltp_pred_q14 = if signal_type == FrameSignalType::Voiced {
            let mut pred = 2;
            pred = smlawb(pred, s_ltp_q15[pred_lag_ptr], i32::from(b_q14[0]));
            pred = smlawb(pred, s_ltp_q15[pred_lag_ptr - 1], i32::from(b_q14[1]));
            pred = smlawb(pred, s_ltp_q15[pred_lag_ptr - 2], i32::from(b_q14[2]));
            pred = smlawb(pred, s_ltp_q15[pred_lag_ptr - 3], i32::from(b_q14[3]));
            pred = smlawb(pred, s_ltp_q15[pred_lag_ptr - 4], i32::from(b_q14[4]));
            pred_lag_ptr += 1;
            lshift(pred, 1)
        } else {
            0
        };

        let n_ltp_q14 = if lag > 0 {
            let mut n_ltp = smulwb(
                add_sat32(
                    nsq.s_ltp_shp_q14[shp_lag_ptr],
                    nsq.s_ltp_shp_q14[shp_lag_ptr - 2],
                ),
                harm_shape_fir_packed_q14,
            );
            n_ltp = smlawt(
                n_ltp,
                nsq.s_ltp_shp_q14[shp_lag_ptr - 1],
                harm_shape_fir_packed_q14,
            );
            n_ltp = ltp_pred_q14.wrapping_sub(lshift(n_ltp, 1));
            shp_lag_ptr += 1;
            n_ltp
        } else {
            0
        };

        for k in 0..n_states {
            let ps_dd = &mut states[k];
            let ps_ss = &mut sample_state[k];

            ps_dd.seed = rand(ps_dd.seed);

            let lpc_q14_offset = NSQ_LPC_BUF_LENGTH - 1 + i;
            let lpc_pred_q14 = lshift(
                crate::silk::nsq::noise_shape_short_prediction(
                    &ps_dd.s_lpc_q14,
                    lpc_q14_offset,
                    a_q12,
                    predict_lpc_order,
                ),
                4,
            );

            let mut tmp2 = smlawb(ps_dd.diff_q14, ps_dd.s_ar2_q14[0], warping_q16);
            let mut tmp1 = smlawb(
                ps_dd.s_ar2_q14[0],
                sub32_ovflw(ps_dd.s_ar2_q14[1], tmp2),
                warping_q16,
            );
            ps_dd.s_ar2_q14[0] = tmp2;

            let mut n_ar_q14 = (shaping_lpc_order as i32) >> 1;
            n_ar_q14 = smlawb(n_ar_q14, tmp2, i32::from(ar_shp_q13[0]));

            for j in (2..shaping_lpc_order).step_by(2) {
                tmp2 = smlawb(
                    ps_dd.s_ar2_q14[j - 1],
                    sub32_ovflw(ps_dd.s_ar2_q14[j], tmp1),
                    warping_q16,
                );
                ps_dd.s_ar2_q14[j - 1] = tmp1;
                n_ar_q14 = smlawb(n_ar_q14, tmp1, i32::from(ar_shp_q13[j - 1]));

                tmp1 = smlawb(
                    ps_dd.s_ar2_q14[j],
                    sub32_ovflw(ps_dd.s_ar2_q14[j + 1], tmp2),
                    warping_q16,
                );
                ps_dd.s_ar2_q14[j] = tmp2;
                n_ar_q14 = smlawb(n_ar_q14, tmp2, i32::from(ar_shp_q13[j]));
            }
            ps_dd.s_ar2_q14[shaping_lpc_order - 1] = tmp1;
            n_ar_q14 = smlawb(n_ar_q14, tmp1, i32::from(ar_shp_q13[shaping_lpc_order - 1]));

            n_ar_q14 = lshift(n_ar_q14, 1);
            n_ar_q14 = smlawb(n_ar_q14, ps_dd.lf_ar_q14, tilt_q14);
            n_ar_q14 = lshift(n_ar_q14, 2);

            let mut n_lf_q14 = smulwb(ps_dd.shape_q14[*smpl_buf_idx], lf_shp_q14);
            n_lf_q14 = smlawt(n_lf_q14, ps_dd.lf_ar_q14, lf_shp_q14);
            n_lf_q14 = lshift(n_lf_q14, 2);

            let mut tmp1 = add_sat32(n_ar_q14, n_lf_q14);
            let tmp2 = add32_ovflw(n_ltp_q14, lpc_pred_q14);
            tmp1 = tmp2.saturating_sub(tmp1);
            tmp1 = rshift_round(tmp1, 4);

            let mut r_q10 = x_q10[i].wrapping_sub(tmp1);
            if ps_dd.seed < 0 {
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
            let mut rd1_q10;
            let mut rd2_q10;
            if q1_q0 > 0 {
                q1_q10 = add32_ovflw(lshift(q1_q0, 10), -QUANT_LEVEL_ADJUST_Q10);
                q1_q10 = add32_ovflw(q1_q10, offset_q10);
                q2_q10 = add32_ovflw(q1_q10, 1024);
                rd1_q10 = smulbb(q1_q10, lambda_q10);
                rd2_q10 = smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == 0 {
                q1_q10 = offset_q10;
                q2_q10 = add32_ovflw(q1_q10, 1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_q10 = smulbb(q1_q10, lambda_q10);
                rd2_q10 = smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == -1 {
                q2_q10 = offset_q10;
                q1_q10 = q2_q10 - (1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_q10 = smulbb(-q1_q10, lambda_q10);
                rd2_q10 = smulbb(q2_q10, lambda_q10);
            } else {
                q1_q10 = add32_ovflw(lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10);
                q1_q10 = add32_ovflw(q1_q10, offset_q10);
                q2_q10 = add32_ovflw(q1_q10, 1024);
                rd1_q10 = smulbb(-q1_q10, lambda_q10);
                rd2_q10 = smulbb(-q2_q10, lambda_q10);
            }

            let mut rr_q10 = r_q10.wrapping_sub(q1_q10);
            rd1_q10 = rshift_round(smlabb(rd1_q10, rr_q10, rr_q10), 10);
            rr_q10 = r_q10.wrapping_sub(q2_q10);
            rd2_q10 = rshift_round(smlabb(rd2_q10, rr_q10, rr_q10), 10);

            if rd1_q10 < rd2_q10 {
                ps_ss[0].rd_q10 = add32_ovflw(ps_dd.rd_q10, rd1_q10);
                ps_ss[1].rd_q10 = add32_ovflw(ps_dd.rd_q10, rd2_q10);
                ps_ss[0].q_q10 = q1_q10;
                ps_ss[1].q_q10 = q2_q10;
            } else {
                ps_ss[0].rd_q10 = add32_ovflw(ps_dd.rd_q10, rd2_q10);
                ps_ss[1].rd_q10 = add32_ovflw(ps_dd.rd_q10, rd1_q10);
                ps_ss[0].q_q10 = q2_q10;
                ps_ss[1].q_q10 = q1_q10;
            }

            let mut exc_q14 = lshift(ps_ss[0].q_q10, 4);
            if ps_dd.seed < 0 {
                exc_q14 = -exc_q14;
            }
            let lpc_exc_q14 = exc_q14.wrapping_add(ltp_pred_q14);
            let xq_q14 = add32_ovflw(lpc_exc_q14, lpc_pred_q14);

            ps_ss[0].diff_q14 = sub32_ovflw(xq_q14, lshift(x_q10[i], 4));
            let s_lf_ar_shp_q14 = sub32_ovflw(ps_ss[0].diff_q14, n_ar_q14);
            ps_ss[0].s_ltp_shp_q14 = s_lf_ar_shp_q14.wrapping_sub(n_lf_q14);
            ps_ss[0].lf_ar_q14 = s_lf_ar_shp_q14;
            ps_ss[0].lpc_exc_q14 = lpc_exc_q14;
            ps_ss[0].xq_q14 = xq_q14;

            let mut exc_q14 = lshift(ps_ss[1].q_q10, 4);
            if ps_dd.seed < 0 {
                exc_q14 = -exc_q14;
            }
            let lpc_exc_q14 = exc_q14.wrapping_add(ltp_pred_q14);
            let xq_q14 = add32_ovflw(lpc_exc_q14, lpc_pred_q14);

            ps_ss[1].diff_q14 = sub32_ovflw(xq_q14, lshift(x_q10[i], 4));
            let s_lf_ar_shp_q14 = sub32_ovflw(ps_ss[1].diff_q14, n_ar_q14);
            ps_ss[1].s_ltp_shp_q14 = s_lf_ar_shp_q14.wrapping_sub(n_lf_q14);
            ps_ss[1].lf_ar_q14 = s_lf_ar_shp_q14;
            ps_ss[1].lpc_exc_q14 = lpc_exc_q14;
            ps_ss[1].xq_q14 = xq_q14;
        }

        *smpl_buf_idx = (*smpl_buf_idx + DECISION_DELAY - 1) % DECISION_DELAY;
        let last_smple_idx = (*smpl_buf_idx + decision_delay) % DECISION_DELAY;

        let mut rd_min_q10 = sample_state[0][0].rd_q10;
        let mut winner_ind = 0;
        for k in 1..n_states {
            if sample_state[k][0].rd_q10 < rd_min_q10 {
                rd_min_q10 = sample_state[k][0].rd_q10;
                winner_ind = k;
            }
        }

        let winner_seed = states[winner_ind].rand_state[last_smple_idx];
        for k in 0..n_states {
            if states[k].rand_state[last_smple_idx] != winner_seed {
                sample_state[k][0].rd_q10 = add_sat32(sample_state[k][0].rd_q10, i32::MAX >> 4);
                sample_state[k][1].rd_q10 = add_sat32(sample_state[k][1].rd_q10, i32::MAX >> 4);
            }
        }

        let mut rd_max_q10 = sample_state[0][0].rd_q10;
        let mut rd_min_q10 = sample_state[0][1].rd_q10;
        let mut rd_max_ind = 0;
        let mut rd_min_ind = 0;
        for k in 1..n_states {
            if sample_state[k][0].rd_q10 > rd_max_q10 {
                rd_max_q10 = sample_state[k][0].rd_q10;
                rd_max_ind = k;
            }
            if sample_state[k][1].rd_q10 < rd_min_q10 {
                rd_min_q10 = sample_state[k][1].rd_q10;
                rd_min_ind = k;
            }
        }

        if rd_min_q10 < rd_max_q10 {
            let mut replacement = states[rd_min_ind].clone();
            replacement.rd_q10 = sample_state[rd_min_ind][1].rd_q10;
            states[rd_max_ind] = replacement;
            sample_state[rd_max_ind][0] = sample_state[rd_min_ind][1];
        }

        let winner = &states[winner_ind];
        if subfr > 0 || i >= decision_delay {
            assert!(pulses_offset + i >= decision_delay);
            let out_idx = pulses_offset + i - decision_delay;
            pulses[out_idx] = rshift_round(winner.q_q10[last_smple_idx], 10) as i8;
            assert!(xq_offset + i >= decision_delay);
            let xq_idx = xq_offset + i - decision_delay;
            nsq.xq[xq_idx] = sat16(rshift_round(
                smulww(
                    winner.xq_q14[last_smple_idx],
                    delayed_gain_q10[last_smple_idx],
                ),
                8,
            ));
            assert!(nsq.s_ltp_shp_buf_idx >= decision_delay);
            let shp_idx = nsq.s_ltp_shp_buf_idx - decision_delay;
            nsq.s_ltp_shp_q14[shp_idx] = winner.shape_q14[last_smple_idx];
            assert!(nsq.s_ltp_buf_idx >= decision_delay);
            let pred_idx = nsq.s_ltp_buf_idx - decision_delay;
            s_ltp_q15[pred_idx] = winner.pred_q15[last_smple_idx];
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        for k in 0..n_states {
            let ps_dd = &mut states[k];
            let ps_ss = sample_state[k][0];
            ps_dd.lf_ar_q14 = ps_ss.lf_ar_q14;
            ps_dd.diff_q14 = ps_ss.diff_q14;
            ps_dd.s_lpc_q14[NSQ_LPC_BUF_LENGTH + i] = ps_ss.xq_q14;
            ps_dd.xq_q14[*smpl_buf_idx] = ps_ss.xq_q14;
            ps_dd.q_q10[*smpl_buf_idx] = ps_ss.q_q10;
            ps_dd.pred_q15[*smpl_buf_idx] = lshift(ps_ss.lpc_exc_q14, 1);
            ps_dd.shape_q14[*smpl_buf_idx] = ps_ss.s_ltp_shp_q14;
            ps_dd.seed = add32_ovflw(ps_dd.seed, rshift_round(ps_ss.q_q10, 10));
            ps_dd.rand_state[*smpl_buf_idx] = ps_dd.seed;
            ps_dd.rd_q10 = ps_ss.rd_q10;
        }
        delayed_gain_q10[*smpl_buf_idx] = gain_q10;
    }

    for state in states.iter_mut() {
        state.s_lpc_q14.copy_within(
            encoder.subfr_length..encoder.subfr_length + NSQ_LPC_BUF_LENGTH,
            0,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn nsq_del_dec_scale_states(
    encoder: &EncoderStateCommon,
    nsq: &mut NoiseShapingQuantizerState,
    states: &mut [DelayedDecisionState],
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    n_states: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: FrameSignalType,
    decision_delay: usize,
) {
    let lag = pitch_l[subfr];
    assert!(lag >= 0);
    assert_eq!(x16.len(), encoder.subfr_length);
    assert_eq!(x_sc_q10.len(), encoder.subfr_length);
    let inv_gain_q31 = inverse32_varq(max(gains_q16[subfr], 1), 47);
    assert_ne!(inv_gain_q31, 0);

    let inv_gain_q26 = rshift_round(inv_gain_q31, 5);
    for (dst, &sample) in x_sc_q10.iter_mut().zip(x16.iter()) {
        *dst = smulww(i32::from(sample), inv_gain_q26);
    }

    if nsq.rewhite_flag {
        let mut inv_gain_q31 = inv_gain_q31;
        if subfr == 0 {
            inv_gain_q31 = lshift(smulwb(inv_gain_q31, ltp_scale_q14), 2);
        }
        let lag_usize = lag as usize;
        assert!(nsq.s_ltp_buf_idx >= lag_usize + (LTP_ORDER / 2));
        let start = nsq.s_ltp_buf_idx - lag_usize - (LTP_ORDER / 2);
        for i in start..nsq.s_ltp_buf_idx {
            assert!(i < s_ltp_q15.len());
            s_ltp_q15[i] = smulwb(inv_gain_q31, i32::from(s_ltp[i]));
        }
    }

    if gains_q16[subfr] != nsq.prev_gain_q16 {
        let gain_adj_q16 = div32_varq(nsq.prev_gain_q16, gains_q16[subfr], 16);

        assert!(nsq.s_ltp_shp_buf_idx >= encoder.ltp_mem_length);
        let start = nsq.s_ltp_shp_buf_idx - encoder.ltp_mem_length;
        for val in &mut nsq.s_ltp_shp_q14[start..nsq.s_ltp_shp_buf_idx] {
            *val = smulww(gain_adj_q16, *val);
        }

        if signal_type == FrameSignalType::Voiced && !nsq.rewhite_flag {
            let lag_usize = lag as usize;
            assert!(nsq.s_ltp_buf_idx >= lag_usize + (LTP_ORDER / 2));
            let start = nsq.s_ltp_buf_idx - lag_usize - (LTP_ORDER / 2);
            let end = nsq.s_ltp_buf_idx - decision_delay;
            for val in &mut s_ltp_q15[start..end] {
                *val = smulww(gain_adj_q16, *val);
            }
        }

        for state in states.iter_mut().take(n_states) {
            state.lf_ar_q14 = smulww(gain_adj_q16, state.lf_ar_q14);
            state.diff_q14 = smulww(gain_adj_q16, state.diff_q14);

            for val in &mut state.s_lpc_q14[..NSQ_LPC_BUF_LENGTH] {
                *val = smulww(gain_adj_q16, *val);
            }
            for val in &mut state.s_ar2_q14[..MAX_SHAPE_LPC_ORDER] {
                *val = smulww(gain_adj_q16, *val);
            }
            for val in &mut state.pred_q15[..DECISION_DELAY] {
                *val = smulww(gain_adj_q16, *val);
            }
            for val in &mut state.shape_q14[..DECISION_DELAY] {
                *val = smulww(gain_adj_q16, *val);
            }
        }

        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}
