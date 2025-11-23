//! Floating-point wrappers that reuse the fixed-point SILK helpers.
//!
//! Mirrors the conversion shims in `silk/float/wrappers_FLP.c`, translating
//! floating-point control/state into the fixed-point representations expected
//! by the shared predictors, NLSF tools, and NSQ implementations.

use crate::silk::a2nlsf::a2nlsf;
use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::encoder::control_flp::EncoderControlFlp;
use crate::silk::encoder::state::{
    EncoderStateCommon, NoiseShapingQuantizerState, MAX_FRAME_LENGTH,
};
use crate::silk::encoder::state_flp::EncoderStateFlp;
use crate::silk::nsq::silk_nsq;
use crate::silk::nsq_del_dec::silk_nsq_del_dec;
use crate::silk::nlsf2a::nlsf2a;
use crate::silk::process_nlsfs::{process_nlsfs, ProcessNlsfConfig};
use crate::silk::quant_ltp_gains::silk_quant_ltp_gains;
use crate::silk::sigproc_flp::silk_float2int;
use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER};
use core::convert::TryFrom;

const Q12_SCALE: f32 = 1.0 / 4096.0;
const Q14_SCALE: f32 = 1.0 / 16384.0;
const Q16_SCALE: f32 = 1.0 / 65536.0;

/// Convert LPC prediction coefficients (f32) to NLSF Q15.
#[allow(clippy::cast_possible_truncation)]
pub fn silk_a2nlsf_flp(nlsf_q15: &mut [i16], ar: &[f32], lpc_order: usize) {
    assert!(lpc_order.is_multiple_of(2), "LPC order must be even");
    assert!(lpc_order <= MAX_LPC_ORDER, "LPC order exceeds MAX_LPC_ORDER");
    assert!(
        ar.len() >= lpc_order,
        "input LPC slice shorter than requested order"
    );
    assert!(
        nlsf_q15.len() >= lpc_order,
        "output NLSF slice shorter than requested order"
    );

    let mut a_q16 = [0i32; MAX_LPC_ORDER];
    for (dst, &src) in a_q16.iter_mut().zip(ar.iter()).take(lpc_order) {
        *dst = silk_float2int(src * Q16_SCALE.recip());
    }

    a2nlsf(&mut nlsf_q15[..lpc_order], &mut a_q16[..lpc_order]);
}

/// Convert NLSF Q15 vectors back to floating-point LPC coefficients.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn silk_nlsf2a_flp(p_ar: &mut [f32], nlsf_q15: &[i16], lpc_order: usize, arch: i32) {
    assert!(lpc_order <= MAX_LPC_ORDER, "LPC order exceeds MAX_LPC_ORDER");
    assert!(
        nlsf_q15.len() >= lpc_order,
        "input NLSF slice shorter than requested order"
    );
    assert!(
        p_ar.len() >= lpc_order,
        "output LPC slice shorter than requested order"
    );

    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    nlsf2a(
        &mut a_q12[..lpc_order],
        &nlsf_q15[..lpc_order],
        arch,
    );

    for (dst, &src) in p_ar.iter_mut().zip(a_q12.iter()).take(lpc_order) {
        *dst = f32::from(src) * Q12_SCALE;
    }
}

/// Floating-point wrapper around the fixed-point NLSF processing helper.
#[allow(clippy::cast_sign_loss)]
pub fn silk_process_nlsfs_flp(
    common: &mut EncoderStateCommon,
    pred_coef: &mut [[f32; MAX_LPC_ORDER]; 2],
    nlsf_q15: &mut [i16],
    prev_nlsf_q15: &[i16],
) {
    let order = common.predict_lpc_order;
    assert!(matches!(order, 10 | 16), "SILK supports 10 or 16 taps");
    assert!(
        nlsf_q15.len() >= order,
        "NLSF vector shorter than predict_lpc_order"
    );
    assert!(
        prev_nlsf_q15.len() >= order,
        "previous NLSF vector shorter than predict_lpc_order"
    );

    let mut pred_coef_q12 = [[0i16; MAX_LPC_ORDER]; 2];
    let cfg = ProcessNlsfConfig {
        speech_activity_q8: common.speech_activity_q8,
        nb_subframes: common.nb_subfr,
        predict_lpc_order: order,
        use_interpolated_nlsfs: common.use_interpolated_nlsfs,
        nlsf_msvq_survivors: common.nlsf_msvq_survivors as usize,
        codebook: common.ps_nlsf_cb,
        arch: common.arch,
    };
    process_nlsfs(
        &cfg,
        &mut common.indices,
        &mut pred_coef_q12,
        &mut nlsf_q15[..order],
        &prev_nlsf_q15[..order],
    );

    for (dst_row, src_row) in pred_coef.iter_mut().zip(pred_coef_q12.iter()) {
        for (dst, &src) in dst_row.iter_mut().zip(src_row.iter()).take(order) {
            *dst = f32::from(src) * Q12_SCALE;
        }
    }
}

/// Floating-point front-end for the SILK noise-shaping quantiser.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn silk_nsq_wrapper_flp(
    enc: &EncoderStateFlp,
    enc_ctrl: &EncoderControlFlp,
    indices: &mut SideInfoIndices,
    nsq: &mut NoiseShapingQuantizerState,
    pulses: &mut [i8],
    x: &[f32],
) {
    let common = &enc.common;
    let nb_subfr = common.nb_subfr;
    let frame_length = common.frame_length;
    let shaping_order = common.shaping_lpc_order as usize;
    let predict_order = common.predict_lpc_order;

    assert_eq!(pulses.len(), frame_length);
    assert_eq!(x.len(), frame_length);
    assert!(
        nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2,
        "nb_subfr must be 4 or 2"
    );
    assert!(shaping_order <= MAX_SHAPE_LPC_ORDER);
    assert!(predict_order <= MAX_LPC_ORDER);
    assert!(
        frame_length <= MAX_FRAME_LENGTH,
        "frame_length exceeds MAX_FRAME_LENGTH"
    );

    let mut x16 = [0i16; MAX_FRAME_LENGTH];
    for (dst, &src) in x16.iter_mut().zip(x.iter()).take(frame_length) {
        *dst = silk_float2int(src) as i16;
    }

    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    for (dst, &src) in gains_q16.iter_mut().zip(enc_ctrl.gains.iter()).take(nb_subfr) {
        *dst = silk_float2int(src * Q16_SCALE.recip());
        assert!(*dst > 0, "quantiser gain must stay positive");
    }

    let mut pred_coef_q12 = [[0i16; MAX_LPC_ORDER]; 2];
    for (dst_row, src_row) in pred_coef_q12
        .iter_mut()
        .zip(enc_ctrl.pred_coef.iter())
    {
        for (dst, &src) in dst_row.iter_mut().zip(src_row.iter()).take(predict_order) {
            *dst = silk_float2int(src * Q12_SCALE.recip()) as i16;
        }
    }

    let mut ltp_coef_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    for (dst, &src) in ltp_coef_q14
        .iter_mut()
        .zip(enc_ctrl.ltp_coef.iter())
        .take(nb_subfr * LTP_ORDER)
    {
        *dst = silk_float2int(src * Q14_SCALE.recip()) as i16;
    }

    let mut ar_q13 = [0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
    for subfr in 0..nb_subfr {
        let base = subfr * MAX_SHAPE_LPC_ORDER;
        let src_base = subfr * MAX_SHAPE_LPC_ORDER;
        for j in 0..shaping_order {
            ar_q13[base + j] =
                silk_float2int(enc_ctrl.ar[src_base + j] * Q14_SCALE.recip() / 2.0) as i16;
        }
    }

    let mut lf_shp_q14 = [0i32; MAX_NB_SUBFR];
    let mut tilt_q14 = [0i32; MAX_NB_SUBFR];
    let mut harm_shape_gain_q14 = [0i32; MAX_NB_SUBFR];
    for i in 0..nb_subfr {
        let ar_q14 = silk_float2int(enc_ctrl.lf_ar_shp[i] * Q14_SCALE.recip());
        let ma_q14 = silk_float2int(enc_ctrl.lf_ma_shp[i] * Q14_SCALE.recip());
        lf_shp_q14[i] = (ar_q14 << 16) | (ma_q14 & 0xFFFF);
        tilt_q14[i] = silk_float2int(enc_ctrl.tilt[i] * Q14_SCALE.recip());
        harm_shape_gain_q14[i] =
            silk_float2int(enc_ctrl.harm_shape_gain[i] * Q14_SCALE.recip());
    }

    let lambda_q10 = silk_float2int(enc_ctrl.lambda * 1024.0);
    let ltp_scale_q14: i32 = if matches!(indices.signal_type, FrameSignalType::Voiced) {
        let idx = usize::try_from(indices.ltp_scale_index).expect("ltp_scale_index fits in usize");
        i32::from(SILK_LTPSCALES_TABLE_Q14[idx])
    } else {
        0
    };

    if common.n_states_delayed_decision > 1 || common.warping_q16 > 0 {
        silk_nsq_del_dec(
            common,
            nsq,
            indices,
            &x16[..frame_length],
            pulses,
            &pred_coef_q12[0][..predict_order],
            &ltp_coef_q14[..nb_subfr * LTP_ORDER],
            &ar_q13[..nb_subfr * MAX_SHAPE_LPC_ORDER],
            &harm_shape_gain_q14[..nb_subfr],
            &tilt_q14[..nb_subfr],
            &lf_shp_q14[..nb_subfr],
            &gains_q16[..nb_subfr],
            &enc_ctrl.pitch_l[..nb_subfr],
            lambda_q10,
            ltp_scale_q14,
        );
    } else {
        silk_nsq(
            common,
            nsq,
            indices,
            &x16[..frame_length],
            pulses,
            &pred_coef_q12[0][..predict_order],
            &ltp_coef_q14[..nb_subfr * LTP_ORDER],
            &ar_q13[..nb_subfr * MAX_SHAPE_LPC_ORDER],
            &harm_shape_gain_q14[..nb_subfr],
            &tilt_q14[..nb_subfr],
            &lf_shp_q14[..nb_subfr],
            &gains_q16[..nb_subfr],
            &enc_ctrl.pitch_l[..nb_subfr],
            lambda_q10,
            ltp_scale_q14,
        );
    }
}

/// Floating-point wrapper for the LTP gain quantiser.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn silk_quant_ltp_gains_flp(
    b: &mut [f32],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_db: &mut f32,
    xx: &[f32],
    x_x: &[f32],
    subfr_len: i32,
    nb_subfr: usize,
    _arch: i32,
) {
    assert_eq!(b.len(), nb_subfr * LTP_ORDER);
    assert_eq!(cbk_index.len(), nb_subfr);
    assert_eq!(xx.len(), nb_subfr * LTP_ORDER * LTP_ORDER);
    assert_eq!(x_x.len(), nb_subfr * LTP_ORDER);
    assert!(
        nb_subfr <= MAX_NB_SUBFR,
        "nb_subfr exceeds MAX_NB_SUBFR"
    );

    let mut b_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    let mut xx_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
    let mut x_x_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER];

    for (dst, &src) in xx_q17.iter_mut().zip(xx.iter()).take(xx.len()) {
        *dst = silk_float2int(src * 131_072.0);
    }
    for (dst, &src) in x_x_q17.iter_mut().zip(x_x.iter()).take(x_x.len()) {
        *dst = silk_float2int(src * 131_072.0);
    }

    let mut pred_gain_db_q7 = 0;
    silk_quant_ltp_gains(
        &mut b_q14[..nb_subfr * LTP_ORDER],
        &mut cbk_index[..nb_subfr],
        periodicity_index,
        sum_log_gain_q7,
        &mut pred_gain_db_q7,
        &xx_q17[..nb_subfr * LTP_ORDER * LTP_ORDER],
        &x_x_q17[..nb_subfr * LTP_ORDER],
        subfr_len,
        nb_subfr,
    );

    for (dst, &src) in b.iter_mut().zip(b_q14.iter()).take(nb_subfr * LTP_ORDER) {
        *dst = f32::from(src) * Q14_SCALE;
    }
    *pred_gain_db = (pred_gain_db_q7 as f32) * (1.0 / 128.0);
}
