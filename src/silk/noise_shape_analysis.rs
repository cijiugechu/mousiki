//! Port of `silk_noise_shape_analysis_FIX`.
//!
//! Mirrors `silk/fixed/noise_shape_analysis_FIX.c`, deriving the LPC shaping
//! filters, per-subframe gains, and spectral tilt parameters that drive the
//! noise-shaping quantiser.
use crate::silk::apply_sine_window::apply_sine_window;
use crate::silk::autocorr::autocorr;
use crate::silk::bwexpander_32::bwexpander_32;
use crate::silk::encoder::control::EncoderControl;
use crate::silk::encoder::state::{EncoderChannelState, SHAPE_LPC_WIN_MAX};
use crate::silk::k2a_q16::k2a_q16;
use crate::silk::lpc_fit::lpc_fit;
use crate::silk::lpc_inv_pred_gain::inverse32_varq;
use crate::silk::lin2log::lin2log;
use crate::silk::log2lin::log2lin;
use crate::silk::schur64::schur64;
use crate::silk::sum_sqr_shift::sum_sqr_shift;
use crate::silk::tuning_parameters::{
    BANDWIDTH_EXPANSION, BG_SNR_DECR_DB, ENERGY_VARIATION_THRESHOLD_QNT_OFFSET,
    FIND_PITCH_WHITE_NOISE_FRACTION, HARMONIC_SHAPING, HARM_HP_NOISE_COEF, HARM_SNR_INCR_DB,
    HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING, HP_NOISE_COEF, LOW_FREQ_SHAPING,
    LOW_QUALITY_LOW_FREQ_SHAPING_DECR, SHAPE_WHITE_NOISE_FRACTION, SUBFR_SMTH_COEF,
};
use crate::silk::warped_autocorrelation::{warped_autocorrelation, MAX_SHAPE_LPC_ORDER};
use crate::silk::{FrameQuantizationOffsetType, FrameSignalType, MAX_NB_SUBFR};

const MIN_QGAIN_DB: i32 = 2;
const USE_HARM_SHAPING: bool = true;

const BG_SNR_DECR_DB_Q7: i32 = (BG_SNR_DECR_DB * (1 << 7) as f32 + 0.5) as i32;
const HARM_SNR_INCR_DB_Q8: i32 = (HARM_SNR_INCR_DB * (1 << 8) as f32 + 0.5) as i32;
const ENERGY_VARIATION_THRESHOLD_Q7: i32 =
    (ENERGY_VARIATION_THRESHOLD_QNT_OFFSET * (1 << 7) as f32 + 0.5) as i32;
const FIND_PITCH_WHITE_NOISE_FRACTION_Q16: i32 =
    (FIND_PITCH_WHITE_NOISE_FRACTION * (1 << 16) as f32 + 0.5) as i32;
const BANDWIDTH_EXPANSION_Q16: i32 = (BANDWIDTH_EXPANSION * (1 << 16) as f32 + 0.5) as i32;
const SHAPE_WHITE_NOISE_FRACTION_Q20: i32 =
    (SHAPE_WHITE_NOISE_FRACTION * (1 << 20) as f32 + 0.5) as i32;
const LOW_FREQ_SHAPING_Q4: i32 = (LOW_FREQ_SHAPING * (1 << 4) as f32 + 0.5) as i32;
const LOW_QUALITY_LOW_FREQ_SHAPING_DECR_Q13: i32 =
    (LOW_QUALITY_LOW_FREQ_SHAPING_DECR * (1 << 13) as f32 + 0.5) as i32;
const HP_NOISE_COEF_Q16: i32 = (HP_NOISE_COEF * (1 << 16) as f32 + 0.5) as i32;
const HARM_HP_NOISE_COEF_Q24: i32 = (HARM_HP_NOISE_COEF * (1 << 24) as f32 + 0.5) as i32;
const HARMONIC_SHAPING_Q16: i32 = (HARMONIC_SHAPING * (1 << 16) as f32 + 0.5) as i32;
const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING_Q16: i32 =
    (HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING * (1 << 16) as f32 + 0.5) as i32;
const SUBFR_SMTH_COEF_Q16: i32 = (SUBFR_SMTH_COEF * (1 << 16) as f32 + 0.5) as i32;
const ONE_Q16: i32 = 1 << 16;
const ONE_Q14: i32 = 1 << 14;

/// Mirrors `silk_noise_shape_analysis_FIX`.
#[allow(clippy::too_many_arguments)]
pub fn noise_shape_analysis(
    encoder: &mut EncoderChannelState,
    control: &mut EncoderControl,
    pitch_res: &[i16],
    x: &[i16],
    arch: i32,
) {
    let nb_subframes = encoder.common.nb_subfr;
    assert!(
        nb_subframes == MAX_NB_SUBFR || nb_subframes == MAX_NB_SUBFR / 2,
        "noise-shape analysis expects 2 or 4 subframes"
    );

    let lpc_order = encoder.common.shaping_lpc_order as usize;
    assert!(
        (1..=MAX_SHAPE_LPC_ORDER).contains(&lpc_order),
        "shaping_lpc_order must be between 1 and {MAX_SHAPE_LPC_ORDER}"
    );
    assert!(
        encoder.common.shape_win_length <= SHAPE_LPC_WIN_MAX,
        "shape window exceeds maximum"
    );

    let la_shape = encoder.common.la_shape as usize;
    let expected_signal_len = encoder.common.frame_length + 2 * la_shape;
    assert!(
        x.len() >= expected_signal_len,
        "x must include la_shape lookahead on both sides of the frame"
    );
    assert!(
        pitch_res.len() >= encoder.common.frame_length,
        "pitch residual must span the frame"
    );

    let mut snr_adj_db_q7 = encoder.common.snr_db_q7;

    control.input_quality_q14 = rshift(encoder.common.input_quality_bands_q15[0]
        + encoder.common.input_quality_bands_q15[1], 2);
    control.coding_quality_q14 =
        rshift(sigm_q15(rshift_round(snr_adj_db_q7 - (20 << 7), 4)), 1);

    if !encoder.common.use_cbr {
        let mut b_q8 = ONE_Q8 - encoder.common.speech_activity_q8;
        b_q8 = smulwb(b_q8 << 8, b_q8);
        let quality_mul = smulwb(ONE_Q14 + control.input_quality_q14, control.coding_quality_q14);
        let snr_term = smulbb(-(BG_SNR_DECR_DB_Q7) >> 5, b_q8);
        snr_adj_db_q7 = smlawb(snr_adj_db_q7, snr_term, quality_mul);
    }

    if encoder.common.indices.signal_type == FrameSignalType::Voiced {
        snr_adj_db_q7 = smlawb(snr_adj_db_q7, HARM_SNR_INCR_DB_Q8, encoder.ltp_corr_q15);
        encoder.common.indices.quant_offset_type = FrameQuantizationOffsetType::Low;
    } else {
        let tmp = smlawb(
            (6.0_f32 * (1 << 9) as f32 + 0.5) as i32,
            -(0.4_f32 * (1 << 18) as f32 + 0.5) as i32,
            encoder.common.snr_db_q7,
        );
        snr_adj_db_q7 = smlawb(
            snr_adj_db_q7,
            tmp,
            ONE_Q14 - control.input_quality_q14,
        );

        let n_samples = (encoder.common.fs_khz << 1) as usize;
        let n_segs = nb_subframes * crate::silk::encoder::state::SUB_FRAME_LENGTH_MS / 2;
        let mut pitch_res_ptr = 0usize;
        let mut energy_variation_q7 = 0;
        let mut log_energy_prev_q7 = 0;

        for _ in 0..n_segs {
            let segment = &pitch_res[pitch_res_ptr..pitch_res_ptr + n_samples];
            let (mut nrg, scale) = sum_sqr_shift(segment);
            nrg += n_samples as i32 >> scale;

            let log_energy_q7 = lin2log(nrg);
            if pitch_res_ptr > 0 {
                energy_variation_q7 += (log_energy_q7 - log_energy_prev_q7).abs();
            }
            log_energy_prev_q7 = log_energy_q7;
            pitch_res_ptr += n_samples;
        }

        if energy_variation_q7 > ENERGY_VARIATION_THRESHOLD_Q7 * (n_segs as i32 - 1) {
            encoder.common.indices.quant_offset_type = FrameQuantizationOffsetType::Low;
        } else {
            encoder.common.indices.quant_offset_type = FrameQuantizationOffsetType::High;
        }
    }

    let strength_q16 =
        smulwb(control.pred_gain_q16, FIND_PITCH_WHITE_NOISE_FRACTION_Q16);
    let bwexp_q16 = div32_varq(
        BANDWIDTH_EXPANSION_Q16,
        smlaww(ONE_Q16, strength_q16, strength_q16),
        16,
    );

    let warping_q16 = if encoder.common.warping_q16 > 0 {
        smlawb(
            encoder.common.warping_q16,
            control.coding_quality_q14,
            (0.01_f32 * (1 << 18) as f32 + 0.5) as i32,
        )
    } else {
        0
    };

    let mut x_ptr = 0usize;
    let mut ar_q24 = [0i32; MAX_SHAPE_LPC_ORDER];
    let mut auto_corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut refl_coef_q16 = [0i32; MAX_SHAPE_LPC_ORDER];
    let mut x_windowed = [0i16; SHAPE_LPC_WIN_MAX as usize];

    for k in 0..nb_subframes {
        let flat_part = encoder.common.fs_khz * 3;
        let slope_part = (encoder.common.shape_win_length - flat_part) >> 1;

        let slope = slope_part as usize;
        apply_sine_window(
            &mut x_windowed[..slope],
            &x[x_ptr..x_ptr + slope],
            1,
        );
        let mut shift = slope;
        let flat = flat_part as usize;
        x_windowed[shift..shift + flat]
            .copy_from_slice(&x[x_ptr + shift..x_ptr + shift + flat]);
        shift += flat;
        apply_sine_window(
            &mut x_windowed[shift..shift + slope],
            &x[x_ptr + shift..x_ptr + shift + slope],
            2,
        );

        x_ptr += encoder.common.subfr_length;

        let scale = if warping_q16 > 0 {
            warped_autocorrelation(
                &mut auto_corr,
                &x_windowed[..encoder.common.shape_win_length as usize],
                warping_q16,
                lpc_order,
            )
        } else {
            autocorr(
                &mut auto_corr,
                &x_windowed[..encoder.common.shape_win_length as usize],
                lpc_order + 1,
                arch,
            )
        };

        auto_corr[0] = auto_corr[0].wrapping_add(
            smulwb(auto_corr[0] >> 4, SHAPE_WHITE_NOISE_FRACTION_Q20).max(1),
        );

        let mut nrg = schur64(&mut refl_coef_q16[..lpc_order], &auto_corr, lpc_order);

        k2a_q16(&mut ar_q24, &refl_coef_q16[..lpc_order]);

        let mut qnrg = -scale;
        if (qnrg & 1) != 0 {
            qnrg -= 1;
            nrg >>= 1;
        }
        let tmp32 = sqrt_approx(nrg);
        qnrg >>= 1;
        control.gains_q16[k] = lshift_sat32(tmp32, 16 - qnrg);

        if warping_q16 > 0 {
            let gain_mult_q16 = warped_gain(&ar_q24[..lpc_order], warping_q16, lpc_order);
            if control.gains_q16[k] < ONE_Q16 / 4 {
                control.gains_q16[k] = smulww(control.gains_q16[k], gain_mult_q16);
            } else {
                control.gains_q16[k] = smulww(rshift_round(control.gains_q16[k], 1), gain_mult_q16);
                control.gains_q16[k] = if control.gains_q16[k] >= (i32::MAX >> 1) {
                    i32::MAX
                } else {
                    control.gains_q16[k] << 1
                };
            }
        }

        bwexpander_32(&mut ar_q24[..lpc_order], bwexp_q16);

        if warping_q16 > 0 {
            limit_warped_coefs(
                &mut ar_q24[..lpc_order],
                warping_q16,
                (3.999_f32 * (1 << 24) as f32 + 0.5) as i32,
                lpc_order,
            );
            for (dst, &coef) in control.ar_q13[k * MAX_SHAPE_LPC_ORDER
                ..k * MAX_SHAPE_LPC_ORDER + lpc_order]
                .iter_mut()
                .zip(&ar_q24[..lpc_order])
            {
                *dst = sat16(rshift_round(coef, 11));
            }
        } else {
            lpc_fit(
                &mut control.ar_q13[k * MAX_SHAPE_LPC_ORDER
                    ..k * MAX_SHAPE_LPC_ORDER + lpc_order],
                &mut ar_q24[..lpc_order],
                13,
                24,
            );
        }
    }

    let gain_mult_q16 = log2lin(-smlawb(-(16 << 7), snr_adj_db_q7, (0.16_f32 * (1 << 16) as f32 + 0.5) as i32));
    let gain_add_q16 =
        log2lin(smlawb(16 << 7, MIN_QGAIN_DB << 7, (0.16_f32 * (1 << 16) as f32 + 0.5) as i32));
    for gain in control.gains_q16.iter_mut().take(nb_subframes) {
        *gain = smulww(*gain, gain_mult_q16);
        *gain = add_pos_sat32(*gain, gain_add_q16);
    }

    let mut strength_q16 = mul(
        LOW_FREQ_SHAPING_Q4,
        smlawb(
            1 << 12,
            LOW_QUALITY_LOW_FREQ_SHAPING_DECR_Q13,
            encoder.common.input_quality_bands_q15[0] - (1 << 15),
        ),
    );
    strength_q16 = rshift(strength_q16 * encoder.common.speech_activity_q8, 8);

    let (tilt_q16, lf_shp_q14) = if encoder.common.indices.signal_type == FrameSignalType::Voiced {
        let fs_khz_inv =
            div32_16((0.2_f32 * (1 << 14) as f32 + 0.5) as i32, encoder.common.fs_khz);
        let mut lf = [0i32; MAX_NB_SUBFR];
        for (k, lf_slot) in lf.iter_mut().take(nb_subframes).enumerate() {
            let b_q14 = fs_khz_inv
                + div32_16(
                    (3.0_f32 * (1 << 14) as f32 + 0.5) as i32,
                    control.pitch_l[k],
                );
            let packed = ((ONE_Q14
                - b_q14
                - smulwb(
                    strength_q16,
                    smulwb((0.6_f32 * (1 << 16) as f32 + 0.5) as i32, b_q14),
                ))
                << 16)
                | ((b_q14 - ONE_Q14) & 0xFFFF);
            *lf_slot = packed;
        }
        let tilt = -HP_NOISE_COEF_Q16
            - smulwb(
                ONE_Q16 - HP_NOISE_COEF_Q16,
                smulwb(HARM_HP_NOISE_COEF_Q24, encoder.common.speech_activity_q8),
            );
        (tilt, lf)
    } else {
        let b_q14 = div32_16(21299, encoder.common.fs_khz);
        let packed = ((ONE_Q14
            - b_q14
            - smulwb(
                strength_q16,
                smulwb((0.6_f32 * (1 << 16) as f32 + 0.5) as i32, b_q14),
            ))
            << 16)
            | ((b_q14 - ONE_Q14) & 0xFFFF);
        let lf = [packed; MAX_NB_SUBFR];
        (-HP_NOISE_COEF_Q16, lf)
    };

    let mut harm_shape_gain_q16 = 0;
    if USE_HARM_SHAPING && encoder.common.indices.signal_type == FrameSignalType::Voiced {
        harm_shape_gain_q16 = smlawb(
            HARMONIC_SHAPING_Q16,
            ONE_Q16
                - smulwb(
                    (1 << 18) - (control.coding_quality_q14 << 4),
                    control.input_quality_q14,
                ),
            HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING_Q16,
        );
        harm_shape_gain_q16 = smulwb(
            harm_shape_gain_q16 << 1,
            sqrt_approx(encoder.ltp_corr_q15 << 15),
        );
    }

    for (k, lf) in lf_shp_q14.iter().enumerate().take(MAX_NB_SUBFR) {
        encoder.shape_state.harm_shape_gain_smth_q16 = smlawb(
            encoder.shape_state.harm_shape_gain_smth_q16,
            harm_shape_gain_q16 - encoder.shape_state.harm_shape_gain_smth_q16,
            SUBFR_SMTH_COEF_Q16,
        );
        encoder.shape_state.tilt_smth_q16 = smlawb(
            encoder.shape_state.tilt_smth_q16,
            tilt_q16 - encoder.shape_state.tilt_smth_q16,
            SUBFR_SMTH_COEF_Q16,
        );

        control.harm_shape_gain_q14[k] =
            rshift_round(encoder.shape_state.harm_shape_gain_smth_q16, 2);
        control.tilt_q14[k] = rshift_round(encoder.shape_state.tilt_smth_q16, 2);
        control.lf_shp_q14[k] = *lf;
    }
}

fn warped_gain(coefs_q24: &[i32], lambda_q16: i32, order: usize) -> i32 {
    let mut gain_q24 = coefs_q24[order - 1];
    let lambda = -lambda_q16;
    for &coef in coefs_q24[..order - 1].iter().rev() {
        gain_q24 = smlawb(coef, gain_q24, lambda);
    }
    gain_q24 = smlawb(1 << 24, gain_q24, -lambda);
    inverse32_varq(gain_q24, 40)
}

fn limit_warped_coefs(coefs_q24: &mut [i32], lambda_q16: i32, limit_q24: i32, order: usize) {
    let mut lambda = -lambda_q16;
    for i in (1..order).rev() {
        coefs_q24[i - 1] = smlawb(coefs_q24[i - 1], coefs_q24[i], lambda);
    }
    lambda = -lambda;
    let mut nom_q16 = smlawb(ONE_Q16, -lambda, lambda);
    let mut den_q24 = smlawb(1 << 24, coefs_q24[0], lambda);
    let mut gain_q16 = div32_varq(nom_q16, den_q24, 24);
    for coef in coefs_q24.iter_mut().take(order) {
        *coef = smulww(gain_q16, *coef);
    }
    let limit_q20 = limit_q24 >> 4;

    for iter in 0..10 {
        let (maxabs_q24, idx) = coefs_q24
            .iter()
            .take(order)
            .enumerate()
            .map(|(i, &c)| (c.abs(), i))
            .max_by(|a, b| a.0.cmp(&b.0))
            .unwrap();

        let maxabs_q20 = maxabs_q24 >> 4;
        if maxabs_q20 <= limit_q20 {
            return;
        }

        for i in 1..order {
            coefs_q24[i - 1] = smlawb(coefs_q24[i - 1], coefs_q24[i], lambda);
        }
        gain_q16 = inverse32_varq(gain_q16, 32);
        for coef in coefs_q24.iter_mut().take(order) {
            *coef = smulww(gain_q16, *coef);
        }

        let chirp_q16 = ((0.99_f32 * (1 << 16) as f32 + 0.5) as i32)
            - div32_varq(
                smulwb(
                    maxabs_q20 - limit_q20,
                    smlabb(
                        (0.8_f32 * (1 << 10) as f32 + 0.5) as i32,
                        (0.1_f32 * (1 << 10) as f32 + 0.5) as i32,
                        iter,
                    ),
                ),
                mul(maxabs_q20, (idx + 1) as i32),
                22,
            );
        bwexpander_32(&mut coefs_q24[..order], chirp_q16);

        lambda = -lambda;
        for i in (1..order).rev() {
            coefs_q24[i - 1] = smlawb(coefs_q24[i - 1], coefs_q24[i], lambda);
        }
        lambda = -lambda;
        nom_q16 = smlawb(ONE_Q16, -lambda, lambda);
        den_q24 = smlawb(1 << 24, coefs_q24[0], lambda);
        gain_q16 = div32_varq(nom_q16, den_q24, 24);
        for coef in coefs_q24.iter_mut().take(order) {
            *coef = smulww(gain_q16, *coef);
        }
    }
    panic!("limit_warped_coefs failed to converge within 10 iterations");
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from((a as i16).wrapping_mul(b as i16))
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(i32::from((b as i16).wrapping_mul(c as i16)))
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((i64::from(b) * i64::from(c as i16)) >> 16) as i32)
}

fn smlaww(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((i64::from(b) * i64::from(c)) >> 16) as i32)
}

fn div32_varq(a32: i32, b32: i32, q_res: i32) -> i32 {
    crate::silk::stereo_find_predictor::div32_varq(a32, b32, q_res)
}

fn rshift(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else {
        value >> shift
    }
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else {
        (value + (1 << (shift - 1))) >> shift
    }
}

fn lshift_sat32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        rshift_round(value, -shift)
    } else {
        let shifted = (i64::from(value)) << shift;
        if shifted > i64::from(i32::MAX) {
            i32::MAX
        } else if shifted < i64::from(i32::MIN) {
            i32::MIN
        } else {
            shifted as i32
        }
    }
}

fn sat16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn div32_16(a: i32, b: i32) -> i32 {
    assert_ne!(b, 0, "division by zero in div32_16");
    a / b
}

fn add_pos_sat32(a: i32, b: i32) -> i32 {
    let sum = i64::from(a) + i64::from(b);
    if sum > i64::from(i32::MAX) {
        i32::MAX
    } else {
        sum as i32
    }
}

fn mul(a: i32, b: i32) -> i32 {
    (i64::from(a) * i64::from(b)) as i32
}

fn sigm_q15(input_q5: i32) -> i32 {
    crate::silk::sigm_q15::sigm_q15(input_q5)
}

fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32_768 } else { 46_214 };
    y >>= lz >> 1;
    smlawb(y, y, smulbb(213, frac_q7))
}

fn clz_frac(x: i32) -> (i32, i32) {
    let ux = x as u32;
    let lz = ux.leading_zeros() as i32;
    let rotate = ((24 - lz) & 31) as u32;
    let frac = (ux.rotate_right(rotate) & 0x7f) as i32;
    (lz, frac)
}

const ONE_Q8: i32 = 1 << 8;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn with_basic_encoder(signal_type: FrameSignalType) -> (EncoderChannelState, EncoderControl) {
        let mut encoder = EncoderChannelState::default();
        encoder.common.indices.signal_type = signal_type;
        encoder.common.shaping_lpc_order = 16;
        encoder.common.shape_win_length = 240;
        encoder.common.la_shape = 80;
        encoder.common.subfr_length = 80;
        encoder.common.frame_length = 320;
        encoder.ltp_corr_q15 = 0;
        let mut control = EncoderControl::default();
        control.pitch_l.fill(40);
        (encoder, control)
    }

    #[test]
    fn quant_offset_stays_low_for_voiced_frames() {
        let (mut encoder, mut control) = with_basic_encoder(FrameSignalType::Voiced);
        let pitch_res = vec![0i16; encoder.common.frame_length];
        let x = vec![0i16; encoder.common.frame_length + 2 * encoder.common.la_shape as usize];

        noise_shape_analysis(&mut encoder, &mut control, &pitch_res, &x, 0);

        assert_eq!(
            encoder.common.indices.quant_offset_type,
            FrameQuantizationOffsetType::Low
        );
    }

    #[test]
    fn sparse_unvoiced_frames_pick_high_quant_offset() {
        let (mut encoder, mut control) = with_basic_encoder(FrameSignalType::Unvoiced);
        let pitch_res = vec![0i16; encoder.common.frame_length];
        let x = vec![0i16; encoder.common.frame_length + 2 * encoder.common.la_shape as usize];

        noise_shape_analysis(&mut encoder, &mut control, &pitch_res, &x, 0);

        assert_eq!(
            encoder.common.indices.quant_offset_type,
            FrameQuantizationOffsetType::High
        );
    }

    #[test]
    fn warped_path_populates_shaping_coefs() {
        let (mut encoder, mut control) = with_basic_encoder(FrameSignalType::Unvoiced);
        encoder.common.warping_q16 = 1 << 14;
        let mut pitch_res = vec![0i16; encoder.common.frame_length];
        for (i, slot) in pitch_res.iter_mut().enumerate() {
            *slot = (i as i16).wrapping_mul(3);
        }
        let mut x =
            vec![0i16; encoder.common.frame_length + 2 * encoder.common.la_shape as usize];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = (i as i16).wrapping_sub(50);
        }

        noise_shape_analysis(&mut encoder, &mut control, &pitch_res, &x, 0);

        let first_band = &control.ar_q13[..encoder.common.shaping_lpc_order as usize];
        assert!(
            first_band.iter().any(|&c| c != 0),
            "warped analysis should generate non-zero AR coefficients"
        );
        assert!(
            first_band
                .iter()
                .all(|&c| c != i16::MAX && c != i16::MIN),
            "AR coefficients should stay away from saturation"
        );
    }
}
