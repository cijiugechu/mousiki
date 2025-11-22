//! Port of `silk_noise_shape_analysis_FLP`.
//!
//! Mirrors `silk/float/noise_shape_analysis_FLP.c`, deriving the floating-point
//! shaping filters, per-subframe gains, and tilt/harmonic controls used by the
//! FLP encoder path.
use crate::silk::apply_sine_window_flp::apply_sine_window_flp;
use crate::silk::autocorrelation_flp::autocorrelation;
use crate::silk::bwexpander_flp::bwexpander;
use crate::silk::encoder::control_flp::EncoderControlFlp;
use crate::silk::encoder::state::{SHAPE_LPC_WIN_MAX, SUB_FRAME_LENGTH_MS};
use crate::silk::encoder::state_flp::EncoderStateFlp;
use crate::silk::energy_flp::energy;
use crate::silk::k2a_flp::k2a_flp;
use crate::silk::schur_flp::silk_schur_flp;
use crate::silk::sigproc_flp::{silk_abs_float, silk_log2, silk_sigmoid};
use crate::silk::tuning_parameters::{
    BANDWIDTH_EXPANSION, BG_SNR_DECR_DB, ENERGY_VARIATION_THRESHOLD_QNT_OFFSET,
    FIND_PITCH_WHITE_NOISE_FRACTION, HARM_HP_NOISE_COEF, HARM_SNR_INCR_DB, HARMONIC_SHAPING,
    HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING, HP_NOISE_COEF, LOW_FREQ_SHAPING,
    LOW_QUALITY_LOW_FREQ_SHAPING_DECR, SHAPE_WHITE_NOISE_FRACTION, SUBFR_SMTH_COEF,
};
use crate::silk::warped_autocorrelation_flp::warped_autocorrelation_flp;
use crate::silk::{
    FrameQuantizationOffsetType, FrameSignalType, MAX_NB_SUBFR, MAX_SHAPE_LPC_ORDER,
};
use libm::{powf, sqrtf};

const MIN_QGAIN_DB: f32 = 2.0;
const USE_HARM_SHAPING: bool = true;

fn warped_gain(coefs: &[f32], lambda: f32, order: usize) -> f32 {
    assert!(order > 0, "order must be positive");
    assert!(
        coefs.len() >= order,
        "coefficient slice must contain at least {order} entries"
    );

    let lambda = -lambda;
    let mut gain = coefs[order - 1];
    for &coef in coefs[..order - 1].iter().rev() {
        gain = lambda * gain + coef;
    }
    1.0 / (1.0 - lambda * gain)
}

fn warped_true2monic_coefs(coefs: &mut [f32], lambda: f32, limit: f32, order: usize) {
    assert!(order > 0, "order must be positive");
    assert!(
        coefs.len() >= order,
        "coefficient slice must contain at least {order} entries"
    );

    let lambda = -lambda;

    for i in (1..order).rev() {
        coefs[i - 1] -= lambda * coefs[i];
    }
    let mut gain = (1.0 - lambda * lambda) / (1.0 + lambda * coefs[0]);
    for coef in coefs.iter_mut().take(order) {
        *coef *= gain;
    }

    for iter in 0..10 {
        let mut maxabs = -1.0f32;
        let mut ind = 0;
        for (i, &coef) in coefs.iter().take(order).enumerate() {
            let tmp = silk_abs_float(coef);
            if tmp > maxabs {
                maxabs = tmp;
                ind = i;
            }
        }
        if maxabs <= limit {
            return;
        }

        for i in 1..order {
            coefs[i - 1] += lambda * coefs[i];
        }
        gain = 1.0 / gain;
        for coef in coefs.iter_mut().take(order) {
            *coef *= gain;
        }

        let chirp = 0.99 - (0.8 + 0.1 * iter as f32) * (maxabs - limit)
            / (maxabs * (ind as f32 + 1.0));
        bwexpander(&mut coefs[..order], chirp);

        for i in (1..order).rev() {
            coefs[i - 1] -= lambda * coefs[i];
        }
        gain = (1.0 - lambda * lambda) / (1.0 + lambda * coefs[0]);
        for coef in coefs.iter_mut().take(order) {
            *coef *= gain;
        }
    }
    panic!("failed to clamp warped coefficients within the expected iteration budget");
}

fn limit_coefs(coefs: &mut [f32], limit: f32, order: usize) {
    assert!(order > 0, "order must be positive");
    assert!(
        coefs.len() >= order,
        "coefficient slice must contain at least {order} entries"
    );

    for iter in 0..10 {
        let mut maxabs = -1.0f32;
        let mut ind = 0;
        for (i, &coef) in coefs.iter().take(order).enumerate() {
            let tmp = silk_abs_float(coef);
            if tmp > maxabs {
                maxabs = tmp;
                ind = i;
            }
        }
        if maxabs <= limit {
            return;
        }

        let chirp = 0.99 - (0.8 + 0.1 * iter as f32) * (maxabs - limit)
            / (maxabs * (ind as f32 + 1.0));
        bwexpander(&mut coefs[..order], chirp);
    }
    panic!("failed to clamp coefficients within the expected iteration budget");
}

/// Mirrors `silk_noise_shape_analysis_FLP`.
#[allow(clippy::too_many_lines)]
pub fn noise_shape_analysis_flp(
    encoder: &mut EncoderStateFlp,
    control: &mut EncoderControlFlp,
    pitch_res: &[f32],
    x: &[f32],
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

    let mut snr_adj_db = encoder.common.snr_db_q7 as f32 * (1.0 / 128.0);

    control.input_quality = 0.5
        * (encoder.common.input_quality_bands_q15[0] as f32
            + encoder.common.input_quality_bands_q15[1] as f32)
        * (1.0 / 32768.0);
    control.coding_quality = silk_sigmoid(0.25 * (snr_adj_db - 20.0));

    if !encoder.common.use_cbr {
        let b = 1.0 - encoder.common.speech_activity_q8 as f32 * (1.0 / 256.0);
        snr_adj_db -= BG_SNR_DECR_DB
            * control.coding_quality
            * (0.5 + 0.5 * control.input_quality)
            * b
            * b;
    }

    if matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced) {
        snr_adj_db += HARM_SNR_INCR_DB * encoder.ltp_corr;
        encoder.common.indices.quant_offset_type = FrameQuantizationOffsetType::Low;
    } else {
        let n_samples = 2 * encoder.common.fs_khz as usize;
        let n_segs = SUB_FRAME_LENGTH_MS * nb_subframes / 2;
        let mut pitch_res_ptr = 0usize;
        let mut energy_variation = 0.0f32;
        let mut log_energy_prev = 0.0f32;

        for k in 0..n_segs {
            assert!(
                pitch_res_ptr + n_samples <= pitch_res.len(),
                "pitch residual length is insufficient for the requested analysis window"
            );
            let nrg =
                n_samples as f64 + energy(&pitch_res[pitch_res_ptr..pitch_res_ptr + n_samples]);
            let log_energy = silk_log2(nrg);
            if k > 0 {
                energy_variation += silk_abs_float(log_energy - log_energy_prev);
            }
            log_energy_prev = log_energy;
            pitch_res_ptr += n_samples;
        }

        if energy_variation > ENERGY_VARIATION_THRESHOLD_QNT_OFFSET * (n_segs as f32 - 1.0) {
            encoder.common.indices.quant_offset_type = FrameQuantizationOffsetType::Low;
        } else {
            encoder.common.indices.quant_offset_type = FrameQuantizationOffsetType::High;
        }
    }

    let strength = FIND_PITCH_WHITE_NOISE_FRACTION * control.pred_gain;
    let bwexp = BANDWIDTH_EXPANSION / (1.0 + strength * strength);

    let warping = if encoder.common.warping_q16 > 0 {
        encoder.common.warping_q16 as f32 * (1.0 / 65536.0) + 0.01 * control.coding_quality
    } else {
        0.0
    };

    let mut x_ptr = 0usize;
    let mut x_windowed = [0.0f32; SHAPE_LPC_WIN_MAX as usize];
    let mut auto_corr = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];
    let mut rc = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];

    for k in 0..nb_subframes {
        let flat_part = (encoder.common.fs_khz * 3) as usize;
        let slope_part = ((encoder.common.shape_win_length - encoder.common.fs_khz * 3) / 2)
            as usize;

        apply_sine_window_flp(
            &mut x_windowed[..slope_part],
            &x[x_ptr..x_ptr + slope_part],
            1,
        );
        let mut shift = slope_part;
        x_windowed[shift..shift + flat_part]
            .copy_from_slice(&x[x_ptr + shift..x_ptr + shift + flat_part]);
        shift += flat_part;
        apply_sine_window_flp(
            &mut x_windowed[shift..shift + slope_part],
            &x[x_ptr + shift..x_ptr + shift + slope_part],
            2,
        );

        x_ptr += encoder.common.subfr_length;

        auto_corr.fill(0.0);
        if encoder.common.warping_q16 > 0 {
            warped_autocorrelation_flp(
                &mut auto_corr,
                &x_windowed[..encoder.common.shape_win_length as usize],
                warping,
                lpc_order,
            );
        } else {
            autocorrelation(
                &mut auto_corr,
                &x_windowed[..encoder.common.shape_win_length as usize],
                lpc_order + 1,
            );
        }

        auto_corr[0] += auto_corr[0] * SHAPE_WHITE_NOISE_FRACTION + 1.0;

        let nrg = silk_schur_flp(&mut rc[..lpc_order], &auto_corr, lpc_order);
        debug_assert!(nrg >= 0.0);
        control.gains[k] = sqrtf(nrg);

        let ar_offset = k * MAX_SHAPE_LPC_ORDER;
        k2a_flp(
            &mut control.ar[ar_offset..ar_offset + lpc_order],
            &rc[..lpc_order],
        );

        if encoder.common.warping_q16 > 0 {
            control.gains[k] *= warped_gain(&control.ar[ar_offset..], warping, lpc_order);
        }

        bwexpander(&mut control.ar[ar_offset..ar_offset + lpc_order], bwexp);

        if encoder.common.warping_q16 > 0 {
            warped_true2monic_coefs(
                &mut control.ar[ar_offset..],
                warping,
                3.999,
                lpc_order,
            );
        } else {
            limit_coefs(&mut control.ar[ar_offset..], 3.999, lpc_order);
        }
    }

    let gain_mult = powf(2.0, -0.16 * snr_adj_db);
    let gain_add = powf(2.0, 0.16 * MIN_QGAIN_DB);
    for gain in control.gains.iter_mut().take(nb_subframes) {
        *gain = *gain * gain_mult + gain_add;
    }

    let mut strength = LOW_FREQ_SHAPING
        * (1.0
            + LOW_QUALITY_LOW_FREQ_SHAPING_DECR
                * (encoder.common.input_quality_bands_q15[0] as f32 * (1.0 / 32768.0) - 1.0));
    strength *= encoder.common.speech_activity_q8 as f32 * (1.0 / 256.0);

    let tilt: f32;
    if matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced) {
        for (k, pitch_lag) in control
            .pitch_l
            .iter()
            .take(nb_subframes)
            .copied()
            .enumerate()
        {
            assert_ne!(pitch_lag, 0, "pitch lag must be non-zero");
            let b = 0.2 / encoder.common.fs_khz as f32 + 3.0 / pitch_lag as f32;
            control.lf_ma_shp[k] = -1.0 + b;
            control.lf_ar_shp[k] = 1.0 - b - b * strength;
        }

        tilt = -HP_NOISE_COEF
            - (1.0 - HP_NOISE_COEF)
                * HARM_HP_NOISE_COEF
                * encoder.common.speech_activity_q8 as f32
                * (1.0 / 256.0);
    } else {
        let b = 1.3 / encoder.common.fs_khz as f32;
        control.lf_ma_shp[0] = -1.0 + b;
        control.lf_ar_shp[0] = 1.0 - b - b * strength * 0.6;
        for k in 1..nb_subframes {
            control.lf_ma_shp[k] = control.lf_ma_shp[0];
            control.lf_ar_shp[k] = control.lf_ar_shp[0];
        }
        tilt = -HP_NOISE_COEF;
    }

    let harmonic_shape_gain = if USE_HARM_SHAPING
        && matches!(encoder.common.indices.signal_type, FrameSignalType::Voiced)
    {
        let mut gain = HARMONIC_SHAPING
            + HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING
                * (1.0 - (1.0 - control.coding_quality) * control.input_quality);
        debug_assert!(encoder.ltp_corr >= 0.0);
        gain *= sqrtf(encoder.ltp_corr.max(0.0));
        gain
    } else {
        0.0
    };

    for k in 0..nb_subframes {
        encoder.shape_state.harm_shape_gain_smth +=
            SUBFR_SMTH_COEF * (harmonic_shape_gain - encoder.shape_state.harm_shape_gain_smth);
        control.harm_shape_gain[k] = encoder.shape_state.harm_shape_gain_smth;

        encoder.shape_state.tilt_smth += SUBFR_SMTH_COEF * (tilt - encoder.shape_state.tilt_smth);
        control.tilt[k] = encoder.shape_state.tilt_smth;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use super::noise_shape_analysis_flp;
    use crate::silk::encoder::control_flp::EncoderControlFlp;
    use crate::silk::encoder::state_flp::EncoderStateFlp;
    use crate::silk::sigproc_flp::silk_sigmoid;
    use crate::silk::tuning_parameters::{
        HARMONIC_SHAPING, HARM_SNR_INCR_DB, HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING,
        HP_NOISE_COEF, SUBFR_SMTH_COEF,
    };
    use crate::silk::{FrameSignalType, MAX_NB_SUBFR};
    use libm::{powf, sqrtf};

    #[test]
    fn noise_shape_analysis_flp_updates_shaping_state() {
        let mut encoder = EncoderStateFlp::default();
        encoder.common.use_cbr = true;
        encoder.common.indices.signal_type = FrameSignalType::Voiced;
        encoder.common.shaping_lpc_order = 1;
        encoder.common.warping_q16 = 0;
        encoder.common.snr_db_q7 = 0;
        encoder.common.input_quality_bands_q15[0] = 0;
        encoder.common.input_quality_bands_q15[1] = 0;
        encoder.ltp_corr = 0.25;

        let mut control = EncoderControlFlp::default();
        control.pred_gain = 0.0;
        control.pitch_l[..encoder.common.nb_subfr].fill(80);

        let frame_length = encoder.common.frame_length;
        let la_shape = encoder.common.la_shape as usize;
        let expected_signal_len = frame_length + 2 * la_shape;
        let pitch_res = vec![0.0f32; frame_length];
        let x = vec![0.0f32; expected_signal_len];

        noise_shape_analysis_flp(&mut encoder, &mut control, &pitch_res, &x);

        let mut snr_adj_db = encoder.common.snr_db_q7 as f32 * (1.0 / 128.0);
        snr_adj_db += HARM_SNR_INCR_DB * encoder.ltp_corr;
        let gain_mult = powf(2.0, -0.16 * snr_adj_db);
        let gain_add = powf(2.0, 0.16 * super::MIN_QGAIN_DB);
        let expected_gain = gain_mult + gain_add;

        for gain in control.gains.iter().take(encoder.common.nb_subfr) {
            assert!(
                (gain - expected_gain).abs() < 1e-6,
                "gain {gain} deviates from expected {expected_gain}"
            );
        }

        let expected_lf_ma = -0.95;
        let expected_lf_ar = 0.95;
        for k in 0..encoder.common.nb_subfr {
            assert!(
                (control.lf_ma_shp[k] - expected_lf_ma).abs() < 1e-6,
                "lf_ma_shp[{k}] = {} differs from {expected_lf_ma}",
                control.lf_ma_shp[k]
            );
            assert!(
                (control.lf_ar_shp[k] - expected_lf_ar).abs() < 1e-6,
                "lf_ar_shp[{k}] = {} differs from {expected_lf_ar}",
                control.lf_ar_shp[k]
            );
        }

        let coding_quality = silk_sigmoid(-5.0);
        let target_harm = (HARMONIC_SHAPING
            + HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING * (1.0 - coding_quality))
            * sqrtf(encoder.ltp_corr);
        let target_tilt = -HP_NOISE_COEF;

        let mut expected_harm = 0.0;
        let mut expected_tilt = 0.0;
        const TOLERANCE: f32 = 1e-3;
        for k in 0..encoder.common.nb_subfr {
            expected_harm += SUBFR_SMTH_COEF * (target_harm - expected_harm);
            expected_tilt += SUBFR_SMTH_COEF * (target_tilt - expected_tilt);
            assert!(
                (control.harm_shape_gain[k] - expected_harm).abs() < TOLERANCE,
                "harmonic gain[{k}] = {} deviates from {expected_harm}",
                control.harm_shape_gain[k]
            );
            assert!(
                (control.tilt[k] - expected_tilt).abs() < TOLERANCE,
                "tilt[{k}] = {} deviates from {expected_tilt}",
                control.tilt[k]
            );
        }

        assert!(
            encoder.shape_state.harm_shape_gain_smth == control.harm_shape_gain[encoder.common.nb_subfr - 1]
        );
        assert!(
            encoder.shape_state.tilt_smth == control.tilt[encoder.common.nb_subfr - 1]
        );

        for coef in control.ar.iter().take(MAX_NB_SUBFR * super::MAX_SHAPE_LPC_ORDER) {
            assert!(coef.abs() <= 3.999);
        }
    }
}
