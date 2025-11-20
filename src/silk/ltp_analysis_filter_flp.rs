//! Port of `silk/float/LTP_analysis_filter_FLP.c`.
//!
//! This floating-point helper subtracts the long-term prediction (LTP)
//! contribution from the input signal and scales the residual by the inverse
//! quantisation gains, mirroring the reference SILK encoder analysis path.

use crate::silk::vq_wmat_ec::LTP_ORDER;

const LTP_CENTER: usize = LTP_ORDER / 2;

/// Mirrors `silk_LTP_analysis_filter_FLP`.
#[allow(clippy::too_many_arguments)]
pub fn ltp_analysis_filter_flp(
    ltp_res: &mut [f32],
    x: &[f32],
    x_ptr_offset: usize,
    ltp_coef: &[f32],
    pitch_l: &[i32],
    inv_gains: &[f32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let chunk = subfr_length + pre_length;
    assert!(chunk > 0, "subframe chunk must be non-zero");

    let total_samples = nb_subfr
        .checked_mul(chunk)
        .expect("ltp_res length overflow");
    assert!(
        ltp_res.len() >= total_samples,
        "ltp_res buffer must hold nb_subfr × (subfr_length + pre_length)"
    );
    assert!(
        pitch_l.len() >= nb_subfr,
        "pitchL slice must cover nb_subfr"
    );
    assert!(
        inv_gains.len() >= nb_subfr,
        "invGains slice must cover nb_subfr"
    );
    assert!(
        ltp_coef.len() >= nb_subfr * LTP_ORDER,
        "LTP coefficients slice too short"
    );

    let mut res_offset = 0;
    let mut x_ptr_idx = x_ptr_offset;

    for subfr in 0..nb_subfr {
        let pitch = pitch_l[subfr];
        assert!(pitch > 0, "pitch lag must be positive");
        let pitch_usize = pitch as usize;

        let chunk_end = x_ptr_idx.checked_add(chunk).expect("x_ptr index overflow");
        assert!(
            chunk_end <= x.len(),
            "x buffer too short for subframe {subfr}"
        );

        let lag_base = x_ptr_idx
            .checked_sub(pitch_usize)
            .expect("insufficient pitch history");
        assert!(
            lag_base >= LTP_CENTER,
            "pitch history must include {LTP_CENTER} guard samples"
        );

        let tap_ceiling = lag_base
            .checked_add(chunk - 1 + LTP_CENTER)
            .expect("lag range overflow");
        assert!(
            tap_ceiling < x.len(),
            "x buffer too short for LTP taps in subframe {subfr}"
        );

        let taps_offset = subfr * LTP_ORDER;
        let taps = &ltp_coef[taps_offset..taps_offset + LTP_ORDER];
        let inv_gain = inv_gains[subfr];
        let res_slice = &mut ltp_res[res_offset..res_offset + chunk];

        for (i, sample_out) in res_slice.iter_mut().enumerate() {
            let lag = lag_base + i;
            let prediction = taps.iter().enumerate().fold(0.0f32, |acc, (tap_idx, tap)| {
                let offset = tap_idx.abs_diff(LTP_CENTER);
                let sample_index = if tap_idx <= LTP_CENTER {
                    lag + offset
                } else {
                    lag - offset
                };
                acc + tap * x[sample_index]
            });

            let residual = x[x_ptr_idx + i] - prediction;
            *sample_out = residual * inv_gain;
        }

        res_offset += chunk;
        x_ptr_idx += subfr_length;
    }
}

#[cfg(test)]
mod tests {
    use super::{LTP_CENTER, LTP_ORDER, ltp_analysis_filter_flp};
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn copies_and_scales_without_prediction() {
        let subfr_length = 4;
        let pre_length = 2;
        let nb_subfr = 1;
        let chunk = subfr_length + pre_length;
        let pitch = 6i32;
        let x_ptr_offset = 10;

        let x: Vec<f32> = (0..32).map(|n| n as f32 * 0.25).collect();
        let mut ltp_res = vec![0.0f32; chunk];
        let ltp_coef = vec![0.0f32; LTP_ORDER];
        let pitch_l = [pitch];
        let inv_gains = [0.5f32];

        ltp_analysis_filter_flp(
            &mut ltp_res,
            &x,
            x_ptr_offset,
            &ltp_coef,
            &pitch_l,
            &inv_gains,
            subfr_length,
            nb_subfr,
            pre_length,
        );

        let expected: Vec<f32> = x[x_ptr_offset..x_ptr_offset + chunk]
            .iter()
            .map(|value| value * inv_gains[0])
            .collect();
        assert_eq!(ltp_res, expected);
    }

    #[test]
    fn subtracts_lagged_prediction() {
        let subfr_length = 3;
        let pre_length = 2;
        let nb_subfr = 1;
        let chunk = subfr_length + pre_length;
        let pitch = 5i32;
        let x_ptr_offset = 8;

        let x: Vec<f32> = (0..24).map(|n| n as f32 * 0.5).collect();
        let mut ltp_res = vec![0.0f32; chunk];
        let mut ltp_coef = vec![0.0f32; LTP_ORDER];
        ltp_coef[LTP_CENTER] = 1.0;
        let pitch_l = [pitch];
        let inv_gains = [1.5f32];

        ltp_analysis_filter_flp(
            &mut ltp_res,
            &x,
            x_ptr_offset,
            &ltp_coef,
            &pitch_l,
            &inv_gains,
            subfr_length,
            nb_subfr,
            pre_length,
        );

        let expected: Vec<f32> = (0..chunk)
            .map(|i| {
                let current = x[x_ptr_offset + i];
                let prediction = x[x_ptr_offset - pitch as usize + i];
                (current - prediction) * inv_gains[0]
            })
            .collect();
        assert_eq!(ltp_res, expected);
    }
}
