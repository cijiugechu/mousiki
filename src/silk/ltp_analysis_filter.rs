//! Port of `silk/fixed/LTP_analysis_filter_FIX.c`.
//!
//! Creates the LTP residual by subtracting the long-term predictor estimate
//! and scaling the output by the inverse quantisation gains.

use crate::silk::vq_wmat_ec::LTP_ORDER;

const LTP_CENTER: usize = LTP_ORDER / 2;

/// Apply the SILK long-term prediction analysis filter.
///
/// The `x_ptr_offset` parameter mirrors the C implementation's pointer
/// arithmetic: it marks the index in `x` that corresponds to `x_ptr = x -
/// pre_length`.  Callers must therefore provide enough pitch history before
/// `x_ptr_offset` so that `pitch_l[k] + 2` samples may be read.
#[allow(clippy::too_many_arguments)]
pub fn ltp_analysis_filter(
    ltp_res: &mut [i16],
    x: &[i16],
    x_ptr_offset: usize,
    ltp_coef_q14: &[i16],
    pitch_l: &[i32],
    inv_gains_q16: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let chunk = subfr_length + pre_length;
    assert!(chunk > 0, "subframe chunk must be non-zero");

    let total_samples = nb_subfr
        .checked_mul(chunk)
        .expect("ltp_res length overflow");
    assert!(ltp_res.len() >= total_samples, "ltp_res buffer too small");
    assert!(pitch_l.len() >= nb_subfr, "pitchL slice must cover nb_subfr");
    assert!(inv_gains_q16.len() >= nb_subfr, "invGains slice must cover nb_subfr");
    assert!(ltp_coef_q14.len() >= nb_subfr * LTP_ORDER, "LTP coefficients slice too short");

    let mut res_offset = 0;
    let mut x_ptr_idx = x_ptr_offset;

    for subfr in 0..nb_subfr {
        let pitch = pitch_l[subfr];
        assert!(pitch > 0, "pitch lag must be positive");
        let pitch_usize = pitch as usize;

        let chunk_end = x_ptr_idx
            .checked_add(chunk)
            .expect("x_ptr index overflow");
        assert!(chunk_end <= x.len(), "x buffer too short for subframe {subfr}");

        let lag_base = x_ptr_idx
            .checked_sub(pitch_usize)
            .expect("insufficient pitch history");
        assert!(lag_base >= 2, "pitch history must include two guard samples");

        let lag_max = lag_base
            .checked_add(chunk - 1 + LTP_CENTER)
            .expect("lag range overflow");
        assert!(lag_max < x.len(), "x buffer too short for LTP taps");

        let taps_offset = subfr * LTP_ORDER;
        let taps = &ltp_coef_q14[taps_offset..taps_offset + LTP_ORDER];

        let res_slice = &mut ltp_res[res_offset..res_offset + chunk];
        let mut lag_index = lag_base as isize;
        for (i, out) in res_slice.iter_mut().enumerate() {
            let sample = x[x_ptr_idx + i];

            let mut ltp_est = smulbb(x[(lag_index + LTP_CENTER as isize) as usize], taps[0]);
            ltp_est = smlabb_ovflw(ltp_est, x[(lag_index + 1) as usize], taps[1]);
            ltp_est = smlabb_ovflw(ltp_est, x[lag_index as usize], taps[2]);
            ltp_est = smlabb_ovflw(ltp_est, x[(lag_index - 1) as usize], taps[3]);
            ltp_est = smlabb_ovflw(ltp_est, x[(lag_index - 2) as usize], taps[4]);

            let prediction_q0 = rshift_round(ltp_est, 14);
            let residual = i32::from(sample) - prediction_q0;
            let residual_sat = clamp_to_i16(residual);
            let scaled = smulwb(inv_gains_q16[subfr], residual_sat);
            *out = clamp_to_i16(scaled);

            lag_index += 1;
        }

        res_offset += chunk;
        x_ptr_idx += subfr_length;
    }
}

fn smulbb(a: i16, b: i16) -> i32 {
    i32::from(a) * i32::from(b)
}

fn smlabb_ovflw(acc: i32, sample: i16, coeff: i16) -> i32 {
    acc.wrapping_add(smulbb(sample, coeff))
}

fn rshift_round(value: i32, shift: u32) -> i32 {
    debug_assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn smulwb(a: i32, b: i16) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

fn clamp_to_i16(value: i32) -> i16 {
    if value > i32::from(i16::MAX) {
        i16::MAX
    } else if value < i32::from(i16::MIN) {
        i16::MIN
    } else {
        value as i16
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::ltp_analysis_filter;
    use super::LTP_ORDER;

    #[test]
    fn copies_when_no_prediction() {
        let subfr_length = 4;
        let pre_length = 2;
        let nb_subfr = 1;
        let chunk = subfr_length + pre_length;
        let pitch = 6;
        let x_ptr_offset = 10;

        let x: Vec<i16> = (0..32).map(|n| (n * 10) as i16).collect();
        let mut ltp_res = vec![0i16; chunk];
        let ltp_coef = [0i16; LTP_ORDER];
        let pitch_l = [pitch];
        let inv_gains = [1 << 16];

        ltp_analysis_filter(
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

        assert_eq!(&ltp_res, &x[x_ptr_offset..x_ptr_offset + chunk]);
    }

    #[test]
    fn subtracts_lagged_samples() {
        let subfr_length = 4;
        let pre_length = 2;
        let nb_subfr = 1;
        let chunk = subfr_length + pre_length;
        let pitch = 4;
        let x_ptr_offset = 10;

        let x: Vec<i16> = (0..32).map(|n| (n * 100) as i16).collect();
        let mut ltp_res = vec![0i16; chunk];
        let mut ltp_coef = [0i16; LTP_ORDER];
        ltp_coef[2] = 1 << 14;
        let pitch_l = [pitch];
        let inv_gains = [1 << 15]; // 0.5 in Q16

        ltp_analysis_filter(
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

        let pitch_usize = pitch as usize;
        let expected: Vec<i16> = (0..chunk)
            .map(|i| {
                let sample_idx = x_ptr_offset + i;
                let diff = i32::from(x[sample_idx]) - i32::from(x[sample_idx - pitch_usize]);
                ((diff * (1 << 15)) >> 16) as i16
            })
            .collect();

        assert_eq!(ltp_res, expected);
    }
}
