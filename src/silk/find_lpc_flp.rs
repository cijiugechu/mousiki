//! Port of `silk/float/find_LPC_FLP.c`.
//!
//! Computes floating-point LPC predictors for the encoder analysis path,
//! optionally searching for an interpolated NLSF vector that reduces the
//! first-half residual energy when interpolation is enabled.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::int_plus_one
)]

use crate::silk::a2nlsf::a2nlsf;
use crate::silk::burg_modified_flp::silk_burg_modified_flp;
use crate::silk::encoder::state::{EncoderStateCommon, MAX_FRAME_LENGTH};
use crate::silk::energy_flp::energy;
use crate::silk::interpolate::interpolate;
use crate::silk::lpc_analysis_filter_flp::lpc_analysis_filter_flp;
use crate::silk::nlsf2a::nlsf2a;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR};
use libm::rintf;

/// Mirrors `silk_find_LPC_FLP` from the reference implementation.
pub fn find_lpc_flp(
    state: &mut EncoderStateCommon,
    nlsf_q15: &mut [i16],
    x: &[f32],
    min_inv_gain: f32,
) {
    let order = state.predict_lpc_order;
    assert!(
        order <= MAX_LPC_ORDER,
        "predictor order exceeds MAX_LPC_ORDER"
    );
    assert!(
        matches!(order, 6 | 8 | 10 | 12 | 16),
        "unsupported LPC order: {order}"
    );
    assert!(nlsf_q15.len() >= order, "NLSF buffer shorter than LPC order");
    assert!(
        state.nb_subfr > 0 && state.nb_subfr <= MAX_NB_SUBFR,
        "invalid subframe count"
    );

    let subfr_length = state
        .subfr_length
        .checked_add(order)
        .expect("subframe length overflow");
    let frame_length = subfr_length
        .checked_mul(state.nb_subfr)
        .expect("frame length overflow");
    assert!(x.len() >= frame_length, "input shorter than frame length");
    assert!(min_inv_gain > 0.0, "min_inv_gain must be positive");

    state.indices.nlsf_interp_coef_q2 = 4;

    let mut a = [0f32; MAX_LPC_ORDER];
    let mut res_nrg = silk_burg_modified_flp(
        &mut a[..order],
        &x[..frame_length],
        min_inv_gain,
        subfr_length,
        state.nb_subfr,
        order,
        state.arch,
    );

    if state.use_interpolated_nlsfs
        && !state.first_frame_after_reset
        && state.nb_subfr == MAX_NB_SUBFR
    {
        let tail_offset = (MAX_NB_SUBFR / 2) * subfr_length;
        let tail_end = tail_offset + (MAX_NB_SUBFR / 2) * subfr_length;
        assert!(tail_end <= x.len(), "input shorter than interpolation tail");

        let mut a_tmp = [0f32; MAX_LPC_ORDER];
        res_nrg -= silk_burg_modified_flp(
            &mut a_tmp[..order],
            &x[tail_offset..tail_end],
            min_inv_gain,
            subfr_length,
            MAX_NB_SUBFR / 2,
            order,
            state.arch,
        );
        a2nlsf_flp(&mut nlsf_q15[..order], &a_tmp[..order], order);

        let mut res_nrg_2nd = f32::MAX;
        let mut interp_nlsf = [0i16; MAX_LPC_ORDER];
        let mut a_interp = [0f32; MAX_LPC_ORDER];
        let mut lpc_res = [0f32; MAX_FRAME_LENGTH + MAX_NB_SUBFR * MAX_LPC_ORDER];

        let valid_len = subfr_length - order;
        assert!(valid_len > 0, "subframe length must exceed LPC order");

        let head_len = 2 * subfr_length;
        assert!(head_len <= frame_length, "frame too short for interpolation");

        for k in (0..=3).rev() {
            interpolate(
                &mut interp_nlsf[..order],
                &state.prev_nlsf_q15[..order],
                &nlsf_q15[..order],
                k,
            );
            nlsf2a_flp(&mut a_interp[..order], &interp_nlsf[..order], order, state.arch);

            lpc_analysis_filter_flp(
                &mut lpc_res[..head_len],
                &a_interp[..order],
                &x[..head_len],
                head_len,
                order,
            );

            let start0 = order;
            let start1 = order + subfr_length;
            let res0 = energy(&lpc_res[start0..start0 + valid_len]);
            let res1 = energy(&lpc_res[start1..start1 + valid_len]);
            let res_nrg_interp = (res0 + res1) as f32;

            if res_nrg_interp < res_nrg {
                res_nrg = res_nrg_interp;
                state.indices.nlsf_interp_coef_q2 = k as i8;
            } else if res_nrg_interp > res_nrg_2nd {
                break;
            }

            res_nrg_2nd = res_nrg_interp;
        }
    }

    if state.indices.nlsf_interp_coef_q2 == 4 {
        a2nlsf_flp(&mut nlsf_q15[..order], &a[..order], order);
    }

    assert!(
        state.indices.nlsf_interp_coef_q2 == 4
            || (state.use_interpolated_nlsfs
                && !state.first_frame_after_reset
                && state.nb_subfr == MAX_NB_SUBFR),
        "interpolation coefficient set outside valid configuration"
    );
}

fn a2nlsf_flp(nlsf_q15: &mut [i16], a: &[f32], order: usize) {
    assert!(
        nlsf_q15.len() >= order && a.len() >= order,
        "buffer length shorter than LPC order"
    );

    let mut a_q16 = [0i32; MAX_LPC_ORDER];
    for (dst, &coeff) in a_q16.iter_mut().zip(a.iter()).take(order) {
        *dst = float_to_int(coeff * 65536.0);
    }

    a2nlsf(&mut nlsf_q15[..order], &mut a_q16[..order]);
}

fn nlsf2a_flp(a: &mut [f32], nlsf_q15: &[i16], order: usize, arch: i32) {
    assert!(
        a.len() >= order && nlsf_q15.len() >= order,
        "buffer length shorter than LPC order"
    );

    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    nlsf2a(&mut a_q12[..order], &nlsf_q15[..order], arch);

    for (dst, &src) in a.iter_mut().zip(a_q12.iter()).take(order) {
        *dst = f32::from(src) * (1.0 / 4096.0);
    }
}

#[inline]
fn float_to_int(value: f32) -> i32 {
    rintf(value) as i32
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use super::find_lpc_flp;
    use alloc::vec;
    use alloc::vec::Vec;
    use crate::silk::encoder::state::EncoderStateCommon;
    use crate::silk::MAX_LPC_ORDER;

    fn monotonic(nlsf: &[i16]) -> bool {
        nlsf.windows(2).all(|pair| pair[0] <= pair[1])
    }

    #[test]
    fn computes_lpc_without_interpolation() {
        let mut state = EncoderStateCommon::default();
        state.predict_lpc_order = 10;
        let frame_length = (state.subfr_length + state.predict_lpc_order) * state.nb_subfr;
        let x: Vec<f32> = (0..frame_length)
            .map(|i| (i as f32).sin() * 0.5f32)
            .collect();

        let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
        find_lpc_flp(&mut state, &mut nlsf_q15, &x, 1.0);

        assert_eq!(state.indices.nlsf_interp_coef_q2, 4);
        assert!(monotonic(&nlsf_q15[..state.predict_lpc_order]));
        assert!(nlsf_q15[..state.predict_lpc_order]
            .iter()
            .any(|&value| value != 0));
    }

    #[test]
    fn handles_interpolation_configuration() {
        let mut state = EncoderStateCommon::default();
        state.predict_lpc_order = 10;
        state.use_interpolated_nlsfs = true;
        state.first_frame_after_reset = false;
        for (i, value) in state.prev_nlsf_q15.iter_mut().enumerate() {
            *value = (i as i16 + 1) * 512;
        }

        let frame_length = (state.subfr_length + state.predict_lpc_order) * state.nb_subfr;
        let mut x = vec![0.0f32; frame_length];
        for (i, sample) in x.iter_mut().enumerate() {
            let t = i as f32 / frame_length as f32;
            *sample = (t * 6.283185307179586).cos();
        }

        let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
        find_lpc_flp(&mut state, &mut nlsf_q15, &x, 1.0);

        assert!(state.indices.nlsf_interp_coef_q2 <= 4);
        assert!(monotonic(&nlsf_q15[..state.predict_lpc_order]));
    }
}
