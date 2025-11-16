//! Port of `silk/fixed/find_LPC_FIX.c`.
//!
//! Converts the input frame into LPC predictor coefficients via the Burg
//! method, optionally searching for an interpolated NLSF vector that lowers
//! the residual energy in the first half of the frame.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::int_plus_one
)]

use crate::silk::a2nlsf::a2nlsf;
use crate::silk::burg_modified::silk_burg_modified;
use crate::silk::decoder_set_fs::MAX_SUB_FRAME_LENGTH;
use crate::silk::encoder::state::EncoderStateCommon;
use crate::silk::interpolate::interpolate;
use crate::silk::lpc_analysis_filter::lpc_analysis_filter;
use crate::silk::nlsf2a::nlsf2a;
use crate::silk::sum_sqr_shift::sum_sqr_shift;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR};

/// Mirrors `silk_find_LPC_FIX` from the fixed-point SILK encoder.
pub fn find_lpc(
    state: &mut EncoderStateCommon,
    nlsf_q15: &mut [i16],
    x: &[i16],
    min_inv_gain_q30: i32,
) {
    let order = state.predict_lpc_order;
    assert!(order <= MAX_LPC_ORDER, "predictor order exceeds MAX_LPC_ORDER");
    assert!(nlsf_q15.len() >= order, "NLSF buffer shorter than LPC order");

    let subfr_length = state.subfr_length + order;
    assert!(
        subfr_length <= MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER,
        "subframe length exceeds supported maximum"
    );
    let nb_subfr = state.nb_subfr;
    assert!((1..=MAX_NB_SUBFR).contains(&nb_subfr));

    let frame_len = subfr_length
        .checked_mul(nb_subfr)
        .expect("frame length overflow");
    assert!(x.len() >= frame_len, "input shorter than frame length");

    state.indices.nlsf_interp_coef_q2 = 4;

    let mut a_q16 = [0i32; MAX_LPC_ORDER];
    let mut res_nrg = 0;
    let mut res_nrg_q = 0;
    silk_burg_modified(
        &mut res_nrg,
        &mut res_nrg_q,
        &mut a_q16[..order],
        &x[..frame_len],
        min_inv_gain_q30,
        subfr_length,
        nb_subfr,
        order,
        state.arch,
    );

    if state.use_interpolated_nlsfs
        && !state.first_frame_after_reset
        && nb_subfr == MAX_NB_SUBFR
    {
        run_interpolated_search(
            state,
            nlsf_q15,
            &x[..frame_len],
            min_inv_gain_q30,
            subfr_length,
            order,
            (&mut res_nrg, &mut res_nrg_q),
        );
    }

    if state.indices.nlsf_interp_coef_q2 == 4 {
        a2nlsf(&mut nlsf_q15[..order], &mut a_q16[..order]);
    }
}

fn run_interpolated_search(
    state: &mut EncoderStateCommon,
    nlsf_q15: &mut [i16],
    x: &[i16],
    min_inv_gain_q30: i32,
    subfr_length: usize,
    order: usize,
    residual: (&mut i32, &mut i32),
) {
    let (res_nrg, res_nrg_q) = residual;
    let tail_start = 2 * subfr_length;
    let tail_end = tail_start + 2 * subfr_length;
    assert!(tail_end <= x.len());

    let mut tail_res = 0;
    let mut tail_res_q = 0;
    let mut tail_a_q16 = [0i32; MAX_LPC_ORDER];
    silk_burg_modified(
        &mut tail_res,
        &mut tail_res_q,
        &mut tail_a_q16[..order],
        &x[tail_start..tail_end],
        min_inv_gain_q30,
        subfr_length,
        2,
        order,
        state.arch,
    );

    (*res_nrg, *res_nrg_q) = subtract_energy(*res_nrg, *res_nrg_q, tail_res, tail_res_q);

    let mut candidate_nlsf = [0i16; MAX_LPC_ORDER];
    a2nlsf(&mut candidate_nlsf[..order], &mut tail_a_q16[..order]);

    let mut best_coef = 4i8;
    let mut best_nrg = *res_nrg;
    let mut best_q = *res_nrg_q;

    let mut interp_nlsf = [0i16; MAX_LPC_ORDER];
    let mut a_tmp_q12 = [0i16; MAX_LPC_ORDER];
    let mut lpc_res = [0i16; 2 * (MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER)];
    let half_len = 2 * subfr_length;
    let valid_len = subfr_length - order;
    assert!(valid_len > 0, "subframe length must exceed LPC order");

    for k in (0..=3).rev() {
        interpolate(
            &mut interp_nlsf[..order],
            &state.prev_nlsf_q15[..order],
            &candidate_nlsf[..order],
            k,
        );
        nlsf2a(&mut a_tmp_q12[..order], &interp_nlsf[..order], state.arch);

        lpc_analysis_filter(
            &mut lpc_res[..half_len],
            &x[..half_len],
            &a_tmp_q12[..order],
            half_len,
            order,
        );

        let (res0, shift0) = sum_sqr_shift(&lpc_res[order..order + valid_len]);
        let start = order + subfr_length;
        let (res1, shift1) = sum_sqr_shift(&lpc_res[start..start + valid_len]);
        let (interp_nrg, interp_q) = combine_energies(res0, shift0, res1, shift1);

        if energy_is_lower(best_nrg, best_q, interp_nrg, interp_q) {
            best_nrg = interp_nrg;
            best_q = interp_q;
            best_coef = k as i8;
        }
    }

    state.indices.nlsf_interp_coef_q2 = best_coef;
    *res_nrg = best_nrg;
    *res_nrg_q = best_q;

    if best_coef != 4 {
        nlsf_q15[..order].copy_from_slice(&candidate_nlsf[..order]);
    }
}

fn subtract_energy(res_nrg: i32, res_q: i32, remove: i32, remove_q: i32) -> (i32, i32) {
    let shift = remove_q - res_q;
    if shift >= 0 {
        if shift < 32 {
            (
                res_nrg.wrapping_sub(rshift_unsigned(remove, shift)),
                res_q,
            )
        } else {
            (res_nrg, res_q)
        }
    } else {
        assert!(shift > -32, "Q-domain difference too large");
        (
            rshift_unsigned(res_nrg, -shift).wrapping_sub(remove),
            remove_q,
        )
    }
}

fn combine_energies(res0: i32, shift0: i32, res1: i32, shift1: i32) -> (i32, i32) {
    let diff = shift0 - shift1;
    if diff >= 0 {
        let sum = res0.wrapping_add(rshift_unsigned(res1, diff));
        (sum, -shift0)
    } else {
        let sum = rshift_unsigned(res0, -diff).wrapping_add(res1);
        (sum, -shift1)
    }
}

fn energy_is_lower(base_nrg: i32, base_q: i32, candidate: i32, candidate_q: i32) -> bool {
    let shift = candidate_q - base_q;
    if shift >= 0 {
        if shift < 32 {
            rshift_unsigned(candidate, shift) < base_nrg
        } else {
            false
        }
    } else if -shift < 32 {
        candidate < rshift_unsigned(base_nrg, -shift)
    } else {
        false
    }
}

fn rshift_unsigned(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value.wrapping_shl((-shift) as u32)
    } else if shift >= 31 {
        0
    } else {
        ((value as u32) >> shift) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use crate::silk::encoder::state::EncoderStateCommon;

    #[test]
    fn computes_full_frame_nlsf_when_interpolation_disabled() {
        let mut state = EncoderStateCommon::default();
        state.use_interpolated_nlsfs = false;
        let order = state.predict_lpc_order;
        let subfr_length = state.subfr_length + order;
        let frame_len = state.nb_subfr * subfr_length;
        let mut input = vec![0i16; frame_len];
        for (i, sample) in input.iter_mut().enumerate() {
            let value = ((i as i32 * 37) % 4000) - 2000;
            *sample = value as i16;
        }

        let mut nlsf = [0i16; MAX_LPC_ORDER];
        find_lpc(&mut state, &mut nlsf, &input, 1 << 30);

        assert_eq!(state.indices.nlsf_interp_coef_q2, 4);

        let mut expected_res = 0;
        let mut expected_q = 0;
        let mut expected_a = [0i32; MAX_LPC_ORDER];
        silk_burg_modified(
            &mut expected_res,
            &mut expected_q,
            &mut expected_a[..order],
            &input,
            1 << 30,
            subfr_length,
            state.nb_subfr,
            order,
            state.arch,
        );
        let mut expected_nlsf = [0i16; MAX_LPC_ORDER];
        a2nlsf(&mut expected_nlsf[..order], &mut expected_a[..order]);

        assert_eq!(&nlsf[..order], &expected_nlsf[..order]);
    }
}
