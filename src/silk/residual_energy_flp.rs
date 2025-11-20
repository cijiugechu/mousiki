//! Port of `silk/float/residual_energy_FLP.c`.
//!
//! These helpers compute the weighted residual energy for floating-point input
//! by filtering each half-frame with the LPC predictors and accumulating the
//! per-subframe energies with the squared quantisation gains.

use crate::silk::decoder_set_fs::MAX_FRAME_LENGTH;
use crate::silk::energy_flp::energy;
use crate::silk::lpc_analysis_filter_flp::lpc_analysis_filter_flp;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR};

const MAX_ITERATIONS_RESIDUAL_NRG: usize = 10;
const REGULARIZATION_FACTOR: f32 = 1e-8;
const HALF_MAX_SUBFR: usize = MAX_NB_SUBFR / 2;
const MAX_RESIDUAL: usize =
    (MAX_FRAME_LENGTH / 2) + ((MAX_NB_SUBFR * MAX_LPC_ORDER) / 2);

/// Mirrors `silk_residual_energy_covar_FLP`.
pub fn residual_energy_covar_flp(
    c: &[f32],
    w_xx: &mut [f32],
    w_xx_vec: &[f32],
    wxx: f32,
    dim: usize,
) -> f32 {
    assert!(dim > 0, "dimension must be positive");
    assert!(
        w_xx.len() >= dim * dim,
        "wXX must hold dim × dim entries"
    );
    assert!(c.len() >= dim, "coefficient slice too short");
    assert!(w_xx_vec.len() >= dim, "correlation vector too short");

    let mut regularization = REGULARIZATION_FACTOR * (w_xx[0] + w_xx[dim * dim - 1]);

    for _ in 0..MAX_ITERATIONS_RESIDUAL_NRG {
        let mut tmp = 0.0f32;
        let mut nrg = wxx;

        for i in 0..dim {
            tmp += w_xx_vec[i] * c[i];
        }
        nrg -= 2.0 * tmp;

        for i in 0..dim {
            let mut acc = 0.0f32;
            for j in (i + 1)..dim {
                acc += w_xx[i + dim * j] * c[j];
            }
            let diag = w_xx[i + dim * i];
            nrg += c[i] * (2.0 * acc + diag * c[i]);
        }

        if nrg > 0.0 {
            return nrg;
        }

        if regularization > 0.0 {
            for i in 0..dim {
                let diag_idx = i + dim * i;
                w_xx[diag_idx] += regularization;
            }
            regularization *= 2.0;
        }
    }

    1.0
}

/// Calculates the per-subframe residual energies using floating-point LPC taps.
#[allow(clippy::too_many_arguments)]
pub fn residual_energy_flp(
    nrgs: &mut [f32; MAX_NB_SUBFR],
    x: &[f32],
    a: &[[f32; MAX_LPC_ORDER]; HALF_MAX_SUBFR],
    gains: &[f32],
    subfr_length: usize,
    nb_subfr: usize,
    lpc_order: usize,
) {
    assert!(
        nb_subfr == MAX_NB_SUBFR || nb_subfr == HALF_MAX_SUBFR,
        "nb_subfr must be {} or {}",
        MAX_NB_SUBFR,
        HALF_MAX_SUBFR
    );
    assert!(
        matches!(lpc_order, 6 | 8 | 10 | 12 | 16),
        "unsupported LPC order: {lpc_order}"
    );
    assert!(
        gains.len() >= nb_subfr,
        "gains slice must hold nb_subfr entries"
    );

    nrgs.fill(0.0);

    let shift = lpc_order
        .checked_add(subfr_length)
        .expect("subframe length overflow");
    let block_len = shift
        .checked_mul(2)
        .expect("block length overflow");
    assert!(
        block_len <= MAX_RESIDUAL,
        "block length exceeds scratch buffer"
    );

    let needed = (nb_subfr / 2)
        .checked_mul(block_len)
        .expect("total input length overflow");
    assert!(x.len() >= needed, "input slice too short");

    let mut lpc_res = [0f32; MAX_RESIDUAL];

    lpc_analysis_filter_flp(
        &mut lpc_res[..block_len],
        &a[0][..lpc_order],
        &x[..block_len],
        block_len,
        lpc_order,
    );

    let lpc_res_ptr = lpc_order;
    nrgs[0] = gains[0]
        * gains[0]
        * (energy(&lpc_res[lpc_res_ptr..lpc_res_ptr + subfr_length]) as f32);
    nrgs[1] = gains[1]
        * gains[1]
        * (energy(
            &lpc_res[lpc_res_ptr + shift..lpc_res_ptr + shift + subfr_length],
        ) as f32);

    if nb_subfr == MAX_NB_SUBFR {
        let start = block_len;
        lpc_analysis_filter_flp(
            &mut lpc_res[..block_len],
            &a[1][..lpc_order],
            &x[start..start + block_len],
            block_len,
            lpc_order,
        );

        nrgs[2] = gains[2]
            * gains[2]
            * (energy(&lpc_res[lpc_res_ptr..lpc_res_ptr + subfr_length]) as f32);
        nrgs[3] = gains[3]
            * gains[3]
            * (energy(
                &lpc_res[lpc_res_ptr + shift..lpc_res_ptr + shift + subfr_length],
            ) as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::{residual_energy_covar_flp, residual_energy_flp};
    use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR};
    use alloc::vec::Vec;

    fn assert_close(lhs: f32, rhs: f32) {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 1e-6,
            "difference exceeds tolerance: {diff} > 1e-6 (lhs={lhs}, rhs={rhs})"
        );
    }

    #[test]
    fn residual_energy_covar_positive_case() {
        let c = [0.1f32, -0.2];
        let mut w_xx = [1.0f32, 0.0, 0.0, 2.0];
        let w_xx_vec = [0.2f32, 0.4];
        let nrg = residual_energy_covar_flp(&c, &mut w_xx, &w_xx_vec, 0.5, 2);
        assert_close(nrg, 0.71);
    }

    #[test]
    fn residual_energy_covar_regularizes_after_multiple_attempts() {
        let c = [0.0f32, 0.0];
        let mut w_xx = [0.0f32; 4];
        let w_xx_vec = [0.0f32; 2];
        let nrg = residual_energy_covar_flp(&c, &mut w_xx, &w_xx_vec, 0.0, 2);
        assert_close(nrg, 1.0);
    }

    #[test]
    fn residual_energy_two_subframes_matches_expected() {
        const LPC_ORDER: usize = 6;
        const SUBFR: usize = 4;
        const NB_SUBFR: usize = MAX_NB_SUBFR / 2;

        let mut nrgs = [0.0f32; MAX_NB_SUBFR];
        let x: Vec<f32> = (0..20).map(|n| n as f32).collect();
        let a = [[0.0f32; MAX_LPC_ORDER]; 2];
        let gains = [1.0f32, 1.0];

        residual_energy_flp(
            &mut nrgs,
            &x,
            &a,
            &gains,
            SUBFR,
            NB_SUBFR,
            LPC_ORDER,
        );

        assert_close(nrgs[0], 230.0);
        assert_close(nrgs[1], 1230.0);
    }

    #[test]
    fn residual_energy_four_subframes_applies_gains() {
        const LPC_ORDER: usize = 6;
        const SUBFR: usize = 4;
        const NB_SUBFR: usize = MAX_NB_SUBFR;

        let mut nrgs = [0.0f32; MAX_NB_SUBFR];
        let x: Vec<f32> = (0..40).map(|n| n as f32).collect();
        let a = [[0.0f32; MAX_LPC_ORDER]; 2];
        let gains = [1.0f32, 0.5, 0.25, 0.75];

        residual_energy_flp(
            &mut nrgs,
            &x,
            &a,
            &gains,
            SUBFR,
            NB_SUBFR,
            LPC_ORDER,
        );

        assert_close(nrgs[0], 230.0);
        assert_close(nrgs[1], 307.5);
        assert_close(nrgs[2], 189.375);
        assert_close(nrgs[3], 3166.875);
    }
}
