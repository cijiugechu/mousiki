//! Port of the encoder-side `silk_process_NLSFs` helper.
//!
//! The original C routine (see `silk/process_NLSFs.c`) limits, stabilises, and
//! quantises the target NLSF vector before converting it to LPC predictor
//! coefficients for the current and (optionally interpolated) half-frame. This
//! Rust translation mirrors the fixed-point arithmetic so the encoder can drive
//! the same NLSF path selection as the reference implementation.

use core::convert::TryFrom;

use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::interpolate::interpolate;
use crate::silk::nlsf2a::nlsf2a;
use crate::silk::nlsf_encode::nlsf_encode;
use crate::silk::nlsf_vq_weights_laroia::nlsf_vq_weights_laroia;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR, SilkNlsfCb};

const NLSF_MU_BASE_Q20: i32 = 3_146; // SILK_FIX_CONST(0.003, 20)
const NLSF_MU_SLOPE_Q28: i32 = -268_434; // SILK_FIX_CONST(-0.001, 28)
const MAX_SPEECH_ACTIVITY_Q8: i32 = 1 << 8;
const MAX_NLSF_SURVIVORS: usize = 32;

/// Configuration needed to process the encoder NLSFs.
#[derive(Debug, Clone, Copy)]
pub struct ProcessNlsfConfig<'a> {
    pub speech_activity_q8: i32,
    pub nb_subframes: usize,
    pub predict_lpc_order: usize,
    pub use_interpolated_nlsfs: bool,
    pub nlsf_msvq_survivors: usize,
    pub codebook: &'a SilkNlsfCb,
    pub arch: i32,
}

impl<'a> ProcessNlsfConfig<'a> {
    fn validate(&self) {
        assert!(
            self.nb_subframes == MAX_NB_SUBFR || self.nb_subframes == MAX_NB_SUBFR / 2,
            "nb_subframes must be 4 or 2"
        );
        assert!(
            matches!(self.predict_lpc_order, 10 | 16),
            "predict_lpc_order must be 10 or 16"
        );
        assert!(
            self.predict_lpc_order <= MAX_LPC_ORDER,
            "predict_lpc_order exceeds MAX_LPC_ORDER"
        );
        assert!(
            (1..=MAX_NLSF_SURVIVORS).contains(&self.nlsf_msvq_survivors),
            "nlsf_msvq_survivors must be within 1..=32"
        );
        let codebook_order = usize::try_from(self.codebook.order)
            .expect("codebook order must fit into usize");
        assert_eq!(
            codebook_order, self.predict_lpc_order,
            "codebook order mismatch"
        );
        let codebook_vectors = usize::try_from(self.codebook.n_vectors)
            .expect("codebook vector count must fit into usize");
        assert!(
            self.nlsf_msvq_survivors <= codebook_vectors,
            "survivor count exceeds codebook size"
        );
    }
}

/// Limit, stabilise, quantise, and convert the encoder NLSFs to LPC taps.
pub fn process_nlsfs(
    cfg: &ProcessNlsfConfig<'_>,
    indices: &mut SideInfoIndices,
    pred_coef_q12: &mut [[i16; MAX_LPC_ORDER]; 2],
    nlsf_q15: &mut [i16],
    prev_nlsf_q15: &[i16],
) {
    cfg.validate();
    assert_eq!(
        nlsf_q15.len(),
        cfg.predict_lpc_order,
        "NLSF vector must match predict_lpc_order"
    );
    assert_eq!(
        prev_nlsf_q15.len(),
        cfg.predict_lpc_order,
        "previous NLSF vector must match predict_lpc_order"
    );
    debug_assert!((0..=MAX_SPEECH_ACTIVITY_Q8).contains(&cfg.speech_activity_q8));
    debug_assert!((0..=4).contains(&indices.nlsf_interp_coef_q2));
    debug_assert!(
        cfg.use_interpolated_nlsfs || indices.nlsf_interp_coef_q2 == 4,
        "interpolation disabled but interpolation factor not forced to 4"
    );

    let order = cfg.predict_lpc_order;
    let nlsf_current = &mut nlsf_q15[..order];
    let prev_nlsf = &prev_nlsf_q15[..order];

    let mut weights_q = [0i16; MAX_LPC_ORDER];
    nlsf_vq_weights_laroia(&mut weights_q[..order], nlsf_current);

    let mut interpolated_q15 = [0i16; MAX_LPC_ORDER];
    let mut interpolated_weights_q = [0i16; MAX_LPC_ORDER];
    let do_interpolate = cfg.use_interpolated_nlsfs && indices.nlsf_interp_coef_q2 < 4;

    if do_interpolate {
        interpolate(
            &mut interpolated_q15[..order],
            prev_nlsf,
            nlsf_current,
            i32::from(indices.nlsf_interp_coef_q2),
        );
        nlsf_vq_weights_laroia(
            &mut interpolated_weights_q[..order],
            &interpolated_q15[..order],
        );

        let coef_q2 = i32::from(indices.nlsf_interp_coef_q2);
        let i_sqr_q15 = smulbb(coef_q2, coef_q2) << 11;
        for i in 0..order {
            let base = i32::from(weights_q[i]) >> 1;
            let contrib = (i32::from(interpolated_weights_q[i]) * i_sqr_q15) >> 16;
            let updated = base + contrib;
            debug_assert!(updated >= 1);
            weights_q[i] = updated as i16;
        }
    }

    let nlsf_mu_q20 = compute_nlsf_mu(cfg.speech_activity_q8, cfg.nb_subframes);
    nlsf_encode(
        &mut indices.nlsf_indices[..order + 1],
        nlsf_current,
        cfg.codebook,
        &weights_q[..order],
        nlsf_mu_q20,
        cfg.nlsf_msvq_survivors,
        i32::from(indices.signal_type),
    );

    let predictors = pred_coef_q12.as_mut_slice();
    let (first_half, second_half) = predictors.split_at_mut(1);
    let first_row = &mut first_half[0][..order];
    let second_row = &mut second_half[0][..order];

    nlsf2a(second_row, nlsf_current, cfg.arch);

    if do_interpolate {
        interpolate(
            &mut interpolated_q15[..order],
            prev_nlsf,
            nlsf_current,
            i32::from(indices.nlsf_interp_coef_q2),
        );
        nlsf2a(first_row, &interpolated_q15[..order], cfg.arch);
    } else {
        first_row.copy_from_slice(second_row);
    }
}

fn compute_nlsf_mu(speech_activity_q8: i32, nb_subframes: usize) -> i32 {
    let mut mu_q20 = smlawb(NLSF_MU_BASE_Q20, NLSF_MU_SLOPE_Q28, speech_activity_q8);
    if nb_subframes == MAX_NB_SUBFR / 2 {
        mu_q20 = add_rshift32(mu_q20, mu_q20, 1);
    }
    mu_q20
}

#[inline]
fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

#[inline]
fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add((b.wrapping_mul(i32::from(c as i16))) >> 16)
}

#[inline]
fn add_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b >> shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::nlsf_decode::nlsf_decode;
    use crate::silk::FrameSignalType;
    use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;

    fn sample_nlsf(order: usize, offset: i16) -> [i16; MAX_LPC_ORDER] {
        let mut out = [0i16; MAX_LPC_ORDER];
        for (i, value) in out.iter_mut().enumerate().take(order) {
            *value = offset + i as i16 * 1_500;
        }
        out
    }

    #[test]
    fn produces_lpc_coefficients_without_interpolation() {
        let order = 16;
        let mut nlsf = sample_nlsf(order, 600);
        let prev = sample_nlsf(order, 400);
        let mut indices = SideInfoIndices::default();
        indices.signal_type = FrameSignalType::Voiced;
        indices.nlsf_interp_coef_q2 = 4;

        let mut predictors = [[0i16; MAX_LPC_ORDER]; 2];
        let cfg = ProcessNlsfConfig {
            speech_activity_q8: 180,
            nb_subframes: MAX_NB_SUBFR,
            predict_lpc_order: order,
            use_interpolated_nlsfs: false,
            nlsf_msvq_survivors: 8,
            codebook: &SILK_NLSF_CB_WB,
            arch: 0,
        };

        process_nlsfs(&cfg, &mut indices, &mut predictors, &mut nlsf[..order], &prev[..order]);

        assert_eq!(&predictors[0][..order], &predictors[1][..order]);

        let mut decoded = [0i16; MAX_LPC_ORDER];
        nlsf_decode(
            &mut decoded[..order],
            &indices.nlsf_indices[..order + 1],
            &SILK_NLSF_CB_WB,
        );
        assert_eq!(&decoded[..order], &nlsf[..order]);
    }

    #[test]
    fn interpolates_half_frame_lpc_coefficients() {
        let order = 16;
        let mut nlsf = sample_nlsf(order, 800);
        let prev = sample_nlsf(order, 200);
        let mut indices = SideInfoIndices::default();
        indices.signal_type = FrameSignalType::Unvoiced;
        indices.nlsf_interp_coef_q2 = 2;

        let mut predictors = [[0i16; MAX_LPC_ORDER]; 2];
        let cfg = ProcessNlsfConfig {
            speech_activity_q8: 200,
            nb_subframes: MAX_NB_SUBFR / 2,
            predict_lpc_order: order,
            use_interpolated_nlsfs: true,
            nlsf_msvq_survivors: 6,
            codebook: &SILK_NLSF_CB_WB,
            arch: 0,
        };

        process_nlsfs(&cfg, &mut indices, &mut predictors, &mut nlsf[..order], &prev[..order]);

        let mut expected_full = [0i16; MAX_LPC_ORDER];
        nlsf2a(&mut expected_full[..order], &nlsf[..order], cfg.arch);
        assert_eq!(&predictors[1][..order], &expected_full[..order]);

        let mut interpolated = [0i16; MAX_LPC_ORDER];
        interpolate(
            &mut interpolated[..order],
            &prev[..order],
            &nlsf[..order],
            i32::from(indices.nlsf_interp_coef_q2),
        );
        let mut expected_half = [0i16; MAX_LPC_ORDER];
        nlsf2a(&mut expected_half[..order], &interpolated[..order], cfg.arch);
        assert_eq!(&predictors[0][..order], &expected_half[..order]);
    }
}
