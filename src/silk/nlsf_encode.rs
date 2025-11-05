//! NLSF vector encoder for the SILK fixed-point pipeline.
//!
//! Port of `silk_NLSF_encode` from the reference implementation. The routine
//! stabilises the target NLSF vector, performs a two-stage codebook search
//! using a delayed-decision trellis for the residual, and returns the selected
//! codebook path alongside the accumulated rate/distortion cost.

use core::convert::TryFrom;

use super::nlsf_decode::nlsf_decode;
use super::nlsf_del_dec_quant::nlsf_del_dec_quant;
use super::nlsf_stabilize::nlsf_stabilize;
use super::nlsf_unpack::nlsf_unpack;
use super::nlsf_vq::nlsf_vq;
use super::sort::insertion_sort_increasing;
use super::{MAX_LPC_ORDER, SilkNlsfCb};
use crate::silk::lin2log::lin2log;
use crate::silk::stereo_find_predictor::div32_varq;

const NLSF_VQ_MAX_VECTORS: usize = 32;
const NLSF_QUANT_MAX_AMPLITUDE_EXT: i8 = 10;

/// Encode an NLSF vector using the supplied codebook and weighting information.
///
/// * `nlsf_indices` - Output buffer receiving the stage-one index followed by
///   `order` residual indices.
/// * `nlsf_q15` - In/out buffer containing the target NLSF vector in Q15; it is
///   replaced by the quantised NLSF on return.
/// * `codebook` - Codebook metadata (stage-one/ two tables, predictors, rates).
/// * `weights_q2` - Per-coefficient weights in Q2.
/// * `nlsf_mu_q20` - Rate/distortion trade-off parameter in Q20.
/// * `n_survivors` - Number of stage-one survivors to evaluate (max 32).
/// * `signal_type` - Frame signal type (0: inactive/unvoiced, 2: voiced).
pub fn nlsf_encode(
    nlsf_indices: &mut [i8],
    nlsf_q15: &mut [i16],
    codebook: &SilkNlsfCb,
    weights_q2: &[i16],
    nlsf_mu_q20: i32,
    n_survivors: usize,
    signal_type: i32,
) -> i32 {
    let order =
        usize::try_from(codebook.order).expect("codebook order must be representable as usize");
    let n_vectors = usize::try_from(codebook.n_vectors)
        .expect("codebook vector count must be representable as usize");

    assert_eq!(
        nlsf_q15.len(),
        order,
        "NLSF buffer must match the codebook order"
    );
    assert_eq!(
        weights_q2.len(),
        order,
        "weight vector must match the codebook order"
    );
    assert_eq!(
        nlsf_indices.len(),
        order + 1,
        "index buffer must hold stage-one and residual entries"
    );
    assert!(order <= MAX_LPC_ORDER, "order exceeds MAX_LPC_ORDER");
    assert!(n_survivors > 0, "must evaluate at least one survivor");
    assert!(
        n_survivors <= n_vectors && n_survivors <= NLSF_VQ_MAX_VECTORS,
        "survivor count out of range"
    );

    nlsf_stabilize(nlsf_q15, &codebook.delta_min_q15[..order + 1]);

    let mut err_q24 = [0i32; NLSF_VQ_MAX_VECTORS];
    nlsf_vq(
        &mut err_q24[..n_vectors],
        nlsf_q15,
        &codebook.cb1_nlsf_q8[..order * n_vectors],
        &codebook.cb1_wght_q9[..order * n_vectors],
    );

    let mut temp_indices1 = [0i32; NLSF_VQ_MAX_VECTORS];
    insertion_sort_increasing(
        &mut err_q24[..n_vectors],
        &mut temp_indices1[..n_survivors],
        n_survivors,
    );

    let mut rd_q25 = [0i32; NLSF_VQ_MAX_VECTORS];
    let mut temp_indices2 = [0i8; NLSF_VQ_MAX_VECTORS * MAX_LPC_ORDER];
    let mut res_q10 = [0i16; MAX_LPC_ORDER];
    let mut nlsf_tmp_q15 = [0i16; MAX_LPC_ORDER];
    let mut w_adj_q5 = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    let mut ec_ix = [0i16; MAX_LPC_ORDER];

    for (s, &ind1_raw) in temp_indices1.iter().take(n_survivors).enumerate() {
        let ind1 = usize::try_from(ind1_raw).expect("stage-one index must be non-negative");
        assert!(ind1 < n_vectors, "stage-one index out of range");

        let base = ind1
            .checked_mul(order)
            .expect("stage-one index multiplication overflow");
        let cb1_nlsf = &codebook.cb1_nlsf_q8[base..base + order];
        let cb1_wght = &codebook.cb1_wght_q9[base..base + order];

        for i in 0..order {
            nlsf_tmp_q15[i] = i16::from(cb1_nlsf[i]) << 7;
            let diff_q15 = i32::from(nlsf_q15[i]) - i32::from(nlsf_tmp_q15[i]);
            let w_tmp_q9 = i32::from(cb1_wght[i]);
            let res = rshift(smulbb(diff_q15, w_tmp_q9), 14);
            res_q10[i] = clamp_to_i16(res);

            let w_prod = smulbb(w_tmp_q9, w_tmp_q9);
            let adj = div32_varq(i32::from(weights_q2[i]), w_prod, 21);
            w_adj_q5[i] = clamp_to_i16(adj);
        }

        nlsf_unpack(&mut ec_ix[..order], &mut pred_q8[..order], codebook, ind1);

        let indices_slice = &mut temp_indices2[s * MAX_LPC_ORDER..][..order];
        rd_q25[s] = nlsf_del_dec_quant(
            indices_slice,
            &res_q10[..order],
            &w_adj_q5[..order],
            &pred_q8[..order],
            &ec_ix[..order],
            codebook.ec_rates_q5,
            i32::from(codebook.quant_step_size_q16),
            codebook.inv_quant_step_size_q6,
            nlsf_mu_q20,
        );

        let icdf_band = ((signal_type >> 1).clamp(0, 1) as usize) * n_vectors;
        let icdf = &codebook.cb1_icdf[icdf_band..icdf_band + n_vectors];
        let prob_q8 = if ind1 == 0 {
            256 - i32::from(icdf[ind1])
        } else {
            i32::from(icdf[ind1 - 1]) - i32::from(icdf[ind1])
        };
        let bits_q7 = (8 << 7) - lin2log(prob_q8);
        rd_q25[s] = smlabb(rd_q25[s], bits_q7, nlsf_mu_q20 >> 2);
    }

    let mut best_index_buf = [0i32; 1];
    insertion_sort_increasing(&mut rd_q25[..n_survivors], &mut best_index_buf, 1);
    let best_survivor =
        usize::try_from(best_index_buf[0]).expect("best survivor index must be non-negative");
    assert!(best_survivor < n_survivors);

    nlsf_indices[0] = temp_indices1[best_survivor] as i8;
    let residual_src = &temp_indices2[best_survivor * MAX_LPC_ORDER..][..order];
    for (dst, &src) in nlsf_indices[1..order + 1].iter_mut().zip(residual_src) {
        *dst = src;
    }

    debug_assert!(nlsf_indices[0] >= 0);
    debug_assert!(nlsf_indices[0] < codebook.n_vectors as i8);
    let valid_range = (-NLSF_QUANT_MAX_AMPLITUDE_EXT)..=NLSF_QUANT_MAX_AMPLITUDE_EXT;
    debug_assert!(
        nlsf_indices[1..order + 1]
            .iter()
            .all(|&idx| valid_range.contains(&idx))
    );

    nlsf_decode(nlsf_q15, &nlsf_indices[..order + 1], codebook);

    rd_q25[0]
}

#[inline]
fn smulbb(a: i32, b: i32) -> i32 {
    let a16 = i32::from(a as i16);
    let b16 = i32::from(b as i16);
    a16 * b16
}

#[inline]
fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulbb(b, c))
}

#[inline]
fn rshift(value: i32, shift: i32) -> i32 {
    value >> shift
}

#[inline]
fn clamp_to_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use super::nlsf_encode;
    use super::{NLSF_QUANT_MAX_AMPLITUDE_EXT, NLSF_VQ_MAX_VECTORS};
    use crate::silk::MAX_LPC_ORDER;
    use crate::silk::nlsf_decode::nlsf_decode;
    use crate::silk::nlsf_vq_weights_laroia::nlsf_vq_weights_laroia;
    use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;

    #[test]
    fn encodes_wideband_vector_and_updates_indices() {
        let codebook = &SILK_NLSF_CB_WB;
        let order = codebook.order as usize;
        let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];

        // Start from a non-trivial target NLSF vector.
        for (i, value) in nlsf_q15[..order].iter_mut().enumerate() {
            *value = (i as i16 * 2048 + 512).min(30_000);
        }

        let mut weights_q2 = [0i16; MAX_LPC_ORDER];
        nlsf_vq_weights_laroia(&mut weights_q2[..order], &nlsf_q15[..order]);

        let mut indices = [0i8; MAX_LPC_ORDER + 1];
        let rd_q25 = nlsf_encode(
            &mut indices[..order + 1],
            &mut nlsf_q15[..order],
            codebook,
            &weights_q2[..order],
            1 << 14,
            8,
            2,
        );

        assert!(rd_q25 >= 0);
        assert!(indices[0] >= 0 && indices[0] < codebook.n_vectors as i8);
        assert!(indices[1..order + 1].iter().all(|&idx| {
            idx >= -NLSF_QUANT_MAX_AMPLITUDE_EXT && idx <= NLSF_QUANT_MAX_AMPLITUDE_EXT
        }));

        let mut decoded = [0i16; MAX_LPC_ORDER];
        nlsf_decode(&mut decoded[..order], &indices[..order + 1], codebook);
        assert_eq!(&decoded[..order], &nlsf_q15[..order]);
    }

    #[test]
    fn clamps_survivor_count_to_table_size() {
        let codebook = &SILK_NLSF_CB_WB;
        let order = codebook.order as usize;
        let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
        for (i, value) in nlsf_q15[..order].iter_mut().enumerate() {
            *value = (i as i16 * 1500 + 300).min(30_000);
        }

        let mut weights_q2 = [0i16; MAX_LPC_ORDER];
        nlsf_vq_weights_laroia(&mut weights_q2[..order], &nlsf_q15[..order]);

        let mut indices = [0i8; MAX_LPC_ORDER + 1];
        let rd_q25 = nlsf_encode(
            &mut indices[..order + 1],
            &mut nlsf_q15[..order],
            codebook,
            &weights_q2[..order],
            1 << 13,
            NLSF_VQ_MAX_VECTORS.min(codebook.n_vectors as usize),
            0,
        );

        assert!(rd_q25 >= 0);
    }
}
