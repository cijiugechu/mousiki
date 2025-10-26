//! Port of the SILK NLSF decoder from `silk/NLSF_decode.c`.
//!
//! Reconstructs a quantised normalised line spectral frequency (NLSF) vector
//! from codebook indices by unpacking the stage-one predictors, de-quantising
//! the residual, and finally stabilising the result. The implementation mirrors
//! the fixed-point arithmetic used by the reference C routine while exposing a
//! safe Rust interface.

use core::convert::TryFrom;

use super::nlsf_stabilize::nlsf_stabilize;
use super::nlsf_unpack::nlsf_unpack;
use super::SilkNlsfCb;
use super::MAX_LPC_ORDER;

const NLSF_QUANT_LEVEL_ADJ_Q10: i32 = 102; // SILK_FIX_CONST(0.1, 10)

/// Decode an NLSF vector from the supplied codebook indices.
///
/// * `nlsf_q15` - Output buffer that receives the decoded NLSF coefficients in
///   Q15 format. Its length must match the order of the codebook.
/// * `indices` - Codebook path vector whose first element selects the
///   stage-one entry while the remaining `order` elements hold the stage-two
///   residual indices.
/// * `codebook` - Metadata describing the NLSF codebook to use.
pub fn nlsf_decode(nlsf_q15: &mut [i16], indices: &[i8], codebook: &SilkNlsfCb) {
    let order = usize::try_from(codebook.order).expect("NLSF order must be representable as usize");
    assert!(
        order <= MAX_LPC_ORDER,
        "codebook order must not exceed MAX_LPC_ORDER"
    );
    assert_eq!(
        nlsf_q15.len(),
        order,
        "output buffer must match codebook order"
    );
    assert_eq!(
        indices.len(),
        order + 1,
        "indices must be [cb1_index, residuals[order]]"
    );

    let cb1_index = usize::try_from(indices[0]).expect("stage-one index must be non-negative");

    let mut ec_ix_buf = [0i16; MAX_LPC_ORDER];
    let mut pred_q8_buf = [0u8; MAX_LPC_ORDER];
    nlsf_unpack(
        &mut ec_ix_buf[..order],
        &mut pred_q8_buf[..order],
        codebook,
        cb1_index,
    );

    let mut res_q10_buf = [0i16; MAX_LPC_ORDER];
    nlsf_residual_dequant(
        &mut res_q10_buf[..order],
        &indices[1..order + 1],
        &pred_q8_buf[..order],
        codebook,
    );
    let res_q10 = &res_q10_buf[..order];

    let start = cb1_index
        .checked_mul(order)
        .expect("stage-one index multiplication overflowed");
    let cb1 = &codebook.cb1_nlsf_q8[start..start + order];
    let cb1_wght = &codebook.cb1_wght_q9[start..start + order];

    for i in 0..order {
        let residual = i32::from(res_q10[i]);
        let weight = i32::from(cb1_wght[i]);
        debug_assert!(weight > 0, "cb1_wght must be positive");
        let correction = div32_16(lshift(residual, 14), weight as i16);
        let base = i32::from(i16::from(cb1[i]));
        let value = add_lshift32(correction, base, 7);
        nlsf_q15[i] = clamp_to_u15(value);
    }

    nlsf_stabilize(nlsf_q15, codebook.delta_min_q15);
}

fn nlsf_residual_dequant(
    output_q10: &mut [i16],
    indices: &[i8],
    pred_coef_q8: &[u8],
    codebook: &SilkNlsfCb,
) {
    assert_eq!(output_q10.len(), pred_coef_q8.len());
    assert_eq!(indices.len(), output_q10.len());

    let mut out_q10 = 0i32;
    let quant_step_size_q16 = i32::from(codebook.quant_step_size_q16);

    for (i, (&index, &pred_coef)) in indices.iter().zip(pred_coef_q8.iter()).enumerate().rev() {
        let pred_q10 = rshift(smulbb(out_q10, i32::from(pred_coef)), 8);

        let mut quantised = lshift(i32::from(index), 10);
        if quantised > 0 {
            quantised -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if quantised < 0 {
            quantised += NLSF_QUANT_LEVEL_ADJ_Q10;
        }

        out_q10 = smlawb(pred_q10, quantised, quant_step_size_q16);
        output_q10[i] = out_q10.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
}

fn smulbb(a32: i32, b32: i32) -> i32 {
    i32::from((a32 as i16).wrapping_mul(b32 as i16))
}

fn smlawb(a32: i32, b32: i32, c32: i32) -> i32 {
    let product = (i64::from(b32) * i64::from(c32 as i16)) >> 16;
    a32.wrapping_add(product as i32)
}

fn add_lshift32(a32: i32, b32: i32, shift: i32) -> i32 {
    a32.wrapping_add(b32.wrapping_shl(shift as u32))
}

fn lshift(value: i32, shift: i32) -> i32 {
    value.wrapping_shl(shift as u32)
}

fn rshift(value: i32, shift: i32) -> i32 {
    value >> shift
}

fn div32_16(a32: i32, b16: i16) -> i32 {
    a32 / i32::from(b16)
}

fn clamp_to_u15(value: i32) -> i16 {
    value.clamp(0, 0x7fff) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;

    #[test]
    fn decodes_base_vector_when_residual_is_zero() {
        let order = usize::try_from(SILK_NLSF_CB_WB.order).unwrap();
        let mut output = [0i16; MAX_LPC_ORDER];
        let mut indices = [0i8; MAX_LPC_ORDER + 1];
        indices[0] = 0; // first codebook vector

        nlsf_decode(
            &mut output[..order],
            &indices[..order + 1],
            &SILK_NLSF_CB_WB,
        );

        let start = 0;
        let mut expected = [0i16; MAX_LPC_ORDER];
        for (dst, &src) in expected[..order]
            .iter_mut()
            .zip(SILK_NLSF_CB_WB.cb1_nlsf_q8[start..start + order].iter())
        {
            *dst = i16::from(src) << 7;
        }

        assert_eq!(&output[..order], &expected[..order]);
    }

    #[test]
    fn produces_sorted_stable_output_for_non_zero_residual() {
        let order = usize::try_from(SILK_NLSF_CB_WB.order).unwrap();
        let mut output = [0i16; MAX_LPC_ORDER];
        let mut indices = [0i8; MAX_LPC_ORDER + 1];
        indices[0] = 5; // pick a non-trivial stage-one entry
        for (idx, value) in indices.iter_mut().enumerate().skip(1) {
            *value = match idx % 3 {
                0 => 2,
                1 => -3,
                _ => 0,
            };
        }

        nlsf_decode(
            &mut output[..order],
            &indices[..order + 1],
            &SILK_NLSF_CB_WB,
        );

        for window in output[..order].windows(2) {
            assert!(window[0] <= window[1]);
        }
        assert!(output[0] >= SILK_NLSF_CB_WB.delta_min_q15[0]);
        let delta_min = &SILK_NLSF_CB_WB.delta_min_q15;
        for i in 0..order - 1 {
            let diff = i32::from(output[i + 1]) - i32::from(output[i]);
            assert!(diff >= i32::from(delta_min[i + 1]));
        }
        let upper_guard = (1 << 15) - i32::from(delta_min[order]);
        assert!(i32::from(output[order - 1]) <= upper_guard);
    }

    #[test]
    fn matches_reference_output_for_entropy_weighted_path() {
        let order = usize::try_from(SILK_NLSF_CB_WB.order).unwrap();
        let mut output = [0i16; MAX_LPC_ORDER];
        let mut indices = [0i8; MAX_LPC_ORDER + 1];

        indices[0] = 9;
        indices[1..=order]
            .copy_from_slice(&[-1, 2, 0, -3, 1, -2, 3, -1, 0, 2, -2, 1, -3, 2, 0, -1]);

        nlsf_decode(
            &mut output[..order],
            &indices[..order + 1],
            &SILK_NLSF_CB_WB,
        );

        let mut expected = [0i16; MAX_LPC_ORDER];
        expected[..order].copy_from_slice(&[
            480, 5399, 5689, 5692, 10036, 10039, 14962, 14976, 17231, 20376, 20387, 22638, 22646,
            28263, 28458, 30080,
        ]);

        assert_eq!(&output[..order], &expected[..order]);
    }
}
