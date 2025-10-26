//! Port of the SILK NLSF decoder from `silk/NLSF_decode.c`.
//!
//! Reconstructs a quantised normalised line spectral frequency (NLSF) vector
//! from codebook indices by unpacking the stage-one predictors, de-quantising
//! the residual, and finally stabilising the result. The implementation mirrors
//! the fixed-point arithmetic used by the reference C routine while exposing a
//! safe Rust interface.

use alloc::vec;
use core::convert::TryFrom;

use super::nlsf_stabilize::nlsf_stabilize;
use super::nlsf_unpack::nlsf_unpack;
use super::SilkNlsfCb;

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
    let order = usize::try_from(codebook.order)
        .expect("NLSF order must be representable as usize");
    assert_eq!(nlsf_q15.len(), order, "output buffer must match codebook order");
    assert!(
        indices.len() > order,
        "indices must contain stage-one entry followed by stage-two residuals",
    );

    let cb1_index = usize::try_from(indices[0]).expect("stage-one index must be non-negative");

    let mut ec_ix = vec![0i16; order];
    let mut pred_q8 = vec![0u8; order];
    nlsf_unpack(&mut ec_ix, &mut pred_q8, codebook, cb1_index);

    let mut res_q10 = vec![0i16; order];
    nlsf_residual_dequant(&mut res_q10, &indices[1..order + 1], &pred_q8, codebook);

    let start = cb1_index
        .checked_mul(order)
        .expect("stage-one index multiplication overflowed");
    let cb1 = &codebook.cb1_nlsf_q8[start..start + order];
    let cb1_wght = &codebook.cb1_wght_q9[start..start + order];

    for i in 0..order {
        let residual = i32::from(res_q10[i]);
        let weight = i32::from(cb1_wght[i]);
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
    debug_assert_eq!(output_q10.len(), pred_coef_q8.len());
    debug_assert_eq!(indices.len(), output_q10.len());

    let mut out_q10 = 0i32;
    let quant_step_size_q16 = i32::from(codebook.quant_step_size_q16);

    for (i, (&index, &pred_coef)) in indices
        .iter()
        .zip(pred_coef_q8.iter())
        .enumerate()
        .rev()
    {
        let pred_q10 = rshift(smulbb(out_q10, i32::from(pred_coef)), 8);

        let mut quantised = lshift(i32::from(index), 10);
        if quantised > 0 {
            quantised -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if quantised < 0 {
            quantised += NLSF_QUANT_LEVEL_ADJ_Q10;
        }

        out_q10 = smlawb(pred_q10, quantised, quant_step_size_q16);
        output_q10[i] = out_q10 as i16;
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
    use alloc::vec::Vec;
    use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;

    #[test]
    fn decodes_base_vector_when_residual_is_zero() {
        let order = usize::try_from(SILK_NLSF_CB_WB.order).unwrap();
        let mut output = vec![0i16; order];
        let mut indices = vec![0i8; order + 1];
        indices[0] = 0; // first codebook vector

        nlsf_decode(&mut output, &indices, &SILK_NLSF_CB_WB);

        let start = 0;
        let expected: Vec<i16> = SILK_NLSF_CB_WB.cb1_nlsf_q8[start..start + order]
            .iter()
            .map(|&v| i16::from(v) << 7)
            .collect();

        assert_eq!(output, expected);
    }

    #[test]
    fn produces_sorted_stable_output_for_non_zero_residual() {
        let order = usize::try_from(SILK_NLSF_CB_WB.order).unwrap();
        let mut output = vec![0i16; order];
        let mut indices = vec![0i8; order + 1];
        indices[0] = 5; // pick a non-trivial stage-one entry
        for (idx, value) in indices.iter_mut().enumerate().skip(1) {
            *value = match idx % 3 {
                0 => 2,
                1 => -3,
                _ => 0,
            };
        }

        nlsf_decode(&mut output, &indices, &SILK_NLSF_CB_WB);

        for window in output.windows(2) {
            assert!(window[0] <= window[1]);
        }
        assert!(output[0] >= SILK_NLSF_CB_WB.delta_min_q15[0]);
        let upper_guard = (1 << 15) - i32::from(*SILK_NLSF_CB_WB.delta_min_q15.last().unwrap());
        assert!(i32::from(*output.last().unwrap()) <= upper_guard);
    }
}
