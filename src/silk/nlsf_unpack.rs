//! Port of the `silk_NLSF_unpack` helper from `silk/NLSF_unpack.c` in the
//! reference SILK implementation.
//!
//! The routine unpacks per-vector entropy table indices and predictor entries
//! for the stage-two NLSF codebooks. It mirrors the bit manipulations in the C
//! sources, providing Rust callers with a type-safe interface that works with
//! the [`SilkNlsfCb`] metadata already ported to this crate.

use super::SilkNlsfCb;

const NLSF_QUANT_MAX_AMPLITUDE: i16 = 4;
const NLSF_QUANT_STEP: i16 = 2 * NLSF_QUANT_MAX_AMPLITUDE + 1;

/// Mirrors the behaviour of `silk_NLSF_unpack` from the reference C sources.
///
/// # Panics
///
/// This function expects the `ec_ix` and `pred_q8` slices to match the order of
/// the supplied codebook and for `cb1_index` to be in-bounds. Any violation of
/// these preconditions results in a panic, mirroring the debug assertions in the
/// original implementation.
pub fn nlsf_unpack(ec_ix: &mut [i16], pred_q8: &mut [u8], codebook: &SilkNlsfCb, cb1_index: usize) {
    let order = codebook.order as usize;
    assert!(order.is_multiple_of(2), "NLSF order must be even");
    assert_eq!(ec_ix.len(), order, "entropy-index buffer must match order");
    assert_eq!(pred_q8.len(), order, "predictor buffer must match order");
    assert!(cb1_index < codebook.n_vectors as usize);

    let stride = order / 2;
    let start = cb1_index.checked_mul(stride).expect("index overflow");
    let ec_sel = &codebook.ec_sel[start..start + stride];
    let pred_table = codebook.pred_q8;
    let pred_period = order - 1;

    for (pair_idx, &entry) in ec_sel.iter().enumerate() {
        let i = pair_idx * 2;

        let left_mul = i16::from((entry >> 1) & 7);
        ec_ix[i] = left_mul * NLSF_QUANT_STEP;

        let pred_base0 = i + ((entry & 1) as usize) * pred_period;
        debug_assert!(pred_base0 < pred_table.len());
        pred_q8[i] = pred_table[pred_base0];

        let right_mul = i16::from((entry >> 5) & 7);
        ec_ix[i + 1] = right_mul * NLSF_QUANT_STEP;

        let pred_base1 = i + (((entry >> 4) & 1) as usize) * pred_period + 1;
        debug_assert!(pred_base1 < pred_table.len());
        pred_q8[i + 1] = pred_table[pred_base1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;

    #[test]
    fn unpack_first_wideband_vector_matches_reference() {
        let mut ec_ix = [0i16; 16];
        let mut pred_q8 = [0u8; 16];

        nlsf_unpack(&mut ec_ix, &mut pred_q8, &SILK_NLSF_CB_WB, 0);

        assert_eq!(ec_ix, [0; 16]);
        assert_eq!(
            pred_q8,
            [
                175, 148, 160, 176, 178, 173, 174, 164, 177, 174, 196, 182, 198, 192, 155, 68,
            ]
        );
    }

    #[test]
    fn unpack_wideband_vector_with_mixed_entries() {
        let mut ec_ix = [0i16; 16];
        let mut pred_q8 = [0u8; 16];

        nlsf_unpack(&mut ec_ix, &mut pred_q8, &SILK_NLSF_CB_WB, 5);

        assert_eq!(
            ec_ix,
            [
                0, 27, 45, 45, 36, 27, 27, 45, 27, 27, 27, 27, 27, 27, 18, 36
            ]
        );
        assert_eq!(
            pred_q8,
            [
                175, 148, 66, 176, 178, 173, 174, 164, 177, 174, 196, 182, 198, 192, 182, 68,
            ]
        );
    }
}
