//! Port of `silk/stereo_encode_pred.c`.
//!
//! Entropy-codes the quantised mid/side stereo predictor indices and the
//! optional mid-only flag using the same range coder contexts as the
//! reference SILK implementation.

use core::convert::TryFrom;

use crate::range::RangeEncoder;
use crate::silk::icdf::{STEREO_ONLY_CODE_MID, STEREO_PREDICTOR_JOINT, UNIFORM3, UNIFORM5};
use crate::silk::tables_other::STEREO_QUANT_SUB_STEPS;

/// Encode the mid/side stereo predictor indices into the range coder.
///
/// Mirrors the behaviour of `silk_stereo_encode_pred` by jointly coding the
/// quotient indices for both predictors, followed by the individual stage and
/// sub-step indices for each channel.
pub fn stereo_encode_pred(range_encoder: &mut RangeEncoder, indices: &[[i8; 3]; 2]) {
    let joint = 5 * i32::from(indices[0][2]) + i32::from(indices[1][2]);
    debug_assert!((0..25).contains(&joint));

    range_encoder.encode_symbol_with_icdf(joint as usize, STEREO_PREDICTOR_JOINT);

    for idx in indices {
        let stage = usize::try_from(idx[0]).expect("stage index must be non-negative");
        let sub_step = usize::try_from(idx[1]).expect("sub-step index must be non-negative");

        debug_assert!(stage < 3);
        debug_assert!(sub_step < STEREO_QUANT_SUB_STEPS);

        range_encoder.encode_symbol_with_icdf(stage, UNIFORM3);
        range_encoder.encode_symbol_with_icdf(sub_step, UNIFORM5);
    }
}

/// Encode the mid-only flag that signals stereo collapse into the range coder.
pub fn stereo_encode_mid_only(range_encoder: &mut RangeEncoder, mid_only: bool) {
    range_encoder.encode_symbol_with_icdf(usize::from(mid_only), STEREO_ONLY_CODE_MID);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt::EcDec;
    use crate::range::RangeEncoder;
    use crate::silk::stereo_decode_pred::{stereo_decode_mid_only, stereo_decode_pred};
    use crate::silk::stereo_quant_pred::stereo_quant_pred;

    #[test]
    fn encodes_predictor_indices_and_mid_only_flag() {
        let mut predictors = [-5000, 3000];
        let indices = stereo_quant_pred(&mut predictors);

        let mut encoder = RangeEncoder::new();
        stereo_encode_pred(&mut encoder, &indices);
        stereo_encode_mid_only(&mut encoder, true);
        let mut storage = encoder.finish();
        let mut decoder = EcDec::new(storage.as_mut_slice());
        let mut decoded = [0; 2];
        stereo_decode_pred(&mut decoder, &mut decoded);
        assert_eq!(decoded, predictors);
        assert!(stereo_decode_mid_only(&mut decoder));
    }
}
