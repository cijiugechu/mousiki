//! Port of `silk/stereo_decode_pred.c`.
//!
//! Decodes the mid/side stereo predictor indices and reconstructs the
//! quantised predictor pair as described in the SILK reference implementation.

use crate::range::RangeDecoder;
use crate::silk::icdf::{STEREO_ONLY_CODE_MID, STEREO_PREDICTOR_JOINT, UNIFORM3, UNIFORM5};
use crate::silk::tables_other::{SILK_STEREO_PRED_QUANT_Q13, STEREO_QUANT_SUB_STEPS};

const HALF_STEP_Q16: i32 =
    ((1 << 15) + (STEREO_QUANT_SUB_STEPS as i32 / 2)) / STEREO_QUANT_SUB_STEPS as i32;

/// Decode the quantised mid/side predictor pair from the entropy coder.
///
/// Mirrors `silk_stereo_decode_pred` from the C sources by first unpacking
/// the jointly-coded quotient index, then reading the residual indices for
/// each predictor and reconstructing the quantised predictor pair.
pub fn stereo_decode_pred(range_decoder: &mut RangeDecoder<'_>, pred_q13: &mut [i32; 2]) {
    let joint_index = range_decoder.decode_symbol_with_icdf(STEREO_PREDICTOR_JOINT) as i32;

    let mut indices = [[0i32; 3]; 2];
    indices[0][2] = joint_index / 5;
    indices[1][2] = joint_index - 5 * indices[0][2];

    for idx in &mut indices {
        idx[0] = range_decoder.decode_symbol_with_icdf(UNIFORM3) as i32;
        idx[1] = range_decoder.decode_symbol_with_icdf(UNIFORM5) as i32;
    }

    for (channel, idx) in indices.iter_mut().enumerate() {
        idx[0] += 3 * idx[2];

        let base = idx[0] as usize;
        let low_q13 = i32::from(SILK_STEREO_PRED_QUANT_Q13[base]);
        let next = i32::from(SILK_STEREO_PRED_QUANT_Q13[base + 1]);
        let step_q13 = smulwb(next - low_q13, HALF_STEP_Q16);

        pred_q13[channel] = smlabb(low_q13, step_q13, 2 * idx[1] + 1);
    }

    pred_q13[0] = pred_q13[0].wrapping_sub(pred_q13[1]);
}

/// Decode the flag that signals whether only the mid channel was coded.
pub fn stereo_decode_mid_only(range_decoder: &mut RangeDecoder<'_>) -> bool {
    range_decoder.decode_symbol_with_icdf(STEREO_ONLY_CODE_MID) == 1
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    let prod = i32::from(b as i16).wrapping_mul(i32::from(c as i16));
    a.wrapping_add(prod)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::RangeEncoder;
    use alloc::vec::Vec;

    fn to_raw_icdf(ctx: crate::silk::icdf::ICDFContext) -> Vec<u16> {
        ctx.dist_table
            .iter()
            .map(|&value| (ctx.total - value as u32) as u16)
            .collect()
    }

    #[test]
    fn decodes_predictors_and_mid_only_flag() {
        let mut encoder = RangeEncoder::new();

        encoder.encode_icdf16(7, &to_raw_icdf(STEREO_PREDICTOR_JOINT), 8);
        encoder.encode_icdf16(1, &to_raw_icdf(UNIFORM3), 8);
        encoder.encode_icdf16(3, &to_raw_icdf(UNIFORM5), 8);
        encoder.encode_icdf16(2, &to_raw_icdf(UNIFORM3), 8);
        encoder.encode_icdf16(4, &to_raw_icdf(UNIFORM5), 8);
        encoder.encode_icdf16(1, &to_raw_icdf(STEREO_ONLY_CODE_MID), 8);

        let data = encoder.finish();
        let mut decoder = RangeDecoder::init(&data);
        let mut predictors = [0; 2];

        stereo_decode_pred(&mut decoder, &mut predictors);

        let mut expected = [0; 2];
        let mut indices = [[0i32; 3]; 2];
        indices[0][2] = 7 / 5;
        indices[1][2] = 7 - 5 * indices[0][2];
        indices[0][0] = 1;
        indices[0][1] = 3;
        indices[1][0] = 2;
        indices[1][1] = 4;

        for (channel, idx) in indices.iter_mut().enumerate() {
            idx[0] += 3 * idx[2];
            let base = idx[0] as usize;
            let low_q13 = i32::from(SILK_STEREO_PRED_QUANT_Q13[base]);
            let next = i32::from(SILK_STEREO_PRED_QUANT_Q13[base + 1]);
            let step_q13 = smulwb(next - low_q13, HALF_STEP_Q16);
            expected[channel] = smlabb(low_q13, step_q13, 2 * idx[1] + 1);
        }

        expected[0] = expected[0].wrapping_sub(expected[1]);

        assert_eq!(predictors, expected);
        assert!(stereo_decode_mid_only(&mut decoder));
    }
}
