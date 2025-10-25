//! Port of `silk/stereo_quant_pred.c`.
//!
//! Quantises the mid/side stereo predictors prior to entropy coding. The
//! implementation mirrors the fixed-point search over the quantisation tables
//! used by the reference C code while relying purely on Rust primitives.

use crate::silk::tables_other::{
    SILK_STEREO_PRED_QUANT_Q13, STEREO_QUANT_SUB_STEPS, STEREO_QUANT_TAB_SIZE,
};

const STEP_Q16: i32 =
    ((1 << 15) + (STEREO_QUANT_SUB_STEPS as i32 / 2)) / STEREO_QUANT_SUB_STEPS as i32;

/// Quantise the mid/side predictors and emit the associated entropy indices.
///
/// `pred_q13` contains the unquantised predictors on entry. On return the
/// predictors are replaced with their quantised counterparts, matching the
/// in-place semantics of the C routine. The returned 2×3 array carries the
/// quantiser stage index, sub-step, and the quotient used by the joint coding
/// path for each predictor.
pub fn stereo_quant_pred(pred_q13: &mut [i32; 2]) -> [[i8; 3]; 2] {
    let mut indices = [[0i8; 3]; 2];

    for (n, pred) in pred_q13.iter_mut().enumerate() {
        let mut err_min_q13 = i32::MAX;
        let mut quant_pred_q13 = 0;
        let mut best_stage = 0;
        let mut best_sub_step = 0;

        'search: for stage in 0..(STEREO_QUANT_TAB_SIZE - 1) {
            let low_q13 = i32::from(SILK_STEREO_PRED_QUANT_Q13[stage]);
            let diff_q13 = i32::from(SILK_STEREO_PRED_QUANT_Q13[stage + 1]) - low_q13;
            let step_q13 = smulwb(diff_q13, STEP_Q16);

            for sub_step in 0..STEREO_QUANT_SUB_STEPS {
                let level_q13 = smlabb(low_q13, step_q13, (2 * sub_step + 1) as i32);
                let err_q13 = abs_q31(pred.wrapping_sub(level_q13));

                if err_q13 < err_min_q13 {
                    err_min_q13 = err_q13;
                    quant_pred_q13 = level_q13;
                    best_stage = stage as i32;
                    best_sub_step = sub_step as i32;
                } else {
                    break 'search;
                }
            }
        }

        indices[n][0] = best_stage as i8;
        indices[n][1] = best_sub_step as i8;
        indices[n][2] = div32_16(best_stage, 3) as i8;
        indices[n][0] = (best_stage - i32::from(indices[n][2]) * 3) as i8;

        *pred = quant_pred_q13;
    }

    pred_q13[0] = pred_q13[0].wrapping_sub(pred_q13[1]);

    indices
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(i32::from((b as i16).wrapping_mul(c as i16)))
}

fn div32_16(a: i32, b: i32) -> i32 {
    a / b
}

fn abs_q31(a: i32) -> i32 {
    if a >= 0 { a } else { a.wrapping_neg() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantises_known_predictors() {
        let mut preds = [-5000, 3000];
        let indices = stereo_quant_pred(&mut preds);
        assert_eq!(preds, [-8305, 3155]);
        assert_eq!(indices, [[1, 4, 1], [0, 0, 3]]);

        let mut preds = [1000, -2000];
        let indices = stereo_quant_pred(&mut preds);
        assert_eq!(preds, [2918, -1885]);
        assert_eq!(indices, [[2, 0, 2], [0, 2, 2]]);

        let mut preds = [12000, 8000];
        let indices = stereo_quant_pred(&mut preds);
        assert_eq!(preds, [3846, 8044]);
        assert_eq!(indices, [[2, 2, 4], [0, 3, 4]]);

        let mut preds = [-14000, -12000];
        let indices = stereo_quant_pred(&mut preds);
        assert_eq!(preds, [-1472, -11892]);
        assert_eq!(indices, [[0, 0, 0], [0, 2, 0]]);

        let mut preds = [4000, 4000];
        let indices = stereo_quant_pred(&mut preds);
        assert_eq!(preds, [0, 3975]);
        assert_eq!(indices, [[0, 2, 3], [0, 2, 3]]);
    }
}
