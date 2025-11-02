use crate::silk::lin2log::lin2log;
use crate::silk::log2lin::log2lin;
use crate::silk::tables_gain::N_LEVELS_QGAIN;

const N_LEVELS_QGAIN_I32: i32 = N_LEVELS_QGAIN as i32;
const MIN_QGAIN_DB: i32 = 2;
const MAX_QGAIN_DB: i32 = 88;
const MIN_DELTA_GAIN_QUANT: i32 = -4;
const MAX_DELTA_GAIN_QUANT: i32 = 36;
const LOG_RANGE_Q7: i32 = ((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6;
const OFFSET: i32 = ((MIN_QGAIN_DB * 128) / 6) + (16 * 128);
const SCALE_Q16: i32 = (65536 * (N_LEVELS_QGAIN_I32 - 1)) / LOG_RANGE_Q7;
const INV_SCALE_Q16: i32 = (65536 * LOG_RANGE_Q7) / (N_LEVELS_QGAIN_I32 - 1);
const MAX_LOG_INPUT_Q7: i32 = 3967;

pub const MAX_NB_SUBFR: usize = 4;

/// Gain scalar quantisation with hysteresis, uniform on the log scale.
pub fn silk_gains_quant(
    ind: &mut [i8],
    gain_q16: &mut [i32],
    prev_ind: &mut i8,
    conditional: bool,
) {
    debug_assert_eq!(ind.len(), gain_q16.len());
    debug_assert!(ind.len() <= MAX_NB_SUBFR);

    let mut prev = i32::from(*prev_ind);
    for (k, (index, gain)) in ind.iter_mut().zip(gain_q16.iter_mut()).enumerate() {
        let mut idx = smulwb(SCALE_Q16, lin2log(*gain) - OFFSET);

        if idx < prev {
            idx += 1;
        }
        idx = limit(idx, 0, N_LEVELS_QGAIN_I32 - 1);

        if k == 0 && !conditional {
            idx = limit(idx, prev + MIN_DELTA_GAIN_QUANT, N_LEVELS_QGAIN_I32 - 1);
            prev = idx;
        } else {
            idx -= prev;
            let threshold = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN_I32 + prev;
            if idx > threshold {
                idx = threshold + ((idx - threshold + 1) >> 1);
            }
            idx = limit(idx, MIN_DELTA_GAIN_QUANT, MAX_DELTA_GAIN_QUANT);
            if idx > threshold {
                prev += (idx << 1) - threshold;
                prev = prev.min(N_LEVELS_QGAIN_I32 - 1);
            } else {
                prev += idx;
            }
            idx -= MIN_DELTA_GAIN_QUANT;
        }

        *index = idx as i8;
        let logits = smulwb(INV_SCALE_Q16, prev) + OFFSET;
        let logits_clamped = logits.min(MAX_LOG_INPUT_Q7);
        *gain = log2lin(logits_clamped);
    }

    *prev_ind = prev as i8;
}

/// Gain scalar dequantisation, uniform on the log scale.
pub fn silk_gains_dequant(gain_q16: &mut [i32], ind: &[i8], prev_ind: &mut i8, conditional: bool) {
    debug_assert_eq!(ind.len(), gain_q16.len());
    debug_assert!(ind.len() <= MAX_NB_SUBFR);

    let mut prev = i32::from(*prev_ind);
    for (k, gain) in gain_q16.iter_mut().enumerate() {
        if k == 0 && !conditional {
            prev = prev.saturating_sub(16);
            prev = prev.max(i32::from(ind[k]));
        } else {
            let ind_tmp = i32::from(ind[k]) + MIN_DELTA_GAIN_QUANT;
            let threshold = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN_I32 + prev;
            if ind_tmp > threshold {
                prev += (ind_tmp << 1) - threshold;
            } else {
                prev += ind_tmp;
            }
        }
        prev = limit(prev, 0, N_LEVELS_QGAIN_I32 - 1);

        let logits = smulwb(INV_SCALE_Q16, prev) + OFFSET;
        let logits_clamped = logits.min(MAX_LOG_INPUT_Q7);
        *gain = log2lin(logits_clamped);
    }

    *prev_ind = prev as i8;
}

/// Compute a unique identifier for the gain index vector.
pub fn silk_gains_id(ind: &[i8]) -> i32 {
    let mut gains_id = 0i32;
    for &value in ind {
        gains_id = gains_id.wrapping_shl(8).wrapping_add(i32::from(value));
    }
    gains_id
}

fn limit(a: i32, limit1: i32, limit2: i32) -> i32 {
    let (min, max) = if limit1 < limit2 {
        (limit1, limit2)
    } else {
        (limit2, limit1)
    };
    a.clamp(min, max)
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silk_gains_quant_matches_reference_case_a() {
        let mut gains = [65536, 98304, 147_456, 229_376];
        let mut indices = [0i8; MAX_NB_SUBFR];
        let mut prev = 10i8;

        silk_gains_quant(&mut indices, &mut gains, &mut prev, false);

        assert_eq!(indices, [6, 0, 5, 7]);
        for (actual, expected) in gains.iter().zip([210_944, 112_640, 131_072, 210_944]) {
            assert_eq!(actual, &expected);
        }
        assert_eq!(prev, 6);

        let expected_gains = [210_944, 112_640, 131_072, 210_944];
        let actual_gains = gains;
        for (actual, expected) in actual_gains.iter().zip(expected_gains.iter()) {
            assert_eq!(actual, expected);
        }

        let mut gains_out = [0; MAX_NB_SUBFR];
        let mut prev_deq = 10i8;
        silk_gains_dequant(&mut gains_out, &indices, &mut prev_deq, false);

        let dequant_gains = gains_out;
        for (actual, expected) in dequant_gains.iter().zip(expected_gains.iter()) {
            assert_eq!(actual, expected);
        }
        assert_eq!(prev_deq, 6);

        assert_eq!(silk_gains_id(&indices), 100_664_583);
    }

    #[test]
    fn silk_gains_quant_matches_reference_case_b() {
        let mut gains = [32_768, 65_536, 180_224, 45_056];
        let mut indices = [0i8; MAX_NB_SUBFR];
        let mut prev = 4i8;

        silk_gains_quant(&mut indices, &mut gains, &mut prev, true);

        assert_eq!(indices, [0, 4, 8, 0]);
        let expected_gains = [81_920, 81_920, 153_600, 81_920];
        let actual_gains = gains;
        for (actual, expected) in actual_gains.iter().zip(expected_gains.iter()) {
            assert_eq!(actual, expected);
        }
        assert_eq!(prev, 0);

        let mut gains_out = [0; MAX_NB_SUBFR];
        let mut prev_deq = 4i8;
        silk_gains_dequant(&mut gains_out, &indices, &mut prev_deq, true);

        let dequant_gains = gains_out;
        for (actual, expected) in dequant_gains.iter().zip(expected_gains.iter()) {
            assert_eq!(actual, expected);
        }
        assert_eq!(prev_deq, 0);

        assert_eq!(silk_gains_id(&indices), 264_192);
    }
}
