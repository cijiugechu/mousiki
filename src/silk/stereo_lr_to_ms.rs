//! Port of `silk/stereo_LR_to_MS.c`.
//!
//! Converts left/right stereo samples to an adaptive mid/side representation
//! while updating the encoder-side stereo prediction state and emitting the
//! entropy-coding indices used by the SILK encoder.

use alloc::vec;
use alloc::vec::Vec;

use crate::silk::stereo_find_predictor::stereo_find_predictor;
use crate::silk::stereo_quant_pred::stereo_quant_pred;

/// Number of milliseconds over which predictor interpolation occurs.
const STEREO_INTERP_LEN_MS: i32 = 8;

/// Look-ahead in milliseconds for the shaping window that controls how long the
/// encoder keeps transmitting the side channel after collapsing to mono.
const LA_SHAPE_MS: i32 = 5;

/// Maximum number of SILK frames carried in a single Opus packet.
const MAX_FRAMES_PER_PACKET: usize = 3;

/// Fixed-point representation of `STEREO_RATIO_SMOOTH_COEF` in Q16.
const STEREO_RATIO_SMOOTH_COEF_Q16: i32 = 655;

/// Fixed-point representation of `STEREO_RATIO_SMOOTH_COEF / 2` in Q16.
const STEREO_RATIO_SMOOTH_HALF_COEF_Q16: i32 = 328;

/// Fixed-point representation of `1.0` in Q16.
const ONE_Q16: i32 = 1 << 16;

/// Fixed-point representation of `1.0` in Q14.
const ONE_Q14: i32 = 1 << 14;

/// Fixed-point representation of `0.05` in Q14.
const ZERO_POINT_ZERO_FIVE_Q14: i32 = 819;

/// Fixed-point representation of `0.02` in Q14.
const ZERO_POINT_ZERO_TWO_Q14: i32 = 328;

/// Fixed-point representation of `0.95` in Q14.
const ZERO_POINT_NINETY_FIVE_Q14: i32 = 15_565;

/// Fixed-point representation of `13.0` in Q16.
const THIRTEEN_Q16: i32 = 13 * ONE_Q16;

/// Encoder-side stereo state mirrored from `stereo_enc_state` in the C
/// implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StereoEncState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
    pub mid_side_amp_q0: [[i32; 2]; 2],
    pub smth_width_q14: i16,
    pub width_prev_q14: i16,
    pub silent_side_len: i16,
    pub pred_ix: [[[i8; 3]; 2]; MAX_FRAMES_PER_PACKET],
    pub mid_only_flags: [i8; MAX_FRAMES_PER_PACKET],
}

impl Default for StereoEncState {
    fn default() -> Self {
        Self {
            pred_prev_q13: [0; 2],
            s_mid: [0; 2],
            s_side: [0; 2],
            mid_side_amp_q0: [[0; 2]; 2],
            smth_width_q14: 0,
            width_prev_q14: 0,
            silent_side_len: 0,
            pred_ix: [[[0; 3]; 2]; MAX_FRAMES_PER_PACKET],
            mid_only_flags: [0; MAX_FRAMES_PER_PACKET],
        }
    }
}

/// Result returned by [`StereoEncState::lr_to_ms`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StereoConversionResult {
    pub indices: [[i8; 3]; 2],
    pub mid_only_flag: bool,
    pub mid_side_rates_bps: [i32; 2],
}

impl StereoEncState {
    /// Convert left/right stereo samples into an adaptive mid/side
    /// representation.
    ///
    /// Mirrors the behaviour of `silk_stereo_LR_to_MS` by first deriving basic
    /// mid/side signals, tracking their smoothed magnitudes, allocating bitrate
    /// between the channels, and finally applying the predictor interpolation
    /// that removes the correlated mid component from the side channel.
    pub fn lr_to_ms(
        &mut self,
        left: &mut [i16],
        right: &mut [i16],
        mut total_rate_bps: i32,
        prev_speech_act_q8: i32,
        to_mono: bool,
        fs_khz: i32,
    ) -> StereoConversionResult {
        debug_assert_eq!(left.len(), right.len());
        let frame_length = left.len();
        if frame_length == 0 {
            return StereoConversionResult {
                indices: [[0; 3]; 2],
                mid_only_flag: false,
                mid_side_rates_bps: [0; 2],
            };
        }

        let mut mid = Vec::with_capacity(frame_length + 2);
        mid.extend_from_slice(&self.s_mid);
        let mut side = Vec::with_capacity(frame_length + 2);
        side.extend_from_slice(&self.s_side);

        for (&l, &r) in left.iter().zip(right.iter()) {
            let sum = i32::from(l) + i32::from(r);
            let diff = i32::from(l) - i32::from(r);
            mid.push(sat16(rshift_round(sum, 1)));
            side.push(sat16(rshift_round(diff, 1)));
        }

        self.s_mid
            .copy_from_slice(&mid[frame_length..frame_length + 2]);
        self.s_side
            .copy_from_slice(&side[frame_length..frame_length + 2]);

        let mut lp_mid = Vec::with_capacity(frame_length);
        let mut hp_mid = Vec::with_capacity(frame_length);
        let mut lp_side = Vec::with_capacity(frame_length);
        let mut hp_side = Vec::with_capacity(frame_length);

        for n in 0..frame_length {
            let sum = rshift_round(
                add_lshift32(
                    i32::from(mid[n]) + i32::from(mid[n + 2]),
                    i32::from(mid[n + 1]),
                    1,
                ),
                2,
            );
            let lp = sat16(sum);
            lp_mid.push(lp);
            hp_mid.push(sat16(i32::from(mid[n + 1]) - sum));

            let sum = rshift_round(
                add_lshift32(
                    i32::from(side[n]) + i32::from(side[n + 2]),
                    i32::from(side[n + 1]),
                    1,
                ),
                2,
            );
            let lp_s = sat16(sum);
            lp_side.push(lp_s);
            hp_side.push(sat16(i32::from(side[n + 1]) - sum));
        }

        let is_10ms_frame = frame_length as i32 == 10 * fs_khz;
        let mut smooth_coef_q16 = if is_10ms_frame {
            STEREO_RATIO_SMOOTH_HALF_COEF_Q16
        } else {
            STEREO_RATIO_SMOOTH_COEF_Q16
        };
        smooth_coef_q16 = smulwb(
            smulbb(prev_speech_act_q8, prev_speech_act_q8),
            smooth_coef_q16,
        );

        let (pred_lp_q13, lp_ratio_q14) = stereo_find_predictor(
            &lp_mid,
            &lp_side,
            &mut self.mid_side_amp_q0[0],
            smooth_coef_q16,
        );
        let (pred_hp_q13, hp_ratio_q14) = stereo_find_predictor(
            &hp_mid,
            &hp_side,
            &mut self.mid_side_amp_q0[1],
            smooth_coef_q16,
        );
        let mut pred_q13 = [pred_lp_q13, pred_hp_q13];

        let mut frac_q16 = smlabb(hp_ratio_q14, lp_ratio_q14, 3);
        frac_q16 = frac_q16.min(ONE_Q16);

        total_rate_bps -= if is_10ms_frame { 1200 } else { 600 };
        if total_rate_bps < 1 {
            total_rate_bps = 1;
        }
        let min_mid_rate_bps = smlabb(2000, fs_khz, 600);

        let mut mid_side_rates_bps = [0; 2];
        let frac_3_q16 = mul(3, frac_q16);
        mid_side_rates_bps[0] = div32_varq(total_rate_bps, THIRTEEN_Q16 + frac_3_q16, 19);

        let mut width_q14;
        let indices;
        let mut mid_only_flag = false;

        if mid_side_rates_bps[0] < min_mid_rate_bps {
            mid_side_rates_bps[0] = min_mid_rate_bps;
            mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
            let numerator = (mid_side_rates_bps[1] << 1) - min_mid_rate_bps;
            let denom = smulwb(ONE_Q16 + frac_3_q16, min_mid_rate_bps);
            width_q14 = div32_varq(numerator, denom, 16 + 2);
            width_q14 = width_q14.clamp(0, ONE_Q14);
        } else {
            mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
            width_q14 = ONE_Q14;
        }

        let mut smth_width_q14 = i32::from(self.smth_width_q14);
        smth_width_q14 = smlawb(smth_width_q14, width_q14 - smth_width_q14, smooth_coef_q16);
        self.smth_width_q14 = sat16(smth_width_q14);

        if to_mono {
            width_q14 = 0;
            pred_q13[0] = 0;
            pred_q13[1] = 0;
            indices = stereo_quant_pred(&mut pred_q13);
        } else if self.width_prev_q14 == 0
            && (8 * total_rate_bps < 13 * min_mid_rate_bps
                || smulwb(frac_q16, smth_width_q14) < ZERO_POINT_ZERO_FIVE_Q14)
        {
            pred_q13[0] = rshift(smulbb(smth_width_q14, pred_q13[0]), 14);
            pred_q13[1] = rshift(smulbb(smth_width_q14, pred_q13[1]), 14);
            indices = stereo_quant_pred(&mut pred_q13);
            width_q14 = 0;
            pred_q13 = [0, 0];
            mid_side_rates_bps[0] = total_rate_bps;
            mid_side_rates_bps[1] = 0;
            mid_only_flag = true;
        } else if self.width_prev_q14 != 0
            && (8 * total_rate_bps < 11 * min_mid_rate_bps
                || smulwb(frac_q16, smth_width_q14) < ZERO_POINT_ZERO_TWO_Q14)
        {
            pred_q13[0] = rshift(smulbb(smth_width_q14, pred_q13[0]), 14);
            pred_q13[1] = rshift(smulbb(smth_width_q14, pred_q13[1]), 14);
            indices = stereo_quant_pred(&mut pred_q13);
            width_q14 = 0;
            pred_q13 = [0, 0];
        } else if smth_width_q14 > ZERO_POINT_NINETY_FIVE_Q14 {
            indices = stereo_quant_pred(&mut pred_q13);
            width_q14 = ONE_Q14;
        } else {
            pred_q13[0] = rshift(smulbb(smth_width_q14, pred_q13[0]), 14);
            pred_q13[1] = rshift(smulbb(smth_width_q14, pred_q13[1]), 14);
            indices = stereo_quant_pred(&mut pred_q13);
            width_q14 = smth_width_q14;
        }

        if mid_only_flag {
            let delta = frame_length as i32 - STEREO_INTERP_LEN_MS * fs_khz;
            let mut silent_side_len = i32::from(self.silent_side_len) + delta;
            if silent_side_len < LA_SHAPE_MS * fs_khz {
                mid_only_flag = false;
            } else {
                silent_side_len = 10_000;
            }
            self.silent_side_len = sat16(silent_side_len);
        } else {
            self.silent_side_len = 0;
        }

        if !mid_only_flag && mid_side_rates_bps[1] < 1 {
            mid_side_rates_bps[1] = 1;
            mid_side_rates_bps[0] = mid_side_rates_bps[0].max(total_rate_bps - 1).max(1);
        }

        let interp_len = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
        let denom_q16 = div32_16(ONE_Q16, STEREO_INTERP_LEN_MS * fs_khz);
        let mut pred0_q13 = -i32::from(self.pred_prev_q13[0]);
        let mut pred1_q13 = -i32::from(self.pred_prev_q13[1]);
        let mut w_q24 = lshift(i32::from(self.width_prev_q14), 10);
        let delta0_q13 = -rshift_round(
            smulbb(pred_q13[0] - i32::from(self.pred_prev_q13[0]), denom_q16),
            16,
        );
        let delta1_q13 = -rshift_round(
            smulbb(pred_q13[1] - i32::from(self.pred_prev_q13[1]), denom_q16),
            16,
        );
        let deltaw_q24 = lshift(
            smulwb(width_q14 - i32::from(self.width_prev_q14), denom_q16),
            10,
        );

        let mut side_out = vec![0i16; frame_length];
        let interp_end = interp_len.min(frame_length);
        for n in 0..interp_end {
            pred0_q13 = pred0_q13.wrapping_add(delta0_q13);
            pred1_q13 = pred1_q13.wrapping_add(delta1_q13);
            w_q24 = w_q24.wrapping_add(deltaw_q24);

            let sum = lshift(
                add_lshift32(
                    i32::from(mid[n]) + i32::from(mid[n + 2]),
                    i32::from(mid[n + 1]),
                    1,
                ),
                9,
            );
            let sum = smlawb(smulwb(w_q24, i32::from(side[n + 1])), sum, pred0_q13);
            let sum = smlawb(sum, lshift(i32::from(mid[n + 1]), 11), pred1_q13);
            side_out[n] = sat16(rshift_round(sum, 8));
        }

        pred0_q13 = -pred_q13[0];
        pred1_q13 = -pred_q13[1];
        w_q24 = lshift(width_q14, 10);
        for n in interp_end..frame_length {
            let sum = lshift(
                add_lshift32(
                    i32::from(mid[n]) + i32::from(mid[n + 2]),
                    i32::from(mid[n + 1]),
                    1,
                ),
                9,
            );
            let sum = smlawb(smulwb(w_q24, i32::from(side[n + 1])), sum, pred0_q13);
            let sum = smlawb(sum, lshift(i32::from(mid[n + 1]), 11), pred1_q13);
            side_out[n] = sat16(rshift_round(sum, 8));
        }

        self.pred_prev_q13[0] = sat16(pred_q13[0]);
        self.pred_prev_q13[1] = sat16(pred_q13[1]);
        self.width_prev_q14 = sat16(width_q14);

        for (dst, &value) in left.iter_mut().zip(mid.iter().skip(2)) {
            *dst = value;
        }
        right.copy_from_slice(&side_out);

        StereoConversionResult {
            indices,
            mid_only_flag,
            mid_side_rates_bps,
        }
    }
}

fn add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b.wrapping_shl(shift as u32))
}

fn lshift(value: i32, shift: i32) -> i32 {
    value.wrapping_shl(shift as u32)
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from((a as i16).wrapping_mul(b as i16))
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(i32::from((b as i16).wrapping_mul(c as i16)))
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let product = (i64::from(b) * i64::from(c as i16)) >> 16;
    a.wrapping_add(product as i32)
}

fn mul(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

fn rshift(value: i32, shift: i32) -> i32 {
    if shift <= 0 { value } else { value >> shift }
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else {
        (value + (1 << (shift - 1))) >> shift
    }
}

fn sat16(value: i32) -> i16 {
    if value > i32::from(i16::MAX) {
        i16::MAX
    } else if value < i32::from(i16::MIN) {
        i16::MIN
    } else {
        value as i16
    }
}

fn div32_16(a: i32, b: i32) -> i32 {
    a / b
}

fn div32_varq(a32: i32, b32: i32, q_res: i32) -> i32 {
    let abs_a = if a32 == i32::MIN { i32::MAX } else { a32.abs() };
    let abs_b = if b32 == i32::MIN { i32::MAX } else { b32.abs() };
    let a_headroom = clz32(abs_a) - 1;
    let mut a_norm = lshift(a32, a_headroom);
    let b_headroom = clz32(abs_b) - 1;
    let b_norm = lshift(b32, b_headroom);

    let denom16 = rshift(b_norm, 16);
    debug_assert!(denom16 != 0);
    let b_inv = div32_16(i32::MAX >> 2, denom16);

    let mut result = smulwb(a_norm, b_inv);
    let correction = lshift(smmul(b_norm, result), 3);
    a_norm = a_norm.wrapping_sub(correction);
    result = smlawb(result, a_norm, b_inv);

    let lshift = 29 + a_headroom - b_headroom - q_res;
    if lshift < 0 {
        lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        rshift(result, lshift)
    } else {
        0
    }
}

fn clz32(value: i32) -> i32 {
    if value == 0 {
        32
    } else {
        (value as u32).leading_zeros() as i32
    }
}

fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

fn lshift_sat32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value
    } else {
        let shifted = (i64::from(value)) << shift;
        if shifted > i64::from(i32::MAX) {
            i32::MAX
        } else if shifted < i64::from(i32::MIN) {
            i32::MIN
        } else {
            shifted as i32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn converts_left_right_to_mid_side_with_zero_state() {
        let mut state = StereoEncState::default();
        let mut left = vec![1200, -800, 400, -200, 100, -50, 25, -12];
        let mut right = left.clone();
        let original_left = left.clone();
        let original_right = right.clone();

        let result = state.lr_to_ms(&mut left, &mut right, 24_000, 0, false, 8);

        let expected_mid: Vec<i16> = original_left
            .iter()
            .zip(original_right.iter())
            .map(|(&l, &r)| sat16(rshift_round(i32::from(l) + i32::from(r), 1)))
            .collect();
        assert_eq!(left, expected_mid);
        assert!(right.iter().all(|&sample| sample == 0));
        assert_eq!(state.s_mid, [left[left.len() - 2], left[left.len() - 1]]);
        assert_eq!(state.s_side, [0, 0]);
        assert!(result.mid_side_rates_bps[0] >= 1);
        assert!(result.mid_side_rates_bps[1] >= 0);
    }

    #[test]
    fn collapses_width_when_forced_to_mono() {
        let mut state = StereoEncState::default();
        let mut left = vec![500, -400, 300, -200, 100, 0, -100, 200, -300, 400];
        let mut right = vec![-500, 400, -300, 200, -100, 0, 100, -200, 300, -400];

        let result = state.lr_to_ms(&mut left, &mut right, 18_000, 64, true, 8);

        assert_eq!(state.width_prev_q14, 0);
        assert!(!result.mid_only_flag);
    }
}
