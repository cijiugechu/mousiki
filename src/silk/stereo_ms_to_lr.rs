//! Port of `silk/stereo_MS_to_LR.c`.
//!
//! Converts an adaptive mid/side representation back to left/right stereo
//! samples while updating the decoder's stereo prediction state.

/// Number of milliseconds over which predictor interpolation occurs.
const STEREO_INTERP_LEN_MS: i32 = 8;

/// Decoder-side state that buffers the previous stereo predictors and the
/// two-sample overlap used when reconstructing left/right channels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StereoDecState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
}

impl StereoDecState {
    /// Convert adaptive mid/side samples back to left/right stereo.
    ///
    /// Mirrors the behaviour of `silk_stereo_MS_to_LR` by first restoring the
    /// decoder's two-sample overlap history, then interpolating the predictor
    /// coefficients for the first `STEREO_INTERP_LEN_MS` milliseconds before
    /// applying the steady-state predictors. The updated side channel is finally
    /// combined with the mid channel to yield left and right PCM samples.
    pub fn ms_to_lr(
        &mut self,
        mid: &mut [i16],
        side: &mut [i16],
        pred_q13: &[i32; 2],
        fs_khz: i32,
        frame_length: usize,
    ) {
        debug_assert!(frame_length + 2 <= mid.len());
        debug_assert!(frame_length + 2 <= side.len());

        // Restore the two-sample history for the predictor filters.
        mid[..2].copy_from_slice(&self.s_mid);
        side[..2].copy_from_slice(&self.s_side);

        // Persist the final two samples for the next call.
        self.s_mid
            .copy_from_slice(&mid[frame_length..frame_length + 2]);
        self.s_side
            .copy_from_slice(&side[frame_length..frame_length + 2]);

        let mut pred0_q13 = i32::from(self.pred_prev_q13[0]);
        let mut pred1_q13 = i32::from(self.pred_prev_q13[1]);

        let interp_samples = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
        let denom_q16 = div32_16(1 << 16, STEREO_INTERP_LEN_MS * fs_khz);
        let delta0_q13 = rshift_round(smulbb(pred_q13[0].wrapping_sub(pred0_q13), denom_q16), 16);
        let delta1_q13 = rshift_round(smulbb(pred_q13[1].wrapping_sub(pred1_q13), denom_q16), 16);

        for n in 0..interp_samples.min(frame_length) {
            pred0_q13 = pred0_q13.wrapping_add(delta0_q13);
            pred1_q13 = pred1_q13.wrapping_add(delta1_q13);

            let sum = lshift(
                add_lshift32(
                    i32::from(mid[n]).wrapping_add(i32::from(mid[n + 2])),
                    i32::from(mid[n + 1]),
                    1,
                ),
                9,
            );
            let sum = smlawb(lshift(i32::from(side[n + 1]), 8), sum, pred0_q13);
            let sum = smlawb(sum, lshift(i32::from(mid[n + 1]), 11), pred1_q13);
            side[n + 1] = sat16(rshift_round(sum, 8));
        }

        pred0_q13 = pred_q13[0];
        pred1_q13 = pred_q13[1];

        for n in interp_samples.min(frame_length)..frame_length {
            let sum = lshift(
                add_lshift32(
                    i32::from(mid[n]).wrapping_add(i32::from(mid[n + 2])),
                    i32::from(mid[n + 1]),
                    1,
                ),
                9,
            );
            let sum = smlawb(lshift(i32::from(side[n + 1]), 8), sum, pred0_q13);
            let sum = smlawb(sum, lshift(i32::from(mid[n + 1]), 11), pred1_q13);
            side[n + 1] = sat16(rshift_round(sum, 8));
        }

        self.pred_prev_q13[0] = pred_q13[0] as i16;
        self.pred_prev_q13[1] = pred_q13[1] as i16;

        for n in 0..frame_length {
            let mid_val = i32::from(mid[n + 1]);
            let side_val = i32::from(side[n + 1]);
            let sum = mid_val.wrapping_add(side_val);
            let diff = mid_val.wrapping_sub(side_val);
            mid[n + 1] = sat16(sum);
            side[n + 1] = sat16(diff);
        }
    }
}

fn div32_16(a: i32, b: i32) -> i32 {
    a / b
}

fn add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b.wrapping_shl(shift as u32))
}

fn lshift(value: i32, shift: i32) -> i32 {
    value.wrapping_shl(shift as u32)
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16).wrapping_mul(i32::from(b as i16))
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let c16 = i64::from(c as i16);
    let prod = (i64::from(b) * c16) >> 16;
    a.wrapping_add(prod as i32)
}

fn rshift_round(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        value
    } else if shift == 1 {
        (value >> 1).wrapping_add(value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn converts_mid_side_to_left_right_with_zero_predictor() {
        let mut state = StereoDecState::default();
        state.s_mid = [10, -11];
        state.s_side = [7, -6];

        let frame_length = 16;
        let mut mid = vec![0i16; frame_length + 2];
        let mut side = vec![0i16; frame_length + 2];

        for (idx, sample) in (2..frame_length + 2).enumerate() {
            mid[sample] = (idx as i16).wrapping_mul(3);
            side[sample] = -(idx as i16).wrapping_mul(2);
        }

        let mut expected_mid = mid.clone();
        let mut expected_side = side.clone();
        let mut expected_state = state;

        expected_mid[..2].copy_from_slice(&expected_state.s_mid);
        expected_side[..2].copy_from_slice(&expected_state.s_side);

        expected_state.s_mid = [expected_mid[frame_length], expected_mid[frame_length + 1]];
        expected_state.s_side = [expected_side[frame_length], expected_side[frame_length + 1]];

        for n in 0..frame_length {
            let mid_val = i32::from(expected_mid[n + 1]);
            let side_val = i32::from(expected_side[n + 1]);
            expected_mid[n + 1] = sat16(mid_val + side_val);
            expected_side[n + 1] = sat16(mid_val - side_val);
        }

        state.ms_to_lr(&mut mid, &mut side, &[0; 2], 16, frame_length);

        assert_eq!(state, expected_state);
        assert_eq!(mid, expected_mid);
        assert_eq!(side, expected_side);
    }

    #[test]
    fn updates_state_and_applies_predictors() {
        let mut state = StereoDecState {
            pred_prev_q13: [500, -300],
            s_mid: [2, -3],
            s_side: [-4, 5],
        };

        let frame_length = 20;
        let mut mid = vec![0i16; frame_length + 2];
        let mut side = vec![0i16; frame_length + 2];

        for n in 2..frame_length + 2 {
            mid[n] = (100 - n as i32) as i16;
            side[n] = (n as i32 - 50) as i16;
        }

        let pred_q13 = [1200, -900];

        let mut expected_state = state;
        let mut expected_mid = mid.clone();
        let mut expected_side = side.clone();

        expected_mid[..2].copy_from_slice(&expected_state.s_mid);
        expected_side[..2].copy_from_slice(&expected_state.s_side);
        expected_state
            .s_mid
            .copy_from_slice(&expected_mid[frame_length..frame_length + 2]);
        expected_state
            .s_side
            .copy_from_slice(&expected_side[frame_length..frame_length + 2]);

        let interp_samples = (STEREO_INTERP_LEN_MS * 8) as usize;
        let denom_q16 = div32_16(1 << 16, STEREO_INTERP_LEN_MS * 8);
        let mut pred0_q13 = i32::from(expected_state.pred_prev_q13[0]);
        let mut pred1_q13 = i32::from(expected_state.pred_prev_q13[1]);
        let delta0_q13 = rshift_round(smulbb(pred_q13[0] - pred0_q13, denom_q16), 16);
        let delta1_q13 = rshift_round(smulbb(pred_q13[1] - pred1_q13, denom_q16), 16);

        for n in 0..interp_samples.min(frame_length) {
            pred0_q13 = pred0_q13.wrapping_add(delta0_q13);
            pred1_q13 = pred1_q13.wrapping_add(delta1_q13);

            let sum = lshift(
                add_lshift32(
                    i32::from(expected_mid[n]).wrapping_add(i32::from(expected_mid[n + 2])),
                    i32::from(expected_mid[n + 1]),
                    1,
                ),
                9,
            );
            let sum = smlawb(lshift(i32::from(expected_side[n + 1]), 8), sum, pred0_q13);
            let sum = smlawb(sum, lshift(i32::from(expected_mid[n + 1]), 11), pred1_q13);
            expected_side[n + 1] = sat16(rshift_round(sum, 8));
        }

        pred0_q13 = pred_q13[0];
        pred1_q13 = pred_q13[1];

        for n in interp_samples.min(frame_length)..frame_length {
            let sum = lshift(
                add_lshift32(
                    i32::from(expected_mid[n]).wrapping_add(i32::from(expected_mid[n + 2])),
                    i32::from(expected_mid[n + 1]),
                    1,
                ),
                9,
            );
            let sum = smlawb(lshift(i32::from(expected_side[n + 1]), 8), sum, pred0_q13);
            let sum = smlawb(sum, lshift(i32::from(expected_mid[n + 1]), 11), pred1_q13);
            expected_side[n + 1] = sat16(rshift_round(sum, 8));
        }

        expected_state.pred_prev_q13[0] = pred_q13[0] as i16;
        expected_state.pred_prev_q13[1] = pred_q13[1] as i16;

        for n in 0..frame_length {
            let mid_val = i32::from(expected_mid[n + 1]);
            let side_val = i32::from(expected_side[n + 1]);
            expected_mid[n + 1] = sat16(mid_val + side_val);
            expected_side[n + 1] = sat16(mid_val - side_val);
        }

        state.ms_to_lr(&mut mid, &mut side, &pred_q13, 8, frame_length);

        assert_eq!(state, expected_state);
        assert_eq!(mid, expected_mid);
        assert_eq!(side, expected_side);
    }
}
