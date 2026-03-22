use core::arch::aarch64::{
    int32x4_t, vaddq_s32, vaddq_s64, vdupq_n_s32, vextq_s32, vget_high_s32, vget_low_s32,
    vgetq_lane_s32, vld1q_s32, vld1q_s64, vmull_s32, vqdmulhq_s32, vshrq_n_s64, vst1q_s64,
    vsubq_s32,
};

use crate::silk::warped_autocorrelation::{MAX_SHAPE_LPC_ORDER, QC, QS, clz64};

const MAX_NEON_INPUT_LEN: usize = 320;
const INPUT_BUF_LEN: usize = MAX_NEON_INPUT_LEN + 2 * MAX_SHAPE_LPC_ORDER + 4;
const STATE_BUF_LEN: usize = MAX_NEON_INPUT_LEN + MAX_SHAPE_LPC_ORDER + 4;
#[inline]
unsafe fn calc_corr(input_qs: *const i32, corr_qc: *mut i64, offset: usize, state: int32x4_t) {
    let input_vec = unsafe { vld1q_s32(input_qs.add(offset)) };

    let mut corr0 = unsafe { vld1q_s64(corr_qc.add(offset)) };
    let mut corr1 = unsafe { vld1q_s64(corr_qc.add(offset + 2)) };

    let prod0 = unsafe { vmull_s32(vget_low_s32(state), vget_low_s32(input_vec)) };
    let prod1 = unsafe { vmull_s32(vget_high_s32(state), vget_high_s32(input_vec)) };

    corr0 = unsafe { vaddq_s64(corr0, vshrq_n_s64::<16>(prod0)) };
    corr1 = unsafe { vaddq_s64(corr1, vshrq_n_s64::<16>(prod1)) };

    unsafe {
        vst1q_s64(corr_qc.add(offset), corr0);
        vst1q_s64(corr_qc.add(offset + 2), corr1);
    }
}

#[inline]
unsafe fn calc_state(
    state_qs0: int32x4_t,
    state_qs0_1: int32x4_t,
    state_qs1_1: int32x4_t,
    warping_q16: int32x4_t,
) -> int32x4_t {
    let delta = unsafe { vsubq_s32(state_qs0, state_qs0_1) };
    let warped = unsafe { vqdmulhq_s32(delta, warping_q16) };
    unsafe { vaddq_s32(state_qs1_1, warped) }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(super) fn warped_autocorrelation(
    corr: &mut [i32],
    input: &[i16],
    warping_q16: i32,
    order: usize,
) -> i32 {
    if order < 6 || input.len() > MAX_NEON_INPUT_LEN {
        return super::warped_autocorrelation_scalar(corr, input, warping_q16, order);
    }

    let order_t = (order + 3) & !3;
    let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];

    let mut input_qst = [0i32; INPUT_BUF_LEN];
    let converted_start = MAX_SHAPE_LPC_ORDER;
    for (dst, &sample) in input_qst[converted_start..converted_start + input.len()]
        .iter_mut()
        .zip(input.iter())
    {
        *dst = i32::from(sample) << QS;
    }

    let input_base = MAX_SHAPE_LPC_ORDER - order_t;
    let len_plus_order = input.len() + order;

    unsafe {
        let warping_q16_vec = vdupq_n_s32(warping_q16 << 15);
        let base_ptr = input_qst.as_ptr().add(input_base);
        let mut in_ptr = base_ptr.add(order_t);
        let mut o = order_t;
        let mut state = [0i32; STATE_BUF_LEN];

        while o > 4 {
            let mut state0_0 = vdupq_n_s32(0);
            let mut state0_1 = vdupq_n_s32(0);
            let mut state1_0 = vdupq_n_s32(0);
            let mut state1_1 = vdupq_n_s32(0);

            for n in 0..len_plus_order {
                calc_corr(base_ptr.add(n), corr_qc.as_mut_ptr(), o - 8, state0_0);
                calc_corr(base_ptr.add(n), corr_qc.as_mut_ptr(), o - 4, state0_1);

                let mut state2_1 = vld1q_s32(in_ptr.add(n));
                state[n] = vgetq_lane_s32::<0>(state0_0);
                let state2_0 = vextq_s32(state0_0, state0_1, 1);
                state2_1 = vextq_s32(state0_1, state2_1, 1);

                state0_0 = calc_state(state0_0, state2_0, state1_0, warping_q16_vec);
                state0_1 = calc_state(state0_1, state2_1, state1_1, warping_q16_vec);

                state1_0 = state2_0;
                state1_1 = state2_1;
            }

            in_ptr = state.as_ptr();
            o -= 8;
        }

        if o == 4 {
            let mut state0_0 = vdupq_n_s32(0);
            let mut state1_0 = vdupq_n_s32(0);
            let mut state_ptr = state.as_mut_ptr();
            let mut input_ptr = base_ptr;

            for _ in 0..len_plus_order {
                calc_corr(input_ptr, corr_qc.as_mut_ptr(), 0, state0_0);
                let mut state2_0 = vld1q_s32(state_ptr);
                *state_ptr = vgetq_lane_s32::<0>(state0_0);
                state2_0 = vextq_s32(state0_0, state2_0, 1);

                state0_0 = calc_state(state0_0, state2_0, state1_0, warping_q16_vec);
                state1_0 = state2_0;

                input_ptr = input_ptr.add(1);
                state_ptr = state_ptr.add(1);
            }
        }
    }

    let mut corr_qc_order_t = 0i64;
    for &sample in input {
        let sample64 = i64::from(sample);
        corr_qc_order_t += sample64 * sample64;
    }
    corr_qc_order_t <<= QC;
    corr_qc[order_t] = corr_qc_order_t;

    let corr_qct_start = order_t - order;
    let mut lsh = clz64(corr_qc_order_t) - 35;
    lsh = lsh.clamp(-12 - QC, 30 - QC);
    let scale = -(QC + lsh);

    if lsh >= 0 {
        let shift = lsh as u32;
        for (i, dst) in corr.iter_mut().take(order + 1).enumerate() {
            *dst = corr_qc[corr_qct_start + order - i].wrapping_shl(shift) as i32;
        }
    } else {
        let shift = (-lsh) as u32;
        for (i, dst) in corr.iter_mut().take(order + 1).enumerate() {
            *dst = (corr_qc[corr_qct_start + order - i] >> shift) as i32;
        }
    }

    scale
}
