use core::arch::aarch64::{
    vcvtq_s32_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32, vminq_f32, vminvq_f32,
    vmulq_n_f32, vqmovn_s32, vrndnq_f32, vst1_s16, vst1q_f32,
};

use crate::celt::float_cast;

/// AArch64 NEON specialization for `opus_limit2_checkwithin1`.
pub(super) fn opus_limit2_checkwithin1(samples: &mut [f32]) -> bool {
    if samples.is_empty() {
        return true;
    }

    const HARDCLIP_MIN: f32 = -2.0;
    const HARDCLIP_MAX: f32 = 2.0;
    const BLOCK_SIZE: usize = 16;
    let blocked_size = samples.len() / BLOCK_SIZE * BLOCK_SIZE;

    let mut exceeding1 = false;
    let mut next_index = 0usize;

    if blocked_size > 0 {
        let mut min_all_0 = unsafe { vdupq_n_f32(0.0) };
        let mut min_all_1 = unsafe { vdupq_n_f32(0.0) };
        let mut max_all_0 = unsafe { vdupq_n_f32(0.0) };
        let mut max_all_1 = unsafe { vdupq_n_f32(0.0) };

        unsafe {
            let samples_ptr = samples.as_ptr();
            for i in (0..blocked_size).step_by(BLOCK_SIZE) {
                let orig_a = vld1q_f32(samples_ptr.add(i));
                let orig_b = vld1q_f32(samples_ptr.add(i + 4));
                let orig_c = vld1q_f32(samples_ptr.add(i + 8));
                let orig_d = vld1q_f32(samples_ptr.add(i + 12));

                max_all_0 = vmaxq_f32(max_all_0, vmaxq_f32(orig_a, orig_b));
                max_all_1 = vmaxq_f32(max_all_1, vmaxq_f32(orig_c, orig_d));
                min_all_0 = vminq_f32(min_all_0, vminq_f32(orig_a, orig_b));
                min_all_1 = vminq_f32(min_all_1, vminq_f32(orig_c, orig_d));
            }

            let max = vmaxvq_f32(vmaxq_f32(max_all_0, max_all_1));
            let min = vminvq_f32(vminq_f32(min_all_0, min_all_1));

            if min < HARDCLIP_MIN || max > HARDCLIP_MAX {
                let hardclip_min = vdupq_n_f32(HARDCLIP_MIN);
                let hardclip_max = vdupq_n_f32(HARDCLIP_MAX);
                let samples_ptr = samples.as_mut_ptr();
                for i in (0..blocked_size).step_by(BLOCK_SIZE) {
                    let orig_a = vld1q_f32(samples_ptr.add(i));
                    let orig_b = vld1q_f32(samples_ptr.add(i + 4));
                    let orig_c = vld1q_f32(samples_ptr.add(i + 8));
                    let orig_d = vld1q_f32(samples_ptr.add(i + 12));
                    let clipped_a = vminq_f32(hardclip_max, vmaxq_f32(orig_a, hardclip_min));
                    let clipped_b = vminq_f32(hardclip_max, vmaxq_f32(orig_b, hardclip_min));
                    let clipped_c = vminq_f32(hardclip_max, vmaxq_f32(orig_c, hardclip_min));
                    let clipped_d = vminq_f32(hardclip_max, vmaxq_f32(orig_d, hardclip_min));
                    vst1q_f32(samples_ptr.add(i), clipped_a);
                    vst1q_f32(samples_ptr.add(i + 4), clipped_b);
                    vst1q_f32(samples_ptr.add(i + 8), clipped_c);
                    vst1q_f32(samples_ptr.add(i + 12), clipped_d);
                }
            }

            exceeding1 = max > 1.0 || min < -1.0;
        }

        next_index = blocked_size;
    }

    for sample in &mut samples[next_index..] {
        let orig_val = *sample;
        *sample = orig_val.clamp(HARDCLIP_MIN, HARDCLIP_MAX);
        exceeding1 |= orig_val > 1.0 || orig_val < -1.0;
    }

    !exceeding1
}

/// AArch64 NEON specialization for `celt_float2int16`.
pub(super) fn celt_float2int16(input: &[f32], output: &mut [i16]) {
    const BLOCK_SIZE: usize = 16;
    let blocked_size = input.len() / BLOCK_SIZE * BLOCK_SIZE;

    unsafe {
        let clamp_min = vdupq_n_f32(-32_768.0);
        let clamp_max = vdupq_n_f32(32_767.0);

        for i in (0..blocked_size).step_by(BLOCK_SIZE) {
            let input_ptr = input.as_ptr().add(i);
            let output_ptr = output.as_mut_ptr().add(i);

            let orig_a = vld1q_f32(input_ptr);
            let orig_b = vld1q_f32(input_ptr.add(4));
            let orig_c = vld1q_f32(input_ptr.add(8));
            let orig_d = vld1q_f32(input_ptr.add(12));

            let scaled_a = vminq_f32(
                clamp_max,
                vmaxq_f32(clamp_min, vmulq_n_f32(orig_a, float_cast::CELT_SIG_SCALE)),
            );
            let scaled_b = vminq_f32(
                clamp_max,
                vmaxq_f32(clamp_min, vmulq_n_f32(orig_b, float_cast::CELT_SIG_SCALE)),
            );
            let scaled_c = vminq_f32(
                clamp_max,
                vmaxq_f32(clamp_min, vmulq_n_f32(orig_c, float_cast::CELT_SIG_SCALE)),
            );
            let scaled_d = vminq_f32(
                clamp_max,
                vmaxq_f32(clamp_min, vmulq_n_f32(orig_d, float_cast::CELT_SIG_SCALE)),
            );

            let as_short_a = vqmovn_s32(vcvtq_s32_f32(vrndnq_f32(scaled_a)));
            let as_short_b = vqmovn_s32(vcvtq_s32_f32(vrndnq_f32(scaled_b)));
            let as_short_c = vqmovn_s32(vcvtq_s32_f32(vrndnq_f32(scaled_c)));
            let as_short_d = vqmovn_s32(vcvtq_s32_f32(vrndnq_f32(scaled_d)));

            vst1_s16(output_ptr, as_short_a);
            vst1_s16(output_ptr.add(4), as_short_b);
            vst1_s16(output_ptr.add(8), as_short_c);
            vst1_s16(output_ptr.add(12), as_short_d);
        }
    }

    super::celt_float2int16_scalar(&input[blocked_size..], &mut output[blocked_size..]);
}
