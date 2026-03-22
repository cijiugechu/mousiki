use core::arch::arm::{
    vaddq_f32, vandq_u32, vcvtq_s32_f32, vdupq_n_f32, vdupq_n_u32, vget_high_f32, vget_lane_f32,
    vget_low_f32, vld1q_f32, vmax_f32, vmaxq_f32, vmin_f32, vminq_f32, vmulq_n_f32, vorrq_u32,
    vqmovn_s32, vreinterpretq_f32_u32, vreinterpretq_u32_f32, vst1_s16, vst1q_f32,
};

use crate::celt::float_cast;

#[inline]
unsafe fn vroundf(x: core::arch::arm::float32x4_t) -> core::arch::arm::int32x4_t {
    // Mirror opus-c arm/mathops_arm.h non-aarch64 fallback:
    // vcvtq_s32_f32(x + copysign(0.5, x))
    let sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x8000_0000));
    let bias = vdupq_n_u32(0x3f00_0000);
    vcvtq_s32_f32(vaddq_f32(x, vreinterpretq_f32_u32(vorrq_u32(bias, sign))))
}

#[inline]
unsafe fn vminvf(a: core::arch::arm::float32x4_t) -> f32 {
    let xy = vmin_f32(vget_low_f32(a), vget_high_f32(a));
    let x = vget_lane_f32(xy, 0);
    let y = vget_lane_f32(xy, 1);
    if x < y { x } else { y }
}

#[inline]
unsafe fn vmaxvf(a: core::arch::arm::float32x4_t) -> f32 {
    let xy = vmax_f32(vget_low_f32(a), vget_high_f32(a));
    let x = vget_lane_f32(xy, 0);
    let y = vget_lane_f32(xy, 1);
    if x > y { x } else { y }
}

/// ARM NEON specialization for `opus_limit2_checkwithin1`.
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

            let max = vmaxvf(vmaxq_f32(max_all_0, max_all_1));
            let min = vminvf(vminq_f32(min_all_0, min_all_1));

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

/// ARM NEON specialization for `celt_float2int16`.
pub(super) fn celt_float2int16(input: &[f32], output: &mut [i16]) {
    const BLOCK_SIZE: usize = 16;
    let blocked_size = input.len() / BLOCK_SIZE * BLOCK_SIZE;

    unsafe {
        for i in (0..blocked_size).step_by(BLOCK_SIZE) {
            let input_ptr = input.as_ptr().add(i);
            let output_ptr = output.as_mut_ptr().add(i);

            let orig_a = vld1q_f32(input_ptr);
            let orig_b = vld1q_f32(input_ptr.add(4));
            let orig_c = vld1q_f32(input_ptr.add(8));
            let orig_d = vld1q_f32(input_ptr.add(12));

            let as_short_a = vqmovn_s32(vroundf(vmulq_n_f32(orig_a, float_cast::CELT_SIG_SCALE)));
            let as_short_b = vqmovn_s32(vroundf(vmulq_n_f32(orig_b, float_cast::CELT_SIG_SCALE)));
            let as_short_c = vqmovn_s32(vroundf(vmulq_n_f32(orig_c, float_cast::CELT_SIG_SCALE)));
            let as_short_d = vqmovn_s32(vroundf(vmulq_n_f32(orig_d, float_cast::CELT_SIG_SCALE)));

            vst1_s16(output_ptr, as_short_a);
            vst1_s16(output_ptr.add(4), as_short_b);
            vst1_s16(output_ptr.add(8), as_short_c);
            vst1_s16(output_ptr.add(12), as_short_d);
        }
    }

    super::celt_float2int16_scalar(&input[blocked_size..], &mut output[blocked_size..]);
}
