//! Top-level Opus helpers ported from `opus.c`.
//!
//! Provides the public-facing soft-clipping routine used by the decoder to
//! bound floating-point PCM output to the [-1, 1] range while smoothing the
//! transition across frames.

use crate::celt::opus_limit2_checkwithin1;

/// Soft-clip helper mirroring `opus_pcm_soft_clip_impl` from the reference code.
///
/// The routine modifies the interleaved `pcm` samples in place, updating the
/// per-channel `softclip_mem` state so subsequent frames continue the same
/// non-linearity until the waveform crosses zero. The `arch` hint is accepted
/// for parity with the C signature but ignored because the scalar Rust helper
/// cannot take advantage of architecture-specific implementations.
pub fn opus_pcm_soft_clip_impl(
    pcm: &mut [f32],
    frame_size: usize,
    channels: usize,
    softclip_mem: &mut [f32],
    _arch: i32,
) {
    if frame_size == 0 || channels == 0 {
        return;
    }

    let Some(total_samples) = frame_size.checked_mul(channels) else {
        return;
    };
    if pcm.len() < total_samples || softclip_mem.len() < channels {
        return;
    }

    let samples = &mut pcm[..total_samples];

    // Clamp to [-2, 2] and optionally skip out-of-bound checks when the SIMD
    // implementation can guarantee all values already lay in [-1, 1]. The
    // scalar helper always returns false for non-empty slices.
    let all_within_neg1pos1 = opus_limit2_checkwithin1(samples);

    for channel in 0..channels {
        let mut a = softclip_mem[channel];

        // Continue the previous frame's non-linearity until the waveform
        // crosses zero to avoid a discontinuity at the stitch point.
        let mut i = 0;
        while i < frame_size {
            let idx = i * channels + channel;
            if samples[idx] * a >= 0.0 {
                break;
            }
            let sample = samples[idx];
            samples[idx] = sample + a * sample * sample;
            i += 1;
        }

        let mut curr = 0usize;
        let x0 = samples[channel];
        loop {
            // Detection for early exit can be skipped if hinted by
            // `all_within_neg1pos1`.
            let i = if all_within_neg1pos1 {
                frame_size
            } else {
                let mut scan = curr;
                while scan < frame_size {
                    let value = samples[scan * channels + channel];
                    if !(-1.0..=1.0).contains(&value) {
                        break;
                    }
                    scan += 1;
                }
                scan
            };

            if i == frame_size {
                a = 0.0;
                break;
            }

            let mut peak_pos = i;
            let mut start = i;
            let mut end = i;
            let clipped_sample = samples[i * channels + channel];
            let mut maxval = clipped_sample.abs();

            // Look for the first zero crossing before clipping.
            while start > 0
                && clipped_sample * samples[(start - 1) * channels + channel] >= 0.0
            {
                start -= 1;
            }

            // Look for the first zero crossing after clipping while tracking
            // the highest magnitude in the region.
            while end < frame_size
                && clipped_sample * samples[end * channels + channel] >= 0.0
            {
                let abs_val = samples[end * channels + channel].abs();
                if abs_val > maxval {
                    maxval = abs_val;
                    peak_pos = end;
                }
                end += 1;
            }

            let special = start == 0 && clipped_sample * samples[channel] >= 0.0;

            // Compute the soft-clipping coefficient such that maxval + a*maxval^2 = 1.
            a = (maxval - 1.0) / (maxval * maxval);
            a += a * 2.4e-7;
            if clipped_sample > 0.0 {
                a = -a;
            }

            // Apply soft clipping for the current region.
            for frame_idx in start..end {
                let idx = frame_idx * channels + channel;
                let sample = samples[idx];
                samples[idx] = sample + a * sample * sample;
            }

            if special && peak_pos >= 2 {
                // Add a linear ramp from the first sample to the signal peak to
                // avoid a discontinuity at the start of the frame.
                let mut offset = x0 - samples[channel];
                let delta = offset / peak_pos as f32;
                for frame_idx in curr..peak_pos {
                    offset -= delta;
                    let idx = frame_idx * channels + channel;
                    samples[idx] += offset;
                    samples[idx] = samples[idx].clamp(-1.0, 1.0);
                }
            }

            curr = end;
            if curr == frame_size {
                break;
            }
        }

        softclip_mem[channel] = a;
    }
}

/// Public wrapper that ignores the architecture hint.
#[inline]
pub fn opus_pcm_soft_clip(pcm: &mut [f32], frame_size: usize, channels: usize, softclip_mem: &mut [f32]) {
    opus_pcm_soft_clip_impl(pcm, frame_size, channels, softclip_mem, 0);
}

#[cfg(test)]
mod tests {
    use super::{opus_pcm_soft_clip, opus_pcm_soft_clip_impl};

    #[test]
    fn in_range_samples_reset_state_without_modification() {
        let mut pcm = [0.1_f32, -0.6, 0.9, 0.3];
        let mut state = [0.5_f32, -0.25];

        opus_pcm_soft_clip_impl(&mut pcm, 2, 2, &mut state, 0);

        assert_eq!(pcm, [0.1, -0.6, 0.9, 0.3]);
        assert_eq!(state, [0.0, 0.0]);
    }

    #[test]
    fn clips_peak_and_updates_memory() {
        let mut pcm = [2.0_f32];
        let mut state = [0.0_f32];

        opus_pcm_soft_clip(&mut pcm, 1, 1, &mut state);

        assert!((pcm[0] - 0.999_999_76).abs() < 1e-7);
        assert!((state[0] + 0.250_000_06).abs() < 1e-7);
    }

    #[test]
    fn continues_previous_nonlinearity_until_zero_crossing() {
        let mut pcm = [0.5_f32, 0.25, -0.25];
        let mut state = [-0.25_f32];

        opus_pcm_soft_clip_impl(&mut pcm, 3, 1, &mut state, 0);

        assert!((pcm[0] - 0.437_5).abs() < 1e-6);
        assert!((pcm[1] - 0.234_375).abs() < 1e-6);
        assert!((pcm[2] + 0.25).abs() < 1e-6);
        assert_eq!(state, [0.0]);
    }

    #[test]
    fn applies_ramp_when_clipping_before_first_zero_crossing() {
        let mut pcm = [0.5_f32, 1.5, 1.7, -0.4];
        let mut state = [0.0_f32];

        opus_pcm_soft_clip(&mut pcm, 4, 1, &mut state);

        assert!((pcm[0] - 0.469_723_17).abs() < 1e-6);
        assert!((pcm[1] - 0.955_016_2).abs() < 1e-6);
        assert!((pcm[2] - 1.0).abs() < 2e-6);
        assert!((pcm[3] + 0.4).abs() < 1e-6);
        assert_eq!(state, [0.0]);
    }
}
