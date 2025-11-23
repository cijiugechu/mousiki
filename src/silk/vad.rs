//! Voice activity detector ported from `silk/VAD.c`.
//!
//! The routines here mirror the fixed-point SILK VAD that band-splits the
//! incoming PCM stream, tracks per-band noise levels, and derives a smoothed
//! speech-activity probability alongside band quality metrics.

use crate::silk::ana_filt_bank_1::ana_filt_bank_1;
use crate::silk::encoder::state::{
    EncoderChannelState, EncoderStateCommon, MAX_FRAME_LENGTH_MS, MAX_FS_KHZ, VAD_N_BANDS, VadState,
};
use crate::silk::lin2log::lin2log;
use crate::silk::sigm_q15::sigm_q15;

const VAD_INTERNAL_SUBFRAMES_LOG2: usize = 2;
const VAD_INTERNAL_SUBFRAMES: usize = 1 << VAD_INTERNAL_SUBFRAMES_LOG2;
const VAD_NOISE_LEVEL_SMOOTH_COEF_Q16: i32 = 1024;
const VAD_SNR_FACTOR_Q16: i32 = 45_000;
const VAD_NEGATIVE_OFFSET_Q5: i32 = 128;
const VAD_SNR_SMOOTH_COEF_Q18: i32 = 4096;
const SILK_UINT8_MAX: i32 = u8::MAX as i32;
const TILT_WEIGHTS: [i32; VAD_N_BANDS] = [30_000, 6_000, -12_000, -12_000];
const MAX_FRAME_LENGTH: usize = MAX_FRAME_LENGTH_MS * MAX_FS_KHZ;
const MAX_VAD_BUFFER_LENGTH: usize = MAX_FRAME_LENGTH * 5 / 4;
const MAX_HALF_FRAME_LENGTH: usize = MAX_FRAME_LENGTH / 2;

/// Updates the speech-activity probability (Q8) and per-band quality metrics.
pub fn compute_speech_activity_q8(channel: &mut EncoderChannelState, input: &[i16]) -> u8 {
    let frame_length = channel.common().frame_length;
    assert!(
        frame_length <= MAX_FRAME_LENGTH,
        "unexpected frame length {frame_length}"
    );
    assert_eq!(
        frame_length,
        (frame_length >> 3) << 3,
        "frame length must be divisible by 8"
    );
    assert_eq!(
        input.len(),
        frame_length,
        "input length does not match the configured frame length"
    );

    let fs_khz = channel.common().fs_khz;
    let (common, vad_state) = channel.parts_mut();
    analyse_frame(common, vad_state, fs_khz, frame_length, input)
}

/// Updates speech-activity probability for a standalone encoder state.
pub fn compute_speech_activity_q8_common(common: &mut EncoderStateCommon, input: &[i16]) -> u8 {
    let frame_length = common.frame_length;
    assert!(
        frame_length <= MAX_FRAME_LENGTH,
        "unexpected frame length {frame_length}"
    );
    assert_eq!(
        frame_length,
        (frame_length >> 3) << 3,
        "frame length must be divisible by 8"
    );
    assert_eq!(
        input.len(),
        frame_length,
        "input length does not match the configured frame length"
    );

    let fs_khz = common.fs_khz;
    let common_ptr: *mut EncoderStateCommon = common;
    // Split the mutable borrow of `common` to hand matching lifetimes to `analyse_frame`.
    let vad_state = unsafe { &mut (*common_ptr).vad_state };
    analyse_frame(common, vad_state, fs_khz, frame_length, input)
}

fn analyse_frame(
    common: &mut EncoderStateCommon,
    vad_state: &mut VadState,
    fs_khz: i32,
    frame_length: usize,
    input: &[i16],
) -> u8 {
    let decimated1 = frame_length >> 1;
    let decimated2 = frame_length >> 2;
    let decimated = frame_length >> 3;

    let mut x = [0i16; MAX_VAD_BUFFER_LENGTH];
    let mut scratch = [0i16; MAX_HALF_FRAME_LENGTH];
    let mut offsets = [0usize; VAD_N_BANDS];
    offsets[0] = 0;
    offsets[1] = decimated + decimated2;
    offsets[2] = offsets[1] + decimated;
    offsets[3] = offsets[2] + decimated2;

    // Split full-band audio into 0-4 kHz and 4-8 kHz.
    {
        let (prefix, tail) = x.split_at_mut(offsets[3]);
        let (low, _) = prefix.split_at_mut(decimated1);
        let high = &mut tail[..decimated1];
        ana_filt_bank_1(&mut vad_state.ana_state, low, high, input);
    }
    // Split the 0-4 kHz band into 0-2 kHz and 2-4 kHz.
    scratch[..decimated1].copy_from_slice(&x[..decimated1]);
    {
        let (prefix, tail) = x.split_at_mut(offsets[2]);
        let (low, _) = prefix.split_at_mut(decimated2);
        let high = &mut tail[..decimated2];
        ana_filt_bank_1(&mut vad_state.ana_state1, low, high, &scratch[..decimated1]);
    }
    // Split the 0-2 kHz band into 0-1 kHz and 1-2 kHz.
    scratch[..decimated2].copy_from_slice(&x[..decimated2]);
    if decimated > 0 {
        let (prefix, tail) = x.split_at_mut(offsets[1]);
        let (low, _) = prefix.split_at_mut(decimated);
        let high = &mut tail[..decimated];
        ana_filt_bank_1(&mut vad_state.ana_state2, low, high, &scratch[..decimated2]);
    }

    highpass_lowest_band(&mut x[..decimated], vad_state);

    let mut xnrg = [0i32; VAD_N_BANDS];
    accumulate_band_energies(&mut xnrg, &x, vad_state, frame_length, &offsets);
    update_noise_levels(&xnrg, vad_state);

    let mut energy_ratios_q8 = [0i32; VAD_N_BANDS];
    let mut sum_squared = 0;
    let mut input_tilt = 0;
    for (b, &band_energy) in xnrg.iter().enumerate() {
        let mut speech_nrg = band_energy - vad_state.nl[b];
        if speech_nrg > 0 {
            let ratio_q8 = if (band_energy as u32 & 0xFF80_0000) == 0 {
                div32(safe_lshift(band_energy, 8), vad_state.nl[b] + 1)
            } else {
                div32(band_energy, (vad_state.nl[b] >> 8) + 1)
            };
            energy_ratios_q8[b] = ratio_q8;

            let mut snr_q7 = lin2log(ratio_q8) - (8 * 128);
            sum_squared = smlabb(sum_squared, snr_q7, snr_q7);

            if speech_nrg < (1 << 20) {
                speech_nrg = sqrt_approx(speech_nrg);
                let scaled = safe_lshift(speech_nrg, 6);
                snr_q7 = smulwb(scaled, snr_q7);
            }
            input_tilt = smlawb(input_tilt, TILT_WEIGHTS[b], snr_q7);
        } else {
            energy_ratios_q8[b] = 256;
        }
    }

    sum_squared = div32_16(sum_squared, VAD_N_BANDS as i32);
    let snr_db_q7 = 3 * sqrt_approx(sum_squared);
    let mut sa_q15 = sigm_q15(smulwb(VAD_SNR_FACTOR_Q16, snr_db_q7) - VAD_NEGATIVE_OFFSET_Q5);
    common.input_tilt_q15 = safe_lshift(sigm_q15(input_tilt) - 16_384, 1);

    let mut speech_nrg = 0i64;
    for (band_idx, (&band_energy, &noise_level)) in xnrg.iter().zip(vad_state.nl.iter()).enumerate()
    {
        let excess = (band_energy - noise_level) >> 4;
        speech_nrg += i64::from(band_idx as i32 + 1) * i64::from(excess);
    }
    if frame_length == 20 * fs_khz as usize {
        speech_nrg >>= 1;
    }
    if speech_nrg <= 0 {
        sa_q15 >>= 1;
    } else if speech_nrg < 16_384 {
        let mut scaled = safe_lshift(speech_nrg as i32, 16);
        scaled = sqrt_approx(scaled);
        sa_q15 = smulwb(32_768 + scaled, sa_q15);
    }
    let speech_activity_q8 = (sa_q15 >> 7).clamp(0, SILK_UINT8_MAX);
    common.speech_activity_q8 = speech_activity_q8;

    let mut smooth_coef_q16 = smulwb(VAD_SNR_SMOOTH_COEF_Q18, smulwb(sa_q15, sa_q15));
    if frame_length == 10 * fs_khz as usize {
        smooth_coef_q16 >>= 1;
    }

    for ((ratio_smth, quality_band), &ratio_q8) in vad_state
        .nrg_ratio_smth_q8
        .iter_mut()
        .zip(common.input_quality_bands_q15.iter_mut())
        .zip(energy_ratios_q8.iter())
    {
        *ratio_smth = smlawb(*ratio_smth, ratio_q8 - *ratio_smth, smooth_coef_q16);
        let snr_q7 = 3 * (lin2log(*ratio_smth) - 8 * 128);
        *quality_band = sigm_q15((snr_q7 - 16 * 128) >> 4);
    }

    speech_activity_q8 as u8
}

fn highpass_lowest_band(band: &mut [i16], vad_state: &mut VadState) {
    if band.is_empty() {
        return;
    }

    let last_idx = band.len() - 1;
    band[last_idx] = (i32::from(band[last_idx]) >> 1) as i16;
    let hp_state_tmp = band[last_idx];
    for i in (1..band.len()).rev() {
        band[i - 1] = (i32::from(band[i - 1]) >> 1) as i16;
        let diff = i32::from(band[i]) - i32::from(band[i - 1]);
        band[i] = diff as i16;
    }
    band[0] = (i32::from(band[0]) - i32::from(vad_state.hp_state)) as i16;
    vad_state.hp_state = hp_state_tmp;
}

fn accumulate_band_energies(
    xnrg: &mut [i32; VAD_N_BANDS],
    x: &[i16; MAX_VAD_BUFFER_LENGTH],
    vad_state: &mut VadState,
    frame_length: usize,
    offsets: &[usize; VAD_N_BANDS],
) {
    for (b, ((offset, xnrg_slot), state_slot)) in offsets
        .iter()
        .zip(xnrg.iter_mut())
        .zip(vad_state.xnrg_subfr.iter_mut())
        .enumerate()
    {
        let shift = (VAD_N_BANDS - b).min(VAD_N_BANDS - 1);
        let decimated_length = frame_length >> shift;
        let mut dec_subframe_len = decimated_length >> VAD_INTERNAL_SUBFRAMES_LOG2;
        if dec_subframe_len == 0 {
            dec_subframe_len = 1;
        }

        let band = &x[*offset..*offset + decimated_length];
        let mut total = *state_slot;
        let mut last_sum = 0;
        let mut offset = 0usize;
        for s in 0..VAD_INTERNAL_SUBFRAMES {
            if offset >= band.len() {
                break;
            }
            let chunk_len = dec_subframe_len.min(band.len() - offset).max(1);
            let mut acc = 0;
            for sample in &band[offset..offset + chunk_len] {
                let reduced = (i32::from(*sample)) >> 3;
                acc = smlabb(acc, reduced, reduced);
            }
            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                total = add_pos_sat32(total, acc);
            } else {
                total = add_pos_sat32(total, acc >> 1);
            }
            last_sum = acc;
            offset += chunk_len;
        }
        *state_slot = last_sum;
        *xnrg_slot = total;
    }
}

fn update_noise_levels(xnrg: &[i32; VAD_N_BANDS], vad_state: &mut VadState) {
    let mut min_coef = 0;
    if vad_state.counter < 1000 {
        min_coef = div32_16(i32::from(i16::MAX), (vad_state.counter >> 4) + 1);
        vad_state.counter += 1;
    }

    for (((nl, inv_nl), bias), &band_energy) in vad_state
        .nl
        .iter_mut()
        .zip(vad_state.inv_nl.iter_mut())
        .zip(vad_state.noise_level_bias.iter())
        .zip(xnrg.iter())
    {
        let mut nrg = add_pos_sat32(band_energy, *bias);
        if nrg <= 0 {
            nrg = 1;
        }
        let inv_nrg = div32(i32::MAX, nrg);
        let mut coef = if nrg > (*nl << 3) {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3
        } else if nrg < *nl {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16
        } else {
            smulwb(smulww(inv_nrg, *nl), VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1)
        };
        coef = coef.max(min_coef);

        *inv_nl = smlawb(*inv_nl, inv_nrg - *inv_nl, coef);
        let mut nl_new = if *inv_nl > 0 {
            div32(i32::MAX, *inv_nl)
        } else {
            0
        };
        nl_new = nl_new.min(0x00FF_FFFF);
        *nl = nl_new;
    }
}

#[inline]
fn add_pos_sat32(a: i32, b: i32) -> i32 {
    let sum = i64::from(a) + i64::from(b);
    if sum < 0 {
        0
    } else if sum > i64::from(i32::MAX) {
        i32::MAX
    } else {
        sum as i32
    }
}

#[inline]
fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

#[inline]
fn smlabb(acc: i32, a: i32, b: i32) -> i32 {
    acc.wrapping_add(smulbb(a, b))
}

#[inline]
fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

#[inline]
fn smlawb(acc: i32, a: i32, b: i32) -> i32 {
    acc.wrapping_add(smulwb(a, b))
}

#[inline]
fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

#[inline]
fn div32_16(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}

#[inline]
fn div32(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}

#[inline]
fn safe_lshift(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value >> (-shift)
    } else if shift >= 31 {
        0
    } else {
        value.wrapping_shl(shift as u32)
    }
}

const SQRT_COEF_Q7: i32 = 213;

fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }

    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32_768 } else { 46_214 };
    y >>= lz >> 1;
    smlawb(y, y, smulbb(SQRT_COEF_Q7, frac_q7))
}

fn clz_frac(x: i32) -> (i32, i32) {
    let ux = x as u32;
    let lz = ux.leading_zeros() as i32;
    let rotate = ((24 - lz) & 31) as u32;
    let frac = (ux.rotate_right(rotate) & 0x7f) as i32;
    (lz, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn silence_keeps_activity_low() {
        let mut channel = EncoderChannelState::default();
        let frame = vec![0i16; channel.common().frame_length];
        let activity = compute_speech_activity_q8(&mut channel, &frame);
        assert_eq!(activity, 2);
        assert_eq!(channel.common().speech_activity_q8, 2);
    }

    #[test]
    fn strong_signal_triggers_activity() {
        let mut channel = EncoderChannelState::default();
        let frame = vec![2000i16; channel.common().frame_length];
        let activity = compute_speech_activity_q8(&mut channel, &frame);
        assert!(activity > 0);
        assert!(
            channel
                .common()
                .input_quality_bands_q15
                .iter()
                .any(|band| *band > 0)
        );
    }

    #[test]
    fn supports_ten_ms_frames() {
        let mut channel = EncoderChannelState::default();
        {
            let common = channel.common_mut();
            common.frame_length =
                (crate::silk::encoder::state::SUB_FRAME_LENGTH_MS * 2) * MAX_FS_KHZ;
        }
        let frame = vec![500i16; channel.common().frame_length];
        let activity = compute_speech_activity_q8(&mut channel, &frame);
        assert!(activity > 0);
    }
}
