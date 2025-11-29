#![allow(dead_code)]

//! Decoder scaffolding ported from `celt/celt_decoder.c`.
//!
//! The reference implementation combines the primary decoder state with a
//! trailing buffer that stores the pitch predictor history, LPC coefficients,
//! and band energy memories.  This module mirrors the allocation strategy so
//! that higher level decode routines can be ported gradually while continuing
//! to rely on the Rust ownership model for safety.
//!
//! Only the allocation helpers are provided for now.  The full decoding loop,
//! packet loss concealment, and post-filter plumbing still live in the C
//! sources and will be translated in follow-up patches.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "fixed_point"))]
use crate::celt::BandCodingState;
use crate::celt::bands::denormalise_bands;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::bands::{anti_collapse, celt_lcg_rand, quant_all_bands};
use crate::celt::celt::{
    COMBFILTER_MINPERIOD, TF_SELECT_TABLE, comb_filter, init_caps, resampling_factor,
};
use crate::celt::cpu_support::{OPUS_ARCHMASK, opus_select_arch};
use crate::celt::entcode::{self, BITRES};
use crate::celt::entdec::EcDec;
use crate::celt::float_cast::CELT_SIG_SCALE;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::float_cast::{float2int, float2int16};
#[cfg(not(feature = "fixed_point"))]
use crate::celt::lpc::{celt_autocorr, celt_iir, celt_lpc};
#[cfg(not(feature = "fixed_point"))]
use crate::celt::math::{celt_sqrt, frac_div32};
use crate::celt::mdct::clt_mdct_backward;
use crate::celt::modes::opus_custom_mode_find_static;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::quant_bands::unquant_energy_finalise;
use crate::celt::quant_bands::{unquant_coarse_energy, unquant_fine_energy};
use crate::celt::rate::clt_compute_allocation;
use crate::celt::types::{
    CeltGlog, CeltNorm, CeltSig, OpusCustomDecoder, OpusCustomMode, OpusInt32, OpusRes, OpusUint32,
    OpusVal16, OpusVal32,
};
use crate::celt::vq::SPREAD_NORMAL;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::vq::renormalise_vector;
use core::cell::UnsafeCell;
#[cfg(not(feature = "fixed_point"))]
use core::cmp::Ordering;
use core::cmp::{max, min};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};
use libm::sqrtf;

/// Linear prediction order used by the decoder side filters.
///
/// Mirrors the `LPC_ORDER` constant from the reference implementation.  The
/// value is surfaced here so future ports that rely on the LPC history length
/// can share the same constant.
const LPC_ORDER: usize = 24;

/// Size of the rolling decode buffer maintained per channel.
///
/// Matches the `DECODE_BUFFER_SIZE` constant from the C implementation.  The
/// reference decoder keeps a two kilobyte circular history in front of the
/// overlap region so packet loss concealment and the post-filter can operate on
/// previously synthesised samples.  Mirroring the same storage requirements in
/// Rust keeps the allocation layout compatible with the ported routines that
/// will eventually consume these buffers.
pub(crate) const DECODE_BUFFER_SIZE: usize = 2048;

/// Maximum pitch period considered by the PLC pitch search.
const MAX_PERIOD: i32 = 1024;

/// Upper bound on the pitch lag probed by the PLC search.
const PLC_PITCH_LAG_MAX: i32 = 720;

/// Lower bound on the pitch lag probed by the PLC search.
const PLC_PITCH_LAG_MIN: i32 = 100;

/// Saturation limit applied to the IMDCT output during synthesis.
const SIG_SAT: CeltSig = 536_870_911.0;

fn apply_inverse_mdct(
    mode: &OpusCustomMode<'_>,
    freq: &[CeltSig],
    output: &mut [CeltSig],
    bands: usize,
    nb: usize,
    shift: usize,
) {
    if bands == 0 {
        return;
    }

    let stride = bands;
    assert!(freq.len() >= nb.saturating_mul(stride));
    assert!(output.len() >= nb.saturating_mul(stride));

    let mut temp = vec![0.0f32; nb];
    for band in 0..bands {
        for (idx, sample) in temp.iter_mut().enumerate() {
            let src_index = band + idx * stride;
            *sample = freq.get(src_index).copied().unwrap_or_default();
        }

        let start = band * nb;
        clt_mdct_backward(
            &mode.mdct,
            &temp,
            &mut output[start..],
            mode.window,
            mode.overlap,
            shift,
            1,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn celt_synthesis(
    mode: &OpusCustomMode<'_>,
    x: &[CeltNorm],
    out_syn: &mut [&mut [CeltSig]],
    old_band_e: &[CeltGlog],
    start: usize,
    eff_end: usize,
    coded_channels: usize,
    output_channels: usize,
    is_transient: bool,
    lm: usize,
    downsample: usize,
    silence: bool,
    arch: i32,
) {
    let _ = arch;

    assert!(output_channels <= out_syn.len());
    assert!(coded_channels <= 2);
    assert!(output_channels <= 2);
    assert!(lm <= mode.max_lm);
    assert!(eff_end <= mode.num_ebands);
    assert!(downsample > 0);

    let nb_ebands = mode.num_ebands;
    let n = mode.short_mdct_size << lm;
    let m = 1 << lm;

    assert!(x.len() >= coded_channels * n);
    assert!(old_band_e.len() >= coded_channels * nb_ebands);
    for channel in out_syn.iter_mut().take(output_channels) {
        assert!(channel.len() >= n);
    }

    let (bands, nb, shift) = if is_transient {
        (m, mode.short_mdct_size, mode.max_lm)
    } else {
        (1, mode.short_mdct_size << lm, mode.max_lm - lm)
    };

    let mut freq = vec![0.0f32; n];

    match (output_channels, coded_channels) {
        (2, 1) => {
            let (left, right) = out_syn.split_at_mut(1);
            let left_out = &mut *left[0];
            let right_out = &mut *right[0];

            denormalise_bands(
                mode,
                &x[..n],
                &mut freq,
                &old_band_e[..nb_ebands],
                start,
                eff_end,
                m,
                downsample,
                silence,
            );

            let freq_copy = freq.clone();
            apply_inverse_mdct(mode, &freq_copy, left_out, bands, nb, shift);
            apply_inverse_mdct(mode, &freq, right_out, bands, nb, shift);
        }
        (1, 2) => {
            let out = &mut *out_syn[0];

            denormalise_bands(
                mode,
                &x[..n],
                &mut freq,
                &old_band_e[..nb_ebands],
                start,
                eff_end,
                m,
                downsample,
                silence,
            );

            let mut freq_other = vec![0.0f32; n];
            denormalise_bands(
                mode,
                &x[n..n * 2],
                &mut freq_other,
                &old_band_e[nb_ebands..nb_ebands * 2],
                start,
                eff_end,
                m,
                downsample,
                silence,
            );

            for (lhs, rhs) in freq.iter_mut().zip(freq_other.iter()) {
                *lhs = 0.5 * (*lhs + *rhs);
            }

            apply_inverse_mdct(mode, &freq, out, bands, nb, shift);
        }
        _ => {
            for channel in 0..output_channels {
                let spectrum = &x[channel * n..(channel + 1) * n];
                let energy = &old_band_e[channel * nb_ebands..(channel + 1) * nb_ebands];
                denormalise_bands(
                    mode, spectrum, &mut freq, energy, start, eff_end, m, downsample, silence,
                );

                let output = &mut *out_syn[channel];
                apply_inverse_mdct(mode, &freq, output, bands, nb, shift);
            }
        }
    }

    for channel in out_syn.iter_mut().take(output_channels) {
        for sample in (*channel).iter_mut().take(n) {
            *sample = sample.clamp(-SIG_SAT, SIG_SAT);
        }
    }
}

/// Runs the pitch search used when concealing packet loss.
///
/// The helper inspects the averaged decoder history, computes normalised
/// correlations for lags within the PLC pitch range, and selects the period
/// with the strongest match. Sub-harmonics and harmonics are re-evaluated so
/// that tonal inputs favour the expected fundamental frequency.
fn celt_plc_pitch_search(decode_mem: &[&[CeltSig]], channels: usize, _arch: i32) -> i32 {
    if channels == 0 {
        return PLC_PITCH_LAG_MAX;
    }

    let mut channel_views = Vec::with_capacity(channels);
    for (idx, channel) in decode_mem.iter().take(channels).enumerate() {
        debug_assert!(
            channel.len() >= DECODE_BUFFER_SIZE,
            "channel {idx} must expose at least DECODE_BUFFER_SIZE samples",
        );
        let end = DECODE_BUFFER_SIZE.min(channel.len());
        channel_views.push(&channel[..end]);
    }

    if channel_views.is_empty() {
        return PLC_PITCH_LAG_MAX;
    }

    let mut mono = vec![0.0f32; DECODE_BUFFER_SIZE];
    for channel in channel_views {
        for (dst, &sample) in mono.iter_mut().zip(channel.iter()) {
            *dst += sample;
        }
    }

    let scale = 1.0 / channels as f32;
    for sample in &mut mono {
        *sample *= scale;
    }

    let target_start = PLC_PITCH_LAG_MAX as usize;
    if mono.len() <= target_start {
        return PLC_PITCH_LAG_MAX;
    }

    let target = &mono[target_start..];
    let target_len = target.len();
    if target_len == 0 {
        return PLC_PITCH_LAG_MAX;
    }

    let target_energy: f32 = target.iter().map(|v| v * v).sum();
    if target_energy == 0.0 {
        return PLC_PITCH_LAG_MAX;
    }

    let mut scores = Vec::with_capacity((PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN + 1) as usize);

    for pitch in PLC_PITCH_LAG_MIN..=PLC_PITCH_LAG_MAX {
        let lag = pitch as usize;
        if lag > target_start {
            break;
        }
        let reference = &mono[target_start - lag..target_start - lag + target_len];
        let reference_energy: f32 = reference.iter().map(|v| v * v).sum();
        if reference_energy == 0.0 {
            continue;
        }
        let dot: f32 = target
            .iter()
            .zip(reference.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let corr = dot / (sqrtf(target_energy) * sqrtf(reference_energy));
        if corr.is_finite() {
            scores.push((pitch, corr));
        }
    }

    if scores.is_empty() {
        return PLC_PITCH_LAG_MAX;
    }

    let (mut best_pitch, mut best_corr) = scores
        .iter()
        .copied()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
        .unwrap();

    let consider = |candidate: i32| -> Option<f32> {
        if !(PLC_PITCH_LAG_MIN..=PLC_PITCH_LAG_MAX).contains(&candidate) {
            return None;
        }
        scores
            .iter()
            .find(|(pitch, _)| *pitch == candidate)
            .map(|(_, corr)| *corr)
    };

    if let Some(half_corr) = consider(best_pitch / 2)
        && half_corr >= best_corr * 0.95
    {
        best_pitch /= 2;
        best_corr = half_corr;
    }

    if let Some(double_corr) = consider(best_pitch * 2)
        && double_corr > best_corr * 1.02
    {
        best_pitch *= 2;
    }

    best_pitch
}

fn prefilter_and_fold(decoder: &mut OpusCustomDecoder<'_>, n: usize) {
    let channels = decoder.channels;
    if channels == 0 {
        return;
    }

    let overlap = decoder.overlap;
    if overlap == 0 {
        return;
    }

    debug_assert!(n <= DECODE_BUFFER_SIZE, "prefilter span exceeds history");

    let stride = DECODE_BUFFER_SIZE + overlap;
    debug_assert_eq!(decoder.decode_mem.len(), stride * channels);

    let start = DECODE_BUFFER_SIZE
        .checked_sub(n)
        .expect("prefilter span exceeds decode buffer");
    debug_assert!(
        start + overlap <= stride,
        "decode buffer lacks overlap tail"
    );

    debug_assert!(decoder.postfilter_tapset_old >= 0);
    debug_assert!(decoder.postfilter_tapset >= 0);
    let tapset0 = decoder.postfilter_tapset_old.max(0) as usize;
    let tapset1 = decoder.postfilter_tapset.max(0) as usize;

    let mut etmp = vec![OpusVal32::default(); overlap];
    let window = decoder.mode.window;
    debug_assert!(window.len() >= overlap);

    for channel in 0..channels {
        let offset = channel * stride;
        let channel_mem = &mut decoder.decode_mem[offset..offset + stride];

        comb_filter(
            &mut etmp,
            channel_mem,
            start,
            overlap,
            decoder.postfilter_period_old,
            decoder.postfilter_period,
            -decoder.postfilter_gain_old,
            -decoder.postfilter_gain,
            tapset0,
            tapset1,
            &[],
            0,
            decoder.arch,
        );

        for i in 0..(overlap / 2) {
            let forward = window[i] * etmp[overlap - 1 - i];
            let reverse = window[overlap - 1 - i] * etmp[i];
            channel_mem[start + i] = forward + reverse;
        }
    }
}

#[cfg(not(feature = "fixed_point"))]
pub(crate) fn celt_decode_lost(decoder: &mut OpusCustomDecoder<'_>, n: usize, lm: usize) {
    let channels = decoder.channels;
    if channels == 0 || n == 0 {
        let increment = ((1usize) << lm) as i32;
        decoder.loss_duration = min(10_000, decoder.loss_duration + increment);
        return;
    }

    assert!(
        channels <= MAX_CHANNELS,
        "decoder only supports mono or stereo"
    );
    assert!(n <= DECODE_BUFFER_SIZE, "frame size exceeds decode history");

    let mode = decoder.mode;
    let overlap = mode.overlap;
    let stride = DECODE_BUFFER_SIZE + overlap;
    assert!(decoder.decode_mem.len() >= stride * channels);

    let nb_ebands = mode.num_ebands;
    assert!(mode.e_bands.len() > nb_ebands);

    let raw_start = max(decoder.start_band, 0);
    let start_band = min(raw_start as usize, nb_ebands);
    let raw_end = max(decoder.end_band, raw_start);
    let end_band = min(raw_end as usize, nb_ebands);
    let eff_end = max(start_band, min(end_band, mode.effective_ebands));

    let decode_mem_ptr = decoder.decode_mem.as_mut_ptr();
    let loss_duration = decoder.loss_duration;
    let noise_based = loss_duration >= 40 || start_band != 0 || decoder.skip_plc;

    if noise_based {
        let move_len = DECODE_BUFFER_SIZE
            .checked_sub(n)
            .expect("frame size larger than decode buffer")
            + overlap;
        for ch in 0..channels {
            let channel_slice =
                unsafe { core::slice::from_raw_parts_mut(decode_mem_ptr.add(ch * stride), stride) };
            channel_slice.copy_within(n..n + move_len, 0);
        }

        if decoder.prefilter_and_fold {
            prefilter_and_fold(decoder, n);
        }

        let decay = if loss_duration == 0 { 1.5_f32 } else { 0.5_f32 };
        for ch in 0..channels {
            let base = ch * nb_ebands;
            for band in start_band..end_band {
                let idx = base + band;
                let background = decoder.background_log_e[idx];
                let current = decoder.old_ebands[idx];
                decoder.old_ebands[idx] = background.max(current - decay);
            }
        }

        let mut seed = decoder.rng;
        let mut spectrum = vec![0.0f32; channels * n];
        for ch in 0..channels {
            for band in start_band..eff_end {
                let band_start = (mode.e_bands[band] as usize) << lm;
                let width = mode.e_bands[band + 1] - mode.e_bands[band];
                debug_assert!(width >= 0);
                let band_width = ((width as usize) << lm).min(n.saturating_sub(band_start));
                if band_width == 0 {
                    continue;
                }
                let offset = ch * n + band_start;
                let slice = &mut spectrum[offset..offset + band_width];
                for sample in slice.iter_mut() {
                    seed = celt_lcg_rand(seed);
                    *sample = ((seed as i32) >> 20) as f32;
                }
                renormalise_vector(slice, band_width, 1.0, decoder.arch);
            }
        }
        decoder.rng = seed;

        let mut outputs = Vec::with_capacity(channels);
        let start = DECODE_BUFFER_SIZE - n;
        for ch in 0..channels {
            let channel_slice =
                unsafe { core::slice::from_raw_parts_mut(decode_mem_ptr.add(ch * stride), stride) };
            outputs.push(&mut channel_slice[start..]);
        }

        let downsample = max(decoder.downsample, 1) as usize;
        celt_synthesis(
            mode,
            &spectrum,
            &mut outputs,
            decoder.old_ebands,
            start_band,
            eff_end,
            channels,
            channels,
            false,
            lm,
            downsample,
            false,
            decoder.arch,
        );

        decoder.prefilter_and_fold = false;
        decoder.skip_plc = true;
    } else {
        let pitch_index = if loss_duration == 0 {
            let mut views = Vec::with_capacity(channels);
            for ch in 0..channels {
                let channel_slice =
                    unsafe { core::slice::from_raw_parts(decode_mem_ptr.add(ch * stride), stride) };
                views.push(channel_slice);
            }
            let search = celt_plc_pitch_search(&views, channels, decoder.arch);
            decoder.last_pitch_index = search;
            search
        } else {
            decoder.last_pitch_index
        };

        let fade = if loss_duration == 0 { 1.0f32 } else { 0.8_f32 };

        let max_period = MAX_PERIOD as usize;
        let pitch_index = pitch_index.clamp(PLC_PITCH_LAG_MIN, PLC_PITCH_LAG_MAX);
        let pitch_index_usize = min(pitch_index as usize, max_period);

        let exc_length = min(2 * pitch_index_usize, max_period);

        let mut exc = vec![0.0f32; max_period + LPC_ORDER];
        let mut fir_tmp = vec![0.0f32; exc_length];

        for ch in 0..channels {
            let channel_slice =
                unsafe { core::slice::from_raw_parts_mut(decode_mem_ptr.add(ch * stride), stride) };

            for (i, slot) in exc.iter_mut().enumerate() {
                let src = DECODE_BUFFER_SIZE + overlap - max_period - LPC_ORDER + i;
                *slot = channel_slice[src];
            }

            if loss_duration == 0 {
                let mut ac = [0.0f32; LPC_ORDER + 1];
                let input = &exc[LPC_ORDER..LPC_ORDER + max_period];
                let window = if overlap == 0 {
                    None
                } else {
                    Some(&mode.window[..overlap])
                };
                celt_autocorr(input, &mut ac, window, overlap, LPC_ORDER, decoder.arch);
                ac[0] *= 1.0001;
                for (i, entry) in ac.iter_mut().enumerate().skip(1) {
                    let factor = 0.008_f32 * 0.008_f32 * (i * i) as f32;
                    *entry -= *entry * factor;
                }
                let base_idx = ch * LPC_ORDER;
                {
                    let lpc_slice = &mut decoder.lpc[base_idx..base_idx + LPC_ORDER];
                    celt_lpc(lpc_slice, &ac);
                }
            }

            let base_idx = ch * LPC_ORDER;
            let lpc_coeffs = &decoder.lpc[base_idx..base_idx + LPC_ORDER];
            let start = max_period - exc_length;
            for (idx, value) in fir_tmp.iter_mut().enumerate() {
                let mut acc = exc[LPC_ORDER + start + idx];
                for (tap, coeff) in lpc_coeffs.iter().enumerate() {
                    let hist_index = LPC_ORDER + start + idx - 1 - tap;
                    acc += coeff * exc[hist_index];
                }
                *value = acc;
            }
            for (idx, value) in fir_tmp.iter().enumerate() {
                exc[LPC_ORDER + start + idx] = *value;
            }

            let mut e1 = 1.0f32;
            let mut e2 = 1.0f32;
            let decay_length = exc_length / 2;
            if decay_length > 0 {
                let exc_slice = &exc[LPC_ORDER..];
                for i in 0..decay_length {
                    let a = exc_slice[max_period - decay_length + i];
                    e1 += a * a;
                    let b = exc_slice[max_period - 2 * decay_length + i];
                    e2 += b * b;
                }
            }
            e1 = e1.min(e2);
            let decay = celt_sqrt(frac_div32(0.5 * e1, e2));

            let move_len = DECODE_BUFFER_SIZE
                .checked_sub(n)
                .expect("frame size larger than decode buffer");
            channel_slice.copy_within(n..n + move_len, 0);

            let extrapolation_offset = max_period - pitch_index_usize;
            let extrapolation_len = n + overlap;
            let mut attenuation = fade * decay;
            let mut j = 0usize;
            let start_index = DECODE_BUFFER_SIZE - n;
            let reference_base = DECODE_BUFFER_SIZE - max_period - n + extrapolation_offset;
            let mut s1 = 0.0f32;
            for i in 0..extrapolation_len {
                if j >= pitch_index_usize {
                    j -= pitch_index_usize;
                    attenuation *= decay;
                }
                let sample = attenuation * exc[LPC_ORDER + extrapolation_offset + j];
                channel_slice[start_index + i] = sample;
                let reference = channel_slice[reference_base + j];
                s1 += reference * reference;
                j += 1;
            }

            let mut lpc_mem = vec![0.0f32; LPC_ORDER];
            for (idx, mem) in lpc_mem.iter_mut().enumerate() {
                *mem = channel_slice[start_index - 1 - idx];
            }

            let mut filtered = channel_slice[start_index..start_index + extrapolation_len].to_vec();
            let input = filtered.clone();
            celt_iir(&input, lpc_coeffs, &mut filtered, &mut lpc_mem);
            channel_slice[start_index..start_index + extrapolation_len].copy_from_slice(&filtered);

            let mut s2 = 0.0f32;
            for sample in &filtered {
                s2 += sample * sample;
            }

            let threshold = 0.2 * s2;
            if matches!(s1.partial_cmp(&threshold), Some(Ordering::Greater)) {
                if matches!(s1.partial_cmp(&s2), Some(Ordering::Less)) {
                    let ratio = celt_sqrt(frac_div32(0.5 * s1 + 1.0, s2 + 1.0));
                    for i in 0..overlap {
                        let gain = 1.0 - mode.window[i] * (1.0 - ratio);
                        channel_slice[start_index + i] *= gain;
                    }
                    for i in overlap..extrapolation_len {
                        channel_slice[start_index + i] *= ratio;
                    }
                }
            } else {
                for value in &mut channel_slice[start_index..start_index + extrapolation_len] {
                    *value = 0.0;
                }
            }
        }

        decoder.prefilter_and_fold = true;
    }

    let increment = ((1usize) << lm) as i32;
    decoder.loss_duration = min(10_000, decoder.loss_duration + increment);
}

/// Maximum number of channels supported by the initial CELT decoder port.
///
/// The reference implementation restricts the custom decoder to mono or stereo
/// streams.  The helper routines below mirror the same validation so the
/// call-sites can rely on early argument checking just like the C helpers.
const MAX_CHANNELS: usize = 2;

const MODE_UNINITIALISED: u8 = 0;
const MODE_INITIALISING: u8 = 1;
const MODE_READY: u8 = 2;

struct CanonicalModeCell {
    state: AtomicU8,
    value: UnsafeCell<Option<OpusCustomMode<'static>>>,
}

unsafe impl Sync for CanonicalModeCell {}

static CANONICAL_MODE: CanonicalModeCell = CanonicalModeCell {
    state: AtomicU8::new(MODE_UNINITIALISED),
    value: UnsafeCell::new(None),
};

pub(crate) fn canonical_mode() -> Option<&'static OpusCustomMode<'static>> {
    loop {
        match CANONICAL_MODE.state.load(AtomicOrdering::Acquire) {
            MODE_READY => unsafe {
                return (*CANONICAL_MODE.value.get()).as_ref();
            },
            MODE_UNINITIALISED => {
                if CANONICAL_MODE
                    .state
                    .compare_exchange(
                        MODE_UNINITIALISED,
                        MODE_INITIALISING,
                        AtomicOrdering::Acquire,
                        AtomicOrdering::Relaxed,
                    )
                    .is_ok()
                {
                    let mode = opus_custom_mode_find_static(48_000, 960);
                    unsafe {
                        *CANONICAL_MODE.value.get() = mode;
                    }
                    CANONICAL_MODE
                        .state
                        .store(MODE_READY, AtomicOrdering::Release);
                    unsafe {
                        return (*CANONICAL_MODE.value.get()).as_ref();
                    }
                }
            }
            _ => core::hint::spin_loop(),
        }
    }
}

/// Cumulative distribution used to decode the global allocation trim.
const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// Spread decision probabilities used by the transient classifier.
const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];

/// Probability model for the three post-filter tapset candidates.
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// Scalar used to decode the post-filter gain from the coarse index.
const POSTFILTER_GAIN_SCALE: OpusVal16 = 0.09375;

const VERY_SMALL: CeltSig = 1.0e-30;
const INV_CELT_SIG_SCALE: f32 = 1.0 / CELT_SIG_SCALE;

#[inline]
fn multiply_coef(coef: OpusVal16, value: CeltSig) -> CeltSig {
    coef * value
}

#[inline]
fn sig_to_res(value: CeltSig) -> OpusRes {
    value * INV_CELT_SIG_SCALE
}

#[inline]
fn add_res(lhs: OpusRes, rhs: OpusRes) -> OpusRes {
    lhs + rhs
}

#[inline]
fn preprocess_sample(sample: CeltSig, mem: CeltSig) -> CeltSig {
    sample + mem + VERY_SMALL
}

#[inline]
fn shl32(value: CeltSig, _shift: i32) -> CeltSig {
    value
}

#[inline]
fn sub_celt(lhs: CeltSig, rhs: CeltSig) -> CeltSig {
    lhs - rhs
}

fn deemphasis_stereo_simple(
    input: &[&[CeltSig]],
    pcm: &mut [OpusRes],
    n: usize,
    coef0: OpusVal16,
    mem: &mut [CeltSig],
) {
    debug_assert!(input.len() >= 2, "stereo deemphasis requires two channels");
    debug_assert!(
        pcm.len() >= 2 * n,
        "PCM buffer must hold interleaved stereo samples"
    );
    debug_assert!(
        mem.len() >= 2,
        "pre-emphasis memory must expose two channels"
    );

    let left = input[0];
    let right = input[1];
    debug_assert!(left.len() >= n, "left channel does not expose N samples");
    debug_assert!(right.len() >= n, "right channel does not expose N samples");

    let mut mem_left = mem[0];
    let mut mem_right = mem[1];

    for (j, (&left_sample, &right_sample)) in left.iter().zip(right.iter()).take(n).enumerate() {
        let tmp_left = preprocess_sample(left_sample, mem_left);
        let tmp_right = preprocess_sample(right_sample, mem_right);

        mem_left = multiply_coef(coef0, tmp_left);
        mem_right = multiply_coef(coef0, tmp_right);

        pcm[2 * j] = sig_to_res(tmp_left);
        pcm[2 * j + 1] = sig_to_res(tmp_right);
    }

    mem[0] = mem_left;
    mem[1] = mem_right;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn deemphasis(
    input: &[&[CeltSig]],
    pcm: &mut [OpusRes],
    n: usize,
    channels: usize,
    downsample: usize,
    coef: &[OpusVal16],
    mem: &mut [CeltSig],
    accum: bool,
) {
    if n == 0 || channels == 0 {
        return;
    }

    debug_assert!(downsample > 0, "downsample factor must be non-zero");
    debug_assert!(
        input.len() >= channels,
        "input must expose one slice per channel"
    );
    debug_assert!(
        mem.len() >= channels,
        "memory buffer must expose one value per channel"
    );
    debug_assert!(
        !coef.is_empty(),
        "pre-emphasis coefficients must not be empty"
    );

    let expected_samples = if downsample > 1 { n / downsample } else { n };
    debug_assert!(
        pcm.len() >= expected_samples * channels,
        "PCM buffer too small for deemphasis output",
    );

    if downsample == 1 && channels == 2 && !accum {
        deemphasis_stereo_simple(input, pcm, n, coef[0], mem);
        return;
    }

    let mut scratch = vec![CeltSig::default(); n];
    let coef0 = coef[0];
    let nd = n / downsample;

    for channel in 0..channels {
        let samples = input[channel];
        debug_assert!(
            samples.len() >= n,
            "channel {} does not expose N samples",
            channel
        );

        let mut m = mem[channel];
        let mut apply_downsampling = false;

        if coef.len() > 3 && coef[1] != 0.0 {
            let coef1 = coef[1];
            let coef3 = coef[3];
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample(sample, m);
                m = sub_celt(multiply_coef(coef0, tmp), multiply_coef(coef1, sample));
                scratch[j] = shl32(multiply_coef(coef3, tmp), 2);
            }
            apply_downsampling = true;
        } else if downsample > 1 {
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample(sample, m);
                m = multiply_coef(coef0, tmp);
                scratch[j] = tmp;
            }
            apply_downsampling = true;
        } else if accum {
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample(sample, m);
                m = multiply_coef(coef0, tmp);
                let idx = j * channels + channel;
                let converted = sig_to_res(tmp);
                pcm[idx] = add_res(pcm[idx], converted);
            }
        } else {
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample(sample, m);
                m = multiply_coef(coef0, tmp);
                let idx = j * channels + channel;
                pcm[idx] = sig_to_res(tmp);
            }
        }

        mem[channel] = m;

        if apply_downsampling {
            for j in 0..nd {
                let idx = j * channels + channel;
                let converted = sig_to_res(scratch[j * downsample]);
                if accum {
                    pcm[idx] = add_res(pcm[idx], converted);
                } else {
                    pcm[idx] = converted;
                }
            }
        }
    }
}

/// Layout stub mirroring the portion of the C decoder that precedes the
/// variable-length trailing buffers.
///
/// The reference implementation stores the primary decoder fields followed by
/// a single-sample `_decode_mem` placeholder. Additional per-channel history,
/// LPC coefficients, and energy memories are allocated immediately afterwards.
/// Recreating the fixed prefix here lets the Rust port reproduce the sizing
/// calculations performed by `opus_custom_decoder_get_size()` without relying
/// on raw pointer arithmetic.
#[repr(C)]
struct DecoderLayoutStub {
    mode: *const (),
    overlap: i32,
    channels: i32,
    stream_channels: i32,
    downsample: i32,
    start_band: i32,
    end_band: i32,
    signalling: i32,
    disable_inv: i32,
    complexity: i32,
    arch: i32,
    rng: OpusUint32,
    error: i32,
    last_pitch_index: i32,
    loss_duration: i32,
    skip_plc: i32,
    postfilter_period: i32,
    postfilter_period_old: i32,
    postfilter_gain: OpusVal16,
    postfilter_gain_old: OpusVal16,
    postfilter_tapset: i32,
    postfilter_tapset_old: i32,
    prefilter_and_fold: i32,
    preemph_mem_decoder: [CeltSig; 2],
    decode_mem_head: [CeltSig; 1],
}

/// Size of the fixed decoder prefix in bytes.
const DECODER_PREFIX_SIZE: usize = core::mem::size_of::<DecoderLayoutStub>();

/// Returns the number of bytes required to allocate a decoder for `mode`.
#[must_use]
pub(crate) fn opus_custom_decoder_get_size(
    mode: &OpusCustomMode<'_>,
    channels: usize,
) -> Option<usize> {
    if channels == 0 || channels > MAX_CHANNELS {
        return None;
    }

    let decode_mem = channels * (DECODE_BUFFER_SIZE + mode.overlap);
    let lpc = channels * LPC_ORDER;
    let band_history = 2 * mode.num_ebands;

    let size = DECODER_PREFIX_SIZE
        + (decode_mem - 1) * core::mem::size_of::<CeltSig>()
        + lpc * core::mem::size_of::<OpusVal16>()
        + 4 * band_history * core::mem::size_of::<CeltGlog>();
    Some(size)
}

/// Returns the size of the canonical CELT decoder operating at 48 kHz/960.
#[must_use]
pub(crate) fn celt_decoder_get_size(channels: usize) -> Option<usize> {
    opus_custom_mode_find_static(48_000, 960)
        .and_then(|mode| opus_custom_decoder_get_size(&mode, channels))
}

/// Owning wrapper around [`OpusCustomDecoder`] and its backing allocation.
///
/// The C implementation returns an opaque pointer whose trailing buffers live in
/// the same allocation as the primary decoder fields.  In Rust we model this by
/// keeping the [`CeltDecoderAlloc`] inside a [`Box`] and storing the decoder
/// view alongside it.  This ensures the borrowed slices remain valid for the
/// lifetime of the wrapper while keeping the ownership semantics of
/// `opus_custom_decoder_create()` intact.
#[derive(Debug)]
pub(crate) struct OwnedCeltDecoder<'mode> {
    decoder: OpusCustomDecoder<'mode>,
    alloc: Box<CeltDecoderAlloc>,
}

impl<'mode> OwnedCeltDecoder<'mode> {
    /// Borrows the underlying decoder state.
    #[must_use]
    pub fn decoder(&mut self) -> &mut OpusCustomDecoder<'mode> {
        &mut self.decoder
    }

    /// Creates a new owned decoder for `mode` and `channels`.
    pub fn new(
        mode: &'mode OpusCustomMode<'mode>,
        channels: usize,
    ) -> Result<Self, CeltDecoderInitError> {
        opus_custom_decoder_create(mode, channels)
    }
}

impl<'mode> AsRef<OwnedCeltDecoder<'mode>> for OwnedCeltDecoder<'mode> {
    fn as_ref(&self) -> &OwnedCeltDecoder<'mode> {
        self
    }
}

impl<'mode> AsMut<OwnedCeltDecoder<'mode>> for OwnedCeltDecoder<'mode> {
    fn as_mut(&mut self) -> &mut OwnedCeltDecoder<'mode> {
        self
    }
}

impl<'mode> core::ops::Deref for OwnedCeltDecoder<'mode> {
    type Target = OpusCustomDecoder<'mode>;

    fn deref(&self) -> &Self::Target {
        &self.decoder
    }
}

impl<'mode> core::ops::DerefMut for OwnedCeltDecoder<'mode> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.decoder
    }
}

impl<'mode> AsRef<OpusCustomDecoder<'mode>> for OwnedCeltDecoder<'mode> {
    fn as_ref(&self) -> &OpusCustomDecoder<'mode> {
        &self.decoder
    }
}

impl<'mode> AsMut<OpusCustomDecoder<'mode>> for OwnedCeltDecoder<'mode> {
    fn as_mut(&mut self) -> &mut OpusCustomDecoder<'mode> {
        &mut self.decoder
    }
}

/// Initialises a decoder for the canonical 48 kHz / 960 sample configuration.
///
/// Mirrors `celt_decoder_init()` from `celt/celt_decoder.c` by borrowing the
/// statically defined mode, delegating to [`opus_custom_decoder_init`], and
/// updating the downsampling factor based on the caller-provided sampling rate.
pub(crate) fn celt_decoder_init<'a>(
    alloc: &'a mut CeltDecoderAlloc,
    sampling_rate: OpusInt32,
    channels: usize,
) -> Result<OpusCustomDecoder<'a>, CeltDecoderInitError> {
    let mode = canonical_mode().ok_or(CeltDecoderInitError::CanonicalModeUnavailable)?;
    let mut decoder = alloc.init_decoder(mode, channels, channels)?;

    let factor = resampling_factor(sampling_rate);
    if factor == 0 {
        return Err(CeltDecoderInitError::UnsupportedSampleRate);
    }

    decoder.downsample = factor as i32;

    Ok(decoder)
}

/// Errors that can be reported while preparing to decode a CELT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltDecodeError {
    /// Input arguments were inconsistent with the current decoder state.
    BadArgument,
    /// The supplied packet was too short to decode a frame and should trigger PLC.
    PacketLoss,
    /// The packet signalled an invalid configuration or ran out of bits.
    InvalidPacket,
}

/// Range decoder storage mirroring the temporary buffer used by the C decoder.
#[derive(Debug)]
pub(crate) struct RangeDecoderState {
    decoder: EcDec<'static>,
    storage: NonNull<[u8]>,
}

impl RangeDecoderState {
    /// Creates a new range decoder backed by an owned copy of the packet payload.
    pub(crate) fn new(data: &[u8]) -> Self {
        let boxed = data.to_vec().into_boxed_slice();
        let storage = unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) };
        // SAFETY: The raw pointer originates from a boxed slice that is retained and
        // later reconstructed in `Drop`. The lifetime extension is therefore sound
        // because the memory outlives the decoder instance.
        let decoder = {
            let slice: &'static mut [u8] = unsafe { &mut *storage.as_ptr() };
            EcDec::new(slice)
        };
        Self { decoder, storage }
    }

    /// Borrows the underlying range decoder with the appropriate lifetime.
    pub(crate) fn decoder(&mut self) -> &mut EcDec<'static> {
        &mut self.decoder
    }

    /// Borrows the decoder using the mode lifetime expected by band quantisation helpers.
    pub(crate) fn decoder_for_mode<'mode>(
        &mut self,
        _mode: &'mode OpusCustomMode<'mode>,
    ) -> &mut EcDec<'mode> {
        // SAFETY: The backing buffer is owned by `self` and outlives the returned borrow.
        unsafe { core::mem::transmute::<&mut EcDec<'static>, &mut EcDec<'mode>>(&mut self.decoder) }
    }
}

impl Drop for RangeDecoderState {
    fn drop(&mut self) {
        // SAFETY: `storage` was created from a `Box<[u8]>` via `Box::into_raw` and is
        // only reconstructed once the decoder (which holds the unique borrow) has
        // been dropped.
        unsafe {
            let _ = Box::from_raw(self.storage.as_ptr());
        }
    }
}

/// Debug-time validation mirroring `validate_celt_decoder()` from the C sources.
///
/// The reference implementation relies on this helper to assert that the decoder
/// state remains internally consistent after initialisation and before decoding
/// a frame. The Rust translation mirrors the same invariants so regressions are
/// caught early while the remaining decode path is ported.
pub(crate) fn validate_celt_decoder(decoder: &OpusCustomDecoder<'_>) {
    debug_assert_eq!(
        decoder.overlap, decoder.mode.overlap,
        "decoder overlap must match the mode configuration",
    );

    let mode_band_limit = decoder.mode.num_ebands as i32;
    let standard_limit = 21;
    let custom_limit = 25;
    let end_limit = if mode_band_limit <= standard_limit {
        mode_band_limit
    } else {
        mode_band_limit.min(custom_limit)
    };
    debug_assert!(
        decoder.end_band <= end_limit,
        "end band {} exceeds supported limit {}",
        decoder.end_band,
        end_limit
    );

    debug_assert!(
        decoder.channels == 1 || decoder.channels == 2,
        "decoder must be mono or stereo",
    );
    debug_assert!(
        decoder.stream_channels == 1 || decoder.stream_channels == 2,
        "stream must be mono or stereo",
    );
    debug_assert!(
        decoder.downsample > 0,
        "downsample factor must be strictly positive",
    );
    debug_assert!(
        decoder.start_band == 0 || decoder.start_band == 17,
        "decoder start band must be 0 or 17",
    );
    debug_assert!(
        decoder.start_band < decoder.end_band,
        "start band must precede end band",
    );

    debug_assert!(
        decoder.arch >= 0 && decoder.arch <= OPUS_ARCHMASK,
        "architecture selection out of range",
    );

    debug_assert!(
        decoder.last_pitch_index <= PLC_PITCH_LAG_MAX,
        "last pitch index exceeds maximum lag",
    );
    debug_assert!(
        decoder.last_pitch_index >= PLC_PITCH_LAG_MIN || decoder.last_pitch_index == 0,
        "last pitch index below minimum lag",
    );

    debug_assert!(
        decoder.postfilter_period < MAX_PERIOD,
        "postfilter period must remain below MAX_PERIOD",
    );
    debug_assert!(
        decoder.postfilter_period >= COMBFILTER_MINPERIOD as i32 || decoder.postfilter_period == 0,
        "postfilter period must be zero or above the comb-filter minimum",
    );
    debug_assert!(
        decoder.postfilter_period_old < MAX_PERIOD,
        "previous postfilter period must remain below MAX_PERIOD",
    );
    debug_assert!(
        decoder.postfilter_period_old >= COMBFILTER_MINPERIOD as i32
            || decoder.postfilter_period_old == 0,
        "previous postfilter period must be zero or above the comb-filter minimum",
    );

    debug_assert!(
        (0..=2).contains(&decoder.postfilter_tapset),
        "postfilter tapset must be in the inclusive range [0, 2]",
    );
    debug_assert!(
        (0..=2).contains(&decoder.postfilter_tapset_old),
        "previous postfilter tapset must be in the inclusive range [0, 2]",
    );
}

/// Metadata describing the parsed frame header and bit allocation.
#[derive(Debug)]
pub(crate) struct FramePreparation {
    pub range_decoder: Option<RangeDecoderState>,
    pub tf_res: Vec<OpusInt32>,
    pub cap: Vec<OpusInt32>,
    pub offsets: Vec<OpusInt32>,
    pub fine_quant: Vec<OpusInt32>,
    pub pulses: Vec<OpusInt32>,
    pub fine_priority: Vec<OpusInt32>,
    pub spread_decision: OpusInt32,
    pub is_transient: bool,
    pub short_blocks: OpusInt32,
    pub intra_ener: bool,
    pub silence: bool,
    pub alloc_trim: OpusInt32,
    pub anti_collapse_rsv: OpusInt32,
    pub intensity: OpusInt32,
    pub dual_stereo: OpusInt32,
    pub balance: OpusInt32,
    pub coded_bands: OpusInt32,
    pub postfilter_pitch: OpusInt32,
    pub postfilter_gain: OpusVal16,
    pub postfilter_tapset: OpusInt32,
    pub total_bits: OpusInt32,
    pub tell: OpusInt32,
    pub bits: OpusInt32,
    pub start: usize,
    pub end: usize,
    pub eff_end: usize,
    pub lm: usize,
    pub m: usize,
    pub n: usize,
    pub c: usize,
    pub cc: usize,
    pub packet_loss: bool,
}

impl FramePreparation {
    #[allow(clippy::too_many_arguments)]
    fn new_packet_loss(
        start: usize,
        end: usize,
        eff_end: usize,
        nb_ebands: usize,
        lm: usize,
        m: usize,
        n: usize,
        c: usize,
        cc: usize,
    ) -> Self {
        let tf_res = vec![0; nb_ebands];
        let cap = vec![0; nb_ebands];
        let zeros = vec![0; nb_ebands];
        Self {
            range_decoder: None,
            tf_res,
            cap,
            offsets: zeros.clone(),
            fine_quant: zeros.clone(),
            pulses: zeros.clone(),
            fine_priority: zeros,
            spread_decision: 0,
            is_transient: false,
            short_blocks: 0,
            intra_ener: false,
            silence: false,
            alloc_trim: 0,
            anti_collapse_rsv: 0,
            intensity: 0,
            dual_stereo: 0,
            balance: 0,
            coded_bands: 0,
            postfilter_pitch: 0,
            postfilter_gain: 0.0,
            postfilter_tapset: 0,
            total_bits: 0,
            tell: 0,
            bits: 0,
            start,
            end,
            eff_end,
            lm,
            m,
            n,
            c,
            cc,
            packet_loss: true,
        }
    }
}

fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [OpusInt32],
    lm: usize,
    dec: &mut EcDec<'_>,
) {
    let mut budget = dec.ctx().storage * 8;
    let mut tell = entcode::ec_tell(dec.ctx()) as u32;
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    if tf_select_rsv {
        budget -= 1;
    }

    let mut curr = 0;
    let mut tf_changed = 0;
    for slot in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            let bit = dec.dec_bit_logp(logp);
            curr ^= bit;
            tell = entcode::ec_tell(dec.ctx()) as u32;
            tf_changed |= curr;
        }
        *slot = curr;
        logp = if is_transient { 4 } else { 5 };
    }

    let mut tf_select = 0;
    if tf_select_rsv {
        let base = 4 * usize::from(is_transient);
        if TF_SELECT_TABLE[lm][base + tf_changed as usize]
            != TF_SELECT_TABLE[lm][base + 2 + tf_changed as usize]
        {
            tf_select = dec.dec_bit_logp(1) as OpusInt32;
        }
    }

    let base = 4 * usize::from(is_transient);
    for slot in tf_res.iter_mut().take(end).skip(start) {
        let idx = base + 2 * tf_select as usize + *slot as usize;
        *slot = OpusInt32::from(TF_SELECT_TABLE[lm][idx]);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_frame(
    decoder: &mut OpusCustomDecoder<'_>,
    packet: &[u8],
    frame_size: usize,
) -> Result<FramePreparation, CeltDecodeError> {
    let cc = decoder.channels;
    let c = decoder.stream_channels;
    if cc == 0 || c == 0 || cc > MAX_CHANNELS {
        return Err(CeltDecodeError::BadArgument);
    }

    if frame_size == 0 {
        return Err(CeltDecodeError::BadArgument);
    }

    let mode = decoder.mode;
    let nb_ebands = mode.num_ebands;
    let start = decoder.start_band as usize;
    let end = decoder.end_band as usize;

    let downsample = decoder.downsample as usize;
    let scaled_frame = frame_size
        .checked_mul(downsample)
        .ok_or(CeltDecodeError::BadArgument)?;
    if scaled_frame > (mode.short_mdct_size << mode.max_lm) {
        return Err(CeltDecodeError::BadArgument);
    }

    let lm = (0..=mode.max_lm)
        .find(|&cand| mode.short_mdct_size << cand == scaled_frame)
        .ok_or(CeltDecodeError::BadArgument)?;
    let m = 1 << lm;
    let n = m * mode.short_mdct_size;

    if packet.len() > 1275 {
        return Err(CeltDecodeError::BadArgument);
    }

    let eff_end = min(end, mode.effective_ebands);

    if packet.len() <= 1 {
        return Ok(FramePreparation::new_packet_loss(
            start, end, eff_end, nb_ebands, lm, m, n, c, cc,
        ));
    }

    if decoder.loss_duration == 0 {
        decoder.skip_plc = false;
    }

    let mut range_decoder = RangeDecoderState::new(packet);
    let dec = range_decoder.decoder();

    if c == 1 {
        for band in 0..nb_ebands {
            let idx = band;
            let paired = nb_ebands + band;
            decoder.old_ebands[idx] = decoder.old_ebands[idx].max(decoder.old_ebands[paired]);
        }
    }

    let len_bits = (packet.len() * 8) as OpusInt32;
    let mut tell = entcode::ec_tell(dec.ctx());
    let mut silence = false;
    if tell >= len_bits {
        silence = true;
    } else if tell == 1 {
        silence = dec.dec_bit_logp(15) != 0;
    }
    if silence {
        let consumed = entcode::ec_tell(dec.ctx());
        dec.ctx_mut().nbits_total += len_bits - consumed;
        tell = len_bits;
    } else {
        tell = entcode::ec_tell(dec.ctx());
    }

    let mut postfilter_gain = 0.0;
    let mut postfilter_pitch = 0;
    let mut postfilter_tapset = 0;
    if start == 0 && tell + 16 <= len_bits {
        if dec.dec_bit_logp(1) != 0 {
            let octave = dec.dec_uint(6) as OpusInt32;
            let low_bits = dec.dec_bits((4 + octave) as u32) as OpusInt32;
            postfilter_pitch = ((16 << octave) + low_bits) - 1;
            let qg = dec.dec_bits(3) as OpusInt32;
            if entcode::ec_tell(dec.ctx()) + 2 <= len_bits {
                postfilter_tapset = dec.dec_icdf(&TAPSET_ICDF, 2);
            }
            postfilter_gain = POSTFILTER_GAIN_SCALE * ((qg + 1) as OpusVal16);
        }
        tell = entcode::ec_tell(dec.ctx());
    }

    let mut is_transient = false;
    if lm > 0 && tell + 3 <= len_bits {
        is_transient = dec.dec_bit_logp(3) != 0;
        tell = entcode::ec_tell(dec.ctx());
    }
    let short_blocks = if is_transient { m as OpusInt32 } else { 0 };

    let mut intra_ener = false;
    if tell + 3 <= len_bits {
        intra_ener = dec.dec_bit_logp(3) != 0;
    }

    if !intra_ener && decoder.loss_duration != 0 {
        let missing = min(10, decoder.loss_duration >> (lm as u32));
        let safety = match lm {
            0 => 1.5,
            1 => 0.5,
            _ => 0.0,
        };

        for ch in 0..2 {
            for band in start..end {
                let idx = ch * nb_ebands + band;
                let mut e0 = decoder.old_ebands[idx];
                let e1 = decoder.old_log_e[idx];
                let e2 = decoder.old_log_e2[idx];
                if e0 < e1.max(e2) {
                    let mut slope = (e1 - e0).max(0.5 * (e2 - e0));
                    slope = slope.min(2.0);
                    let reduction = (((missing + 1) as f32) * slope).max(0.0);
                    e0 -= reduction;
                    decoder.old_ebands[idx] = e0.max(-20.0);
                } else {
                    decoder.old_ebands[idx] = decoder.old_ebands[idx].min(e1.min(e2));
                }
                decoder.old_ebands[idx] -= safety;
            }
        }
    }

    unquant_coarse_energy(mode, start, end, decoder.old_ebands, intra_ener, dec, c, lm);

    let mut tf_res = vec![0; nb_ebands];
    tf_decode(start, end, is_transient, &mut tf_res, lm, dec);

    tell = entcode::ec_tell(dec.ctx());
    let mut spread_decision = SPREAD_NORMAL;
    if tell + 4 <= len_bits {
        spread_decision = dec.dec_icdf(&SPREAD_ICDF, 5);
    }

    let mut cap = vec![0; nb_ebands];
    init_caps(mode, &mut cap, lm, c);

    let mut offsets = vec![0; nb_ebands];
    let mut dynalloc_logp = 6;
    let mut total_bits = len_bits << BITRES;
    let mut tell_frac = entcode::ec_tell_frac(dec.ctx()) as OpusInt32;

    for band in start..end {
        let band_width = i32::from(mode.e_bands[band + 1] - mode.e_bands[band]);
        let width = (c as OpusInt32 * band_width) << lm;
        let six_bits = (6 << BITRES) as OpusInt32;
        let quanta = min(width << BITRES, max(six_bits, width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let mut boost = 0;
        while tell_frac + ((dynalloc_loop_logp as OpusInt32) << BITRES) < total_bits
            && boost < cap[band]
        {
            let flag = dec.dec_bit_logp(dynalloc_loop_logp as u32);
            tell_frac = entcode::ec_tell_frac(dec.ctx()) as OpusInt32;
            if flag == 0 {
                break;
            }
            boost += quanta;
            total_bits -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[band] = boost;
        if boost > 0 {
            dynalloc_logp = max(2, dynalloc_logp - 1);
        }
    }

    let mut fine_quant = vec![0; nb_ebands];
    let alloc_trim = if tell_frac + ((6 << BITRES) as OpusInt32) <= total_bits {
        dec.dec_icdf(&TRIM_ICDF, 7)
    } else {
        5
    };

    let mut bits = ((len_bits << BITRES) - entcode::ec_tell_frac(dec.ctx()) as OpusInt32) - 1;
    let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm as OpusInt32 + 2) << BITRES)
    {
        (1 << BITRES) as OpusInt32
    } else {
        0
    };
    bits -= anti_collapse_rsv;

    let mut pulses = vec![0; nb_ebands];
    let mut fine_priority = vec![0; nb_ebands];
    let mut intensity = 0;
    let mut dual_stereo = 0;
    let mut balance = 0;

    let coded_bands = clt_compute_allocation(
        mode,
        start,
        end,
        &offsets,
        &cap,
        alloc_trim,
        &mut intensity,
        &mut dual_stereo,
        bits,
        &mut balance,
        &mut pulses,
        &mut fine_quant,
        &mut fine_priority,
        c as OpusInt32,
        lm as OpusInt32,
        None,
        Some(dec),
        0,
        0,
    );

    unquant_fine_energy(mode, start, end, decoder.old_ebands, &fine_quant, dec, c);

    let tell = entcode::ec_tell(dec.ctx());

    Ok(FramePreparation {
        range_decoder: Some(range_decoder),
        tf_res,
        cap,
        offsets,
        fine_quant,
        pulses,
        fine_priority,
        spread_decision,
        is_transient,
        short_blocks,
        intra_ener,
        silence,
        alloc_trim,
        anti_collapse_rsv,
        intensity,
        dual_stereo,
        balance,
        coded_bands,
        postfilter_pitch,
        postfilter_gain,
        postfilter_tapset,
        total_bits,
        tell,
        bits,
        start,
        end,
        eff_end,
        lm,
        m,
        n,
        c,
        cc,
        packet_loss: false,
    })
}

#[cfg(not(feature = "fixed_point"))]
fn res_to_int24(sample: OpusRes) -> i32 {
    let scale = CELT_SIG_SCALE * 256.0;
    let scaled = (sample * scale).clamp(-8_388_608.0, 8_388_607.0);
    float2int(scaled)
}

#[cfg(not(feature = "fixed_point"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn celt_decode_with_ec_dred(
    decoder: &mut OpusCustomDecoder<'_>,
    packet: Option<&[u8]>,
    pcm: &mut [OpusRes],
    frame_size: usize,
    range_decoder: Option<&mut EcDec<'_>>,
    accum: bool,
) -> Result<usize, CeltDecodeError> {
    // The reference implementation keeps the range decoder internal to the DRED
    // helper, so passing an external instance would diverge from the validated
    // control flow.  Mirror that constraint until the stand-alone range decoder
    // path is ported.
    if range_decoder.into_iter().next().is_some() {
        return Err(CeltDecodeError::BadArgument);
    }

    validate_celt_decoder(decoder);

    let data = packet.unwrap_or(&[]);
    let mut frame = prepare_frame(decoder, data, frame_size)?;
    let mode = decoder.mode;
    let nb_ebands = mode.num_ebands;
    let overlap = mode.overlap;
    let stride = DECODE_BUFFER_SIZE + overlap;

    let downsample = decoder.downsample.max(1) as usize;
    let cc = frame.cc;
    if cc == 0 {
        return Err(CeltDecodeError::BadArgument);
    }

    let n = frame.n;
    let output_samples = n / downsample;
    let required_pcm = output_samples
        .checked_mul(cc)
        .ok_or(CeltDecodeError::BadArgument)?;
    if pcm.len() < required_pcm {
        return Err(CeltDecodeError::BadArgument);
    }

    if frame.packet_loss {
        celt_decode_lost(decoder, n, frame.lm);

        let mut inputs: Vec<&[CeltSig]> = Vec::with_capacity(cc);
        for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
            let start = DECODE_BUFFER_SIZE
                .checked_sub(n)
                .ok_or(CeltDecodeError::BadArgument)?;
            let (_, rest) = channel_slice.split_at(start);
            let (output, _) = rest.split_at(n);
            inputs.push(output);
        }

        deemphasis(
            &inputs,
            pcm,
            n,
            cc,
            downsample,
            &mode.pre_emphasis,
            &mut decoder.preemph_mem_decoder,
            accum,
        );

        return Ok(output_samples);
    }

    let mut range_state = frame
        .range_decoder
        .take()
        .ok_or(CeltDecodeError::InvalidPacket)?;
    let FramePreparation {
        range_decoder: _,
        tf_res,
        cap: _,
        offsets: _,
        fine_quant,
        pulses,
        fine_priority,
        spread_decision,
        is_transient,
        short_blocks,
        intra_ener: _,
        silence,
        alloc_trim: _,
        anti_collapse_rsv,
        intensity,
        dual_stereo,
        balance,
        coded_bands,
        postfilter_pitch,
        postfilter_gain,
        postfilter_tapset,
        total_bits,
        tell: _,
        bits: _,
        start,
        end,
        eff_end,
        lm,
        m,
        n,
        c,
        cc,
        packet_loss: _,
    } = frame;

    let packet_len = data.len();

    for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
        let move_len = DECODE_BUFFER_SIZE
            .checked_sub(n)
            .ok_or(CeltDecodeError::BadArgument)?
            + overlap;
        channel_slice.copy_within(n..n + move_len, 0);
    }

    let mut collapse_masks = vec![0u8; c * nb_ebands];
    let mut spectrum = vec![0.0f32; c * n];
    let total_available = total_bits - anti_collapse_rsv;
    let (first_channel, second_channel_opt) = if c == 2 {
        let (left, right) = spectrum.split_at_mut(n);
        (left, Some(right))
    } else {
        (&mut spectrum[..], None)
    };

    {
        let dec = range_state.decoder_for_mode(mode);
        let mut coder = BandCodingState::Decoder(dec);

        quant_all_bands(
            false,
            mode,
            start,
            end,
            first_channel,
            second_channel_opt,
            &mut collapse_masks,
            &[],
            &pulses,
            short_blocks != 0,
            spread_decision,
            dual_stereo != 0,
            intensity.max(0) as usize,
            &tf_res,
            total_available,
            balance,
            &mut coder,
            lm as i32,
            coded_bands.max(0) as usize,
            &mut decoder.rng,
            decoder.complexity,
            decoder.arch,
            decoder.disable_inv,
        );
    }
    let mut anti_collapse_on = false;
    let dec = range_state.decoder_for_mode(mode);

    if anti_collapse_rsv > 0 {
        anti_collapse_on = dec.dec_bits(1) != 0;
    }

    let remaining_bits = (packet_len as OpusInt32 * 8) - entcode::ec_tell(dec.ctx());
    unquant_energy_finalise(
        mode,
        start,
        end,
        decoder.old_ebands,
        &fine_quant,
        &fine_priority,
        remaining_bits,
        dec,
        c,
    );

    if anti_collapse_on {
        anti_collapse(
            mode,
            &mut spectrum,
            &collapse_masks,
            lm,
            c,
            n,
            start,
            end,
            decoder.old_ebands,
            decoder.old_log_e,
            decoder.old_log_e2,
            &pulses,
            decoder.rng,
            false,
            decoder.arch,
        );
    }

    if silence {
        decoder.old_ebands.fill(-28.0);
    }

    if decoder.prefilter_and_fold {
        prefilter_and_fold(decoder, n);
    }

    let mut out_slices: Vec<&mut [CeltSig]> = Vec::with_capacity(cc);
    for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
        let start_idx = DECODE_BUFFER_SIZE
            .checked_sub(n)
            .ok_or(CeltDecodeError::BadArgument)?;
        let (_, rest) = channel_slice.split_at_mut(start_idx);
        let (output, _) = rest.split_at_mut(n);
        out_slices.push(output);
    }

    celt_synthesis(
        mode,
        &spectrum,
        &mut out_slices,
        decoder.old_ebands,
        start,
        eff_end,
        c,
        cc,
        is_transient,
        lm,
        downsample,
        silence,
        decoder.arch,
    );

    drop(out_slices);

    decoder.postfilter_period = decoder.postfilter_period.max(COMBFILTER_MINPERIOD as i32);
    decoder.postfilter_period_old = decoder
        .postfilter_period_old
        .max(COMBFILTER_MINPERIOD as i32);

    for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
        let output_start = DECODE_BUFFER_SIZE - n;
        let channel_len = channel_slice.len();
        let channel_ptr = channel_slice.as_ptr();
        let output = &mut channel_slice[output_start..output_start + n];
        let full_channel = unsafe { core::slice::from_raw_parts(channel_ptr, channel_len) };

        let first_len = mode.short_mdct_size.min(output.len());
        if first_len > 0 {
            let (first_block, tail_block) = output.split_at_mut(first_len);
            comb_filter(
                first_block,
                full_channel,
                output_start,
                first_len,
                decoder.postfilter_period_old,
                decoder.postfilter_period,
                decoder.postfilter_gain_old,
                decoder.postfilter_gain,
                decoder.postfilter_tapset_old.max(0) as usize,
                decoder.postfilter_tapset.max(0) as usize,
                mode.window,
                overlap,
                decoder.arch,
            );

            if lm != 0 && !tail_block.is_empty() {
                let tail_start = output_start + first_len;
                comb_filter(
                    tail_block,
                    full_channel,
                    tail_start,
                    tail_block.len(),
                    decoder.postfilter_period,
                    postfilter_pitch,
                    decoder.postfilter_gain,
                    postfilter_gain,
                    decoder.postfilter_tapset.max(0) as usize,
                    postfilter_tapset.max(0) as usize,
                    mode.window,
                    overlap,
                    decoder.arch,
                );
            }
        }
    }

    decoder.postfilter_period_old = decoder.postfilter_period;
    decoder.postfilter_gain_old = decoder.postfilter_gain;
    decoder.postfilter_tapset_old = decoder.postfilter_tapset;
    decoder.postfilter_period = postfilter_pitch;
    decoder.postfilter_gain = postfilter_gain;
    decoder.postfilter_tapset = postfilter_tapset;
    if lm != 0 {
        decoder.postfilter_period_old = decoder.postfilter_period;
        decoder.postfilter_gain_old = decoder.postfilter_gain;
        decoder.postfilter_tapset_old = decoder.postfilter_tapset;
    }

    if c == 1 {
        let (left, right) = decoder.old_ebands.split_at_mut(nb_ebands);
        right.copy_from_slice(left);
    }

    if is_transient {
        for (log_e, band_e) in decoder.old_log_e.iter_mut().zip(decoder.old_ebands.iter()) {
            *log_e = (*log_e).min(*band_e);
        }
    } else {
        decoder.old_log_e2.copy_from_slice(decoder.old_log_e);
        decoder.old_log_e.copy_from_slice(decoder.old_ebands);
    }

    let increase = ((decoder.loss_duration + m as i32).min(160) as f32) * 0.001;
    for (background, band_e) in decoder
        .background_log_e
        .iter_mut()
        .zip(decoder.old_ebands.iter())
    {
        *background = (*background + increase).min(*band_e);
    }

    for ch in 0..2 {
        let base = ch * nb_ebands;
        for band in 0..start {
            let idx = base + band;
            decoder.old_ebands[idx] = 0.0;
            decoder.old_log_e[idx] = -28.0;
            decoder.old_log_e2[idx] = -28.0;
        }
        for band in end..nb_ebands {
            let idx = base + band;
            decoder.old_ebands[idx] = 0.0;
            decoder.old_log_e[idx] = -28.0;
            decoder.old_log_e2[idx] = -28.0;
        }
    }

    decoder.rng = dec.ctx().rng;

    // TODO: The temporary vectors in this decode path mirror the C implementation's
    // scratch allocations. Reuse decoder-owned scratch storage once functional
    // parity is fully established to avoid repeated heap allocations on hot paths.
    let mut deemph_inputs: Vec<&[CeltSig]> = Vec::with_capacity(cc);
    for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
        let start_idx = DECODE_BUFFER_SIZE - n;
        let (_, rest) = channel_slice.split_at_mut(start_idx);
        let (output, _) = rest.split_at_mut(n);
        deemph_inputs.push(output);
    }

    deemphasis(
        &deemph_inputs,
        pcm,
        n,
        cc,
        downsample,
        &mode.pre_emphasis,
        &mut decoder.preemph_mem_decoder,
        accum,
    );

    decoder.loss_duration = 0;
    decoder.prefilter_and_fold = false;

    if entcode::ec_tell(dec.ctx()) > (packet_len as OpusInt32 * 8) {
        return Err(CeltDecodeError::InvalidPacket);
    }

    if dec.ctx().error() != 0 {
        decoder.error = 1;
    }

    Ok(output_samples)
}

#[cfg(not(feature = "fixed_point"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn celt_decode_with_ec(
    decoder: &mut OpusCustomDecoder<'_>,
    packet: Option<&[u8]>,
    pcm: &mut [OpusRes],
    frame_size: usize,
    range_decoder: Option<&mut EcDec<'_>>,
    accum: bool,
) -> Result<usize, CeltDecodeError> {
    celt_decode_with_ec_dred(decoder, packet, pcm, frame_size, range_decoder, accum)
}

#[cfg(not(feature = "fixed_point"))]
pub(crate) fn opus_custom_decode(
    decoder: &mut OpusCustomDecoder<'_>,
    packet: Option<&[u8]>,
    pcm: &mut [i16],
    frame_size: usize,
) -> Result<usize, CeltDecodeError> {
    let channels = decoder.channels;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(CeltDecodeError::BadArgument)?;
    if pcm.len() < required {
        return Err(CeltDecodeError::BadArgument);
    }

    let mut temp = vec![0.0f32; required];
    let samples = celt_decode_with_ec(decoder, packet, &mut temp, frame_size, None, false)?;
    for (dst, &src) in pcm.iter_mut().zip(temp.iter().take(samples * channels)) {
        *dst = float2int16(src);
    }
    Ok(samples)
}

#[cfg(not(feature = "fixed_point"))]
pub(crate) fn opus_custom_decode24(
    decoder: &mut OpusCustomDecoder<'_>,
    packet: Option<&[u8]>,
    pcm: &mut [i32],
    frame_size: usize,
) -> Result<usize, CeltDecodeError> {
    let channels = decoder.channels;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(CeltDecodeError::BadArgument)?;
    if pcm.len() < required {
        return Err(CeltDecodeError::BadArgument);
    }

    let mut temp = vec![0.0f32; required];
    let samples = celt_decode_with_ec(decoder, packet, &mut temp, frame_size, None, false)?;
    for (dst, &src) in pcm.iter_mut().zip(temp.iter().take(samples * channels)) {
        *dst = res_to_int24(src);
    }
    Ok(samples)
}

#[cfg(not(feature = "fixed_point"))]
pub(crate) fn opus_custom_decode_float(
    decoder: &mut OpusCustomDecoder<'_>,
    packet: Option<&[u8]>,
    pcm: &mut [f32],
    frame_size: usize,
) -> Result<usize, CeltDecodeError> {
    celt_decode_with_ec(decoder, packet, pcm, frame_size, None, false)
}

/// Errors that can be reported when initialising a CELT decoder instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltDecoderInitError {
    /// Channel count was zero or larger than the supported maximum.
    InvalidChannelCount,
    /// Requested stream channel layout is not compatible with the physical
    /// channels configured for the decoder.
    InvalidStreamChannels,
    /// The provided mode uses a sampling rate that cannot be resampled from the
    /// 48 kHz CELT reference clock.
    UnsupportedSampleRate,
    /// The canonical 48 kHz / 960 sample mode could not be constructed.
    CanonicalModeUnavailable,
}

/// Errors that can be emitted by [`opus_custom_decoder_ctl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltDecoderCtlError {
    /// The provided argument is outside the range accepted by the request.
    InvalidArgument,
    /// The request has not been implemented by the Rust port yet.
    Unimplemented,
}

/// Strongly-typed replacement for the decoder-side varargs CTL dispatcher.
pub(crate) enum DecoderCtlRequest<'dec, 'req> {
    SetComplexity(i32),
    GetComplexity(&'req mut i32),
    SetStartBand(i32),
    SetEndBand(i32),
    SetChannels(usize),
    GetAndClearError(&'req mut i32),
    GetLookahead(&'req mut i32),
    ResetState,
    GetPitch(&'req mut i32),
    GetMode(&'req mut Option<&'dec OpusCustomMode<'dec>>),
    SetSignalling(i32),
    GetFinalRange(&'req mut OpusUint32),
    SetPhaseInversionDisabled(bool),
    GetPhaseInversionDisabled(&'req mut bool),
}

/// Helper owning the trailing buffers that back [`OpusCustomDecoder`].
///
/// The C implementation allocates the decoder struct followed by a number of
/// variable-length arrays.  Keeping the storage separate in Rust avoids unsafe
/// pointer arithmetic and simplifies sharing the buffers across temporary
/// decoder views used during reset or PLC.
#[derive(Debug, Default)]
pub(crate) struct CeltDecoderAlloc {
    decode_mem: Vec<CeltSig>,
    lpc: Vec<OpusVal16>,
    old_ebands: Vec<CeltGlog>,
    old_log_e: Vec<CeltGlog>,
    old_log_e2: Vec<CeltGlog>,
    background_log_e: Vec<CeltGlog>,
}

impl CeltDecoderAlloc {
    /// Creates a new allocation suitable for the provided mode and channel
    /// configuration.
    ///
    /// The decoder requires per-channel history buffers for the overlap region
    /// as well as twice the number of energy bands tracked by the mode.  The
    /// allocations follow the layout of the C implementation while leveraging
    /// Rust's `Vec` to manage the backing storage.
    pub(crate) fn new(mode: &OpusCustomMode<'_>, channels: usize) -> Self {
        assert!(channels > 0, "decoder must contain at least one channel");

        let overlap = mode.overlap;
        let decode_mem = channels * (DECODE_BUFFER_SIZE + overlap);
        let lpc = LPC_ORDER * channels;
        let band_count = 2 * mode.num_ebands;

        Self {
            decode_mem: vec![0.0; decode_mem],
            lpc: vec![0.0; lpc],
            old_ebands: vec![0.0; band_count],
            old_log_e: vec![0.0; band_count],
            old_log_e2: vec![0.0; band_count],
            background_log_e: vec![0.0; band_count],
        }
    }

    /// Returns the total size in bytes consumed by the allocation.
    ///
    /// Mirrors the behaviour of `celt_decoder_get_size()` in spirit by exposing
    /// how much storage is required for the decoder and its trailing buffers.
    /// The actual C helper only depends on the channel count; we include the
    /// mode so the calculation reflects the precise band layout in use.  A
    /// follow-up port of the fixed allocation used by the reference
    /// implementation will replace this helper with a fully bit-exact
    /// translation.
    pub(crate) fn size_in_bytes(&self) -> usize {
        let channels = self.lpc.len() / LPC_ORDER;
        debug_assert!(channels > 0 && channels <= MAX_CHANNELS);

        let decode_mem = self.decode_mem.len();
        debug_assert!(decode_mem >= 1);

        let band_history = self.old_ebands.len();
        debug_assert_eq!(self.old_log_e.len(), band_history);
        debug_assert_eq!(self.old_log_e2.len(), band_history);
        debug_assert_eq!(self.background_log_e.len(), band_history);

        DECODER_PREFIX_SIZE
            + (decode_mem - 1) * core::mem::size_of::<CeltSig>()
            + self.lpc.len() * core::mem::size_of::<OpusVal16>()
            + 4 * band_history * core::mem::size_of::<CeltGlog>()
    }

    /// Borrows the allocation as an [`OpusCustomDecoder`] tied to the provided
    /// mode.
    ///
    /// Each call returns a fresh decoder view referencing the same backing
    /// buffers.  This mirrors the C layout where the state and trailing memory
    /// occupy a single blob, enabling the caller to reset or reuse the decoder
    /// without reallocating.
    pub(crate) fn as_decoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
    ) -> OpusCustomDecoder<'a> {
        OpusCustomDecoder::new(
            mode,
            channels,
            stream_channels,
            self.decode_mem.as_mut_slice(),
            self.lpc.as_mut_slice(),
            self.old_ebands.as_mut_slice(),
            self.old_log_e.as_mut_slice(),
            self.old_log_e2.as_mut_slice(),
            self.background_log_e.as_mut_slice(),
        )
    }

    /// Resets the allocation contents to zero.
    pub(crate) fn reset(&mut self) {
        for sample in &mut self.decode_mem {
            *sample = 0.0;
        }
        for coeff in &mut self.lpc {
            *coeff = 0.0;
        }
        for history in &mut self.old_ebands {
            *history = 0.0;
        }
        for history in &mut self.old_log_e {
            *history = 0.0;
        }
        for history in &mut self.old_log_e2 {
            *history = 0.0;
        }
        for history in &mut self.background_log_e {
            *history = 0.0;
        }
    }

    /// Prepares a decoder view that mirrors the default initialisation state.
    ///
    /// The helper ports the zeroing performed by `opus_custom_decoder_init()`
    /// and the follow-up `OPUS_RESET_STATE` call by validating the channel
    /// layout, borrowing the trailing buffers, and clearing the runtime state.
    fn prepare_decoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
    ) -> Result<OpusCustomDecoder<'a>, CeltDecoderInitError> {
        validate_channel_layout(channels, stream_channels)?;

        let mut decoder = self.as_decoder(mode, channels, stream_channels);
        decoder.reset_runtime_state();
        decoder.downsample = 1;
        decoder.start_band = 0;
        decoder.end_band = mode.effective_ebands as i32;
        decoder.signalling = 1;
        decoder.disable_inv = channels == 1;
        decoder.arch = opus_select_arch();

        Ok(decoder)
    }

    /// Returns a freshly initialised decoder state.
    ///
    /// The helper mirrors `celt_decoder_init()` by validating the channel
    /// configuration, clearing the trailing buffers, and populating the fields
    /// that depend on the current mode and sampling rate.  Callers receive a
    /// fully formed [`OpusCustomDecoder`] that borrows the allocation's backing
    /// storage.
    pub(crate) fn init_decoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
    ) -> Result<OpusCustomDecoder<'a>, CeltDecoderInitError> {
        let mut decoder = self.prepare_decoder(mode, channels, stream_channels)?;

        let downsample = resampling_factor(mode.sample_rate);
        if downsample == 0 {
            return Err(CeltDecoderInitError::UnsupportedSampleRate);
        }

        decoder.downsample = downsample as i32;

        Ok(decoder)
    }
}

/// Initialises a decoder allocation for a custom mode.
///
/// Mirrors `opus_custom_decoder_init()` by validating the channel count,
/// clearing the trailing buffers, and returning a decoder view that reflects the
/// freshly reset state.  The helper leaves the downsampling factor at unity, as
/// the C implementation derives any alternative stride from the caller-provided
/// sampling rate after this routine completes.
pub(crate) fn opus_custom_decoder_init<'a>(
    alloc: &'a mut CeltDecoderAlloc,
    mode: &'a OpusCustomMode<'a>,
    channels: usize,
) -> Result<OpusCustomDecoder<'a>, CeltDecoderInitError> {
    alloc.prepare_decoder(mode, channels, channels)
}

/// Allocates and initialises a decoder for a custom mode.
///
/// Mirrors `opus_custom_decoder_create()` by allocating the trailing buffers,
/// validating the channel layout, and returning an owned wrapper that keeps the
/// decoder state and backing storage alive for the duration of the ported API.
pub(crate) fn opus_custom_decoder_create<'mode>(
    mode: &'mode OpusCustomMode<'mode>,
    channels: usize,
) -> Result<OwnedCeltDecoder<'mode>, CeltDecoderInitError> {
    validate_channel_layout(channels, channels)?;
    let alloc = Box::new(CeltDecoderAlloc::new(mode, channels));
    let ptr = Box::into_raw(alloc);
    // SAFETY: `ptr` originates from `Box::into_raw` and remains valid until it is
    // reconstructed below. The pointer is never null.
    let result = unsafe { (*ptr).prepare_decoder(mode, channels, channels) };
    match result {
        Ok(decoder) => {
            // SAFETY: Rebuilds the `Box` returned by `Box::into_raw` so the
            // allocation is managed by Rust again.
            let alloc = unsafe { Box::from_raw(ptr) };
            Ok(OwnedCeltDecoder { decoder, alloc })
        }
        Err(err) => {
            // SAFETY: The allocation must be reclaimed when initialisation fails
            // to avoid leaking memory.
            unsafe {
                let _ = Box::from_raw(ptr);
            }
            Err(err)
        }
    }
}

/// Releases the resources owned by [`OwnedCeltDecoder`].
///
/// The wrapper owns the trailing buffers through its boxed allocation, so
/// consuming it mirrors the behaviour of `opus_custom_decoder_destroy()` in C.
/// Dropping the wrapper performs all necessary cleanup, so this helper is a
/// no-op that exists for API parity.
#[inline]
pub(crate) fn opus_custom_decoder_destroy(_decoder: OwnedCeltDecoder<'_>) {}

fn validate_channel_layout(
    channels: usize,
    stream_channels: usize,
) -> Result<(), CeltDecoderInitError> {
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(CeltDecoderInitError::InvalidChannelCount);
    }
    if stream_channels == 0 || stream_channels > channels {
        return Err(CeltDecoderInitError::InvalidStreamChannels);
    }
    Ok(())
}

/// Applies a control request to the provided decoder state.
pub(crate) fn opus_custom_decoder_ctl<'dec, 'req>(
    decoder: &mut OpusCustomDecoder<'dec>,
    request: DecoderCtlRequest<'dec, 'req>,
) -> Result<(), CeltDecoderCtlError> {
    match request {
        DecoderCtlRequest::SetComplexity(value) => {
            if !(0..=10).contains(&value) {
                return Err(CeltDecoderCtlError::InvalidArgument);
            }
            decoder.complexity = value;
        }
        DecoderCtlRequest::GetComplexity(slot) => {
            *slot = decoder.complexity;
        }
        DecoderCtlRequest::SetStartBand(value) => {
            let max = decoder.mode.num_ebands as i32;
            if value < 0 || value >= max {
                return Err(CeltDecoderCtlError::InvalidArgument);
            }
            decoder.start_band = value;
        }
        DecoderCtlRequest::SetEndBand(value) => {
            let max = decoder.mode.num_ebands as i32;
            if value < 1 || value > max {
                return Err(CeltDecoderCtlError::InvalidArgument);
            }
            decoder.end_band = value;
        }
        DecoderCtlRequest::SetChannels(value) => {
            if value == 0 || value > decoder.channels {
                return Err(CeltDecoderCtlError::InvalidArgument);
            }
            decoder.stream_channels = value;
        }
        DecoderCtlRequest::GetAndClearError(slot) => {
            *slot = decoder.error;
            decoder.error = 0;
        }
        DecoderCtlRequest::GetLookahead(slot) => {
            let downsample = decoder.downsample;
            if downsample <= 0 {
                return Err(CeltDecoderCtlError::InvalidArgument);
            }
            *slot = (decoder.overlap as i32) / downsample;
        }
        DecoderCtlRequest::ResetState => {
            decoder.reset_runtime_state();
        }
        DecoderCtlRequest::GetPitch(slot) => {
            *slot = decoder.postfilter_period;
        }
        DecoderCtlRequest::GetMode(slot) => {
            *slot = Some(decoder.mode);
        }
        DecoderCtlRequest::SetSignalling(value) => {
            decoder.signalling = value;
        }
        DecoderCtlRequest::GetFinalRange(slot) => {
            *slot = decoder.rng;
        }
        DecoderCtlRequest::SetPhaseInversionDisabled(value) => {
            decoder.disable_inv = value;
        }
        DecoderCtlRequest::GetPhaseInversionDisabled(slot) => {
            *slot = decoder.disable_inv;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "fixed_point"))]
    use super::celt_decode_lost;
    #[cfg(not(feature = "fixed_point"))]
    use super::celt_plc_pitch_search;
    #[cfg(not(feature = "fixed_point"))]
    use super::deemphasis;
    use super::{
        CeltDecodeError, CeltDecoderAlloc, CeltDecoderCtlError, CeltDecoderInitError,
        DECODE_BUFFER_SIZE, DecoderCtlRequest, LPC_ORDER, MAX_CHANNELS, RangeDecoderState,
        celt_decoder_get_size, celt_decoder_init, comb_filter, opus_custom_decoder_create,
        opus_custom_decoder_ctl, opus_custom_decoder_get_size, opus_custom_decoder_init,
        prefilter_and_fold, prepare_frame, tf_decode, validate_celt_decoder,
        validate_channel_layout,
    };
    #[cfg(not(feature = "fixed_point"))]
    use crate::celt::float_cast::CELT_SIG_SCALE;
    use crate::celt::modes::opus_custom_mode_create;
    use crate::celt::opus_select_arch;
    use crate::celt::types::{MdctLookup, OpusCustomMode, PulseCacheData};
    use alloc::vec;
    use alloc::vec::Vec;
    #[cfg(not(feature = "fixed_point"))]
    use core::f32::consts::PI;
    use core::ptr;

    #[test]
    fn tf_decode_returns_default_table_entries() {
        let mut range = RangeDecoderState::new(&[0u8]);
        let mut tf_res = vec![0; 4];
        tf_decode(0, tf_res.len(), false, &mut tf_res, 0, range.decoder());
        assert!(tf_res.iter().all(|&v| v == 0));
    }

    #[test]
    #[cfg(not(feature = "fixed_point"))]
    fn celt_plc_pitch_search_detects_mono_period() {
        let target_period = 320i32;
        let mut channel = vec![0.0; super::DECODE_BUFFER_SIZE];
        for (i, sample) in channel.iter_mut().enumerate() {
            let phase = 2.0 * PI * (i as f32) / target_period as f32;
            *sample = phase.sin();
        }

        let decode_mem = [&channel[..]];
        let pitch = celt_plc_pitch_search(&decode_mem, decode_mem.len(), 0);
        assert!((pitch - target_period).abs() <= 2);
    }

    #[test]
    #[cfg(not(feature = "fixed_point"))]
    fn celt_plc_pitch_search_handles_stereo_average() {
        let target_period = 480i32;
        let mut left = vec![0.0; super::DECODE_BUFFER_SIZE];
        let mut right = vec![0.0; super::DECODE_BUFFER_SIZE];
        for i in 0..super::DECODE_BUFFER_SIZE {
            let phase = 2.0 * PI * (i as f32) / target_period as f32;
            left[i] = phase.sin();
            right[i] = (phase + PI / 3.0).sin();
        }

        let decode_mem = [&left[..], &right[..]];
        let pitch = celt_plc_pitch_search(&decode_mem, decode_mem.len(), 0);
        assert!((pitch - target_period).abs() <= 4);
    }

    #[test]
    fn allocates_expected_band_buffers() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 2);
        assert_eq!(
            alloc.decode_mem.len(),
            2 * (super::DECODE_BUFFER_SIZE + mode.overlap)
        );
        assert_eq!(alloc.lpc.len(), LPC_ORDER * 2);
        assert_eq!(alloc.old_ebands.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e2.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.background_log_e.len(), 2 * mode.num_ebands);

        // Ensure the reset helper clears all buffers.
        alloc.decode_mem.fill(1.0);
        alloc.lpc.fill(1.0);
        alloc.old_ebands.fill(1.0);
        alloc.old_log_e.fill(1.0);
        alloc.old_log_e2.fill(1.0);
        alloc.background_log_e.fill(1.0);
        alloc.reset();

        assert!(alloc.decode_mem.iter().all(|&v| v == 0.0));
        assert!(alloc.lpc.iter().all(|&v| v == 0.0));
        assert!(alloc.old_ebands.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e2.iter().all(|&v| v == 0.0));
        assert!(alloc.background_log_e.iter().all(|&v| v == 0.0));

        let expected_size = opus_custom_decoder_get_size(&mode, 2).expect("decoder size");
        assert_eq!(alloc.size_in_bytes(), expected_size);
    }

    #[test]
    fn celt_decoder_get_size_honours_channel_limits() {
        assert!(celt_decoder_get_size(0).is_none());
        assert!(celt_decoder_get_size(3).is_none());
        assert!(celt_decoder_get_size(1).is_some());
        assert!(celt_decoder_get_size(2).is_some());
    }

    #[test]
    fn celt_decoder_init_sets_downsampling_factor() {
        let mode = super::canonical_mode().expect("canonical mode");
        let mut alloc = CeltDecoderAlloc::new(mode, 1);
        let decoder = celt_decoder_init(&mut alloc, 16_000, 1).expect("decoder");
        assert_eq!(decoder.downsample, 3);
    }

    #[test]
    fn celt_decoder_init_rejects_unsupported_rate() {
        let mode = super::canonical_mode().expect("canonical mode");
        let mut alloc = CeltDecoderAlloc::new(mode, 1);
        let err = celt_decoder_init(&mut alloc, 44_100, 1).unwrap_err();
        assert_eq!(err, CeltDecoderInitError::UnsupportedSampleRate);
    }

    #[test]
    fn opus_custom_decoder_create_initialises_state() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();

        let decoder = opus_custom_decoder_create(&mode, 2).expect("decoder");

        assert_eq!(decoder.channels, 2);
        assert_eq!(decoder.stream_channels, 2);
        assert_eq!(decoder.downsample, 1);
        assert_eq!(decoder.start_band, 0);
        assert_eq!(decoder.end_band, mode.effective_ebands as i32);
        assert_eq!(decoder.signalling, 1);
        assert!(!decoder.disable_inv);
        assert_eq!(decoder.arch, opus_select_arch());
    }

    #[test]
    fn opus_custom_decoder_create_rejects_invalid_channel_layouts() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();

        assert!(opus_custom_decoder_create(&mode, 0).is_err());
        assert!(opus_custom_decoder_create(&mode, 3).is_err());
    }

    #[test]
    fn validate_celt_decoder_accepts_default_configuration() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let decoder = alloc.as_decoder(&mode, 1, 1);

        validate_celt_decoder(&decoder);
    }

    #[test]
    #[should_panic]
    fn validate_celt_decoder_rejects_invalid_channel_count() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc.as_decoder(&mode, 1, 1);
        decoder.channels = 3;

        validate_celt_decoder(&decoder);
    }

    #[test]
    fn validate_channel_layout_rejects_invalid_configurations() {
        assert_eq!(
            validate_channel_layout(0, 0),
            Err(CeltDecoderInitError::InvalidChannelCount)
        );
        assert_eq!(
            validate_channel_layout(MAX_CHANNELS + 1, 1),
            Err(CeltDecoderInitError::InvalidChannelCount)
        );
        assert_eq!(
            validate_channel_layout(1, 0),
            Err(CeltDecoderInitError::InvalidStreamChannels)
        );
        assert_eq!(
            validate_channel_layout(1, 2),
            Err(CeltDecoderInitError::InvalidStreamChannels)
        );
    }

    #[test]
    fn init_decoder_populates_expected_defaults() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let decoder = alloc
            .init_decoder(&mode, 1, 1)
            .expect("initialisation should succeed");

        assert_eq!(decoder.overlap, mode.overlap);
        assert_eq!(decoder.downsample, 1);
        assert_eq!(decoder.end_band, mode.effective_ebands as i32);
        assert_eq!(decoder.arch, 0);
        assert_eq!(decoder.rng, 0);
        assert_eq!(decoder.loss_duration, 0);
        assert_eq!(decoder.postfilter_period, 0);
        assert_eq!(decoder.postfilter_gain, 0.0);
        assert_eq!(decoder.postfilter_tapset, 0);
        assert!(decoder.skip_plc);
    }

    #[test]
    fn opus_custom_decoder_init_matches_reference_defaults() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            12_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let decoder = opus_custom_decoder_init(&mut alloc, &mode, 1)
            .expect("custom decoder initialisation should succeed");

        assert_eq!(decoder.downsample, 1);
        assert_eq!(decoder.start_band, 0);
        assert_eq!(decoder.end_band, mode.effective_ebands as i32);
        assert_eq!(decoder.signalling, 1);
        assert!(decoder.disable_inv);
        assert!(decoder.skip_plc);
        assert_eq!(decoder.arch, 0);
    }

    #[test]
    fn decoder_ctl_handles_configuration_requests() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 2);
        let mut decoder = alloc
            .init_decoder(&mode, 2, 2)
            .expect("decoder initialisation should succeed");

        let err = opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetComplexity(11))
            .unwrap_err();
        assert_eq!(err, CeltDecoderCtlError::InvalidArgument);
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetComplexity(7)).unwrap();
        let mut complexity = 0;
        opus_custom_decoder_ctl(
            &mut decoder,
            DecoderCtlRequest::GetComplexity(&mut complexity),
        )
        .unwrap();
        assert_eq!(complexity, 7);

        let max = mode.num_ebands as i32;
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetStartBand(1)).unwrap();
        assert_eq!(decoder.start_band, 1);
        let err = opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetStartBand(max))
            .unwrap_err();
        assert_eq!(err, CeltDecoderCtlError::InvalidArgument);

        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetEndBand(max)).unwrap();
        assert_eq!(decoder.end_band, max);
        let err =
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetEndBand(0)).unwrap_err();
        assert_eq!(err, CeltDecoderCtlError::InvalidArgument);

        let err =
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(3)).unwrap_err();
        assert_eq!(err, CeltDecoderCtlError::InvalidArgument);
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(1)).unwrap();
        assert_eq!(decoder.stream_channels, 1);
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(2)).unwrap();
        assert_eq!(decoder.stream_channels, 2);

        decoder.error = -57;
        let mut reported_error = 0;
        opus_custom_decoder_ctl(
            &mut decoder,
            DecoderCtlRequest::GetAndClearError(&mut reported_error),
        )
        .unwrap();
        assert_eq!(reported_error, -57);
        assert_eq!(decoder.error, 0);

        decoder.overlap = 6;
        decoder.downsample = 2;
        let mut lookahead = 0;
        opus_custom_decoder_ctl(
            &mut decoder,
            DecoderCtlRequest::GetLookahead(&mut lookahead),
        )
        .unwrap();
        assert_eq!(lookahead, 3);

        decoder.postfilter_period = 321;
        let mut pitch = 0;
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::GetPitch(&mut pitch)).unwrap();
        assert_eq!(pitch, 321);

        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetSignalling(5)).unwrap();
        assert_eq!(decoder.signalling, 5);

        decoder.rng = 0xDEADBEEF;
        let mut rng = 0;
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::GetFinalRange(&mut rng)).unwrap();
        assert_eq!(rng, 0xDEADBEEF);

        opus_custom_decoder_ctl(
            &mut decoder,
            DecoderCtlRequest::SetPhaseInversionDisabled(true),
        )
        .unwrap();
        assert!(decoder.disable_inv);
        let mut disabled = false;
        opus_custom_decoder_ctl(
            &mut decoder,
            DecoderCtlRequest::GetPhaseInversionDisabled(&mut disabled),
        )
        .unwrap();
        assert!(disabled);

        let mut mode_slot = None;
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::GetMode(&mut mode_slot)).unwrap();
        let mode_ref = mode_slot.expect("mode reference");
        assert!(ptr::eq(mode_ref, &mode));
    }

    #[test]
    fn decoder_ctl_reset_state_matches_reference() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc
            .init_decoder(&mode, 1, 1)
            .expect("decoder initialisation should succeed");

        decoder.rng = 1234;
        decoder.error = -1;
        decoder.last_pitch_index = 77;
        decoder.loss_duration = 99;
        decoder.skip_plc = false;
        decoder.postfilter_period = 12;
        decoder.postfilter_period_old = 34;
        decoder.postfilter_gain = 0.5;
        decoder.postfilter_gain_old = 0.25;
        decoder.postfilter_tapset = 2;
        decoder.postfilter_tapset_old = 1;
        decoder.prefilter_and_fold = true;
        decoder.preemph_mem_decoder = [0.1, -0.2];
        decoder.decode_mem.fill(1.0);
        decoder.lpc.fill(0.5);
        decoder.old_ebands.fill(0.75);
        decoder.old_log_e.fill(1.0);
        decoder.old_log_e2.fill(1.5);
        decoder.background_log_e.fill(0.125);

        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::ResetState).unwrap();

        assert_eq!(decoder.rng, 0);
        assert_eq!(decoder.error, 0);
        assert_eq!(decoder.last_pitch_index, 0);
        assert_eq!(decoder.loss_duration, 0);
        assert!(decoder.skip_plc);
        assert_eq!(decoder.postfilter_period, 0);
        assert_eq!(decoder.postfilter_period_old, 0);
        assert_eq!(decoder.postfilter_gain, 0.0);
        assert_eq!(decoder.postfilter_gain_old, 0.0);
        assert_eq!(decoder.postfilter_tapset, 0);
        assert_eq!(decoder.postfilter_tapset_old, 0);
        assert!(!decoder.prefilter_and_fold);
        assert_eq!(decoder.preemph_mem_decoder, [0.0, 0.0]);
        assert!(decoder.decode_mem.iter().all(|&v| v == 0.0));
        assert!(decoder.lpc.iter().all(|&v| v == 0.0));
        assert!(decoder.old_ebands.iter().all(|&v| v == 0.0));
        assert!(decoder.old_log_e.iter().all(|&v| v == -28.0));
        assert!(decoder.old_log_e2.iter().all(|&v| v == -28.0));
        assert!(decoder.background_log_e.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn prepare_frame_handles_packet_loss() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mut mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );
        mode.short_mdct_size = 2;
        mode.max_lm = 2;

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc
            .init_decoder(&mode, 1, 1)
            .expect("decoder initialisation should succeed");

        let frame = prepare_frame(&mut decoder, &[], mode.short_mdct_size)
            .expect("packet loss preparation should succeed");
        assert!(frame.packet_loss);
    }

    #[test]
    fn prepare_frame_rejects_mismatched_frame_size() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mut mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );
        mode.short_mdct_size = 2;
        mode.max_lm = 2;

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc
            .init_decoder(&mode, 1, 1)
            .expect("decoder initialisation should succeed");

        let err = prepare_frame(&mut decoder, &[0u8; 2], mode.short_mdct_size + 1)
            .expect_err("invalid frame size must be rejected");
        assert_eq!(err, CeltDecodeError::BadArgument);
    }

    #[test]
    fn prefilter_and_fold_rebuilds_overlap_tail() {
        let e_bands = [0, 1];
        let alloc_vectors = [0u8; 1];
        let log_n = [0i16; 1];
        let window = [0.25f32, 0.5, 0.5, 0.25];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0i16; 2], vec![0; 2], vec![0; 2]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc
            .init_decoder(&mode, 1, 1)
            .expect("decoder initialisation should succeed");

        let overlap = mode.overlap;
        let stride = DECODE_BUFFER_SIZE + overlap;
        assert_eq!(decoder.decode_mem.len(), stride);

        for (idx, sample) in decoder.decode_mem.iter_mut().enumerate() {
            *sample = idx as f32;
        }

        let n = 32;
        let start = DECODE_BUFFER_SIZE - n;
        let original = decoder.decode_mem[start..start + overlap].to_vec();

        decoder.postfilter_gain = 0.0;
        decoder.postfilter_gain_old = 0.0;
        decoder.postfilter_tapset = 0;
        decoder.postfilter_tapset_old = 0;

        prefilter_and_fold(&mut decoder, n);

        let tolerance = 1e-6f32;
        for i in 0..(overlap / 2) {
            let expected =
                window[i] * original[overlap - 1 - i] + window[overlap - 1 - i] * original[i];
            let actual = decoder.decode_mem[start + i];
            assert!(
                (expected - actual).abs() <= tolerance,
                "folded sample {i} differs: expected {expected}, got {actual}",
            );
        }

        for i in overlap / 2..overlap {
            let idx = start + i;
            assert_eq!(decoder.decode_mem[idx], original[i]);
        }
    }

    #[test]
    fn prefilter_and_fold_filters_overlap_tail_with_gain() {
        let e_bands = [0, 1];
        let alloc_vectors = [0u8; 1];
        let log_n = [0i16; 1];
        let window = [0.25f32, 0.5, 0.5, 0.25];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0i16; 2], vec![0; 2], vec![0; 2]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc
            .init_decoder(&mode, 1, 1)
            .expect("decoder initialisation should succeed");

        let overlap = mode.overlap;
        let stride = DECODE_BUFFER_SIZE + overlap;
        assert_eq!(decoder.decode_mem.len(), stride);

        for (idx, sample) in decoder.decode_mem.iter_mut().enumerate() {
            *sample = (idx as f32) * 0.125;
        }

        let n = 64;
        let start = DECODE_BUFFER_SIZE - n;
        let baseline: Vec<f32> = decoder.decode_mem.iter().copied().collect();
        let original = baseline[start..start + overlap].to_vec();

        decoder.postfilter_gain_old = 0.2;
        decoder.postfilter_gain = 0.35;
        decoder.postfilter_period_old = 24;
        decoder.postfilter_period = 32;
        decoder.postfilter_tapset_old = 0;
        decoder.postfilter_tapset = 1;

        prefilter_and_fold(&mut decoder, n);

        let mut expected_filtered = vec![0.0; overlap];
        comb_filter(
            &mut expected_filtered,
            &baseline,
            start,
            overlap,
            decoder.postfilter_period_old,
            decoder.postfilter_period,
            -decoder.postfilter_gain_old,
            -decoder.postfilter_gain,
            decoder.postfilter_tapset_old as usize,
            decoder.postfilter_tapset as usize,
            &[],
            0,
            decoder.arch,
        );

        let tolerance = 1e-6f32;
        for i in 0..(overlap / 2) {
            let expected = window[i] * expected_filtered[overlap - 1 - i]
                + window[overlap - 1 - i] * expected_filtered[i];
            let actual = decoder.decode_mem[start + i];
            assert!(
                (expected - actual).abs() <= tolerance,
                "filtered fold {i} differs: expected {expected}, got {actual}",
            );
        }

        for i in overlap / 2..overlap {
            let idx = start + i;
            assert_eq!(decoder.decode_mem[idx], original[i]);
        }
    }

    #[test]
    fn prefilter_and_fold_handles_stereo_channels_independently() {
        let e_bands = [0, 1];
        let alloc_vectors = [0u8; 1];
        let log_n = [0i16; 1];
        let window = [0.25f32, 0.5, 0.5, 0.25];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0i16; 2], vec![0; 2], vec![0; 2]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 2);
        let mut decoder = alloc
            .init_decoder(&mode, 2, 2)
            .expect("decoder initialisation should succeed");

        let overlap = mode.overlap;
        let stride = DECODE_BUFFER_SIZE + overlap;
        assert_eq!(decoder.decode_mem.len(), 2 * stride);

        for i in 0..stride {
            decoder.decode_mem[i] = (i as f32) * 0.25;
            decoder.decode_mem[stride + i] = 500.0 + (i as f32) * 0.5;
        }

        let n = 48;
        let start = DECODE_BUFFER_SIZE - n;
        let baseline: Vec<f32> = decoder.decode_mem.iter().copied().collect();

        decoder.postfilter_gain_old = 0.1;
        decoder.postfilter_gain = 0.3;
        decoder.postfilter_period_old = 20;
        decoder.postfilter_period = 28;
        decoder.postfilter_tapset_old = 2;
        decoder.postfilter_tapset = 1;

        prefilter_and_fold(&mut decoder, n);

        let tolerance = 1e-6f32;
        for channel in 0..2 {
            let offset = channel * stride;
            let original = &baseline[offset + start..offset + start + overlap];
            let mut expected_filtered = vec![0.0; overlap];
            comb_filter(
                &mut expected_filtered,
                &baseline[offset..offset + stride],
                start,
                overlap,
                decoder.postfilter_period_old,
                decoder.postfilter_period,
                -decoder.postfilter_gain_old,
                -decoder.postfilter_gain,
                decoder.postfilter_tapset_old as usize,
                decoder.postfilter_tapset as usize,
                &[],
                0,
                decoder.arch,
            );

            for i in 0..(overlap / 2) {
                let expected = window[i] * expected_filtered[overlap - 1 - i]
                    + window[overlap - 1 - i] * expected_filtered[i];
                let actual = decoder.decode_mem[offset + start + i];
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "channel {channel} fold {i} differs: expected {expected}, got {actual}",
                );
            }

            for i in overlap / 2..overlap {
                let idx = offset + start + i;
                assert_eq!(decoder.decode_mem[idx], original[i]);
            }
        }
    }

    #[cfg(not(feature = "fixed_point"))]
    #[test]
    fn celt_decode_lost_noise_branch_updates_state() {
        use crate::celt::modes::opus_custom_mode_find_static;

        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc.as_decoder(&mode, 1, 1);
        decoder.downsample = 1;
        decoder.end_band = mode.num_ebands as i32;
        decoder.loss_duration = 0;
        decoder.skip_plc = true;
        decoder.prefilter_and_fold = true;
        decoder.rng = 0x1234_5678;
        decoder.background_log_e.fill(-8.0);
        decoder.old_ebands.fill(-4.0);

        let n = mode.short_mdct_size;
        celt_decode_lost(&mut decoder, n, 0);

        assert!(decoder.skip_plc);
        assert!(!decoder.prefilter_and_fold);
        assert_ne!(decoder.rng, 0x1234_5678);
        assert_eq!(decoder.loss_duration, 1);
    }

    #[cfg(not(feature = "fixed_point"))]
    #[test]
    fn celt_decode_lost_pitch_branch_generates_output() {
        use crate::celt::modes::opus_custom_mode_find_static;

        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc.as_decoder(&mode, 1, 1);
        decoder.downsample = 1;
        decoder.end_band = mode.num_ebands as i32;
        decoder.loss_duration = 2;
        decoder.skip_plc = false;
        decoder.last_pitch_index = 200;
        decoder.prefilter_and_fold = false;

        let _stride = DECODE_BUFFER_SIZE + mode.overlap;
        for (idx, sample) in decoder.decode_mem.iter_mut().enumerate() {
            *sample = (idx % 23) as f32 * 0.001;
        }

        let n = mode.short_mdct_size;
        celt_decode_lost(&mut decoder, n, 0);

        assert!(decoder.prefilter_and_fold);
        assert!(!decoder.skip_plc);
        assert_eq!(decoder.loss_duration, 3);

        let start = DECODE_BUFFER_SIZE - n;
        let produced = &decoder.decode_mem[start..start + n];
        let energy: f32 = produced.iter().map(|v| v.abs()).sum();
        assert!(energy > 0.0);
    }

    #[cfg(not(feature = "fixed_point"))]
    #[test]
    fn deemphasis_stereo_simple_matches_reference() {
        let left = [0.25_f32, -0.5, 0.75];
        let right = [-0.125_f32, 0.5, -0.25];
        let input: [&[f32]; 2] = [&left, &right];
        let mut pcm = vec![0.0_f32; left.len() * 2];
        let mut mem = [0.1_f32, -0.2_f32];
        let coef = [0.5_f32];

        deemphasis(&input, &mut pcm, left.len(), 2, 1, &coef, &mut mem, false);

        const VERY_SMALL: f32 = 1.0e-30;
        let mut expected_mem = [0.1_f32, -0.2_f32];
        let mut expected = [Vec::new(), Vec::new()];

        for (channel, samples) in [left.as_slice(), right.as_slice()].iter().enumerate() {
            let mut m = expected_mem[channel];
            for &sample in *samples {
                let tmp = sample + m + VERY_SMALL;
                expected[channel].push(tmp);
                m = coef[0] * tmp;
            }
            expected_mem[channel] = m;
        }

        for j in 0..left.len() {
            assert!((pcm[2 * j] - expected[0][j] / CELT_SIG_SCALE).abs() < 1e-6);
            assert!((pcm[2 * j + 1] - expected[1][j] / CELT_SIG_SCALE).abs() < 1e-6);
        }

        assert!((mem[0] - expected_mem[0]).abs() < 1e-6);
        assert!((mem[1] - expected_mem[1]).abs() < 1e-6);
    }

    #[cfg(not(feature = "fixed_point"))]
    #[test]
    fn deemphasis_downsamples_with_accumulation() {
        let samples = [0.5_f32, -0.25, 0.75, -0.5];
        let input: [&[f32]; 1] = [&samples];
        let mut pcm = vec![0.1_f32, -0.2_f32];
        let mut mem = [0.0_f32];
        let coef = [0.25_f32];

        deemphasis(&input, &mut pcm, samples.len(), 1, 2, &coef, &mut mem, true);

        const VERY_SMALL: f32 = 1.0e-30;
        let mut m = 0.0_f32;
        let mut scratch = Vec::new();
        for &sample in &samples {
            let tmp = sample + m + VERY_SMALL;
            scratch.push(tmp);
            m = coef[0] * tmp;
        }

        let expected = [
            0.1_f32 + scratch[0] / CELT_SIG_SCALE,
            -0.2_f32 + scratch[2] / CELT_SIG_SCALE,
        ];

        for (actual, expected) in pcm.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        assert!((mem[0] - m).abs() < 1e-6);
    }
}
