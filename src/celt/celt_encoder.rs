#![allow(dead_code)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]

//! Encoder scaffolding ported from `celt/celt_encoder.c`.
//!
//! The reference implementation stores the primary encoder state followed by a
//! number of variable-length buffers.  This module mirrors the allocation
//! strategy so that higher level encoding routines can be translated
//! incrementally while keeping the memory layout compatible with the C code.
//! Future patches will extend this file with the analysis, bit allocation, and
//! entropy coding paths that still live in the C sources.

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::bands::{compute_band_energies, haar1};
use crate::celt::celt::{COMBFILTER_MINPERIOD, TF_SELECT_TABLE, comb_filter, resampling_factor};
use crate::celt::cpu_support::opus_select_arch;
use crate::celt::entcode::{BITRES, ec_tell};
use crate::celt::entenc::EcEnc;
use crate::celt::float_cast::{CELT_SIG_SCALE, float2int16};
#[cfg(feature = "fixed_point")]
use crate::celt::math::celt_ilog2;
use crate::celt::math::{celt_exp2, celt_log2, celt_sqrt, frac_div32_q29};
#[cfg(feature = "fixed_point")]
use crate::celt::math_fixed::celt_sqrt as celt_sqrt_fixed;
use crate::celt::mdct::clt_mdct_forward;
use crate::celt::modes::opus_custom_mode_find_static;
use crate::celt::pitch::{celt_inner_prod, pitch_downsample, pitch_search, remove_doubling};
use crate::celt::quant_bands::{E_MEANS, amp2_log2};
use crate::celt::types::{
    AnalysisInfo, CeltGlog, CeltNorm, CeltSig, OpusCustomEncoder, OpusCustomMode, OpusInt16,
    OpusInt32, OpusRes, OpusUint32, OpusVal16, OpusVal32, SilkInfo,
};
use alloc::boxed::Box;
use core::cmp::{max, min};
use core::f32::consts::FRAC_1_SQRT_2;
use core::ptr::NonNull;
#[cfg(not(feature = "fixed_point"))]
use libm::acosf;
use libm::floorf;

/// Maximum number of channels supported by the scalar encoder path.
const MAX_CHANNELS: usize = 2;

/// Size of the comb-filter history kept per channel by the encoder prefilter.
const COMBFILTER_MAXPERIOD: usize = 1024;

/// Maximum number of energy bands handled during the time/frequency analysis.
const MAX_TF_BANDS: usize = 50;
/// Upper bound on the number of coefficients examined per band during TF analysis.
///
/// The C reference rejects modes where the widest band, scaled by the maximum LM,
/// exceeds 208 coefficients. Using the same limit lets us pre-allocate the
/// temporary buffers on the stack.
const MAX_TF_BAND_SIZE: usize = 208;

/// Number of bands that participate in the leak boost analysis.
const LEAK_BANDS: usize = 19;

/// Maximum amplitude allowed when clipping the pre-emphasised input.
const PREEMPHASIS_CLIP_LIMIT: CeltSig = 65_536.0;

/// Special bitrate value used by Opus to request the maximum possible rate.
const OPUS_BITRATE_MAX: OpusInt32 = -1;

/// Errors that can be reported while initialising a CELT encoder instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltEncoderInitError {
    /// The requested number of channels exceeds the supported range.
    InvalidChannelCount,
    /// The number of coded stream channels is inconsistent with the layout.
    InvalidStreamChannels,
    /// The chosen sampling rate cannot be derived from the 48 kHz reference.
    UnsupportedSampleRate,
}

/// Errors that can be emitted by [`opus_custom_encoder_ctl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltEncoderCtlError {
    /// The provided argument is outside the range accepted by the request.
    InvalidArgument,
    /// The request has not been implemented by the Rust port yet.
    Unimplemented,
}

/// Errors that can arise while encoding a frame with [`celt_encode_with_ec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltEncodeError {
    /// The caller did not supply enough PCM samples for the configured layout.
    InsufficientPcm,
    /// The provided frame size is not compatible with the encoder mode.
    InvalidFrameSize,
    /// No output buffer or range encoder was supplied.
    MissingOutput,
}

/// Strongly-typed replacement for the varargs CTL dispatcher used by the C implementation.
#[allow(clippy::large_enum_variant)]
pub(crate) enum EncoderCtlRequest<'enc, 'req> {
    SetComplexity(i32),
    SetStartBand(i32),
    SetEndBand(i32),
    SetPrediction(i32),
    SetPacketLossPerc(i32),
    SetVbrConstraint(bool),
    SetVbr(bool),
    SetBitrate(OpusInt32),
    SetChannels(usize),
    SetLsbDepth(i32),
    GetLsbDepth(&'req mut i32),
    SetPhaseInversionDisabled(bool),
    GetPhaseInversionDisabled(&'req mut bool),
    ResetState,
    SetInputClipping(bool),
    SetSignalling(i32),
    SetAnalysis(&'enc AnalysisInfo),
    SetSilkInfo(&'enc SilkInfo),
    GetMode(&'req mut Option<&'enc OpusCustomMode<'enc>>),
    GetFinalRange(&'req mut OpusUint32),
    SetLfe(bool),
    SetEnergyMask(Option<&'enc [CeltGlog]>),
}

/// Returns the number of bytes required to allocate an encoder for `mode`.
#[must_use]
pub(crate) fn opus_custom_encoder_get_size(mode: &OpusCustomMode<'_>, channels: usize) -> usize {
    let in_mem = channels * mode.overlap;
    let prefilter_mem = channels * COMBFILTER_MAXPERIOD;
    let band_count = channels * mode.num_ebands;

    in_mem * core::mem::size_of::<CeltSig>()
        + prefilter_mem * core::mem::size_of::<CeltSig>()
        + 4 * band_count * core::mem::size_of::<CeltGlog>()
}

/// Returns the size of the canonical CELT encoder operating at 48 kHz/960.
#[must_use]
pub(crate) fn celt_encoder_get_size(channels: usize) -> Option<usize> {
    opus_custom_mode_find_static(48_000, 960)
        .map(|mode| opus_custom_encoder_get_size(&mode, channels))
}

/// Mirrors `opus_custom_encoder_init_arch()` from the reference encoder.
pub(crate) fn opus_custom_encoder_init_arch<'a>(
    alloc: &'a mut CeltEncoderAlloc,
    mode: &'a OpusCustomMode<'a>,
    channels: usize,
    arch: i32,
    rng_seed: OpusUint32,
) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
    alloc.init_custom_encoder_with_arch(mode, channels, channels, arch, rng_seed)
}

/// Mirrors `opus_custom_encoder_init()` by selecting the runtime architecture automatically.
pub(crate) fn opus_custom_encoder_init<'a>(
    alloc: &'a mut CeltEncoderAlloc,
    mode: &'a OpusCustomMode<'a>,
    channels: usize,
    rng_seed: OpusUint32,
) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
    alloc.init_custom_encoder(mode, channels, channels, rng_seed)
}

/// Mirrors `celt_encoder_init()` by initialising the canonical Opus encoder mode.
pub(crate) fn celt_encoder_init<'a>(
    alloc: &'a mut CeltEncoderAlloc,
    sampling_rate: OpusInt32,
    channels: usize,
    arch: i32,
    rng_seed: OpusUint32,
) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
    let upsample = resampling_factor(sampling_rate);
    if upsample == 0 {
        return Err(CeltEncoderInitError::UnsupportedSampleRate);
    }
    if alloc.static_mode.is_none() {
        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let boxed = Box::new(mode);
        let ptr = Box::into_raw(boxed);
        // SAFETY: `Box::into_raw` never returns null.
        alloc.static_mode = Some(unsafe { NonNull::new_unchecked(ptr) });
    }
    // SAFETY: `static_mode` stores a pointer obtained from `Box::into_raw` and remains
    // valid for the lifetime of `alloc`. The reference is coerced to match the borrow
    // used by the encoder initialisation.
    let mode_ref: &OpusCustomMode<'static> = unsafe {
        alloc
            .static_mode
            .expect("static mode is stored for canonical init")
            .as_ref()
    };
    let mut encoder =
        alloc.init_custom_encoder_with_arch(mode_ref, channels, channels, arch, rng_seed)?;
    encoder.upsample = upsample as i32;
    Ok(encoder)
}

/// Mirrors `opus_custom_encoder_destroy()` which simply releases the encoder allocation.
pub(crate) fn opus_custom_encoder_destroy(_alloc: CeltEncoderAlloc) {}

impl Drop for CeltEncoderAlloc {
    fn drop(&mut self) {
        if let Some(ptr) = self.static_mode.take() {
            // SAFETY: `static_mode` was initialised from `Box::into_raw` and is only
            // reconstructed here when the allocation is dropped.
            unsafe {
                let _ = Box::from_raw(ptr.as_ptr());
            }
        }
    }
}

/// Computes the L1 norm used by the time/frequency resolution heuristics.
///
/// Mirrors the helper of the same name in `celt/celt_encoder.c`. The function
/// sums the absolute values in `tmp[..n]` and applies the bias term that favors
/// finer frequency resolution when the MDCT has been split into shorter
/// windows.
fn l1_metric(tmp: &[OpusVal16], n: usize, lm: i32, bias: OpusVal16) -> OpusVal32 {
    assert!(n <= tmp.len());

    let mut l1: OpusVal32 = 0.0;
    for &value in &tmp[..n] {
        l1 += value.abs() as OpusVal32;
    }

    let freq_bias = (lm as OpusVal32) * bias as OpusVal32;
    l1 + freq_bias * l1
}

/// Mirrors the stereo mode decision helper from `celt/celt_encoder.c`.
///
/// The function measures how well a stereo pair can be represented using
/// mid/side coding by comparing the L/R and M/S L1 norms across the first 13
/// bands. The reference implementation returns a non-zero integer when the
/// entropy of the mid/side representation is lower; we translate that into a
/// boolean result for the Rust port.
fn stereo_analysis(mode: &OpusCustomMode<'_>, x: &[CeltNorm], lm: usize, n0: usize) -> bool {
    const EPSILON: f32 = 1.0e-15;

    debug_assert!(
        mode.num_ebands >= 14,
        "stereo analysis expects at least 14 bands"
    );
    debug_assert!(
        x.len() >= 2 * n0,
        "stereo analysis requires two channel buffers"
    );

    let mut sum_lr = EPSILON;
    let mut sum_ms = EPSILON;

    for band in 0..13 {
        let start = (mode.e_bands[band] as usize) << lm;
        let end = (mode.e_bands[band + 1] as usize) << lm;
        if end <= start || end > n0 {
            continue;
        }

        for idx in start..end {
            let left = x[idx];
            let right = x[n0 + idx];
            let mid = left + right;
            let side = left - right;
            sum_lr += left.abs() + right.abs();
            sum_ms += mid.abs() + side.abs();
        }
    }

    sum_ms *= FRAC_1_SQRT_2;
    let mut thetas = 13i32;
    if lm <= 1 {
        thetas -= 8;
    }

    let base = i32::from(mode.e_bands[13]) << (lm + 1);
    let lhs = (base + thetas) as f32 * sum_ms;
    let rhs = base as f32 * sum_lr;
    lhs > rhs
}

#[allow(clippy::too_many_arguments)]
fn tf_analysis(
    mode: &OpusCustomMode<'_>,
    len: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[CeltNorm],
    n0: usize,
    lm: usize,
    tf_estimate: OpusVal16,
    tf_chan: usize,
    importance: &[i32],
) -> i32 {
    debug_assert!(lm < TF_SELECT_TABLE.len());
    debug_assert!(len <= tf_res.len());
    debug_assert!(len <= importance.len());
    debug_assert!(len < mode.e_bands.len());

    if len == 0 {
        return 0;
    }

    let bias = 0.04 * (0.5 - tf_estimate).max(-0.25);

    let mut max_band = 0usize;
    for band in 0..len {
        let start = mode.e_bands[band] as usize;
        let end = mode.e_bands[band + 1] as usize;
        let width = end.saturating_sub(start);
        max_band = max(max_band, width << lm);
    }

    debug_assert!(len <= MAX_TF_BANDS);
    debug_assert!(max_band <= MAX_TF_BAND_SIZE);

    let mut metric_storage = [0i32; MAX_TF_BANDS];
    let mut path0_storage = [0i32; MAX_TF_BANDS];
    let mut path1_storage = [0i32; MAX_TF_BANDS];
    let mut tmp_storage = [0.0f32; MAX_TF_BAND_SIZE];
    let mut tmp_alt_storage = [0.0f32; MAX_TF_BAND_SIZE];

    let metric = &mut metric_storage[..len];
    let path0 = &mut path0_storage[..len];
    let path1 = &mut path1_storage[..len];
    let tmp = &mut tmp_storage[..max_band.max(1)];
    let tmp_alt = &mut tmp_alt_storage[..max_band.max(1)];

    let lm_i32 = lm as i32;

    for band in 0..len {
        let start = mode.e_bands[band] as usize;
        let end = mode.e_bands[band + 1] as usize;
        let width = end.saturating_sub(start);
        let n = width << lm;
        if n == 0 {
            continue;
        }

        let offset = tf_chan * n0 + (start << lm);
        debug_assert!(offset + n <= x.len());
        tmp[..n].copy_from_slice(&x[offset..offset + n]);

        let narrow = width == 1;
        let mut best_level = 0i32;
        let mut best_l1 = l1_metric(&tmp[..n], n, if is_transient { lm_i32 } else { 0 }, bias);

        if is_transient && !narrow {
            tmp_alt[..n].copy_from_slice(&tmp[..n]);
            let blocks = n >> lm;
            if blocks > 0 {
                haar1(&mut tmp_alt[..n], blocks, 1 << lm);
                let l1 = l1_metric(&tmp_alt[..n], n, lm_i32 + 1, bias);
                if l1 < best_l1 {
                    best_l1 = l1;
                    best_level = -1;
                }
            }
        }

        let extra = if is_transient || narrow { 0 } else { 1 };
        for k in 0..(lm + extra) {
            let blocks = n >> k;
            if blocks == 0 {
                break;
            }

            haar1(&mut tmp[..n], blocks, 1 << k);
            let b = if is_transient {
                lm_i32 - k as i32 - 1
            } else {
                k as i32 + 1
            };

            let l1 = l1_metric(&tmp[..n], n, b, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k as i32 + 1;
            }
        }

        let mut value = if is_transient {
            2 * best_level
        } else {
            -2 * best_level
        };
        if narrow && (value == 0 || value == -2 * lm_i32) {
            value -= 1;
        }
        metric[band] = value;
    }

    let table = &TF_SELECT_TABLE[lm];
    let base_index = if is_transient { 4 } else { 0 };
    let mut selcost = [0i32; 2];

    for sel in 0..2 {
        let idx0 = base_index + 2 * sel;
        let idx1 = idx0 + 1;
        let target0 = 2 * i32::from(table[idx0]);
        let target1 = 2 * i32::from(table[idx1]);

        let mut cost0 = importance[0] * (metric[0] - target0).abs();
        let mut cost1 = importance[0] * (metric[0] - target1).abs();
        if !is_transient {
            cost1 += lambda;
        }

        for band in 1..len {
            let from0 = cost0;
            let from1 = cost1 + lambda;
            let curr0;
            if from0 < from1 {
                curr0 = from0;
                path0[band] = 0;
            } else {
                curr0 = from1;
                path0[band] = 1;
            }

            let from0 = cost0 + lambda;
            let from1 = cost1;
            let curr1;
            if from0 < from1 {
                curr1 = from0;
                path1[band] = 0;
            } else {
                curr1 = from1;
                path1[band] = 1;
            }

            cost0 = curr0 + importance[band] * (metric[band] - target0).abs();
            cost1 = curr1 + importance[band] * (metric[band] - target1).abs();
        }

        selcost[sel] = cost0.min(cost1);
    }

    let mut tf_select = 0i32;
    if is_transient && selcost[1] < selcost[0] {
        tf_select = 1;
    }

    let idx0 = base_index + 2 * tf_select as usize;
    let idx1 = idx0 + 1;
    let target0 = 2 * i32::from(table[idx0]);
    let target1 = 2 * i32::from(table[idx1]);

    let mut cost0 = importance[0] * (metric[0] - target0).abs();
    let mut cost1 = importance[0] * (metric[0] - target1).abs();
    if !is_transient {
        cost1 += lambda;
    }

    for band in 1..len {
        let from0 = cost0;
        let from1 = cost1 + lambda;
        let curr0;
        if from0 < from1 {
            curr0 = from0;
            path0[band] = 0;
        } else {
            curr0 = from1;
            path0[band] = 1;
        }

        let from0 = cost0 + lambda;
        let from1 = cost1;
        let curr1;
        if from0 < from1 {
            curr1 = from0;
            path1[band] = 0;
        } else {
            curr1 = from1;
            path1[band] = 1;
        }

        cost0 = curr0 + importance[band] * (metric[band] - target0).abs();
        cost1 = curr1 + importance[band] * (metric[band] - target1).abs();
    }

    tf_res[len - 1] = if cost0 < cost1 { 0 } else { 1 };
    if len >= 2 {
        for band in (0..=(len - 2)).rev() {
            let next = tf_res[band + 1];
            tf_res[band] = if next == 1 {
                path1[band + 1]
            } else {
                path0[band + 1]
            };
        }
    }

    tf_select
}

/// Evaluates the trim selector used by the dynamic allocation heuristics.
///
/// This ports `alloc_trim_analysis()` from `celt/celt_encoder.c`. The helper
/// inspects stereo correlation, spectral tilt, transient strength, and the
/// surround analysis results to choose one of eleven trim presets. It updates
/// the stereo saving accumulator as a side effect, mirroring the behaviour of
/// the C implementation.
#[allow(clippy::too_many_arguments)]
fn alloc_trim_analysis(
    mode: &OpusCustomMode<'_>,
    x: &[CeltNorm],
    band_log_e: &[CeltGlog],
    end: usize,
    lm: usize,
    channels: usize,
    n0: usize,
    analysis: &AnalysisInfo,
    stereo_saving: &mut OpusVal16,
    tf_estimate: OpusVal16,
    intensity: usize,
    surround_trim: CeltGlog,
    equiv_rate: OpusInt32,
    _arch: i32,
) -> i32 {
    debug_assert!(channels == 1 || channels == 2);
    debug_assert!(
        x.len() >= channels * n0,
        "insufficient MDCT samples for alloc_trim_analysis"
    );
    debug_assert!(band_log_e.len() >= channels * mode.num_ebands);

    let mut trim = 5.0f32;
    if equiv_rate < 64_000 {
        trim = 4.0;
    } else if equiv_rate < 80_000 {
        let frac = ((equiv_rate - 64_000) >> 10) as f32;
        trim = 4.0 + (1.0 / 16.0) * frac;
    }

    if channels == 2 {
        let mut sum = 0.0f32;
        let limit = intensity.min(mode.num_ebands);

        for band in 0..8.min(mode.num_ebands) {
            let start = (mode.e_bands[band] as usize) << lm;
            let end = (mode.e_bands[band + 1] as usize) << lm;
            if end <= start || end > n0 {
                continue;
            }
            let left = &x[start..end];
            let right = &x[n0 + start..n0 + end];
            let partial = celt_inner_prod(left, right);
            sum += partial;
        }

        sum *= 1.0 / 8.0;
        sum = sum.abs().min(1.0);
        let mut min_xc = sum;

        for band in 8..limit {
            let start = (mode.e_bands[band] as usize) << lm;
            let end = (mode.e_bands[band + 1] as usize) << lm;
            if end <= start || end > n0 {
                continue;
            }
            let left = &x[start..end];
            let right = &x[n0 + start..n0 + end];
            let partial = celt_inner_prod(left, right).abs().min(1.0);
            if partial < min_xc {
                min_xc = partial;
            }
        }

        let log_xc = celt_log2(1.001 - sum * sum);
        let alt = celt_log2(1.001 - min_xc * min_xc);
        let mut log_xc2 = 0.5 * log_xc;
        if alt > log_xc2 {
            log_xc2 = alt;
        }

        let adjustment = (0.75 * log_xc).max(-4.0);
        trim += adjustment;

        let candidate = (-0.5 * log_xc2).min(*stereo_saving + 0.25);
        *stereo_saving = candidate;
    }

    let nb_ebands = mode.num_ebands;
    let mut diff = 0.0f32;
    if end > 1 {
        for ch in 0..channels {
            let base = ch * nb_ebands;
            for band in 0..(end - 1) {
                let weight = (2 + 2 * band as i32 - end as i32) as f32;
                diff += band_log_e[base + band] * weight;
            }
        }
        diff /= (channels * (end - 1)) as f32;
    }

    let slope = ((diff + 1.0) / 6.0).clamp(-2.0, 2.0);
    trim -= slope;
    trim -= surround_trim;
    trim -= 2.0 * tf_estimate;

    if analysis.valid {
        let tonal = 2.0 * (analysis.tonality_slope + 0.05);
        trim -= tonal.clamp(-2.0, 2.0);
    }

    let mut trim_index = floorf(trim + 0.5) as i32;
    trim_index = trim_index.clamp(0, 10);
    trim_index
}

/// Applies the MDCT to each sub-frame for all channels, mirroring
/// `compute_mdcts()` from `celt/celt_encoder.c`.
#[allow(clippy::too_many_arguments)]
fn compute_mdcts(
    mode: &OpusCustomMode<'_>,
    short_blocks: usize,
    input: &[CeltSig],
    output: &mut [CeltSig],
    coded_channels: usize,
    total_channels: usize,
    lm: usize,
    upsample: usize,
    arch: i32,
) {
    assert!(coded_channels > 0 && coded_channels <= total_channels);
    assert!(upsample > 0);
    assert!(lm <= mode.max_lm);

    let overlap = mode.overlap;
    let (block_count, shift) = if short_blocks != 0 {
        (short_blocks, mode.max_lm)
    } else {
        (1, mode.max_lm - lm)
    };
    let transform_len = mode.mdct.effective_len(shift);
    assert!(transform_len.is_multiple_of(2));
    let frame_len = transform_len >> 1;

    let channel_input_stride = block_count * transform_len + overlap;
    let channel_output_stride = block_count * frame_len;

    assert!(input.len() >= total_channels * channel_input_stride);
    assert!(output.len() >= total_channels * channel_output_stride);

    for channel in 0..total_channels {
        let input_base = channel * channel_input_stride;
        let output_base = channel * channel_output_stride;

        for block in 0..block_count {
            let input_offset = input_base + block * transform_len;
            let output_offset = output_base + block;
            let input_end = input_offset + overlap + transform_len;
            let output_end = output_base + channel_output_stride;

            clt_mdct_forward(
                &mode.mdct,
                &input[input_offset..input_end],
                &mut output[output_offset..output_end],
                mode.window,
                overlap,
                shift,
                block_count,
            );
        }
    }

    if total_channels == 2 && coded_channels == 1 {
        let band_len = block_count * frame_len;
        for i in 0..band_len {
            output[i] = 0.5 * (output[i] + output[band_len + i]);
        }
    }

    if upsample != 1 {
        for channel in 0..coded_channels {
            let base = channel * channel_output_stride;
            let bound = channel_output_stride / upsample;
            let (to_scale, to_zero) =
                output[base..base + channel_output_stride].split_at_mut(bound);
            for value in to_scale.iter_mut() {
                *value *= upsample as CeltSig;
            }
            to_zero.fill(0.0);
        }
    }

    let _ = arch;
}

fn ensure_pcm_capacity(pcmp: &[OpusRes], channels: usize, samples: usize) {
    if samples == 0 {
        return;
    }

    let Some(last_frame_index) = samples.checked_sub(1) else {
        return;
    };
    let required = channels
        .checked_mul(last_frame_index)
        .and_then(|value| value.checked_add(1))
        .expect("pcm length calculation overflowed");

    assert!(
        pcmp.len() >= required,
        "PCM slice is shorter than the requested frame"
    );
}

/// Mirrors `celt_preemphasis()` from `celt/celt_encoder.c` for the float build.
///
/// The helper converts the interleaved PCM input to the internal signal
/// representation, applies the high-pass pre-emphasis filter, and updates the
/// running filter memory. Only the first `n` samples of `inp` are modified; the
/// caller is responsible for providing sufficient capacity in the destination
/// buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn celt_preemphasis(
    pcmp: &[OpusRes],
    inp: &mut [CeltSig],
    n: usize,
    channels: usize,
    upsample: usize,
    coef: &[OpusVal16; 4],
    mem: &mut CeltSig,
    clip: bool,
) {
    assert!(channels > 0, "channel count must be positive");
    assert!(upsample > 0, "upsample factor must be positive");
    assert!(
        inp.len() >= n,
        "output buffer too small for requested frame"
    );

    let coef0 = coef[0];
    let mut m = *mem;

    if coef[1] == 0.0 && upsample == 1 && !clip {
        ensure_pcm_capacity(pcmp, channels, n);

        for i in 0..n {
            let x = pcmp[channels * i] * CELT_SIG_SCALE;
            inp[i] = x - m;
            m = coef0 * x;
        }

        *mem = m;
        return;
    }

    let nu = n / upsample;
    if upsample != 1 {
        inp[..n].fill(0.0);
    }

    ensure_pcm_capacity(pcmp, channels, nu);

    for i in 0..nu {
        inp[i * upsample] = pcmp[channels * i] * CELT_SIG_SCALE;
    }

    if clip {
        for i in 0..nu {
            let index = i * upsample;
            inp[index] = inp[index].clamp(-PREEMPHASIS_CLIP_LIMIT, PREEMPHASIS_CLIP_LIMIT);
        }
    }

    if coef[1] == 0.0 {
        for value in &mut inp[..n] {
            let x = *value;
            *value = x - m;
            m = coef0 * x;
        }
    } else {
        let coef1 = coef[1];
        let coef2 = coef[2];

        for value in &mut inp[..n] {
            let x = *value;
            let tmp = coef2 * x;
            *value = tmp + m;
            m = coef1 * *value - coef0 * tmp;
        }
    }

    *mem = m;
}

/// Helper owning the trailing buffers that back [`OpusCustomEncoder`].
#[derive(Debug, Default)]
pub(crate) struct CeltEncoderAlloc {
    in_mem: Vec<CeltSig>,
    prefilter_mem: Vec<CeltSig>,
    old_band_e: Vec<CeltGlog>,
    old_log_e: Vec<CeltGlog>,
    old_log_e2: Vec<CeltGlog>,
    energy_error: Vec<CeltGlog>,
    static_mode: Option<NonNull<OpusCustomMode<'static>>>,
}

impl CeltEncoderAlloc {
    /// Creates a new allocation suitable for the provided mode and channel layout.
    pub(crate) fn new(mode: &OpusCustomMode<'_>, channels: usize) -> Self {
        assert!(
            channels > 0 && channels <= MAX_CHANNELS,
            "unsupported channel layout"
        );

        let overlap = mode.overlap * channels;
        let band_count = channels * mode.num_ebands;

        Self {
            in_mem: vec![0.0; overlap],
            prefilter_mem: vec![0.0; channels * COMBFILTER_MAXPERIOD],
            old_band_e: vec![0.0; band_count],
            old_log_e: vec![0.0; band_count],
            old_log_e2: vec![0.0; band_count],
            energy_error: vec![0.0; band_count],
            static_mode: None,
        }
    }

    /// Returns the number of bytes consumed by the allocation.
    #[must_use]
    pub(crate) fn size_in_bytes(&self) -> usize {
        self.in_mem.len() * core::mem::size_of::<CeltSig>()
            + self.prefilter_mem.len() * core::mem::size_of::<CeltSig>()
            + (self.old_band_e.len()
                + self.old_log_e.len()
                + self.old_log_e2.len()
                + self.energy_error.len())
                * core::mem::size_of::<CeltGlog>()
    }

    /// Borrows the allocation as an [`OpusCustomEncoder`] tied to the provided mode.
    pub(crate) fn as_encoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        energy_mask: Option<&'a [CeltGlog]>,
    ) -> OpusCustomEncoder<'a> {
        OpusCustomEncoder::new(
            mode,
            channels,
            stream_channels,
            energy_mask,
            self.in_mem.as_mut_slice(),
            self.prefilter_mem.as_mut_slice(),
            self.old_band_e.as_mut_slice(),
            self.old_log_e.as_mut_slice(),
            self.old_log_e2.as_mut_slice(),
            self.energy_error.as_mut_slice(),
        )
    }

    /// Clears the buffers and restores the reference reset state.
    pub(crate) fn reset(&mut self) {
        self.in_mem.fill(0.0);
        self.prefilter_mem.fill(0.0);
        self.old_band_e.fill(0.0);
        self.energy_error.fill(0.0);
        self.old_log_e.fill(-28.0);
        self.old_log_e2.fill(-28.0);
    }

    /// Internal helper shared by the public initialisation routines.
    fn init_internal<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        upsample: u32,
        arch: i32,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(CeltEncoderInitError::InvalidChannelCount);
        }
        if stream_channels == 0 || stream_channels > channels {
            return Err(CeltEncoderInitError::InvalidStreamChannels);
        }

        self.reset();
        let mut encoder = self.as_encoder(mode, channels, stream_channels, None);
        encoder.reset_runtime_state();
        encoder.upsample = upsample as i32;
        encoder.start_band = 0;
        encoder.end_band = mode.effective_ebands as i32;
        encoder.signalling = 1;
        encoder.arch = arch;
        encoder.constrained_vbr = true;
        encoder.clip = true;
        encoder.bitrate = OPUS_BITRATE_MAX;
        encoder.use_vbr = false;
        encoder.force_intra = false;
        encoder.complexity = 5;
        encoder.lsb_depth = 24;
        encoder.loss_rate = 0;
        encoder.lfe = false;
        encoder.disable_prefilter = false;
        encoder.disable_inv = false;
        encoder.rng = rng_seed;

        Ok(encoder)
    }

    /// Mirrors `opus_custom_encoder_init_arch()` by allowing the caller to
    /// specify the architecture hint used by the encoder heuristics.
    pub(crate) fn init_custom_encoder_with_arch<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        arch: i32,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
        self.init_internal(mode, channels, stream_channels, 1, arch, rng_seed)
    }

    /// Mirrors `opus_custom_encoder_init()` by configuring the encoder for a custom mode.
    pub(crate) fn init_custom_encoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
        self.init_custom_encoder_with_arch(
            mode,
            channels,
            stream_channels,
            opus_select_arch(),
            rng_seed,
        )
    }

    /// Mirrors `celt_encoder_init()` by configuring the encoder for a public Opus mode.
    pub(crate) fn init_encoder_for_rate<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        sampling_rate: OpusInt32,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
        let upsample = resampling_factor(sampling_rate);
        if upsample == 0 {
            return Err(CeltEncoderInitError::UnsupportedSampleRate);
        }
        self.init_internal(
            mode,
            channels,
            stream_channels,
            upsample,
            opus_select_arch(),
            rng_seed,
        )
    }
}

/// Applies a control request to the provided encoder state.
pub(crate) fn opus_custom_encoder_ctl<'enc, 'req>(
    encoder: &mut OpusCustomEncoder<'enc>,
    request: EncoderCtlRequest<'enc, 'req>,
) -> Result<(), CeltEncoderCtlError> {
    match request {
        EncoderCtlRequest::SetComplexity(value) => {
            if !(0..=10).contains(&value) {
                return Err(CeltEncoderCtlError::InvalidArgument);
            }
            encoder.complexity = value;
        }
        EncoderCtlRequest::SetStartBand(value) => {
            let max = encoder.mode.num_ebands as i32;
            if value < 0 || value >= max {
                return Err(CeltEncoderCtlError::InvalidArgument);
            }
            encoder.start_band = value;
        }
        EncoderCtlRequest::SetEndBand(value) => {
            let max = encoder.mode.num_ebands as i32;
            if value < 1 || value > max {
                return Err(CeltEncoderCtlError::InvalidArgument);
            }
            encoder.end_band = value;
        }
        EncoderCtlRequest::SetPrediction(value) => {
            if !(0..=2).contains(&value) {
                return Err(CeltEncoderCtlError::InvalidArgument);
            }
            encoder.disable_prefilter = value <= 1;
            encoder.force_intra = value == 0;
        }
        EncoderCtlRequest::SetPacketLossPerc(value) => {
            if !(0..=100).contains(&value) {
                return Err(CeltEncoderCtlError::InvalidArgument);
            }
            encoder.loss_rate = value;
        }
        EncoderCtlRequest::SetVbrConstraint(value) => {
            encoder.constrained_vbr = value;
        }
        EncoderCtlRequest::SetVbr(value) => {
            encoder.use_vbr = value;
        }
        EncoderCtlRequest::SetBitrate(value) => {
            if value <= 500 && value != OPUS_BITRATE_MAX {
                return Err(CeltEncoderCtlError::InvalidArgument);
            }
            let capped = min(value, 260_000 * encoder.channels as OpusInt32);
            encoder.bitrate = capped;
        }
        EncoderCtlRequest::SetChannels(value) => {
            if value == 0 || value > encoder.channels {
                return Err(CeltEncoderCtlError::InvalidArgument);
            }
            encoder.stream_channels = value;
        }
        EncoderCtlRequest::SetLsbDepth(value) => {
            if !(8..=24).contains(&value) {
                return Err(CeltEncoderCtlError::InvalidArgument);
            }
            encoder.lsb_depth = value;
        }
        EncoderCtlRequest::GetLsbDepth(out) => {
            *out = encoder.lsb_depth;
        }
        EncoderCtlRequest::SetPhaseInversionDisabled(value) => {
            encoder.disable_inv = value;
        }
        EncoderCtlRequest::GetPhaseInversionDisabled(out) => {
            *out = encoder.disable_inv;
        }
        EncoderCtlRequest::ResetState => {
            encoder.reset_runtime_state();
        }
        EncoderCtlRequest::SetInputClipping(value) => {
            encoder.clip = value;
        }
        EncoderCtlRequest::SetSignalling(value) => {
            encoder.signalling = value;
        }
        EncoderCtlRequest::SetAnalysis(info) => {
            encoder.analysis = info.clone();
        }
        EncoderCtlRequest::SetSilkInfo(info) => {
            encoder.silk_info = info.clone();
        }
        EncoderCtlRequest::GetMode(slot) => {
            *slot = Some(encoder.mode);
        }
        EncoderCtlRequest::GetFinalRange(slot) => {
            *slot = encoder.rng;
        }
        EncoderCtlRequest::SetLfe(value) => {
            encoder.lfe = value;
        }
        EncoderCtlRequest::SetEnergyMask(mask) => {
            encoder.energy_mask = mask;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn transient_analysis(
    input: &[OpusVal32],
    len: usize,
    channels: usize,
    tf_estimate: &mut OpusVal16,
    tf_chan: &mut usize,
    allow_weak_transients: bool,
    weak_transient: &mut bool,
    tone_freq: OpusVal16,
    toneishness: OpusVal32,
) -> bool {
    const INV_TABLE: [u8; 128] = [
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18,
        17, 16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9,
        8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    ];

    debug_assert!(channels * len <= input.len());

    let mut tmp = vec![0.0f32; len];
    *weak_transient = false;

    let mut forward_decay = 0.0625f32;
    if allow_weak_transients {
        forward_decay = 0.03125f32;
    }

    let len2 = len / 2;
    let mut mask_metric = 0.0f32;
    *tf_chan = 0;

    for c in 0..channels {
        let mut mem0 = 0.0f32;
        let mut mem1 = 0.0f32;
        for i in 0..len {
            let x = input[c * len + i];
            let y = mem0 + x;
            let mem00 = mem0;
            mem0 = mem0 - x + 0.5 * mem1;
            mem1 = x - mem00;
            tmp[i] = y;
        }

        for value in tmp.iter_mut().take(len.min(12)) {
            *value = 0.0;
        }

        let mut mean = 0.0f32;
        mem0 = 0.0;
        for i in 0..len2 {
            let x0 = tmp[2 * i];
            let x1 = tmp[2 * i + 1];
            let x2 = x0 * x0 + x1 * x1;
            mean += x2;
            mem0 = x2 + (1.0 - forward_decay) * mem0;
            tmp[i] = forward_decay * mem0;
        }

        mem0 = 0.0;
        let mut max_e = 0.0f32;
        for i in (0..len2).rev() {
            mem0 = tmp[i] + 0.875 * mem0;
            let value = 0.125 * mem0;
            tmp[i] = value;
            if value > max_e {
                max_e = value;
            }
        }

        let frame_energy = celt_sqrt(mean * max_e * 0.5 * len2 as f32);
        // Equivalent to the floating-point branch of the original `norm` computation.
        let norm = (len2 as f32) / (frame_energy * 0.5 + 1e-15f32);

        let mut unmask = 0i32;
        for i in (12..len2.saturating_sub(5)).step_by(4) {
            debug_assert!(!tmp[i].is_nan());
            debug_assert!(!norm.is_nan());
            let scaled = floorf(64.0 * norm * (tmp[i] + 1e-15f32));
            let clamped = scaled.clamp(0.0, 127.0) as usize;
            unmask += i32::from(INV_TABLE[clamped]);
        }

        if len2 > 17 {
            let denom = 6 * (len2 as i32 - 17);
            if denom > 0 {
                let value = (64 * unmask * 4) as f32 / denom as f32;
                if value > mask_metric {
                    mask_metric = value;
                    *tf_chan = c;
                }
            }
        }
    }

    let mut is_transient = mask_metric > 200.0;
    if toneishness > 0.98 && tone_freq < 0.026 {
        is_transient = false;
    }
    if allow_weak_transients && is_transient && mask_metric < 600.0 {
        is_transient = false;
        *weak_transient = true;
    }

    let mut tf_max = celt_sqrt(27.0 * mask_metric) - 42.0;
    if tf_max < 0.0 {
        tf_max = 0.0;
    }
    let tf_max = tf_max.min(163.0);
    let value = (0.0069 * tf_max - 0.139).max(0.0);
    *tf_estimate = celt_sqrt(value);
    is_transient
}

fn patch_transient_decision(
    new_e: &[CeltGlog],
    old_e: &[CeltGlog],
    nb_ebands: usize,
    start: usize,
    end: usize,
    channels: usize,
) -> bool {
    debug_assert!(new_e.len() >= channels * nb_ebands);
    debug_assert!(old_e.len() >= channels * nb_ebands);
    debug_assert!(start < end);
    debug_assert!(end <= nb_ebands);

    let mut spread_old = vec![0.0f32; nb_ebands];

    if channels == 1 {
        spread_old[start] = old_e[start];
        for i in (start + 1)..end {
            let prev = spread_old[i - 1] - 1.0;
            spread_old[i] = prev.max(old_e[i]);
        }
    } else {
        spread_old[start] = old_e[start].max(old_e[start + nb_ebands]);
        for i in (start + 1)..end {
            let prev = spread_old[i - 1] - 1.0;
            let pair = old_e[i].max(old_e[i + nb_ebands]);
            spread_old[i] = prev.max(pair);
        }
    }

    if end >= 2 {
        for i in (start..=(end - 2)).rev() {
            let next = spread_old[i + 1] - 1.0;
            if next > spread_old[i] {
                spread_old[i] = next;
            }
        }
    }

    let start_i = start.max(2);
    let mut mean_diff = 0.0f32;
    for c in 0..channels {
        let base = c * nb_ebands;
        for i in start_i..(end.saturating_sub(1)) {
            let x1 = new_e[base + i].max(0.0);
            let x2 = spread_old[i].max(0.0);
            let diff = (x1 - x2).max(0.0);
            mean_diff += diff;
        }
    }

    let denom = (channels * (end.saturating_sub(1).saturating_sub(start_i))) as f32;
    if denom > 0.0 {
        mean_diff /= denom;
    }

    mean_diff > 1.0
}

#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn dynalloc_analysis(
    band_log_e: &[CeltGlog],
    band_log_e2: &[CeltGlog],
    old_band_e: &[CeltGlog],
    nb_ebands: usize,
    start: usize,
    end: usize,
    channels: usize,
    offsets: &mut [i32],
    lsb_depth: i32,
    log_n: &[i16],
    is_transient: bool,
    vbr: bool,
    constrained_vbr: bool,
    e_bands: &[i16],
    lm: i32,
    effective_bytes: i32,
    tot_boost: &mut i32,
    lfe: bool,
    surround_dynalloc: &mut [CeltGlog],
    analysis: &AnalysisInfo,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: OpusVal16,
    toneishness: OpusVal32,
) -> CeltGlog {
    debug_assert!(channels <= MAX_CHANNELS);
    debug_assert!(band_log_e.len() >= channels * nb_ebands);
    debug_assert!(band_log_e2.len() >= channels * nb_ebands);
    debug_assert!(old_band_e.len() >= channels * nb_ebands);
    debug_assert!(offsets.len() >= nb_ebands);
    debug_assert!(importance.len() >= nb_ebands);
    debug_assert!(spread_weight.len() >= nb_ebands);
    debug_assert!(log_n.len() >= end);
    debug_assert!(e_bands.len() > end);
    debug_assert!(surround_dynalloc.len() >= end);

    offsets.iter_mut().for_each(|value| *value = 0);
    importance.iter_mut().for_each(|value| *value = 0);
    spread_weight.iter_mut().for_each(|value| *value = 0);

    let mut follower = vec![0.0f32; channels * nb_ebands];
    let mut noise_floor = vec![0.0f32; nb_ebands];
    let mut band_log_e3 = vec![0.0f32; nb_ebands];

    let mut max_depth = -31.9f32;
    let depth_shift = (9 - lsb_depth) as f32;

    for i in 0..end {
        let log_n_val = f32::from(log_n[i]);
        let mean = E_MEANS
            .get(i)
            .copied()
            .unwrap_or_else(|| *E_MEANS.last().expect("non-empty e_means"));
        let index = (i + 5) as f32;
        noise_floor[i] = 0.0625 * log_n_val + 0.5 + depth_shift - mean + 0.0062 * index * index;
    }

    for c in 0..channels {
        let base = c * nb_ebands;
        for i in 0..end {
            let depth = band_log_e[base + i] - noise_floor[i];
            if depth > max_depth {
                max_depth = depth;
            }
        }
    }

    let mut mask = vec![0.0f32; nb_ebands];
    let mut sig = vec![0.0f32; nb_ebands];
    for i in 0..end {
        let mut value = band_log_e[i] - noise_floor[i];
        if channels == 2 {
            let other = band_log_e[nb_ebands + i] - noise_floor[i];
            if other > value {
                value = other;
            }
        }
        mask[i] = value;
        sig[i] = value;
    }
    for i in 1..end {
        let candidate = mask[i - 1] - 2.0;
        if candidate > mask[i] {
            mask[i] = candidate;
        }
    }
    if end >= 2 {
        for i in (0..=end - 2).rev() {
            let candidate = mask[i + 1] - 3.0;
            if candidate > mask[i] {
                mask[i] = candidate;
            }
        }
    }

    let base_threshold = (max_depth - 12.0).max(0.0);
    for i in 0..end {
        let clamp = base_threshold.max(mask[i]);
        let smr = sig[i] - clamp;
        let rounded = floorf(smr + 0.5);
        let mut shift = -(rounded as i32);
        shift = shift.clamp(0, 5);
        spread_weight[i] = 32 >> shift;
    }

    let mut total_boost_bits = 0i32;

    if effective_bytes >= (30 + 5 * lm) && !lfe {
        let mut last = 0usize;
        for c in 0..channels {
            let base = c * nb_ebands;
            band_log_e3[..end].copy_from_slice(&band_log_e2[base..base + end]);
            if lm == 0 {
                for i in 0..end.min(8) {
                    let idx = base + i;
                    let current = band_log_e2[idx];
                    let previous = old_band_e[idx];
                    band_log_e3[i] = current.max(previous);
                }
            }

            let follower_slice = &mut follower[base..base + nb_ebands];
            if end > 0 {
                follower_slice[0] = band_log_e3[0];
            }
            for i in 1..end {
                if band_log_e3[i] > band_log_e3[i - 1] + 0.5 {
                    last = i;
                }
                let candidate = follower_slice[i - 1] + 1.5;
                follower_slice[i] = band_log_e3[i].min(candidate);
            }

            let mut idx = last;
            while idx > 0 {
                let prev = idx - 1;
                let candidate = follower_slice[idx] + 2.0;
                let min_val = candidate.min(band_log_e3[prev]);
                if min_val < follower_slice[prev] {
                    follower_slice[prev] = min_val;
                }
                idx -= 1;
            }

            if end >= 3 {
                let median_start = median_of_3(&band_log_e3[..3]) - 1.0;
                follower_slice[0] = follower_slice[0].max(median_start);
                if end > 1 {
                    follower_slice[1] = follower_slice[1].max(median_start);
                }
                let median_end = median_of_3(&band_log_e3[end - 3..end]) - 1.0;
                if end >= 2 {
                    follower_slice[end - 2] = follower_slice[end - 2].max(median_end);
                }
                follower_slice[end - 1] = follower_slice[end - 1].max(median_end);
            }
            if end > 4 {
                for i in 2..end - 2 {
                    let median = median_of_5(&band_log_e3[i - 2..i + 3]) - 1.0;
                    if median > follower_slice[i] {
                        follower_slice[i] = median;
                    }
                }
            }

            for i in 0..end {
                if noise_floor[i] > follower_slice[i] {
                    follower_slice[i] = noise_floor[i];
                }
            }
        }

        if channels == 2 {
            for i in start..end {
                let left_idx = i;
                let right_idx = nb_ebands + i;
                let updated_right = follower[right_idx].max(follower[left_idx] - 4.0);
                follower[right_idx] = updated_right;
                let updated_left = follower[left_idx].max(updated_right - 4.0);
                follower[left_idx] = updated_left;
                let left_depth = (band_log_e[left_idx] - follower[left_idx]).max(0.0);
                let right_depth = (band_log_e[right_idx] - follower[right_idx]).max(0.0);
                follower[left_idx] = 0.5 * (left_depth + right_depth);
            }
        } else {
            for i in start..end {
                follower[i] = (band_log_e[i] - follower[i]).max(0.0);
            }
        }

        for i in start..end {
            let surround = surround_dynalloc[i];
            if surround > follower[i] {
                follower[i] = surround;
            }
        }

        for i in start..end {
            let capped = follower[i].min(4.0);
            let weight = 13.0 * celt_exp2(capped);
            importance[i] = floorf(weight + 0.5) as i32;
        }

        if ((!vbr) || constrained_vbr) && !is_transient {
            for value in &mut follower[start..end] {
                *value *= 0.5;
            }
        }

        for i in start..end {
            if i < 8 {
                follower[i] *= 2.0;
            }
            if i >= 12 {
                follower[i] *= 0.5;
            }
        }

        if toneishness > 0.98 {
            let freq_bin = floorf(tone_freq * (120.0 / core::f32::consts::PI) + 0.5) as i32;
            for i in start..end {
                let band_low = i32::from(e_bands[i]);
                let band_high = i32::from(e_bands[i + 1]);
                if freq_bin >= band_low && freq_bin <= band_high {
                    follower[i] += 2.0;
                }
                if freq_bin >= band_low - 1 && freq_bin <= band_high + 1 {
                    follower[i] += 1.0;
                }
                if freq_bin >= band_low - 2 && freq_bin <= band_high + 2 {
                    follower[i] += 1.0;
                }
                if freq_bin >= band_low - 3 && freq_bin <= band_high + 3 {
                    follower[i] += 0.5;
                }
            }
        }

        if analysis.valid {
            let leak_len = end.min(LEAK_BANDS).min(analysis.leak_boost.len());
            for i in start..leak_len {
                follower[i] += f32::from(analysis.leak_boost[i]) / 64.0;
            }
        }

        for i in start..end {
            let follower_val = follower[i].min(4.0);
            let band_width = i32::from(e_bands[i + 1]) - i32::from(e_bands[i]);
            let width = (channels as i32 * band_width) << lm;
            let (boost, boost_bits) = if width < 6 {
                let boost = follower_val as i32;
                let bits = (boost * width) << BITRES;
                (boost, bits)
            } else if width > 48 {
                let boost = (follower_val * 8.0) as i32;
                let bits = ((boost * width) << BITRES) / 8;
                (boost, bits)
            } else {
                let boost = (follower_val * width as f32 / 6.0) as i32;
                let bits = (boost * 6) << BITRES;
                (boost, bits)
            };

            if ((!vbr) || (constrained_vbr && !is_transient))
                && (((total_boost_bits + boost_bits) >> BITRES) >> 3) > (2 * effective_bytes / 3)
            {
                let cap = (2 * effective_bytes / 3) << (BITRES + 3);
                offsets[i] = cap - total_boost_bits;
                total_boost_bits = cap;
                break;
            }

            offsets[i] = boost;
            total_boost_bits += boost_bits;
        }
    } else {
        for value in &mut importance[start..end] {
            *value = 13;
        }
    }

    *tot_boost = total_boost_bits;
    max_depth
}

#[allow(clippy::too_many_arguments)]
fn run_prefilter(
    encoder: &mut OpusCustomEncoder<'_>,
    input: &mut [CeltSig],
    prefilter_mem: &mut [CeltSig],
    channels: usize,
    n: usize,
    prefilter_tapset: i32,
    pitch: &mut i32,
    gain: &mut OpusVal16,
    qgain: &mut i32,
    enabled: bool,
    tf_estimate: OpusVal16,
    nb_available_bytes: i32,
    analysis: &AnalysisInfo,
    mut tone_freq: OpusVal16,
    toneishness: OpusVal32,
) -> bool {
    assert!(channels > 0, "run_prefilter requires at least one channel");
    assert!(n > 0, "run_prefilter expects a positive frame size");

    let mode = encoder.mode;
    let overlap = mode.overlap;
    let stride = overlap + n;
    let history_len = COMBFILTER_MAXPERIOD;

    assert!(
        input.len() >= channels * stride,
        "time buffer must expose channels * (n + overlap) samples",
    );
    assert!(
        prefilter_mem.len() >= channels * history_len,
        "prefilter history must expose channels * COMBFILTER_MAXPERIOD samples",
    );

    let mut pre = vec![0.0; channels * (n + history_len)];
    for ch in 0..channels {
        let pre_offset = ch * (n + history_len);
        let pre_slice = &mut pre[pre_offset..pre_offset + history_len + n];
        let history = &prefilter_mem[ch * history_len..(ch + 1) * history_len];
        pre_slice[..history_len].copy_from_slice(history);

        let input_offset = ch * stride;
        let input_slice = &input[input_offset + overlap..input_offset + overlap + n];
        pre_slice[history_len..history_len + n].copy_from_slice(input_slice);
    }

    let mut channel_views: [&[f32]; MAX_CHANNELS] = [&[]; MAX_CHANNELS];
    for ch in 0..channels {
        let start = ch * (n + history_len);
        let end = start + history_len + n;
        channel_views[ch] = &pre[start..end];
    }

    let mut pitch_index = COMBFILTER_MINPERIOD as i32;
    let mut gain1 = 0.0;

    if enabled {
        let downsample_len = history_len + n;
        let mut pitch_buf = vec![0.0; downsample_len >> 1];
        pitch_downsample(&channel_views[..channels], &mut pitch_buf, downsample_len, encoder.arch);

        let search_span = history_len - 3 * COMBFILTER_MINPERIOD;
        if search_span > 0 {
            let offset = history_len >> 1;
            if offset < pitch_buf.len() {
                let result = pitch_search(
                    &pitch_buf[offset..],
                    &pitch_buf,
                    n,
                    search_span,
                    encoder.arch,
                );
                pitch_index = result;
            }
        }

        gain1 = remove_doubling(
            &pitch_buf,
            history_len,
            COMBFILTER_MINPERIOD,
            n,
            &mut pitch_index,
            encoder.prefilter_period,
            encoder.prefilter_gain,
            encoder.arch,
        );
        let max_period = (history_len - 2) as i32;
        if pitch_index > max_period {
            pitch_index = max_period;
        }
        gain1 *= 0.7;

        if toneishness > 0.99 {
            while tone_freq >= 0.39 {
                tone_freq *= 0.5;
            }
            if tone_freq > 0.006_148 {
                let candidate = floorf(0.5 + 2.0 * core::f32::consts::PI / tone_freq) as i32;
                pitch_index = candidate.min(max_period);
            } else {
                pitch_index = COMBFILTER_MINPERIOD as i32;
            }
            gain1 = 0.75;
        }

        if encoder.loss_rate > 2 {
            gain1 *= 0.5;
        }
        if encoder.loss_rate > 4 {
            gain1 *= 0.5;
        }
        if encoder.loss_rate > 8 {
            gain1 = 0.0;
        }
    }

    if analysis.valid {
        gain1 *= analysis.max_pitch_ratio;
    }

    let mut pf_threshold: f32 = 0.2;

    if (pitch_index - encoder.prefilter_period).abs() * 10 > pitch_index {
        pf_threshold += 0.2;
        if tf_estimate > 0.98 {
            gain1 = 0.0;
        }
    }
    if nb_available_bytes < 25 {
        pf_threshold += 0.1;
    }
    if nb_available_bytes < 35 {
        pf_threshold += 0.1;
    }
    if encoder.prefilter_gain > 0.4 {
        pf_threshold -= 0.1;
    }
    if encoder.prefilter_gain > 0.55 {
        pf_threshold -= 0.1;
    }

    pf_threshold = pf_threshold.max(0.2);

    let mut pf_on = false;
    let mut qg_local = 0;
    if gain1 < pf_threshold {
        gain1 = 0.0;
        pitch_index = COMBFILTER_MINPERIOD as i32;
    } else {
        if (gain1 - encoder.prefilter_gain).abs() < 0.1 {
            gain1 = encoder.prefilter_gain;
        }
        let mut quant = floorf(0.5 + gain1 * 32.0 / 3.0) as i32 - 1;
        quant = quant.clamp(0, 7);
        gain1 = 0.093_75 * (quant + 1) as f32;
        qg_local = quant;
        pf_on = true;
    }

    let mut before = [0.0f32; MAX_CHANNELS];
    let mut after = [0.0f32; MAX_CHANNELS];
    let mut cancel_pitch = false;

    let prev_tapset = encoder.prefilter_tapset.max(0) as usize;
    let new_tapset = prefilter_tapset.max(0) as usize;
    let offset = mode.short_mdct_size.saturating_sub(overlap).min(n);

    encoder.prefilter_period = encoder.prefilter_period.max(COMBFILTER_MINPERIOD as i32);

    for ch in 0..channels {
        let input_offset = ch * stride;
        let (head, tail) = input[input_offset..input_offset + stride].split_at_mut(overlap);
        head.copy_from_slice(&encoder.in_mem[ch * overlap..(ch + 1) * overlap]);

        let mut sum_before = 0.0;
        for sample in tail.iter().take(n) {
            sum_before += sample.abs();
        }
        before[ch] = sum_before;

        let pre_offset = ch * (n + history_len);
        let pre_channel = &pre[pre_offset..pre_offset + history_len + n];

        if offset > 0 {
            let (first, rest) = tail.split_at_mut(offset);
            comb_filter(
                first,
                pre_channel,
                history_len,
                offset,
                encoder.prefilter_period,
                encoder.prefilter_period,
                -encoder.prefilter_gain,
                -encoder.prefilter_gain,
                prev_tapset,
                prev_tapset,
                &[],
                0,
                encoder.arch,
            );
            comb_filter(
                rest,
                pre_channel,
                history_len + offset,
                n - offset,
                encoder.prefilter_period,
                pitch_index,
                -encoder.prefilter_gain,
                -gain1,
                prev_tapset,
                new_tapset,
                mode.window,
                overlap,
                encoder.arch,
            );
        } else {
            comb_filter(
                tail,
                pre_channel,
                history_len,
                n,
                encoder.prefilter_period,
                pitch_index,
                -encoder.prefilter_gain,
                -gain1,
                prev_tapset,
                new_tapset,
                mode.window,
                overlap,
                encoder.arch,
            );
        }

        let mut sum_after = 0.0;
        for sample in tail.iter().take(n) {
            sum_after += sample.abs();
        }
        after[ch] = sum_after;
    }

    if channels == 2 {
        let thresh0 = 0.25 * gain1 * before[0] + 0.01 * before[1];
        let thresh1 = 0.25 * gain1 * before[1] + 0.01 * before[0];
        if (after[0] - before[0]) > thresh0 || (after[1] - before[1]) > thresh1 {
            cancel_pitch = true;
        }
        if (before[0] - after[0]) < thresh0 && (before[1] - after[1]) < thresh1 {
            cancel_pitch = true;
        }
    } else if after[0] > before[0] {
        cancel_pitch = true;
    }

    if cancel_pitch {
        for ch in 0..channels {
            let input_offset = ch * stride;
            let channel = &mut input[input_offset..input_offset + stride];
            let pre_offset = ch * (n + history_len);
            let pre_channel = &pre[pre_offset..pre_offset + history_len + n];

            channel[overlap..overlap + n]
                .copy_from_slice(&pre_channel[history_len..history_len + n]);

            if overlap > 0 && offset < n {
                let span = overlap.min(n - offset);
                let start = overlap + offset;
                let end = start + span;
                comb_filter(
                    &mut channel[start..end],
                    pre_channel,
                    history_len + offset,
                    span,
                    encoder.prefilter_period,
                    pitch_index,
                    -encoder.prefilter_gain,
                    0.0,
                    prev_tapset,
                    new_tapset,
                    mode.window,
                    span,
                    encoder.arch,
                );
            }
        }
        gain1 = 0.0;
        qg_local = 0;
        pf_on = false;
    }

    for ch in 0..channels {
        let input_offset = ch * stride;
        let channel = &input[input_offset + n..input_offset + n + overlap];
        encoder.in_mem[ch * overlap..(ch + 1) * overlap].copy_from_slice(channel);

        let pre_offset = ch * (n + history_len);
        let pre_channel = &pre[pre_offset..pre_offset + history_len + n];
        let mem = &mut prefilter_mem[ch * history_len..(ch + 1) * history_len];
        if n > history_len {
            mem.copy_from_slice(&pre_channel[n..n + history_len]);
        } else {
            let shift = history_len - n;
            mem.copy_within(n..history_len, 0);
            mem[shift..].copy_from_slice(&pre_channel[history_len..history_len + n]);
        }
    }

    *gain = gain1;
    *pitch = pitch_index;
    *qgain = qg_local;
    pf_on
}

fn tf_encode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: usize,
    mut tf_select: i32,
    enc: &mut EcEnc<'_>,
) {
    debug_assert!(start <= end);
    debug_assert!(end <= tf_res.len());
    debug_assert!(lm < TF_SELECT_TABLE.len());

    let mut budget = enc.ctx().storage * 8;
    let mut tell = ec_tell(enc.ctx()) as OpusUint32;
    let mut logp = if is_transient { 2u32 } else { 4u32 };
    let mut curr = 0;
    let mut tf_changed = 0;

    let reserve_select = lm > 0 && tell + logp < budget;
    if reserve_select {
        budget -= 1;
    }

    for slot in tf_res[start..end].iter_mut() {
        if tell + logp <= budget {
            let symbol = OpusInt32::from(*slot ^ curr);
            enc.enc_bit_logp(symbol, logp);
            tell = ec_tell(enc.ctx()) as OpusUint32;
            curr = *slot;
            tf_changed |= curr;
        } else {
            *slot = curr;
        }
        logp = if is_transient { 4u32 } else { 5u32 };
    }

    let base = 4 * usize::from(is_transient);

    if reserve_select
        && TF_SELECT_TABLE[lm][base + tf_changed as usize]
            != TF_SELECT_TABLE[lm][base + 2 + tf_changed as usize]
    {
        enc.enc_bit_logp(tf_select, 1);
    } else {
        tf_select = 0;
    }

    debug_assert!((0..=1).contains(&tf_select));

    for slot in tf_res[start..end].iter_mut() {
        debug_assert!((0..=1).contains(slot));
        let offset = base + 2 * tf_select as usize + *slot as usize;
        *slot = i32::from(TF_SELECT_TABLE[lm][offset]);
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_vbr(
    mode: &OpusCustomMode<'_>,
    analysis: &AnalysisInfo,
    base_target: OpusInt32,
    lm: i32,
    bitrate: OpusInt32,
    last_coded_bands: i32,
    channels: usize,
    intensity: i32,
    constrained_vbr: bool,
    stereo_saving: OpusVal16,
    tot_boost: OpusInt32,
    tf_estimate: OpusVal16,
    pitch_change: bool,
    max_depth: CeltGlog,
    lfe: bool,
    has_surround_mask: bool,
    surround_masking: CeltGlog,
    temporal_vbr: CeltGlog,
) -> OpusInt32 {
    use crate::celt::entcode::BITRES;

    let bitres = BITRES as i32;
    let nb_ebands = mode.num_ebands;
    let e_bands = mode.e_bands;

    let mut coded_bands = if last_coded_bands > 0 {
        last_coded_bands as usize
    } else {
        nb_ebands
    };
    coded_bands = coded_bands.min(nb_ebands);

    let mut coded_bins = i32::from(e_bands[coded_bands]) << lm;
    if channels == 2 {
        let stereo_index = intensity.clamp(0, coded_bands as i32) as usize;
        coded_bins += i32::from(e_bands[stereo_index]) << lm;
    }

    let mut target = base_target;

    if analysis.valid && analysis.activity < 0.4 {
        let coded = (i64::from(coded_bins) << bitres) as f32;
        let reduction = (coded * (0.4 - analysis.activity)) as i32;
        target -= reduction;
    }

    if channels == 2 && coded_bins > 0 {
        let stereo_bands = intensity.clamp(0, coded_bands as i32) as usize;
        let stereo_dof = (i32::from(e_bands[stereo_bands]) << lm) - stereo_bands as i32;
        if stereo_dof > 0 {
            let max_frac = 0.8f32 * stereo_dof as f32 / coded_bins as f32;
            let capped_saving = stereo_saving.min(1.0);
            let term1 = (max_frac * target as f32) as i32;
            let raw = capped_saving - 0.1f32;
            let term2 = (raw * (i64::from(stereo_dof) << bitres) as f32) as i32;
            target -= term1.min(term2);
        }
    }

    target += tot_boost - (19 << lm);

    let tf_calibration = 0.044f32;
    let tf_adjust = 2.0f32 * (tf_estimate - tf_calibration) * target as f32;
    target += tf_adjust as i32;

    if analysis.valid && !lfe {
        let tonal = (analysis.tonality - 0.15f32).max(0.0) - 0.12f32;
        if tonal != 0.0 {
            let coded = (i64::from(coded_bins) << bitres) as f32;
            let mut tonal_target = target as f32 + 1.2f32 * coded * tonal;
            if pitch_change {
                tonal_target += 0.8f32 * coded;
            }
            target = tonal_target as i32;
        }
    }

    if has_surround_mask && !lfe {
        let surround_delta = (surround_masking * (i64::from(coded_bins) << bitres) as f32) as i32;
        let surround_target = target + surround_delta;
        target = max(target / 4, surround_target);
    }

    if nb_ebands >= 2 {
        let bins = i32::from(e_bands[nb_ebands - 2]) << lm;
        let floor_depth = ((i64::from(channels as i32 * bins) << bitres) as f32 * max_depth) as i32;
        let floor_depth = max(floor_depth, target >> 2);
        target = min(target, floor_depth);
    }

    if (!has_surround_mask || lfe) && constrained_vbr {
        let delta = (target - base_target) as f32;
        target = base_target + (0.67f32 * delta) as i32;
    }

    if !has_surround_mask && tf_estimate < 0.2f32 {
        let clamp = (96_000 - bitrate).clamp(0, 32_000);
        let amount = 0.0000031f32 * clamp as f32;
        let tvbr_factor = temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }

    let doubled = base_target.saturating_mul(2);
    target = min(doubled, target);

    target
}

fn resolve_lm(mode: &OpusCustomMode<'_>, frame_size: usize) -> Option<usize> {
    let mut n = mode.short_mdct_size;
    for lm in 0..=mode.max_lm {
        if n == frame_size {
            return Some(lm);
        }
        n <<= 1;
    }
    None
}

fn mdct_shift_for_frame(mode: &OpusCustomMode<'_>, frame_size: usize) -> Option<usize> {
    (0..=mode.mdct.max_shift()).find(|&shift| mode.mdct.effective_len(shift) == frame_size)
}

fn prepare_time_domain(
    encoder: &mut OpusCustomEncoder<'_>,
    pcm: &[CeltSig],
    frame_size: usize,
) -> Result<Vec<CeltSig>, CeltEncodeError> {
    let channels = encoder.channels;
    if pcm.len() < channels * frame_size {
        return Err(CeltEncodeError::InsufficientPcm);
    }

    let overlap = encoder.mode.overlap;
    let stride = frame_size + overlap;
    let mut buffer = vec![0.0f32; channels * stride];

    for ch in 0..channels {
        let input_stride = channels;
        let dst_offset = ch * stride;
        let prev = &encoder.in_mem[ch * overlap..(ch + 1) * overlap];
        buffer[dst_offset..dst_offset + overlap].copy_from_slice(prev);

        let mut mem = encoder.preemph_mem_encoder[ch];
        for i in 0..frame_size {
            let raw = pcm[i * input_stride + ch];
            let emphasised = raw - mem;
            buffer[dst_offset + overlap + i] = emphasised;
            mem = encoder.mode.pre_emphasis[0] * raw;
        }
        encoder.preemph_mem_encoder[ch] = mem;

        if overlap > 0 {
            let available = frame_size.min(overlap);
            let copy_start = dst_offset + overlap + frame_size - available;
            let dst_start = dst_offset + frame_size;
            let tail = buffer[copy_start..copy_start + available].to_vec();
            buffer[dst_start..dst_start + available].copy_from_slice(&tail);
            if available < overlap {
                buffer[dst_start + available..dst_start + overlap].fill(0.0);
            }
        }
    }

    Ok(buffer)
}

#[allow(clippy::too_many_arguments)]
fn compute_mdct_spectrum(
    mode: &OpusCustomMode<'_>,
    short_blocks: bool,
    time: &[CeltSig],
    freq: &mut [CeltSig],
    encoded_channels: usize,
    total_channels: usize,
    frame_size: usize,
    shift: usize,
) {
    let overlap = mode.overlap;
    let stride = frame_size + overlap;

    let blocks = if short_blocks { 1usize << shift } else { 1 };

    for ch in 0..encoded_channels {
        let src_index = ch * stride;
        let input = &time[src_index..src_index + overlap + frame_size];
        let output = &mut freq[ch * (frame_size / 2)..(ch + 1) * (frame_size / 2)];
        clt_mdct_forward(
            &mode.mdct,
            input,
            output,
            mode.window,
            overlap,
            shift,
            blocks,
        );
    }

    if total_channels == 2 && encoded_channels == 1 {
        for i in 0..(frame_size / 2) {
            let left = freq[i];
            let right = freq[frame_size / 2 + i];
            freq[i] = 0.5 * (left + right);
        }
    }
}

fn update_overlap_history(
    encoder: &mut OpusCustomEncoder<'_>,
    time: &[CeltSig],
    frame_size: usize,
) {
    let channels = encoder.channels;
    let overlap = encoder.mode.overlap;
    let stride = frame_size + overlap;

    for ch in 0..channels {
        let src = &time[ch * stride + frame_size..ch * stride + frame_size + overlap];
        encoder.in_mem[ch * overlap..(ch + 1) * overlap].copy_from_slice(src);
    }
}

/// Performs the analysis stages of the CELT encoder and updates the runtime
/// state using the provided range encoder. This mirrors the portions of the C
/// implementation that prepare the spectrum before quantisation.
fn encode_internal(
    encoder: &mut OpusCustomEncoder<'_>,
    pcm: &[CeltSig],
    frame_size: usize,
    enc: &mut EcEnc<'_>,
) -> Result<(), CeltEncodeError> {
    let mode = encoder.mode;
    let Some(lm) = resolve_lm(mode, frame_size) else {
        return Err(CeltEncodeError::InvalidFrameSize);
    };
    let Some(shift) = mdct_shift_for_frame(mode, frame_size) else {
        return Err(CeltEncodeError::InvalidFrameSize);
    };

    let time = prepare_time_domain(encoder, pcm, frame_size)?;
    update_overlap_history(encoder, &time, frame_size);

    let freq_bins = frame_size / 2;
    let mut freq = vec![0.0f32; encoder.stream_channels * freq_bins];
    compute_mdct_spectrum(
        mode,
        false,
        &time,
        &mut freq,
        encoder.stream_channels,
        encoder.channels,
        frame_size,
        shift,
    );

    let end_band = encoder.end_band.clamp(0, mode.num_ebands as i32) as usize;
    let mut band_e = vec![0.0f32; encoder.stream_channels * mode.num_ebands];
    compute_band_energies(
        mode,
        &freq,
        &mut band_e,
        end_band,
        encoder.stream_channels,
        lm,
        encoder.arch,
    );

    let mut band_log = vec![0.0f32; encoder.stream_channels * mode.num_ebands];
    amp2_log2(
        mode,
        end_band,
        end_band,
        &band_e,
        &mut band_log,
        encoder.stream_channels,
    );

    let stride = mode.num_ebands;
    for (dst, src) in encoder
        .old_band_e
        .chunks_mut(stride)
        .zip(band_e.chunks(stride))
        .take(encoder.stream_channels)
    {
        dst[..end_band].copy_from_slice(&src[..end_band]);
    }
    for (dst, src) in encoder
        .old_log_e
        .chunks_mut(stride)
        .zip(band_log.chunks(stride))
        .take(encoder.stream_channels)
    {
        dst[..end_band].copy_from_slice(&src[..end_band]);
    }
    for chunk in encoder
        .old_log_e2
        .chunks_mut(stride)
        .take(encoder.stream_channels)
    {
        for slot in &mut chunk[..end_band] {
            *slot = (*slot * 0.75) + 0.25 * -28.0;
        }
    }

    encoder.last_coded_bands = end_band as i32;
    encoder.tapset_decision = 0;
    encoder.consec_transient = 0;

    // The simplified port does not emit additional data yet but ensures the
    // entropy coder remains in a valid state.
    enc.enc_bits(0, 0);

    Ok(())
}

/// Minimal Rust translation of the reference `celt_encode_with_ec()` entry point.
///
/// The implementation focuses on the analysis scaffolding required by later
/// ports. It performs the PCM pre-emphasis, MDCT evaluation, and band energy
/// computation so that the encoder state remains internally consistent even
/// though the full bitstream packing path is still pending.
#[allow(clippy::too_many_arguments)]
pub(crate) fn celt_encode_with_ec(
    encoder: &mut OpusCustomEncoder<'_>,
    pcm: Option<&[CeltSig]>,
    frame_size: usize,
    compressed: Option<&mut [u8]>,
    range_encoder: Option<&mut EcEnc<'_>>,
) -> Result<usize, CeltEncodeError> {
    let pcm = pcm.ok_or(CeltEncodeError::InsufficientPcm)?;

    if let Some(enc) = range_encoder {
        encode_internal(encoder, pcm, frame_size, enc)?;
        enc.enc_done();
        let error = enc.ctx().error;
        encoder.rng = enc.ctx().rng;
        if error != 0 {
            return Err(CeltEncodeError::MissingOutput);
        }
        return Ok(enc.range_bytes() as usize);
    }

    if let Some(buf) = compressed {
        let mut local = EcEnc::new(buf);
        encode_internal(encoder, pcm, frame_size, &mut local)?;
        local.enc_done();
        let error = local.ctx().error;
        encoder.rng = local.ctx().rng;
        if error != 0 {
            return Err(CeltEncodeError::MissingOutput);
        }
        return Ok(local.range_bytes() as usize);
    }

    Err(CeltEncodeError::MissingOutput)
}

fn required_pcm_samples(channels: usize, frame_size: usize) -> Result<usize, CeltEncodeError> {
    channels
        .checked_mul(frame_size)
        .ok_or(CeltEncodeError::InsufficientPcm)
}

fn convert_i16_to_celt_sig(pcm: &[OpusInt16], required: usize) -> Vec<CeltSig> {
    pcm.iter()
        .take(required)
        .map(|&sample| CeltSig::from(sample))
        .collect()
}

fn convert_i24_to_celt_sig(pcm: &[OpusInt32], required: usize) -> Vec<CeltSig> {
    pcm.iter()
        .take(required)
        .map(|&sample| {
            let rounded = (sample + (1 << 7)) >> 8;
            rounded.clamp(-32_768, 32_767) as CeltSig
        })
        .collect()
}

fn convert_f32_to_celt_sig(pcm: &[f32], required: usize) -> Vec<CeltSig> {
    pcm.iter()
        .take(required)
        .map(|&sample| CeltSig::from(float2int16(sample)))
        .collect()
}

fn encode_with_converted_pcm(
    encoder: &mut OpusCustomEncoder<'_>,
    pcm: &[CeltSig],
    frame_size: usize,
    compressed: &mut [u8],
    nb_compressed_bytes: usize,
) -> Result<usize, CeltEncodeError> {
    if nb_compressed_bytes > compressed.len() {
        return Err(CeltEncodeError::MissingOutput);
    }
    if nb_compressed_bytes < 2 {
        return Err(CeltEncodeError::MissingOutput);
    }

    celt_encode_with_ec(
        encoder,
        Some(pcm),
        frame_size,
        Some(&mut compressed[..nb_compressed_bytes]),
        None,
    )
}

/// Ports the 16-bit PCM wrapper `opus_custom_encode()` from `celt/celt_encoder.c`.
pub(crate) fn opus_custom_encode(
    encoder: &mut OpusCustomEncoder<'_>,
    pcm: &[OpusInt16],
    frame_size: usize,
    compressed: &mut [u8],
    nb_compressed_bytes: usize,
) -> Result<usize, CeltEncodeError> {
    let required = required_pcm_samples(encoder.channels, frame_size)?;
    if pcm.len() < required {
        return Err(CeltEncodeError::InsufficientPcm);
    }

    let converted = convert_i16_to_celt_sig(pcm, required);
    encode_with_converted_pcm(
        encoder,
        &converted,
        frame_size,
        compressed,
        nb_compressed_bytes,
    )
}

/// Ports the 24-bit PCM wrapper `opus_custom_encode24()` from `celt/celt_encoder.c`.
pub(crate) fn opus_custom_encode24(
    encoder: &mut OpusCustomEncoder<'_>,
    pcm: &[OpusInt32],
    frame_size: usize,
    compressed: &mut [u8],
    nb_compressed_bytes: usize,
) -> Result<usize, CeltEncodeError> {
    let required = required_pcm_samples(encoder.channels, frame_size)?;
    if pcm.len() < required {
        return Err(CeltEncodeError::InsufficientPcm);
    }

    let converted = convert_i24_to_celt_sig(pcm, required);
    encode_with_converted_pcm(
        encoder,
        &converted,
        frame_size,
        compressed,
        nb_compressed_bytes,
    )
}

/// Ports the float PCM wrapper `opus_custom_encode_float()` from `celt/celt_encoder.c`.
pub(crate) fn opus_custom_encode_float(
    encoder: &mut OpusCustomEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    compressed: &mut [u8],
    nb_compressed_bytes: usize,
) -> Result<usize, CeltEncodeError> {
    let required = required_pcm_samples(encoder.channels, frame_size)?;
    if pcm.len() < required {
        return Err(CeltEncodeError::InsufficientPcm);
    }

    let converted = convert_f32_to_celt_sig(pcm, required);
    encode_with_converted_pcm(
        encoder,
        &converted,
        frame_size,
        compressed,
        nb_compressed_bytes,
    )
}

/// Returns the median of five consecutive logarithmic band energies.
///
/// The helper mirrors `median_of_5()` from `celt/celt_encoder.c` and keeps the
/// branching structure used by the C implementation so future ports that rely
/// on its exact behaviour (such as `dynalloc_analysis()`) observe the same
/// decisions when fed identical inputs.
fn median_of_5(values: &[CeltGlog]) -> CeltGlog {
    debug_assert!(values.len() >= 5);

    let (mut t0, mut t1) = if values[0] > values[1] {
        (values[1], values[0])
    } else {
        (values[0], values[1])
    };
    let t2 = values[2];
    let (mut t3, mut t4) = if values[3] > values[4] {
        (values[4], values[3])
    } else {
        (values[3], values[4])
    };

    if t0 > t3 {
        core::mem::swap(&mut t0, &mut t3);
        core::mem::swap(&mut t1, &mut t4);
    }

    if t2 > t1 {
        if t1 < t3 { t2.min(t3) } else { t4.min(t1) }
    } else if t2 < t3 {
        t1.min(t3)
    } else {
        t2.min(t4)
    }
}

/// Solves the two-tap LPC system used by the tone detector.
///
/// Mirrors the helper of the same name in `celt/celt_encoder.c`. The function
/// accumulates the forward and backward autocorrelation terms for a lag of
/// `delay` samples and applies the covariance method to derive the prediction
/// coefficients. It returns `true` when the linear system is ill-conditioned,
/// matching the non-zero failure return of the C implementation.
pub(crate) fn tone_lpc(x: &[OpusVal16], delay: usize, lpc: &mut [OpusVal32; 2]) -> bool {
    let len = x.len();
    assert!(len > 2 * delay, "tone_lpc requires len > 2 * delay");

    let mut r00 = 0.0f32;
    let mut r01 = 0.0f32;
    let mut r02 = 0.0f32;

    let limit = len - 2 * delay;
    for i in 0..limit {
        let xi = x[i];
        r00 += xi * xi;
        r01 += xi * x[i + delay];
        r02 += xi * x[i + 2 * delay];
    }

    let mut edges = 0.0f32;
    let tail2_base = len - 2 * delay;
    for i in 0..delay {
        let tail = x[tail2_base + i];
        let head = x[i];
        edges += tail * tail - head * head;
    }
    let mut r11 = r00 + edges;

    edges = 0.0;
    let tail1_base = len - delay;
    for i in 0..delay {
        let tail = x[tail1_base + i];
        let head = x[i + delay];
        edges += tail * tail - head * head;
    }
    let r22 = r11 + edges;

    edges = 0.0;
    for i in 0..delay {
        let head0 = x[i];
        let head1 = x[i + delay];
        let tail0 = x[tail2_base + i];
        let tail1 = x[tail1_base + i];
        edges += tail0 * tail1 - head0 * head1;
    }
    let mut r12 = r01 + edges;

    let r00_total = r00 + r22;
    let r01_total = r01 + r12;
    let r11_total = 2.0 * r11;
    let r02_total = 2.0 * r02;
    let r12_total = r12 + r01;

    r00 = r00_total;
    r01 = r01_total;
    r11 = r11_total;
    r02 = r02_total;
    r12 = r12_total;

    let den = (r00 * r11) - (r01 * r01);
    if den < 0.001 * (r00 * r11) {
        return true;
    }

    let num1 = (r02 * r11) - (r01 * r12);
    if num1 >= den {
        lpc[1] = 1.0;
    } else if num1 <= -den {
        lpc[1] = -1.0;
    } else {
        lpc[1] = frac_div32_q29(num1, den);
    }

    let num0 = (r00 * r12) - (r02 * r01);
    if 0.5 * num0 >= den {
        lpc[0] = 1.999_999;
    } else if 0.5 * num0 <= -den {
        lpc[0] = -1.999_999;
    } else {
        lpc[0] = frac_div32_q29(num0, den);
    }

    false
}

/// Detects narrowband tones in the pre-filter input.
///
/// Mirrors `tone_detect()` from `celt/celt_encoder.c`. The helper analyses the
/// pre-emphasised signal, attempting to fit a two-tap LPC model whose complex
/// roots indicate the presence of a strong sinusoid. It returns the detected
/// tone frequency in radians/sample (or `-1.0` when no stable tone is present)
/// and writes a "toneishness" score into `toneishness` so callers can gauge how
/// narrowly peaked the spectrum is.
pub(crate) fn tone_detect(
    input: &[CeltSig],
    channels: usize,
    n: usize,
    toneishness: &mut OpusVal32,
    fs: OpusInt32,
) -> OpusVal16 {
    debug_assert!(channels == 1 || channels == 2);
    debug_assert!(n > 0);
    debug_assert!(input.len() >= channels * n);

    let mut workspace = vec![0.0f32; n];
    if channels == 2 {
        let stride = n;
        for i in 0..n {
            workspace[i] = input[i] + input[stride + i];
        }
    } else {
        workspace.copy_from_slice(&input[..n]);
    }

    normalize_tone_input(&mut workspace);

    let mut lpc = [0.0f32; 2];
    let mut delay = 1usize;
    let mut fail = tone_lpc(&workspace, delay, &mut lpc);
    let mut max_delay = fs.max(0) as usize / 3000;
    if max_delay == 0 {
        max_delay = 1;
    }

    while delay <= max_delay && (fail || (lpc[0] > 1.0 && lpc[1] < 0.0)) {
        delay *= 2;
        if 2 * delay >= n {
            fail = true;
            break;
        }
        fail = tone_lpc(&workspace, delay, &mut lpc);
    }

    if !fail && (lpc[0] * lpc[0] + 3.999_999 * lpc[1]) < 0.0 {
        *toneishness = -lpc[1];
        let angle = {
            #[cfg(feature = "fixed_point")]
            {
                acos_approx(0.5 * lpc[0])
            }
            #[cfg(not(feature = "fixed_point"))]
            {
                acosf(0.5 * lpc[0])
            }
        };
        (angle / delay as OpusVal32) as OpusVal16
    } else {
        *toneishness = 0.0;
        -1.0
    }
}

/// Normalises the tone detector input to avoid overflow in the fixed-point build.
///
/// The C implementation rescales the temporary tone buffer so that the
/// subsequent LPC analysis can square the samples without exceeding the Q15
/// dynamic range. The float variant of CELT performs all computations in
/// `f32`, so no scaling is necessary; the helper only performs work when the
/// crate is compiled with the `fixed_point` feature enabled.
#[cfg(feature = "fixed_point")]
pub(crate) fn normalize_tone_input(x: &mut [OpusVal16]) {
    if x.is_empty() {
        return;
    }

    let mut ac0: OpusInt32 = x.len() as OpusInt32;
    for &sample in x.iter() {
        let sample32: OpusInt32 = sample as OpusInt32;
        ac0 = ac0.wrapping_add((sample32 * sample32) >> 10);
    }

    let shift = 5 - ((28 - celt_ilog2(ac0)) >> 1);
    if shift > 0 {
        let bias = 1 << (shift - 1);
        for sample in x.iter_mut() {
            let value: OpusInt32 = (*sample) as OpusInt32;
            let scaled = (value + bias) >> shift;
            *sample = scaled as OpusVal16;
        }
    }
}

/// Float build stub matching the no-op behaviour of the reference
/// implementation.
#[cfg(not(feature = "fixed_point"))]
pub(crate) fn normalize_tone_input(_x: &mut [OpusVal16]) {}

/// Approximates `acos(x)` using the fixed-point polynomial used by CELT.
///
/// The reference implementation exposes this helper only when operating in
/// fixed-point mode.  Replicating it in Rust keeps the tone detector logic
/// numerically equivalent, including the mirrored handling of negative inputs
/// and the square-root refinement of the polynomial.
#[cfg(feature = "fixed_point")]
pub(crate) fn acos_approx(mut x: OpusVal32) -> OpusVal32 {
    // Emulate the CELT fixed-point acos approximation using integer math.
    // Input `x` is a real value in [-1, 1]. We convert it to Q29, run the
    // original integer polynomial, which produces an angle in Q14 radians,
    // then convert back to `f32`.
    let flip = x < 0.0;
    if flip {
        x = -x;
    }

    // Clamp to [0, 1] and convert to Q29.
    let x_q29: i32 = (x.clamp(0.0, 1.0) * (1u32 << 29) as f32) as i32;

    // Polynomial and refinement in the fixed-point domain.
    let x14: i32 = x_q29 >> 15; // Q14
    let mut tmp: i32 = ((762 * x14) >> 14) - 3_308;
    tmp = ((tmp * x14) >> 14) + 25_726;
    let radicand: i32 = max(0, (1 << 30) - (x_q29 << 1)); // Q30
    tmp = (tmp * celt_sqrt_fixed(radicand)) >> 16; // Q14

    // Mirror negative inputs and convert Q14 -> f32 radians.
    let tmp_q14 = if flip { 25_736 - tmp } else { tmp };
    tmp_q14 as f32 / 16_384.0
}

/// Float variant that falls back to the standard library implementation.
#[cfg(not(feature = "fixed_point"))]
pub(crate) fn acos_approx(x: OpusVal32) -> OpusVal32 {
    acosf(x.clamp(-1.0, 1.0))
}

/// Returns the median of three consecutive logarithmic band energies.
///
/// This mirrors the scalar helper `median_of_3()` from `celt/celt_encoder.c`
/// and provides the same branching behaviour for compatibility with the
/// dynamic allocation heuristics that will be ported later.
fn median_of_3(values: &[CeltGlog]) -> CeltGlog {
    debug_assert!(values.len() >= 3);

    let (t0, t1) = if values[0] > values[1] {
        (values[1], values[0])
    } else {
        (values[0], values[1])
    };
    let t2 = values[2];

    if t1 < t2 {
        t1
    } else if t0 < t2 {
        t2
    } else {
        t0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        COMBFILTER_MAXPERIOD, CeltEncodeError, CeltEncoderAlloc, CeltEncoderInitError, CeltSig,
        EncoderCtlRequest, MAX_CHANNELS, OPUS_BITRATE_MAX, OpusCustomEncoder, OpusCustomMode,
        OpusUint32, PREEMPHASIS_CLIP_LIMIT, celt_encoder_init, celt_preemphasis,
        convert_f32_to_celt_sig, convert_i16_to_celt_sig, convert_i24_to_celt_sig, float2int16,
        opus_custom_encode, opus_custom_encoder_destroy, opus_custom_encoder_init,
        opus_custom_encoder_init_arch,
    };
    use super::{
        CeltEncoderCtlError, alloc_trim_analysis, compute_mdcts, compute_vbr, dynalloc_analysis,
        l1_metric, median_of_3, median_of_5, opus_custom_encoder_ctl, patch_transient_decision,
        stereo_analysis, tf_analysis, tf_encode, tone_detect, tone_lpc, transient_analysis,
    };
    #[cfg(not(feature = "fixed_point"))]
    use super::{acos_approx, normalize_tone_input};
    use crate::celt::OpusVal16;
    use crate::celt::celt::TF_SELECT_TABLE;
    use crate::celt::cpu_support::opus_select_arch;
    use crate::celt::entenc::EcEnc;
    use crate::celt::float_cast::CELT_SIG_SCALE;
    use crate::celt::math::celt_log2;
    use crate::celt::modes::{
        compute_preemphasis, opus_custom_mode_create, opus_custom_mode_find_static,
    };
    use crate::celt::types::{AnalysisInfo, OpusRes};
    use crate::celt::vq::SPREAD_NORMAL;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::{FRAC_1_SQRT_2, PI};
    use libm::floorf;
    use libm::sinf;

    const EPSILON: f32 = 1e-6;

    fn assert_slice_close(actual: &[CeltSig], expected: &[CeltSig]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < EPSILON,
                "mismatch at index {index}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn l1_metric_matches_reference_bias() {
        let tmp: [OpusVal16; 4] = [1.0, -2.0, 0.5, -0.25];

        let unbiased = l1_metric(&tmp, tmp.len(), 0, 0.125);
        assert!((unbiased - 3.75).abs() < EPSILON);

        let biased = l1_metric(&tmp, tmp.len(), 2, 0.5);
        assert!((biased - 7.5).abs() < EPSILON);
    }

    #[test]
    fn tf_analysis_prefers_frequency_resolution_for_flat_spectrum() {
        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let lm = 0;
        let len = 4;
        let n0 = (mode.e_bands[len] as usize) << lm;
        let x = vec![0.0; n0];
        let mut tf_res = vec![1; len];
        let importance = vec![1; len];

        let tf_select = tf_analysis(
            &mode,
            len,
            false,
            &mut tf_res,
            100,
            &x,
            n0,
            lm,
            0.0,
            0,
            &importance,
        );

        assert_eq!(tf_select, 0);
        assert!(tf_res.iter().take(len).all(|&value| value == 0));
    }

    #[test]
    fn tf_analysis_enables_tf_select_for_transient_pattern() {
        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let lm = 1;
        let len = 9;
        let n0 = (mode.e_bands[len] as usize) << lm;
        let mut x = vec![0.0; n0];
        let pattern = [-9.119_444, -7.347_349, 9.822_017, -6.768_198];
        let start = (mode.e_bands[8] as usize) << lm;
        x[start..start + pattern.len()].copy_from_slice(&pattern);
        let mut tf_res = vec![0; len];
        let importance = vec![1; len];

        let tf_select = tf_analysis(
            &mode,
            len,
            true,
            &mut tf_res,
            80,
            &x,
            n0,
            lm,
            0.0,
            0,
            &importance,
        );

        assert_eq!(tf_select, 1);
        assert_eq!(tf_res[8], 1);
    }

    #[test]
    fn compute_mdcts_matches_manual_mdct() {
        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let short_blocks = 0;
        let lm = 0;
        let upsample = 1;
        let total_channels = 2;
        let coded_channels = 2;
        let block_count = 1;
        let shift = mode.max_lm - lm;
        let transform_len = mode.mdct.effective_len(shift);
        let frame_len = transform_len >> 1;
        let overlap = mode.overlap;
        let channel_input_stride = block_count * transform_len + overlap;
        let channel_output_stride = block_count * frame_len;
        let mut input = vec![0.0; total_channels * channel_input_stride];
        for (index, sample) in input.iter_mut().enumerate() {
            *sample = index as f32;
        }

        let mut expected = vec![0.0; total_channels * channel_output_stride];
        for channel in 0..total_channels {
            let input_offset = channel * channel_input_stride;
            let output_offset = channel * channel_output_stride;
            crate::celt::mdct::clt_mdct_forward(
                &mode.mdct,
                &input[input_offset..input_offset + overlap + transform_len],
                &mut expected[output_offset..output_offset + channel_output_stride],
                mode.window,
                overlap,
                shift,
                block_count,
            );
        }

        let mut output = vec![0.0; total_channels * channel_output_stride];
        compute_mdcts(
            &mode,
            short_blocks,
            &input,
            &mut output,
            coded_channels,
            total_channels,
            lm,
            upsample,
            0,
        );

        assert_slice_close(&output, &expected);
    }

    #[test]
    fn compute_mdcts_downmixes_stereo() {
        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let short_blocks = 0;
        let lm = 0;
        let upsample = 1;
        let total_channels = 2;
        let block_count = 1;
        let shift = mode.max_lm - lm;
        let transform_len = mode.mdct.effective_len(shift);
        let frame_len = transform_len >> 1;
        let overlap = mode.overlap;
        let channel_input_stride = block_count * transform_len + overlap;
        let channel_output_stride = block_count * frame_len;
        let mut input = vec![0.0; total_channels * channel_input_stride];
        for (index, sample) in input.iter_mut().enumerate() {
            *sample = (index as f32) / 5.0;
        }

        let mut stereo = vec![0.0; total_channels * channel_output_stride];
        compute_mdcts(
            &mode,
            short_blocks,
            &input,
            &mut stereo,
            total_channels,
            total_channels,
            lm,
            upsample,
            0,
        );

        let mut mono = vec![0.0; total_channels * channel_output_stride];
        compute_mdcts(
            &mode,
            short_blocks,
            &input,
            &mut mono,
            1,
            total_channels,
            lm,
            upsample,
            0,
        );

        for i in 0..block_count * frame_len {
            let expected = 0.5 * (stereo[i] + stereo[block_count * frame_len + i]);
            assert!((mono[i] - expected).abs() < EPSILON);
        }
    }

    #[test]
    fn compute_mdcts_scales_for_upsampling() {
        let mode = opus_custom_mode_find_static(48_000, 960).expect("static mode");
        let short_blocks = 0;
        let lm = 0;
        let total_channels = 1;
        let block_count = 1;
        let shift = mode.max_lm - lm;
        let transform_len = mode.mdct.effective_len(shift);
        let frame_len = transform_len >> 1;
        let overlap = mode.overlap;
        let channel_input_stride = block_count * transform_len + overlap;
        let channel_output_stride = block_count * frame_len;
        let mut input = vec![0.0; total_channels * channel_input_stride];
        for (index, sample) in input.iter_mut().enumerate() {
            *sample = (index as f32) / 7.0;
        }

        let mut baseline = vec![0.0; total_channels * channel_output_stride];
        compute_mdcts(
            &mode,
            short_blocks,
            &input,
            &mut baseline,
            total_channels,
            total_channels,
            lm,
            1,
            0,
        );

        let upsample = 2;
        let mut scaled = vec![0.0; total_channels * channel_output_stride];
        compute_mdcts(
            &mode,
            short_blocks,
            &input,
            &mut scaled,
            total_channels,
            total_channels,
            lm,
            upsample,
            0,
        );

        let bound = block_count * frame_len / upsample;
        for i in 0..bound {
            assert!((scaled[i] - baseline[i] * upsample as f32).abs() < EPSILON);
        }
        for value in &scaled[bound..block_count * frame_len] {
            assert!(value.abs() < EPSILON);
        }
    }

    #[test]
    fn tf_encode_applies_select_when_budget_allows() {
        let mut buffer = [0u8; 16];
        let mut enc = EcEnc::new(&mut buffer);
        let mut tf_res = [0, 1, 1, 0];

        tf_encode(0, tf_res.len(), false, &mut tf_res, 1, 1, &mut enc);

        let expected = [
            i32::from(TF_SELECT_TABLE[1][2]),
            i32::from(TF_SELECT_TABLE[1][3]),
            i32::from(TF_SELECT_TABLE[1][3]),
            i32::from(TF_SELECT_TABLE[1][2]),
        ];
        assert_eq!(tf_res, expected);
    }

    #[test]
    fn tf_encode_clamps_to_previous_when_budget_is_exhausted() {
        let mut buffer = [0u8; 0];
        let mut enc = EcEnc::new(&mut buffer);
        let mut tf_res = [1, 0];

        tf_encode(0, tf_res.len(), false, &mut tf_res, 0, 1, &mut enc);

        assert_eq!(tf_res, [0, 0]);
    }

    #[test]
    fn celt_preemphasis_fast_path_matches_reference() {
        let coef = compute_preemphasis(48_000);
        let pcm: [OpusRes; 4] = [0.0, 0.25, -0.5, 1.0];
        let n = pcm.len();
        let mut output = vec![0.0; n];
        let mut expected = vec![0.0; n];
        let mut state = 0.0;
        let mut expected_state = state;

        for i in 0..n {
            let x = pcm[i] * CELT_SIG_SCALE;
            expected[i] = x - expected_state;
            expected_state = coef[0] * x;
        }

        celt_preemphasis(&pcm, &mut output, n, 1, 1, &coef, &mut state, false);

        assert_slice_close(&output, &expected);
        assert!((state - expected_state).abs() < EPSILON);
    }

    #[test]
    fn celt_preemphasis_handles_upsampling_and_clipping() {
        let coef = compute_preemphasis(48_000);
        let n = 6;
        let upsample = 2;
        let channels = 2;
        let pcm: [OpusRes; 6] = [1.0, 3.5, -2.0, -4.0, 0.25, -0.75];
        let pcmp = &pcm[1..];
        let mut output = vec![42.0; n];
        let mut expected = vec![0.0; n];
        let mut state = 123.0;
        let mut expected_state = state;

        let nu = n / upsample;
        expected.fill(0.0);
        for i in 0..nu {
            let sample = pcmp[channels * i];
            expected[i * upsample] = sample * CELT_SIG_SCALE;
        }
        for i in 0..nu {
            let index = i * upsample;
            expected[index] =
                expected[index].clamp(-PREEMPHASIS_CLIP_LIMIT, PREEMPHASIS_CLIP_LIMIT);
        }
        for value in &mut expected[..n] {
            let x = *value;
            *value = x - expected_state;
            expected_state = coef[0] * x;
        }

        celt_preemphasis(
            pcmp,
            &mut output,
            n,
            channels,
            upsample,
            &coef,
            &mut state,
            true,
        );

        assert_slice_close(&output, &expected);
        assert!((state - expected_state).abs() < EPSILON);
    }

    #[test]
    fn celt_preemphasis_three_tap_path_matches_reference() {
        let coef = compute_preemphasis(16_000);
        let pcm: [OpusRes; 5] = [0.5, -0.25, 0.75, -0.5, 0.0];
        let n = pcm.len();
        let mut output = vec![0.0; n];
        let mut expected = vec![0.0; n];
        let mut state = -321.0;
        let mut expected_state = state;

        for i in 0..n {
            expected[i] = pcm[i] * CELT_SIG_SCALE;
        }
        for value in &mut expected {
            let x = *value;
            let tmp = coef[2] * x;
            *value = tmp + expected_state;
            expected_state = coef[1] * *value - coef[0] * tmp;
        }

        celt_preemphasis(&pcm, &mut output, n, 1, 1, &coef, &mut state, false);

        assert_slice_close(&output, &expected);
        assert!((state - expected_state).abs() < EPSILON);
    }

    fn get_lsb_depth(encoder: &mut OpusCustomEncoder<'_>) -> i32 {
        let mut value = 0;
        opus_custom_encoder_ctl(encoder, EncoderCtlRequest::GetLsbDepth(&mut value)).unwrap();
        value
    }

    fn get_phase_disabled(encoder: &mut OpusCustomEncoder<'_>) -> bool {
        let mut value = false;
        opus_custom_encoder_ctl(
            encoder,
            EncoderCtlRequest::GetPhaseInversionDisabled(&mut value),
        )
        .unwrap();
        value
    }

    fn get_final_range(encoder: &mut OpusCustomEncoder<'_>) -> OpusUint32 {
        let mut value = 0;
        opus_custom_encoder_ctl(encoder, EncoderCtlRequest::GetFinalRange(&mut value)).unwrap();
        value
    }

    fn assert_mode_matches(encoder: &mut OpusCustomEncoder<'_>, expected: *const OpusCustomMode) {
        let mut slot = None;
        opus_custom_encoder_ctl(encoder, EncoderCtlRequest::GetMode(&mut slot)).unwrap();
        let mode_ref = slot.expect("mode");
        assert_eq!(mode_ref as *const OpusCustomMode, expected);
    }

    #[test]
    fn transient_analysis_outputs_valid_metrics() {
        let len = 64;
        let mut input = vec![0.0f32; len];
        input[0] = 10.0;
        let mut tf_estimate = 0.0f32;
        let mut tf_chan = 0usize;
        let mut weak = false;

        let _detected = transient_analysis(
            &input,
            len,
            1,
            &mut tf_estimate,
            &mut tf_chan,
            false,
            &mut weak,
            0.1,
            0.0,
        );

        assert!(tf_estimate >= 0.0);
        assert_eq!(tf_chan, 0);
        assert!(!weak);
    }

    #[test]
    fn transient_analysis_rejects_flat_signal() {
        let len = 64;
        let input = vec![0.5f32; len];
        let mut tf_estimate = 0.0f32;
        let mut tf_chan = 0usize;
        let mut weak = false;

        let detected = transient_analysis(
            &input,
            len,
            1,
            &mut tf_estimate,
            &mut tf_chan,
            true,
            &mut weak,
            0.0,
            1.0,
        );

        assert!(!detected);
        assert!(!weak);
        assert_eq!(tf_chan, 0);
        assert!(tf_estimate >= 0.0);
    }

    #[test]
    fn patch_transient_decision_returns_boolean() {
        let nb_ebands = 5;
        let start = 0;
        let end = nb_ebands;
        let channels = 1;
        let mut new_e = vec![0.0f32; nb_ebands];
        let mut old_e = vec![0.0f32; nb_ebands];

        old_e.fill(-2.0);
        new_e.fill(-2.0);
        new_e[2] = 2.0;

        let increase = patch_transient_decision(&new_e, &old_e, nb_ebands, start, end, channels);
        let baseline = patch_transient_decision(&old_e, &old_e, nb_ebands, start, end, channels);

        assert!(u8::from(increase) >= u8::from(baseline));
    }

    #[test]
    fn dynalloc_analysis_defaults_when_disabled() {
        let nb_ebands = 4;
        let channels = 1;
        let start = 0;
        let end = nb_ebands;
        let band_log_e = vec![0.5f32; channels * nb_ebands];
        let band_log_e2 = band_log_e.clone();
        let old_band_e = vec![-28.0f32; channels * nb_ebands];
        let log_n = vec![10i16; nb_ebands];
        let e_bands = [0i16, 1, 2, 3, 4];
        let mut offsets = vec![1i32; nb_ebands];
        let mut importance = vec![0i32; nb_ebands];
        let mut spread_weight = vec![0i32; nb_ebands];
        let mut surround_dynalloc = vec![0.0f32; nb_ebands];
        let mut tot_boost = -1;

        let max_depth = dynalloc_analysis(
            &band_log_e,
            &band_log_e2,
            &old_band_e,
            nb_ebands,
            start,
            end,
            channels,
            &mut offsets,
            8,
            &log_n,
            false,
            true,
            false,
            &e_bands,
            0,
            10,
            &mut tot_boost,
            false,
            &mut surround_dynalloc,
            &AnalysisInfo::default(),
            &mut importance,
            &mut spread_weight,
            0.0,
            0.0,
        );

        assert!(max_depth > 0.0);
        assert_eq!(tot_boost, 0);
        assert!(offsets.iter().all(|&value| value == 0));
        assert_eq!(&importance[start..end], &[13, 13, 13, 13]);
    }

    #[test]
    fn dynalloc_analysis_accounts_for_surround_boost() {
        let nb_ebands = 4;
        let channels = 1;
        let start = 0;
        let end = nb_ebands;
        let band_log_e = vec![6.0f32; channels * nb_ebands];
        let band_log_e2 = band_log_e.clone();
        let old_band_e = vec![5.0f32; channels * nb_ebands];
        let log_n = vec![8i16; nb_ebands];
        let e_bands = [0i16, 1, 2, 3, 4];
        let mut offsets = vec![0i32; nb_ebands];
        let mut importance = vec![0i32; nb_ebands];
        let mut spread_weight = vec![0i32; nb_ebands];
        let mut surround_dynalloc = vec![0.0f32; nb_ebands];
        surround_dynalloc[0] = 3.0;
        let mut tot_boost = 0;

        let toneishness = 0.0f32;
        let max_depth = dynalloc_analysis(
            &band_log_e,
            &band_log_e2,
            &old_band_e,
            nb_ebands,
            start,
            end,
            channels,
            &mut offsets,
            6,
            &log_n,
            false,
            true,
            false,
            &e_bands,
            0,
            60,
            &mut tot_boost,
            false,
            &mut surround_dynalloc,
            &AnalysisInfo::default(),
            &mut importance,
            &mut spread_weight,
            0.05,
            toneishness,
        );

        assert!(max_depth > 0.0);
        assert!(importance[0] > 13);
        assert_eq!(offsets[0], 4);
        assert_eq!(tot_boost, 32);
    }

    #[test]
    fn stereo_analysis_matches_manual_decision() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let lm = 0usize;
        let n0 = mode.short_mdct_size << lm;
        let mut x = vec![0.0f32; 2 * n0];
        for i in 0..n0 {
            let sample = ((i % 7) as f32 - 3.0) * 0.125;
            x[i] = sample;
            x[n0 + i] = 0.6 * sample;
        }

        let result = stereo_analysis(&mode, &x, lm, n0);

        let mut sum_lr = 1.0e-15f32;
        let mut sum_ms = 1.0e-15f32;
        for band in 0..13 {
            let start = (mode.e_bands[band] as usize) << lm;
            let end = (mode.e_bands[band + 1] as usize) << lm;
            for idx in start..end {
                let left = x[idx];
                let right = x[n0 + idx];
                let mid = left + right;
                let side = left - right;
                sum_lr += left.abs() + right.abs();
                sum_ms += mid.abs() + side.abs();
            }
        }

        sum_ms *= FRAC_1_SQRT_2;
        let mut thetas = 13i32;
        if lm <= 1 {
            thetas -= 8;
        }
        let base = i32::from(mode.e_bands[13]) << (lm + 1);
        let expected = (base + thetas) as f32 * sum_ms > base as f32 * sum_lr;

        assert_eq!(result, expected);
    }

    #[test]
    fn alloc_trim_analysis_matches_reference_flow() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let lm = 0usize;
        let n0 = mode.short_mdct_size << lm;
        let channels = 2;
        let mut x = vec![0.0f32; channels * n0];
        for i in 0..n0 {
            let sample = (0.005 * i as f32).sin();
            x[i] = sample;
            x[n0 + i] = 0.5 * sample + 0.1;
        }

        let nb_ebands = mode.num_ebands;
        let mut band_log_e = vec![0.0f32; channels * nb_ebands];
        for c in 0..channels {
            for b in 0..nb_ebands {
                band_log_e[c * nb_ebands + b] = 0.1 * (b as f32 + c as f32);
            }
        }

        let mut analysis = AnalysisInfo::default();
        analysis.valid = true;
        analysis.tonality_slope = 0.075;

        let mut stereo_saving = 0.0f32;
        let tf_estimate = 0.35;
        let surround_trim = 0.2;
        let end = nb_ebands.min(15);
        let intensity = end;
        let equiv_rate = 72_000;

        let trim_index = alloc_trim_analysis(
            &mode,
            &x,
            &band_log_e,
            end,
            lm,
            channels,
            n0,
            &analysis,
            &mut stereo_saving,
            tf_estimate,
            intensity,
            surround_trim,
            equiv_rate,
            0,
        );

        let mut expected_trim = if equiv_rate < 64_000 {
            4.0
        } else if equiv_rate < 80_000 {
            4.0 + ((equiv_rate - 64_000) >> 10) as f32 / 16.0
        } else {
            5.0
        };

        let mut sum = 0.0f32;
        for band in 0..8.min(mode.num_ebands) {
            let start = (mode.e_bands[band] as usize) << lm;
            let end = (mode.e_bands[band + 1] as usize) << lm;
            for idx in start..end {
                sum += x[idx] * x[n0 + idx];
            }
        }
        sum *= 1.0 / 8.0;
        sum = sum.abs().min(1.0);
        let mut min_xc = sum;
        for band in 8..intensity.min(mode.num_ebands) {
            let start = (mode.e_bands[band] as usize) << lm;
            let end = (mode.e_bands[band + 1] as usize) << lm;
            for idx in start..end {
                let partial = (x[idx] * x[n0 + idx]).abs().min(1.0);
                if partial < min_xc {
                    min_xc = partial;
                }
            }
        }

        let log_xc = celt_log2(1.001 - sum * sum);
        let alt = celt_log2(1.001 - min_xc * min_xc);
        let half_log = 0.5 * log_xc;
        let log_xc2 = if alt > half_log { alt } else { half_log };
        expected_trim += (0.75 * log_xc).max(-4.0);
        let expected_stereo = (-0.5 * log_xc2).min(0.25);

        let mut diff = 0.0f32;
        if end > 1 {
            for c in 0..channels {
                let base = c * nb_ebands;
                for band in 0..(end - 1) {
                    let weight = (2 + 2 * band as i32 - end as i32) as f32;
                    diff += band_log_e[base + band] * weight;
                }
            }
            diff /= (channels * (end - 1)) as f32;
        }

        expected_trim -= ((diff + 1.0) / 6.0).clamp(-2.0, 2.0);
        expected_trim -= surround_trim;
        expected_trim -= 2.0 * tf_estimate;
        if analysis.valid {
            let tonal = 2.0 * (analysis.tonality_slope + 0.05);
            expected_trim -= tonal.clamp(-2.0, 2.0);
        }

        let mut expected_index = floorf(expected_trim + 0.5) as i32;
        expected_index = expected_index.clamp(0, 10);

        assert_eq!(trim_index, expected_index);
        assert!(
            (stereo_saving - expected_stereo).abs() < 1e-6,
            "stereo_saving={} expected={}",
            stereo_saving,
            expected_stereo
        );
    }

    #[test]
    fn compute_vbr_penalises_quiet_analysis() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut analysis = AnalysisInfo::default();
        analysis.valid = true;
        analysis.activity = 0.0;

        let base_target = 12_000;
        let target = compute_vbr(
            &mode,
            &analysis,
            base_target,
            0,
            64_000,
            0,
            1,
            mode.effective_ebands as i32,
            false,
            0.0,
            0,
            0.0,
            false,
            10.0,
            false,
            false,
            0.0,
            0.0,
        );

        assert!(target < base_target);
    }

    #[test]
    fn compute_vbr_caps_to_twice_base() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut analysis = AnalysisInfo::default();
        analysis.valid = true;
        analysis.tonality = 1.0;
        analysis.activity = 0.5;

        let base_target = 10_000;
        let target = compute_vbr(
            &mode,
            &analysis,
            base_target,
            0,
            64_000,
            0,
            2,
            mode.effective_ebands as i32,
            false,
            1.0,
            2_000,
            1.0,
            true,
            10.0,
            false,
            false,
            0.0,
            0.5,
        );

        assert!(target <= base_target * 2);
    }

    #[test]
    fn allocation_matches_reference_layout() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let alloc = CeltEncoderAlloc::new(&mode, 2);

        assert_eq!(alloc.in_mem.len(), mode.overlap * 2);
        assert_eq!(alloc.prefilter_mem.len(), 2 * COMBFILTER_MAXPERIOD);
        assert_eq!(alloc.old_band_e.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e2.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.energy_error.len(), 2 * mode.num_ebands);

        let bytes = alloc.size_in_bytes();
        assert_eq!(bytes, super::opus_custom_encoder_get_size(&mode, 2));
    }

    #[test]
    fn reset_initialises_energy_histories() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        alloc.old_log_e.fill(0.5);
        alloc.old_log_e2.fill(0.25);
        alloc.old_band_e.fill(1.0);
        alloc.energy_error.fill(1.0);
        alloc.prefilter_mem.fill(1.0);
        alloc.in_mem.fill(1.0);

        alloc.reset();

        assert!(alloc.in_mem.iter().all(|&v| v == 0.0));
        assert!(alloc.prefilter_mem.iter().all(|&v| v == 0.0));
        assert!(alloc.old_band_e.iter().all(|&v| v == 0.0));
        assert!(alloc.energy_error.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e.iter().all(|&v| (v + 28.0).abs() < 1e-6));
        assert!(alloc.old_log_e2.iter().all(|&v| (v + 28.0).abs() < 1e-6));
    }

    #[test]
    fn init_custom_encoder_sets_reference_defaults() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 2);

        let encoder = alloc
            .init_custom_encoder(&mode, 2, 2, 0xDEADBEEF)
            .expect("encoder");

        assert_eq!(encoder.channels, 2);
        assert_eq!(encoder.stream_channels, 2);
        assert_eq!(encoder.upsample, 1);
        assert_eq!(encoder.start_band, 0);
        assert_eq!(encoder.end_band, mode.effective_ebands as i32);
        assert_eq!(encoder.signalling, 1);
        assert_eq!(encoder.arch, opus_select_arch());
        assert!(encoder.constrained_vbr);
        assert!(encoder.clip);
        assert_eq!(encoder.bitrate, OPUS_BITRATE_MAX);
        assert!(!encoder.use_vbr);
        assert_eq!(encoder.complexity, 5);
        assert_eq!(encoder.lsb_depth, 24);
        assert_eq!(encoder.spread_decision, SPREAD_NORMAL);
        assert!((encoder.delayed_intra - 1.0).abs() < 1e-6);
        assert_eq!(encoder.tonal_average, 256);
        assert_eq!(encoder.hf_average, 0);
        assert_eq!(encoder.tapset_decision, 0);
        assert_eq!(encoder.rng, 0xDEADBEEF);
        assert!(encoder.old_log_e.iter().all(|&v| (v + 28.0).abs() < 1e-6));
        assert!(encoder.old_log_e2.iter().all(|&v| (v + 28.0).abs() < 1e-6));
    }

    #[test]
    fn opus_custom_encoder_init_arch_honours_requested_architecture() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        let encoder =
            opus_custom_encoder_init_arch(&mut alloc, &mode, 1, 11, 1234).expect("encoder");

        assert_eq!(encoder.arch, 11);
        assert_eq!(encoder.stream_channels, 1);
    }

    #[test]
    fn opus_custom_encoder_init_defaults_stream_channels() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 2);

        let encoder = opus_custom_encoder_init(&mut alloc, &mode, 2, 77).expect("encoder");

        assert_eq!(encoder.channels, 2);
        assert_eq!(encoder.stream_channels, 2);
        assert_eq!(encoder.arch, opus_select_arch());
    }

    #[test]
    fn celt_encoder_init_sets_resampling_factor() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        let encoder = celt_encoder_init(&mut alloc, 24_000, 1, 3, 0).expect("encoder");

        assert_eq!(encoder.upsample, 2);
        assert_eq!(encoder.arch, 3);
    }

    #[test]
    fn celt_encoder_init_rejects_invalid_rate() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        let err = celt_encoder_init(&mut alloc, 44_100, 1, 0, 0).unwrap_err();
        assert_eq!(err, CeltEncoderInitError::UnsupportedSampleRate);
    }

    #[test]
    fn opus_custom_encoder_destroy_drops_allocation() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let alloc = CeltEncoderAlloc::new(&mode, 1);

        opus_custom_encoder_destroy(alloc);
    }

    #[test]
    fn init_encoder_for_rate_rejects_unsupported_sampling_rate() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        let err = alloc
            .init_encoder_for_rate(&mode, 1, 1, 44_100, 0)
            .unwrap_err();
        assert_eq!(err, CeltEncoderInitError::UnsupportedSampleRate);
    }

    #[test]
    fn channel_validation_matches_reference_limits() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        let err = alloc.init_custom_encoder(&mode, 0, 0, 0).unwrap_err();
        assert_eq!(err, CeltEncoderInitError::InvalidChannelCount);

        let err = alloc
            .init_custom_encoder(&mode, MAX_CHANNELS + 1, 1, 0)
            .unwrap_err();
        assert_eq!(err, CeltEncoderInitError::InvalidChannelCount);

        let err = alloc.init_custom_encoder(&mode, 1, 0, 0).unwrap_err();
        assert_eq!(err, CeltEncoderInitError::InvalidStreamChannels);

        let err = alloc.init_custom_encoder(&mode, 2, 3, 0).unwrap_err();
        assert_eq!(err, CeltEncoderInitError::InvalidStreamChannels);
    }

    #[test]
    fn ctl_updates_encoder_state() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 2);
        let mut encoder = alloc
            .init_custom_encoder(&mode, 2, 2, 1234)
            .expect("encoder");

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetComplexity(7)).unwrap();
        assert_eq!(encoder.complexity, 7);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetStartBand(1)).unwrap();
        assert_eq!(encoder.start_band, 1);

        opus_custom_encoder_ctl(
            &mut encoder,
            EncoderCtlRequest::SetEndBand(mode.num_ebands as i32),
        )
        .unwrap();
        assert_eq!(encoder.end_band, mode.num_ebands as i32);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetPrediction(2)).unwrap();
        assert!(!encoder.disable_prefilter);
        assert!(!encoder.force_intra);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetPacketLossPerc(25)).unwrap();
        assert_eq!(encoder.loss_rate, 25);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetVbrConstraint(false)).unwrap();
        assert!(!encoder.constrained_vbr);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetVbr(true)).unwrap();
        assert!(encoder.use_vbr);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetBitrate(700_000)).unwrap();
        assert_eq!(encoder.bitrate, 260_000 * 2);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetChannels(1)).unwrap();
        assert_eq!(encoder.stream_channels, 1);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetLsbDepth(16)).unwrap();
        assert_eq!(encoder.lsb_depth, 16);

        opus_custom_encoder_ctl(
            &mut encoder,
            EncoderCtlRequest::SetPhaseInversionDisabled(true),
        )
        .unwrap();
        assert!(encoder.disable_inv);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetInputClipping(false)).unwrap();
        assert!(!encoder.clip);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetSignalling(0)).unwrap();
        assert_eq!(encoder.signalling, 0);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetLfe(true)).unwrap();
        assert!(encoder.lfe);

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetEnergyMask(None)).unwrap();
        assert!(encoder.energy_mask.is_none());

        let lsb_depth = get_lsb_depth(&mut encoder);
        assert_eq!(lsb_depth, encoder.lsb_depth);

        let phase_disabled = get_phase_disabled(&mut encoder);
        assert!(phase_disabled);

        let final_range = get_final_range(&mut encoder);
        assert_eq!(final_range, encoder.rng);

        encoder.rng = 42;
        encoder.old_log_e.fill(0.0);
        encoder.energy_mask = Some(&[]);
        let expected_mode_ptr = encoder.mode as *const OpusCustomMode;
        assert_mode_matches(&mut encoder, expected_mode_ptr);
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::ResetState).unwrap();
        assert_eq!(encoder.rng, 0);
        assert!(encoder.old_log_e.iter().all(|&v| (v + 28.0).abs() < 1e-6));
        assert!(encoder.energy_mask.is_none());
    }

    #[test]
    fn ctl_validates_arguments() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);
        let mut encoder = alloc
            .init_custom_encoder(&mode, 1, 1, 9876)
            .expect("encoder");

        let err = opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetComplexity(11))
            .unwrap_err();
        assert_eq!(err, CeltEncoderCtlError::InvalidArgument);

        let err =
            opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetStartBand(-1)).unwrap_err();
        assert_eq!(err, CeltEncoderCtlError::InvalidArgument);

        let err =
            opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetEndBand(0)).unwrap_err();
        assert_eq!(err, CeltEncoderCtlError::InvalidArgument);

        let err =
            opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetPrediction(3)).unwrap_err();
        assert_eq!(err, CeltEncoderCtlError::InvalidArgument);

        let err = opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetPacketLossPerc(101))
            .unwrap_err();
        assert_eq!(err, CeltEncoderCtlError::InvalidArgument);

        let err =
            opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetBitrate(400)).unwrap_err();
        assert_eq!(err, CeltEncoderCtlError::InvalidArgument);

        let err =
            opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetChannels(2)).unwrap_err();
        assert_eq!(err, CeltEncoderCtlError::InvalidArgument);

        let err =
            opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetLsbDepth(6)).unwrap_err();
        assert_eq!(err, CeltEncoderCtlError::InvalidArgument);
    }

    #[test]
    fn opus_custom_encode_errors_on_short_pcm() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);
        let mut encoder = alloc
            .init_custom_encoder(&mode, 1, 1, 123)
            .expect("encoder");

        let pcm = vec![0i16; 100];
        let mut compressed = vec![0u8; 16];
        let limit = compressed.len();

        let err = opus_custom_encode(&mut encoder, &pcm, 960, &mut compressed, limit).unwrap_err();
        assert_eq!(err, CeltEncodeError::InsufficientPcm);
    }

    #[test]
    fn opus_custom_encode_errors_when_output_too_small() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);
        let mut encoder = alloc
            .init_custom_encoder(&mode, 1, 1, 234)
            .expect("encoder");

        let pcm = vec![0i16; 960];
        let mut compressed = vec![0u8; 8];
        let err = opus_custom_encode(&mut encoder, &pcm, 960, &mut compressed, 16).unwrap_err();
        assert_eq!(err, CeltEncodeError::MissingOutput);
    }

    #[test]
    fn opus_custom_encode_errors_when_nb_compressed_bytes_below_minimum() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);
        let mut encoder = alloc
            .init_custom_encoder(&mode, 1, 1, 345)
            .expect("encoder");

        let pcm = vec![0i16; 960];
        let mut compressed = vec![0u8; 16];
        let err = opus_custom_encode(&mut encoder, &pcm, 960, &mut compressed, 1).unwrap_err();
        assert_eq!(err, CeltEncodeError::MissingOutput);
    }

    #[test]
    fn convert_i16_to_celt_sig_preserves_values() {
        let input = [0i16, -32_768, 32_767, 1_234];
        let converted = convert_i16_to_celt_sig(&input, input.len());
        assert_eq!(converted, vec![0.0, -32_768.0, 32_767.0, 1_234.0]);
    }

    #[test]
    fn convert_i24_to_celt_sig_matches_reference_shift() {
        let input = [0i32, 100_000, -150_000, 255, -255, 8_388_607, -8_388_608];
        let converted = convert_i24_to_celt_sig(&input, input.len());
        assert_eq!(
            converted,
            vec![0.0, 391.0, -586.0, 1.0, -1.0, 32_767.0, -32_768.0]
        );
    }

    #[test]
    fn convert_f32_to_celt_sig_matches_float2int16() {
        let input = [0.0f32, 0.5 / CELT_SIG_SCALE, -1.5];
        let converted = convert_f32_to_celt_sig(&input, input.len());
        let expected: Vec<CeltSig> = input
            .iter()
            .map(|&sample| CeltSig::from(float2int16(sample)))
            .collect();
        assert_eq!(converted, expected);
    }

    #[test]
    fn median_of_5_matches_sorted_middle() {
        let samples = [
            [1.0f32, 5.0, 3.0, 2.0, 4.0],
            [9.0, -1.0, 2.0, 2.0, 8.0],
            [12.0, 12.0, 11.0, 13.0, 12.5],
        ];

        for data in samples {
            let mut sorted = data;
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
            let expected = sorted[2];
            assert_eq!(median_of_5(&data), expected);
        }
    }

    #[test]
    fn median_of_3_selects_middle_value() {
        let samples = [[1.0f32, 3.0, 2.0], [5.0, -2.0, 4.0], [7.5, 7.5, 7.0]];

        for data in samples {
            let mut sorted = data;
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
            let expected = sorted[1];
            assert_eq!(median_of_3(&data), expected);
        }
    }

    #[test]
    fn tone_lpc_recovers_sinusoid_predictor() {
        let len = 240;
        let delay = 1;
        let mut samples = vec![0.0f32; len];
        let omega = 2.0 * PI * 0.1;
        for (n, slot) in samples.iter_mut().enumerate() {
            *slot = (omega * n as f32).sin();
        }

        let mut lpc = [0.0f32; 2];
        let failed = tone_lpc(&samples, delay, &mut lpc);
        assert!(
            !failed,
            "tone_lpc should succeed for a well-conditioned tone"
        );

        let expected_cos = 2.0 * omega.cos();
        assert!((lpc[0] - expected_cos).abs() < 1e-3);
        assert!((lpc[1] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn tone_detect_identifies_sinusoid() {
        let fs = 48_000;
        let n = 960;
        let target_hz = 440.0;
        let omega = 2.0 * PI * target_hz / fs as f32;
        let mut input = vec![0.0f32; n];
        for (i, sample) in input.iter_mut().enumerate() {
            *sample = sinf(omega * i as f32);
        }

        let mut toneishness = 0.0f32;
        let freq = tone_detect(&input, 1, n, &mut toneishness, fs);

        assert!(freq > 0.0, "freq {freq} omega {omega}");
        assert!(freq < 0.1, "freq {freq} omega {omega}");
        assert!(toneishness > 0.8);
    }

    #[test]
    fn tone_detect_rejects_silence() {
        let n = 240;
        let input = vec![0.0f32; n];
        let mut toneishness = 1.0f32;
        let freq = tone_detect(&input, 1, n, &mut toneishness, 48_000);

        assert_eq!(freq, -1.0);
        assert_eq!(toneishness, 0.0);
    }

    #[cfg(not(feature = "fixed_point"))]
    #[test]
    fn normalize_tone_input_is_noop_for_float_build() {
        let mut data = [0.125f32, -0.5, 1.25, -1.75];
        let original = data;
        normalize_tone_input(&mut data);
        assert_eq!(data, original);
    }

    #[cfg(not(feature = "fixed_point"))]
    #[test]
    #[cfg_attr(miri, ignore = "libm relies on inline assembly under Miri")]
    fn acos_approx_matches_libm_for_float_build() {
        let samples = [-1.0f32, -0.5, 0.0, 0.3, 0.75, 1.0];

        for &value in &samples {
            let expected = libm::acosf(value);
            let approx = acos_approx(value);
            assert!(
                (approx - expected).abs() < 1e-6,
                "approximation should match libm for value {value}"
            );
        }
    }
}
