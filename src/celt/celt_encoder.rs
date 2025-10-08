#![allow(dead_code)]

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

use crate::celt::celt::resampling_factor;
use crate::celt::cpu_support::opus_select_arch;
use crate::celt::math::celt_sqrt;
use crate::celt::modes::opus_custom_mode_find_static;
use crate::celt::types::{
    AnalysisInfo, CeltGlog, CeltSig, OpusCustomEncoder, OpusCustomMode, OpusInt32, OpusUint32,
    OpusVal16, OpusVal32, SilkInfo,
};
use core::cmp::{max, min};
use libm::floorf;

/// Maximum number of channels supported by the scalar encoder path.
const MAX_CHANNELS: usize = 2;

/// Size of the comb-filter history kept per channel by the encoder prefilter.
const COMBFILTER_MAXPERIOD: usize = 1024;

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

/// Helper owning the trailing buffers that back [`OpusCustomEncoder`].
#[derive(Debug, Default)]
pub(crate) struct CeltEncoderAlloc {
    in_mem: Vec<CeltSig>,
    prefilter_mem: Vec<CeltSig>,
    old_band_e: Vec<CeltGlog>,
    old_log_e: Vec<CeltGlog>,
    old_log_e2: Vec<CeltGlog>,
    energy_error: Vec<CeltGlog>,
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

    /// Mirrors `opus_custom_encoder_init()` by configuring the encoder for a custom mode.
    pub(crate) fn init_custom_encoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
        self.init_internal(
            mode,
            channels,
            stream_channels,
            1,
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

// TODO: Port the main `celt_encode_with_ec()` path once the supporting modules
//       are available in Rust.

#[cfg(test)]
mod tests {
    use super::{
        COMBFILTER_MAXPERIOD, CeltEncoderAlloc, CeltEncoderInitError, EncoderCtlRequest,
        MAX_CHANNELS, OPUS_BITRATE_MAX, OpusCustomEncoder, OpusCustomMode, OpusUint32,
    };
    use super::{
        CeltEncoderCtlError, compute_vbr, opus_custom_encoder_ctl, patch_transient_decision,
        transient_analysis,
    };
    use crate::celt::cpu_support::opus_select_arch;
    use crate::celt::modes::opus_custom_mode_create;
    use crate::celt::types::AnalysisInfo;
    use crate::celt::vq::SPREAD_NORMAL;
    use alloc::vec;

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
}
