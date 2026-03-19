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

use alloc::vec;
use alloc::vec::Vec;
#[cfg(test)]
extern crate std;

use crate::celt::BandCodingState;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::bands::anti_collapse;
use crate::celt::bands::celt_lcg_rand;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::bands::denormalise_bands;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::bands::quant_all_bands;
#[cfg(feature = "fixed_point")]
use crate::celt::bands::{anti_collapse_fixed, quant_all_bands_decode_fixed};
#[cfg(feature = "fixed_point")]
use crate::celt::bands::{denormalise_bands_fixed, denormalise_bands_fixed_native};
#[cfg(any(not(feature = "fixed_point"), test))]
use crate::celt::celt::comb_filter;
#[cfg(feature = "fixed_point")]
use crate::celt::celt::comb_filter_fixed;
#[cfg(feature = "fixed_point")]
use crate::celt::celt::comb_filter_fixed_in_place;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::celt::comb_filter_in_place;
use crate::celt::celt::{COMBFILTER_MINPERIOD, TF_SELECT_TABLE, init_caps, resampling_factor};
use crate::celt::cpu_support::{OPUS_ARCHMASK, opus_select_arch};
use crate::celt::entcode::{self, BITRES};
use crate::celt::entdec::EcDec;
#[cfg(feature = "fixed_point")]
use crate::celt::fixed_arch::{
    DB_SHIFT, Q15_ONE, SIG_SHIFT, int16tosig, res2float as fixed_res_to_float,
    sig2res as fixed_sig_to_res, sig2word16,
};
#[cfg(feature = "fixed_point")]
use crate::celt::fixed_ops::{
    add32_ovflw, mult16_16_q15, mult16_32_q15, pshr32, qconst16, qconst16_clamped, qconst32,
    shl32 as shl32_fixed,
};
use crate::celt::float_cast::CELT_SIG_SCALE;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::float_cast::float2int;
use crate::celt::float_cast::float2int16;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::lpc::{celt_autocorr, celt_iir, celt_lpc};
#[cfg(feature = "fixed_point")]
use crate::celt::lpc::{celt_autocorr_fixed, celt_fir_fixed, celt_iir_fixed, celt_lpc_fixed};
#[cfg(feature = "fixed_point")]
use crate::celt::math::celt_zlog2;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::math::{celt_sqrt, frac_div32};
#[cfg(feature = "fixed_point")]
use crate::celt::math_fixed::{
    celt_maxabs16, celt_sqrt as celt_sqrt_fixed, frac_div32 as frac_div32_fixed,
};
#[cfg(not(feature = "fixed_point"))]
use crate::celt::mdct::clt_mdct_backward;
#[cfg(feature = "fixed_point")]
use crate::celt::mdct_fixed::{FixedMdctLookup, clt_mdct_backward_fixed};
use crate::celt::modes::{opus_custom_mode_find_static, opus_custom_mode_find_static_ref};
#[cfg(not(feature = "fixed_point"))]
use crate::celt::pitch::{pitch_downsample, pitch_search};
#[cfg(feature = "fixed_point")]
use crate::celt::pitch::{pitch_downsample_fixed, pitch_search_fixed};
#[cfg(not(feature = "fixed_point"))]
use crate::celt::quant_bands::unquant_energy_finalise;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::quant_bands::{unquant_coarse_energy, unquant_fine_energy};
#[cfg(feature = "fixed_point")]
use crate::celt::quant_bands::{
    unquant_coarse_energy_fixed, unquant_energy_finalise_fixed, unquant_fine_energy_fixed,
};
use crate::celt::rate::clt_compute_allocation;
#[cfg(any(feature = "fixed_point", feature = "deep_plc"))]
use crate::celt::types::OpusInt16;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::types::OpusVal32;
use crate::celt::types::{
    CeltGlog, CeltNorm, CeltSig, OpusCustomDecoder, OpusCustomMode, OpusInt32, OpusRes, OpusUint32,
    OpusVal16,
};
#[cfg(feature = "fixed_point")]
use crate::celt::types::{
    FixedCeltCoef, FixedCeltGlog, FixedCeltNorm, FixedCeltSig, FixedOpusVal16,
};
use crate::celt::vq::SPREAD_NORMAL;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::vq::renormalise_vector;
#[cfg(feature = "fixed_point")]
use crate::celt::vq::renormalise_vector_fixed;
#[cfg(feature = "deep_plc")]
use crate::celt::{
    LpcNetPlcState, PLC_FRAME_SIZE, PLC_UPDATE_SAMPLES, PREEMPHASIS, SINC_FILTER, SINC_ORDER,
    update_plc_state,
};
#[cfg(not(feature = "fixed_point"))]
use core::cmp::Ordering;
use core::cmp::{max, min};

#[cfg(feature = "deep_plc")]
type PlcHandle<'a> = Option<&'a mut LpcNetPlcState>;
#[cfg(not(feature = "deep_plc"))]
type PlcHandle<'a> = ();

#[cfg(feature = "fixed_point")]
fn glog_from_fixed(value: FixedCeltGlog) -> f32 {
    value as f32 / (1u32 << DB_SHIFT) as f32
}

#[cfg(feature = "fixed_point")]
fn glog_to_fixed(value: f32) -> FixedCeltGlog {
    qconst32(value as f64, DB_SHIFT)
}

#[cfg(feature = "fixed_point")]
fn sync_loge_from_fixed(dst: &mut [CeltGlog], src: &[FixedCeltGlog]) {
    for (out, &value) in dst.iter_mut().zip(src.iter()) {
        *out = glog_from_fixed(value);
    }
}

#[cfg(feature = "fixed_point")]
fn sync_loge_to_fixed(dst: &mut [FixedCeltGlog], src: &[CeltGlog]) {
    for (out, &value) in dst.iter_mut().zip(src.iter()) {
        *out = glog_to_fixed(value);
    }
}

#[cfg(feature = "fixed_point")]
#[inline]
fn lpc_from_fixed(value: FixedOpusVal16) -> OpusVal16 {
    value as f32 / (1 << 12) as f32
}

#[cfg(feature = "fixed_point")]
#[inline]
fn lpc_to_fixed(value: OpusVal16) -> FixedOpusVal16 {
    qconst16(value as f64, 12)
}

#[cfg(feature = "fixed_point")]
fn sync_from_fixed_primary_to_float_cache(decoder: &mut OpusCustomDecoder<'_>) {
    debug_assert_eq!(decoder.decode_mem_fixed.len(), decoder.decode_mem.len());
    debug_assert_eq!(decoder.lpc_fixed.len(), decoder.lpc.len());
    debug_assert_eq!(decoder.old_ebands_fixed.len(), decoder.old_ebands.len());
    debug_assert_eq!(decoder.old_log_e_fixed.len(), decoder.old_log_e.len());
    debug_assert_eq!(decoder.old_log_e2_fixed.len(), decoder.old_log_e2.len());
    debug_assert_eq!(
        decoder.background_log_e_fixed.len(),
        decoder.background_log_e.len()
    );

    for (dst, &src) in decoder
        .decode_mem
        .iter_mut()
        .zip(decoder.decode_mem_fixed.iter())
    {
        *dst = fixed_sig_to_float(src);
    }
    for (dst, &src) in decoder.lpc.iter_mut().zip(decoder.lpc_fixed.iter()) {
        *dst = lpc_from_fixed(src);
    }
    for (dst, &src) in decoder
        .preemph_mem_decoder
        .iter_mut()
        .zip(decoder.fixed_preemph_mem_decoder.iter())
    {
        *dst = fixed_sig_to_float(src);
    }
    sync_loge_from_fixed(&mut decoder.old_ebands, &decoder.old_ebands_fixed);
    sync_loge_from_fixed(&mut decoder.old_log_e, &decoder.old_log_e_fixed);
    sync_loge_from_fixed(&mut decoder.old_log_e2, &decoder.old_log_e2_fixed);
    sync_loge_from_fixed(
        &mut decoder.background_log_e,
        &decoder.background_log_e_fixed,
    );
}

#[cfg(feature = "fixed_point")]
fn sync_from_float_cache_to_fixed_primary(decoder: &mut OpusCustomDecoder<'_>) {
    debug_assert_eq!(decoder.decode_mem_fixed.len(), decoder.decode_mem.len());
    debug_assert_eq!(decoder.lpc_fixed.len(), decoder.lpc.len());
    debug_assert_eq!(decoder.old_ebands_fixed.len(), decoder.old_ebands.len());
    debug_assert_eq!(decoder.old_log_e_fixed.len(), decoder.old_log_e.len());
    debug_assert_eq!(decoder.old_log_e2_fixed.len(), decoder.old_log_e2.len());
    debug_assert_eq!(
        decoder.background_log_e_fixed.len(),
        decoder.background_log_e.len()
    );

    for (dst, &src) in decoder
        .decode_mem_fixed
        .iter_mut()
        .zip(decoder.decode_mem.iter())
    {
        *dst = celt_sig_to_fixed(src);
    }
    for (dst, &src) in decoder.lpc_fixed.iter_mut().zip(decoder.lpc.iter()) {
        *dst = lpc_to_fixed(src);
    }
    for (dst, &src) in decoder
        .fixed_preemph_mem_decoder
        .iter_mut()
        .zip(decoder.preemph_mem_decoder.iter())
    {
        *dst = celt_sig_to_fixed(src);
    }
    sync_loge_to_fixed(&mut decoder.old_ebands_fixed, &decoder.old_ebands);
    sync_loge_to_fixed(&mut decoder.old_log_e_fixed, &decoder.old_log_e);
    sync_loge_to_fixed(&mut decoder.old_log_e2_fixed, &decoder.old_log_e2);
    sync_loge_to_fixed(
        &mut decoder.background_log_e_fixed,
        &decoder.background_log_e,
    );
}

#[cfg(feature = "fixed_point")]
fn sync_fixed_output_window_to_float_cache(
    decoder: &mut OpusCustomDecoder<'_>,
    channels: usize,
    output_start: usize,
    output_len: usize,
) {
    let stride = DECODE_BUFFER_SIZE + decoder.overlap;
    for channel in 0..channels {
        let base = channel
            .checked_mul(stride)
            .expect("channel stride multiplication overflow");
        for i in 0..output_len {
            let idx = base + output_start + i;
            decoder.decode_mem[idx] = fixed_sig_to_float(decoder.decode_mem_fixed[idx]);
        }
    }
}

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
#[cfg(feature = "fixed_point")]
const FIXED_SIG_SAT: FixedCeltSig = 536_870_911;

#[cfg(feature = "fixed_point")]
type FixedSynthesisCtx<'a> = (&'a FixedMdctLookup, &'a [FixedCeltCoef], usize);
#[cfg(not(feature = "fixed_point"))]
type FixedSynthesisCtx<'a> = ();

#[cfg(feature = "fixed_point")]
fn fixed_sig_to_float(value: FixedCeltSig) -> f32 {
    value as f32 / (1u32 << SIG_SHIFT) as f32
}

#[cfg(feature = "fixed_point")]
#[inline]
fn celt_sig_to_fixed(value: CeltSig) -> FixedCeltSig {
    libm::rintf(value * (1u32 << SIG_SHIFT) as f32).clamp(i32::MIN as f32, i32::MAX as f32)
        as FixedCeltSig
}

#[cfg(feature = "fixed_point")]
#[inline]
fn fixed_sig_to_word16(value: CeltSig) -> OpusInt16 {
    sig2word16(celt_sig_to_fixed(value))
}

#[cfg(feature = "fixed_point")]
#[inline]
fn q15_to_float(value: FixedOpusVal16) -> f32 {
    value as f32 / (1u32 << 15) as f32
}

#[cfg(not(feature = "fixed_point"))]
#[inline]
fn decoder_noise_renormalise_runtime(x: &mut [OpusVal16], n: usize, gain: f32, arch: i32) {
    renormalise_vector(x, n, gain, arch);
}

#[cfg(feature = "fixed_point")]
#[inline]
fn fixed_norm_to_float(value: FixedOpusVal16) -> OpusVal16 {
    f32::from(value) * (1.0 / 32_768.0)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn fixed_norm_slice_to_float(dst: &mut [OpusVal16], src: &[FixedCeltNorm]) {
    debug_assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src.iter()) {
        *out = fixed_norm_to_float(value);
    }
}

#[cfg(feature = "fixed_point")]
#[inline]
fn float_norm_slice_to_fixed(dst: &mut [FixedCeltNorm], src: &[OpusVal16]) {
    debug_assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src.iter()) {
        *out = float2int16(value);
    }
}

#[cfg(feature = "fixed_point")]
#[inline]
fn gain_to_q31(gain: f32) -> i32 {
    qconst32(f64::from(gain), 31)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn raw_norm_float_to_i16(value: OpusVal16) -> FixedOpusVal16 {
    libm::rintf(value).clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as FixedOpusVal16
}

#[cfg(feature = "fixed_point")]
fn decoder_noise_renormalise_runtime(x: &mut [OpusVal16], n: usize, gain: f32, arch: i32) {
    assert!(x.len() >= n, "input vector shorter than band size");
    // In the fixed decoder PLC-noise path the spectrum is generated as raw
    // i16 magnitudes (rng>>20), not unit-range floats. Preserve that domain
    // before calling the fixed renormaliser.
    let mut fixed_x: Vec<FixedOpusVal16> = x
        .iter()
        .take(n)
        .map(|&sample| raw_norm_float_to_i16(sample))
        .collect();
    renormalise_vector_fixed(&mut fixed_x, n, gain_to_q31(gain), arch);
    for (dst, &sample) in x.iter_mut().take(n).zip(fixed_x.iter()) {
        *dst = fixed_norm_to_float(sample);
    }
}

#[cfg(feature = "fixed_point")]
fn plc_decay_terms_fixed(exc_sig: &[CeltSig], exc_length: usize) -> (i32, i32, FixedOpusVal16) {
    debug_assert!(!exc_sig.is_empty(), "excitation history cannot be empty");
    debug_assert!(
        exc_length <= exc_sig.len(),
        "excitation length exceeds history"
    );

    let exc_i16: Vec<OpusInt16> = exc_sig
        .iter()
        .map(|&sample| fixed_sig_to_word16(sample))
        .collect();
    plc_decay_terms_fixed_native(&exc_i16, exc_length)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn plc_ratio_from_energies_fixed(s1: i32, s2: i32) -> FixedOpusVal16 {
    celt_sqrt_fixed(frac_div32_fixed(
        (s1 >> 1).wrapping_add(1),
        s2.wrapping_add(1),
    )) as FixedOpusVal16
}

#[cfg(feature = "fixed_point")]
fn plc_ratio_terms_fixed(old_sig: &[CeltSig], new_sig: &[CeltSig]) -> (i32, i32, FixedOpusVal16) {
    debug_assert_eq!(
        old_sig.len(),
        new_sig.len(),
        "energy windows must have equal length"
    );

    let old_fixed: Vec<FixedCeltSig> = old_sig
        .iter()
        .map(|&sample| celt_sig_to_fixed(sample))
        .collect();
    let new_fixed: Vec<FixedCeltSig> = new_sig
        .iter()
        .map(|&sample| celt_sig_to_fixed(sample))
        .collect();
    plc_ratio_terms_fixed_native(&old_fixed, &new_fixed)
}

#[cfg(feature = "fixed_point")]
fn plc_decay_terms_fixed_native(
    exc: &[FixedOpusVal16],
    exc_length: usize,
) -> (i32, i32, FixedOpusVal16) {
    debug_assert!(!exc.is_empty(), "excitation history cannot be empty");
    debug_assert!(exc_length <= exc.len(), "excitation length exceeds history");

    let history_len = exc.len();
    let start = history_len.saturating_sub(exc_length);
    let maxabs = celt_maxabs16(&exc[start..]);
    let shift = max(0, 2 * celt_zlog2(maxabs) - 20) as u32;

    let decay_length = exc_length >> 1;
    let mut e1 = 1i32;
    let mut e2 = 1i32;
    for i in 0..decay_length {
        let a = i32::from(exc[history_len - decay_length + i]);
        let b = i32::from(exc[history_len - 2 * decay_length + i]);
        e1 = e1.wrapping_add((a.wrapping_mul(a)) >> shift);
        e2 = e2.wrapping_add((b.wrapping_mul(b)) >> shift);
    }
    e1 = min(e1, e2);
    let decay = celt_sqrt_fixed(frac_div32_fixed(e1 >> 1, e2)) as FixedOpusVal16;
    (e1, e2, decay)
}

#[cfg(feature = "fixed_point")]
fn plc_ratio_terms_fixed_native(
    old_sig: &[FixedCeltSig],
    new_sig: &[FixedCeltSig],
) -> (i32, i32, FixedOpusVal16) {
    debug_assert_eq!(
        old_sig.len(),
        new_sig.len(),
        "energy windows must have equal length"
    );

    let mut s1 = 0i32;
    let mut s2 = 0i32;
    for (&old_sample, &new_sample) in old_sig.iter().zip(new_sig.iter()) {
        let old = i32::from(sig2word16(old_sample));
        let new = i32::from(sig2word16(new_sample));
        s1 = s1.wrapping_add((old.wrapping_mul(old)) >> 10);
        s2 = s2.wrapping_add((new.wrapping_mul(new)) >> 10);
    }
    let ratio = plc_ratio_from_energies_fixed(s1, s2);
    (s1, s2, ratio)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn postfilter_gain_to_fixed(value: OpusVal16) -> FixedOpusVal16 {
    crate::celt::fixed_ops::qconst16_clamped(f64::from(value), 15)
}

#[cfg(not(feature = "fixed_point"))]
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

#[cfg(feature = "fixed_point")]
#[allow(clippy::too_many_arguments)]
fn apply_inverse_mdct_fixed(
    fixed_mdct: &FixedMdctLookup,
    fixed_window: &[FixedCeltCoef],
    overlap: usize,
    freq: &[FixedCeltSig],
    output: &mut [CeltSig],
    bands: usize,
    nb: usize,
    shift: usize,
) {
    let mut fixed_output = vec![0; output.len()];
    apply_inverse_mdct_fixed_native(
        fixed_mdct,
        fixed_window,
        overlap,
        freq,
        &mut fixed_output,
        bands,
        nb,
        shift,
    );
    for (dst, &src) in output.iter_mut().zip(fixed_output.iter()) {
        *dst = fixed_sig_to_float(src);
    }
}

#[cfg(feature = "fixed_point")]
#[allow(clippy::too_many_arguments)]
fn apply_inverse_mdct_fixed_native(
    fixed_mdct: &FixedMdctLookup,
    fixed_window: &[FixedCeltCoef],
    overlap: usize,
    freq: &[FixedCeltSig],
    output: &mut [FixedCeltSig],
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

    let mut temp = vec![0; nb];
    let mdct = fixed_mdct;
    let window = fixed_window;

    for band in 0..bands {
        for (idx, slot) in temp.iter_mut().enumerate() {
            let src_index = band + idx * stride;
            *slot = freq.get(src_index).copied().unwrap_or_default();
        }

        let start = band * nb;
        clt_mdct_backward_fixed(mdct, &temp, &mut output[start..], window, overlap, shift, 1);
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
    fixed_ctx: FixedSynthesisCtx<'_>,
) {
    assert!(output_channels <= out_syn.len());
    assert!(coded_channels <= 2);
    assert!(output_channels <= 2);
    assert!(lm <= mode.max_lm);
    assert!(eff_end <= mode.num_ebands);
    assert!(downsample > 0);
    #[cfg(not(feature = "fixed_point"))]
    #[allow(clippy::let_unit_value)]
    let _ = fixed_ctx;

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

    #[cfg(feature = "fixed_point")]
    let (fixed_mdct, fixed_window, overlap) = fixed_ctx;

    #[cfg(not(feature = "fixed_point"))]
    let mut freq = vec![0.0f32; n];
    #[cfg(feature = "fixed_point")]
    let mut freq = vec![0; n];

    match (output_channels, coded_channels) {
        (2, 1) => {
            let (left, right) = out_syn.split_at_mut(1);
            let left_out = &mut *left[0];
            let right_out = &mut *right[0];
            #[cfg(not(feature = "fixed_point"))]
            {
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
            #[cfg(feature = "fixed_point")]
            {
                denormalise_bands_fixed(
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
                apply_inverse_mdct_fixed(
                    fixed_mdct,
                    fixed_window,
                    overlap,
                    &freq_copy,
                    left_out,
                    bands,
                    nb,
                    shift,
                );
                apply_inverse_mdct_fixed(
                    fixed_mdct,
                    fixed_window,
                    overlap,
                    &freq,
                    right_out,
                    bands,
                    nb,
                    shift,
                );
            }
        }
        (1, 2) => {
            let out = &mut *out_syn[0];
            #[cfg(not(feature = "fixed_point"))]
            {
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
            #[cfg(feature = "fixed_point")]
            {
                denormalise_bands_fixed(
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

                let mut freq_other = vec![0; n];
                denormalise_bands_fixed(
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
                    *lhs = pshr32(add32_ovflw(*lhs, *rhs), 1);
                }

                apply_inverse_mdct_fixed(
                    fixed_mdct,
                    fixed_window,
                    overlap,
                    &freq,
                    out,
                    bands,
                    nb,
                    shift,
                );
            }
        }
        _ => {
            for channel in 0..output_channels {
                let spectrum = &x[channel * n..(channel + 1) * n];
                let energy = &old_band_e[channel * nb_ebands..(channel + 1) * nb_ebands];
                #[cfg(not(feature = "fixed_point"))]
                {
                    denormalise_bands(
                        mode, spectrum, &mut freq, energy, start, eff_end, m, downsample, silence,
                    );

                    let output = &mut *out_syn[channel];
                    apply_inverse_mdct(mode, &freq, output, bands, nb, shift);
                }
                #[cfg(feature = "fixed_point")]
                {
                    denormalise_bands_fixed(
                        mode, spectrum, &mut freq, energy, start, eff_end, m, downsample, silence,
                    );

                    let output = &mut *out_syn[channel];
                    apply_inverse_mdct_fixed(
                        fixed_mdct,
                        fixed_window,
                        overlap,
                        &freq,
                        output,
                        bands,
                        nb,
                        shift,
                    );
                }
            }
        }
    }

    for channel in out_syn.iter_mut().take(output_channels) {
        for sample in (*channel).iter_mut().take(n) {
            *sample = sample.clamp(-SIG_SAT, SIG_SAT);
        }
    }
}

#[cfg(feature = "fixed_point")]
#[allow(clippy::too_many_arguments)]
fn celt_synthesis_fixed_native(
    mode: &OpusCustomMode<'_>,
    x: &[FixedCeltNorm],
    out_syn: &mut [&mut [FixedCeltSig]],
    old_band_e: &[FixedCeltGlog],
    start: usize,
    eff_end: usize,
    coded_channels: usize,
    output_channels: usize,
    is_transient: bool,
    lm: usize,
    downsample: usize,
    silence: bool,
    fixed_mdct: &FixedMdctLookup,
    fixed_window: &[FixedCeltCoef],
    overlap: usize,
) {
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

    let mut freq = vec![0; n];

    match (output_channels, coded_channels) {
        (2, 1) => {
            let (left, right) = out_syn.split_at_mut(1);
            let left_out = &mut *left[0];
            let right_out = &mut *right[0];

            denormalise_bands_fixed_native(
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
            apply_inverse_mdct_fixed_native(
                fixed_mdct,
                fixed_window,
                overlap,
                &freq_copy,
                left_out,
                bands,
                nb,
                shift,
            );
            apply_inverse_mdct_fixed_native(
                fixed_mdct,
                fixed_window,
                overlap,
                &freq,
                right_out,
                bands,
                nb,
                shift,
            );
        }
        (1, 2) => {
            let out = &mut *out_syn[0];
            denormalise_bands_fixed_native(
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

            let mut freq_other = vec![0; n];
            denormalise_bands_fixed_native(
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
                *lhs = pshr32(add32_ovflw(*lhs, *rhs), 1);
            }

            apply_inverse_mdct_fixed_native(
                fixed_mdct,
                fixed_window,
                overlap,
                &freq,
                out,
                bands,
                nb,
                shift,
            );
        }
        _ => {
            for channel in 0..output_channels {
                let spectrum = &x[channel * n..(channel + 1) * n];
                let energy = &old_band_e[channel * nb_ebands..(channel + 1) * nb_ebands];
                denormalise_bands_fixed_native(
                    mode, spectrum, &mut freq, energy, start, eff_end, m, downsample, silence,
                );

                let output = &mut *out_syn[channel];
                apply_inverse_mdct_fixed_native(
                    fixed_mdct,
                    fixed_window,
                    overlap,
                    &freq,
                    output,
                    bands,
                    nb,
                    shift,
                );
            }
        }
    }

    for channel in out_syn.iter_mut().take(output_channels) {
        for sample in (*channel).iter_mut().take(n) {
            *sample = (*sample).clamp(-FIXED_SIG_SAT, FIXED_SIG_SAT);
        }
    }
}

/// Runs the PLC pitch search using the same downsampling and lag sweep as the
/// reference `celt_decoder.c` implementation.
fn celt_plc_pitch_search(decode_mem: &[&[CeltSig]], channels: usize, arch: i32) -> i32 {
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

    let offset = (PLC_PITCH_LAG_MAX >> 1) as usize;
    let max_pitch = (PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN) as usize;
    let target_len = DECODE_BUFFER_SIZE.saturating_sub(PLC_PITCH_LAG_MAX as usize);
    if target_len == 0 {
        return PLC_PITCH_LAG_MAX;
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        let mut lp_pitch_buf = vec![0.0f32; DECODE_BUFFER_SIZE >> 1];
        pitch_downsample(&channel_views, &mut lp_pitch_buf, DECODE_BUFFER_SIZE, arch);
        if offset >= lp_pitch_buf.len() {
            return PLC_PITCH_LAG_MAX;
        }
        let pitch_index = pitch_search(
            &lp_pitch_buf[offset..],
            &lp_pitch_buf,
            target_len,
            max_pitch,
            arch,
        );
        PLC_PITCH_LAG_MAX - pitch_index
    }
    #[cfg(feature = "fixed_point")]
    {
        let mut fixed_channels = Vec::with_capacity(channel_views.len());
        for channel in &channel_views {
            let mut fixed = Vec::with_capacity(DECODE_BUFFER_SIZE);
            for &sample in *channel {
                fixed.push(celt_sig_to_fixed(sample));
            }
            fixed_channels.push(fixed);
        }
        let fixed_views: Vec<&[FixedCeltSig]> = fixed_channels.iter().map(Vec::as_slice).collect();

        let mut lp_pitch_buf = vec![0i16; DECODE_BUFFER_SIZE >> 1];
        pitch_downsample_fixed(&fixed_views, &mut lp_pitch_buf, DECODE_BUFFER_SIZE, arch);
        if offset >= lp_pitch_buf.len() {
            return PLC_PITCH_LAG_MAX;
        }
        let pitch_index = pitch_search_fixed(
            &lp_pitch_buf[offset..],
            &lp_pitch_buf,
            target_len,
            max_pitch,
            arch,
        );
        PLC_PITCH_LAG_MAX - pitch_index
    }
}

#[cfg(feature = "fixed_point")]
fn celt_plc_pitch_search_fixed(decode_mem: &[&[FixedCeltSig]], channels: usize, arch: i32) -> i32 {
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

    let offset = (PLC_PITCH_LAG_MAX >> 1) as usize;
    let max_pitch = (PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN) as usize;
    let target_len = DECODE_BUFFER_SIZE.saturating_sub(PLC_PITCH_LAG_MAX as usize);
    if target_len == 0 {
        return PLC_PITCH_LAG_MAX;
    }

    let mut lp_pitch_buf = vec![0i16; DECODE_BUFFER_SIZE >> 1];
    pitch_downsample_fixed(&channel_views, &mut lp_pitch_buf, DECODE_BUFFER_SIZE, arch);
    if offset >= lp_pitch_buf.len() {
        return PLC_PITCH_LAG_MAX;
    }
    let pitch_index = pitch_search_fixed(
        &lp_pitch_buf[offset..],
        &lp_pitch_buf,
        target_len,
        max_pitch,
        arch,
    );
    PLC_PITCH_LAG_MAX - pitch_index
}

#[cfg(feature = "fixed_point")]
fn plc_bandwidth_expand_for_iir(lpc: &mut [FixedOpusVal16]) {
    loop {
        let mut sum = i32::from(qconst16(1.0, SIG_SHIFT));
        for &coeff in lpc.iter() {
            sum = sum.wrapping_add(i32::from(coeff).abs());
        }
        if sum < 65_535 {
            break;
        }

        let mut tmp = Q15_ONE;
        for coeff in lpc.iter_mut() {
            tmp = mult16_16_q15(qconst16(0.99, 15), tmp);
            *coeff = mult16_16_q15(*coeff, tmp);
        }
    }
}

#[cfg(feature = "fixed_point")]
fn celt_decode_lost_pitch_fixed(
    decoder: &mut OpusCustomDecoder<'_>,
    n: usize,
    overlap: usize,
    pitch_index: i32,
    fade: FixedOpusVal16,
    arch: i32,
) {
    let channels = decoder.channels;
    let stride = DECODE_BUFFER_SIZE + overlap;
    let max_period = MAX_PERIOD as usize;
    debug_assert!(pitch_index > 0);
    let loss_duration = decoder.loss_duration;
    let pitch_index_usize = pitch_index as usize;
    let exc_length = min(2 * pitch_index_usize, max_period);
    let start_index = DECODE_BUFFER_SIZE - n;
    let extrapolation_len = n + overlap;
    let fixed_window = decoder.fixed_window.clone();
    let (decode_mem_fixed, lpc_fixed) = (&mut decoder.decode_mem_fixed, &mut decoder.lpc_fixed);

    for ch in 0..channels {
        let channel_base = ch * stride;
        let lpc_base = ch * LPC_ORDER;
        let channel_mem = &mut decode_mem_fixed[channel_base..channel_base + stride];

        let mut exc = vec![0i16; max_period + LPC_ORDER];
        for (i, slot) in exc.iter_mut().enumerate() {
            let src = DECODE_BUFFER_SIZE - max_period - LPC_ORDER + i;
            *slot = sig2word16(channel_mem[src]);
        }

        if loss_duration == 0 {
            let mut ac = [0i32; LPC_ORDER + 1];
            let input = &exc[LPC_ORDER..LPC_ORDER + max_period];
            let window = if overlap == 0 {
                None
            } else {
                Some(&fixed_window[..overlap])
            };
            celt_autocorr_fixed(input, &mut ac, window, overlap, LPC_ORDER, arch);
            ac[0] = ac[0].wrapping_add(ac[0] >> 13);
            for (i, entry) in ac.iter_mut().enumerate().skip(1) {
                *entry = entry.wrapping_sub(mult16_32_q15((2 * i * i) as i16, *entry));
            }
            let lpc = &mut lpc_fixed[lpc_base..lpc_base + LPC_ORDER];
            #[cfg(test)]
            if ch == 0 {
                crate::test_trace::trace_println!("rust exc_first8={:?}", &input[..8]);
                for (idx, value) in ac.iter().enumerate() {
                    crate::test_trace::trace_println!("rust ac[{idx}]={value}");
                }
            }
            celt_lpc_fixed(lpc, &ac);
            #[cfg(test)]
            if ch == 0 {
                crate::test_trace::trace_println!("rust lpc_after_solver={:?}", &lpc[..]);
            }
            plc_bandwidth_expand_for_iir(lpc);
        }

        let lpc_coeffs = &lpc_fixed[lpc_base..lpc_base + LPC_ORDER];
        let fir_start = max_period - exc_length;
        let mut fir_tmp = vec![0i16; exc_length];
        #[cfg(test)]
        if ch == 0 {
            let mut h = 2166136261u32;
            for &v in &exc {
                let x = v as u16;
                h = (h ^ u32::from(x & 0xFF)).wrapping_mul(16777619);
                h = (h ^ u32::from(x >> 8)).wrapping_mul(16777619);
            }
            crate::test_trace::trace_println!("rust raw_exc_hash=0x{h:08x}");
            crate::test_trace::trace_println!(
                "rust fir_hist_first24={:?}",
                &exc[fir_start..fir_start + LPC_ORDER]
            );
            crate::test_trace::trace_println!(
                "rust fir_cur_first16={:?}",
                &exc[fir_start + LPC_ORDER..fir_start + LPC_ORDER + 16]
            );
        }
        celt_fir_fixed(
            &exc[fir_start..fir_start + LPC_ORDER + exc_length],
            lpc_coeffs,
            &mut fir_tmp,
        );
        exc[LPC_ORDER + fir_start..LPC_ORDER + fir_start + exc_length].copy_from_slice(&fir_tmp);
        #[cfg(test)]
        if ch == 0 {
            let mut h = 2166136261u32;
            for &v in &exc {
                let x = v as u16;
                h = (h ^ u32::from(x & 0xFF)).wrapping_mul(16777619);
                h = (h ^ u32::from(x >> 8)).wrapping_mul(16777619);
            }
            crate::test_trace::trace_println!("rust postfir_exc_hash=0x{h:08x}");
            for i in 0..16 {
                crate::test_trace::trace_println!(
                    "rust postfir[{i}]={}",
                    exc[LPC_ORDER + fir_start + i]
                );
            }
        }

        let (_, _, decay) =
            plc_decay_terms_fixed_native(&exc[LPC_ORDER..LPC_ORDER + max_period], exc_length);

        let move_len = DECODE_BUFFER_SIZE - n;
        channel_mem.copy_within(n..n + move_len, 0);

        let extrapolation_offset = max_period - pitch_index_usize;
        #[cfg(test)]
        if ch == 0 {
            crate::test_trace::trace_println!(
                "rust extrap_src_with_ord={:?}",
                &exc[LPC_ORDER + extrapolation_offset..LPC_ORDER + extrapolation_offset + 16]
            );
            crate::test_trace::trace_println!(
                "rust extrap_src_raw={:?}",
                &exc[extrapolation_offset..extrapolation_offset + 16]
            );
        }
        let reference_base = DECODE_BUFFER_SIZE - max_period - n + extrapolation_offset;
        let mut attenuation = mult16_16_q15(fade, decay);
        let mut s1 = 0i32;
        let mut j = 0usize;
        for i in 0..extrapolation_len {
            if j >= pitch_index_usize {
                j -= pitch_index_usize;
                attenuation = mult16_16_q15(attenuation, decay);
            }
            let sample = mult16_16_q15(attenuation, exc[LPC_ORDER + extrapolation_offset + j]);
            channel_mem[start_index + i] = int16tosig(sample);
            let reference = i32::from(sig2word16(channel_mem[reference_base + j]));
            s1 = s1.wrapping_add((reference.wrapping_mul(reference)) >> 10);
            j += 1;
        }

        #[cfg(test)]
        if ch == 0 {
            let mut h = 2166136261u32;
            for &v in &channel_mem[start_index..start_index + extrapolation_len] {
                let x = v as u32;
                for b in [x as u8, (x >> 8) as u8, (x >> 16) as u8, (x >> 24) as u8] {
                    h = (h ^ u32::from(b)).wrapping_mul(16777619);
                }
            }
            crate::test_trace::trace_println!("rust preiir_hash=0x{h:08x} s1={s1} decay={decay}");
            for i in 0..16 {
                crate::test_trace::trace_println!(
                    "rust preiir[{i}]={}",
                    sig2word16(channel_mem[start_index + i])
                );
            }
        }
        let mut lpc_mem = vec![0i16; LPC_ORDER];
        for (idx, mem) in lpc_mem.iter_mut().enumerate() {
            *mem = sig2word16(channel_mem[start_index - 1 - idx]);
        }
        #[cfg(test)]
        if ch == 0 {
            let mut h = 2166136261u32;
            for &v in &lpc_mem {
                let x = v as u16;
                h = (h ^ u32::from(x & 0xFF)).wrapping_mul(16777619);
                h = (h ^ u32::from(x >> 8)).wrapping_mul(16777619);
            }
            crate::test_trace::trace_println!("rust lpc_mem_hash=0x{h:08x}");
        }

        let input = channel_mem[start_index..start_index + extrapolation_len].to_vec();
        let mut filtered = vec![0i32; extrapolation_len];
        celt_iir_fixed(&input, lpc_coeffs, &mut filtered, &mut lpc_mem);
        for (dst, &sample) in channel_mem[start_index..start_index + extrapolation_len]
            .iter_mut()
            .zip(filtered.iter())
        {
            *dst = sample.clamp(-FIXED_SIG_SAT, FIXED_SIG_SAT);
        }
        #[cfg(test)]
        if ch == 0 {
            let mut h = 2166136261u32;
            for &v in &channel_mem[start_index..start_index + extrapolation_len] {
                let x = v as u32;
                for b in [x as u8, (x >> 8) as u8, (x >> 16) as u8, (x >> 24) as u8] {
                    h = (h ^ u32::from(b)).wrapping_mul(16777619);
                }
            }
            let mut hm = 2166136261u32;
            for &v in &lpc_mem {
                let x = v as u16;
                hm = (hm ^ u32::from(x & 0xFF)).wrapping_mul(16777619);
                hm = (hm ^ u32::from(x >> 8)).wrapping_mul(16777619);
            }
            crate::test_trace::trace_println!(
                "rust postiir_hash=0x{h:08x} lpc_mem_after=0x{hm:08x}"
            );
            for i in 0..16 {
                crate::test_trace::trace_println!(
                    "rust postiir[{i}]={}",
                    sig2word16(channel_mem[start_index + i])
                );
            }
        }

        let mut s2 = 0i32;
        for &sample in &channel_mem[start_index..start_index + extrapolation_len] {
            let word = i32::from(sig2word16(sample));
            s2 = s2.wrapping_add((word.wrapping_mul(word)) >> 10);
        }

        #[cfg(test)]
        if ch == 0 {
            crate::test_trace::trace_println!("rust s2={s2}");
        }
        if s1 <= (s2 >> 2) {
            for value in &mut channel_mem[start_index..start_index + extrapolation_len] {
                *value = 0;
            }
        } else if s1 < s2 {
            let ratio = plc_ratio_from_energies_fixed(s1, s2);
            for i in 0..overlap {
                let gain = Q15_ONE
                    .wrapping_sub(mult16_16_q15(fixed_window[i], Q15_ONE.wrapping_sub(ratio)));
                channel_mem[start_index + i] = mult16_32_q15(gain, channel_mem[start_index + i]);
            }
            for i in overlap..extrapolation_len {
                channel_mem[start_index + i] = mult16_32_q15(ratio, channel_mem[start_index + i]);
            }
        }
    }

    decoder.prefilter_and_fold = true;
    sync_from_fixed_primary_to_float_cache(decoder);
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

    #[cfg(not(feature = "fixed_point"))]
    {
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

    #[cfg(feature = "fixed_point")]
    {
        debug_assert!(decoder.fixed_window.len() >= overlap);
        let g0 = postfilter_gain_to_fixed(decoder.postfilter_gain_old).wrapping_neg();
        let g1 = postfilter_gain_to_fixed(decoder.postfilter_gain).wrapping_neg();

        for channel in 0..channels {
            let offset = channel * stride;
            let channel_mem_fixed = &mut decoder.decode_mem_fixed[offset..offset + stride];
            let mut fixed_etmp = vec![0; overlap];

            comb_filter_fixed(
                &mut fixed_etmp,
                channel_mem_fixed,
                start,
                overlap,
                decoder.postfilter_period_old,
                decoder.postfilter_period,
                g0,
                g1,
                tapset0,
                tapset1,
                &decoder.fixed_window[..0],
                0,
                decoder.arch,
            );

            for i in 0..(overlap / 2) {
                let forward = mult16_32_q15(decoder.fixed_window[i], fixed_etmp[overlap - 1 - i]);
                let reverse = mult16_32_q15(decoder.fixed_window[overlap - 1 - i], fixed_etmp[i]);
                let folded = add32_ovflw(forward, reverse);
                let idx = start + i;
                channel_mem_fixed[idx] = folded;
                decoder.decode_mem[offset + idx] = fixed_sig_to_float(folded);
            }
        }
    }
}

pub(crate) fn celt_decode_lost(
    decoder: &mut OpusCustomDecoder<'_>,
    n: usize,
    lm: usize,
    lpcnet: PlcHandle<'_>,
) {
    #[cfg(feature = "fixed_point")]
    sync_from_fixed_primary_to_float_cache(decoder);
    celt_decode_lost_impl(decoder, n, lm, lpcnet);
}

fn celt_decode_lost_impl(
    decoder: &mut OpusCustomDecoder<'_>,
    n: usize,
    lm: usize,
    lpcnet: PlcHandle<'_>,
) {
    #[cfg(feature = "deep_plc")]
    let mut lpcnet = lpcnet;
    #[cfg(not(feature = "deep_plc"))]
    let _ = lpcnet;
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

    let loss_duration = decoder.loss_duration;
    #[cfg(feature = "deep_plc")]
    let noise_based = if let Some(lpcnet) = lpcnet.as_ref() {
        start_band != 0 || (lpcnet.fec_fill_pos == 0 && (decoder.skip_plc || loss_duration >= 80))
    } else {
        loss_duration >= 40 || start_band != 0 || decoder.skip_plc
    };
    #[cfg(not(feature = "deep_plc"))]
    let noise_based = loss_duration >= 40 || start_band != 0 || decoder.skip_plc;
    let arch = decoder.arch;

    if noise_based {
        let move_len = DECODE_BUFFER_SIZE
            .checked_sub(n)
            .expect("frame size larger than decode buffer")
            + overlap;
        for channel_slice in decoder.decode_mem.chunks_mut(stride).take(channels) {
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
                decoder_noise_renormalise_runtime(slice, band_width, 1.0, arch);
            }
        }
        decoder.rng = seed;

        let start = DECODE_BUFFER_SIZE - n;
        let downsample = max(decoder.downsample, 1) as usize;
        #[cfg(feature = "fixed_point")]
        let fixed_ctx = (
            &decoder.fixed_mdct,
            decoder.fixed_window.as_slice(),
            decoder.overlap,
        );
        #[cfg(not(feature = "fixed_point"))]
        let fixed_ctx = ();
        {
            let (decode_mem, old_ebands) = (&mut decoder.decode_mem, &decoder.old_ebands);
            let mut outputs = Vec::with_capacity(channels);
            for channel_slice in decode_mem.chunks_mut(stride).take(channels) {
                outputs.push(&mut channel_slice[start..]);
            }
            // Keep borrow tracking intact by using disjoint slices instead of raw pointers.
            celt_synthesis(
                mode,
                &spectrum,
                &mut outputs,
                old_ebands,
                start_band,
                eff_end,
                channels,
                channels,
                false,
                lm,
                downsample,
                false,
                fixed_ctx,
            );
        }

        decoder.prefilter_and_fold = false;
        decoder.skip_plc = true;
        #[cfg(feature = "fixed_point")]
        sync_from_float_cache_to_fixed_primary(decoder);
    } else {
        #[cfg(feature = "fixed_point")]
        {
            let pitch_index = if loss_duration == 0 {
                #[cfg(feature = "deep_plc")]
                if let Some(lpcnet) = lpcnet.as_deref_mut() {
                    if lpcnet.loaded {
                        let mut views = Vec::with_capacity(channels);
                        for channel_slice in decoder.decode_mem.chunks(stride).take(channels) {
                            views.push(channel_slice);
                        }
                        update_plc_state(lpcnet, &views, &mut decoder.plc_preemphasis_mem);
                    }
                }

                let mut fixed_views = Vec::with_capacity(channels);
                for channel_slice in decoder.decode_mem_fixed.chunks(stride).take(channels) {
                    fixed_views.push(channel_slice);
                }
                let search = celt_plc_pitch_search_fixed(&fixed_views, channels, arch);
                decoder.last_pitch_index = search;
                search
            } else {
                decoder.last_pitch_index
            };

            let fade = if loss_duration == 0 {
                Q15_ONE
            } else {
                qconst16(0.8, 15)
            };
            celt_decode_lost_pitch_fixed(decoder, n, overlap, pitch_index, fade, arch);
        }

        #[cfg(not(feature = "fixed_point"))]
        {
            let pitch_index = if loss_duration == 0 {
                let mut views = Vec::with_capacity(channels);
                for channel_slice in decoder.decode_mem.chunks(stride).take(channels) {
                    views.push(channel_slice);
                }
                #[cfg(feature = "deep_plc")]
                if let Some(lpcnet) = lpcnet.as_deref_mut() {
                    if lpcnet.loaded {
                        update_plc_state(lpcnet, &views, &mut decoder.plc_preemphasis_mem);
                    }
                }
                let search = celt_plc_pitch_search(&views, channels, arch);
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

            let (decode_mem, lpc) = (&mut decoder.decode_mem, &mut decoder.lpc);
            for (ch, channel_slice) in decode_mem.chunks_mut(stride).take(channels).enumerate() {
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
                    celt_autocorr(input, &mut ac, window, overlap, LPC_ORDER, arch);
                    ac[0] *= 1.0001;
                    for (i, entry) in ac.iter_mut().enumerate().skip(1) {
                        let factor = 0.008_f32 * 0.008_f32 * (i * i) as f32;
                        *entry -= *entry * factor;
                    }
                    let base_idx = ch * LPC_ORDER;
                    let lpc_slice = &mut lpc[base_idx..base_idx + LPC_ORDER];
                    celt_lpc(lpc_slice, &ac);
                }

                let base_idx = ch * LPC_ORDER;
                let lpc_coeffs = &lpc[base_idx..base_idx + LPC_ORDER];
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

                let mut filtered =
                    channel_slice[start_index..start_index + extrapolation_len].to_vec();
                let input = filtered.clone();
                celt_iir(&input, lpc_coeffs, &mut filtered, &mut lpc_mem);
                channel_slice[start_index..start_index + extrapolation_len]
                    .copy_from_slice(&filtered);

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

        #[cfg(feature = "deep_plc")]
        if let Some(lpcnet) = lpcnet.as_deref_mut() {
            if lpcnet.loaded && (decoder.complexity >= 5 || lpcnet.fec_fill_pos > 0) {
                let start_index = DECODE_BUFFER_SIZE - n;
                let mut buf_copy = vec![0.0f32; channels * overlap];
                for (ch, channel_slice) in
                    decoder.decode_mem.chunks(stride).take(channels).enumerate()
                {
                    let src = &channel_slice[start_index..start_index + overlap];
                    let base = ch * overlap;
                    buf_copy[base..base + overlap].copy_from_slice(src);
                }

                let samples_needed16k = (n + SINC_ORDER + overlap) / 3;
                if loss_duration == 0 {
                    decoder.plc_fill = 0;
                }
                debug_assert!(decoder.plc_fill >= 0);
                while (decoder.plc_fill as usize) < samples_needed16k {
                    let fill = decoder.plc_fill as usize;
                    let end = fill + PLC_FRAME_SIZE;
                    debug_assert!(end <= decoder.plc_pcm.len());
                    lpcnet.lpcnet_plc_conceal(&mut decoder.plc_pcm[fill..end]);
                    decoder.plc_fill += PLC_FRAME_SIZE as i32;
                }

                let plc_pcm = &mut decoder.plc_pcm;
                let plc_fill = &mut decoder.plc_fill;
                let plc_preemphasis_mem = &mut decoder.plc_preemphasis_mem;
                let decode_mem = &mut decoder.decode_mem;

                let (first, rest) = decode_mem.split_at_mut(stride);
                for i in 0..(n + overlap) / 3 {
                    let mut sum = 0.0f32;
                    for j in 0..17 {
                        sum += 3.0 * f32::from(plc_pcm[i + j]) * SINC_FILTER[3 * j];
                    }
                    first[start_index + 3 * i] = sum;
                    let mut sum = 0.0f32;
                    for j in 0..16 {
                        sum += 3.0 * f32::from(plc_pcm[i + j + 1]) * SINC_FILTER[3 * j + 2];
                    }
                    first[start_index + 3 * i + 1] = sum;
                    let mut sum = 0.0f32;
                    for j in 0..16 {
                        sum += 3.0 * f32::from(plc_pcm[i + j + 1]) * SINC_FILTER[3 * j + 1];
                    }
                    first[start_index + 3 * i + 2] = sum;
                }

                let shift = n / 3;
                let fill = (*plc_fill).max(0) as usize;
                debug_assert!(fill >= shift);
                if fill > shift {
                    plc_pcm.copy_within(shift..fill, 0);
                }
                *plc_fill -= shift as i32;

                for i in 0..n {
                    let tmp = first[start_index + i];
                    first[start_index + i] -= PREEMPHASIS * *plc_preemphasis_mem;
                    *plc_preemphasis_mem = tmp;
                }
                let mut overlap_mem = *plc_preemphasis_mem;
                for i in 0..overlap {
                    let tmp = first[DECODE_BUFFER_SIZE + i];
                    first[DECODE_BUFFER_SIZE + i] -= PREEMPHASIS * overlap_mem;
                    overlap_mem = tmp;
                }

                if channels == 2 {
                    let second = &mut rest[..stride];
                    second.copy_from_slice(first);
                }

                if loss_duration == 0 {
                    for (ch, channel_slice) in
                        decode_mem.chunks_mut(stride).take(channels).enumerate()
                    {
                        let base = ch * overlap;
                        for i in 0..overlap {
                            channel_slice[start_index + i] = (1.0 - mode.window[i])
                                * buf_copy[base + i]
                                + mode.window[i] * channel_slice[start_index + i];
                        }
                    }
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

pub(crate) fn canonical_mode() -> Option<&'static OpusCustomMode<'static>> {
    opus_custom_mode_find_static_ref(48_000, 960)
}

/// Cumulative distribution used to decode the global allocation trim.
const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// Spread decision probabilities used by the transient classifier.
const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];

/// Probability model for the three post-filter tapset candidates.
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// Scalar used to decode the post-filter gain from the coarse index.
const POSTFILTER_GAIN_SCALE: OpusVal16 = 0.09375;

const FROM_OPUS_TABLE: [u8; 16] = [
    0x80, 0x88, 0x90, 0x98, 0x40, 0x48, 0x50, 0x58, 0x20, 0x28, 0x30, 0x38, 0x00, 0x08, 0x10, 0x18,
];

fn from_opus(value: u8) -> Option<u8> {
    if value < 0x80 {
        None
    } else {
        let idx = ((value >> 3) as usize).saturating_sub(16);
        FROM_OPUS_TABLE
            .get(idx)
            .map(|mapped| mapped | (value & 0x7))
    }
}

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

#[cfg(feature = "fixed_point")]
#[inline]
fn deemphasis_coef0_q15(coef: &[OpusVal16]) -> FixedOpusVal16 {
    qconst16_clamped(f64::from(coef[0]), 15)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn deemphasis_coef1_q15(coef: &[OpusVal16]) -> FixedOpusVal16 {
    qconst16_clamped(f64::from(coef[1]), 15)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn deemphasis_coef3_q13(coef: &[OpusVal16]) -> FixedOpusVal16 {
    qconst16_clamped(f64::from(coef[3]), 13)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn preprocess_sample_fixed(sample: FixedCeltSig, mem: FixedCeltSig) -> FixedCeltSig {
    (sample + mem).clamp(-FIXED_SIG_SAT, FIXED_SIG_SAT)
}

#[cfg(feature = "fixed_point")]
fn deemphasis_fixed(
    input: &[&[FixedCeltSig]],
    pcm: &mut [OpusRes],
    n: usize,
    channels: usize,
    downsample: usize,
    coef: &[OpusVal16],
    mem: &mut [FixedCeltSig],
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

    let coef0 = deemphasis_coef0_q15(coef);

    if downsample == 1 && channels == 2 && !accum {
        let left = input[0];
        let right = input[1];
        let mut mem_left = mem[0];
        let mut mem_right = mem[1];

        for j in 0..n {
            let tmp_left = preprocess_sample_fixed(left[j], mem_left);
            let tmp_right = preprocess_sample_fixed(right[j], mem_right);
            mem_left = mult16_32_q15(coef0, tmp_left);
            mem_right = mult16_32_q15(coef0, tmp_right);
            pcm[2 * j] = fixed_res_to_float(fixed_sig_to_res(tmp_left));
            pcm[2 * j + 1] = fixed_res_to_float(fixed_sig_to_res(tmp_right));
        }

        mem[0] = mem_left;
        mem[1] = mem_right;
        return;
    }

    let mut scratch = vec![0; n];
    let nd = n / downsample;

    for channel in 0..channels {
        let samples = input[channel];
        let mut m = mem[channel];
        let mut apply_downsampling = false;

        if coef.len() > 3 && coef[1] != 0.0 {
            let coef1 = deemphasis_coef1_q15(coef);
            let coef3 = deemphasis_coef3_q13(coef);
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample_fixed(sample, m);
                m = mult16_32_q15(coef0, tmp) - mult16_32_q15(coef1, sample);
                scratch[j] = shl32_fixed(mult16_32_q15(coef3, tmp), 2);
            }
            apply_downsampling = true;
        } else if downsample > 1 {
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample_fixed(sample, m);
                m = mult16_32_q15(coef0, tmp);
                scratch[j] = tmp;
            }
            apply_downsampling = true;
        } else if accum {
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample_fixed(sample, m);
                m = mult16_32_q15(coef0, tmp);
                let idx = j * channels + channel;
                pcm[idx] += fixed_res_to_float(fixed_sig_to_res(tmp));
            }
        } else {
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample_fixed(sample, m);
                m = mult16_32_q15(coef0, tmp);
                let idx = j * channels + channel;
                pcm[idx] = fixed_res_to_float(fixed_sig_to_res(tmp));
            }
        }

        mem[channel] = m;

        if apply_downsampling {
            if accum {
                for j in 0..nd {
                    let idx = j * channels + channel;
                    pcm[idx] += fixed_res_to_float(fixed_sig_to_res(scratch[j * downsample]));
                }
            } else {
                for j in 0..nd {
                    let idx = j * channels + channel;
                    pcm[idx] = fixed_res_to_float(fixed_sig_to_res(scratch[j * downsample]));
                }
            }
        }
    }
}

#[cfg(feature = "fixed_point")]
fn deemphasis_fixed_to_int16(
    input: &[&[FixedCeltSig]],
    pcm: &mut [i16],
    n: usize,
    channels: usize,
    downsample: usize,
    coef: &[OpusVal16],
    mem: &mut [FixedCeltSig],
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

    let expected_samples = if downsample > 1 { n / downsample } else { n };
    debug_assert!(
        pcm.len() >= expected_samples * channels,
        "PCM buffer too small for deemphasis output",
    );

    let coef0 = deemphasis_coef0_q15(coef);

    if downsample == 1 && channels == 2 {
        let left = input[0];
        let right = input[1];
        let mut mem_left = mem[0];
        let mut mem_right = mem[1];

        for j in 0..n {
            let tmp_left = preprocess_sample_fixed(left[j], mem_left);
            let tmp_right = preprocess_sample_fixed(right[j], mem_right);
            mem_left = mult16_32_q15(coef0, tmp_left);
            mem_right = mult16_32_q15(coef0, tmp_right);
            pcm[2 * j] = sig2word16(tmp_left);
            pcm[2 * j + 1] = sig2word16(tmp_right);
        }

        mem[0] = mem_left;
        mem[1] = mem_right;
        return;
    }

    let mut scratch = vec![0; n];
    let nd = n / downsample;

    for channel in 0..channels {
        let samples = input[channel];
        let mut m = mem[channel];

        if coef.len() > 3 && coef[1] != 0.0 {
            let coef1 = deemphasis_coef1_q15(coef);
            let coef3 = deemphasis_coef3_q13(coef);
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample_fixed(sample, m);
                m = mult16_32_q15(coef0, tmp) - mult16_32_q15(coef1, sample);
                scratch[j] = shl32_fixed(mult16_32_q15(coef3, tmp), 2);
            }
            for j in 0..nd {
                let idx = j * channels + channel;
                pcm[idx] = sig2word16(scratch[j * downsample]);
            }
        } else if downsample > 1 {
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample_fixed(sample, m);
                m = mult16_32_q15(coef0, tmp);
                scratch[j] = tmp;
            }
            for j in 0..nd {
                let idx = j * channels + channel;
                pcm[idx] = sig2word16(scratch[j * downsample]);
            }
        } else {
            for (j, &sample) in samples.iter().take(n).enumerate() {
                let tmp = preprocess_sample_fixed(sample, m);
                m = mult16_32_q15(coef0, tmp);
                let idx = j * channels + channel;
                pcm[idx] = sig2word16(tmp);
            }
        }

        mem[channel] = m;
    }
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
    #[cfg(feature = "deep_plc")]
    plc_pcm: [OpusInt16; PLC_UPDATE_SAMPLES],
    #[cfg(feature = "deep_plc")]
    plc_fill: OpusInt32,
    #[cfg(feature = "deep_plc")]
    plc_preemphasis_mem: f32,
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
        + 4 * band_history * core::mem::size_of::<CeltGlog>()
        + {
            #[cfg(feature = "fixed_point")]
            {
                decode_mem * core::mem::size_of::<FixedCeltSig>()
                    + lpc * core::mem::size_of::<FixedOpusVal16>()
                    + 4 * band_history * core::mem::size_of::<FixedCeltGlog>()
            }
            #[cfg(not(feature = "fixed_point"))]
            {
                0
            }
        };
    Some(size)
}

/// Returns the size of the canonical CELT decoder operating at 48 kHz/960.
#[must_use]
pub(crate) fn celt_decoder_get_size(channels: usize) -> Option<usize> {
    opus_custom_mode_find_static(48_000, 960)
        .and_then(|mode| opus_custom_decoder_get_size(&mode, channels))
}

/// Owning wrapper around [`OpusCustomDecoder`].
///
/// The decoder now owns its trailing buffers directly, but the wrapper keeps
/// the historical "owned decoder handle" surface used by the Opus front-end
/// and existing tests.
#[derive(Debug)]
pub(crate) struct OwnedCeltDecoder<'mode> {
    decoder: OpusCustomDecoder<'mode>,
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
pub(crate) fn celt_decoder_init(
    alloc: &mut CeltDecoderAlloc,
    sampling_rate: OpusInt32,
    channels: usize,
) -> Result<OpusCustomDecoder<'static>, CeltDecoderInitError> {
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

/// Range decoder wrapper that can either own or borrow the underlying decoder.
#[derive(Debug)]
pub(crate) enum RangeDecoderHandle<'a> {
    Owned(EcDec<'a>),
    External(&'a mut EcDec<'a>),
}

impl<'a> RangeDecoderHandle<'a> {
    fn new(packet: &'a [u8], external: Option<&'a mut EcDec<'a>>) -> Self {
        match external {
            Some(dec) => Self::External(dec),
            None => Self::Owned(EcDec::new(packet)),
        }
    }

    fn decoder(&mut self) -> &mut EcDec<'a> {
        match self {
            Self::Owned(decoder) => decoder,
            Self::External(dec) => dec,
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
        decoder.start_band >= 0,
        "decoder start band must be non-negative",
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

    let stride = DECODE_BUFFER_SIZE + decoder.overlap;
    debug_assert_eq!(
        decoder.decode_mem.len(),
        stride * decoder.channels,
        "decode history must match channel-stride layout",
    );
    debug_assert_eq!(
        decoder.lpc.len(),
        LPC_ORDER * decoder.channels,
        "LPC history size must match channel count",
    );
    let band_count = 2 * decoder.mode.num_ebands;
    debug_assert_eq!(decoder.old_ebands.len(), band_count);
    debug_assert_eq!(decoder.old_log_e.len(), band_count);
    debug_assert_eq!(decoder.old_log_e2.len(), band_count);
    debug_assert_eq!(decoder.background_log_e.len(), band_count);
    #[cfg(feature = "fixed_point")]
    {
        debug_assert_eq!(decoder.decode_mem_fixed.len(), decoder.decode_mem.len());
        debug_assert_eq!(decoder.lpc_fixed.len(), decoder.lpc.len());
        debug_assert_eq!(decoder.old_ebands_fixed.len(), band_count);
        debug_assert_eq!(decoder.old_log_e_fixed.len(), band_count);
        debug_assert_eq!(decoder.old_log_e2_fixed.len(), band_count);
        debug_assert_eq!(
            decoder.background_log_e_fixed.len(),
            decoder.background_log_e.len()
        );
    }
}

/// Metadata describing the parsed frame header and bit allocation.
#[derive(Debug)]
pub(crate) struct FramePreparation<'a> {
    pub range_decoder: Option<RangeDecoderHandle<'a>>,
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

impl<'a> FramePreparation<'a> {
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
pub(crate) fn prepare_frame<'mode, 'pkt>(
    decoder: &mut OpusCustomDecoder<'mode>,
    packet: &'pkt [u8],
    frame_size: usize,
    range_decoder: Option<&'pkt mut EcDec<'pkt>>,
) -> Result<FramePreparation<'pkt>, CeltDecodeError>
where
    'mode: 'pkt,
{
    let cc = decoder.channels;
    let mut c = decoder.stream_channels;
    if cc == 0 || c == 0 || cc > MAX_CHANNELS {
        return Err(CeltDecodeError::BadArgument);
    }

    if frame_size == 0 {
        return Err(CeltDecodeError::BadArgument);
    }

    let mode = decoder.mode;
    let nb_ebands = mode.num_ebands;
    let start = decoder.start_band as usize;
    let mut end = decoder.end_band as usize;

    let downsample = decoder.downsample as usize;
    let mut scaled_frame = frame_size
        .checked_mul(downsample)
        .ok_or(CeltDecodeError::BadArgument)?;
    let mut packet = packet;
    let mut header_lm: Option<usize> = None;

    if decoder.signalling != 0 && !packet.is_empty() {
        let mut data0 = packet[0];
        if mode.sample_rate == 48_000 && mode.short_mdct_size == 120 {
            data0 = from_opus(data0).ok_or(CeltDecodeError::InvalidPacket)?;
        }
        end = max(
            1,
            mode.effective_ebands
                .saturating_sub(2 * (data0 as usize >> 5)),
        );
        decoder.end_band = end as i32;
        c = 1 + ((data0 >> 2) & 0x1) as usize;
        header_lm = Some(((data0 >> 3) & 0x3) as usize);

        if (packet[0] & 0x03) == 0x03 {
            packet = packet.get(1..).ok_or(CeltDecodeError::InvalidPacket)?;
            if packet.is_empty() {
                return Err(CeltDecodeError::InvalidPacket);
            }
            if (packet[0] & 0x40) != 0 {
                packet = packet.get(1..).ok_or(CeltDecodeError::InvalidPacket)?;
                if packet.is_empty() {
                    return Err(CeltDecodeError::InvalidPacket);
                }
                let mut len = packet.len() as i32;
                let mut padding = 0i32;
                loop {
                    if packet.is_empty() {
                        return Err(CeltDecodeError::InvalidPacket);
                    }
                    let p = packet[0];
                    packet = &packet[1..];
                    len -= 1;
                    let tmp = if p == 255 { 254 } else { p as i32 };
                    len -= tmp;
                    padding += tmp;
                    if p != 255 {
                        break;
                    }
                }
                padding -= 1;
                if len <= 0 || padding < 0 {
                    return Err(CeltDecodeError::InvalidPacket);
                }
                packet = packet
                    .get(..len as usize)
                    .ok_or(CeltDecodeError::InvalidPacket)?;
            }
        } else {
            packet = packet.get(1..).ok_or(CeltDecodeError::InvalidPacket)?;
        }

        let lm = header_lm.unwrap_or(0);
        if lm > mode.max_lm {
            return Err(CeltDecodeError::InvalidPacket);
        }
        let required = mode.short_mdct_size << lm;
        if scaled_frame < required {
            return Err(CeltDecodeError::BadArgument);
        }
        scaled_frame = required;
    }

    if scaled_frame > (mode.short_mdct_size << mode.max_lm) {
        return Err(CeltDecodeError::BadArgument);
    }

    let lm = if let Some(lm) = header_lm {
        lm
    } else {
        (0..=mode.max_lm)
            .find(|&cand| mode.short_mdct_size << cand == scaled_frame)
            .ok_or(CeltDecodeError::BadArgument)?
    };
    let m = 1 << lm;
    let n = m * mode.short_mdct_size;

    if c == 0 || c > MAX_CHANNELS {
        return Err(CeltDecodeError::BadArgument);
    }

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

    let mut range_decoder = RangeDecoderHandle::new(packet, range_decoder);
    let is_owned = matches!(&range_decoder, RangeDecoderHandle::Owned(_));
    let dec = range_decoder.decoder();
    if is_owned {
        debug_assert_eq!(
            dec.ctx().storage as usize,
            packet.len(),
            "range decoder storage must match packet length",
        );
    }

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

    #[cfg(feature = "fixed_point")]
    {
        sync_loge_to_fixed(&mut decoder.old_ebands_fixed, &decoder.old_ebands);
        unquant_coarse_energy_fixed(
            mode,
            start,
            end,
            &mut decoder.old_ebands_fixed,
            intra_ener,
            dec,
            c,
            lm,
        );
        sync_loge_from_fixed(&mut decoder.old_ebands, &decoder.old_ebands_fixed);
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        unquant_coarse_energy(
            mode,
            start,
            end,
            &mut decoder.old_ebands,
            intra_ener,
            dec,
            c,
            lm,
        );
    }

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
    let total_bits = len_bits << BITRES;
    let mut dynalloc_total_bits = total_bits;
    let mut tell_frac = entcode::ec_tell_frac(dec.ctx()) as OpusInt32;

    for band in start..end {
        let band_width = i32::from(mode.e_bands[band + 1] - mode.e_bands[band]);
        let width = (c as OpusInt32 * band_width) << lm;
        let six_bits = (6 << BITRES) as OpusInt32;
        let quanta = min(width << BITRES, max(six_bits, width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let mut boost = 0;
        while tell_frac + ((dynalloc_loop_logp as OpusInt32) << BITRES) < dynalloc_total_bits
            && boost < cap[band]
        {
            let flag = dec.dec_bit_logp(dynalloc_loop_logp as u32);
            tell_frac = entcode::ec_tell_frac(dec.ctx()) as OpusInt32;
            if flag == 0 {
                break;
            }
            boost += quanta;
            dynalloc_total_bits -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[band] = boost;
        if boost > 0 {
            dynalloc_logp = max(2, dynalloc_logp - 1);
        }
    }

    let mut fine_quant = vec![0; nb_ebands];
    let alloc_trim = if tell_frac + ((6 << BITRES) as OpusInt32) <= dynalloc_total_bits {
        dec.dec_icdf(&TRIM_ICDF, 7)
    } else {
        5
    };

    #[cfg(feature = "fixed_point")]
    let frame_total_bits = total_bits;
    #[cfg(not(feature = "fixed_point"))]
    let frame_total_bits = dynalloc_total_bits;

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

    #[cfg(feature = "fixed_point")]
    {
        unquant_fine_energy_fixed(
            mode,
            start,
            end,
            &mut decoder.old_ebands_fixed,
            &fine_quant,
            dec,
            c,
        );
        sync_loge_from_fixed(&mut decoder.old_ebands, &decoder.old_ebands_fixed);
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        unquant_fine_energy(
            mode,
            start,
            end,
            &mut decoder.old_ebands,
            &fine_quant,
            dec,
            c,
        );
    }

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
        total_bits: frame_total_bits,
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

fn res_to_int24(sample: OpusRes) -> i32 {
    #[cfg(feature = "fixed_point")]
    {
        crate::celt::res2int24(crate::celt::float2res(sample))
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        let scale = CELT_SIG_SCALE * 256.0;
        let scaled = (sample * scale).clamp(-8_388_608.0, 8_388_607.0);
        float2int(scaled)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn celt_decode_with_ec_dred<'mode, 'pkt>(
    decoder: &mut OpusCustomDecoder<'mode>,
    packet: Option<&'pkt [u8]>,
    pcm: &mut [OpusRes],
    frame_size: usize,
    range_decoder: Option<&'pkt mut EcDec<'pkt>>,
    accum: bool,
    plc: PlcHandle<'_>,
) -> Result<usize, CeltDecodeError>
where
    'mode: 'pkt,
{
    #[cfg(feature = "fixed_point")]
    sync_from_fixed_primary_to_float_cache(decoder);
    let result =
        celt_decode_with_ec_dred_impl(decoder, packet, pcm, frame_size, range_decoder, accum, plc);
    #[cfg(feature = "fixed_point")]
    if matches!(packet, Some(data) if data.len() > 1) {
        sync_loge_to_fixed(&mut decoder.old_ebands_fixed, &decoder.old_ebands);
        sync_loge_to_fixed(&mut decoder.old_log_e_fixed, &decoder.old_log_e);
        sync_loge_to_fixed(&mut decoder.old_log_e2_fixed, &decoder.old_log_e2);
        sync_loge_to_fixed(
            &mut decoder.background_log_e_fixed,
            &decoder.background_log_e,
        );
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn celt_decode_with_ec_dred_impl<'mode, 'pkt>(
    decoder: &mut OpusCustomDecoder<'mode>,
    packet: Option<&'pkt [u8]>,
    pcm: &mut [OpusRes],
    frame_size: usize,
    range_decoder: Option<&'pkt mut EcDec<'pkt>>,
    accum: bool,
    plc: PlcHandle<'_>,
) -> Result<usize, CeltDecodeError>
where
    'mode: 'pkt,
{
    validate_celt_decoder(decoder);

    let data = packet.unwrap_or(&[]);
    let mut frame = prepare_frame(decoder, data, frame_size, range_decoder)?;
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
        celt_decode_lost(decoder, n, frame.lm, plc);
        let start = DECODE_BUFFER_SIZE
            .checked_sub(n)
            .ok_or(CeltDecodeError::BadArgument)?;

        #[cfg(feature = "fixed_point")]
        {
            debug_assert!(cc <= MAX_CHANNELS);
            let mut inputs: [&[FixedCeltSig]; MAX_CHANNELS] = [&[]; MAX_CHANNELS];
            for (channel, channel_slice) in decoder
                .decode_mem_fixed
                .chunks_mut(stride)
                .take(cc)
                .enumerate()
            {
                let (_, rest) = channel_slice.split_at(start);
                let (output, _) = rest.split_at(n);
                inputs[channel] = output;
            }

            deemphasis_fixed(
                &inputs[..cc],
                pcm,
                n,
                cc,
                downsample,
                &mode.pre_emphasis,
                &mut decoder.fixed_preemph_mem_decoder,
                accum,
            );
            for (dst, &src) in decoder
                .preemph_mem_decoder
                .iter_mut()
                .zip(decoder.fixed_preemph_mem_decoder.iter())
            {
                *dst = fixed_sig_to_float(src);
            }
        }

        #[cfg(not(feature = "fixed_point"))]
        {
            debug_assert!(cc <= MAX_CHANNELS);
            let mut inputs: [&[CeltSig]; MAX_CHANNELS] = [&[]; MAX_CHANNELS];
            for (channel, channel_slice) in
                decoder.decode_mem.chunks_mut(stride).take(cc).enumerate()
            {
                let (_, rest) = channel_slice.split_at(start);
                let (output, _) = rest.split_at(n);
                inputs[channel] = output;
            }

            deemphasis(
                &inputs[..cc],
                pcm,
                n,
                cc,
                downsample,
                &mode.pre_emphasis,
                &mut decoder.preemph_mem_decoder,
                accum,
            );
        }

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

    let move_len = DECODE_BUFFER_SIZE
        .checked_sub(n)
        .ok_or(CeltDecodeError::BadArgument)?
        + overlap;
    for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
        channel_slice.copy_within(n..n + move_len, 0);
    }
    #[cfg(feature = "fixed_point")]
    for channel_slice in decoder.decode_mem_fixed.chunks_mut(stride).take(cc) {
        channel_slice.copy_within(n..n + move_len, 0);
    }

    let mut collapse_masks = vec![0u8; c * nb_ebands];
    let total_available = total_bits - anti_collapse_rsv;
    #[cfg(feature = "fixed_point")]
    let mut spectrum_fixed = vec![0i16; c * n];
    #[cfg(not(feature = "fixed_point"))]
    let mut spectrum = vec![0.0f32; c * n];

    {
        let dec = range_state.decoder();
        let mut coder = BandCodingState::Decoder(dec);

        #[cfg(feature = "fixed_point")]
        {
            #[cfg(test)]
            let use_stereo_float_bridge =
                c == 2 && std::env::var("CELT_EXP_STEREO_FLOAT_BRIDGE").is_ok();
            #[cfg(not(test))]
            let use_stereo_float_bridge = false;

            if use_stereo_float_bridge {
                let mut spectrum_float = vec![0.0f32; c * n];
                let (left, right) = spectrum_float.split_at_mut(n);
                crate::celt::bands::quant_all_bands(
                    false,
                    mode,
                    start,
                    end,
                    left,
                    Some(right),
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
                float_norm_slice_to_fixed(&mut spectrum_fixed, &spectrum_float);
            } else {
                let (first_channel, second_channel_opt) = if c == 2 {
                    let (left, right) = spectrum_fixed.split_at_mut(n);
                    (left, Some(right))
                } else {
                    (&mut spectrum_fixed[..], None)
                };
                quant_all_bands_decode_fixed(
                    mode,
                    start,
                    end,
                    first_channel,
                    second_channel_opt,
                    &mut collapse_masks,
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
                    decoder.arch,
                    decoder.disable_inv,
                );
            }
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            let (first_channel, second_channel_opt) = if c == 2 {
                let (left, right) = spectrum.split_at_mut(n);
                (left, Some(right))
            } else {
                (&mut spectrum[..], None)
            };
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
    }
    let mut anti_collapse_on = false;
    let dec = range_state.decoder();

    if anti_collapse_rsv > 0 {
        anti_collapse_on = dec.dec_bits(1) != 0;
    }

    let payload_len_bits = (dec.ctx().storage as OpusInt32) * 8;
    let remaining_bits = payload_len_bits - entcode::ec_tell(dec.ctx());
    #[cfg(all(test, feature = "fixed_point"))]
    if std::env::var("CELT_TRACE_POST_BANDS").is_ok() {
        let pre_final_hash = decoder.old_ebands_fixed[..mode.num_ebands].iter().fold(
            2166136261u32,
            |hash, &value| {
                let v = value as u32;
                let hash = (hash ^ (v & 0xFF)).wrapping_mul(16777619);
                let hash = (hash ^ ((v >> 8) & 0xFF)).wrapping_mul(16777619);
                let hash = (hash ^ ((v >> 16) & 0xFF)).wrapping_mul(16777619);
                (hash ^ ((v >> 24) & 0xFF)).wrapping_mul(16777619)
            },
        );
        crate::test_trace::trace_println!(
            "pre_finalise.eb_hash=0x{:08x}.eb_values={:?}.remaining_bits={} fine_quant={:?} fine_priority={:?}",
            pre_final_hash,
            &decoder.old_ebands_fixed[..mode.num_ebands],
            remaining_bits,
            &fine_quant[start..end],
            &fine_priority[start..end]
        );
    }
    #[cfg(feature = "fixed_point")]
    {
        unquant_energy_finalise_fixed(
            mode,
            start,
            end,
            &mut decoder.old_ebands_fixed,
            &fine_quant,
            &fine_priority,
            remaining_bits,
            dec,
            c,
        );
        sync_loge_from_fixed(&mut decoder.old_ebands, &decoder.old_ebands_fixed);
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        unquant_energy_finalise(
            mode,
            start,
            end,
            &mut decoder.old_ebands,
            &fine_quant,
            &fine_priority,
            remaining_bits,
            dec,
            c,
        );
    }

    #[cfg(all(test, feature = "fixed_point"))]
    if std::env::var("CELT_TRACE_POST_BANDS").is_ok() && spectrum_fixed.len() >= 232 {
        crate::test_trace::trace_println!(
            "actual_band15.stage=pre_anti.on={anti_collapse_on}.hash=0x{:08x}.coeffs={:?}",
            spectrum_fixed.iter().fold(2166136261u32, |hash, &value| {
                let v = value as u16;
                let hash = (hash ^ u32::from(v & 0xFF)).wrapping_mul(16777619);
                (hash ^ u32::from(v >> 8)).wrapping_mul(16777619)
            }),
            &spectrum_fixed[224..272]
        );
    }

    if anti_collapse_on {
        #[cfg(feature = "fixed_point")]
        {
            anti_collapse_fixed(
                mode,
                &mut spectrum_fixed,
                &collapse_masks,
                lm,
                c,
                n,
                start,
                end,
                &decoder.old_ebands_fixed,
                &decoder.old_log_e_fixed,
                &decoder.old_log_e2_fixed,
                &pulses,
                decoder.rng,
                false,
                decoder.arch,
            );
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            anti_collapse(
                mode,
                &mut spectrum,
                &collapse_masks,
                lm,
                c,
                n,
                start,
                end,
                &decoder.old_ebands,
                &decoder.old_log_e,
                &decoder.old_log_e2,
                &pulses,
                decoder.rng,
                false,
                decoder.arch,
            );
        }
    }

    #[cfg(all(test, feature = "fixed_point"))]
    if std::env::var("CELT_TRACE_POST_BANDS").is_ok() && spectrum_fixed.len() >= 232 {
        let spectrum_hash = spectrum_fixed.iter().fold(2166136261u32, |hash, &value| {
            let v = value as u16;
            let hash = (hash ^ u32::from(v & 0xFF)).wrapping_mul(16777619);
            (hash ^ u32::from(v >> 8)).wrapping_mul(16777619)
        });
        let eb_hash = decoder.old_ebands_fixed[..mode.num_ebands].iter().fold(
            2166136261u32,
            |hash, &value| {
                let v = value as u32;
                let hash = (hash ^ (v & 0xFF)).wrapping_mul(16777619);
                let hash = (hash ^ ((v >> 8) & 0xFF)).wrapping_mul(16777619);
                let hash = (hash ^ ((v >> 16) & 0xFF)).wrapping_mul(16777619);
                (hash ^ ((v >> 24) & 0xFF)).wrapping_mul(16777619)
            },
        );
        crate::test_trace::trace_println!(
            "actual_band15.stage=post_anti.on={anti_collapse_on}.hash=0x{:08x}.eb_hash=0x{:08x}.eb_values={:?}.coeffs={:?}",
            spectrum_hash,
            eb_hash,
            &decoder.old_ebands_fixed[..mode.num_ebands],
            &spectrum_fixed[224..272]
        );
        for band_hash in 0..mode.num_ebands {
            let band_start = (mode.e_bands[band_hash] as usize) << lm;
            let band_end = (mode.e_bands[band_hash + 1] as usize) << lm;
            let hash =
                spectrum_fixed[band_start..band_end]
                    .iter()
                    .fold(2166136261u32, |hash, &value| {
                        let v = value as u16;
                        let hash = (hash ^ u32::from(v & 0xFF)).wrapping_mul(16777619);
                        (hash ^ u32::from(v >> 8)).wrapping_mul(16777619)
                    });
            crate::test_trace::trace_println!("post_anti_band_hash[{}]=0x{:08x}", band_hash, hash);
        }
    }

    if silence {
        decoder.old_ebands.fill(-28.0);
        #[cfg(feature = "fixed_point")]
        {
            sync_loge_to_fixed(&mut decoder.old_ebands_fixed, &decoder.old_ebands);
        }
    }

    if decoder.prefilter_and_fold {
        prefilter_and_fold(decoder, n);
    }

    #[cfg(feature = "fixed_point")]
    {
        let start_idx = DECODE_BUFFER_SIZE
            .checked_sub(n)
            .ok_or(CeltDecodeError::BadArgument)?;
        #[cfg(test)]
        if cc == 2 && std::env::var("CELT_EXP_FLOAT_SYNTH").is_ok() {
            let mut spectrum_float = vec![0.0f32; spectrum_fixed.len()];
            fixed_norm_slice_to_float(&mut spectrum_float, &spectrum_fixed);
            let mut out_slices_float: Vec<&mut [CeltSig]> = Vec::with_capacity(cc);
            for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
                out_slices_float.push(&mut channel_slice[start_idx..]);
            }
            celt_synthesis(
                mode,
                &spectrum_float,
                &mut out_slices_float,
                &decoder.old_ebands,
                start,
                eff_end,
                c,
                cc,
                is_transient,
                lm,
                downsample,
                silence,
                (
                    &decoder.fixed_mdct,
                    decoder.fixed_window.as_slice(),
                    decoder.overlap,
                ),
            );
            for channel in 0..cc {
                let float_base = channel * stride + start_idx;
                let fixed_base = channel * stride + start_idx;
                for i in 0..(n + overlap) {
                    decoder.decode_mem_fixed[fixed_base + i] =
                        celt_sig_to_fixed(decoder.decode_mem[float_base + i]);
                }
            }
        } else {
            let mut out_slices_fixed: Vec<&mut [FixedCeltSig]> = Vec::with_capacity(cc);
            for channel_slice in decoder.decode_mem_fixed.chunks_mut(stride).take(cc) {
                out_slices_fixed.push(&mut channel_slice[start_idx..]);
            }
            celt_synthesis_fixed_native(
                mode,
                &spectrum_fixed,
                &mut out_slices_fixed,
                &decoder.old_ebands_fixed,
                start,
                eff_end,
                c,
                cc,
                is_transient,
                lm,
                downsample,
                silence,
                &decoder.fixed_mdct,
                decoder.fixed_window.as_slice(),
                decoder.overlap,
            );
        }
        #[cfg(not(test))]
        {
            let mut out_slices_fixed: Vec<&mut [FixedCeltSig]> = Vec::with_capacity(cc);
            for channel_slice in decoder.decode_mem_fixed.chunks_mut(stride).take(cc) {
                out_slices_fixed.push(&mut channel_slice[start_idx..]);
            }
            celt_synthesis_fixed_native(
                mode,
                &spectrum_fixed,
                &mut out_slices_fixed,
                &decoder.old_ebands_fixed,
                start,
                eff_end,
                c,
                cc,
                is_transient,
                lm,
                downsample,
                silence,
                &decoder.fixed_mdct,
                decoder.fixed_window.as_slice(),
                decoder.overlap,
            );
        }
        #[cfg(test)]
        if std::env::var("CELT_TRACE_SYNTH_CMP").is_ok() && c == 1 && cc == 1 {
            let mut spectrum_float = vec![0.0f32; spectrum_fixed.len()];
            fixed_norm_slice_to_float(&mut spectrum_float, &spectrum_fixed);
            let mut freq_native = vec![0i32; n];
            denormalise_bands_fixed_native(
                mode,
                &spectrum_fixed,
                &mut freq_native,
                &decoder.old_ebands_fixed,
                start,
                eff_end,
                1 << lm,
                downsample,
                silence,
            );
            let mut freq_bridge = vec![0i32; n];
            denormalise_bands_fixed(
                mode,
                &spectrum_float,
                &mut freq_bridge,
                &decoder.old_ebands,
                start,
                eff_end,
                1 << lm,
                downsample,
                silence,
            );
            let mut hash_freq_native = 2166136261u32;
            for &value in freq_native.iter() {
                let v = value as u32;
                hash_freq_native = (hash_freq_native ^ (v & 0xFF)).wrapping_mul(16777619);
                hash_freq_native = (hash_freq_native ^ ((v >> 8) & 0xFF)).wrapping_mul(16777619);
                hash_freq_native = (hash_freq_native ^ ((v >> 16) & 0xFF)).wrapping_mul(16777619);
                hash_freq_native = (hash_freq_native ^ ((v >> 24) & 0xFF)).wrapping_mul(16777619);
            }
            let mut hash_freq_bridge = 2166136261u32;
            for &value in freq_bridge.iter() {
                let v = value as u32;
                hash_freq_bridge = (hash_freq_bridge ^ (v & 0xFF)).wrapping_mul(16777619);
                hash_freq_bridge = (hash_freq_bridge ^ ((v >> 8) & 0xFF)).wrapping_mul(16777619);
                hash_freq_bridge = (hash_freq_bridge ^ ((v >> 16) & 0xFF)).wrapping_mul(16777619);
                hash_freq_bridge = (hash_freq_bridge ^ ((v >> 24) & 0xFF)).wrapping_mul(16777619);
            }
            crate::test_trace::trace_println!(
                "freqcmp native=0x{:08x} bridge=0x{:08x} first_diff={:?}",
                hash_freq_native,
                hash_freq_bridge,
                freq_native
                    .iter()
                    .zip(freq_bridge.iter())
                    .position(|(lhs, rhs)| lhs != rhs)
            );
            let mut bridge_out = vec![0.0f32; n + overlap];
            let mut bridge_views: Vec<&mut [CeltSig]> = vec![bridge_out.as_mut_slice()];
            celt_synthesis(
                mode,
                &spectrum_float,
                &mut bridge_views,
                &decoder.old_ebands,
                start,
                eff_end,
                c,
                cc,
                is_transient,
                lm,
                downsample,
                silence,
                (
                    &decoder.fixed_mdct,
                    decoder.fixed_window.as_slice(),
                    decoder.overlap,
                ),
            );
            let synth_start = DECODE_BUFFER_SIZE - n;
            let actual = &decoder.decode_mem_fixed[synth_start..synth_start + n];
            let mut bridge_fixed = vec![0i32; n];
            for (dst, &src) in bridge_fixed.iter_mut().zip(bridge_out.iter()) {
                *dst = celt_sig_to_fixed(src);
            }
            let mut hash_actual = 2166136261u32;
            for &value in actual.iter() {
                let v = value as u32;
                hash_actual = (hash_actual ^ (v & 0xFF)).wrapping_mul(16777619);
                hash_actual = (hash_actual ^ ((v >> 8) & 0xFF)).wrapping_mul(16777619);
                hash_actual = (hash_actual ^ ((v >> 16) & 0xFF)).wrapping_mul(16777619);
                hash_actual = (hash_actual ^ ((v >> 24) & 0xFF)).wrapping_mul(16777619);
            }
            let mut hash_bridge = 2166136261u32;
            for &value in bridge_fixed.iter() {
                let v = value as u32;
                hash_bridge = (hash_bridge ^ (v & 0xFF)).wrapping_mul(16777619);
                hash_bridge = (hash_bridge ^ ((v >> 8) & 0xFF)).wrapping_mul(16777619);
                hash_bridge = (hash_bridge ^ ((v >> 16) & 0xFF)).wrapping_mul(16777619);
                hash_bridge = (hash_bridge ^ ((v >> 24) & 0xFF)).wrapping_mul(16777619);
            }
            let first_diff = actual
                .iter()
                .zip(bridge_fixed.iter())
                .position(|(lhs, rhs)| lhs != rhs);
            crate::test_trace::trace_println!(
                "synthcmp actual=0x{:08x} bridge=0x{:08x} first_diff={:?}",
                hash_actual,
                hash_bridge,
                first_diff
            );
            if let Some(idx) = first_diff {
                crate::test_trace::trace_println!(
                    "synthcmp coeff[{idx}] actual={} bridge={} tail_actual=0x{:08x} tail_bridge=0x{:08x}",
                    actual[idx],
                    bridge_fixed[idx],
                    {
                        let mut hash = 2166136261u32;
                        for &value in actual[n - 64..n].iter() {
                            let v = value as u32;
                            hash = (hash ^ (v & 0xFF)).wrapping_mul(16777619);
                            hash = (hash ^ ((v >> 8) & 0xFF)).wrapping_mul(16777619);
                            hash = (hash ^ ((v >> 16) & 0xFF)).wrapping_mul(16777619);
                            hash = (hash ^ ((v >> 24) & 0xFF)).wrapping_mul(16777619);
                        }
                        hash
                    },
                    {
                        let mut hash = 2166136261u32;
                        for &value in bridge_fixed[n - 64..n].iter() {
                            let v = value as u32;
                            hash = (hash ^ (v & 0xFF)).wrapping_mul(16777619);
                            hash = (hash ^ ((v >> 8) & 0xFF)).wrapping_mul(16777619);
                            hash = (hash ^ ((v >> 16) & 0xFF)).wrapping_mul(16777619);
                            hash = (hash ^ ((v >> 24) & 0xFF)).wrapping_mul(16777619);
                        }
                        hash
                    }
                );
            }
        }
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        let mut out_slices: Vec<&mut [CeltSig]> = Vec::with_capacity(cc);
        for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
            let start_idx = DECODE_BUFFER_SIZE
                .checked_sub(n)
                .ok_or(CeltDecodeError::BadArgument)?;
            out_slices.push(&mut channel_slice[start_idx..]);
        }
        celt_synthesis(
            mode,
            &spectrum,
            &mut out_slices,
            &decoder.old_ebands,
            start,
            eff_end,
            c,
            cc,
            is_transient,
            lm,
            downsample,
            silence,
            (),
        );
    }

    decoder.postfilter_period = decoder.postfilter_period.max(COMBFILTER_MINPERIOD as i32);
    decoder.postfilter_period_old = decoder
        .postfilter_period_old
        .max(COMBFILTER_MINPERIOD as i32);

    let output_start = DECODE_BUFFER_SIZE - n;
    #[cfg(not(feature = "fixed_point"))]
    for channel_slice in decoder.decode_mem.chunks_mut(stride).take(cc) {
        let first_len = mode.short_mdct_size.min(n);
        if first_len > 0 {
            comb_filter_in_place(
                channel_slice,
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

            if lm != 0 && first_len < n {
                comb_filter_in_place(
                    channel_slice,
                    output_start + first_len,
                    n - first_len,
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

    #[cfg(feature = "fixed_point")]
    {
        let g0 = postfilter_gain_to_fixed(decoder.postfilter_gain_old);
        let g1 = postfilter_gain_to_fixed(decoder.postfilter_gain);
        #[cfg(test)]
        if std::env::var("CELT_DUMP_POSTFILTER_INPUT").is_ok() {
            static POSTFILTER_DUMP_COUNTER: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            if POSTFILTER_DUMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
                let synth = &decoder.decode_mem_fixed[output_start..output_start + n];
                crate::test_trace::trace_println!(
                    "postfilter_input len={} first_len={} overlap={} old_period={} cur_period={} next_period={} g0={} g1={} g_next={} tap0={} tap1={} tap_next={}",
                    n,
                    mode.short_mdct_size.min(n),
                    overlap,
                    decoder
                        .postfilter_period_old
                        .max(COMBFILTER_MINPERIOD as i32),
                    decoder.postfilter_period.max(COMBFILTER_MINPERIOD as i32),
                    postfilter_pitch,
                    g0,
                    g1,
                    postfilter_gain_to_fixed(postfilter_gain),
                    decoder.postfilter_tapset_old.max(0),
                    decoder.postfilter_tapset.max(0),
                    postfilter_tapset.max(0),
                );
                for (idx, value) in synth.iter().enumerate() {
                    crate::test_trace::trace_println!("postfilter_input[{idx}]={value}");
                }
            }
        }
        for channel_slice in decoder.decode_mem_fixed.chunks_mut(stride).take(cc) {
            let first_len = mode.short_mdct_size.min(n);

            if first_len > 0 {
                comb_filter_fixed_in_place(
                    channel_slice,
                    output_start,
                    first_len,
                    decoder.postfilter_period_old,
                    decoder.postfilter_period,
                    g0,
                    g1,
                    decoder.postfilter_tapset_old.max(0) as usize,
                    decoder.postfilter_tapset.max(0) as usize,
                    decoder.fixed_window.as_slice(),
                    overlap,
                    decoder.arch,
                );

                if lm != 0 && first_len < n {
                    comb_filter_fixed_in_place(
                        channel_slice,
                        output_start + first_len,
                        n - first_len,
                        decoder.postfilter_period,
                        postfilter_pitch,
                        g1,
                        postfilter_gain_to_fixed(postfilter_gain),
                        decoder.postfilter_tapset.max(0) as usize,
                        postfilter_tapset.max(0) as usize,
                        decoder.fixed_window.as_slice(),
                        overlap,
                        decoder.arch,
                    );
                }
            }
        }
        sync_fixed_output_window_to_float_cache(decoder, cc, output_start, n + (overlap >> 1));
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
        decoder.old_log_e2.copy_from_slice(&decoder.old_log_e);
        decoder.old_log_e.copy_from_slice(&decoder.old_ebands);
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
    #[cfg(feature = "fixed_point")]
    {
        debug_assert!(cc <= MAX_CHANNELS);
        let mut deemph_inputs: [&[FixedCeltSig]; MAX_CHANNELS] = [&[]; MAX_CHANNELS];
        for (channel, channel_slice) in decoder
            .decode_mem_fixed
            .chunks_mut(stride)
            .take(cc)
            .enumerate()
        {
            let start_idx = DECODE_BUFFER_SIZE - n;
            let (_, rest) = channel_slice.split_at_mut(start_idx);
            let (output, _) = rest.split_at_mut(n);
            deemph_inputs[channel] = output;
        }

        deemphasis_fixed(
            &deemph_inputs[..cc],
            pcm,
            n,
            cc,
            downsample,
            &mode.pre_emphasis,
            &mut decoder.fixed_preemph_mem_decoder,
            accum,
        );
        for (dst, &src) in decoder
            .preemph_mem_decoder
            .iter_mut()
            .zip(decoder.fixed_preemph_mem_decoder.iter())
        {
            *dst = fixed_sig_to_float(src);
        }
    }

    #[cfg(not(feature = "fixed_point"))]
    {
        debug_assert!(cc <= MAX_CHANNELS);
        let mut deemph_inputs: [&[CeltSig]; MAX_CHANNELS] = [&[]; MAX_CHANNELS];
        for (channel, channel_slice) in decoder.decode_mem.chunks_mut(stride).take(cc).enumerate() {
            let start_idx = DECODE_BUFFER_SIZE - n;
            let (_, rest) = channel_slice.split_at_mut(start_idx);
            let (output, _) = rest.split_at_mut(n);
            deemph_inputs[channel] = output;
        }

        deemphasis(
            &deemph_inputs[..cc],
            pcm,
            n,
            cc,
            downsample,
            &mode.pre_emphasis,
            &mut decoder.preemph_mem_decoder,
            accum,
        );
    }

    decoder.loss_duration = 0;
    decoder.prefilter_and_fold = false;

    if entcode::ec_tell(dec.ctx()) > payload_len_bits {
        return Err(CeltDecodeError::InvalidPacket);
    }

    if dec.ctx().error() != 0 {
        decoder.error = 1;
    }

    Ok(output_samples)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn celt_decode_with_ec<'mode, 'pkt>(
    decoder: &mut OpusCustomDecoder<'mode>,
    packet: Option<&'pkt [u8]>,
    pcm: &mut [OpusRes],
    frame_size: usize,
    range_decoder: Option<&'pkt mut EcDec<'pkt>>,
    accum: bool,
) -> Result<usize, CeltDecodeError>
where
    'mode: 'pkt,
{
    #[cfg(feature = "deep_plc")]
    let plc = None;
    #[cfg(not(feature = "deep_plc"))]
    let plc = ();
    celt_decode_with_ec_dred(decoder, packet, pcm, frame_size, range_decoder, accum, plc)
}

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

    #[cfg(feature = "fixed_point")]
    let start_preemph_mem = decoder.fixed_preemph_mem_decoder;
    let mut temp = vec![0.0f32; required];
    let samples = celt_decode_with_ec(decoder, packet, &mut temp, frame_size, None, false)?;
    #[cfg(all(test, feature = "fixed_point"))]
    if channels == 1 && samples >= 891 {
        for &idx in &[208usize, 680, 890] {
            crate::test_trace::trace_println!(
                "opus_custom_decode temp[{idx}]={} scaled={}",
                temp[idx],
                temp[idx] * CELT_SIG_SCALE
            );
        }
    }

    #[cfg(feature = "fixed_point")]
    {
        let n = samples * decoder.downsample.max(1) as usize;
        let stride = DECODE_BUFFER_SIZE + decoder.overlap;
        let start_idx = DECODE_BUFFER_SIZE - n;
        debug_assert!(channels <= MAX_CHANNELS);
        let mut inputs: [&[FixedCeltSig]; MAX_CHANNELS] = [&[]; MAX_CHANNELS];
        for (channel, channel_slice) in decoder
            .decode_mem_fixed
            .chunks(stride)
            .take(channels)
            .enumerate()
        {
            inputs[channel] = &channel_slice[start_idx..start_idx + n];
        }
        let mut mem = start_preemph_mem;
        deemphasis_fixed_to_int16(
            &inputs[..channels],
            pcm,
            n,
            channels,
            decoder.downsample.max(1) as usize,
            &decoder.mode.pre_emphasis,
            &mut mem,
        );
    }

    #[cfg(not(feature = "fixed_point"))]
    for (dst, &src) in pcm.iter_mut().zip(temp.iter().take(samples * channels)) {
        *dst = float2int16(src);
    }
    Ok(samples)
}

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
    #[cfg(feature = "fixed_point")]
    decode_mem_fixed: Vec<FixedCeltSig>,
    #[cfg(feature = "fixed_point")]
    lpc_fixed: Vec<FixedOpusVal16>,
    #[cfg(feature = "fixed_point")]
    old_ebands_fixed: Vec<FixedCeltGlog>,
    #[cfg(feature = "fixed_point")]
    old_log_e_fixed: Vec<FixedCeltGlog>,
    #[cfg(feature = "fixed_point")]
    old_log_e2_fixed: Vec<FixedCeltGlog>,
    #[cfg(feature = "fixed_point")]
    background_log_e_fixed: Vec<FixedCeltGlog>,
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
            #[cfg(feature = "fixed_point")]
            decode_mem_fixed: vec![0; decode_mem],
            #[cfg(feature = "fixed_point")]
            lpc_fixed: vec![0; lpc],
            #[cfg(feature = "fixed_point")]
            old_ebands_fixed: vec![0; band_count],
            #[cfg(feature = "fixed_point")]
            old_log_e_fixed: vec![0; band_count],
            #[cfg(feature = "fixed_point")]
            old_log_e2_fixed: vec![0; band_count],
            #[cfg(feature = "fixed_point")]
            background_log_e_fixed: vec![0; band_count],
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
        #[cfg(feature = "fixed_point")]
        {
            debug_assert_eq!(self.decode_mem_fixed.len(), decode_mem);
            debug_assert_eq!(self.lpc_fixed.len(), self.lpc.len());
            debug_assert_eq!(self.old_ebands_fixed.len(), band_history);
            debug_assert_eq!(self.old_log_e_fixed.len(), band_history);
            debug_assert_eq!(self.old_log_e2_fixed.len(), band_history);
            debug_assert_eq!(self.background_log_e_fixed.len(), band_history);
        }

        DECODER_PREFIX_SIZE
            + (decode_mem - 1) * core::mem::size_of::<CeltSig>()
            + self.lpc.len() * core::mem::size_of::<OpusVal16>()
            + 4 * band_history * core::mem::size_of::<CeltGlog>()
            + {
                #[cfg(feature = "fixed_point")]
                {
                    self.decode_mem_fixed.len() * core::mem::size_of::<FixedCeltSig>()
                        + self.lpc_fixed.len() * core::mem::size_of::<FixedOpusVal16>()
                        + (self.old_ebands_fixed.len()
                            + self.old_log_e_fixed.len()
                            + self.old_log_e2_fixed.len()
                            + self.background_log_e_fixed.len())
                            * core::mem::size_of::<FixedCeltGlog>()
                }
                #[cfg(not(feature = "fixed_point"))]
                {
                    0
                }
            }
    }

    /// Borrows the allocation as an [`OpusCustomDecoder`] tied to the provided
    /// mode.
    ///
    /// Each call returns a fresh decoder view referencing the same backing
    /// buffers.  This mirrors the C layout where the state and trailing memory
    /// occupy a single blob, enabling the caller to reset or reuse the decoder
    /// without reallocating.
    pub(crate) fn as_decoder<'mode>(
        &mut self,
        mode: &'mode OpusCustomMode<'mode>,
        channels: usize,
        stream_channels: usize,
    ) -> OpusCustomDecoder<'mode> {
        #[cfg(feature = "fixed_point")]
        {
            OpusCustomDecoder::new(
                mode,
                channels,
                stream_channels,
                core::mem::take(&mut self.decode_mem_fixed).into_boxed_slice(),
                core::mem::take(&mut self.lpc_fixed).into_boxed_slice(),
                core::mem::take(&mut self.old_ebands_fixed).into_boxed_slice(),
                core::mem::take(&mut self.old_log_e_fixed).into_boxed_slice(),
                core::mem::take(&mut self.old_log_e2_fixed).into_boxed_slice(),
                core::mem::take(&mut self.background_log_e_fixed).into_boxed_slice(),
                core::mem::take(&mut self.decode_mem).into_boxed_slice(),
                core::mem::take(&mut self.lpc).into_boxed_slice(),
                core::mem::take(&mut self.old_ebands).into_boxed_slice(),
                core::mem::take(&mut self.old_log_e).into_boxed_slice(),
                core::mem::take(&mut self.old_log_e2).into_boxed_slice(),
                core::mem::take(&mut self.background_log_e).into_boxed_slice(),
            )
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            OpusCustomDecoder::new(
                mode,
                channels,
                stream_channels,
                core::mem::take(&mut self.decode_mem).into_boxed_slice(),
                core::mem::take(&mut self.lpc).into_boxed_slice(),
                core::mem::take(&mut self.old_ebands).into_boxed_slice(),
                core::mem::take(&mut self.old_log_e).into_boxed_slice(),
                core::mem::take(&mut self.old_log_e2).into_boxed_slice(),
                core::mem::take(&mut self.background_log_e).into_boxed_slice(),
            )
        }
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
        #[cfg(feature = "fixed_point")]
        {
            self.decode_mem_fixed.fill(0);
            self.lpc_fixed.fill(0);
            self.old_ebands_fixed.fill(0);
            self.old_log_e_fixed.fill(0);
            self.old_log_e2_fixed.fill(0);
            self.background_log_e_fixed.fill(0);
        }
    }

    /// Prepares a decoder view that mirrors the default initialisation state.
    ///
    /// The helper ports the zeroing performed by `opus_custom_decoder_init()`
    /// and the follow-up `OPUS_RESET_STATE` call by validating the channel
    /// layout, borrowing the trailing buffers, and clearing the runtime state.
    fn prepare_decoder<'mode>(
        &mut self,
        mode: &'mode OpusCustomMode<'mode>,
        channels: usize,
        stream_channels: usize,
    ) -> Result<OpusCustomDecoder<'mode>, CeltDecoderInitError> {
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
    pub(crate) fn init_decoder<'mode>(
        &mut self,
        mode: &'mode OpusCustomMode<'mode>,
        channels: usize,
        stream_channels: usize,
    ) -> Result<OpusCustomDecoder<'mode>, CeltDecoderInitError> {
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
pub(crate) fn opus_custom_decoder_init<'mode>(
    alloc: &mut CeltDecoderAlloc,
    mode: &'mode OpusCustomMode<'mode>,
    channels: usize,
) -> Result<OpusCustomDecoder<'mode>, CeltDecoderInitError> {
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
    let mut alloc = CeltDecoderAlloc::new(mode, channels);
    let decoder = alloc.prepare_decoder(mode, channels, channels)?;
    Ok(OwnedCeltDecoder { decoder })
}

/// Releases the resources owned by [`OwnedCeltDecoder`].
///
/// Consuming it mirrors the behaviour of `opus_custom_decoder_destroy()` in C.
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
            if value == 0 || value > MAX_CHANNELS {
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
    extern crate std;

    #[cfg(feature = "fixed_point")]
    use super::FIXED_SIG_SAT;
    use super::celt_decode_lost;
    use super::celt_plc_pitch_search;
    #[cfg(feature = "fixed_point")]
    use super::celt_plc_pitch_search_fixed;
    #[cfg(feature = "fixed_point")]
    use super::decoder_noise_renormalise_runtime;
    use super::deemphasis;
    #[cfg(feature = "fixed_point")]
    use super::opus_custom_decode;
    use super::{
        CeltDecodeError, CeltDecoderAlloc, CeltDecoderCtlError, CeltDecoderInitError,
        DECODE_BUFFER_SIZE, DecoderCtlRequest, LPC_ORDER, MAX_CHANNELS, celt_decoder_get_size,
        celt_decoder_init, comb_filter, opus_custom_decoder_create, opus_custom_decoder_ctl,
        opus_custom_decoder_get_size, opus_custom_decoder_init, prefilter_and_fold, prepare_frame,
        tf_decode, validate_celt_decoder, validate_channel_layout,
    };
    use crate::celt::EcDec;
    #[cfg(feature = "fixed_point")]
    use crate::celt::bands::BandCodingState;
    #[cfg(feature = "fixed_point")]
    use crate::celt::celt_encoder::{
        EncoderCtlRequest, opus_custom_encode, opus_custom_encoder_create, opus_custom_encoder_ctl,
    };
    #[cfg(feature = "fixed_point")]
    use crate::celt::cwrs::decode_pulses_debug;
    #[cfg(feature = "fixed_point")]
    use crate::celt::entcode;
    #[cfg(feature = "fixed_point")]
    use crate::celt::entcode::celt_sudiv;
    #[cfg(feature = "fixed_point")]
    use crate::celt::fixed_arch::{DB_SHIFT, sig2word16};
    #[cfg(feature = "fixed_point")]
    use crate::celt::fixed_ops::{mult16_32_q15, qconst16_clamped, qconst32};
    use crate::celt::float_cast::CELT_SIG_SCALE;
    #[cfg(feature = "fixed_point")]
    use crate::celt::float_cast::float2int16;
    use crate::celt::modes::{opus_custom_mode_create, opus_custom_mode_find_static};
    use crate::celt::opus_select_arch;
    #[cfg(feature = "fixed_point")]
    use crate::celt::rate::{bits2pulses, get_pulses};
    #[cfg(feature = "fixed_point")]
    use crate::celt::types::FixedCeltSig;
    use crate::celt::types::{
        CeltGlog, CeltNorm, CeltSig, MdctLookup, OpusCustomMode, PulseCacheData,
    };
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::PI;
    use core::ptr;

    #[test]
    fn tf_decode_returns_default_table_entries() {
        let mut range = EcDec::new(&[0u8]);
        let mut tf_res = vec![0; 4];
        tf_decode(0, tf_res.len(), false, &mut tf_res, 0, &mut range);
        assert!(tf_res.iter().all(|&v| v == 0));
    }

    #[test]
    fn celt_plc_pitch_search_detects_mono_period() {
        let target_period = 320i32;
        let mut channel = vec![0.0; super::DECODE_BUFFER_SIZE];
        for (i, sample) in channel.iter_mut().enumerate() {
            let phase = 2.0 * PI * (i as f32) / target_period as f32;
            *sample = phase.sin();
        }

        let decode_mem = [&channel[..]];
        let pitch = celt_plc_pitch_search(&decode_mem, decode_mem.len(), 0);
        let doubled = target_period * 2;
        let matches = (pitch - target_period).abs() <= 2
            || (doubled <= super::PLC_PITCH_LAG_MAX && (pitch - doubled).abs() <= 2);
        assert!(
            matches,
            "pitch {pitch} deviates from target {target_period}",
        );
    }

    #[test]
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

    #[cfg(feature = "fixed_point")]
    fn fill_plc_ctest_periodic(
        channel: &mut [f32],
        period: usize,
        amp: i32,
        phase: usize,
        invert: bool,
    ) {
        let half = period / 2;
        for (i, sample) in channel.iter_mut().enumerate() {
            let pos = (i + phase) % period;
            let tri = if pos < half { pos } else { period - pos };
            let centered = (tri as i32 * 2) - half as i32;
            let mut value = (centered * amp) / half as i32;
            if invert {
                value = -value;
            }
            *sample = value as f32 / 32768.0;
        }
    }

    #[cfg(feature = "fixed_point")]
    fn fill_plc_ctest_periodic_sig(
        channel: &mut [f32],
        period: usize,
        amp: i32,
        phase: usize,
        invert: bool,
    ) {
        let half = period / 2;
        for (i, sample) in channel.iter_mut().enumerate() {
            let pos = (i + phase) % period;
            let tri = if pos < half { pos } else { period - pos };
            let centered = (tri as i32 * 2) - half as i32;
            let mut value = (centered * amp) / half as i32;
            if invert {
                value = -value;
            }
            *sample = value as f32;
        }
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn celt_plc_pitch_search_matches_ctest_periodic_mono_shape() {
        let mut mono = vec![0.0f32; super::DECODE_BUFFER_SIZE];
        fill_plc_ctest_periodic(&mut mono, 96, 14_000, 0, false);

        let decode_mem = [&mono[..]];
        let pitch = celt_plc_pitch_search(&decode_mem, decode_mem.len(), 0);
        let pitch_again = celt_plc_pitch_search(&decode_mem, decode_mem.len(), 0);
        assert_eq!(pitch, pitch_again, "pitch search should be deterministic");
        assert!(
            (1..=super::PLC_PITCH_LAG_MAX).contains(&pitch),
            "ctest-like mono pattern produced out-of-range pitch {pitch}",
        );
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn celt_plc_pitch_search_matches_ctest_periodic_stereo_shape() {
        let mut left = vec![0.0f32; super::DECODE_BUFFER_SIZE];
        let mut right = vec![0.0f32; super::DECODE_BUFFER_SIZE];
        fill_plc_ctest_periodic(&mut left, 96, 14_000, 0, false);
        fill_plc_ctest_periodic(&mut right, 96, 11_500, 23, true);

        let decode_mem = [&left[..], &right[..]];
        let pitch = celt_plc_pitch_search(&decode_mem, decode_mem.len(), 0);
        let pitch_again = celt_plc_pitch_search(&decode_mem, decode_mem.len(), 0);
        assert_eq!(pitch, pitch_again, "pitch search should be deterministic");
        assert!(
            (1..=super::PLC_PITCH_LAG_MAX).contains(&pitch),
            "ctest-like stereo pattern produced out-of-range pitch {pitch}",
        );
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn decoder_plc_decay_terms_match_ctest_vectors() {
        let mut exc = vec![0.0f32; 320];

        fill_plc_ctest_periodic_sig(&mut exc, 96, 14_000, 0, false);
        let (e1_a, e2_a, decay_a) = super::plc_decay_terms_fixed(&exc, exc.len());
        assert_eq!(e1_a, 148_945_491);
        assert_eq!(e2_a, 172_083_276);
        assert_eq!(decay_a, 30_486);

        fill_plc_ctest_periodic_sig(&mut exc, 96, 14_000, 0, false);
        let half = exc.len() / 2;
        for sample in &mut exc[half..] {
            let quantised = super::fixed_sig_to_word16(*sample);
            *sample = (quantised / 4) as f32;
        }
        let (e1_b, e2_b, decay_b) = super::plc_decay_terms_fixed(&exc, exc.len());
        assert_eq!(e1_b, 9_306_492);
        assert_eq!(e2_b, 172_083_276);
        assert_eq!(decay_b, 7_620);
        assert!(decay_b < decay_a);

        fill_plc_ctest_periodic_sig(&mut exc, 64, 3_000, 7, true);
        let third = exc.len() / 3;
        for sample in &mut exc[..third] {
            *sample = 0.0;
        }
        let (e1_c, e2_c, decay_c) = super::plc_decay_terms_fixed(&exc, exc.len());
        assert_eq!(e1_c, 45_578_035);
        assert_eq!(e2_c, 45_578_035);
        assert_eq!(decay_c, 32_767);
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn decoder_plc_ratio_terms_match_ctest_vectors() {
        let mut old_sig = vec![0.0f32; 144];
        let mut new_sig = vec![0.0f32; 144];

        fill_plc_ctest_periodic_sig(&mut old_sig, 72, 7_000, 0, false);
        fill_plc_ctest_periodic_sig(&mut new_sig, 72, 12_000, 11, true);
        let (s1_a, s2_a, ratio_a) = super::plc_ratio_terms_fixed(&old_sig, &new_sig);
        assert_eq!(s1_a, 2_299_964);
        assert_eq!(s2_a, 6_759_820);
        assert_eq!(ratio_a, 19_113);

        fill_plc_ctest_periodic_sig(&mut old_sig, 72, 12_000, 11, true);
        fill_plc_ctest_periodic_sig(&mut new_sig, 72, 7_000, 0, false);
        let (s1_b, s2_b, ratio_b) = super::plc_ratio_terms_fixed(&old_sig, &new_sig);
        assert_eq!(s1_b, 6_759_820);
        assert_eq!(s2_b, 2_299_964);
        assert_eq!(ratio_b, 32_767);

        old_sig.fill(0.0);
        new_sig.fill(0.0);
        let (s1_c, s2_c, ratio_c) = super::plc_ratio_terms_fixed(&old_sig, &new_sig);
        assert_eq!(s1_c, 0);
        assert_eq!(s2_c, 0);
        assert_eq!(ratio_c, 32_767);
    }

    #[test]
    fn allocates_expected_band_buffers() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new_test(
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
        #[cfg(feature = "fixed_point")]
        {
            assert_eq!(
                alloc.decode_mem_fixed.len(),
                2 * (super::DECODE_BUFFER_SIZE + mode.overlap)
            );
            assert_eq!(alloc.lpc_fixed.len(), LPC_ORDER * 2);
            assert_eq!(alloc.old_ebands_fixed.len(), 2 * mode.num_ebands);
            assert_eq!(alloc.old_log_e_fixed.len(), 2 * mode.num_ebands);
            assert_eq!(alloc.old_log_e2_fixed.len(), 2 * mode.num_ebands);
            assert_eq!(alloc.background_log_e_fixed.len(), 2 * mode.num_ebands);
        }

        // Ensure the reset helper clears all buffers.
        alloc.decode_mem.fill(1.0);
        alloc.lpc.fill(1.0);
        alloc.old_ebands.fill(1.0);
        alloc.old_log_e.fill(1.0);
        alloc.old_log_e2.fill(1.0);
        alloc.background_log_e.fill(1.0);
        #[cfg(feature = "fixed_point")]
        {
            alloc.decode_mem_fixed.fill(7);
            alloc.lpc_fixed.fill(7);
            alloc.old_ebands_fixed.fill(7);
            alloc.old_log_e_fixed.fill(7);
            alloc.old_log_e2_fixed.fill(7);
            alloc.background_log_e_fixed.fill(7);
        }
        alloc.reset();

        assert!(alloc.decode_mem.iter().all(|&v| v == 0.0));
        assert!(alloc.lpc.iter().all(|&v| v == 0.0));
        assert!(alloc.old_ebands.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e2.iter().all(|&v| v == 0.0));
        assert!(alloc.background_log_e.iter().all(|&v| v == 0.0));
        #[cfg(feature = "fixed_point")]
        {
            assert!(alloc.decode_mem_fixed.iter().all(|&v| v == 0));
            assert!(alloc.lpc_fixed.iter().all(|&v| v == 0));
            assert!(alloc.old_ebands_fixed.iter().all(|&v| v == 0));
            assert!(alloc.old_log_e_fixed.iter().all(|&v| v == 0));
            assert!(alloc.old_log_e2_fixed.iter().all(|&v| v == 0));
            assert!(alloc.background_log_e_fixed.iter().all(|&v| v == 0));
        }

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
        let mode = OpusCustomMode::new_test(
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
    #[cfg_attr(not(debug_assertions), ignore = "debug_assertions disabled in release")]
    #[should_panic]
    fn validate_celt_decoder_rejects_invalid_channel_count() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new_test(
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
        let mode = OpusCustomMode::new_test(
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
        let mode = OpusCustomMode::new_test(
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
        let mode = OpusCustomMode::new_test(
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
        let mode = OpusCustomMode::new_test(
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
        #[cfg(feature = "fixed_point")]
        {
            decoder.decode_mem_fixed.fill(7);
            decoder.lpc_fixed.fill(7);
            decoder.old_ebands_fixed.fill(7);
            decoder.old_log_e_fixed.fill(7);
            decoder.old_log_e2_fixed.fill(7);
            decoder.background_log_e_fixed.fill(7);
        }

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
        #[cfg(feature = "fixed_point")]
        {
            assert!(decoder.decode_mem_fixed.iter().all(|&v| v == 0));
            assert!(decoder.lpc_fixed.iter().all(|&v| v == 0));
            assert!(decoder.old_ebands_fixed.iter().all(|&v| v == 0));
            assert!(
                decoder
                    .old_log_e_fixed
                    .iter()
                    .all(|&v| v == qconst32(-28.0, DB_SHIFT))
            );
            assert!(
                decoder
                    .old_log_e2_fixed
                    .iter()
                    .all(|&v| v == qconst32(-28.0, DB_SHIFT))
            );
            assert!(decoder.background_log_e_fixed.iter().all(|&v| v == 0));
        }
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
        let mut mode = OpusCustomMode::new_test(
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

        let frame = prepare_frame(&mut decoder, &[], mode.short_mdct_size, None)
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
        let mut mode = OpusCustomMode::new_test(
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
        decoder.signalling = 0;

        let err = prepare_frame(&mut decoder, &[0u8; 2], mode.short_mdct_size + 1, None)
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
        let mode = OpusCustomMode::new_test(
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
            *sample = (idx as f32) * 0.0002;
        }
        #[cfg(feature = "fixed_point")]
        for (dst, &src) in decoder
            .decode_mem_fixed
            .iter_mut()
            .zip(decoder.decode_mem.iter())
        {
            *dst = super::celt_sig_to_fixed(src);
        }

        let n = 32;
        let start = DECODE_BUFFER_SIZE - n;
        let original = decoder.decode_mem[start..start + overlap].to_vec();

        decoder.postfilter_gain = 0.0;
        decoder.postfilter_gain_old = 0.0;
        decoder.postfilter_tapset = 0;
        decoder.postfilter_tapset_old = 0;

        prefilter_and_fold(&mut decoder, n);

        let tolerance = if cfg!(feature = "fixed_point") {
            2e-3f32
        } else {
            1e-6f32
        };
        for i in 0..(overlap / 2) {
            let expected =
                window[i] * original[overlap - 1 - i] + window[overlap - 1 - i] * original[i];
            let actual = decoder.decode_mem[start + i];
            assert!(
                (expected - actual).abs() <= tolerance,
                "folded sample {i} differs: expected {expected}, got {actual}",
            );
            #[cfg(feature = "fixed_point")]
            assert_eq!(
                decoder.decode_mem_fixed[start + i],
                super::celt_sig_to_fixed(actual),
                "fixed cache mismatch for folded sample {i}",
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
        let mode = OpusCustomMode::new_test(
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
            *sample = (idx as f32) * 0.0002;
        }
        #[cfg(feature = "fixed_point")]
        for (dst, &src) in decoder
            .decode_mem_fixed
            .iter_mut()
            .zip(decoder.decode_mem.iter())
        {
            *dst = super::celt_sig_to_fixed(src);
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

        let tolerance = if cfg!(feature = "fixed_point") {
            2e-3f32
        } else {
            1e-6f32
        };
        for i in 0..(overlap / 2) {
            let expected = window[i] * expected_filtered[overlap - 1 - i]
                + window[overlap - 1 - i] * expected_filtered[i];
            let actual = decoder.decode_mem[start + i];
            assert!(
                (expected - actual).abs() <= tolerance,
                "filtered fold {i} differs: expected {expected}, got {actual}",
            );
            #[cfg(feature = "fixed_point")]
            assert!(
                (decoder.decode_mem_fixed[start + i] - super::celt_sig_to_fixed(actual)).abs() <= 1,
                "fixed cache mismatch for filtered fold {i}: fixed={}, float2sig={}",
                decoder.decode_mem_fixed[start + i],
                super::celt_sig_to_fixed(actual),
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
        let mode = OpusCustomMode::new_test(
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
            decoder.decode_mem[i] = (i as f32) * 0.0002;
            decoder.decode_mem[stride + i] = -0.3 + (i as f32) * 0.00025;
        }
        #[cfg(feature = "fixed_point")]
        for (dst, &src) in decoder
            .decode_mem_fixed
            .iter_mut()
            .zip(decoder.decode_mem.iter())
        {
            *dst = super::celt_sig_to_fixed(src);
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

        let tolerance = if cfg!(feature = "fixed_point") {
            2e-3f32
        } else {
            1e-6f32
        };
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
                #[cfg(feature = "fixed_point")]
                assert!(
                    (decoder.decode_mem_fixed[offset + start + i]
                        - super::celt_sig_to_fixed(actual))
                    .abs()
                        <= 1,
                    "channel {channel} fixed cache mismatch at fold {i}: fixed={}, float2sig={}",
                    decoder.decode_mem_fixed[offset + start + i],
                    super::celt_sig_to_fixed(actual),
                );
            }

            for i in overlap / 2..overlap {
                let idx = offset + start + i;
                assert_eq!(decoder.decode_mem[idx], original[i]);
            }
        }
    }

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
        #[cfg(feature = "fixed_point")]
        {
            decoder
                .background_log_e_fixed
                .fill(qconst32(-8.0, DB_SHIFT));
            decoder.old_ebands_fixed.fill(qconst32(-4.0, DB_SHIFT));
        }

        let n = mode.short_mdct_size;
        #[cfg(feature = "deep_plc")]
        let plc = None;
        #[cfg(not(feature = "deep_plc"))]
        let plc = ();
        celt_decode_lost(&mut decoder, n, 0, plc);

        assert!(decoder.skip_plc);
        assert!(!decoder.prefilter_and_fold);
        assert_ne!(decoder.rng, 0x1234_5678);
        assert_eq!(decoder.loss_duration, 1);
    }

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
        let mut seed = vec![0.0f32; decoder.decode_mem.len()];
        fill_plc_ctest_periodic_sig(&mut seed, 96, 14_000, 0, false);
        for (idx, sample) in decoder.decode_mem.iter_mut().enumerate() {
            let value = seed[idx];
            *sample = value;
            #[cfg(feature = "fixed_point")]
            {
                decoder.decode_mem_fixed[idx] = super::celt_sig_to_fixed(value);
            }
        }

        let n = mode.short_mdct_size;
        #[cfg(feature = "deep_plc")]
        let plc = None;
        #[cfg(not(feature = "deep_plc"))]
        let plc = ();
        celt_decode_lost(&mut decoder, n, 0, plc);

        assert!(decoder.prefilter_and_fold);
        assert!(!decoder.skip_plc);
        assert_eq!(decoder.loss_duration, 3);

        let start = DECODE_BUFFER_SIZE - n;
        let produced = &decoder.decode_mem[start..start + n];
        let energy: f32 = produced.iter().map(|v| v.abs()).sum();
        assert!(energy > 0.0);
    }

    #[cfg(feature = "fixed_point")]
    const POSTFILTER_CASE_FRAMES: usize = 6;
    #[cfg(feature = "fixed_point")]
    const POSTFILTER_SAMPLE_RATE: i32 = 48_000;
    #[cfg(feature = "fixed_point")]
    const POSTFILTER_FRAME_SIZE: usize = 960;
    #[cfg(feature = "fixed_point")]
    const POSTFILTER_MAX_PACKET_SIZE: usize = 1276;
    #[cfg(feature = "fixed_point")]
    const POSTFILTER_MAX_PITCH: i32 = 1024;

    #[cfg(feature = "fixed_point")]
    struct DecoderPostfilterCase {
        name: &'static str,
        channels: usize,
        bitrate: i32,
        max_bytes: usize,
        min_nonzero_pitch_frames: usize,
        require_pitch_changes: bool,
        expected_pitch: [i32; POSTFILTER_CASE_FRAMES],
        expected_packet_hash: [u32; POSTFILTER_CASE_FRAMES],
        expected_pcm_hash: [u32; POSTFILTER_CASE_FRAMES],
    }

    #[cfg(feature = "fixed_point")]
    fn fnv1a_update(mut hash: u32, byte: u8) -> u32 {
        hash ^= u32::from(byte);
        hash.wrapping_mul(16_777_619)
    }

    #[cfg(feature = "fixed_point")]
    fn fnv1a_bytes(data: &[u8]) -> u32 {
        data.iter()
            .fold(2_166_136_261, |hash, byte| fnv1a_update(hash, *byte))
    }

    #[cfg(feature = "fixed_point")]
    fn fnv1a_pcm_le(data: &[i16]) -> u32 {
        let mut hash = 2_166_136_261u32;
        for sample in data {
            for byte in sample.to_le_bytes() {
                hash = fnv1a_update(hash, byte);
            }
        }
        hash
    }

    #[cfg(feature = "fixed_point")]
    fn fill_postfilter_case_pcm(
        pcm: &mut [i16],
        frame_size: usize,
        channels: usize,
        frame_idx: usize,
    ) {
        for i in 0..frame_size {
            let s0 = ((i * 7 + frame_idx * 31) % 3000) as i32 - 1500;
            let s1 = ((i * 11 + frame_idx * 19) % 2600) as i32 - 1300;
            let shaped0 = s0 + if (i & 7) == 0 { 900 } else { -300 };
            let shaped1 = s1 + if (i % 5) == 0 { -700 } else { 250 };
            if channels == 1 {
                pcm[i] = (shaped0 * 8) as i16;
            } else {
                pcm[2 * i] = (shaped0 * 7) as i16;
                pcm[2 * i + 1] = (shaped1 * 7) as i16;
            }
        }
    }

    #[cfg(feature = "fixed_point")]
    fn run_decoder_postfilter_case(mode: &OpusCustomMode<'_>, case: &DecoderPostfilterCase) {
        let strict_ctest_vectors = std::env::var_os("RUST_CTEST_STRICT_HASHES").is_some();
        let mut encoder =
            opus_custom_encoder_create(mode, POSTFILTER_SAMPLE_RATE, case.channels, 0)
                .unwrap_or_else(|err| panic!("{}: encoder create failed: {err:?}", case.name));
        let mut decoder = opus_custom_decoder_create(mode, case.channels)
            .unwrap_or_else(|err| panic!("{}: decoder create failed: {err:?}", case.name));

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetBitrate(case.bitrate))
            .unwrap_or_else(|err| panic!("{}: set bitrate failed: {err:?}", case.name));
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetVbr(false))
            .unwrap_or_else(|err| panic!("{}: set vbr failed: {err:?}", case.name));
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetComplexity(10))
            .unwrap_or_else(|err| panic!("{}: set complexity failed: {err:?}", case.name));
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetLsbDepth(16))
            .unwrap_or_else(|err| panic!("{}: set lsb_depth failed: {err:?}", case.name));

        let mut invalid_output = vec![0i16; 123 * case.channels];
        assert!(
            opus_custom_decode(&mut decoder, Some(&[0]), &mut invalid_output, 123).is_err(),
            "{}: decode with invalid frame size should fail",
            case.name
        );
        assert_eq!(
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetComplexity(99))
                .expect_err("invalid decoder complexity must fail"),
            CeltDecoderCtlError::InvalidArgument,
            "{}: invalid decoder complexity should map to InvalidArgument",
            case.name
        );

        let mut nonzero_pitch_frames = 0usize;
        let mut pitch_changes = 0usize;
        let mut packet_hash_changes = 0usize;
        let mut pcm_hash_changes = 0usize;
        let mut prev_pitch: Option<i32> = None;
        let mut prev_packet_hash: Option<u32> = None;
        let mut prev_pcm_hash: Option<u32> = None;

        let mut packet = vec![0u8; POSTFILTER_MAX_PACKET_SIZE];
        let mut pcm = vec![0i16; POSTFILTER_FRAME_SIZE * case.channels];
        let mut decoded = vec![0i16; POSTFILTER_FRAME_SIZE * case.channels];

        for frame_idx in 0..POSTFILTER_CASE_FRAMES {
            fill_postfilter_case_pcm(&mut pcm, POSTFILTER_FRAME_SIZE, case.channels, frame_idx);
            let packet_len = opus_custom_encode(
                &mut encoder,
                &pcm,
                POSTFILTER_FRAME_SIZE,
                &mut packet,
                case.max_bytes,
            )
            .unwrap_or_else(|err| {
                panic!("{} frame {frame_idx}: encode failed: {err:?}", case.name)
            });

            let decoded_len = opus_custom_decode(
                &mut decoder,
                Some(&packet[..packet_len]),
                &mut decoded,
                POSTFILTER_FRAME_SIZE,
            )
            .unwrap_or_else(|err| {
                panic!("{} frame {frame_idx}: decode failed: {err:?}", case.name)
            });
            assert_eq!(
                decoded_len, POSTFILTER_FRAME_SIZE,
                "{} frame {}: decode length mismatch",
                case.name, frame_idx
            );

            let mut pitch = 0;
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::GetPitch(&mut pitch))
                .unwrap_or_else(|err| {
                    panic!("{} frame {frame_idx}: get pitch failed: {err:?}", case.name)
                });
            assert!(
                (0..=POSTFILTER_MAX_PITCH).contains(&pitch),
                "{} frame {}: pitch {} out of range",
                case.name,
                frame_idx,
                pitch
            );

            if pitch > 0 {
                nonzero_pitch_frames += 1;
            }
            if let Some(prev) = prev_pitch {
                if prev != pitch {
                    pitch_changes += 1;
                }
            }
            prev_pitch = Some(pitch);

            let packet_hash = fnv1a_bytes(&packet[..packet_len]);
            let pcm_hash = fnv1a_pcm_le(&decoded[..POSTFILTER_FRAME_SIZE * case.channels]);

            if let Some(prev) = prev_packet_hash {
                if prev != packet_hash {
                    packet_hash_changes += 1;
                }
            }
            prev_packet_hash = Some(packet_hash);
            if let Some(prev) = prev_pcm_hash {
                if prev != pcm_hash {
                    pcm_hash_changes += 1;
                }
            }
            prev_pcm_hash = Some(pcm_hash);

            if strict_ctest_vectors {
                if case.expected_pitch[frame_idx] != 0 {
                    assert_eq!(
                        pitch, case.expected_pitch[frame_idx],
                        "{} frame {}: pitch mismatch",
                        case.name, frame_idx
                    );
                }
                if case.expected_packet_hash[frame_idx] != 0 {
                    assert_eq!(
                        packet_hash, case.expected_packet_hash[frame_idx],
                        "{} frame {}: packet hash mismatch",
                        case.name, frame_idx
                    );
                }
                if case.expected_pcm_hash[frame_idx] != 0 {
                    assert_eq!(
                        pcm_hash, case.expected_pcm_hash[frame_idx],
                        "{} frame {}: pcm hash mismatch",
                        case.name, frame_idx
                    );
                }
            }
        }

        if case.min_nonzero_pitch_frames > 0 {
            assert!(
                nonzero_pitch_frames >= case.min_nonzero_pitch_frames,
                "{}: expected at least {} non-zero pitch frames, got {}",
                case.name,
                case.min_nonzero_pitch_frames,
                nonzero_pitch_frames
            );
        }
        if case.require_pitch_changes {
            assert!(
                pitch_changes > 0,
                "{}: expected at least one pitch change across frames",
                case.name
            );
        }
        assert!(
            packet_hash_changes > 0,
            "{}: expected encoded packet hashes to change across frames",
            case.name
        );
        assert!(
            pcm_hash_changes > 0,
            "{}: expected decoded PCM hashes to change across frames",
            case.name
        );
        assert!(
            opus_custom_decode(
                &mut decoder,
                Some(&[0]),
                &mut decoded,
                POSTFILTER_FRAME_SIZE
            )
            .is_err(),
            "{}: tiny packet decode should fail",
            case.name
        );
    }

    #[cfg(feature = "fixed_point")]
    const DECODER_STATE_WARM_FRAMES: usize = 8;
    #[cfg(feature = "fixed_point")]
    const DECODER_STATE_LOSS_FRAMES: usize = 6;
    #[cfg(feature = "fixed_point")]
    const DECODER_DATAFLOW_FRAMES: usize = 13;
    #[cfg(feature = "fixed_point")]
    const DECODER_DATAFLOW_NB_EBANDS: i32 = 21;

    #[cfg(feature = "fixed_point")]
    fn fill_decoder_state_pcm(pcm: &mut [i16], frame_size: usize, phase: usize) {
        let period = 96usize;
        let half = period / 2;
        let amp = 13_000i32;
        for (i, slot) in pcm.iter_mut().take(frame_size).enumerate() {
            let pos = (i + phase) % period;
            let tri = if pos < half { pos } else { period - pos };
            let centered = (tri as i32 * 2) - half as i32;
            let mut shaped = centered * amp / half as i32;
            shaped += if (i % 9) == 0 { 1200 } else { -300 };
            *slot = shaped as i16;
        }
    }

    #[cfg(feature = "fixed_point")]
    fn fill_decoder_dataflow_pcm(pcm: &mut [i16], frame_size: usize, phase: usize) {
        assert!(
            pcm.len() >= frame_size * 2,
            "stereo dataflow pcm buffer must contain 2*frame_size samples",
        );
        let period = 120usize;
        let half = period / 2;
        let amp_l = 15_000i32;
        let amp_r = 11_200i32;
        for i in 0..frame_size {
            let pos_l = (i + phase) % period;
            let pos_r = (i + phase + 29) % period;
            let tri_l = if pos_l < half { pos_l } else { period - pos_l };
            let tri_r = if pos_r < half { pos_r } else { period - pos_r };
            let centered_l = (tri_l as i32 * 2) - half as i32;
            let centered_r = (tri_r as i32 * 2) - half as i32;
            let mut shaped_l = centered_l * amp_l / half as i32;
            let mut shaped_r = -(centered_r * amp_r / half as i32);
            shaped_l += if (i % 11) == 0 { 900 } else { -350 };
            shaped_r += if (i % 7) == 0 { -700 } else { 220 };
            pcm[2 * i] = shaped_l as i16;
            pcm[2 * i + 1] = shaped_r as i16;
        }
    }

    #[cfg(feature = "fixed_point")]
    fn pcm_energy_i64(samples: &[i16]) -> i64 {
        samples.iter().fold(0i64, |acc, &sample| {
            acc + i64::from(sample) * i64::from(sample)
        })
    }

    #[cfg(feature = "fixed_point")]
    fn count_hash_changes(hashes: &[u32]) -> usize {
        hashes.windows(2).filter(|pair| pair[0] != pair[1]).count()
    }

    #[cfg(feature = "fixed_point")]
    fn count_energy_drops(energies: &[i64]) -> usize {
        energies.windows(2).filter(|pair| pair[1] < pair[0]).count()
    }

    #[cfg(feature = "fixed_point")]
    const DECODER_PLC_IIR_PITCH_LOSS_FRAMES: usize = 2;
    #[cfg(feature = "fixed_point")]
    const DECODER_PLC_IIR_NOISE_START_BAND: i32 = 17;

    #[cfg(feature = "fixed_point")]
    fn fnv1a_sig32_le(data: &[i32]) -> u32 {
        let mut hash = 2_166_136_261u32;
        for sample in data {
            for byte in sample.to_le_bytes() {
                hash = fnv1a_update(hash, byte);
            }
        }
        hash
    }

    #[cfg(feature = "fixed_point")]
    fn decoder_plc_iir_lpc_hash(decoder: &super::OpusCustomDecoder<'_>) -> u32 {
        fnv1a_pcm_le(&decoder.lpc_fixed)
    }

    #[cfg(feature = "fixed_point")]
    fn decoder_plc_iir_tail_hash(decoder: &super::OpusCustomDecoder<'_>, frame_size: usize) -> u32 {
        let stride = DECODE_BUFFER_SIZE + decoder.overlap;
        let tail_len = frame_size + decoder.overlap;
        let mut hash = 2_166_136_261u32;
        for channel in 0..decoder.channels {
            let base = channel * stride + DECODE_BUFFER_SIZE - frame_size;
            let tail = &decoder.decode_mem_fixed[base..base + tail_len];
            hash ^= fnv1a_sig32_le(tail);
            hash = hash.wrapping_mul(16_777_619);
        }
        hash
    }

    #[cfg(feature = "fixed_point")]
    fn parse_hex_packet(hex: &str) -> Vec<u8> {
        assert_eq!(
            hex.len() % 2,
            0,
            "hex packet strings must have an even length"
        );
        let mut packet = Vec::with_capacity(hex.len() / 2);
        for idx in (0..hex.len()).step_by(2) {
            let byte = u8::from_str_radix(&hex[idx..idx + 2], 16)
                .expect("packet hex should contain only hexadecimal digits");
            packet.push(byte);
        }
        packet
    }

    #[cfg(feature = "fixed_point")]
    fn prime_decoder_plc_iir(
        decoder: &mut super::OpusCustomDecoder<'_>,
        decoded: &mut [i16],
        packets: &[&str],
    ) -> i32 {
        let mut pitch = 0;
        for packet_hex in packets.iter() {
            let packet = parse_hex_packet(packet_hex);
            let decoded_len =
                opus_custom_decode(decoder, Some(&packet), decoded, POSTFILTER_FRAME_SIZE)
                    .expect("PLC/IIR priming decode should succeed");
            assert_eq!(decoded_len, POSTFILTER_FRAME_SIZE);
            opus_custom_decoder_ctl(decoder, DecoderCtlRequest::GetPitch(&mut pitch))
                .expect("get pitch should succeed");
            let mut final_range = 0u32;
            opus_custom_decoder_ctl(decoder, DecoderCtlRequest::GetFinalRange(&mut final_range))
                .expect("get final range should succeed");
            crate::test_trace::trace_println!(
                "prime decoded={} pitch={} range=0x{:08x} pcm=0x{:08x} tail=0x{:08x} lpc=0x{:08x} eb=0x{:08x}",
                decoded_len,
                pitch,
                final_range,
                fnv1a_pcm_le(decoded),
                decoder_plc_iir_tail_hash(decoder, POSTFILTER_FRAME_SIZE),
                decoder_plc_iir_lpc_hash(decoder),
                fnv1a_sig32_le(&decoder.old_ebands_fixed)
            );
            if pitch > 0 {
                return pitch;
            }
        }
        panic!(
            "failed to prime decoder pitch after {} frames",
            packets.len()
        );
    }

    #[cfg(feature = "fixed_point")]
    fn decode_packet_band_spectrum(
        decoder: &mut super::OpusCustomDecoder<'_>,
        packet: &[u8],
        use_fixed_native: bool,
    ) -> (Vec<i16>, Vec<u8>) {
        let mode = decoder.mode;
        let nb_ebands = mode.num_ebands;
        let frame = prepare_frame(decoder, packet, POSTFILTER_FRAME_SIZE, None)
            .expect("frame preparation should succeed");
        let super::FramePreparation {
            mut range_decoder,
            tf_res,
            fine_quant: _,
            pulses,
            spread_decision,
            short_blocks,
            intra_ener: _,
            anti_collapse_rsv,
            intensity,
            dual_stereo,
            balance,
            coded_bands,
            total_bits,
            start,
            end,
            lm,
            n,
            c,
            ..
        } = frame;
        let mut range_state = range_decoder.take().expect("range decoder must exist");
        let dec = range_state.decoder();

        let mut collapse_masks = vec![0u8; c * nb_ebands];
        let total_available = total_bits - anti_collapse_rsv;
        let mut spectrum_fixed = vec![0i16; c * n];
        let mut coder = BandCodingState::Decoder(dec);

        if use_fixed_native {
            let (first_channel, second_channel_opt) = if c == 2 {
                let (left, right) = spectrum_fixed.split_at_mut(n);
                (left, Some(right))
            } else {
                (&mut spectrum_fixed[..], None)
            };

            crate::celt::bands::quant_all_bands_decode_fixed(
                mode,
                start,
                end,
                first_channel,
                second_channel_opt,
                &mut collapse_masks,
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
                decoder.arch,
                decoder.disable_inv,
            );
        } else {
            let mut spectrum_float = vec![0.0f32; c * n];
            let (first_channel, second_channel_opt) = if c == 2 {
                let (left, right) = spectrum_float.split_at_mut(n);
                (left, Some(right))
            } else {
                (&mut spectrum_float[..], None)
            };

            crate::celt::bands::quant_all_bands(
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
            super::float_norm_slice_to_fixed(&mut spectrum_fixed, &spectrum_float);
        }

        (spectrum_fixed, collapse_masks)
    }

    #[cfg(feature = "fixed_point")]
    fn debug_first_band_pulses(
        decoder: &mut super::OpusCustomDecoder<'_>,
        packet: &[u8],
    ) -> (usize, i32, u32, u32, Vec<i32>, u32) {
        let mode = decoder.mode;
        let frame = prepare_frame(decoder, packet, POSTFILTER_FRAME_SIZE, None)
            .expect("frame preparation should succeed");
        let super::FramePreparation {
            mut range_decoder,
            pulses,
            total_bits,
            anti_collapse_rsv,
            balance,
            coded_bands,
            start,
            lm,
            ..
        } = frame;
        let mut range_state = range_decoder.take().expect("range decoder must exist");
        let dec = range_state.decoder();

        let band = start;
        let total_available = total_bits - anti_collapse_rsv;
        let tell = entcode::ec_tell_frac(dec.ctx()) as i32;
        let remaining_bits = total_available - tell - 1;
        let remaining_coded = (coded_bands.max(0) as usize).saturating_sub(band).min(3) as i32;
        let curr_balance = if remaining_coded > 0 {
            celt_sudiv(balance, remaining_coded)
        } else {
            0
        };
        let b = (remaining_bits + 1)
            .min(pulses[band] + curr_balance)
            .clamp(0, 16_383);
        let q = bits2pulses(mode, band, lm as i32, b);
        let k = get_pulses(q) as usize;
        let n = ((mode.e_bands[band + 1] - mode.e_bands[band]) as usize) << (lm as usize);
        let mut pulse_vec = vec![0i32; n];
        let (index, total, _) = decode_pulses_debug(&mut pulse_vec, n, k, dec);
        let collapse_mask =
            crate::celt::vq::extract_collapse_mask(&pulse_vec, n, 1usize << (lm as usize));
        (n, q, index, total, pulse_vec, collapse_mask)
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn decoder_plc_iir_matches_ctest_vectors() {
        let owned_mode = opus_custom_mode_create(48_000, POSTFILTER_FRAME_SIZE)
            .expect("mode creation should succeed");
        let mode = owned_mode.mode();
        let mono_prime_packets = [
            "f87b5ade71db5cb86cc9a42d87bbde1e71f2afd45d5a88bb16581b72a6ca16de9eae4b",
            "f8afe90df1a3c02ade92c45dbb015dd97f46a387235b2befcd07fe0d3f70757d6ab161",
        ];
        let stereo_prime_packets = [
            "fc7f06c5823a668788d0fc75ad6cc856faca7625c7fc39ac0bb9a60782cca363b2ac94cc2ead16432c2db2b60c4a1bf5d8b0db4cfb5dcecea0c7e319dc388d203aab9cada1d8fca1735f2273de589eb62975a41340011c8c8752c5848f073f0c8b4b7df18f482a9b96b4f48b988711a3c51703fcd97b59cb338539cdbc007ede6ea8ff6089adae678bb6f91ee118b045bd6b41c255a773f962f87ed3bcf22326",
            "fcb4050bfdbad3a7e0e30e26a55e15a50a352c8324108e1b086d04ea69125c1bd5c099b59cae0bf512802803d8bedc99b2ea3ae09ddcadb681924fedb949c16f5d73330cdf509498645b0dbb97b7d8ff3d51ee030b70f88d581a993441324991f4a1e31f8137687fc9b8a9c56298d8657a9437028a34c2a6c8d1a276f73df195974263dad177f3300c027683b7115ac0a91d33b4a3bdb51c79b0041d720d07a1",
        ];

        let mono_expected_lpc = [0x5159_7095u32, 0x5159_7095u32];
        let mono_expected_tail = [0xdfaa_14e9u32, 0xefd8_0164u32];
        let mono_expected_pcm = [0xc1b6_b08du32, 0xc2a4_65edu32];
        let mono_expected_noise_lpc = 0xc655_ff85u32;
        let mono_expected_noise_tail = 0x05e4_dd21u32;

        let stereo_expected_lpc = [0xd31a_6449u32, 0xd31a_6449u32];
        let stereo_expected_tail = [0x7bf7_ff22u32, 0xa5d2_03d2u32];
        let stereo_expected_pcm = [0x03cb_2a69u32, 0x16d3_9455u32];

        {
            let first_packet = parse_hex_packet(mono_prime_packets[0]);
            let mut decoder_tf =
                opus_custom_decoder_create(&mode, 1).expect("tf decoder creation should succeed");
            let mut decoder_native = opus_custom_decoder_create(&mode, 1)
                .expect("native decoder creation should succeed");
            let mut decoder_bridge = opus_custom_decoder_create(&mode, 1)
                .expect("bridge decoder creation should succeed");
            let tf_frame =
                prepare_frame(&mut decoder_tf, &first_packet, POSTFILTER_FRAME_SIZE, None)
                    .expect("frame preparation should succeed");
            crate::test_trace::trace_println!(
                "frame0 tf_res_first4={:?}",
                &tf_frame.tf_res[..4.min(tf_frame.tf_res.len())]
            );
            let (band0_n, band0_q, band0_index, band0_total, band0_pulses, band0_mask) =
                debug_first_band_pulses(&mut decoder_native, &first_packet);
            crate::test_trace::trace_println!(
                "band0 pulses n={} q={} index={} total={} mask=0x{:02x} pulses={:?}",
                band0_n,
                band0_q,
                band0_index,
                band0_total,
                band0_mask,
                band0_pulses
            );
            let (native_spectrum, native_masks) =
                decode_packet_band_spectrum(&mut decoder_native, &first_packet, true);
            let (bridge_spectrum, bridge_masks) =
                decode_packet_band_spectrum(&mut decoder_bridge, &first_packet, false);
            let first_diff = native_spectrum
                .iter()
                .zip(bridge_spectrum.iter())
                .position(|(lhs, rhs)| lhs != rhs);
            crate::test_trace::trace_println!(
                "bandcmp frame0 native=0x{:08x} bridge=0x{:08x} masks_native=0x{:08x} masks_bridge=0x{:08x} first_diff={:?}",
                fnv1a_pcm_le(&native_spectrum),
                fnv1a_pcm_le(&bridge_spectrum),
                fnv1a_bytes(&native_masks),
                fnv1a_bytes(&bridge_masks),
                first_diff
            );
            crate::test_trace::trace_println!(
                "bandcmp frame0 masks_native_full={:?}",
                native_masks
            );
            let lm_dbg = (POSTFILTER_FRAME_SIZE / mode.short_mdct_size).ilog2() as usize;
            for band_hash in 0..mode.num_ebands {
                let band_start = (mode.e_bands[band_hash] as usize) << lm_dbg;
                let band_end = (mode.e_bands[band_hash + 1] as usize) << lm_dbg;
                let native_hash = fnv1a_pcm_le(&native_spectrum[band_start..band_end]);
                crate::test_trace::trace_println!(
                    "bandcmp frame0 native_band_hash[{}]=0x{:08x}",
                    band_hash,
                    native_hash
                );
            }
            if let Some(idx) = first_diff {
                let lm_dbg = (POSTFILTER_FRAME_SIZE / mode.short_mdct_size).ilog2() as usize;
                let mut band_idx = 0usize;
                while band_idx + 1 < mode.num_ebands
                    && ((mode.e_bands[band_idx + 1] as usize) << lm_dbg) <= idx
                {
                    band_idx += 1;
                }
                let band_start = (mode.e_bands[band_idx] as usize) << lm_dbg;
                crate::test_trace::trace_println!(
                    "bandcmp frame0 coeff[{idx}] band={} band_off={} native={} bridge={}",
                    band_idx,
                    idx - band_start,
                    native_spectrum[idx],
                    bridge_spectrum[idx]
                );
            }
        }

        {
            let first_packet = parse_hex_packet(stereo_prime_packets[0]);
            let mut decoder_native = opus_custom_decoder_create(&mode, 2)
                .expect("stereo native decoder creation should succeed");
            let mut decoder_bridge = opus_custom_decoder_create(&mode, 2)
                .expect("stereo bridge decoder creation should succeed");
            let (native_spectrum, native_masks) =
                decode_packet_band_spectrum(&mut decoder_native, &first_packet, true);
            let (bridge_spectrum, bridge_masks) =
                decode_packet_band_spectrum(&mut decoder_bridge, &first_packet, false);
            let first_diff = native_spectrum
                .iter()
                .zip(bridge_spectrum.iter())
                .position(|(lhs, rhs)| lhs != rhs);
            crate::test_trace::trace_println!(
                "stereo bandcmp frame0 native=0x{:08x} bridge=0x{:08x} masks_native=0x{:08x} masks_bridge=0x{:08x} first_diff={:?}",
                fnv1a_pcm_le(&native_spectrum),
                fnv1a_pcm_le(&bridge_spectrum),
                fnv1a_bytes(&native_masks),
                fnv1a_bytes(&bridge_masks),
                first_diff
            );
            crate::test_trace::trace_println!(
                "stereo bandcmp frame0 first8 native={:?} bridge={:?}",
                &native_spectrum[..8],
                &bridge_spectrum[..8]
            );
        }

        {
            let mut decoder =
                opus_custom_decoder_create(&mode, 1).expect("mono decoder creation should succeed");

            assert_eq!(
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetComplexity(11))
                    .expect_err("invalid decoder complexity must fail"),
                CeltDecoderCtlError::InvalidArgument
            );

            let mut decoded = vec![0i16; POSTFILTER_FRAME_SIZE];
            let primed_pitch =
                prime_decoder_plc_iir(&mut decoder, &mut decoded, &mono_prime_packets);
            assert!(primed_pitch > 0);
            crate::test_trace::trace_println!("mono primed_pitch={primed_pitch}");
            {
                let stride = DECODE_BUFFER_SIZE + decoder.overlap;
                let float_views: Vec<&[CeltSig]> = decoder
                    .decode_mem
                    .chunks(stride)
                    .take(decoder.channels)
                    .collect();
                let fixed_views: Vec<&[FixedCeltSig]> = decoder
                    .decode_mem_fixed
                    .chunks(stride)
                    .take(decoder.channels)
                    .collect();
                let float_pitch =
                    celt_plc_pitch_search(&float_views, decoder.channels, decoder.arch);
                let fixed_pitch =
                    celt_plc_pitch_search_fixed(&fixed_views, decoder.channels, decoder.arch);
                crate::test_trace::trace_println!(
                    "mono search float={float_pitch} fixed={fixed_pitch}"
                );
            }

            let baseline_lpc = decoder_plc_iir_lpc_hash(&decoder);
            let baseline_tail = decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE);
            crate::test_trace::trace_println!(
                "mono baseline_lpc=0x{baseline_lpc:08x} tail=0x{baseline_tail:08x}"
            );
            let tiny_packet = [0u8];

            assert!(
                opus_custom_decode(
                    &mut decoder,
                    Some(&tiny_packet),
                    &mut decoded,
                    POSTFILTER_FRAME_SIZE,
                )
                .is_err(),
                "tiny mono packet should fail",
            );
            assert_eq!(decoder_plc_iir_lpc_hash(&decoder), baseline_lpc);
            assert_eq!(
                decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE),
                baseline_tail
            );

            assert!(
                opus_custom_decode(&mut decoder, None, &mut decoded, POSTFILTER_FRAME_SIZE - 1)
                    .is_err(),
                "invalid mono frame size should fail",
            );
            assert_eq!(decoder_plc_iir_lpc_hash(&decoder), baseline_lpc);
            assert_eq!(
                decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE),
                baseline_tail
            );

            let mut lpc_hashes = [0u32; DECODER_PLC_IIR_PITCH_LOSS_FRAMES];
            let mut tail_hashes = [0u32; DECODER_PLC_IIR_PITCH_LOSS_FRAMES];
            let mut pcm_hashes = [0u32; DECODER_PLC_IIR_PITCH_LOSS_FRAMES];
            for loss in 0..DECODER_PLC_IIR_PITCH_LOSS_FRAMES {
                let start_preemph_mem = decoder.fixed_preemph_mem_decoder[0];
                let decoded_len =
                    opus_custom_decode(&mut decoder, None, &mut decoded, POSTFILTER_FRAME_SIZE)
                        .expect("mono PLC loss decode should succeed");
                assert_eq!(decoded_len, POSTFILTER_FRAME_SIZE);
                lpc_hashes[loss] = decoder_plc_iir_lpc_hash(&decoder);
                tail_hashes[loss] = decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE);
                pcm_hashes[loss] = fnv1a_pcm_le(&decoded);
                crate::test_trace::trace_println!(
                    "mono loss{loss} pitch={} lpc=0x{:08x} tail=0x{:08x} pcm=0x{:08x}",
                    decoder.last_pitch_index,
                    lpc_hashes[loss],
                    tail_hashes[loss],
                    pcm_hashes[loss]
                );
                crate::test_trace::trace_println!(
                    "mono loss{loss} pcm_first16={:?}",
                    &decoded[..16]
                );
                crate::test_trace::trace_println!(
                    "mono loss{loss} pcm_last16={:?}",
                    &decoded[POSTFILTER_FRAME_SIZE - 16..]
                );
                crate::test_trace::trace_println!(
                    "mono loss{loss} pcm_hash_halves=0x{:08x}/0x{:08x}",
                    fnv1a_pcm_le(&decoded[..POSTFILTER_FRAME_SIZE / 2]),
                    fnv1a_pcm_le(&decoded[POSTFILTER_FRAME_SIZE / 2..]),
                );
                for block in 0..(POSTFILTER_FRAME_SIZE / 64) {
                    let start = block * 64;
                    crate::test_trace::trace_println!(
                        "mono loss{loss} block{block:02}=0x{:08x}",
                        fnv1a_pcm_le(&decoded[start..start + 64])
                    );
                }
                if loss == 0 {
                    crate::test_trace::trace_println!(
                        "mono loss0 block03_samples={:?}",
                        &decoded[192..256]
                    );
                    crate::test_trace::trace_println!(
                        "mono loss0 block10_samples={:?}",
                        &decoded[640..704]
                    );
                } else {
                    crate::test_trace::trace_println!(
                        "mono loss1 block13_samples={:?}",
                        &decoded[832..896]
                    );
                }
                if loss == 0 {
                    crate::test_trace::trace_println!(
                        "mono loss0 lpc_coeffs={:?}",
                        &decoder.lpc_fixed[..LPC_ORDER]
                    );
                }
                let stride = DECODE_BUFFER_SIZE + decoder.overlap;
                let start_idx = DECODE_BUFFER_SIZE - POSTFILTER_FRAME_SIZE;
                let signal =
                    &decoder.decode_mem_fixed[start_idx..start_idx + POSTFILTER_FRAME_SIZE];
                let coef0 = qconst16_clamped(f64::from(mode.pre_emphasis[0]), 15);
                let trace_points: &[usize] = if loss == 0 { &[208, 680] } else { &[890] };
                let mut mem_sig = start_preemph_mem;
                for (j, &sample) in signal.iter().enumerate() {
                    let tmp = (sample + mem_sig).clamp(-FIXED_SIG_SAT, FIXED_SIG_SAT);
                    let out = sig2word16(tmp);
                    let next_mem = mult16_32_q15(coef0, tmp);
                    if trace_points.contains(&j) {
                        crate::test_trace::trace_println!(
                            "mono loss{loss} trace[{j}] sample={} mem_before={} tmp={} out={} mem_after={}",
                            sample,
                            mem_sig,
                            tmp,
                            out,
                            next_mem
                        );
                    }
                    mem_sig = next_mem;
                }
                let _ = stride;
                assert!(decoded.iter().any(|&sample| sample != 0));
            }

            assert_eq!(lpc_hashes, mono_expected_lpc);
            assert_eq!(tail_hashes, mono_expected_tail);
            assert_eq!(pcm_hashes, mono_expected_pcm);

            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::ResetState)
                .expect("mono reset should succeed");
            opus_custom_decoder_ctl(
                &mut decoder,
                DecoderCtlRequest::SetStartBand(DECODER_PLC_IIR_NOISE_START_BAND),
            )
            .expect("mono set noise start band should succeed");

            let decoded_len =
                opus_custom_decode(&mut decoder, None, &mut decoded, POSTFILTER_FRAME_SIZE)
                    .expect("mono noise-path PLC should succeed");
            assert_eq!(decoded_len, POSTFILTER_FRAME_SIZE);
            assert!(decoded.iter().any(|&sample| sample != 0));
            assert_eq!(decoder_plc_iir_lpc_hash(&decoder), mono_expected_noise_lpc);
            assert_eq!(
                decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE),
                mono_expected_noise_tail
            );
        }

        {
            let mut decoder = opus_custom_decoder_create(&mode, 2)
                .expect("stereo decoder creation should succeed");

            let mut decoded = vec![0i16; POSTFILTER_FRAME_SIZE * 2];

            assert!(
                opus_custom_decode(
                    &mut decoder,
                    Some(&[0u8]),
                    &mut decoded,
                    POSTFILTER_FRAME_SIZE - 1,
                )
                .is_err(),
                "invalid stereo frame size should fail",
            );

            let primed_pitch =
                prime_decoder_plc_iir(&mut decoder, &mut decoded, &stereo_prime_packets);
            assert!(primed_pitch > 0);

            let baseline_lpc = decoder_plc_iir_lpc_hash(&decoder);
            let baseline_tail = decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE);
            let tiny_packet = [0u8];
            assert!(
                opus_custom_decode(
                    &mut decoder,
                    Some(&tiny_packet),
                    &mut decoded,
                    POSTFILTER_FRAME_SIZE,
                )
                .is_err(),
                "tiny stereo packet should fail",
            );
            assert_eq!(decoder_plc_iir_lpc_hash(&decoder), baseline_lpc);
            assert_eq!(
                decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE),
                baseline_tail
            );

            let mut lpc_hashes = [0u32; DECODER_PLC_IIR_PITCH_LOSS_FRAMES];
            let mut tail_hashes = [0u32; DECODER_PLC_IIR_PITCH_LOSS_FRAMES];
            let mut pcm_hashes = [0u32; DECODER_PLC_IIR_PITCH_LOSS_FRAMES];
            for loss in 0..DECODER_PLC_IIR_PITCH_LOSS_FRAMES {
                let decoded_len =
                    opus_custom_decode(&mut decoder, None, &mut decoded, POSTFILTER_FRAME_SIZE)
                        .expect("stereo PLC loss decode should succeed");
                assert_eq!(decoded_len, POSTFILTER_FRAME_SIZE);
                lpc_hashes[loss] = decoder_plc_iir_lpc_hash(&decoder);
                tail_hashes[loss] = decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE);
                pcm_hashes[loss] = fnv1a_pcm_le(&decoded);
                crate::test_trace::trace_println!(
                    "stereo loss{loss} pitch={} lpc=0x{:08x} tail=0x{:08x} pcm=0x{:08x}",
                    decoder.last_pitch_index,
                    lpc_hashes[loss],
                    tail_hashes[loss],
                    pcm_hashes[loss]
                );
                assert!(decoded.iter().any(|&sample| sample != 0));
            }

            assert_eq!(lpc_hashes, stereo_expected_lpc);
            assert_eq!(tail_hashes, stereo_expected_tail);
            assert_eq!(pcm_hashes, stereo_expected_pcm);
        }
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn debug_single_packet_actual_decode_trace() {
        let owned_mode = opus_custom_mode_create(48_000, POSTFILTER_FRAME_SIZE)
            .expect("mode creation should succeed");
        let mode = owned_mode.mode();
        let packet = parse_hex_packet(
            "f87b5ade71db5cb86cc9a42d87bbde1e71f2afd45d5a88bb16581b72a6ca16de9eae4b",
        );
        let mut decoder =
            opus_custom_decoder_create(&mode, 1).expect("decoder creation should succeed");
        if std::env::var("CELT_TRACE_MDCT_TABLES").is_ok() {
            let twiddles = decoder.fixed_mdct.twiddles(3);
            let mut ht = 2166136261u32;
            for &value in twiddles.iter() {
                let v = value as u16;
                ht = (ht ^ u32::from(v & 0xFF)).wrapping_mul(16777619);
                ht = (ht ^ u32::from(v >> 8)).wrapping_mul(16777619);
            }
            let bitrev = decoder.fixed_mdct.inverse_plan(3).bitrev();
            let mut hb = 2166136261u32;
            for &value in bitrev.iter() {
                let v = value as u16;
                hb = (hb ^ u32::from(v & 0xFF)).wrapping_mul(16777619);
                hb = (hb ^ u32::from(v >> 8)).wrapping_mul(16777619);
            }
            crate::test_trace::trace_println!(
                "mdct_tables twiddle_hash=0x{:08x} first8={:?} bitrev_hash=0x{:08x} bitrev_first=[{}, {}, {}]",
                ht,
                &twiddles[..8],
                hb,
                bitrev[0],
                bitrev[1],
                bitrev[29],
            );
        }
        let mut pcm = vec![0i16; POSTFILTER_FRAME_SIZE];
        let decoded =
            opus_custom_decode(&mut decoder, Some(&packet), &mut pcm, POSTFILTER_FRAME_SIZE)
                .expect("decode should succeed");
        crate::test_trace::trace_println!(
            "single_packet decoded={} range=0x{:08x} pcm=0x{:08x} tail=0x{:08x}",
            decoded,
            decoder.rng,
            fnv1a_pcm_le(&pcm[..decoded]),
            fnv1a_sig32_le(&decoder.decode_mem_fixed[DECODE_BUFFER_SIZE - 64..DECODE_BUFFER_SIZE])
        );
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn debug_single_stereo_packet_actual_decode_trace() {
        let owned_mode = opus_custom_mode_create(48_000, POSTFILTER_FRAME_SIZE)
            .expect("mode creation should succeed");
        let mode = owned_mode.mode();
        let packet = parse_hex_packet(
            "fc7f06c5823a668788d0fc75ad6cc856faca7625c7fc39ac0bb9a60782cca363b2ac94cc2ead16432c2db2b60c4a1bf5d8b0db4cfb5dcecea0c7e319dc388d203aab9cada1d8fca1735f2273de589eb62975a41340011c8c8752c5848f073f0c8b4b7df18f482a9b96b4f48b988711a3c51703fcd97b59cb338539cdbc007ede6ea8ff6089adae678bb6f91ee118b045bd6b41c255a773f962f87ed3bcf22326",
        );
        let mut spectrum_decoder =
            opus_custom_decoder_create(&mode, 2).expect("decoder creation should succeed");
        let (spectrum, masks) = decode_packet_band_spectrum(&mut spectrum_decoder, &packet, true);
        crate::test_trace::trace_println!(
            "single_stereo_packet spectrum=0x{:08x} masks=0x{:08x}",
            fnv1a_pcm_le(&spectrum),
            fnv1a_bytes(&masks)
        );
        crate::test_trace::trace_println!("single_stereo_packet masks_full={:?}", masks);
        let lm_dbg = (POSTFILTER_FRAME_SIZE / mode.short_mdct_size).ilog2() as usize;
        let channel_stride = spectrum.len() / 2;
        for band in 0..mode.num_ebands {
            let band_start = (mode.e_bands[band] as usize) << lm_dbg;
            let band_end = (mode.e_bands[band + 1] as usize) << lm_dbg;
            crate::test_trace::trace_println!(
                "single_stereo_packet band_hash_l[{band}]=0x{:08x}",
                fnv1a_pcm_le(&spectrum[band_start..band_end])
            );
            crate::test_trace::trace_println!(
                "single_stereo_packet band_hash_r[{band}]=0x{:08x}",
                fnv1a_pcm_le(&spectrum[channel_stride + band_start..channel_stride + band_end])
            );
        }
        let band17_start = (mode.e_bands[17] as usize) << lm_dbg;
        crate::test_trace::trace_println!(
            "single_stereo_packet band17_left={:?}",
            &spectrum[band17_start..band17_start + 8]
        );
        crate::test_trace::trace_println!(
            "single_stereo_packet band17_right={:?}",
            &spectrum[channel_stride + band17_start..channel_stride + band17_start + 8]
        );
        let mut decoder =
            opus_custom_decoder_create(&mode, 2).expect("decoder creation should succeed");
        let mut pcm = vec![0i16; POSTFILTER_FRAME_SIZE * 2];
        let decoded =
            opus_custom_decode(&mut decoder, Some(&packet), &mut pcm, POSTFILTER_FRAME_SIZE)
                .expect("decode should succeed");
        crate::test_trace::trace_println!(
            "single_stereo_packet decoded={} range=0x{:08x} pcm=0x{:08x} tail=0x{:08x}",
            decoded,
            decoder.rng,
            fnv1a_pcm_le(&pcm[..decoded * 2]),
            decoder_plc_iir_tail_hash(&decoder, POSTFILTER_FRAME_SIZE)
        );
        crate::test_trace::trace_println!("single_stereo_packet first16={:?}", &pcm[..32]);
        crate::test_trace::trace_println!("single_stereo_packet first64={:?}", &pcm[..128]);
        crate::test_trace::trace_println!(
            "single_stereo_packet full_pcm={:?}",
            &pcm[..decoded * 2]
        );
        for block in 0..(POSTFILTER_FRAME_SIZE / 64) {
            let start = block * 64 * 2;
            crate::test_trace::trace_println!(
                "single_stereo_packet block{block:02}=0x{:08x}",
                fnv1a_pcm_le(&pcm[start..start + 128])
            );
        }
        let stride = DECODE_BUFFER_SIZE + decoder.overlap;
        let left_tail = &decoder.decode_mem_fixed[DECODE_BUFFER_SIZE - POSTFILTER_FRAME_SIZE
            ..DECODE_BUFFER_SIZE - POSTFILTER_FRAME_SIZE + POSTFILTER_FRAME_SIZE + decoder.overlap];
        let right_base = stride;
        let right_tail = &decoder.decode_mem_fixed[right_base + DECODE_BUFFER_SIZE
            - POSTFILTER_FRAME_SIZE
            ..right_base + DECODE_BUFFER_SIZE - POSTFILTER_FRAME_SIZE
                + POSTFILTER_FRAME_SIZE
                + decoder.overlap];
        crate::test_trace::trace_println!(
            "single_stereo_packet tail0=0x{:08x} tail1=0x{:08x}",
            fnv1a_sig32_le(left_tail),
            fnv1a_sig32_le(right_tail)
        );
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn decoder_state_transitions_match_ctest_vectors() {
        let strict_ctest_vectors = std::env::var_os("RUST_CTEST_STRICT_HASHES").is_some();
        let expected_warm_packet_hashes: [u32; DECODER_STATE_WARM_FRAMES] = [
            0x1204_0bb9,
            0xfa16_704e,
            0xc40f_7772,
            0x7c72_6ef3,
            0x3ec0_1bf0,
            0x108e_4248,
            0x0341_cfd7,
            0x8419_87b6,
        ];
        let expected_warm_pcm_hashes: [u32; DECODER_STATE_WARM_FRAMES] = [
            0x7025_93e3,
            0xe891_961f,
            0xb876_40ee,
            0xd9ab_f25c,
            0x0d59_b40f,
            0xa453_23c0,
            0x387e_b7e3,
            0xde85_f2a0,
        ];
        let expected_loss_hashes: [u32; DECODER_STATE_LOSS_FRAMES] = [
            0x88e9_15a8,
            0xca42_f30e,
            0x657c_aa87,
            0x746d_415b,
            0x2e3e_688f,
            0xdd03_0e3b,
        ];
        let expected_recover_hash_after_loss = 0x948f_f204u32;
        let expected_recover_hash_without_loss = 0x7456_1161u32;
        let expected_reset_loss_hash = 0x9811_584du32;

        let owned_mode =
            opus_custom_mode_create(48_000, 960).expect("mode creation should succeed");
        let mode = owned_mode.mode();
        let mut encoder = opus_custom_encoder_create(&mode, 48_000, 1, 0)
            .expect("encoder creation should succeed");
        let mut decoder_a =
            opus_custom_decoder_create(&mode, 1).expect("decoder A creation should succeed");
        let mut decoder_b =
            opus_custom_decoder_create(&mode, 1).expect("decoder B creation should succeed");

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetBitrate(18_000))
            .expect("set bitrate should succeed");
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetVbr(false))
            .expect("set vbr should succeed");
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetComplexity(10))
            .expect("set complexity should succeed");
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetLsbDepth(16))
            .expect("set lsb depth should succeed");
        opus_custom_decoder_ctl(&mut decoder_a, DecoderCtlRequest::SetComplexity(10))
            .expect("decoder A complexity should succeed");
        opus_custom_decoder_ctl(&mut decoder_b, DecoderCtlRequest::SetComplexity(10))
            .expect("decoder B complexity should succeed");

        let mut packet = vec![0u8; POSTFILTER_MAX_PACKET_SIZE];
        let mut pcm = vec![0i16; POSTFILTER_FRAME_SIZE];
        let mut out_a = vec![0i16; POSTFILTER_FRAME_SIZE];
        let mut out_b = vec![0i16; POSTFILTER_FRAME_SIZE];

        assert_eq!(
            opus_custom_decoder_ctl(&mut decoder_a, DecoderCtlRequest::SetComplexity(11))
                .expect_err("invalid complexity must fail"),
            CeltDecoderCtlError::InvalidArgument
        );
        assert!(
            opus_custom_decode(&mut decoder_a, None, &mut out_a, POSTFILTER_FRAME_SIZE - 1)
                .is_err(),
            "decode with invalid frame size should fail",
        );
        assert!(
            opus_custom_decode(
                &mut decoder_a,
                Some(&[0]),
                &mut out_a,
                POSTFILTER_FRAME_SIZE
            )
            .is_err(),
            "tiny packet decode should fail",
        );
        #[cfg(feature = "deep_plc")]
        let plc = None;
        #[cfg(not(feature = "deep_plc"))]
        let plc = ();
        let mut tiny_pcm: [f32; 0] = [];
        assert!(
            super::celt_decode_with_ec_dred(
                &mut decoder_a,
                None,
                &mut tiny_pcm,
                POSTFILTER_FRAME_SIZE,
                None,
                false,
                plc
            )
            .is_err(),
            "decode with empty PCM output should fail",
        );

        let mut warm_packet_hashes = [0u32; DECODER_STATE_WARM_FRAMES];
        let mut warm_pcm_hashes = [0u32; DECODER_STATE_WARM_FRAMES];
        for frame in 0..DECODER_STATE_WARM_FRAMES {
            fill_decoder_state_pcm(&mut pcm, POSTFILTER_FRAME_SIZE, frame * 7);
            let packet_len = opus_custom_encode(
                &mut encoder,
                &pcm,
                POSTFILTER_FRAME_SIZE,
                &mut packet,
                POSTFILTER_MAX_PACKET_SIZE,
            )
            .expect("warmup encode should succeed");
            let decoded_a = opus_custom_decode(
                &mut decoder_a,
                Some(&packet[..packet_len]),
                &mut out_a,
                POSTFILTER_FRAME_SIZE,
            )
            .expect("decoder A warmup decode should succeed");
            let decoded_b = opus_custom_decode(
                &mut decoder_b,
                Some(&packet[..packet_len]),
                &mut out_b,
                POSTFILTER_FRAME_SIZE,
            )
            .expect("decoder B warmup decode should succeed");
            assert_eq!(decoded_a, POSTFILTER_FRAME_SIZE);
            assert_eq!(decoded_b, POSTFILTER_FRAME_SIZE);

            warm_packet_hashes[frame] = fnv1a_bytes(&packet[..packet_len]);
            warm_pcm_hashes[frame] = fnv1a_pcm_le(&out_a[..POSTFILTER_FRAME_SIZE]);
            let warm_pcm_hash_b = fnv1a_pcm_le(&out_b[..POSTFILTER_FRAME_SIZE]);
            assert_eq!(warm_pcm_hashes[frame], warm_pcm_hash_b);
        }
        assert!(count_hash_changes(&warm_packet_hashes) > 0);
        assert!(count_hash_changes(&warm_pcm_hashes) > 0);
        if strict_ctest_vectors {
            assert_eq!(warm_packet_hashes, expected_warm_packet_hashes);
            assert_eq!(warm_pcm_hashes, expected_warm_pcm_hashes);
        }

        let mut loss_hashes = [0u32; DECODER_STATE_LOSS_FRAMES];
        let mut loss_energies = [0i64; DECODER_STATE_LOSS_FRAMES];
        for i in 0..DECODER_STATE_LOSS_FRAMES {
            let decoded =
                opus_custom_decode(&mut decoder_a, None, &mut out_a, POSTFILTER_FRAME_SIZE)
                    .expect("loss decode should succeed");
            assert_eq!(decoded, POSTFILTER_FRAME_SIZE);
            loss_hashes[i] = fnv1a_pcm_le(&out_a[..POSTFILTER_FRAME_SIZE]);
            loss_energies[i] = pcm_energy_i64(&out_a[..POSTFILTER_FRAME_SIZE]);
        }
        assert!(
            decoder_a.decode_mem.iter().any(|&sample| sample != 0.0),
            "decoder history should carry non-zero PLC state"
        );
        if strict_ctest_vectors {
            assert_eq!(loss_hashes, expected_loss_hashes);
        }

        fill_decoder_state_pcm(&mut pcm, POSTFILTER_FRAME_SIZE, 93);
        let recover_packet_len = opus_custom_encode(
            &mut encoder,
            &pcm,
            POSTFILTER_FRAME_SIZE,
            &mut packet,
            POSTFILTER_MAX_PACKET_SIZE,
        )
        .expect("recovery encode should succeed");
        opus_custom_decode(
            &mut decoder_a,
            Some(&packet[..recover_packet_len]),
            &mut out_a,
            POSTFILTER_FRAME_SIZE,
        )
        .expect("decoder A recovery decode should succeed");
        opus_custom_decode(
            &mut decoder_b,
            Some(&packet[..recover_packet_len]),
            &mut out_b,
            POSTFILTER_FRAME_SIZE,
        )
        .expect("decoder B recovery decode should succeed");
        let recover_hash_after_loss = fnv1a_pcm_le(&out_a[..POSTFILTER_FRAME_SIZE]);
        let recover_hash_without_loss = fnv1a_pcm_le(&out_b[..POSTFILTER_FRAME_SIZE]);
        if strict_ctest_vectors {
            assert_eq!(recover_hash_after_loss, expected_recover_hash_after_loss);
            assert_eq!(
                recover_hash_without_loss,
                expected_recover_hash_without_loss
            );
        }

        opus_custom_decoder_ctl(&mut decoder_a, DecoderCtlRequest::ResetState)
            .expect("decoder A reset should succeed");
        opus_custom_decoder_ctl(&mut decoder_b, DecoderCtlRequest::ResetState)
            .expect("decoder B reset should succeed");
        opus_custom_decode(&mut decoder_a, None, &mut out_a, POSTFILTER_FRAME_SIZE)
            .expect("decoder A post-reset loss decode should succeed");
        opus_custom_decode(&mut decoder_b, None, &mut out_b, POSTFILTER_FRAME_SIZE)
            .expect("decoder B post-reset loss decode should succeed");
        let reset_hash_a = fnv1a_pcm_le(&out_a[..POSTFILTER_FRAME_SIZE]);
        let reset_hash_b = fnv1a_pcm_le(&out_b[..POSTFILTER_FRAME_SIZE]);
        assert_eq!(reset_hash_a, reset_hash_b);
        assert_ne!(reset_hash_a, 0);
        if strict_ctest_vectors {
            assert_eq!(reset_hash_a, expected_reset_loss_hash);
            assert_eq!(reset_hash_b, expected_reset_loss_hash);
        }
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn decoder_dataflow_matches_ctest_vectors() {
        let strict_ctest_vectors = std::env::var_os("RUST_CTEST_STRICT_HASHES").is_some();
        let expected_packet_hashes: [u32; DECODER_DATAFLOW_FRAMES] = [
            0x1e1e_1269,
            0x2dde_7528,
            0x8e00_0bce,
            0x6790_92e8,
            0x143e_3a1e,
            0x6986_4649,
            0xe63d_689d,
            0x9af6_d244,
            0x6c7a_cb7a,
            0x6c5b_cdd9,
            0xbead_c667,
            0x3e08_723e,
            0x834e_54b0,
        ];
        let expected_pcm_hashes: [u32; DECODER_DATAFLOW_FRAMES] = [
            0x72d2_ff52,
            0xfdbe_8b96,
            0xf442_2f17,
            0x15d7_fb47,
            0x4e10_ba6e,
            0xbb83_37f9,
            0x0942_b6c6,
            0x8bee_73ad,
            0xdd70_c41d,
            0x630c_948a,
            0xf86a_2a05,
            0xf047_b0c5,
            0x5502_c350,
        ];
        let expected_final_ranges: [u32; DECODER_DATAFLOW_FRAMES] = [
            0x212d_dc00,
            0x26b1_e200,
            0x0392_e900,
            0x1073_2300,
            0x6328_5f00,
            0x1946_6300,
            0x0b72_b000,
            0x0462_6100,
            0x69b1_6100,
            0x1bd0_aa00,
            0x067a_c700,
            0x0831_5200,
            0x3611_a800,
        ];

        let owned_mode = opus_custom_mode_create(48_000, POSTFILTER_FRAME_SIZE)
            .expect("mode creation should succeed");
        let mode = owned_mode.mode();
        assert_eq!(mode.num_ebands as i32, DECODER_DATAFLOW_NB_EBANDS);

        let mut encoder = opus_custom_encoder_create(&mode, 48_000, 2, 0)
            .expect("encoder creation should succeed");
        let mut decoder =
            opus_custom_decoder_create(&mode, 2).expect("decoder creation should succeed");

        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetBitrate(64_000))
            .expect("set bitrate should succeed");
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetVbr(false))
            .expect("set vbr should succeed");
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetComplexity(10))
            .expect("set complexity should succeed");
        opus_custom_encoder_ctl(&mut encoder, EncoderCtlRequest::SetLsbDepth(16))
            .expect("set lsb depth should succeed");
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetComplexity(10))
            .expect("decoder complexity should succeed");

        assert_eq!(
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetComplexity(11))
                .expect_err("invalid complexity must fail"),
            CeltDecoderCtlError::InvalidArgument
        );
        assert_eq!(
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetStartBand(-1))
                .expect_err("negative start band must fail"),
            CeltDecoderCtlError::InvalidArgument
        );
        assert_eq!(
            opus_custom_decoder_ctl(
                &mut decoder,
                DecoderCtlRequest::SetStartBand(DECODER_DATAFLOW_NB_EBANDS)
            )
            .expect_err("out-of-range start band must fail"),
            CeltDecoderCtlError::InvalidArgument
        );
        assert_eq!(
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetEndBand(0))
                .expect_err("zero end band must fail"),
            CeltDecoderCtlError::InvalidArgument
        );
        assert_eq!(
            opus_custom_decoder_ctl(
                &mut decoder,
                DecoderCtlRequest::SetEndBand(DECODER_DATAFLOW_NB_EBANDS + 1)
            )
            .expect_err("out-of-range end band must fail"),
            CeltDecoderCtlError::InvalidArgument
        );
        assert_eq!(
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(0))
                .expect_err("zero stream channels must fail"),
            CeltDecoderCtlError::InvalidArgument
        );
        assert_eq!(
            opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(3))
                .expect_err("three stream channels must fail"),
            CeltDecoderCtlError::InvalidArgument
        );

        let mut bad_output = vec![0i16; 123 * 2];
        assert!(
            opus_custom_decode(&mut decoder, Some(&[0]), &mut bad_output, 123).is_err(),
            "decode with invalid frame size should fail",
        );
        #[cfg(feature = "deep_plc")]
        let plc = None;
        #[cfg(not(feature = "deep_plc"))]
        let plc = ();
        let mut empty_pcm: [f32; 0] = [];
        assert!(
            super::celt_decode_with_ec_dred(
                &mut decoder,
                None,
                &mut empty_pcm,
                POSTFILTER_FRAME_SIZE,
                None,
                false,
                plc
            )
            .is_err(),
            "decode with empty PCM output should fail",
        );

        let mut packet = vec![0u8; POSTFILTER_MAX_PACKET_SIZE];
        let mut pcm = vec![0i16; POSTFILTER_FRAME_SIZE * 2];
        let mut decoded = vec![0i16; POSTFILTER_FRAME_SIZE * 2];
        let mut packet_hashes = [0u32; DECODER_DATAFLOW_FRAMES];
        let mut pcm_hashes = [0u32; DECODER_DATAFLOW_FRAMES];
        let mut final_ranges = [0u32; DECODER_DATAFLOW_FRAMES];

        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetStartBand(0))
            .expect("set start band should succeed");
        opus_custom_decoder_ctl(
            &mut decoder,
            DecoderCtlRequest::SetEndBand(DECODER_DATAFLOW_NB_EBANDS),
        )
        .expect("set end band should succeed");
        opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(2))
            .expect("set stream channels should succeed");

        for frame in 0..DECODER_DATAFLOW_FRAMES {
            if frame == 4 {
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetStartBand(3))
                    .expect("set start band should succeed");
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetEndBand(18))
                    .expect("set end band should succeed");
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(2))
                    .expect("set stream channels should succeed");
            } else if frame == 8 {
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetStartBand(2))
                    .expect("set start band should succeed");
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetEndBand(19))
                    .expect("set end band should succeed");
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(1))
                    .expect("set stream channels should succeed");
            } else if frame == 12 {
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::ResetState)
                    .expect("decoder reset should succeed");
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetStartBand(0))
                    .expect("set start band should succeed");
                opus_custom_decoder_ctl(
                    &mut decoder,
                    DecoderCtlRequest::SetEndBand(DECODER_DATAFLOW_NB_EBANDS),
                )
                .expect("set end band should succeed");
                opus_custom_decoder_ctl(&mut decoder, DecoderCtlRequest::SetChannels(2))
                    .expect("set stream channels should succeed");
            }

            fill_decoder_dataflow_pcm(&mut pcm, POSTFILTER_FRAME_SIZE, frame * 13);
            let packet_len = opus_custom_encode(
                &mut encoder,
                &pcm,
                POSTFILTER_FRAME_SIZE,
                &mut packet,
                POSTFILTER_MAX_PACKET_SIZE,
            )
            .expect("encode should succeed");
            let decoded_len = opus_custom_decode(
                &mut decoder,
                Some(&packet[..packet_len]),
                &mut decoded,
                POSTFILTER_FRAME_SIZE,
            )
            .expect("decode should succeed");
            assert_eq!(decoded_len, POSTFILTER_FRAME_SIZE);

            let mut final_range = 0u32;
            opus_custom_decoder_ctl(
                &mut decoder,
                DecoderCtlRequest::GetFinalRange(&mut final_range),
            )
            .expect("get final range should succeed");

            packet_hashes[frame] = fnv1a_bytes(&packet[..packet_len]);
            pcm_hashes[frame] = fnv1a_pcm_le(&decoded[..POSTFILTER_FRAME_SIZE * 2]);
            final_ranges[frame] = final_range;
        }

        assert!(count_hash_changes(&packet_hashes) > 0);
        assert!(count_hash_changes(&pcm_hashes) > 0);
        assert!(count_hash_changes(&final_ranges) > 0);
        if strict_ctest_vectors {
            assert_eq!(packet_hashes, expected_packet_hashes);
            assert_eq!(pcm_hashes, expected_pcm_hashes);
            assert_eq!(final_ranges, expected_final_ranges);
        }

        assert!(
            opus_custom_decode(
                &mut decoder,
                Some(&[0]),
                &mut decoded,
                POSTFILTER_FRAME_SIZE,
            )
            .is_err(),
            "tiny packet decode should fail",
        );

        assert!(
            opus_custom_decode(
                &mut decoder,
                Some(&vec![0u8; POSTFILTER_MAX_PACKET_SIZE + 1]),
                &mut decoded,
                POSTFILTER_FRAME_SIZE,
            )
            .is_err(),
            "oversize packet decode should fail",
        );
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn decoder_postfilter_matches_ctest_fixed_vectors() {
        let owned_mode = opus_custom_mode_create(POSTFILTER_SAMPLE_RATE, POSTFILTER_FRAME_SIZE)
            .expect("mode creation should succeed");
        let mode = owned_mode.mode();

        let cases = [
            DecoderPostfilterCase {
                name: "mono_postfilter",
                channels: 1,
                bitrate: 24_000,
                max_bytes: 96,
                min_nonzero_pitch_frames: 3,
                require_pitch_changes: true,
                expected_pitch: [430, 954, 954, 0, 528, 528],
                expected_packet_hash: [
                    0xd9db_cb3a,
                    0x5e4e_25b9,
                    0x5de8_bc73,
                    0x3d80_f8c0,
                    0x68d6_2f7c,
                    0x50eb_f0b5,
                ],
                expected_pcm_hash: [
                    0xba7e_fc92,
                    0xc441_26c4,
                    0xa582_4709,
                    0x4d7c_01ed,
                    0x85bb_0370,
                    0x6911_f513,
                ],
            },
            DecoderPostfilterCase {
                name: "stereo_postfilter",
                channels: 2,
                bitrate: 64_000,
                max_bytes: 180,
                min_nonzero_pitch_frames: 0,
                require_pitch_changes: false,
                expected_pitch: [0, 480, 480, 480, 480, 480],
                expected_packet_hash: [
                    0x5cdf_3e0c,
                    0x6fc9_dad3,
                    0x9406_08ff,
                    0x49c0_9e95,
                    0xe63d_158e,
                    0x418b_6748,
                ],
                expected_pcm_hash: [
                    0x042e_972d,
                    0x547d_2986,
                    0x8c2e_ee5d,
                    0x9d66_7d9a,
                    0xbb76_e17f,
                    0xdb24_1031,
                ],
            },
        ];

        for case in &cases {
            run_decoder_postfilter_case(&mode, case);
        }
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn decoder_noise_renorm_runtime_matches_ctest_vectors() {
        // Mirrors the fixed-point renormalise vectors covered by ctests/vq_test.c.
        let mut zeros = [0.0f32; 8];
        decoder_noise_renormalise_runtime(&mut zeros, 8, 1.0, 0);
        let zeros_q15: Vec<i16> = zeros.iter().map(|&sample| float2int16(sample)).collect();
        assert_eq!(zeros_q15, [0; 8], "zero-energy vector should remain zero");

        let mut mixed = [
            1000.0, -2000.0, 3000.0, -4000.0, 500.0, -600.0, 700.0, -800.0,
        ];
        decoder_noise_renormalise_runtime(&mut mixed, 8, 1.0, 0);
        let mixed_q15: Vec<i16> = mixed.iter().map(|&sample| float2int16(sample)).collect();
        assert_eq!(
            mixed_q15,
            [2908, -5816, 8724, -11632, 1454, -1745, 2036, -2326],
            "mixed renormalise result should match C fixed output",
        );

        let mut large = [30000.0, -30000.0, 20000.0, -10000.0];
        decoder_noise_renormalise_runtime(&mut large, 4, 1.0, 0);
        let large_q15: Vec<i16> = large.iter().map(|&sample| float2int16(sample)).collect();
        assert_eq!(
            large_q15,
            [10248, -10248, 6832, -3416],
            "large-magnitude renormalise result should match C fixed output",
        );

        let mut half_gain = [30000.0, -30000.0, 20000.0, -10000.0];
        decoder_noise_renormalise_runtime(&mut half_gain, 4, 0.5, 0);
        let half_gain_q15: Vec<i16> = half_gain
            .iter()
            .map(|&sample| float2int16(sample))
            .collect();
        assert_eq!(
            half_gain_q15,
            [5124, -5124, 3416, -1708],
            "gain-scaled renormalise result should match C fixed output",
        );
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    #[should_panic(expected = "input vector shorter than band size")]
    fn decoder_noise_renorm_runtime_panics_on_short_input() {
        let mut short = [1000.0f32, -500.0];
        decoder_noise_renormalise_runtime(&mut short, 3, 1.0, 0);
    }

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

    #[test]
    fn celt_synthesis_compare_output() {
        let mode = opus_custom_mode_find_static(48_000, 120)
            .expect("static 48k/120 mode should be available");
        let lm = 0usize;
        let is_transient = false;
        let coded_channels = 1usize;
        let output_channels = 1usize;
        let start = 0usize;
        let eff_end = mode.effective_ebands;
        let downsample = 1usize;
        let silence = false;

        let n = mode.short_mdct_size << lm;
        let nb_ebands = mode.num_ebands;
        let shift = if is_transient {
            mode.max_lm
        } else {
            mode.max_lm - lm
        };
        let mdct_len = mode.mdct.effective_len(shift);
        let n2 = mdct_len >> 1;
        let output_len = (mode.overlap >> 1) + n2;

        let mut x = vec![0.0f32; coded_channels * n];
        for (i, sample) in x.iter_mut().enumerate() {
            *sample = i as CeltNorm * 0.01 - 0.5;
        }

        let mut old_band_e = vec![0.0f32; coded_channels * nb_ebands];
        for (i, value) in old_band_e.iter_mut().enumerate() {
            *value = 0.5 + i as CeltGlog * 0.01;
        }

        #[cfg(feature = "fixed_point")]
        let fixed_mdct =
            crate::celt::mdct_fixed::FixedMdctLookup::new(mode.mdct.len(), mode.mdct.max_shift());
        #[cfg(feature = "fixed_point")]
        let fixed_window: Vec<crate::celt::types::FixedCeltCoef> = mode
            .window
            .iter()
            .map(|&value| crate::celt::fixed_ops::qconst16_clamped(f64::from(value), 15))
            .collect();
        #[cfg(feature = "fixed_point")]
        let fixed_ctx = (&fixed_mdct, fixed_window.as_slice(), mode.overlap);
        #[cfg(not(feature = "fixed_point"))]
        let fixed_ctx = ();

        let mut output = vec![0.0f32; output_len];
        {
            let mut outputs: Vec<&mut [CeltSig]> = vec![output.as_mut_slice()];
            super::celt_synthesis(
                &mode,
                &x,
                &mut outputs,
                &old_band_e,
                start,
                eff_end,
                coded_channels,
                output_channels,
                is_transient,
                lm,
                downsample,
                silence,
                fixed_ctx,
            );
        }

        for (i, sample) in output.iter().enumerate() {
            crate::test_trace::trace_println!("celt_synthesis_out[{}]={:.9e}", i, sample);
        }
    }
}
