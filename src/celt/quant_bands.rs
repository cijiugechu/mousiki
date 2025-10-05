#![allow(dead_code)]

//! Quantisation helpers for band energies.
//!
//! This module gathers routines from `celt/quant_bands.c` that have few
//! dependencies so they can be ported in isolation. The helpers operate on the
//! logarithmic band energy buffers shared between the encoder and decoder.

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::entcode::{EcWindow, ec_tell, ec_tell_frac};
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::celt::rate::MAX_FINE_BITS;
use crate::celt::types::{CeltGlog, OpusCustomMode};
use libm::floorf;

use crate::celt::math::{celt_exp2, celt_log2};
const TOTAL_FREQ: u32 = 1 << 15;
const LAPLACE_MINP: u32 = 1;
const LAPLACE_NMIN: u32 = 16;

const INV_Q15: f32 = 1.0 / 16_384.0;

#[derive(Clone)]
struct EcEncSnapshot {
    storage: u32,
    end_offs: u32,
    end_window: EcWindow,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    error: i32,
    buffer: Vec<u8>,
}

impl EcEncSnapshot {
    fn capture(enc: &EcEnc<'_>) -> Self {
        let ctx = enc.ctx();
        Self {
            storage: ctx.storage,
            end_offs: ctx.end_offs,
            end_window: ctx.end_window,
            nend_bits: ctx.nend_bits,
            nbits_total: ctx.nbits_total,
            offs: ctx.offs,
            rng: ctx.rng,
            val: ctx.val,
            ext: ctx.ext,
            rem: ctx.rem,
            error: ctx.error,
            buffer: ctx.buffer().to_vec(),
        }
    }

    fn restore(&self, enc: &mut EcEnc<'_>) {
        let ctx = enc.ctx_mut();
        assert_eq!(self.buffer.len(), ctx.buffer().len());
        ctx.storage = self.storage;
        ctx.end_offs = self.end_offs;
        ctx.end_window = self.end_window;
        ctx.nend_bits = self.nend_bits;
        ctx.nbits_total = self.nbits_total;
        ctx.offs = self.offs;
        ctx.rng = self.rng;
        ctx.val = self.val;
        ctx.ext = self.ext;
        ctx.rem = self.rem;
        ctx.error = self.error;
        ctx.buffer_mut().copy_from_slice(&self.buffer);
    }
}

fn laplace_get_freq1(fs0: u32, decay: u32) -> u32 {
    let remaining = TOTAL_FREQ - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    if decay >= 16_384 {
        0
    } else {
        let factor = 16_384 - decay;
        ((remaining as u64 * factor as u64) >> 15) as u32
    }
}

fn apply_sign(value: i32, sign: i32) -> i32 {
    (value + sign) ^ sign
}

fn laplace_encode(enc: &mut EcEnc<'_>, value: &mut i32, mut fs: u32, decay: u32) {
    let mut fl = 0u32;
    let mut val = *value;

    if val != 0 {
        let sign = if val < 0 { -1 } else { 0 };
        val = apply_sign(val, sign);
        let mut i = 1;
        fl = fs;
        fs = laplace_get_freq1(fs, decay);

        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            fs = ((fs as u64 * decay as u64) >> 15) as u32;
            i += 1;
        }

        if fs == 0 {
            let mut ndi_max = ((TOTAL_FREQ - fl + LAPLACE_MINP - 1) >> 0) as i32;
            ndi_max = (ndi_max - sign) >> 1;
            let di = core::cmp::min(val - i, ndi_max - 1);
            fl += ((2 * di + 1 + sign) as u32) * LAPLACE_MINP;
            fs = core::cmp::min(LAPLACE_MINP, TOTAL_FREQ - fl);
            *value = apply_sign(i + di, sign);
        } else {
            fs += LAPLACE_MINP;
            if sign == 0 {
                fl += fs;
            }
        }

        debug_assert!(fl + fs <= TOTAL_FREQ);
        debug_assert!(fs > 0);
    }

    let high = (fl + fs).min(TOTAL_FREQ);
    enc.encode_bin(fl, high, 15);
}

fn laplace_decode(dec: &mut EcDec<'_>, mut fs: u32, decay: u32) -> i32 {
    let mut val = 0i32;
    let mut fl = 0u32;
    let fm = dec.decode_bin(15);

    if fm >= fs {
        val += 1;
        fl = fs;
        fs = laplace_get_freq1(fs, decay) + LAPLACE_MINP;

        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = (((fs - 2 * LAPLACE_MINP) as u64 * decay as u64) >> 15) as u32;
            fs += LAPLACE_MINP;
            val += 1;
        }

        if fs <= LAPLACE_MINP {
            let di = ((fm - fl) >> 1) as i32;
            val += di;
            fl += 2 * di as u32 * LAPLACE_MINP;
        }

        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }

    let high = (fl + fs).min(TOTAL_FREQ);
    dec.update(fl, high, TOTAL_FREQ);

    val
}

/// Mean band energies mirroring `eMeans` from `celt/quant_bands.c`.
#[allow(dead_code)]
pub(crate) const E_MEANS: [f32; 25] = [
    6.437_5, 6.25, 5.75, 5.312_5, 5.062_5, 4.812_5, 4.5, 4.375, 4.875, 4.687_5, 4.562_5, 4.437_5,
    4.875, 4.625, 4.312_5, 4.5, 4.375, 4.625, 4.75, 4.437_5, 3.75, 3.75, 3.75, 3.75, 3.75,
];

/// Prediction coefficients (`pred_coef`) converted to floating point.
#[allow(dead_code)]
pub(crate) const PRED_COEF: [f32; 4] = [
    29_440.0 / 32_768.0,
    26_112.0 / 32_768.0,
    21_248.0 / 32_768.0,
    16_384.0 / 32_768.0,
];

/// `beta_coef` prediction feedback constants from the reference implementation.
#[allow(dead_code)]
pub(crate) const BETA_COEF: [f32; 4] = [
    30_147.0 / 32_768.0,
    22_282.0 / 32_768.0,
    12_124.0 / 32_768.0,
    6_554.0 / 32_768.0,
];

/// Intra-frame beta coefficient (`beta_intra`) from `celt/quant_bands.c`.
#[allow(dead_code)]
pub(crate) const BETA_INTRA: f32 = 4_915.0 / 32_768.0;

/// Laplace model parameters (`e_prob_model`) indexed by frame size, prediction
/// type, and band.
#[allow(dead_code)]
pub(crate) const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

/// Small energy inverse CDF table from `celt/quant_bands.c`.
#[allow(dead_code)]
pub(crate) const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// Returns a conservative distortion score between the current and previous
/// band energies.
///
/// Mirrors the `loss_distortion()` helper from `celt/quant_bands.c`. The C
/// routine iterates over the encoded bands for each channel, scales the
/// difference between the newly computed energies and the historical values,
/// and accumulates a squared error metric. In the floating-point build the
/// scaling macros collapse to no-ops, so the score is simply the sum of squared
/// differences, clamped to an upper bound of `200.0`.
pub(crate) fn loss_distortion(
    e_bands: &[CeltGlog],
    old_e_bands: &[CeltGlog],
    start: usize,
    end: usize,
    bands_per_channel: usize,
    channels: usize,
) -> f32 {
    assert_eq!(
        e_bands.len(),
        old_e_bands.len(),
        "energy buffers must match"
    );
    assert!(
        end <= bands_per_channel,
        "end band must lie within the channel span"
    );
    assert!(start <= end, "start band cannot exceed end band");
    assert!(channels * bands_per_channel <= e_bands.len());

    let mut distortion = 0.0f32;

    for channel in 0..channels {
        let base = channel * bands_per_channel;
        for band in start..end {
            let idx = base + band;
            let delta = e_bands[idx] - old_e_bands[idx];
            distortion += delta * delta;
        }
    }

    distortion.min(200.0)
}

#[allow(clippy::too_many_arguments)]
fn quant_coarse_energy_impl(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    e_bands: &[CeltGlog],
    old_e_bands: &mut [CeltGlog],
    budget: i32,
    initial_tell: i32,
    prob_model: &[u8],
    error: &mut [CeltGlog],
    enc: &mut EcEnc<'_>,
    channels: usize,
    lm: usize,
    intra: bool,
    max_decay: f32,
    lfe: bool,
) -> i32 {
    assert!(lm < PRED_COEF.len());
    assert_eq!(old_e_bands.len(), channels * mode.num_ebands);
    assert_eq!(e_bands.len(), channels * mode.num_ebands);
    assert_eq!(error.len(), channels * mode.num_ebands);
    assert!(end <= mode.num_ebands);
    assert!(start <= end);
    assert!(prob_model.len() >= 2 * core::cmp::min(end, 21));

    let stride = mode.num_ebands;
    let mut prev = vec![0.0f32; channels];
    let coef = if intra { 0.0 } else { PRED_COEF[lm] };
    let beta = if intra { BETA_INTRA } else { BETA_COEF[lm] };
    let mut badness = 0;
    let channels_i32 = channels as i32;

    if initial_tell + 3 <= budget {
        enc.enc_bit_logp(intra as i32, 3);
    }

    for band in start..end {
        for channel in 0..channels {
            let idx = channel * stride + band;
            let x = e_bands[idx];
            let old_e = old_e_bands[idx].max(-9.0);
            let f = x - coef * old_e - prev[channel];
            let mut qi = floorf(f + 0.5) as i32;
            let decay_bound = old_e_bands[idx].max(-28.0) - max_decay;
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }

            let qi0 = qi;
            let tell = ec_tell(enc.ctx());
            let bits_left = budget - tell - 3 * channels_i32 * (end - band) as i32;
            if band != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            if lfe && band >= 2 {
                qi = qi.min(0);
            }

            if budget - tell >= 15 {
                let pi = 2 * core::cmp::min(band, 20);
                let mut symbol = qi;
                laplace_encode(
                    enc,
                    &mut symbol,
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as u32) << 6,
                );
                qi = symbol;
            } else if budget - tell >= 2 {
                qi = qi.clamp(-1, 1);
                let symbol = ((2 * qi) ^ -i32::from(qi < 0)) as usize;
                enc.enc_icdf(symbol, &SMALL_ENERGY_ICDF, 2);
            } else if budget - tell >= 1 {
                qi = qi.min(0);
                enc.enc_bit_logp((-qi) as i32, 1);
            } else {
                qi = -1;
            }

            error[idx] = f - qi as f32;
            badness += (qi0 - qi).abs();
            let q = qi as f32;
            let tmp = (coef * old_e) + prev[channel] + q;
            old_e_bands[idx] = tmp.max(-28.0);
            prev[channel] += q - beta * q;
        }
    }

    if lfe { 0 } else { badness }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn quant_coarse_energy(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    eff_end: usize,
    e_bands: &[CeltGlog],
    old_e_bands: &mut [CeltGlog],
    budget: u32,
    error: &mut [CeltGlog],
    enc: &mut EcEnc<'_>,
    channels: usize,
    lm: usize,
    nb_available_bytes: i32,
    force_intra: bool,
    delayed_intra: &mut f32,
    mut two_pass: bool,
    loss_rate: i32,
    lfe: bool,
) {
    assert!(end <= mode.num_ebands);
    assert!(eff_end <= end);
    assert_eq!(e_bands.len(), channels * mode.num_ebands);
    assert_eq!(old_e_bands.len(), channels * mode.num_ebands);
    assert_eq!(error.len(), channels * mode.num_ebands);
    assert!(lm < PRED_COEF.len());

    let channels_i32 = channels as i32;
    let band_span = (end - start) as i32;
    let mut intra = force_intra
        || (!two_pass
            && *delayed_intra > 2.0 * channels as f32 * band_span as f32
            && nb_available_bytes > band_span * channels_i32);

    let budget_i32 = budget as i32;
    let initial_tell = ec_tell(enc.ctx());
    if initial_tell + 3 > budget_i32 {
        two_pass = false;
        intra = false;
    }

    let intra_bias =
        ((budget as f32) * *delayed_intra * loss_rate as f32 / (channels as f32 * 512.0)) as i32;
    let new_distortion = loss_distortion(
        e_bands,
        old_e_bands,
        start,
        eff_end,
        mode.num_ebands,
        channels,
    );

    let mut max_decay = 16.0f32;
    if end - start > 10 {
        max_decay = max_decay.min(0.125f32 * nb_available_bytes as f32);
    }
    if lfe {
        max_decay = 3.0;
    }

    let start_snapshot = EcEncSnapshot::capture(enc);
    let mut old_intra = vec![0.0f32; old_e_bands.len()];
    let mut error_intra = vec![0.0f32; error.len()];
    let mut intra_snapshot = None;
    let mut badness_intra = 0;
    let mut tell_intra = 0u32;

    if two_pass || intra {
        old_intra.copy_from_slice(old_e_bands);
        error_intra.copy_from_slice(error);
        badness_intra = quant_coarse_energy_impl(
            mode,
            start,
            end,
            e_bands,
            &mut old_intra,
            budget_i32,
            initial_tell,
            &E_PROB_MODEL[lm][1],
            &mut error_intra,
            enc,
            channels,
            lm,
            true,
            max_decay,
            lfe,
        );
        intra_snapshot = Some(EcEncSnapshot::capture(enc));
        tell_intra = ec_tell_frac(enc.ctx());
    }

    if !intra {
        start_snapshot.restore(enc);
        let badness_inter = quant_coarse_energy_impl(
            mode,
            start,
            end,
            e_bands,
            old_e_bands,
            budget_i32,
            initial_tell,
            &E_PROB_MODEL[lm][0],
            error,
            enc,
            channels,
            lm,
            false,
            max_decay,
            lfe,
        );

        if two_pass
            && (badness_intra < badness_inter
                || (badness_intra == badness_inter
                    && (ec_tell_frac(enc.ctx()) as i32 + intra_bias) > tell_intra as i32))
        {
            if let Some(snapshot) = &intra_snapshot {
                snapshot.restore(enc);
            }
            old_e_bands.copy_from_slice(&old_intra);
            error.copy_from_slice(&error_intra);
            intra = true;
        }
    } else {
        if let Some(snapshot) = &intra_snapshot {
            snapshot.restore(enc);
        }
        old_e_bands.copy_from_slice(&old_intra);
        error.copy_from_slice(&error_intra);
    }

    if intra {
        *delayed_intra = new_distortion;
    } else {
        let coef = PRED_COEF[lm];
        *delayed_intra = coef * coef * *delayed_intra + new_distortion;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn unquant_coarse_energy(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_e_bands: &mut [CeltGlog],
    intra: bool,
    dec: &mut EcDec<'_>,
    channels: usize,
    lm: usize,
) {
    assert!(end <= mode.num_ebands);
    assert_eq!(old_e_bands.len(), channels * mode.num_ebands);
    assert!(lm < PRED_COEF.len());

    let stride = mode.num_ebands;
    let prob_model = &E_PROB_MODEL[lm][usize::from(intra)];
    let mut prev = vec![0.0f32; channels];
    let coef = if intra { 0.0 } else { PRED_COEF[lm] };
    let beta = if intra { BETA_INTRA } else { BETA_COEF[lm] };
    let budget = (dec.ctx().storage * 8) as i32;

    for band in start..end {
        for channel in 0..channels {
            let idx = channel * stride + band;
            let tell = ec_tell(dec.ctx());
            let qi = if budget - tell >= 15 {
                let pi = 2 * core::cmp::min(band, 20);
                laplace_decode(
                    dec,
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as u32) << 6,
                )
            } else if budget - tell >= 2 {
                let sym = dec.dec_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (sym >> 1) ^ -((sym & 1) as i32)
            } else if budget - tell >= 1 {
                -dec.dec_bit_logp(1)
            } else {
                -1
            };

            old_e_bands[idx] = old_e_bands[idx].max(-9.0);
            let q = qi as f32;
            let tmp = coef * old_e_bands[idx] + prev[channel] + q;
            old_e_bands[idx] = tmp.clamp(-28.0, 28.0);
            prev[channel] += q - beta * q;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn amp2_log2(
    mode: &OpusCustomMode<'_>,
    eff_end: usize,
    end: usize,
    band_e: &[CeltGlog],
    band_log_e: &mut [CeltGlog],
    channels: usize,
) {
    assert!(eff_end <= end);
    assert!(end <= mode.num_ebands);
    assert_eq!(band_e.len(), channels * mode.num_ebands);
    assert_eq!(band_log_e.len(), channels * mode.num_ebands);

    let stride = mode.num_ebands;
    for channel in 0..channels {
        for band in 0..eff_end {
            let idx = channel * stride + band;
            band_log_e[idx] = celt_log2(band_e[idx]) - E_MEANS[band];
        }
        for band in eff_end..end {
            let idx = channel * stride + band;
            band_log_e[idx] = -14.0;
        }
    }
}

/// Converts logarithmic band energies back to linear amplitudes.
///
/// Mirrors the float variant of `log2Amp()` from `celt/quant_bands.c`. The
/// helper reverses the transform performed by [`amp2_log2`] by reapplying the
/// per-band energy means and evaluating the base-2 exponential.
#[allow(clippy::too_many_arguments)]
pub(crate) fn log2_amp(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    e_bands: &mut [CeltGlog],
    old_e_bands: &[CeltGlog],
    channels: usize,
) {
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    let stride = mode.num_ebands;
    assert!(
        channels * stride <= e_bands.len(),
        "insufficient energy storage"
    );
    assert!(
        channels * stride <= old_e_bands.len(),
        "insufficient log energy storage"
    );

    for band in start..end {
        let mean = E_MEANS[band];
        for channel in 0..channels {
            let idx = channel * stride + band;
            e_bands[idx] = celt_exp2(old_e_bands[idx] + mean);
        }
    }
}

fn fine_energy_scale(fine: usize) -> f32 {
    debug_assert!(fine <= 14);
    let shift = 14usize.saturating_sub(fine);
    ((1usize << shift) as f32) * INV_Q15
}

fn fine_energy_final_scale(fine: usize) -> f32 {
    debug_assert!(fine <= 13);
    let shift = 14usize.saturating_sub(fine + 1);
    ((1usize << shift) as f32) * INV_Q15
}

/// Quantises the finer energy resolution bits for each band.
///
/// Mirrors the float portion of `quant_fine_energy()` from
/// `celt/quant_bands.c`. The function scans the requested bands, quantises the
/// fractional energy error, and accumulates the residual back into
/// `old_ebands`/`error` while appending the raw bits to `enc`.
pub(crate) fn quant_fine_energy(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_ebands: &mut [CeltGlog],
    error: &mut [CeltGlog],
    fine_quant: &[i32],
    enc: &mut EcEnc<'_>,
    channels: usize,
) {
    assert_eq!(old_ebands.len(), error.len(), "band buffers must align");
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(fine_quant.len() >= end, "fine quantiser metadata too short");
    assert!(
        channels * mode.num_ebands <= old_ebands.len(),
        "insufficient band data"
    );

    let stride = mode.num_ebands;

    for band in start..end {
        let fine = fine_quant[band];
        if fine <= 0 {
            continue;
        }

        let fine_bits = fine as usize;
        let frac = 1i32 << fine_bits;
        let max_q = frac - 1;
        let scale = fine_energy_scale(fine_bits);

        for channel in 0..channels {
            let idx = channel * stride + band;
            let target = error[idx] + 0.5;
            let mut q2 = floorf(target * frac as f32) as i32;
            q2 = q2.clamp(0, max_q);

            enc.enc_bits(q2 as u32, fine_bits as u32);

            let offset = (q2 as f32 + 0.5) * scale - 0.5;
            old_ebands[idx] += offset;
            error[idx] -= offset;
        }
    }
}

/// Consumes the remaining fine energy bits based on their priority.
///
/// Ports the float implementation of `quant_energy_finalise()` which allocates
/// leftover bits to low-priority bands and updates the running energy/error
/// estimates accordingly.
pub(crate) fn quant_energy_finalise(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_ebands: &mut [CeltGlog],
    error: &mut [CeltGlog],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut EcEnc<'_>,
    channels: usize,
) {
    assert_eq!(old_ebands.len(), error.len(), "band buffers must align");
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(fine_quant.len() >= end, "fine quantiser metadata too short");
    assert!(
        fine_priority.len() >= end,
        "fine priority metadata too short"
    );
    assert!(
        channels * mode.num_ebands <= old_ebands.len(),
        "insufficient band data"
    );

    let stride = mode.num_ebands;
    let channels_i32 = channels as i32;

    for priority in 0..2 {
        for band in start..end {
            if bits_left < channels_i32 {
                break;
            }

            let fine = fine_quant[band];
            if fine >= MAX_FINE_BITS || fine_priority[band] != priority {
                continue;
            }

            let fine_bits = fine.max(0) as usize;
            let scale = fine_energy_final_scale(fine_bits);

            for channel in 0..channels {
                let idx = channel * stride + band;
                let q2 = if error[idx] < 0.0 { 0 } else { 1 };
                enc.enc_bits(q2 as u32, 1);

                let offset = (q2 as f32 - 0.5) * scale;
                old_ebands[idx] += offset;
                error[idx] -= offset;
                bits_left -= 1;
            }
        }
    }
}

/// Restores the fine energy quantisation from the bit-stream.
///
/// Mirrors the float path of `unquant_fine_energy()` by replaying the raw bits
/// written by [`quant_fine_energy`] and accumulating the reconstructed energy
/// back into `old_ebands`.
pub(crate) fn unquant_fine_energy(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_ebands: &mut [CeltGlog],
    fine_quant: &[i32],
    dec: &mut EcDec<'_>,
    channels: usize,
) {
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(fine_quant.len() >= end, "fine quantiser metadata too short");
    assert!(
        channels * mode.num_ebands <= old_ebands.len(),
        "insufficient band data"
    );

    let stride = mode.num_ebands;

    for band in start..end {
        let fine = fine_quant[band];
        if fine <= 0 {
            continue;
        }

        let fine_bits = fine as usize;
        let scale = fine_energy_scale(fine_bits);

        for channel in 0..channels {
            let idx = channel * stride + band;
            let q2 = dec.dec_bits(fine_bits as u32) as i32;
            let offset = (q2 as f32 + 0.5) * scale - 0.5;
            old_ebands[idx] += offset;
        }
    }
}

/// Replays the final fine energy decisions for the decoder.
///
/// Ports the float build of `unquant_energy_finalise()` which consumes the
/// leftover single-bit decisions and updates the reconstructed energy buffer.
pub(crate) fn unquant_energy_finalise(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_ebands: &mut [CeltGlog],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut EcDec<'_>,
    channels: usize,
) {
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(fine_quant.len() >= end, "fine quantiser metadata too short");
    assert!(
        fine_priority.len() >= end,
        "fine priority metadata too short"
    );
    assert!(
        channels * mode.num_ebands <= old_ebands.len(),
        "insufficient band data"
    );

    let stride = mode.num_ebands;
    let channels_i32 = channels as i32;

    for priority in 0..2 {
        for band in start..end {
            if bits_left < channels_i32 {
                break;
            }

            let fine = fine_quant[band];
            if fine >= MAX_FINE_BITS || fine_priority[band] != priority {
                continue;
            }

            let fine_bits = fine.max(0) as usize;
            let scale = fine_energy_final_scale(fine_bits);

            for channel in 0..channels {
                let idx = channel * stride + band;
                let q2 = dec.dec_bits(1) as i32;
                let offset = (q2 as f32 - 0.5) * scale;
                old_ebands[idx] += offset;
                bits_left -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{
        BETA_COEF, BETA_INTRA, E_MEANS, E_PROB_MODEL, PRED_COEF, SMALL_ENERGY_ICDF, amp2_log2,
        loss_distortion, quant_coarse_energy, quant_energy_finalise, quant_fine_energy,
        unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy,
    };
    use crate::celt::entcode::ec_tell;
    use crate::celt::entdec::EcDec;
    use crate::celt::entenc::EcEnc;
    use crate::celt::types::{MdctLookup, OpusCustomMode, PulseCacheData};

    #[test]
    fn loss_distortion_matches_manual_accumulation() {
        let e = [
            // Channel 0
            -2.0f32, -1.0, 0.5, 0.25, // Channel 1
            1.5, 2.0, -0.75, 0.0,
        ];
        let old = [
            // Channel 0
            -1.5f32, -0.5, 0.0, 0.0, // Channel 1
            1.0, 1.25, -1.25, -0.25,
        ];

        let manual = e
            .iter()
            .zip(old.iter())
            .map(|(current, previous)| {
                let diff = current - previous;
                diff * diff
            })
            .sum::<f32>();

        let computed = loss_distortion(&e, &old, 0, 4, 4, 2);
        assert!((computed - manual).abs() <= f32::EPSILON * 32.0);
    }

    #[test]
    fn loss_distortion_clamps_to_upper_bound() {
        let e = [50.0f32; 6];
        let old = [0.0f32; 6];
        let result = loss_distortion(&e, &old, 0, 3, 3, 2);
        assert_eq!(result, 200.0);
    }

    #[test]
    fn quant_bands_constants_match_reference_values() {
        let expected_means = [
            6.437_5, 6.25, 5.75, 5.312_5, 5.062_5, 4.812_5, 4.5, 4.375, 4.875, 4.687_5, 4.562_5,
            4.437_5, 4.875, 4.625, 4.312_5, 4.5, 4.375, 4.625, 4.75, 4.437_5, 3.75, 3.75, 3.75,
            3.75, 3.75,
        ];
        assert_eq!(E_MEANS, expected_means);

        let expected_pred = [
            29_440.0 / 32_768.0,
            26_112.0 / 32_768.0,
            21_248.0 / 32_768.0,
            16_384.0 / 32_768.0,
        ];
        assert_eq!(PRED_COEF, expected_pred);

        let expected_beta = [
            30_147.0 / 32_768.0,
            22_282.0 / 32_768.0,
            12_124.0 / 32_768.0,
            6_554.0 / 32_768.0,
        ];
        assert_eq!(BETA_COEF, expected_beta);

        assert_eq!(BETA_INTRA, 4_915.0 / 32_768.0);
        assert_eq!(SMALL_ENERGY_ICDF, [2, 1, 0]);

        // Spot-check a couple of Laplace model entries to guard against typos.
        assert_eq!(E_PROB_MODEL[0][0][0], 72);
        assert_eq!(E_PROB_MODEL[0][1][1], 179);
        assert_eq!(E_PROB_MODEL[1][0][10], 90);
        assert_eq!(E_PROB_MODEL[2][1][20], 105);
        assert_eq!(E_PROB_MODEL[3][0][41], 15);
    }

    #[test]
    fn fine_energy_quantisation_round_trips() {
        let e_bands = [0i16, 1, 2, 3, 4];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 4];
        let window = [0.0f32; 4];
        let _twiddle = [0.0f32; 1];
        let mdct = MdctLookup::new(4, 0);
        let mode = OpusCustomMode::new(
            48_000,
            0,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            PulseCacheData::default(),
        );

        let mut encoded_old = [0.0f32; 8];
        let mut error = [0.6, -0.25, 0.1, -0.05, 0.4, -0.3, 0.0, 0.2];
        let fine_quant = [2, 1, 0, 3];
        let fine_priority = [0, 1, 0, 1];
        let bits_left = 4;
        let channels = 2;

        let mut buffer = vec![0u8; 32];
        {
            let mut enc = EcEnc::new(&mut buffer);
            quant_fine_energy(
                &mode,
                0,
                4,
                &mut encoded_old,
                &mut error,
                &fine_quant,
                &mut enc,
                channels,
            );
            quant_energy_finalise(
                &mode,
                0,
                4,
                &mut encoded_old,
                &mut error,
                &fine_quant,
                &fine_priority,
                bits_left,
                &mut enc,
                channels,
            );
            enc.enc_done();
        }

        let mut decoded_old = [0.0f32; 8];
        {
            let mut dec = EcDec::new(&mut buffer);
            unquant_fine_energy(
                &mode,
                0,
                4,
                &mut decoded_old,
                &fine_quant,
                &mut dec,
                channels,
            );
            unquant_energy_finalise(
                &mode,
                0,
                4,
                &mut decoded_old,
                &fine_quant,
                &fine_priority,
                bits_left,
                &mut dec,
                channels,
            );
        }

        for (enc, dec) in encoded_old.iter().zip(decoded_old.iter()) {
            assert!((enc - dec).abs() <= 1e-6);
        }
    }

    #[test]
    fn coarse_energy_round_trip_matches_encoder() {
        let e_bands = [0i16, 1, 2, 3, 4, 5, 6];
        let alloc_vectors = [0u8; 6];
        let log_n = [0i16; 6];
        let window = [0.0f32; 6];
        let mdct = MdctLookup::new(8, 0);
        let mode = OpusCustomMode::new(
            48_000,
            0,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            PulseCacheData::default(),
        );

        let channels = 1usize;
        let start = 0usize;
        let end = 4usize;
        let eff_end = 4usize;
        let lm = 0usize;
        let budget = 64u32;
        let nb_available_bytes = 12;
        let mut delayed_intra = 0.0f32;
        let mut old = [-2.0f32, -1.0, -0.5, 0.0, 0.0, 0.0];
        let mut error = [0.0f32; 6];
        let original_old = old;
        let e = [1.2f32, 0.5, -0.3, 2.0, 0.0, 0.0];

        let mut buffer = vec![0u8; 64];
        {
            let mut enc = EcEnc::new(&mut buffer);
            quant_coarse_energy(
                &mode,
                start,
                end,
                eff_end,
                &e,
                &mut old,
                budget,
                &mut error,
                &mut enc,
                channels,
                lm,
                nb_available_bytes,
                false,
                &mut delayed_intra,
                true,
                0,
                false,
            );
            enc.enc_done();
        }

        let mut decoded_old = original_old;
        {
            let mut dec = EcDec::new(&mut buffer);
            let tell = ec_tell(dec.ctx());
            let intra = if tell + 3 <= budget as i32 {
                dec.dec_bit_logp(3) != 0
            } else {
                false
            };

            unquant_coarse_energy(
                &mode,
                start,
                end,
                &mut decoded_old,
                intra,
                &mut dec,
                channels,
                lm,
            );
        }

        for (enc, dec) in old.iter().zip(decoded_old.iter()) {
            assert!((enc - dec).abs() <= 1e-5);
        }
    }

    #[test]
    fn amp2_log2_matches_expected_logarithm() {
        let e_bands = [0i16, 1, 2, 3, 4, 5, 6];
        let alloc_vectors = [0u8; 6];
        let log_n = [0i16; 6];
        let window = [0.0f32; 6];
        let mdct = MdctLookup::new(8, 0);
        let mode = OpusCustomMode::new(
            48_000,
            0,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            PulseCacheData::default(),
        );

        let channels = 1usize;
        let mut band_log_e = [0.0f32; 6];
        let band_e = [1.0f32, 2.0, 4.0, 8.0, 1.0, 1.0];

        amp2_log2(&mode, 3, 4, &band_e, &mut band_log_e, channels);

        assert!((band_log_e[0] - (band_e[0].log2() - E_MEANS[0])).abs() <= 1e-6);
        assert!((band_log_e[1] - (band_e[1].log2() - E_MEANS[1])).abs() <= 1e-6);
        assert_eq!(band_log_e[3], -14.0);
    }

    #[test]
    fn log2_amp_restores_linear_energies() {
        let e_bands = [0i16, 1, 2, 3, 4, 5, 6];
        let alloc_vectors = [0u8; 6];
        let log_n = [0i16; 6];
        let window = [0.0f32; 6];
        let mdct = MdctLookup::new(8, 0);
        let mode = OpusCustomMode::new(
            48_000,
            0,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            PulseCacheData::default(),
        );

        let channels = 2usize;
        let mut e = [0.0f32; 12];
        let log_e = [
            0.1f32, -0.3, 0.0, -1.0, 0.0, 0.0, 0.4, -0.2, 0.2, 0.0, 0.0, 0.0,
        ];

        super::log2_amp(&mode, 1, 4, &mut e, &log_e, channels);

        for channel in 0..channels {
            for band in 1..4 {
                let idx = channel * mode.num_ebands + band;
                let expected = crate::celt::math::celt_exp2(log_e[idx] + E_MEANS[band]);
                assert!((e[idx] - expected).abs() <= 1e-6);
            }
        }

        // Bands outside the requested range remain untouched.
        assert_eq!(e[0], 0.0);
        assert_eq!(e[mode.num_ebands], 0.0);
    }
}
