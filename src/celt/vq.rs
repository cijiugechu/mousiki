#![allow(dead_code)]

//! Vector quantisation helpers ported from `celt/vq.c`.
//!
//! The routines in this module operate on the normalised coefficient buffers
//! used by CELT's pulse shaping stage.  They are largely self-contained and
//! map closely to their C counterparts, making them ideal candidates for early
//! porting efforts.

use alloc::vec;
use core::convert::TryFrom;
use core::f32::consts::FRAC_2_PI;

use crate::celt::cwrs::{decode_pulses, encode_pulses};
use crate::celt::entcode::celt_udiv;
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::celt::math::{
    celt_cos_norm, celt_div, celt_rcp, celt_rsqrt_norm, celt_sqrt, fast_atan2f,
};
use crate::celt::pitch::celt_inner_prod;
use crate::celt::types::{OpusInt32, OpusVal16, OpusVal32};
use libm::floorf;
#[cfg(feature = "fixed_point")]
use crate::celt::fixed_ops::{extract16, mult16_16, mult16_16_q15, mult32_32_q31, pshr32, shr16, vshr32};
#[cfg(feature = "fixed_point")]
use crate::celt::math::celt_ilog2;
#[cfg(feature = "fixed_point")]
use crate::celt::math_fixed::{celt_atan2p, celt_rsqrt_norm as celt_rsqrt_norm_fixed, celt_sqrt as celt_sqrt_fixed};
#[cfg(feature = "fixed_point")]
use crate::celt::types::{FixedOpusVal16, FixedOpusVal32};

/// Spread decisions mirrored from `celt/bands.h`.
pub(crate) const SPREAD_NONE: i32 = 0;
pub(crate) const SPREAD_LIGHT: i32 = 1;
pub(crate) const SPREAD_NORMAL: i32 = 2;
pub(crate) const SPREAD_AGGRESSIVE: i32 = 3;

const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
const Q15_ONE: OpusVal16 = 1.0;
const EPSILON: OpusVal32 = 1e-15;

fn exp_rotation1(x: &mut [OpusVal16], stride: usize, c: OpusVal16, s: OpusVal16) {
    let len = x.len();
    if stride == 0 || len <= stride {
        return;
    }

    let ms = -s;

    for i in 0..(len - stride) {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }

    if len > 2 * stride {
        let limit = len - 2 * stride - 1;
        for i in (0..=limit).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            x[i + stride] = c * x2 + s * x1;
            x[i] = c * x1 + ms * x2;
        }
    }
}

/// Port of `exp_rotation()` from `celt/vq.c`.
///
/// Applies a spreading rotation to the coefficient buffer in-place.  The logic
/// matches the float build of the reference implementation, relying on Rust's
/// slice handling for safety while keeping the numerical behaviour intact.
pub(crate) fn exp_rotation(
    x: &mut [OpusVal16],
    len: usize,
    dir: i32,
    stride: usize,
    k: i32,
    spread: i32,
) {
    if len == 0 || stride == 0 {
        return;
    }

    let slice_len = len.min(x.len());
    let x = &mut x[..slice_len];

    if 2 * k >= slice_len as i32 || spread == SPREAD_NONE {
        return;
    }

    let spread_index = match spread {
        SPREAD_LIGHT => 0,
        SPREAD_NORMAL => 1,
        SPREAD_AGGRESSIVE => 2,
        _ => return,
    };

    let factor = SPREAD_FACTOR[spread_index];
    let gain = celt_div(
        Q15_ONE * slice_len as OpusVal16,
        (slice_len + (factor * k) as usize) as OpusVal16,
    );
    let theta = 0.5 * gain * gain;
    let c = celt_cos_norm(theta);
    let s = celt_cos_norm(Q15_ONE - theta);

    let mut stride2 = 0usize;
    if slice_len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < slice_len {
            stride2 += 1;
        }
    }

    let len_div = slice_len / stride;
    if len_div == 0 {
        return;
    }

    for band in 0..stride {
        let start = band * len_div;
        let end = start + len_div;
        let band_slice = &mut x[start..end];
        if dir < 0 {
            if stride2 > 0 {
                exp_rotation1(band_slice, stride2, s, c);
            }
            exp_rotation1(band_slice, 1, c, s);
        } else {
            exp_rotation1(band_slice, 1, c, -s);
            if stride2 > 0 {
                exp_rotation1(band_slice, stride2, s, -c);
            }
        }
    }
}

/// Port of `normalise_residual()` from `celt/vq.c`.
///
/// The helper mixes the decoded PVQ pulses with the pitch vector so that the
/// resulting excitation has unit energy. The float build performs the scaling
/// by computing the reciprocal square root of the accumulated pulse energy and
/// multiplying by the supplied gain. The Rust port follows the same approach
/// while clamping the slice lengths to avoid overruns.
pub(crate) fn normalise_residual(
    pulses: &[i32],
    x: &mut [OpusVal16],
    n: usize,
    ryy: OpusVal32,
    gain: OpusVal32,
) {
    if n == 0 {
        return;
    }

    debug_assert!(pulses.len() >= n, "pulse buffer shorter than band size");
    debug_assert!(x.len() >= n, "output buffer shorter than band size");

    let len = n.min(pulses.len()).min(x.len());
    if len == 0 {
        return;
    }

    let scale = celt_rsqrt_norm(ryy) * gain;
    for (dst, &pulse) in x.iter_mut().take(len).zip(pulses.iter()) {
        *dst = scale * pulse as OpusVal16;
    }
}

#[cfg(feature = "fixed_point")]
pub(crate) fn normalise_residual_fixed(
    pulses: &[i32],
    x: &mut [FixedOpusVal16],
    n: usize,
    ryy: FixedOpusVal32,
    gain: FixedOpusVal32,
) {
    if n == 0 {
        return;
    }

    debug_assert!(pulses.len() >= n, "pulse buffer shorter than band size");
    debug_assert!(x.len() >= n, "output buffer shorter than band size");

    let len = n.min(pulses.len()).min(x.len());
    if len == 0 || ryy == 0 {
        return;
    }

    let k = celt_ilog2(ryy) >> 1;
    let t = vshr32(ryy, 2 * (k - 7));
    let g = mult32_32_q31(i32::from(celt_rsqrt_norm_fixed(t)), gain) as i16;

    for (dst, &pulse) in x.iter_mut().take(len).zip(pulses.iter()) {
        *dst = extract16(pshr32(mult16_16(g, pulse as i16), (k + 1) as u32));
    }
}

/// Float port of `op_pvq_search_c()` from `celt/vq.c`.
///
/// The helper distributes `K` algebraic pulses across the `N`-dimensional
/// coefficient vector `x`, returning the squared energy of the chosen pulse
/// vector. The routine mirrors the reference implementation by performing a
/// greedy search that maximises the correlation proxy `Rxy / sqrt(Ryy)` without
/// taking expensive square roots inside the inner loop.
pub(crate) fn op_pvq_search(
    x: &mut [OpusVal16],
    pulses: &mut [OpusInt32],
    n: usize,
    k: i32,
    _arch: i32,
) -> OpusVal32 {
    assert!(n > 0, "vector dimension must be positive");
    assert!(k >= 0, "pulse count must be non-negative");
    assert!(x.len() >= n, "coefficient buffer shorter than band size");
    assert!(pulses.len() >= n, "pulse buffer shorter than band size");

    let mut y = vec![0.0f32; n];
    let mut sign = vec![false; n];

    for (idx, sample) in x.iter_mut().enumerate().take(n) {
        let value = *sample;
        sign[idx] = value < 0.0;
        *sample = value.abs();
        pulses[idx] = 0;
        y[idx] = 0.0;
    }

    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    let mut pulses_left = k;

    if k > ((n as i32) >> 1) {
        let mut sum = 0.0f32;
        for &sample in x.iter().take(n) {
            sum += sample;
        }

        if !(sum > EPSILON && sum < 64.0) {
            if n > 0 {
                x[0] = 1.0;
                for coeff in x.iter_mut().take(n).skip(1) {
                    *coeff = 0.0;
                }
            }
            sum = 1.0;
        }

        let rcp = (k as OpusVal32 + 0.8) * celt_rcp(sum);
        for idx in 0..n {
            let projected = floorf(rcp * x[idx]);
            let pulse = projected as OpusInt32;
            pulses[idx] = pulse;
            let val = pulse as OpusVal16;
            y[idx] = val;
            yy += val * val;
            xy += x[idx] * val;
            y[idx] *= 2.0;
            pulses_left -= pulse;
        }
    }

    debug_assert!(pulses_left >= 0, "pulse allocation exceeded target count");
    if pulses_left < 0 {
        pulses_left = 0;
    }

    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as OpusVal16;
        yy += tmp * tmp;
        yy += tmp * y[0];
        pulses[0] += pulses_left;
        pulses_left = 0;
    }

    for _ in 0..pulses_left {
        yy += 1.0;

        let mut best_id = 0usize;
        let mut best_den = yy + y[0];
        let mut best_num = (xy + x[0]) * (xy + x[0]);

        for idx in 1..n {
            let rxy = xy + x[idx];
            let ryy = yy + y[idx];
            let num = rxy * rxy;
            if best_den * num > ryy * best_num {
                best_den = ryy;
                best_num = num;
                best_id = idx;
            }
        }

        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        pulses[best_id] += 1;
    }

    for (idx, pulse) in pulses.iter_mut().take(n).enumerate() {
        if sign[idx] {
            *pulse = -*pulse;
        }
    }

    yy
}

/// Algebraic pulse quantiser from `celt/vq.c`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn alg_quant(
    x: &mut [OpusVal16],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    enc: &mut EcEnc<'_>,
    gain: OpusVal32,
    resynth: bool,
    arch: i32,
) -> u32 {
    assert!(k > 0, "alg_quant requires at least one pulse");
    assert!(n > 1, "alg_quant requires at least two dimensions");
    assert!(x.len() >= n, "input vector shorter than band size");

    let mut pulses = vec![0i32; n + 3];

    exp_rotation(x, n, 1, b, k, spread);

    let yy = op_pvq_search(x, &mut pulses, n, k, arch);

    let total_pulses = usize::try_from(k).expect("pulse count must fit in usize");
    encode_pulses(&pulses[..n], n, total_pulses, enc);

    if resynth {
        normalise_residual(&pulses[..n], x, n, yy, gain);
        exp_rotation(x, n, -1, b, k, spread);
    }

    extract_collapse_mask(&pulses[..n], n, b)
}

/// Algebraic pulse decoder mirroring `alg_unquant()`.
pub(crate) fn alg_unquant(
    x: &mut [OpusVal16],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    dec: &mut EcDec<'_>,
    gain: OpusVal32,
) -> u32 {
    assert!(k > 0, "alg_unquant requires at least one pulse");
    assert!(n > 1, "alg_unquant requires at least two dimensions");
    assert!(x.len() >= n, "input vector shorter than band size");

    let mut pulses = vec![0i32; n];
    let total_pulses = usize::try_from(k).expect("pulse count must fit in usize");
    let ryy = decode_pulses(&mut pulses, n, total_pulses, dec);
    normalise_residual(&pulses, x, n, ryy, gain);
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(&pulses, n, b)
}

/// Renormalises a vector to unit gain, matching `renormalise_vector()`.
pub(crate) fn renormalise_vector(x: &mut [OpusVal16], n: usize, gain: OpusVal32, arch: i32) {
    assert!(x.len() >= n, "input vector shorter than band size");
    let len = n.min(x.len());
    let slice = &mut x[..len];

    let energy = EPSILON + celt_inner_prod(slice, slice);
    let scale = celt_rsqrt_norm(energy) * gain;

    if arch != 0 {
        let _ = arch;
    }

    for sample in slice.iter_mut() {
        *sample *= scale;
    }
}

/// Fixed-point version of `renormalise_vector()`.
#[cfg(feature = "fixed_point")]
pub(crate) fn renormalise_vector_fixed(
    x: &mut [FixedOpusVal16],
    n: usize,
    gain: FixedOpusVal32,
    _arch: i32,
) {
    use crate::celt::pitch::celt_inner_prod_fixed;
    
    assert!(x.len() >= n, "input vector shorter than band size");
    let len = n.min(x.len());
    
    // Compute energy with EPSILON offset (matches C code: E = EPSILON + celt_inner_prod(X, X, N, arch))
    let e = 1i32 + celt_inner_prod_fixed(&x[..len], &x[..len]);
    
    let k = celt_ilog2(e) >> 1;
    let t = vshr32(e, 2 * (k - 7));
    let g = mult32_32_q31(i32::from(celt_rsqrt_norm_fixed(t)), gain) as i16;
    
    for sample in x.iter_mut().take(len) {
        *sample = extract16(pshr32(mult16_16(g, *sample), (k + 1) as u32));
    }
}

/// Computes the stereo intensity angle mirroring `stereo_itheta()`.
pub(crate) fn stereo_itheta(
    x: &[OpusVal16],
    y: &[OpusVal16],
    stereo: bool,
    n: usize,
    arch: i32,
) -> i32 {
    assert!(x.len() >= n, "mid channel shorter than requested length");
    assert!(y.len() >= n, "side channel shorter than requested length");

    let len = n.min(x.len()).min(y.len());
    let mut emid = EPSILON;
    let mut eside = EPSILON;

    if stereo {
        for i in 0..len {
            let m = 0.5 * (x[i] + y[i]);
            let s = 0.5 * (x[i] - y[i]);
            emid += m * m;
            eside += s * s;
        }
    } else {
        let mid = &x[..len];
        let side = &y[..len];
        emid += celt_inner_prod(mid, mid);
        eside += celt_inner_prod(side, side);
    }

    if arch != 0 {
        let _ = arch;
    }

    let mid = celt_sqrt(emid);
    let side = celt_sqrt(eside);
    let angle = fast_atan2f(side, mid);

    floorf(0.5 + 16_384.0 * FRAC_2_PI * angle) as i32
}

/// Fixed-point version of `stereo_itheta()`.
#[cfg(feature = "fixed_point")]
pub(crate) fn stereo_itheta_fixed(
    x: &[FixedOpusVal16],
    y: &[FixedOpusVal16],
    stereo: bool,
    n: usize,
    _arch: i32,
) -> i32 {
    assert!(x.len() >= n, "mid channel shorter than requested length");
    assert!(y.len() >= n, "side channel shorter than requested length");

    let len = n.min(x.len()).min(y.len());
    let mut emid: FixedOpusVal32 = 1; // EPSILON in fixed-point
    let mut eside: FixedOpusVal32 = 1;

    if stereo {
        for i in 0..len {
            // m = (x[i] + y[i]) / 2 => SHR16(x[i] + y[i], 1)
            // s = (x[i] - y[i]) / 2 => SHR16(x[i] - y[i], 1)
            let m = shr16(x[i].wrapping_add(y[i]), 1);
            let s = shr16(x[i].wrapping_sub(y[i]), 1);
            emid = emid.wrapping_add((i32::from(m) * i32::from(m)) >> 15);
            eside = eside.wrapping_add((i32::from(s) * i32::from(s)) >> 15);
        }
    } else {
        for i in 0..len {
            let xval = i32::from(x[i]);
            let yval = i32::from(y[i]);
            emid = emid.wrapping_add((xval * xval) >> 15);
            eside = eside.wrapping_add((yval * yval) >> 15);
        }
    }

    let mid = celt_sqrt_fixed(emid) as i16;
    let side = celt_sqrt_fixed(eside) as i16;
    
    // 0.63662 = 2/pi in Q15 format
    const TWO_OVER_PI_Q15: i16 = 20_861; // floor(0.63662 * 32768 + 0.5)
    let itheta = mult16_16_q15(TWO_OVER_PI_Q15, celt_atan2p(side, mid));
    
    i32::from(itheta)
}

/// Mirrors `extract_collapse_mask()` from `celt/vq.c`.
///
/// The helper inspects the quantised PVQ pulses and determines which of the
/// folded bands received any energy. The mask is later used to decide whether
/// a band "collapsed" during quantisation and therefore needs spectral
/// spreading in the decoder.
#[must_use]
pub(crate) fn extract_collapse_mask(pulses: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }

    if n == 0 {
        return 0;
    }

    debug_assert!(pulses.len() >= n, "pulse buffer shorter than band size");

    let n0 = celt_udiv(n as u32, b as u32) as usize;
    debug_assert!(n0 > 0, "sub-band width must be non-zero");

    let mut collapse_mask = 0u32;
    for band in 0..b {
        let start = band * n0;
        let end = (start + n0).min(pulses.len());
        let mut accumulator = 0;
        for &value in &pulses[start..end] {
            accumulator |= value;
        }
        if accumulator != 0 {
            collapse_mask |= 1 << band;
        }
    }

    collapse_mask
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{
        SPREAD_NORMAL, alg_quant, alg_unquant, exp_rotation, extract_collapse_mask,
        normalise_residual, renormalise_vector, stereo_itheta,
    };
    #[cfg(feature = "fixed_point")]
    use super::{normalise_residual_fixed, renormalise_vector_fixed, stereo_itheta_fixed};
    use crate::celt::entdec::EcDec;
    use crate::celt::entenc::EcEnc;

    fn seed_samples(len: usize) -> Vec<f32> {
        let mut seed = 0x1234_5678u32;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let sample = ((seed >> 16) & 0x7fff) as i32 - 16_384;
            out.push(sample as f32);
        }
        out
    }

    fn snr_db(original: &[f32], processed: &[f32]) -> f64 {
        let mut err = 0.0;
        let mut ener = 0.0;
        for (&orig, &proc) in original.iter().zip(processed.iter()) {
            let diff = f64::from(orig - proc);
            err += diff * diff;
            ener += f64::from(orig) * f64::from(orig);
        }
        if err == 0.0 {
            return f64::INFINITY;
        }
        20.0 * (ener / err).log10()
    }

    fn rotation_case(len: usize, k: i32) {
        let baseline = seed_samples(len);
        let mut rotated = baseline.clone();

        exp_rotation(&mut rotated, len, 1, 1, k, SPREAD_NORMAL);
        let forward_snr = snr_db(&baseline, &rotated);

        exp_rotation(&mut rotated, len, -1, 1, k, SPREAD_NORMAL);
        let inverse_snr = snr_db(&baseline, &rotated);

        assert!(inverse_snr > 60.0, "inverse SNR too low: {inverse_snr}");
        assert!(
            forward_snr < 20.0,
            "forward SNR unexpectedly high: {forward_snr}"
        );
    }

    #[test]
    fn rotation_matches_reference_behaviour() {
        for &(len, k) in &[(15, 3), (23, 5), (50, 3), (80, 1)] {
            rotation_case(len, k);
        }
    }

    #[test]
    fn residual_normalisation_scales_by_gain() {
        let pulses = [2, -1, 0, 3];
        let mut output = [0.0f32; 4];
        let ryy = 25.0;
        let gain = 0.5;

        normalise_residual(&pulses, &mut output, pulses.len(), ryy, gain);

        let expected_scale = gain * (1.0 / ryy.sqrt());
        for (value, &pulse) in output.iter().zip(&pulses) {
            let expected = expected_scale * pulse as f32;
            assert!(
                (value - expected).abs() <= 1e-6,
                "expected {expected}, found {value}"
            );
        }
    }

    #[test]
    fn collapse_mask_sets_bits_for_active_bands() {
        let pulses = [0, 0, 1, 0, 0, 0, 2, 3];
        let mask = extract_collapse_mask(&pulses, pulses.len(), 4);
        assert_eq!(mask, 0b1010);
    }

    #[test]
    fn collapse_mask_for_single_band_is_one() {
        let pulses = [0, 0, 0, 0];
        assert_eq!(extract_collapse_mask(&pulses, pulses.len(), 1), 1);
    }

    #[test]
    fn op_pvq_search_distributes_requested_pulses() {
        let mut coeffs = [0.6f32, -0.4, 0.2, 0.1];
        let mut pulses = vec![0i32; coeffs.len()];
        let len = coeffs.len();
        let energy = super::op_pvq_search(&mut coeffs, &mut pulses, len, 5, 0);

        let total: i32 = pulses.iter().map(|&p| p.abs()).sum();
        assert_eq!(total, 5);
        assert!(energy > 0.0);
    }

    #[test]
    fn alg_quant_and_unquant_round_trip() {
        let coeffs = seed_samples(6);
        let mut encoded = coeffs.clone();
        let gain = 1.0f32;
        let n = coeffs.len();
        let k = 5;
        let spread = SPREAD_NORMAL;
        let b = 2usize;
        let arch = 0;

        let mut buffer = vec![0u8; 64];
        let mask;
        {
            let mut enc = EcEnc::new(&mut buffer);
            mask = alg_quant(&mut encoded, n, k, spread, b, &mut enc, gain, true, arch);
            enc.enc_done();
        }

        let mut decoded = coeffs.clone();
        let mut dec = EcDec::new(&mut buffer);
        let mask_dec = alg_unquant(&mut decoded, n, k, spread, b, &mut dec, gain);

        assert_eq!(mask, mask_dec);
        let tolerance = 1e-6;
        for (a, b) in encoded.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < tolerance);
        }
    }

    #[test]
    fn renormalise_vector_matches_expected_gain() {
        let mut data = seed_samples(8);
        let gain = 0.75f32;
        let len = data.len();
        renormalise_vector(&mut data, len, gain, 0);
        let energy: f32 = data.iter().map(|&v| v * v).sum();
        assert!((energy - gain * gain).abs() < 1e-5);
    }

    #[test]
    fn stereo_itheta_returns_expected_half_pi_value() {
        let x = [0.0f32; 8];
        let y = [1.0f32; 8];
        let angle = stereo_itheta(&x, &y, false, x.len(), 0);
        assert!((angle - 16_384).abs() <= 1);
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn fixed_residual_normalisation_preserves_signs() {
        let pulses = [3, -2, 1, 0];
        let mut output = [0i16; 4];
        let ryy = 1 << 14;
        let gain = 1 << 14;

        normalise_residual_fixed(&pulses, &mut output, pulses.len(), ryy, gain);

        for (value, &pulse) in output.iter().zip(&pulses) {
            if pulse == 0 {
                assert_eq!(*value, 0);
            } else {
                let sign = i32::from(value.signum());
                assert!(sign == 0 || sign == pulse.signum());
            }
        }
    }

    #[test]
    #[cfg(feature = "fixed_point")]
    fn renormalise_vector_fixed_runs_without_panic() {
        let mut x = vec![1000i16, 2000, 1500, -1000];
        let n = x.len();
        let gain = 16384i32; // 0.5 gain in Q15
        
        // Just ensure the function runs without panic
        renormalise_vector_fixed(&mut x, n, gain, 0);
        
        // The function should complete successfully
        assert!(true);
    }

    #[test]
    #[cfg(feature = "fixed_point")]
    fn stereo_itheta_fixed_returns_valid_angle() {
        let x = vec![100i16, 200, 150, 100];
        let y = vec![50i16, 100, 75, 50];
        
        let angle = stereo_itheta_fixed(&x, &y, false, x.len(), 0);
        
        // Angle should be in valid range [0, 16384] representing [0, pi/2]
        assert!(angle >= 0 && angle <= 16384, "angle {} out of range", angle);
    }
}
