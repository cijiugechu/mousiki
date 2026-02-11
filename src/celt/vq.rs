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
use crate::celt::fixed_ops::{
    add32, extract16, mac16_16, mult16_16, mult16_16_q15, mult16_32_q16, mult32_32_q31, pshr32,
    shr32, vshr32,
};
#[cfg(feature = "fixed_point")]
use crate::celt::math::celt_ilog2;
#[cfg(feature = "fixed_point")]
use crate::celt::entcode::ec_ilog;
#[cfg(feature = "fixed_point")]
use crate::celt::math_fixed::{celt_cos_norm as celt_cos_norm_fixed, celt_rcp as celt_rcp_fixed, celt_rsqrt_norm as celt_rsqrt_norm_fixed};
#[cfg(feature = "fixed_point")]
use crate::celt::types::{FixedOpusVal16, FixedOpusVal32};
#[cfg(feature = "fixed_point")]
use crate::celt::fixed_arch::{EPSILON as FIXED_EPSILON, Q15_ONE as Q15_ONE_FIXED};
#[cfg(feature = "fixed_point")]
use crate::celt::pitch::celt_inner_prod_fixed;

/// Spread decisions mirrored from `celt/bands.h`.
pub(crate) const SPREAD_NONE: i32 = 0;
pub(crate) const SPREAD_LIGHT: i32 = 1;
pub(crate) const SPREAD_NORMAL: i32 = 2;
pub(crate) const SPREAD_AGGRESSIVE: i32 = 3;

const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
const Q15_ONE: OpusVal16 = 1.0;
const EPSILON: OpusVal32 = 1e-15;

#[cfg(feature = "fixed_point")]
#[inline]
fn add16(a: FixedOpusVal16, b: FixedOpusVal16) -> FixedOpusVal16 {
    a.wrapping_add(b)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn sub16(a: FixedOpusVal16, b: FixedOpusVal16) -> FixedOpusVal16 {
    a.wrapping_sub(b)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn neg16(a: FixedOpusVal16) -> FixedOpusVal16 {
    a.wrapping_neg()
}

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

#[cfg(feature = "fixed_point")]
fn exp_rotation1_fixed(
    x: &mut [FixedOpusVal16],
    stride: usize,
    c: FixedOpusVal16,
    s: FixedOpusVal16,
) {
    let len = x.len();
    if stride == 0 || len <= stride {
        return;
    }

    let ms = neg16(s);

    for i in 0..(len - stride) {
        let x1 = x[i];
        let x2 = x[i + stride];
        let acc1 = mac16_16(mult16_16(c, x2), s, x1);
        let acc2 = mac16_16(mult16_16(c, x1), ms, x2);
        x[i + stride] = extract16(pshr32(acc1, 15));
        x[i] = extract16(pshr32(acc2, 15));
    }

    if len > 2 * stride {
        let limit = len - 2 * stride - 1;
        for i in (0..=limit).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            let acc1 = mac16_16(mult16_16(c, x2), s, x1);
            let acc2 = mac16_16(mult16_16(c, x1), ms, x2);
            x[i + stride] = extract16(pshr32(acc1, 15));
            x[i] = extract16(pshr32(acc2, 15));
        }
    }
}

#[cfg(feature = "fixed_point")]
pub(crate) fn exp_rotation_fixed(
    x: &mut [FixedOpusVal16],
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
    let denom = slice_len as i32 + factor * k;
    let numer = mult16_16(Q15_ONE_FIXED, slice_len as i16);
    let gain = extract16(mult32_32_q31(numer, celt_rcp_fixed(denom)));
    let theta = (mult16_16_q15(gain, gain) >> 1) as i16;
    let c = celt_cos_norm_fixed(i32::from(theta));
    let s = celt_cos_norm_fixed(i32::from(sub16(Q15_ONE_FIXED, theta)));

    let mut stride2 = 0usize;
    if slice_len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < slice_len {
            stride2 += 1;
        }
    }

    let len_div = celt_udiv(slice_len as u32, stride as u32) as usize;
    if len_div == 0 {
        return;
    }

    for band in 0..stride {
        let start = band * len_div;
        let end = start + len_div;
        let band_slice = &mut x[start..end];
        if dir < 0 {
            if stride2 > 0 {
                exp_rotation1_fixed(band_slice, stride2, s, c);
            }
            exp_rotation1_fixed(band_slice, 1, c, s);
        } else {
            exp_rotation1_fixed(band_slice, 1, c, neg16(s));
            if stride2 > 0 {
                exp_rotation1_fixed(band_slice, stride2, s, neg16(c));
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

#[cfg(feature = "fixed_point")]
pub(crate) fn op_pvq_search_fixed(
    x: &mut [FixedOpusVal16],
    pulses: &mut [OpusInt32],
    n: usize,
    k: i32,
    _arch: i32,
) -> FixedOpusVal16 {
    assert!(n > 0, "vector dimension must be positive");
    assert!(k >= 0, "pulse count must be non-negative");
    assert!(x.len() >= n, "coefficient buffer shorter than band size");
    assert!(pulses.len() >= n, "pulse buffer shorter than band size");

    let mut y = vec![0i16; n];
    let mut signx = vec![0i32; n];

    for j in 0..n {
        let value = x[j];
        signx[j] = i32::from(value < 0);
        x[j] = if value < 0 { value.wrapping_neg() } else { value };
        pulses[j] = 0;
        y[j] = 0;
    }

    let mut xy: i32 = 0;
    let mut yy: i16 = 0;
    let mut pulses_left = k;

    if k > ((n as i32) >> 1) {
        let mut sum: i32 = 0;
        for &sample in x.iter().take(n) {
            sum = sum.wrapping_add(sample as i32);
        }

        if sum <= k {
            if n > 0 {
                x[0] = 16_384;
                for coeff in x.iter_mut().take(n).skip(1) {
                    *coeff = 0;
                }
            }
            sum = 16_384;
        }

        let rcp = extract16(mult16_32_q16(k as i16, celt_rcp_fixed(sum)));
        for j in 0..n {
            let iy = mult16_16_q15(x[j], rcp) as i32;
            pulses[j] = iy;
            let yj = iy as i16;
            y[j] = yj;
            yy = extract16(mac16_16(i32::from(yy), yj, yj));
            xy = mac16_16(xy, x[j], yj);
            y[j] = y[j].wrapping_shl(1);
            pulses_left -= iy;
        }
    }

    debug_assert!(pulses_left >= 0, "pulse allocation exceeded target count");
    if pulses_left < 0 {
        pulses_left = 0;
    }

    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as i16;
        yy = extract16(mac16_16(i32::from(yy), tmp, tmp));
        yy = extract16(mac16_16(i32::from(yy), tmp, y[0]));
        pulses[0] += pulses_left;
        pulses_left = 0;
    }

    for i in 0..pulses_left {
        let rshift = 1 + celt_ilog2(k - pulses_left + i + 1);
        yy = add16(yy, 1);

        let mut best_id = 0usize;
        let mut rxy = extract16(shr32(add32(xy, i32::from(x[0])), rshift as u32));
        let mut ryy = add16(yy, y[0]);
        let mut rxy_sq = mult16_16_q15(rxy, rxy);
        let mut best_den = ryy;
        let mut best_num: i32 = i32::from(rxy_sq);

        for j in 1..n {
            rxy = extract16(shr32(add32(xy, i32::from(x[j])), rshift as u32));
            ryy = add16(yy, y[j]);
            rxy_sq = mult16_16_q15(rxy, rxy);
            let left = mult16_16(best_den, rxy_sq);
            let right = i32::from(ryy).wrapping_mul(best_num);
            if left > right {
                best_den = ryy;
                best_num = i32::from(rxy_sq);
                best_id = j;
            }
        }

        xy = add32(xy, i32::from(x[best_id]));
        yy = add16(yy, y[best_id]);
        y[best_id] = y[best_id].wrapping_add(2);
        pulses[best_id] += 1;
    }

    for j in 0..n {
        let sign = -signx[j];
        pulses[j] = (pulses[j] ^ sign) + signx[j];
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

#[cfg(feature = "fixed_point")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn alg_quant_fixed(
    x: &mut [FixedOpusVal16],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    enc: &mut EcEnc<'_>,
    gain: FixedOpusVal32,
    resynth: bool,
    arch: i32,
) -> u32 {
    assert!(k > 0, "alg_quant requires at least one pulse");
    assert!(n > 1, "alg_quant requires at least two dimensions");
    assert!(x.len() >= n, "input vector shorter than band size");

    let mut pulses = vec![0i32; n + 3];

    exp_rotation_fixed(x, n, 1, b, k, spread);

    let yy = op_pvq_search_fixed(x, &mut pulses, n, k, arch);

    let total_pulses = usize::try_from(k).expect("pulse count must fit in usize");
    encode_pulses(&pulses[..n], n, total_pulses, enc);

    if resynth {
        normalise_residual_fixed(&pulses[..n], x, n, i32::from(yy), gain);
        exp_rotation_fixed(x, n, -1, b, k, spread);
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

#[cfg(feature = "fixed_point")]
pub(crate) fn alg_unquant_fixed(
    x: &mut [FixedOpusVal16],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    dec: &mut EcDec<'_>,
    gain: FixedOpusVal32,
) -> u32 {
    assert!(k > 0, "alg_unquant requires at least one pulse");
    assert!(n > 1, "alg_unquant requires at least two dimensions");
    assert!(x.len() >= n, "input vector shorter than band size");

    let mut pulses = vec![0i32; n];
    let total_pulses = usize::try_from(k).expect("pulse count must fit in usize");
    let _ = decode_pulses(&mut pulses, n, total_pulses, dec);

    let mut ryy: i32 = 0;
    for &pulse in pulses.iter().take(n) {
        ryy = mac16_16(ryy, pulse as i16, pulse as i16);
    }

    normalise_residual_fixed(&pulses, x, n, ryy, gain);
    exp_rotation_fixed(x, n, -1, b, k, spread);
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

#[cfg(feature = "fixed_point")]
pub(crate) fn renormalise_vector_fixed(
    x: &mut [FixedOpusVal16],
    n: usize,
    gain: FixedOpusVal32,
    arch: i32,
) {
    assert!(x.len() >= n, "input vector shorter than band size");
    let len = n.min(x.len());
    let slice = &mut x[..len];

    let mut energy = i32::from(FIXED_EPSILON);
    energy = energy.wrapping_add(celt_inner_prod_fixed(slice, slice));
    let k = (ec_ilog(energy as u32) - 1) >> 1;
    let t = vshr32(energy, 2 * (k - 7));
    let g = mult32_32_q31(i32::from(celt_rsqrt_norm_fixed(t)), gain) as i16;

    if arch != 0 {
        let _ = arch;
    }

    for sample in slice.iter_mut() {
        let scaled = mult16_16(g, *sample);
        *sample = extract16(pshr32(scaled, (k + 1) as u32));
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

    #[cfg(feature = "fixed_point")]
    mod fixed_tests {
        use crate::celt::vq::{
            SPREAD_AGGRESSIVE, SPREAD_LIGHT, SPREAD_NONE, SPREAD_NORMAL, alg_quant_fixed,
            alg_unquant_fixed, exp_rotation_fixed, op_pvq_search_fixed, renormalise_vector_fixed,
        };
        use crate::celt::entdec::EcDec;
        use crate::celt::entenc::EcEnc;
        use crate::celt::fixed_arch::Q31_ONE;
        use crate::celt::fixed_ops::{extract16, mult16_16, mult16_16_q15, mult32_32_q31};
        use crate::celt::math_fixed::{celt_cos_norm as celt_cos_norm_fixed, celt_rcp as celt_rcp_fixed};
        use crate::celt::fixed_arch::Q15_ONE as Q15_ONE_FIXED;
        use crate::celt::vq::normalise_residual_fixed;

        fn assert_i16_slice(label: &str, got: &[i16], expected: &[i16]) {
            assert_eq!(got.len(), expected.len(), "{label} length mismatch");
            for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(g, e, "{label}[{idx}] mismatch: {g} != {e}");
            }
        }

        fn assert_i32_slice(label: &str, got: &[i32], expected: &[i32]) {
            assert_eq!(got.len(), expected.len(), "{label} length mismatch");
            for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(g, e, "{label}[{idx}] mismatch: {g} != {e}");
            }
        }

        #[test]
        fn fixed_normalise_residual_matches_reference_case() {
            let pulses = [1, -1, 1, 0, 0, 0, 0, 0];
            let mut output = [0i16; 8];
            normalise_residual_fixed(&pulses, &mut output, pulses.len(), 3, Q31_ONE);
            let expected = [9459, -9458, 9459, 0, 0, 0, 0, 0];
            assert_i16_slice("normalise_case1", &output, &expected);
        }

        #[test]
        fn fixed_exp_rotation_coefficients_match_reference_case() {
            let len = 8i32;
            let k = 3i32;
            let factor = 10i32;
            let denom = len + factor * k;
            let numer = mult16_16(Q15_ONE_FIXED, len as i16);
            let gain = extract16(mult32_32_q31(numer, celt_rcp_fixed(denom)));
            let theta = (mult16_16_q15(gain, gain) >> 1) as i16;
            let c = celt_cos_norm_fixed(i32::from(theta));
            let s = celt_cos_norm_fixed(i32::from(Q15_ONE_FIXED.wrapping_sub(theta)));

            assert_eq!(gain, 6898);
            assert_eq!(theta, 726);
            assert_eq!(c, 32748);
            assert_eq!(s, 1142);
        }

        #[test]
        fn fixed_exp_rotation_forward_matches_reference_case() {
            let mut data = [12000, -8000, 6000, -4000, 2000, 0, -1000, 500];
            let len = data.len();
            exp_rotation_fixed(&mut data, len, 1, 2, 3, SPREAD_NORMAL);
            let expected = [11429, -8383, 6429, -4217, 1993, -209, -975, 534];
            assert_i16_slice("exp_rotation_forward_case1", &data, &expected);
        }

        #[test]
        fn fixed_exp_rotation_inverse_matches_reference_case() {
            let mut data = [9459, -9458, 9459, 0, 0, 0, 0, 0];
            let len = data.len();
            exp_rotation_fixed(&mut data, len, -1, 2, 3, SPREAD_NORMAL);
            let expected = [10117, -9412, 8794, 318, 0, 0, 0, 0];
            assert_i16_slice("exp_rotation_case1", &data, &expected);
        }

        #[test]
        fn fixed_renormalise_vector_matches_reference_cases() {
            let mut zeros = [0i16; 8];
            let expected_zeros = [0i16; 8];
            let len = zeros.len();
            renormalise_vector_fixed(&mut zeros, len, Q31_ONE, 0);
            assert_i16_slice("renorm_zero", &zeros, &expected_zeros);

            let mut mixed = [1000, -2000, 3000, -4000, 500, -600, 700, -800];
            let expected_mixed = [2908, -5816, 8724, -11632, 1454, -1745, 2036, -2326];
            let len = mixed.len();
            renormalise_vector_fixed(&mut mixed, len, Q31_ONE, 0);
            assert_i16_slice("renorm_mixed", &mixed, &expected_mixed);

            let mut large = [30000, -30000, 20000, -10000];
            let expected_large = [10248, -10248, 6832, -3416];
            let len = large.len();
            renormalise_vector_fixed(&mut large, len, Q31_ONE, 0);
            assert_i16_slice("renorm_large", &large, &expected_large);

            let mut half_gain = [30000, -30000, 20000, -10000];
            let expected_half = [5124, -5124, 3416, -1708];
            let len = half_gain.len();
            renormalise_vector_fixed(&mut half_gain, len, 1_073_741_824, 0);
            assert_i16_slice("renorm_gain_half", &half_gain, &expected_half);
        }

        #[test]
        fn fixed_alg_quant_and_unquant_match_reference_cases() {
            {
                let mut buffer = [0u8; 256];
                let mut x = [12000, -8000, 6000, -4000, 2000, 0, -1000, 500];
                let expected = [10117, -9412, 8794, 318, 0, 0, 0, 0];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        3,
                        SPREAD_NORMAL,
                        2,
                        &mut enc,
                        Q31_ONE,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 8];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 3, SPREAD_NORMAL, 2, &mut dec, Q31_ONE);
                assert_eq!(mask, 1);
                assert_eq!(umask, 1);
                assert_i16_slice("alg_case1_q", &x, &expected);
                assert_i16_slice("alg_case1_u", &xu, &expected);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [12000, -8000, 6000, -4000, 2000, 0, -1000, 500];
                let expected_q = [11429, 8383, 6429, 4217, 1993, 209, 975, 534];
                let expected_u = [10117, -9412, 8794, 318, 0, 0, 0, 0];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        3,
                        SPREAD_NORMAL,
                        2,
                        &mut enc,
                        Q31_ONE,
                        false,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 8];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 3, SPREAD_NORMAL, 2, &mut dec, Q31_ONE);
                assert_eq!(mask, 1);
                assert_eq!(umask, 1);
                assert_i16_slice("alg_case1_resynth0_q", &x, &expected_q);
                assert_i16_slice("alg_case1_resynth0_u", &xu, &expected_u);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [0, 16000, -16000, 8000, -8000, 4000, 0, 2000, -2000, 0];
                let expected = [0, 9459, -9458, 9459, 0, 0, 0, 0, 0, 0];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        3,
                        SPREAD_NONE,
                        5,
                        &mut enc,
                        Q31_ONE,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 10];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 3, SPREAD_NONE, 5, &mut dec, Q31_ONE);
                assert_eq!(mask, 3);
                assert_eq!(umask, 3);
                assert_i16_slice("alg_case2_q", &x, &expected);
                assert_i16_slice("alg_case2_u", &xu, &expected);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [8000, -4000, 2000, -1000];
                let expected = [11585, -11584, 0, 0];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        2,
                        SPREAD_NORMAL,
                        1,
                        &mut enc,
                        Q31_ONE,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 4];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 2, SPREAD_NORMAL, 1, &mut dec, Q31_ONE);
                assert_eq!(mask, 1);
                assert_eq!(umask, 1);
                assert_i16_slice("alg_case3_q", &x, &expected);
                assert_i16_slice("alg_case3_u", &xu, &expected);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [12000, -8000];
                let expected = [16383, 0];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        1,
                        SPREAD_AGGRESSIVE,
                        1,
                        &mut enc,
                        Q31_ONE,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 2];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 1, SPREAD_AGGRESSIVE, 1, &mut dec, Q31_ONE);
                assert_eq!(mask, 1);
                assert_eq!(umask, 1);
                assert_i16_slice("alg_case4_q", &x, &expected);
                assert_i16_slice("alg_case4_u", &xu, &expected);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [
                    5000, -4000, 3000, -2000, 1000, -500, 250, -125, 6000, -5000, 4000, -3000,
                    2000, -1000, 500, -250,
                ];
                let expected = [
                    9376, 1011, 108, -501, -51, -57, 507, 28, 10390, -8309, -901, -611, 449, -5,
                    538, -483,
                ];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        3,
                        SPREAD_LIGHT,
                        2,
                        &mut enc,
                        Q31_ONE,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 16];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 3, SPREAD_LIGHT, 2, &mut dec, Q31_ONE);
                assert_eq!(mask, 3);
                assert_eq!(umask, 3);
                assert_i16_slice("alg_case5_q", &x, &expected);
                assert_i16_slice("alg_case5_u", &xu, &expected);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [
                    0, 10000, -9000, 8000, -7000, 6000, -5000, 4000, -3000, 2000, -1000, 500,
                    -250, 125, -60, 30,
                ];
                let expected = [
                    -1153, 6368, -3849, 2819, 157, 1165, -687, 805, 0, 0, 0, 0, 0, 0, 0, 0,
                ];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        3,
                        SPREAD_AGGRESSIVE,
                        2,
                        &mut enc,
                        1_073_741_824,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 16];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask =
                    alg_unquant_fixed(&mut xu, n, 3, SPREAD_AGGRESSIVE, 2, &mut dec, 1_073_741_824);
                assert_eq!(mask, 1);
                assert_eq!(umask, 1);
                assert_i16_slice("alg_case6_q", &x, &expected);
                assert_i16_slice("alg_case6_u", &xu, &expected);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [16000, 8000, 4000, 2000, 1000, 500, 250, 125];
                let expected = [13135, 8757, 4378, 0, 0, 0, 0, 0];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        6,
                        SPREAD_NONE,
                        2,
                        &mut enc,
                        Q31_ONE,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 8];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 6, SPREAD_NONE, 2, &mut dec, Q31_ONE);
                assert_eq!(mask, 1);
                assert_eq!(umask, 1);
                assert_i16_slice("alg_case7_q", &x, &expected);
                assert_i16_slice("alg_case7_u", &xu, &expected);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [6000, -6000, 6000, -6000, 6000, -6000, 6000, -6000];
                let expected = [5792, -5792, 5792, -5792, 5792, -5792, 5792, -5792];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        8,
                        SPREAD_NONE,
                        4,
                        &mut enc,
                        Q31_ONE,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 8];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 8, SPREAD_NONE, 4, &mut dec, Q31_ONE);
                assert_eq!(mask, 15);
                assert_eq!(umask, 15);
                assert_i16_slice("alg_case8_q", &x, &expected);
                assert_i16_slice("alg_case8_u", &xu, &expected);
            }

            {
                let mut buffer = [0u8; 256];
                let mut x = [0, 0, 0, 0, 0, 0, 12000, -8000];
                let expected = [0, 0, 0, 0, 0, 0, 14654, -7327];
                let mask;
                {
                    let n = x.len();
                    let mut enc = EcEnc::new(&mut buffer);
                    mask = alg_quant_fixed(
                        &mut x,
                        n,
                        3,
                        SPREAD_NONE,
                        4,
                        &mut enc,
                        Q31_ONE,
                        true,
                        0,
                    );
                    enc.enc_done();
                }
                let mut xu = [0i16; 8];
                let n = xu.len();
                let mut dec = EcDec::new(&mut buffer);
                let umask = alg_unquant_fixed(&mut xu, n, 3, SPREAD_NONE, 4, &mut dec, Q31_ONE);
                assert_eq!(mask, 8);
                assert_eq!(umask, 8);
                assert_i16_slice("alg_case9_q", &x, &expected);
                assert_i16_slice("alg_case9_u", &xu, &expected);
            }
        }

        #[test]
        fn fixed_op_pvq_search_matches_reference_cases() {
            {
                let mut x = [0i16; 8];
                let mut pulses = [0i32; 8];
                let yy = op_pvq_search_fixed(&mut x, &mut pulses, 8, 6, 0);
                let expected_x = [16384, 0, 0, 0, 0, 0, 0, 0];
                let expected_pulses = [6, 0, 0, 0, 0, 0, 0, 0];
                assert_eq!(yy, 36);
                assert_i16_slice("op_silence_x", &x, &expected_x);
                assert_i32_slice("op_silence_iy", &pulses, &expected_pulses);
            }

            {
                let mut x = [0i16; 2];
                let mut pulses = [0i32; 2];
                let yy = op_pvq_search_fixed(&mut x, &mut pulses, 2, 10, 0);
                let expected_x = [16384, 0];
                let expected_pulses = [10, 0];
                assert_eq!(yy, 100);
                assert_i16_slice("op_bigk_x", &x, &expected_x);
                assert_i32_slice("op_bigk_iy", &pulses, &expected_pulses);
            }

            {
                let mut x = [100, -50, 25, -12, 6, -3, 2, -1];
                let mut pulses = [0i32; 8];
                let yy = op_pvq_search_fixed(&mut x, &mut pulses, 8, 6, 0);
                let expected_x = [100, 50, 25, 12, 6, 3, 2, 1];
                let expected_pulses = [5, -1, 0, 0, 0, 0, 0, 0];
                assert_eq!(yy, 26);
                assert_i16_slice("op_small_x", &x, &expected_x);
                assert_i32_slice("op_small_iy", &pulses, &expected_pulses);
            }

            {
                let mut x = [11429, -8383, 6429, -4217, 1993, -209, -975, 534];
                let mut pulses = [0i32; 8];
                let yy = op_pvq_search_fixed(&mut x, &mut pulses, 8, 3, 0);
                let expected_x = [11429, 8383, 6429, 4217, 1993, 209, 975, 534];
                let expected_pulses = [1, -1, 1, 0, 0, 0, 0, 0];
                assert_eq!(yy, 3);
                assert_i16_slice("op_case1_x", &x, &expected_x);
                assert_i32_slice("op_case1_iy", &pulses, &expected_pulses);
            }
        }
    }
}
