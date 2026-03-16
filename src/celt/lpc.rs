#![allow(dead_code)]

//! Linear prediction helpers mirrored from `celt/celt_lpc.c`.
//!
//! The Levinson-Durbin recursion exposed as `_celt_lpc()` in the C
//! implementation has minimal dependencies on the rest of the encoder.  This
//! makes it a convenient building block to port early, as later modules such as
//! the pitch analysis and postfilter reuse it.

use alloc::borrow::Cow;
#[cfg(feature = "fixed_point")]
use alloc::vec;
#[cfg(feature = "fixed_point")]
use alloc::vec::Vec;

use crate::celt::math::{frac_div32, mul_add_f32};
use crate::celt::pitch::celt_pitch_xcorr;
use crate::celt::types::{CeltCoef, OpusVal16, OpusVal32};
#[cfg(feature = "fixed_point")]
use crate::celt::entcode::ec_ilog;
#[cfg(feature = "fixed_point")]
use crate::celt::fixed_arch::SIG_SHIFT;
#[cfg(feature = "fixed_point")]
use crate::celt::fixed_ops::{
    add32, div32, extract16, mac16_16, mult16_16, mult16_16_q15, mult32_32_32, mult32_32_q16,
    mult32_32_q31, pshr32, qconst32, shl32, shr32,
};
#[cfg(feature = "fixed_point")]
use crate::celt::math::celt_ilog2;
#[cfg(feature = "fixed_point")]
use crate::celt::math_fixed::frac_div32 as frac_div32_fixed;
#[cfg(feature = "fixed_point")]
use crate::celt::pitch::celt_pitch_xcorr_fixed;
#[cfg(feature = "fixed_point")]
use crate::celt::types::{FixedCeltCoef, FixedOpusVal16, FixedOpusVal32};
#[cfg(feature = "fixed_point")]
use core::cmp::min;

#[cfg(feature = "fixed_point")]
#[inline]
fn sround16_fixed(x: FixedOpusVal32, shift: u32) -> FixedOpusVal16 {
    let rounded = pshr32(x, shift);
    rounded.clamp(-32_767, 32_767) as FixedOpusVal16
}

#[cfg(feature = "fixed_point")]
#[inline]
fn xcorr_kernel_fixed(x: &[FixedOpusVal16], y: &[FixedOpusVal16], sum: &mut [FixedOpusVal32; 4]) {
    let len = x.len();
    assert!(len >= 3, "xcorr_kernel requires at least three samples");
    assert!(y.len() >= len + 3, "xcorr_kernel needs len + 3 samples from y");

    let mut y_idx = 0usize;
    let mut y0 = y[y_idx];
    y_idx += 1;
    let mut y1 = y[y_idx];
    y_idx += 1;
    let mut y2 = y[y_idx];
    y_idx += 1;
    let mut y3 = 0i16;

    let mut j = 0usize;
    while j + 3 < len {
        let tmp0 = x[j];
        y3 = y[y_idx];
        y_idx += 1;
        sum[0] = mac16_16(sum[0], tmp0, y0);
        sum[1] = mac16_16(sum[1], tmp0, y1);
        sum[2] = mac16_16(sum[2], tmp0, y2);
        sum[3] = mac16_16(sum[3], tmp0, y3);

        let tmp1 = x[j + 1];
        y0 = y[y_idx];
        y_idx += 1;
        sum[0] = mac16_16(sum[0], tmp1, y1);
        sum[1] = mac16_16(sum[1], tmp1, y2);
        sum[2] = mac16_16(sum[2], tmp1, y3);
        sum[3] = mac16_16(sum[3], tmp1, y0);

        let tmp2 = x[j + 2];
        y1 = y[y_idx];
        y_idx += 1;
        sum[0] = mac16_16(sum[0], tmp2, y2);
        sum[1] = mac16_16(sum[1], tmp2, y3);
        sum[2] = mac16_16(sum[2], tmp2, y0);
        sum[3] = mac16_16(sum[3], tmp2, y1);

        let tmp3 = x[j + 3];
        y2 = y[y_idx];
        y_idx += 1;
        sum[0] = mac16_16(sum[0], tmp3, y3);
        sum[1] = mac16_16(sum[1], tmp3, y0);
        sum[2] = mac16_16(sum[2], tmp3, y1);
        sum[3] = mac16_16(sum[3], tmp3, y2);

        j += 4;
    }

    if j < len {
        let tmp = x[j];
        y3 = y[y_idx];
        y_idx += 1;
        sum[0] = mac16_16(sum[0], tmp, y0);
        sum[1] = mac16_16(sum[1], tmp, y1);
        sum[2] = mac16_16(sum[2], tmp, y2);
        sum[3] = mac16_16(sum[3], tmp, y3);
        j += 1;
    }
    if j < len {
        let tmp = x[j];
        y0 = y[y_idx];
        y_idx += 1;
        sum[0] = mac16_16(sum[0], tmp, y1);
        sum[1] = mac16_16(sum[1], tmp, y2);
        sum[2] = mac16_16(sum[2], tmp, y3);
        sum[3] = mac16_16(sum[3], tmp, y0);
        j += 1;
    }
    if j < len {
        let tmp = x[j];
        y1 = y[y_idx];
        sum[0] = mac16_16(sum[0], tmp, y2);
        sum[1] = mac16_16(sum[1], tmp, y3);
        sum[2] = mac16_16(sum[2], tmp, y0);
        sum[3] = mac16_16(sum[3], tmp, y1);
    }
}

/// Computes LPC coefficients from the autocorrelation sequence using the
/// Levinson-Durbin recursion.
///
/// Mirrors the float build of `_celt_lpc()` from `celt/celt_lpc.c`. The caller
/// provides the first `order + 1` autocorrelation entries in `ac`, with
/// `ac[0]` containing the zero-lag value. The resulting predictor coefficients
/// are written to `lpc`.
///
/// The routine aborts early when the prediction error falls below
/// `0.001 * ac[0]`, matching the conservative bailout used by the reference
/// implementation to avoid unstable filters when the signal energy becomes
/// negligible.
pub(crate) fn celt_lpc(lpc: &mut [OpusVal16], ac: &[OpusVal32]) {
    let order = lpc.len();
    assert!(
        ac.len() > order,
        "autocorrelation must provide order + 1 samples"
    );

    for coeff in lpc.iter_mut() {
        *coeff = 0.0;
    }

    if order == 0 {
        return;
    }

    let ac0 = ac[0];
    if ac0 <= 1e-10 {
        return;
    }

    let mut error = ac0;

    for i in 0..order {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr = mul_add_f32(lpc[j], ac[i - j], rr);
        }
        rr += ac[i + 1];

        let reflection = -frac_div32(rr, error);
        lpc[i] = reflection;

        let half = (i + 1) >> 1;
        for j in 0..half {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = mul_add_f32(reflection, tmp2, tmp1);
            lpc[i - 1 - j] = mul_add_f32(reflection, tmp1, tmp2);
        }

        let r2 = reflection * reflection;
        error = mul_add_f32(-r2, error, error);
        if error <= 0.001 * ac0 {
            break;
        }
    }
}

/// Fixed-point LPC solver used by CELT when `fixed_point` is enabled.
///
/// Mirrors the `FIXED_POINT` branch of `_celt_lpc()` from `celt/celt_lpc.c`,
/// including the chirp-based bandwidth expansion that ensures the final
/// coefficients fit into Q12 precision.
#[cfg(feature = "fixed_point")]
pub(crate) fn celt_lpc_fixed(lpc: &mut [FixedOpusVal16], ac: &[FixedOpusVal32]) {
    let order = lpc.len();
    assert!(
        ac.len() > order,
        "autocorrelation must provide order + 1 samples"
    );

    lpc.fill(0);

    if order == 0 || ac[0] == 0 {
        return;
    }

    let mut lpc32 = vec![0i32; order];
    let mut error = ac[0];

    for i in 0..order {
        let mut acc = 0i64;
        for j in 0..i {
            acc += i64::from(lpc32[j]) * i64::from(ac[i - j]);
        }
        let mut rr = (acc >> 31) as i32;
        rr = rr.wrapping_add(shr32(ac[i + 1], 6));
        let r = -frac_div32_fixed(shl32(rr, 6), error);
        lpc32[i] = shr32(r, 6);

        for j in 0..((i + 1) >> 1) {
            let tmp1 = lpc32[j];
            let tmp2 = lpc32[i - 1 - j];
            lpc32[j] = tmp1.wrapping_add(mult32_32_q31(r, tmp2));
            lpc32[i - 1 - j] = tmp2.wrapping_add(mult32_32_q31(r, tmp1));
        }

        error = error.wrapping_sub(mult32_32_q31(mult32_32_q31(r, r), error));
        if error <= shr32(ac[0], 10) {
            break;
        }
    }

    let mut iter = 0;
    let mut idx = 0usize;
    while iter < 10 {
        let mut maxabs = 0i32;
        for (i, &value) in lpc32.iter().enumerate() {
            let absval = value.abs();
            if absval > maxabs {
                maxabs = absval;
                idx = i;
            }
        }

        maxabs = pshr32(maxabs, 13);
        if maxabs <= 32_767 {
            break;
        }

        maxabs = min(maxabs, 163_838);
        let denom = shr32(mult32_32_32(maxabs, (idx + 1) as i32), 2);
        let numer = shl32(maxabs.wrapping_sub(32_767), 14);
        let mut chirp_q16 = qconst32(0.999, 16).wrapping_sub(div32(numer, denom));
        let chirp_minus_one_q16 = chirp_q16.wrapping_sub(65_536);

        for i in 0..order.saturating_sub(1) {
            lpc32[i] = mult32_32_q16(chirp_q16, lpc32[i]);
            chirp_q16 = chirp_q16.wrapping_add(pshr32(
                mult32_32_32(chirp_q16, chirp_minus_one_q16),
                16,
            ));
        }
        if let Some(last) = lpc32.last_mut() {
            *last = mult32_32_q16(chirp_q16, *last);
        }

        iter += 1;
    }

    if iter == 10 {
        lpc.fill(0);
        if let Some(first) = lpc.first_mut() {
            *first = 4096;
        }
    } else {
        for (dst, &value) in lpc.iter_mut().zip(lpc32.iter()) {
            *dst = extract16(pshr32(value, 13));
        }
    }
}

/// Applies a causal FIR filter to the input sequence.
///
/// Mirrors the behaviour of `celt_fir_c()` from `celt/celt_lpc.c` for the float
/// build where the optimisation-specific helpers collapse to straightforward
/// scalar operations. The `x` slice must contain the `ord` samples of history
/// followed by `N` new samples, where `N` matches the length of `y`.
///
/// The implementation intentionally keeps the history layout identical to the C
/// code: `x[ord + i]` corresponds to the current sample while `x[ord + i - 1 - j]`
/// exposes the `j`-th past value.
pub(crate) fn celt_fir(x: &[OpusVal16], num: &[OpusVal16], y: &mut [OpusVal16]) {
    debug_assert!(!core::ptr::eq(x.as_ptr(), y.as_ptr()));

    let ord = num.len();
    let n = y.len();
    assert!(x.len() >= ord + n, "input must provide ord history samples");

    for i in 0..n {
        let mut acc = x[ord + i];
        for (tap, coeff) in num.iter().enumerate() {
            acc += coeff * x[ord + i - 1 - tap];
        }
        y[i] = acc;
    }
}

/// Fixed-point FIR used by CELT PLC in `FIXED_POINT` builds.
///
/// Mirrors the scalar semantics of `celt_fir_c()` from `celt/celt_lpc.c`.
/// The input slice stores `ord` history samples followed by `N` current
/// samples, where `N == y.len()`.
#[cfg(feature = "fixed_point")]
pub(crate) fn celt_fir_fixed(
    x: &[FixedOpusVal16],
    num: &[FixedOpusVal16],
    y: &mut [FixedOpusVal16],
) {
    let ord = num.len();
    let n = y.len();
    assert!(x.len() >= ord + n, "input must provide ord history samples");

    for i in 0..n {
        let mut acc = shl32(i32::from(x[ord + i]), SIG_SHIFT);
        for tap in 0..ord {
            let coeff = num[ord - 1 - tap];
            acc = mac16_16(acc, coeff, x[i + tap]);
        }
        y[i] = sround16_fixed(acc, SIG_SHIFT);
    }
}

/// Applies an all-pole IIR filter and updates the provided memory buffer.
///
/// Mirrors the small-footprint implementation of `celt_iir()` in
/// `celt/celt_lpc.c` for the float build. The denominator coefficients in
/// `den` encode the autoregressive part of the filter and the `mem` slice stores
/// the past outputs (`y[n-1]`, `y[n-2]`, ...).
pub(crate) fn celt_iir(
    x: &[OpusVal32],
    den: &[OpusVal16],
    y: &mut [OpusVal32],
    mem: &mut [OpusVal16],
) {
    let ord = den.len();
    assert_eq!(mem.len(), ord, "IIR memory must match denominator order");
    assert_eq!(
        x.len(),
        y.len(),
        "input and output must have the same length"
    );

    if ord == 0 {
        y.copy_from_slice(x);
        return;
    }

    for (input, output) in x.iter().zip(y.iter_mut()) {
        let mut acc = *input;
        for (coeff, state) in den.iter().zip(mem.iter()) {
            acc -= coeff * *state;
        }

        *output = acc;

        for idx in (1..ord).rev() {
            mem[idx] = mem[idx - 1];
        }
        mem[0] = acc as OpusVal16;
    }
}

/// Fixed-point all-pole IIR used by CELT PLC in `FIXED_POINT` builds.
///
/// Mirrors the scalar semantics of the non-`SMALL_FOOTPRINT` implementation
/// of `celt_iir()` from `celt/celt_lpc.c`.
#[cfg(feature = "fixed_point")]
pub(crate) fn celt_iir_fixed(
    x: &[FixedOpusVal32],
    den: &[FixedOpusVal16],
    y: &mut [FixedOpusVal32],
    mem: &mut [FixedOpusVal16],
) {
    let ord = den.len();
    assert_eq!(mem.len(), ord, "IIR memory must match denominator order");
    assert_eq!(
        x.len(),
        y.len(),
        "input and output must have the same length"
    );

    if ord == 0 {
        y.copy_from_slice(x);
        return;
    }

    debug_assert_eq!(ord & 3, 0, "celt_iir_fixed expects order divisible by 4");

    let mut rden = vec![0i16; ord];
    for (idx, slot) in rden.iter_mut().enumerate() {
        *slot = den[ord - 1 - idx];
    }

    let mut hist = vec![0i16; x.len() + ord];
    for i in 0..ord {
        hist[i] = mem[ord - 1 - i].wrapping_neg();
    }

    let mut i = 0usize;
    while i + 3 < x.len() {
        let mut sum = [x[i], x[i + 1], x[i + 2], x[i + 3]];
        xcorr_kernel_fixed(&rden, &hist[i..], &mut sum);

        hist[i + ord] = sround16_fixed(sum[0], SIG_SHIFT).wrapping_neg();
        y[i] = sum[0];

        sum[1] = mac16_16(sum[1], hist[i + ord], den[0]);
        hist[i + ord + 1] = sround16_fixed(sum[1], SIG_SHIFT).wrapping_neg();
        y[i + 1] = sum[1];

        sum[2] = mac16_16(sum[2], hist[i + ord + 1], den[0]);
        sum[2] = mac16_16(sum[2], hist[i + ord], den[1]);
        hist[i + ord + 2] = sround16_fixed(sum[2], SIG_SHIFT).wrapping_neg();
        y[i + 2] = sum[2];

        sum[3] = mac16_16(sum[3], hist[i + ord + 2], den[0]);
        sum[3] = mac16_16(sum[3], hist[i + ord + 1], den[1]);
        sum[3] = mac16_16(sum[3], hist[i + ord], den[2]);
        hist[i + ord + 3] = sround16_fixed(sum[3], SIG_SHIFT).wrapping_neg();
        y[i + 3] = sum[3];

        i += 4;
    }

    while i < x.len() {
        let mut sum = x[i];
        for j in 0..ord {
            sum = sum.wrapping_sub(mult16_16(rden[j], hist[i + j]));
        }
        hist[i + ord] = sround16_fixed(sum, SIG_SHIFT);
        y[i] = sum;
        i += 1;
    }

    for i in 0..ord {
        mem[i] = extract16(y[x.len() - 1 - i]);
    }
}

/// Computes the autocorrelation sequence of the input signal.
///
/// Mirrors the float configuration of `_celt_autocorr()` from `celt/celt_lpc.c`.
/// The routine optionally applies a symmetric analysis window spanning
/// `overlap` samples at the start and end of `x` before evaluating the
/// autocorrelation up to `lag`.
///
/// The returned shift value is specific to the fixed-point build in the
/// reference implementation; for the float configuration it always resolves to
/// zero and is provided only for API compatibility with future ports that may
/// consume the output.
pub(crate) fn celt_autocorr(
    x: &[OpusVal16],
    ac: &mut [OpusVal32],
    window: Option<&[CeltCoef]>,
    overlap: usize,
    lag: usize,
    arch: i32,
) -> i32 {
    assert!(
        !x.is_empty(),
        "input signal must contain at least one sample"
    );
    assert!(
        ac.len() > lag,
        "autocorrelation buffer must hold lag + 1 values"
    );
    assert!(lag <= x.len(), "lag must not exceed the input length");
    assert!(
        overlap <= x.len(),
        "window overlap cannot exceed the input length"
    );

    let n = x.len();
    let fast_n = n - lag;

    let xptr_cow = if overlap == 0 {
        Cow::Borrowed(x)
    } else {
        let window = window.expect("window coefficients required when overlap > 0");
        assert!(
            window.len() >= overlap,
            "window must provide at least overlap coefficients"
        );

        let mut buffer = x.to_vec();
        for i in 0..overlap {
            let w = window[i];
            buffer[i] *= w;
            let tail = n - i - 1;
            buffer[tail] *= w;
        }

        Cow::Owned(buffer)
    };
    let xptr = xptr_cow.as_ref();

    let _ = arch;

    celt_pitch_xcorr(xptr, xptr, fast_n, lag + 1, ac);

    for (k, slot) in ac.iter_mut().enumerate().take(lag + 1) {
        let mut d = 0.0;
        for i in k + fast_n..n {
            d += xptr[i] * xptr[i - k];
        }
        *slot += d;
    }

    0
}

/// Fixed-point autocorrelation helper used by the pitch search and PLC paths.
///
/// Mirrors the `FIXED_POINT` branch of `_celt_autocorr()` from
/// `celt/celt_lpc.c`, including the dynamic rescaling that keeps the
/// autocorrelation values within range.
#[cfg(feature = "fixed_point")]
pub(crate) fn celt_autocorr_fixed(
    x: &[FixedOpusVal16],
    ac: &mut [FixedOpusVal32],
    window: Option<&[FixedCeltCoef]>,
    overlap: usize,
    lag: usize,
    arch: i32,
) -> i32 {
    assert!(
        !x.is_empty(),
        "input signal must contain at least one sample"
    );
    assert!(
        ac.len() > lag,
        "autocorrelation buffer must hold lag + 1 values"
    );
    assert!(lag <= x.len(), "lag must not exceed the input length");
    assert!(
        overlap <= x.len(),
        "window overlap cannot exceed the input length"
    );

    let n = x.len();
    let fast_n = n - lag;
    let _ = arch;

    let mut buffer = if overlap > 0 {
        let window = window.expect("window coefficients required when overlap > 0");
        assert!(
            window.len() >= overlap,
            "window must provide at least overlap coefficients"
        );

        let mut tmp = x.to_vec();
        for i in 0..overlap {
            let w = window[i];
            tmp[i] = mult16_16_q15(tmp[i], w);
            let tail = n - i - 1;
            tmp[tail] = mult16_16_q15(tmp[tail], w);
        }
        Some(tmp)
    } else {
        None
    };
    let mut xptr: &[FixedOpusVal16] = buffer.as_ref().map_or(x, |buf| buf);

    let mut ac0 = 1i32 + ((n as i32) << 7);
    if n & 1 != 0 {
        ac0 = ac0.wrapping_add(shr32(mult16_16(xptr[0], xptr[0]), 9));
    }
    let mut i = (n & 1) as usize;
    while i < n {
        ac0 = ac0.wrapping_add(shr32(mult16_16(xptr[i], xptr[i]), 9));
        ac0 = ac0.wrapping_add(shr32(mult16_16(xptr[i + 1], xptr[i + 1]), 9));
        i += 2;
    }

    let mut shift = (celt_ilog2(ac0) - 30 + 10) / 2;
    if shift > 0 {
        let mut shifted = Vec::with_capacity(n);
        for i in 0..n {
            shifted.push(pshr32(i32::from(xptr[i]), shift as u32) as i16);
        }
        buffer = Some(shifted);
        xptr = buffer.as_ref().expect("shifted buffer must exist");
    } else {
        shift = 0;
    }

    celt_pitch_xcorr_fixed(xptr, xptr, fast_n, lag + 1, ac);
    for k in 0..=lag {
        let mut d = 0i32;
        for i in k + fast_n..n {
            d = add32(d, mult16_16(xptr[i], xptr[i - k]));
        }
        ac[k] = ac[k].wrapping_add(d);
    }

    shift *= 2;
    if shift <= 0 {
        ac[0] = ac[0].wrapping_add(shl32(1, (-shift) as u32));
    }
    if ac[0] < 268_435_456 {
        let shift2 = 29 - ec_ilog(ac[0] as u32);
        for value in ac.iter_mut().take(lag + 1) {
            *value = shl32(*value, shift2 as u32);
        }
        shift -= shift2;
    } else if ac[0] >= 536_870_912 {
        let mut shift2 = 1;
        if ac[0] >= 1_073_741_824 {
            shift2 += 1;
        }
        for value in ac.iter_mut().take(lag + 1) {
            *value = shr32(*value, shift2 as u32);
        }
        shift += shift2;
    }

    shift
}

#[cfg(test)]
mod tests {
    use super::{celt_autocorr, celt_fir, celt_iir, celt_lpc};
    #[cfg(feature = "fixed_point")]
    use super::{celt_fir_fixed, celt_iir_fixed, sround16_fixed, xcorr_kernel_fixed};
    #[cfg(feature = "fixed_point")]
    use crate::celt::fixed_arch::{int16tosig, SIG_SHIFT};
    #[cfg(feature = "fixed_point")]
    use crate::celt::fixed_ops::{mac16_16, mult16_16, pshr32, shl32};
    use crate::celt::types::{OpusVal16, OpusVal32};
    #[cfg(feature = "fixed_point")]
    use crate::celt::types::{FixedOpusVal16, FixedOpusVal32};
    use alloc::vec;
    use alloc::vec::Vec;

    fn reference_lpc(ac: &[f64], order: usize) -> Vec<f64> {
        let mut lpc = vec![0.0f64; order];
        if order == 0 {
            return lpc;
        }

        let ac0 = ac[0];
        if ac0 <= 1e-10 {
            return lpc;
        }

        let mut error = ac0;
        for i in 0..order {
            let mut rr = 0.0f64;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];

            let reflection = -rr / error;
            lpc[i] = reflection;

            let half = (i + 1) >> 1;
            for j in 0..half {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + reflection * tmp2;
                lpc[i - 1 - j] = tmp2 + reflection * tmp1;
            }

            error -= (reflection * reflection) * error;
            if error <= 0.001 * ac0 {
                break;
            }
        }

        lpc
    }

    fn generate_autocorrelation(order: usize, len: usize) -> Vec<f64> {
        let mut seed = 0x1234_5678u32;
        let mut signal = vec![0.0f64; len + order];

        for n in order..signal.len() {
            let noise = {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let val = f64::from(seed >> 1) / f64::from(u32::MAX >> 1);
                (val - 0.5) * 0.1
            };

            let mut value = noise;
            for k in 1..=order {
                value += 0.3f64.powi(k as i32) * signal[n - k];
            }
            signal[n] = value;
        }

        let mut ac = vec![0.0f64; order + 1];
        for lag in 0..=order {
            let mut sum = 0.0f64;
            for n in lag..signal.len() {
                sum += signal[n] * signal[n - lag];
            }
            ac[lag] = sum;
        }
        ac
    }

    fn reference_autocorr(signal: &[OpusVal16], lag: usize) -> Vec<OpusVal32> {
        let n = signal.len();
        (0..=lag)
            .map(|k| {
                let mut acc = 0.0f32;
                for i in 0..n.saturating_sub(k) {
                    acc += signal[i] * signal[i + k];
                }
                acc
            })
            .collect()
    }

    #[test]
    fn autocorr_matches_reference_without_window() {
        let n = 32;
        let lag = 6;
        let mut seed = 0x2468_acdfu32;
        let mut signal = vec![0.0f32; n];
        for sample in &mut signal {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = ((seed >> 9) as f32 / (u32::MAX >> 1) as f32) - 1.0;
        }

        let mut ac = vec![0.0f32; lag + 1];
        let shift = celt_autocorr(&signal, &mut ac, None, 0, lag, 0);

        let expected = reference_autocorr(&signal, lag);
        for (lhs, rhs) in ac.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
        assert_eq!(shift, 0);
    }

    #[test]
    fn autocorr_applies_window_symmetrically() {
        let n = 24;
        let lag = 4;
        let overlap = 6;
        let mut signal = vec![0.0f32; n];
        for (idx, sample) in signal.iter_mut().enumerate() {
            *sample = (idx as f32 * 0.1).sin();
        }

        let mut window = vec![0.0f32; overlap];
        for (i, slot) in window.iter_mut().enumerate() {
            let phase = core::f32::consts::PI * (i as f32 + 0.5) / overlap as f32;
            *slot = phase.sin();
        }

        let mut windowed = signal.clone();
        for i in 0..overlap {
            windowed[i] *= window[i];
            let tail = n - i - 1;
            windowed[tail] *= window[i];
        }

        let mut ac = vec![0.0f32; lag + 1];
        celt_autocorr(&signal, &mut ac, Some(window.as_slice()), overlap, lag, 0);

        let expected = reference_autocorr(&windowed, lag);
        for (lhs, rhs) in ac.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn lpc_matches_reference_for_randomish_signal() {
        let order = 8;
        let ac = generate_autocorrelation(order, 128);
        let expected = reference_lpc(&ac, order);

        let mut coeffs = vec![0.0f32; order];
        let ac_f32: Vec<OpusVal32> = ac.iter().map(|&v| v as OpusVal32).collect();
        celt_lpc(&mut coeffs, &ac_f32);

        for (got, want) in coeffs.iter().zip(expected.iter()) {
            assert!(
                (f64::from(*got) - *want).abs() <= 1e-5,
                "got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lpc_leaves_coefficients_zero_for_low_energy() {
        let mut coeffs = [1.0f32, -2.0, 3.0];
        let ac: [OpusVal32; 4] = [1e-12, 0.0, 0.0, 0.0];
        celt_lpc(&mut coeffs, &ac);
        assert_eq!(coeffs, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn lpc_handles_zero_order() {
        let mut coeffs: [OpusVal16; 0] = [];
        let ac: [OpusVal32; 1] = [1.0];
        celt_lpc(&mut coeffs, &ac);
    }

    #[test]
    fn fir_matches_reference_response() {
        let ord = 4;
        let taps = [0.2f32, -0.15, 0.05, 0.1];
        let history = [0.5f32, -0.25, 0.1, 0.0];
        let input = [0.3f32, -0.4, 0.2, -0.1, 0.05, 0.6];

        let mut buffer = history.to_vec();
        buffer.extend_from_slice(&input);

        let mut output = vec![0.0f32; input.len()];
        celt_fir(&buffer, &taps, &mut output);

        for (i, got) in output.iter().enumerate() {
            let mut expected = buffer[ord + i];
            for k in 0..ord {
                expected += taps[k] * buffer[ord + i - 1 - k];
            }
            assert!(
                (expected - *got).abs() <= 1e-6,
                "idx {i}: got {got}, want {expected}"
            );
        }
    }

    #[test]
    fn iir_matches_reference_response() {
        let den = [0.4f32, -0.2, 0.1];
        let input = [0.5f32, 0.1, -0.3, 0.2, 0.0, -0.1, 0.4];

        let mut mem = vec![0.0f32; den.len()];
        let mut output = vec![0.0f32; input.len()];
        celt_iir(&input, &den, &mut output, &mut mem);

        let mut ref_mem = vec![0.0f32; den.len()];
        let mut expected = vec![0.0f32; input.len()];
        for (idx, (&x, y)) in input.iter().zip(expected.iter_mut()).enumerate() {
            let mut acc = x;
            for (coeff, state) in den.iter().zip(ref_mem.iter()) {
                acc -= coeff * *state;
            }
            *y = acc;
            for j in (1..den.len()).rev() {
                ref_mem[j] = ref_mem[j - 1];
            }
            if !den.is_empty() {
                ref_mem[0] = acc;
            }
            assert!(
                (output[idx] - acc).abs() <= 1e-6,
                "idx {idx}: got {}, want {acc}",
                output[idx]
            );
        }

        assert_eq!(mem.len(), ref_mem.len());
        for (idx, (got, want)) in mem.iter().zip(ref_mem.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6,
                "mem {idx}: got {got}, want {want}"
            );
        }
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn fir_fixed_matches_reference_response() {
        let taps: [FixedOpusVal16; 4] = [768, -512, 256, 128];
        let history: [FixedOpusVal16; 4] = [3000, -2500, 1700, -900];
        let input: [FixedOpusVal16; 6] = [2100, -1800, 1400, -1100, 700, 400];

        let mut buffer = history.to_vec();
        buffer.extend_from_slice(&input);

        let mut output = vec![0i16; input.len()];
        celt_fir_fixed(&buffer, &taps, &mut output);

        for i in 0..input.len() {
            let mut acc = shl32(i32::from(buffer[taps.len() + i]), SIG_SHIFT);
            for tap in 0..taps.len() {
                let coeff = taps[taps.len() - 1 - tap];
                acc = mac16_16(acc, coeff, buffer[i + tap]);
            }
            let expected = sround16_fixed(acc, SIG_SHIFT);
            assert_eq!(output[i], expected, "idx {i}");
        }
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn iir_fixed_matches_reference_response() {
        let den: [FixedOpusVal16; 4] = [1536, -896, 320, -192];
        let input: [FixedOpusVal32; 8] = [
            int16tosig(2400),
            int16tosig(1200),
            int16tosig(-1800),
            int16tosig(900),
            int16tosig(0),
            int16tosig(-700),
            int16tosig(1600),
            int16tosig(-500),
        ];

        let mut mem: Vec<FixedOpusVal16> = vec![180, -240, 75, -120];
        let mut output = vec![0i32; input.len()];
        celt_iir_fixed(&input, &den, &mut output, &mut mem);

        let mut rden = vec![0i16; den.len()];
        for (idx, slot) in rden.iter_mut().enumerate() {
            *slot = den[den.len() - 1 - idx];
        }
        let mut ref_mem: Vec<FixedOpusVal16> = vec![180, -240, 75, -120];
        let mut expected = vec![0i32; input.len()];
        let mut hist = vec![0i16; input.len() + den.len()];
        for i in 0..den.len() {
            hist[i] = ref_mem[den.len() - 1 - i].wrapping_neg();
        }

        let mut i = 0usize;
        while i + 3 < input.len() {
            let mut sum = [input[i], input[i + 1], input[i + 2], input[i + 3]];
            xcorr_kernel_fixed(&rden, &hist[i..], &mut sum);

            hist[i + den.len()] = sround16_fixed(sum[0], SIG_SHIFT).wrapping_neg();
            expected[i] = sum[0];

            sum[1] = mac16_16(sum[1], hist[i + den.len()], den[0]);
            hist[i + den.len() + 1] = sround16_fixed(sum[1], SIG_SHIFT).wrapping_neg();
            expected[i + 1] = sum[1];

            sum[2] = mac16_16(sum[2], hist[i + den.len() + 1], den[0]);
            sum[2] = mac16_16(sum[2], hist[i + den.len()], den[1]);
            hist[i + den.len() + 2] = sround16_fixed(sum[2], SIG_SHIFT).wrapping_neg();
            expected[i + 2] = sum[2];

            sum[3] = mac16_16(sum[3], hist[i + den.len() + 2], den[0]);
            sum[3] = mac16_16(sum[3], hist[i + den.len() + 1], den[1]);
            sum[3] = mac16_16(sum[3], hist[i + den.len()], den[2]);
            hist[i + den.len() + 3] = sround16_fixed(sum[3], SIG_SHIFT).wrapping_neg();
            expected[i + 3] = sum[3];

            i += 4;
        }

        while i < input.len() {
            let mut sum = input[i];
            for j in 0..den.len() {
                sum = sum.wrapping_sub(mult16_16(rden[j], hist[i + j]));
            }
            hist[i + den.len()] = sround16_fixed(sum, SIG_SHIFT);
            expected[i] = sum;
            i += 1;
        }

        for i in 0..den.len() {
            ref_mem[i] = expected[input.len() - 1 - i] as FixedOpusVal16;
        }

        assert_eq!(output, expected);
        assert_eq!(mem, ref_mem);
    }
}
