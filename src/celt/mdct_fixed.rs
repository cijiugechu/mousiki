#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;

use super::fixed_ops::{
    abs32, add32_ovflw, neg32_ovflw, pshr32, pshr32_ovflw, shl32, shr32, sub32_ovflw,
};
use super::fixed_ops::{mult16_32_q15, mult16_32_q16};
use super::kiss_fft_fixed::{FixedKissFftCpx, FixedKissFftState};
use super::math::{celt_ilog2, celt_zlog2};
use super::math_fixed::celt_cos_norm;
use super::types::{FixedCeltCoef, FixedCeltSig};

#[derive(Debug, Clone)]
pub struct FixedMdctLookup {
    pub len: usize,
    pub max_shift: usize,
    pub forward: Vec<FixedKissFftState>,
    pub inverse: Vec<FixedKissFftState>,
    pub twiddle: Vec<FixedCeltCoef>,
    pub twiddle_offsets: Vec<usize>,
}

impl FixedMdctLookup {
    #[must_use]
    pub fn new(len: usize, max_shift: usize) -> Self {
        assert!(len.is_multiple_of(2), "MDCT length must be even");
        assert!(max_shift < 8, "unsupported MDCT shift");
        assert!(
            len >> max_shift > 0,
            "MDCT length too small for requested shift"
        );

        let mut forward = Vec::with_capacity(max_shift + 1);
        let mut inverse = Vec::with_capacity(max_shift + 1);
        let n0 = len;
        assert!(
            n0.is_multiple_of(4),
            "MDCT length must be a multiple of four"
        );
        let base = FixedKissFftState::new(n0 >> 2);
        forward.push(base.clone());
        inverse.push(base);
        for shift in 1..=max_shift {
            let n = len >> shift;
            assert!(
                n.is_multiple_of(4),
                "MDCT length must be a multiple of four"
            );
            let base_state = FixedKissFftState::with_base(n >> 2, Some(&forward[0]));
            let base_inverse = FixedKissFftState::with_base(n >> 2, Some(&inverse[0]));
            forward.push(base_state);
            inverse.push(base_inverse);
        }

        let mut twiddle = Vec::new();
        let mut offsets = Vec::with_capacity(max_shift + 2);
        offsets.push(0);
        let mut n = len;
        let mut n2 = n >> 1;
        for _ in 0..=max_shift {
            for i in 0..n2 {
                let phase =
                    shl32(i as i32, 17).wrapping_add(n2 as i32 + 16_384) / n as i32;
                twiddle.push(celt_cos_norm(phase));
            }
            offsets.push(twiddle.len());
            n2 >>= 1;
            n >>= 1;
        }

        Self {
            len,
            max_shift,
            forward,
            inverse,
            twiddle,
            twiddle_offsets: offsets,
        }
    }

    #[inline]
    #[must_use]
    pub fn effective_len(&self, shift: usize) -> usize {
        assert!(shift <= self.max_shift);
        self.len >> shift
    }

    #[inline]
    #[must_use]
    pub fn forward_plan(&self, shift: usize) -> &FixedKissFftState {
        assert!(shift < self.forward.len());
        &self.forward[shift]
    }

    #[inline]
    #[must_use]
    pub fn inverse_plan(&self, shift: usize) -> &FixedKissFftState {
        assert!(shift < self.inverse.len());
        &self.inverse[shift]
    }

    #[inline]
    #[must_use]
    pub fn twiddles(&self, shift: usize) -> &[FixedCeltCoef] {
        assert!(shift <= self.max_shift);
        let start = self.twiddle_offsets[shift];
        let end = self.twiddle_offsets[shift + 1];
        &self.twiddle[start..end]
    }
}

#[inline]
fn s_mul(a: FixedCeltSig, b: FixedCeltCoef) -> FixedCeltSig {
    mult16_32_q15(b, a)
}

#[inline]
fn s_mul2(a: FixedCeltSig, b: FixedCeltCoef) -> FixedCeltSig {
    mult16_32_q16(b, a)
}

fn fold_input(
    input: &[FixedCeltSig],
    window: &[FixedCeltCoef],
    overlap: usize,
    n2: usize,
) -> Vec<FixedCeltSig> {
    let n4 = n2 >> 1;
    let quarter_overlap = (overlap + 3) >> 2;
    let half_overlap = overlap >> 1;

    let mut folded = vec![0; n2];
    let mut yp = 0usize;

    let mut xp1 = half_overlap as isize;
    let mut xp2 = (half_overlap + n2 - 1) as isize;
    let mut wp1 = half_overlap as isize;
    let mut wp2 = half_overlap as isize - 1;

    let n2_isize = n2 as isize;

    for _ in 0..quarter_overlap {
        let re = add32_ovflw(
            s_mul(input[(xp1 + n2_isize) as usize], window[wp2 as usize]),
            s_mul(input[xp2 as usize], window[wp1 as usize]),
        );
        let im = sub32_ovflw(
            s_mul(input[xp1 as usize], window[wp1 as usize]),
            s_mul(input[(xp2 - n2_isize) as usize], window[wp2 as usize]),
        );
        folded[yp] = re;
        folded[yp + 1] = im;
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
        wp1 += 2;
        wp2 -= 2;
    }

    for _ in quarter_overlap..(n4 - quarter_overlap) {
        let re = input[xp2 as usize];
        let im = input[xp1 as usize];
        folded[yp] = re;
        folded[yp + 1] = im;
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
    }

    wp1 = 0;
    wp2 = overlap as isize - 1;

    for _ in (n4 - quarter_overlap)..n4 {
        let re = add32_ovflw(
            neg32_ovflw(s_mul(input[(xp1 - n2_isize) as usize], window[wp1 as usize])),
            s_mul(input[xp2 as usize], window[wp2 as usize]),
        );
        let im = add32_ovflw(
            s_mul(input[xp1 as usize], window[wp2 as usize]),
            s_mul(input[(xp2 + n2_isize) as usize], window[wp1 as usize]),
        );
        folded[yp] = re;
        folded[yp + 1] = im;
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
        wp1 += 2;
        wp2 -= 2;
    }

    folded
}

fn pre_rotate_forward(
    folded: &[FixedCeltSig],
    twiddles: &[FixedCeltCoef],
    n4: usize,
    fft: &FixedKissFftState,
) -> (Vec<FixedKissFftCpx>, i32) {
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let mut out = vec![FixedKissFftCpx::default(); n4];
    let scale = fft.scale();
    let scale_shift = fft.scale_shift() - 1;
    let mut maxval = 1i32;

    for i in 0..n4 {
        let re = folded[2 * i];
        let im = folded[2 * i + 1];
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        let yr = sub32_ovflw(s_mul(re, t0), s_mul(im, t1));
        let yi = add32_ovflw(s_mul(im, t0), s_mul(re, t1));
        let yc = FixedKissFftCpx::new(s_mul2(yr, scale), s_mul2(yi, scale));
        maxval = maxval.max(abs32(yc.r)).max(abs32(yc.i));
        out[fft.bitrev()[i]] = yc;
    }

    let headroom = if maxval > 0 {
        let limit = 28 - celt_ilog2(maxval);
        limit.clamp(0, scale_shift)
    } else {
        0
    };

    (out, headroom)
}

fn post_rotate_forward(
    freq: &[FixedKissFftCpx],
    twiddles: &[FixedCeltCoef],
    out: &mut [FixedCeltSig],
    stride: usize,
    headroom: i32,
) {
    let n4 = freq.len();
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let n2 = n4 * 2;
    let mut left = 0usize;
    let mut right = (n2 - 1) * stride;
    for i in 0..n4 {
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        let yr = pshr32(
            sub32_ovflw(s_mul(freq[i].i, t1), s_mul(freq[i].r, t0)),
            headroom as u32,
        );
        let yi = pshr32(
            add32_ovflw(s_mul(freq[i].r, t1), s_mul(freq[i].i, t0)),
            headroom as u32,
        );
        out[left] = yr;
        out[right] = yi;
        left += 2 * stride;
        if right >= 2 * stride {
            right -= 2 * stride;
        } else {
            right = 0;
        }
    }
}

fn pre_rotate_backward(
    input: &[FixedCeltSig],
    twiddles: &[FixedCeltCoef],
    stride: usize,
    fft: &FixedKissFftState,
    pre_shift: i32,
) -> Vec<FixedKissFftCpx> {
    let n2 = input.len() / stride;
    let n4 = n2 / 2;
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let mut out = vec![FixedKissFftCpx::default(); n4];
    let stride_isize = stride as isize;
    let mut xp1 = 0isize;
    let mut xp2 = (n2 as isize - 1) * stride_isize;
    for (i, &rev) in fft.bitrev().iter().take(n4).enumerate() {
        let x1 = shl32(input[xp1 as usize], pre_shift as u32);
        let x2 = shl32(input[xp2 as usize], pre_shift as u32);
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        let yr = add32_ovflw(s_mul(x2, t0), s_mul(x1, t1));
        let yi = sub32_ovflw(s_mul(x1, t0), s_mul(x2, t1));
        out[rev] = FixedKissFftCpx::new(yi, yr);
        xp1 += 2 * stride_isize;
        xp2 -= 2 * stride_isize;
    }
    out
}

fn post_rotate_backward(
    freq: &[FixedKissFftCpx],
    twiddles: &[FixedCeltCoef],
    out: &mut [FixedCeltSig],
    window: &[FixedCeltCoef],
    overlap: usize,
    post_shift: i32,
) {
    let n4 = freq.len();
    let n2 = n4 * 2;
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let half_overlap = overlap >> 1;
    let mut temp = vec![0; n2];

    let pairs = (n4 + 1) >> 1;
    for i in 0..pairs {
        let re_front = freq[i].i;
        let im_front = freq[i].r;
        let t0_front = cos_part[i];
        let t1_front = sin_part[i];
        let yr_front = pshr32_ovflw(
            add32_ovflw(s_mul(re_front, t0_front), s_mul(im_front, t1_front)),
            post_shift as u32,
        );
        let yi_front = pshr32_ovflw(
            sub32_ovflw(s_mul(re_front, t1_front), s_mul(im_front, t0_front)),
            post_shift as u32,
        );

        let back_index = n4 - i - 1;
        let (yr_back, yi_back) = if back_index == i {
            (yr_front, yi_front)
        } else {
            let re_back = freq[back_index].i;
            let im_back = freq[back_index].r;
            let t0_back = cos_part[back_index];
            let t1_back = sin_part[back_index];
            (
                pshr32_ovflw(
                    add32_ovflw(s_mul(re_back, t0_back), s_mul(im_back, t1_back)),
                    post_shift as u32,
                ),
                pshr32_ovflw(
                    sub32_ovflw(s_mul(re_back, t1_back), s_mul(im_back, t0_back)),
                    post_shift as u32,
                ),
            )
        };

        let front_even = 2 * i;
        let front_odd = front_even + 1;
        let back_even = n2.saturating_sub(2 * (i + 1));
        let back_odd = back_even + 1;

        temp[front_even] = yr_front;
        temp[front_odd] = yi_back;
        temp[back_even] = yr_back;
        temp[back_odd] = yi_front;
    }

    for (dst, src) in out[half_overlap..half_overlap + n2]
        .iter_mut()
        .zip(temp.iter())
    {
        *dst = *src;
    }

    if overlap == 0 {
        return;
    }

    for i in 0..(overlap >> 1) {
        let wp1 = i;
        let wp2 = overlap - 1 - i;
        let yp1 = i;
        let xp1 = overlap - 1 - i;
        let x1 = out[xp1];
        let x2 = out[yp1];
        out[yp1] = sub32_ovflw(s_mul(x2, window[wp2]), s_mul(x1, window[wp1]));
        out[xp1] = add32_ovflw(s_mul(x2, window[wp1]), s_mul(x1, window[wp2]));
    }
}

pub fn clt_mdct_forward_fixed(
    lookup: &FixedMdctLookup,
    input: &[FixedCeltSig],
    output: &mut [FixedCeltSig],
    window: &[FixedCeltCoef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let n = lookup.effective_len(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    assert!(input.len() >= overlap + n);
    assert!(window.len() >= overlap);
    let required = (n2 - 1) * stride + 1;
    assert!(output.len() >= required);
    assert!(stride > 0);

    let twiddles = lookup.twiddles(shift);
    let folded = fold_input(input, window, overlap, n2);
    let fft = lookup.forward_plan(shift);
    let (mut spectrum, headroom) = pre_rotate_forward(&folded, twiddles, n4, fft);
    fft.process(&mut spectrum, fft.scale_shift() - 1 - headroom);
    post_rotate_forward(&spectrum, twiddles, output, stride, headroom);
}

pub fn clt_mdct_backward_fixed(
    lookup: &FixedMdctLookup,
    input: &[FixedCeltSig],
    output: &mut [FixedCeltSig],
    window: &[FixedCeltCoef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let n = lookup.effective_len(shift);
    let n2 = n >> 1;
    assert!(input.len() >= stride * n2);
    assert!(window.len() >= overlap);
    let half_overlap = overlap >> 1;
    assert!(output.len() >= overlap);
    assert!(output.len() >= half_overlap + n2);
    assert!(stride > 0);

    let twiddles = lookup.twiddles(shift);
    output.fill(0);

    let mut sumval = n2 as i32;
    let mut maxval = 0i32;
    for i in 0..n2 {
        let sample = input[i * stride];
        maxval = maxval.max(abs32(sample));
        sumval = sumval.wrapping_add(abs32(shr32(sample, 11)));
    }

    let pre_shift = (29 - celt_zlog2(add32_ovflw(maxval, 1))).max(0);
    let mut post_shift = (19 - celt_ilog2(abs32(sumval))).max(0);
    post_shift = post_shift.min(pre_shift);
    let fft_shift = pre_shift - post_shift;

    let fft = lookup.inverse_plan(shift);
    let mut pre = pre_rotate_backward(input, twiddles, stride, fft, pre_shift);
    fft.process(&mut pre, fft_shift);
    post_rotate_backward(&pre, twiddles, output, window, overlap, post_shift);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt::fixed_arch::{float2sig, SIG_SHIFT};
    use crate::celt::float_cast::CELT_SIG_SCALE;
    use crate::celt::mdct::{clt_mdct_backward, clt_mdct_forward};
    use crate::celt::modes::compute_mdct_window;
    use crate::celt::types::MdctLookup;

    fn sig_to_float(value: FixedCeltSig) -> f32 {
        let scale = CELT_SIG_SCALE * (1u32 << SIG_SHIFT) as f32;
        value as f32 / scale
    }

    fn window_to_float(window: &[FixedCeltCoef]) -> Vec<f32> {
        window
            .iter()
            .map(|&w| w as f32 / 32768.0)
            .collect()
    }

    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let mut num = 0.0f32;
        let mut den_a = 0.0f32;
        let mut den_b = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            num += x * y;
            den_a += x * x;
            den_b += y * y;
        }
        if den_a == 0.0 || den_b == 0.0 {
            return 0.0;
        }
        num / (den_a.sqrt() * den_b.sqrt())
    }

    #[test]
    fn fixed_forward_matches_float_reference() {
        let sizes = [16usize, 32];
        for &n in &sizes {
            let overlap = n / 2;
            let float_window = compute_mdct_window(overlap);
            let fixed_window: Vec<FixedCeltCoef> = float_window
                .iter()
                .map(|&w| (w * 32768.0).round() as FixedCeltCoef)
                .collect();

            let mdct = MdctLookup::new(n, 0);
            let fixed_mdct = FixedMdctLookup::new(n, 0);
            let float_window_fixed = window_to_float(&fixed_window);

            let mut input = vec![0.0f32; overlap + n];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = (i as f32 * 0.37).sin();
            }
            let fixed_input: Vec<FixedCeltSig> = input.iter().map(|&v| float2sig(v)).collect();

            let mut float_out = vec![0.0f32; n / 2];
            clt_mdct_forward(
                &mdct,
                &input,
                &mut float_out,
                &float_window_fixed,
                overlap,
                0,
                1,
            );

            let mut fixed_out = vec![0; n / 2];
            clt_mdct_forward_fixed(
                &fixed_mdct,
                &fixed_input,
                &mut fixed_out,
                &fixed_window,
                overlap,
                0,
                1,
            );

            let fixed_out_float: Vec<f32> = fixed_out.into_iter().map(sig_to_float).collect();
            let max_fixed = fixed_out_float
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()));
            let max_float = float_out
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = if max_fixed > 0.0 {
                max_float / max_fixed
            } else {
                1.0
            };
            let scaled: Vec<f32> = fixed_out_float.iter().map(|v| v * scale).collect();
            let corr = correlation(&scaled, &float_out);
            assert!(corr > 0.99, "correlation {corr}");
        }
    }

    #[test]
    fn fixed_backward_matches_float_reference() {
        let sizes = [16usize, 32];
        for &n in &sizes {
            let overlap = n / 2;
            let float_window = compute_mdct_window(overlap);
            let fixed_window: Vec<FixedCeltCoef> = float_window
                .iter()
                .map(|&w| (w * 32768.0).round() as FixedCeltCoef)
                .collect();

            let mdct = MdctLookup::new(n, 0);
            let fixed_mdct = FixedMdctLookup::new(n, 0);
            let float_window_fixed = window_to_float(&fixed_window);

            let mut input = vec![0.0f32; n / 2];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = (i as f32 * 0.19).cos();
            }
            let fixed_input: Vec<FixedCeltSig> = input.iter().map(|&v| float2sig(v)).collect();

            let mut float_out = vec![0.0f32; overlap.max((overlap >> 1) + n / 2)];
            clt_mdct_backward(
                &mdct,
                &input,
                &mut float_out,
                &float_window_fixed,
                overlap,
                0,
                1,
            );

            let mut fixed_out = vec![0; overlap.max((overlap >> 1) + n / 2)];
            clt_mdct_backward_fixed(
                &fixed_mdct,
                &fixed_input,
                &mut fixed_out,
                &fixed_window,
                overlap,
                0,
                1,
            );

            let fixed_out_float: Vec<f32> = fixed_out.into_iter().map(sig_to_float).collect();
            let max_fixed = fixed_out_float
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()));
            let max_float = float_out
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = if max_fixed > 0.0 {
                max_float / max_fixed
            } else {
                1.0
            };
            let scaled: Vec<f32> = fixed_out_float.iter().map(|v| v * scale).collect();
            let corr = correlation(&scaled, &float_out);
            assert!(corr > 0.99, "correlation {corr}");
        }
    }
}
