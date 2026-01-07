#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;

use super::mini_kfft::KissFftCpx;
use super::types::{CeltCoef, MdctLookup};

fn fold_input(input: &[f32], window: &[CeltCoef], overlap: usize, n2: usize) -> Vec<f32> {
    let n4 = n2 >> 1;
    let quarter_overlap = (overlap + 3) >> 2;
    let half_overlap = overlap >> 1;

    let mut folded = vec![0.0f32; n2];
    let mut yp = 0usize;

    let mut xp1 = half_overlap as isize;
    let mut xp2 = (half_overlap + n2 - 1) as isize;
    let mut wp1 = half_overlap as isize;
    let mut wp2 = half_overlap as isize - 1;

    let n2_isize = n2 as isize;

    for _ in 0..quarter_overlap {
        let re = input[(xp1 + n2_isize) as usize] * window[wp2 as usize]
            + input[xp2 as usize] * window[wp1 as usize];
        let im = input[xp1 as usize] * window[wp1 as usize]
            - input[(xp2 - n2_isize) as usize] * window[wp2 as usize];
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
        let re = -input[(xp1 - n2_isize) as usize] * window[wp1 as usize]
            + input[xp2 as usize] * window[wp2 as usize];
        let im = input[xp1 as usize] * window[wp2 as usize]
            + input[(xp2 + n2_isize) as usize] * window[wp1 as usize];
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

fn pre_rotate_forward(folded: &[f32], twiddles: &[f32], n4: usize) -> Vec<KissFftCpx> {
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let mut out = vec![KissFftCpx::default(); n4];
    for i in 0..n4 {
        let re = folded[2 * i];
        let im = folded[2 * i + 1];
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        let yr = re * t0 - im * t1;
        let yi = im * t0 + re * t1;
        out[i] = KissFftCpx::new(yr, yi);
    }
    out
}

fn post_rotate_forward(freq: &[KissFftCpx], twiddles: &[f32], out: &mut [f32], stride: usize) {
    let n4 = freq.len();
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let n2 = n4 * 2;
    let mut left = 0usize;
    let mut right = (n2 - 1) * stride;
    for i in 0..n4 {
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        let yr = freq[i].i * t1 - freq[i].r * t0;
        let yi = freq[i].r * t1 + freq[i].i * t0;
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

fn pre_rotate_backward(input: &[f32], twiddles: &[f32], stride: usize) -> Vec<KissFftCpx> {
    let n2 = input.len() / stride;
    let n4 = n2 / 2;
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let mut out = vec![KissFftCpx::default(); n4];
    let stride_isize = stride as isize;
    let mut xp1 = 0isize;
    let mut xp2 = (n2 as isize - 1) * stride_isize;
    for i in 0..n4 {
        let x1 = input[xp1 as usize];
        let x2 = input[xp2 as usize];
        let t0 = cos_part[i];
        let t1 = sin_part[i];
        let re = x2 * t0 + x1 * t1;
        let im = x1 * t0 - x2 * t1;
        out[i] = KissFftCpx::new(re, im);
        xp1 += 2 * stride_isize;
        xp2 -= 2 * stride_isize;
    }
    out
}

fn post_rotate_backward(
    freq: &[KissFftCpx],
    twiddles: &[f32],
    out: &mut [f32],
    window: &[CeltCoef],
    overlap: usize,
) {
    let n4 = freq.len();
    let n2 = n4 * 2;
    let (cos_part, sin_part) = twiddles.split_at(n4);
    let half_overlap = overlap >> 1;
    let mut temp = vec![0.0f32; n2];

    let pairs = (n4 + 1) >> 1;
    for i in 0..pairs {
        let f_front = freq[i];
        let t0_front = cos_part[i];
        let t1_front = sin_part[i];
        let yr_front = f_front.r * t0_front + f_front.i * t1_front;
        let yi_front = f_front.r * t1_front - f_front.i * t0_front;

        let back_index = n4 - i - 1;
        let (yr_back, yi_back) = if back_index == i {
            (yr_front, yi_front)
        } else {
            let f_back = freq[back_index];
            let t0_back = cos_part[back_index];
            let t1_back = sin_part[back_index];
            (
                f_back.r * t0_back + f_back.i * t1_back,
                f_back.r * t1_back - f_back.i * t0_back,
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

    for (offset, (&w1, &w2)) in window
        .iter()
        .zip(window.iter().rev())
        .take(overlap >> 1)
        .enumerate()
    {
        let yp1 = offset;
        let xp1 = overlap - 1 - offset;
        let x1 = out[xp1];
        let x2 = out[yp1];
        out[yp1] = x2 * w2 - x1 * w1;
        out[xp1] = x2 * w1 + x1 * w2;
    }
}

pub fn clt_mdct_forward(
    lookup: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[CeltCoef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let n = lookup.effective_len(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    assert!(input.len() >= overlap + n2);
    assert!(window.len() >= overlap);
    assert!(output.len() >= stride * n2);
    assert!(stride > 0);

    let twiddles = lookup.twiddles(shift);
    let folded = fold_input(input, window, overlap, n2);
    let spectrum = pre_rotate_forward(&folded, twiddles, n4);
    let mut fft_out = vec![KissFftCpx::default(); n4];
    lookup.forward_plan(shift).process(&spectrum, &mut fft_out);
    post_rotate_forward(&fft_out, twiddles, output, stride);
}

pub fn clt_mdct_backward(
    lookup: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[CeltCoef],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let n = lookup.effective_len(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    assert!(input.len() >= stride * n2);
    assert!(window.len() >= overlap);
    let half_overlap = overlap >> 1;
    assert!(output.len() >= overlap);
    assert!(output.len() >= half_overlap + n2);
    assert!(stride > 0);

    let twiddles = lookup.twiddles(shift);
    output.fill(0.0);
    let pre = pre_rotate_backward(input, twiddles, stride);
    let mut fft_out = vec![KissFftCpx::default(); n4];
    lookup.inverse_plan(shift).process(&pre, &mut fft_out);
    post_rotate_backward(&fft_out, twiddles, output, window, overlap);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::f32::consts::PI;

    fn naive_fft(input: &[KissFftCpx], inverse: bool) -> Vec<KissFftCpx> {
        let n = input.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut sum = KissFftCpx::default();
            for (n_idx, sample) in input.iter().enumerate() {
                let angle = 2.0 * PI * (k * n_idx) as f32 / n as f32;
                let (sin_term, cos_term) = if inverse {
                    angle.sin_cos()
                } else {
                    let (s, c) = (-angle).sin_cos();
                    (s, c)
                };
                sum.r += sample.r * cos_term - sample.i * sin_term;
                sum.i += sample.r * sin_term + sample.i * cos_term;
            }
            out.push(sum);
        }
        out
    }

    fn reference_forward(
        lookup: &MdctLookup,
        input: &[f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) -> Vec<f32> {
        let n = lookup.effective_len(shift);
        let n2 = n >> 1;
        let n4 = n >> 2;
        let twiddles = lookup.twiddles(shift);
        let folded = fold_input(input, window, overlap, n2);
        let spectrum = pre_rotate_forward(&folded, twiddles, n4);
        let freq = naive_fft(&spectrum, false);
        let mut out = vec![0.0f32; stride * n2];
        post_rotate_forward(&freq, twiddles, &mut out, stride);
        out
    }

    fn reference_backward(
        lookup: &MdctLookup,
        input: &[f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) -> Vec<f32> {
        let n = lookup.effective_len(shift);
        let n2 = n >> 1;
        let twiddles = lookup.twiddles(shift);
        let pre = pre_rotate_backward(input, twiddles, stride);
        let freq = naive_fft(&pre, true);
        let mut out = vec![0.0f32; overlap.max((overlap >> 1) + n2)];
        post_rotate_backward(&freq, twiddles, &mut out, window, overlap);
        out
    }

    fn make_sine_window(overlap: usize) -> Vec<f32> {
        (0..overlap)
            .map(|i| (PI * (i as f32 + 0.5) / overlap as f32).sin())
            .collect()
    }

    #[test]
    fn forward_matches_reference_for_small_sizes() {
        let sizes = [16usize, 32];
        for &n in &sizes {
            let mdct = MdctLookup::new(n, 0);
            let overlap = n / 2;
            let window = make_sine_window(overlap);
            let mut input = vec![0.0f32; overlap + n];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = (i as f32 * 0.37).sin();
            }
            let mut output = vec![0.0f32; n / 2];
            clt_mdct_forward(&mdct, &input, &mut output, &window, overlap, 0, 1);
            let reference = reference_forward(&mdct, &input, &window, overlap, 0, 1);
            for (lhs, rhs) in output.iter().zip(reference.iter()) {
                assert!((lhs - rhs).abs() < 1e-4, "{} vs {}", lhs, rhs);
            }
        }
    }

    #[test]
    fn backward_matches_reference_for_small_sizes() {
        let sizes = [16usize, 32];
        for &n in &sizes {
            let mdct = MdctLookup::new(n, 0);
            let overlap = n / 2;
            let window = make_sine_window(overlap);
            let mut input = vec![0.0f32; n / 2];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = (i as f32 * 0.19).cos();
            }
            let mut output = vec![0.0f32; overlap.max((overlap >> 1) + n / 2)];
            clt_mdct_backward(&mdct, &input, &mut output, &window, overlap, 0, 1);
            let reference = reference_backward(&mdct, &input, &window, overlap, 0, 1);
            for (lhs, rhs) in output.iter().zip(reference.iter()) {
                assert!((lhs - rhs).abs() < 1e-4, "{} vs {}", lhs, rhs);
            }
        }
    }

    /// Tests MDCT for multiple transform sizes.
    ///
    /// This is an adapted port of `test_unit_mdct.c` that validates the
    /// forward and backward MDCT transforms match the naive reference
    /// implementation across various sizes used by Opus.
    #[test]
    fn mdct_multiple_sizes() {
        // Test power-of-2 sizes up to a reasonable limit
        for &n in &[32usize, 64, 128, 256, 512] {
            let mdct = MdctLookup::new(n, 0);
            let overlap = n / 2;
            let window = make_sine_window(overlap);

            // Tolerance scales with size due to accumulated floating point errors
            let tol = 1e-3 + (n as f32) * 1e-5;

            // Test forward transform
            let mut input = vec![0.0f32; overlap + n];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = ((i as f32 * 0.37) + (i as f32 * 0.13).cos()).sin();
            }
            let mut output = vec![0.0f32; n / 2];
            clt_mdct_forward(&mdct, &input, &mut output, &window, overlap, 0, 1);
            let reference = reference_forward(&mdct, &input, &window, overlap, 0, 1);
            for (j, (lhs, rhs)) in output.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (lhs - rhs).abs() < tol,
                    "forward n={} bin {}: {} vs {}",
                    n,
                    j,
                    lhs,
                    rhs
                );
            }

            // Test backward transform
            let mut freq = vec![0.0f32; n / 2];
            for (i, sample) in freq.iter_mut().enumerate() {
                *sample = (i as f32 * 0.19).cos();
            }
            let mut time = vec![0.0f32; overlap.max((overlap >> 1) + n / 2)];
            clt_mdct_backward(&mdct, &freq, &mut time, &window, overlap, 0, 1);
            let reference_back = reference_backward(&mdct, &freq, &window, overlap, 0, 1);
            for (j, (lhs, rhs)) in time.iter().zip(reference_back.iter()).enumerate() {
                assert!(
                    (lhs - rhs).abs() < tol,
                    "backward n={} sample {}: {} vs {}",
                    n,
                    j,
                    lhs,
                    rhs
                );
            }
        }
    }
}
