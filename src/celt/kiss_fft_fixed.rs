#![allow(dead_code)]

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use core::f64::consts::FRAC_1_SQRT_2;

use super::fixed_arch::Q15_ONE;
use super::fixed_ops::{add32_ovflw, neg32_ovflw, pshr32, qconst16, shl32, shr32, sub32_ovflw};
use super::fixed_ops::{mult16_32_q15, mult16_32_q16};
use super::math::celt_ilog2;
use super::math_fixed::celt_cos_norm;
use super::types::{FixedOpusVal16, FixedOpusVal32};

const MAXFACTORS: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FixedKissFftCpx {
    pub r: FixedOpusVal32,
    pub i: FixedOpusVal32,
}

impl FixedKissFftCpx {
    #[inline]
    pub const fn new(r: FixedOpusVal32, i: FixedOpusVal32) -> Self {
        Self { r, i }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FixedKissTwiddleCpx {
    pub r: FixedOpusVal16,
    pub i: FixedOpusVal16,
}

#[inline]
fn s_mul(a: FixedOpusVal32, b: FixedOpusVal16) -> FixedOpusVal32 {
    mult16_32_q15(b, a)
}

#[inline]
fn s_mul2(a: FixedOpusVal32, b: FixedOpusVal16) -> FixedOpusVal32 {
    mult16_32_q16(b, a)
}

#[inline]
fn c_add(a: FixedKissFftCpx, b: FixedKissFftCpx) -> FixedKissFftCpx {
    FixedKissFftCpx::new(add32_ovflw(a.r, b.r), add32_ovflw(a.i, b.i))
}

#[inline]
fn c_sub(a: FixedKissFftCpx, b: FixedKissFftCpx) -> FixedKissFftCpx {
    FixedKissFftCpx::new(sub32_ovflw(a.r, b.r), sub32_ovflw(a.i, b.i))
}

#[inline]
fn c_mul(a: FixedKissFftCpx, b: FixedKissTwiddleCpx) -> FixedKissFftCpx {
    FixedKissFftCpx::new(
        sub32_ovflw(s_mul(a.r, b.r), s_mul(a.i, b.i)),
        add32_ovflw(s_mul(a.r, b.i), s_mul(a.i, b.r)),
    )
}

#[inline]
fn c_mul_by_scalar(a: FixedKissFftCpx, s: FixedOpusVal16) -> FixedKissFftCpx {
    FixedKissFftCpx::new(s_mul(a.r, s), s_mul(a.i, s))
}

#[inline]
fn half_of(x: FixedOpusVal32) -> FixedOpusVal32 {
    shr32(x, 1)
}

#[derive(Clone, Debug)]
pub struct FixedKissFftState {
    nfft: usize,
    scale: FixedOpusVal16,
    scale_shift: i32,
    shift: Option<usize>,
    factors: Vec<usize>,
    bitrev: Vec<usize>,
    twiddles: Arc<[FixedKissTwiddleCpx]>,
}

impl FixedKissFftState {
    #[must_use]
    pub fn new(nfft: usize) -> Self {
        Self::with_base(nfft, None)
    }

    #[must_use]
    pub fn with_base(nfft: usize, base: Option<&FixedKissFftState>) -> Self {
        assert!(nfft > 0, "FFT size must be non-zero");
        let (twiddles, shift) = if let Some(base_state) = base {
            let mut shift = 0usize;
            while (nfft << shift) < base_state.nfft {
                shift += 1;
            }
            assert_eq!(
                nfft << shift,
                base_state.nfft,
                "base FFT length must be a power-of-two multiple of the requested length"
            );
            (Arc::clone(&base_state.twiddles), Some(shift))
        } else {
            (Arc::<[FixedKissTwiddleCpx]>::from(compute_twiddles(nfft)), None)
        };

        let factors = kf_factor(nfft);
        assert!(
            factors.len() <= 2 * MAXFACTORS,
            "factor buffer overflow: {} entries",
            factors.len()
        );
        let mut bitrev = vec![0usize; nfft];
        compute_bitrev_table(0, &mut bitrev, 1, 1, &factors);

        let scale_shift = celt_ilog2(nfft as i32);
        let scale = if nfft == (1usize << scale_shift) {
            Q15_ONE
        } else {
            let numerator = (1i64 << 30) + (nfft as i64 / 2);
            let mut value = numerator / nfft as i64;
            let adjust = 15 - scale_shift;
            if adjust > 0 {
                value >>= adjust as u32;
            } else if adjust < 0 {
                value <<= (-adjust) as u32;
            }
            value as FixedOpusVal16
        };

        Self {
            nfft,
            scale,
            scale_shift,
            shift,
            factors,
            bitrev,
            twiddles,
        }
    }

    #[inline]
    #[must_use]
    pub fn nfft(&self) -> usize {
        self.nfft
    }

    #[inline]
    #[must_use]
    pub fn bitrev(&self) -> &[usize] {
        &self.bitrev
    }

    #[inline]
    #[must_use]
    pub(crate) fn scale(&self) -> FixedOpusVal16 {
        self.scale
    }

    #[inline]
    #[must_use]
    pub(crate) fn scale_shift(&self) -> i32 {
        self.scale_shift
    }

    pub fn fft(&self, fin: &[FixedKissFftCpx], fout: &mut [FixedKissFftCpx]) {
        assert_eq!(fin.len(), self.nfft, "input length must match FFT size");
        assert_eq!(fout.len(), self.nfft, "output length must match FFT size");
        assert!(
            !core::ptr::eq(fin.as_ptr(), fout.as_mut_ptr()),
            "in-place FFT not supported"
        );

        for (src, &rev) in fin.iter().zip(self.bitrev.iter()) {
            fout[rev] = FixedKissFftCpx::new(s_mul2(src.r, self.scale), s_mul2(src.i, self.scale));
        }
        let downshift = self.scale_shift - 1;
        self.fft_impl(fout, downshift);
    }

    pub fn ifft(&self, fin: &[FixedKissFftCpx], fout: &mut [FixedKissFftCpx]) {
        assert_eq!(fin.len(), self.nfft, "input length must match FFT size");
        assert_eq!(fout.len(), self.nfft, "output length must match FFT size");
        assert!(
            !core::ptr::eq(fin.as_ptr(), fout.as_mut_ptr()),
            "in-place FFT not supported"
        );

        for (src, &rev) in fin.iter().zip(self.bitrev.iter()) {
            fout[rev] = *src;
        }
        for sample in fout.iter_mut() {
            sample.i = neg32_ovflw(sample.i);
        }
        self.fft_impl(fout, 0);
        for sample in fout.iter_mut() {
            sample.i = neg32_ovflw(sample.i);
        }
    }

    pub(crate) fn process(&self, fout: &mut [FixedKissFftCpx], downshift: i32) {
        self.fft_impl(fout, downshift);
    }

    fn fft_impl(&self, fout: &mut [FixedKissFftCpx], mut downshift: i32) {
        let mut fstride = [0usize; MAXFACTORS + 1];
        fstride[0] = 1;
        let mut stages = 0usize;
        loop {
            let p = self.factors[2 * stages];
            let m = self.factors[2 * stages + 1];
            fstride[stages + 1] = fstride[stages] * p;
            stages += 1;
            if m == 1 {
                break;
            }
        }

        let mut m = self.factors[2 * stages - 1];
        let shift = self.shift.unwrap_or(0);
        for stage in (0..stages).rev() {
            let p = self.factors[2 * stage];
            let m2 = if stage != 0 {
                self.factors[2 * stage - 1]
            } else {
                1
            };
            match p {
                2 => {
                    fft_downshift(fout, &mut downshift, 1);
                    kf_bfly2(fout, m, fstride[stage]);
                }
                3 => {
                    fft_downshift(fout, &mut downshift, 2);
                    kf_bfly3(fout, fstride[stage] << shift, self, m, fstride[stage], m2);
                }
                4 => {
                    fft_downshift(fout, &mut downshift, 2);
                    kf_bfly4(fout, fstride[stage] << shift, self, m, fstride[stage], m2);
                }
                5 => {
                    fft_downshift(fout, &mut downshift, 3);
                    kf_bfly5(fout, fstride[stage] << shift, self, m, fstride[stage], m2);
                }
                _ => panic!("unsupported radix {p} in factorisation"),
            }
            m = m2;
        }
        let remaining = downshift;
        fft_downshift(fout, &mut downshift, remaining);
    }
}

fn fft_downshift(
    fout: &mut [FixedKissFftCpx],
    total: &mut i32,
    step: i32,
) {
    if step <= 0 {
        return;
    }
    let shift = step.min(*total).max(0) as u32;
    *total -= shift as i32;
    if shift == 0 {
        return;
    }
    if shift == 1 {
        for value in fout.iter_mut() {
            value.r = shr32(value.r, 1);
            value.i = shr32(value.i, 1);
        }
    } else {
        for value in fout.iter_mut() {
            value.r = pshr32(value.r, shift);
            value.i = pshr32(value.i, shift);
        }
    }
}

#[must_use]
fn compute_twiddles(nfft: usize) -> Vec<FixedKissTwiddleCpx> {
    let mut twiddles = Vec::with_capacity(nfft);
    for i in 0..nfft {
        let phase = -(i as i32);
        let phase = shl32(phase, 17) / nfft as i32;
        let r = celt_cos_norm(phase);
        let i = celt_cos_norm(phase - 32_768);
        twiddles.push(FixedKissTwiddleCpx { r, i });
    }
    twiddles
}

fn compute_bitrev_table(
    fout: usize,
    table: &mut [usize],
    fstride: usize,
    in_stride: usize,
    factors: &[usize],
) {
    let p = factors[0];
    let m = factors[1];
    if m == 1 {
        for j in 0..p {
            table[j * fstride * in_stride] = fout + j;
        }
    } else {
        let mut fout_base = fout;
        let mut table_offset = 0usize;
        for _ in 0..p {
            compute_bitrev_table(
                fout_base,
                &mut table[table_offset..],
                fstride * p,
                in_stride,
                &factors[2..],
            );
            table_offset += fstride * in_stride;
            fout_base += m;
        }
    }
}

fn kf_factor(mut n: usize) -> Vec<usize> {
    let mut factors = [0usize; 2 * MAXFACTORS];
    let mut p = 4usize;
    let mut stages = 0usize;
    let nbak = n;

    loop {
        while !n.is_multiple_of(p) {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p > 32000 || p.saturating_mul(p) > n {
                p = n;
            }
        }
        n /= p;
        assert!(p <= 5, "unsupported FFT radix {p}");
        factors[2 * stages] = p;
        if p == 2 && stages > 1 {
            factors[2 * stages] = 4;
            factors[2] = 2;
        }
        stages += 1;
        if n == 1 {
            break;
        }
    }

    let mut n = nbak;
    for i in 0..(stages / 2) {
        factors.swap(2 * i, 2 * (stages - i - 1));
    }

    let mut out = Vec::with_capacity(2 * stages);
    for i in 0..stages {
        n /= factors[2 * i];
        factors[2 * i + 1] = n;
        out.push(factors[2 * i]);
        out.push(factors[2 * i + 1]);
    }
    out
}

fn kf_bfly2(fout: &mut [FixedKissFftCpx], m: usize, n: usize) {
    if m == 1 {
        for i in 0..n {
            let base = 2 * i;
            let t = fout[base + 1];
            fout[base + 1] = c_sub(fout[base], t);
            fout[base] = c_add(fout[base], t);
        }
    } else {
        debug_assert_eq!(m, 4);
        let tw = qconst16(FRAC_1_SQRT_2, 15);
        for i in 0..n {
            let base = i * 2 * m;
            let t0 = fout[base + 4];
            fout[base + 4] = c_sub(fout[base], t0);
            fout[base] = c_add(fout[base], t0);

            let t1 = FixedKissFftCpx::new(
                s_mul(add32_ovflw(fout[base + 5].r, fout[base + 5].i), tw),
                s_mul(sub32_ovflw(fout[base + 5].i, fout[base + 5].r), tw),
            );
            fout[base + 5] = c_sub(fout[base + 1], t1);
            fout[base + 1] = c_add(fout[base + 1], t1);

            let t2 = FixedKissFftCpx::new(fout[base + 6].i, neg32_ovflw(fout[base + 6].r));
            fout[base + 6] = c_sub(fout[base + 2], t2);
            fout[base + 2] = c_add(fout[base + 2], t2);

            let t3 = FixedKissFftCpx::new(
                s_mul(sub32_ovflw(fout[base + 7].i, fout[base + 7].r), tw),
                s_mul(neg32_ovflw(add32_ovflw(fout[base + 7].i, fout[base + 7].r)), tw),
            );
            fout[base + 7] = c_sub(fout[base + 3], t3);
            fout[base + 3] = c_add(fout[base + 3], t3);
        }
    }
}

fn kf_bfly3(
    fout: &mut [FixedKissFftCpx],
    fstride: usize,
    st: &FixedKissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    let epi3 = FixedKissTwiddleCpx {
        r: 0,
        i: -qconst16(0.866_025_40_f64, 15),
    };
    for i in 0..n {
        let base = i * mm;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;
        for k in 0..m {
            let scratch1 = c_mul(fout[base + m + k], st.twiddles[tw1]);
            let scratch2 = c_mul(fout[base + m2 + k], st.twiddles[tw2]);
            let scratch3 = c_add(scratch1, scratch2);
            let scratch0 = c_sub(scratch1, scratch2);
            tw1 += fstride;
            tw2 += fstride * 2;

            let mut fout_m = FixedKissFftCpx::new(
                sub32_ovflw(fout[base + k].r, half_of(scratch3.r)),
                sub32_ovflw(fout[base + k].i, half_of(scratch3.i)),
            );
            let scratch0 = c_mul_by_scalar(scratch0, epi3.i);
            let fout0 = c_add(fout[base + k], scratch3);

            fout[base + m2 + k] = FixedKissFftCpx::new(
                add32_ovflw(fout_m.r, scratch0.i),
                sub32_ovflw(fout_m.i, scratch0.r),
            );
            fout_m = FixedKissFftCpx::new(
                sub32_ovflw(fout_m.r, scratch0.i),
                add32_ovflw(fout_m.i, scratch0.r),
            );

            fout[base + k] = fout0;
            fout[base + m + k] = fout_m;
        }
    }
}

fn kf_bfly4(
    fout: &mut [FixedKissFftCpx],
    fstride: usize,
    st: &FixedKissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    if m == 1 {
        for i in 0..n {
            let base = i * mm;
            let scratch0 = c_sub(fout[base], fout[base + 2]);
            let scratch1 = c_add(fout[base + 1], fout[base + 3]);
            let scratch1b = c_sub(fout[base + 1], fout[base + 3]);

            let mut fout0 = c_add(fout[base], fout[base + 2]);
            fout[base + 2] = c_sub(fout0, scratch1);
            fout0 = c_add(fout0, scratch1);

            fout[base + 1] = FixedKissFftCpx::new(
                add32_ovflw(scratch0.r, scratch1b.i),
                sub32_ovflw(scratch0.i, scratch1b.r),
            );
            fout[base + 3] = FixedKissFftCpx::new(
                sub32_ovflw(scratch0.r, scratch1b.i),
                add32_ovflw(scratch0.i, scratch1b.r),
            );
            fout[base] = fout0;
        }
    } else {
        let m2 = 2 * m;
        let m3 = 3 * m;
        for i in 0..n {
            let base = i * mm;
            let mut tw1 = 0usize;
            let mut tw2 = 0usize;
            let mut tw3 = 0usize;
            for j in 0..m {
                let scratch0 = c_mul(fout[base + j + m], st.twiddles[tw1]);
                let scratch1 = c_mul(fout[base + j + m2], st.twiddles[tw2]);
                let scratch2 = c_mul(fout[base + j + m3], st.twiddles[tw3]);

                tw1 += fstride;
                tw2 += fstride * 2;
                tw3 += fstride * 3;

                let scratch5 = c_sub(fout[base + j], scratch1);
                let mut fout0 = c_add(fout[base + j], scratch1);
                let scratch3 = c_add(scratch0, scratch2);
                let scratch4 = c_sub(scratch0, scratch2);

                fout[base + j + m2] = c_sub(fout0, scratch3);
                fout0 = c_add(fout0, scratch3);

                let fout_m = FixedKissFftCpx::new(
                    add32_ovflw(scratch5.r, scratch4.i),
                    sub32_ovflw(scratch5.i, scratch4.r),
                );
                let fout_m3 = FixedKissFftCpx::new(
                    sub32_ovflw(scratch5.r, scratch4.i),
                    add32_ovflw(scratch5.i, scratch4.r),
                );

                fout[base + j] = fout0;
                fout[base + j + m] = fout_m;
                fout[base + j + m3] = fout_m3;
            }
        }
    }
}

fn kf_bfly5(
    fout: &mut [FixedKissFftCpx],
    fstride: usize,
    st: &FixedKissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    let ya = FixedKissTwiddleCpx {
        r: qconst16(0.309_016_99_f64, 15),
        i: -qconst16(0.951_056_52_f64, 15),
    };
    let yb = FixedKissTwiddleCpx {
        r: -qconst16(0.809_016_99_f64, 15),
        i: -qconst16(0.587_785_25_f64, 15),
    };
    for i in 0..n {
        let base = i * mm;
        for u in 0..m {
            let scratch0 = fout[base + u];
            let scratch1 = c_mul(fout[base + m + u], st.twiddles[u * fstride]);
            let scratch2 = c_mul(fout[base + 2 * m + u], st.twiddles[2 * u * fstride]);
            let scratch3 = c_mul(fout[base + 3 * m + u], st.twiddles[3 * u * fstride]);
            let scratch4 = c_mul(fout[base + 4 * m + u], st.twiddles[4 * u * fstride]);

            let scratch7 = c_add(scratch1, scratch4);
            let scratch10 = c_sub(scratch1, scratch4);
            let scratch8 = c_add(scratch2, scratch3);
            let scratch9 = c_sub(scratch2, scratch3);

            fout[base + u].r = add32_ovflw(scratch0.r, add32_ovflw(scratch7.r, scratch8.r));
            fout[base + u].i = add32_ovflw(scratch0.i, add32_ovflw(scratch7.i, scratch8.i));

            let scratch5 = FixedKissFftCpx::new(
                add32_ovflw(
                    scratch0.r,
                    add32_ovflw(s_mul(scratch7.r, ya.r), s_mul(scratch8.r, yb.r)),
                ),
                add32_ovflw(
                    scratch0.i,
                    add32_ovflw(s_mul(scratch7.i, ya.r), s_mul(scratch8.i, yb.r)),
                ),
            );
            let scratch6 = FixedKissFftCpx::new(
                add32_ovflw(s_mul(scratch10.i, ya.i), s_mul(scratch9.i, yb.i)),
                neg32_ovflw(add32_ovflw(s_mul(scratch10.r, ya.i), s_mul(scratch9.r, yb.i))),
            );

            fout[base + m + u] = c_sub(scratch5, scratch6);
            fout[base + 4 * m + u] = c_add(scratch5, scratch6);

            let scratch11 = FixedKissFftCpx::new(
                add32_ovflw(
                    scratch0.r,
                    add32_ovflw(s_mul(scratch7.r, yb.r), s_mul(scratch8.r, ya.r)),
                ),
                add32_ovflw(
                    scratch0.i,
                    add32_ovflw(s_mul(scratch7.i, yb.r), s_mul(scratch8.i, ya.r)),
                ),
            );
            let scratch12 = FixedKissFftCpx::new(
                sub32_ovflw(s_mul(scratch9.i, ya.i), s_mul(scratch10.i, yb.i)),
                sub32_ovflw(s_mul(scratch10.r, yb.i), s_mul(scratch9.r, ya.i)),
            );

            fout[base + 2 * m + u] = c_add(scratch11, scratch12);
            fout[base + 3 * m + u] = c_sub(scratch11, scratch12);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *seed
    }

    fn generate_input(nfft: usize, seed: &mut u32) -> Vec<FixedKissFftCpx> {
        let mut buf = Vec::with_capacity(nfft);
        for _ in 0..nfft {
            let r = (lcg(seed) & 0x7fff) as i32 - 16384;
            let i = (lcg(seed) & 0x7fff) as i32 - 16384;
            buf.push(FixedKissFftCpx::new(r, i));
        }
        buf
    }

    #[test]
    fn fixed_fft_impulse_produces_flat_spectrum() {
        let sizes = [32usize, 64, 128, 256];
        for &nfft in &sizes {
            let state = FixedKissFftState::new(nfft);
            let mut input = vec![FixedKissFftCpx::default(); nfft];
            let impulse = 16_384;
            input[0] = FixedKissFftCpx::new(impulse, 0);
            let mut output = vec![FixedKissFftCpx::default(); nfft];
            state.fft(&input, &mut output);
            let expected = impulse / nfft as i32;
            let tol = 2;
            for bin in output {
                assert!(
                    (bin.r - expected).abs() <= tol && bin.i.abs() <= tol,
                    "unexpected bin ({}, {}) for nfft={nfft}",
                    bin.r,
                    bin.i
                );
            }
        }
    }

    #[test]
    fn with_base_matches_standalone_fft() {
        let base = FixedKissFftState::new(256);
        let sizes = [128usize, 64, 32];
        for &nfft in &sizes {
            let state = FixedKissFftState::new(nfft);
            let state_base = FixedKissFftState::with_base(nfft, Some(&base));
            let mut seed = 7u32;
            let input = generate_input(nfft, &mut seed);
            let mut out_new = vec![FixedKissFftCpx::default(); nfft];
            let mut out_base = vec![FixedKissFftCpx::default(); nfft];
            state.fft(&input, &mut out_new);
            state_base.fft(&input, &mut out_base);
            assert_eq!(out_new, out_base);
        }
    }
}
