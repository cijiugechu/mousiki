#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;
use libm::{cosf, floor, sinf, sqrt};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KissFftCpx {
    pub r: f32,
    pub i: f32,
}

impl KissFftCpx {
    #[inline]
    pub const fn new(r: f32, i: f32) -> Self {
        Self { r, i }
    }
}

#[inline]
fn c_add(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx::new(a.r + b.r, a.i + b.i)
}

#[inline]
fn c_sub(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx::new(a.r - b.r, a.i - b.i)
}

#[inline]
fn c_mul(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx::new(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r)
}

#[inline]
fn c_mul_by_scalar(a: KissFftCpx, s: f32) -> KissFftCpx {
    KissFftCpx::new(a.r * s, a.i * s)
}

#[inline]
fn half_of(x: f32) -> f32 {
    0.5 * x
}

const MAXFACTORS: usize = 32;

#[derive(Clone, Debug)]
pub struct MiniKissFft {
    nfft: usize,
    inverse: bool,
    factors: Vec<i32>,
    twiddles: Vec<KissFftCpx>,
}

impl MiniKissFft {
    pub fn new(nfft: usize, inverse_fft: bool) -> Self {
        assert!(nfft > 0, "FFT size must be greater than zero");
        let twiddles = (0..nfft)
            .map(|i| {
                let mut phase = -2.0 * PI * i as f32 / nfft as f32;
                if inverse_fft {
                    phase = -phase;
                }
                KissFftCpx::new(cosf(phase), sinf(phase))
            })
            .collect();
        let factors = kf_factor(nfft);
        assert!(
            factors.len() <= 2 * MAXFACTORS,
            "factor buffer overflow: {} entries",
            factors.len()
        );
        Self {
            nfft,
            inverse: inverse_fft,
            factors,
            twiddles,
        }
    }

    pub fn nfft(&self) -> usize {
        self.nfft
    }

    pub fn is_inverse(&self) -> bool {
        self.inverse
    }

    pub fn process_stride(&self, fin: &[KissFftCpx], fout: &mut [KissFftCpx], in_stride: usize) {
        assert_eq!(fout.len(), self.nfft);
        assert!(in_stride > 0);
        assert!(fin.len() > (self.nfft - 1) * in_stride);
        self.kf_work(fout, fin, 0, 1, in_stride, 0);
    }

    pub fn process(&self, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
        self.process_stride(fin, fout, 1);
    }

    fn kf_work(
        &self,
        fout: &mut [KissFftCpx],
        fin: &[KissFftCpx],
        fin_offset: usize,
        fstride: usize,
        in_stride: usize,
        factors_pos: usize,
    ) {
        let p = self.factors[factors_pos] as usize;
        let m = self.factors[factors_pos + 1] as usize;

        debug_assert_eq!(fout.len(), p * m);

        if m == 1 {
            let mut fin_index = fin_offset;
            for fout_elem in fout.iter_mut().take(p) {
                *fout_elem = fin[fin_index];
                fin_index += fstride * in_stride;
            }
        } else {
            let mut fin_index = fin_offset;
            for chunk in fout.chunks_mut(m).take(p) {
                self.kf_work(
                    chunk,
                    fin,
                    fin_index,
                    fstride * p,
                    in_stride,
                    factors_pos + 2,
                );
                fin_index += fstride * in_stride;
            }
        }

        match p {
            2 => self.kf_bfly2(fout, fstride, m),
            3 => self.kf_bfly3(fout, fstride, m),
            4 => self.kf_bfly4(fout, fstride, m),
            5 => self.kf_bfly5(fout, fstride, m),
            _ => panic!("unsupported radix {p}"),
        }
    }

    fn kf_bfly2(&self, fout: &mut [KissFftCpx], fstride: usize, m: usize) {
        for k in 0..m {
            let tw = self.twiddles[k * fstride];
            let temp = fout[k];
            let t = c_mul(fout[m + k], tw);
            fout[m + k] = c_sub(temp, t);
            fout[k] = c_add(temp, t);
        }
    }

    fn kf_bfly3(&self, fout: &mut [KissFftCpx], fstride: usize, m: usize) {
        let m2 = 2 * m;
        let epi3 = self.twiddles[fstride * m];
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;
        for k in 0..m {
            let scratch1 = c_mul(fout[m + k], self.twiddles[tw1]);
            let scratch2 = c_mul(fout[m2 + k], self.twiddles[tw2]);
            let scratch3 = c_add(scratch1, scratch2);
            let scratch0 = c_sub(scratch1, scratch2);

            tw1 += fstride;
            tw2 += fstride * 2;

            let mut fout_m = KissFftCpx::new(
                fout[k].r - half_of(scratch3.r),
                fout[k].i - half_of(scratch3.i),
            );
            let scratch0 = c_mul_by_scalar(scratch0, epi3.i);
            let fout0 = c_add(fout[k], scratch3);

            let fout_m2 = KissFftCpx::new(fout_m.r + scratch0.i, fout_m.i - scratch0.r);
            fout_m = KissFftCpx::new(fout_m.r - scratch0.i, fout_m.i + scratch0.r);

            fout[k] = fout0;
            fout[m + k] = fout_m;
            fout[m2 + k] = fout_m2;
        }
    }

    fn kf_bfly4(&self, fout: &mut [KissFftCpx], fstride: usize, m: usize) {
        let m2 = 2 * m;
        let m3 = 3 * m;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;
        let mut tw3 = 0usize;
        for k in 0..m {
            let scratch0 = c_mul(fout[m + k], self.twiddles[tw1]);
            let scratch1 = c_mul(fout[m2 + k], self.twiddles[tw2]);
            let scratch2 = c_mul(fout[m3 + k], self.twiddles[tw3]);

            tw1 += fstride;
            tw2 += fstride * 2;
            tw3 += fstride * 3;

            let scratch5 = c_sub(fout[k], scratch1);
            let mut fout0 = c_add(fout[k], scratch1);
            let scratch3 = c_add(scratch0, scratch2);
            let scratch4 = c_sub(scratch0, scratch2);

            fout[m2 + k] = c_sub(fout0, scratch3);
            fout0 = c_add(fout0, scratch3);

            let (fout_m, fout_m3) = if self.inverse {
                (
                    KissFftCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r),
                    KissFftCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r),
                )
            } else {
                (
                    KissFftCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r),
                    KissFftCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r),
                )
            };

            fout[k] = fout0;
            fout[m + k] = fout_m;
            fout[m3 + k] = fout_m3;
        }
    }

    fn kf_bfly5(&self, fout: &mut [KissFftCpx], fstride: usize, m: usize) {
        let ya = self.twiddles[fstride * m];
        let yb = self.twiddles[fstride * 2 * m];
        for u in 0..m {
            let scratch0 = fout[u];
            let scratch1 = c_mul(fout[m + u], self.twiddles[u * fstride]);
            let scratch2 = c_mul(fout[2 * m + u], self.twiddles[2 * u * fstride]);
            let scratch3 = c_mul(fout[3 * m + u], self.twiddles[3 * u * fstride]);
            let scratch4 = c_mul(fout[4 * m + u], self.twiddles[4 * u * fstride]);

            let scratch7 = c_add(scratch1, scratch4);
            let scratch10 = c_sub(scratch1, scratch4);
            let scratch8 = c_add(scratch2, scratch3);
            let scratch9 = c_sub(scratch2, scratch3);

            let fout0 = c_add(c_add(scratch0, scratch7), scratch8);

            let scratch5 = KissFftCpx::new(
                scratch0.r + scratch7.r * ya.r + scratch8.r * yb.r,
                scratch0.i + scratch7.i * ya.r + scratch8.i * yb.r,
            );
            let scratch6 = KissFftCpx::new(
                scratch10.i * ya.i + scratch9.i * yb.i,
                -scratch10.r * ya.i - scratch9.r * yb.i,
            );

            fout[m + u] = c_sub(scratch5, scratch6);
            fout[4 * m + u] = c_add(scratch5, scratch6);

            let scratch11 = KissFftCpx::new(
                scratch0.r + scratch7.r * yb.r + scratch8.r * ya.r,
                scratch0.i + scratch7.i * yb.r + scratch8.i * ya.r,
            );
            let scratch12 = KissFftCpx::new(
                -scratch10.i * yb.i + scratch9.i * ya.i,
                scratch10.r * yb.i - scratch9.r * ya.i,
            );

            fout[2 * m + u] = c_add(scratch11, scratch12);
            fout[3 * m + u] = c_sub(scratch11, scratch12);
            fout[u] = fout0;
        }
    }
}

#[derive(Clone, Debug)]
pub struct MiniKissFftr {
    substate: MiniKissFft,
    pack_buffer: Vec<KissFftCpx>,
    tmpbuf: Vec<KissFftCpx>,
    super_twiddles: Vec<KissFftCpx>,
}

impl MiniKissFftr {
    pub fn new(nfft: usize, inverse_fft: bool) -> Self {
        assert!(nfft.is_multiple_of(2), "Real FFT requires an even length");
        let ncfft = nfft / 2;
        let substate = MiniKissFft::new(ncfft, inverse_fft);
        let pack_buffer = vec![KissFftCpx::default(); ncfft];
        let tmpbuf = vec![KissFftCpx::default(); ncfft];
        let super_twiddles = (0..ncfft / 2)
            .map(|i| {
                let mut phase = -PI * ((i + 1) as f32 / ncfft as f32 + 0.5);
                if inverse_fft {
                    phase = -phase;
                }
                KissFftCpx::new(cosf(phase), sinf(phase))
            })
            .collect();
        Self {
            substate,
            pack_buffer,
            tmpbuf,
            super_twiddles,
        }
    }

    pub fn process(&mut self, timedata: &[f32], freqdata: &mut [KissFftCpx]) {
        let ncfft = self.substate.nfft();
        assert_eq!(timedata.len(), ncfft * 2);
        assert_eq!(freqdata.len(), ncfft + 1);

        for (chunk, packed) in timedata.chunks_exact(2).zip(self.pack_buffer.iter_mut()) {
            *packed = KissFftCpx::new(chunk[0], chunk[1]);
        }

        self.substate.process(&self.pack_buffer, &mut self.tmpbuf);

        let tdc = self.tmpbuf[0];
        freqdata[0] = KissFftCpx::new(tdc.r + tdc.i, 0.0);
        freqdata[ncfft] = KissFftCpx::new(tdc.r - tdc.i, 0.0);

        for k in 1..=ncfft / 2 {
            let fpk = self.tmpbuf[k];
            let fpnk = KissFftCpx::new(self.tmpbuf[ncfft - k].r, -self.tmpbuf[ncfft - k].i);

            let f1k = c_add(fpk, fpnk);
            let f2k = c_sub(fpk, fpnk);
            let tw = c_mul(f2k, self.super_twiddles[k - 1]);

            freqdata[k] = KissFftCpx::new(half_of(f1k.r + tw.r), half_of(f1k.i + tw.i));
            freqdata[ncfft - k] = KissFftCpx::new(half_of(f1k.r - tw.r), half_of(tw.i - f1k.i));
        }
    }
}

fn kf_factor(mut n: usize) -> Vec<i32> {
    let mut factors = Vec::with_capacity(2 * MAXFACTORS);
    let mut p = 4usize;
    let floor_sqrt = floor(sqrt(n as f64)) as usize;
    while n > 1 {
        while !n.is_multiple_of(p) {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p > floor_sqrt {
                p = n;
            }
        }
        n /= p;
        factors.push(p as i32);
        factors.push(n as i32);
    }
    factors
}

#[cfg(test)]
mod tests {
    use super::*;
    use libm::{cosf, sinf};

    fn naive_fft(input: &[KissFftCpx]) -> Vec<KissFftCpx> {
        let n = input.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut sum = KissFftCpx::default();
            for (n_index, sample) in input.iter().enumerate() {
                let angle = -2.0 * PI * (k * n_index) as f32 / n as f32;
                let tw = KissFftCpx::new(cosf(angle), sinf(angle));
                sum = c_add(sum, c_mul(*sample, tw));
            }
            out.push(sum);
        }
        out
    }

    fn approx_eq(a: KissFftCpx, b: KissFftCpx) {
        let eps = 1e-4;
        assert!(
            (a.r - b.r).abs() <= eps,
            "real mismatch: {} vs {}",
            a.r,
            b.r
        );
        assert!(
            (a.i - b.i).abs() <= eps,
            "imag mismatch: {} vs {}",
            a.i,
            b.i
        );
    }

    #[test]
    fn forward_fft_matches_naive() {
        for &n in &[2usize, 3, 4, 5, 6, 8] {
            let input: Vec<_> = (0..n)
                .map(|i| KissFftCpx::new((i + 1) as f32 * 0.25, (i * 2) as f32 * 0.1))
                .collect();
            let naive = naive_fft(&input);
            let fft = MiniKissFft::new(n, false);
            let mut output = vec![KissFftCpx::default(); n];
            fft.process(&input, &mut output);
            for (lhs, rhs) in output.into_iter().zip(naive.into_iter()) {
                approx_eq(lhs, rhs);
            }
        }
    }

    #[test]
    fn inverse_fft_inverts_forward() {
        let n = 8;
        let fft = MiniKissFft::new(n, false);
        let ifft = MiniKissFft::new(n, true);
        let input: Vec<_> = (0..n)
            .map(|i| KissFftCpx::new(sinf(i as f32), cosf(i as f32)))
            .collect();
        let mut freq = vec![KissFftCpx::default(); n];
        fft.process(&input, &mut freq);
        let mut time = vec![KissFftCpx::default(); n];
        ifft.process(&freq, &mut time);
        for (original, reconstructed) in input.iter().zip(time.iter()) {
            approx_eq(
                *original,
                KissFftCpx::new(reconstructed.r / n as f32, reconstructed.i / n as f32),
            );
        }
    }

    #[test]
    fn real_fft_matches_complex_reference() {
        let n = 16;
        let mut fftr = MiniKissFftr::new(n, false);
        let time: Vec<f32> = (0..n).map(|i| sinf(i as f32 / 3.0)).collect();
        let mut packed_freq = vec![KissFftCpx::default(); n / 2 + 1];
        fftr.process(&time, &mut packed_freq);

        let complex_input: Vec<_> = time.iter().map(|&x| KissFftCpx::new(x, 0.0)).collect();
        let reference = naive_fft(&complex_input);
        for k in 0..=n / 2 {
            approx_eq(packed_freq[k], reference[k]);
        }
    }
}
