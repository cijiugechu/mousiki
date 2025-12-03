#![allow(dead_code)]

use core::f32::consts::{FRAC_1_SQRT_2, PI};

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use super::mini_kfft::KissFftCpx;
use libm::{cosf, sinf};

const MAXFACTORS: usize = 32;

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

#[derive(Clone, Debug)]
pub struct KissFftState {
    nfft: usize,
    scale: f32,
    shift: Option<usize>,
    factors: Vec<usize>,
    bitrev: Vec<usize>,
    twiddles: Arc<[KissFftCpx]>,
}

impl KissFftState {
    /// Creates a new FFT state for the provided transform length.
    #[must_use]
    pub fn new(nfft: usize) -> Self {
        Self::with_base(nfft, None)
    }

    /// Creates a new FFT state, optionally reusing the twiddle table from a larger base plan.
    #[must_use]
    pub fn with_base(nfft: usize, base: Option<&KissFftState>) -> Self {
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
            (Arc::<[KissFftCpx]>::from(compute_twiddles(nfft)), None)
        };

        let factors = kf_factor(nfft);
        assert!(
            factors.len() <= 2 * MAXFACTORS,
            "factor buffer overflow: {} entries",
            factors.len()
        );
        let mut bitrev = vec![0usize; nfft];
        compute_bitrev_table(0, &mut bitrev, 1, 1, &factors);

        Self {
            nfft,
            scale: 1.0 / nfft as f32,
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
    pub fn scale(&self) -> f32 {
        self.scale
    }

    #[inline]
    #[must_use]
    pub fn bitrev(&self) -> &[usize] {
        &self.bitrev
    }

    /// Computes the forward complex FFT with 1/N scaling.
    pub fn fft(&self, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
        assert_eq!(fin.len(), self.nfft, "input length must match FFT size");
        assert_eq!(fout.len(), self.nfft, "output length must match FFT size");
        assert!(
            !core::ptr::eq(fin.as_ptr(), fout.as_mut_ptr()),
            "in-place FFT not supported"
        );

        for (src, &rev) in fin.iter().zip(self.bitrev.iter()) {
            fout[rev] = KissFftCpx::new(src.r * self.scale, src.i * self.scale);
        }
        self.fft_impl(fout);
    }

    /// Computes the inverse complex FFT (no scaling).
    pub fn ifft(&self, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
        assert_eq!(fin.len(), self.nfft, "input length must match FFT size");
        assert_eq!(fout.len(), self.nfft, "output length must match FFT size");
        assert!(
            !core::ptr::eq(fin.as_ptr(), fout.as_mut_ptr()),
            "in-place FFT not supported"
        );

        for (src, &rev) in fin.iter().zip(self.bitrev.iter()) {
            fout[rev] = KissFftCpx::new(src.r, -src.i);
        }
        self.fft_impl(fout);
        for val in fout.iter_mut() {
            val.i = -val.i;
        }
    }

    fn fft_impl(&self, fout: &mut [KissFftCpx]) {
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
                2 => kf_bfly2(fout, m, fstride[stage]),
                3 => kf_bfly3(
                    fout,
                    fstride[stage] << shift,
                    self,
                    m,
                    fstride[stage],
                    m2,
                ),
                4 => kf_bfly4(
                    fout,
                    fstride[stage] << shift,
                    self,
                    m,
                    fstride[stage],
                    m2,
                ),
                5 => kf_bfly5(
                    fout,
                    fstride[stage] << shift,
                    self,
                    m,
                    fstride[stage],
                    m2,
                ),
                _ => panic!("unsupported radix {p} in factorisation"),
            }
            m = m2;
        }
    }
}

#[must_use]
fn compute_twiddles(nfft: usize) -> Vec<KissFftCpx> {
    (0..nfft)
        .map(|i| {
            let phase = -2.0 * PI * i as f32 / nfft as f32;
            KissFftCpx::new(cosf(phase), sinf(phase))
        })
        .collect()
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

fn kf_bfly2(fout: &mut [KissFftCpx], m: usize, n: usize) {
    if m == 1 {
        for i in 0..n {
            let base = 2 * i;
            let t = fout[base + 1];
            fout[base + 1] = c_sub(fout[base], t);
            fout[base] = c_add(fout[base], t);
        }
    } else {
        debug_assert_eq!(m, 4);
        let tw = FRAC_1_SQRT_2;
        for i in 0..n {
            let base = i * 2 * m;
            let t0 = fout[base + 4];
            fout[base + 4] = c_sub(fout[base], t0);
            fout[base] = c_add(fout[base], t0);

            let mut t1 =
                KissFftCpx::new((fout[base + 5].r + fout[base + 5].i) * tw, (fout[base + 5].i - fout[base + 5].r) * tw);
            fout[base + 5] = c_sub(fout[base + 1], t1);
            fout[base + 1] = c_add(fout[base + 1], t1);

            let t2 = KissFftCpx::new(fout[base + 6].i, -fout[base + 6].r);
            fout[base + 6] = c_sub(fout[base + 2], t2);
            fout[base + 2] = c_add(fout[base + 2], t2);

            t1 = KissFftCpx::new(
                (fout[base + 7].i - fout[base + 7].r) * tw,
                -(fout[base + 7].i + fout[base + 7].r) * tw,
            );
            fout[base + 7] = c_sub(fout[base + 3], t1);
            fout[base + 3] = c_add(fout[base + 3], t1);
        }
    }
}

fn kf_bfly3(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    let epi3 = st.twiddles[fstride * m];
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

            let mut fout_m = KissFftCpx::new(
                fout[base + k].r - half_of(scratch3.r),
                fout[base + k].i - half_of(scratch3.i),
            );
            let scratch0 = c_mul_by_scalar(scratch0, epi3.i);
            let fout0 = c_add(fout[base + k], scratch3);

            fout[base + m2 + k] =
                KissFftCpx::new(fout_m.r + scratch0.i, fout_m.i - scratch0.r);
            fout_m = KissFftCpx::new(fout_m.r - scratch0.i, fout_m.i + scratch0.r);

            fout[base + k] = fout0;
            fout[base + m + k] = fout_m;
        }
    }
}

fn kf_bfly4(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
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

            fout[base + 1] = KissFftCpx::new(scratch0.r + scratch1b.i, scratch0.i - scratch1b.r);
            fout[base + 3] = KissFftCpx::new(scratch0.r - scratch1b.i, scratch0.i + scratch1b.r);
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

                let fout_m = KissFftCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
                let fout_m3 = KissFftCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);

                fout[base + j] = fout0;
                fout[base + j + m] = fout_m;
                fout[base + j + m3] = fout_m3;
            }
        }
    }
}

fn kf_bfly5(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    let ya = st.twiddles[fstride * m];
    let yb = st.twiddles[fstride * 2 * m];
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

            let fout0 = c_add(c_add(scratch0, scratch7), scratch8);

            let scratch5 = KissFftCpx::new(
                scratch0.r + scratch7.r * ya.r + scratch8.r * yb.r,
                scratch0.i + scratch7.i * ya.r + scratch8.i * yb.r,
            );
            let scratch6 = KissFftCpx::new(
                scratch10.i * ya.i + scratch9.i * yb.i,
                -(scratch10.r * ya.i + scratch9.r * yb.i),
            );

            fout[base + m + u] = c_sub(scratch5, scratch6);
            fout[base + 4 * m + u] = c_add(scratch5, scratch6);

            let scratch11 = KissFftCpx::new(
                scratch0.r + scratch7.r * yb.r + scratch8.r * ya.r,
                scratch0.i + scratch7.i * yb.r + scratch8.i * ya.r,
            );
            let scratch12 = KissFftCpx::new(
                -scratch10.i * yb.i + scratch9.i * ya.i,
                scratch10.r * yb.i - scratch9.r * ya.i,
            );

            fout[base + 2 * m + u] = c_add(scratch11, scratch12);
            fout[base + 3 * m + u] = c_sub(scratch11, scratch12);
            fout[base + u] = fout0;
        }
    }
}

/// Convenience wrapper matching the C helper `opus_fft_alloc`.
#[must_use]
pub fn opus_fft_alloc(nfft: usize) -> KissFftState {
    KissFftState::new(nfft)
}

/// Convenience wrapper mirroring `opus_fft_alloc_twiddles`.
#[must_use]
pub fn opus_fft_alloc_twiddles(nfft: usize, base: Option<&KissFftState>) -> KissFftState {
    KissFftState::with_base(nfft, base)
}

/// Computes the forward FFT using a pre-allocated state.
pub fn opus_fft(state: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    state.fft(fin, fout);
}

/// Computes the inverse FFT using a pre-allocated state.
pub fn opus_ifft(state: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    state.ifft(fin, fout);
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use libm::{cos, sin};

    use super::*;

    fn check(fin: &[KissFftCpx], fout: &[KissFftCpx], inverse: bool) -> f64 {
        let nfft = fin.len();
        let mut errpow = 0.0f64;
        let mut sigpow = 0.0f64;
        for bin in 0..nfft {
            let mut ansr = 0.0f64;
            let mut ansi = 0.0f64;
            for (k, sample) in fin.iter().enumerate() {
                let phase = -2.0 * PI as f64 * bin as f64 * k as f64 / nfft as f64;
                let mut re = cos(phase);
                let mut im = sin(phase);
                if inverse {
                    im = -im;
                } else {
                    re /= nfft as f64;
                    im /= nfft as f64;
                }
                ansr += sample.r as f64 * re - sample.i as f64 * im;
                ansi += sample.r as f64 * im + sample.i as f64 * re;
            }
            let difr = ansr - fout[bin].r as f64;
            let difi = ansi - fout[bin].i as f64;
            errpow += difr * difr + difi * difi;
            sigpow += ansr * ansr + ansi * ansi;
        }
        10.0 * (sigpow / errpow).log10()
    }

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        *seed
    }

    fn generate_input(nfft: usize, inverse: bool, seed: &mut u32) -> Vec<KissFftCpx> {
        let mut buf = Vec::with_capacity(nfft);
        for _ in 0..nfft {
            let r = (lcg(seed) & 0x7fff) as f32 - 16384.0;
            let i = (lcg(seed) & 0x7fff) as f32 - 16384.0;
            buf.push(KissFftCpx::new(r * 32768.0, i * 32768.0));
        }
        if inverse {
            let scale = 1.0 / nfft as f32;
            for v in &mut buf {
                v.r *= scale;
                v.i *= scale;
            }
        }
        buf
    }

    fn run_case(nfft: usize, inverse: bool) {
        let mut seed = 1u32;
        let state = KissFftState::new(nfft);
        let input = generate_input(nfft, inverse, &mut seed);
        let mut output = vec![KissFftCpx::default(); nfft];
        if inverse {
            state.ifft(&input, &mut output);
        } else {
            state.fft(&input, &mut output);
        }
        let snr = check(&input, &output, inverse);
        assert!(
            snr >= 60.0,
            "poor SNR {snr:.2} dB for nfft={nfft} inverse={inverse}"
        );
    }

    #[test]
    fn fft_matches_reference_across_sizes() {
        let sizes = [32usize, 128, 256, 36, 50, 60, 120, 240, 480];
        for &nfft in &sizes {
            run_case(nfft, false);
            run_case(nfft, true);
        }
    }
}
