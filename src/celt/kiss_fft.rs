#![allow(dead_code)]

use core::cell::RefCell;

use alloc::vec;
use alloc::vec::Vec;

use super::mini_kfft::{KissFftCpx, MiniKissFft};

/// Safe Rust implementation of the scalar KISS FFT routines used by CELT.
///
/// The original C version exposes an allocation API that returns a pointer to an
/// opaque state structure. The Rust port keeps the interface ergonomic by
/// wrapping the reusable buffers and twiddle tables inside this struct while
/// relying on interior mutability for the temporary workspace required during a
/// transform. This mirrors the behaviour of the reference implementation where
/// the state itself remains logically immutable once constructed.
#[derive(Clone, Debug)]
pub struct KissFftState {
    nfft: usize,
    scale: f32,
    forward: MiniKissFft,
    inverse: MiniKissFft,
    scratch: RefCell<Vec<KissFftCpx>>,
}

impl KissFftState {
    /// Creates a new FFT state for the provided transform length.
    #[must_use]
    pub fn new(nfft: usize) -> Self {
        assert!(nfft > 0, "FFT size must be non-zero");
        let forward = MiniKissFft::new(nfft, false);
        let inverse = MiniKissFft::new(nfft, true);
        let scratch = RefCell::new(vec![KissFftCpx::default(); nfft]);
        Self {
            nfft,
            scale: 1.0 / nfft as f32,
            forward,
            inverse,
            scratch,
        }
    }

    /// Returns the length of the transform configured for this state.
    #[inline]
    #[must_use]
    pub fn nfft(&self) -> usize {
        self.nfft
    }

    /// Returns the scale applied to inputs when computing the forward FFT.
    #[inline]
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Computes the forward complex FFT.
    ///
    /// The CELT build of KISS FFT normalises the forward transform by the FFT
    /// length. We mirror that behaviour here so that the inverse transform can
    /// recover the original signal without any additional scaling.
    pub fn fft(&self, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
        assert_eq!(fin.len(), self.nfft, "input length must match FFT size");
        assert_eq!(fout.len(), self.nfft, "output length must match FFT size");

        let mut scratch = self.scratch.borrow_mut();
        debug_assert_eq!(scratch.len(), self.nfft);
        for (dst, src) in scratch.iter_mut().zip(fin.iter()) {
            *dst = KissFftCpx::new(src.r * self.scale, src.i * self.scale);
        }
        self.forward.process(&scratch, fout);
    }

    /// Computes the inverse complex FFT.
    ///
    /// The inverse path mirrors the reference implementation: it does not apply
    /// any scaling, meaning that running `ifft(fft(x))` will return the original
    /// sequence.
    pub fn ifft(&self, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
        assert_eq!(fin.len(), self.nfft, "input length must match FFT size");
        assert_eq!(fout.len(), self.nfft, "output length must match FFT size");
        self.inverse.process(fin, fout);
    }
}

/// Convenience wrapper matching the C helper `opus_fft_alloc`.
#[must_use]
pub fn opus_fft_alloc(nfft: usize) -> KissFftState {
    KissFftState::new(nfft)
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
    use libm::{cosf, sinf};

    use super::*;

    fn naive_fft(input: &[KissFftCpx]) -> Vec<KissFftCpx> {
        let n = input.len();
        let mut out = vec![KissFftCpx::default(); n];
        for k in 0..n {
            let mut sum = KissFftCpx::default();
            for (n_index, sample) in input.iter().enumerate() {
                let angle = -2.0 * core::f32::consts::PI * (k * n_index) as f32 / n as f32;
                let tw = KissFftCpx::new(cosf(angle), sinf(angle));
                sum.r += sample.r * tw.r - sample.i * tw.i;
                sum.i += sample.r * tw.i + sample.i * tw.r;
            }
            out[k] = sum;
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
    fn forward_fft_matches_naive_with_normalisation() {
        for &n in &[2usize, 3, 4, 5, 6, 8] {
            let state = KissFftState::new(n);
            let input: Vec<_> = (0..n)
                .map(|i| KissFftCpx::new((i + 1) as f32 * 0.25, (i * 2) as f32 * 0.1))
                .collect();
            let naive = naive_fft(&input);
            let mut output = vec![KissFftCpx::default(); n];
            state.fft(&input, &mut output);
            for (lhs, rhs) in output.into_iter().zip(naive.into_iter()) {
                approx_eq(lhs, KissFftCpx::new(rhs.r / n as f32, rhs.i / n as f32));
            }
        }
    }

    #[test]
    fn inverse_fft_round_trip() {
        let n = 16;
        let state = KissFftState::new(n);
        let input: Vec<_> = (0..n)
            .map(|i| KissFftCpx::new(sinf(i as f32), cosf(i as f32)))
            .collect();
        let mut freq = vec![KissFftCpx::default(); n];
        state.fft(&input, &mut freq);
        let mut time = vec![KissFftCpx::default(); n];
        state.ifft(&freq, &mut time);
        for (original, reconstructed) in input.iter().zip(time.iter()) {
            approx_eq(*original, *reconstructed);
        }
    }
}
