#![allow(dead_code)]

#[cfg(test)]
extern crate std;

#[cfg(test)]
use crate::celt::fft_bitrev_480::FFT_BITREV_480;
use crate::celt::fft_twiddles_48000_960::FFT_TWIDDLES_48000_960;
use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;
use core::f64::consts::PI as PI64;
use libm::{cos, cosf, sin, sinf};

const C_FACTORS_480: [i32; 10] = [5, 96, 3, 32, 4, 8, 2, 4, 4, 1];

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
fn fft_use_fma() -> bool {
    #[cfg(test)]
    {
        let val = std::env::var("CELT_TRACE_KFFT_NO_FMA")
            .ok()
            .unwrap_or_default()
            .to_ascii_lowercase();
        !(val == "1" || val == "true" || val == "yes" || val == "on")
    }
    #[cfg(not(test))]
    {
        true
    }
}

#[inline]
fn c_mul(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    if fft_use_fma() {
        // Keep fused multiply-add to match C's contraction behavior (bit-level parity).
        let r = fused_mul_add(a.r, b.r, -a.i * b.i);
        let i = fused_mul_add(a.r, b.i, a.i * b.r);
        KissFftCpx::new(r, i)
    } else {
        // Match strict mul/add order when FMA is disabled.
        let r = a.r * b.r - a.i * b.i;
        let i = a.r * b.i + a.i * b.r;
        KissFftCpx::new(r, i)
    }
}

#[inline]
fn fused_mul_add(a: f32, b: f32, c: f32) -> f32 {
    // Keep this helper (do not replace with a*b + c); C relies on fused rounding.
    // Regression coverage lives in `fused_mul_add_matches_fma_bits`.
    #[cfg(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "android",
        target_os = "ios"
    ))]
    unsafe {
        unsafe extern "C" {
            fn fmaf(x: f32, y: f32, z: f32) -> f32;
        }
        fmaf(a, b, c)
    }
    #[cfg(not(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "android",
        target_os = "ios"
    )))]
    {
        libm::fmaf(a, b, c)
    }
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
        let twiddles = if nfft == FFT_TWIDDLES_48000_960.len() {
            if inverse_fft {
                FFT_TWIDDLES_48000_960
                    .iter()
                    .map(|c| KissFftCpx::new(c.r, -c.i))
                    .collect()
            } else {
                FFT_TWIDDLES_48000_960.to_vec()
            }
        } else {
            (0..nfft)
                .map(|i| {
                    let mut phase = -2.0 * PI64 * i as f64 / nfft as f64;
                    if inverse_fft {
                        phase = -phase;
                    }
                    KissFftCpx::new(cos(phase) as f32, sin(phase) as f32)
                })
                .collect()
        };
        let factors = if nfft == 480 {
            C_FACTORS_480.to_vec()
        } else {
            kf_factor(nfft)
        };
        assert!(
            factors.len() <= 2 * MAXFACTORS,
            "factor buffer overflow: {} entries",
            factors.len()
        );
        #[cfg(test)]
        twiddle_trace::maybe_dump(nfft, &twiddles);
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
        #[cfg(test)]
        kfft_trace::trace_if_enabled(self, fin);
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
        if m == 1 {
            let mut idx = 0usize;
            while idx + 3 < fout.len() {
                let mut f0 = fout[idx];
                let f1 = fout[idx + 1];
                let f2 = fout[idx + 2];
                let f3 = fout[idx + 3];

                let scratch0 = c_sub(f0, f2);
                f0 = c_add(f0, f2);
                let mut scratch1 = c_add(f1, f3);
                let f2_new = c_sub(f0, scratch1);
                f0 = c_add(f0, scratch1);
                scratch1 = c_sub(f1, f3);

                let f1_new = KissFftCpx::new(scratch0.r + scratch1.i, scratch0.i - scratch1.r);
                let f3_new = KissFftCpx::new(scratch0.r - scratch1.i, scratch0.i + scratch1.r);

                fout[idx] = f0;
                fout[idx + 1] = f1_new;
                fout[idx + 2] = f2_new;
                fout[idx + 3] = f3_new;
                idx += 4;
            }
            return;
        }
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

            // Match C order: scratch0 + (scratch7 + scratch8).
            let scratch78 = c_add(scratch7, scratch8);
            let fout0 = c_add(scratch0, scratch78);

            // Preserve C's sum order with explicit temporaries to avoid
            // optimizer re-association and FMA differences.
            let term7_ya_r = scratch7.r * ya.r;
            let term7_ya_i = scratch7.i * ya.r;
            let term8_yb_r = scratch8.r * yb.r;
            let term8_yb_i = scratch8.i * yb.r;
            let sum78_ya_yb_r = fused_mul_add(scratch7.r, ya.r, term8_yb_r);
            let sum78_ya_yb_i = fused_mul_add(scratch7.i, ya.r, term8_yb_i);
            let scratch5 = KissFftCpx::new(scratch0.r + sum78_ya_yb_r, scratch0.i + sum78_ya_yb_i);

            let term10_yai_r = scratch10.i * ya.i;
            let term10_yai_i = scratch10.r * ya.i;
            let term9_ybi_r = scratch9.i * yb.i;
            let term9_ybi_i = scratch9.r * yb.i;
            // Match C's fused order: fma(scratch10.*, ya.i, term9_ybi).
            let sum10ya_9yb_r = fused_mul_add(scratch10.i, ya.i, term9_ybi_r);
            let sum10ya_9yb_i = fused_mul_add(scratch10.r, ya.i, term9_ybi_i);
            let scratch6 = KissFftCpx::new(sum10ya_9yb_r, -sum10ya_9yb_i);

            fout[m + u] = c_sub(scratch5, scratch6);
            fout[4 * m + u] = c_add(scratch5, scratch6);

            let term7_yb_r = scratch7.r * yb.r;
            let term7_yb_i = scratch7.i * yb.r;
            let term8_ya_r = scratch8.r * ya.r;
            let term8_ya_i = scratch8.i * ya.r;
            // Use fused add here to mirror C compiler contraction in the
            // (a*b + c*d) sums observed to drift by 1 ULP.
            // Use libm::fmaf for consistent fused rounding with the C build.
            let sum78_yb_ya_r = fused_mul_add(scratch7.r, yb.r, term8_ya_r);
            let sum78_yb_ya_i = fused_mul_add(scratch7.i, yb.r, term8_ya_i);
            let scratch11 = KissFftCpx::new(scratch0.r + sum78_yb_ya_r, scratch0.i + sum78_yb_ya_i);

            let term10_ybi_r = scratch10.i * yb.i;
            let term10_ybi_i = scratch10.r * yb.i;
            let term9_yai_r = scratch9.i * ya.i;
            let term9_yai_i = scratch9.r * ya.i;
            let sum9ya_10yb_r = fused_mul_add(scratch9.i, ya.i, -term10_ybi_r);
            let sum9ya_10yb_i = fused_mul_add(scratch10.r, yb.i, -term9_yai_i);
            let scratch12 = KissFftCpx::new(sum9ya_10yb_r, sum9ya_10yb_i);

            fout[2 * m + u] = c_add(scratch11, scratch12);
            fout[3 * m + u] = c_sub(scratch11, scratch12);
            fout[u] = fout0;
        }
    }
}

#[cfg(test)]
pub(crate) mod kfft_trace {
    extern crate std;

    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::env;
    use std::sync::OnceLock;

    use super::{c_add, c_mul, c_mul_by_scalar, c_sub, KissFftCpx, MiniKissFft};
    use crate::celt::fft_bitrev_480::FFT_BITREV_480;

    const C_FACTORS_480: [i16; 10] = [5, 96, 3, 32, 4, 8, 2, 4, 4, 1];

    pub(crate) struct TraceConfig {
        frame: Option<usize>,
        stage: Option<usize>,
        want_bits: bool,
        start: usize,
        count: usize,
        detail: bool,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static FRAME_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);
    static CALL_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);
    static CHANNEL_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);
    static BLOCK_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);
    static TAG_INDEX: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn set_mdct_context(
        frame: usize,
        call: usize,
        channel: usize,
        block: usize,
        tag: usize,
    ) {
        FRAME_INDEX.store(frame, Ordering::Relaxed);
        CALL_INDEX.store(call, Ordering::Relaxed);
        CHANNEL_INDEX.store(channel, Ordering::Relaxed);
        BLOCK_INDEX.store(block, Ordering::Relaxed);
        TAG_INDEX.store(tag, Ordering::Relaxed);
    }

    fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = env_truthy("CELT_TRACE_KFFT_STAGE");
                if !enabled {
                    return None;
                }
                let frame = env::var("CELT_TRACE_KFFT_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let stage = env::var("CELT_TRACE_KFFT_STAGE_INDEX")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let want_bits = env_truthy("CELT_TRACE_KFFT_BITS");
                let start = env::var("CELT_TRACE_KFFT_START")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0);
                let count = env::var("CELT_TRACE_KFFT_COUNT")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(64);
                let detail = env_truthy("CELT_TRACE_KFFT_DETAIL");
                Some(TraceConfig {
                    frame,
                    stage,
                    want_bits,
                    start,
                    count,
                    detail,
                })
            })
            .as_ref()
    }

    fn env_truthy(name: &str) -> bool {
        match env::var(name) {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        }
    }

    fn tag_name(tag: usize) -> &'static str {
        if tag == 1 {
            "mdct2"
        } else {
            "main"
        }
    }

    fn should_dump_stage(stage: usize) -> Option<&'static TraceConfig> {
        let cfg = config()?;
        let frame = FRAME_INDEX.load(Ordering::Relaxed);
        if frame == usize::MAX {
            return None;
        }
        if let Some(target) = cfg.frame {
            if target != frame {
                return None;
            }
        }
        if let Some(target_stage) = cfg.stage {
            if target_stage != stage {
                return None;
            }
        }
        Some(cfg)
    }

    fn should_dump_detail(stage: usize) -> Option<&'static TraceConfig> {
        let cfg = should_dump_stage(stage)?;
        if !cfg.detail {
            return None;
        }
        Some(cfg)
    }

    fn should_dump_detail_idx(stage: usize, idx: usize) -> Option<&'static TraceConfig> {
        let cfg = should_dump_detail(stage)?;
        let start = cfg.start;
        let end = start.saturating_add(cfg.count);
        if idx < start || idx >= end {
            return None;
        }
        Some(cfg)
    }

    pub(crate) fn trace_if_enabled(plan: &MiniKissFft, fin: &[KissFftCpx]) {
        if config().is_none() {
            return;
        }
        if plan.inverse || fin.len() != plan.nfft {
            return;
        }
        if FRAME_INDEX.load(Ordering::Relaxed) == usize::MAX {
            return;
        }

        let mut fout = vec![KissFftCpx::default(); plan.nfft];
        let bitrev = bitrev_for_trace(plan);
        for (src, &dst) in bitrev.iter().enumerate() {
            fout[dst] = fin[src];
        }

        let factors = factors_for_trace(plan);
        let stages = factors.len() / 2;
        let mut fstride = vec![0usize; stages + 1];
        fstride[0] = 1;
        let mut m = 0usize;
        for stage in 0..stages {
            let p = factors[2 * stage];
            m = factors[2 * stage + 1];
            fstride[stage + 1] = fstride[stage] * p;
        }

        let mut stage_index = 0usize;
        let mut m_cur = factors[2 * stages - 1];
        for stage in (0..stages).rev() {
            let p = factors[2 * stage];
            let m2 = if stage != 0 {
                factors[2 * stage - 1]
            } else {
                1
            };
            let n = fstride[stage];
            match p {
                2 => trace_bfly2(&mut fout, plan, n, m_cur, m2),
                3 => trace_bfly3(&mut fout, plan, n, m_cur, m2, stage_index),
                4 => trace_bfly4(&mut fout, plan, n, m_cur, m2, stage_index),
                5 => trace_bfly5(&mut fout, plan, n, m_cur, m2, stage_index),
                _ => break,
            }

            if let Some(cfg) = should_dump_stage(stage_index) {
                dump_stage(cfg, stage_index, p, m_cur, n, &fout);
            }

            m_cur = m2;
            stage_index += 1;
        }
    }

    fn factors_for_trace(plan: &MiniKissFft) -> Vec<usize> {
        if plan.nfft == 480 {
            C_FACTORS_480.iter().map(|&value| value as usize).collect()
        } else {
            plan.factors.iter().map(|&value| value as usize).collect()
        }
    }

    fn bitrev_for_trace(plan: &MiniKissFft) -> Vec<usize> {
        if plan.nfft == 480 {
            return FFT_BITREV_480.iter().map(|&value| value as usize).collect();
        }
        let mut bitrev = vec![0usize; plan.nfft];
        compute_bitrev_table(0, &mut bitrev, 0, 1, 1, &plan.factors);
        bitrev
    }

    fn compute_bitrev_table(
        fout: usize,
        bitrev: &mut [usize],
        start: usize,
        fstride: usize,
        in_stride: usize,
        factors: &[i32],
    ) {
        if factors.len() < 2 {
            return;
        }
        let p = factors[0] as usize;
        let m = factors[1] as usize;
        if m == 1 {
            let mut idx = start;
            for j in 0..p {
                if idx < bitrev.len() {
                    bitrev[idx] = fout + j;
                }
                idx += fstride * in_stride;
            }
        } else {
            let mut idx = start;
            let mut fout_base = fout;
            for _ in 0..p {
                compute_bitrev_table(
                    fout_base,
                    bitrev,
                    idx,
                    fstride * p,
                    in_stride,
                    &factors[2..],
                );
                idx += fstride * in_stride;
                fout_base += m;
            }
        }
    }

    fn dump_stage(
        cfg: &TraceConfig,
        stage: usize,
        p: usize,
        m: usize,
        fstride: usize,
        spectrum: &[KissFftCpx],
    ) {
        let frame = FRAME_INDEX.load(Ordering::Relaxed);
        let call = CALL_INDEX.load(Ordering::Relaxed);
        let channel = CHANNEL_INDEX.load(Ordering::Relaxed);
        let block = BLOCK_INDEX.load(Ordering::Relaxed);
        let tag = TAG_INDEX.load(Ordering::Relaxed);
        let len = spectrum.len();
        let start = cfg.start.min(len);
        let end = start.saturating_add(cfg.count).min(len);

        std::println!(
            "celt_kfft_stage[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].p={p}",
            tag_name(tag)
        );
        std::println!(
            "celt_kfft_stage[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].m={m}",
            tag_name(tag)
        );
        std::println!(
            "celt_kfft_stage[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].fstride={fstride}",
            tag_name(tag)
        );

        for i in start..end {
            let value = spectrum[i];
            std::println!(
                "celt_kfft_stage[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].idx[{i}].r={:.9e}",
                tag_name(tag),
                value.r
            );
            std::println!(
                "celt_kfft_stage[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].idx[{i}].i={:.9e}",
                tag_name(tag),
                value.i
            );
            if cfg.want_bits {
                std::println!(
                    "celt_kfft_stage[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].idx_bits[{i}].r=0x{:08x}",
                    tag_name(tag),
                    value.r.to_bits()
                );
                std::println!(
                    "celt_kfft_stage[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].idx_bits[{i}].i=0x{:08x}",
                    tag_name(tag),
                    value.i.to_bits()
                );
            }
        }
    }

    fn trace_bfly2(fout: &mut [KissFftCpx], plan: &MiniKissFft, n: usize, m: usize, mm: usize) {
        if m == 4 {
            let tw = 0.7071067812f32;
            for group in 0..n {
                let base = group * mm;
                let f0 = fout[base];
                let f2 = fout[base + 4];
                let mut t = f2;
                fout[base + 4] = c_sub(f0, t);
                fout[base] = c_add(f0, t);

                let f1 = fout[base + 1];
                let f3 = fout[base + 5];
                t = KissFftCpx::new((f3.r + f3.i) * tw, (f3.i - f3.r) * tw);
                fout[base + 5] = c_sub(f1, t);
                fout[base + 1] = c_add(f1, t);

                let f2 = fout[base + 2];
                let f4 = fout[base + 6];
                t = KissFftCpx::new(f4.i, -f4.r);
                fout[base + 6] = c_sub(f2, t);
                fout[base + 2] = c_add(f2, t);

                let f3 = fout[base + 3];
                let f5 = fout[base + 7];
                t = KissFftCpx::new((f5.i - f5.r) * tw, -(f5.i + f5.r) * tw);
                fout[base + 7] = c_sub(f3, t);
                fout[base + 3] = c_add(f3, t);
            }
            return;
        }

        let group_size = 2 * m;
        for group in 0..n {
            let base = group * mm;
            let slice = &mut fout[base..base + group_size];
            for k in 0..m {
                let tw = plan.twiddles[k * n];
                let temp = slice[k];
                let t = c_mul(slice[m + k], tw);
                slice[m + k] = c_sub(temp, t);
                slice[k] = c_add(temp, t);
            }
        }
    }

    fn trace_bfly3(
        fout: &mut [KissFftCpx],
        plan: &MiniKissFft,
        n: usize,
        m: usize,
        mm: usize,
        stage: usize,
    ) {
        let group_size = 3 * m;
        for group in 0..n {
            let base = group * mm;
            let slice = &mut fout[base..base + group_size];
            let m2 = 2 * m;
            let epi3 = plan.twiddles[n * m];
            let mut tw1 = 0usize;
            let mut tw2 = 0usize;
            for k in 0..m {
                let idx0 = base + k;
                let idx1 = idx0 + m;
                let idx2 = idx0 + m2;
                let want_detail = should_dump_detail_idx(stage, idx0)
                    .or_else(|| should_dump_detail_idx(stage, idx1))
                    .or_else(|| should_dump_detail_idx(stage, idx2))
                    .is_some();

                let in0 = slice[k];
                let in1 = slice[m + k];
                let in2 = slice[m2 + k];
                let tw1_val = plan.twiddles[tw1];
                let tw2_val = plan.twiddles[tw2];
                let scratch1 = c_mul(in1, tw1_val);
                let scratch2 = c_mul(in2, tw2_val);
                let scratch3 = c_add(scratch1, scratch2);
                let scratch0 = c_sub(scratch1, scratch2);

                tw1 += n;
                tw2 += n * 2;

                let mut fout_m =
                    KissFftCpx::new(in0.r - 0.5 * scratch3.r, in0.i - 0.5 * scratch3.i);
                let scratch0_scaled = c_mul_by_scalar(scratch0, epi3.i);
                let fout0 = c_add(in0, scratch3);

                let fout_m2 =
                    KissFftCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);
                fout_m =
                    KissFftCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);

                if want_detail {
                    dump_detail_cpx("tw1", stage, idx0, tw1_val);
                    dump_detail_cpx("tw2", stage, idx0, tw2_val);
                    dump_detail_cpx("epi3", stage, idx0, epi3);
                    dump_detail_cpx("in0", stage, idx0, in0);
                    dump_detail_cpx("in1", stage, idx0, in1);
                    dump_detail_cpx("in2", stage, idx0, in2);
                    dump_detail_cpx("scratch1", stage, idx0, scratch1);
                    dump_detail_cpx("scratch2", stage, idx0, scratch2);
                    dump_detail_cpx("scratch3", stage, idx0, scratch3);
                    dump_detail_cpx("scratch0", stage, idx0, scratch0);
                    dump_detail_cpx("scratch0_scaled", stage, idx0, scratch0_scaled);
                    dump_detail_cpx("out0", stage, idx0, fout0);
                    dump_detail_cpx("out1", stage, idx0, fout_m);
                    dump_detail_cpx("out2", stage, idx0, fout_m2);
                }

                slice[k] = fout0;
                slice[m + k] = fout_m;
                slice[m2 + k] = fout_m2;
            }
        }
    }

    fn trace_bfly4(
        fout: &mut [KissFftCpx],
        plan: &MiniKissFft,
        n: usize,
        m: usize,
        mm: usize,
        stage: usize,
    ) {
        if m == 1 {
            for group in 0..n {
                let base = group * mm;
                let mut f0 = fout[base];
                let f1 = fout[base + 1];
                let f2 = fout[base + 2];
                let f3 = fout[base + 3];

                let scratch0 = c_sub(f0, f2);
                f0 = c_add(f0, f2);
                let mut scratch1 = c_add(f1, f3);
                let f2_new = c_sub(f0, scratch1);
                f0 = c_add(f0, scratch1);
                scratch1 = c_sub(f1, f3);

                let f1_new = KissFftCpx::new(scratch0.r + scratch1.i, scratch0.i - scratch1.r);
                let f3_new = KissFftCpx::new(scratch0.r - scratch1.i, scratch0.i + scratch1.r);

                fout[base] = f0;
                fout[base + 1] = f1_new;
                fout[base + 2] = f2_new;
                fout[base + 3] = f3_new;
            }
            return;
        }
        let group_size = 4 * m;
        for group in 0..n {
            let base = group * mm;
            let slice = &mut fout[base..base + group_size];
            let m2 = 2 * m;
            let m3 = 3 * m;
            let mut tw1 = 0usize;
            let mut tw2 = 0usize;
            let mut tw3 = 0usize;
            for k in 0..m {
                let idx0 = base + k;
                let idx1 = idx0 + m;
                let idx2 = idx0 + m2;
                let idx3 = idx0 + m3;
                let want_detail = should_dump_detail_idx(stage, idx0)
                    .or_else(|| should_dump_detail_idx(stage, idx1))
                    .or_else(|| should_dump_detail_idx(stage, idx2))
                    .or_else(|| should_dump_detail_idx(stage, idx3))
                    .is_some();

                let in0 = slice[k];
                let in1 = slice[m + k];
                let in2 = slice[m2 + k];
                let in3 = slice[m3 + k];
                let tw1_val = plan.twiddles[tw1];
                let tw2_val = plan.twiddles[tw2];
                let tw3_val = plan.twiddles[tw3];
                let scratch0 = c_mul(in1, tw1_val);
                let scratch1 = c_mul(in2, tw2_val);
                let scratch2 = c_mul(in3, tw3_val);

                tw1 += n;
                tw2 += n * 2;
                tw3 += n * 3;

                let scratch5 = c_sub(in0, scratch1);
                let mut fout0 = c_add(in0, scratch1);
                let scratch3 = c_add(scratch0, scratch2);
                let scratch4 = c_sub(scratch0, scratch2);

                let out2 = c_sub(fout0, scratch3);
                fout0 = c_add(fout0, scratch3);

                let (fout_m, fout_m3) = if plan.inverse {
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

                if want_detail {
                    dump_detail_cpx("tw1", stage, idx0, tw1_val);
                    dump_detail_cpx("tw2", stage, idx0, tw2_val);
                    dump_detail_cpx("tw3", stage, idx0, tw3_val);
                    dump_detail_cpx("in0", stage, idx0, in0);
                    dump_detail_cpx("in1", stage, idx0, in1);
                    dump_detail_cpx("in2", stage, idx0, in2);
                    dump_detail_cpx("in3", stage, idx0, in3);
                    dump_detail_cpx("scratch0", stage, idx0, scratch0);
                    dump_detail_cpx("scratch1", stage, idx0, scratch1);
                    dump_detail_cpx("scratch2", stage, idx0, scratch2);
                    dump_detail_cpx("scratch3", stage, idx0, scratch3);
                    dump_detail_cpx("scratch4", stage, idx0, scratch4);
                    dump_detail_cpx("scratch5", stage, idx0, scratch5);
                    dump_detail_cpx("out0", stage, idx0, fout0);
                    dump_detail_cpx("out1", stage, idx0, fout_m);
                    dump_detail_cpx("out2", stage, idx0, out2);
                    dump_detail_cpx("out3", stage, idx0, fout_m3);
                }

                slice[k] = fout0;
                slice[m + k] = fout_m;
                slice[m2 + k] = out2;
                slice[m3 + k] = fout_m3;
            }
        }
    }

    fn trace_bfly5(
        fout: &mut [KissFftCpx],
        plan: &MiniKissFft,
        n: usize,
        m: usize,
        mm: usize,
        stage: usize,
    ) {
        let group_size = 5 * m;
        let ya = plan.twiddles[n * m];
        let yb = plan.twiddles[n * 2 * m];
        let detail_cfg = should_dump_detail(stage);
        for group in 0..n {
            let base = group * mm;
            let slice = &mut fout[base..base + group_size];
            for u in 0..m {
                let want_detail = detail_cfg
                    .map(|cfg| {
                        let start = cfg.start.min(m);
                        let end = start.saturating_add(cfg.count).min(m);
                        u >= start && u < end
                    })
                    .unwrap_or(false);

                let scratch0 = slice[u];
                let in1 = slice[m + u];
                let in2 = slice[2 * m + u];
                let in3 = slice[3 * m + u];
                let in4 = slice[4 * m + u];
                let tw0 = plan.twiddles[u * n];
                let tw1 = plan.twiddles[2 * u * n];
                let tw2 = plan.twiddles[3 * u * n];
                let tw3 = plan.twiddles[4 * u * n];
                let scratch1 = c_mul(in1, tw0);
                let scratch2 = c_mul(in2, tw1);
                let scratch3 = c_mul(in3, tw2);
                let scratch4 = c_mul(in4, tw3);

                let scratch7 = c_add(scratch1, scratch4);
                let scratch10 = c_sub(scratch1, scratch4);
                let scratch8 = c_add(scratch2, scratch3);
                let scratch9 = c_sub(scratch2, scratch3);

                // Match C order: scratch0 + (scratch7 + scratch8).
                let scratch78 = c_add(scratch7, scratch8);
                let fout0 = c_add(scratch0, scratch78);

                // Preserve C's sum order with explicit temporaries.
                let term7_ya_r = scratch7.r * ya.r;
                let term7_ya_i = scratch7.i * ya.r;
                let term8_yb_r = scratch8.r * yb.r;
                let term8_yb_i = scratch8.i * yb.r;
                let sum78_ya_yb_r = super::fused_mul_add(scratch7.r, ya.r, term8_yb_r);
                let sum78_ya_yb_i = super::fused_mul_add(scratch7.i, ya.r, term8_yb_i);
                let scratch5 = KissFftCpx::new(scratch0.r + sum78_ya_yb_r, scratch0.i + sum78_ya_yb_i);
                let scratch6 = KissFftCpx::new(
                    super::fused_mul_add(scratch10.i, ya.i, scratch9.i * yb.i),
                    -super::fused_mul_add(scratch10.r, ya.i, scratch9.r * yb.i),
                );

                let out1 = c_sub(scratch5, scratch6);
                let out4 = c_add(scratch5, scratch6);

                let term7_yb_r = scratch7.r * yb.r;
                let term7_yb_i = scratch7.i * yb.r;
                let term8_ya_r = scratch8.r * ya.r;
                let term8_ya_i = scratch8.i * ya.r;
                let sum78_yb_ya_r = super::fused_mul_add(scratch7.r, yb.r, term8_ya_r);
                let sum78_yb_ya_i = super::fused_mul_add(scratch7.i, yb.r, term8_ya_i);
                let scratch11 = KissFftCpx::new(scratch0.r + sum78_yb_ya_r, scratch0.i + sum78_yb_ya_i);
                let scratch12 = KissFftCpx::new(
                    super::fused_mul_add(scratch9.i, ya.i, -(scratch10.i * yb.i)),
                    super::fused_mul_add(scratch10.r, yb.i, -(scratch9.r * ya.i)),
                );

                let out2 = c_add(scratch11, scratch12);
                let out3 = c_sub(scratch11, scratch12);

                if want_detail {
                    dump_detail_cpx("tw0", stage, u, tw0);
                    dump_detail_cpx("tw1", stage, u, tw1);
                    dump_detail_cpx("tw2", stage, u, tw2);
                    dump_detail_cpx("tw3", stage, u, tw3);
                    dump_detail_cpx("ya", stage, u, ya);
                    dump_detail_cpx("yb", stage, u, yb);
                    dump_detail_cpx("scratch0", stage, u, scratch0);
                    dump_detail_cpx("in1", stage, u, in1);
                    dump_detail_cpx("in2", stage, u, in2);
                    dump_detail_cpx("in3", stage, u, in3);
                    dump_detail_cpx("in4", stage, u, in4);
                    dump_detail_cpx("scratch1", stage, u, scratch1);
                    dump_detail_cpx("scratch2", stage, u, scratch2);
                    dump_detail_cpx("scratch3", stage, u, scratch3);
                    dump_detail_cpx("scratch4", stage, u, scratch4);
                    dump_detail_cpx("scratch7", stage, u, scratch7);
                    dump_detail_cpx("scratch8", stage, u, scratch8);
                    dump_detail_cpx("scratch9", stage, u, scratch9);
                    dump_detail_cpx("scratch10", stage, u, scratch10);
                    dump_detail_cpx(
                        "term7_ya",
                        stage,
                        u,
                        KissFftCpx::new(scratch7.r * ya.r, scratch7.i * ya.r),
                    );
                    dump_detail_cpx(
                        "term8_yb",
                        stage,
                        u,
                        KissFftCpx::new(scratch8.r * yb.r, scratch8.i * yb.r),
                    );
                    dump_detail_cpx(
                        "term7_yb",
                        stage,
                        u,
                        KissFftCpx::new(scratch7.r * yb.r, scratch7.i * yb.r),
                    );
                    dump_detail_cpx(
                        "term8_ya",
                        stage,
                        u,
                        KissFftCpx::new(scratch8.r * ya.r, scratch8.i * ya.r),
                    );
                    dump_detail_cpx(
                        "term10_yai",
                        stage,
                        u,
                        KissFftCpx::new(scratch10.i * ya.i, scratch10.r * ya.i),
                    );
                    dump_detail_cpx(
                        "term9_ybi",
                        stage,
                        u,
                        KissFftCpx::new(scratch9.i * yb.i, scratch9.r * yb.i),
                    );
                    dump_detail_cpx(
                        "term10_ybi",
                        stage,
                        u,
                        KissFftCpx::new(scratch10.i * yb.i, scratch10.r * yb.i),
                    );
                    dump_detail_cpx(
                        "term9_yai",
                        stage,
                        u,
                        KissFftCpx::new(scratch9.i * ya.i, scratch9.r * ya.i),
                    );
                    let trace_sum78_ya_yb = KissFftCpx::new(
                        scratch7.r * ya.r + scratch8.r * yb.r,
                        scratch7.i * ya.r + scratch8.i * yb.r,
                    );
                    dump_detail_cpx("sum78_ya_yb", stage, u, trace_sum78_ya_yb);
                    dump_detail_cpx(
                        "sum0_7ya",
                        stage,
                        u,
                        KissFftCpx::new(
                            scratch0.r + scratch7.r * ya.r,
                            scratch0.i + scratch7.i * ya.r,
                        ),
                    );
                    dump_detail_cpx(
                        "sum0_8yb",
                        stage,
                        u,
                        KissFftCpx::new(
                            super::fused_mul_add(scratch8.r, yb.r, scratch0.r),
                            super::fused_mul_add(scratch8.i, yb.r, scratch0.i),
                        ),
                    );
                    let trace_sum78_yb_ya = KissFftCpx::new(
                        super::fused_mul_add(scratch7.r, yb.r, scratch8.r * ya.r),
                        super::fused_mul_add(scratch7.i, yb.r, scratch8.i * ya.r),
                    );
                    dump_detail_cpx("sum78_yb_ya", stage, u, trace_sum78_yb_ya);
                    dump_detail_cpx(
                        "sum0_7yb",
                        stage,
                        u,
                        KissFftCpx::new(
                            super::fused_mul_add(scratch7.r, yb.r, scratch0.r),
                            super::fused_mul_add(scratch7.i, yb.r, scratch0.i),
                        ),
                    );
                    dump_detail_cpx(
                        "sum0_8ya",
                        stage,
                        u,
                        KissFftCpx::new(
                            scratch0.r + scratch8.r * ya.r,
                            scratch0.i + scratch8.i * ya.r,
                        ),
                    );
                    let trace_sum10ya_9yb = KissFftCpx::new(
                        super::fused_mul_add(scratch10.i, ya.i, scratch9.i * yb.i),
                        super::fused_mul_add(scratch10.r, ya.i, scratch9.r * yb.i),
                    );
                    dump_detail_cpx("sum10ya_9yb", stage, u, trace_sum10ya_9yb);
                    let trace_sum9ya_10yb = KissFftCpx::new(
                        super::fused_mul_add(scratch9.i, ya.i, -(scratch10.i * yb.i)),
                        super::fused_mul_add(scratch10.r, yb.i, -(scratch9.r * ya.i)),
                    );
                    dump_detail_cpx("sum9ya_10yb", stage, u, trace_sum9ya_10yb);
                    dump_detail_cpx("scratch5", stage, u, scratch5);
                    dump_detail_cpx("scratch6", stage, u, scratch6);
                    dump_detail_cpx("scratch11", stage, u, scratch11);
                    dump_detail_cpx("scratch12", stage, u, scratch12);
                    dump_detail_cpx("out0", stage, u, fout0);
                    dump_detail_cpx("out1", stage, u, out1);
                    dump_detail_cpx("out2", stage, u, out2);
                    dump_detail_cpx("out3", stage, u, out3);
                    dump_detail_cpx("out4", stage, u, out4);
                }

                slice[m + u] = out1;
                slice[4 * m + u] = out4;
                slice[2 * m + u] = out2;
                slice[3 * m + u] = out3;
                slice[u] = fout0;
            }
        }
    }

    fn dump_detail_cpx(name: &str, stage: usize, u: usize, value: KissFftCpx) {
        let frame = FRAME_INDEX.load(Ordering::Relaxed);
        let call = CALL_INDEX.load(Ordering::Relaxed);
        let channel = CHANNEL_INDEX.load(Ordering::Relaxed);
        let block = BLOCK_INDEX.load(Ordering::Relaxed);
        let tag = TAG_INDEX.load(Ordering::Relaxed);
        std::println!(
            "celt_kfft_detail[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].u[{u}].{name}.r={:.9e}",
            tag_name(tag),
            value.r
        );
        std::println!(
            "celt_kfft_detail[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].u[{u}].{name}.i={:.9e}",
            tag_name(tag),
            value.i
        );
        if let Some(cfg) = config() {
            if cfg.want_bits {
                std::println!(
                    "celt_kfft_detail[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].u[{u}].{name}_bits.r=0x{:08x}",
                    tag_name(tag),
                    value.r.to_bits()
                );
                std::println!(
                    "celt_kfft_detail[{frame}].{}.call[{call}].ch[{channel}].block[{block}].stage[{stage}].u[{u}].{name}_bits.i=0x{:08x}",
                    tag_name(tag),
                    value.i.to_bits()
                );
            }
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
                let mut phase = -PI64 * ((i + 1) as f64 / ncfft as f64 + 0.5);
                if inverse_fft {
                    phase = -phase;
                }
                KissFftCpx::new(cos(phase) as f32, sin(phase) as f32)
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
    let floor_sqrt = floor_sqrt_usize(n);
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

fn floor_sqrt_usize(n: usize) -> usize {
    if n <= 1 {
        return n;
    }

    let mut low = 1usize;
    let mut high = n;
    let mut best = 1usize;

    while low <= high {
        let mid = low + (high - low) / 2;
        match mid.checked_mul(mid) {
            Some(prod) if prod == n => return mid,
            Some(prod) if prod < n => {
                best = mid;
                low = mid + 1;
            }
            _ => {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            }
        }
    }

    best
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

    #[test]
    fn fused_mul_add_matches_fma_bits() {
        // Regression: ensure fused rounding stays intact for C parity.
        // These bits were found to differ between fma and a*b+c on this target.
        let a = f32::from_bits(0x3f000003);
        let b = f32::from_bits(0x3f800005);
        let c = f32::from_bits(0x3f000015);
        let expected = 0x3f80000f;
        let naive = (a * b + c).to_bits();
        let fused = fused_mul_add(a, b, c).to_bits();
        assert_ne!(
            naive, expected,
            "naive path unexpectedly matches fused result"
        );
        assert_eq!(fused, expected, "fused_mul_add must keep fma rounding");
    }
}

#[cfg(test)]
mod twiddle_trace {
    extern crate std;

    use std::env;

    use super::KissFftCpx;

    pub(crate) fn maybe_dump(nfft: usize, twiddles: &[KissFftCpx]) {
        let enabled = match env::var("CELT_TRACE_TWIDDLES") {
            Ok(value) => !value.is_empty() && value != "0",
            Err(_) => false,
        };
        if !enabled {
            return;
        }
        let match_nfft = env::var("CELT_TRACE_TWIDDLES_NFFT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map_or(true, |target| target == nfft);
        if !match_nfft {
            return;
        }
        for (i, tw) in twiddles.iter().enumerate() {
            std::println!("celt_twiddle[{nfft}].idx[{i}].r={:.9e}", tw.r as f64);
            std::println!("celt_twiddle[{nfft}].idx[{i}].i={:.9e}", tw.i as f64);
            std::println!(
                "celt_twiddle[{nfft}].idx[{i}].r_bits=0x{:08x}",
                tw.r.to_bits()
            );
            std::println!(
                "celt_twiddle[{nfft}].idx[{i}].i_bits=0x{:08x}",
                tw.i.to_bits()
            );
        }
    }
}
