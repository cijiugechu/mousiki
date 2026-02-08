#![allow(dead_code)]

use core::f32::consts::FRAC_1_SQRT_2;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use super::fft_twiddles_48000_960::FFT_TWIDDLES_48000_960;
use super::math::mul_add_f32;
use super::mini_kfft::KissFftCpx;

const MAXFACTORS: usize = 32;
// Keep the literal to match C twiddle generation; avoid consts::PI to preserve bits.
#[allow(clippy::approx_constant)]
const PI_F64: f64 = 3.14159265358979323846264338327;

#[cfg(test)]
mod fft_stage_trace {
    extern crate std;

    use std::env;
    use std::format;
    use std::string::String;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::OnceLock;
    use std::vec::Vec;

    use super::KissFftCpx;

    pub(crate) struct TraceConfig {
        bins: Vec<usize>,
        all_bins: bool,
        frame_filter: Option<usize>,
        manual: bool,
        bfly_stage: Option<usize>,
        bfly_index: Option<usize>,
        bfly_hex: bool,
        twiddle_dump: bool,
        bitrev_src: bool,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static TRACE_FRAME_INDEX: AtomicUsize = AtomicUsize::new(0);
    static CURRENT_FRAME: AtomicUsize = AtomicUsize::new(usize::MAX);
    static OVERRIDE_FRAME: AtomicUsize = AtomicUsize::new(usize::MAX);
    static OVERRIDE_PENDING: AtomicBool = AtomicBool::new(false);
    static BFLY_ENABLED: AtomicBool = AtomicBool::new(false);
    static BFLY_INDEX: AtomicUsize = AtomicUsize::new(0);
    static BFLY_STAGE: AtomicUsize = AtomicUsize::new(usize::MAX);
    static BFLY_HEX: AtomicBool = AtomicBool::new(false);
    static BFLY_DETAIL_INDEX: AtomicUsize = AtomicUsize::new(4);
    static TWIDDLE_DUMPED: AtomicBool = AtomicBool::new(false);

    pub(crate) fn set_frame_override(frame_idx: usize) {
        OVERRIDE_FRAME.store(frame_idx, Ordering::Relaxed);
        OVERRIDE_PENDING.store(true, Ordering::Relaxed);
    }

    pub(crate) fn begin_call() {
        let Some(cfg) = config() else {
            return;
        };
        if OVERRIDE_PENDING.swap(false, Ordering::Relaxed) {
            let frame = OVERRIDE_FRAME.load(Ordering::Relaxed);
            CURRENT_FRAME.store(frame, Ordering::Relaxed);
            return;
        }
        if cfg.manual {
            CURRENT_FRAME.store(usize::MAX, Ordering::Relaxed);
        } else {
            let frame = TRACE_FRAME_INDEX.fetch_add(1, Ordering::Relaxed);
            CURRENT_FRAME.store(frame, Ordering::Relaxed);
        }
    }

    pub(crate) fn end_call() {
        CURRENT_FRAME.store(usize::MAX, Ordering::Relaxed);
    }

    fn current_frame() -> Option<usize> {
        let value = CURRENT_FRAME.load(Ordering::Relaxed);
        if value == usize::MAX {
            None
        } else {
            Some(value)
        }
    }

    pub(crate) fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = env::var_os("ANALYSIS_TRACE_STAGE").is_some()
                    || env::var_os("KISS_FFT_TRACE_STAGE").is_some()
                    || env::var_os("ANALYSIS_TRACE_BFLY_STAGE").is_some()
                    || env::var_os("KISS_FFT_TRACE_BFLY_STAGE").is_some()
                    || env::var_os("ANALYSIS_TRACE_BITREV_SRC").is_some()
                    || env::var_os("KISS_FFT_TRACE_BITREV_SRC").is_some();
                let twiddle_dump = env::var_os("ANALYSIS_TRACE_TWIDDLES").is_some()
                    || env::var_os("KISS_FFT_TRACE_TWIDDLES").is_some();
                if !enabled && !twiddle_dump {
                    return None;
                }
                let bins = match env::var("ANALYSIS_TRACE_BINS") {
                    Ok(value) => parse_bins(&value),
                    Err(_) => default_bins(),
                };
                let frame_filter = env::var("ANALYSIS_TRACE_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let bfly_stage = env::var("ANALYSIS_TRACE_BFLY_STAGE")
                    .or_else(|_| env::var("KISS_FFT_TRACE_BFLY_STAGE"))
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let bfly_index = env::var("ANALYSIS_TRACE_BFLY_INDEX")
                    .or_else(|_| env::var("KISS_FFT_TRACE_BFLY_INDEX"))
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let bfly_hex = env::var_os("ANALYSIS_TRACE_BFLY_HEX").is_some()
                    || env::var_os("KISS_FFT_TRACE_BFLY_HEX").is_some();
                let manual = env::var_os("ANALYSIS_TRACE_STAGE_MANUAL").is_some()
                    || env::var_os("KISS_FFT_TRACE_MANUAL").is_some();
                let bitrev_src = env::var_os("ANALYSIS_TRACE_BITREV_SRC").is_some()
                    || env::var_os("KISS_FFT_TRACE_BITREV_SRC").is_some();
                Some(TraceConfig {
                    bins: bins.bins,
                    all_bins: bins.all_bins,
                    frame_filter,
                    manual,
                    bfly_stage,
                    bfly_index,
                    bfly_hex,
                    twiddle_dump,
                    bitrev_src,
                })
            })
            .as_ref()
    }

    pub(crate) fn dump_bitrev_source(fin: &[KissFftCpx], bitrev: &[usize], scale: f32) {
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let Some(cfg) = config() else {
            return;
        };
        if !cfg.bitrev_src || !should_trace_frame(cfg, frame_idx) {
            return;
        }
        for (src_idx, &dest) in bitrev.iter().enumerate() {
            if !should_trace_bin(cfg, dest) {
                continue;
            }
            let src = fin[src_idx];
            let scaled = KissFftCpx::new(src.r * scale, src.i * scale);
            std::println!("fft_stage[{frame_idx}].bitrev_src.bin[{dest}].src={src_idx}");
            std::println!(
                "fft_stage[{frame_idx}].bitrev_src.bin[{dest}].input.r={}",
                format_value(src.r as f64)
            );
            std::println!(
                "fft_stage[{frame_idx}].bitrev_src.bin[{dest}].input.i={}",
                format_value(src.i as f64)
            );
            std::println!(
                "fft_stage[{frame_idx}].bitrev_src.bin[{dest}].input_bits.r=0x{:08x}",
                src.r.to_bits()
            );
            std::println!(
                "fft_stage[{frame_idx}].bitrev_src.bin[{dest}].input_bits.i=0x{:08x}",
                src.i.to_bits()
            );
            std::println!(
                "fft_stage[{frame_idx}].bitrev_src.bin[{dest}].scaled.r={}",
                format_value(scaled.r as f64)
            );
            std::println!(
                "fft_stage[{frame_idx}].bitrev_src.bin[{dest}].scaled.i={}",
                format_value(scaled.i as f64)
            );
            std::println!(
                "fft_stage[{frame_idx}].bitrev_src.bin[{dest}].scaled_bits.r=0x{:08x}",
                scaled.r.to_bits()
            );
            std::println!(
                "fft_stage[{frame_idx}].bitrev_src.bin[{dest}].scaled_bits.i=0x{:08x}",
                scaled.i.to_bits()
            );
        }
    }

    pub(crate) fn dump_bitrev(buf: &[KissFftCpx]) {
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let Some(cfg) = config() else {
            return;
        };
        if !should_trace_frame(cfg, frame_idx) {
            return;
        }
        for bin in 0..buf.len() {
            if !should_trace_bin(cfg, bin) {
                continue;
            }
            std::println!(
                "fft_stage[{frame_idx}].bitrev.bin[{bin}].r={}",
                format_value(buf[bin].r as f64)
            );
            std::println!(
                "fft_stage[{frame_idx}].bitrev.bin[{bin}].i={}",
                format_value(buf[bin].i as f64)
            );
        }
    }

    pub(crate) fn dump_twiddles(nfft: usize, twiddles: &[KissFftCpx]) {
        let Some(cfg) = config() else {
            return;
        };
        if !cfg.twiddle_dump {
            return;
        }
        if TWIDDLE_DUMPED.swap(true, Ordering::Relaxed) {
            return;
        }
        let count = nfft.min(twiddles.len());
        std::println!("fft_twiddles.nfft={count}");
        for (idx, value) in twiddles.iter().take(count).enumerate() {
            let r = value.r.to_bits();
            let i = value.i.to_bits();
            std::println!("fft_twiddles[{idx}].bits.r=0x{r:08x}");
            std::println!("fft_twiddles[{idx}].bits.i=0x{i:08x}");
        }
    }

    pub(crate) fn dump_twiddle_phase(index: usize, phase: f64) {
        let Some(cfg) = config() else {
            return;
        };
        if !cfg.twiddle_dump || (index != 120 && index != 240) {
            return;
        }
        let bits = phase.to_bits();
        std::println!("fft_twiddle_phase[{index}].bits=0x{bits:016x}");
    }

    pub(crate) fn dump_stage(stage: usize, buf: &[KissFftCpx]) {
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let Some(cfg) = config() else {
            return;
        };
        if !should_trace_frame(cfg, frame_idx) {
            return;
        }
        for bin in 0..buf.len() {
            if !should_trace_bin(cfg, bin) {
                continue;
            }
            std::println!(
                "fft_stage[{frame_idx}].stage[{stage}].bin[{bin}].r={}",
                format_value(buf[bin].r as f64)
            );
            std::println!(
                "fft_stage[{frame_idx}].stage[{stage}].bin[{bin}].i={}",
                format_value(buf[bin].i as f64)
            );
        }
    }

    pub(crate) fn begin_bfly_stage(stage: usize) {
        let Some(cfg) = config() else {
            return;
        };
        let Some(target) = cfg.bfly_stage else {
            return;
        };
        let Some(frame_idx) = current_frame() else {
            return;
        };
        if target != stage || !should_trace_frame(cfg, frame_idx) {
            return;
        }
        BFLY_STAGE.store(stage, Ordering::Relaxed);
        BFLY_INDEX.store(0, Ordering::Relaxed);
        BFLY_ENABLED.store(true, Ordering::Relaxed);
        BFLY_HEX.store(cfg.bfly_hex, Ordering::Relaxed);
        BFLY_DETAIL_INDEX.store(cfg.bfly_index.unwrap_or(4), Ordering::Relaxed);
    }

    pub(crate) fn end_bfly_stage(stage: usize) {
        if BFLY_ENABLED.load(Ordering::Relaxed)
            && BFLY_STAGE.load(Ordering::Relaxed) == stage
        {
            BFLY_ENABLED.store(false, Ordering::Relaxed);
            BFLY_HEX.store(false, Ordering::Relaxed);
            BFLY_DETAIL_INDEX.store(4, Ordering::Relaxed);
        }
    }

    pub(crate) fn bfly_active() -> bool {
        BFLY_ENABLED.load(Ordering::Relaxed)
    }

    pub(crate) fn current_bfly_index() -> Option<usize> {
        if !BFLY_ENABLED.load(Ordering::Relaxed) {
            return None;
        }
        Some(BFLY_INDEX.load(Ordering::Relaxed))
    }

    pub(crate) fn bfly_detail_index() -> usize {
        BFLY_DETAIL_INDEX.load(Ordering::Relaxed)
    }

    pub(crate) fn dump_bfly_value(bfly_idx: usize, label: &str, value: KissFftCpx) {
        if !BFLY_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let stage = BFLY_STAGE.load(Ordering::Relaxed);
        std::println!(
            "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].{label}.r={}",
            format_value(value.r as f64)
        );
        std::println!(
            "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].{label}.i={}",
            format_value(value.i as f64)
        );
    }

    pub(crate) fn dump_bfly_bits(bfly_idx: usize, label: &str, value: KissFftCpx) {
        if !BFLY_ENABLED.load(Ordering::Relaxed) || !BFLY_HEX.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let stage = BFLY_STAGE.load(Ordering::Relaxed);
        std::println!(
            "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].{label}.bits.r=0x{bits:08x}",
            bits = value.r.to_bits()
        );
        std::println!(
            "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].{label}.bits.i=0x{bits:08x}",
            bits = value.i.to_bits()
        );
    }

    pub(crate) fn dump_bfly_scalar(bfly_idx: usize, label: &str, value: f32) {
        if !BFLY_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let stage = BFLY_STAGE.load(Ordering::Relaxed);
        std::println!(
            "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].{label}={}",
            format_value(value as f64)
        );
    }

    pub(crate) fn dump_bfly_scalar_bits(bfly_idx: usize, label: &str, value: f32) {
        if !BFLY_ENABLED.load(Ordering::Relaxed) || !BFLY_HEX.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let stage = BFLY_STAGE.load(Ordering::Relaxed);
        std::println!(
            "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].{label}_bits=0x{bits:08x}",
            bits = value.to_bits()
        );
    }

    pub(crate) fn dump_bfly_indices(bfly_idx: usize, indices: [usize; 5]) {
        if !BFLY_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let stage = BFLY_STAGE.load(Ordering::Relaxed);
        for (slot, idx) in indices.into_iter().enumerate() {
            std::println!(
                "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].idx[{slot}]={idx}"
            );
        }
    }

    pub(crate) fn dump_bfly_twiddle_index(bfly_idx: usize, label: &str, index: usize) {
        if !BFLY_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let stage = BFLY_STAGE.load(Ordering::Relaxed);
        std::println!(
            "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].{label}={index}"
        );
    }

    pub(crate) fn dump_bfly(before: &[KissFftCpx], after: &[KissFftCpx]) {
        if !BFLY_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame_idx) = current_frame() else {
            return;
        };
        let stage = BFLY_STAGE.load(Ordering::Relaxed);
        let bfly_idx = BFLY_INDEX.fetch_add(1, Ordering::Relaxed);
        for (idx, value) in before.iter().enumerate() {
            std::println!(
                "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].in[{idx}].r={}",
                format_value(value.r as f64)
            );
            std::println!(
                "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].in[{idx}].i={}",
                format_value(value.i as f64)
            );
            let out = after[idx];
            std::println!(
                "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].out[{idx}].r={}",
                format_value(out.r as f64)
            );
            std::println!(
                "fft_stage[{frame_idx}].stage[{stage}].bfly[{bfly_idx}].out[{idx}].i={}",
                format_value(out.i as f64)
            );
        }
    }

    fn format_value(value: f64) -> String {
        let raw = format!("{:.9e}", value);
        let Some(pos) = raw.find('e') else {
            return raw;
        };
        let (mant, exp) = raw.split_at(pos);
        let mut digits = String::from(&exp[1..]);
        let mut sign = '+';
        if let Some(rest) = digits.strip_prefix('-') {
            sign = '-';
            digits = String::from(rest);
        } else if let Some(rest) = digits.strip_prefix('+') {
            sign = '+';
            digits = String::from(rest);
        }
        if digits.len() == 1 {
            digits.insert(0, '0');
        }
        format!("{mant}e{sign}{digits}")
    }

    fn should_trace_bin(cfg: &TraceConfig, bin: usize) -> bool {
        if cfg.all_bins {
            return true;
        }
        cfg.bins.iter().any(|&value| value == bin)
    }

    fn should_trace_frame(cfg: &TraceConfig, frame_idx: usize) -> bool {
        cfg.frame_filter.map_or(true, |value| value == frame_idx)
    }

    struct BinConfig {
        bins: Vec<usize>,
        all_bins: bool,
    }

    fn default_bins() -> BinConfig {
        let mut bins = Vec::new();
        bins.push(1);
        bins.push(61);
        BinConfig {
            bins,
            all_bins: false,
        }
    }

    fn parse_bins(value: &str) -> BinConfig {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return default_bins();
        }
        if trimmed.eq_ignore_ascii_case("all") {
            return BinConfig {
                bins: Vec::new(),
                all_bins: true,
            };
        }
        let mut bins = Vec::new();
        for token in trimmed.split(|c| c == ',' || c == ' ' || c == '\t') {
            if token.is_empty() {
                continue;
            }
            if let Ok(bin) = token.parse::<usize>() {
                if bin < 480 {
                    bins.push(bin);
                }
            }
        }
        if bins.is_empty() {
            default_bins()
        } else {
            BinConfig {
                bins,
                all_bins: false,
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn set_fft_trace_frame(frame_idx: usize) {
    fft_stage_trace::set_frame_override(frame_idx);
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
    // Use mul_add to mirror C's fused multiply-add behavior in twiddle multiplies.
    let real = mul_add_f32(a.r, b.r, -(a.i * b.i));
    let imag = mul_add_f32(a.r, b.i, a.i * b.r);
    KissFftCpx::new(real, imag)
}

#[inline]
fn c_mul_by_scalar(a: KissFftCpx, s: f32) -> KissFftCpx {
    KissFftCpx::new(a.r * s, a.i * s)
}

#[inline]
fn half_of(x: f32) -> f32 {
    0.5 * x
}

#[inline]
fn fft_scale(nfft: usize) -> f32 {
    // Match the C static-mode scale literals to keep FFT output bit-identical.
    match nfft {
        60 => 0.016_666_667,
        120 => 0.008_333_333,
        240 => 0.004_166_667,
        480 => 0.002_083_333,
        _ => 1.0 / nfft as f32,
    }
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
            let twiddles = if nfft == 480 {
                Arc::<[KissFftCpx]>::from(&FFT_TWIDDLES_48000_960[..])
            } else {
                Arc::<[KissFftCpx]>::from(compute_twiddles(nfft))
            };
            (twiddles, None)
        };
        #[cfg(test)]
        if fft_stage_trace::config().is_some() {
            fft_stage_trace::dump_twiddles(nfft, twiddles.as_ref());
        }

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
            scale: fft_scale(nfft),
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

        #[cfg(test)]
        if fft_stage_trace::config().is_some() {
            fft_stage_trace::begin_call();
        }

        #[cfg(test)]
        if fft_stage_trace::config().is_some() {
            fft_stage_trace::dump_bitrev_source(fin, &self.bitrev, self.scale);
        }

        for (src, &rev) in fin.iter().zip(self.bitrev.iter()) {
            fout[rev] = KissFftCpx::new(src.r * self.scale, src.i * self.scale);
        }

        #[cfg(test)]
        if fft_stage_trace::config().is_some() {
            fft_stage_trace::dump_bitrev(fout);
        }

        self.fft_impl(fout);

        #[cfg(test)]
        if fft_stage_trace::config().is_some() {
            fft_stage_trace::end_call();
        }
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
        #[cfg(test)]
        let total_stages = stages;
        for stage in (0..stages).rev() {
            let p = self.factors[2 * stage];
            let m2 = if stage != 0 {
                self.factors[2 * stage - 1]
            } else {
                1
            };
            #[cfg(test)]
            let applied_stage = total_stages - 1 - stage;
            #[cfg(test)]
            if fft_stage_trace::config().is_some() {
                fft_stage_trace::begin_bfly_stage(applied_stage);
            }
            match p {
                2 => kf_bfly2(fout, m, fstride[stage]),
                3 => kf_bfly3(fout, fstride[stage] << shift, self, m, fstride[stage], m2),
                4 => kf_bfly4(fout, fstride[stage] << shift, self, m, fstride[stage], m2),
                5 => kf_bfly5(fout, fstride[stage] << shift, self, m, fstride[stage], m2),
                _ => panic!("unsupported radix {p} in factorisation"),
            }
            #[cfg(test)]
            if fft_stage_trace::config().is_some() {
                fft_stage_trace::end_bfly_stage(applied_stage);
                fft_stage_trace::dump_stage(applied_stage, fout);
            }
            m = m2;
        }
    }
}

#[must_use]
fn twiddle_phase(index: usize, nfft: usize) -> f64 {
    (-2.0 * PI_F64 / nfft as f64) * index as f64
}

fn compute_twiddles(nfft: usize) -> Vec<KissFftCpx> {
    (0..nfft)
        .map(|i| {
            // Match opus-c: compute twiddles in f64, then cast to f32.
            let phase = twiddle_phase(i, nfft);
            #[cfg(test)]
            if fft_stage_trace::config().is_some() {
                fft_stage_trace::dump_twiddle_phase(i, phase);
            }
            let (re, im) = twiddle_math::cos_sin(phase);
            KissFftCpx::new(re as f32, im as f32)
        })
        .collect()
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod twiddle_math {
    #[link(name = "System")]
    unsafe extern "C" {
        fn sin(x: f64) -> f64;
        fn cos(x: f64) -> f64;
    }

    pub(super) fn cos_sin(phase: f64) -> (f64, f64) {
        // SAFETY: libSystem provides sin/cos.
        unsafe { (cos(phase), sin(phase)) }
    }
}

#[cfg(all(unix, not(any(target_os = "macos", target_os = "ios"))))]
mod twiddle_math {
    #[link(name = "m")]
    unsafe extern "C" {
        fn sin(x: f64) -> f64;
        fn cos(x: f64) -> f64;
    }

    pub(super) fn cos_sin(phase: f64) -> (f64, f64) {
        // SAFETY: libm provides sin/cos.
        unsafe { (cos(phase), sin(phase)) }
    }
}

#[cfg(not(unix))]
mod twiddle_math {
    use libm::{cos, sin};

    pub(super) fn cos_sin(phase: f64) -> (f64, f64) {
        (cos(phase), sin(phase))
    }
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
    #[cfg(test)]
    let trace_enabled = fft_stage_trace::bfly_active();
    if m == 1 {
        for i in 0..n {
            let base = 2 * i;
            #[cfg(test)]
            let mut before = [KissFftCpx::new(0.0, 0.0); 2];
            #[cfg(test)]
            if trace_enabled {
                before[0] = fout[base];
                before[1] = fout[base + 1];
            }
            let t = fout[base + 1];
            fout[base + 1] = c_sub(fout[base], t);
            fout[base] = c_add(fout[base], t);
            #[cfg(test)]
            if trace_enabled {
                let after = [fout[base], fout[base + 1]];
                fft_stage_trace::dump_bfly(&before, &after);
            }
        }
    } else {
        debug_assert_eq!(m, 4);
        let tw = FRAC_1_SQRT_2;
        for i in 0..n {
            let base = i * 2 * m;
            #[cfg(test)]
            let mut before = [KissFftCpx::new(0.0, 0.0); 8];
            #[cfg(test)]
            if trace_enabled {
                for k in 0..8 {
                    before[k] = fout[base + k];
                }
            }
            let t0 = fout[base + 4];
            fout[base + 4] = c_sub(fout[base], t0);
            fout[base] = c_add(fout[base], t0);

            let mut t1 = KissFftCpx::new(
                (fout[base + 5].r + fout[base + 5].i) * tw,
                (fout[base + 5].i - fout[base + 5].r) * tw,
            );
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
            #[cfg(test)]
            if trace_enabled {
                let mut after = [KissFftCpx::new(0.0, 0.0); 8];
                for k in 0..8 {
                    after[k] = fout[base + k];
                }
                fft_stage_trace::dump_bfly(&before, &after);
            }
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
    #[cfg(test)]
    let trace_enabled = fft_stage_trace::bfly_active();
    let m2 = 2 * m;
    let epi3 = st.twiddles[fstride * m];
    for i in 0..n {
        let base = i * mm;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;
        for k in 0..m {
            #[cfg(test)]
            let mut before = [KissFftCpx::new(0.0, 0.0); 3];
            #[cfg(test)]
            if trace_enabled {
                before[0] = fout[base + k];
                before[1] = fout[base + m + k];
                before[2] = fout[base + m2 + k];
            }
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

            fout[base + m2 + k] = KissFftCpx::new(fout_m.r + scratch0.i, fout_m.i - scratch0.r);
            fout_m = KissFftCpx::new(fout_m.r - scratch0.i, fout_m.i + scratch0.r);

            fout[base + k] = fout0;
            fout[base + m + k] = fout_m;
            #[cfg(test)]
            if trace_enabled {
                let after = [
                    fout[base + k],
                    fout[base + m + k],
                    fout[base + m2 + k],
                ];
                fft_stage_trace::dump_bfly(&before, &after);
            }
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
    #[cfg(test)]
    let trace_enabled = fft_stage_trace::bfly_active();
    if m == 1 {
        for i in 0..n {
            let base = i * mm;
            #[cfg(test)]
            let mut before = [KissFftCpx::new(0.0, 0.0); 4];
            #[cfg(test)]
            if trace_enabled {
                for k in 0..4 {
                    before[k] = fout[base + k];
                }
            }
            #[cfg(test)]
            let bfly_idx = if trace_enabled {
                fft_stage_trace::current_bfly_index()
            } else {
                None
            };
            let scratch0 = c_sub(fout[base], fout[base + 2]);
            let scratch1 = c_add(fout[base + 1], fout[base + 3]);
            let scratch1b = c_sub(fout[base + 1], fout[base + 3]);
            #[cfg(test)]
            if let Some(bfly_idx) = bfly_idx {
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch0", scratch0);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch1", scratch1);
            }

            let mut fout0 = c_add(fout[base], fout[base + 2]);
            fout[base + 2] = c_sub(fout0, scratch1);
            fout0 = c_add(fout0, scratch1);

            fout[base + 1] = KissFftCpx::new(scratch0.r + scratch1b.i, scratch0.i - scratch1b.r);
            fout[base + 3] = KissFftCpx::new(scratch0.r - scratch1b.i, scratch0.i + scratch1b.r);
            fout[base] = fout0;
            #[cfg(test)]
            if let Some(bfly_idx) = bfly_idx {
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch1b", scratch1b);
            }
            #[cfg(test)]
            if trace_enabled {
                let mut after = [KissFftCpx::new(0.0, 0.0); 4];
                for k in 0..4 {
                    after[k] = fout[base + k];
                }
                fft_stage_trace::dump_bfly(&before, &after);
            }
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
                #[cfg(test)]
                let mut before = [KissFftCpx::new(0.0, 0.0); 4];
                #[cfg(test)]
                if trace_enabled {
                    before[0] = fout[base + j];
                    before[1] = fout[base + j + m];
                    before[2] = fout[base + j + m2];
                    before[3] = fout[base + j + m3];
                }
                #[cfg(test)]
                let bfly_idx = if trace_enabled {
                    fft_stage_trace::current_bfly_index()
                } else {
                    None
                };
                let scratch0 = c_mul(fout[base + j + m], st.twiddles[tw1]);
                let scratch1 = c_mul(fout[base + j + m2], st.twiddles[tw2]);
                let scratch2 = c_mul(fout[base + j + m3], st.twiddles[tw3]);
                #[cfg(test)]
                if let Some(bfly_idx) = bfly_idx {
                    fft_stage_trace::dump_bfly_value(bfly_idx, "mul_in0", before[1]);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "mul_in1", before[2]);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "mul_in2", before[3]);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "mul_in0", before[1]);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "mul_in1", before[2]);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "mul_in2", before[3]);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "tw1", st.twiddles[tw1]);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "tw2", st.twiddles[tw2]);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "tw3", st.twiddles[tw3]);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "tw1", st.twiddles[tw1]);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "tw2", st.twiddles[tw2]);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "tw3", st.twiddles[tw3]);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "mul0", scratch0);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "mul1", scratch1);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "mul2", scratch2);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "mul0", scratch0);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "mul1", scratch1);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "mul2", scratch2);
                }

                tw1 += fstride;
                tw2 += fstride * 2;
                tw3 += fstride * 3;

                let scratch5 = c_sub(fout[base + j], scratch1);
                let mut fout0 = c_add(fout[base + j], scratch1);
                let scratch3 = c_add(scratch0, scratch2);
                let scratch4 = c_sub(scratch0, scratch2);
                #[cfg(test)]
                if let Some(bfly_idx) = bfly_idx {
                    fft_stage_trace::dump_bfly_value(bfly_idx, "scratch5", scratch5);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "scratch3", scratch3);
                    fft_stage_trace::dump_bfly_value(bfly_idx, "scratch4", scratch4);
                }

                fout[base + j + m2] = c_sub(fout0, scratch3);
                fout0 = c_add(fout0, scratch3);

                let fout_m = KissFftCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
                let fout_m3 = KissFftCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);

                fout[base + j] = fout0;
                fout[base + j + m] = fout_m;
                fout[base + j + m3] = fout_m3;
                #[cfg(test)]
                if trace_enabled {
                    let after = [
                        fout[base + j],
                        fout[base + j + m],
                        fout[base + j + m2],
                        fout[base + j + m3],
                    ];
                    fft_stage_trace::dump_bfly(&before, &after);
                }
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
    #[cfg(test)]
    let trace_enabled = fft_stage_trace::bfly_active();
    let ya = st.twiddles[fstride * m];
    let yb = st.twiddles[fstride * 2 * m];
    for i in 0..n {
        let base = i * mm;
        for u in 0..m {
            #[cfg(test)]
            let mut before = [KissFftCpx::new(0.0, 0.0); 5];
            #[cfg(test)]
            if trace_enabled {
                before[0] = fout[base + u];
                before[1] = fout[base + m + u];
                before[2] = fout[base + 2 * m + u];
                before[3] = fout[base + 3 * m + u];
                before[4] = fout[base + 4 * m + u];
            }
            #[cfg(test)]
            let bfly_idx = if trace_enabled {
                fft_stage_trace::current_bfly_index()
            } else {
                None
            };
            #[cfg(test)]
            if let Some(bfly_idx) = bfly_idx {
                fft_stage_trace::dump_bfly_indices(
                    bfly_idx,
                    [
                        base + u,
                        base + m + u,
                        base + 2 * m + u,
                        base + 3 * m + u,
                        base + 4 * m + u,
                    ],
                );
                if bfly_idx == 0 {
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "ya", ya);
                    fft_stage_trace::dump_bfly_bits(bfly_idx, "yb", yb);
                }
            }
            let scratch0 = fout[base + u];
            #[cfg(test)]
            if let Some(bfly_idx) = bfly_idx {
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch0", scratch0);
            }
            let scratch1 = c_mul(fout[base + m + u], st.twiddles[u * fstride]);
            let scratch2 = c_mul(fout[base + 2 * m + u], st.twiddles[2 * u * fstride]);
            let scratch3 = c_mul(fout[base + 3 * m + u], st.twiddles[3 * u * fstride]);
            let scratch4 = c_mul(fout[base + 4 * m + u], st.twiddles[4 * u * fstride]);
            #[cfg(test)]
            if let Some(bfly_idx) = bfly_idx {
                let tw4_idx = 4 * u * fstride;
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch1", scratch1);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch2", scratch2);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch3", scratch3);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch4", scratch4);
                fft_stage_trace::dump_bfly_bits(bfly_idx, "mul_in4", before[4]);
                fft_stage_trace::dump_bfly_twiddle_index(bfly_idx, "tw4.idx", tw4_idx);
                fft_stage_trace::dump_bfly_bits(bfly_idx, "tw4", st.twiddles[tw4_idx]);
                fft_stage_trace::dump_bfly_bits(bfly_idx, "scratch4", scratch4);
            }

            let scratch7 = c_add(scratch1, scratch4);
            let scratch10 = c_sub(scratch1, scratch4);
            let scratch8 = c_add(scratch2, scratch3);
            let scratch9 = c_sub(scratch2, scratch3);
            #[cfg(test)]
            if let Some(bfly_idx) = bfly_idx {
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch7", scratch7);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch10", scratch10);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch8", scratch8);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch9", scratch9);
                if bfly_idx == fft_stage_trace::bfly_detail_index() {
                    let s6a = scratch10.i * ya.i;
                    let s6b = scratch9.i * yb.i;
                    let s6sum = s6a + s6b;
                    let s11ar = scratch7.r * yb.r;
                    let s11br = scratch8.r * ya.r;
                    let s11sum_r = s11ar + s11br;
                    let s11ai = scratch7.i * yb.r;
                    let s11bi = scratch8.i * ya.r;
                    let s11sum_i = s11ai + s11bi;
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch6_term0", s6a);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch6_term0", s6a);
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch6_term1", s6b);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch6_term1", s6b);
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch6_sum", s6sum);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch6_sum", s6sum);
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch11_term0_r", s11ar);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch11_term0_r", s11ar);
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch11_term1_r", s11br);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch11_term1_r", s11br);
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch11_sum_r", s11sum_r);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch11_sum_r", s11sum_r);
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch11_term0_i", s11ai);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch11_term0_i", s11ai);
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch11_term1_i", s11bi);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch11_term1_i", s11bi);
                    fft_stage_trace::dump_bfly_scalar(bfly_idx, "scratch11_sum_i", s11sum_i);
                    fft_stage_trace::dump_bfly_scalar_bits(bfly_idx, "scratch11_sum_i", s11sum_i);
                }
            }

            let fout0 = c_add(scratch0, c_add(scratch7, scratch8));

            // Mirror C's FP contraction in the radix-5 butterfly.
            let scratch5_r = mul_add_f32(scratch7.r, ya.r, scratch8.r * yb.r);
            let scratch5_i = mul_add_f32(scratch7.i, ya.r, scratch8.i * yb.r);
            let scratch5 = KissFftCpx::new(scratch0.r + scratch5_r, scratch0.i + scratch5_i);
            // Use mul_add to mirror C's FP contraction in the radix-5 butterfly.
            let scratch6 = KissFftCpx::new(
                mul_add_f32(scratch10.i, ya.i, scratch9.i * yb.i),
                -mul_add_f32(scratch10.r, ya.i, scratch9.r * yb.i),
            );
            #[cfg(test)]
            if let Some(bfly_idx) = bfly_idx {
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch5", scratch5);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch6", scratch6);
                fft_stage_trace::dump_bfly_bits(bfly_idx, "scratch6", scratch6);
            }

            fout[base + m + u] = c_sub(scratch5, scratch6);
            fout[base + 4 * m + u] = c_add(scratch5, scratch6);

            let scratch11_r = mul_add_f32(scratch7.r, yb.r, scratch8.r * ya.r);
            let scratch11_i = mul_add_f32(scratch7.i, yb.r, scratch8.i * ya.r);
            let scratch11 =
                KissFftCpx::new(scratch0.r + scratch11_r, scratch0.i + scratch11_i);
            let scratch12 = KissFftCpx::new(
                mul_add_f32(scratch9.i, ya.i, -scratch10.i * yb.i),
                mul_add_f32(scratch10.r, yb.i, -scratch9.r * ya.i),
            );
            #[cfg(test)]
            if let Some(bfly_idx) = bfly_idx {
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch11", scratch11);
                fft_stage_trace::dump_bfly_value(bfly_idx, "scratch12", scratch12);
                fft_stage_trace::dump_bfly_bits(bfly_idx, "scratch11", scratch11);
                fft_stage_trace::dump_bfly_bits(bfly_idx, "scratch12", scratch12);
            }

            fout[base + 2 * m + u] = c_add(scratch11, scratch12);
            fout[base + 3 * m + u] = c_sub(scratch11, scratch12);
            fout[base + u] = fout0;
            #[cfg(test)]
            if trace_enabled {
                let after = [
                    fout[base + u],
                    fout[base + m + u],
                    fout[base + 2 * m + u],
                    fout[base + 3 * m + u],
                    fout[base + 4 * m + u],
                ];
                if let Some(bfly_idx) = bfly_idx {
                    if bfly_idx == fft_stage_trace::bfly_detail_index() {
                        fft_stage_trace::dump_bfly_bits(bfly_idx, "out0", after[0]);
                        fft_stage_trace::dump_bfly_bits(bfly_idx, "out1", after[1]);
                        fft_stage_trace::dump_bfly_bits(bfly_idx, "out2", after[2]);
                        fft_stage_trace::dump_bfly_bits(bfly_idx, "out3", after[3]);
                        fft_stage_trace::dump_bfly_bits(bfly_idx, "out4", after[4]);
                    }
                }
                fft_stage_trace::dump_bfly(&before, &after);
            }
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
                let phase = -2.0 * PI_F64 * bin as f64 * k as f64 / nfft as f64;
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
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
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

    #[test]
    fn twiddle_phase_matches_opus_c_order() {
        let nfft = 480usize;
        let cases = [
            (120usize, 0xbff921fb54442d18u64),
            (240usize, 0xc00921fb54442d18u64),
        ];
        for (index, expected) in cases {
            let phase = twiddle_phase(index, nfft);
            assert_eq!(
                phase.to_bits(),
                expected,
                "phase bits mismatch for index={index}"
            );
        }
    }
}
