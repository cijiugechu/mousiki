#![allow(dead_code)]

use alloc::vec::Vec;

use super::mini_kfft::MiniKissFft;
use libm::cosf;

/// Corresponds to `opus_int16` in the C implementation.
pub type OpusInt16 = i16;
/// Corresponds to `opus_int32` in the C implementation.
pub type OpusInt32 = i32;
/// Corresponds to `opus_uint32` in the C implementation.
pub type OpusUint32 = u32;
/// Floating-point representation used for `opus_val16` in CELT's float build.
pub type OpusVal16 = f32;
/// Floating-point representation used for `opus_val32` in CELT's float build.
pub type OpusVal32 = f32;
/// Internal CELT signal precision.
pub type CeltSig = OpusVal32;
/// Internal CELT logarithmic energy precision.
pub type CeltGlog = OpusVal32;
/// Coefficients used by the MDCT windows.
pub type CeltCoef = OpusVal16;

/// Scalar type used by the KISS FFT tables.
pub type KissTwiddleScalar = f32;

/// Lookup table required by CELT's MDCT implementation.
///
/// Mirrors the layout of `mdct_lookup` from `celt/mdct.h` while relying on Rust
/// slices to express borrowed data. The lookup owns no memory itself; the
/// lifetimes keep the dependency graph explicit without resorting to raw
/// pointers.
#[derive(Debug, Clone)]
pub struct MdctLookup {
    pub len: usize,
    pub max_shift: usize,
    pub forward: Vec<MiniKissFft>,
    pub inverse: Vec<MiniKissFft>,
    pub twiddle: Vec<KissTwiddleScalar>,
    pub twiddle_offsets: Vec<usize>,
}

impl MdctLookup {
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
        for shift in 0..=max_shift {
            let n = len >> shift;
            assert!(
                n.is_multiple_of(4),
                "MDCT length must be a multiple of four"
            );
            forward.push(MiniKissFft::new(n >> 2, false));
            inverse.push(MiniKissFft::new(n >> 2, true));
        }

        let mut twiddle = Vec::new();
        let mut offsets = Vec::with_capacity(max_shift + 2);
        offsets.push(0);
        for shift in 0..=max_shift {
            let n = len >> shift;
            let n2 = n >> 1;
            for i in 0..n2 {
                let angle = 2.0 * core::f32::consts::PI * (i as f32 + 0.125) / n as f32;
                twiddle.push(cosf(angle));
            }
            offsets.push(twiddle.len());
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
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    #[must_use]
    pub fn max_shift(&self) -> usize {
        self.max_shift
    }

    #[inline]
    #[must_use]
    pub fn effective_len(&self, shift: usize) -> usize {
        assert!(shift <= self.max_shift);
        self.len >> shift
    }

    #[inline]
    #[must_use]
    pub fn forward_plan(&self, shift: usize) -> &MiniKissFft {
        assert!(shift < self.forward.len());
        &self.forward[shift]
    }

    #[inline]
    #[must_use]
    pub fn inverse_plan(&self, shift: usize) -> &MiniKissFft {
        assert!(shift < self.inverse.len());
        &self.inverse[shift]
    }

    #[inline]
    #[must_use]
    pub fn twiddles(&self, shift: usize) -> &[KissTwiddleScalar] {
        assert!(shift <= self.max_shift);
        let start = self.twiddle_offsets[shift];
        let end = self.twiddle_offsets[shift + 1];
        &self.twiddle[start..end]
    }
}

/// Borrowed view of the pulse cache information embedded inside `OpusCustomMode`.
#[derive(Debug, Clone, Copy)]
pub struct PulseCache<'a> {
    pub size: usize,
    pub index: &'a [i16],
    pub bits: &'a [u8],
    pub caps: &'a [u8],
}

/// Owned storage for the pulse cache tables referenced by custom modes.
#[derive(Debug, Clone, Default)]
pub struct PulseCacheData {
    pub size: usize,
    pub index: Vec<i16>,
    pub bits: Vec<u8>,
    pub caps: Vec<u8>,
}

impl PulseCacheData {
    /// Creates a new cache from fully-populated buffers.
    pub fn new(index: Vec<i16>, bits: Vec<u8>, caps: Vec<u8>) -> Self {
        let size = bits.len();
        Self {
            size,
            index,
            bits,
            caps,
        }
    }

    /// Returns a borrowed representation of the cached tables.
    #[must_use]
    pub fn as_view(&self) -> PulseCache<'_> {
        PulseCache {
            size: self.size,
            index: &self.index,
            bits: &self.bits,
            caps: &self.caps,
        }
    }
}

/// Rust port of the opaque `OpusCustomMode`/`CELTMode` type.
#[derive(Debug, Clone)]
pub struct OpusCustomMode<'a> {
    pub sample_rate: OpusInt32,
    pub overlap: usize,
    pub num_ebands: usize,
    pub effective_ebands: usize,
    pub pre_emphasis: [OpusVal16; 4],
    pub e_bands: &'a [i16],
    pub max_lm: usize,
    pub num_short_mdcts: usize,
    pub short_mdct_size: usize,
    pub num_alloc_vectors: usize,
    pub alloc_vectors: &'a [u8],
    pub log_n: &'a [i16],
    pub window: &'a [CeltCoef],
    pub mdct: MdctLookup,
    pub cache: PulseCacheData,
}

impl<'a> OpusCustomMode<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: OpusInt32,
        overlap: usize,
        e_bands: &'a [i16],
        alloc_vectors: &'a [u8],
        log_n: &'a [i16],
        window: &'a [CeltCoef],
        mdct: MdctLookup,
        cache: PulseCacheData,
    ) -> Self {
        let num_ebands = e_bands.len().saturating_sub(1);
        let num_alloc_vectors = if num_ebands > 0 {
            alloc_vectors.len() / num_ebands
        } else {
            0
        };
        Self {
            sample_rate,
            overlap,
            num_ebands,
            effective_ebands: num_ebands,
            pre_emphasis: [0.0; 4],
            e_bands,
            max_lm: 0,
            num_short_mdcts: 0,
            short_mdct_size: 0,
            num_alloc_vectors,
            alloc_vectors,
            log_n,
            window,
            mdct,
            cache,
        }
    }

    /// Returns a borrowed view of the cached pulse tables.
    #[must_use]
    pub fn pulse_cache(&self) -> PulseCache<'_> {
        self.cache.as_view()
    }
}

/// CELT analysis metadata shared between SILK and CELT.
#[derive(Debug, Clone, Default)]
pub struct AnalysisInfo {
    pub valid: bool,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub noisiness: f32,
    pub activity: f32,
    pub music_prob: f32,
    pub music_prob_min: f32,
    pub music_prob_max: f32,
    pub bandwidth: i32,
    pub activity_probability: f32,
    pub max_pitch_ratio: f32,
    pub leak_boost: [u8; 19],
}

/// Minimal port of the auxiliary SILK information embedded in the encoder.
#[derive(Debug, Clone, Default)]
pub struct SilkInfo {
    pub signal_type: i32,
    pub offset: i32,
}

/// Primary encoder state for CELT.
#[derive(Debug)]
pub struct OpusCustomEncoder<'a> {
    pub mode: &'a OpusCustomMode<'a>,
    pub channels: usize,
    pub stream_channels: usize,
    pub force_intra: bool,
    pub clip: bool,
    pub disable_prefilter: bool,
    pub complexity: i32,
    pub upsample: i32,
    pub start_band: i32,
    pub end_band: i32,
    pub bitrate: OpusInt32,
    pub use_vbr: bool,
    pub signalling: i32,
    pub constrained_vbr: bool,
    pub loss_rate: i32,
    pub lsb_depth: i32,
    pub lfe: bool,
    pub disable_inv: bool,
    pub arch: i32,
    pub rng: OpusUint32,
    pub spread_decision: i32,
    pub delayed_intra: OpusVal32,
    pub tonal_average: i32,
    pub last_coded_bands: i32,
    pub hf_average: i32,
    pub tapset_decision: i32,
    pub prefilter_period: i32,
    pub prefilter_gain: OpusVal16,
    pub prefilter_tapset: i32,
    pub consec_transient: i32,
    pub analysis: AnalysisInfo,
    pub silk_info: SilkInfo,
    pub preemph_mem_encoder: [OpusVal32; 2],
    pub preemph_mem_decoder: [OpusVal32; 2],
    pub vbr_reservoir: OpusInt32,
    pub vbr_drift: OpusInt32,
    pub vbr_offset: OpusInt32,
    pub vbr_count: OpusInt32,
    pub overlap_max: OpusVal32,
    pub stereo_saving: OpusVal16,
    pub intensity: i32,
    pub energy_mask: Option<&'a [CeltGlog]>,
    pub spec_avg: CeltGlog,
    pub in_mem: &'a mut [CeltSig],
    pub prefilter_mem: &'a mut [CeltSig],
    pub old_band_e: &'a mut [CeltGlog],
    pub old_log_e: &'a mut [CeltGlog],
    pub old_log_e2: &'a mut [CeltGlog],
    pub energy_error: &'a mut [CeltGlog],
}

impl<'a> OpusCustomEncoder<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        energy_mask: Option<&'a [CeltGlog]>,
        in_mem: &'a mut [CeltSig],
        prefilter_mem: &'a mut [CeltSig],
        old_band_e: &'a mut [CeltGlog],
        old_log_e: &'a mut [CeltGlog],
        old_log_e2: &'a mut [CeltGlog],
        energy_error: &'a mut [CeltGlog],
    ) -> Self {
        let overlap = mode.overlap * channels;
        debug_assert_eq!(in_mem.len(), overlap);
        let band_count = channels * mode.num_ebands;
        debug_assert_eq!(old_band_e.len(), band_count);
        debug_assert_eq!(old_log_e.len(), band_count);
        debug_assert_eq!(old_log_e2.len(), band_count);
        debug_assert_eq!(energy_error.len(), band_count);
        Self {
            mode,
            channels,
            stream_channels,
            force_intra: false,
            clip: false,
            disable_prefilter: false,
            complexity: 0,
            upsample: 1,
            start_band: 0,
            end_band: mode.num_ebands as i32,
            bitrate: 0,
            use_vbr: false,
            signalling: 0,
            constrained_vbr: false,
            loss_rate: 0,
            lsb_depth: 0,
            lfe: false,
            disable_inv: false,
            arch: 0,
            rng: 0,
            spread_decision: 0,
            delayed_intra: 0.0,
            tonal_average: 0,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            consec_transient: 0,
            analysis: AnalysisInfo::default(),
            silk_info: SilkInfo::default(),
            preemph_mem_encoder: [0.0; 2],
            preemph_mem_decoder: [0.0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0.0,
            stereo_saving: 0.0,
            intensity: 0,
            energy_mask,
            spec_avg: 0.0,
            in_mem,
            prefilter_mem,
            old_band_e,
            old_log_e,
            old_log_e2,
            energy_error,
        }
    }
}

/// Primary decoder state for CELT.
#[derive(Debug)]
pub struct OpusCustomDecoder<'a> {
    pub mode: &'a OpusCustomMode<'a>,
    pub overlap: usize,
    pub channels: usize,
    pub stream_channels: usize,
    pub downsample: i32,
    pub start_band: i32,
    pub end_band: i32,
    pub signalling: i32,
    pub disable_inv: bool,
    pub complexity: i32,
    pub arch: i32,
    pub rng: OpusUint32,
    pub error: i32,
    pub last_pitch_index: i32,
    pub loss_duration: i32,
    pub skip_plc: bool,
    pub postfilter_period: i32,
    pub postfilter_period_old: i32,
    pub postfilter_gain: OpusVal16,
    pub postfilter_gain_old: OpusVal16,
    pub postfilter_tapset: i32,
    pub postfilter_tapset_old: i32,
    pub prefilter_and_fold: bool,
    pub preemph_mem_decoder: [CeltSig; 2],
    pub decode_mem: &'a mut [CeltSig],
    pub lpc: &'a mut [OpusVal16],
    pub old_ebands: &'a mut [CeltGlog],
    pub old_log_e: &'a mut [CeltGlog],
    pub old_log_e2: &'a mut [CeltGlog],
    pub background_log_e: &'a mut [CeltGlog],
}

impl<'a> OpusCustomDecoder<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        decode_mem: &'a mut [CeltSig],
        lpc: &'a mut [OpusVal16],
        old_ebands: &'a mut [CeltGlog],
        old_log_e: &'a mut [CeltGlog],
        old_log_e2: &'a mut [CeltGlog],
        background_log_e: &'a mut [CeltGlog],
    ) -> Self {
        let overlap = mode.overlap * channels;
        debug_assert!(decode_mem.len() >= overlap);
        let band_count = 2 * mode.num_ebands;
        debug_assert_eq!(old_ebands.len(), band_count);
        debug_assert_eq!(old_log_e.len(), band_count);
        debug_assert_eq!(old_log_e2.len(), band_count);
        debug_assert_eq!(background_log_e.len(), band_count);
        Self {
            mode,
            overlap,
            channels,
            stream_channels,
            downsample: 1,
            start_band: 0,
            end_band: mode.num_ebands as i32,
            signalling: 0,
            disable_inv: false,
            complexity: 0,
            arch: 0,
            rng: 0,
            error: 0,
            last_pitch_index: 0,
            loss_duration: 0,
            skip_plc: false,
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            prefilter_and_fold: false,
            preemph_mem_decoder: [0.0; 2],
            decode_mem,
            lpc,
            old_ebands,
            old_log_e,
            old_log_e2,
            background_log_e,
        }
    }
}
