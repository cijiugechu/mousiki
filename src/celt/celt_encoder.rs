#![allow(dead_code)]

//! Encoder scaffolding ported from `celt/celt_encoder.c`.
//!
//! The reference implementation stores the primary encoder state followed by a
//! number of variable-length buffers.  This module mirrors the allocation
//! strategy so that higher level encoding routines can be translated
//! incrementally while keeping the memory layout compatible with the C code.
//! Future patches will extend this file with the analysis, bit allocation, and
//! entropy coding paths that still live in the C sources.

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::celt::resampling_factor;
use crate::celt::cpu_support::opus_select_arch;
use crate::celt::modes::opus_custom_mode_find_static;
use crate::celt::types::{
    AnalysisInfo, CeltGlog, CeltSig, OpusCustomEncoder, OpusCustomMode, OpusInt32, OpusUint32,
    SilkInfo,
};
use crate::celt::vq::SPREAD_NORMAL;

/// Maximum number of channels supported by the scalar encoder path.
const MAX_CHANNELS: usize = 2;

/// Size of the comb-filter history kept per channel by the encoder prefilter.
const COMBFILTER_MAXPERIOD: usize = 1024;

/// Special bitrate value used by Opus to request the maximum possible rate.
const OPUS_BITRATE_MAX: OpusInt32 = -1;

/// Errors that can be reported while initialising a CELT encoder instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltEncoderInitError {
    /// The requested number of channels exceeds the supported range.
    InvalidChannelCount,
    /// The number of coded stream channels is inconsistent with the layout.
    InvalidStreamChannels,
    /// The chosen sampling rate cannot be derived from the 48 kHz reference.
    UnsupportedSampleRate,
}

/// Returns the number of bytes required to allocate an encoder for `mode`.
#[must_use]
pub(crate) fn opus_custom_encoder_get_size(mode: &OpusCustomMode<'_>, channels: usize) -> usize {
    let in_mem = channels * mode.overlap;
    let prefilter_mem = channels * COMBFILTER_MAXPERIOD;
    let band_count = channels * mode.num_ebands;

    in_mem * core::mem::size_of::<CeltSig>()
        + prefilter_mem * core::mem::size_of::<CeltSig>()
        + 4 * band_count * core::mem::size_of::<CeltGlog>()
}

/// Returns the size of the canonical CELT encoder operating at 48 kHz/960.
#[must_use]
pub(crate) fn celt_encoder_get_size(channels: usize) -> Option<usize> {
    opus_custom_mode_find_static(48_000, 960)
        .map(|mode| opus_custom_encoder_get_size(&mode, channels))
}

/// Helper owning the trailing buffers that back [`OpusCustomEncoder`].
#[derive(Debug, Default)]
pub(crate) struct CeltEncoderAlloc {
    in_mem: Vec<CeltSig>,
    prefilter_mem: Vec<CeltSig>,
    old_band_e: Vec<CeltGlog>,
    old_log_e: Vec<CeltGlog>,
    old_log_e2: Vec<CeltGlog>,
    energy_error: Vec<CeltGlog>,
}

impl CeltEncoderAlloc {
    /// Creates a new allocation suitable for the provided mode and channel layout.
    pub(crate) fn new(mode: &OpusCustomMode<'_>, channels: usize) -> Self {
        assert!(
            channels > 0 && channels <= MAX_CHANNELS,
            "unsupported channel layout"
        );

        let overlap = mode.overlap * channels;
        let band_count = channels * mode.num_ebands;

        Self {
            in_mem: vec![0.0; overlap],
            prefilter_mem: vec![0.0; channels * COMBFILTER_MAXPERIOD],
            old_band_e: vec![0.0; band_count],
            old_log_e: vec![0.0; band_count],
            old_log_e2: vec![0.0; band_count],
            energy_error: vec![0.0; band_count],
        }
    }

    /// Returns the number of bytes consumed by the allocation.
    #[must_use]
    pub(crate) fn size_in_bytes(&self) -> usize {
        self.in_mem.len() * core::mem::size_of::<CeltSig>()
            + self.prefilter_mem.len() * core::mem::size_of::<CeltSig>()
            + (self.old_band_e.len()
                + self.old_log_e.len()
                + self.old_log_e2.len()
                + self.energy_error.len())
                * core::mem::size_of::<CeltGlog>()
    }

    /// Borrows the allocation as an [`OpusCustomEncoder`] tied to the provided mode.
    pub(crate) fn as_encoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        energy_mask: Option<&'a [CeltGlog]>,
    ) -> OpusCustomEncoder<'a> {
        OpusCustomEncoder::new(
            mode,
            channels,
            stream_channels,
            energy_mask,
            self.in_mem.as_mut_slice(),
            self.prefilter_mem.as_mut_slice(),
            self.old_band_e.as_mut_slice(),
            self.old_log_e.as_mut_slice(),
            self.old_log_e2.as_mut_slice(),
            self.energy_error.as_mut_slice(),
        )
    }

    /// Clears the buffers and restores the reference reset state.
    pub(crate) fn reset(&mut self) {
        self.in_mem.fill(0.0);
        self.prefilter_mem.fill(0.0);
        self.old_band_e.fill(0.0);
        self.energy_error.fill(0.0);
        self.old_log_e.fill(-28.0);
        self.old_log_e2.fill(-28.0);
    }

    /// Internal helper shared by the public initialisation routines.
    fn init_internal<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        upsample: u32,
        arch: i32,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(CeltEncoderInitError::InvalidChannelCount);
        }
        if stream_channels == 0 || stream_channels > channels {
            return Err(CeltEncoderInitError::InvalidStreamChannels);
        }

        self.reset();
        let mut encoder = self.as_encoder(mode, channels, stream_channels, None);
        encoder.upsample = upsample as i32;
        encoder.start_band = 0;
        encoder.end_band = mode.effective_ebands as i32;
        encoder.signalling = 1;
        encoder.arch = arch;
        encoder.constrained_vbr = true;
        encoder.clip = true;
        encoder.bitrate = OPUS_BITRATE_MAX;
        encoder.use_vbr = false;
        encoder.force_intra = false;
        encoder.complexity = 5;
        encoder.lsb_depth = 24;
        encoder.loss_rate = 0;
        encoder.lfe = false;
        encoder.disable_prefilter = false;
        encoder.disable_inv = false;
        encoder.rng = rng_seed;
        encoder.spread_decision = SPREAD_NORMAL;
        encoder.delayed_intra = 1.0;
        encoder.tonal_average = 256;
        encoder.last_coded_bands = 0;
        encoder.hf_average = 0;
        encoder.tapset_decision = 0;
        encoder.prefilter_period = 0;
        encoder.prefilter_gain = 0.0;
        encoder.prefilter_tapset = 0;
        encoder.consec_transient = 0;
        encoder.analysis = AnalysisInfo::default();
        encoder.silk_info = SilkInfo::default();
        encoder.preemph_mem_encoder = [0.0; 2];
        encoder.preemph_mem_decoder = [0.0; 2];
        encoder.vbr_reservoir = 0;
        encoder.vbr_drift = 0;
        encoder.vbr_offset = 0;
        encoder.vbr_count = 0;
        encoder.overlap_max = 0.0;
        encoder.stereo_saving = 0.0;
        encoder.intensity = 0;
        encoder.energy_mask = None;
        encoder.spec_avg = 0.0;

        Ok(encoder)
    }

    /// Mirrors `opus_custom_encoder_init()` by configuring the encoder for a custom mode.
    pub(crate) fn init_custom_encoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
        self.init_internal(
            mode,
            channels,
            stream_channels,
            1,
            opus_select_arch(),
            rng_seed,
        )
    }

    /// Mirrors `celt_encoder_init()` by configuring the encoder for a public Opus mode.
    pub(crate) fn init_encoder_for_rate<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        sampling_rate: OpusInt32,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomEncoder<'a>, CeltEncoderInitError> {
        let upsample = resampling_factor(sampling_rate);
        if upsample == 0 {
            return Err(CeltEncoderInitError::UnsupportedSampleRate);
        }
        self.init_internal(
            mode,
            channels,
            stream_channels,
            upsample,
            opus_select_arch(),
            rng_seed,
        )
    }
}

// TODO: Port the remaining encoding routines (transient analysis, VBR control,
//       and the main `celt_encode_with_ec()` path) once the supporting modules
//       are available in Rust.

#[cfg(test)]
mod tests {
    use super::{
        COMBFILTER_MAXPERIOD, CeltEncoderAlloc, CeltEncoderInitError, MAX_CHANNELS,
        OPUS_BITRATE_MAX, SPREAD_NORMAL,
    };
    use crate::celt::cpu_support::opus_select_arch;
    use crate::celt::modes::opus_custom_mode_create;

    #[test]
    fn allocation_matches_reference_layout() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let alloc = CeltEncoderAlloc::new(&mode, 2);

        assert_eq!(alloc.in_mem.len(), mode.overlap * 2);
        assert_eq!(alloc.prefilter_mem.len(), 2 * COMBFILTER_MAXPERIOD);
        assert_eq!(alloc.old_band_e.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e2.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.energy_error.len(), 2 * mode.num_ebands);

        let bytes = alloc.size_in_bytes();
        assert_eq!(bytes, super::opus_custom_encoder_get_size(&mode, 2));
    }

    #[test]
    fn reset_initialises_energy_histories() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        alloc.old_log_e.fill(0.5);
        alloc.old_log_e2.fill(0.25);
        alloc.old_band_e.fill(1.0);
        alloc.energy_error.fill(1.0);
        alloc.prefilter_mem.fill(1.0);
        alloc.in_mem.fill(1.0);

        alloc.reset();

        assert!(alloc.in_mem.iter().all(|&v| v == 0.0));
        assert!(alloc.prefilter_mem.iter().all(|&v| v == 0.0));
        assert!(alloc.old_band_e.iter().all(|&v| v == 0.0));
        assert!(alloc.energy_error.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e.iter().all(|&v| (v + 28.0).abs() < 1e-6));
        assert!(alloc.old_log_e2.iter().all(|&v| (v + 28.0).abs() < 1e-6));
    }

    #[test]
    fn init_custom_encoder_sets_reference_defaults() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 2);

        let encoder = alloc
            .init_custom_encoder(&mode, 2, 2, 0xDEADBEEF)
            .expect("encoder");

        assert_eq!(encoder.channels, 2);
        assert_eq!(encoder.stream_channels, 2);
        assert_eq!(encoder.upsample, 1);
        assert_eq!(encoder.start_band, 0);
        assert_eq!(encoder.end_band, mode.effective_ebands as i32);
        assert_eq!(encoder.signalling, 1);
        assert_eq!(encoder.arch, opus_select_arch());
        assert!(encoder.constrained_vbr);
        assert!(encoder.clip);
        assert_eq!(encoder.bitrate, OPUS_BITRATE_MAX);
        assert!(!encoder.use_vbr);
        assert_eq!(encoder.complexity, 5);
        assert_eq!(encoder.lsb_depth, 24);
        assert_eq!(encoder.spread_decision, SPREAD_NORMAL);
        assert!((encoder.delayed_intra - 1.0).abs() < 1e-6);
        assert_eq!(encoder.tonal_average, 256);
        assert_eq!(encoder.hf_average, 0);
        assert_eq!(encoder.tapset_decision, 0);
        assert_eq!(encoder.rng, 0xDEADBEEF);
        assert!(encoder.old_log_e.iter().all(|&v| (v + 28.0).abs() < 1e-6));
        assert!(encoder.old_log_e2.iter().all(|&v| (v + 28.0).abs() < 1e-6));
    }

    #[test]
    fn init_encoder_for_rate_rejects_unsupported_sampling_rate() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        let err = alloc
            .init_encoder_for_rate(&mode, 1, 1, 44_100, 0)
            .unwrap_err();
        assert_eq!(err, CeltEncoderInitError::UnsupportedSampleRate);
    }

    #[test]
    fn channel_validation_matches_reference_limits() {
        let owned = opus_custom_mode_create(48_000, 960).expect("mode");
        let mode = owned.mode();
        let mut alloc = CeltEncoderAlloc::new(&mode, 1);

        let err = alloc.init_custom_encoder(&mode, 0, 0, 0).unwrap_err();
        assert_eq!(err, CeltEncoderInitError::InvalidChannelCount);

        let err = alloc
            .init_custom_encoder(&mode, MAX_CHANNELS + 1, 1, 0)
            .unwrap_err();
        assert_eq!(err, CeltEncoderInitError::InvalidChannelCount);

        let err = alloc.init_custom_encoder(&mode, 1, 0, 0).unwrap_err();
        assert_eq!(err, CeltEncoderInitError::InvalidStreamChannels);

        let err = alloc.init_custom_encoder(&mode, 2, 3, 0).unwrap_err();
        assert_eq!(err, CeltEncoderInitError::InvalidStreamChannels);
    }
}
