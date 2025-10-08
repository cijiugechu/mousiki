#![allow(dead_code)]

//! Decoder scaffolding ported from `celt/celt_decoder.c`.
//!
//! The reference implementation combines the primary decoder state with a
//! trailing buffer that stores the pitch predictor history, LPC coefficients,
//! and band energy memories.  This module mirrors the allocation strategy so
//! that higher level decode routines can be ported gradually while continuing
//! to rely on the Rust ownership model for safety.
//!
//! Only the allocation helpers are provided for now.  The full decoding loop,
//! packet loss concealment, and post-filter plumbing still live in the C
//! sources and will be translated in follow-up patches.

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::celt::resampling_factor;
use crate::celt::cpu_support::opus_select_arch;
use crate::celt::types::{
    CeltGlog, CeltSig, OpusCustomDecoder, OpusCustomMode, OpusUint32, OpusVal16,
};

/// Linear prediction order used by the decoder side filters.
///
/// Mirrors the `LPC_ORDER` constant from the reference implementation.  The
/// value is surfaced here so future ports that rely on the LPC history length
/// can share the same constant.
const LPC_ORDER: usize = 24;

/// Size of the rolling decode buffer maintained per channel.
///
/// Matches the `DECODE_BUFFER_SIZE` constant from the C implementation.  The
/// reference decoder keeps a two kilobyte circular history in front of the
/// overlap region so packet loss concealment and the post-filter can operate on
/// previously synthesised samples.  Mirroring the same storage requirements in
/// Rust keeps the allocation layout compatible with the ported routines that
/// will eventually consume these buffers.
const DECODE_BUFFER_SIZE: usize = 2048;

/// Maximum number of channels supported by the initial CELT decoder port.
///
/// The reference implementation restricts the custom decoder to mono or stereo
/// streams.  The helper routines below mirror the same validation so the
/// call-sites can rely on early argument checking just like the C helpers.
const MAX_CHANNELS: usize = 2;

/// Errors that can be reported when initialising a CELT decoder instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltDecoderInitError {
    /// Channel count was zero or larger than the supported maximum.
    InvalidChannelCount,
    /// Requested stream channel layout is not compatible with the physical
    /// channels configured for the decoder.
    InvalidStreamChannels,
    /// The provided mode uses a sampling rate that cannot be resampled from the
    /// 48 kHz CELT reference clock.
    UnsupportedSampleRate,
}

/// Helper owning the trailing buffers that back [`OpusCustomDecoder`].
///
/// The C implementation allocates the decoder struct followed by a number of
/// variable-length arrays.  Keeping the storage separate in Rust avoids unsafe
/// pointer arithmetic and simplifies sharing the buffers across temporary
/// decoder views used during reset or PLC.
#[derive(Debug, Default)]
pub(crate) struct CeltDecoderAlloc {
    decode_mem: Vec<CeltSig>,
    lpc: Vec<OpusVal16>,
    old_ebands: Vec<CeltGlog>,
    old_log_e: Vec<CeltGlog>,
    old_log_e2: Vec<CeltGlog>,
    background_log_e: Vec<CeltGlog>,
}

impl CeltDecoderAlloc {
    /// Creates a new allocation suitable for the provided mode and channel
    /// configuration.
    ///
    /// The decoder requires per-channel history buffers for the overlap region
    /// as well as twice the number of energy bands tracked by the mode.  The
    /// allocations follow the layout of the C implementation while leveraging
    /// Rust's `Vec` to manage the backing storage.
    pub(crate) fn new(mode: &OpusCustomMode<'_>, channels: usize) -> Self {
        assert!(channels > 0, "decoder must contain at least one channel");

        let overlap = mode.overlap;
        let decode_mem = channels * (DECODE_BUFFER_SIZE + overlap);
        let lpc = LPC_ORDER * channels;
        let band_count = 2 * mode.num_ebands;

        Self {
            decode_mem: vec![0.0; decode_mem],
            lpc: vec![0.0; lpc],
            old_ebands: vec![0.0; band_count],
            old_log_e: vec![0.0; band_count],
            old_log_e2: vec![0.0; band_count],
            background_log_e: vec![0.0; band_count],
        }
    }

    /// Returns the total size in bytes consumed by the allocation.
    ///
    /// Mirrors the behaviour of `celt_decoder_get_size()` in spirit by exposing
    /// how much storage is required for the decoder and its trailing buffers.
    /// The actual C helper only depends on the channel count; we include the
    /// mode so the calculation reflects the precise band layout in use.  A
    /// follow-up port of the fixed allocation used by the reference
    /// implementation will replace this helper with a fully bit-exact
    /// translation.
    pub(crate) fn size_in_bytes(&self) -> usize {
        self.decode_mem.len() * core::mem::size_of::<CeltSig>()
            + self.lpc.len() * core::mem::size_of::<OpusVal16>()
            + (self.old_ebands.len()
                + self.old_log_e.len()
                + self.old_log_e2.len()
                + self.background_log_e.len())
                * core::mem::size_of::<CeltGlog>()
    }

    /// Borrows the allocation as an [`OpusCustomDecoder`] tied to the provided
    /// mode.
    ///
    /// Each call returns a fresh decoder view referencing the same backing
    /// buffers.  This mirrors the C layout where the state and trailing memory
    /// occupy a single blob, enabling the caller to reset or reuse the decoder
    /// without reallocating.
    pub(crate) fn as_decoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
    ) -> OpusCustomDecoder<'a> {
        OpusCustomDecoder::new(
            mode,
            channels,
            stream_channels,
            self.decode_mem.as_mut_slice(),
            self.lpc.as_mut_slice(),
            self.old_ebands.as_mut_slice(),
            self.old_log_e.as_mut_slice(),
            self.old_log_e2.as_mut_slice(),
            self.background_log_e.as_mut_slice(),
        )
    }

    /// Resets the allocation contents to zero.
    pub(crate) fn reset(&mut self) {
        for sample in &mut self.decode_mem {
            *sample = 0.0;
        }
        for coeff in &mut self.lpc {
            *coeff = 0.0;
        }
        for history in &mut self.old_ebands {
            *history = 0.0;
        }
        for history in &mut self.old_log_e {
            *history = 0.0;
        }
        for history in &mut self.old_log_e2 {
            *history = 0.0;
        }
        for history in &mut self.background_log_e {
            *history = 0.0;
        }
    }

    /// Returns a freshly initialised decoder state.
    ///
    /// The helper mirrors `opus_custom_decoder_init()` by validating the
    /// channel configuration, clearing the trailing buffers, and populating the
    /// fields that depend on the current mode.  Callers receive a fully formed
    /// [`OpusCustomDecoder`] that borrows the allocation's backing storage.
    pub(crate) fn init_decoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomDecoder<'a>, CeltDecoderInitError> {
        validate_channel_layout(channels, stream_channels)?;

        let downsample = resampling_factor(mode.sample_rate);
        if downsample == 0 {
            return Err(CeltDecoderInitError::UnsupportedSampleRate);
        }

        self.reset();
        let mut decoder = self.as_decoder(mode, channels, stream_channels);
        decoder.downsample = downsample as i32;
        decoder.end_band = mode.effective_ebands as i32;
        decoder.arch = opus_select_arch();
        decoder.rng = rng_seed;
        decoder.error = 0;
        decoder.loss_duration = 0;
        decoder.skip_plc = false;
        decoder.postfilter_period = 0;
        decoder.postfilter_period_old = 0;
        decoder.postfilter_gain = 0.0;
        decoder.postfilter_gain_old = 0.0;
        decoder.postfilter_tapset = 0;
        decoder.postfilter_tapset_old = 0;
        decoder.prefilter_and_fold = false;
        decoder.preemph_mem_decoder = [0.0; 2];
        decoder.last_pitch_index = 0;

        Ok(decoder)
    }
}

fn validate_channel_layout(
    channels: usize,
    stream_channels: usize,
) -> Result<(), CeltDecoderInitError> {
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(CeltDecoderInitError::InvalidChannelCount);
    }
    if stream_channels == 0 || stream_channels > channels {
        return Err(CeltDecoderInitError::InvalidStreamChannels);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CeltDecoderAlloc, CeltDecoderInitError, LPC_ORDER, MAX_CHANNELS, validate_channel_layout,
    };
    use crate::celt::types::{MdctLookup, OpusCustomMode, PulseCacheData};
    use alloc::vec;

    #[test]
    fn allocates_expected_band_buffers() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 2);
        assert_eq!(
            alloc.decode_mem.len(),
            2 * (super::DECODE_BUFFER_SIZE + mode.overlap)
        );
        assert_eq!(alloc.lpc.len(), LPC_ORDER * 2);
        assert_eq!(alloc.old_ebands.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e2.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.background_log_e.len(), 2 * mode.num_ebands);

        // Ensure the reset helper clears all buffers.
        alloc.decode_mem.fill(1.0);
        alloc.lpc.fill(1.0);
        alloc.old_ebands.fill(1.0);
        alloc.old_log_e.fill(1.0);
        alloc.old_log_e2.fill(1.0);
        alloc.background_log_e.fill(1.0);
        alloc.reset();

        assert!(alloc.decode_mem.iter().all(|&v| v == 0.0));
        assert!(alloc.lpc.iter().all(|&v| v == 0.0));
        assert!(alloc.old_ebands.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e2.iter().all(|&v| v == 0.0));
        assert!(alloc.background_log_e.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn validate_channel_layout_rejects_invalid_configurations() {
        assert_eq!(
            validate_channel_layout(0, 0),
            Err(CeltDecoderInitError::InvalidChannelCount)
        );
        assert_eq!(
            validate_channel_layout(MAX_CHANNELS + 1, 1),
            Err(CeltDecoderInitError::InvalidChannelCount)
        );
        assert_eq!(
            validate_channel_layout(1, 0),
            Err(CeltDecoderInitError::InvalidStreamChannels)
        );
        assert_eq!(
            validate_channel_layout(1, 2),
            Err(CeltDecoderInitError::InvalidStreamChannels)
        );
    }

    #[test]
    fn init_decoder_populates_expected_defaults() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let decoder = alloc
            .init_decoder(&mode, 1, 1, 1234)
            .expect("initialisation should succeed");

        assert_eq!(decoder.overlap, mode.overlap);
        assert_eq!(decoder.downsample, 1);
        assert_eq!(decoder.end_band, mode.effective_ebands as i32);
        assert_eq!(decoder.arch, 0);
        assert_eq!(decoder.rng, 1234);
        assert_eq!(decoder.loss_duration, 0);
        assert_eq!(decoder.postfilter_period, 0);
        assert_eq!(decoder.postfilter_gain, 0.0);
        assert_eq!(decoder.postfilter_tapset, 0);
        assert!(!decoder.skip_plc);
    }
}
