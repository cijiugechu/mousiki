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

use crate::celt::types::{CeltGlog, CeltSig, OpusCustomDecoder, OpusCustomMode, OpusVal16};

/// Linear prediction order used by the decoder side filters.
///
/// Mirrors the `LPC_ORDER` constant from the reference implementation.  The
/// value is surfaced here so future ports that rely on the LPC history length
/// can share the same constant.
const LPC_ORDER: usize = 24;

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

        let overlap = mode.overlap * channels;
        let lpc = LPC_ORDER * channels;
        let band_count = 2 * mode.num_ebands;

        Self {
            decode_mem: vec![0.0; overlap],
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
}

#[cfg(test)]
mod tests {
    use super::{CeltDecoderAlloc, LPC_ORDER};
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
        assert_eq!(alloc.decode_mem.len(), mode.overlap * 2);
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
}
