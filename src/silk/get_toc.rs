//! Port of the optional `silk_get_TOC` helper from `silk/dec_API.c`.
//!
//! The reference implementation exposes a small utility that inspects the
//! first byte of a SILK payload and reconstructs the per-frame VAD and FEC
//! flags stored in the table of contents. This Rust translation mirrors that
//! behaviour for callers that need a lightweight summary of a packet without
//! running the full decoder.

use bitflags::bitflags;

use crate::silk::MAX_FRAMES_PER_PACKET;
use crate::silk::errors::SilkError;

bitflags! {
    /// Bitflags that encode the per-frame VAD bits plus the in-band FEC flag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TocFlags: u8 {
        /// Packet carries low-bit-rate redundancy (FEC).
        const INBAND_FEC = 1 << 0;
        /// Frame 0 is voice-active.
        const VAD0 = 1 << 1;
        /// Frame 1 is voice-active.
        const VAD1 = 1 << 2;
        /// Frame 2 is voice-active.
        const VAD2 = 1 << 3;
    }
}

/// Simplified table-of-contents view for a SILK packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toc {
    /// Number of SILK frames represented by this TOC (1–3).
    pub frames_per_payload: usize,
    /// Encoded flag bits for VAD and in-band FEC.
    pub flags: TocFlags,
}

impl Toc {
    /// Returns `true` when any frame is marked as voice-active.
    pub fn vad_flag(&self) -> bool {
        self.flags
            .intersects(TocFlags::VAD0 | TocFlags::VAD1 | TocFlags::VAD2)
    }

    /// Returns the per-frame VAD decisions as a boolean array.
    #[must_use]
    pub fn vad_flags(&self) -> [bool; MAX_FRAMES_PER_PACKET] {
        let mut out = [false; MAX_FRAMES_PER_PACKET];
        for (idx, flag) in [TocFlags::VAD0, TocFlags::VAD1, TocFlags::VAD2]
            .iter()
            .enumerate()
        {
            out[idx] = self.flags.contains(*flag);
        }
        out
    }

    /// Returns whether in-band FEC is present.
    #[must_use]
    pub fn inband_fec_flag(&self) -> bool {
        self.flags.contains(TocFlags::INBAND_FEC)
    }
}

impl Default for Toc {
    fn default() -> Self {
        Self {
            frames_per_payload: 1,
            flags: TocFlags::empty(),
        }
    }
}

/// Mirrors `silk_get_TOC`.
///
/// # Errors
/// Returns [`SilkError::DecInvalidFrameSize`] when the payload is empty or when
/// `frames_per_payload` exceeds `MAX_FRAMES_PER_PACKET`.
pub fn silk_get_toc(payload: &[u8], frames_per_payload: usize) -> Result<Toc, SilkError> {
    if payload.is_empty() || frames_per_payload > MAX_FRAMES_PER_PACKET {
        return Err(SilkError::DecInvalidFrameSize);
    }

    let mut toc = Toc {
        frames_per_payload,
        flags: TocFlags::empty(),
    };
    let mut flags =
        (payload[0] >> (7 - frames_per_payload)) & ((1 << (frames_per_payload + 1)) - 1);

    if flags & 1 != 0 {
        toc.flags.insert(TocFlags::INBAND_FEC);
    }

    for frame_idx in (0..frames_per_payload).rev() {
        flags >>= 1;
        let voiced = flags & 1 != 0;
        if voiced {
            match frame_idx {
                0 => toc.flags.insert(TocFlags::VAD0),
                1 => toc.flags.insert(TocFlags::VAD1),
                2 => toc.flags.insert(TocFlags::VAD2),
                _ => {}
            }
        }
    }

    Ok(toc)
}

#[cfg(test)]
mod tests {
    use super::{Toc, TocFlags, silk_get_toc};
    use crate::silk::MAX_FRAMES_PER_PACKET;
    use crate::silk::errors::SilkError;

    #[test]
    fn rejects_empty_payloads() {
        let err = silk_get_toc(&[], 1);
        assert_eq!(err, Err(SilkError::DecInvalidFrameSize));
    }

    #[test]
    fn rejects_excess_frames_per_payload() {
        let payload = [0u8; 1];
        let err = silk_get_toc(&payload, 4);
        assert_eq!(err, Err(SilkError::DecInvalidFrameSize));
    }

    #[test]
    fn decodes_toc_flags_for_single_frame() {
        // First byte: 1xxx xxxx -> VAD set, in-band FEC clear for a 1-frame packet.
        let payload = [0b1000_0000u8];
        let toc = silk_get_toc(&payload, 1).unwrap();
        assert_eq!(
            toc,
            Toc {
                frames_per_payload: 1,
                flags: TocFlags::VAD0
            }
        );
        assert!(toc.vad_flag());
        assert_eq!(toc.vad_flags(), [true, false, false]);
        assert!(!toc.inband_fec_flag());
    }

    #[test]
    fn decodes_toc_flags_for_three_frames() {
        // Packets with three frames encode the VAD/FEC bits in the upper nibble.
        // payload[0] layout for 3 frames: ???F V2 V1 V0
        let payload = [0b0010_1000u8];
        let toc = silk_get_toc(&payload, 3).unwrap();
        assert_eq!(
            toc,
            Toc {
                frames_per_payload: 3,
                flags: TocFlags::VAD2
            }
        );
        assert!(!toc.inband_fec_flag());
        assert!(toc.vad_flag());
        assert_eq!(toc.vad_flags(), [false, false, true]);
    }

    #[test]
    fn vad_flags_are_zeroed_for_unused_frames() {
        let payload = [0b1100_0000u8]; // FEC + VAD0 for a single-frame payload.
        let toc = silk_get_toc(&payload, 1).unwrap();
        let mut expected = [false; MAX_FRAMES_PER_PACKET];
        expected[0] = true;
        assert_eq!(toc.vad_flags(), expected);
        assert!(toc.inband_fec_flag());
    }
}
