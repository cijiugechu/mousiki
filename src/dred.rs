//! Deep REDundancy (DRED) stubs and helpers.
//!
//! The DRED feature is not fully ported yet. The APIs mirror the C surface
//! while returning `Unimplemented` until the model and coding paths land.

use crate::celt::opus_select_arch;
use crate::extensions::{ExtensionError, OpusExtensionIterator};
use crate::opus_decoder::OpusDecoder;
use crate::packet::{opus_packet_get_samples_per_frame, opus_packet_parse_impl, PacketError};

const DRED_EXTENSION_ID: u8 = 126;
const DRED_EXPERIMENTAL_VERSION: u8 = 10;
const DRED_EXPERIMENTAL_BYTES: usize = 2;
const DRED_FRAME_OFFSET_DIVISOR: i32 = 120;

/// Errors surfaced by the DRED helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusDredError {
    BadArgument,
    BufferTooSmall,
    InvalidPacket,
    InternalError,
    Unimplemented,
}

impl OpusDredError {
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::InvalidPacket => -4,
            Self::Unimplemented => -5,
        }
    }
}

impl From<PacketError> for OpusDredError {
    #[inline]
    fn from(value: PacketError) -> Self {
        match value {
            PacketError::BadArgument => Self::BadArgument,
            PacketError::InvalidPacket => Self::InvalidPacket,
        }
    }
}

impl From<ExtensionError> for OpusDredError {
    #[inline]
    fn from(value: ExtensionError) -> Self {
        match value {
            ExtensionError::BadArgument => Self::BadArgument,
            ExtensionError::BufferTooSmall => Self::BufferTooSmall,
            ExtensionError::InvalidPacket => Self::InvalidPacket,
        }
    }
}

/// Opaque DRED decoder state.
#[derive(Debug, Default)]
pub struct OpusDredDecoder {
    loaded: bool,
    arch: i32,
}

/// Opaque DRED packet state.
#[derive(Debug, Default)]
pub struct OpusDred {
    process_stage: i32,
}

#[derive(Debug, Clone, Copy)]
struct DredPayload<'a> {
    payload: &'a [u8],
    dred_frame_offset: i32,
}

fn dred_find_payload<'a>(data: &'a [u8]) -> Result<Option<DredPayload<'a>>, OpusDredError> {
    let parsed = opus_packet_parse_impl(data, data.len(), false)?;
    let frame_size = opus_packet_get_samples_per_frame(data, 48_000)?;
    let frame_size =
        i32::try_from(frame_size).map_err(|_| OpusDredError::InvalidPacket)?;
    let mut iter = OpusExtensionIterator::new(parsed.padding, parsed.frame_count);

    loop {
        let Some(ext) = iter.find(DRED_EXTENSION_ID)? else {
            return Ok(None);
        };

        let ext_len = usize::try_from(ext.len).map_err(|_| OpusDredError::InvalidPacket)?;
        if ext_len > ext.data.len() {
            return Err(OpusDredError::InvalidPacket);
        }

        let dred_frame_offset = ext
            .frame
            .checked_mul(frame_size)
            .and_then(|value| value.checked_div(DRED_FRAME_OFFSET_DIVISOR))
            .ok_or(OpusDredError::InvalidPacket)?;

        if ext_len > DRED_EXPERIMENTAL_BYTES
            && ext.data.len() >= DRED_EXPERIMENTAL_BYTES
            && ext.data[0] == b'D'
            && ext.data[1] == DRED_EXPERIMENTAL_VERSION
        {
            let payload = &ext.data[DRED_EXPERIMENTAL_BYTES..ext_len];
            return Ok(Some(DredPayload {
                payload,
                dred_frame_offset,
            }));
        }
    }
}

/// Mirrors `opus_dred_decoder_get_size`.
#[inline]
pub fn opus_dred_decoder_get_size() -> usize {
    core::mem::size_of::<OpusDredDecoder>()
}

/// Mirrors `opus_dred_decoder_init`.
pub fn opus_dred_decoder_init(decoder: &mut OpusDredDecoder) -> Result<(), OpusDredError> {
    decoder.loaded = false;
    decoder.arch = opus_select_arch();
    Ok(())
}

/// Mirrors `opus_dred_decoder_create`.
pub fn opus_dred_decoder_create() -> Result<OpusDredDecoder, OpusDredError> {
    let mut decoder = OpusDredDecoder::default();
    opus_dred_decoder_init(&mut decoder)?;
    Ok(decoder)
}

/// Mirrors `opus_dred_decoder_destroy`.
#[inline]
pub fn opus_dred_decoder_destroy(_decoder: OpusDredDecoder) {}

/// Strongly-typed replacement for the DRED decoder CTL dispatcher.
pub enum OpusDredDecoderCtlRequest<'req> {
    SetDnnBlob(&'req [u8]),
}

/// Mirrors `opus_dred_decoder_ctl`.
pub fn opus_dred_decoder_ctl(
    _decoder: &mut OpusDredDecoder,
    _request: OpusDredDecoderCtlRequest<'_>,
) -> Result<(), OpusDredError> {
    Err(OpusDredError::Unimplemented)
}

/// Mirrors `opus_dred_get_size`.
pub fn opus_dred_get_size() -> usize {
    if cfg!(feature = "dred") {
        core::mem::size_of::<OpusDred>()
    } else {
        0
    }
}

/// Mirrors `opus_dred_alloc`.
pub fn opus_dred_alloc() -> Result<OpusDred, OpusDredError> {
    if cfg!(feature = "dred") {
        Ok(OpusDred::default())
    } else {
        Err(OpusDredError::Unimplemented)
    }
}

/// Mirrors `opus_dred_free`.
#[inline]
pub fn opus_dred_free(_dred: OpusDred) {}

/// Mirrors `opus_dred_parse`.
#[allow(clippy::too_many_arguments)]
pub fn opus_dred_parse(
    decoder: &OpusDredDecoder,
    dred: &mut OpusDred,
    data: &[u8],
    max_dred_samples: i32,
    sampling_rate: i32,
    dred_end: Option<&mut i32>,
    defer_processing: bool,
) -> Result<i32, OpusDredError> {
    if !cfg!(feature = "dred") {
        return Err(OpusDredError::Unimplemented);
    }

    if !decoder.loaded {
        return Err(OpusDredError::Unimplemented);
    }

    dred.process_stage = -1;

    if let Some(payload) = dred_find_payload(data)? {
        let DredPayload {
            payload: dred_payload,
            dred_frame_offset,
        } = payload;
        let _ = (
            max_dred_samples,
            sampling_rate,
            defer_processing,
            dred_payload,
            dred_frame_offset,
        );
        return Err(OpusDredError::Unimplemented);
    }

    if let Some(out) = dred_end {
        *out = 0;
    }
    Ok(0)
}

/// Mirrors `opus_dred_process`.
pub fn opus_dred_process(
    _decoder: &mut OpusDredDecoder,
    _src: &OpusDred,
    _dst: &mut OpusDred,
) -> Result<(), OpusDredError> {
    Err(OpusDredError::Unimplemented)
}

/// Mirrors `opus_decoder_dred_decode`.
pub fn opus_decoder_dred_decode(
    _decoder: &mut OpusDecoder<'_>,
    _dred: &OpusDred,
    _dred_offset: i32,
    _pcm: &mut [i16],
    _frame_size: usize,
) -> Result<usize, OpusDredError> {
    Err(OpusDredError::Unimplemented)
}

/// Mirrors `opus_decoder_dred_decode24`.
pub fn opus_decoder_dred_decode24(
    _decoder: &mut OpusDecoder<'_>,
    _dred: &OpusDred,
    _dred_offset: i32,
    _pcm: &mut [i32],
    _frame_size: usize,
) -> Result<usize, OpusDredError> {
    Err(OpusDredError::Unimplemented)
}

/// Mirrors `opus_decoder_dred_decode_float`.
pub fn opus_decoder_dred_decode_float(
    _decoder: &mut OpusDecoder<'_>,
    _dred: &OpusDred,
    _dred_offset: i32,
    _pcm: &mut [f32],
    _frame_size: usize,
) -> Result<usize, OpusDredError> {
    Err(OpusDredError::Unimplemented)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::{opus_packet_extensions_generate, OpusExtensionData};
    use alloc::vec::Vec;

    fn build_packet_with_padding(frame_count: usize, frame_len: usize, padding: &[u8]) -> Vec<u8> {
        assert!(frame_count > 0 && frame_count < 64);
        assert!(padding.len() > 0 && padding.len() < 255);

        let mut packet =
            Vec::with_capacity(3 + frame_count * frame_len + padding.len());
        packet.push(0x03);
        packet.push(0x40 | frame_count as u8);
        packet.push(padding.len() as u8);
        packet.resize(packet.len() + frame_count * frame_len, 0);
        packet.extend_from_slice(padding);
        packet
    }

    fn build_dred_padding(frame_count: usize, frame: i32, payload: &[u8]) -> Vec<u8> {
        let mut ext_bytes = Vec::with_capacity(DRED_EXPERIMENTAL_BYTES + payload.len());
        ext_bytes.push(b'D');
        ext_bytes.push(DRED_EXPERIMENTAL_VERSION);
        ext_bytes.extend_from_slice(payload);

        let ext = OpusExtensionData {
            id: DRED_EXTENSION_ID,
            frame,
            data: &ext_bytes,
            len: i32::try_from(ext_bytes.len()).expect("ext len fits i32"),
        };

        let max_len = 255usize;
        let required = opus_packet_extensions_generate(None, max_len, &[ext], frame_count, false)
            .expect("generate ext len");
        let mut padding = Vec::with_capacity(required);
        padding.resize(required, 0);
        let written = opus_packet_extensions_generate(
            Some(&mut padding),
            required,
            &[ext],
            frame_count,
            false,
        )
        .expect("generate ext bytes");
        assert_eq!(written, required);
        padding
    }

    #[test]
    fn dred_size_and_alloc_match_feature_state() {
        let size = opus_dred_get_size();
        if cfg!(feature = "dred") {
            assert!(size > 0);
            let dred = opus_dred_alloc().expect("dred alloc");
            opus_dred_free(dred);
        } else {
            assert_eq!(size, 0);
            assert_eq!(opus_dred_alloc().unwrap_err(), OpusDredError::Unimplemented);
        }
    }

    #[test]
    fn dred_parse_reports_unimplemented_without_model() {
        if !cfg!(feature = "dred") {
            return;
        }

        let decoder = opus_dred_decoder_create().expect("decoder");
        let mut dred = opus_dred_alloc().expect("dred alloc");
        let data = [0u8; 4];
        let err = opus_dred_parse(&decoder, &mut dred, &data, 48000, 48000, None, false)
            .unwrap_err();
        assert_eq!(err, OpusDredError::Unimplemented);
        opus_dred_free(dred);
        opus_dred_decoder_destroy(decoder);
    }

    #[test]
    fn dred_find_payload_returns_none_without_extension() {
        let packet = [0x00u8];
        let payload = dred_find_payload(&packet).expect("parse");
        assert!(payload.is_none());
    }

    #[test]
    fn dred_find_payload_extracts_payload_and_offset() {
        let payload_bytes = [0xAA, 0xBB];
        let padding = build_dred_padding(2, 1, &payload_bytes);
        let packet = build_packet_with_padding(2, 1, &padding);

        let payload = dred_find_payload(&packet)
            .expect("parse")
            .expect("payload");
        assert_eq!(payload.payload, payload_bytes);
        let frame_size = opus_packet_get_samples_per_frame(&packet, 48_000)
            .expect("frame size") as i32;
        let expected_offset = frame_size / DRED_FRAME_OFFSET_DIVISOR;
        assert_eq!(payload.dred_frame_offset, expected_offset);
    }
}
