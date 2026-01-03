//! Deep REDundancy (DRED) stubs and helpers.
//!
//! The DRED feature is not fully ported yet. The APIs mirror the C surface
//! while returning `Unimplemented` until the model and coding paths land.

use crate::celt::opus_select_arch;
use crate::opus_decoder::OpusDecoder;

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

/// Opaque DRED decoder state.
#[derive(Debug, Default)]
pub struct OpusDredDecoder {
    loaded: bool,
    arch: i32,
}

/// Opaque DRED packet state.
#[derive(Debug, Default)]
pub struct OpusDred {
    _process_stage: i32,
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

    let _ = dred;
    let _ = data;
    let _ = max_dred_samples;
    let _ = sampling_rate;
    let _ = defer_processing;

    if !decoder.loaded {
        return Err(OpusDredError::Unimplemented);
    }

    if let Some(out) = dred_end {
        *out = 0;
    }

    Err(OpusDredError::Unimplemented)
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
}
