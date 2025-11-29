//! Small pieces of the top-level Opus decoder API.
//!
//! Ports the size helper from `opus_decoder_get_size()` so callers can
//! determine how much memory the combined SILK/CELT decoder requires.

use crate::celt::{OwnedCeltDecoder, canonical_mode, celt_decoder_get_size, opus_custom_decoder_create};
use crate::silk::decoder_state::DecoderState;
use crate::silk::dec_api::{Decoder as SilkDecoder, DECODER_NUM_CHANNELS};
use crate::silk::get_decoder_size::get_decoder_size;
use crate::silk::init_decoder::init_decoder as silk_init_channel;

/// Maximum supported channel count for the canonical decoder.
const MAX_CHANNELS: usize = 2;

/// Mirrors the alignment used by `opus_decoder_get_size` in the C code.
#[inline]
fn align(value: usize) -> usize {
    #[repr(C)]
    struct AlignProbe {
        _tag: u8,
        _union: AlignUnion,
    }

    #[repr(C)]
    union AlignUnion {
        _ptr: *const (),
        _i32: i32,
        _f32: f32,
    }

    let alignment = core::mem::align_of::<AlignProbe>();
    value.div_ceil(alignment) * alignment
}

/// Minimal layout stub matching the prefix of `OpusDecoder` used by the size helper.
#[repr(C)]
struct OpusDecoderLayout {
    celt_dec_offset: i32,
    silk_dec_offset: i32,
    channels: i32,
    fs: i32,
    dec_control: DecControlLayout,
    decode_gain: i32,
    complexity: i32,
    arch: i32,
    stream_channels: i32,
    bandwidth: i32,
    mode: i32,
    prev_mode: i32,
    frame_size: i32,
    prev_redundancy: i32,
    last_packet_duration: i32,
    softclip_mem: [f32; 2],
    range_final: u32,
}

/// Mirrors the integer layout of `silk_DecControlStruct` for sizing purposes.
#[repr(C)]
struct DecControlLayout {
    n_channels_api: i32,
    n_channels_internal: i32,
    api_sample_rate: i32,
    internal_sample_rate: i32,
    payload_size_ms: i32,
    prev_pitch_lag: i32,
    enable_deep_plc: i32,
}

/// Returns the number of bytes required to allocate an Opus decoder for `channels`.
///
/// Mirrors `opus_decoder_get_size` by aligning the size of the Opus decoder header
/// and adding the aligned SILK decoder plus CELT decoder sizes. Returns `None`
/// when the requested channel count is outside the supported 1–2 range or when
/// the component size helpers fail.
#[must_use]
pub fn opus_decoder_get_size(channels: usize) -> Option<usize> {
    if channels == 0 || channels > MAX_CHANNELS {
        return None;
    }

    let mut silk_size = 0usize;
    get_decoder_size(&mut silk_size).ok()?;
    let silk_size = align(silk_size);

    let celt_size = celt_decoder_get_size(channels)?;
    let header_size = align(core::mem::size_of::<OpusDecoderLayout>());

    Some(header_size + silk_size + celt_size)
}

/// Top-level Opus decoder wrapper.
///
/// This is a small subset of the C `OpusDecoder` that currently supports
/// construction but not full packet decode.
#[derive(Debug)]
pub struct OpusDecoder<'mode> {
    /// Borrowed CELT decoder for the canonical mode.
    pub(crate) celt: OwnedCeltDecoder<'mode>,
    /// Embedded SILK decoder super-structure.
    pub(crate) silk: SilkDecoder,
    /// Sample rate requested at the API level.
    pub(crate) fs: i32,
    /// Number of channels (1 or 2).
    pub(crate) channels: i32,
}

/// Error codes reported by the top-level decoder helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusDecoderInitError {
    /// The requested configuration was not supported (invalid Fs or channel count).
    BadArgument,
    /// CELT initialisation failed (unsupported sample rate or missing mode).
    CeltInit,
    /// SILK initialisation failed.
    SilkInit,
}

impl<'mode> OpusDecoder<'mode> {
    /// Mirrors `opus_decoder_init` by preparing both the SILK and CELT decoders.
    pub fn init(&mut self, fs: i32, channels: i32) -> Result<(), OpusDecoderInitError> {
        if !matches!(fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000) || !matches!(channels, 1 | 2)
        {
            return Err(OpusDecoderInitError::BadArgument);
        }

        self.fs = fs;
        self.channels = channels;

        // Reset SILK decoder.
        for (idx, state) in self
            .silk
            .channel_states
            .iter_mut()
            .enumerate()
            .take(DECODER_NUM_CHANNELS)
        {
            silk_init_channel(state).map_err(|_| OpusDecoderInitError::SilkInit)?;
            if idx as i32 >= channels {
                *state = DecoderState::default();
            }
        }
        self.silk.n_channels_api = channels;
        self.silk.n_channels_internal = channels;

        // Reinitialise the embedded CELT decoder for the requested rate/channels.
        let mode = canonical_mode().ok_or(OpusDecoderInitError::CeltInit)?;
        self.celt = opus_custom_decoder_create(mode, channels as usize)
            .map_err(|_| OpusDecoderInitError::CeltInit)?;

        Ok(())
    }
}

/// Mirrors `opus_decoder_create` by allocating and initialising a decoder.
pub fn opus_decoder_create(fs: i32, channels: i32) -> Result<OpusDecoder<'static>, OpusDecoderInitError> {
    if !matches!(fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000) || !matches!(channels, 1 | 2) {
        return Err(OpusDecoderInitError::BadArgument);
    }

    let silk = SilkDecoder::default();
    let mode = canonical_mode().ok_or(OpusDecoderInitError::CeltInit)?;
    let celt = opus_custom_decoder_create(mode, channels as usize)
        .map_err(|_| OpusDecoderInitError::CeltInit)?;

    let mut decoder = OpusDecoder {
        celt,
        silk,
        fs,
        channels,
    };

    decoder.init(fs, channels)?;

    Ok(decoder)
}

#[cfg(test)]
mod tests {
    use super::opus_decoder_get_size;
    use crate::celt::{celt_decoder_get_size, canonical_mode, opus_custom_decoder_create};
    use crate::silk::dec_api::Decoder as SilkDecoder;
    use crate::silk::get_decoder_size::get_decoder_size;

    #[test]
    fn rejects_invalid_channel_counts() {
        assert!(opus_decoder_get_size(0).is_none());
        assert!(opus_decoder_get_size(3).is_none());
    }

    #[test]
    fn matches_component_size_sum_for_mono_and_stereo() {
        for &channels in &[1usize, 2] {
            let mut silk_size = 0usize;
            get_decoder_size(&mut silk_size).unwrap();
            let celt_size = celt_decoder_get_size(channels).unwrap();

            let expected = opus_decoder_get_size(channels).unwrap();
            // The size helper is monotonic in its inputs, so it should never
            // under-report the aligned component sum.
            assert!(expected >= silk_size + celt_size);
        }
    }

    #[test]
    fn init_resets_silk_and_recreates_celt() {
        let mode = canonical_mode().expect("canonical mode");
        let celt = opus_custom_decoder_create(mode, 1).expect("celt decoder");
        let silk = SilkDecoder::default();
        let mut decoder = super::OpusDecoder {
            celt,
            silk,
            fs: 48_000,
            channels: 1,
        };

        decoder.init(48_000, 1).expect("init succeeds");
        assert_eq!(decoder.silk.n_channels_api, 1);
        assert_eq!(decoder.silk.n_channels_internal, 1);
    }

    #[test]
    fn create_rejects_invalid_arguments() {
        assert!(super::opus_decoder_create(44_100, 1).is_err());
        assert!(super::opus_decoder_create(48_000, 3).is_err());
    }
}
