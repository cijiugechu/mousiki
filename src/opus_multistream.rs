//! Channel layout helpers mirrored from `opus_multistream.c`.
#![cfg_attr(not(test), allow(dead_code))]

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::{OpusRes, isqrt32};
#[cfg(not(feature = "fixed_point"))]
use crate::celt::{float2int, float2int16};
use crate::opus_decoder::{
    OpusDecodeError, OpusDecoder, OpusDecoderCtlError, OpusDecoderCtlRequest, OpusDecoderInitError,
    opus_decode_native, opus_decoder_create, opus_decoder_ctl, opus_decoder_get_size,
};
use crate::opus_encoder::{
    OpusEncodeError, OpusEncoder, OpusEncoderCtlError, OpusEncoderCtlRequest, OpusEncoderInitError,
    opus_encode, opus_encoder_create, opus_encoder_ctl, opus_encoder_get_size,
};
use crate::packet::{PacketError, opus_packet_get_nb_samples, opus_packet_parse_impl};

/// Sentinel used by the reference encoder when auto-selecting the bitrate.
pub(crate) const OPUS_AUTO: i32 = -1000;
/// Maximum bitrate marker mirrored from the public Opus defines.
pub(crate) const OPUS_BITRATE_MAX: i32 = -1;

/// Mirrors the mapping type enum embedded in the multistream encoder state.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MappingType {
    None,
    Surround,
    Ambisonics,
}

/// Internal multistream channel layout description.
///
/// Mirrors the layout prefix embedded in the multistream encoder/decoder
/// states. The `mapping` table uses `255` as a sentinel for channels that are
/// omitted from the encoded streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelLayout {
    pub nb_channels: usize,
    pub nb_streams: usize,
    pub nb_coupled_streams: usize,
    pub mapping: [u8; 256],
}

/// Errors surfaced by the multistream decoder front-end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusMultistreamDecoderError {
    BadArgument,
    BufferTooSmall,
    InternalError,
    InvalidPacket,
    Unimplemented,
    DecoderInit(OpusDecoderInitError),
    DecoderCtl(OpusDecoderCtlError),
    DecoderDecode(OpusDecodeError),
}

impl OpusMultistreamDecoderError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::InvalidPacket => -4,
            Self::Unimplemented => -5,
            Self::DecoderInit(OpusDecoderInitError::BadArgument) => -1,
            Self::DecoderInit(_) => -3,
            Self::DecoderCtl(OpusDecoderCtlError::BadArgument) => -1,
            Self::DecoderCtl(OpusDecoderCtlError::Unimplemented) => -5,
            Self::DecoderCtl(OpusDecoderCtlError::Silk(_)) => -3,
            Self::DecoderDecode(err) => err.code(),
        }
    }
}

impl From<PacketError> for OpusMultistreamDecoderError {
    #[inline]
    fn from(value: PacketError) -> Self {
        match value {
            PacketError::BadArgument => Self::BadArgument,
            PacketError::InvalidPacket => Self::InvalidPacket,
        }
    }
}

impl From<OpusDecoderInitError> for OpusMultistreamDecoderError {
    #[inline]
    fn from(value: OpusDecoderInitError) -> Self {
        Self::DecoderInit(value)
    }
}

impl From<OpusDecoderCtlError> for OpusMultistreamDecoderError {
    #[inline]
    fn from(value: OpusDecoderCtlError) -> Self {
        Self::DecoderCtl(value)
    }
}

impl From<OpusDecodeError> for OpusMultistreamDecoderError {
    #[inline]
    fn from(value: OpusDecodeError) -> Self {
        Self::DecoderDecode(value)
    }
}

#[repr(C)]
struct ChannelLayoutLayout {
    nb_channels: i32,
    nb_streams: i32,
    nb_coupled_streams: i32,
    mapping: [u8; 256],
}

#[repr(C)]
struct OpusMsDecoderLayout {
    layout: ChannelLayoutLayout,
}

#[repr(C)]
struct OpusMsEncoderLayout {
    layout: ChannelLayoutLayout,
}

/// Mirrors the alignment helper from `opus_private.h`.
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

/// Returns the number of bytes required to allocate a multistream decoder.
#[must_use]
pub fn opus_multistream_decoder_get_size(
    nb_streams: usize,
    nb_coupled_streams: usize,
) -> Option<usize> {
    if nb_streams == 0 || nb_coupled_streams > nb_streams {
        return None;
    }

    let coupled_size = opus_decoder_get_size(2)?;
    let mono_size = opus_decoder_get_size(1)?;
    let header_size = align(core::mem::size_of::<OpusMsDecoderLayout>());

    let coupled_total = nb_coupled_streams.checked_mul(align(coupled_size))?;
    let mono_total = nb_streams
        .checked_sub(nb_coupled_streams)?
        .checked_mul(align(mono_size))?;

    header_size
        .checked_add(coupled_total)?
        .checked_add(mono_total)
}

/// Multistream decoder state mirroring `OpusMSDecoder` from the reference code.
#[derive(Debug)]
pub struct OpusMultistreamDecoder<'mode> {
    layout: ChannelLayout,
    decoders: Vec<OpusDecoder<'mode>>,
}

impl<'mode> OpusMultistreamDecoder<'mode> {
    #[inline]
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    #[inline]
    fn sample_rate(&self) -> Option<i32> {
        self.decoders.first().map(|decoder| decoder.fs)
    }

    /// Resets the decoder to a new layout and sample rate.
    pub fn init(
        &mut self,
        sample_rate: i32,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
    ) -> Result<(), OpusMultistreamDecoderError> {
        let layout = build_layout(channels, streams, coupled_streams, mapping)?;
        let decoders = build_stream_decoders(sample_rate, streams, coupled_streams)?;

        self.layout = layout;
        self.decoders = decoders;
        Ok(())
    }

    /// Returns a mutable reference to the decoder for `stream_id`, mirroring
    /// `OPUS_MULTISTREAM_GET_DECODER_STATE`.
    pub fn decoder_state(&mut self, stream_id: usize) -> Option<&mut OpusDecoder<'mode>> {
        self.decoders.get_mut(stream_id)
    }
}

/// Mirrors `opus_multistream_decoder_create` by allocating and initialising all
/// component decoders.
pub fn opus_multistream_decoder_create(
    sample_rate: i32,
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Result<OpusMultistreamDecoder<'static>, OpusMultistreamDecoderError> {
    let layout = build_layout(channels, streams, coupled_streams, mapping)?;
    let decoders = build_stream_decoders(sample_rate, streams, coupled_streams)?;

    Ok(OpusMultistreamDecoder { layout, decoders })
}

/// Mirrors `opus_multistream_decoder_init` by resetting an existing decoder instance.
pub fn opus_multistream_decoder_init(
    decoder: &mut OpusMultistreamDecoder<'_>,
    sample_rate: i32,
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Result<(), OpusMultistreamDecoderError> {
    decoder.init(sample_rate, channels, streams, coupled_streams, mapping)
}

fn build_layout(
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Result<ChannelLayout, OpusMultistreamDecoderError> {
    if channels == 0
        || channels > 255
        || coupled_streams > streams
        || streams == 0
        || streams > 255 - coupled_streams
    {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    if mapping.len() < channels {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let mut layout = ChannelLayout {
        nb_channels: channels,
        nb_streams: streams,
        nb_coupled_streams: coupled_streams,
        mapping: [u8::MAX; 256],
    };
    layout.mapping[..channels].copy_from_slice(&mapping[..channels]);
    if !validate_layout(&layout) {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    Ok(layout)
}

fn build_stream_decoders(
    sample_rate: i32,
    streams: usize,
    coupled_streams: usize,
) -> Result<Vec<OpusDecoder<'static>>, OpusMultistreamDecoderError> {
    if sample_rate <= 0 {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let mut decoders = Vec::with_capacity(streams);
    for stream in 0..streams {
        let channels: i32 = if stream < coupled_streams { 2 } else { 1 };
        decoders.push(opus_decoder_create(sample_rate, channels)?);
    }

    Ok(decoders)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VorbisLayout {
    nb_streams: usize,
    nb_coupled_streams: usize,
    mapping: [u8; 8],
}

/* Index is nb_channel-1 */
#[allow(dead_code)]
const VORBIS_MAPPINGS: [VorbisLayout; 8] = [
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 0,
        mapping: [0, 255, 255, 255, 255, 255, 255, 255],
    }, /* 1: mono */
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 1,
        mapping: [0, 1, 255, 255, 255, 255, 255, 255],
    }, /* 2: stereo */
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 1,
        mapping: [0, 2, 1, 255, 255, 255, 255, 255],
    }, /* 3: 1-d surround */
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 2,
        mapping: [0, 1, 2, 3, 255, 255, 255, 255],
    }, /* 4: quadraphonic surround */
    VorbisLayout {
        nb_streams: 3,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 255, 255, 255],
    }, /* 5: 5-channel surround */
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 5, 255, 255],
    }, /* 6: 5.1 surround */
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 3,
        mapping: [0, 4, 1, 2, 3, 5, 6, 255],
    }, /* 7: 6.1 surround */
    VorbisLayout {
        nb_streams: 5,
        nb_coupled_streams: 3,
        mapping: [0, 6, 1, 2, 3, 4, 5, 7],
    }, /* 8: 7.1 surround */
];

/// Verifies that the layout only references stream indices that exist.
#[must_use]
pub(crate) fn validate_layout(layout: &ChannelLayout) -> bool {
    let Some(max_channel) = layout.nb_streams.checked_add(layout.nb_coupled_streams) else {
        return false;
    };

    if max_channel > u8::MAX as usize {
        return false;
    }

    if layout.nb_channels > layout.mapping.len() {
        return false;
    }

    layout
        .mapping
        .iter()
        .take(layout.nb_channels)
        .all(|&value| value == u8::MAX || usize::from(value) < max_channel)
}

/// Ensures each stream in the layout has a channel mapping.
#[must_use]
pub(crate) fn validate_encoder_layout(layout: &ChannelLayout) -> bool {
    for stream in 0..layout.nb_streams {
        if stream < layout.nb_coupled_streams {
            if get_left_channel(layout, stream, None).is_none() {
                return false;
            }
            if get_right_channel(layout, stream, None).is_none() {
                return false;
            }
        } else if get_mono_channel(layout, stream, None).is_none() {
            return false;
        }
    }

    true
}

/// Validates the ambisonics channel count and returns the derived stream layout.
#[must_use]
pub(crate) fn validate_ambisonics(channels: usize) -> Option<(usize, usize)> {
    if !(1..=227).contains(&channels) {
        return None;
    }

    let order_plus_one = isqrt32(channels as u32) as usize;
    let acn_channels = order_plus_one.checked_mul(order_plus_one)?;
    let nondiegetic_channels = channels.checked_sub(acn_channels)?;

    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return None;
    }

    let streams = acn_channels + usize::from(nondiegetic_channels != 0);
    let coupled_streams = usize::from(nondiegetic_channels != 0);
    Some((streams, coupled_streams))
}

fn surround_rate_allocation(
    layout: &ChannelLayout,
    bitrate_bps: i32,
    lfe_stream: Option<usize>,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<()> {
    let nb_streams = layout.nb_streams;
    let nb_coupled = layout.nb_coupled_streams;
    let nb_lfe = usize::from(lfe_stream.is_some());
    if nb_streams == 0 || nb_coupled > nb_streams || nb_streams < nb_coupled + nb_lfe {
        return None;
    }
    if frame_size == 0 || sample_rate <= 0 {
        return None;
    }
    if rates.len() < nb_streams {
        return None;
    }

    let nb_uncoupled = nb_streams - nb_coupled - nb_lfe;
    let nb_normal = 2 * nb_coupled + nb_uncoupled;
    if nb_normal == 0 {
        return None;
    }

    let frame_rate = sample_rate / frame_size as i32;
    let channel_offset = 40 * frame_rate.max(50);
    let bitrate = if bitrate_bps == OPUS_AUTO {
        nb_normal as i32 * (channel_offset + sample_rate + 10_000) + 8000 * nb_lfe as i32
    } else if bitrate_bps == OPUS_BITRATE_MAX {
        nb_normal as i32 * 300_000 + nb_lfe as i32 * 128_000
    } else {
        bitrate_bps
    };

    let lfe_offset = bitrate
        .checked_div(20)
        .map(|value| value.min(3000))
        .and_then(|value| value.checked_add(15 * frame_rate.max(50)))?;
    let stream_offset =
        (((bitrate - channel_offset * nb_normal as i32 - lfe_offset * nb_lfe as i32)
            / nb_normal as i32)
            / 2)
        .clamp(0, 20_000);
    let coupled_ratio = 512;
    let lfe_ratio = 32;

    let total = ((nb_uncoupled as i32) << 8)
        + coupled_ratio * nb_coupled as i32
        + lfe_ratio * nb_lfe as i32;
    if total == 0 {
        return None;
    }
    let channel_rate = 256
        * (bitrate
            - lfe_offset * nb_lfe as i32
            - stream_offset * (nb_coupled as i32 + nb_uncoupled as i32)
            - channel_offset * nb_normal as i32)
        / total;

    for (stream, slot) in rates.iter_mut().take(nb_streams).enumerate() {
        let value = if stream < nb_coupled {
            2 * channel_offset + (stream_offset + ((channel_rate * coupled_ratio) >> 8)).max(0)
        } else if lfe_stream == Some(stream) {
            (lfe_offset + ((channel_rate * lfe_ratio) >> 8)).max(0)
        } else {
            channel_offset + (stream_offset + channel_rate).max(0)
        };
        *slot = value;
    }

    Some(())
}

fn ambisonics_rate_allocation(
    layout: &ChannelLayout,
    bitrate_bps: i32,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<()> {
    if frame_size == 0 || sample_rate <= 0 || layout.nb_streams == 0 {
        return None;
    }
    if rates.len() < layout.nb_streams {
        return None;
    }

    let nb_channels = layout.nb_streams + layout.nb_coupled_streams;
    let total_rate = if bitrate_bps == OPUS_AUTO {
        let term = sample_rate + 60 * sample_rate / frame_size as i32;
        (layout.nb_coupled_streams + layout.nb_streams) as i32 * term
            + layout.nb_streams as i32 * 15_000
    } else if bitrate_bps == OPUS_BITRATE_MAX {
        nb_channels as i32 * 320_000
    } else {
        bitrate_bps
    };

    let per_stream_rate = total_rate / layout.nb_streams as i32;
    for slot in rates.iter_mut().take(layout.nb_streams) {
        *slot = per_stream_rate;
    }

    Some(())
}

/// Computes the bitrate distribution across all streams.
#[must_use]
pub(crate) fn rate_allocation(
    layout: &ChannelLayout,
    mapping_type: MappingType,
    bitrate_bps: i32,
    lfe_stream: Option<usize>,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<i32> {
    match mapping_type {
        MappingType::Ambisonics => {
            ambisonics_rate_allocation(layout, bitrate_bps, frame_size, sample_rate, rates)?
        }
        _ => surround_rate_allocation(
            layout,
            bitrate_bps,
            lfe_stream,
            frame_size,
            sample_rate,
            rates,
        )?,
    }

    let mut sum = 0i64;
    for rate in rates.iter_mut().take(layout.nb_streams) {
        *rate = (*rate).max(500);
        sum += i64::from(*rate);
    }

    i32::try_from(sum).ok()
}

fn next_index(prev: Option<usize>) -> Option<usize> {
    match prev {
        Some(idx) => idx.checked_add(1),
        None => Some(0),
    }
}

fn find_channel(layout: &ChannelLayout, start: usize, target: usize) -> Option<usize> {
    let limit = layout.nb_channels.min(layout.mapping.len());
    (start..limit).find(|&i| usize::from(layout.mapping[i]) == target)
}

/// Returns the next channel mapped to the left slot of `stream_id`.
#[must_use]
pub(crate) fn get_left_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_mul(2)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

/// Returns the next channel mapped to the right slot of `stream_id`.
#[must_use]
pub(crate) fn get_right_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_mul(2)?.checked_add(1)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

/// Returns the next channel mapped to the mono stream `stream_id`.
#[must_use]
pub(crate) fn get_mono_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_add(layout.nb_coupled_streams)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

/// Strongly-typed replacement for the multistream decoder CTL dispatcher.
pub enum OpusMultistreamDecoderCtlRequest<'req> {
    SetGain(i32),
    GetGain(&'req mut i32),
    SetComplexity(i32),
    GetComplexity(&'req mut i32),
    GetBandwidth(&'req mut i32),
    GetSampleRate(&'req mut i32),
    GetFinalRange(&'req mut u32),
    ResetState,
    GetLastPacketDuration(&'req mut i32),
    SetPhaseInversionDisabled(bool),
    GetPhaseInversionDisabled(&'req mut bool),
}

/// Applies a control request across the embedded decoders.
pub fn opus_multistream_decoder_ctl<'req>(
    decoder: &mut OpusMultistreamDecoder<'_>,
    request: OpusMultistreamDecoderCtlRequest<'req>,
) -> Result<(), OpusMultistreamDecoderError> {
    if decoder.decoders.is_empty() {
        return Err(OpusMultistreamDecoderError::InternalError);
    }

    match request {
        OpusMultistreamDecoderCtlRequest::SetGain(value) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetGain(value))?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetGain(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetGain(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::SetComplexity(value) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetComplexity(value))?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetComplexity(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetComplexity(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::GetBandwidth(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetBandwidth(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::GetSampleRate(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetSampleRate(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::GetFinalRange(slot) => {
            let mut acc = 0u32;
            for dec in &mut decoder.decoders {
                let mut value = 0u32;
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::GetFinalRange(&mut value))?;
                acc ^= value;
            }
            *slot = acc;
        }
        OpusMultistreamDecoderCtlRequest::ResetState => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::ResetState)?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetLastPacketDuration(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetLastPacketDuration(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::SetPhaseInversionDisabled(value) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetPhaseInversionDisabled(value))?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetPhaseInversionDisabled(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetPhaseInversionDisabled(slot),
            )?;
        }
    }

    Ok(())
}

fn opus_multistream_packet_validate(
    data: &[u8],
    len: usize,
    nb_streams: usize,
    sample_rate: i32,
) -> Result<usize, OpusMultistreamDecoderError> {
    if len > data.len() || nb_streams == 0 || sample_rate <= 0 {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let mut remaining = len;
    let mut cursor = data;
    let mut samples: Option<usize> = None;

    for stream in 0..nb_streams {
        if remaining == 0 {
            return Err(OpusMultistreamDecoderError::InvalidPacket);
        }

        let parsed = opus_packet_parse_impl(cursor, remaining, stream + 1 != nb_streams)?;
        let tmp_samples =
            opus_packet_get_nb_samples(cursor, parsed.packet_offset, sample_rate as u32)?;
        if let Some(prev) = samples {
            if prev != tmp_samples {
                return Err(OpusMultistreamDecoderError::InvalidPacket);
            }
        } else {
            samples = Some(tmp_samples);
        }

        cursor = &cursor[parsed.packet_offset..];
        remaining = remaining
            .checked_sub(parsed.packet_offset)
            .ok_or(OpusMultistreamDecoderError::InvalidPacket)?;
    }

    samples.ok_or(OpusMultistreamDecoderError::InvalidPacket)
}

#[cfg(feature = "fixed_point")]
const OPTIONAL_CLIP: bool = false;
#[cfg(not(feature = "fixed_point"))]
const OPTIONAL_CLIP: bool = true;

fn opus_multistream_decode_native<T: PcmSample>(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [T],
    frame_size: usize,
    decode_fec: bool,
    soft_clip: bool,
) -> Result<usize, OpusMultistreamDecoderError> {
    opus_multistream_decode_native_with_handler(
        decoder,
        data,
        len,
        pcm,
        frame_size,
        decode_fec,
        soft_clip,
        |dst, dst_stride, dst_channel, src, frame_size, src_offset| {
            copy_channel_out(dst, dst_stride, dst_channel, src, frame_size, src_offset);
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn opus_multistream_decode_native_with_handler<T>(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [T],
    mut frame_size: usize,
    decode_fec: bool,
    soft_clip: bool,
    mut handler: impl FnMut(&mut [T], usize, usize, Option<(&[OpusRes], usize)>, usize, usize),
) -> Result<usize, OpusMultistreamDecoderError> {
    if frame_size == 0 {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    if len > data.len() {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let nb_streams = decoder.layout.nb_streams;
    if nb_streams == 0 {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let sample_rate = decoder
        .sample_rate()
        .ok_or(OpusMultistreamDecoderError::InternalError)?;

    let max_frame = sample_rate as usize / 25 * 3;
    frame_size = frame_size.min(max_frame);

    let required = frame_size
        .checked_mul(decoder.layout.nb_channels)
        .ok_or(OpusMultistreamDecoderError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusMultistreamDecoderError::BufferTooSmall);
    }

    let do_plc = len == 0;
    if !do_plc {
        let minimum = nb_streams
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(OpusMultistreamDecoderError::BadArgument)?;
        if len < minimum {
            return Err(OpusMultistreamDecoderError::InvalidPacket);
        }

        let samples = opus_multistream_packet_validate(data, len, nb_streams, sample_rate)?;
        if samples > frame_size {
            return Err(OpusMultistreamDecoderError::BufferTooSmall);
        }
    }

    let mut scratch = vec![OpusRes::default(); 2 * frame_size];
    let mut cursor = data;
    let mut remaining = len;
    let mut decoded_frame_size = frame_size;

    for stream in 0..nb_streams {
        let self_delimited = stream + 1 != nb_streams;

        let decoder_state = decoder
            .decoders
            .get_mut(stream)
            .ok_or(OpusMultistreamDecoderError::InternalError)?;

        let mut packet_offset = 0usize;
        let decoded = if do_plc {
            opus_decode_native(
                decoder_state,
                None,
                0,
                &mut scratch,
                decoded_frame_size,
                decode_fec,
                self_delimited,
                None,
                soft_clip,
            )?
        } else {
            if remaining == 0 {
                return Err(OpusMultistreamDecoderError::InternalError);
            }
            opus_decode_native(
                decoder_state,
                Some(cursor),
                remaining,
                &mut scratch,
                decoded_frame_size,
                decode_fec,
                self_delimited,
                Some(&mut packet_offset),
                soft_clip,
            )?
        };

        if !do_plc {
            if packet_offset > remaining {
                return Err(OpusMultistreamDecoderError::InvalidPacket);
            }
            cursor = &cursor[packet_offset..];
            remaining -= packet_offset;
        }

        if decoded == 0 {
            return Err(OpusMultistreamDecoderError::InternalError);
        }
        decoded_frame_size = decoded;

        if stream < decoder.layout.nb_coupled_streams {
            let mut prev = None;
            while let Some(chan) = get_left_channel(&decoder.layout, stream, prev) {
                handler(
                    pcm,
                    decoder.layout.nb_channels,
                    chan,
                    Some((&scratch[..], 2)),
                    decoded_frame_size,
                    0,
                );
                prev = Some(chan);
            }

            let mut prev = None;
            while let Some(chan) = get_right_channel(&decoder.layout, stream, prev) {
                handler(
                    pcm,
                    decoder.layout.nb_channels,
                    chan,
                    Some((&scratch[1..], 2)),
                    decoded_frame_size,
                    0,
                );
                prev = Some(chan);
            }
        } else {
            let mut prev = None;
            while let Some(chan) = get_mono_channel(&decoder.layout, stream, prev) {
                handler(
                    pcm,
                    decoder.layout.nb_channels,
                    chan,
                    Some((&scratch[..], 1)),
                    decoded_frame_size,
                    0,
                );
                prev = Some(chan);
            }
        }
    }

    // Handle muted channels.
    for channel in 0..decoder.layout.nb_channels {
        if decoder.layout.mapping[channel] == u8::MAX {
            handler(
                pcm,
                decoder.layout.nb_channels,
                channel,
                None,
                decoded_frame_size,
                0,
            );
        }
    }

    Ok(decoded_frame_size)
}

pub fn opus_multistream_decode(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [i16],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusMultistreamDecoderError> {
    opus_multistream_decode_native(
        decoder,
        data,
        len,
        pcm,
        frame_size,
        decode_fec,
        OPTIONAL_CLIP,
    )
}

pub fn opus_multistream_decode24(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [i32],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusMultistreamDecoderError> {
    opus_multistream_decode_native(decoder, data, len, pcm, frame_size, decode_fec, false)
}

pub fn opus_multistream_decode_float(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [f32],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusMultistreamDecoderError> {
    opus_multistream_decode_native(decoder, data, len, pcm, frame_size, decode_fec, false)
}

trait PcmSample: Copy + Default {
    fn from_opus_res(value: OpusRes) -> Self;
}

impl PcmSample for i16 {
    #[inline]
    fn from_opus_res(value: OpusRes) -> Self {
        #[cfg(feature = "fixed_point")]
        {
            value as i16
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            float2int16(value)
        }
    }
}

impl PcmSample for i32 {
    #[inline]
    fn from_opus_res(value: OpusRes) -> Self {
        #[cfg(feature = "fixed_point")]
        {
            i32::from(value as i16) << 8
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            float2int(value * 8_388_608.0)
        }
    }
}

impl PcmSample for f32 {
    #[inline]
    fn from_opus_res(value: OpusRes) -> Self {
        #[cfg(feature = "fixed_point")]
        {
            f32::from(value as i16) * (1.0 / 32_768.0)
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            value
        }
    }
}

fn copy_channel_out<T: PcmSample>(
    dst: &mut [T],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<(&[OpusRes], usize)>,
    frame_size: usize,
    src_offset: usize,
) {
    if dst_stride == 0 || frame_size == 0 {
        return;
    }
    for i in 0..frame_size {
        let dst_index = i * dst_stride + dst_channel;
        if dst_index >= dst.len() {
            break;
        }
        dst[dst_index] = match src {
            Some((src_data, src_stride)) => {
                let src_index = src_offset + i * src_stride;
                let value = src_data.get(src_index).copied().unwrap_or_default();
                T::from_opus_res(value)
            }
            None => T::default(),
        };
    }
}

/// Errors surfaced by the multistream encoder front-end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusMultistreamEncoderError {
    BadArgument,
    BufferTooSmall,
    InternalError,
    Unimplemented,
    EncoderInit(OpusEncoderInitError),
    EncoderCtl(OpusEncoderCtlError),
    Encode(OpusEncodeError),
}

impl OpusMultistreamEncoderError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::Unimplemented => -5,
            Self::EncoderInit(err) => err.code(),
            Self::EncoderCtl(err) => err.code(),
            Self::Encode(err) => err.code(),
        }
    }
}

impl From<OpusEncoderInitError> for OpusMultistreamEncoderError {
    #[inline]
    fn from(value: OpusEncoderInitError) -> Self {
        Self::EncoderInit(value)
    }
}

impl From<OpusEncoderCtlError> for OpusMultistreamEncoderError {
    #[inline]
    fn from(value: OpusEncoderCtlError) -> Self {
        Self::EncoderCtl(value)
    }
}

impl From<OpusEncodeError> for OpusMultistreamEncoderError {
    #[inline]
    fn from(value: OpusEncodeError) -> Self {
        match value {
            OpusEncodeError::BadArgument => Self::BadArgument,
            OpusEncodeError::BufferTooSmall => Self::BufferTooSmall,
            OpusEncodeError::InternalError => Self::InternalError,
            OpusEncodeError::Unimplemented => Self::Unimplemented,
            OpusEncodeError::Silk(_) => Self::Encode(value),
        }
    }
}

/// Multistream encoder state mirroring `OpusMSEncoder` from the reference code.
#[derive(Debug)]
pub struct OpusMultistreamEncoder<'mode> {
    layout: ChannelLayout,
    encoders: Vec<OpusEncoder<'mode>>,
    mapping_type: MappingType,
    lfe_stream: Option<usize>,
    sample_rate: i32,
    application: i32,
    bitrate_bps: i32,
}

impl<'mode> OpusMultistreamEncoder<'mode> {
    #[inline]
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Resets the encoder to a new layout and sample rate.
    pub fn init(
        &mut self,
        sample_rate: i32,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
        application: i32,
    ) -> Result<(), OpusMultistreamEncoderError> {
        let layout = build_encoder_layout(channels, streams, coupled_streams, mapping)?;
        let encoders = build_stream_encoders(sample_rate, streams, coupled_streams, application)?;

        self.layout = layout;
        self.encoders = encoders;
        self.mapping_type = MappingType::None;
        self.lfe_stream = None;
        self.sample_rate = sample_rate;
        self.application = application;
        self.bitrate_bps = OPUS_AUTO;
        Ok(())
    }

    /// Returns a mutable reference to the encoder for `stream_id`, mirroring
    /// `OPUS_MULTISTREAM_GET_ENCODER_STATE`.
    pub fn encoder_state(&mut self, stream_id: usize) -> Option<&mut OpusEncoder<'mode>> {
        self.encoders.get_mut(stream_id)
    }
}

/// Returns the number of bytes required to allocate a multistream encoder.
#[must_use]
pub fn opus_multistream_encoder_get_size(
    nb_streams: usize,
    nb_coupled_streams: usize,
) -> Option<usize> {
    if nb_streams == 0 || nb_coupled_streams > nb_streams {
        return None;
    }

    let coupled_size = opus_encoder_get_size(2)?;
    let mono_size = opus_encoder_get_size(1)?;
    let header_size = align(core::mem::size_of::<OpusMsEncoderLayout>());

    let coupled_total = nb_coupled_streams.checked_mul(align(coupled_size))?;
    let mono_total = nb_streams
        .checked_sub(nb_coupled_streams)?
        .checked_mul(align(mono_size))?;

    header_size
        .checked_add(coupled_total)?
        .checked_add(mono_total)
}

/// Mirrors `opus_multistream_encoder_create` by allocating and initialising all
/// component encoders.
pub fn opus_multistream_encoder_create(
    sample_rate: i32,
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
    application: i32,
) -> Result<OpusMultistreamEncoder<'static>, OpusMultistreamEncoderError> {
    let layout = build_encoder_layout(channels, streams, coupled_streams, mapping)?;
    let encoders = build_stream_encoders(sample_rate, streams, coupled_streams, application)?;

    Ok(OpusMultistreamEncoder {
        layout,
        encoders,
        mapping_type: MappingType::None,
        lfe_stream: None,
        sample_rate,
        application,
        bitrate_bps: OPUS_AUTO,
    })
}

fn build_encoder_layout(
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Result<ChannelLayout, OpusMultistreamEncoderError> {
    let layout = build_layout(channels, streams, coupled_streams, mapping)
        .map_err(|err| match err {
            OpusMultistreamDecoderError::BadArgument => OpusMultistreamEncoderError::BadArgument,
            _ => OpusMultistreamEncoderError::InternalError,
        })?;
    if streams
        .checked_add(coupled_streams)
        .is_none_or(|total| total > channels)
    {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    if !validate_encoder_layout(&layout) {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    Ok(layout)
}

fn build_stream_encoders(
    sample_rate: i32,
    streams: usize,
    coupled_streams: usize,
    application: i32,
) -> Result<Vec<OpusEncoder<'static>>, OpusMultistreamEncoderError> {
    if sample_rate <= 0 {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    let mut encoders = Vec::with_capacity(streams);
    for stream in 0..streams {
        let channels: i32 = if stream < coupled_streams { 2 } else { 1 };
        encoders.push(opus_encoder_create(sample_rate, channels, application)?);
    }
    Ok(encoders)
}

/// Strongly-typed replacement for the multistream encoder CTL dispatcher.
pub enum OpusMultistreamEncoderCtlRequest<'req> {
    SetBitrate(i32),
    GetBitrate(&'req mut i32),
    SetVbr(bool),
    GetVbr(&'req mut bool),
    SetVbrConstraint(bool),
    GetVbrConstraint(&'req mut bool),
    SetComplexity(i32),
    GetComplexity(&'req mut i32),
    SetPacketLossPerc(i32),
    GetPacketLossPerc(&'req mut i32),
    SetInbandFec(bool),
    GetInbandFec(&'req mut bool),
    SetDtx(bool),
    GetDtx(&'req mut bool),
    GetFinalRange(&'req mut u32),
    ResetState,
}

/// Applies a control request across the embedded encoders.
pub fn opus_multistream_encoder_ctl<'req>(
    encoder: &mut OpusMultistreamEncoder<'_>,
    request: OpusMultistreamEncoderCtlRequest<'req>,
) -> Result<(), OpusMultistreamEncoderError> {
    if encoder.encoders.is_empty() {
        return Err(OpusMultistreamEncoderError::InternalError);
    }

    match request {
        OpusMultistreamEncoderCtlRequest::SetBitrate(value) => {
            if value != OPUS_AUTO && value != OPUS_BITRATE_MAX && value <= 0 {
                return Err(OpusMultistreamEncoderError::BadArgument);
            }
            encoder.bitrate_bps = value;
        }
        OpusMultistreamEncoderCtlRequest::GetBitrate(out) => {
            *out = encoder.bitrate_bps;
        }
        OpusMultistreamEncoderCtlRequest::SetVbr(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetVbr(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetVbr(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetVbr(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetVbrConstraint(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetVbrConstraint(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetVbrConstraint(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetVbrConstraint(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetComplexity(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetComplexity(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetComplexity(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetComplexity(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetPacketLossPerc(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetPacketLossPerc(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetPacketLossPerc(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetInbandFec(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetInbandFec(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetInbandFec(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetInbandFec(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetDtx(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetDtx(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetDtx(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetDtx(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::GetFinalRange(out) => {
            let mut acc = 0u32;
            for enc in &mut encoder.encoders {
                let mut value = 0u32;
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::GetFinalRange(&mut value))?;
                acc ^= value;
            }
            *out = acc;
        }
        OpusMultistreamEncoderCtlRequest::ResetState => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::ResetState)?;
            }
        }
    }

    Ok(())
}

fn encode_size(size: usize, data: &mut [u8]) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    if size < 252 {
        data[0] = size as u8;
        Some(1)
    } else {
        if data.len() < 2 {
            return None;
        }
        data[0] = 252 + (size & 0x3) as u8;
        data[1] = ((size - usize::from(data[0])) >> 2) as u8;
        Some(2)
    }
}

fn extract_i16_channel(
    input: &[i16],
    input_channels: usize,
    channel: usize,
    frame_size: usize,
    output: &mut [i16],
    output_stride: usize,
    output_offset: usize,
) -> Result<(), OpusMultistreamEncoderError> {
    for i in 0..frame_size {
        let src = i
            .checked_mul(input_channels)
            .and_then(|base| base.checked_add(channel))
            .ok_or(OpusMultistreamEncoderError::BadArgument)?;
        let dst = output_offset
            .checked_add(i.checked_mul(output_stride).ok_or(OpusMultistreamEncoderError::BadArgument)?)
            .ok_or(OpusMultistreamEncoderError::BadArgument)?;
        output[dst] = *input.get(src).ok_or(OpusMultistreamEncoderError::BadArgument)?;
    }
    Ok(())
}

pub fn opus_multistream_encode(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[i16],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusMultistreamEncoderError> {
    let channels = encoder.layout.nb_channels;
    if channels == 0 || frame_size == 0 {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    let required_pcm = channels
        .checked_mul(frame_size)
        .ok_or(OpusMultistreamEncoderError::BadArgument)?;
    if pcm.len() < required_pcm {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }

    let nb_streams = encoder.layout.nb_streams;
    if nb_streams == 0 {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }

    let smallest_packet = nb_streams
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(OpusMultistreamEncoderError::BadArgument)?;
    if data.len() < smallest_packet {
        return Err(OpusMultistreamEncoderError::BufferTooSmall);
    }

    let mut rates = vec![0i32; nb_streams];
    let _ = rate_allocation(
        &encoder.layout,
        encoder.mapping_type,
        encoder.bitrate_bps,
        encoder.lfe_stream,
        frame_size,
        encoder.sample_rate,
        &mut rates,
    );

    let mut total_written = 0usize;
    for stream in 0..nb_streams {
        let self_delimited = stream + 1 != nb_streams;
        let remaining_streams = nb_streams - stream - 1;
        let reserve_min = remaining_streams
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .unwrap_or(0);
        let available = data
            .len()
            .checked_sub(total_written)
            .and_then(|value| value.checked_sub(reserve_min))
            .ok_or(OpusMultistreamEncoderError::BufferTooSmall)?;

        // Worst-case 2 bytes for the self-delimiting size.
        let size_overhead = if self_delimited { 2 } else { 0 };
        if available <= size_overhead + 1 {
            return Err(OpusMultistreamEncoderError::BufferTooSmall);
        }
        let mut tmp = vec![0u8; available - size_overhead];

        if let Some(rate) = rates.get(stream).copied() {
            let _ = opus_encoder_ctl(
                encoder
                    .encoders
                    .get_mut(stream)
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::SetBitrate(rate),
            );
        }

        if stream < encoder.layout.nb_coupled_streams {
            let left = get_left_channel(&encoder.layout, stream, None)
                .ok_or(OpusMultistreamEncoderError::BadArgument)?;
            let right = get_right_channel(&encoder.layout, stream, None)
                .ok_or(OpusMultistreamEncoderError::BadArgument)?;

            let mut coupled_pcm = vec![0i16; frame_size * 2];
            extract_i16_channel(pcm, channels, left, frame_size, &mut coupled_pcm, 2, 0)?;
            extract_i16_channel(pcm, channels, right, frame_size, &mut coupled_pcm, 2, 1)?;

            let len = opus_encode(
                encoder
                    .encoders
                    .get_mut(stream)
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                &coupled_pcm,
                frame_size,
                &mut tmp,
            )?;

            let written = write_stream_packet(&tmp[..len], self_delimited, &mut data[total_written..])
                .ok_or(OpusMultistreamEncoderError::BufferTooSmall)?;
            total_written += written;
        } else {
            let chan = get_mono_channel(&encoder.layout, stream, None)
                .ok_or(OpusMultistreamEncoderError::BadArgument)?;
            let mut mono_pcm = vec![0i16; frame_size];
            extract_i16_channel(pcm, channels, chan, frame_size, &mut mono_pcm, 1, 0)?;

            let len = opus_encode(
                encoder
                    .encoders
                    .get_mut(stream)
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                &mono_pcm,
                frame_size,
                &mut tmp,
            )?;

            let written = write_stream_packet(&tmp[..len], self_delimited, &mut data[total_written..])
                .ok_or(OpusMultistreamEncoderError::BufferTooSmall)?;
            total_written += written;
        }
    }

    Ok(total_written)
}

fn write_stream_packet(stream_packet: &[u8], self_delimited: bool, out: &mut [u8]) -> Option<usize> {
    if !self_delimited {
        if out.len() < stream_packet.len() {
            return None;
        }
        out[..stream_packet.len()].copy_from_slice(stream_packet);
        return Some(stream_packet.len());
    }
    if stream_packet.is_empty() {
        return None;
    }
    let toc = stream_packet[0];
    let frame = &stream_packet[1..];
    if out.is_empty() {
        return None;
    }
    out[0] = toc;
    let size_bytes = encode_size(frame.len(), &mut out[1..])?;
    let start = 1 + size_bytes;
    if out.len() < start + frame.len() {
        return None;
    }
    out[start..start + frame.len()].copy_from_slice(frame);
    Some(start + frame.len())
}

pub fn opus_multistream_encode_float(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusMultistreamEncoderError> {
    let channels = encoder.layout.nb_channels;
    let required_pcm = channels
        .checked_mul(frame_size)
        .ok_or(OpusMultistreamEncoderError::BadArgument)?;
    if pcm.len() < required_pcm {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    let mut tmp = vec![0i16; required_pcm];
    for (dst, &sample) in tmp.iter_mut().zip(pcm.iter().take(required_pcm)) {
        let scaled = libm::roundf(sample * 32_768.0);
        *dst = scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
    }
    opus_multistream_encode(encoder, &tmp, frame_size, data)
}

pub fn opus_multistream_encode24(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusMultistreamEncoderError> {
    let channels = encoder.layout.nb_channels;
    let required_pcm = channels
        .checked_mul(frame_size)
        .ok_or(OpusMultistreamEncoderError::BadArgument)?;
    if pcm.len() < required_pcm {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    let mut tmp = vec![0i16; required_pcm];
    for (dst, &sample) in tmp.iter_mut().zip(pcm.iter().take(required_pcm)) {
        let shifted = (sample >> 8).clamp(i32::from(i16::MIN), i32::from(i16::MAX));
        *dst = shifted as i16;
    }
    opus_multistream_encode(encoder, &tmp, frame_size, data)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusMultistreamSurroundLayout {
    pub streams: usize,
    pub coupled_streams: usize,
    pub mapping: Vec<u8>,
}

pub fn opus_multistream_surround_encoder_get_size(
    channels: usize,
    mapping_family: u8,
) -> Option<usize> {
    let (streams, coupled_streams, _) = surround_layout(channels, mapping_family).ok()?;
    let mut size = opus_multistream_encoder_get_size(streams, coupled_streams)?;
    if channels > 2 {
        size = size.checked_add(
            channels
                .checked_mul(
                    120usize
                        .checked_mul(core::mem::size_of::<crate::celt::OpusVal32>())?
                        .checked_add(core::mem::size_of::<crate::celt::OpusVal32>())?,
                )?,
        )?;
    }
    Some(size)
}

pub fn opus_multistream_surround_encoder_create(
    sample_rate: i32,
    channels: usize,
    mapping_family: u8,
    application: i32,
) -> Result<(OpusMultistreamEncoder<'static>, OpusMultistreamSurroundLayout), OpusMultistreamEncoderError>
{
    let (streams, coupled_streams, mapping) = surround_layout(channels, mapping_family)?;
    let mut encoder = opus_multistream_encoder_create(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        &mapping,
        application,
    )?;

    let (mapping_type, lfe_stream) = surround_mapping_type(channels, mapping_family, streams);
    encoder.mapping_type = mapping_type;
    encoder.lfe_stream = lfe_stream;

    Ok((
        encoder,
        OpusMultistreamSurroundLayout {
            streams,
            coupled_streams,
            mapping,
        },
    ))
}

pub fn opus_multistream_surround_encoder_init(
    encoder: &mut OpusMultistreamEncoder<'_>,
    sample_rate: i32,
    channels: usize,
    mapping_family: u8,
    application: i32,
) -> Result<OpusMultistreamSurroundLayout, OpusMultistreamEncoderError> {
    let (streams, coupled_streams, mapping) = surround_layout(channels, mapping_family)?;
    encoder.init(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        &mapping,
        application,
    )?;

    let (mapping_type, lfe_stream) = surround_mapping_type(channels, mapping_family, streams);
    encoder.mapping_type = mapping_type;
    encoder.lfe_stream = lfe_stream;

    Ok(OpusMultistreamSurroundLayout {
        streams,
        coupled_streams,
        mapping,
    })
}

fn surround_mapping_type(
    channels: usize,
    mapping_family: u8,
    streams: usize,
) -> (MappingType, Option<usize>) {
    match mapping_family {
        1 if channels > 2 => (
            MappingType::Surround,
            if channels >= 6 {
                streams.checked_sub(1)
            } else {
                None
            },
        ),
        2 => (MappingType::Ambisonics, None),
        _ => (MappingType::None, None),
    }
}

fn surround_layout(
    channels: usize,
    mapping_family: u8,
) -> Result<(usize, usize, Vec<u8>), OpusMultistreamEncoderError> {
    if channels == 0 || channels > 255 {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }

    let mut mapping = Vec::with_capacity(channels);
    match mapping_family {
        0 => match channels {
            1 => {
                mapping.push(0);
                Ok((1, 0, mapping))
            }
            2 => {
                mapping.extend_from_slice(&[0, 1]);
                Ok((1, 1, mapping))
            }
            _ => Err(OpusMultistreamEncoderError::Unimplemented),
        },
        1 => {
            if !(1..=8).contains(&channels) {
                return Err(OpusMultistreamEncoderError::Unimplemented);
            }
            let layout = &VORBIS_MAPPINGS[channels - 1];
            mapping.extend_from_slice(&layout.mapping[..channels]);
            Ok((layout.nb_streams, layout.nb_coupled_streams, mapping))
        }
        255 => {
            mapping.extend((0..channels).map(|v| v as u8));
            Ok((channels, 0, mapping))
        }
        2 => {
            let (streams, coupled_streams) = validate_ambisonics(channels)
                .ok_or(OpusMultistreamEncoderError::BadArgument)?;
            let uncoupled_streams = streams - coupled_streams;
            for i in 0..uncoupled_streams {
                mapping.push((i + coupled_streams * 2) as u8);
            }
            for i in 0..coupled_streams * 2 {
                mapping.push(i as u8);
            }
            Ok((streams, coupled_streams, mapping))
        }
        _ => Err(OpusMultistreamEncoderError::Unimplemented),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use crate::opus_decoder::{opus_decode, opus_decoder_create};

    fn layout_from_mapping(
        nb_channels: usize,
        nb_streams: usize,
        nb_coupled_streams: usize,
        mapping_slice: &[u8],
    ) -> ChannelLayout {
        let mut mapping = [u8::MAX; 256];
        let count = mapping_slice.len().min(mapping.len());
        mapping[..count].copy_from_slice(&mapping_slice[..count]);
        ChannelLayout {
            nb_channels,
            nb_streams,
            nb_coupled_streams,
            mapping,
        }
    }

    #[test]
    fn accepts_valid_layout() {
        let layout = layout_from_mapping(3, 2, 1, &[0, 1, 2]);
        assert!(validate_layout(&layout));
    }

    #[test]
    fn rejects_out_of_range_stream_indices() {
        let layout = layout_from_mapping(1, 1, 0, &[1]);
        assert!(!validate_layout(&layout));
    }

    #[test]
    fn rejects_when_stream_count_exceeds_byte_limit() {
        let layout = layout_from_mapping(2, 200, 60, &[0, 1]);
        assert!(!validate_layout(&layout));
    }

    #[test]
    fn iterates_all_channels_for_a_stream() {
        let layout = layout_from_mapping(4, 3, 1, &[0, 0, 1, 2]);

        assert_eq!(get_left_channel(&layout, 0, None), Some(0));
        assert_eq!(get_left_channel(&layout, 0, Some(0)), Some(1));
        assert_eq!(get_left_channel(&layout, 0, Some(1)), None);

        assert_eq!(get_right_channel(&layout, 0, None), Some(2));
        assert_eq!(get_right_channel(&layout, 0, Some(2)), None);

        assert_eq!(get_mono_channel(&layout, 1, None), Some(3));
        assert_eq!(get_mono_channel(&layout, 1, Some(3)), None);
    }

    #[test]
    fn validate_encoder_layout_rejects_missing_channel() {
        let layout = layout_from_mapping(2, 1, 1, &[0, u8::MAX]);
        assert!(!validate_encoder_layout(&layout));
    }

    #[test]
    fn validate_encoder_layout_accepts_complete_mapping() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        assert!(validate_encoder_layout(&layout));
    }

    #[test]
    fn validate_ambisonics_computes_stream_and_coupling_counts() {
        assert_eq!(validate_ambisonics(1), Some((1, 0)));
        assert_eq!(validate_ambisonics(6), Some((5, 1)));
        assert_eq!(validate_ambisonics(7), None);
        assert_eq!(validate_ambisonics(228), None);
    }

    #[test]
    fn rate_allocation_handles_stereo_surround_defaults() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        let mut rates = [0; 1];

        let sum = rate_allocation(
            &layout,
            MappingType::Surround,
            OPUS_AUTO,
            None,
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 120_000);
        assert_eq!(rates, [120_000]);
    }

    #[test]
    fn rate_allocation_accounts_for_lfe_stream() {
        let layout = layout_from_mapping(6, 4, 2, &[0, 4, 1, 2, 3, 5]);
        let mut rates = [0; 4];

        let sum = rate_allocation(
            &layout,
            MappingType::Surround,
            256_000,
            Some(3),
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 255_995);
        assert_eq!(rates, [95_120, 95_120, 57_560, 8_195]);
    }

    #[test]
    fn rate_allocation_splits_evenly_for_ambisonics() {
        let layout = layout_from_mapping(4, 4, 0, &[0, 1, 2, 3]);
        let mut rates = [0; 4];

        let sum = rate_allocation(
            &layout,
            MappingType::Ambisonics,
            OPUS_AUTO,
            None,
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 264_000);
        assert_eq!(rates, [66_000; 4]);
    }

    #[test]
    fn rate_allocation_rejects_insufficient_rate_storage() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        let mut rates: [i32; 0] = [];

        assert!(
            rate_allocation(
                &layout,
                MappingType::Surround,
                64_000,
                None,
                960,
                48_000,
                &mut rates
            )
            .is_none()
        );
    }

    #[test]
    fn multistream_decoder_size_matches_aligned_components() {
        let coupled = opus_decoder_get_size(2).expect("coupled size");
        let mono = opus_decoder_get_size(1).expect("mono size");
        let expected =
            align(core::mem::size_of::<OpusMsDecoderLayout>()) + align(coupled) + align(mono);
        let reported = opus_multistream_decoder_get_size(2, 1).expect("reported size");

        assert_eq!(reported, expected);
        assert!(opus_multistream_decoder_get_size(0, 0).is_none());
        assert!(opus_multistream_decoder_get_size(2, 3).is_none());
    }

    #[test]
    fn multistream_decoder_creation_validates_arguments() {
        let mapping = [0u8, 1];
        let err = opus_multistream_decoder_create(48_000, 0, 1, 0, &mapping).unwrap_err();
        assert_eq!(err, OpusMultistreamDecoderError::BadArgument);

        let short_map = [0u8];
        let err = opus_multistream_decoder_create(48_000, 2, 1, 1, &short_map).unwrap_err();
        assert_eq!(err, OpusMultistreamDecoderError::BadArgument);
    }

    #[test]
    fn multistream_decoder_ctl_round_trips_gain_and_complexity() {
        let mapping = [0u8, 1];
        let mut decoder =
            opus_multistream_decoder_create(48_000, 2, 1, 1, &mapping).expect("decoder");

        opus_multistream_decoder_ctl(&mut decoder, OpusMultistreamDecoderCtlRequest::SetGain(-12))
            .unwrap();

        let mut gain = 0;
        opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::GetGain(&mut gain),
        )
        .unwrap();
        assert_eq!(gain, -12);

        opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::SetComplexity(5),
        )
        .unwrap();
        let mut complexity = 0;
        opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::GetComplexity(&mut complexity),
        )
        .unwrap();
        assert_eq!(complexity, 5);

        let mut fs = 0;
        opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::GetSampleRate(&mut fs),
        )
        .unwrap();
        assert_eq!(fs, 48_000);

        let err = opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::SetComplexity(11),
        )
        .unwrap_err();
        assert_eq!(
            err,
            OpusMultistreamDecoderError::DecoderCtl(OpusDecoderCtlError::BadArgument)
        );
    }

    #[test]
    fn packet_validation_accepts_self_delimited_streams() {
        // First stream is self-delimited (size=1), second stream is the tail packet.
        let packet = [0x00, 0x01, 0xAA, 0x00, 0xBB];
        let samples =
            opus_multistream_packet_validate(&packet, packet.len(), 2, 48_000).expect("packet");
        assert_eq!(samples, 480);
    }

    #[test]
    fn packet_validation_rejects_mismatched_sample_counts() {
        // Second stream advertises a 60 ms frame, which does not match the first stream.
        let packet = [0x00, 0x01, 0xAA, 0x18, 0xBB];
        let err = opus_multistream_packet_validate(&packet, packet.len(), 2, 48_000).unwrap_err();
        assert_eq!(err, OpusMultistreamDecoderError::InvalidPacket);
    }

    #[test]
    fn decode_returns_unimplemented_after_validation() {
        let mapping = [0u8, 1];
        let mut decoder =
            opus_multistream_decoder_create(48_000, 2, 1, 1, &mapping).expect("decoder");
        let packet = [0x00, 0xAA];
        let mut pcm = vec![0i16; 2 * 960];

        let decoded =
            opus_multistream_decode(&mut decoder, &packet, packet.len(), &mut pcm, 960, false)
                .expect("decode");
        assert!(decoded > 0);
    }

    #[test]
    fn multistream_decode_matches_single_stream_decoder_for_stereo() {
        let mapping = [0u8, 1];
        let mut encoder = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        let pcm_in = vec![0i16; 2 * 960];
        let mut packet = vec![0u8; 1500];
        let len = opus_encode(&mut encoder, &pcm_in, 960, &mut packet).expect("encode");

        let mut ms_decoder =
            opus_multistream_decoder_create(48_000, 2, 1, 1, &mapping).expect("ms decoder");
        let mut single = opus_decoder_create(48_000, 2).expect("single decoder");

        let mut ms_out = vec![0i16; 2 * 960];
        let mut single_out = vec![0i16; 2 * 960];

        let ms_decoded =
            opus_multistream_decode(&mut ms_decoder, &packet, len, &mut ms_out, 960, false)
                .expect("ms decode");
        let single_decoded =
            opus_decode(&mut single, Some(&packet[..len]), len, &mut single_out, 960, false)
                .expect("single decode");
        assert_eq!(ms_decoded, single_decoded);
        assert_eq!(ms_out[..ms_decoded * 2], single_out[..single_decoded * 2]);
    }

    #[test]
    fn multistream_decode_routes_two_coupled_streams() {
        let mapping = [0u8, 1, 2, 3];
        let mut ms_encoder =
            opus_multistream_encoder_create(48_000, 4, 2, 2, &mapping, 2048).expect("ms encoder");
        let pcm_in = vec![0i16; 4 * 960];
        let mut packet = vec![0u8; 4000];
        let len = opus_multistream_encode(&mut ms_encoder, &pcm_in, 960, &mut packet).expect("encode");

        let mut ms_decoder =
            opus_multistream_decoder_create(48_000, 4, 2, 2, &mapping).expect("ms decoder");

        let mut ms_out = vec![0i16; 4 * 960];
        let decoded = opus_multistream_decode(&mut ms_decoder, &packet, len, &mut ms_out, 960, false)
            .expect("ms decode");
        assert!(decoded > 0);

        let mut offset = 0usize;
        let mut remaining = len;
        let mut expected = vec![0i16; 4 * decoded];
        for stream in 0..2 {
            let self_delimited = stream == 0;
            let parsed = opus_packet_parse_impl(&packet[offset..], remaining, self_delimited).expect("parse");
            let packet_offset = parsed.packet_offset;
            let mut dec = opus_decoder_create(48_000, 2).expect("decoder");
            let mut stream_out = vec![0i16; 2 * decoded];
            let stream_packet = if self_delimited {
                let toc = packet[offset];
                let payload_start = offset + parsed.payload_offset;
                let payload_end = offset + parsed.packet_offset;
                let mut reconstructed = vec![0u8; 1 + (payload_end - payload_start)];
                reconstructed[0] = toc;
                reconstructed[1..].copy_from_slice(&packet[payload_start..payload_end]);
                reconstructed
            } else {
                packet[offset..offset + packet_offset].to_vec()
            };

            let stream_decoded = opus_decode(
                &mut dec,
                Some(&stream_packet),
                stream_packet.len(),
                &mut stream_out,
                decoded,
                false,
            )
            .expect("decode");
            assert_eq!(stream_decoded, decoded);

            for i in 0..decoded {
                expected[i * 4 + stream * 2] = stream_out[i * 2];
                expected[i * 4 + stream * 2 + 1] = stream_out[i * 2 + 1];
            }

            offset += packet_offset;
            remaining -= packet_offset;
        }

        assert_eq!(ms_out[..decoded * 4], expected[..decoded * 4]);
    }

    #[test]
    fn multistream_decode_zero_fills_muted_channel() {
        let mapping = [0u8, 1, u8::MAX];
        let mut encoder = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        let pcm_in = vec![0i16; 2 * 960];
        let mut packet = vec![0u8; 1500];
        let len = opus_encode(&mut encoder, &pcm_in, 960, &mut packet).expect("encode");

        let mut ms_decoder =
            opus_multistream_decoder_create(48_000, 3, 1, 1, &mapping).expect("ms decoder");
        let mut out = vec![0i16; 3 * 960];
        let decoded = opus_multistream_decode(&mut ms_decoder, &packet, len, &mut out, 960, false)
            .expect("decode");

        for i in 0..decoded {
            assert_eq!(out[i * 3 + 2], 0);
        }
    }

    #[test]
    fn multistream_encode_respects_minimum_packet_size() {
        let mapping = [0u8, 1];
        let mut ms_encoder =
            opus_multistream_encoder_create(48_000, 2, 1, 1, &mapping, 2048).expect("ms encoder");
        let pcm_in = vec![0i16; 2 * 960];
        let mut packet = vec![0u8; 2];
        let err = opus_multistream_encode(&mut ms_encoder, &pcm_in, 960, &mut packet).unwrap_err();
        assert_eq!(err, OpusMultistreamEncoderError::BufferTooSmall);
    }

    #[test]
    fn surround_encoder_layout_matches_vorbis_mapping() {
        let (encoder, layout) =
            opus_multistream_surround_encoder_create(48_000, 6, 1, 2048).expect("encoder");

        assert_eq!(layout.streams, 4);
        assert_eq!(layout.coupled_streams, 2);
        assert_eq!(layout.mapping, vec![0, 4, 1, 2, 3, 5]);
        assert_eq!(encoder.layout.nb_channels, 6);
        assert_eq!(encoder.layout.nb_streams, 4);
        assert_eq!(encoder.layout.nb_coupled_streams, 2);

        // LFE is the last stream for 5.1+ layouts.
        assert_eq!(encoder.lfe_stream, Some(3));
        assert_eq!(encoder.mapping_type, MappingType::Surround);
    }

    #[test]
    fn surround_encoder_ambisonics_layout_matches_reference_ordering() {
        let (_encoder, layout) =
            opus_multistream_surround_encoder_create(48_000, 6, 2, 2048).expect("encoder");
        assert_eq!(layout.streams, 5);
        assert_eq!(layout.coupled_streams, 1);
        assert_eq!(layout.mapping, vec![2, 3, 4, 5, 0, 1]);
    }

    #[test]
    fn surround_encoder_get_size_matches_reference_overhead() {
        let size = opus_multistream_surround_encoder_get_size(6, 1).expect("size");
        let base = opus_multistream_encoder_get_size(4, 2).expect("base");
        let overhead = 6 * (120 * core::mem::size_of::<crate::celt::OpusVal32>()
            + core::mem::size_of::<crate::celt::OpusVal32>());
        assert_eq!(size, base + overhead);
    }
}
