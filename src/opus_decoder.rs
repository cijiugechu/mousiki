//! Small pieces of the top-level Opus decoder API.
//!
//! Ports the size helper from `opus_decoder_get_size()` so callers can
//! determine how much memory the combined SILK/CELT decoder requires.

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::{
    canonical_mode, celt_decoder_get_size, celt_exp2, opus_custom_decoder_create,
    opus_custom_decoder_ctl, opus_select_arch, CeltDecoderCtlError,
    DecoderCtlRequest as CeltDecoderCtlRequest, OpusRes, OwnedCeltDecoder,
};
use crate::celt::{CeltCoef, OpusCustomMode};
#[cfg(not(feature = "fixed_point"))]
use crate::celt::celt_decode_with_ec_dred;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::CeltDecodeError;
use crate::celt::CELT_SIG_SCALE;
#[cfg(not(feature = "fixed_point"))]
use crate::opus::opus_pcm_soft_clip_impl;
use crate::packet::{
    Bandwidth, Mode, PacketError, ParsedPacket, opus_packet_get_bandwidth, opus_packet_get_mode,
    opus_packet_get_nb_channels, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
    opus_packet_parse_impl,
};
use crate::silk::dec_api::{
    DECODER_NUM_CHANNELS, DecControl, Decoder as SilkDecoder, reset_decoder as silk_reset_decoder,
    silk_decode,
};
use crate::silk::decoder_state::DecoderState;
use crate::silk::decode_frame::DecodeFlag;
use crate::silk::errors::SilkError;
use crate::silk::get_decoder_size::get_decoder_size;
use crate::silk::init_decoder::init_decoder as silk_init_channel;
use crate::range::RangeDecoder;

/// Maximum supported channel count for the canonical decoder.
const MAX_CHANNELS: usize = 2;
/// Scale factor that converts the quarter-dB decode gain to a base-2 exponent.
const DECODE_GAIN_SCALE: f32 = core::f32::consts::LOG2_10 / 5120.0;
/// Mode tag mirrored from `opus_private.h`.
pub(crate) const MODE_SILK_ONLY: i32 = 1000;
/// Mode tag mirrored from `opus_private.h`.
pub(crate) const MODE_HYBRID: i32 = 1001;
/// Mode tag mirrored from `opus_private.h`.
pub(crate) const MODE_CELT_ONLY: i32 = 1002;

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

#[inline]
fn opus_mode_to_int(mode: Mode) -> i32 {
    match mode {
        Mode::SILK => MODE_SILK_ONLY,
        Mode::HYBRID => MODE_HYBRID,
        Mode::CELT => MODE_CELT_ONLY,
    }
}

fn smooth_fade(
    in1: &[OpusRes],
    in2: &[OpusRes],
    out: &mut [OpusRes],
    overlap: usize,
    channels: usize,
    window: &[CeltCoef],
    fs: i32,
) {
    if channels == 0 || overlap == 0 || fs <= 0 {
        return;
    }

    let inc = match 48_000i32.checked_div(fs) {
        Some(step) if step > 0 => step as usize,
        _ => return,
    };
    let Some(required) = overlap.checked_mul(channels) else {
        return;
    };
    if in1.len() < required || in2.len() < required || out.len() < required {
        return;
    }

    for c in 0..channels {
        for i in 0..overlap {
            let w_idx = i.saturating_mul(inc);
            if w_idx >= window.len() {
                break;
            }
            let weight = window[w_idx] * window[w_idx];
            let idx = i * channels + c;
            out[idx] = weight * in2[idx] + (1.0 - weight) * in1[idx];
        }
    }
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
    /// Control block passed to the embedded SILK decoder.
    dec_control: DecControl,
    /// Decoder gain offset applied in quarter-dB steps.
    decode_gain: i32,
    /// Complexity hint mirrored from the C reference state.
    complexity: i32,
    /// Architecture selection hint propagated to CELT helpers.
    arch: i32,
    /// Number of coded channels in the current stream.
    stream_channels: i32,
    /// Decoder bandwidth advertised by the most recent packet.
    bandwidth: i32,
    /// Mode signalled by the most recent packet.
    mode: i32,
    /// Previous decode mode used for PLC decisions.
    prev_mode: i32,
    /// Frame size in samples per channel for the last decoded packet.
    frame_size: i32,
    /// Tracks whether the previous frame carried redundancy.
    prev_redundancy: i32,
    /// Duration in samples of the last decoded packet.
    last_packet_duration: i32,
    /// Soft-clipping memory when decoding to floating-point PCM.
    #[cfg(not(feature = "fixed_point"))]
    softclip_mem: [f32; 2],
    /// Final range of the last decoded packet.
    range_final: u32,
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

/// Errors that can be emitted by [`opus_decoder_ctl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusDecoderCtlError {
    /// The provided argument is outside the range accepted by the request.
    BadArgument,
    /// The requested operation is not implemented by the decoder.
    Unimplemented,
    /// The request failed inside the SILK decoder.
    Silk(SilkError),
}

/// Errors surfaced by the top-level decoder front-end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusDecodeError {
    BadArgument,
    BufferTooSmall,
    InvalidPacket,
    InternalError,
    Unimplemented,
}

impl OpusDecodeError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::InvalidPacket => -4,
            Self::Unimplemented => -5,
        }
    }
}

impl From<CeltDecoderCtlError> for OpusDecoderCtlError {
    fn from(value: CeltDecoderCtlError) -> Self {
        match value {
            CeltDecoderCtlError::InvalidArgument => Self::BadArgument,
            CeltDecoderCtlError::Unimplemented => Self::Unimplemented,
        }
    }
}

impl From<SilkError> for OpusDecoderCtlError {
    fn from(value: SilkError) -> Self {
        Self::Silk(value)
    }
}

impl From<PacketError> for OpusDecodeError {
    #[inline]
    fn from(value: PacketError) -> Self {
        match value {
            PacketError::BadArgument => Self::BadArgument,
            PacketError::InvalidPacket => Self::InvalidPacket,
        }
    }
}

/// Strongly-typed replacement for the decoder-side varargs CTL dispatcher.
pub enum OpusDecoderCtlRequest<'req> {
    SetGain(i32),
    GetGain(&'req mut i32),
    SetComplexity(i32),
    GetComplexity(&'req mut i32),
    GetBandwidth(&'req mut i32),
    GetSampleRate(&'req mut i32),
    GetPitch(&'req mut i32),
    GetFinalRange(&'req mut u32),
    ResetState,
    GetLastPacketDuration(&'req mut i32),
    SetPhaseInversionDisabled(bool),
    GetPhaseInversionDisabled(&'req mut bool),
}

/// Packet metadata extracted from the top-level decoder front-end.
///
/// Mirrors the header parsing performed by `opus_decode_native`, including the
/// optional self-delimited framing used when decoding multistream packets.
#[derive(Debug, Clone)]
pub struct ParsedPacketMetadata<'a> {
    pub mode: Mode,
    pub bandwidth: Bandwidth,
    pub frame_size: usize,
    pub stream_channels: usize,
    pub parsed: ParsedPacket<'a>,
}

impl<'mode> OpusDecoder<'mode> {
    /// Mirrors `opus_decoder_init` by preparing both the SILK and CELT decoders.
    pub fn init(&mut self, fs: i32, channels: i32) -> Result<(), OpusDecoderInitError> {
        if !matches!(fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000) || !matches!(channels, 1 | 2) {
            return Err(OpusDecoderInitError::BadArgument);
        }

        self.fs = fs;
        self.channels = channels;
        self.reset_dec_control();
        self.decode_gain = 0;
        self.complexity = 0;
        self.bandwidth = 0;
        self.mode = 0;
        self.prev_mode = 0;
        self.prev_redundancy = 0;
        self.last_packet_duration = 0;
        #[cfg(not(feature = "fixed_point"))]
        {
            self.softclip_mem = [0.0; 2];
        }
        self.range_final = 0;

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

        self.arch = opus_select_arch();
        self.reset_runtime_fields();

        Ok(())
    }

    /// Returns the number of PCM samples in the provided packet for this decoder's sample rate.
    #[inline]
    pub fn get_nb_samples(&self, packet: &[u8], len: usize) -> Result<usize, PacketError> {
        debug_assert!(matches!(self.fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000));
        opus_packet_get_nb_samples(packet, len, self.fs as u32)
    }

    /// Parses packet metadata for the decode front-end.
    ///
    /// Mirrors the header parsing performed by `opus_decode_native`, including
    /// the optional self-delimited framing used for multistream decoding.
    pub fn parse_packet<'a>(
        &self,
        packet: &'a [u8],
        len: usize,
        self_delimited: bool,
    ) -> Result<ParsedPacketMetadata<'a>, PacketError> {
        if len == 0 || len > packet.len() {
            return Err(PacketError::BadArgument);
        }

        debug_assert!(matches!(self.fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000));
        debug_assert!(matches!(self.channels, 1 | 2));

        let parsed = opus_packet_parse_impl(packet, len, self_delimited)?;
        let mode = opus_packet_get_mode(packet)?;
        let bandwidth = opus_packet_get_bandwidth(packet)?;
        let fs = u32::try_from(self.fs).map_err(|_| PacketError::BadArgument)?;
        let frame_size = opus_packet_get_samples_per_frame(packet, fs)?;
        let stream_channels = opus_packet_get_nb_channels(packet)?;

        Ok(ParsedPacketMetadata {
            mode,
            bandwidth,
            frame_size,
            stream_channels,
            parsed,
        })
    }

    /// Decodes (or conceals) a single Opus frame.
    ///
    /// Mirrors the high-level control flow from `opus_decode_frame`, wiring the
    /// SILK and CELT back-ends while handling PLC, CELT/SILK accumulation, the
    /// optional redundancy frames used for SILK↔CELT transitions, and the
    /// windowed fades that smooth decoder mode switches.
    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_frame(
        &mut self,
        data: Option<&[u8]>,
        len: usize,
        pcm: &mut [OpusRes],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize, OpusDecodeError> {
        let channels =
            usize::try_from(self.channels).map_err(|_| OpusDecodeError::BadArgument)?;
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return Err(OpusDecodeError::BadArgument);
        }

        if !matches!(self.fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000) {
            return Err(OpusDecodeError::BadArgument);
        }

        if data.is_some() && len == 0 {
            return Err(OpusDecodeError::BadArgument);
        }

        let fs = usize::try_from(self.fs).map_err(|_| OpusDecodeError::BadArgument)?;
        let f20 = fs / 50;
        let f10 = f20 / 2;
        let f5 = f10 / 2;
        let f2_5 = f5 / 2;

        if frame_size < f2_5 {
            return Err(OpusDecodeError::BufferTooSmall);
        }

        let max_frame = (fs / 25) * 3;
        let mut frame_size = frame_size.min(max_frame);

        let mut packet_len = len;
        if let Some(packet) = data && packet_len > packet.len() {
            return Err(OpusDecodeError::BadArgument);
        }
        let mut packet = data.map(|packet| &packet[..packet_len]);

        if packet.is_none() || packet_len <= 1 {
            packet = None;
            let current_size =
                usize::try_from(self.frame_size).map_err(|_| OpusDecodeError::BadArgument)?;
            frame_size = frame_size.min(current_size);
        }

        let mut audiosize;
        let mode;
        let bandwidth;
        let mut transition = false;
        let mut redundancy = false;
        let mut celt_to_silk = false;
        let mut redundant_rng = 0u32;
        let mut pcm_transition: Option<Vec<OpusRes>> = None;
        let mut redundant_audio: Option<Vec<OpusRes>> = None;
        let mut redundant_packet: Option<&[u8]> = None;

        if packet.is_some() {
            audiosize = usize::try_from(self.frame_size).map_err(|_| OpusDecodeError::BadArgument)?;
            mode = self.mode;
            bandwidth = self.bandwidth;
        } else {
            audiosize = frame_size;
            mode = if self.prev_redundancy != 0 {
                MODE_CELT_ONLY
            } else {
                self.prev_mode
            };
            bandwidth = 0;

            if mode == 0 {
                let samples = audiosize
                    .checked_mul(channels)
                    .ok_or(OpusDecodeError::BadArgument)?;
                if pcm.len() < samples {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                pcm[..samples].fill(0.0);
                self.prev_mode = 0;
                self.prev_redundancy = 0;
                self.range_final = 0;
                return Ok(audiosize);
            }

            if audiosize > f20 {
                let mut decoded = 0usize;
                while decoded < audiosize {
                    let chunk = (audiosize - decoded).min(f20);
                    let offset = decoded
                        .checked_mul(channels)
                        .ok_or(OpusDecodeError::BadArgument)?;
                    let ret = self.decode_frame(
                        None,
                        0,
                        &mut pcm[offset..],
                        chunk,
                        false,
                    )?;
                    if ret != chunk {
                        return Err(OpusDecodeError::InternalError);
                    }
                    decoded = decoded
                        .checked_add(ret)
                        .ok_or(OpusDecodeError::BadArgument)?;
                }
                self.prev_mode = mode;
                self.prev_redundancy = 0;
                self.range_final = 0;
                return Ok(audiosize);
            } else if audiosize < f20 {
                if audiosize > f10 {
                    audiosize = f10;
                } else if mode != MODE_CELT_ONLY && audiosize > f5 && audiosize < f10 {
                    audiosize = f5;
                }
            }
        }

        if packet.is_some()
            && self.prev_mode > 0
            && ((mode == MODE_CELT_ONLY
                && self.prev_mode != MODE_CELT_ONLY
                && self.prev_redundancy == 0)
                || (mode != MODE_CELT_ONLY && self.prev_mode == MODE_CELT_ONLY))
        {
            transition = true;
            if mode == MODE_CELT_ONLY {
                let transition_len = f5
                    .checked_mul(channels)
                    .ok_or(OpusDecodeError::BadArgument)?;
                let mut buffer = vec![0.0; transition_len];
                let transition_size = audiosize.min(f5);
                let ret = self.decode_frame(None, 0, &mut buffer, transition_size, false)?;
                if ret != transition_size {
                    return Err(OpusDecodeError::InternalError);
                }
                pcm_transition = Some(buffer);
            }
        }

        if audiosize > frame_size {
            return Err(OpusDecodeError::BadArgument);
        }

        let celt_accum = mode != MODE_CELT_ONLY;
        let mut range_final = 0u32;

        if mode != MODE_CELT_ONLY {
            let pcm_too_small = audiosize < f10;
            let pcm_silk_len = if pcm_too_small { f10 } else { audiosize };
            let mut silk_pcm: Option<Vec<OpusRes>> = pcm_too_small.then(|| {
                vec![0.0; pcm_silk_len.saturating_mul(channels)]
            });

            let payload_ms = audiosize
                .checked_mul(1000)
                .and_then(|value| value.checked_div(fs))
                .ok_or(OpusDecodeError::BadArgument)?
                .max(10);
            let payload_ms =
                i32::try_from(payload_ms).map_err(|_| OpusDecodeError::BadArgument)?;

            let control = &mut self.dec_control;
            control.n_channels_api = self.channels;
            control.api_sample_rate = self.fs;
            control.payload_size_ms = payload_ms;
            control.enable_deep_plc = self.complexity >= 5;

            if packet.is_some() {
                control.n_channels_internal = self.stream_channels;
                control.internal_sample_rate =
                    Bandwidth::from_opus_int(bandwidth).map_or(16_000, |packet_bandwidth| {
                        match mode {
                            MODE_SILK_ONLY => match packet_bandwidth {
                                Bandwidth::Narrow => 8_000,
                                Bandwidth::Medium => 12_000,
                                Bandwidth::Wide => 16_000,
                                _ => 16_000,
                            },
                            _ => 16_000,
                        }
                    });
            } else if control.internal_sample_rate == 0
                && let Some(fs_khz) = self
                    .silk
                    .channel_states
                    .first()
                    .map(|state| state.sample_rate.fs_khz)
                    .filter(|&fs_khz| fs_khz > 0)
            {
                control.internal_sample_rate = fs_khz * 1000;
            }

            if self.prev_mode == MODE_CELT_ONLY {
                silk_reset_decoder(&mut self.silk).map_err(|_| OpusDecodeError::InternalError)?;
            }

            let mut range_decoder = RangeDecoder::init(packet.unwrap_or(&[]));
            let mut silk_output = vec![0i16; pcm_silk_len.saturating_mul(channels)];
            let mut decoded_samples = 0usize;
            let mut write_offset = 0usize;
            while decoded_samples < audiosize {
                let new_packet = decoded_samples == 0;
                let max_chunk = audiosize
                    .checked_sub(decoded_samples)
                    .ok_or(OpusDecodeError::BadArgument)?;
                let samples_available = max_chunk
                    .checked_mul(channels)
                    .ok_or(OpusDecodeError::BadArgument)?;
                if write_offset + samples_available > silk_output.len() {
                    return Err(OpusDecodeError::BadArgument);
                }
                let result = silk_decode(
                    &mut self.silk,
                    control,
                    if packet.is_some() {
                        if decode_fec {
                            DecodeFlag::Lbrr
                        } else {
                            DecodeFlag::Normal
                        }
                    } else {
                        DecodeFlag::PacketLoss
                    },
                    new_packet,
                    &mut range_decoder,
                    &mut silk_output[write_offset..write_offset + samples_available],
                    self.arch,
                );

                let written = match result {
                    Ok(value) => value,
                    Err(_) if packet.is_none() => {
                        silk_output[write_offset..write_offset + samples_available].fill(0);
                        max_chunk
                    }
                    Err(_) => return Err(OpusDecodeError::InternalError),
                };

                if written == 0 {
                    return Err(OpusDecodeError::InternalError);
                }
                decoded_samples = decoded_samples
                    .checked_add(written)
                    .ok_or(OpusDecodeError::BadArgument)?;
                write_offset = decoded_samples
                    .checked_mul(channels)
                    .ok_or(OpusDecodeError::BadArgument)?;
            }

            let convert_scale = 1.0 / CELT_SIG_SCALE;
            let silk_samples = decoded_samples
                .checked_mul(channels)
                .ok_or(OpusDecodeError::BadArgument)?;
            if let Some(ref mut temp) = silk_pcm {
                if temp.len() < silk_samples {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                for (dst, &src) in temp.iter_mut().zip(silk_output.iter().take(silk_samples)) {
                    *dst = OpusRes::from(src) * convert_scale;
                }
            } else {
                if pcm.len() < silk_samples {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                for (dst, &src) in pcm
                    .iter_mut()
                    .take(silk_samples)
                    .zip(silk_output.iter().take(silk_samples))
                {
                    *dst = OpusRes::from(src) * convert_scale;
                }
            }

            if let Some(temp) = silk_pcm {
                let copy_len = audiosize
                    .checked_mul(channels)
                    .ok_or(OpusDecodeError::BadArgument)?;
                if temp.len() < copy_len || pcm.len() < copy_len {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                pcm[..copy_len].copy_from_slice(&temp[..copy_len]);
            }

            if !decode_fec && packet.is_some() {
                let tell = range_decoder.tell();
                let threshold = 17 + if mode == MODE_HYBRID { 20 } else { 0 };
                if tell + threshold <= (8 * packet_len) as i32 {
                    redundancy = if mode == MODE_HYBRID {
                        range_decoder.decode_symbol_logp(12) != 0
                    } else {
                        true
                    };
                    if redundancy {
                        celt_to_silk = range_decoder.decode_symbol_logp(1) != 0;
                        let bytes = if mode == MODE_HYBRID {
                            usize::try_from(range_decoder.decode_uint(256).saturating_add(2))
                                .map_err(|_| OpusDecodeError::BadArgument)?
                        } else {
                            let used_bytes = ((range_decoder.tell() + 7) >> 3) as usize;
                            packet_len
                                .checked_sub(used_bytes)
                                .ok_or(OpusDecodeError::BadArgument)?
                        };
                        if bytes > packet_len {
                            return Err(OpusDecodeError::BadArgument);
                        }
                        let cutoff = packet_len
                            .checked_sub(bytes)
                            .ok_or(OpusDecodeError::BadArgument)?;
                        if let Some(data) = packet {
                            redundant_packet = data.get(cutoff..cutoff + bytes);
                        }
                        packet_len = cutoff;
                        if packet_len
                            .checked_mul(8)
                            .is_none_or(|value| value < range_decoder.tell() as usize)
                        {
                            packet_len = 0;
                            redundancy = false;
                            redundant_packet = None;
                        } else if redundancy && redundant_packet.is_none() {
                            return Err(OpusDecodeError::BadArgument);
                        }
                    }
                }
            }

            if redundancy {
                transition = false;
            } else if transition {
                let transition_len = f5
                    .checked_mul(channels)
                    .ok_or(OpusDecodeError::BadArgument)?;
                let mut buffer = vec![0.0; transition_len];
                let transition_size = audiosize.min(f5);
                let ret = self.decode_frame(None, 0, &mut buffer, transition_size, false)?;
                if ret != transition_size {
                    return Err(OpusDecodeError::InternalError);
                }
                pcm_transition = Some(buffer);
            }

            range_final = if packet.is_some() && packet_len > 1 {
                range_decoder.range_size
            } else {
                0
            };
        }

        if let Some(data) = packet {
            if packet_len > data.len() {
                return Err(OpusDecodeError::BadArgument);
            }
            packet = Some(&data[..packet_len]);
        }

        let mut start_band = 0;
        if mode != MODE_CELT_ONLY {
            start_band = 17;
        }

        if let Some(packet) = Bandwidth::from_opus_int(bandwidth) {
            let end_band = match packet {
                Bandwidth::Narrow => 13,
                Bandwidth::Medium | Bandwidth::Wide => 17,
                Bandwidth::SuperWide => 19,
                Bandwidth::Full => 21,
            };
            opus_custom_decoder_ctl(
                self.celt.decoder(),
                CeltDecoderCtlRequest::SetEndBand(end_band),
            )
            .map_err(|_| OpusDecodeError::BadArgument)?;
        }

        opus_custom_decoder_ctl(
            self.celt.decoder(),
            CeltDecoderCtlRequest::SetChannels(channels),
        )
        .map_err(|_| OpusDecodeError::BadArgument)?;

        if redundancy && celt_to_silk {
            let redundant_len = f5
                .checked_mul(channels)
                .ok_or(OpusDecodeError::BadArgument)?;
            let mut buffer = vec![0.0; redundant_len];
            opus_custom_decoder_ctl(
                self.celt.decoder(),
                CeltDecoderCtlRequest::SetStartBand(0),
            )
            .map_err(|_| OpusDecodeError::BadArgument)?;
            if let Some(data) = redundant_packet {
                let ret = decode_celt_frame(&mut self.celt, Some(data), &mut buffer, f5, false)?;
                if ret != f5 {
                    return Err(OpusDecodeError::InternalError);
                }
                opus_custom_decoder_ctl(
                    self.celt.decoder(),
                    CeltDecoderCtlRequest::GetFinalRange(&mut redundant_rng),
                )
                .map_err(|_| OpusDecodeError::BadArgument)?;
            } else {
                return Err(OpusDecodeError::BadArgument);
            }
            redundant_audio = Some(buffer);
        }

        opus_custom_decoder_ctl(
            self.celt.decoder(),
            CeltDecoderCtlRequest::SetStartBand(start_band),
        )
        .map_err(|_| OpusDecodeError::BadArgument)?;

        if mode == MODE_SILK_ONLY {
            if !celt_accum {
                let total = audiosize
                    .checked_mul(channels)
                    .ok_or(OpusDecodeError::BadArgument)?;
                if pcm.len() < total {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                pcm[..total].fill(0.0);
            }
            if self.prev_mode == MODE_HYBRID
                && !(redundancy && celt_to_silk && self.prev_redundancy != 0)
            {
                opus_custom_decoder_ctl(
                    self.celt.decoder(),
                    CeltDecoderCtlRequest::SetStartBand(0),
                )
                .map_err(|_| OpusDecodeError::BadArgument)?;
                let silence = [0xFFu8, 0xFF];
                let ret =
                    decode_celt_frame(&mut self.celt, Some(&silence), pcm, f2_5, celt_accum)?;
                if ret != f2_5 {
                    return Err(OpusDecodeError::InternalError);
                }
            }
        } else {
            if mode != self.prev_mode && self.prev_mode > 0 && self.prev_redundancy == 0 {
                opus_custom_decoder_ctl(self.celt.decoder(), CeltDecoderCtlRequest::ResetState)
                    .map_err(|_| OpusDecodeError::BadArgument)?;
            }
            let celt_frame = audiosize.min(f20);
            let celt_packet = if decode_fec { None } else { packet };
            let celt_ret = decode_celt_frame(
                &mut self.celt,
                celt_packet,
                pcm,
                celt_frame,
                celt_accum,
            )?;

            if celt_ret != celt_frame {
                return Err(OpusDecodeError::InternalError);
            }
        }

        let mut mode_slot: Option<&OpusCustomMode<'_>> = None;
        opus_custom_decoder_ctl(
            self.celt.decoder(),
            CeltDecoderCtlRequest::GetMode(&mut mode_slot),
        )
        .map_err(|_| OpusDecodeError::BadArgument)?;
        let Some(celt_mode) = mode_slot else {
            return Err(OpusDecodeError::InternalError);
        };
        let window = celt_mode.window;
        let fade_len = f2_5
            .checked_mul(channels)
            .ok_or(OpusDecodeError::BadArgument)?;

        if redundancy && !celt_to_silk && redundant_audio.is_none() {
            let redundant_len = f5
                .checked_mul(channels)
                .ok_or(OpusDecodeError::BadArgument)?;
            let mut buffer = vec![0.0; redundant_len];
            opus_custom_decoder_ctl(self.celt.decoder(), CeltDecoderCtlRequest::ResetState)
                .map_err(|_| OpusDecodeError::BadArgument)?;
            opus_custom_decoder_ctl(
                self.celt.decoder(),
                CeltDecoderCtlRequest::SetStartBand(0),
            )
            .map_err(|_| OpusDecodeError::BadArgument)?;
            if let Some(data) = redundant_packet {
                let ret = decode_celt_frame(&mut self.celt, Some(data), &mut buffer, f5, false)?;
                if ret != f5 {
                    return Err(OpusDecodeError::InternalError);
                }
                opus_custom_decoder_ctl(
                    self.celt.decoder(),
                    CeltDecoderCtlRequest::GetFinalRange(&mut redundant_rng),
                )
                .map_err(|_| OpusDecodeError::BadArgument)?;
            } else {
                return Err(OpusDecodeError::BadArgument);
            }
            redundant_audio = Some(buffer);
        }

        if redundancy {
            if !celt_to_silk {
                if let Some(buffer) = redundant_audio.as_ref() {
                    let offset = audiosize
                        .checked_sub(f2_5)
                        .and_then(|value| value.checked_mul(channels))
                        .ok_or(OpusDecodeError::BadArgument)?;
                    let mut current = vec![0.0; fade_len];
                    current.copy_from_slice(&pcm[offset..offset + fade_len]);
                    smooth_fade(
                        &current,
                        &buffer[fade_len..],
                        &mut pcm[offset..offset + fade_len],
                        f2_5,
                        channels,
                        window,
                        self.fs,
                    );
                }
            } else if (self.prev_mode != MODE_SILK_ONLY || self.prev_redundancy != 0)
                && let Some(buffer) = redundant_audio.as_ref()
            {
                if pcm.len() < fade_len.saturating_mul(2) {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                for (dst, &src) in pcm.iter_mut().take(fade_len).zip(buffer.iter()) {
                    *dst = src;
                }
                let mut tail = vec![0.0; fade_len];
                tail.copy_from_slice(&pcm[fade_len..fade_len + fade_len]);
                smooth_fade(
                    &buffer[fade_len..],
                    &tail,
                    &mut pcm[fade_len..fade_len + fade_len],
                    f2_5,
                    channels,
                    window,
                    self.fs,
                );
            }
        } else if transition && let Some(transition_pcm) = pcm_transition.as_ref() {
            if audiosize >= f5 {
                if transition_pcm.len() < fade_len || pcm.len() < fade_len {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                pcm[..fade_len].copy_from_slice(&transition_pcm[..fade_len]);
                if pcm.len() < fade_len.saturating_mul(2) {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                let mut tail = vec![0.0; fade_len];
                tail.copy_from_slice(&pcm[fade_len..fade_len + fade_len]);
                smooth_fade(
                    &transition_pcm[fade_len..],
                    &tail,
                    &mut pcm[fade_len..fade_len + fade_len],
                    f2_5,
                    channels,
                    window,
                    self.fs,
                );
            } else {
                if pcm.len() < fade_len {
                    return Err(OpusDecodeError::BufferTooSmall);
                }
                let mut current = vec![0.0; fade_len];
                current.copy_from_slice(&pcm[..fade_len]);
                smooth_fade(
                    transition_pcm,
                    &current,
                    &mut pcm[..fade_len],
                    f2_5,
                    channels,
                    window,
                    self.fs,
                );
            }
        }

        self.prev_mode = mode;
        self.prev_redundancy = i32::from(redundancy && !celt_to_silk);
        self.range_final = range_final ^ redundant_rng;

        Ok(audiosize)
    }

    /// Ports the FEC/PLC glue from `opus_decode_native`, delegating the actual
    /// frame decode to `decode_frame`.
    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_native_with<F>(
        &mut self,
        data: Option<&[u8]>,
        len: usize,
        pcm: &mut [OpusRes],
        frame_size: usize,
        decode_fec: bool,
        self_delimited: bool,
        packet_offset: Option<&mut usize>,
        soft_clip: bool,
        decode_frame: &mut F,
    ) -> Result<usize, OpusDecodeError>
    where
        F: FnMut(
            &mut OpusDecoder<'mode>,
            Option<&[u8]>,
            usize,
            &mut [OpusRes],
            usize,
            bool,
        ) -> Result<usize, OpusDecodeError>,
    {
        if frame_size == 0 {
            return Err(OpusDecodeError::BadArgument);
        }

        let channels =
            usize::try_from(self.channels).map_err(|_| OpusDecodeError::BadArgument)?;
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(OpusDecodeError::BadArgument);
        }

        let total_samples = frame_size
            .checked_mul(channels)
            .ok_or(OpusDecodeError::BadArgument)?;
        if pcm.len() < total_samples {
            return Err(OpusDecodeError::BufferTooSmall);
        }

        let samples_per_2_5_ms =
            usize::try_from(self.fs / 400).map_err(|_| OpusDecodeError::BadArgument)?;
        if (decode_fec || len == 0 || data.is_none())
            && samples_per_2_5_ms != 0
            && !frame_size.is_multiple_of(samples_per_2_5_ms)
        {
            return Err(OpusDecodeError::BadArgument);
        }

        if len == 0 || data.is_none() {
            let mut pcm_count = 0usize;
            while pcm_count < frame_size {
                let offset = pcm_count
                    .checked_mul(channels)
                    .ok_or(OpusDecodeError::BadArgument)?;
                let remaining = frame_size
                    .checked_sub(pcm_count)
                    .ok_or(OpusDecodeError::BadArgument)?;
                let pcm_slice_end = offset
                    .checked_add(
                        remaining
                            .checked_mul(channels)
                            .ok_or(OpusDecodeError::BadArgument)?,
                    )
                    .ok_or(OpusDecodeError::BadArgument)?;
                let ret = decode_frame(
                    self,
                    None,
                    0,
                    &mut pcm[offset..pcm_slice_end],
                    remaining,
                    false,
                )?;
                if ret == 0 {
                    return Err(OpusDecodeError::InternalError);
                }
                pcm_count = pcm_count
                    .checked_add(ret)
                    .ok_or(OpusDecodeError::BadArgument)?;
            }
            debug_assert_eq!(pcm_count, frame_size);
            self.last_packet_duration =
                i32::try_from(pcm_count).map_err(|_| OpusDecodeError::BadArgument)?;
            // Apply decode gain but skip soft-clipping on PLC-only output.
            self.apply_decode_gain_and_soft_clip(pcm, pcm_count, false);
            return Ok(pcm_count);
        }

        let packet = data.unwrap_or(&[]);
        if len > packet.len() {
            return Err(OpusDecodeError::BadArgument);
        }
        let packet = &packet[..len];

        let packet_mode = opus_packet_get_mode(packet)?;
        let packet_bandwidth = opus_packet_get_bandwidth(packet)?;
        let fs = u32::try_from(self.fs).map_err(|_| OpusDecodeError::BadArgument)?;
        let packet_frame_size = opus_packet_get_samples_per_frame(packet, fs)?;
        let packet_stream_channels = opus_packet_get_nb_channels(packet)?;

        let parsed = opus_packet_parse_impl(packet, len, self_delimited)?;
        if let Some(slot) = packet_offset {
            *slot = parsed.packet_offset;
        }

        if decode_fec {
            if frame_size < packet_frame_size
                || opus_mode_to_int(packet_mode) == MODE_CELT_ONLY
                || self.mode == MODE_CELT_ONLY
            {
                return self.decode_native_with(
                    None,
                    0,
                    pcm,
                    frame_size,
                    false,
                    false,
                    None,
                    soft_clip,
                    decode_frame,
                );
            }

            let duration_copy = self.last_packet_duration;
            if frame_size != packet_frame_size {
                let leading = frame_size
                    .checked_sub(packet_frame_size)
                    .ok_or(OpusDecodeError::BadArgument)?;
                let ret = self.decode_native_with(
                    None,
                    0,
                    pcm,
                    leading,
                    false,
                    false,
                    None,
                    soft_clip,
                    decode_frame,
                );
                if let Err(err) = ret {
                    self.last_packet_duration = duration_copy;
                    return Err(err);
                }
                let ret = ret?;
                if ret != leading {
                    self.last_packet_duration = duration_copy;
                    return Err(OpusDecodeError::InternalError);
                }
            }

            self.mode = opus_mode_to_int(packet_mode);
            self.bandwidth = packet_bandwidth.to_opus_int();
            self.frame_size =
                i32::try_from(packet_frame_size).map_err(|_| OpusDecodeError::BadArgument)?;
            self.stream_channels = i32::try_from(packet_stream_channels)
                .map_err(|_| OpusDecodeError::BadArgument)?;

            let offset = (frame_size - packet_frame_size)
                .checked_mul(channels)
                .ok_or(OpusDecodeError::BadArgument)?;
            let frame_samples = packet_frame_size
                .checked_mul(channels)
                .ok_or(OpusDecodeError::BadArgument)?;
            let end = offset
                .checked_add(frame_samples)
                .ok_or(OpusDecodeError::BadArgument)?;
            let ret = decode_frame(
                self,
                Some(parsed.frames[0]),
                usize::from(parsed.frame_sizes[0]),
                &mut pcm[offset..end],
                packet_frame_size,
                true,
            )?;
            debug_assert_eq!(ret, packet_frame_size);
            self.last_packet_duration =
                i32::try_from(frame_size).map_err(|_| OpusDecodeError::BadArgument)?;
            self.apply_decode_gain_and_soft_clip(pcm, frame_size, false);
            return Ok(frame_size);
        }

        if parsed.frame_count * packet_frame_size > frame_size {
            return Err(OpusDecodeError::BufferTooSmall);
        }

        self.mode = opus_mode_to_int(packet_mode);
        self.bandwidth = packet_bandwidth.to_opus_int();
        self.frame_size =
            i32::try_from(packet_frame_size).map_err(|_| OpusDecodeError::BadArgument)?;
        self.stream_channels = i32::try_from(packet_stream_channels)
            .map_err(|_| OpusDecodeError::BadArgument)?;

        let mut nb_samples = 0usize;
        for (frame, &size_bytes) in parsed
            .frames
            .iter()
            .zip(parsed.frame_sizes.iter())
            .take(parsed.frame_count)
        {
            let offset = nb_samples
                .checked_mul(channels)
                .ok_or(OpusDecodeError::BadArgument)?;
            let remaining = frame_size
                .checked_sub(nb_samples)
                .ok_or(OpusDecodeError::BadArgument)?;
            let pcm_samples = remaining
                .checked_mul(channels)
                .ok_or(OpusDecodeError::BadArgument)?;
            let end = offset
                .checked_add(pcm_samples)
                .ok_or(OpusDecodeError::BadArgument)?;
            let ret = decode_frame(
                self,
                Some(*frame),
                usize::from(size_bytes),
                &mut pcm[offset..end],
                remaining,
                false,
            )?;
            debug_assert_eq!(ret, packet_frame_size);
            nb_samples = nb_samples
                .checked_add(ret)
                .ok_or(OpusDecodeError::BadArgument)?;
        }

        self.last_packet_duration =
            i32::try_from(nb_samples).map_err(|_| OpusDecodeError::BadArgument)?;
        self.apply_decode_gain_and_soft_clip(pcm, nb_samples, soft_clip);

        Ok(nb_samples)
    }

    /// Applies the decoder gain and optional soft clipping to the decoded PCM.
    ///
    /// Mirrors the tail of `opus_decode_native`, scaling the interleaved `pcm`
    /// samples by the quarter-dB decode gain before running the optional
    /// floating-point soft clipper. The clipping state is reset when clipping
    /// is disabled to match the reference behaviour.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn apply_decode_gain_and_soft_clip(
        &mut self,
        pcm: &mut [OpusRes],
        frame_size: usize,
        soft_clip: bool,
    ) {
        let channels = usize::try_from(self.channels).unwrap_or_default();
        debug_assert!(matches!(channels, 1 | 2));
        let Some(total_samples) = frame_size.checked_mul(channels) else {
            return;
        };
        debug_assert!(pcm.len() >= total_samples);
        if pcm.len() < total_samples {
            return;
        }
        let pcm = &mut pcm[..total_samples];

        if self.decode_gain != 0 {
            let gain = celt_exp2(DECODE_GAIN_SCALE * self.decode_gain as f32);
            for sample in pcm.iter_mut() {
                *sample *= gain;
            }
        }

        #[cfg(not(feature = "fixed_point"))]
        {
            if soft_clip {
                opus_pcm_soft_clip_impl(
                    pcm,
                    frame_size,
                    channels,
                    &mut self.softclip_mem,
                    self.arch,
                );
            } else {
                self.softclip_mem = [0.0; 2];
            }
        }
        #[cfg(feature = "fixed_point")]
        {
            let _ = soft_clip;
        }
    }

    /// Resets the SILK control block to the defaults applied by `opus_decoder_init`.
    fn reset_dec_control(&mut self) {
        self.dec_control = DecControl {
            n_channels_api: self.channels,
            n_channels_internal: 0,
            api_sample_rate: self.fs,
            internal_sample_rate: 0,
            payload_size_ms: 0,
            prev_pitch_lag: 0,
            enable_deep_plc: false,
        };
    }

    /// Clears runtime decoder fields that are reset by both `opus_decoder_init` and `OPUS_RESET_STATE`.
    fn reset_runtime_fields(&mut self) {
        self.stream_channels = self.channels;
        self.bandwidth = 0;
        self.mode = 0;
        self.prev_mode = 0;
        self.frame_size = self.fs / 400;
        self.prev_redundancy = 0;
        self.last_packet_duration = 0;
        #[cfg(not(feature = "fixed_point"))]
        {
            self.softclip_mem = [0.0; 2];
        }
        self.range_final = 0;
    }

    /// Mirrors `OPUS_RESET_STATE` by clearing runtime fields and resetting the component decoders.
    fn reset_state(&mut self) -> Result<(), OpusDecoderCtlError> {
        opus_custom_decoder_ctl(self.celt.decoder(), CeltDecoderCtlRequest::ResetState)?;
        silk_reset_decoder(&mut self.silk)?;
        self.reset_runtime_fields();
        Ok(())
    }
}

#[cfg(not(feature = "fixed_point"))]
fn map_celt_error(err: CeltDecodeError) -> OpusDecodeError {
    match err {
        CeltDecodeError::BadArgument => OpusDecodeError::BadArgument,
        CeltDecodeError::InvalidPacket | CeltDecodeError::PacketLoss => {
            OpusDecodeError::InvalidPacket
        }
    }
}

#[cfg(not(feature = "fixed_point"))]
fn decode_celt_frame(
    decoder: &mut OwnedCeltDecoder<'_>,
    packet: Option<&[u8]>,
    pcm: &mut [OpusRes],
    frame_size: usize,
    accum: bool,
) -> Result<usize, OpusDecodeError> {
    celt_decode_with_ec_dred(decoder.decoder(), packet, pcm, frame_size, None, accum)
        .map_err(map_celt_error)
}

#[cfg(feature = "fixed_point")]
fn decode_celt_frame(
    decoder: &mut OwnedCeltDecoder<'_>,
    packet: Option<&[u8]>,
    pcm: &mut [OpusRes],
    frame_size: usize,
    accum: bool,
) -> Result<usize, OpusDecodeError> {
    let _ = (decoder, packet, pcm, frame_size, accum);
    Err(OpusDecodeError::Unimplemented)
}

/// Mirrors `opus_decoder_create` by allocating and initialising a decoder.
pub fn opus_decoder_create(
    fs: i32,
    channels: i32,
) -> Result<OpusDecoder<'static>, OpusDecoderInitError> {
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
        dec_control: DecControl::default(),
        decode_gain: 0,
        complexity: 0,
        arch: 0,
        stream_channels: 0,
        bandwidth: 0,
        mode: 0,
        prev_mode: 0,
        frame_size: 0,
        prev_redundancy: 0,
        last_packet_duration: 0,
        #[cfg(not(feature = "fixed_point"))]
        softclip_mem: [0.0; 2],
        range_final: 0,
    };

    decoder.init(fs, channels)?;

    Ok(decoder)
}

/// Mirrors `opus_decoder_get_nb_samples` by delegating to the packet helper with the decoder's Fs.
#[inline]
pub fn opus_decoder_get_nb_samples(
    decoder: &OpusDecoder<'_>,
    packet: &[u8],
    len: usize,
) -> Result<usize, PacketError> {
    decoder.get_nb_samples(packet, len)
}

/// Mirrors `opus_decode_native` while delegating frame decode to the embedded closure.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub fn opus_decode_native(
    decoder: &mut OpusDecoder<'_>,
    data: Option<&[u8]>,
    len: usize,
    pcm: &mut [OpusRes],
    frame_size: usize,
    decode_fec: bool,
    self_delimited: bool,
    packet_offset: Option<&mut usize>,
    soft_clip: bool,
) -> Result<usize, OpusDecodeError> {
    decoder.decode_native_with(
        data,
        len,
        pcm,
        frame_size,
        decode_fec,
        self_delimited,
        packet_offset,
        soft_clip,
        &mut |st, frame_data, frame_len, out, size, fec| {
            st.decode_frame(frame_data, frame_len, out, size, fec)
        },
    )
}

/// Applies a control request to the provided decoder state.
pub fn opus_decoder_ctl<'req>(
    decoder: &mut OpusDecoder<'_>,
    request: OpusDecoderCtlRequest<'req>,
) -> Result<(), OpusDecoderCtlError> {
    match request {
        OpusDecoderCtlRequest::GetBandwidth(slot) => {
            *slot = decoder.bandwidth;
        }
        OpusDecoderCtlRequest::GetSampleRate(slot) => {
            *slot = decoder.fs;
        }
        OpusDecoderCtlRequest::GetPitch(slot) => {
            if decoder.prev_mode == MODE_CELT_ONLY {
                opus_custom_decoder_ctl(
                    decoder.celt.decoder(),
                    CeltDecoderCtlRequest::GetPitch(slot),
                )?;
            } else {
                *slot = decoder.dec_control.prev_pitch_lag;
            }
        }
        OpusDecoderCtlRequest::SetGain(value) => {
            if !(-32_768..=32_767).contains(&value) {
                return Err(OpusDecoderCtlError::BadArgument);
            }
            decoder.decode_gain = value;
        }
        OpusDecoderCtlRequest::GetGain(slot) => {
            *slot = decoder.decode_gain;
        }
        OpusDecoderCtlRequest::SetComplexity(value) => {
            if !(0..=10).contains(&value) {
                return Err(OpusDecoderCtlError::BadArgument);
            }
            opus_custom_decoder_ctl(
                decoder.celt.decoder(),
                CeltDecoderCtlRequest::SetComplexity(value),
            )?;
            decoder.complexity = value;
        }
        OpusDecoderCtlRequest::GetComplexity(slot) => {
            *slot = decoder.complexity;
        }
        OpusDecoderCtlRequest::ResetState => decoder.reset_state()?,
        OpusDecoderCtlRequest::GetLastPacketDuration(slot) => {
            *slot = decoder.last_packet_duration;
        }
        OpusDecoderCtlRequest::GetFinalRange(slot) => {
            *slot = decoder.range_final;
        }
        OpusDecoderCtlRequest::SetPhaseInversionDisabled(value) => {
            opus_custom_decoder_ctl(
                decoder.celt.decoder(),
                CeltDecoderCtlRequest::SetPhaseInversionDisabled(value),
            )?;
        }
        OpusDecoderCtlRequest::GetPhaseInversionDisabled(slot) => {
            opus_custom_decoder_ctl(
                decoder.celt.decoder(),
                CeltDecoderCtlRequest::GetPhaseInversionDisabled(slot),
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DecControl, OpusDecodeError, OpusDecoderCtlError, OpusDecoderCtlRequest, MODE_CELT_ONLY,
        MODE_SILK_ONLY, opus_decode_native, opus_decoder_create, opus_decoder_ctl, smooth_fade,
        opus_decoder_get_size,
    };
    #[cfg(not(feature = "fixed_point"))]
    use super::MODE_HYBRID;
    use alloc::vec;
    use alloc::vec::Vec;
    use crate::celt::{
        OpusRes, canonical_mode, celt_decoder_get_size, celt_exp2, opus_custom_decoder_create,
    };
    use crate::packet::{Bandwidth, Mode, PacketError};
    use crate::silk::dec_api::Decoder as SilkDecoder;
    use crate::silk::get_decoder_size::get_decoder_size;

    fn simple_packet(toc: u8, payload_len: usize) -> Vec<u8> {
        let mut packet = Vec::with_capacity(payload_len + 1);
        packet.push(toc);
        packet.extend(core::iter::repeat(0u8).take(payload_len));
        packet
    }

    #[test]
    fn smooth_fade_blends_with_squared_window() {
        let in1 = [1.0f32, -1.0, 0.5, -0.5];
        let in2 = [0.0f32, 0.5, 1.0, -1.0];
        let mut out = [0.0f32; 4];
        let window = [0.0f32, 0.5, 1.0, 1.0];

        smooth_fade(&in1, &in2, &mut out, 2, 2, &window, 48_000);

        assert!((out[0] - 1.0).abs() < 1e-7);
        assert!((out[1] + 1.0).abs() < 1e-7);
        assert!((out[2] - 0.625).abs() < 1e-7);
        assert!((out[3] + 0.625).abs() < 1e-7);
    }

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
            dec_control: DecControl::default(),
            decode_gain: 0,
            complexity: 0,
            arch: 0,
            stream_channels: 0,
            bandwidth: 0,
            mode: 0,
            prev_mode: 0,
            frame_size: 0,
            prev_redundancy: 0,
            last_packet_duration: 0,
            #[cfg(not(feature = "fixed_point"))]
            softclip_mem: [0.0; 2],
            range_final: 0,
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

    #[test]
    fn decoder_gain_round_trips_and_validates_range() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");

        opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::SetGain(-15)).unwrap();
        let mut gain = 0;
        opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::GetGain(&mut gain)).unwrap();
        assert_eq!(gain, -15);

        let err =
            opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::SetGain(40_000)).unwrap_err();
        assert_eq!(err, OpusDecoderCtlError::BadArgument);

        opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::GetGain(&mut gain)).unwrap();
        assert_eq!(gain, -15);
    }

    #[test]
    fn complexity_ctl_round_trips_and_validates_range() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");

        opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::SetComplexity(7)).unwrap();

        let mut complexity = 0;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetComplexity(&mut complexity),
        )
        .unwrap();
        assert_eq!(complexity, 7);

        let err =
            opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::SetComplexity(11)).unwrap_err();
        assert_eq!(err, OpusDecoderCtlError::BadArgument);

        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetComplexity(&mut complexity),
        )
        .unwrap();
        assert_eq!(complexity, 7);
    }

    #[test]
    fn reset_state_preserves_gain_and_resets_runtime_fields() {
        let mut decoder = opus_decoder_create(48_000, 2).expect("decoder should initialise");

        decoder.decode_gain = 123;
        decoder.stream_channels = 1;
        decoder.prev_mode = 1;
        decoder.prev_redundancy = 1;
        decoder.last_packet_duration = 960;
        decoder.range_final = 42;
        decoder.silk.prev_decode_only_middle = true;
        #[cfg(not(feature = "fixed_point"))]
        {
            decoder.softclip_mem = [0.5, -0.25];
        }

        opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::ResetState).unwrap();

        assert_eq!(decoder.decode_gain, 123);
        assert_eq!(decoder.stream_channels, decoder.channels);
        assert_eq!(decoder.prev_mode, 0);
        assert_eq!(decoder.prev_redundancy, 0);
        assert_eq!(decoder.last_packet_duration, 0);
        assert_eq!(decoder.range_final, 0);
        assert_eq!(decoder.frame_size, decoder.fs / 400);
        assert!(!decoder.silk.prev_decode_only_middle);
        #[cfg(not(feature = "fixed_point"))]
        {
            assert_eq!(decoder.softclip_mem, [0.0, 0.0]);
        }
    }

    #[test]
    #[cfg(not(feature = "fixed_point"))]
    fn apply_decode_gain_scales_pcm_and_resets_softclip_mem_when_disabled() {
        let mut decoder = opus_decoder_create(48_000, 2).expect("decoder should initialise");
        decoder.decode_gain = 256; // +1 dB.
        decoder.softclip_mem = [0.5, -0.25];

        let mut pcm: [OpusRes; 4] = [0.5, -0.25, -0.75, 0.1];
        decoder.apply_decode_gain_and_soft_clip(&mut pcm, 2, false);

        let gain = celt_exp2(super::DECODE_GAIN_SCALE * 256.0);
        assert!((pcm[0] - 0.5 * gain).abs() < 1e-6);
        assert!((pcm[1] + 0.25 * gain).abs() < 1e-6);
        assert!((pcm[2] + 0.75 * gain).abs() < 1e-6);
        assert!((pcm[3] - 0.1 * gain).abs() < 1e-6);
        assert_eq!(decoder.softclip_mem, [0.0, 0.0]);
    }

    #[test]
    #[cfg(not(feature = "fixed_point"))]
    fn apply_decode_gain_invokes_soft_clip_before_returning() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        decoder.decode_gain = 256; // +1 dB.

        let mut pcm: [OpusRes; 1] = [0.9];
        decoder.apply_decode_gain_and_soft_clip(&mut pcm, 1, true);

        let gain = celt_exp2(super::DECODE_GAIN_SCALE * 256.0);
        let mut expected = [0.9 * gain];
        let mut softclip_mem = [0.0; 2];
        crate::opus::opus_pcm_soft_clip_impl(&mut expected, 1, 1, &mut softclip_mem, decoder.arch);

        assert!((pcm[0] - expected[0]).abs() < 1e-6);
        assert!((decoder.softclip_mem[0] - softclip_mem[0]).abs() < 1e-6);
        assert_eq!(decoder.softclip_mem[1], 0.0);
    }

    #[test]
    #[cfg(feature = "fixed_point")]
    fn apply_decode_gain_scales_pcm_in_fixed_point_builds() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        decoder.decode_gain = 128; // +0.5 dB.

        let mut pcm: [OpusRes; 1] = [0.5];
        decoder.apply_decode_gain_and_soft_clip(&mut pcm, 1, false);

        let gain = celt_exp2(super::DECODE_GAIN_SCALE * 128.0);
        assert!((pcm[0] - 0.5 * gain).abs() < 1e-6);
    }

    #[test]
    fn exposes_sample_rate_bandwidth_and_final_range() {
        let mut decoder = opus_decoder_create(48_000, 2).expect("decoder should initialise");
        decoder.bandwidth = 1105;
        decoder.range_final = 42;

        let mut fs = 0;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetSampleRate(&mut fs),
        )
        .unwrap();
        assert_eq!(fs, 48_000);

        let mut bandwidth = 0;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetBandwidth(&mut bandwidth),
        )
        .unwrap();
        assert_eq!(bandwidth, 1105);

        let mut final_range = 0;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut final_range),
        )
        .unwrap();
        assert_eq!(final_range, 42);
    }

    #[test]
    fn reports_last_packet_duration() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        decoder.last_packet_duration = 960;

        let mut duration = 0;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetLastPacketDuration(&mut duration),
        )
        .unwrap();
        assert_eq!(duration, 960);

        opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::ResetState).unwrap();
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetLastPacketDuration(&mut duration),
        )
        .unwrap();
        assert_eq!(duration, 0);
    }

    #[test]
    fn phase_inversion_ctl_forwards_to_celt() {
        let mut decoder = opus_decoder_create(48_000, 2).expect("decoder should initialise");

        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::SetPhaseInversionDisabled(true),
        )
        .unwrap();

        let mut disabled = false;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetPhaseInversionDisabled(&mut disabled),
        )
        .unwrap();
        assert!(disabled);
    }

    #[test]
    fn get_pitch_reports_silk_prev_pitch_lag() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        decoder.prev_mode = MODE_SILK_ONLY;
        decoder.dec_control.prev_pitch_lag = 123;

        let mut pitch = 0;
        opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::GetPitch(&mut pitch)).unwrap();
        assert_eq!(pitch, 123);
    }

    #[test]
    fn get_pitch_forwards_to_celt_when_prev_mode_is_celt() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        decoder.prev_mode = MODE_CELT_ONLY;
        {
            let celt = decoder.celt.decoder();
            celt.postfilter_period = 321;
        }

        let mut pitch = 0;
        opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::GetPitch(&mut pitch)).unwrap();
        assert_eq!(pitch, 321);
    }

    #[test]
    fn parse_packet_reports_self_delimited_metadata() {
        let decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        let packet: [u8; 7] = [0x00, 0x05, 1, 2, 3, 4, 5];

        let parsed = decoder
            .parse_packet(&packet, packet.len(), true)
            .expect("parse succeeds");

        assert_eq!(parsed.mode, Mode::SILK);
        assert_eq!(parsed.bandwidth, Bandwidth::Narrow);
        assert_eq!(parsed.frame_size, 480);
        assert_eq!(parsed.stream_channels, 1);
        assert_eq!(parsed.parsed.frame_count, 1);
        assert_eq!(parsed.parsed.frame_sizes[0], 5);
        assert_eq!(parsed.parsed.payload_offset, 2);
        assert_eq!(parsed.parsed.packet_offset, packet.len());
        assert!(parsed.parsed.padding.is_empty());
    }

    #[test]
    fn parse_packet_validates_length() {
        let decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        let packet: [u8; 2] = [0x00, 0x00];

        let err = decoder
            .parse_packet(&packet, packet.len() + 1, true)
            .unwrap_err();
        assert_eq!(err, PacketError::BadArgument);
    }

    #[test]
    fn decode_native_rejects_misaligned_plc_frame_size() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        let mut pcm = [0.0; 240];

        let err = decoder
            .decode_native_with(
                None,
                0,
                &mut pcm,
                100,
                false,
                false,
                None,
                false,
                &mut |_st, _data, _len, _pcm, _frame_size, _decode_fec| {
                    unreachable!("decode_frame should not be called on invalid input")
                },
            )
            .unwrap_err();

        assert_eq!(err, OpusDecodeError::BadArgument);
    }

    #[test]
    fn decode_frame_rejects_frame_sizes_shorter_than_2_5_ms() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        let mut pcm = [0.0; 10];

        let err = decoder
            .decode_frame(None, 0, &mut pcm, 60, false)
            .unwrap_err();

        assert_eq!(err, OpusDecodeError::BufferTooSmall);
    }

    #[test]
    #[cfg(not(feature = "fixed_point"))]
    fn hybrid_plc_frames_decode_without_unimplemented() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        decoder.prev_mode = MODE_HYBRID;
        decoder.frame_size = 960;
        let mut pcm = vec![0.0; 960];

        let decoded = decoder
            .decode_frame(None, 0, &mut pcm, 960, false)
            .expect("hybrid PLC decode should succeed");

        assert_eq!(decoded, 960);
        assert_eq!(decoder.prev_mode, MODE_HYBRID);
        assert_eq!(decoder.prev_redundancy, 0);
    }

    #[test]
    fn decode_native_runs_plc_via_frame_decoder() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        let mut pcm = [0.0; 480];

        let decoded =
            opus_decode_native(&mut decoder, None, 0, &mut pcm, 480, false, false, None, false)
                .expect("PLC decode should succeed");

        assert_eq!(decoded, 480);
        assert_eq!(decoder.last_packet_duration, 480);
        assert!(pcm.iter().all(|&sample| sample == 0.0));
    }

    #[test]
    fn decode_native_runs_plc_path_when_packet_missing() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        let mut pcm = [0.0; 480];
        let mut calls = 0usize;

        let decoded = decoder
            .decode_native_with(
                None,
                0,
                &mut pcm,
                480,
                false,
                false,
                None,
                false,
                &mut |st, data, len, out, requested, decode_fec| {
                    assert!(data.is_none());
                    assert_eq!(len, 0);
                    assert_eq!(requested, 480);
                    assert!(!decode_fec);
                    calls += 1;

                    let channels = st.channels as usize;
                    for sample in out.iter_mut().take(requested * channels) {
                        *sample = 1.0;
                    }

                    Ok(requested)
                },
            )
            .expect("PLC decode should succeed");

        assert_eq!(decoded, 480);
        assert_eq!(calls, 1);
        assert_eq!(decoder.last_packet_duration, 480);
        assert!(pcm[..480].iter().all(|&sample| (sample - 1.0).abs() < 1e-6));
    }

    #[test]
    fn decode_native_handles_partial_fec_and_updates_state() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        let packet = simple_packet(0x00, 2);
        let mut pcm = [0.0; 960];
        let mut calls = Vec::new();

        let decoded = decoder
            .decode_native_with(
                Some(&packet),
                packet.len(),
                &mut pcm,
                960,
                true,
                false,
                None,
                false,
                &mut |st, data, len, out, requested, decode_fec| {
                    calls.push((data.is_some(), len, requested, decode_fec));
                    let channels = st.channels as usize;
                    for sample in out.iter_mut().take(requested * channels) {
                        *sample = if data.is_none() { -1.0 } else { 2.0 };
                    }
                    Ok(requested)
                },
            )
            .expect("FEC decode should succeed");

        assert_eq!(decoded, 960);
        assert_eq!(decoder.mode, MODE_SILK_ONLY);
        assert_eq!(decoder.bandwidth, Bandwidth::Narrow.to_opus_int());
        assert_eq!(decoder.frame_size, 480);
        assert_eq!(decoder.stream_channels, 1);
        assert!(pcm[..480].iter().all(|&sample| (sample + 1.0).abs() < 1e-6));
        assert!(pcm[480..960].iter().all(|&sample| (sample - 2.0).abs() < 1e-6));
        assert_eq!(
            calls,
            vec![
                (false, 0, 480, false),
                (true, packet.len() - 1, 480, true)
            ]
        );
    }

    #[test]
    fn decode_native_fec_falls_back_to_plc_when_celt_only() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        decoder.mode = MODE_CELT_ONLY;
        let packet = simple_packet(0x80, 2);
        let mut pcm = [0.0; 240];
        let mut calls = 0usize;

        let decoded = decoder
            .decode_native_with(
                Some(&packet),
                packet.len(),
                &mut pcm,
                240,
                true,
                false,
                None,
                false,
                &mut |st, data, _len, out, requested, decode_fec| {
                    assert!(data.is_none());
                    assert!(!decode_fec);
                    calls += 1;

                    let channels = st.channels as usize;
                    for sample in out.iter_mut().take(requested * channels) {
                        *sample = 3.0;
                    }

                    Ok(requested)
                },
            )
            .expect("PLC fallback should succeed");

        assert_eq!(decoded, 240);
        assert_eq!(calls, 1);
        assert_eq!(decoder.mode, MODE_CELT_ONLY);
        assert!(pcm[..240].iter().all(|&sample| (sample - 3.0).abs() < 1e-6));
    }

    #[test]
    fn decode_native_restores_last_duration_when_plc_fails_during_fec() {
        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder should initialise");
        decoder.last_packet_duration = 320;
        let packet = simple_packet(0x00, 2);
        let mut pcm = [0.0; 960];

        let err = decoder
            .decode_native_with(
                Some(&packet),
                packet.len(),
                &mut pcm,
                960,
                true,
                false,
                None,
                false,
                &mut |_st, data, _len, _out, _requested, _decode_fec| {
                    if data.is_none() {
                        Err(OpusDecodeError::InvalidPacket)
                    } else {
                        Ok(0)
                    }
                },
            )
            .unwrap_err();

        assert_eq!(err, OpusDecodeError::InvalidPacket);
        assert_eq!(decoder.last_packet_duration, 320);
    }
}
