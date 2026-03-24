//! Rust-idiomatic wrappers around the top-level Opus encoder/decoder.
//!
//! The lower-level `opus_encoder`/`opus_decoder` modules stay available for
//! callers that want a closer match to the C API. This module adds method-based
//! wrappers, typed configuration enums, and frame-size inference from PCM slice
//! lengths.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "fixed_point"))]
use crate::opus_decoder::opus_decode_float;
use crate::opus_decoder::{
    OpusDecodeError, OpusDecoder, OpusDecoderCtlError, OpusDecoderCtlRequest, OpusDecoderInitError,
    opus_decode, opus_decode24, opus_decoder_create, opus_decoder_ctl, opus_decoder_get_nb_samples,
};
use crate::opus_encoder::{
    OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_FRAMESIZE_2_5_MS,
    OPUS_FRAMESIZE_5_MS, OPUS_FRAMESIZE_10_MS, OPUS_FRAMESIZE_20_MS, OPUS_FRAMESIZE_40_MS,
    OPUS_FRAMESIZE_60_MS, OPUS_FRAMESIZE_80_MS, OPUS_FRAMESIZE_100_MS, OPUS_FRAMESIZE_120_MS,
    OPUS_FRAMESIZE_ARG, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE, OpusEncodeError, OpusEncoder,
    OpusEncoderCtlError, OpusEncoderCtlRequest, OpusEncoderInitError, opus_encode,
    opus_encode_float, opus_encode24, opus_encoder_create, opus_encoder_ctl,
};
use crate::opus_multistream::{OPUS_AUTO, OPUS_BITRATE_MAX};
use crate::packet::PacketError;

/// Intended encoder tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Application {
    Voip,
    Audio,
    LowDelay,
}

impl Application {
    #[inline]
    const fn to_opus_int(self) -> i32 {
        match self {
            Self::Voip => 2048,
            Self::Audio => 2049,
            Self::LowDelay => 2051,
        }
    }
}

/// Supported channel layouts for the canonical top-level Opus API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Mono,
    Stereo,
}

impl Channels {
    #[inline]
    pub const fn count(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }

    #[inline]
    const fn to_opus_int(self) -> i32 {
        self.count() as i32
    }
}

/// Encoder bitrate selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitrate {
    Auto,
    Max,
    Bits(i32),
}

impl Bitrate {
    #[inline]
    const fn to_opus_int(self) -> i32 {
        match self {
            Self::Auto => OPUS_AUTO,
            Self::Max => OPUS_BITRATE_MAX,
            Self::Bits(value) => value,
        }
    }

    #[inline]
    const fn from_opus_int(value: i32) -> Self {
        match value {
            OPUS_AUTO => Self::Auto,
            OPUS_BITRATE_MAX => Self::Max,
            _ => Self::Bits(value),
        }
    }
}

/// Encoder/decoder bandwidth selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bandwidth {
    Narrowband,
    Mediumband,
    Wideband,
    Superwideband,
    Fullband,
}

impl Bandwidth {
    #[inline]
    const fn to_opus_int(self) -> i32 {
        match self {
            Self::Narrowband => OPUS_BANDWIDTH_NARROWBAND,
            Self::Mediumband => OPUS_BANDWIDTH_MEDIUMBAND,
            Self::Wideband => OPUS_BANDWIDTH_WIDEBAND,
            Self::Superwideband => OPUS_BANDWIDTH_SUPERWIDEBAND,
            Self::Fullband => OPUS_BANDWIDTH_FULLBAND,
        }
    }

    #[inline]
    const fn from_opus_int(value: i32) -> Option<Self> {
        match value {
            OPUS_BANDWIDTH_NARROWBAND => Some(Self::Narrowband),
            OPUS_BANDWIDTH_MEDIUMBAND => Some(Self::Mediumband),
            OPUS_BANDWIDTH_WIDEBAND => Some(Self::Wideband),
            OPUS_BANDWIDTH_SUPERWIDEBAND => Some(Self::Superwideband),
            OPUS_BANDWIDTH_FULLBAND => Some(Self::Fullband),
            _ => None,
        }
    }
}

/// Signal hint for encoder mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Auto,
    Voice,
    Music,
}

impl Signal {
    #[inline]
    const fn to_opus_int(self) -> i32 {
        match self {
            Self::Auto => OPUS_AUTO,
            Self::Voice => OPUS_SIGNAL_VOICE,
            Self::Music => OPUS_SIGNAL_MUSIC,
        }
    }

    #[inline]
    const fn from_opus_int(value: i32) -> Option<Self> {
        match value {
            OPUS_AUTO => Some(Self::Auto),
            OPUS_SIGNAL_VOICE => Some(Self::Voice),
            OPUS_SIGNAL_MUSIC => Some(Self::Music),
            _ => None,
        }
    }
}

/// Expert frame-duration override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDuration {
    Auto,
    Ms2_5,
    Ms5,
    Ms10,
    Ms20,
    Ms40,
    Ms60,
    Ms80,
    Ms100,
    Ms120,
}

impl FrameDuration {
    #[inline]
    const fn to_opus_int(self) -> i32 {
        match self {
            Self::Auto => OPUS_FRAMESIZE_ARG,
            Self::Ms2_5 => OPUS_FRAMESIZE_2_5_MS,
            Self::Ms5 => OPUS_FRAMESIZE_5_MS,
            Self::Ms10 => OPUS_FRAMESIZE_10_MS,
            Self::Ms20 => OPUS_FRAMESIZE_20_MS,
            Self::Ms40 => OPUS_FRAMESIZE_40_MS,
            Self::Ms60 => OPUS_FRAMESIZE_60_MS,
            Self::Ms80 => OPUS_FRAMESIZE_80_MS,
            Self::Ms100 => OPUS_FRAMESIZE_100_MS,
            Self::Ms120 => OPUS_FRAMESIZE_120_MS,
        }
    }

    #[inline]
    const fn from_opus_int(value: i32) -> Option<Self> {
        match value {
            OPUS_FRAMESIZE_ARG => Some(Self::Auto),
            OPUS_FRAMESIZE_2_5_MS => Some(Self::Ms2_5),
            OPUS_FRAMESIZE_5_MS => Some(Self::Ms5),
            OPUS_FRAMESIZE_10_MS => Some(Self::Ms10),
            OPUS_FRAMESIZE_20_MS => Some(Self::Ms20),
            OPUS_FRAMESIZE_40_MS => Some(Self::Ms40),
            OPUS_FRAMESIZE_60_MS => Some(Self::Ms60),
            OPUS_FRAMESIZE_80_MS => Some(Self::Ms80),
            OPUS_FRAMESIZE_100_MS => Some(Self::Ms100),
            OPUS_FRAMESIZE_120_MS => Some(Self::Ms120),
            _ => None,
        }
    }
}

/// Errors returned when building an [`Encoder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderBuilderError {
    Init(OpusEncoderInitError),
    Ctl(OpusEncoderCtlError),
}

impl From<OpusEncoderInitError> for EncoderBuilderError {
    #[inline]
    fn from(value: OpusEncoderInitError) -> Self {
        Self::Init(value)
    }
}

impl From<OpusEncoderCtlError> for EncoderBuilderError {
    #[inline]
    fn from(value: OpusEncoderCtlError) -> Self {
        Self::Ctl(value)
    }
}

/// Errors returned when building a [`Decoder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderBuilderError {
    Init(OpusDecoderInitError),
    Ctl(OpusDecoderCtlError),
}

impl From<OpusDecoderInitError> for DecoderBuilderError {
    #[inline]
    fn from(value: OpusDecoderInitError) -> Self {
        Self::Init(value)
    }
}

impl From<OpusDecoderCtlError> for DecoderBuilderError {
    #[inline]
    fn from(value: OpusDecoderCtlError) -> Self {
        Self::Ctl(value)
    }
}

/// Builder for the high-level [`Encoder`] wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderBuilder {
    sample_rate: u32,
    channels: Channels,
    application: Application,
    bitrate: Option<Bitrate>,
    complexity: Option<i32>,
    vbr: Option<bool>,
    vbr_constraint: Option<bool>,
    max_bandwidth: Option<Bandwidth>,
    signal: Option<Signal>,
    inband_fec: Option<bool>,
    packet_loss_perc: Option<i32>,
    dtx: Option<bool>,
    lsb_depth: Option<i32>,
    frame_duration: Option<FrameDuration>,
    prediction_disabled: Option<bool>,
}

impl EncoderBuilder {
    #[inline]
    pub const fn new(sample_rate: u32, channels: Channels, application: Application) -> Self {
        Self {
            sample_rate,
            channels,
            application,
            bitrate: None,
            complexity: None,
            vbr: None,
            vbr_constraint: None,
            max_bandwidth: None,
            signal: None,
            inband_fec: None,
            packet_loss_perc: None,
            dtx: None,
            lsb_depth: None,
            frame_duration: None,
            prediction_disabled: None,
        }
    }

    #[inline]
    pub const fn bitrate(mut self, value: Bitrate) -> Self {
        self.bitrate = Some(value);
        self
    }

    #[inline]
    pub const fn complexity(mut self, value: i32) -> Self {
        self.complexity = Some(value);
        self
    }

    #[inline]
    pub const fn vbr(mut self, value: bool) -> Self {
        self.vbr = Some(value);
        self
    }

    #[inline]
    pub const fn vbr_constraint(mut self, value: bool) -> Self {
        self.vbr_constraint = Some(value);
        self
    }

    #[inline]
    pub const fn max_bandwidth(mut self, value: Bandwidth) -> Self {
        self.max_bandwidth = Some(value);
        self
    }

    #[inline]
    pub const fn signal(mut self, value: Signal) -> Self {
        self.signal = Some(value);
        self
    }

    #[inline]
    pub const fn inband_fec(mut self, value: bool) -> Self {
        self.inband_fec = Some(value);
        self
    }

    #[inline]
    pub const fn packet_loss_perc(mut self, value: i32) -> Self {
        self.packet_loss_perc = Some(value);
        self
    }

    #[inline]
    pub const fn dtx(mut self, value: bool) -> Self {
        self.dtx = Some(value);
        self
    }

    #[inline]
    pub const fn lsb_depth(mut self, value: i32) -> Self {
        self.lsb_depth = Some(value);
        self
    }

    #[inline]
    pub const fn frame_duration(mut self, value: FrameDuration) -> Self {
        self.frame_duration = Some(value);
        self
    }

    #[inline]
    pub const fn prediction_disabled(mut self, value: bool) -> Self {
        self.prediction_disabled = Some(value);
        self
    }

    pub fn build(self) -> Result<Encoder, EncoderBuilderError> {
        let mut encoder = Encoder::new(self.sample_rate, self.channels, self.application)?;
        if let Some(value) = self.bitrate {
            encoder.set_bitrate(value)?;
        }
        if let Some(value) = self.complexity {
            encoder.set_complexity(value)?;
        }
        if let Some(value) = self.vbr {
            encoder.set_vbr(value)?;
        }
        if let Some(value) = self.vbr_constraint {
            encoder.set_vbr_constraint(value)?;
        }
        if let Some(value) = self.max_bandwidth {
            encoder.set_max_bandwidth(value)?;
        }
        if let Some(value) = self.signal {
            encoder.set_signal(value)?;
        }
        if let Some(value) = self.inband_fec {
            encoder.set_inband_fec(value)?;
        }
        if let Some(value) = self.packet_loss_perc {
            encoder.set_packet_loss_perc(value)?;
        }
        if let Some(value) = self.dtx {
            encoder.set_dtx(value)?;
        }
        if let Some(value) = self.lsb_depth {
            encoder.set_lsb_depth(value)?;
        }
        if let Some(value) = self.frame_duration {
            encoder.set_frame_duration(value)?;
        }
        if let Some(value) = self.prediction_disabled {
            encoder.set_prediction_disabled(value)?;
        }
        Ok(encoder)
    }
}

/// Builder for the high-level [`Decoder`] wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderBuilder {
    sample_rate: u32,
    channels: Channels,
    gain: Option<i32>,
    complexity: Option<i32>,
    phase_inversion_disabled: Option<bool>,
}

impl DecoderBuilder {
    #[inline]
    pub const fn new(sample_rate: u32, channels: Channels) -> Self {
        Self {
            sample_rate,
            channels,
            gain: None,
            complexity: None,
            phase_inversion_disabled: None,
        }
    }

    #[inline]
    pub const fn gain(mut self, value: i32) -> Self {
        self.gain = Some(value);
        self
    }

    #[inline]
    pub const fn complexity(mut self, value: i32) -> Self {
        self.complexity = Some(value);
        self
    }

    #[inline]
    pub const fn phase_inversion_disabled(mut self, value: bool) -> Self {
        self.phase_inversion_disabled = Some(value);
        self
    }

    pub fn build(self) -> Result<Decoder, DecoderBuilderError> {
        let mut decoder = Decoder::new(self.sample_rate, self.channels)?;
        if let Some(value) = self.gain {
            decoder.set_gain(value)?;
        }
        if let Some(value) = self.complexity {
            decoder.set_complexity(value)?;
        }
        if let Some(value) = self.phase_inversion_disabled {
            decoder.set_phase_inversion_disabled(value)?;
        }
        Ok(decoder)
    }
}

/// High-level Opus encoder.
#[derive(Debug)]
pub struct Encoder {
    inner: OpusEncoder<'static>,
    sample_rate: u32,
    channels: Channels,
    application: Application,
}

impl Encoder {
    #[inline]
    pub const fn builder(
        sample_rate: u32,
        channels: Channels,
        application: Application,
    ) -> EncoderBuilder {
        EncoderBuilder::new(sample_rate, channels, application)
    }

    pub fn new(
        sample_rate: u32,
        channels: Channels,
        application: Application,
    ) -> Result<Self, OpusEncoderInitError> {
        let sample_rate_i32 =
            i32::try_from(sample_rate).map_err(|_| OpusEncoderInitError::BadArgument)?;
        let inner = opus_encoder_create(
            sample_rate_i32,
            channels.to_opus_int(),
            application.to_opus_int(),
        )?;
        Ok(Self {
            inner,
            sample_rate,
            channels,
            application,
        })
    }

    #[inline]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    pub const fn channels(&self) -> Channels {
        self.channels
    }

    #[inline]
    pub const fn application(&self) -> Application {
        self.application
    }

    #[inline]
    pub fn as_raw(&self) -> &OpusEncoder<'static> {
        &self.inner
    }

    #[inline]
    pub fn as_raw_mut(&mut self) -> &mut OpusEncoder<'static> {
        &mut self.inner
    }

    #[inline]
    pub fn into_raw(self) -> OpusEncoder<'static> {
        self.inner
    }

    #[inline]
    fn pcm_frame_size<T>(&self, pcm: &[T]) -> Result<usize, OpusEncodeError> {
        let channels = self.channels.count();
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return Err(OpusEncodeError::BadArgument);
        }
        Ok(pcm.len() / channels)
    }

    #[inline]
    pub fn encode(&mut self, pcm: &[i16], packet: &mut [u8]) -> Result<usize, OpusEncodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_encode(&mut self.inner, pcm, frame_size, packet)
    }

    pub fn encode_vec(
        &mut self,
        pcm: &[i16],
        max_packet_len: usize,
    ) -> Result<Vec<u8>, OpusEncodeError> {
        let mut packet = vec![0; max_packet_len];
        let len = self.encode(pcm, &mut packet)?;
        packet.truncate(len);
        Ok(packet)
    }

    #[inline]
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        packet: &mut [u8],
    ) -> Result<usize, OpusEncodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_encode_float(&mut self.inner, pcm, frame_size, packet)
    }

    #[inline]
    pub fn encode_24bit(
        &mut self,
        pcm: &[i32],
        packet: &mut [u8],
    ) -> Result<usize, OpusEncodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_encode24(&mut self.inner, pcm, frame_size, packet)
    }

    #[inline]
    pub fn set_bitrate(&mut self, value: Bitrate) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::SetBitrate(value.to_opus_int()),
        )
    }

    pub fn bitrate(&mut self) -> Result<Bitrate, OpusEncoderCtlError> {
        let mut value = 0;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetBitrate(&mut value),
        )?;
        Ok(Bitrate::from_opus_int(value))
    }

    #[inline]
    pub fn set_vbr(&mut self, value: bool) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(&mut self.inner, OpusEncoderCtlRequest::SetVbr(value))
    }

    pub fn vbr(&mut self) -> Result<bool, OpusEncoderCtlError> {
        let mut value = false;
        opus_encoder_ctl(&mut self.inner, OpusEncoderCtlRequest::GetVbr(&mut value))?;
        Ok(value)
    }

    #[inline]
    pub fn set_vbr_constraint(&mut self, value: bool) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::SetVbrConstraint(value),
        )
    }

    pub fn vbr_constraint(&mut self) -> Result<bool, OpusEncoderCtlError> {
        let mut value = false;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetVbrConstraint(&mut value),
        )?;
        Ok(value)
    }

    #[inline]
    pub fn set_complexity(&mut self, value: i32) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(&mut self.inner, OpusEncoderCtlRequest::SetComplexity(value))
    }

    pub fn complexity(&mut self) -> Result<i32, OpusEncoderCtlError> {
        let mut value = 0;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetComplexity(&mut value),
        )?;
        Ok(value)
    }

    #[inline]
    pub fn set_signal(&mut self, value: Signal) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::SetSignal(value.to_opus_int()),
        )
    }

    pub fn signal(&mut self) -> Result<Signal, OpusEncoderCtlError> {
        let mut value = 0;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetSignal(&mut value),
        )?;
        Signal::from_opus_int(value).ok_or(OpusEncoderCtlError::InternalError)
    }

    #[inline]
    pub fn set_max_bandwidth(&mut self, value: Bandwidth) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::SetMaxBandwidth(value.to_opus_int()),
        )
    }

    pub fn max_bandwidth(&mut self) -> Result<Bandwidth, OpusEncoderCtlError> {
        let mut value = 0;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetMaxBandwidth(&mut value),
        )?;
        Bandwidth::from_opus_int(value).ok_or(OpusEncoderCtlError::InternalError)
    }

    #[inline]
    pub fn set_inband_fec(&mut self, value: bool) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(&mut self.inner, OpusEncoderCtlRequest::SetInbandFec(value))
    }

    pub fn inband_fec(&mut self) -> Result<bool, OpusEncoderCtlError> {
        let mut value = false;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetInbandFec(&mut value),
        )?;
        Ok(value)
    }

    #[inline]
    pub fn set_packet_loss_perc(&mut self, value: i32) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::SetPacketLossPerc(value),
        )
    }

    pub fn packet_loss_perc(&mut self) -> Result<i32, OpusEncoderCtlError> {
        let mut value = 0;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetPacketLossPerc(&mut value),
        )?;
        Ok(value)
    }

    #[inline]
    pub fn set_dtx(&mut self, value: bool) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(&mut self.inner, OpusEncoderCtlRequest::SetDtx(value))
    }

    pub fn dtx(&mut self) -> Result<bool, OpusEncoderCtlError> {
        let mut value = false;
        opus_encoder_ctl(&mut self.inner, OpusEncoderCtlRequest::GetDtx(&mut value))?;
        Ok(value)
    }

    #[inline]
    pub fn set_lsb_depth(&mut self, value: i32) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(&mut self.inner, OpusEncoderCtlRequest::SetLsbDepth(value))
    }

    pub fn lsb_depth(&mut self) -> Result<i32, OpusEncoderCtlError> {
        let mut value = 0;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetLsbDepth(&mut value),
        )?;
        Ok(value)
    }

    #[inline]
    pub fn set_frame_duration(&mut self, value: FrameDuration) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::SetExpertFrameDuration(value.to_opus_int()),
        )
    }

    pub fn frame_duration(&mut self) -> Result<FrameDuration, OpusEncoderCtlError> {
        let mut value = 0;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetExpertFrameDuration(&mut value),
        )?;
        FrameDuration::from_opus_int(value).ok_or(OpusEncoderCtlError::InternalError)
    }

    #[inline]
    pub fn set_prediction_disabled(&mut self, value: bool) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::SetPredictionDisabled(value),
        )
    }

    pub fn prediction_disabled(&mut self) -> Result<bool, OpusEncoderCtlError> {
        let mut value = false;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetPredictionDisabled(&mut value),
        )?;
        Ok(value)
    }

    pub fn final_range(&mut self) -> Result<u32, OpusEncoderCtlError> {
        let mut value = 0;
        opus_encoder_ctl(
            &mut self.inner,
            OpusEncoderCtlRequest::GetFinalRange(&mut value),
        )?;
        Ok(value)
    }

    #[inline]
    pub fn reset_state(&mut self) -> Result<(), OpusEncoderCtlError> {
        opus_encoder_ctl(&mut self.inner, OpusEncoderCtlRequest::ResetState)
    }
}

/// High-level Opus decoder.
#[derive(Debug)]
pub struct Decoder {
    inner: OpusDecoder<'static>,
    sample_rate: u32,
    channels: Channels,
}

impl Decoder {
    #[inline]
    pub const fn builder(sample_rate: u32, channels: Channels) -> DecoderBuilder {
        DecoderBuilder::new(sample_rate, channels)
    }

    pub fn new(sample_rate: u32, channels: Channels) -> Result<Self, OpusDecoderInitError> {
        let sample_rate_i32 =
            i32::try_from(sample_rate).map_err(|_| OpusDecoderInitError::BadArgument)?;
        let inner = opus_decoder_create(sample_rate_i32, channels.to_opus_int())?;
        Ok(Self {
            inner,
            sample_rate,
            channels,
        })
    }

    #[inline]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    pub const fn channels(&self) -> Channels {
        self.channels
    }

    #[inline]
    pub fn as_raw(&self) -> &OpusDecoder<'static> {
        &self.inner
    }

    #[inline]
    pub fn as_raw_mut(&mut self) -> &mut OpusDecoder<'static> {
        &mut self.inner
    }

    #[inline]
    pub fn into_raw(self) -> OpusDecoder<'static> {
        self.inner
    }

    #[inline]
    fn pcm_frame_size<T>(&self, pcm: &[T]) -> Result<usize, OpusDecodeError> {
        let channels = self.channels.count();
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return Err(OpusDecodeError::BadArgument);
        }
        Ok(pcm.len() / channels)
    }

    #[inline]
    pub fn packet_samples(&self, packet: &[u8]) -> Result<usize, PacketError> {
        opus_decoder_get_nb_samples(&self.inner, packet, packet.len())
    }

    #[inline]
    pub fn decode(
        &mut self,
        packet: &[u8],
        pcm: &mut [i16],
        fec: bool,
    ) -> Result<usize, OpusDecodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_decode(
            &mut self.inner,
            Some(packet),
            packet.len(),
            pcm,
            frame_size,
            fec,
        )
    }

    #[inline]
    pub fn conceal(&mut self, pcm: &mut [i16]) -> Result<usize, OpusDecodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_decode(&mut self.inner, None, 0, pcm, frame_size, false)
    }

    #[inline]
    pub fn decode_24bit(
        &mut self,
        packet: &[u8],
        pcm: &mut [i32],
        fec: bool,
    ) -> Result<usize, OpusDecodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_decode24(
            &mut self.inner,
            Some(packet),
            packet.len(),
            pcm,
            frame_size,
            fec,
        )
    }

    #[inline]
    pub fn conceal_24bit(&mut self, pcm: &mut [i32]) -> Result<usize, OpusDecodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_decode24(&mut self.inner, None, 0, pcm, frame_size, false)
    }

    #[cfg(not(feature = "fixed_point"))]
    #[inline]
    pub fn decode_float(
        &mut self,
        packet: &[u8],
        pcm: &mut [f32],
        fec: bool,
    ) -> Result<usize, OpusDecodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_decode_float(
            &mut self.inner,
            Some(packet),
            packet.len(),
            pcm,
            frame_size,
            fec,
        )
    }

    #[cfg(not(feature = "fixed_point"))]
    #[inline]
    pub fn conceal_float(&mut self, pcm: &mut [f32]) -> Result<usize, OpusDecodeError> {
        let frame_size = self.pcm_frame_size(pcm)?;
        opus_decode_float(&mut self.inner, None, 0, pcm, frame_size, false)
    }

    #[inline]
    pub fn set_gain(&mut self, value: i32) -> Result<(), OpusDecoderCtlError> {
        opus_decoder_ctl(&mut self.inner, OpusDecoderCtlRequest::SetGain(value))
    }

    pub fn gain(&mut self) -> Result<i32, OpusDecoderCtlError> {
        let mut value = 0;
        opus_decoder_ctl(&mut self.inner, OpusDecoderCtlRequest::GetGain(&mut value))?;
        Ok(value)
    }

    #[inline]
    pub fn set_complexity(&mut self, value: i32) -> Result<(), OpusDecoderCtlError> {
        opus_decoder_ctl(&mut self.inner, OpusDecoderCtlRequest::SetComplexity(value))
    }

    pub fn complexity(&mut self) -> Result<i32, OpusDecoderCtlError> {
        let mut value = 0;
        opus_decoder_ctl(
            &mut self.inner,
            OpusDecoderCtlRequest::GetComplexity(&mut value),
        )?;
        Ok(value)
    }

    pub fn bandwidth(&mut self) -> Result<Option<Bandwidth>, OpusDecoderCtlError> {
        let mut value = 0;
        opus_decoder_ctl(
            &mut self.inner,
            OpusDecoderCtlRequest::GetBandwidth(&mut value),
        )?;
        if value == 0 {
            return Ok(None);
        }
        Bandwidth::from_opus_int(value)
            .map(Some)
            .ok_or(OpusDecoderCtlError::Unimplemented)
    }

    pub fn pitch(&mut self) -> Result<i32, OpusDecoderCtlError> {
        let mut value = 0;
        opus_decoder_ctl(&mut self.inner, OpusDecoderCtlRequest::GetPitch(&mut value))?;
        Ok(value)
    }

    pub fn last_packet_duration(&mut self) -> Result<usize, OpusDecoderCtlError> {
        let mut value = 0;
        opus_decoder_ctl(
            &mut self.inner,
            OpusDecoderCtlRequest::GetLastPacketDuration(&mut value),
        )?;
        usize::try_from(value).map_err(|_| OpusDecoderCtlError::BadArgument)
    }

    pub fn final_range(&mut self) -> Result<u32, OpusDecoderCtlError> {
        let mut value = 0;
        opus_decoder_ctl(
            &mut self.inner,
            OpusDecoderCtlRequest::GetFinalRange(&mut value),
        )?;
        Ok(value)
    }

    #[inline]
    pub fn set_phase_inversion_disabled(&mut self, value: bool) -> Result<(), OpusDecoderCtlError> {
        opus_decoder_ctl(
            &mut self.inner,
            OpusDecoderCtlRequest::SetPhaseInversionDisabled(value),
        )
    }

    pub fn phase_inversion_disabled(&mut self) -> Result<bool, OpusDecoderCtlError> {
        let mut value = false;
        opus_decoder_ctl(
            &mut self.inner,
            OpusDecoderCtlRequest::GetPhaseInversionDisabled(&mut value),
        )?;
        Ok(value)
    }

    #[inline]
    pub fn reset_state(&mut self) -> Result<(), OpusDecoderCtlError> {
        opus_decoder_ctl(&mut self.inner, OpusDecoderCtlRequest::ResetState)
    }
}
