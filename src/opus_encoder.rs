//! Top-level Opus encoder front-end.
//!
//! This module begins the port of `opus_encoder.c` by providing the encoder
//! size/init/CTL entry points and a minimal `opus_encode` implementation for
//! SILK-only packets. Hybrid/CELT modes will be wired once the remaining CELT
//! bitstream packing path is ported.

use alloc::vec::Vec;

#[cfg(not(feature = "fixed_point"))]
use crate::analysis::{
    TonalityAnalysisState, run_analysis, tonality_analysis_init, tonality_analysis_reset,
};
use crate::celt::AnalysisInfo;
use crate::celt::{
    CeltEncoderCtlError, CeltEncoderInitError, EncoderCtlRequest as CeltEncoderCtlRequest,
    OwnedCeltEncoder, canonical_mode, opus_custom_encoder_create, opus_custom_encoder_ctl,
    opus_select_arch,
};
use crate::opus_multistream::{OPUS_AUTO, OPUS_BITRATE_MAX};
use crate::packet::Bandwidth;
use crate::range::RangeEncoder;
use crate::silk::enc_api::{PrefillMode, silk_encode, silk_init_encoder};
use crate::silk::errors::SilkError;
use crate::silk::EncControl as SilkEncControl;

const MAX_CHANNELS: usize = 2;

pub(crate) const MODE_SILK_ONLY: i32 = 1000;
#[allow(dead_code)]
pub(crate) const MODE_HYBRID: i32 = 1001;
pub(crate) const MODE_CELT_ONLY: i32 = 1002;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusApplication {
    Voip,
    Audio,
    RestrictedLowDelay,
}

impl OpusApplication {
    #[inline]
    fn from_opus_int(value: i32) -> Option<Self> {
        match value {
            2048 => Some(Self::Voip),
            2049 => Some(Self::Audio),
            2051 => Some(Self::RestrictedLowDelay),
            _ => None,
        }
    }

    #[inline]
    #[cfg(not(feature = "fixed_point"))]
    const fn to_opus_int(self) -> i32 {
        match self {
            Self::Voip => 2048,
            Self::Audio => 2049,
            Self::RestrictedLowDelay => 2051,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusEncoderInitError {
    BadArgument,
    SilkInit,
    CeltInit,
}

impl OpusEncoderInitError {
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::SilkInit | Self::CeltInit => -3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusEncoderCtlError {
    BadArgument,
    Unimplemented,
    Silk(SilkError),
    InternalError,
}

impl OpusEncoderCtlError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::Unimplemented => -5,
            Self::Silk(_) | Self::InternalError => -3,
        }
    }
}

impl From<SilkError> for OpusEncoderCtlError {
    #[inline]
    fn from(value: SilkError) -> Self {
        Self::Silk(value)
    }
}

impl From<CeltEncoderInitError> for OpusEncoderCtlError {
    #[inline]
    fn from(value: CeltEncoderInitError) -> Self {
        let _ = value;
        Self::InternalError
    }
}

impl From<CeltEncoderCtlError> for OpusEncoderCtlError {
    #[inline]
    fn from(value: CeltEncoderCtlError) -> Self {
        let _ = value;
        Self::InternalError
    }
}

pub enum OpusEncoderCtlRequest<'req> {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusEncodeError {
    BadArgument,
    BufferTooSmall,
    InternalError,
    Unimplemented,
    Silk(SilkError),
}

impl OpusEncodeError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError | Self::Silk(_) => -3,
            Self::Unimplemented => -5,
        }
    }
}

impl From<SilkError> for OpusEncodeError {
    #[inline]
    fn from(value: SilkError) -> Self {
        Self::Silk(value)
    }
}

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

#[repr(C)]
struct StereoWidthStateLayout {
    xx: f32,
    xy: f32,
    yy: f32,
    smoothed_width: f32,
    max_follower: f32,
}

#[repr(C)]
struct OpusEncoderLayout {
    celt_enc_offset: i32,
    silk_enc_offset: i32,
    silk_mode: SilkEncControlLayout,
    application: i32,
    channels: i32,
    delay_compensation: i32,
    force_channels: i32,
    signal_type: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    user_forced_mode: i32,
    voice_ratio: i32,
    fs: i32,
    use_vbr: i32,
    vbr_constraint: i32,
    variable_duration: i32,
    bitrate_bps: i32,
    user_bitrate_bps: i32,
    lsb_depth: i32,
    encoder_buffer: i32,
    lfe: i32,
    arch: i32,
    use_dtx: i32,
    fec_config: i32,
    #[cfg(not(feature = "fixed_point"))]
    analysis: TonalityAnalysisState,
    stream_channels: i32,
    hybrid_stereo_width_q14: i16,
    variable_hp_smth2_q15: i32,
    prev_hb_gain: f32,
    hp_mem: [f32; 4],
    mode: i32,
    prev_mode: i32,
    prev_channels: i32,
    prev_framesize: i32,
    bandwidth: i32,
    auto_bandwidth: i32,
    silk_bw_switch: i32,
    first: i32,
    energy_masking: *const f32,
    width_mem: StereoWidthStateLayout,
    delay_buffer: [f32; 960],
    #[cfg(not(feature = "fixed_point"))]
    detected_bandwidth: i32,
    #[cfg(not(feature = "fixed_point"))]
    nb_no_activity_ms_q1: i32,
    #[cfg(not(feature = "fixed_point"))]
    peak_signal_energy: f32,
    nonfinal_frame: i32,
    range_final: u32,
}

#[repr(C)]
struct SilkEncControlLayout {
    n_channels_api: i32,
    n_channels_internal: i32,
    api_sample_rate: i32,
    max_internal_sample_rate: i32,
    min_internal_sample_rate: i32,
    desired_internal_sample_rate: i32,
    payload_size_ms: i32,
    bit_rate: i32,
    packet_loss_percentage: i32,
    complexity: i32,
    use_in_band_fec: i32,
    use_dred: i32,
    lbrr_coded: i32,
    use_dtx: i32,
    use_cbr: i32,
    max_bits: i32,
    to_mono: bool,
    opus_can_switch: bool,
    reduced_dependency: bool,
    internal_sample_rate: i32,
    allow_bandwidth_switch: bool,
    in_wb_mode_without_variable_lp: bool,
    stereo_width_q14: i32,
    switch_ready: bool,
    signal_type: i32,
    offset: i32,
}

#[must_use]
pub fn opus_encoder_get_size(channels: usize) -> Option<usize> {
    if channels == 0 || channels > MAX_CHANNELS {
        return None;
    }

    let mut silk_size = 0usize;
    crate::silk::get_encoder_size::get_encoder_size(&mut silk_size).ok()?;
    let silk_size = align(silk_size);

    let celt_size = crate::celt::celt_encoder_get_size(channels)?;
    let header_size = align(core::mem::size_of::<OpusEncoderLayout>());

    Some(header_size + silk_size + celt_size)
}

#[derive(Debug)]
pub struct OpusEncoder<'mode> {
    celt: OwnedCeltEncoder<'mode>,
    silk: crate::silk::encoder::state::Encoder,
    silk_mode: SilkEncControl,
    #[cfg(not(feature = "fixed_point"))]
    analysis: TonalityAnalysisState,
    analysis_info: AnalysisInfo,
    application: OpusApplication,
    channels: i32,
    stream_channels: i32,
    fs: i32,
    arch: i32,
    use_vbr: bool,
    vbr_constraint: bool,
    user_bitrate_bps: i32,
    bitrate_bps: i32,
    packet_loss_perc: i32,
    complexity: i32,
    inband_fec: bool,
    use_dtx: bool,
    mode: i32,
    bandwidth: Bandwidth,
    range_final: u32,
}

impl<'mode> OpusEncoder<'mode> {
    pub fn init(&mut self, fs: i32, channels: i32, application: i32) -> Result<(), OpusEncoderInitError> {
        if !matches!(fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000) || !matches!(channels, 1 | 2)
        {
            return Err(OpusEncoderInitError::BadArgument);
        }
        let application = OpusApplication::from_opus_int(application)
            .ok_or(OpusEncoderInitError::BadArgument)?;

        let mode = canonical_mode().ok_or(OpusEncoderInitError::CeltInit)?;
        self.celt =
            opus_custom_encoder_create(mode, fs, channels as usize, 0).map_err(|_| OpusEncoderInitError::CeltInit)?;

        self.arch = opus_select_arch();
        self.channels = channels;
        self.stream_channels = channels;
        self.fs = fs;
        self.application = application;

        // Reset SILK encoder.
        silk_init_encoder(&mut self.silk, self.arch, &mut self.silk_mode)
            .map_err(|_| OpusEncoderInitError::SilkInit)?;

        // Default SILK parameters from `opus_encoder_init`.
        self.silk_mode.n_channels_api = channels;
        self.silk_mode.n_channels_internal = channels;
        self.silk_mode.api_sample_rate = fs;
        self.silk_mode.max_internal_sample_rate = 16_000;
        self.silk_mode.min_internal_sample_rate = 8_000;
        self.silk_mode.desired_internal_sample_rate = 16_000;
        self.silk_mode.payload_size_ms = 20;
        self.silk_mode.bit_rate = 25_000;
        self.silk_mode.packet_loss_percentage = 0;
        self.silk_mode.complexity = 9;
        self.silk_mode.use_in_band_fec = 0;
        self.silk_mode.use_dred = 0;
        self.silk_mode.use_dtx = 0;
        self.silk_mode.use_cbr = 0;
        self.silk_mode.reduced_dependency = false;

        // Keep CELT's signalling disabled for later frame packing.
        opus_custom_encoder_ctl(self.celt.encoder(), CeltEncoderCtlRequest::SetSignalling(0))
            .map_err(|_| OpusEncoderInitError::CeltInit)?;
        opus_custom_encoder_ctl(
            self.celt.encoder(),
            CeltEncoderCtlRequest::SetComplexity(self.silk_mode.complexity),
        )
        .map_err(|_| OpusEncoderInitError::CeltInit)?;

        self.use_vbr = true;
        self.vbr_constraint = true;
        self.user_bitrate_bps = OPUS_AUTO;
        self.bitrate_bps = 3000 + fs * channels;
        self.packet_loss_perc = 0;
        self.complexity = self.silk_mode.complexity;
        self.inband_fec = false;
        self.use_dtx = false;

        self.mode = MODE_SILK_ONLY;
        self.bandwidth = Bandwidth::Wide;
        self.range_final = 0;

        #[cfg(not(feature = "fixed_point"))]
        {
            tonality_analysis_init(&mut self.analysis, fs);
        }
        self.analysis_info = AnalysisInfo::default();

        Ok(())
    }

    fn reset_state(&mut self) -> Result<(), OpusEncoderCtlError> {
        silk_init_encoder(&mut self.silk, self.arch, &mut self.silk_mode)?;
        opus_custom_encoder_ctl(self.celt.encoder(), CeltEncoderCtlRequest::ResetState)?;
        #[cfg(not(feature = "fixed_point"))]
        {
            tonality_analysis_reset(&mut self.analysis);
        }
        self.analysis_info = AnalysisInfo::default();
        self.range_final = 0;
        Ok(())
    }

    fn configure_silk_control(&mut self, frame_rate: i32, max_data_bytes: usize) {
        // Frame size in milliseconds is derived from the API-level sampling rate.
        let payload_size_ms = 1000 / frame_rate;
        self.silk_mode.payload_size_ms = payload_size_ms;
        self.silk_mode.n_channels_api = self.channels;
        self.silk_mode.n_channels_internal = self.stream_channels;
        self.silk_mode.api_sample_rate = self.fs;
        self.silk_mode.max_internal_sample_rate = 16_000;
        self.silk_mode.min_internal_sample_rate = 8_000;
        self.silk_mode.desired_internal_sample_rate = 16_000;
        self.silk_mode.packet_loss_percentage = self.packet_loss_perc;
        self.silk_mode.complexity = self.complexity;
        self.silk_mode.use_in_band_fec = i32::from(self.inband_fec);
        self.silk_mode.use_dtx = i32::from(self.use_dtx);
        self.silk_mode.use_cbr = i32::from(!self.use_vbr);
        self.silk_mode.reduced_dependency = false;
        self.silk_mode.opus_can_switch = false;
        self.silk_mode.max_bits = (max_data_bytes.saturating_mul(8)).min(i32::MAX as usize) as i32;

        let bitrate = match self.user_bitrate_bps {
            OPUS_AUTO => self.bitrate_bps,
            OPUS_BITRATE_MAX => 80_000,
            value => value,
        };
        self.silk_mode.bit_rate = bitrate.clamp(5_000, 80_000);
    }

    fn bandwidth_from_silk_control(control: &SilkEncControl) -> Bandwidth {
        match control.internal_sample_rate {
            8_000 => Bandwidth::Narrow,
            12_000 => Bandwidth::Medium,
            _ => Bandwidth::Wide,
        }
    }
}

fn gen_toc(mode: i32, framerate: i32, bandwidth: Bandwidth, channels: i32) -> u8 {
    let mut framerate = framerate;
    let mut period = 0i32;
    while framerate < 400 {
        framerate <<= 1;
        period += 1;
    }

    let bw_int = bandwidth.to_opus_int();
    let mut toc = if mode == MODE_SILK_ONLY {
        let bw_index = (bw_int - Bandwidth::Narrow.to_opus_int()).clamp(0, 3);
        let period_index = (period - 2).clamp(0, 3);
        ((bw_index as u8) << 5) | ((period_index as u8) << 3)
    } else if mode == MODE_CELT_ONLY {
        let mut tmp = bw_int - Bandwidth::Medium.to_opus_int();
        if tmp < 0 {
            tmp = 0;
        }
        let period_index = period.clamp(0, 3);
        0x80 | ((tmp as u8) << 5) | ((period_index as u8) << 3)
    } else {
        // Hybrid
        let bw_flag = if bandwidth == Bandwidth::Full { 1 } else { 0 };
        let period_index = (period - 2).clamp(0, 3);
        0x60 | ((bw_flag as u8) << 4) | ((period_index as u8) << 3)
    };

    if channels == 2 {
        toc |= 0x04;
    }
    toc
}

pub fn opus_encoder_create<'mode>(
    fs: i32,
    channels: i32,
    application: i32,
) -> Result<OpusEncoder<'mode>, OpusEncoderInitError> {
    if !matches!(fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000)
        || !matches!(channels, 1 | 2)
        || OpusApplication::from_opus_int(application).is_none()
    {
        return Err(OpusEncoderInitError::BadArgument);
    }

    let mode = canonical_mode().ok_or(OpusEncoderInitError::CeltInit)?;
    let celt = opus_custom_encoder_create(mode, fs, channels as usize, 0)
        .map_err(|_| OpusEncoderInitError::CeltInit)?;

    let mut encoder = OpusEncoder {
        celt,
        silk: crate::silk::encoder::state::Encoder::default(),
        silk_mode: SilkEncControl::default(),
        #[cfg(not(feature = "fixed_point"))]
        analysis: TonalityAnalysisState::new(fs),
        analysis_info: AnalysisInfo::default(),
        application: OpusApplication::Voip,
        channels,
        stream_channels: channels,
        fs,
        arch: opus_select_arch(),
        use_vbr: true,
        vbr_constraint: true,
        user_bitrate_bps: OPUS_AUTO,
        bitrate_bps: 0,
        packet_loss_perc: 0,
        complexity: 9,
        inband_fec: false,
        use_dtx: false,
        mode: MODE_SILK_ONLY,
        bandwidth: Bandwidth::Wide,
        range_final: 0,
    };

    encoder.init(fs, channels, application)?;
    Ok(encoder)
}

pub fn opus_encoder_ctl<'req>(
    encoder: &mut OpusEncoder<'_>,
    request: OpusEncoderCtlRequest<'req>,
) -> Result<(), OpusEncoderCtlError> {
    match request {
        OpusEncoderCtlRequest::SetBitrate(value) => {
            if value != OPUS_AUTO && value != OPUS_BITRATE_MAX && value <= 0 {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.user_bitrate_bps = value;
            if value != OPUS_AUTO {
                encoder.bitrate_bps = value;
            }
        }
        OpusEncoderCtlRequest::GetBitrate(out) => {
            *out = encoder.user_bitrate_bps;
        }
        OpusEncoderCtlRequest::SetVbr(value) => {
            encoder.use_vbr = value;
        }
        OpusEncoderCtlRequest::GetVbr(out) => {
            *out = encoder.use_vbr;
        }
        OpusEncoderCtlRequest::SetVbrConstraint(value) => {
            encoder.vbr_constraint = value;
        }
        OpusEncoderCtlRequest::GetVbrConstraint(out) => {
            *out = encoder.vbr_constraint;
        }
        OpusEncoderCtlRequest::SetComplexity(value) => {
            if !(0..=10).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.complexity = value;
            opus_custom_encoder_ctl(encoder.celt.encoder(), CeltEncoderCtlRequest::SetComplexity(value))?;
        }
        OpusEncoderCtlRequest::GetComplexity(out) => {
            *out = encoder.complexity;
        }
        OpusEncoderCtlRequest::SetPacketLossPerc(value) => {
            if !(0..=100).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.packet_loss_perc = value;
        }
        OpusEncoderCtlRequest::GetPacketLossPerc(out) => {
            *out = encoder.packet_loss_perc;
        }
        OpusEncoderCtlRequest::SetInbandFec(value) => {
            encoder.inband_fec = value;
        }
        OpusEncoderCtlRequest::GetInbandFec(out) => {
            *out = encoder.inband_fec;
        }
        OpusEncoderCtlRequest::SetDtx(value) => {
            encoder.use_dtx = value;
        }
        OpusEncoderCtlRequest::GetDtx(out) => {
            *out = encoder.use_dtx;
        }
        OpusEncoderCtlRequest::GetFinalRange(out) => {
            *out = encoder.range_final;
        }
        OpusEncoderCtlRequest::ResetState => {
            encoder.reset_state()?;
        }
    }
    Ok(())
}

pub fn opus_encode(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[i16],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusEncodeError> {
    let channels = usize::try_from(encoder.channels).map_err(|_| OpusEncodeError::BadArgument)?;
    if channels == 0 || channels > MAX_CHANNELS || frame_size == 0 {
        return Err(OpusEncodeError::BadArgument);
    }
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusEncodeError::BadArgument);
    }
    if data.len() < 2 {
        return Err(OpusEncodeError::BufferTooSmall);
    }

    if encoder.mode != MODE_SILK_ONLY {
        return Err(OpusEncodeError::Unimplemented);
    }

    let frame_size_i32 = i32::try_from(frame_size).map_err(|_| OpusEncodeError::BadArgument)?;
    let frame_rate = encoder
        .fs
        .checked_div(frame_size_i32)
        .ok_or(OpusEncodeError::BadArgument)?;
    if frame_rate != 50 {
        // Matches the current crate limitation of 20 ms frames while the
        // multi-frame packing path is ported.
        return Err(OpusEncodeError::Unimplemented);
    }

    let max_payload_bytes = data.len().saturating_sub(1);
    encoder.configure_silk_control(frame_rate, max_payload_bytes);

    #[cfg(feature = "fixed_point")]
    let activity = 1i32;
    #[cfg(not(feature = "fixed_point"))]
    let activity = {
        encoder.analysis.application = encoder.application.to_opus_int();
        run_analysis(
            &mut encoder.analysis,
            encoder.celt.mode,
            Some(&pcm[..required]),
            frame_size,
            frame_size,
            0,
            -1,
            encoder.channels,
            encoder.fs,
            16,
            &mut encoder.analysis_info,
        );
        if encoder.analysis_info.valid && encoder.analysis_info.activity < 0.02 {
            0
        } else {
            1
        }
    };

    let mut range_encoder = RangeEncoder::new();
    let mut bytes_out = 0i32;
    silk_encode(
        &mut encoder.silk,
        &mut encoder.silk_mode,
        &pcm[..required],
        &mut range_encoder,
        &mut bytes_out,
        PrefillMode::None,
        activity,
    )?;

    let range_final = range_encoder.range_final();
    let payload = range_encoder.finish();
    if payload.len() > max_payload_bytes {
        return Err(OpusEncodeError::BufferTooSmall);
    }

    let bandwidth = OpusEncoder::bandwidth_from_silk_control(&encoder.silk_mode);
    let toc = gen_toc(MODE_SILK_ONLY, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;

    data[0] = toc;
    data[1..1 + payload.len()].copy_from_slice(&payload);
    encoder.bandwidth = bandwidth;
    encoder.range_final = range_final;

    Ok(1 + payload.len())
}

pub fn opus_encode_float(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusEncodeError> {
    let channels = usize::try_from(encoder.channels).map_err(|_| OpusEncodeError::BadArgument)?;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusEncodeError::BadArgument);
    }

    let mut tmp = Vec::with_capacity(required);
    for &sample in pcm.iter().take(required) {
        let scaled = libm::roundf(sample * 32768.0);
        tmp.push(
            scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16,
        );
    }

    opus_encode(encoder, &tmp, frame_size, data)
}

#[cfg(test)]
mod tests {
    use super::{
        OpusEncodeError, OpusEncoderCtlRequest, OpusEncoderInitError, opus_encode,
        opus_encoder_create, opus_encoder_ctl, opus_encoder_get_size,
    };
    use crate::packet::{Bandwidth, opus_packet_get_bandwidth, opus_packet_get_mode};

    #[test]
    fn encoder_get_size_matches_components() {
        let size = opus_encoder_get_size(1).expect("size");
        assert!(size > 0);
    }

    #[test]
    fn create_rejects_invalid_arguments() {
        assert_eq!(
            opus_encoder_create(44_100, 1, 2048).unwrap_err(),
            OpusEncoderInitError::BadArgument
        );
        assert_eq!(
            opus_encoder_create(48_000, 3, 2048).unwrap_err(),
            OpusEncoderInitError::BadArgument
        );
        assert_eq!(
            opus_encoder_create(48_000, 1, 123).unwrap_err(),
            OpusEncoderInitError::BadArgument
        );
    }

    #[test]
    fn encodes_silk_only_frame_with_valid_toc() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        let pcm = [0i16; 960];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 960, &mut out).expect("encode");
        assert!(len > 1);
        assert_eq!(opus_packet_get_mode(&out[..len]).unwrap(), crate::packet::Mode::SILK);
        assert_eq!(opus_packet_get_bandwidth(&out[..len]).unwrap(), Bandwidth::Wide);
    }

    #[test]
    fn ctl_round_trips_basic_settings() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbr(false)).unwrap();
        let mut vbr = true;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetVbr(&mut vbr)).unwrap();
        assert!(!vbr);
    }

    #[test]
    fn encode_rejects_non_20ms_frames_for_now() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        let pcm = [0i16; 480];
        let mut out = [0u8; 4000];
        let err = opus_encode(&mut enc, &pcm, 480, &mut out).unwrap_err();
        assert_eq!(err, OpusEncodeError::Unimplemented);
    }
}
