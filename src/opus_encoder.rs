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
pub(crate) const OPUS_SIGNAL_VOICE: i32 = 3001;
pub(crate) const OPUS_SIGNAL_MUSIC: i32 = 3002;
pub(crate) const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
pub(crate) const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
pub(crate) const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
pub(crate) const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
pub(crate) const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;
pub(crate) const OPUS_FRAMESIZE_ARG: i32 = 5000;
pub(crate) const OPUS_FRAMESIZE_2_5_MS: i32 = 5001;
pub(crate) const OPUS_FRAMESIZE_5_MS: i32 = 5002;
pub(crate) const OPUS_FRAMESIZE_10_MS: i32 = 5003;
pub(crate) const OPUS_FRAMESIZE_20_MS: i32 = 5004;
pub(crate) const OPUS_FRAMESIZE_40_MS: i32 = 5005;
pub(crate) const OPUS_FRAMESIZE_60_MS: i32 = 5006;
pub(crate) const OPUS_FRAMESIZE_80_MS: i32 = 5007;
pub(crate) const OPUS_FRAMESIZE_100_MS: i32 = 5008;
pub(crate) const OPUS_FRAMESIZE_120_MS: i32 = 5009;

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
    SetForceChannels(i32),
    GetForceChannels(&'req mut i32),
    SetMaxBandwidth(i32),
    GetMaxBandwidth(&'req mut i32),
    SetBandwidth(i32),
    GetBandwidth(&'req mut i32),
    SetVbr(bool),
    GetVbr(&'req mut bool),
    SetVbrConstraint(bool),
    GetVbrConstraint(&'req mut bool),
    SetComplexity(i32),
    GetComplexity(&'req mut i32),
    SetSignal(i32),
    GetSignal(&'req mut i32),
    SetPacketLossPerc(i32),
    GetPacketLossPerc(&'req mut i32),
    SetInbandFec(bool),
    GetInbandFec(&'req mut bool),
    SetDtx(bool),
    GetDtx(&'req mut bool),
    SetLsbDepth(i32),
    GetLsbDepth(&'req mut i32),
    SetExpertFrameDuration(i32),
    GetExpertFrameDuration(&'req mut i32),
    SetPredictionDisabled(bool),
    GetPredictionDisabled(&'req mut bool),
    SetPhaseInversionDisabled(bool),
    GetPhaseInversionDisabled(&'req mut bool),
    SetForceMode(i32),
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
    force_channels: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    signal_type: i32,
    user_forced_mode: i32,
    lsb_depth: i32,
    variable_duration: i32,
    prediction_disabled: bool,
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
        self.force_channels = OPUS_AUTO;
        self.user_bandwidth = OPUS_AUTO;
        self.max_bandwidth = OPUS_BANDWIDTH_FULLBAND;
        self.signal_type = OPUS_AUTO;
        self.user_forced_mode = OPUS_AUTO;
        self.lsb_depth = 24;
        self.variable_duration = OPUS_FRAMESIZE_ARG;
        self.prediction_disabled = false;

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

    fn configure_silk_control(&mut self, frame_size: i32, max_data_bytes: usize) {
        // Frame size in milliseconds is derived from the API-level sampling rate.
        let payload_size_ms = (1000i64 * i64::from(frame_size) / i64::from(self.fs)) as i32;
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
        self.silk_mode.reduced_dependency = self.prediction_disabled;
        self.silk_mode.opus_can_switch = false;
        self.silk_mode.max_bits = (max_data_bytes.saturating_mul(8)).min(i32::MAX as usize) as i32;

        self.silk_mode.max_internal_sample_rate =
            max_internal_sample_rate_for_bandwidth(self.user_bandwidth, self.max_bandwidth);

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

fn frame_size_select(frame_size: i32, variable_duration: i32, fs: i32) -> Option<i32> {
    if frame_size < fs / 400 {
        return None;
    }

    let new_size = if variable_duration == OPUS_FRAMESIZE_ARG {
        frame_size
    } else if (OPUS_FRAMESIZE_2_5_MS..=OPUS_FRAMESIZE_120_MS).contains(&variable_duration) {
        if variable_duration <= OPUS_FRAMESIZE_40_MS {
            (fs / 400) << (variable_duration - OPUS_FRAMESIZE_2_5_MS)
        } else {
            (variable_duration - OPUS_FRAMESIZE_2_5_MS - 2) * fs / 50
        }
    } else {
        return None;
    };

    if new_size > frame_size {
        return None;
    }

    let valid = 400 * new_size == fs
        || 200 * new_size == fs
        || 100 * new_size == fs
        || 50 * new_size == fs
        || 25 * new_size == fs
        || 50 * new_size == 3 * fs
        || 50 * new_size == 4 * fs
        || 50 * new_size == 5 * fs
        || 50 * new_size == 6 * fs;

    valid.then_some(new_size)
}

fn select_bandwidth(user_bandwidth: i32, max_bandwidth: i32) -> Bandwidth {
    let requested = if user_bandwidth == OPUS_AUTO {
        max_bandwidth
    } else {
        user_bandwidth
    };
    Bandwidth::from_opus_int(requested).unwrap_or(Bandwidth::Wide)
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
        force_channels: OPUS_AUTO,
        user_bandwidth: OPUS_AUTO,
        max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
        signal_type: OPUS_AUTO,
        user_forced_mode: OPUS_AUTO,
        lsb_depth: 24,
        variable_duration: OPUS_FRAMESIZE_ARG,
        prediction_disabled: false,
        mode: MODE_SILK_ONLY,
        bandwidth: Bandwidth::Wide,
        range_final: 0,
    };

    encoder.init(fs, channels, application)?;
    Ok(encoder)
}

fn max_internal_sample_rate_for_bandwidth(user_bandwidth: i32, max_bandwidth: i32) -> i32 {
    let selected = if user_bandwidth == OPUS_AUTO {
        max_bandwidth
    } else {
        user_bandwidth
    };
    match selected {
        OPUS_BANDWIDTH_NARROWBAND => 8_000,
        OPUS_BANDWIDTH_MEDIUMBAND => 12_000,
        OPUS_BANDWIDTH_WIDEBAND | OPUS_BANDWIDTH_SUPERWIDEBAND | OPUS_BANDWIDTH_FULLBAND => 16_000,
        _ => 16_000,
    }
}

fn is_valid_bandwidth(value: i32) -> bool {
    matches!(
        value,
        OPUS_BANDWIDTH_NARROWBAND
            | OPUS_BANDWIDTH_MEDIUMBAND
            | OPUS_BANDWIDTH_WIDEBAND
            | OPUS_BANDWIDTH_SUPERWIDEBAND
            | OPUS_BANDWIDTH_FULLBAND
    )
}

fn is_valid_signal(value: i32) -> bool {
    matches!(value, OPUS_AUTO | OPUS_SIGNAL_VOICE | OPUS_SIGNAL_MUSIC)
}

fn is_valid_expert_frame_duration(value: i32) -> bool {
    matches!(
        value,
        OPUS_FRAMESIZE_ARG
            | OPUS_FRAMESIZE_2_5_MS
            | OPUS_FRAMESIZE_5_MS
            | OPUS_FRAMESIZE_10_MS
            | OPUS_FRAMESIZE_20_MS
            | OPUS_FRAMESIZE_40_MS
            | OPUS_FRAMESIZE_60_MS
            | OPUS_FRAMESIZE_80_MS
            | OPUS_FRAMESIZE_100_MS
            | OPUS_FRAMESIZE_120_MS
    )
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
        OpusEncoderCtlRequest::SetForceChannels(value) => {
            if value != OPUS_AUTO && (value < 1 || value > encoder.channels) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.force_channels = value;
        }
        OpusEncoderCtlRequest::GetForceChannels(out) => {
            *out = encoder.force_channels;
        }
        OpusEncoderCtlRequest::SetMaxBandwidth(value) => {
            if !is_valid_bandwidth(value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.max_bandwidth = value;
            encoder.silk_mode.max_internal_sample_rate =
                max_internal_sample_rate_for_bandwidth(encoder.user_bandwidth, encoder.max_bandwidth);
        }
        OpusEncoderCtlRequest::GetMaxBandwidth(out) => {
            *out = encoder.max_bandwidth;
        }
        OpusEncoderCtlRequest::SetBandwidth(value) => {
            if value != OPUS_AUTO && !is_valid_bandwidth(value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.user_bandwidth = value;
            encoder.silk_mode.max_internal_sample_rate =
                max_internal_sample_rate_for_bandwidth(encoder.user_bandwidth, encoder.max_bandwidth);
        }
        OpusEncoderCtlRequest::GetBandwidth(out) => {
            *out = encoder.bandwidth.to_opus_int();
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
        OpusEncoderCtlRequest::SetSignal(value) => {
            if !is_valid_signal(value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.signal_type = value;
        }
        OpusEncoderCtlRequest::GetSignal(out) => {
            *out = encoder.signal_type;
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
        OpusEncoderCtlRequest::SetLsbDepth(value) => {
            if !(8..=24).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.lsb_depth = value;
        }
        OpusEncoderCtlRequest::GetLsbDepth(out) => {
            *out = encoder.lsb_depth;
        }
        OpusEncoderCtlRequest::SetExpertFrameDuration(value) => {
            if !is_valid_expert_frame_duration(value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.variable_duration = value;
        }
        OpusEncoderCtlRequest::GetExpertFrameDuration(out) => {
            *out = encoder.variable_duration;
        }
        OpusEncoderCtlRequest::SetPredictionDisabled(value) => {
            encoder.prediction_disabled = value;
            encoder.silk_mode.reduced_dependency = value;
        }
        OpusEncoderCtlRequest::GetPredictionDisabled(out) => {
            *out = encoder.prediction_disabled;
        }
        OpusEncoderCtlRequest::SetPhaseInversionDisabled(value) => {
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetPhaseInversionDisabled(value),
            )?;
        }
        OpusEncoderCtlRequest::GetPhaseInversionDisabled(out) => {
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::GetPhaseInversionDisabled(out),
            )?;
        }
        OpusEncoderCtlRequest::SetForceMode(value) => {
            if value != OPUS_AUTO && !(MODE_SILK_ONLY..=MODE_CELT_ONLY).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.user_forced_mode = value;
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
    let required_input = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < required_input {
        return Err(OpusEncodeError::BadArgument);
    }
    if data.len() < 2 {
        return Err(OpusEncodeError::BufferTooSmall);
    }

    let frame_size_i32 = i32::try_from(frame_size).map_err(|_| OpusEncodeError::BadArgument)?;
    let frame_size_i32 = frame_size_select(frame_size_i32, encoder.variable_duration, encoder.fs)
        .ok_or(OpusEncodeError::BadArgument)?;
    let frame_size = usize::try_from(frame_size_i32).map_err(|_| OpusEncodeError::BadArgument)?;

    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;

    let frame_rate = encoder
        .fs
        .checked_div(frame_size_i32)
        .ok_or(OpusEncodeError::BadArgument)?;

    let max_payload_bytes = data.len().saturating_sub(1);

    let stream_channels = if encoder.force_channels == OPUS_AUTO {
        encoder.channels
    } else {
        encoder.force_channels
    };
    encoder.stream_channels = stream_channels;

    let mut mode = if encoder.user_forced_mode == OPUS_AUTO {
        encoder.mode
    } else {
        encoder.user_forced_mode
    };
    let min_celt = usize::try_from(encoder.fs / 100).map_err(|_| OpusEncodeError::BadArgument)?;
    if mode == OPUS_AUTO {
        if frame_size <= min_celt {
            mode = MODE_CELT_ONLY;
        } else {
            mode = MODE_SILK_ONLY;
        }
    }

    let max_celt = usize::try_from(encoder.fs / 50).map_err(|_| OpusEncodeError::BadArgument)?;
    if mode != MODE_SILK_ONLY && frame_size > max_celt {
        return Err(OpusEncodeError::Unimplemented);
    }
    if mode == MODE_SILK_ONLY && frame_size != max_celt {
        return Err(OpusEncodeError::Unimplemented);
    }

    let mut bandwidth = select_bandwidth(encoder.user_bandwidth, encoder.max_bandwidth);

    match mode {
        MODE_SILK_ONLY => {
            encoder.configure_silk_control(frame_size_i32, max_payload_bytes);

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

            bandwidth = OpusEncoder::bandwidth_from_silk_control(&encoder.silk_mode);
            let toc = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;

            data[0] = toc;
            data[1..1 + payload.len()].copy_from_slice(&payload);
            encoder.bandwidth = bandwidth;
            encoder.range_final = range_final;
            encoder.mode = mode;

            Ok(1 + payload.len())
        }
        MODE_CELT_ONLY | MODE_HYBRID => {
            if mode == MODE_CELT_ONLY && bandwidth == Bandwidth::Medium {
                bandwidth = Bandwidth::Wide;
            }

            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetLsbDepth(encoder.lsb_depth),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetChannels(
                    usize::try_from(encoder.stream_channels)
                        .map_err(|_| OpusEncodeError::BadArgument)?,
                ),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetVbr(encoder.use_vbr),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetVbrConstraint(encoder.vbr_constraint),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetBitrate(encoder.bitrate_bps),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;

            let bytes = crate::celt::opus_custom_encode(
                encoder.celt.encoder(),
                &pcm[..required],
                frame_size,
                &mut data[1..],
                max_payload_bytes,
            )
            .map_err(|_| OpusEncodeError::InternalError)?;

            let mut celt_range = 0u32;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::GetFinalRange(&mut celt_range),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            let range_final = celt_range;

            let toc = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;
            data[0] = toc;
            encoder.bandwidth = bandwidth;
            encoder.range_final = range_final;
            encoder.mode = mode;

            Ok(1 + bytes)
        }
        _ => Err(OpusEncodeError::BadArgument),
    }
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
        MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND,
        OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_FRAMESIZE_20_MS, OPUS_SIGNAL_MUSIC,
        OpusEncodeError, OpusEncoderCtlError, OpusEncoderCtlRequest, OpusEncoderInitError,
        opus_encode, opus_encoder_create, opus_encoder_ctl, opus_encoder_get_size,
    };
    use crate::packet::{
        Bandwidth, opus_packet_get_bandwidth, opus_packet_get_mode,
        opus_packet_get_samples_per_frame,
    };

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
    fn encodes_celt_only_frame_with_valid_toc() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY)).unwrap();
        let pcm = [0i16; 480];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 480, &mut out).expect("encode");
        assert!(len >= 1);
        assert_eq!(opus_packet_get_mode(&out[..len]).unwrap(), crate::packet::Mode::CELT);
        assert_eq!(opus_packet_get_samples_per_frame(&out[..len], 48_000).unwrap(), 480);
    }

    #[test]
    fn encodes_hybrid_frame_with_valid_toc() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_HYBRID)).unwrap();
        let pcm = [0i16; 960];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 960, &mut out).expect("encode");
        assert!(len >= 1);
        assert_eq!(opus_packet_get_mode(&out[..len]).unwrap(), crate::packet::Mode::HYBRID);
        assert_eq!(opus_packet_get_samples_per_frame(&out[..len], 48_000).unwrap(), 960);
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
    fn ctl_round_trips_extended_settings() {
        let mut enc = opus_encoder_create(48_000, 2, 2048).expect("encoder");

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceChannels(1)).unwrap();
        let mut force_channels = 0;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetForceChannels(&mut force_channels),
        )
        .unwrap();
        assert_eq!(force_channels, 1);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND),
        )
        .unwrap();
        let mut max_bandwidth = 0;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetMaxBandwidth(&mut max_bandwidth),
        )
        .unwrap();
        assert_eq!(max_bandwidth, OPUS_BANDWIDTH_SUPERWIDEBAND);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_NARROWBAND),
        )
        .unwrap();

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
        )
        .unwrap();
        let mut signal = 0;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetSignal(&mut signal)).unwrap();
        assert_eq!(signal, OPUS_SIGNAL_MUSIC);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetLsbDepth(16)).unwrap();
        let mut lsb_depth = 0;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetLsbDepth(&mut lsb_depth)).unwrap();
        assert_eq!(lsb_depth, 16);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetExpertFrameDuration(OPUS_FRAMESIZE_20_MS),
        )
        .unwrap();
        let mut frame_duration = 0;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetExpertFrameDuration(&mut frame_duration),
        )
        .unwrap();
        assert_eq!(frame_duration, OPUS_FRAMESIZE_20_MS);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetPredictionDisabled(true)).unwrap();
        let mut prediction_disabled = false;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetPredictionDisabled(&mut prediction_disabled),
        )
        .unwrap();
        assert!(prediction_disabled);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetPhaseInversionDisabled(true)).unwrap();
        let mut phase_inversion_disabled = false;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetPhaseInversionDisabled(&mut phase_inversion_disabled),
        )
        .unwrap();
        assert!(phase_inversion_disabled);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_SILK_ONLY)).unwrap();
    }

    #[test]
    fn ctl_rejects_invalid_extended_settings() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");

        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceChannels(3)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetMaxBandwidth(999)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(42)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetLsbDepth(7)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetExpertFrameDuration(4242)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY + 1))
                .unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
    }

    #[test]
    fn encode_rejects_unsupported_frames() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        let pcm = [0i16; 479];
        let mut out = [0u8; 4000];
        let err = opus_encode(&mut enc, &pcm, 479, &mut out).unwrap_err();
        assert_eq!(err, OpusEncodeError::BadArgument);
    }
}
