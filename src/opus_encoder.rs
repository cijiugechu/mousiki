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
    tonality_get_info,
};
use crate::celt::AnalysisInfo;
use crate::celt::{
    CELT_SIG_SCALE, CeltEncodeError, CeltEncoderCtlError, CeltEncoderInitError, SilkInfo,
    EncoderCtlRequest as CeltEncoderCtlRequest, OpusRes, OwnedCeltEncoder, canonical_mode,
    celt_encode_with_ec, celt_sqrt, convert_i16_to_celt_sig, frac_div32,
    opus_custom_encoder_create, opus_custom_encoder_ctl, opus_select_arch,
};
use crate::opus_multistream::{OPUS_AUTO, OPUS_BITRATE_MAX};
use crate::packet::Bandwidth;
use crate::range::RangeEncoder;
use crate::repacketizer::{opus_packet_pad, OpusRepacketizer, RepacketizerError};
use crate::silk::enc_api::{PrefillMode, silk_encode, silk_init_encoder};
use crate::silk::errors::SilkError;
use crate::silk::EncControl as SilkEncControl;
use crate::silk::lin2log::lin2log;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;

const MAX_CHANNELS: usize = 2;
const MAX_ENCODER_BUFFER: usize = 480;
const DELAY_BUFFER_SAMPLES: usize = MAX_ENCODER_BUFFER * 2;
const MAX_PACKET_BYTES: i32 = 1276 * 6;
const MAX_REPACKETIZER_BYTES: usize = MAX_PACKET_BYTES as usize + 6;

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

const DRED_MAX_FRAMES: i32 = 104;
const FEC_THRESHOLDS: [i32; 10] = [
    12_000, 1_000, 14_000, 1_000, 16_000, 1_000, 20_000, 1_000, 22_000, 1_000,
];
const FEC_RATE_SCALE_Q16: i32 = 655;
#[allow(dead_code)]
const SILK_RATE_TABLE: [[i32; 5]; 7] = [
    [0, 0, 0, 0, 0],
    [12_000, 10_000, 10_000, 11_000, 11_000],
    [16_000, 13_500, 13_500, 15_000, 15_000],
    [20_000, 16_000, 16_000, 18_000, 18_000],
    [24_000, 18_000, 18_000, 21_000, 21_000],
    [32_000, 22_000, 22_000, 28_000, 28_000],
    [64_000, 38_000, 38_000, 50_000, 50_000],
];
const MONO_VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9_000, 700, 9_000, 700, 13_500, 1_000, 14_000, 2_000,
];
const MONO_MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9_000, 700, 9_000, 700, 11_000, 1_000, 12_000, 2_000,
];
const STEREO_VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9_000, 700, 9_000, 700, 13_500, 1_000, 14_000, 2_000,
];
const STEREO_MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9_000, 700, 9_000, 700, 11_000, 1_000, 12_000, 2_000,
];
const STEREO_VOICE_THRESHOLD: i32 = 19_000;
const STEREO_MUSIC_THRESHOLD: i32 = 17_000;
const MODE_THRESHOLDS: [[i32; 2]; 2] = [[64_000, 10_000], [44_000, 10_000]];
const Q15_ONE: i32 = 1 << 15;

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
    SetDredDuration(i32),
    GetDredDuration(&'req mut i32),
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
#[derive(Clone, Copy, Debug, Default)]
struct StereoWidthState {
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
    width_mem: StereoWidthState,
    delay_buffer: [OpusRes; DELAY_BUFFER_SAMPLES],
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
    fec_config: i32,
    force_channels: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    signal_type: i32,
    user_forced_mode: i32,
    voice_ratio: i32,
    lsb_depth: i32,
    variable_duration: i32,
    prediction_disabled: bool,
    hybrid_stereo_width_q14: i16,
    variable_hp_smth2_q15: i32,
    prev_hb_gain: f32,
    hp_mem: [f32; 4],
    mode: i32,
    prev_mode: i32,
    prev_channels: i32,
    prev_framesize: i32,
    bandwidth: Bandwidth,
    auto_bandwidth: i32,
    silk_bw_switch: bool,
    first: bool,
    width_mem: StereoWidthState,
    delay_buffer: [OpusRes; DELAY_BUFFER_SAMPLES],
    #[cfg(not(feature = "fixed_point"))]
    detected_bandwidth: i32,
    nonfinal_frame: bool,
    range_final: u32,
    dred_duration: i32,
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
        self.fec_config = 0;
        self.force_channels = OPUS_AUTO;
        self.user_bandwidth = OPUS_AUTO;
        self.max_bandwidth = OPUS_BANDWIDTH_FULLBAND;
        self.signal_type = OPUS_AUTO;
        self.user_forced_mode = OPUS_AUTO;
        self.voice_ratio = -1;
        self.lsb_depth = 24;
        self.variable_duration = OPUS_FRAMESIZE_ARG;
        self.prediction_disabled = false;

        self.hybrid_stereo_width_q14 = 1_i16 << 14;
        self.variable_hp_smth2_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        self.prev_hb_gain = 1.0;
        self.hp_mem = [0.0; 4];
        self.mode = MODE_HYBRID;
        self.prev_mode = 0;
        self.prev_channels = 0;
        self.prev_framesize = 0;
        self.bandwidth = Bandwidth::Full;
        self.auto_bandwidth = 0;
        self.silk_bw_switch = false;
        self.first = true;
        self.width_mem = StereoWidthState::default();
        self.delay_buffer = [OpusRes::default(); DELAY_BUFFER_SAMPLES];
        #[cfg(not(feature = "fixed_point"))]
        {
            self.detected_bandwidth = 0;
        }
        self.nonfinal_frame = false;
        self.range_final = 0;
        self.dred_duration = 0;

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
        self.dred_duration = 0;
        self.silk_mode.use_dred = 0;
        self.stream_channels = self.channels;
        self.hybrid_stereo_width_q14 = 1_i16 << 14;
        self.prev_hb_gain = 1.0;
        self.first = true;
        self.mode = MODE_HYBRID;
        self.bandwidth = Bandwidth::Full;
        self.variable_hp_smth2_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        self.hp_mem = [0.0; 4];
        self.prev_mode = 0;
        self.prev_channels = 0;
        self.prev_framesize = 0;
        self.auto_bandwidth = 0;
        self.silk_bw_switch = false;
        self.width_mem = StereoWidthState::default();
        self.delay_buffer = [OpusRes::default(); DELAY_BUFFER_SAMPLES];
        #[cfg(not(feature = "fixed_point"))]
        {
            self.detected_bandwidth = 0;
        }
        self.nonfinal_frame = false;
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

        let max_internal =
            max_internal_sample_rate_for_bandwidth(self.user_bandwidth, self.max_bandwidth);
        self.silk_mode.max_internal_sample_rate = max_internal;
        self.silk_mode.desired_internal_sample_rate = max_internal;

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

fn user_bitrate_to_bitrate(encoder: &OpusEncoder<'_>, mut frame_size: i32, max_data_bytes: i32) -> i32 {
    if frame_size == 0 {
        frame_size = encoder.fs / 400;
    }
    if encoder.user_bitrate_bps == OPUS_AUTO {
        let base = i64::from(60 * encoder.fs / frame_size);
        let channels = i64::from(encoder.channels * encoder.fs);
        (base + channels).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    } else if encoder.user_bitrate_bps == OPUS_BITRATE_MAX {
        let bitrate =
            i64::from(max_data_bytes) * 8 * i64::from(encoder.fs) / i64::from(frame_size);
        bitrate.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    } else {
        encoder.user_bitrate_bps
    }
}

fn compute_stereo_width(
    pcm: &[i16],
    frame_size: usize,
    fs: i32,
    mem: &mut StereoWidthState,
) -> f32 {
    let frame_rate = fs / frame_size as i32;
    if frame_rate <= 0 {
        return 0.0;
    }
    let short_alpha = 1.0 - 25.0 / (frame_rate.max(50) as f32);
    let mut xx = 0.0f32;
    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    let scale = 1.0 / CELT_SIG_SCALE;

    for i in (0..frame_size.saturating_sub(3)).step_by(4) {
        let mut pxx = 0.0f32;
        let mut pxy = 0.0f32;
        let mut pyy = 0.0f32;

        let x0 = f32::from(pcm[2 * i]) * scale;
        let y0 = f32::from(pcm[2 * i + 1]) * scale;
        pxx += x0 * x0;
        pxy += x0 * y0;
        pyy += y0 * y0;

        let x1 = f32::from(pcm[2 * i + 2]) * scale;
        let y1 = f32::from(pcm[2 * i + 3]) * scale;
        pxx += x1 * x1;
        pxy += x1 * y1;
        pyy += y1 * y1;

        let x2 = f32::from(pcm[2 * i + 4]) * scale;
        let y2 = f32::from(pcm[2 * i + 5]) * scale;
        pxx += x2 * x2;
        pxy += x2 * y2;
        pyy += y2 * y2;

        let x3 = f32::from(pcm[2 * i + 6]) * scale;
        let y3 = f32::from(pcm[2 * i + 7]) * scale;
        pxx += x3 * x3;
        pxy += x3 * y3;
        pyy += y3 * y3;

        xx += pxx;
        xy += pxy;
        yy += pyy;
    }

    if xx >= 1.0e9 || xx.is_nan() || yy >= 1.0e9 || yy.is_nan() {
        xx = 0.0;
        xy = 0.0;
        yy = 0.0;
    }

    mem.xx += short_alpha * (xx - mem.xx);
    mem.xy += short_alpha * (xy - mem.xy);
    mem.yy += short_alpha * (yy - mem.yy);
    mem.xx = mem.xx.max(0.0);
    mem.xy = mem.xy.max(0.0);
    mem.yy = mem.yy.max(0.0);

    const WIDTH_THRESHOLD: f32 = 8.0e-4;
    const EPSILON: f32 = 1.0e-15;
    if mem.xx.max(mem.yy) > WIDTH_THRESHOLD {
        let sqrt_xx = celt_sqrt(mem.xx);
        let sqrt_yy = celt_sqrt(mem.yy);
        let qrrt_xx = celt_sqrt(sqrt_xx);
        let qrrt_yy = celt_sqrt(sqrt_yy);
        mem.xy = mem.xy.min(sqrt_xx * sqrt_yy);
        let corr = frac_div32(mem.xy, EPSILON + sqrt_xx * sqrt_yy);
        let ldiff = (qrrt_xx - qrrt_yy).abs() / (EPSILON + qrrt_xx + qrrt_yy);
        let width = celt_sqrt((1.0 - corr * corr).max(0.0)).min(1.0) * ldiff;
        mem.smoothed_width += (width - mem.smoothed_width) / frame_rate as f32;
        mem.max_follower =
            (mem.max_follower - 0.02 / frame_rate as f32).max(mem.smoothed_width);
    }

    (20.0 * mem.max_follower).min(1.0)
}

fn decide_fec(
    use_in_band_fec: bool,
    packet_loss_perc: i32,
    last_fec: bool,
    mode: i32,
    bandwidth: &mut i32,
    rate: i32,
) -> bool {
    if !use_in_band_fec || packet_loss_perc == 0 || mode == MODE_CELT_ONLY {
        return false;
    }

    let orig_bandwidth = *bandwidth;
    loop {
        let idx = usize::try_from(2 * (*bandwidth - OPUS_BANDWIDTH_NARROWBAND)).unwrap_or(0);
        let mut lbrr_rate_thres_bps = *FEC_THRESHOLDS.get(idx).unwrap_or(&0);
        let hysteresis = *FEC_THRESHOLDS.get(idx + 1).unwrap_or(&0);
        if last_fec {
            lbrr_rate_thres_bps -= hysteresis;
        }
        if !last_fec {
            lbrr_rate_thres_bps += hysteresis;
        }

        let loss_scale = 125 - packet_loss_perc.min(25);
        let scaled = (i64::from(lbrr_rate_thres_bps) * i64::from(loss_scale)
            * i64::from(FEC_RATE_SCALE_Q16))
            >> 16;
        lbrr_rate_thres_bps = scaled.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;

        if rate > lbrr_rate_thres_bps {
            return true;
        }
        if packet_loss_perc <= 5 {
            return false;
        }
        if *bandwidth > OPUS_BANDWIDTH_NARROWBAND {
            *bandwidth -= 1;
        } else {
            break;
        }
    }

    *bandwidth = orig_bandwidth;
    false
}

#[allow(dead_code)]
fn compute_silk_rate_for_hybrid(
    mut rate: i32,
    bandwidth: i32,
    frame_20ms: bool,
    vbr: bool,
    fec: bool,
    channels: i32,
) -> i32 {
    rate /= channels;
    let entry = 1 + i32::from(frame_20ms) + 2 * i32::from(fec);

    let mut idx = 1;
    while idx < SILK_RATE_TABLE.len() && SILK_RATE_TABLE[idx][0] <= rate {
        idx += 1;
    }

    let mut silk_rate = if idx == SILK_RATE_TABLE.len() {
        let base = SILK_RATE_TABLE[idx - 1][entry as usize];
        base + (rate - SILK_RATE_TABLE[idx - 1][0]) / 2
    } else {
        let lo = SILK_RATE_TABLE[idx - 1][entry as usize];
        let hi = SILK_RATE_TABLE[idx][entry as usize];
        let x0 = SILK_RATE_TABLE[idx - 1][0];
        let x1 = SILK_RATE_TABLE[idx][0];
        let num = i64::from(lo) * i64::from(x1 - rate) + i64::from(hi) * i64::from(rate - x0);
        (num / i64::from(x1 - x0)) as i32
    };

    if !vbr {
        silk_rate += 100;
    }
    if bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND {
        silk_rate += 300;
    }
    silk_rate *= channels;
    if channels == 2 && rate >= 12_000 {
        silk_rate -= 1_000;
    }
    silk_rate
}

fn compute_equiv_rate(
    bitrate: i32,
    channels: i32,
    frame_rate: i32,
    vbr: bool,
    mode: i32,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = i64::from(bitrate);
    if frame_rate > 50 {
        equiv -= i64::from((40 * channels + 20) * (frame_rate - 50));
    }
    if !vbr {
        equiv -= equiv / 12;
    }
    equiv = equiv * i64::from(90 + complexity) / 100;
    if mode == MODE_SILK_ONLY || mode == MODE_HYBRID {
        if complexity < 2 {
            equiv = equiv * 4 / 5;
        }
        equiv -= equiv * i64::from(loss) / i64::from(6 * loss + 10);
    } else if mode == MODE_CELT_ONLY {
        if complexity < 5 {
            equiv = equiv * 9 / 10;
        }
    } else {
        equiv -= equiv * i64::from(loss) / i64::from(12 * loss + 20);
    }
    equiv.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[cfg(not(feature = "fixed_point"))]
fn is_digital_silence(pcm: &[i16], frame_size: usize, channels: usize, lsb_depth: i32) -> bool {
    let total = frame_size.saturating_mul(channels);
    if pcm.len() < total || lsb_depth <= 0 {
        return false;
    }
    let mut sample_max = 0i32;
    for &sample in &pcm[..total] {
        sample_max = sample_max.max(i32::from(sample).abs());
    }
    if lsb_depth >= 15 {
        sample_max == 0
    } else {
        let threshold = 1i32 << (15 - lsb_depth);
        sample_max <= threshold
    }
}

#[allow(dead_code)]
fn compute_redundancy_bytes(
    max_data_bytes: i32,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: i32,
) -> i32 {
    if frame_rate <= 0 {
        return 0;
    }
    let base_bits = i64::from(40 * channels + 20);
    let mut redundancy_rate =
        i64::from(bitrate_bps) + base_bits * i64::from(200 - frame_rate);
    redundancy_rate = 3 * redundancy_rate / 2;
    let mut redundancy_bytes = redundancy_rate / 1_600;

    let available_bits = i64::from(max_data_bytes) * 8 - 2 * base_bits;
    let denom = i64::from(240 + 48_000 / frame_rate);
    let redundancy_bytes_cap = (available_bits * 240 / denom + base_bits) / 8;
    redundancy_bytes = redundancy_bytes.min(redundancy_bytes_cap);

    if redundancy_bytes > i64::from(4 + 8 * channels) {
        redundancy_bytes = redundancy_bytes.min(257);
    } else {
        redundancy_bytes = 0;
    }

    redundancy_bytes.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn finish_encode(encoder: &mut OpusEncoder<'_>, mode: i32, to_celt: bool, frame_size: i32) {
    encoder.mode = mode;
    encoder.prev_mode = if to_celt { MODE_CELT_ONLY } else { mode };
    encoder.prev_channels = encoder.stream_channels;
    encoder.prev_framesize = frame_size;
    encoder.first = false;
}

fn encode_frame_native<'mode>(
    encoder: &mut OpusEncoder<'mode>,
    pcm: &[i16],
    frame_size: usize,
    data: &mut [u8],
    lsb_depth: i32,
    silk_use_dtx: bool,
    is_silence: bool,
    redundancy: bool,
    celt_to_silk: bool,
    prefill: PrefillMode,
    bandwidth_int: i32,
    mode: i32,
    to_celt: bool,
) -> Result<usize, OpusEncodeError> {
    if data.len() < 2 {
        return Err(OpusEncodeError::BufferTooSmall);
    }

    let channels = usize::try_from(encoder.channels).map_err(|_| OpusEncodeError::BadArgument)?;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusEncodeError::BadArgument);
    }

    let frame_size_i32 = i32::try_from(frame_size).map_err(|_| OpusEncodeError::BadArgument)?;
    let frame_rate = encoder
        .fs
        .checked_div(frame_size_i32)
        .ok_or(OpusEncodeError::BadArgument)?;
    let max_data_bytes = i32::try_from(data.len()).map_err(|_| OpusEncodeError::BadArgument)?;

    #[cfg(feature = "fixed_point")]
    let _ = is_silence;

    let mut bandwidth = Bandwidth::from_opus_int(bandwidth_int).unwrap_or(Bandwidth::Wide);

    let max_frame_bytes_i32 = max_data_bytes.min(1276);
    let max_frame_bytes =
        usize::try_from(max_frame_bytes_i32).map_err(|_| OpusEncodeError::BadArgument)?;
    let max_payload_bytes = max_frame_bytes.saturating_sub(1);

    match mode {
        MODE_SILK_ONLY => {
            encoder.configure_silk_control(frame_size_i32, max_payload_bytes);
            encoder.silk_mode.use_dtx = i32::from(silk_use_dtx);
            encoder.silk_mode.desired_internal_sample_rate = match bandwidth_int {
                OPUS_BANDWIDTH_NARROWBAND => 8_000,
                OPUS_BANDWIDTH_MEDIUMBAND => 12_000,
                _ => 16_000,
            };
            encoder.silk_mode.min_internal_sample_rate = 8_000;

            #[cfg(feature = "fixed_point")]
            let activity = 1i32;
            #[cfg(not(feature = "fixed_point"))]
            let activity = {
                if is_silence {
                    0
                } else if encoder.analysis_info.valid && encoder.analysis_info.activity < 0.02 {
                    0
                } else {
                    1
                }
            };

            if encoder.silk.state_fxx[0].resampler_state.fs_in_khz() == 0
                || encoder.silk.state_fxx[0].resampler_state.fs_out_khz() == 0
            {
                encoder
                    .silk
                    .state_fxx[0]
                    .resampler_state
                    .silk_resampler_init(
                        encoder.silk_mode.api_sample_rate,
                        encoder.silk_mode.desired_internal_sample_rate,
                        true,
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                if encoder.silk_mode.n_channels_internal == 2 {
                    encoder.silk.state_fxx[1].resampler_state =
                        encoder.silk.state_fxx[0].resampler_state.clone();
                }
            }

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
            let bytes_out =
                usize::try_from(bytes_out).map_err(|_| OpusEncodeError::InternalError)?;

            bandwidth = OpusEncoder::bandwidth_from_silk_control(&encoder.silk_mode);
            let toc = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;

            data[0] = toc;
            if bytes_out == 0 {
                encoder.bandwidth = bandwidth;
                encoder.range_final = 0;
                finish_encode(encoder, mode, to_celt, frame_size_i32);
                return Ok(1);
            }

            let payload = range_encoder.finish();
            let payload_len = payload.len();
            if payload_len > max_payload_bytes {
                return Err(OpusEncodeError::BufferTooSmall);
            }

            data[1..1 + payload_len].copy_from_slice(&payload);
            encoder.bandwidth = bandwidth;
            encoder.range_final = range_final;
            finish_encode(encoder, mode, to_celt, frame_size_i32);

            Ok(1 + payload_len)
        }
        MODE_CELT_ONLY => {
            if bandwidth == Bandwidth::Medium {
                bandwidth = Bandwidth::Wide;
            }

            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetLsbDepth(lsb_depth),
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

            let celt_frame_size = frame_size;
            let celt_pcm: &[i16] = &pcm[..required];

            let mut bytes = crate::celt::opus_custom_encode(
                encoder.celt.encoder(),
                celt_pcm,
                celt_frame_size,
                &mut data[1..],
                max_payload_bytes,
            )
            .map_err(|err| match err {
                CeltEncodeError::MissingOutput => OpusEncodeError::BufferTooSmall,
                _ => OpusEncodeError::InternalError,
            })?;
            if bytes == 0 {
                if max_payload_bytes == 0 {
                    return Err(OpusEncodeError::BufferTooSmall);
                }
                data[1] = 0;
                bytes = 1;
            }

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
            finish_encode(encoder, mode, to_celt, frame_size_i32);

            Ok(1 + bytes)
        }
        MODE_HYBRID => {
            let mut redundancy = redundancy;
            let mut celt_to_silk = celt_to_silk;
            let mut prefill = prefill;
            let mut redundancy_bytes = 0i32;
            let mut redundant_rng = 0u32;

            if encoder.silk_bw_switch {
                redundancy = true;
                celt_to_silk = true;
                encoder.silk_bw_switch = false;
                prefill = PrefillMode::PrefillWithState;
            }

            if redundancy {
                redundancy_bytes = compute_redundancy_bytes(
                    max_frame_bytes_i32,
                    encoder.bitrate_bps,
                    frame_rate,
                    encoder.stream_channels,
                );
                if redundancy_bytes == 0 {
                    redundancy = false;
                }
            }

            let bytes_target = (max_frame_bytes_i32 - redundancy_bytes)
                .min(encoder.bitrate_bps * frame_size_i32 / (encoder.fs * 8))
                - 1;
            let bytes_target = bytes_target.max(0);

            encoder.silk_mode.payload_size_ms =
                (i64::from(frame_size_i32) * 1000 / i64::from(encoder.fs)) as i32;
            encoder.silk_mode.n_channels_api = encoder.channels;
            encoder.silk_mode.n_channels_internal = encoder.stream_channels;
            encoder.silk_mode.api_sample_rate = encoder.fs;
            encoder.silk_mode.packet_loss_percentage = encoder.packet_loss_perc;
            encoder.silk_mode.complexity = encoder.complexity;
            encoder.silk_mode.use_in_band_fec = i32::from(encoder.inband_fec);
            encoder.silk_mode.use_dtx = i32::from(silk_use_dtx);
            encoder.silk_mode.use_cbr = i32::from(!encoder.use_vbr);
            encoder.silk_mode.reduced_dependency = encoder.prediction_disabled;
            encoder.silk_mode.opus_can_switch = false;

            encoder.silk_mode.desired_internal_sample_rate = match bandwidth_int {
                OPUS_BANDWIDTH_NARROWBAND => 8_000,
                OPUS_BANDWIDTH_MEDIUMBAND => 12_000,
                _ => 16_000,
            };
            debug_assert!(
                mode == MODE_HYBRID || bandwidth_int == OPUS_BANDWIDTH_WIDEBAND,
                "unexpected SILK internal bandwidth selection",
            );
            encoder.silk_mode.min_internal_sample_rate = 16_000;
            encoder.silk_mode.max_internal_sample_rate = 16_000;

            encoder.silk_mode.max_bits = (max_frame_bytes_i32 - 1).saturating_mul(8);
            if redundancy && redundancy_bytes >= 2 {
                encoder.silk_mode.max_bits = encoder
                    .silk_mode
                    .max_bits
                    .saturating_sub(redundancy_bytes * 8 + 1);
                encoder.silk_mode.max_bits = encoder.silk_mode.max_bits.saturating_sub(20);
            }

            let frame_20ms = frame_size_i32 * 50 == encoder.fs;
            let total_bitrate = (i64::from(bytes_target) * 8 * i64::from(frame_rate))
                .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            encoder.silk_mode.bit_rate = compute_silk_rate_for_hybrid(
                total_bitrate,
                bandwidth_int,
                frame_20ms,
                encoder.use_vbr,
                encoder.silk_mode.lbrr_coded != 0,
                encoder.stream_channels,
            )
            .clamp(5_000, 80_000);

            if encoder.silk_mode.use_cbr != 0 {
                let other_bits = (encoder.silk_mode.max_bits
                    - encoder.silk_mode.bit_rate * frame_size_i32 / encoder.fs)
                    .max(0);
                encoder.silk_mode.max_bits =
                    (encoder.silk_mode.max_bits - other_bits * 3 / 4).max(0);
                encoder.silk_mode.use_cbr = 0;
            } else {
                let max_bit_rate = compute_silk_rate_for_hybrid(
                    encoder.silk_mode.max_bits * encoder.fs / frame_size_i32,
                    bandwidth_int,
                    frame_20ms,
                    encoder.use_vbr,
                    encoder.silk_mode.lbrr_coded != 0,
                    encoder.stream_channels,
                );
                encoder.silk_mode.max_bits = max_bit_rate * frame_size_i32 / encoder.fs;
            }

            #[cfg(feature = "fixed_point")]
            let activity = 1i32;
            #[cfg(not(feature = "fixed_point"))]
            let activity = {
                if is_silence {
                    0
                } else if encoder.analysis_info.valid && encoder.analysis_info.activity < 0.02 {
                    0
                } else {
                    1
                }
            };

            if !matches!(prefill, PrefillMode::None) {
                let prefill_samples =
                    usize::try_from(encoder.fs / 100).map_err(|_| OpusEncodeError::BadArgument)?;
                let prefill_len = prefill_samples
                    .checked_mul(channels)
                    .ok_or(OpusEncodeError::BadArgument)?;
                if prefill_len <= required {
                    let mut prefill_encoder = RangeEncoder::new();
                    let mut prefill_bytes = 0i32;
                    silk_encode(
                        &mut encoder.silk,
                        &mut encoder.silk_mode,
                        &pcm[..prefill_len],
                        &mut prefill_encoder,
                        &mut prefill_bytes,
                        prefill,
                        activity,
                    )?;
                    encoder.silk_mode.opus_can_switch = false;
                }
            }

            let mut range_encoder = RangeEncoder::with_capacity(max_payload_bytes);
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

            if bytes_out == 0 {
                let toc = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;
                data[0] = toc;
                encoder.bandwidth = bandwidth;
                encoder.range_final = 0;
                finish_encode(encoder, mode, to_celt, frame_size_i32);
                return Ok(1);
            }

            debug_assert!(
                encoder.silk_mode.internal_sample_rate == 16_000,
                "hybrid SILK internal sample rate must remain at 16 kHz",
            );

            encoder.silk_mode.opus_can_switch =
                encoder.silk_mode.switch_ready && !encoder.nonfinal_frame;
            if encoder.silk_mode.opus_can_switch {
                redundancy_bytes = compute_redundancy_bytes(
                    max_frame_bytes_i32,
                    encoder.bitrate_bps,
                    frame_rate,
                    encoder.stream_channels,
                );
                redundancy = redundancy_bytes != 0;
                celt_to_silk = false;
                encoder.silk_bw_switch = true;
            }

            if range_encoder.tell() + 17 + 20 <= 8 * (max_frame_bytes_i32 - 1) {
                range_encoder.encode_bit_logp(i32::from(redundancy), 12);
                if redundancy {
                    range_encoder.encode_bit_logp(i32::from(celt_to_silk), 1);
                    let max_redundancy = (max_frame_bytes_i32 - 1)
                        - ((range_encoder.tell() + 8 + 3 + 7) >> 3);
                    redundancy_bytes = redundancy_bytes.min(max_redundancy);
                    redundancy_bytes = redundancy_bytes.clamp(2, 257);
                    range_encoder.encode_uint((redundancy_bytes - 2) as u32, 256);
                }
            } else {
                redundancy = false;
            }

            if !redundancy {
                encoder.silk_bw_switch = false;
                redundancy_bytes = 0;
            }

            let nb_compr_bytes = usize::try_from(
                (max_frame_bytes_i32 - 1 - redundancy_bytes).max(0),
            )
            .map_err(|_| OpusEncodeError::BadArgument)?;
            if nb_compr_bytes < 2 {
                return Err(OpusEncodeError::BufferTooSmall);
            }
            range_encoder.shrink(nb_compr_bytes);

            let end_band = match bandwidth {
                Bandwidth::Narrow => 13,
                Bandwidth::Medium | Bandwidth::Wide => 17,
                Bandwidth::SuperWide => 19,
                Bandwidth::Full => 21,
            };
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetEndBand(end_band),
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
                CeltEncoderCtlRequest::SetLsbDepth(lsb_depth),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetBitrate(OPUS_BITRATE_MAX),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetPrediction(if encoder.silk_mode.reduced_dependency {
                    0
                } else {
                    2
                }),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;

            #[cfg(not(feature = "fixed_point"))]
            {
                encoder.celt.encoder().analysis = encoder.analysis_info.clone();
            }

            encoder.celt.encoder().silk_info = SilkInfo {
                signal_type: i32::from(encoder.silk_mode.signal_type),
                offset: encoder.silk_mode.offset,
            };

            let celt_pcm = convert_i16_to_celt_sig(&pcm[..required], required);

            let mut redundancy_buf = [0u8; 257];
            let mut redundancy_len = 0usize;
            if redundancy && celt_to_silk {
                let n2 = usize::try_from(encoder.fs / 200)
                    .map_err(|_| OpusEncodeError::BadArgument)?;
                let red_len = usize::try_from(redundancy_bytes)
                    .map_err(|_| OpusEncodeError::BadArgument)?;
                if n2 > 0 && red_len >= 2 {
                    debug_assert!(red_len <= redundancy_buf.len());
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetStartBand(0),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetVbr(false),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetBitrate(OPUS_BITRATE_MAX),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    let used = celt_encode_with_ec(
                        encoder.celt.encoder(),
                        Some(&celt_pcm[..n2 * channels]),
                        n2,
                        Some(&mut redundancy_buf[..red_len]),
                        None,
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    redundancy_len = used;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::GetFinalRange(&mut redundant_rng),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::ResetState,
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                }
            }

            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetStartBand(17),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetVbr(encoder.use_vbr),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            if encoder.use_vbr {
                let celt_rate = encoder.bitrate_bps - encoder.silk_mode.bit_rate;
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetBitrate(celt_rate),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetVbrConstraint(false),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
            }

            if encoder.prev_mode != mode && encoder.prev_mode > 0 {
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::ResetState,
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                let n4 = usize::try_from(encoder.fs / 400)
                    .map_err(|_| OpusEncodeError::BadArgument)?;
                if n4 > 0 && celt_pcm.len() >= n4 * channels {
                    let mut dummy = [0u8; 2];
                    let _ = celt_encode_with_ec(
                        encoder.celt.encoder(),
                        Some(&celt_pcm[..n4 * channels]),
                        n4,
                        Some(&mut dummy),
                        None,
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                }
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetPrediction(0),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
            }

            let mut enc_done = false;
            if range_encoder.tell() <= (nb_compr_bytes * 8) as i32 {
                let _ = celt_encode_with_ec(
                    encoder.celt.encoder(),
                    Some(&celt_pcm[..]),
                    frame_size,
                    None,
                    Some(range_encoder.encoder_mut()),
                )
                .map_err(|err| match err {
                    CeltEncodeError::MissingOutput => OpusEncodeError::BufferTooSmall,
                    _ => OpusEncodeError::InternalError,
                })?;
                enc_done = true;
            }

            if redundancy && !celt_to_silk {
                let n2 = usize::try_from(encoder.fs / 200)
                    .map_err(|_| OpusEncodeError::BadArgument)?;
                let n4 = usize::try_from(encoder.fs / 400)
                    .map_err(|_| OpusEncodeError::BadArgument)?;
                let red_len = usize::try_from(redundancy_bytes)
                    .map_err(|_| OpusEncodeError::BadArgument)?;
                if n2 > 0 && red_len >= 2 {
                    debug_assert!(red_len <= redundancy_buf.len());
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::ResetState,
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetStartBand(0),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetPrediction(0),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetVbr(false),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetBitrate(OPUS_BITRATE_MAX),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;

                    if n4 > 0 {
                        let prefill_start = frame_size
                            .saturating_sub(n2 + n4)
                            .saturating_mul(channels);
                        let prefill_end =
                            prefill_start.saturating_add(n4.saturating_mul(channels));
                        if prefill_end <= celt_pcm.len() {
                            let mut dummy = [0u8; 2];
                            let _ = celt_encode_with_ec(
                                encoder.celt.encoder(),
                                Some(&celt_pcm[prefill_start..prefill_end]),
                                n4,
                                Some(&mut dummy),
                                None,
                            )
                            .map_err(|_| OpusEncodeError::InternalError)?;
                        }
                    }

                    let red_start = frame_size
                        .saturating_sub(n2)
                        .saturating_mul(channels);
                    let red_end = red_start.saturating_add(n2.saturating_mul(channels));
                    if red_end <= celt_pcm.len() {
                        let used = celt_encode_with_ec(
                            encoder.celt.encoder(),
                            Some(&celt_pcm[red_start..red_end]),
                            n2,
                            Some(&mut redundancy_buf[..red_len]),
                            None,
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        redundancy_len = used;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::GetFinalRange(&mut redundant_rng),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                    }
                }
            }

            let range_final = range_encoder.range_final();
            let payload = if enc_done {
                range_encoder.finish_without_done()
            } else {
                range_encoder.finish()
            };
            let payload_len = payload.len();
            let total_len = payload_len + redundancy_len;
            if total_len > max_payload_bytes {
                return Err(OpusEncodeError::BufferTooSmall);
            }

            let toc = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;
            data[0] = toc;
            data[1..1 + payload_len].copy_from_slice(&payload);
            if redundancy_len != 0 {
                data[1 + payload_len..1 + total_len]
                    .copy_from_slice(&redundancy_buf[..redundancy_len]);
            }

            encoder.bandwidth = bandwidth;
            encoder.range_final = range_final ^ redundant_rng;
            finish_encode(encoder, mode, to_celt, frame_size_i32);

            Ok(1 + total_len)
        }
        _ => Err(OpusEncodeError::BadArgument),
    }
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
        fec_config: 0,
        force_channels: OPUS_AUTO,
        user_bandwidth: OPUS_AUTO,
        max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
        signal_type: OPUS_AUTO,
        user_forced_mode: OPUS_AUTO,
        voice_ratio: 0,
        lsb_depth: 24,
        variable_duration: OPUS_FRAMESIZE_ARG,
        prediction_disabled: false,
        hybrid_stereo_width_q14: 0,
        variable_hp_smth2_q15: 0,
        prev_hb_gain: 0.0,
        hp_mem: [0.0; 4],
        mode: MODE_SILK_ONLY,
        prev_mode: 0,
        prev_channels: 0,
        prev_framesize: 0,
        bandwidth: Bandwidth::Wide,
        auto_bandwidth: 0,
        silk_bw_switch: false,
        first: false,
        width_mem: StereoWidthState::default(),
        delay_buffer: [OpusRes::default(); DELAY_BUFFER_SAMPLES],
        #[cfg(not(feature = "fixed_point"))]
        detected_bandwidth: 0,
        nonfinal_frame: false,
        range_final: 0,
        dred_duration: 0,
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
        OpusEncoderCtlRequest::SetDredDuration(value) => {
            if !cfg!(feature = "dred") {
                return Err(OpusEncoderCtlError::Unimplemented);
            }
            if !(0..=DRED_MAX_FRAMES).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.dred_duration = value;
            encoder.silk_mode.use_dred = if value > 0 { 1 } else { 0 };
        }
        OpusEncoderCtlRequest::GetDredDuration(out) => {
            if !cfg!(feature = "dred") {
                return Err(OpusEncoderCtlError::Unimplemented);
            }
            *out = encoder.dred_duration;
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

    let mut max_data_bytes = i32::try_from(data.len()).map_err(|_| OpusEncodeError::BadArgument)?;
    max_data_bytes = max_data_bytes.min(MAX_PACKET_BYTES);
    encoder.bitrate_bps = user_bitrate_to_bitrate(encoder, frame_size_i32, max_data_bytes);

    let lsb_depth = encoder.lsb_depth.min(16);

    let mut stereo_width = 0.0f32;
    if encoder.channels == 2 && encoder.force_channels != 1 {
        stereo_width = compute_stereo_width(
            &pcm[..required],
            frame_size,
            encoder.fs,
            &mut encoder.width_mem,
        );
    }
    let stereo_width_q15 =
        libm::roundf(stereo_width * Q15_ONE as f32).clamp(0.0, Q15_ONE as f32) as i32;

    #[cfg(not(feature = "fixed_point"))]
    let mut is_silence = false;
    #[cfg(feature = "fixed_point")]
    let is_silence = false;
    #[cfg(not(feature = "fixed_point"))]
    let mut analysis_read_state = None;
    #[cfg(not(feature = "fixed_point"))]
    {
        encoder.analysis_info.valid = false;
        if encoder.silk_mode.complexity >= 7 && encoder.fs >= 16_000 {
            is_silence = is_digital_silence(&pcm[..required], frame_size, channels, lsb_depth);
            analysis_read_state = Some(encoder.analysis.snapshot_read_state());
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
                lsb_depth,
                &mut encoder.analysis_info,
            );
        } else {
            tonality_analysis_reset(&mut encoder.analysis);
        }

        if !is_silence {
            encoder.voice_ratio = -1;
        }
        encoder.detected_bandwidth = 0;
        if encoder.analysis_info.valid {
            if encoder.signal_type == OPUS_AUTO {
                let prob = if encoder.prev_mode == 0 {
                    encoder.analysis_info.music_prob
                } else if encoder.prev_mode == MODE_CELT_ONLY {
                    encoder.analysis_info.music_prob_max
                } else {
                    encoder.analysis_info.music_prob_min
                };
                encoder.voice_ratio = libm::floorf(0.5 + 100.0 * (1.0 - prob)) as i32;
            }

            let analysis_bandwidth = encoder.analysis_info.bandwidth;
            encoder.detected_bandwidth = if analysis_bandwidth <= 12 {
                OPUS_BANDWIDTH_NARROWBAND
            } else if analysis_bandwidth <= 14 {
                OPUS_BANDWIDTH_MEDIUMBAND
            } else if analysis_bandwidth <= 16 {
                OPUS_BANDWIDTH_WIDEBAND
            } else if analysis_bandwidth <= 18 {
                OPUS_BANDWIDTH_SUPERWIDEBAND
            } else {
                OPUS_BANDWIDTH_FULLBAND
            };
        }
    }
    #[cfg(feature = "fixed_point")]
    {
        encoder.voice_ratio = -1;
    }

    let mut equiv_rate = compute_equiv_rate(
        encoder.bitrate_bps,
        encoder.channels,
        frame_rate,
        encoder.use_vbr,
        0,
        encoder.complexity,
        encoder.packet_loss_perc,
    );

    let voice_est = if encoder.signal_type == OPUS_SIGNAL_VOICE {
        127
    } else if encoder.signal_type == OPUS_SIGNAL_MUSIC {
        0
    } else if encoder.voice_ratio >= 0 {
        let mut est = (encoder.voice_ratio * 327) >> 8;
        if matches!(encoder.application, OpusApplication::Audio) {
            est = est.min(115);
        }
        est
    } else if matches!(encoder.application, OpusApplication::Voip) {
        115
    } else {
        48
    };

    let prev_stream_channels = encoder.stream_channels;
    let stream_channels = if encoder.force_channels != OPUS_AUTO && encoder.channels == 2 {
        encoder.force_channels
    } else if encoder.channels == 2 {
        let mut stereo_threshold = STEREO_MUSIC_THRESHOLD
            + ((i64::from(voice_est) * i64::from(voice_est)
                * i64::from(STEREO_VOICE_THRESHOLD - STEREO_MUSIC_THRESHOLD))
                >> 14) as i32;
        if prev_stream_channels == 2 {
            stereo_threshold -= 1000;
        } else {
            stereo_threshold += 1000;
        }
        if equiv_rate > stereo_threshold { 2 } else { 1 }
    } else {
        encoder.channels
    };
    encoder.stream_channels = stream_channels;

    equiv_rate = compute_equiv_rate(
        encoder.bitrate_bps,
        encoder.stream_channels,
        frame_rate,
        encoder.use_vbr,
        0,
        encoder.complexity,
        encoder.packet_loss_perc,
    );

    #[cfg(not(feature = "fixed_point"))]
    let silk_use_dtx = encoder.use_dtx && !(encoder.analysis_info.valid || is_silence);
    #[cfg(feature = "fixed_point")]
    let silk_use_dtx = encoder.use_dtx;
    encoder.silk_mode.use_dtx = i32::from(silk_use_dtx);

    let mut mode = if matches!(encoder.application, OpusApplication::RestrictedLowDelay) {
        MODE_CELT_ONLY
    } else if encoder.user_forced_mode != OPUS_AUTO {
        encoder.user_forced_mode
    } else {
        let q15_one_minus = Q15_ONE - stereo_width_q15;
        let mode_voice = ((i64::from(q15_one_minus) * i64::from(MODE_THRESHOLDS[0][0])
            + i64::from(stereo_width_q15) * i64::from(MODE_THRESHOLDS[1][0]))
            >> 15) as i32;
        let mode_music = ((i64::from(q15_one_minus) * i64::from(MODE_THRESHOLDS[1][1])
            + i64::from(stereo_width_q15) * i64::from(MODE_THRESHOLDS[1][1]))
            >> 15) as i32;

        let mut threshold = mode_music
            + ((i64::from(voice_est)
                * i64::from(voice_est)
                * i64::from(mode_voice - mode_music))
                >> 14) as i32;
        if matches!(encoder.application, OpusApplication::Voip) {
            threshold += 8000;
        }
        if encoder.prev_mode == MODE_CELT_ONLY {
            threshold -= 4000;
        } else if encoder.prev_mode > 0 {
            threshold += 4000;
        }

        let mut selected = if equiv_rate >= threshold {
            MODE_CELT_ONLY
        } else {
            MODE_SILK_ONLY
        };

        if encoder.inband_fec
            && encoder.packet_loss_perc > (128 - voice_est) >> 4
            && (encoder.fec_config != 2 || voice_est > 25)
        {
            selected = MODE_SILK_ONLY;
        }
        if silk_use_dtx && voice_est > 100 {
            selected = MODE_SILK_ONLY;
        }
        let rate_threshold = if frame_rate > 50 { 9000 } else { 6000 };
        let threshold_bytes =
            i64::from(rate_threshold) * i64::from(frame_size_i32) / i64::from(encoder.fs * 8);
        if i64::from(max_data_bytes) < threshold_bytes {
            selected = MODE_CELT_ONLY;
        }
        selected
    };

    let min_celt = encoder.fs / 100;
    if mode != MODE_CELT_ONLY && frame_size_i32 < min_celt {
        mode = MODE_CELT_ONLY;
    }

    let mut redundancy = false;
    let mut celt_to_silk = false;
    let mut to_celt = false;
    let mut prefill = PrefillMode::None;
    if encoder.user_forced_mode != MODE_CELT_ONLY
        && encoder.prev_mode > 0
        && ((mode != MODE_CELT_ONLY && encoder.prev_mode == MODE_CELT_ONLY)
            || (mode == MODE_CELT_ONLY && encoder.prev_mode != MODE_CELT_ONLY))
    {
        redundancy = true;
        celt_to_silk = mode != MODE_CELT_ONLY;
        if !celt_to_silk {
            if frame_size_i32 >= min_celt {
                mode = encoder.prev_mode;
                to_celt = true;
            } else {
                redundancy = false;
            }
        }
    }

    if encoder.stream_channels == 1
        && encoder.prev_channels == 2
        && !encoder.silk_mode.to_mono
        && mode != MODE_CELT_ONLY
        && encoder.prev_mode != MODE_CELT_ONLY
    {
        encoder.silk_mode.to_mono = true;
        encoder.stream_channels = 2;
    } else {
        encoder.silk_mode.to_mono = false;
    }

    equiv_rate = compute_equiv_rate(
        encoder.bitrate_bps,
        encoder.stream_channels,
        frame_rate,
        encoder.use_vbr,
        mode,
        encoder.complexity,
        encoder.packet_loss_perc,
    );

    if mode != MODE_CELT_ONLY && encoder.prev_mode == MODE_CELT_ONLY {
        let mut dummy = SilkEncControl::default();
        silk_init_encoder(&mut encoder.silk, encoder.arch, &mut dummy)?;
        prefill = PrefillMode::Prefill;
    }

    let mut bandwidth_int = encoder.bandwidth.to_opus_int();
    if mode == MODE_CELT_ONLY || encoder.first || encoder.silk_mode.allow_bandwidth_switch {
        let (voice_thresholds, music_thresholds) =
            if encoder.channels == 2 && encoder.force_channels != 1 {
                (
                    &STEREO_VOICE_BANDWIDTH_THRESHOLDS,
                    &STEREO_MUSIC_BANDWIDTH_THRESHOLDS,
                )
            } else {
                (&MONO_VOICE_BANDWIDTH_THRESHOLDS, &MONO_MUSIC_BANDWIDTH_THRESHOLDS)
            };
        let mut bandwidth_thresholds = [0i32; 8];
        for i in 0..bandwidth_thresholds.len() {
            bandwidth_thresholds[i] = music_thresholds[i]
                + ((i64::from(voice_est)
                    * i64::from(voice_est)
                    * i64::from(voice_thresholds[i] - music_thresholds[i]))
                    >> 14) as i32;
        }

        bandwidth_int = OPUS_BANDWIDTH_FULLBAND;
        loop {
            let idx = usize::try_from(2 * (bandwidth_int - OPUS_BANDWIDTH_MEDIUMBAND))
                .unwrap_or(0);
            let mut threshold = *bandwidth_thresholds.get(idx).unwrap_or(&0);
            let hysteresis = *bandwidth_thresholds.get(idx + 1).unwrap_or(&0);
            if !encoder.first {
                if encoder.auto_bandwidth >= bandwidth_int {
                    threshold -= hysteresis;
                } else {
                    threshold += hysteresis;
                }
            }
            if equiv_rate >= threshold {
                break;
            }
            if bandwidth_int <= OPUS_BANDWIDTH_NARROWBAND {
                break;
            }
            bandwidth_int -= 1;
        }
        if bandwidth_int == OPUS_BANDWIDTH_MEDIUMBAND {
            bandwidth_int = OPUS_BANDWIDTH_WIDEBAND;
        }
        encoder.auto_bandwidth = bandwidth_int;
        if !encoder.first
            && mode != MODE_CELT_ONLY
            && !encoder.silk_mode.in_wb_mode_without_variable_lp
            && bandwidth_int > OPUS_BANDWIDTH_WIDEBAND
        {
            bandwidth_int = OPUS_BANDWIDTH_WIDEBAND;
        }
    }

    if bandwidth_int > encoder.max_bandwidth {
        bandwidth_int = encoder.max_bandwidth;
    }
    if encoder.user_bandwidth != OPUS_AUTO {
        bandwidth_int = encoder.user_bandwidth;
    }
    let max_rate = frame_rate.saturating_mul(max_data_bytes).saturating_mul(8);
    if mode != MODE_CELT_ONLY && max_rate < 15_000 {
        bandwidth_int = bandwidth_int.min(OPUS_BANDWIDTH_WIDEBAND);
    }
    if encoder.fs <= 24_000 && bandwidth_int > OPUS_BANDWIDTH_SUPERWIDEBAND {
        bandwidth_int = OPUS_BANDWIDTH_SUPERWIDEBAND;
    }
    if encoder.fs <= 16_000 && bandwidth_int > OPUS_BANDWIDTH_WIDEBAND {
        bandwidth_int = OPUS_BANDWIDTH_WIDEBAND;
    }
    if encoder.fs <= 12_000 && bandwidth_int > OPUS_BANDWIDTH_MEDIUMBAND {
        bandwidth_int = OPUS_BANDWIDTH_MEDIUMBAND;
    }
    if encoder.fs <= 8_000 && bandwidth_int > OPUS_BANDWIDTH_NARROWBAND {
        bandwidth_int = OPUS_BANDWIDTH_NARROWBAND;
    }
    #[cfg(not(feature = "fixed_point"))]
    if encoder.detected_bandwidth != 0 && encoder.user_bandwidth == OPUS_AUTO {
        let min_detected_bandwidth = if equiv_rate
            <= 18_000 * encoder.stream_channels
            && mode == MODE_CELT_ONLY
        {
            OPUS_BANDWIDTH_NARROWBAND
        } else if equiv_rate <= 24_000 * encoder.stream_channels && mode == MODE_CELT_ONLY {
            OPUS_BANDWIDTH_MEDIUMBAND
        } else if equiv_rate <= 30_000 * encoder.stream_channels {
            OPUS_BANDWIDTH_WIDEBAND
        } else if equiv_rate <= 44_000 * encoder.stream_channels {
            OPUS_BANDWIDTH_SUPERWIDEBAND
        } else {
            OPUS_BANDWIDTH_FULLBAND
        };
        encoder.detected_bandwidth = encoder.detected_bandwidth.max(min_detected_bandwidth);
        bandwidth_int = bandwidth_int.min(encoder.detected_bandwidth);
    }

    let use_fec = decide_fec(
        encoder.inband_fec,
        encoder.packet_loss_perc,
        encoder.silk_mode.lbrr_coded != 0,
        mode,
        &mut bandwidth_int,
        equiv_rate,
    );
    encoder.silk_mode.lbrr_coded = i32::from(use_fec);

    if mode == MODE_CELT_ONLY && bandwidth_int == OPUS_BANDWIDTH_MEDIUMBAND {
        bandwidth_int = OPUS_BANDWIDTH_WIDEBAND;
    }

    let curr_bandwidth_int = bandwidth_int;
    if mode == MODE_SILK_ONLY && curr_bandwidth_int > OPUS_BANDWIDTH_WIDEBAND {
        mode = MODE_HYBRID;
    }
    if mode == MODE_HYBRID && curr_bandwidth_int <= OPUS_BANDWIDTH_WIDEBAND {
        mode = MODE_SILK_ONLY;
    }

    let bandwidth = Bandwidth::from_opus_int(bandwidth_int).unwrap_or(Bandwidth::Wide);
    encoder.bandwidth = bandwidth;

    let max_celt = usize::try_from(encoder.fs / 50).map_err(|_| OpusEncodeError::BadArgument)?;
    if mode == MODE_SILK_ONLY && frame_size * 2 != max_celt {
        let multiples = frame_size / max_celt;
        if !frame_size.is_multiple_of(max_celt) || !(1..=6).contains(&multiples) {
            return Err(OpusEncodeError::Unimplemented);
        }
    }

    if (frame_size > max_celt && mode != MODE_SILK_ONLY) || frame_size > max_celt * 3 {
        let fs = encoder.fs;
        let enc_frame_size_i32 = if mode == MODE_SILK_ONLY {
            if frame_size_i32 == 2 * fs / 25 {
                fs / 25
            } else if frame_size_i32 == 3 * fs / 25 {
                3 * fs / 50
            } else {
                fs / 50
            }
        } else {
            fs / 50
        };
        if enc_frame_size_i32 <= 0 || frame_size_i32 % enc_frame_size_i32 != 0 {
            return Err(OpusEncodeError::Unimplemented);
        }
        let enc_frame_size = usize::try_from(enc_frame_size_i32)
            .map_err(|_| OpusEncodeError::BadArgument)?;
        let nb_frames = frame_size / enc_frame_size;
        if !(1..=6).contains(&nb_frames) {
            return Err(OpusEncodeError::Unimplemented);
        }

        #[cfg(not(feature = "fixed_point"))]
        if let Some(read_state) = analysis_read_state {
            encoder.analysis.restore_read_state(read_state);
        }

        let max_header_bytes = if nb_frames == 2 {
            3
        } else {
            2 + (nb_frames - 1) * 2
        };
        let nb_frames_i32 = i32::try_from(nb_frames).map_err(|_| OpusEncodeError::BadArgument)?;
        let max_header_bytes_i32 =
            i32::try_from(max_header_bytes).map_err(|_| OpusEncodeError::BadArgument)?;
        let max_len_sum = nb_frames_i32
            .checked_add(max_data_bytes)
            .and_then(|value| value.checked_sub(max_header_bytes_i32))
            .ok_or(OpusEncodeError::BufferTooSmall)?;
        if max_len_sum < 2 * nb_frames_i32 {
            return Err(OpusEncodeError::BufferTooSmall);
        }
        let max_len_sum_usize =
            usize::try_from(max_len_sum).map_err(|_| OpusEncodeError::BadArgument)?;
        if max_len_sum_usize > MAX_REPACKETIZER_BYTES {
            return Err(OpusEncodeError::BufferTooSmall);
        }

        let mut tmp_data = [0u8; MAX_REPACKETIZER_BYTES];
        let tmp_data = &mut tmp_data[..max_len_sum_usize];
        let mut repacketizer = OpusRepacketizer::new();
        repacketizer.opus_repacketizer_init();

        let bak_to_mono = encoder.silk_mode.to_mono;
        if bak_to_mono {
            encoder.force_channels = 1;
        } else {
            encoder.prev_channels = encoder.stream_channels;
        }

        let mut tot_size = 0i32;
        let mut dtx_count = 0usize;
        for frame_idx in 0..nb_frames {
            encoder.silk_mode.to_mono = false;
            encoder.nonfinal_frame = frame_idx < nb_frames - 1;
            let frame_to_celt = to_celt && frame_idx == nb_frames - 1;
            let frame_redundancy = redundancy && (frame_to_celt || (!to_celt && frame_idx == 0));

            let frames_left =
                i32::try_from(nb_frames - frame_idx).map_err(|_| OpusEncodeError::BadArgument)?;
            let max_len_per_frame = (max_len_sum - tot_size) / frames_left;
            let mut curr_max = max_len_per_frame;
            if encoder.use_vbr {
                let rate_bytes =
                    (i64::from(encoder.bitrate_bps) * i64::from(enc_frame_size_i32))
                        / (8 * i64::from(encoder.fs));
                let rate_bytes =
                    rate_bytes.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                curr_max = curr_max.min(rate_bytes);
            }
            if curr_max < 2 {
                return Err(OpusEncodeError::BufferTooSmall);
            }

            let tot_size_usize =
                usize::try_from(tot_size).map_err(|_| OpusEncodeError::BadArgument)?;
            let max_len_per_frame_usize =
                usize::try_from(max_len_per_frame).map_err(|_| OpusEncodeError::BadArgument)?;
            if tot_size_usize + max_len_per_frame_usize > tmp_data.len() {
                return Err(OpusEncodeError::BufferTooSmall);
            }
            let curr_max_usize =
                usize::try_from(curr_max).map_err(|_| OpusEncodeError::BadArgument)?;

            #[cfg(not(feature = "fixed_point"))]
            if analysis_read_state.is_some() {
                tonality_get_info(&mut encoder.analysis, &mut encoder.analysis_info, enc_frame_size);
            }

            let start = frame_idx
                .checked_mul(enc_frame_size)
                .and_then(|value| value.checked_mul(channels))
                .ok_or(OpusEncodeError::BadArgument)?;
            let end = start
                .checked_add(enc_frame_size * channels)
                .ok_or(OpusEncodeError::BadArgument)?;
            let frame_buf =
                &mut tmp_data[tot_size_usize..tot_size_usize + max_len_per_frame_usize];
            let len = match encode_frame_native(
                encoder,
                &pcm[start..end],
                enc_frame_size,
                &mut frame_buf[..curr_max_usize],
                lsb_depth,
                silk_use_dtx,
                is_silence,
                frame_redundancy,
                celt_to_silk,
                prefill,
                bandwidth_int,
                mode,
                frame_to_celt,
            ) {
                Ok(len) => len,
                Err(OpusEncodeError::BufferTooSmall) if curr_max < max_len_per_frame => {
                    encode_frame_native(
                        encoder,
                        &pcm[start..end],
                        enc_frame_size,
                        &mut frame_buf[..max_len_per_frame_usize],
                        lsb_depth,
                        silk_use_dtx,
                        is_silence,
                        frame_redundancy,
                        celt_to_silk,
                        prefill,
                        bandwidth_int,
                        mode,
                        frame_to_celt,
                    )?
                }
                Err(err) => return Err(err),
            };
            if len == 1 {
                dtx_count += 1;
            }
            repacketizer
                .opus_repacketizer_cat(&frame_buf[..len], len)
                .map_err(|err| match err {
                    RepacketizerError::BufferTooSmall => OpusEncodeError::BufferTooSmall,
                    _ => OpusEncodeError::InternalError,
                })?;
            tot_size = tot_size.saturating_add(len as i32);
        }

        let maxlen = usize::try_from(max_data_bytes).map_err(|_| OpusEncodeError::BadArgument)?;
        if maxlen > data.len() {
            return Err(OpusEncodeError::BufferTooSmall);
        }
        let mut written = repacketizer
            .opus_repacketizer_out(&mut data[..], maxlen)
            .map_err(|err| match err {
                RepacketizerError::BufferTooSmall => OpusEncodeError::BufferTooSmall,
                _ => OpusEncodeError::InternalError,
            })?;
        if !encoder.use_vbr && dtx_count != nb_frames {
            opus_packet_pad(data, written, maxlen).map_err(|err| match err {
                RepacketizerError::BufferTooSmall => OpusEncodeError::BufferTooSmall,
                _ => OpusEncodeError::InternalError,
            })?;
            written = maxlen;
        }

        encoder.silk_mode.to_mono = bak_to_mono;
        encoder.nonfinal_frame = false;
        return Ok(written);
    }

    encoder.nonfinal_frame = false;
    encode_frame_native(
        encoder,
        &pcm[..required],
        frame_size,
        data,
        lsb_depth,
        silk_use_dtx,
        is_silence,
        redundancy,
        celt_to_silk,
        prefill,
        bandwidth_int,
        mode,
        to_celt,
    )
}

/// Wrapper for encoding 24-bit PCM stored in `i32`, mirroring `opus_encode24`.
pub fn opus_encode24(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[i32],
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
        let scaled = libm::roundf(sample as f32 / 256.0);
        tmp.push(
            scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16,
        );
    }

    opus_encode(encoder, &tmp, frame_size, data)
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
    use alloc::vec;
    use super::{
        DELAY_BUFFER_SAMPLES, DRED_MAX_FRAMES, MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY,
        OPUS_AUTO, OPUS_BITRATE_MAX, OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
        OPUS_BANDWIDTH_WIDEBAND, OPUS_FRAMESIZE_20_MS, OPUS_FRAMESIZE_40_MS, OPUS_SIGNAL_MUSIC,
        OpusEncodeError,
        OpusEncoderCtlError, OpusEncoderCtlRequest, OpusEncoderInitError, StereoWidthState,
        compute_equiv_rate, compute_redundancy_bytes, compute_silk_rate_for_hybrid,
        compute_stereo_width, decide_fec, lin2log, opus_encode, opus_encode24,
        opus_encoder_create, opus_encoder_ctl, opus_encoder_get_size, user_bitrate_to_bitrate,
        VARIABLE_HP_MIN_CUTOFF_HZ,
    };
    use crate::packet::{
        Bandwidth, opus_packet_get_bandwidth, opus_packet_get_mode, opus_packet_get_nb_frames,
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
    fn init_sets_hybrid_state_defaults() {
        let enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");

        assert_eq!(enc.voice_ratio, -1);
        assert_eq!(enc.hybrid_stereo_width_q14, 1_i16 << 14);
        assert_eq!(
            enc.variable_hp_smth2_q15,
            lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8
        );
        assert_eq!(enc.prev_hb_gain, 1.0);
        assert!(enc.hp_mem.iter().all(|&value| value == 0.0));
        assert_eq!(enc.mode, MODE_HYBRID);
        assert_eq!(enc.bandwidth, Bandwidth::Full);
        assert_eq!(enc.auto_bandwidth, 0);
        assert_eq!(enc.prev_mode, 0);
        assert_eq!(enc.prev_channels, 0);
        assert_eq!(enc.prev_framesize, 0);
        assert!(!enc.silk_bw_switch);
        assert!(enc.first);
        assert_eq!(enc.delay_buffer.len(), DELAY_BUFFER_SAMPLES);
        assert!(enc.delay_buffer.iter().all(|&value| value == 0.0));
        assert_eq!(enc.width_mem.xx, 0.0);
        assert_eq!(enc.width_mem.xy, 0.0);
        assert_eq!(enc.width_mem.yy, 0.0);
        assert_eq!(enc.width_mem.smoothed_width, 0.0);
        assert_eq!(enc.width_mem.max_follower, 0.0);
        assert!(!enc.nonfinal_frame);
        #[cfg(not(feature = "fixed_point"))]
        assert_eq!(enc.detected_bandwidth, 0);
    }

    #[test]
    fn encodes_silk_only_frame_with_valid_toc() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(12_000)).unwrap();
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
    fn encodes_hybrid_multiframe_packet() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_HYBRID)).unwrap();
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetExpertFrameDuration(OPUS_FRAMESIZE_40_MS),
        )
        .unwrap();
        let pcm = [0i16; 1920];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 1920, &mut out).expect("encode");
        assert!(len >= 1);
        assert_eq!(opus_packet_get_mode(&out[..len]).unwrap(), crate::packet::Mode::HYBRID);
        assert_eq!(opus_packet_get_nb_frames(&out[..len], len).unwrap(), 2);
        assert_eq!(opus_packet_get_samples_per_frame(&out[..len], 48_000).unwrap(), 960);
    }

    #[test]
    fn restricted_low_delay_forces_celt() {
        let mut enc = opus_encoder_create(48_000, 1, 2051).expect("encoder");
        let pcm = [0i16; 960];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 960, &mut out).expect("encode");
        assert_eq!(opus_packet_get_mode(&out[..len]).unwrap(), crate::packet::Mode::CELT);
    }

    #[test]
    fn bandwidth_clamps_to_sample_rate_limits() {
        let mut enc = opus_encoder_create(16_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY)).unwrap();
        let pcm = [0i16; 320];
        let mut out = [0u8; 4000];

        let _ = opus_encode(&mut enc, &pcm, 320, &mut out).expect("encode");
        assert!(enc.bandwidth.to_opus_int() <= OPUS_BANDWIDTH_WIDEBAND);
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

    #[cfg(feature = "dred")]
    #[test]
    fn ctl_round_trips_dred_duration() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDredDuration(12)).unwrap();
        let mut duration = 0;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetDredDuration(&mut duration)).unwrap();
        assert_eq!(duration, 12);
    }

    #[cfg(feature = "dred")]
    #[test]
    fn ctl_rejects_invalid_dred_duration() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");

        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDredDuration(-1)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetDredDuration(DRED_MAX_FRAMES + 1),
            )
            .unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
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

    #[test]
    fn opus_encode24_accepts_int24_input() {
        let mut enc = opus_encoder_create(48_000, 1, 2051).expect("encoder");
        let pcm = vec![0i32; 960];
        let mut out = vec![0u8; 1500];
        let result = opus_encode24(&mut enc, &pcm, 960, &mut out);
        assert!(result.is_ok(), "opus_encode24 should accept int24 input");
    }

    #[test]
    fn user_bitrate_to_bitrate_respects_auto_and_max() {
        let mut enc = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        let frame_size = 960;
        let max_data_bytes = 200;

        enc.user_bitrate_bps = OPUS_AUTO;
        let expected_auto = 60 * 48_000 / frame_size + 48_000 * 2;
        assert_eq!(
            user_bitrate_to_bitrate(&enc, frame_size, max_data_bytes),
            expected_auto
        );

        enc.user_bitrate_bps = OPUS_BITRATE_MAX;
        let expected_max = max_data_bytes * 8 * 48_000 / frame_size;
        assert_eq!(
            user_bitrate_to_bitrate(&enc, frame_size, max_data_bytes),
            expected_max
        );

        enc.user_bitrate_bps = 12_345;
        assert_eq!(
            user_bitrate_to_bitrate(&enc, frame_size, max_data_bytes),
            12_345
        );
    }

    #[test]
    fn compute_equiv_rate_matches_reference_math() {
        let equiv = compute_equiv_rate(10_000, 1, 100, true, MODE_CELT_ONLY, 10, 0);
        assert_eq!(equiv, 7_000);
    }

    #[test]
    fn decide_fec_restores_bandwidth_when_unavailable() {
        let mut bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        let enabled = decide_fec(true, 20, false, MODE_SILK_ONLY, &mut bandwidth, 1_000);
        assert!(!enabled);
        assert_eq!(bandwidth, OPUS_BANDWIDTH_WIDEBAND);
    }

    #[test]
    fn compute_redundancy_bytes_respects_caps() {
        let redundancy = compute_redundancy_bytes(100, 20_000, 50, 1);
        assert_eq!(redundancy, 24);
    }

    #[test]
    fn compute_silk_rate_for_hybrid_from_table() {
        let silk_rate = compute_silk_rate_for_hybrid(
            24_000,
            OPUS_BANDWIDTH_SUPERWIDEBAND,
            true,
            true,
            false,
            1,
        );
        assert_eq!(silk_rate, 18_300);
    }

    #[test]
    fn compute_stereo_width_stays_zero_for_silence() {
        let pcm = vec![0i16; 2 * 96];
        let mut mem = StereoWidthState::default();
        let width = compute_stereo_width(&pcm, 96, 48_000, &mut mem);
        assert_eq!(width, 0.0);
        assert_eq!(mem.xx, 0.0);
        assert_eq!(mem.xy, 0.0);
        assert_eq!(mem.yy, 0.0);
        assert_eq!(mem.smoothed_width, 0.0);
        assert_eq!(mem.max_follower, 0.0);
    }
}
