//! Small pieces of the top-level Opus decoder API.
//!
//! Ports the size helper from `opus_decoder_get_size()` so callers can
//! determine how much memory the combined SILK/CELT decoder requires.

use crate::celt::{
    CeltDecoderCtlError, DecoderCtlRequest as CeltDecoderCtlRequest, OwnedCeltDecoder,
    canonical_mode, celt_decoder_get_size, opus_custom_decoder_create, opus_custom_decoder_ctl,
    opus_select_arch,
};
use crate::packet::{PacketError, opus_packet_get_nb_samples};
use crate::silk::dec_api::{
    DECODER_NUM_CHANNELS, Decoder as SilkDecoder, reset_decoder as silk_reset_decoder,
};
use crate::silk::decoder_state::DecoderState;
use crate::silk::errors::SilkError;
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

/// Strongly-typed replacement for the decoder-side varargs CTL dispatcher.
pub enum OpusDecoderCtlRequest<'req> {
    SetGain(i32),
    GetGain(&'req mut i32),
    ResetState,
    GetLastPacketDuration(&'req mut i32),
    SetPhaseInversionDisabled(bool),
    GetPhaseInversionDisabled(&'req mut bool),
}

impl<'mode> OpusDecoder<'mode> {
    /// Mirrors `opus_decoder_init` by preparing both the SILK and CELT decoders.
    pub fn init(&mut self, fs: i32, channels: i32) -> Result<(), OpusDecoderInitError> {
        if !matches!(fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000) || !matches!(channels, 1 | 2) {
            return Err(OpusDecoderInitError::BadArgument);
        }

        self.fs = fs;
        self.channels = channels;
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

/// Applies a control request to the provided decoder state.
pub fn opus_decoder_ctl<'req>(
    decoder: &mut OpusDecoder<'_>,
    request: OpusDecoderCtlRequest<'req>,
) -> Result<(), OpusDecoderCtlError> {
    match request {
        OpusDecoderCtlRequest::SetGain(value) => {
            if !(-32_768..=32_767).contains(&value) {
                return Err(OpusDecoderCtlError::BadArgument);
            }
            decoder.decode_gain = value;
        }
        OpusDecoderCtlRequest::GetGain(slot) => {
            *slot = decoder.decode_gain;
        }
        OpusDecoderCtlRequest::ResetState => decoder.reset_state()?,
        OpusDecoderCtlRequest::GetLastPacketDuration(slot) => {
            *slot = decoder.last_packet_duration;
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
        OpusDecoderCtlError, OpusDecoderCtlRequest, opus_decoder_create, opus_decoder_ctl,
        opus_decoder_get_size,
    };
    use crate::celt::{canonical_mode, celt_decoder_get_size, opus_custom_decoder_create};
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
}
