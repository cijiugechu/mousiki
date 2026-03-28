extern crate std;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use std::fs::File;
use std::io::Write;

use crate::libopusenc::ogg_packer::OggPacker;
use crate::libopusenc::opus_header::{
    CommentError, OpusHeader, comment_add, comment_init, comment_pad, opus_header_get_size,
    opus_header_to_packet,
};
use crate::libopusenc::picture::{
    parse_picture_specification, parse_picture_specification_from_memory,
};
use crate::libopusenc::resample::SpeexResampler;
use crate::opus_multistream::{
    OpusMultistreamEncoder, OpusMultistreamEncoderCtlRequest, OpusMultistreamEncoderError,
    opus_multistream_encode_float, opus_multistream_encoder_create,
    opus_multistream_encoder_ctl, opus_multistream_surround_encoder_create,
};
use crate::projection::{
    OpusProjectionEncoder, OpusProjectionEncoderCtlRequest, OpusProjectionEncoderError,
    opus_projection_ambisonics_encoder_create, opus_projection_encode_float,
    opus_projection_encoder_ctl,
};

const FRAME_SIZE_20_MS: usize = 960;
const MAX_PACKET_SIZE: usize = 1277 * 6 * 255 + 2;
const OPE_ABI_VERSION: i32 = 0;
const MAX_LOOKAHEAD: i32 = 96_000;
const LPC_PADDING: usize = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpeError {
    Ok,
    BadArg,
    InternalError,
    Unimplemented,
    AllocFail,
    CannotOpen,
    TooLate,
    InvalidPicture,
    InvalidIcon,
    WriteFail,
    CloseFail,
}

impl From<CommentError> for OpeError {
    fn from(value: CommentError) -> Self {
        match value {
            CommentError::AllocationFailed => Self::AllocFail,
        }
    }
}

pub trait OpusEncCallbacks {
    fn write(&mut self, data: &[u8]) -> Result<(), OpeError>;

    fn close(&mut self) -> Result<(), OpeError> {
        Ok(())
    }
}

pub type PacketCallback = dyn FnMut(&[u8], u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OggOpusComments {
    comment: Vec<u8>,
    seen_file_icons: i32,
}

impl OggOpusComments {
    pub fn create() -> Result<Self, OpeError> {
        let vendor = format!("libopusenc {}", env!("CARGO_PKG_VERSION"));
        Ok(Self {
            comment: comment_init(&vendor)?,
            seen_file_icons: 0,
        })
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    pub fn add(&mut self, tag: &str, value: &str) -> Result<(), OpeError> {
        if tag.contains('=') {
            return Err(OpeError::BadArg);
        }
        comment_add(&mut self.comment, Some(tag), value)?;
        Ok(())
    }

    pub fn add_string(&mut self, tag_and_value: &str) -> Result<(), OpeError> {
        if !tag_and_value.contains('=') {
            return Err(OpeError::BadArg);
        }
        comment_add(&mut self.comment, None, tag_and_value)?;
        Ok(())
    }

    pub fn add_picture(
        &mut self,
        filename: &str,
        picture_type: i32,
        description: Option<&str>,
    ) -> Result<(), OpeError> {
        let picture = parse_picture_specification(
            filename,
            picture_type,
            description,
            &mut self.seen_file_icons,
        )?;
        comment_add(&mut self.comment, Some("METADATA_BLOCK_PICTURE"), &picture)?;
        Ok(())
    }

    pub fn add_picture_from_memory(
        &mut self,
        picture: &[u8],
        picture_type: i32,
        description: Option<&str>,
    ) -> Result<(), OpeError> {
        let picture = parse_picture_specification_from_memory(
            picture,
            picture_type,
            description,
            &mut self.seen_file_icons,
        )?;
        comment_add(&mut self.comment, Some("METADATA_BLOCK_PICTURE"), &picture)?;
        Ok(())
    }

    fn padded_bytes(&self, padding: i32) -> Result<Vec<u8>, OpeError> {
        let mut comment = self.comment.clone();
        comment_pad(&mut comment, padding)?;
        Ok(comment)
    }
}

enum OutputMode {
    Pull,
    Callbacks(Box<dyn OpusEncCallbacks>),
    File(File),
}

impl core::fmt::Debug for OutputMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Pull => f.write_str("Pull"),
            Self::Callbacks(_) => f.write_str("Callbacks(..)"),
            Self::File(_) => f.write_str("File(..)"),
        }
    }
}

#[derive(Debug)]
struct EncStream {
    comments: OggOpusComments,
    serialno: Option<i32>,
    stream_is_init: bool,
    header_is_frozen: bool,
    header_gain: i32,
    preskip: i32,
    end_granule: usize,
    granule_offset: i64,
    close_at_end: bool,
    next_output: Option<OutputMode>,
}

impl EncStream {
    fn new(comments: &OggOpusComments, next_output: Option<OutputMode>) -> Self {
        Self {
            comments: comments.copy(),
            serialno: None,
            stream_is_init: false,
            header_is_frozen: false,
            header_gain: 0,
            preskip: 0,
            end_granule: 0,
            granule_offset: 0,
            close_at_end: true,
            next_output,
        }
    }
}

#[derive(Debug)]
enum EncoderBackend {
    Multistream(OpusMultistreamEncoder<'static>),
    Projection(OpusProjectionEncoder<'static>),
}

impl EncoderBackend {
    fn encode_float(&mut self, pcm: &[f32], frame_size: usize, data: &mut [u8]) -> Result<usize, OpeError> {
        match self {
            Self::Multistream(encoder) => {
                opus_multistream_encode_float(encoder, pcm, frame_size, data)
                    .map_err(map_multistream_error)
            }
            Self::Projection(encoder) => {
                opus_projection_encode_float(encoder, pcm, frame_size, data)
                    .map_err(map_projection_error)
            }
        }
    }

    fn lookahead(&mut self) -> Result<i32, OpeError> {
        let mut value = 0;
        match self {
            Self::Multistream(encoder) => opus_multistream_encoder_ctl(
                encoder,
                OpusMultistreamEncoderCtlRequest::GetLookahead(&mut value),
            )
            .map_err(map_multistream_error)?,
            Self::Projection(encoder) => opus_projection_encoder_ctl(
                encoder,
                OpusProjectionEncoderCtlRequest::Multistream(
                    OpusMultistreamEncoderCtlRequest::GetLookahead(&mut value),
                ),
            )
            .map_err(map_projection_error)?,
        }
        Ok(value)
    }

    fn set_prediction_disabled(&mut self, value: bool) -> Result<(), OpeError> {
        match self {
            Self::Multistream(encoder) => opus_multistream_encoder_ctl(
                encoder,
                OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(value),
            )
            .map_err(map_multistream_error),
            Self::Projection(encoder) => opus_projection_encoder_ctl(
                encoder,
                OpusProjectionEncoderCtlRequest::Multistream(
                    OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(value),
                ),
            )
            .map_err(map_projection_error),
        }
    }

    fn prediction_disabled(&mut self) -> Result<bool, OpeError> {
        let mut value = false;
        match self {
            Self::Multistream(encoder) => opus_multistream_encoder_ctl(
                encoder,
                OpusMultistreamEncoderCtlRequest::GetPredictionDisabled(&mut value),
            )
            .map_err(map_multistream_error)?,
            Self::Projection(encoder) => opus_projection_encoder_ctl(
                encoder,
                OpusProjectionEncoderCtlRequest::Multistream(
                    OpusMultistreamEncoderCtlRequest::GetPredictionDisabled(&mut value),
                ),
            )
            .map_err(map_projection_error)?,
        }
        Ok(value)
    }

    fn set_expert_frame_duration(&mut self, value: i32) -> Result<(), OpeError> {
        match self {
            Self::Multistream(encoder) => opus_multistream_encoder_ctl(
                encoder,
                OpusMultistreamEncoderCtlRequest::SetExpertFrameDuration(value),
            )
            .map_err(map_multistream_error),
            Self::Projection(encoder) => opus_projection_encoder_ctl(
                encoder,
                OpusProjectionEncoderCtlRequest::Multistream(
                    OpusMultistreamEncoderCtlRequest::SetExpertFrameDuration(value),
                ),
            )
            .map_err(map_projection_error),
        }
    }
}

pub struct OggOpusEnc {
    backend: Option<EncoderBackend>,
    oggp: Option<OggPacker>,
    output: OutputMode,
    pending_pages: VecDeque<Vec<u8>>,
    streams: VecDeque<EncStream>,
    header: OpusHeader,
    rate: u32,
    channels: usize,
    frame_size: usize,
    frame_size_request: i32,
    decision_delay: i32,
    comment_padding: i32,
    max_ogg_delay: i32,
    pcm_buffer: Vec<f32>,
    pcm_buffer_start: usize,
    input_buffer: Vec<f32>,
    frame_buffer: Vec<f32>,
    packet_buffer: Vec<u8>,
    resampler_output_buffer: Vec<f32>,
    resampler_zero_buffer: Vec<f32>,
    resampler: Option<SpeexResampler>,
    write_granule: usize,
    resampled_granule: usize,
    curr_granule: usize,
    global_granule_offset: Option<usize>,
    drained: bool,
    fatal_error: Option<OpeError>,
    packet_callback: Option<Box<PacketCallback>>,
    chaining_keyframe: Option<Vec<u8>>,
    next_generated_serial: i32,
}

impl OggOpusEnc {
    pub fn create_pull(
        comments: &OggOpusComments,
        rate: i32,
        channels: i32,
        family: i32,
    ) -> Result<Self, OpeError> {
        Self::create(OutputMode::Pull, comments, rate, channels, family)
    }

    pub fn create_callbacks(
        callbacks: Box<dyn OpusEncCallbacks>,
        comments: &OggOpusComments,
        rate: i32,
        channels: i32,
        family: i32,
    ) -> Result<Self, OpeError> {
        Self::create(OutputMode::Callbacks(callbacks), comments, rate, channels, family)
    }

    pub fn create_file(
        path: &str,
        comments: &OggOpusComments,
        rate: i32,
        channels: i32,
        family: i32,
    ) -> Result<Self, OpeError> {
        let file = File::create(path).map_err(|_| OpeError::CannotOpen)?;
        Self::create(OutputMode::File(file), comments, rate, channels, family)
    }

    fn create(
        output: OutputMode,
        comments: &OggOpusComments,
        rate: i32,
        channels: i32,
        family: i32,
    ) -> Result<Self, OpeError> {
        if rate <= 0 || !(1..=255).contains(&channels) {
            return Err(OpeError::BadArg);
        }
        if !matches!(family, -1 | 0 | 1 | 2 | 3 | 255) {
            return Err(if family < -1 || family > 255 {
                OpeError::BadArg
            } else {
                OpeError::Unimplemented
            });
        }

        let backend = if family == -1 {
            None
        } else {
            Some(create_backend(channels as usize, family)?)
        };
        let mut header = OpusHeader {
            channels,
            input_sample_rate: rate as u32,
            channel_mapping: family,
            ..Default::default()
        };
        if let Some(backend_ref) = backend.as_ref() {
            fill_header_layout(&mut header, backend_ref);
        }

        let mut resampler = if rate != 48_000 {
            let mut resampler = SpeexResampler::new(channels as u32, rate as u32, 48_000, 5)
                .map_err(|_| OpeError::BadArg)?;
            resampler.skip_zeros().map_err(|_| OpeError::InternalError)?;
            Some(resampler)
        } else {
            None
        };

        let mut streams = VecDeque::new();
        streams.push_back(EncStream::new(comments, None));

        let mut encoder = Self {
            backend,
            oggp: None,
            output,
            pending_pages: VecDeque::new(),
            streams,
            header,
            rate: rate as u32,
            channels: channels as usize,
            frame_size: FRAME_SIZE_20_MS,
            frame_size_request: 5004,
            decision_delay: MAX_LOOKAHEAD,
            comment_padding: 512,
            max_ogg_delay: 48_000,
            pcm_buffer: Vec::new(),
            pcm_buffer_start: 0,
            input_buffer: Vec::new(),
            frame_buffer: vec![0.0; FRAME_SIZE_20_MS * channels as usize],
            packet_buffer: vec![0; MAX_PACKET_SIZE],
            resampler_output_buffer: Vec::new(),
            resampler_zero_buffer: Vec::new(),
            resampler: resampler.take(),
            write_granule: 0,
            resampled_granule: 0,
            curr_granule: 0,
            global_granule_offset: None,
            drained: false,
            fatal_error: if family == -1 {
                Some(OpeError::TooLate)
            } else {
                None
            },
            packet_callback: None,
            chaining_keyframe: None,
            next_generated_serial: 0,
        };
        if let Some(backend) = encoder.backend.as_mut() {
            backend.set_expert_frame_duration(encoder.frame_size_request)?;
        }
        Ok(encoder)
    }

    pub fn deferred_init_with_mapping(
        &mut self,
        family: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
    ) -> Result<(), OpeError> {
        if family < 0 || family > 255 {
            return Err(OpeError::BadArg);
        }
        if !matches!(family, 1 | 2 | 255) {
            return Err(OpeError::Unimplemented);
        }
        if streams <= 0
            || streams > 255
            || coupled_streams < 0
            || coupled_streams >= 128
            || streams + coupled_streams > 255
            || mapping.len() != (streams + coupled_streams) as usize
        {
            return Err(OpeError::BadArg);
        }
        if self.streams.front().is_some_and(|stream| stream.stream_is_init)
            || self.streams.back().is_some_and(|stream| stream.header_is_frozen)
        {
            return Err(OpeError::TooLate);
        }

        let mut backend = EncoderBackend::Multistream(
            opus_multistream_encoder_create(
                48_000,
                self.channels,
                streams as usize,
                coupled_streams as usize,
                mapping,
                2049,
            )
            .map_err(map_multistream_error)?,
        );
        backend.set_expert_frame_duration(self.frame_size_request)?;
        self.backend = Some(backend);
        self.header.channel_mapping = family;
        self.header.nb_streams = streams;
        self.header.nb_coupled = coupled_streams;
        self.header.stream_map.fill(0);
        self.header.stream_map[..mapping.len()].copy_from_slice(mapping);
        self.fatal_error = None;
        Ok(())
    }

    pub fn flush_header(&mut self) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        if self.drained {
            return Err(OpeError::TooLate);
        }
        let Some(current) = self.streams.front() else {
            return Err(OpeError::TooLate);
        };
        if current.header_is_frozen || current.stream_is_init {
            return Err(OpeError::TooLate);
        }
        self.init_stream()
    }

    pub fn write(&mut self, pcm: &[i16], samples_per_channel: usize) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        if self.drained {
            return Err(OpeError::TooLate);
        }
        if pcm.len() < samples_per_channel.saturating_mul(self.channels) {
            return Err(OpeError::BadArg);
        }
        let Some(last) = self.streams.back_mut() else {
            return Err(OpeError::TooLate);
        };
        last.header_is_frozen = true;
        last.end_granule = self.write_granule.saturating_add(samples_per_channel);
        if !self.streams.front().is_some_and(|stream| stream.stream_is_init) {
            self.init_stream()?;
        }

        let sample_count = samples_per_channel * self.channels;
        let mut input_buffer = core::mem::take(&mut self.input_buffer);
        if input_buffer.len() != sample_count {
            input_buffer.resize(sample_count, 0.0);
        }
        for (dst, &sample) in input_buffer.iter_mut().zip(&pcm[..sample_count]) {
            *dst = sample as f32 / 32768.0;
        }
        self.write_granule += samples_per_channel;
        let append_result = self.append_pcm(&input_buffer, samples_per_channel);
        self.input_buffer = input_buffer;
        append_result?;
        self.process_ready_packets(false)
    }

    pub fn write_float(&mut self, pcm: &[f32], samples_per_channel: usize) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        if self.drained {
            return Err(OpeError::TooLate);
        }
        if pcm.len() < samples_per_channel.saturating_mul(self.channels) {
            return Err(OpeError::BadArg);
        }
        let Some(last) = self.streams.back_mut() else {
            return Err(OpeError::TooLate);
        };
        last.header_is_frozen = true;
        last.end_granule = self.write_granule.saturating_add(samples_per_channel);
        if !self.streams.front().is_some_and(|stream| stream.stream_is_init) {
            self.init_stream()?;
        }
        self.write_granule += samples_per_channel;
        self.append_pcm(&pcm[..samples_per_channel * self.channels], samples_per_channel)?;
        self.process_ready_packets(false)
    }

    pub fn drain(&mut self) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        if self.drained {
            return Err(OpeError::TooLate);
        }
        if self.streams.is_empty() {
            return Err(OpeError::TooLate);
        }
        if !self.streams.front().is_some_and(|stream| stream.stream_is_init) {
            self.init_stream()?;
        }

        let target_frames = self.input_samples_to_granule(self.write_granule)?;
        if self.resampled_granule < target_frames {
            self.flush_resampler(target_frames)?;
        }
        self.pad_for_drain()?;
        self.process_ready_packets(true)?;
        if !self.streams.is_empty() {
            return Err(OpeError::InternalError);
        }
        self.pcm_buffer.clear();
        self.pcm_buffer_start = 0;
        self.emit_pages(false)?;
        self.close_output()?;
        self.drained = true;
        Ok(())
    }

    pub fn get_page(&mut self, flush: bool) -> Result<Option<Vec<u8>>, OpeError> {
        self.ensure_not_fatal()?;
        if flush {
            self.emit_pages(true)?;
        }
        Ok(self.pending_pages.pop_front())
    }

    pub fn set_decision_delay(&mut self, value: i32) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        if value < 0 {
            return Err(OpeError::BadArg);
        }
        self.decision_delay = value.min(MAX_LOOKAHEAD);
        Ok(())
    }

    pub const fn decision_delay(&self) -> i32 {
        self.decision_delay
    }

    pub fn set_comment_padding(&mut self, value: i32) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        if value < 0 {
            return Err(OpeError::BadArg);
        }
        self.comment_padding = value;
        Ok(())
    }

    pub const fn comment_padding(&self) -> i32 {
        self.comment_padding
    }

    pub fn set_serialno(&mut self, value: i32) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        let Some(last) = self.streams.back_mut() else {
            return Err(OpeError::TooLate);
        };
        if last.header_is_frozen {
            return Err(OpeError::TooLate);
        }
        last.serialno = Some(value);
        Ok(())
    }

    pub fn serialno(&mut self) -> Result<i32, OpeError> {
        self.ensure_not_fatal()?;
        let Some(_) = self.streams.back() else {
            return Err(OpeError::TooLate);
        };
        if self.streams.back().is_some_and(|stream| stream.serialno.is_none()) {
            let serialno = self.generate_serialno();
            if let Some(last) = self.streams.back_mut() {
                last.serialno = Some(serialno);
            }
        }
        Ok(self
            .streams
            .back()
            .and_then(|stream| stream.serialno)
            .expect("serialno just initialised"))
    }

    pub fn set_header_gain(&mut self, value: i32) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        let Some(last) = self.streams.back_mut() else {
            return Err(OpeError::TooLate);
        };
        if last.header_is_frozen {
            return Err(OpeError::TooLate);
        }
        last.header_gain = value;
        Ok(())
    }

    pub fn header_gain(&self) -> i32 {
        self.streams.back().map_or(0, |stream| stream.header_gain)
    }

    pub fn set_muxing_delay(&mut self, value: i32) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        if value < 0 {
            return Err(OpeError::BadArg);
        }
        self.max_ogg_delay = value;
        if let Some(oggp) = &mut self.oggp {
            oggp.set_muxing_delay(value as u64);
        }
        Ok(())
    }

    pub const fn muxing_delay(&self) -> i32 {
        self.max_ogg_delay
    }

    pub const fn nb_streams(&self) -> i32 {
        self.header.nb_streams
    }

    pub const fn nb_coupled_streams(&self) -> i32 {
        self.header.nb_coupled
    }

    pub fn set_packet_callback(&mut self, callback: Option<Box<PacketCallback>>) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        self.packet_callback = callback;
        Ok(())
    }

    pub fn chain_current(&mut self, comments: &OggOpusComments) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        let Some(last) = self.streams.back_mut() else {
            return Err(OpeError::TooLate);
        };
        last.close_at_end = false;
        let mut stream = EncStream::new(comments, None);
        stream.header_gain = last.header_gain;
        self.streams.push_back(stream);
        Ok(())
    }

    pub fn continue_new_file(
        &mut self,
        path: &str,
        comments: &OggOpusComments,
    ) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        let file = File::create(path).map_err(|_| OpeError::CannotOpen)?;
        self.continue_with_output(OutputMode::File(file), comments)
    }

    pub fn continue_new_callbacks(
        &mut self,
        callbacks: Box<dyn OpusEncCallbacks>,
        comments: &OggOpusComments,
    ) -> Result<(), OpeError> {
        self.ensure_not_fatal()?;
        self.continue_with_output(OutputMode::Callbacks(callbacks), comments)
    }

    fn continue_with_output(
        &mut self,
        output: OutputMode,
        comments: &OggOpusComments,
    ) -> Result<(), OpeError> {
        let header_gain = self.streams.back().map_or(0, |stream| stream.header_gain);
        let mut stream = EncStream::new(comments, Some(output));
        stream.header_gain = header_gain;
        stream.end_granule = self.write_granule;
        self.streams.push_back(stream);
        Ok(())
    }

    fn backend_mut(&mut self) -> Result<&mut EncoderBackend, OpeError> {
        self.backend.as_mut().ok_or(OpeError::TooLate)
    }

    fn ensure_not_fatal(&self) -> Result<(), OpeError> {
        if let Some(err) = self.fatal_error {
            Err(err)
        } else {
            Ok(())
        }
    }

    fn init_stream(&mut self) -> Result<(), OpeError> {
        let Some(_) = self.streams.front() else {
            return Err(OpeError::TooLate);
        };
        if self.streams.front().is_some_and(|stream| stream.stream_is_init) {
            return Ok(());
        }
        if self.streams.front().is_some_and(|stream| stream.serialno.is_none()) {
            let serialno = self.generate_serialno();
            if let Some(current) = self.streams.front_mut() {
                current.serialno = Some(serialno);
            }
        }

        let serialno = self
            .streams
            .front()
            .and_then(|stream| stream.serialno)
            .expect("serialno initialised");
        if let Some(oggp) = &mut self.oggp {
            oggp.chain(serialno);
        } else {
            let mut oggp = OggPacker::new(serialno);
            oggp.set_muxing_delay(self.max_ogg_delay as u64);
            self.oggp = Some(oggp);
        }

        if self.global_granule_offset.is_none() {
            let preskip = self.backend_mut()?.lookahead()?;
            if let Some(current) = self.streams.front_mut() {
                current.preskip = preskip;
            }
            self.global_granule_offset = Some(preskip as usize);
        }

        let mut header = self.header;
        header.preskip = self.streams.front().map_or(0, |stream| stream.preskip);
        header.gain = self.streams.front().map_or(0, |stream| stream.header_gain);
        let header_size = opus_header_get_size(&header);
        let mut header_packet = vec![0u8; header_size];
        let packet_size = match self.backend.as_ref() {
            Some(EncoderBackend::Projection(encoder)) => opus_header_to_packet(
                &header,
                &mut header_packet,
                Some(encoder.projection_layout()),
            ),
            _ => opus_header_to_packet(&header, &mut header_packet, None),
        }
        .map_err(|_| OpeError::InternalError)?;
        header_packet.truncate(packet_size);
        self.commit_packet(&header_packet, 0, false)?;
        self.flush_current_page()?;

        let comments = self
            .streams
            .front()
            .ok_or(OpeError::TooLate)?
            .comments
            .padded_bytes(self.comment_padding)?;
        self.commit_packet(&comments, 0, false)?;
        self.flush_current_page()?;

        if let Some(current) = self.streams.front_mut() {
            current.stream_is_init = true;
        }
        Ok(())
    }

    fn commit_packet(&mut self, packet: &[u8], granulepos: u64, eos: bool) -> Result<(), OpeError> {
        if let Some(callback) = &mut self.packet_callback {
            callback(packet, 0);
        }
        let oggp = self.oggp.as_mut().ok_or(OpeError::InternalError)?;
        let buffer = oggp
            .get_packet_buffer(packet.len())
            .ok_or(OpeError::AllocFail)?;
        buffer.copy_from_slice(packet);
        oggp.commit_packet(packet.len(), granulepos, eos)
            .map_err(|_| OpeError::InternalError)?;
        Ok(())
    }

    fn flush_current_page(&mut self) -> Result<(), OpeError> {
        if let Some(oggp) = &mut self.oggp {
            oggp.flush_page();
        }
        self.emit_pages(false)
    }

    fn emit_pages(&mut self, flush: bool) -> Result<(), OpeError> {
        if let Some(oggp) = &mut self.oggp
            && flush
        {
            oggp.flush_page();
        }
        while let Some(page) = self.oggp.as_mut().and_then(OggPacker::get_next_page) {
            match &mut self.output {
                OutputMode::Pull => self.pending_pages.push_back(page),
                OutputMode::Callbacks(callbacks) => callbacks.write(&page)?,
                OutputMode::File(file) => file.write_all(&page).map_err(|_| OpeError::WriteFail)?,
            }
        }
        Ok(())
    }

    fn append_pcm(&mut self, pcm: &[f32], samples_per_channel: usize) -> Result<(), OpeError> {
        let mut resampler_output_buffer = core::mem::take(&mut self.resampler_output_buffer);
        if let Some(resampler) = &mut self.resampler {
            let mut start = 0usize;
            let mut remaining = samples_per_channel;
            while remaining > 0 {
                let out_capacity_frames = (((remaining as u64) * 48_000 + self.rate as u64 - 1)
                    / self.rate as u64) as usize
                    + resampler.output_latency() as usize
                    + 16;
                let out_capacity_samples = out_capacity_frames * self.channels;
                if resampler_output_buffer.len() < out_capacity_samples {
                    resampler_output_buffer.resize(out_capacity_samples, 0.0);
                }
                let mut in_len = remaining as u32;
                let mut out_len = out_capacity_frames as u32;
                let resample_result = resampler
                    .process_interleaved_float(
                        Some(&pcm[start * self.channels..]),
                        &mut in_len,
                        &mut resampler_output_buffer,
                        &mut out_len,
                    )
                    .map_err(|_| OpeError::InternalError);
                if let Err(err) = resample_result {
                    self.resampler_output_buffer = resampler_output_buffer;
                    return Err(err);
                }
                if in_len == 0 && out_len == 0 {
                    self.resampler_output_buffer = resampler_output_buffer;
                    return Err(OpeError::InternalError);
                }
                self.pcm_buffer
                    .extend_from_slice(&resampler_output_buffer[..out_len as usize * self.channels]);
                self.resampled_granule += out_len as usize;
                start += in_len as usize;
                remaining -= in_len as usize;
            }
        } else {
            self.pcm_buffer
                .extend_from_slice(&pcm[..samples_per_channel * self.channels]);
            self.resampled_granule += samples_per_channel;
        }
        self.resampler_output_buffer = resampler_output_buffer;
        Ok(())
    }

    fn buffered_frames(&self) -> usize {
        self.pcm_buffer
            .len()
            .checked_div(self.channels)
            .unwrap_or(0)
            .saturating_sub(self.pcm_buffer_start)
    }

    fn compact_pcm_buffer(&mut self) {
        if self.pcm_buffer_start == 0 {
            return;
        }
        let consumed_samples = self.pcm_buffer_start * self.channels;
        if consumed_samples >= self.pcm_buffer.len() {
            self.pcm_buffer.clear();
            self.pcm_buffer_start = 0;
            return;
        }
        self.pcm_buffer.copy_within(consumed_samples.., 0);
        self.pcm_buffer.truncate(self.pcm_buffer.len() - consumed_samples);
        self.pcm_buffer_start = 0;
    }

    fn process_ready_packets(&mut self, draining: bool) -> Result<(), OpeError> {
        let decision_delay = if draining {
            0
        } else {
            self.decision_delay.max(0) as usize
        };

        while !self.streams.is_empty() && self.buffered_frames() > self.frame_size + decision_delay {
            let active_end_input = self.streams.front().ok_or(OpeError::TooLate)?.end_granule;
            let active_end_audio = self.input_samples_to_granule(active_end_input)?;
            let active_end_granule = active_end_audio
                .checked_add(self.global_granule_offset.unwrap_or(0))
                .ok_or(OpeError::InternalError)?;
            let has_next_stream = self.streams.len() > 1;
            let is_keyframe = has_next_stream
                && self
                    .curr_granule
                    .saturating_add(self.frame_size.saturating_mul(2))
                    >= active_end_granule;

            let previous_prediction = if is_keyframe {
                let prev = self.backend_mut()?.prediction_disabled()?;
                self.backend_mut()?.set_prediction_disabled(true)?;
                Some(prev)
            } else {
                None
            };

            let frame_samples = self.frame_size * self.channels;
            let src_start = self.pcm_buffer_start * self.channels;
            let src_end = src_start + frame_samples;
            let mut frame_buffer = core::mem::take(&mut self.frame_buffer);
            if frame_buffer.len() != frame_samples {
                frame_buffer.resize(frame_samples, 0.0);
            }
            frame_buffer.copy_from_slice(&self.pcm_buffer[src_start..src_end]);

            let mut packet_buffer = core::mem::take(&mut self.packet_buffer);
            if packet_buffer.len() != MAX_PACKET_SIZE {
                packet_buffer.resize(MAX_PACKET_SIZE, 0);
            }
            let packet_len = {
                let frame_size = self.frame_size;
                let backend = self.backend_mut()?;
                backend.encode_float(&frame_buffer, frame_size, &mut packet_buffer)?
            };

            if let Some(prev) = previous_prediction {
                self.backend_mut()?.set_prediction_disabled(prev)?;
            }

            self.curr_granule = self
                .curr_granule
                .checked_add(self.frame_size)
                .ok_or(OpeError::InternalError)?;
            self.pcm_buffer_start += self.frame_size;

            let packet_is_last = self.curr_granule >= active_end_granule;
            let granulepos = {
                let stream = self.streams.front().ok_or(OpeError::TooLate)?;
                if packet_is_last {
                    adjusted_granule(active_end_granule, stream.granule_offset)?
                } else {
                    adjusted_granule(self.curr_granule, stream.granule_offset)?
                }
            };

            if is_keyframe {
                self.chaining_keyframe = Some(packet_buffer[..packet_len].to_vec());
            } else if !packet_is_last {
                self.chaining_keyframe = None;
            }

            let commit_result =
                self.commit_packet(&packet_buffer[..packet_len], granulepos, packet_is_last);
            self.frame_buffer = frame_buffer;
            self.packet_buffer = packet_buffer;
            commit_result?;
            if packet_is_last {
                let chaining_keyframe = self.chaining_keyframe.take();
                self.finish_active_stream(active_end_granule, chaining_keyframe)?;
            } else {
                self.emit_pages(false)?;
            }

            if self.pcm_buffer_start >= self.frame_size * 8 {
                self.compact_pcm_buffer();
            }
        }

        if self.streams.is_empty() {
            self.compact_pcm_buffer();
        }
        Ok(())
    }

    fn flush_resampler(&mut self, target_frames: usize) -> Result<(), OpeError> {
        let mut resampler_output_buffer = core::mem::take(&mut self.resampler_output_buffer);
        let mut resampler_zero_buffer = core::mem::take(&mut self.resampler_zero_buffer);
        let Some(resampler) = &mut self.resampler else {
            self.resampler_output_buffer = resampler_output_buffer;
            self.resampler_zero_buffer = resampler_zero_buffer;
            return Ok(());
        };
        while self.resampled_granule < target_frames {
            let remaining = target_frames - self.resampled_granule;
            let zero_in_frames = (((remaining as u64) * self.rate as u64 + 47_999) / 48_000) as usize
                + resampler.input_latency() as usize
                + 1;
            let zero_in_samples = zero_in_frames * self.channels;
            if resampler_zero_buffer.len() < zero_in_samples {
                resampler_zero_buffer.resize(zero_in_samples, 0.0);
            }
            let out_capacity_frames = (((zero_in_frames as u64) * 48_000 + self.rate as u64 - 1)
                / self.rate as u64) as usize
                + resampler.output_latency() as usize
                + 16;
            let out_capacity_samples = out_capacity_frames * self.channels;
            if resampler_output_buffer.len() < out_capacity_samples {
                resampler_output_buffer.resize(out_capacity_samples, 0.0);
            }
            let mut in_len = zero_in_frames as u32;
            let mut out_len = out_capacity_frames as u32;
            let resample_result = resampler
                .process_interleaved_float(
                    Some(&resampler_zero_buffer[..zero_in_samples]),
                    &mut in_len,
                    &mut resampler_output_buffer,
                    &mut out_len,
                )
                .map_err(|_| OpeError::InternalError);
            if let Err(err) = resample_result {
                self.resampler_output_buffer = resampler_output_buffer;
                self.resampler_zero_buffer = resampler_zero_buffer;
                return Err(err);
            }
            if in_len == 0 && out_len == 0 {
                self.resampler_output_buffer = resampler_output_buffer;
                self.resampler_zero_buffer = resampler_zero_buffer;
                return Err(OpeError::InternalError);
            }
            let take_frames = remaining.min(out_len as usize);
            self.pcm_buffer
                .extend_from_slice(&resampler_output_buffer[..take_frames * self.channels]);
            self.resampled_granule += take_frames;
        }
        self.resampler_output_buffer = resampler_output_buffer;
        self.resampler_zero_buffer = resampler_zero_buffer;
        Ok(())
    }

    fn pad_for_drain(&mut self) -> Result<(), OpeError> {
        let resampler_drain = self
            .resampler
            .as_ref()
            .map_or(0usize, |resampler| resampler.output_latency() as usize);
        let pad_frames = LPC_PADDING.max(
            self.global_granule_offset
                .unwrap_or(0)
                .checked_add(self.frame_size)
                .and_then(|value| value.checked_add(resampler_drain))
                .and_then(|value| value.checked_add(1))
                .ok_or(OpeError::InternalError)?,
        );
        self.pcm_buffer
            .resize(self.pcm_buffer.len() + pad_frames * self.channels, 0.0);
        self.resampled_granule = self
            .resampled_granule
            .checked_add(pad_frames)
            .ok_or(OpeError::InternalError)?;
        Ok(())
    }

    fn finish_active_stream(
        &mut self,
        end_granule: usize,
        chaining_keyframe: Option<Vec<u8>>,
    ) -> Result<(), OpeError> {
        if let Some(oggp) = &mut self.oggp {
            oggp.flush_page();
        }
        self.emit_pages(false)?;

        let completed = self.streams.pop_front().ok_or(OpeError::TooLate)?;
        let had_next = !self.streams.is_empty();
        if completed.close_at_end && had_next {
            self.close_output()?;
        }
        if let Some(next) = self.streams.front_mut() {
            if let Some(output) = next.next_output.take() {
                self.output = output;
            }
            next.preskip = end_granule
                .checked_add(self.frame_size)
                .and_then(|value| value.checked_sub(self.curr_granule))
                .ok_or(OpeError::InternalError)? as i32;
            next.granule_offset = self
                .curr_granule
                .checked_sub(self.frame_size)
                .ok_or(OpeError::InternalError)? as i64;
            if chaining_keyframe.is_some() {
                next.preskip = next
                    .preskip
                    .checked_add(self.frame_size as i32)
                    .ok_or(OpeError::InternalError)?;
                next.granule_offset -= self.frame_size as i64;
            }
            self.init_stream()?;
            if let Some(packet) = chaining_keyframe {
                let granulepos = self
                    .curr_granule
                    .checked_sub(self.frame_size)
                    .ok_or(OpeError::InternalError)?;
                let granulepos = adjusted_granule(
                    granulepos,
                    self.streams.front().ok_or(OpeError::TooLate)?.granule_offset,
                )?;
                self.commit_packet(&packet, granulepos, false)?;
            }
        }
        Ok(())
    }

    fn close_output(&mut self) -> Result<(), OpeError> {
        match &mut self.output {
            OutputMode::Callbacks(callbacks) => callbacks.close(),
            OutputMode::File(file) => file.flush().map_err(|_| OpeError::CloseFail),
            OutputMode::Pull => Ok(()),
        }
    }

    fn input_samples_to_granule(&self, samples: usize) -> Result<usize, OpeError> {
        samples
            .checked_mul(48_000)
            .and_then(|value| value.checked_add(self.rate as usize - 1))
            .map(|value| value / self.rate as usize)
            .ok_or(OpeError::InternalError)
    }

    fn generate_serialno(&mut self) -> i32 {
        let serialno = self.next_generated_serial;
        self.next_generated_serial = self.next_generated_serial.wrapping_add(1);
        serialno
    }
}

fn create_backend(channels: usize, family: i32) -> Result<EncoderBackend, OpeError> {
    match family {
        3 => {
            let (encoder, streams, coupled) =
                opus_projection_ambisonics_encoder_create(48_000, channels, family as u8, 2049)
                    .map_err(map_projection_error)?;
            let mut header = OpusHeader::default();
            header.nb_streams = streams as i32;
            header.nb_coupled = coupled as i32;
            let _ = header;
            Ok(EncoderBackend::Projection(encoder))
        }
        0 | 1 | 2 | 255 => {
            let (encoder, _) =
                opus_multistream_surround_encoder_create(48_000, channels, family as u8, 2049)
                    .map_err(map_multistream_error)?;
            Ok(EncoderBackend::Multistream(encoder))
        }
        _ => Err(OpeError::Unimplemented),
    }
}

fn fill_header_layout(header: &mut OpusHeader, backend: &EncoderBackend) {
    match backend {
        EncoderBackend::Multistream(encoder) => {
            header.nb_streams = encoder.layout().nb_streams as i32;
            header.nb_coupled = encoder.layout().nb_coupled_streams as i32;
            header.stream_map[..header.channels as usize]
                .copy_from_slice(&encoder.layout().mapping[..header.channels as usize]);
        }
        EncoderBackend::Projection(encoder) => {
            let layout = encoder.projection_layout();
            header.nb_streams = layout.streams as i32;
            header.nb_coupled = layout.coupled_streams as i32;
            for (index, value) in header
                .stream_map
                .iter_mut()
                .take(header.channels as usize)
                .enumerate()
            {
                *value = index as u8;
            }
        }
    }
}

fn map_multistream_error(error: OpusMultistreamEncoderError) -> OpeError {
    match error.code() {
        -1 => OpeError::BadArg,
        -3 => OpeError::InternalError,
        -5 => OpeError::Unimplemented,
        -7 => OpeError::AllocFail,
        _ => OpeError::InternalError,
    }
}

fn map_projection_error(error: OpusProjectionEncoderError) -> OpeError {
    match error {
        OpusProjectionEncoderError::BadArgument => OpeError::BadArg,
        OpusProjectionEncoderError::Unimplemented => OpeError::Unimplemented,
        OpusProjectionEncoderError::SizeOverflow => OpeError::InternalError,
        OpusProjectionEncoderError::Multistream(inner) => map_multistream_error(inner),
    }
}

fn adjusted_granule(base: usize, granule_offset: i64) -> Result<u64, OpeError> {
    let value = (base as i64)
        .checked_sub(granule_offset)
        .ok_or(OpeError::InternalError)?;
    u64::try_from(value).map_err(|_| OpeError::InternalError)
}

#[must_use]
pub fn get_version_string() -> &'static str {
    concat!("libopusenc ", env!("CARGO_PKG_VERSION"))
}

#[must_use]
pub const fn get_abi_version() -> i32 {
    OPE_ABI_VERSION
}

#[must_use]
pub const fn strerror(error: OpeError) -> &'static str {
    match error {
        OpeError::Ok => "success",
        OpeError::BadArg => "invalid argument",
        OpeError::InternalError => "internal error",
        OpeError::Unimplemented => "unimplemented",
        OpeError::AllocFail => "allocation failed",
        OpeError::CannotOpen => "cannot open file",
        OpeError::TooLate => "call cannot be made at this point",
        OpeError::InvalidPicture => "invalid picture file",
        OpeError::InvalidIcon => "invalid icon file (pictures of type 1 MUST be 32x32 PNGs)",
        OpeError::WriteFail => "write failed",
        OpeError::CloseFail => "close failed",
    }
}
