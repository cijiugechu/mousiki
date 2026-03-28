extern crate std;

use alloc::collections::VecDeque;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::libopusenc::error::LibopusencError;
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
    opus_multistream_encode_float, opus_multistream_encoder_create, opus_multistream_encoder_ctl,
    opus_multistream_surround_encoder_create,
};
use crate::projection::{
    OpusProjectionEncoder, OpusProjectionEncoderCtlRequest, OpusProjectionEncoderError,
    opus_projection_ambisonics_encoder_create, opus_projection_encode_float,
    opus_projection_encoder_ctl,
};

const FRAME_SIZE_20_MS: usize = 960;
const MAX_PACKET_SIZE: usize = 1277 * 6 * 255 + 2;
const MAX_LOOKAHEAD: u32 = 96_000;
const LPC_PADDING: usize = 120;

pub trait PacketHandler {
    fn on_packet(&mut self, packet: &[u8], flags: u32);
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NoPacketHandler;

impl PacketHandler for NoPacketHandler {
    fn on_packet(&mut self, _packet: &[u8], _flags: u32) {}
}

impl<F> PacketHandler for F
where
    F: FnMut(&[u8], u32),
{
    fn on_packet(&mut self, packet: &[u8], flags: u32) {
        self(packet, flags);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingFamily {
    MonoStereo,
    Surround,
    Ambisonics,
    Projection,
    Independent,
}

impl MappingFamily {
    const fn as_header_code(self) -> i32 {
        match self {
            Self::MonoStereo => 0,
            Self::Surround => 1,
            Self::Ambisonics => 2,
            Self::Projection => 3,
            Self::Independent => 255,
        }
    }

    const fn as_encoder_code(self) -> u8 {
        self.as_header_code() as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MuxingDelaySamples(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PictureType {
    Other = 0,
    FileIcon = 1,
    OtherFileIcon = 2,
    FrontCover = 3,
    BackCover = 4,
    Leaflet = 5,
    Media = 6,
    LeadArtist = 7,
    Artist = 8,
    Conductor = 9,
    Band = 10,
    Composer = 11,
    Lyricist = 12,
    RecordingLocation = 13,
    DuringRecording = 14,
    DuringPerformance = 15,
    VideoScreenCapture = 16,
    BrightColouredFish = 17,
    Illustration = 18,
    BandLogo = 19,
    PublisherLogo = 20,
}

impl PictureType {
    const fn as_i32(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OggOpusComments {
    comment: Vec<u8>,
    seen_file_icons: i32,
}

impl OggOpusComments {
    pub fn new() -> Result<Self, LibopusencError> {
        let vendor = format!("libopusenc {}", env!("CARGO_PKG_VERSION"));
        Ok(Self {
            comment: comment_init(&vendor)?,
            seen_file_icons: 0,
        })
    }

    pub fn add(&mut self, tag: &str, value: &str) -> Result<(), LibopusencError> {
        if tag.contains('=') {
            return Err(LibopusencError::InvalidArgument);
        }
        comment_add(&mut self.comment, Some(tag), value)?;
        Ok(())
    }

    pub fn add_string(&mut self, tag_and_value: &str) -> Result<(), LibopusencError> {
        if !tag_and_value.contains('=') {
            return Err(LibopusencError::InvalidArgument);
        }
        comment_add(&mut self.comment, None, tag_and_value)?;
        Ok(())
    }

    pub fn add_picture(
        &mut self,
        filename: &str,
        picture_type: PictureType,
        description: Option<&str>,
    ) -> Result<(), LibopusencError> {
        let picture = parse_picture_specification(
            filename,
            picture_type.as_i32(),
            description,
            &mut self.seen_file_icons,
        )?;
        comment_add(&mut self.comment, Some("METADATA_BLOCK_PICTURE"), &picture)?;
        Ok(())
    }

    pub fn add_picture_from_memory(
        &mut self,
        picture: &[u8],
        picture_type: PictureType,
        description: Option<&str>,
    ) -> Result<(), LibopusencError> {
        let picture = parse_picture_specification_from_memory(
            picture,
            picture_type.as_i32(),
            description,
            &mut self.seen_file_icons,
        )?;
        comment_add(&mut self.comment, Some("METADATA_BLOCK_PICTURE"), &picture)?;
        Ok(())
    }

    fn padded_bytes(&self, padding: u32) -> Result<Vec<u8>, LibopusencError> {
        let mut comment = self.comment.clone();
        comment_pad(&mut comment, padding as i32)?;
        Ok(comment)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExplicitMapping {
    streams: i32,
    coupled_streams: i32,
    mapping: Vec<u8>,
}

pub struct OggOpusEncoderBuilder<C = NoPacketHandler> {
    comments: OggOpusComments,
    sample_rate: u32,
    channels: u8,
    family: MappingFamily,
    explicit_mapping: Option<ExplicitMapping>,
    decision_delay: u32,
    comment_padding: u32,
    serialno: Option<i32>,
    header_gain: i16,
    muxing_delay: MuxingDelaySamples,
    packet_handler: C,
}

impl OggOpusEncoderBuilder<NoPacketHandler> {
    pub fn new(
        comments: OggOpusComments,
        sample_rate: u32,
        channels: u8,
        family: MappingFamily,
    ) -> Result<Self, LibopusencError> {
        if sample_rate == 0 || channels == 0 {
            return Err(LibopusencError::InvalidArgument);
        }
        Ok(Self {
            comments,
            sample_rate,
            channels,
            family,
            explicit_mapping: None,
            decision_delay: MAX_LOOKAHEAD,
            comment_padding: 512,
            serialno: None,
            header_gain: 0,
            muxing_delay: MuxingDelaySamples(48_000),
            packet_handler: NoPacketHandler,
        })
    }
}

impl<C: PacketHandler> OggOpusEncoderBuilder<C> {
    #[must_use]
    pub fn decision_delay(mut self, value: u32) -> Self {
        self.decision_delay = value.min(MAX_LOOKAHEAD);
        self
    }

    #[must_use]
    pub fn comment_padding(mut self, value: u32) -> Self {
        self.comment_padding = value;
        self
    }

    #[must_use]
    pub fn serialno(mut self, value: i32) -> Self {
        self.serialno = Some(value);
        self
    }

    #[must_use]
    pub fn header_gain(mut self, value: i16) -> Self {
        self.header_gain = value;
        self
    }

    #[must_use]
    pub fn muxing_delay(mut self, value: MuxingDelaySamples) -> Self {
        self.muxing_delay = value;
        self
    }

    #[must_use]
    pub fn packet_callback<N>(self, callback: N) -> OggOpusEncoderBuilder<N>
    where
        N: PacketHandler,
    {
        OggOpusEncoderBuilder {
            comments: self.comments,
            sample_rate: self.sample_rate,
            channels: self.channels,
            family: self.family,
            explicit_mapping: self.explicit_mapping,
            decision_delay: self.decision_delay,
            comment_padding: self.comment_padding,
            serialno: self.serialno,
            header_gain: self.header_gain,
            muxing_delay: self.muxing_delay,
            packet_handler: callback,
        }
    }

    pub fn mapping(
        mut self,
        streams: u8,
        coupled_streams: u8,
        mapping: &[u8],
    ) -> Result<Self, LibopusencError> {
        if !matches!(
            self.family,
            MappingFamily::Surround | MappingFamily::Ambisonics | MappingFamily::Independent
        ) {
            return Err(LibopusencError::InvalidArgument);
        }
        if streams == 0
            || coupled_streams >= 128
            || u16::from(streams) + u16::from(coupled_streams) > 255
            || usize::from(self.channels) != mapping.len()
            || usize::from(self.channels) != usize::from(streams) + usize::from(coupled_streams)
        {
            return Err(LibopusencError::InvalidArgument);
        }
        self.explicit_mapping = Some(ExplicitMapping {
            streams: i32::from(streams),
            coupled_streams: i32::from(coupled_streams),
            mapping: mapping.to_vec(),
        });
        Ok(self)
    }

    pub fn build_writer<W: Write>(self, writer: W) -> Result<OggOpusEncoder<W, C>, LibopusencError> {
        Ok(OggOpusEncoder {
            core: EncoderCore::new(self, WriterOutput::new(writer))?,
        })
    }

    pub fn build_file(
        self,
        path: impl AsRef<Path>,
    ) -> Result<OggOpusEncoder<BufWriter<File>, C>, LibopusencError> {
        let writer = BufWriter::new(File::create(path).map_err(LibopusencError::Io)?);
        self.build_writer(writer)
    }

    pub fn build_pull(self) -> Result<OggOpusPullEncoder<C>, LibopusencError> {
        Ok(OggOpusPullEncoder {
            core: EncoderCore::new(self, PullOutput::default())?,
        })
    }
}

pub struct OggOpusEncoder<W: Write, C: PacketHandler = NoPacketHandler> {
    core: EncoderCore<WriterOutput<W>, C>,
}

impl<W: Write, C: PacketHandler> OggOpusEncoder<W, C> {
    pub fn flush_headers(&mut self) -> Result<(), LibopusencError> {
        self.core.flush_headers()
    }

    pub fn write(&mut self, pcm: &[i16], samples_per_channel: usize) -> Result<(), LibopusencError> {
        self.core.write(pcm, samples_per_channel)
    }

    pub fn write_float(
        &mut self,
        pcm: &[f32],
        samples_per_channel: usize,
    ) -> Result<(), LibopusencError> {
        self.core.write_float(pcm, samples_per_channel)
    }

    pub fn start_next_stream(
        &mut self,
        comments: OggOpusComments,
    ) -> Result<(), LibopusencError> {
        self.core.start_next_stream(comments)
    }

    pub fn finish(mut self) -> Result<W, LibopusencError> {
        self.core.finish()?;
        Ok(self.core.output.into_inner())
    }
}

pub struct OggOpusPullEncoder<C: PacketHandler = NoPacketHandler> {
    core: EncoderCore<PullOutput, C>,
}

impl<C: PacketHandler> OggOpusPullEncoder<C> {
    pub fn flush_headers(&mut self) -> Result<(), LibopusencError> {
        self.core.flush_headers()
    }

    pub fn write(&mut self, pcm: &[i16], samples_per_channel: usize) -> Result<(), LibopusencError> {
        self.core.write(pcm, samples_per_channel)
    }

    pub fn write_float(
        &mut self,
        pcm: &[f32],
        samples_per_channel: usize,
    ) -> Result<(), LibopusencError> {
        self.core.write_float(pcm, samples_per_channel)
    }

    pub fn start_next_stream(
        &mut self,
        comments: OggOpusComments,
    ) -> Result<(), LibopusencError> {
        self.core.start_next_stream(comments)
    }

    pub fn finish(&mut self) -> Result<(), LibopusencError> {
        self.core.finish()
    }

    pub fn flush_pages(&mut self) -> Result<(), LibopusencError> {
        self.core.flush_pages()
    }

    pub fn next_page(&mut self) -> Result<Option<Vec<u8>>, LibopusencError> {
        self.flush_pages()?;
        Ok(self.core.output.next_page())
    }
}

trait OutputTarget {
    fn write_page(&mut self, page: Vec<u8>) -> Result<(), LibopusencError>;
    fn finish(&mut self) -> Result<(), LibopusencError>;
}

struct WriterOutput<W: Write> {
    writer: Option<W>,
}

impl<W: Write> WriterOutput<W> {
    fn new(writer: W) -> Self {
        Self {
            writer: Some(writer),
        }
    }

    fn writer_mut(&mut self) -> Result<&mut W, LibopusencError> {
        self.writer.as_mut().ok_or(LibopusencError::Internal)
    }

    fn into_inner(mut self) -> W {
        self.writer.take().expect("writer still present after finish")
    }
}

impl<W: Write> OutputTarget for WriterOutput<W> {
    fn write_page(&mut self, page: Vec<u8>) -> Result<(), LibopusencError> {
        self.writer_mut()?.write_all(&page).map_err(LibopusencError::Io)
    }

    fn finish(&mut self) -> Result<(), LibopusencError> {
        self.writer_mut()?.flush().map_err(LibopusencError::Io)
    }
}

#[derive(Default)]
struct PullOutput {
    pending_pages: VecDeque<Vec<u8>>,
}

impl PullOutput {
    fn next_page(&mut self) -> Option<Vec<u8>> {
        self.pending_pages.pop_front()
    }
}

impl OutputTarget for PullOutput {
    fn write_page(&mut self, page: Vec<u8>) -> Result<(), LibopusencError> {
        self.pending_pages.push_back(page);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), LibopusencError> {
        Ok(())
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
}

impl EncStream {
    fn new(comments: OggOpusComments, serialno: Option<i32>, header_gain: i16) -> Self {
        Self {
            comments,
            serialno,
            stream_is_init: false,
            header_is_frozen: false,
            header_gain: i32::from(header_gain),
            preskip: 0,
            end_granule: 0,
            granule_offset: 0,
        }
    }
}

#[derive(Debug)]
enum EncoderBackend {
    Multistream(OpusMultistreamEncoder<'static>),
    Projection(OpusProjectionEncoder<'static>),
}

impl EncoderBackend {
    fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        data: &mut [u8],
    ) -> Result<usize, LibopusencError> {
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

    fn lookahead(&mut self) -> Result<i32, LibopusencError> {
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

    fn set_prediction_disabled(&mut self, value: bool) -> Result<(), LibopusencError> {
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

    fn prediction_disabled(&mut self) -> Result<bool, LibopusencError> {
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

    fn set_expert_frame_duration(&mut self, value: i32) -> Result<(), LibopusencError> {
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

struct EncoderCore<O: OutputTarget, C: PacketHandler> {
    backend: EncoderBackend,
    oggp: Option<OggPacker>,
    output: O,
    streams: VecDeque<EncStream>,
    header: OpusHeader,
    rate: u32,
    channels: usize,
    frame_size: usize,
    decision_delay: u32,
    comment_padding: u32,
    max_ogg_delay: u32,
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
    packet_handler: C,
    chaining_keyframe: Option<Vec<u8>>,
    next_generated_serial: i32,
}

impl<O: OutputTarget, C: PacketHandler> EncoderCore<O, C> {
    fn new(builder: OggOpusEncoderBuilder<C>, output: O) -> Result<Self, LibopusencError> {
        let explicit_mapping = builder.explicit_mapping.clone();
        let mut backend = create_backend(
            usize::from(builder.channels),
            builder.family,
            explicit_mapping.as_ref(),
        )?;
        backend.set_expert_frame_duration(5004)?;

        let mut header = OpusHeader {
            channels: i32::from(builder.channels),
            input_sample_rate: builder.sample_rate,
            channel_mapping: builder.family.as_header_code(),
            ..Default::default()
        };
        fill_header_layout(&mut header, &backend, explicit_mapping.as_ref());

        let resampler = if builder.sample_rate == 48_000 {
            None
        } else {
            let mut resampler =
                SpeexResampler::new(builder.channels as u32, builder.sample_rate, 48_000, 5)
                    .map_err(|_| LibopusencError::InvalidArgument)?;
            resampler.skip_zeros().map_err(|_| LibopusencError::Internal)?;
            Some(resampler)
        };

        let mut streams = VecDeque::new();
        streams.push_back(EncStream::new(
            builder.comments,
            builder.serialno,
            builder.header_gain,
        ));

        Ok(Self {
            backend,
            oggp: None,
            output,
            streams,
            header,
            rate: builder.sample_rate,
            channels: usize::from(builder.channels),
            frame_size: FRAME_SIZE_20_MS,
            decision_delay: builder.decision_delay,
            comment_padding: builder.comment_padding,
            max_ogg_delay: builder.muxing_delay.0,
            pcm_buffer: Vec::new(),
            pcm_buffer_start: 0,
            input_buffer: Vec::new(),
            frame_buffer: vec![0.0; FRAME_SIZE_20_MS * usize::from(builder.channels)],
            packet_buffer: vec![0; MAX_PACKET_SIZE],
            resampler_output_buffer: Vec::new(),
            resampler_zero_buffer: Vec::new(),
            resampler,
            write_granule: 0,
            resampled_granule: 0,
            curr_granule: 0,
            global_granule_offset: None,
            drained: false,
            packet_handler: builder.packet_handler,
            chaining_keyframe: None,
            next_generated_serial: 0,
        })
    }

    fn flush_headers(&mut self) -> Result<(), LibopusencError> {
        self.ensure_can_write()?;
        let current = self.streams.front().ok_or(LibopusencError::InvalidState)?;
        if current.header_is_frozen || current.stream_is_init {
            return Err(LibopusencError::InvalidState);
        }
        self.init_stream()
    }

    fn write(&mut self, pcm: &[i16], samples_per_channel: usize) -> Result<(), LibopusencError> {
        self.ensure_can_write()?;
        if pcm.len() < samples_per_channel.saturating_mul(self.channels) {
            return Err(LibopusencError::InvalidArgument);
        }
        let last = self
            .streams
            .back_mut()
            .ok_or(LibopusencError::InvalidState)?;
        last.header_is_frozen = true;
        last.end_granule = self.write_granule.saturating_add(samples_per_channel);
        if !self
            .streams
            .front()
            .is_some_and(|stream| stream.stream_is_init)
        {
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

    fn write_float(
        &mut self,
        pcm: &[f32],
        samples_per_channel: usize,
    ) -> Result<(), LibopusencError> {
        self.ensure_can_write()?;
        if pcm.len() < samples_per_channel.saturating_mul(self.channels) {
            return Err(LibopusencError::InvalidArgument);
        }
        let last = self
            .streams
            .back_mut()
            .ok_or(LibopusencError::InvalidState)?;
        last.header_is_frozen = true;
        last.end_granule = self.write_granule.saturating_add(samples_per_channel);
        if !self
            .streams
            .front()
            .is_some_and(|stream| stream.stream_is_init)
        {
            self.init_stream()?;
        }
        self.write_granule += samples_per_channel;
        self.append_pcm(
            &pcm[..samples_per_channel * self.channels],
            samples_per_channel,
        )?;
        self.process_ready_packets(false)
    }

    fn start_next_stream(&mut self, comments: OggOpusComments) -> Result<(), LibopusencError> {
        self.ensure_can_write()?;
        let last = self
            .streams
            .back()
            .ok_or(LibopusencError::InvalidState)?;
        if !last.stream_is_init && !last.header_is_frozen {
            return Err(LibopusencError::InvalidState);
        }
        let mut stream = EncStream::new(comments, None, last.header_gain as i16);
        stream.end_granule = self.write_granule;
        self.streams.push_back(stream);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), LibopusencError> {
        if self.drained {
            return Err(LibopusencError::InvalidState);
        }
        if self.streams.is_empty() {
            return Err(LibopusencError::InvalidState);
        }
        if !self
            .streams
            .front()
            .is_some_and(|stream| stream.stream_is_init)
        {
            self.init_stream()?;
        }

        let target_frames = self.input_samples_to_granule(self.write_granule)?;
        if self.resampled_granule < target_frames {
            self.flush_resampler(target_frames)?;
        }
        self.pad_for_drain()?;
        self.process_ready_packets(true)?;
        if !self.streams.is_empty() {
            return Err(LibopusencError::Internal);
        }
        self.pcm_buffer.clear();
        self.pcm_buffer_start = 0;
        self.emit_pages(false)?;
        self.output.finish()?;
        self.drained = true;
        Ok(())
    }

    fn flush_pages(&mut self) -> Result<(), LibopusencError> {
        if let Some(oggp) = &mut self.oggp {
            oggp.flush_page();
        }
        self.emit_pages(false)
    }

    fn ensure_can_write(&self) -> Result<(), LibopusencError> {
        if self.drained {
            Err(LibopusencError::InvalidState)
        } else {
            Ok(())
        }
    }

    fn init_stream(&mut self) -> Result<(), LibopusencError> {
        if self
            .streams
            .front()
            .ok_or(LibopusencError::InvalidState)?
            .stream_is_init
        {
            return Ok(());
        }
        if self
            .streams
            .front()
            .is_some_and(|stream| stream.serialno.is_none())
        {
            let serialno = self.generate_serialno();
            if let Some(current) = self.streams.front_mut() {
                current.serialno = Some(serialno);
            }
        }

        let serialno = self
            .streams
            .front()
            .and_then(|stream| stream.serialno)
            .expect("serial initialised");
        if let Some(oggp) = &mut self.oggp {
            oggp.chain(serialno);
        } else {
            let mut oggp = OggPacker::new(serialno);
            oggp.set_muxing_delay(self.max_ogg_delay as u64);
            self.oggp = Some(oggp);
        }

        if self.global_granule_offset.is_none() {
            let preskip = self.backend.lookahead()?;
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
        let packet_size = match &self.backend {
            EncoderBackend::Projection(encoder) => {
                opus_header_to_packet(&header, &mut header_packet, Some(encoder.projection_layout()))
            }
            _ => opus_header_to_packet(&header, &mut header_packet, None),
        }
        .map_err(|_| LibopusencError::Internal)?;
        header_packet.truncate(packet_size);
        self.commit_packet(&header_packet, 0, false)?;
        self.flush_current_page()?;

        let comments = self
            .streams
            .front()
            .ok_or(LibopusencError::InvalidState)?
            .comments
            .padded_bytes(self.comment_padding)?;
        self.commit_packet(&comments, 0, false)?;
        self.flush_current_page()?;

        if let Some(current) = self.streams.front_mut() {
            current.stream_is_init = true;
        }
        Ok(())
    }

    fn commit_packet(
        &mut self,
        packet: &[u8],
        granulepos: u64,
        eos: bool,
    ) -> Result<(), LibopusencError> {
        self.packet_handler.on_packet(packet, 0);
        let oggp = self.oggp.as_mut().ok_or(LibopusencError::Internal)?;
        let buffer = oggp
            .get_packet_buffer(packet.len())
            .ok_or(LibopusencError::Internal)?;
        buffer.copy_from_slice(packet);
        oggp.commit_packet(packet.len(), granulepos, eos)
            .map_err(|_| LibopusencError::Internal)?;
        Ok(())
    }

    fn flush_current_page(&mut self) -> Result<(), LibopusencError> {
        if let Some(oggp) = &mut self.oggp {
            oggp.flush_page();
        }
        self.emit_pages(false)
    }

    fn emit_pages(&mut self, flush: bool) -> Result<(), LibopusencError> {
        if let Some(oggp) = &mut self.oggp
            && flush
        {
            oggp.flush_page();
        }
        while let Some(page) = self.oggp.as_mut().and_then(OggPacker::get_next_page) {
            self.output.write_page(page)?;
        }
        Ok(())
    }

    fn append_pcm(
        &mut self,
        pcm: &[f32],
        samples_per_channel: usize,
    ) -> Result<(), LibopusencError> {
        let mut resampler_output_buffer = core::mem::take(&mut self.resampler_output_buffer);
        if let Some(resampler) = &mut self.resampler {
            let mut start = 0usize;
            let mut remaining = samples_per_channel;
            while remaining > 0 {
                let out_capacity_frames = ((remaining as u64) * 48_000)
                    .div_ceil(self.rate as u64) as usize
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
                    .map_err(|_| LibopusencError::Internal);
                if let Err(err) = resample_result {
                    self.resampler_output_buffer = resampler_output_buffer;
                    return Err(err);
                }
                if in_len == 0 && out_len == 0 {
                    self.resampler_output_buffer = resampler_output_buffer;
                    return Err(LibopusencError::Internal);
                }
                self.pcm_buffer.extend_from_slice(
                    &resampler_output_buffer[..out_len as usize * self.channels],
                );
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
        self.pcm_buffer
            .truncate(self.pcm_buffer.len() - consumed_samples);
        self.pcm_buffer_start = 0;
    }

    fn process_ready_packets(&mut self, draining: bool) -> Result<(), LibopusencError> {
        let decision_delay = if draining {
            0
        } else {
            self.decision_delay as usize
        };

        while !self.streams.is_empty() && self.buffered_frames() > self.frame_size + decision_delay
        {
            let active_end_input = self
                .streams
                .front()
                .ok_or(LibopusencError::InvalidState)?
                .end_granule;
            let active_end_audio = self.input_samples_to_granule(active_end_input)?;
            let active_end_granule = active_end_audio
                .checked_add(self.global_granule_offset.unwrap_or(0))
                .ok_or(LibopusencError::Internal)?;
            let has_next_stream = self.streams.len() > 1;
            let is_keyframe = has_next_stream
                && self
                    .curr_granule
                    .saturating_add(self.frame_size.saturating_mul(2))
                    >= active_end_granule;

            let previous_prediction = if is_keyframe {
                let prev = self.backend.prediction_disabled()?;
                self.backend.set_prediction_disabled(true)?;
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
            let packet_len = self
                .backend
                .encode_float(&frame_buffer, self.frame_size, &mut packet_buffer)?;

            if let Some(prev) = previous_prediction {
                self.backend.set_prediction_disabled(prev)?;
            }

            self.curr_granule = self
                .curr_granule
                .checked_add(self.frame_size)
                .ok_or(LibopusencError::Internal)?;
            self.pcm_buffer_start += self.frame_size;

            let packet_is_last = self.curr_granule >= active_end_granule;
            let granulepos = {
                let stream = self
                    .streams
                    .front()
                    .ok_or(LibopusencError::InvalidState)?;
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

    fn flush_resampler(&mut self, target_frames: usize) -> Result<(), LibopusencError> {
        let mut resampler_output_buffer = core::mem::take(&mut self.resampler_output_buffer);
        let mut resampler_zero_buffer = core::mem::take(&mut self.resampler_zero_buffer);
        let Some(resampler) = &mut self.resampler else {
            self.resampler_output_buffer = resampler_output_buffer;
            self.resampler_zero_buffer = resampler_zero_buffer;
            return Ok(());
        };
        while self.resampled_granule < target_frames {
            let remaining = target_frames - self.resampled_granule;
            let zero_in_frames = ((remaining as u64) * self.rate as u64).div_ceil(48_000)
                as usize
                + resampler.input_latency() as usize
                + 1;
            let zero_in_samples = zero_in_frames * self.channels;
            if resampler_zero_buffer.len() < zero_in_samples {
                resampler_zero_buffer.resize(zero_in_samples, 0.0);
            }
            let out_capacity_frames = ((zero_in_frames as u64) * 48_000)
                .div_ceil(self.rate as u64) as usize
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
                .map_err(|_| LibopusencError::Internal);
            if let Err(err) = resample_result {
                self.resampler_output_buffer = resampler_output_buffer;
                self.resampler_zero_buffer = resampler_zero_buffer;
                return Err(err);
            }
            if in_len == 0 && out_len == 0 {
                self.resampler_output_buffer = resampler_output_buffer;
                self.resampler_zero_buffer = resampler_zero_buffer;
                return Err(LibopusencError::Internal);
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

    fn pad_for_drain(&mut self) -> Result<(), LibopusencError> {
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
                .ok_or(LibopusencError::Internal)?,
        );
        self.pcm_buffer
            .resize(self.pcm_buffer.len() + pad_frames * self.channels, 0.0);
        self.resampled_granule = self
            .resampled_granule
            .checked_add(pad_frames)
            .ok_or(LibopusencError::Internal)?;
        Ok(())
    }

    fn finish_active_stream(
        &mut self,
        end_granule: usize,
        chaining_keyframe: Option<Vec<u8>>,
    ) -> Result<(), LibopusencError> {
        if let Some(oggp) = &mut self.oggp {
            oggp.flush_page();
        }
        self.emit_pages(false)?;

        self.streams
            .pop_front()
            .ok_or(LibopusencError::InvalidState)?;
        if let Some(next) = self.streams.front_mut() {
            next.preskip = end_granule
                .checked_add(self.frame_size)
                .and_then(|value| value.checked_sub(self.curr_granule))
                .ok_or(LibopusencError::Internal)? as i32;
            next.granule_offset = self
                .curr_granule
                .checked_sub(self.frame_size)
                .ok_or(LibopusencError::Internal)? as i64;
            if chaining_keyframe.is_some() {
                next.preskip = next
                    .preskip
                    .checked_add(self.frame_size as i32)
                    .ok_or(LibopusencError::Internal)?;
                next.granule_offset -= self.frame_size as i64;
            }
            self.init_stream()?;
            if let Some(packet) = chaining_keyframe {
                let granulepos = self
                    .curr_granule
                    .checked_sub(self.frame_size)
                    .ok_or(LibopusencError::Internal)?;
                let granulepos = adjusted_granule(
                    granulepos,
                    self.streams
                        .front()
                        .ok_or(LibopusencError::InvalidState)?
                        .granule_offset,
                )?;
                self.commit_packet(&packet, granulepos, false)?;
            }
        }
        Ok(())
    }

    fn input_samples_to_granule(&self, samples: usize) -> Result<usize, LibopusencError> {
        samples
            .checked_mul(48_000)
            .and_then(|value| value.checked_add(self.rate as usize - 1))
            .map(|value| value / self.rate as usize)
            .ok_or(LibopusencError::Internal)
    }

    fn generate_serialno(&mut self) -> i32 {
        let serialno = self.next_generated_serial;
        self.next_generated_serial = self.next_generated_serial.wrapping_add(1);
        serialno
    }
}

fn create_backend(
    channels: usize,
    family: MappingFamily,
    explicit_mapping: Option<&ExplicitMapping>,
) -> Result<EncoderBackend, LibopusencError> {
    if let Some(explicit) = explicit_mapping {
        return match family {
            MappingFamily::Surround | MappingFamily::Ambisonics | MappingFamily::Independent => {
                let encoder = opus_multistream_encoder_create(
                    48_000,
                    channels,
                    explicit.streams as usize,
                    explicit.coupled_streams as usize,
                    &explicit.mapping,
                    2049,
                )
                .map_err(map_multistream_error)?;
                Ok(EncoderBackend::Multistream(encoder))
            }
            _ => Err(LibopusencError::InvalidArgument),
        };
    }

    match family {
        MappingFamily::Projection => {
            let (encoder, _, _) =
                opus_projection_ambisonics_encoder_create(48_000, channels, 3, 2049)
                    .map_err(map_projection_error)?;
            Ok(EncoderBackend::Projection(encoder))
        }
        MappingFamily::MonoStereo
        | MappingFamily::Surround
        | MappingFamily::Ambisonics
        | MappingFamily::Independent => {
            let (encoder, _) = opus_multistream_surround_encoder_create(
                48_000,
                channels,
                family.as_encoder_code(),
                2049,
            )
            .map_err(map_multistream_error)?;
            Ok(EncoderBackend::Multistream(encoder))
        }
    }
}

fn fill_header_layout(
    header: &mut OpusHeader,
    backend: &EncoderBackend,
    explicit_mapping: Option<&ExplicitMapping>,
) {
    if let Some(explicit) = explicit_mapping {
        header.nb_streams = explicit.streams;
        header.nb_coupled = explicit.coupled_streams;
        header.stream_map.fill(0);
        header.stream_map[..explicit.mapping.len()].copy_from_slice(&explicit.mapping);
        return;
    }

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

fn map_multistream_error(error: OpusMultistreamEncoderError) -> LibopusencError {
    match error.code() {
        -1 => LibopusencError::InvalidArgument,
        -5 => LibopusencError::Unsupported,
        _ => LibopusencError::Internal,
    }
}

fn map_projection_error(error: OpusProjectionEncoderError) -> LibopusencError {
    match error {
        OpusProjectionEncoderError::BadArgument => LibopusencError::InvalidArgument,
        OpusProjectionEncoderError::Unimplemented => LibopusencError::Unsupported,
        OpusProjectionEncoderError::SizeOverflow => LibopusencError::Internal,
        OpusProjectionEncoderError::Multistream(inner) => map_multistream_error(inner),
    }
}

impl From<CommentError> for LibopusencError {
    fn from(_: CommentError) -> Self {
        Self::Internal
    }
}

fn adjusted_granule(base: usize, granule_offset: i64) -> Result<u64, LibopusencError> {
    let value = (base as i64)
        .checked_sub(granule_offset)
        .ok_or(LibopusencError::Internal)?;
    u64::try_from(value).map_err(|_| LibopusencError::Internal)
}

#[must_use]
pub fn get_version_string() -> &'static str {
    concat!("libopusenc ", env!("CARGO_PKG_VERSION"))
}
