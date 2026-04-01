extern crate std;

use alloc::{vec, vec::Vec};
use std::fs;
use std::io::Read;
use std::path::Path;

use crate::ogg::Page;
use crate::ogg::StreamDecoder as OggStreamDecoder;
use crate::opus_multistream::{
    OpusMultistreamDecoderError, opus_multistream_decode_float, opus_multistream_decoder_create,
};
use crate::opusfile::{OpusHead, OpusTags};
use crate::packet::{opus_packet_get_nb_samples, PacketError};
use crate::opusfile::{OpusfileError, OpusfileOpenError};

const DOWNMIX_SQRT_HALF: f32 = core::f32::consts::FRAC_1_SQRT_2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainType {
    Header,
    Album,
    Track,
    Absolute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadSamples {
    pub samples_per_channel: usize,
    pub link_index: usize,
}

#[derive(Debug, Clone)]
struct PacketIndex {
    raw_offset: usize,
    pcm_offset: usize,
}

#[derive(Debug, Clone)]
struct AudioPacket {
    data: Vec<u8>,
    granule_position: i64,
    raw_offset: usize,
}

#[derive(Debug, Clone)]
struct Link {
    serialno: u32,
    head: OpusHead,
    tags: OpusTags,
    raw_start: usize,
    raw_end: usize,
    pcm_start: usize,
    pcm_end: usize,
    pcm: Vec<f32>,
    packet_index: Vec<PacketIndex>,
}

impl Link {
    #[must_use]
    fn channel_count(&self) -> usize {
        usize::from(self.head.channel_count)
    }

    #[must_use]
    fn pcm_len(&self) -> usize {
        self.pcm_end.saturating_sub(self.pcm_start)
    }
}

#[derive(Debug)]
struct PageRecord {
    offset: usize,
    end_offset: usize,
    page: Page,
}

#[derive(Debug)]
struct LinkBuilder {
    serialno: u32,
    head: OpusHead,
    tags: Option<OpusTags>,
    decoder: OggStreamDecoder,
    raw_start: usize,
    last_page_end: usize,
    packets: Vec<AudioPacket>,
}

impl LinkBuilder {
    fn from_bos_page(record: &PageRecord) -> Result<Option<Self>, OpusfileError> {
        let serialno = record.page.stream_serial();
        let mut decoder = OggStreamDecoder::new(serialno);
        decoder
            .push_page(&record.page)
            .map_err(|_| OpusfileError::BadHeader)?;
        let Some(packet) = decoder
            .next_packet()
            .map_err(|_| OpusfileError::BadHeader)?
        else {
            return Err(OpusfileError::BadHeader);
        };
        let head = match OpusHead::parse(packet.data()) {
            Ok(head) => head,
            Err(OpusfileError::NotFormat) => return Ok(None),
            Err(err) => return Err(err),
        };
        let mut builder = Self {
            serialno: serialno as u32,
            head,
            tags: None,
            decoder,
            raw_start: record.offset,
            last_page_end: record.end_offset,
            packets: Vec::new(),
        };
        builder.drain_packets(record.offset)?;
        Ok(Some(builder))
    }

    fn push_page(&mut self, record: &PageRecord) -> Result<(), OpusfileError> {
        self.decoder
            .push_page(&record.page)
            .map_err(|_| OpusfileError::BadHeader)?;
        self.last_page_end = record.end_offset;
        self.drain_packets(record.offset)
    }

    fn drain_packets(&mut self, raw_offset: usize) -> Result<(), OpusfileError> {
        while let Some(packet) = self
            .decoder
            .next_packet()
            .map_err(|_| OpusfileError::BadHeader)?
        {
            if self.tags.is_none() {
                self.tags = Some(OpusTags::parse(packet.data())?);
                continue;
            }
            let granule_position = packet.granule_position();
            self.packets.push(AudioPacket {
                data: packet.into_data(),
                granule_position,
                raw_offset,
            });
        }
        Ok(())
    }

    fn finish(self, pcm_start: usize) -> Result<Link, OpusfileError> {
        let Some(tags) = self.tags else {
            return Err(OpusfileError::BadHeader);
        };
        decode_link(
            self.serialno,
            self.head,
            tags,
            self.raw_start,
            self.last_page_end,
            pcm_start,
            &self.packets,
        )
    }
}

/// Rust-first buffered Ogg Opus reader.
///
/// This implementation buffers the full source in memory during open so it can
/// expose simple `read_*` and `seek_*` operations on top of the existing
/// `mousiki-ogg` and multistream decoder layers.
#[derive(Debug, Clone)]
pub struct OpusFile {
    links: Vec<Link>,
    raw_total: usize,
    pcm_cursor: usize,
    gain_type: GainType,
    gain_offset_q8: i32,
    dither_enabled: bool,
    bytes_tracked: usize,
    samples_tracked: usize,
}

impl OpusFile {
    pub fn open_memory(data: &[u8]) -> Result<Self, OpusfileError> {
        let (mut links, raw_total) = parse_links(data)?;
        if links.is_empty() {
            return Err(OpusfileError::NotFormat);
        }
        for index in 0..links.len().saturating_sub(1) {
            links[index].raw_end = links[index + 1].raw_start;
        }
        if let Some(last) = links.last_mut() {
            last.raw_end = raw_total;
        }
        Ok(Self {
            links,
            raw_total,
            pcm_cursor: 0,
            gain_type: GainType::Header,
            gain_offset_q8: 0,
            dither_enabled: true,
            bytes_tracked: 0,
            samples_tracked: 0,
        })
    }

    pub fn open_reader<R: Read>(mut reader: R) -> Result<Self, OpusfileOpenError> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        Self::open_memory(&data).map_err(Into::into)
    }

    pub fn open_file(path: impl AsRef<Path>) -> Result<Self, OpusfileOpenError> {
        let data = fs::read(path)?;
        Self::open_memory(&data).map_err(Into::into)
    }

    #[must_use]
    pub fn is_seekable(&self) -> bool {
        true
    }

    #[must_use]
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    pub fn serialno(&self, link_index: Option<usize>) -> Result<u32, OpusfileError> {
        Ok(self.link(link_index)?.serialno)
    }

    pub fn channel_count(&self, link_index: Option<usize>) -> Result<usize, OpusfileError> {
        Ok(self.link(link_index)?.channel_count())
    }

    pub fn raw_total(&self, link_index: Option<usize>) -> Result<usize, OpusfileError> {
        match link_index {
            None => Ok(self.raw_total),
            Some(index) => Ok(self.link_by_index(index)?.raw_end - self.link_by_index(index)?.raw_start),
        }
    }

    pub fn pcm_total(&self, link_index: Option<usize>) -> Result<usize, OpusfileError> {
        match link_index {
            None => self
                .links
                .last()
                .map(|link| link.pcm_end)
                .ok_or(OpusfileError::BadArgument),
            Some(index) => Ok(self.link_by_index(index)?.pcm_len()),
        }
    }

    pub fn head(&self, link_index: Option<usize>) -> Result<&OpusHead, OpusfileError> {
        Ok(&self.link(link_index)?.head)
    }

    pub fn tags(&self, link_index: Option<usize>) -> Result<&OpusTags, OpusfileError> {
        Ok(&self.link(link_index)?.tags)
    }

    pub fn current_link(&self) -> Result<usize, OpusfileError> {
        self.link_index_for_pcm(self.pcm_cursor)
            .ok_or(OpusfileError::BadArgument)
    }

    #[must_use]
    pub fn raw_tell(&self) -> usize {
        self.raw_offset_for_pcm(self.pcm_cursor)
    }

    #[must_use]
    pub fn pcm_tell(&self) -> usize {
        self.pcm_cursor
    }

    pub fn raw_seek(&mut self, byte_offset: usize) -> Result<(), OpusfileError> {
        if byte_offset > self.raw_total {
            return Err(OpusfileError::BadArgument);
        }
        let target_pcm = self.pcm_for_raw_offset(byte_offset);
        self.reset_tracking();
        self.pcm_cursor = target_pcm;
        Ok(())
    }

    pub fn pcm_seek(&mut self, pcm_offset: usize) -> Result<(), OpusfileError> {
        let total = self.pcm_total(None)?;
        if pcm_offset > total {
            return Err(OpusfileError::BadArgument);
        }
        self.reset_tracking();
        self.pcm_cursor = pcm_offset;
        Ok(())
    }

    pub fn set_gain_offset(&mut self, gain_type: GainType, gain_offset_q8: i32) {
        self.gain_type = gain_type;
        self.gain_offset_q8 = gain_offset_q8.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    }

    #[must_use]
    pub fn gain_offset(&self) -> (GainType, i32) {
        (self.gain_type, self.gain_offset_q8)
    }

    pub fn set_dither_enabled(&mut self, enabled: bool) {
        self.dither_enabled = enabled;
    }

    #[must_use]
    pub fn dither_enabled(&self) -> bool {
        self.dither_enabled
    }

    pub fn bitrate(&self, link_index: Option<usize>) -> Result<i32, OpusfileError> {
        let bytes = self.raw_total(link_index)? as u64;
        let samples = self.pcm_total(link_index)? as u64;
        Ok(calculate_bitrate(bytes, samples))
    }

    pub fn bitrate_instant(&mut self) -> Result<i32, OpusfileError> {
        if self.samples_tracked == 0 {
            return Err(OpusfileError::False);
        }
        let bitrate = calculate_bitrate(self.bytes_tracked as u64, self.samples_tracked as u64);
        self.reset_tracking();
        Ok(bitrate)
    }

    pub fn read_float(&mut self, pcm: &mut [f32]) -> Result<ReadSamples, OpusfileError> {
        let link_index = self.current_link()?;
        let link = &self.links[link_index];
        let channels = link.channel_count();
        let available = link.pcm_end.saturating_sub(self.pcm_cursor);
        let samples = available.min(pcm.len() / channels);
        if samples == 0 {
            return Ok(ReadSamples {
                samples_per_channel: 0,
                link_index,
            });
        }
        let local_start = self.pcm_cursor - link.pcm_start;
        let gain = self.gain_scale(link);
        let src = &link.pcm[local_start * channels..(local_start + samples) * channels];
        for (dst, sample) in pcm[..samples * channels].iter_mut().zip(src.iter().copied()) {
            *dst = sample * gain;
        }
        self.advance_tracking(samples, self.pcm_cursor, self.pcm_cursor + samples);
        self.pcm_cursor += samples;
        Ok(ReadSamples {
            samples_per_channel: samples,
            link_index,
        })
    }

    pub fn read(&mut self, pcm: &mut [i16]) -> Result<ReadSamples, OpusfileError> {
        let link_index = self.current_link()?;
        let link = &self.links[link_index];
        let channels = link.channel_count();
        let available = link.pcm_end.saturating_sub(self.pcm_cursor);
        let samples = available.min(pcm.len() / channels);
        if samples == 0 {
            return Ok(ReadSamples {
                samples_per_channel: 0,
                link_index,
            });
        }
        let local_start = self.pcm_cursor - link.pcm_start;
        let gain = self.gain_scale(link);
        let src = &link.pcm[local_start * channels..(local_start + samples) * channels];
        for (dst, sample) in pcm[..samples * channels].iter_mut().zip(src.iter().copied()) {
            *dst = float_to_i16(sample * gain);
        }
        self.advance_tracking(samples, self.pcm_cursor, self.pcm_cursor + samples);
        self.pcm_cursor += samples;
        Ok(ReadSamples {
            samples_per_channel: samples,
            link_index,
        })
    }

    pub fn read_float_stereo(&mut self, pcm: &mut [f32]) -> Result<ReadSamples, OpusfileError> {
        let link_index = self.current_link()?;
        let link = &self.links[link_index];
        let available = link.pcm_end.saturating_sub(self.pcm_cursor);
        let samples = available.min(pcm.len() / 2);
        if samples == 0 {
            return Ok(ReadSamples {
                samples_per_channel: 0,
                link_index,
            });
        }
        let local_start = self.pcm_cursor - link.pcm_start;
        let gain = self.gain_scale(link);
        let channels = link.channel_count();
        let src = &link.pcm[local_start * channels..(local_start + samples) * channels];
        downmix_to_stereo(src, channels, &mut pcm[..samples * 2], gain);
        self.advance_tracking(samples, self.pcm_cursor, self.pcm_cursor + samples);
        self.pcm_cursor += samples;
        Ok(ReadSamples {
            samples_per_channel: samples,
            link_index,
        })
    }

    pub fn read_stereo(&mut self, pcm: &mut [i16]) -> Result<ReadSamples, OpusfileError> {
        let mut scratch = vec![0.0f32; pcm.len()];
        let result = self.read_float_stereo(&mut scratch)?;
        for (dst, sample) in pcm[..result.samples_per_channel * 2]
            .iter_mut()
            .zip(scratch.into_iter())
        {
            *dst = float_to_i16(sample);
        }
        Ok(result)
    }

    fn link(&self, link_index: Option<usize>) -> Result<&Link, OpusfileError> {
        match link_index {
            Some(index) => self.link_by_index(index),
            None => self
                .link_index_for_pcm(self.pcm_cursor)
                .and_then(|index| self.links.get(index))
                .ok_or(OpusfileError::BadArgument),
        }
    }

    fn link_by_index(&self, index: usize) -> Result<&Link, OpusfileError> {
        self.links.get(index).ok_or(OpusfileError::BadArgument)
    }

    fn link_index_for_pcm(&self, pcm_offset: usize) -> Option<usize> {
        self.links
            .iter()
            .position(|link| pcm_offset >= link.pcm_start && pcm_offset < link.pcm_end)
            .or_else(|| {
                (!self.links.is_empty() && pcm_offset == self.links.last()?.pcm_end)
                    .then_some(self.links.len() - 1)
            })
    }

    fn raw_offset_for_pcm(&self, pcm_offset: usize) -> usize {
        let Some(link_index) = self.link_index_for_pcm(pcm_offset) else {
            return self.raw_total;
        };
        let link = &self.links[link_index];
        let local_pcm = pcm_offset.saturating_sub(link.pcm_start);
        if local_pcm == 0 {
            return link.raw_start;
        }
        let mut raw = link.raw_start;
        for packet in &link.packet_index {
            if packet.pcm_offset <= local_pcm {
                raw = packet.raw_offset;
            } else {
                break;
            }
        }
        raw
    }

    fn pcm_for_raw_offset(&self, byte_offset: usize) -> usize {
        for link in &self.links {
            if byte_offset < link.raw_start {
                return link.pcm_start;
            }
            if byte_offset >= link.raw_end {
                continue;
            }
            let mut pcm = link.pcm_start;
            for packet in &link.packet_index {
                if packet.raw_offset <= byte_offset {
                    pcm = link.pcm_start + packet.pcm_offset;
                } else {
                    break;
                }
            }
            return pcm;
        }
        self.links.last().map_or(0, |link| link.pcm_end)
    }

    fn gain_scale(&self, link: &Link) -> f32 {
        let header_gain = i32::from(link.head.output_gain);
        let gain_q8 = match self.gain_type {
            GainType::Header => header_gain + self.gain_offset_q8,
            GainType::Album => header_gain
                + i32::from(link.tags.album_gain_q8().unwrap_or(0))
                + self.gain_offset_q8,
            GainType::Track => header_gain
                + i32::from(link.tags.track_gain_q8().unwrap_or(0))
                + self.gain_offset_q8,
            GainType::Absolute => self.gain_offset_q8,
        }
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX));
        libm::powf(10.0, gain_q8 as f32 / (20.0 * 256.0))
    }

    fn advance_tracking(&mut self, samples: usize, old_pcm: usize, new_pcm: usize) {
        let old_raw = self.raw_offset_for_pcm(old_pcm);
        let new_raw = self.raw_offset_for_pcm(new_pcm);
        self.bytes_tracked = self
            .bytes_tracked
            .saturating_add(new_raw.saturating_sub(old_raw));
        self.samples_tracked = self.samples_tracked.saturating_add(samples);
    }

    fn reset_tracking(&mut self) {
        self.bytes_tracked = 0;
        self.samples_tracked = 0;
    }
}

pub fn probe_opus_stream(initial_data: &[u8]) -> Result<OpusHead, OpusfileError> {
    if initial_data.len() < 47 {
        return Err(OpusfileError::False);
    }
    if initial_data.get(..4) != Some(b"OggS") {
        return Err(OpusfileError::NotFormat);
    }
    let pages = collect_pages(initial_data, true)?;
    for record in pages {
        if !record.page.is_beginning_of_stream() {
            return Err(OpusfileError::NotFormat);
        }
        if let Some(builder) = LinkBuilder::from_bos_page(&record)? {
            return Ok(builder.head);
        }
    }
    Err(OpusfileError::False)
}

pub fn probe_reader<R: Read>(mut reader: R) -> Result<OpusHead, OpusfileOpenError> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    probe_opus_stream(&data).map_err(Into::into)
}

pub fn probe_file(path: impl AsRef<Path>) -> Result<OpusHead, OpusfileOpenError> {
    let data = fs::read(path)?;
    probe_opus_stream(&data).map_err(Into::into)
}

fn parse_links(data: &[u8]) -> Result<(Vec<Link>, usize), OpusfileError> {
    let page_records = collect_pages(data, false)?;
    let raw_total = page_records.last().map_or(0, |record| record.end_offset);
    let mut links = Vec::new();
    let mut current = None::<LinkBuilder>;
    let mut pcm_start = 0usize;

    for record in page_records {
        if let Some(builder) = current.as_mut()
            && record.page.stream_serial() == builder.serialno as i32
        {
            builder.push_page(&record)?;
            if record.page.is_end_of_stream() {
                let link = current
                    .take()
                    .expect("active builder should exist")
                    .finish(pcm_start)?;
                pcm_start = link.pcm_end;
                links.push(link);
            }
            continue;
        }

        if current.is_none()
            && record.page.is_beginning_of_stream()
            && let Some(builder) = LinkBuilder::from_bos_page(&record)?
        {
            if record.page.is_end_of_stream() {
                let link = builder.finish(pcm_start)?;
                pcm_start = link.pcm_end;
                links.push(link);
            } else {
                current = Some(builder);
            }
        }
    }

    if current.is_some() {
        return Err(OpusfileError::BadLink);
    }
    if links.is_empty() {
        return Err(OpusfileError::NotFormat);
    }
    Ok((links, raw_total))
}

fn collect_pages(data: &[u8], partial_ok: bool) -> Result<Vec<PageRecord>, OpusfileError> {
    let mut pages = Vec::new();
    let mut cursor = 0usize;
    let mut saw_page = false;

    while cursor < data.len() {
        let remaining = data.len() - cursor;
        if remaining < 27 {
            if partial_ok && !saw_page {
                return Err(OpusfileError::False);
            }
            break;
        }
        if &data[cursor..cursor + 4] != b"OggS" {
            if saw_page {
                break;
            }
            return Err(OpusfileError::NotFormat);
        }
        let segment_count = usize::from(data[cursor + 26]);
        let header_len = 27 + segment_count;
        if remaining < header_len {
            return Err(if partial_ok { OpusfileError::False } else { OpusfileError::BadHeader });
        }
        let body_len: usize = data[cursor + 27..cursor + header_len]
            .iter()
            .map(|&value| usize::from(value))
            .sum();
        if remaining < header_len + body_len {
            return Err(if partial_ok { OpusfileError::False } else { OpusfileError::BadHeader });
        }
        let mut page = Page::from_parts(
            data[cursor..cursor + header_len].to_vec(),
            data[cursor + header_len..cursor + header_len + body_len].to_vec(),
        );
        if !page.is_checksum_valid() {
            return Err(OpusfileError::BadHeader);
        }
        pages.push(PageRecord {
            offset: cursor,
            end_offset: cursor + header_len + body_len,
            page: {
                page.update_checksum();
                page
            },
        });
        saw_page = true;
        cursor += header_len + body_len;
    }

    Ok(pages)
}

fn decode_link(
    serialno: u32,
    head: OpusHead,
    tags: OpusTags,
    raw_start: usize,
    raw_end: usize,
    pcm_start: usize,
    packets: &[AudioPacket],
) -> Result<Link, OpusfileError> {
    let channels = usize::from(head.channel_count);
    let mut decoder = opus_multistream_decoder_create(
        48_000,
        channels,
        usize::from(head.stream_count),
        usize::from(head.coupled_count),
        &head.mapping[..channels],
    )
    .map_err(|err| map_decoder_error(&err))?;

    let mut pcm = Vec::new();
    let mut packet_index = Vec::with_capacity(packets.len());
    let mut total_decoded = 0usize;
    let mut last_granule_position = None::<i64>;

    for packet in packets {
        let nsamples = opus_packet_get_nb_samples(&packet.data, packet.data.len(), 48_000)
            .map_err(map_packet_error)?;
        let mut decoded = vec![0.0f32; nsamples * channels];
        let samples = opus_multistream_decode_float(
            &mut decoder,
            &packet.data,
            packet.data.len(),
            &mut decoded,
            nsamples,
            false,
        )
        .map_err(|err| map_decoder_error(&err))?;
        packet_index.push(PacketIndex {
            raw_offset: packet.raw_offset,
            pcm_offset: total_decoded,
        });
        pcm.extend_from_slice(&decoded[..samples * channels]);
        total_decoded += samples;
        if packet.granule_position >= 0 {
            last_granule_position = Some(packet.granule_position);
        }
    }

    let pre_skip = usize::from(head.pre_skip).min(total_decoded);
    let declared_pcm_len = last_granule_position
        .and_then(|granule| head.granule_sample(granule))
        .map(|samples| samples.max(0) as usize)
        .unwrap_or_else(|| total_decoded.saturating_sub(pre_skip));
    let trimmed_pcm_len = declared_pcm_len.min(total_decoded.saturating_sub(pre_skip));
    let trimmed_start = pre_skip * channels;
    let trimmed_end = trimmed_start + trimmed_pcm_len * channels;
    let trimmed_pcm = if trimmed_start <= pcm.len() && trimmed_end <= pcm.len() {
        pcm[trimmed_start..trimmed_end].to_vec()
    } else {
        Vec::new()
    };
    for packet in &mut packet_index {
        packet.pcm_offset = packet.pcm_offset.saturating_sub(pre_skip);
    }

    Ok(Link {
        serialno,
        head,
        tags,
        raw_start,
        raw_end,
        pcm_start,
        pcm_end: pcm_start + trimmed_pcm_len,
        pcm: trimmed_pcm,
        packet_index,
    })
}

fn calculate_bitrate(bytes: u64, samples: u64) -> i32 {
    if samples == 0 {
        return i32::MAX;
    }
    let bitrate = (bytes.saturating_mul(48_000).saturating_mul(8) + (samples >> 1)) / samples;
    bitrate.min(i32::MAX as u64) as i32
}

fn map_packet_error(error: PacketError) -> OpusfileError {
    match error {
        PacketError::BadArgument | PacketError::InvalidPacket => OpusfileError::BadPacket,
    }
}

fn map_decoder_error(error: &OpusMultistreamDecoderError) -> OpusfileError {
    match error {
        OpusMultistreamDecoderError::BadArgument
        | OpusMultistreamDecoderError::InvalidPacket
        | OpusMultistreamDecoderError::DecoderDecode(_) => OpusfileError::BadPacket,
        OpusMultistreamDecoderError::Unimplemented => OpusfileError::Unimplemented,
        OpusMultistreamDecoderError::BufferTooSmall
        | OpusMultistreamDecoderError::InternalError
        | OpusMultistreamDecoderError::DecoderInit(_)
        | OpusMultistreamDecoderError::DecoderCtl(_) => OpusfileError::Fault,
    }
}

fn float_to_i16(sample: f32) -> i16 {
    let scaled = (sample * 32768.0).round().clamp(-32768.0, 32767.0);
    scaled as i16
}

fn downmix_to_stereo(src: &[f32], channels: usize, dst: &mut [f32], gain: f32) {
    debug_assert_eq!(src.len() % channels, 0);
    debug_assert_eq!(dst.len() % 2, 0);
    let frames = dst.len() / 2;
    for frame in 0..frames {
        let in_frame = &src[frame * channels..(frame + 1) * channels];
        let (mut left, mut right) = match channels {
            0 => (0.0, 0.0),
            1 => (in_frame[0], in_frame[0]),
            2 => (in_frame[0], in_frame[1]),
            3 => (
                in_frame[0] + DOWNMIX_SQRT_HALF * in_frame[1],
                in_frame[2] + DOWNMIX_SQRT_HALF * in_frame[1],
            ),
            4 => (
                in_frame[0] + DOWNMIX_SQRT_HALF * in_frame[2],
                in_frame[1] + DOWNMIX_SQRT_HALF * in_frame[3],
            ),
            5 => (
                in_frame[0] + DOWNMIX_SQRT_HALF * in_frame[1] + DOWNMIX_SQRT_HALF * in_frame[3],
                in_frame[2] + DOWNMIX_SQRT_HALF * in_frame[1] + DOWNMIX_SQRT_HALF * in_frame[4],
            ),
            6 => (
                in_frame[0]
                    + DOWNMIX_SQRT_HALF * in_frame[1]
                    + DOWNMIX_SQRT_HALF * in_frame[3]
                    + 0.5 * in_frame[5],
                in_frame[2]
                    + DOWNMIX_SQRT_HALF * in_frame[1]
                    + DOWNMIX_SQRT_HALF * in_frame[4]
                    + 0.5 * in_frame[5],
            ),
            _ => {
                let left = in_frame
                    .iter()
                    .step_by(2)
                    .copied()
                    .fold(0.0, |acc, value| acc + value);
                let right = in_frame
                    .iter()
                    .skip(1)
                    .step_by(2)
                    .copied()
                    .fold(0.0, |acc, value| acc + value);
                (left, right)
            }
        };
        left *= gain;
        right *= gain;
        dst[frame * 2] = left;
        dst[frame * 2 + 1] = right;
    }
}

#[cfg(all(test, feature = "libopusenc"))]
mod tests {
    use alloc::{format, vec, vec::Vec};
    use std::io::Cursor;

    use super::{GainType, OpusFile, probe_file, probe_opus_stream, probe_reader};
    use crate::libopusenc::{MappingFamily, OggOpusComments, OggOpusEncoderBuilder};

    fn encode_test_file(channels: usize, amplitude: f32) -> Vec<u8> {
        let comments = OggOpusComments::new().expect("comments");
        let channel_count = u8::try_from(channels).expect("test channels should fit in u8");
        let mut encoder =
            OggOpusEncoderBuilder::new(comments, 48_000, channel_count, MappingFamily::MonoStereo)
                .expect("builder")
                .build_pull()
                .expect("pull encoder");
        let frame = vec![amplitude; 960 * channels];
        encoder.flush_headers().expect("flush headers");
        encoder.write_float(&frame, 960).expect("write frame 1");
        encoder.write_float(&frame, 960).expect("write frame 2");
        encoder.finish().expect("finish");

        let mut encoded = Vec::new();
        while let Some(page) = encoder.next_page().expect("next page") {
            encoded.extend_from_slice(&page);
        }
        encoded
    }

    #[test]
    fn probe_and_open_memory_work() {
        let encoded = encode_test_file(2, 0.0);
        let head = probe_opus_stream(&encoded).expect("probe");
        assert_eq!(head.channel_count, 2);

        let file = OpusFile::open_memory(&encoded).expect("open");
        assert_eq!(file.link_count(), 1);
        assert_eq!(file.channel_count(Some(0)).expect("channels"), 2);
        assert!(file.raw_total(None).expect("raw total") > 0);
        assert!(file.pcm_total(None).expect("pcm total") > 0);
    }

    #[test]
    fn probe_reader_and_open_file_work() {
        let encoded = encode_test_file(2, 0.0);
        let head = probe_reader(Cursor::new(&encoded)).expect("probe reader");
        assert_eq!(head.channel_count, 2);

        let temp_path = std::env::temp_dir().join(format!(
            "mousiki-opusfile-test-{}-{}.opus",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        std::fs::write(&temp_path, &encoded).expect("write temp opus");

        let file = OpusFile::open_file(&temp_path).expect("open file");
        assert_eq!(file.link_count(), 1);
        let probed = probe_file(&temp_path).expect("probe file");
        assert_eq!(probed.channel_count, 2);

        std::fs::remove_file(&temp_path).expect("remove temp opus");
    }

    #[test]
    fn read_and_seek_round_trip() {
        let encoded = encode_test_file(2, 0.0);
        let mut file = OpusFile::open_memory(&encoded).expect("open");
        let total = file.pcm_total(None).expect("pcm total");
        let mut pcm = vec![0.0f32; 1024];
        let first = file.read_float(&mut pcm).expect("read");
        assert!(first.samples_per_channel > 0);
        assert_eq!(first.link_index, 0);

        file.pcm_seek(total / 2).expect("seek");
        assert_eq!(file.pcm_tell(), total / 2);
        let second = file.read_float_stereo(&mut pcm).expect("read stereo");
        assert!(second.samples_per_channel > 0);

        file.raw_seek(0).expect("raw seek");
        assert!(file.raw_tell() <= file.raw_total(None).expect("raw total"));
        assert!(file.pcm_tell() <= total);
        let third = file.read(&mut vec![0i16; 1024]).expect("read i16");
        assert!(third.samples_per_channel > 0);
    }

    #[test]
    fn gain_offset_changes_output_scale() {
        let encoded = encode_test_file(2, 0.5);
        let mut file = OpusFile::open_memory(&encoded).expect("open");
        file.set_gain_offset(GainType::Absolute, -256);
        let (_, gain_q8) = file.gain_offset();
        assert_eq!(gain_q8, -256);
        let mut pcm = vec![0.0f32; 960 * 2];
        let read = file.read_float(&mut pcm).expect("read");
        assert!(read.samples_per_channel > 0);
        assert!(pcm.iter().any(|sample| *sample != 0.0));
    }

    #[test]
    fn chained_files_advance_across_link_boundaries() {
        let mut encoded = encode_test_file(2, 0.0);
        encoded.extend_from_slice(&encode_test_file(1, 0.25));

        let mut file = OpusFile::open_memory(&encoded).expect("open chained file");
        assert_eq!(file.link_count(), 2);
        assert_eq!(file.channel_count(Some(0)).expect("link 0 channels"), 2);
        assert_eq!(file.channel_count(Some(1)).expect("link 1 channels"), 1);

        let first_total = file.pcm_total(Some(0)).expect("link 0 total");
        file.pcm_seek(first_total).expect("seek to second link");
        assert_eq!(file.current_link().expect("current link"), 1);

        let mut stereo = vec![0.0f32; 960 * 2];
        let read = file.read_float_stereo(&mut stereo).expect("read chained stereo");
        assert!(read.samples_per_channel > 0);
        assert_eq!(read.link_index, 1);
        assert!(stereo.iter().any(|sample| *sample != 0.0));
    }
}
