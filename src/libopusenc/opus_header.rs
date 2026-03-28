use alloc::vec::Vec;

use crate::projection::{ProjectionLayout, demixing_matrix_gain, write_demixing_matrix_subset};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusHeader {
    pub version: i32,
    pub channels: i32,
    pub preskip: i32,
    pub input_sample_rate: u32,
    pub gain: i32,
    pub channel_mapping: i32,
    pub nb_streams: i32,
    pub nb_coupled: i32,
    pub stream_map: [u8; 255],
}

impl Default for OpusHeader {
    fn default() -> Self {
        Self {
            version: 0,
            channels: 0,
            preskip: 0,
            input_sample_rate: 0,
            gain: 0,
            channel_mapping: 0,
            nb_streams: 0,
            nb_coupled: 0,
            stream_map: [0; 255],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentError {
    AllocationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderEncodeError {
    BufferTooSmall,
    MissingProjectionLayout,
    ProjectionMatrixWriteFailed,
}

#[must_use]
pub const fn use_projection(channel_mapping: i32) -> bool {
    channel_mapping == 3
}

#[must_use]
pub fn opus_header_get_size(header: &OpusHeader) -> usize {
    if use_projection(header.channel_mapping) {
        21 + header.channels as usize * (header.nb_streams + header.nb_coupled) as usize * 2
    } else {
        21 + header.channels as usize
    }
}

pub fn opus_header_to_packet(
    header: &OpusHeader,
    packet: &mut [u8],
    projection: Option<&ProjectionLayout>,
) -> Result<usize, HeaderEncodeError> {
    if packet.len() < 19 {
        return Err(HeaderEncodeError::BufferTooSmall);
    }

    let mut writer = PacketWriter::new(packet);
    writer.write_bytes(b"OpusHead")?;
    writer.write_u8(1)?;
    writer.write_u8(header.channels as u8)?;
    writer.write_u16(header.preskip as u16)?;
    writer.write_u32(header.input_sample_rate)?;

    let gain = if use_projection(header.channel_mapping) {
        let Some(layout) = projection else {
            return Err(HeaderEncodeError::MissingProjectionLayout);
        };
        header.gain + demixing_matrix_gain(layout)
    } else {
        header.gain
    };
    writer.write_u16(gain as u16)?;
    writer.write_u8(header.channel_mapping as u8)?;

    if header.channel_mapping != 0 {
        writer.write_u8(header.nb_streams as u8)?;
        writer.write_u8(header.nb_coupled as u8)?;

        if use_projection(header.channel_mapping) {
            let Some(layout) = projection else {
                return Err(HeaderEncodeError::MissingProjectionLayout);
            };
            let matrix_len = layout
                .demixing_subset_size_bytes()
                .map_err(|_| HeaderEncodeError::ProjectionMatrixWriteFailed)?;
            let dst = writer.take(matrix_len)?;
            write_demixing_matrix_subset(layout, dst)
                .map_err(|_| HeaderEncodeError::ProjectionMatrixWriteFailed)?;
        } else {
            let channels = header.channels as usize;
            writer.write_bytes(&header.stream_map[..channels])?;
        }
    }

    Ok(writer.pos)
}

pub fn comment_init(vendor_string: &str) -> Result<Vec<u8>, CommentError> {
    let vendor_length = vendor_string.len();
    let len = 8 + 4 + vendor_length + 4;
    let mut comments = Vec::new();
    comments
        .try_reserve_exact(len)
        .map_err(|_| CommentError::AllocationFailed)?;
    comments.extend_from_slice(b"OpusTags");
    comments.extend_from_slice(&(vendor_length as u32).to_le_bytes());
    comments.extend_from_slice(vendor_string.as_bytes());
    comments.extend_from_slice(&0u32.to_le_bytes());
    Ok(comments)
}

pub fn comment_add(
    comments: &mut Vec<u8>,
    tag: Option<&str>,
    value: &str,
) -> Result<(), CommentError> {
    if comments.len() < 16 || &comments[..8] != b"OpusTags" {
        return Err(CommentError::AllocationFailed);
    }

    let vendor_length = read_u32_le(comments, 8) as usize;
    let comment_count_offset = 8 + 4 + vendor_length;
    let user_comment_list_length = read_u32_le(comments, comment_count_offset);
    let tag_len = tag.map_or(0, |tag| tag.len() + 1);
    let value_len = value.len();
    let new_comment_len = 4 + tag_len + value_len;

    comments
        .try_reserve_exact(new_comment_len)
        .map_err(|_| CommentError::AllocationFailed)?;
    comments.extend_from_slice(&((tag_len + value_len) as u32).to_le_bytes());
    if let Some(tag) = tag {
        comments.extend_from_slice(tag.as_bytes());
        comments.push(b'=');
    }
    comments.extend_from_slice(value.as_bytes());
    comments[comment_count_offset..comment_count_offset + 4]
        .copy_from_slice(&(user_comment_list_length + 1).to_le_bytes());
    Ok(())
}

pub fn comment_pad(comments: &mut Vec<u8>, amount: i32) -> Result<(), CommentError> {
    if amount <= 0 {
        return Ok(());
    }

    let original_len = comments.len();
    let new_len = ((original_len + amount as usize + 255) / 255) * 255 - 1;
    if new_len <= original_len {
        return Ok(());
    }

    comments
        .try_reserve_exact(new_len - original_len)
        .map_err(|_| CommentError::AllocationFailed)?;
    comments.resize(new_len, 0);
    Ok(())
}

fn read_u32_le(buf: &[u8], base: usize) -> u32 {
    u32::from_le_bytes([buf[base], buf[base + 1], buf[base + 2], buf[base + 3]])
}

struct PacketWriter<'a> {
    packet: &'a mut [u8],
    pos: usize,
}

impl<'a> PacketWriter<'a> {
    fn new(packet: &'a mut [u8]) -> Self {
        Self { packet, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&mut [u8], HeaderEncodeError> {
        if self.pos > self.packet.len().saturating_sub(len) {
            return Err(HeaderEncodeError::BufferTooSmall);
        }
        let start = self.pos;
        self.pos += len;
        Ok(&mut self.packet[start..self.pos])
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), HeaderEncodeError> {
        self.take(bytes.len())?.copy_from_slice(bytes);
        Ok(())
    }

    fn write_u8(&mut self, value: u8) -> Result<(), HeaderEncodeError> {
        self.write_bytes(&[value])
    }

    fn write_u16(&mut self, value: u16) -> Result<(), HeaderEncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_u32(&mut self, value: u32) -> Result<(), HeaderEncodeError> {
        self.write_bytes(&value.to_le_bytes())
    }
}
