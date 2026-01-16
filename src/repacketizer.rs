use alloc::vec;
use alloc::vec::Vec;

use crate::extensions;
use crate::packet::{
    MAX_FRAMES_PER_PACKET, PacketError, opus_packet_get_nb_frames,
    opus_packet_get_samples_per_frame, opus_packet_parse_impl,
};

pub use crate::extensions::OpusExtensionData;

/// Errors surfaced by the repacketizer helpers, mirroring the C API codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepacketizerError {
    BadArgument,
    BufferTooSmall,
    InternalError,
    InvalidPacket,
}

impl RepacketizerError {
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            RepacketizerError::BadArgument => -1,
            RepacketizerError::BufferTooSmall => -2,
            RepacketizerError::InternalError => -3,
            RepacketizerError::InvalidPacket => -4,
        }
    }
}

impl From<PacketError> for RepacketizerError {
    #[inline]
    fn from(value: PacketError) -> Self {
        match value {
            PacketError::BadArgument => RepacketizerError::InvalidPacket,
            PacketError::InvalidPacket => RepacketizerError::InvalidPacket,
        }
    }
}

fn map_extension_error(err: extensions::ExtensionError) -> RepacketizerError {
    match err {
        extensions::ExtensionError::BadArgument => RepacketizerError::BadArgument,
        extensions::ExtensionError::BufferTooSmall => RepacketizerError::BufferTooSmall,
        extensions::ExtensionError::InvalidPacket => RepacketizerError::InvalidPacket,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Frame {
    start: usize,
    len: u16,
    padding_start: Option<usize>,
}

/// Repacketizer state mirroring the C layout, storing frame metadata and a backing copy
/// of the packet payloads referenced by the frames.
pub struct OpusRepacketizer {
    toc: u8,
    nb_frames: usize,
    frames: [Frame; MAX_FRAMES_PER_PACKET],
    framesize: usize,
    padding_len: [usize; MAX_FRAMES_PER_PACKET],
    padding_nb_frames: [u8; MAX_FRAMES_PER_PACKET],
    buffer: Vec<u8>,
}

impl Default for OpusRepacketizer {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusRepacketizer {
    #[inline]
    pub fn new() -> Self {
        OpusRepacketizer {
            toc: 0,
            nb_frames: 0,
            frames: [Frame::default(); MAX_FRAMES_PER_PACKET],
            framesize: 0,
            padding_len: [0; MAX_FRAMES_PER_PACKET],
            padding_nb_frames: [0; MAX_FRAMES_PER_PACKET],
            buffer: Vec::new(),
        }
    }

    #[inline]
    pub const fn opus_repacketizer_get_size() -> usize {
        core::mem::size_of::<OpusRepacketizer>()
    }

    #[inline]
    pub fn opus_repacketizer_init(&mut self) -> &mut Self {
        self.nb_frames = 0;
        self.frames = [Frame::default(); MAX_FRAMES_PER_PACKET];
        self.padding_len = [0; MAX_FRAMES_PER_PACKET];
        self.padding_nb_frames = [0; MAX_FRAMES_PER_PACKET];
        self.framesize = 0;
        self.buffer.clear();
        self
    }

    fn cat_impl(
        &mut self,
        data: &[u8],
        len: usize,
        self_delimited: bool,
    ) -> Result<(), RepacketizerError> {
        if len < 1 || len > data.len() {
            return Err(RepacketizerError::InvalidPacket);
        }
        if self.nb_frames == 0 {
            self.toc = data[0];
            self.framesize = opus_packet_get_samples_per_frame(data, 8000)? as usize;
        } else if (self.toc & 0xFC) != (data[0] & 0xFC) {
            return Err(RepacketizerError::InvalidPacket);
        }

        let curr_nb_frames = opus_packet_get_nb_frames(data, len)? as usize;
        if (curr_nb_frames + self.nb_frames) * self.framesize > 960 {
            return Err(RepacketizerError::InvalidPacket);
        }

        let parsed = opus_packet_parse_impl(data, len, self_delimited)?;
        if parsed.frame_count == 0 {
            return Err(RepacketizerError::InvalidPacket);
        }

        let base = self.buffer.len();
        self.buffer.extend_from_slice(&data[..len]);

        let mut cursor = base + parsed.payload_offset;
        for (slot, size) in
            (self.nb_frames..self.nb_frames + parsed.frame_count).zip(parsed.frame_sizes.iter())
        {
            self.frames[slot] = Frame {
                start: cursor,
                len: *size,
                padding_start: None,
            };
            cursor = cursor
                .checked_add(*size as usize)
                .ok_or(RepacketizerError::InvalidPacket)?;
        }

        self.padding_len[self.nb_frames] = parsed.padding.len();
        self.padding_nb_frames[self.nb_frames] = parsed.frame_count as u8;
        if !parsed.padding.is_empty() {
            let pad_start = base + (parsed.packet_offset - parsed.padding.len());
            self.frames[self.nb_frames].padding_start = Some(pad_start);
        }
        for slot in self.nb_frames + 1..self.nb_frames + parsed.frame_count {
            self.padding_len[slot] = 0;
            self.padding_nb_frames[slot] = 0;
            self.frames[slot].padding_start = None;
        }

        self.nb_frames += parsed.frame_count;
        Ok(())
    }

    pub fn opus_repacketizer_cat(
        &mut self,
        data: &[u8],
        len: usize,
    ) -> Result<(), RepacketizerError> {
        self.cat_impl(data, len, false)
    }

    #[inline]
    pub fn opus_repacketizer_get_nb_frames(&self) -> usize {
        self.nb_frames
    }

    #[allow(clippy::too_many_arguments)]
    fn opus_repacketizer_out_range_impl(
        &self,
        begin: usize,
        end: usize,
        data: &mut [u8],
        maxlen: usize,
        self_delimited: bool,
        pad: bool,
        extensions: &[OpusExtensionData<'_>],
    ) -> Result<usize, RepacketizerError> {
        if begin >= end || end > self.nb_frames {
            return Err(RepacketizerError::BadArgument);
        }

        let count = end - begin;
        let frames = &self.frames[begin..end];
        let first_len = frames[0].len as usize;
        let last_len = frames[count - 1].len as usize;

        let mut ones_begin = 0usize;
        let mut ones_end = 0usize;
        let mut ext_begin = 0usize;
        let mut ext_len = 0usize;

        let mut total_ext_count = extensions.len();
        for i in begin..end {
            let pad_len = self.padding_len[i];
            let nb_pad_frames = usize::from(self.padding_nb_frames[i]);
            if pad_len == 0 || nb_pad_frames == 0 {
                continue;
            }
            let pad_start = self.frames[i]
                .padding_start
                .ok_or(RepacketizerError::InternalError)?;
            let pad_end = pad_start
                .checked_add(pad_len)
                .filter(|end| *end <= self.buffer.len())
                .ok_or(RepacketizerError::InternalError)?;
            let padding = &self.buffer[pad_start..pad_end];
            let count = extensions::opus_packet_extensions_count(padding, pad_len, nb_pad_frames)
                .map_err(|_| RepacketizerError::InternalError)?;
            total_ext_count = total_ext_count
                .checked_add(count)
                .ok_or(RepacketizerError::InternalError)?;
        }

        let mut all_extensions: Vec<OpusExtensionData<'_>> = Vec::with_capacity(total_ext_count);
        if !extensions.is_empty() {
            all_extensions.extend_from_slice(extensions);
        }

        for i in begin..end {
            let pad_len = self.padding_len[i];
            let nb_pad_frames = usize::from(self.padding_nb_frames[i]);
            if pad_len == 0 || nb_pad_frames == 0 {
                continue;
            }
            let pad_start = self.frames[i]
                .padding_start
                .ok_or(RepacketizerError::InternalError)?;
            let pad_end = pad_start
                .checked_add(pad_len)
                .filter(|end| *end <= self.buffer.len())
                .ok_or(RepacketizerError::InternalError)?;
            let padding = &self.buffer[pad_start..pad_end];
            let frame_count =
                extensions::opus_packet_extensions_count(padding, pad_len, nb_pad_frames)
                    .map_err(|_| RepacketizerError::InternalError)?;
            if frame_count == 0 {
                continue;
            }
            let mut frame_exts = vec![OpusExtensionData::default(); frame_count];
            extensions::opus_packet_extensions_parse(
                padding,
                pad_len,
                nb_pad_frames,
                &mut frame_exts,
            )
            .map_err(|_| RepacketizerError::InternalError)?;
            for mut ext in frame_exts {
                ext.frame += (i - begin) as i32;
                all_extensions.push(ext);
            }
        }

        let ext_count = all_extensions.len();

        let mut ptr = 0usize;
        let mut tot_size = if self_delimited {
            1 + usize::from(last_len >= 252)
        } else {
            0
        };

        if count == 1 {
            tot_size += first_len + 1;
            if tot_size > maxlen {
                return Err(RepacketizerError::BufferTooSmall);
            }
            data[ptr] = self.toc & 0xFC;
            ptr += 1;
        } else if count == 2 {
            let second_len = frames[1].len as usize;
            if second_len == first_len {
                tot_size += 2 * first_len + 1;
                if tot_size > maxlen {
                    return Err(RepacketizerError::BufferTooSmall);
                }
                data[ptr] = (self.toc & 0xFC) | 0x1;
                ptr += 1;
            } else {
                tot_size += first_len + second_len + 2 + usize::from(first_len >= 252);
                if tot_size > maxlen {
                    return Err(RepacketizerError::BufferTooSmall);
                }
                data[ptr] = (self.toc & 0xFC) | 0x2;
                ptr += 1;
                ptr += encode_size(first_len, &mut data[ptr..]);
            }
        }

        if count > 2 || (pad && tot_size < maxlen) || ext_count > 0 {
            let mut vbr = false;
            let mut pad_amount = 0usize;

            // Restart for padding path.
            ptr = 0;
            tot_size = if self_delimited {
                1 + usize::from(last_len >= 252)
            } else {
                0
            };

            for frame in frames.iter().take(count).skip(1) {
                if frame.len as usize != first_len {
                    vbr = true;
                    break;
                }
            }

            if vbr {
                tot_size += 2;
                for frame in frames.iter().take(count - 1) {
                    let len = frame.len as usize;
                    tot_size += 1 + usize::from(len >= 252) + len;
                }
                tot_size += last_len;
                if tot_size > maxlen {
                    return Err(RepacketizerError::BufferTooSmall);
                }
                data[ptr] = (self.toc & 0xFC) | 0x3;
                ptr += 1;
                data[ptr] = (count as u8) | 0x80;
                ptr += 1;
            } else {
                tot_size += count * first_len + 2;
                if tot_size > maxlen {
                    return Err(RepacketizerError::BufferTooSmall);
                }
                data[ptr] = (self.toc & 0xFC) | 0x3;
                ptr += 1;
                data[ptr] = count as u8;
                ptr += 1;
            }

            if pad && tot_size < maxlen {
                pad_amount = maxlen - tot_size;
            }

            if ext_count > 0 {
                ext_len = extensions::opus_packet_extensions_generate(
                    None,
                    maxlen.saturating_sub(tot_size),
                    &all_extensions,
                    count,
                    false,
                )
                .map_err(map_extension_error)?;
                if !pad {
                    pad_amount = ext_len + ext_len / 254 + 1;
                }
            }

            if pad_amount != 0 {
                let nb_255s = (pad_amount - 1) / 255;
                let size_ok = tot_size
                    .checked_add(ext_len)
                    .and_then(|s| s.checked_add(nb_255s + 1))
                    .is_none_or(|needed| needed > maxlen);
                if size_ok {
                    return Err(RepacketizerError::BufferTooSmall);
                }
                ext_begin = tot_size + pad_amount - ext_len;
                ones_begin = tot_size + nb_255s + 1;
                ones_end = tot_size + pad_amount - ext_len;

                data[1] |= 0x40;
                for _ in 0..nb_255s {
                    data[ptr] = 255;
                    ptr += 1;
                }
                data[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                ptr += 1;
                tot_size += pad_amount;
            }

            if vbr {
                for frame in frames.iter().take(count - 1) {
                    let len = frame.len as usize;
                    ptr += encode_size(len, &mut data[ptr..]);
                }
            }
        }

        if self_delimited {
            ptr += encode_size(last_len, &mut data[ptr..]);
        }

        for frame in frames.iter() {
            let len = frame.len as usize;
            let src_start = frame.start;
            let src_end = src_start + len;
            if src_end > self.buffer.len() {
                return Err(RepacketizerError::InternalError);
            }
            let dst_end = ptr + len;
            if dst_end > data.len() {
                return Err(RepacketizerError::BufferTooSmall);
            }
            data[ptr..dst_end].copy_from_slice(&self.buffer[src_start..src_end]);
            ptr = dst_end;
        }

        if ext_len > 0 {
            let generated = extensions::opus_packet_extensions_generate(
                Some(&mut data[ext_begin..ext_begin + ext_len]),
                ext_len,
                &all_extensions,
                count,
                false,
            )
            .map_err(map_extension_error)?;
            debug_assert_eq!(generated, ext_len);
        }

        for byte in data.iter_mut().take(ones_end).skip(ones_begin) {
            *byte = 0x01;
        }

        if pad && ext_count == 0 {
            for byte in data.iter_mut().take(maxlen).skip(ptr) {
                *byte = 0;
            }
        }

        Ok(tot_size)
    }

    #[inline]
    pub fn opus_repacketizer_out_range(
        &self,
        begin: usize,
        end: usize,
        data: &mut [u8],
        maxlen: usize,
    ) -> Result<usize, RepacketizerError> {
        self.opus_repacketizer_out_range_impl(begin, end, data, maxlen, false, false, &[])
    }

    #[inline]
    pub fn opus_repacketizer_out(
        &self,
        data: &mut [u8],
        maxlen: usize,
    ) -> Result<usize, RepacketizerError> {
        self.opus_repacketizer_out_range_impl(0, self.nb_frames, data, maxlen, false, false, &[])
    }
}

fn encode_size(size: usize, data: &mut [u8]) -> usize {
    if size < 252 {
        data[0] = size as u8;
        1
    } else {
        data[0] = 252 + (size & 0x3) as u8;
        data[1] = ((size - usize::from(data[0])) >> 2) as u8;
        2
    }
}

/// Pads a packet to `new_len`, preserving frame ordering and ToC header.
pub fn opus_packet_pad(
    data: &mut [u8],
    len: usize,
    new_len: usize,
) -> Result<(), RepacketizerError> {
    opus_packet_pad_impl(data, len, new_len, true)
}

pub(crate) fn opus_packet_pad_with_extensions(
    data: &mut [u8],
    len: usize,
    new_len: usize,
    pad: bool,
    extensions: &[OpusExtensionData<'_>],
) -> Result<usize, RepacketizerError> {
    if len < 1 {
        return Err(RepacketizerError::BadArgument);
    }
    if len == new_len {
        return Ok(len);
    }
    if len > new_len || new_len > data.len() {
        return Err(RepacketizerError::BadArgument);
    }

    let mut copy = Vec::with_capacity(len);
    copy.extend_from_slice(&data[..len]);

    let mut rp = OpusRepacketizer::new();
    rp.opus_repacketizer_cat(&copy, len)?;
    let written =
        rp.opus_repacketizer_out_range_impl(0, rp.nb_frames, data, new_len, false, pad, extensions)?;

    if written > 0 {
        Ok(written)
    } else {
        Err(RepacketizerError::InternalError)
    }
}

fn opus_packet_pad_impl(
    data: &mut [u8],
    len: usize,
    new_len: usize,
    pad: bool,
) -> Result<(), RepacketizerError> {
    if len < 1 {
        return Err(RepacketizerError::BadArgument);
    }
    if len == new_len {
        return Ok(());
    }
    if len > new_len || new_len > data.len() {
        return Err(RepacketizerError::BadArgument);
    }

    let mut copy = Vec::with_capacity(len);
    copy.extend_from_slice(&data[..len]);

    let mut rp = OpusRepacketizer::new();
    rp.opus_repacketizer_cat(&copy, len)?;
    let written =
        rp.opus_repacketizer_out_range_impl(0, rp.nb_frames, data, new_len, false, pad, &[])?;

    if written > 0 {
        Ok(())
    } else {
        Err(RepacketizerError::InternalError)
    }
}

/// Removes all padding and extensions from a packet, returning the new size.
pub fn opus_packet_unpad(data: &mut [u8], len: usize) -> Result<usize, RepacketizerError> {
    if len < 1 {
        return Err(RepacketizerError::BadArgument);
    }

    let mut rp = OpusRepacketizer::new();
    rp.opus_repacketizer_cat(data, len)?;
    for slot in 0..rp.nb_frames {
        rp.padding_len[slot] = 0;
        rp.padding_nb_frames[slot] = 0;
        rp.frames[slot].padding_start = None;
    }
    let written =
        rp.opus_repacketizer_out_range_impl(0, rp.nb_frames, data, len, false, false, &[])?;
    debug_assert!(written > 0 && written <= len);
    if written == 0 || written > len {
        Err(RepacketizerError::InternalError)
    } else {
        Ok(written)
    }
}

pub fn opus_multistream_packet_pad(
    data: &mut [u8],
    len: usize,
    new_len: usize,
    nb_streams: usize,
) -> Result<(), RepacketizerError> {
    if len < 1 {
        return Err(RepacketizerError::BadArgument);
    }
    if len == new_len {
        return Ok(());
    }
    if len > new_len || new_len > data.len() {
        return Err(RepacketizerError::BadArgument);
    }

    let mut offset = 0usize;
    let mut remaining = len;
    for _ in 0..nb_streams.saturating_sub(1) {
        if remaining == 0 {
            return Err(RepacketizerError::InvalidPacket);
        }
        let parsed = opus_packet_parse_impl(&data[offset..], remaining, true)?;
        if parsed.packet_offset > remaining {
            return Err(RepacketizerError::InvalidPacket);
        }
        offset += parsed.packet_offset;
        remaining -= parsed.packet_offset;
    }

    opus_packet_pad(&mut data[offset..], remaining, remaining + (new_len - len))
}

pub fn opus_multistream_packet_unpad(
    data: &mut [u8],
    len: usize,
    nb_streams: usize,
) -> Result<usize, RepacketizerError> {
    if len < 1 {
        return Err(RepacketizerError::BadArgument);
    }

    let mut offset = 0usize;
    let mut remaining = len;
    let mut dst_len = 0usize;

    for stream in 0..nb_streams {
        let self_delimited = stream + 1 != nb_streams;
        if remaining == 0 {
            return Err(RepacketizerError::InvalidPacket);
        }

        let packet_offset = {
            let parsed = opus_packet_parse_impl(&data[offset..], remaining, self_delimited)?;
            if parsed.packet_offset > remaining {
                return Err(RepacketizerError::InvalidPacket);
            }
            parsed.packet_offset
        };

        let packet_end = offset + packet_offset;
        let mut packet_copy = Vec::with_capacity(packet_offset);
        packet_copy.extend_from_slice(&data[offset..packet_end]);

        let mut rp = OpusRepacketizer::new();
        rp.cat_impl(&packet_copy, packet_copy.len(), self_delimited)?;
        for slot in 0..rp.nb_frames {
            rp.padding_len[slot] = 0;
            rp.padding_nb_frames[slot] = 0;
            rp.frames[slot].padding_start = None;
        }

        let out_len = rp.opus_repacketizer_out_range_impl(
            0,
            rp.nb_frames,
            &mut data[dst_len..],
            len - dst_len,
            self_delimited,
            false,
            &[],
        )?;

        dst_len += out_len;
        offset += packet_offset;
        remaining = remaining
            .checked_sub(packet_offset)
            .ok_or(RepacketizerError::InvalidPacket)?;
    }

    Ok(dst_len)
}
