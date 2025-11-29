/// The remaining two bits of the `TOC` byte, labeled `c`, code the number
/// of frames per packet (codes 0 to 3) as follows
///
/// See [section-3.1](https://datatracker.ietf.org/doc/html/rfc6716#section-3.1)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum FrameCountCode {
    /// 1 frame in the packet
    Single = 0,
    /// 2 frames in the packet, each with equal compressed size
    DoubleEqual = 1,
    /// 2 frames in the packet, with different compressed sizes
    DoubleDifferent = 2,
    /// an arbitrary number of frames in the packet
    // invariant: max_count = 48
    // see https://datatracker.ietf.org/doc/html/rfc6716#section-3.2.5
    Arbitrary = 3,
}

/// See [section-3.1](https://datatracker.ietf.org/doc/html/rfc6716#section-3.1)
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Mode {
    SILK,
    CELT,
    HYBRID,
}

/// Derives the coding mode from the TOC byte.
#[inline]
pub fn opus_packet_get_mode(data: &[u8]) -> Result<Mode, PacketError> {
    let toc = *data.first().ok_or(PacketError::BadArgument)?;

    let mode = if toc & 0x80 != 0 {
        Mode::CELT
    } else if toc & 0x60 == 0x60 {
        Mode::HYBRID
    } else {
        Mode::SILK
    };

    Ok(mode)
}

/// Bandwidth
///
/// See [section-2](https://datatracker.ietf.org/doc/html/rfc6716#section-2)
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Bandwidth {
    Narrow,
    Medium,
    Wide,
    SuperWide,
    Full,
}

impl Bandwidth {
    #[inline]
    pub const fn from_opus_int(value: i32) -> Option<Self> {
        match value {
            1101 => Some(Self::Narrow),
            1102 => Some(Self::Medium),
            1103 => Some(Self::Wide),
            1104 => Some(Self::SuperWide),
            1105 => Some(Self::Full),
            _ => None,
        }
    }

    #[inline]
    pub const fn to_opus_int(&self) -> i32 {
        match self {
            Bandwidth::Narrow => 1101,
            Bandwidth::Medium => 1102,
            Bandwidth::Wide => 1103,
            Bandwidth::SuperWide => 1104,
            Bandwidth::Full => 1105,
        }
    }

    #[inline]
    pub const fn audio_band_width(&self) -> u16 {
        match self {
            Bandwidth::Narrow => 4000,
            Bandwidth::Medium => 6000,
            Bandwidth::Wide => 8000,
            Bandwidth::SuperWide => 12000,
            Bandwidth::Full => 20000,
        }
    }

    #[inline]
    pub const fn sample_rate(&self) -> u16 {
        match self {
            Bandwidth::Narrow => 8000,
            Bandwidth::Medium => 12000,
            Bandwidth::Wide => 16000,
            Bandwidth::SuperWide => 24000,
            Bandwidth::Full => 48000,
        }
    }

    /// let n be the number of samples in a subframe (40 for NB, 60 for
    /// MB, and 80 for WB)
    ///
    /// See [section-4.2.7.9](https://www.rfc-editor.org/rfc/rfc6716.html#section-4.2.7.9)
    #[inline]
    pub const fn samples_in_subframe(&self) -> u8 {
        match self {
            Bandwidth::Narrow => 40,
            Bandwidth::Medium => 60,
            Bandwidth::Wide => 80,
            _ => 0,
        }
    }
}

/// See [section-2.1.4](https://datatracker.ietf.org/doc/html/rfc6716#section-2.1.4)
#[derive(Clone, Copy, PartialEq)]
pub enum FrameDuration {
    /// 2.5 ms
    Ms2_5,
    /// 5 ms
    Ms5,
    /// 10 ms
    Ms10,
    /// 20 ms
    Ms20,
    /// 40 ms
    Ms40,
    /// 60 ms
    Ms60,
}

impl core::fmt::Debug for FrameDuration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FrameDuration::Ms2_5 => write!(f, "2.5 ms"),
            FrameDuration::Ms5 => write!(f, "5 ms"),
            FrameDuration::Ms10 => write!(f, "10 ms"),
            FrameDuration::Ms20 => write!(f, "20 ms"),
            FrameDuration::Ms40 => write!(f, "40 ms"),
            FrameDuration::Ms60 => write!(f, "60 ms"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Packet<'a> {
    frame_count_code: FrameCountCode,
    variable_bitrate: bool,
    stereo: bool,
    // TODO: determine invariant
    opus_padding: u16,
    mode: Mode,
    bandwidth: Bandwidth,
    frame_duration: FrameDuration,
    raw_data: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketError {
    BadArgument,
    InvalidPacket,
}

impl PacketError {
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            PacketError::BadArgument => -1,
            PacketError::InvalidPacket => -4,
        }
    }
}

#[inline]
pub fn opus_packet_get_bandwidth(data: &[u8]) -> Result<Bandwidth, PacketError> {
    let toc = *data.first().ok_or(PacketError::BadArgument)?;

    let bandwidth = if toc & 0x80 != 0 {
        match (toc >> 5) & 0x03 {
            0 => Bandwidth::Narrow,
            1 => Bandwidth::Wide,
            2 => Bandwidth::SuperWide,
            _ => Bandwidth::Full,
        }
    } else if toc & 0x60 == 0x60 {
        if toc & 0x10 != 0 {
            Bandwidth::Full
        } else {
            Bandwidth::SuperWide
        }
    } else {
        match (toc >> 5) & 0x03 {
            0 => Bandwidth::Narrow,
            1 => Bandwidth::Medium,
            2 => Bandwidth::Wide,
            _ => Bandwidth::SuperWide,
        }
    };

    Ok(bandwidth)
}

#[inline]
pub fn opus_packet_get_nb_channels(data: &[u8]) -> Result<usize, PacketError> {
    let toc = *data.first().ok_or(PacketError::BadArgument)?;
    Ok(if toc & 0x04 != 0 { 2 } else { 1 })
}

#[inline]
pub fn opus_packet_get_samples_per_frame(data: &[u8], fs_hz: u32) -> Result<usize, PacketError> {
    let toc = *data.first().ok_or(PacketError::BadArgument)?;

    let audiosize = if toc & 0x80 != 0 {
        let shift = u32::from((toc >> 3) & 0x03);
        fs_hz.checked_shl(shift).map(|value| value / 400)
    } else if toc & 0x60 == 0x60 {
        Some(if toc & 0x08 != 0 {
            fs_hz / 50
        } else {
            fs_hz / 100
        })
    } else {
        let size_code = (toc >> 3) & 0x03;
        if size_code == 3 {
            fs_hz.checked_mul(60).map(|value| value / 1000)
        } else {
            fs_hz.checked_shl(size_code.into()).map(|value| value / 100)
        }
    }
    .ok_or(PacketError::BadArgument)?;

    Ok(audiosize as usize)
}

#[inline]
pub fn opus_packet_get_nb_frames(packet: &[u8], len: usize) -> Result<usize, PacketError> {
    if len == 0 || len > packet.len() {
        return Err(PacketError::BadArgument);
    }

    let count = packet[0] & 0x03;
    if count == 0 {
        return Ok(1);
    }
    if count != 3 {
        return Ok(2);
    }
    if len < 2 {
        return Err(PacketError::InvalidPacket);
    }

    Ok((packet[1] & 0x3F) as usize)
}

#[inline]
pub fn opus_packet_get_nb_samples(
    packet: &[u8],
    len: usize,
    fs_hz: u32,
) -> Result<usize, PacketError> {
    let count = opus_packet_get_nb_frames(packet, len)?;
    let samples_per_frame = opus_packet_get_samples_per_frame(packet, fs_hz)?;
    let samples = count
        .checked_mul(samples_per_frame)
        .ok_or(PacketError::InvalidPacket)?;

    let max_samples = u64::from(fs_hz).saturating_mul(3);
    let scaled = (samples as u64).saturating_mul(25);
    if scaled > max_samples {
        Err(PacketError::InvalidPacket)
    } else {
        Ok(samples)
    }
}

/// Maximum number of frames allowed in a single Opus packet.
pub const MAX_FRAMES_PER_PACKET: usize = 48;

/// Maximum encoded size in bytes for a single frame when not explicitly delimited.
const MAX_FRAME_BYTES: usize = 1275;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedPacket<'a> {
    pub toc: u8,
    pub frame_count: usize,
    pub frames: [&'a [u8]; MAX_FRAMES_PER_PACKET],
    pub frame_sizes: [u16; MAX_FRAMES_PER_PACKET],
    pub payload_offset: usize,
    pub packet_offset: usize,
    pub padding: &'a [u8],
}

#[inline]
fn parse_size(data: &[u8]) -> Result<(usize, usize), PacketError> {
    let Some(&first) = data.first() else {
        return Err(PacketError::InvalidPacket);
    };

    if first < 252 {
        Ok((1, usize::from(first)))
    } else {
        let Some(&second) = data.get(1) else {
            return Err(PacketError::InvalidPacket);
        };
        Ok((2, 4 * usize::from(second) + usize::from(first)))
    }
}

/// Parses an Opus packet, optionally treating it as self-delimited.
///
/// Mirrors `opus_packet_parse_impl` from the reference C implementation. On
/// success, returns the number of frames plus their sizes and slices, along
/// with bookkeeping offsets used by the multistream decode path.
pub fn opus_packet_parse_impl<'a>(
    packet: &'a [u8],
    len: usize,
    self_delimited: bool,
) -> Result<ParsedPacket<'a>, PacketError> {
    if len > packet.len() {
        return Err(PacketError::BadArgument);
    }
    if len == 0 {
        return Err(PacketError::InvalidPacket);
    }

    let mut frame_sizes = [0u16; MAX_FRAMES_PER_PACKET];
    let mut frames = [&packet[..0]; MAX_FRAMES_PER_PACKET];

    let mut idx = 0usize;
    let mut remaining = len;

    let toc = packet[0];
    idx += 1;
    remaining -= 1;

    let framesize = opus_packet_get_samples_per_frame(packet, 48_000)?;

    let mut cbr = false;
    let frame_count: usize;
    let mut last_size: isize = remaining as isize;
    let mut pad = 0usize;

    match toc & 0x03 {
        0 => {
            frame_count = 1;
        }
        1 => {
            frame_count = 2;
            cbr = true;
            if !self_delimited {
                if remaining & 0x1 != 0 {
                    return Err(PacketError::InvalidPacket);
                }
                last_size = (remaining / 2) as isize;
                frame_sizes[0] = last_size as u16;
            }
        }
        2 => {
            frame_count = 2;
            let (size_bytes, size) = parse_size(&packet[idx..len])?;
            if size > remaining.saturating_sub(size_bytes) {
                return Err(PacketError::InvalidPacket);
            }
            idx += size_bytes;
            remaining -= size_bytes;
            frame_sizes[0] = size as u16;
            last_size = remaining
                .checked_sub(size)
                .ok_or(PacketError::InvalidPacket)? as isize;
        }
        _ => {
            if remaining == 0 {
                return Err(PacketError::InvalidPacket);
            }
            let ch = packet[idx];
            idx += 1;
            remaining -= 1;

            frame_count = usize::from(ch & 0x3F);
            if frame_count == 0
                || frame_count > MAX_FRAMES_PER_PACKET
                || framesize
                    .checked_mul(frame_count)
                    .is_none_or(|total| total > 5760)
            {
                return Err(PacketError::InvalidPacket);
            }

            if ch & 0x40 != 0 {
                loop {
                    if remaining == 0 {
                        return Err(PacketError::InvalidPacket);
                    }
                    let p = packet[idx];
                    idx += 1;
                    remaining -= 1;

                    let tmp = if p == 255 { 254usize } else { usize::from(p) };
                    pad = pad.checked_add(tmp).ok_or(PacketError::InvalidPacket)?;
                    if remaining < tmp {
                        return Err(PacketError::InvalidPacket);
                    }
                    remaining -= tmp;
                    if p != 255 {
                        break;
                    }
                }
            }

            cbr = (ch & 0x80) == 0;
            if !cbr {
                last_size = remaining as isize;
                for slot in frame_sizes.iter_mut().take(frame_count - 1) {
                    let (size_bytes, size) = parse_size(&packet[idx..len])?;
                    if size > remaining.saturating_sub(size_bytes) {
                        return Err(PacketError::InvalidPacket);
                    }
                    idx += size_bytes;
                    remaining -= size_bytes;

                    *slot = size as u16;
                    last_size -= (size_bytes + size) as isize;
                    if last_size < 0 {
                        return Err(PacketError::InvalidPacket);
                    }
                }
            } else if !self_delimited {
                let per_frame = remaining / frame_count;
                if per_frame * frame_count != remaining {
                    return Err(PacketError::InvalidPacket);
                }
                last_size = per_frame as isize;
                for slot in frame_sizes.iter_mut().take(frame_count - 1) {
                    *slot = per_frame as u16;
                }
            }
        }
    }

    if self_delimited {
        let (size_bytes, size) = parse_size(&packet[idx..len])?;
        if size > remaining.saturating_sub(size_bytes) {
            return Err(PacketError::InvalidPacket);
        }
        idx += size_bytes;
        remaining -= size_bytes;
        frame_sizes[frame_count - 1] = size as u16;

        if cbr {
            let total = size
                .checked_mul(frame_count)
                .ok_or(PacketError::InvalidPacket)?;
            if total > remaining {
                return Err(PacketError::InvalidPacket);
            }
            for slot in frame_sizes.iter_mut().take(frame_count - 1) {
                *slot = size as u16;
            }
        } else if size_bytes + size > last_size as usize {
            return Err(PacketError::InvalidPacket);
        }
    } else {
        if last_size < 0 || last_size as usize > MAX_FRAME_BYTES {
            return Err(PacketError::InvalidPacket);
        }
        frame_sizes[frame_count - 1] = last_size as u16;
    }

    let payload_offset = idx;
    let mut cursor = idx;

    for (frame_slot, &size) in frames.iter_mut().zip(frame_sizes.iter()).take(frame_count) {
        let sz = usize::from(size);
        let end = cursor.checked_add(sz).ok_or(PacketError::InvalidPacket)?;
        if end > len {
            return Err(PacketError::InvalidPacket);
        }
        *frame_slot = &packet[cursor..end];
        cursor = end;
    }

    let padding_end = cursor.checked_add(pad).ok_or(PacketError::InvalidPacket)?;
    if padding_end > len {
        return Err(PacketError::InvalidPacket);
    }

    Ok(ParsedPacket {
        toc,
        frame_count,
        frames,
        frame_sizes,
        payload_offset,
        packet_offset: padding_end,
        padding: &packet[cursor..padding_end],
    })
}

/// Convenience wrapper for non-self-delimited packets.
#[inline]
pub fn opus_packet_parse<'a>(
    packet: &'a [u8],
    len: usize,
) -> Result<ParsedPacket<'a>, PacketError> {
    opus_packet_parse_impl(packet, len, false)
}
