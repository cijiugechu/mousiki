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
