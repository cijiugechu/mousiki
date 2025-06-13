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
    pub fn audio_band_width(&self) -> u16 {
        match self {
            Bandwidth::Narrow => 4000,
            Bandwidth::Medium => 6000,
            Bandwidth::Wide => 8000,
            Bandwidth::SuperWide => 12000,
            Bandwidth::Full => 20000,
        }
    }

    #[inline]
    pub fn sample_rate(&self) -> u16 {
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
    pub fn samples_in_subframe(&self) -> u8 {
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
