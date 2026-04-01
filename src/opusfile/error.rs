extern crate std;

use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusfileError {
    False,
    EndOfFile,
    Hole,
    Read,
    Fault,
    Unimplemented,
    BadArgument,
    NotFormat,
    BadHeader,
    Version,
    NotAudio,
    BadPacket,
    BadLink,
    NoSeek,
    BadTimestamp,
}

impl OpusfileError {
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::False => -1,
            Self::EndOfFile => -2,
            Self::Hole => -3,
            Self::Read => -128,
            Self::Fault => -129,
            Self::Unimplemented => -130,
            Self::BadArgument => -131,
            Self::NotFormat => -132,
            Self::BadHeader => -133,
            Self::Version => -134,
            Self::NotAudio => -135,
            Self::BadPacket => -136,
            Self::BadLink => -137,
            Self::NoSeek => -138,
            Self::BadTimestamp => -139,
        }
    }

    #[must_use]
    pub const fn from_code(code: i32) -> Option<Self> {
        match code {
            -1 => Some(Self::False),
            -2 => Some(Self::EndOfFile),
            -3 => Some(Self::Hole),
            -128 => Some(Self::Read),
            -129 => Some(Self::Fault),
            -130 => Some(Self::Unimplemented),
            -131 => Some(Self::BadArgument),
            -132 => Some(Self::NotFormat),
            -133 => Some(Self::BadHeader),
            -134 => Some(Self::Version),
            -135 => Some(Self::NotAudio),
            -136 => Some(Self::BadPacket),
            -137 => Some(Self::BadLink),
            -138 => Some(Self::NoSeek),
            -139 => Some(Self::BadTimestamp),
            _ => None,
        }
    }
}

impl core::fmt::Display for OpusfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::False => f.write_str("request did not succeed"),
            Self::EndOfFile => f.write_str("end of file"),
            Self::Hole => f.write_str("hole in page sequence"),
            Self::Read => f.write_str("stream read/seek/tell failed"),
            Self::Fault => f.write_str("internal fault or allocation failure"),
            Self::Unimplemented => f.write_str("feature not implemented"),
            Self::BadArgument => f.write_str("invalid argument"),
            Self::NotFormat => f.write_str("not an Ogg Opus stream"),
            Self::BadHeader => f.write_str("invalid Opus header"),
            Self::Version => f.write_str("unsupported Opus header version"),
            Self::NotAudio => f.write_str("not audio"),
            Self::BadPacket => f.write_str("bad Opus packet"),
            Self::BadLink => f.write_str("bad chained link"),
            Self::NoSeek => f.write_str("stream is not seekable"),
            Self::BadTimestamp => f.write_str("bad granule position"),
        }
    }
}

impl core::error::Error for OpusfileError {}

#[derive(Debug)]
pub enum OpusfileOpenError {
    Io(io::Error),
    Opusfile(OpusfileError),
}

impl PartialEq for OpusfileOpenError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Io(lhs), Self::Io(rhs)) => lhs.kind() == rhs.kind(),
            (Self::Opusfile(lhs), Self::Opusfile(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl core::fmt::Display for OpusfileOpenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Opusfile(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for OpusfileOpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Opusfile(err) => Some(err),
        }
    }
}

impl From<io::Error> for OpusfileOpenError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<OpusfileError> for OpusfileOpenError {
    fn from(value: OpusfileError) -> Self {
        Self::Opusfile(value)
    }
}
