extern crate std;

use std::fmt;
use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureErrorKind {
    InvalidPicture,
    InvalidIcon,
}

#[derive(Debug)]
pub enum LibopusencError {
    InvalidArgument,
    InvalidState,
    Unsupported,
    Io(io::Error),
    Picture(PictureErrorKind),
    Internal,
}

impl LibopusencError {
    #[must_use]
    pub fn io_kind(&self) -> Option<io::ErrorKind> {
        match self {
            Self::Io(err) => Some(err.kind()),
            _ => None,
        }
    }
}

impl PartialEq for LibopusencError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::InvalidArgument, Self::InvalidArgument)
            | (Self::InvalidState, Self::InvalidState)
            | (Self::Unsupported, Self::Unsupported)
            | (Self::Internal, Self::Internal) => true,
            (Self::Picture(lhs), Self::Picture(rhs)) => lhs == rhs,
            (Self::Io(lhs), Self::Io(rhs)) => lhs.kind() == rhs.kind(),
            _ => false,
        }
    }
}

impl fmt::Display for LibopusencError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument => f.write_str("invalid argument"),
            Self::InvalidState => {
                f.write_str("operation is not valid in the current encoder state")
            }
            Self::Unsupported => f.write_str("unsupported libopusenc configuration"),
            Self::Io(err) => write!(f, "{err}"),
            Self::Picture(PictureErrorKind::InvalidPicture) => f.write_str("invalid picture data"),
            Self::Picture(PictureErrorKind::InvalidIcon) => {
                f.write_str("invalid icon data (type 1 must be a 32x32 PNG)")
            }
            Self::Internal => f.write_str("internal libopusenc error"),
        }
    }
}

impl std::error::Error for LibopusencError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for LibopusencError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
