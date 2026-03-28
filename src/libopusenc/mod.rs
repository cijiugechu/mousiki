//! Feature-gated high-level Ogg Opus encoder utilities ported from libopusenc.

mod error;
mod encoder;
pub(crate) mod ogg_packer;
pub(crate) mod opus_header;
pub(crate) mod picture;
pub(crate) mod resample;

pub use encoder::{
    MappingFamily, MuxingDelaySamples, NoPacketHandler, OggOpusComments, OggOpusEncoder,
    OggOpusEncoderBuilder, OggOpusPullEncoder, PacketHandler, PictureType, get_version_string,
};
pub use error::{LibopusencError, PictureErrorKind};
