//! Feature-gated high-level Ogg Opus encoder utilities ported from libopusenc.

mod encoder;
pub(crate) mod ogg_packer;
pub(crate) mod opus_header;
pub(crate) mod picture;
pub(crate) mod resample;

pub use encoder::{
    OggOpusComments, OggOpusEnc, OpeError, OpusEncCallbacks, PacketCallback, get_abi_version,
    get_version_string, strerror,
};
