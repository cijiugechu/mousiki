#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod test_trace;

mod analysis;
pub mod bitdepth;
mod celt;
mod codec;
pub mod decoder;
#[cfg(feature = "deep_plc")]
mod dnn_utils;
#[cfg(feature = "dred")]
mod dnn_weights;
mod dred;
mod dred_constants;
#[cfg(feature = "dred")]
mod dred_encoder;
#[cfg(feature = "dred")]
mod dred_rdovae_dec;
#[cfg(feature = "dred")]
mod dred_rdovae_dec_data;
#[cfg(feature = "dred")]
mod dred_rdovae_enc;
#[cfg(feature = "dred")]
mod dred_rdovae_enc_data;
mod dred_stats_data;
mod extensions;
#[cfg(feature = "deep_plc")]
pub mod fargan;
#[cfg(feature = "libopusenc")]
pub mod libopusenc;
#[cfg(feature = "dred")]
mod lpcnet_enc;
mod mapping_matrix;
mod math;
mod mlp;
mod mlp_data;
#[cfg(feature = "dred")]
mod nnet;
pub mod oggreader;
mod opus;
mod opus_decoder;
mod opus_encoder;
mod opus_multistream;
mod packet;
#[cfg(feature = "dred")]
mod pitchdnn;
#[cfg(feature = "dred")]
mod pitchdnn_data;
#[cfg(feature = "deep_plc")]
mod plc_model;
mod projection;
pub mod range;
mod repacketizer;
pub mod resample;
pub mod silk;

pub use crate::codec::{
    Application, Bandwidth, Bitrate, Channels, Decoder, DecoderBuilder, DecoderBuilderError,
    Encoder, EncoderBuilder, EncoderBuilderError, FrameDuration, Signal,
};
pub use crate::opus_decoder::{OpusDecodeError, OpusDecoderCtlError, OpusDecoderInitError};
pub use crate::opus_encoder::{OpusEncodeError, OpusEncoderCtlError, OpusEncoderInitError};
pub use crate::packet::PacketError;

/// Low-level APIs that intentionally mirror the original C-style libopus surface.
pub mod c_style_api {
    /// Low-level DRED helpers.
    pub mod dred {
        pub use crate::dred::*;
    }

    /// Low-level packet extension helpers.
    pub mod extensions {
        pub use crate::extensions::*;
    }

    /// Low-level mapping-matrix helpers.
    pub mod mapping_matrix {
        pub use crate::mapping_matrix::*;
    }

    /// Low-level helpers corresponding to `opus.h`.
    pub mod opus {
        pub use crate::opus::*;
    }

    /// Low-level decoder API mirroring libopus.
    pub mod opus_decoder {
        pub use crate::opus_decoder::*;
    }

    /// Low-level encoder API mirroring libopus.
    pub mod opus_encoder {
        pub use crate::opus_encoder::*;
    }

    /// Low-level multistream API mirroring libopus.
    pub mod opus_multistream {
        pub use crate::opus_multistream::*;
    }

    /// Low-level packet inspection helpers.
    pub mod packet {
        pub use crate::packet::*;
    }

    /// Low-level projection API mirroring libopus.
    pub mod projection {
        pub use crate::projection::*;
    }

    /// Low-level repacketizer API mirroring libopus.
    pub mod repacketizer {
        pub use crate::repacketizer::*;
    }
}

/// Returns the textual version identifier for the library, matching
/// `opus_get_version_string` from the reference implementation.
#[must_use]
pub fn opus_get_version_string() -> &'static str {
    crate::celt::opus_get_version_string()
}
