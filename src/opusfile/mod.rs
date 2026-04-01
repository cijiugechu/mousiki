//! Feature-gated high-level Ogg Opus reader utilities ported from `opusfile`.

mod error;
mod head;
mod picture;
mod reader;
mod tags;
mod util;

pub use error::{OpusfileError, OpusfileOpenError};
pub use head::{OPUS_CHANNEL_COUNT_MAX, OpusHead};
pub use picture::{OpusPictureFormat, OpusPictureTag};
pub use reader::{GainType, OpusFile, ReadSamples, probe_file, probe_opus_stream, probe_reader};
pub use tags::{OpusTags, tag_compare, tag_ncompare};
