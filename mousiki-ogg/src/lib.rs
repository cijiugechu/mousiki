#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod c_style_api;
mod crc;
mod pack;
mod packet;
mod page;
mod stream;
mod sync;

pub use pack::{BitOrder, BitPacker, BitPackerError, BitUnpacker};
pub use packet::{Packet, PacketMetadata};
pub use page::{Page, PageFlags, SegmentSlices};
pub use stream::{StreamDecodeError, StreamDecoder, StreamEncodeError, StreamEncoder};
pub use sync::{PageReader, PageReaderError};
