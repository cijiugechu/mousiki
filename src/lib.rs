#![no_std]

extern crate alloc;

mod analysis;
pub mod bitdepth;
mod celt;
pub mod decoder;
pub mod mapping_matrix;
mod math;
mod mlp;
mod mlp_data;
pub mod oggreader;
pub mod opus;
pub mod opus_decoder;
mod opus_multistream;
pub mod packet;
pub mod projection;
pub mod range;
pub mod repacketizer;
pub mod resample;
pub mod silk;

/// Returns the textual version identifier for the library, matching
/// `opus_get_version_string` from the reference implementation.
#[must_use]
pub fn opus_get_version_string() -> &'static str {
    crate::celt::opus_get_version_string()
}
