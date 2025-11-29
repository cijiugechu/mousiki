#![no_std]

extern crate alloc;

pub mod bitdepth;
mod celt;
pub mod decoder;
mod math;
pub mod opus;
pub mod oggreader;
pub mod packet;
pub mod range;
pub mod resample;
pub mod silk;
pub mod opus_decoder;

/// Returns the textual version identifier for the library, matching
/// `opus_get_version_string` from the reference implementation.
#[must_use]
pub fn opus_get_version_string() -> &'static str {
    crate::celt::opus_get_version_string()
}
