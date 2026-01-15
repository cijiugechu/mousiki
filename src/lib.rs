#![no_std]

extern crate alloc;

mod analysis;
pub mod bitdepth;
mod celt;
pub mod decoder;
pub mod dred;
mod dred_constants;
#[cfg(feature = "dred")]
mod dred_rdovae_dec;
#[cfg(feature = "dred")]
mod dred_rdovae_dec_data;
mod dred_stats_data;
pub mod extensions;
pub mod mapping_matrix;
mod math;
mod mlp;
mod mlp_data;
#[cfg(feature = "dred")]
mod nnet;
pub mod oggreader;
pub mod opus;
pub mod opus_decoder;
pub mod opus_encoder;
pub mod opus_multistream;
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
