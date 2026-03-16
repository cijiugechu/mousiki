#![no_std]

extern crate alloc;

#[cfg(test)]
mod test_trace;

mod analysis;
pub mod bitdepth;
mod celt;
pub mod decoder;
#[cfg(feature = "deep_plc")]
mod dnn_utils;
#[cfg(feature = "dred")]
mod dnn_weights;
pub mod dred;
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
pub mod extensions;
#[cfg(feature = "deep_plc")]
pub mod fargan;
#[cfg(feature = "dred")]
mod lpcnet_enc;
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
#[cfg(feature = "dred")]
mod pitchdnn;
#[cfg(feature = "dred")]
mod pitchdnn_data;
#[cfg(feature = "deep_plc")]
mod plc_model;
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
