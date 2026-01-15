// DRED stats data live in the weights crate when the feature is enabled.

#[cfg(feature = "dred")]
pub use mousiki_dred_weights::dred_stats_data::*;

#[cfg(not(feature = "dred"))]
pub(crate) const DRED_LATENT_QUANT_SCALES_Q8: [u8; 336] = [0; 336];
#[cfg(not(feature = "dred"))]
pub(crate) const DRED_LATENT_DEAD_ZONE_Q8: [u8; 336] = [0; 336];
#[cfg(not(feature = "dred"))]
pub(crate) const DRED_LATENT_R_Q8: [u8; 336] = [0; 336];
#[cfg(not(feature = "dred"))]
pub(crate) const DRED_LATENT_P0_Q8: [u8; 336] = [0; 336];
#[cfg(not(feature = "dred"))]
pub(crate) const DRED_STATE_QUANT_SCALES_Q8: [u8; 304] = [0; 304];
#[cfg(not(feature = "dred"))]
pub(crate) const DRED_STATE_DEAD_ZONE_Q8: [u8; 304] = [0; 304];
#[cfg(not(feature = "dred"))]
pub(crate) const DRED_STATE_R_Q8: [u8; 304] = [0; 304];
#[cfg(not(feature = "dred"))]
pub(crate) const DRED_STATE_P0_Q8: [u8; 304] = [0; 304];
