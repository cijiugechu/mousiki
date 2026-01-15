// Auto-generated from `opus-c/dnn/dred_config.h` and `opus-c/dnn/dred_rdovae_constants.h`.

pub(crate) const DRED_NUM_FEATURES: usize = 20;
pub(crate) const DRED_LATENT_DIM: usize = 21;
pub(crate) const DRED_STATE_DIM: usize = 19;
pub(crate) const DRED_PADDED_LATENT_DIM: usize = 24;
pub(crate) const DRED_PADDED_STATE_DIM: usize = 24;
pub(crate) const DRED_NUM_QUANTIZATION_LEVELS: usize = 16;
pub(crate) const DRED_MAX_RNN_NEURONS: usize = 96;
pub(crate) const DRED_MAX_CONV_INPUTS: usize = 1536;
pub(crate) const DRED_ENC_MAX_RNN_NEURONS: usize = 1536;
pub(crate) const DRED_ENC_MAX_CONV_INPUTS: usize = 1536;
pub(crate) const DRED_DEC_MAX_RNN_NEURONS: usize = 96;

pub(crate) const DRED_MIN_BYTES: usize = 8;
pub(crate) const DRED_SILK_ENCODER_DELAY: i32 = 79 + 12 - 80;
pub(crate) const DRED_FRAME_SIZE: usize = 160;
pub(crate) const DRED_DFRAME_SIZE: usize = 2 * DRED_FRAME_SIZE;
pub(crate) const DRED_MAX_DATA_SIZE: usize = 1000;
pub(crate) const DRED_ENC_Q0: i32 = 6;
pub(crate) const DRED_ENC_Q1: i32 = 15;
pub(crate) const DRED_MAX_LATENTS: usize = 26;
pub(crate) const DRED_NUM_REDUNDANCY_FRAMES: usize = 2 * DRED_MAX_LATENTS;
pub(crate) const DRED_MAX_FRAMES: usize = 4 * DRED_MAX_LATENTS;
