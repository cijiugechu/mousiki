//! SILK encoder state scaffolding.
//!
//! This module gradually mirrors the encoder-side state structures defined in
//! `silk/structs.h`.  Only the fields required by the already-ported helpers are
//! modelled for now; additional members will be introduced alongside their Rust
//! counterparts.

pub mod state;

pub use state::{
    ENCODER_NUM_CHANNELS, Encoder, EncoderChannelState, EncoderStateCommon, VAD_N_BANDS,
};
