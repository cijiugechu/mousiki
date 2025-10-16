//! CELT module internals.
//!
//! This module contains foundational types for the Rust port of the CELT
//! implementation.  The definitions are intentionally close to the original C
//! structures so that subsequent ports can translate field-by-field logic while
//! relying on Rust's ownership and lifetime tracking for safety.

mod bands;
#[allow(clippy::module_inception)]
mod celt;
mod celt_decoder;
mod celt_encoder;
mod cpu_support;
mod cwrs;
#[cfg(feature = "deep_plc")]
mod deep_plc;
mod entcode;
mod entdec;
mod entenc;
mod float_cast;
mod kiss_fft;
mod laplace;
mod lpc;
mod math;
pub(crate) mod math_fixed;
mod mdct;
mod mini_kfft;
mod modes;
mod pitch;
mod quant_bands;
mod rate;
mod types;
mod vq;

#[allow(unused_imports)]
pub(crate) use bands::*;
#[allow(unused_imports)]
pub(crate) use celt::*;
#[allow(unused_imports)]
pub(crate) use celt_decoder::*;
#[allow(unused_imports)]
pub(crate) use celt_encoder::*;
#[allow(unused_imports)]
pub(crate) use cpu_support::*;
#[allow(unused_imports)]
pub(crate) use cwrs::*;
#[cfg(feature = "deep_plc")]
#[allow(unused_imports)]
pub(crate) use deep_plc::*;
#[allow(unused_imports)]
pub(crate) use entcode::*;
#[allow(unused_imports)]
pub(crate) use entdec::*;
#[allow(unused_imports)]
pub(crate) use entenc::*;
#[allow(unused_imports)]
pub(crate) use float_cast::*;
#[allow(unused_imports)]
pub(crate) use kiss_fft::*;
#[allow(unused_imports)]
pub(crate) use laplace::*;
#[allow(unused_imports)]
pub(crate) use lpc::*;
#[allow(unused_imports)]
pub(crate) use math::*;
#[allow(unused_imports)]
pub(crate) use mdct::*;
#[allow(unused_imports)]
pub(crate) use mini_kfft::*;
#[allow(unused_imports)]
pub(crate) use modes::*;
#[allow(unused_imports)]
pub(crate) use pitch::*;
#[allow(unused_imports)]
pub(crate) use quant_bands::*;
#[allow(unused_imports)]
pub(crate) use rate::*;
#[allow(unused_imports)]
pub(crate) use types::*;
#[allow(unused_imports)]
pub(crate) use vq::*;
