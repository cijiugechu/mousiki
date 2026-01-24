//! CELT module internals.
//!
//! This module contains foundational types for the Rust port of the CELT
//! implementation.  The definitions are intentionally close to the original C
//! structures so that subsequent ports can translate field-by-field logic while
//! relying on Rust's ownership and lifetime tracking for safety.

mod arm_celt_map;
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
mod fft_bitrev_480;
mod fft_twiddles_48000_960;
mod mdct_twiddles_48000_960;
mod window_48000_960;
#[cfg(feature = "fixed_point")]
mod fixed_arch;
#[cfg(feature = "fixed_point")]
mod fixed_ops;
mod float_cast;
mod kiss_fft;
#[cfg(feature = "fixed_point")]
mod kiss_fft_fixed;
mod laplace;
mod lpc;
mod math;
pub(crate) mod math_fixed;
mod mdct;
#[cfg(feature = "fixed_point")]
mod mdct_fixed;
mod mini_kfft;
mod modes;
mod pitch;
mod quant_bands;
mod rate;
mod types;
mod vq;
mod x86_celt_map;

#[allow(unused_imports)]
pub(crate) use arm_celt_map::*;
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
#[cfg(feature = "fixed_point")]
#[allow(unused_imports)]
pub(crate) use fixed_arch::*;
#[cfg(feature = "fixed_point")]
#[allow(unused_imports)]
pub(crate) use fixed_ops::*;
#[allow(unused_imports)]
pub(crate) use float_cast::*;
#[allow(unused_imports)]
pub(crate) use kiss_fft::*;
#[cfg(feature = "fixed_point")]
#[allow(unused_imports)]
pub(crate) use kiss_fft_fixed::*;
#[allow(unused_imports)]
pub(crate) use laplace::*;
#[allow(unused_imports)]
pub(crate) use lpc::*;
pub(crate) use math::isqrt32;
#[allow(unused_imports)]
pub(crate) use math::*;
#[allow(unused_imports)]
pub(crate) use mdct::*;
#[cfg(feature = "fixed_point")]
#[allow(unused_imports)]
pub(crate) use mdct_fixed::*;
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
#[allow(unused_imports)]
pub(crate) use x86_celt_map::*;
