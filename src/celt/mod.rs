//! CELT module internals.
//!
//! This module contains foundational types for the Rust port of the CELT
//! implementation.  The definitions are intentionally close to the original C
//! structures so that subsequent ports can translate field-by-field logic while
//! relying on Rust's ownership and lifetime tracking for safety.

mod types;

#[allow(unused_imports)]
pub(crate) use types::*;
