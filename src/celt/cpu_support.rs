#![allow(dead_code)]

//! Runtime CPU feature detection helpers.
//!
//! The reference implementation exposes `opus_select_arch()` through
//! `celt/cpu_support.h`. When runtime CPU detection is not enabled, the C code
//! falls back to a stub that always returns zero. This module ports that
//! behaviour so the Rust translation can depend on the same helper without
//! pulling in the platform-specific assembly back-ends yet.

/// Bitmask describing the available architecture-specific optimisations.
///
/// In the reference build this value expands to zero when runtime CPU
/// detection is disabled, matching the behaviour of this initial port.
pub(crate) const OPUS_ARCHMASK: i32 = 0;

/// Selects the architecture variant for CELT's optional optimised kernels.
///
/// This mirrors the fallback inline function defined in `celt/cpu_support.h`
/// which returns zero when runtime dispatch support is disabled. Future ports
/// can extend this implementation to detect platform features dynamically.
#[inline]
pub(crate) fn opus_select_arch() -> i32 {
    0
}

#[cfg(test)]
mod tests {
    use super::{OPUS_ARCHMASK, opus_select_arch};

    #[test]
    fn opus_select_arch_matches_default_stub() {
        assert_eq!(opus_select_arch(), 0);
        assert_eq!(OPUS_ARCHMASK, 0);
    }
}
