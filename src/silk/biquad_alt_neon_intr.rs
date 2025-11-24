//! Stub NEON entry point mirroring `silk/arm/biquad_alt_neon_intr.c`.
//!
//! The Rust port does not yet wire up ARM runtime CPU detection, so this
//! implementation simply delegates to the scalar `biquad_alt_stride2` helper
//! while keeping the NEON symbol available for future dispatch wiring.

use crate::silk::biquad_alt::biquad_alt_stride2;

#[allow(clippy::arithmetic_side_effects)]
pub fn biquad_alt_stride2_neon(
    input: &[i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    state: &mut [i32; 4],
    output: &mut [i16],
) {
    biquad_alt_stride2(input, b_q28, a_q28, state, output);
}
