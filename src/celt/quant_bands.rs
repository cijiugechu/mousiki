#![allow(dead_code)]

//! Quantisation helpers for band energies.
//!
//! This module gathers small routines from `celt/quant_bands.c` that have few
//! dependencies so they can be ported in isolation. The helpers operate on the
//! logarithmic band energy buffers shared between the encoder and decoder.

use crate::celt::types::CeltGlog;

/// Returns a conservative distortion score between the current and previous
/// band energies.
///
/// Mirrors the `loss_distortion()` helper from `celt/quant_bands.c`. The C
/// routine iterates over the encoded bands for each channel, scales the
/// difference between the newly computed energies and the historical values,
/// and accumulates a squared error metric. In the floating-point build the
/// scaling macros collapse to no-ops, so the score is simply the sum of squared
/// differences, clamped to an upper bound of `200.0`.
pub(crate) fn loss_distortion(
    e_bands: &[CeltGlog],
    old_e_bands: &[CeltGlog],
    start: usize,
    end: usize,
    bands_per_channel: usize,
    channels: usize,
) -> f32 {
    assert_eq!(
        e_bands.len(),
        old_e_bands.len(),
        "energy buffers must match"
    );
    assert!(
        end <= bands_per_channel,
        "end band must lie within the channel span"
    );
    assert!(start <= end, "start band cannot exceed end band");
    assert!(channels * bands_per_channel <= e_bands.len());

    let mut distortion = 0.0f32;

    for channel in 0..channels {
        let base = channel * bands_per_channel;
        for band in start..end {
            let idx = base + band;
            let delta = e_bands[idx] - old_e_bands[idx];
            distortion += delta * delta;
        }
    }

    distortion.min(200.0)
}

#[cfg(test)]
mod tests {
    use super::loss_distortion;

    #[test]
    fn loss_distortion_matches_manual_accumulation() {
        let e = [
            // Channel 0
            -2.0f32, -1.0, 0.5, 0.25, // Channel 1
            1.5, 2.0, -0.75, 0.0,
        ];
        let old = [
            // Channel 0
            -1.5f32, -0.5, 0.0, 0.0, // Channel 1
            1.0, 1.25, -1.25, -0.25,
        ];

        let manual = e
            .iter()
            .zip(old.iter())
            .map(|(current, previous)| {
                let diff = current - previous;
                diff * diff
            })
            .sum::<f32>();

        let computed = loss_distortion(&e, &old, 0, 4, 4, 2);
        assert!((computed - manual).abs() <= f32::EPSILON * 32.0);
    }

    #[test]
    fn loss_distortion_clamps_to_upper_bound() {
        let e = [50.0f32; 6];
        let old = [0.0f32; 6];
        let result = loss_distortion(&e, &old, 0, 3, 3, 2);
        assert_eq!(result, 200.0);
    }
}
