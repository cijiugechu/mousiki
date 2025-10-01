#![allow(dead_code)]

//! Quantisation helpers for band energies.
//!
//! This module gathers small routines from `celt/quant_bands.c` that have few
//! dependencies so they can be ported in isolation. The helpers operate on the
//! logarithmic band energy buffers shared between the encoder and decoder.

use crate::celt::types::CeltGlog;

/// Mean band energies mirroring `eMeans` from `celt/quant_bands.c`.
#[allow(dead_code)]
pub(crate) const E_MEANS: [f32; 25] = [
    6.437_5, 6.25, 5.75, 5.312_5, 5.062_5, 4.812_5, 4.5, 4.375, 4.875, 4.687_5,
    4.562_5, 4.437_5, 4.875, 4.625, 4.312_5, 4.5, 4.375, 4.625, 4.75, 4.437_5,
    3.75, 3.75, 3.75, 3.75, 3.75,
];

/// Prediction coefficients (`pred_coef`) converted to floating point.
#[allow(dead_code)]
pub(crate) const PRED_COEF: [f32; 4] = [
    29_440.0 / 32_768.0,
    26_112.0 / 32_768.0,
    21_248.0 / 32_768.0,
    16_384.0 / 32_768.0,
];

/// `beta_coef` prediction feedback constants from the reference implementation.
#[allow(dead_code)]
pub(crate) const BETA_COEF: [f32; 4] = [
    30_147.0 / 32_768.0,
    22_282.0 / 32_768.0,
    12_124.0 / 32_768.0,
    6_554.0 / 32_768.0,
];

/// Intra-frame beta coefficient (`beta_intra`) from `celt/quant_bands.c`.
#[allow(dead_code)]
pub(crate) const BETA_INTRA: f32 = 4_915.0 / 32_768.0;

/// Laplace model parameters (`e_prob_model`) indexed by frame size, prediction
/// type, and band.
#[allow(dead_code)]
pub(crate) const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78,
            92, 79, 92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161,
            12, 176, 10, 177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114,
            70, 96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37,
            97, 43, 78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36,
            117, 34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190,
            8, 177, 9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80,
            92, 66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97,
            38, 77, 45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132,
            27, 136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6,
            175, 9, 159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72,
            105, 58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98,
            35, 77, 42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

/// Small energy inverse CDF table from `celt/quant_bands.c`.
#[allow(dead_code)]
pub(crate) const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

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
    use super::{
        loss_distortion, BETA_COEF, BETA_INTRA, E_MEANS, E_PROB_MODEL, PRED_COEF, SMALL_ENERGY_ICDF,
    };

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

    #[test]
    fn quant_bands_constants_match_reference_values() {
        let expected_means = [
            6.437_5, 6.25, 5.75, 5.312_5, 5.062_5, 4.812_5, 4.5, 4.375, 4.875, 4.687_5, 4.562_5,
            4.437_5, 4.875, 4.625, 4.312_5, 4.5, 4.375, 4.625, 4.75, 4.437_5, 3.75, 3.75, 3.75,
            3.75, 3.75,
        ];
        assert_eq!(E_MEANS, expected_means);

        let expected_pred = [
            29_440.0 / 32_768.0,
            26_112.0 / 32_768.0,
            21_248.0 / 32_768.0,
            16_384.0 / 32_768.0,
        ];
        assert_eq!(PRED_COEF, expected_pred);

        let expected_beta = [
            30_147.0 / 32_768.0,
            22_282.0 / 32_768.0,
            12_124.0 / 32_768.0,
            6_554.0 / 32_768.0,
        ];
        assert_eq!(BETA_COEF, expected_beta);

        assert_eq!(BETA_INTRA, 4_915.0 / 32_768.0);
        assert_eq!(SMALL_ENERGY_ICDF, [2, 1, 0]);

        // Spot-check a couple of Laplace model entries to guard against typos.
        assert_eq!(E_PROB_MODEL[0][0][0], 72);
        assert_eq!(E_PROB_MODEL[0][1][1], 179);
        assert_eq!(E_PROB_MODEL[1][0][10], 90);
        assert_eq!(E_PROB_MODEL[2][1][20], 105);
        assert_eq!(E_PROB_MODEL[3][0][41], 15);
    }
}
