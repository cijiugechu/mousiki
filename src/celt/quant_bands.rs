#![allow(dead_code)]

//! Quantisation helpers for band energies.
//!
//! This module gathers small routines from `celt/quant_bands.c` that have few
//! dependencies so they can be ported in isolation. The helpers operate on the
//! logarithmic band energy buffers shared between the encoder and decoder.

use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::celt::rate::MAX_FINE_BITS;
use crate::celt::types::{CeltGlog, OpusCustomMode};
use libm::floorf;

const INV_Q15: f32 = 1.0 / 16_384.0;

/// Mean band energies mirroring `eMeans` from `celt/quant_bands.c`.
#[allow(dead_code)]
pub(crate) const E_MEANS: [f32; 25] = [
    6.437_5, 6.25, 5.75, 5.312_5, 5.062_5, 4.812_5, 4.5, 4.375, 4.875, 4.687_5, 4.562_5, 4.437_5,
    4.875, 4.625, 4.312_5, 4.5, 4.375, 4.625, 4.75, 4.437_5, 3.75, 3.75, 3.75, 3.75, 3.75,
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
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
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

fn fine_energy_scale(fine: usize) -> f32 {
    debug_assert!(fine <= 14);
    let shift = 14usize.saturating_sub(fine);
    ((1usize << shift) as f32) * INV_Q15
}

fn fine_energy_final_scale(fine: usize) -> f32 {
    debug_assert!(fine <= 13);
    let shift = 14usize.saturating_sub(fine + 1);
    ((1usize << shift) as f32) * INV_Q15
}

/// Quantises the finer energy resolution bits for each band.
///
/// Mirrors the float portion of `quant_fine_energy()` from
/// `celt/quant_bands.c`. The function scans the requested bands, quantises the
/// fractional energy error, and accumulates the residual back into
/// `old_ebands`/`error` while appending the raw bits to `enc`.
pub(crate) fn quant_fine_energy(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_ebands: &mut [CeltGlog],
    error: &mut [CeltGlog],
    fine_quant: &[i32],
    enc: &mut EcEnc<'_>,
    channels: usize,
) {
    assert_eq!(old_ebands.len(), error.len(), "band buffers must align");
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(fine_quant.len() >= end, "fine quantiser metadata too short");
    assert!(
        channels * mode.num_ebands <= old_ebands.len(),
        "insufficient band data"
    );

    let stride = mode.num_ebands;

    for band in start..end {
        let fine = fine_quant[band];
        if fine <= 0 {
            continue;
        }

        let fine_bits = fine as usize;
        let frac = 1i32 << fine_bits;
        let max_q = frac - 1;
        let scale = fine_energy_scale(fine_bits);

        for channel in 0..channels {
            let idx = channel * stride + band;
            let target = error[idx] + 0.5;
            let mut q2 = floorf(target * frac as f32) as i32;
            q2 = q2.clamp(0, max_q);

            enc.enc_bits(q2 as u32, fine_bits as u32);

            let offset = (q2 as f32 + 0.5) * scale - 0.5;
            old_ebands[idx] += offset;
            error[idx] -= offset;
        }
    }
}

/// Consumes the remaining fine energy bits based on their priority.
///
/// Ports the float implementation of `quant_energy_finalise()` which allocates
/// leftover bits to low-priority bands and updates the running energy/error
/// estimates accordingly.
pub(crate) fn quant_energy_finalise(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_ebands: &mut [CeltGlog],
    error: &mut [CeltGlog],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut EcEnc<'_>,
    channels: usize,
) {
    assert_eq!(old_ebands.len(), error.len(), "band buffers must align");
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(fine_quant.len() >= end, "fine quantiser metadata too short");
    assert!(
        fine_priority.len() >= end,
        "fine priority metadata too short"
    );
    assert!(
        channels * mode.num_ebands <= old_ebands.len(),
        "insufficient band data"
    );

    let stride = mode.num_ebands;
    let channels_i32 = channels as i32;

    for priority in 0..2 {
        for band in start..end {
            if bits_left < channels_i32 {
                break;
            }

            let fine = fine_quant[band];
            if fine >= MAX_FINE_BITS || fine_priority[band] != priority {
                continue;
            }

            let fine_bits = fine.max(0) as usize;
            let scale = fine_energy_final_scale(fine_bits);

            for channel in 0..channels {
                let idx = channel * stride + band;
                let q2 = if error[idx] < 0.0 { 0 } else { 1 };
                enc.enc_bits(q2 as u32, 1);

                let offset = (q2 as f32 - 0.5) * scale;
                old_ebands[idx] += offset;
                error[idx] -= offset;
                bits_left -= 1;
            }
        }
    }
}

/// Restores the fine energy quantisation from the bit-stream.
///
/// Mirrors the float path of `unquant_fine_energy()` by replaying the raw bits
/// written by [`quant_fine_energy`] and accumulating the reconstructed energy
/// back into `old_ebands`.
pub(crate) fn unquant_fine_energy(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_ebands: &mut [CeltGlog],
    fine_quant: &[i32],
    dec: &mut EcDec<'_>,
    channels: usize,
) {
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(fine_quant.len() >= end, "fine quantiser metadata too short");
    assert!(
        channels * mode.num_ebands <= old_ebands.len(),
        "insufficient band data"
    );

    let stride = mode.num_ebands;

    for band in start..end {
        let fine = fine_quant[band];
        if fine <= 0 {
            continue;
        }

        let fine_bits = fine as usize;
        let scale = fine_energy_scale(fine_bits);

        for channel in 0..channels {
            let idx = channel * stride + band;
            let q2 = dec.dec_bits(fine_bits as u32) as i32;
            let offset = (q2 as f32 + 0.5) * scale - 0.5;
            old_ebands[idx] += offset;
        }
    }
}

/// Replays the final fine energy decisions for the decoder.
///
/// Ports the float build of `unquant_energy_finalise()` which consumes the
/// leftover single-bit decisions and updates the reconstructed energy buffer.
pub(crate) fn unquant_energy_finalise(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    old_ebands: &mut [CeltGlog],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut EcDec<'_>,
    channels: usize,
) {
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(fine_quant.len() >= end, "fine quantiser metadata too short");
    assert!(
        fine_priority.len() >= end,
        "fine priority metadata too short"
    );
    assert!(
        channels * mode.num_ebands <= old_ebands.len(),
        "insufficient band data"
    );

    let stride = mode.num_ebands;
    let channels_i32 = channels as i32;

    for priority in 0..2 {
        for band in start..end {
            if bits_left < channels_i32 {
                break;
            }

            let fine = fine_quant[band];
            if fine >= MAX_FINE_BITS || fine_priority[band] != priority {
                continue;
            }

            let fine_bits = fine.max(0) as usize;
            let scale = fine_energy_final_scale(fine_bits);

            for channel in 0..channels {
                let idx = channel * stride + band;
                let q2 = dec.dec_bits(1) as i32;
                let offset = (q2 as f32 - 0.5) * scale;
                old_ebands[idx] += offset;
                bits_left -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{
        BETA_COEF, BETA_INTRA, E_MEANS, E_PROB_MODEL, PRED_COEF, SMALL_ENERGY_ICDF,
        loss_distortion, quant_energy_finalise, quant_fine_energy, unquant_energy_finalise,
        unquant_fine_energy,
    };
    use crate::celt::entdec::EcDec;
    use crate::celt::entenc::EcEnc;
    use crate::celt::types::{MdctLookup, OpusCustomMode, PulseCacheData};

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

    #[test]
    fn fine_energy_quantisation_round_trips() {
        let e_bands = [0i16; 4];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 4];
        let window = [0.0f32; 4];
        let twiddle = [0.0f32; 1];
        let mdct = MdctLookup::new(1, 0, &twiddle);
        let mode = OpusCustomMode::new(
            48_000,
            0,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            PulseCacheData::default(),
        );

        let mut encoded_old = [0.0f32; 8];
        let mut error = [0.6, -0.25, 0.1, -0.05, 0.4, -0.3, 0.0, 0.2];
        let fine_quant = [2, 1, 0, 3];
        let fine_priority = [0, 1, 0, 1];
        let bits_left = 4;
        let channels = 2;

        let mut buffer = vec![0u8; 32];
        {
            let mut enc = EcEnc::new(&mut buffer);
            quant_fine_energy(
                &mode,
                0,
                4,
                &mut encoded_old,
                &mut error,
                &fine_quant,
                &mut enc,
                channels,
            );
            quant_energy_finalise(
                &mode,
                0,
                4,
                &mut encoded_old,
                &mut error,
                &fine_quant,
                &fine_priority,
                bits_left,
                &mut enc,
                channels,
            );
            enc.enc_done();
        }

        let mut decoded_old = [0.0f32; 8];
        {
            let mut dec = EcDec::new(&mut buffer);
            unquant_fine_energy(
                &mode,
                0,
                4,
                &mut decoded_old,
                &fine_quant,
                &mut dec,
                channels,
            );
            unquant_energy_finalise(
                &mode,
                0,
                4,
                &mut decoded_old,
                &fine_quant,
                &fine_priority,
                bits_left,
                &mut dec,
                channels,
            );
        }

        for (enc, dec) in encoded_old.iter().zip(decoded_old.iter()) {
            assert!((enc - dec).abs() <= 1e-6);
        }
    }
}
