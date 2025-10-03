#![allow(dead_code)]

//! Vector quantisation helpers ported from `celt/vq.c`.
//!
//! The routines in this module operate on the normalised coefficient buffers
//! used by CELT's pulse shaping stage.  They are largely self-contained and
//! map closely to their C counterparts, making them ideal candidates for early
//! porting efforts.

use crate::celt::entcode::celt_udiv;
use crate::celt::math::{celt_cos_norm, celt_div, celt_rsqrt_norm};
use crate::celt::types::{OpusVal16, OpusVal32};

/// Spread decisions mirrored from `celt/bands.h`.
pub(crate) const SPREAD_NONE: i32 = 0;
pub(crate) const SPREAD_LIGHT: i32 = 1;
pub(crate) const SPREAD_NORMAL: i32 = 2;
pub(crate) const SPREAD_AGGRESSIVE: i32 = 3;

const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
const Q15_ONE: OpusVal16 = 1.0;

fn exp_rotation1(x: &mut [OpusVal16], stride: usize, c: OpusVal16, s: OpusVal16) {
    let len = x.len();
    if stride == 0 || len <= stride {
        return;
    }

    let ms = -s;

    for i in 0..(len - stride) {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }

    if len > 2 * stride {
        let limit = len - 2 * stride - 1;
        for i in (0..=limit).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            x[i + stride] = c * x2 + s * x1;
            x[i] = c * x1 + ms * x2;
        }
    }
}

/// Port of `exp_rotation()` from `celt/vq.c`.
///
/// Applies a spreading rotation to the coefficient buffer in-place.  The logic
/// matches the float build of the reference implementation, relying on Rust's
/// slice handling for safety while keeping the numerical behaviour intact.
pub(crate) fn exp_rotation(
    x: &mut [OpusVal16],
    len: usize,
    dir: i32,
    stride: usize,
    k: i32,
    spread: i32,
) {
    if len == 0 || stride == 0 {
        return;
    }

    let slice_len = len.min(x.len());
    let x = &mut x[..slice_len];

    if 2 * k >= slice_len as i32 || spread == SPREAD_NONE {
        return;
    }

    let spread_index = match spread {
        SPREAD_LIGHT => 0,
        SPREAD_NORMAL => 1,
        SPREAD_AGGRESSIVE => 2,
        _ => return,
    };

    let factor = SPREAD_FACTOR[spread_index];
    let gain = celt_div(
        Q15_ONE * slice_len as OpusVal16,
        (slice_len + (factor * k) as usize) as OpusVal16,
    );
    let theta = 0.5 * gain * gain;
    let c = celt_cos_norm(theta);
    let s = celt_cos_norm(Q15_ONE - theta);

    let mut stride2 = 0usize;
    if slice_len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < slice_len {
            stride2 += 1;
        }
    }

    let len_div = slice_len / stride;
    if len_div == 0 {
        return;
    }

    for band in 0..stride {
        let start = band * len_div;
        let end = start + len_div;
        let band_slice = &mut x[start..end];
        if dir < 0 {
            if stride2 > 0 {
                exp_rotation1(band_slice, stride2, s, c);
            }
            exp_rotation1(band_slice, 1, c, s);
        } else {
            exp_rotation1(band_slice, 1, c, -s);
            if stride2 > 0 {
                exp_rotation1(band_slice, stride2, s, -c);
            }
        }
    }
}

/// Port of `normalise_residual()` from `celt/vq.c`.
///
/// The helper mixes the decoded PVQ pulses with the pitch vector so that the
/// resulting excitation has unit energy. The float build performs the scaling
/// by computing the reciprocal square root of the accumulated pulse energy and
/// multiplying by the supplied gain. The Rust port follows the same approach
/// while clamping the slice lengths to avoid overruns.
pub(crate) fn normalise_residual(
    pulses: &[i32],
    x: &mut [OpusVal16],
    n: usize,
    ryy: OpusVal32,
    gain: OpusVal32,
) {
    if n == 0 {
        return;
    }

    debug_assert!(pulses.len() >= n, "pulse buffer shorter than band size");
    debug_assert!(x.len() >= n, "output buffer shorter than band size");

    let len = n.min(pulses.len()).min(x.len());
    if len == 0 {
        return;
    }

    let scale = celt_rsqrt_norm(ryy) * gain;
    for (dst, &pulse) in x.iter_mut().take(len).zip(pulses.iter()) {
        *dst = scale * pulse as OpusVal16;
    }
}

/// Mirrors `extract_collapse_mask()` from `celt/vq.c`.
///
/// The helper inspects the quantised PVQ pulses and determines which of the
/// folded bands received any energy. The mask is later used to decide whether
/// a band "collapsed" during quantisation and therefore needs spectral
/// spreading in the decoder.
#[must_use]
pub(crate) fn extract_collapse_mask(pulses: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }

    if n == 0 {
        return 0;
    }

    debug_assert!(pulses.len() >= n, "pulse buffer shorter than band size");

    let n0 = celt_udiv(n as u32, b as u32) as usize;
    debug_assert!(n0 > 0, "sub-band width must be non-zero");

    let mut collapse_mask = 0u32;
    for band in 0..b {
        let start = band * n0;
        let end = (start + n0).min(pulses.len());
        let mut accumulator = 0;
        for &value in &pulses[start..end] {
            accumulator |= value;
        }
        if accumulator != 0 {
            collapse_mask |= 1 << band;
        }
    }

    collapse_mask
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{SPREAD_NORMAL, exp_rotation, extract_collapse_mask, normalise_residual};

    fn seed_samples(len: usize) -> Vec<f32> {
        let mut seed = 0x1234_5678u32;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let sample = ((seed >> 16) & 0x7fff) as i32 - 16_384;
            out.push(sample as f32);
        }
        out
    }

    fn snr_db(original: &[f32], processed: &[f32]) -> f64 {
        let mut err = 0.0;
        let mut ener = 0.0;
        for (&orig, &proc) in original.iter().zip(processed.iter()) {
            let diff = (orig - proc) as f64;
            err += diff * diff;
            ener += (orig as f64) * (orig as f64);
        }
        if err == 0.0 {
            return f64::INFINITY;
        }
        20.0 * (ener / err).log10()
    }

    fn rotation_case(len: usize, k: i32) {
        let baseline = seed_samples(len);
        let mut rotated = baseline.clone();

        exp_rotation(&mut rotated, len, 1, 1, k, SPREAD_NORMAL);
        let forward_snr = snr_db(&baseline, &rotated);

        exp_rotation(&mut rotated, len, -1, 1, k, SPREAD_NORMAL);
        let inverse_snr = snr_db(&baseline, &rotated);

        assert!(inverse_snr > 60.0, "inverse SNR too low: {inverse_snr}");
        assert!(
            forward_snr < 20.0,
            "forward SNR unexpectedly high: {forward_snr}"
        );
    }

    #[test]
    fn rotation_matches_reference_behaviour() {
        for &(len, k) in &[(15, 3), (23, 5), (50, 3), (80, 1)] {
            rotation_case(len, k);
        }
    }

    #[test]
    fn residual_normalisation_scales_by_gain() {
        let pulses = [2, -1, 0, 3];
        let mut output = [0.0f32; 4];
        let ryy = 25.0;
        let gain = 0.5;

        normalise_residual(&pulses, &mut output, pulses.len(), ryy, gain);

        let expected_scale = gain * (1.0 / ryy.sqrt());
        for (value, &pulse) in output.iter().zip(&pulses) {
            let expected = expected_scale * pulse as f32;
            assert!(
                (value - expected).abs() <= 1e-6,
                "expected {expected}, found {value}"
            );
        }
    }

    #[test]
    fn collapse_mask_sets_bits_for_active_bands() {
        let pulses = [0, 0, 1, 0, 0, 0, 2, 3];
        let mask = extract_collapse_mask(&pulses, pulses.len(), 4);
        assert_eq!(mask, 0b1010);
    }

    #[test]
    fn collapse_mask_for_single_band_is_one() {
        let pulses = [0, 0, 0, 0];
        assert_eq!(extract_collapse_mask(&pulses, pulses.len(), 1), 1);
    }
}
