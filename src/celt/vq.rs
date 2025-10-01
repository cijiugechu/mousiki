#![allow(dead_code)]

//! Vector quantisation helpers ported from `celt/vq.c`.
//!
//! The routines in this module operate on the normalised coefficient buffers
//! used by CELT's pulse shaping stage.  They are largely self-contained and
//! map closely to their C counterparts, making them ideal candidates for early
//! porting efforts.

use crate::celt::math::{celt_cos_norm, celt_div};
use crate::celt::types::OpusVal16;

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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{SPREAD_NORMAL, exp_rotation};

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
}
