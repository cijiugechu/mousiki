//! Fixed-point residual energy helper from `silk/fixed/residual_energy16_FIX.c`.
//!
//! The routine computes the weighted prediction error
//! `wxx - 2 * wXx * c + c' * wXX * c` using only 32-bit arithmetic by
//! dynamically scaling the LPC vector `c`. It is primarily used by the encoder
//! when evaluating long-term prediction candidates.

use core::cmp::{max, min};

use super::MAX_LPC_ORDER;

const MAX_MATRIX_SIZE: usize = MAX_LPC_ORDER;

/// Mirrors `silk_residual_energy16_covar_FIX`.
///
/// * `c` — prediction vector holding at least `dim` entries.
/// * `w_xx` — flattened, symmetric `dim × dim` correlation matrix stored in row-major order.
/// * `w_xx_vec` — correlation vector with `dim` elements.
/// * `wxx` — scalar correlation term.
/// * `dim` — matrix/vector dimension (1–=`MAX_LPC_ORDER`).
/// * `c_q` — Q value describing the fixed-point precision of `c`.
#[allow(clippy::too_many_arguments)]
pub fn residual_energy16_covar(
    c: &[i16],
    w_xx: &[i32],
    w_xx_vec: &[i32],
    wxx: i32,
    dim: usize,
    c_q: i32,
) -> i32 {
    assert!((1..16).contains(&c_q), "c_q must be in the range 1..=15");
    assert!(dim > 0, "dimension must be positive");
    assert!(dim <= MAX_MATRIX_SIZE, "dimension exceeds MAX_MATRIX_SIZE");
    assert!(
        c.len() >= dim && w_xx_vec.len() >= dim && w_xx.len() >= dim * dim,
        "correlation inputs must provide at least `dim` elements"
    );

    let dim_i32 = dim as i32;
    let mut lshifts = 16 - c_q;
    let mut qxtra = lshifts;

    let mut c_max = 0;
    for &value in &c[..dim] {
        c_max = max(c_max, abs_i32(i32::from(value)));
    }
    qxtra = min(qxtra, clz32(c_max) - 17);

    let last = dim * dim - 1;
    let w_max = max(w_xx[0], w_xx[last]);
    let scaled = dim_i32.wrapping_mul(smulwb(w_max, c_max) >> 4);
    qxtra = min(qxtra, clz32(scaled) - 5);
    qxtra = max(qxtra, 0);

    let mut cn = [0i32; MAX_MATRIX_SIZE];
    for i in 0..dim {
        cn[i] = i32::from(c[i]).wrapping_shl(qxtra as u32);
        debug_assert!(abs_i32(cn[i]) <= i32::from(i16::MAX) + 1);
    }

    lshifts -= qxtra;

    let shift = (1 + lshifts) as u32;
    let mut tmp = 0;
    for i in 0..dim {
        tmp = smlawb(tmp, w_xx_vec[i], cn[i]);
    }
    let mut nrg = (wxx >> shift).wrapping_sub(tmp);

    let mut tmp2 = 0;
    for i in 0..dim {
        let mut row_acc = 0;
        let base = i * dim;
        for j in (i + 1)..dim {
            row_acc = smlawb(row_acc, w_xx[base + j], cn[j]);
        }
        let diag = w_xx[base + i] >> 1;
        row_acc = smlawb(row_acc, diag, cn[i]);
        tmp2 = smlawb(tmp2, row_acc, cn[i]);
    }
    nrg = add_lshift32(nrg, tmp2, lshifts);

    if nrg < 1 {
        1
    } else {
        let threshold = i32::MAX >> (lshifts + 2);
        if nrg > threshold {
            i32::MAX >> 1
        } else {
            nrg.wrapping_shl((lshifts + 1) as u32)
        }
    }
}

#[inline]
fn abs_i32(value: i32) -> i32 {
    let mask = value >> 31;
    (value ^ mask).wrapping_sub(mask)
}

#[inline]
fn clz32(value: i32) -> i32 {
    if value == 0 {
        32
    } else {
        (value as u32).leading_zeros() as i32
    }
}

#[inline]
fn smulwb(a: i32, b: i32) -> i32 {
    let b_low = i32::from(b as i16);
    ((i64::from(a) * i64::from(b_low)) >> 16) as i32
}

#[inline]
fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let c_low = i32::from(c as i16);
    let product = (i64::from(b) * i64::from(c_low)) >> 16;
    a.wrapping_add(product as i32)
}

#[inline]
fn add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    debug_assert!(shift >= 0);
    a.wrapping_add(b.wrapping_shl(shift as u32))
}

#[cfg(test)]
mod tests {
    use super::{MAX_MATRIX_SIZE, residual_energy16_covar};

    #[test]
    fn matches_reference_for_small_dimensions() {
        let c = [1234, -2500, 16384, -8192, 400];
        let w_xx_vec = [45_000, -12_000, 31_000, 4_200, -7_500];
        let w_xx = [
            70_000, 3_000, 2_000, 1_000, 500, //
            3_000, 68_000, 2_100, 1_200, 600, //
            2_000, 2_100, 66_000, 1_400, 700, //
            1_000, 1_200, 1_400, 64_000, 800, //
            500, 600, 700, 800, 62_000,
        ];
        for dim in 1..=5 {
            for &c_q in &[4, 9, 15] {
                let expected =
                    reference_residual_energy16(&c, &w_xx, &w_xx_vec, 1_234_567_890, dim, c_q);
                let actual = residual_energy16_covar(&c, &w_xx, &w_xx_vec, 1_234_567_890, dim, c_q);
                assert_eq!(
                    actual, expected,
                    "dim={dim}, c_q={c_q} produced mismatched energy"
                );
            }
        }
    }

    #[test]
    fn clamps_to_minimum_when_negative() {
        let c = [3000; MAX_MATRIX_SIZE];
        let w_xx_vec = [10_000; MAX_MATRIX_SIZE];
        let w_xx = [0; MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
        let result = residual_energy16_covar(&c, &w_xx, &w_xx_vec, 0, 3, 8);
        assert_eq!(result, 1);
    }

    #[test]
    fn clamps_to_maximum_when_headroom_exceeded() {
        let c = [0; MAX_MATRIX_SIZE];
        let mut w_xx = [0; MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
        w_xx[0] = 1;
        let w_xx_vec = [0; MAX_MATRIX_SIZE];
        let result = residual_energy16_covar(&c, &w_xx, &w_xx_vec, i32::MAX, 1, 8);
        assert_eq!(result, i32::MAX >> 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_residual_energy16(
        c: &[i16],
        w_xx: &[i32],
        w_xx_vec: &[i32],
        wxx: i32,
        dim: usize,
        c_q: i32,
    ) -> i32 {
        assert!(dim > 0 && dim <= MAX_MATRIX_SIZE);
        assert!((1..16).contains(&c_q));
        let mut lshifts = 16 - c_q;
        let mut qxtra = lshifts;

        let mut c_max = 0;
        for &value in &c[..dim] {
            c_max = c_max.max(abs(i32::from(value)));
        }
        qxtra = qxtra.min(clz(i32::from(c_max)) - 17);

        let last = dim * dim - 1;
        let w_max = w_xx[0].max(w_xx[last]);
        let scaled = (dim as i32).wrapping_mul(smulwb_ref(w_max, c_max) >> 4);
        qxtra = qxtra.min(clz(scaled) - 5);
        qxtra = qxtra.max(0);

        let mut cn = [0i32; MAX_MATRIX_SIZE];
        for i in 0..dim {
            cn[i] = i32::from(c[i]).wrapping_shl(qxtra as u32);
        }

        lshifts -= qxtra;
        let mut tmp: i32 = 0;
        for i in 0..dim {
            tmp = tmp.wrapping_add(smlawb_ref(0, w_xx_vec[i], cn[i]));
        }
        let mut nrg: i32 = (wxx >> ((1 + lshifts) as u32)).wrapping_sub(tmp);

        let mut tmp2: i32 = 0;
        for i in 0..dim {
            let mut row: i32 = 0;
            let base = i * dim;
            for j in (i + 1)..dim {
                row = row.wrapping_add(smlawb_ref(0, w_xx[base + j], cn[j]));
            }
            let diag = w_xx[base + i] >> 1;
            row = row.wrapping_add(smlawb_ref(0, diag, cn[i]));
            tmp2 = tmp2.wrapping_add(smlawb_ref(0, row, cn[i]));
        }

        nrg = nrg.wrapping_add(tmp2.wrapping_shl(lshifts as u32));
        if nrg < 1 {
            1
        } else {
            let threshold = i32::MAX >> (lshifts + 2);
            if nrg > threshold {
                i32::MAX >> 1
            } else {
                nrg.wrapping_shl((lshifts + 1) as u32)
            }
        }
    }

    #[inline]
    fn abs(value: i32) -> i32 {
        let mask = value >> 31;
        (value ^ mask).wrapping_sub(mask)
    }

    #[inline]
    fn clz(value: i32) -> i32 {
        if value == 0 {
            32
        } else {
            (value as u32).leading_zeros() as i32
        }
    }

    #[inline]
    fn smulwb_ref(a: i32, b: i32) -> i32 {
        let b_low = i32::from(b as i16);
        ((i64::from(a) * i64::from(b_low)) >> 16) as i32
    }

    #[inline]
    fn smlawb_ref(_: i32, b: i32, c: i32) -> i32 {
        let c_low = i32::from(c as i16);
        ((i64::from(b) * i64::from(c_low)) >> 16) as i32
    }
}
