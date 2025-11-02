//! Port of the fixed-point `silk_A2NLSF` helper from the SILK reference
//! implementation (`silk/A2NLSF.c`). The routine converts monic prediction
//! filter coefficients (Q16) into normalised line spectral frequencies (NLSFs)
//! in Q15 precision, mirroring the polynomial transformations, root searches,
//! and bandwidth expansion fallbacks used by the original C code.

use super::bwexpander_32::bwexpander_32;
use super::lpc_inv_pred_gain::SILK_MAX_ORDER_LPC;
use super::table_lsf_cos::{LSF_COS_TAB_SZ_FIX, SILK_LSF_COS_TAB_FIX_Q12};

const BIN_DIV_STEPS_A2NLSF_FIX: usize = 3;
const MAX_ITERATIONS_A2NLSF_FIX: usize = 16;

/// Convert prediction filter coefficients (Q16) to normalised line spectral
/// frequencies (Q15).
///
/// The input slice must contain an even number of coefficients not exceeding
/// `SILK_MAX_ORDER_LPC`. On success the output slice receives the computed
/// NLSFs in ascending order.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn a2nlsf(nlsf_q15: &mut [i16], a_q16: &mut [i32]) {
    let d = nlsf_q15.len();
    assert_eq!(d, a_q16.len(), "filter order mismatch");
    assert!(d % 2 == 0, "SILK requires an even LPC order");
    assert!(d <= SILK_MAX_ORDER_LPC, "order exceeds SILK_MAX_ORDER_LPC");

    let dd = d / 2;
    let mut p = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    let mut q = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];

    a2nlsf_init(&a_q16[..d], &mut p[..=dd], &mut q[..=dd], dd);

    let mut xlo = i32::from(SILK_LSF_COS_TAB_FIX_Q12[0]);
    let mut ylo = a2nlsf_eval_poly(&p[..=dd], xlo, dd);
    let mut root_ix = 0usize;

    if ylo < 0 {
        nlsf_q15[0] = 0;
        root_ix = 1;
        ylo = a2nlsf_eval_poly(&q[..=dd], xlo, dd);
    }

    let mut k = 1usize;
    let mut iteration = 0usize;
    let mut thr = 0;

    while root_ix < d {
        if k > LSF_COS_TAB_SZ_FIX {
            iteration += 1;
            if iteration > MAX_ITERATIONS_A2NLSF_FIX {
                let spacing = div32_16(1 << 15, (d + 1) as i32) as i16;
                nlsf_q15[0] = spacing;
                for idx in 1..d {
                    nlsf_q15[idx] = nlsf_q15[idx - 1].wrapping_add(spacing);
                }
                return;
            }

            let chirp_q16 = (1 << 16) - (1 << iteration);
            bwexpander_32(&mut a_q16[..d], chirp_q16);
            a2nlsf_init(&a_q16[..d], &mut p[..=dd], &mut q[..=dd], dd);

            xlo = i32::from(SILK_LSF_COS_TAB_FIX_Q12[0]);
            ylo = a2nlsf_eval_poly(&p[..=dd], xlo, dd);
            if ylo < 0 {
                nlsf_q15[0] = 0;
                root_ix = 1;
                ylo = a2nlsf_eval_poly(&q[..=dd], xlo, dd);
            } else {
                root_ix = 0;
            }

            k = 1;
            thr = 0;
            continue;
        }

        let mut xhi = i32::from(SILK_LSF_COS_TAB_FIX_Q12[k]);
        let mut yhi = if (root_ix & 1) == 0 {
            a2nlsf_eval_poly(&p[..=dd], xhi, dd)
        } else {
            a2nlsf_eval_poly(&q[..=dd], xhi, dd)
        };

        if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
            thr = if yhi == 0 { 1 } else { 0 };

            let mut ffrac = -256;
            for m in 0..BIN_DIV_STEPS_A2NLSF_FIX {
                let xmid = rshift_round32(xlo.wrapping_add(xhi), 1);
                let ymid = if (root_ix & 1) == 0 {
                    a2nlsf_eval_poly(&p[..=dd], xmid, dd)
                } else {
                    a2nlsf_eval_poly(&q[..=dd], xmid, dd)
                };

                if (ylo <= 0 && ymid >= 0) || (ylo >= 0 && ymid <= 0) {
                    xhi = xmid;
                    yhi = ymid;
                } else {
                    xlo = xmid;
                    ylo = ymid;
                    ffrac = add_rshift(ffrac, 128, m);
                }
            }

            if ylo.abs() < 65_536 {
                let den = ylo.wrapping_sub(yhi);
                if den != 0 {
                    let nom =
                        (ylo << (8 - BIN_DIV_STEPS_A2NLSF_FIX)).wrapping_add(rshift32(den, 1));
                    ffrac = ffrac.wrapping_add(div32(nom, den));
                }
            } else {
                let denom = rshift32(ylo.wrapping_sub(yhi), 8 - BIN_DIV_STEPS_A2NLSF_FIX);
                if denom != 0 {
                    ffrac = ffrac.wrapping_add(div32(ylo, denom));
                }
            }

            let value = ((k as i32) << 8).wrapping_add(ffrac);
            let clamped = value.clamp(0, i32::from(i16::MAX));
            debug_assert!(clamped >= 0, "NLSF values must be non-negative");
            nlsf_q15[root_ix] = clamped as i16;

            root_ix += 1;
            if root_ix >= d {
                break;
            }

            xlo = i32::from(SILK_LSF_COS_TAB_FIX_Q12[k - 1]);
            ylo = (1 - ((root_ix & 2) as i32)) << 12;
        } else {
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;
        }
    }
}

fn a2nlsf_trans_poly(poly: &mut [i32], dd: usize) {
    for k in 2..=dd {
        for n in ((k + 1)..=dd).rev() {
            let idx = n - 2;
            poly[idx] = poly[idx].wrapping_sub(poly[n]);
        }
        let idx = k - 2;
        poly[idx] = poly[idx].wrapping_sub(poly[k] << 1);
    }
}

fn a2nlsf_eval_poly(poly: &[i32], x: i32, dd: usize) -> i32 {
    let mut y32 = poly[dd];
    let x_q16 = x << 4;

    if dd == 8 {
        y32 = smlaaw(poly[7], y32, x_q16);
        y32 = smlaaw(poly[6], y32, x_q16);
        y32 = smlaaw(poly[5], y32, x_q16);
        y32 = smlaaw(poly[4], y32, x_q16);
        y32 = smlaaw(poly[3], y32, x_q16);
        y32 = smlaaw(poly[2], y32, x_q16);
        y32 = smlaaw(poly[1], y32, x_q16);
        smlaaw(poly[0], y32, x_q16)
    } else {
        for n in (0..dd).rev() {
            y32 = smlaaw(poly[n], y32, x_q16);
        }
        y32
    }
}

fn a2nlsf_init(a_q16: &[i32], p: &mut [i32], q: &mut [i32], dd: usize) {
    p[dd] = 1 << 16;
    q[dd] = 1 << 16;

    for k in 0..dd {
        let even = a_q16[dd - k - 1];
        let odd = a_q16[dd + k];
        let sum = even.wrapping_add(odd);
        let diff = odd.wrapping_sub(even);
        p[k] = sum.wrapping_neg();
        q[k] = diff;
    }

    for k in (1..=dd).rev() {
        p[k - 1] = p[k - 1].wrapping_sub(p[k]);
        q[k - 1] = q[k - 1].wrapping_add(q[k]);
    }

    a2nlsf_trans_poly(p, dd);
    a2nlsf_trans_poly(q, dd);
}

fn add_rshift(value: i32, addend: i32, shift: usize) -> i32 {
    value.wrapping_add(addend >> shift)
}

fn rshift_round32(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        return value;
    }

    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn rshift32(value: i32, shift: usize) -> i32 {
    if shift == 0 { value } else { value >> shift }
}

fn div32(a: i32, b: i32) -> i32 {
    a / b
}

fn div32_16(a: i32, b: i32) -> i32 {
    a / b
}

fn smlaaw(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((i64::from(b) * i64::from(c)) >> 16) as i32)
}

#[cfg(test)]
mod tests {
    use super::SILK_MAX_ORDER_LPC;
    use super::a2nlsf;

    #[test]
    fn matches_reference_output_for_known_lpc() {
        let order = 16;
        let mut a_q16 = [0i32; SILK_MAX_ORDER_LPC];
        a_q16[..order].copy_from_slice(&[
            15520, 2208, 4400, 8720, -1360, 13632, 1152, 7184, -1312, -6496, -15904, 3872, 11968,
            -10720, 8272, -7616,
        ]);

        let mut output = [0i16; SILK_MAX_ORDER_LPC];
        a2nlsf(&mut output[..order], &mut a_q16[..order]);

        let expected = [
            1496, 2925, 5334, 8052, 9524, 10640, 13688, 15291, 16759, 19462, 21048, 22212, 25217,
            26443, 29500, 31037,
        ];

        assert_eq!(&output[..order], &expected);
    }
}
