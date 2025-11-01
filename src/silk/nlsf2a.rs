//! Port of the fixed-point `silk_NLSF2A` helper from the SILK reference
//! implementation (`silk/NLSF2A.c`). The routine converts a normalised line
//! spectral frequency (NLSF) vector into prediction filter coefficients (LPC) in
//! Q12 precision, mirroring the lookup-table interpolation, polynomial
//! generation, and stability checks performed by the original C code.

use core::cmp::Ordering;

use super::bwexpander_32::bwexpander_32;
use super::lpc_fit::lpc_fit;
use super::lpc_inv_pred_gain::{SILK_MAX_ORDER_LPC, lpc_inverse_pred_gain};
use super::table_lsf_cos::{LSF_COS_TAB_SZ_FIX, SILK_LSF_COS_TAB_FIX_Q12};

const QA: i32 = 16;
const MAX_LPC_STABILIZE_ITERATIONS: usize = 16;
const SHIFT_QA1_TO_Q12: i32 = QA + 1 - 12;

const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

/// Convert an NLSF vector in Q15 format into LPC coefficients in Q12.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn nlsf2a(a_q12: &mut [i16], nlsf_q15: &[i16], _arch: i32) {
    assert_eq!(a_q12.len(), nlsf_q15.len(), "order mismatch");
    let order = a_q12.len();
    assert!(
        order == 10 || order == 16,
        "SILK only supports 10 or 16 taps"
    );
    assert!(order <= SILK_MAX_ORDER_LPC, "order exceeds MAX_ORDER_LPC");

    let ordering = if order == 16 {
        &ORDERING16[..]
    } else {
        &ORDERING10[..]
    };

    let mut cos_lsf_qa = [0i32; SILK_MAX_ORDER_LPC];
    for (k, &index) in ordering.iter().enumerate().take(order) {
        let nlsf = i32::from(nlsf_q15[k]);
        assert!(nlsf >= 0, "NLSF values must be non-negative");

        let f_int = (nlsf >> (15 - 7)) as usize;
        let f_frac = nlsf - ((f_int as i32) << (15 - 7));
        assert!(
            f_int < LSF_COS_TAB_SZ_FIX,
            "index out of cosine table range"
        );

        let cos_val = i32::from(SILK_LSF_COS_TAB_FIX_Q12[f_int]);
        let delta =
            i32::from(SILK_LSF_COS_TAB_FIX_Q12[f_int + 1] - SILK_LSF_COS_TAB_FIX_Q12[f_int]);
        let interpolated = rshift_round64(
            (i64::from(cos_val) << 8) + i64::from(delta) * i64::from(f_frac),
            20 - QA,
        );
        cos_lsf_qa[index] = interpolated;
    }

    let dd = order / 2;

    let mut p = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    let mut q = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    nlsf2a_find_poly(&mut p[..=dd], &cos_lsf_qa[..order], dd);
    nlsf2a_find_poly(&mut q[..=dd], &cos_lsf_qa[1..order], dd);

    let mut a32_qa1 = [0i32; SILK_MAX_ORDER_LPC];
    for k in 0..dd {
        let ptmp = p[k + 1].wrapping_add(p[k]);
        let qtmp = q[k + 1].wrapping_sub(q[k]);
        a32_qa1[k] = qtmp.wrapping_neg().wrapping_sub(ptmp);
        a32_qa1[order - k - 1] = qtmp.wrapping_sub(ptmp);
    }

    lpc_fit(a_q12, &mut a32_qa1[..order], 12, QA + 1);

    let mut iteration = 0;
    while iteration < MAX_LPC_STABILIZE_ITERATIONS {
        if lpc_inverse_pred_gain(a_q12) != 0 {
            return;
        }

        let chirp_q16 = (1 << 16) - (2 << iteration);
        bwexpander_32(&mut a32_qa1[..order], chirp_q16);
        for (dst, &value) in a_q12.iter_mut().zip(a32_qa1[..order].iter()) {
            *dst = rshift_round(value, SHIFT_QA1_TO_Q12) as i16;
        }

        iteration += 1;
    }
}

fn nlsf2a_find_poly(out: &mut [i32], clsf: &[i32], dd: usize) {
    assert!(dd > 0, "polynomial order must be positive");
    assert!(out.len() > dd);
    let required = if dd <= 1 { 1 } else { 2 * dd - 1 };
    assert!(clsf.len() >= required);

    out[0] = 1 << QA;
    out[1] = -clsf[0];

    for k in 1..dd {
        let ftmp = clsf[2 * k];
        out[k + 1] =
            (out[k - 1] << 1).wrapping_sub(rshift_round64(i64::from(ftmp) * i64::from(out[k]), QA));
        for n in (2..=k).rev() {
            let product = rshift_round64(i64::from(ftmp) * i64::from(out[n - 1]), QA);
            out[n] = out[n].wrapping_add(out[n - 2]).wrapping_sub(product);
        }
        out[1] = out[1].wrapping_sub(ftmp);
    }
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    match shift.cmp(&0) {
        Ordering::Equal => value,
        Ordering::Greater => {
            if shift == 1 {
                (value >> 1) + (value & 1)
            } else {
                ((value >> (shift - 1)) + 1) >> 1
            }
        }
        Ordering::Less => value.wrapping_shl((-shift) as u32),
    }
}

fn rshift_round64(value: i64, shift: i32) -> i32 {
    if shift <= 0 {
        (value << (-shift)) as i32
    } else if shift == 1 {
        ((value >> 1) + (value & 1)) as i32
    } else {
        (((value >> (shift - 1)) + 1) >> 1) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::SILK_MAX_ORDER_LPC;
    use super::nlsf2a;
    use crate::silk::nlsf_decode::nlsf_decode;
    use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;

    #[test]
    fn produces_stable_lpc_for_wideband_vector() {
        let order = usize::try_from(SILK_NLSF_CB_WB.order).unwrap();
        let mut nlsf_q15 = [0i16; SILK_MAX_ORDER_LPC];
        let mut indices = [0i8; SILK_MAX_ORDER_LPC + 1];
        indices[0] = 5; // choose a non-trivial stage-one entry
        for value in indices.iter_mut().take(order + 1).skip(1) {
            *value = 1;
        }

        nlsf_decode(
            &mut nlsf_q15[..order],
            &indices[..order + 1],
            &SILK_NLSF_CB_WB,
        );

        let mut a_q12 = [0i16; SILK_MAX_ORDER_LPC];
        nlsf2a(&mut a_q12[..order], &nlsf_q15[..order], 0);

        let gain = super::lpc_inverse_pred_gain(&a_q12[..order]);
        assert!(gain > 0, "LPC coefficients should be stable");
    }
}
