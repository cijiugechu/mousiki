//! Port of `silk/float/burg_modified_FLP.c`.
//!
//! Computes floating-point LPC coefficients with the Burg method while
//! preserving the reference gain conditioning and reflection-coefficient
//! clamping behaviour used by the SILK FLP analysis path.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]

use crate::silk::energy_flp::energy;
use crate::silk::inner_product_flp::inner_product_flp;
use crate::silk::lpc_inv_pred_gain::SILK_MAX_ORDER_LPC;
use crate::silk::tuning_parameters::FIND_LPC_COND_FAC;
use libm::sqrt;

const MAX_FRAME_SIZE: usize = 384;

/// Floating-point Burg method mirroring `silk_burg_modified_FLP`.
pub fn silk_burg_modified_flp(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    order: usize,
    arch: i32,
) -> f32 {
    assert!(
        order <= SILK_MAX_ORDER_LPC,
        "order {} exceeds SILK_MAX_ORDER_LPC ({})",
        order,
        SILK_MAX_ORDER_LPC
    );
    assert!(
        a.len() >= order,
        "output buffer too small for LPC order {}",
        order
    );
    assert!(
        subfr_length >= order,
        "subframe length must cover the LPC order"
    );

    let frame_length = subfr_length
        .checked_mul(nb_subfr)
        .expect("frame length overflow");
    assert!(
        frame_length <= x.len(),
        "input shorter than subframe stack length"
    );
    assert!(
        frame_length <= MAX_FRAME_SIZE,
        "frame exceeds MAX_FRAME_SIZE"
    );

    let _ = arch;
    let x = &x[..frame_length];

    let mut c_first_row = [0f64; SILK_MAX_ORDER_LPC];
    let mut c_last_row = [0f64; SILK_MAX_ORDER_LPC];
    let mut caf = [0f64; SILK_MAX_ORDER_LPC + 1];
    let mut cab = [0f64; SILK_MAX_ORDER_LPC + 1];
    let mut af = [0f64; SILK_MAX_ORDER_LPC];

    let c0 = energy(x);
    for s in 0..nb_subfr {
        let start = s * subfr_length;
        let x_ptr = &x[start..start + subfr_length];
        for n in 1..=order {
            let len = subfr_length - n;
            let dot = inner_product_flp(&x_ptr[..len], &x_ptr[n..n + len]);
            c_first_row[n - 1] += dot;
        }
    }
    c_last_row[..order].copy_from_slice(&c_first_row[..order]);

    let cond = f64::from(FIND_LPC_COND_FAC) * c0 + 1e-9f64;
    let base = c0 + cond;
    caf[0] = base;
    cab[0] = base;

    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;

    for n in 0..order {
        for s in 0..nb_subfr {
            let start = s * subfr_length;
            let x_ptr = &x[start..start + subfr_length];

            let mut tmp1 = f64::from(x_ptr[n]);
            let mut tmp2 = f64::from(x_ptr[subfr_length - n - 1]);
            for k in 0..n {
                c_first_row[k] -= f64::from(x_ptr[n]) * f64::from(x_ptr[n - k - 1]);
                c_last_row[k] -=
                    f64::from(x_ptr[subfr_length - n - 1]) * f64::from(x_ptr[subfr_length - n + k]);

                let atmp = af[k];
                tmp1 += f64::from(x_ptr[n - k - 1]) * atmp;
                tmp2 += f64::from(x_ptr[subfr_length - n + k]) * atmp;
            }

            for k in 0..=n {
                caf[k] -= tmp1 * f64::from(x_ptr[n - k]);
                cab[k] -= tmp2 * f64::from(x_ptr[subfr_length - n + k - 1]);
            }
        }

        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }

        assert!(nrg_f > 0.0, "forward prediction energy must stay positive");
        assert!(nrg_b > 0.0, "backward prediction energy must stay positive");

        let mut rc = -2.0 * num / (nrg_f + nrg_b);
        assert!(
            rc > -1.0 && rc < 1.0,
            "reflection coefficient out of bounds"
        );

        let next_inv_gain = inv_gain * (1.0 - rc * rc);
        if next_inv_gain <= f64::from(min_inv_gain) {
            rc = sqrt(1.0 - f64::from(min_inv_gain) / inv_gain);
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = f64::from(min_inv_gain);
            reached_max_gain = true;
        } else {
            inv_gain = next_inv_gain;
        }

        let half = n.div_ceil(2);
        for k in 0..half {
            let tmp_l = af[k];
            let tmp_r = af[n - k - 1];
            af[k] = tmp_l + rc * tmp_r;
            af[n - k - 1] = tmp_r + rc * tmp_l;
        }
        af[n] = rc;

        if reached_max_gain {
            for coef in &mut af[n + 1..order] {
                *coef = 0.0;
            }
            break;
        }

        for (k, caf_slot) in caf.iter_mut().take(n + 2).enumerate() {
            let idx = n + 1 - k;
            let tmp_l = *caf_slot;
            let tmp_r = cab[idx];
            *caf_slot = tmp_l + rc * tmp_r;
            cab[idx] = tmp_r + rc * tmp_l;
        }
    }

    let residual = if reached_max_gain {
        for (dst, &coef) in a.iter_mut().take(order).zip(af.iter()) {
            *dst = -(coef as f32);
        }

        let mut c0_adjusted = c0;
        for s in 0..nb_subfr {
            let start = s * subfr_length;
            c0_adjusted -= energy(&x[start..start + order]);
        }
        c0_adjusted * inv_gain
    } else {
        let mut nrg_f = caf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..order {
            let atmp = af[k];
            nrg_f += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = -(atmp as f32);
        }

        nrg_f - f64::from(FIND_LPC_COND_FAC) * c0 * tmp1
    };

    residual as f32
}

#[cfg(test)]
mod tests {
    use super::silk_burg_modified_flp;

    #[test]
    fn computes_first_order_coefficients() {
        let mut coeffs = [0.0f32; 1];
        let x = [1.0f32, 0.5, 0.25];
        let residual = silk_burg_modified_flp(&mut coeffs, &x, 0.1, 3, 1, 1, 0);

        assert!((coeffs[0] - 0.7999866).abs() < 1e-6);
        assert!((residual - 0.11248992).abs() < 1e-8);
    }

    #[test]
    fn clamps_when_prediction_gain_exceeds_limit() {
        let mut coeffs = [0.0f32; 1];
        let x = [1.0f32, 0.5, 0.25];
        let residual = silk_burg_modified_flp(&mut coeffs, &x, 0.9, 3, 1, 1, 0);

        assert!((coeffs[0] - 0.31622776).abs() < 1e-6);
        assert!((residual - 0.28125).abs() < 1e-8);
    }
}
