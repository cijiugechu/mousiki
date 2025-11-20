//! Port of the floating-point `silk_LPC_inverse_pred_gain_FLP` helper from the
//! reference SILK implementation.
//!
//! The routine validates LPC stability and computes the inverse prediction
//! gain, returning zero when the coefficient set would exceed the maximum
//! allowed prediction power gain.

use super::lpc_inv_pred_gain::SILK_MAX_ORDER_LPC;

// 1e4 from `silk/define.h`.
const MAX_PREDICTION_POWER_GAIN: f64 = 1.0e4;

/// Computes the inverse prediction gain of floating-point LPC coefficients.
///
/// Returns `0.0` when the coefficient set is unstable or exceeds the reference
/// gain constraint.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn lpc_inverse_pred_gain_flp(a: &[f32]) -> f32 {
    assert!(
        !a.is_empty(),
        "LPC order must be strictly positive for inverse prediction gain"
    );
    assert!(
        a.len() <= SILK_MAX_ORDER_LPC,
        "order {} exceeds {}",
        a.len(),
        SILK_MAX_ORDER_LPC
    );

    let order = a.len();
    let mut a_tmp = [0f32; SILK_MAX_ORDER_LPC];
    a_tmp[..order].copy_from_slice(a);

    let mut inv_gain = 1.0f64;

    for k in (1..order).rev() {
        let rc = -f64::from(a_tmp[k]);
        let rc_mult1 = 1.0 - rc * rc;
        inv_gain *= rc_mult1;
        if inv_gain * MAX_PREDICTION_POWER_GAIN < 1.0 {
            return 0.0;
        }

        let rc_mult2 = 1.0 / rc_mult1;
        for n in 0..((k + 1) >> 1) {
            let tmp1 = f64::from(a_tmp[n]);
            let tmp2 = f64::from(a_tmp[k - n - 1]);
            a_tmp[n] = ((tmp1 - tmp2 * rc) * rc_mult2) as f32;
            a_tmp[k - n - 1] = ((tmp2 - tmp1 * rc) * rc_mult2) as f32;
        }
    }

    let rc = -f64::from(a_tmp[0]);
    let rc_mult1 = 1.0 - rc * rc;
    inv_gain *= rc_mult1;
    if inv_gain * MAX_PREDICTION_POWER_GAIN < 1.0 {
        0.0
    } else {
        inv_gain as f32
    }
}

#[cfg(test)]
mod tests {
    use super::lpc_inverse_pred_gain_flp;
    use crate::silk::lpc_inv_pred_gain::lpc_inverse_pred_gain;
    use alloc::vec::Vec;

    #[test]
    fn computes_first_order_gain() {
        let gain = lpc_inverse_pred_gain_flp(&[0.5]);
        assert!((gain - 0.75).abs() < 1e-6);
    }

    #[test]
    fn rejects_high_power_predictor() {
        let gain = lpc_inverse_pred_gain_flp(&[0.99999]);
        assert_eq!(gain, 0.0);
    }

    #[test]
    fn tracks_fixed_point_reference() {
        let coeffs = [0.2, -0.1, 0.05];
        let flp_gain = lpc_inverse_pred_gain_flp(&coeffs);

        let q12: Vec<i16> = coeffs.iter().map(|c| (c * 4096.0).round() as i16).collect();
        let fixed_gain_q30 = lpc_inverse_pred_gain(&q12);
        let fixed_gain = fixed_gain_q30 as f32 / (1u64 << 30) as f32;

        assert!((flp_gain - fixed_gain).abs() < 2e-3);
    }
}
