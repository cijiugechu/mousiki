//! Floating-point Schur recursion from `silk/float/schur_FLP.c`.
//!
//! The helper derives reflection coefficients and residual energy from an
//! autocorrelation sequence using double-precision accumulators to match the
//! C reference behaviour.

use crate::silk::lpc_inv_pred_gain::SILK_MAX_ORDER_LPC;

/// Computes reflection coefficients and residual energy for an LPC
/// autocorrelation sequence.
///
/// Mirrors `silk_schur_FLP` by clamping the energy denominator away from zero
/// and updating the correlation matrix in place.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn silk_schur_flp(refl_coef: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    assert!(
        order <= SILK_MAX_ORDER_LPC,
        "order {} exceeds SILK_MAX_ORDER_LPC ({})",
        order,
        SILK_MAX_ORDER_LPC
    );
    assert!(
        refl_coef.len() >= order,
        "reflection coefficient slice too short for order {}",
        order
    );
    assert!(
        auto_corr.len() > order,
        "auto correlation slice must contain at least order + 1 entries"
    );

    let mut c = [[0f64; 2]; SILK_MAX_ORDER_LPC + 1];
    for (dst, &src) in c.iter_mut().zip(auto_corr.iter()).take(order + 1) {
        let value = f64::from(src);
        dst[0] = value;
        dst[1] = value;
    }

    for k in 0..order {
        let rc_tmp = -c[k + 1][0] / c[0][1].max(1e-9f64);
        refl_coef[k] = rc_tmp as f32;

        for n in 0..(order - k) {
            let ctmp1 = c[n + k + 1][0];
            let ctmp2 = c[n][1];
            c[n + k + 1][0] = ctmp1 + ctmp2 * rc_tmp;
            c[n][1] = ctmp2 + ctmp1 * rc_tmp;
        }
    }

    c[0][1] as f32
}

#[cfg(test)]
mod tests {
    use super::silk_schur_flp;

    #[test]
    fn returns_autocorr_for_zero_order() {
        let auto_corr = [0.75f32];
        let mut refl: [f32; 0] = [];

        let residual = silk_schur_flp(&mut refl, &auto_corr, 0);

        assert!((residual - 0.75).abs() < 1e-12);
    }

    #[test]
    fn computes_first_order_reflection() {
        let auto_corr = [10.0f32, 3.0];
        let mut refl = [0.0f32; 1];

        let residual = silk_schur_flp(&mut refl, &auto_corr, 1);

        assert!((refl[0] + 0.3).abs() < 1e-6);
        assert!((residual - 9.1).abs() < 1e-6);
    }

    #[test]
    fn computes_second_order_example() {
        let auto_corr = [10.0f32, 4.0, 1.0];
        let mut refl = [0.0f32; 2];

        let residual = silk_schur_flp(&mut refl, &auto_corr, 2);

        assert!((refl[0] + 0.4).abs() < 1e-6);
        assert!((refl[1] - 0.071_428_575).abs() < 1e-6);
        assert!((residual - 8.357_142_45).abs() < 1e-5);
    }
}
