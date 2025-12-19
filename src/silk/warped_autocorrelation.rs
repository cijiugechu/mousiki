//! Ports the fixed-point warped autocorrelation helper from
//! `silk/fixed/warped_autocorrelation_FIX.c`.
//!
//! The routine computes autocorrelation values on a warped frequency axis and
//! returns a scaling factor so the results fit into 32-bit accumulators. It is
//! used by the SILK encoder's noise-shaping analysis when deriving warped LPC
//! coefficients.

/// Maximum LPC order supported by the warped autocorrelation helper.
pub const MAX_SHAPE_LPC_ORDER: usize = 24;

const QC: i32 = 10;
const QS: i32 = 13;
const CORR_SHIFT_QC: i32 = 2 * QS - QC; // equals 16 in the reference code

/// Computes warped autocorrelations for a Q0 input vector.
///
/// The `order` must be even and no larger than [`MAX_SHAPE_LPC_ORDER`]. The
/// `corr` slice must provide room for `order + 1` values. The function returns
/// the scaling factor that the C version writes to `scale`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub fn warped_autocorrelation(
    corr: &mut [i32],
    input: &[i16],
    warping_q16: i32,
    order: usize,
) -> i32 {
    assert!(
        order <= MAX_SHAPE_LPC_ORDER,
        "order must be <= {MAX_SHAPE_LPC_ORDER}"
    );
    assert!(order.is_multiple_of(2), "order must be even");
    assert!(
        corr.len() > order,
        "corr must expose at least {}",
        order + 1
    );

    let mut state_qs = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];

    for &sample in input {
        let mut tmp1_qs = (i32::from(sample)) << QS;

        let mut section = 0;
        while section < order {
            let next = section + 1;
            let tail = section + 2;

            let diff = state_qs[next].wrapping_sub(tmp1_qs);
            let tmp2_qs = smlawb(state_qs[section], diff, warping_q16);
            state_qs[section] = tmp1_qs;
            corr_qc[section] += mul_qc(tmp1_qs, state_qs[0]);

            let diff2 = state_qs[tail].wrapping_sub(tmp2_qs);
            tmp1_qs = smlawb(state_qs[next], diff2, warping_q16);
            state_qs[next] = tmp2_qs;
            corr_qc[next] += mul_qc(tmp2_qs, state_qs[0]);

            section += 2;
        }

        state_qs[order] = tmp1_qs;
        corr_qc[order] += mul_qc(tmp1_qs, state_qs[0]);
    }

    debug_assert!(
        corr_qc[0] >= 0,
        "corr_qc[0] should stay non-negative after accumulation"
    );

    let mut lsh = clz64(corr_qc[0]) - 35;
    lsh = lsh.clamp(-12 - QC, 30 - QC);

    let scale = -(QC + lsh);
    debug_assert!((-30..=12).contains(&scale));

    if lsh >= 0 {
        let shift = lsh as u32;
        for (dst, src) in corr.iter_mut().take(order + 1).zip(&corr_qc) {
            let value = src.wrapping_shl(shift);
            debug_assert!(
                value <= i64::from(i32::MAX) && value >= i64::from(i32::MIN),
                "scaled correlation overflows 32-bit range"
            );
            *dst = value as i32;
        }
    } else {
        let shift = (-lsh) as u32;
        for (dst, src) in corr.iter_mut().take(order + 1).zip(&corr_qc) {
            let value = src >> shift;
            debug_assert!(
                value <= i64::from(i32::MAX) && value >= i64::from(i32::MIN),
                "scaled correlation overflows 32-bit range"
            );
            *dst = value as i32;
        }
    }

    debug_assert!(
        corr_qc[0] >= 0,
        "corr_qc[0] should stay non-negative after scaling"
    );

    scale
}

fn clz64(value: i64) -> i32 {
    if value == 0 {
        64
    } else {
        (value as u64).leading_zeros() as i32
    }
}

#[inline]
fn smlawb(acc: i32, b: i32, c: i32) -> i32 {
    let c16 = i32::from(c as i16);
    let product = (i64::from(b) * i64::from(c16)) >> 16;
    acc.wrapping_add(product as i32)
}

#[inline]
fn mul_qc(a: i32, b: i32) -> i64 {
    (i64::from(a) * i64::from(b)) >> CORR_SHIFT_QC
}

#[cfg(test)]
mod tests {
    use super::{MAX_SHAPE_LPC_ORDER, warped_autocorrelation};

    #[test]
    fn matches_reference_values() {
        let input = [3276, -1638, 819, -410, 205, -102, 51, -25];
        let mut corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
        let scale = warped_autocorrelation(&mut corr, &input, 29_491, 4);

        assert_eq!(scale, -5);
        assert_eq!(
            &corr[..5],
            &[
                457_911_552,
                -355_118_103,
                275_370_536,
                -213_637_770,
                165_476_668
            ]
        );
    }

    #[test]
    fn matches_reference_values_higher_order() {
        let input = [
            1234, 2345, -3456, 4567, -5678, 6789, -7890, 3210, -210, 1111, -999, 555,
        ];
        let mut corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
        let scale = warped_autocorrelation(&mut corr, &input, 16_384, 6);

        assert_eq!(scale, -1);
        assert_eq!(
            &corr[..7],
            &[
                386_588_116,
                -358_683_256,
                315_108_010,
                -274_226_201,
                232_894_203,
                -201_623_736,
                166_257_710
            ]
        );
    }

    #[test]
    fn zero_input_produces_min_scale() {
        let input = [0i16; 8];
        let mut corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
        let scale = warped_autocorrelation(&mut corr, &input, 1_000, 4);

        assert_eq!(scale, -30);
        assert!(corr.iter().take(5).all(|&v| v == 0));
    }
}
