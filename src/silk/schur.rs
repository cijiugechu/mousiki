use core::cmp::Ordering;

use crate::silk::lpc_inv_pred_gain::SILK_MAX_ORDER_LPC;

const ALMOST_ONE_Q15: i16 = ((99 * (1 << 15) + 50) / 100) as i16;

fn clz32(value: i32) -> i32 {
    if value == 0 {
        32
    } else {
        (value as u32).leading_zeros() as i32
    }
}

fn saturate_to_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let c16 = i64::from(c as i16);
    let product = i64::from(b) * c16;
    a.wrapping_add((product >> 16) as i32)
}

pub fn silk_schur(rc_q15: &mut [i16], c: &[i32], order: usize) -> i32 {
    assert!(order <= SILK_MAX_ORDER_LPC);
    assert!(rc_q15.len() >= order);
    assert!(c.len() > order);

    let mut c_mat = [[0i32; 2]; SILK_MAX_ORDER_LPC + 1];

    let lz = clz32(c[0]);
    match lz.cmp(&2) {
        Ordering::Less => {
            for (dst, &src) in c_mat.iter_mut().zip(c.iter()).take(order + 1) {
                let val = src >> 1;
                dst[0] = val;
                dst[1] = val;
            }
        }
        Ordering::Greater => {
            let shift = (lz - 2) as u32;
            for i in 0..=order {
                let val = c[i].wrapping_shl(shift);
                c_mat[i][0] = val;
                c_mat[i][1] = val;
            }
        }
        Ordering::Equal => {
            for (dst, &src) in c_mat.iter_mut().zip(c.iter()).take(order + 1) {
                dst[0] = src;
                dst[1] = src;
            }
        }
    }

    let mut k = 0usize;
    while k < order {
        if c_mat[k + 1][0].abs() >= c_mat[0][1] {
            rc_q15[k] = if c_mat[k + 1][0] > 0 {
                -ALMOST_ONE_Q15
            } else {
                ALMOST_ONE_Q15
            };
            k += 1;
            break;
        }

        let denom = (c_mat[0][1] >> 15).max(1);
        let mut rc_tmp_q15 = -c_mat[k + 1][0] / denom;
        rc_tmp_q15 = i32::from(saturate_to_i16(rc_tmp_q15));
        rc_q15[k] = rc_tmp_q15 as i16;

        for n in 0..(order - k) {
            let ctmp1 = c_mat[n + k + 1][0];
            let ctmp2 = c_mat[n][1];
            c_mat[n + k + 1][0] = smlawb(ctmp1, ctmp2 << 1, rc_tmp_q15);
            c_mat[n][1] = smlawb(ctmp2, ctmp1 << 1, rc_tmp_q15);
        }

        k += 1;
    }

    while k < order {
        rc_q15[k] = 0;
        k += 1;
    }

    c_mat[0][1].max(1)
}
