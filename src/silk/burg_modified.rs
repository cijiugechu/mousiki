//! Port of `silk/fixed/burg_modified_FIX.c`.
//!
//! Computes the LPC predictor coefficients via the Burg method while tracking
//! the same Q-domain shifts, headroom clamps, and inverse-gain limits as the
//! fixed-point reference implementation.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]

use crate::silk::MAX_LPC_ORDER;
use crate::silk::inner_prod_aligned::inner_prod_aligned;
use crate::silk::stereo_find_predictor::div32_varq;
use crate::silk::vector_ops::inner_prod16;

const MAX_FRAME_SIZE: usize = 384;
const QA: i32 = 25;
const N_BITS_HEAD_ROOM: i32 = 3;
const MIN_RSHIFTS: i32 = -16;
const MAX_RSHIFTS: i32 = 32 - QA;
const FIND_LPC_COND_FAC_Q32: i32 = 42_950;

/// Fixed-point Burg method mirroring `silk_burg_modified_c`.
pub fn silk_burg_modified(
    res_nrg: &mut i32,
    res_nrg_q: &mut i32,
    a_q16: &mut [i32],
    x: &[i16],
    min_inv_gain_q30: i32,
    subfr_length: usize,
    nb_subfr: usize,
    order: usize,
    arch: i32,
) {
    assert!(
        order <= MAX_LPC_ORDER,
        "predictor order exceeds MAX_LPC_ORDER"
    );
    assert!(
        a_q16.len() >= order,
        "output buffer too small for LPC order"
    );
    assert!(
        subfr_length >= order,
        "subframe length must cover the predictor order"
    );

    let total_length = subfr_length
        .checked_mul(nb_subfr)
        .expect("subframe length overflow");
    assert!(
        total_length <= x.len(),
        "input buffer shorter than requested frame"
    );
    assert!(
        total_length <= MAX_FRAME_SIZE,
        "frame longer than MAX_FRAME_SIZE"
    );

    if order == 0 {
        *res_nrg = 0;
        *res_nrg_q = 0;
        return;
    }

    let mut c_first_row = [0i32; MAX_LPC_ORDER];
    let mut c_last_row = [0i32; MAX_LPC_ORDER];
    let mut af_qa = [0i32; MAX_LPC_ORDER];
    let mut caf = [0i32; MAX_LPC_ORDER + 1];
    let mut cab = [0i32; MAX_LPC_ORDER + 1];
    let mut xcorr = [0i32; MAX_LPC_ORDER];

    let c0_64 = inner_prod16(&x[..total_length], &x[..total_length]);
    if c0_64 == 0 {
        // Skip the Burg loop entirely when the frame has no energy to avoid
        // introducing the +1 bias from the reference's noise floor term.
        a_q16[..order].fill(0);
        *res_nrg = 0;
        *res_nrg_q = 0;
        return;
    }
    let lz = clz64(c0_64);
    let mut rshifts = 32 + 1 + N_BITS_HEAD_ROOM - lz;
    rshifts = rshifts.clamp(MIN_RSHIFTS, MAX_RSHIFTS);

    let mut c0 = if rshifts > 0 {
        (c0_64 >> rshifts) as i32
    } else {
        (c0_64 as i32).wrapping_shl((-rshifts) as u32)
    };

    let base = c0
        .wrapping_add(smmul(FIND_LPC_COND_FAC_Q32, c0))
        .wrapping_add(1);
    caf[0] = base;
    cab[0] = base;

    if rshifts > 0 {
        for s in 0..nb_subfr {
            let frame = &x[s * subfr_length..(s + 1) * subfr_length];
            for n in 1..=order {
                let len = subfr_length - n;
                let prod = inner_prod16(&frame[..len], &frame[n..n + len]);
                c_first_row[n - 1] = c_first_row[n - 1].wrapping_add((prod >> rshifts) as i32);
            }
        }
    } else {
        for s in 0..nb_subfr {
            let frame = &x[s * subfr_length..(s + 1) * subfr_length];
            let len = subfr_length - order;
            if len > 0 {
                pitch_xcorr(frame, &frame[1..], len, order, &mut xcorr[..order]);
            } else {
                xcorr[..order].fill(0);
            }
            for n in 1..=order {
                let start = n + subfr_length - order;
                let mut tail = 0i32;
                for i in start..subfr_length {
                    let sample = i32::from(frame[i]);
                    let delayed = i32::from(frame[i - n]);
                    tail = tail.wrapping_add(sample.wrapping_mul(delayed));
                }
                xcorr[n - 1] = xcorr[n - 1].wrapping_add(tail);
            }
            let shift = (-rshifts) as u32;
            for n in 1..=order {
                c_first_row[n - 1] =
                    c_first_row[n - 1].wrapping_add(xcorr[n - 1].wrapping_shl(shift));
            }
        }
    }

    c_last_row[..order].copy_from_slice(&c_first_row[..order]);
    caf[0] = base;
    cab[0] = base;

    let mut inv_gain_q30 = 1 << 30;
    let mut reached_max_gain = false;

    for n in 0..order {
        if rshifts > -2 {
            for s in 0..nb_subfr {
                let frame = &x[s * subfr_length..(s + 1) * subfr_length];
                let x1 = -((i32::from(frame[n])) << (16 - rshifts));
                let x2 = -((i32::from(frame[subfr_length - n - 1])) << (16 - rshifts));
                let mut tmp1 = i32::from(frame[n]) << (QA - 16);
                let mut tmp2 = i32::from(frame[subfr_length - n - 1]) << (QA - 16);
                for k in 0..n {
                    c_first_row[k] = smlawb(c_first_row[k], x1, i32::from(frame[n - k - 1]));
                    c_last_row[k] =
                        smlawb(c_last_row[k], x2, i32::from(frame[subfr_length - n + k]));
                    let atmp = af_qa[k];
                    tmp1 = smlawb(tmp1, atmp, i32::from(frame[n - k - 1]));
                    tmp2 = smlawb(tmp2, atmp, i32::from(frame[subfr_length - n + k]));
                }
                let shift = (32 - QA - rshifts) as u32;
                tmp1 = (-tmp1).wrapping_shl(shift);
                tmp2 = (-tmp2).wrapping_shl(shift);
                for k in 0..=n {
                    caf[k] = smlawb(caf[k], tmp1, i32::from(frame[n - k]));
                    cab[k] = smlawb(cab[k], tmp2, i32::from(frame[subfr_length - n + k - 1]));
                }
            }
        } else {
            for s in 0..nb_subfr {
                let frame = &x[s * subfr_length..(s + 1) * subfr_length];
                let x1 = -((i32::from(frame[n])) << (-rshifts));
                let x2 = -((i32::from(frame[subfr_length - n - 1])) << (-rshifts));
                let mut tmp1 = i32::from(frame[n]) << 17;
                let mut tmp2 = i32::from(frame[subfr_length - n - 1]) << 17;
                for k in 0..n {
                    c_first_row[k] = mla(c_first_row[k], x1, i32::from(frame[n - k - 1]));
                    c_last_row[k] = mla(c_last_row[k], x2, i32::from(frame[subfr_length - n + k]));
                    let atmp = rshift_round(af_qa[k], QA - 17);
                    tmp1 = mla_ovflw(tmp1, i32::from(frame[n - k - 1]), atmp);
                    tmp2 = mla_ovflw(tmp2, i32::from(frame[subfr_length - n + k]), atmp);
                }
                tmp1 = -tmp1;
                tmp2 = -tmp2;
                let shift = (-rshifts - 1) as u32;
                for k in 0..=n {
                    let head = i32::from(frame[n - k]).wrapping_shl(shift);
                    let tail = i32::from(frame[subfr_length - n + k - 1]).wrapping_shl(shift);
                    caf[k] = smla_w_w(caf[k], tmp1, head);
                    cab[k] = smla_w_w(cab[k], tmp2, tail);
                }
            }
        }

        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        let mut num = 0;
        let mut nrg = cab[0].wrapping_add(caf[0]);

        for k in 0..n {
            let atmp = af_qa[k];
            let mut shift = clz32(atmp.abs()) - 1;
            if shift > 32 - QA {
                shift = 32 - QA;
            }
            let atmp1 = atmp.wrapping_shl(shift as u32);
            let adjust = (32 - QA - shift) as u32;
            tmp1 = add_lshift(tmp1, smmul(c_last_row[n - k - 1], atmp1), adjust);
            tmp2 = add_lshift(tmp2, smmul(c_first_row[n - k - 1], atmp1), adjust);
            num = add_lshift(num, smmul(cab[n - k], atmp1), adjust);
            let sum = cab[k + 1].wrapping_add(caf[k + 1]);
            nrg = add_lshift(nrg, smmul(sum, atmp1), adjust);
        }

        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;
        num = num.wrapping_add(tmp2);
        num = num.wrapping_shl(1).wrapping_neg();

        let mut rc_q31 = if num.abs() < nrg {
            div32_varq(num, nrg, 31)
        } else if num > 0 {
            i32::MAX
        } else {
            i32::MIN
        };

        let mut tmp_gain = (1_i32 << 30).wrapping_sub(smmul(rc_q31, rc_q31));
        tmp_gain = lshift32(smmul(inv_gain_q30, tmp_gain), 2);
        if tmp_gain <= min_inv_gain_q30 {
            let limit = (1_i32 << 30).wrapping_sub(div32_varq(min_inv_gain_q30, inv_gain_q30, 30));
            rc_q31 = sqrt_approx(limit);
            if rc_q31 > 0 {
                rc_q31 = rshift_round(rc_q31 + div32(limit, rc_q31), 1);
                rc_q31 = rc_q31.wrapping_shl(16);
                if num < 0 {
                    rc_q31 = -rc_q31;
                }
            }
            inv_gain_q30 = min_inv_gain_q30;
            reached_max_gain = true;
        } else {
            inv_gain_q30 = tmp_gain;
        }

        let half = (n + 1) >> 1;
        for k in 0..half {
            let tmp_l = af_qa[k];
            let tmp_r = af_qa[n - k - 1];
            af_qa[k] = add_lshift(tmp_l, smmul(tmp_r, rc_q31), 1);
            af_qa[n - k - 1] = add_lshift(tmp_r, smmul(tmp_l, rc_q31), 1);
        }
        af_qa[n] = rc_q31 >> (31 - QA);

        if reached_max_gain {
            for slot in &mut af_qa[n + 1..order] {
                *slot = 0;
            }
            for k in 0..order {
                a_q16[k] = -rshift_round(af_qa[k], QA - 16);
            }
            if rshifts > 0 {
                for s in 0..nb_subfr {
                    let frame = &x[s * subfr_length..(s + 1) * subfr_length];
                    let prod = inner_prod16(&frame[..order], &frame[..order]);
                    c0 = c0.wrapping_sub((prod >> rshifts) as i32);
                }
            } else {
                for s in 0..nb_subfr {
                    let frame = &x[s * subfr_length..(s + 1) * subfr_length];
                    let prod = inner_prod_aligned(&frame[..order], &frame[..order], arch);
                    c0 = c0.wrapping_sub(prod.wrapping_shl((-rshifts) as u32));
                }
            }
            *res_nrg = lshift32(smmul(inv_gain_q30, c0), 2);
            *res_nrg_q = -rshifts;
            return;
        }

        for (k, caf_slot) in caf.iter_mut().take(n + 2).enumerate() {
            let tmp_l = *caf_slot;
            let idx = n + 1 - k;
            let tmp_r = cab[idx];
            *caf_slot = add_lshift(tmp_l, smmul(tmp_r, rc_q31), 1);
            cab[idx] = add_lshift(tmp_r, smmul(tmp_l, rc_q31), 1);
        }
    }

    let mut nrg = caf[0];
    let mut tmp1 = 1 << 16;
    for k in 0..order {
        let atmp = rshift_round(af_qa[k], QA - 16);
        nrg = smla_w_w(nrg, caf[k + 1], atmp);
        tmp1 = smla_w_w(tmp1, atmp, atmp);
        a_q16[k] = -atmp;
    }
    let cond = smmul(FIND_LPC_COND_FAC_Q32, c0);
    *res_nrg = smla_w_w(nrg, cond, -tmp1);
    *res_nrg_q = -rshifts;
}

fn pitch_xcorr(x: &[i16], y: &[i16], len: usize, max_pitch: usize, out: &mut [i32]) {
    assert!(out.len() >= max_pitch);
    if len == 0 {
        out[..max_pitch].fill(0);
        return;
    }
    for lag in 0..max_pitch {
        debug_assert!(y.len() >= lag + len);
        let mut sum = 0i64;
        for i in 0..len {
            sum += i64::from(x[i]) * i64::from(y[lag + i]);
        }
        out[lag] = sum as i32;
    }
}

fn clz32(value: i32) -> i32 {
    if value == 0 {
        32
    } else {
        value.leading_zeros() as i32
    }
}

fn clz64(value: i64) -> i32 {
    if value == 0 {
        64
    } else {
        value.leading_zeros() as i32
    }
}

fn smlawb(acc: i32, b: i32, c: i32) -> i32 {
    let product = (i64::from(b) * i64::from(c as i16)) >> 16;
    acc.wrapping_add(product as i32)
}

fn smla_w_w(acc: i32, b: i32, c: i32) -> i32 {
    let product = (i64::from(b) * i64::from(c)) >> 16;
    acc.wrapping_add(product as i32)
}

fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

fn mla(acc: i32, b: i32, c: i32) -> i32 {
    acc.wrapping_add(b.wrapping_mul(c))
}

fn mla_ovflw(acc: i32, b: i32, c: i32) -> i32 {
    acc.wrapping_add(b.wrapping_mul(c))
}

fn add_lshift(value: i32, other: i32, shift: u32) -> i32 {
    value.wrapping_add(other.wrapping_shl(shift))
}

fn lshift32(value: i32, shift: i32) -> i32 {
    value.wrapping_shl(shift as u32)
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn div32(a: i32, b: i32) -> i32 {
    debug_assert!(b != 0, "division by zero");
    a / b
}

fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 == 1 { 32_768 } else { 46_214 };
    y >>= lz >> 1;
    y = smlawb(y, y, smulbb(213, frac_q7));
    y
}

fn clz_frac(x: i32) -> (i32, i32) {
    let lz = clz32(x);
    let frac = ((x << (lz + 1)) >> 25) & 0x7f;
    (lz, frac)
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

#[cfg(test)]
mod tests {
    use super::silk_burg_modified;

    #[test]
    fn zero_input_yields_zero_lpc() {
        let mut res_nrg = 0;
        let mut res_nrg_q = 0;
        let mut coeffs = [0i32; 16];
        let input = [0i16; 384];
        silk_burg_modified(
            &mut res_nrg,
            &mut res_nrg_q,
            &mut coeffs,
            &input,
            1 << 16,
            20,
            4,
            16,
            0,
        );
        assert!(coeffs.iter().all(|&c| c == 0));
        assert_eq!(res_nrg, 0);
    }
}
