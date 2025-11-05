//! Delayed-decision quantiser for NLSF residuals.
//!
//! Port of `silk_NLSF_del_dec_quant` from the reference SILK implementation.
//! The routine quantises a residual vector using a small trellis whose states
//! represent candidate amplitude indices. It mirrors the fixed-point arithmetic
//! of the C version so that the resulting rate/distortion cost and selected
//! indices remain bit-exact with the original codec.

use core::convert::TryInto;

use super::MAX_LPC_ORDER;

const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
const NLSF_QUANT_MAX_AMPLITUDE_EXT: i32 = 10;
const NLSF_QUANT_LEVEL_ADJ_Q10: i32 = 102; // SILK_FIX_CONST(0.1, 10)
const NLSF_QUANT_DEL_DEC_STATES_LOG2: usize = 2;
const NLSF_QUANT_DEL_DEC_STATES: usize = 1 << NLSF_QUANT_DEL_DEC_STATES_LOG2;

/// Trellis-based quantiser for NLSF residuals.
///
/// * `indices` - Output buffer receiving the selected amplitude indices.
/// * `x_q10` - Target residual vector in Q10.
/// * `w_q5` - Per-coefficient weights in Q5.
/// * `pred_coef_q8` - Backward predictor coefficients from the codebook.
/// * `ec_ix` - Offsets into the entropy-rate tables.
/// * `ec_rates_q5` - Entropy-rate table in Q5, as provided by the codebook.
/// * `quant_step_size_q16` - Quantisation step size in Q16.
/// * `inv_quant_step_size_q6` - Inverse step size in Q6.
/// * `mu_q20` - Rate/distortion trade-off weight in Q20.
///
/// Returns the accumulated rate/distortion cost in Q25.
#[allow(clippy::too_many_arguments)]
pub fn nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
) -> i32 {
    let order = x_q10.len();
    assert_eq!(indices.len(), order, "index buffer must match order");
    assert_eq!(w_q5.len(), order, "weight buffer must match order");
    assert_eq!(
        pred_coef_q8.len(),
        order,
        "predictor buffer must match order"
    );
    assert_eq!(ec_ix.len(), order, "entropy index buffer must match order");
    assert!(!x_q10.is_empty(), "NLSF order must be strictly positive");
    assert!(
        order <= MAX_LPC_ORDER,
        "order must not exceed MAX_LPC_ORDER"
    );

    let mut out0_q10_table = [0i32; 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize];
    let mut out1_q10_table = [0i32; 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize];

    for (offset, i) in (-NLSF_QUANT_MAX_AMPLITUDE_EXT..NLSF_QUANT_MAX_AMPLITUDE_EXT).enumerate() {
        let mut out0_q10 = i << 10;
        let mut out1_q10 = (i + 1) << 10;
        if i > 0 {
            out0_q10 -= NLSF_QUANT_LEVEL_ADJ_Q10;
            out1_q10 -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if i == 0 {
            out1_q10 -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if i == -1 {
            out0_q10 += NLSF_QUANT_LEVEL_ADJ_Q10;
        } else {
            out0_q10 += NLSF_QUANT_LEVEL_ADJ_Q10;
            out1_q10 += NLSF_QUANT_LEVEL_ADJ_Q10;
        }

        out0_q10_table[offset] = rshift(smulbb(out0_q10, quant_step_size_q16), 16);
        out1_q10_table[offset] = rshift(smulbb(out1_q10, quant_step_size_q16), 16);
    }

    let mut ind = [[0i8; MAX_LPC_ORDER]; NLSF_QUANT_DEL_DEC_STATES];
    let mut ind_sort = [0i32; NLSF_QUANT_DEL_DEC_STATES];
    let mut prev_out_q10 = [0i16; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_q25 = [0i32; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_min_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_max_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];

    let mut n_states = 1usize;
    rd_q25[0] = 0;
    prev_out_q10[0] = 0;

    for i in (0..order).rev() {
        let ec_offset = ec_ix[i]
            .try_into()
            .expect("entropy-table offset must be non-negative");
        debug_assert!(
            ec_offset + (2 * NLSF_QUANT_MAX_AMPLITUDE as usize + 1) <= ec_rates_q5.len(),
            "entropy-rate table must provide enough entries"
        );
        let rates_q5 = &ec_rates_q5[ec_offset..];
        let in_q10 = i32::from(x_q10[i]);

        for j in 0..n_states {
            let pred_q10 = rshift(
                smulbb(i32::from(pred_coef_q8[i]), i32::from(prev_out_q10[j])),
                8,
            );
            let res_q10 = sub16(in_q10, pred_q10);
            let mut ind_tmp = rshift(smulbb(i32::from(inv_quant_step_size_q6), res_q10), 16);
            ind_tmp = limit(
                ind_tmp,
                -NLSF_QUANT_MAX_AMPLITUDE_EXT,
                NLSF_QUANT_MAX_AMPLITUDE_EXT - 1,
            );
            ind[j][i] = ind_tmp as i8;

            let table_index = (ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
            let out0_q10_i16 = (out0_q10_table[table_index] + pred_q10) as i16;
            let out1_q10_i16 = (out1_q10_table[table_index] + pred_q10) as i16;
            prev_out_q10[j] = out0_q10_i16;
            prev_out_q10[j + n_states] = out1_q10_i16;
            let out0_q10 = i32::from(out0_q10_i16);
            let out1_q10 = i32::from(out1_q10_i16);

            let (rate0_q5, rate1_q5) = compute_rates(ind_tmp, rates_q5);

            let rd_tmp_q25 = rd_q25[j];
            let diff0_q10 = sub16(in_q10, out0_q10);
            rd_q25[j] = smlabb(
                mla(rd_tmp_q25, smulbb(diff0_q10, diff0_q10), i32::from(w_q5[i])),
                mu_q20,
                rate0_q5,
            );

            let diff1_q10 = sub16(in_q10, out1_q10);
            rd_q25[j + n_states] = smlabb(
                mla(rd_tmp_q25, smulbb(diff1_q10, diff1_q10), i32::from(w_q5[i])),
                mu_q20,
                rate1_q5,
            );
        }

        if n_states <= NLSF_QUANT_DEL_DEC_STATES / 2 {
            for j in 0..n_states {
                ind[j + n_states][i] = ind[j][i].wrapping_add(1);
            }
            n_states <<= 1;
            for j in n_states..NLSF_QUANT_DEL_DEC_STATES {
                ind[j][i] = ind[j - n_states][i];
            }
        } else {
            for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                let upper_idx = j + NLSF_QUANT_DEL_DEC_STATES;
                if rd_q25[j] > rd_q25[upper_idx] {
                    rd_max_q25[j] = rd_q25[j];
                    rd_min_q25[j] = rd_q25[upper_idx];
                    rd_q25[j] = rd_min_q25[j];
                    rd_q25[upper_idx] = rd_max_q25[j];
                    prev_out_q10.swap(j, upper_idx);
                    ind_sort[j] = (j + NLSF_QUANT_DEL_DEC_STATES) as i32;
                } else {
                    rd_min_q25[j] = rd_q25[j];
                    rd_max_q25[j] = rd_q25[upper_idx];
                    ind_sort[j] = j as i32;
                }
            }

            loop {
                let mut min_max_q25 = i32::MAX;
                let mut max_min_q25 = i32::MIN;
                let mut ind_min_max = 0usize;
                let mut ind_max_min = 0usize;
                for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                    if rd_max_q25[j] < min_max_q25 {
                        min_max_q25 = rd_max_q25[j];
                        ind_min_max = j;
                    }
                    if rd_min_q25[j] > max_min_q25 {
                        max_min_q25 = rd_min_q25[j];
                        ind_max_min = j;
                    }
                }
                if min_max_q25 >= max_min_q25 {
                    break;
                }

                let swap_src = ind_min_max;
                let swap_dst = ind_max_min;
                ind_sort[swap_dst] = ind_sort[swap_src] ^ NLSF_QUANT_DEL_DEC_STATES as i32;
                rd_q25[swap_dst] = rd_q25[swap_src + NLSF_QUANT_DEL_DEC_STATES];
                prev_out_q10[swap_dst] = prev_out_q10[swap_src + NLSF_QUANT_DEL_DEC_STATES];
                rd_min_q25[swap_dst] = 0;
                rd_max_q25[swap_src] = i32::MAX;
                ind[swap_dst] = ind[swap_src];
            }

            for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                ind[j][i] += ((ind_sort[j] >> NLSF_QUANT_DEL_DEC_STATES_LOG2) & 1) as i8;
            }
        }
    }

    let mut best_state = 0usize;
    let mut best_cost = i32::MAX;
    for (j, &cost) in rd_q25.iter().enumerate() {
        if cost < best_cost {
            best_cost = cost;
            best_state = j;
        }
    }

    let state_mask = NLSF_QUANT_DEL_DEC_STATES - 1;
    let base_state = best_state & state_mask;
    indices[..order].copy_from_slice(&ind[base_state][..order]);
    indices[0] = indices[0].wrapping_add((best_state >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8);
    debug_assert!(indices.iter().all(|&idx| {
        idx >= -NLSF_QUANT_MAX_AMPLITUDE_EXT as i8 && idx <= NLSF_QUANT_MAX_AMPLITUDE_EXT as i8
    }));

    debug_assert!(best_cost >= 0);
    best_cost
}

fn compute_rates(ind_tmp: i32, rates_q5: &[u8]) -> (i32, i32) {
    if ind_tmp + 1 >= NLSF_QUANT_MAX_AMPLITUDE {
        if ind_tmp + 1 == NLSF_QUANT_MAX_AMPLITUDE {
            let idx = (ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize;
            (i32::from(rates_q5[idx]), 280)
        } else {
            let base = 280 - 43 * NLSF_QUANT_MAX_AMPLITUDE;
            let rate0 = base + 43 * ind_tmp;
            (rate0, rate0 + 43)
        }
    } else if ind_tmp <= -NLSF_QUANT_MAX_AMPLITUDE {
        if ind_tmp == -NLSF_QUANT_MAX_AMPLITUDE {
            let idx = (ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize;
            (280, i32::from(rates_q5[idx]))
        } else {
            let base = 280 - 43 * NLSF_QUANT_MAX_AMPLITUDE;
            let rate0 = base - 43 * ind_tmp;
            (rate0, rate0 - 43)
        }
    } else {
        let idx0 = (ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize;
        let idx1 = idx0 + 1;
        (i32::from(rates_q5[idx0]), i32::from(rates_q5[idx1]))
    }
}

#[inline]
fn smulbb(a: i32, b: i32) -> i32 {
    let a16 = i32::from(a as i16);
    let b16 = i32::from(b as i16);
    a16 * b16
}

#[inline]
fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulbb(b, c))
}

#[inline]
fn mla(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(b.wrapping_mul(c))
}

#[inline]
fn rshift(value: i32, shift: i32) -> i32 {
    value >> shift
}

#[inline]
fn sub16(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline]
fn limit(value: i32, min: i32, max: i32) -> i32 {
    value.clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_input_returns_zero_cost() {
        let mut indices = [42i8; 4];
        let x_q10 = [0i16; 4];
        let w_q5 = [1i16; 4];
        let pred = [0u8; 4];
        let ec_ix = [0i16; 4];
        let mut ec_rates = [0u8; 16];
        for (idx, rate) in ec_rates.iter_mut().enumerate() {
            *rate = (idx as u8).wrapping_mul(5);
        }

        let cost = nlsf_del_dec_quant(
            &mut indices,
            &x_q10,
            &w_q5,
            &pred,
            &ec_ix,
            &ec_rates,
            1 << 16,
            1 << 6,
            0,
        );

        assert_eq!(cost, 0);
        assert!(indices.iter().all(|&idx| idx == 0));
    }
}
