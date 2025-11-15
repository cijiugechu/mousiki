//! Port of `silk/fixed/residual_energy_FIX.c`.
//!
//! The helper filters each half-frame with the Q12 LPC prediction taps,
//! measures the residual power of every subframe, and applies the squared
//! quantiser gains so later encoder stages can operate on normalised energies.

use crate::silk::decoder_set_fs::MAX_SUB_FRAME_LENGTH;
use crate::silk::lpc_analysis_filter::lpc_analysis_filter;
use crate::silk::sum_sqr_shift::sum_sqr_shift;
use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR};

const HALF_MAX_SUBFR: usize = MAX_NB_SUBFR / 2;
const MAX_RESIDUAL: usize = HALF_MAX_SUBFR * (MAX_LPC_ORDER + MAX_SUB_FRAME_LENGTH);

/// Mirrors `silk_residual_energy_FIX`.
#[allow(clippy::too_many_arguments)]
pub fn residual_energy(
    nrgs: &mut [i32; MAX_NB_SUBFR],
    nrgs_q: &mut [i32; MAX_NB_SUBFR],
    input: &[i16],
    coeffs_q12: &[[i16; MAX_LPC_ORDER]; HALF_MAX_SUBFR],
    gains_q16: &[i32; MAX_NB_SUBFR],
    subfr_length: usize,
    nb_subfr: usize,
    lpc_order: usize,
    arch: i32,
) {
    let _ = arch;
    assert!(
        nb_subfr == MAX_NB_SUBFR || nb_subfr == HALF_MAX_SUBFR,
        "nb_subfr must be {} or {}",
        MAX_NB_SUBFR,
        HALF_MAX_SUBFR
    );
    assert!(
        subfr_length > 0 && subfr_length <= MAX_SUB_FRAME_LENGTH,
        "invalid subframe length {subfr_length}"
    );
    assert!(
        (6..=MAX_LPC_ORDER).contains(&lpc_order) && lpc_order.is_multiple_of(2),
        "LPC order must be an even number within 6..={MAX_LPC_ORDER}"
    );

    let offset = lpc_order + subfr_length;
    assert!(
        offset <= MAX_LPC_ORDER + MAX_SUB_FRAME_LENGTH,
        "offset exceeds stack buffer"
    );
    let block_len = HALF_MAX_SUBFR * offset;
    let total_needed = (nb_subfr >> 1) * block_len;
    assert!(
        input.len() >= total_needed,
        "input slice too short: need {total_needed} samples"
    );

    let mut lpc_res = [0i16; MAX_RESIDUAL];
    let mut input_idx = 0;
    for (half, coeffs) in coeffs_q12.iter().enumerate().take(nb_subfr >> 1) {
        let res_slice = &mut lpc_res[..block_len];
        let window = &input[input_idx..input_idx + block_len];
        lpc_analysis_filter(
            res_slice,
            window,
            &coeffs[..lpc_order],
            block_len,
            lpc_order,
        );

        let mut res_ptr = lpc_order;
        for j in 0..HALF_MAX_SUBFR {
            let idx = half * HALF_MAX_SUBFR + j;
            let start = res_ptr;
            let end = start + subfr_length;
            let (energy, shift) = sum_sqr_shift(&res_slice[start..end]);
            nrgs[idx] = energy;
            nrgs_q[idx] = -shift;
            res_ptr += offset;
        }

        input_idx += block_len;
    }

    for i in 0..nb_subfr {
        let lz1 = clz32(nrgs[i]) - 1;
        let lz2 = clz32(gains_q16[i]) - 1;
        let gain_shifted = lshift32(gains_q16[i], lz2);
        let gain_squared = smmul(gain_shifted, gain_shifted);
        nrgs[i] = smmul(gain_squared, lshift32(nrgs[i], lz1));
        nrgs_q[i] += lz1 + (lz2 * 2) - 64;
    }
}

fn clz32(value: i32) -> i32 {
    if value == 0 {
        32
    } else {
        (value as u32).leading_zeros() as i32
    }
}

fn lshift32(value: i32, shift: i32) -> i32 {
    debug_assert!(shift >= 0);
    value.wrapping_shl(shift as u32)
}

fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

#[cfg(test)]
mod tests {
    use super::residual_energy;
    use crate::silk::{MAX_LPC_ORDER, MAX_NB_SUBFR};
    use alloc::vec::Vec;

    #[test]
    fn computes_residuals_for_four_subframes() {
        const LPC_ORDER: usize = 10;
        const SUBFR: usize = 40;
        const NB_SUBFR: usize = MAX_NB_SUBFR;

        let mut nrgs = [0; MAX_NB_SUBFR];
        let mut nrgs_q = [0; MAX_NB_SUBFR];
        let input = test_input(NB_SUBFR, LPC_ORDER, SUBFR, -16_000, 1_103_515_245, 12_345);
        let coeffs = test_coeffs(1234, 1);
        let gains = [65_536, 98_304, 147_456, 229_376];

        residual_energy(
            &mut nrgs,
            &mut nrgs_q,
            &input,
            &coeffs,
            &gains,
            SUBFR,
            NB_SUBFR,
            LPC_ORDER,
            0,
        );

        assert_eq!(nrgs[..NB_SUBFR], [123_386_601, 283_447_072, 85_303_613, 347_417_940]);
        assert_eq!(nrgs_q[..NB_SUBFR], [-40, -40, -43, -42]);
    }

    #[test]
    fn handles_two_subframe_mode() {
        const LPC_ORDER: usize = 12;
        const SUBFR: usize = 20;
        const NB_SUBFR: usize = MAX_NB_SUBFR / 2;

        let mut nrgs = [0; MAX_NB_SUBFR];
        let mut nrgs_q = [0; MAX_NB_SUBFR];
        let input = test_input(NB_SUBFR, LPC_ORDER, SUBFR, 123_456_789, 1_664_525, 1_013_904_223);
        let coeffs = test_coeffs(321, 2);
        let gains = [65_536, 131_072, 98_304, 81_920];

        residual_energy(
            &mut nrgs,
            &mut nrgs_q,
            &input,
            &coeffs,
            &gains,
            SUBFR,
            NB_SUBFR,
            LPC_ORDER,
            0,
        );

        assert_eq!(nrgs[..NB_SUBFR], [67_666_475, 128_629_840]);
        assert_eq!(nrgs_q[..NB_SUBFR], [-40, -41]);
    }

    fn test_input(
        nb_subfr: usize,
        lpc_order: usize,
        subfr_length: usize,
        mut seed: i64,
        mult: i64,
        incr: i64,
    ) -> Vec<i16> {
        let total = nb_subfr * (lpc_order + subfr_length);
        let mut out = Vec::with_capacity(total);
        for _ in 0..total {
            seed = (seed.wrapping_mul(mult).wrapping_add(incr)) & 0x7FFF_FFFF;
            let sample = ((seed >> 15) & 0xFFFF) as i64 - 32_768;
            out.push(sample as i16);
        }
        out
    }

    fn test_coeffs(scale: i32, stride: i32) -> [[i16; MAX_LPC_ORDER]; super::HALF_MAX_SUBFR] {
        let mut coeffs = [[0i16; MAX_LPC_ORDER]; super::HALF_MAX_SUBFR];
        for (row_idx, row) in coeffs.iter_mut().enumerate() {
            for (k, coeff) in row.iter_mut().enumerate() {
                let raw = ((row_idx as i32 + stride) * (k as i32 + 2) * scale) & 0x3FFF;
                *coeff = if raw & 0x2000 != 0 {
                    (raw as i32 - 0x4000) as i16
                } else {
                    raw as i16
                };
            }
        }
        coeffs
    }
}
