//! Port of the SILK packet-loss concealment helpers from `silk/PLC.c`.
//!
//! The routines in this module reconstruct an excitation signal when packets
//! go missing, glue the resulting concealment frames to the next valid frame,
//! and keep the PLC scratch state in sync with the decoder. The code mirrors
//! the fixed-point arithmetic and guard rails from the C implementation so
//! that future ports of the decoder entry points can reuse the logic without
//! diverging from the reference behaviour.

use core::cmp::{max, min};

use crate::silk::bwexpander::bwexpander;
use crate::silk::cng::silk_rand;
use crate::silk::decoder_control::DecoderControl;
use crate::silk::decoder_set_fs::{MAX_FRAME_LENGTH, MAX_SUB_FRAME_LENGTH};
use crate::silk::decoder_state::DecoderState;
use crate::silk::lpc_analysis_filter::lpc_analysis_filter;
use crate::silk::lpc_inv_pred_gain::{inverse32_varq, lpc_inverse_pred_gain};
use crate::silk::sum_sqr_shift::sum_sqr_shift;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{FrameSignalType, MAX_LPC_ORDER};

const NB_ATTENUATION_STEPS: usize = 2;
const HARM_ATT_Q15: [i16; NB_ATTENUATION_STEPS] = [32_440, 31_130]; // 0.99, 0.95
const RAND_ATTENUATE_VOICED_Q15: [i16; NB_ATTENUATION_STEPS] = [31_130, 26_214]; // 0.95, 0.8
const RAND_ATTENUATE_UNVOICED_Q15: [i16; NB_ATTENUATION_STEPS] = [32_440, 29_491]; // 0.99, 0.9
const BWE_COEF_Q16: i32 = 64_881; // SILK_FIX_CONST(0.99, 16)
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11_469; // 0.7 in Q14
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15_565; // 0.95 in Q14
const MAX_PITCH_LAG_MS: i32 = 18;
const RAND_BUF_SIZE: usize = 128;
const RAND_BUF_MASK: usize = RAND_BUF_SIZE - 1;
const LOG2_INV_LPC_GAIN_HIGH_THRES: i32 = 3;
const LOG2_INV_LPC_GAIN_LOW_THRES: i32 = 8;
const PITCH_DRIFT_FAC_Q16: i32 = 655; // 0.01 in Q16
const MAX_LTP_MEM_LENGTH: usize = 4 * MAX_SUB_FRAME_LENGTH;
const MAX_LTP_STATE: usize = MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH;

/// Resets the PLC helper, mirroring `silk_PLC_Reset`.
pub fn silk_plc_reset(state: &mut DecoderState) {
    let frame_length = state.sample_rate.frame_length;
    state.plc_state.reset(frame_length);
}

/// Mirrors `silk_PLC`: updates or synthesises the PLC output for one frame.
pub fn silk_plc(
    state: &mut DecoderState,
    control: &mut DecoderControl,
    frame: &mut [i16],
    lost: bool,
    arch: i32,
) {
    assert_eq!(
        frame.len(),
        state.sample_rate.frame_length,
        "PLC frame length must match decoder state"
    );

    if state.sample_rate.fs_khz != state.plc_state.fs_khz {
        silk_plc_reset(state);
        state.plc_state.fs_khz = state.sample_rate.fs_khz;
    }

    if lost {
        silk_plc_conceal(state, control, frame, arch);
        state.loss_count = state.loss_count.saturating_add(1);
    } else {
        silk_plc_update(state, control);
    }
}

/// Mirrors `silk_PLC_glue_frames`: smooths the transition from concealed audio.
pub fn silk_plc_glue_frames(state: &mut DecoderState, frame: &mut [i16]) {
    let plc = &mut state.plc_state;
    if state.loss_count > 0 {
        let (energy, shift) = sum_sqr_shift(frame);
        plc.conc_energy = energy;
        plc.conc_energy_shift = shift;
        plc.last_frame_lost = 1;
        return;
    }

    if plc.last_frame_lost == 0 {
        return;
    }

    let (mut energy, energy_shift) = sum_sqr_shift(frame);
    if energy_shift > plc.conc_energy_shift {
        let shift = energy_shift - plc.conc_energy_shift;
        plc.conc_energy >>= shift;
    } else if energy_shift < plc.conc_energy_shift {
        energy >>= plc.conc_energy_shift - energy_shift;
    }

    if energy > plc.conc_energy {
        let leading_zeros = (plc.conc_energy.max(1) as u32).leading_zeros() as i32 - 1;
        plc.conc_energy <<= leading_zeros;
        let right_shift = max(24 - leading_zeros, 0);
        if right_shift > 0 {
            energy >>= right_shift;
        }
        let frac_q24 = div32(plc.conc_energy, max(energy, 1));
        let mut gain_q16 = lshift(sqrt_approx(frac_q24), 4);
        let mut slope_q16 = div32_16((1 << 16) - gain_q16, frame.len() as i32);
        slope_q16 <<= 2;

        for sample in frame.iter_mut() {
            *sample = sat16(smulwb(gain_q16, i32::from(*sample)));
            gain_q16 = gain_q16.saturating_add(slope_q16);
            if gain_q16 > (1 << 16) {
                break;
            }
        }
    }

    plc.last_frame_lost = 0;
}

fn silk_plc_update(state: &mut DecoderState, control: &DecoderControl) {
    let sr = &mut state.sample_rate;
    let plc = &mut state.plc_state;

    sr.prev_signal_type = state.indices.signal_type;

    let nb_subfr = sr.nb_subfr;
    let subfr_length = sr.subfr_length as i32;
    let lpc_order = sr.lpc_order;
    let fs_khz = sr.fs_khz;

    assert!(nb_subfr >= 2, "PLC update requires at least two subframes");
    assert!(lpc_order <= MAX_LPC_ORDER);

    let mut ltp_gain_q14 = 0;
    if matches!(state.indices.signal_type, FrameSignalType::Voiced) {
        let target_pitch = control.pitch_l[nb_subfr - 1];
        let mut j = 0usize;
        while j < nb_subfr && (j as i32) * subfr_length < target_pitch {
            let subframe = nb_subfr - 1 - j;
            let start = subframe * LTP_ORDER;
            let mut temp_gain = 0;
            for coef in &control.ltp_coef_q14[start..start + LTP_ORDER] {
                temp_gain += i32::from(*coef);
            }
            if temp_gain > ltp_gain_q14 {
                ltp_gain_q14 = temp_gain;
                plc.pitch_l_q8 = control.pitch_l[subframe] << 8;
            }
            j += 1;
        }

        plc.ltp_coef_q14.fill(0);
        plc.ltp_coef_q14[LTP_ORDER / 2] = clamp_to_i16(ltp_gain_q14);

        if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let tmp = lshift(V_PITCH_GAIN_START_MIN_Q14, 10);
            let scale_q10 = div32(tmp, max(ltp_gain_q14, 1));
            for coef in &mut plc.ltp_coef_q14 {
                let scaled = smulbb(i32::from(*coef), scale_q10);
                *coef = clamp_to_i16(rshift(scaled, 10));
            }
        } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let tmp = lshift(V_PITCH_GAIN_START_MAX_Q14, 14);
            let scale_q14 = div32(tmp, max(ltp_gain_q14, 1));
            for coef in &mut plc.ltp_coef_q14 {
                let scaled = smulbb(i32::from(*coef), scale_q14);
                *coef = clamp_to_i16(rshift(scaled, 14));
            }
        }
    } else {
        plc.pitch_l_q8 = lshift(smulbb(fs_khz, 18), 8);
        plc.ltp_coef_q14.fill(0);
    }

    let order = lpc_order.min(MAX_LPC_ORDER);
    plc.prev_lpc_q12[..order].copy_from_slice(&control.pred_coef_q12[1][..order]);
    plc.prev_lpc_q12[order..].fill(0);
    plc.prev_ltp_scale_q14 = clamp_to_i16(control.ltp_scale_q14);

    plc.prev_gain_q16[0] = control.gains_q16[nb_subfr - 2];
    plc.prev_gain_q16[1] = control.gains_q16[nb_subfr - 1];
    plc.subfr_length = sr.subfr_length as i32;
    plc.nb_subfr = nb_subfr as i32;
}

fn silk_plc_conceal(
    state: &mut DecoderState,
    control: &mut DecoderControl,
    frame: &mut [i16],
    arch: i32,
) {
    let frame_length = state.sample_rate.frame_length;
    let nb_subfr = state.sample_rate.nb_subfr;
    let subfr_length = state.sample_rate.subfr_length;
    let ltp_mem_length = state.sample_rate.ltp_mem_length;
    let lpc_order = state.sample_rate.lpc_order;
    let fs_khz = state.sample_rate.fs_khz;
    let prev_gain_q10 = [
        state.plc_state.prev_gain_q16[0] >> 6,
        state.plc_state.prev_gain_q16[1] >> 6,
    ];

    let ((energy1, shift1), (energy2, shift2)) =
        silk_plc_energy(&state.exc_q14, prev_gain_q10, subfr_length, nb_subfr);

    let rand_slice = if rshift(energy1, shift2) < rshift(energy2, shift1) {
        pick_rand_slice(
            &state.exc_q14,
            state.plc_state.nb_subfr,
            state.plc_state.subfr_length,
            false,
        )
    } else {
        pick_rand_slice(
            &state.exc_q14,
            state.plc_state.nb_subfr,
            state.plc_state.subfr_length,
            true,
        )
    };

    let sr = &mut state.sample_rate;
    let plc = &mut state.plc_state;

    let _ = arch;

    assert!(frame_length <= MAX_FRAME_LENGTH);
    assert!(ltp_mem_length <= MAX_LTP_MEM_LENGTH);

    if sr.first_frame_after_reset {
        plc.prev_lpc_q12.fill(0);
    }

    let b_q14 = &mut plc.ltp_coef_q14;
    let mut rand_scale_q14 = i32::from(plc.rand_scale_q14);
    let max_att_idx = min(state.loss_count as usize, NB_ATTENUATION_STEPS - 1);
    let harm_gain_q15 = i32::from(HARM_ATT_Q15[max_att_idx]);
    let rand_gain_q15 = if matches!(sr.prev_signal_type, FrameSignalType::Voiced) {
        i32::from(RAND_ATTENUATE_VOICED_Q15[max_att_idx])
    } else {
        i32::from(RAND_ATTENUATE_UNVOICED_Q15[max_att_idx])
    };

    bwexpander(&mut plc.prev_lpc_q12[..lpc_order], BWE_COEF_Q16);
    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    a_q12[..lpc_order].copy_from_slice(&plc.prev_lpc_q12[..lpc_order]);

    if state.loss_count == 0 {
        rand_scale_q14 = 1 << 14;
        if matches!(sr.prev_signal_type, FrameSignalType::Voiced) {
            for &coef in b_q14.iter() {
                rand_scale_q14 -= i32::from(coef);
            }
            rand_scale_q14 = max(3_277, rand_scale_q14);
            rand_scale_q14 = rshift(
                smulbb(rand_scale_q14, i32::from(plc.prev_ltp_scale_q14)),
                14,
            );
        } else {
            let inv_gain_q30 = lpc_inverse_pred_gain(&plc.prev_lpc_q12[..lpc_order]);
            let mut down_scale = min(rshift(1 << 30, LOG2_INV_LPC_GAIN_HIGH_THRES), inv_gain_q30);
            down_scale = max(rshift(1 << 30, LOG2_INV_LPC_GAIN_LOW_THRES), down_scale);
            down_scale = lshift(down_scale, LOG2_INV_LPC_GAIN_HIGH_THRES);
            rand_scale_q14 = rshift(smulwb(down_scale, rand_gain_q15), 14);
        }
    }

    let mut rand_seed = plc.rand_seed;
    let mut lag = rshift_round(plc.pitch_l_q8, 8);
    let mut s_ltp_q14 = [0i32; MAX_LTP_STATE];
    let mut s_ltp = [0i16; MAX_LTP_MEM_LENGTH];

    let idx =
        ltp_mem_length as isize - lag as isize - lpc_order as isize - (LTP_ORDER / 2) as isize;
    assert!(idx > 0, "invalid PLC re-whitening index");
    let whitening_start = idx as usize;
    let filter_len = ltp_mem_length - whitening_start;
    lpc_analysis_filter(
        &mut s_ltp[whitening_start..ltp_mem_length],
        &state.sample_rate.out_buf[whitening_start..whitening_start + filter_len],
        &a_q12[..lpc_order],
        filter_len,
        lpc_order,
    );

    let mut inv_gain_q30 = inverse32_varq(plc.prev_gain_q16[1], 46);
    inv_gain_q30 = min(inv_gain_q30, i32::MAX >> 1);
    for offset in whitening_start + lpc_order..ltp_mem_length {
        s_ltp_q14[offset] = smulwb(inv_gain_q30, i32::from(s_ltp[offset]));
    }

    let mut s_ltp_buf_idx = ltp_mem_length;
    for _ in 0..nb_subfr {
        for _ in 0..subfr_length {
            let mut ltp_pred_q12 = 2;
            for (tap, coeff) in b_q14.iter().enumerate() {
                let tap_offset = tap as isize - (LTP_ORDER as isize / 2);
                let ref_idx = s_ltp_buf_idx as isize - lag as isize + tap_offset;
                let sample = s_ltp_q14[ref_idx as usize];
                ltp_pred_q12 = smlawb(ltp_pred_q12, sample, i32::from(*coeff));
            }
            rand_seed = silk_rand(rand_seed);
            let noise_idx = ((rand_seed >> 25) as usize) & RAND_BUF_MASK;
            let excitation = smlawb(ltp_pred_q12, rand_slice[noise_idx], rand_scale_q14);
            s_ltp_q14[s_ltp_buf_idx] = lshift_sat32(excitation, 2);
            s_ltp_buf_idx += 1;
        }

        for coef in b_q14.iter_mut() {
            *coef = clamp_to_i16(rshift(smulbb(harm_gain_q15, i32::from(*coef)), 15));
        }
        rand_scale_q14 = rshift(smulbb(rand_scale_q14, rand_gain_q15), 15);

        plc.pitch_l_q8 = plc.pitch_l_q8 + smulwb(plc.pitch_l_q8, PITCH_DRIFT_FAC_Q16);
        let max_lag_q8 = lshift(smulbb(MAX_PITCH_LAG_MS, fs_khz), 8);
        plc.pitch_l_q8 = min(plc.pitch_l_q8, max_lag_q8);
        lag = rshift_round(plc.pitch_l_q8, 8);
    }

    let s_lpc_start = ltp_mem_length - MAX_LPC_ORDER;
    let total_len = MAX_LPC_ORDER + frame_length;
    let s_lpc = &mut s_ltp_q14[s_lpc_start..s_lpc_start + total_len];
    s_lpc[..MAX_LPC_ORDER].copy_from_slice(&state.sample_rate.s_lpc_q14_buf[..MAX_LPC_ORDER]);

    for i in 0..frame_length {
        let mut lpc_pred_q10 = (lpc_order as i32) >> 1;
        for j in 0..lpc_order {
            let sample = s_lpc[MAX_LPC_ORDER + i - j - 1];
            lpc_pred_q10 = smlawb(lpc_pred_q10, sample, i32::from(a_q12[j]));
        }

        let idx = MAX_LPC_ORDER + i;
        let updated = add_sat32(s_lpc[idx], lshift_sat32(lpc_pred_q10, 4));
        s_lpc[idx] = updated;
        let scaled = rshift_round(smulww(updated, prev_gain_q10[1]), 8);
        frame[i] = sat16(scaled);
    }

    state
        .sample_rate
        .s_lpc_q14_buf
        .copy_from_slice(&s_lpc[frame_length..frame_length + MAX_LPC_ORDER]);

    plc.rand_seed = rand_seed;
    plc.rand_scale_q14 = clamp_to_i16(rand_scale_q14);
    control
        .pitch_l
        .iter_mut()
        .for_each(|lag_slot| *lag_slot = lag);
}

fn silk_plc_energy(
    exc_q14: &[i32; MAX_FRAME_LENGTH],
    prev_gain_q10: [i32; 2],
    subfr_length: usize,
    nb_subfr: usize,
) -> ((i32, i32), (i32, i32)) {
    debug_assert!(subfr_length <= MAX_SUB_FRAME_LENGTH);
    debug_assert!(nb_subfr >= 2);

    let mut exc_buf = [0i16; 2 * MAX_SUB_FRAME_LENGTH];
    for k in 0..2 {
        let base = (k + nb_subfr - 2) * subfr_length;
        for i in 0..subfr_length {
            let idx = base + i;
            let sample = exc_q14.get(idx).copied().unwrap_or(0);
            let scaled = smulww(sample, prev_gain_q10[k]);
            exc_buf[k * MAX_SUB_FRAME_LENGTH + i] = sat16(rshift(scaled, 8));
        }
    }

    let first = sum_sqr_shift(&exc_buf[..subfr_length]);
    let second = sum_sqr_shift(&exc_buf[MAX_SUB_FRAME_LENGTH..MAX_SUB_FRAME_LENGTH + subfr_length]);
    (first, second)
}

fn pick_rand_slice(
    exc_q14: &[i32; MAX_FRAME_LENGTH],
    nb_subfr: i32,
    subfr_length: i32,
    second: bool,
) -> [i32; RAND_BUF_SIZE] {
    let subfr_length = max(subfr_length, 1) as usize;
    let nb_subfr = max(nb_subfr, 2) as usize;
    let base = if second {
        nb_subfr * subfr_length
    } else {
        nb_subfr.saturating_sub(1) * subfr_length
    };
    let max_start = MAX_FRAME_LENGTH.saturating_sub(RAND_BUF_SIZE);
    let start = min(base.saturating_sub(RAND_BUF_SIZE), max_start);
    let mut buf = [0i32; RAND_BUF_SIZE];
    buf.copy_from_slice(&exc_q14[start..start + RAND_BUF_SIZE]);
    buf
}

fn rshift(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value.wrapping_shl((-shift) as u32)
    } else {
        value >> shift
    }
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

fn smlawb(acc: i32, x: i32, y: i32) -> i32 {
    acc.wrapping_add(smulwb(x, y))
}

fn add_sat32(a: i32, b: i32) -> i32 {
    (i64::from(a) + i64::from(b)).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn lshift(value: i32, shift: i32) -> i32 {
    value.wrapping_shl(shift as u32)
}

fn lshift_sat32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value >> (-shift)
    } else {
        let widened = i64::from(value) << shift;
        widened.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return value.wrapping_shl((-shift) as u32);
    }
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn sat16(value: i32) -> i16 {
    clamp_to_i16(value)
}

fn clamp_to_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn div32(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}

fn div32_16(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}

fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let leading = x.leading_zeros() as i32;
    let frac = ((x as u32).rotate_right(((24 - leading) & 31) as u32) & 0x7f) as i32;
    let mut y = if leading & 1 != 0 { 32_768 } else { 46_214 };
    y >>= leading >> 1;
    smlawb(y, y, smulbb(213, frac))
}

#[cfg(test)]
mod tests {
    use super::{silk_plc_glue_frames, silk_plc_reset, silk_plc_update};
    use crate::silk::decoder_control::DecoderControl;
    use crate::silk::decoder_state::DecoderState;
    use crate::silk::vq_wmat_ec::LTP_ORDER;
    use crate::silk::{FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR};

    fn decoder_state() -> DecoderState {
        let mut state = DecoderState::default();
        state.sample_rate.frame_length = 160;
        state.sample_rate.subfr_length = 40;
        state.sample_rate.nb_subfr = 4;
        state.sample_rate.fs_khz = 16;
        state.sample_rate.ltp_mem_length = 320;
        state.sample_rate.lpc_order = MAX_LPC_ORDER;
        state
    }

    #[test]
    fn reset_primes_pitch_and_gains() {
        let mut state = decoder_state();
        silk_plc_reset(&mut state);
        assert_eq!(state.plc_state.pitch_l_q8, 160 << 7);
        assert_eq!(state.plc_state.prev_gain_q16, [1 << 16; 2]);
    }

    #[test]
    fn update_tracks_signal_type_and_gains() {
        let mut state = decoder_state();
        state.indices.signal_type = FrameSignalType::Voiced;
        let mut control = DecoderControl::default();
        control.pitch_l = [60; MAX_NB_SUBFR];
        control.ltp_coef_q14 = [0; MAX_NB_SUBFR * LTP_ORDER];
        control.ltp_coef_q14[LTP_ORDER / 2] = 8_000;
        control.gains_q16 = [1 << 16; MAX_NB_SUBFR];
        control.pred_coef_q12[1] = [7; MAX_LPC_ORDER];
        silk_plc_update(&mut state, &control);
        assert_eq!(state.sample_rate.prev_signal_type, FrameSignalType::Voiced);
        assert_eq!(state.plc_state.prev_lpc_q12[0], 7);
        assert_eq!(state.plc_state.prev_gain_q16[1], 1 << 16);
    }

    #[test]
    fn glue_frames_records_concealment_energy() {
        let mut state = decoder_state();
        state.loss_count = 1;
        let mut frame = [1i16; 160];
        silk_plc_glue_frames(&mut state, &mut frame);
        assert!(state.plc_state.conc_energy > 0);
        assert_eq!(state.plc_state.last_frame_lost, 1);
    }
}
