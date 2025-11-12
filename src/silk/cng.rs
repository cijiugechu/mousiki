//! Port of the SILK comfort-noise generator (`silk/CNG.c`).
//!
//! This module mirrors the fixed-point helper that smooths decoder-side
//! excitation statistics and synthesises artificial noise whenever packet loss
//! occurs.  The routines operate entirely on stack-backed slices mirroring the
//! C reference implementation to avoid heap allocations.

use alloc::vec;
use core::cmp::Ordering;

use crate::silk::nlsf2a::nlsf2a;
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR};

const SUBFRAME_MS: usize = 5;
const MAX_FS_KHZ: usize = 16;
const MAX_SUBFRAME_LENGTH: usize = SUBFRAME_MS * MAX_FS_KHZ;
const MAX_FRAME_LENGTH: usize = MAX_SUBFRAME_LENGTH * MAX_NB_SUBFR;

const CNG_BUF_MASK_MAX: i32 = 255;
const CNG_GAIN_SMTH_Q16: i32 = 4_634;
const CNG_GAIN_SMTH_THRESHOLD_Q16: i32 = 46_396;
const CNG_NLSF_SMTH_Q16: i32 = 16_348;
const INITIAL_RAND_SEED: i32 = 3_176_576;
const RAND_MULTIPLIER: i32 = 196_314_165;
const RAND_INCREMENT: i32 = 907_633_515;
const INVALID_FS_KHZ: i32 = -1;

/// Decoder-side comfort-noise state (`silk_CNG_struct`).
#[derive(Clone, Debug)]
pub struct CngState {
    exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    synth_state: [i32; MAX_LPC_ORDER],
    smth_gain_q16: i32,
    rand_seed: i32,
    fs_khz: i32,
}

impl Default for CngState {
    fn default() -> Self {
        Self {
            exc_buf_q14: [0; MAX_FRAME_LENGTH],
            smth_nlsf_q15: [0; MAX_LPC_ORDER],
            synth_state: [0; MAX_LPC_ORDER],
            smth_gain_q16: 0,
            rand_seed: INITIAL_RAND_SEED,
            fs_khz: INVALID_FS_KHZ,
        }
    }
}

impl CngState {
    /// Creates a zeroed comfort-noise state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reinitialises the smoothed NLSF grid and resets the internal gain/seed.
    pub fn reset(&mut self, lpc_order: usize) {
        assert!(
            (1..=MAX_LPC_ORDER).contains(&lpc_order),
            "invalid LPC order {lpc_order}"
        );
        let step_q15 = div32_16(i32::from(i16::MAX), (lpc_order + 1) as i32);
        let mut acc_q15: i32 = 0;
        for value in self.smth_nlsf_q15.iter_mut().take(lpc_order) {
            acc_q15 = acc_q15.saturating_add(step_q15);
            *value = acc_q15 as i16;
        }
        for value in self.smth_nlsf_q15.iter_mut().skip(lpc_order) {
            *value = 0;
        }
        self.smth_gain_q16 = 0;
        self.rand_seed = INITIAL_RAND_SEED;
        self.synth_state = [0; MAX_LPC_ORDER];
    }

    /// Returns the current smoothed LPC gain in Q16.
    #[must_use]
    pub fn smoothed_gain_q16(&self) -> i32 {
        self.smth_gain_q16
    }

    /// Borrows the smoothed NLSF vector.
    #[must_use]
    pub fn smoothed_nlsf_q15(&self) -> &[i16; MAX_LPC_ORDER] {
        &self.smth_nlsf_q15
    }

    /// Borrows the rolling excitation buffer.
    #[must_use]
    pub fn excitation_buffer(&self) -> &[i32; MAX_FRAME_LENGTH] {
        &self.exc_buf_q14
    }

    /// Mutably borrows the rolling excitation buffer.
    pub fn excitation_buffer_mut(&mut self) -> &mut [i32; MAX_FRAME_LENGTH] {
        &mut self.exc_buf_q14
    }

    /// Borrows the LPC synthesis history buffer (Q14).
    #[must_use]
    pub fn synth_state(&self) -> &[i32; MAX_LPC_ORDER] {
        &self.synth_state
    }

    /// Returns the last internal sampling rate (kHz) tracked by the state.
    #[must_use]
    pub fn fs_khz(&self) -> i32 {
        self.fs_khz
    }
}

/// Packet-loss concealment summary used by the CNG helper.
#[derive(Clone, Debug, Default)]
pub struct PlcState {
    pub rand_scale_q14: i32,
    pub prev_gain_q16: [i32; 2],
}

/// Minimal decoder control stub containing the per-subframe decoder parameters.
#[derive(Clone, Debug, Default)]
pub struct DecoderControl {
    /// Q16 gains per subframe.
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// LPC predictor coefficients for each half-frame (Q12).
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    /// Pitch lags per subframe (samples).
    pub pitch_l: [i16; MAX_NB_SUBFR],
    /// LTP coefficients for each subframe (Q14).
    pub ltp_coef_q14: [i16; MAX_NB_SUBFR * LTP_ORDER],
    /// LTP scaling factor (Q14).
    pub ltp_scale_q14: i16,
}

/// Parameters required to update the comfort-noise state.
#[derive(Clone, Debug)]
pub struct ComfortNoiseInputs<'a> {
    pub fs_khz: i32,
    pub lpc_order: usize,
    pub nb_subfr: usize,
    pub subfr_length: usize,
    pub prev_signal_type: FrameSignalType,
    pub loss_count: i32,
    pub prev_nlsf_q15: &'a [i16],
    pub exc_q14: &'a [i32],
}

impl<'a> ComfortNoiseInputs<'a> {
    #[must_use]
    pub fn frame_length(&self) -> usize {
        self.nb_subfr * self.subfr_length
    }
}

/// Mirrors `silk_CNG`: updates the comfort-noise statistics and injects noise
/// when the decoder reports packet loss.
pub fn apply_cng(
    state: &mut CngState,
    plc: &PlcState,
    control: &DecoderControl,
    inputs: &ComfortNoiseInputs<'_>,
    frame: &mut [i16],
) {
    validate_inputs(inputs, frame);

    if inputs.fs_khz != state.fs_khz {
        if state.fs_khz == INVALID_FS_KHZ {
            state
                .synth_state
                .iter_mut()
                .take(inputs.lpc_order)
                .for_each(|value| *value = 0);
        } else {
            state.reset(inputs.lpc_order);
        }
        state.smth_nlsf_q15[..inputs.lpc_order]
            .copy_from_slice(&inputs.prev_nlsf_q15[..inputs.lpc_order]);
        state
            .smth_nlsf_q15
            .iter_mut()
            .skip(inputs.lpc_order)
            .for_each(|value| *value = 0);
        state.fs_khz = inputs.fs_khz;
    }

    if inputs.loss_count == 0 && inputs.prev_signal_type == FrameSignalType::Inactive {
        smooth_nlsf(state, inputs);
        update_excitation_buffer(state, inputs, control);
        smooth_gain(state, control, inputs.nb_subfr);
    }

    if inputs.loss_count > 0 {
        synthesize_noise(state, plc, inputs, frame);
    } else {
        state
            .synth_state
            .iter_mut()
            .take(inputs.lpc_order)
            .for_each(|value| *value = 0);
    }
}

fn validate_inputs(inputs: &ComfortNoiseInputs<'_>, frame: &[i16]) {
    assert!(
        (1..=MAX_LPC_ORDER).contains(&inputs.lpc_order),
        "invalid LPC order {}",
        inputs.lpc_order
    );
    assert!(
        inputs.nb_subfr > 0 && inputs.nb_subfr <= MAX_NB_SUBFR,
        "invalid subframe count {}",
        inputs.nb_subfr
    );
    assert!(inputs.subfr_length > 0 && inputs.subfr_length <= MAX_SUBFRAME_LENGTH);
    assert!(
        frame.len() == inputs.frame_length(),
        "frame length {} does not match {}",
        frame.len(),
        inputs.frame_length()
    );
    assert!(
        frame.len() <= MAX_FRAME_LENGTH,
        "frame length {} exceeds MAX_FRAME_LENGTH {}",
        frame.len(),
        MAX_FRAME_LENGTH
    );
    assert!(
        inputs.prev_nlsf_q15.len() >= inputs.lpc_order,
        "prev_nlsf slice too small"
    );
    assert!(
        inputs.exc_q14.len() >= inputs.frame_length(),
        "excitation buffer too small"
    );
}

fn smooth_nlsf(state: &mut CngState, inputs: &ComfortNoiseInputs<'_>) {
    for (idx, target) in state
        .smth_nlsf_q15
        .iter_mut()
        .take(inputs.lpc_order)
        .enumerate()
    {
        let prev = i32::from(inputs.prev_nlsf_q15[idx]);
        let current = i32::from(*target);
        let updated = current + smulwb(prev - current, CNG_NLSF_SMTH_Q16);
        *target = updated as i16;
    }
}

fn update_excitation_buffer(
    state: &mut CngState,
    inputs: &ComfortNoiseInputs<'_>,
    control: &DecoderControl,
) {
    let mut max_gain_q16 = 0;
    let mut strongest_subfr = 0usize;
    for (idx, &gain) in control.gains_q16.iter().take(inputs.nb_subfr).enumerate() {
        if gain > max_gain_q16 {
            max_gain_q16 = gain;
            strongest_subfr = idx;
        }
    }

    if inputs.nb_subfr == 0 {
        return;
    }

    let move_len = inputs.subfr_length * (inputs.nb_subfr - 1);
    if move_len > 0 {
        state
            .exc_buf_q14
            .copy_within(0..move_len, inputs.subfr_length);
    }

    let start = strongest_subfr * inputs.subfr_length;
    let end = start + inputs.subfr_length;
    state.exc_buf_q14[..inputs.subfr_length].copy_from_slice(&inputs.exc_q14[start..end]);
}

fn smooth_gain(state: &mut CngState, control: &DecoderControl, nb_subfr: usize) {
    for gain in control.gains_q16.iter().take(nb_subfr) {
        state.smth_gain_q16 =
            state.smth_gain_q16 + smulwb(*gain - state.smth_gain_q16, CNG_GAIN_SMTH_Q16);
        if smulww(state.smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16) > *gain {
            state.smth_gain_q16 = *gain;
        }
    }
}

fn synthesize_noise(
    state: &mut CngState,
    plc: &PlcState,
    inputs: &ComfortNoiseInputs<'_>,
    frame: &mut [i16],
) {
    let length = frame.len();
    let mut cng_sig_q14 = vec![0i32; length + MAX_LPC_ORDER];

    let mut gain_q16 = smulww(plc.rand_scale_q14, plc.prev_gain_q16[1]);
    if gain_q16 >= (1 << 21) || state.smth_gain_q16 > (1 << 23) {
        gain_q16 = smultt(gain_q16, gain_q16);
        gain_q16 = sub_lshift32(
            smultt(state.smth_gain_q16, state.smth_gain_q16),
            gain_q16,
            5,
        );
        gain_q16 = lshift_sat32(sqrt_approx(gain_q16), 16);
    } else {
        gain_q16 = smulww(gain_q16, gain_q16);
        gain_q16 = sub_lshift32(
            smulww(state.smth_gain_q16, state.smth_gain_q16),
            gain_q16,
            5,
        );
        gain_q16 = lshift_sat32(sqrt_approx(gain_q16), 8);
    }
    let gain_q10 = gain_q16 >> 6;
    generate_excitation(
        &mut cng_sig_q14[MAX_LPC_ORDER..],
        &state.exc_buf_q14,
        length,
        &mut state.rand_seed,
    );

    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    nlsf2a(
        &mut a_q12[..inputs.lpc_order],
        &state.smth_nlsf_q15[..inputs.lpc_order],
        0,
    );

    cng_sig_q14[..MAX_LPC_ORDER].copy_from_slice(&state.synth_state);

    for i in 0..length {
        let mut lpc_pred_q10 = (inputs.lpc_order as i32) >> 1;
        for (tap, coeff) in a_q12.iter().take(inputs.lpc_order).enumerate() {
            let hist_idx = MAX_LPC_ORDER + i - 1 - tap;
            lpc_pred_q10 = smlawb(lpc_pred_q10, cng_sig_q14[hist_idx], i32::from(*coeff));
        }
        let updated = add_sat32(
            cng_sig_q14[MAX_LPC_ORDER + i],
            lshift_sat32(lpc_pred_q10, 4),
        );
        cng_sig_q14[MAX_LPC_ORDER + i] = updated;

        let scaled = smulww(updated, gain_q10);
        let sample = sat16(rshift_round(scaled, 8));
        frame[i] = add_sat16(frame[i], sample);
    }

    state
        .synth_state
        .copy_from_slice(&cng_sig_q14[length..length + MAX_LPC_ORDER]);
}

fn generate_excitation(
    out: &mut [i32],
    exc_buf: &[i32; MAX_FRAME_LENGTH],
    length: usize,
    seed: &mut i32,
) {
    if out.is_empty() {
        return;
    }
    let mut exc_mask = CNG_BUF_MASK_MAX;
    let target = length as i32;
    while exc_mask > target {
        exc_mask >>= 1;
    }

    let mut current_seed = *seed;
    for sample in out.iter_mut() {
        current_seed = silk_rand(current_seed);
        let idx =
            ((current_seed >> 24) & exc_mask).clamp(0, (MAX_FRAME_LENGTH - 1) as i32) as usize;
        *sample = exc_buf[idx];
    }
    *seed = current_seed;
}

pub(crate) fn silk_rand(seed: i32) -> i32 {
    RAND_INCREMENT.wrapping_add(seed.wrapping_mul(RAND_MULTIPLIER))
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(i32::from(b as i16))) >> 16) as i32
}

fn smlawb(acc: i32, b: i32, c: i32) -> i32 {
    acc.wrapping_add(smulwb(b, c))
}

fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

fn smultt(a: i32, b: i32) -> i32 {
    (a >> 16) * (b >> 16)
}

fn div32_16(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}

fn add_sat32(a: i32, b: i32) -> i32 {
    let sum = i64::from(a) + i64::from(b);
    sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn add_sat16(a: i16, b: i16) -> i16 {
    sat16(i32::from(a) + i32::from(b))
}

fn sat16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn lshift_sat32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value >> (-shift)
    } else if shift >= 31 {
        match value.cmp(&0) {
            Ordering::Greater => i32::MAX,
            Ordering::Less => i32::MIN,
            Ordering::Equal => 0,
        }
    } else {
        let min_val = i32::MIN >> shift;
        let max_val = i32::MAX >> shift;
        (value.clamp(min_val, max_val)) << shift
    }
}

fn sub_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_sub(b.wrapping_shl(shift as u32))
}

fn rshift_round(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        value.wrapping_shl((-shift) as u32)
    } else if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32_768 } else { 46_214 };
    y >>= lz >> 1;
    smlawb(y, y, smulbb(213, frac_q7))
}

fn clz_frac(x: i32) -> (i32, i32) {
    let ux = x as u32;
    let lz = ux.leading_zeros() as i32;
    let rotate = ((24 - lz) & 31) as u32;
    let frac = (ux.rotate_right(rotate) & 0x7f) as i32;
    (lz, frac)
}

fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn default_inputs<'a>(
        prev_nlsf_q15: &'a [i16],
        exc_q14: &'a [i32],
        loss_count: i32,
        prev_signal_type: FrameSignalType,
        subfr_length: usize,
        nb_subfr: usize,
        lpc_order: usize,
    ) -> ComfortNoiseInputs<'a> {
        ComfortNoiseInputs {
            fs_khz: 16,
            lpc_order,
            nb_subfr,
            subfr_length,
            prev_signal_type,
            loss_count,
            prev_nlsf_q15,
            exc_q14,
        }
    }

    #[test]
    fn reset_seeds_nlsf_grid() {
        let mut state = CngState::new();
        state.reset(10);
        let step_q15 = div32_16(i32::from(i16::MAX), 11);
        for (idx, &value) in state.smth_nlsf_q15.iter().take(10).enumerate() {
            let expected = ((idx as i32 + 1) * step_q15) as i16;
            assert_eq!(value, expected);
        }
        assert_eq!(state.smoothed_gain_q16(), 0);
        assert_eq!(state.rand_seed, INITIAL_RAND_SEED);
    }

    #[test]
    fn update_tracks_highest_gain_subframe() {
        let mut state = CngState::new();
        let plc = PlcState::default();
        let mut control = DecoderControl::default();
        control.gains_q16 = [1 << 16, 2 << 16, 4 << 16, 3 << 16];

        let subfr_length = 4;
        let nb_subfr = 4;
        let lpc_order = 10;
        let frame_len = subfr_length * nb_subfr;

        let prev_nlsf: Vec<i16> = (0..MAX_LPC_ORDER).map(|idx| (idx as i16) * 123).collect();
        let mut exc_q14 = vec![0i32; frame_len];
        for (idx, sample) in exc_q14.iter_mut().enumerate() {
            *sample = (idx as i32 + 1) * 10;
        }
        let inputs = default_inputs(
            &prev_nlsf,
            &exc_q14,
            0,
            FrameSignalType::Inactive,
            subfr_length,
            nb_subfr,
            lpc_order,
        );
        let mut frame = vec![0i16; frame_len];

        apply_cng(&mut state, &plc, &control, &inputs, &mut frame);

        let mut expected_gain = 0;
        for gain in control.gains_q16.iter().take(nb_subfr) {
            expected_gain += smulwb(*gain - expected_gain, CNG_GAIN_SMTH_Q16);
            if smulww(expected_gain, CNG_GAIN_SMTH_THRESHOLD_Q16) > *gain {
                expected_gain = *gain;
            }
        }
        assert_eq!(state.smoothed_gain_q16(), expected_gain);
        assert_eq!(frame, vec![0i16; frame_len]);
        let max_subfr_start = 2 * subfr_length;
        assert_eq!(
            &state.exc_buf_q14[..subfr_length],
            &exc_q14[max_subfr_start..max_subfr_start + subfr_length]
        );
        assert!(
            state
                .smth_nlsf_q15
                .iter()
                .take(lpc_order)
                .zip(prev_nlsf.iter())
                .all(|(&smoothed, &target)| smoothed.abs() <= target.abs()),
            "smoothed_nlsf={:?} prev={:?}",
            &state.smth_nlsf_q15[..lpc_order],
            &prev_nlsf[..lpc_order]
        );
    }

    #[test]
    fn loss_path_injects_noise() {
        let mut state = CngState::new();
        state.reset(10);
        state.smth_gain_q16 = 1 << 20;
        state
            .smth_nlsf_q15
            .iter_mut()
            .enumerate()
            .for_each(|(idx, value)| *value = 200 + (idx as i16) * 50);
        state
            .exc_buf_q14
            .iter_mut()
            .enumerate()
            .for_each(|(idx, sample)| *sample = (idx as i32) * 113);
        let mut state_clone = state.clone();

        let mut control = DecoderControl::default();
        control.gains_q16 = [1 << 16; MAX_NB_SUBFR];
        let mut plc = PlcState::default();
        plc.rand_scale_q14 = 1 << 14;
        plc.prev_gain_q16 = [1 << 16, 1 << 16];

        let subfr_length = 8;
        let nb_subfr = 2;
        let lpc_order = 10;
        let frame_len = subfr_length * nb_subfr;
        let prev_nlsf = vec![300i16; MAX_LPC_ORDER];
        let exc_q14 = vec![0i32; frame_len];
        let mut frame = vec![0i16; frame_len];
        let inputs = default_inputs(
            &prev_nlsf,
            &exc_q14,
            1,
            FrameSignalType::Voiced,
            subfr_length,
            nb_subfr,
            lpc_order,
        );

        apply_cng(&mut state, &plc, &control, &inputs, &mut frame);
        assert!(frame.iter().any(|&sample| sample != 0), "frame={:?}", frame);
        assert!(
            state
                .synth_state
                .iter()
                .take(lpc_order)
                .any(|&value| value != 0)
        );

        // Deterministic seed evolution ensures identical results when re-running.
        let mut expected_frame = vec![0i16; frame_len];
        apply_cng(
            &mut state_clone,
            &plc,
            &control,
            &inputs,
            &mut expected_frame,
        );
        assert_eq!(frame, expected_frame);
    }
}
