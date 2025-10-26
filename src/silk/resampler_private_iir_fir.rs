//! Port of SILK's hybrid IIR/FIR resampler.
//!
//! This module translates `silk_resampler_private_IIR_FIR` from
//! `silk/resampler_private_IIR_FIR.c`. The routine upsamples incoming 16-bit
//! audio by a factor of two using the shared high-quality all-pass sections and
//! then applies 1/12th-phase FIR interpolation driven by the fractional step
//! stored in `inv_ratio_q16`.

use alloc::vec;
use alloc::vec::Vec;

use super::resampler_private_up2_hq::resampler_private_up2_hq;
use super::resampler_rom::{RESAMPLER_ORDER_FIR_12, SILK_RESAMPLER_FRAC_FIR_12};

/// Minimal state required by the hybrid IIR/FIR resampler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResamplerStateIirFir {
    /// Delay elements for the high-quality 2× upsampler (Q10 domain).
    s_iir: [i32; 6],
    /// Tail of the FIR delay line preserved between calls (Q0 samples).
    s_fir: [i16; RESAMPLER_ORDER_FIR_12],
    /// Number of input samples processed per iteration.
    batch_size: usize,
    /// Fixed-point step size between output samples (Q16 domain).
    inv_ratio_q16: i32,
    /// Scratch buffer reused to avoid per-call allocation.
    scratch: Vec<i16>,
}

impl ResamplerStateIirFir {
    /// Creates a new resampler state with zeroed delay elements.
    pub fn new(batch_size: usize, inv_ratio_q16: i32) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than zero");
        assert!(inv_ratio_q16 > 0, "inv_ratio_q16 must be positive");
        let scratch_len = 2 * batch_size + RESAMPLER_ORDER_FIR_12;
        Self {
            s_iir: [0; 6],
            s_fir: [0; RESAMPLER_ORDER_FIR_12],
            batch_size,
            inv_ratio_q16,
            scratch: vec![0; scratch_len],
        }
    }

    /// Returns the configured batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the configured Q16 step size.
    pub fn inv_ratio_q16(&self) -> i32 {
        self.inv_ratio_q16
    }

    /// Exposes the FIR tail retained between calls.
    pub fn fir_tail(&self) -> &[i16; RESAMPLER_ORDER_FIR_12] {
        &self.s_fir
    }

    /// Exposes the internal IIR delay elements.
    pub fn iir_state(&self) -> &[i32; 6] {
        &self.s_iir
    }

    fn ensure_scratch_capacity(&mut self) {
        let required = 2 * self.batch_size + RESAMPLER_ORDER_FIR_12;
        if self.scratch.len() < required {
            self.scratch.resize(required, 0);
        }
    }

    #[cfg(test)]
    fn iir_state_mut(&mut self) -> &mut [i32; 6] {
        &mut self.s_iir
    }

    #[cfg(test)]
    fn fir_tail_mut(&mut self) -> &mut [i16; RESAMPLER_ORDER_FIR_12] {
        &mut self.s_fir
    }
}

/// Runs the hybrid IIR/FIR resampler on `input`, writing results into `output`.
///
/// Returns the number of produced samples.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn resampler_private_iir_fir(
    state: &mut ResamplerStateIirFir,
    output: &mut [i16],
    input: &[i16],
) -> usize {
    if input.is_empty() {
        return 0;
    }

    state.ensure_scratch_capacity();
    let batch_size = state.batch_size;
    let inv_ratio_q16 = state.inv_ratio_q16;
    debug_assert!(inv_ratio_q16 > 0);

    let mut remaining = input.len();
    let mut in_index = 0usize;
    let mut out_index = 0usize;
    let mut last_n_samples_in = 0usize;

    let buf_len = 2 * batch_size + RESAMPLER_ORDER_FIR_12;
    let buf = &mut state.scratch[..buf_len];
    buf[..RESAMPLER_ORDER_FIR_12].copy_from_slice(&state.s_fir);

    while remaining > 0 {
        let n_samples_in = remaining.min(batch_size);
        let upsampled_len = n_samples_in * 2;
        let range = RESAMPLER_ORDER_FIR_12..RESAMPLER_ORDER_FIR_12 + upsampled_len;
        resampler_private_up2_hq(
            &mut state.s_iir,
            &mut buf[range.clone()],
            &input[in_index..in_index + n_samples_in],
        );

        let max_index_q16 = (n_samples_in as i32) << 17;
        out_index += resampler_private_iir_fir_interpol(
            buf,
            max_index_q16,
            inv_ratio_q16,
            &mut output[out_index..],
        );

        in_index += n_samples_in;
        remaining -= n_samples_in;
        last_n_samples_in = n_samples_in;

        if remaining > 0 {
            buf.copy_within(upsampled_len..upsampled_len + RESAMPLER_ORDER_FIR_12, 0);
        }
    }

    if last_n_samples_in > 0 {
        let tail_offset = last_n_samples_in * 2;
        state
            .s_fir
            .copy_from_slice(&buf[tail_offset..tail_offset + RESAMPLER_ORDER_FIR_12]);
    }

    out_index
}

fn resampler_private_iir_fir_interpol(
    buf: &[i16],
    max_index_q16: i32,
    index_increment_q16: i32,
    output: &mut [i16],
) -> usize {
    debug_assert!(index_increment_q16 > 0);
    if index_increment_q16 > 0 && max_index_q16 > 0 {
        let required =
            ((i64::from(max_index_q16 - 1) / i64::from(index_increment_q16)) + 1) as usize;
        assert!(
            required <= output.len(),
            "output buffer too small: need at least {} samples",
            required
        );
    }

    let mut out_index = 0usize;
    let mut index_q16 = 0i32;
    while index_q16 < max_index_q16 {
        let frac = index_q16 & 0xFFFF;
        let table_index = smulwb(frac, 12) as usize;
        let base = (index_q16 >> 16) as usize;
        let buf_ptr = &buf[base..base + RESAMPLER_ORDER_FIR_12];

        let forward = SILK_RESAMPLER_FRAC_FIR_12[table_index];
        let backward = SILK_RESAMPLER_FRAC_FIR_12[11 - table_index];

        let mut acc = smulbb(buf_ptr[0], forward[0]);
        acc = smlabb(acc, buf_ptr[1], forward[1]);
        acc = smlabb(acc, buf_ptr[2], forward[2]);
        acc = smlabb(acc, buf_ptr[3], forward[3]);
        acc = smlabb(acc, buf_ptr[4], backward[3]);
        acc = smlabb(acc, buf_ptr[5], backward[2]);
        acc = smlabb(acc, buf_ptr[6], backward[1]);
        acc = smlabb(acc, buf_ptr[7], backward[0]);

        output[out_index] = sat16(rshift_round(acc, 15));
        out_index += 1;
        index_q16 = index_q16.wrapping_add(index_increment_q16);
    }

    out_index
}

#[inline]
fn smulwb(a: i32, b: i32) -> i32 {
    let product = i64::from(a) * i64::from(b as i16);
    (product >> 16) as i32
}

#[inline]
fn smulbb(a: i16, b: i16) -> i32 {
    i32::from(a) * i32::from(b)
}

#[inline]
fn smlabb(acc: i32, a: i16, b: i16) -> i32 {
    acc.wrapping_add(smulbb(a, b))
}

#[inline]
fn sat16(value: i32) -> i16 {
    if value > i32::from(i16::MAX) {
        i16::MAX
    } else if value < i32::from(i16::MIN) {
        i16::MIN
    } else {
        value as i16
    }
}

#[inline]
fn rshift_round(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        value
    } else {
        let offset = 1 << (shift - 1);
        value.wrapping_add(offset) >> shift
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{ResamplerStateIirFir, resampler_private_iir_fir};

    #[test]
    fn produces_zero_output_for_zero_input() {
        let mut state = ResamplerStateIirFir::new(8, 1 << 16);
        let input = [0i16; 12];
        let mut output = [1i16; 32];
        let produced = resampler_private_iir_fir(&mut state, &mut output, &input);
        assert_eq!(produced, 24);
        assert!(output[..produced].iter().all(|&sample| sample == 0));
        assert_eq!(state.iir_state(), &[0; 6]);
        assert_eq!(state.fir_tail(), &[0; super::RESAMPLER_ORDER_FIR_12]);
    }

    #[test]
    fn matches_reference_implementation() {
        let mut state = ResamplerStateIirFir::new(10, 50_000);
        let mut reference = state.clone();
        let input = [
            1200i16, -800, 600, -400, 300, -200, 150, -90, 60, -40, 20, -10, 5, -2,
        ];
        let expected = reference_resampler_private_iir_fir(&mut reference, &input);
        let mut output = vec![0i16; expected.len()];
        let produced = resampler_private_iir_fir(&mut state, &mut output, &input);
        assert_eq!(produced, expected.len());
        assert_eq!(&output[..produced], expected.as_slice());
        assert_eq!(state.fir_tail(), reference.fir_tail());
        assert_eq!(state.iir_state(), reference.iir_state());
    }

    #[test]
    fn handles_multiple_batches() {
        let mut state = ResamplerStateIirFir::new(6, 35_000);
        let mut reference = state.clone();
        let input = [400i16, -300, 250, -200, 150, -120, 90, -60, 30, -15];
        let expected = reference_resampler_private_iir_fir(&mut reference, &input);
        let mut output = vec![0i16; expected.len()];
        let produced = resampler_private_iir_fir(&mut state, &mut output, &input);
        assert_eq!(produced, expected.len());
        assert_eq!(&output[..produced], expected.as_slice());
        assert_eq!(state.fir_tail(), reference.fir_tail());
        assert_eq!(state.iir_state(), reference.iir_state());
    }

    fn reference_resampler_private_iir_fir(
        state: &mut ResamplerStateIirFir,
        input: &[i16],
    ) -> Vec<i16> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut scratch = vec![0i16; 2 * state.batch_size() + super::RESAMPLER_ORDER_FIR_12];
        scratch[..super::RESAMPLER_ORDER_FIR_12].copy_from_slice(state.fir_tail());

        let mut outputs = Vec::new();
        let mut remaining = input.len();
        let mut in_index = 0usize;
        let mut last_n_samples_in = 0usize;

        while remaining > 0 {
            let n_samples_in = remaining.min(state.batch_size());
            let upsampled_len = n_samples_in * 2;
            let range =
                super::RESAMPLER_ORDER_FIR_12..super::RESAMPLER_ORDER_FIR_12 + upsampled_len;
            super::resampler_private_up2_hq(
                state.iir_state_mut(),
                &mut scratch[range.clone()],
                &input[in_index..in_index + n_samples_in],
            );

            let mut index_q16 = 0i32;
            let max_index_q16 = (n_samples_in as i32) << 17;
            while index_q16 < max_index_q16 {
                let frac = index_q16 & 0xFFFF;
                let table_index = super::smulwb(frac, 12) as usize;
                let base = (index_q16 >> 16) as usize;
                let buf_ptr = &scratch[base..base + super::RESAMPLER_ORDER_FIR_12];
                let forward = super::SILK_RESAMPLER_FRAC_FIR_12[table_index];
                let backward = super::SILK_RESAMPLER_FRAC_FIR_12[11 - table_index];
                let mut acc = super::smulbb(buf_ptr[0], forward[0]);
                acc = super::smlabb(acc, buf_ptr[1], forward[1]);
                acc = super::smlabb(acc, buf_ptr[2], forward[2]);
                acc = super::smlabb(acc, buf_ptr[3], forward[3]);
                acc = super::smlabb(acc, buf_ptr[4], backward[3]);
                acc = super::smlabb(acc, buf_ptr[5], backward[2]);
                acc = super::smlabb(acc, buf_ptr[6], backward[1]);
                acc = super::smlabb(acc, buf_ptr[7], backward[0]);
                outputs.push(super::sat16(super::rshift_round(acc, 15)));
                index_q16 = index_q16.wrapping_add(state.inv_ratio_q16());
            }

            in_index += n_samples_in;
            remaining -= n_samples_in;
            last_n_samples_in = n_samples_in;

            if remaining > 0 {
                scratch.copy_within(
                    upsampled_len..upsampled_len + super::RESAMPLER_ORDER_FIR_12,
                    0,
                );
            }
        }

        if last_n_samples_in > 0 {
            let tail_offset = last_n_samples_in * 2;
            state.fir_tail_mut().copy_from_slice(
                &scratch[tail_offset..tail_offset + super::RESAMPLER_ORDER_FIR_12],
            );
        }

        outputs
    }
}
