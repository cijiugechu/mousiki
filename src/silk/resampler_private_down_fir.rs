//! Port of the SILK fractional downsampler that combines a second-order AR stage
//! with polyphase FIR interpolation.
//!
//! This mirrors `silk_resampler_private_down_FIR` from
//! `silk/resampler_private_down_FIR.c`. The routine maintains a small IIR state,
//! preserves the tail of the FIR delay line between calls, and emits a stream of
//! decimated 16-bit samples whose spacing is governed by `inv_ratio_q16`.

use alloc::vec;
use alloc::vec::Vec;

use super::resampler_private_ar2::resampler_private_ar2;
use super::resampler_rom::{
    RESAMPLER_DOWN_ORDER_FIR0, RESAMPLER_DOWN_ORDER_FIR1, RESAMPLER_DOWN_ORDER_FIR2,
};

/// Minimal state required by the SILK fractional downsampler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResamplerStateDownFIR<'a> {
    /// Two-element IIR delay line stored in Q18 format.
    pub s_iir: [i32; 2],
    /// FIR delay line holding `fir_order` Q8 samples.
    pub s_fir: Vec<i32>,
    /// Number of input samples processed per inner iteration.
    pub batch_size: usize,
    /// Fixed-point ratio between input and output sample positions (Q16).
    pub inv_ratio_q16: i32,
    /// FIR tap count (must equal one of the `RESAMPLER_DOWN_ORDER_FIR*` constants).
    pub fir_order: usize,
    /// Number of fractional interpolation tables.
    pub fir_fracs: usize,
    /// Concatenated IIR and FIR coefficients sourced from `resampler_rom`.
    pub coefs: &'a [i16],
}

impl<'a> ResamplerStateDownFIR<'a> {
    /// Creates a new downsampler state with zeroed delay elements.
    pub fn new(
        batch_size: usize,
        inv_ratio_q16: i32,
        fir_order: usize,
        fir_fracs: usize,
        coefs: &'a [i16],
    ) -> Self {
        assert!(fir_order % 2 == 0, "FIR order must be even");
        assert!(fir_fracs > 0, "at least one fractional table is required");
        assert!(
            fir_order == RESAMPLER_DOWN_ORDER_FIR0
                || fir_order == RESAMPLER_DOWN_ORDER_FIR1
                || fir_order == RESAMPLER_DOWN_ORDER_FIR2,
            "unexpected FIR order: {}",
            fir_order
        );
        let min_len = 2 + (fir_order / 2) * fir_fracs;
        assert!(
            coefs.len() >= min_len,
            "coefficient slice too short: need at least {} entries",
            min_len
        );

        Self {
            s_iir: [0; 2],
            s_fir: vec![0; fir_order],
            batch_size,
            inv_ratio_q16,
            fir_order,
            fir_fracs,
            coefs,
        }
    }
}

/// Downsamples `input` into `output`, returning the number of produced samples.
///
/// The routine consumes `input` in batches of `batch_size`, filtering each block
/// with the shared AR section and applying polyphase FIR interpolation. Any
/// unconsumed FIR samples are stored back into `state` for the next invocation.
#[allow(clippy::similar_names)]
pub fn resampler_private_down_fir(
    state: &mut ResamplerStateDownFIR<'_>,
    output: &mut [i16],
    input: &[i16],
) -> usize {
    if input.is_empty() {
        return 0;
    }

    let mut buf = vec![0i32; state.batch_size + state.fir_order];
    buf[..state.fir_order].copy_from_slice(&state.s_fir);

    let fir_coefs = &state.coefs[2..];
    let ar_coefs = [state.coefs[0], state.coefs[1]];

    let mut remaining = input.len();
    let mut in_index = 0;
    let mut out_index = 0usize;
    let mut last_n_samples_in = 0usize;

    while remaining > 0 {
        let n_samples_in = remaining.min(state.batch_size);
        let buf_range = state.fir_order..state.fir_order + n_samples_in;
        resampler_private_ar2(
            &mut state.s_iir,
            &mut buf[buf_range.clone()],
            &input[in_index..in_index + n_samples_in],
            &ar_coefs,
        );

        let max_index_q16 = (n_samples_in as i32) << 16;
        out_index += resampler_private_down_fir_interpol(
            &buf,
            fir_coefs,
            state.fir_order,
            state.fir_fracs,
            max_index_q16,
            state.inv_ratio_q16,
            &mut output[out_index..],
        );

        in_index += n_samples_in;
        remaining -= n_samples_in;
        last_n_samples_in = n_samples_in;

        if remaining > 1 {
            for idx in 0..state.fir_order {
                buf[idx] = buf[n_samples_in + idx];
            }
        } else {
            break;
        }
    }

    if last_n_samples_in > 0 {
        state
            .s_fir
            .copy_from_slice(&buf[last_n_samples_in..last_n_samples_in + state.fir_order]);
    }

    out_index
}

fn resampler_private_down_fir_interpol(
    buf: &[i32],
    fir_coefs: &[i16],
    fir_order: usize,
    fir_fracs: usize,
    max_index_q16: i32,
    index_increment_q16: i32,
    output: &mut [i16],
) -> usize {
    let mut out_index = 0usize;
    let mut index_q16 = 0i32;
    let half_order = fir_order / 2;

    while index_q16 < max_index_q16 {
        let base = (index_q16 >> 16) as usize;
        let buf_ptr = &buf[base..base + fir_order];

        let sample = match fir_order {
            RESAMPLER_DOWN_ORDER_FIR0 => {
                let interpol_ind = smulwb(index_q16 & 0xFFFF, fir_fracs as i32) as usize;
                let start = half_order * interpol_ind;
                let forward = &fir_coefs[start..start + half_order];
                let mirror_index = half_order * (fir_fracs - 1 - interpol_ind);
                let backward = &fir_coefs[mirror_index..mirror_index + half_order];

                let mut acc = smulwb(buf_ptr[0], i32::from(forward[0]));
                for k in 1..half_order {
                    acc = smlawb(acc, buf_ptr[k], i32::from(forward[k]));
                }
                for (k, &coef) in backward.iter().enumerate() {
                    let buf_idx = fir_order - 1 - k;
                    acc = smlawb(acc, buf_ptr[buf_idx], i32::from(coef));
                }
                acc
            }
            RESAMPLER_DOWN_ORDER_FIR1 => {
                let mut acc = smulwb(
                    buf_ptr[0].wrapping_add(buf_ptr[fir_order - 1]),
                    i32::from(fir_coefs[0]),
                );
                for k in 1..half_order {
                    let sum = buf_ptr[k].wrapping_add(buf_ptr[fir_order - 1 - k]);
                    acc = smlawb(acc, sum, i32::from(fir_coefs[k]));
                }
                acc
            }
            RESAMPLER_DOWN_ORDER_FIR2 => {
                let mut acc = smulwb(
                    buf_ptr[0].wrapping_add(buf_ptr[fir_order - 1]),
                    i32::from(fir_coefs[0]),
                );
                for k in 1..half_order {
                    let sum = buf_ptr[k].wrapping_add(buf_ptr[fir_order - 1 - k]);
                    acc = smlawb(acc, sum, i32::from(fir_coefs[k]));
                }
                acc
            }
            _ => unreachable!("unsupported FIR order: {fir_order}"),
        };

        assert!(
            out_index < output.len(),
            "output buffer too small: need at least {} samples",
            out_index + 1
        );
        output[out_index] = sat16(rshift_round(sample, 6));
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
fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwb(b, c))
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
        (value.wrapping_add(offset)) >> shift
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::ResamplerStateDownFIR;
    use super::resampler_private_down_fir;
    use crate::silk::resampler_rom::{
        RESAMPLER_DOWN_ORDER_FIR0, RESAMPLER_DOWN_ORDER_FIR1, RESAMPLER_DOWN_ORDER_FIR2,
        SILK_RESAMPLER_1_2_COEFS, SILK_RESAMPLER_1_3_COEFS, SILK_RESAMPLER_3_4_COEFS,
    };

    #[test]
    fn processes_fir0_configuration() {
        let mut state = ResamplerStateDownFIR::new(
            8,
            44_000,
            RESAMPLER_DOWN_ORDER_FIR0,
            3,
            &SILK_RESAMPLER_3_4_COEFS,
        );
        let mut reference = state.clone();
        let input = [1000i16, -500, 750, -250, 125, -60, 30, -10, 5, -2, 1, 0];
        let expected = reference_resampler_private_down_fir(&mut reference, &input);
        let mut output = vec![0i16; expected.len()];
        let produced = resampler_private_down_fir(&mut state, &mut output, &input);
        assert_eq!(produced, expected.len());
        assert_eq!(&output[..produced], expected.as_slice());
        assert_eq!(state.s_fir, reference.s_fir);
        assert_eq!(state.s_iir, reference.s_iir);
    }

    #[test]
    fn processes_fir1_configuration() {
        let mut state = ResamplerStateDownFIR::new(
            12,
            1 << 16,
            RESAMPLER_DOWN_ORDER_FIR1,
            1,
            &SILK_RESAMPLER_1_2_COEFS,
        );
        let mut reference = state.clone();
        let input = [500i16, 400, -300, 200, -100, 50, -25, 12, -6, 3, -1, 0];
        let expected = reference_resampler_private_down_fir(&mut reference, &input);
        let mut output = vec![0i16; expected.len()];
        let produced = resampler_private_down_fir(&mut state, &mut output, &input);
        assert_eq!(produced, expected.len());
        assert_eq!(&output[..produced], expected.as_slice());
        assert_eq!(state.s_fir, reference.s_fir);
        assert_eq!(state.s_iir, reference.s_iir);
    }

    #[test]
    fn processes_fir2_configuration() {
        let mut state = ResamplerStateDownFIR::new(
            10,
            32_768,
            RESAMPLER_DOWN_ORDER_FIR2,
            1,
            &SILK_RESAMPLER_1_3_COEFS,
        );
        let mut reference = state.clone();
        let input = [
            1200i16, -800, 600, -400, 300, -200, 150, -90, 60, -40, 20, -10, 5, -2,
        ];
        let expected = reference_resampler_private_down_fir(&mut reference, &input);
        let mut output = vec![0i16; expected.len()];
        let produced = resampler_private_down_fir(&mut state, &mut output, &input);
        assert_eq!(produced, expected.len());
        assert_eq!(&output[..produced], expected.as_slice());
        assert_eq!(state.s_fir, reference.s_fir);
        assert_eq!(state.s_iir, reference.s_iir);
    }

    fn reference_resampler_private_down_fir(
        state: &mut ResamplerStateDownFIR<'_>,
        input: &[i16],
    ) -> Vec<i16> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut buf = vec![0i32; state.batch_size + state.fir_order];
        buf[..state.fir_order].copy_from_slice(&state.s_fir);
        let fir_coefs = &state.coefs[2..];
        let ar_coefs = [state.coefs[0], state.coefs[1]];

        let mut outputs = Vec::new();
        let mut remaining = input.len();
        let mut in_index = 0usize;
        let mut last_n_samples_in = 0usize;

        while remaining > 0 {
            let n_samples_in = remaining.min(state.batch_size);
            let range = state.fir_order..state.fir_order + n_samples_in;
            super::resampler_private_ar2(
                &mut state.s_iir,
                &mut buf[range.clone()],
                &input[in_index..in_index + n_samples_in],
                &ar_coefs,
            );

            let mut index_q16 = 0i32;
            let max_index_q16 = (n_samples_in as i32) << 16;
            while index_q16 < max_index_q16 {
                let base = (index_q16 >> 16) as usize;
                let buf_ptr = &buf[base..base + state.fir_order];
                let sample = match state.fir_order {
                    RESAMPLER_DOWN_ORDER_FIR0 => {
                        let half = state.fir_order / 2;
                        let interpol_ind =
                            super::smulwb(index_q16 & 0xFFFF, state.fir_fracs as i32) as usize;
                        let forward = &fir_coefs[half * interpol_ind..half * (interpol_ind + 1)];
                        let mirror_index = half * (state.fir_fracs - 1 - interpol_ind);
                        let backward = &fir_coefs[mirror_index..mirror_index + half];
                        let mut acc = super::smulwb(buf_ptr[0], i32::from(forward[0]));
                        for k in 1..half {
                            acc = super::smlawb(acc, buf_ptr[k], i32::from(forward[k]));
                        }
                        for (offset, &coef) in backward.iter().enumerate() {
                            let buf_idx = state.fir_order - 1 - offset;
                            acc = super::smlawb(acc, buf_ptr[buf_idx], i32::from(coef));
                        }
                        acc
                    }
                    RESAMPLER_DOWN_ORDER_FIR1 | RESAMPLER_DOWN_ORDER_FIR2 => {
                        let half = state.fir_order / 2;
                        let mut acc = super::smulwb(
                            buf_ptr[0].wrapping_add(buf_ptr[state.fir_order - 1]),
                            i32::from(fir_coefs[0]),
                        );
                        for k in 1..half {
                            let sum = buf_ptr[k].wrapping_add(buf_ptr[state.fir_order - 1 - k]);
                            acc = super::smlawb(acc, sum, i32::from(fir_coefs[k]));
                        }
                        acc
                    }
                    _ => unreachable!("unexpected FIR order"),
                };
                outputs.push(super::sat16(super::rshift_round(sample, 6)));
                index_q16 = index_q16.wrapping_add(state.inv_ratio_q16);
            }

            in_index += n_samples_in;
            remaining -= n_samples_in;
            last_n_samples_in = n_samples_in;

            if remaining > 1 {
                for idx in 0..state.fir_order {
                    buf[idx] = buf[n_samples_in + idx];
                }
            } else {
                break;
            }
        }

        if last_n_samples_in > 0 {
            state
                .s_fir
                .copy_from_slice(&buf[last_n_samples_in..last_n_samples_in + state.fir_order]);
        }

        outputs
    }
}
