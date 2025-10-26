//! Port of the SILK low-quality 2/3 downsampler.
//!
//! This mirrors `silk_resampler_down2_3` from `silk/resampler_down2_3.c`, which chains
//! the shared second-order AR section with a short FIR interpolator to decimate a
//! 16-bit input stream by a factor of 2/3. The routine keeps four Q8 samples of FIR
//! history alongside the two-element AR state and processes input in small batches to
//! avoid unbounded stack allocation in the reference C implementation.

use alloc::vec;

use core::convert::TryInto;

use super::resampler_private_ar2::resampler_private_ar2;
use super::resampler_rom::SILK_RESAMPLER_2_3_COEFS_LQ;

const ORDER_FIR: usize = 4;
const RESAMPLER_MAX_BATCH_SIZE_IN: usize = 480;

/// Downsamples `input` by a factor of 2/3, returning the number of produced samples.
///
/// The function consumes the provided samples in batches of up to 480 frames, updating
/// `state` in-place and writing the decimated result into `output`. Any trailing input
/// samples that do not form a complete triplet are preserved in the state and folded
/// into the next call.
///
/// # Panics
///
/// * If `output.len()` is smaller than `2 * input.len() / 3`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::similar_names
)]
pub fn resampler_down2_3(
    state: &mut [i32; ORDER_FIR + 2],
    output: &mut [i16],
    input: &[i16],
) -> usize {
    if input.is_empty() {
        return 0;
    }

    let required = (2 * input.len()) / 3;
    assert!(
        output.len() >= required,
        "output buffer too small: need {} samples",
        required
    );

    let (fir_state, ar_state_slice) = state.split_at_mut(ORDER_FIR);
    let ar_state: &mut [i32; 2] = ar_state_slice
        .try_into()
        .expect("state must contain four FIR taps and two AR elements");

    let ar_coefs = [
        SILK_RESAMPLER_2_3_COEFS_LQ[0],
        SILK_RESAMPLER_2_3_COEFS_LQ[1],
    ];
    let fir0 = i32::from(SILK_RESAMPLER_2_3_COEFS_LQ[2]);
    let fir1 = i32::from(SILK_RESAMPLER_2_3_COEFS_LQ[3]);
    let fir2 = i32::from(SILK_RESAMPLER_2_3_COEFS_LQ[4]);
    let fir3 = i32::from(SILK_RESAMPLER_2_3_COEFS_LQ[5]);

    let mut buf = vec![0i32; RESAMPLER_MAX_BATCH_SIZE_IN + ORDER_FIR];
    buf[..ORDER_FIR].copy_from_slice(fir_state);

    let mut produced = 0usize;
    let mut processed = 0usize;
    let mut last_block_len = 0usize;

    while processed < input.len() {
        let remaining = input.len() - processed;
        let block_len = remaining.min(RESAMPLER_MAX_BATCH_SIZE_IN);
        let end = processed + block_len;
        last_block_len = block_len;

        let buf_len = ORDER_FIR + block_len;
        resampler_private_ar2(
            ar_state,
            &mut buf[ORDER_FIR..buf_len],
            &input[processed..end],
            &ar_coefs,
        );

        let mut buf_idx = 0usize;
        let mut counter = block_len;
        while counter > 2 {
            let mut res_q6 = smulwb(buf[buf_idx], fir0);
            res_q6 = smlawb(res_q6, buf[buf_idx + 1], fir1);
            res_q6 = smlawb(res_q6, buf[buf_idx + 2], fir3);
            res_q6 = smlawb(res_q6, buf[buf_idx + 3], fir2);
            output[produced] = sat16(rshift_round(res_q6, 6));
            produced += 1;

            let mut res_q6 = smulwb(buf[buf_idx + 1], fir2);
            res_q6 = smlawb(res_q6, buf[buf_idx + 2], fir3);
            res_q6 = smlawb(res_q6, buf[buf_idx + 3], fir1);
            res_q6 = smlawb(res_q6, buf[buf_idx + 4], fir0);
            output[produced] = sat16(rshift_round(res_q6, 6));
            produced += 1;

            buf_idx += 3;
            counter -= 3;
        }

        processed = end;

        if processed < input.len() {
            buf.copy_within(block_len..block_len + ORDER_FIR, 0);
        }
    }

    fir_state.copy_from_slice(&buf[last_block_len..last_block_len + ORDER_FIR]);

    produced
}

#[inline]
fn smlawb(a: i32, b: i32, coef_q15: i32) -> i32 {
    let product = i64::from(b) * i64::from(coef_q15 as i16);
    a.wrapping_add((product >> 16) as i32)
}

#[inline]
fn smulwb(a: i32, coef_q15: i32) -> i32 {
    let product = i64::from(a) * i64::from(coef_q15 as i16);
    (product >> 16) as i32
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
    assert!(shift > 0);
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

#[cfg(test)]
mod tests {
    use super::resampler_down2_3;

    #[test]
    fn handles_zero_input() {
        let mut state = [0i32; 6];
        let mut output = [0i16; 8];
        let produced = resampler_down2_3(&mut state, &mut output, &[]);
        assert_eq!(produced, 0);
        assert_eq!(output, [0; 8]);
        assert_eq!(state, [0; 6]);
    }

    #[test]
    fn processes_basic_sequence() {
        let mut state = [0i32; 6];
        let input = [1000i16, -1000, 2000, -2000, 1500, -1500];
        let mut output = [0i16; 4];
        let produced = resampler_down2_3(&mut state, &mut output, &input);
        assert_eq!(produced, 4);
        assert_eq!(output, [0, 287, 236, 158]);
        assert_eq!(state[..4], [461_492, -471_755, 281_250, -244_654]);
    }

    #[test]
    fn preserves_state_between_calls() {
        let mut state = [12_345i32, -54_321, 98_765, -12_345, 4_321, -9_876];
        let input = [
            25_340i16, -4_753, 19_673, 28_343, -2_438, -27_347, -13_032, 3_506, 1_845, -3_463,
        ];
        let mut output = [0i16; 6];
        let produced = resampler_down2_3(&mut state, &mut output, &input);
        assert_eq!(produced, 6);
        assert_eq!(output, [65, 7_412, 13_067, 13_750, 13_278, -28_142]);
        assert_eq!(
            state,
            [-488_595, 4_766_869, -147_410, -2_754_553, 528_788, 1_093_986]
        );
    }
}
