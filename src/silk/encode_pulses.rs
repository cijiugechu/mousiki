//! Port of `silk/encode_pulses.c`.
//!
//! This module mirrors the SILK encoder helper that entropy-codes the
//! shell-block pulse magnitudes, their refinement bits, and the associated
//! sign information.  The implementation stays close to the fixed-point C
//! reference so that the range-coder stream matches bit-for-bit.

use core::{convert::TryFrom, slice};

use crate::range::RangeEncoder;
use crate::silk::MAX_NB_SUBFR;
use crate::silk::code_signs::silk_encode_signs;
use crate::silk::shell_coder::silk_shell_encoder;
use crate::silk::tables_other::SILK_LSB_ICDF;
use crate::silk::tables_pulses_per_block::{
    SILK_MAX_PULSES_TABLE, SILK_PULSES_PER_BLOCK_BITS_Q5, SILK_PULSES_PER_BLOCK_ICDF,
    SILK_RATE_LEVELS_BITS_Q5, SILK_RATE_LEVELS_ICDF,
};

const SHELL_CODEC_FRAME_LENGTH: usize = 16;
const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
const SILK_MAX_PULSES: usize = 16;
const N_RATE_LEVELS: usize = 10;
const MAX_LSB_COUNT: usize = 10;
const SUB_FRAME_LENGTH_MS: usize = 5;
const MAX_FS_KHZ: usize = 16;
const MAX_FRAME_LENGTH_MS: usize = SUB_FRAME_LENGTH_MS * MAX_NB_SUBFR;
const MAX_FRAME_LENGTH: usize = MAX_FRAME_LENGTH_MS * MAX_FS_KHZ;
const MAX_PADDED_FRAME_LENGTH: usize = MAX_FRAME_LENGTH + SHELL_CODEC_FRAME_LENGTH;
const MAX_SHELL_BLOCKS: usize = MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH;
const TWELVE_KHZ_10_MS_FRAME: usize = 12 * 10;

#[derive(Default)]
struct CombineScratch {
    level8: [i32; SHELL_CODEC_FRAME_LENGTH / 2],
    level4: [i32; SHELL_CODEC_FRAME_LENGTH / 4],
    level2: [i32; SHELL_CODEC_FRAME_LENGTH / 8],
}

fn combine_and_check(output: &mut [i32], input: &[i32], max_pulses: i32) -> bool {
    debug_assert_eq!(input.len(), output.len() * 2);

    for (index, value) in output.iter_mut().enumerate() {
        let sum = input[2 * index] + input[2 * index + 1];
        if sum > max_pulses {
            return true;
        }
        *value = sum;
    }

    false
}

fn number_of_shell_blocks(frame_length: usize) -> usize {
    let mut blocks = frame_length >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if blocks * SHELL_CODEC_FRAME_LENGTH < frame_length {
        debug_assert_eq!(frame_length, TWELVE_KHZ_10_MS_FRAME);
        blocks += 1;
    }
    blocks
}

/// Encodes the quantised excitation pulses into the provided range encoder.
pub fn silk_encode_pulses(
    encoder: &mut RangeEncoder,
    signal_type: i32,
    quant_offset_type: i32,
    pulses: &mut [i8],
    frame_length: usize,
) {
    assert!(frame_length > 0, "frame_length must be positive");
    assert!(
        frame_length <= pulses.len(),
        "pulse buffer shorter than frame length"
    );
    assert!(
        frame_length <= MAX_FRAME_LENGTH,
        "frame length {frame_length} exceeds MAX_FRAME_LENGTH {MAX_FRAME_LENGTH}"
    );

    let num_shell_blocks = number_of_shell_blocks(frame_length);
    debug_assert!(num_shell_blocks > 0);
    assert!(
        num_shell_blocks <= MAX_SHELL_BLOCKS,
        "too many shell blocks: {num_shell_blocks}"
    );
    debug_assert_eq!(1 << LOG2_SHELL_CODEC_FRAME_LENGTH, SHELL_CODEC_FRAME_LENGTH);

    let padded_frame_length = num_shell_blocks * SHELL_CODEC_FRAME_LENGTH;
    debug_assert!(padded_frame_length <= MAX_PADDED_FRAME_LENGTH);

    let mut padded = [0i8; MAX_PADDED_FRAME_LENGTH];
    padded[..frame_length].copy_from_slice(&pulses[..frame_length]);

    let mut abs_pulses = [0i32; MAX_PADDED_FRAME_LENGTH];
    for index in 0..padded_frame_length {
        abs_pulses[index] = i32::from(padded[index]).abs();
    }

    let mut sum_pulses = [0i32; MAX_SHELL_BLOCKS];
    let mut n_rshifts = [0u8; MAX_SHELL_BLOCKS];

    for block in 0..num_shell_blocks {
        let mut scratch = CombineScratch::default();
        let block_start = block * SHELL_CODEC_FRAME_LENGTH;
        let block_end = block_start + SHELL_CODEC_FRAME_LENGTH;
        let block_slice = &mut abs_pulses[block_start..block_end];

        loop {
            let mut scale_down = combine_and_check(
                &mut scratch.level8,
                block_slice,
                i32::from(SILK_MAX_PULSES_TABLE[0]),
            );
            scale_down |= combine_and_check(
                &mut scratch.level4,
                &scratch.level8,
                i32::from(SILK_MAX_PULSES_TABLE[1]),
            );
            scale_down |= combine_and_check(
                &mut scratch.level2,
                &scratch.level4,
                i32::from(SILK_MAX_PULSES_TABLE[2]),
            );
            scale_down |= combine_and_check(
                slice::from_mut(&mut sum_pulses[block]),
                &scratch.level2,
                i32::from(SILK_MAX_PULSES_TABLE[3]),
            );

            if scale_down {
                n_rshifts[block] = n_rshifts[block].saturating_add(1);
                debug_assert!(
                    usize::from(n_rshifts[block]) <= MAX_LSB_COUNT,
                    "more than {MAX_LSB_COUNT} LSB passes for shell block {block}"
                );
                for value in block_slice.iter_mut() {
                    *value >>= 1;
                }
            } else {
                break;
            }
        }
    }

    let signal_index = (signal_type >> 1) as usize;
    assert!(
        signal_index < SILK_RATE_LEVELS_BITS_Q5.len(),
        "invalid signal type {signal_type}"
    );

    let mut min_sum_bits_q5 = i32::MAX;
    let mut rate_level_index = 0usize;

    for level in 0..(N_RATE_LEVELS - 1) {
        let mut sum_bits_q5 = i32::from(SILK_RATE_LEVELS_BITS_Q5[signal_index][level]);
        let bits_table = &SILK_PULSES_PER_BLOCK_BITS_Q5[level];

        for block in 0..num_shell_blocks {
            let symbol = if n_rshifts[block] == 0 {
                usize::try_from(sum_pulses[block]).expect("sum_pulses negative")
            } else {
                SILK_MAX_PULSES + 1
            };
            sum_bits_q5 += i32::from(bits_table[symbol]);
        }

        if sum_bits_q5 < min_sum_bits_q5 {
            min_sum_bits_q5 = sum_bits_q5;
            rate_level_index = level;
        }
    }

    encoder.encode_icdf(rate_level_index, &SILK_RATE_LEVELS_ICDF[signal_index], 8);

    let base_cdf = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    let escape_cdf = &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1];

    for (shift_count, sum) in n_rshifts
        .iter()
        .take(num_shell_blocks)
        .zip(sum_pulses.iter())
    {
        if *shift_count == 0 {
            let symbol = usize::try_from(*sum).expect("sum_pulses negative");
            encoder.encode_icdf(symbol, base_cdf, 8);
        } else {
            encoder.encode_icdf(SILK_MAX_PULSES + 1, base_cdf, 8);
            let extra_shifts = usize::from(*shift_count) - 1;
            for _ in 0..extra_shifts {
                encoder.encode_icdf(SILK_MAX_PULSES + 1, escape_cdf, 8);
            }
            let symbol = usize::try_from(*sum).expect("sum_pulses negative");
            encoder.encode_icdf(symbol, escape_cdf, 8);
        }
    }

    for (sum, block_slice) in sum_pulses
        .iter()
        .take(num_shell_blocks)
        .zip(abs_pulses[..padded_frame_length].chunks_exact(SHELL_CODEC_FRAME_LENGTH))
    {
        if *sum > 0 {
            silk_shell_encoder(encoder, block_slice);
        }
    }

    for (shift_count, block_slice) in n_rshifts
        .iter()
        .take(num_shell_blocks)
        .zip(padded[..padded_frame_length].chunks_exact(SHELL_CODEC_FRAME_LENGTH))
    {
        let shift_count = usize::from(*shift_count);
        if shift_count > 0 {
            let n_ls = shift_count - 1;
            for &pulse in block_slice {
                let abs_q = i32::from(pulse).abs();
                if n_ls > 0 {
                    for shift in (1..=n_ls).rev() {
                        let bit = usize::try_from((abs_q >> shift) & 1).expect("bit extraction");
                        encoder.encode_icdf(bit, &SILK_LSB_ICDF, 8);
                    }
                }
                let bit = usize::try_from(abs_q & 1).expect("bit extraction");
                encoder.encode_icdf(bit, &SILK_LSB_ICDF, 8);
            }
        }
    }

    silk_encode_signs(
        encoder,
        &pulses[..frame_length],
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses[..num_shell_blocks],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    use crate::range::RangeDecoder;
    use crate::silk::code_signs::silk_decode_signs;
    use crate::silk::shell_coder::silk_shell_decoder;

    fn decode_reference(
        encoded: &[u8],
        signal_type: i32,
        quant_offset_type: i32,
        frame_length: usize,
    ) -> Vec<i8> {
        let mut decoder = RangeDecoder::init(encoded);
        let signal_index = (signal_type >> 1) as usize;
        assert!(signal_index < SILK_RATE_LEVELS_ICDF.len());
        let rate_level_index =
            decoder.decode_icdf(&SILK_RATE_LEVELS_ICDF[signal_index], 8) as usize;

        let num_shell_blocks = number_of_shell_blocks(frame_length);
        let mut sum_pulses = vec![0i32; num_shell_blocks];
        let mut n_lshifts = vec![0u8; num_shell_blocks];

        let base_cdf = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
        for block in 0..num_shell_blocks {
            let mut value = i32::try_from(decoder.decode_icdf(base_cdf, 8))
                .expect("rate-level symbol overflow");
            while value == i32::try_from(SILK_MAX_PULSES + 1).unwrap() {
                n_lshifts[block] = n_lshifts[block].saturating_add(1);
                let escape_cdf = &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1];
                let disallow_escape = usize::from(n_lshifts[block]) == MAX_LSB_COUNT;
                let tail = if disallow_escape {
                    &escape_cdf[1..]
                } else {
                    escape_cdf
                };
                value =
                    i32::try_from(decoder.decode_icdf(tail, 8)).expect("escape symbol overflow");
            }
            sum_pulses[block] = value;
        }

        let padded_frame_length = num_shell_blocks * SHELL_CODEC_FRAME_LENGTH;
        let mut pulses = vec![0i16; padded_frame_length];

        for block in 0..num_shell_blocks {
            let block_start = block * SHELL_CODEC_FRAME_LENGTH;
            let block_slice = &mut pulses[block_start..block_start + SHELL_CODEC_FRAME_LENGTH];
            if sum_pulses[block] > 0 {
                silk_shell_decoder(block_slice, &mut decoder, sum_pulses[block]);
            } else {
                block_slice.fill(0);
            }
        }

        for block in 0..num_shell_blocks {
            let n_ls = usize::from(n_lshifts[block]);
            if n_ls > 0 {
                let block_start = block * SHELL_CODEC_FRAME_LENGTH;
                let block_slice = &mut pulses[block_start..block_start + SHELL_CODEC_FRAME_LENGTH];
                for value in block_slice.iter_mut() {
                    let mut abs_q = i32::from(*value);
                    for _ in 0..n_ls {
                        abs_q = (abs_q << 1)
                            + i32::try_from(decoder.decode_icdf(&SILK_LSB_ICDF, 8))
                                .expect("LSB symbol overflow");
                    }
                    *value = abs_q as i16;
                }
                sum_pulses[block] |= i32::from(n_lshifts[block]) << 5;
            }
        }

        silk_decode_signs(
            &mut decoder,
            &mut pulses,
            frame_length,
            signal_type,
            quant_offset_type,
            &sum_pulses,
        );

        pulses[..frame_length]
            .iter()
            .map(|&value| i8::try_from(value).expect("pulse outside i8 range"))
            .collect()
    }

    fn sample_pulses(frame_length: usize) -> Vec<i8> {
        (0..frame_length)
            .map(|i| {
                let pattern = ((i * 7 + 3) % 21) as i32 - 10;
                if (i & 1) == 0 {
                    pattern as i8
                } else {
                    (-pattern).clamp(-20, 20) as i8
                }
            })
            .collect()
    }

    fn round_trip_case(frame_length: usize, signal_type: i32, quant_offset_type: i32) {
        let pulses = sample_pulses(frame_length);
        let mut work = pulses.clone();
        let mut encoder = RangeEncoder::new();

        silk_encode_pulses(
            &mut encoder,
            signal_type,
            quant_offset_type,
            &mut work,
            frame_length,
        );

        let encoded = encoder.finish();
        let decoded = decode_reference(&encoded, signal_type, quant_offset_type, frame_length);
        assert_eq!(decoded, pulses);
    }

    #[test]
    fn encode_decode_roundtrip_unvoiced() {
        round_trip_case(160, 0, 0);
    }

    #[test]
    fn encode_decode_roundtrip_partial_block() {
        round_trip_case(TWELVE_KHZ_10_MS_FRAME, 2, 1);
    }
}
