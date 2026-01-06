//! Port of `silk/decode_pulses.c`.
//!
//! Mirrors the SILK helper that rebuilds the shell-coded excitation pulses
//! from the range decoder, including the rate-level selection, per-block
//! escape handling, LSB refinement, and sign decoding stages.

use core::convert::TryFrom;

use crate::silk::MAX_NB_SUBFR;
use crate::silk::SilkRangeDecoder;
use crate::silk::code_signs::silk_decode_signs;
use crate::silk::shell_coder::silk_shell_decoder;
use crate::silk::tables_other::SILK_LSB_ICDF;
use crate::silk::tables_pulses_per_block::{SILK_PULSES_PER_BLOCK_ICDF, SILK_RATE_LEVELS_ICDF};

const SHELL_CODEC_FRAME_LENGTH: usize = 16;
const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
const SILK_MAX_PULSES: usize = 16;
const ESCAPE_PULSES: usize = SILK_MAX_PULSES + 1;
const N_RATE_LEVELS: usize = 10;
const MAX_LSB_COUNT: usize = 10;
const SUB_FRAME_LENGTH_MS: usize = 5;
const MAX_FS_KHZ: usize = 16;
const MAX_FRAME_LENGTH_MS: usize = SUB_FRAME_LENGTH_MS * MAX_NB_SUBFR;
const MAX_FRAME_LENGTH: usize = MAX_FRAME_LENGTH_MS * MAX_FS_KHZ;
const MAX_PADDED_FRAME_LENGTH: usize = MAX_FRAME_LENGTH + SHELL_CODEC_FRAME_LENGTH;
const MAX_SHELL_BLOCKS: usize = MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH;
const TWELVE_KHZ_10_MS_FRAME: usize = 12 * 10;

fn number_of_shell_blocks(frame_length: usize) -> usize {
    let mut blocks = frame_length >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if blocks * SHELL_CODEC_FRAME_LENGTH < frame_length {
        debug_assert_eq!(
            frame_length, TWELVE_KHZ_10_MS_FRAME,
            "only the 10 ms @ 12 kHz frame requires padding"
        );
        blocks += 1;
    }
    blocks
}

fn padded_frame_length(frame_length: usize) -> usize {
    number_of_shell_blocks(frame_length) * SHELL_CODEC_FRAME_LENGTH
}

/// Mirrors `silk_decode_pulses` from the reference C sources.
pub fn silk_decode_pulses(
    decoder: &mut impl SilkRangeDecoder,
    pulses: &mut [i16],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: usize,
) {
    assert!(frame_length > 0, "frame length must be positive");
    assert!(
        frame_length <= MAX_FRAME_LENGTH,
        "frame length {frame_length} exceeds MAX_FRAME_LENGTH {MAX_FRAME_LENGTH}"
    );
    assert!(
        quant_offset_type == 0 || quant_offset_type == 1,
        "invalid quantization offset type {quant_offset_type}"
    );

    let num_shell_blocks = number_of_shell_blocks(frame_length);
    assert!(num_shell_blocks > 0, "at least one shell block is required");
    assert!(
        num_shell_blocks <= MAX_SHELL_BLOCKS,
        "too many shell blocks: {num_shell_blocks}"
    );

    let padded_length = padded_frame_length(frame_length);
    assert!(
        padded_length <= pulses.len(),
        "pulse buffer shorter than padded frame length {padded_length}"
    );
    assert!(
        padded_length <= MAX_PADDED_FRAME_LENGTH,
        "padded frame length {padded_length} exceeds MAX_PADDED_FRAME_LENGTH {MAX_PADDED_FRAME_LENGTH}"
    );

    let signal_index = ((signal_type >> 1).clamp(0, 1)) as usize;
    let rate_level_index = decoder.decode_icdf(&SILK_RATE_LEVELS_ICDF[signal_index], 8);
    let rate_level_index = rate_level_index.min(N_RATE_LEVELS - 1);

    let base_cdf = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    let escape_cdf = &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1];

    let mut sum_pulses = [0i32; MAX_SHELL_BLOCKS];
    let mut n_lshifts = [0u8; MAX_SHELL_BLOCKS];

    for (shift_count, sum_slot) in n_lshifts
        .iter_mut()
        .zip(sum_pulses.iter_mut())
        .take(num_shell_blocks)
    {
        let mut sum = decoder.decode_icdf(base_cdf, 8);
        while sum == ESCAPE_PULSES {
            *shift_count = shift_count.saturating_add(1);
            let escape_slice = if usize::from(*shift_count) == MAX_LSB_COUNT {
                &escape_cdf[1..]
            } else {
                escape_cdf
            };
            sum = decoder.decode_icdf(escape_slice, 8);
        }
        *sum_slot = sum as i32;
    }

    {
        let mut block_chunks = pulses[..padded_length].chunks_exact_mut(SHELL_CODEC_FRAME_LENGTH);
        for (sum, block_slice) in sum_pulses
            .iter()
            .take(num_shell_blocks)
            .zip(block_chunks.by_ref())
        {
            if *sum > 0 {
                silk_shell_decoder(block_slice, decoder, *sum);
            } else {
                block_slice.fill(0);
            }
        }
    }

    {
        let block_chunks = pulses[..padded_length].chunks_exact_mut(SHELL_CODEC_FRAME_LENGTH);
        for ((sum, shift_count), block_slice) in sum_pulses
            .iter_mut()
            .zip(n_lshifts.iter())
            .take(num_shell_blocks)
            .zip(block_chunks)
        {
            if *shift_count == 0 {
                continue;
            }

            for sample in block_slice.iter_mut() {
                let mut abs_q = i32::from(*sample);
                for _ in 0..*shift_count {
                    abs_q += abs_q;
                    abs_q += i32::try_from(decoder.decode_icdf(&SILK_LSB_ICDF, 8))
                        .expect("decoded LSB exceeds i32 range");
                }
                *sample = i16::try_from(abs_q).expect("pulse magnitude exceeds i16 range");
            }

            *sum |= i32::from(*shift_count) << 5;
        }
    }

    silk_decode_signs(
        decoder,
        pulses,
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

    use crate::celt::EcDec;
    use crate::range::RangeEncoder;
    use crate::silk::encode_pulses::silk_encode_pulses;

    fn reference_pulses(frame_length: usize) -> Vec<i8> {
        let mut pulses = vec![0i8; frame_length];
        for (index, value) in pulses.iter_mut().enumerate() {
            *value = match index % 5 {
                0 => 3,
                1 => -2,
                2 => 0,
                3 => 1,
                _ => -1,
            };
        }
        pulses
    }

    fn decode_round_trip(frame_length: usize, signal_type: i32, quant_offset_type: i32) {
        let original = reference_pulses(frame_length);
        let mut encoder = RangeEncoder::new();
        let mut encoder_pulses = original.clone();
        silk_encode_pulses(
            &mut encoder,
            signal_type,
            quant_offset_type,
            &mut encoder_pulses,
            frame_length,
        );
        let mut payload = encoder.finish();

        let mut decoder = EcDec::new(payload.as_mut_slice());
        let padded_length = padded_frame_length(frame_length);
        let mut decoded = vec![0i16; padded_length];
        silk_decode_pulses(
            &mut decoder,
            &mut decoded,
            signal_type,
            quant_offset_type,
            frame_length,
        );

        for (expected, actual) in original.iter().zip(decoded.iter().take(frame_length)) {
            assert_eq!(*expected as i16, *actual);
        }
    }

    #[test]
    fn round_trip_voiced_frame() {
        decode_round_trip(320, 2, 0);
    }

    #[test]
    fn round_trip_12khz_10ms_frame() {
        decode_round_trip(TWELVE_KHZ_10_MS_FRAME, 0, 1);
    }
}
