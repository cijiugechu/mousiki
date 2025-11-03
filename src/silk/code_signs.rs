//! Port of the SILK excitation sign coder from `silk/code_signs.c`.
//!
//! These helpers attach or extract the sign bits associated with
//! shell-coded excitation pulses. They operate alongside the shell
//! coder (`silk/shell_coder.c`) and reuse the shared sign entropy
//! tables exposed in `tables_pulses_per_block`.

use crate::range::{RangeDecoder, RangeEncoder};
use crate::silk::tables_pulses_per_block::SILK_SIGN_ICDF;

const SHELL_CODEC_FRAME_LENGTH: usize = 16;
const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
#[inline]
fn silk_enc_map(value: i16) -> usize {
    ((i32::from(value) >> 15) + 1) as usize
}

#[inline]
fn silk_dec_map(symbol: usize) -> i16 {
    debug_assert!(symbol <= 1);
    ((symbol as i16) << 1) - 1
}

#[inline]
fn sign_icdf_base(signal_type: i32, quant_offset_type: i32) -> usize {
    let index = 7 * (quant_offset_type + (signal_type << 1));
    debug_assert!(
        (0..=(SILK_SIGN_ICDF.len() as i32 - 7)).contains(&index),
        "invalid range-coder context for pulse signs"
    );
    index as usize
}

fn number_of_shell_blocks(frame_length: usize) -> usize {
    (frame_length + (SHELL_CODEC_FRAME_LENGTH / 2)) >> LOG2_SHELL_CODEC_FRAME_LENGTH
}

/// Encodes the pulse sign bits into the provided range encoder.
pub fn silk_encode_signs(
    encoder: &mut RangeEncoder,
    pulses: &[i8],
    frame_length: usize,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    assert!(
        frame_length <= pulses.len(),
        "pulse buffer shorter than frame length"
    );

    let num_blocks = number_of_shell_blocks(frame_length);
    assert!(
        sum_pulses.len() >= num_blocks,
        "sum_pulses slice shorter than required shell blocks"
    );

    let mut icdf = [0u8; 2];
    icdf[1] = 0;

    let icdf_ptr = &SILK_SIGN_ICDF[sign_icdf_base(signal_type, quant_offset_type)..];
    let mut pulse_index = 0usize;

    for &total in sum_pulses.iter().take(num_blocks) {
        if total > 0 {
            let table_index = ((total & 0x1F) as usize).min(6);
            icdf[0] = icdf_ptr[table_index];

            let block_end = (pulse_index + SHELL_CODEC_FRAME_LENGTH).min(frame_length);
            for &pulse in &pulses[pulse_index..block_end] {
                if pulse != 0 {
                    let symbol = silk_enc_map(pulse.into());
                    encoder.encode_icdf(symbol, &icdf, 8);
                }
            }
        }

        pulse_index += SHELL_CODEC_FRAME_LENGTH;
    }
}

/// Decodes sign bits and applies them in-place to the absolute pulse magnitudes.
pub fn silk_decode_signs(
    decoder: &mut RangeDecoder<'_>,
    pulses: &mut [i16],
    frame_length: usize,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    assert!(
        frame_length <= pulses.len(),
        "pulse buffer shorter than frame length"
    );

    let num_blocks = number_of_shell_blocks(frame_length);
    assert!(
        sum_pulses.len() >= num_blocks,
        "sum_pulses slice shorter than required shell blocks"
    );

    let mut icdf = [0u8; 2];
    icdf[1] = 0;

    let icdf_ptr = &SILK_SIGN_ICDF[sign_icdf_base(signal_type, quant_offset_type)..];
    let mut pulse_index = 0usize;

    for &total in sum_pulses.iter().take(num_blocks) {
        if total > 0 {
            let table_index = ((total & 0x1F) as usize).min(6);
            icdf[0] = icdf_ptr[table_index];

            let block_end = (pulse_index + SHELL_CODEC_FRAME_LENGTH).min(frame_length);
            for pulse in &mut pulses[pulse_index..block_end] {
                if *pulse > 0 {
                    let symbol = decoder.decode_icdf(&icdf, 8);
                    *pulse *= silk_dec_map(symbol);
                }
            }
        }

        pulse_index += SHELL_CODEC_FRAME_LENGTH;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    fn sums_for_blocks(pulses: &[i8]) -> Vec<i32> {
        let num_blocks = number_of_shell_blocks(pulses.len());
        let mut sums = vec![0i32; num_blocks];

        for (block_index, chunk) in pulses
            .chunks(SHELL_CODEC_FRAME_LENGTH)
            .take(num_blocks)
            .enumerate()
        {
            let sum = chunk.iter().map(|value| i32::from(value.abs())).sum();
            sums[block_index] = sum;
        }

        sums
    }

    #[test]
    fn encode_decode_roundtrip() {
        let frame_length = 32;
        let pulses = [
            3, -1, 0, 2, -2, 0, 1, -4, 0, 0, 2, -1, 0, 1, 0, -1, //
            -2, 1, 0, -1, 3, 0, 0, -2, 0, 1, 0, 2, -1, 0, 1, 0,
        ];
        let sums = sums_for_blocks(&pulses);
        let signal_type = 2; // voiced
        let quant_offset_type = 0;

        let mut encoder = RangeEncoder::new();
        silk_encode_signs(
            &mut encoder,
            &pulses,
            frame_length,
            signal_type,
            quant_offset_type,
            &sums,
        );
        let encoded = encoder.finish();

        let mut magnitudes: Vec<i16> = pulses.iter().map(|&value| i16::from(value.abs())).collect();
        let mut decoder = RangeDecoder::init(&encoded);
        silk_decode_signs(
            &mut decoder,
            &mut magnitudes,
            frame_length,
            signal_type,
            quant_offset_type,
            &sums,
        );

        let reconstructed: Vec<i8> = magnitudes.iter().map(|&value| value as i8).collect();
        assert_eq!(reconstructed, pulses);
    }

    #[test]
    fn zero_sum_blocks_emit_no_bits() {
        let frame_length = SHELL_CODEC_FRAME_LENGTH;
        let pulses = [0i8; SHELL_CODEC_FRAME_LENGTH];
        let sums = vec![0];
        let signal_type = 0;
        let quant_offset_type = 1;

        let mut encoder = RangeEncoder::new();
        silk_encode_signs(
            &mut encoder,
            &pulses,
            frame_length,
            signal_type,
            quant_offset_type,
            &sums,
        );
        let encoded = encoder.finish();
        assert!(encoded.is_empty());

        let mut magnitudes = [0i16; SHELL_CODEC_FRAME_LENGTH];
        let mut decoder = RangeDecoder::init(&encoded);
        silk_decode_signs(
            &mut decoder,
            &mut magnitudes,
            frame_length,
            signal_type,
            quant_offset_type,
            &sums,
        );
        assert!(magnitudes.iter().all(|&value| value == 0));
    }
}
