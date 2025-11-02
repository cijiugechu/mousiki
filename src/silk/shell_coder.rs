//! Port of the SILK shell coder used to encode and decode 16-sample pulse blocks.
//!
//! Mirrors the fixed-point routines from `silk/shell_coder.c`, which build a
//! binary tree over non-negative pulse amplitudes. Each split is entropy coded
//! using tables from `tables_pulses_per_block`.

use crate::range::{RangeDecoder, RangeEncoder};
use crate::silk::tables_pulses_per_block::{
    SILK_SHELL_CODE_TABLE0, SILK_SHELL_CODE_TABLE1, SILK_SHELL_CODE_TABLE2,
    SILK_SHELL_CODE_TABLE3, SILK_SHELL_CODE_TABLE_OFFSETS,
};

const SHELL_CODEC_FRAME_LENGTH: usize = 16;

fn combine_pulses(output: &mut [i32], input: &[i32]) {
    debug_assert_eq!(input.len(), output.len() * 2);

    for (k, value) in output.iter_mut().enumerate() {
        *value = input[2 * k] + input[2 * k + 1];
    }
}

fn shell_table_slice(table: &[u8], pulses: usize) -> &[u8] {
    let start = usize::from(SILK_SHELL_CODE_TABLE_OFFSETS[pulses]);
    let length = pulses + 1;
    debug_assert!(start + length <= table.len());
    &table[start..start + length]
}

fn encode_split(encoder: &mut RangeEncoder, first_child: i32, total: i32, table: &[u8]) {
    if total > 0 {
        debug_assert!(first_child >= 0 && first_child <= total);
        let slice = shell_table_slice(table, total as usize);
        encoder.encode_icdf(first_child as usize, slice, 8);
    }
}

fn decode_split(decoder: &mut RangeDecoder<'_>, total: i32, table: &[u8]) -> (i32, i32) {
    if total > 0 {
        let slice = shell_table_slice(table, total as usize);
        let first = decoder.decode_icdf(slice, 8) as i32;
        (first, total - first)
    } else {
        (0, 0)
    }
}

/// Shell encoder, operates on one shell code frame of 16 pulses.
pub fn silk_shell_encoder(encoder: &mut RangeEncoder, pulses0: &[i32]) {
    assert_eq!(pulses0.len(), SHELL_CODEC_FRAME_LENGTH);
    debug_assert!(pulses0.iter().all(|&value| value >= 0));

    let mut pulses1 = [0i32; SHELL_CODEC_FRAME_LENGTH / 2];
    let mut pulses2 = [0i32; SHELL_CODEC_FRAME_LENGTH / 4];
    let mut pulses3 = [0i32; SHELL_CODEC_FRAME_LENGTH / 8];
    let mut pulses4 = [0i32; 1];

    combine_pulses(&mut pulses1, pulses0);
    combine_pulses(&mut pulses2, &pulses1);
    combine_pulses(&mut pulses3, &pulses2);
    combine_pulses(&mut pulses4, &pulses3);

    encode_split(encoder, pulses3[0], pulses4[0], &SILK_SHELL_CODE_TABLE3);

    encode_split(encoder, pulses2[0], pulses3[0], &SILK_SHELL_CODE_TABLE2);

    encode_split(encoder, pulses1[0], pulses2[0], &SILK_SHELL_CODE_TABLE1);
    encode_split(encoder, pulses0[0], pulses1[0], &SILK_SHELL_CODE_TABLE0);
    encode_split(encoder, pulses0[2], pulses1[1], &SILK_SHELL_CODE_TABLE0);

    encode_split(encoder, pulses1[2], pulses2[1], &SILK_SHELL_CODE_TABLE1);
    encode_split(encoder, pulses0[4], pulses1[2], &SILK_SHELL_CODE_TABLE0);
    encode_split(encoder, pulses0[6], pulses1[3], &SILK_SHELL_CODE_TABLE0);

    encode_split(encoder, pulses2[2], pulses3[1], &SILK_SHELL_CODE_TABLE2);

    encode_split(encoder, pulses1[4], pulses2[2], &SILK_SHELL_CODE_TABLE1);
    encode_split(encoder, pulses0[8], pulses1[4], &SILK_SHELL_CODE_TABLE0);
    encode_split(encoder, pulses0[10], pulses1[5], &SILK_SHELL_CODE_TABLE0);

    encode_split(encoder, pulses1[6], pulses2[3], &SILK_SHELL_CODE_TABLE1);
    encode_split(encoder, pulses0[12], pulses1[6], &SILK_SHELL_CODE_TABLE0);
    encode_split(encoder, pulses0[14], pulses1[7], &SILK_SHELL_CODE_TABLE0);
}

/// Shell decoder, operates on one shell code frame of 16 pulses.
pub fn silk_shell_decoder(
    pulses0: &mut [i16],
    decoder: &mut RangeDecoder<'_>,
    total_pulses: i32,
) {
    assert_eq!(pulses0.len(), SHELL_CODEC_FRAME_LENGTH);
    debug_assert!(total_pulses >= 0);

    let mut pulses3 = [0i32; 2];
    let mut pulses2 = [0i32; 4];
    let mut pulses1 = [0i32; 8];
    let mut temp = [0i32; SHELL_CODEC_FRAME_LENGTH];

    (pulses3[0], pulses3[1]) = decode_split(decoder, total_pulses, &SILK_SHELL_CODE_TABLE3);

    (pulses2[0], pulses2[1]) = decode_split(decoder, pulses3[0], &SILK_SHELL_CODE_TABLE2);

    (pulses1[0], pulses1[1]) = decode_split(decoder, pulses2[0], &SILK_SHELL_CODE_TABLE1);
    (temp[0], temp[1]) = decode_split(decoder, pulses1[0], &SILK_SHELL_CODE_TABLE0);
    (temp[2], temp[3]) = decode_split(decoder, pulses1[1], &SILK_SHELL_CODE_TABLE0);

    (pulses1[2], pulses1[3]) = decode_split(decoder, pulses2[1], &SILK_SHELL_CODE_TABLE1);
    (temp[4], temp[5]) = decode_split(decoder, pulses1[2], &SILK_SHELL_CODE_TABLE0);
    (temp[6], temp[7]) = decode_split(decoder, pulses1[3], &SILK_SHELL_CODE_TABLE0);

    (pulses2[2], pulses2[3]) = decode_split(decoder, pulses3[1], &SILK_SHELL_CODE_TABLE2);

    (pulses1[4], pulses1[5]) = decode_split(decoder, pulses2[2], &SILK_SHELL_CODE_TABLE1);
    (temp[8], temp[9]) = decode_split(decoder, pulses1[4], &SILK_SHELL_CODE_TABLE0);
    (temp[10], temp[11]) = decode_split(decoder, pulses1[5], &SILK_SHELL_CODE_TABLE0);

    (pulses1[6], pulses1[7]) = decode_split(decoder, pulses2[3], &SILK_SHELL_CODE_TABLE1);
    (temp[12], temp[13]) = decode_split(decoder, pulses1[6], &SILK_SHELL_CODE_TABLE0);
    (temp[14], temp[15]) = decode_split(decoder, pulses1[7], &SILK_SHELL_CODE_TABLE0);

    for (dst, &value) in pulses0.iter_mut().zip(temp.iter()) {
        *dst = value as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn round_trip_shell_coder() {
        let pulses = [
            3, 0, 1, 2, 0, 0, 0, 1, 0, 1, 0, 0, 2, 1, 0, 0,
        ];

        let mut encoder = RangeEncoder::new();
        silk_shell_encoder(&mut encoder, &pulses);
        let data = encoder.finish();

        let mut decoder = RangeDecoder::init(&data);
        let mut decoded = [0i16; SHELL_CODEC_FRAME_LENGTH];
        silk_shell_decoder(
            &mut decoded,
            &mut decoder,
            pulses.iter().copied().sum::<i32>(),
        );

        let recovered: Vec<i32> = decoded.iter().map(|&value| i32::from(value)).collect();
        assert_eq!(recovered.as_slice(), pulses);
    }

    #[test]
    fn decoding_zero_pulses_yields_zeros() {
        let mut decoder = RangeDecoder::init(&[]);
        let mut pulses = [1i16; SHELL_CODEC_FRAME_LENGTH];
        silk_shell_decoder(&mut pulses, &mut decoder, 0);
        assert!(pulses.iter().all(|&value| value == 0));
    }

    #[test]
    fn handles_maximum_pulse_count() {
        let pulses = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let mut encoder = RangeEncoder::new();
        silk_shell_encoder(&mut encoder, &pulses);
        let data = encoder.finish();

        let mut decoder = RangeDecoder::init(&data);
        let mut decoded = [0i16; SHELL_CODEC_FRAME_LENGTH];
        silk_shell_decoder(&mut decoded, &mut decoder, 16);

        let recovered: Vec<i32> = decoded.iter().map(|&value| i32::from(value)).collect();
        assert_eq!(recovered.as_slice(), pulses);
    }
}
