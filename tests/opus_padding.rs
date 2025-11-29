//! Port of the padding overflow regression test from
//! `opus-c/tests/test_opus_padding.c`, adapted to the current Rust decoder API.

use mousiki::decoder::{Decoder, DecoderError};
use mousiki::opus_get_version_string;

// 16,909,318 bytes mirrors the pathological packet length from the C test.
const PACKET_SIZE: usize = 16_909_318;
// The current decoder always produces 320 SILK samples per frame, upsampled by
// a factor of three and stored as little-endian i16.
const OUTPUT_BYTES: usize = 320 * 3 * 2;

#[test]
fn padding_overflow_packet_is_rejected() {
    let version = opus_get_version_string();
    assert!(
        !version.is_empty(),
        "version string should be available for diagnostics"
    );

    let mut packet = vec![0xffu8; PACKET_SIZE];
    packet[1] = 0x41;
    packet[PACKET_SIZE - 1] = 0x0b;

    let mut decoder = Decoder::new();
    let mut output = vec![0u8; OUTPUT_BYTES];

    let err = decoder
        .decode(&packet, &mut output)
        .expect_err("invalid padded packet must be rejected");
    assert!(
        matches!(err, DecoderError::UnsupportedFrameCode(3)),
        "decoder should reject malformed TOC before touching padding"
    );
}
