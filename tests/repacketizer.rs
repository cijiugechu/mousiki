use mousiki::repacketizer::{
    opus_multistream_packet_pad, opus_multistream_packet_unpad, opus_packet_pad,
    opus_packet_unpad, OpusRepacketizer, RepacketizerError,
};

#[test]
fn size_and_init_reset_state() {
    let size = OpusRepacketizer::opus_repacketizer_get_size();
    assert!(size > 0);

    let mut rp = OpusRepacketizer::new();
    rp.opus_repacketizer_init();
    assert_eq!(rp.opus_repacketizer_get_nb_frames(), 0);
}

#[test]
fn rejects_empty_and_toc_mismatch() {
    let mut rp = OpusRepacketizer::new();
    let packet = [0u8];
    assert_eq!(
        rp.opus_repacketizer_cat(&packet, 0),
        Err(RepacketizerError::InvalidPacket)
    );

    let packet_ok = [0u8, 0xAA, 0xBB];
    rp.opus_repacketizer_cat(&packet_ok, packet_ok.len())
        .expect("first packet should be accepted");

    let mut packet_bad = packet_ok;
    packet_bad[0] = 0x04; // change upper ToC bits
    assert_eq!(
        rp.opus_repacketizer_cat(&packet_bad, packet_bad.len()),
        Err(RepacketizerError::InvalidPacket)
    );
}

#[test]
fn out_roundtrips_single_frame_packet() {
    let mut rp = OpusRepacketizer::new();
    let packet = [0u8, 0xAA, 0xBB];
    rp.opus_repacketizer_cat(&packet, packet.len())
        .expect("cat should succeed");

    let mut out = [0u8; 8];
    let capacity = out.len();
    let written = rp
        .opus_repacketizer_out(&mut out, capacity)
        .expect("out should succeed");

    assert_eq!(written, packet.len());
    assert_eq!(&out[..written], &packet);
}

#[test]
fn combines_two_equal_coded_frames_into_code1_packet() {
    let mut rp = OpusRepacketizer::new();
    let packet = [1u8, 0xAA, 0xBB, 0xCC, 0xDD];

    rp.opus_repacketizer_cat(&packet, packet.len())
        .expect("first packet should be accepted");
    rp.opus_repacketizer_cat(&packet, packet.len())
        .expect("second packet should be accepted");

    let mut out = [0u8; 12];
    let capacity = out.len();
    let written = rp
        .opus_repacketizer_out(&mut out, capacity)
        .expect("out should succeed");

    // Four CBR frames should be encoded as Code 3 with a 4-frame count.
    assert_eq!(written, 10);
    assert_eq!(out[0] & 0x03, 3); // code 3 (arbitrary frames)
    assert_eq!(out[1] & 0x3F, 4); // frame count
    assert_eq!(&out[2..written], &[0xAA, 0xBB, 0xCC, 0xDD, 0xAA, 0xBB, 0xCC, 0xDD]);
}

#[test]
fn returns_buffer_too_small_when_output_slice_is_short() {
    let mut rp = OpusRepacketizer::new();
    let packet = [0u8, 0xAA, 0xBB];
    rp.opus_repacketizer_cat(&packet, packet.len())
        .expect("cat should succeed");

    let mut out = [0u8; 2];
    let capacity = out.len();
    let err = rp
        .opus_repacketizer_out(&mut out, capacity)
        .expect_err("out should fail due to capacity");
    assert_eq!(err, RepacketizerError::BufferTooSmall);
}

#[test]
fn pad_and_unpad_preserve_payload() {
    let mut packet = [0u8, 1, 2, 3, 0, 0];
    opus_packet_pad(&mut packet, 4, 6).expect("pad should succeed");
    let new_len = opus_packet_unpad(&mut packet, 6).expect("unpad should succeed");
    assert_eq!(new_len, 4);
    assert_eq!(&packet[..new_len], &[0, 1, 2, 3]);
}

#[test]
fn multistream_pad_and_unpad_roundtrip() {
    let mut packet = [0u8, 0xAA, 0xBB, 0, 0];
    opus_multistream_packet_pad(&mut packet, 3, 5, 1).expect("pad should succeed");
    let new_len =
        opus_multistream_packet_unpad(&mut packet, 5, 1).expect("unpad should succeed");
    assert_eq!(new_len, 3);
    assert_eq!(&packet[..new_len], &[0, 0xAA, 0xBB]);
}

