use mousiki::opus_decoder::{opus_decoder_create, opus_decoder_get_nb_samples};
use mousiki::packet::{
    Bandwidth, Mode, PacketError, opus_packet_get_bandwidth, opus_packet_get_mode,
    opus_packet_get_nb_channels, opus_packet_get_nb_frames, opus_packet_get_nb_samples,
    opus_packet_get_samples_per_frame, opus_packet_parse_impl,
};

const SAMPLE_RATES: [u32; 5] = [8000, 12_000, 16_000, 24_000, 48_000];

fn reference_bandwidth(toc: u8) -> Bandwidth {
    let bw = toc >> 4;
    let code = 1101
        + ((((i32::from(bw & 7) * 9) & (63 - i32::from(bw & 8)))
            + 2
            + 12 * if bw & 8 != 0 { 1 } else { 0 })
            >> 4);

    Bandwidth::from_opus_int(code).expect("reference bandwidth code should map to enum")
}

fn reference_fp3s(toc: u8) -> i32 {
    let mut fp3s = (toc >> 3) as i32;
    fp3s = ((((3 - (fp3s & 3)) * 13 & 119) + 9) >> 2)
        * (((fp3s > 13) as i32 * (3 - ((fp3s & 3) == 3) as i32)) + 1)
        * 25;
    fp3s
}

#[test]
fn bandwidth_matches_reference_for_all_toc_values() {
    for toc in 0u8..=255 {
        assert_eq!(
            opus_packet_get_bandwidth(&[toc]).unwrap(),
            reference_bandwidth(toc),
            "failed for toc byte {toc:#04x}"
        );
    }

    assert_eq!(
        opus_packet_get_bandwidth(&[]),
        Err(PacketError::BadArgument)
    );
}

#[test]
fn samples_per_frame_matches_reference() {
    for toc in 0u8..=255 {
        let fp3s = reference_fp3s(toc);
        assert_ne!(fp3s, 0);

        for &rate in &SAMPLE_RATES {
            let expected = (rate as i32 * 3 / fp3s) as usize;
            let actual = opus_packet_get_samples_per_frame(&[toc], rate).unwrap();
            assert_eq!(actual, expected, "toc {toc:#04x}, rate {rate}");
        }
    }

    assert_eq!(
        opus_packet_get_samples_per_frame(&[], SAMPLE_RATES[0]),
        Err(PacketError::BadArgument)
    );
}

#[test]
fn frame_count_matches_reference_cases() {
    let mut packet = [0u8; 2];

    assert_eq!(
        opus_packet_get_nb_frames(&packet[..], 0),
        Err(PacketError::BadArgument)
    );
    assert_eq!(
        opus_packet_get_nb_frames(&packet[..1], 2),
        Err(PacketError::BadArgument)
    );

    for toc in 0u8..=255 {
        packet[0] = toc;
        let l1_expected = match toc & 0x03 {
            0 => Ok(1),
            1 | 2 => Ok(2),
            _ => Err(PacketError::InvalidPacket),
        };

        assert_eq!(
            opus_packet_get_nb_frames(&packet[..], 1),
            l1_expected,
            "len=1 toc {toc:#04x}"
        );

        for second in 0u8..=255 {
            packet[1] = second;
            let expected = if toc & 0x03 != 3 {
                l1_expected
            } else {
                Ok((second & 0x3F) as usize)
            };

            assert_eq!(
                opus_packet_get_nb_frames(&packet[..], 2),
                expected,
                "len=2 toc {toc:#04x} second {second:#04x}"
            );
        }
    }
}

#[test]
fn channel_count_follows_header_flag() {
    assert_eq!(opus_packet_get_nb_channels(&[0]).unwrap(), 1);
    assert_eq!(opus_packet_get_nb_channels(&[0x04]).unwrap(), 2);
    assert_eq!(
        opus_packet_get_nb_channels(&[]),
        Err(PacketError::BadArgument)
    );
}

#[test]
fn sample_count_matches_reference_api_expectations() {
    let mut packet = [0u8; 2];
    packet[0] = 0;

    assert_eq!(
        opus_packet_get_nb_samples(&packet[..], 1, 48_000).unwrap(),
        480
    );
    assert_eq!(
        opus_packet_get_nb_samples(&packet[..], 1, 96_000).unwrap(),
        960
    );
    assert_eq!(
        opus_packet_get_nb_samples(&packet[..], 1, 32_000).unwrap(),
        320
    );
    assert_eq!(
        opus_packet_get_nb_samples(&packet[..], 1, 8_000).unwrap(),
        80
    );

    packet[0] = 3;
    assert_eq!(
        opus_packet_get_nb_samples(&packet[..], 1, 24_000),
        Err(PacketError::InvalidPacket)
    );

    packet[0] = (63 << 2) | 3;
    packet[1] = 63;
    assert_eq!(
        opus_packet_get_nb_samples(&packet[..], 0, 24_000),
        Err(PacketError::BadArgument)
    );
    assert_eq!(
        opus_packet_get_nb_samples(&packet[..], 2, 48_000),
        Err(PacketError::InvalidPacket)
    );

    let decoder = opus_decoder_create(48_000, 2).expect("decoder should initialise");
    assert_eq!(
        opus_decoder_get_nb_samples(&decoder, &packet[..], 2),
        Err(PacketError::InvalidPacket)
    );
}

#[test]
fn packet_mode_matches_bit_layout() {
    assert_eq!(opus_packet_get_mode(&[0x00]).unwrap(), Mode::SILK);
    assert_eq!(opus_packet_get_mode(&[0x60]).unwrap(), Mode::HYBRID);
    assert_eq!(opus_packet_get_mode(&[0x80]).unwrap(), Mode::CELT);

    assert_eq!(opus_packet_get_mode(&[]), Err(PacketError::BadArgument));
}

#[test]
fn parses_self_delimited_single_frame() {
    let packet: [u8; 7] = [0x00, 0x05, 1, 2, 3, 4, 5];
    let parsed = opus_packet_parse_impl(&packet, packet.len(), true).unwrap();

    assert_eq!(parsed.toc, 0x00);
    assert_eq!(parsed.frame_count, 1);
    assert_eq!(parsed.frame_sizes[0], 5);
    assert_eq!(parsed.payload_offset, 2);
    assert_eq!(parsed.packet_offset, packet.len());
    assert_eq!(parsed.frames[0], &packet[2..]);
    assert!(parsed.padding.is_empty());
}

#[test]
fn parses_self_delimited_cbr_with_padding() {
    let packet: [u8; 12] = [
        0x03, // toc: code 3 -> arbitrary frames
        0x42, // padding flag set, 2 frames, CBR
        0x02, // padding length
        0x03, // self-delimited size for both frames
        0xAA, 0xAB, 0xAC, // frame 0
        0xBA, 0xBB, 0xBC, // frame 1
        0xEE, 0xEF, // padding bytes
    ];

    let parsed = opus_packet_parse_impl(&packet, packet.len(), true).unwrap();

    assert_eq!(parsed.frame_count, 2);
    assert_eq!(parsed.frame_sizes[0], 3);
    assert_eq!(parsed.frame_sizes[1], 3);
    assert_eq!(parsed.payload_offset, 4);
    assert_eq!(parsed.packet_offset, packet.len());
    assert_eq!(parsed.frames[0], &packet[4..7]);
    assert_eq!(parsed.frames[1], &packet[7..10]);
    assert_eq!(parsed.padding, &packet[10..]);
}

#[test]
fn rejects_oversized_self_delimited_frame() {
    // Frame size claims four bytes but only two bytes of payload remain.
    let packet: [u8; 5] = [0x01, 0x04, 0xAA, 0xBB, 0xCC];

    let err = opus_packet_parse_impl(&packet, packet.len(), true).unwrap_err();
    assert_eq!(err, PacketError::InvalidPacket);
}
