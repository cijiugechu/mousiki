use mousiki::opus_decoder::{opus_decoder_create, opus_decoder_get_nb_samples};
use mousiki::packet::{
    Bandwidth, PacketError, opus_packet_get_bandwidth, opus_packet_get_nb_channels,
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
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
