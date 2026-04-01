use mousiki_ogg::c_style_api::*;
use mousiki_ogg::{Packet, PacketMetadata};

#[test]
fn upstream_bitwise_lsb_and_msb_vectors_match() {
    let testbuffer1: [u32; 43] = [
        18, 12, 103948, 4325, 543, 76, 432, 52, 3, 65, 4, 56, 32, 42, 34, 21, 1, 23, 32, 546, 456,
        7, 567, 56, 8, 8, 55, 3, 52, 342, 341, 4, 265, 7, 67, 86, 2199, 21, 7, 1, 5, 1, 4,
    ];
    let expected_lsb: [u8; 33] = [
        146, 25, 44, 151, 195, 15, 153, 176, 233, 131, 196, 65, 85, 172, 47, 40, 34, 242, 223, 136,
        35, 222, 211, 86, 171, 50, 225, 135, 214, 75, 172, 223, 4,
    ];
    let expected_msb: [u8; 33] = [
        150, 101, 131, 33, 203, 15, 204, 216, 105, 193, 156, 65, 84, 85, 222, 8, 139, 145, 227,
        126, 34, 55, 244, 171, 85, 100, 39, 195, 173, 18, 245, 251, 128,
    ];

    let mut lsb = OggPackBuffer::default();
    oggpack_writeinit(&mut lsb);
    for &value in &testbuffer1 {
        let bits = 32 - value.leading_zeros();
        assert_eq!(0, oggpack_write(&mut lsb, value, bits as i32));
    }
    assert_eq!(&expected_lsb, oggpack_get_buffer(&lsb));

    let mut msb = OggPackBuffer::default();
    oggpack_b_writeinit(&mut msb);
    for &value in &testbuffer1 {
        let bits = 32 - value.leading_zeros();
        assert_eq!(0, oggpack_b_write(&mut msb, value, bits as i32));
    }
    assert_eq!(&expected_msb, oggpack_b_get_buffer(&msb));
}

#[test]
fn upstream_single_page_header_matches() {
    let expected_header: [u8; 28] = [
        0x4f, 0x67, 0x67, 0x53, 0, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x02, 0x03, 0x04, 0, 0, 0, 0, 0x15, 0xed, 0xec, 0x91, 1, 17,
    ];
    let mut stream = OggStreamState::new(0x0403_0201);
    let packet = Packet::with_metadata(
        (0..17).map(|value| value as u8).collect(),
        PacketMetadata {
            end_of_stream: true,
            granule_position: 7,
            ..PacketMetadata::default()
        },
    );
    assert_eq!(0, ogg_stream_packetin(&mut stream, &packet));
    let page = ogg_stream_pageout(&mut stream).expect("page");
    assert_eq!(&expected_header, page.header_bytes());
    assert_eq!(
        (0..17).map(|value| value as u8).collect::<Vec<_>>(),
        page.body_bytes()
    );
}
