use mousiki_ogg::{BitOrder, BitPacker, Packet, StreamState};

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

    let mut lsb = BitPacker::new(BitOrder::Lsb);
    for &value in &testbuffer1 {
        let bits = 32 - value.leading_zeros();
        lsb.write(value, bits as i32).expect("lsb write");
    }
    assert_eq!(&expected_lsb, lsb.buffer());

    let mut msb = BitPacker::new(BitOrder::Msb);
    for &value in &testbuffer1 {
        let bits = 32 - value.leading_zeros();
        msb.write(value, bits as i32).expect("msb write");
    }
    assert_eq!(&expected_msb, msb.buffer());
}

#[test]
fn upstream_single_page_header_matches() {
    let expected_header: [u8; 28] = [
        0x4f, 0x67, 0x67, 0x53, 0, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x02, 0x03, 0x04, 0, 0, 0, 0, 0x15, 0xed, 0xec, 0x91, 1, 17,
    ];
    let mut stream = StreamState::new(0x0403_0201);
    let packet = Packet::new(
        (0..17).map(|value| value as u8).collect(),
        false,
        true,
        7,
        0,
    );
    assert_eq!(0, stream.packet_in(&packet));
    let page = stream.page_out().expect("page");
    assert_eq!(&expected_header, page.header.as_slice());
    assert_eq!(
        (0..17).map(|value| value as u8).collect::<Vec<_>>(),
        page.body
    );
}
