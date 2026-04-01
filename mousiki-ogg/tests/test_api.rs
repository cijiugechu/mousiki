use mousiki_ogg::{
    BitOrder, BitPacker, BitUnpacker, Packet, PacketMetadata, PageReader, StreamDecoder,
    StreamEncoder,
};

#[test]
fn bitpack_rust_api_tracks_bit_lengths() {
    let mut packer = BitPacker::new(BitOrder::Lsb);
    assert!(packer.is_valid());
    packer.write_bits(5, 3).expect("write");
    assert_eq!(3, packer.bit_len());
    assert_eq!(1, packer.byte_len());
    packer.align_to_byte().expect("align");
    assert_eq!(8, packer.bit_len());
    packer.truncate_bits(3);
    assert_eq!(3, packer.bit_len());

    let mut unpacker = BitUnpacker::new(BitOrder::Lsb, packer.as_bytes());
    assert_eq!(5, unpacker.read_bits(3).expect("read"));

    packer.clear();
    assert!(!packer.is_valid());
}

#[test]
fn stream_encoder_decoder_roundtrip_packets() {
    let mut encoder = StreamEncoder::new(0x1234_5678);
    let metadata = PacketMetadata {
        beginning_of_stream: true,
        end_of_stream: true,
        granule_position: 321,
        sequence_number: 0,
    };
    encoder
        .push_packet_data(&[0x11, 0x22, 0x33], metadata)
        .expect("packet");
    assert!(encoder.is_end_of_stream());
    assert_eq!(1, encoder.pending_segment_count());

    let page = encoder.flush_page().expect("page");
    assert_eq!(0, page.version());
    assert!(!page.is_continued());
    assert!(page.is_beginning_of_stream());
    assert!(page.is_end_of_stream());
    assert_eq!(0, page.granule_position());
    assert_eq!(0x1234_5678_u32 as i32, page.stream_serial());
    assert_eq!(0, page.sequence_number());
    assert_eq!(1, page.packet_count());

    let mut decoder = StreamDecoder::new(page.stream_serial());
    decoder.push_page(&page).expect("page in");
    let packet = decoder.next_packet().expect("packet out").expect("packet");
    assert_eq!(&[0x11, 0x22, 0x33], packet.data());
    assert!(packet.is_beginning_of_stream());
    assert!(packet.is_end_of_stream());
    assert_eq!(0, packet.sequence_number());
}

#[test]
fn page_reader_extracts_pages_from_buffered_input() {
    let mut encoder = StreamEncoder::new(0x0bad_beef_u32 as i32);
    let packet = Packet::with_metadata(
        vec![1, 2, 3, 4],
        PacketMetadata {
            beginning_of_stream: true,
            granule_position: 77,
            ..PacketMetadata::default()
        },
    );
    encoder.push_packet(&packet).expect("packet");
    let page_bytes = encoder.flush_page().expect("page").into_bytes();

    let mut reader = PageReader::new();
    reader.push_bytes(&page_bytes[..5]).expect("first chunk");
    assert!(reader.next_page().expect("page state").is_none());
    reader.push_bytes(&page_bytes[5..]).expect("second chunk");
    let page = reader.next_page().expect("page state").expect("page");
    assert_eq!(0x0bad_beef_u32 as i32, page.stream_serial());
    assert_eq!(&[1, 2, 3, 4], page.body_bytes());
}
