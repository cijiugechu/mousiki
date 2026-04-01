use mousiki_ogg::c_style_api::*;
use mousiki_ogg::{Packet, PacketMetadata};

#[test]
fn c_style_api_matches_upstream_baseline() {
    let mut pack_lsb = OggPackBuffer::default();
    oggpack_writeinit(&mut pack_lsb);
    assert_eq!(0, oggpack_writecheck(&pack_lsb));
    assert_eq!(0, oggpack_write(&mut pack_lsb, 5, 3));
    assert_eq!(3, oggpack_bits(&pack_lsb));
    assert_eq!(1, oggpack_bytes(&pack_lsb));
    assert_eq!(0, oggpack_writealign(&mut pack_lsb));
    assert_eq!(8, oggpack_bits(&pack_lsb));
    oggpack_writetrunc(&mut pack_lsb, 3);
    assert_eq!(3, oggpack_bits(&pack_lsb));

    let pack_lsb_bytes = oggpack_get_buffer(&pack_lsb).to_vec();
    oggpack_readinit(&mut pack_lsb, &pack_lsb_bytes);
    assert_eq!(5, oggpack_read(&mut pack_lsb, 3));
    oggpack_writeclear(&mut pack_lsb);
    assert_eq!(-1, oggpack_writecheck(&pack_lsb));

    let mut stream = OggStreamState::new(0x1234_5678);
    assert_eq!(0, ogg_stream_check(&stream));
    assert_eq!(0, ogg_stream_eos(&stream));

    let packet = Packet::with_metadata(
        vec![0x11, 0x22, 0x33],
        PacketMetadata {
            beginning_of_stream: true,
            end_of_stream: true,
            granule_position: 321,
            ..PacketMetadata::default()
        },
    );
    assert_eq!(0, ogg_stream_packetin(&mut stream, &packet));
    assert_ne!(0, ogg_stream_eos(&stream));

    let page = ogg_stream_flush(&mut stream).expect("page");
    assert_eq!(0, ogg_page_version(&page));
    assert_eq!(0, ogg_page_continued(&page));
    assert_ne!(0, ogg_page_bos(&page));
    assert_ne!(0, ogg_page_eos(&page));
    assert_eq!(0, ogg_page_granulepos(&page));
    assert_eq!(0x1234_5678_u32 as i32, ogg_page_serialno(&page));
    assert_eq!(0, ogg_page_pageno(&page));
    assert_eq!(1, ogg_page_packets(&page));

    let mut checksum_page = page.clone();
    checksum_page.update_checksum();
    assert_eq!(
        &checksum_page.header_bytes()[22..26],
        &page.header_bytes()[22..26]
    );

    ogg_stream_reset_serialno(&mut stream, 0x0bad_beef_u32 as i32);
    let followup = Packet::new(vec![0xaa]);
    assert_eq!(0, ogg_stream_packetin(&mut stream, &followup));
    let followup_page = ogg_stream_flush(&mut stream).expect("followup page");
    assert_eq!(0x0bad_beef_u32 as i32, ogg_page_serialno(&followup_page));
    assert_eq!(0, ogg_page_pageno(&followup_page));

    let page_bytes = page.to_bytes();
    let mut sync = OggSyncState::new();
    assert_eq!(0, ogg_sync_check(&sync));
    {
        let buf = ogg_sync_buffer(&mut sync, page_bytes.len() + 2);
        buf[0] = b'x';
        buf[1] = b'y';
        buf[2..].copy_from_slice(&page_bytes);
    }
    assert_eq!(0, ogg_sync_wrote(&mut sync, page_bytes.len() + 2));
    assert_eq!(-2, ogg_sync_pageseek(&mut sync).expect_err("skip garbage"));
    let synced_page = ogg_sync_pageseek(&mut sync)
        .expect("pageseek")
        .expect("page");
    assert_eq!(ogg_page_serialno(&page), ogg_page_serialno(&synced_page));

    ogg_sync_reset(&mut sync);
    {
        let buf = ogg_sync_buffer(&mut sync, page_bytes.len());
        buf.copy_from_slice(&page_bytes);
    }
    assert_eq!(0, ogg_sync_wrote(&mut sync, page_bytes.len()));
    let synced_page = ogg_sync_pageout(&mut sync).expect("pageout").expect("page");
    assert_eq!(1, ogg_page_packets(&synced_page));

    ogg_sync_clear(&mut sync);
    assert_eq!(0, ogg_sync_check(&sync));
    assert_eq!(16, ogg_sync_buffer(&mut sync, 16).len());

    ogg_stream_clear(&mut stream);
    assert_ne!(0, ogg_stream_check(&stream));

    let mut cleared_packet = Packet::with_metadata(
        vec![0; 8],
        PacketMetadata {
            beginning_of_stream: true,
            end_of_stream: true,
            granule_position: 99,
            sequence_number: 7,
        },
    );
    ogg_packet_clear(&mut cleared_packet);
    assert!(cleared_packet.data().is_empty());
    assert!(!cleared_packet.is_beginning_of_stream());
    assert!(!cleared_packet.is_end_of_stream());
    assert_eq!(0, cleared_packet.granule_position());
    assert_eq!(0, cleared_packet.sequence_number());
}
