use mousiki_ogg::{BitOrder, BitPacker, BitUnpacker, Packet, StreamState, SyncState};

#[test]
fn bitpack_api_matches_upstream_baseline() {
    let mut pack_lsb = BitPacker::new(BitOrder::Lsb);
    assert_eq!(0, pack_lsb.writecheck());
    pack_lsb.write(5, 3).expect("write");
    assert_eq!(3, pack_lsb.bits());
    assert_eq!(1, pack_lsb.bytes());
    pack_lsb.writealign().expect("align");
    assert_eq!(8, pack_lsb.bits());
    pack_lsb.writetrunc(3);
    assert_eq!(3, pack_lsb.bits());
    assert_eq!(1, pack_lsb.bytes());

    let mut read_lsb = BitUnpacker::new(BitOrder::Lsb, pack_lsb.buffer());
    assert_eq!(5, read_lsb.read(3).expect("read"));

    pack_lsb.writeclear();
    assert_eq!(-1, pack_lsb.writecheck());

    let mut pack_msb = BitPacker::new(BitOrder::Msb);
    assert_eq!(0, pack_msb.writecheck());
    pack_msb.write(5, 3).expect("write");
    assert_eq!(3, pack_msb.bits());
    pack_msb.writealign().expect("align");
    assert_eq!(8, pack_msb.bits());
    pack_msb.writetrunc(3);
    assert_eq!(3, pack_msb.bits());

    let mut read_msb = BitUnpacker::new(BitOrder::Msb, pack_msb.buffer());
    assert_eq!(5, read_msb.read(3).expect("read"));

    pack_msb.writeclear();
    assert_eq!(-1, pack_msb.writecheck());
}

#[test]
fn page_stream_sync_api_matches_upstream_baseline() {
    let mut stream = StreamState::new(0x1234_5678);
    assert_eq!(0, stream.check());
    assert_eq!(0, stream.eos());

    let packet = Packet::new(vec![0x11, 0x22, 0x33], true, true, 321, 0);
    assert_eq!(0, stream.packet_in(&packet));
    assert_ne!(0, stream.eos());

    let page = stream.flush().expect("page");
    assert_eq!(0, page.version());
    assert_eq!(0, page.continued());
    assert_ne!(0, page.bos());
    assert_ne!(0, page.eos());
    assert_eq!(0, page.granulepos());
    assert_eq!(0x1234_5678_u32 as i32, page.serialno());
    assert_eq!(0, page.pageno());
    assert_eq!(1, page.packets());

    let mut checksum_page = page.clone();
    checksum_page.header[22..26].fill(0);
    checksum_page.checksum_set();
    assert_eq!(&checksum_page.header[22..26], &page.header[22..26]);

    stream.reset_serialno(0x0bad_beef_u32 as i32);
    assert_eq!(0x0bad_beef_u32 as i32, stream.serialno);
    assert_eq!(-1, stream.pageno);
    assert_eq!(0, stream.eos());

    let page_bytes = page.to_bytes();
    let mut sync = SyncState::new();
    assert_eq!(0, sync.check());
    {
        let buf = sync.buffer(page_bytes.len() + 2);
        buf[0] = b'x';
        buf[1] = b'y';
        buf[2..].copy_from_slice(&page_bytes);
    }
    assert_eq!(0, sync.wrote(page_bytes.len() + 2));
    assert_eq!(-2, sync.pageseek().expect_err("skip garbage"));
    let synced_page = sync.pageseek().expect("pageseek").expect("page");
    assert_eq!(page.serialno(), synced_page.serialno());

    sync.reset();
    {
        let buf = sync.buffer(page_bytes.len());
        buf.copy_from_slice(&page_bytes);
    }
    assert_eq!(0, sync.wrote(page_bytes.len()));
    let synced_page = sync.pageout().expect("pageout").expect("page");
    assert_eq!(1, synced_page.packets());

    sync.clear();
    assert_eq!(0, sync.check());
    assert_eq!(16, sync.buffer(16).len());

    stream.clear();
    assert_ne!(0, stream.check());
}

#[test]
fn packet_clear_matches_upstream_baseline() {
    let mut packet = Packet::new(vec![0; 8], true, true, 99, 7);
    packet.clear();
    assert!(packet.packet.is_empty());
    assert_eq!(0, packet.b_o_s);
    assert_eq!(0, packet.e_o_s);
    assert_eq!(0, packet.granulepos);
    assert_eq!(0, packet.packetno);
}
