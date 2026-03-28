#![cfg(feature = "libopusenc")]

mod common;

use mousiki::libopusenc::ogg_packer::OggPacker;

use crate::common::libopusenc::TestBuffer;

const OGG_FLAG_CONTINUED: u8 = 0x01;
const OGG_FLAG_BOS: u8 = 0x02;
const OGG_FLAG_EOS: u8 = 0x04;

#[derive(Debug)]
struct ParsedPage<'a> {
    granulepos: u64,
    serialno: i32,
    pageno: i32,
    lacing: &'a [u8],
    body: &'a [u8],
    len: usize,
    header_len: usize,
    body_len: usize,
    flags: u8,
    segment_count: usize,
}

fn parse_page(page: &[u8]) -> ParsedPage<'_> {
    assert!(page.len() >= 27);
    assert_eq!(&page[..4], b"OggS");
    let header_len = 27 + page[26] as usize;
    assert!(header_len <= page.len());
    ParsedPage {
        granulepos: u64::from_le_bytes(page[6..14].try_into().unwrap()),
        serialno: u32::from_le_bytes(page[14..18].try_into().unwrap()) as i32,
        pageno: u32::from_le_bytes(page[18..22].try_into().unwrap()) as i32,
        lacing: &page[27..header_len],
        body: &page[header_len..],
        len: page.len(),
        header_len,
        body_len: page.len() - header_len,
        flags: page[5],
        segment_count: page[26] as usize,
    }
}

fn compute_ogg_crc(data: &[u8]) -> u32 {
    let mut crc_reg = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        let value = if (22..26).contains(&i) { 0 } else { byte };
        crc_reg ^= (value as u32) << 24;
        for _ in 0..8 {
            if crc_reg & 0x8000_0000 != 0 {
                crc_reg = (crc_reg << 1) ^ 0x04c1_1db7;
            } else {
                crc_reg <<= 1;
            }
        }
    }
    crc_reg
}

fn fill_payload(payload: &mut [u8], seed: u8) {
    for (i, byte) in payload.iter_mut().enumerate() {
        *byte = seed.wrapping_add((31 * i) as u8).wrapping_add((i >> 1) as u8);
    }
}

fn commit_payload(oggp: &mut OggPacker, payload: &[u8], granulepos: u64, eos: bool) {
    let buffer = oggp.get_packet_buffer(payload.len()).expect("packet buffer");
    buffer.copy_from_slice(payload);
    oggp.commit_packet(payload.len(), granulepos, eos)
        .expect("commit packet");
}

fn commit_pattern_packet(oggp: &mut OggPacker, len: usize, granulepos: u64, eos: bool, seed: u8) {
    let mut payload = vec![0u8; len];
    fill_payload(&mut payload, seed);
    commit_payload(oggp, &payload, granulepos, eos);
}

fn test_page_crc_and_shape(page: &[u8]) {
    let parsed = parse_page(page);
    assert_eq!(b"OggS", &page[..4]);
    assert_eq!(0, page[4]);
    assert_eq!(parsed.len, parsed.header_len + parsed.body_len);
    assert_eq!(
        parsed.body_len,
        parsed.lacing.iter().map(|&value| value as usize).sum::<usize>()
    );
    assert_eq!(
        u32::from_le_bytes(page[22..26].try_into().unwrap()),
        compute_ogg_crc(page)
    );
}

fn expect_no_page(oggp: &mut OggPacker) {
    assert!(oggp.get_next_page().is_none());
}

#[test]
fn empty_state_matches_ctest() {
    let mut oggp = OggPacker::new(1234);
    expect_no_page(&mut oggp);
    assert!(!oggp.flush_page());
    expect_no_page(&mut oggp);
}

#[test]
fn single_packet_happy_path_matches_ctest() {
    let mut payload = [0u8; 7];
    fill_payload(&mut payload, 3);
    let mut oggp = OggPacker::new(777);

    commit_payload(&mut oggp, &payload, 123, false);
    assert!(oggp.flush_page());
    let page = oggp.get_next_page().expect("page");
    test_page_crc_and_shape(&page);
    let parsed = parse_page(&page);

    assert_eq!(0, parsed.flags & OGG_FLAG_CONTINUED);
    assert_eq!(OGG_FLAG_BOS, parsed.flags & OGG_FLAG_BOS);
    assert_eq!(0, parsed.flags & OGG_FLAG_EOS);
    assert_eq!(777, parsed.serialno);
    assert_eq!(0, parsed.pageno);
    assert_eq!(123, parsed.granulepos);
    assert_eq!(1, parsed.segment_count);
    assert_eq!(payload.len(), parsed.lacing[0] as usize);
    assert_eq!(payload.as_slice(), parsed.body);

    expect_no_page(&mut oggp);
}

#[test]
fn eos_behavior_matches_ctest() {
    let mut payload = [0u8; 4];
    fill_payload(&mut payload, 7);
    let mut oggp = OggPacker::new(501);

    commit_payload(&mut oggp, &payload, 44, true);
    assert!(oggp.flush_page());
    let page = oggp.get_next_page().expect("page");
    test_page_crc_and_shape(&page);
    let parsed = parse_page(&page);
    assert_eq!(OGG_FLAG_BOS, parsed.flags & OGG_FLAG_BOS);
    assert_eq!(OGG_FLAG_EOS, parsed.flags & OGG_FLAG_EOS);
    assert_eq!(0, parsed.flags & OGG_FLAG_CONTINUED);
    expect_no_page(&mut oggp);
}

#[test]
fn packet_size_edges_match_ctest() {
    for (i, size) in [0usize, 1, 254, 255, 256].into_iter().enumerate() {
        let mut oggp = OggPacker::new(900 + i as i32);
        let mut payload = vec![0u8; size];
        fill_payload(&mut payload, 11 + i as u8);

        commit_payload(&mut oggp, &payload, (i + 1) as u64, false);
        assert!(oggp.flush_page());
        let page = oggp.get_next_page().expect("page");
        test_page_crc_and_shape(&page);
        let parsed = parse_page(&page);

        match size {
            0 => {
                assert_eq!(1, parsed.segment_count);
                assert_eq!(0, parsed.lacing[0]);
                assert_eq!(0, parsed.body_len);
            }
            1 | 254 => {
                assert_eq!(1, parsed.segment_count);
                assert_eq!(size, parsed.lacing[0] as usize);
            }
            255 => {
                assert_eq!(2, parsed.segment_count);
                assert_eq!(255, parsed.lacing[0]);
                assert_eq!(0, parsed.lacing[1]);
            }
            256 => {
                assert_eq!(2, parsed.segment_count);
                assert_eq!(255, parsed.lacing[0]);
                assert_eq!(1, parsed.lacing[1]);
            }
            _ => unreachable!(),
        }
        assert_eq!(payload.as_slice(), parsed.body);
        expect_no_page(&mut oggp);
    }
}

#[test]
fn multiple_packets_on_one_page_matches_ctest() {
    let mut payload_a = [0u8; 3];
    let mut payload_b = [0u8; 4];
    let mut payload_c = [0u8; 5];
    fill_payload(&mut payload_a, 1);
    fill_payload(&mut payload_b, 2);
    fill_payload(&mut payload_c, 3);

    let mut oggp = OggPacker::new(314);
    commit_payload(&mut oggp, &payload_a, 10, false);
    commit_payload(&mut oggp, &payload_b, 20, false);
    commit_payload(&mut oggp, &payload_c, 30, false);
    assert!(oggp.flush_page());
    let page = oggp.get_next_page().expect("page");
    test_page_crc_and_shape(&page);
    let parsed = parse_page(&page);

    assert_eq!(3, parsed.segment_count);
    assert_eq!(3, parsed.lacing[0]);
    assert_eq!(4, parsed.lacing[1]);
    assert_eq!(5, parsed.lacing[2]);
    assert_eq!(30, parsed.granulepos);

    let mut expected = TestBuffer::default();
    expected.append(&payload_a);
    expected.append(&payload_b);
    expected.append(&payload_c);
    assert_eq!(expected.data.as_slice(), parsed.body);
}

#[test]
fn muxing_delay_auto_flush_matches_ctest() {
    let mut oggp = OggPacker::new(81);
    oggp.set_muxing_delay(10);

    commit_pattern_packet(&mut oggp, 5, 10, false, 1);
    let first_page = oggp.get_next_page().expect("first page");
    test_page_crc_and_shape(&first_page);
    let first = parse_page(&first_page);
    assert_eq!(10, first.granulepos);
    assert_eq!(0, first.pageno);

    commit_pattern_packet(&mut oggp, 6, 11, false, 2);
    expect_no_page(&mut oggp);

    commit_pattern_packet(&mut oggp, 7, 22, false, 3);
    let second_page = oggp.get_next_page().expect("second page");
    let third_page = oggp.get_next_page().expect("third page");
    test_page_crc_and_shape(&second_page);
    test_page_crc_and_shape(&third_page);
    let second = parse_page(&second_page);
    let third = parse_page(&third_page);

    assert_eq!(11, second.granulepos);
    assert_eq!(22, third.granulepos);
    assert_eq!(1, second.pageno);
    assert_eq!(2, third.pageno);
    expect_no_page(&mut oggp);
}

#[test]
fn large_packet_continuation_matches_ctest() {
    let payload_size = 70_000usize;
    let mut payload = vec![0u8; payload_size];
    fill_payload(&mut payload, 9);

    let mut oggp = OggPacker::new(404);
    commit_payload(&mut oggp, &payload, 999, false);
    assert!(oggp.flush_page());

    let first_page = oggp.get_next_page().expect("first page");
    let second_page = oggp.get_next_page().expect("second page");
    test_page_crc_and_shape(&first_page);
    test_page_crc_and_shape(&second_page);
    let first = parse_page(&first_page);
    let second = parse_page(&second_page);

    assert_eq!(255, first.segment_count);
    assert_eq!(65_025, first.body_len);
    assert_eq!(OGG_FLAG_BOS, first.flags & OGG_FLAG_BOS);
    assert_eq!(0, first.flags & OGG_FLAG_CONTINUED);
    assert_eq!(u64::MAX, first.granulepos);

    assert_eq!(20, second.segment_count);
    assert_eq!(4_975, second.body_len);
    assert_eq!(OGG_FLAG_CONTINUED, second.flags & OGG_FLAG_CONTINUED);
    assert_eq!(0, second.flags & OGG_FLAG_BOS);
    assert_eq!(999, second.granulepos);

    let mut reconstructed = TestBuffer::default();
    reconstructed.append(first.body);
    reconstructed.append(second.body);
    assert_eq!(payload_size, reconstructed.data.len());
    assert_eq!(payload, reconstructed.data);
    expect_no_page(&mut oggp);
}

#[test]
fn chaining_matches_ctest() {
    let mut payload = [0u8; 3];
    fill_payload(&mut payload, 5);
    let mut oggp = OggPacker::new(1001);

    commit_payload(&mut oggp, &payload, 1, false);
    assert!(oggp.flush_page());
    commit_payload(&mut oggp, &payload, 2, false);
    oggp.chain(2002);
    commit_payload(&mut oggp, &payload, 1, false);
    assert!(oggp.flush_page());

    let old_page0 = oggp.get_next_page().expect("old page 0");
    let old_page1 = oggp.get_next_page().expect("old page 1");
    let new_page0 = oggp.get_next_page().expect("new page 0");
    test_page_crc_and_shape(&old_page0);
    test_page_crc_and_shape(&old_page1);
    test_page_crc_and_shape(&new_page0);
    let old0 = parse_page(&old_page0);
    let old1 = parse_page(&old_page1);
    let new0 = parse_page(&new_page0);

    assert_eq!(1, old0.granulepos);
    assert_eq!(2, old1.granulepos);
    assert_eq!(1, new0.granulepos);
    assert_eq!(0, old0.pageno);
    assert_eq!(1, old1.pageno);
    assert_eq!(0, new0.pageno);
    assert_eq!(OGG_FLAG_BOS, old0.flags & OGG_FLAG_BOS);
    assert_eq!(0, old1.flags & OGG_FLAG_BOS);
    assert_eq!(OGG_FLAG_BOS, new0.flags & OGG_FLAG_BOS);
    assert_eq!(0, new0.flags & OGG_FLAG_EOS);
    assert_eq!(2002, new0.serialno);
    expect_no_page(&mut oggp);
}

#[test]
fn queue_consumption_semantics_match_ctest() {
    let mut first_payload = [0u8; 2];
    let mut second_payload = [0u8; 3];
    fill_payload(&mut first_payload, 12);
    fill_payload(&mut second_payload, 13);
    let mut oggp = OggPacker::new(909);

    commit_payload(&mut oggp, &first_payload, 5, false);
    assert!(oggp.flush_page());
    commit_payload(&mut oggp, &second_payload, 9, false);
    assert!(oggp.flush_page());

    let first_page = oggp.get_next_page().expect("first page");
    let second_page = oggp.get_next_page().expect("second page");
    expect_no_page(&mut oggp);

    let mut first_snapshot = TestBuffer::default();
    first_snapshot.append(&first_page);
    let mut second_snapshot = TestBuffer::default();
    second_snapshot.append(&second_page);

    assert_eq!(first_snapshot.data, first_page);
    assert_eq!(second_snapshot.data, second_page);
    test_page_crc_and_shape(&first_page);
    test_page_crc_and_shape(&second_page);
    let first = parse_page(&first_page);
    let second = parse_page(&second_page);
    assert_eq!(first_payload.as_slice(), first.body);
    assert_eq!(second_payload.as_slice(), second.body);
}
