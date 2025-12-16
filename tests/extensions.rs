use mousiki::extensions::{
    ExtensionError, OpusExtensionData, opus_packet_extensions_count,
    opus_packet_extensions_count_ext, opus_packet_extensions_generate,
    opus_packet_extensions_parse, opus_packet_extensions_parse_ext,
};
use mousiki::packet::opus_packet_parse_impl;
use mousiki::repacketizer::OpusRepacketizer;

fn extension_with_len<'a>(id: u8, frame: i32, data: &'a [u8], len: i32) -> OpusExtensionData<'a> {
    OpusExtensionData {
        id,
        frame,
        data,
        len,
    }
}

#[derive(Clone)]
struct FastRand {
    rz: u32,
    rw: u32,
}

impl FastRand {
    fn new(seed: u32) -> Self {
        Self { rz: seed, rw: seed }
    }

    fn next(&mut self) -> u32 {
        self.rz = 36969u32
            .wrapping_mul(self.rz & 0xFFFF)
            .wrapping_add(self.rz >> 16);
        self.rw = 18000u32
            .wrapping_mul(self.rw & 0xFFFF)
            .wrapping_add(self.rw >> 16);
        (self.rz << 16).wrapping_add(self.rw)
    }
}

const NB_RANDOM_EXTENSIONS: usize = 5_000;
const MAX_EXTENSION_SIZE: usize = 200;
const MAX_NB_EXTENSIONS: usize = (MAX_EXTENSION_SIZE - 1) * 48;
const NB_EXT: usize = 13;

#[test]
fn extensions_generate_success() {
    let ext = [
        extension_with_len(3, 0, b"a", 1),
        extension_with_len(32, 10, b"DRED", 4),
        extension_with_len(33, 1, b"NOT DRED", 8),
        extension_with_len(4, 4, &[], 0),
    ];

    let mut packet = [0u8; 32];
    let result = opus_packet_extensions_generate(Some(&mut packet), 27, &ext, 11, true).unwrap();
    assert_eq!(result, 27);

    let mut offset = 0usize;
    assert_eq!(&packet[offset..offset + 4], &[1, 1, 1, 1]);
    offset += 4;

    assert_eq!(packet[offset] >> 1, 3);
    assert_eq!(packet[offset] & 0x01, 1);
    assert_eq!(packet[offset + 1], b'a');
    offset += 2;

    assert_eq!(packet[offset], 0x02);
    offset += 1;
    assert_eq!(packet[offset] >> 1, 33);
    assert_eq!(packet[offset] & 0x01, 1);
    assert_eq!(packet[offset + 1], ext[2].len as u8);
    offset += 2;
    assert_eq!(&packet[offset..offset + ext[2].len as usize], ext[2].data);
    offset += ext[2].len as usize;

    assert_eq!(packet[offset], 0x03);
    assert_eq!(packet[offset + 1], 0x03);
    offset += 2;

    assert_eq!(packet[offset] >> 1, 4);
    assert_eq!(packet[offset] & 0x01, 0);
    offset += 1;

    assert_eq!(packet[offset], 0x03);
    assert_eq!(packet[offset + 1], 0x06);
    offset += 2;

    assert_eq!(packet[offset] >> 1, 32);
    assert_eq!(packet[offset] & 0x01, 0);
    offset += 1;
    assert_eq!(&packet[offset..offset + ext[1].len as usize], ext[1].data);
}

#[test]
fn extensions_generate_zero() {
    let mut packet = [0u8; 32];
    let result = opus_packet_extensions_generate(Some(&mut packet), 0, &[], 0, true).unwrap();
    assert_eq!(result, 0);
}

#[test]
fn extensions_generate_no_padding() {
    let ext = [
        extension_with_len(3, 0, b"a", 1),
        extension_with_len(32, 10, b"DRED", 4),
        extension_with_len(33, 1, b"NOT DRED", 8),
        extension_with_len(4, 4, &[], 0),
    ];

    let mut packet = [0u8; 32];
    let capacity = packet.len();
    let result =
        opus_packet_extensions_generate(Some(&mut packet), capacity, &ext, 11, false).unwrap();
    assert_eq!(result, 23);
}

#[test]
fn extensions_generate_fail() {
    let ext = [
        extension_with_len(3, 0, b"a", 1),
        extension_with_len(32, 10, b"DRED", 4),
        extension_with_len(33, 1, b"NOT DRED", 8),
        extension_with_len(4, 4, &[], 0),
    ];

    let mut packet = [0u8; 100];
    let capacity = packet.len();

    for len in 0..23 {
        for byte in packet.iter_mut().skip(len) {
            *byte = 0xFE;
        }
        let err = opus_packet_extensions_generate(Some(&mut packet), len, &ext, 11, true)
            .expect_err("generation should fail when the buffer is too small");
        assert_eq!(err, ExtensionError::BufferTooSmall);
        assert!(packet[len..].iter().all(|&b| b == 0xFE));
    }

    let id_too_big = [extension_with_len(255, 0, b"a", 1)];
    let err = opus_packet_extensions_generate(Some(&mut packet), capacity, &id_too_big, 11, true)
        .expect_err("id over 127 must be rejected");
    assert_eq!(err, ExtensionError::BadArgument);

    let id_too_small = [extension_with_len(2, 0, b"a", 1)];
    let err = opus_packet_extensions_generate(Some(&mut packet), capacity, &id_too_small, 11, true)
        .expect_err("id under 3 must be rejected");
    assert_eq!(err, ExtensionError::BadArgument);

    let frame_too_big = [extension_with_len(33, 11, b"a", 1)];
    let err =
        opus_packet_extensions_generate(Some(&mut packet), capacity, &frame_too_big, 49, true)
            .expect_err("nb_frames over 48 must be rejected");
    assert_eq!(err, ExtensionError::BadArgument);

    let frame_too_small = [extension_with_len(33, -1, b"a", 1)];
    let err =
        opus_packet_extensions_generate(Some(&mut packet), capacity, &frame_too_small, 11, true)
            .expect_err("negative frame index must be rejected");
    assert_eq!(err, ExtensionError::BadArgument);

    let frame_index_too_big = [extension_with_len(33, 11, b"a", 1)];
    let err = opus_packet_extensions_generate(
        Some(&mut packet),
        capacity,
        &frame_index_too_big,
        11,
        true,
    )
    .expect_err("frame index beyond nb_frames must be rejected");
    assert_eq!(err, ExtensionError::BadArgument);

    let size_too_big = [extension_with_len(3, 0, b"abcd", 4)];
    let err = opus_packet_extensions_generate(Some(&mut packet), capacity, &size_too_big, 1, true)
        .expect_err("short extension must be length 0 or 1");
    assert_eq!(err, ExtensionError::BadArgument);

    let neg_size = [extension_with_len(3, 0, &[], -4)];
    let err = opus_packet_extensions_generate(Some(&mut packet), capacity, &neg_size, 1, true)
        .expect_err("negative short length must be rejected");
    assert_eq!(err, ExtensionError::BadArgument);

    let neg_size_long = [extension_with_len(33, 0, &[], -4)];
    let err = opus_packet_extensions_generate(Some(&mut packet), capacity, &neg_size_long, 1, true)
        .expect_err("negative long length must be rejected");
    assert_eq!(err, ExtensionError::BadArgument);
}

#[test]
fn extensions_parse_success() {
    let ext = [
        extension_with_len(3, 0, b"a", 1),
        extension_with_len(32, 10, b"DRED", 4),
        extension_with_len(33, 1, b"NOT DRED", 8),
        extension_with_len(4, 4, &[], 0),
    ];
    let mut ext_out = [OpusExtensionData::default(); 10];
    let mut packet = [0u8; 32];
    let capacity = packet.len();

    let len = opus_packet_extensions_generate(Some(&mut packet), capacity, &ext, 11, true).unwrap();
    assert_eq!(len, 32);

    let count = opus_packet_extensions_count(&packet, len, 11).unwrap();
    assert_eq!(count, 4);

    let parsed = opus_packet_extensions_parse(&packet, len, 11, &mut ext_out).unwrap();
    assert_eq!(parsed, 4);

    assert_eq!(ext_out[0].id, 3);
    assert_eq!(ext_out[0].frame, 0);
    assert_eq!(ext_out[0].len, 1);
    assert_eq!(ext_out[0].data, &ext[0].data[..1]);

    assert_eq!(ext_out[1].id, 33);
    assert_eq!(ext_out[1].frame, 1);
    assert_eq!(ext_out[1].len, 8);
    assert_eq!(ext_out[1].data, ext[2].data);

    assert_eq!(ext_out[2].id, 4);
    assert_eq!(ext_out[2].frame, 4);
    assert_eq!(ext_out[2].len, 0);

    assert_eq!(ext_out[3].id, 32);
    assert_eq!(ext_out[3].frame, 10);
    assert_eq!(ext_out[3].len, 4);
    assert_eq!(ext_out[3].data, ext[1].data);
}

#[test]
fn extensions_parse_zero() {
    let ext = [extension_with_len(32, 1, b"DRED", 4)];
    let mut packet = [0u8; 32];
    let capacity = packet.len();

    let len = opus_packet_extensions_generate(Some(&mut packet), capacity, &ext, 2, true).unwrap();
    assert_eq!(len, 32);

    let err = opus_packet_extensions_parse(&packet, len, 2, &mut []).expect_err("buffer too small");
    assert_eq!(err, ExtensionError::BufferTooSmall);
}

#[test]
fn extensions_parse_fail() {
    let ext = [
        extension_with_len(3, 0, b"a", 1),
        extension_with_len(33, 1, b"NOT DRED", 8),
        extension_with_len(4, 4, &[], 0),
        extension_with_len(32, 10, b"DRED", 4),
        extension_with_len(32, 9, b"DRED", 4),
        extension_with_len(4, 9, b"b", 1),
        extension_with_len(4, 10, b"c", 1),
    ];

    let mut ext_out = [OpusExtensionData::default(); 10];
    let mut packet = [0u8; 32];
    let capacity = packet.len();

    let mut len =
        opus_packet_extensions_generate(Some(&mut packet), capacity, &ext[..4], 11, false).unwrap();
    packet[4] = 255;
    let err = opus_packet_extensions_parse(&packet, len, 11, &mut ext_out)
        .expect_err("invalid length must be rejected");
    assert_eq!(err, ExtensionError::InvalidPacket);
    let count = opus_packet_extensions_count(&packet, len, 11)
        .expect("count should still succeed despite invalid length marker");
    assert_eq!(count, 1);

    len =
        opus_packet_extensions_generate(Some(&mut packet), capacity, &ext[..4], 11, false).unwrap();
    ext_out = [OpusExtensionData::default(); 10];
    let err = opus_packet_extensions_parse(&packet, len, 5, &mut ext_out)
        .expect_err("too few frames must be rejected");
    assert_eq!(err, ExtensionError::InvalidPacket);

    let count = opus_packet_extensions_count(&packet, len, 5).unwrap();
    assert_eq!(count, 3);

    ext_out = [OpusExtensionData::default(); 10];
    packet[14] = 255;
    let err = opus_packet_extensions_parse(&packet, len, 11, &mut ext_out)
        .expect_err("invalid frame increment must be rejected");
    assert_eq!(err, ExtensionError::InvalidPacket);

    let count = opus_packet_extensions_count(&packet, len, 11).unwrap();
    assert_eq!(count, 2);

    ext_out = [OpusExtensionData::default(); 10];
    let err = opus_packet_extensions_parse(&packet, len, 11, &mut ext_out[..1])
        .expect_err("insufficient output capacity must fail");
    assert_eq!(err, ExtensionError::BufferTooSmall);

    len =
        opus_packet_extensions_generate(Some(&mut packet), capacity, &ext[..7], 11, false).unwrap();
    len -= 5;
    ext_out = [OpusExtensionData::default(); 10];
    let err = opus_packet_extensions_parse(&packet, len, 11, &mut ext_out)
        .expect_err("truncated repeated extension must fail");
    assert_eq!(err, ExtensionError::InvalidPacket);
    let count = opus_packet_extensions_count(&packet, len, 11).unwrap();
    assert_eq!(count, 5);

    let lensize = ((1u32 << 31) / 255 + 1) as usize;
    let mut buf = vec![0u8; lensize + 1];
    buf[0] = 33 << 1 | 1;
    buf[1..lensize].fill(0xFF);
    buf[lensize] = 0xFE;
    let err = opus_packet_extensions_parse(&buf, buf.len(), 1, &mut ext_out)
        .expect_err("overflowing length must be rejected");
    assert_eq!(err, ExtensionError::InvalidPacket);
}

fn check_ext_data(ext_in: &[OpusExtensionData<'_>], ext_out: &[OpusExtensionData<'_>]) {
    let mut prev_frame = -1;
    let mut j = 0usize;
    for ext in ext_out.iter() {
        assert!(
            ext.frame >= prev_frame,
            "expected parsed extensions to be returned in frame order"
        );
        if ext.frame > prev_frame {
            j = 0;
        }
        while j < ext_in.len() && ext_in[j].frame != ext.frame {
            j += 1;
        }
        assert!(j < ext_in.len(), "expected matching frame in source data");
        assert_eq!(ext_in[j].id, ext.id);
        assert_eq!(ext_in[j].len, ext.len);
        assert_eq!(ext_in[j].data, ext.data);
        prev_frame = ext.frame;
        j += 1;
    }
}

#[test]
fn extensions_repeating() {
    let ext = [
        extension_with_len(3, 0, b"a", 1),
        extension_with_len(3, 1, b"b", 1),
        extension_with_len(3, 2, b"c", 1),
        extension_with_len(4, 0, b"d", 1),
        extension_with_len(4, 1, &[], 0),
        extension_with_len(4, 2, &[], 0),
        extension_with_len(32, 2, b"DRED2", 5),
        extension_with_len(32, 1, b"DRED", 4),
        extension_with_len(5, 1, &[], 0),
        extension_with_len(5, 2, &[], 0),
        extension_with_len(6, 2, b"f", 1),
        extension_with_len(6, 1, b"e", 1),
        extension_with_len(32, 2, b"DREDthree", 9),
    ];
    let encoded_len = [0, 2, 5, 5, 7, 9, 10, 16, 21, 23, 22, 26, 25, 37];

    for nb_ext in 0..=NB_EXT {
        let mut packet = [0u8; 64];
        let capacity = packet.len();
        let mut nb_frame_exts = [0i32; 48];
        let mut ext_out = vec![OpusExtensionData::default(); NB_EXT];
        let len =
            opus_packet_extensions_generate(Some(&mut packet), capacity, &ext[..nb_ext], 3, false)
                .unwrap();
        assert_eq!(len, encoded_len[nb_ext]);
        let count = opus_packet_extensions_count_ext(&packet, len, &mut nb_frame_exts, 3).unwrap();
        assert_eq!(count, nb_ext);
        let parsed =
            opus_packet_extensions_parse_ext(&packet, len, 3, &mut ext_out[..], &nb_frame_exts)
                .unwrap();
        assert_eq!(parsed, nb_ext);
        check_ext_data(&ext[..nb_ext], &ext_out[..nb_ext]);
        ext_out = vec![OpusExtensionData::default(); NB_EXT];

        let mut len = len;
        if nb_ext == 6 {
            packet[len] = 2 << 1 | 0;
            packet[len + 1] = 3 << 1 | 0;
            len += 2;
        } else if nb_ext == 8 {
            packet.copy_within(15..len, 16);
            packet[15] = 0x01;
            len += 1;
        } else if nb_ext == 10 {
            packet.copy_within(15..len, 17);
            packet[15] = 0x03;
            packet[16] = 0;
            len += 2;
        } else if nb_ext == 13 {
            packet[26] = 2 << 1 | 0;
        } else {
            continue;
        }
        let count = opus_packet_extensions_count_ext(&packet, len, &mut nb_frame_exts, 3).unwrap();
        assert_eq!(count, nb_ext);
        let parsed =
            opus_packet_extensions_parse_ext(&packet, len, 3, &mut ext_out[..], &nb_frame_exts)
                .unwrap();
        assert_eq!(parsed, nb_ext);
        check_ext_data(&ext[..nb_ext], &ext_out[..nb_ext]);
        ext_out = vec![OpusExtensionData::default(); NB_EXT];

        if nb_ext == 8 {
            packet.copy_within(9..len, 10);
            packet[9] = 2 << 1 | 1;
            len += 1;
            packet.copy_within(5..len, 6);
            packet[5] = 2 << 1 | 1;
            len += 1;

            let count =
                opus_packet_extensions_count_ext(&packet, len, &mut nb_frame_exts, 3).unwrap();
            assert_eq!(count, nb_ext);
            let parsed =
                opus_packet_extensions_parse_ext(&packet, len, 3, &mut ext_out[..], &nb_frame_exts)
                    .unwrap();
            assert_eq!(parsed, nb_ext);
            check_ext_data(&ext[..nb_ext], &ext_out[..nb_ext]);
        }
    }
}

#[test]
fn random_extensions_parse() {
    let mut rng = FastRand::new(0x1234_5678);
    for _ in 0..NB_RANDOM_EXTENSIONS {
        let mut ext_out = vec![OpusExtensionData::default(); MAX_NB_EXTENSIONS];
        let len = (rng.next() as usize) % (MAX_EXTENSION_SIZE + 1);
        let mut payload = vec![0u8; len];
        for byte in &mut payload {
            *byte = rng.next() as u8;
        }
        let nb_ext = (rng.next() as usize) % (MAX_NB_EXTENSIONS + 1);
        let nb_frames = (rng.next() as usize % 48) + 1;
        let mut capacity = nb_ext.min(MAX_NB_EXTENSIONS);
        let result =
            opus_packet_extensions_parse(&payload, len, nb_frames, &mut ext_out[..capacity]);
        match result {
            Ok(parsed) => capacity = parsed,
            Err(ExtensionError::BufferTooSmall) | Err(ExtensionError::InvalidPacket) => {}
            Err(err) => panic!("unexpected parse error: {err:?}"),
        }
        for ext in ext_out.iter().take(capacity).filter(|e| e.id != 0) {
            assert!(
                (0..nb_frames as i32).contains(&ext.frame),
                "frame should be within range"
            );
            assert!(
                (2..=127).contains(&ext.id),
                "id should be between 2 and 127"
            );
            assert!(
                {
                    let data_ptr = ext.data.as_ptr() as usize;
                    let start = payload.as_ptr() as usize;
                    let end = start + payload.len();
                    data_ptr >= start && data_ptr <= end
                },
                "data must point inside the payload"
            );
        }
        if let Ok(parsed) = result {
            let mut payload2 = [0u8; MAX_EXTENSION_SIZE + 1];
            let payload2_capacity = payload2.len();
            let len2 = opus_packet_extensions_generate(
                Some(&mut payload2),
                payload2_capacity,
                &ext_out[..parsed],
                nb_frames,
                false,
            )
            .unwrap();
            let mut nb_frame_exts = [0i32; 48];
            let count2 =
                opus_packet_extensions_count_ext(&payload2, len2, &mut nb_frame_exts, nb_frames)
                    .unwrap();
            assert_eq!(count2, parsed);
            let mut ext_out2 = vec![OpusExtensionData::default(); MAX_NB_EXTENSIONS];
            let parsed2 = opus_packet_extensions_parse_ext(
                &payload2,
                len2,
                nb_frames,
                &mut ext_out2,
                &nb_frame_exts,
            )
            .unwrap();
            assert_eq!(parsed2, parsed);
            check_ext_data(&ext_out[..parsed], &ext_out2[..parsed]);
        }
    }
}

#[test]
fn opus_repacketizer_out_range_impl_extensions() {
    let mut rp = OpusRepacketizer::new();
    let mut packet = [0u8; 1024];
    let mut packet_out = [0u8; 1024];
    let capacity = packet.len();
    let input_ext = [
        extension_with_len(33, 0, b"abcdefg", 7),
        extension_with_len(100, 0, b"uvwxyz", 6),
    ];

    packet.fill(0);
    packet[0] = (15 << 3) | 3;
    packet[1] = 1 << 6 | 1;

    let len =
        opus_packet_extensions_generate(Some(&mut packet[4..]), capacity - 4, &input_ext, 1, false)
            .unwrap();
    packet[2] = len as u8;

    rp.opus_repacketizer_cat(&packet[..4 + len], 4 + len)
        .expect("first frame should be accepted");
    packet[1] = 1;
    rp.opus_repacketizer_cat(&packet[..4], 4)
        .expect("second frame should be accepted");
    packet[1] = 1 << 6 | 1;
    rp.opus_repacketizer_cat(&packet[..4 + len], 4 + len)
        .expect("third frame should be accepted");

    assert_eq!(rp.opus_repacketizer_get_nb_frames(), 3);
    let out_capacity = packet_out.len();
    let written = rp
        .opus_repacketizer_out(&mut packet_out, out_capacity)
        .expect("out should succeed");
    let parsed = opus_packet_parse_impl(&packet_out, written, false).unwrap();
    let padding = parsed.padding;
    let mut ext_out = [OpusExtensionData::default(); 10];
    let nb_ext = opus_packet_extensions_parse(padding, padding.len(), 3, &mut ext_out).unwrap();
    assert_eq!(nb_ext, 4);

    let mut first_count = 0;
    let mut second_count = 0;
    for (idx, ext) in ext_out.iter().take(nb_ext).enumerate() {
        match ext.id {
            33 => {
                assert_eq!(ext.len, input_ext[0].len);
                assert_eq!(ext.data, input_ext[0].data);
                first_count += 1;
            }
            100 => {
                assert_eq!(ext.len, input_ext[1].len);
                assert_eq!(ext.data, input_ext[1].data);
                second_count += 1;
            }
            _ => panic!("unexpected extension id {}", ext.id),
        }
        if idx < 2 {
            assert_eq!(ext.frame, 0);
        } else {
            assert_eq!(ext.frame, 2);
        }
    }
    assert_eq!(first_count, 2);
    assert_eq!(second_count, 2);
}
