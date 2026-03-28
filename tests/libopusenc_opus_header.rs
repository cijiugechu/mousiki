#![cfg(feature = "libopusenc")]

use mousiki::libopusenc::opus_header::{
    CommentError, OpusHeader, comment_add, comment_init, comment_pad, opus_header_get_size,
    opus_header_to_packet,
};
use mousiki::c_style_api::projection::opus_projection_ambisonics_encoder_create;

#[derive(Debug)]
struct ParsedCommentPacket<'a> {
    vendor: &'a [u8],
    comment_count: usize,
    comments: Vec<&'a [u8]>,
}

#[derive(Debug)]
struct ParsedHeaderPacket<'a> {
    version: u8,
    channels: u8,
    preskip: u16,
    input_sample_rate: u32,
    gain_raw: u16,
    channel_mapping: u8,
    nb_streams: Option<u8>,
    nb_coupled: Option<u8>,
    stream_map: &'a [u8],
    matrix: &'a [u8],
}

fn parse_comment_packet(packet: &[u8]) -> ParsedCommentPacket<'_> {
    assert!(packet.len() >= 16);
    assert_eq!(&packet[..8], b"OpusTags");
    let vendor_length = u32::from_le_bytes(packet[8..12].try_into().unwrap()) as usize;
    let vendor = &packet[12..12 + vendor_length];
    let mut offset = 12 + vendor_length;
    let comment_count = u32::from_le_bytes(packet[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let mut comments = Vec::with_capacity(comment_count);
    for _ in 0..comment_count {
        let len = u32::from_le_bytes(packet[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        comments.push(&packet[offset..offset + len]);
        offset += len;
    }
    ParsedCommentPacket {
        vendor,
        comment_count,
        comments,
    }
}

fn parse_header_packet(packet: &[u8]) -> ParsedHeaderPacket<'_> {
    assert!(packet.len() >= 19);
    assert_eq!(&packet[..8], b"OpusHead");
    let version = packet[8];
    let channels = packet[9];
    let preskip = u16::from_le_bytes(packet[10..12].try_into().unwrap());
    let input_sample_rate = u32::from_le_bytes(packet[12..16].try_into().unwrap());
    let gain_raw = u16::from_le_bytes(packet[16..18].try_into().unwrap());
    let channel_mapping = packet[18];
    if channel_mapping == 0 {
        ParsedHeaderPacket {
            version,
            channels,
            preskip,
            input_sample_rate,
            gain_raw,
            channel_mapping,
            nb_streams: None,
            nb_coupled: None,
            stream_map: &[],
            matrix: &[],
        }
    } else {
        let nb_streams = packet[19];
        let nb_coupled = packet[20];
        let payload = &packet[21..];
        if channel_mapping == 3 {
            ParsedHeaderPacket {
                version,
                channels,
                preskip,
                input_sample_rate,
                gain_raw,
                channel_mapping,
                nb_streams: Some(nb_streams),
                nb_coupled: Some(nb_coupled),
                stream_map: &[],
                matrix: payload,
            }
        } else {
            ParsedHeaderPacket {
                version,
                channels,
                preskip,
                input_sample_rate,
                gain_raw,
                channel_mapping,
                nb_streams: Some(nb_streams),
                nb_coupled: Some(nb_coupled),
                stream_map: payload,
                matrix: &[],
            }
        }
    }
}

fn fill_stream_map(stream_map: &mut [u8], channels: usize) {
    for (i, item) in stream_map.iter_mut().take(channels).enumerate() {
        *item = ((i * 3) % channels) as u8;
    }
}

#[test]
fn comment_init_matches_ctest() {
    let comments = comment_init("libopusenc-test").expect("init");
    let parsed = parse_comment_packet(&comments);
    assert_eq!(parsed.vendor, b"libopusenc-test");
    assert_eq!(parsed.comment_count, 0);

    let comments = comment_init("").expect("init");
    let parsed = parse_comment_packet(&comments);
    assert_eq!(parsed.vendor, b"");
    assert_eq!(parsed.comment_count, 0);
}

#[test]
fn comment_add_matches_ctest() {
    let mut comments = comment_init("vendor").expect("init");
    comment_add(&mut comments, Some("ARTIST"), "Name").expect("artist");
    comment_add(&mut comments, None, "TITLE=Track").expect("title");
    comment_add(&mut comments, Some("ALBUM"), "Record").expect("album");
    let parsed = parse_comment_packet(&comments);

    assert_eq!(parsed.comment_count, 3);
    assert_eq!(parsed.comments[0], b"ARTIST=Name");
    assert_eq!(parsed.comments[1], b"TITLE=Track");
    assert_eq!(parsed.comments[2], b"ALBUM=Record");
}

#[test]
fn comment_pad_matches_ctest() {
    let mut comments = comment_init("vendor").expect("init");
    comment_add(&mut comments, Some("ARTIST"), "Name").expect("artist");
    let original_len = comments.len();

    comment_pad(&mut comments, 0).expect("pad zero");
    assert_eq!(original_len, comments.len());

    comment_pad(&mut comments, -5).expect("pad negative");
    assert_eq!(original_len, comments.len());

    comment_pad(&mut comments, 10).expect("pad positive");
    assert_eq!(254, comments.len());
    assert!(comments[original_len..].iter().all(|&byte| byte == 0));

    let parsed = parse_comment_packet(&comments[..original_len]);
    assert_eq!(parsed.comment_count, 1);
}

#[test]
fn header_mapping_zero_matches_ctest() {
    let header = OpusHeader {
        version: 9,
        channels: 2,
        preskip: 312,
        input_sample_rate: 44_100,
        gain: -15,
        channel_mapping: 0,
        ..Default::default()
    };
    assert_eq!(23, opus_header_get_size(&header));

    let mut packet = [0u8; 64];
    let packet_size = opus_header_to_packet(&header, &mut packet, None).expect("packet");
    assert_eq!(19, packet_size);

    let parsed = parse_header_packet(&packet[..packet_size]);
    assert_eq!(1, parsed.version);
    assert_eq!(2, parsed.channels);
    assert_eq!(312, parsed.preskip);
    assert_eq!(44_100, parsed.input_sample_rate);
    assert_eq!((-15i16) as u16, parsed.gain_raw);
    assert_eq!(0, parsed.channel_mapping);
    assert!(parsed.nb_streams.is_none());
    assert!(parsed.nb_coupled.is_none());
    assert!(parsed.stream_map.is_empty());
}

#[test]
fn header_mapped_multistream_matches_ctest() {
    let mut header = OpusHeader {
        version: 5,
        channels: 3,
        preskip: 120,
        input_sample_rate: 48_000,
        gain: 256,
        channel_mapping: 255,
        nb_streams: 2,
        nb_coupled: 1,
        ..Default::default()
    };
    fill_stream_map(&mut header.stream_map, header.channels as usize);

    let expected_size = opus_header_get_size(&header);
    assert_eq!(24, expected_size);

    let mut packet = [0u8; 64];
    let packet_size = opus_header_to_packet(&header, &mut packet, None).expect("packet");
    assert_eq!(expected_size, packet_size);

    let parsed = parse_header_packet(&packet[..packet_size]);
    assert_eq!(1, parsed.version);
    assert_eq!(3, parsed.channels);
    assert_eq!(120, parsed.preskip);
    assert_eq!(48_000, parsed.input_sample_rate);
    assert_eq!(256, parsed.gain_raw);
    assert_eq!(255, parsed.channel_mapping);
    assert_eq!(Some(2), parsed.nb_streams);
    assert_eq!(Some(1), parsed.nb_coupled);
    assert_eq!(&header.stream_map[..3], parsed.stream_map);
}

#[test]
fn header_short_buffer_failures_match_ctest() {
    let header = OpusHeader {
        channels: 2,
        preskip: 1,
        input_sample_rate: 48_000,
        channel_mapping: 0,
        ..Default::default()
    };
    let mut packet = [0u8; 128];
    assert!(opus_header_to_packet(&header, &mut packet[..18], None).is_err());

    let mut header = OpusHeader {
        channels: 4,
        preskip: 1,
        input_sample_rate: 48_000,
        channel_mapping: 1,
        nb_streams: 2,
        nb_coupled: 2,
        ..Default::default()
    };
    fill_stream_map(&mut header.stream_map, header.channels as usize);
    let short_len = opus_header_get_size(&header) - 1;
    assert!(opus_header_to_packet(&header, &mut packet[..short_len], None).is_err());
}

#[test]
fn header_projection_matches_ctest() {
    let (encoder, streams, coupled_streams) =
        opus_projection_ambisonics_encoder_create(48_000, 4, 3, 2_049)
            .expect("projection encoder");
    let layout = *encoder.projection_layout();

    let header = OpusHeader {
        version: 7,
        channels: 4,
        preskip: 384,
        input_sample_rate: 96_000,
        gain: -64,
        channel_mapping: 3,
        nb_streams: streams as i32,
        nb_coupled: coupled_streams as i32,
        ..Default::default()
    };

    let expected_size = opus_header_get_size(&header);
    assert_eq!(
        21 + header.channels as usize * (header.nb_streams + header.nb_coupled) as usize * 2,
        expected_size
    );
    let matrix_size = layout.demixing_subset_size_bytes().expect("matrix size");
    assert_eq!(expected_size - 21, matrix_size);

    let mut packet = vec![0u8; expected_size];
    let packet_size = opus_header_to_packet(&header, &mut packet, Some(&layout)).expect("packet");
    assert_eq!(expected_size, packet_size);

    let parsed = parse_header_packet(&packet[..packet_size]);
    assert_eq!(1, parsed.version);
    assert_eq!(4, parsed.channels);
    assert_eq!(384, parsed.preskip);
    assert_eq!(96_000, parsed.input_sample_rate);
    assert_eq!((header.gain + layout.demixing.gain_db) as i16 as u16, parsed.gain_raw);
    assert_eq!(3, parsed.channel_mapping);
    assert_eq!(Some(streams as u8), parsed.nb_streams);
    assert_eq!(Some(coupled_streams as u8), parsed.nb_coupled);
    assert_eq!(matrix_size, parsed.matrix.len());

    assert!(opus_header_to_packet(&header, &mut packet[..expected_size - 1], Some(&layout)).is_err());
}

#[test]
fn comment_add_rejects_non_opus_tags_buffers() {
    let mut comments = vec![0u8; 4];
    let err = comment_add(&mut comments, Some("ARTIST"), "Name").unwrap_err();
    assert_eq!(CommentError::AllocationFailed, err);
}

#[test]
fn projection_header_requires_projection_layout() {
    let header = OpusHeader {
        channels: 4,
        preskip: 384,
        input_sample_rate: 96_000,
        gain: -64,
        channel_mapping: 3,
        nb_streams: 4,
        nb_coupled: 0,
        ..Default::default()
    };
    let mut packet = [0u8; 64];
    assert!(opus_header_to_packet(&header, &mut packet, None).is_err());
}
