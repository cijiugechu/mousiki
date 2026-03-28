#![cfg(feature = "libopusenc")]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use mousiki::libopusenc::encoder::OpeError;
use mousiki::libopusenc::picture::{
    parse_picture_specification, parse_picture_specification_from_memory,
};

static PNG_32X32_RGBA: &[u8] = &[
    0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H', b'D',
    b'R', 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x20, 0x08, 0x06, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
];

static PNG_16X16_RGBA: &[u8] = &[
    0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H', b'D',
    b'R', 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x08, 0x06, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
];

static GIF_2X3: &[u8] = &[
    b'G', b'I', b'F', b'8', b'9', b'a', 0x02, 0x00, 0x03, 0x00, 0x81, 0x00, 0x00, 0x00,
];

static JPEG_32X16: &[u8] = &[
    0xff, 0xd8, 0xff, 0xc0, 0x00, 0x08, 0x08, 0x00, 0x10, 0x00, 0x20, 0x03, 0xff, 0xd9,
];

static INVALID_BYTES: &[u8] = &[0x00, 0x11, 0x22, 0x33, 0x44];

#[derive(Debug)]
struct ParsedPictureBlock {
    picture_type: u32,
    mime_len: u32,
    mime: Vec<u8>,
    description_len: u32,
    description: Vec<u8>,
    width: u32,
    height: u32,
    depth: u32,
    colors: u32,
    data_len: u32,
    data: Vec<u8>,
}

fn b64_value(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn decode_base64(input: &str) -> Vec<u8> {
    assert_eq!(0, input.len() % 4);
    let mut decoded = Vec::with_capacity(input.len() / 4 * 3);
    let bytes = input.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let a = b64_value(bytes[i]).unwrap();
        let b = b64_value(bytes[i + 1]).unwrap();
        let c = if bytes[i + 2] == b'=' {
            None
        } else {
            Some(b64_value(bytes[i + 2]).unwrap())
        };
        let d = if bytes[i + 3] == b'=' {
            None
        } else {
            Some(b64_value(bytes[i + 3]).unwrap())
        };
        decoded.push((a << 2) | (b >> 4));
        if let Some(c) = c {
            decoded.push(((b & 0x0f) << 4) | (c >> 2));
            if let Some(d) = d {
                decoded.push(((c & 0x03) << 6) | d);
            }
        }
        i += 4;
    }
    decoded
}

fn read_u32be(data: &[u8]) -> u32 {
    u32::from_be_bytes([data[0], data[1], data[2], data[3]])
}

fn parse_picture_block(data: &[u8]) -> ParsedPictureBlock {
    let mut offset = 0usize;
    let picture_type = read_u32be(&data[offset..offset + 4]);
    offset += 4;
    let mime_len = read_u32be(&data[offset..offset + 4]);
    offset += 4;
    let mime = data[offset..offset + mime_len as usize].to_vec();
    offset += mime_len as usize;
    let description_len = read_u32be(&data[offset..offset + 4]);
    offset += 4;
    let description = data[offset..offset + description_len as usize].to_vec();
    offset += description_len as usize;
    let width = read_u32be(&data[offset..offset + 4]);
    offset += 4;
    let height = read_u32be(&data[offset..offset + 4]);
    offset += 4;
    let depth = read_u32be(&data[offset..offset + 4]);
    offset += 4;
    let colors = read_u32be(&data[offset..offset + 4]);
    offset += 4;
    let data_len = read_u32be(&data[offset..offset + 4]);
    offset += 4;
    let picture_data = data[offset..offset + data_len as usize].to_vec();

    ParsedPictureBlock {
        picture_type,
        mime_len,
        mime,
        description_len,
        description,
        width,
        height,
        depth,
        colors,
        data_len,
        data: picture_data,
    }
}

fn parse_picture_b64(base64: &str) -> ParsedPictureBlock {
    let decoded = decode_base64(base64);
    parse_picture_block(&decoded)
}

static UNIQUE_SUFFIX: AtomicU64 = AtomicU64::new(0);

fn create_temp_path() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let suffix = UNIQUE_SUFFIX.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("mousiki-libopusenc-picture-{unique}-{suffix}.bin"))
}

#[test]
fn parse_png_from_memory_matches_ctest() {
    let mut seen_file_icons = 0;
    let base64 = parse_picture_specification_from_memory(
        PNG_32X32_RGBA,
        -1,
        Some("cover"),
        &mut seen_file_icons,
    )
    .expect("parse png");
    assert_eq!(0, seen_file_icons);
    let parsed = parse_picture_b64(&base64);
    assert_eq!(3, parsed.picture_type);
    assert_eq!(9, parsed.mime_len);
    assert_eq!(b"image/png", parsed.mime.as_slice());
    assert_eq!(5, parsed.description_len);
    assert_eq!(b"cover", parsed.description.as_slice());
    assert_eq!(32, parsed.width);
    assert_eq!(32, parsed.height);
    assert_eq!(32, parsed.depth);
    assert_eq!(0, parsed.colors);
    assert_eq!(PNG_32X32_RGBA.len() as u32, parsed.data_len);
    assert_eq!(PNG_32X32_RGBA, parsed.data.as_slice());
}

#[test]
fn parse_gif_and_jpeg_from_memory_match_ctest() {
    let mut seen_file_icons = 0;
    let base64 = parse_picture_specification_from_memory(GIF_2X3, 4, None, &mut seen_file_icons)
        .expect("parse gif");
    let parsed = parse_picture_b64(&base64);
    assert_eq!(4, parsed.picture_type);
    assert_eq!(b"image/gif", parsed.mime.as_slice());
    assert_eq!(0, parsed.description_len);
    assert_eq!(2, parsed.width);
    assert_eq!(3, parsed.height);
    assert_eq!(24, parsed.depth);
    assert_eq!(4, parsed.colors);

    let base64 =
        parse_picture_specification_from_memory(JPEG_32X16, 5, Some("jpeg"), &mut seen_file_icons)
            .expect("parse jpeg");
    let parsed = parse_picture_b64(&base64);
    assert_eq!(5, parsed.picture_type);
    assert_eq!(b"image/jpeg", parsed.mime.as_slice());
    assert_eq!(4, parsed.description_len);
    assert_eq!(b"jpeg", parsed.description.as_slice());
    assert_eq!(32, parsed.width);
    assert_eq!(16, parsed.height);
    assert_eq!(24, parsed.depth);
    assert_eq!(0, parsed.colors);
}

#[test]
fn parse_png_from_file_matches_ctest() {
    let path = create_temp_path();
    fs::write(&path, PNG_32X32_RGBA).expect("write file");
    let mut seen_file_icons = 0;
    let base64 = parse_picture_specification(
        path.to_str().unwrap(),
        3,
        Some("file"),
        &mut seen_file_icons,
    )
    .expect("parse file");
    fs::remove_file(path).expect("remove file");
    let parsed = parse_picture_b64(&base64);
    assert_eq!(3, parsed.picture_type);
    assert_eq!(PNG_32X32_RGBA.len() as u32, parsed.data_len);
    assert_eq!(PNG_32X32_RGBA, parsed.data.as_slice());
}

#[test]
fn invalid_inputs_match_ctest() {
    let mut seen_file_icons = 0;
    assert_eq!(
        Err(OpeError::InvalidPicture),
        parse_picture_specification_from_memory(
            INVALID_BYTES,
            3,
            Some("bad"),
            &mut seen_file_icons
        )
    );
    assert_eq!(
        Err(OpeError::InvalidPicture),
        parse_picture_specification_from_memory(
            PNG_32X32_RGBA,
            21,
            Some("bad"),
            &mut seen_file_icons
        )
    );
    assert_eq!(
        Err(OpeError::CannotOpen),
        parse_picture_specification(
            "/tmp/definitely-missing-libopusenc-picture",
            3,
            Some("missing"),
            &mut seen_file_icons
        )
    );
}

#[test]
fn icon_rules_match_ctest() {
    let mut seen_file_icons = 0;
    let base64 =
        parse_picture_specification_from_memory(PNG_32X32_RGBA, 1, Some("icon"), &mut seen_file_icons)
            .expect("parse icon");
    assert_eq!(1, seen_file_icons);
    let parsed = parse_picture_b64(&base64);
    assert_eq!(1, parsed.picture_type);
    assert_eq!(b"image/png", parsed.mime.as_slice());

    assert_eq!(
        Err(OpeError::InvalidPicture),
        parse_picture_specification_from_memory(
            PNG_32X32_RGBA,
            1,
            Some("icon-again"),
            &mut seen_file_icons
        )
    );

    let mut fresh_seen_file_icons = 0;
    assert_eq!(
        Err(OpeError::InvalidIcon),
        parse_picture_specification_from_memory(
            PNG_16X16_RGBA,
            1,
            Some("bad-icon"),
            &mut fresh_seen_file_icons
        )
    );
    assert_eq!(
        Err(OpeError::InvalidIcon),
        parse_picture_specification_from_memory(
            JPEG_32X16,
            1,
            Some("jpeg-icon"),
            &mut fresh_seen_file_icons
        )
    );

    parse_picture_specification_from_memory(
        PNG_32X32_RGBA,
        2,
        Some("other-icon"),
        &mut seen_file_icons,
    )
    .expect("parse other icon");
    assert_eq!(3, seen_file_icons);
    assert_eq!(
        Err(OpeError::InvalidPicture),
        parse_picture_specification_from_memory(
            PNG_32X32_RGBA,
            2,
            Some("other-icon-again"),
            &mut seen_file_icons
        )
    );
}
