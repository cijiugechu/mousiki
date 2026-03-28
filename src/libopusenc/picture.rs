extern crate std;

use alloc::string::String;
use alloc::vec::Vec;

use std::fs;

use crate::libopusenc::{LibopusencError, PictureErrorKind};

const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const PNG_SIGNATURE: &[u8] = b"\x89PNG\x0D\x0A\x1A\x0A";

pub(crate) fn parse_picture_specification(
    filename: &str,
    picture_type: i32,
    description: Option<&str>,
    seen_file_icons: &mut i32,
) -> Result<String, LibopusencError> {
    let picture_type = normalize_picture_type(picture_type);
    if !validate_picture_type(picture_type, *seen_file_icons) {
        return Err(LibopusencError::Picture(PictureErrorKind::InvalidPicture));
    }
    let data = fs::read(filename).map_err(LibopusencError::Io)?;
    parse_picture_specification_from_memory(&data, picture_type, description, seen_file_icons)
}

pub(crate) fn parse_picture_specification_from_memory(
    mem: &[u8],
    picture_type: i32,
    description: Option<&str>,
    seen_file_icons: &mut i32,
) -> Result<String, LibopusencError> {
    let picture_type = normalize_picture_type(picture_type);
    if !validate_picture_type(picture_type, *seen_file_icons) {
        return Err(LibopusencError::Picture(PictureErrorKind::InvalidPicture));
    }
    let description = description.unwrap_or("");
    let (mime_type, width, height, depth, colors) =
        parse_image(mem).ok_or(LibopusencError::Picture(PictureErrorKind::InvalidPicture))?;
    if picture_type == 1 && (width != 32 || height != 32 || mime_type != "image/png") {
        return Err(LibopusencError::Picture(PictureErrorKind::InvalidIcon));
    }

    let mut block = Vec::with_capacity(32 + mime_type.len() + description.len() + mem.len());
    push_u32be(&mut block, picture_type as u32);
    push_u32be(&mut block, mime_type.len() as u32);
    block.extend_from_slice(mime_type.as_bytes());
    push_u32be(&mut block, description.len() as u32);
    block.extend_from_slice(description.as_bytes());
    push_u32be(&mut block, width);
    push_u32be(&mut block, height);
    push_u32be(&mut block, depth);
    push_u32be(&mut block, colors);
    push_u32be(&mut block, mem.len() as u32);
    block.extend_from_slice(mem);

    let encoded = base64_encode(&block);
    if (1..=2).contains(&picture_type) {
        *seen_file_icons |= picture_type;
    }
    Ok(encoded)
}

fn normalize_picture_type(picture_type: i32) -> i32 {
    if picture_type < 0 { 3 } else { picture_type }
}

fn validate_picture_type(picture_type: i32, seen_file_icons: i32) -> bool {
    if picture_type > 20 {
        return false;
    }
    if (1..=2).contains(&picture_type) && (seen_file_icons & picture_type) != 0 {
        return false;
    }
    true
}

fn parse_image(data: &[u8]) -> Option<(&'static str, u32, u32, u32, u32)> {
    if is_jpeg(data) {
        extract_jpeg_params(data).map(|(w, h, d, c)| ("image/jpeg", w, h, d, c))
    } else if is_png(data) {
        extract_png_params(data).map(|(w, h, d, c)| ("image/png", w, h, d, c))
    } else if is_gif(data) {
        extract_gif_params(data).map(|(w, h, d, c)| ("image/gif", w, h, d, c))
    } else {
        None
    }
}

fn is_jpeg(buf: &[u8]) -> bool {
    buf.len() >= 3 && &buf[..3] == b"\xFF\xD8\xFF"
}

fn is_png(buf: &[u8]) -> bool {
    buf.len() >= 8 && &buf[..8] == PNG_SIGNATURE
}

fn is_gif(buf: &[u8]) -> bool {
    buf.len() >= 6 && (&buf[..6] == b"GIF87a" || &buf[..6] == b"GIF89a")
}

fn extract_png_params(data: &[u8]) -> Option<(u32, u32, u32, u32)> {
    if !is_png(data) {
        return None;
    }
    let mut width = 0u32;
    let mut height = 0u32;
    let mut depth = 0u32;
    let mut colors = 0u32;
    let mut has_palette = -1i32;
    let mut offs = 8usize;
    while data.len().saturating_sub(offs) >= 12 {
        let chunk_len = read_u32be(&data[offs..offs + 4]) as usize;
        if chunk_len > data.len().saturating_sub(offs + 12) {
            break;
        }
        if chunk_len == 13 && &data[offs + 4..offs + 8] == b"IHDR" {
            width = read_u32be(&data[offs + 8..offs + 12]);
            height = read_u32be(&data[offs + 12..offs + 16]);
            let color_type = data[offs + 17];
            if color_type == 3 {
                depth = 24;
                has_palette = 1;
            } else {
                let sample_depth = data[offs + 16] as u32;
                depth = match color_type {
                    0 => sample_depth,
                    2 => sample_depth * 3,
                    4 => sample_depth * 2,
                    6 => sample_depth * 4,
                    _ => 0,
                };
                colors = 0;
                break;
            }
        } else if has_palette > 0 && &data[offs + 4..offs + 8] == b"PLTE" {
            colors = (chunk_len / 3) as u32;
            break;
        }
        offs += 12 + chunk_len;
    }
    if width == 0 || height == 0 || depth == 0 {
        Some((0, 0, 0, 0))
    } else {
        Some((width, height, depth, colors))
    }
}

fn extract_gif_params(data: &[u8]) -> Option<(u32, u32, u32, u32)> {
    if !is_gif(data) || data.len() < 14 {
        return None;
    }
    let width = u16::from_le_bytes([data[6], data[7]]) as u32;
    let height = u16::from_le_bytes([data[8], data[9]]) as u32;
    let depth = 24;
    let colors = 1u32 << ((data[10] & 7) + 1);
    Some((width, height, depth, colors))
}

fn extract_jpeg_params(data: &[u8]) -> Option<(u32, u32, u32, u32)> {
    if !is_jpeg(data) {
        return None;
    }
    let mut offs = 2usize;
    loop {
        while offs < data.len() && data[offs] != 0xFF {
            offs += 1;
        }
        while offs < data.len() && data[offs] == 0xFF {
            offs += 1;
        }
        if offs >= data.len() {
            break;
        }
        let marker = data[offs];
        offs += 1;
        if offs >= data.len() || (0xD8..=0xDA).contains(&marker) {
            break;
        }
        if (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if data.len().saturating_sub(offs) < 2 {
            break;
        }
        let segment_len = u16::from_be_bytes([data[offs], data[offs + 1]]) as usize;
        if segment_len < 2 || data.len().saturating_sub(offs) < segment_len {
            break;
        }
        if marker == 0xC0 || (marker > 0xC0 && marker < 0xD0 && (marker & 3) != 0) {
            if segment_len >= 8 {
                let height = u16::from_be_bytes([data[offs + 3], data[offs + 4]]) as u32;
                let width = u16::from_be_bytes([data[offs + 5], data[offs + 6]]) as u32;
                let depth = data[offs + 2] as u32 * data[offs + 7] as u32;
                return Some((width, height, depth, 0));
            }
            break;
        }
        offs += segment_len;
    }
    Some((0, 0, 0, 0))
}

fn push_u32be(dst: &mut Vec<u8>, value: u32) {
    dst.extend_from_slice(&value.to_be_bytes());
}

fn read_u32be(data: &[u8]) -> u32 {
    u32::from_be_bytes([data[0], data[1], data[2], data[3]])
}

fn base64_encode(src: &[u8]) -> String {
    let mut out = String::with_capacity(src.len().div_ceil(3) * 4);
    let groups = src.len() / 3;
    for i in 0..groups {
        let s0 = src[3 * i];
        let s1 = src[3 * i + 1];
        let s2 = src[3 * i + 2];
        out.push(BASE64_TABLE[(s0 >> 2) as usize] as char);
        out.push(BASE64_TABLE[((s0 & 3) << 4 | (s1 >> 4)) as usize] as char);
        out.push(BASE64_TABLE[((s1 & 15) << 2 | (s2 >> 6)) as usize] as char);
        out.push(BASE64_TABLE[(s2 & 63) as usize] as char);
    }
    match src.len() - groups * 3 {
        1 => {
            let s0 = src[3 * groups];
            out.push(BASE64_TABLE[(s0 >> 2) as usize] as char);
            out.push(BASE64_TABLE[((s0 & 3) << 4) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let s0 = src[3 * groups];
            let s1 = src[3 * groups + 1];
            out.push(BASE64_TABLE[(s0 >> 2) as usize] as char);
            out.push(BASE64_TABLE[((s0 & 3) << 4 | (s1 >> 4)) as usize] as char);
            out.push(BASE64_TABLE[((s1 & 15) << 2) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec::Vec;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{parse_picture_specification, parse_picture_specification_from_memory};
    use crate::libopusenc::{LibopusencError, PictureErrorKind};

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
        let base64 =
            parse_picture_specification_from_memory(GIF_2X3, 4, None, &mut seen_file_icons)
                .expect("parse gif");
        let parsed = parse_picture_b64(&base64);
        assert_eq!(4, parsed.picture_type);
        assert_eq!(b"image/gif", parsed.mime.as_slice());
        assert_eq!(0, parsed.description_len);
        assert_eq!(2, parsed.width);
        assert_eq!(3, parsed.height);
        assert_eq!(24, parsed.depth);
        assert_eq!(4, parsed.colors);

        let base64 = parse_picture_specification_from_memory(
            JPEG_32X16,
            5,
            Some("jpeg"),
            &mut seen_file_icons,
        )
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
            Err(LibopusencError::Picture(PictureErrorKind::InvalidPicture)),
            parse_picture_specification_from_memory(
                INVALID_BYTES,
                3,
                Some("bad"),
                &mut seen_file_icons
            )
        );
        assert_eq!(
            Err(LibopusencError::Picture(PictureErrorKind::InvalidPicture)),
            parse_picture_specification_from_memory(
                PNG_32X32_RGBA,
                21,
                Some("bad"),
                &mut seen_file_icons
            )
        );
        assert_eq!(
            Err(LibopusencError::Io(std::io::Error::from(std::io::ErrorKind::NotFound))),
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
        let base64 = parse_picture_specification_from_memory(
            PNG_32X32_RGBA,
            1,
            Some("icon"),
            &mut seen_file_icons,
        )
        .expect("parse icon");
        assert_eq!(1, seen_file_icons);
        let parsed = parse_picture_b64(&base64);
        assert_eq!(1, parsed.picture_type);
        assert_eq!(b"image/png", parsed.mime.as_slice());

        assert_eq!(
            Err(LibopusencError::Picture(PictureErrorKind::InvalidPicture)),
            parse_picture_specification_from_memory(
                PNG_32X32_RGBA,
                1,
                Some("icon-again"),
                &mut seen_file_icons
            )
        );

        let mut fresh_seen_file_icons = 0;
        assert_eq!(
            Err(LibopusencError::Picture(PictureErrorKind::InvalidIcon)),
            parse_picture_specification_from_memory(
                PNG_16X16_RGBA,
                1,
                Some("bad-icon"),
                &mut fresh_seen_file_icons
            )
        );
        assert_eq!(
            Err(LibopusencError::Picture(PictureErrorKind::InvalidIcon)),
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
            Err(LibopusencError::Picture(PictureErrorKind::InvalidPicture)),
            parse_picture_specification_from_memory(
                PNG_32X32_RGBA,
                2,
                Some("other-icon-again"),
                &mut seen_file_icons
            )
        );
    }
}
