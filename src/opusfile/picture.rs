use alloc::{vec, vec::Vec};

use crate::opusfile::OpusfileError;
use crate::opusfile::tags::tag_ncompare;
use crate::opusfile::util::parse_u32_be;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusPictureFormat {
    Unknown,
    Url,
    Jpeg,
    Png,
    Gif,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusPictureTag {
    pub picture_type: i32,
    mime_type: Vec<u8>,
    description: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub colors: u32,
    data: Vec<u8>,
    pub format: OpusPictureFormat,
}

impl OpusPictureTag {
    pub fn parse(tag: &str) -> Result<Self, OpusfileError> {
        Self::parse_bytes(tag.as_bytes())
    }

    pub fn parse_bytes(tag: &[u8]) -> Result<Self, OpusfileError> {
        let tag = if matches!(
            tag_ncompare(b"METADATA_BLOCK_PICTURE", 22, tag),
            core::cmp::Ordering::Equal
        ) {
            &tag[23..]
        } else {
            tag
        };
        if tag.len() % 4 != 0 {
            return Err(OpusfileError::NotFormat);
        }
        let base64_chunks = tag.len() / 4;
        let mut decoded_len = 3usize
            .checked_mul(base64_chunks)
            .ok_or(OpusfileError::Fault)?;
        if decoded_len < 32 {
            return Err(OpusfileError::NotFormat);
        }
        if tag.ends_with(b"=") {
            decoded_len -= 1;
        }
        if tag.len() >= 2 && &tag[tag.len() - 2..] == b"==" {
            decoded_len -= 1;
        }
        if decoded_len < 32 {
            return Err(OpusfileError::NotFormat);
        }
        let decoded = decode_base64(tag, decoded_len)?;
        parse_picture_block(&decoded)
    }

    #[must_use]
    pub fn mime_type_bytes(&self) -> &[u8] {
        &self.mime_type
    }

    #[must_use]
    pub fn mime_type(&self) -> Option<&str> {
        core::str::from_utf8(&self.mime_type).ok()
    }

    #[must_use]
    pub fn description_bytes(&self) -> &[u8] {
        &self.description
    }

    #[must_use]
    pub fn description(&self) -> Option<&str> {
        core::str::from_utf8(&self.description).ok()
    }

    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

fn decode_base64(tag: &[u8], decoded_len: usize) -> Result<Vec<u8>, OpusfileError> {
    let mut out = vec![0u8; decoded_len];
    for chunk_index in 0..(tag.len() / 4) {
        let mut value = 0u32;
        for part in 0..4 {
            let char_index = chunk_index * 4 + part;
            let c = tag[char_index];
            let digit = match c {
                b'+' => 62,
                b'/' => 63,
                b'0'..=b'9' => 52 + u32::from(c - b'0'),
                b'a'..=b'z' => 26 + u32::from(c - b'a'),
                b'A'..=b'Z' => u32::from(c - b'A'),
                b'=' if 3 * chunk_index + part > decoded_len => 0,
                _ => return Err(OpusfileError::NotFormat),
            };
            value = (value << 6) | digit;
        }
        let out_index = 3 * chunk_index;
        out[out_index] = (value >> 16) as u8;
        if out_index + 1 < decoded_len {
            out[out_index + 1] = (value >> 8) as u8;
        }
        if out_index + 2 < decoded_len {
            out[out_index + 2] = value as u8;
        }
    }
    Ok(out)
}

fn parse_picture_block(block: &[u8]) -> Result<OpusPictureTag, OpusfileError> {
    let mut cursor = 0usize;
    let picture_type = parse_u32_be(&block[cursor..cursor + 4]) as i32;
    cursor += 4;

    let mime_type_len = parse_u32_be(&block[cursor..cursor + 4]) as usize;
    cursor += 4;
    if mime_type_len > block.len().saturating_sub(32) {
        return Err(OpusfileError::NotFormat);
    }
    let mime_type = block[cursor..cursor + mime_type_len].to_vec();
    cursor += mime_type_len;

    let description_len = parse_u32_be(&block[cursor..cursor + 4]) as usize;
    cursor += 4;
    if description_len > block.len().saturating_sub(mime_type_len + 32) {
        return Err(OpusfileError::NotFormat);
    }
    let description = block[cursor..cursor + description_len].to_vec();
    cursor += description_len;

    let mut width = parse_u32_be(&block[cursor..cursor + 4]);
    cursor += 4;
    let mut height = parse_u32_be(&block[cursor..cursor + 4]);
    cursor += 4;
    let mut depth = parse_u32_be(&block[cursor..cursor + 4]);
    cursor += 4;
    let mut colors = parse_u32_be(&block[cursor..cursor + 4]);
    cursor += 4;
    let colors_set = width != 0 || height != 0 || depth != 0 || colors != 0;
    if (width == 0 || height == 0 || depth == 0) && colors_set {
        return Err(OpusfileError::NotFormat);
    }

    let data_length = parse_u32_be(&block[cursor..cursor + 4]) as usize;
    cursor += 4;
    if data_length > block.len().saturating_sub(cursor) {
        return Err(OpusfileError::NotFormat);
    }
    let data = &block[cursor..cursor + data_length];

    let mime_str = core::str::from_utf8(&mime_type).ok();
        let (format, extracted) = if mime_type_len == 3 && mime_type.as_slice() == b"-->" {
        if picture_type == 1 && (width != 0 || height != 0) && (width != 32 || height != 32) {
            return Err(OpusfileError::NotFormat);
        }
        (OpusPictureFormat::Url, None)
    } else {
        let format = if mime_str == Some("image/jpeg") {
            if is_jpeg(data) {
                OpusPictureFormat::Jpeg
            } else {
                OpusPictureFormat::Unknown
            }
        } else if mime_str == Some("image/png") {
            if is_png(data) {
                OpusPictureFormat::Png
            } else {
                OpusPictureFormat::Unknown
            }
        } else if mime_str == Some("image/gif") {
            if is_gif(data) {
                OpusPictureFormat::Gif
            } else {
                OpusPictureFormat::Unknown
            }
        } else if mime_type.is_empty() || mime_str == Some("image/") {
            if is_jpeg(data) {
                OpusPictureFormat::Jpeg
            } else if is_png(data) {
                OpusPictureFormat::Png
            } else if is_gif(data) {
                OpusPictureFormat::Gif
            } else {
                OpusPictureFormat::Unknown
            }
        } else {
            OpusPictureFormat::Unknown
        };
        let extracted = match format {
            OpusPictureFormat::Jpeg => extract_jpeg_params(data),
            OpusPictureFormat::Png => extract_png_params(data),
            OpusPictureFormat::Gif => extract_gif_params(data),
            OpusPictureFormat::Unknown | OpusPictureFormat::Url => None,
        };
        if picture_type == 1 {
            let (w, h, _, _) = extracted.unwrap_or((width, height, depth, colors));
            if format != OpusPictureFormat::Png || w != 32 || h != 32 {
                return Err(OpusfileError::NotFormat);
            }
        }
        (format, extracted)
    };

    if let Some((file_width, file_height, file_depth, file_colors)) = extracted {
        width = file_width;
        height = file_height;
        depth = file_depth;
        colors = file_colors;
    }

    Ok(OpusPictureTag {
        picture_type,
        mime_type,
        description,
        width,
        height,
        depth,
        colors,
        data: data.to_vec(),
        format,
    })
}

fn is_jpeg(buf: &[u8]) -> bool {
    buf.len() >= 3 && &buf[..3] == b"\xFF\xD8\xFF"
}

fn is_png(buf: &[u8]) -> bool {
    buf.len() >= 8 && &buf[..8] == b"\x89PNG\x0D\x0A\x1A\x0A"
}

fn is_gif(buf: &[u8]) -> bool {
    buf.len() >= 6 && (&buf[..6] == b"GIF87a" || &buf[..6] == b"GIF89a")
}

fn extract_jpeg_params(data: &[u8]) -> Option<(u32, u32, u32, u32)> {
    if !is_jpeg(data) {
        return None;
    }
    let mut offset = 2usize;
    loop {
        while offset < data.len() && data[offset] != 0xFF {
            offset += 1;
        }
        while offset < data.len() && data[offset] == 0xFF {
            offset += 1;
        }
        if offset >= data.len() {
            break;
        }
        let marker = data[offset];
        offset += 1;
        if offset >= data.len() || (0xD8..=0xDA).contains(&marker) {
            break;
        }
        if (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if data.len().saturating_sub(offset) < 2 {
            break;
        }
        let segment_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        if segment_len < 2 || data.len().saturating_sub(offset) < segment_len {
            break;
        }
        if marker == 0xC0 || (marker > 0xC0 && marker < 0xD0 && (marker & 3) != 0) {
            if segment_len >= 8 {
                return Some((
                    u16::from_be_bytes([data[offset + 5], data[offset + 6]]) as u32,
                    u16::from_be_bytes([data[offset + 3], data[offset + 4]]) as u32,
                    u32::from(data[offset + 2]) * u32::from(data[offset + 7]),
                    0,
                ));
            }
            break;
        }
        offset += segment_len;
    }
    None
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
    let mut offset = 8usize;
    while data.len().saturating_sub(offset) >= 12 {
        let chunk_len = parse_u32_be(&data[offset..offset + 4]) as usize;
        if chunk_len > data.len().saturating_sub(offset + 12) {
            break;
        }
        if chunk_len == 13 && &data[offset + 4..offset + 8] == b"IHDR" {
            width = parse_u32_be(&data[offset + 8..offset + 12]);
            height = parse_u32_be(&data[offset + 12..offset + 16]);
            let color_type = data[offset + 17];
            if color_type == 3 {
                depth = 24;
                has_palette = 1;
            } else {
                let sample_depth = u32::from(data[offset + 16]);
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
        } else if has_palette > 0 && &data[offset + 4..offset + 8] == b"PLTE" {
            colors = (chunk_len / 3) as u32;
            break;
        }
        offset += 12 + chunk_len;
    }
    (width != 0 && height != 0 && depth != 0).then_some((width, height, depth, colors))
}

fn extract_gif_params(data: &[u8]) -> Option<(u32, u32, u32, u32)> {
    if !is_gif(data) || data.len() < 14 {
        return None;
    }
    Some((
        u16::from_le_bytes([data[6], data[7]]) as u32,
        u16::from_le_bytes([data[8], data[9]]) as u32,
        24,
        1u32 << ((data[10] & 7) + 1),
    ))
}

#[cfg(test)]
mod tests {
    use alloc::{format, string::String, vec, vec::Vec};

    use super::{OpusPictureFormat, OpusPictureTag};

    fn encode_base64(data: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
        let groups = data.len() / 3;
        for index in 0..groups {
            let s0 = data[3 * index];
            let s1 = data[3 * index + 1];
            let s2 = data[3 * index + 2];
            out.push(TABLE[(s0 >> 2) as usize] as char);
            out.push(TABLE[((s0 & 3) << 4 | (s1 >> 4)) as usize] as char);
            out.push(TABLE[((s1 & 15) << 2 | (s2 >> 6)) as usize] as char);
            out.push(TABLE[(s2 & 63) as usize] as char);
        }
        match data.len() - groups * 3 {
            1 => {
                let s0 = data[3 * groups];
                out.push(TABLE[(s0 >> 2) as usize] as char);
                out.push(TABLE[((s0 & 3) << 4) as usize] as char);
                out.push('=');
                out.push('=');
            }
            2 => {
                let s0 = data[3 * groups];
                let s1 = data[3 * groups + 1];
                out.push(TABLE[(s0 >> 2) as usize] as char);
                out.push(TABLE[((s0 & 3) << 4 | (s1 >> 4)) as usize] as char);
                out.push(TABLE[((s1 & 15) << 2) as usize] as char);
                out.push('=');
            }
            _ => {}
        }
        out
    }

    fn png_32x32_rgba_stub() -> Vec<u8> {
        vec![
            0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, b'I', b'H', b'D',
            b'R', 0, 0, 0, 32, 0, 0, 0, 32, 8, 6, 0, 0, 0, 0, 0, 0, 0,
        ]
    }

    fn metadata_block_picture_tag(picture_type: u32, mime: &[u8], description: &[u8], data: &[u8]) -> String {
        let mut block = Vec::new();
        block.extend_from_slice(&picture_type.to_be_bytes());
        block.extend_from_slice(&(mime.len() as u32).to_be_bytes());
        block.extend_from_slice(mime);
        block.extend_from_slice(&(description.len() as u32).to_be_bytes());
        block.extend_from_slice(description);
        block.extend_from_slice(&0u32.to_be_bytes());
        block.extend_from_slice(&0u32.to_be_bytes());
        block.extend_from_slice(&0u32.to_be_bytes());
        block.extend_from_slice(&0u32.to_be_bytes());
        block.extend_from_slice(&(data.len() as u32).to_be_bytes());
        block.extend_from_slice(data);
        encode_base64(&block)
    }

    #[test]
    fn parses_png_picture_and_overrides_dimensions() {
        let png = png_32x32_rgba_stub();
        let tag = metadata_block_picture_tag(3, b"image/png", b"cover", &png);
        let picture = OpusPictureTag::parse(&tag).expect("valid picture tag");
        assert_eq!(picture.picture_type, 3);
        assert_eq!(picture.mime_type(), Some("image/png"));
        assert_eq!(picture.description(), Some("cover"));
        assert_eq!(picture.width, 32);
        assert_eq!(picture.height, 32);
        assert_eq!(picture.depth, 32);
        assert_eq!(picture.colors, 0);
        assert_eq!(picture.format, OpusPictureFormat::Png);
        assert_eq!(picture.data(), png.as_slice());
    }

    #[test]
    fn parses_prefixed_metadata_block_picture_tag() {
        let png = png_32x32_rgba_stub();
        let tag = metadata_block_picture_tag(3, b"image/png", b"", &png);
        let with_prefix = format!("METADATA_BLOCK_PICTURE={tag}");
        let picture = OpusPictureTag::parse(&with_prefix).expect("valid prefixed tag");
        assert_eq!(picture.format, OpusPictureFormat::Png);
    }

    #[test]
    fn rejects_non_png_file_icon() {
        let gif = b"GIF89a\x20\x00\x20\x00\x00\x00\x00\x00".to_vec();
        let tag = metadata_block_picture_tag(1, b"image/gif", b"", &gif);
        assert!(OpusPictureTag::parse(&tag).is_err());
    }
}
