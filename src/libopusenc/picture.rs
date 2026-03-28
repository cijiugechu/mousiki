extern crate std;

use alloc::string::String;
use alloc::vec::Vec;

use std::fs;

use crate::libopusenc::encoder::OpeError;

const BASE64_TABLE: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const PNG_SIGNATURE: &[u8] = b"\x89PNG\x0D\x0A\x1A\x0A";

pub fn parse_picture_specification(
    filename: &str,
    picture_type: i32,
    description: Option<&str>,
    seen_file_icons: &mut i32,
) -> Result<String, OpeError> {
    let picture_type = normalize_picture_type(picture_type);
    if !validate_picture_type(picture_type, *seen_file_icons) {
        return Err(OpeError::InvalidPicture);
    }
    let data = fs::read(filename).map_err(|_| OpeError::CannotOpen)?;
    parse_picture_specification_from_memory(&data, picture_type, description, seen_file_icons)
}

pub fn parse_picture_specification_from_memory(
    mem: &[u8],
    picture_type: i32,
    description: Option<&str>,
    seen_file_icons: &mut i32,
) -> Result<String, OpeError> {
    let picture_type = normalize_picture_type(picture_type);
    if !validate_picture_type(picture_type, *seen_file_icons) {
        return Err(OpeError::InvalidPicture);
    }
    let description = description.unwrap_or("");
    let (mime_type, width, height, depth, colors) =
        parse_image(mem).ok_or(OpeError::InvalidPicture)?;
    if picture_type == 1
        && (width != 32 || height != 32 || mime_type != "image/png")
    {
        return Err(OpeError::InvalidIcon);
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
