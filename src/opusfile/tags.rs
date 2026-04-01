use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::opusfile::OpusfileError;
use crate::opusfile::util::{parse_u32_le, tag_ncompare_impl};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpusTags {
    vendor: Vec<u8>,
    comments: Vec<Vec<u8>>,
    binary_suffix: Vec<u8>,
}

impl OpusTags {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse(data: &[u8]) -> Result<Self, OpusfileError> {
        if data.len() < 8 {
            return Err(OpusfileError::NotFormat);
        }
        if &data[..8] != b"OpusTags" {
            return Err(OpusfileError::NotFormat);
        }
        if data.len() < 16 {
            return Err(OpusfileError::BadHeader);
        }

        let mut cursor = 8usize;
        let vendor_len = parse_u32_le(&data[cursor..cursor + 4]) as usize;
        cursor += 4;
        if vendor_len > data.len().saturating_sub(cursor) {
            return Err(OpusfileError::BadHeader);
        }
        let vendor = data[cursor..cursor + vendor_len].to_vec();
        cursor += vendor_len;

        if data.len().saturating_sub(cursor) < 4 {
            return Err(OpusfileError::BadHeader);
        }
        let comment_count = parse_u32_le(&data[cursor..cursor + 4]) as usize;
        cursor += 4;
        if comment_count > data.len().saturating_sub(cursor) >> 2 {
            return Err(OpusfileError::BadHeader);
        }
        if comment_count > i32::MAX as usize - 1 {
            return Err(OpusfileError::Fault);
        }

        let mut comments = Vec::with_capacity(comment_count);
        for index in 0..comment_count {
            if comment_count - index > data.len().saturating_sub(cursor) >> 2 {
                return Err(OpusfileError::BadHeader);
            }
            let comment_len = parse_u32_le(&data[cursor..cursor + 4]) as usize;
            cursor += 4;
            if comment_len > data.len().saturating_sub(cursor) {
                return Err(OpusfileError::BadHeader);
            }
            if comment_len > i32::MAX as usize {
                return Err(OpusfileError::Fault);
            }
            comments.push(data[cursor..cursor + comment_len].to_vec());
            cursor += comment_len;
        }

        let binary_suffix = if cursor < data.len() && (data[cursor] & 1) != 0 {
            data[cursor..].to_vec()
        } else {
            Vec::new()
        };

        Ok(Self {
            vendor,
            comments,
            binary_suffix,
        })
    }

    #[must_use]
    pub fn vendor_bytes(&self) -> &[u8] {
        &self.vendor
    }

    #[must_use]
    pub fn vendor(&self) -> Option<&str> {
        core::str::from_utf8(&self.vendor).ok()
    }

    #[must_use]
    pub fn comment_count(&self) -> usize {
        self.comments.len()
    }

    pub fn comments(&self) -> impl Iterator<Item = &[u8]> {
        self.comments.iter().map(Vec::as_slice)
    }

    #[must_use]
    pub fn comment(&self, index: usize) -> Option<&[u8]> {
        self.comments.get(index).map(Vec::as_slice)
    }

    pub fn add(&mut self, tag: &str, value: &str) -> Result<(), OpusfileError> {
        let tag_len = tag.len();
        let value_len = value.len();
        let total_len = tag_len
            .checked_add(value_len)
            .and_then(|len| len.checked_add(1))
            .ok_or(OpusfileError::Fault)?;
        if total_len > i32::MAX as usize {
            return Err(OpusfileError::Fault);
        }
        let mut comment = Vec::with_capacity(total_len);
        comment.extend_from_slice(tag.as_bytes());
        comment.push(b'=');
        comment.extend_from_slice(value.as_bytes());
        self.comments.push(comment);
        Ok(())
    }

    pub fn add_comment(&mut self, comment: &str) -> Result<(), OpusfileError> {
        if comment.len() > i32::MAX as usize {
            return Err(OpusfileError::Fault);
        }
        self.comments.push(comment.as_bytes().to_vec());
        Ok(())
    }

    pub fn set_binary_suffix(&mut self, data: &[u8]) -> Result<(), OpusfileError> {
        if !data.is_empty() && (data[0] & 1) == 0 {
            return Err(OpusfileError::BadArgument);
        }
        if data.len() > i32::MAX as usize {
            return Err(OpusfileError::Fault);
        }
        self.binary_suffix.clear();
        self.binary_suffix.extend_from_slice(data);
        Ok(())
    }

    #[must_use]
    pub fn binary_suffix(&self) -> Option<&[u8]> {
        (!self.binary_suffix.is_empty()).then_some(self.binary_suffix.as_slice())
    }

    #[must_use]
    pub fn query(&self, tag: &str, count: usize) -> Option<&[u8]> {
        let mut found = 0usize;
        for comment in &self.comments {
            if matches!(tag_ncompare(tag.as_bytes(), tag.len(), comment), Ordering::Equal) {
                if found == count {
                    return comment.get(tag.len() + 1..);
                }
                found += 1;
            }
        }
        None
    }

    #[must_use]
    pub fn query_str(&self, tag: &str, count: usize) -> Option<&str> {
        core::str::from_utf8(self.query(tag, count)?).ok()
    }

    #[must_use]
    pub fn query_count(&self, tag: &str) -> usize {
        self.comments
            .iter()
            .filter(|comment| matches!(tag_ncompare(tag.as_bytes(), tag.len(), comment), Ordering::Equal))
            .count()
    }

    #[must_use]
    pub fn album_gain_q8(&self) -> Option<i16> {
        gain_from_tags(self, b"R128_ALBUM_GAIN")
    }

    #[must_use]
    pub fn track_gain_q8(&self) -> Option<i16> {
        gain_from_tags(self, b"R128_TRACK_GAIN")
    }
}

#[must_use]
pub fn tag_compare(tag_name: &str, comment: &[u8]) -> Ordering {
    tag_ncompare(tag_name.as_bytes(), tag_name.len(), comment)
}

#[must_use]
pub fn tag_ncompare(tag_name: &[u8], tag_len: usize, comment: &[u8]) -> Ordering {
    tag_ncompare_impl(tag_name, tag_len, comment)
}

fn gain_from_tags(tags: &OpusTags, tag_name: &[u8]) -> Option<i16> {
    for comment in &tags.comments {
        if !matches!(tag_ncompare(tag_name, tag_name.len(), comment), Ordering::Equal) {
            continue;
        }
        let mut input = comment.get(tag_name.len() + 1..)?;
        let mut negative = false;
        if let Some(rest) = input.strip_prefix(b"-") {
            negative = true;
            input = rest;
        } else if let Some(rest) = input.strip_prefix(b"+") {
            input = rest;
        }
        if input.is_empty() || !input.iter().all(u8::is_ascii_digit) {
            continue;
        }
        let mut gain: i32 = 0;
        for &digit in input {
            gain = gain.saturating_mul(10).saturating_add(i32::from(digit - b'0'));
            if gain > i32::from(i16::MAX) + i32::from(negative) {
                break;
            }
        }
        if gain > i32::from(i16::MAX) + i32::from(negative) {
            continue;
        }
        let signed = if negative { -gain } else { gain };
        if let Ok(gain) = i16::try_from(signed) {
            return Some(gain);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{OpusTags, tag_compare};
    use crate::opusfile::OpusfileError;
    use core::cmp::Ordering;

    fn build_tags_packet() -> Vec<u8> {
        let vendor = b"mousiki-tests";
        let comments = [b"ARTIST=nemurubaka".as_slice(), b"R128_ALBUM_GAIN=-123".as_slice()];
        let suffix = [1u8, 2, 3, 4];
        let mut packet = Vec::new();
        packet.extend_from_slice(b"OpusTags");
        packet.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        packet.extend_from_slice(vendor);
        packet.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for comment in comments {
            packet.extend_from_slice(&(comment.len() as u32).to_le_bytes());
            packet.extend_from_slice(comment);
        }
        packet.extend_from_slice(&suffix);
        packet
    }

    #[test]
    fn parses_comment_packet_and_binary_suffix() {
        let tags = OpusTags::parse(&build_tags_packet()).expect("valid tags packet");
        assert_eq!(tags.vendor(), Some("mousiki-tests"));
        assert_eq!(tags.comment_count(), 2);
        assert_eq!(tags.query_str("artist", 0), Some("nemurubaka"));
        assert_eq!(tags.query_count("artist"), 1);
        assert_eq!(tags.album_gain_q8(), Some(-123));
        assert_eq!(tags.track_gain_q8(), None);
        assert_eq!(tags.binary_suffix(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn ignores_invalid_binary_suffix_on_parse() {
        let vendor = b"x";
        let mut packet = Vec::new();
        packet.extend_from_slice(b"OpusTags");
        packet.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        packet.extend_from_slice(vendor);
        packet.extend_from_slice(&0u32.to_le_bytes());
        packet.extend_from_slice(&[0u8, 9, 9]);
        let tags = OpusTags::parse(&packet).expect("packet still parses");
        assert_eq!(tags.binary_suffix(), None);
    }

    #[test]
    fn supports_add_and_suffix_update() {
        let mut tags = OpusTags::new();
        tags.add("TITLE", "opusfile port").expect("add title");
        tags.add_comment("ARTIST=nemurubaka").expect("add comment");
        tags.set_binary_suffix(&[1, 9]).expect("valid suffix");
        assert_eq!(tags.query_str("title", 0), Some("opusfile port"));
        assert_eq!(tags.query_str("artist", 0), Some("nemurubaka"));
        assert_eq!(tags.binary_suffix(), Some(&[1, 9][..]));
        assert_eq!(
            tags.set_binary_suffix(&[0, 9]),
            Err(OpusfileError::BadArgument)
        );
    }

    #[test]
    fn tag_compare_is_ascii_case_insensitive() {
        assert_eq!(tag_compare("artist", b"ARTIST=value"), Ordering::Equal);
        assert_eq!(tag_compare("artist", b"ARTISTX=value"), Ordering::Less);
        assert_eq!(tag_compare("artist", b"ARTIS=value"), Ordering::Greater);
        assert_eq!(tag_compare("artist", b"title=value"), Ordering::Less);
    }

    #[test]
    fn gain_parser_skips_invalid_values() {
        let mut tags = OpusTags::new();
        tags.add_comment("R128_TRACK_GAIN=12x").expect("invalid first");
        tags.add_comment("R128_TRACK_GAIN=+321").expect("valid second");
        assert_eq!(tags.track_gain_q8(), Some(321));
    }
}
