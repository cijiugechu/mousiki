use core::cmp::Ordering;

pub(crate) fn parse_u16_le(data: &[u8]) -> u16 {
    u16::from_le_bytes([data[0], data[1]])
}

pub(crate) fn parse_i16_le(data: &[u8]) -> i16 {
    i16::from_le_bytes([data[0], data[1]])
}

pub(crate) fn parse_u32_le(data: &[u8]) -> u32 {
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

pub(crate) fn parse_u32_be(data: &[u8]) -> u32 {
    u32::from_be_bytes([data[0], data[1], data[2], data[3]])
}

#[inline]
fn ascii_fold(byte: u8) -> u8 {
    if byte.is_ascii_lowercase() {
        byte - (b'a' - b'A')
    } else {
        byte
    }
}

pub(crate) fn ascii_nocase_compare(a: &[u8], b: &[u8]) -> Ordering {
    for (&lhs, &rhs) in a.iter().zip(b.iter()) {
        let lhs = ascii_fold(lhs);
        let rhs = ascii_fold(rhs);
        match lhs.cmp(&rhs) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
    }
    a.len().cmp(&b.len())
}

pub(crate) fn tag_ncompare_impl(tag_name: &[u8], tag_len: usize, comment: &[u8]) -> Ordering {
    let compare_len = tag_len.min(tag_name.len());
    let comment_prefix = &comment[..comment.len().min(compare_len)];
    match ascii_nocase_compare(&tag_name[..compare_len], comment_prefix) {
        Ordering::Equal => {
            if compare_len > comment_prefix.len() {
                Ordering::Greater
            } else {
                b'='.cmp(&comment.get(compare_len).copied().unwrap_or(0))
            }
        }
        non_eq => non_eq,
    }
}

pub(crate) fn add_granule_position(src: i64, delta: i32) -> Option<i64> {
    debug_assert_ne!(src, -1);
    if delta > 0 && src < 0 && src >= -1 - i64::from(delta) {
        return None;
    }
    if delta < 0 && src >= 0 && src < -i64::from(delta) {
        return None;
    }
    Some((src as u64).wrapping_add(delta as i64 as u64) as i64)
}
