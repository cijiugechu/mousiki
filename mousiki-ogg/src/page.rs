extern crate alloc;

use alloc::vec::Vec;

use crate::crc::update_crc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PageFlags {
    pub continued: bool,
    pub beginning_of_stream: bool,
    pub end_of_stream: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Page {
    header: Vec<u8>,
    body: Vec<u8>,
}

pub struct SegmentSlices<'a> {
    page: &'a Page,
    index: usize,
    offset: usize,
}

impl Page {
    #[must_use]
    pub fn from_parts(header: Vec<u8>, body: Vec<u8>) -> Self {
        Self { header, body }
    }

    #[must_use]
    pub fn header_len(&self) -> usize {
        self.header.len()
    }

    #[must_use]
    pub fn body_len(&self) -> usize {
        self.body.len()
    }

    #[must_use]
    pub fn header_bytes(&self) -> &[u8] {
        &self.header
    }

    #[must_use]
    pub fn body_bytes(&self) -> &[u8] {
        &self.body
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.header.len() + self.body.len());
        out.extend_from_slice(&self.header);
        out.extend_from_slice(&self.body);
        out
    }

    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.header.len() + self.body.len());
        out.extend_from_slice(&self.header);
        out.extend_from_slice(&self.body);
        out
    }

    #[must_use]
    pub fn version(&self) -> i32 {
        self.header.get(4).copied().unwrap_or(0) as i32
    }

    #[must_use]
    pub fn flags(&self) -> PageFlags {
        let header_type = self.header.get(5).copied().unwrap_or(0);
        PageFlags {
            continued: (header_type & 0x01) != 0,
            beginning_of_stream: (header_type & 0x02) != 0,
            end_of_stream: (header_type & 0x04) != 0,
        }
    }

    #[must_use]
    pub fn is_continued(&self) -> bool {
        self.flags().continued
    }

    #[must_use]
    pub fn is_beginning_of_stream(&self) -> bool {
        self.flags().beginning_of_stream
    }

    #[must_use]
    pub fn is_end_of_stream(&self) -> bool {
        self.flags().end_of_stream
    }

    #[must_use]
    pub fn granule_position(&self) -> i64 {
        if self.header.len() < 14 {
            return 0;
        }
        i64::from_le_bytes(self.header[6..14].try_into().expect("slice length checked"))
    }

    #[must_use]
    pub fn stream_serial(&self) -> i32 {
        if self.header.len() < 18 {
            return 0;
        }
        u32::from_le_bytes(
            self.header[14..18]
                .try_into()
                .expect("slice length checked"),
        ) as i32
    }

    #[must_use]
    pub fn sequence_number(&self) -> i64 {
        if self.header.len() < 22 {
            return 0;
        }
        u32::from_le_bytes(
            self.header[18..22]
                .try_into()
                .expect("slice length checked"),
        ) as i64
    }

    #[must_use]
    pub fn packet_count(&self) -> i32 {
        let Some(&segments) = self.header.get(26) else {
            return 0;
        };
        let mut count = 0;
        for &value in &self.header[27..27 + segments as usize] {
            if value < 255 {
                count += 1;
            }
        }
        count
    }

    pub fn update_checksum(&mut self) {
        if self.header.len() < 26 {
            return;
        }
        self.header[22..26].fill(0);
        let crc = update_crc(update_crc(0, &self.header), &self.body);
        self.header[22..26].copy_from_slice(&crc.to_le_bytes());
    }

    #[must_use]
    pub fn is_checksum_valid(&self) -> bool {
        if self.header.len() < 26 {
            return false;
        }
        let mut recomputed = self.clone();
        let expected = recomputed.header[22..26].to_vec();
        recomputed.update_checksum();
        recomputed.header[22..26] == expected
    }

    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.header.get(26).copied().unwrap_or(0) as usize
    }

    #[must_use]
    pub fn lacing_values(&self) -> &[u8] {
        let count = self.segment_count();
        &self.header[27..27 + count]
    }

    #[must_use]
    pub fn segment_slices(&self) -> SegmentSlices<'_> {
        SegmentSlices {
            page: self,
            index: 0,
            offset: 0,
        }
    }
}

impl<'a> Iterator for SegmentSlices<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let len = *self.page.lacing_values().get(self.index)? as usize;
        let start = self.offset;
        let end = start + len;
        self.index += 1;
        self.offset = end;
        Some(&self.page.body[start..end])
    }
}
