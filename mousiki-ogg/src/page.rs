extern crate alloc;

use alloc::vec::Vec;

use crate::crc::update_crc;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Page {
    pub header: Vec<u8>,
    pub body: Vec<u8>,
}

pub struct Segments<'a> {
    page: &'a Page,
    index: usize,
    offset: usize,
}

impl Page {
    #[must_use]
    pub fn new(header: Vec<u8>, body: Vec<u8>) -> Self {
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
    pub fn version(&self) -> i32 {
        self.header.get(4).copied().unwrap_or(0) as i32
    }

    #[must_use]
    pub fn continued(&self) -> i32 {
        (self.header.get(5).copied().unwrap_or(0) & 0x01) as i32
    }

    #[must_use]
    pub fn bos(&self) -> i32 {
        ((self.header.get(5).copied().unwrap_or(0) & 0x02) != 0) as i32
    }

    #[must_use]
    pub fn eos(&self) -> i32 {
        ((self.header.get(5).copied().unwrap_or(0) & 0x04) != 0) as i32
    }

    #[must_use]
    pub fn granulepos(&self) -> i64 {
        if self.header.len() < 14 {
            return 0;
        }
        i64::from_le_bytes(self.header[6..14].try_into().expect("slice length checked"))
    }

    #[must_use]
    pub fn serialno(&self) -> i32 {
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
    pub fn pageno(&self) -> i64 {
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
    pub fn packets(&self) -> i32 {
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

    pub fn checksum_set(&mut self) {
        if self.header.len() < 26 {
            return;
        }
        self.header[22] = 0;
        self.header[23] = 0;
        self.header[24] = 0;
        self.header[25] = 0;
        let crc = update_crc(update_crc(0, &self.header), &self.body);
        self.header[22..26].copy_from_slice(&crc.to_le_bytes());
    }

    #[must_use]
    pub fn checksum_valid(&self) -> bool {
        if self.header.len() < 26 {
            return false;
        }
        let mut recomputed = self.clone();
        let expected = recomputed.header[22..26].to_vec();
        recomputed.checksum_set();
        recomputed.header[22..26] == expected
    }

    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.header.len() + self.body.len());
        out.extend_from_slice(&self.header);
        out.extend_from_slice(&self.body);
        out
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
    pub fn segments(&self) -> Segments<'_> {
        Segments {
            page: self,
            index: 0,
            offset: 0,
        }
    }
}

impl<'a> Iterator for Segments<'a> {
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
