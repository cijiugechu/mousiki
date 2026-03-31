extern crate alloc;

use alloc::vec::Vec;

use crate::page::Page;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyncState {
    data: Vec<u8>,
    fill: usize,
    returned: usize,
    unsynced: bool,
    headerbytes: usize,
    bodybytes: usize,
}

impl SyncState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn check(&self) -> i32 {
        0
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.fill = 0;
        self.returned = 0;
        self.unsynced = false;
        self.headerbytes = 0;
        self.bodybytes = 0;
    }

    pub fn reset(&mut self) {
        self.fill = 0;
        self.returned = 0;
        self.unsynced = false;
        self.headerbytes = 0;
        self.bodybytes = 0;
    }

    pub fn buffer(&mut self, size: usize) -> &mut [u8] {
        if self.returned > 0 {
            self.data.drain(..self.returned);
            self.fill -= self.returned;
            self.returned = 0;
        }
        let start = self.fill;
        let needed = self.fill + size;
        if needed > self.data.len() {
            self.data.resize(needed, 0);
        }
        &mut self.data[start..needed]
    }

    pub fn wrote(&mut self, bytes: usize) -> i32 {
        let new_fill = match self.fill.checked_add(bytes) {
            Some(value) => value,
            None => return -1,
        };
        if new_fill > self.data.len() {
            return -1;
        }
        self.fill = new_fill;
        0
    }

    pub fn pageseek(&mut self) -> Result<Option<Page>, i64> {
        let start = self.returned;
        let end = self.fill;
        let bytes = end - start;

        if self.headerbytes == 0 {
            if bytes < 27 {
                return Ok(None);
            }
            if &self.data[start..start + 4] != b"OggS" {
                return self.sync_fail(start, end);
            }
            self.headerbytes = usize::from(self.data[start + 26]) + 27;
            if bytes < self.headerbytes {
                return Ok(None);
            }
            self.bodybytes = self.data[start + 27..start + self.headerbytes]
                .iter()
                .map(|&value| usize::from(value))
                .sum();
        }

        if self.headerbytes + self.bodybytes > bytes {
            return Ok(None);
        }

        let header = self.data[start..start + self.headerbytes].to_vec();
        let body =
            self.data[start + self.headerbytes..start + self.headerbytes + self.bodybytes].to_vec();
        let mut recomputed = Page::new(header.clone(), body.clone());
        let checksum = header[22..26].to_vec();
        recomputed.checksum_set();
        if recomputed.header[22..26] != checksum {
            return self.sync_fail(start, end);
        }

        let total = self.headerbytes + self.bodybytes;
        self.unsynced = false;
        self.returned += total;
        self.headerbytes = 0;
        self.bodybytes = 0;
        Ok(Some(Page::new(header, body)))
    }

    fn sync_fail(&mut self, start: usize, end: usize) -> Result<Option<Page>, i64> {
        self.headerbytes = 0;
        self.bodybytes = 0;
        let next = self.data[start + 1..end]
            .iter()
            .position(|&byte| byte == b'O')
            .map_or(self.fill, |offset| start + 1 + offset);
        let skipped = next - start;
        self.returned = next;
        Err(-(skipped as i64))
    }

    pub fn pageout(&mut self) -> Result<Option<Page>, i32> {
        loop {
            match self.pageseek() {
                Ok(Some(page)) => return Ok(Some(page)),
                Ok(None) => return Ok(None),
                Err(_) if !self.unsynced => {
                    self.unsynced = true;
                    return Err(-1);
                }
                Err(_) => {}
            }
        }
    }
}
