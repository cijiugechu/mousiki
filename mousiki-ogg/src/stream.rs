extern crate alloc;

use alloc::vec::Vec;

use crate::crc::update_crc;
use crate::packet::{Packet, PacketMetadata};
use crate::page::Page;

const LACING_START: i32 = 0x100;
const LACING_END_OF_STREAM: i32 = 0x200;
const LACING_HOLE: i32 = 0x400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEncodeError {
    InvalidPacket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamDecodeError {
    InvalidPage,
    Gap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawStreamState {
    body_data: Vec<u8>,
    body_returned: usize,
    lacing_vals: Vec<i32>,
    granule_vals: Vec<i64>,
    lacing_packet: usize,
    lacing_returned: usize,
    header: [u8; 282],
    header_fill: usize,
    e_o_s: i32,
    b_o_s: i32,
    serialno: i32,
    pageno: i64,
    packetno: i64,
    granulepos: i64,
    ready: bool,
}

impl RawStreamState {
    #[must_use]
    pub fn new(serialno: i32) -> Self {
        let mut this = Self {
            body_data: Vec::with_capacity(16 * 1024),
            body_returned: 0,
            lacing_vals: Vec::with_capacity(1024),
            granule_vals: Vec::with_capacity(1024),
            lacing_packet: 0,
            lacing_returned: 0,
            header: [0; 282],
            header_fill: 0,
            e_o_s: 0,
            b_o_s: 0,
            serialno,
            pageno: -1,
            packetno: 0,
            granulepos: 0,
            ready: true,
        };
        this.reset();
        this.serialno = serialno;
        this
    }

    #[must_use]
    pub fn check(&self) -> i32 {
        if self.ready { 0 } else { -1 }
    }

    pub fn clear(&mut self) {
        self.body_data.clear();
        self.lacing_vals.clear();
        self.granule_vals.clear();
        self.body_returned = 0;
        self.lacing_packet = 0;
        self.lacing_returned = 0;
        self.header_fill = 0;
        self.e_o_s = 0;
        self.b_o_s = 0;
        self.pageno = 0;
        self.packetno = 0;
        self.granulepos = 0;
        self.ready = false;
    }

    pub fn reset(&mut self) {
        if self.check() != 0 {
            return;
        }
        self.body_data.clear();
        self.body_returned = 0;
        self.lacing_vals.clear();
        self.granule_vals.clear();
        self.lacing_packet = 0;
        self.lacing_returned = 0;
        self.header_fill = 0;
        self.e_o_s = 0;
        self.b_o_s = 0;
        self.pageno = -1;
        self.packetno = 0;
        self.granulepos = 0;
    }

    pub fn reset_serialno(&mut self, serialno: i32) {
        self.reset();
        self.serialno = serialno;
    }

    #[must_use]
    pub fn eos(&self) -> i32 {
        if self.check() != 0 { 1 } else { self.e_o_s }
    }

    #[must_use]
    pub fn pending_segment_count(&self) -> usize {
        self.lacing_vals.len()
    }

    fn compact_body(&mut self) {
        if self.body_returned > 0 {
            let retained = self.body_data.len().saturating_sub(self.body_returned);
            self.body_data.copy_within(self.body_returned.., 0);
            self.body_data.truncate(retained);
            self.body_returned = 0;
        }
    }

    fn compact_lacing(&mut self) {
        if self.lacing_returned > 0 {
            let retained = self.lacing_vals.len().saturating_sub(self.lacing_returned);
            self.lacing_vals.copy_within(self.lacing_returned.., 0);
            self.lacing_vals.truncate(retained);
            self.granule_vals.copy_within(self.lacing_returned.., 0);
            self.granule_vals.truncate(retained);
            self.lacing_packet = self.lacing_packet.saturating_sub(self.lacing_returned);
            self.lacing_returned = 0;
        }
    }

    pub fn packet_in(&mut self, packet: &Packet) -> i32 {
        self.iovec_in(
            &[packet.data()],
            packet.is_end_of_stream(),
            packet.granule_position(),
        )
    }

    pub fn packet_in_slice(&mut self, packet: &[u8], end_of_stream: bool, granulepos: i64) -> i32 {
        self.iovec_in(&[packet], end_of_stream, granulepos)
    }

    pub fn iovec_in(&mut self, chunks: &[&[u8]], end_of_stream: bool, granulepos: i64) -> i32 {
        if self.check() != 0 {
            return -1;
        }

        let mut bytes = 0usize;
        for chunk in chunks {
            bytes = match bytes.checked_add(chunk.len()) {
                Some(value) => value,
                None => return -1,
            };
        }

        let lacing_vals = bytes / 255 + 1;

        self.compact_body();
        self.body_data.reserve(bytes);
        for chunk in chunks {
            self.body_data.extend_from_slice(chunk);
        }

        self.lacing_vals.reserve(lacing_vals);
        self.granule_vals.reserve(lacing_vals);
        for _ in 0..lacing_vals.saturating_sub(1) {
            self.lacing_vals.push(255);
            self.granule_vals.push(self.granulepos);
        }
        self.lacing_vals.push((bytes % 255) as i32);
        self.granule_vals.push(granulepos);
        self.granulepos = granulepos;

        if let Some(first) = self.lacing_vals.len().checked_sub(lacing_vals) {
            self.lacing_vals[first] |= LACING_START;
        }

        self.packetno += 1;
        if end_of_stream {
            self.e_o_s = 1;
        }
        0
    }

    fn select_page_segments(&mut self, mut force: bool, nfill: i32) -> Option<(usize, usize)> {
        if self.check() != 0 {
            return None;
        }
        let maxvals = self.lacing_vals.len().min(255);
        if maxvals == 0 {
            return None;
        }

        let mut vals = 0usize;
        let mut bytes = 0usize;
        let mut acc = 0i64;
        let mut granule_pos = -1i64;

        if self.b_o_s == 0 {
            granule_pos = 0;
            while vals < maxvals {
                if (self.lacing_vals[vals] & 0xff) < 255 {
                    vals += 1;
                    break;
                }
                vals += 1;
            }
        } else {
            let mut packets_done = 0;
            let mut packet_just_done = 0;
            while vals < maxvals {
                if acc > i64::from(nfill) && packet_just_done >= 4 {
                    force = true;
                    break;
                }
                acc += i64::from(self.lacing_vals[vals] & 0xff);
                if (self.lacing_vals[vals] & 0xff) < 255 {
                    granule_pos = self.granule_vals[vals];
                    packets_done += 1;
                    packet_just_done = packets_done;
                } else {
                    packet_just_done = 0;
                }
                vals += 1;
            }
            if vals == 255 {
                force = true;
            }
        }

        if !force {
            return None;
        }

        self.header[..4].copy_from_slice(b"OggS");
        self.header[4] = 0;
        self.header[5] = 0;
        if (self.lacing_vals[0] & LACING_START) == 0 {
            self.header[5] |= 0x01;
        }
        if self.b_o_s == 0 {
            self.header[5] |= 0x02;
        }
        if self.e_o_s != 0 && self.lacing_vals.len() == vals {
            self.header[5] |= 0x04;
        }
        self.b_o_s = 1;

        self.header[6..14].copy_from_slice(&granule_pos.to_le_bytes());
        self.header[14..18].copy_from_slice(&(self.serialno as u32).to_le_bytes());
        if self.pageno == -1 {
            self.pageno = 0;
        }
        let pageno = self.pageno as u32;
        self.pageno += 1;
        self.header[18..22].copy_from_slice(&pageno.to_le_bytes());
        self.header[22..26].fill(0);
        self.header[26] = vals as u8;
        for i in 0..vals {
            let segment = (self.lacing_vals[i] & 0xff) as u8;
            self.header[27 + i] = segment;
            bytes += usize::from(segment);
        }
        self.header_fill = vals + 27;

        Some((vals, bytes))
    }

    fn finish_page(&mut self, vals: usize, bytes: usize) {
        let retained = self.lacing_vals.len().saturating_sub(vals);
        self.lacing_vals.copy_within(vals.., 0);
        self.lacing_vals.truncate(retained);
        self.granule_vals.copy_within(vals.., 0);
        self.granule_vals.truncate(retained);
        self.body_returned += bytes;
    }

    fn flush_internal(&mut self, force: bool, nfill: i32) -> Option<Page> {
        let (vals, bytes) = self.select_page_segments(force, nfill)?;
        let header = self.header[..self.header_fill].to_vec();
        let body = self.body_data[self.body_returned..self.body_returned + bytes].to_vec();
        self.finish_page(vals, bytes);
        let mut page = Page::from_parts(header, body);
        page.update_checksum();
        Some(page)
    }

    fn flush_internal_bytes(&mut self, force: bool, nfill: i32) -> Option<Vec<u8>> {
        let (vals, bytes) = self.select_page_segments(force, nfill)?;
        let header_fill = self.header_fill;
        let mut page = Vec::with_capacity(header_fill + bytes);
        page.extend_from_slice(&self.header[..header_fill]);
        page.extend_from_slice(&self.body_data[self.body_returned..self.body_returned + bytes]);

        page[22..26].fill(0);
        let crc = update_crc(
            update_crc(0, &page[..header_fill]),
            &page[header_fill..header_fill + bytes],
        );
        page[22..26].copy_from_slice(&crc.to_le_bytes());

        self.finish_page(vals, bytes);
        Some(page)
    }

    pub fn flush(&mut self) -> Option<Page> {
        self.flush_internal(true, 4096)
    }

    pub fn flush_fill(&mut self, nfill: i32) -> Option<Page> {
        self.flush_internal(true, nfill)
    }

    pub fn flush_bytes(&mut self) -> Option<Vec<u8>> {
        self.flush_internal_bytes(true, 4096)
    }

    pub fn flush_fill_bytes(&mut self, nfill: i32) -> Option<Vec<u8>> {
        self.flush_internal_bytes(true, nfill)
    }

    pub fn page_out(&mut self) -> Option<Page> {
        let force = (self.e_o_s != 0 && !self.lacing_vals.is_empty())
            || (!self.lacing_vals.is_empty() && self.b_o_s == 0);
        self.flush_internal(force, 4096)
    }

    pub fn page_out_fill(&mut self, nfill: i32) -> Option<Page> {
        let force = (self.e_o_s != 0 && !self.lacing_vals.is_empty())
            || (!self.lacing_vals.is_empty() && self.b_o_s == 0);
        self.flush_internal(force, nfill)
    }

    pub fn page_out_bytes(&mut self) -> Option<Vec<u8>> {
        let force = (self.e_o_s != 0 && !self.lacing_vals.is_empty())
            || (!self.lacing_vals.is_empty() && self.b_o_s == 0);
        self.flush_internal_bytes(force, 4096)
    }

    pub fn page_out_fill_bytes(&mut self, nfill: i32) -> Option<Vec<u8>> {
        let force = (self.e_o_s != 0 && !self.lacing_vals.is_empty())
            || (!self.lacing_vals.is_empty() && self.b_o_s == 0);
        self.flush_internal_bytes(force, nfill)
    }

    pub fn page_in(&mut self, page: &Page) -> i32 {
        let header = page.header_bytes();
        if self.check() != 0 || header.len() < 27 {
            return -1;
        }

        self.compact_body();
        self.compact_lacing();

        let version = page.version();
        let mut bos = i32::from(page.is_beginning_of_stream());
        let eos = i32::from(page.is_end_of_stream());
        let continued = i32::from(page.is_continued());
        let granulepos = page.granule_position();
        let serialno = page.stream_serial();
        let pageno = page.sequence_number();
        let segments = header[26] as usize;

        if serialno != self.serialno || version > 0 {
            return -1;
        }

        if pageno != self.pageno {
            for value in &self.lacing_vals[self.lacing_packet..] {
                let len = (value & 0xff) as usize;
                let new_len = self.body_data.len().saturating_sub(len);
                self.body_data.truncate(new_len);
            }
            self.lacing_vals.truncate(self.lacing_packet);
            self.granule_vals.truncate(self.lacing_packet);
            if self.pageno != -1 {
                self.lacing_vals.push(LACING_HOLE);
                self.granule_vals.push(-1);
                self.lacing_packet += 1;
            }
        }

        let mut segptr = 0usize;
        let mut body_offset = 0usize;
        let mut bodysize = page.body_len();

        if continued != 0
            && (self.lacing_vals.is_empty()
                || (self.lacing_vals[self.lacing_vals.len() - 1] & 0xff) < 255
                || self.lacing_vals[self.lacing_vals.len() - 1] == LACING_HOLE)
        {
            bos = 0;
            while segptr < segments {
                let val = header[27 + segptr] as usize;
                body_offset += val;
                bodysize = bodysize.saturating_sub(val);
                segptr += 1;
                if val < 255 {
                    break;
                }
            }
        }

        if bodysize > 0 {
            self.body_data
                .extend_from_slice(&page.body_bytes()[body_offset..body_offset + bodysize]);
        }

        let mut saved = None;
        while segptr < segments {
            let val = header[27 + segptr] as i32;
            let mut lace = val;
            if bos != 0 {
                lace |= LACING_START;
                bos = 0;
            }
            if val < 255 {
                saved = Some(self.lacing_vals.len());
            }
            self.lacing_vals.push(lace);
            self.granule_vals.push(-1);
            segptr += 1;
            if val < 255 {
                self.lacing_packet = self.lacing_vals.len();
            }
        }

        if let Some(saved) = saved {
            self.granule_vals[saved] = granulepos;
        }

        if eos != 0 {
            self.e_o_s = 1;
            if let Some(last) = self.lacing_vals.last_mut() {
                *last |= LACING_END_OF_STREAM;
            }
        }

        self.pageno = pageno + 1;
        0
    }

    fn packet_out_internal(&mut self, advance: bool) -> Result<Option<Packet>, i32> {
        let ptr = self.lacing_returned;
        if self.lacing_packet <= ptr {
            return Ok(None);
        }
        if (self.lacing_vals[ptr] & LACING_HOLE) != 0 {
            self.lacing_returned += 1;
            self.packetno += 1;
            return Err(-1);
        }

        let mut size = (self.lacing_vals[ptr] & 0xff) as usize;
        let mut bytes = size;
        let mut eos = self.lacing_vals[ptr] & LACING_END_OF_STREAM;
        let bos = self.lacing_vals[ptr] & LACING_START;
        let mut current = ptr;
        while size == 255 {
            current += 1;
            let val = self.lacing_vals[current];
            size = (val & 0xff) as usize;
            if (val & LACING_END_OF_STREAM) != 0 {
                eos = LACING_END_OF_STREAM;
            }
            bytes += size;
        }

        let packet = Packet::from_raw_parts(
            self.body_data[self.body_returned..self.body_returned + bytes].to_vec(),
            bos != 0,
            eos != 0,
            self.granule_vals[current],
            self.packetno,
        );

        if advance {
            self.body_returned += bytes;
            self.lacing_returned = current + 1;
            self.packetno += 1;
        }
        Ok(Some(packet))
    }

    pub fn packet_out(&mut self) -> Result<Option<Packet>, i32> {
        if self.check() != 0 {
            return Ok(None);
        }
        self.packet_out_internal(true)
    }

    pub fn packet_peek(&mut self) -> Result<Option<Packet>, i32> {
        if self.check() != 0 {
            return Ok(None);
        }
        self.packet_out_internal(false)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamEncoder {
    raw: RawStreamState,
}

impl StreamEncoder {
    #[must_use]
    pub fn new(serialno: i32) -> Self {
        Self {
            raw: RawStreamState::new(serialno),
        }
    }

    #[must_use]
    pub fn stream_serial(&self) -> i32 {
        self.raw.serialno
    }

    #[must_use]
    pub fn pending_segment_count(&self) -> usize {
        self.raw.pending_segment_count()
    }

    #[must_use]
    pub fn is_end_of_stream(&self) -> bool {
        self.raw.eos() != 0
    }

    pub fn push_packet(&mut self, packet: &Packet) -> Result<(), StreamEncodeError> {
        self.push_packet_data(packet.data(), packet.metadata())
    }

    pub fn push_packet_data(
        &mut self,
        packet: &[u8],
        metadata: PacketMetadata,
    ) -> Result<(), StreamEncodeError> {
        if self
            .raw
            .packet_in_slice(packet, metadata.end_of_stream, metadata.granule_position)
            != 0
        {
            return Err(StreamEncodeError::InvalidPacket);
        }
        Ok(())
    }

    pub fn flush_page(&mut self) -> Option<Page> {
        self.raw.flush()
    }

    pub fn flush_page_with_fill(&mut self, nfill: i32) -> Option<Page> {
        self.raw.flush_fill(nfill)
    }

    pub fn flush_page_bytes(&mut self) -> Option<Vec<u8>> {
        self.raw.flush_bytes()
    }

    pub fn flush_page_bytes_with_fill(&mut self, nfill: i32) -> Option<Vec<u8>> {
        self.raw.flush_fill_bytes(nfill)
    }

    pub fn next_page(&mut self) -> Option<Page> {
        self.raw.page_out()
    }

    pub fn next_page_with_fill(&mut self, nfill: i32) -> Option<Page> {
        self.raw.page_out_fill(nfill)
    }

    pub fn next_page_bytes(&mut self) -> Option<Vec<u8>> {
        self.raw.page_out_bytes()
    }

    pub fn next_page_bytes_with_fill(&mut self, nfill: i32) -> Option<Vec<u8>> {
        self.raw.page_out_fill_bytes(nfill)
    }

    pub fn reset_stream(&mut self, serialno: i32) {
        self.raw.reset_serialno(serialno);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamDecoder {
    raw: RawStreamState,
}

impl StreamDecoder {
    #[must_use]
    pub fn new(serialno: i32) -> Self {
        Self {
            raw: RawStreamState::new(serialno),
        }
    }

    #[must_use]
    pub fn stream_serial(&self) -> i32 {
        self.raw.serialno
    }

    pub fn push_page(&mut self, page: &Page) -> Result<(), StreamDecodeError> {
        if self.raw.page_in(page) != 0 {
            return Err(StreamDecodeError::InvalidPage);
        }
        Ok(())
    }

    pub fn next_packet(&mut self) -> Result<Option<Packet>, StreamDecodeError> {
        self.raw.packet_out().map_err(|_| StreamDecodeError::Gap)
    }

    pub fn peek_packet(&mut self) -> Result<Option<Packet>, StreamDecodeError> {
        self.raw.packet_peek().map_err(|_| StreamDecodeError::Gap)
    }
}
