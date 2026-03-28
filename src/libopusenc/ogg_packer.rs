use alloc::vec;
use alloc::vec::Vec;

const MAX_HEADER_SIZE: usize = 27 + 255;
const MAX_PAGE_SIZE: usize = 255 * 255 + MAX_HEADER_SIZE;

const CRC_LOOKUP: [u32; 256] = [
    0x00000000, 0x04c11db7, 0x09823b6e, 0x0d4326d9, 0x130476dc, 0x17c56b6b, 0x1a864db2,
    0x1e475005, 0x2608edb8, 0x22c9f00f, 0x2f8ad6d6, 0x2b4bcb61, 0x350c9b64, 0x31cd86d3,
    0x3c8ea00a, 0x384fbdbd, 0x4c11db70, 0x48d0c6c7, 0x4593e01e, 0x4152fda9, 0x5f15adac,
    0x5bd4b01b, 0x569796c2, 0x52568b75, 0x6a1936c8, 0x6ed82b7f, 0x639b0da6, 0x675a1011,
    0x791d4014, 0x7ddc5da3, 0x709f7b7a, 0x745e66cd, 0x9823b6e0, 0x9ce2ab57, 0x91a18d8e,
    0x95609039, 0x8b27c03c, 0x8fe6dd8b, 0x82a5fb52, 0x8664e6e5, 0xbe2b5b58, 0xbaea46ef,
    0xb7a96036, 0xb3687d81, 0xad2f2d84, 0xa9ee3033, 0xa4ad16ea, 0xa06c0b5d, 0xd4326d90,
    0xd0f37027, 0xddb056fe, 0xd9714b49, 0xc7361b4c, 0xc3f706fb, 0xceb42022, 0xca753d95,
    0xf23a8028, 0xf6fb9d9f, 0xfbb8bb46, 0xff79a6f1, 0xe13ef6f4, 0xe5ffeb43, 0xe8bccd9a,
    0xec7dd02d, 0x34867077, 0x30476dc0, 0x3d044b19, 0x39c556ae, 0x278206ab, 0x23431b1c,
    0x2e003dc5, 0x2ac12072, 0x128e9dcf, 0x164f8078, 0x1b0ca6a1, 0x1fcdbb16, 0x018aeb13,
    0x054bf6a4, 0x0808d07d, 0x0cc9cdca, 0x7897ab07, 0x7c56b6b0, 0x71159069, 0x75d48dde,
    0x6b93dddb, 0x6f52c06c, 0x6211e6b5, 0x66d0fb02, 0x5e9f46bf, 0x5a5e5b08, 0x571d7dd1,
    0x53dc6066, 0x4d9b3063, 0x495a2dd4, 0x44190b0d, 0x40d816ba, 0xaca5c697, 0xa864db20,
    0xa527fdf9, 0xa1e6e04e, 0xbfa1b04b, 0xbb60adfc, 0xb6238b25, 0xb2e29692, 0x8aad2b2f,
    0x8e6c3698, 0x832f1041, 0x87ee0df6, 0x99a95df3, 0x9d684044, 0x902b669d, 0x94ea7b2a,
    0xe0b41de7, 0xe4750050, 0xe9362689, 0xedf73b3e, 0xf3b06b3b, 0xf771768c, 0xfa325055,
    0xfef34de2, 0xc6bcf05f, 0xc27dede8, 0xcf3ecb31, 0xcbffd686, 0xd5b88683, 0xd1799b34,
    0xdc3abded, 0xd8fba05a, 0x690ce0ee, 0x6dcdfd59, 0x608edb80, 0x644fc637, 0x7a089632,
    0x7ec98b85, 0x738aad5c, 0x774bb0eb, 0x4f040d56, 0x4bc510e1, 0x46863638, 0x42472b8f,
    0x5c007b8a, 0x58c1663d, 0x558240e4, 0x51435d53, 0x251d3b9e, 0x21dc2629, 0x2c9f00f0,
    0x285e1d47, 0x36194d42, 0x32d850f5, 0x3f9b762c, 0x3b5a6b9b, 0x0315d626, 0x07d4cb91,
    0x0a97ed48, 0x0e56f0ff, 0x1011a0fa, 0x14d0bd4d, 0x19939b94, 0x1d528623, 0xf12f560e,
    0xf5ee4bb9, 0xf8ad6d60, 0xfc6c70d7, 0xe22b20d2, 0xe6ea3d65, 0xeba91bbc, 0xef68060b,
    0xd727bbb6, 0xd3e6a601, 0xdea580d8, 0xda649d6f, 0xc423cd6a, 0xc0e2d0dd, 0xcda1f604,
    0xc960ebb3, 0xbd3e8d7e, 0xb9ff90c9, 0xb4bcb610, 0xb07daba7, 0xae3afba2, 0xaafbe615,
    0xa7b8c0cc, 0xa379dd7b, 0x9b3660c6, 0x9ff77d71, 0x92b45ba8, 0x9675461f, 0x8832161a,
    0x8cf30bad, 0x81b02d74, 0x857130c3, 0x5d8a9099, 0x594b8d2e, 0x5408abf7, 0x50c9b640,
    0x4e8ee645, 0x4a4ffbf2, 0x470cdd2b, 0x43cdc09c, 0x7b827d21, 0x7f436096, 0x7200464f,
    0x76c15bf8, 0x68860bfd, 0x6c47164a, 0x61043093, 0x65c52d24, 0x119b4be9, 0x155a565e,
    0x18197087, 0x1cd86d30, 0x029f3d35, 0x065e2082, 0x0b1d065b, 0x0fdc1bec, 0x3793a651,
    0x3352bbe6, 0x3e119d3f, 0x3ad08088, 0x2497d08d, 0x2056cd3a, 0x2d15ebe3, 0x29d4f654,
    0xc5a92679, 0xc1683bce, 0xcc2b1d17, 0xc8ea00a0, 0xd6ad50a5, 0xd26c4d12, 0xdf2f6bcb,
    0xdbee767c, 0xe3a1cbc1, 0xe760d676, 0xea23f0af, 0xeee2ed18, 0xf0a5bd1d, 0xf464a0aa,
    0xf9278673, 0xfde69bc4, 0x89b8fd09, 0x8d79e0be, 0x803ac667, 0x84fbdbd0, 0x9abc8bd5,
    0x9e7d9662, 0x933eb0bb, 0x97ffad0c, 0xafb010b1, 0xab710d06, 0xa6322bdf, 0xa2f33668,
    0xbcb4666d, 0xb8757bda, 0xb5365d03, 0xb1f740b4,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OggPage {
    granulepos: u64,
    buf_pos: usize,
    buf_size: usize,
    lacing_pos: usize,
    lacing_size: usize,
    flags: u8,
    pageno: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OggPacker {
    serialno: i32,
    buf: Vec<u8>,
    user_buf_start: Option<usize>,
    user_buf_reserved: usize,
    buf_fill: usize,
    buf_begin: usize,
    lacing: Vec<u8>,
    lacing_fill: usize,
    lacing_begin: usize,
    pages: Vec<OggPage>,
    muxing_delay: u64,
    is_eos: bool,
    curr_granule: u64,
    last_granule: u64,
    pageno: usize,
}

impl OggPacker {
    #[must_use]
    pub fn new(serialno: i32) -> Self {
        Self {
            serialno,
            buf: vec![0; MAX_PAGE_SIZE],
            user_buf_start: None,
            user_buf_reserved: 0,
            buf_fill: 0,
            buf_begin: 0,
            lacing: vec![0; 256],
            lacing_fill: 0,
            lacing_begin: 0,
            pages: Vec::with_capacity(10),
            muxing_delay: 0,
            is_eos: false,
            curr_granule: 0,
            last_granule: 0,
            pageno: 0,
        }
    }

    pub fn set_muxing_delay(&mut self, delay: u64) {
        self.muxing_delay = delay;
    }

    pub fn get_packet_buffer(&mut self, bytes: usize) -> Option<&mut [u8]> {
        let required = self.buf_fill.checked_add(bytes)?;
        if required > self.buf.len() {
            self.shift_buffer();
            if required > self.buf.len() {
                let new_size = required.checked_add(MAX_HEADER_SIZE)?.checked_mul(3)? / 2;
                self.buf.resize(new_size, 0);
            }
        }
        self.user_buf_start = Some(self.buf_fill);
        self.user_buf_reserved = bytes;
        Some(&mut self.buf[self.buf_fill..self.buf_fill + bytes])
    }

    pub fn commit_packet(&mut self, bytes: usize, granulepos: u64, eos: bool) -> Result<(), ()> {
        let nb_255s = bytes / 255;
        if self.lacing_fill - self.lacing_begin + nb_255s + 1 > 255
            || (self.muxing_delay != 0
                && granulepos.saturating_sub(self.last_granule) > self.muxing_delay)
        {
            self.flush_page();
        }

        let Some(start) = self.user_buf_start.take() else {
            return Err(());
        };
        if bytes > self.user_buf_reserved {
            return Err(());
        }
        self.user_buf_reserved = 0;
        self.buf_fill = start + bytes;

        if self.lacing_fill + nb_255s + 1 > self.lacing.len() {
            self.shift_buffer();
            if self.lacing_fill + nb_255s + 1 > self.lacing.len() {
                let new_size = (self.lacing_fill + nb_255s + 1) * 3 / 2;
                self.lacing.resize(new_size, 0);
            }
        }

        for value in &mut self.lacing[self.lacing_fill..self.lacing_fill + nb_255s] {
            *value = 255;
        }
        self.lacing[self.lacing_fill + nb_255s] = (bytes - 255 * nb_255s) as u8;
        self.lacing_fill += nb_255s + 1;
        self.curr_granule = granulepos;
        self.is_eos = eos;
        if self.muxing_delay != 0
            && granulepos.saturating_sub(self.last_granule) >= self.muxing_delay
        {
            self.flush_page();
        }
        Ok(())
    }

    pub fn flush_page(&mut self) -> bool {
        if self.lacing_fill == self.lacing_begin {
            return false;
        }

        let mut nb_lacing = self.lacing_fill - self.lacing_begin;
        let mut cont = 0u8;
        while nb_lacing > 0 {
            let mut page = OggPage {
                granulepos: self.curr_granule,
                buf_pos: self.buf_begin,
                buf_size: 0,
                lacing_pos: self.lacing_begin,
                lacing_size: nb_lacing,
                flags: cont,
                pageno: self.pageno,
            };

            if page.lacing_size > 255 {
                page.lacing_size = 255;
                page.buf_size = self.lacing[self.lacing_begin..self.lacing_begin + 255]
                    .iter()
                    .map(|&value| value as usize)
                    .sum();
                page.granulepos = u64::MAX;
                cont = 0x01;
            } else {
                page.buf_size = self.buf_fill - self.buf_begin;
                if self.is_eos {
                    page.flags |= 0x04;
                }
            }

            if page.pageno == 0 {
                page.flags |= 0x02;
            }
            self.pageno += 1;
            nb_lacing -= page.lacing_size;
            self.lacing_begin += page.lacing_size;
            self.buf_begin += page.buf_size;
            self.pages.push(page);
        }

        self.last_granule = self.curr_granule;
        true
    }

    pub fn get_next_page(&mut self) -> Option<Vec<u8>> {
        let page = *self.pages.first()?;
        let header_size = 27 + page.lacing_size;
        let len = header_size + page.buf_size;
        let mut out = vec![0u8; len];

        out[..4].copy_from_slice(b"OggS");
        out[4] = 0;
        out[5] = page.flags;
        out[6..14].copy_from_slice(&page.granulepos.to_le_bytes());
        out[14..18].copy_from_slice(&(self.serialno as u32).to_le_bytes());
        out[18..22].copy_from_slice(&(page.pageno as u32).to_le_bytes());
        out[26] = page.lacing_size as u8;
        out[27..27 + page.lacing_size]
            .copy_from_slice(&self.lacing[page.lacing_pos..page.lacing_pos + page.lacing_size]);
        out[header_size..].copy_from_slice(&self.buf[page.buf_pos..page.buf_pos + page.buf_size]);
        ogg_page_checksum_set(&mut out);

        self.pages.remove(0);
        self.shift_buffer();
        Some(out)
    }

    pub fn chain(&mut self, serialno: i32) {
        self.flush_page();
        self.serialno = serialno;
        self.curr_granule = 0;
        self.last_granule = 0;
        self.is_eos = false;
        self.pageno = 0;
    }

    fn shift_buffer(&mut self) {
        let buf_shift = self.pages.first().map_or(self.buf_begin, |page| page.buf_pos);
        let lacing_shift = self
            .pages
            .first()
            .map_or(self.lacing_begin, |page| page.lacing_pos);

        if 4 * lacing_shift > self.lacing_fill {
            let retained = self.lacing_fill - lacing_shift;
            self.lacing.copy_within(lacing_shift..self.lacing_fill, 0);
            for page in &mut self.pages {
                page.lacing_pos -= lacing_shift;
            }
            self.lacing_fill = retained;
            self.lacing_begin -= lacing_shift;
        }

        if 4 * buf_shift > self.buf_fill {
            let retained = self.buf_fill - buf_shift;
            self.buf.copy_within(buf_shift..self.buf_fill, 0);
            for page in &mut self.pages {
                page.buf_pos -= buf_shift;
            }
            self.buf_fill = retained;
            self.buf_begin -= buf_shift;
            if let Some(start) = self.user_buf_start.as_mut() {
                *start -= buf_shift;
            }
        }
    }
}

fn ogg_page_checksum_set(page: &mut [u8]) {
    page[22] = 0;
    page[23] = 0;
    page[24] = 0;
    page[25] = 0;

    let mut crc_reg = 0u32;
    for &byte in page.iter() {
        crc_reg = (crc_reg << 8) ^ CRC_LOOKUP[((crc_reg >> 24) as u8 ^ byte) as usize];
    }

    page[22..26].copy_from_slice(&crc_reg.to_le_bytes());
}
