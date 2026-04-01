extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

const BUFFER_INCREMENT: usize = 256;

const MASK: [u32; 33] = [
    0x00000000, 0x00000001, 0x00000003, 0x00000007, 0x0000000f, 0x0000001f, 0x0000003f, 0x0000007f,
    0x000000ff, 0x000001ff, 0x000003ff, 0x000007ff, 0x00000fff, 0x00001fff, 0x00003fff, 0x00007fff,
    0x0000ffff, 0x0001ffff, 0x0003ffff, 0x0007ffff, 0x000fffff, 0x001fffff, 0x003fffff, 0x007fffff,
    0x00ffffff, 0x01ffffff, 0x03ffffff, 0x07ffffff, 0x0fffffff, 0x1fffffff, 0x3fffffff, 0x7fffffff,
    0xffffffff,
];

const MASK8B: [u8; 9] = [0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, 0xff];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOrder {
    Lsb,
    Msb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitPackerError {
    InvalidBitCount,
    ReadPastEnd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawBitPacker {
    order: BitOrder,
    buffer: Vec<u8>,
    endbyte: usize,
    endbit: usize,
    valid: bool,
}

impl RawBitPacker {
    #[must_use]
    pub(crate) fn new(order: BitOrder) -> Self {
        Self {
            order,
            buffer: vec![0; BUFFER_INCREMENT],
            endbyte: 0,
            endbit: 0,
            valid: true,
        }
    }

    #[must_use]
    pub(crate) fn writecheck(&self) -> i32 {
        if self.valid { 0 } else { -1 }
    }

    fn ensure_capacity(&mut self, needed_tail: usize) {
        let needed = self.endbyte.saturating_add(needed_tail);
        if needed >= self.buffer.len() {
            self.buffer.resize(needed + BUFFER_INCREMENT, 0);
        }
    }

    pub(crate) fn reset(&mut self) {
        if !self.buffer.is_empty() {
            self.buffer.fill(0);
        }
        self.endbyte = 0;
        self.endbit = 0;
        self.valid = true;
    }

    pub(crate) fn writeclear(&mut self) {
        self.buffer.clear();
        self.endbyte = 0;
        self.endbit = 0;
        self.valid = false;
    }

    pub(crate) fn writetrunc(&mut self, bits: usize) {
        if !self.valid {
            return;
        }
        let bytes = bits >> 3;
        let rem = bits - bytes * 8;
        self.endbyte = bytes;
        self.endbit = rem;
        if self.buffer.len() <= self.endbyte {
            self.buffer.resize(self.endbyte + 1, 0);
        }
        match self.order {
            BitOrder::Lsb => self.buffer[self.endbyte] &= MASK[rem] as u8,
            BitOrder::Msb => self.buffer[self.endbyte] &= MASK8B[rem],
        }
    }

    pub(crate) fn writealign(&mut self) -> Result<(), BitPackerError> {
        let bits = 8 - self.endbit;
        if bits < 8 {
            self.write(0, bits as i32)?;
        }
        Ok(())
    }

    pub(crate) fn writecopy(&mut self, source: &[u8], bits: usize) -> Result<(), BitPackerError> {
        let bytes = bits / 8;
        let trailing_bits = bits - bytes * 8;

        if self.endbit == 0 {
            self.ensure_capacity(bits.div_ceil(8) + 1);
            self.buffer[self.endbyte..self.endbyte + bytes].copy_from_slice(&source[..bytes]);
            self.endbyte += bytes;
            if self.endbyte >= self.buffer.len() {
                self.buffer.resize(self.endbyte + 1, 0);
            }
            self.buffer[self.endbyte] = 0;
        } else {
            for &byte in &source[..bytes] {
                self.write(byte as u32, 8)?;
            }
        }

        if trailing_bits > 0 {
            let value = match self.order {
                BitOrder::Lsb => source[bytes] as u32,
                BitOrder::Msb => (source[bytes] as u32) >> (8 - trailing_bits),
            };
            self.write(value, trailing_bits as i32)?;
        }
        Ok(())
    }

    pub(crate) fn write(&mut self, mut value: u32, bits: i32) -> Result<(), BitPackerError> {
        if !(0..=32).contains(&bits) {
            self.writeclear();
            return Err(BitPackerError::InvalidBitCount);
        }
        let bits = bits as usize;
        if !self.valid {
            return Ok(());
        }
        self.ensure_capacity(5);
        match self.order {
            BitOrder::Lsb => {
                value &= MASK[bits];
                let total_bits = bits + self.endbit;
                self.buffer[self.endbyte] |= (value << self.endbit) as u8;
                if total_bits >= 8 {
                    self.buffer[self.endbyte + 1] = (value >> (8 - self.endbit)) as u8;
                    if total_bits >= 16 {
                        self.buffer[self.endbyte + 2] = (value >> (16 - self.endbit)) as u8;
                        if total_bits >= 24 {
                            self.buffer[self.endbyte + 3] = (value >> (24 - self.endbit)) as u8;
                            if total_bits >= 32 {
                                self.buffer[self.endbyte + 4] = if self.endbit == 0 {
                                    0
                                } else {
                                    (value >> (32 - self.endbit)) as u8
                                };
                            }
                        }
                    }
                }
                self.endbyte += total_bits / 8;
                self.endbit = total_bits & 7;
            }
            BitOrder::Msb => {
                value = (value & MASK[bits]) << (32 - bits);
                let total_bits = bits + self.endbit;
                self.buffer[self.endbyte] |= (value >> (24 + self.endbit)) as u8;
                if total_bits >= 8 {
                    self.buffer[self.endbyte + 1] = (value >> (16 + self.endbit)) as u8;
                    if total_bits >= 16 {
                        self.buffer[self.endbyte + 2] = (value >> (8 + self.endbit)) as u8;
                        if total_bits >= 24 {
                            self.buffer[self.endbyte + 3] = (value >> self.endbit) as u8;
                            if total_bits >= 32 {
                                self.buffer[self.endbyte + 4] = if self.endbit == 0 {
                                    0
                                } else {
                                    (value << (8 - self.endbit)) as u8
                                };
                            }
                        }
                    }
                }
                self.endbyte += total_bits / 8;
                self.endbit = total_bits & 7;
            }
        }
        Ok(())
    }

    #[must_use]
    pub(crate) fn bytes(&self) -> usize {
        self.endbyte + usize::from(self.endbit > 0)
    }

    #[must_use]
    pub(crate) fn bits(&self) -> usize {
        self.endbyte * 8 + self.endbit
    }

    #[must_use]
    pub(crate) fn buffer(&self) -> &[u8] {
        &self.buffer[..self.bytes()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawBitUnpacker {
    order: BitOrder,
    buffer: Vec<u8>,
    storage: usize,
    endbyte: usize,
    endbit: usize,
}

impl RawBitUnpacker {
    #[must_use]
    pub(crate) fn new(order: BitOrder, bytes: &[u8]) -> Self {
        Self {
            order,
            buffer: bytes.to_vec(),
            storage: bytes.len(),
            endbyte: 0,
            endbit: 0,
        }
    }

    pub(crate) fn look(&self, bits: i32) -> Result<u32, BitPackerError> {
        if !(0..=32).contains(&bits) {
            return Err(BitPackerError::InvalidBitCount);
        }
        let bits = bits as usize;
        if self.endbyte >= self.storage {
            return if bits == 0 {
                Ok(0)
            } else {
                Err(BitPackerError::ReadPastEnd)
            };
        }
        let available = self.storage * 8 - self.endbyte * 8 - self.endbit;
        if bits > available {
            return Err(BitPackerError::ReadPastEnd);
        }
        let ptr = self.endbyte;
        let result = match self.order {
            BitOrder::Lsb => {
                let mut value = (self.buffer[ptr] as u32) >> self.endbit;
                if bits > 8 - self.endbit {
                    value |= (self.buffer[ptr + 1] as u32) << (8 - self.endbit);
                    if bits > 16 - self.endbit {
                        value |= (self.buffer[ptr + 2] as u32) << (16 - self.endbit);
                        if bits > 24 - self.endbit {
                            value |= (self.buffer[ptr + 3] as u32) << (24 - self.endbit);
                            if bits > 32 - self.endbit && self.endbit != 0 {
                                value |= (self.buffer[ptr + 4] as u32) << (32 - self.endbit);
                            }
                        }
                    }
                }
                value & MASK[bits]
            }
            BitOrder::Msb => {
                let mut value = (self.buffer[ptr] as u32) << (24 + self.endbit);
                if bits > 8 - self.endbit {
                    value |= (self.buffer[ptr + 1] as u32) << (16 + self.endbit);
                    if bits > 16 - self.endbit {
                        value |= (self.buffer[ptr + 2] as u32) << (8 + self.endbit);
                        if bits > 24 - self.endbit {
                            value |= (self.buffer[ptr + 3] as u32) << self.endbit;
                            if bits > 32 - self.endbit && self.endbit != 0 {
                                value |= (self.buffer[ptr + 4] as u32) >> (8 - self.endbit);
                            }
                        }
                    }
                }
                (value >> (32 - bits)) & MASK[bits]
            }
        };
        Ok(result)
    }

    pub(crate) fn look1(&self) -> Result<u32, BitPackerError> {
        self.look(1)
    }

    pub(crate) fn adv(&mut self, bits: i32) -> Result<(), BitPackerError> {
        if bits < 0 {
            return Err(BitPackerError::InvalidBitCount);
        }
        let bits = bits as usize;
        let available = self.storage * 8 - self.endbyte * 8 - self.endbit;
        if bits > available {
            self.endbyte = self.storage;
            self.endbit = 0;
            return Err(BitPackerError::ReadPastEnd);
        }
        let total = self.endbit + bits;
        self.endbyte += total / 8;
        self.endbit = total & 7;
        Ok(())
    }

    pub(crate) fn adv1(&mut self) -> Result<(), BitPackerError> {
        self.adv(1)
    }

    pub(crate) fn read(&mut self, bits: i32) -> Result<u32, BitPackerError> {
        let value = self.look(bits)?;
        self.adv(bits)?;
        Ok(value)
    }

    pub(crate) fn read1(&mut self) -> Result<u32, BitPackerError> {
        self.read(1)
    }

    #[must_use]
    pub(crate) fn bytes(&self) -> usize {
        self.endbyte + usize::from(self.endbit > 0)
    }

    #[must_use]
    pub(crate) fn bits(&self) -> usize {
        self.endbyte * 8 + self.endbit
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitPacker {
    raw: RawBitPacker,
}

impl BitPacker {
    #[must_use]
    pub fn new(order: BitOrder) -> Self {
        Self {
            raw: RawBitPacker::new(order),
        }
    }

    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.raw.writecheck() == 0
    }

    pub fn reset(&mut self) {
        self.raw.reset();
    }

    pub fn clear(&mut self) {
        self.raw.writeclear();
    }

    pub fn truncate_bits(&mut self, bits: usize) {
        self.raw.writetrunc(bits);
    }

    pub fn align_to_byte(&mut self) -> Result<(), BitPackerError> {
        self.raw.writealign()
    }

    pub fn copy_bits(&mut self, source: &[u8], bits: usize) -> Result<(), BitPackerError> {
        self.raw.writecopy(source, bits)
    }

    pub fn write_bits(&mut self, value: u32, bits: i32) -> Result<(), BitPackerError> {
        self.raw.write(value, bits)
    }

    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.raw.bytes()
    }

    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.raw.bits()
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.raw.buffer()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitUnpacker {
    raw: RawBitUnpacker,
}

impl BitUnpacker {
    #[must_use]
    pub fn new(order: BitOrder, bytes: &[u8]) -> Self {
        Self {
            raw: RawBitUnpacker::new(order, bytes),
        }
    }

    pub fn peek_bits(&self, bits: i32) -> Result<u32, BitPackerError> {
        self.raw.look(bits)
    }

    pub fn peek_one(&self) -> Result<u32, BitPackerError> {
        self.raw.look1()
    }

    pub fn skip_bits(&mut self, bits: i32) -> Result<(), BitPackerError> {
        self.raw.adv(bits)
    }

    pub fn skip_one(&mut self) -> Result<(), BitPackerError> {
        self.raw.adv1()
    }

    pub fn read_bits(&mut self, bits: i32) -> Result<u32, BitPackerError> {
        self.raw.read(bits)
    }

    pub fn read_one(&mut self) -> Result<u32, BitPackerError> {
        self.raw.read1()
    }

    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.raw.bytes()
    }

    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.raw.bits()
    }
}
