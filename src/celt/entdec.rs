#![allow(dead_code)]

//! Range decoder implementation mirroring `celt/entdec.c`.
//!
//! The decoder operates on the shared [`EcCtx`](crate::celt::entcode::EcCtx)
//! structure and provides the scalar range decoding logic used throughout the
//! CELT bitstream reader. The port keeps the arithmetic and control flow close
//! to the C implementation to ease future verification against the reference
//! sources.

use core::cmp::min;

use crate::celt::entcode::{
    EC_CODE_BITS, EC_CODE_BOT, EC_CODE_EXTRA, EC_CODE_TOP, EC_SYM_BITS, EC_SYM_MAX, EC_UINT_BITS,
    EC_WINDOW_SIZE, EcCtx, EcWindow, celt_udiv,
};
use crate::celt::types::{OpusInt32, OpusUint32};

/// Range decoder operating on a mutable byte buffer.
#[derive(Debug)]
pub struct EcDec<'a> {
    ctx: EcCtx<'a>,
}

impl<'a> EcDec<'a> {
    /// Creates a new decoder backed by the provided buffer.
    #[must_use]
    pub fn new(buf: &'a mut [u8]) -> Self {
        let storage = buf.len() as OpusUint32;
        let mut ctx = EcCtx::new(buf);
        ctx.storage = storage;
        ctx.end_offs = 0;
        ctx.end_window = 0;
        ctx.nend_bits = 0;
        ctx.nbits_total = (EC_CODE_BITS as OpusInt32) + 1
            - (((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) as OpusInt32)
                * EC_SYM_BITS as OpusInt32;
        ctx.offs = 0;
        ctx.rng = 1u32 << EC_CODE_EXTRA;
        ctx.rem = 0;
        ctx.ext = 0;
        ctx.error = 0;

        let mut dec = Self { ctx };
        dec.ctx.rem = OpusInt32::from(dec.read_byte());
        dec.ctx.val =
            dec.ctx.rng - 1 - ((dec.ctx.rem as OpusUint32) >> (EC_SYM_BITS - EC_CODE_EXTRA));
        dec.normalize();
        dec
    }

    /// Borrows the underlying entropy context.
    #[must_use]
    pub fn ctx(&self) -> &EcCtx<'a> {
        &self.ctx
    }

    /// Borrows the underlying entropy context mutably.
    #[must_use]
    pub fn ctx_mut(&mut self) -> &mut EcCtx<'a> {
        &mut self.ctx
    }

    fn read_byte(&mut self) -> u8 {
        if self.ctx.offs < self.ctx.storage {
            let byte = self.ctx.buf[self.ctx.offs as usize];
            self.ctx.offs += 1;
            byte
        } else {
            0
        }
    }

    fn read_byte_from_end(&mut self) -> u8 {
        if self.ctx.end_offs < self.ctx.storage {
            self.ctx.end_offs += 1;
            let idx = (self.ctx.storage - self.ctx.end_offs) as usize;
            self.ctx.buf[idx]
        } else {
            0
        }
    }

    fn normalize(&mut self) {
        while self.ctx.rng <= EC_CODE_BOT {
            self.ctx.nbits_total += EC_SYM_BITS as OpusInt32;
            self.ctx.rng <<= EC_SYM_BITS;
            let mut sym = self.ctx.rem as OpusUint32;
            self.ctx.rem = OpusInt32::from(self.read_byte());
            sym = ((sym << EC_SYM_BITS) | (self.ctx.rem as OpusUint32))
                >> (EC_SYM_BITS - EC_CODE_EXTRA);
            let sub = EC_SYM_MAX & !sym;
            self.ctx.val = ((self.ctx.val << EC_SYM_BITS) + sub) & (EC_CODE_TOP - 1);
        }
    }

    /// Decodes a symbol with total frequency `ft`.
    #[must_use]
    pub fn decode(&mut self, ft: OpusUint32) -> OpusUint32 {
        let ext = celt_udiv(self.ctx.rng, ft);
        self.ctx.ext = ext;
        let s = (self.ctx.val / ext) as OpusUint32;
        ft - min(s + 1, ft)
    }

    /// Decodes a binary symbol with `bits` precision.
    #[must_use]
    pub fn decode_bin(&mut self, bits: u32) -> OpusUint32 {
        self.ctx.ext = self.ctx.rng >> bits;
        let s = (self.ctx.val / self.ctx.ext) as OpusUint32;
        (1u32 << bits) - min(s + 1, 1u32 << bits)
    }

    /// Updates the decoder interval with the symbol range `[fl, fh)`.
    pub fn update(&mut self, fl: OpusUint32, fh: OpusUint32, ft: OpusUint32) {
        let s = self.ctx.ext.wrapping_mul(ft - fh);
        self.ctx.val = self.ctx.val.wrapping_sub(s);
        self.ctx.rng = if fl > 0 {
            self.ctx.ext.wrapping_mul(fh - fl)
        } else {
            self.ctx.rng.wrapping_sub(s)
        };
        self.normalize();
    }

    /// Decodes a bit with probability `1/(1<<logp)` of being 1.
    #[must_use]
    pub fn dec_bit_logp(&mut self, logp: u32) -> OpusInt32 {
        let r = self.ctx.rng;
        let d = self.ctx.val;
        let s = r >> logp;
        let ret = OpusInt32::from(d < s);
        if ret == 0 {
            self.ctx.val = d - s;
        }
        self.ctx.rng = if ret != 0 { s } else { r - s };
        self.normalize();
        ret
    }

    /// Decodes a symbol using an inverse cumulative distribution in 8-bit form.
    #[must_use]
    pub fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> OpusInt32 {
        let mut s = self.ctx.rng;
        let d = self.ctx.val;
        let r = s >> ftb;
        let mut ret: OpusInt32 = -1;
        loop {
            ret += 1;
            let t = s;
            let idx = ret as usize;
            debug_assert!(idx < icdf.len());
            s = r.wrapping_mul(OpusUint32::from(icdf[idx]));
            if d >= s {
                self.ctx.val = d.wrapping_sub(s);
                self.ctx.rng = t.wrapping_sub(s);
                self.normalize();
                return ret;
            }
        }
    }

    /// Decodes a symbol using a 16-bit inverse cumulative distribution.
    #[must_use]
    pub fn dec_icdf16(&mut self, icdf: &[u16], ftb: u32) -> OpusInt32 {
        let mut s = self.ctx.rng;
        let d = self.ctx.val;
        let r = s >> ftb;
        let mut ret: OpusInt32 = -1;
        loop {
            ret += 1;
            let t = s;
            let idx = ret as usize;
            debug_assert!(idx < icdf.len());
            s = r.wrapping_mul(OpusUint32::from(icdf[idx]));
            if d >= s {
                self.ctx.val = d.wrapping_sub(s);
                self.ctx.rng = t.wrapping_sub(s);
                self.normalize();
                return ret;
            }
        }
    }

    /// Decodes an unsigned integer in `[0, ft)`.
    #[must_use]
    pub fn dec_uint(&mut self, mut ft: OpusUint32) -> OpusUint32 {
        assert!(ft > 1);
        ft -= 1;
        let mut ftb = 32 - ft.leading_zeros();
        if ftb > EC_UINT_BITS as u32 {
            ftb -= EC_UINT_BITS as u32;
            let ft_small = (ft >> ftb) + 1;
            let s = self.decode(ft_small);
            self.update(s, s + 1, ft_small);
            let t = ((s as OpusUint32) << ftb) | self.dec_bits(ftb);
            if t <= ft {
                return t;
            }
            self.ctx.error = 1;
            ft
        } else {
            ft += 1;
            let s = self.decode(ft);
            self.update(s, s + 1, ft);
            s
        }
    }

    /// Reads raw bits from the tail of the stream.
    #[must_use]
    pub fn dec_bits(&mut self, bits: u32) -> OpusUint32 {
        let mut window: EcWindow = self.ctx.end_window;
        let mut available: OpusInt32 = self.ctx.nend_bits;
        if (available as u32) < bits {
            while available <= (EC_WINDOW_SIZE as OpusInt32) - EC_SYM_BITS as OpusInt32 {
                window |= EcWindow::from(self.read_byte_from_end()) << (available as u32);
                available += EC_SYM_BITS as OpusInt32;
            }
        }
        let mask = (1u32 << bits) - 1;
        let ret = (window & mask) as OpusUint32;
        window >>= bits;
        available -= bits as OpusInt32;
        self.ctx.end_window = window;
        self.ctx.nend_bits = available;
        self.ctx.nbits_total += bits as OpusInt32;
        ret
    }

    /// Returns the number of whole bytes read from the range coder.
    #[must_use]
    pub fn range_bytes(&self) -> OpusUint32 {
        self.ctx.range_bytes()
    }
}

impl<'a> core::ops::Deref for EcDec<'a> {
    type Target = EcCtx<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl<'a> core::ops::DerefMut for EcDec<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::EcDec;
    use crate::celt::EcEnc;
    use crate::celt::entcode::{EC_CODE_BITS, EC_CODE_EXTRA, EC_SYM_BITS, EC_WINDOW_SIZE};
    use crate::celt::types::{OpusInt32, OpusUint32};

    fn reference_initial_state(
        buf: &[u8],
    ) -> (OpusInt32, OpusUint32, OpusUint32, OpusUint32, OpusInt32) {
        let storage = buf.len() as OpusUint32;
        let mut offs = 0u32;
        let mut rng = 1u32 << EC_CODE_EXTRA;
        let nbits_total = (EC_CODE_BITS as OpusInt32) + 1
            - (((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) as OpusInt32)
                * EC_SYM_BITS as OpusInt32;
        let rem = if offs < storage {
            let b = buf[offs as usize];
            offs += 1;
            OpusInt32::from(b)
        } else {
            0
        };
        let mut val = rng - 1 - ((rem as OpusUint32) >> (EC_SYM_BITS - EC_CODE_EXTRA));
        let mut nbits_total = nbits_total;
        let mut rem = rem;
        while rng <= super::EC_CODE_BOT {
            nbits_total += EC_SYM_BITS as OpusInt32;
            rng <<= EC_SYM_BITS;
            let mut sym = rem as OpusUint32;
            rem = if offs < storage {
                let b = buf[offs as usize];
                offs += 1;
                OpusInt32::from(b)
            } else {
                0
            };
            sym = ((sym << EC_SYM_BITS) | (rem as OpusUint32)) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            let sub = super::EC_SYM_MAX & !sym;
            val = ((val << EC_SYM_BITS) + sub) & (super::EC_CODE_TOP - 1);
        }
        (nbits_total, rng, val, offs, rem)
    }

    #[test]
    fn decoder_initialises_like_reference() {
        let mut buf = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let reference = reference_initial_state(&buf);
        let dec = EcDec::new(&mut buf);
        assert_eq!(dec.nbits_total, reference.0);
        assert_eq!(dec.rng, reference.1);
        assert_eq!(dec.val, reference.2);
        assert_eq!(dec.offs, reference.3);
        assert_eq!(dec.rem, reference.4);
    }

    #[test]
    fn dec_bits_reads_from_end() {
        let mut buf = vec![0u8; 4];
        buf[3] = 0b1011_0101;
        let mut dec = EcDec::new(&mut buf);
        dec.end_offs = 0;
        dec.end_window = 0;
        dec.nend_bits = 0;
        dec.nbits_total = 0;
        let bits = dec.dec_bits(4);
        assert_eq!(bits, 0b0101);
        assert_eq!(dec.nbits_total, 4);
        assert_eq!(dec.nend_bits, EC_WINDOW_SIZE as OpusInt32 - 4);
    }

    #[test]
    fn dec_uint_respects_requested_range() {
        let mut buf = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let mut dec = EcDec::new(&mut buf);
        let value = dec.dec_uint(16);
        assert!(value < 16);
        assert_eq!(dec.error, 0);
    }

    #[test]
    fn dec_icdf_matches_reference_for_simple_case() {
        let icdf = [192u8, 128, 64, 0];
        let mut storage = vec![0u8; 16];
        let mut enc = EcEnc::new(storage.as_mut_slice());
        enc.enc_icdf(2, &icdf, 8);
        enc.enc_done();
        let size = enc.range_bytes() as usize;

        let mut data = storage[..size].to_vec();
        let mut dec = EcDec::new(data.as_mut_slice());
        let ret = dec.dec_icdf(&icdf, 8);
        assert_eq!(ret, 2);
    }
}
