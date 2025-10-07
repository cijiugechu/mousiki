#![allow(dead_code)]

//! Range encoder implementation mirroring `celt/entenc.c`.
//!
//! The encoder operates on the shared [`EcCtx`](crate::celt::entcode::EcCtx)
//! structure and maintains behaviour close to the C implementation to ease
//! verification against the reference sources.

use crate::celt::entcode::{
    EC_CODE_BITS, EC_CODE_BOT, EC_CODE_SHIFT, EC_CODE_TOP, EC_SYM_BITS, EC_SYM_MAX, EC_UINT_BITS,
    EC_WINDOW_SIZE, EcCtx, EcWindow, celt_udiv, ec_ilog,
};
use crate::celt::types::{OpusInt32, OpusUint32};
use alloc::vec::Vec;

/// Range encoder backed by a mutable byte slice.
#[derive(Debug)]
pub struct EcEnc<'a> {
    ctx: EcCtx<'a>,
}

impl<'a> EcEnc<'a> {
    /// Creates a new encoder using the provided output buffer.
    #[must_use]
    pub fn new(buf: &'a mut [u8]) -> Self {
        let storage = buf.len() as OpusUint32;
        let mut ctx = EcCtx::new(buf);
        ctx.storage = storage;
        ctx.end_offs = 0;
        ctx.end_window = 0;
        ctx.nend_bits = 0;
        ctx.nbits_total = EC_CODE_BITS as OpusInt32 + 1;
        ctx.offs = 0;
        ctx.rng = EC_CODE_TOP;
        ctx.rem = -1;
        ctx.val = 0;
        ctx.ext = 0;
        ctx.error = 0;
        Self { ctx }
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

    fn write_byte(&mut self, value: OpusUint32) -> OpusInt32 {
        if self.ctx.offs + self.ctx.end_offs >= self.ctx.storage {
            -1
        } else {
            let idx = self.ctx.offs as usize;
            self.ctx.buf[idx] = value as u8;
            self.ctx.offs += 1;
            0
        }
    }

    fn write_byte_at_end(&mut self, value: OpusUint32) -> OpusInt32 {
        if self.ctx.offs + self.ctx.end_offs >= self.ctx.storage {
            -1
        } else {
            self.ctx.end_offs += 1;
            let idx = (self.ctx.storage - self.ctx.end_offs) as usize;
            self.ctx.buf[idx] = value as u8;
            0
        }
    }

    fn carry_out(&mut self, c: OpusInt32) {
        if c != EC_SYM_MAX as OpusInt32 {
            let carry = c >> EC_SYM_BITS;
            if self.ctx.rem >= 0 {
                let value = (self.ctx.rem + carry) as OpusUint32;
                self.ctx.error |= self.write_byte(value);
            }
            if self.ctx.ext > 0 {
                let sym = (EC_SYM_MAX + carry as OpusUint32) & EC_SYM_MAX;
                while self.ctx.ext > 0 {
                    self.ctx.error |= self.write_byte(sym);
                    self.ctx.ext -= 1;
                }
            }
            self.ctx.rem = c & EC_SYM_MAX as OpusInt32;
        } else {
            self.ctx.ext = self.ctx.ext.wrapping_add(1);
        }
    }

    fn normalize(&mut self) {
        while self.ctx.rng <= EC_CODE_BOT {
            self.carry_out((self.ctx.val >> EC_CODE_SHIFT) as OpusInt32);
            self.ctx.val = (self.ctx.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.ctx.rng <<= EC_SYM_BITS;
            self.ctx.nbits_total += EC_SYM_BITS as OpusInt32;
        }
    }

    /// Encodes a symbol using cumulative frequencies.
    pub fn encode(&mut self, fl: OpusUint32, fh: OpusUint32, ft: OpusUint32) {
        let r = celt_udiv(self.ctx.rng, ft);
        if fl > 0 {
            let diff = ft - fl;
            self.ctx.val = self
                .ctx
                .val
                .wrapping_add(self.ctx.rng.wrapping_sub(r.wrapping_mul(diff)));
            self.ctx.rng = r.wrapping_mul(fh - fl);
        } else {
            self.ctx.rng = self.ctx.rng.wrapping_sub(r.wrapping_mul(ft - fh));
        }
        self.normalize();
    }

    /// Encodes a binary symbol with `_bits` bits of precision.
    pub fn encode_bin(&mut self, fl: OpusUint32, fh: OpusUint32, bits: u32) {
        let r = self.ctx.rng >> bits;
        let total = 1u32 << bits;
        if fl > 0 {
            self.ctx.val = self
                .ctx
                .val
                .wrapping_add(self.ctx.rng.wrapping_sub(r.wrapping_mul(total - fl)));
            self.ctx.rng = r.wrapping_mul(fh - fl);
        } else {
            self.ctx.rng = self.ctx.rng.wrapping_sub(r.wrapping_mul(total - fh));
        }
        self.normalize();
    }

    /// Encodes a bit with probability `1/(1<<logp)` of being one.
    pub fn enc_bit_logp(&mut self, val: OpusInt32, logp: u32) {
        let r = self.ctx.rng;
        let l = self.ctx.val;
        let s = r >> logp;
        let r_minus_s = r - s;
        if val != 0 {
            self.ctx.val = l.wrapping_add(r_minus_s);
            self.ctx.rng = s;
        } else {
            self.ctx.rng = r_minus_s;
        }
        self.normalize();
    }

    /// Encodes a symbol using an inverse CDF table with 8-bit entries.
    pub fn enc_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.ctx.rng >> ftb;
        if s > 0 {
            let high = icdf[s - 1] as OpusUint32;
            self.ctx.val = self
                .ctx
                .val
                .wrapping_add(self.ctx.rng.wrapping_sub(r.wrapping_mul(high)));
            self.ctx.rng = r.wrapping_mul(high - icdf[s] as OpusUint32);
        } else {
            self.ctx.rng = self
                .ctx
                .rng
                .wrapping_sub(r.wrapping_mul(icdf[s] as OpusUint32));
        }
        self.normalize();
    }

    /// Encodes a symbol using an inverse CDF table with 16-bit entries.
    pub fn enc_icdf16(&mut self, s: usize, icdf: &[u16], ftb: u32) {
        let r = self.ctx.rng >> ftb;
        if s > 0 {
            let high = icdf[s - 1] as OpusUint32;
            self.ctx.val = self
                .ctx
                .val
                .wrapping_add(self.ctx.rng.wrapping_sub(r.wrapping_mul(high)));
            self.ctx.rng = r.wrapping_mul(high - icdf[s] as OpusUint32);
        } else {
            self.ctx.rng = self
                .ctx
                .rng
                .wrapping_sub(r.wrapping_mul(icdf[s] as OpusUint32));
        }
        self.normalize();
    }

    /// Encodes an unsigned integer in `[0, ft)`.
    pub fn enc_uint(&mut self, fl: OpusUint32, ft: OpusUint32) {
        assert!(ft > 1);
        let ft = ft - 1;
        let mut ftb = (32 - ft.leading_zeros()) as OpusInt32;
        if ftb as usize > EC_UINT_BITS {
            ftb -= EC_UINT_BITS as OpusInt32;
            let ft_small = (ft >> ftb) + 1;
            let fl_small = fl >> ftb;
            self.encode(fl_small, fl_small + 1, ft_small);
            let mask = (1u32 << ftb) - 1;
            self.enc_bits(fl & mask, ftb as u32);
        } else {
            self.encode(fl, fl + 1, ft + 1);
        }
    }

    /// Appends raw bits to the tail of the stream.
    pub fn enc_bits(&mut self, fl: OpusUint32, bits: u32) {
        debug_assert!(bits > 0);
        let mut window = self.ctx.end_window;
        let mut used = self.ctx.nend_bits;
        if used as u32 + bits > EC_WINDOW_SIZE as u32 {
            while used >= EC_SYM_BITS as OpusInt32 {
                self.ctx.error |=
                    self.write_byte_at_end((window & EC_SYM_MAX as EcWindow) as OpusUint32);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS as OpusInt32;
            }
        }
        window |= (fl as EcWindow) << (used as u32);
        used += bits as OpusInt32;
        self.ctx.end_window = window;
        self.ctx.nend_bits = used;
        self.ctx.nbits_total += bits as OpusInt32;
    }

    /// Patches bits at the beginning of the stream once encoding has started.
    pub fn enc_patch_initial_bits(&mut self, val: OpusUint32, nbits: u32) {
        assert!(nbits <= EC_SYM_BITS);
        let shift = EC_SYM_BITS - nbits;
        let mask = ((1u32 << nbits) - 1) << shift;
        let val_masked = (val & ((1u32 << nbits) - 1)) << shift;
        if self.ctx.offs > 0 {
            let byte = self.ctx.buf[0] as OpusUint32;
            self.ctx.buf[0] = ((byte & !mask) | val_masked) as u8;
        } else if self.ctx.rem >= 0 {
            let rem = self.ctx.rem as OpusUint32;
            self.ctx.rem = ((rem & !mask) | val_masked) as OpusInt32;
        } else if self.ctx.rng <= (EC_CODE_TOP >> nbits) {
            let mask_shifted = (mask as OpusUint32) << EC_CODE_SHIFT;
            self.ctx.val =
                (self.ctx.val & !mask_shifted) | ((val_masked as OpusUint32) << EC_CODE_SHIFT);
        } else {
            self.ctx.error = -1;
        }
    }

    /// Shrinks the backing buffer to the requested size.
    pub fn enc_shrink(&mut self, size: OpusUint32) {
        assert!(self.ctx.offs + self.ctx.end_offs <= size);
        if size < self.ctx.storage {
            let len = self.ctx.end_offs as usize;
            if len > 0 {
                let src_start = (self.ctx.storage - self.ctx.end_offs) as usize;
                let dst_start = (size - self.ctx.end_offs) as usize;
                self.ctx
                    .buf
                    .copy_within(src_start..src_start + len, dst_start);
            }
            self.ctx.storage = size;
        }
    }

    /// Finalises the encoding process and flushes buffered data.
    pub fn enc_done(&mut self) {
        let mut window = self.ctx.end_window;
        let mut used = self.ctx.nend_bits;
        let mut l: OpusInt32 = EC_CODE_BITS as OpusInt32 - ec_ilog(self.ctx.rng);
        let mut msk = (EC_CODE_TOP - 1) >> l;
        let mut end = (self.ctx.val + msk) & !msk;
        if (end | msk) >= self.ctx.val.wrapping_add(self.ctx.rng) {
            l += 1;
            msk >>= 1;
            end = (self.ctx.val + msk) & !msk;
        }
        while l > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as OpusInt32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS as OpusInt32;
        }
        if self.ctx.rem >= 0 || self.ctx.ext > 0 {
            self.carry_out(0);
        }
        while used >= EC_SYM_BITS as OpusInt32 {
            self.ctx.error |=
                self.write_byte_at_end((window & EC_SYM_MAX as EcWindow) as OpusUint32);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS as OpusInt32;
        }
        if self.ctx.error == 0 {
            let start = self.ctx.offs as usize;
            let end_idx = (self.ctx.storage - self.ctx.end_offs) as usize;
            for slot in &mut self.ctx.buf[start..end_idx] {
                *slot = 0;
            }
            if used > 0 {
                if self.ctx.end_offs >= self.ctx.storage {
                    self.ctx.error = -1;
                } else {
                    let l_remaining = -l;
                    if self.ctx.offs + self.ctx.end_offs >= self.ctx.storage && l_remaining < used {
                        if l_remaining > 0 {
                            window &= ((1u32 << l_remaining as u32) - 1) as EcWindow;
                        } else {
                            window = 0;
                        }
                        self.ctx.error = -1;
                    }
                    let idx = (self.ctx.storage - self.ctx.end_offs - 1) as usize;
                    self.ctx.buf[idx] |= window as u8;
                }
            }
        }
        self.ctx.end_window = window;
        self.ctx.nend_bits = used;
    }

    /// Returns the number of bytes written to the range portion of the stream.
    #[must_use]
    pub fn range_bytes(&self) -> OpusUint32 {
        self.ctx.range_bytes()
    }
}

/// Snapshot of the encoder state used to perform RDO experiments.
#[derive(Clone)]
pub struct EcEncSnapshot {
    storage: OpusUint32,
    end_offs: OpusUint32,
    end_window: EcWindow,
    nend_bits: OpusInt32,
    nbits_total: OpusInt32,
    offs: OpusUint32,
    rng: OpusUint32,
    val: OpusUint32,
    ext: OpusUint32,
    rem: OpusInt32,
    error: OpusInt32,
    buffer: Vec<u8>,
}

impl EcEncSnapshot {
    /// Captures the current encoder state, including the output buffer.
    #[must_use]
    pub fn capture(enc: &EcEnc<'_>) -> Self {
        let ctx = enc.ctx();
        Self {
            storage: ctx.storage,
            end_offs: ctx.end_offs,
            end_window: ctx.end_window,
            nend_bits: ctx.nend_bits,
            nbits_total: ctx.nbits_total,
            offs: ctx.offs,
            rng: ctx.rng,
            val: ctx.val,
            ext: ctx.ext,
            rem: ctx.rem,
            error: ctx.error,
            buffer: ctx.buffer().to_vec(),
        }
    }

    /// Restores a previously captured encoder state.
    pub fn restore(&self, enc: &mut EcEnc<'_>) {
        let ctx = enc.ctx_mut();
        assert_eq!(self.buffer.len(), ctx.buffer().len());
        ctx.storage = self.storage;
        ctx.end_offs = self.end_offs;
        ctx.end_window = self.end_window;
        ctx.nend_bits = self.nend_bits;
        ctx.nbits_total = self.nbits_total;
        ctx.offs = self.offs;
        ctx.rng = self.rng;
        ctx.val = self.val;
        ctx.ext = self.ext;
        ctx.rem = self.rem;
        ctx.error = self.error;
        ctx.buffer_mut().copy_from_slice(&self.buffer);
    }
}

impl<'a> core::ops::Deref for EcEnc<'a> {
    type Target = EcCtx<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl<'a> core::ops::DerefMut for EcEnc<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::EcEnc;
    use crate::celt::entcode::{EC_CODE_BITS, EC_CODE_TOP, EC_WINDOW_SIZE};

    #[test]
    fn encoder_initialises_like_reference() {
        let mut buf = vec![0u8; 8];
        let enc = EcEnc::new(&mut buf);
        assert_eq!(enc.storage, 8);
        assert_eq!(enc.end_offs, 0);
        assert_eq!(enc.end_window, 0);
        assert_eq!(enc.nend_bits, 0);
        assert_eq!(enc.nbits_total, EC_CODE_BITS as i32 + 1);
        assert_eq!(enc.offs, 0);
        assert_eq!(enc.rng, EC_CODE_TOP);
        assert_eq!(enc.rem, -1);
        assert_eq!(enc.val, 0);
        assert_eq!(enc.ext, 0);
        assert_eq!(enc.error, 0);
    }

    #[test]
    fn enc_bits_appends_to_tail() {
        let mut buf = vec![0u8; 4];
        let mut enc = EcEnc::new(&mut buf);
        enc.enc_bits(0b1010, 4);
        assert_eq!(enc.end_window, 0b1010);
        assert_eq!(enc.nend_bits, 4);
        assert_eq!(enc.nbits_total, EC_CODE_BITS as i32 + 1 + 4);
    }

    #[test]
    fn patch_initial_bits_updates_finalised_byte() {
        let mut buf = vec![0u8; 2];
        let mut enc = EcEnc::new(&mut buf);
        enc.offs = 1;
        enc.buf[0] = 0b1111_0000;
        enc.enc_patch_initial_bits(0b1010, 4);
        assert_eq!(enc.buf[0], 0b1010_0000);
    }

    #[test]
    fn enc_shrink_moves_tail_bits() {
        let mut buf = vec![0u8; 4];
        let mut enc = EcEnc::new(&mut buf);
        enc.end_offs = 1;
        enc.storage = 4;
        enc.buf[3] = 0xAA;
        enc.enc_shrink(3);
        assert_eq!(enc.storage, 3);
        assert_eq!(enc.buf[2], 0xAA);
    }

    #[test]
    fn enc_done_flushes_raw_bits() {
        let mut buf = vec![0u8; 4];
        let mut enc = EcEnc::new(&mut buf);
        enc.enc_bits(0b1011, 4);
        enc.enc_done();
        assert_eq!(enc.buf[3], 0b1011);
        assert_eq!(enc.error, 0);
        assert!(enc.nend_bits < EC_WINDOW_SIZE as i32);
    }
}
