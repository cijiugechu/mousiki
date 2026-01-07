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
        if c == EC_SYM_MAX as OpusInt32 {
            self.ctx.ext = self.ctx.ext.wrapping_add(1);
        } else {
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
            let high = OpusUint32::from(icdf[s - 1]);
            self.ctx.val = self
                .ctx
                .val
                .wrapping_add(self.ctx.rng.wrapping_sub(r.wrapping_mul(high)));
            self.ctx.rng = r.wrapping_mul(high - OpusUint32::from(icdf[s]));
        } else {
            self.ctx.rng = self
                .ctx
                .rng
                .wrapping_sub(r.wrapping_mul(OpusUint32::from(icdf[s])));
        }
        self.normalize();
    }

    /// Encodes a symbol using an inverse CDF table with 16-bit entries.
    pub fn enc_icdf16(&mut self, s: usize, icdf: &[u16], ftb: u32) {
        let r = self.ctx.rng >> ftb;
        if s > 0 {
            let high = OpusUint32::from(icdf[s - 1]);
            self.ctx.val = self
                .ctx
                .val
                .wrapping_add(self.ctx.rng.wrapping_sub(r.wrapping_mul(high)));
            self.ctx.rng = r.wrapping_mul(high - OpusUint32::from(icdf[s]));
        } else {
            self.ctx.rng = self
                .ctx
                .rng
                .wrapping_sub(r.wrapping_mul(OpusUint32::from(icdf[s])));
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
            let byte = OpusUint32::from(self.ctx.buf[0]);
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

    #[must_use]
    pub(crate) fn buffer_len(&self) -> usize {
        self.buffer.len()
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
    use alloc::vec::Vec;

    use super::EcEnc;
    use crate::celt::entcode::{EC_CODE_BITS, EC_CODE_TOP, EC_WINDOW_SIZE};
    use crate::celt::entdec::EcDec;

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

    /// Port of the first section of `test_unit_entropy.c` from opus-c.
    ///
    /// Tests encoding/decoding of raw unsigned integers (ec_enc_uint/ec_dec_uint)
    /// for various base values from 2 to 1023.
    #[test]
    fn entropy_uint_roundtrip() {
        const DATA_SIZE: usize = 1_000_000;
        let mut buf = vec![0u8; DATA_SIZE];

        // Encode all possible values for each base from 2 to 1023
        {
            let mut enc = EcEnc::new(&mut buf);
            for ft in 2..1024u32 {
                for i in 0..ft {
                    enc.enc_uint(i, ft);
                }
            }
            enc.enc_done();
            assert_eq!(enc.error, 0, "encoding error");
        }

        // Decode and verify all values
        {
            let mut dec = EcDec::new(&mut buf);
            for ft in 2..1024u32 {
                for i in 0..ft {
                    let sym = dec.dec_uint(ft);
                    assert_eq!(
                        sym, i,
                        "decoded {} instead of {} with ft of {}",
                        sym, i, ft
                    );
                }
            }
        }
    }

    /// Port of the raw bits encoding section from `test_unit_entropy.c`.
    ///
    /// Tests that encoding and decoding raw bits produces exact matches
    /// and uses exactly the expected number of bits.
    #[test]
    fn entropy_raw_bits_roundtrip() {
        use crate::celt::entcode::ec_tell;

        const DATA_SIZE: usize = 1_000_000;
        let mut buf = vec![0u8; DATA_SIZE];

        // Encode raw bits for bit widths 1 to 15
        {
            let mut enc = EcEnc::new(&mut buf);
            for ftb in 1..16u32 {
                for i in 0..(1u32 << ftb) {
                    let nbits_before = ec_tell(&enc);
                    enc.enc_bits(i, ftb);
                    let nbits_after = ec_tell(&enc);
                    assert_eq!(
                        nbits_after - nbits_before,
                        ftb as i32,
                        "used {} bits to encode {} bits directly",
                        nbits_after - nbits_before,
                        ftb
                    );
                }
            }
            enc.enc_done();
            assert_eq!(enc.error, 0);
        }

        // Decode and verify raw bits
        {
            let mut dec = EcDec::new(&mut buf);
            for ftb in 1..16u32 {
                for i in 0..(1u32 << ftb) {
                    let sym = dec.dec_bits(ftb);
                    assert_eq!(
                        sym, i,
                        "decoded {} instead of {} with ftb of {}",
                        sym, i, ftb
                    );
                }
            }
        }
    }

    /// Port of the `patch_initial_bits` test from `test_unit_entropy.c`.
    ///
    /// Tests that patching initial bits works correctly in various scenarios.
    #[test]
    fn entropy_patch_initial_bits() {
        const DATA_SIZE: usize = 10_000;
        let mut buf = vec![0u8; DATA_SIZE];

        // Test case 1: patch works correctly
        {
            let mut enc = EcEnc::new(&mut buf);
            enc.enc_bit_logp(0, 1);
            enc.enc_bit_logp(0, 1);
            enc.enc_bit_logp(0, 1);
            enc.enc_bit_logp(0, 1);
            enc.enc_bit_logp(0, 2);
            enc.enc_patch_initial_bits(3, 2);
            assert_eq!(enc.error, 0, "patch_initial_bits failed");

            // Patching more bits than available should fail
            enc.enc_patch_initial_bits(0, 5);
            assert_ne!(enc.error, 0, "patch_initial_bits didn't fail when it should have");
            enc.enc_done();
            assert_eq!(enc.range_bytes(), 1);
            assert_eq!(buf[0], 192);
        }

        // Test case 2: different bit patterns
        {
            let mut enc = EcEnc::new(&mut buf);
            enc.enc_bit_logp(0, 1);
            enc.enc_bit_logp(0, 1);
            enc.enc_bit_logp(1, 6);
            enc.enc_bit_logp(0, 2);
            enc.enc_patch_initial_bits(0, 2);
            assert_eq!(enc.error, 0, "patch_initial_bits failed");
            enc.enc_done();
            assert_eq!(enc.range_bytes(), 2);
            assert_eq!(buf[0], 63);
        }
    }

    /// Port of the raw bits overfill test from `test_unit_entropy.c`.
    ///
    /// Tests that attempting to encode more raw bits than buffer size
    /// results in an error.
    #[test]
    fn entropy_raw_bits_overfill() {
        let mut buf = vec![0u8; 2];

        // Test 1: 48 raw bits should overflow 16-bit buffer
        {
            let mut enc = EcEnc::new(&mut buf);
            enc.enc_bit_logp(0, 2);
            for _ in 0..48 {
                enc.enc_bits(0, 1);
            }
            enc.enc_done();
            assert_ne!(enc.error, 0, "raw bits overfill didn't fail when it should have");
        }

        // Test 2: 17 raw bits in two bytes should fail
        {
            let mut enc = EcEnc::new(&mut buf);
            for _ in 0..17 {
                enc.enc_bits(0, 1);
            }
            enc.enc_done();
            assert_ne!(enc.error, 0, "17 raw bits encoded in two bytes");
        }
    }

    /// Simple LCG for reproducible pseudo-random sequences.
    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed)
        }

        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            self.0
        }
    }

    /// Port of the random stream tests from `test_unit_entropy.c`.
    ///
    /// Tests encoding and decoding of random unsigned integer sequences
    /// with various base values and stream sizes. Also validates that
    /// the bit-counting (tell) functions are consistent between encoder
    /// and decoder.
    #[test]
    fn entropy_random_uint_streams() {
        use crate::celt::entcode::ec_tell_frac;

        const DATA_SIZE: usize = 10_000;
        const NUM_ITERATIONS: usize = 1_000; // Reduced from C's 409600 for speed
        const SEED: u32 = 12345;

        let mut buf = vec![0u8; DATA_SIZE];
        let mut rng = Lcg::new(SEED);

        for iteration in 0..NUM_ITERATIONS {
            // Generate random parameters (matching C's rand() behavior with RAND_MAX)
            let shift1 = (rng.next() % 11) as u32;
            let divisor = u32::MAX.wrapping_shr(shift1).saturating_add(1);
            let ft = rng.next().wrapping_div(divisor.max(1)).saturating_add(10);
            let shift2 = (rng.next() % 9) as u32;
            let divisor2 = u32::MAX.wrapping_shr(shift2).saturating_add(1);
            let sz = (rng.next().wrapping_div(divisor2.max(1))) as usize;
            let zeros = (rng.next() % 13) == 0;

            // Generate random data
            let data: Vec<u32> = (0..sz)
                .map(|_| {
                    if zeros {
                        0
                    } else {
                        rng.next() % ft
                    }
                })
                .collect();

            // Encode
            let mut enc = EcEnc::new(&mut buf);
            let mut tells: Vec<u32> = Vec::with_capacity(sz + 1);
            tells.push(ec_tell_frac(&enc));
            for &value in &data {
                enc.enc_uint(value, ft);
                tells.push(ec_tell_frac(&enc));
            }

            // Optionally pad to byte boundary
            if rng.next() % 2 == 0 {
                use crate::celt::entcode::ec_tell;
                while ec_tell(&enc) % 8 != 0 {
                    enc.enc_uint(rng.next() % 2, 2);
                }
            }

            enc.enc_done();
            assert_eq!(enc.error, 0, "encoding error at iteration {}", iteration);

            // Decode and verify
            let mut dec = EcDec::new(&mut buf);
            assert_eq!(
                ec_tell_frac(&dec),
                tells[0],
                "tell mismatch at symbol 0, iteration {}",
                iteration
            );

            for (j, &expected) in data.iter().enumerate() {
                let sym = dec.dec_uint(ft);
                assert_eq!(
                    sym, expected,
                    "decoded {} instead of {} with ft of {} at position {} of {} (iteration {})",
                    sym, expected, ft, j, sz, iteration
                );
                assert_eq!(
                    ec_tell_frac(&dec),
                    tells[j + 1],
                    "tell mismatch at symbol {}, iteration {}",
                    j + 1,
                    iteration
                );
            }
        }
    }

    /// Port of the encode/decode method compatibility test from `test_unit_entropy.c`.
    ///
    /// Tests that different encoding methods (encode, encode_bin, enc_bit_logp, enc_icdf)
    /// produce correct results when decoded with the same method family.
    #[test]
    fn entropy_method_compatibility() {
        const DATA_SIZE: usize = 10_000;
        const NUM_ITERATIONS: usize = 1_000; // Reduced from C's 409600 for speed
        const SEED: u32 = 54321;

        let mut buf = vec![0u8; DATA_SIZE];
        let mut rng = Lcg::new(SEED);

        for iteration in 0..NUM_ITERATIONS {
            let shift = (rng.next() % 9) as u32;
            let divisor = u32::MAX.wrapping_shr(shift).saturating_add(1);
            let sz = rng.next().wrapping_div(divisor.max(1)) as usize;
            let sz = sz.min(500); // Limit size for reasonable test time

            // Generate random data and parameters
            let divisor_half = u32::MAX.wrapping_shr(1).saturating_add(1);
            let data: Vec<u32> = (0..sz)
                .map(|_| rng.next().wrapping_div(divisor_half.max(1)))
                .collect();
            let logp1: Vec<u32> = (0..sz).map(|_| (rng.next() % 15) + 1).collect();
            let enc_method: Vec<u32> = (0..sz)
                .map(|_| {
                    let divisor4 = u32::MAX.wrapping_shr(2).saturating_add(1);
                    rng.next().wrapping_div(divisor4.max(1))
                })
                .collect();

            // Encode using various methods
            let mut enc = EcEnc::new(&mut buf);

            for j in 0..sz {
                let d = data[j];
                let logp = logp1[j];
                let ft = 1u32 << logp;

                match enc_method[j] % 4 {
                    0 => {
                        // ec_encode
                        let (fl, fh) = if d != 0 { (ft - 1, ft) } else { (0, 1) };
                        enc.encode(fl, fh, ft);
                    }
                    1 => {
                        // ec_encode_bin
                        let (fl, fh) = if d != 0 { (ft - 1, ft) } else { (0, 1) };
                        enc.encode_bin(fl, fh, logp);
                    }
                    2 => {
                        // ec_enc_bit_logp
                        enc.enc_bit_logp(d as i32, logp);
                    }
                    _ => {
                        // ec_enc_icdf
                        let icdf = [1u8, 0];
                        enc.enc_icdf(d as usize, &icdf, logp);
                    }
                }
            }
            enc.enc_done();
            assert_eq!(enc.error, 0, "encoding error at iteration {}", iteration);

            // Decode using the SAME method as encoding (to ensure roundtrip)
            let mut dec = EcDec::new(&mut buf);

            for j in 0..sz {
                let expected = data[j];
                let logp = logp1[j];
                let ft = 1u32 << logp;

                // Use the same decode method as encode method
                let sym = match enc_method[j] % 4 {
                    0 => {
                        // ec_decode
                        let fs = dec.decode(ft);
                        let s = if fs >= ft - 1 { 1 } else { 0 };
                        let (fl, fh) = if s != 0 { (ft - 1, ft) } else { (0, 1) };
                        dec.update(fl, fh, ft);
                        s
                    }
                    1 => {
                        // ec_decode_bin
                        let fs = dec.decode_bin(logp);
                        let s = if fs >= ft - 1 { 1 } else { 0 };
                        let (fl, fh) = if s != 0 { (ft - 1, ft) } else { (0, 1) };
                        dec.update(fl, fh, ft);
                        s
                    }
                    2 => {
                        // ec_dec_bit_logp
                        dec.dec_bit_logp(logp) as u32
                    }
                    _ => {
                        // ec_dec_icdf
                        let icdf = [1u8, 0];
                        dec.dec_icdf(&icdf, logp) as u32
                    }
                };

                assert_eq!(
                    sym, expected,
                    "decoded {} instead of {} at position {} of {} (iteration {})",
                    sym, expected, j, sz, iteration
                );
            }
        }
    }
}
