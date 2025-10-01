#![allow(dead_code)]

use crate::celt::types::{OpusInt32, OpusUint32};

/// Window type used by the entropy coder.
pub type EcWindow = OpusUint32;

/// Number of bits in the range coder's window.
pub const EC_WINDOW_SIZE: usize = core::mem::size_of::<EcWindow>() * 8;

/// Number of bits output at a time by the range coder.
pub const EC_SYM_BITS: u32 = 8;

/// Total number of bits in the coder's internal state registers.
pub const EC_CODE_BITS: u32 = 32;

/// Maximum value that a coded symbol can take.
pub const EC_SYM_MAX: OpusUint32 = (1u32 << EC_SYM_BITS) - 1;

/// Carry bit associated with the high-order range symbol.
pub const EC_CODE_TOP: OpusUint32 = 1u32 << (EC_CODE_BITS - 1);

/// Low-order bit of the high-order range symbol.
pub const EC_CODE_BOT: OpusUint32 = EC_CODE_TOP >> EC_SYM_BITS;

/// Number of extra bits stored in the range coder state.
pub const EC_CODE_EXTRA: u32 = ((EC_CODE_BITS - 2) % EC_SYM_BITS) + 1;

/// Number of bits consumed when encoding unsigned integers.
pub const EC_UINT_BITS: usize = 8;

/// Resolution of fractional bit counts reported by [`ec_tell_frac`].
pub const BITRES: u32 = 3;

/// Entropy coder context used by both the encoder and the decoder.
#[derive(Debug)]
pub struct EcCtx<'a> {
    pub buf: &'a mut [u8],
    pub storage: OpusUint32,
    pub end_offs: OpusUint32,
    pub end_window: EcWindow,
    pub nend_bits: OpusInt32,
    pub nbits_total: OpusInt32,
    pub offs: OpusUint32,
    pub rng: OpusUint32,
    pub val: OpusUint32,
    pub ext: OpusUint32,
    pub rem: OpusInt32,
    pub error: OpusInt32,
}

impl<'a> EcCtx<'a> {
    /// Constructs a new entropy coder context backed by the provided buffer.
    #[must_use]
    pub fn new(buf: &'a mut [u8]) -> Self {
        let storage = buf.len() as OpusUint32;
        Self {
            buf,
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: 0,
            offs: 0,
            rng: 1,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        }
    }

    /// Returns the number of bytes currently stored in the range coder.
    #[must_use]
    pub fn range_bytes(&self) -> OpusUint32 {
        self.offs
    }

    /// Returns a shared view of the underlying I/O buffer.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        &*self.buf
    }

    /// Returns a mutable view of the underlying I/O buffer.
    #[must_use]
    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut *self.buf
    }

    /// Returns the current error flag for the coder.
    #[must_use]
    pub fn error(&self) -> OpusInt32 {
        self.error
    }
}

/// Counts the integer binary logarithm of the supplied value.
#[must_use]
pub fn ec_ilog(mut v: OpusUint32) -> OpusInt32 {
    let mut ret = OpusInt32::from(v != 0);
    let mut m = OpusInt32::from((v & 0xFFFF_0000) != 0) << 4;
    v >>= m;
    ret |= m;

    m = OpusInt32::from((v & 0xFF00) != 0) << 3;
    v >>= m;
    ret |= m;

    m = OpusInt32::from((v & 0xF0) != 0) << 2;
    v >>= m;
    ret |= m;

    m = OpusInt32::from((v & 0xC) != 0) << 1;
    v >>= m;
    ret |= m;

    ret + OpusInt32::from((v & 0x2) != 0)
}

/// Returns the number of whole bits consumed by the coder so far.
#[must_use]
pub fn ec_tell(ctx: &EcCtx<'_>) -> OpusInt32 {
    ctx.nbits_total - ec_ilog(ctx.rng)
}

/// Returns the number of bits consumed by the coder at a resolution of 1/8th bits.
#[must_use]
pub fn ec_tell_frac(ctx: &EcCtx<'_>) -> OpusUint32 {
    const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];

    let nbits = (ctx.nbits_total as OpusUint32) << BITRES;
    let mut l = ec_ilog(ctx.rng);
    debug_assert!(l >= 16);
    let r = ctx.rng >> ((l - 16) as u32);
    let b = (r >> 12) as i32 - 8;
    debug_assert!((0..CORRECTION.len() as i32).contains(&b));
    let mut b = b as usize;
    if r > CORRECTION[b] {
        b += 1;
    }
    l = (l << 3) + b as OpusInt32;
    nbits.wrapping_sub(l as OpusUint32)
}

/// Lookup table used by the small divisor optimisation in [`celt_udiv`].
#[allow(clippy::unreadable_literal)]
pub const SMALL_DIV_TABLE: [OpusUint32; 128] = [
    0xFFFF_FFFF,
    0x5555_5555,
    0x3333_3333,
    0x2492_4924,
    0x1C71_C71C,
    0x1745_D174,
    0x13B1_3B13,
    0x1111_1111,
    0x0F0F_0F0F,
    0x0D79_435E,
    0x0C30_C30C,
    0x0B21_642C,
    0x0A3D_70A3,
    0x097B_425E,
    0x08D3_DCB0,
    0x0842_1084,
    0x07C1_F07C,
    0x0750_7507,
    0x06EB_3E45,
    0x0690_6906,
    0x063E_7063,
    0x05F4_17D0,
    0x05B0_5B05,
    0x0572_620A,
    0x0539_7829,
    0x0505_0505,
    0x04D4_873E,
    0x04A7_904A,
    0x047D_C11F,
    0x0456_C797,
    0x0432_5C53,
    0x0410_4104,
    0x03F0_3F03,
    0x03D2_2635,
    0x03B5_CC0E,
    0x039B_0AD1,
    0x0381_C0E0,
    0x0369_D036,
    0x0353_1DEC,
    0x033D_91D2,
    0x0329_161F,
    0x0315_9721,
    0x0303_0303,
    0x02F1_4990,
    0x02E0_5C0B,
    0x02D0_2D02,
    0x02C0_B02C,
    0x02B1_DA46,
    0x02A3_A0FD,
    0x0295_FAD4,
    0x0288_DF0C,
    0x027C_4597,
    0x0270_2702,
    0x0264_7C69,
    0x0259_3F69,
    0x024E_6A17,
    0x0243_F6F0,
    0x0239_E0D5,
    0x0230_2302,
    0x0226_B902,
    0x021D_9EAD,
    0x0214_D021,
    0x020C_49BA,
    0x0204_0810,
    0x01FC_07F0,
    0x01F4_4659,
    0x01EC_C07B,
    0x01E5_73AC,
    0x01DE_5D6E,
    0x01D7_7B65,
    0x01D0_CB58,
    0x01CA_4B30,
    0x01C3_F8F0,
    0x01BD_D2B8,
    0x01B7_D6C3,
    0x01B2_0364,
    0x01AC_5701,
    0x01A6_D01A,
    0x01A1_6D3F,
    0x019C_2D14,
    0x0197_0E4F,
    0x0192_0FB4,
    0x018D_3018,
    0x0188_6E5F,
    0x0183_C977,
    0x017F_405F,
    0x017A_D220,
    0x0176_7DCE,
    0x0172_4287,
    0x016E_1F76,
    0x016A_13CD,
    0x0166_1EC6,
    0x0162_3FA7,
    0x015E_75BB,
    0x015A_C056,
    0x0157_1ED3,
    0x0153_9094,
    0x0150_1501,
    0x014C_AB88,
    0x0149_539E,
    0x0146_0CBC,
    0x0142_D662,
    0x013F_B013,
    0x013C_995A,
    0x0139_91C2,
    0x0136_98DF,
    0x0133_AE45,
    0x0130_D190,
    0x012E_025C,
    0x012B_404A,
    0x0128_8B01,
    0x0125_E227,
    0x0123_4567,
    0x0120_B470,
    0x011E_2EF3,
    0x011B_B4A4,
    0x0119_4538,
    0x0116_E068,
    0x0114_85F0,
    0x0112_358E,
    0x010F_EF01,
    0x010D_B20A,
    0x010B_7E6E,
    0x0109_53F3,
    0x0107_3260,
    0x0105_197F,
    0x0103_091B,
    0x0101_0101,
];

/// Unsigned division helper mirroring the CELT implementation.
#[must_use]
pub fn celt_udiv(n: OpusUint32, d: OpusUint32) -> OpusUint32 {
    debug_assert!(d > 0);
    if d > 256 {
        n / d
    } else {
        let t = ec_ilog(d & d.wrapping_neg()) as u32;
        let shift = t.saturating_sub(1);
        let q = ((SMALL_DIV_TABLE[(d >> t) as usize] as u64) * ((n >> shift) as u64)) >> 32;
        let q = q as OpusUint32;
        q + OpusUint32::from(n - q * d >= d)
    }
}

/// Signed division helper mirroring the CELT implementation.
#[must_use]
pub fn celt_sudiv(n: OpusInt32, d: OpusInt32) -> OpusInt32 {
    debug_assert!(d > 0);
    if n < 0 {
        -(celt_udiv(n.unsigned_abs(), d as OpusUint32) as OpusInt32)
    } else {
        celt_udiv(n as OpusUint32, d as OpusUint32) as OpusInt32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ec_ilog_matches_reference_values() {
        let expected = [
            (0u32, 0),
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 3),
            (7, 3),
            (8, 4),
            (15, 4),
            (16, 5),
            (31, 5),
            (32, 6),
            (255, 8),
            (256, 9),
            (1023, 10),
            (1024, 11),
        ];
        for (input, output) in expected {
            assert_eq!(ec_ilog(input), output, "ec_ilog({input})");
        }
    }

    fn slow_tell_frac(ctx: &EcCtx<'_>) -> OpusUint32 {
        let mut l = ec_ilog(ctx.rng);
        debug_assert!(l >= 16);
        let mut r = ctx.rng >> ((l - 16) as u32);
        for _ in 0..BITRES as usize {
            r = (r * r) >> 15;
            let b = (r >> 16) as i32;
            l = (l << 1) | b;
            r >>= b;
        }
        ((ctx.nbits_total as OpusUint32) << BITRES).wrapping_sub(l as OpusUint32)
    }

    #[test]
    fn fast_tell_frac_matches_reference() {
        let mut scratch = [0u8; 1];
        let mut ctx = EcCtx::new(&mut scratch);
        let samples = [
            (0x8000u32, 0),
            (0xFFFFu32, 5),
            (0x10000u32, 17),
            (0x23456u32, 42),
            (0x7FFF_FFFFu32, 64),
        ];
        for (rng, nbits_total) in samples {
            ctx.rng = rng;
            ctx.nbits_total = nbits_total;
            let fast = ec_tell_frac(&ctx);
            let slow = slow_tell_frac(&ctx);
            assert_eq!(fast, slow, "rng={rng:#x}, nbits_total={nbits_total}");
        }
    }

    #[test]
    fn small_division_matches_builtin() {
        for d in 1..=256u32 {
            for &n in &[0u32, 1, 7, 255, 256, 511, 1024, u32::MAX] {
                assert_eq!(celt_udiv(n, d), n / d, "n={n}, d={d}");
            }
        }
    }

    #[test]
    fn signed_division_matches_builtin() {
        let denominators = [1, 2, 3, 7, 19, 127, 255];
        let numerators = [-65_535, -1234, -1, 0, 1, 1234, 65_535, OpusInt32::MAX];
        for &d in &denominators {
            for &n in &numerators {
                assert_eq!(celt_sudiv(n, d), n / d, "n={n}, d={d}");
            }
        }
    }
}
