use crate::celt::{EcDec, ec_tell};
use crate::range::RangeDecoder;
use crate::silk::icdf::ICDFContext;

pub trait SilkRangeDecoder {
    fn decode_symbol_logp(&mut self, logp: usize) -> u32;
    fn decode_symbol_with_icdf(&mut self, icdf_ctx: ICDFContext) -> u32;
    fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize;
    #[allow(dead_code)]
    fn decode_icdf16(&mut self, icdf: &[u16], ftb: u32) -> usize;
    fn decode_uint(&mut self, total: u32) -> u32;
    fn tell(&self) -> i32;
    fn range_final(&self) -> u32;
}

impl<'a> SilkRangeDecoder for RangeDecoder<'a> {
    fn decode_symbol_logp(&mut self, logp: usize) -> u32 {
        self.decode_symbol_logp(logp)
    }

    fn decode_symbol_with_icdf(&mut self, icdf_ctx: ICDFContext) -> u32 {
        self.decode_symbol_with_icdf(icdf_ctx)
    }

    fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        self.decode_icdf(icdf, ftb)
    }

    fn decode_icdf16(&mut self, icdf: &[u16], ftb: u32) -> usize {
        self.decode_icdf16(icdf, ftb)
    }

    fn decode_uint(&mut self, total: u32) -> u32 {
        self.decode_uint(total)
    }

    fn tell(&self) -> i32 {
        self.tell()
    }

    fn range_final(&self) -> u32 {
        self.range_size
    }
}

impl<'a> SilkRangeDecoder for EcDec<'a> {
    fn decode_symbol_logp(&mut self, logp: usize) -> u32 {
        self.dec_bit_logp(logp as u32) as u32
    }

    fn decode_symbol_with_icdf(&mut self, icdf_ctx: ICDFContext) -> u32 {
        let ICDFContext { total, dist_table } = icdf_ctx;
        debug_assert!(!dist_table.is_empty(), "icdf tables must not be empty");
        let symbol = self.decode(total);

        let mut index = 0usize;
        while index < dist_table.len() && (dist_table[index] as u32) <= symbol {
            index += 1;
        }
        if index == dist_table.len() {
            index = 0;
        }

        let high = dist_table[index] as u32;
        let low = if index > 0 {
            dist_table[index - 1] as u32
        } else {
            0
        };
        self.update(low, high, total);

        index as u32
    }

    fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let value = self.dec_icdf(icdf, ftb);
        debug_assert!(value >= 0);
        value as usize
    }

    fn decode_icdf16(&mut self, icdf: &[u16], ftb: u32) -> usize {
        let value = self.dec_icdf16(icdf, ftb);
        debug_assert!(value >= 0);
        value as usize
    }

    fn decode_uint(&mut self, total: u32) -> u32 {
        self.dec_uint(total)
    }

    fn tell(&self) -> i32 {
        ec_tell(self.ctx())
    }

    fn range_final(&self) -> u32 {
        self.ctx().rng
    }
}
