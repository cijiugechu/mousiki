#![allow(dead_code)]

//! Laplace probability model helpers from `celt/laplace.c`.
//!
//! The Laplace coder is used by CELT to model the distribution of band energy
//! deltas.  It sits on top of the range coder and has limited dependencies,
//! making it a good candidate for early porting efforts.

use core::cmp::{max, min};

use crate::range::{RangeDecoder, RangeEncoder};

const LAPLACE_LOG_MINP: u32 = 0;
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;
const LAPLACE_NMIN: u32 = 16;
const TOTAL_FREQ: u32 = 1 << 15;

fn laplace_get_freq1(fs0: u32, decay: u32) -> u32 {
    let ft = TOTAL_FREQ - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    if decay >= 16384 {
        0
    } else {
        let factor = 16384u32 - decay;
        ((u64::from(ft) * u64::from(factor)) >> 15) as u32
    }
}

fn apply_sign(value: i32, sign: i32) -> i32 {
    (value + sign) ^ sign
}

pub(crate) fn ec_laplace_encode(enc: &mut RangeEncoder, value: &mut i32, mut fs: u32, decay: u32) {
    let mut fl = 0u32;
    let mut val = *value;

    if val != 0 {
        let sign = if val < 0 { -1 } else { 0 };
        val = apply_sign(val, sign);
        let mut i = 1i32;
        fl = fs;
        fs = laplace_get_freq1(fs, decay);

        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            fs = ((u64::from(fs) * u64::from(decay)) >> 15) as u32;
            i += 1;
        }

        if fs == 0 {
            let mut ndi_max = ((TOTAL_FREQ - fl + LAPLACE_MINP - 1) >> LAPLACE_LOG_MINP) as i32;
            ndi_max = (ndi_max - sign) >> 1;
            let di = min(val - i, ndi_max - 1);
            fl += ((2 * di + 1 + sign) as u32) * LAPLACE_MINP;
            fs = min(LAPLACE_MINP, TOTAL_FREQ - fl);
            *value = apply_sign(i + di, sign);
        } else {
            fs += LAPLACE_MINP;
            if sign == 0 {
                fl += fs;
            }
        }

        debug_assert!(fl + fs <= TOTAL_FREQ);
        debug_assert!(fs > 0);
    }

    let high = (fl + fs).min(TOTAL_FREQ);
    enc.encode_bin(fl, high, 15);
}

pub(crate) fn ec_laplace_decode(dec: &mut RangeDecoder, mut fs: u32, decay: u32) -> i32 {
    let mut val = 0i32;
    let mut fl = 0u32;
    let fm = dec.decode_bin(15);

    if fm >= fs {
        val += 1;
        fl = fs;
        fs = laplace_get_freq1(fs, decay) + LAPLACE_MINP;

        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = ((u64::from(fs - 2 * LAPLACE_MINP) * u64::from(decay)) >> 15) as u32;
            fs += LAPLACE_MINP;
            val += 1;
        }

        if fs <= LAPLACE_MINP {
            let di = ((fm - fl) >> (LAPLACE_LOG_MINP + 1)) as i32;
            val += di;
            fl += 2 * di as u32 * LAPLACE_MINP;
        }

        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }

    let scale = dec.range_size >> 15;
    let high = (fl + fs).min(TOTAL_FREQ);
    dec.update(scale, fl, high, TOTAL_FREQ);

    val
}

pub(crate) fn ec_laplace_encode_p0(enc: &mut RangeEncoder, value: i32, p0: u16, decay: u16) {
    let mut sign_icdf = [0u16; 3];
    sign_icdf[0] = 32768 - p0;
    sign_icdf[1] = sign_icdf[0] / 2;
    sign_icdf[2] = 0;

    let sign_symbol = match value.cmp(&0) {
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
        core::cmp::Ordering::Less => 2,
    };
    enc.encode_icdf16(sign_symbol as usize, &sign_icdf, 15);

    let magnitude = value.abs();
    if magnitude != 0 {
        let mut icdf = [0u16; 8];
        icdf[0] = max(7u32, u32::from(decay)) as u16;
        for i in 1..7 {
            let baseline = max(0, 7 - i as i32) as u32;
            let decayed = (u32::from(icdf[i - 1]) * u32::from(decay)) >> 15;
            icdf[i] = max(baseline, decayed) as u16;
        }
        icdf[7] = 0;

        let mut remaining = magnitude - 1;
        loop {
            let symbol = remaining.min(7) as usize;
            enc.encode_icdf16(symbol, &icdf, 15);
            remaining -= 7;
            if remaining < 0 {
                break;
            }
        }
    }
}

pub(crate) fn ec_laplace_decode_p0(dec: &mut RangeDecoder, p0: u16, decay: u16) -> i32 {
    let mut sign_icdf = [0u16; 3];
    sign_icdf[0] = 32768 - p0;
    sign_icdf[1] = sign_icdf[0] / 2;
    sign_icdf[2] = 0;

    let mut sign = dec.decode_icdf16(&sign_icdf, 15) as i32;
    if sign == 2 {
        sign = -1;
    }

    if sign != 0 {
        let mut icdf = [0u16; 8];
        icdf[0] = max(7u32, u32::from(decay)) as u16;
        for i in 1..7 {
            let baseline = max(0, 7 - i as i32) as u32;
            let decayed = (u32::from(icdf[i - 1]) * u32::from(decay)) >> 15;
            icdf[i] = max(baseline, decayed) as u16;
        }
        icdf[7] = 0;

        let mut value = 1;
        loop {
            let v = dec.decode_icdf16(&icdf, 15) as i32;
            value += v;
            if v != 7 {
                return sign * value;
            }
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed)
        }

        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            self.0 >> 16
        }
    }

    fn laplace_start_freq(decay: u32) -> u32 {
        let ft = TOTAL_FREQ - LAPLACE_MINP * (2 * LAPLACE_NMIN + 1);
        let numerator = u64::from(ft) * u64::from(16384 - decay);
        let denominator = u64::from(16384 + decay);
        (numerator / denominator) as u32 + LAPLACE_MINP
    }

    #[test]
    fn laplace_encode_decode_roundtrip() {
        let mut encoder = RangeEncoder::new();
        let decay = 12000;
        let fs = 5000;
        let inputs = [-4, -1, 0, 1, 3, 5];
        let mut encoded = inputs;

        for value in &mut encoded {
            ec_laplace_encode(&mut encoder, value, fs, decay);
        }

        let data = encoder.finish();
        let mut decoder = RangeDecoder::init(&data);

        for expected in &encoded {
            let decoded = ec_laplace_decode(&mut decoder, fs, decay);
            assert_eq!(decoded, *expected);
        }
    }

    #[test]
    fn laplace_encode_decode_p0_roundtrip() {
        let mut encoder = RangeEncoder::new();
        let values = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let p0 = 16000u16;
        let decay = 16000u16;

        for &value in &values {
            ec_laplace_encode_p0(&mut encoder, value, p0, decay);
        }

        let data = encoder.finish();
        let mut decoder = RangeDecoder::init(&data);

        for &expected in &values {
            let decoded = ec_laplace_decode_p0(&mut decoder, p0, decay);
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn laplace_reference_harness_roundtrip() {
        const SAMPLES: usize = 10_000;
        let mut values = [0i32; SAMPLES];
        let mut decays = [0u32; SAMPLES];

        values[0] = 3;
        values[1] = 0;
        values[2] = -1;
        decays[0] = 6000;
        decays[1] = 5800;
        decays[2] = 5600;

        let mut rng = Lcg::new(1);
        for i in 3..SAMPLES {
            values[i] = (rng.next() % 15) as i32 - 7;
            decays[i] = (rng.next() % 11_000) + 5_000;
        }

        let mut encoder = RangeEncoder::new();
        for i in 0..SAMPLES {
            let fs = laplace_start_freq(decays[i]);
            ec_laplace_encode(&mut encoder, &mut values[i], fs, decays[i]);
        }

        let data = encoder.finish();
        let mut decoder = RangeDecoder::init(&data);

        for i in 0..SAMPLES {
            let fs = laplace_start_freq(decays[i]);
            let decoded = ec_laplace_decode(&mut decoder, fs, decays[i]);
            assert_eq!(decoded, values[i], "mismatch at sample {}", i);
        }
    }
}
