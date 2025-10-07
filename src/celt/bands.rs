#![allow(dead_code)]

//! Helper routines from `celt/bands.c` that are self-contained enough to port
//! ahead of the rest of the band analysis logic.
//!
//! The goal is to translate building blocks that have little coupling with the
//! more complex pieces of the encoder so that future ports can focus on the
//! higher-level control flow.

use alloc::vec;
use alloc::vec::Vec;

use core::f32::consts::{FRAC_1_SQRT_2, SQRT_2};

use crate::celt::entcode::ec_tell_frac;
use crate::celt::{
    BITRES, EcDec, EcEnc, SPREAD_AGGRESSIVE, SPREAD_LIGHT, SPREAD_NONE, SPREAD_NORMAL, celt_exp2,
    celt_inner_prod, celt_rsqrt, celt_rsqrt_norm, celt_sqrt, celt_sudiv, celt_udiv,
    dual_inner_prod, ec_ilog,
    math::isqrt32,
    quant_bands::E_MEANS,
    rate::{QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE},
    renormalise_vector,
    types::{CeltGlog, CeltSig, OpusCustomMode, OpusVal16, OpusVal32},
    vq::stereo_itheta,
};
use core::convert::TryFrom;

/// Small positive constant used throughout the CELT band tools to avoid divisions by zero.
const EPSILON: f32 = 1e-15;

/// Scaling factor applied to unit-norm vectors in the float build.
const NORM_SCALING: OpusVal16 = 1.0;

/// Shared context mirroring the state tracked by `struct band_ctx` in `celt/bands.c`.
#[derive(Debug)]
pub(crate) struct BandCtx<'a> {
    /// Whether the caller is encoding (`true`) or decoding (`false`).
    pub encode: bool,
    /// When `true`, the quantiser should resynthesise the canonical unit vector.
    pub resynth: bool,
    /// Active CELT mode driving the band configuration.
    pub mode: &'a OpusCustomMode<'a>,
    /// Index of the band currently being processed.
    pub band: usize,
    /// First band where intensity stereo becomes active.
    pub intensity: usize,
    /// Spreading decision selected for the frame.
    pub spread: i32,
    /// Time/frequency resolution change applied to the band.
    pub tf_change: i32,
    /// Remaining fractional bits available to the band quantiser.
    pub remaining_bits: i32,
    /// Per-band energy targets.
    pub band_e: &'a [CeltGlog],
    /// Random seed used for collapse prevention.
    pub seed: u32,
    /// Architecture hint for platform-specific optimisations.
    pub arch: i32,
    /// Theta rounding mode used by the stereo splitting logic.
    pub theta_round: i32,
    /// Whether inverse signalling is disabled for this band.
    pub disable_inv: bool,
    /// Forces deterministic synthesis when splitting noise.
    pub avoid_split_noise: bool,
}

/// Abstraction over the entropy coder used by the band quantisers.
pub(crate) enum BandCodingState<'a, 'b> {
    /// Encoding path using [`EcEnc`].
    Encoder(&'b mut EcEnc<'a>),
    /// Decoding path using [`EcDec`].
    Decoder(&'b mut EcDec<'a>),
}

impl<'a, 'b> BandCodingState<'a, 'b> {
    #[inline]
    fn is_encoder(&self) -> bool {
        matches!(self, Self::Encoder(_))
    }

    #[inline]
    fn encode_bits(&mut self, value: u32, bits: u32) {
        match self {
            Self::Encoder(enc) => enc.enc_bits(value, bits),
            Self::Decoder(_) => unreachable!("encoding requested on a decoder"),
        }
    }

    #[inline]
    fn decode_bits(&mut self, bits: u32) -> u32 {
        match self {
            Self::Decoder(dec) => dec.dec_bits(bits),
            Self::Encoder(_) => unreachable!("decoding requested on an encoder"),
        }
    }

    #[inline]
    fn tell_frac(&self) -> u32 {
        match self {
            Self::Encoder(enc) => ec_tell_frac(enc.ctx()),
            Self::Decoder(dec) => ec_tell_frac(dec.ctx()),
        }
    }

    #[inline]
    fn encode_range(&mut self, fl: u32, fh: u32, ft: u32) {
        match self {
            Self::Encoder(enc) => enc.encode(fl, fh, ft),
            Self::Decoder(_) => unreachable!("encoding requested on a decoder"),
        }
    }

    #[inline]
    fn decode_range(&mut self, ft: u32) -> u32 {
        match self {
            Self::Decoder(dec) => dec.decode(ft),
            Self::Encoder(_) => unreachable!("decoding requested on an encoder"),
        }
    }

    #[inline]
    fn update_range(&mut self, fl: u32, fh: u32, ft: u32) {
        match self {
            Self::Decoder(dec) => dec.update(fl, fh, ft),
            Self::Encoder(_) => unreachable!("decoder update invoked on an encoder"),
        }
    }

    #[inline]
    fn encode_uint(&mut self, value: u32, total: u32) {
        match self {
            Self::Encoder(enc) => enc.enc_uint(value, total),
            Self::Decoder(_) => unreachable!("encoding requested on a decoder"),
        }
    }

    #[inline]
    fn decode_uint(&mut self, total: u32) -> u32 {
        match self {
            Self::Decoder(dec) => dec.dec_uint(total),
            Self::Encoder(_) => unreachable!("decoding requested on an encoder"),
        }
    }

    #[inline]
    fn encode_bit_logp(&mut self, bit: i32, logp: u32) {
        match self {
            Self::Encoder(enc) => enc.enc_bit_logp(bit, logp),
            Self::Decoder(_) => unreachable!("encoding requested on a decoder"),
        }
    }

    #[inline]
    fn decode_bit_logp(&mut self, logp: u32) -> i32 {
        match self {
            Self::Decoder(dec) => dec.dec_bit_logp(logp),
            Self::Encoder(_) => unreachable!("decoding requested on an encoder"),
        }
    }
}

/// Split context mirroring the temporary state tracked in the C implementation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitCtx {
    pub inv: bool,
    pub imid: i32,
    pub iside: i32,
    pub delta: i32,
    pub itheta: i32,
    pub qalloc: i32,
}

fn mask_from_bits(bits: i32) -> u32 {
    if bits <= 0 {
        0
    } else if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_theta<'a, 'b>(
    ctx: &mut BandCtx<'a>,
    sctx: &mut SplitCtx,
    x: &mut [OpusVal16],
    y: &mut [OpusVal16],
    n: usize,
    b: &mut i32,
    b_current: i32,
    b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut u32,
    coder: &mut BandCodingState<'a, 'b>,
) {
    debug_assert!(n > 0, "band size must be positive");
    debug_assert!(x.len() >= n, "mid buffer shorter than band length");
    debug_assert!(y.len() >= n, "side buffer shorter than band length");

    let encode = ctx.encode;
    let mode = ctx.mode;
    let band = ctx.band;
    let intensity = ctx.intensity;
    let band_e = ctx.band_e;

    let log_n = i32::from(mode.log_n[band]);
    let pulse_cap = log_n + lm * (1_i32 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };

    let mut qn = compute_qn(n as i32, *b, offset, pulse_cap, stereo);
    if stereo && band >= intensity {
        qn = 1;
    }

    let mut itheta = if encode {
        stereo_itheta(x, y, stereo, n, ctx.arch)
    } else {
        0
    };

    let tell_before = coder.tell_frac() as i32;
    let mut inv = false;
    let imid;
    let iside;
    let mut delta;

    if qn != 1 {
        if encode {
            if !stereo || ctx.theta_round == 0 {
                itheta = ((itheta * qn) + 8192) >> 14;
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    let unquantized = celt_udiv((itheta * 16_384) as u32, qn as u32) as i32;
                    let mid = bitexact_cos(unquantized as i16) as i32;
                    let side = bitexact_cos((16_384 - unquantized) as i16) as i32;
                    let log_ratio = bitexact_log2tan(side, mid);
                    let scale = ((n as i32 - 1) << 7).max(0);
                    delta = frac_mul16(scale, log_ratio);
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                let bias = if itheta > 8192 {
                    32_767 / qn
                } else {
                    -32_767 / qn
                };
                let mut down = ((itheta * qn) + bias + 8_191) >> 14;
                down = down.clamp(0, qn - 1);
                itheta = if ctx.theta_round < 0 { down } else { down + 1 };
            }
        }

        if stereo && n > 2 {
            let p0 = 3;
            let mut x_val = itheta;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            if encode {
                let (fl, fh) = if x_val <= x0 {
                    (p0 * x_val, p0 * (x_val + 1))
                } else {
                    let base = (x0 + 1) * p0;
                    (base + (x_val - 1 - x0), base + (x_val - x0))
                };
                coder.encode_range(fl as u32, fh as u32, ft as u32);
            } else {
                let fs = coder.decode_range(ft as u32) as i32;
                x_val = if fs < (x0 + 1) * p0 {
                    fs / p0
                } else {
                    x0 + 1 + (fs - (x0 + 1) * p0)
                };
                let (fl, fh) = if x_val <= x0 {
                    (p0 * x_val, p0 * (x_val + 1))
                } else {
                    let base = (x0 + 1) * p0;
                    (base + (x_val - 1 - x0), base + (x_val - x0))
                };
                coder.update_range(fl as u32, fh as u32, ft as u32);
                itheta = x_val;
            }
        } else if b0 > 1 || stereo {
            if encode {
                coder.encode_uint(itheta as u32, (qn + 1) as u32);
            } else {
                itheta = coder.decode_uint((qn + 1) as u32) as i32;
            }
        } else {
            let half_qn = qn >> 1;
            let ft = (half_qn + 1) * (half_qn + 1);
            if encode {
                let (fl, fs) = if itheta <= half_qn {
                    let fl = (itheta * (itheta + 1)) >> 1;
                    (fl, itheta + 1)
                } else {
                    let fs = qn + 1 - itheta;
                    let fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                    (fl, fs)
                };
                coder.encode_range(fl as u32, (fl + fs) as u32, ft as u32);
            } else {
                let fm = coder.decode_range(ft as u32) as i32;
                let threshold = (half_qn * (half_qn + 1)) >> 1;
                let (fl, fs);
                if fm < threshold {
                    let root = isqrt32((8 * fm + 1) as u32) as i32;
                    itheta = (root - 1) >> 1;
                    fl = (itheta * (itheta + 1)) >> 1;
                    fs = itheta + 1;
                } else {
                    let root = isqrt32((8 * (ft - fm - 1) + 1) as u32) as i32;
                    itheta = (2 * (qn + 1) - root) >> 1;
                    fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                    fs = qn + 1 - itheta;
                }
                coder.update_range(fl as u32, (fl + fs) as u32, ft as u32);
            }
        }

        debug_assert!(itheta >= 0);
        if qn > 0 {
            itheta = celt_udiv((itheta * 16_384) as u32, qn as u32) as i32;
        }
        if encode && stereo {
            if itheta == 0 {
                intensity_stereo(mode, x, y, band_e, band, n);
            } else {
                let x_band = &mut x[..n];
                let y_band = &mut y[..n];
                stereo_split(x_band, y_band);
            }
        }
    } else if stereo {
        if encode {
            inv = itheta > 8_192 && !ctx.disable_inv;
            if inv {
                for sample in y.iter_mut().take(n) {
                    *sample = -*sample;
                }
            }
            intensity_stereo(mode, x, y, band_e, band, n);
        }

        let threshold = 2 << BITRES;
        if *b > threshold && ctx.remaining_bits > threshold {
            if encode {
                coder.encode_bit_logp(if inv { 1 } else { 0 }, 2);
            } else {
                inv = coder.decode_bit_logp(2) != 0;
            }
        } else {
            inv = false;
        }

        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    }

    let tell_after = coder.tell_frac() as i32;
    let qalloc = tell_after - tell_before;
    *b -= qalloc;

    let b_mask = mask_from_bits(b_current);
    let band_scale = ((n as i32 - 1) << 7).max(0);

    if itheta == 0 {
        imid = 32_767;
        iside = 0;
        *fill &= b_mask;
        delta = -16_384;
    } else if itheta == 16_384 {
        imid = 0;
        iside = 32_767;
        let shifted = if b_current <= 0 {
            0
        } else if b_current >= 32 {
            u32::MAX
        } else {
            b_mask << (b_current as u32)
        };
        *fill &= shifted;
        delta = 16_384;
    } else {
        imid = i32::from(bitexact_cos(itheta as i16));
        iside = i32::from(bitexact_cos((16_384 - itheta) as i16));
        delta = frac_mul16(band_scale, bitexact_log2tan(iside, imid));
    }

    sctx.inv = inv;
    sctx.imid = imid;
    sctx.iside = iside;
    sctx.delta = delta;
    sctx.itheta = itheta;
    sctx.qalloc = qalloc;
}

/// Indexing table for converting natural-order Hadamard coefficients into the
/// "ordery" permutation used by CELT's spreading analysis.
///
/// The layout mirrors the compact array embedded in `celt/bands.c`, grouping
/// permutations for strides of 2, 4, 8, and 16. The Hadamard interleaving logic
/// selects the slice corresponding to the current stride when the `hadamard`
/// flag is active.
const ORDERY_TABLES: [&[usize]; 4] = [
    &[1, 0],
    &[3, 0, 2, 1],
    &[7, 0, 4, 3, 6, 1, 5, 2],
    &[15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5],
];

fn hadamard_ordery(stride: usize) -> Option<&'static [usize]> {
    match stride {
        2 => Some(ORDERY_TABLES[0]),
        4 => Some(ORDERY_TABLES[1]),
        8 => Some(ORDERY_TABLES[2]),
        16 => Some(ORDERY_TABLES[3]),
        _ => None,
    }
}

/// Fixed-point fractional multiply mirroring the `FRAC_MUL16` macro from the C
/// sources.
#[inline]
fn frac_mul16(a: i32, b: i32) -> i32 {
    let a = a as i16;
    let b = b as i16;
    (16_384 + i32::from(a) * i32::from(b)) >> 15
}

/// Bit-exact cosine approximation used by the band analysis heuristics.
///
/// Mirrors `bitexact_cos()` from `celt/bands.c`. The helper operates entirely
/// in 16-bit fixed-point arithmetic so that it matches the reference
/// implementation across platforms.
#[must_use]
pub(crate) fn bitexact_cos(x: i16) -> i16 {
    let tmp = (4_096 + i32::from(x) * i32::from(x)) >> 13;
    let mut x2 = tmp;
    x2 = (32_767 - x2) + frac_mul16(x2, -7_651 + frac_mul16(x2, 8_277 + frac_mul16(-626, x2)));
    (1 + x2) as i16
}

/// Bit-exact logarithmic tangent helper used by the stereo analysis logic.
///
/// Mirrors `bitexact_log2tan()` from `celt/bands.c`, relying on the shared
/// range coder log helper to normalise the sine and cosine magnitudes before
/// evaluating the polynomial approximation.
#[must_use]
pub(crate) fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32) as i32;
    let ls = ec_ilog(isin as u32) as i32;

    let shift_cos = 15 - lc;
    debug_assert!(shift_cos >= 0);
    let icos = icos << shift_cos;

    let shift_sin = 15 - ls;
    debug_assert!(shift_sin >= 0);
    let isin = isin << shift_sin;

    ((ls - lc) << 11) + frac_mul16(isin, frac_mul16(isin, -2_597) + 7_932)
        - frac_mul16(icos, frac_mul16(icos, -2_597) + 7_932)
}

/// Applies a hysteresis decision to a scalar value.
///
/// Mirrors `hysteresis_decision()` from `celt/bands.c`. The helper walks the
/// provided threshold table and returns the first band whose threshold exceeds
/// the current value. Hysteresis offsets are used to avoid flapping between
/// adjacent bands when the value is close to the threshold shared by two
/// regions. The `prev` argument supplies the previously selected band.
#[must_use]
pub(crate) fn hysteresis_decision(
    value: OpusVal16,
    thresholds: &[OpusVal16],
    hysteresis: &[OpusVal16],
    prev: usize,
) -> usize {
    debug_assert_eq!(thresholds.len(), hysteresis.len());
    let count = thresholds.len();
    debug_assert!(prev <= count, "prev index must be within the table bounds");

    let mut index = 0;
    while index < count {
        if value < thresholds[index] {
            break;
        }
        index += 1;
    }

    if prev < count && index > prev && value < thresholds[prev] + hysteresis[prev] {
        index = prev;
    }

    if prev > 0 && index < prev && value > thresholds[prev - 1] - hysteresis[prev - 1] {
        index = prev;
    }

    index
}

/// Linear congruential pseudo-random number generator used by the band tools.
///
/// Mirrors `celt_lcg_rand()` from `celt/bands.c`. The generator matches the
/// parameters from Numerical Recipes and returns a new 32-bit seed value.
#[must_use]
#[inline]
pub(crate) fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// Computes the number of quantisation levels available for the stereo split angle.
///
/// Mirrors the `compute_qn()` helper from `celt/bands.c`. The routine determines how
/// finely the mid/side angle can be quantised given the number of available bits and
/// the band configuration. The return value is always even so that the subsequent
/// entropy coding can mirror the reference implementation's lookup table.
pub(crate) fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];

    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }

    let mut qb = celt_sudiv(b + n2 * offset, n2);
    let pulse_guard = b - pulse_cap - (4 << BITRES);
    qb = qb.min(pulse_guard);
    qb = qb.min(8 << BITRES);

    let threshold = (1 << BITRES) >> 1;
    let qn = if qb < threshold {
        1
    } else {
        let index = (qb & 0x7) as usize;
        let shift = 14 - (qb >> BITRES);
        let mut value = EXP2_TABLE8[index] >> shift;
        value = (value + 1) >> 1;
        value << 1
    };

    debug_assert!(qn <= 256);
    qn
}

fn quant_band_n1_channel<'a, 'b>(
    ctx: &mut BandCtx<'a>,
    samples: &mut [OpusVal16],
    coder: &mut BandCodingState<'a, 'b>,
) {
    assert!(
        !samples.is_empty(),
        "quant_band_n1 expects non-empty coefficient slices",
    );

    let mut sign = 0;
    let bit_budget = 1_i32 << BITRES;
    if ctx.remaining_bits >= bit_budget {
        if ctx.encode {
            debug_assert!(coder.is_encoder());
            sign = (samples[0] < 0.0) as i32;
            coder.encode_bits(sign as u32, 1);
        } else {
            debug_assert!(!coder.is_encoder());
            sign = coder.decode_bits(1) as i32;
        }
        ctx.remaining_bits -= bit_budget;
    }

    if ctx.resynth {
        samples[0] = if sign != 0 {
            -NORM_SCALING
        } else {
            NORM_SCALING
        };
    }
}

/// Handles the single-pulse PVQ case where only the sign needs to be coded.
///
/// Mirrors `quant_band_n1()` from `celt/bands.c`, emitting (or consuming) a raw
/// sign bit when enough range coder budget is available. The helper optionally
/// resynthesises the canonical unit vector so that future spreading and collapse
/// prevention logic can operate on the reconstructed coefficients.
pub(crate) fn quant_band_n1<'a, 'b>(
    ctx: &mut BandCtx<'a>,
    x: &mut [OpusVal16],
    y: Option<&mut [OpusVal16]>,
    lowband_out: Option<&mut [OpusVal16]>,
    coder: &mut BandCodingState<'a, 'b>,
) -> usize {
    debug_assert_eq!(ctx.encode, coder.is_encoder());

    quant_band_n1_channel(ctx, x, coder);
    if let Some(y_samples) = y {
        quant_band_n1_channel(ctx, y_samples, coder);
    }

    if let Some(lowband) = lowband_out.filter(|lowband| !lowband.is_empty()) {
        lowband[0] = x[0];
    }

    1
}

/// Computes stereo weighting factors used when balancing channel distortion.
///
/// Mirrors `compute_channel_weights()` from `celt/bands.c`. The helper adjusts
/// the per-channel energies by a fraction of the smaller energy so that the
/// stereo weighting is slightly more conservative than a pure proportional
/// split.
#[must_use]
pub(crate) fn compute_channel_weights(ex: OpusVal32, ey: OpusVal32) -> [OpusVal16; 2] {
    let min_energy = ex.min(ey);
    let adjusted_ex = ex + min_energy / 3.0;
    let adjusted_ey = ey + min_energy / 3.0;
    [adjusted_ex, adjusted_ey]
}

/// Collapses an intensity-coded stereo band back into the mid channel.
///
/// Mirrors the float configuration of `intensity_stereo()` from `celt/bands.c`.
/// The helper derives linear weights from the per-channel band energies and
/// mixes the encoded side channel into the mid channel while preserving the
/// overall energy of the pair.
pub(crate) fn intensity_stereo(
    mode: &OpusCustomMode<'_>,
    x: &mut [OpusVal16],
    y: &[OpusVal16],
    band_e: &[OpusVal32],
    band_id: usize,
    n: usize,
) {
    assert!(
        band_id < mode.num_ebands,
        "band index must be within the mode range"
    );
    assert!(x.len() >= n, "output band must contain at least n samples");
    assert!(y.len() >= n, "side band must contain at least n samples");

    let stride = mode.num_ebands;
    assert!(
        band_e.len() >= stride * 2,
        "band energy buffer must store both channel energies",
    );
    assert!(
        band_id + stride < band_e.len(),
        "band energy buffer too small for right channel",
    );

    let left = band_e[band_id];
    let right = band_e[band_id + stride];
    let norm = EPSILON + celt_sqrt(EPSILON + left * left + right * right);
    let a1 = left / norm;
    let a2 = right / norm;

    for idx in 0..n {
        let l = x[idx];
        let r = y[idx];
        x[idx] = a1 * l + a2 * r;
    }
}

/// Converts a mid/side representation into left/right stereo samples.
///
/// Mirrors `stereo_split()` from `celt/bands.c`. The helper applies the
/// orthonormal transform that maps a mid (sum) signal and a side (difference)
/// signal back to the left/right domain while preserving energy. CELT encodes
/// mid/side pairs using Q15 fixed-point arithmetic; the float build operates on
/// `f32`, so the Rust port multiplies by `FRAC_1_SQRT_2` instead of the
/// `QCONST16(0.70710678f, 15)` constant used in the original source.
pub(crate) fn stereo_split(x: &mut [f32], y: &mut [f32]) {
    assert_eq!(
        x.len(),
        y.len(),
        "stereo_split expects slices of equal length",
    );

    for (left, right) in x.iter_mut().zip(y.iter_mut()) {
        let mid = FRAC_1_SQRT_2 * *left;
        let side = FRAC_1_SQRT_2 * *right;
        *left = mid + side;
        *right = side - mid;
    }
}

/// Restores energy to bands that collapsed during transient coding.
///
/// Mirrors the float build of `anti_collapse()` from `celt/bands.c`. When a
/// short MDCT band loses all pulses the decoder injects shaped noise with a
/// gain derived from recent band energies. The helper mirrors the reference
/// pseudo-random sequence, energy guards, and subsequent renormalisation so the
/// decoder matches the C implementation bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn anti_collapse(
    mode: &OpusCustomMode<'_>,
    x: &mut [OpusVal16],
    collapse_masks: &[u8],
    lm: usize,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[CeltGlog],
    prev1_log_e: &[CeltGlog],
    prev2_log_e: &[CeltGlog],
    pulses: &[i32],
    mut seed: u32,
    encode: bool,
    arch: i32,
) {
    assert!(channels > 0, "anti_collapse requires at least one channel");
    assert!(start <= end, "start band must not exceed end band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(
        collapse_masks.len() >= channels * end,
        "collapse masks too short"
    );
    assert!(
        log_e.len() >= channels * mode.num_ebands,
        "logE buffer too small"
    );
    assert!(
        prev1_log_e.len() >= channels * mode.num_ebands,
        "prev1 buffer too small"
    );
    assert!(
        prev2_log_e.len() >= channels * mode.num_ebands,
        "prev2 buffer too small"
    );
    assert!(
        pulses.len() >= end,
        "pulse buffer too small for requested bands"
    );

    let expected_stride = mode.short_mdct_size << lm;
    assert_eq!(
        size, expected_stride,
        "channel stride must match the MDCT length for the block size",
    );
    assert!(
        x.len() >= channels * size,
        "spectrum buffer shorter than the requested channel span",
    );

    let block_count = 1usize << lm;
    let band_stride = mode.num_ebands;

    for (band, &pulses_for_band) in pulses
        .iter()
        .enumerate()
        .take(end)
        .skip(start)
    {
        let band_begin =
            usize::try_from(mode.e_bands[band]).expect("band index must be non-negative");
        let band_end =
            usize::try_from(mode.e_bands[band + 1]).expect("band index must be non-negative");
        let width = band_end.saturating_sub(band_begin);
        if width == 0 {
            continue;
        }

        assert!(pulses_for_band >= 0, "pulse counts must be non-negative");
        let numerator = u32::try_from(pulses_for_band)
            .expect("pulse count fits in u32")
            .wrapping_add(1);
        let denom = u32::try_from(width).expect("band width fits in u32");
        debug_assert!(denom > 0, "band width must be positive");
        let depth = (celt_udiv(numerator, denom) >> lm) as i32;

        let thresh = 0.5 * celt_exp2(-0.125 * depth as f32);
        let sqrt_1 = celt_rsqrt((width << lm) as f32);

        for channel in 0..channels {
            let mask = collapse_masks[band * channels + channel] as u32;
            let channel_base = channel * size;
            let band_base = channel_base + (band_begin << lm);
            let band_len = width << lm;
            assert!(
                band_base + band_len <= x.len(),
                "band slice exceeds spectrum length"
            );

            let mut prev1 = prev1_log_e[channel * band_stride + band];
            let mut prev2 = prev2_log_e[channel * band_stride + band];

            if !encode && channels == 1 {
                let alt = band_stride + band;
                if alt < prev1_log_e.len() {
                    prev1 = prev1.max(prev1_log_e[alt]);
                }
                if alt < prev2_log_e.len() {
                    prev2 = prev2.max(prev2_log_e[alt]);
                }
            }

            let mut ediff = log_e[channel * band_stride + band] - prev1.min(prev2);
            if ediff < 0.0 {
                ediff = 0.0;
            }

            let mut r = 2.0 * celt_exp2(-ediff);
            if lm == 3 {
                r *= SQRT_2;
            }
            r = r.min(thresh);
            r *= sqrt_1;

            let mut needs_renorm = false;

            for k in 0..block_count {
                if mask & (1u32 << k) == 0 {
                    for j in 0..width {
                        seed = celt_lcg_rand(seed);
                        let idx = band_base + (j << lm) + k;
                        x[idx] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    needs_renorm = true;
                }
            }

            if needs_renorm {
                let end_idx = band_base + band_len;
                renormalise_vector(&mut x[band_base..end_idx], band_len, 1.0, arch);
            }
        }
    }
}

/// Reconstructs left/right stereo samples from a mid/side representation.
///
/// Mirrors the float configuration of `stereo_merge()` from `celt/bands.c`.
/// The helper evaluates the energies of the `X ± Y` combinations to derive
/// normalisation gains, then applies the inverse transform to recover the
/// left and right channels. If either energy falls below the conservative
/// threshold used by the reference implementation, the side channel is
/// replaced by the mid channel to avoid amplifying near-silent noise.
pub(crate) fn stereo_merge(x: &mut [OpusVal16], y: &mut [OpusVal16], mid: OpusVal32) {
    assert_eq!(
        x.len(),
        y.len(),
        "stereo_merge expects slices of equal length",
    );

    if x.is_empty() {
        return;
    }

    let (mut cross, side_energy) = dual_inner_prod(y, x, y);
    cross *= mid;
    let mid_energy = mid * mid;
    let el = mid_energy + side_energy - 2.0 * cross;
    let er = mid_energy + side_energy + 2.0 * cross;

    if er < 6e-4 || el < 6e-4 {
        y.copy_from_slice(x);
        return;
    }

    let lgain = celt_rsqrt_norm(el);
    let rgain = celt_rsqrt_norm(er);

    for (left, right) in x.iter_mut().zip(y.iter_mut()) {
        let mid_scaled = mid * *left;
        let side_val = *right;
        *left = lgain * (mid_scaled - side_val);
        *right = rgain * (mid_scaled + side_val);
    }
}

/// Duplicates the initial hybrid folding samples needed by the next band.
///
/// Mirrors the `special_hybrid_folding()` helper from `celt/bands.c`. The
/// function ensures that enough low-frequency PVQ coefficients are available
/// when the second hybrid band needs to fold spectrum from the first one. The
/// decoder and the resynthesis-enabled encoder both depend on this behaviour to
/// match the reference bitstream exactly.
pub(crate) fn special_hybrid_folding(
    mode: &OpusCustomMode,
    norm: &mut [OpusVal16],
    norm2: Option<&mut [OpusVal16]>,
    start: usize,
    m: usize,
    dual_stereo: bool,
) {
    debug_assert!(start + 2 < mode.e_bands.len(), "hybrid folding requires two successor bands");

    let e_bands = mode.e_bands;
    let n1 = m * (e_bands[start + 1] - e_bands[start]) as usize;
    let n2 = m * (e_bands[start + 2] - e_bands[start + 1]) as usize;

    if n2 <= n1 {
        return;
    }

    let copy_len = n2 - n1;
    let src_start = 2 * n1 - n2;

    debug_assert!(n1 + copy_len <= norm.len(), "destination slice exceeds bounds");
    debug_assert!(src_start + copy_len <= norm.len(), "source slice exceeds bounds");

    let temp: Vec<OpusVal16> = norm[src_start..src_start + copy_len].to_vec();
    norm[n1..n1 + copy_len].copy_from_slice(&temp);

    if let (true, Some(norm2)) = (dual_stereo, norm2) {
        debug_assert!(n1 + copy_len <= norm2.len(), "destination slice exceeds bounds");
        debug_assert!(src_start + copy_len <= norm2.len(), "source slice exceeds bounds");

        let temp2: Vec<OpusVal16> = norm2[src_start..src_start + copy_len].to_vec();
        norm2[n1..n1 + copy_len].copy_from_slice(&temp2);
    }
}

/// Decides how aggressively PVQ pulses should be spread in the current frame.
///
/// Mirrors the float configuration of `spreading_decision()` from
/// `celt/bands.c`. The helper analyses the normalised spectrum stored in `x`
/// and classifies each band based on the proportion of low-energy coefficients.
/// The resulting score is filtered through a simple recursive average and a
/// small hysteresis term controlled by the previous decision. High-frequency
/// statistics optionally update the pitch tapset selector when `update_hf` is
/// `true`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spreading_decision(
    mode: &OpusCustomMode,
    x: &[OpusVal16],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: usize,
    channels: usize,
    m: usize,
    spread_weight: &[i32],
) -> i32 {
    assert!(end > 0, "band range must contain at least one band");
    assert!(end <= mode.num_ebands, "band range exceeds mode span");
    assert!(spread_weight.len() >= end, "insufficient spread weights");

    let n0 = m * mode.short_mdct_size;
    assert!(x.len() >= channels * n0, "spectrum buffer too small");

    let last_band_width =
        m * (mode.e_bands[end] as usize).saturating_sub(mode.e_bands[end - 1] as usize);
    if last_band_width <= 8 {
        return SPREAD_NONE;
    }

    let mut sum = 0i32;
    let mut nb_bands = 0i32;
    let mut hf_sum = 0i32;

    for c in 0..channels {
        let channel_base = c * n0;
        for (band, &weight) in spread_weight.iter().take(end).enumerate() {
            let start = m * (mode.e_bands[band] as usize);
            let stop = m * (mode.e_bands[band + 1] as usize);
            let n = stop - start;
            if n <= 8 {
                continue;
            }

            let slice = &x[channel_base + start..channel_base + stop];
            let mut tcount = [0i32; 3];
            for &value in slice {
                let x2n = value * value * n as OpusVal16;
                if x2n < 0.25 {
                    tcount[0] += 1;
                }
                if x2n < 0.0625 {
                    tcount[1] += 1;
                }
                if x2n < 0.015625 {
                    tcount[2] += 1;
                }
            }

            if band + 4 > mode.num_ebands {
                let numerator = 32 * (tcount[1] + tcount[0]);
                hf_sum += celt_udiv(numerator as u32, n as u32) as i32;
            }

            let mut tmp = 0i32;
            if 2 * tcount[2] >= n as i32 {
                tmp += 1;
            }
            if 2 * tcount[1] >= n as i32 {
                tmp += 1;
            }
            if 2 * tcount[0] >= n as i32 {
                tmp += 1;
            }

            sum += tmp * weight;
            nb_bands += weight;
        }
    }

    if update_hf {
        if hf_sum != 0 {
            let denom = (channels as i32) * (4 - mode.num_ebands as i32 + end as i32);
            if denom > 0 {
                hf_sum = celt_udiv(hf_sum as u32, denom as u32) as i32;
            } else {
                hf_sum = 0;
            }
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        match *tapset_decision {
            2 => hf_sum += 4,
            0 => hf_sum -= 4,
            _ => {}
        }
        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }

    assert!(
        nb_bands > 0,
        "spreading analysis requires at least one band"
    );
    let scaled = ((sum as i64) << 8) as u32;
    let denom = nb_bands as u32;
    let mut sum = celt_udiv(scaled, denom) as i32;
    sum = (sum + *average) >> 1;
    *average = sum;

    let hysteresis = ((3 - last_decision) << 7) + 64;
    sum = (3 * sum + hysteresis + 2) >> 2;

    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

/// Restores the natural band ordering after a Hadamard transform.
///
/// Mirrors `deinterleave_hadamard()` from `celt/bands.c`. The routine copies the
/// interleaved coefficients into a temporary buffer before writing them back in
/// natural band order. When `hadamard` is `true`, the function applies the
/// "ordery" permutation so that the Hadamard DC term appears at the end of the
/// output sequence, matching the reference implementation.
pub(crate) fn deinterleave_hadamard(x: &mut [OpusVal16], n0: usize, stride: usize, hadamard: bool) {
    if stride == 0 {
        return;
    }

    let n = n0.checked_mul(stride).expect("stride * n0 overflowed");
    assert!(x.len() >= n, "input buffer too small for deinterleave");

    if n == 0 {
        return;
    }

    let mut tmp = vec![0.0f32; n];

    if hadamard {
        let ordery = hadamard_ordery(stride)
            .expect("hadamard interleave only defined for strides of 2, 4, 8, or 16");
        assert_eq!(ordery.len(), stride);
        for (i, &ord) in ordery.iter().enumerate() {
            for j in 0..n0 {
                tmp[ord * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }

    x[..n].copy_from_slice(&tmp);
}

/// Applies the Hadamard interleaving used by CELT's spreading decisions.
///
/// Mirrors `interleave_hadamard()` from `celt/bands.c`. The helper stores the
/// natural-order coefficients into a temporary buffer, optionally applying the
/// "ordery" permutation when `hadamard` is `true`. The resulting layout matches
/// the reference code, ensuring that deinterleaving reverses the transform
/// exactly.
pub(crate) fn interleave_hadamard(x: &mut [OpusVal16], n0: usize, stride: usize, hadamard: bool) {
    if stride == 0 {
        return;
    }

    let n = n0.checked_mul(stride).expect("stride * n0 overflowed");
    assert!(x.len() >= n, "input buffer too small for interleave");

    if n == 0 {
        return;
    }

    let mut tmp = vec![0.0f32; n];

    if hadamard {
        let ordery = hadamard_ordery(stride)
            .expect("hadamard interleave only defined for strides of 2, 4, 8, or 16");
        assert_eq!(ordery.len(), stride);
        for (i, &ord) in ordery.iter().enumerate() {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ord * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }

    x[..n].copy_from_slice(&tmp);
}

/// Applies a single-level Haar transform across interleaved coefficients.
///
/// Mirrors `haar1()` from `celt/bands.c`, scaling each pair of samples by
/// `1/sqrt(2)` before computing their sum and difference. The coefficients are
/// laid out in `stride` interleaved groups, matching the memory layout used by
/// CELT's band folding routines.
pub(crate) fn haar1(x: &mut [OpusVal16], n0: usize, stride: usize) {
    if stride == 0 || n0 < 2 {
        return;
    }

    let half = n0 / 2;
    if half == 0 {
        return;
    }

    let required = stride * n0;
    assert!(
        x.len() >= required,
        "haar1 expects at least stride * n0 coefficients"
    );

    let scale = FRAC_1_SQRT_2 as OpusVal16;

    for i in 0..stride {
        for j in 0..half {
            let idx0 = stride * (2 * j) + i;
            let idx1 = idx0 + stride;
            debug_assert!(idx1 < x.len());

            let tmp1 = scale * x[idx0];
            let tmp2 = scale * x[idx1];
            x[idx0] = tmp1 + tmp2;
            x[idx1] = tmp1 - tmp2;
        }
    }
}

/// Computes the per-band energy for the supplied channels.
///
/// Ports the float build of `compute_band_energies()` from `celt/bands.c`. The
/// helper sums the squared magnitudes within each critical band and stores the
/// square-rooted result in `band_e`. A small bias of `1e-27` mirrors the
/// reference implementation and keeps the normalisation stable even for silent
/// bands.
pub(crate) fn compute_band_energies(
    mode: &OpusCustomMode<'_>,
    x: &[CeltSig],
    band_e: &mut [CeltGlog],
    end: usize,
    channels: usize,
    lm: usize,
    arch: i32,
) {
    let _ = arch;

    assert!(
        end <= mode.num_ebands,
        "end band must not exceed mode bands"
    );
    assert!(
        mode.e_bands.len() > end,
        "eBands must contain end + 1 entries"
    );

    let n = mode.short_mdct_size << lm;
    assert!(
        x.len() >= channels * n,
        "input spectrum is too short for the mode"
    );

    let stride = mode.num_ebands;
    assert!(
        band_e.len() >= channels * stride,
        "band energy buffer too small"
    );

    for c in 0..channels {
        let signal_base = c * n;
        let energy_base = c * stride;

        for band in 0..end {
            let band_start = (mode.e_bands[band] as usize) << lm;
            let band_end = (mode.e_bands[band + 1] as usize) << lm;
            assert!(band_end <= n, "band end exceeds MDCT length");

            let slice = &x[signal_base + band_start..signal_base + band_end];
            let sum = 1e-27_f32 + celt_inner_prod(slice, slice);
            band_e[energy_base + band] = celt_sqrt(sum);
        }
    }
}

/// Normalises each band to unit energy.
///
/// Mirrors the float implementation of `normalise_bands()` from `celt/bands.c`
/// by scaling the MDCT spectrum in-place. The gain for each band is computed
/// from the `band_e` table produced by [`compute_band_energies`], with the same
/// `1e-27` bias to guard against division by zero.
pub(crate) fn normalise_bands(
    mode: &OpusCustomMode<'_>,
    freq: &[CeltSig],
    x: &mut [OpusVal16],
    band_e: &[CeltGlog],
    end: usize,
    channels: usize,
    m: usize,
) {
    assert!(
        end <= mode.num_ebands,
        "end band must not exceed mode bands"
    );
    assert!(
        mode.e_bands.len() > end,
        "eBands must contain end + 1 entries"
    );

    let n = m * mode.short_mdct_size;
    assert!(freq.len() >= channels * n, "frequency buffer too small");
    assert!(x.len() >= channels * n, "normalisation buffer too small");

    let stride = mode.num_ebands;
    assert!(
        band_e.len() >= channels * stride,
        "band energy buffer too small"
    );

    for c in 0..channels {
        let freq_base = c * n;
        let energy_base = c * stride;

        for band in 0..end {
            let start = m * (mode.e_bands[band] as usize);
            let stop = m * (mode.e_bands[band + 1] as usize);
            assert!(stop <= n, "band end exceeds MDCT length");

            let gain = 1.0 / (1e-27_f32 + band_e[energy_base + band]);
            for idx in start..stop {
                x[freq_base + idx] = freq[freq_base + idx] * gain;
            }
        }
    }
}

/// Rescales the unit-energy coefficients back to their target magnitudes.
///
/// Mirrors the float variant of `denormalise_bands()` from `celt/bands.c` by
/// multiplying each normalised coefficient by the exponential of its target
/// log-energy. The helper also zeroes samples preceding the `start` band and
/// clears the tail of the buffer based on the downsampling factor so that the
/// reconstruction matches the reference implementation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn denormalise_bands(
    mode: &OpusCustomMode<'_>,
    x: &[OpusVal16],
    freq: &mut [CeltSig],
    band_log_e: &[CeltGlog],
    mut start: usize,
    mut end: usize,
    m: usize,
    downsample: usize,
    silence: bool,
) {
    assert!(
        end <= mode.num_ebands,
        "end band must not exceed mode bands"
    );
    assert!(
        mode.e_bands.len() > end,
        "eBands must contain end + 1 entries"
    );
    assert!(band_log_e.len() >= end, "bandLogE must provide end entries");

    let n = m * mode.short_mdct_size;
    assert!(freq.len() >= n, "frequency buffer too small");
    assert!(x.len() >= n, "normalised spectrum buffer too small");
    assert!(downsample > 0, "downsample factor must be non-zero");

    let mut bound = m * usize::try_from(mode.e_bands[end])
        .expect("band edge must be non-negative")
        .min(n);

    if downsample != 1 {
        bound = bound.min(n / downsample);
    }

    if silence {
        bound = 0;
        start = 0;
        end = 0;
    }

    let start_edge =
        m * usize::try_from(mode.e_bands[start]).expect("band edge must be non-negative");
    freq[..start_edge].fill(0.0);

    let mut freq_idx = start_edge;
    let mut x_idx = start_edge;

    for (band, &band_gain_log) in band_log_e
        .iter()
        .enumerate()
        .take(end)
        .skip(start)
    {
        let band_end =
            m * usize::try_from(mode.e_bands[band + 1]).expect("band edge must be non-negative");
        assert!(band_end <= n, "band end exceeds MDCT length");
        assert!(band < E_MEANS.len(), "E_MEANS lacks entry for band");

        let gain = celt_exp2((band_gain_log + E_MEANS[band]).min(32.0));
        while freq_idx < band_end {
            freq[freq_idx] = x[x_idx] * gain;
            freq_idx += 1;
            x_idx += 1;
        }
    }

    freq[bound..n].fill(0.0);
}

#[cfg(test)]
mod tests {
    use super::{
        BandCodingState, BandCtx, EPSILON, NORM_SCALING, SplitCtx, anti_collapse, bitexact_cos,
        bitexact_log2tan, celt_lcg_rand, compute_band_energies, compute_channel_weights,
        compute_qn, compute_theta, deinterleave_hadamard, denormalise_bands, frac_mul16, haar1,
        hysteresis_decision, intensity_stereo, interleave_hadamard, normalise_bands,
        quant_band_n1, special_hybrid_folding, spreading_decision, stereo_merge, stereo_split,
    };
    use crate::celt::entcode::BITRES;
    use crate::celt::math::celt_exp2;
    use crate::celt::quant_bands::E_MEANS;
    use crate::celt::types::{CeltSig, MdctLookup, OpusCustomMode, PulseCacheData};
    use crate::celt::{
        EcDec, EcEnc, SPREAD_AGGRESSIVE, SPREAD_NONE, SPREAD_NORMAL, celt_rsqrt_norm,
        dual_inner_prod,
    };
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn hysteresis_matches_reference_logic() {
        // Synthetic thresholds with simple hysteresis offsets.
        let thresholds = [0.2_f32, 0.4, 0.6, 0.8];
        let hysteresis = [0.05_f32; 4];

        fn reference(value: f32, thresholds: &[f32], hysteresis: &[f32], prev: usize) -> usize {
            let count = thresholds.len();
            let mut i = 0;
            while i < count {
                if value < thresholds[i] {
                    break;
                }
                i += 1;
            }

            if i > prev && prev < count && value < thresholds[prev] + hysteresis[prev] {
                i = prev;
            }
            if i < prev && prev > 0 && value > thresholds[prev - 1] - hysteresis[prev - 1] {
                i = prev;
            }
            i
        }

        let values = [0.0, 0.15, 0.25, 0.39, 0.41, 0.59, 0.61, 0.79, 0.81, 0.95];

        for prev in 0..=thresholds.len() {
            for &value in &values {
                let expected = reference(value, &thresholds, &hysteresis, prev);
                assert_eq!(
                    hysteresis_decision(value, &thresholds, &hysteresis, prev),
                    expected,
                    "value {value}, prev {prev}",
                );
            }
        }
    }

    #[test]
    fn haar1_preserves_signal_when_applied_twice() {
        let mut data = vec![
            0.25_f32, -1.5, 3.5, 0.75, -2.25, 1.0, 0.5, -0.125, 2.0, -3.0, 1.5, 0.25,
        ];
        let original = data.clone();

        // Apply the transform twice; the Haar matrix is orthonormal so the
        // second application inverts the first.
        haar1(&mut data, 12, 1);
        haar1(&mut data, 12, 1);

        for (expected, observed) in original.iter().zip(data.iter()) {
            assert!(
                (expected - observed).abs() <= 1e-6,
                "expected {expected}, got {observed}"
            );
        }
    }

    #[test]
    fn channel_weights_match_reference_formula() {
        let cases = [
            (0.0, 0.0),
            (1.0, 4.0),
            (4.0, 1.0),
            (10.0, 10.0),
            (3.75, 0.25),
        ];

        for &(ex, ey) in &cases {
            let weights = compute_channel_weights(ex, ey);
            let min_energy = ex.min(ey);
            let reference_ex = ex + min_energy / 3.0;
            let reference_ey = ey + min_energy / 3.0;

            assert!((weights[0] - reference_ex).abs() <= f32::EPSILON * 4.0);
            assert!((weights[1] - reference_ey).abs() <= f32::EPSILON * 4.0);
        }
    }

    #[test]
    fn intensity_stereo_matches_reference_weights() {
        let e_bands = [0i16, 2, 4, 6, 8];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 4];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(4, 0);
        let mode = OpusCustomMode::new(
            48_000,
            0,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            PulseCacheData::default(),
        );

        let mut x = vec![0.5, -0.75, 0.25, -0.125];
        let y = vec![0.25, 0.5, -0.5, 0.75];
        let mut band_e = vec![0.0f32; mode.num_ebands * 2];
        band_e[2] = 1.8;
        band_e[2 + mode.num_ebands] = 0.9;

        let left = band_e[2];
        let right = band_e[2 + mode.num_ebands];
        let norm = EPSILON + (EPSILON + left * left + right * right).sqrt();
        let a1 = left / norm;
        let a2 = right / norm;
        let mut expected = x.clone();
        for (idx, value) in expected.iter_mut().enumerate() {
            *value = a1 * *value + a2 * y[idx];
        }

        intensity_stereo(&mode, &mut x, &y, &band_e, 2, y.len());

        for (idx, (&observed, &reference)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (observed - reference).abs() <= 1e-6,
                "sample {idx}: observed={observed}, expected={reference}"
            );
        }
    }

    #[test]
    fn lcg_rand_produces_expected_sequence() {
        let mut seed = 0xDEAD_BEEF_u32;
        let mut expected = [0_u32; 5];
        for slot in &mut expected {
            let next = ((1_664_525_u64 * u64::from(seed)) + 1_013_904_223_u64) & 0xFFFF_FFFF;
            *slot = next as u32;
            seed = next as u32;
        }

        seed = 0xDEAD_BEEF_u32;
        for &value in &expected {
            seed = celt_lcg_rand(seed);
            assert_eq!(seed, value);
        }
    }

    #[test]
    fn compute_qn_covers_reference_cases() {
        let cases = [
            ((1, 12, 3, 8, false), 1),
            ((2, 20, 2, 16, true), 1),
            ((4, 64, 1, 24, false), 2),
            ((8, 200, 0, 32, true), 4),
            ((3, 48, 4, 12, false), 2),
        ];

        for ((n, b, offset, pulse_cap, stereo), expected) in cases {
            assert_eq!(
                compute_qn(n, b, offset, pulse_cap, stereo),
                expected,
                "compute_qn({n}, {b}, {offset}, {pulse_cap}, {stereo})",
            );
        }
    }

    #[test]
    fn spreading_returns_aggressive_for_concentrated_energy() {
        let e_bands = [0i16, 16, 32];
        let mode = dummy_mode(&e_bands, 32);
        let channels = 1;
        let m = 1;
        let end = 2;
        let spread_weight = [1, 1];
        let spectrum = vec![1.0f32; channels * m * mode.short_mdct_size];
        let mut average = 0;
        let mut hf_average = 0;
        let mut tapset = 1;

        let decision = spreading_decision(
            &mode,
            &spectrum,
            &mut average,
            SPREAD_NORMAL,
            &mut hf_average,
            &mut tapset,
            false,
            end,
            channels,
            m,
            &spread_weight,
        );

        assert_eq!(decision, SPREAD_AGGRESSIVE);
        assert_eq!(tapset, 1);
        assert_eq!(hf_average, 0);
        assert_eq!(average, 0);
    }

    #[test]
    fn spreading_returns_normal_when_single_threshold_met() {
        let e_bands = [0i16, 16, 32];
        let mode = dummy_mode(&e_bands, 32);
        let channels = 1;
        let m = 1;
        let end = 2;
        let spread_weight = [1, 1];
        let mut spectrum = vec![1.0f32; channels * m * mode.short_mdct_size];
        for idx in 0..8 {
            spectrum[idx] = 0.1;
        }
        let mut average = 0;
        let mut hf_average = 0;
        let mut tapset = 1;

        let decision = spreading_decision(
            &mode,
            &spectrum,
            &mut average,
            SPREAD_NORMAL,
            &mut hf_average,
            &mut tapset,
            false,
            end,
            channels,
            m,
            &spread_weight,
        );

        assert_eq!(decision, SPREAD_NORMAL);
        assert_eq!(tapset, 1);
        assert_eq!(hf_average, 0);
        assert_eq!(average, 64);
    }

    #[test]
    fn spreading_returns_none_when_all_thresholds_met() {
        let e_bands = [0i16, 16, 32];
        let mode = dummy_mode(&e_bands, 32);
        let channels = 1;
        let m = 1;
        let end = 1;
        let spread_weight = [1];
        let mut spectrum = vec![1.0f32; channels * m * mode.short_mdct_size];
        for idx in 0..8 {
            spectrum[idx] = 0.01;
        }
        let mut average = 0;
        let mut hf_average = 0;
        let mut tapset = 1;

        let decision = spreading_decision(
            &mode,
            &spectrum,
            &mut average,
            SPREAD_NONE,
            &mut hf_average,
            &mut tapset,
            false,
            end,
            channels,
            m,
            &spread_weight,
        );

        assert_eq!(decision, SPREAD_NONE);
        assert_eq!(tapset, 1);
        assert_eq!(hf_average, 0);
        assert_eq!(average, 384);
    }

    #[test]
    fn spreading_updates_hf_tracking() {
        let e_bands: Vec<i16> = (0..=10).map(|i| (i * 24) as i16).collect();
        let mode = dummy_mode(&e_bands, 256);
        let channels = 1;
        let m = 1;
        let end = mode.num_ebands;
        let spread_weight = vec![1; end];
        let spectrum = vec![0.0f32; channels * m * mode.short_mdct_size];
        let mut average = 0;
        let mut hf_average = 0;
        let mut tapset = 1;

        let decision = spreading_decision(
            &mode,
            &spectrum,
            &mut average,
            SPREAD_NONE,
            &mut hf_average,
            &mut tapset,
            true,
            end,
            channels,
            m,
            &spread_weight,
        );

        assert_eq!(decision, SPREAD_NONE);
        assert_eq!(tapset, 2);
        assert_eq!(hf_average, 24);
        assert_eq!(average, 384);
    }

    #[test]
    fn stereo_split_matches_reference_transform() {
        let mut mid = [1.0_f32, -1.5, 0.25, 0.0];
        let mut side = [0.5_f32, 2.0, -0.75, 1.25];

        let mut expected_mid = mid;
        let mut expected_side = side;
        for idx in 0..mid.len() {
            let m = core::f32::consts::FRAC_1_SQRT_2 * expected_mid[idx];
            let s = core::f32::consts::FRAC_1_SQRT_2 * expected_side[idx];
            expected_mid[idx] = m + s;
            expected_side[idx] = s - m;
        }

        stereo_split(&mut mid, &mut side);

        for (observed, reference) in mid.iter().zip(expected_mid.iter()) {
            assert!((observed - reference).abs() <= f32::EPSILON * 16.0);
        }
        for (observed, reference) in side.iter().zip(expected_side.iter()) {
            assert!((observed - reference).abs() <= f32::EPSILON * 16.0);
        }
    }

    fn reference_stereo_merge(x: &[f32], y: &[f32], mid: f32) -> (Vec<f32>, Vec<f32>) {
        let mut left = x.to_vec();
        let mut right = y.to_vec();

        let (mut cross, side_energy) = dual_inner_prod(&right, &left, &right);
        cross *= mid;
        let mid_energy = mid * mid;
        let el = mid_energy + side_energy - 2.0 * cross;
        let er = mid_energy + side_energy + 2.0 * cross;

        if er < 6e-4 || el < 6e-4 {
            right.copy_from_slice(&left);
            return (left, right);
        }

        let lgain = celt_rsqrt_norm(el);
        let rgain = celt_rsqrt_norm(er);

        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mid_scaled = mid * *l;
            let side_val = *r;
            *l = lgain * (mid_scaled - side_val);
            *r = rgain * (mid_scaled + side_val);
        }

        (left, right)
    }

    #[test]
    fn stereo_merge_matches_reference_transform() {
        let mut left = [0.8, -0.25, 0.5, -0.75, 0.1, 0.3];
        let mut right = [-0.2, 0.4, -0.6, 0.3, -0.1, 0.2];
        let mid = 0.9;

        let (expected_left, expected_right) = reference_stereo_merge(&left, &right, mid);
        stereo_merge(&mut left, &mut right, mid);

        for (idx, (&value, &expected)) in left.iter().zip(expected_left.iter()).enumerate() {
            assert!(
                (value - expected).abs() <= 1e-6,
                "left[{idx}] mismatch: value={value}, expected={expected}"
            );
        }

        for (idx, (&value, &expected)) in right.iter().zip(expected_right.iter()).enumerate() {
            assert!(
                (value - expected).abs() <= 1e-6,
                "right[{idx}] mismatch: value={value}, expected={expected}"
            );
        }
    }

    #[test]
    fn stereo_merge_copies_mid_for_low_energy() {
        let mut left = [0.0f32; 4];
        let mut right = [1e-3f32, -1e-3, 2e-3, -2e-3];
        stereo_merge(&mut left, &mut right, 0.0);
        assert_eq!(right, left);
    }

    #[test]
    fn special_hybrid_folding_duplicates_primary_when_needed() {
        let e_bands = [0i16, 1, 3, 6];
        let mode = dummy_mode(&e_bands, 8);
        let mut norm = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];

        special_hybrid_folding(&mode, &mut norm, None, 0, 2, false);

        assert_eq!(norm, vec![0.1, 0.2, 0.1, 0.2, 0.5, 0.6]);
    }

    #[test]
    fn special_hybrid_folding_updates_secondary_when_dual_stereo() {
        let e_bands = [0i16, 1, 3, 6];
        let mode = dummy_mode(&e_bands, 8);
        let mut norm = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut norm2 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        special_hybrid_folding(&mode, &mut norm, Some(&mut norm2), 0, 2, true);

        assert_eq!(norm, vec![0.1, 0.2, 0.1, 0.2, 0.5, 0.6]);
        assert_eq!(norm2, vec![1.0, 2.0, 1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn special_hybrid_folding_noops_when_second_band_not_wider() {
        let e_bands = [0i16, 2, 3, 5];
        let mode = dummy_mode(&e_bands, 4);
        let mut norm = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut norm2 = vec![4.0f32, 5.0, 6.0, 7.0];
        let expected_norm = norm.clone();
        let expected_norm2 = norm2.clone();

        special_hybrid_folding(&mode, &mut norm, Some(&mut norm2), 0, 1, true);

        assert_eq!(norm, expected_norm);
        assert_eq!(norm2, expected_norm2);
    }

    fn reference_ordery(stride: usize) -> &'static [usize] {
        match stride {
            2 => &[1, 0],
            4 => &[3, 0, 2, 1],
            8 => &[7, 0, 4, 3, 6, 1, 5, 2],
            16 => &[15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5],
            _ => panic!("unsupported stride"),
        }
    }

    #[test]
    fn hadamard_interleave_matches_reference_layout() {
        for &stride in &[2, 4, 8, 16] {
            let n0 = 4usize;
            let n = n0 * stride;
            let mut data: Vec<f32> = (0..n).map(|v| v as f32).collect();
            let mut expected = data.clone();

            let ordery = reference_ordery(stride);
            for (i, &ord) in ordery.iter().enumerate() {
                for j in 0..n0 {
                    expected[j * stride + i] = data[ord * n0 + j];
                }
            }

            interleave_hadamard(&mut data, n0, stride, true);
            assert_eq!(data[..n], expected[..n]);
        }
    }

    #[test]
    fn interleave_and_deinterleave_round_trip() {
        let cases = [
            (4usize, 2usize, false),
            (4, 2, true),
            (8, 4, false),
            (8, 4, true),
        ];

        for &(n0, stride, hadamard) in &cases {
            let n = n0 * stride;
            let data: Vec<f32> = (0..n).map(|v| (v as f32) * 0.5 - 3.0).collect();
            let mut transformed = data.clone();
            interleave_hadamard(&mut transformed, n0, stride, hadamard);
            deinterleave_hadamard(&mut transformed, n0, stride, hadamard);
            assert_eq!(transformed[..n], data[..n]);
        }
    }

    #[test]
    fn frac_mul16_matches_c_macro() {
        // Compare a handful of values against a direct evaluation of the C
        // macro written in Rust.
        fn reference(a: i32, b: i32) -> i32 {
            let a = a as i16;
            let b = b as i16;
            (16_384 + i32::from(a) * i32::from(b)) >> 15
        }

        let samples = [
            (-32_768, -32_768),
            (-32_768, 32_767),
            (-20_000, 16_000),
            (-626, 8_000),
            (8_277, -5_000),
            (7_932, 2_000),
            (32_767, 32_767),
        ];

        for &(a, b) in &samples {
            assert_eq!(frac_mul16(a, b), reference(a, b));
        }
    }

    #[test]
    fn bitexact_cos_matches_reference_samples() {
        let inputs = [-16_383, -12_000, -6_000, -1, 0, 1, 6_000, 12_000, 16_383];
        let expected = [
            3, 13_371, 27_494, -32_768, -32_768, -32_768, 27_494, 13_371, 3,
        ];

        for (&input, &value) in inputs.iter().zip(expected.iter()) {
            assert_eq!(bitexact_cos(input), value);
        }
    }

    #[test]
    fn bitexact_log2tan_matches_reference_samples() {
        let inputs = [
            (23_170, 32_767),
            (11_585, 32_767),
            (16_384, 23_170),
            (30_000, 12_345),
            (12_345, 30_000),
            (1, 32_767),
            (32_767, 1),
        ];
        let expected = [-1_025, -3_073, -993, 2_631, -2_631, -30_690, 30_690];

        for ((isin, icos), &value) in inputs.iter().zip(expected.iter()) {
            assert_eq!(bitexact_log2tan(*isin, *icos), value);
        }
    }

    fn dummy_mode<'a>(e_bands: &'a [i16], short_mdct_size: usize) -> OpusCustomMode<'a> {
        let mdct = MdctLookup::new(short_mdct_size, 0);
        OpusCustomMode {
            sample_rate: 48_000,
            overlap: 0,
            num_ebands: e_bands.len() - 1,
            effective_ebands: e_bands.len() - 1,
            pre_emphasis: [0.0; 4],
            e_bands,
            max_lm: 0,
            num_short_mdcts: 1,
            short_mdct_size,
            num_alloc_vectors: 0,
            alloc_vectors: &[],
            log_n: &[],
            window: &[],
            mdct,
            cache: PulseCacheData::default(),
        }
    }

    #[test]
    fn quant_band_n1_round_trips_sign_information() {
        let e_bands = [0i16, 1];
        let mode = dummy_mode(&e_bands, 4);
        let bit_budget = (1_i32) << BITRES;

        let mut storage = vec![0u8; 8];
        let mut x = [-0.5_f32];
        let mut y = [0.25_f32];
        let mut lowband = [0.0_f32];

        {
            let mut enc = EcEnc::new(&mut storage);
            let mut ctx = BandCtx {
                encode: true,
                resynth: true,
                mode: &mode,
                band: 0,
                intensity: 0,
                spread: 0,
                tf_change: 0,
                remaining_bits: 2 * bit_budget,
                band_e: &[],
                seed: 0,
                arch: 0,
                theta_round: 0,
                disable_inv: false,
                avoid_split_noise: false,
            };
            {
                let mut coder = BandCodingState::Encoder(&mut enc);
                let coded = quant_band_n1(
                    &mut ctx,
                    &mut x,
                    Some(&mut y),
                    Some(&mut lowband),
                    &mut coder,
                );
                assert_eq!(coded, 1);
            }
            assert_eq!(ctx.remaining_bits, 0);
            assert!((x[0] + NORM_SCALING).abs() <= f32::EPSILON * 2.0);
            assert!((y[0] - NORM_SCALING).abs() <= f32::EPSILON * 2.0);
            assert_eq!(lowband[0], x[0]);
            enc.enc_done();
        }

        let mut decode_buf = storage.clone();
        let mut dec = EcDec::new(&mut decode_buf);
        let mut ctx = BandCtx {
            encode: false,
            resynth: true,
            mode: &mode,
            band: 0,
            intensity: 0,
            spread: 0,
            tf_change: 0,
            remaining_bits: 2 * bit_budget,
            band_e: &[],
            seed: 0,
            arch: 0,
            theta_round: 0,
            disable_inv: false,
            avoid_split_noise: false,
        };
        let mut x_dec = [0.0_f32];
        let mut y_dec = [0.0_f32];
        {
            let mut coder = BandCodingState::Decoder(&mut dec);
            let coded = quant_band_n1(&mut ctx, &mut x_dec, Some(&mut y_dec), None, &mut coder);
            assert_eq!(coded, 1);
        }
        assert_eq!(ctx.remaining_bits, 0);
        assert!((x_dec[0] + NORM_SCALING).abs() <= f32::EPSILON * 2.0);
        assert!((y_dec[0] - NORM_SCALING).abs() <= f32::EPSILON * 2.0);
    }

    #[test]
    fn compute_theta_encode_decode_round_trip() {
        let e_bands = [0i16, 2, 4];
        let log_n = [0i16, 0];
        let mdct = MdctLookup::new(4, 0);
        let mode = OpusCustomMode {
            sample_rate: 48_000,
            overlap: 0,
            num_ebands: e_bands.len() - 1,
            effective_ebands: e_bands.len() - 1,
            pre_emphasis: [0.0; 4],
            e_bands: &e_bands,
            max_lm: 0,
            num_short_mdcts: 1,
            short_mdct_size: 4,
            num_alloc_vectors: 0,
            alloc_vectors: &[],
            log_n: &log_n,
            window: &[],
            mdct,
            cache: PulseCacheData::default(),
        };
        let band_e = vec![0.75_f32, 0.6, 0.65, 0.7];
        let n = 4;
        let initial_b = 48 << BITRES;
        let b_current = 2;
        let b0 = 2;
        let lm = 1;
        let stereo = true;
        let initial_fill = (1u32 << (b0 as u32)) - 1;

        let mut x_encode = vec![0.45_f32, -0.2, 0.05, -0.35];
        let mut y_encode = vec![0.15_f32, 0.32, -0.28, 0.4];
        let x_original = x_encode.clone();
        let y_original = y_encode.clone();

        let mut ctx_encode = BandCtx {
            encode: true,
            resynth: false,
            mode: &mode,
            band: 1,
            intensity: mode.num_ebands + 4,
            spread: SPREAD_NORMAL,
            tf_change: 0,
            remaining_bits: 160 << BITRES,
            band_e: &band_e,
            seed: 0x4567_89ab,
            arch: 0,
            theta_round: 0,
            disable_inv: false,
            avoid_split_noise: false,
        };

        let mut sctx_encode = SplitCtx::default();
        let mut b_encode = initial_b;
        let mut fill_encode = initial_fill;

        let mut storage = vec![0u8; 32];
        {
            let mut enc = EcEnc::new(&mut storage);
            {
                let mut coder = BandCodingState::Encoder(&mut enc);
                compute_theta(
                    &mut ctx_encode,
                    &mut sctx_encode,
                    &mut x_encode,
                    &mut y_encode,
                    n,
                    &mut b_encode,
                    b_current,
                    b0,
                    lm,
                    stereo,
                    &mut fill_encode,
                    &mut coder,
                );
            }
            enc.enc_done();
        }

        let mut x_decode = x_original;
        let mut y_decode = y_original;
        let mut ctx_decode = BandCtx {
            encode: false,
            resynth: false,
            mode: &mode,
            band: 1,
            intensity: mode.num_ebands + 4,
            spread: SPREAD_NORMAL,
            tf_change: 0,
            remaining_bits: 160 << BITRES,
            band_e: &band_e,
            seed: 0x4567_89ab,
            arch: 0,
            theta_round: 0,
            disable_inv: false,
            avoid_split_noise: false,
        };

        let mut sctx_decode = SplitCtx::default();
        let mut b_decode = initial_b;
        let mut fill_decode = initial_fill;

        let mut decode_buf = storage.clone();
        {
            let mut dec = EcDec::new(&mut decode_buf);
            let mut coder = BandCodingState::Decoder(&mut dec);
            compute_theta(
                &mut ctx_decode,
                &mut sctx_decode,
                &mut x_decode,
                &mut y_decode,
                n,
                &mut b_decode,
                b_current,
                b0,
                lm,
                stereo,
                &mut fill_decode,
                &mut coder,
            );
        }

        assert_eq!(sctx_encode, sctx_decode);
        assert_eq!(b_encode, b_decode);
        assert_eq!(fill_encode, fill_decode);
    }

    #[test]
    fn compute_band_energies_matches_manual_sum() {
        let e_bands = [0i16, 2, 4];
        let mode = dummy_mode(&e_bands, 4);
        let channels = 2;
        let lm = 1usize;
        let n = mode.short_mdct_size << lm;

        let mut spectrum = Vec::with_capacity(channels * n);
        for idx in 0..channels * n {
            spectrum.push((idx as f32 * 0.13 - 0.5).sin());
        }

        let mut band_e = vec![0.0; mode.num_ebands * channels];
        compute_band_energies(
            &mode,
            &spectrum,
            &mut band_e,
            mode.num_ebands,
            channels,
            lm,
            0,
        );

        for c in 0..channels {
            for b in 0..mode.num_ebands {
                let start = ((mode.e_bands[b] as usize) << lm) + c * n;
                let stop = ((mode.e_bands[b + 1] as usize) << lm) + c * n;
                let sum: f32 = spectrum[start..stop].iter().map(|v| v * v).sum();
                let expected = (1e-27_f32 + sum).sqrt();
                let idx = b + c * mode.num_ebands;
                assert!(
                    (band_e[idx] - expected).abs() <= 1e-6,
                    "channel {c}, band {b}"
                );
            }
        }
    }

    #[test]
    fn normalise_bands_scales_by_inverse_energy() {
        let e_bands = [0i16, 2, 4];
        let mode = dummy_mode(&e_bands, 4);
        let channels = 1usize;
        let m = 2usize;
        let n = mode.short_mdct_size * m;

        let freq: Vec<CeltSig> = (0..n).map(|i| (i as f32 * 0.21 - 0.4).cos()).collect();
        let mut norm = vec![0.0f32; freq.len()];

        let mut band_e = vec![0.0f32; mode.num_ebands * channels];
        for b in 0..mode.num_ebands {
            let start = m * (mode.e_bands[b] as usize);
            let stop = m * (mode.e_bands[b + 1] as usize);
            let sum: f32 = freq[start..stop].iter().map(|v| v * v).sum();
            band_e[b] = (1e-27_f32 + sum).sqrt();
        }

        normalise_bands(
            &mode,
            &freq,
            &mut norm,
            &band_e,
            mode.num_ebands,
            channels,
            m,
        );

        for b in 0..mode.num_ebands {
            let start = m * (mode.e_bands[b] as usize);
            let stop = m * (mode.e_bands[b + 1] as usize);
            let gain = 1.0 / (1e-27_f32 + band_e[b]);
            for j in start..stop {
                assert!(
                    (norm[j] - freq[j] * gain).abs() <= 1e-6,
                    "band {b}, index {j}"
                );
            }
        }
    }

    #[test]
    fn denormalise_bands_restores_scaled_spectrum() {
        let e_bands = [0i16, 2, 4];
        let mode = dummy_mode(&e_bands, 4);
        let m = 1usize;
        let n = mode.short_mdct_size * m;

        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17 - 0.3).sin()).collect();
        let mut freq = vec![1.0f32; n];
        let band_log_e = vec![0.5f32, -0.25];

        denormalise_bands(
            &mode,
            &x,
            &mut freq,
            &band_log_e,
            0,
            mode.num_ebands,
            m,
            1,
            false,
        );

        let mut expected = vec![0.0f32; n];
        let mut idx = 0usize;
        for band in 0..mode.num_ebands {
            let band_end = m * (mode.e_bands[band + 1] as usize);
            let gain = celt_exp2((band_log_e[band] + E_MEANS[band]).min(32.0));
            while idx < band_end {
                expected[idx] = x[idx] * gain;
                idx += 1;
            }
        }

        for (i, (&observed, &reference)) in freq.iter().zip(expected.iter()).enumerate() {
            assert!(
                (observed - reference).abs() <= 1e-6,
                "index {i}: observed={observed}, expected={reference}"
            );
        }
    }

    #[test]
    fn denormalise_bands_honours_downsample_and_silence() {
        let e_bands = [0i16, 2, 4];
        let mode = dummy_mode(&e_bands, 4);
        let m = 1usize;
        let n = mode.short_mdct_size * m;

        let x = vec![0.5f32, -0.25, 0.125, -0.375];
        let mut freq = vec![0.75f32; n];
        let band_log_e = vec![0.0f32];

        denormalise_bands(&mode, &x, &mut freq, &band_log_e, 0, 1, m, 2, false);

        let gain = celt_exp2((band_log_e[0] + E_MEANS[0]).min(32.0));
        assert!((freq[0] - x[0] * gain).abs() <= 1e-6);
        assert!((freq[1] - x[1] * gain).abs() <= 1e-6);
        assert_eq!(freq[2], 0.0);
        assert_eq!(freq[3], 0.0);

        denormalise_bands(&mode, &x, &mut freq, &band_log_e, 0, 1, m, 1, true);
        assert!(freq.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn anti_collapse_fills_collapsed_band_with_noise() {
        let e_bands = [0i16, 2];
        let mode = dummy_mode(&e_bands, 4);
        let lm = 1usize;
        let channels = 1usize;
        let size = mode.short_mdct_size << lm;
        let mut spectrum = vec![0.0f32; channels * size];
        let collapse_masks = vec![0u8; mode.num_ebands * channels];
        let log_e = vec![5.0f32; mode.num_ebands * channels];
        let prev1 = vec![0.0f32; mode.num_ebands * channels];
        let prev2 = vec![0.0f32; mode.num_ebands * channels];
        let pulses = vec![0i32; mode.num_ebands];

        anti_collapse(
            &mode,
            &mut spectrum,
            &collapse_masks,
            lm,
            channels,
            size,
            0,
            mode.num_ebands,
            &log_e,
            &prev1,
            &prev2,
            &pulses,
            0xDEAD_BEEF,
            false,
            0,
        );

        let band_width = usize::try_from(e_bands[1] - e_bands[0]).unwrap();
        let samples = band_width << lm;
        let energy: f32 = spectrum[..samples].iter().map(|v| v * v).sum();

        assert!(spectrum[..samples].iter().any(|v| *v != 0.0));
        assert!(energy > 0.0);
        assert!((energy - 1.0).abs() <= 1e-3, "renormalised energy {energy}");
    }
}
