//! Helpers from the CELT rate control module.
//!
//! The reference implementation in `celt/rate.c` exposes a number of
//! lightweight helpers that other translation units depend on.  This module
//! begins porting that surface by translating the constant tables and the
//! inline helpers that describe the pseudo-pulse grid.

#![allow(dead_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::{max, min};
use core::convert::TryFrom;

use crate::celt::cwrs::get_required_bits;
use crate::celt::entcode::{BITRES, celt_udiv};
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::celt::types::{OpusCustomMode, OpusInt16, OpusInt32, OpusUint32, PulseCacheData};

/// Maximum pseudo-pulse index described in the C headers.
pub(crate) const MAX_PSEUDO: i32 = 40;
/// Base-2 logarithm of [`MAX_PSEUDO`] used by the search helpers.
pub(crate) const LOG_MAX_PSEUDO: i32 = 6;
/// Maximum pulses tracked by the allocation helpers.
pub(crate) const CELT_MAX_PULSES: usize = 128;
/// Maximum number of fine bits stored per band.
pub(crate) const MAX_FINE_BITS: i32 = 8;
/// Fine energy quantiser offset.
pub(crate) const FINE_OFFSET: i32 = 21;
/// Offset applied to the qtheta bit allocation for the single phase search.
pub(crate) const QTHETA_OFFSET: i32 = 4;
/// Offset applied when performing the two-phase qtheta search.
pub(crate) const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// Fractional log2 look-up table used when reserving intensity bits.
pub(crate) const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

/// Returns the number of pulses represented by the pseudo-pulse index `i`.
///
/// This mirrors the inline helper from `celt/rate.h`.  The first eight entries
/// map one-to-one, after which the sequence doubles every eight indices while
/// repeating the base pattern modulo eight.
pub(crate) fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Determines if `V(N, K)` fits inside an unsigned 32-bit integer.
///
/// In the reference C implementation this guard is only compiled for custom
/// modes.  It precomputes the limits for both `N` and `K` and applies the same
/// branching logic, allowing the port to reuse the pulse cache generation logic
/// without pulling in the full rate control module.
pub(crate) fn fits_in32(n: i32, k: i32) -> bool {
    const MAX_N: [i16; 15] = [
        32767, 32767, 32767, 1476, 283, 109, 60, 40, 29, 24, 20, 18, 16, 14, 13,
    ];
    const MAX_K: [i16; 15] = [
        32767, 32767, 32767, 32767, 1172, 238, 95, 53, 36, 27, 22, 18, 16, 15, 13,
    ];

    if n >= 14 {
        if k >= 14 {
            false
        } else {
            n <= MAX_N[k as usize] as i32
        }
    } else {
        k <= MAX_K[n as usize] as i32
    }
}

/// Recomputes the pulse cache used by custom modes.
///
/// Mirrors `compute_pulse_cache()` from `celt/rate.c`, porting the allocation of
/// the PVQ lookup tables and the per-band bit caps to safe Rust containers. The
/// helper is written to match the original control flow closely so that the
/// results can be compared against the C reference when validating future
/// translations.
#[allow(clippy::too_many_lines)]
pub(crate) fn compute_pulse_cache(
    e_bands: &[OpusInt16],
    log_n: &[OpusInt16],
    lm: usize,
) -> PulseCacheData {
    let nb_ebands = e_bands.len().saturating_sub(1);
    let rows = nb_ebands * (lm + 2);
    let mut index = vec![-1i32; rows];
    let mut entry_n = Vec::new();
    let mut entry_k = Vec::new();
    let mut entry_offset = Vec::new();
    let mut curr = 0i32;

    for i in 0..=(lm + 1) {
        for j in 0..nb_ebands {
            let mut n = i32::from(e_bands[j + 1] - e_bands[j]);
            n = (n << i) >> 1;
            let row = i * nb_ebands + j;
            index[row] = -1;

            for k in 0..=i {
                for n_idx in 0..nb_ebands {
                    if k == i && n_idx >= j {
                        break;
                    }
                    let mut other = i32::from(e_bands[n_idx + 1] - e_bands[n_idx]);
                    other = (other << k) >> 1;
                    if n == other {
                        index[row] = index[k * nb_ebands + n_idx];
                        break;
                    }
                }
                if index[row] != -1 {
                    break;
                }
            }

            if index[row] == -1 && n != 0 {
                let mut k = 0;
                while k < MAX_PSEUDO && fits_in32(n, get_pulses(k + 1)) {
                    k += 1;
                }
                entry_n.push(n);
                entry_k.push(k);
                entry_offset.push(curr);
                index[row] = curr;
                curr += k + 1;
            }
        }
    }

    let mut bits = vec![0u8; curr.max(0) as usize];
    for idx in 0..entry_n.len() {
        let n = entry_n[idx] as usize;
        let k = entry_k[idx] as usize;
        let offset = entry_offset[idx] as usize;
        let mut scratch = vec![0 as OpusInt16; CELT_MAX_PULSES + 1];
        let max_k = get_pulses(entry_k[idx]) as usize;
        get_required_bits(&mut scratch, n, max_k, BITRES as OpusInt32);

        bits[offset] = k as u8;
        for j in 1..=k {
            let pulses = get_pulses(j as i32) as usize;
            let value = scratch[pulses] - 1;
            debug_assert!((0..=u8::MAX as OpusInt16).contains(&value));
            bits[offset + j] = value as u8;
        }
    }

    let mut caps = vec![0u8; (lm + 1) * 2 * nb_ebands];
    let shift = BITRES as usize;
    for i in 0..=lm {
        for c in 1..=2 {
            let c_i32 = c as i32;
            for j in 0..nb_ebands {
                let mut n0 = i32::from(e_bands[j + 1] - e_bands[j]);
                let max_bits = if (n0 << i) == 1 {
                    (c_i32 * (1 + MAX_FINE_BITS)) << shift
                } else {
                    let mut lm0 = 0i32;
                    if n0 > 2 {
                        n0 >>= 1;
                        lm0 -= 1;
                    } else if n0 <= 1 {
                        lm0 = i32::min(i as i32, 1);
                        n0 <<= lm0 as usize;
                    }

                    let row = ((lm0 + 1) as usize) * nb_ebands + j;
                    let cache_offset = index[row];
                    debug_assert!(cache_offset >= 0, "pulse cache entry should exist");
                    let cache_offset = cache_offset as usize;
                    let entry_k = bits[cache_offset] as i32;
                    let base_idx = cache_offset + entry_k as usize;
                    let mut local_bits = i32::from(bits[base_idx]) + 1;
                    let iterations = (i as i32 - lm0).max(0);
                    let mut n = n0;
                    for k_iter in 0..iterations {
                        local_bits <<= 1;
                        let offset = ((i32::from(log_n[j]) + ((lm0 + k_iter) << shift)) >> 1)
                            - QTHETA_OFFSET;
                        let two_n_minus_one = 2 * n - 1;
                        let num = 459 * (two_n_minus_one * offset + local_bits);
                        let den = (two_n_minus_one << 9) - 459;
                        let mut qb = ((num + (den >> 1)) / den).min(57);
                        debug_assert!(qb >= 0);
                        if qb < 0 {
                            qb = 0;
                        }
                        local_bits += qb;
                        let offset_guard = offset.max(0);
                        local_bits = local_bits.max(c_i32 * (offset_guard + (4 << shift)));
                        n <<= 1;
                    }

                    let ndof = (c_i32 * n0 - (c_i32 - 1)).max(1);
                    local_bits = local_bits.min(c_i32 * n0 << shift);
                    let pulses = get_pulses(entry_k) + 2 * (ndof - 1);
                    let floor = c_i32 * ((pulses / (2 * ndof)) << shift);
                    local_bits = local_bits.max(floor);
                    if (n0 << i) == 2 {
                        let extra = ((c_i32 * (1 + 16)) >> 1) << shift;
                        local_bits = local_bits.max(extra);
                    }
                    if c == 2 && (n0 << i) >= 2 {
                        let extra = ((2 * (1 + 24)) >> 1) << shift;
                        local_bits = local_bits.max(extra);
                    }

                    local_bits
                };

                let cap_idx = i * 2 * nb_ebands + (c - 1) * nb_ebands + j;
                if !caps.is_empty() {
                    caps[cap_idx] = ((max_bits >> shift).min(255)) as u8;
                }
            }
        }
    }

    let index = index
        .into_iter()
        .map(|value| i16::try_from(value).expect("pulse cache index exceeds 16-bit range"))
        .collect();

    PulseCacheData::new(index, bits, caps)
}

/// Interpolates between allocation vectors and converts the resulting bit budget
/// to PVQ pulses for each band.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub(crate) fn interp_bits2pulses(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[OpusInt32],
    bits2: &[OpusInt32],
    thresh: &[OpusInt32],
    cap: &[OpusInt32],
    mut total: OpusInt32,
    balance: &mut OpusInt32,
    skip_rsv: OpusInt32,
    intensity: &mut OpusInt32,
    mut intensity_rsv: OpusInt32,
    dual_stereo: &mut OpusInt32,
    dual_stereo_rsv: OpusInt32,
    bits: &mut [OpusInt32],
    ebits: &mut [OpusInt32],
    fine_priority: &mut [OpusInt32],
    channels: OpusInt32,
    lm: OpusInt32,
    mut encoder: Option<&mut EcEnc<'_>>,
    mut decoder: Option<&mut EcDec<'_>>,
    prev: OpusInt32,
    signal_bandwidth: OpusInt32,
) -> OpusInt32 {
    debug_assert!(start <= end);
    debug_assert!(bits.len() >= end);
    debug_assert!(ebits.len() >= end);
    debug_assert!(fine_priority.len() >= end);
    debug_assert!(bits1.len() >= end);
    debug_assert!(bits2.len() >= end);
    debug_assert!(thresh.len() >= end);
    debug_assert!(cap.len() >= end);

    const ALLOC_STEPS: OpusInt32 = 6;

    let alloc_floor = channels << BITRES;
    let stereo_shift = if channels > 1 { 1 } else { 0 };
    let log_m = lm << BITRES;

    let mut lo: OpusInt32 = 0;
    let mut hi: OpusInt32 = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += min(tmp, cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let mut psum = 0;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            if tmp >= alloc_floor {
                tmp = alloc_floor;
            } else {
                tmp = 0;
            }
        } else {
            done = true;
        }
        tmp = min(tmp, cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    let mut coded_bands = end as OpusInt32;
    while coded_bands > start as OpusInt32 {
        let band = coded_bands - 1;
        let j = band as usize;
        let band_start = OpusInt32::from(mode.e_bands[start]);
        let band_end = OpusInt32::from(mode.e_bands[coded_bands as usize]);
        let band_prev = OpusInt32::from(mode.e_bands[j]);
        let band_width = band_end - band_prev;

        if band <= skip_start as OpusInt32 {
            total += skip_rsv;
            break;
        }

        let mut left = total - psum;
        let denom = max(band_end - band_start, 1);
        let per_coeff = celt_udiv(left.max(0) as OpusUint32, denom as OpusUint32) as OpusInt32;
        left -= denom * per_coeff;
        let rem = max(left - (band_prev - band_start), 0);
        let mut band_bits = bits[j] + per_coeff * band_width + rem;
        let thresh_j = max(thresh[j], alloc_floor + (1 << BITRES));

        if band_bits >= thresh_j {
            let mut skip = false;
            if let Some(enc) = encoder.as_deref_mut() {
                let decision = if coded_bands > 17 {
                    let depth_threshold = if j as OpusInt32 <= prev { 7 } else { 9 };
                    let split_shift = (lm + BITRES as OpusInt32) as u32;
                    band_bits > ((depth_threshold * band_width) << split_shift) >> 4
                        && (j as OpusInt32) <= signal_bandwidth
                } else {
                    true
                };
                enc.enc_bit_logp(decision as OpusInt32, 1);
                if decision {
                    skip = true;
                }
            } else if let Some(dec) = decoder.as_deref_mut() {
                if dec.dec_bit_logp(1) != 0 {
                    skip = true;
                }
            }

            if skip {
                break;
            }

            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }

        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = OpusInt32::from(LOG2_FRAC_TABLE[j - start]);
        }
        psum += intensity_rsv;

        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }

        coded_bands -= 1;
    }

    debug_assert!(coded_bands > start as OpusInt32);

    if intensity_rsv > 0 {
        if let Some(enc) = encoder.as_deref_mut() {
            let limit = coded_bands + 1 - start as OpusInt32;
            let clamped = min(*intensity, coded_bands);
            enc.enc_uint((clamped - start as OpusInt32) as OpusUint32, limit as u32);
        } else if let Some(dec) = decoder.as_deref_mut() {
            let limit = coded_bands + 1 - start as OpusInt32;
            let value = dec.dec_uint(limit as u32) as OpusInt32;
            *intensity = start as OpusInt32 + value;
        }
    } else {
        *intensity = 0;
    }

    if *intensity <= start as OpusInt32 {
        total += dual_stereo_rsv;
    }

    if dual_stereo_rsv > 0 {
        if let Some(enc) = encoder.as_deref_mut() {
            enc.enc_bit_logp(*dual_stereo, 1);
        } else if let Some(dec) = decoder.as_deref_mut() {
            *dual_stereo = dec.dec_bit_logp(1);
        }
    } else {
        *dual_stereo = 0;
    }

    let denom = max(
        OpusInt32::from(mode.e_bands[coded_bands as usize]) - OpusInt32::from(mode.e_bands[start]),
        1,
    );
    let mut left = total - psum;
    let per_coeff = celt_udiv(left.max(0) as OpusUint32, denom as OpusUint32) as OpusInt32;
    left -= denom * per_coeff;
    for j in start..coded_bands as usize {
        let width = OpusInt32::from(mode.e_bands[j + 1] - mode.e_bands[j]);
        bits[j] += per_coeff * width;
    }
    for j in start..coded_bands as usize {
        let width = OpusInt32::from(mode.e_bands[j + 1] - mode.e_bands[j]);
        let add = min(width, left);
        bits[j] += add;
        left -= add;
    }

    let mut local_balance = 0;
    for j in start..coded_bands as usize {
        let n0 = OpusInt32::from(mode.e_bands[j + 1] - mode.e_bands[j]);
        let n = n0 << lm;
        let bit = bits[j] + local_balance;

        if n > 1 {
            let excess = max(bit - cap[j], 0);
            bits[j] = bit - excess;

            let mut den = channels * n;
            if channels == 2 && n > 2 && *dual_stereo == 0 && (j as OpusInt32) < *intensity {
                den += 1;
            }
            let nclogn = den * (OpusInt32::from(mode.log_n[j]) + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << (BITRES - 2);
            }
            if bits[j] + offset < den * 2 << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < den * 3 << BITRES {
                offset += nclogn >> 3;
            }

            let mut eb = max(0, bits[j] + offset + (den << (BITRES - 1)));
            eb = (celt_udiv(eb as OpusUint32, den as OpusUint32) as OpusInt32) >> BITRES;
            if channels * eb > (bits[j] >> stereo_shift) >> BITRES {
                eb = bits[j] >> stereo_shift >> BITRES;
            }
            eb = min(eb, MAX_FINE_BITS);
            fine_priority[j] = if eb * (den << BITRES) >= bits[j] + offset {
                1
            } else {
                0
            };
            bits[j] -= channels * eb << BITRES;
            ebits[j] = eb;

            if excess > 0 {
                let extra_fine = min(excess >> (stereo_shift + BITRES), MAX_FINE_BITS - ebits[j]);
                ebits[j] += extra_fine;
                let extra_bits = extra_fine * channels << BITRES;
                if extra_bits >= excess - local_balance {
                    fine_priority[j] = 1;
                }
                local_balance = excess - extra_bits;
            } else {
                local_balance = excess;
            }
        } else {
            let excess = max(0, bit - (channels << BITRES));
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
            local_balance = excess;
        }

        debug_assert!(bits[j] >= 0);
        debug_assert!(ebits[j] >= 0);
    }

    *balance = local_balance;
    coded_bands
}

/// Computes the full band allocation curve for the supplied mode and bit budget.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub(crate) fn clt_compute_allocation(
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    offsets: &[OpusInt32],
    cap: &[OpusInt32],
    alloc_trim: OpusInt32,
    intensity: &mut OpusInt32,
    dual_stereo: &mut OpusInt32,
    mut total: OpusInt32,
    balance: &mut OpusInt32,
    pulses: &mut [OpusInt32],
    ebits: &mut [OpusInt32],
    fine_priority: &mut [OpusInt32],
    channels: OpusInt32,
    lm: OpusInt32,
    encoder: Option<&mut EcEnc<'_>>,
    decoder: Option<&mut EcDec<'_>>,
    prev: OpusInt32,
    signal_bandwidth: OpusInt32,
) -> OpusInt32 {
    debug_assert!(offsets.len() >= end);
    debug_assert!(cap.len() >= end);
    debug_assert!(pulses.len() >= end);
    debug_assert!(ebits.len() >= end);
    debug_assert!(fine_priority.len() >= end);

    total = max(total, 0);
    let len = mode.num_ebands;
    let mut skip_start = start;

    let mut skip_rsv = 0;
    if total >= 1 << BITRES {
        skip_rsv = 1 << BITRES;
        total -= skip_rsv;
    }

    let mut intensity_rsv = 0;
    let mut dual_stereo_rsv = 0;
    if channels == 2 {
        let candidate = OpusInt32::from(LOG2_FRAC_TABLE[end - start]);
        if candidate <= total {
            intensity_rsv = candidate;
            total -= intensity_rsv;
            if total >= 1 << BITRES {
                dual_stereo_rsv = 1 << BITRES;
                total -= dual_stereo_rsv;
            }
        }
    }

    let mut bits1 = vec![0; len];
    let mut bits2 = vec![0; len];
    let mut thresh = vec![0; len];
    let mut trim_offset = vec![0; len];

    for j in start..end {
        let n = OpusInt32::from(mode.e_bands[j + 1] - mode.e_bands[j]);
        let alloc_shift = (lm + BITRES as OpusInt32) as u32;
        thresh[j] = max(channels << BITRES, (3 * n) << alloc_shift >> 4);
        let split_shift = (lm + BITRES as OpusInt32) as u32;
        trim_offset[j] = channels
            * n
            * (alloc_trim - 5 - lm)
            * OpusInt32::try_from(end - j - 1).unwrap()
            * (1 << split_shift)
            >> 6;
        if (n << lm) == 1 {
            trim_offset[j] -= channels << BITRES;
        }
    }

    let mut lo: OpusInt32 = 1;
    let mut hi: OpusInt32 = mode.num_alloc_vectors as OpusInt32 - 1;
    while lo <= hi {
        let mid = (lo + hi) >> 1;
        let mut done = false;
        let mut psum = 0;
        for j in (start..end).rev() {
            let n = OpusInt32::from(mode.e_bands[j + 1] - mode.e_bands[j]);
            let mut bitsj =
                channels * n * OpusInt32::from(mode.alloc_vectors[mid as usize * len + j]) << lm
                    >> 2;
            if bitsj > 0 {
                bitsj = max(0, bitsj + trim_offset[j]);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += min(bitsj, cap[j]);
            } else if bitsj >= channels << BITRES {
                psum += channels << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    hi = lo;
    lo -= 1;

    for j in start..end {
        let n = OpusInt32::from(mode.e_bands[j + 1] - mode.e_bands[j]);
        let mut bits1j =
            channels * n * OpusInt32::from(mode.alloc_vectors[lo as usize * len + j]) << lm >> 2;
        let mut bits2j = if hi as usize >= mode.num_alloc_vectors {
            cap[j]
        } else {
            channels * n * OpusInt32::from(mode.alloc_vectors[hi as usize * len + j]) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = max(0, bits1j + trim_offset[j]);
        }
        if bits2j > 0 {
            bits2j = max(0, bits2j + trim_offset[j]);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = max(0, bits2j - bits1j);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    interp_bits2pulses(
        mode,
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        balance,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        channels,
        lm,
        encoder,
        decoder,
        prev,
        signal_bandwidth,
    )
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;

    use super::{
        LOG2_FRAC_TABLE, clt_compute_allocation, compute_pulse_cache, fits_in32, get_pulses,
        interp_bits2pulses,
    };
    use crate::celt::entcode::BITRES;
    use crate::celt::entdec::EcDec;
    use crate::celt::entenc::EcEnc;
    use crate::celt::types::{MdctLookup, OpusCustomMode, OpusInt32, PulseCacheData};

    #[test]
    fn get_pulses_matches_reference_pattern() {
        let expected = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30,
        ];
        for (i, &value) in expected.iter().enumerate() {
            assert_eq!(get_pulses(i as i32), value);
        }

        // Spot-check the first few entries of the next doubling interval.
        assert_eq!(get_pulses(24), 32);
        assert_eq!(get_pulses(31), 60);
    }

    #[test]
    fn fits_in32_replicates_thresholds() {
        // For n < 14 the max K threshold is provided by MAX_K.
        assert!(fits_in32(13, 15));
        assert!(!fits_in32(13, 16));

        // For n >= 14 the logic flips to checking max N.
        assert!(fits_in32(14, 13));
        assert!(!fits_in32(14, 14));

        // Boundaries around the large "always fits" region.
        assert!(fits_in32(0, 32767));
        assert!(fits_in32(1, 32767));
        assert!(fits_in32(2, 32767));

        // Large values that violate the final MAX_N entry should fail.
        assert!(!fits_in32(15, 13));
    }

    #[test]
    fn compute_pulse_cache_assigns_shared_offsets() {
        let e_bands = [0i16, 2, 6];
        let log_n = [6i16, 7];
        let lm = 1usize;
        let cache = compute_pulse_cache(&e_bands, &log_n, lm);

        let nb_ebands = e_bands.len() - 1;
        assert_eq!(cache.size, cache.bits.len());
        assert_eq!(cache.index.len(), nb_ebands * (lm + 2));
        assert_eq!(cache.caps.len(), (lm + 1) * 2 * nb_ebands);

        let mut seen = BTreeMap::new();
        for i in 0..=(lm + 1) {
            for j in 0..nb_ebands {
                let n = i32::from(e_bands[j + 1] - e_bands[j]);
                let n = (n << i) >> 1;
                if n == 0 {
                    continue;
                }
                let offset = cache.index[i * nb_ebands + j];
                if let Some(&expected) = seen.get(&n) {
                    assert_eq!(offset, expected);
                } else {
                    seen.insert(n, offset);
                    let offset = offset as usize;
                    let k = cache.bits[offset] as usize;
                    assert!(offset + k < cache.bits.len());
                }
            }
        }
    }

    fn simple_mode<'a>(
        e_bands: &'a [i16],
        alloc_vectors: &'a [u8],
        log_n: &'a [i16],
        cache: PulseCacheData,
    ) -> OpusCustomMode<'a> {
        OpusCustomMode {
            sample_rate: 48_000,
            overlap: 0,
            num_ebands: e_bands.len(),
            effective_ebands: e_bands.len(),
            pre_emphasis: [0.0; 4],
            e_bands,
            max_lm: 2,
            num_short_mdcts: 0,
            short_mdct_size: 0,
            num_alloc_vectors: alloc_vectors.len() / e_bands.len(),
            alloc_vectors,
            log_n,
            window: &[],
            mdct: MdctLookup::new(0, 0, &[]),
            cache,
        }
    }

    #[test]
    fn interp_bits2pulses_matches_encode_decode() {
        let e_bands = [0i16, 2, 4];
        let log_n = [7i16, 8];
        let alloc_vectors = [6u8, 7, 9, 10];
        let cache = compute_pulse_cache(&e_bands, &log_n, 1);
        let mode = simple_mode(&e_bands, &alloc_vectors, &log_n, cache);

        let cap = vec![1 << (BITRES + 6); mode.num_ebands];
        let bits1 = vec![20 << BITRES; mode.num_ebands];
        let bits2 = vec![5 << BITRES; mode.num_ebands];
        let thresh = vec![8 << BITRES; mode.num_ebands];
        let mut bits_encode = vec![0; mode.num_ebands];
        let mut bits_decode = vec![0; mode.num_ebands];
        let mut ebits_encode = vec![0; mode.num_ebands];
        let mut ebits_decode = vec![0; mode.num_ebands];
        let mut fine_encode = vec![0; mode.num_ebands];
        let mut fine_decode = vec![0; mode.num_ebands];
        let mut balance_encode = 0;
        let mut balance_decode = 0;
        let mut intensity_encode = 0;
        let mut intensity_decode = 0;
        let mut dual_stereo_encode = 0;
        let mut dual_stereo_decode = 0;
        let total = 120 << BITRES;

        let mut buffer = vec![0u8; 64];
        {
            let mut enc = EcEnc::new(&mut buffer);
            interp_bits2pulses(
                &mode,
                0,
                2,
                0,
                &bits1,
                &bits2,
                &thresh,
                &cap,
                total,
                &mut balance_encode,
                1 << BITRES,
                &mut intensity_encode,
                OpusInt32::from(LOG2_FRAC_TABLE[2]),
                &mut dual_stereo_encode,
                1 << BITRES,
                &mut bits_encode,
                &mut ebits_encode,
                &mut fine_encode,
                1,
                1,
                Some(&mut enc),
                None,
                0,
                2,
            );
            enc.enc_done();
        }

        let mut decode_buf = buffer.clone();
        {
            let mut dec = EcDec::new(&mut decode_buf);
            interp_bits2pulses(
                &mode,
                0,
                2,
                0,
                &bits1,
                &bits2,
                &thresh,
                &cap,
                total,
                &mut balance_decode,
                1 << BITRES,
                &mut intensity_decode,
                OpusInt32::from(LOG2_FRAC_TABLE[2]),
                &mut dual_stereo_decode,
                1 << BITRES,
                &mut bits_decode,
                &mut ebits_decode,
                &mut fine_decode,
                1,
                1,
                None,
                Some(&mut dec),
                0,
                2,
            );
        }

        assert_eq!(bits_encode, bits_decode);
        assert_eq!(ebits_encode, ebits_decode);
        assert_eq!(fine_encode, fine_decode);
        assert_eq!(balance_encode, balance_decode);
        assert_eq!(intensity_encode, intensity_decode);
        assert_eq!(dual_stereo_encode, dual_stereo_decode);
    }

    #[test]
    fn clt_compute_allocation_round_trip() {
        let e_bands = [0i16, 2, 4];
        let log_n = [7i16, 8];
        let alloc_vectors = [6u8, 8, 9, 11];
        let cache = compute_pulse_cache(&e_bands, &log_n, 1);
        let mode = simple_mode(&e_bands, &alloc_vectors, &log_n, cache);

        let offsets = vec![0; mode.num_ebands];
        let cap = vec![1 << (BITRES + 6); mode.num_ebands];
        let total = 140 << BITRES;

        let mut pulses_encode = vec![0; mode.num_ebands];
        let mut pulses_decode = vec![0; mode.num_ebands];
        let mut ebits_encode = vec![0; mode.num_ebands];
        let mut ebits_decode = vec![0; mode.num_ebands];
        let mut fine_encode = vec![0; mode.num_ebands];
        let mut fine_decode = vec![0; mode.num_ebands];
        let mut balance_encode = 0;
        let mut balance_decode = 0;
        let mut intensity_encode = 0;
        let mut intensity_decode = 0;
        let mut dual_stereo_encode = 0;
        let mut dual_stereo_decode = 0;

        let coded_bands_encode;
        let mut buffer = vec![0u8; 64];
        {
            let mut enc = EcEnc::new(&mut buffer);
            coded_bands_encode = clt_compute_allocation(
                &mode,
                0,
                2,
                &offsets,
                &cap,
                5,
                &mut intensity_encode,
                &mut dual_stereo_encode,
                total,
                &mut balance_encode,
                &mut pulses_encode,
                &mut ebits_encode,
                &mut fine_encode,
                1,
                1,
                Some(&mut enc),
                None,
                0,
                2,
            );
            enc.enc_done();
        }

        let mut decode_buf = buffer.clone();
        {
            let mut dec = EcDec::new(&mut decode_buf);
            let coded_bands_decode = clt_compute_allocation(
                &mode,
                0,
                2,
                &offsets,
                &cap,
                5,
                &mut intensity_decode,
                &mut dual_stereo_decode,
                total,
                &mut balance_decode,
                &mut pulses_decode,
                &mut ebits_decode,
                &mut fine_decode,
                1,
                1,
                None,
                Some(&mut dec),
                0,
                2,
            );
            assert_eq!(coded_bands_decode, coded_bands_encode);
        }

        assert_eq!(pulses_encode, pulses_decode);
        assert_eq!(ebits_encode, ebits_decode);
        assert_eq!(fine_encode, fine_decode);
        assert_eq!(balance_encode, balance_decode);
        assert_eq!(intensity_encode, intensity_decode);
        assert_eq!(dual_stereo_encode, dual_stereo_decode);
    }
}
