//! Port of the floating-point pitch analysis core.
//!
//! Mirrors `silk_pitch_analysis_core_FLP` from `silk/float/pitch_analysis_core_FLP.c`,
//! running the three-stage correlation search in the 4/8 kHz domains before
//! refining the lags at the native sample rate.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::explicit_counter_loop,
    clippy::indexing_slicing,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unnecessary_cast
)]

use crate::celt::celt_pitch_xcorr;
use crate::silk::energy_flp::energy;
use crate::silk::inner_product_flp::inner_product_flp;
use crate::silk::pitch_est_tables::{
    PE_D_SRCH_LENGTH, PE_FLATCONTOUR_BIAS, PE_LTP_MEM_LENGTH_MS, PE_MAX_FS_KHZ, PE_MAX_LAG,
    PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2, PE_NB_CBKS_STAGE2_10_MS,
    PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10_MS, PE_NB_CBKS_STAGE3_MAX, PE_NB_STAGE3_LAGS,
    PE_PREVLAG_BIAS, PE_SHORTLAG_BIAS, PE_SUBFR_LENGTH_MS, SILK_CB_LAGS_STAGE2,
    SILK_CB_LAGS_STAGE2_10_MS, SILK_CB_LAGS_STAGE3, SILK_CB_LAGS_STAGE3_10_MS,
    SILK_LAG_RANGE_STAGE3, SILK_LAG_RANGE_STAGE3_10_MS, SILK_NB_CBK_SEARCHS_STAGE3,
    SILK_PE_MAX_COMPLEX, SILK_PE_MIN_COMPLEX,
};
use crate::silk::resampler_down2::resampler_down2;
use crate::silk::resampler_down2_3::resampler_down2_3;
use crate::silk::sigproc_flp::{
    silk_float2int, silk_float2short_array, silk_log2, silk_short2float_array,
};
use crate::silk::sort::insertion_sort_decreasing_f32;
use core::cmp::{max, min};

const SCRATCH_SIZE: usize = 22;
const MAX_FRAME_LENGTH: usize =
    (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * PE_MAX_FS_KHZ;
const MAX_FRAME_LENGTH_8_KHZ: usize =
    (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * 8;
const MAX_FRAME_LENGTH_4_KHZ: usize =
    (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * 4;
const MIN_LAG_4_KHZ: usize = PE_MIN_LAG_MS * 4;
const MAX_LAG_4_KHZ: usize = PE_MAX_LAG_MS * 4;
const CSTRIDE: usize = (PE_MAX_LAG >> 1) + 5;

/// Floating-point pitch analysis. Returns `0` for voiced frames and `1` for unvoiced.
#[allow(clippy::too_many_lines)]
pub fn pitch_analysis_core_flp(
    frame: &[f32],
    pitch_out: &mut [i32],
    lag_index: &mut i16,
    contour_index: &mut i8,
    ltp_corr: &mut f32,
    mut prev_lag: i32,
    search_thres1: f32,
    search_thres2: f32,
    fs_khz: i32,
    complexity: i32,
    nb_subfr: i32,
    arch: i32,
) -> i32 {
    assert!(matches!(fs_khz, 8 | 12 | 16), "unsupported sampling rate");
    assert!(
        (SILK_PE_MIN_COMPLEX as i32..=SILK_PE_MAX_COMPLEX as i32).contains(&complexity),
        "invalid complexity"
    );
    assert!(
        (0.0..=1.0).contains(&search_thres1) && (0.0..=1.0).contains(&search_thres2),
        "search thresholds must be in [0, 1]"
    );

    let nb_subfr_usize = nb_subfr as usize;
    assert!(
        nb_subfr_usize > 0 && nb_subfr_usize <= PE_MAX_NB_SUBFR,
        "invalid subframe count"
    );
    assert!(
        pitch_out.len() >= nb_subfr_usize,
        "pitch_out must hold {nb_subfr} entries"
    );

    let frame_length_ms = PE_LTP_MEM_LENGTH_MS + nb_subfr_usize * PE_SUBFR_LENGTH_MS;
    let frame_length = frame_length_ms * fs_khz as usize;
    let frame_length_8khz = frame_length_ms * 8;
    let frame_length_4khz = frame_length_ms * 4;
    let sf_length = PE_SUBFR_LENGTH_MS * fs_khz as usize;
    let sf_length_8khz = PE_SUBFR_LENGTH_MS * 8;
    let sf_length_4khz = PE_SUBFR_LENGTH_MS * 4;
    let min_lag = PE_MIN_LAG_MS * fs_khz as usize;
    let min_lag_8khz = PE_MIN_LAG_MS * 8;
    let max_lag = PE_MAX_LAG_MS * fs_khz as usize - 1;
    let max_lag_8khz = PE_MAX_LAG_MS * 8 - 1;

    assert!(
        frame.len() >= frame_length,
        "frame shorter than required input"
    );

    // Downsample to 8 kHz.
    let mut frame_8_fix = [0i16; MAX_FRAME_LENGTH];
    let mut frame_8khz = [0f32; MAX_FRAME_LENGTH_8_KHZ];
    let mut filter_state = [0i32; 6];
    match fs_khz {
        16 => {
            let mut frame_16_fix = [0i16; MAX_FRAME_LENGTH];
            silk_float2short_array(&mut frame_16_fix[..frame_length], &frame[..frame_length]);
            let state: &mut [i32; 2] = (&mut filter_state[..2]).try_into().unwrap();
            resampler_down2(
                state,
                &mut frame_8_fix[..frame_length_8khz],
                &frame_16_fix[..frame_length],
            );
            silk_short2float_array(
                &mut frame_8khz[..frame_length_8khz],
                &frame_8_fix[..frame_length_8khz],
            );
        }
        12 => {
            let mut frame_12_fix = [0i16; MAX_FRAME_LENGTH];
            silk_float2short_array(&mut frame_12_fix[..frame_length], &frame[..frame_length]);
            let produced = resampler_down2_3(
                &mut filter_state,
                &mut frame_8_fix[..frame_length_8khz],
                &frame_12_fix[..frame_length],
            );
            debug_assert_eq!(produced, frame_length_8khz);
            silk_short2float_array(
                &mut frame_8khz[..frame_length_8khz],
                &frame_8_fix[..frame_length_8khz],
            );
        }
        _ => {
            // Fs_khz == 8, no resampling needed for the 8 kHz path.
            silk_float2short_array(
                &mut frame_8_fix[..frame_length_8khz],
                &frame[..frame_length],
            );
        }
    }

    // Downsample again to 4 kHz.
    let mut frame_4_fix = [0i16; MAX_FRAME_LENGTH_4_KHZ];
    let mut frame_4khz = [0f32; MAX_FRAME_LENGTH_4_KHZ];
    let state: &mut [i32; 2] = (&mut filter_state[..2]).try_into().unwrap();
    resampler_down2(
        state,
        &mut frame_4_fix[..frame_length_4khz],
        &frame_8_fix[..frame_length_8khz],
    );
    silk_short2float_array(
        &mut frame_4khz[..frame_length_4khz],
        &frame_4_fix[..frame_length_4khz],
    );

    // Low-pass filter.
    for i in (1..frame_length_4khz).rev() {
        frame_4khz[i] = add_sat16_as_float(frame_4khz[i], frame_4khz[i - 1]);
    }

    // Stage 1: coarse correlation search at 4 kHz.
    let mut c = [[0f32; CSTRIDE]; PE_MAX_NB_SUBFR];
    let mut d_srch = [0i32; PE_D_SRCH_LENGTH];
    let mut d_comp = [0i16; CSTRIDE];
    let mut xcorr = [0f32; MAX_LAG_4_KHZ - MIN_LAG_4_KHZ + 1];

    let mut target_idx = sf_length_4khz << 2;
    for _k in 0..(nb_subfr_usize >> 1) {
        let target = &frame_4khz[target_idx..target_idx + sf_length_8khz];
        let basis_start = target_idx
            .checked_sub(MAX_LAG_4_KHZ)
            .expect("basis_start underflow");
        let basis_end = basis_start + sf_length_8khz + (MAX_LAG_4_KHZ - MIN_LAG_4_KHZ);
        assert!(basis_end <= frame_length_4khz);

        celt_pitch_xcorr(
            target,
            &frame_4khz[basis_start..],
            sf_length_8khz,
            xcorr.len(),
            &mut xcorr,
        );

        let mut basis_idx = target_idx - MIN_LAG_4_KHZ;
        let mut cross_corr = f64::from(xcorr[MAX_LAG_4_KHZ - MIN_LAG_4_KHZ]);
        let mut normalizer = energy(target)
            + energy(&frame_4khz[basis_idx..basis_idx + sf_length_8khz])
            + f64::from(sf_length_8khz as i32 * 4000);
        c[0][MIN_LAG_4_KHZ] += (2.0 * cross_corr / normalizer) as f32;

        for d in (MIN_LAG_4_KHZ + 1)..=MAX_LAG_4_KHZ {
            basis_idx -= 1;
            cross_corr = f64::from(xcorr[MAX_LAG_4_KHZ - d]);
            let head = f64::from(frame_4khz[basis_idx]);
            let tail = f64::from(frame_4khz[basis_idx + sf_length_8khz]);
            normalizer += head * head - tail * tail;
            c[0][d] += (2.0 * cross_corr / normalizer) as f32;
        }

        target_idx += sf_length_8khz;
    }

    for i in MIN_LAG_4_KHZ..=MAX_LAG_4_KHZ {
        c[0][i] -= c[0][i] * (i as f32) / 4096.0;
    }

    let mut length_d_srch = 4 + 2 * complexity as usize;
    insertion_sort_decreasing_f32(
        &mut c[0][MIN_LAG_4_KHZ..=MAX_LAG_4_KHZ],
        &mut d_srch,
        length_d_srch,
    );

    let cmax = c[0][MIN_LAG_4_KHZ];
    if cmax < 0.2 {
        pitch_out[..nb_subfr_usize].fill(0);
        *ltp_corr = 0.0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    let threshold = search_thres1 * cmax;
    for i in 0..length_d_srch {
        let slot = MIN_LAG_4_KHZ + i;
        if c[0][slot] > threshold {
            d_srch[i] = ((d_srch[i] + MIN_LAG_4_KHZ as i32) << 1) as i32;
        } else {
            length_d_srch = i;
            break;
        }
    }
    assert!(length_d_srch > 0);

    for i in (min_lag_8khz.saturating_sub(5))..=max_lag_8khz + 5 {
        d_comp[i] = 0;
    }
    for i in 0..length_d_srch {
        let idx = d_srch[i] as usize;
        d_comp[idx] = 1;
    }

    for i in (min_lag_8khz + 3)..=max_lag_8khz + 3 {
        d_comp[i] = d_comp[i]
            .saturating_add(d_comp[i - 1])
            .saturating_add(d_comp[i - 2]);
    }

    length_d_srch = 0;
    for i in min_lag_8khz..=max_lag_8khz {
        if d_comp[i + 1] > 0 {
            if length_d_srch >= PE_D_SRCH_LENGTH {
                break;
            }
            d_srch[length_d_srch] = i as i32;
            length_d_srch += 1;
        }
    }

    for i in (min_lag_8khz + 3)..=max_lag_8khz + 3 {
        d_comp[i] = d_comp[i]
            .saturating_add(d_comp[i - 1])
            .saturating_add(d_comp[i - 2])
            .saturating_add(d_comp[i - 3]);
    }

    let mut length_d_comp = 0;
    for i in min_lag_8khz..=max_lag_8khz + 3 {
        if d_comp[i] > 0 {
            d_comp[length_d_comp] = (i as i16 - 2) as i16;
            length_d_comp += 1;
        }
    }

    // Stage 2: refined search at 8 kHz.
    c.iter_mut().for_each(|row| row.fill(0.0));
    let mut cc = [0f32; PE_NB_CBKS_STAGE2_EXT];

    let signal_8khz: &[f32] = if fs_khz == 8 {
        frame
    } else {
        &frame_8khz[..frame_length_8khz]
    };

    let mut target_idx = PE_LTP_MEM_LENGTH_MS * 8;
    for k in 0..nb_subfr_usize {
        let target = &signal_8khz[target_idx..target_idx + sf_length_8khz];
        let energy_tmp = energy(target) + 1.0;
        for j in 0..length_d_comp {
            let d = d_comp[j] as usize;
            let basis_idx = target_idx.checked_sub(d).expect("basis underflow");
            let basis = &signal_8khz[basis_idx..basis_idx + sf_length_8khz];
            let cross_corr = inner_product_flp(basis, target);
            c[k][d] = if cross_corr > 0.0 {
                let energy_basis = energy(basis);
                (2.0 * cross_corr / (energy_basis + energy_tmp)) as f32
            } else {
                0.0
            };
        }
        target_idx += sf_length_8khz;
    }

    let mut ccmax = 0.0f32;
    let mut ccmax_b = -1000.0f32;
    let mut cbimax = 0usize;
    let mut lag = -1i32;

    let prev_lag_log2 = if prev_lag > 0 {
        if fs_khz == 12 {
            prev_lag = (prev_lag << 1) / 3;
        } else if fs_khz == 16 {
            prev_lag >>= 1;
        }
        silk_log2(f64::from(prev_lag))
    } else {
        0.0
    };

    let use_stage2_10ms = nb_subfr_usize != PE_MAX_NB_SUBFR;
    let nb_cbk_search = if use_stage2_10ms {
        PE_NB_CBKS_STAGE2_10_MS
    } else if fs_khz == 8 && complexity > SILK_PE_MIN_COMPLEX as i32 {
        PE_NB_CBKS_STAGE2_EXT
    } else {
        PE_NB_CBKS_STAGE2
    };

    for k in 0..length_d_srch {
        let d = d_srch[k] as usize;
        for j in 0..nb_cbk_search {
            cc[j] = 0.0;
            for i in 0..nb_subfr_usize {
                let code = if use_stage2_10ms {
                    SILK_CB_LAGS_STAGE2_10_MS[i][j]
                } else {
                    SILK_CB_LAGS_STAGE2[i][j]
                };
                let idx = d as i32 + i32::from(code);
                assert!(
                    idx >= 0 && (idx as usize) < CSTRIDE,
                    "lag index {} out of range (stride {})",
                    idx,
                    CSTRIDE
                );
                let idx = idx as usize;
                assert!(
                    idx < CSTRIDE,
                    "lag index {idx} out of range (stride {CSTRIDE})"
                );
                cc[j] += c[i][idx];
            }
        }

        let mut ccmax_new = -1000.0f32;
        let mut cbimax_new = 0usize;
        for i in 0..nb_cbk_search {
            if cc[i] > ccmax_new {
                ccmax_new = cc[i];
                cbimax_new = i;
            }
        }

        let lag_log2 = silk_log2(d as f64);
        let mut ccmax_new_b = ccmax_new - PE_SHORTLAG_BIAS * nb_subfr as f32 * lag_log2;
        if prev_lag > 0 {
            let mut delta = lag_log2 - prev_lag_log2;
            delta *= delta;
            ccmax_new_b -= PE_PREVLAG_BIAS * nb_subfr as f32 * (*ltp_corr) * delta / (delta + 0.5);
        }

        if ccmax_new_b > ccmax_b && ccmax_new > nb_subfr as f32 * search_thres2 {
            ccmax_b = ccmax_new_b;
            ccmax = ccmax_new;
            lag = d as i32;
            cbimax = cbimax_new;
        }
    }

    if lag == -1 {
        pitch_out[..nb_subfr_usize].fill(0);
        *ltp_corr = 0.0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    *ltp_corr = ccmax / nb_subfr as f32;
    assert!(*ltp_corr >= 0.0);

    if fs_khz > 8 {
        if fs_khz == 12 {
            lag = ((lag * 3) + 1) >> 1;
        } else {
            lag <<= 1;
        }
        lag = limit_int(lag, min_lag as i32, max_lag as i32);
        let start_lag = max(lag - 2, min_lag as i32);
        let end_lag = min(lag + 2, max_lag as i32);
        let mut lag_new = lag;
        cbimax = 0;
        ccmax = -1000.0;

        let mut cross_corr_st3 =
            [[[0f32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR];
        let mut energies_st3 =
            [[[0f32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR];

        p_ana_calc_corr_st3_flp(
            &mut cross_corr_st3,
            frame,
            start_lag,
            sf_length,
            nb_subfr_usize,
            complexity as usize,
            arch,
        );
        p_ana_calc_energy_st3_flp(
            &mut energies_st3,
            frame,
            start_lag,
            sf_length,
            nb_subfr_usize,
            complexity as usize,
        );

        let mut lag_counter = 0;
        let contour_bias = PE_FLATCONTOUR_BIAS / lag as f32;
        let use_full_cb = nb_subfr_usize == PE_MAX_NB_SUBFR;
        let nb_cbk_search = if use_full_cb {
            SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize] as usize
        } else {
            PE_NB_CBKS_STAGE3_10_MS
        };

        let target_idx = PE_LTP_MEM_LENGTH_MS * fs_khz as usize;
        let energy_tmp = energy(&frame[target_idx..target_idx + nb_subfr_usize * sf_length]) + 1.0;

        for d in start_lag..=end_lag {
            for j in 0..nb_cbk_search {
                let mut cross_corr = 0.0f64;
                let mut energy = energy_tmp;
                for k in 0..nb_subfr_usize {
                    cross_corr += f64::from(cross_corr_st3[k][j][lag_counter]);
                    energy += f64::from(energies_st3[k][j][lag_counter]);
                }

                let mut ccmax_new = if cross_corr > 0.0 {
                    let mut value = (2.0 * cross_corr / energy) as f32;
                    value *= 1.0 - contour_bias * j as f32;
                    value
                } else {
                    0.0
                };

                let cb_delta = if use_full_cb {
                    i32::from(SILK_CB_LAGS_STAGE3[0][j])
                } else {
                    i32::from(SILK_CB_LAGS_STAGE3_10_MS[0][j])
                };
                if d + cb_delta > max_lag as i32 {
                    ccmax_new = 0.0;
                }

                if ccmax_new > ccmax {
                    ccmax = ccmax_new;
                    lag_new = d;
                    cbimax = j;
                }
            }
            lag_counter += 1;
        }

        for k in 0..nb_subfr_usize {
            let delta = if use_full_cb {
                SILK_CB_LAGS_STAGE3[k][cbimax]
            } else {
                SILK_CB_LAGS_STAGE3_10_MS[k][cbimax]
            };
            pitch_out[k] = lag_new + i32::from(delta);
            pitch_out[k] = limit_int(pitch_out[k], min_lag as i32, PE_MAX_LAG_MS as i32 * fs_khz);
        }
        *lag_index = (lag_new - min_lag as i32) as i16;
        *contour_index = cbimax as i8;
    } else {
        for k in 0..nb_subfr_usize {
            let delta = if use_stage2_10ms {
                SILK_CB_LAGS_STAGE2_10_MS[k][cbimax]
            } else {
                SILK_CB_LAGS_STAGE2[k][cbimax]
            };
            pitch_out[k] = lag + i32::from(delta);
            pitch_out[k] = limit_int(pitch_out[k], min_lag_8khz as i32, PE_MAX_LAG_MS as i32 * 8);
        }
        *lag_index = (lag - min_lag_8khz as i32) as i16;
        *contour_index = cbimax as i8;
    }

    assert!(*lag_index >= 0);
    0
}

fn p_ana_calc_corr_st3_flp(
    cross_corr_st3: &mut [[[f32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR],
    frame: &[f32],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    arch: i32,
) {
    assert!((SILK_PE_MIN_COMPLEX..=SILK_PE_MAX_COMPLEX).contains(&complexity));

    let use_full_cb = nb_subfr == PE_MAX_NB_SUBFR;
    let nb_cbk_search = if use_full_cb {
        SILK_NB_CBK_SEARCHS_STAGE3[complexity] as usize
    } else {
        PE_NB_CBKS_STAGE3_10_MS
    };

    let mut target_idx = sf_length << 2;
    for k in 0..nb_subfr {
        let (lag_low, lag_high) = if use_full_cb {
            (
                i32::from(SILK_LAG_RANGE_STAGE3[complexity][k][0]),
                i32::from(SILK_LAG_RANGE_STAGE3[complexity][k][1]),
            )
        } else {
            (
                i32::from(SILK_LAG_RANGE_STAGE3_10_MS[k][0]),
                i32::from(SILK_LAG_RANGE_STAGE3_10_MS[k][1]),
            )
        };
        let xcorr_len = (lag_high - lag_low + 1) as usize;
        let mut xcorr = [0f32; SCRATCH_SIZE];

        celt_pitch_xcorr(
            &frame[target_idx..target_idx + sf_length],
            &frame[(target_idx as i32 - start_lag - lag_high) as usize..],
            sf_length,
            xcorr_len,
            &mut xcorr[..xcorr_len],
        );

        let mut scratch_mem = [0f32; SCRATCH_SIZE];
        for (lag_counter, slot) in (lag_low..=lag_high).rev().enumerate() {
            scratch_mem[lag_counter] = xcorr[(lag_high - slot) as usize];
        }

        let delta = lag_low;
        for i in 0..nb_cbk_search {
            let code = if use_full_cb {
                SILK_CB_LAGS_STAGE3[k][i]
            } else {
                SILK_CB_LAGS_STAGE3_10_MS[k][i]
            };
            let idx = i32::from(code) - delta;
            for j in 0..PE_NB_STAGE3_LAGS {
                cross_corr_st3[k][i][j] = scratch_mem[(idx + j as i32) as usize];
            }
        }
        target_idx += sf_length;
    }
    let _ = arch;
}

fn p_ana_calc_energy_st3_flp(
    energies_st3: &mut [[[f32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR],
    frame: &[f32],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
) {
    assert!((SILK_PE_MIN_COMPLEX..=SILK_PE_MAX_COMPLEX).contains(&complexity));

    let use_full_cb = nb_subfr == PE_MAX_NB_SUBFR;
    let nb_cbk_search = if use_full_cb {
        SILK_NB_CBK_SEARCHS_STAGE3[complexity] as usize
    } else {
        PE_NB_CBKS_STAGE3_10_MS
    };

    let mut target_idx = sf_length << 2;
    for k in 0..nb_subfr {
        let (lag_low, lag_high) = if use_full_cb {
            (
                i32::from(SILK_LAG_RANGE_STAGE3[complexity][k][0]),
                i32::from(SILK_LAG_RANGE_STAGE3[complexity][k][1]),
            )
        } else {
            (
                i32::from(SILK_LAG_RANGE_STAGE3_10_MS[k][0]),
                i32::from(SILK_LAG_RANGE_STAGE3_10_MS[k][1]),
            )
        };
        let mut scratch_mem = [0f64; SCRATCH_SIZE];

        let basis_idx = (target_idx as i32 - (start_lag + lag_low)) as usize;
        let mut e = energy(&frame[basis_idx..basis_idx + sf_length]);
        scratch_mem[0] = e;

        let lag_diff = (lag_high - lag_low + 1) as usize;
        for i in 1..lag_diff {
            let remove = basis_idx + sf_length - i;
            let add = basis_idx - i;
            e -= f64::from(frame[remove]) * f64::from(frame[remove]);
            e += f64::from(frame[add]) * f64::from(frame[add]);
            scratch_mem[i] = e;
        }

        let delta = lag_low;
        for i in 0..nb_cbk_search {
            let code = if use_full_cb {
                SILK_CB_LAGS_STAGE3[k][i]
            } else {
                SILK_CB_LAGS_STAGE3_10_MS[k][i]
            };
            let idx = i32::from(code) - delta;
            for j in 0..PE_NB_STAGE3_LAGS {
                energies_st3[k][i][j] = scratch_mem[(idx + j as i32) as usize] as f32;
            }
        }
        target_idx += sf_length;
    }
}

fn add_sat16_as_float(a: f32, b: f32) -> f32 {
    let sum = silk_float2int(a).saturating_add(silk_float2int(b));
    sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as f32
}

fn limit_int(value: i32, min_v: i32, max_v: i32) -> i32 {
    value.clamp(min_v, max_v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::pitch_analysis_core::pitch_analysis_core;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn returns_unvoiced_for_silence() {
        let fs_khz = 8;
        let nb_subfr = 4;
        let frame_length = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz;
        let frame = vec![0.0f32; frame_length];

        let mut pitch_out = [0i32; PE_MAX_NB_SUBFR];
        let mut lag_index = -1i16;
        let mut contour_index = -1i8;
        let mut ltp_corr = -1.0f32;

        let result = pitch_analysis_core_flp(
            &frame,
            &mut pitch_out[..nb_subfr],
            &mut lag_index,
            &mut contour_index,
            &mut ltp_corr,
            0,
            0.3,
            0.3,
            fs_khz as i32,
            1,
            nb_subfr as i32,
            0,
        );

        assert_eq!(result, 1);
        assert!(pitch_out[..nb_subfr].iter().all(|&p| p == 0));
        assert_eq!(lag_index, 0);
        assert_eq!(contour_index, 0);
        assert_eq!(ltp_corr, 0.0);
    }

    #[test]
    fn matches_fixed_point_on_periodic_input() {
        let fs_khz = 8;
        let nb_subfr = 4;
        let frame_length = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz;
        let sample_rate = (fs_khz * 1000) as f32;
        let freq_hz = 200.0f32;

        let mut frame = vec![0.0f32; frame_length];
        for (i, sample) in frame.iter_mut().enumerate() {
            let phase = 2.0 * core::f32::consts::PI * freq_hz * (i as f32) / sample_rate;
            *sample = phase.sin() * 5000.0;
        }

        let mut pitch_out_flp = [0i32; PE_MAX_NB_SUBFR];
        let mut lag_index_flp = -1i16;
        let mut contour_index_flp = -1i8;
        let mut ltp_corr_flp = 0.0f32;

        let flp_res = pitch_analysis_core_flp(
            &frame,
            &mut pitch_out_flp[..nb_subfr],
            &mut lag_index_flp,
            &mut contour_index_flp,
            &mut ltp_corr_flp,
            0,
            0.3,
            0.3,
            fs_khz as i32,
            1,
            nb_subfr as i32,
            0,
        );

        let frame_fixed: Vec<i16> = frame.iter().map(|s| *s as i16).collect();
        let mut pitch_out_fix = [0i32; PE_MAX_NB_SUBFR];
        let mut lag_index_fix = -1i16;
        let mut contour_index_fix = -1i8;
        let mut ltp_corr_q15 = 0;

        let fix_res = pitch_analysis_core(
            &frame_fixed,
            &mut pitch_out_fix[..nb_subfr],
            &mut lag_index_fix,
            &mut contour_index_fix,
            &mut ltp_corr_q15,
            0,
            (0.3 * 65536.0) as i32,
            (0.3 * 8192.0) as i32,
            fs_khz as i32,
            1,
            nb_subfr as i32,
            0,
        );

        assert_eq!(flp_res, fix_res);
        assert_eq!(flp_res, 0);

        for idx in 0..nb_subfr {
            assert!(
                (pitch_out_flp[idx] - pitch_out_fix[idx]).abs() <= 1,
                "lag mismatch: flp={} fix={}",
                pitch_out_flp[idx],
                pitch_out_fix[idx]
            );
        }

        let ltp_corr_fix = ltp_corr_q15 as f32 / 32768.0;
        assert!(
            (ltp_corr_flp - ltp_corr_fix).abs() < 0.05,
            "ltp corr mismatch: flp={} fix={}",
            ltp_corr_flp,
            ltp_corr_fix
        );
        assert_eq!(contour_index_flp, contour_index_fix);
        assert_eq!(lag_index_flp, lag_index_fix);
    }
}
