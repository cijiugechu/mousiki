//! Port of the fixed-point pitch analysis routine.
//!
//! Mirrors `silk_pitch_analysis_core` from `silk/fixed/pitch_analysis_core_FIX.c`,
//! including the three-stage correlation search, bias terms, and stage-three
//! refinement helpers.

use crate::silk::inner_prod_aligned::inner_prod_aligned;
use crate::silk::lin2log::lin2log;
use crate::silk::pitch_est_tables::{
    PE_D_SRCH_LENGTH, PE_LTP_MEM_LENGTH_MS, PE_MAX_FS_KHZ, PE_MAX_LAG_MS, PE_MAX_NB_SUBFR,
    PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2, PE_NB_CBKS_STAGE2_10_MS, PE_NB_CBKS_STAGE2_EXT,
    PE_NB_CBKS_STAGE3_10_MS, PE_NB_CBKS_STAGE3_MAX, PE_NB_STAGE3_LAGS, PE_PREVLAG_BIAS,
    PE_SHORTLAG_BIAS, PE_SUBFR_LENGTH_MS, SILK_CB_LAGS_STAGE2, SILK_CB_LAGS_STAGE2_10_MS,
    SILK_CB_LAGS_STAGE3, SILK_CB_LAGS_STAGE3_10_MS, SILK_LAG_RANGE_STAGE3,
    SILK_LAG_RANGE_STAGE3_10_MS, SILK_NB_CBK_SEARCHS_STAGE3,
};
use crate::silk::resampler_down2::resampler_down2;
use crate::silk::resampler_down2_3::resampler_down2_3;
use crate::silk::sort::insertion_sort_decreasing_int16;
use crate::silk::stereo_find_predictor::div32_varq;
use crate::silk::sum_sqr_shift::sum_sqr_shift;

const SCRATCH_SIZE: usize = 22;
const SF_LENGTH_4_KHZ: usize = PE_SUBFR_LENGTH_MS * 4;
const SF_LENGTH_8_KHZ: usize = PE_SUBFR_LENGTH_MS * 8;
const MIN_LAG_4_KHZ: usize = 2 * 4;
const MAX_LAG_4_KHZ: usize = PE_MAX_LAG_MS * 4;
const MIN_LAG_8_KHZ: usize = 2 * 8;
const MAX_LAG_8_KHZ: usize = PE_MAX_LAG_MS * 8 - 1;
const CSTRIDE_4_KHZ: usize = MAX_LAG_4_KHZ + 1 - MIN_LAG_4_KHZ;
const CSTRIDE_8_KHZ: usize = MAX_LAG_8_KHZ + 3 - (MIN_LAG_8_KHZ - 2);
const D_COMP_STRIDE: usize = (MAX_LAG_8_KHZ + 4) - (MIN_LAG_8_KHZ - 3);
const D_COMP_MIN: i32 = MIN_LAG_8_KHZ as i32 - 3;
const D_COMP_MAX: i32 = MAX_LAG_8_KHZ as i32 + 4;
const CMAX_UNVOICED_Q14: i32 = (0.2 * (1 << 14) as f32 + 0.5) as i32;
const MAX_FRAME_LENGTH: usize =
    (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * PE_MAX_FS_KHZ;
const MAX_FRAME_LENGTH_8_KHZ: usize =
    (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * 8;
const MAX_FRAME_LENGTH_4_KHZ: usize =
    (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * 4;

const PE_SHORTLAG_BIAS_Q13: i32 = (PE_SHORTLAG_BIAS * (1 << 13) as f32) as i32;
const PE_PREVLAG_BIAS_Q13: i32 = (PE_PREVLAG_BIAS * (1 << 13) as f32) as i32;
const PE_FLATCONTOUR_BIAS_Q15: i32 = (0.05 * (1 << 15) as f32) as i32;

/// Fixed-point core pitch analysis. Returns `0` for voiced frames and `1` for unvoiced.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
pub fn pitch_analysis_core(
    frame_unscaled: &[i16],
    pitch_out: &mut [i32],
    lag_index: &mut i16,
    contour_index: &mut i8,
    ltp_corr_q15: &mut i32,
    prev_lag: i32,
    search_thres1_q16: i32,
    search_thres2_q13: i32,
    fs_khz: i32,
    complexity: i32,
    nb_subfr: i32,
    arch: i32,
) -> i32 {
    assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);
    assert!((0..=2).contains(&complexity));

    let nb_subfr_usize = nb_subfr as usize;
    assert!(
        pitch_out.len() >= nb_subfr_usize,
        "pitch_out must hold {nb_subfr} lags"
    );

    let frame_length_ms = PE_LTP_MEM_LENGTH_MS as i32 + nb_subfr * PE_SUBFR_LENGTH_MS as i32;
    let frame_length = frame_length_ms * fs_khz;
    let frame_length_8khz = frame_length_ms * 8;
    let frame_length_4khz = frame_length_ms * 4;
    let sf_length = PE_SUBFR_LENGTH_MS as i32 * fs_khz;
    let min_lag = PE_MIN_LAG_MS as i32 * fs_khz;
    let max_lag = PE_MAX_LAG_MS as i32 * fs_khz - 1;

    let frame_length_usize = frame_length as usize;
    let frame_length_8khz_usize = frame_length_8khz as usize;
    let frame_length_4khz_usize = frame_length_4khz as usize;

    assert!(
        frame_unscaled.len() >= frame_length_usize,
        "input does not contain enough samples"
    );

    // Rescale to guarantee a few guard bits before running the correlation.
    let (energy, mut shift) = sum_sqr_shift(&frame_unscaled[..frame_length_usize]);
    shift += 3 - clz32(energy);

    let mut frame_scaled = [0i16; MAX_FRAME_LENGTH];
    let frame = if shift > 0 {
        let shift = rshift(shift + 1, 1);
        for i in 0..frame_length_usize {
            frame_scaled[i] = rshift_signed(frame_unscaled[i], shift);
        }
        &frame_scaled[..frame_length_usize]
    } else {
        &frame_unscaled[..frame_length_usize]
    };

    // Downsample to 8 kHz.
    let mut frame_8khz_buf = [0i16; MAX_FRAME_LENGTH_8_KHZ];
    let mut filter_state = [0i32; 6];
    let frame_8khz = match fs_khz {
        16 => {
            let state: &mut [i32; 2] = (&mut filter_state[..2]).try_into().unwrap();
            resampler_down2(state, &mut frame_8khz_buf[..frame_length_8khz_usize], frame);
            &frame_8khz_buf[..frame_length_8khz_usize]
        }
        12 => {
            let produced = resampler_down2_3(
                &mut filter_state,
                &mut frame_8khz_buf[..frame_length_8khz_usize],
                frame,
            );
            debug_assert_eq!(produced, frame_length_8khz_usize);
            &frame_8khz_buf[..produced]
        }
        _ => frame,
    };

    // Second downsampling stage to 4 kHz, plus a simple low-pass.
    filter_state[..2].fill(0);
    let mut frame_4khz = [0i16; MAX_FRAME_LENGTH_4_KHZ];
    let state: &mut [i32; 2] = (&mut filter_state[..2]).try_into().unwrap();
    resampler_down2(
        state,
        &mut frame_4khz[..frame_length_4khz_usize],
        frame_8khz,
    );
    for i in (1..frame_length_4khz_usize).rev() {
        let acc = i32::from(frame_4khz[i]) + i32::from(frame_4khz[i - 1]);
        frame_4khz[i] = sat16(acc);
    }

    // Stage 1: coarse correlation at 4 kHz.
    let mut c = [0i16; PE_MAX_NB_SUBFR * CSTRIDE_8_KHZ];
    let mut d_srch = [0i32; PE_D_SRCH_LENGTH];
    let mut d_comp = [0i16; D_COMP_STRIDE];
    let mut xcorr32 = [0i32; MAX_LAG_4_KHZ - MIN_LAG_4_KHZ + 1];

    let half_subfr = nb_subfr_usize >> 1;
    let mut target_idx = SF_LENGTH_4_KHZ << 2;
    for k in 0..half_subfr {
        let target = &frame_4khz[target_idx..target_idx + SF_LENGTH_8_KHZ];
        let basis_start = target_idx - MAX_LAG_4_KHZ;
        let basis = &frame_4khz[basis_start..basis_start + SF_LENGTH_8_KHZ + MAX_LAG_4_KHZ];

        pitch_xcorr(
            target,
            basis,
            SF_LENGTH_8_KHZ,
            MAX_LAG_4_KHZ - MIN_LAG_4_KHZ + 1,
            &mut xcorr32,
        );

        let cross_corr = xcorr32[MAX_LAG_4_KHZ - MIN_LAG_4_KHZ];
        let basis_energy = &frame_4khz[basis_start..basis_start + SF_LENGTH_8_KHZ];
        let mut normalizer = inner_prod_aligned(target, target, arch);
        normalizer = normalizer
            .wrapping_add(inner_prod_aligned(basis_energy, basis_energy, arch))
            .wrapping_add(SF_LENGTH_8_KHZ as i32 * 4000);

        c[k * CSTRIDE_4_KHZ] = div32_varq(cross_corr, normalizer, 14) as i16;

        let mut norm = normalizer;
        for d in (MIN_LAG_4_KHZ + 1)..=MAX_LAG_4_KHZ {
            let offset = MAX_LAG_4_KHZ - d;
            let basis_ptr = target_idx - d;
            norm = norm
                .wrapping_add(
                    i32::from(frame_4khz[basis_ptr]).wrapping_mul(i32::from(frame_4khz[basis_ptr])),
                )
                .wrapping_sub(
                    i32::from(frame_4khz[basis_ptr + SF_LENGTH_8_KHZ])
                        .wrapping_mul(i32::from(frame_4khz[basis_ptr + SF_LENGTH_8_KHZ])),
                );

            c[k * CSTRIDE_4_KHZ + d - MIN_LAG_4_KHZ] = div32_varq(xcorr32[offset], norm, 14) as i16;
        }

        target_idx += SF_LENGTH_8_KHZ;
    }

    let (combined, tail) = c.split_at_mut(CSTRIDE_4_KHZ);
    if nb_subfr_usize == PE_MAX_NB_SUBFR {
        for i in (MIN_LAG_4_KHZ..=MAX_LAG_4_KHZ).rev() {
            let mut sum =
                i32::from(combined[i - MIN_LAG_4_KHZ]) + i32::from(tail[i - MIN_LAG_4_KHZ]);
            sum = smlawb(sum, sum, -((i as i32) << 4));
            combined[i - MIN_LAG_4_KHZ] = sum as i16;
        }
    } else {
        for i in (MIN_LAG_4_KHZ..=MAX_LAG_4_KHZ).rev() {
            let mut sum = i32::from(combined[i - MIN_LAG_4_KHZ]) << 1;
            sum = smlawb(sum, sum, -((i as i32) << 4));
            combined[i - MIN_LAG_4_KHZ] = sum as i16;
        }
    }

    let mut length_d_srch = add_lshift32(4, complexity, 1) as usize;
    insertion_sort_decreasing_int16(combined, &mut d_srch, length_d_srch.min(combined.len()));

    let cmax = i32::from(combined[0]);
    if cmax < CMAX_UNVOICED_Q14 {
        pitch_out[..nb_subfr_usize].fill(0);
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    let threshold = smulwb(search_thres1_q16, cmax);
    for i in 0..length_d_srch {
        if i32::from(combined[i]) > threshold {
            d_srch[i] = (d_srch[i] + MIN_LAG_4_KHZ as i32) << 1;
        } else {
            length_d_srch = i;
            break;
        }
    }
    assert!(length_d_srch > 0);

    d_comp.fill(0);
    for &idx in d_srch.iter().take(length_d_srch) {
        let offset = (idx - D_COMP_MIN) as usize;
        d_comp[offset] = 1;
    }

    for i in (MIN_LAG_8_KHZ as i32..=MAX_LAG_8_KHZ as i32).rev() {
        let base = (i + 1 - D_COMP_MIN) as usize;
        d_comp[base] = d_comp[base] + d_comp[base - 1] + d_comp[base - 2];
    }

    length_d_srch = 0;
    for i in MIN_LAG_8_KHZ as i32..=MAX_LAG_8_KHZ as i32 {
        if d_comp[(i + 1 - D_COMP_MIN) as usize] > 0 {
            d_srch[length_d_srch] = i;
            length_d_srch += 1;
        }
    }

    let mut length_d_comp = 0;
    for i in (MIN_LAG_8_KHZ as i32..=MAX_LAG_8_KHZ as i32 + 3).rev() {
        let base = (i - D_COMP_MIN) as usize;
        d_comp[base] = d_comp[base] + d_comp[base - 1] + d_comp[base - 2] + d_comp[base - 3];
    }

    for i in MIN_LAG_8_KHZ as i32..D_COMP_MAX {
        if d_comp[(i - D_COMP_MIN) as usize] > 0 {
            d_comp[length_d_comp] = (i - 2) as i16;
            length_d_comp += 1;
        }
    }

    // Stage 2: refine at 8 kHz.
    c.fill(0);

    let mut target_ptr = PE_SUBFR_LENGTH_MS * 8 * 4;
    for k in 0..nb_subfr_usize {
        let energy_target = inner_prod_aligned(
            &frame_8khz[target_ptr..target_ptr + SF_LENGTH_8_KHZ],
            &frame_8khz[target_ptr..target_ptr + SF_LENGTH_8_KHZ],
            arch,
        ) + 1;
        for &d in d_comp.iter().take(length_d_comp) {
            let basis_ptr = target_ptr - d as usize;
            let cross_corr = inner_prod_aligned(
                &frame_8khz[target_ptr..target_ptr + SF_LENGTH_8_KHZ],
                &frame_8khz[basis_ptr..basis_ptr + SF_LENGTH_8_KHZ],
                arch,
            );
            if cross_corr > 0 {
                let energy_basis = inner_prod_aligned(
                    &frame_8khz[basis_ptr..basis_ptr + SF_LENGTH_8_KHZ],
                    &frame_8khz[basis_ptr..basis_ptr + SF_LENGTH_8_KHZ],
                    arch,
                );
                let idx = k * CSTRIDE_8_KHZ + (d as usize - (MIN_LAG_8_KHZ - 2));
                c[idx] =
                    div32_varq(cross_corr, energy_target.wrapping_add(energy_basis), 14) as i16;
            }
        }
        target_ptr += SF_LENGTH_8_KHZ;
    }

    let mut ccmax = i32::MIN;
    let mut ccmax_b = i32::MIN;
    let mut cbimax = 0;
    let mut lag = -1;
    let mut prev_lag_log2_q7 = 0;
    if prev_lag > 0 {
        prev_lag_log2_q7 = lin2log(prev_lag);
    }

    let (_cbk_size, nb_cbk_search) = if nb_subfr_usize == PE_MAX_NB_SUBFR {
        let search = if fs_khz == 8 && complexity > 0 {
            PE_NB_CBKS_STAGE2_EXT
        } else {
            PE_NB_CBKS_STAGE2
        };
        (PE_NB_CBKS_STAGE2_EXT, search)
    } else {
        (PE_NB_CBKS_STAGE2_10_MS, PE_NB_CBKS_STAGE2_10_MS)
    };

    let mut cc = [0i32; PE_NB_CBKS_STAGE2_EXT];
    for &d in d_srch.iter().take(length_d_srch) {
        for j in 0..nb_cbk_search {
            let mut acc = 0;
            for i in 0..nb_subfr_usize {
                let delta = if nb_subfr_usize == PE_MAX_NB_SUBFR {
                    i32::from(SILK_CB_LAGS_STAGE2[i][j])
                } else {
                    i32::from(SILK_CB_LAGS_STAGE2_10_MS[i][j])
                };
                let idx = i * CSTRIDE_8_KHZ + (d + delta - (MIN_LAG_8_KHZ as i32 - 2)) as usize;
                acc += i32::from(c[idx]);
            }
            cc[j] = acc;
        }

        let (ccmax_new, cbimax_new) = cc
            .iter()
            .take(nb_cbk_search)
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(idx, &v)| (v, idx))
            .unwrap();

        let lag_log2_q7 = lin2log(d);
        let mut ccmax_new_b = ccmax_new
            - rshift(
                smulbb(nb_subfr * PE_SHORTLAG_BIAS_Q13, lag_log2_q7),
                7,
            );

        if prev_lag > 0 {
            let mut delta_q7 = lag_log2_q7 - prev_lag_log2_q7;
            delta_q7 = rshift(smulbb(delta_q7, delta_q7), 7);
            let mut prev_bias_q13 = smulwb(nb_subfr * PE_PREVLAG_BIAS_Q13, *ltp_corr_q15);
            prev_bias_q13 = div32(prev_bias_q13 * delta_q7, delta_q7 + (1 << 6));
            ccmax_new_b -= prev_bias_q13;
        }

        let valid_lag = if nb_subfr_usize == PE_MAX_NB_SUBFR {
            i32::from(SILK_CB_LAGS_STAGE2[0][cbimax_new]) <= MIN_LAG_8_KHZ as i32
        } else {
            i32::from(SILK_CB_LAGS_STAGE2_10_MS[0][cbimax_new]) <= MIN_LAG_8_KHZ as i32
        };
        if ccmax_new_b > ccmax_b && ccmax_new > nb_subfr * search_thres2_q13 && valid_lag {
            ccmax_b = ccmax_new_b;
            ccmax = ccmax_new;
            lag = d;
            cbimax = cbimax_new;
        }
    }

    if lag == -1 {
        pitch_out[..nb_subfr_usize].fill(0);
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    *ltp_corr_q15 = (div32_16(ccmax, nb_subfr) << 2).clamp(0, i32::MAX);

    let use_full_cb = nb_subfr_usize == PE_MAX_NB_SUBFR;
    let nb_cbk_search = if use_full_cb {
        SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize] as usize
    } else {
        PE_NB_CBKS_STAGE3_10_MS
    };

    if fs_khz > 8 {
        let mut lag_adjusted = if fs_khz == 12 {
            div32_16(lag << 1, 3)
        } else if fs_khz == 16 {
            lag << 1
        } else {
            lag * 3
        };
        lag_adjusted = limit_int(lag_adjusted, min_lag, max_lag);

        let start_lag = (lag_adjusted - 2).max(min_lag);
        let end_lag = (lag_adjusted + 2).min(max_lag);
        let mut lag_new = lag_adjusted;
        let mut cbimax_stage3 = 0;
        ccmax = i32::MIN;

        for k in 0..nb_subfr_usize {
            let delta = if use_full_cb {
                i32::from(SILK_CB_LAGS_STAGE3[k][cbimax])
            } else {
                i32::from(SILK_CB_LAGS_STAGE3_10_MS[k][cbimax])
            };
            pitch_out[k] = lag_adjusted + 2 * delta;
        }

        let mut energies_st3 = [[0i32; PE_NB_STAGE3_LAGS]; PE_MAX_NB_SUBFR * PE_NB_CBKS_STAGE3_MAX];
        let mut cross_corr_st3 =
            [[0i32; PE_NB_STAGE3_LAGS]; PE_MAX_NB_SUBFR * PE_NB_CBKS_STAGE3_MAX];
        p_ana_calc_corr_st3(
            &mut cross_corr_st3,
            frame,
            start_lag,
            sf_length as usize,
            nb_subfr_usize,
            complexity as usize,
            arch,
        );
        p_ana_calc_energy_st3(
            &mut energies_st3,
            frame,
            start_lag,
            sf_length as usize,
            nb_subfr_usize,
            complexity as usize,
            arch,
        );

        let contour_bias_q15 = div32_16(PE_FLATCONTOUR_BIAS_Q15, lag_adjusted);
        let target_ptr = PE_LTP_MEM_LENGTH_MS * fs_khz as usize;
        let energy_target = inner_prod_aligned(
            &frame[target_ptr..target_ptr + nb_subfr_usize * sf_length as usize],
            &frame[target_ptr..target_ptr + nb_subfr_usize * sf_length as usize],
            arch,
        ) + 1;

        for (lag_counter, d) in (start_lag..=end_lag).enumerate() {
            for j in 0..nb_cbk_search {
                let mut cross_corr = 0;
                let mut energy = energy_target;
                for k in 0..nb_subfr_usize {
                    cross_corr += cross_corr_st3[k * nb_cbk_search + j][lag_counter];
                    energy += energies_st3[k * nb_cbk_search + j][lag_counter];
                }

                let ccmax_new = if cross_corr > 0 {
                    let ratio = div32_varq(cross_corr, energy, 14);
                    let diff = i32::from(i16::MAX) - smulbb(contour_bias_q15, j as i32);
                    smulwb(ratio, diff)
                } else {
                    0
                };

                let lag_delta = if use_full_cb {
                    i32::from(SILK_CB_LAGS_STAGE3[0][j])
                } else {
                    i32::from(SILK_CB_LAGS_STAGE3_10_MS[0][j])
                };
                if ccmax_new > ccmax && (d + lag_delta) <= max_lag {
                    ccmax = ccmax_new;
                    lag_new = d;
                    cbimax_stage3 = j;
                }
            }
        }

        for k in 0..nb_subfr_usize {
            let delta = if use_full_cb {
                i32::from(SILK_CB_LAGS_STAGE3[k][cbimax_stage3])
            } else {
                i32::from(SILK_CB_LAGS_STAGE3_10_MS[k][cbimax_stage3])
            };
            pitch_out[k] = lag_new + delta;
            pitch_out[k] = limit_int(pitch_out[k], min_lag, PE_MAX_LAG_MS as i32 * fs_khz);
        }
        *lag_index = (lag_new - min_lag) as i16;
        *contour_index = cbimax_stage3 as i8;
    } else {
        let lag_usize = lag as usize;
        for k in 0..nb_subfr_usize {
            let delta = if use_full_cb {
                i32::from(SILK_CB_LAGS_STAGE3[k][cbimax])
            } else {
                i32::from(SILK_CB_LAGS_STAGE3_10_MS[k][cbimax])
            };
            pitch_out[k] = lag + delta;
            pitch_out[k] = limit_int(pitch_out[k], MIN_LAG_8_KHZ as i32, PE_MAX_LAG_MS as i32 * 8);
        }
        *lag_index = (lag_usize - MIN_LAG_8_KHZ) as i16;
        *contour_index = cbimax as i8;
    }

    0
}

fn p_ana_calc_corr_st3(
    cross_corr_st3: &mut [[i32; PE_NB_STAGE3_LAGS]],
    frame: &[i16],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    arch: i32,
) {
    let use_full_cb = nb_subfr == PE_MAX_NB_SUBFR;
    let lag_range: &[[i8; 2]] = if use_full_cb {
        &SILK_LAG_RANGE_STAGE3[complexity]
    } else {
        &SILK_LAG_RANGE_STAGE3_10_MS
    };
    let nb_cbk_search = if use_full_cb {
        SILK_NB_CBK_SEARCHS_STAGE3[complexity] as usize
    } else {
        PE_NB_CBKS_STAGE3_10_MS
    };

    let mut scratch_mem = [0i32; SCRATCH_SIZE];
    let mut xcorr32 = [0i32; SCRATCH_SIZE];
    let mut target_idx = sf_length << 2;
    for k in 0..nb_subfr {
        let lag_low = i32::from(lag_range[k][0]);
        let lag_high = i32::from(lag_range[k][1]);
        pitch_xcorr(
            &frame[target_idx..target_idx + sf_length],
            &frame[(target_idx as i32 - start_lag - lag_high) as usize..],
            sf_length,
            (lag_high - lag_low + 1) as usize,
            &mut xcorr32,
        );

        for (lag_counter, slot) in (lag_low..=lag_high).rev().enumerate() {
            scratch_mem[lag_counter] = xcorr32[(lag_high - slot) as usize];
        }

        let delta = i32::from(lag_range[k][0]);
        for i in 0..nb_cbk_search {
            let code = if use_full_cb {
                SILK_CB_LAGS_STAGE3[k][i]
            } else {
                SILK_CB_LAGS_STAGE3_10_MS[k][i]
            };
            let idx = i32::from(code) - delta;
            for j in 0..PE_NB_STAGE3_LAGS {
                cross_corr_st3[k * nb_cbk_search + i][j] = scratch_mem[(idx + j as i32) as usize];
            }
        }
        target_idx += sf_length;
    }
    let _ = arch;
}

fn p_ana_calc_energy_st3(
    energies_st3: &mut [[i32; PE_NB_STAGE3_LAGS]],
    frame: &[i16],
    start_lag: i32,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    arch: i32,
) {
    let use_full_cb = nb_subfr == PE_MAX_NB_SUBFR;
    let lag_range: &[[i8; 2]] = if use_full_cb {
        &SILK_LAG_RANGE_STAGE3[complexity]
    } else {
        &SILK_LAG_RANGE_STAGE3_10_MS
    };
    let nb_cbk_search = if use_full_cb {
        SILK_NB_CBK_SEARCHS_STAGE3[complexity] as usize
    } else {
        PE_NB_CBKS_STAGE3_10_MS
    };

    let mut scratch_mem = [0i32; SCRATCH_SIZE];
    let mut target_idx = sf_length << 2;

    for k in 0..nb_subfr {
        let basis_ptr =
            (target_idx as i32 - (start_lag + i32::from(lag_range[k][0]))) as usize;
        let mut energy = inner_prod_aligned(
            &frame[basis_ptr..basis_ptr + sf_length],
            &frame[basis_ptr..basis_ptr + sf_length],
            arch,
        );
        scratch_mem[0] = energy;

        let lag_diff = (lag_range[k][1] - lag_range[k][0] + 1) as usize;
        for i in 1..lag_diff {
            energy = energy
                .wrapping_sub(smulbb(
                    i32::from(frame[basis_ptr + sf_length - i]),
                    i32::from(frame[basis_ptr + sf_length - i]),
                ))
                .wrapping_add(smulbb(
                    i32::from(frame[basis_ptr - i]),
                    i32::from(frame[basis_ptr - i]),
                ));
            scratch_mem[i] = energy;
        }

        let delta = i32::from(lag_range[k][0]);
        for i in 0..nb_cbk_search {
            let code = if use_full_cb {
                SILK_CB_LAGS_STAGE3[k][i]
            } else {
                SILK_CB_LAGS_STAGE3_10_MS[k][i]
            };
            let idx = i32::from(code) - delta;
            for j in 0..PE_NB_STAGE3_LAGS {
                energies_st3[k * nb_cbk_search + i][j] = scratch_mem[(idx + j as i32) as usize];
            }
        }
        target_idx += sf_length;
    }
    let _ = arch;
}

fn pitch_xcorr(x: &[i16], y: &[i16], len: usize, max_pitch: usize, out: &mut [i32]) {
    assert!(out.len() >= max_pitch);
    assert!(x.len() >= len);
    assert!(y.len() >= len + max_pitch - 1);

    for delay in 0..max_pitch {
        let mut acc = 0i64;
        let y_slice = &y[delay..delay + len];
        for (&a, &b) in x.iter().zip(y_slice.iter()) {
            acc += i64::from(a) * i64::from(b);
        }
        out[delay] = acc as i32;
    }
}

fn smulbb(a: i32, b: i32) -> i32 {
    let lhs = i32::from(a as i16);
    let rhs = i32::from(b as i16);
    lhs.wrapping_mul(rhs)
}

fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwb(b, c))
}

fn div32_16(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}

fn div32(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}

fn add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b.wrapping_shl(shift as u32))
}

fn rshift(value: i32, shift: i32) -> i32 {
    value >> shift
}

fn rshift_signed(value: i16, shift: i32) -> i16 {
    let widened = i32::from(value);
    (widened >> shift) as i16
}

fn sat16(value: i32) -> i16 {
    if value > i32::from(i16::MAX) {
        i16::MAX
    } else if value < i32::from(i16::MIN) {
        i16::MIN
    } else {
        value as i16
    }
}

fn clz32(value: i32) -> i32 {
    if value == 0 {
        32
    } else {
        (value as u32).leading_zeros() as i32
    }
}

fn limit_int(value: i32, min: i32, max: i32) -> i32 {
    value.clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn returns_unvoiced_for_silence() {
        let fs_khz = 8;
        let nb_subfr = 4;
        let frame_length =
            (PE_LTP_MEM_LENGTH_MS + nb_subfr as usize * PE_SUBFR_LENGTH_MS) * fs_khz as usize;
        let frame = vec![0i16; frame_length];

        let mut pitch_out = [0i32; PE_MAX_NB_SUBFR];
        let mut lag_index = -1i16;
        let mut contour_index = -1i8;
        let mut ltp_corr_q15 = -1;

        let result = pitch_analysis_core(
            &frame,
            &mut pitch_out[..nb_subfr as usize],
            &mut lag_index,
            &mut contour_index,
            &mut ltp_corr_q15,
            0,
            (0.3f32 * 65536.0) as i32,
            (0.3f32 * 8192.0) as i32,
            fs_khz,
            1,
            nb_subfr,
            0,
        );

        assert_eq!(result, 1);
        assert!(pitch_out[..nb_subfr as usize].iter().all(|&p| p == 0));
        assert_eq!(lag_index, 0);
        assert_eq!(contour_index, 0);
        assert_eq!(ltp_corr_q15, 0);
    }

    #[test]
    fn detects_periodic_wave_as_voiced() {
        let fs_khz = 8;
        let nb_subfr = 4;
        let frame_length =
            (PE_LTP_MEM_LENGTH_MS + nb_subfr as usize * PE_SUBFR_LENGTH_MS) * fs_khz as usize;
        let mut frame = vec![0i16; frame_length];

        let period = 40;
        for (i, sample) in frame.iter_mut().enumerate() {
            *sample = if (i % period) < period / 2 { 800 } else { -800 };
        }

        let mut pitch_out = [0i32; PE_MAX_NB_SUBFR];
        let mut lag_index = 0i16;
        let mut contour_index = 0i8;
        let mut ltp_corr_q15 = 0;

        let result = pitch_analysis_core(
            &frame,
            &mut pitch_out[..nb_subfr as usize],
            &mut lag_index,
            &mut contour_index,
            &mut ltp_corr_q15,
            0,
            (0.25f32 * 65536.0) as i32,
            (0.2f32 * 8192.0) as i32,
            fs_khz,
            1,
            nb_subfr,
            0,
        );

        assert_eq!(result, 0);
        assert!(pitch_out[..nb_subfr as usize].iter().all(|&p| p > 0));
        assert!(lag_index >= 0);
        assert!(ltp_corr_q15 > 0);
    }
}
