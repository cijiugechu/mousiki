//! Port of `silk/decode_pitch.c`.
//!
//! This helper reconstructs the four per-subframe pitch lags from the
//! absolute lag index and the decoded contour entry.  It mirrors the reference
//! SILK routine so callers can share the same stage-two/stage-three codebooks
//! regardless of whether the decoder is running at 8, 12, or 16 kHz.

use core::convert::TryFrom;

use crate::silk::pitch_est_tables::{
    PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2_10_MS,
    PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10_MS, PE_NB_CBKS_STAGE3_MAX,
    SILK_CB_LAGS_STAGE2, SILK_CB_LAGS_STAGE2_10_MS, SILK_CB_LAGS_STAGE3,
    SILK_CB_LAGS_STAGE3_10_MS,
};

const NB_SUBFR_10_MS: usize = PE_MAX_NB_SUBFR / 2;

/// Decode the per-subframe pitch lags from the quantised lag index and contour.
///
/// * `lag_index` — Absolute lag index without the minimum-lag bias.
/// * `contour_index` — Codebook entry that selects the contour offsets.
/// * `pitch_lags` — Output buffer that receives `nb_subfr` pitch values.
/// * `fs_khz` — Internal sampling rate in kHz (8/12/16).
/// * `nb_subfr` — Number of 5 ms subframes (4 for 20 ms frames, 2 for 10 ms).
pub fn silk_decode_pitch(
    lag_index: i16,
    contour_index: i8,
    pitch_lags: &mut [i16],
    fs_khz: i32,
    nb_subfr: usize,
) {
    debug_assert!(pitch_lags.len() >= nb_subfr);
    debug_assert!(matches!(fs_khz, 8 | 12 | 16));
    debug_assert!(nb_subfr == PE_MAX_NB_SUBFR || nb_subfr == NB_SUBFR_10_MS);

    let cbk_size = codebook_size(fs_khz, nb_subfr);
    let contour = usize::try_from(contour_index).expect("contour index must be non-negative");
    debug_assert!(contour < cbk_size);

    let min_lag = (PE_MIN_LAG_MS as i32) * fs_khz;
    let max_lag = (PE_MAX_LAG_MS as i32) * fs_khz;
    let base_lag = min_lag + i32::from(lag_index);

    for (subframe_idx, lag) in pitch_lags.iter_mut().enumerate().take(nb_subfr) {
        let offset = codebook_entry(fs_khz, nb_subfr, subframe_idx, contour);
        let candidate = (base_lag + offset).clamp(min_lag, max_lag);
        *lag = candidate as i16;
    }

    for lag in pitch_lags.iter_mut().skip(nb_subfr) {
        *lag = 0;
    }
}

fn codebook_size(fs_khz: i32, nb_subfr: usize) -> usize {
    match (fs_khz == 8, nb_subfr) {
        (true, PE_MAX_NB_SUBFR) => PE_NB_CBKS_STAGE2_EXT,
        (true, NB_SUBFR_10_MS) => PE_NB_CBKS_STAGE2_10_MS,
        (false, PE_MAX_NB_SUBFR) => PE_NB_CBKS_STAGE3_MAX,
        (false, NB_SUBFR_10_MS) => PE_NB_CBKS_STAGE3_10_MS,
        _ => panic!("unsupported fs/nb_subfr combination"),
    }
}

fn codebook_entry(fs_khz: i32, nb_subfr: usize, subframe_idx: usize, contour_idx: usize) -> i32 {
    match (fs_khz == 8, nb_subfr) {
        (true, PE_MAX_NB_SUBFR) => {
            debug_assert!(subframe_idx < PE_MAX_NB_SUBFR);
            debug_assert!(contour_idx < PE_NB_CBKS_STAGE2_EXT);
            i32::from(SILK_CB_LAGS_STAGE2[subframe_idx][contour_idx])
        }
        (true, NB_SUBFR_10_MS) => {
            debug_assert!(subframe_idx < NB_SUBFR_10_MS);
            debug_assert!(contour_idx < PE_NB_CBKS_STAGE2_10_MS);
            i32::from(SILK_CB_LAGS_STAGE2_10_MS[subframe_idx][contour_idx])
        }
        (false, PE_MAX_NB_SUBFR) => {
            debug_assert!(subframe_idx < PE_MAX_NB_SUBFR);
            debug_assert!(contour_idx < PE_NB_CBKS_STAGE3_MAX);
            i32::from(SILK_CB_LAGS_STAGE3[subframe_idx][contour_idx])
        }
        (false, NB_SUBFR_10_MS) => {
            debug_assert!(subframe_idx < NB_SUBFR_10_MS);
            debug_assert!(contour_idx < PE_NB_CBKS_STAGE3_10_MS);
            i32::from(SILK_CB_LAGS_STAGE3_10_MS[subframe_idx][contour_idx])
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::silk_decode_pitch;
    use crate::silk::MAX_NB_SUBFR;

    #[test]
    fn decode_pitch_wideband_stage3() {
        let mut lags = [0i16; MAX_NB_SUBFR];
        silk_decode_pitch(0, 1, &mut lags, 16, MAX_NB_SUBFR);
        assert_eq!(&lags, &[32, 32, 33, 33]);
    }

    #[test]
    fn decode_pitch_narrowband_stage2() {
        let mut lags = [0i16; MAX_NB_SUBFR];
        silk_decode_pitch(5, 2, &mut lags, 8, MAX_NB_SUBFR);
        assert_eq!(&lags, &[20, 21, 22, 23]);
    }

    #[test]
    fn decode_pitch_stage3_10ms() {
        let mut lags = [0i16; MAX_NB_SUBFR];
        silk_decode_pitch(3, 5, &mut lags, 16, MAX_NB_SUBFR / 2);
        assert_eq!(&lags, &[34, 37, 0, 0]);
    }

    #[test]
    fn decode_pitch_stage2_10ms() {
        let mut lags = [0i16; MAX_NB_SUBFR];
        silk_decode_pitch(2, 1, &mut lags, 8, MAX_NB_SUBFR / 2);
        assert_eq!(&lags, &[19, 18, 0, 0]);
    }
}
