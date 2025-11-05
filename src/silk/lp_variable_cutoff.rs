//! Ports the variable cut-off low-pass filter from `silk/LP_variable_cutoff.c`.
//!
//! This module implements an elliptic/Cauer filter with 0.1 dB passband ripple,
//! 80 dB minimum stopband attenuation, and normalized cut-off frequencies ranging
//! from 0.95 down to 0.35. The cut-off frequency is controlled via piece-wise linear
//! interpolation between pre-computed filter coefficient tables during bandwidth
//! transitions.

use crate::silk::biquad_alt::biquad_alt_stride1_inplace;
use crate::silk::tables_other::{
    SILK_TRANSITION_LP_A_Q28, SILK_TRANSITION_LP_B_Q28, TRANSITION_INT_NUM, TRANSITION_NA,
    TRANSITION_NB,
};

/// Maximum frame length in milliseconds (5ms subframe × 4 subframes).
const MAX_FRAME_LENGTH_MS: i32 = 20;

/// Total transition time in milliseconds.
const TRANSITION_TIME_MS: i32 = 5120;

/// Number of frames over which the transition occurs.
const TRANSITION_FRAMES: i32 = TRANSITION_TIME_MS / MAX_FRAME_LENGTH_MS;

/// Number of interpolation steps between coefficient tables.
const TRANSITION_INT_STEPS: i32 = TRANSITION_FRAMES / (TRANSITION_INT_NUM as i32 - 1);

/// State structure for the variable cut-off low-pass filter.
///
/// This mirrors `silk_LP_state` from the C implementation. The filter can be
/// activated and deactivated dynamically, and the cut-off frequency changes
/// gradually over multiple frames during bandwidth transitions.
#[derive(Clone, Debug, Default)]
pub struct LpState {
    /// Two-element Q12 state vector for the biquad filter.
    pub in_lp_state: [i32; 2],
    /// Frame counter mapping to the current cut-off frequency (0..=TRANSITION_FRAMES).
    pub transition_frame_no: i32,
    /// Operating mode: <0 = switch down, >0 = switch up, 0 = do nothing.
    pub mode: i32,
    /// If non-zero, holds the last sampling rate (in kHz) before a bandwidth switching reset.
    pub saved_fs_khz: i32,
}

impl LpState {
    /// Creates a new `LpState` with all fields initialized to zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Low-pass filters a frame with variable cut-off frequency.
    ///
    /// The cut-off frequency is determined by `self.transition_frame_no` and changes
    /// gradually over time when `self.mode != 0`. The filter coefficients are interpolated
    /// from pre-computed elliptic filter tables.
    ///
    /// # Parameters
    /// - `frame`: Mutable slice of Q0 samples to be filtered in-place.
    ///
    /// # Panics
    /// Panics if `self.transition_frame_no` is outside the range `[0, TRANSITION_FRAMES]`.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::arithmetic_side_effects
    )]
    pub fn lp_variable_cutoff(&mut self, frame: &mut [i16]) {
        assert!(
            self.transition_frame_no >= 0 && self.transition_frame_no <= TRANSITION_FRAMES,
            "transition_frame_no must be in range [0, {}]",
            TRANSITION_FRAMES
        );

        // Run filter only if mode is active
        if self.mode == 0 {
            return;
        }

        // Calculate index and interpolation factor for interpolation
        let mut fac_q16 = if TRANSITION_INT_STEPS == 64 {
            // Optimized shift when TRANSITION_INT_STEPS == 64
            (TRANSITION_FRAMES - self.transition_frame_no) << (16 - 6)
        } else {
            // General division
            ((TRANSITION_FRAMES - self.transition_frame_no) << 16) / TRANSITION_FRAMES
        };

        let ind = (fac_q16 >> 16) as usize;
        fac_q16 -= (ind as i32) << 16;

        assert!(
            ind < TRANSITION_INT_NUM,
            "index must be less than {}",
            TRANSITION_INT_NUM
        );

        // Interpolate filter coefficients
        let mut b_q28 = [0i32; TRANSITION_NB];
        let mut a_q28 = [0i32; TRANSITION_NA];
        lp_interpolate_filter_taps(&mut b_q28, &mut a_q28, ind, fac_q16);

        // Update transition frame number for next frame
        self.transition_frame_no =
            limit(self.transition_frame_no + self.mode, 0, TRANSITION_FRAMES);

        // ARMA low-pass filtering
        // Note: TRANSITION_NB == 3 and TRANSITION_NA == 2 (hardcoded in tables)
        assert_eq!(TRANSITION_NB, 3);
        assert_eq!(TRANSITION_NA, 2);

        biquad_alt_stride1_inplace(frame, &b_q28, &a_q28, &mut self.in_lp_state);
    }
}

/// Interpolates filter taps between pre-computed coefficient tables.
///
/// This helper function performs piece-wise linear interpolation of the numerator (B)
/// and denominator (A) coefficients based on the given index and interpolation factor.
///
/// # Parameters
/// - `b_q28`: Output array for interpolated B (numerator) coefficients in Q28.
/// - `a_q28`: Output array for interpolated A (denominator) coefficients in Q28.
/// - `ind`: Index into the coefficient tables (must be less than
///   `TRANSITION_INT_NUM - 1` for interpolation to occur).
/// - `fac_q16`: Q16 interpolation factor within the current interval.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::arithmetic_side_effects
)]
fn lp_interpolate_filter_taps(
    b_q28: &mut [i32; TRANSITION_NB],
    a_q28: &mut [i32; TRANSITION_NA],
    ind: usize,
    fac_q16: i32,
) {
    if ind < TRANSITION_INT_NUM - 1 {
        if fac_q16 > 0 {
            if fac_q16 < 32768 {
                // fac_q16 is in range of a 16-bit int
                // Piece-wise linear interpolation of B and A
                for nb in 0..TRANSITION_NB {
                    b_q28[nb] = smlawb(
                        SILK_TRANSITION_LP_B_Q28[ind][nb],
                        SILK_TRANSITION_LP_B_Q28[ind + 1][nb] - SILK_TRANSITION_LP_B_Q28[ind][nb],
                        fac_q16,
                    );
                }
                for na in 0..TRANSITION_NA {
                    a_q28[na] = smlawb(
                        SILK_TRANSITION_LP_A_Q28[ind][na],
                        SILK_TRANSITION_LP_A_Q28[ind + 1][na] - SILK_TRANSITION_LP_A_Q28[ind][na],
                        fac_q16,
                    );
                }
            } else {
                // (fac_q16 - (1 << 16)) is in range of a 16-bit int
                assert_eq!(
                    fac_q16 - (1 << 16),
                    sat16(fac_q16 - (1 << 16)),
                    "fac_q16 - 65536 must be in 16-bit range"
                );
                // Piece-wise linear interpolation of B and A
                for nb in 0..TRANSITION_NB {
                    b_q28[nb] = smlawb(
                        SILK_TRANSITION_LP_B_Q28[ind + 1][nb],
                        SILK_TRANSITION_LP_B_Q28[ind + 1][nb] - SILK_TRANSITION_LP_B_Q28[ind][nb],
                        fac_q16 - (1 << 16),
                    );
                }
                for na in 0..TRANSITION_NA {
                    a_q28[na] = smlawb(
                        SILK_TRANSITION_LP_A_Q28[ind + 1][na],
                        SILK_TRANSITION_LP_A_Q28[ind + 1][na] - SILK_TRANSITION_LP_A_Q28[ind][na],
                        fac_q16 - (1 << 16),
                    );
                }
            }
        } else {
            // fac_q16 == 0, no interpolation needed
            b_q28.copy_from_slice(&SILK_TRANSITION_LP_B_Q28[ind]);
            a_q28.copy_from_slice(&SILK_TRANSITION_LP_A_Q28[ind]);
        }
    } else {
        // Use last table entry
        b_q28.copy_from_slice(&SILK_TRANSITION_LP_B_Q28[TRANSITION_INT_NUM - 1]);
        a_q28.copy_from_slice(&SILK_TRANSITION_LP_A_Q28[TRANSITION_INT_NUM - 1]);
    }
}

/// Fixed-point multiply-accumulate: a + ((b * c[15:0]) >> 16).
///
/// This mirrors the `silk_SMLAWB` macro from the C implementation. It multiplies
/// the full 32-bit `b` by the low 16 bits of `c` (sign-extended), shifts the
/// 48-bit product right by 16 bits, and adds the result to `a`.
#[inline]
fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    let product = i64::from(b) * i64::from(c as i16);
    a.wrapping_add((product >> 16) as i32)
}

/// Saturates a 32-bit value to 16-bit signed range.
#[inline]
fn sat16(value: i32) -> i32 {
    if value > i32::from(i16::MAX) {
        i32::from(i16::MAX)
    } else if value < i32::from(i16::MIN) {
        i32::from(i16::MIN)
    } else {
        value
    }
}

/// Clamps a value to the given range.
#[inline]
fn limit(value: i32, min: i32, max: i32) -> i32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_state_default_is_zeroed() {
        let state = LpState::default();
        assert_eq!(state.in_lp_state, [0, 0]);
        assert_eq!(state.transition_frame_no, 0);
        assert_eq!(state.mode, 0);
        assert_eq!(state.saved_fs_khz, 0);
    }

    #[test]
    fn lp_variable_cutoff_does_nothing_when_mode_is_zero() {
        let mut state = LpState::new();
        state.mode = 0;
        let mut frame = [100i16, -200, 300, -400, 500];
        let original = frame;

        state.lp_variable_cutoff(&mut frame);

        assert_eq!(frame, original, "frame should be unchanged when mode is 0");
        assert_eq!(
            state.transition_frame_no, 0,
            "transition_frame_no should not change"
        );
    }

    #[test]
    fn lp_variable_cutoff_updates_transition_frame_no() {
        let mut state = LpState::new();
        state.mode = 1;
        state.transition_frame_no = 100;
        let mut frame = [0i16; 160];

        state.lp_variable_cutoff(&mut frame);

        assert_eq!(
            state.transition_frame_no, 101,
            "transition_frame_no should increment by mode"
        );
    }

    #[test]
    fn lp_variable_cutoff_clamps_transition_frame_no() {
        let mut state = LpState::new();
        state.mode = 10;
        state.transition_frame_no = TRANSITION_FRAMES - 5;
        let mut frame = [0i16; 160];

        state.lp_variable_cutoff(&mut frame);

        assert_eq!(
            state.transition_frame_no, TRANSITION_FRAMES,
            "transition_frame_no should be clamped to TRANSITION_FRAMES"
        );
    }

    #[test]
    fn lp_interpolate_uses_last_table_when_ind_at_end() {
        let mut b = [0i32; TRANSITION_NB];
        let mut a = [0i32; TRANSITION_NA];

        lp_interpolate_filter_taps(&mut b, &mut a, TRANSITION_INT_NUM - 1, 12345);

        assert_eq!(b, SILK_TRANSITION_LP_B_Q28[TRANSITION_INT_NUM - 1]);
        assert_eq!(a, SILK_TRANSITION_LP_A_Q28[TRANSITION_INT_NUM - 1]);
    }

    #[test]
    fn lp_interpolate_copies_when_fac_is_zero() {
        let mut b = [0i32; TRANSITION_NB];
        let mut a = [0i32; TRANSITION_NA];

        lp_interpolate_filter_taps(&mut b, &mut a, 2, 0);

        assert_eq!(b, SILK_TRANSITION_LP_B_Q28[2]);
        assert_eq!(a, SILK_TRANSITION_LP_A_Q28[2]);
    }

    #[test]
    fn smlawb_matches_reference() {
        // smlawb(a, b, c) = a + ((b * c[15:0]) >> 16)
        // where c[15:0] is the low 16 bits of c interpreted as signed i16
        assert_eq!(smlawb(1000, 0x10000, 0x8000), -31768);
        // 1000 + ((65536 * -32768) >> 16) = 1000 + (-32768) = -31768

        assert_eq!(smlawb(0, 100_000, -1), -2);
        // 0 + ((100000 * -1) >> 16) = 0 + (-2) = -2

        assert_eq!(smlawb(500, -200_000, 0x1234), -13722);
        // 500 + ((-200000 * 4660) >> 16) = 500 + (-14222) = -13722
    }

    #[test]
    fn limit_clamps_below_min() {
        assert_eq!(limit(-10, 0, 100), 0);
    }

    #[test]
    fn limit_clamps_above_max() {
        assert_eq!(limit(150, 0, 100), 100);
    }

    #[test]
    fn limit_preserves_in_range() {
        assert_eq!(limit(50, 0, 100), 50);
    }
}
