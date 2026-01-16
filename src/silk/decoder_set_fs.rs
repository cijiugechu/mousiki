//! Port of `silk/decoder_set_fs.c`.
//!
//! The SILK decoder keeps a small chunk of mutable state that depends on the
//! currently configured internal (8/12/16 kHz) and external (API) sampling
//! rates.  The C implementation exposes `silk_decoder_set_fs` to update those
//! members whenever the application switches bandwidth.  This module mirrors
//! that helper together with a lightweight struct that captures just the
//! fields touched by the original routine.

use core::fmt;

use crate::silk::resampler::{Resampler, ResamplerInitError};
use crate::silk::tables_nlsf_cb_nb_mb::SILK_NLSF_CB_NB_MB;
use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;
use crate::silk::tables_other::{SILK_UNIFORM4_ICDF, SILK_UNIFORM6_ICDF, SILK_UNIFORM8_ICDF};
use crate::silk::tables_pitch_lag::{
    PITCH_CONTOUR_10_MS_ICDF, PITCH_CONTOUR_10_MS_NB_ICDF, PITCH_CONTOUR_ICDF,
    PITCH_CONTOUR_NB_ICDF,
};
use crate::silk::{FrameSignalType, MAX_LPC_ORDER, MAX_NB_SUBFR, MIN_LPC_ORDER, SilkNlsfCb};

pub(crate) const SUB_FRAME_LENGTH_MS: usize = 5;
const LTP_MEM_LENGTH_MS: usize = 20;
const MAX_FS_KHZ: usize = 16;
pub(crate) const MAX_SUB_FRAME_LENGTH: usize = SUB_FRAME_LENGTH_MS * MAX_FS_KHZ;
const MAX_FRAME_LENGTH_MS: usize = SUB_FRAME_LENGTH_MS * MAX_NB_SUBFR;
pub(crate) const MAX_FRAME_LENGTH: usize = MAX_FRAME_LENGTH_MS * MAX_FS_KHZ;
pub(crate) const MAX_DECODER_BUFFER: usize = MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH;

/// Errors that can occur when switching decoder sampling rates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderSetFsError {
    /// `fs_khz` was not one of {8, 12, 16}.
    UnsupportedInternalSampleRate(i32),
    /// The decoder state tracks neither 20 ms (4 subframe) nor 10 ms (2 subframe) packets.
    InvalidSubframeCount(usize),
    /// Reinitialising the resampler failed.
    Resampler(ResamplerInitError),
}

impl fmt::Display for DecoderSetFsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedInternalSampleRate(rate) => {
                write!(f, "unsupported internal sampling rate {rate} kHz")
            }
            Self::InvalidSubframeCount(count) => {
                write!(f, "invalid decoder subframe count {count}")
            }
            Self::Resampler(err) => write!(f, "failed to init SILK resampler: {err}"),
        }
    }
}

impl From<ResamplerInitError> for DecoderSetFsError {
    fn from(value: ResamplerInitError) -> Self {
        Self::Resampler(value)
    }
}

/// Minimal subset of the decoder state touched by `silk_decoder_set_fs`.
#[derive(Clone, Debug)]
pub struct DecoderSampleRateState {
    /// Internal sampling rate (kHz).
    pub fs_khz: i32,
    /// API sampling rate (Hz).
    pub fs_api_hz: i32,
    /// Number of subframes per packet (2 for 10 ms, 4 for 20 ms).
    pub nb_subfr: usize,
    /// Samples per subframe.
    pub subfr_length: usize,
    /// Samples per frame.
    pub frame_length: usize,
    /// LTP memory length (samples).
    pub ltp_mem_length: usize,
    /// Active LPC order (either [`MIN_LPC_ORDER`] or [`MAX_LPC_ORDER`]).
    pub lpc_order: usize,
    /// Flag updated when the decoder needs to disable NLSF interpolation.
    pub first_frame_after_reset: bool,
    /// Previous decoded pitch lag.
    pub lag_prev: i32,
    /// Previous gain index.
    pub last_gain_index: i32,
    /// Previous decoded signal type.
    pub prev_signal_type: FrameSignalType,
    /// Pointer to the appropriate pitch-lag low-bit iCDF table.
    pub pitch_lag_low_bits_icdf: &'static [u8],
    /// Pointer to the pitch-contour iCDF table.
    pub pitch_contour_icdf: &'static [u8],
    /// Active NLSF codebook.
    pub ps_nlsf_cb: &'static SilkNlsfCb,
    /// Decoder-side resampler state used by the API wrapper.
    pub resampler_state: Resampler,
    /// Output ring buffer used when transitioning sample rates.
    pub out_buf: [i16; MAX_DECODER_BUFFER],
    /// Past LPC samples expressed in Q14.
    pub s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
}

impl Default for DecoderSampleRateState {
    fn default() -> Self {
        Self {
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: MAX_NB_SUBFR,
            subfr_length: 0,
            frame_length: 0,
            ltp_mem_length: 0,
            lpc_order: MAX_LPC_ORDER,
            first_frame_after_reset: true,
            lag_prev: 0,
            last_gain_index: 0,
            prev_signal_type: FrameSignalType::Inactive,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM4_ICDF,
            pitch_contour_icdf: &PITCH_CONTOUR_ICDF,
            ps_nlsf_cb: &SILK_NLSF_CB_WB,
            resampler_state: Resampler::default(),
            out_buf: [0; MAX_DECODER_BUFFER],
            s_lpc_q14_buf: [0; MAX_LPC_ORDER],
        }
    }
}

impl DecoderSampleRateState {
    /// Construct a decoder sample-rate state for the given number of subframes.
    pub fn with_subframes(nb_subfr: usize) -> Self {
        assert!(
            nb_subfr == MAX_NB_SUBFR || nb_subfr == MAX_NB_SUBFR / 2,
            "nb_subfr must be 2 or 4"
        );
        Self {
            nb_subfr,
            ..Self::default()
        }
    }

    /// Mirrors the reference `silk_decoder_set_fs`.
    pub fn set_sample_rates(
        &mut self,
        fs_khz: i32,
        fs_api_hz: i32,
    ) -> Result<(), DecoderSetFsError> {
        if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
            return Err(DecoderSetFsError::UnsupportedInternalSampleRate(fs_khz));
        }
        if self.nb_subfr != MAX_NB_SUBFR && self.nb_subfr != MAX_NB_SUBFR / 2 {
            return Err(DecoderSetFsError::InvalidSubframeCount(self.nb_subfr));
        }

        let fs_khz_usize = fs_khz as usize;
        let subfr_length = SUB_FRAME_LENGTH_MS * fs_khz_usize;
        let frame_length = self.nb_subfr * subfr_length;

        if self.fs_khz != fs_khz || self.fs_api_hz != fs_api_hz {
            let fs_hz = fs_khz
                .checked_mul(1000)
                .expect("fs_khz conversion to Hz must not overflow");
            self.resampler_state
                .silk_resampler_init(fs_hz, fs_api_hz, false)?;
            self.fs_api_hz = fs_api_hz;
        }

        if self.fs_khz != fs_khz || self.frame_length != frame_length {
            self.pitch_contour_icdf =
                select_pitch_contour_table(fs_khz, self.nb_subfr == MAX_NB_SUBFR);

            if self.fs_khz != fs_khz {
                self.ltp_mem_length = LTP_MEM_LENGTH_MS * fs_khz_usize;
                if fs_khz == 8 || fs_khz == 12 {
                    self.lpc_order = MIN_LPC_ORDER;
                    self.ps_nlsf_cb = &SILK_NLSF_CB_NB_MB;
                } else {
                    self.lpc_order = MAX_LPC_ORDER;
                    self.ps_nlsf_cb = &SILK_NLSF_CB_WB;
                }

                self.pitch_lag_low_bits_icdf = match fs_khz {
                    8 => &SILK_UNIFORM4_ICDF,
                    12 => &SILK_UNIFORM6_ICDF,
                    16 => &SILK_UNIFORM8_ICDF,
                    _ => unreachable!("fs_khz validated above"),
                };

                self.first_frame_after_reset = true;
                self.lag_prev = 100;
                self.last_gain_index = 10;
                self.prev_signal_type = FrameSignalType::Inactive;
                self.out_buf.fill(0);
                self.s_lpc_q14_buf.fill(0);
            }

            self.fs_khz = fs_khz;
            self.subfr_length = subfr_length;
            self.frame_length = frame_length;
            debug_assert!(
                self.frame_length > 0 && self.frame_length <= MAX_FRAME_LENGTH,
                "frame length must be within SILK bounds"
            );
        }

        Ok(())
    }
}

fn select_pitch_contour_table(fs_khz: i32, is_20_ms: bool) -> &'static [u8] {
    match (fs_khz, is_20_ms) {
        (8, true) => &PITCH_CONTOUR_NB_ICDF,
        (8, false) => &PITCH_CONTOUR_10_MS_NB_ICDF,
        (_, true) => &PITCH_CONTOUR_ICDF,
        (_, false) => &PITCH_CONTOUR_10_MS_ICDF,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(debug_assertions)]
    fn set_sample_rates_updates_wideband_state() {
        let mut state = DecoderSampleRateState::default();

        state.set_sample_rates(16, 48_000).unwrap();

        assert_eq!(state.fs_khz, 16);
        assert_eq!(state.fs_api_hz, 48_000);
        assert_eq!(state.subfr_length, SUB_FRAME_LENGTH_MS * 16);
        assert_eq!(state.frame_length, MAX_NB_SUBFR * SUB_FRAME_LENGTH_MS * 16);
        assert_eq!(state.ltp_mem_length, LTP_MEM_LENGTH_MS * 16);
        assert_eq!(state.lpc_order, MAX_LPC_ORDER);
        assert!(core::ptr::eq(
            state.pitch_contour_icdf,
            &PITCH_CONTOUR_ICDF as &[_]
        ));
        assert!(core::ptr::eq(
            state.pitch_lag_low_bits_icdf,
            &SILK_UNIFORM8_ICDF as &[_]
        ));
        assert!(core::ptr::eq(state.ps_nlsf_cb, &SILK_NLSF_CB_WB));
        assert!(state.first_frame_after_reset);
        assert_eq!(state.lag_prev, 100);
        assert_eq!(state.last_gain_index, 10);
        assert_eq!(state.prev_signal_type, FrameSignalType::Inactive);
        assert!(state.out_buf.iter().all(|sample| *sample == 0));
        assert!(state.s_lpc_q14_buf.iter().all(|sample| *sample == 0));
        assert_eq!(state.resampler_state.fs_in_khz(), 16);
        assert_eq!(state.resampler_state.fs_out_khz(), 48);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn narrowband_10_ms_selects_nb_tables() {
        let mut state = DecoderSampleRateState::with_subframes(MAX_NB_SUBFR / 2);

        state.set_sample_rates(8, 8_000).unwrap();

        assert_eq!(state.subfr_length, SUB_FRAME_LENGTH_MS * 8);
        assert_eq!(
            state.frame_length,
            (MAX_NB_SUBFR / 2) * SUB_FRAME_LENGTH_MS * 8
        );
        assert_eq!(state.lpc_order, MIN_LPC_ORDER);
        assert_eq!(state.ltp_mem_length, LTP_MEM_LENGTH_MS * 8);
        assert!(core::ptr::eq(
            state.pitch_contour_icdf,
            &PITCH_CONTOUR_10_MS_NB_ICDF as &[_]
        ));
        assert!(core::ptr::eq(
            state.pitch_lag_low_bits_icdf,
            &SILK_UNIFORM4_ICDF as &[_]
        ));
        assert!(core::ptr::eq(state.ps_nlsf_cb, &SILK_NLSF_CB_NB_MB));
    }

    #[test]
    fn resampler_reconfigured_on_rate_change() {
        let mut state = DecoderSampleRateState::default();
        state.set_sample_rates(16, 48_000).unwrap();

        // Change only the API rate to ensure we still reconfigure the resampler.
        state.set_sample_rates(16, 44_100).unwrap();
        assert_eq!(state.fs_api_hz, 44_100);
        assert_eq!(state.resampler_state.fs_out_khz(), 44);
    }
}
