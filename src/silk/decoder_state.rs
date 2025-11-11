//! Typed representation of `silk_decoder_state`.
//!
//! The original SILK implementation stores the decoder working state inside
//! `silk_decoder_state` defined in `silk/structs.h`.  This module mirrors the
//! relevant fields so that higher-level helpers such as `silk_decoder_set_fs`
//! and `silk_reset_decoder` can operate entirely within Rust without touching
//! the C structs.

use core::convert::TryFrom;

use crate::silk::cng::CngState;
use crate::silk::decode_indices::SideInfoIndices;
use crate::silk::decoder_set_fs::{DecoderSampleRateState, MAX_FRAME_LENGTH};
use crate::silk::vq_wmat_ec::LTP_ORDER;
use crate::silk::{FrameSignalType, MAX_FRAMES_PER_PACKET, MAX_LPC_ORDER};

const UNITY_Q16: i32 = 1 << 16;
const DEFAULT_PLC_SUBFRAME_LENGTH: i32 = 20;
const DEFAULT_PLC_SUBFRAME_COUNT: i32 = 2;

/// Decoder-side packet-loss concealment summary (`silk_PLC_struct`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PacketLossConcealmentState {
    /// Pitch lag expressed in Q8.
    pub pitch_l_q8: i32,
    /// LTP coefficients used during voiced concealment.
    pub ltp_coef_q14: [i16; LTP_ORDER],
    /// Previous LPC coefficients in Q12.
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    /// Tracks whether the last frame was lost.
    pub last_frame_lost: i32,
    /// Random seed for generating unvoiced noise.
    pub rand_seed: i32,
    /// Scaling factor for unvoiced random excitation (Q14).
    pub rand_scale_q14: i16,
    /// Energy of the synthesized concealment signal.
    pub conc_energy: i32,
    /// Right shift applied to the concealment energy.
    pub conc_energy_shift: i32,
    /// Scaling factor applied to the LTP state during concealment (Q14).
    pub prev_ltp_scale_q14: i16,
    /// Previous subframe gains in Q16.
    pub prev_gain_q16: [i32; 2],
    /// Internal sampling rate tracked by the PLC helper.
    pub fs_khz: i32,
    /// Number of subframes stored internally.
    pub nb_subfr: i32,
    /// Subframe length tracked by the PLC helper.
    pub subfr_length: i32,
    /// Enables the optional deep-PLC path.
    pub enable_deep_plc: bool,
}

impl Default for PacketLossConcealmentState {
    fn default() -> Self {
        Self {
            pitch_l_q8: 0,
            ltp_coef_q14: [0; LTP_ORDER],
            prev_lpc_q12: [0; MAX_LPC_ORDER],
            last_frame_lost: 0,
            rand_seed: 0,
            rand_scale_q14: 0,
            conc_energy: 0,
            conc_energy_shift: 0,
            prev_ltp_scale_q14: 0,
            prev_gain_q16: [UNITY_Q16; 2],
            fs_khz: 0,
            nb_subfr: DEFAULT_PLC_SUBFRAME_COUNT,
            subfr_length: DEFAULT_PLC_SUBFRAME_LENGTH,
            enable_deep_plc: false,
        }
    }
}

impl PacketLossConcealmentState {
    /// Mirrors `silk_PLC_Reset`.
    pub fn reset(&mut self, frame_length: usize) {
        let samples = i32::try_from(frame_length)
            .expect("decoder frame length must fit in a signed 32-bit integer");
        self.pitch_l_q8 = samples << 7;
        self.prev_gain_q16 = [UNITY_Q16; 2];
        self.subfr_length = DEFAULT_PLC_SUBFRAME_LENGTH;
        self.nb_subfr = DEFAULT_PLC_SUBFRAME_COUNT;
        self.last_frame_lost = 0;
        self.rand_seed = 0;
        self.rand_scale_q14 = 0;
        self.conc_energy = 0;
        self.conc_energy_shift = 0;
        self.prev_ltp_scale_q14 = 0;
        self.fs_khz = 0;
        self.enable_deep_plc = false;
        self.ltp_coef_q14 = [0; LTP_ORDER];
        self.prev_lpc_q12 = [0; MAX_LPC_ORDER];
    }
}

/// Rust equivalent of `silk_decoder_state`.
#[derive(Debug)]
pub struct DecoderState {
    /// Subset of members managed by `silk_decoder_set_fs`.
    pub sample_rate: DecoderSampleRateState,
    /// Previous decoded gain (Q16).
    pub prev_gain_q16: i32,
    /// Excitation buffer expressed in Q14.
    pub exc_q14: [i32; MAX_FRAME_LENGTH],
    /// Cached NLSF vector from the previous frame.
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    /// Side-information indices decoded for the current frame.
    pub indices: SideInfoIndices,
    /// Number of frames decoded in the current packet.
    pub n_frames_decoded: i32,
    /// Target number of frames per packet.
    pub n_frames_per_packet: i32,
    /// Previous signal type tracked for conditional coding decisions.
    pub ec_prev_signal_type: FrameSignalType,
    /// Previous lag index tracked for conditional coding.
    pub ec_prev_lag_index: i16,
    /// Cached VAD decisions for frames buffered in the current packet.
    pub vad_flags: [i32; MAX_FRAMES_PER_PACKET],
    /// Indicates whether low-bit-rate redundancy was present.
    pub lbrr_flag: i32,
    /// LBRR flags per frame in the buffered packet.
    pub lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    /// Comfort-noise generator state.
    pub cng_state: CngState,
    /// Running loss counter.
    pub loss_count: i32,
    /// Architecture capabilities determined at runtime.
    pub arch: i32,
    /// Packet-loss concealment helper.
    pub plc_state: PacketLossConcealmentState,
}

impl Default for DecoderState {
    fn default() -> Self {
        Self {
            sample_rate: DecoderSampleRateState::default(),
            prev_gain_q16: UNITY_Q16,
            exc_q14: [0; MAX_FRAME_LENGTH],
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            indices: SideInfoIndices::default(),
            n_frames_decoded: 0,
            n_frames_per_packet: 0,
            ec_prev_signal_type: FrameSignalType::Inactive,
            ec_prev_lag_index: 0,
            vad_flags: [0; MAX_FRAMES_PER_PACKET],
            lbrr_flag: 0,
            lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            cng_state: CngState::default(),
            loss_count: 0,
            arch: 0,
            plc_state: PacketLossConcealmentState::default(),
        }
    }
}

impl DecoderState {
    /// Borrow the decoder's sample-rate specific members.
    pub fn sample_rate(&self) -> &DecoderSampleRateState {
        &self.sample_rate
    }

    /// Mutably borrow the decoder's sample-rate specific members.
    pub fn sample_rate_mut(&mut self) -> &mut DecoderSampleRateState {
        &mut self.sample_rate
    }

    /// Returns the internal excitation buffer.
    pub fn excitation_q14(&self) -> &[i32; MAX_FRAME_LENGTH] {
        &self.exc_q14
    }

    /// Returns a mutable view of the excitation buffer.
    pub fn excitation_q14_mut(&mut self) -> &mut [i32; MAX_FRAME_LENGTH] {
        &mut self.exc_q14
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DecoderState, PacketLossConcealmentState, DEFAULT_PLC_SUBFRAME_COUNT,
        DEFAULT_PLC_SUBFRAME_LENGTH, UNITY_Q16,
    };
    use crate::silk::decoder_set_fs::MAX_FRAME_LENGTH;
    use crate::silk::{MAX_FRAMES_PER_PACKET, MAX_LPC_ORDER};

    #[test]
    fn decoder_state_defaults_are_zeroed() {
        let state = DecoderState::default();
        assert_eq!(state.prev_gain_q16, UNITY_Q16);
        assert_eq!(state.exc_q14, [0; MAX_FRAME_LENGTH]);
        assert_eq!(state.prev_nlsf_q15, [0; MAX_LPC_ORDER]);
        assert_eq!(state.n_frames_decoded, 0);
        assert_eq!(state.n_frames_per_packet, 0);
        assert_eq!(state.loss_count, 0);
        assert_eq!(state.lbrr_flag, 0);
        assert_eq!(state.lbrr_flags, [0; MAX_FRAMES_PER_PACKET]);
    }

    #[test]
    fn plc_reset_matches_reference_defaults() {
        let mut plc = PacketLossConcealmentState::default();
        plc.reset(160);
        assert_eq!(plc.pitch_l_q8, 160 << 7);
        assert_eq!(plc.prev_gain_q16, [UNITY_Q16; 2]);
        assert_eq!(plc.subfr_length, DEFAULT_PLC_SUBFRAME_LENGTH);
        assert_eq!(plc.nb_subfr, DEFAULT_PLC_SUBFRAME_COUNT);
    }
}
