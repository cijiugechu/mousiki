#![allow(dead_code)]

//! Neural PLC helpers used by the decoder when `ENABLE_DEEP_PLC` is active.
//!
//! The reference implementation relies on an auxiliary LPCNet model to refine
//! packet loss concealment when the decoder complexity is high enough.  The
//! routines here mirror the small pieces of that pipeline that are required by
//! `celt_decode_lost()`.  The full neural PLC stack has many moving parts; this
//! module focuses on the state update helper so that future ports can integrate
//! the remaining neural components incrementally.

use crate::celt::celt_decoder::DECODE_BUFFER_SIZE;
use crate::celt::float_cast::float2int;
use crate::celt::types::CeltSig;

/// Number of 16 kHz samples produced per neural PLC update.
pub(crate) const PLC_FRAME_SIZE: usize = 160;

/// Number of frames fed to the neural PLC when refreshing its history.
pub(crate) const PLC_UPDATE_FRAMES: usize = 4;

/// Total number of 16 kHz samples pushed through the neural PLC update.
pub(crate) const PLC_UPDATE_SAMPLES: usize = PLC_UPDATE_FRAMES * PLC_FRAME_SIZE;

/// Number of past feature vectors retained by the neural PLC.
const CONT_VECTORS: usize = 5;

/// Size of the floating-point history buffer maintained by the neural PLC.
pub(crate) const PLC_BUF_SIZE: usize = (CONT_VECTORS + 10) * PLC_FRAME_SIZE;

/// Pre-emphasis constant shared with the LPCNet helpers.
pub(crate) const PREEMPHASIS: f32 = 0.85;

/// Order of the sinc-based resampler used when converting from 48 kHz to 16 kHz.
const SINC_ORDER: usize = 48;

/// Low-pass filter used to resample the decoder history to 16 kHz.
///
/// Mirrors the coefficients embedded in `celt_decoder.c`.
const SINC_FILTER: [f32; SINC_ORDER + 1] = [
    4.2931e-05,
    -0.000190293,
    -0.000816132,
    -0.000637162,
    0.00141662,
    0.00354764,
    0.00184368,
    -0.00428274,
    -0.00856105,
    -0.0034003,
    0.00930201,
    0.0159616,
    0.00489785,
    -0.0169649,
    -0.0259484,
    -0.00596856,
    0.0286551,
    0.0405872,
    0.00649994,
    -0.0509284,
    -0.0716655,
    -0.00665212,
    0.134336,
    0.278927,
    0.339995,
    0.278927,
    0.134336,
    -0.00665212,
    -0.0716655,
    -0.0509284,
    0.00649994,
    0.0405872,
    0.0286551,
    -0.00596856,
    -0.0259484,
    -0.0169649,
    0.00489785,
    0.0159616,
    0.00930201,
    -0.0034003,
    -0.00856105,
    -0.00428274,
    0.00184368,
    0.00354764,
    0.00141662,
    -0.000637162,
    -0.000816132,
    -0.000190293,
    4.2931e-05,
];

/// Scaling factor applied when normalising 16-bit PCM to floating point.
const PCM_NORMALISATION: f32 = 1.0 / 32_768.0;

/// Minimal representation of the neural PLC state required by `update_plc_state()`.
///
/// The complete C structure stores the neural network weights, feature queues,
/// and other caches.  Only a handful of fields are touched by the state update
/// helper, so the Rust port tracks the subset that is relevant for the
/// downsampling and history maintenance performed here.  Additional fields will
/// be introduced alongside future ports of the neural PLC logic.
#[derive(Debug, Clone)]
pub(crate) struct LpcNetPlcState {
    /// Whether the neural PLC model has been loaded successfully.
    pub loaded: bool,
    /// Index of the next FEC feature vector to consume.
    pub fec_read_pos: i32,
    /// Number of FEC frames that should be skipped.
    pub fec_skip: i32,
    /// Tracks gaps in the analysis history.
    pub analysis_gap: i32,
    /// Offset of the next analysis window within [`Self::pcm`].
    pub analysis_pos: i32,
    /// Offset of the next prediction window within [`Self::pcm`].
    pub predict_pos: i32,
    /// Rolling 16 kHz PCM history used by the neural PLC.
    pub pcm: [f32; PLC_BUF_SIZE],
    /// Number of consecutive concealed frames produced by the neural PLC.
    pub loss_count: i32,
    /// Blend factor used when merging neural PLC output with waveform PLC.
    pub blend: i32,
}

impl Default for LpcNetPlcState {
    fn default() -> Self {
        Self {
            loaded: false,
            fec_read_pos: 0,
            fec_skip: 0,
            analysis_gap: 1,
            analysis_pos: PLC_BUF_SIZE as i32,
            predict_pos: PLC_BUF_SIZE as i32,
            pcm: [0.0; PLC_BUF_SIZE],
            loss_count: 0,
            blend: 0,
        }
    }
}

impl LpcNetPlcState {
    /// Mirrors `lpcnet_plc_update()` from `dnn/lpcnet_plc.c`.
    pub fn lpcnet_plc_update(&mut self, pcm: &mut [i16]) -> i32 {
        assert_eq!(
            pcm.len(),
            PLC_FRAME_SIZE,
            "PCM frame must contain 10 ms of audio"
        );

        if self.analysis_pos - PLC_FRAME_SIZE as i32 >= 0 {
            self.analysis_pos -= PLC_FRAME_SIZE as i32;
        } else {
            self.analysis_gap = 1;
        }

        if self.predict_pos - PLC_FRAME_SIZE as i32 >= 0 {
            self.predict_pos -= PLC_FRAME_SIZE as i32;
        }

        // Shift the rolling PCM buffer left by one frame.
        self.pcm.copy_within(PLC_FRAME_SIZE.., 0);

        let start = PLC_BUF_SIZE - PLC_FRAME_SIZE;
        for (index, sample) in pcm.iter().enumerate() {
            self.pcm[start + index] = f32::from(*sample) * PCM_NORMALISATION;
        }

        self.loss_count = 0;
        self.blend = 0;

        0
    }
}

/// Updates the neural PLC state with the most recent decoder history.
///
/// The helper down-samples the 48 kHz decoder buffer to 16 kHz using a windowed
/// sinc filter, applies the same pre-emphasis as the LPCNet analysis path, and
/// feeds four 10 ms frames into the neural PLC state.  The FEC cursors are
/// preserved so that the update does not consume queued feature vectors.
pub(crate) fn update_plc_state(
    lpcnet: &mut LpcNetPlcState,
    decode_mem: &[&[CeltSig]],
    plc_preemphasis_mem: &mut f32,
) {
    if decode_mem.is_empty() || !lpcnet.loaded {
        return;
    }

    let channels = decode_mem.len();
    debug_assert!(channels == 1 || channels == 2);
    for channel in decode_mem {
        debug_assert!(channel.len() >= DECODE_BUFFER_SIZE);
    }

    let mut buf48k = [0.0f32; DECODE_BUFFER_SIZE];
    match channels {
        1 => {
            buf48k.copy_from_slice(&decode_mem[0][..DECODE_BUFFER_SIZE]);
        }
        2 => {
            let left = &decode_mem[0][..DECODE_BUFFER_SIZE];
            let right = &decode_mem[1][..DECODE_BUFFER_SIZE];
            for index in 0..DECODE_BUFFER_SIZE {
                buf48k[index] = 0.5 * (left[index] + right[index]);
            }
        }
        _ => unreachable!("decoder only supports mono or stereo histories"),
    }

    for index in 1..DECODE_BUFFER_SIZE {
        buf48k[index] += PREEMPHASIS * buf48k[index - 1];
    }

    *plc_preemphasis_mem = buf48k[DECODE_BUFFER_SIZE - 1];

    let offset = DECODE_BUFFER_SIZE - SINC_ORDER - 1 - 3 * (PLC_UPDATE_SAMPLES - 1);
    debug_assert!(
        3 * (PLC_UPDATE_SAMPLES - 1) + SINC_ORDER + offset == DECODE_BUFFER_SIZE - 1,
        "resampler offset must match the C reference"
    );

    let mut buf16k = [0i16; PLC_UPDATE_SAMPLES];
    for (frame_index, sample) in buf16k.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for tap in 0..=SINC_ORDER {
            sum += buf48k[3 * frame_index + tap + offset] * SINC_FILTER[tap];
        }
        let clamped = sum.clamp(-32_767.0, 32_767.0);
        *sample = float2int(clamped) as i16;
    }

    let saved_read_pos = lpcnet.fec_read_pos;
    let saved_skip = lpcnet.fec_skip;

    for frame in buf16k.chunks_exact_mut(PLC_FRAME_SIZE) {
        let _ = lpcnet.lpcnet_plc_update(frame);
    }

    lpcnet.fec_read_pos = saved_read_pos;
    lpcnet.fec_skip = saved_skip;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn reference_downsample(history: &[&[CeltSig]]) -> (Vec<i16>, f32) {
        let mut buf48k = [0.0f32; DECODE_BUFFER_SIZE];
        if history.len() == 1 {
            buf48k.copy_from_slice(&history[0][..DECODE_BUFFER_SIZE]);
        } else {
            for index in 0..DECODE_BUFFER_SIZE {
                buf48k[index] = 0.5 * (history[0][index] + history[1][index]);
            }
        }

        for index in 1..DECODE_BUFFER_SIZE {
            buf48k[index] += PREEMPHASIS * buf48k[index - 1];
        }

        let preemph_mem = buf48k[DECODE_BUFFER_SIZE - 1];

        let offset = DECODE_BUFFER_SIZE - SINC_ORDER - 1 - 3 * (PLC_UPDATE_SAMPLES - 1);
        let mut buf16k = vec![0i16; PLC_UPDATE_SAMPLES];
        for (frame_index, sample) in buf16k.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for tap in 0..=SINC_ORDER {
                sum += buf48k[3 * frame_index + tap + offset] * SINC_FILTER[tap];
            }
            let clamped = sum.clamp(-32_767.0, 32_767.0);
            *sample = float2int(clamped) as i16;
        }

        (buf16k, preemph_mem)
    }

    #[test]
    fn update_plc_state_refreshes_single_channel_history() {
        let mut left = vec![0.0f32; DECODE_BUFFER_SIZE];
        for (index, sample) in left.iter_mut().enumerate() {
            *sample = (index as f32).sin();
        }

        let mut state = LpcNetPlcState::default();
        state.loaded = true;
        state.fec_read_pos = 3;
        state.fec_skip = 2;
        state.analysis_pos = PLC_FRAME_SIZE as i32;
        state.predict_pos = PLC_FRAME_SIZE as i32;
        for (index, sample) in state.pcm.iter_mut().enumerate() {
            *sample = index as f32;
        }

        let original_pcm = state.pcm;

        let mut preemph_mem = 0.0;
        update_plc_state(&mut state, &[&left], &mut preemph_mem);

        assert_eq!(state.fec_read_pos, 3);
        assert_eq!(state.fec_skip, 2);
        assert_eq!(state.analysis_pos, 0);
        assert_eq!(state.predict_pos, 0);
        assert_eq!(state.loss_count, 0);
        assert_eq!(state.blend, 0);

        // Verify the PCM history shifted by the four 16 kHz frames consumed by the update.
        for (index, (after, before)) in state.pcm[..PLC_BUF_SIZE - PLC_UPDATE_SAMPLES]
            .iter()
            .zip(&original_pcm[PLC_UPDATE_SAMPLES..])
            .enumerate()
        {
            assert!(
                (after - before).abs() < 1e-6,
                "history mismatch at {}: after={} before={}",
                index,
                after,
                before
            );
        }

        let (expected_pcm, expected_preemph) = reference_downsample(&[&left]);
        assert!((preemph_mem - expected_preemph).abs() < 1e-6);

        let tail = &state.pcm[PLC_BUF_SIZE - PLC_UPDATE_SAMPLES..];
        for (sample, expected) in tail.iter().zip(expected_pcm.iter()) {
            assert!((sample - (*expected as f32) * PCM_NORMALISATION).abs() < 1e-6);
        }
    }

    #[test]
    fn update_plc_state_averages_stereo_history() {
        let mut left = vec![0.0f32; DECODE_BUFFER_SIZE];
        let mut right = vec![0.0f32; DECODE_BUFFER_SIZE];
        for index in 0..DECODE_BUFFER_SIZE {
            left[index] = index as f32;
            right[index] = (DECODE_BUFFER_SIZE - index) as f32;
        }

        let mut state = LpcNetPlcState::default();
        state.loaded = true;

        let mut preemph_mem = 0.0;
        update_plc_state(&mut state, &[&left, &right], &mut preemph_mem);

        let (expected_pcm, expected_preemph) = reference_downsample(&[&left, &right]);
        assert!((preemph_mem - expected_preemph).abs() < 1e-6);

        let tail = &state.pcm[PLC_BUF_SIZE - PLC_FRAME_SIZE..];
        for (sample, expected) in tail.iter().zip(expected_pcm.iter()) {
            assert!((sample - (*expected as f32) * PCM_NORMALISATION).abs() < 1e-6);
        }
    }
}
