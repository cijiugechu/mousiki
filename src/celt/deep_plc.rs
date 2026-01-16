#![allow(dead_code)]

//! Neural PLC helpers used by the decoder when `ENABLE_DEEP_PLC` is active.
//!
//! The reference implementation relies on an auxiliary LPCNet model to refine
//! packet loss concealment when the decoder complexity is high enough.  The
//! routines here mirror the small pieces of that pipeline that are required by
//! `celt_decode_lost()`.  The full neural PLC stack has many moving parts; this
//! module focuses on the state update helper so that future ports can integrate
//! the remaining neural components incrementally.

use alloc::vec::Vec;
use crate::celt::celt_decoder::DECODE_BUFFER_SIZE;
use crate::celt::opus_select_arch;
use crate::celt::float_cast::float2int;
use crate::celt::types::CeltSig;
use crate::celt::{KissFftCpx, KissFftState};
use crate::dred_constants::DRED_NUM_FEATURES;
use crate::fargan::{FarganState, FARGAN_CONT_SAMPLES};
use crate::lpcnet_enc::{lpcnet_compute_single_frame_features_float, LpcNetEncState};
use crate::nnet::{compute_generic_dense, compute_generic_gru, ACTIVATION_LINEAR, ACTIVATION_TANH};
use crate::plc_model::PlcModel;
use libm::{cosf, log10f, powf, sqrt, sqrtf};

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

/// Maximum number of queued FEC feature frames.
const PLC_MAX_FEC: usize = 100;

/// Pre-emphasis constant shared with the LPCNet helpers.
pub(crate) const PREEMPHASIS: f32 = 0.85;

const NB_BANDS: usize = 18;
const NB_FEATURES: usize = DRED_NUM_FEATURES;
const NB_TOTAL_FEATURES: usize = 36;
const LPC_ORDER: usize = 16;
const WINDOW_SIZE_5MS: usize = 4;
const OVERLAP_SIZE: usize = PLC_FRAME_SIZE;
const WINDOW_SIZE: usize = PLC_FRAME_SIZE + OVERLAP_SIZE;
const FREQ_SIZE: usize = WINDOW_SIZE / 2 + 1;
const PLC_FEATURES_LEN: usize = 2 * NB_BANDS + NB_FEATURES + 1;

const MAX_FRAME_SIZE: usize = 384;
const FIND_LPC_COND_FAC: f64 = 1.0e-5;

const EBAND_5MS: [i16; NB_BANDS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];

const ATT_TABLE: [f32; 10] = [0.0, 0.0, -0.2, -0.2, -0.4, -0.4, -0.8, -0.8, -1.6, -1.6];

/// Order of the sinc-based resampler used when converting from 48 kHz to 16 kHz.
pub(crate) const SINC_ORDER: usize = 48;

/// Low-pass filter used to resample the decoder history to 16 kHz.
///
/// Mirrors the coefficients embedded in `celt_decoder.c`.
pub(crate) const SINC_FILTER: [f32; SINC_ORDER + 1] = [
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

#[derive(Debug, Clone, Default)]
pub(crate) struct PlcNetState {
    pub gru1_state: Vec<f32>,
    pub gru2_state: Vec<f32>,
}

impl PlcNetState {
    pub fn resize(&mut self, gru1_len: usize, gru2_len: usize) {
        self.gru1_state.resize(gru1_len, 0.0);
        self.gru2_state.resize(gru2_len, 0.0);
    }

    pub fn reset(&mut self) {
        self.gru1_state.fill(0.0);
        self.gru2_state.fill(0.0);
    }

    pub fn copy_from(&mut self, other: &Self) {
        if self.gru1_state.len() != other.gru1_state.len()
            || self.gru2_state.len() != other.gru2_state.len()
        {
            self.resize(other.gru1_state.len(), other.gru2_state.len());
        }
        self.gru1_state.copy_from_slice(&other.gru1_state);
        self.gru2_state.copy_from_slice(&other.gru2_state);
    }
}

/// Minimal representation of the neural PLC state required by `update_plc_state()`.
///
/// The complete C structure stores the neural network weights, feature queues,
/// and other caches.  Only a handful of fields are touched by the state update
/// helper, so the Rust port tracks the subset that is relevant for the
/// downsampling, history maintenance, and queued FEC features performed here.
/// Additional fields will be introduced alongside future ports of the neural
/// PLC logic.
#[derive(Debug, Clone)]
pub(crate) struct LpcNetPlcState {
    /// Whether the neural PLC model has been loaded successfully.
    pub loaded: bool,
    /// PLC prediction model weights.
    pub model: PlcModel,
    /// FARGAN synthesis state.
    pub fargan: FarganState,
    /// Feature extraction state.
    pub enc: LpcNetEncState,
    /// Architecture selector for neural network helpers.
    pub arch: i32,
    /// Queue of FEC feature vectors supplied by DRED.
    pub fec: [[f32; DRED_NUM_FEATURES]; PLC_MAX_FEC],
    /// Index of the next FEC feature vector to consume.
    pub fec_read_pos: i32,
    /// Index of the next FEC feature slot to fill.
    pub fec_fill_pos: i32,
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
    /// Most recent LPCNet feature vector.
    pub features: [f32; NB_TOTAL_FEATURES],
    /// Contiguous history of the most recent feature vectors.
    pub cont_features: [f32; CONT_VECTORS * NB_FEATURES],
    /// Current PLC network state.
    pub plc_net: PlcNetState,
    /// Backup PLC network states.
    pub plc_bak: [PlcNetState; 2],
    plc_tmp: Vec<f32>,
    burg_fft: KissFftState,
    burg_dct: [f32; NB_BANDS * NB_BANDS],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlcModelError {
    BadArgument,
}

impl Default for LpcNetPlcState {
    fn default() -> Self {
        let mut burg_dct = [0.0f32; NB_BANDS * NB_BANDS];
        init_burg_dct_table(&mut burg_dct);
        Self {
            loaded: false,
            model: PlcModel::default(),
            fargan: FarganState::new(),
            enc: LpcNetEncState::default(),
            arch: opus_select_arch(),
            fec: [[0.0; DRED_NUM_FEATURES]; PLC_MAX_FEC],
            fec_read_pos: 0,
            fec_fill_pos: 0,
            fec_skip: 0,
            analysis_gap: 1,
            analysis_pos: PLC_BUF_SIZE as i32,
            predict_pos: PLC_BUF_SIZE as i32,
            pcm: [0.0; PLC_BUF_SIZE],
            loss_count: 0,
            blend: 0,
            features: [0.0; NB_TOTAL_FEATURES],
            cont_features: [0.0; CONT_VECTORS * NB_FEATURES],
            plc_net: PlcNetState::default(),
            plc_bak: [PlcNetState::default(), PlcNetState::default()],
            plc_tmp: Vec::new(),
            burg_fft: KissFftState::new(WINDOW_SIZE),
            burg_dct,
        }
    }
}

impl LpcNetPlcState {
    /// Resets the PLC state while keeping the model loaded flag intact.
    pub fn reset(&mut self) {
        self.fec.fill([0.0; DRED_NUM_FEATURES]);
        self.fec_read_pos = 0;
        self.fec_fill_pos = 0;
        self.fec_skip = 0;
        self.analysis_gap = 1;
        self.analysis_pos = PLC_BUF_SIZE as i32;
        self.predict_pos = PLC_BUF_SIZE as i32;
        self.pcm.fill(0.0);
        self.loss_count = 0;
        self.blend = 0;
        self.features.fill(0.0);
        self.cont_features.fill(0.0);
        self.plc_net.reset();
        for backup in &mut self.plc_bak {
            backup.reset();
        }
        self.plc_tmp.fill(0.0);
        self.fargan.reset();
        self.enc.reset();
    }

    /// Mirrors `lpcnet_plc_fec_clear()` by resetting the FEC queue cursors.
    pub fn fec_clear(&mut self) {
        self.fec_read_pos = 0;
        self.fec_fill_pos = 0;
        self.fec_skip = 0;
    }

    /// Mirrors `lpcnet_plc_fec_add()` by queueing a DRED feature vector.
    pub fn fec_add(&mut self, features: Option<&[f32]>) {
        let Some(features) = features else {
            self.fec_skip = self.fec_skip.saturating_add(1);
            return;
        };

        debug_assert!(
            features.len() >= DRED_NUM_FEATURES,
            "FEC features must contain DRED_NUM_FEATURES entries"
        );

        if self.fec_fill_pos as usize == PLC_MAX_FEC {
            let read_pos = self.fec_read_pos.clamp(0, self.fec_fill_pos);
            let remaining = self.fec_fill_pos.saturating_sub(read_pos);
            let remaining_usize = remaining as usize;
            if remaining_usize > 0 {
                let src_start = read_pos as usize;
                for idx in 0..remaining_usize {
                    self.fec[idx] = self.fec[src_start + idx];
                }
            }
            self.fec_fill_pos = remaining;
            self.fec_read_pos = 0;
        }

        let fill_pos = self.fec_fill_pos as usize;
        if fill_pos < PLC_MAX_FEC {
            self.fec[fill_pos].copy_from_slice(&features[..DRED_NUM_FEATURES]);
            self.fec_fill_pos += 1;
        }
    }

    /// Mirrors `lpcnet_plc_load_model()` by accepting an external model blob.
    pub fn load_model(&mut self, data: &[u8]) -> Result<(), PlcModelError> {
        if data.is_empty() {
            return Err(PlcModelError::BadArgument);
        }
        let model = PlcModel::from_weights(data).map_err(|_| PlcModelError::BadArgument)?;
        self.enc
            .load_model(data)
            .map_err(|_| PlcModelError::BadArgument)?;
        self.fargan
            .load_model(data)
            .map_err(|_| PlcModelError::BadArgument)?;

        let gru1_len = model.plc_gru1_recurrent.nb_inputs;
        let gru2_len = model.plc_gru2_recurrent.nb_inputs;
        self.plc_net.resize(gru1_len, gru2_len);
        for backup in &mut self.plc_bak {
            backup.resize(gru1_len, gru2_len);
        }
        self.plc_tmp.resize(model.plc_dense_in.nb_outputs, 0.0);
        self.model = model;
        self.loaded = true;
        self.reset();
        Ok(())
    }

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

    fn compute_plc_pred(&mut self, out: &mut [f32], input: &[f32]) {
        debug_assert!(self.loaded);
        debug_assert!(input.len() >= self.model.plc_dense_in.nb_inputs);
        debug_assert!(out.len() >= self.model.plc_dense_out.nb_outputs);

        let tmp_len = self.model.plc_dense_in.nb_outputs;
        if self.plc_tmp.len() < tmp_len {
            self.plc_tmp.resize(tmp_len, 0.0);
        }

        compute_generic_dense(
            &self.model.plc_dense_in,
            &mut self.plc_tmp[..tmp_len],
            &input[..self.model.plc_dense_in.nb_inputs],
            ACTIVATION_TANH,
            self.arch,
        );
        compute_generic_gru(
            &self.model.plc_gru1_input,
            &self.model.plc_gru1_recurrent,
            &mut self.plc_net.gru1_state,
            &self.plc_tmp[..tmp_len],
            self.arch,
        );
        compute_generic_gru(
            &self.model.plc_gru2_input,
            &self.model.plc_gru2_recurrent,
            &mut self.plc_net.gru2_state,
            &self.plc_net.gru1_state,
            self.arch,
        );
        compute_generic_dense(
            &self.model.plc_dense_out,
            out,
            &self.plc_net.gru2_state,
            ACTIVATION_LINEAR,
            self.arch,
        );
    }

    fn get_fec_or_pred(&mut self, out: &mut [f32]) -> bool {
        if self.fec_read_pos != self.fec_fill_pos && self.fec_skip == 0 {
            let read_pos = self.fec_read_pos as usize;
            out[..NB_FEATURES].copy_from_slice(&self.fec[read_pos]);
            self.fec_read_pos = self.fec_read_pos.saturating_add(1);

            let mut plc_features = [0.0f32; PLC_FEATURES_LEN];
            plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES]
                .copy_from_slice(&out[..NB_FEATURES]);
            plc_features[2 * NB_BANDS + NB_FEATURES] = -1.0;
            let mut discard = [0.0f32; NB_FEATURES];
            self.compute_plc_pred(&mut discard, &plc_features);
            true
        } else {
            let zeros = [0.0f32; PLC_FEATURES_LEN];
            self.compute_plc_pred(out, &zeros);
            if self.fec_skip > 0 {
                self.fec_skip = self.fec_skip.saturating_sub(1);
            }
            false
        }
    }

    fn shift_plc_backup(&mut self) {
        let backup = self.plc_bak[1].clone();
        self.plc_bak[0].copy_from(&backup);
        self.plc_bak[1].copy_from(&self.plc_net);
    }

    fn queue_features(&mut self, features: &[f32]) {
        self.cont_features.copy_within(NB_FEATURES.., 0);
        let start = (CONT_VECTORS - 1) * NB_FEATURES;
        self.cont_features[start..start + NB_FEATURES].copy_from_slice(&features[..NB_FEATURES]);
    }

    fn burg_cepstral_analysis(&self, cepstrum: &mut [f32; 2 * NB_BANDS], x: &[f32; PLC_FRAME_SIZE]) {
        let (first, second) = cepstrum.split_at_mut(NB_BANDS);
        compute_burg_cepstrum(
            first,
            &x[..PLC_FRAME_SIZE / 2],
            &self.burg_fft,
            &self.burg_dct,
        );
        compute_burg_cepstrum(
            second,
            &x[PLC_FRAME_SIZE / 2..],
            &self.burg_fft,
            &self.burg_dct,
        );

        for i in 0..NB_BANDS {
            let c0 = cepstrum[i];
            let c1 = cepstrum[NB_BANDS + i];
            cepstrum[i] = 0.5 * (c0 + c1);
            cepstrum[NB_BANDS + i] = c0 - c1;
        }
    }

    /// Mirrors `lpcnet_plc_conceal()` by generating PLC audio from the neural model.
    pub fn lpcnet_plc_conceal(&mut self, pcm: &mut [i16]) -> i32 {
        assert_eq!(
            pcm.len(),
            PLC_FRAME_SIZE,
            "PCM frame must contain 10 ms of audio"
        );
        debug_assert!(self.loaded, "PLC conceal requested without a model");

        if self.blend == 0 {
            let mut count = 0;
            self.plc_net.copy_from(&self.plc_bak[0]);
            while self.analysis_pos + PLC_FRAME_SIZE as i32 <= PLC_BUF_SIZE as i32 {
                let mut x = [0.0f32; PLC_FRAME_SIZE];
                let start = self.analysis_pos as usize;
                for i in 0..PLC_FRAME_SIZE {
                    x[i] = 32_768.0 * self.pcm[start + i];
                }

                let mut plc_features = [0.0f32; PLC_FEATURES_LEN];
                let mut cepstrum = [0.0f32; 2 * NB_BANDS];
                self.burg_cepstral_analysis(&mut cepstrum, &x);
                plc_features[..2 * NB_BANDS].copy_from_slice(&cepstrum);
                let _ = lpcnet_compute_single_frame_features_float(
                    &mut self.enc,
                    &x,
                    &mut self.features,
                    self.arch,
                );
                let mut current_features = [0.0f32; NB_FEATURES];
                current_features.copy_from_slice(&self.features[..NB_FEATURES]);

                if (self.analysis_gap == 0 || count > 0) && self.analysis_pos >= self.predict_pos {
                    self.queue_features(&current_features);
                    plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES]
                        .copy_from_slice(&current_features);
                    plc_features[2 * NB_BANDS + NB_FEATURES] = 1.0;
                    self.shift_plc_backup();
                    let mut predicted = [0.0f32; NB_FEATURES];
                    self.compute_plc_pred(&mut predicted, &plc_features);
                    self.features[..NB_FEATURES].copy_from_slice(&predicted);
                }

                self.analysis_pos += PLC_FRAME_SIZE as i32;
                count += 1;
            }

            self.shift_plc_backup();
            let mut predicted = [0.0f32; NB_FEATURES];
            self.get_fec_or_pred(&mut predicted);
            self.features[..NB_FEATURES].copy_from_slice(&predicted);
            self.queue_features(&predicted);
            self.shift_plc_backup();
            let mut predicted = [0.0f32; NB_FEATURES];
            self.get_fec_or_pred(&mut predicted);
            self.features[..NB_FEATURES].copy_from_slice(&predicted);
            self.queue_features(&predicted);
            let cont_start = PLC_BUF_SIZE - FARGAN_CONT_SAMPLES;
            self.fargan
                .fargan_cont(&self.pcm[cont_start..], &self.cont_features);
            self.analysis_gap = 0;
        }

        self.shift_plc_backup();
        let mut predicted = [0.0f32; NB_FEATURES];
        if self.get_fec_or_pred(&mut predicted) {
            self.loss_count = 0;
        } else {
            self.loss_count = self.loss_count.saturating_add(1);
        }
        self.features[..NB_FEATURES].copy_from_slice(&predicted);

        if self.loss_count >= 10 {
            let attenuation = ATT_TABLE[9] - 2.0 * (self.loss_count - 9) as f32;
            self.features[0] = (self.features[0] + attenuation).max(-10.0);
        } else {
            let idx = self.loss_count as usize;
            self.features[0] = (self.features[0] + ATT_TABLE[idx]).max(-10.0);
        }

        self.fargan
            .fargan_synthesize_int(pcm, &self.features[..NB_FEATURES]);
        let mut current_features = [0.0f32; NB_FEATURES];
        current_features.copy_from_slice(&self.features[..NB_FEATURES]);
        self.queue_features(&current_features);

        if self.analysis_pos - PLC_FRAME_SIZE as i32 >= 0 {
            self.analysis_pos -= PLC_FRAME_SIZE as i32;
        } else {
            self.analysis_gap = 1;
        }
        self.predict_pos = PLC_BUF_SIZE as i32;

        self.pcm.copy_within(PLC_FRAME_SIZE.., 0);
        let start = PLC_BUF_SIZE - PLC_FRAME_SIZE;
        for (index, sample) in pcm.iter().enumerate() {
            self.pcm[start + index] = f32::from(*sample) * PCM_NORMALISATION;
        }

        self.blend = 1;

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

    let prev = *plc_preemphasis_mem;
    buf48k[0] += PREEMPHASIS * prev;
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
        let clamped = sum.clamp(f32::from(i16::MIN) + 1.0, f32::from(i16::MAX));
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

fn init_burg_dct_table(table: &mut [f32; NB_BANDS * NB_BANDS]) {
    let nb_bands = NB_BANDS as f32;
    let scale = sqrtf(0.5);
    for i in 0..NB_BANDS {
        for j in 0..NB_BANDS {
            let mut value = cosf((i as f32 + 0.5) * j as f32 * core::f32::consts::PI / nb_bands);
            if j == 0 {
                value *= scale;
            }
            table[i * NB_BANDS + j] = value;
        }
    }
}

fn forward_transform(
    fft: &KissFftState,
    output: &mut [KissFftCpx; FREQ_SIZE],
    input: &[f32; WINDOW_SIZE],
) {
    let mut x = [KissFftCpx::default(); WINDOW_SIZE];
    let mut y = [KissFftCpx::default(); WINDOW_SIZE];
    for (dst, &src) in x.iter_mut().zip(input.iter()) {
        dst.r = src;
        dst.i = 0.0;
    }
    fft.fft(&x, &mut y);
    output.copy_from_slice(&y[..FREQ_SIZE]);
}

fn compute_band_energy_inverse(
    band_energy: &mut [f32; NB_BANDS],
    freq: &[KissFftCpx; FREQ_SIZE],
) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize * WINDOW_SIZE_5MS;
        let band_start = EBAND_5MS[i] as usize * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = band_start + j;
            let tmp = freq[idx].r * freq[idx].r + freq[idx].i * freq[idx].i;
            let tmp = 1.0 / (tmp + 1.0e-9);
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    band_energy.copy_from_slice(&sum);
}

fn dct(out: &mut [f32; NB_BANDS], input: &[f32; NB_BANDS], dct_table: &[f32; NB_BANDS * NB_BANDS]) {
    let scale = sqrtf(2.0 / NB_BANDS as f32);
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * dct_table[j * NB_BANDS + i];
        }
        out[i] = sum * scale;
    }
}

fn compute_burg_cepstrum(
    out: &mut [f32],
    pcm: &[f32],
    fft: &KissFftState,
    dct_table: &[f32; NB_BANDS * NB_BANDS],
) {
    let len = pcm.len();
    debug_assert!(len <= PLC_FRAME_SIZE);

    let mut burg_in = [0.0f32; PLC_FRAME_SIZE];
    let mut burg_lpc = [0.0f32; LPC_ORDER];
    let mut response = [0.0f32; WINDOW_SIZE];
    let mut e_burg = [0.0f32; NB_BANDS];
    let mut freq = [KissFftCpx::default(); FREQ_SIZE];
    let mut ly = [0.0f32; NB_BANDS];

    for i in 0..len.saturating_sub(1) {
        burg_in[i] = pcm[i + 1] - PREEMPHASIS * pcm[i];
    }

    let mut energy = silk_burg_analysis(
        &mut burg_lpc,
        &burg_in[..len.saturating_sub(1)],
        1.0e-3,
        len.saturating_sub(1),
        1,
        LPC_ORDER,
    );
    let denom = (len as f32) - 2.0 * (LPC_ORDER as f32 - 1.0);
    if denom > 0.0 {
        energy /= denom;
    }

    response.fill(0.0);
    response[0] = 1.0;
    for i in 0..LPC_ORDER {
        response[i + 1] = -burg_lpc[i] * powf(0.995, (i + 1) as f32);
    }
    forward_transform(fft, &mut freq, &response);
    compute_band_energy_inverse(&mut e_burg, &freq);
    let scale = 0.45 * energy / (WINDOW_SIZE as f32 * WINDOW_SIZE as f32 * WINDOW_SIZE as f32);
    for i in 0..NB_BANDS {
        e_burg[i] *= scale;
    }

    let mut log_max = -2.0f32;
    let mut follow = -2.0f32;
    for i in 0..NB_BANDS {
        let mut value = log10f(1.0e-2 + e_burg[i]);
        value = value.max(log_max - 8.0).max(follow - 2.5);
        log_max = log_max.max(value);
        follow = (follow - 2.5).max(value);
        ly[i] = value;
    }

    dct(out.try_into().expect("NB_BANDS mismatch"), &ly, dct_table);
    out[0] -= 4.0;
}

fn silk_burg_analysis(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    order: usize,
) -> f32 {
    debug_assert!(order <= LPC_ORDER);
    debug_assert!(subfr_length * nb_subfr <= MAX_FRAME_SIZE);

    let mut c_first_row = [0.0f64; LPC_ORDER];
    let mut c_last_row = [0.0f64; LPC_ORDER];
    let mut c_af = [0.0f64; LPC_ORDER + 1];
    let mut c_ab = [0.0f64; LPC_ORDER + 1];
    let mut a_f = [0.0f64; LPC_ORDER];

    let c0 = silk_energy(x);
    for s in 0..nb_subfr {
        let offset = s * subfr_length;
        let frame = &x[offset..offset + subfr_length];
        for n in 1..=order {
            c_first_row[n - 1] += silk_inner_product(frame, &frame[n..], subfr_length - n);
        }
    }
    c_last_row.copy_from_slice(&c_first_row);

    c_af[0] = c0 + FIND_LPC_COND_FAC * c0 + 1.0e-9;
    c_ab[0] = c_af[0];
    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;

    for n in 0..order {
        for s in 0..nb_subfr {
            let offset = s * subfr_length;
            let frame = &x[offset..offset + subfr_length];
            let mut tmp1 = frame[n] as f64;
            let mut tmp2 = frame[subfr_length - n - 1] as f64;

            for k in 0..n {
                c_first_row[k] -= frame[n] as f64 * frame[n - k - 1] as f64;
                c_last_row[k] -= frame[subfr_length - n - 1] as f64
                    * frame[subfr_length - n + k] as f64;
                let atmp = a_f[k];
                tmp1 += frame[n - k - 1] as f64 * atmp;
                tmp2 += frame[subfr_length - n + k] as f64 * atmp;
            }
            for k in 0..=n {
                c_af[k] -= tmp1 * frame[n - k] as f64;
                c_ab[k] -= tmp2 * frame[subfr_length - n + k - 1] as f64;
            }
        }

        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = a_f[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }
        c_af[n + 1] = tmp1;
        c_ab[n + 1] = tmp2;

        let mut num = c_ab[n + 1];
        let mut nrg_b = c_ab[0];
        let mut nrg_f = c_af[0];
        for k in 0..n {
            let atmp = a_f[k];
            num += c_ab[n - k] * atmp;
            nrg_b += c_ab[k + 1] * atmp;
            nrg_f += c_af[k + 1] * atmp;
        }

        let mut rc = -2.0 * num / (nrg_f + nrg_b);
        let tmp = inv_gain * (1.0 - rc * rc);
        if tmp <= min_inv_gain as f64 {
            rc = sqrt(1.0 - min_inv_gain as f64 / inv_gain);
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain as f64;
            reached_max_gain = true;
        } else {
            inv_gain = tmp;
        }

        for k in 0..(n + 1) / 2 {
            let tmp1 = a_f[k];
            let tmp2 = a_f[n - k - 1];
            a_f[k] = tmp1 + rc * tmp2;
            a_f[n - k - 1] = tmp2 + rc * tmp1;
        }
        a_f[n] = rc;

        if reached_max_gain {
            for k in n + 1..order {
                a_f[k] = 0.0;
            }
            break;
        }

        for k in 0..=n + 1 {
            let idx = n + 1 - k;
            let tmp1 = c_af[k];
            c_af[k] += rc * c_ab[idx];
            c_ab[idx] += rc * tmp1;
        }
    }

    let energy = if reached_max_gain {
        for k in 0..order {
            a[k] = (-a_f[k]) as f32;
        }
        let mut c0 = c0;
        for s in 0..nb_subfr {
            let offset = s * subfr_length;
            c0 -= silk_energy(&x[offset..offset + order]);
        }
        c0 * inv_gain
    } else {
        let mut nrg_f = c_af[0];
        let mut tmp1 = 1.0f64;
        for k in 0..order {
            let atmp = a_f[k];
            nrg_f += c_af[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = (-atmp) as f32;
        }
        nrg_f - FIND_LPC_COND_FAC * c0 * tmp1
    };

    energy.max(0.0) as f32
}

fn silk_energy(data: &[f32]) -> f64 {
    data.iter().map(|&value| (value as f64) * (value as f64)).sum()
}

fn silk_inner_product(data1: &[f32], data2: &[f32], len: usize) -> f64 {
    data1
        .iter()
        .take(len)
        .zip(data2.iter().take(len))
        .map(|(&a, &b)| (a as f64) * (b as f64))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn reference_downsample(history: &[&[CeltSig]], prev_mem: f32) -> (Vec<i16>, f32) {
        let mut buf48k = [0.0f32; DECODE_BUFFER_SIZE];
        if history.len() == 1 {
            buf48k.copy_from_slice(&history[0][..DECODE_BUFFER_SIZE]);
        } else {
            for index in 0..DECODE_BUFFER_SIZE {
                buf48k[index] = 0.5 * (history[0][index] + history[1][index]);
            }
        }

        buf48k[0] += PREEMPHASIS * prev_mem;
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
            let clamped = sum.clamp(f32::from(i16::MIN) + 1.0, f32::from(i16::MAX));
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

        let (expected_pcm, expected_preemph) = reference_downsample(&[&left], 0.0);
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

        let (expected_pcm, expected_preemph) = reference_downsample(&[&left, &right], 0.0);
        assert!((preemph_mem - expected_preemph).abs() < 1e-6);

        let tail = &state.pcm[PLC_BUF_SIZE - PLC_FRAME_SIZE..];
        let start = PLC_UPDATE_SAMPLES - PLC_FRAME_SIZE;
        for (sample, expected) in tail.iter().zip(expected_pcm[start..].iter()) {
            assert!((sample - (*expected as f32) * PCM_NORMALISATION).abs() < 1e-6);
        }
    }

    #[test]
    fn fec_queue_tracks_fill_and_skip() {
        let mut state = LpcNetPlcState::default();
        let features = [1.0f32; DRED_NUM_FEATURES];

        state.fec_add(None);
        assert_eq!(state.fec_skip, 1);
        assert_eq!(state.fec_fill_pos, 0);

        state.fec_add(Some(&features));
        assert_eq!(state.fec_read_pos, 0);
        assert_eq!(state.fec_fill_pos, 1);
        assert_eq!(state.fec_skip, 1);
        assert_eq!(state.fec[0], features);

        state.fec_clear();
        assert_eq!(state.fec_read_pos, 0);
        assert_eq!(state.fec_fill_pos, 0);
        assert_eq!(state.fec_skip, 0);
    }

    #[test]
    fn load_model_rejects_empty_blob() {
        let mut state = LpcNetPlcState::default();
        assert_eq!(state.load_model(&[]), Err(PlcModelError::BadArgument));
    }

    #[test]
    fn burg_cepstral_analysis_produces_finite_values() {
        let state = LpcNetPlcState::default();
        let pcm = [0.0f32; PLC_FRAME_SIZE];
        let mut cepstrum = [0.0f32; 2 * NB_BANDS];

        state.burg_cepstral_analysis(&mut cepstrum, &pcm);

        assert!(cepstrum.iter().all(|value| value.is_finite()));
    }
}
