#![cfg(feature = "deep_plc")]

use crate::celt::opus_select_arch;
use crate::dnn_utils::linear_layer_from_blob;
use crate::dnn_weights::{WeightBlob, WeightError};
use crate::dred_constants::DRED_NUM_FEATURES;
use crate::nnet::{
    compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu, LinearLayer,
    ACTIVATION_LINEAR, ACTIVATION_SIGMOID, ACTIVATION_TANH,
};
use crate::pitchdnn::PITCH_MAX_PERIOD;
use alloc::vec;
use alloc::vec::Vec;
use libm::{expf, floorf, powf};

pub const FARGAN_CONT_SAMPLES: usize = 320;
const FARGAN_NB_SUBFRAMES: usize = 4;
const FARGAN_SUBFRAME_SIZE: usize = 40;
pub const FARGAN_FRAME_SIZE: usize = FARGAN_NB_SUBFRAMES * FARGAN_SUBFRAME_SIZE;
const FARGAN_DEEMPHASIS: f32 = 0.85;
const NB_BANDS: usize = 18;

#[derive(Clone, Debug, Default)]
struct FarganModel {
    pub cond_net_pembed: LinearLayer,
    pub cond_net_fdense1: LinearLayer,
    pub cond_net_fconv1: LinearLayer,
    pub cond_net_fdense2: LinearLayer,
    pub sig_net_cond_gain_dense: LinearLayer,
    pub sig_net_fwc0_conv: LinearLayer,
    pub sig_net_fwc0_glu_gate: LinearLayer,
    pub sig_net_gru1_input: LinearLayer,
    pub sig_net_gru1_recurrent: LinearLayer,
    pub sig_net_gru1_glu_gate: LinearLayer,
    pub sig_net_gru2_input: LinearLayer,
    pub sig_net_gru2_recurrent: LinearLayer,
    pub sig_net_gru2_glu_gate: LinearLayer,
    pub sig_net_gru3_input: LinearLayer,
    pub sig_net_gru3_recurrent: LinearLayer,
    pub sig_net_gru3_glu_gate: LinearLayer,
    pub sig_net_skip_dense: LinearLayer,
    pub sig_net_skip_glu_gate: LinearLayer,
    pub sig_net_sig_dense_out: LinearLayer,
    pub sig_net_gain_dense_out: LinearLayer,
}

#[derive(Clone, Debug, Default)]
pub struct FarganState {
    model: FarganModel,
    arch: i32,
    cont_initialized: bool,
    deemph_mem: f32,
    pitch_buf: Vec<f32>,
    cond_conv1_state: Vec<f32>,
    fwc0_mem: Vec<f32>,
    gru1_state: Vec<f32>,
    gru2_state: Vec<f32>,
    gru3_state: Vec<f32>,
    last_period: i32,
    dense_in: Vec<f32>,
    conv1_in: Vec<f32>,
    fdense2_in: Vec<f32>,
    cond: Vec<f32>,
    fwc0_in: Vec<f32>,
    gru1_in: Vec<f32>,
    gru2_in: Vec<f32>,
    gru3_in: Vec<f32>,
    pred: Vec<f32>,
    prev: Vec<f32>,
    pitch_gate: Vec<f32>,
    skip_cat: Vec<f32>,
    skip_out: Vec<f32>,
    glu_in: Vec<f32>,
}

impl FarganState {
    pub fn new() -> Self {
        Self {
            arch: opus_select_arch(),
            ..Self::default()
        }
    }

    pub fn reset(&mut self) {
        self.cont_initialized = false;
        self.deemph_mem = 0.0;
        self.last_period = 0;
        self.pitch_buf.fill(0.0);
        self.cond_conv1_state.fill(0.0);
        self.fwc0_mem.fill(0.0);
        self.gru1_state.fill(0.0);
        self.gru2_state.fill(0.0);
        self.gru3_state.fill(0.0);
    }

    pub fn load_model(&mut self, data: &[u8]) -> Result<(), WeightError> {
        let blob = WeightBlob::parse(data)?;
        let mut model = FarganModel::default();
        init_fargan_from_weights(&mut model, &blob)?;
        self.model = model;
        self.resize_buffers()?;
        self.reset();
        Ok(())
    }

    pub fn fargan_cont(&mut self, pcm0: &[f32], features0: &[f32]) {
        let mut cond = vec![0.0f32; self.model.cond_net_fdense2.nb_outputs];
        let mut x0 = [0.0f32; FARGAN_CONT_SAMPLES];
        let mut dummy = [0.0f32; FARGAN_SUBFRAME_SIZE];
        let mut period = 0;

        for idx in 0..5 {
            let start = idx * DRED_NUM_FEATURES;
            let features = &features0[start..start + DRED_NUM_FEATURES];
            self.last_period = period;
            period = period_from_features(features);
            self.compute_fargan_cond(&mut cond, features, period);
        }

        x0[0] = 0.0;
        for i in 1..FARGAN_CONT_SAMPLES {
            x0[i] = pcm0[i] - FARGAN_DEEMPHASIS * pcm0[i - 1];
        }

        let base = PITCH_MAX_PERIOD - FARGAN_FRAME_SIZE;
        self.pitch_buf[base..base + FARGAN_FRAME_SIZE].copy_from_slice(&x0[..FARGAN_FRAME_SIZE]);
        self.cont_initialized = true;

        let cond_size = self.cond_size();
        for i in 0..FARGAN_NB_SUBFRAMES {
            let start = i * cond_size;
            self.run_fargan_subframe(&mut dummy, &cond[start..start + cond_size], self.last_period);
            let src_start = FARGAN_FRAME_SIZE + i * FARGAN_SUBFRAME_SIZE;
            let dst_start = PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE;
            self.pitch_buf[dst_start..dst_start + FARGAN_SUBFRAME_SIZE]
                .copy_from_slice(&x0[src_start..src_start + FARGAN_SUBFRAME_SIZE]);
        }

        self.deemph_mem = pcm0[FARGAN_CONT_SAMPLES - 1];
    }

    pub fn fargan_synthesize(&mut self, pcm: &mut [f32], features: &[f32]) {
        let period = period_from_features(features);
        let mut cond = vec![0.0f32; self.model.cond_net_fdense2.nb_outputs];
        self.compute_fargan_cond(&mut cond, features, period);

        let cond_size = self.cond_size();
        for subframe in 0..FARGAN_NB_SUBFRAMES {
            let start = subframe * cond_size;
            let out_start = subframe * FARGAN_SUBFRAME_SIZE;
            self.run_fargan_subframe(
                &mut pcm[out_start..out_start + FARGAN_SUBFRAME_SIZE],
                &cond[start..start + cond_size],
                self.last_period,
            );
        }
        self.last_period = period;
    }

    pub fn fargan_synthesize_int(&mut self, pcm: &mut [i16], features: &[f32]) {
        let mut fpcm = [0.0f32; FARGAN_FRAME_SIZE];
        self.fargan_synthesize(&mut fpcm, features);
        for (dst, &src) in pcm.iter_mut().zip(fpcm.iter()) {
            let scaled = (src * 32768.0).clamp(-32767.0, 32767.0);
            *dst = floorf(0.5 + scaled) as i16;
        }
    }

    fn cond_size(&self) -> usize {
        self.model.cond_net_fdense2.nb_outputs / FARGAN_NB_SUBFRAMES
    }

    fn resize_buffers(&mut self) -> Result<(), WeightError> {
        if self.model.cond_net_fdense2.nb_outputs % FARGAN_NB_SUBFRAMES != 0 {
            return Err(WeightError::InvalidBlob);
        }

        let cond_size = self.cond_size();
        let sig_net_input_size = cond_size + 2 * FARGAN_SUBFRAME_SIZE + 4;
        if self.model.sig_net_gru1_input.nb_inputs
            != self.model.sig_net_fwc0_conv.nb_outputs + 2 * FARGAN_SUBFRAME_SIZE
        {
            return Err(WeightError::InvalidBlob);
        }
        if self.model.sig_net_gru2_input.nb_inputs
            != self.model.sig_net_gru1_recurrent.nb_inputs + 2 * FARGAN_SUBFRAME_SIZE
        {
            return Err(WeightError::InvalidBlob);
        }
        if self.model.sig_net_gru3_input.nb_inputs
            != self.model.sig_net_gru2_recurrent.nb_inputs + 2 * FARGAN_SUBFRAME_SIZE
        {
            return Err(WeightError::InvalidBlob);
        }
        if self.model.sig_net_skip_glu_gate.nb_outputs != self.model.sig_net_skip_dense.nb_outputs {
            return Err(WeightError::InvalidBlob);
        }
        if self.model.sig_net_sig_dense_out.nb_inputs != self.model.sig_net_skip_dense.nb_outputs {
            return Err(WeightError::InvalidBlob);
        }

        let cond_dense_in = DRED_NUM_FEATURES + self.model.cond_net_pembed.nb_outputs;
        self.dense_in.resize(cond_dense_in, 0.0);
        self.conv1_in
            .resize(self.model.cond_net_fdense1.nb_outputs, 0.0);
        self.fdense2_in
            .resize(self.model.cond_net_fconv1.nb_outputs, 0.0);
        self.cond.resize(self.model.cond_net_fdense2.nb_outputs, 0.0);

        self.fwc0_in.resize(sig_net_input_size, 0.0);
        self.gru1_in.resize(
            self.model.sig_net_fwc0_conv.nb_outputs + 2 * FARGAN_SUBFRAME_SIZE,
            0.0,
        );
        self.gru2_in.resize(
            self.model.sig_net_gru1_recurrent.nb_inputs + 2 * FARGAN_SUBFRAME_SIZE,
            0.0,
        );
        self.gru3_in.resize(
            self.model.sig_net_gru2_recurrent.nb_inputs + 2 * FARGAN_SUBFRAME_SIZE,
            0.0,
        );
        self.pred.resize(FARGAN_SUBFRAME_SIZE + 4, 0.0);
        self.prev.resize(FARGAN_SUBFRAME_SIZE, 0.0);
        self.pitch_gate
            .resize(self.model.sig_net_gain_dense_out.nb_outputs, 0.0);

        let skip_cat_len = self.model.sig_net_gru1_recurrent.nb_inputs
            + self.model.sig_net_gru2_recurrent.nb_inputs
            + self.model.sig_net_gru3_recurrent.nb_inputs
            + self.model.sig_net_fwc0_conv.nb_outputs
            + 2 * FARGAN_SUBFRAME_SIZE;
        self.skip_cat.resize(skip_cat_len, 0.0);
        self.skip_out
            .resize(self.model.sig_net_skip_dense.nb_outputs, 0.0);
        let max_glu = self
            .model
            .sig_net_fwc0_glu_gate
            .nb_outputs
            .max(self.model.sig_net_skip_glu_gate.nb_outputs);
        self.glu_in.resize(max_glu, 0.0);

        let cond_state_len = self.model.cond_net_fconv1.nb_inputs
            .checked_sub(self.model.cond_net_fdense1.nb_outputs)
            .ok_or(WeightError::InvalidBlob)?;
        self.cond_conv1_state.resize(cond_state_len, 0.0);

        let fwc0_state_len = self.model.sig_net_fwc0_conv.nb_inputs
            .checked_sub(sig_net_input_size)
            .ok_or(WeightError::InvalidBlob)?;
        self.fwc0_mem.resize(fwc0_state_len, 0.0);

        self.pitch_buf.resize(PITCH_MAX_PERIOD, 0.0);
        self.gru1_state
            .resize(self.model.sig_net_gru1_recurrent.nb_inputs, 0.0);
        self.gru2_state
            .resize(self.model.sig_net_gru2_recurrent.nb_inputs, 0.0);
        self.gru3_state
            .resize(self.model.sig_net_gru3_recurrent.nb_inputs, 0.0);

        Ok(())
    }

    fn compute_fargan_cond(&mut self, cond: &mut [f32], features: &[f32], period: i32) {
        let embed_dim = self.model.cond_net_pembed.nb_outputs;
        let embed_count = self.model.cond_net_pembed.nb_inputs;
        let period_idx = (period - 32).clamp(0, embed_count as i32 - 1) as usize;

        let embed_start = period_idx * embed_dim;
        let embed_end = embed_start + embed_dim;
        if let Some(weights) = self.model.cond_net_pembed.float_weights {
            self.dense_in[DRED_NUM_FEATURES..DRED_NUM_FEATURES + embed_dim]
                .copy_from_slice(&weights[embed_start..embed_end]);
        } else {
            self.dense_in[DRED_NUM_FEATURES..DRED_NUM_FEATURES + embed_dim].fill(0.0);
        }
        self.dense_in[..DRED_NUM_FEATURES].copy_from_slice(features);

        compute_generic_dense(
            &self.model.cond_net_fdense1,
            &mut self.conv1_in[..self.model.cond_net_fdense1.nb_outputs],
            &self.dense_in[..self.model.cond_net_fdense1.nb_inputs],
            ACTIVATION_TANH,
            self.arch,
        );
        compute_generic_conv1d(
            &self.model.cond_net_fconv1,
            &mut self.fdense2_in[..self.model.cond_net_fconv1.nb_outputs],
            &mut self.cond_conv1_state,
            &self.conv1_in[..self.model.cond_net_fdense1.nb_outputs],
            self.model.cond_net_fdense1.nb_outputs,
            ACTIVATION_TANH,
            self.arch,
        );
        compute_generic_dense(
            &self.model.cond_net_fdense2,
            &mut cond[..self.model.cond_net_fdense2.nb_outputs],
            &self.fdense2_in[..self.model.cond_net_fconv1.nb_outputs],
            ACTIVATION_TANH,
            self.arch,
        );
    }

    fn run_fargan_subframe(&mut self, pcm: &mut [f32], cond: &[f32], period: i32) {
        debug_assert!(self.cont_initialized);

        let mut gain = [0.0f32; 1];
        compute_generic_dense(
            &self.model.sig_net_cond_gain_dense,
            &mut gain,
            cond,
            ACTIVATION_LINEAR,
            self.arch,
        );
        let gain = expf(gain[0]);
        let gain_inv = 1.0 / (1.0e-5 + gain);

        let mut pos = PITCH_MAX_PERIOD as i32 - period - 2;
        for i in 0..self.pred.len() {
            let sample = self.pitch_buf[pos.max(0) as usize];
            self.pred[i] = (gain_inv * sample).clamp(-1.0, 1.0);
            pos += 1;
            if pos == PITCH_MAX_PERIOD as i32 {
                pos -= period;
            }
        }

        let prev_start = PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE;
        for i in 0..FARGAN_SUBFRAME_SIZE {
            self.prev[i] = (gain_inv * self.pitch_buf[prev_start + i]).clamp(-1.0, 1.0);
        }

        let cond_size = self.cond_size();
        self.fwc0_in[..cond_size].copy_from_slice(cond);
        self.fwc0_in[cond_size..cond_size + self.pred.len()].copy_from_slice(&self.pred);
        let prev_offset = cond_size + self.pred.len();
        self.fwc0_in[prev_offset..prev_offset + self.prev.len()].copy_from_slice(&self.prev);

        compute_generic_conv1d(
            &self.model.sig_net_fwc0_conv,
            &mut self.gru1_in[..self.model.sig_net_fwc0_conv.nb_outputs],
            &mut self.fwc0_mem,
            &self.fwc0_in,
            self.fwc0_in.len(),
            ACTIVATION_TANH,
            self.arch,
        );
        let glu_len = self.model.sig_net_fwc0_glu_gate.nb_outputs;
        self.glu_in[..glu_len].copy_from_slice(&self.gru1_in[..glu_len]);
        compute_glu(
            &self.model.sig_net_fwc0_glu_gate,
            &mut self.gru1_in[..glu_len],
            &self.glu_in[..glu_len],
            self.arch,
        );

        compute_generic_dense(
            &self.model.sig_net_gain_dense_out,
            &mut self.pitch_gate,
            &self.gru1_in[..self.model.sig_net_fwc0_conv.nb_outputs],
            ACTIVATION_SIGMOID,
            self.arch,
        );

        for i in 0..FARGAN_SUBFRAME_SIZE {
            let idx = self.model.sig_net_fwc0_glu_gate.nb_outputs + i;
            self.gru1_in[idx] = self.pitch_gate[0] * self.pred[i + 2];
        }
        let prev_start = self.model.sig_net_fwc0_glu_gate.nb_outputs + FARGAN_SUBFRAME_SIZE;
        self.gru1_in[prev_start..prev_start + FARGAN_SUBFRAME_SIZE].copy_from_slice(&self.prev);

        compute_generic_gru(
            &self.model.sig_net_gru1_input,
            &self.model.sig_net_gru1_recurrent,
            &mut self.gru1_state,
            &self.gru1_in[..self.model.sig_net_gru1_input.nb_inputs],
            self.arch,
        );
        compute_glu(
            &self.model.sig_net_gru1_glu_gate,
            &mut self.gru2_in[..self.model.sig_net_gru1_recurrent.nb_inputs],
            &self.gru1_state,
            self.arch,
        );

        for i in 0..FARGAN_SUBFRAME_SIZE {
            let idx = self.model.sig_net_gru1_recurrent.nb_inputs + i;
            self.gru2_in[idx] = self.pitch_gate[1] * self.pred[i + 2];
        }
        let prev_start = self.model.sig_net_gru1_recurrent.nb_inputs + FARGAN_SUBFRAME_SIZE;
        self.gru2_in[prev_start..prev_start + FARGAN_SUBFRAME_SIZE].copy_from_slice(&self.prev);

        compute_generic_gru(
            &self.model.sig_net_gru2_input,
            &self.model.sig_net_gru2_recurrent,
            &mut self.gru2_state,
            &self.gru2_in[..self.model.sig_net_gru2_input.nb_inputs],
            self.arch,
        );
        compute_glu(
            &self.model.sig_net_gru2_glu_gate,
            &mut self.gru3_in[..self.model.sig_net_gru2_recurrent.nb_inputs],
            &self.gru2_state,
            self.arch,
        );

        for i in 0..FARGAN_SUBFRAME_SIZE {
            let idx = self.model.sig_net_gru2_recurrent.nb_inputs + i;
            self.gru3_in[idx] = self.pitch_gate[2] * self.pred[i + 2];
        }
        let prev_start = self.model.sig_net_gru2_recurrent.nb_inputs + FARGAN_SUBFRAME_SIZE;
        self.gru3_in[prev_start..prev_start + FARGAN_SUBFRAME_SIZE].copy_from_slice(&self.prev);

        compute_generic_gru(
            &self.model.sig_net_gru3_input,
            &self.model.sig_net_gru3_recurrent,
            &mut self.gru3_state,
            &self.gru3_in[..self.model.sig_net_gru3_input.nb_inputs],
            self.arch,
        );
        let skip_offset = self.model.sig_net_gru1_recurrent.nb_inputs
            + self.model.sig_net_gru2_recurrent.nb_inputs;
        compute_glu(
            &self.model.sig_net_gru3_glu_gate,
            &mut self.skip_cat[skip_offset..skip_offset + self.model.sig_net_gru3_recurrent.nb_inputs],
            &self.gru3_state,
            self.arch,
        );

        self.skip_cat[..self.model.sig_net_gru1_recurrent.nb_inputs]
            .copy_from_slice(&self.gru2_in[..self.model.sig_net_gru1_recurrent.nb_inputs]);
        let offset = self.model.sig_net_gru1_recurrent.nb_inputs;
        self.skip_cat[offset..offset + self.model.sig_net_gru2_recurrent.nb_inputs]
            .copy_from_slice(&self.gru3_in[..self.model.sig_net_gru2_recurrent.nb_inputs]);
        let offset = offset + self.model.sig_net_gru2_recurrent.nb_inputs
            + self.model.sig_net_gru3_recurrent.nb_inputs;
        self.skip_cat[offset..offset + self.model.sig_net_fwc0_conv.nb_outputs]
            .copy_from_slice(&self.gru1_in[..self.model.sig_net_fwc0_conv.nb_outputs]);
        let offset = offset + self.model.sig_net_fwc0_conv.nb_outputs;
        for i in 0..FARGAN_SUBFRAME_SIZE {
            self.skip_cat[offset + i] = self.pitch_gate[3] * self.pred[i + 2];
        }
        let offset = offset + FARGAN_SUBFRAME_SIZE;
        self.skip_cat[offset..offset + FARGAN_SUBFRAME_SIZE].copy_from_slice(&self.prev);

        compute_generic_dense(
            &self.model.sig_net_skip_dense,
            &mut self.skip_out,
            &self.skip_cat,
            ACTIVATION_TANH,
            self.arch,
        );
        let skip_len = self.model.sig_net_skip_glu_gate.nb_outputs;
        self.glu_in[..skip_len].copy_from_slice(&self.skip_out[..skip_len]);
        compute_glu(
            &self.model.sig_net_skip_glu_gate,
            &mut self.skip_out[..skip_len],
            &self.glu_in[..skip_len],
            self.arch,
        );

        compute_generic_dense(
            &self.model.sig_net_sig_dense_out,
            pcm,
            &self.skip_out,
            ACTIVATION_TANH,
            self.arch,
        );
        for sample in pcm.iter_mut() {
            *sample *= gain;
        }

        self.pitch_buf
            .copy_within(FARGAN_SUBFRAME_SIZE.., 0);
        let start = PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE;
        self.pitch_buf[start..start + FARGAN_SUBFRAME_SIZE].copy_from_slice(pcm);
        self.fargan_deemphasis(pcm);
    }

    fn fargan_deemphasis(&mut self, pcm: &mut [f32]) {
        for sample in pcm.iter_mut() {
            *sample += FARGAN_DEEMPHASIS * self.deemph_mem;
            self.deemph_mem = *sample;
        }
    }
}

fn period_from_features(features: &[f32]) -> i32 {
    let pitch = features[NB_BANDS] + 1.5;
    let pow = powf(2.0, (1.0 / 60.0) * (pitch * 60.0));
    floorf(0.5 + 256.0 / pow) as i32
}

fn init_fargan_from_weights(
    model: &mut FarganModel,
    blob: &WeightBlob<'_>,
) -> Result<(), WeightError> {
    model.cond_net_pembed = linear_layer_from_blob(
        blob,
        Some("cond_net_pembed_bias"),
        None,
        None,
        Some("cond_net_pembed_weights_float"),
        None,
        None,
        None,
        None,
        None,
    )?;
    model.cond_net_fdense1 = linear_layer_from_blob(
        blob,
        Some("cond_net_fdense1_bias"),
        None,
        None,
        Some("cond_net_fdense1_weights_float"),
        None,
        None,
        None,
        Some(DRED_NUM_FEATURES + model.cond_net_pembed.nb_outputs),
        None,
    )?;
    model.cond_net_fconv1 = linear_layer_from_blob(
        blob,
        Some("cond_net_fconv1_bias"),
        None,
        None,
        Some("cond_net_fconv1_weights_float"),
        None,
        None,
        None,
        None,
        None,
    )?;
    model.cond_net_fdense2 = linear_layer_from_blob(
        blob,
        Some("cond_net_fdense2_bias"),
        None,
        None,
        Some("cond_net_fdense2_weights_float"),
        None,
        None,
        None,
        Some(model.cond_net_fconv1.nb_outputs),
        None,
    )?;

    model.sig_net_cond_gain_dense = linear_layer_from_blob(
        blob,
        Some("sig_net_cond_gain_dense_bias"),
        None,
        None,
        Some("sig_net_cond_gain_dense_weights_float"),
        None,
        None,
        None,
        Some(model.cond_net_fdense2.nb_outputs / FARGAN_NB_SUBFRAMES),
        Some(1),
    )?;
    model.sig_net_fwc0_conv = linear_layer_from_blob(
        blob,
        Some("sig_net_fwc0_conv_bias"),
        None,
        None,
        Some("sig_net_fwc0_conv_weights_float"),
        None,
        None,
        None,
        None,
        None,
    )?;
    model.sig_net_fwc0_glu_gate = linear_layer_from_blob(
        blob,
        Some("sig_net_fwc0_glu_gate_bias"),
        None,
        None,
        Some("sig_net_fwc0_glu_gate_weights_float"),
        None,
        None,
        None,
        Some(model.sig_net_fwc0_conv.nb_outputs),
        Some(model.sig_net_fwc0_conv.nb_outputs),
    )?;

    model.sig_net_gru1_input = linear_layer_from_blob(
        blob,
        None,
        None,
        None,
        Some("sig_net_gru1_input_weights_float"),
        None,
        None,
        None,
        None,
        None,
    )?;
    if model.sig_net_gru1_input.nb_outputs % 3 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let gru1_units = model.sig_net_gru1_input.nb_outputs / 3;
    model.sig_net_gru1_recurrent = linear_layer_from_blob(
        blob,
        None,
        None,
        None,
        Some("sig_net_gru1_recurrent_weights_float"),
        None,
        None,
        None,
        Some(gru1_units),
        Some(model.sig_net_gru1_input.nb_outputs),
    )?;
    model.sig_net_gru1_glu_gate = linear_layer_from_blob(
        blob,
        Some("sig_net_gru1_glu_gate_bias"),
        None,
        None,
        Some("sig_net_gru1_glu_gate_weights_float"),
        None,
        None,
        None,
        Some(gru1_units),
        Some(gru1_units),
    )?;

    model.sig_net_gru2_input = linear_layer_from_blob(
        blob,
        None,
        None,
        None,
        Some("sig_net_gru2_input_weights_float"),
        None,
        None,
        None,
        None,
        None,
    )?;
    if model.sig_net_gru2_input.nb_outputs % 3 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let gru2_units = model.sig_net_gru2_input.nb_outputs / 3;
    model.sig_net_gru2_recurrent = linear_layer_from_blob(
        blob,
        None,
        None,
        None,
        Some("sig_net_gru2_recurrent_weights_float"),
        None,
        None,
        None,
        Some(gru2_units),
        Some(model.sig_net_gru2_input.nb_outputs),
    )?;
    model.sig_net_gru2_glu_gate = linear_layer_from_blob(
        blob,
        Some("sig_net_gru2_glu_gate_bias"),
        None,
        None,
        Some("sig_net_gru2_glu_gate_weights_float"),
        None,
        None,
        None,
        Some(gru2_units),
        Some(gru2_units),
    )?;

    model.sig_net_gru3_input = linear_layer_from_blob(
        blob,
        None,
        None,
        None,
        Some("sig_net_gru3_input_weights_float"),
        None,
        None,
        None,
        None,
        None,
    )?;
    if model.sig_net_gru3_input.nb_outputs % 3 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let gru3_units = model.sig_net_gru3_input.nb_outputs / 3;
    model.sig_net_gru3_recurrent = linear_layer_from_blob(
        blob,
        None,
        None,
        None,
        Some("sig_net_gru3_recurrent_weights_float"),
        None,
        None,
        None,
        Some(gru3_units),
        Some(model.sig_net_gru3_input.nb_outputs),
    )?;
    model.sig_net_gru3_glu_gate = linear_layer_from_blob(
        blob,
        Some("sig_net_gru3_glu_gate_bias"),
        None,
        None,
        Some("sig_net_gru3_glu_gate_weights_float"),
        None,
        None,
        None,
        Some(gru3_units),
        Some(gru3_units),
    )?;

    model.sig_net_skip_glu_gate = linear_layer_from_blob(
        blob,
        Some("sig_net_skip_glu_gate_bias"),
        None,
        None,
        Some("sig_net_skip_glu_gate_weights_float"),
        None,
        None,
        None,
        None,
        None,
    )?;
    model.sig_net_skip_dense = linear_layer_from_blob(
        blob,
        Some("sig_net_skip_dense_bias"),
        None,
        None,
        Some("sig_net_skip_dense_weights_float"),
        None,
        None,
        None,
        None,
        None,
    )?;
    model.sig_net_sig_dense_out = linear_layer_from_blob(
        blob,
        Some("sig_net_sig_dense_out_bias"),
        None,
        None,
        Some("sig_net_sig_dense_out_weights_float"),
        None,
        None,
        None,
        Some(model.sig_net_skip_dense.nb_outputs),
        Some(FARGAN_SUBFRAME_SIZE),
    )?;
    model.sig_net_gain_dense_out = linear_layer_from_blob(
        blob,
        Some("sig_net_gain_dense_out_bias"),
        None,
        None,
        Some("sig_net_gain_dense_out_weights_float"),
        None,
        None,
        None,
        Some(model.sig_net_fwc0_conv.nb_outputs),
        Some(4),
    )?;

    Ok(())
}
