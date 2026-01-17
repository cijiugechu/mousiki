use alloc::boxed::Box;
use alloc::vec::Vec;
use crate::dnn_weights::{optional_bytes, require_bytes, WeightBlob, WeightError};
use crate::nnet::{
    compute_conv2d, compute_generic_dense, compute_generic_gru, Conv2dLayer, LinearLayer,
    ACTIVATION_LINEAR, ACTIVATION_TANH,
};
use crate::pitchdnn_data::*;

pub(crate) const PITCH_MIN_PERIOD: usize = 32;
pub(crate) const PITCH_MAX_PERIOD: usize = 256;
pub(crate) const NB_XCORR_FEATURES: usize = PITCH_MAX_PERIOD - PITCH_MIN_PERIOD;
pub(crate) const PITCH_IF_MAX_FREQ: usize = 30;
pub(crate) const PITCH_IF_FEATURES: usize = 3 * PITCH_IF_MAX_FREQ - 2;

const MAX_CONV_CHANNELS: usize = 8;
const XCORR_MEM1_SIZE: usize = (NB_XCORR_FEATURES + 2) * 2;
const XCORR_MEM2_SIZE: usize = (NB_XCORR_FEATURES + 2) * 2 * MAX_CONV_CHANNELS;
const OUTPUT_BINS: usize = 180;
const CONV_KERNEL: usize = 3;

const DENSE_IF_UPSAMPLER_1_OUT_SIZE: usize = DENSE_IF_UPSAMPLER_1_BIAS.len();
const DENSE_IF_UPSAMPLER_2_OUT_SIZE: usize = DENSE_IF_UPSAMPLER_2_BIAS.len();
const DENSE_DOWNSAMPLER_OUT_SIZE: usize = DENSE_DOWNSAMPLER_BIAS.len();
const DENSE_FINAL_UPSAMPLER_OUT_SIZE: usize = DENSE_FINAL_UPSAMPLER_BIAS.len();
const GRU_1_STATE_SIZE: usize = GRU_1_RECURRENT_BIAS.len() / 3;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PitchDnn {
    pub dense_if_upsampler_1: LinearLayer,
    pub dense_if_upsampler_2: LinearLayer,
    pub dense_downsampler: LinearLayer,
    pub dense_final_upsampler: LinearLayer,
    pub gru_1_input: LinearLayer,
    pub gru_1_recurrent: LinearLayer,
    pub conv2d_1: Conv2dLayer,
    pub conv2d_2: Conv2dLayer,
}

impl PitchDnn {
    pub(crate) fn new() -> Self {
        let mut model = Self::default();
        init_pitchdnn(&mut model);
        model
    }

    pub(crate) fn from_weights(data: &[u8]) -> Result<Self, WeightError> {
        let blob = WeightBlob::parse(data)?;
        let mut model = Self::default();
        init_pitchdnn_from_weights(&mut model, &blob)?;
        Ok(model)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PitchDnnState {
    pub model: PitchDnn,
    pub gru_state: [f32; GRU_1_STATE_SIZE],
    pub xcorr_mem1: [f32; XCORR_MEM1_SIZE],
    pub xcorr_mem2: [f32; XCORR_MEM2_SIZE],
}

impl Default for PitchDnnState {
    fn default() -> Self {
        Self {
            model: PitchDnn::new(),
            gru_state: [0.0; GRU_1_STATE_SIZE],
            xcorr_mem1: [0.0; XCORR_MEM1_SIZE],
            xcorr_mem2: [0.0; XCORR_MEM2_SIZE],
        }
    }
}

impl PitchDnnState {
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.gru_state.fill(0.0);
        self.xcorr_mem1.fill(0.0);
        self.xcorr_mem2.fill(0.0);
    }

    pub fn load_model(&mut self, data: &[u8]) -> Result<(), WeightError> {
        self.model = PitchDnn::from_weights(data)?;
        Ok(())
    }
}

pub(crate) fn compute_pitchdnn(
    state: &mut PitchDnnState,
    if_features: &[f32],
    xcorr_features: &[f32],
    arch: i32,
) -> f32 {
    debug_assert_eq!(if_features.len(), PITCH_IF_FEATURES);
    debug_assert_eq!(xcorr_features.len(), NB_XCORR_FEATURES);

    let mut if1_out = [0.0f32; DENSE_IF_UPSAMPLER_1_OUT_SIZE];
    let mut downsampler_in = [0.0f32; NB_XCORR_FEATURES + DENSE_IF_UPSAMPLER_2_OUT_SIZE];
    let mut downsampler_out = [0.0f32; DENSE_DOWNSAMPLER_OUT_SIZE];
    let mut conv1_tmp1 = [0.0f32; (NB_XCORR_FEATURES + 2) * MAX_CONV_CHANNELS];
    let mut conv1_tmp2 = [0.0f32; (NB_XCORR_FEATURES + 2) * MAX_CONV_CHANNELS];
    let mut output = [0.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];

    compute_generic_dense(
        &state.model.dense_if_upsampler_1,
        &mut if1_out[..state.model.dense_if_upsampler_1.nb_outputs],
        if_features,
        ACTIVATION_TANH,
        arch,
    );
    compute_generic_dense(
        &state.model.dense_if_upsampler_2,
        &mut downsampler_in[NB_XCORR_FEATURES
            ..NB_XCORR_FEATURES + state.model.dense_if_upsampler_2.nb_outputs],
        &if1_out[..state.model.dense_if_upsampler_1.nb_outputs],
        ACTIVATION_TANH,
        arch,
    );

    conv1_tmp1[1..1 + NB_XCORR_FEATURES].copy_from_slice(xcorr_features);
    let mem1_len = (state.model.conv2d_1.ktime - 1)
        * state.model.conv2d_1.in_channels
        * (NB_XCORR_FEATURES + state.model.conv2d_1.kheight - 1);
    let mem2_len = (state.model.conv2d_2.ktime - 1)
        * state.model.conv2d_2.in_channels
        * (NB_XCORR_FEATURES + state.model.conv2d_2.kheight - 1);
    debug_assert!(mem1_len <= state.xcorr_mem1.len());
    debug_assert!(mem2_len <= state.xcorr_mem2.len());

    compute_conv2d(
        &state.model.conv2d_1,
        &mut conv1_tmp2[1..],
        &mut state.xcorr_mem1[..mem1_len],
        &conv1_tmp1,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES + 2,
        ACTIVATION_TANH,
        arch,
    );
    compute_conv2d(
        &state.model.conv2d_2,
        &mut downsampler_in[..NB_XCORR_FEATURES],
        &mut state.xcorr_mem2[..mem2_len],
        &conv1_tmp2,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES,
        ACTIVATION_TANH,
        arch,
    );

    compute_generic_dense(
        &state.model.dense_downsampler,
        &mut downsampler_out[..state.model.dense_downsampler.nb_outputs],
        &downsampler_in[..NB_XCORR_FEATURES + state.model.dense_if_upsampler_2.nb_outputs],
        ACTIVATION_TANH,
        arch,
    );
    compute_generic_gru(
        &state.model.gru_1_input,
        &state.model.gru_1_recurrent,
        &mut state.gru_state[..state.model.gru_1_recurrent.nb_inputs],
        &downsampler_out[..state.model.dense_downsampler.nb_outputs],
        arch,
    );
    compute_generic_dense(
        &state.model.dense_final_upsampler,
        &mut output[..state.model.dense_final_upsampler.nb_outputs],
        &state.gru_state[..state.model.gru_1_recurrent.nb_inputs],
        ACTIVATION_LINEAR,
        arch,
    );

    let bins = OUTPUT_BINS.min(state.model.dense_final_upsampler.nb_outputs);
    let mut pos = 0usize;
    let mut maxval = -1.0f32;
    for i in 0..bins {
        if output[i] > maxval {
            pos = i;
            maxval = output[i];
        }
    }

    let start = pos.saturating_sub(2);
    let end = (pos + 2).min(bins.saturating_sub(1));
    let mut sum = 0.0f32;
    let mut count = 0.0f32;
    for i in start..=end {
        let p = libm::expf(output[i]);
        sum += p * i as f32;
        count += p;
    }

    if count > 0.0 {
        (1.0 / 60.0) * (sum / count) - 1.5
    } else {
        -1.5
    }
}

fn init_pitchdnn(model: &mut PitchDnn) {
    let dense_if_upsampler_1_outputs = DENSE_IF_UPSAMPLER_1_BIAS.len();
    let dense_if_upsampler_1_inputs =
        DENSE_IF_UPSAMPLER_1_WEIGHTS_FLOAT.len() / dense_if_upsampler_1_outputs;
    model.dense_if_upsampler_1 = linear_layer(
        Some(&DENSE_IF_UPSAMPLER_1_BIAS),
        Some(&DENSE_IF_UPSAMPLER_1_SUBIAS),
        Some(&DENSE_IF_UPSAMPLER_1_WEIGHTS_INT8),
        Some(&DENSE_IF_UPSAMPLER_1_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DENSE_IF_UPSAMPLER_1_SCALE),
        dense_if_upsampler_1_inputs,
        dense_if_upsampler_1_outputs,
    );

    let dense_if_upsampler_2_outputs = DENSE_IF_UPSAMPLER_2_BIAS.len();
    let dense_if_upsampler_2_inputs =
        DENSE_IF_UPSAMPLER_2_WEIGHTS_FLOAT.len() / dense_if_upsampler_2_outputs;
    model.dense_if_upsampler_2 = linear_layer(
        Some(&DENSE_IF_UPSAMPLER_2_BIAS),
        Some(&DENSE_IF_UPSAMPLER_2_SUBIAS),
        Some(&DENSE_IF_UPSAMPLER_2_WEIGHTS_INT8),
        Some(&DENSE_IF_UPSAMPLER_2_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DENSE_IF_UPSAMPLER_2_SCALE),
        dense_if_upsampler_2_inputs,
        dense_if_upsampler_2_outputs,
    );

    let dense_downsampler_outputs = DENSE_DOWNSAMPLER_BIAS.len();
    let dense_downsampler_inputs =
        DENSE_DOWNSAMPLER_WEIGHTS_FLOAT.len() / dense_downsampler_outputs;
    model.dense_downsampler = linear_layer(
        Some(&DENSE_DOWNSAMPLER_BIAS),
        Some(&DENSE_DOWNSAMPLER_SUBIAS),
        Some(&DENSE_DOWNSAMPLER_WEIGHTS_INT8),
        Some(&DENSE_DOWNSAMPLER_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DENSE_DOWNSAMPLER_SCALE),
        dense_downsampler_inputs,
        dense_downsampler_outputs,
    );

    let dense_final_outputs = DENSE_FINAL_UPSAMPLER_BIAS.len();
    let dense_final_inputs =
        DENSE_FINAL_UPSAMPLER_WEIGHTS_FLOAT.len() / dense_final_outputs;
    model.dense_final_upsampler = linear_layer(
        Some(&DENSE_FINAL_UPSAMPLER_BIAS),
        Some(&DENSE_FINAL_UPSAMPLER_SUBIAS),
        Some(&DENSE_FINAL_UPSAMPLER_WEIGHTS_INT8),
        Some(&DENSE_FINAL_UPSAMPLER_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DENSE_FINAL_UPSAMPLER_SCALE),
        dense_final_inputs,
        dense_final_outputs,
    );

    model.gru_1_input = linear_layer(
        Some(&GRU_1_INPUT_BIAS),
        Some(&GRU_1_INPUT_SUBIAS),
        Some(&GRU_1_INPUT_WEIGHTS_INT8),
        Some(&GRU_1_INPUT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&GRU_1_INPUT_SCALE),
        dense_downsampler_outputs,
        GRU_1_INPUT_BIAS.len(),
    );
    model.gru_1_recurrent = linear_layer(
        Some(&GRU_1_RECURRENT_BIAS),
        Some(&GRU_1_RECURRENT_SUBIAS),
        Some(&GRU_1_RECURRENT_WEIGHTS_INT8),
        Some(&GRU_1_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&GRU_1_RECURRENT_SCALE),
        GRU_1_STATE_SIZE,
        GRU_1_RECURRENT_BIAS.len(),
    );

    let conv2d_1_out = CONV2D_1_BIAS.len();
    let conv2d_1_in = CONV2D_1_WEIGHT_FLOAT.len()
        / (conv2d_1_out * CONV_KERNEL * CONV_KERNEL);
    model.conv2d_1 = Conv2dLayer {
        bias: Some(&CONV2D_1_BIAS),
        float_weights: Some(&CONV2D_1_WEIGHT_FLOAT),
        in_channels: conv2d_1_in,
        out_channels: conv2d_1_out,
        ktime: CONV_KERNEL,
        kheight: CONV_KERNEL,
    };

    let conv2d_2_out = CONV2D_2_BIAS.len();
    let conv2d_2_in = CONV2D_2_WEIGHT_FLOAT.len()
        / (conv2d_2_out * CONV_KERNEL * CONV_KERNEL);
    model.conv2d_2 = Conv2dLayer {
        bias: Some(&CONV2D_2_BIAS),
        float_weights: Some(&CONV2D_2_WEIGHT_FLOAT),
        in_channels: conv2d_2_in,
        out_channels: conv2d_2_out,
        ktime: CONV_KERNEL,
        kheight: CONV_KERNEL,
    };
}

fn init_pitchdnn_from_weights(
    model: &mut PitchDnn,
    blob: &WeightBlob<'_>,
) -> Result<(), WeightError> {
    let dense_if_upsampler_1_outputs = DENSE_IF_UPSAMPLER_1_BIAS.len();
    let dense_if_upsampler_1_inputs =
        DENSE_IF_UPSAMPLER_1_WEIGHTS_FLOAT.len() / dense_if_upsampler_1_outputs;
    model.dense_if_upsampler_1 = linear_layer_from_weights(
        blob,
        Some("dense_if_upsampler_1_bias"),
        Some("dense_if_upsampler_1_subias"),
        Some("dense_if_upsampler_1_weights_int8"),
        Some("dense_if_upsampler_1_weights_float"),
        None,
        None,
        Some("dense_if_upsampler_1_scale"),
        dense_if_upsampler_1_inputs,
        dense_if_upsampler_1_outputs,
    )?;

    let dense_if_upsampler_2_outputs = DENSE_IF_UPSAMPLER_2_BIAS.len();
    let dense_if_upsampler_2_inputs =
        DENSE_IF_UPSAMPLER_2_WEIGHTS_FLOAT.len() / dense_if_upsampler_2_outputs;
    model.dense_if_upsampler_2 = linear_layer_from_weights(
        blob,
        Some("dense_if_upsampler_2_bias"),
        Some("dense_if_upsampler_2_subias"),
        Some("dense_if_upsampler_2_weights_int8"),
        Some("dense_if_upsampler_2_weights_float"),
        None,
        None,
        Some("dense_if_upsampler_2_scale"),
        dense_if_upsampler_2_inputs,
        dense_if_upsampler_2_outputs,
    )?;

    let dense_downsampler_outputs = DENSE_DOWNSAMPLER_BIAS.len();
    let dense_downsampler_inputs =
        DENSE_DOWNSAMPLER_WEIGHTS_FLOAT.len() / dense_downsampler_outputs;
    model.dense_downsampler = linear_layer_from_weights(
        blob,
        Some("dense_downsampler_bias"),
        Some("dense_downsampler_subias"),
        Some("dense_downsampler_weights_int8"),
        Some("dense_downsampler_weights_float"),
        None,
        None,
        Some("dense_downsampler_scale"),
        dense_downsampler_inputs,
        dense_downsampler_outputs,
    )?;

    let dense_final_outputs = DENSE_FINAL_UPSAMPLER_BIAS.len();
    let dense_final_inputs =
        DENSE_FINAL_UPSAMPLER_WEIGHTS_FLOAT.len() / dense_final_outputs;
    model.dense_final_upsampler = linear_layer_from_weights(
        blob,
        Some("dense_final_upsampler_bias"),
        Some("dense_final_upsampler_subias"),
        Some("dense_final_upsampler_weights_int8"),
        Some("dense_final_upsampler_weights_float"),
        None,
        None,
        Some("dense_final_upsampler_scale"),
        dense_final_inputs,
        dense_final_outputs,
    )?;

    model.gru_1_input = linear_layer_from_weights(
        blob,
        Some("gru_1_input_bias"),
        Some("gru_1_input_subias"),
        Some("gru_1_input_weights_int8"),
        Some("gru_1_input_weights_float"),
        None,
        None,
        Some("gru_1_input_scale"),
        dense_downsampler_outputs,
        GRU_1_INPUT_BIAS.len(),
    )?;
    model.gru_1_recurrent = linear_layer_from_weights(
        blob,
        Some("gru_1_recurrent_bias"),
        Some("gru_1_recurrent_subias"),
        Some("gru_1_recurrent_weights_int8"),
        Some("gru_1_recurrent_weights_float"),
        None,
        None,
        Some("gru_1_recurrent_scale"),
        GRU_1_STATE_SIZE,
        GRU_1_RECURRENT_BIAS.len(),
    )?;

    let conv2d_1_out = CONV2D_1_BIAS.len();
    let conv2d_1_in = CONV2D_1_WEIGHT_FLOAT.len()
        / (conv2d_1_out * CONV_KERNEL * CONV_KERNEL);
    model.conv2d_1 = conv2d_layer_from_weights(
        blob,
        Some("conv2d_1_bias"),
        Some("conv2d_1_weight_float"),
        conv2d_1_in,
        conv2d_1_out,
        CONV_KERNEL,
        CONV_KERNEL,
    )?;

    let conv2d_2_out = CONV2D_2_BIAS.len();
    let conv2d_2_in = CONV2D_2_WEIGHT_FLOAT.len()
        / (conv2d_2_out * CONV_KERNEL * CONV_KERNEL);
    model.conv2d_2 = conv2d_layer_from_weights(
        blob,
        Some("conv2d_2_bias"),
        Some("conv2d_2_weight_float"),
        conv2d_2_in,
        conv2d_2_out,
        CONV_KERNEL,
        CONV_KERNEL,
    )?;

    Ok(())
}

fn linear_layer(
    bias: Option<&'static [f32]>,
    subias: Option<&'static [f32]>,
    weights: Option<&'static [i8]>,
    float_weights: Option<&'static [f32]>,
    weights_idx: Option<&'static [i32]>,
    diag: Option<&'static [f32]>,
    scale: Option<&'static [f32]>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> LinearLayer {
    if let Some(bias) = bias {
        debug_assert_eq!(bias.len(), nb_outputs);
    }
    if let Some(subias) = subias {
        debug_assert_eq!(subias.len(), nb_outputs);
    }
    if let Some(scale) = scale {
        debug_assert_eq!(scale.len(), nb_outputs);
    }
    if let Some(float_weights) = float_weights {
        if let Some(weights_idx) = weights_idx {
            let blocks = sparse_block_count(weights_idx, nb_inputs, nb_outputs);
            debug_assert_eq!(float_weights.len(), blocks * 32);
        } else {
            debug_assert_eq!(float_weights.len(), nb_inputs * nb_outputs);
        }
    }
    if let Some(weights) = weights {
        if let Some(weights_idx) = weights_idx {
            let blocks = sparse_block_count(weights_idx, nb_inputs, nb_outputs);
            debug_assert_eq!(weights.len(), blocks * 32);
        } else {
            debug_assert_eq!(weights.len(), nb_inputs * nb_outputs);
        }
    }

    LinearLayer {
        bias,
        subias,
        weights,
        float_weights,
        weights_idx,
        diag,
        scale,
        nb_inputs,
        nb_outputs,
    }
}

fn linear_layer_from_weights(
    blob: &WeightBlob<'_>,
    bias: Option<&'static str>,
    subias: Option<&'static str>,
    weights: Option<&'static str>,
    float_weights: Option<&'static str>,
    weights_idx: Option<&'static str>,
    diag: Option<&'static str>,
    scale: Option<&'static str>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer, WeightError> {
    let bias = match bias {
        Some(name) => Some(leak_f32(require_bytes(
            blob,
            name,
            nb_outputs * core::mem::size_of::<f32>(),
        )?)?),
        None => None,
    };
    let subias = match subias {
        Some(name) => Some(leak_f32(require_bytes(
            blob,
            name,
            nb_outputs * core::mem::size_of::<f32>(),
        )?)?),
        None => None,
    };
    let diag = match diag {
        Some(name) => Some(leak_f32(require_bytes(
            blob,
            name,
            nb_outputs * core::mem::size_of::<f32>(),
        )?)?),
        None => None,
    };

    let mut total_blocks = None;
    let weights_idx = match weights_idx {
        Some(name) => {
            let data = require_bytes(blob, name, bytes_len(blob, name)?)?;
            let idx = leak_i32(data)?;
            let blocks = validate_sparse_idx(idx, nb_inputs, nb_outputs, name)?;
            total_blocks = Some(blocks);
            Some(idx)
        }
        None => None,
    };

    let (weights, float_weights) = if let Some(blocks) = total_blocks {
        let expected_i8 = 32usize
            .checked_mul(blocks)
            .ok_or(WeightError::InvalidBlob)?;
        let weights = match weights {
            Some(name) => Some(leak_i8(require_bytes(blob, name, expected_i8)?)?),
            None => None,
        };
        let expected_f32 = expected_i8
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(WeightError::InvalidBlob)?;
        let float_weights = match float_weights {
            Some(name) => match optional_bytes(blob, name, expected_f32)? {
                Some(data) => Some(leak_f32(data)?),
                None => None,
            },
            None => None,
        };
        (weights, float_weights)
    } else {
        let expected_i8 = nb_inputs
            .checked_mul(nb_outputs)
            .ok_or(WeightError::InvalidBlob)?;
        let weights = match weights {
            Some(name) => Some(leak_i8(require_bytes(blob, name, expected_i8)?)?),
            None => None,
        };
        let expected_f32 = expected_i8
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(WeightError::InvalidBlob)?;
        let float_weights = match float_weights {
            Some(name) => match optional_bytes(blob, name, expected_f32)? {
                Some(data) => Some(leak_f32(data)?),
                None => None,
            },
            None => None,
        };
        (weights, float_weights)
    };

    let scale = match scale {
        Some(name) => match weights {
            Some(_) => Some(leak_f32(require_bytes(
                blob,
                name,
                nb_outputs * core::mem::size_of::<f32>(),
            )?)?),
            None => None,
        },
        None => None,
    };

    Ok(LinearLayer {
        bias,
        subias,
        weights,
        float_weights,
        weights_idx,
        diag,
        scale,
        nb_inputs,
        nb_outputs,
    })
}

fn conv2d_layer_from_weights(
    blob: &WeightBlob<'_>,
    bias: Option<&'static str>,
    float_weights: Option<&'static str>,
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
) -> Result<Conv2dLayer, WeightError> {
    let bias = match bias {
        Some(name) => Some(leak_f32(require_bytes(
            blob,
            name,
            out_channels * core::mem::size_of::<f32>(),
        )?)?),
        None => None,
    };
    let float_weights = match float_weights {
        Some(name) => Some(leak_f32(require_bytes(
            blob,
            name,
            out_channels
                .checked_mul(in_channels)
                .and_then(|value| value.checked_mul(ktime))
                .and_then(|value| value.checked_mul(kheight))
                .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
                .ok_or(WeightError::InvalidBlob)?,
        )?)?),
        None => None,
    };

    Ok(Conv2dLayer {
        bias,
        float_weights,
        in_channels,
        out_channels,
        ktime,
        kheight,
    })
}

fn sparse_block_count(idx: &[i32], nb_inputs: usize, nb_outputs: usize) -> usize {
    let mut remain = idx.len() as i32;
    let mut out = nb_outputs as i32;
    let mut total_blocks = 0i32;
    let mut pos = 0usize;
    while remain > 0 {
        let nb_blocks = idx[pos];
        pos += 1;
        remain -= 1;
        if nb_blocks < 0 || remain < nb_blocks {
            return 0;
        }
        for _ in 0..nb_blocks {
            let offset = idx[pos];
            pos += 1;
            remain -= 1;
            if offset < 0 || offset + 3 >= nb_inputs as i32 || (offset & 0x3) != 0 {
                return 0;
            }
        }
        out -= 8;
        total_blocks += nb_blocks;
    }
    if out != 0 {
        return 0;
    }
    total_blocks as usize
}

fn leak_f32(data: &[u8]) -> Result<&'static [f32], WeightError> {
    if data.len() % 4 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let mut values = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().map_err(|_| WeightError::InvalidBlob)?;
        values.push(f32::from_le_bytes(bytes));
    }
    Ok(Box::leak(values.into_boxed_slice()))
}

fn leak_i8(data: &[u8]) -> Result<&'static [i8], WeightError> {
    let mut values = Vec::with_capacity(data.len());
    values.extend(data.iter().map(|&byte| byte as i8));
    Ok(Box::leak(values.into_boxed_slice()))
}

fn leak_i32(data: &[u8]) -> Result<&'static [i32], WeightError> {
    if data.len() % 4 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let mut values = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().map_err(|_| WeightError::InvalidBlob)?;
        values.push(i32::from_le_bytes(bytes));
    }
    Ok(Box::leak(values.into_boxed_slice()))
}

fn bytes_len(blob: &WeightBlob<'_>, name: &'static str) -> Result<usize, WeightError> {
    blob.find(name)
        .map(|array| array.size)
        .ok_or(WeightError::MissingArray(name))
}

fn validate_sparse_idx(
    idx: &[i32],
    nb_inputs: usize,
    nb_outputs: usize,
    name: &'static str,
) -> Result<usize, WeightError> {
    let mut remain = idx.len() as i32;
    let mut out = nb_outputs as i32;
    let mut pos = 0usize;
    let mut total_blocks = 0i32;
    while remain > 0 {
        let nb_blocks = idx[pos];
        pos += 1;
        remain -= 1;
        if nb_blocks < 0 || remain < nb_blocks {
            return Err(WeightError::InvalidIndex(name));
        }
        for _ in 0..nb_blocks {
            let offset = idx[pos];
            pos += 1;
            remain -= 1;
            if offset < 0 || offset + 3 >= nb_inputs as i32 || (offset & 0x3) != 0 {
                return Err(WeightError::InvalidIndex(name));
            }
        }
        out -= 8;
        total_blocks += nb_blocks;
    }
    if out != 0 {
        return Err(WeightError::InvalidIndex(name));
    }
    Ok(total_blocks as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_pitchdnn_runs_with_zero_features() {
        let mut state = PitchDnnState::default();
        let if_features = [0.0f32; PITCH_IF_FEATURES];
        let xcorr_features = [0.0f32; NB_XCORR_FEATURES];
        let pitch = compute_pitchdnn(&mut state, &if_features, &xcorr_features, 0);
        assert!(pitch.is_finite());
    }
}
