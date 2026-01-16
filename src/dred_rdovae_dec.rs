//! RDOVAE decoder model and inference helpers.

use crate::dred_constants::{DRED_LATENT_DIM, DRED_NUM_FEATURES, DRED_STATE_DIM};
use crate::dred_rdovae_dec_data::*;
use crate::dnn_weights::{optional_bytes, require_bytes, WeightBlob, WeightError};
use crate::nnet::{
    compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu,
    LinearLayer, ACTIVATION_LINEAR, ACTIVATION_TANH,
};
use alloc::boxed::Box;
use alloc::vec::Vec;

pub(crate) const DEC_DENSE1_OUT_SIZE: usize = 96;
pub(crate) const DEC_GLU1_OUT_SIZE: usize = 96;
pub(crate) const DEC_GLU2_OUT_SIZE: usize = 96;
pub(crate) const DEC_GLU3_OUT_SIZE: usize = 96;
pub(crate) const DEC_GLU4_OUT_SIZE: usize = 96;
pub(crate) const DEC_GLU5_OUT_SIZE: usize = 96;
pub(crate) const DEC_OUTPUT_OUT_SIZE: usize = 80;
pub(crate) const DEC_HIDDEN_INIT_OUT_SIZE: usize = 128;
pub(crate) const DEC_GRU_INIT_OUT_SIZE: usize = 480;
pub(crate) const DEC_GRU1_OUT_SIZE: usize = 96;
pub(crate) const DEC_GRU1_STATE_SIZE: usize = 96;
pub(crate) const DEC_GRU2_OUT_SIZE: usize = 96;
pub(crate) const DEC_GRU2_STATE_SIZE: usize = 96;
pub(crate) const DEC_GRU2_IN_SIZE: usize = 224;
pub(crate) const DEC_GRU3_OUT_SIZE: usize = 96;
pub(crate) const DEC_GRU3_STATE_SIZE: usize = 96;
pub(crate) const DEC_GRU3_IN_SIZE: usize = 352;
pub(crate) const DEC_GRU4_OUT_SIZE: usize = 96;
pub(crate) const DEC_GRU4_STATE_SIZE: usize = 96;
pub(crate) const DEC_GRU4_IN_SIZE: usize = 480;
pub(crate) const DEC_GRU5_OUT_SIZE: usize = 96;
pub(crate) const DEC_GRU5_STATE_SIZE: usize = 96;
pub(crate) const DEC_GRU5_IN_SIZE: usize = 608;
pub(crate) const DEC_CONV1_OUT_SIZE: usize = 32;
pub(crate) const DEC_CONV1_IN_SIZE: usize = 192;
pub(crate) const DEC_CONV1_STATE_SIZE: usize = 192;
#[allow(dead_code)]
pub(crate) const DEC_CONV1_DELAY: usize = 0;
pub(crate) const DEC_CONV2_OUT_SIZE: usize = 32;
pub(crate) const DEC_CONV2_IN_SIZE: usize = 320;
pub(crate) const DEC_CONV2_STATE_SIZE: usize = 320;
#[allow(dead_code)]
pub(crate) const DEC_CONV2_DELAY: usize = 0;
pub(crate) const DEC_CONV3_OUT_SIZE: usize = 32;
pub(crate) const DEC_CONV3_IN_SIZE: usize = 448;
pub(crate) const DEC_CONV3_STATE_SIZE: usize = 448;
#[allow(dead_code)]
pub(crate) const DEC_CONV3_DELAY: usize = 0;
pub(crate) const DEC_CONV4_OUT_SIZE: usize = 32;
pub(crate) const DEC_CONV4_IN_SIZE: usize = 576;
pub(crate) const DEC_CONV4_STATE_SIZE: usize = 576;
#[allow(dead_code)]
pub(crate) const DEC_CONV4_DELAY: usize = 0;
pub(crate) const DEC_CONV5_OUT_SIZE: usize = 32;
pub(crate) const DEC_CONV5_IN_SIZE: usize = 704;
pub(crate) const DEC_CONV5_STATE_SIZE: usize = 704;
#[allow(dead_code)]
pub(crate) const DEC_CONV5_DELAY: usize = 0;

const DEC_BUFFER_SIZE: usize = DEC_DENSE1_OUT_SIZE
    + DEC_GRU1_OUT_SIZE
    + DEC_GRU2_OUT_SIZE
    + DEC_GRU3_OUT_SIZE
    + DEC_GRU4_OUT_SIZE
    + DEC_GRU5_OUT_SIZE
    + DEC_CONV1_OUT_SIZE
    + DEC_CONV2_OUT_SIZE
    + DEC_CONV3_OUT_SIZE
    + DEC_CONV4_OUT_SIZE
    + DEC_CONV5_OUT_SIZE;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RdovaeDec {
    pub dec_dense1: LinearLayer,
    pub dec_glu1: LinearLayer,
    pub dec_glu2: LinearLayer,
    pub dec_glu3: LinearLayer,
    pub dec_glu4: LinearLayer,
    pub dec_glu5: LinearLayer,
    pub dec_output: LinearLayer,
    pub dec_hidden_init: LinearLayer,
    pub dec_gru_init: LinearLayer,
    pub dec_gru1_input: LinearLayer,
    pub dec_gru1_recurrent: LinearLayer,
    pub dec_gru2_input: LinearLayer,
    pub dec_gru2_recurrent: LinearLayer,
    pub dec_gru3_input: LinearLayer,
    pub dec_gru3_recurrent: LinearLayer,
    pub dec_gru4_input: LinearLayer,
    pub dec_gru4_recurrent: LinearLayer,
    pub dec_gru5_input: LinearLayer,
    pub dec_gru5_recurrent: LinearLayer,
    pub dec_conv1: LinearLayer,
    pub dec_conv2: LinearLayer,
    pub dec_conv3: LinearLayer,
    pub dec_conv4: LinearLayer,
    pub dec_conv5: LinearLayer,
}

impl RdovaeDec {
    pub(crate) fn new() -> Self {
        let mut model = Self::default();
        init_rdovaedec(&mut model);
        model
    }

    pub(crate) fn from_weights(data: &[u8]) -> Result<Self, WeightError> {
        let blob = WeightBlob::parse(data)?;
        let mut model = Self::default();
        init_rdovaedec_from_weights(&mut model, &blob)?;
        Ok(model)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RdovaeDecState {
    pub initialized: i32,
    pub gru1_state: [f32; DEC_GRU1_STATE_SIZE],
    pub gru2_state: [f32; DEC_GRU2_STATE_SIZE],
    pub gru3_state: [f32; DEC_GRU3_STATE_SIZE],
    pub gru4_state: [f32; DEC_GRU4_STATE_SIZE],
    pub gru5_state: [f32; DEC_GRU5_STATE_SIZE],
    pub conv1_state: [f32; DEC_CONV1_STATE_SIZE],
    pub conv2_state: [f32; DEC_CONV2_STATE_SIZE],
    pub conv3_state: [f32; DEC_CONV3_STATE_SIZE],
    pub conv4_state: [f32; DEC_CONV4_STATE_SIZE],
    pub conv5_state: [f32; DEC_CONV5_STATE_SIZE],
}

impl Default for RdovaeDecState {
    fn default() -> Self {
        Self {
            initialized: 0,
            gru1_state: [0.0; DEC_GRU1_STATE_SIZE],
            gru2_state: [0.0; DEC_GRU2_STATE_SIZE],
            gru3_state: [0.0; DEC_GRU3_STATE_SIZE],
            gru4_state: [0.0; DEC_GRU4_STATE_SIZE],
            gru5_state: [0.0; DEC_GRU5_STATE_SIZE],
            conv1_state: [0.0; DEC_CONV1_STATE_SIZE],
            conv2_state: [0.0; DEC_CONV2_STATE_SIZE],
            conv3_state: [0.0; DEC_CONV3_STATE_SIZE],
            conv4_state: [0.0; DEC_CONV4_STATE_SIZE],
            conv5_state: [0.0; DEC_CONV5_STATE_SIZE],
        }
    }
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
        let expected_i8 = SPARSE_BLOCK_SIZE
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

fn init_rdovaedec_from_weights(
    model: &mut RdovaeDec,
    blob: &WeightBlob<'_>,
) -> Result<(), WeightError> {
    model.dec_dense1 = linear_layer_from_weights(
        blob,
        Some("dec_dense1_bias"),
        None,
        None,
        Some("dec_dense1_weights_float"),
        None,
        None,
        None,
        DRED_LATENT_DIM,
        DEC_DENSE1_OUT_SIZE,
    )?;
    model.dec_glu1 = linear_layer_from_weights(
        blob,
        Some("dec_glu1_bias"),
        Some("dec_glu1_subias"),
        Some("dec_glu1_weights_int8"),
        Some("dec_glu1_weights_float"),
        None,
        None,
        Some("dec_glu1_scale"),
        DEC_GLU1_OUT_SIZE,
        DEC_GLU1_OUT_SIZE,
    )?;
    model.dec_glu2 = linear_layer_from_weights(
        blob,
        Some("dec_glu2_bias"),
        Some("dec_glu2_subias"),
        Some("dec_glu2_weights_int8"),
        Some("dec_glu2_weights_float"),
        None,
        None,
        Some("dec_glu2_scale"),
        DEC_GLU2_OUT_SIZE,
        DEC_GLU2_OUT_SIZE,
    )?;
    model.dec_glu3 = linear_layer_from_weights(
        blob,
        Some("dec_glu3_bias"),
        Some("dec_glu3_subias"),
        Some("dec_glu3_weights_int8"),
        Some("dec_glu3_weights_float"),
        None,
        None,
        Some("dec_glu3_scale"),
        DEC_GLU3_OUT_SIZE,
        DEC_GLU3_OUT_SIZE,
    )?;
    model.dec_glu4 = linear_layer_from_weights(
        blob,
        Some("dec_glu4_bias"),
        Some("dec_glu4_subias"),
        Some("dec_glu4_weights_int8"),
        Some("dec_glu4_weights_float"),
        None,
        None,
        Some("dec_glu4_scale"),
        DEC_GLU4_OUT_SIZE,
        DEC_GLU4_OUT_SIZE,
    )?;
    model.dec_glu5 = linear_layer_from_weights(
        blob,
        Some("dec_glu5_bias"),
        Some("dec_glu5_subias"),
        Some("dec_glu5_weights_int8"),
        Some("dec_glu5_weights_float"),
        None,
        None,
        Some("dec_glu5_scale"),
        DEC_GLU5_OUT_SIZE,
        DEC_GLU5_OUT_SIZE,
    )?;
    model.dec_output = linear_layer_from_weights(
        blob,
        Some("dec_output_bias"),
        Some("dec_output_subias"),
        Some("dec_output_weights_int8"),
        Some("dec_output_weights_float"),
        None,
        None,
        Some("dec_output_scale"),
        DEC_BUFFER_SIZE,
        DEC_OUTPUT_OUT_SIZE,
    )?;
    model.dec_hidden_init = linear_layer_from_weights(
        blob,
        Some("dec_hidden_init_bias"),
        None,
        None,
        Some("dec_hidden_init_weights_float"),
        None,
        None,
        None,
        DRED_STATE_DIM,
        DEC_HIDDEN_INIT_OUT_SIZE,
    )?;
    model.dec_gru_init = linear_layer_from_weights(
        blob,
        Some("dec_gru_init_bias"),
        Some("dec_gru_init_subias"),
        Some("dec_gru_init_weights_int8"),
        Some("dec_gru_init_weights_float"),
        None,
        None,
        Some("dec_gru_init_scale"),
        DEC_HIDDEN_INIT_OUT_SIZE,
        DEC_GRU_INIT_OUT_SIZE,
    )?;
    model.dec_gru1_input = linear_layer_from_weights(
        blob,
        Some("dec_gru1_input_bias"),
        Some("dec_gru1_input_subias"),
        Some("dec_gru1_input_weights_int8"),
        Some("dec_gru1_input_weights_float"),
        Some("dec_gru1_input_weights_idx"),
        None,
        Some("dec_gru1_input_scale"),
        DEC_GRU1_OUT_SIZE,
        DEC_GRU1_STATE_SIZE * 3,
    )?;
    model.dec_gru1_recurrent = linear_layer_from_weights(
        blob,
        Some("dec_gru1_recurrent_bias"),
        Some("dec_gru1_recurrent_subias"),
        Some("dec_gru1_recurrent_weights_int8"),
        Some("dec_gru1_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru1_recurrent_scale"),
        DEC_GRU1_OUT_SIZE,
        DEC_GRU1_STATE_SIZE * 3,
    )?;
    model.dec_gru2_input = linear_layer_from_weights(
        blob,
        Some("dec_gru2_input_bias"),
        Some("dec_gru2_input_subias"),
        Some("dec_gru2_input_weights_int8"),
        Some("dec_gru2_input_weights_float"),
        Some("dec_gru2_input_weights_idx"),
        None,
        Some("dec_gru2_input_scale"),
        DEC_GRU2_IN_SIZE,
        DEC_GRU2_STATE_SIZE * 3,
    )?;
    model.dec_gru2_recurrent = linear_layer_from_weights(
        blob,
        Some("dec_gru2_recurrent_bias"),
        Some("dec_gru2_recurrent_subias"),
        Some("dec_gru2_recurrent_weights_int8"),
        Some("dec_gru2_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru2_recurrent_scale"),
        DEC_GRU2_OUT_SIZE,
        DEC_GRU2_STATE_SIZE * 3,
    )?;
    model.dec_gru3_input = linear_layer_from_weights(
        blob,
        Some("dec_gru3_input_bias"),
        Some("dec_gru3_input_subias"),
        Some("dec_gru3_input_weights_int8"),
        Some("dec_gru3_input_weights_float"),
        Some("dec_gru3_input_weights_idx"),
        None,
        Some("dec_gru3_input_scale"),
        DEC_GRU3_IN_SIZE,
        DEC_GRU3_STATE_SIZE * 3,
    )?;
    model.dec_gru3_recurrent = linear_layer_from_weights(
        blob,
        Some("dec_gru3_recurrent_bias"),
        Some("dec_gru3_recurrent_subias"),
        Some("dec_gru3_recurrent_weights_int8"),
        Some("dec_gru3_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru3_recurrent_scale"),
        DEC_GRU3_OUT_SIZE,
        DEC_GRU3_STATE_SIZE * 3,
    )?;
    model.dec_gru4_input = linear_layer_from_weights(
        blob,
        Some("dec_gru4_input_bias"),
        Some("dec_gru4_input_subias"),
        Some("dec_gru4_input_weights_int8"),
        Some("dec_gru4_input_weights_float"),
        Some("dec_gru4_input_weights_idx"),
        None,
        Some("dec_gru4_input_scale"),
        DEC_GRU4_IN_SIZE,
        DEC_GRU4_STATE_SIZE * 3,
    )?;
    model.dec_gru4_recurrent = linear_layer_from_weights(
        blob,
        Some("dec_gru4_recurrent_bias"),
        Some("dec_gru4_recurrent_subias"),
        Some("dec_gru4_recurrent_weights_int8"),
        Some("dec_gru4_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru4_recurrent_scale"),
        DEC_GRU4_OUT_SIZE,
        DEC_GRU4_STATE_SIZE * 3,
    )?;
    model.dec_gru5_input = linear_layer_from_weights(
        blob,
        Some("dec_gru5_input_bias"),
        Some("dec_gru5_input_subias"),
        Some("dec_gru5_input_weights_int8"),
        Some("dec_gru5_input_weights_float"),
        Some("dec_gru5_input_weights_idx"),
        None,
        Some("dec_gru5_input_scale"),
        DEC_GRU5_IN_SIZE,
        DEC_GRU5_STATE_SIZE * 3,
    )?;
    model.dec_gru5_recurrent = linear_layer_from_weights(
        blob,
        Some("dec_gru5_recurrent_bias"),
        Some("dec_gru5_recurrent_subias"),
        Some("dec_gru5_recurrent_weights_int8"),
        Some("dec_gru5_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru5_recurrent_scale"),
        DEC_GRU5_OUT_SIZE,
        DEC_GRU5_STATE_SIZE * 3,
    )?;
    model.dec_conv1 = linear_layer_from_weights(
        blob,
        Some("dec_conv1_bias"),
        Some("dec_conv1_subias"),
        Some("dec_conv1_weights_int8"),
        Some("dec_conv1_weights_float"),
        None,
        None,
        Some("dec_conv1_scale"),
        DEC_CONV1_IN_SIZE * 2,
        DEC_CONV1_OUT_SIZE,
    )?;
    model.dec_conv2 = linear_layer_from_weights(
        blob,
        Some("dec_conv2_bias"),
        Some("dec_conv2_subias"),
        Some("dec_conv2_weights_int8"),
        Some("dec_conv2_weights_float"),
        None,
        None,
        Some("dec_conv2_scale"),
        DEC_CONV2_IN_SIZE * 2,
        DEC_CONV2_OUT_SIZE,
    )?;
    model.dec_conv3 = linear_layer_from_weights(
        blob,
        Some("dec_conv3_bias"),
        Some("dec_conv3_subias"),
        Some("dec_conv3_weights_int8"),
        Some("dec_conv3_weights_float"),
        None,
        None,
        Some("dec_conv3_scale"),
        DEC_CONV3_IN_SIZE * 2,
        DEC_CONV3_OUT_SIZE,
    )?;
    model.dec_conv4 = linear_layer_from_weights(
        blob,
        Some("dec_conv4_bias"),
        Some("dec_conv4_subias"),
        Some("dec_conv4_weights_int8"),
        Some("dec_conv4_weights_float"),
        None,
        None,
        Some("dec_conv4_scale"),
        DEC_CONV4_IN_SIZE * 2,
        DEC_CONV4_OUT_SIZE,
    )?;
    model.dec_conv5 = linear_layer_from_weights(
        blob,
        Some("dec_conv5_bias"),
        Some("dec_conv5_subias"),
        Some("dec_conv5_weights_int8"),
        Some("dec_conv5_weights_float"),
        None,
        None,
        Some("dec_conv5_scale"),
        DEC_CONV5_IN_SIZE * 2,
        DEC_CONV5_OUT_SIZE,
    )?;
    Ok(())
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

const SPARSE_BLOCK_SIZE: usize = 32;

fn init_rdovaedec(model: &mut RdovaeDec) {
    model.dec_dense1 = linear_layer(
        Some(&DEC_DENSE1_BIAS),
        None,
        None,
        Some(&DEC_DENSE1_WEIGHTS_FLOAT),
        None,
        None,
        None,
        DRED_LATENT_DIM,
        DEC_DENSE1_OUT_SIZE,
    );
    model.dec_glu1 = linear_layer(
        Some(&DEC_GLU1_BIAS),
        Some(&DEC_GLU1_SUBIAS),
        Some(&DEC_GLU1_WEIGHTS_INT8),
        Some(&DEC_GLU1_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GLU1_SCALE),
        DEC_GLU1_OUT_SIZE,
        DEC_GLU1_OUT_SIZE,
    );
    model.dec_glu2 = linear_layer(
        Some(&DEC_GLU2_BIAS),
        Some(&DEC_GLU2_SUBIAS),
        Some(&DEC_GLU2_WEIGHTS_INT8),
        Some(&DEC_GLU2_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GLU2_SCALE),
        DEC_GLU2_OUT_SIZE,
        DEC_GLU2_OUT_SIZE,
    );
    model.dec_glu3 = linear_layer(
        Some(&DEC_GLU3_BIAS),
        Some(&DEC_GLU3_SUBIAS),
        Some(&DEC_GLU3_WEIGHTS_INT8),
        Some(&DEC_GLU3_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GLU3_SCALE),
        DEC_GLU3_OUT_SIZE,
        DEC_GLU3_OUT_SIZE,
    );
    model.dec_glu4 = linear_layer(
        Some(&DEC_GLU4_BIAS),
        Some(&DEC_GLU4_SUBIAS),
        Some(&DEC_GLU4_WEIGHTS_INT8),
        Some(&DEC_GLU4_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GLU4_SCALE),
        DEC_GLU4_OUT_SIZE,
        DEC_GLU4_OUT_SIZE,
    );
    model.dec_glu5 = linear_layer(
        Some(&DEC_GLU5_BIAS),
        Some(&DEC_GLU5_SUBIAS),
        Some(&DEC_GLU5_WEIGHTS_INT8),
        Some(&DEC_GLU5_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GLU5_SCALE),
        DEC_GLU5_OUT_SIZE,
        DEC_GLU5_OUT_SIZE,
    );
    model.dec_output = linear_layer(
        Some(&DEC_OUTPUT_BIAS),
        Some(&DEC_OUTPUT_SUBIAS),
        Some(&DEC_OUTPUT_WEIGHTS_INT8),
        Some(&DEC_OUTPUT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_OUTPUT_SCALE),
        DEC_BUFFER_SIZE,
        DEC_OUTPUT_OUT_SIZE,
    );
    model.dec_hidden_init = linear_layer(
        Some(&DEC_HIDDEN_INIT_BIAS),
        None,
        None,
        Some(&DEC_HIDDEN_INIT_WEIGHTS_FLOAT),
        None,
        None,
        None,
        DRED_STATE_DIM,
        DEC_HIDDEN_INIT_OUT_SIZE,
    );
    model.dec_gru_init = linear_layer(
        Some(&DEC_GRU_INIT_BIAS),
        Some(&DEC_GRU_INIT_SUBIAS),
        Some(&DEC_GRU_INIT_WEIGHTS_INT8),
        Some(&DEC_GRU_INIT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GRU_INIT_SCALE),
        DEC_HIDDEN_INIT_OUT_SIZE,
        DEC_GRU_INIT_OUT_SIZE,
    );
    model.dec_gru1_input = linear_layer(
        Some(&DEC_GRU1_INPUT_BIAS),
        Some(&DEC_GRU1_INPUT_SUBIAS),
        Some(&DEC_GRU1_INPUT_WEIGHTS_INT8),
        Some(&DEC_GRU1_INPUT_WEIGHTS_FLOAT),
        Some(&DEC_GRU1_INPUT_WEIGHTS_IDX),
        None,
        Some(&DEC_GRU1_INPUT_SCALE),
        DEC_GRU1_OUT_SIZE,
        DEC_GRU1_OUT_SIZE * 3,
    );
    model.dec_gru1_recurrent = linear_layer(
        Some(&DEC_GRU1_RECURRENT_BIAS),
        Some(&DEC_GRU1_RECURRENT_SUBIAS),
        Some(&DEC_GRU1_RECURRENT_WEIGHTS_INT8),
        Some(&DEC_GRU1_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GRU1_RECURRENT_SCALE),
        DEC_GRU1_STATE_SIZE,
        DEC_GRU1_OUT_SIZE * 3,
    );
    model.dec_gru2_input = linear_layer(
        Some(&DEC_GRU2_INPUT_BIAS),
        Some(&DEC_GRU2_INPUT_SUBIAS),
        Some(&DEC_GRU2_INPUT_WEIGHTS_INT8),
        Some(&DEC_GRU2_INPUT_WEIGHTS_FLOAT),
        Some(&DEC_GRU2_INPUT_WEIGHTS_IDX),
        None,
        Some(&DEC_GRU2_INPUT_SCALE),
        DEC_GRU2_IN_SIZE,
        DEC_GRU2_OUT_SIZE * 3,
    );
    model.dec_gru2_recurrent = linear_layer(
        Some(&DEC_GRU2_RECURRENT_BIAS),
        Some(&DEC_GRU2_RECURRENT_SUBIAS),
        Some(&DEC_GRU2_RECURRENT_WEIGHTS_INT8),
        Some(&DEC_GRU2_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GRU2_RECURRENT_SCALE),
        DEC_GRU2_STATE_SIZE,
        DEC_GRU2_OUT_SIZE * 3,
    );
    model.dec_gru3_input = linear_layer(
        Some(&DEC_GRU3_INPUT_BIAS),
        Some(&DEC_GRU3_INPUT_SUBIAS),
        Some(&DEC_GRU3_INPUT_WEIGHTS_INT8),
        Some(&DEC_GRU3_INPUT_WEIGHTS_FLOAT),
        Some(&DEC_GRU3_INPUT_WEIGHTS_IDX),
        None,
        Some(&DEC_GRU3_INPUT_SCALE),
        DEC_GRU3_IN_SIZE,
        DEC_GRU3_OUT_SIZE * 3,
    );
    model.dec_gru3_recurrent = linear_layer(
        Some(&DEC_GRU3_RECURRENT_BIAS),
        Some(&DEC_GRU3_RECURRENT_SUBIAS),
        Some(&DEC_GRU3_RECURRENT_WEIGHTS_INT8),
        Some(&DEC_GRU3_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GRU3_RECURRENT_SCALE),
        DEC_GRU3_STATE_SIZE,
        DEC_GRU3_OUT_SIZE * 3,
    );
    model.dec_gru4_input = linear_layer(
        Some(&DEC_GRU4_INPUT_BIAS),
        Some(&DEC_GRU4_INPUT_SUBIAS),
        Some(&DEC_GRU4_INPUT_WEIGHTS_INT8),
        Some(&DEC_GRU4_INPUT_WEIGHTS_FLOAT),
        Some(&DEC_GRU4_INPUT_WEIGHTS_IDX),
        None,
        Some(&DEC_GRU4_INPUT_SCALE),
        DEC_GRU4_IN_SIZE,
        DEC_GRU4_OUT_SIZE * 3,
    );
    model.dec_gru4_recurrent = linear_layer(
        Some(&DEC_GRU4_RECURRENT_BIAS),
        Some(&DEC_GRU4_RECURRENT_SUBIAS),
        Some(&DEC_GRU4_RECURRENT_WEIGHTS_INT8),
        Some(&DEC_GRU4_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GRU4_RECURRENT_SCALE),
        DEC_GRU4_STATE_SIZE,
        DEC_GRU4_OUT_SIZE * 3,
    );
    model.dec_gru5_input = linear_layer(
        Some(&DEC_GRU5_INPUT_BIAS),
        Some(&DEC_GRU5_INPUT_SUBIAS),
        Some(&DEC_GRU5_INPUT_WEIGHTS_INT8),
        Some(&DEC_GRU5_INPUT_WEIGHTS_FLOAT),
        Some(&DEC_GRU5_INPUT_WEIGHTS_IDX),
        None,
        Some(&DEC_GRU5_INPUT_SCALE),
        DEC_GRU5_IN_SIZE,
        DEC_GRU5_OUT_SIZE * 3,
    );
    model.dec_gru5_recurrent = linear_layer(
        Some(&DEC_GRU5_RECURRENT_BIAS),
        Some(&DEC_GRU5_RECURRENT_SUBIAS),
        Some(&DEC_GRU5_RECURRENT_WEIGHTS_INT8),
        Some(&DEC_GRU5_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_GRU5_RECURRENT_SCALE),
        DEC_GRU5_STATE_SIZE,
        DEC_GRU5_OUT_SIZE * 3,
    );
    model.dec_conv1 = linear_layer(
        Some(&DEC_CONV1_BIAS),
        Some(&DEC_CONV1_SUBIAS),
        Some(&DEC_CONV1_WEIGHTS_INT8),
        Some(&DEC_CONV1_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_CONV1_SCALE),
        DEC_CONV1_IN_SIZE * 2,
        DEC_CONV1_OUT_SIZE,
    );
    model.dec_conv2 = linear_layer(
        Some(&DEC_CONV2_BIAS),
        Some(&DEC_CONV2_SUBIAS),
        Some(&DEC_CONV2_WEIGHTS_INT8),
        Some(&DEC_CONV2_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_CONV2_SCALE),
        DEC_CONV2_IN_SIZE * 2,
        DEC_CONV2_OUT_SIZE,
    );
    model.dec_conv3 = linear_layer(
        Some(&DEC_CONV3_BIAS),
        Some(&DEC_CONV3_SUBIAS),
        Some(&DEC_CONV3_WEIGHTS_INT8),
        Some(&DEC_CONV3_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_CONV3_SCALE),
        DEC_CONV3_IN_SIZE * 2,
        DEC_CONV3_OUT_SIZE,
    );
    model.dec_conv4 = linear_layer(
        Some(&DEC_CONV4_BIAS),
        Some(&DEC_CONV4_SUBIAS),
        Some(&DEC_CONV4_WEIGHTS_INT8),
        Some(&DEC_CONV4_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_CONV4_SCALE),
        DEC_CONV4_IN_SIZE * 2,
        DEC_CONV4_OUT_SIZE,
    );
    model.dec_conv5 = linear_layer(
        Some(&DEC_CONV5_BIAS),
        Some(&DEC_CONV5_SUBIAS),
        Some(&DEC_CONV5_WEIGHTS_INT8),
        Some(&DEC_CONV5_WEIGHTS_FLOAT),
        None,
        None,
        Some(&DEC_CONV5_SCALE),
        DEC_CONV5_IN_SIZE * 2,
        DEC_CONV5_OUT_SIZE,
    );
}

fn conv1_cond_init(mem: &mut [f32], len: usize, dilation: usize, init: &mut i32) {
    if *init == 0 {
        for i in 0..dilation {
            let start = i * len;
            mem[start..start + len].fill(0.0);
        }
    }
    *init = 1;
}

pub(crate) fn rdovae_dec_init_states(
    state: &mut RdovaeDecState,
    model: &RdovaeDec,
    initial_state: &[f32],
    arch: i32,
) {
    let mut hidden = [0.0f32; DEC_HIDDEN_INIT_OUT_SIZE];
    let mut state_init = [0.0f32; DEC_GRU_INIT_OUT_SIZE];
    compute_generic_dense(&model.dec_hidden_init, &mut hidden, initial_state, ACTIVATION_TANH, arch);
    compute_generic_dense(&model.dec_gru_init, &mut state_init, &hidden, ACTIVATION_TANH, arch);
    let mut counter = 0usize;
    state.gru1_state.copy_from_slice(&state_init[counter..counter + DEC_GRU1_STATE_SIZE]);
    counter += DEC_GRU1_STATE_SIZE;
    state.gru2_state.copy_from_slice(&state_init[counter..counter + DEC_GRU2_STATE_SIZE]);
    counter += DEC_GRU2_STATE_SIZE;
    state.gru3_state.copy_from_slice(&state_init[counter..counter + DEC_GRU3_STATE_SIZE]);
    counter += DEC_GRU3_STATE_SIZE;
    state.gru4_state.copy_from_slice(&state_init[counter..counter + DEC_GRU4_STATE_SIZE]);
    counter += DEC_GRU4_STATE_SIZE;
    state.gru5_state.copy_from_slice(&state_init[counter..counter + DEC_GRU5_STATE_SIZE]);
    state.initialized = 0;
}

pub(crate) fn rdovae_decode_qframe(
    dec_state: &mut RdovaeDecState,
    model: &RdovaeDec,
    qframe: &mut [f32],
    input: &[f32],
    arch: i32,
) {
    debug_assert_eq!(qframe.len(), DEC_OUTPUT_OUT_SIZE);
    debug_assert_eq!(input.len(), DRED_LATENT_DIM);
    let mut buffer = [0.0f32; DEC_BUFFER_SIZE];
    let mut output_index = 0usize;

    compute_generic_dense(
        &model.dec_dense1,
        &mut buffer[output_index..output_index + DEC_DENSE1_OUT_SIZE],
        input,
        ACTIVATION_TANH,
        arch,
    );
    output_index += DEC_DENSE1_OUT_SIZE;

    compute_generic_gru(
        &model.dec_gru1_input,
        &model.dec_gru1_recurrent,
        &mut dec_state.gru1_state,
        &buffer[..output_index],
        arch,
    );
    compute_glu(
        &model.dec_glu1,
        &mut buffer[output_index..output_index + DEC_GRU1_OUT_SIZE],
        &dec_state.gru1_state,
        arch,
    );
    output_index += DEC_GRU1_OUT_SIZE;
    debug_assert_eq!(output_index, DEC_CONV1_IN_SIZE);
    conv1_cond_init(&mut dec_state.conv1_state, output_index, 1, &mut dec_state.initialized);
    let (conv1_in, conv1_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d(
        &model.dec_conv1,
        &mut conv1_out[..DEC_CONV1_OUT_SIZE],
        &mut dec_state.conv1_state,
        conv1_in,
        DEC_CONV1_IN_SIZE,
        ACTIVATION_TANH,
        arch,
    );
    output_index += DEC_CONV1_OUT_SIZE;

    compute_generic_gru(
        &model.dec_gru2_input,
        &model.dec_gru2_recurrent,
        &mut dec_state.gru2_state,
        &buffer[..output_index],
        arch,
    );
    compute_glu(
        &model.dec_glu2,
        &mut buffer[output_index..output_index + DEC_GRU2_OUT_SIZE],
        &dec_state.gru2_state,
        arch,
    );
    output_index += DEC_GRU2_OUT_SIZE;
    debug_assert_eq!(output_index, DEC_CONV2_IN_SIZE);
    conv1_cond_init(&mut dec_state.conv2_state, output_index, 1, &mut dec_state.initialized);
    let (conv2_in, conv2_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d(
        &model.dec_conv2,
        &mut conv2_out[..DEC_CONV2_OUT_SIZE],
        &mut dec_state.conv2_state,
        conv2_in,
        DEC_CONV2_IN_SIZE,
        ACTIVATION_TANH,
        arch,
    );
    output_index += DEC_CONV2_OUT_SIZE;

    compute_generic_gru(
        &model.dec_gru3_input,
        &model.dec_gru3_recurrent,
        &mut dec_state.gru3_state,
        &buffer[..output_index],
        arch,
    );
    compute_glu(
        &model.dec_glu3,
        &mut buffer[output_index..output_index + DEC_GRU3_OUT_SIZE],
        &dec_state.gru3_state,
        arch,
    );
    output_index += DEC_GRU3_OUT_SIZE;
    debug_assert_eq!(output_index, DEC_CONV3_IN_SIZE);
    conv1_cond_init(&mut dec_state.conv3_state, output_index, 1, &mut dec_state.initialized);
    let (conv3_in, conv3_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d(
        &model.dec_conv3,
        &mut conv3_out[..DEC_CONV3_OUT_SIZE],
        &mut dec_state.conv3_state,
        conv3_in,
        DEC_CONV3_IN_SIZE,
        ACTIVATION_TANH,
        arch,
    );
    output_index += DEC_CONV3_OUT_SIZE;

    compute_generic_gru(
        &model.dec_gru4_input,
        &model.dec_gru4_recurrent,
        &mut dec_state.gru4_state,
        &buffer[..output_index],
        arch,
    );
    compute_glu(
        &model.dec_glu4,
        &mut buffer[output_index..output_index + DEC_GRU4_OUT_SIZE],
        &dec_state.gru4_state,
        arch,
    );
    output_index += DEC_GRU4_OUT_SIZE;
    debug_assert_eq!(output_index, DEC_CONV4_IN_SIZE);
    conv1_cond_init(&mut dec_state.conv4_state, output_index, 1, &mut dec_state.initialized);
    let (conv4_in, conv4_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d(
        &model.dec_conv4,
        &mut conv4_out[..DEC_CONV4_OUT_SIZE],
        &mut dec_state.conv4_state,
        conv4_in,
        DEC_CONV4_IN_SIZE,
        ACTIVATION_TANH,
        arch,
    );
    output_index += DEC_CONV4_OUT_SIZE;

    compute_generic_gru(
        &model.dec_gru5_input,
        &model.dec_gru5_recurrent,
        &mut dec_state.gru5_state,
        &buffer[..output_index],
        arch,
    );
    compute_glu(
        &model.dec_glu5,
        &mut buffer[output_index..output_index + DEC_GRU5_OUT_SIZE],
        &dec_state.gru5_state,
        arch,
    );
    output_index += DEC_GRU5_OUT_SIZE;
    debug_assert_eq!(output_index, DEC_CONV5_IN_SIZE);
    conv1_cond_init(&mut dec_state.conv5_state, output_index, 1, &mut dec_state.initialized);
    let (conv5_in, conv5_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d(
        &model.dec_conv5,
        &mut conv5_out[..DEC_CONV5_OUT_SIZE],
        &mut dec_state.conv5_state,
        conv5_in,
        DEC_CONV5_IN_SIZE,
        ACTIVATION_TANH,
        arch,
    );
    output_index += DEC_CONV5_OUT_SIZE;

    compute_generic_dense(
        &model.dec_output,
        qframe,
        &buffer[..output_index],
        ACTIVATION_LINEAR,
        arch,
    );
}

pub(crate) fn rdovae_decode_all(
    model: &RdovaeDec,
    features: &mut [f32],
    state: &[f32],
    latents: &[f32],
    nb_latents: i32,
    arch: i32,
) {
    debug_assert!(state.len() >= DRED_STATE_DIM);
    let mut dec = RdovaeDecState::default();
    rdovae_dec_init_states(&mut dec, model, state, arch);
    let nb_latents = nb_latents.max(0) as usize;
    debug_assert!(latents.len() >= nb_latents * DRED_LATENT_DIM);
    debug_assert!(features.len() >= 4 * nb_latents * DRED_NUM_FEATURES);
    for i in (0..2 * nb_latents).step_by(2) {
        let feature_offset = 2 * i * DRED_NUM_FEATURES;
        let latent_offset = (i / 2) * DRED_LATENT_DIM;
        rdovae_decode_qframe(
            &mut dec,
            model,
            &mut features[feature_offset..feature_offset + DEC_OUTPUT_OUT_SIZE],
            &latents[latent_offset..latent_offset + DRED_LATENT_DIM],
            arch,
        );
    }
}
