#![cfg(feature = "deep_plc")]

use alloc::boxed::Box;
use alloc::vec::Vec;
use crate::dnn_weights::{WeightArray, WeightBlob, WeightError};
use crate::nnet::LinearLayer;

fn find_array<'a>(
    blob: &'a WeightBlob<'a>,
    name: &'static str,
) -> Result<&'a WeightArray<'a>, WeightError> {
    blob.find(name).ok_or(WeightError::MissingArray(name))
}

#[allow(dead_code)]
fn array_len(array: &WeightArray<'_>, elem_size: usize) -> Result<usize, WeightError> {
    if array.size == 0 || array.size % elem_size != 0 {
        return Err(WeightError::InvalidBlob);
    }
    Ok(array.size / elem_size)
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
    for &byte in data {
        values.push(byte as i8);
    }
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

fn load_optional_f32(
    blob: &WeightBlob<'_>,
    name: Option<&'static str>,
) -> Result<Option<&'static [f32]>, WeightError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let array = find_array(blob, name)?;
    Ok(Some(leak_f32(array.data)?))
}

fn load_optional_i8(
    blob: &WeightBlob<'_>,
    name: Option<&'static str>,
) -> Result<Option<&'static [i8]>, WeightError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let array = find_array(blob, name)?;
    Ok(Some(leak_i8(array.data)?))
}

fn load_optional_i32(
    blob: &WeightBlob<'_>,
    name: Option<&'static str>,
) -> Result<Option<&'static [i32]>, WeightError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let array = find_array(blob, name)?;
    Ok(Some(leak_i32(array.data)?))
}

fn len_optional(array: Option<&'static [f32]>) -> Option<usize> {
    array.map(<[f32]>::len)
}

pub(crate) fn linear_layer_from_blob(
    blob: &WeightBlob<'_>,
    bias_name: Option<&'static str>,
    subias_name: Option<&'static str>,
    weights_name: Option<&'static str>,
    float_weights_name: Option<&'static str>,
    weights_idx_name: Option<&'static str>,
    diag_name: Option<&'static str>,
    scale_name: Option<&'static str>,
    expected_inputs: Option<usize>,
    expected_outputs: Option<usize>,
) -> Result<LinearLayer, WeightError> {
    let bias = load_optional_f32(blob, bias_name)?;
    let subias = load_optional_f32(blob, subias_name)?;
    let float_weights = load_optional_f32(blob, float_weights_name)?;
    let weights = load_optional_i8(blob, weights_name)?;
    let weights_idx = load_optional_i32(blob, weights_idx_name)?;
    let diag = load_optional_f32(blob, diag_name)?;
    let scale = load_optional_f32(blob, scale_name)?;

    let weight_len = if let Some(weights) = float_weights {
        weights.len()
    } else if let Some(weights) = weights {
        weights.len()
    } else {
        return Err(WeightError::InvalidBlob);
    };

    let mut nb_outputs = expected_outputs
        .or_else(|| len_optional(bias))
        .or_else(|| len_optional(subias))
        .or_else(|| len_optional(scale));

    if nb_outputs.is_none() {
        if let Some(inputs) = expected_inputs {
            if inputs == 0 || weight_len % inputs != 0 {
                return Err(WeightError::InvalidBlob);
            }
            nb_outputs = Some(weight_len / inputs);
        }
    }

    let Some(nb_outputs) = nb_outputs else {
        return Err(WeightError::InvalidBlob);
    };
    if nb_outputs == 0 {
        return Err(WeightError::InvalidBlob);
    }

    let nb_inputs = if let Some(inputs) = expected_inputs {
        inputs
    } else {
        if weight_len % nb_outputs != 0 {
            return Err(WeightError::InvalidBlob);
        }
        weight_len / nb_outputs
    };

    if nb_inputs == 0 || nb_inputs * nb_outputs != weight_len {
        return Err(WeightError::InvalidBlob);
    }

    if let Some(bias) = bias {
        if bias.len() != nb_outputs {
            return Err(WeightError::InvalidBlob);
        }
    }
    if let Some(subias) = subias {
        if subias.len() != nb_outputs {
            return Err(WeightError::InvalidBlob);
        }
    }
    if let Some(scale) = scale {
        if scale.len() != nb_outputs {
            return Err(WeightError::InvalidBlob);
        }
    }
    if let Some(diag) = diag {
        if diag.len() % 3 != 0 {
            return Err(WeightError::InvalidBlob);
        }
    }

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

#[allow(dead_code)]
pub(crate) fn array_f32_len(
    blob: &WeightBlob<'_>,
    name: &'static str,
) -> Result<usize, WeightError> {
    let array = find_array(blob, name)?;
    array_len(array, 4)
}
