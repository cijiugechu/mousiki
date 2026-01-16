#![cfg(feature = "deep_plc")]

use crate::dnn_utils::linear_layer_from_blob;
use crate::dnn_weights::{WeightBlob, WeightError};
use crate::dred_constants::DRED_NUM_FEATURES;
use crate::nnet::LinearLayer;

const NB_BANDS: usize = 18;
const PLC_FEATURES_LEN: usize = 2 * NB_BANDS + DRED_NUM_FEATURES + 1;

#[derive(Clone, Debug, Default)]
pub(crate) struct PlcModel {
    pub plc_dense_in: LinearLayer,
    pub plc_gru1_input: LinearLayer,
    pub plc_gru1_recurrent: LinearLayer,
    pub plc_gru2_input: LinearLayer,
    pub plc_gru2_recurrent: LinearLayer,
    pub plc_dense_out: LinearLayer,
}

impl PlcModel {
    pub(crate) fn from_weights(data: &[u8]) -> Result<Self, WeightError> {
        let blob = WeightBlob::parse(data)?;
        let mut model = Self::default();
        init_plc_model_from_weights(&mut model, &blob)?;
        Ok(model)
    }
}

fn init_plc_model_from_weights(
    model: &mut PlcModel,
    blob: &WeightBlob<'_>,
) -> Result<(), WeightError> {
    model.plc_dense_in = linear_layer_from_blob(
        blob,
        Some("plc_dense_in_bias"),
        None,
        None,
        Some("plc_dense_in_weights_float"),
        None,
        None,
        None,
        Some(PLC_FEATURES_LEN),
        None,
    )?;

    model.plc_gru1_input = linear_layer_from_blob(
        blob,
        Some("plc_gru1_input_bias"),
        None,
        None,
        Some("plc_gru1_input_weights_float"),
        None,
        None,
        None,
        Some(model.plc_dense_in.nb_outputs),
        None,
    )?;
    if model.plc_gru1_input.nb_outputs % 3 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let gru1_units = model.plc_gru1_input.nb_outputs / 3;
    model.plc_gru1_recurrent = linear_layer_from_blob(
        blob,
        Some("plc_gru1_recurrent_bias"),
        None,
        None,
        Some("plc_gru1_recurrent_weights_float"),
        None,
        None,
        None,
        Some(gru1_units),
        Some(model.plc_gru1_input.nb_outputs),
    )?;

    model.plc_gru2_input = linear_layer_from_blob(
        blob,
        Some("plc_gru2_input_bias"),
        None,
        None,
        Some("plc_gru2_input_weights_float"),
        None,
        None,
        None,
        Some(gru1_units),
        None,
    )?;
    if model.plc_gru2_input.nb_outputs % 3 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let gru2_units = model.plc_gru2_input.nb_outputs / 3;
    model.plc_gru2_recurrent = linear_layer_from_blob(
        blob,
        Some("plc_gru2_recurrent_bias"),
        None,
        None,
        Some("plc_gru2_recurrent_weights_float"),
        None,
        None,
        None,
        Some(gru2_units),
        Some(model.plc_gru2_input.nb_outputs),
    )?;

    model.plc_dense_out = linear_layer_from_blob(
        blob,
        Some("plc_dense_out_bias"),
        None,
        None,
        Some("plc_dense_out_weights_float"),
        None,
        None,
        None,
        Some(gru2_units),
        Some(DRED_NUM_FEATURES),
    )?;

    Ok(())
}
