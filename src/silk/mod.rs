pub mod ana_filt_bank_1;
pub mod biquad_alt;
pub mod bwexpander;
pub mod bwexpander_32;
pub mod codebook;
pub mod decoder;
pub mod icdf;
pub mod inner_prod_aligned;
pub mod interpolate;
pub mod k2a;
pub mod lin2log;
pub mod log2lin;
pub mod lpc_inv_pred_gain;
pub mod nlsf_stabilize;
pub mod nlsf_unpack;
pub mod nlsf_vq_weights_laroia;
pub mod pitch_est_tables;
pub mod resampler_down2;
pub mod resampler_private_ar2;
pub mod resampler_private_up2_hq;
pub mod resampler_rom;
pub mod sigm_q15;
pub mod sort;
pub mod stereo_decode_pred;
pub mod stereo_encode_pred;
pub mod stereo_find_predictor;
pub mod stereo_ms_to_lr;
pub mod stereo_quant_pred;
pub mod sum_sqr_shift;
pub mod table_lsf_cos;
pub mod tables_gain;
pub mod tables_ltp;
pub mod tables_nlsf_cb_nb_mb;
pub mod tables_nlsf_cb_wb;
pub mod tables_other;
pub mod tables_pitch_lag;
pub mod tables_pulses_per_block;

pub use interpolate::MAX_LPC_ORDER;
pub use tables_nlsf_cb_wb::SilkNlsfCb;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameSignalType {
    Inactive,
    Unvoiced,
    Voiced,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameQuantizationOffsetType {
    Low,
    High,
}
