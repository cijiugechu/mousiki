pub mod a2nlsf;
pub mod ana_filt_bank_1;
pub mod apply_sine_window;
pub mod biquad_alt;
pub mod bwexpander;
pub mod bwexpander_32;
pub mod check_control_input;
pub mod code_signs;
pub mod codebook;
pub mod decoder;
pub mod errors;
pub mod gain_quant;
pub mod icdf;
pub mod inner_prod_aligned;
pub mod interpolate;
pub mod k2a;
pub mod k2a_q16;
pub mod lin2log;
pub mod log2lin;
pub mod lp_variable_cutoff;
pub mod lpc_analysis_filter;
pub mod lpc_fit;
pub mod lpc_inv_pred_gain;
pub mod nlsf2a;
pub mod nlsf_decode;
pub mod nlsf_del_dec_quant;
pub mod nlsf_encode;
pub mod nlsf_stabilize;
pub mod nlsf_unpack;
pub mod nlsf_vq;
pub mod nlsf_vq_weights_laroia;
pub mod pitch_est_tables;
pub mod regularize_correlations;
pub mod resampler;
pub mod resampler_down2;
pub mod resampler_down2_3;
pub mod resampler_private_ar2;
pub mod resampler_private_down_fir;
pub mod resampler_private_iir_fir;
pub mod resampler_private_up2_hq;
pub mod resampler_rom;
pub mod schur;
pub mod shell_coder;
pub mod sigm_q15;
pub mod sort;
pub mod stereo_decode_pred;
pub mod stereo_encode_pred;
pub mod stereo_find_predictor;
pub mod stereo_lr_to_ms;
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
pub mod tuning_parameters;
pub mod vq_wmat_ec;

pub use check_control_input::EncControl;
pub use gain_quant::MAX_NB_SUBFR;
pub use interpolate::MAX_LPC_ORDER;
pub use stereo_lr_to_ms::{StereoConversionResult, StereoEncState};
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
