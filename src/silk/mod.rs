pub mod a2nlsf;
pub mod ana_filt_bank_1;
pub mod apply_sine_window;
pub mod autocorr;
pub mod biquad_alt;
pub mod bwexpander;
pub mod bwexpander_32;
pub mod check_control_input;
pub mod cng;
pub mod code_signs;
pub mod codebook;
pub mod control_snr;
pub mod corr_matrix;
pub mod decode_indices;
pub mod decoder;
pub mod decoder_set_fs;
pub mod encoder;
pub mod errors;
pub mod gain_quant;
pub mod hp_variable_cutoff;
pub mod icdf;
pub mod inner_prod_aligned;
pub mod interpolate;
pub mod k2a;
pub mod k2a_q16;
pub mod lin2log;
pub mod log2lin;
pub mod load_osce_models;
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
pub mod process_nlsfs;
pub mod quant_ltp_gains;
pub mod regularize_correlations;
pub mod resampler;
pub mod resampler_down2;
pub mod resampler_down2_3;
pub mod resampler_private_ar2;
pub mod resampler_private_down_fir;
pub mod resampler_private_iir_fir;
pub mod resampler_private_up2_hq;
pub mod resampler_rom;
pub mod residual_energy16;
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
pub mod vad;
pub mod vector_ops;
pub mod vq_wmat_ec;
pub mod warped_autocorrelation;

pub use check_control_input::EncControl;
pub use gain_quant::MAX_NB_SUBFR;
pub use interpolate::MAX_LPC_ORDER;
pub use stereo_lr_to_ms::{StereoConversionResult, StereoEncState};
pub use tables_nlsf_cb_wb::SilkNlsfCb;
pub use warped_autocorrelation::MAX_SHAPE_LPC_ORDER;
pub const MIN_LPC_ORDER: usize = 10;
pub const MAX_FRAMES_PER_PACKET: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSignalType {
    Inactive,
    Unvoiced,
    Voiced,
}

impl From<FrameSignalType> for i32 {
    fn from(value: FrameSignalType) -> Self {
        match value {
            FrameSignalType::Inactive => 0,
            FrameSignalType::Unvoiced => 1,
            FrameSignalType::Voiced => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameQuantizationOffsetType {
    Low,
    High,
}
