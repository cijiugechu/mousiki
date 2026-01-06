pub mod a2nlsf;
pub mod ana_filt_bank_1;
pub mod apply_sine_window;
pub mod apply_sine_window_flp;
pub mod arm_silk_map;
pub mod autocorr;
pub mod autocorrelation_flp;
pub mod biquad_alt;
pub mod biquad_alt_neon_intr;
pub mod burg_modified;
pub mod burg_modified_flp;
pub mod bwexpander;
pub mod bwexpander_32;
pub mod bwexpander_flp;
pub mod check_control_input;
pub mod cng;
pub mod code_signs;
pub mod codebook;
pub mod control_audio_bandwidth;
pub mod control_codec;
pub mod control_snr;
pub mod corr_matrix;
pub mod corr_matrix_flp;
pub mod debug;
pub mod dec_api;
pub mod decode_core;
pub mod decode_frame;
pub mod decode_indices;
pub mod decode_parameters;
pub mod decode_pitch;
pub mod decode_pulses;
pub mod decoder;
pub mod decoder_control;
pub mod decoder_set_fs;
pub mod decoder_state;
pub mod enc_api;
pub mod encode_frame;
pub mod encode_frame_flp;
pub mod encode_indices;
pub mod encode_pulses;
pub mod encoder;
pub mod energy_flp;
pub mod errors;
pub mod find_lpc;
pub mod find_lpc_flp;
pub mod find_ltp;
pub mod find_ltp_flp;
pub mod find_pitch_lags;
pub mod find_pitch_lags_flp;
pub mod find_pred_coefs;
pub mod find_pred_coefs_flp;
pub mod gain_quant;
pub mod get_decoder_size;
pub mod get_encoder_size;
pub mod get_toc;
pub mod hp_variable_cutoff;
pub mod icdf;
pub mod init_decoder;
pub mod init_encoder;
pub mod inner_prod_aligned;
pub mod inner_product_flp;
pub mod inner_product_flp_avx2;
pub mod interpolate;
pub mod k2a;
pub mod k2a_flp;
pub mod k2a_q16;
pub mod lin2log;
pub mod load_osce_models;
pub mod log2lin;
pub mod lp_variable_cutoff;
pub mod lpc_analysis_filter;
pub mod lpc_analysis_filter_flp;
pub mod lpc_fit;
pub mod lpc_inv_pred_gain;
pub mod lpc_inv_pred_gain_flp;
pub mod ltp_analysis_filter;
pub mod ltp_analysis_filter_flp;
pub mod ltp_scale_ctrl;
pub mod ltp_scale_ctrl_flp;
pub mod nlsf2a;
pub mod nlsf_decode;
pub mod nlsf_del_dec_quant;
pub mod nlsf_encode;
pub mod nlsf_stabilize;
pub mod nlsf_unpack;
pub mod nlsf_vq;
pub mod nlsf_vq_weights_laroia;
pub mod noise_shape_analysis;
pub mod noise_shape_analysis_flp;
pub mod nsq;
pub mod nsq_del_dec;
pub mod nsq_del_dec_avx2;
pub mod nsq_del_dec_sse4_1;
pub mod nsq_sse4_1;
pub mod pitch_analysis_core;
pub mod pitch_analysis_core_flp;
pub mod pitch_est_tables;
pub mod plc;
pub mod process_gains;
pub mod process_gains_flp;
pub mod process_nlsfs;
pub mod quant_ltp_gains;
pub mod regularize_correlations;
pub mod regularize_correlations_flp;
pub mod resampler;
pub mod resampler_down2;
pub mod resampler_down2_3;
pub mod resampler_private_ar2;
pub mod resampler_private_down_fir;
pub mod resampler_private_iir_fir;
pub mod resampler_private_up2_hq;
pub mod resampler_rom;
pub mod residual_energy;
pub mod residual_energy16;
pub mod residual_energy_flp;
pub mod range_decoder;
pub mod scale_vector;
pub mod schur;
pub mod schur64;
pub mod schur_flp;
pub mod shell_coder;
pub mod sigm_q15;
pub mod sigproc_flp;
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
pub mod vad_sse4_1;
pub mod vector_ops;
pub mod vector_ops_fix_sse4_1;
pub mod vq_wmat_ec;
pub mod vq_wmat_ec_sse4_1;
pub mod warped_autocorrelation;
pub mod warped_autocorrelation_flp;
pub mod wrappers_flp;
pub mod x86_silk_map;

pub use check_control_input::EncControl;
pub use range_decoder::SilkRangeDecoder;
pub use decode_frame::{DecodeFlag, silk_decode_frame};
pub use decoder_control::DecoderControl;
pub use decoder_state::{DecoderState, PacketLossConcealmentState};
pub use gain_quant::MAX_NB_SUBFR;
pub use get_toc::{Toc, silk_get_toc};
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
