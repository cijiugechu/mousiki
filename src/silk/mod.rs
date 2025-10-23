pub mod bwexpander;
pub mod bwexpander_32;
pub mod codebook;
pub mod decoder;
pub mod icdf;
pub mod interpolate;
pub mod lin2log;
pub mod log2lin;
pub mod resampler_down2;
pub mod sigm_q15;
pub mod sort;
pub mod sum_sqr_shift;
pub mod table_lsf_cos;
pub mod tables_gain;
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
