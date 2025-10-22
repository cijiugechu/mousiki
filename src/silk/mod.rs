pub mod codebook;
pub mod decoder;
pub mod icdf;
pub mod interpolate;
pub mod lin2log;
pub mod log2lin;
pub mod sigm_q15;
pub mod sort;
pub mod sum_sqr_shift;
pub mod tables_pulses_per_block;
pub mod tables_pitch_lag;

pub use interpolate::MAX_LPC_ORDER;

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
