pub mod codebook;
pub mod decoder;
pub mod icdf;
pub mod lin2log;
pub mod log2lin;
pub mod sum_sqr_shift;

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
