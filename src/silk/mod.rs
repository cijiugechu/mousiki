pub mod codebook;
pub mod decoder;
pub mod icdf;
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
