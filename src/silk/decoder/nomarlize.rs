use crate::silk::codebook::NORMALIZED_LSF_STAGE_TWO_INDEX_WIDEBAND;
use core::ops::{Deref, DerefMut};

const fn get_max_d_lpc() -> usize {
    const WIDE_LEN: usize = NORMALIZED_LSF_STAGE_TWO_INDEX_WIDEBAND[0].len();
    WIDE_LEN
}

pub(crate) const MAX_D_LPC: usize = get_max_d_lpc();

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResQ10 {
    Wide([i16; 16]),
    NarrowOrMedium([i16; 10]),
}

impl Deref for ResQ10 {
    type Target = [i16];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Wide(arr) => arr,
            Self::NarrowOrMedium(arr) => arr,
        }
    }
}

impl DerefMut for ResQ10 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Wide(arr) => arr,
            Self::NarrowOrMedium(arr) => arr,
        }
    }
}

impl ResQ10 {
    pub const fn d_lpc(&self) -> usize {
        match self {
            Self::Wide(_) => 16,
            Self::NarrowOrMedium(_) => 10,
        }
    }
}
