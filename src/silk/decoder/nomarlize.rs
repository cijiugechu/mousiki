use crate::silk::codebook::NORMALIZED_LSF_STAGE_TWO_INDEX_WIDEBAND;
use core::ops::{Deref, DerefMut};

const fn get_max_d_lpc() -> usize {
    const WIDE_LEN: usize = NORMALIZED_LSF_STAGE_TWO_INDEX_WIDEBAND[0].len();
    WIDE_LEN
}

pub(crate) const MAX_D_LPC: usize = get_max_d_lpc();

pub(crate) const MAX_A_Q12_SETS: usize = 4;

pub(crate) const MAX_D2_LPC: usize = (MAX_D_LPC / 2) + 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlsfQ15 {
    data: [i16; MAX_D_LPC],
    len: usize,
}

impl NlsfQ15 {
    pub fn new(len: usize) -> Self {
        debug_assert!(len <= MAX_D_LPC);

        Self {
            data: [0; MAX_D_LPC],
            len,
        }
    }

    pub fn from_slice(slice: &[i16]) -> Self {
        let mut instance = Self::new(slice.len());
        instance.as_mut_slice().copy_from_slice(slice);
        instance
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[i16] {
        &self.data[..self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [i16] {
        &mut self.data[..self.len]
    }
}

impl Deref for NlsfQ15 {
    type Target = [i16];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for NlsfQ15 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct A32Q17 {
    data: [i32; MAX_D_LPC],
    len: usize,
}

impl A32Q17 {
    pub const fn new(len: usize) -> Self {
        debug_assert!(len <= MAX_D_LPC);

        Self {
            data: [0; MAX_D_LPC],
            len,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[i32] {
        &self.data[..self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [i32] {
        &mut self.data[..self.len]
    }
}

impl Deref for A32Q17 {
    type Target = [i32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for A32Q17 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aq12Coefficients {
    data: [f32; MAX_D_LPC],
    len: usize,
}

impl Aq12Coefficients {
    pub const fn new(len: usize) -> Self {
        debug_assert!(len <= MAX_D_LPC);

        Self {
            data: [0.0; MAX_D_LPC],
            len,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data[..self.len]
    }
}

impl Deref for Aq12Coefficients {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for Aq12Coefficients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Aq12List {
    data: [[f32; MAX_D_LPC]; MAX_A_Q12_SETS],
    lens: [usize; MAX_A_Q12_SETS],
    len: usize,
}

impl Aq12List {
    pub const fn new() -> Self {
        Self {
            data: [[0.0; MAX_D_LPC]; MAX_A_Q12_SETS],
            lens: [0; MAX_A_Q12_SETS],
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, coeffs: &Aq12Coefficients) {
        debug_assert!(self.len < MAX_A_Q12_SETS);

        let idx = self.len;
        self.data[idx][..coeffs.len()].copy_from_slice(coeffs.as_slice());
        self.lens[idx] = coeffs.len();
        self.len += 1;
    }

    pub fn get(&self, idx: usize) -> &[f32] {
        debug_assert!(idx < self.len);
        &self.data[idx][..self.lens[idx]]
    }
}
