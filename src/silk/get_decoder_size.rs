//! Port of `silk_Get_Decoder_Size` from `silk/dec_API.c`.
//!
//! The original helper simply reports the number of bytes needed to hold the
//! decoder super-structure so callers can preallocate memory before calling
//! `silk_InitDecoder`. The Rust port mirrors that behaviour by measuring the
//! [`Decoder`](crate::silk::dec_api::Decoder) type that backs the translated SILK
//! decoder implementation.

use core::mem;

use super::dec_api::Decoder;
use super::errors::SilkError;

/// Mirrors `silk_Get_Decoder_Size`.
///
/// # Returns
/// * [`Ok`]`(())` and writes the number of bytes required to hold the current
///   [`Decoder`] state into `size_bytes`.
pub fn get_decoder_size(size_bytes: &mut usize) -> Result<(), SilkError> {
    *size_bytes = mem::size_of::<Decoder>();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::get_decoder_size;
    use crate::silk::dec_api::Decoder;

    #[test]
    fn reports_decoder_size_in_bytes() {
        let mut size = 0usize;
        assert!(get_decoder_size(&mut size).is_ok());
        assert_eq!(size, core::mem::size_of::<Decoder>());
    }
}
