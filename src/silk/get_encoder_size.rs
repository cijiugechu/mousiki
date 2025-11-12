//! Port of `silk_Get_Encoder_Size` from `silk/enc_API.c`.
//!
//! The original helper reports the number of bytes required to hold the SILK
//! encoder super-structure so callers can allocate raw storage before
//! initialising the encoder state. The Rust port mirrors that behaviour by
//! measuring the [`Encoder`](crate::silk::encoder::Encoder) type that backs the
//! current encoder scaffolding.

use core::mem;

use super::encoder::Encoder;
use super::errors::SilkError;

/// Mirrors `silk_Get_Encoder_Size`.
///
/// # Returns
/// * [`Ok`]`(())` and writes the number of bytes required to hold the current
///   [`Encoder`] state into `size_bytes`.
pub fn get_encoder_size(size_bytes: &mut usize) -> Result<(), SilkError> {
    *size_bytes = mem::size_of::<Encoder>();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::get_encoder_size;
    use crate::silk::encoder::Encoder;

    #[test]
    fn reports_encoder_size_in_bytes() {
        let mut size = 0usize;
        assert!(get_encoder_size(&mut size).is_ok());
        assert_eq!(size, core::mem::size_of::<Encoder>());
    }
}
