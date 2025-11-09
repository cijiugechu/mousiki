//! Port of `silk_LoadOSCEModels` from `silk/dec_API.c`.
//!
//! The upstream helper wires optional OSCE (Opus Speech Coding Enhancer)
//! models into the decoder state. The current Rust port builds without OSCE
//! support, so the helper mirrors the reference behaviour in non-OSCE builds
//! by simply reporting success.

use super::decoder::Decoder;
use super::errors::SilkError;

/// Mirrors `silk_LoadOSCEModels`.
///
/// When OSCE is disabled the C reference returns [`SilkError::NoError`] without
/// touching the decoder state. We match that behaviour by ignoring the
/// optional payload completely until an OSCE-capable implementation lands.
pub fn load_osce_models(_decoder: &mut Decoder, _data: Option<&[u8]>) -> Result<(), SilkError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::load_osce_models;
    use crate::silk::decoder::DecoderBuilder;

    #[test]
    fn succeeds_without_payload() {
        let mut decoder = DecoderBuilder::new().build();
        assert_eq!(load_osce_models(&mut decoder, None), Ok(()));
    }

    #[test]
    fn accepts_inline_payloads() {
        let mut decoder = DecoderBuilder::new().build();
        let blob = [1_u8, 2, 3, 4];
        assert_eq!(load_osce_models(&mut decoder, Some(&blob)), Ok(()));
    }
}
