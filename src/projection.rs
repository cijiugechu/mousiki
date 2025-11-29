//! Ambisonics projection helpers and layout selection.
//!
//! This module begins the port of `opus_projection_{encoder,decoder}.c` by
//! exposing the channel validation and matrix selection logic for ambisonics
//! mapping family 3. The full projection encoder/decoder front-ends still
//! depend on the unported multistream glue, but the matrix plumbing and size
//! calculations are available for subsequent integration.

use crate::celt::isqrt32;
use crate::mapping_matrix::{
    mapping_matrix_get_size, MappingMatrixView, MAPPING_MATRIX_FIFTHOA_DEMIXING,
    MAPPING_MATRIX_FIFTHOA_MIXING, MAPPING_MATRIX_FOA_DEMIXING, MAPPING_MATRIX_FOA_MIXING,
    MAPPING_MATRIX_FOURTHOA_DEMIXING, MAPPING_MATRIX_FOURTHOA_MIXING,
    MAPPING_MATRIX_SOA_DEMIXING, MAPPING_MATRIX_SOA_MIXING, MAPPING_MATRIX_TOA_DEMIXING,
    MAPPING_MATRIX_TOA_MIXING,
};

/// Errors surfaced by the projection helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionError {
    /// Inputs fall outside the valid ambisonics ranges.
    BadArgument,
    /// The requested mapping family or order is not yet ported.
    Unimplemented,
    /// Intermediate size calculations overflowed `usize`.
    SizeOverflow,
}

/// Derived projection layout for an ambisonics configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionLayout {
    pub channels: usize,
    pub streams: usize,
    pub coupled_streams: usize,
    pub order_plus_one: usize,
    pub mixing: MappingMatrixView<'static>,
    pub demixing: MappingMatrixView<'static>,
    pub mixing_matrix_size_bytes: usize,
    pub demixing_matrix_size_bytes: usize,
}

impl ProjectionLayout {
    /// Returns the size in bytes of the demixing submatrix that is exposed via
    /// the projection CTLs (channels x input streams, 16-bit little-endian).
    pub fn demixing_subset_size_bytes(&self) -> Result<usize, ProjectionError> {
        let nb_input_streams = self
            .streams
            .checked_add(self.coupled_streams)
            .ok_or(ProjectionError::SizeOverflow)?;
        self.channels
            .checked_mul(nb_input_streams)
            .and_then(|cells| cells.checked_mul(core::mem::size_of::<i16>()))
            .ok_or(ProjectionError::SizeOverflow)
    }
}

/// Selects the ambisonics layout and precomputed matrices for mapping family 3.
///
/// Mirrors the validation flow inside `opus_projection_ambisonics_encoder_init`
/// without wiring the multistream encoder yet.
pub fn projection_layout(
    channels: usize,
    mapping_family: u8,
) -> Result<ProjectionLayout, ProjectionError> {
    const PROJECTION_FAMILY: u8 = 3;
    if mapping_family != PROJECTION_FAMILY {
        return Err(ProjectionError::Unimplemented);
    }

    let order_plus_one = get_order_plus_one_from_channels(channels)
        .ok_or(ProjectionError::BadArgument)?;
    let (streams, coupled_streams) = get_streams_from_channels(channels, order_plus_one)
        .ok_or(ProjectionError::BadArgument)?;
    let (mixing, demixing) = select_matrices(order_plus_one)
        .ok_or(ProjectionError::Unimplemented)?;

    // Ensure the selected matrices can cover the requested layout.
    if streams + coupled_streams > mixing.rows
        || channels > mixing.cols
        || channels > demixing.rows
        || streams + coupled_streams > demixing.cols
    {
        return Err(ProjectionError::BadArgument);
    }

    let mixing_matrix_size_bytes =
        mapping_matrix_get_size(mixing.rows, mixing.cols).ok_or(ProjectionError::BadArgument)?;
    let demixing_matrix_size_bytes =
        mapping_matrix_get_size(demixing.rows, demixing.cols).ok_or(ProjectionError::BadArgument)?;

    Ok(ProjectionLayout {
        channels,
        streams,
        coupled_streams,
        order_plus_one,
        mixing,
        demixing,
        mixing_matrix_size_bytes,
        demixing_matrix_size_bytes,
    })
}

/// Writes the exposed portion of the demixing matrix into `output` in the same
/// layout expected by the projection encoder CTL.
pub fn write_demixing_matrix_subset(
    layout: &ProjectionLayout,
    output: &mut [u8],
) -> Result<(), ProjectionError> {
    let nb_input_streams = layout
        .streams
        .checked_add(layout.coupled_streams)
        .ok_or(ProjectionError::SizeOverflow)?;
    let expected_size = layout.demixing_subset_size_bytes()?;
    if output.len() != expected_size {
        return Err(ProjectionError::BadArgument);
    }

    let mut offset = 0;
    for input_stream in 0..nb_input_streams {
        for channel in 0..layout.channels {
            let value = layout.demixing.cell(channel, input_stream).to_le_bytes();
            output[offset] = value[0];
            output[offset + 1] = value[1];
            offset += 2;
        }
    }

    Ok(())
}

/// Returns the exposed demixing submatrix size in bytes.
pub fn demixing_matrix_size(layout: &ProjectionLayout) -> Result<usize, ProjectionError> {
    layout.demixing_subset_size_bytes()
}

/// Returns the demixing matrix gain in 7.8 fixed-point dB.
pub fn demixing_matrix_gain(layout: &ProjectionLayout) -> i32 {
    layout.demixing.gain_db
}

fn get_order_plus_one_from_channels(channels: usize) -> Option<usize> {
    // Allowed channel counts: (1 + n)^2 + 2j for n = 0..14 and j = 0 or 1.
    if !(1..=227).contains(&channels) {
        return None;
    }

    let order_plus_one = isqrt32(channels as u32) as usize;
    let acn_channels = order_plus_one.checked_mul(order_plus_one)?;
    let nondiegetic_channels = channels.checked_sub(acn_channels)?;
    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return None;
    }

    Some(order_plus_one)
}

fn get_streams_from_channels(
    channels: usize,
    order_plus_one: usize,
) -> Option<(usize, usize)> {
    // Mapping family 3 only supports orders with precomputed matrices.
    if !(2..=6).contains(&order_plus_one) {
        return None;
    }

    let streams = channels.div_ceil(2);
    let coupled_streams = channels / 2;
    Some((streams, coupled_streams))
}

fn select_matrices(order_plus_one: usize) -> Option<(MappingMatrixView<'static>, MappingMatrixView<'static>)> {
    match order_plus_one {
        2 => Some((MAPPING_MATRIX_FOA_MIXING, MAPPING_MATRIX_FOA_DEMIXING)),
        3 => Some((MAPPING_MATRIX_SOA_MIXING, MAPPING_MATRIX_SOA_DEMIXING)),
        4 => Some((MAPPING_MATRIX_TOA_MIXING, MAPPING_MATRIX_TOA_DEMIXING)),
        5 => Some((
            MAPPING_MATRIX_FOURTHOA_MIXING,
            MAPPING_MATRIX_FOURTHOA_DEMIXING,
        )),
        6 => Some((MAPPING_MATRIX_FIFTHOA_MIXING, MAPPING_MATRIX_FIFTHOA_DEMIXING)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn rejects_unhandled_mapping_family() {
        let err = projection_layout(4, 0).unwrap_err();
        assert_eq!(err, ProjectionError::Unimplemented);
    }

    #[test]
    fn rejects_invalid_channel_counts() {
        let err = projection_layout(7, 3).unwrap_err();
        assert_eq!(err, ProjectionError::BadArgument);
    }

    #[test]
    fn computes_layout_for_foa_channels() {
        let layout = projection_layout(4, 3).expect("layout");
        assert_eq!(layout.order_plus_one, 2);
        assert_eq!(layout.streams, 2);
        assert_eq!(layout.coupled_streams, 2);
        assert_eq!(layout.mixing, MAPPING_MATRIX_FOA_MIXING);
        assert_eq!(layout.demixing, MAPPING_MATRIX_FOA_DEMIXING);
        // Ensure size helpers use the aligned matrix footprint.
        assert!(layout.mixing_matrix_size_bytes > 0);
        assert!(layout.demixing_matrix_size_bytes > 0);
    }

    #[test]
    fn writes_demixing_subset_in_expected_order() {
        let layout = projection_layout(4, 3).expect("layout");
        let mut buffer = vec![0u8; layout.demixing_subset_size_bytes().unwrap()];
        write_demixing_matrix_subset(&layout, &mut buffer).expect("write");

        let nb_input_streams = layout.streams + layout.coupled_streams;
        let decoded: Vec<i16> = buffer
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let mut expected = Vec::new();
        for input_stream in 0..nb_input_streams {
            for channel in 0..layout.channels {
                expected.push(
                    layout
                        .demixing
                        .data[layout.demixing.rows * input_stream + channel],
                );
            }
        }

        assert_eq!(decoded, expected);
    }

    #[test]
    fn demixing_helpers_export_metadata_and_matrix() {
        let layout = projection_layout(9, 3).expect("layout");

        let size = demixing_matrix_size(&layout).expect("size");
        let expected_size = layout.demixing_subset_size_bytes().unwrap();
        assert_eq!(size, expected_size);

        let gain = demixing_matrix_gain(&layout);
        assert_eq!(gain, layout.demixing.gain_db);

        let mut buffer = vec![0u8; size];
        write_demixing_matrix_subset(&layout, &mut buffer).expect("matrix");

        let decoded: Vec<i16> = buffer
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let nb_input_streams = layout.streams + layout.coupled_streams;
        let mut expected = Vec::new();
        for input_stream in 0..nb_input_streams {
            for channel in 0..layout.channels {
                expected.push(
                    layout
                        .demixing
                        .data[layout.demixing.rows * input_stream + channel],
                );
            }
        }

        assert_eq!(decoded, expected);
    }

    #[test]
    fn demixing_matrix_rejects_mismatched_buffer_size() {
        let layout = projection_layout(4, 3).expect("layout");
        let size = layout.demixing_subset_size_bytes().unwrap();
        let mut buffer = vec![0u8; size - 1];

        let err = write_demixing_matrix_subset(&layout, &mut buffer).unwrap_err();
        assert_eq!(err, ProjectionError::BadArgument);
    }
}
