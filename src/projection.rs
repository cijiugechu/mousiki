//! Ambisonics projection helpers, layout selection, and the projection
//! encoder/decoder front-ends.
//!
//! Mirrors `opus_projection_{encoder,decoder}.c` for mapping family 3 by
//! selecting the appropriate precomputed mixing/demixing matrices and wiring
//! them through the multistream encoder/decoder glue.

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::isqrt32;
use crate::celt::float2int16;
use crate::mapping_matrix::{
    MAPPING_MATRIX_FIFTHOA_DEMIXING, MAPPING_MATRIX_FIFTHOA_MIXING, MAPPING_MATRIX_FOA_DEMIXING,
    MAPPING_MATRIX_FOA_MIXING, MAPPING_MATRIX_FOURTHOA_DEMIXING, MAPPING_MATRIX_FOURTHOA_MIXING,
    MAPPING_MATRIX_SOA_DEMIXING, MAPPING_MATRIX_SOA_MIXING, MAPPING_MATRIX_TOA_DEMIXING,
    MAPPING_MATRIX_TOA_MIXING, MappingMatrix, MappingMatrixView, mapping_matrix_get_size,
    mapping_matrix_init, mapping_matrix_multiply_channel_in_float,
    mapping_matrix_multiply_channel_in_int24, mapping_matrix_multiply_channel_in_short,
    mapping_matrix_multiply_channel_out_float, mapping_matrix_multiply_channel_out_int24,
};
#[cfg(not(feature = "fixed_point"))]
use crate::mapping_matrix::mapping_matrix_multiply_channel_out_short;
use crate::opus_multistream::{
    OpusMultistreamDecoder, OpusMultistreamDecoderCtlRequest, OpusMultistreamDecoderError,
    OpusMultistreamEncoder, OpusMultistreamEncoderCtlRequest, OpusMultistreamEncoderError,
    opus_multistream_decode_native_with_handler, opus_multistream_decoder_create,
    opus_multistream_decoder_ctl, opus_multistream_decoder_get_size, opus_multistream_encode,
    opus_multistream_encoder_create, opus_multistream_encoder_ctl,
    opus_multistream_encoder_get_size,
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

    let order_plus_one =
        get_order_plus_one_from_channels(channels).ok_or(ProjectionError::BadArgument)?;
    let (streams, coupled_streams) =
        get_streams_from_channels(channels, order_plus_one).ok_or(ProjectionError::BadArgument)?;
    let (mixing, demixing) =
        select_matrices(order_plus_one).ok_or(ProjectionError::Unimplemented)?;

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
    let demixing_matrix_size_bytes = mapping_matrix_get_size(demixing.rows, demixing.cols)
        .ok_or(ProjectionError::BadArgument)?;

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

#[repr(C)]
struct OpusProjectionEncoderLayout {
    mixing_matrix_size_in_bytes: i32,
    demixing_matrix_size_in_bytes: i32,
}

#[repr(C)]
struct OpusProjectionDecoderLayout {
    demixing_matrix_size_in_bytes: i32,
}

#[inline]
fn align(value: usize) -> usize {
    #[repr(C)]
    struct AlignProbe {
        _tag: u8,
        _union: AlignUnion,
    }

    #[repr(C)]
    union AlignUnion {
        _ptr: *const (),
        _i32: i32,
        _f32: f32,
    }

    let alignment = core::mem::align_of::<AlignProbe>();
    value.div_ceil(alignment) * alignment
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusProjectionEncoderError {
    BadArgument,
    Unimplemented,
    SizeOverflow,
    Multistream(OpusMultistreamEncoderError),
}

impl From<ProjectionError> for OpusProjectionEncoderError {
    #[inline]
    fn from(value: ProjectionError) -> Self {
        match value {
            ProjectionError::BadArgument => Self::BadArgument,
            ProjectionError::Unimplemented => Self::Unimplemented,
            ProjectionError::SizeOverflow => Self::SizeOverflow,
        }
    }
}

impl From<OpusMultistreamEncoderError> for OpusProjectionEncoderError {
    #[inline]
    fn from(value: OpusMultistreamEncoderError) -> Self {
        Self::Multistream(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusProjectionDecoderError {
    BadArgument,
    Multistream(OpusMultistreamDecoderError),
}

impl From<OpusMultistreamDecoderError> for OpusProjectionDecoderError {
    #[inline]
    fn from(value: OpusMultistreamDecoderError) -> Self {
        Self::Multistream(value)
    }
}

/// Projection encoder state mirroring `OpusProjectionEncoder` from the
/// reference implementation.
#[derive(Debug)]
pub struct OpusProjectionEncoder<'mode> {
    layout: ProjectionLayout,
    ms_encoder: OpusMultistreamEncoder<'mode>,
}

impl<'mode> OpusProjectionEncoder<'mode> {
    #[inline]
    pub fn projection_layout(&self) -> &ProjectionLayout {
        &self.layout
    }

    #[inline]
    pub fn multistream_encoder(&mut self) -> &mut OpusMultistreamEncoder<'mode> {
        &mut self.ms_encoder
    }
}

/// Returns the number of bytes required to allocate a projection encoder.
#[must_use]
pub fn opus_projection_ambisonics_encoder_get_size(channels: usize, mapping_family: u8) -> Option<usize> {
    let layout = projection_layout(channels, mapping_family).ok()?;
    let ms_size = opus_multistream_encoder_get_size(layout.streams, layout.coupled_streams)?;
    align(core::mem::size_of::<OpusProjectionEncoderLayout>())
        .checked_add(layout.mixing_matrix_size_bytes)?
        .checked_add(layout.demixing_matrix_size_bytes)?
        .checked_add(ms_size)
}

/// Allocates and initialises a projection encoder state.
pub fn opus_projection_ambisonics_encoder_create(
    sample_rate: i32,
    channels: usize,
    mapping_family: u8,
    application: i32,
) -> Result<(OpusProjectionEncoder<'static>, usize, usize), OpusProjectionEncoderError> {
    let layout = projection_layout(channels, mapping_family)?;
    let mapping: Vec<u8> = (0..channels).map(|v| v as u8).collect();
    let ms_encoder = opus_multistream_encoder_create(
        sample_rate,
        channels,
        layout.streams,
        layout.coupled_streams,
        &mapping,
        application,
    )?;

    Ok((
        OpusProjectionEncoder { layout, ms_encoder },
        layout.streams,
        layout.coupled_streams,
    ))
}

/// Resets a projection encoder instance with a new ambisonics configuration.
pub fn opus_projection_ambisonics_encoder_init(
    encoder: &mut OpusProjectionEncoder<'_>,
    sample_rate: i32,
    channels: usize,
    mapping_family: u8,
    application: i32,
) -> Result<(usize, usize), OpusProjectionEncoderError> {
    let layout = projection_layout(channels, mapping_family)?;
    let mapping: Vec<u8> = (0..channels).map(|v| v as u8).collect();
    encoder.ms_encoder.init(
        sample_rate,
        channels,
        layout.streams,
        layout.coupled_streams,
        &mapping,
        application,
    )?;
    encoder.layout = layout;
    Ok((layout.streams, layout.coupled_streams))
}

pub enum OpusProjectionEncoderCtlRequest<'req> {
    GetDemixingMatrixSize(&'req mut usize),
    GetDemixingMatrixGain(&'req mut i32),
    GetDemixingMatrix(&'req mut [u8]),
    Multistream(OpusMultistreamEncoderCtlRequest<'req>),
}

pub fn opus_projection_encoder_ctl<'req>(
    encoder: &mut OpusProjectionEncoder<'_>,
    request: OpusProjectionEncoderCtlRequest<'req>,
) -> Result<(), OpusProjectionEncoderError> {
    match request {
        OpusProjectionEncoderCtlRequest::GetDemixingMatrixSize(out) => {
            *out = encoder
                .layout
                .demixing_subset_size_bytes()
                .map_err(OpusProjectionEncoderError::from)?;
        }
        OpusProjectionEncoderCtlRequest::GetDemixingMatrixGain(out) => {
            *out = encoder.layout.demixing.gain_db;
        }
        OpusProjectionEncoderCtlRequest::GetDemixingMatrix(out) => {
            write_demixing_matrix_subset(&encoder.layout, out).map_err(OpusProjectionEncoderError::from)?;
        }
        OpusProjectionEncoderCtlRequest::Multistream(req) => {
            opus_multistream_encoder_ctl(&mut encoder.ms_encoder, req)?;
        }
    }
    Ok(())
}

fn mix_i16(
    mixing_matrix: MappingMatrixView<'static>,
    channels: usize,
    pcm: &[i16],
    frame_size: usize,
) -> Vec<i16> {
    let mut mixed_res = vec![0.0f32; channels * frame_size];
    for row in 0..channels {
        mapping_matrix_multiply_channel_in_short(
            mixing_matrix,
            pcm,
            channels,
            &mut mixed_res[row..],
            row,
            channels,
            frame_size,
        );
    }

    let mut mixed_pcm = vec![0i16; channels * frame_size];
    for (dst, &sample) in mixed_pcm.iter_mut().zip(mixed_res.iter()) {
        *dst = float2int16(sample);
    }
    mixed_pcm
}

fn mix_float(
    mixing_matrix: MappingMatrixView<'static>,
    channels: usize,
    pcm: &[f32],
    frame_size: usize,
) -> Vec<i16> {
    let mut mixed_res = vec![0.0f32; channels * frame_size];
    for row in 0..channels {
        mapping_matrix_multiply_channel_in_float(
            mixing_matrix,
            pcm,
            channels,
            &mut mixed_res[row..],
            row,
            channels,
            frame_size,
        );
    }

    let mut mixed_pcm = vec![0i16; channels * frame_size];
    for (dst, &sample) in mixed_pcm.iter_mut().zip(mixed_res.iter()) {
        *dst = float2int16(sample);
    }
    mixed_pcm
}

fn mix_int24(
    mixing_matrix: MappingMatrixView<'static>,
    channels: usize,
    pcm: &[i32],
    frame_size: usize,
) -> Vec<i16> {
    let mut mixed_res = vec![0.0f32; channels * frame_size];
    for row in 0..channels {
        mapping_matrix_multiply_channel_in_int24(
            mixing_matrix,
            pcm,
            channels,
            &mut mixed_res[row..],
            row,
            channels,
            frame_size,
        );
    }

    let mut mixed_pcm = vec![0i16; channels * frame_size];
    for (dst, &sample) in mixed_pcm.iter_mut().zip(mixed_res.iter()) {
        *dst = float2int16(sample);
    }
    mixed_pcm
}

pub fn opus_projection_encode(
    encoder: &mut OpusProjectionEncoder<'_>,
    pcm: &[i16],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusProjectionEncoderError> {
    let channels = encoder.layout.channels;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusProjectionEncoderError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusProjectionEncoderError::BadArgument);
    }

    let mixed = mix_i16(encoder.layout.mixing, channels, &pcm[..required], frame_size);
    Ok(opus_multistream_encode(
        &mut encoder.ms_encoder,
        &mixed,
        frame_size,
        data,
    )?)
}

pub fn opus_projection_encode_float(
    encoder: &mut OpusProjectionEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusProjectionEncoderError> {
    let channels = encoder.layout.channels;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusProjectionEncoderError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusProjectionEncoderError::BadArgument);
    }

    let mixed = mix_float(encoder.layout.mixing, channels, &pcm[..required], frame_size);
    Ok(opus_multistream_encode(
        &mut encoder.ms_encoder,
        &mixed,
        frame_size,
        data,
    )?)
}

pub fn opus_projection_encode24(
    encoder: &mut OpusProjectionEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusProjectionEncoderError> {
    let channels = encoder.layout.channels;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusProjectionEncoderError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusProjectionEncoderError::BadArgument);
    }

    let mixed = mix_int24(encoder.layout.mixing, channels, &pcm[..required], frame_size);
    Ok(opus_multistream_encode(
        &mut encoder.ms_encoder,
        &mixed,
        frame_size,
        data,
    )?)
}

/// Projection decoder state mirroring `OpusProjectionDecoder` from the
/// reference implementation.
#[derive(Debug)]
pub struct OpusProjectionDecoder<'mode> {
    channels: usize,
    demixing_matrix: MappingMatrix,
    ms_decoder: OpusMultistreamDecoder<'mode>,
}

impl<'mode> OpusProjectionDecoder<'mode> {
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    pub fn multistream_decoder(&mut self) -> &mut OpusMultistreamDecoder<'mode> {
        &mut self.ms_decoder
    }
}

#[must_use]
pub fn opus_projection_decoder_get_size(
    channels: usize,
    streams: usize,
    coupled_streams: usize,
) -> Option<usize> {
    let matrix_size = mapping_matrix_get_size(streams + coupled_streams, channels)?;
    let decoder_size = opus_multistream_decoder_get_size(streams, coupled_streams)?;
    align(core::mem::size_of::<OpusProjectionDecoderLayout>())
        .checked_add(matrix_size)?
        .checked_add(decoder_size)
}

pub fn opus_projection_decoder_create(
    sample_rate: i32,
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    demixing_matrix: &[u8],
) -> Result<OpusProjectionDecoder<'static>, OpusProjectionDecoderError> {
    let nb_input_streams = streams + coupled_streams;
    let expected_size = nb_input_streams
        .checked_mul(channels)
        .and_then(|cells| cells.checked_mul(core::mem::size_of::<i16>()))
        .ok_or(OpusProjectionDecoderError::BadArgument)?;
    if demixing_matrix.len() != expected_size {
        return Err(OpusProjectionDecoderError::BadArgument);
    }

    let mut matrix_data = Vec::with_capacity(nb_input_streams * channels);
    for chunk in demixing_matrix.chunks_exact(2) {
        matrix_data.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }

    let demixing_matrix = mapping_matrix_init(
        channels,
        nb_input_streams,
        0,
        &matrix_data,
        demixing_matrix.len(),
    );
    let mapping: Vec<u8> = (0..channels).map(|v| v as u8).collect();
    let ms_decoder =
        opus_multistream_decoder_create(sample_rate, channels, streams, coupled_streams, &mapping)?;

    Ok(OpusProjectionDecoder {
        channels,
        demixing_matrix,
        ms_decoder,
    })
}

pub fn opus_projection_decoder_init(
    decoder: &mut OpusProjectionDecoder<'_>,
    sample_rate: i32,
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    demixing_matrix: &[u8],
) -> Result<(), OpusProjectionDecoderError> {
    let nb_input_streams = streams + coupled_streams;
    let expected_size = nb_input_streams
        .checked_mul(channels)
        .and_then(|cells| cells.checked_mul(core::mem::size_of::<i16>()))
        .ok_or(OpusProjectionDecoderError::BadArgument)?;
    if demixing_matrix.len() != expected_size {
        return Err(OpusProjectionDecoderError::BadArgument);
    }

    let mut matrix_data = Vec::with_capacity(nb_input_streams * channels);
    for chunk in demixing_matrix.chunks_exact(2) {
        matrix_data.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }

    decoder.demixing_matrix = mapping_matrix_init(
        channels,
        nb_input_streams,
        0,
        &matrix_data,
        demixing_matrix.len(),
    );
    decoder.channels = channels;
    let mapping: Vec<u8> = (0..channels).map(|v| v as u8).collect();
    decoder
        .ms_decoder
        .init(sample_rate, channels, streams, coupled_streams, &mapping)?;
    Ok(())
}

#[cfg(feature = "fixed_point")]
const OPTIONAL_CLIP: bool = false;
#[cfg(not(feature = "fixed_point"))]
const OPTIONAL_CLIP: bool = true;

pub fn opus_projection_decode(
    decoder: &mut OpusProjectionDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [i16],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusProjectionDecoderError> {
    let channels = decoder.channels;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusProjectionDecoderError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusProjectionDecoderError::BadArgument);
    }

    pcm[..required].fill(0);
    let matrix = decoder.demixing_matrix.as_view();

    Ok(opus_multistream_decode_native_with_handler(
        &mut decoder.ms_decoder,
        data,
        len,
        pcm,
        frame_size,
        decode_fec,
        OPTIONAL_CLIP,
        |dst, dst_stride, dst_channel, src, frame_size, src_offset| {
            let Some((src_data, src_stride)) = src else {
                return;
            };
            if src_offset >= src_data.len() {
                return;
            }
            #[cfg(feature = "fixed_point")]
            {
                for frame in 0..frame_size {
                    let src_index = src_offset + frame * src_stride;
                    let Some(sample) = src_data.get(src_index).copied() else {
                        break;
                    };
                    let input_sample = sample as i16;
                    for row in 0..dst_stride {
                        let coeff = i32::from(matrix.cell(row, dst_channel));
                        let tmp = coeff * i32::from(input_sample);
                        let delta = (tmp + 16_384) >> 15;
                        let dst_index = frame * dst_stride + row;
                        if dst_index >= dst.len() {
                            break;
                        }
                        let updated = i32::from(dst[dst_index]).wrapping_add(delta);
                        dst[dst_index] = updated as i16;
                    }
                }
            }
            #[cfg(not(feature = "fixed_point"))]
            mapping_matrix_multiply_channel_out_short(
                matrix,
                &src_data[src_offset..],
                dst_channel,
                src_stride,
                dst,
                dst_stride,
                frame_size,
            );
        },
    )?)
}

pub fn opus_projection_decode_float(
    decoder: &mut OpusProjectionDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [f32],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusProjectionDecoderError> {
    let channels = decoder.channels;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusProjectionDecoderError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusProjectionDecoderError::BadArgument);
    }

    pcm[..required].fill(0.0);
    let matrix = decoder.demixing_matrix.as_view();

    Ok(opus_multistream_decode_native_with_handler(
        &mut decoder.ms_decoder,
        data,
        len,
        pcm,
        frame_size,
        decode_fec,
        false,
        |dst, dst_stride, dst_channel, src, frame_size, src_offset| {
            let Some((src_data, src_stride)) = src else {
                return;
            };
            if src_offset >= src_data.len() {
                return;
            }
            mapping_matrix_multiply_channel_out_float(
                matrix,
                &src_data[src_offset..],
                dst_channel,
                src_stride,
                dst,
                dst_stride,
                frame_size,
            );
        },
    )?)
}

pub fn opus_projection_decode24(
    decoder: &mut OpusProjectionDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [i32],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusProjectionDecoderError> {
    let channels = decoder.channels;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusProjectionDecoderError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusProjectionDecoderError::BadArgument);
    }

    pcm[..required].fill(0);
    let matrix = decoder.demixing_matrix.as_view();

    Ok(opus_multistream_decode_native_with_handler(
        &mut decoder.ms_decoder,
        data,
        len,
        pcm,
        frame_size,
        decode_fec,
        false,
        |dst, dst_stride, dst_channel, src, frame_size, src_offset| {
            let Some((src_data, src_stride)) = src else {
                return;
            };
            if src_offset >= src_data.len() {
                return;
            }
            mapping_matrix_multiply_channel_out_int24(
                matrix,
                &src_data[src_offset..],
                dst_channel,
                src_stride,
                dst,
                dst_stride,
                frame_size,
            );
        },
    )?)
}

pub enum OpusProjectionDecoderCtlRequest<'req> {
    Multistream(OpusMultistreamDecoderCtlRequest<'req>),
}

pub fn opus_projection_decoder_ctl<'req>(
    decoder: &mut OpusProjectionDecoder<'_>,
    request: OpusProjectionDecoderCtlRequest<'req>,
) -> Result<(), OpusProjectionDecoderError> {
    match request {
        OpusProjectionDecoderCtlRequest::Multistream(req) => {
            opus_multistream_decoder_ctl(&mut decoder.ms_decoder, req)?;
        }
    }
    Ok(())
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

fn get_streams_from_channels(channels: usize, order_plus_one: usize) -> Option<(usize, usize)> {
    // Mapping family 3 only supports orders with precomputed matrices.
    if !(2..=6).contains(&order_plus_one) {
        return None;
    }

    let streams = channels.div_ceil(2);
    let coupled_streams = channels / 2;
    Some((streams, coupled_streams))
}

fn select_matrices(
    order_plus_one: usize,
) -> Option<(MappingMatrixView<'static>, MappingMatrixView<'static>)> {
    match order_plus_one {
        2 => Some((MAPPING_MATRIX_FOA_MIXING, MAPPING_MATRIX_FOA_DEMIXING)),
        3 => Some((MAPPING_MATRIX_SOA_MIXING, MAPPING_MATRIX_SOA_DEMIXING)),
        4 => Some((MAPPING_MATRIX_TOA_MIXING, MAPPING_MATRIX_TOA_DEMIXING)),
        5 => Some((
            MAPPING_MATRIX_FOURTHOA_MIXING,
            MAPPING_MATRIX_FOURTHOA_DEMIXING,
        )),
        6 => Some((
            MAPPING_MATRIX_FIFTHOA_MIXING,
            MAPPING_MATRIX_FIFTHOA_DEMIXING,
        )),
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
                expected.push(layout.demixing.data[layout.demixing.rows * input_stream + channel]);
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
                expected.push(layout.demixing.data[layout.demixing.rows * input_stream + channel]);
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

    #[test]
    fn projection_encoder_and_decoder_round_trip_silence() {
        let (mut encoder, streams, coupled) =
            opus_projection_ambisonics_encoder_create(48_000, 4, 3, 2048).expect("encoder");
        assert_eq!(streams, 2);
        assert_eq!(coupled, 2);

        let mut matrix_size = 0usize;
        opus_projection_encoder_ctl(
            &mut encoder,
            OpusProjectionEncoderCtlRequest::GetDemixingMatrixSize(&mut matrix_size),
        )
        .expect("ctl");
        assert_eq!(matrix_size, 4 * 4 * 2);

        let mut demix = vec![0u8; matrix_size];
        opus_projection_encoder_ctl(
            &mut encoder,
            OpusProjectionEncoderCtlRequest::GetDemixingMatrix(demix.as_mut_slice()),
        )
        .expect("ctl");

        let mut decoder =
            opus_projection_decoder_create(48_000, 4, streams, coupled, &demix).expect("decoder");

        let pcm_in = vec![0i16; 4 * 960];
        let mut packet = vec![0u8; 4000];
        let len = opus_projection_encode(&mut encoder, &pcm_in, 960, &mut packet).expect("encode");
        assert!(len > 0);

        let mut pcm_out = vec![0i16; 4 * 960];
        let decoded =
            opus_projection_decode(&mut decoder, &packet, len, &mut pcm_out, 960, false)
                .expect("decode");
        assert_eq!(decoded, 960);

        let mapping = [0u8, 1, 2, 3];
        let mut ms_decoder =
            opus_multistream_decoder_create(48_000, 4, streams, coupled, &mapping).expect("ms");
        let mut ms_pcm = vec![0i16; 4 * 960];
        let _ = crate::opus_multistream::opus_multistream_decode(
            &mut ms_decoder,
            &packet,
            len,
            &mut ms_pcm,
            960,
            false,
        )
        .expect("ms decode");
        let ms_max_abs = ms_pcm.iter().map(|&v| i32::from(v).abs()).max().unwrap_or(0);

        let max_abs = pcm_out
            .iter()
            .map(|&v| i32::from(v).abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_abs <= 64,
            "projection max abs sample was {max_abs} (multistream max was {ms_max_abs})"
        );
    }
}
