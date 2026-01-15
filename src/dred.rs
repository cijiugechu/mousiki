//! Deep REDundancy (DRED) stubs and helpers.
//!
//! The DRED feature is not fully ported yet. The APIs mirror the C surface
//! while returning `Unimplemented` until the model and coding paths land.

use alloc::vec;
use alloc::vec::Vec;
use crate::celt::{ec_laplace_decode_p0, ec_tell, EcDec, OpusRes};
use crate::celt::opus_select_arch;
use crate::celt::select_celt_float2int16_impl;
#[cfg(not(feature = "fixed_point"))]
use crate::celt::{CELT_SIG_SCALE, float2int};
use crate::dred_constants::{
    DRED_LATENT_DIM, DRED_MAX_DATA_SIZE, DRED_NUM_FEATURES, DRED_NUM_REDUNDANCY_FRAMES,
    DRED_STATE_DIM,
};
use crate::dred_stats_data::{
    DRED_LATENT_P0_Q8, DRED_LATENT_QUANT_SCALES_Q8, DRED_LATENT_R_Q8, DRED_STATE_P0_Q8,
    DRED_STATE_QUANT_SCALES_Q8, DRED_STATE_R_Q8,
};
#[cfg(feature = "dred")]
use crate::dred_rdovae_dec::{rdovae_decode_all, RdovaeDec};
use crate::extensions::{ExtensionError, OpusExtensionIterator};
use crate::opus_decoder::{OpusDecodeError, OpusDecoder, opus_decode_native};
use crate::packet::{opus_packet_get_samples_per_frame, opus_packet_parse_impl, PacketError};

const DRED_EXTENSION_ID: u8 = 126;
const DRED_EXPERIMENTAL_VERSION: u8 = 10;
const DRED_EXPERIMENTAL_BYTES: usize = 2;
const DRED_FRAME_OFFSET_DIVISOR: i32 = 120;
const DRED_FEC_FEATURES_LEN: usize =
    2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES;
const DRED_LATENTS_LEN: usize = (DRED_NUM_REDUNDANCY_FRAMES / 2) * DRED_LATENT_DIM;

/// Errors surfaced by the DRED helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusDredError {
    BadArgument,
    BufferTooSmall,
    InvalidPacket,
    InternalError,
    Unimplemented,
}

impl OpusDredError {
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::InvalidPacket => -4,
            Self::Unimplemented => -5,
        }
    }
}

impl From<PacketError> for OpusDredError {
    #[inline]
    fn from(value: PacketError) -> Self {
        match value {
            PacketError::BadArgument => Self::BadArgument,
            PacketError::InvalidPacket => Self::InvalidPacket,
        }
    }
}

impl From<ExtensionError> for OpusDredError {
    #[inline]
    fn from(value: ExtensionError) -> Self {
        match value {
            ExtensionError::BadArgument => Self::BadArgument,
            ExtensionError::BufferTooSmall => Self::BufferTooSmall,
            ExtensionError::InvalidPacket => Self::InvalidPacket,
        }
    }
}

impl From<OpusDecodeError> for OpusDredError {
    #[inline]
    fn from(value: OpusDecodeError) -> Self {
        match value {
            OpusDecodeError::BadArgument => Self::BadArgument,
            OpusDecodeError::BufferTooSmall => Self::BufferTooSmall,
            OpusDecodeError::InvalidPacket => Self::InvalidPacket,
            OpusDecodeError::InternalError => Self::InternalError,
            OpusDecodeError::Unimplemented => Self::Unimplemented,
        }
    }
}

/// Opaque DRED decoder state.
#[derive(Debug, Default)]
pub struct OpusDredDecoder {
    #[cfg(feature = "dred")]
    model: RdovaeDec,
    loaded: bool,
    arch: i32,
}

/// Opaque DRED packet state.
#[derive(Debug, Clone)]
pub struct OpusDred {
    fec_features: [f32; DRED_FEC_FEATURES_LEN],
    state: [f32; DRED_STATE_DIM],
    latents: [f32; DRED_LATENTS_LEN],
    nb_latents: i32,
    process_stage: i32,
    dred_offset: i32,
}

impl Default for OpusDred {
    fn default() -> Self {
        Self {
            fec_features: [0.0; DRED_FEC_FEATURES_LEN],
            state: [0.0; DRED_STATE_DIM],
            latents: [0.0; DRED_LATENTS_LEN],
            nb_latents: 0,
            process_stage: 0,
            dred_offset: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DredPayload<'a> {
    payload: &'a [u8],
    dred_frame_offset: i32,
}

fn dred_decode_latents(
    dec: &mut EcDec<'_>,
    output: &mut [f32],
    scale: &[u8],
    r: &[u8],
    p0: &[u8],
) {
    debug_assert_eq!(output.len(), scale.len());
    debug_assert_eq!(output.len(), r.len());
    debug_assert_eq!(output.len(), p0.len());

    for (idx, out) in output.iter_mut().enumerate() {
        let q = if r[idx] == 0 || p0[idx] == 255 {
            0
        } else {
            ec_laplace_decode_p0(
                dec,
                u16::from(p0[idx]) << 7,
                u16::from(r[idx]) << 7,
            )
        };
        let denom = if scale[idx] == 0 { 1 } else { scale[idx] };
        *out = (q as f32) * 256.0 / f32::from(denom);
    }
}

fn compute_quantizer(q0: i32, d_q: i32, qmax: i32, index: i32) -> i32 {
    const D_Q_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];
    debug_assert!(
        (0..D_Q_TABLE.len() as i32).contains(&d_q),
        "dQ index out of range"
    );
    let quant = q0 + (D_Q_TABLE[d_q as usize] * index + 8) / 16;
    quant.min(qmax)
}

fn dred_ec_decode(
    dec: &mut OpusDred,
    bytes: &[u8],
    min_feature_frames: i32,
    dred_frame_offset: i32,
) -> Result<i32, OpusDredError> {
    debug_assert!(
        DRED_NUM_REDUNDANCY_FRAMES % 2 == 0,
        "redundancy frame count must be even"
    );

    let mut buffer = [0u8; DRED_MAX_DATA_SIZE];
    if bytes.len() > buffer.len() {
        return Err(OpusDredError::BufferTooSmall);
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    let mut ec = EcDec::new(&mut buffer[..bytes.len()]);

    let q0 = ec.dec_uint(16) as i32;
    let d_q = ec.dec_uint(8) as i32;
    let extra_offset = if ec.dec_uint(2) != 0 {
        32 * ec.dec_uint(256) as i32
    } else {
        0
    };
    dec.dred_offset = 16 - ec.dec_uint(32) as i32 - extra_offset + dred_frame_offset;

    let mut qmax = 15;
    if q0 < 14 && d_q > 0 {
        let nvals = 15 - (q0 + 1);
        let ft = 2 * nvals;
        let s = ec.decode(ft as u32) as i32;
        if s >= nvals {
            qmax = q0 + (s - nvals) + 1;
            ec.update(s as u32, (s + 1) as u32, ft as u32);
        } else {
            ec.update(0, nvals as u32, ft as u32);
        }
    }

    let state_qoffset = (q0 * DRED_STATE_DIM as i32) as usize;
    dred_decode_latents(
        &mut ec,
        &mut dec.state,
        &DRED_STATE_QUANT_SCALES_Q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
        &DRED_STATE_R_Q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
        &DRED_STATE_P0_Q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
    );

    let max_frames = ((min_feature_frames + 1) / 2)
        .clamp(0, DRED_NUM_REDUNDANCY_FRAMES as i32) as usize;
    let mut i = 0usize;
    while i < max_frames {
        if 8 * bytes.len() as i32 - ec_tell(ec.ctx()) <= 7 {
            break;
        }
        let q_level = compute_quantizer(q0, d_q, qmax, (i / 2) as i32);
        let offset = (q_level * DRED_LATENT_DIM as i32) as usize;
        let latent_start = (i / 2) * DRED_LATENT_DIM;
        dred_decode_latents(
            &mut ec,
            &mut dec.latents[latent_start..latent_start + DRED_LATENT_DIM],
            &DRED_LATENT_QUANT_SCALES_Q8[offset..offset + DRED_LATENT_DIM],
            &DRED_LATENT_R_Q8[offset..offset + DRED_LATENT_DIM],
            &DRED_LATENT_P0_Q8[offset..offset + DRED_LATENT_DIM],
        );
        i += 2;
    }

    dec.process_stage = 1;
    dec.nb_latents = (i / 2) as i32;
    Ok(dec.nb_latents)
}

fn dred_find_payload<'a>(data: &'a [u8]) -> Result<Option<DredPayload<'a>>, OpusDredError> {
    let parsed = opus_packet_parse_impl(data, data.len(), false)?;
    let frame_size = opus_packet_get_samples_per_frame(data, 48_000)?;
    let frame_size =
        i32::try_from(frame_size).map_err(|_| OpusDredError::InvalidPacket)?;
    let mut iter = OpusExtensionIterator::new(parsed.padding, parsed.frame_count);

    loop {
        let Some(ext) = iter.find(DRED_EXTENSION_ID)? else {
            return Ok(None);
        };

        let ext_len = usize::try_from(ext.len).map_err(|_| OpusDredError::InvalidPacket)?;
        if ext_len > ext.data.len() {
            return Err(OpusDredError::InvalidPacket);
        }

        let dred_frame_offset = ext
            .frame
            .checked_mul(frame_size)
            .and_then(|value| value.checked_div(DRED_FRAME_OFFSET_DIVISOR))
            .ok_or(OpusDredError::InvalidPacket)?;

        if ext_len > DRED_EXPERIMENTAL_BYTES
            && ext.data.len() >= DRED_EXPERIMENTAL_BYTES
            && ext.data[0] == b'D'
            && ext.data[1] == DRED_EXPERIMENTAL_VERSION
        {
            let payload = &ext.data[DRED_EXPERIMENTAL_BYTES..ext_len];
            return Ok(Some(DredPayload {
                payload,
                dred_frame_offset,
            }));
        }
    }
}

/// Mirrors `opus_dred_decoder_get_size`.
#[inline]
pub fn opus_dred_decoder_get_size() -> usize {
    core::mem::size_of::<OpusDredDecoder>()
}

/// Mirrors `opus_dred_decoder_init`.
pub fn opus_dred_decoder_init(decoder: &mut OpusDredDecoder) -> Result<(), OpusDredError> {
    decoder.loaded = false;
    decoder.arch = opus_select_arch();
    #[cfg(feature = "dred")]
    {
        decoder.model = RdovaeDec::new();
        decoder.loaded = true;
    }
    Ok(())
}

/// Mirrors `opus_dred_decoder_create`.
pub fn opus_dred_decoder_create() -> Result<OpusDredDecoder, OpusDredError> {
    let mut decoder = OpusDredDecoder::default();
    opus_dred_decoder_init(&mut decoder)?;
    Ok(decoder)
}

/// Mirrors `opus_dred_decoder_destroy`.
#[inline]
pub fn opus_dred_decoder_destroy(_decoder: OpusDredDecoder) {}

/// Strongly-typed replacement for the DRED decoder CTL dispatcher.
pub enum OpusDredDecoderCtlRequest<'req> {
    SetDnnBlob(&'req [u8]),
}

/// Mirrors `opus_dred_decoder_ctl`.
pub fn opus_dred_decoder_ctl(
    _decoder: &mut OpusDredDecoder,
    _request: OpusDredDecoderCtlRequest<'_>,
) -> Result<(), OpusDredError> {
    Err(OpusDredError::Unimplemented)
}

/// Mirrors `opus_dred_get_size`.
pub fn opus_dred_get_size() -> usize {
    if cfg!(feature = "dred") {
        core::mem::size_of::<OpusDred>()
    } else {
        0
    }
}

/// Mirrors `opus_dred_alloc`.
pub fn opus_dred_alloc() -> Result<OpusDred, OpusDredError> {
    if cfg!(feature = "dred") {
        Ok(OpusDred::default())
    } else {
        Err(OpusDredError::Unimplemented)
    }
}

/// Mirrors `opus_dred_free`.
#[inline]
pub fn opus_dred_free(_dred: OpusDred) {}

/// Mirrors `opus_dred_parse`.
#[allow(clippy::too_many_arguments)]
pub fn opus_dred_parse(
    decoder: &OpusDredDecoder,
    dred: &mut OpusDred,
    data: &[u8],
    max_dred_samples: i32,
    sampling_rate: i32,
    dred_end: Option<&mut i32>,
    defer_processing: bool,
) -> Result<i32, OpusDredError> {
    if !cfg!(feature = "dred") {
        return Err(OpusDredError::Unimplemented);
    }

    if !decoder.loaded {
        return Err(OpusDredError::Unimplemented);
    }

    dred.process_stage = -1;

    if let Some(payload) = dred_find_payload(data)? {
        let DredPayload {
            payload: dred_payload,
            dred_frame_offset,
        } = payload;
        let offset = 100 * max_dred_samples / sampling_rate;
        let min_feature_frames = (2 + offset)
            .min((2 * DRED_NUM_REDUNDANCY_FRAMES) as i32);
        dred_ec_decode(dred, dred_payload, min_feature_frames, dred_frame_offset)?;
        if !defer_processing {
            let src = dred.clone();
            opus_dred_process(decoder, &src, dred)?;
        }
        if let Some(out) = dred_end {
            *out = (-(dred.dred_offset) * sampling_rate / 400).max(0);
        }
        return Ok(
            (dred.nb_latents * sampling_rate / 25
                - dred.dred_offset * sampling_rate / 400)
                .max(0),
        );
    }

    if let Some(out) = dred_end {
        *out = 0;
    }
    Ok(0)
}

/// Mirrors `opus_dred_process`.
pub fn opus_dred_process(
    decoder: &OpusDredDecoder,
    src: &OpusDred,
    dst: &mut OpusDred,
) -> Result<(), OpusDredError> {
    if !cfg!(feature = "dred") {
        return Err(OpusDredError::Unimplemented);
    }

    if src.process_stage != 1 && src.process_stage != 2 {
        return Err(OpusDredError::BadArgument);
    }

    if !decoder.loaded {
        return Err(OpusDredError::Unimplemented);
    }

    if !core::ptr::eq(src, dst) {
        *dst = src.clone();
    }

    if dst.process_stage == 2 {
        return Ok(());
    }

    #[cfg(feature = "dred")]
    rdovae_decode_all(
        &decoder.model,
        &mut dst.fec_features,
        &dst.state,
        &dst.latents,
        dst.nb_latents,
        decoder.arch,
    );
    dst.process_stage = 2;
    Ok(())
}

fn res_to_int24(sample: OpusRes) -> i32 {
    #[cfg(feature = "fixed_point")]
    {
        crate::celt::res2int24(crate::celt::float2res(sample))
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        let scale = CELT_SIG_SCALE * 256.0;
        let scaled = (sample * scale).clamp(-8_388_608.0, 8_388_607.0);
        float2int(scaled)
    }
}

/// Mirrors `opus_decoder_dred_decode`.
pub fn opus_decoder_dred_decode(
    decoder: &mut OpusDecoder<'_>,
    _dred: &OpusDred,
    _dred_offset: i32,
    pcm: &mut [i16],
    frame_size: usize,
) -> Result<usize, OpusDredError> {
    if !cfg!(feature = "dred") {
        return Err(OpusDredError::Unimplemented);
    }

    if frame_size == 0 {
        return Err(OpusDredError::BadArgument);
    }

    let channels = usize::try_from(decoder.channels).map_err(|_| OpusDredError::BadArgument)?;
    if channels == 0 || channels > 2 {
        return Err(OpusDredError::BadArgument);
    }

    let total_samples = frame_size
        .checked_mul(channels)
        .ok_or(OpusDredError::BadArgument)?;
    if pcm.len() < total_samples {
        return Err(OpusDredError::BufferTooSmall);
    }

    let mut out: Vec<OpusRes> = vec![OpusRes::default(); total_samples];
    let decoded = opus_decode_native(
        decoder,
        None,
        0,
        &mut out,
        frame_size,
        false,
        false,
        None,
        true,
    )?;

    let decoded_samples = decoded
        .checked_mul(channels)
        .ok_or(OpusDredError::BadArgument)?;
    if pcm.len() < decoded_samples {
        return Err(OpusDredError::BufferTooSmall);
    }

    select_celt_float2int16_impl(decoder.arch())(
        &out[..decoded_samples],
        &mut pcm[..decoded_samples],
    );

    Ok(decoded)
}

/// Mirrors `opus_decoder_dred_decode24`.
pub fn opus_decoder_dred_decode24(
    decoder: &mut OpusDecoder<'_>,
    _dred: &OpusDred,
    _dred_offset: i32,
    pcm: &mut [i32],
    frame_size: usize,
) -> Result<usize, OpusDredError> {
    if !cfg!(feature = "dred") {
        return Err(OpusDredError::Unimplemented);
    }

    if frame_size == 0 {
        return Err(OpusDredError::BadArgument);
    }

    let channels = usize::try_from(decoder.channels).map_err(|_| OpusDredError::BadArgument)?;
    if channels == 0 || channels > 2 {
        return Err(OpusDredError::BadArgument);
    }

    let total_samples = frame_size
        .checked_mul(channels)
        .ok_or(OpusDredError::BadArgument)?;
    if pcm.len() < total_samples {
        return Err(OpusDredError::BufferTooSmall);
    }

    let mut out: Vec<OpusRes> = vec![OpusRes::default(); total_samples];
    let decoded = opus_decode_native(
        decoder,
        None,
        0,
        &mut out,
        frame_size,
        false,
        false,
        None,
        true,
    )?;

    let decoded_samples = decoded
        .checked_mul(channels)
        .ok_or(OpusDredError::BadArgument)?;
    if pcm.len() < decoded_samples {
        return Err(OpusDredError::BufferTooSmall);
    }

    for (dst, &src) in pcm.iter_mut().take(decoded_samples).zip(out.iter()) {
        *dst = res_to_int24(src);
    }

    Ok(decoded)
}

/// Mirrors `opus_decoder_dred_decode_float`.
pub fn opus_decoder_dred_decode_float(
    decoder: &mut OpusDecoder<'_>,
    _dred: &OpusDred,
    _dred_offset: i32,
    pcm: &mut [f32],
    frame_size: usize,
) -> Result<usize, OpusDredError> {
    if !cfg!(feature = "dred") {
        return Err(OpusDredError::Unimplemented);
    }

    if frame_size == 0 {
        return Err(OpusDredError::BadArgument);
    }

    opus_decode_native(decoder, None, 0, pcm, frame_size, false, false, None, false)
        .map_err(OpusDredError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::{opus_packet_extensions_generate, OpusExtensionData};
    use crate::opus_decoder::opus_decoder_create;
    use alloc::vec::Vec;
    use alloc::vec;

    fn build_packet_with_padding(frame_count: usize, frame_len: usize, padding: &[u8]) -> Vec<u8> {
        assert!(frame_count > 0 && frame_count < 64);
        assert!(padding.len() > 0 && padding.len() < 255);

        let mut packet =
            Vec::with_capacity(3 + frame_count * frame_len + padding.len());
        packet.push(0x03);
        packet.push(0x40 | frame_count as u8);
        packet.push(padding.len() as u8);
        packet.resize(packet.len() + frame_count * frame_len, 0);
        packet.extend_from_slice(padding);
        packet
    }

    fn build_dred_padding(frame_count: usize, frame: i32, payload: &[u8]) -> Vec<u8> {
        let mut ext_bytes = Vec::with_capacity(DRED_EXPERIMENTAL_BYTES + payload.len());
        ext_bytes.push(b'D');
        ext_bytes.push(DRED_EXPERIMENTAL_VERSION);
        ext_bytes.extend_from_slice(payload);

        let ext = OpusExtensionData {
            id: DRED_EXTENSION_ID,
            frame,
            data: &ext_bytes,
            len: i32::try_from(ext_bytes.len()).expect("ext len fits i32"),
        };

        let max_len = 255usize;
        let required = opus_packet_extensions_generate(None, max_len, &[ext], frame_count, false)
            .expect("generate ext len");
        let mut padding = Vec::with_capacity(required);
        padding.resize(required, 0);
        let written = opus_packet_extensions_generate(
            Some(&mut padding),
            required,
            &[ext],
            frame_count,
            false,
        )
        .expect("generate ext bytes");
        assert_eq!(written, required);
        padding
    }

    #[test]
    fn dred_size_and_alloc_match_feature_state() {
        let size = opus_dred_get_size();
        if cfg!(feature = "dred") {
            assert!(size > 0);
            let dred = opus_dred_alloc().expect("dred alloc");
            opus_dred_free(dred);
        } else {
            assert_eq!(size, 0);
            assert_eq!(opus_dred_alloc().unwrap_err(), OpusDredError::Unimplemented);
        }
    }

    #[test]
    fn dred_parse_succeeds_without_payload() {
        if !cfg!(feature = "dred") {
            return;
        }

        let decoder = opus_dred_decoder_create().expect("decoder");
        let mut dred = opus_dred_alloc().expect("dred alloc");
        let data = [0u8; 4];
        let mut dred_end = -1;
        let decoded =
            opus_dred_parse(&decoder, &mut dred, &data, 48000, 48000, Some(&mut dred_end), false)
                .expect("parse");
        assert_eq!(decoded, 0);
        assert_eq!(dred_end, 0);
        opus_dred_free(dred);
        opus_dred_decoder_destroy(decoder);
    }

    #[test]
    fn dred_find_payload_returns_none_without_extension() {
        let packet = [0x00u8];
        let payload = dred_find_payload(&packet).expect("parse");
        assert!(payload.is_none());
    }

    #[test]
    fn dred_find_payload_extracts_payload_and_offset() {
        let payload_bytes = [0xAA, 0xBB];
        let padding = build_dred_padding(2, 1, &payload_bytes);
        let packet = build_packet_with_padding(2, 1, &padding);

        let payload = dred_find_payload(&packet)
            .expect("parse")
            .expect("payload");
        assert_eq!(payload.payload, payload_bytes);
        let frame_size = opus_packet_get_samples_per_frame(&packet, 48_000)
            .expect("frame size") as i32;
        let expected_offset = frame_size / DRED_FRAME_OFFSET_DIVISOR;
        assert_eq!(payload.dred_frame_offset, expected_offset);
    }

    #[test]
    fn compute_quantizer_matches_reference() {
        assert_eq!(super::compute_quantizer(6, 0, 15, 0), 6);
        assert_eq!(super::compute_quantizer(6, 1, 15, 8), 7);
        assert_eq!(super::compute_quantizer(14, 7, 15, 10), 15);
    }

    #[test]
    fn dred_random_payloads_do_not_break_processing() {
        if !cfg!(feature = "dred") {
            return;
        }

        const MAX_EXTENSION_SIZE: usize = 200;
        const ITERATIONS: usize = 1_000;

        #[derive(Clone)]
        struct FastRand {
            rz: u32,
            rw: u32,
        }

        impl FastRand {
            fn new(seed: u32) -> Self {
                Self { rz: seed, rw: seed }
            }

            fn next(&mut self) -> u32 {
                self.rz = 36969u32
                    .wrapping_mul(self.rz & 0xFFFF)
                    .wrapping_add(self.rz >> 16);
                self.rw = 18000u32
                    .wrapping_mul(self.rw & 0xFFFF)
                    .wrapping_add(self.rw >> 16);
                (self.rz << 16).wrapping_add(self.rw)
            }
        }

        let decoder = opus_dred_decoder_create().expect("decoder");
        let mut dred = opus_dred_alloc().expect("dred alloc");
        let mut rng = FastRand::new(0xDEAD_BEEF);
        let mut payload = [0u8; MAX_EXTENSION_SIZE];

        for _ in 0..ITERATIONS {
            let len = (rng.next() as usize) % (MAX_EXTENSION_SIZE + 1);
            for byte in payload.iter_mut().take(len) {
                *byte = (rng.next() & 0xFF) as u8;
            }
            let mut dred_end = 0;
            let defer_processing = (rng.next() & 0x1) != 0;
            let res = opus_dred_parse(
                &decoder,
                &mut dred,
                &payload[..len],
                48_000,
                48_000,
                Some(&mut dred_end),
                defer_processing,
            );
            if let Ok(samples) = res {
                if samples > 0 {
                    let src = dred.clone();
                    let process = opus_dred_process(&decoder, &src, &mut dred);
                    assert_eq!(process, Ok(()));
                    assert!(samples >= dred_end);
                }
            }
        }

        opus_dred_free(dred);
        opus_dred_decoder_destroy(decoder);
    }

    #[test]
    fn dred_decode_int16_runs_on_plc() {
        if !cfg!(feature = "dred") {
            return;
        }

        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder");
        let dred = OpusDred::default();
        let frame_size = 480;
        let mut pcm = vec![1i16; frame_size];

        let decoded =
            opus_decoder_dred_decode(&mut decoder, &dred, 0, &mut pcm, frame_size)
                .expect("dred decode");
        assert_eq!(decoded, frame_size);
        assert!(pcm.iter().all(|&value| value == 0));
    }

    #[test]
    fn dred_decode_int24_runs_on_plc() {
        if !cfg!(feature = "dred") {
            return;
        }

        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder");
        let dred = OpusDred::default();
        let frame_size = 480;
        let mut pcm = vec![1i32; frame_size];

        let decoded =
            opus_decoder_dred_decode24(&mut decoder, &dred, 0, &mut pcm, frame_size)
                .expect("dred decode");
        assert_eq!(decoded, frame_size);
        assert!(pcm.iter().all(|&value| value == 0));
    }

    #[test]
    fn dred_decode_float_runs_on_plc() {
        if !cfg!(feature = "dred") {
            return;
        }

        let mut decoder = opus_decoder_create(48_000, 1).expect("decoder");
        let dred = OpusDred::default();
        let frame_size = 480;
        let mut pcm = vec![1.0f32; frame_size];

        let decoded =
            opus_decoder_dred_decode_float(&mut decoder, &dred, 0, &mut pcm, frame_size)
                .expect("dred decode");
        assert_eq!(decoded, frame_size);
        assert!(pcm.iter().all(|&value| value == 0.0));
    }
}
