#![allow(dead_code)]

//! Decoder scaffolding ported from `celt/celt_decoder.c`.
//!
//! The reference implementation combines the primary decoder state with a
//! trailing buffer that stores the pitch predictor history, LPC coefficients,
//! and band energy memories.  This module mirrors the allocation strategy so
//! that higher level decode routines can be ported gradually while continuing
//! to rely on the Rust ownership model for safety.
//!
//! Only the allocation helpers are provided for now.  The full decoding loop,
//! packet loss concealment, and post-filter plumbing still live in the C
//! sources and will be translated in follow-up patches.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::celt::celt::{COMBFILTER_MINPERIOD, TF_SELECT_TABLE, init_caps, resampling_factor};
use crate::celt::cpu_support::{OPUS_ARCHMASK, opus_select_arch};
use crate::celt::entcode::{self, BITRES};
use crate::celt::entdec::EcDec;
use crate::celt::quant_bands::{unquant_coarse_energy, unquant_fine_energy};
use crate::celt::rate::clt_compute_allocation;
use crate::celt::types::{
    CeltGlog, CeltSig, OpusCustomDecoder, OpusCustomMode, OpusInt32, OpusUint32, OpusVal16,
};
use crate::celt::vq::SPREAD_NORMAL;
use core::cmp::{max, min};

/// Linear prediction order used by the decoder side filters.
///
/// Mirrors the `LPC_ORDER` constant from the reference implementation.  The
/// value is surfaced here so future ports that rely on the LPC history length
/// can share the same constant.
const LPC_ORDER: usize = 24;

/// Size of the rolling decode buffer maintained per channel.
///
/// Matches the `DECODE_BUFFER_SIZE` constant from the C implementation.  The
/// reference decoder keeps a two kilobyte circular history in front of the
/// overlap region so packet loss concealment and the post-filter can operate on
/// previously synthesised samples.  Mirroring the same storage requirements in
/// Rust keeps the allocation layout compatible with the ported routines that
/// will eventually consume these buffers.
const DECODE_BUFFER_SIZE: usize = 2048;

/// Maximum pitch period considered by the PLC pitch search.
const MAX_PERIOD: i32 = 1024;

/// Upper bound on the pitch lag probed by the PLC search.
const PLC_PITCH_LAG_MAX: i32 = 720;

/// Lower bound on the pitch lag probed by the PLC search.
const PLC_PITCH_LAG_MIN: i32 = 100;

/// Maximum number of channels supported by the initial CELT decoder port.
///
/// The reference implementation restricts the custom decoder to mono or stereo
/// streams.  The helper routines below mirror the same validation so the
/// call-sites can rely on early argument checking just like the C helpers.
const MAX_CHANNELS: usize = 2;

/// Cumulative distribution used to decode the global allocation trim.
const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// Spread decision probabilities used by the transient classifier.
const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];

/// Probability model for the three post-filter tapset candidates.
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// Scalar used to decode the post-filter gain from the coarse index.
const POSTFILTER_GAIN_SCALE: OpusVal16 = 0.09375;

/// Errors that can be reported while preparing to decode a CELT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltDecodeError {
    /// Input arguments were inconsistent with the current decoder state.
    BadArgument,
    /// The supplied packet was too short to decode a frame and should trigger PLC.
    PacketLoss,
    /// The packet signalled an invalid configuration or ran out of bits.
    InvalidPacket,
}

/// Range decoder storage mirroring the temporary buffer used by the C decoder.
#[derive(Debug)]
pub(crate) struct RangeDecoderState {
    decoder: EcDec<'static>,
    storage: Box<[u8]>,
}

impl RangeDecoderState {
    /// Creates a new range decoder backed by an owned copy of the packet payload.
    pub(crate) fn new(data: &[u8]) -> Self {
        let mut storage = data.to_vec().into_boxed_slice();
        // SAFETY: `EcDec` only requires that the backing slice outlives the decoder
        // instance. The boxed slice is moved into the struct alongside the decoder,
        // ensuring the borrow remains valid for the lifetime of `Self`.
        let decoder = {
            // Borrow explicitly to keep the compiler aware of the lifetimes.
            let slice: &mut [u8] = &mut storage;
            // Extend the borrow to 'static to satisfy the generic parameter. This is
            // safe because `storage` is moved into the struct and dropped after
            // `decoder`, preserving the required lifetime ordering.
            let slice: &'static mut [u8] = unsafe { core::mem::transmute(slice) };
            EcDec::new(slice)
        };
        Self { decoder, storage }
    }

    /// Borrows the underlying range decoder with the appropriate lifetime.
    pub(crate) fn decoder(&mut self) -> &mut EcDec<'static> {
        &mut self.decoder
    }
}

/// Debug-time validation mirroring `validate_celt_decoder()` from the C sources.
///
/// The reference implementation relies on this helper to assert that the decoder
/// state remains internally consistent after initialisation and before decoding
/// a frame. The Rust translation mirrors the same invariants so regressions are
/// caught early while the remaining decode path is ported.
pub(crate) fn validate_celt_decoder(decoder: &OpusCustomDecoder<'_>) {
    debug_assert_eq!(
        decoder.overlap, decoder.mode.overlap,
        "decoder overlap must match the mode configuration",
    );

    let mode_band_limit = decoder.mode.num_ebands as i32;
    let standard_limit = 21;
    let custom_limit = 25;
    let end_limit = if mode_band_limit <= standard_limit {
        mode_band_limit
    } else {
        mode_band_limit.min(custom_limit)
    };
    debug_assert!(
        decoder.end_band <= end_limit,
        "end band {} exceeds supported limit {}",
        decoder.end_band,
        end_limit
    );

    debug_assert!(
        decoder.channels == 1 || decoder.channels == 2,
        "decoder must be mono or stereo",
    );
    debug_assert!(
        decoder.stream_channels == 1 || decoder.stream_channels == 2,
        "stream must be mono or stereo",
    );
    debug_assert!(
        decoder.downsample > 0,
        "downsample factor must be strictly positive",
    );
    debug_assert!(
        decoder.start_band == 0 || decoder.start_band == 17,
        "decoder start band must be 0 or 17",
    );
    debug_assert!(
        decoder.start_band < decoder.end_band,
        "start band must precede end band",
    );

    debug_assert!(
        decoder.arch >= 0 && decoder.arch <= OPUS_ARCHMASK,
        "architecture selection out of range",
    );

    debug_assert!(
        decoder.last_pitch_index <= PLC_PITCH_LAG_MAX,
        "last pitch index exceeds maximum lag",
    );
    debug_assert!(
        decoder.last_pitch_index >= PLC_PITCH_LAG_MIN || decoder.last_pitch_index == 0,
        "last pitch index below minimum lag",
    );

    debug_assert!(
        decoder.postfilter_period < MAX_PERIOD,
        "postfilter period must remain below MAX_PERIOD",
    );
    debug_assert!(
        decoder.postfilter_period >= COMBFILTER_MINPERIOD as i32 || decoder.postfilter_period == 0,
        "postfilter period must be zero or above the comb-filter minimum",
    );
    debug_assert!(
        decoder.postfilter_period_old < MAX_PERIOD,
        "previous postfilter period must remain below MAX_PERIOD",
    );
    debug_assert!(
        decoder.postfilter_period_old >= COMBFILTER_MINPERIOD as i32
            || decoder.postfilter_period_old == 0,
        "previous postfilter period must be zero or above the comb-filter minimum",
    );

    debug_assert!(
        (0..=2).contains(&decoder.postfilter_tapset),
        "postfilter tapset must be in the inclusive range [0, 2]",
    );
    debug_assert!(
        (0..=2).contains(&decoder.postfilter_tapset_old),
        "previous postfilter tapset must be in the inclusive range [0, 2]",
    );
}

/// Metadata describing the parsed frame header and bit allocation.
#[derive(Debug)]
pub(crate) struct FramePreparation {
    pub range_decoder: Option<RangeDecoderState>,
    pub tf_res: Vec<OpusInt32>,
    pub cap: Vec<OpusInt32>,
    pub offsets: Vec<OpusInt32>,
    pub fine_quant: Vec<OpusInt32>,
    pub pulses: Vec<OpusInt32>,
    pub fine_priority: Vec<OpusInt32>,
    pub spread_decision: OpusInt32,
    pub is_transient: bool,
    pub short_blocks: OpusInt32,
    pub intra_ener: bool,
    pub silence: bool,
    pub alloc_trim: OpusInt32,
    pub anti_collapse_rsv: OpusInt32,
    pub intensity: OpusInt32,
    pub dual_stereo: OpusInt32,
    pub balance: OpusInt32,
    pub coded_bands: OpusInt32,
    pub postfilter_pitch: OpusInt32,
    pub postfilter_gain: OpusVal16,
    pub postfilter_tapset: OpusInt32,
    pub total_bits: OpusInt32,
    pub tell: OpusInt32,
    pub bits: OpusInt32,
    pub start: usize,
    pub end: usize,
    pub eff_end: usize,
    pub lm: usize,
    pub m: usize,
    pub n: usize,
    pub c: usize,
    pub cc: usize,
    pub packet_loss: bool,
}

impl FramePreparation {
    #[allow(clippy::too_many_arguments)]
    fn new_packet_loss(
        start: usize,
        end: usize,
        eff_end: usize,
        nb_ebands: usize,
        lm: usize,
        m: usize,
        n: usize,
        c: usize,
        cc: usize,
    ) -> Self {
        let tf_res = vec![0; nb_ebands];
        let cap = vec![0; nb_ebands];
        let zeros = vec![0; nb_ebands];
        Self {
            range_decoder: None,
            tf_res,
            cap,
            offsets: zeros.clone(),
            fine_quant: zeros.clone(),
            pulses: zeros.clone(),
            fine_priority: zeros,
            spread_decision: 0,
            is_transient: false,
            short_blocks: 0,
            intra_ener: false,
            silence: false,
            alloc_trim: 0,
            anti_collapse_rsv: 0,
            intensity: 0,
            dual_stereo: 0,
            balance: 0,
            coded_bands: 0,
            postfilter_pitch: 0,
            postfilter_gain: 0.0,
            postfilter_tapset: 0,
            total_bits: 0,
            tell: 0,
            bits: 0,
            start,
            end,
            eff_end,
            lm,
            m,
            n,
            c,
            cc,
            packet_loss: true,
        }
    }
}

fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [OpusInt32],
    lm: usize,
    dec: &mut EcDec<'_>,
) {
    let mut budget = dec.ctx().storage * 8;
    let mut tell = entcode::ec_tell(dec.ctx()) as u32;
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    if tf_select_rsv {
        budget -= 1;
    }

    let mut curr = 0;
    let mut tf_changed = 0;
    for slot in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            let bit = dec.dec_bit_logp(logp);
            curr ^= bit;
            tell = entcode::ec_tell(dec.ctx()) as u32;
            tf_changed |= curr;
        }
        *slot = curr;
        logp = if is_transient { 4 } else { 5 };
    }

    let mut tf_select = 0;
    if tf_select_rsv {
        let base = 4 * usize::from(is_transient);
        if TF_SELECT_TABLE[lm][base + tf_changed as usize]
            != TF_SELECT_TABLE[lm][base + 2 + tf_changed as usize]
        {
            tf_select = dec.dec_bit_logp(1) as OpusInt32;
        }
    }

    let base = 4 * usize::from(is_transient);
    for slot in tf_res.iter_mut().take(end).skip(start) {
        let idx = base + 2 * tf_select as usize + *slot as usize;
        *slot = OpusInt32::from(TF_SELECT_TABLE[lm][idx]);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_frame(
    decoder: &mut OpusCustomDecoder<'_>,
    packet: &[u8],
    frame_size: usize,
) -> Result<FramePreparation, CeltDecodeError> {
    let cc = decoder.channels;
    let c = decoder.stream_channels;
    if cc == 0 || c == 0 || cc > MAX_CHANNELS {
        return Err(CeltDecodeError::BadArgument);
    }

    if frame_size == 0 {
        return Err(CeltDecodeError::BadArgument);
    }

    let mode = decoder.mode;
    let nb_ebands = mode.num_ebands;
    let start = decoder.start_band as usize;
    let end = decoder.end_band as usize;

    let downsample = decoder.downsample as usize;
    let scaled_frame = frame_size
        .checked_mul(downsample)
        .ok_or(CeltDecodeError::BadArgument)?;
    if scaled_frame > (mode.short_mdct_size << mode.max_lm) {
        return Err(CeltDecodeError::BadArgument);
    }

    let lm = (0..=mode.max_lm)
        .find(|&cand| mode.short_mdct_size << cand == scaled_frame)
        .ok_or(CeltDecodeError::BadArgument)?;
    let m = 1 << lm;
    let n = m * mode.short_mdct_size;

    if packet.len() > 1275 {
        return Err(CeltDecodeError::BadArgument);
    }

    let eff_end = min(end, mode.effective_ebands);

    if packet.len() <= 1 {
        return Ok(FramePreparation::new_packet_loss(
            start, end, eff_end, nb_ebands, lm, m, n, c, cc,
        ));
    }

    if decoder.loss_duration == 0 {
        decoder.skip_plc = false;
    }

    let mut range_decoder = RangeDecoderState::new(packet);
    let dec = range_decoder.decoder();

    if c == 1 {
        for band in 0..nb_ebands {
            let idx = band;
            let paired = nb_ebands + band;
            decoder.old_ebands[idx] = decoder.old_ebands[idx].max(decoder.old_ebands[paired]);
        }
    }

    let len_bits = (packet.len() * 8) as OpusInt32;
    let mut tell = entcode::ec_tell(dec.ctx());
    let mut silence = false;
    if tell >= len_bits {
        silence = true;
    } else if tell == 1 {
        silence = dec.dec_bit_logp(15) != 0;
    }
    if silence {
        let consumed = entcode::ec_tell(dec.ctx());
        dec.ctx_mut().nbits_total += len_bits - consumed;
        tell = len_bits;
    } else {
        tell = entcode::ec_tell(dec.ctx());
    }

    let mut postfilter_gain = 0.0;
    let mut postfilter_pitch = 0;
    let mut postfilter_tapset = 0;
    if start == 0 && tell + 16 <= len_bits {
        if dec.dec_bit_logp(1) != 0 {
            let octave = dec.dec_uint(6) as OpusInt32;
            let low_bits = dec.dec_bits((4 + octave) as u32) as OpusInt32;
            postfilter_pitch = ((16 << octave) + low_bits) - 1;
            let qg = dec.dec_bits(3) as OpusInt32;
            if entcode::ec_tell(dec.ctx()) + 2 <= len_bits {
                postfilter_tapset = dec.dec_icdf(&TAPSET_ICDF, 2);
            }
            postfilter_gain = POSTFILTER_GAIN_SCALE * ((qg + 1) as OpusVal16);
        }
        tell = entcode::ec_tell(dec.ctx());
    }

    let mut is_transient = false;
    if lm > 0 && tell + 3 <= len_bits {
        is_transient = dec.dec_bit_logp(3) != 0;
        tell = entcode::ec_tell(dec.ctx());
    }
    let short_blocks = if is_transient { m as OpusInt32 } else { 0 };

    let mut intra_ener = false;
    if tell + 3 <= len_bits {
        intra_ener = dec.dec_bit_logp(3) != 0;
    }

    if !intra_ener && decoder.loss_duration != 0 {
        let missing = min(10, decoder.loss_duration >> (lm as u32));
        let safety = match lm {
            0 => 1.5,
            1 => 0.5,
            _ => 0.0,
        };

        for ch in 0..2 {
            for band in start..end {
                let idx = ch * nb_ebands + band;
                let mut e0 = decoder.old_ebands[idx];
                let e1 = decoder.old_log_e[idx];
                let e2 = decoder.old_log_e2[idx];
                if e0 < e1.max(e2) {
                    let mut slope = (e1 - e0).max(0.5 * (e2 - e0));
                    slope = slope.min(2.0);
                    let reduction = (((missing + 1) as f32) * slope).max(0.0);
                    e0 -= reduction;
                    decoder.old_ebands[idx] = e0.max(-20.0);
                } else {
                    decoder.old_ebands[idx] = decoder.old_ebands[idx].min(e1.min(e2));
                }
                decoder.old_ebands[idx] -= safety;
            }
        }
    }

    unquant_coarse_energy(mode, start, end, decoder.old_ebands, intra_ener, dec, c, lm);

    let mut tf_res = vec![0; nb_ebands];
    tf_decode(start, end, is_transient, &mut tf_res, lm, dec);

    tell = entcode::ec_tell(dec.ctx());
    let mut spread_decision = SPREAD_NORMAL;
    if tell + 4 <= len_bits {
        spread_decision = dec.dec_icdf(&SPREAD_ICDF, 5);
    }

    let mut cap = vec![0; nb_ebands];
    init_caps(mode, &mut cap, lm, c);

    let mut offsets = vec![0; nb_ebands];
    let mut dynalloc_logp = 6;
    let mut total_bits = len_bits << BITRES;
    let mut tell_frac = entcode::ec_tell_frac(dec.ctx()) as OpusInt32;

    for band in start..end {
        let band_width = i32::from(mode.e_bands[band + 1] - mode.e_bands[band]);
        let width = (c as OpusInt32 * band_width) << lm;
        let six_bits = (6 << BITRES) as OpusInt32;
        let quanta = min(width << BITRES, max(six_bits, width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let mut boost = 0;
        while tell_frac + ((dynalloc_loop_logp as OpusInt32) << BITRES) < total_bits
            && boost < cap[band]
        {
            let flag = dec.dec_bit_logp(dynalloc_loop_logp as u32);
            tell_frac = entcode::ec_tell_frac(dec.ctx()) as OpusInt32;
            if flag == 0 {
                break;
            }
            boost += quanta;
            total_bits -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[band] = boost;
        if boost > 0 {
            dynalloc_logp = max(2, dynalloc_logp - 1);
        }
    }

    let mut fine_quant = vec![0; nb_ebands];
    let alloc_trim = if tell_frac + ((6 << BITRES) as OpusInt32) <= total_bits {
        dec.dec_icdf(&TRIM_ICDF, 7)
    } else {
        5
    };

    let mut bits = ((len_bits << BITRES) - entcode::ec_tell_frac(dec.ctx()) as OpusInt32) - 1;
    let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm as OpusInt32 + 2) << BITRES)
    {
        (1 << BITRES) as OpusInt32
    } else {
        0
    };
    bits -= anti_collapse_rsv;

    let mut pulses = vec![0; nb_ebands];
    let mut fine_priority = vec![0; nb_ebands];
    let mut intensity = 0;
    let mut dual_stereo = 0;
    let mut balance = 0;

    let coded_bands = clt_compute_allocation(
        mode,
        start,
        end,
        &offsets,
        &cap,
        alloc_trim,
        &mut intensity,
        &mut dual_stereo,
        bits,
        &mut balance,
        &mut pulses,
        &mut fine_quant,
        &mut fine_priority,
        c as OpusInt32,
        lm as OpusInt32,
        None,
        Some(dec),
        0,
        0,
    );

    unquant_fine_energy(mode, start, end, decoder.old_ebands, &fine_quant, dec, c);

    let tell = entcode::ec_tell(dec.ctx());

    Ok(FramePreparation {
        range_decoder: Some(range_decoder),
        tf_res,
        cap,
        offsets,
        fine_quant,
        pulses,
        fine_priority,
        spread_decision,
        is_transient,
        short_blocks,
        intra_ener,
        silence,
        alloc_trim,
        anti_collapse_rsv,
        intensity,
        dual_stereo,
        balance,
        coded_bands,
        postfilter_pitch,
        postfilter_gain,
        postfilter_tapset,
        total_bits,
        tell,
        bits,
        start,
        end,
        eff_end,
        lm,
        m,
        n,
        c,
        cc,
        packet_loss: false,
    })
}

/// Errors that can be reported when initialising a CELT decoder instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeltDecoderInitError {
    /// Channel count was zero or larger than the supported maximum.
    InvalidChannelCount,
    /// Requested stream channel layout is not compatible with the physical
    /// channels configured for the decoder.
    InvalidStreamChannels,
    /// The provided mode uses a sampling rate that cannot be resampled from the
    /// 48 kHz CELT reference clock.
    UnsupportedSampleRate,
}

/// Helper owning the trailing buffers that back [`OpusCustomDecoder`].
///
/// The C implementation allocates the decoder struct followed by a number of
/// variable-length arrays.  Keeping the storage separate in Rust avoids unsafe
/// pointer arithmetic and simplifies sharing the buffers across temporary
/// decoder views used during reset or PLC.
#[derive(Debug, Default)]
pub(crate) struct CeltDecoderAlloc {
    decode_mem: Vec<CeltSig>,
    lpc: Vec<OpusVal16>,
    old_ebands: Vec<CeltGlog>,
    old_log_e: Vec<CeltGlog>,
    old_log_e2: Vec<CeltGlog>,
    background_log_e: Vec<CeltGlog>,
}

impl CeltDecoderAlloc {
    /// Creates a new allocation suitable for the provided mode and channel
    /// configuration.
    ///
    /// The decoder requires per-channel history buffers for the overlap region
    /// as well as twice the number of energy bands tracked by the mode.  The
    /// allocations follow the layout of the C implementation while leveraging
    /// Rust's `Vec` to manage the backing storage.
    pub(crate) fn new(mode: &OpusCustomMode<'_>, channels: usize) -> Self {
        assert!(channels > 0, "decoder must contain at least one channel");

        let overlap = mode.overlap;
        let decode_mem = channels * (DECODE_BUFFER_SIZE + overlap);
        let lpc = LPC_ORDER * channels;
        let band_count = 2 * mode.num_ebands;

        Self {
            decode_mem: vec![0.0; decode_mem],
            lpc: vec![0.0; lpc],
            old_ebands: vec![0.0; band_count],
            old_log_e: vec![0.0; band_count],
            old_log_e2: vec![0.0; band_count],
            background_log_e: vec![0.0; band_count],
        }
    }

    /// Returns the total size in bytes consumed by the allocation.
    ///
    /// Mirrors the behaviour of `celt_decoder_get_size()` in spirit by exposing
    /// how much storage is required for the decoder and its trailing buffers.
    /// The actual C helper only depends on the channel count; we include the
    /// mode so the calculation reflects the precise band layout in use.  A
    /// follow-up port of the fixed allocation used by the reference
    /// implementation will replace this helper with a fully bit-exact
    /// translation.
    pub(crate) fn size_in_bytes(&self) -> usize {
        self.decode_mem.len() * core::mem::size_of::<CeltSig>()
            + self.lpc.len() * core::mem::size_of::<OpusVal16>()
            + (self.old_ebands.len()
                + self.old_log_e.len()
                + self.old_log_e2.len()
                + self.background_log_e.len())
                * core::mem::size_of::<CeltGlog>()
    }

    /// Borrows the allocation as an [`OpusCustomDecoder`] tied to the provided
    /// mode.
    ///
    /// Each call returns a fresh decoder view referencing the same backing
    /// buffers.  This mirrors the C layout where the state and trailing memory
    /// occupy a single blob, enabling the caller to reset or reuse the decoder
    /// without reallocating.
    pub(crate) fn as_decoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
    ) -> OpusCustomDecoder<'a> {
        OpusCustomDecoder::new(
            mode,
            channels,
            stream_channels,
            self.decode_mem.as_mut_slice(),
            self.lpc.as_mut_slice(),
            self.old_ebands.as_mut_slice(),
            self.old_log_e.as_mut_slice(),
            self.old_log_e2.as_mut_slice(),
            self.background_log_e.as_mut_slice(),
        )
    }

    /// Resets the allocation contents to zero.
    pub(crate) fn reset(&mut self) {
        for sample in &mut self.decode_mem {
            *sample = 0.0;
        }
        for coeff in &mut self.lpc {
            *coeff = 0.0;
        }
        for history in &mut self.old_ebands {
            *history = 0.0;
        }
        for history in &mut self.old_log_e {
            *history = 0.0;
        }
        for history in &mut self.old_log_e2 {
            *history = 0.0;
        }
        for history in &mut self.background_log_e {
            *history = 0.0;
        }
    }

    /// Returns a freshly initialised decoder state.
    ///
    /// The helper mirrors `opus_custom_decoder_init()` by validating the
    /// channel configuration, clearing the trailing buffers, and populating the
    /// fields that depend on the current mode.  Callers receive a fully formed
    /// [`OpusCustomDecoder`] that borrows the allocation's backing storage.
    pub(crate) fn init_decoder<'a>(
        &'a mut self,
        mode: &'a OpusCustomMode<'a>,
        channels: usize,
        stream_channels: usize,
        rng_seed: OpusUint32,
    ) -> Result<OpusCustomDecoder<'a>, CeltDecoderInitError> {
        validate_channel_layout(channels, stream_channels)?;

        let downsample = resampling_factor(mode.sample_rate);
        if downsample == 0 {
            return Err(CeltDecoderInitError::UnsupportedSampleRate);
        }

        self.reset();
        let mut decoder = self.as_decoder(mode, channels, stream_channels);
        decoder.downsample = downsample as i32;
        decoder.end_band = mode.effective_ebands as i32;
        decoder.arch = opus_select_arch();
        decoder.rng = rng_seed;
        decoder.error = 0;
        decoder.loss_duration = 0;
        decoder.skip_plc = false;
        decoder.postfilter_period = 0;
        decoder.postfilter_period_old = 0;
        decoder.postfilter_gain = 0.0;
        decoder.postfilter_gain_old = 0.0;
        decoder.postfilter_tapset = 0;
        decoder.postfilter_tapset_old = 0;
        decoder.prefilter_and_fold = false;
        decoder.preemph_mem_decoder = [0.0; 2];
        decoder.last_pitch_index = 0;

        Ok(decoder)
    }
}

fn validate_channel_layout(
    channels: usize,
    stream_channels: usize,
) -> Result<(), CeltDecoderInitError> {
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(CeltDecoderInitError::InvalidChannelCount);
    }
    if stream_channels == 0 || stream_channels > channels {
        return Err(CeltDecoderInitError::InvalidStreamChannels);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CeltDecodeError, CeltDecoderAlloc, CeltDecoderInitError, LPC_ORDER, MAX_CHANNELS,
        RangeDecoderState, prepare_frame, tf_decode, validate_celt_decoder,
        validate_channel_layout,
    };
    use crate::celt::types::{MdctLookup, OpusCustomMode, PulseCacheData};
    use alloc::vec;

    #[test]
    fn tf_decode_returns_default_table_entries() {
        let mut range = RangeDecoderState::new(&[0u8]);
        let mut tf_res = vec![0; 4];
        tf_decode(0, tf_res.len(), false, &mut tf_res, 0, range.decoder());
        assert!(tf_res.iter().all(|&v| v == 0));
    }

    #[test]
    fn allocates_expected_band_buffers() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 2);
        assert_eq!(
            alloc.decode_mem.len(),
            2 * (super::DECODE_BUFFER_SIZE + mode.overlap)
        );
        assert_eq!(alloc.lpc.len(), LPC_ORDER * 2);
        assert_eq!(alloc.old_ebands.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.old_log_e2.len(), 2 * mode.num_ebands);
        assert_eq!(alloc.background_log_e.len(), 2 * mode.num_ebands);

        // Ensure the reset helper clears all buffers.
        alloc.decode_mem.fill(1.0);
        alloc.lpc.fill(1.0);
        alloc.old_ebands.fill(1.0);
        alloc.old_log_e.fill(1.0);
        alloc.old_log_e2.fill(1.0);
        alloc.background_log_e.fill(1.0);
        alloc.reset();

        assert!(alloc.decode_mem.iter().all(|&v| v == 0.0));
        assert!(alloc.lpc.iter().all(|&v| v == 0.0));
        assert!(alloc.old_ebands.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e.iter().all(|&v| v == 0.0));
        assert!(alloc.old_log_e2.iter().all(|&v| v == 0.0));
        assert!(alloc.background_log_e.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn validate_celt_decoder_accepts_default_configuration() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let decoder = alloc.as_decoder(&mode, 1, 1);

        validate_celt_decoder(&decoder);
    }

    #[test]
    #[should_panic]
    fn validate_celt_decoder_rejects_invalid_channel_count() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc.as_decoder(&mode, 1, 1);
        decoder.channels = 3;

        validate_celt_decoder(&decoder);
    }

    #[test]
    fn validate_channel_layout_rejects_invalid_configurations() {
        assert_eq!(
            validate_channel_layout(0, 0),
            Err(CeltDecoderInitError::InvalidChannelCount)
        );
        assert_eq!(
            validate_channel_layout(MAX_CHANNELS + 1, 1),
            Err(CeltDecoderInitError::InvalidChannelCount)
        );
        assert_eq!(
            validate_channel_layout(1, 0),
            Err(CeltDecoderInitError::InvalidStreamChannels)
        );
        assert_eq!(
            validate_channel_layout(1, 2),
            Err(CeltDecoderInitError::InvalidStreamChannels)
        );
    }

    #[test]
    fn init_decoder_populates_expected_defaults() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let decoder = alloc
            .init_decoder(&mode, 1, 1, 1234)
            .expect("initialisation should succeed");

        assert_eq!(decoder.overlap, mode.overlap);
        assert_eq!(decoder.downsample, 1);
        assert_eq!(decoder.end_band, mode.effective_ebands as i32);
        assert_eq!(decoder.arch, 0);
        assert_eq!(decoder.rng, 1234);
        assert_eq!(decoder.loss_duration, 0);
        assert_eq!(decoder.postfilter_period, 0);
        assert_eq!(decoder.postfilter_gain, 0.0);
        assert_eq!(decoder.postfilter_tapset, 0);
        assert!(!decoder.skip_plc);
    }

    #[test]
    fn prepare_frame_handles_packet_loss() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mut mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );
        mode.short_mdct_size = 2;
        mode.max_lm = 2;

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc
            .init_decoder(&mode, 1, 1, 0)
            .expect("decoder initialisation should succeed");

        let frame = prepare_frame(&mut decoder, &[], mode.short_mdct_size)
            .expect("packet loss preparation should succeed");
        assert!(frame.packet_loss);
    }

    #[test]
    fn prepare_frame_rejects_mismatched_frame_size() {
        let e_bands = [0, 2, 5];
        let alloc_vectors = [0u8; 4];
        let log_n = [0i16; 2];
        let window = [0.0f32; 4];
        let mdct = MdctLookup::new(8, 0);
        let cache = PulseCacheData::new(vec![0; 6], vec![0; 6], vec![0; 6]);
        let mut mode = OpusCustomMode::new(
            48_000,
            4,
            &e_bands,
            &alloc_vectors,
            &log_n,
            &window,
            mdct,
            cache,
        );
        mode.short_mdct_size = 2;
        mode.max_lm = 2;

        let mut alloc = CeltDecoderAlloc::new(&mode, 1);
        let mut decoder = alloc
            .init_decoder(&mode, 1, 1, 0)
            .expect("decoder initialisation should succeed");

        let err = prepare_frame(&mut decoder, &[0u8; 2], mode.short_mdct_size + 1)
            .expect_err("invalid frame size must be rejected");
        assert_eq!(err, CeltDecodeError::BadArgument);
    }
}
