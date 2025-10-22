# SILK Porting Status

## Current Rust Coverage
- `src/silk/mod.rs` exposes the `codebook`, `decoder`, `icdf`, `interpolate`, and `sum_sqr_shift` modules, re-exporting the `FrameSignalType`, `FrameQuantizationOffsetType`, and `MAX_LPC_ORDER` items used when parsing the bitstream.【src/silk/mod.rs†L1-L18】
- `src/silk/icdf.rs` ports the inverse cumulative distribution function tables that drive entropy decoding of gains, LSF codebooks, LTP parameters, and related side information.【src/silk/icdf.rs†L1-L185】
- `src/silk/codebook.rs` mirrors the SILK stage-two LSF vector-quantiser tables used during decoding.【src/silk/codebook.rs†L1-L200】
- `src/silk/decoder.rs` implements parts of the SILK frame decoder, including routines for classifying frame types, decoding gain and LSF indices, reconstructing LPC coefficients, recovering pitch lags, and synthesising excitation via LTP; it also contains helper data structures such as `Decoder`, `DecoderBuilder`, `ShellBlockCounts`, `ExcitationQ23`, and `PitchLagInfo`.【src/silk/decoder.rs†L200-L700】
- `src/silk/decoder/nomarlize.rs` (sic) provides Rust representations of several C helper types (`NlsfQ15`, `ResQ10`, `A32Q17`, `Aq12Coefficients`, `Aq12List`) that back the LSF interpolation logic.【src/silk/decoder/nomarlize.rs†L1-L220】
- `src/silk/sum_sqr_shift.rs` ports the fixed-point helper that accumulates the energy of 16-bit sample blocks while determining the right-shift needed to avoid 32-bit overflow, mirroring `silk_sum_sqr_shift` from the C code.【src/silk/sum_sqr_shift.rs†L1-L101】
- `src/silk/lin2log.rs` mirrors the fixed-point approximation of `128 * log2(x)` used throughout the signal-processing helpers, matching `silk_lin2log` from the C sources.【src/silk/lin2log.rs†L1-L72】
- `src/silk/log2lin.rs` provides the companion fixed-point exponential that maps a Q7 log-domain value back to linear amplitude, matching `silk_log2lin` from the C implementation.【src/silk/log2lin.rs†L1-L71】
- `src/silk/interpolate.rs` ports the fixed-point helper that interpolates between LPC parameter vectors using a Q2 factor, mirroring `silk_interpolate` from the reference C sources.【src/silk/interpolate.rs†L1-L74】
- `src/silk/sigm_q15.rs` ports the lookup-table-based sigmoid approximation leveraged by SILK's predictor tuning helpers, mirroring `silk_sigm_Q15` from the reference C sources.【src/silk/sigm_q15.rs†L1-L86】
- `src/silk/sort.rs` ports the insertion-sort helpers that maintain partially ordered fixed-point vectors and their indices, mirroring `silk/sort.c`.【src/silk/sort.rs†L1-L159】
- `src/silk/tables_pulses_per_block.rs` mirrors the shell coder probability tables for rate levels, pulse counts, and sign decoding, replacing the lookup data from `silk/tables_pulses_per_block.c`.【src/silk/tables_pulses_per_block.rs】
- `src/silk/tables_pitch_lag.rs` ports the iCDF tables that parameterise SILK's pitch-lag estimation and contour decoding, matching `silk/tables_pitch_lag.c`.【src/silk/tables_pitch_lag.rs†L1-L29】
- `src/silk/tables_gain.rs` mirrors the quantization gain iCDF tables and associated delta-gain bounds defined in `silk/tables_gain.c`, supplying the lookup data used when decoding independent and delta gain indices.【src/silk/tables_gain.rs†L1-L31】【opus-c/silk/tables_gain.c†L1-L55】
- `src/silk/table_lsf_cos.rs` ports the fixed-point cosine lookup table used when converting LSFs to cosines, matching the values from `silk/table_LSF_cos.c`.【src/silk/table_lsf_cos.rs†L1-L26】【opus-c/opus-main/silk/table_LSF_cos.c†L1-L69】
- `src/silk/tables_other.rs` now provides the remaining stereo predictor, LBRR flag, uniform entropy, quantisation offset, and transition filter tables sourced from `silk/tables_other.c`, along with associated constants shared across SILK's stereo and bandwidth-transition paths.【src/silk/tables_other.rs†L1-L75】
- `src/silk/resampler_down2.rs` ports the fixed-point 2× downsampler used by SILK's resampling pipeline, mirroring the all-pass structure in `silk/resampler_down2.c` and relying on the ROM coefficients from `silk/resampler_rom.h`.【src/silk/resampler_down2.rs†L1-L82】
- `src/silk/bwexpander.rs` mirrors the chirp-based LPC bandwidth expander from `silk/bwexpander.c`, applying the fixed-point helper that shrinks LPC magnitudes during prediction setup.【src/silk/bwexpander.rs†L1-L83】【opus-c/opus-main/silk/bwexpander.c†L34-L49】
- `src/silk/bwexpander_32.rs` ports the 32-bit variant of the LPC bandwidth-expansion helper from `silk/bwexpander_32.c`, applying the same chirp logic to full-precision predictor coefficients used in resampling and analysis paths.【src/silk/bwexpander_32.rs†L1-L66】【opus-c/opus-main/silk/bwexpander_32.c†L34-L51】

The existing Rust implementation therefore covers only a subset of the full SILK decoder pipeline and omits all encoder- and platform-specific code.

## C Module Inventory

### Public API and State Management
- `API.h` declares the public encoder and decoder entry points (`silk_Encode`, `silk_Decode`) plus lifecycle helpers for sizing, initialising, and resetting codec state, alongside the externally visible control structures.【22c9aa†L37-L121】
- `enc_API.c` wires the high-level encoder API, exposing `silk_Encode` and driving channel/state transitions, VAD hooks, and configuration updates.【29ff48†L18-L66】
- `dec_API.c` implements the matching decoder super-structure (`silk_decoder`) and API functions such as `silk_LoadOSCEModels`, `silk_Get_Decoder_Size`, `silk_ResetDecoder`, `silk_InitDecoder`, and `silk_Decode`, orchestrating per-channel decode calls and packet handling.【49e9b1†L1-L120】
- `structs.h` defines the comprehensive encoder and decoder state structures (`silk_encoder_state`, `silk_decoder_state`, `silk_decoder_control`) that hold buffers, configuration, PLC/CNG state, entropy indices, and architecture-specific metadata.【6988cc†L1-L60】【9c7ced†L1-L56】
- `init_encoder.c`, `init_decoder.c`, and `decoder_set_fs.c` set up and reset codec state, manage sample-rate transitions, and enforce buffer invariants during initialisation.【1cceb7†L1-L40】【608c41†L1-L52】

None of these lifecycle or state-management layers have been ported to Rust; the Rust tree lacks equivalents for the public C API wrappers and the large codec state structures.

### Decoder Pipeline
- `decode_frame.c` sequences the SILK frame decode by invoking index, pulse, parameter, and core reconstruction stages, then updates PLC/CNG state and output buffers.【adaec1†L1-L80】
- `decode_core.c`, `decode_parameters.c`, `decode_indices.c`, `decode_pitch.c`, and `decode_pulses.c` contain the detailed algorithms that unpack side information, reconstruct gains and predictor coefficients, decode pitch contours, and run the inverse noise-shaping quantiser.【b6184a†L1-L76】【b563fc†L1-L78】
- `PLC.c` and `PLC.h` implement packet-loss concealment, while `CNG.c` performs comfort-noise generation that depends on decoder state.【f552ca†L1-L94】【a5a877†L1-L64】
- `VAD.c`, `shell_coder.c`, `code_signs.c`, and related helpers contribute to excitation decoding and voice-activity detection.【fa37d9†L1-L104】【75f2da†L1-L112】【87f287†L1-L32】

The Rust decoder implements only a slice of this logic: there are no ports yet for the top-level `silk_decode_frame` orchestration, PLC/CNG handling, entropy decoding of pulses, or the auxiliary VAD/shell modules.

### Encoder Pipeline
- `enc_API.c` delegates to encoder control helpers like `control_codec.c`, `control_SNR.c`, `control_audio_bandwidth.c`, `check_control_input.c`, and gain/pitch analysis routines spread across `gain_quant.c`, `quant_LTP_gains.c`, `NSQ.c`, and `NSQ_del_dec.c`. These files handle bandwidth switching, rate control, long-term prediction updates, and noise-shaping quantisation.【f1c0fa†L1-L64】【57d235†L1-L60】【da49cd†L1-L74】【7b30e4†L1-L56】【760d41†L1-L68】【7f3cdd†L1-L76】【cbc6a2†L1-L120】
- `encode_indices.c`, `encode_pulses.c`, and numerous files in `fixed/` and `float/` provide the forward (encoder) versions of the entropy coding, LSF quantisation, and signal analysis pipeline.【7cf3d9†L1-L108】【424aff†L1-L88】【a500cb†L1-L80】【c56351†L1-L88】

No encoder functionality exists in the Rust tree, leaving these modules entirely unported.

### Signal Processing Libraries
- The `fixed/` directory implements fixed-point DSP kernels for the encoder and decoder, including LPC/LTP analysis, pitch detection, residual energy estimation, and vector operations.【424aff†L1-L88】【a500cb†L1-L96】
- The `float/` directory mirrors these routines with floating-point implementations used in certain build configurations.【c56351†L1-L88】
- Platform-specific optimisations under `arm/`, `x86/`, `mips/`, and `xtensa/` provide SIMD kernels and specialised math headers.【c0a208†L1-L80】【4e0c2f†L1-L96】【cbc6a2†L1-L120】【880630†L1-L80】

These support libraries are prerequisites for a full port but have not yet been translated to Rust.

### Resampling and Utility Modules
- `resampler.c`, `resampler_private_*.c`, `resampler_rom.c`, and `resampler_structs.h` implement the multi-stage resamplers used on the encoder side and for decoder bandwidth transitions.【77a597†L1-L120】【723e57†L1-L100】
- Helper utilities such as `sum_sqr_shift.c`, `interpolate.c`, `log2lin.c`, and table files (`tables_*.c`) supply math helpers and lookup data; Rust now mirrors the `sum_sqr_shift`, `log2lin`, `interpolate`, `tables_pitch_lag`, `tables_pulses_per_block`, `tables_gain`, and `table_LSF_cos` helpers via `src/silk/sum_sqr_shift.rs`, `src/silk/log2lin.rs`, `src/silk/interpolate.rs`, `src/silk/tables_pitch_lag.rs`, `src/silk/tables_pulses_per_block.rs`, `src/silk/tables_gain.rs`, and `src/silk/table_lsf_cos.rs`, but the remaining lookup tables stay in C.【28a6dc†L1-L44】【a6d7bc†L1-L48】【src/silk/sum_sqr_shift.rs†L1-L101】【src/silk/log2lin.rs†L1-L71】【src/silk/interpolate.rs†L1-L74】【src/silk/tables_pitch_lag.rs†L1-L29】【src/silk/tables_pulses_per_block.rs】【src/silk/tables_gain.rs†L1-L31】【src/silk/table_lsf_cos.rs†L1-L26】
- `sort.c` is now mirrored by `src/silk/sort.rs`, providing the insertion-sort routines used across decoder helpers.【src/silk/sort.rs†L1-L159】

Rust still lacks the more elaborate fractional and multi-stage resampler implementations noted above; only the basic 2× downsampler has been ported so far via `src/silk/resampler_down2.rs`.【src/silk/resampler_down2.rs†L1-L82】

### Stereo, Bandwidth Extension, and Optional Features
- Stereo prediction, MS/LR transforms, and predictor quantisation live in `stereo_*.c`, while bandwidth extension and tuning logic appear in files such as `HP_variable_cutoff.c`, `LP_variable_cutoff.c`, and `tuning_parameters.h`. Optional OSCE support is wired through additional headers referenced by the decoder API.【6e5ae6†L1-L44】【03d532†L1-L60】
- Architecture-specific noise-shaping and vectorisation code extends many of these features for NEON, SSE4.1, AVX2, and other targets.【cbc6a2†L1-L160】【880630†L1-L80】

These stereo/bandwidth-extension paths have no representation in the Rust tree.

## Missing Functions and Types

| Kind | C definition | Purpose | Rust status |
| --- | --- | --- | --- |
| Function | `silk_Get_Encoder_Size` (`enc_API.c`) | Returns the size of the encoder state object for allocation.【22c9aa†L47-L72】 | No Rust encoder API; `src/silk` exposes no encoder module.【src/silk/mod.rs†L1-L16】 |
| Function | `silk_InitEncoder` (`enc_API.c`) | Initialises or resets encoder channels and configuration.【29ff48†L18-L60】 | Missing from Rust; no encoder lifecycle implementation exists.【src/silk/mod.rs†L1-L16】 |
| Function | `silk_Encode` (`enc_API.c`) | Top-level frame encode driver managing buffers, transitions, and range coding.【29ff48†L20-L66】 | Not ported; Rust tree lacks encoder entry points.【src/silk/mod.rs†L1-L16】 |
| Function | `silk_LoadOSCEModels` (`dec_API.c`) | Loads optional neural enhancement models into the decoder state.【49e9b1†L34-L68】 | Absent; Rust decoder has no OSCE integration or equivalent hooks.【src/silk/decoder.rs†L949-L1206】 |
| Function | `silk_Get_Decoder_Size` (`dec_API.c`) | Reports the size of the decoder super-structure.【49e9b1†L86-L112】 | Not implemented; Rust exposes a simplified `DecoderBuilder` without size queries.【src/silk/decoder.rs†L36-L81】 |
| Function | `silk_ResetDecoder` / `silk_InitDecoder` (`dec_API.c`) | Reset and initialisation routines for multi-channel decoder state.【1cceb7†L1-L40】【d0e901†L1-L28】 | Missing; Rust lacks channel management and reset APIs.【src/silk/mod.rs†L1-L16】 |
| Function | `silk_Decode` (`dec_API.c`) | High-level frame decode orchestration including PLC and stereo handling.【d0e901†L15-L112】 | Partially covered by `Decoder::decode`, but PLC/stereo paths are absent.【src/silk/decoder.rs†L949-L1206】 |
| Function | `silk_decode_frame` (`decode_frame.c`) | Core per-frame decoder tying together index, pulse, parameter, and PLC steps.【adaec1†L1-L80】 | No direct Rust analogue; current decoder skips PLC, shell, and stereo updates.【src/silk/decoder.rs†L949-L1206】 |
| Function | `silk_decode_parameters` (`decode_parameters.c`) | Reconstructs predictor coefficients, gains, and interpolation weights from bitstream indices.【b6184a†L1-L76】 | Not ported; Rust reconstructs some LSF/LTP data but omits this combined routine.【src/silk/decoder.rs†L208-L700】 |
| Function | `silk_decode_pulses` (`decode_pulses.c`) | Expands entropy-coded pulse data into excitation vectors.【b563fc†L1-L78】 | Partially mirrored by `Decoder::decode_excitation`, but there is no standalone Rust module that matches the C entry point or its shell coder helpers.【src/silk/decoder.rs†L1143-L1314】 |
| Function | `silk_PLC` (`PLC.c`) | Performs packet-loss concealment using decoder history.【f552ca†L1-L94】 | Not implemented; Rust decode path lacks PLC integration.【src/silk/decoder.rs†L949-L1206】 |
| Function | `silk_CNG` (`CNG.c`) | Generates comfort noise during silent periods or packet loss.【a5a877†L1-L64】 | Missing from Rust decoder pipeline.【src/silk/decoder.rs†L949-L1206】 |
| Type | `silk_encoder_state` (`structs.h`) | Full encoder working state including buffers, resamplers, and entropy indices.【6988cc†L1-L60】 | No Rust encoder state struct exists.【src/silk/mod.rs†L1-L16】 |
| Type | `silk_decoder_state` (`structs.h`) | Decoder state with LPC buffers, PLC/CNG members, and channel metadata.【9c7ced†L1-L36】 | Rust `Decoder` is a minimal subset without these fields.【src/silk/decoder.rs†L82-L160】 |
| Type | `silk_decoder_control` (`structs.h`) | Holds decoded predictor coefficients, gains, and LTP parameters for synthesis.【9c7ced†L36-L48】 | Not represented; Rust code uses ad-hoc structures for partial data only.【src/silk/decoder.rs†L530-L598】 |

This inventory highlights that the Rust port currently implements only portions of the SILK decoder’s entropy and LPC reconstruction logic. The remaining decoder orchestration, PLC/CNG support, encoder pipeline, and supporting DSP libraries still need to be translated from the C sources.
