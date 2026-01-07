Opus Porting Status
===================

This file tracks how much of the reference `opus-c` tree is covered by the Rust port in `src/`.
Detailed status for the CELT and SILK subcomponents lives in `src/celt/PORTING_STATUS.md`
and `src/silk/PORTING_STATUS.md`; the notes below focus on the top-level libopus glue.

Current Rust coverage
---------------------
- CELT and SILK scalar implementations are ported (see the sub-status files above); platform
  SIMD back-ends still default to scalar helpers.
- Packet parsing, repacketizer helpers, mapping matrices, and a subset of projection layout
  selection are available. Projection encoder/decoder front-ends for mapping family 3 are
  now wired on top of the multistream glue.
- Padding extension helpers from `extensions.c` are ported, including iterator/count/parse/generate
  support wired through the repacketizer.
- Multistream glue includes channel layout helpers plus ambisonics validation and
  bitrate-allocation utilities from `opus_multistream_encoder.c`. Decoder sizing/init/CTL
  dispatch and packet validation are ported; the top-level decode glue now mirrors the
  `opus_decode_native` FEC/PLC/front-end control flow and invokes the translated
  per-frame decoder (SILK PLC/FEC plus CELT output on non-fixed builds), including the
  CELT↔SILK transition smoothing and redundancy fades used during bandwidth switches.
  Multistream decode now includes per-stream
  `opus_decode_native` dispatch and PCM routing. The multistream encoder front-end is
  available for generic layouts and wraps the current Rust `opus_encode` implementation
  (SILK-only supports 10/20/40/60 ms packets plus repacketized >60 ms frames, CELT-only
  supports multiframe packing; hybrid remains minimal single-frame 10/20 ms payloads).
  Surround/projection helper
  entry points are available for computing layouts and wiring the projection matrices.
- A minimal top-level encoder front-end is available via `src/opus_encoder.rs`, including
  `opus_encoder_get_size`, create/init/reset helpers, an expanded CTL surface (bitrate,
  VBR, force channels, bandwidth caps, signal type, lsb depth, expert frame duration,
  prediction disable, phase inversion disable, forced mode), plus a basic `opus_encode`
  implementation that emits SILK-only 10/20/40/60 ms packets (and repacketized >60 ms frames),
  plus CELT-only multiframe payloads, with updated TOC/final-range handling; hybrid remains
  limited to single-frame 10/20 ms payloads. Unit tests cover
  the encoder CTL round-trips, validation cases, and TOC/frame-size outputs.
- Feature-gated Deep REDundancy (DRED) stubs are available in `src/dred.rs`, with matching
  encoder/multistream CTLs gated behind the `dred` feature.
- Tonality analysis mirrors `analysis.c/h` and the supporting MLP (`mlp.c`, `mlp_data.c`),
  including the RNN-based music/speech classifier, bandwidth detector, and tonality metadata
  extraction used by the encoder heuristics.
- The public soft-clip helper from `opus.c` is ported as `opus_pcm_soft_clip{,_impl}`.
- The decode-side gain and soft-clip tail from `opus_decode_native` are wired into
  `opus_decode_native` via `OpusDecoder::apply_decode_gain_and_soft_clip`. Decoder CTL dispatch
  now covers the full reference decoder surface (gain/complexity/bandwidth/sample-rate/pitch/
  last-packet-duration/final-range/phase-inversion/reset), exposing CELT post-filter state or the
  SILK control block as in the reference.
- `opus_decoder_get_size` and related layout sizing helpers are ported. Packet/header parsing,
  including the self-delimited variant used by multistream decode, feeds the ported
  `opus_decode_native` and its 16/24-bit and float wrappers. The `fixed_point` feature now
  wires the fixed-point MDCT/FFT kernels into the CELT decode IMDCT and uses the reference
  PLC pitch downsample/search flow (still float-backed), but the broader fixed-point backend
  is still pending.

Remaining modules to port
-------------------------
- Top-level decoder: optional Deep PLC / OSCE CTLs (e.g. `OPUS_SET_DNN_BLOB`,
  `OPUS_SET_OSCE_BWE`) remain pending alongside a true fixed-point decode backend and the
  shared SILK/CELT range-decoder path for hybrid packets.
- True fixed-point decode backend (align `--features fixed_point` with `opus-c`'s `FIXED_POINT` build):
  - Fixed-point type aliases plus the `arch.h` scaling/conversion helpers now live in
    `src/celt/types.rs` and `src/celt/fixed_arch.rs` (including RES16/RES24 variants, SIG/RES
    conversions, and PCM helpers), so downstream ports can reuse a single model.
  - Remove float-only dependencies from the fixed-point decode graph:
    - Eliminate `libm`/`f32` math from CELT decode when `fixed_point` is enabled (e.g. replace
      `celt_exp2`, `celt_cos_norm`, `celt_div`, `celt_sqrt`, `celt_rsqrt` with fixed-point equivalents).
    - Ensure no float-only helpers (including float-specific test assertions) are required to build
      `cargo test --features fixed_point`.
  - Port CELT's fixed-point DSP backend (most of the work; the fixed-point MDCT/FFT kernels are
    now in place, but much of the decode graph still mirrors the float path):
    - FFT/MDCT: fixed-point `kiss_fft` + MDCT kernels are ported (`kiss_fft_fixed.rs`,
      `mdct_fixed.rs`) and wired into the decode IMDCT path.
    - Bands/VQ/quantisation: port the `FIXED_POINT` branches in `celt/bands.c`, `celt/vq.c`,
      and `celt/quant_bands.c` (normalisation, renormalisation, PVQ search, stereo energy maths,
      and the fixed-point energy/log-domain conversions).
    - LPC helpers: port `celt/celt_lpc.c` fixed-point branches used by PLC and post-filter paths.
    - Decoder glue: update `celt_decoder` paths that currently assume float (`CELT_SIG_SCALE` and
      float PCM scaling) to match `celt_decoder.c`'s fixed-point output handling, including the
      `ENABLE_RES24`-style conversions where applicable.
  - Align top-level decode wrappers with the fixed-point reference behaviour:
    - Mirror `opus-c/src/opus_decoder.c`’s fixed-point wrapper behaviour where it decodes directly
      to `i16`/`i32` output buffers (avoiding an intermediate float `opus_res` vector when possible).
    - Keep `OPUS_SET_GAIN` behaviour but implement the gain scaling without floats in fixed builds
      (fixed-point `exp2`/lookup path).
    - Match the reference’s soft-clip behaviour (`#ifndef FIXED_POINT` in `opus_decode_native`):
      keep `OPTIONAL_CLIP` disabled for fixed builds unless an integer soft-clip is implemented.
  - Tests and validation for the fixed backend:
    - Add `--features fixed_point` coverage that validates fixed-point bit-exactness against known
      vectors (or against the `opus-c` fixed-point build) for CELT-only, SILK-only, and hybrid packets,
      including PLC/FEC cases and transition smoothing.
    - Keep the existing float tests, but add fixed-point-specific assertions around saturation/scaling
      boundaries (e.g. `SIG_SAT` behaviour and `RES2INT16/24` conversions).
- Top-level encoder: `opus_encoder.c` and `analysis.h` entry points (`opus_encode`,
  `_encode_float/_encode_native`, FEC/DTX/LBRR glue, encoder CTLs, per-frame state updates).
  The current Rust port supports SILK-only 10/20/40/60 ms packets (plus repacketized >60 ms
  frames) and CELT-only multiframe packing; hybrid payloads remain minimal single-frame 10/20 ms.
  Full CELT/HYBRID packing, variable-duration/multiframe framing for hybrid, redundancy, and
  remaining CTL coverage (e.g. voice ratio, lookahead, application changes, LFE/energy mask,
  DNN hooks) are still pending.
- Multistream: Generic encoder/decoder front-ends are ported (per-stream encode/decode dispatch,
  self-delimited framing for all but the last stream, and PCM routing). Surround/projection-specific
  multistream encoder tuning (surround analysis and forced modes/bandwidth decisions) remains pending.
- Projection: `opus_projection_encoder.c` / `opus_projection_decoder.c` front-ends are ported and
  wired through the multistream glue, including demixing-matrix CTL equivalents.
- Neural/DNN extras: the entire `opus-c/dnn/` tree (DRED/LPCNet/OSCE/Deep PLC tooling and demos)
  is not ported; Rust currently only stubs the optional deep-PLC hooks and the DRED API surface.
- Architecture-specific SIMD back-ends and runtime CPU detection for CELT/SILK (arm/x86/mips)
  remain unported; dispatch tables default to scalar implementations.
- Demos/tests/tools: `opus_demo.c`, `opus_compare.c`, and other CLI/test harnesses are not
  reproduced in Rust; `test_opus_encode.c` is ported as `tests/test_opus_encode.rs`.

Porting plan (tracked work)
---------------------------
- Step 1 (done): map `test_opus_encode` call chain and identify missing APIs/CTLs/features.
- Step 2 (done): port single-stream encoder CTLs needed by `test_opus_encode`
  (force channels, bandwidth caps, signal type, lsb depth, expert frame duration,
  prediction/phase inversion flags, forced mode) with Rust tests.
- Step 3 (done): add multistream encoder CTL parity and expose encoder-state access
  used by `test_opus_encode`; extend packet padding/unpadding test coverage.
- Step 4 (done): extend `opus_encode` with frame-size selection and basic Hybrid/CELT paths,
  updating TOC and final-range handling (SILK supports 10/20/40/60 ms plus repacketized >60 ms;
  CELT-only multiframe packing available; hybrid multiframe pending).
- Step 5 (done): port regression vectors from `opus_encode_regressions.c` into Rust tests.
- Step 6 (done): feature-gate DRED APIs/CTLs while stubbing the unported DRED paths.
