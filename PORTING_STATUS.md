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
  CELT↔SILK transition smoothing and redundancy fades used during bandwidth switches; hybrid
  frames now run through the same path. Multistream decode now includes per-stream
  `opus_decode_native` dispatch and PCM routing. The multistream encoder front-end is
  available for generic layouts and wraps the current Rust `opus_encode` implementation
  (still limited to SILK-only single-frame 20 ms packets). Surround/projection helper
  entry points are available for computing layouts and wiring the projection matrices.
- A minimal top-level encoder front-end is available via `src/opus_encoder.rs`, including
  `opus_encoder_get_size`, create/init/reset helpers, a small CTL surface, and a SILK-only
  `opus_encode` implementation capable of emitting single-frame 20 ms packets.
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
  `opus_decode_native` and its 16/24-bit and float wrappers. The `fixed_point` feature builds
  the same floating-point CELT/SILK decode path today; a true fixed-point backend is still
  pending.

Remaining modules to port
-------------------------
- Top-level decoder: optional Deep PLC / OSCE CTLs (e.g. `OPUS_SET_DNN_BLOB`,
  `OPUS_SET_OSCE_BWE`) remain pending alongside a true fixed-point decode backend.
- True fixed-point decode backend (align `--features fixed_point` with `opus-c`'s `FIXED_POINT` build):
  - Decide and encode the fixed-point type/scale model from `opus-c/celt/arch.h`:
    - `opus_val16/opus_val32/celt_sig/celt_norm/celt_coef/opus_res` become integer types in
      fixed builds (e.g. Q15/Q27 for CELT, plus `RES_SHIFT` selection for 16-bit vs 24-bit output).
    - Port/centralise the scaling constants and conversion macros used throughout the C tree
      (`SIG_SHIFT`, `SIG_SAT`, `NORM_SCALING`, `RES2INT16/RES2INT24`, `INT16TORES/INT24TORES`,
      `SIG2RES/RES2SIG`, `RES2FLOAT/FLOAT2RES`, etc.) so Rust call sites can stay readable.
      - Initial RES16 helpers are now available in `src/celt/fixed_arch.rs` (including
        `SIG_SHIFT`, `SIG_SAT`, `NORM_SCALING`, `SIG2RES/RES2SIG`, `RES2INT16/RES2INT24`,
        `INT16TORES/INT24TORES`, and `ADD_RES`); `ENABLE_RES24`-specific variants remain pending.
    - Refactor `src/celt/types.rs` so `OpusRes`/`CeltSig`/`CeltNorm`/`CeltCoef` are `cfg(feature="fixed_point")`
      aware, or introduce parallel fixed-point types and keep the float build unchanged.
  - Remove float-only dependencies from the fixed-point decode graph:
    - Eliminate `libm`/`f32` math from CELT decode when `fixed_point` is enabled (e.g. replace
      `celt_exp2`, `celt_cos_norm`, `celt_div`, `celt_sqrt`, `celt_rsqrt` with fixed-point equivalents).
    - Ensure no float-only helpers (including float-specific test assertions) are required to build
      `cargo test --features fixed_point`.
  - Port CELT's fixed-point DSP backend (most of the work; today Rust mirrors the float path):
    - FFT/MDCT: implement the `FIXED_POINT` branches of `celt/kiss_fft.c` and `celt/mdct.c`
      (twiddle tables, scaling/shift strategy, and saturation rules), then re-plumb `MdctLookup`
      and its callers to use the fixed-point kernels.
    - Pitch/search: port the `FIXED_POINT` branches of `celt/pitch.c` (downsampling, xcorr kernels,
      gain computations, and the fixed-point normalisation steps).
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
  The current Rust port supports SILK-only single-frame 20 ms packets; Hybrid/CELT packing,
  variable-duration/multiframe framing, and the full CTL surface are still pending.
- Multistream: Generic encoder/decoder front-ends are ported (per-stream encode/decode dispatch,
  self-delimited framing for all but the last stream, and PCM routing). Surround/projection-specific
  multistream encoder tuning (surround analysis and forced modes/bandwidth decisions) remains pending.
- Projection: `opus_projection_encoder.c` / `opus_projection_decoder.c` front-ends are ported and
  wired through the multistream glue, including demixing-matrix CTL equivalents.
- Neural/DNN extras: the entire `opus-c/dnn/` tree (DRED/LPCNet/OSCE/Deep PLC tooling and demos)
  is not ported; Rust currently only stubs the optional deep-PLC hooks.
- Architecture-specific SIMD back-ends and runtime CPU detection for CELT/SILK (arm/x86/mips)
  remain unported; dispatch tables default to scalar implementations.
- Demos/tests/tools: `opus_demo.c`, `opus_compare.c`, and other CLI/test harnesses are not
  reproduced in Rust.
