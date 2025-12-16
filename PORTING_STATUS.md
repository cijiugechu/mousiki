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
  selection are available. The ambisonics helpers stop after matrix sizing/selection and
  explicitly defer multistream wiring.
- Padding extension helpers from `extensions.c` are ported, including iterator/count/parse/generate
  support wired through the repacketizer.
- Multistream glue includes channel layout helpers plus ambisonics validation and
  bitrate-allocation utilities from `opus_multistream_encoder.c`. Decoder sizing/init/CTL
  dispatch and packet validation are ported; the top-level decode glue now mirrors the
  `opus_decode_native` FEC/PLC/front-end control flow and invokes the translated
  per-frame decoder (SILK PLC/FEC plus CELT output on non-fixed builds), including the
  CELT↔SILK transition smoothing and redundancy fades used during bandwidth switches; hybrid
  frames now run through the same path. Encoder front-ends remain stubbed.
- Tonality analysis mirrors `analysis.c/h` and the supporting MLP (`mlp.c`, `mlp_data.c`),
  including the RNN-based music/speech classifier, bandwidth detector, and tonality metadata
  extraction used by the encoder heuristics.
- The public soft-clip helper from `opus.c` is ported as `opus_pcm_soft_clip{,_impl}`.
- The decode-side gain and soft-clip tail from `opus_decode_native` are wired into
  `opus_decode_native` via `OpusDecoder::apply_decode_gain_and_soft_clip`. Decoder CTL dispatch
  now covers the frame pitch query (`OPUS_GET_PITCH`), exposing CELT post-filter state or the SILK
  control block as in the reference.
- `opus_decoder_get_size` and related layout sizing helpers are ported. Packet/header parsing,
  including the self-delimited variant used by multistream decode, feeds the ported
  `opus_decode_native` and its 16/24-bit and float wrappers; fixed-point CELT output is still
  omitted.

Remaining modules to port
-------------------------
- Top-level decoder: remaining CTLs beyond the existing pitch/bandwidth/gain/final-range set.
  Fixed-point CELT output remains unported.
- Top-level encoder: `opus_encoder.c` and `analysis.h` entry points (`opus_encode`,
  `_encode_float/_encode_native`, FEC/DTX/LBRR glue, encoder CTLs, per-frame state updates).
- Multistream: Encoder front-end (`opus_multistream_encoder.c`) remains unimplemented apart from
  padding/unpadding helpers. Decoder side has size/init/CTL wiring and packet validation but still
  lacks the per-stream decode/PCM routing layered over the now-ported `opus_decode_native`.
- Projection: `opus_projection_encoder.c` / `opus_projection_decoder.c` front-ends are missing;
  only mapping/matrix selection is present and still depends on multistream glue.
  ```4:10:src/projection.rs
  //! This module begins the port of `opus_projection_{encoder,decoder}.c` ...
  ```
- Neural/DNN extras: the entire `opus-c/dnn/` tree (DRED/LPCNet/OSCE/Deep PLC tooling and demos)
  is not ported; Rust currently only stubs the optional deep-PLC hooks.
- Architecture-specific SIMD back-ends and runtime CPU detection for CELT/SILK (arm/x86/mips)
  remain unported; dispatch tables default to scalar implementations.
- Demos/tests/tools: `opus_demo.c`, `opus_compare.c`, and other CLI/test harnesses are not
  reproduced in Rust.
