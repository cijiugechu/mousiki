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
- Tonality analysis mirrors `analysis.c/h` and the supporting MLP (`mlp.c`, `mlp_data.c`),
  including the RNN-based music/speech classifier, bandwidth detector, and tonality metadata
  extraction used by the encoder heuristics.
- The public soft-clip helper from `opus.c` is ported as `opus_pcm_soft_clip{,_impl}`.
- `opus_decoder_get_size` and related layout sizing helpers are ported; full decoding is not.

Remaining modules to port
-------------------------
- Top-level decoder: `opus_decoder.c` (`opus_decode_native`, `opus_decode{,_float,_24}`,
  self-delimited parsing, decode-gain/soft-clip integration, FEC/PLC glue, CTL handling).
- Top-level encoder: `opus_encoder.c` and `analysis.h` entry points (`opus_encode`,
  `_encode_float/_encode_native`, FEC/DTX/LBRR glue, encoder CTLs, per-frame state updates).
- Extensions/CTL shims: `extensions.c` (API wrappers and extra CTLs referenced by applications).
- Multistream: Channel layout helpers from `opus_multistream.c` are ported; the encoder/decoder
  front-ends (`opus_multistream_encoder.c`, `opus_multistream_decoder.c`) remain unimplemented
  aside from packet padding/unpadding helpers.
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
