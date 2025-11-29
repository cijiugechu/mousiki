# Porting Status

This file tracks how the reference C integration and end-to-end tests map onto the current Rust test suite.

## C integration / E2E tests and Rust coverage
- `opus-c/tests/test_opus_padding.c` — padding overflow regression; **Ported** as `tests/opus_padding.rs` against the Rust decoder API.
- `opus-c/tests/opus_decode_fuzzer.c` — libFuzzer-style streaming decode harness; **Ported (seed-only)** as `tests/fuzz_decoder.rs`, which replays the bundled Ogg seeds but does not yet wire libFuzzer.
- `opus-c/silk/tests/test_unit_LPC_inv_pred_gain.c` — randomized LPC stability check; **Ported** as `tests/lpc_inv_pred_gain.rs` with a trimmed iteration count.
- `opus-c/tests/test_opus_projection.c` — mapping matrix helper coverage; **Partially ported** as the `simple_matrix_operations_follow_reference` unit test in `src/mapping_matrix.rs`, which exercises the short and float helper paths. Ambisonics projection layout validation and demixing matrix export live in `src/projection.rs`.

## Still unported from the C tree
- `opus-c/tests/test_opus_api.c`, `test_opus_decode.c`, `test_opus_encode.c`, `test_opus_custom.c`, `test_opus_extensions.c`, `test_opus_projection.c`, `test_opus_dred.c`, and `opus_encode_regressions.c` — API compliance, encode/decode regressions, custom modes, SIMD extensions, ambisonics/projection, and DRED regression coverage remain C-only.
- `opus-c/tests/run_vectors.sh`, `random_config.sh`, `dred_vectors.sh`, and `opus_build_test.sh` — shell harnesses that orchestrate vector/regression suites and randomized config sweeps are not mirrored.
- `opus-c/celt/tests/test_unit_{laplace,mathops,entropy,types,mdct,cwrs32,rotation,dft}.c` — CELT unit tests are not yet represented in Rust.
- `opus-c/dnn/adaconvtest.c`, `dnn/test_vec.c`, and `dnn/training_tf2/test_{lpcnet,plc}.py` — neural/PLC regression checks are not ported.

## Dependency gaps for the unported C tests
- `test_opus_api.c` exercises the public encoder/decoder, CTLs, repacketizer, packet pad/unpad helpers, and the multistream front-ends. The Rust tree lacks the `opus_decode{,_float,_native}` path, the `OpusEncoder`/`opus_encode` wrappers, the repacketizer, packet padding/unpadding, and the `opus_multistream_*` encode/decode shims.
- `test_opus_decode.c` expects a fully wired `OpusDecoder` that runs SILK↔CELT hybrid decoding, FEC/PLC handling, self-delimited parsing, and decode gain/soft-clip updates. Those top-level decode entry points and the SILK–CELT glue are still C-only.
- `test_opus_encode.c` and `opus_encode_regressions.c` rely on the canonical `OpusEncoder` driver (`opus_encoder.c`) that orchestrates SILK/CELT analysis, VBR/CBR switching, in-band FEC/DTX/LBRR, and repacketizer-backed packet assembly; the Rust codebase does not yet expose a top-level encoder or the hybrid packetiser.
- `test_opus_custom.c` depends on the public `opus_custom_*` encoder/decoder API and custom-mode construction. While CELT custom components exist internally, there is no public `OpusCustomEncoder/Decoder` front-end or end-to-end encode/decode hook-up in Rust.
- `test_opus_extensions.c` exercises the packet extension helpers (`opus_packet_extensions_*`) and repacketizer extension passthrough; Rust currently lacks the extension encode/decode logic and the repacketizer implementation they rely on.
- `test_opus_projection.c` still needs the ambisonics projection encoder/decoder and the projection decode/encode harnesses. The `mapping_matrix_*` helpers now live in `src/mapping_matrix.rs`, and `src/projection.rs` covers the ambisonics channel validation/matrix selection and demixing export helpers, but no projection multistream front-ends or Ogg header plumbing have been ported.
- `test_opus_dred.c` covers the DRED extension negotiation and decode path. The Rust decoder still omits DRED parsing, buffering, and redundancy synthesis.
- `run_vectors.sh`, `random_config.sh`, `dred_vectors.sh`, and `opus_build_test.sh` drive the CLI/vector harnesses (`opus_demo`, `opus_compare`, API tests) over full encode/decode pipelines; the Rust workspace lacks those command-line drivers and the completed top-level encode/decode flow they assume.
- `celt/tests/test_unit_{laplace,mathops,entropy,types,mdct,cwrs32,rotation,dft}.c` validate the CELT primitives against reference vectors. Although many primitives are already ported (range coder, Laplace models, CWRS, MDCT, rotation), there is no Rust harness mirroring these vector checks, and the remaining DFT/KISS-FFT plumbing stays C-only.
- `dnn/adaconvtest.c`, `dnn/test_vec.c`, and the TF2 scripts exercise the neural PLC/DNN stack, which has not been translated to Rust.
