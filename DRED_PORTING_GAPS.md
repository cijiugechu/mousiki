# DRED Porting Gaps (C vs Rust)

This document summarizes DRED-related gaps by comparing the C reference
implementation under `opus-c` with the Rust port in this repository.

## High-level gaps

- The Rust DRED decoder now includes entropy decoding plus the RDOVAE decoder,
  queues DRED FEC features into the deep PLC state, and the neural PLC output
  path is implemented (PLC model + FARGAN synthesis). The Rust port can now
  auto-load an embedded DNN blob when `deep_plc_weights` is enabled; without
  that feature, callers must still provide a DNN blob via `SetDnnBlob`.
- DRED vector tooling is ported (`dred_vectors`), but the reference vector
  files are not stored in this repository. Vector validation requires external
  test data plus a DNN blob.

## Missing modules and data

C provides the full DRED pipeline in:
- `opus-c/dnn/dred_encoder.c`, `opus-c/dnn/dred_decoder.c`,
  `opus-c/dnn/dred_coding.c`
- RDO-VAE and constants/data:
  `opus-c/dnn/dred_rdovae_enc.h`, `opus-c/dnn/dred_rdovae_dec.h`,
  `opus-c/dnn/dred_rdovae_enc_data.h`, `opus-c/dnn/dred_rdovae_dec_data.h`,
  `opus-c/dnn/dred_rdovae_constants.h`, `opus-c/dnn/dred_rdovae_stats_data.h`

Rust currently includes the decoder-side model and data:
- RDOVAE decoder implementation in `src/dred_rdovae_dec.rs`
- Generated decoder weights and stats data via `mousiki-dred-weights`
  (`src/dred_rdovae_dec_data.rs`, `src/dred_stats_data.rs`)

Rust now includes the encoder-side model and data:
- RDOVAE encoder implementation in `src/dred_rdovae_enc.rs`
- LPCNet encoder feature extraction in `src/lpcnet_enc.rs`
- PitchDNN model in `src/pitchdnn.rs`
- Generated encoder weights via `mousiki-dred-weights`
  (`src/dred_rdovae_enc_data.rs`, `src/pitchdnn_data.rs`)

## Data structures

C structures hold DRED state and model data:
- `struct OpusDRED` and `struct OpusDREDDecoder` in
  `opus-c/dnn/dred_decoder.h`

Rust mirrors these fields for decoder-side state in:
- `OpusDred`, `OpusDredDecoder` in `src/dred.rs`

## Missing decoder API behavior

C implements full DRED decode behavior in:
- `opus-c/src/opus_decoder.c` (`opus_dred_decoder_ctl`,
  `opus_dred_parse`, `opus_dred_process`, `opus_decoder_dred_decode*`)

Rust currently implements:
- `opus_dred_parse`/`opus_dred_process` (entropy decode + RDOVAE decode)
- Experimental DRED payload discovery in `dred_find_payload`
- `opus_decoder_dred_decode*` entrypoints, with DRED features queued for deep
  PLC when the model is loaded and the feature is enabled

Still missing:
- DRED vector tooling and reference test coverage

## Extension parsing and payload wiring

C locates DRED payloads via the packet padding extension:
- `dred_find_payload` in `opus-c/src/opus_decoder.c`
- DRED extension constants in `opus-c/dnn/dred_config.h`

Rust implements DRED-specific extension IDs and payload parsing via
`OpusExtensionIterator` in `src/dred.rs` (experimental header only, matching the
current C configuration). Encoder payload insertion is now wired in
`src/opus_encoder.rs`.

## Encoder integration status

The Rust encoder now mirrors the C DRED pipeline:
- DRED model load/reset (`dred_encoder_load_model`, `dred_encoder_reset`)
- Latent computation and DRED encoding (`dred_compute_latents`,
  `dred_encode_silk_frame`)
- Packet extension insertion for DRED payloads (experimental header)

This path is gated behind the `dred` feature and requires weights provided by
`mousiki-dred-weights`.

## PLC/FEC integration status

C injects DRED features into the PLC/FEC path in:
- `opus-c/src/opus_decoder.c` (DRED-backed FEC in `opus_decode_native`)

Rust now mirrors this path: DRED decode entrypoints queue FEC features into the
deep PLC state and the CELT PLC callsite consumes them via the neural PLC
pipeline (`src/celt/deep_plc.rs`). When `deep_plc_weights` is enabled, the PLC
model is preloaded from the embedded blob (matching the C default); otherwise
the DNN blob must be provided explicitly.

## Missing tests and tools

C includes DRED-specific tests and vector tooling:
- `opus-c/tests/test_opus_dred.c`
- `opus-c/tests/dred_vectors.sh`

Rust now mirrors the randomized DRED parse/process test (ported from
`test_opus_dred.c` in `src/dred.rs`) and includes a `dred_vectors` binary that
replays the vector workflow in-process. Vector validation still depends on
external test data and a DNN blob (either embedded via `deep_plc_weights` or
supplied externally) for the FARGAN synthesis stage.
