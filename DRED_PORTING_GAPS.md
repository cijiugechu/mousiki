# DRED Porting Gaps (C vs Rust)

This document summarizes DRED-related gaps by comparing the C reference
implementation under `opus-c` with the Rust port in this repository.

## High-level gaps

- The core DRED model and coding pipeline (encoder/decoder, RDO-VAE, and
  associated constants/data) are not implemented on the Rust side.
- The Rust DRED API surface exists as stubs and always reports
  `Unimplemented` for functional calls.
- DRED packet extension parsing and payload insertion are missing in Rust.
- Integration with the PLC/FEC feature path is absent in Rust.
- DRED-specific tests and vector tooling are not ported.

## Missing modules and data

C provides the full DRED pipeline in:
- `opus-c/dnn/dred_encoder.c`, `opus-c/dnn/dred_decoder.c`,
  `opus-c/dnn/dred_coding.c`
- RDO-VAE and constants/data:
  `opus-c/dnn/dred_rdovae_enc.h`, `opus-c/dnn/dred_rdovae_dec.h`,
  `opus-c/dnn/dred_rdovae_enc_data.h`, `opus-c/dnn/dred_rdovae_dec_data.h`,
  `opus-c/dnn/dred_rdovae_constants.h`, `opus-c/dnn/dred_rdovae_stats_data.h`

Rust has no corresponding implementations or data modules; only stubs exist in:
- `src/dred.rs`

## Missing data structures

C structures hold DRED state and model data:
- `struct OpusDRED` and `struct OpusDREDDecoder` in
  `opus-c/dnn/dred_decoder.h`

Rust types are skeletal and do not mirror the C state:
- `OpusDred`, `OpusDredDecoder` in `src/dred.rs`

## Missing decoder API behavior

C implements full DRED decode behavior in:
- `opus-c/src/opus_decoder.c` (`opus_dred_decoder_ctl`,
  `opus_dred_parse`, `opus_dred_process`, `opus_decoder_dred_decode*`)

Rust counterparts in `src/dred.rs` are stubbed and always return
`OpusDredError::Unimplemented`.

## Missing extension parsing and payload wiring

C locates DRED payloads via the packet padding extension:
- `dred_find_payload` in `opus-c/src/opus_decoder.c`
- DRED extension constants in `opus-c/dnn/dred_config.h`

Rust does not implement DRED-specific extension IDs or parsing, and the
generic extension helpers are not connected to DRED.

## Missing encoder integration

C encoder path includes:
- DRED model load/reset (`dred_encoder_load_model`,
  `dred_encoder_reset`) in `opus-c/src/opus_encoder.c`
- Latent computation and DRED encoding (`dred_compute_latents`,
  `dred_encode_silk_frame`)
- Packet extension insertion for DRED payloads

Rust only contains bitrate allocation and activity history logic in
`src/opus_encoder.rs`, without any model loading or payload generation.

## Missing PLC/FEC integration

C injects DRED features into the PLC/FEC path in:
- `opus-c/src/opus_decoder.c` (DRED-backed FEC in `opus_decode_native`)

Rust has a limited PLC helper in `src/celt/deep_plc.rs` and does not
consume DRED features or offsets.

## Missing tests and tools

C includes DRED-specific tests and vector tooling:
- `opus-c/tests/test_opus_dred.c`
- `opus-c/tests/dred_vectors.sh`

Rust has no equivalent DRED test coverage.
