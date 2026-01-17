# Deep PLC Weights Plan

## Goal

Add a built-in DNN blob that matches the C reference "embedded weights" path
(i.e., `#ifndef USE_WEIGHTS_FILE`), so deep PLC/DRED can run without requiring
an external `OPUS_SET_DNN_BLOB` call.

## Scope

- New workspace crate: `mousiki-deep-plc-weights`
- Build-time generation of a single `weights_blob.bin`
- Auto-load the blob in Rust when the feature is enabled (to match C default)
- Keep existing external blob path unchanged (`SetDnnBlob` still works)

Out of scope:
- OSCE/LACE/NOLACE weights (not currently wired in Rust)
- Training pipeline changes

## Data Alignment With C

Match `opus-c/dnn/write_lpcnet_weights.c` payloads exactly:

- `dred_rdovae_enc_data.c`
- `dred_rdovae_dec_data.c`
- `plc_data.c`
- `fargan_data.c`
- `pitchdnn_data.c`

Notes:
- `dred_rdovae_stats_data.c` is *not* in the C blob and should remain separate.
- This keeps Rust aligned with C's `weights_blob.bin` layout.

## Crate Design: `mousiki-deep-plc-weights`

### Cargo

- `no_std`
- Feature: `fetch` (allow downloading the model tarball at build time)

### Build Inputs

- `DNN_WEIGHTS_PATH`: directory or tarball containing the `dnn/*_data.c` files
- `DNN_WEIGHTS_URL`: optional override for the tarball URL
- `DNN_WEIGHTS_SHA256`: optional override for the tarball checksum

Behavior:
- If `DNN_WEIGHTS_PATH` is set, use it (dir or tarball).
- Else, if `fetch` is enabled, download and verify the tarball.
- Else, fail with a clear error message.

### Build Output

Generate `OUT_DIR/weights_blob.bin` with the same on-disk format as C:

- 64-byte `WeightHead` records, `head = "DNNw"`, `version = 0`
- payload bytes padded to 64-byte block size
- little-endian encoding for numeric types

The crate should expose:

```rust
pub const DNN_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/weights_blob.bin"));
```

### Parsing Strategy

Re-use the existing C-array parsing logic (from `mousiki-dred-weights`) and
extend it to:

- Parse the `WeightArray` list in each `*_data.c` file
- Map array names to parsed values
- Emit the blob in the same order as the `*_arrays` list

This avoids hard-coding names and preserves C ordering.

## Workspace Integration

### Dependencies and Features

Add optional dependency:

- `mousiki-deep-plc-weights = { path = "mousiki-deep-plc-weights", optional = true }`

Add features:

- `deep_plc_weights = ["deep_plc", "mousiki-deep-plc-weights"]`
- `deep_plc_weights_fetch = ["deep_plc_weights", "mousiki-deep-plc-weights/fetch"]`

### Auto-Load To Match C

Implement `LpcNetPlcState::load_default_model()` under
`cfg(feature = "deep_plc_weights")`, and call it during
`LpcNetPlcState::default()` (or immediately after construction) so that the
model is preloaded by default, matching the C path when `USE_WEIGHTS_FILE` is
*not* defined.

This should:

- Call `load_model(DNN_BLOB)`
- Set `loaded = true`
- Keep `SetDnnBlob` behavior unchanged (still overrides model at runtime)

### DRED Interaction

`OpusDredDecoderCtlRequest::SetDnnBlob` already accepts the blob. No behavior
changes required.

If we want a convenience path:

- Add `opus_dred_decoder_load_default_model()` that uses `DNN_BLOB`
  (feature-gated to avoid auto-loading on the DRED side, which would diverge
  from current behavior).

## Relationship With `mousiki-dred-weights`

- Keep `mousiki-dred-weights` for `dred_rdovae_stats_data` and the direct
  array-backed defaults used by the current DRED encoder/decoder paths.
- Optional follow-up: refactor shared tarball download/extract into a
  small helper module to avoid duplicated logic.

## Tests

Add minimal tests (feature-gated):

- `LpcNetPlcState::default()` sets `loaded = true` when
  `deep_plc_weights` is enabled
- `opus_decoder_ctl(SetDnnBlob(DNN_BLOB))` succeeds

Run required checks after implementation:

- `cargo check --all-features`
- `TEST_OPUS_NOFUZZ=1 cargo test --all-features --release`
- `cargo test --all-features --lib`
- `cargo clippy --all-features`

## Implementation Steps

1. (Done) Create `mousiki-deep-plc-weights` crate with `build.rs`, `README.md`,
   and `src/lib.rs`.
2. (Done) Implement C-array + `WeightArray` parsing and blob emission.
3. (Done) Add workspace dependency and features in root `Cargo.toml`.
4. (Done) Add `load_default_model()` in `src/celt/deep_plc.rs` and wire auto-load.
5. (Done) Add the feature-gated tests.
6. (Done) Update repository docs if needed (`README.md`, `DRED_PORTING_GAPS.md`).
7. (Done) Allow `dred_vectors` to default to embedded weights when
   `deep_plc_weights` is enabled; keep external blob override available.
