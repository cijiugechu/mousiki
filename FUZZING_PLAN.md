Fuzzing Plan (Rust, cargo-fuzz)
==============================

Goal
----
Port the C `opus_decode_fuzzer.c` behavior into a Rust fuzz target using
`cargo-fuzz` (libFuzzer backend) while keeping the same input contract
and decoder exercise pattern.

Step-by-step
------------
1. Audit the C fuzzer (`opus-c/tests/opus_decode_fuzzer.c`) to capture:
   - Input framing (4-byte length + 4-byte range + packet bytes).
   - ToC-based decoder init (sample rate + channels).
   - FEC bit semantics (byte 4, bit 0).
   - Loss handling when length is zero.
   - Decode loop cap (`MAX_DECODES`) and size limits.

2. Initialize `cargo-fuzz`:
   - Run `cargo fuzz init`.
   - Choose a target name (e.g. `decode_fuzzer`).
   - Keep `fuzz/` outside default test runs.

3. Implement the harness (`fuzz/fuzz_targets/decode_fuzzer.rs`):
   - Parse the same framing as the C fuzzer.
   - Use the first ToC to create the decoder.
   - Allocate a PCM buffer sized for `MAX_FRAME_SAMP * channels`.
   - Iterate over framed packets up to `MAX_DECODES`, calling
     `opus_decode` (or `opus_decode_float` if needed) with optional FEC.
   - Mirror the C behavior for zero-length (PLC) packets.

4. Seed corpus:
   - Create `fuzz/corpus/decode_fuzzer/`.
   - Add small valid Opus packets and boundary cases.
   - Optionally extract packets from `testdata/tiny.ogg`.

5. Document usage and CI policy:
   - Add short instructions in `README.md` or `docs/README.md`.
   - Decide whether fuzzing runs manually, nightly, or on-demand in CI.

Notes
-----
- The fuzz target should not change public APIs.
- Keep sizes/constants aligned with the C fuzzer for parity.
