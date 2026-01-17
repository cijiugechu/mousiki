Fuzzing Plan (Rust, cargo-fuzz)
==============================

Goal
----
Port the C `opus_decode_fuzzer.c` behavior into a Rust fuzz target using
`cargo-fuzz` (libFuzzer backend) while keeping the same input contract
and decoder exercise pattern.

Step-by-step
------------
1. [done] Audit the C fuzzer (`opus-c/tests/opus_decode_fuzzer.c`) to capture:
   - Input framing (4-byte length + 4-byte range + packet bytes).
   - ToC-based decoder init (sample rate + channels).
   - FEC bit semantics (byte 4, bit 0).
   - Loss handling when length is zero.
   - Decode loop cap (`MAX_DECODES`) and size limits.
   Findings:
   - Guard: require `size >= SETUP_BYTE_COUNT + 1` to read the first ToC.
   - Frame format: big-endian 4-byte length, then 4-byte range; ToC is at
     `data[SETUP_BYTE_COUNT]`.
   - ToC init: `opus_packet_get_bandwidth(toc)` selects `samp_freqs = [8000,
     12000, 16000, 24000, 48000]` via `bandwidth - OPUS_BANDWIDTH_NARROWBAND`;
     channels from `opus_packet_get_nb_channels(toc)`.
   - FEC: `fec = data[i + 4] & 1` (byte 4 in the 8-byte header at each packet).
   - Loss handling: if `len == 0`, call `OPUS_GET_LAST_PACKET_DURATION` and then
     `opus_decode(dec, NULL, len, pcm, frame_size, fec)`.
   - Limits: `MAX_FRAME_SAMP = 5760`, `MAX_PACKET = 1500`, `MAX_DECODES = 12`.
   - Loop: `while i + 8 < size && num_decodes++ < MAX_DECODES`, break on
     `len > MAX_PACKET`, `len < 0`, or `i + 8 + len > size`; increment `i` by
     `8 + len`.

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
- Cargo-fuzz documentation: https://rust-fuzz.github.io/book/cargo-fuzz.html
- When you need details about cargo-fuzz behavior or options, consult the docs above or run `cargo fuzz --help` / `cargo fuzz <subcommand> --help`.
