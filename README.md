# mousiki

A Rust port of the Xiph `opus-c` reference implementation. The core crate is
`#![no_std]` and uses `alloc` (some APIs allocate).

### Current coverage (high level)
- Opus decode path mirrors `opus_decode_native`, including SILK/CELT/Hybrid, PLC/FEC,
  final-range reporting, and packet parsing; multistream decode and projection
  front-ends are wired.
- Opus encode front-end supports SILK-only 10/20/40/60 ms packets (plus repacketized
  >60 ms frames), CELT-only multiframe packing, and hybrid single-frame 10/20 ms
  payloads. The full encode/decode loop can run the `examples/trivial_example`
  round-trip on `testdata/ehren-paper_lights-96.pcm`.
- Repacketizer, packet helpers, extension padding, mapping/projection matrices, and
  tonality analysis are available.

### Known gaps
- Fixed-point decode backend, SIMD back-ends, and optional DRED/Deep PLC paths are
  not complete.
- See `PORTING_STATUS.md` for detailed status.


## Quick start

### Run the examples
- Decode Ogg Opus (SILK-only path used by `decoder::Decoder`) to a PCM file:

```bash
cargo run --example decode -- testdata/tiny.ogg output_mono.pcm
```

- Play directly (requires an audio output device; uses `cpal`):

```bash
cargo run --example playback -- testdata/tiny.ogg
```

- Round-trip a full 48 kHz stereo PCM sample through the trivial encoder/decoder:

```bash
cargo run --example trivial_example -- \
  testdata/ehren-paper_lights-96.pcm ehren-paper_lights-96_trivial_out.pcm
```

### Run the tests
- Full test suite:

```bash
cargo test --all-features
```

- Decode integration test (ported from `opus-c/tests/test_opus_decode.c`):

```bash
cargo test --all-features --test test_opus_decode
```

- Opt-in to the fuzz-heavy decode section (longer runtime), or to strict final-range checks:

```bash
TEST_OPUS_FUZZ=1 cargo test --all-features --test test_opus_decode
TEST_OPUS_STRICT_FINAL_RANGE=1 cargo test --all-features --test test_opus_decode
```

- DRED vector validation (optional; vectors are distributed separately):

```bash
# Fetch vectors into testdata/dred_vectors (requires DRED_VECTORS_URL).
./scripts/fetch_dred_vectors.sh --url <vector-archive-url>

# Run the vector checks (uses DRED_VECTORS_PATH or testdata/dred_vectors).
# If deep_plc_weights is disabled, set DNN_BLOB or pass --dnn-blob.
DRED_VECTORS_PATH=testdata/dred_vectors cargo test --all-features --test dred_vectors
```

### Fuzzing (manual/on-demand)
Fuzzing uses `cargo-fuzz` and is not part of CI by default.

```bash
cargo install cargo-fuzz
rustup toolchain install nightly
rustup run nightly cargo fuzz run decode_fuzzer
```

Seed corpus lives in `fuzz/corpus/decode_fuzzer/`.

### Use in your code
The snippet below uses the lightweight `decoder::Decoder` (SILK-only, single-frame)
to decode a single Opus packet into `i16` PCM (mono, 48 kHz):

```rust
use mousiki::decoder::Decoder;

// `packet` is a single Opus SILK-only, mono packet (already decontainerized; not an Ogg page).
let packet: &[u8] = /* your Opus packet */;

// Output buffer (20 ms -> 960 samples, each sample is 2 bytes for i16)
let mut pcm_bytes = [0u8; 1920];

let mut decoder = Decoder::new();
let (_bandwidth, stereo) = decoder.decode(packet, &mut pcm_bytes)?;
assert!(!stereo, "mono only for now");
// `pcm_bytes` now contains 48 kHz i16 little-endian PCM data
```

For `f32` output, use `decode_float32` and a buffer of length 960 for a 20 ms frame:

```rust
use mousiki::decoder::Decoder;

let packet: &[u8] = /* your Opus packet */;
let mut pcm_f32 = [0.0f32; 960];

let mut decoder = Decoder::new();
let (_bandwidth, stereo) = decoder.decode_float32(packet, &mut pcm_f32)?;
assert!(!stereo);
```

Tip: If you need the full Opus API surface (stereo, CELT/Hybrid, multistream),
use `opus_decoder` instead of `decoder::Decoder`. For Ogg input, see the
`oggreader` example to extract raw Opus packets.


## License and acknowledgements
- License: MIT (see `LICENSE`).
- Thanks to the upstream `pion/opus` (Go) implementation and the community.
