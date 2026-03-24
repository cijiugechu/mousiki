# mousiki

A Rust port of the Xiph `opus-c` reference implementation. The core crate is
`#![no_std]` and uses `alloc` (some APIs allocate).

### Known gaps
- Fixed-point decode backend


## Quick start

### Run the examples
- Decode Ogg Opus (SILK-only path used by `decoder::Decoder`) to a PCM file:

```bash
cargo run --example decode -- testdata/tiny.ogg output_mono.pcm
```

- Play directly (requires an audio output device; uses `cpal`):

```bash
cargo run -p playback -- testdata/tiny.ogg
```

- Round-trip a full 48 kHz stereo PCM sample through the trivial encoder/decoder:

```bash
cargo run --example trivial_example -- \
  testdata/ehren-paper_lights-96.pcm ehren-paper_lights-96_trivial_out.pcm
```

- Convert the raw PCM output to WAV for playback:

```bash
ffmpeg -y -f s16le -ar 48000 -ac 2 \
  -i ehren-paper_lights-96_trivial_out.pcm \
  ehren-paper_lights-96_trivial_out.wav
```

### Run the tests
- Full test suite:

```bash
cargo test --all-features
```

- Default-build golden regression check for the trivial 48 kHz stereo round-trip
  configuration:

```bash
cargo test --test trivial_example trivial_example_default_build_golden_hash
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
The preferred high-level API is exported at the crate root and wraps the full
Opus front-end (SILK/CELT/Hybrid, stereo) with typed enums and method-based
state:

```rust
use mousiki::{Application, Bitrate, Channels, Decoder, Encoder};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_SIZE: usize = 960;
const MAX_FRAME_SIZE: usize = 6 * 960;
const MAX_PACKET_SIZE: usize = 3 * 1276;

let mut encoder = Encoder::builder(SAMPLE_RATE, Channels::Stereo, Application::Audio)
    .bitrate(Bitrate::Bits(64_000))
    .build()?;
let mut decoder = Decoder::new(SAMPLE_RATE, Channels::Stereo)?;

let pcm_in = [0i16; FRAME_SIZE * Channels::Stereo.count()];
let mut packet = [0u8; MAX_PACKET_SIZE];
let packet_len = encoder.encode(&pcm_in, &mut packet)?;

let mut pcm_out = [0i16; MAX_FRAME_SIZE * Channels::Stereo.count()];
let decoded = decoder.decode(&packet[..packet_len], &mut pcm_out, false)?;
let total_samples = decoded * Channels::Stereo.count();
let _decoded_pcm = &pcm_out[..total_samples];
```

If you need API parity with the C entry points, use
`mousiki::c_style_api::*`.

If you only need the lightweight SILK-only, single-frame decoder (mono,
48 kHz), use `decoder::Decoder` directly:

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

For `f32` output, use `decode_float32` and a buffer of length 960 for a 20 ms frame.
For Ogg input, see the `decode` example and the `mousiki::oggreader` module to
extract raw Opus packets.


## License and acknowledgements
- License: MIT (see `LICENSE`).
- Thanks to the upstream `pion/opus` (Go) implementation and the community.
