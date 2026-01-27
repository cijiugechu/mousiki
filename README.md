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
The full Opus front-end (SILK/CELT/Hybrid, stereo) is exposed via
`opus_encoder`/`opus_decoder`, matching the `trivial_example` round-trip:

```rust
use mousiki::opus_decoder::{opus_decode, opus_decoder_create};
use mousiki::opus_encoder::{
    opus_encode, opus_encoder_create, opus_encoder_ctl, OpusEncoderCtlRequest,
};

const SAMPLE_RATE: i32 = 48_000;
const CHANNELS: i32 = 2;
const APPLICATION: i32 = 2049; // OPUS_APPLICATION_AUDIO
const FRAME_SIZE: usize = 960; // 20 ms at 48 kHz
const MAX_FRAME_SIZE: usize = 6 * 960;
const MAX_PACKET_SIZE: usize = 3 * 1276;

let mut encoder = opus_encoder_create(SAMPLE_RATE, CHANNELS, APPLICATION)?;
opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetBitrate(64_000))?;
let mut decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS)?;

let pcm_in = [0i16; FRAME_SIZE * 2]; // interleaved stereo
let mut packet = [0u8; MAX_PACKET_SIZE];
let packet_len = opus_encode(&mut encoder, &pcm_in, FRAME_SIZE, &mut packet)?;

let mut pcm_out = [0i16; MAX_FRAME_SIZE * 2];
let decoded = opus_decode(
    &mut decoder,
    Some(&packet[..packet_len]),
    packet_len,
    &mut pcm_out,
    MAX_FRAME_SIZE,
    false,
)?;
let total_samples = decoded * CHANNELS as usize;
let _decoded_pcm = &pcm_out[..total_samples];
```

If you only need the lightweight SILK-only, single-frame decoder (mono, 48 kHz),
use `decoder::Decoder` directly:

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
For Ogg input, see the `oggreader` example to extract raw Opus packets.

## TODO
- Make the public API more Rust-idiomatic.
- Remove all `unsafe` code.

## License and acknowledgements
- License: MIT (see `LICENSE`).
- Thanks to the upstream `pion/opus` (Go) implementation and the community.
