# mousiki

An Opus decoder library in Rust, ported from the Go implementation in `pion/opus`. The core crate is `#![no_std]` and performs zero heap allocations.


### Capabilities and limitations (current)
- Mono only; stereo is not supported yet.
- SILK-only mode; Hybrid and CELT are not supported yet.
- Single-frame packets only; multi-frame/variable-frame packets currently return “unsupported”.
- Output is 48 kHz PCM:
  - Either `i16` little-endian (LE) or `f32`, depending on the API you call.
  - A 20 ms frame yields 960 samples (mono), i.e., 1920 bytes for `i16` output.


## Quick start

### Run the examples
- Decode Ogg Opus (mono, SILK-only) to a PCM file:

```bash
cargo run --example decode -- testdata/tiny.ogg output_mono.pcm
```

- Play directly (requires an audio output device; uses `cpal`):

```bash
cargo run --example playback -- testdata/tiny.ogg
```

### Use in your code
The snippet below decodes a single Opus packet into `i16` PCM (mono, 48 kHz):

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

Tip: If your input is an Ogg-encapsulated Opus stream, see the example `oggreader` usage to extract raw Opus packets before decoding.

## Compatibility and roadmap
- Target environments: no_std, constrained memory, real-time systems, no dynamic allocation.
- Implemented: SILK-only, mono, single-frame packets, 48 kHz output, basic examples (decode/playback).
- Planned:
  - Stereo support.
  - Hybrid/CELT decode paths.
  - Richer frame structures (multi-frame/variable-frame) support.

## License and acknowledgements
- License: MIT (see `LICENSE`).
- Thanks to the upstream `pion/opus` (Go) implementation and the community.

