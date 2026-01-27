# CELT Fixed-Point Decoder Testing Guide

This guide explains how to test the CELT fixed-point DSP backend with CELT-only audio files.

## Overview

The CELT fixed-point DSP backend has been fully ported from the C reference implementation. This includes:

- Fixed-point MDCT/FFT kernels (kiss_fft_fixed, mdct_fixed)
- Fixed-point band energy computation and normalization
- Fixed-point PVQ (Pyramid Vector Quantization) operations
- Fixed-point LPC helpers for PLC and post-filter
- Fixed-point decoder output handling

## Building with Fixed-Point Support

To build the library with fixed-point support:

```bash
cargo build --lib --features fixed_point
```

To run tests with fixed-point support:

```bash
# Run all tests
TEST_OPUS_NOFUZZ=1 cargo test --lib --features fixed_point --release

# Run just CELT tests
cargo test --lib --features fixed_point celt

# Run CELT fixed-point integration tests
cargo test --test test_celt_fixed_point --features fixed_point
```

## Decoding CELT Audio Files

### Using the decode_celt_fixed Example

The `decode_celt_fixed` example demonstrates how to decode CELT-only Opus files using the fixed-point decoder:

```bash
cargo run --example decode_celt_fixed --features fixed_point <input.opus> <output.pcm>
```

For example, with the test files mentioned in the issue (outside1.opus):

```bash
cargo run --example decode_celt_fixed --features fixed_point outside1.opus outside1_decoded.pcm
```

The output will be raw PCM data (16-bit signed, little-endian) that can be played or compared with reference implementations.

### Creating Test Files

If you don't have CELT test files, you can create them using the C opus tools:

```bash
# Using opus-tools (opusenc with CELT-only mode)
opusenc --hard-cbr --bitrate 128 --framesize 20 --music input.wav output.opus

# Or use the existing decoder example to create PCM from any opus file
cargo run --example decode --features fixed_point input.opus output.pcm
```

## Comparing with Reference Implementation

To verify correctness, you can compare the fixed-point output with:

1. **Float decoder output**: Run the same file through the decoder without `fixed_point` feature
2. **C reference implementation**: Use the opus-tools or test_opus_decode from opus-c
3. **Expected WAV file**: Compare the decoded PCM with outside1.wav using audio comparison tools

### Example Comparison Workflow

```bash
# Decode with fixed-point
cargo run --example decode_celt_fixed --features fixed_point outside1.opus outside1_fixed.pcm

# Decode with float (for comparison)
cargo run --example decode outside1.opus outside1_float.pcm

# Compare the outputs (they should be very similar but not bit-exact)
# Use audio tools like ffmpeg to convert PCM to WAV for listening:
ffmpeg -f s16le -ar 48000 -ac 1 -i outside1_fixed.pcm outside1_fixed.wav
```

## Test Coverage

The fixed-point implementation includes:

1. **Unit tests** - Integrated into each module (bands.rs, vq.rs, lpc.rs, mdct_fixed.rs)
2. **Integration tests** - test_celt_fixed_point.rs validates the full decode path
3. **Regression tests** - All 800 existing tests pass with `--features fixed_point`

## Troubleshooting

### Build Issues

If you encounter build errors:

1. Ensure you're using the correct feature flag: `--features fixed_point`
2. Check that opus-c submodule is initialized: `git submodule update --init --recursive`
3. Use `cargo clean` if you're switching between fixed_point and float builds

### Decode Issues

If decoding fails:

1. Verify the input file is a valid Opus file (Ogg container with Opus packets)
2. Check that the file contains CELT frames (not SILK-only)
3. Enable verbose logging to see decode progress

### Accuracy Issues

Fixed-point and float implementations may differ slightly due to:

1. Rounding differences in fixed-point arithmetic
2. Different precision in intermediate calculations  
3. Saturation behavior in fixed-point math

These differences should be minimal and inaudible in practice.

## Performance Notes

The fixed-point decoder is designed for:

- **Lower CPU usage** - Integer arithmetic is faster than floating-point on some platforms
- **Better embedded support** - Works on platforms without FPU
- **Deterministic behavior** - Fixed-point is more predictable across platforms

On modern x86/ARM CPUs with SIMD, float performance may be comparable or better.

## References

- [Opus CELT Specification](https://opus-codec.org/docs/)
- [Original opus-c Implementation](https://github.com/xiph/opus)
- [PORTING_STATUS.md](../PORTING_STATUS.md) - Overall porting progress
