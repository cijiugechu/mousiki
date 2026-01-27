# ctests

This directory holds standalone C tests that exercise the upstream C Opus
implementation without modifying the `opus-c` sources. The goal is to compare
C and Rust outputs for the same inputs (for example, `compute_band_energies`),
so the Rust port can be validated against the reference behavior.

## Running the tests

Use the helper script:

```sh
./ctests/run.sh
```

Common options:

```sh
./ctests/run.sh --clean -j 8
./ctests/run.sh --cmake-arg -DOPUS_CUSTOM_MODES=ON
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  --cmake-arg -DOPUS_DISABLE_INTRINSICS=ON
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_mathops_test
./ctests/run.sh -- --verbose
```

Run `./ctests/run.sh --help` to see all available options.

Fixed-point runs only build the minimal CELT subset needed by the fixed-point
tests (e.g., band energy + pitch), so the mdct/synthesis/analysis tools are
skipped in that configuration.

## Packet stream tools

The following utilities help cross-check Rust encoder/decoder behavior using
a simple OPUSPKT1 packet stream format:

```sh
ctests/build/opus_packet_encode input.pcm output.opuspkt
ctests/build/opus_packet_decode input.opuspkt output.pcm
```

Rust-side helpers live in `examples/opus_packet_tool.rs`.

## Analysis comparison

To dump the C analysis output for a PCM file:

```sh
ctests/build/analysis_compare input.pcm 64
```

To dump FFT/angle/fast_atan2/float2int intermediates from the C analysis path:

```sh
ANALYSIS_TRACE_BINS=1,61 ctests/build/analysis_fft_trace input.pcm 64
```

To dump per-stage FFT outputs (bit-reverse + each radix stage):

```sh
ANALYSIS_TRACE_BINS=1,61 ANALYSIS_TRACE_FRAME=12 \
  ctests/build/analysis_fft_stage_trace input.pcm 64
```
