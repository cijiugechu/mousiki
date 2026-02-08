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
  -- --tests-regex 'celt_(comb_filter|prefilter)_test'
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_fixed_encode_test
CELT_FIXED_ENC_DUMP=1 ./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_fixed_encode_test
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_mathops_test
# Includes fixed-point mathops coverage (e.g., celt_sqrt, frac_div32, frac_div32_q29).
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_decoder_math_test
# Covers decoder PLC intermediate math expressions (decay/ratio) derived from celt_decoder.c.
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_decoder_noise_renorm_test
# Covers decoder packet-loss noise renormalisation behavior in the fixed-point loss path.
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_vq_test
# Covers fixed-point PVQ/VQ core (renormalise_vector, alg_quant, alg_unquant).
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_decoder_plc_test
# Covers decoder PLC/postfilter behavior (mono+stereo, long loss runs, reset, and invalid-input paths).
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_decoder_postfilter_test
# Covers normal decode postfilter comb path (mono+stereo) and invalid-input/error paths.
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_decoder_state_test
# Covers decoder core state transitions (warmup/loss/recovery/reset) and invalid-input/error paths.
./ctests/run.sh --cmake-arg -DOPUS_CTESTS_FIXED_POINT=ON \
  --cmake-arg -DOPUS_CTESTS_ENABLE_FLOAT_API=OFF \
  -- --tests-regex celt_decoder_dataflow_test
# Covers celt_decode_with_ec_dred decode dataflow (start/end band + stream channel transitions) and invalid-input/error paths.
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
