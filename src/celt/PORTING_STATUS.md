# CELT Porting Status

This document tracks which pieces of the reference CELT implementation have been
ported to Rust and which remain to be translated. The goal is to keep an
up-to-date map of dependencies so that future porting work can sequence modules
safely.

## Rust code that already mirrors the C sources

### `bands.rs`
- `hysteresis_decision` &rarr; ports the decision helper from `celt/bands.c`
  that keeps the spread classification stable when values hover near
  thresholds.
- `celt_lcg_rand` &rarr; mirrors the linear congruential pseudo-random number
  generator used by the band analysis heuristics in `celt/bands.c`.
- `bitexact_cos` and `bitexact_log2tan` &rarr; translate the bit-exact cosine
  and logarithmic tangent approximations used by the stereo analysis tools in
  `celt/bands.c`.
- `compute_channel_weights` &rarr; ports the stereo weighting helper from
  `celt/bands.c` that balances distortion across channels using adjusted
  energy estimates.
- `compute_band_energies` &rarr; ports the float helper from `celt/bands.c`
  that accumulates per-band MDCT magnitudes before normalisation.
- `normalise_bands` &rarr; mirrors the float implementation from
  `celt/bands.c` that scales each MDCT band to unit energy using the computed
  magnitudes.
- `stereo_split` &rarr; ports the mid/side-to-left/right transform from
  `celt/bands.c`, applying the orthonormal scaling used when decoding stereo
  bands.
- `stereo_merge` &rarr; mirrors the float mid/side reconstruction helper from
  `celt/bands.c`, including the energy guards and normalisation gains used when
  converting encoded mid/side pairs back to left/right channels.

### `math.rs`
- `fast_atan2f` &rarr; mirrors the helper of the same name in
  `celt/mathops.h`.
- `celt_log2`, `celt_exp2`, `celt_div`, and `celt_cos_norm` &rarr; float build
  math helpers implemented in `celt/mathops.h`/`mathops.c`.
- `celt_ilog2` and `celt_zlog2` &rarr; wrap the integer logarithm helpers from
  `celt/mathops.h`, forwarding to the shared range coder bit count while
  handling zero inputs.
- `celt_sqrt`, `celt_rsqrt`, `celt_rsqrt_norm`, `celt_rcp`, `frac_div32`, and
  `frac_div32_q29` &rarr; remaining scalar helpers from `celt/mathops.c`.
- `isqrt32` &rarr; integer square root routine from `celt/mathops.c`.
- `opus_limit2_checkwithin1` &rarr; scalar sample clamp helper from
  `celt/mathops.c`.
- `celt_float2int16` &rarr; float-to-int16 conversion helper from
  `celt/mathops.c`.
- `celt_maxabs16` and `celt_maxabs32` &rarr; helpers returning the largest
  absolute sample magnitude from `celt/mathops.c`.

### `math_fixed.rs`
- `celt_rsqrt_norm`, `celt_sqrt`, and `celt_rcp` &rarr; port the fixed-point
  reciprocal square root, square root, and reciprocal helpers from
  `celt/mathops.c`, maintaining the Q-format arithmetic used by the scalar
  build.
- `_celt_cos_pi_2` and `celt_cos_norm` &rarr; translate the Q15 cosine
  approximation employed by the MDCT window generation routines when CELT is
  built without floating-point support.
- `frac_div32_q29` and `frac_div32` &rarr; port the fractional division helpers
  that keep intermediate values within range while producing Q29/Q31 quotients.

### `types.rs`
- Scalar aliases `OpusInt32`, `OpusUint32`, `OpusVal16`, `OpusVal32`,
  `CeltSig`, `CeltGlog`, and `CeltCoef` &rarr; match the primitive type
  definitions in `celt/celt.h`.
- `KissFftState` marker type and `KissTwiddleScalar` &rarr; cover the KISS FFT
  state and scalar types from `celt/kiss_fft.h` and `celt/mdct.h`.
- `MdctLookup` &rarr; ports the `mdct_lookup` struct from `celt/mdct.h`.
- `PulseCache` &rarr; covers the pulse cache fields referenced from
  `celt/modes.h`.
- `OpusCustomMode` &rarr; mirrors `struct OpusCustomMode` from `celt/celt.h` and
  related headers, capturing the mode metadata (band layout, MDCT, caches).
- `AnalysisInfo` &rarr; ports the struct declared in `celt/celt.h`.
- `SilkInfo` &rarr; mirrors the auxiliary SILK state stored inside the encoder
  (`celt/celt.h`).
- `OpusCustomEncoder` &rarr; Rust equivalent of the encoder state defined in
  `celt/celt_encoder.c`/`celt/celt.h`, including the dynamic buffers it points
  to in C.
- `OpusCustomDecoder` &rarr; Rust equivalent of the decoder state defined in
  `celt/celt_decoder.c`/`celt/celt.h`.

### `vq.rs`
- Spread constants `SPREAD_*` and rotation helpers &rarr; ported from the
  definitions in `celt/bands.h` and `celt/vq.c`.
- `exp_rotation` and the helper `exp_rotation1` &rarr; mirror the coefficient
  rotation routines in `celt/vq.c`.

### `laplace.rs`
- `ec_laplace_encode`, `ec_laplace_decode`, `ec_laplace_encode_p0`, and
  `ec_laplace_decode_p0` &rarr; port the Laplace probability model from
  `celt/laplace.c`, including the helper `ec_laplace_get_freq1`.

### `entcode.rs`
- `EcCtx`, `EcWindow`, and helper accessors &rarr; mirror the shared range coder
  context declared in `celt/entcode.h`.
- `ec_ilog`, `ec_tell`, and `ec_tell_frac` &rarr; port the bit accounting
  routines implemented in `celt/entcode.c`.
- `SMALL_DIV_TABLE`, `celt_udiv`, and `celt_sudiv` &rarr; translate the
  optimised small divisor helpers from `celt/entcode.c`/`entcode.h`.

### `entdec.rs`
- `EcDec` and its helpers &rarr; port the scalar range decoder in `celt/entdec.c`,
  covering normalisation, Laplace/ICDF decoding, unsigned integer decoding, and
  raw bit extraction from the tail of the stream.

### `entenc.rs`
- `EcEnc` and helper routines &rarr; port the scalar range encoder in
  `celt/entenc.c`, including carry propagation, binary/ICDF symbol coding,
  unsigned integer support, raw bit packing, and buffer finalisation.

### `cwrs.rs`
- `log2_frac` &rarr; ports the conservative fractional logarithm estimator used
  by the pulse codeword enumerator in `celt/cwrs.c`.
- `unext`, `uprev`, and `ncwrs_urow` &rarr; port the small-footprint PVQ
  recurrence helpers from `celt/cwrs.c` that build `U(n, k)` rows without the
  precomputed tables.
- `icwrs1`, `icwrs`, `cwrsi`, `encode_pulses`, and `decode_pulses` &rarr; port
  the small-footprint PVQ indexer, decoder, and entropy coder glue from
  `celt/cwrs.c`.
- `get_required_bits` &rarr; ports the custom-mode helper from `celt/cwrs.c`
  that computes the fractional bit requirements for each pulse count when
  building pulse caches.

### `lpc.rs`
- `celt_lpc` &rarr; ports the float Levinson-Durbin recursion `_celt_lpc()` from
  `celt/celt_lpc.c`, producing predictor coefficients from an autocorrelation
  sequence.
- `celt_fir` and `celt_iir` &rarr; translate the FIR/IIR helpers in
  `celt/celt_lpc.c` for the float build, supplying the filter primitives used by
  the pitch analysis and postfilter paths.
- `celt_autocorr` &rarr; evaluates the autocorrelation sequence with optional
  analysis windowing, matching `_celt_autocorr()` in `celt/celt_lpc.c` for the
  float configuration.

### `pitch.rs`
- `celt_inner_prod` &rarr; scalar dot product helper from `celt/pitch.c` used by the
  pitch correlation routines.
- `dual_inner_prod` &rarr; simultaneous dot-product helper mirroring
  `dual_inner_prod_c()` in `celt/pitch.c`.
- `find_best_pitch` &rarr; selects the two strongest normalised pitch candidates
  from `celt/pitch.c`'s coarse correlation sweep.
- `compute_pitch_gain` &rarr; float pitch gain normalisation matching the
  `compute_pitch_gain()` utility in `celt/pitch.c`.
- `celt_pitch_xcorr` &rarr; scalar pitch cross-correlation routine from
  `celt/pitch.c` that evaluates delayed inner products between the excitation and
  target windows.
- `celt_fir5` &rarr; ports the 5-tap FIR helper from `celt/pitch.c` used during
  the downsampling pre-filter.
- `pitch_downsample` &rarr; mirrors the low-pass downsampling stage implemented
  in `celt/pitch.c`, including the LPC-based pre-emphasis used before the pitch
  search.

### `rate.rs`
- `MAX_PSEUDO`, `LOG_MAX_PSEUDO`, `CELT_MAX_PULSES`, `MAX_FINE_BITS`,
  `FINE_OFFSET`, `QTHETA_OFFSET`, and `QTHETA_OFFSET_TWOPHASE` &rarr; translate
  the constant definitions shared through `celt/rate.h`.
- `get_pulses` &rarr; mirrors the inline helper that maps pseudo-pulse indices
  to their pulse counts in `celt/rate.h`.
- `fits_in32` &rarr; ports the custom-modes guard from `celt/rate.c` that checks
  whether `V(N, K)` fits inside an unsigned 32-bit integer when building the
  pulse cache.
- `compute_pulse_cache` &rarr; recreates the PVQ cache generation routine from
  `celt/rate.c`, including the per-band bit caps used by custom modes.

### `quant_bands.rs`
- `loss_distortion` &rarr; ports the distortion metric helper from
  `celt/quant_bands.c` that scores how far the current coarse band energies have
  drifted from the historical estimates, clamping the accumulated squared
  difference to a conservative upper bound.
- `E_MEANS`, `PRED_COEF`, `BETA_COEF`, `BETA_INTRA`, `E_PROB_MODEL`, and
  `SMALL_ENERGY_ICDF` &rarr; port the constant tables from `celt/quant_bands.c`
  used by the coarse energy quantiser and small energy Laplace model.

### `mini_kfft.rs`
- `MiniKissFft` and butterfly helpers &rarr; port the minimalist complex FFT
  kernels from `celt/mini_kfft.c`, including radix-2/3/4/5 butterflies and the
  recursive factor planner.
- `MiniKissFftr` &rarr; ports the real FFT wrapper from `celt/mini_kfft.c`,
  including the packing buffers and super-twiddle generation used by the MDCT
  paths.

## Remaining C modules and their dependencies

The table below lists the major `.c` files under `celt/` in the reference tree
that have not yet been ported. Dependencies are derived from the files they
`#include`, focusing on CELT-specific modules rather than generic platform
support headers.

| C module | Responsibilities | Depends on |
| --- | --- | --- |
| `bands.c` | Band energy analysis, spreading, quantisation. | `modes`, `vq`, `cwrs`, `rate`, `quant_bands`, `pitch`, `mathops` |
| `celt.c` | Top-level encoder/decoder glue (frame dispatch, overlap-add). | `mdct`, `pitch`, `bands`, `modes`, `entcode`, `quant_bands`, `rate`, `mathops`, `celt_lpc`, `vq` |
| `celt_decoder.c` | Decoder main loop, PLC, postfilter. | `mdct`, `pitch`, `bands`, `modes`, `entcode`, `quant_bands`, `rate`, `mathops`, `celt_lpc`, `vq`, `lpcnet` |
| `celt_encoder.c` | Encoder analysis, bit allocation, transient detection. | `mdct`, `pitch`, `bands`, `modes`, `entcode`, `quant_bands`, `rate`, `mathops`, `celt_lpc`, `vq` |
| `kiss_fft.c` | KISS FFT backend used by the MDCT. | `kiss_fft`, `mathops`, `stack_alloc` |
| `mdct.c` | Forward/inverse MDCT built on top of KISS FFT. | `mdct`, `kiss_fft`, `mathops` |
| `modes.c` | Mode construction, static tables, precomputed caches. | `celt`, `modes`, `rate`, `quant_bands` |
| `pitch.c` | Pitch correlation/search and postfilter helpers. | `modes`, `mathops`, `celt_lpc` |
| `quant_bands.c` | Band quantisation tables and rate allocation. | `quant_bands`, `laplace`, `mathops`, `rate` |
| `rate.c` | Remaining bitrate distribution heuristics (the helper constants and pulse cache builder now live in Rust). | `modes`, `cwrs`, `entcode`, `rate` |
| `vq.c` (remaining parts) | Pulse allocation, PVQ search, and quantiser core. | `mathops`, `cwrs`, `bands`, `rate`, `pitch` |

Additional directories (`arm/`, `mips/`, `x86/`) contain architecture-specific
optimisations that depend on the scalar implementations above and remain to be
ported once the scalar logic is in place.

## Modules intentionally left unported

### `os_support.h`
- The C helpers in `celt/os_support.h` wrap manual heap management and raw
  memory utilities. Rust's standard library already provides safe and idiomatic
  equivalents (`Vec`, RAII-managed drops, and slice copying/clearing
  primitives), so introducing a dedicated wrapper module would only duplicate
  existing functionality without aiding the porting effort.
