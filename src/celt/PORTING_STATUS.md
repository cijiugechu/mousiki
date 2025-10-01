# CELT Porting Status

This document tracks which pieces of the reference CELT implementation have been
ported to Rust and which remain to be translated. The goal is to keep an
up-to-date map of dependencies so that future porting work can sequence modules
safely.

## Rust code that already mirrors the C sources

### `math.rs`
- `fast_atan2f` &rarr; mirrors the helper of the same name in
  `celt/mathops.h`.
- `celt_log2`, `celt_exp2`, `celt_div`, and `celt_cos_norm` &rarr; float build
  math helpers implemented in `celt/mathops.h`/`mathops.c`.
- `celt_sqrt`, `celt_rsqrt`, `celt_rsqrt_norm`, `celt_rcp`, `frac_div32`, and
  `frac_div32_q29` &rarr; remaining scalar helpers from `celt/mathops.c`.
- `isqrt32` &rarr; integer square root routine from `celt/mathops.c`.
- `opus_limit2_checkwithin1` &rarr; scalar sample clamp helper from
  `celt/mathops.c`.
- `celt_float2int16` &rarr; float-to-int16 conversion helper from
  `celt/mathops.c`.
- `celt_maxabs16` and `celt_maxabs32` &rarr; helpers returning the largest
  absolute sample magnitude from `celt/mathops.c`.

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
| `celt_lpc.c` | LPC analysis helpers (short-term prediction). | `celt_lpc`, `mathops`, `pitch` |
| `cwrs.c` | Combinatorial pulse encoding/decoding (excluding `log2_frac`, now ported). | `cwrs`, `mathops` |
| `entcode.c` | Range encoder utilities shared by `entenc`/`entdec`. | `entcode` |
| `kiss_fft.c` | KISS FFT backend used by the MDCT. | `kiss_fft`, `mathops`, `stack_alloc` |
| `mathops.c` | Fixed- and float-point math helpers beyond the ones already ported. | `mathops`, `float_cast` |
| `mdct.c` | Forward/inverse MDCT built on top of KISS FFT. | `mdct`, `kiss_fft`, `mathops` |
| `mini_kfft.c` | Reduced FFT variant for small MDCT sizes. | `kiss_fft` |
| `modes.c` | Mode construction, static tables, precomputed caches. | `celt`, `modes`, `rate`, `quant_bands` |
| `pitch.c` | Pitch correlation/search and postfilter helpers. | `modes`, `mathops`, `celt_lpc` |
| `quant_bands.c` | Band quantisation tables and rate allocation. | `quant_bands`, `laplace`, `mathops`, `rate` |
| `rate.c` | Bitrate distribution and pulse cache logic. | `modes`, `cwrs`, `entcode`, `rate` |
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
