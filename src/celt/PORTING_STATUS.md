# CELT Porting Status

This document tracks which pieces of the reference CELT implementation have been
ported to Rust and which remain to be translated. The goal is to keep an
up-to-date map of dependencies so that future porting work can sequence modules
safely.

## Rust code that already mirrors the C sources

### `bands.rs`
- `quant_partition` &rarr; ports the recursive mono-partition helper from
  `celt/bands.c` that splits bands when sufficient bits are available for
  additional time/frequency resolution.
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
- `intensity_stereo` &rarr; mirrors the intensity mixing helper from
  `celt/bands.c` that reconstructs mid-channel samples from the
  intensity-coded side data using the per-band energy weights.
- `anti_collapse` &rarr; recreates the transient noise injection used in
  `celt/bands.c` to repopulate collapsed short MDCT bands and renormalises the
  affected coefficients after seeding the pseudo-random spectrum.
- `compute_band_energies` &rarr; ports the float helper from `celt/bands.c`
  that accumulates per-band MDCT magnitudes before normalisation.
- `normalise_bands` &rarr; mirrors the float implementation from
  `celt/bands.c` that scales each MDCT band to unit energy using the computed
  magnitudes.
- `denormalise_bands` &rarr; ports the float helper from `celt/bands.c` that
  restores the MDCT spectrum by applying the exponentiated log-energy targets
  and clearing samples outside the active bandwidth when downsampling or
  silence flags are in effect.
- `stereo_split` &rarr; ports the mid/side-to-left/right transform from
  `celt/bands.c`, applying the orthonormal scaling used when decoding stereo
  bands.
- `stereo_merge` &rarr; mirrors the float mid/side reconstruction helper from
  `celt/bands.c`, including the energy guards and normalisation gains used when
  converting encoded mid/side pairs back to left/right channels.
- `special_hybrid_folding` &rarr; ports the hybrid folding helper from
  `celt/bands.c` that duplicates the initial PVQ samples so the second hybrid
  band can reuse the first band's spectrum when folding.
- `haar1` &rarr; ports the single-level Haar transform from `celt/bands.c` that
  mixes adjacent interleaved coefficients using an orthonormal sum/difference
  stage.
- `deinterleave_hadamard` &rarr; mirrors the helper from `celt/bands.c` that
  reshuffles Hadamard-transformed coefficients back into natural order using the
  "ordery" permutation employed by the spreading analysis.
- `interleave_hadamard` &rarr; ports the Hadamard interleaver from
  `celt/bands.c`, applying the stride-specific permutation that positions the DC
  term at the end of each band block.
- `compute_qn` &rarr; ports the helper from `celt/bands.c` that selects the
  number of quantisation levels used for the stereo angle, capping the
  resolution to guarantee that at least one pulse remains available for the side
  channel.
- `BandCtx` and `BandCodingState` &rarr; provide the Rust equivalents of the
  `struct band_ctx` bookkeeping and entropy coder dispatch used throughout
  `celt/bands.c`.
- `quant_band_n1` &rarr; ports the single-pulse PVQ special case from
  `celt/bands.c`, coding a raw sign bit when only one coefficient is active and
  resynthesising the unit vector for collapse prevention.
- `compute_theta` &rarr; ports the stereo/partition angle quantiser from
  `celt/bands.c`, handling probability modelling, intensity detection, and fill
  mask updates while accounting for the available fractional bit budget.
- `spreading_decision` &rarr; ports the frame-level spreading classifier from
  `celt/bands.c`, building per-band histograms, smoothing the score, and
  applying the hysteresis used to stabilise PVQ spreading decisions while
  updating the high-frequency tapset selection heuristics.
 - `quant_all_bands` &rarr; covers the top-level PVQ loop from `celt/bands.c`
   that allocates per-band bit budgets, tracks folding state, evaluates the
   two-pass theta RDO experiment, and invokes the mono/stereo band quantisers
   while updating the collapse masks.

### `celt.rs`
- `resampling_factor` &rarr; mirrors the sampling-rate-to-downsampling-factor
  mapping in `celt/celt.c`, returning the ratios used by the pitch analysis
  when converting from the 48 kHz reference rate.
- `TF_SELECT_TABLE` &rarr; ports the TF change lookup table from `celt/celt.c`
  that trades time resolution for frequency resolution based on transient
  analysis.
- `init_caps` &rarr; mirrors the allocation cap initialiser from `celt/celt.c`,
  scaling the cached per-band limits by the channel count and effective band
  width derived from the current `LM`.
- `opus_strerror` &rarr; ports the error-to-string helper from `celt/celt.c`,
  returning human-readable diagnostics for the core Opus error codes.
- `opus_get_version_string` &rarr; mirrors the version helper from `celt/celt.c`,
  exposing the library identifier used by applications to detect build
  variants.
- `comb_filter_const` &rarr; ports the scalar `comb_filter_const_c()` helper from
  `celt/celt.c`, reusing the constant tapset during the post-filter overlap
  phase while validating the slice history requirements in Rust.
- `comb_filter` &rarr; mirrors the main comb filter from `celt/celt.c`,
  including the overlap ramp, tapset interpolation, and the optimisation that
  collapses to `comb_filter_const` once the overlap region completes.

### `celt_decoder.rs`
- `CeltDecoderAlloc` &rarr; owns the trailing decoder buffers (`decode_mem`, LPC
  history, and band energy arrays) that follow `CELTDecoder` in the C layout,
  allocating them with Rust `Vec`s and exposing a safe `as_decoder()` helper to
  obtain an `OpusCustomDecoder` view.
- `DECODE_BUFFER_SIZE` &rarr; mirrors the two-kilobyte circular history kept per
  channel by `celt_decoder.c`, allowing future PLC and post-filter ports to rely
  on an accurate backing buffer size.
- `LPC_ORDER` &rarr; surfaces the decoder-side LPC history length so future
  ports of the PLC and post-filter routines share the same constant as the
  reference implementation.
- `size_in_bytes`/`reset` &rarr; utility helpers mirroring the allocation sizing
  and zeroing performed by the reference implementation when creating and
  reinitialising decoder states.
- `opus_custom_decoder_get_size`/`celt_decoder_get_size` &rarr; reproduce the
  allocation sizing helpers from `celt/celt_decoder.c`, accounting for the
  embedded `_decode_mem` sample so Rust calculations stay byte-for-byte with the
  flexible array layout.
- `init_decoder` &rarr; validates the channel layout, configures architecture
  selection, and initialises the runtime fields exposed by `OpusCustomDecoder`
  while mirroring the zeroing behaviour of `opus_custom_decoder_init()`.
- `RangeDecoderState`, `FramePreparation`, `tf_decode`, and `prepare_frame`
  &rarr; translate the frame-header parsing and bit-allocation bookkeeping that
  feed `celt_decode_with_ec()`, including range-decoder setup, dynamic
  allocation boosts, TF selection, and the post-filter parameter decoding.
- `validate_celt_decoder` &rarr; mirrors the debug-time sanity checks from
  `celt/celt_decoder.c`, ensuring the decoder state remains internally
  consistent before the synthesis path is executed.
- `deemphasis` and `deemphasis_stereo_simple` &rarr; port the post-filter output
  stage from `celt/celt_decoder.c`, providing the stereo fast path, optional
  downsampling, and accumulation into existing PCM buffers while updating the
  per-channel pre-emphasis history.
- **Still to port:** the synthesis side is largely unimplemented. The C
  routines `celt_decoder_init()` and the public wrappers such as
  `opus_custom_decode()`/`opus_custom_decode_float()` remain to be mirrored so
  the allocation matches the reference layout exactly. The signal reconstruction
  helpers (`deemphasis[_stereo]_simple()`,
  `celt_synthesis()`), PLC pipeline (`celt_plc_pitch_search()`,
  `prefilter_and_fold()`, `update_plc_state()`, `celt_decode_lost()`), and the
  main `celt_decode_with_ec()` entry point with its `*_dred` variant are still
  pending as well as the CTL dispatcher `opus_custom_decoder_ctl()`.

### `celt_encoder.rs`
- `CeltEncoderAlloc` &rarr; mirrors the encoder-side trailing buffers (`in_mem`,
  prefilter history, and band energy tracking) so the Rust port can borrow the
  same storage layout as `celt/celt_encoder.c`.
- `opus_custom_encoder_get_size`/`celt_encoder_get_size` &rarr; reproduce the
  allocation sizing helpers from the C implementation for custom and canonical
  modes.
- `CeltEncoderInitError`, `init_custom_encoder`, and
  `init_encoder_for_rate` &rarr; translate the encoder initialisation logic,
  including the reset defaults applied by `OPUS_RESET_STATE` and the
  resampling-factor validation used by `celt_encoder_init()`.
- `opus_custom_encoder_init_arch`, `opus_custom_encoder_init`,
  `celt_encoder_init`, and `opus_custom_encoder_destroy` &rarr; mirror the
  public initialisation and teardown wrappers from `celt/celt_encoder.c`,
  providing architecture selection, static-mode setup, and explicit
  deallocation hooks for callers that follow the C API layout.
- `EncoderCtlRequest` and `opus_custom_encoder_ctl` &rarr; port the encoder CTL
  dispatcher from `celt/celt_encoder.c`, replacing the varargs interface with a
  strongly typed request enum that preserves the validation and reset
  semantics of the C implementation.
- `transient_analysis` &rarr; ports the temporal masking detector from
  `celt/celt_encoder.c`, producing the per-channel transient estimate used by
  the TF resolution heuristics.
- `patch_transient_decision` &rarr; mirrors the energy-spread comparison helper
  from `celt/celt_encoder.c` that patches the transient detector when sudden
  band energy increases are observed across frames.
- `normalize_tone_input` &rarr; ports the fixed-point tone-detector scaler from
  `celt/celt_encoder.c`, rescaling the scratch buffer under the `fixed_point`
  feature while remaining a no-op for the float build.
- `compute_vbr` &rarr; ports the VBR target calculator from `celt/celt_encoder.c`
  that adjusts the per-frame bit budget based on activity, tonality, stereo
  savings, and temporal masking while respecting the constrained-VBR reservoir
  tracking rules.
- `tone_lpc` &rarr; ports the two-tap LPC solver from `celt/celt_encoder.c`
  that estimates the autoregressive tone model used by the detector, preserving
  the covariance accumulation, conditioning checks, and saturation limits.
- `acos_approx` &rarr; mirrors the fixed-point arccosine approximation from
  `celt/celt_encoder.c`, delegating to the float `acos` helper when the
  `fixed_point` feature is disabled.
- `median_of_5` &rarr; mirrors the five-sample median helper from
  `celt/celt_encoder.c` that smooths coarse band energy estimates during the
  dynamic allocation analysis.
- `median_of_3` &rarr; ports the three-sample median helper from
  `celt/celt_encoder.c`, preserving the tie-breaking behaviour relied upon by
  the energy follower bootstrap in the allocation heuristics.
- `celt_encode_with_ec` &rarr; establishes the Rust-side analysis path, covering
  pre-emphasis, MDCT evaluation, and band energy bookkeeping so the encoder
  state remains in sync with the reference implementation while bitstream
  packing is ported incrementally.
- `celt_preemphasis` &rarr; ports the input high-pass filter from
  `celt/celt_encoder.c`, converting interleaved PCM into the internal signal
  representation, applying optional clipping, handling upsampling, and keeping
  the one-sample filter memory in sync with the C layout.
- **Still to port:** key analysis and bitstream routines continue to live in
  C. The MDCT staging and comb-filter driver (`compute_mdcts()`,
  `run_prefilter()`), time/frequency allocation helpers
  (`l1_metric()`, `tf_analysis()`, `tf_encode()`, `alloc_trim_analysis()`,
  `dynalloc_analysis()`), stereo/tone detectors (`stereo_analysis()`,
  `tone_detect()`), the median filters used by the tonality estimator, and the
  public packet writers (`opus_custom_encode{,_float,_24}()` along with the
  canonical initialisation wrappers) still need Rust translations before the
  encoder can emit full CELT frames.

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
- `normalise_residual` &rarr; ports the decoder/encoder helper in `celt/vq.c`
  that scales the quantised pulse vector by the pitch gain so the mixed
  excitation preserves unit energy.
- `extract_collapse_mask` &rarr; mirrors the collapse mask generator from
  `celt/vq.c`, grouping PVQ pulses per band to flag spectral regions that need
  spreading after quantisation.
- `op_pvq_search` &rarr; ports the greedy PVQ pulse search from `celt/vq.c`,
  distributing pulses while maximising the correlation proxy used by the
  encoder.
- `alg_quant` and `alg_unquant` &rarr; translate the algebraic quantiser pair
  that wraps the PVQ search, entropy coder integration, and optional
  resynthesis of the excitation vector.
- `renormalise_vector` &rarr; mirrors the float helper that rescales excitation
  vectors to a target gain using the reciprocal square root approximation.
- `stereo_itheta` &rarr; ports the stereo intensity angle estimator that derives
  the mid/side rotation from the per-channel energies.

### `cpu_support.rs`
- `OPUS_ARCHMASK` and `opus_select_arch` &rarr; mirror the fallback CPU
  detection stub from `celt/cpu_support.h`, returning zero when runtime dispatch
  is disabled.

### `float_cast.rs`
- `CELT_SIG_SCALE` &rarr; exposes the floating-point scaling factor defined in
  `celt/float_cast.h` that bridges the public float API and the fixed-point
  internals.
- `float2int` &rarr; ports the rounding helper built on top of `lrintf()` to
  obtain saturated 32-bit integers from `f32` inputs.
- `float2int16` &rarr; mirrors the `FLOAT2INT16` macro that scales, clamps, and
  rounds float samples to signed 16-bit values.

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
- `pitch_search` &rarr; ports the coarse-to-fine open-loop lag search from
  `celt/pitch.c`, including the pseudo-interpolation that refines the best
  candidate after the decimated sweeps.
- `remove_doubling` &rarr; mirrors the subharmonic inspection helper from
  `celt/pitch.c` that suppresses doubled pitch estimates and returns the
  adjusted gain.

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
- `bits2pulses` and `pulses2bits` &rarr; translate the inline helpers from
  `celt/rate.h` that map between the cached pulse tables and their bit
  consumption, performing the same binary search used by the C implementation.
- `interp_bits2pulses` &rarr; ports the allocation interpolation helper from
  `celt/rate.c`, including the encode/decode logic for skip, intensity, and
  dual-stereo signalling.
- `clt_compute_allocation` &rarr; mirrors the top-level CELT allocation sweep
  that searches the static vectors, applies trim offsets, and delegates to the
  pulse interpolation helper.

### `quant_bands.rs`
- `loss_distortion` &rarr; ports the distortion metric helper from
  `celt/quant_bands.c` that scores how far the current coarse band energies have
  drifted from the historical estimates, clamping the accumulated squared
  difference to a conservative upper bound.
- `quant_coarse_energy` &rarr; mirrors the coarse band energy quantiser from
  `celt/quant_bands.c`, including the intra/inter frame decision logic and
  Laplace-coded residual tracking.
- `quant_fine_energy` &rarr; translates the float quantiser that refines band
  energies and pushes the raw decisions into the entropy encoder's tail bits.
- `quant_energy_finalise` &rarr; ports the float helper that allocates any
  remaining fine energy bits based on per-band priorities and updates the
  running error estimates.
- `unquant_coarse_energy` &rarr; ports the decoder-side reconstruction of the
  coarse energy decisions emitted by `quant_coarse_energy`.
- `unquant_fine_energy` &rarr; mirrors the decoder-side reconstruction of the
  fine energy steps produced by `quant_fine_energy`.
- `unquant_energy_finalise` &rarr; ports the float routine that consumes the
  final one-bit decisions used to top up the fine energy resolution.
- `amp2_log2` &rarr; ports the amplitude-to-log-energy conversion helper that
  subtracts the per-band means prior to coarse quantisation.
- `log2_amp` &rarr; mirrors the inverse conversion from `celt/quant_bands.c`
  that reconstructs linear band energies from their quantised logarithmic
  representation.
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

### `modes.rs`
- `compute_ebands` &rarr; ports the Bark-scale band layout generator from
  `celt/modes.c`, returning the dynamic band edges used when constructing custom
  CELT modes.
- `compute_allocation_table` &rarr; ports the allocation vector interpolation
  helper from `celt/modes.c`, remapping the 5 ms reference bit-allocation table
  to the dynamically generated Bark-based band layout used by custom modes.
- `compute_preemphasis` &rarr; mirrors the sampling-rate-dependent pre-emphasis
  tap selection from `celt/modes.c`, returning the four filter coefficients used
  to initialise `mode->preemph` during custom mode construction.
- `compute_mdct_window` &rarr; ports the nested sine window generation performed
  by `opus_custom_mode_create()` in `celt/modes.c`, producing the overlap-add
  coefficients required by CELT's MDCT.
- `compute_log_band_widths` &rarr; mirrors the loop in `opus_custom_mode_create()`
  that fills the `logN` table by applying `log2_frac()` to each Bark-derived
  band width, preserving the `BITRES` fractional precision.
- `opus_custom_mode_create` and `OwnedOpusCustomMode` &rarr; port the dynamic
  mode constructor from `celt/modes.c`, validating inputs, generating the Bark
  layout, allocation tables, MDCT lookup, and pulse caches, and wrapping the
  results in a safe owner that mirrors the lifetime of `CELTMode`.
- `opus_custom_mode_find_static` &rarr; recreates the `static_mode_list`
  dispatch from `celt/modes.c`, lazily constructing the shared 48 kHz / 960
  sample mode so that standard configurations reuse the precomputed
  allocation tables.

### `kiss_fft.rs`
- `KissFftState` &rarr; safe Rust wrapper around the scalar KISS FFT routines in
  `celt/kiss_fft.c`, keeping the cached twiddle tables and scratch buffers
  inside a reusable state object.
- `opus_fft_alloc`, `opus_fft`, and `opus_ifft` &rarr; expose the allocation and
  transform entry points required by the MDCT, matching the forward 1/`N`
  normalisation and unscaled inverse behaviour of the reference code.

### `mdct.rs`
- `MdctLookup::new` &rarr; owns the per-shift FFT plans and twiddle tables used
  by CELT's MDCT, mirroring the allocation performed by `clt_mdct_init()` in
  `celt/mdct.c` while enforcing Rust's safety checks on the transform sizes.
- `clt_mdct_forward` &rarr; ports the forward MDCT including the windowed
  folding step, pre/post-rotations, and N/4 complex FFT driven by the KISS FFT
  kernel, yielding the 4/`N` scaled coefficients expected by the rest of CELT.
- `clt_mdct_backward` &rarr; mirrors the inverse MDCT and TDAC overlap-add paths
  from `celt/mdct.c`, including the twiddle symmetry, inverse FFT, and final
  window mixing used to reconstruct the time-domain signal.

## Outstanding pieces of the reference sources

Only a handful of routines in the C tree remain untranslated after the work
listed above. Tracking them explicitly helps future ports focus on the pieces
that still gate a full end-to-end encoder/decoder.

| Source file | Remaining routines | Notes |
| --- | --- | --- |
| `celt/celt_decoder.c` | `celt_decoder_init()`, `celt_synthesis()`, `celt_plc_pitch_search()`, `prefilter_and_fold()`, `update_plc_state()`, `celt_decode_lost()`, `celt_decode_with_ec()`/`celt_decode_with_ec_dred()`, `opus_custom_decode{,_float,_24}()`, `opus_custom_decoder_ctl()` | The parser scaffolding is in Rust, but the synthesis/PLC loops and the public decode entry points still live in C and must be ported to complete the decoder. |
| `celt/celt_encoder.c` | `compute_mdcts()`, `l1_metric()`, `tf_analysis()`, `tf_encode()`, `alloc_trim_analysis()`, `stereo_analysis()`, `dynalloc_analysis()`, `tone_detect()`, `run_prefilter()`, `opus_custom_encode{,_float,_24}()` | The encoder currently performs the analysis preamble but still lacks the tone/stereo heuristics, dynamic allocation, prefilter, and packet emission paths that the C implementation provides. |

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
