# Fixed-Point Alignment Plan (Code-Based Status)

This file reflects the **current status based on code inspection only**.
All previous checklist items are removed and replaced with the findings below.

## Current conclusion (as of code review)
Fixed-point backend is **not complete**. The fixed MDCT/energy path is now wired,
encoder prefilter/normalisation are switched to fixed-point paths, and decoder
comb/filter postfilter paths are now wired to fixed-point comb filtering.
Decoder PLC pitch search and PLC math paths are now wired to fixed-point helpers.
However, some decoder runtime paths still use float implementations.

## Verified gaps (from code)
- **Decoder packet-loss noise renormalisation still uses float helper**:
  - `renormalise_vector` is still used in the decoder loss branch (`src/celt/celt_decoder.rs`).
  - The fixed-point build still routes this local renormalisation through float.

## What is already wired (confirmed)
- Fixed MDCT is used for encoder energy computation (no float->fixed MDCT mixing).
- Fixed band energy + fixed log energy (amp2_log2_fixed) now consume fixed MDCT output.
- Fixed encoder prefilter path is wired (`run_prefilter_fixed` and fixed pitch/comb helpers).
- Fixed decoder comb/filter postfilter path is wired under `fixed_point`:
  - `prefilter_and_fold` uses `comb_filter_fixed` in fixed-point builds.
  - Decoder postfilter sections use `comb_filter_fixed` in fixed-point builds.
- Fixed decoder PLC pitch path is wired under `fixed_point`:
  - `celt_plc_pitch_search` now uses `pitch_downsample_fixed` / `pitch_search_fixed`
    in fixed-point builds.
- Fixed PVQ/VQ runtime path is wired in bands quant/dequant:
  - `pvq_alg_quant_runtime` / `pvq_alg_unquant_runtime` route to
    `alg_quant_fixed` / `alg_unquant_fixed` under `fixed_point`.
  - Runtime renormalisation in bands uses `pvq_renormalise_runtime` and
    `renormalise_vector_fixed` under `fixed_point`.
- Fixed encoder normalisation path is wired (`normalise_bands_fixed` under `fixed_point`).
- PVQ/VQ tests are now in place:
  - C tests: `ctests/vq_test.c`, `ctests/quant_bands_test.c` (+ their fixed-point coverage additions).
  - Rust unit tests:
    `fixed_runtime_pvq_wrappers_match_direct_fixed_impl`,
    `fixed_runtime_renormalise_wrapper_matches_direct_fixed_impl`
    in `src/celt/bands.rs`.
- Decoder postfilter alignment tests are in place:
  - C tests: `ctests/celt_decoder_postfilter_test.c` (fixed-point run covered).
  - Rust unit test: `decoder_postfilter_matches_ctest_fixed_vectors`
    in `src/celt/celt_decoder.rs`.
- Decoder PLC pitch alignment tests are in place:
  - C tests: `ctests/celt_decoder_plc_test.c`, `ctests/celt_pitch_test.c`
    (fixed-point run covered).
  - Rust unit tests:
    `celt_plc_pitch_search_matches_ctest_periodic_mono_shape`,
    `celt_plc_pitch_search_matches_ctest_periodic_stereo_shape`
    in `src/celt/celt_decoder.rs`.
- Decoder PLC math alignment tests are in place:
  - C test: `ctests/celt_decoder_math_test.c` (fixed-point run covered).
  - Rust unit tests:
    `decoder_plc_decay_terms_match_ctest_vectors`,
    `decoder_plc_ratio_terms_match_ctest_vectors`
    in `src/celt/celt_decoder.rs`.

## Next alignment steps (code-driven)
1) `[done]` Port fixed-point `comb_filter` and wire it in encoder prefilter path.
2) `[done]` Wire decoder postfilter/PLC comb path to fixed-point runtime.
3) `[done]` Replace decoder math usage (`celt_sqrt`, `frac_div32`) with fixed-point equivalents where needed.
4) `[done]` Use fixed-point pitch downsample/search in decoder PLC path.
5) `[done]` Wire fixed-point PVQ path (`vq` fixed variants) through quant/dequant.
6) `[done]` Switch encoder normalisation to fixed path when `fixed_point` is enabled.
7) `[done]` Fill PVQ/VQ test gaps (ctests + Rust unit tests for fixed runtime wrappers).
8) `[todo]` Port decoder packet-loss noise renormalisation path to fixed-point (`renormalise_vector` usage in decoder loss branch).
