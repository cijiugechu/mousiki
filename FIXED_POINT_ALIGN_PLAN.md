# Fixed-Point Alignment Plan (Code-Based Status)

This file reflects the **current status based on code inspection only**.
All previous checklist items are removed and replaced with the findings below.

## Current conclusion (as of code review)
Fixed-point backend is **not complete**. The fixed MDCT/energy path is now wired,
encoder prefilter/normalisation are switched to fixed-point paths, and decoder
comb/filter postfilter paths are now wired to fixed-point comb filtering.
Decoder PLC pitch search, PLC math paths, and packet-loss noise renormalisation
are now wired to fixed-point helpers. Decoder core state/buffers now have
fixed-native primary storage in fixed builds, with float caches kept as
transitional decode working buffers. Decoder `celt_decode_with_ec_dred`
band synthesis/dataflow now uses fixed-native spectrum buffers as the fixed
build center path, with localized bridge conversions only around
`quant_all_bands`/`anti_collapse`.
The previously tracked decoder checklist is complete, but global fixed-point
alignment is still pending in decode dataflow and removal of transitional
float-cache bridges.

## Verified gaps (from code)
- Decoder core runtime state remains dual-representation during transition:
  - fixed-native primary slices (`decode_mem_fixed`, `lpc_fixed`,
    `old_ebands_fixed`, `old_log_e_fixed`, `old_log_e2_fixed`,
    `background_log_e_fixed`) are now present, but decode still runs through
    float cache slices with sync barriers (`src/celt/types.rs`,
    `src/celt/celt_decoder.rs`).
- Decoder dataflow still uses localized fixed<->float bridge wrappers around
  bands PVQ entry points:
  - fixed-native spectrum is now primary in `celt_decode_with_ec_dred`, but
    `quant_all_bands` / `anti_collapse` currently require temporary float
    bridge buffers (`src/celt/celt_decoder.rs`, `src/celt/bands.rs`).
- Decoder PLC pitch path is only partially fixed:
  - fixed helpers are used for some ratios/energies, but excitation/LPC/IIR
    path remains float-buffer based (`src/celt/celt_decoder.rs`).
- Bands PVQ runtime in fixed builds still uses wrapper-style float<->fixed conversion:
  - `pvq_alg_*_runtime` and `pvq_renormalise_runtime` convert `OpusVal16`
    slices to `i16` and back (`src/celt/bands.rs`).
- Encoder fixed path still depends on frequent float/fixed state sync:
  - `sync_loge_to_fixed` / `sync_loge_from_fixed` around coarse/fine energy
    stages (`src/celt/celt_encoder.rs`).
- Public decode API still decodes to float temp then quantises to PCM:
  - `opus_custom_decode` uses temporary float buffer in fixed build too
    (`src/celt/celt_decoder.rs`).
- Project-level status still flags fixed decode backend as incomplete
  (`README.md`).

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
- Decoder packet-loss noise renormalisation is wired to fixed-point runtime:
  - Decoder loss branch now calls fixed runtime wrapper under `fixed_point` and
    routes to `renormalise_vector_fixed`.
  - C test: `ctests/celt_decoder_noise_renorm_test.c` (fixed-point run covered,
    includes non-happy-path checks).
  - Rust unit tests:
    `decoder_noise_renorm_runtime_matches_ctest_vectors`,
    `decoder_noise_renorm_runtime_panics_on_short_input`
    in `src/celt/celt_decoder.rs`.
- Decoder core state transition alignment tests are in place:
  - C test: `ctests/celt_decoder_state_test.c` (fixed-point run covered,
    includes non-happy-path checks).
  - Rust unit test: `decoder_state_transitions_match_ctest_vectors`
    in `src/celt/celt_decoder.rs`.
- Decoder dataflow alignment tests are in place:
  - C test: `ctests/celt_decoder_dataflow_test.c` (fixed-point run covered,
    includes non-happy-path checks).
  - Rust unit test: `decoder_dataflow_matches_ctest_vectors`
    in `src/celt/celt_decoder.rs`.

## Completed milestone checklist (already done)
1) `[done]` Port fixed-point `comb_filter` and wire it in encoder prefilter path.
2) `[done]` Wire decoder postfilter/PLC comb path to fixed-point runtime.
3) `[done]` Replace decoder math usage (`celt_sqrt`, `frac_div32`) with fixed-point equivalents where needed.
4) `[done]` Use fixed-point pitch downsample/search in decoder PLC path.
5) `[done]` Wire fixed-point PVQ path (`vq` fixed variants) through quant/dequant.
6) `[done]` Switch encoder normalisation to fixed path when `fixed_point` is enabled.
7) `[done]` Fill PVQ/VQ test gaps (ctests + Rust unit tests for fixed runtime wrappers).
8) `[done]` Port decoder packet-loss noise renormalisation path to fixed-point (`renormalise_vector` usage in decoder loss branch).
9) `[done]` Port decoder core state/buffers to fixed-native primary storage in fixed builds (with fixed<->float cache sync barriers).
10) `[done]` Port `celt_decode_with_ec_dred` band synthesis/dataflow to fixed-native buffers, removing float `spectrum` as fixed-build center path.
11) `[done]` Remove per-frame float<->fixed conversion in decoder postfilter by keeping postfilter working set fixed-native end-to-end.

## Remaining global alignment work (priority order)
1) `[todo]` Port decoder pitch/PLC excitation-LPC-IIR path to fixed-native buffers and arithmetic (not only helper-level fixed maths).
2) `[todo]` Replace bands PVQ fixed runtime wrappers with fixed-native interfaces to avoid repeated float<->i16 conversion.
3) `[todo]` Reduce encoder float/fixed log-energy synchronization points by converging on a fixed-native state flow under `fixed_point`.
4) `[todo]` Provide fixed-native `opus_custom_decode` sample path without mandatory float temp buffer in fixed builds.
5) `[todo]` After the above refactors, rerun ctests + Rust parity vectors and then update `README.md` fixed-point gap status.
