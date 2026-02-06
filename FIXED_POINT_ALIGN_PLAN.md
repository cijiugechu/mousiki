# Fixed-Point Alignment Plan (Code-Based Status)

This file reflects the **current status based on code inspection only**.
All previous checklist items are removed and replaced with the findings below.

## Current conclusion (as of code review)
Fixed-point backend is **not complete**. The fixed MDCT/energy path is now wired,
and encoder prefilter/normalisation are now switched to fixed-point paths.
However, several decoder runtime paths still use float implementations.

## Verified gaps (from code)
- **Decoder comb/filter path still uses float runtime buffers**:
  - Fixed comb helpers exist (`comb_filter_fixed` / `comb_filter_const_fixed`) and are used in encoder prefilter.
  - Decoder-side PLC/postfilter path still operates on float (`CeltSig = f32`) and calls float comb filtering.

- **Decoder math still uses float helpers**:
  - `celt_sqrt`, `frac_div32` (float) are used directly in `src/celt/celt_decoder.rs`.
  - Fixed-point build still depends on float math here.

- **Decoder pitch/PLC path is float**:
  - `pitch_downsample` / `pitch_search` used unconditionally in `src/celt/celt_decoder.rs`.
  - Fixed-point variants are not used in the decoder path.

- **Decoder packet-loss noise renormalisation still uses float helper**:
  - `renormalise_vector` is still used in the decoder loss branch (`src/celt/celt_decoder.rs`).
  - The fixed-point build still routes this local renormalisation through float.

## What is already wired (confirmed)
- Fixed MDCT is used for encoder energy computation (no float->fixed MDCT mixing).
- Fixed band energy + fixed log energy (amp2_log2_fixed) now consume fixed MDCT output.
- Fixed encoder prefilter path is wired (`run_prefilter_fixed` and fixed pitch/comb helpers).
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

## Next alignment steps (code-driven)
1) `[done]` Port fixed-point `comb_filter` and wire it in encoder prefilter path.
2) `[todo]` Wire decoder postfilter/PLC comb path to fixed-point runtime.
3) `[todo]` Replace decoder math usage (`celt_sqrt`, `frac_div32`) with fixed-point equivalents where needed.
4) `[todo]` Use fixed-point pitch downsample/search in decoder PLC path.
5) `[done]` Wire fixed-point PVQ path (`vq` fixed variants) through quant/dequant.
6) `[done]` Switch encoder normalisation to fixed path when `fixed_point` is enabled.
7) `[done]` Fill PVQ/VQ test gaps (ctests + Rust unit tests for fixed runtime wrappers).
