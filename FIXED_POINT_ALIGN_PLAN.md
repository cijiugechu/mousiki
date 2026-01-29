# Fixed-Point Alignment Plan (Code-Based Status)

This file reflects the **current status based on code inspection only**.
All previous checklist items are removed and replaced with the findings below.

## Current conclusion (as of code review)
Fixed-point backend is **not complete**. The fixed MDCT/energy path is now wired,
but several critical paths still use float implementations or lack fixed-point
equivalents in the runtime code.

## Verified gaps (from code)
- **Prefilter / comb filter are float-only**:
  - `comb_filter` / `comb_filter_const` operate on `OpusVal32` in `src/celt/celt.rs`.
  - Encoder and decoder use these float versions; no fixed-point variants exist or are wired.

- **Decoder math still uses float helpers**:
  - `celt_sqrt`, `frac_div32` (float) are used directly in `src/celt/celt_decoder.rs`.
  - Fixed-point build still depends on float math here.

- **Decoder pitch/PLC path is float**:
  - `pitch_downsample` / `pitch_search` used unconditionally in `src/celt/celt_decoder.rs`.
  - Fixed-point variants are not used in the decoder path.

- **PVQ/VQ core path is float**:
  - `renormalise_vector`, `alg_quant`, `alg_unquant` in `src/celt/vq.rs` are float paths.
  - Only partial fixed helpers exist; no fixed-only end-to-end path is wired.

- **Encoder prefilter is float**:
  - `run_prefilter` in `src/celt/celt_encoder.rs` uses float buffers.
  - Fixed-point data is currently a mirror of float output.

- **Encoder normalisation path is float**:
  - Main encode path uses `normalise_bands` (float).
  - `normalise_bands_fixed` exists but is not used in the main path.

## What is already wired (confirmed)
- Fixed MDCT is used for encoder energy computation (no float->fixed MDCT mixing).
- Fixed band energy + fixed log energy (amp2_log2_fixed) now consume fixed MDCT output.

## Next alignment steps (code-driven)
1) Implement/port fixed-point `comb_filter` and wire it for encoder + decoder.
2) Replace decoder math usage (`celt_sqrt`, `frac_div32`) with fixed-point equivalents where needed.
3) Use fixed-point pitch downsample/search in decoder PLC path.
4) Wire fixed-point PVQ path (`vq` fixed variants) through quant/dequant.
5) Switch encoder normalisation to fixed path when `fixed_point` is enabled.

