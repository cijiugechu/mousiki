# Hybrid Decode Gap Closure Plan

This note summarizes the missing pieces needed to run `opus-c/tests/test_opus_decode.c`
against the Rust port, focusing on the hybrid (SILK+CELT) decode path.

## Current Gap Summary (updated)
- Hybrid packets are still treated as CELT-only in `src/opus_decoder.rs`, so the
  shared range-decoder path is not exercised for hybrid decode.
- The CELT decoder (`src/celt/celt_decoder.rs`) still rejects external range decoders,
  preventing reuse of the SILK range state for hybrid packets.
- Redundancy + `range_final` handling is not aligned with `opus_decoder.c`.
- Hybrid-specific decode tests from `opus-c/tests/test_opus_decode.c` are not ported.

## Step-by-Step Plan
1) **Unify the range decoder implementation** (DONE)
   - Move SILK decode paths to use `EcDec` (or wrap `RangeDecoder` around `EcDec`).
   - Update SILK decode helpers to accept a mutable `EcDec` instead of `RangeDecoder`.
   - Target files:
     - `src/silk/dec_api.rs`
     - `src/silk/decode_frame.rs`
     - `src/silk/decode_indices.rs`
     - `src/silk/decode_pulses.rs`
     - `src/silk/shell_coder.rs`
     - `src/silk/code_signs.rs`
     - `src/silk/stereo_decode_pred.rs`
   - Completed:
     - SILK decode now consumes `EcDec` via the `SilkRangeDecoder` trait.
     - `RangeEncoder` is backed by `EcEnc`, aligning round-trip tests with `EcDec`.
     - `entdec` icdf decode logic matches the reference loop condition.
     - Laplace decode paths now use `EcDec`.

2) **Allow CELT to accept an external range decoder**
   - In `src/celt/celt_decoder.rs`, remove the "external decoder is unsupported"
     guard in `celt_decode_with_ec_dred`.
   - Teach `prepare_frame` to reuse the supplied `EcDec` (no re-init), while
     preserving raw-bits handling.

3) **Restore the hybrid flow in the top-level decoder**
   - In `src/opus_decoder.rs`, treat only `MODE_CELT_ONLY` as CELT-only.
   - For hybrid packets, create a single `EcDec` and pass it through SILK,
     then reuse the same decoder for CELT.
   - This also fixes hybrid FEC paths that are currently forced into PLC.

4) **Match redundancy + rangeFinal handling**
   - Replicate the reference logic that reduces decoder storage by
     `redundancy_bytes` and uses `ec_tell` for bounds checks.
   - Compute `range_final` from the shared decoder (`dec.ctx().rng`) and XOR
     with `redundant_rng`, mirroring `opus_decoder.c`.

5) **Update tests and porting status**
   - Port the hybrid sections of `opus-c/tests/test_opus_decode.c`.
   - Update `PORTING_STATUS.md` and `src/celt/PORTING_STATUS.md` to remove
     the "hybrid is CELT-only" and "external range decoder unsupported" notes.

## Reference C Entry Points
- `opus-c/src/opus_decoder.c` (`opus_decode_frame`, `opus_decode_native`)
- `opus-c/silk/dec_API.c` (SILK decode range usage)
- `opus-c/celt/entdec.c` (range decoder used by CELT)
