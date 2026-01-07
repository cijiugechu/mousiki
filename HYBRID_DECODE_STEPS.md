# Hybrid Decode Completion Steps

This document captures the step-by-step work needed to fully enable hybrid
packet decode (shared SILK/CELT range-decoder path) in the Rust port.

## Step-by-step

1) Wire the shared range decoder for hybrid frames (done)
   - Rust now reuses a single `EcDec` for SILK then CELT when packet data is
     present, including the CELT-only path to mirror C's shared decoder
     lifecycle in `opus_decode_frame`.
   - The CELT decode call uses `decode_celt_frame_with_ec(...)` whenever a
     shared decoder exists, avoiding internal reinitialization.

2) Align CELT external-decoder handling with C
   - When an external decoder is provided, avoid altering its state or buffer
     ownership semantics.
   - Verify `packet_len`, `tell`, `ctx.storage`, and `range_final` match C after
     redundancy-byte trimming and decoder handoff.

3) Validate hybrid control flow and redundancy parsing
   - Match C logic for `redundancy`, `celt_to_silk`, and `bytes` extraction.
   - Ensure the final range reported for hybrid packets mirrors `opus_decode_native`.

4) Add hybrid decode tests
   - Use known-good hybrid packets to validate PCM output, `final_range`,
     transition smoothing, and PLC/FEC paths.
   - Place tests alongside existing decoder tests (e.g., `src/opus_decoder.rs`
     or `tests/`).

5) Update porting status docs
   - Remove the "hybrid packets decode as CELT-only" note from
     `PORTING_STATUS.md` and `src/celt/PORTING_STATUS.md` once tests pass.

## Relevant files

- `src/opus_decoder.rs`
- `src/celt/celt_decoder.rs`
- `PORTING_STATUS.md`
- `src/celt/PORTING_STATUS.md`
