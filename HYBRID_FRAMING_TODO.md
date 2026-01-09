# Hybrid framing TODO (from opus-c)

This document summarizes what remains to finish Hybrid framing in the Rust port,
using `opus-c/src/opus_encoder.c` as the reference.

## Step-by-step checklist

1. **Expand encoder state and initialization** (done)
   - Mirror the Hybrid-related fields and defaults from the C encoder state.
   - Key fields: `hybrid_stereo_width_Q14`, `width_mem`, `delay_buffer`,
     `hp_mem`, `variable_hp_smth2_Q15`, `prev_hb_gain`, `auto_bandwidth`,
     `prev_mode`, `prev_channels`, `prev_framesize`, `silk_bw_switch`,
     `first`, `nonfinal_frame`, `detected_bandwidth`, `voice_ratio`.
   - Rust file: `src/opus_encoder.rs`.

2. **Port Hybrid helper functions and wire them in** (done)
   - Required helpers from `opus-c/src/opus_encoder.c`:
     - `compute_silk_rate_for_hybrid`
     - `compute_redundancy_bytes`
     - `compute_equiv_rate`
     - `decide_fec`
     - `compute_stereo_width`
     - `user_bitrate_to_bitrate`
   - Wired into `opus_encode()` for bitrate derivation, stereo width tracking,
     FEC decision, and redundancy sizing (still before full hybrid framing).
   - Rust file: `src/opus_encoder.rs`.

3. **Implement full mode/bandwidth/channel selection (encode_native)** (done)
   - Replicate `opus_encode_native()` decision flow:
     - auto mode selection, including Hybrid eligibility
     - bandwidth auto-selection and constraints
     - redundancy switch logic (`redundancy`, `celt_to_silk`, `to_celt`)
     - SILK prefill when switching from CELT to SILK/Hybrid
   - Rust file: `src/opus_encoder.rs`.

4. **Implement Hybrid framing in the per-frame encoder** (done)
   - Port `opus_encode_frame_native()` for Hybrid packing:
     - shared range-encoder flow (SILK then CELT)
     - Hybrid redundancy flag + length signaling
     - `start_band = 17` and `CELT_SET_SILK_INFO`
     - CELT encode via `celt_encode_with_ec()` using the same range encoder
     - redundancy 5 ms frames for SILK<->CELT transitions
     - final-range handling (`rangeFinal` xor redundant range)
   - Rust files: `src/opus_encoder.rs`, `src/celt/celt_encoder.rs`,
     `src/range.rs`.

5. **Handle Hybrid multi-frame and long-frame repacketization** (partial)
   - Port the multi-frame path in `opus_encode_native()` for
     `frame_size > 20 ms` in Hybrid/CELT modes.
   - Ensure per-frame `to_celt` is only requested on the last frame.
   - Current state: long-frame Hybrid falls back to CELT-only multiframe;
     Hybrid multiframe packing still missing.
   - Rust file: `src/opus_encoder.rs`.

6. **Update and add tests** (partial)
   - Replace the current test that asserts Hybrid outputs CELT-only.
   - Add Hybrid framing vectors/expectations (TOC, bandwidth, final range).
   - Add Hybrid multi-frame and redundancy transition tests.
   - Rust files: `tests/test_opus_encode.rs`, plus new fixtures if needed.

## Reference locations

- C reference: `opus-c/src/opus_encoder.c`
- Rust encoder: `src/opus_encoder.rs`
- CELT encoder: `src/celt/celt_encoder.rs`
- Range encoder: `src/range.rs`
