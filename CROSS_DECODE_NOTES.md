# Cross Encoder/Decoder Notes

## Context

- Input: `ehren-paper_lights-96.opus` (Ogg/Opus, 48 kHz, stereo).
- Converted to PCM for testing:
  - `ffmpeg -i ehren-paper_lights-96.opus -f s16le -ac 2 -ar 48000 ehren-paper_lights-96.pcm`
- The example binaries expect raw 16-bit little-endian PCM (no WAV header).
- `trivial_example` reads only whole 960-sample frames, so output can be
  slightly shorter than input when the final partial frame is dropped.

## Packet Stream Tools Added

A minimal packet stream format (OPUSPKT1) is used to compare Rust/C
encode/decode directly.

- Rust tool: `examples/opus_packet_tool.rs`
  - `encode <input.pcm> <output.opuspkt>`
  - `decode <input.opuspkt> <output.pcm>`
- C tools (ctests):
  - `ctests/opus_packet_encode.c`
  - `ctests/opus_packet_decode.c`

Build C tools:
```
cmake -S ctests -B ctests/build
cmake --build ctests/build
```

## Cross Test Commands

Rust encoder -> C decoder:
```
cargo run --example opus_packet_tool -- encode ehren-paper_lights-96.pcm rust_packets.opuspkt
ctests/build/opus_packet_decode rust_packets.opuspkt rust_to_c_out.pcm
```

C encoder -> Rust decoder:
```
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm c_packets.opuspkt
cargo run --example opus_packet_tool -- decode c_packets.opuspkt c_to_rust_out.pcm
```

WAV conversion for listening:
```
ffmpeg -f s16le -ar 48000 -ac 2 -i rust_to_c_out.pcm rust_to_c_out.wav
ffmpeg -f s16le -ar 48000 -ac 2 -i c_to_rust_out.pcm c_to_rust_out.wav
```

## Observed Results (Confirmed)

- `rust_to_c_out.wav` sounds like noise, nothing like the input.
- `c_to_rust_out.wav` is in the right ballpark (audibly similar to input).
- This points to the Rust encoder path as the likely source of the issue,
  since the Rust decoder can handle C-produced packets.

Quick numeric check (RMS, approximate):
- Input RMS ~ 4435
- Rust -> C RMS ~ 32055 (very high)
- C -> Rust RMS ~ 4057 (close to input)

## Packet Header Comparison (Confirmed)

Packet bytes were compared for Rust/C encoders using the same PCM input.

- First packet matches exactly (len=3, bytes `fc ff fe`):
  - TOC: `0xfc` -> CELT, Full bandwidth, stereo, 960 samples, 1 frame.
- First payload divergence: frame 12 (TOC matches, payload length differs).
- First TOC divergence: frame 47 (Rust `0xdc` vs C `0xfc`).
- Summary across 11405 frames:
  - `length_mismatch=7900`
  - `toc_mismatch=2465`
  - `payload_mismatch=11283`

Interpretation: the TOC starts correct but payload diverges early, implying
encoder internals (analysis/CELT) diverge before header decisions change.

## CELT Allocation Trace (New)

Added targeted traces to compare `clt_compute_allocation` outputs between
Rust and C at the first payload mismatch frame.

Trace knobs:
- C: `CELT_TRACE_ALLOC=1 CELT_TRACE_ALLOC_FRAME=12`
- Rust: `CELT_TRACE_ALLOC=1 CELT_TRACE_ALLOC_FRAME=12`
  - Test driver: `CELT_TRACE_PCM=ehren-paper_lights-96.pcm CELT_TRACE_FRAMES=64 \
    cargo test -p mousiki --lib celt_alloc_trace_output -- --nocapture`

Where:
- C: `opus-c/celt/celt_encoder.c` (after `clt_compute_allocation`, before
  `quant_all_bands`)
- Rust: `src/celt/celt_encoder.rs` (same point), plus a small test harness
  `celt_alloc_trace_output` to feed PCM.

Findings (frame 12):
- C: `bits=9013`, `coded_bands=20`
- Rust: `bits=18878`, `coded_bands=21`
- `pulses/fine_quant/fine_priority` differ from band 0 onward (expected once
  `bits` diverges).

Interpretation: the earliest divergence is *before* `clt_compute_allocation`.
We then instrumented the `bits` input chain and found the root cause is
already at the `use_external` / `nb_compressed_bytes` layer.

### CELT Allocation Inputs (New)

Added `use_external`, `header_bytes`, `nb_compressed_bytes`, `tell0_frac`,
`tell`, `nb_filled_bytes`, and `tell_frac` to the allocation trace.

Findings (frame 12):
- C:
  - `use_external=1`
  - `nb_compressed_bytes=160`
  - `tell0_frac=8`, `tell=1218`, `tell_frac=1218`
  - `bits=9013`
- Rust:
  - `use_external=0`
  - `nb_compressed_bytes=313`
  - `tell0_frac=1`, `tell=1`, `tell_frac=1145`
  - `bits=18878`

Conclusion: the divergence is caused by Rust taking the non-external encoder
path (no shared `ec_enc`), which doubles the available bytes and inflates
`bits` before allocation. This is upstream of PVQ/quantization and likely
explains the early payload length mismatch.

### Opus Encode Chain Alignment (New)

Investigated the Rust `opus_encode` call chain and aligned the CELT-only path
with C’s external range-coder usage.

Change:
- `src/opus_encoder.rs` (MODE_CELT_ONLY): now creates a `RangeEncoder` and
  passes `Some(range_encoder.encoder_mut())` to `celt_encode_with_ec`, instead
  of passing a direct output buffer (which set `use_external=0`).

After change, CELT allocation trace (frame 12):
- Rust: `use_external=1`, `tell0_frac=8` (now matches C).
- Remaining mismatch: Rust `tell=1`, `nb_compressed_bytes=313` vs C
  `tell=1218`, `nb_compressed_bytes=160`.

Interpretation: with external EC aligned, Rust still diverges because the
active mode or range-coder state differs. In C, `tell` reflects earlier SILK
bits (HYBRID path); in Rust, `tell` staying at 1 suggests CELT-only or a trace
frame misalignment (prefill/redundancy calls).

### CELT Allocation: Pulse Caps + Skipped Bands (Progress)

While tracing the first payload mismatch (frame 12), band 20 diverged in
allocation:
- C: `pulses=0`, `fine_quant=1`
- Rust: `pulses=16`, `fine_quant=0`

Trace (`CELT_TRACE_ALLOC_INTERP`) showed `bits1/bits2/thresh` were identical,
but the cap differed (C=9152 vs Rust=7832), which pointed to the cached caps
table.

Fixes applied:
- Rewrote `compute_pulse_cache` caps generation in Rust to match
  `opus-c/celt/rate.c` (qtheta/fine-bit estimation + final `(4*max_bits/(C*N))-64`
  scaling).
- Added the missing skipped-band handling in `interp_bits2pulses` so bands
  beyond `coded_bands` move their bits into `ebits` and zero the PVQ bits
  (mirrors the C loop after fine allocation).

After these changes, Rust now matches C for band 20:
- `celt_alloc_interp[12].band[20].cap=9152`
- `celt_alloc[12].band[20].pulses=0`, `fine_quant=1`, `fine_priority=0`

Next: re-run packet compare / CELT RC traces to see if the frame-12 payload
mismatch is resolved.

## Analysis Module Comparison (Confirmed)

New tools were added to dump analysis output per frame for C and Rust.

- C tool: `ctests/analysis_compare.c`
  - Build: `cmake -S ctests -B ctests/build && cmake --build ctests/build`
  - Run: `ctests/build/analysis_compare input.pcm 64`
- Rust tool: `analysis_compare_output` test in `src/analysis.rs`
  - Run: `ANALYSIS_PCM=input.pcm ANALYSIS_FRAMES=64 cargo test -p mousiki --lib analysis_compare_output -- --nocapture`

Results on `ehren-paper_lights-96.pcm` (first 64 frames):
- First integer mismatch: frame 47, `analysis_info.bandwidth` (C=20, Rust=18).
- Largest float diffs appear by frame 12:
  - `tonality` diff ~ 5.69e-3 (frame 12)
  - `tonality_slope` diff ~ 7.35e-3 (frame 12)
  - `noisiness` diff ~ 1.11e-1 (frame 12)
  - `activity` diff ~ 5.55e-2 (frame 12)
  - `max_pitch_ratio` diff ~ 9.57e-1 (frame 12)

Interpretation: analysis diverges early (around frame 12), and bandwidth
selection diverges by frame 47. This aligns with the first TOC mismatch.

## Tonality Trace: Bin 27/28 (Current)

- Re-ran `analysis_compare` after switching FFT twiddles to the C static table.
  The first numeric mismatch remains at frame 12 (e.g. `analysis_info[12].tonality`
  differs by ~1e-6), so analysis still diverges.
- Narrowed the first tonality mismatch to bin 28 (band 5) in frame 12:
  - Bin 27 `tonality_raw` now matches between C and Rust.
  - Bin 27 `tonality_smooth` still differs because `tonality2_next` comes from
    bin 28, which is off by 1 ULP.
- Added a minimal bin-28 trace (C + Rust) to print only
  `x2r/x2i/fast_atan2/angle2/d2_angle2`.
  This shows `x2r/x2i` are identical but `fast_atan2` differs by ~1 ULP:
  - C: `fast_atan2=9.356510639e-01`, `angle2=1.489134878e-01`
  - Rust: `fast_atan2=9.356510043e-01`, `angle2=1.489134729e-01`
  So the remaining divergence is inside `fast_atan2f` itself (not its inputs).

Notes:
- Removed the duplicate `analysis_tonality_raw` printing from
  `ctests/analysis_fft_trace.c` (opus-c already prints it in `analysis.c`).
- Added focused `fast_atan2f` tracing in both C (`opus-c/celt/mathops.h`) and
  Rust (`src/celt/math.rs`) with `ANALYSIS_TRACE_FAST_ATAN2*` env vars, plus
  bin-28 trace in `analysis` to pinpoint the remaining 1-ULP mismatch.
- First mismatch was isolated to `t1 = y2 + cB*x2` inside `fast_atan2f`:
  `x2/y2/num/t2` matched, but `t1` differed by 1 ULP, causing `den` and
  `fast_atan2` to drift.
- Forced Rust to compute `t1/t2` via `mul_add`, aligning `t1`, `den`,
  `fast_atan2`, `angle2`, and `d2_angle2` for frame 12/bin 28.
- Re-ran `analysis_tonality_raw` for frame 12 across all bins; the new earliest
  mismatch is bin 29 (`d2_angle2_bits` differs by 1 ULP), which propagates to
  `mod2/avg_mod/tonality/tonality2` for that bin.

## Analysis Internals (Tonality) Comparison (Confirmed)

## Opus PCM Buffer / Delay Alignment (New)

Goal: confirm whether the preemphasis input divergence originates upstream in
Opus-level PCM buffering/delay compensation (before CELT prefilter).

New trace knobs:
- C (opus-c): `OPUS_TRACE_PCM_DUMP=1`, `OPUS_TRACE_PCM_FRAME=12`,
  `OPUS_TRACE_PCM_BITS=1`, `OPUS_TRACE_PCM_START=0`, `OPUS_TRACE_PCM_COUNT=8`
- Rust: same env vars, run the `opus_mode_trace_output` test harness.

Trace points:
- C: `opus-c/src/opus_encoder.c`
  - `delay_copy`: after copying delay history into `pcm_buf`
  - `pcm_buf`: after HP/DC filtering into `pcm_buf`
  - `delay_buf`: after updating `st->delay_buffer`
- Rust: `src/opus_encoder.rs`
  - `delay_copy`: inside `prepare_pcm_buffer` (scratch prefix)
  - `pcm_buf`: after HP/DC filtering
  - `delay_buf`: after `update_delay_buffer`

Commands used:
```
OPUS_TRACE_PCM_DUMP=1 OPUS_TRACE_PCM_FRAME=12 OPUS_TRACE_PCM_BITS=1 \
OPUS_TRACE_PCM_START=0 OPUS_TRACE_PCM_COUNT=8 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/opus_c_trace.opuspkt \
  > /tmp/opus_c_pcm_trace.txt

OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_PCM_DUMP=1 \
OPUS_TRACE_PCM_FRAME=12 OPUS_TRACE_PCM_BITS=1 OPUS_TRACE_PCM_START=0 \
OPUS_TRACE_PCM_COUNT=8 \
cargo test --all-features opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_pcm_trace.txt
```

Findings (frame 12):
- C `delay_copy` already non-zero (tiny values), e.g.
  `opus_pcm[12].delay_copy.ch[0].sample[0]=-2.51036456e-27`,
  bits `0x9346e43e`
- Rust `delay_copy` is exactly `-0.0`, bits `0x00000000`
- `pcm_buf` and `delay_buf` also diverge accordingly.

Conclusion:
- The **first divergence is at Opus delay history** (delay_buffer contents).
- This happens *before* CELT preemphasis/prefilter/MDCT, so the source is
  Opus-level buffering/delay compensation (`prepare_pcm_buffer` /
  `update_delay_buffer`) rather than CELT analysis itself.

## Opus PCM Buffer: Earliest Drift (New)

To find the first drift, re-ran traces on the **first 64 frames** (no frame
filter) using a truncated PCM input (`/tmp/opus64.pcm`).

Observation:
- The first substantive difference is already in **frame 0**, and it shows up
  in `pcm_buf` at indices **672..679** (these are the samples copied into
  `delay_buffer[0..]` because `frame_size + total_buffer > encoder_buffer`).
- C example (frame 0, ch0, sample 672): `-4.37445905e-28`,
  bits `0x920aa1d4`
- Rust example: `-4.374443164e-28`,
  bits `0x920aa1b3`
- Same pattern for the mirror channel.

Interpretation:
- The drift begins in **`dc_reject` output**, not in delay buffer indexing.
  The value difference is ~1 ULP in the `VERY_SMALL` accumulation path.
- This means Opus-level PCM buffering is *functionally aligned*, but the
  floating-point update chain differs slightly from C from the very first
  frame.

## CELT Quant + Band Energy + MDCT Input Traces (Latest)

### Frame Alignment Check (Confirmed)
- Using `OPUS_TRACE_MODE=1` on both C and Rust: this input is **all CELT-only**
  (`mode=1002`) for every frame.
- Therefore **CELT frame index == Opus frame index** (e.g., `celt frame 12`
  is the same frame on both sides).

### quant_all_bands pre/post trace (Confirmed)
New trace knobs:
- C: `CELT_TRACE_QUANT_BANDS=1 CELT_TRACE_QUANT_BANDS_FRAME=12 CELT_TRACE_QUANT_BANDS_BITS=1`
- Rust: same env vars (hooked in `src/celt/bands.rs`)

Findings (frame 12):
- `pulses` and `bandE` differ from **band 0 onward** already in `pre`.
- Example: band 0 pre
  - C: `pulses=238`, `bandE=0.0114397248`
  - Rust: `pulses=336`, `bandE=0.686383784`

Conclusion: divergence occurs **before quant_all_bands** (inputs already off).

### compute_band_energies trace (Confirmed)
New trace knobs:
- C: `CELT_TRACE_BAND_ENERGY=1 CELT_TRACE_BAND_ENERGY_FRAME=12 CELT_TRACE_BAND_ENERGY_BITS=1`
- Rust: same env vars (hooked in `src/celt/celt_encoder.rs`)

Findings (frame 12):
- `bandE` already diverges in both `mdct2` and `main` paths.
- Example (main, band 0):
  - C: `0.0114397248` (bits `0x3c3b6daf`)
  - Rust: `0.686383784` (bits `0x3f2fb6d9`)

Conclusion: divergence exists **at or before MDCT output**.

### Prefilter input trace (Updated)
New trace knobs:
- C: `CELT_TRACE_PREFILTER=1 CELT_TRACE_PREFILTER_FRAME=12 ...`
- Rust: same env vars (hooked in `src/celt/celt_encoder.rs`)

Findings (frame 12):
- `pre`/`post` values appear identical in decimal at very small magnitudes,
  but **bit-level traces show actual mismatches** in the prefilter input
  samples (e.g., `pre ch0 idx0` bits differ: C `0x996e8820` vs Rust `0x996e84c8`).
- The differences are tiny (~1e-23), but they occur **before** MDCT.

Conclusion: the time-domain input to MDCT is already drifting at the
`celt_preemphasis` output stage.

### MDCT input / windowed trace (Confirmed)
New trace knobs (C + Rust):
- `CELT_TRACE_MDCT_IN=1`
- `CELT_TRACE_MDCT_WINDOW=1`
- `CELT_TRACE_MDCT_FRAME=12`
- `CELT_TRACE_MDCT_BITS=1`
- `CELT_TRACE_MDCT_START=0`
- `CELT_TRACE_MDCT_COUNT=128`

Findings (frame 12):
- `celt_mdct_in` differs at **idx 0** for both `mdct2` and `main`.
  - Example (`main`, ch0/block0/idx0): C `0x996e8820` vs Rust `0x996e84c8`
- `celt_mdct_win` (folded/windowed) also differs starting at idx 0.

Conclusion: the MDCT discrepancy is **not** due to FFT/twiddles; the
**MDCT input array already diverges**. Root cause is **upstream of MDCT**
(likely `celt_preemphasis` input scaling or PCM buffer path).

### Instrumentation Summary
Added traces in:
- C: `opus-c/celt/celt_encoder.c`, `opus-c/celt/mdct.c`, `opus-c/celt/mdct.h`
- Rust: `src/celt/celt_encoder.rs`, `src/celt/mdct.rs`, `src/celt/bands.rs`

Env knobs added:
- `CELT_TRACE_QUANT_BANDS*`
- `CELT_TRACE_BAND_ENERGY*`
- `CELT_TRACE_PREFILTER*`
- `CELT_TRACE_MDCT*`, `CELT_TRACE_MDCT_IN`, `CELT_TRACE_MDCT_WINDOW`

## FFT Stage / E_last Alignment (Update)

Summary of new tracing and fixes around the FFT path for frame 12:

- Root cause of remaining 1-ULP drift moved to FFT output (bin 23 / mirror 457).
- Added static scale matching C static modes in the C stage-trace harness:
  - `ctests/analysis_fft_stage_trace.c` now uses `fft_scale()` with the
    static literals (`0.002083333f` for `nfft=480`), matching Rust.
- Added a new trace knob to target a specific radix-5 butterfly:
  - `ANALYSIS_TRACE_BFLY_INDEX` (C + Rust).
- Added/extended radix-5 per-butterfly trace to compare scratch terms.

Key findings (frame 12, bin 23 / mirror 457):
- The mismatch is in stage[4], radix-5 butterfly `bfly[73]` (indices
  `[73,169,265,361,457]`).
- All scratch terms matched *except* `scratch5.i` (1‑ULP).
- That 1‑ULP caused `out4.i` (mirror bin) to drift by 1‑ULP,
  leading to bin energy `binE` differences in band 4.

Fix applied:
- Use FMA in Rust for `scratch5` accumulation to match C’s FP contraction.
  (`scratch5_r` / `scratch5_i` now computed with `mul_add` semantics.)

After fix:
- `bfly[73]` per‑term and output bits match exactly (C vs Rust).
- `analysis_fft_trace` for bin 23 now matches (output FFT bits align).
- Band 4 `E_last` now matches at frame 12.

Implementation notes:
- `mul_add` is now routed through `libm::fmaf` to keep `no_std` builds working.
  (`src/celt/math.rs::mul_add_f32`, used by `src/celt/kiss_fft.rs`.)
- Added clippy exceptions on `src/celt/fft_twiddles_48000_960.rs` to keep
  C‑exact literal twiddle values:
  `#![allow(clippy::approx_constant, clippy::excessive_precision)]`.

Current state after alignment:
- `analysis_fft_trace` bin 23 fully aligned.
- `analysis_compare` frame 12: `E_last` still shows 1‑ULP deltas, now starting
  at band 7 (also bands 9/10/11/17 show small drift).

Next target:
- Start from band 7 and run per‑bin `binE` trace to find the new first bin
  divergence, then trace the corresponding FFT bin path (stage + butterfly).

Dumped `analysis_state` internals per frame for C and Rust to find the first
divergence inside `tonality_analysis` (same input, 64 frames).

Earliest mismatches (frame 12):
- `state.angle[61]` diff ~ 3.0e-7
- `state.d_angle[1]` diff ~ 2.5e-1
- `state.d2_angle[1]` diff ~ 9.0e-3
- `state.prev_band_tonality[0]` diff ~ 1.0e-1
- `state.mem[1]` diff ~ 2.1e-7
- `state.cmean[1]` diff ~ 2.1e-7

Other early/small diffs:
- `state.logE_last[8]` diff ~ 1.9e-6 (frame 13)
- `state.E_last[8]` diff ~ 1.1e-9 (frame 16)
- `state.std[0]` diff ~ 6.9e-5 (frame 17)
- `state.meanE[18]` diff ~ 1.3e-7 (frame 47)

Notes:
- `state.downmix_state[0..2]` matched within 1e-7 for all 64 frames.
- The first substantial deviation is in `d_angle[1]` at frame 12, which then
  propagates to `prev_band_tonality` and the higher-level `analysis_info`
  values.

Interpretation: the earliest divergence is in the angle/phase path (likely
FFT -> `angle` -> `d_angle` / `d2_angle`), not in energy accumulation.

## FFT / Angle Path Deep Dive (Confirmed)

New C/Rust traces were added to locate the first divergence inside the FFT
path that feeds `fast_atan2f`.

Tools:
- C: `ctests/analysis_fft_trace.c` (per-bin FFT/angle trace)
- C: `ctests/analysis_fft_stage_trace.c` (stage-by-stage FFT trace)
- Rust: `analysis_fft_trace_output` test + `src/celt/kiss_fft.rs` stage trace

Builds used:
- Default C build: `ctests/build`
- No intrinsics: `ctests/build-nointr` (`-DOPUS_DISABLE_INTRINSICS=ON`)
- Custom modes only: `ctests/build-custom` (`-DOPUS_CUSTOM_MODES=ON -DCUSTOM_MODES_ONLY`)

Key commands:
```
ANALYSIS_TRACE_BINS=1,61 ANALYSIS_TRACE_FRAME=12 ctests/build/analysis_fft_trace ehren-paper_lights-96.pcm 64
ANALYSIS_TRACE_BINS=1,61 ANALYSIS_TRACE_FRAME=12 ANALYSIS_PCM=ehren-paper_lights-96.pcm ANALYSIS_FRAMES=64 \
  cargo test -p mousiki --lib analysis_fft_trace_output -- --nocapture
```

Stage trace (bin 479 mirror):
```
ANALYSIS_TRACE_BINS=479 ANALYSIS_TRACE_FRAME=12 ctests/build-custom/analysis_fft_stage_trace ehren-paper_lights-96.pcm 64
ANALYSIS_TRACE_STAGE=1 ANALYSIS_TRACE_STAGE_MANUAL=1 ANALYSIS_TRACE_BINS=479 ANALYSIS_TRACE_FRAME=12 \
  ANALYSIS_PCM=ehren-paper_lights-96.pcm ANALYSIS_FRAMES=64 \
  cargo test -p mousiki --lib analysis_fft_trace_output -- --nocapture
```

Findings:
- Default C build: first numeric mismatch is `analysis_fft[12].bin[1].output_fft.{r,i}`
  (diff ~ 1.8e-9). This makes `x1i` non-zero in Rust, triggering `fast_atan2f`
  to return `-pi/2` instead of 0 at frame 12 bin 1.
- Disabling intrinsics did not change the C result; mismatch remained.
- Forcing custom modes (`CUSTOM_MODES_ONLY`) makes `output_fft.{r,i}` match Rust,
  but `output_fft_mirror.i` still differs by ~9.31e-10.
- Stage trace shows the mirror bin (479) diverges around stage 2 by ~1e-10,
  which propagates to the final `output_fft_mirror.i`.
- Attempted to disable FMA with `RUSTFLAGS="-C target-feature=-fma"`, but the
  target did not recognize `-fma` and the trace diffs were unchanged.
- Added per-butterfly stage tracing via `ANALYSIS_TRACE_BFLY_STAGE` (C/Rust),
  and aligned Rust trace formatting to C `%.9e` output for diffing.
- Stage 2 per-butterfly compare (frame 12, custom modes):
  inputs match exactly; first mismatch is `bfly[1].out[3].r` (diff ~1.46e-11),
  max diff observed ~1.17e-10 at `bfly[9].out[0].r`. With 120 butterflies
  (1920 lines / 16 per bfly), stage 2 is a radix-4 pass. This points to
  divergence inside the stage 2 butterfly math (not at stage 2 inputs).
- Added radix-4 internal dumps (`tw*`, `mul*`, `scratch*`) and re-ran stage 2:
  first mismatch now appears at `bfly[1].mul0.i` (diff ~7.3e-12), before any
  add/sub scratch values diverge. Twiddle values for the early bflies match at
  the printed precision. This indicates the first divergence is in the complex
  multiply (twiddle multiply), not the add/sub chain.
- Added hex/bits tracing for the radix-4 multiply inputs/outputs
  (`ANALYSIS_TRACE_BFLY_HEX=1`), printing `*.bits.{r,i}=0x...` for `mul_in*`,
  `tw*`, and `mul*`. Result: `mul_in*` and `tw*` bits match exactly, but the
  first mismatch is `bfly[1].mul0.bits.i` (C `0xb8df1573` vs Rust `0xb8df1572`),
  confirming the inputs are bit-identical and the divergence comes from the
  complex multiply’s arithmetic/rounding.
- With `-ffp-contract=off` (C) and `RUSTFLAGS="-C llvm-args=-fp-contract=off"`
  (Rust), the early `mul0` bit mismatch disappears; the first mismatch now
  shifts to `bfly[4].tw2.bits.r` (very small twiddle real value, C
  `0x248d3132` vs Rust `0x25a34c4c`). This suggests FP contraction/FMA was

## CELT RC / Quantization Trace (Frame 12, Updated)

After aligning the mdct2 FFT **stage trace**, re-ran tracing from the CELT
RC/quantization path for frame 12 to find the first post-MDCT mismatch.

### Range coder entry (pre/post quant_all_bands)
Trace knobs:
- C: `CELT_TRACE_RC=1 CELT_TRACE_RC_FRAME=12`
- Rust: same env vars (run via `opus_mode_trace_output`)

Findings (frame 12):
- `pre_quant_fine_energy` **matches**.
- First mismatch at `pre_quant_all_bands`:
  - C: `nbits_total=263`
  - Rust: `nbits_total=261`

Notes:
- C stage labels: `pre_quant_fine_energy`, `pre_quant_all_bands`, `post_quant`.
- Rust currently logs: `pre_quant_fine_energy`, `pre_quant_all_bands`,
  `post_quant_all_bands` (naming mismatch only).

### Fine energy encode bits
Trace knobs:
- C: `CELT_TRACE_FINE_ENERGY=1 CELT_TRACE_FINE_ENERGY_FRAME=12 CELT_TRACE_FINE_ENERGY_BITS=1`
- Rust: same env vars

First mismatch:
- band 1, ch 1
  - C: `error_before_bits=0x3eac0900`
  - Rust: `error_before_bits=0x3eac0920`
  - `error_after_bits` also differs accordingly.

### Coarse energy encode bits
Trace knobs:
- C: `CELT_TRACE_COARSE_ENERGY=1 CELT_TRACE_COARSE_ENERGY_FRAME=12 CELT_TRACE_COARSE_ENERGY_BITS=1`
- Rust: same env vars

First mismatch:
- pass `inter`, band 1, ch 1
  - `x_bits`: C `0xc15b6cd8` vs Rust `0xc15b6cd7`
  - `f_bits` and `error_before_bits` differ by the same 1-ULP.

### Band energy detail (bin-level)
Trace knobs:
- C: `CELT_TRACE_BAND_ENERGY_DETAIL=1 CELT_TRACE_BAND_ENERGY_DETAIL_FRAME=12 \
       CELT_TRACE_BAND_ENERGY_DETAIL_BAND=1 CELT_TRACE_BAND_ENERGY_DETAIL_BITS=1`
- Rust: same env vars

Findings:
- First overall mismatch (band 1, ch 0, bin 0):
  - `x2_bits`: C `0x35b63335` vs Rust `0x35b6333f`
- For band 1, ch 1, bin 0:
  - `x_bits`: C `0x3a7f111c` vs Rust `0x3a7f111f`
- Multiple later bins are also off by 1–3 ULPs; `sum` bits differ.

### MDCT idx 8 (band 1) output trace
Trace knobs:
- C: `CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_FRAME=12 CELT_TRACE_MDCT_BITS=1 \
       CELT_TRACE_MDCT_START=8 CELT_TRACE_MDCT_COUNT=1`
- Rust: same env vars

Findings (mdct2 output):
- ch 0 idx 8: C `0x3a98b6d1` vs Rust `0x3a98b6d5`
- ch 1 idx 8: C `0x3a7f111c` vs Rust `0x3a7f111f`

### MDCT stage trace (idx 8 path)
Trace knobs:
- C: `CELT_TRACE_MDCT_STAGE=1` plus idx targeting
- Rust: same env vars

Findings:
- Post-rotate `t0/t1` match.
- **FFT output (`fp.r/fp.i`) differs** by 1–2 ULP at idx 8:
  - C `fp.r=0xb9c3fead`, `fp.i=0xbab66878`
  - Rust `fp.r=0xb9c3feab`, `fp.i=0xbab6687a`

Interpretation:
- Even though the *stage-by-stage* mdct2 KFFT trace aligns, the **actual
  mdct2 FFT output still differs** at idx 8 in the real MDCT path.
- This explains the band energy drift (band 1), coarse energy drift, and the
  `pre_quant_all_bands` `nbits_total` mismatch.

### amp2log2 trace note
Rust `CELT_TRACE_AMP2LOG2` currently prints one line per band (no channel),
while C prints per-band per-channel. Diffing needs to account for this
format difference.

## MDCT2 FFT / Post-Rotate Alignment (Frame 12, Updated)

Target: the remaining band-1 energy mismatch (bin 2) traced to `mdct2` output
idx 10 (band start 8 + bin 2). `mdct2` idx 8 is now aligned.

### Change 1: Force C factor order for nfft=480 (MiniKissFft)
Rust `MiniKissFft::new` now uses the C factor list for `nfft=480`:
`[5, 96, 3, 32, 4, 8, 2, 4, 4, 1]` instead of `kf_factor`’s default
`[4, 120, 4, 30, 2, 15, 3, 5, 5, 1]`.

Result:
- `mdct2` FFT output at idx 8 now matches C exactly.

### Change 2: Use FMA in MDCT post-rotate
`post_rotate_forward` now uses `fmaf(a, b, c)` for:
- `yr = fmaf(fp.i, t1, -(fp.r * t0))`
- `yi = fmaf(fp.r, t1, fp.i * t0)`

Result (frame 12):
- `mdct2` post-rotate `yr` at idx 4 now matches C (`0x3a98b6d1`).
- `mdct2` output idx 8 now matches C for both channels:
  - ch0 `0x3a98b6d1`, ch1 `0x3a7f111c`.

### Downstream re-trace (frame 12)
Coarse energy (`CELT_TRACE_COARSE_ENERGY_BITS=1`):
- First mismatch persists at band 1, ch 1:
  - `x_bits`: C `0xc15b6cd8` vs Rust `0xc15b6cd7` (unchanged).

Band energy detail (band 1):
- First mismatch moved:
  - now `bin[2].x_bits`: C `0x3a831ec4` vs Rust `0x3a831ec5`.

MDCT output for idx 10 (band 1 bin 2):
- C: `celt_mdct[12].mdct2.ch[0].idx_bits[10]=0x3a831ec4`
- Rust: `0x3a831ec5` (ch1 also +1 ULP).

MDCT stage (idx 5 ⇒ output idx 10):
- `fp.r` mismatch at `mdct2` idx 5:
  - C `0xba80f737` vs Rust `0xba80f738` (FMA path).
  - Disabling FMA in `c_mul` shifts to `0xba80f736` (still off).

Conclusion:
- `mdct2` idx 8 alignment is fixed (factor order + post-rotate FMA).
- The next mismatch is **FFT output at mdct2 idx 5** (band 1 bin 2),
  causing `mdct2` output idx 10 and coarse energy drift to persist.
  responsible for the earlier `mul*` divergence; remaining differences are
  dominated by twiddle generation for near-zero values (pi constant / sin/cos
  rounding).
- Re-ran the stage 2 hex trace after switching Rust to the same PI literal as C
  (and keeping `-fp-contract=off`): the first mismatch still appears at
  `bfly[4].tw2.bits.r` with the same bit patterns, and subsequent `mul1.bits.i`
  differences follow. The PI literal change alone did not affect the twiddle
  bits mismatch pattern.
- Dumped the full twiddle table (`ANALYSIS_TRACE_TWIDDLES=1`, `-fp-contract=off`)
  and compared C vs Rust: only two entries differ. The first mismatch is at
  `fft_twiddles[120].bits.r` (C `0x248d3132` vs Rust `0x25a34c4c`), and the
  second is at `fft_twiddles[240].bits.i` (C `0xa50d3132` vs Rust `0xa6234c4c`).
  These correspond to near-zero components at quarter/half-cycle phases, so the
  remaining divergence is coming from trig results (cos/sin) for tiny values,
  not from butterfly arithmetic or indexing.
- Checked opus-c `kf_cexp`/`KISS_FFT_*`: `kf_cexp` uses `KISS_FFT_COS/SIN` which
  call `cos(phase)`/`sin(phase)` (no `sincos`/`sincosf`). Updated Rust to use
  the same system libm FFI on Unix (macOS `System`, others `libm`) for all builds.
  Re-dumped twiddles with the FFI path: the same two entries (120/240) still
  differ, so sincos vs sin/cos is not the cause here.
- Dumped `phase` f64 bit patterns for twiddle indices 120/240. C reports
  `fft_twiddle_phase[120].bits=0xbff921fb54442d18` and
  `fft_twiddle_phase[240].bits=0xc00921fb54442d18`, while Rust initially
  reported `0xbff921fb54442d17` / `0xc00921fb54442d17`. This was a 1-ULP
  difference in the phase input itself, pointing to phase computation order
  (C uses `(-2*pi/nfft)*i`, Rust was doing `(-2*pi*i)/nfft`). After changing
  Rust to match C's order, the phase bits now match exactly and the twiddle
  table dump shows no mismatches (with the same `-fp-contract=off` setup).
- With `-fp-contract=off` on both sides, the first remaining mismatch moved to
  stage 4 (final FFT stage): `fft_stage[12].stage[4].bin[61].r` differed by
  ~5.8e-11. Stage 0..3 were identical.
- Added radix-5 scratch tracing in `kf_bfly5` (C + Rust). The first divergence
  appeared in `scratch5.i` / `scratch11` (add-chain), while all input and
  multiply-derived scratch values (`scratch1..4`, `scratch7..10`) matched.
  This pointed to float add ordering inside radix-5 rather than mul/twiddles.
- Updated Rust `kf_bfly5` to mirror C's addition order:
  - `fout0 = scratch0 + (scratch7 + scratch8)`
  - `scratch5/11` computed as `scratch0 + (scratch7*... + scratch8*...)`
  After this, per-butterfly stage-4 traces match exactly for bins 61/419, and
  the stage-4 bin outputs are identical.
- Re-running `analysis_fft_trace` now shows full alignment for frame 12 bins
  1/61 (no mismatches). `analysis_compare` still shows a tiny float diff at
  `analysis_info[12].tonality_slope` (~9.3e-13), so remaining divergence is
  outside the FFT stage trace.

Rust twiddle adjustments:
- `src/celt/kiss_fft.rs` now computes twiddles in f64 (cos/sin) and casts to f32,
  matching opus-c's double-precision twiddle generation.
- Under test cfg, Rust can call system `sin/cos` via FFI (macOS uses `System`),
  but this did not remove the remaining ~1e-10 residuals.

Current state:
- `fast_atan2f` branch divergence at frame 12 bin 1 is resolved (both sides return 0).
- Remaining diffs are small (1e-10 to 1e-5) and appear to be due to floating
  rounding / math library behavior rather than large logic differences.

## Tonality Slope Residual (Current)

Further tracing of the remaining `analysis_info[12].tonality_slope` delta
isolated a 1-ULP mismatch in `prev_band_tonality` at frame 12, band 5:

- Added optional bit dumps to `analysis_compare`:
  - `ANALYSIS_TRACE_E_HISTORY=1` -> `E_hist_bits` for all frames/bands.
  - `ANALYSIS_TRACE_ANGLE_BITS=1` -> `angle_bits`, `d_angle_bits`,
    `d2_angle_bits`.
  - `ANALYSIS_TRACE_BAND_TONALITY_BITS=1` -> `prev_band_tonality_bits`.
- Added FFT trace outputs for `bin_e` and `tonality` (raw + smoothed) bits:
  - C: `ctests/analysis_fft_trace.c`
  - Rust: `src/analysis.rs` `fft_trace` dumps `bin_e_bits`,
    `tonality_bits`, `tonality2_bits`, and `tonality_smoothed_bits`.

Findings (frame 12, full 13-frame run with `ANALYSIS_TRACE_BINS=all`):
- `E_hist_bits` match for all frames/bands (stationarity inputs match).
- `angle_bits`, `d_angle_bits`, `d2_angle_bits` match for all bins/frames.
- `bin_e_bits`, `tonality_bits`, and `tonality_smoothed_bits` match for all
  bins/frames.
- First `prev_band_tonality_bits` mismatch is at frame 12, band 5:
  - C: `0x3d9b6b0d`
  - Rust: `0x3d9b6b0c`
- Recomputing `band_tonality` from the trace data (bin energies + smoothed
  tonality + `E_hist` stationarity) yields `0x3d9b6b0c`, matching Rust.

Interpretation (initial):
- Inputs to `band_tonality` looked bit-identical in the FFT trace path.
- The 1-ULP difference appeared to come from the C-side accumulation/precision
  in `tE/(1e-15+E)` rather than a logic mismatch in Rust.

### Tonality2 / pi4 Alignment (Update)

Further tracing inside the real `analysis.c` path contradicted the initial
assumption above and pinpointed the mismatch in `tonality2` (raw + smoothed).

New traces added:
- C: `opus-c/src/analysis.c` now prints `analysis_tonality_raw[...]`,
  `analysis_tonality_smooth[...]`, and `analysis_tonality_bin[...]` when
  `ANALYSIS_TRACE_TONALITY_SLOPE=1` (with optional `FRAME`, `BANDS`, `BITS`).
- Rust: `src/analysis.rs` prints matching `analysis_tonality_*` lines from the
  same path.

Findings (frame 12, band 5, bins 24/27):
- `binE` bits match; `float2int` and `mod2` (pre/post) bits match.
- The first bit mismatch is in `tonality2`, which propagates through smoothing
  to `tonality` and then `tE`/`band_tonality`.

Root cause and fix:
- C computes `pi4` as `(M_PI*M_PI*M_PI*M_PI)` in double, then casts to float.
- Rust used `PI` from `core::f32::consts` and computed `PI^4` in f32.
- Rust was updated to compute `pi4` as `((M_PI_F64)^4) as f32` using the same
  literal `3.141592653`.
- After this change, `tonality2_bits` and `tonality_bits` align for the traced
  bins.

Status:
- The pi4/tonality2 mismatch is resolved.
- Need to re-run `analysis_compare` end-to-end to confirm whether the
  remaining `analysis_info[12].tonality_slope` 1-ULP delta is gone.

### Stationarity Denom / sqrt Alignment (Update)

Tracing frame 25 showed residual bit mismatches isolated to band 17, which
were traced back to frame 24's stationarity calculation.

Findings:
- `E_bits[]`, `sqrtE_bits[]`, `L1_bits`, and `L2_bits` matched.
- The only remaining mismatch was `denom_bits` from
  `sqrt(1e-15 + NB_FRAMES * L2)`.
- Root cause: Rust used `sqrtf` (f32), while C used `sqrt` on double and cast
  back to float, causing a 1-ULP difference.

Fix:
- Rust now computes the denominator via a helper:
  - `stationarity_denom(l2)` uses `sqrt` in f64, casts to f32.
- This aligns `denom_bits`, and frame 25 band 17 `band_tonality`/`slope_acc`
  bits now match.

Regression test:
- Added a unit test in `src/analysis.rs` that checks
  `stationarity_denom(6.127_892_860e-9)` yields `0x39682ac1`.
- Added a unit test that asserts `pi4_f32()` matches `0x42c2d174` to guard
  against reverting to f32 `PI^4`.

### Strict FP Experiments (No Fix)

- Added per-file strict FP flags for C `analysis.c` only:
  - `ctests/CMakeLists.txt` option `OPUS_ANALYSIS_STRICT_FP`
  - Flags: `-ffp-model=strict -frounding-math -ftrapping-math -fno-vectorize
    -fno-slp-vectorize -fno-associative-math -fno-reciprocal-math`
- Built `ctests/build-ffp-analysis-strict` with `-DOPUS_ANALYSIS_STRICT_FP=ON`
  and `-ffp-contract=off`.
- Result: the 1-ULP `prev_band_tonality` mismatch (frame 12, band 5) persisted.

### Rust f64 Accumulation Attempt (Reverted)

- Tried accumulating `band_e`, `t_e`, `n_e` in f64 inside `tonality_analysis`
  (Rust) to mimic potential C extended-precision accumulation.
- This made mismatches worse (first mismatch moved to band 3 and multiple bands
  differed at frame 12).
- Change was reverted; Rust now uses the original f32 accumulation.

## Opus Mode Selection Trace (New, Pending Run)

Added a mode-selection trace at the same decision point in both C and Rust to
confirm whether the encoders choose the same mode/bandwidth on the same frame.

Where:
- C: `opus-c/src/opus_encoder.c` (end of mode/bandwidth decision, right after
  the SILK/HYBRID final adjustment and before multi-frame handling).
- Rust: `src/opus_encoder.rs` (same point, after `bandwidth_int` -> `Bandwidth`
  mapping and mode adjustments).

Trace knobs:
- C:
  - `OPUS_TRACE_MODE=1`
  - `OPUS_TRACE_MODE_FRAME=<n>` (optional; select a single frame)
- Rust:
  - `OPUS_TRACE_MODE=1`
  - `OPUS_TRACE_MODE_FRAME=<n>` (optional)
  - `OPUS_TRACE_PCM=<path>`
  - `OPUS_TRACE_FRAMES=<n>` (optional, default 64)
  - Driver: `cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture`

Printed fields:
- `mode`, `prev_mode`, `equiv_rate`, `bandwidth`, `stream_channels`,
  `voice_ratio`, `is_silence`, and `analysis_info` (valid/bandwidth/activity/
  tonality/tonality_slope/noisiness/music_prob/activity_probability/max_pitch_ratio).

Goal:
- Verify whether Rust/C pick the same mode on the first payload mismatch frame
  (frame 12) and the first TOC mismatch (frame 47).
- If modes differ, that likely explains the `ec_tell` delta and the early CELT
  allocation mismatch.

Run (frame 12):
- C: `OPUS_TRACE_MODE=1 OPUS_TRACE_MODE_FRAME=12 ctests/build/opus_packet_encode ...`
- Rust: `OPUS_TRACE_MODE=1 OPUS_TRACE_MODE_FRAME=12 OPUS_TRACE_PCM=... \
  OPUS_TRACE_FRAMES=64 cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture`

Findings (frame 12):
- Mode selection matches: `mode=1002 (CELT_ONLY)`, `prev_mode=1002`,
  `equiv_rate=63360`, `bandwidth=1105 (FULLBAND)`, `stream_channels=2`.
- Analysis fields still diverge (e.g. `tonality_slope`, `max_pitch_ratio`),
  which explains why downstream analysis-driven decisions can still drift.

Interpretation:
- Mode choice is *not* the source of the early allocation mismatch.
- Remaining divergence likely still originates inside analysis/tonality, even
  though high-level mode is aligned.

## Max Pitch Ratio Alignment (New)

Added a dedicated trace for `analysis_info.max_pitch_ratio` to follow the
`below_max_pitch/above_max_pitch` accumulation and the high-band energy term.

Trace knobs:
- C:
  - `ANALYSIS_TRACE_PITCH_RATIO=1`
  - `ANALYSIS_TRACE_PITCH_RATIO_FRAME=<n>`
  - `ANALYSIS_TRACE_PITCH_RATIO_BANDS=<list|all>` (optional)
  - `ANALYSIS_TRACE_PITCH_RATIO_BITS=1` (optional)
- Rust:
  - same env vars, using `analysis_compare_output` as the driver

Findings (frame 12, before fix):
- C: `hp_ener=6.704e-08`, `e_high=1.862e-11`, `ratio=4.326e-2`.
- Rust: `hp_ener=0`, `e_high=0`, `ratio=1.0`.
- Root cause: Rust used `tonal.hp_ener_accum` **after** it was reset for the
  next window, and also overwrote the accumulator on early return.

Fix (Rust):
- `src/analysis.rs`: keep `hp_ener` as the accumulated value for the *current*
  analysis window (like C), and do not overwrite `hp_ener_accum` on early
  return. High-band energy now uses the captured `hp_ener`.

Results (frame 12, after fix):
- `hp_ener` and `e_high` are now non-zero and close to C.
- `max_pitch_ratio` now aligns numerically: C `4.326478392e-02`,
  Rust `4.326480627e-02` (tiny ULP-level differences remain in band energy
  sums, likely due to minor float rounding differences).

## HP Resampler Trace (New)

Added a per-sample trace for the high-pass downsampler to confirm the
`downmix_and_resample` path is aligned.

Trace knobs:
- C: `ANALYSIS_TRACE_HP_RESAMPLER=1`
  - `ANALYSIS_TRACE_HP_RESAMPLER_FRAME=<n>`
  - `ANALYSIS_TRACE_HP_RESAMPLER_CALL=<n>`
  - `ANALYSIS_TRACE_HP_RESAMPLER_SAMPLES=all`
  - `ANALYSIS_TRACE_HP_RESAMPLER_BITS=1`
- Rust: same env vars, using `analysis_compare_output` as the driver.

Findings (frame 12, call 0, all samples):
- `analysis_hp_resampler` values (`out32_hp`, `S[2]`) and their bits match
  exactly for every sample.
- Only formatting differences remain (C prints `e+00`, Rust prints `e0`).

Interpretation: `silk_resampler_down2_hp` output is bit-identical between C
and Rust, so the remaining divergence is not coming from the HP resampler or
its state.

## Tonality Bin Energy / FFT Stage 2 (New)

Re-ran per-bin tracing on the real `analysis.c` path for frame 12, band 5.

Findings:
- `analysis_tonality_bin` shows the first mismatch at bin 24:
  - `binE_bits`: C `0x282121fe` vs Rust `0x28212203`
  - `tonality_bits` match; `tE_term` differs only because `binE` differs.
- `analysis_fft_trace` (bins 24–27) shows:
  - `input_fft` values match
  - `output_fft` values diverge at 1–2 ULP
  - `bin_e_bits` mismatch aligns with the `binE_bits` mismatch above.
- `analysis_fft_stage_trace` shows the first divergence at **stage 2** for
  frame 12 (bins 25/26).
- Stage-2 per-butterfly trace (hex enabled, bin 24) shows:
  - `mul_in*` and `tw*` bits match exactly
  - first mismatch appears at `bfly[1].mul0.bits.i`
    (C `0xb8df1573` vs Rust `0xb8df1572`), and `mul1.bits.r`
  - mismatches then propagate through `scratch*` and `out*`.

Interpretation: the remaining divergence is inside the complex multiply
arithmetic of the stage-2 butterfly (likely FMA/contract or operation order),
not in input data, twiddle generation, or add/sub chains.

### FFT Complex Multiply Alignment (Update)

Changes (Rust):
- `src/celt/kiss_fft.rs`: `c_mul` now uses `mul_add` for both real/imag to
  mirror the fused multiply-add behavior seen on the C build.
- `src/celt/kiss_fft.rs`: FFT `scale` uses C’s static-mode literal values for
  `nfft` 60/120/240/480 (e.g. `0.002083333f` for 480) instead of `1.0/nfft`.

Findings after the change (frame 12, bins 24–27):
- Stage‑2 per‑butterfly trace now matches exactly (no `mul*`/`scratch*`
  bit mismatches).
- `analysis_tonality_bin` aligns for bins 24–26; only bin 27 remains at 1‑ULP:
  - `binE_bits`: C `0x28259199` vs Rust `0x28259198`
  - `tE_term_bits` differs accordingly.
- `analysis_fft_trace` shows `output_fft.i` at bin 27 still differs by ~1e‑10,
  which explains the remaining `binE` 1‑ULP delta.

Interpretation:
- The remaining mismatch likely comes from twiddle values (static table in
  `opus-c/celt/static_modes_float.h` vs Rust’s runtime sin/cos), not from the
  butterfly math anymore.
- A quick check against the C static twiddle table shows multiple 1‑ULP
  differences in the imaginary components vs runtime sin/cos, suggesting the
  static table’s decimal truncation is a likely source of the last bin‑level
  difference.

## Bin 33 binE Mismatch (Frame 12, Band 7)

Added finer bin_e component tracing (r^2/i^2/mirror + SCALE_ENER bits) in:
- `ctests/analysis_fft_trace.c` (bin_e_r2/mr2/i2/mi2/sum/scaled + bits)
- `src/analysis.rs` fft_trace (same fields + bits)

Findings (frame 12, bin 33):
- `output_fft` bits match exactly between C and Rust.
- `r2/mr2/i2/mi2` bits match exactly.
- The *trace* sum + scaled bits match (`bin_e_scaled_bits = 0x284f5ed2`).
- But `analysis_compare` reports:
  - C `binE_bits = 0x284f5ed1`
  - Rust `binE_bits = 0x284f5ed2` (1 ULP higher)

Repro with standalone float32 arithmetic:
- Plain f32 mul/add chain yields `0x284f5ed2` (Rust).
- FMA-style accumulation (`fmaf(r,r, fmaf(mr,mr, fmaf(i,i, mi*mi)))`)
  yields `0x284f5ed1` (matches C).

Interpretation:
- C appears to contract the sum-of-squares into FMA(s) (or keep extra precision
  before final rounding), producing a slightly smaller result than strict
  step-by-step f32 adds.
- The remaining binE mismatch is likely due to missing FMA contraction in Rust.

## BinE FMA Alignment (Rust Fix + Verification)

Changes (Rust):
- Added `bin_energy_sum()` in `src/analysis.rs` using `fmaf` (mul_add) to
  mirror C’s contracted sum-of-squares for `r^2 + mr^2 + i^2 + mi^2`.
- Replaced all binE paths (FFT trace, DC band, tonality bands, pitch ratio
  bands) to use `bin_energy_sum()`.
- Added regression test `bin_energy_sum_fma_bits_match_reference` to lock the
  reference bits for frame 12 bin 33.

Verification:
- `analysis_compare_output` (64 frames, `ehren-paper_lights-96.pcm`) now passes
  with no differences reported.

Impact on audible output:
- Re-ran `examples/trivial_example` with `ehren-paper_lights-96.pcm` input.
  The decoded output is audibly improved: the first few seconds carry a clear
  melody; later frames still diverge into noise. This suggests remaining
  mismatch still exists but the earlier analysis drift has been reduced.

## Max Pitch Ratio Alignment (Frame 12)

Goal: align `analysis_info.max_pitch_ratio` between C and Rust.

Work done:
- Traced pitch ratio inputs with `ANALYSIS_TRACE_PITCH_RATIO=1` (frame 12).
- Found remaining mismatches in `bandE_bits` and high-band `hp_ener` that
  propagated into `max_pitch_ratio`.
- Reworked `bin_energy_sum()` ordering to match C's floating expression
  evaluation (FMA chain with `mi^2` outermost, then `i^2`, then `r^2/mr^2`).
  This eliminated *all* `bandE_bits` mismatches.
- Found `hp_ener` mismatch caused by accumulation: C uses FMA-style accumulation
  (`hp_ener = fma(out32_hp, out32_hp, hp_ener)`), while Rust used `+=`.
  Switched to `mul_add_f32` accumulation in `silk_resampler_down2_hp`.

Result:
- `analysis_pitch_ratio` trace for frame 12 now matches exactly:
  - `bandE_bits` aligned for all bands
  - `high.hp_ener_bits`, `high.e_high_bits`, `above_bits`
    and final `ratio_bits` all aligned
- `below_total`, `above_total`, and `ratio` are identical between C/Rust.

Note:
- `opus_mode_trace_output` still shows `analysis_info` differences in Rust,
  but that appears to be due to **different downmix parameters** in
  `opus_encode` (`c2=-1` in Rust vs `c2=-2` in C). If we change Rust to use
  `c2=-2`, the mode trace should align as well. This is a separate decision.

## Opus Encode Downmix Alignment (c2 = -2)

Change (Rust):
- In `opus_encode`, changed the `run_analysis(...)` call to pass `c2=-2`
  (matching the C path), instead of `c2=-1`.

Result:
- `opus_mode_trace_output` for frame 12 now matches C (mode/bandwidth and
  analysis_info values align to trace precision).
- **Packet mismatch remains**: first payload mismatch is still frame 12,
  with lengths C=161 vs Rust=226 and first byte diff at offset 2.

Interpretation:
- Analysis/mode selection is no longer the source of the first mismatch.
- Remaining divergence is downstream in the CELT encoding path (allocation /
  quantization / range coding).

## Opus -> CELT Budget Trace (New, Frame 12)

Goal: verify whether the first payload mismatch is caused by different
Opus-side byte budgeting / range-coder state before entering CELT.

New trace knobs:
- C: `OPUS_TRACE_CELT_BUDGET=1` + optional `OPUS_TRACE_CELT_BUDGET_FRAME=<n>`
- Rust: same env vars, driven by `opus_mode_trace_output` test

Trace fields:
- `max_data_bytes`, `max_frame_bytes`, `max_payload_bytes`
- `frame_size`, `frame_rate`, `equiv_rate`, `bitrate_bps`
- `use_vbr`, `vbr_constraint`, `channels`, `stream_channels`
- `redundancy`, `celt_to_silk`, `to_celt`, `redundancy_bytes`
- `nb_compr_bytes`, `tell_pre`, `tell_frac_pre`, `tell_post`, `tell_frac_post`

Commands:
```
OPUS_TRACE_CELT_BUDGET=1 OPUS_TRACE_CELT_BUDGET_FRAME=12 \
  ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/opus_c_budget.opuspkt \
  > /tmp/opus_c_budget_trace.txt

OPUS_TRACE_CELT_BUDGET=1 OPUS_TRACE_CELT_BUDGET_FRAME=12 \
  OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=64 \
  cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_budget_trace.txt
```

Findings (frame 12):
- C and Rust match on all CELT budget inputs:
  - `mode=1002`, `frame_size=960`, `frame_rate=50`,
    `equiv_rate=63360`, `bitrate_bps=64000`,
    `use_vbr=1`, `vbr_constraint=1`
  - `nb_compr_bytes=1275`
  - `tell_pre=1`, `tell_frac_pre=8`, `tell_post=1`, `tell_frac_post=8`
- Only difference: `max_data_bytes` (C=1276 vs Rust=3828), but
  `max_frame_bytes=1276` and `max_payload_bytes=1275` match, so the
  CELT byte budget entering `celt_encode_with_ec` is aligned.

Conclusion:
- The mismatch is **not** caused by Opus-side byte budgeting or the
  pre-CELT range-coder state.
- Remaining divergence likely occurs **inside CELT** (e.g. CVBR /
  reservoir logic) after entry.

## CELT VBR Budget Trace (New, Frame 12)

Goal: isolate where constrained‑VBR/reservoir logic diverges inside CELT.

New trace knobs:
- C: `CELT_TRACE_VBR_BUDGET=1` + `CELT_TRACE_VBR_BUDGET_FRAME=<n>`
- Rust: same env vars, driven via `opus_mode_trace_output` (CELT encoder is invoked)

Trace stages:
- `pre_cvbr`: before constrained‑VBR limit
- `post_cvbr`: after constrained‑VBR limit (max_allowed)
- `post_target`: after VBR target computation / reservoir update / shrink

Commands:
```
CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=12 \
  ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/opus_c_vbr_budget.opuspkt \
  > /tmp/opus_c_vbr_budget_trace.txt

CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=12 \
  OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=64 \
  cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_vbr_budget_trace.txt
```

Findings (frame 12):
- `pre_cvbr` matches: `nb_compressed_bytes=1275`, `nb_available_bytes=1275`,
  `vbr_rate=10240`, `vbr_reservoir=0`, `tell_frac=8`.
- `post_cvbr` matches: `max_allowed=320`, `nb_compressed_bytes=320`,
  `nb_available_bytes=320`.
- **Divergence happens at `post_target`:**
  - C: `base_target=9440`, `target=7360`, `delta=-2881`,
    `nb_available_bytes=160`, `nb_compressed_bytes=160`,
    `tell_frac=1218`, `vbr_offset=87`, `vbr_drift=-87`, `vbr_reservoir=0`.
  - Rust: `base_target=9440`, `target=20032`, `delta=9785`,
    `nb_available_bytes=313`, `nb_compressed_bytes=313`,
    `tell_frac=1145`, `vbr_offset=-296`, `vbr_drift=296`, `vbr_reservoir=9792`.

Conclusion:
- Constrained‑VBR cap (`max_allowed`) matches; **the first divergence is in the
  VBR target computation** (inside the `vbr_rate>0` block), which then drives
  different reservoir/offset/drift and shrinks `nb_compressed_bytes` to 160
  in C vs 313 in Rust.

Next likely checkpoints:
- Compare inputs to `compute_vbr()` and the `target` adjustment path:
  `tf_estimate`, `tot_boost`, `encoder.stereo_saving`, `encoder.intensity`,
  `encoder.last_coded_bands`, `pitch_change`, `max_depth`,
  `surround_masking`, `temporal_vbr`, `tell_frac`.

## Artifacts Produced During Investigation (Untracked)

- `ehren-paper_lights-96.pcm`
- `ehren-paper_lights-96.wav`
- `ehren-paper_lights-96_out.wav`
- `rust_packets.opuspkt`
- `c_packets.opuspkt`
- `rust_to_c_out.pcm` / `rust_to_c_out.wav`
- `c_to_rust_out.pcm` / `c_to_rust_out.wav`
- `rust_packets.opuspkt` / `c_packets.opuspkt`

## Notes

- CMake configure logs mention:
  - `ERRORRuntime cpu capability detection needed for MAY_HAVE_NEON`
  This did not prevent building the ctests tools.

## 2026-01-22 — Tonality slope / fast_atan2f alignment

- Extended tonality trace to include slope_pre/slope_term and upstream bin raw values (x1/x2, atan1/atan2, angle/angle2, d_angle/d_angle2) with bits in both C (`opus-c/src/analysis.c`) and Rust (`src/analysis.rs`).
- Located first drift in frame 12: band 9, bin 48. `x2r/x2i` bits matched; `fast_atan2f(x2i,x2r)` differed 1‑ULP, propagating through `angle2 -> d_angle2 -> d2_angle2 -> mod2 -> avg_mod -> tonality -> tE_term -> band_tonality`.
- Added extra fast_atan2f trace terms (xy, num_term) in C (`opus-c/celt/mathops.h`) and Rust (`src/celt/math.rs`) to isolate mismatch.
- Root cause: `num_term = x2 + cA*y2` (x2>=y2 branch) computed with different rounding in Rust vs C.
- Fix: Rust `fast_atan2f` now computes `num_term` using the same FMA order as C: `num_term = mul_add_c_order(CA, y2, x2)` (and `mul_add_c_order(CA, x2, y2)` in the other branch). After this, `fast_atan2f` internal bits match for the bin 48 case and tonality bin traces for band 9 align.
- After the fix, the next first drift moved to band 10 `band_tonality_bits`, but per‑bin traces for band 10 show no bin‑level mismatches (binE/tonality/tE_term bits all align). Next likely source is stationarity/prev_band_tonality accumulation at the band level.

## 2026-01-22 — tE accumulation & activity trace rerun

- Updated Rust tE accumulation to use FMA (`accumulate_t_e` -> `fmaf(bin_e, tonality_clamped, acc)`), with regression test ensuring FMA is used and a comment explaining why (mirrors C compiler contraction).
- After switching to FMA, band 10 `tE/energy_ratio/band_tonality` bits now align (the prior 1‑ULP drift disappears).
- Reran `analysis_activity` trace for frame 12:
  - `features_bits` fully aligned (no remaining differences).
  - `layer0_bits` and `rnn_post_bits` still show multiple 1‑ULP differences, so the remaining divergence is inside `analysis_compute_dense` / `analysis_compute_gru` (MLP/GRU math), not in feature extraction.

## 2026-01-23 — CELT compute_vbr inputs alignment (frame 12)

Goal: identify which inputs to `compute_vbr()` diverge to explain the
`target/nb_compressed_bytes` mismatch observed in the CELT VBR budget trace.

Trace knobs:
- C: `CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=12`
- Rust: same env vars, via `opus_mode_trace_output`

Commands:
```
CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=12 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets.opuspkt \
  > /tmp/opus_c_vbr_inputs_trace.txt

CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=12 \
OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=64 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_vbr_inputs_trace.txt
```

Findings (frame 12):
- **Matched**: `tot_boost=0`, `tf_estimate=9.928242564e-01`,
  `stereo_saving=2.500000000e-01`, `intensity=15`, `last_coded_bands=1`,
  `pitch_change=0`, `surround_masking=0.0`, `temporal_vbr=-1.5`
- **Mismatched**:
  - `tell_frac`: C `1218`, Rust `1145`
  - `max_depth`: C `-1.048040390e-01`, Rust `5.802087784e+00`

Conclusion:
- The **first divergence in the compute_vbr chain** is already present in
  `tell_frac` and `max_depth`. Since these are direct inputs to
  `compute_vbr()`, the `target/nb_compressed_bytes` mismatch is expected.
- Next step: trace `max_depth` (from `dynalloc_analysis`) and the range‑coder
  `tell_frac` path to identify why they diverge.

### dynalloc_analysis max_depth trace (frame 12)

Added a max‑depth trace inside `dynalloc_analysis` (C/Rust) gated by
`CELT_TRACE_VBR_BUDGET*` to record the band/channel that produced `max_depth`
plus `band_log_e`/`noise_floor` values and bits.

Findings (frame 12):
- **Max band/channel**: both C and Rust pick **band 0, channel 0**.
- **noise_floor**: matches exactly (`-1.278250027e+01`, bits `0xc14c851f`).
- **band_log_e**: **diverges**:
  - C: `-1.288730431e+01` (bits `0xc14e3266`)
  - Rust: `-6.980412483e+00` (bits `0xc0df5f8a`)
- Resulting `depth`:
  - C: `-1.048040390e-01` (bits `0xbdd6a380`)
  - Rust: `5.802087784e+00` (bits `0x40b9aab4`)

Conclusion:
- `max_depth` mismatch is caused by **band_log_e** mismatch, not noise_floor.
- This points back to **band energy / MDCT input** divergence (already traced
  earlier via `compute_band_energies` and preemphasis/PCM paths).

## 2026-01-23 — band 0 band_log_e chain (amp2_log2 → band_e → MDCT input)

Added a focused band‑0 trace around the `amp2_log2` call (C + Rust) gated by
`CELT_TRACE_VBR_BUDGET*`. This logs `bandE` (input) and `bandLogE` (output)
for both channels at the same frame as the VBR budget trace.

Findings (frame 12, band 0):
- `bandE` already diverges (input to amp2_log2):
  - C: ch0 `1.143972483e-02` (bits `0x3c3b6daf`),
       ch1 `9.672008455e-03` (bits `0x3c1e7758`)
  - Rust: ch0 `6.863837838e-01` (bits `0x3f2fb6d9`),
         ch1 `5.803205967e-01` (bits `0x3f148fe4`)
- `bandLogE` diverges accordingly:
  - C: ch0 `-1.288730431e+01` (bits `0xc14e3266`),
       ch1 `-1.312946892e+01` (bits `0xc152124e`)
  - Rust: ch0 `-6.980412483e+00` (bits `0xc0df5f8a`),
         ch1 `-7.222578049e+00` (bits `0xc0e71f5c`)

Conclusion:
- The `band_log_e` mismatch is **entirely driven by `bandE`**, so the root
  cause is still **upstream of amp2_log2**, i.e., in **compute_band_energies /
  MDCT input** (consistent with earlier MDCT/prefilter traces).

### band 0 per‑bin energy trace (compute_band_energies input)

Added a per‑bin trace for band 0 in the **main MDCT path** (C/Rust), gated by
`CELT_TRACE_VBR_BUDGET*`, printing each bin’s `x`, `x^2`, and the running
sum/sqrt used by `compute_band_energies`.

Findings (frame 12, band 0):
- Both C and Rust have **8 bins** in band 0 (LM=??).
- Bins 0–6 are tiny (sub‑1e‑21 range) on both sides, with small ULP drift.
- **Bin 7 is the dominant contributor and diverges massively**:
  - C: `x=1.143972483e-02` (bits `0x3c3b6daf`)
  - Rust: `x=6.863837838e-01` (bits `0x3f2fb6d9`)
  - This single bin explains the bandE difference.

Conclusion:
- The earliest large divergence in `bandE` is **not from accumulation
  order**; it is a **single MDCT output bin** (band 0, bin 7) that differs by
  orders of magnitude.
- Next step: trace the **MDCT output bin 7** back to its **windowed time‑domain
  samples** (preemphasis / window / overlap‑add) to locate where that bin’s
  inputs diverge.

### MDCT bin 7 root cause (frame 12)

Used MDCT input/output traces (note: C requires `CELT_TRACE_MDCT=1` to enable
`celt_mdct_trace_set_frame`) and compared Rust/C for frame 12:

- MDCT input (`celt_mdct_in`) and windowed folded (`celt_mdct_win`) samples for
  block 7 (main) match within ~1e‑8.
- MDCT output (`celt_mdct`) still differed by **~60x** on non‑zero bins.

Example (main, ch0, idx7):
- C: `0.0114397248`
- Rust: `0.686383784`
- Ratio ≈ **60.0**, which equals **N/4** for this mode (N=240, N/4=60).

Conclusion:
- Rust MDCT forward was **missing the 4/N scale** that C applies via
  `opus_fft`’s `scale=1/nfft` (nfft=N/4).
- This explains the band0 `bandE` mismatch and the downstream `band_log_e /
  max_depth / VBR target` divergence.

Fix (Rust):
- `src/celt/mdct.rs::clt_mdct_forward` now multiplies the pre‑rotated spectrum
  by `1/n4` before FFT, matching C’s scaling.
- Added a comment explaining the 4/N scale, and a regression test
  `forward_matches_reference_with_stride` to cover stride>1.

After fix:
- `celt_mdct[12].main.ch0.idx7` now matches within ~4e‑9.

### VBR inputs after MDCT scale fix

Re‑ran `CELT_TRACE_VBR_BUDGET` after the MDCT scale fix:
- `tell_frac` now matches: C/Rust both `1218`.
- `max_depth` now matches within ~1e‑8 (`-1.048040390e-01` vs `-1.048030853e-01`).

Remaining mismatch (post_target stage):
- C: `target=7360`, `vbr_offset=87`, `vbr_drift=-87`, `tell=1218`
- Rust: `target=8832`, `vbr_offset=42`, `vbr_drift=-42`, `tell=1`

Interpretation:
- The MDCT scaling fix resolves the **bandE / max_depth / tell_frac** mismatch,
  but **compute_vbr output still diverges**, likely due to upstream analysis
  fields not captured in the current input trace (e.g., `analysis` contents).

## 2026-01-23 — compute_vbr internal trace & fixes

Added detailed `compute_vbr` step tracing (C/Rust) gated by
`CELT_TRACE_VBR_BUDGET*`, printing inputs and `target` after each step:
`after_activity`, `after_stereo`, `after_boost`, `after_tf`, `after_tonality`,
`after_surround`, `after_floor`, `after_constrained`, `after_temporal`,
`after_cap`.

Initial findings (before fixes):
- Rust `analysis_valid=0`, `activity=0`, `tonality=0` while C had valid analysis.
- Rust `after_tf` was **~2x larger** than C (factor‑2 scaling).

Fixes applied:
1) **MDCT scale already fixed** (restored 4/N), which aligned `max_depth`/inputs.
2) **compute_vbr tf scaling**: removed extra `*2` in Rust
   (`SHL32` is a no‑op in float C, so no doubling).
3) **CELT analysis injection in CELT‑only path**:
   `encoder.celt.encoder().analysis = encoder.analysis_info.clone();`
   (matches `CELT_SET_ANALYSIS` in C).

After fixes (frame 12):
- `analysis_valid` now matches (both 1).
- `activity/tonality` match exactly.
- All `compute_vbr` step targets match (within 1 LSB for `after_tonality`).
- `post_target` VBR budget now matches:
  - `target=7360`, `delta=-2881`, `vbr_offset=87`, `vbr_drift=-87`,
    `nb_compressed_bytes=160`, `nb_available_bytes=160`.
- The only remaining cosmetic trace mismatch is `tell` in the
  `post_target` dump (C uses `tell_frac` later in the function; Rust still logs
  the earlier `ec_tell` value). This does **not** affect allocation.

## 2026-01-23 — quant_all_bands per-band RC trace (frame 12)

Goal: identify which band first mutates the range-coder state differently.

Setup:
- Added `CELT_TRACE_RC_BAND` trace inside `quant_all_bands` (C/Rust).
  - Dumps RC state per band after it completes (`post_band`).
  - Env: `CELT_TRACE_RC=1`, `CELT_TRACE_RC_FRAME=12`, optional `CELT_TRACE_RC_BAND=<band>`.

Findings (frame 12):
- First RC buffer divergence appears at **band 12**.
  - `buf[36]`: C `0xa6`, Rust `0xa9`.
  - Bands 13..20 retain the same mismatch, so the divergence starts in band 12.
- Conclusion: the first payload mismatch is now localized to band 12 inside
  `quant_all_bands`.

Next step (in progress):
- Add RC dumps around the RDO path inside band 12 (`rdo_pre_round_down`,
  `rdo_post_round_down`, `rdo_post_restore`, `rdo_post_round_up`,
  `rdo_post_select`) to pinpoint whether the mismatch happens during the
  round-down encode, the restore, the round-up encode, or the final selection.

### RDO sub-steps in band 12 (quant_all_bands)

Added RC dumps at:
- `rdo_pre_round_down`
- `rdo_post_round_down`
- `rdo_post_restore`
- `rdo_post_round_up`
- `rdo_post_select`

Findings (frame 12, band 12):
- **Mismatch already present at `rdo_pre_round_down`**, before any encode in
  the RDO block. Fields differ (`nbits_total`, `nend_bits`, `end_window`,
  `tell`), so the divergence existed at band entry.
- `rdo_post_round_down` shows the same buf mismatch as `post_band`:
  `buf[36]` C=0xa6 vs Rust=0xa9 (and `end_buf[0]` C=0xa7 vs Rust=0xa9).
- `rdo_post_round_up` diverges further (different `offs/end_offs/nbits_total`).

Interpretation: the first visible **buf** mismatch happens within band 12, but
**RC state already differs before the RDO block begins**, implying an earlier
mismatch (likely end-window state) that only surfaces in the byte buffer by band 12.

### RC state before quant_all_bands (frame 12)

Added RC dump right after `quant_fine_energy` and before `quant_all_bands`
(`pre_quant_all_bands` stage in C/Rust).

Findings (frame 12):
- `pre_quant_all_bands` already differs:
  - `nbits_total`: C 263 vs Rust 261
  - `nend_bits`: C 14 vs Rust 12
  - `end_window`: C 0x000039ca vs Rust 0x000009ca
  - `tell`: C 237 vs Rust 235
  - `tell_frac`: C 1893 vs Rust 1877

Interpretation:
- The RC divergence is present **before** `quant_all_bands` starts, i.e. either
  earlier in the encoder pipeline or inside `quant_fine_energy` itself.
- Next step: add RC dump just **before** `quant_fine_energy` (or inside it) to
  pinpoint whether `quant_fine_energy` introduces the divergence.

### RC state before quant_fine_energy (frame 12)

Added RC dump immediately before `quant_fine_energy` (`pre_quant_fine_energy`).

Findings (frame 12):
- `pre_quant_fine_energy` **matches exactly** between C and Rust.
- `pre_quant_all_bands` (post-quant_fine_energy) **diverges**.

Conclusion:
- The first RC divergence is introduced **inside `quant_fine_energy`**.
  Next step: add per-band trace inside `quant_fine_energy` to compare `q2`,
  `offset`, and updated `error/oldEBands` values (and the exact `enc_bits` output)
  to pinpoint the first band where the encoded fine bits differ.

### quant_fine_energy per-band trace (frame 12)

New trace env (C/Rust):
- `CELT_TRACE_FINE_ENERGY=1`
- `CELT_TRACE_FINE_ENERGY_FRAME=12`
- `CELT_TRACE_FINE_ENERGY_BAND=<band>` (optional)
- `CELT_TRACE_FINE_ENERGY_BITS=1` for bit dumps

Findings (frame 12):
- First mismatch appears at **band 0, channel 0** before any fine bits are
  encoded:
  - `error_before_bits`: C `0x3de6cd00`, Rust `0x3de6cd80`
  - `old_before_bits` matches (`0xc1500000`).
- `q2`, `offset`, and `tell_before/after` are the same, but because
  `error_before` is already different, `error_after_bits` also diverges.

Conclusion:
- The RC divergence is not caused by `quant_fine_energy` itself; the input
  `error` already differs at band 0. The next step is to trace `quant_coarse_energy`
  (or earlier `band_logE` / `log2_amp` path) to find where `error` diverges.

### quant_coarse_energy per-band trace (frame 12)

Added `CELT_TRACE_COARSE_ENERGY*` traces inside `quant_coarse_energy_impl`.

Findings (frame 12):
- First mismatch is in the **inter pass, band 0, channel 0** at the input
  energy `x`:
  - C `x_bits=0xc14e3266` (x=-12.88730431)
  - Rust `x_bits=0xc14e3265` (x=-12.88730335)
- This 1‑ULP `x` mismatch propagates to `f` and `error_before/error_after`,
  explaining the `error_before` mismatch seen in `quant_fine_energy`.
- `old_before/oldE`, `qi`, `q`, `tmp`, `prev` and `tell` match at this point.

Conclusion:
- The root divergence feeding the RC stream is now localized to `eBands` (the
  log‑energy input to `quant_coarse_energy`) for **band 0**. Next step is to
  trace where `eBands` comes from: `amp2_log2` / `log2_amp` path (band energy
  -> log energy) or even earlier in band energy computation.

### amp2_log2 trace (frame 12)

Trace env:
- `CELT_TRACE_AMP2LOG2=1`
- `CELT_TRACE_AMP2LOG2_FRAME=12`
- `CELT_TRACE_AMP2LOG2_BITS=1`

Findings (frame 12):
- `bandE` inputs to `amp2Log2/amp2_log2` already differ across bands.
  Example band 0:
  - C `energy_bits=0x3b828ddd` (3.9841966e-3)
  - Rust `energy_bits=0x3b5c9f07` (3.3664124e-3)
- `mean_bits` match, so the mismatch is purely in `bandE` (linear energy),
  which then yields `log_bits` differences for every band.

Conclusion:
- The divergence is **upstream of amp2_log2**, in `compute_band_energies` or its
  inputs (`freq` / MDCT output). Next step: trace `compute_band_energies` sums
  for band 0 (per-bin squares and sum) to see where `bandE` first diverges.

### compute_band_energies per-bin trace (frame 12, band 0)

Trace env:
- `CELT_TRACE_BAND_ENERGY_DETAIL=1`
- `CELT_TRACE_BAND_ENERGY_DETAIL_FRAME=12`
- `CELT_TRACE_BAND_ENERGY_DETAIL_BAND=0`
- `CELT_TRACE_BAND_ENERGY_DETAIL_BITS=1`

Findings (frame 12, band 0):
- **First per-bin mismatch at bin 0** (channel 0):
  - C `x_bits=0x3ac5c52c` (1.5088669e-3)
  - Rust `x_bits=0x3ac5c531` (1.5088675e-3)
  - `x2_bits` differ accordingly.
- This means the divergence originates **before** energy accumulation, in the
  frequency-domain input `freq` / MDCT output feeding `compute_band_energies`.

Conclusion:
- Next step: trace the MDCT output (freq) for band 0/bin 0 on the same frame,
  and then walk upstream (prefilter/output PCM buffer) to find the first
  divergence in the time-domain input.

### MDCT window detail (frame 12, main, i=0)

Using `CELT_TRACE_MDCT_WINDOW_DETAIL=1` (index 0), compared the first windowed
fold sample in the MDCT pipeline.

Findings:
- MDCT input (`celt_mdct_in`) matches for the full window length.
- The first windowed output (`celt_mdct_win` idx 0) differs by 1 ULP.
- Window detail shows **input samples match**, but **window coefficients differ**:
  - `w1_bits`: C `0x3f36dee7`, Rust `0x3f36dee8`
  - `w2_bits`: C `0x3f332619`, Rust `0x3f33261a`
  - This yields `re_bits`/`im_bits` differences and propagates to MDCT output.

Conclusion:
- The root divergence is in `compute_mdct_window` (sine window generation),
  likely due to libm `sinf` vs system libm `sinf` differences. Next decision:
  switch Rust to call the system `sinf` (FFI) or embed a precomputed window
  table derived from C.

### MDCT window static table (48k/960)

Change:
- Added `src/celt/window_48000_960.rs` with `WINDOW_120` copied from
  `opus-c/celt/static_modes_float.h` (`dump_modes.c 48000 960`).
- `compute_mdct_window` now returns `WINDOW_120` when `overlap == 120`.

Re-test (frame 12, mdct2, i=0):
- C and Rust now **match** on window coefficients:
  - `w1_bits`: `0x3f36dee7`
  - `w2_bits`: `0x3f332619`
- `a/b/c/d` input bits also match.
- `re_bits`/`im_bits` still differ by ~1 ULP:
  - C `re_bits=0x99a95727`, Rust `re_bits=0x99a95728`
  - C `im_bits=0x965e0f76`, Rust `im_bits=0x965e0f80`

Conclusion:
- Window coefficient mismatch is resolved. Remaining MDCT divergence is now
  in the multiply/add chain (or later FFT scaling), not in the window table.

### MDCT fold add order / FMA alignment

Trace env:
- `CELT_TRACE_MDCT=1`
- `CELT_TRACE_MDCT_WINDOW=1`
- `CELT_TRACE_MDCT_WINDOW_DETAIL=1`
- `CELT_TRACE_MDCT_WINDOW_DETAIL_INDEX=0`
- `CELT_TRACE_MDCT_FRAME=12`
- `CELT_TRACE_MDCT_BITS=1`

Findings (frame 12, mdct2, i=0):
- `mul_aw2/mul_bw1/mul_cw1/mul_dw2` bits match C and Rust.
- C **actual** folded output `celt_mdct_win idx_bits[0]=0x99a95727` and
  `idx_bits[1]=0x965e0f76`.
- Rust **non‑FMA** sum (`re = mul_aw2 + mul_bw1`, `im = mul_cw1 - mul_dw2`)
  yields `re_bits=0x99a95728` / `im_bits=0x965e0f80` (1 ULP off).
- Rust **FMA** (`re_fma = a.mul_add(w2, b*w1)`,
  `im_fma = c.mul_add(w1, -(d*w2))`) matches C actual bits:
  `re_fma_bits=0x99a95727`, `im_fma_bits=0x965e0f76`.

Conclusion:
- The 1‑ULP drift comes from the add path in MDCT fold using FMA on the C side.
- Updated Rust `fold_input` to use `mul_add` in the MDCT fold loops so
  `celt_mdct_win` now matches C for frame 12 (mdct2 idx 0/1).

### Packet mismatch (frame 13) -> coarse energy -> mdct2

Packet compare:
- First payload mismatch at frame 13 (0-based), byte offset 37.
- C packet len 161, Rust packet len 159.

RC trace:
- `celt_rc[13].stage=pre_quant_fine_energy` already diverges.
- `celt_rc_band` shows **pre_band** mismatch starting at band 1 (band 0 not used).

Coarse energy trace (bits, frame 13):
- First mismatch at **band 3**, pass `intra`, channel 0:
  - `x_bits` C `0xc08f8c5d`, Rust `0xc08f8c5c`.

Band energy detail (frame 13, band 3):
- First per-bin mismatch at **bin 0**:
  - `x_bits` C `0x16792459` vs Rust `0x1679245b`.

MDCT output (frame 13):
- `main` MDCT freq idx 3 (ch0) **matches**.
- `mdct2` MDCT freq idx 3 (ch0) **mismatch**:
  - C `0x3f169bc3`
  - Rust `0x3f169bc5` (after static MDCT twiddles + static FFT twiddles + FMA)

Conclusion:
- The remaining divergence appears in **mdct2** (long-block MDCT) output, not main.
- Likely source is inside MDCT pre-rotate/FFT/post-rotate chain for shift=0.
- Next step: add a targeted pre-rotate/FFT stage trace for mdct2 to locate first drift.

### mdct2 pre-rotate vs FFT stage trace (frame 13)

Trace env:
- `CELT_TRACE_MDCT=1`
- `CELT_TRACE_MDCT_STAGE=1`
- `CELT_TRACE_MDCT_FRAME=13`
- `CELT_TRACE_MDCT_BITS=1`
- `CELT_TRACE_MDCT_START=0`
- `CELT_TRACE_MDCT_COUNT=8`

Result (mdct2, ch0/block0, idx 0..):
- **Pre-rotate outputs match** (no diff in `pre_rotate` values/bits).
- **FFT output diverges first** at idx 0 imag component:
  - C: `-1.980147138e-02`
  - Rust: `-1.980148442e-02`

Conclusion:
- The remaining drift is inside the N/4 FFT implementation (post pre-rotate).
- Next step: add per-stage FFT trace inside `mini_kfft` (radix stages) to find
  the first butterfly that diverges.

## 2026-01-24 — Packet compare rerun (ehren-paper_lights-96.pcm)

Commands:
```
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets.opuspkt
cargo run --example opus_packet_tool -- encode ehren-paper_lights-96.pcm /tmp/rust_packets.opuspkt
```

Findings:
- First payload mismatch is **frame 12** (0-based).
- First byte offset: **37**.
- Packet lengths: **C 161**, **Rust 159**.

Conclusion:
- The current earliest divergence is still in frame 12. Continue tracing from
  the CELT path (RC/coarse energy/quant bands) for that frame.

## 2026-01-24 — Frame 12 CELT trace (RC -> fine/coarse energy -> band energy)

RC trace (frame 12):
- `celt_rc[12].stage=pre_quant_fine_energy` **matches**.
- First RC divergence appears at `pre_quant_all_bands`:
  - `nbits_total`: C `263`, Rust `261`
  - `nend_bits`: C `14`, Rust `12`
  - `tell_frac`: C `1893`, Rust `1877`

Fine energy trace (frame 12, bits):
- First mismatch at **band 1, channel 1**:
  - `error_before_bits`: C `0x3eac0900`, Rust `0x3eac0920`.

Coarse energy trace (frame 12, bits):
- First mismatch at **band 1, channel 1** (intra pass):
  - `x_bits`: C `0xc15b6cd8`, Rust `0xc15b6cd7`.

Band energy detail (frame 12, band 1, bits):
- First mismatch at **channel 0, bin 0**:
  - `x_bits`: C `0x3a98b6d1`, Rust `0x3a98b6d7`.

Interpretation:
- The earliest numeric drift for frame 12 is now at `compute_band_energies`
  for **band 1** (global bin indices 8–15 for `lm=3`).
- Next step: trace the MDCT output for **global bin 8** (both `mdct2` and
  `main` tags) to identify which MDCT path diverges first.

### MDCT output trace (frame 12, idx 8)

Trace env:
- `CELT_TRACE_MDCT=1`
- `CELT_TRACE_MDCT_FRAME=12`
- `CELT_TRACE_MDCT_BITS=1`
- `CELT_TRACE_MDCT_START=8`
- `CELT_TRACE_MDCT_COUNT=1`

Findings:
- `main` MDCT output **matches** for both channels at idx 8.
- `mdct2` output **mismatches** at idx 8:
  - ch0: C `0x3a98b6d1`, Rust `0x3a98b6d7`
  - ch1: C `0x3a7f111c`, Rust `0x3a7f1126`

Conclusion:
- The drift for frame 12 band 1 originates in the **mdct2** path.
- Next step: compare `mdct2` **post-rotate** math (t0/t1, fp->r/i, yr/yi) to
  identify where the 1‑ULP difference is introduced.

### mdct2 stage trace (frame 12, first 16 indices)

Trace env:
- `CELT_TRACE_MDCT_STAGE=1`
- `CELT_TRACE_MDCT_FRAME=12`
- `CELT_TRACE_MDCT_BITS=1`
- `CELT_TRACE_MDCT_START=0`
- `CELT_TRACE_MDCT_COUNT=16`

Findings:
- `mdct2` **pre-rotate and FFT outputs match** for indices 0–15 (bits aligned).

Conclusion:
- The mismatch appears **after** the FFT stage, i.e., in the **post-rotate**
  output path for mdct2.

## 2026-01-24 — mdct2 post-rotate trace (frame 12, idx 4 -> output idx 8)

Note:
- C-side MDCT stage tracing requires **both** `CELT_TRACE_MDCT=1` and
  `CELT_TRACE_MDCT_STAGE=1`. `CELT_TRACE_MDCT_STAGE` alone does not enable
  frame indexing in the C encoder path.

Trace env:
- `CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_STAGE=1`
- `CELT_TRACE_MDCT_FRAME=12`
- `CELT_TRACE_MDCT_BITS=1`
- `CELT_TRACE_MDCT_START=4`
- `CELT_TRACE_MDCT_COUNT=1`

Findings (mdct2, call 0/ch0 and call 1/ch1):
- `t0/t1` **match** between C and Rust.
- `fp.r/fp.i` **mismatch** at idx 4:
  - ch0: C `fp.r=0xba9746f2`, Rust `0xba9746f8`
  - ch0: C `fp.i=0xba56f3b9`, Rust `0xba56f3c2`
- `yr/yi` **mismatch** accordingly (matches the `mdct2` output drift).

Pre-rotate trace (same frame/index) with `CELT_TRACE_MDCT_STAGE`:
- `pre_rotate.idx_bits[4].r`: C `0x9532559e`, Rust `0x953255a1`
- `pre_rotate.idx_bits[4].i`: C `0x93e51f7a`, Rust `0x93e51f7e`

Folded input (`celt_mdct_win`) for mdct2:
- `idx_bits[8]` and `idx_bits[9]` **match** between C and Rust.

Conclusion:
- The first drift is **inside mdct2 pre-rotate arithmetic** (even though the
  folded inputs and twiddles match). This strongly suggests a **floating‑point
  contraction/rounding** difference (e.g., FMA/FTZ) in the `re*t0 - im*t1`
  / `im*t0 + re*t1` path.
- Next step: test pre-rotate under `-fp-contract=off` (or force a specific
  mul/add order) to see if the bits align.

### Rust `-fp-contract=off` check (frame 12, mdct2 idx 4)

Command:
```
RUSTFLAGS="-C llvm-args=-fp-contract=off" \
CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_STAGE=1 CELT_TRACE_MDCT_FRAME=12 \
CELT_TRACE_MDCT_BITS=1 CELT_TRACE_MDCT_START=4 CELT_TRACE_MDCT_COUNT=1 \
OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=13 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture
```

Result:
- `fp.r/fp.i` and `yr/yi` **still differ** from C at idx 4.
- Therefore, **disabling FP contraction in Rust alone does not align** the
  pre-rotate outputs with C.

Next step:
- Either force **FMA** in Rust pre‑rotate (match C if it contracts), or
  rebuild C with `-ffp-contract=off` to see if C matches the non‑FMA Rust path.

## 2026-01-24 — mdct2 N/4 FFT stage trace (frame 12)

Goal: find the first FFT stage/butterfly that diverges for mdct2.

Changes (Rust trace harness):
- `src/celt/mini_kfft.rs` trace path now supports **any nfft**, not just 480.
  - Uses `plan.factors` to build the stage schedule.
  - Added a `compute_bitrev_table` helper (matching C's recursive table
    builder) to generate bitrev indices for non-480 sizes.
- `trace_bfly5` output order fixed to match C: `out0 = scratch0 + (scratch7 +
  scratch8)` to prevent trace-only skew.

Trace knobs used (C/Rust):
```
CELT_TRACE_MDCT=1
CELT_TRACE_KFFT_STAGE=1
CELT_TRACE_KFFT_FRAME=12
CELT_TRACE_KFFT_BITS=1
CELT_TRACE_KFFT_START=0
CELT_TRACE_KFFT_COUNT=480
```
Plus detail for a single u:
```
CELT_TRACE_KFFT_STAGE_INDEX=4
CELT_TRACE_KFFT_DETAIL=1
CELT_TRACE_KFFT_START=11
CELT_TRACE_KFFT_COUNT=1
```

Findings (frame 12, mdct2, ch0/block0):
- **Stages 0..3 match exactly** (bit-level).
- **First mismatch appears in stage 4** (radix-5, m=96, fstride=1):
  - First differing output: `stage[4].idx[107].i`
    - C: `0xba9b072f`
    - Rust: `0xba9b0730`
  - `idx 107 = u + 96` => **u=11**.

Detail (stage 4, u=11):
- First mismatching term is `sum78_ya_yb` (i.e., `scratch7*ya + scratch8*yb`):
  - C: `sum78_ya_yb_bits.r=0x36865b4d`, `.i=0xb969d8db`
  - Rust: `sum78_ya_yb_bits.r=0x36865b50`, `.i=0xb969d8dc`
- This 1-ULP drift propagates to `scratch5`, then `out1.i` / `out4.r`, which
  matches the first differing stage output above.

Conclusion:
- The earliest mdct2 FFT divergence is inside the **stage-4 radix-5** add/mul
  chain, specifically the `sum78_ya_yb` accumulation. This points to
  floating-point contraction/rounding differences in that term rather than
  earlier stages or twiddle inputs.

## 2026-01-24 — radix-5 sum78_ya_yb FMA alignment (frame 12)

Change (Rust):
- In `kf_bfly5` and trace `trace_bfly5`, compute `sum78_ya_yb` using FMA:
  `fused_mul_add(scratch7.{r,i}, ya.r, scratch8.{r,i} * yb.r)`, matching the
  C-side contraction behavior for the sum.

Trace re-run (same knobs as above, full stage dump):
- `CELT_TRACE_KFFT_STAGE=1 CELT_TRACE_KFFT_FRAME=12 CELT_TRACE_KFFT_BITS=1
   CELT_TRACE_KFFT_START=0 CELT_TRACE_KFFT_COUNT=480`

Result:
- **All mdct2 KFFT stage outputs now match** between C/Rust for frame 12
  (no `idx_bits` mismatches across stage 0..4).

Interpretation:
- The stage-4 radix‑5 mismatch was due to `sum78_ya_yb` rounding. With FMA,
  the mdct2 FFT path is now bit-aligned at the stage outputs for frame 12.

### Packet compare re-run (after radix-5 FMA fix)

Commands:
```
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets_new.opuspkt
cargo run --example opus_packet_tool -- encode ehren-paper_lights-96.pcm /tmp/rust_packets_new.opuspkt
```

Result:
- Frames: C 11405, Rust 11405.
- **First payload mismatch remains at frame 12** (0-based).
- TOC still matches at that frame (no TOC mismatch at the first payload mismatch).
- Packet lengths: C 161 vs Rust 159.
- First byte offset: 37.

Conclusion:
- mdct2 FFT stage alignment alone is **not sufficient** to eliminate the frame‑12
  payload mismatch. Continue tracing from the CELT quantization / RC path.

## 2026-01-24 — mdct2 post-rotate recheck after scale fix (frame 12, idx 4)

Trace commands (opus encoder path, not analysis_compare):
```
CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_STAGE=1 CELT_TRACE_MDCT_FRAME=12 \
CELT_TRACE_MDCT_BITS=1 CELT_TRACE_MDCT_START=4 CELT_TRACE_MDCT_COUNT=1 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets_tmp.opuspkt

CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_STAGE=1 CELT_TRACE_MDCT_FRAME=12 \
CELT_TRACE_MDCT_BITS=1 CELT_TRACE_MDCT_START=4 CELT_TRACE_MDCT_COUNT=1 \
OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=64 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture
```

Findings (mdct2 call/ch0, idx 4 -> output idx 8):
- **pre_rotate terms now match** (t0/t1/re/im/mul*/yr/yi and bits align).
- **post_rotate still mismatches** at `fp.r/fp.i` and `yr/yi`:
  - C `fp.r=0xba9746f2`, Rust `0xba9746f6`
  - C `fp.i=0xba56f3b9`, Rust `0xba56f3bc`
  - C `yr=0x3a98b6d1`, Rust `0x3a98b6d5`
- `t0/t1` match, so the drift must originate from the **FFT output** fed into
  post‑rotate (not from twiddles or post‑rotate math order).

Conclusion:
- After aligning `mdct2` scale, the **first drift moved to the FFT output**.
- Next step: enable `CELT_TRACE_KFFT_STAGE` (or per‑stage FFT trace) for mdct2
  and locate the first stage/index where `freq[i]` diverges.

## 2026-01-24 — mdct2 KFFT stage trace (frame 12, idx 4)

Trace env (C/Rust):
```
CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_FRAME=12 \
CELT_TRACE_KFFT_STAGE=1 CELT_TRACE_KFFT_FRAME=12 \
CELT_TRACE_KFFT_BITS=1 CELT_TRACE_KFFT_START=4 CELT_TRACE_KFFT_COUNT=1
```

Results (mdct2, ch0: C call[1] vs Rust call[0]):
- **Stage 0–3 match** (r/i bits identical).
- **Stage 4 differs** (first mismatch):
  - C `stage[4].idx_bits[4].r=0xba9746f2`
  - Rust `stage[4].idx_bits[4].r=0xba9746f1`
  - `stage[4].idx_bits[4].i` matches (`0xba56f3b9`).

Conclusion:
- The first FFT divergence is at the **final stage (stage 4)** for idx 4.

## 2026-01-24 — stage 4 (radix‑5) per‑butterfly detail (u=4)

Trace env:
```
CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_FRAME=12 \
CELT_TRACE_KFFT_STAGE=1 CELT_TRACE_KFFT_FRAME=12 \
CELT_TRACE_KFFT_STAGE_INDEX=4 CELT_TRACE_KFFT_DETAIL=1 \
CELT_TRACE_KFFT_BITS=1 CELT_TRACE_KFFT_START=4 CELT_TRACE_KFFT_COUNT=1
```

Findings (mdct2, ch0, stage 4, u=4):
- Twiddles (`tw0..tw3`) and constants (`ya/yb`) **match**.
- **Inputs** to radix‑5 butterfly differ:
  - `in1` (idx 100) **i** bits differ.
  - `in2` (idx 196) **r** bits differ.
  - `in3` (idx 292) **i** bits differ.
  - `in4` (idx 388) **matches**.
- As a result, `scratch1..4` and all downstream scratch/out values differ.

Conclusion:
- Stage‑4 math itself is not the first divergence; the **inputs into stage 4**
  already differ. These inputs are stage‑3 outputs at indices 100/196/292.
- Next step: trace **stage 3** (radix‑3) for indices 100/196/292 to locate the
  earliest mismatch and then drill into `kf_bfly3` for that u/group.

## 2026-01-24 — stage 3 (radix‑3) detail at idx 100

Trace env:
```
CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_FRAME=12 \
CELT_TRACE_KFFT_STAGE=1 CELT_TRACE_KFFT_FRAME=12 \
CELT_TRACE_KFFT_STAGE_INDEX=3 CELT_TRACE_KFFT_DETAIL=1 \
CELT_TRACE_KFFT_BITS=1 CELT_TRACE_KFFT_START=100 CELT_TRACE_KFFT_COUNT=1
```

Findings (mdct2, ch0, stage 3, idx 100):
- `in0.i` **already differs** (C `0x397a2948`, Rust `0x397a2945`).
- `scratch2.r` differs; `scratch0` / `scratch0_scaled` differ accordingly.
- `out0/out1/out2` all differ (propagating the input mismatch).

Conclusion:
- Stage‑3 arithmetic is **not the first drift**; the **input to stage 3**
  at idx 100 is already off. The earliest mismatch is **stage 2 output**
  feeding idx 100.
- Next step: add radix‑4 (stage 2) per‑butterfly detail for **idx 100** to
  find the first mismatch inside `kf_bfly4`.

## 2026-01-24 — stage 2 (radix‑4) detail at idx 100

Trace env:
```
CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_FRAME=12 \
CELT_TRACE_KFFT_STAGE=1 CELT_TRACE_KFFT_FRAME=12 \
CELT_TRACE_KFFT_STAGE_INDEX=2 CELT_TRACE_KFFT_DETAIL=1 \
CELT_TRACE_KFFT_BITS=1 CELT_TRACE_KFFT_START=100 CELT_TRACE_KFFT_COUNT=1
```

Findings (mdct2, ch0, stage 2, idx 100):
- Inputs and twiddles **match** (`in0..in3`, `tw1..tw3` all identical).
- First mismatches appear in the **radix‑4 internal products**:
  - `scratch0.i` differs by 1 ULP.
  - `scratch2.i` differs by 1 ULP.
- `scratch3/scratch4` and `out0/out1/out2/out3` differ accordingly.

Conclusion:
- The earliest drift is **inside stage‑2 radix‑4 math**, specifically the
  complex multiplies feeding `scratch0`/`scratch2`. Inputs/twiddles are
  identical, so this looks like **floating‑point contraction/rounding** in
  `c_mul` (or the compiler emitting FMA on one side).
- Next step: compare `c_mul` behavior by forcing Rust to use `fmaf`
  (or compile C with `-ffp-contract=off`) to see if the 1‑ULP difference
  disappears.

## 2026-01-24 — force FMA in Rust `c_mul` (stage‑2 alignment)

Change:
- Rust `c_mul` now uses `mul_add` (fused) for r/i.

Result:
- **Stage 2 idx 100 now aligns** (scratch0/scratch2/out* bits match).
- **Stage 3 idx 100 aligns** as a consequence.
- **Stage 4 still differs**, but inputs and scratch1..4 now match; first
  mismatch is in the stage‑4 arithmetic (`scratch5/scratch6/scratch12` etc.).

Conclusion:
- The original drift was caused by **non‑fused vs fused c_mul** in stage‑2.
- Remaining mismatch is now isolated to **radix‑5 arithmetic order** in stage‑4
  (likely sum ordering in scratch5/11 or scratch6/12 paths).

## 2026-01-24 — stage 4 term products (radix‑5)

Added per‑term trace for the individual products feeding `scratch5/11/12`
(`term7_ya`, `term8_yb`, `term7_yb`, `term8_ya`, `term10_yai`, `term9_ybi`,
`term10_ybi`, `term9_yai`).

Result (mdct2, ch0, stage 4, u=4):
- **All term products match** between C/Rust.
- `scratch7/8/9/10/11` still match.
- **Mismatch remains** in `scratch5.r`, `scratch6.i`, `scratch12.r` and
  propagated `out*`.

Conclusion:
- The first remaining drift is **not in the multiplies**, but in the **add/sub
  order/rounding** when combining those terms (e.g., how `scratch5` and
  `scratch12` are formed from the term sums).

## 2026-01-24 — stage 4 sum‑order explicit (no FMA in scratch5/11/12)

Change:
- Rust `kf_bfly5` now uses explicit `a + (b + c)` (no `mul_add`) for
  `scratch5/11/12` and matches C’s parenthesized form.

Result:
- **No change**: `scratch5.r`, `scratch6.i`, `scratch12.r` still differ by 1‑ULP,
  and `out*` remain off.

Conclusion:
- Remaining mismatch is not explained by simple parenthesis differences; need
  to capture **effective addition order** (e.g., `(scratch0 + term7) + term8`
  vs `scratch0 + (term7 + term8)`) by logging **intermediate sums**.

## 2026-01-24 — stage 4 intermediate sums

Added sum traces for:
- `sum78_ya_yb`, `sum0_7ya`, `sum0_8yb`
- `sum78_yb_ya`, `sum0_7yb`, `sum0_8ya`
- `sum10ya_9yb`, `sum9ya_10yb`

Result (mdct2, ch0, stage 4, u=4):
- **Mismatches first appear in sums**, not in term products:
  - `sum78_yb_ya.i` differs (C `0x398cdea5`, Rust `0x398cdea4`)
  - `sum10ya_9yb.i` differs (C `0x3b4c82f0`, Rust `0x3b4c82f1`)
  - `sum9ya_10yb.r` differs (C `0x3a21b2d5`, Rust `0x3a21b2d6`)
- These deltas map directly to `scratch6.i` and `scratch12.r`.

Conclusion:
- The drift is now pinpointed to **specific sum operations** inside radix‑5.
- Next step: force Rust to compute these sums **exactly in C order** using
  explicit temporaries (e.g., `sum = termA + termB` then `sum += termC`), and
  avoid fusing or reordering, to see if the 1‑ULP disappears.

## 2026-01-24 — radix-5 sum78 FMA + mdct2 FFT stage alignment

Change:
- In `kf_bfly5`, compute `sum78_ya_yb` with `mul_add` to mirror C’s possible
  contraction (`sum78_ya_yb = term7_ya.mul_add(1.0, term8_yb)` / `mul_add` on
  i), keeping other sums explicit.

Result (frame 12, mdct2, ch0, stage 4, u=4):
- The prior 1‑ULP deltas in `sum78_yb_ya.i` disappear.
- **All mdct2 KFFT stage outputs now match** between C/Rust for frame 12.

Conclusion:
- The remaining drift in the mdct2 FFT path was due to **FMA contraction** in
  this sum path. With `sum78_ya_yb` fused, stage outputs align.

Follow‑up:
- Re‑run packet compare to confirm whether the frame‑12 payload mismatch remains
  (it does; see next section).

## 2026-01-24 — RC/quantization re‑trace after mdct2 alignment (frame 12)

Packet compare:
- **First payload mismatch remains at frame 12** (lengths C=161 vs Rust=159).

RC band trace:
- Added per‑band RC dumps in `quant_all_bands` and RDO sub‑steps.
- With `CELT_TRACE_RC_BAND=8`, the first mismatch appears at:
  - `celt_rc_band[12].band[8].stage=rdo_post_round_down`
  - `nbits_total` C=510 vs Rust=511.
- Band 1 trace matches fully, so the first RC drift is within **band 8**.

Theta trace (band 8):
- Added `compute_theta` dumps in C/Rust.
- All three `compute_theta` calls in band 8 (round_down/round_up + the nested
  split) **match exactly** (`b_in/b_out/qn/itheta/qalloc/delta`, `tell` values).

Conclusion:
- RC divergence in frame 12 is **not** from `compute_theta` in band 8.
- Next target: trace inside **PVQ / quant_partition / quant_band** for band 8
  (rdo round_down) to find the first differing `q`, `K`, `curr_bits`, or
  pulse allocation path.

## 2026-01-24 — PVQ/lowband folding fixes (frame 12)

New trace:
- Added `celt_pvq` traces in C/Rust `quant_partition` (entry + bits) with a
  recursion depth counter to pinpoint `b/q/curr_bits` mismatches.

Findings (frame 12, band 8):
- First PVQ mismatch at depth=1: `b` C=133 vs Rust=137 (and `q/curr_bits`)
  even though `compute_theta` matched.
- Root cause: missing “Give more bits to low‑energy MDCTs…” delta adjustment in
  Rust `quant_partition`.

Fix:
- Ported the `B0>1 && (itheta&0x3fff)` delta adjustment in Rust before
  `mbits/sbits` split (including the `(N<<BITRES>>(5-LM))` clamp).

Result:
- `celt_pvq` trace for band 8 now matches (q/curr_bits align), and
  `celt_rc_band` for band 8 matches through RDO.

Follow‑up mismatch (frame 12, band 1):
- `compute_theta`/PVQ `fill` mismatch (C `0x80`, Rust `0x00`).
- Cause: lowband folding selection differed. Rust used `band_start >=
  first_band_start`, which incorrectly treated band 0 as having a lowband.
- Fix:
  - Match C condition `band_start - N >= first_band_start` (i.e.
    `band_start >= first_band_start + N`).
  - Match C’s `fold_start` decrement loop and threshold (`effective + norm_offset + N`).

Result:
- Band 1 `fill` now matches; RC trace advances further.

Current earliest mismatch (frame 12):
- Band 9 `compute_theta` (stereo): `itheta` C=0 vs Rust=2730.
- RC state still matches up to band 8; the divergence now begins at band 9
  stereo angle quantization.

Next step:
- Trace `stereo_itheta` inputs for band 9 (X/Y vectors or mid/side energies)
  to see whether the mismatch comes from input normalization/folding or the
  stereo angle computation itself.

## 2026-01-24 — RC band_alloc + theta alignment (frame 12)

New traces:
- Added `celt_trace_band_alloc_dump` in C (`opus-c/celt/bands.c`) and
  `dump_band_alloc_if_match` in Rust (`src/celt/bands.rs`) to log per‑band
  allocation state (balance/tell/remaining/curr_balance/pulses/b) under
  `CELT_TRACE_RC*`.
- Reordered Rust `celt_stereo_itheta` trace fields and added `fmt_exp` helper
  so the floating-point dump format matches the C print ordering/format.

Fixes:
- `compute_theta` bias path: removed the extra `+ 8191` in Rust for
  `theta_round != 0` (C does not add it). This corrected band‑9 stereo
  `itheta` (C/Rust both `itheta=0` for the `theta_round=-1` call).
- Reset `ctx.theta_round = 0` after RDO selection in `quant_all_bands` to match
  C’s behavior.

RC trace status (frame 12):
- With the above fixes, `celt_rc_band` and `celt_pvq` traces now match C/Rust
  for all bands when **excluding** `celt_stereo_itheta` float lines.
- Remaining differences are tiny float deltas in `celt_stereo_itheta`
  (`emid/eside/side`), while `itheta_raw` matches. This suggests RC state
  alignment is complete aside from benign float print noise.

Packet compare re-run:
- The first payload mismatch **still** occurs at frame 12 (0‑based).
- After skipping the 16‑byte OPUSPKT1 header, the first differing payload byte
  is at offset 84; packet lengths: C 161 vs Rust 159.

Conclusion:
- RC/quantization appears aligned for frame 12; payload mismatch persists.
- Next step: investigate the small `celt_stereo_itheta` float drift or any
  upstream/non‑RC bitstream path that could alter packet length despite RC
  trace alignment.

## 2026-01-25 — stereo split / itheta input tracing (frame 12)

New traces:
- Added `celt_trace_stereo_itheta_input_dump` (C) and
  `dump_stereo_itheta_input_if_match` (Rust) under
  `CELT_TRACE_STEREO_ITHETA_IN`, called before `stereo_itheta` in
  `compute_theta`.
- Added `celt_trace_norm_in_dump` (C) and `dump_norm_in_if_match` (Rust) under
  `CELT_TRACE_NORM_IN`, called at band entry in `quant_all_bands`.
- Added `celt_trace_stereo_split_dump` (C) and `dump_stereo_split_if_match`
  (Rust) under `CELT_TRACE_STEREO_SPLIT`, called after
  `intensity_stereo`/`stereo_split` in `compute_theta`, and now include
  `bandE` left/right in the dump.

Findings (frame 12, band 10):
- `celt_norm_in` inputs **match** exactly between C/Rust, so the drift happens
  after band entry.
- `celt_stereo_itheta_in` for the top‑level stereo block (n=16, stereo=1)
  matches; the **only mismatch** is inside the split path where
  `stereo=0, n=8` (depth 1 in `quant_partition`).
- `celt_stereo_split` outputs differ by ~1 ULP even though `bandE` matches and
  the preceding inputs align; mismatches show up in `x` (mid) and some `y`
  values.

Attempted alignment:
- In Rust `stereo_split`, changed the scale constant to `0.70710678_f32` to
  mirror C’s `QCONST16(.70710678f, 15)` and tried fused `mul_add` for
  `scale*xl ± scale*yr`. The 1‑ULP drift **persisted**.

Conclusion:
- The remaining mismatch appears to be inside `intensity_stereo`/`stereo_split`
  rounding/contraction rather than upstream inputs or band energies.
- Next step: add deeper per‑term tracing inside `intensity_stereo`/`stereo_split`
  (e.g., `norm`, `a1/a2`, and `scale*left/right` intermediates) to identify the
  first mismatching float op.

## 2026-01-25 — stereo_split detail trace + explicit muls (frame 12)

New trace (detail):
- Added `CELT_TRACE_STEREO_SPLIT_DETAIL=1` to dump `intensity_stereo` and
  `stereo_split` intermediates (`left/right/norm/a1/a2`, per‑sample mul/sum,
  `scale`, `l/r/sum/diff`). Implemented in both C (`opus-c/celt/bands.c`) and
  Rust (`src/celt/bands.rs`) with band gating under `CELT_TRACE_RC*`.

Change:
- Rust `stereo_split` now uses explicit mul/add/sub (`scale*xl`, `scale*yr`,
  `sum`, `diff`) instead of `mul_add_f32`, matching the float C macro path.

Trace commands:
```
CELT_TRACE_RC=1 CELT_TRACE_RC_FRAME=12 CELT_TRACE_RC_BAND=10 \
CELT_TRACE_STEREO_SPLIT_DETAIL=1 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets_trace.opuspkt

CELT_TRACE_RC=1 CELT_TRACE_RC_FRAME=12 CELT_TRACE_RC_BAND=10 \
CELT_TRACE_STEREO_SPLIT_DETAIL=1 RUSTFLAGS="--cfg test" \
cargo run --example opus_packet_tool -- encode ehren-paper_lights-96.pcm /tmp/rust_packets_trace.opuspkt
```

Findings (frame 12, band 10):
- `celt_stereo_split_detail` **matches** between C/Rust for both `intensity` and
  `split` stages (all per‑term intermediates align).
- `celt_stereo_split` outputs now **match** for band 10 (no 1‑ULP drift).

Follow‑up (`stereo_itheta_in`):
- The nested `compute_theta` call with `stereo=0, n=8` still shows 1‑ULP
  differences in `x/y` inputs (e.g., `x[2]`, `y[2]`, `y[7]`), so the remaining
  drift persists in the split‑band inputs downstream of the top‑level stereo
  split.

Packet compare (after change):
- Frames: C 11405, Rust 11405.
- **First payload mismatch remains at frame 12** (0‑based).
- Packet lengths: C 161 vs Rust 159.
- First payload byte mismatch: offset 84 (after 16‑byte OPUSPKT1 header).

Conclusion:
- `stereo_split` is now bit‑aligned with C, but a small drift persists in the
  split‑band (`stereo=0`) `stereo_itheta_in` inputs. Next step: trace the band
  coefficients just before `quant_partition`’s split `compute_theta` to locate
  the first divergence (likely in `haar1`/`deinterleave_hadamard`).

## 2026-01-25 — band prepartition trace + intensity FMA alignment (frame 12)

New trace (pre‑partition):
- Added `CELT_TRACE_BAND_PREPART=1` in both C and Rust to dump band coefficients
  around the Haar/deinterleave steps. New stages: `pre_haar_recombine_*`,
  `haar_recombine_*`, `pre_haar_time_divide_*`, `haar_time_divide_*`,
  `post_haar`, `post_deinterleave`.

Initial finding:
- With `CELT_TRACE_BAND_PREPART=1`, the **first mismatch** shows up at
  `pre_haar_time_divide_0` for frame 12 band 10, so the drift happens *before*
  `haar1`. This pointed back to the stereo pre‑processing (`compute_theta`).

Stereo split vs detail:
- `celt_stereo_split_detail` (intensity stage) **matched** C/Rust, but the
  final `celt_stereo_split` output differed by 1 ULP. In the C log, the
  printed `sum_bits` inside the detail block did **not** match the final
  `x_bits`, implying FMA contraction in the `MAC16_16(a1*l, a2, r)` path.

Fix:
- In Rust `intensity_stereo`, compute the output with fused multiply‑add:
  `x[idx] = a1.mul_add(l, mul2)` (where `mul2 = a2 * r`). This aligns with the
  C output that appears to be FMA‑contracted.
- Also aligned `haar1`’s scale constant to `0.70710678_f32` (C uses
  `QCONST16(.70710678f, 15)` in float builds).

Results after the change:
- `celt_stereo_split` now matches between C/Rust (frame 12 band 10).
- `celt_band_prepartition` stages now match (no diff at pre/post Haar or
  deinterleave).
- `celt_stereo_itheta_in` now matches (previous 1‑ULP drift gone).

Trace commands:
```
CELT_TRACE_RC=1 CELT_TRACE_RC_FRAME=12 CELT_TRACE_RC_BAND=10 \
CELT_TRACE_BAND_PREPART=1 CELT_TRACE_NORM_IN=1 \
CELT_TRACE_STEREO_SPLIT=1 CELT_TRACE_STEREO_SPLIT_DETAIL=1 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets_prepart.opuspkt

CELT_TRACE_RC=1 CELT_TRACE_RC_FRAME=12 CELT_TRACE_RC_BAND=10 \
CELT_TRACE_BAND_PREPART=1 CELT_TRACE_NORM_IN=1 \
CELT_TRACE_STEREO_SPLIT=1 CELT_TRACE_STEREO_SPLIT_DETAIL=1 \
RUSTFLAGS="--cfg test" cargo run --example opus_packet_tool -- encode \
ehren-paper_lights-96.pcm /tmp/rust_packets_prepart.opuspkt

CELT_TRACE_RC=1 CELT_TRACE_RC_FRAME=12 CELT_TRACE_RC_BAND=10 \
CELT_TRACE_STEREO_ITHETA_IN=1 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets_itheta.opuspkt

CELT_TRACE_RC=1 CELT_TRACE_RC_FRAME=12 CELT_TRACE_RC_BAND=10 \
CELT_TRACE_STEREO_ITHETA_IN=1 \
RUSTFLAGS="--cfg test" cargo run --example opus_packet_tool -- encode \
ehren-paper_lights-96.pcm /tmp/rust_packets_itheta.opuspkt
```

Next step:
- Re-run the packet compare to see whether the frame‑12 payload mismatch is
  resolved now that stereo split + pre‑partition stages align.

## 2026-01-25 — frame 12 tonality bin 237 + music_prob min/max alignment

Re-ran `analysis_compare` (64 frames) and isolated the new first drift:
- First diff moved to `analysis_info[12].tonality_slope` (1 ULP), so traced
  tonality slope internals for frame 12.

Tonality slope trace (frame 12, band 17):
- `analysis_tonality` showed the only mismatch was **band 17** (`tE`,
  `energy_ratio_num`, `band_tonality`, `slope_term/slope_acc`).
- `analysis_tonality_bin` showed the first bin mismatch at **bin 237**
  (`tonality` / `tonality2` 1‑ULP); all upstream raw terms matched.

Fix (tonality formula):
- In Rust, compute the tonality denominator via FMA to mirror C contraction:
  `denom = mul_add_f32(scale, avg_mod, 1.0)` and
  `denom2 = mul_add_f32(scale, mod2, 1.0)` (with `scale = 40*16*pi4`).
- After this change, band‑17 bin traces match and frame‑12 tonality slope
  aligns.

Follow-up (music_prob min/max drift):
- With tonality aligned, the remaining diffs were `music_prob_min/max`
  (frame 12). The loop uses a multiply‑add to accumulate `prob_avg` and
  a multiply‑add in the transition penalty term. Rust now mirrors C with
  `mul_add_f32` for these expressions (and uses raw `pos_vad` like C).
- After this change, **frame 12** is clean; the first mismatch now shifts to
  **frame 13** (`analysis_info[13].music_prob` 1‑ULP).

Commands:
```
ctests/build/analysis_compare ehren-paper_lights-96.pcm 64 > /tmp/analysis_compare_c.txt
ANALYSIS_PCM=ehren-paper_lights-96.pcm ANALYSIS_FRAMES=64 \
cargo test -p mousiki --lib analysis_compare_output -- --nocapture \
  > /tmp/analysis_compare_r.txt

ANALYSIS_TRACE_TONALITY_SLOPE=1 ANALYSIS_TRACE_TONALITY_SLOPE_FRAME=12 \
ANALYSIS_TRACE_TONALITY_SLOPE_BANDS=17 ANALYSIS_TRACE_TONALITY_SLOPE_BITS=1 \
ctests/build/analysis_compare ehren-paper_lights-96.pcm 64 \
  > /tmp/analysis_tonality_band17_c.txt

ANALYSIS_TRACE_TONALITY_SLOPE=1 ANALYSIS_TRACE_TONALITY_SLOPE_FRAME=12 \
ANALYSIS_TRACE_TONALITY_SLOPE_BANDS=17 ANALYSIS_TRACE_TONALITY_SLOPE_BITS=1 \
ANALYSIS_PCM=ehren-paper_lights-96.pcm ANALYSIS_FRAMES=64 \
cargo test -p mousiki --lib analysis_compare_output -- --nocapture \
  > /tmp/analysis_tonality_band17_r.txt
```

Next step:
- Trace frame 13 `music_prob` drift using the same tonality/activity traces
  (or add a targeted trace inside `tonality_get_info` around `prob_avg` /
  `prob_min` / `prob_max`).

## 2026-01-25 — GRU trace (frame 13) + h_post alignment

After the tonality + music_prob fixes, the first drift moved to frame 13
`analysis_info[13].music_prob`. Activity trace showed:
- `features` now match (after the FMA tweak on `features[4+i]`).
- The remaining mismatch was only in `rnn_post` (and thus `frame_probs`).

### GRU trace (new)
Added a focused GRU trace in both C and Rust:
- C: `opus-c/src/mlp.c` prints `analysis_gru[frame]` dumps for `z`, `r`,
  `h_pre`, `h_act`, `h_mix1`, `h_mix2`, `h_post` with optional bits.
- Rust: `src/mlp.rs` mirrors the same dumps under a new
  `ANALYSIS_TRACE_GRU*` env; `src/analysis.rs` sets the frame index each
  iteration (`set_gru_trace_frame`).
- C `analysis.c` now always calls `analysis_gru_set_frame(...)` with its own
  per‑frame counter so the GRU trace does not depend on the activity trace.

Findings (frame 13):
- `z`, `r`, `h_pre`, and `h_act` all match between C/Rust.
- The first mismatch was **only** in `h_post`.
- After exposing `h_mix1 = z*state` and `h_mix2 = (1-z)*h_act`, both `h_mix*`
  terms match, so the fix is to use a **plain sum** (no FMA) for `h_post`.

Fix:
- Rust `analysis_compute_gru` now computes:
  `h_mix1 = z*state`, `h_mix2 = (1-z)*h_act`, `h_post = h_mix1 + h_mix2`.
  This aligns the GRU output with C.

Result:
- Activity trace for frame 13 now matches fully.
- `analysis_compare` first diff moved to frame 14:
  `analysis_state[14].cmean[0]` (1‑ULP).

GRU trace commands:
```
ANALYSIS_TRACE_GRU=1 ANALYSIS_TRACE_GRU_FRAME=13 ANALYSIS_TRACE_GRU_BITS=1 \
ctests/build/analysis_compare ehren-paper_lights-96.pcm 64 \
  > /tmp/analysis_gru13_c.txt

ANALYSIS_TRACE_GRU=1 ANALYSIS_TRACE_GRU_FRAME=13 ANALYSIS_TRACE_GRU_BITS=1 \
ANALYSIS_PCM=ehren-paper_lights-96.pcm ANALYSIS_FRAMES=64 \
cargo test -p mousiki --lib analysis_compare_output -- --nocapture \
  > /tmp/analysis_gru13_r.txt
```

Next step:
- Trace frame 14 `cmean[0]` drift with activity trace (check `bfcc` inputs and
  `cmean` update formula for FMA/ordering).

2026-01-25 — Activity/tonality alignment through frame 64

Progression after frame 14 fix:
- Frame 17 mismatch moved into activity/MLP inputs. Activity trace showed
  mismatched `log_e`/`bfcc`/features; tonality slope trace matched.
- Switched `log_e` to use `log` with f64 intermediates to mirror C’s
  `(float)log(...)`, and re-ordered feature[0..3] accumulation with fma to
  match C’s rounding.
- Frame 18 mismatch isolated to `frame_noisiness`/`n_e` accumulation. Fixed by
  using `fma` for `n_e` update: `n_e = fma(bin_e*2, (0.5-noisiness), n_e)`.
- Frame 27 mismatch isolated to `relative_e`. Fixed by keeping the denominator
  as `1e-5 + (high_e - low_e)` explicitly (matches C’s parentheses).
- Frame 47 tonality mismatch traced to `max_frame_tonality` weighting.
  C effectively evaluates `(1 + 0.03*(b-NB_TBANDS))` in higher precision; Rust
  now uses f64 intermediates for the weight and weighted term before casting
  back to f32.

Trace hooks added (debug-only / env-gated):
- Activity trace now dumps `frame_noisiness`, `relative_e`, `activity`, plus
  per-band `band_e`, `band_n_e`, `band_noisiness` (with optional bits).
- Tonality trace now dumps per-band `frame_tonality`, `weight`, `weighted`,
  `max_frame_tonality`, and per-frame `max_frame_tonality` / `frame_tonality`
  details (with optional bits).

Result:
- `analysis_compare` (64 frames, `ehren-paper_lights-96.pcm`) now shows no
  mismatches between C and Rust.

## 2026-01-25 — Frame 13 CELT VBR/alloc alignment (mismatch now at frame 16)

Packet compare (OPUSPKT1, `ehren-paper_lights-96.pcm`) after the range‑coder
fixes:
- First payload mismatch moved from frame 13 to frame 16.
- New first mismatch: frame index 16, payload len C=165 / Rust=165, first
  differing byte at offset 1 (payload includes TOC).

### Root cause for frame 13 (now fixed)
The first mismatch was in CELT VBR/alloc before quantization:
- `celt_ctrl` showed `nb_compressed_bytes`, `tf_estimate`, `alloc_trim`, and
  `bits` diverging.
- `celt_vbr_budget` drifted at `post_target`.
- `celt_alloc_interp` already diverged at `bits1/bits2` for band 0.

Fixes:
1) **Clamp max_data_bytes inside `encode_frame_native`**  
   C clamps `orig_max_data_bytes` to 1276 per frame. Rust was only clamping
   to `MAX_PACKET_BYTES` at the outer layer, so the CELT budget saw 3828
   bytes. Now `encode_frame_native` caps `max_data_bytes` to 1276.
   - `opus_celt_budget` trace matches C (frame 13).

2) **`transient_analysis` float branch**  
   - Use `norm = len2 / (frame_energy + 1e-15)` (float build: `SHR32` is a
     no‑op in C; Rust was dividing by `frame_energy * 0.5`).
   - Keep `mask_metric` as **int** and normalize with integer division
     (`64*unmask*4/(6*(len2-17))`) like C.  
   Result: `tf_estimate` matches bit‑exact.

3) **`compute_vbr` tonality step truncation**  
   C does `target + (int)(coded*1.2*tonal)`; Rust was casting after the sum.
   Now Rust truncates the product before adding, matching C and fixing the
   1‑LSB drift after `after_tonality`.

4) **Trace parity for `tell`**  
   After encoding `alloc_trim`, set `tell = tell_frac` to mirror C’s use of
   `ec_tell_frac` for later traces (`celt_ctrl`, `celt_vbr_budget`).

After these changes (frame 13):
- `celt_vbr_budget`, `celt_ctrl`, `celt_alloc_interp`, `celt_quant` (bit
  dumps), and `celt_rc` all match C.
- The packet mismatch moved forward to frame 16.

Trace commands used:
```
CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=13 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets.opuspkt

CELT_TRACE_PCM=ehren-paper_lights-96.pcm CELT_TRACE_FRAMES=64 \
CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=13 \
cargo test -p mousiki --lib celt_alloc_trace_output -- --nocapture

CELT_TRACE_ALLOC_INTERP=1 CELT_TRACE_ALLOC_INTERP_FRAME=13 CELT_TRACE_ALLOC_INTERP_BAND=0 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets.opuspkt

CELT_TRACE_PCM=ehren-paper_lights-96.pcm CELT_TRACE_FRAMES=64 \
CELT_TRACE_ALLOC_INTERP=1 CELT_TRACE_ALLOC_INTERP_FRAME=13 CELT_TRACE_ALLOC_INTERP_BAND=0 \
cargo test -p mousiki --lib celt_alloc_trace_output -- --nocapture
```

Next:
- Start tracing frame 16 (same CELT control/VBR/alloc path). Check whether the
  first drift is still in VBR target/alloc or later in quantization/RC.

## 2026-01-25 — Prefilter threshold branch keeps pitch_index (frame 12)

Findings (frame 12, `run_prefilter` debug):
- C and Rust agree on `gain1_pre`, `pf_threshold`, and `pf_on_pre_cancel`.
- **Difference**: when `gain1 < pf_threshold`, C **does not** reset
  `pitch_index`; Rust was forcing `pitch_index = COMBFILTER_MINPERIOD`.
- This showed up as `pitch_index_quant` mismatch: C=1021 vs Rust=15.

Fix:
- Removed the `pitch_index = COMBFILTER_MINPERIOD` assignment in the
  `gain1 < pf_threshold` branch of Rust `run_prefilter`, matching C.

Result:
- Prefilter debug outputs now match for frame 12.
- Packet mismatch moved forward again.

Packet compare (OPUSPKT1):
- **First payload mismatch is now frame 21** (0‑based).
- Lengths match: C=161, Rust=161.
- First differing byte offset: 8 (C=76, Rust=81).

Trace commands:
```
CELT_TRACE_PREFILTER=1 CELT_TRACE_PREFILTER_FRAME=12 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets.opuspkt

CELT_TRACE_PCM=ehren-paper_lights-96.pcm CELT_TRACE_FRAMES=64 \
CELT_TRACE_PREFILTER=1 CELT_TRACE_PREFILTER_FRAME=12 \
cargo test -p mousiki --lib celt_alloc_trace_output -- --nocapture
```

## 2026-01-25 — Temporal VBR + CELT-only end_band fixes (mismatch now frame 104)

Findings:
- Frame 21 mismatch traced back to `celt_ctrl` showing `end` mismatch
  (C=19 vs Rust=21) and `tell` drift before `clt_compute_allocation`.
- Opus mode trace for frame 21 shows `bandwidth=1104` (SuperWide) on both sides,
  but Rust **never set `end_band` in the CELT-only path**, so the CELT encoder
  retained the previous (fullband) `end_band`.
- `compute_vbr` delta drift started at frame 16 due to **temporal VBR math**:
  - `follow` decay was wrong (`max(follow, candidate)`), so it never decayed.
  - `temporal_vbr` incorrectly used `frame_avg * 32` even in float build; C’s
    `SHL32` is a no-op, so it should be just `frame_avg - spec_avg`.

Fixes:
- Correct `follow` update in temporal VBR:
  `follow = (follow - 1.0).max(candidate);`
- Remove the `* 32` in `temporal_vbr` (float build matches C’s `SHL32` no-op).
- Set `CELT_SET_END_BAND` in the **CELT-only** encode path, matching the
  hybrid path.

Results:
- `celt_temporal_vbr`/`celt_vbr_budget` now match for frame 16.
- `celt_ctrl` end-band mismatch for frame 21 resolved.
- **First packet mismatch moved to frame 104** (0‑based), length 158; first
  differing byte offset 79 (C=141, Rust=159).

Trace commands:
```
CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=16 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets.opuspkt

CELT_TRACE_PCM=ehren-paper_lights-96.pcm CELT_TRACE_FRAMES=64 \
CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=16 \
cargo test -p mousiki --lib celt_alloc_trace_output -- --nocapture

OPUS_TRACE_MODE=1 OPUS_TRACE_MODE_FRAME=21 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets.opuspkt

OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=64 \
OPUS_TRACE_MODE=1 OPUS_TRACE_MODE_FRAME=21 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture
```

## 2026-01-25 — Prefilter comb_filter uses fused mul-add (mismatch now frame 693)

Findings (frame 104):
- Prefilter params and inputs matched, but `celt_prefilter.post` differed by
  1 ULP at sample 0 (`0x4327a552` vs `0x4327a553`).
- Tracing showed the first drift in the comb filter overlap section
  (prefilter output), before MDCT/quantization.
- C is built with `-ffast-math`, so fused multiply-add contraction is allowed;
  Rust was doing separate mul + add.

Fix:
- Use `mul_add_f32` (libm `fmaf`) when accumulating terms in `comb_filter` and
  `comb_filter_const`, matching C’s fused rounding.

Results:
- `celt_prefilter[104].post.*` sample bits now match exactly.
- **First packet mismatch moved to frame 693** (0‑based), length 170 vs 171;
  first differing byte offset 23 (C=193, Rust=194).

Trace commands:
```
CELT_TRACE_PREFILTER=1 CELT_TRACE_PREFILTER_FRAME=104 CELT_TRACE_PREFILTER_BITS=1 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets.opuspkt

CELT_TRACE_PCM=ehren-paper_lights-96.pcm CELT_TRACE_FRAMES=128 \
CELT_TRACE_PREFILTER=1 CELT_TRACE_PREFILTER_FRAME=104 CELT_TRACE_PREFILTER_BITS=1 \
cargo test -p mousiki --lib celt_alloc_trace_output -- --nocapture
```

## 2026-01-26 — transient_analysis inv_table mismatch fixed (frame 693 VBR aligns)

New trace knobs (C + Rust):
- `CELT_TRACE_TRANSIENT=1`
- `CELT_TRACE_TRANSIENT_FRAME=<n>`
- `CELT_TRACE_TRANSIENT_BITS=1` (optional)

Findings (frame 693):
- `mask_metric` differed by 1: C=175 vs Rust=176, which propagated into
  `tf_estimate` and VBR target.
- Per-step unmask trace showed the first *effective* divergence at
  `channel[1]`, `i=368`: `clamped=110` matched but `inv_table` differed
  (C=3 vs Rust=4).
- Rust `INV_TABLE` had four mismatched entries vs C:
  indices 73, 85, 110, 111.

Fix:
- Update Rust `INV_TABLE` in `transient_analysis` to match the C table.

Results:
- `mask_metric`, `tf_max`, `tf_estimate` now match bit‑exact for frame 693.
- `celt_vbr_budget` now matches C for frame 693
  (`target=10816`, `nb_compressed_bytes=169`).
- Packet compare on the first 700 frames (`/tmp/ehren-700.pcm`) now matches:
  no payload/TOC/length mismatches across frames 0‑699.

Trace commands used:
```
CELT_TRACE_TRANSIENT=1 CELT_TRACE_TRANSIENT_FRAME=693 CELT_TRACE_TRANSIENT_BITS=1 \
ctests/build/opus_packet_encode /tmp/ehren-700.pcm /tmp/opus_c_transient_693.opuspkt \
  > /tmp/opus_c_transient_693.txt

CELT_TRACE_TRANSIENT=1 CELT_TRACE_TRANSIENT_FRAME=693 CELT_TRACE_TRANSIENT_BITS=1 \
OPUS_TRACE_PCM=/tmp/ehren-700.pcm OPUS_TRACE_FRAMES=700 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_transient_693.txt
```

Packet compare (700 frames):
```
ctests/build/opus_packet_encode /tmp/ehren-700.pcm /tmp/c_packets_new.opuspkt
cargo run --example opus_packet_tool -- encode /tmp/ehren-700.pcm /tmp/rust_packets_new.opuspkt
```

Full packet compare (entire file):
- First mismatch now at **frame 6099** (0‑based), len C=149 vs Rust=151,
  first differing byte offset 1 (C=234, Rust=1).

```
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/c_packets_full.opuspkt
cargo run --example opus_packet_tool -- encode ehren-paper_lights-96.pcm /tmp/rust_packets_full.opuspkt
```

### Frame 6099 VBR budget trace (next target)

Initial VBR trace shows the divergence already in inputs:
- `tell_frac`: C=671 vs Rust=566
- `tot_boost`: C=0 vs Rust=48
- `stereo_saving`: C=-7.202e‑4 vs Rust=-3.557e‑4
- `max_depth`: C=13.9433 vs Rust=14.5302
- `temporal_vbr`: C=-0.5208 vs Rust=-0.2624

As a result:
- `target`: C=9472 vs Rust=9600
- `nb_compressed_bytes`: C=148 vs Rust=150

Trace commands:
```
CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=6099 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/opus_c_vbr_budget_6099.opuspkt \
  > /tmp/opus_c_vbr_budget_6099.txt

CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=6099 \
OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=6100 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_vbr_budget_6099.txt
```

Next:
- Trace `tot_boost`, `tell_frac`, `max_depth`, and `temporal_vbr` inputs at the
  earlier stages to locate where they first diverge (likely analysis/RC path
  or temporal VBR state).

### Frame 6099 deeper trace (tot_boost / tell_frac / max_depth / temporal_vbr)

Findings:
- **Temporal VBR drift starts at band 0**:
  - C: `bandLogE[0].ch0=-8.7376e-02`, `bandLogE[0].ch1=-4.6890e-01`
  - Rust: `bandLogE[0].ch0=2.1404e-01`, `bandLogE[0].ch1=-4.0834e-01`
  - This flips the `candidate`/`follow` sequence and yields
    `frame_avg` C=-1.964e-01 vs Rust=6.201e-02.
- **dynalloc_analysis divergence**:
  - C: `tot_boost=0`, `max_depth=1.3943e+01`, offsets all zero.
  - Rust: `tot_boost=48`, `max_depth=1.4530e+01`,
    **offsets[1]=1** (all others zero).
- **tell_frac already diverged before dynalloc bit loop**:
  - C: `tell_frac_pre=647` → `tell_frac_post=651`
  - Rust: `tell_frac_pre=484` → `tell_frac_post=546`
  - So RC state differs *before* dynalloc bits/alloc_trim.

Conclusion:
- The first visible drift at frame 6099 is in **band energies/logE** (band 0),
  which cascades into temporal VBR and dynalloc (tot_boost). RC `tell_frac`
  already differs before dynalloc, so the earlier divergence likely occurs
  **before or during coarse energy/TF/spreading decisions**, not in VBR.

Trace commands:
```
CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=6099 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/opus_c_vbr_budget_6099.opuspkt \
  > /tmp/opus_c_vbr_budget_6099.txt

CELT_TRACE_VBR_BUDGET=1 CELT_TRACE_VBR_BUDGET_FRAME=6099 \
OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=6100 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_vbr_budget_6099.txt
```

Next:
- Trace earlier CELT stages for frame 6099 (prefilter / MDCT / band energies)
  to pinpoint the first time-domain or spectral mismatch.

### Frame 6099 band energy comparison

Using `CELT_TRACE_BAND_ENERGY`, the first mismatch is already in
`compute_band_energies` output:
- **Band 0, ch 0**: C=81.5788956 vs Rust=100.534309387 (diff +18.96)
- Band 0, ch 1 also differs (C=62.6218834 vs Rust=65.306854248)

So the divergence is **before** `amp2_log2` and VBR; next step is to trace
MDCT outputs and prefilter for frame 6099.

Trace commands:
```
CELT_TRACE_BAND_ENERGY=1 CELT_TRACE_BAND_ENERGY_FRAME=6099 CELT_TRACE_BAND_ENERGY_BITS=1 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/opus_c_band_6099.opuspkt \
  > /tmp/opus_c_band_6099.txt

CELT_TRACE_BAND_ENERGY=1 CELT_TRACE_BAND_ENERGY_FRAME=6099 CELT_TRACE_BAND_ENERGY_BITS=1 \
OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=6100 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_band_6099.txt
```

### Frame 6099 MDCT comparison (first 64 bins)

MDCT already diverges (channel 0) at the first bin:
- idx0 ch0: C=-3.98263454 vs Rust=-5.518184185
- Many subsequent bins differ, so the mismatch is **at or before MDCT**.

Trace commands:
```
CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_FRAME=6099 CELT_TRACE_MDCT_BITS=1 \
CELT_TRACE_MDCT_START=0 CELT_TRACE_MDCT_COUNT=64 \
ctests/build/opus_packet_encode ehren-paper_lights-96.pcm /tmp/opus_c_mdct_6099.opuspkt \
  > /tmp/opus_c_mdct_6099.txt

CELT_TRACE_MDCT=1 CELT_TRACE_MDCT_FRAME=6099 CELT_TRACE_MDCT_BITS=1 \
CELT_TRACE_MDCT_START=0 CELT_TRACE_MDCT_COUNT=64 \
OPUS_TRACE_PCM=ehren-paper_lights-96.pcm OPUS_TRACE_FRAMES=6100 \
cargo test -p mousiki --lib opus_mode_trace_output -- --nocapture \
  > /tmp/opus_r_mdct_6099.txt
```

Next:
- Re-run packet compare to see whether the frame‑693 payload mismatch is
  resolved and find the new first mismatch frame (if any).
