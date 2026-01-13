# Hybrid Framing Porting Checklist (C vs Rust)

This checklist captures remaining hybrid-encode path deltas after comparing
`opus-c/src/opus_encoder.c` to the Rust port. It focuses on the hybrid
processing path beyond framing/packetization.

1. ✅ Wire `delay_buffer` into the PCM preparation path.
   - C uses `delay_buffer` + `encoder_buffer`/`delay_compensation` to build
     `pcm_buf` before encoding and to keep overlap history.
   - Rust now stages PCM with delay history and exposes the missing fields.
   - Target: `src/opus_encoder.rs` (with a focused unit test).

2. ✅ Port high-pass filtering and DC rejection on the input PCM.
   - C computes `hp_freq_smth1` and updates `variable_HP_smth2_Q15`, then calls
     `hp_cutoff()` (VoIP) or `dc_reject()` (music).
   - Rust now updates `variable_hp_smth2_q15`, computes cut-off, and applies
     `hp_cutoff()`/`dc_reject()` during PCM staging with unit tests.
   - Target: `src/opus_encoder.rs`.

3. ✅ Port HB gain computation and apply `gain_fade()`.
   - C computes `HB_gain` in hybrid and fades between frames using
     `gain_fade()`, tracking `prev_hb_gain`.
   - Rust now computes `hb_gain` based on CELT bitrate allocation and applies
     `gain_fade()` to smoothly transition between frames before CELT encoding.
   - Target: `src/opus_encoder.rs`.

4. ✅ Port hybrid stereo width handling via `stereo_fade()`.
   - C uses `hybrid_stereo_width_Q14` and `silk_mode.stereoWidth_Q14` to
     attenuate stereo width at low bitrates.
   - Rust now computes stereo width based on equiv_rate and applies
     `stereo_fade()` to smoothly transition between frames, tracking
     `hybrid_stereo_width_q14` for frame-to-frame continuity.
   - Target: `src/opus_encoder.rs`.

5. ✅ Port `energy_masking` influence on SILK/Hybrid bit allocation.
   - C applies surround masking to adjust `silk_mode.bitRate` and HB gain.
   - Rust now stores `energy_masking` pointer and `lfe` flag, computes surround
     masking rate offset to adjust SILK bitrate, skips HB gain attenuation when
     energy_masking is present, and skips stereo width reduction in surround mode.
   - Target: `src/opus_encoder.rs`.

6. ✅ Port the prefill ramp and `tmp_prefill` logic for mode switches.
   - C fades the delay buffer for a smooth onset and uses `tmp_prefill` for
     CELT prefill when switching modes.
   - Rust now ramps the delay buffer for SILK prefill, converts it back to
     PCM for the prefill call, and uses delay-buffer slices for CELT prefill.
   - Target: `src/opus_encoder.rs` (with focused unit tests).

7. ✅ Port the delay buffer shift/copy after encoding.
   - C updates `delay_buffer` after encoding, before gain/stereo fades.
   - Rust now updates `delay_buffer` in SILK-only, CELT-only, and hybrid paths, and keeps
     the copy ahead of gain/stereo fades.
   - Target: `src/opus_encoder.rs` (with focused unit tests).

8. ✅ (Optional) DRED-specific hybrid adjustments.
   - C adjusts `nb_compr_bytes` and activity history in DRED paths, with DRED bitrate budgeting and CELT CBR overrides.
   - Rust now mirrors the per-frame DRED handling (bitrate budgeting, activity history update, CELT budget cap, CBR->VBR override) under the `dred` feature flag with unit coverage.
   - Target: `src/opus_encoder.rs` (feature-gated).

9. ✅ Add tests to cover the newly ported behaviors.
   - Added hybrid encode vector tests for HP filtering + delay-buffer effects and stereo width/final-range.
   - Target: `tests/hybrid_encode.rs` with fixtures in `tests/fixtures/hybrid_encode_vectors.rs`.
