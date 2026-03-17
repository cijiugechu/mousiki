# Unsafe Audit

Last updated: 2026-03-17

## Purpose

This note records the current `unsafe` usage in the Rust codebase, why it exists, and which parts look realistically removable.

The goal is not just to list `unsafe` sites. It is to give a later contributor enough context to decide:

- which `unsafe` is accidental and should be removed,
- which `unsafe` is structural and needs a redesign first,
- which replacement patterns fit this repository's constraints.

Scope:

- included: `src/`
- excluded: `opus-c/`
- excluded: tests-only fixtures under `tests/` unless they affect a design recommendation

As of this audit:

- `rg -n '\bunsafe\b' src` reports 61 matches
- `rg -n 'unsafe impl|unsafe fn|unsafe \{' src` reports 56 concrete `unsafe` items

## Important Context

This crate is `#![no_std]`, with `alloc` available and `std` only in test-only paths.

That matters for replacement strategy:

- `std::sync::OnceLock` is not available in normal library code
- safe replacements should prefer:
  - plain ownership/layout changes
  - safe slice APIs
  - explicit serialization/deserialization
  - no_std-friendly crates when a crate is justified

Also note the porting style:

- many current `unsafe` sites exist because the port mirrors C allocation shape or C API surface
- some of those are still worth keeping if they preserve required aliasing or layout semantics
- some are only transitional and should be removed once the Rust-side ownership model is cleaned up

## Current Distribution

`unsafe` appears in these source files:

- [`src/celt/celt.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs)
- [`src/celt/celt_decoder.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs)
- [`src/celt/celt_encoder.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt_encoder.rs)
- [`src/celt/kiss_fft.rs`](/Users/nemurubaka/repos/mousiki/src/celt/kiss_fft.rs)
- [`src/celt/mini_kfft.rs`](/Users/nemurubaka/repos/mousiki/src/celt/mini_kfft.rs)
- [`src/celt/modes.rs`](/Users/nemurubaka/repos/mousiki/src/celt/modes.rs)
- [`src/mapping_matrix.rs`](/Users/nemurubaka/repos/mousiki/src/mapping_matrix.rs)
- [`src/opus_encoder.rs`](/Users/nemurubaka/repos/mousiki/src/opus_encoder.rs)
- [`src/range.rs`](/Users/nemurubaka/repos/mousiki/src/range.rs)
- [`src/silk/debug.rs`](/Users/nemurubaka/repos/mousiki/src/silk/debug.rs)

`src/silk/encoder/state.rs` mentions `unsafe` only in a comment and does not contain actual `unsafe` code.

## Categories

The existing `unsafe` falls into a few repeatable patterns.

### 1. Hand-rolled FFI calls for math functions

Files:

- [`src/celt/mini_kfft.rs:78`](/Users/nemurubaka/repos/mousiki/src/celt/mini_kfft.rs:78)
- [`src/celt/kiss_fft.rs:778`](/Users/nemurubaka/repos/mousiki/src/celt/kiss_fft.rs:778)
- [`src/celt/kiss_fft.rs:792`](/Users/nemurubaka/repos/mousiki/src/celt/kiss_fft.rs:792)

These use `unsafe extern "C"` declarations for `fmaf`, `sin`, and `cos`.

Observation:

- the codebase already depends on `libm`
- other files already use safe `libm::fmaf`

Best-practice replacement:

- replace hand-written FFI bindings with safe `libm::{fmaf, sin, cos}`

Assessment:

- high-confidence removable
- low risk
- good first cleanup

### 2. Raw bytes reinterpreted as typed layout

File:

- [`src/mapping_matrix.rs`](/Users/nemurubaka/repos/mousiki/src/mapping_matrix.rs)

Sites:

- [`src/mapping_matrix.rs:65`](/Users/nemurubaka/repos/mousiki/src/mapping_matrix.rs:65)
- [`src/mapping_matrix.rs:77`](/Users/nemurubaka/repos/mousiki/src/mapping_matrix.rs:77)
- [`src/mapping_matrix.rs:140`](/Users/nemurubaka/repos/mousiki/src/mapping_matrix.rs:140)
- [`src/mapping_matrix.rs:159`](/Users/nemurubaka/repos/mousiki/src/mapping_matrix.rs:159)

Current design:

- `MappingMatrix` stores `Box<[u8]>`
- header and payload are written/read by pointer casts

Why it exists:

- it preserves the serialized C-style layout in memory

Why it is likely removable:

- internal Rust code does not fundamentally need a byte blob as the canonical owned form
- the public/useful view is already typed: `rows`, `cols`, `gain_db`, `data: &[i16]`

Best-practice replacements:

1. Prefer a typed owned representation:
   - `rows: usize`
   - `cols: usize`
   - `gain_db: i32`
   - `data: Box<[i16]>`
2. When bytes are needed, serialize explicitly with `to_ne_bytes` / `from_ne_bytes`
3. If zero-copy byte views are still desired, consider a dedicated safe layout crate such as `zerocopy` or `bytemuck`

Assessment:

- high-confidence removable
- requires a contained refactor, but not an architectural rewrite

### 3. Raw pointer field used as delayed borrowed slice

File:

- [`src/opus_encoder.rs`](/Users/nemurubaka/repos/mousiki/src/opus_encoder.rs)

Site:

- [`src/opus_encoder.rs:2836`](/Users/nemurubaka/repos/mousiki/src/opus_encoder.rs:2836)

Current design:

- encoder state stores `energy_masking: *const f32`
- later code reconstructs `&[f32]` with `from_raw_parts`

Why it exists:

- the CTL surface mirrors a C-style pointer setter:
  [`src/opus_encoder.rs:732`](/Users/nemurubaka/repos/mousiki/src/opus_encoder.rs:732)

Why it is probably removable from core logic:

- the raw pointer only exists to keep the C-like API shape
- internal encode logic does not need to keep this as a raw pointer

Best-practice replacements:

1. Keep the raw pointer only at the API boundary, but copy into owned storage immediately
   - the size is tiny: 21 bands per channel
2. Store an internal safe representation
   - e.g. `Option<[f32; 42]>` for stereo-capable default Opus encoder
   - or `Option<Vec<f32>>` if flexibility matters more than stack layout
3. If the external API can change, prefer `Option<&[f32]>` at the Rust boundary

Assessment:

- likely removable from the encode hot path
- CTL/API compatibility may still force a raw-pointer boundary somewhere

### 4. Hand-rolled lazy statics using `UnsafeCell`

Files:

- [`src/celt/modes.rs`](/Users/nemurubaka/repos/mousiki/src/celt/modes.rs)
- [`src/celt/celt_decoder.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs)
- [`src/silk/debug.rs`](/Users/nemurubaka/repos/mousiki/src/silk/debug.rs)

Representative sites:

- [`src/celt/modes.rs:142`](/Users/nemurubaka/repos/mousiki/src/celt/modes.rs:142)
- [`src/celt/modes.rs:151`](/Users/nemurubaka/repos/mousiki/src/celt/modes.rs:151)
- [`src/celt/modes.rs:159`](/Users/nemurubaka/repos/mousiki/src/celt/modes.rs:159)
- [`src/celt/celt_decoder.rs:1857`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:1857)
- [`src/celt/celt_decoder.rs:1867`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:1867)
- [`src/celt/celt_decoder.rs:1882`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:1882)
- [`src/celt/celt_decoder.rs:1888`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:1888)
- [`src/silk/debug.rs:41`](/Users/nemurubaka/repos/mousiki/src/silk/debug.rs:41)

There are two different subcases here.

#### 4a. Static mode caches

`modes.rs` already has a lazy static-mode table. `celt_decoder.rs` adds a second `CanonicalModeCell` cache on top.

Observation:

- [`src/celt/modes.rs:605`](/Users/nemurubaka/repos/mousiki/src/celt/modes.rs:605) already exposes `opus_custom_mode_find_static(...)`
- [`src/celt/celt_decoder.rs:1864`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:1864) appears to duplicate the canonical 48k/960 cache path

Best-practice replacement:

- unify canonical-mode access behind one static-mode provider
- avoid maintaining two separate `UnsafeCell`-based caches
- if a crate is acceptable, use a no_std-friendly once-init primitive instead of hand-written atomics + `UnsafeCell`

Assessment:

- promising cleanup target
- medium effort

#### 4b. Debug/timer globals

`src/silk/debug.rs` uses `StaticMut<T>`, `unsafe impl Sync`, and raw access to process-global state.

Assessment:

- this is less urgent than codec-path cleanup
- removal is possible, but value is lower unless debug infrastructure is being redesigned

### 5. Self-referential owner + borrowed view patterns

Files:

- [`src/range.rs`](/Users/nemurubaka/repos/mousiki/src/range.rs)
- [`src/celt/celt_decoder.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs)
- [`src/celt/celt_encoder.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt_encoder.rs)

Representative sites:

- [`src/range.rs:353`](/Users/nemurubaka/repos/mousiki/src/range.rs:353)
- [`src/celt/celt_decoder.rs:2535`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:2535)
- [`src/celt/celt_decoder.rs:2554`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:2554)
- [`src/celt/celt_decoder.rs:2563`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:2563)
- [`src/celt/celt_encoder.rs:1554`](/Users/nemurubaka/repos/mousiki/src/celt/celt_encoder.rs:1554)

Current pattern:

- allocate owner storage
- convert owner to raw pointer
- build a borrowed view from that storage
- store both in one wrapper
- recover the owner in `Drop`

This is effectively hand-written self-referential data.

Examples:

- `RangeEncoder` owns a byte buffer and stores `EcEnc<'static>`
- `RangeDecoderState` owns a byte buffer and stores `EcDec<'static>`
- `OwnedCeltEncoder` / `OwnedCeltDecoder` own alloc structs while also storing borrowed encoder/decoder views

Why it exists:

- the C API model is “one allocation, many interior slices”
- the Rust port chose to keep a long-lived borrowed view instead of reconstructing it on demand

Why this matters:

- this is the most important structural `unsafe` in the repo
- the `transmute` in [`src/celt/celt_decoder.rs:2554`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:2554) and [`src/celt/celt_decoder.rs:2563`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:2563) is a direct consequence of this design

Best-practice replacements:

1. Prefer “owner only” wrappers plus temporary view construction
   - store only the owned allocation
   - construct `EcEnc` / `EcDec` / `OpusCustom*` views in methods like `with_encoder(...)` or `with_decoder(...)`
   - this is the cleanest unsafe-removal path, but changes call patterns
2. If long-lived borrowed views are really required, use a dedicated self-referential helper crate
   - candidates: `self_cell`, `ouroboros`
   - only if a new dependency is acceptable

Assessment:

- removable in principle
- high-value
- highest refactor cost

### 6. Re-borrowing full mutable buffers via raw pointers

File:

- [`src/celt/celt_decoder.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs)

Sites:

- [`src/celt/celt_decoder.rs:3980`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:3980)
- [`src/celt/celt_decoder.rs:4058`](/Users/nemurubaka/repos/mousiki/src/celt/celt_decoder.rs:4058)

Current pattern:

- take `channel_slice`
- derive `output = &mut channel_slice[...]`
- reconstruct a full read-only channel slice from the raw pointer for history reads

Why it exists:

- `comb_filter()` needs read access to the whole channel history and write access to the output window
- borrow checking makes that awkward with the current API shape

Best-practice replacement:

- redesign [`src/celt/celt.rs:134`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs:134) and [`src/celt/celt.rs:322`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs:322) to accept a single channel buffer plus an output range
- then use safe splitting APIs such as `split_at_mut` and range-local writes

Assessment:

- likely removable
- requires API redesign, not just a local edit

### 7. Intentional memmove semantics

File:

- [`src/celt/celt.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs)

Sites:

- [`src/celt/celt.rs:168`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs:168)
- [`src/celt/celt.rs:250`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs:250)
- [`src/celt/celt.rs:355`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs:355)
- [`src/celt/celt.rs:437`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs:437)

These use `ptr::copy` to preserve overlap-safe memmove behavior.

Assessment:

- lower priority
- may still be justifiable even after a full cleanup
- if revisited, prefer safe APIs only where overlap semantics are guaranteed equivalent

## Prioritized Removal Candidates

Recommended order:

1. Replace FFI math `unsafe`
   - `mini_kfft`
   - `kiss_fft`
2. Remove `static_mode` raw pointer storage from `CeltEncoderAlloc`
3. Refactor `mapping_matrix` to typed owned storage
4. Unify canonical/static mode initialization and remove duplicated lazy cells
5. Move `energy_masking` to a safe internal representation
6. Redesign range/celt owner wrappers to avoid self-referential borrowed views
7. Rework comb-filter caller/API shape to avoid raw slice reconstruction

## Suggested Refactor Notes

### Refactor A: `CeltEncoderAlloc::static_mode`

Current state:

- stores `Option<NonNull<OpusCustomMode<'static>>>`

Preferred state:

- store `Option<OpusCustomMode<'static>>`

Why this should work:

- [`OpusCustomMode`](/Users/nemurubaka/repos/mousiki/src/celt/types.rs:262) is `Clone`
- `opus_custom_mode_find_static()` already returns `OpusCustomMode<'static>` by value
- `static_mode` is only used inside [`src/celt/celt_encoder.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt_encoder.rs)

Expected result:

- removes `NonNull`
- removes `Box::into_raw`
- removes manual `Drop` recovery for this field

### Refactor B: `RangeEncoder` / `RangeDecoderState`

Current state:

- wrapper stores both owned storage and a long-lived borrowed coder view

Preferred direction:

- store only the buffer and enough snapshot/state data to reconstruct an `EcEnc` / `EcDec` view when needed
- or use a self-referential helper crate if persistent borrowed views are truly required

Design warning:

- local patching is not enough here
- this is a representation-level refactor

### Refactor C: `MappingMatrix`

Current state:

- canonical owned state is raw bytes

Preferred direction:

- canonical owned state should be typed fields
- conversion to/from bytes should be explicit and safe

This is the cleanest way to remove all current `unsafe` from the file.

## What Is Probably Not Worth Touching First

- [`src/silk/debug.rs`](/Users/nemurubaka/repos/mousiki/src/silk/debug.rs)
  This is mostly debug global-state plumbing, not a codec hot-path risk.

- [`src/celt/celt.rs`](/Users/nemurubaka/repos/mousiki/src/celt/celt.rs) overlap copies
  These encode deliberate memmove semantics. They are less suspicious than the self-referential lifetime code.

## Practical Next Step

If continuing this audit as code work, the best first patch is:

1. remove FFI math `unsafe`
2. replace `static_mode` raw pointer storage with owned `OpusCustomMode<'static>`
3. keep tests and parity checks together with the refactor

That should produce meaningful unsafe reduction with low regression risk before touching the harder self-referential wrappers.
