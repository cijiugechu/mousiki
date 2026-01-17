# Fuzzing Status (Rust vs C Opus)

This document captures the current fuzzing state, recent changes, and how to
run the C fuzzer inside Docker (linux/amd64). It also records the Rust crash
input and its status on the C implementation.

## Current changes in this repo

Committed earlier:
- Seed corpus for `decode_fuzzer` in `fuzz/corpus/decode_fuzzer/`
  (`single_packet.bin`, `two_packets_fec.bin`, `packet_then_loss.bin`).
- `fuzz/.gitignore` now only tracks the three seed files and ignores
  generated corpus/artifacts.
- `Cargo.toml` excludes `fuzz` from the workspace; `fuzz/Cargo.toml` has an
  empty `[workspace]` to avoid workspace mismatch.
- `README.md` documents fuzzing as manual/on-demand and uses nightly one-shot.
- `FUZZING_PLAN.md` marks step 4 and step 5 as done.

Uncommitted files created during C fuzzing:
- `Dockerfile.opus-fuzz` (new file).
- `opus_decode_fuzzer` (built C fuzzer binary in repo root).
- `opus-c/build-linux/` (CMake build dir in the `opus-c` submodule).
- `fuzz/artifacts/opus_decode_fuzzer/` (libFuzzer artifacts dir; ignored).

## Rust fuzzing (cargo-fuzz)

Rust fuzzing hit a crash in the ported decoder.

- Crash location: `src/celt/mdct.rs:242`  
  Assertion: `output.len() >= half_overlap + n2`.
- Crash input (file):  
  `fuzz/artifacts/decode_fuzzer/crash-213fac479ee66d1483486fd15198b7d1e976ca25`
- Input bytes (hex):  
  `00 00 00 0f 00 08 00 00 b8 7c 35 21 75 e5 67 d5 1c ac a2 54 fa ff bf`
- Input bytes (base64):  
  `AAAADwAIAAC4fDUhdeVn1RysolT6/78=`

Reproduce (Rust):
```bash
rustup run nightly cargo fuzz run decode_fuzzer \
  fuzz/artifacts/decode_fuzzer/crash-213fac479ee66d1483486fd15198b7d1e976ca25
```

## C vs Rust comparison (ctests)

### clt_mdct_backward (MDCT inverse)
- C test: `ctests/mdct_backward_compare.c`
- Rust test: `src/celt/mdct.rs` (`mdct_backward_compare_output`)
- Compare run:
  ```bash
  cmake -S ctests -B ctests/build
  cmake --build ctests/build
  ctests/build/celt_mdct_backward_compare > /tmp/mdct_backward_c.txt

  cargo test -p mousiki --lib mdct_backward_compare_output -- --nocapture \
    > /tmp/mdct_backward_rs.txt

  python3 - <<'PY'
  import re
  def load(path):
      out = {}
      with open(path, "r") as f:
          for line in f:
              m = re.search(r"mdct_backward_out\\[(\\d+)\\]=([-+0-9.eE]+)", line)
              if m:
                  out[int(m.group(1))] = float(m.group(2))
      return out
  c = load("/tmp/mdct_backward_c.txt")
  rs = load("/tmp/mdct_backward_rs.txt")
  max_diff = 0.0
  max_idx = None
  for idx in c:
      diff = abs(c[idx] - rs[idx])
      if diff > max_diff:
          max_diff = diff
          max_idx = idx
  print("count", len(c), "max_diff", max_diff, "at", max_idx)
  PY
  ```
- Window: uses `mode->window` / `mode.window` from the 48 kHz / 120 static mode.
- Result: 1020 samples compared, max abs diff `~4.74e-5` (idx 118).

### celt_synthesis (synthesis path)
- C test: `ctests/celt_synthesis_compare.c`
- Rust test: `src/celt/celt_decoder.rs` (`celt_synthesis_compare_output`)
- Compare run:
  ```bash
  cmake -S ctests -B ctests/build
  cmake --build ctests/build
  ctests/build/celt_synthesis_compare > /tmp/celt_synthesis_c.txt

  cargo test -p mousiki --lib celt_synthesis_compare_output -- --nocapture \
    > /tmp/celt_synthesis_rs.txt

  python3 - <<'PY'
  import re
  def load(path):
      out = {}
      with open(path, "r") as f:
          for line in f:
              m = re.search(r"celt_synthesis_out\\[(\\d+)\\]=([-+0-9.eE]+)", line)
              if m:
                  out[int(m.group(1))] = float(m.group(2))
      return out
  c = load("/tmp/celt_synthesis_c.txt")
  rs = load("/tmp/celt_synthesis_rs.txt")
  max_diff = 0.0
  max_idx = None
  for idx in c:
      diff = abs(c[idx] - rs[idx])
      if diff > max_diff:
          max_diff = diff
          max_idx = idx
  print("count", len(c), "max_diff", max_diff, "at", max_idx)
  PY
  ```
- Inputs: static 48 kHz / 120 mode window, `LM=0`, `C=1`, `CC=1`, `start=0`,
  `eff_end=mode.effEBands`, `downsample=1`, `silence=false`, linear `X`/`oldBandE`.
- Result: 180 samples compared, max abs diff `~1.24e-4` (idx 61).

## C fuzzing in Docker (linux/amd64)

The C fuzzer was built and run in an `ubuntu:22.04` container with LLVM 17.
It ran for 30 seconds without a crash.

### Build the Docker image (linux/amd64)
```bash
docker build --platform=linux/amd64 -f Dockerfile.opus-fuzz -t opus-fuzz .
```

### Build and run the C fuzzer
```bash
docker run --rm --platform=linux/amd64 \
  -v /Users/nemurubaka/repos/mousiki:/repo \
  -w /repo opus-fuzz bash -lc '\
    cmake -S opus-c -B opus-c/build-linux \
      -DOPUS_BUILD_SHARED_LIBRARY=OFF \
      -DOPUS_BUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-17 && \
    cmake --build opus-c/build-linux -j && \
    clang-17 -O1 -g -fsanitize=fuzzer,address \
      -Iopus-c/include -Iopus-c/src -Iopus-c/build-linux \
      opus-c/tests/opus_decode_fuzzer.c \
      opus-c/build-linux/libopus.a \
      -o opus_decode_fuzzer && \
    mkdir -p /repo/fuzz/artifacts/opus_decode_fuzzer && \
    ./opus_decode_fuzzer \
      -artifact_prefix=/repo/fuzz/artifacts/opus_decode_fuzzer/ \
      -max_len=1000000 \
      fuzz/corpus/decode_fuzzer \
      -max_total_time=30 \
  '
```

Note: The libFuzzer build in this container did not recognize
`-options_file=opus-c/tests/opus_decode_fuzzer.options`, so `-max_len=1000000`
is passed directly.

### Run the Rust crash input against C
```bash
docker run --rm --platform=linux/amd64 \
  -v /Users/nemurubaka/repos/mousiki:/repo \
  -w /repo opus-fuzz bash -lc '\
    ./opus_decode_fuzzer \
      /repo/fuzz/artifacts/decode_fuzzer/crash-213fac479ee66d1483486fd15198b7d1e976ca25 \
      -runs=1 \
  '
```

Result: the C fuzzer completed the single input without crashing.
