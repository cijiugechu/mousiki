# ctests

This directory holds standalone C tests that exercise the upstream C Opus
implementation without modifying the `opus-c` sources. The goal is to compare
C and Rust outputs for the same inputs (for example, `compute_band_energies`),
so the Rust port can be validated against the reference behavior.

## Running the tests

Use the helper script:

```sh
./ctests/run.sh
```

Common options:

```sh
./ctests/run.sh --clean -j 8
./ctests/run.sh --cmake-arg -DOPUS_CUSTOM_MODES=ON
./ctests/run.sh -- --verbose
```

Run `./ctests/run.sh --help` to see all available options.
