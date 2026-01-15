# DRED discussion notes

## Feature behavior
- Enabling the `dred` feature only compiles DRED-related Rust modules and embedded weight data.
- There is no code that downloads artifacts from Xiph or anywhere else at build/runtime.
- The weights are included as static arrays in Rust source files.

## Size impact
- `src/dred_rdovae_dec_data.rs`: 17,918,340 bytes (~17.1 MiB)
- DRED-related source total: 17,971,502 bytes (~17.1 MiB)
- This already exceeds a 10 MiB crate-size budget, even if the `dred` feature is not enabled.

## Options to keep the crate under 10 MiB
1) Move weights into a separate crate (e.g., `mousiki-dred-weights`) so the main crate stays small.
2) Compress the weights (e.g., zstd) and `include_bytes!` them, then decompress at runtime.
3) Remove embedded weights and load them from an external file provided by the user.
