# Rust port of the C opus library

The directory where the Go version is located is `/opus_c`.

# Porting strategy

- Before porting functions and structures, first find out what their lifetime is.

- Always port the implementation and tests together.
 
- Variable, type, and function naming in Rust needs to follow Rust conventions.