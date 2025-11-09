# Rust port of the C opus library

The directory where the C version is located is `/opus-c`(Which is a git submodule, If you encounter a 403 error when pulling this submodule, you can download the code and unzip it from [this link](https://github.com/xiph/opus/archive/refs/heads/main.zip)).

# Porting strategy

- Before porting functions and structures, first find out what their lifetime is.

- In scenarios where heap allocation is used in C, you should first consider whether it can be converted to using pre-allocated types such as array and struct; if the size of the allocated type is uncertain at runtime, you should allocate its maximum allocable size on the stack and then use mutable borrowing.

- Always port the implementation and tests together.
 
- Variable, type, and function naming in Rust needs to follow Rust conventions.

- When using constants for mathematical operations, give priority to whether rust core provides them.

- Run `cargo check --all-features`, `cargo test --all-features` and `cargo clippy` automatically after making code changes, you need to make sure no errors.