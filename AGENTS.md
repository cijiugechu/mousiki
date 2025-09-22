# Rust port of the go opus library

The directory where the Go version is located is `/opus-go`.

# Porting strategy

- The Rust version needs to take into account the use case of no_std and does not allow any dynamic memory allocation, including but not limited to vec, hashmap, box, etc.

- In scenarios where heap allocation is used in Go, you should first consider whether it can be converted to using pre-allocated types such as array and struct; if the size of the allocated type is uncertain at runtime, you should allocate its maximum allocable size on the stack and then use mutable borrowing.

- Always port the implementation and tests together.
 
- Variable, type, and function naming in Rust needs to follow Rust conventions