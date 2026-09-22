# Native Machine Kernels

These are standalone CPU-native kernels implementing the Native Machine ABI
v2. Each kernel is independently buildable and can be installed into a runtime
registry.

Included kernels:

- `reference-add-one`: `output[i] = input[i] + 1.0`
- `reference-relu`: `output[i] = max(input[i], 0.0)`

Build every kernel from the repository root:

```bash
just build-kernels
```

The command discovers each immediate child of `kernels/` containing a
`Cargo.toml`, so new standalone kernel crates are included automatically.
Each kernel manifest must contain an empty `[workspace]` table so Cargo treats
it as an independent crate rather than an undeclared member of the root
workspace.

Install the resulting platform-specific shared library with the runtime:

```bash
cargo run -p native-machine -- kernel install PATH_TO_SHARED_LIBRARY
cargo run -p native-machine -- kernel list
cargo run -p native-machine -- kernel test PATH_TO_SHARED_LIBRARY
```

Both kernels use `f32` input and output, require no optional CPU features, use
no scratch memory, and perform no heap allocation in their execution loops.
They are reference kernels intended to verify the plugin contract before
adding architecture-specific SIMD implementations.
