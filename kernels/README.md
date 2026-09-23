# Native Machine Kernels

These are standalone CPU-native kernels implementing the Native Machine ABI
v3. Each kernel is independently buildable and can be installed into a runtime
registry.

Included kernels:

- `reference-add-one`: `output[i] = input[i] + 1.0` (elementwise `f32`)
- `reference-relu`: `output[i] = max(input[i], 0.0)` (elementwise `f32`)
- `reference-matmul`: `C[M,N] = A[M,K] x B[K,N]` (row-major contiguous `f32`)
- `reference-xor-shift-add`: `output[i] = (input[i] ^ (input[i] << shift)) + add`
  (`u64`, wrapping)
- `neon-add-one`, `neon-relu`, `neon-matmul`, `neon-xor-shift-add`: the same
  contracts with NEON-accelerated hot loops on AArch64 (scalar fallback
  elsewhere). `neon-matmul` uses a 4x8 register-blocked FMA micro-kernel
  inside 64-column panels.
- `avx2-add-one`, `avx2-relu`, `avx2-matmul`, `avx2-xor-shift-add`: the same
  contracts with an AVX2 tier on x86_64 (runtime-detected, scalar fallback
  elsewhere, so they build and pass tests on any host). `avx2-matmul` uses a
  4x8 micro-kernel with separate mul+add (no FMA3 dependency), so it matches
  the scalar reference bitwise.

The `reference-*` crates are the scalar oracles; the `neon-*` crates declare
the NEON CPU feature bit and the `avx2-*` crates declare the AVX2 bit when
compiled for their target architecture, so the admission feature floor
rejects a compiled SIMD kernel on hosts without the feature. Differential
tests prove the SIMD tiers against the reference crates (bitwise where the
operation is exact, bounded-ULP where FMA contraction applies).

Build every kernel from the repository root:

```bash
just build-kernels
```

Run each kernel's unit tests (correctness against scalar oracles, plus ABI
rejection cases) with:

```bash
just test-kernels
```

The build command discovers each immediate child of `kernels/` containing a
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

`just kernel-demo` installs all four kernels, verifies their manifests, runs
ReLU, a 2x2 matmul, and xor/shift/add, and demonstrates typed rejection of
invalid input.

## ABI v3 contract

Each kernel exports `hologram_kernel_plugin_v3`, returning a descriptor that
declares the ABI version (3), name, input/output type IDs (`F32 = 1`,
`U64 = 2`), operation kind (`ELEMENTWISE = 1`, `MATMUL = 2`,
`XOR_SHIFT_ADD = 3`), required alignment, scratch bytes, required CPU
features, and the entry function. The runtime rejects ABI v2 plugins.

Invocation passes a bounded byte-oriented context: input/output/params
pointers with fixed-width `u64` byte lengths. Kernels validate null pointers,
length equality and element-size divisibility, overflow, alignment, and
parameter bounds before touching any buffer, and return a nonzero status code
on rejection.

Parameter encodings are fixed-width little-endian:

- elementwise: no parameters (`params_bytes = 0`);
- matmul: 12 bytes — `m: u32`, `k: u32`, `n: u32`;
- xor-shift-add: 16 bytes — `shift: u32`, reserved `u32` (must be zero),
  `add: u64`.

## Kernel contracts

### reference-matmul

The input region holds `A` (`M*K` values) immediately followed by `B`
(`K*N` values); the output region holds `M*N` values. All dimensions must be
nonzero and all buffer sizes must match the dimensions exactly. The kernel is
a scalar, deterministic triple loop with no heap allocation, hidden copies,
or implicit layout conversion. Matmul is not O(1) — the work is `M*N*K`
multiply-adds — and this reference exists to pin the contract down as the
oracle for any future optimized kernel.

### reference-xor-shift-add

`shift` must be in `0..=63`; larger shifts are rejected. The left shift
discards bits shifted past bit 63, and the final addition wraps modulo 2^64
(`wrapping_add`). With `shift = 0` the xor cancels and every output equals
`add`. This kernel represents the CPU-native integer path: pure ALU work over
caller-owned buffers with fully specified wrapping behavior.

All four kernels require no optional CPU features, declare no scratch memory,
and perform no heap allocation in their execution paths. They are reference
kernels intended to verify the plugin contract before adding
architecture-specific SIMD implementations.
