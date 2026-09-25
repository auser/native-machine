# Native Machine Architecture

Native Machine is a clean-room CPU-native execution runtime. A compiler
produces an immutable artifact; the runtime validates it, maps it, binds a
kernel table once, and executes bounded operations over caller-owned state.

## Non-negotiable contracts

- CPU instructions only. No accelerator or remote fallback.
- Fixed-width serialized and FFI fields. Never serialize `usize`, pointers, or vtables.
- Immutable artifact bytes after admission.
- No runtime repacking or whole-model materialization.
- Zero allocation in the steady-state operation path.
- Every queue, operation, and output has an explicit capacity.
- Scalar reference behavior exists before SIMD specialization.
- Dynamic plugins are loaded once, validated once, and retained for the session.
- Runtime artifact payloads are read through an immutable memory map; the
  portable borrowed parser remains available for tests and non-mmap callers.
- Session execution uses fixed-capacity caller-visible state and returns typed
  exhaustion errors instead of growing collections.
- Operation records use a fixed-width serialized representation and are parsed
  from borrowed bytes without constructing a runtime operation graph.
- Validated artifacts expose a deterministic full-byte SHA-256 identity for
  provenance; identity is intentionally separate from authorization and trust.
- Provenance is an explicit section kind rather than an implicit runtime value.
- Provenance payloads are canonicalized through `uor-addr` so equivalent JSON
  representations receive the same provenance identity.

## Kernel plugin model

The current ABI is version 3. The host resolves only
`hologram_kernel_plugin_v3`, validates the descriptor before retaining the
library, and rejects incompatible plugins — including ABI v2 plugins — without
semantic fallback. Future ABI versions must use an explicit entry point and
compatibility policy.

The descriptor declares, in fixed-width fields: ABI version, name, input type,
output type, operation kind, required alignment, scratch bytes, required CPU
features, and the entry function. Type IDs are `F32 = 1` and `U64 = 2`;
operation kinds are `ELEMENTWISE = 1`, `MATMUL = 2`, and `XOR_SHIFT_ADD = 3`;
CPU feature bits are `AVX2 = 1`, `NEON = 2`, and `FMA = 4` (x86 FMA3;
AArch64 fused multiply-add is covered by the NEON bit).
`MATMUL_ACT = 4` is the fused variant of matmul: its 16-byte params add a
`u32` activation kind (0 = none, 1 = ReLU) applied to the accumulators
before the single output store, so `C` is written once and never re-read.
Fusion belongs to the kernel, not the runtime: the record executor stays a
thin dispatcher, and a future compiler chooses between emitting separate
matmul and activation records or one fused record.

Invocation passes a bounded byte-oriented context:

```text
input: *const u8, input_bytes: u64
output: *mut u8, output_bytes: u64
params: *const u8, params_bytes: u64
```

No Rust slices, references, `usize`, vtables, or heap-owned structures cross
the boundary. Parameters use fixed-width little-endian encodings: matmul takes
three `u32` dimensions (`m`, `k`, `n`); xor-shift-add takes a `u32` shift, a
reserved zero `u32`, and a `u64` addend.

The registry exposes typed, allocation-free dispatch methods (`run_f32`,
`run_matmul`, `run_u64`). Buffers are caller-owned and passed through without
repacking; for matmul, `A` and `B` must form one contiguous row-major region
so the single input pointer can cover both operands. The success path performs
no heap allocation, verified by an allocation-counting test.

## Reference kernels

Four standalone kernels under `kernels/` implement ABI v3: add-one and ReLU
(elementwise `f32`), matmul (`f32`), and xor-shift-add (`u64`).

Alongside them, `neon-*` crates implement the same contracts with
NEON-accelerated hot loops on AArch64 (scalar fallback elsewhere), and
`avx2-*` crates add an AVX2 tier on x86_64 with runtime detection and scalar
fallback elsewhere, so every crate builds and passes its tests on any host.
SIMD descriptors declare their feature bit (NEON or AVX2) when compiled for
the matching architecture, so the admission feature floor rejects a compiled
SIMD kernel on hosts without the feature. Differential tests prove the SIMD
tiers against the reference oracles: bitwise for exact operations,
bounded-ULP for `neon-matmul`, whose FMA contraction may differ from the
scalar reference in the last mantissa bit. `neon-matmul` uses a 4x8
register-blocked micro-kernel inside 64-column panels, vectorizing across
columns instead of reducing along `K`; `avx2-matmul` uses separate mul+add
(no FMA3) and matches the reference bitwise.

Callers that dispatch the same kernel repeatedly can resolve the name once
with `PluginRegistry::resolve` and dispatch through the returned
`KernelHandle`, skipping the per-call name lookup while keeping full
contract validation.

Matmul is deliberately a scalar, deterministic triple loop. It is not O(1):
`C[M,N] = A[M,K] x B[K,N]` costs `M*N*K` multiply-adds, so no constant-time
implementation is possible, and this reference makes that cost explicit rather
than hiding it behind a library call. It exists to pin down the ABI contract —
layout, parameter encoding, dimension validation, and buffer bounds — before
any blocked, tiled, or SIMD implementation is admitted. A future optimized
kernel must produce bit-identical results against this oracle.

Xor-shift-add (`output[i] = (input[i] ^ (input[i] << shift)) + add`, with
wrapping addition and `shift` in `0..=63`) represents the CPU-native integer
path: pure ALU work with no floating-point, no memory beyond the caller's
buffers, and fully specified wrapping behavior. It exercises the `U64` type
ID and the parameterized operation contract.

## Kernels versus compiled plans

A kernel is a single dynamically loaded function with a fixed operation
contract, admitted once and retained for the session. A compiled plan is the
immutable artifact the runtime validates, maps, and executes as a sequence of
fixed-width operation records; plugin records in a plan dispatch by index into
the admitted kernel table. Kernels provide primitives; plans compose them.
The ABI governs kernels, not plans: changing a plan never changes a kernel's
contract.

Plan records are 16 bytes. Built-in opcodes cover Copy and AddScalar; plugin
opcodes dispatch elementwise (`2`), matmul (`3`: kernel index plus `m`, `k`,
`n` as `u32`), and fused matmul + activation (`4`: index plus `u16`
dimensions, a `u16` activation kind, and a reserved zero word). Plans are
chains: each record consumes the previous record's output, so a matmul
record consumes `m*k + k*n` values (A||B row-major contiguous) and produces
`m*n`, and compilation validates every link before execution. The session
arena is `f32`-typed, so `u64` kernels (xor-shift-add) dispatch through the
typed registry methods rather than plan records. `native-machine run`
compiles the artifact's operation section once and executes the compiled
plan, which also means multi-record artifacts now chain rather than
re-reading the original input per record.

Fusion exists at both levels. Kernel-level fusion is a new operation kind
(`MATMUL_ACT`): the activation is applied to accumulators before the single
output store. Plan-level fusion is a two-phase design: `compile_plan`
validates the record stream once, groups it into segments (fused built-in
runs and plugin dispatches), extracts operands, and stamps the plan with a
canonical UOR content address via `uor-addr`; `execute_compiled_plan` then
dispatches pre-digested segments in O(1) per operation with no parsing, no
validation, and no heap allocation. Fused built-in segments run as
strip-mined SIMD passes (first AddScalar folded into the input copy, the
rest in place per strip), bitwise identical to separate passes while
intermediates stay cache-resident. Compilation and addressing cost ~20us
once per plan and amortize to zero; measured execution reaches parity with
raw unfused native passes, and zero-copy plugin dispatch keeps artifact
record execution at ~1.0x native.

Compiled plans are deduplicated through a bounded `PlanCache` keyed by their
UOR address: `get_or_compile` compiles and inserts once per distinct plan
(content-addressed deduplication), and `get` retrieves a plan by address
with a fixed scan and no allocation. The cache is caller-owned with a hard
capacity; a full cache is a typed error, never a silent eviction.

## In-process plugin trust

Plugins are `dlopen`ed shared libraries running in the runtime's address
space. Manifest hashing and descriptor validation provide integrity and
compatibility checking, not isolation: a malicious or defective plugin can
corrupt or crash the host process. Admission is therefore explicit
(`kernel install`), verified by content hash on every load, bounded by a
determinism probe, and limited to plugins the user deliberately supplies.

## Benchmarks

`just bench` installs the release-mode reference kernels and measures typed
dispatch against in-binary native baselines in a release build. Expected
shape: a fixed dispatch overhead (tens of nanoseconds, visible at size 1)
that amortizes to a ratio near 1.0x at steady-state sizes, zero heap
allocations per dispatch, and admission/identity costs (SHA-256 artifact
identity, UOR provenance address, manifest verification) reported separately
as one-time per-load operations. The benchmark fails if any typed dispatch
allocates on the success path.

## UOR integration

`uor-addr` supplies stable identities for manifests and operation plans. The
runtime treats the address as provenance, not authorization. The prism/partition
integration remains an adapter boundary until its actual API contract is known.
