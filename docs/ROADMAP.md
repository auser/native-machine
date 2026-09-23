# Production Roadmap

## Phase 1 — executable foundation

- [x] versioned dynamic kernel ABI
- [x] built-in scalar kernel
- [x] dynamic Rust kernel example
- [x] content-addressed operation-plan identity
- [x] fixed-width FFI length field
- [x] workspace checks and lint configuration
- [x] first-run setup contract

## Phase 2 — artifact contract

- [x] versioned binary header and checked section table
- [x] typed artifact validator
- [x] borrowed section views
- [x] artifact fixture writer
- [x] section overlap validation
- [x] mmap-backed immutable artifact view
- [ ] manifest admission and provenance

## Phase 3 — zero-allocation execution

- [ ] fixed session arena and layout planner
- [x] fixed session arena and layout planner
- [x] bounded scalar operation executor
- [x] scalar reference executor
- [x] fixed-width operation record parser
- [x] artifact-independent record executor
- [x] CLI end-to-end mapped artifact execution
- [x] strict unknown-section policy
- [x] caller-provided executor scratch path
- [x] deterministic manifest identity groundwork
- [x] full-byte artifact identity
- [x] canonical provenance identity path
- [x] allocation-counting test
- [ ] deterministic trace and byte-read accounting

## Phase 4 — native kernels

- [x] ABI v3 byte-oriented kernel context and descriptor
- [x] typed registry dispatch (elementwise, matmul, xor-shift-add)
- [x] reference kernel suite: add-one, relu, matmul, xor-shift-add
- [x] alignment and feature-floor validation
- [x] zero-allocation typed dispatch test
- [x] NEON kernel tier with descriptor-declared CPU feature floor
- [x] AVX2 kernel tier with runtime detection and scalar fallback
- [x] AVX2+FMA3 matmul tier with descriptor-declared FMA feature floor
- [x] K-blocked panel matmul (L1-resident B slices, C accumulates in memory)
- [x] fused matmul + activation kernel (`MATMUL_ACT`)
- [x] chained plan executor with built-in operation fusion
- [x] zero-copy plugin record dispatch
- [x] SIMD differential tests (NEON vs scalar reference oracles)
- [x] resolve-once kernel handles for hot-loop dispatch
- [ ] one-time scalar/NEON/AVX2/VNNI dispatch in built-in operations
- [x] kernel head-to-head benchmarks (`just bench`, native baseline ratios)

## Phase 5 — compiler

- [ ] source-independent intermediate representation
- [ ] route/plane lowering adapter
- [ ] packed operation emission
- [ ] behavioral certification
- [ ] atomic artifact publication
