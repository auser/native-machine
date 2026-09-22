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
- [ ] allocation-counting test
- [ ] deterministic trace and byte-read accounting

## Phase 4 — native kernels

- [ ] one-time scalar/NEON/AVX2/VNNI dispatch
- [ ] SIMD differential tests
- [ ] alignment and feature-floor validation
- [ ] kernel head-to-head benchmarks

## Phase 5 — compiler

- [ ] source-independent intermediate representation
- [ ] route/plane lowering adapter
- [ ] packed operation emission
- [ ] behavioral certification
- [ ] atomic artifact publication
