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

The current ABI is version 2. It includes explicit input/output type IDs,
bounded scratch requirements, and required CPU feature bits. The host resolves
only `hologram_kernel_plugin_v2`, validates the descriptor before retaining the
library, and rejects incompatible plugins without semantic fallback. Future ABI
versions must use an explicit entry point and compatibility policy.

## UOR integration

`uor-addr` supplies stable identities for manifests and operation plans. The
runtime treats the address as provenance, not authorization. The prism/partition
integration remains an adapter boundary until its actual API contract is known.
