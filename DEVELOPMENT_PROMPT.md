# Development Kickoff Prompt

```text
You are continuing development of Native Machine, a clean-room Rust project
for compiling bounded CPU-native operation artifacts and executing them with
zero-copy immutable data and zero steady-state heap allocation.

Read first: rust-best-practices.md, docs/ARCHITECTURE.md, docs/FIRST_RUN.md,
docs/ROADMAP.md, DEVELOPMENT_PROMPT.md, and the current source tree.

Do not create a toy demo, speculative pseudocode, or unrelated framework.
Implement the next production vertical slice in the existing workspace.

Current priority:
1. Add a bounded provenance-section writer containing source, compiler, layout,
   and transform metadata; validate its JSON shape before canonicalizing it.
2. Add allocation-counting tests for the complete executor path.
3. Add SIMD kernel dispatch and differential tests against the scalar path.

Constraints:
- use Rust and `just`, not Make;
- use clap derive for CLI commands;
- defaults -> config file -> environment -> CLI precedence;
- no panic, unwrap, expect, todo, or unbounded fallback;
- fixed-width fields at serialization and FFI boundaries;
- no runtime repacking or whole-artifact copies;
- preserve a clear scalar reference implementation;
- keep runtime crates allocation-free after initialization;
- check every capacity and range;
- use `uor-addr` for artifact and manifest identity;
- keep prism/partition integration behind a trait or adapter until verified;
- preserve built-in scalar execution while adding optimized/plugin paths;
- keep files under 1,500 lines;
- update docs, tests, roadmap, and just commands with behavior changes.

Before editing, inspect the tree and git status, state files to change, and
identify constraint conflicts. After editing, run `just fmt-fix`, `just lint`,
`just check`, and `just test`; report anything that could not run. Do not claim
production readiness until the artifact validator, fixed arena, allocation test,
plugin admission, and benchmark gates exist.
```
