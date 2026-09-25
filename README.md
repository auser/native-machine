# Native Machine

Clean-room prototype for a zero-copy CPU-native operation runtime.

It demonstrates:

- a versioned C-compatible kernel plugin ABI;
- a statically linked built-in kernel;
- dynamically loaded Rust kernels;
- no runtime model repacking;
- deterministic content addressing of an operation plan with `uor-addr`.

The plugin boundary is intentionally narrow. Plugins receive borrowed input and
output buffers and return a status code. A production artifact will add signed
manifests, CPU-feature requirements, fixed scratch declarations, alignment, and
operation bounds before loading a plugin.

The loader is for initialization, not for dispatching every inner operation.
Load and validate a plugin once, retain its function table, and call the typed
function pointer in the hot path. For the fastest path, the compiler should
also be able to statically link the same kernel ABI into the executable.

## Deliberate development workflow

This project is developed through small pull requests into `main`. Do not push
feature work directly to `main`. Every pull request must pass `just ci`; the
protected `main` branch is updated through GitHub's merge queue.

The intended loop is:

```text
git switch main
git pull --ff-only origin main
git switch -c feat/your-change
# make one focused change
just ci
git add .
git commit -m "feat: describe the change"
git push -u origin feat/your-change
# open a PR, wait for CI, then enqueue it in GitHub
```

The repository must require pull requests, passing CI, and the merge queue for
`main`. Enable the `CI / rust` check for both pull requests and `merge_group`
events in GitHub branch protection. See
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## First run

Install Rust, `just`, and GitHub CLI (`gh`), then authenticate `gh`:

```text
just install-hooks
just init
just doctor
just inspect-host
just build-all
```

`just install-hooks` enables the checked-in pre-commit hook. When a commit
contains Rust files, the hook runs rustfmt and re-stages those files so the
formatting is included in that same commit. It refuses partially staged Rust
files to avoid adding unstaged hunks unexpectedly.

`just` is the canonical interface; use `just` with no arguments to see all
available commands.

## Run

```text
just run
just plugin
just run-plugin
just install-kernels
just kernel-list
just kernel-demo
```

`just install-kernels` initializes the local runtime, builds every standalone
crate under `kernels/`, and installs each resulting shared library. It handles
the platform-specific `.dylib`, `.so`, or `.dll` filename automatically. Run
`just kernel-list` afterward to verify the installed manifests and plugin
identities.

`just kernel-demo` builds and installs the reference kernels, verifies their
manifests, runs ReLU, a 2x2 matmul, and xor/shift/add over `u64` values, and
demonstrates typed rejection of invalid dimensions, shifts, and buffer sizes.

`just bench` installs the release-mode reference kernels and benchmarks typed
dispatch against in-binary native baselines: per-call dispatch overhead at
size 1, throughput at steady-state sizes, the artifact record dispatch path,
admission and identity costs (SHA-256 artifact identity and UOR provenance
address), and a measured zero heap allocations across thousands of typed
dispatches.

Run `just` to see every available development command. `just ci` is the local
release gate.

The planned first-run experience is documented in
[`docs/FIRST_RUN.md`](docs/FIRST_RUN.md). It will initialize a local workspace,
inspect the host, run a self-test, and refuse unverified plugins or artifacts.

The plugin path can also be exercised directly:

```text
native-machine kernel test target/debug/libkernel_plugin.dylib
native-machine init
native-machine kernel install target/debug/libkernel_plugin.dylib
native-machine kernel list
```

Installation writes an adjacent `.manifest.toml` containing the plugin ABI,
name, size, and SHA-256 identity. `kernel list` verifies those manifests and
refuses to admit a changed or unmanifested plugin.

The ABI descriptor is version 3 and declares the ABI version, name,
input/output buffer types (`F32 = 1`, `U64 = 2`), operation kind
(`ELEMENTWISE = 1`, `MATMUL = 2`, `XOR_SHIFT_ADD = 3`), required alignment,
scratch bytes, and required CPU features. The loader resolves only
`hologram_kernel_plugin_v3` and rejects ABI v2 plugins and incompatible entry
points. Invocation passes a bounded byte-oriented context
(`input`/`output`/`params` pointers with fixed-width `u64` byte lengths); the
registry exposes typed, allocation-free dispatch methods (`run_f32`,
`run_matmul`, `run_u64`) that validate dimensions, alignment, buffer bounds,
and parameter encodings before calling the kernel. A future kernel may require
AVX2 or another explicitly supported feature and will be rejected on
incompatible hosts.

Runtime execution enforces `max_artifact_bytes` from the effective TOML
configuration before mapping an artifact. This bounds admission separately
from the fixed session arena and plugin size limit.

Create and validate a local artifact fixture:

```text
native-machine artifact create-fixture ./fixture.nm
native-machine artifact validate ./fixture.nm
native-machine artifact inspect ./fixture.nm
native-machine run --artifact ./fixture.nm --input ./input.f32le
```

Plans can also be written in a small line-based source format and compiled
to an artifact (see `docs/ARCHITECTURE.md` for the grammar):

```text
printf 'input 8\nmatmul avx2-fma-matmul 2 2 2\nelementwise reference-relu\n' > plan.txt
native-machine artifact create-plan plan.txt plan.nm
native-machine run --artifact plan.nm --input ./matmul-input.f32le
```

`input.f32le` is a caller-created file containing little-endian `f32` values.
The runtime bounds it to the fixed session arena before execution.

On Linux use `libkernel_plugin.so`; on Windows use `kernel_plugin.dll`.

## Build, verify, package, and release

```text
just fmt          # check formatting
just fmt-fix      # apply formatting
just check        # compile all workspace targets
just test         # run all tests
just lint         # run Clippy with warnings denied
just build        # debug build of all targets
just build-kernels # build every standalone kernel crate
just build-kernels-release # build every kernel in release mode
just test-kernels  # run every standalone kernel's unit tests
just install-kernels # build and install every standalone kernel
just kernel-demo  # install kernels and run the end-to-end kernel demo
just bench        # benchmark kernel dispatch against native baselines
just build-all    # CI plus release builds and plugin build
just package      # build crates and package them
just release-check
just release version=0.1.0
```

`just release` requires a clean, merged `main` checkout. It creates and pushes
a semantic-version tag; the GitHub Actions release workflow then publishes the
GitHub release and release binary. Releases should only be created after the
change has passed through the pull request and merge queue.
