# First-Run Setup

The first run must be safe, deterministic, reversible, and useful even when no
model or external kernel is installed. It must not download a model, execute
untrusted plugin code, or silently modify a user's global environment.

## User experience

```text
$ native-machine init

Native Machine first-run setup
──────────────────────────────
Workspace: /Users/ari/.native-machine

Host
  architecture: aarch64
  operating system: macOS
  detected kernels: scalar, neon

Created
  config:    ~/.native-machine/config.toml
  cache:     ~/.native-machine/cache/
  artifacts: ~/.native-machine/artifacts/
  plugins:   ~/.native-machine/plugins/
  logs:      ~/.native-machine/logs/

Running self-test ... passed
Running allocation contract test ... passed

Native Machine is ready.
Next steps:
  native-machine inspect-host
  native-machine kernel list
  native-machine artifact validate path/to/artifact
```

The command is idempotent. A second invocation reports existing paths and does
not overwrite configuration or artifacts. This document describes the intended
experience; the sections below mark what is implemented today.

## Commands

The CLI exposes:

```text
native-machine init [--root PATH] [--force]
native-machine inspect-host
native-machine doctor
native-machine config show
native-machine kernel list
native-machine kernel inspect PATH
native-machine kernel test PATH
native-machine kernel install PATH
native-machine kernel demo
native-machine kernel bench
native-machine artifact inspect PATH
native-machine artifact validate PATH
native-machine artifact create-fixture PATH
native-machine artifact create-plan SOURCE OUTPUT
native-machine run --artifact PATH --input PATH
```

`init` creates only local directories and a configuration file. `doctor`
performs checks without modifying anything. `kernel install` is the only
command that admits a plugin. It loads the library, resolves the versioned
ABI v3 entry point, validates the descriptor, runs a bounded determinism
self-test, and only then copies it into the managed plugin directory with an
adjacent `<name>.manifest.toml` recording the plugin's name, ABI version,
size, and SHA-256 identity. `kernel list` re-verifies those manifests on
every load. `kernel demo` and `kernel bench` exercise the installed kernels;
`just kernel-demo` and `just bench` drive them end to end.

## Directory layout

```text
<root>/
├── config.toml
├── cache/
├── artifacts/
├── plugins/
│   ├── libexample.dylib
│   └── libexample.dylib.manifest.toml
├── logs/
└── state/
```

All writes use a temporary file in the same directory followed by validation
and an atomic rename. Existing files are never truncated in place.

## Configuration precedence

Values resolve in this order:

```text
built-in defaults → config.toml → environment → CLI flags
```

The effective configuration should be printable with:

```text
native-machine config show
```

Secrets, if a future feature requires them, must never be printed. Paths are
normalized before use and must remain beneath the configured root unless the
user explicitly opts into an external path.

## Host inspection

Host inspection is read-only and records:

- operating system and architecture;
- pointer width and endianness;
- available CPU feature set;
- supported kernel families;
- page size and alignment capabilities;
- maximum configured artifact and session sizes;
- runtime version and artifact-format version.

Feature detection happens once. The result is converted into a concrete kernel
table and stored in the session. Hot loops do not inspect features.

## Plugin admission

First run must not automatically search arbitrary directories or load arbitrary
dynamic libraries. A plugin becomes eligible only after:

1. the user supplies its path;
2. the file is opened read-only;
3. its manifest is parsed with bounds checks;
4. its ABI and CPU floor are checked;
5. its content address is calculated;
6. its self-test runs in a bounded process or isolated admission path;
7. the user explicitly confirms installation, unless `--yes` is provided.

The plugin is then copied to a content-addressed location. The original file is
never executed again during registration.

## Failure behavior

Every failure is typed and actionable. `init` may safely be rerun after a
failure. Partial directories are harmless; partial artifacts and plugin
registrations are never admitted. The command exits nonzero without panicking.

## Implementation order

1. Add `Cli` and subcommands with `clap` derive.
2. Add `Config` with defaults, TOML loading, environment, and CLI precedence.
3. Implement idempotent `init`.
4. Implement read-only `inspect-host`.
5. Implement `doctor` and the startup self-test.
6. Add plugin manifests and explicit admission.
7. Add artifact inspection and validation.
