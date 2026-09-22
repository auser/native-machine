set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# List the available development commands.
default:
    @just --list

# Check Rust formatting without changing files.
fmt:
    cargo fmt --all -- --check

# Format all Rust source files in place.
fmt-fix:
    cargo fmt --all

# Enable the repository-managed Git hooks for this clone.
install-hooks:
    git rev-parse --is-inside-work-tree >/dev/null
    git config core.hooksPath .githooks
    @echo "installed repository hooks from .githooks"

# Type-check every workspace target.
check:
    cargo check --workspace --all-targets

# Run all workspace tests.
test:
    cargo test --workspace

# Run Clippy for every target and feature with warnings denied.
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Build every workspace target in debug mode.
build:
    cargo build --workspace --all-targets

# Run CI, build the plugin, and produce release binaries.
build-all: ci plugin
    cargo build --workspace --release

# Build and validate publishable crate packages.
package: build-all
    cargo package -p native-machine --allow-dirty
    cargo package -p kernel-plugin --allow-dirty

# Build the dynamic kernel plugin in debug mode.
plugin:
    cargo build -p kernel-plugin

# Run the Native Machine CLI.
run:
    cargo run -p native-machine

# Initialize the local Native Machine workspace.
init:
    cargo run -p native-machine -- init

# Check and repair the local Native Machine workspace.
doctor:
    cargo run -p native-machine -- doctor

# Print the detected host and CPU capabilities.
inspect-host:
    cargo run -p native-machine -- inspect-host

# List installed kernel plugins and their verification status.
kernel-list:
    cargo run -p native-machine -- kernel list

# Create an artifact fixture at the requested path.
artifact-create path="fixture.nm":
    cargo run -p native-machine -- artifact create-fixture {{path}}

# Validate an artifact fixture at the requested path.
artifact-validate path="fixture.nm":
    cargo run -p native-machine -- artifact validate {{path}}

# Inspect an artifact fixture at the requested path.
artifact-inspect path="fixture.nm":
    cargo run -p native-machine -- artifact inspect {{path}}

# Build and exercise the dynamic kernel plugin.
run-plugin: plugin
    case "$(uname -s)" in \
        Darwin) shared_suffix=dylib ;; \
        MINGW*|MSYS*|CYGWIN*) shared_suffix=dll ;; \
        *) shared_suffix=so ;; \
    esac; \
    cargo run -p native-machine -- kernel test "target/debug/libkernel_plugin.${shared_suffix}"

# Run the formatting, lint, type-check, and test gates.
ci: fmt lint check test

# Run every CI, build, and packaging release gate.
release-check: ci build-all package

# Tag and push a verified release from a clean main branch.
release version="0.1.0": release-check
    test "$(git branch --show-current)" = "main"
    test -z "$(git status --porcelain)"
    git tag "v{{version}}"
    git push origin "v{{version}}"

# Create a clean-room source archive of the repository.
archive:
    tar -czf native-machine-clean-room.tar.gz --exclude=target --exclude='*.tar.gz' Cargo.toml README.md rust-toolchain.toml rust-best-practices.md justfile .gitignore .cargo .githooks crates docs .github
