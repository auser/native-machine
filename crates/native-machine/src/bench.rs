//! Release-mode benchmarks comparing native baselines with ABI v3 dispatch.
//!
//! Run with `just bench`, which installs the reference kernels and executes
//! this module in a release build. Numbers are wall-clock medians-free best
//! effort: each measurement warms up, then runs for a fixed time budget and
//! reports total elapsed time divided by the iteration count.

use crate::arena::SessionArena;
use crate::config::Config;
use crate::ops;
use crate::plugin::{load_registry, PluginRegistry};
use std::cell::Cell;
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

const MEASUREMENT_BUDGET: Duration = Duration::from_millis(200);
const WARMUP_CALLS: u32 = 8;
const ALLOCATION_DISPATCHES: u32 = 1000;

struct Row {
    label: String,
    elements: u64,
    native_ns: Option<f64>,
    dispatched_ns: f64,
}

fn measure(mut call: impl FnMut()) -> (u64, Duration) {
    for _ in 0..WARMUP_CALLS {
        call();
    }
    let start = Instant::now();
    let mut iterations = 0_u64;
    loop {
        call();
        iterations += 1;
        if iterations.is_multiple_of(16) && start.elapsed() >= MEASUREMENT_BUDGET {
            break;
        }
    }
    (iterations, start.elapsed())
}

fn nanos(iterations: u64, elapsed: Duration) -> f64 {
    elapsed.as_nanos() as f64 / iterations as f64
}

fn native_add_one(input: &[f32], output: &mut [f32]) {
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = *source + 1.0;
    }
}

fn native_relu(input: &[f32], output: &mut [f32]) {
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = source.max(0.0);
    }
}

fn native_matmul(a: &[f32], b: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0_f32;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + column];
            }
            output[row * n + column] = sum;
        }
    }
}

fn native_xor_shift_add(input: &[u64], output: &mut [u64], shift: u32, add: u64) {
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = (source ^ (source << shift)).wrapping_add(add);
    }
}

fn f32_values(size: usize) -> Vec<f32> {
    (0..size).map(|index| index as f32 * 0.5 - 32.0).collect()
}

fn installed_kernels<'a>(registry: &PluginRegistry, candidates: &[&'a str]) -> Vec<&'a str> {
    candidates
        .iter()
        .copied()
        .filter(|name| registry.kernel_index(name).is_some())
        .collect()
}

fn bench_elementwise_f32(
    registry: &PluginRegistry,
    kernels: &[&str],
    label: &str,
    size: usize,
    native: fn(&[f32], &mut [f32]),
    rows: &mut Vec<Row>,
) -> Result<(), Box<dyn Error>> {
    let input = f32_values(size);
    let mut output = vec![0.0_f32; size];
    let (native_iterations, native_elapsed) =
        measure(|| native(black_box(&input), black_box(&mut output)));
    let native_ns = nanos(native_iterations, native_elapsed);
    for kernel in kernels {
        registry.run_f32(kernel, &input, &mut output)?;
        let failures = Cell::new(0_u64);
        let (iterations, elapsed) = measure(|| {
            if registry
                .run_f32(kernel, black_box(&input), black_box(&mut output))
                .is_err()
            {
                failures.set(failures.get() + 1);
            }
        });
        if failures.get() != 0 {
            return Err(format!("kernel {kernel} failed during benchmarking").into());
        }
        rows.push(Row {
            label: format!("{label} ({size} f32) via {kernel}"),
            elements: size as u64,
            native_ns: Some(native_ns),
            dispatched_ns: nanos(iterations, elapsed),
        });
    }
    Ok(())
}

fn bench_matmul(
    registry: &PluginRegistry,
    kernels: &[&str],
    dimension: u32,
    rows: &mut Vec<Row>,
) -> Result<(), Box<dyn Error>> {
    let dimension_usize = dimension as usize;
    let elements = dimension_usize * dimension_usize;
    let mut ab = f32_values(2 * elements);
    for (index, value) in ab.iter_mut().enumerate() {
        *value = (index % 7) as f32 * 0.25 - 0.5;
    }
    let (a, b) = ab.split_at(elements);
    let mut output = vec![0.0_f32; elements];
    let (native_iterations, native_elapsed) = measure(|| {
        native_matmul(
            black_box(a),
            black_box(b),
            black_box(&mut output),
            dimension_usize,
            dimension_usize,
            dimension_usize,
        );
    });
    let native_ns = nanos(native_iterations, native_elapsed);
    for kernel in kernels {
        registry.run_matmul(kernel, a, b, &mut output, dimension, dimension, dimension)?;
        let failures = Cell::new(0_u64);
        let (iterations, elapsed) = measure(|| {
            if registry
                .run_matmul(
                    kernel,
                    black_box(a),
                    black_box(b),
                    black_box(&mut output),
                    dimension,
                    dimension,
                    dimension,
                )
                .is_err()
            {
                failures.set(failures.get() + 1);
            }
        });
        if failures.get() != 0 {
            return Err(format!("kernel {kernel} failed during benchmarking").into());
        }
        rows.push(Row {
            label: format!("matmul ({dimension}x{dimension}x{dimension} f32) via {kernel}"),
            elements: u64::from(dimension).pow(2),
            native_ns: Some(native_ns),
            dispatched_ns: nanos(iterations, elapsed),
        });
    }
    Ok(())
}

fn bench_xor_shift_add(
    registry: &PluginRegistry,
    kernels: &[&str],
    size: usize,
    rows: &mut Vec<Row>,
) -> Result<(), Box<dyn Error>> {
    let input: Vec<u64> = (0..size)
        .map(|index| (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let mut output = vec![0_u64; size];
    let (native_iterations, native_elapsed) =
        measure(|| native_xor_shift_add(black_box(&input), black_box(&mut output), 13, 7));
    let native_ns = nanos(native_iterations, native_elapsed);
    for kernel in kernels {
        registry.run_u64(kernel, &input, &mut output, 13, 7)?;
        let failures = Cell::new(0_u64);
        let (iterations, elapsed) = measure(|| {
            if registry
                .run_u64(kernel, black_box(&input), black_box(&mut output), 13, 7)
                .is_err()
            {
                failures.set(failures.get() + 1);
            }
        });
        if failures.get() != 0 {
            return Err(format!("kernel {kernel} failed during benchmarking").into());
        }
        rows.push(Row {
            label: format!("xor-shift-add ({size} u64) via {kernel}"),
            elements: size as u64,
            native_ns: Some(native_ns),
            dispatched_ns: nanos(iterations, elapsed),
        });
    }
    Ok(())
}

fn bench_record_path(registry: &PluginRegistry, rows: &mut Vec<Row>) -> Result<(), Box<dyn Error>> {
    let size = crate::arena::MAX_VALUES;
    let index = registry
        .kernel_index("reference-add-one")
        .ok_or("reference-add-one is not installed")?;
    let index = u16::try_from(index).map_err(|_| "kernel index does not fit u16")?;
    let mut record = [0_u8; ops::OPERATION_BYTES];
    record[0..2].copy_from_slice(&ops::PLUGIN_OPCODE.to_le_bytes());
    record[2..4].copy_from_slice(&index.to_le_bytes());
    record[8..12].copy_from_slice(&(size as u32).to_le_bytes());
    record[12..16].copy_from_slice(&(size as u32).to_le_bytes());
    let mut arena = SessionArena::new();
    arena.load_input(&f32_values(size))?;
    let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
    ops::execute_records_with_plugins(&mut arena, &record, &mut scratch, registry)?;
    let failures = Cell::new(0_u64);
    let (iterations, elapsed) = measure(|| {
        if ops::execute_records_with_plugins(
            black_box(&mut arena),
            black_box(&record),
            black_box(&mut scratch),
            registry,
        )
        .is_err()
        {
            failures.set(failures.get() + 1);
        }
    });
    if failures.get() != 0 {
        return Err("record-path dispatch failed during benchmarking".into());
    }
    let input = f32_values(size);
    let mut output = vec![0.0_f32; size];
    let (native_iterations, native_elapsed) =
        measure(|| native_add_one(black_box(&input), black_box(&mut output)));
    rows.push(Row {
        label: format!("artifact record -> plugin ({size} f32)"),
        elements: size as u64,
        native_ns: Some(nanos(native_iterations, native_elapsed)),
        dispatched_ns: nanos(iterations, elapsed),
    });
    Ok(())
}

fn bench_identities(config: &Config, rows: &mut Vec<Row>) -> Result<(), Box<dyn Error>> {
    let fixture_path = std::env::temp_dir().join(format!(
        "native-machine-bench-fixture-{}.nm",
        std::process::id()
    ));
    crate::artifact::create_fixture(&fixture_path)?;
    let mapped = crate::artifact::MappedArtifact::open(&fixture_path, u64::MAX)?;
    let artifact = mapped.view()?;
    let artifact_bytes = artifact.header.artifact_bytes;

    let failures = Cell::new(0_u64);
    let (iterations, elapsed) = measure(|| {
        if black_box(artifact.identity()).is_err() {
            failures.set(failures.get() + 1);
        }
    });
    if failures.get() != 0 {
        return Err("artifact identity failed during benchmarking".into());
    }
    rows.push(Row {
        label: format!("artifact identity sha256 ({artifact_bytes} bytes)"),
        elements: artifact_bytes,
        native_ns: None,
        dispatched_ns: nanos(iterations, elapsed),
    });

    let (iterations, elapsed) = measure(|| {
        if black_box(artifact.provenance_identity()).is_err() {
            failures.set(failures.get() + 1);
        }
    });
    if failures.get() != 0 {
        return Err("provenance identity failed during benchmarking".into());
    }
    rows.push(Row {
        label: "provenance UOR address (uor-addr)".to_string(),
        elements: artifact_bytes,
        native_ns: None,
        dispatched_ns: nanos(iterations, elapsed),
    });

    let (iterations, elapsed) = measure(|| {
        if black_box(load_registry(config)).is_err() {
            failures.set(failures.get() + 1);
        }
    });
    if failures.get() != 0 {
        return Err("registry admission failed during benchmarking".into());
    }
    rows.push(Row {
        label: "plugin admission (manifest verify + load)".to_string(),
        elements: 0,
        native_ns: None,
        dispatched_ns: nanos(iterations, elapsed),
    });

    let _ = std::fs::remove_file(&fixture_path);
    Ok(())
}

fn measure_dispatch_allocations(registry: &PluginRegistry) -> Result<usize, Box<dyn Error>> {
    let input = f32_values(64);
    let mut output = vec![0.0_f32; 64];
    let ab = f32_values(8);
    let (a, b) = ab.split_at(4);
    let mut c = vec![0.0_f32; 4];
    let words: Vec<u64> = (0..64).map(|index| index as u64).collect();
    let mut words_output = vec![0_u64; 64];
    let tracking = crate::allocation::track();
    for _ in 0..ALLOCATION_DISPATCHES {
        for kernel in ["reference-add-one", "neon-add-one"] {
            if registry.kernel_index(kernel).is_some() {
                registry.run_f32(kernel, &input, &mut output)?;
            }
        }
        for kernel in ["reference-matmul", "neon-matmul"] {
            if registry.kernel_index(kernel).is_some() {
                registry.run_matmul(kernel, a, b, &mut c, 2, 2, 2)?;
            }
        }
        for kernel in ["reference-xor-shift-add", "neon-xor-shift-add"] {
            if registry.kernel_index(kernel).is_some() {
                registry.run_u64(kernel, &words, &mut words_output, 13, 7)?;
            }
        }
    }
    let allocations = tracking.count();
    drop(tracking);
    Ok(allocations)
}

fn format_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2} us", ns / 1_000.0)
    } else {
        format!("{ns:.0} ns")
    }
}

pub fn run(config: &Config) -> Result<(), Box<dyn Error>> {
    let registry = load_registry(config)?;
    for required in [
        "reference-add-one",
        "reference-relu",
        "reference-matmul",
        "reference-xor-shift-add",
    ] {
        if registry.kernel_index(required).is_none() {
            return Err(format!(
                "kernel {required} is not installed; run `just install-kernels` first"
            )
            .into());
        }
    }

    let features = crate::cpu::features();
    println!(
        "native-machine kernel benchmarks\nhost: {} (avx2: {}, neon: {})\nmeasurement budget: {:?} per row",
        std::env::consts::ARCH,
        features.avx2,
        features.neon,
        MEASUREMENT_BUDGET
    );

    let mut rows = Vec::new();
    let add_one = installed_kernels(&registry, &["reference-add-one", "neon-add-one"]);
    let relu = installed_kernels(&registry, &["reference-relu", "neon-relu"]);
    let matmul = installed_kernels(&registry, &["reference-matmul", "neon-matmul"]);
    let xor_shift_add = installed_kernels(
        &registry,
        &["reference-xor-shift-add", "neon-xor-shift-add"],
    );
    for size in [1, 64, 4096] {
        bench_elementwise_f32(
            &registry,
            &add_one,
            "add-one",
            size,
            native_add_one,
            &mut rows,
        )?;
    }
    bench_elementwise_f32(&registry, &relu, "relu", 4096, native_relu, &mut rows)?;
    for dimension in [16, 32, 64] {
        bench_matmul(&registry, &matmul, dimension, &mut rows)?;
    }
    for size in [1, 64, 4096] {
        bench_xor_shift_add(&registry, &xor_shift_add, size, &mut rows)?;
    }
    bench_record_path(&registry, &mut rows)?;

    println!(
        "\n{:<42} {:>12} {:>12} {:>10} {:>8}",
        "benchmark", "native", "dispatched", "ns/elem", "ratio"
    );
    for row in &rows {
        let native = row.native_ns.map_or_else(|| "-".to_string(), format_ns);
        let ratio = row.native_ns.map_or_else(
            || "-".to_string(),
            |native_ns| format!("{:.2}x", row.dispatched_ns / native_ns),
        );
        let per_element = if row.elements == 0 {
            "-".to_string()
        } else {
            format!("{:.2}", row.dispatched_ns / row.elements as f64)
        };
        println!(
            "{:<42} {:>12} {:>12} {:>10} {:>8}",
            row.label,
            native,
            format_ns(row.dispatched_ns),
            per_element,
            ratio
        );
    }

    let mut identity_rows = Vec::new();
    bench_identities(config, &mut identity_rows)?;
    println!("\nadmission and identity (per call):");
    for row in &identity_rows {
        println!("  {:<44} {:>12}", row.label, format_ns(row.dispatched_ns));
    }

    let allocations = measure_dispatch_allocations(&registry)?;
    println!(
        "\nheap allocations: {allocations} across {ALLOCATION_DISPATCHES} rounds of typed dispatch over every installed kernel"
    );
    if allocations != 0 {
        return Err("typed dispatch allocated on the success path".into());
    }
    println!("benchmark: passed");
    Ok(())
}
