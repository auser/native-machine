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

fn bench_resolved_dispatch(
    registry: &PluginRegistry,
    rows: &mut Vec<Row>,
) -> Result<(), Box<dyn Error>> {
    let kernel = ["neon-add-one", "reference-add-one"]
        .into_iter()
        .find(|name| registry.kernel_index(name).is_some())
        .ok_or("no add-one kernel is installed")?;
    let handle = registry.resolve(kernel)?;
    for size in [1, 64] {
        let input = f32_values(size);
        let mut output = vec![0.0_f32; size];
        registry.run_f32(kernel, &input, &mut output)?;
        let failures = Cell::new(0_u64);
        let (named_iterations, named_elapsed) = measure(|| {
            if registry
                .run_f32(kernel, black_box(&input), black_box(&mut output))
                .is_err()
            {
                failures.set(failures.get() + 1);
            }
        });
        let (resolved_iterations, resolved_elapsed) = measure(|| {
            if registry
                .run_f32_resolved(handle, black_box(&input), black_box(&mut output))
                .is_err()
            {
                failures.set(failures.get() + 1);
            }
        });
        if failures.get() != 0 {
            return Err(format!("kernel {kernel} failed during benchmarking").into());
        }
        rows.push(Row {
            label: format!("add-one ({size} f32) via {kernel} (resolved handle)"),
            elements: size as u64,
            native_ns: Some(nanos(named_iterations, named_elapsed)),
            dispatched_ns: nanos(resolved_iterations, resolved_elapsed),
        });
    }
    Ok(())
}

/// Compares a chained plan of three built-in add-scalar records executed by
/// the fusing chain executor (one pass over the buffers) against three
/// unfused native passes (each intermediate round-trips through memory).
fn bench_fused_plan_chain(
    registry: &PluginRegistry,
    rows: &mut Vec<Row>,
) -> Result<(), Box<dyn Error>> {
    let size = crate::arena::MAX_VALUES;
    let mut records = Vec::new();
    for _ in 0..3 {
        let mut record = [0_u8; ops::OPERATION_BYTES];
        record[0..2].copy_from_slice(&1_u16.to_le_bytes());
        record[4..8].copy_from_slice(&1.0_f32.to_le_bytes());
        record[8..12].copy_from_slice(&(size as u32).to_le_bytes());
        record[12..16].copy_from_slice(&(size as u32).to_le_bytes());
        records.extend_from_slice(&record);
    }
    let mut arena = SessionArena::new();
    arena.load_input(&f32_values(size))?;
    let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
    ops::execute_chain_with_plugins(&mut arena, &records, &mut scratch, registry)?;

    let failures = Cell::new(0_u64);
    let (fused_iterations, fused_elapsed) = measure(|| {
        if ops::execute_chain_with_plugins(
            black_box(&mut arena),
            black_box(&records),
            black_box(&mut scratch),
            registry,
        )
        .is_err()
        {
            failures.set(failures.get() + 1);
        }
    });
    let input = f32_values(size);
    let mut first = input.clone();
    let mut second = vec![0.0_f32; size];
    let (pass_iterations, pass_elapsed) = measure(|| {
        for (source, destination) in first.iter().zip(second.iter_mut()) {
            *destination = *source + 1.0;
        }
        for (source, destination) in second.iter().zip(first.iter_mut()) {
            *destination = *source + 1.0;
        }
        for (source, destination) in first.iter().zip(second.iter_mut()) {
            *destination = *source + 1.0;
        }
        black_box(&second);
    });
    if failures.get() != 0 {
        return Err("fused plan chain failed during benchmarking".into());
    }
    rows.push(Row {
        label: format!("plan chain add x3 fused ({size} f32)"),
        elements: size as u64,
        native_ns: Some(nanos(pass_iterations, pass_elapsed)),
        dispatched_ns: nanos(fused_iterations, fused_elapsed),
    });
    Ok(())
}

/// Compares a chained matmul + relu (two dispatches, C round-trips through
/// memory) against the fused MATMUL_ACT kernel (activation applied to the
/// accumulators, C written once). Uses a thin-K memory-bound shape where
/// fusion matters and a compute-bound shape where it should be neutral.
fn bench_fused_matmul(
    registry: &PluginRegistry,
    rows: &mut Vec<Row>,
) -> Result<(), Box<dyn Error>> {
    let Some(fused) = ["avx2-fma-matmul-act"]
        .into_iter()
        .find(|name| registry.kernel_index(name).is_some())
    else {
        return Ok(());
    };
    let Some(matmul) = ["avx2-fma-matmul", "neon-matmul", "reference-matmul"]
        .into_iter()
        .find(|name| registry.kernel_index(name).is_some())
    else {
        return Ok(());
    };
    let Some(relu) = ["avx2-relu", "neon-relu", "reference-relu"]
        .into_iter()
        .find(|name| registry.kernel_index(name).is_some())
    else {
        return Ok(());
    };
    for (m, k, n) in [(1024_u32, 8_u32, 1024_u32), (256, 256, 256)] {
        let a_elements = (m * k) as usize;
        let c_elements = (m * n) as usize;
        let mut ab = f32_values(a_elements + (k * n) as usize);
        for (index, value) in ab.iter_mut().enumerate() {
            *value = (index % 7) as f32 * 0.25 - 0.75;
        }
        let (a, b) = ab.split_at(a_elements);
        let mut c = vec![0.0_f32; c_elements];
        let mut c_activated = vec![0.0_f32; c_elements];
        registry.run_matmul(matmul, a, b, &mut c, m, k, n)?;
        registry.run_f32(relu, &c, &mut c_activated)?;
        registry.run_matmul_act(fused, a, b, &mut c_activated, m, k, n, 1)?;

        let failures = Cell::new(0_u64);
        let (chained_iterations, chained_elapsed) = measure(|| {
            let first = registry.run_matmul(
                matmul,
                black_box(a),
                black_box(b),
                black_box(&mut c),
                m,
                k,
                n,
            );
            let second = registry.run_f32(relu, black_box(&c), black_box(&mut c_activated));
            if first.is_err() || second.is_err() {
                failures.set(failures.get() + 1);
            }
        });
        let (fused_iterations, fused_elapsed) = measure(|| {
            if registry
                .run_matmul_act(
                    fused,
                    black_box(a),
                    black_box(b),
                    black_box(&mut c),
                    m,
                    k,
                    n,
                    1,
                )
                .is_err()
            {
                failures.set(failures.get() + 1);
            }
        });
        if failures.get() != 0 {
            return Err("fused matmul benchmarking failed".into());
        }
        let chained_ns = nanos(chained_iterations, chained_elapsed);
        rows.push(Row {
            label: format!("matmul+relu chained ({m}x{k}x{n}, {matmul}+{relu})"),
            elements: u64::from(m) * u64::from(n),
            native_ns: None,
            dispatched_ns: chained_ns,
        });
        rows.push(Row {
            label: format!("matmul+relu fused ({m}x{k}x{n}, {fused})"),
            elements: u64::from(m) * u64::from(n),
            native_ns: Some(chained_ns),
            dispatched_ns: nanos(fused_iterations, fused_elapsed),
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
    // Resolve every installed kernel once, then dispatch by handle: this is
    // the intended hot-loop pattern.
    let f32_handles: Vec<_> = ["reference-add-one", "neon-add-one", "avx2-add-one"]
        .iter()
        .filter_map(|name| registry.resolve(name).ok())
        .collect();
    let matmul_handles: Vec<_> = [
        "reference-matmul",
        "neon-matmul",
        "avx2-matmul",
        "avx2-fma-matmul",
    ]
    .iter()
    .filter_map(|name| registry.resolve(name).ok())
    .collect();
    let u64_handles: Vec<_> = [
        "reference-xor-shift-add",
        "neon-xor-shift-add",
        "avx2-xor-shift-add",
    ]
    .iter()
    .filter_map(|name| registry.resolve(name).ok())
    .collect();
    let tracking = crate::allocation::track();
    for _ in 0..ALLOCATION_DISPATCHES {
        for handle in &f32_handles {
            registry.run_f32_resolved(*handle, &input, &mut output)?;
        }
        for handle in &matmul_handles {
            registry.run_matmul_resolved(*handle, a, b, &mut c, 2, 2, 2)?;
        }
        for handle in &u64_handles {
            registry.run_u64_resolved(*handle, &words, &mut words_output, 13, 7)?;
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
        "native-machine kernel benchmarks\nhost: {} (avx2: {}, neon: {}, fma: {})\nmeasurement budget: {:?} per row",
        std::env::consts::ARCH,
        features.avx2,
        features.neon,
        features.fma,
        MEASUREMENT_BUDGET
    );

    let mut rows = Vec::new();
    let add_one = installed_kernels(
        &registry,
        &["reference-add-one", "neon-add-one", "avx2-add-one"],
    );
    let relu = installed_kernels(&registry, &["reference-relu", "neon-relu", "avx2-relu"]);
    let matmul = installed_kernels(
        &registry,
        &[
            "reference-matmul",
            "neon-matmul",
            "avx2-matmul",
            "avx2-fma-matmul",
        ],
    );
    let xor_shift_add = installed_kernels(
        &registry,
        &[
            "reference-xor-shift-add",
            "neon-xor-shift-add",
            "avx2-xor-shift-add",
        ],
    );
    for size in [1, 64, 4096, 65536, 1048576] {
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
    for dimension in [16, 64, 128, 256, 512] {
        bench_matmul(&registry, &matmul, dimension, &mut rows)?;
    }
    for size in [1, 64, 4096, 65536] {
        bench_xor_shift_add(&registry, &xor_shift_add, size, &mut rows)?;
    }
    bench_resolved_dispatch(&registry, &mut rows)?;
    bench_fused_matmul(&registry, &mut rows)?;
    bench_fused_plan_chain(&registry, &mut rows)?;
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
