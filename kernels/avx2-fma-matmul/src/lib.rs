//! AVX2+FMA3 matmul kernel implementing the Native Machine ABI v3.
//!
//! Contract: `C[M,N] = A[M,K] x B[K,N]` over row-major contiguous `f32`
//! buffers, matching `reference-matmul`. The input region holds `A`
//! immediately followed by `B`; the dimensions are passed as three
//! little-endian `u32` values in `params`.
//!
//! On x86_64 the kernel uses a 4x8 register-blocked micro-kernel (four
//! accumulators, one 8-lane vector of `B` per inner step, four broadcasts of
//! `A`) inside column panels of 64, so one streamed panel of `B` stays
//! cache-resident for large matrices. Unlike `avx2-matmul`, the inner step
//! uses fused multiply-add (`_mm256_fmadd_ps`), roughly doubling FLOP
//! throughput; FMA contraction means results may differ from the scalar
//! reference in the last mantissa bit, and the descriptor declares both the
//! AVX2 and FMA3 feature bits so admission rejects the kernel on hosts
//! without either. The public entry point runtime-detects both features and
//! falls back to the scalar path when either is absent, so the unit tests
//! are safe on any host. Other targets use the scalar fallback
//! unconditionally. The kernel performs no heap allocation, hidden copies,
//! or layout conversion.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_char;

pub const ABI_VERSION: u32 = 3;
const TYPE_F32: u16 = 1;
const OPERATION_MATMUL: u16 = 2;
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const CPU_AVX2: u64 = 1;
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const CPU_FMA: u64 = 4;

#[cfg(target_arch = "x86_64")]
const REQUIRED_FEATURES: u64 = CPU_AVX2 | CPU_FMA;
#[cfg(not(target_arch = "x86_64"))]
const REQUIRED_FEATURES: u64 = 0;

const STATUS_OK: i32 = 0;
const STATUS_NULL_CONTEXT: i32 = 1;
const STATUS_NULL_BUFFER: i32 = 2;
const STATUS_LENGTH_OVERFLOW: i32 = 3;
const STATUS_LENGTH_MISMATCH: i32 = 4;
const STATUS_MISALIGNED: i32 = 5;
const STATUS_INVALID_PARAMS: i32 = 6;
const STATUS_INVALID_DIMENSIONS: i32 = 7;
const STATUS_OUTPUT_TOO_SMALL: i32 = 8;

const PARAMS_BYTES: usize = 12;

#[repr(C)]
pub struct KernelContext {
    pub input: *const u8,
    pub input_bytes: u64,
    pub output: *mut u8,
    pub output_bytes: u64,
    pub params: *const u8,
    pub params_bytes: u64,
}

pub type KernelRun = unsafe extern "C" fn(*mut KernelContext) -> i32;

#[repr(C)]
pub struct KernelPlugin {
    pub abi_version: u32,
    pub name: *const c_char,
    pub run: KernelRun,
    pub input_type: u16,
    pub output_type: u16,
    pub operation: u16,
    pub flags: u32,
    pub required_alignment: u32,
    pub scratch_bytes: u64,
    pub cpu_features: u64,
}

// SAFETY: `KernelPlugin` is an immutable ABI descriptor. Its raw pointer
// targets the static, NUL-terminated `NAME` bytes, and its function pointer
// has no descriptor-owned mutable state.
unsafe impl Sync for KernelPlugin {}

pub fn matmul_scalar(a: &[f32], b: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
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

/// `C[M,N] = A[M,K] x B[K,N]`, AVX2 register-blocked on x86_64.
pub fn matmul(a: &[f32], b: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA3 were just detected at runtime, so the
            // target-feature contract of `matmul_avx2_fma` is satisfied.
            unsafe { matmul_avx2_fma(a, b, output, m, k, n) };
            return;
        }
    }
    matmul_scalar(a, b, output, m, k, n);
}

/// Columns processed per panel; keeps one streamed panel of `B` cache-resident
/// for large matrices.
#[cfg(target_arch = "x86_64")]
const COLUMN_PANEL: usize = 64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_avx2_fma(a: &[f32], b: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    // SAFETY: the caller detected AVX2 and FMA3 at runtime, so the
    // target-feature contract of every intrinsic used here is satisfied.
    // Every load from `b` reads eight lanes starting at inner * n + column,
    // with the column bound checked against the panel end (<= n) and
    // inner < k, staying within b.len() == k * n; every store stays within
    // output.len() == m * n; every `get_unchecked` on `a` stays below
    // a.len() == m * k because row < m and inner < k. The unaligned
    // load/store intrinsics do not require alignment.
    unsafe {
        let mut panel = 0;
        while panel < n {
            let panel_end = (panel + COLUMN_PANEL).min(n);
            let mut row = 0;
            while row + 4 <= m {
                let mut column = panel;
                // 4x8 micro-kernel: four accumulators, one 8-lane B row
                // segment per inner step, four A broadcasts, four FMAs.
                while column + 8 <= panel_end {
                    let mut c0 = _mm256_set1_ps(0.0);
                    let mut c1 = _mm256_set1_ps(0.0);
                    let mut c2 = _mm256_set1_ps(0.0);
                    let mut c3 = _mm256_set1_ps(0.0);
                    for inner in 0..k {
                        let b_row = _mm256_loadu_ps(b.as_ptr().add(inner * n + column));
                        let a0 = _mm256_set1_ps(*a.get_unchecked(row * k + inner));
                        let a1 = _mm256_set1_ps(*a.get_unchecked((row + 1) * k + inner));
                        let a2 = _mm256_set1_ps(*a.get_unchecked((row + 2) * k + inner));
                        let a3 = _mm256_set1_ps(*a.get_unchecked((row + 3) * k + inner));
                        c0 = _mm256_fmadd_ps(a0, b_row, c0);
                        c1 = _mm256_fmadd_ps(a1, b_row, c1);
                        c2 = _mm256_fmadd_ps(a2, b_row, c2);
                        c3 = _mm256_fmadd_ps(a3, b_row, c3);
                    }
                    _mm256_storeu_ps(output.as_mut_ptr().add(row * n + column), c0);
                    _mm256_storeu_ps(output.as_mut_ptr().add((row + 1) * n + column), c1);
                    _mm256_storeu_ps(output.as_mut_ptr().add((row + 2) * n + column), c2);
                    _mm256_storeu_ps(output.as_mut_ptr().add((row + 3) * n + column), c3);
                    column += 8;
                }
                // Scalar tail columns for the four blocked rows.
                while column < panel_end {
                    for blocked_row in row..row + 4 {
                        let mut sum = 0.0_f32;
                        for inner in 0..k {
                            sum += a[blocked_row * k + inner] * b[inner * n + column];
                        }
                        output[blocked_row * n + column] = sum;
                    }
                    column += 1;
                }
                row += 4;
            }
            // Tail rows: single-row vector pass plus scalar tail columns.
            while row < m {
                let mut column = panel;
                while column + 8 <= panel_end {
                    let mut c0 = _mm256_set1_ps(0.0);
                    for inner in 0..k {
                        let b_row = _mm256_loadu_ps(b.as_ptr().add(inner * n + column));
                        c0 = _mm256_fmadd_ps(
                            _mm256_set1_ps(*a.get_unchecked(row * k + inner)),
                            b_row,
                            c0,
                        );
                    }
                    _mm256_storeu_ps(output.as_mut_ptr().add(row * n + column), c0);
                    column += 8;
                }
                while column < panel_end {
                    let mut sum = 0.0_f32;
                    for inner in 0..k {
                        sum += a[row * k + inner] * b[inner * n + column];
                    }
                    output[row * n + column] = sum;
                    column += 1;
                }
                row += 1;
            }
            panel += COLUMN_PANEL;
        }
    }
}

fn checked_elements(rows: usize, columns: usize) -> Result<usize, i32> {
    rows.checked_mul(columns).ok_or(STATUS_LENGTH_OVERFLOW)
}

unsafe extern "C" fn run(context: *mut KernelContext) -> i32 {
    if context.is_null() {
        return STATUS_NULL_CONTEXT;
    }
    // SAFETY: the host supplies a valid context pointer for this call.
    let context = unsafe { &mut *context };
    if context.input.is_null() || context.output.is_null() {
        return STATUS_NULL_BUFFER;
    }
    let params_bytes = match usize::try_from(context.params_bytes) {
        Ok(value) => value,
        Err(_) => return STATUS_LENGTH_OVERFLOW,
    };
    if params_bytes != PARAMS_BYTES || context.params.is_null() {
        return STATUS_INVALID_PARAMS;
    }
    // SAFETY: the params pointer is non-null and PARAMS_BYTES long per the
    // check above.
    let params = unsafe { std::slice::from_raw_parts(context.params, PARAMS_BYTES) };
    let m = u32::from_le_bytes([params[0], params[1], params[2], params[3]]) as usize;
    let k = u32::from_le_bytes([params[4], params[5], params[6], params[7]]) as usize;
    let n = u32::from_le_bytes([params[8], params[9], params[10], params[11]]) as usize;
    if m == 0 || k == 0 || n == 0 {
        return STATUS_INVALID_DIMENSIONS;
    }
    let a_elements = match checked_elements(m, k) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let b_elements = match checked_elements(k, n) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let c_elements = match checked_elements(m, n) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let element_bytes = std::mem::size_of::<f32>();
    let expected_input = match (a_elements + b_elements).checked_mul(element_bytes) {
        Some(value) => value,
        None => return STATUS_LENGTH_OVERFLOW,
    };
    let expected_output = match c_elements.checked_mul(element_bytes) {
        Some(value) => value,
        None => return STATUS_LENGTH_OVERFLOW,
    };
    let input_bytes = match usize::try_from(context.input_bytes) {
        Ok(value) => value,
        Err(_) => return STATUS_LENGTH_OVERFLOW,
    };
    let output_bytes = match usize::try_from(context.output_bytes) {
        Ok(value) => value,
        Err(_) => return STATUS_LENGTH_OVERFLOW,
    };
    if input_bytes != expected_input {
        return STATUS_LENGTH_MISMATCH;
    }
    if output_bytes < expected_output {
        return STATUS_OUTPUT_TOO_SMALL;
    }
    if output_bytes != expected_output {
        return STATUS_LENGTH_MISMATCH;
    }
    let alignment = std::mem::align_of::<f32>();
    if !(context.input as usize).is_multiple_of(alignment)
        || !(context.output as usize).is_multiple_of(alignment)
    {
        return STATUS_MISALIGNED;
    }
    // SAFETY: the host validated both buffers; their byte lengths match the
    // declared dimensions exactly and their alignment was checked.
    let input =
        unsafe { std::slice::from_raw_parts(context.input.cast::<f32>(), a_elements + b_elements) };
    // SAFETY: the host provides writable output storage for M*N values.
    let output =
        unsafe { std::slice::from_raw_parts_mut(context.output.cast::<f32>(), c_elements) };
    let (a, b) = input.split_at(a_elements);
    matmul(a, b, output, m, k, n);
    STATUS_OK
}

static NAME: &[u8] = b"avx2-fma-matmul\0";

static PLUGIN: KernelPlugin = KernelPlugin {
    abi_version: ABI_VERSION,
    name: NAME.as_ptr().cast(),
    run,
    input_type: TYPE_F32,
    output_type: TYPE_F32,
    operation: OPERATION_MATMUL,
    flags: 0,
    required_alignment: 4,
    scratch_bytes: 0,
    cpu_features: REQUIRED_FEATURES,
};

// Export the entry symbol only in non-test builds so unit tests can
// statically link a reference kernel crate, which exports the same ABI
// symbol, without duplicate-symbol link errors (lld on Linux rejects them).
#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn hologram_kernel_plugin_v3() -> *const KernelPlugin {
    &PLUGIN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_bitwise_on_exact_values() {
        // Inputs are multiples of 0.25 with small magnitudes, so products and
        // partial sums are exactly representable and FMA contraction cannot
        // change the result. Sizes exercise the 4x8 micro-kernel, scalar tail
        // columns, tail rows, and multiple panels.
        for (m, k, n) in [(6, 7, 10), (5, 17, 130), (9, 3, 66)] {
            let a: Vec<f32> = (0..m * k)
                .map(|index| (index % 9) as f32 * 0.25 - 1.0)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|index| (index % 5) as f32 * 0.5 - 1.0)
                .collect();
            let mut expected = vec![0.0_f32; m * n];
            let mut actual = vec![0.0_f32; m * n];
            reference_matmul::matmul(&a, &b, &mut expected, m, k, n);
            matmul(&a, &b, &mut actual, m, k, n);
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_eq!(expected.to_bits(), actual.to_bits());
            }
        }
    }

    #[test]
    fn matches_reference_within_ulp_tolerance() {
        let (m, k, n) = (13, 11, 9);
        let a: Vec<f32> = (0..m * k).map(|index| (index as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|index| (index as f32 * 0.2).cos()).collect();
        let mut expected = vec![0.0_f32; m * n];
        let mut actual = vec![0.0_f32; m * n];
        reference_matmul::matmul(&a, &b, &mut expected, m, k, n);
        matmul(&a, &b, &mut actual, m, k, n);
        for (expected, actual) in expected.iter().zip(&actual) {
            let tolerance = 1e-5_f32 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn run_computes_2x2_through_the_abi() {
        let ab = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0_f32; 4];
        let mut params = [0_u8; PARAMS_BYTES];
        params[0..4].copy_from_slice(&2_u32.to_le_bytes());
        params[4..8].copy_from_slice(&2_u32.to_le_bytes());
        params[8..12].copy_from_slice(&2_u32.to_le_bytes());
        let mut context = KernelContext {
            input: ab.as_ptr().cast(),
            input_bytes: 32,
            output: output.as_mut_ptr().cast(),
            output_bytes: 16,
            params: params.as_ptr(),
            params_bytes: PARAMS_BYTES as u64,
        };
        // SAFETY: the context points at the live buffers above.
        assert_eq!(unsafe { run(&mut context) }, STATUS_OK);
        assert_eq!(output, [19.0, 22.0, 43.0, 50.0]);
    }
}
