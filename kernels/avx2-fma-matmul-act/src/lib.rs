//! AVX2+FMA3 fused matmul+activation kernel implementing the Native Machine
//! ABI v3.
//!
//! Contract: `C[M,N] = act(A[M,K] x B[K,N])` over row-major contiguous `f32`
//! buffers, matching `reference-matmul` followed by the activation. The input
//! region holds `A` immediately followed by `B`; the dimensions and the
//! activation are passed as four little-endian `u32` values in `params`
//! (`m, k, n, activation`), where activation 0 is identity and 1 is ReLU
//! (`max(x, 0.0)`). The activation is fused into the matmul: every element of
//! `C` is written exactly once with the activation already applied and never
//! re-read, which is the point of the fusion — no second pass over the output.
//!
//! On x86_64 the kernel uses a 4x16 register-blocked AVX2 micro-kernel (eight
//! accumulators, two 8-lane vectors of `B` and four broadcasts of `A` per
//! inner step, so the eight FMAs outrun the load ports) inside column panels
//! of 64, so one streamed panel of `B` stays cache-resident for large
//! matrices. C accumulates in memory across K blocks, so the activation is
//! applied only when storing the final K block — applying ReLU to a partial
//! sum would be wrong. FMA contraction means results may differ from the
//! scalar reference in the last mantissa bit, and the descriptor declares
//! both the AVX2 and FMA3 feature bits so admission rejects the kernel on
//! hosts without either. The public entry point runtime-detects both
//! features and falls back to the scalar path when either is absent, so the
//! unit tests are safe on any host. Other targets use the scalar fallback
//! unconditionally. The kernel performs no heap allocation, hidden copies,
//! or layout conversion.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_char;

pub const ABI_VERSION: u32 = 3;
const TYPE_F32: u16 = 1;
const OPERATION_MATMUL_ACT: u16 = 4;
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

const PARAMS_BYTES: usize = 16;

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

/// `C[M,N] = act(A[M,K] x B[K,N])`, AVX2 register-blocked on x86_64.
/// Activation 0 is identity; 1 is ReLU.
pub fn matmul_act(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    activation: u32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA3 were just detected at runtime, so the
            // target-feature contract of `matmul_avx2_fma_act` is satisfied.
            unsafe { matmul_avx2_fma_act(a, b, output, m, k, n, activation) };
            return;
        }
    }
    matmul_scalar(a, b, output, m, k, n);
    if activation == 1 {
        for value in output.iter_mut() {
            *value = value.max(0.0);
        }
    }
}

/// Columns processed per panel; keeps one streamed panel of `B` cache-resident
/// for large matrices.
#[cfg(target_arch = "x86_64")]
const COLUMN_PANEL: usize = 64;

/// Inner dimension processed per block; a K_BLOCK x COLUMN_PANEL slice of `B`
/// is 16 KiB and stays L1-resident across row blocks.
#[cfg(target_arch = "x86_64")]
const K_BLOCK: usize = 64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_avx2_fma_act(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    activation: u32,
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };
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
            // K is processed in blocks so each panel slice of B stays
            // cache-resident across row blocks. C accumulates in memory
            // across K blocks, initialised to zero on the first block; the
            // accumulation order per element stays k-ascending, so results
            // are identical to the unblocked kernel. The activation is
            // applied only on the final K block, immediately before the
            // store, so every element of C is written exactly once with the
            // activation already applied.
            let mut k0 = 0;
            while k0 < k {
                let k1 = (k0 + K_BLOCK).min(k);
                let finalize = activation == 1 && k1 == k;
                let mut row = 0;
                while row + 4 <= m {
                    let mut column = panel;
                    // 4x16 micro-kernel: eight accumulators, two 8-lane B row
                    // segments and four A broadcasts per inner step for eight
                    // FMAs. Two loads per four broadcasts amortize the B
                    // stream across enough FMAs to make the kernel
                    // compute-bound.
                    while column + 16 <= panel_end {
                        let (
                            mut c00,
                            mut c01,
                            mut c10,
                            mut c11,
                            mut c20,
                            mut c21,
                            mut c30,
                            mut c31,
                        ) = if k0 == 0 {
                            (
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                            )
                        } else {
                            (
                                _mm256_loadu_ps(output.as_ptr().add(row * n + column)),
                                _mm256_loadu_ps(output.as_ptr().add(row * n + column + 8)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 1) * n + column)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 1) * n + column + 8)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 2) * n + column)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 2) * n + column + 8)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 3) * n + column)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 3) * n + column + 8)),
                            )
                        };
                        for inner in k0..k1 {
                            let b0 = _mm256_loadu_ps(b.as_ptr().add(inner * n + column));
                            let b1 = _mm256_loadu_ps(b.as_ptr().add(inner * n + column + 8));
                            let a0 = _mm256_set1_ps(*a.get_unchecked(row * k + inner));
                            let a1 = _mm256_set1_ps(*a.get_unchecked((row + 1) * k + inner));
                            let a2 = _mm256_set1_ps(*a.get_unchecked((row + 2) * k + inner));
                            let a3 = _mm256_set1_ps(*a.get_unchecked((row + 3) * k + inner));
                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a0, b1, c01);
                            c10 = _mm256_fmadd_ps(a1, b0, c10);
                            c11 = _mm256_fmadd_ps(a1, b1, c11);
                            c20 = _mm256_fmadd_ps(a2, b0, c20);
                            c21 = _mm256_fmadd_ps(a2, b1, c21);
                            c30 = _mm256_fmadd_ps(a3, b0, c30);
                            c31 = _mm256_fmadd_ps(a3, b1, c31);
                        }
                        if finalize {
                            // Value first, zeros second: maxNum NaN semantics
                            // match f32::max.
                            c00 = _mm256_max_ps(c00, _mm256_set1_ps(0.0));
                            c01 = _mm256_max_ps(c01, _mm256_set1_ps(0.0));
                            c10 = _mm256_max_ps(c10, _mm256_set1_ps(0.0));
                            c11 = _mm256_max_ps(c11, _mm256_set1_ps(0.0));
                            c20 = _mm256_max_ps(c20, _mm256_set1_ps(0.0));
                            c21 = _mm256_max_ps(c21, _mm256_set1_ps(0.0));
                            c30 = _mm256_max_ps(c30, _mm256_set1_ps(0.0));
                            c31 = _mm256_max_ps(c31, _mm256_set1_ps(0.0));
                        }
                        _mm256_storeu_ps(output.as_mut_ptr().add(row * n + column), c00);
                        _mm256_storeu_ps(output.as_mut_ptr().add(row * n + column + 8), c01);
                        _mm256_storeu_ps(output.as_mut_ptr().add((row + 1) * n + column), c10);
                        _mm256_storeu_ps(output.as_mut_ptr().add((row + 1) * n + column + 8), c11);
                        _mm256_storeu_ps(output.as_mut_ptr().add((row + 2) * n + column), c20);
                        _mm256_storeu_ps(output.as_mut_ptr().add((row + 2) * n + column + 8), c21);
                        _mm256_storeu_ps(output.as_mut_ptr().add((row + 3) * n + column), c30);
                        _mm256_storeu_ps(output.as_mut_ptr().add((row + 3) * n + column + 8), c31);
                        column += 16;
                    }
                    // 4x8 remainder within the panel.
                    while column + 8 <= panel_end {
                        let (mut c0, mut c1, mut c2, mut c3) = if k0 == 0 {
                            (
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                                _mm256_set1_ps(0.0),
                            )
                        } else {
                            (
                                _mm256_loadu_ps(output.as_ptr().add(row * n + column)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 1) * n + column)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 2) * n + column)),
                                _mm256_loadu_ps(output.as_ptr().add((row + 3) * n + column)),
                            )
                        };
                        for inner in k0..k1 {
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
                        if finalize {
                            c0 = _mm256_max_ps(c0, _mm256_set1_ps(0.0));
                            c1 = _mm256_max_ps(c1, _mm256_set1_ps(0.0));
                            c2 = _mm256_max_ps(c2, _mm256_set1_ps(0.0));
                            c3 = _mm256_max_ps(c3, _mm256_set1_ps(0.0));
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
                            let mut sum = if k0 == 0 {
                                0.0
                            } else {
                                output[blocked_row * n + column]
                            };
                            for inner in k0..k1 {
                                sum += a[blocked_row * k + inner] * b[inner * n + column];
                            }
                            if finalize {
                                sum = sum.max(0.0);
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
                        let mut c0 = if k0 == 0 {
                            _mm256_set1_ps(0.0)
                        } else {
                            _mm256_loadu_ps(output.as_ptr().add(row * n + column))
                        };
                        for inner in k0..k1 {
                            let b_row = _mm256_loadu_ps(b.as_ptr().add(inner * n + column));
                            c0 = _mm256_fmadd_ps(
                                _mm256_set1_ps(*a.get_unchecked(row * k + inner)),
                                b_row,
                                c0,
                            );
                        }
                        if finalize {
                            c0 = _mm256_max_ps(c0, _mm256_set1_ps(0.0));
                        }
                        _mm256_storeu_ps(output.as_mut_ptr().add(row * n + column), c0);
                        column += 8;
                    }
                    while column < panel_end {
                        let mut sum = if k0 == 0 {
                            0.0
                        } else {
                            output[row * n + column]
                        };
                        for inner in k0..k1 {
                            sum += a[row * k + inner] * b[inner * n + column];
                        }
                        if finalize {
                            sum = sum.max(0.0);
                        }
                        output[row * n + column] = sum;
                        column += 1;
                    }
                    row += 1;
                }
                k0 = k1;
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
    let activation = u32::from_le_bytes([params[12], params[13], params[14], params[15]]);
    if activation > 1 {
        return STATUS_INVALID_PARAMS;
    }
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
    matmul_act(a, b, output, m, k, n, activation);
    STATUS_OK
}

static NAME: &[u8] = b"avx2-fma-matmul-act\0";

static PLUGIN: KernelPlugin = KernelPlugin {
    abi_version: ABI_VERSION,
    name: NAME.as_ptr().cast(),
    run,
    input_type: TYPE_F32,
    output_type: TYPE_F32,
    operation: OPERATION_MATMUL_ACT,
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

    // Inputs are multiples of 0.25/0.5 with small magnitudes, so products and
    // partial sums are exactly representable and FMA contraction cannot change
    // the result. Sizes exercise the 4x16 micro-kernel, the 4x8 remainder,
    // scalar tail columns, tail rows, multiple panels, and K blocks beyond
    // K_BLOCK (C accumulation in memory).
    const SIZES: [(usize, usize, usize); 3] = [(6, 7, 10), (5, 17, 130), (4, 257, 16)];

    fn exact_inputs(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..m * k)
            .map(|index| (index % 9) as f32 * 0.25 - 1.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|index| (index % 5) as f32 * 0.5 - 1.0)
            .collect();
        (a, b)
    }

    #[test]
    fn identity_activation_matches_reference_bitwise() {
        for (m, k, n) in SIZES {
            let (a, b) = exact_inputs(m, k, n);
            let mut expected = vec![0.0_f32; m * n];
            let mut actual = vec![0.0_f32; m * n];
            reference_matmul::matmul(&a, &b, &mut expected, m, k, n);
            matmul_act(&a, &b, &mut actual, m, k, n, 0);
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_eq!(expected.to_bits(), actual.to_bits());
            }
        }
    }

    #[test]
    fn relu_activation_matches_reference() {
        for (m, k, n) in SIZES {
            let (a, b) = exact_inputs(m, k, n);
            let mut expected = vec![0.0_f32; m * n];
            let mut actual = vec![0.0_f32; m * n];
            reference_matmul::matmul(&a, &b, &mut expected, m, k, n);
            for value in expected.iter_mut() {
                *value = value.max(0.0);
            }
            matmul_act(&a, &b, &mut actual, m, k, n, 1);
            // Results are exact with these inputs, so compare with equality.
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn run_rejects_invalid_activation() {
        let ab = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0_f32; 4];
        let mut params = [0_u8; PARAMS_BYTES];
        params[0..4].copy_from_slice(&2_u32.to_le_bytes());
        params[4..8].copy_from_slice(&2_u32.to_le_bytes());
        params[8..12].copy_from_slice(&2_u32.to_le_bytes());
        params[12..16].copy_from_slice(&2_u32.to_le_bytes());
        let mut context = KernelContext {
            input: ab.as_ptr().cast(),
            input_bytes: 32,
            output: output.as_mut_ptr().cast(),
            output_bytes: 16,
            params: params.as_ptr(),
            params_bytes: PARAMS_BYTES as u64,
        };
        // SAFETY: the context points at the live buffers above.
        assert_eq!(unsafe { run(&mut context) }, STATUS_INVALID_PARAMS);
    }

    #[test]
    fn run_computes_fused_2x2_through_the_abi() {
        let ab = [1.0_f32, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0];
        let mut output = [0.0_f32; 4];
        let mut params = [0_u8; PARAMS_BYTES];
        params[0..4].copy_from_slice(&2_u32.to_le_bytes());
        params[4..8].copy_from_slice(&2_u32.to_le_bytes());
        params[8..12].copy_from_slice(&2_u32.to_le_bytes());
        params[12..16].copy_from_slice(&1_u32.to_le_bytes());
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
        // matmul gives [1*-5+2*7, 1*6+2*-8, 3*-5+4*7, 3*6+4*-8]
        //             = [9, -10, 13, -14]; ReLU then clamps to [9, 0, 13, 0].
        assert_eq!(output, [9.0, 0.0, 13.0, 0.0]);
    }
}
