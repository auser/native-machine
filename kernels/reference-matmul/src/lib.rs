//! Reference scalar matmul kernel implementing the Native Machine ABI v3.
//!
//! Contract: `C[M,N] = A[M,K] x B[K,N]` over row-major contiguous `f32`
//! buffers. The input region holds `A` immediately followed by `B`; the
//! dimensions are passed as three little-endian `u32` values in `params`.
//! This is a deterministic scalar reference implementation: no heap
//! allocation, no hidden copies, and no implicit layout conversion.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_char;

pub const ABI_VERSION: u32 = 3;
const TYPE_F32: u16 = 1;
const OPERATION_MATMUL: u16 = 2;

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

/// Scalar reference implementation: `C[M,N] = A[M,K] x B[K,N]`, row-major.
pub fn matmul(a: &[f32], b: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
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

static NAME: &[u8] = b"reference-matmul\0";

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
    cpu_features: 0,
};

#[no_mangle]
pub extern "C" fn hologram_kernel_plugin_v3() -> *const KernelPlugin {
    &PLUGIN
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(m: u32, k: u32, n: u32) -> [u8; PARAMS_BYTES] {
        let mut params = [0_u8; PARAMS_BYTES];
        params[0..4].copy_from_slice(&m.to_le_bytes());
        params[4..8].copy_from_slice(&k.to_le_bytes());
        params[8..12].copy_from_slice(&n.to_le_bytes());
        params
    }

    fn run_matmul(ab: &[f32], output: &mut [f32], params: &[u8; PARAMS_BYTES]) -> i32 {
        let mut context = KernelContext {
            input: ab.as_ptr().cast(),
            input_bytes: std::mem::size_of_val(ab) as u64,
            output: output.as_mut_ptr().cast(),
            output_bytes: std::mem::size_of_val(output) as u64,
            params: params.as_ptr(),
            params_bytes: PARAMS_BYTES as u64,
        };
        // SAFETY: the context points at the live buffers above.
        unsafe { run(&mut context) }
    }

    #[test]
    fn matmul_matches_scalar_oracle() {
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut output = [0.0_f32; 4];
        matmul(&a, &b, &mut output, 2, 3, 2);
        for row in 0..2 {
            for column in 0..2 {
                let mut sum = 0.0_f32;
                for inner in 0..3 {
                    sum += a[row * 3 + inner] * b[inner * 2 + column];
                }
                assert_eq!(sum, output[row * 2 + column]);
            }
        }
        assert_eq!(output, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn entry_point_reports_abi_v3() {
        // SAFETY: the entry point returns a static descriptor.
        let descriptor = unsafe { &*hologram_kernel_plugin_v3() };
        assert_eq!(descriptor.abi_version, ABI_VERSION);
        assert_eq!(descriptor.operation, OPERATION_MATMUL);
        assert_eq!(descriptor.input_type, TYPE_F32);
        assert_eq!(descriptor.output_type, TYPE_F32);
    }

    #[test]
    fn run_rejects_null_context() {
        // SAFETY: null context is a defined rejection case.
        assert_eq!(unsafe { run(std::ptr::null_mut()) }, STATUS_NULL_CONTEXT);
    }

    #[test]
    fn run_rejects_null_buffers() {
        let mut output = [0.0_f32; 4];
        let params = params(2, 2, 2);
        let mut context = KernelContext {
            input: std::ptr::null(),
            input_bytes: 32,
            output: output.as_mut_ptr().cast(),
            output_bytes: 16,
            params: params.as_ptr(),
            params_bytes: PARAMS_BYTES as u64,
        };
        // SAFETY: null input is a defined rejection case.
        assert_eq!(unsafe { run(&mut context) }, STATUS_NULL_BUFFER);
    }

    #[test]
    fn run_rejects_missing_params() {
        let ab = [1.0_f32; 8];
        let mut output = [0.0_f32; 4];
        let mut context = KernelContext {
            input: ab.as_ptr().cast(),
            input_bytes: 32,
            output: output.as_mut_ptr().cast(),
            output_bytes: 16,
            params: std::ptr::null(),
            params_bytes: 0,
        };
        // SAFETY: missing params are a defined rejection case.
        assert_eq!(unsafe { run(&mut context) }, STATUS_INVALID_PARAMS);
    }

    #[test]
    fn run_rejects_zero_dimensions() {
        let ab = [1.0_f32; 8];
        let mut output = [0.0_f32; 4];
        assert_eq!(
            run_matmul(&ab, &mut output, &params(0, 2, 2)),
            STATUS_INVALID_DIMENSIONS
        );
    }

    #[test]
    fn run_rejects_input_length_mismatch() {
        let ab = [1.0_f32; 8];
        let mut output = [0.0_f32; 6];
        assert_eq!(
            run_matmul(&ab, &mut output, &params(3, 2, 2)),
            STATUS_LENGTH_MISMATCH
        );
    }

    #[test]
    fn run_rejects_undersized_output() {
        let ab = [1.0_f32; 8];
        let mut output = [0.0_f32; 3];
        assert_eq!(
            run_matmul(&ab, &mut output, &params(2, 2, 2)),
            STATUS_OUTPUT_TOO_SMALL
        );
    }

    #[test]
    fn run_computes_2x2_through_the_abi() {
        let ab = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0_f32; 4];
        assert_eq!(run_matmul(&ab, &mut output, &params(2, 2, 2)), STATUS_OK);
        assert_eq!(output, [19.0, 22.0, 43.0, 50.0]);
    }
}
