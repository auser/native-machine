//! Reference xor-shift-add kernel implementing the Native Machine ABI v3.
//!
//! Contract: `output[i] = (input[i] ^ (input[i] << shift)) + add` over
//! contiguous `u64` buffers, with explicit wrapping semantics:
//!
//! - `shift` must be in `0..=63`; any larger shift is rejected.
//! - The left shift discards bits shifted past bit 63 (natural `u64` shift).
//! - The final addition wraps modulo 2^64 (`wrapping_add`).
//!
//! Parameters are encoded as 16 little-endian bytes: `shift: u32`, a
//! reserved `u32` that must be zero, and `add: u64`. The kernel performs no
//! heap allocation and is deterministic.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_char;

pub const ABI_VERSION: u32 = 3;
const TYPE_U64: u16 = 2;
const OPERATION_XOR_SHIFT_ADD: u16 = 3;

/// Largest accepted shift; `shift` must be in `0..=MAX_SHIFT`.
pub const MAX_SHIFT: u32 = 63;

const STATUS_OK: i32 = 0;
const STATUS_NULL_CONTEXT: i32 = 1;
const STATUS_NULL_BUFFER: i32 = 2;
const STATUS_LENGTH_OVERFLOW: i32 = 3;
const STATUS_LENGTH_MISMATCH: i32 = 4;
const STATUS_MISALIGNED: i32 = 5;
const STATUS_INVALID_PARAMS: i32 = 6;

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

/// Scalar reference implementation with wrapping semantics:
/// `output[i] = (input[i] ^ (input[i] << shift)).wrapping_add(add)`.
pub fn xor_shift_add(input: &[u64], output: &mut [u64], shift: u32, add: u64) {
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = (source ^ (source << shift)).wrapping_add(add);
    }
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
    let shift = u32::from_le_bytes([params[0], params[1], params[2], params[3]]);
    let reserved = u32::from_le_bytes([params[4], params[5], params[6], params[7]]);
    let add = u64::from_le_bytes([
        params[8], params[9], params[10], params[11], params[12], params[13], params[14],
        params[15],
    ]);
    if shift > MAX_SHIFT || reserved != 0 {
        return STATUS_INVALID_PARAMS;
    }
    let input_bytes = match usize::try_from(context.input_bytes) {
        Ok(value) => value,
        Err(_) => return STATUS_LENGTH_OVERFLOW,
    };
    let output_bytes = match usize::try_from(context.output_bytes) {
        Ok(value) => value,
        Err(_) => return STATUS_LENGTH_OVERFLOW,
    };
    let element_bytes = std::mem::size_of::<u64>();
    if input_bytes != output_bytes || !input_bytes.is_multiple_of(element_bytes) {
        return STATUS_LENGTH_MISMATCH;
    }
    let alignment = std::mem::align_of::<u64>();
    if !(context.input as usize).is_multiple_of(alignment)
        || !(context.output as usize).is_multiple_of(alignment)
    {
        return STATUS_MISALIGNED;
    }
    let len = input_bytes / element_bytes;
    // SAFETY: the host validated both buffers; their byte lengths are equal
    // and multiple of the element size, and their alignment was checked.
    let input = unsafe { std::slice::from_raw_parts(context.input.cast::<u64>(), len) };
    // SAFETY: the host provides writable output storage for `len` values.
    let output = unsafe { std::slice::from_raw_parts_mut(context.output.cast::<u64>(), len) };
    xor_shift_add(input, output, shift, add);
    STATUS_OK
}

static NAME: &[u8] = b"reference-xor-shift-add\0";

static PLUGIN: KernelPlugin = KernelPlugin {
    abi_version: ABI_VERSION,
    name: NAME.as_ptr().cast(),
    run,
    input_type: TYPE_U64,
    output_type: TYPE_U64,
    operation: OPERATION_XOR_SHIFT_ADD,
    flags: 0,
    required_alignment: 8,
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

    fn params(shift: u32, reserved: u32, add: u64) -> [u8; PARAMS_BYTES] {
        let mut params = [0_u8; PARAMS_BYTES];
        params[0..4].copy_from_slice(&shift.to_le_bytes());
        params[4..8].copy_from_slice(&reserved.to_le_bytes());
        params[8..16].copy_from_slice(&add.to_le_bytes());
        params
    }

    fn run_buffers(input: &[u64], output: &mut [u64], params: &[u8; PARAMS_BYTES]) -> i32 {
        let mut context = KernelContext {
            input: input.as_ptr().cast(),
            input_bytes: std::mem::size_of_val(input) as u64,
            output: output.as_mut_ptr().cast(),
            output_bytes: std::mem::size_of_val(output) as u64,
            params: params.as_ptr(),
            params_bytes: PARAMS_BYTES as u64,
        };
        // SAFETY: the context points at the live buffers above.
        unsafe { run(&mut context) }
    }

    #[test]
    fn xor_shift_add_matches_scalar_oracle() {
        let input = [0_u64, 1, 42, u64::MAX];
        let mut output = [0_u64; 4];
        xor_shift_add(&input, &mut output, 13, 7);
        for (source, destination) in input.iter().zip(output) {
            assert_eq!((source ^ (source << 13)).wrapping_add(7), destination);
        }
    }

    #[test]
    fn shift_zero_cancels_the_xor() {
        let input = [7_u64, u64::MAX];
        let mut output = [0_u64; 2];
        xor_shift_add(&input, &mut output, 0, 5);
        assert_eq!(output, [5, 5]);
    }

    #[test]
    fn addition_wraps_modulo_2_to_the_64() {
        let input = [u64::MAX];
        let mut output = [0_u64; 1];
        xor_shift_add(&input, &mut output, 0, 1);
        assert_eq!(output, [1]);
        xor_shift_add(&input, &mut output, 63, u64::MAX);
        assert_eq!(
            output[0],
            (u64::MAX ^ (u64::MAX << 63)).wrapping_add(u64::MAX)
        );
    }

    #[test]
    fn entry_point_reports_abi_v3() {
        // SAFETY: the entry point returns a static descriptor.
        let descriptor = unsafe { &*hologram_kernel_plugin_v3() };
        assert_eq!(descriptor.abi_version, ABI_VERSION);
        assert_eq!(descriptor.operation, OPERATION_XOR_SHIFT_ADD);
        assert_eq!(descriptor.input_type, TYPE_U64);
        assert_eq!(descriptor.output_type, TYPE_U64);
    }

    #[test]
    fn run_rejects_null_context() {
        // SAFETY: null context is a defined rejection case.
        assert_eq!(unsafe { run(std::ptr::null_mut()) }, STATUS_NULL_CONTEXT);
    }

    #[test]
    fn run_rejects_null_buffers() {
        let mut output = [0_u64; 1];
        let params = params(1, 0, 0);
        let mut context = KernelContext {
            input: std::ptr::null(),
            input_bytes: 8,
            output: output.as_mut_ptr().cast(),
            output_bytes: 8,
            params: params.as_ptr(),
            params_bytes: PARAMS_BYTES as u64,
        };
        // SAFETY: null input is a defined rejection case.
        assert_eq!(unsafe { run(&mut context) }, STATUS_NULL_BUFFER);
    }

    #[test]
    fn run_rejects_missing_params() {
        let input = [1_u64];
        let mut output = [0_u64; 1];
        let mut context = KernelContext {
            input: input.as_ptr().cast(),
            input_bytes: 8,
            output: output.as_mut_ptr().cast(),
            output_bytes: 8,
            params: std::ptr::null(),
            params_bytes: 0,
        };
        // SAFETY: missing params are a defined rejection case.
        assert_eq!(unsafe { run(&mut context) }, STATUS_INVALID_PARAMS);
    }

    #[test]
    fn run_rejects_shift_above_63() {
        let input = [1_u64];
        let mut output = [0_u64; 1];
        assert_eq!(
            run_buffers(&input, &mut output, &params(64, 0, 0)),
            STATUS_INVALID_PARAMS
        );
    }

    #[test]
    fn run_accepts_shift_63() {
        let input = [3_u64];
        let mut output = [0_u64; 1];
        assert_eq!(
            run_buffers(&input, &mut output, &params(MAX_SHIFT, 0, 0)),
            STATUS_OK
        );
        assert_eq!(output[0], 3 ^ (3 << 63));
    }

    #[test]
    fn run_rejects_nonzero_reserved_field() {
        let input = [1_u64];
        let mut output = [0_u64; 1];
        assert_eq!(
            run_buffers(&input, &mut output, &params(1, 1, 0)),
            STATUS_INVALID_PARAMS
        );
    }

    #[test]
    fn run_rejects_length_mismatch() {
        let input = [1_u64, 2];
        let mut output = [0_u64; 1];
        assert_eq!(
            run_buffers(&input, &mut output, &params(1, 0, 0)),
            STATUS_LENGTH_MISMATCH
        );
    }

    #[test]
    fn run_computes_through_the_abi() {
        let input = [1_u64, 2, 3];
        let mut output = [0_u64; 3];
        assert_eq!(
            run_buffers(&input, &mut output, &params(4, 0, 10)),
            STATUS_OK
        );
        for (source, destination) in input.iter().zip(output) {
            assert_eq!((source ^ (source << 4)).wrapping_add(10), destination);
        }
    }
}
