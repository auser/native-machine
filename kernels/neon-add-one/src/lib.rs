//! NEON add-one kernel implementing the Native Machine ABI v3.
//!
//! Contract: `output[i] = input[i] + 1.0` over contiguous `f32` buffers,
//! identical to `reference-add-one`. On AArch64 the hot loop uses four-way
//! unrolled NEON vector operations; other targets use the scalar fallback.
//! The kernel performs no heap allocation and is deterministic.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_char;

pub const ABI_VERSION: u32 = 3;
const TYPE_F32: u16 = 1;
const OPERATION_ELEMENTWISE: u16 = 1;
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
const CPU_NEON: u64 = 2;

#[cfg(target_arch = "aarch64")]
const REQUIRED_FEATURES: u64 = CPU_NEON;
#[cfg(not(target_arch = "aarch64"))]
const REQUIRED_FEATURES: u64 = 0;

const STATUS_OK: i32 = 0;
const STATUS_NULL_CONTEXT: i32 = 1;
const STATUS_NULL_BUFFER: i32 = 2;
const STATUS_LENGTH_OVERFLOW: i32 = 3;
const STATUS_LENGTH_MISMATCH: i32 = 4;
const STATUS_MISALIGNED: i32 = 5;
const STATUS_INVALID_PARAMS: i32 = 6;

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

pub fn add_one_scalar(input: &[f32], output: &mut [f32]) {
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = *source + 1.0;
    }
}

/// `output[i] = input[i] + 1.0`, NEON-accelerated on AArch64.
pub fn add_one(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    add_one_neon(input, output);
    #[cfg(not(target_arch = "aarch64"))]
    add_one_scalar(input, output);
}

#[cfg(target_arch = "aarch64")]
fn add_one_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vst1q_f32};
    // SAFETY: NEON is mandatory on AArch64, so the target-feature contract of
    // every intrinsic used here is satisfied. All loads and stores stay within
    // the first `vectorized` elements, which is <= input.len() == output.len();
    // NEON accesses do not require alignment.
    unsafe {
        let ones = vdupq_n_f32(1.0);
        let vectorized = input.len() / 16 * 16;
        let mut offset = 0;
        while offset < vectorized {
            let first = vld1q_f32(input.as_ptr().add(offset));
            let second = vld1q_f32(input.as_ptr().add(offset + 4));
            let third = vld1q_f32(input.as_ptr().add(offset + 8));
            let fourth = vld1q_f32(input.as_ptr().add(offset + 12));
            vst1q_f32(output.as_mut_ptr().add(offset), vaddq_f32(first, ones));
            vst1q_f32(output.as_mut_ptr().add(offset + 4), vaddq_f32(second, ones));
            vst1q_f32(output.as_mut_ptr().add(offset + 8), vaddq_f32(third, ones));
            vst1q_f32(
                output.as_mut_ptr().add(offset + 12),
                vaddq_f32(fourth, ones),
            );
            offset += 16;
        }
        add_one_scalar(&input[offset..], &mut output[offset..]);
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
    if context.params_bytes != 0 {
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
    let element_bytes = std::mem::size_of::<f32>();
    if input_bytes != output_bytes || !input_bytes.is_multiple_of(element_bytes) {
        return STATUS_LENGTH_MISMATCH;
    }
    let alignment = std::mem::align_of::<f32>();
    if !(context.input as usize).is_multiple_of(alignment)
        || !(context.output as usize).is_multiple_of(alignment)
    {
        return STATUS_MISALIGNED;
    }
    let len = input_bytes / element_bytes;
    // SAFETY: the host validated both buffers; their byte lengths are equal
    // and multiple of the element size, and their alignment was checked.
    let input = unsafe { std::slice::from_raw_parts(context.input.cast::<f32>(), len) };
    // SAFETY: the host provides writable output storage for `len` values.
    let output = unsafe { std::slice::from_raw_parts_mut(context.output.cast::<f32>(), len) };
    add_one(input, output);
    STATUS_OK
}

static NAME: &[u8] = b"neon-add-one\0";

static PLUGIN: KernelPlugin = KernelPlugin {
    abi_version: ABI_VERSION,
    name: NAME.as_ptr().cast(),
    run,
    input_type: TYPE_F32,
    output_type: TYPE_F32,
    operation: OPERATION_ELEMENTWISE,
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
    fn matches_reference_bitwise() {
        let mut input = vec![0.0_f32; 1027];
        for (index, value) in input.iter_mut().enumerate() {
            *value = index as f32 * 0.5 - 300.0;
        }
        input[0] = -0.0;
        input[1] = f32::INFINITY;
        input[2] = f32::NEG_INFINITY;
        let mut expected = vec![0.0_f32; input.len()];
        let mut actual = vec![0.0_f32; input.len()];
        reference_add_one::add_one(&input, &mut expected);
        add_one(&input, &mut actual);
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
    }

    #[test]
    fn run_computes_through_the_abi() {
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0_f32; 5];
        let mut context = KernelContext {
            input: input.as_ptr().cast(),
            input_bytes: 20,
            output: output.as_mut_ptr().cast(),
            output_bytes: 20,
            params: std::ptr::null(),
            params_bytes: 0,
        };
        // SAFETY: the context points at the live buffers above.
        assert_eq!(unsafe { run(&mut context) }, STATUS_OK);
        assert_eq!(output, [2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
