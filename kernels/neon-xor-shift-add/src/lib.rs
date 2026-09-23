//! NEON xor-shift-add kernel implementing the Native Machine ABI v3.
//!
//! Contract: `output[i] = (input[i] ^ (input[i] << shift)) + add` over
//! contiguous `u64` buffers, with the same wrapping semantics as
//! `reference-xor-shift-add`: `shift` in `0..=63`, the left shift discards
//! bits past bit 63, and the addition wraps modulo 2^64. On AArch64 the hot
//! loop uses NEON two-lane `u64` vector operations; other targets use the
//! scalar fallback. The kernel performs no heap allocation and is
//! deterministic.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_char;

pub const ABI_VERSION: u32 = 3;
const TYPE_U64: u16 = 2;
const OPERATION_XOR_SHIFT_ADD: u16 = 3;
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
const CPU_NEON: u64 = 2;

/// Largest accepted shift; `shift` must be in `0..=MAX_SHIFT`.
pub const MAX_SHIFT: u32 = 63;

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

pub fn xor_shift_add_scalar(input: &[u64], output: &mut [u64], shift: u32, add: u64) {
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = (source ^ (source << shift)).wrapping_add(add);
    }
}

/// `(input[i] ^ (input[i] << shift)).wrapping_add(add)`, NEON-accelerated on
/// AArch64.
pub fn xor_shift_add(input: &[u64], output: &mut [u64], shift: u32, add: u64) {
    #[cfg(target_arch = "aarch64")]
    xor_shift_add_neon(input, output, shift, add);
    #[cfg(not(target_arch = "aarch64"))]
    xor_shift_add_scalar(input, output, shift, add);
}

#[cfg(target_arch = "aarch64")]
fn xor_shift_add_neon(input: &[u64], output: &mut [u64], shift: u32, add: u64) {
    use std::arch::aarch64::{
        vaddq_u64, vdupq_n_s64, vdupq_n_u64, veorq_u64, vld1q_u64, vshlq_u64, vst1q_u64,
    };
    // SAFETY: NEON is mandatory on AArch64, so the target-feature contract of
    // every intrinsic used here is satisfied. All loads and stores stay within
    // the first `vectorized` elements, which is <= input.len() == output.len();
    // NEON accesses do not require alignment. shift <= MAX_SHIFT, so the
    // per-lane shift counts are in range.
    unsafe {
        let shift_counts = vdupq_n_s64(i64::from(shift));
        let addend = vdupq_n_u64(add);
        let vectorized = input.len() / 8 * 8;
        let mut offset = 0;
        while offset < vectorized {
            let first = vld1q_u64(input.as_ptr().add(offset));
            let second = vld1q_u64(input.as_ptr().add(offset + 2));
            let third = vld1q_u64(input.as_ptr().add(offset + 4));
            let fourth = vld1q_u64(input.as_ptr().add(offset + 6));
            let first = vaddq_u64(veorq_u64(first, vshlq_u64(first, shift_counts)), addend);
            let second = vaddq_u64(veorq_u64(second, vshlq_u64(second, shift_counts)), addend);
            let third = vaddq_u64(veorq_u64(third, vshlq_u64(third, shift_counts)), addend);
            let fourth = vaddq_u64(veorq_u64(fourth, vshlq_u64(fourth, shift_counts)), addend);
            vst1q_u64(output.as_mut_ptr().add(offset), first);
            vst1q_u64(output.as_mut_ptr().add(offset + 2), second);
            vst1q_u64(output.as_mut_ptr().add(offset + 4), third);
            vst1q_u64(output.as_mut_ptr().add(offset + 6), fourth);
            offset += 8;
        }
        xor_shift_add_scalar(&input[offset..], &mut output[offset..], shift, add);
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

static NAME: &[u8] = b"neon-xor-shift-add\0";

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
    cpu_features: REQUIRED_FEATURES,
};

#[no_mangle]
pub extern "C" fn hologram_kernel_plugin_v3() -> *const KernelPlugin {
    &PLUGIN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_bitwise_across_shift_range() {
        let input: Vec<u64> = (0..1031_u64)
            .map(|index| index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        for shift in [0, 1, 31, 32, 62, MAX_SHIFT] {
            for add in [0, 1, u64::MAX] {
                let mut expected = vec![0_u64; input.len()];
                let mut actual = vec![0_u64; input.len()];
                reference_xor_shift_add::xor_shift_add(&input, &mut expected, shift, add);
                xor_shift_add(&input, &mut actual, shift, add);
                assert_eq!(expected, actual, "shift {shift}, add {add}");
            }
        }
    }

    #[test]
    fn run_computes_through_the_abi() {
        let input = [1_u64, 2, 3];
        let mut output = [0_u64; 3];
        let mut params = [0_u8; PARAMS_BYTES];
        params[0..4].copy_from_slice(&4_u32.to_le_bytes());
        params[8..16].copy_from_slice(&10_u64.to_le_bytes());
        let mut context = KernelContext {
            input: input.as_ptr().cast(),
            input_bytes: 24,
            output: output.as_mut_ptr().cast(),
            output_bytes: 24,
            params: params.as_ptr(),
            params_bytes: PARAMS_BYTES as u64,
        };
        // SAFETY: the context points at the live buffers above.
        assert_eq!(unsafe { run(&mut context) }, STATUS_OK);
        for (source, destination) in input.iter().zip(output) {
            assert_eq!((source ^ (source << 4)).wrapping_add(10), destination);
        }
    }
}
