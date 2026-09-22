//! Example dynamically loaded CPU-native kernel (Native Machine ABI v3).

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_char;

pub const ABI_VERSION: u32 = 3;
const TYPE_F32: u16 = 1;
const OPERATION_ELEMENTWISE: u16 = 1;

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

// SAFETY: `KernelPlugin` is an immutable ABI descriptor. Its only raw pointer
// targets the static, NUL-terminated `NAME` bytes, and `run` is a function
// pointer with no descriptor-owned mutable state.
unsafe impl Sync for KernelPlugin {}

unsafe extern "C" fn run(context: *mut KernelContext) -> i32 {
    if context.is_null() {
        return STATUS_NULL_CONTEXT;
    }
    // SAFETY: the host contract supplies a valid context pointer for this call.
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
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = *source + 1.0;
    }
    STATUS_OK
}

static NAME: &[u8] = b"scalar-add-one\0";

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
    cpu_features: 0,
};

#[no_mangle]
pub extern "C" fn hologram_kernel_plugin_v3() -> *const KernelPlugin {
    &PLUGIN
}
