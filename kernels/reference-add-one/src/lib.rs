#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_char;

#[repr(C)]
pub struct KernelContext {
    pub input: *const f32,
    pub output: *mut f32,
    pub len: u64,
}

pub type KernelRun = unsafe extern "C" fn(*mut KernelContext) -> i32;

#[repr(C)]
pub struct KernelPlugin {
    pub abi_version: u32,
    pub name: *const c_char,
    pub run: KernelRun,
    pub input_type: u16,
    pub output_type: u16,
    pub flags: u32,
    pub scratch_bytes: u32,
    pub cpu_features: u64,
}

// SAFETY: `KernelPlugin` is an immutable ABI descriptor. Its raw pointer
// targets the static, NUL-terminated `NAME` bytes, and its function pointer
// has no descriptor-owned mutable state.
unsafe impl Sync for KernelPlugin {}

unsafe extern "C" fn run(context: *mut KernelContext) -> i32 {
    if context.is_null() {
        return 1;
    }
    // SAFETY: the host supplies a valid context pointer for this call.
    let context = unsafe { &mut *context };
    if context.input.is_null() || context.output.is_null() {
        return 2;
    }
    let len = match usize::try_from(context.len) {
        Ok(value) => value,
        Err(_) => return 3,
    };
    // SAFETY: the host validates both buffers and their length before calling.
    let input = unsafe { std::slice::from_raw_parts(context.input, len) };
    // SAFETY: the host provides writable output storage for `len` values.
    let output = unsafe { std::slice::from_raw_parts_mut(context.output, len) };
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = *source + 1.0;
    }
    0
}

static NAME: &[u8] = b"reference-add-one\0";

static PLUGIN: KernelPlugin = KernelPlugin {
    abi_version: 2,
    name: NAME.as_ptr().cast(),
    run,
    input_type: 1,
    output_type: 1,
    flags: 0,
    scratch_bytes: 0,
    cpu_features: 0,
};

#[no_mangle]
pub extern "C" fn hologram_kernel_plugin_v2() -> *const KernelPlugin {
    &PLUGIN
}
