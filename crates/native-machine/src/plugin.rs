//! Kernel discovery, admission, and the versioned dynamic plugin boundary.

use crate::config::Config;
use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::{c_char, CStr};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use thiserror::Error;

pub const ABI_VERSION: u32 = 3;
pub const TYPE_F32: u16 = 1;
pub const TYPE_U64: u16 = 2;
pub const OPERATION_ELEMENTWISE: u16 = 1;
pub const OPERATION_MATMUL: u16 = 2;
pub const OPERATION_XOR_SHIFT_ADD: u16 = 3;
pub const MAX_SHIFT: u32 = 63;

const CPU_AVX2: u64 = 1;
const CPU_NEON: u64 = 2;
const ENTRY_SYMBOL: &[u8] = b"hologram_kernel_plugin_v3\0";
const LEGACY_ENTRY_SYMBOL: &[u8] = b"hologram_kernel_plugin_v2\0";
const MANIFEST_SUFFIX: &str = ".manifest.toml";
const MAX_NAME_BYTES: usize = 64;
const MAX_ALIGNMENT: u32 = 64;
const MAX_SCRATCH_BYTES: u64 = 1024 * 1024;
const MATMUL_PARAMS_BYTES: usize = 12;
const XOR_SHIFT_ADD_PARAMS_BYTES: usize = 16;

#[derive(Debug, Serialize, Deserialize)]
struct PluginManifest {
    format: u16,
    name: String,
    abi_version: u32,
    artifact_sha256: String,
    file_bytes: u64,
    #[serde(default)]
    cpu_features: Vec<String>,
}

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

/// Minimal header shared by every ABI version; used to report the exact
/// version of a rejected legacy plugin.
#[repr(C)]
struct AbiHeader {
    abi_version: u32,
}

type PluginEntry = unsafe extern "C" fn() -> *const KernelPlugin;
type LegacyPluginEntry = unsafe extern "C" fn() -> *const AbiHeader;

pub struct LoadedPlugin {
    _library: Option<Library>,
    kernel: LoadedKernel,
}

pub struct LoadedKernel {
    run: KernelRun,
    pub name: String,
    input_type: u16,
    output_type: u16,
    operation: u16,
    required_alignment: u32,
}

#[derive(Debug, Error)]
pub enum PluginError {
    #[error("plugin path does not exist: {0}")]
    Missing(String),
    #[error("plugin exceeds configured size limit")]
    TooLarge,
    #[error("plugin could not be loaded: {0}")]
    Load(#[from] libloading::Error),
    #[error("plugin returned a null descriptor")]
    NullDescriptor,
    #[error("unsupported plugin ABI {0}; this runtime requires ABI {ABI_VERSION}")]
    Abi(u32),
    #[error("plugin name is null")]
    NullName,
    #[error("plugin name is not valid UTF-8")]
    NameEncoding,
    #[error("plugin name is empty or exceeds 64 bytes")]
    NameLength,
    #[error("plugin descriptor declares unsupported buffer types")]
    BufferType,
    #[error("plugin descriptor declares unsupported operation kind {0}")]
    Operation(u16),
    #[error("plugin descriptor declares reserved flags {0:#x}")]
    Flags(u32),
    #[error("plugin descriptor declares unsupported alignment {0}")]
    InvalidAlignment(u32),
    #[error("buffer pointer does not satisfy the kernel's required alignment")]
    BufferAlignment,
    #[error("kernel operation kind does not match the requested invocation shape")]
    OperationMismatch,
    #[error("matmul dimensions must be nonzero")]
    InvalidDimensions,
    #[error("matmul input buffers do not match the M*K and K*N dimensions")]
    MatmulInputLength,
    #[error("matmul inputs A and B must form one contiguous row-major buffer")]
    NonContiguousInputs,
    #[error("matmul output buffer does not match M*N elements")]
    OutputTooSmall,
    #[error("xor-shift-add shift {0} exceeds the valid range 0..={MAX_SHIFT}")]
    InvalidShift(u32),
    #[error("plugin requires unsupported CPU features: {0:#x}")]
    CpuFeatures(u64),
    #[error("plugin declares unsupported scratch memory: {0} bytes")]
    ScratchBytes(u64),
    #[error("plugin kernel failed with status {0}")]
    KernelStatus(i32),
    #[error("plugin length does not fit the ABI")]
    LengthOverflow,
    #[error("plugin input and output lengths differ")]
    BufferLength,
    #[error("plugin admission probe produced nondeterministic or incomplete output")]
    AdmissionOutput,
    #[error("could not inspect or install plugin: {0}")]
    Io(#[from] std::io::Error),
    #[error("could not serialize plugin manifest: {0}")]
    Manifest(#[from] toml::ser::Error),
    #[error("could not parse plugin manifest: {0}")]
    ManifestParse(#[from] toml::de::Error),
    #[error("plugin manifest is missing for {0}")]
    ManifestMissing(String),
    #[error("plugin hash does not match its manifest: {0}")]
    HashMismatch(String),
    #[error("admitted plugin is not registered: {0}")]
    NotFound(String),
    #[error("kernel demo failed: {0}")]
    Demo(&'static str),
}

pub fn print_kernel_list(config: &Config) -> Result<(), PluginError> {
    println!("scalar (built-in, available)");
    let registry = load_registry(config)?;
    if registry.entries.is_empty() {
        println!("dynamic plugins: none admitted");
    }
    for entry in registry.entries {
        println!("{} (admitted, {})", entry.name, entry.path.display());
    }
    Ok(())
}

pub struct PluginRegistry {
    entries: Vec<RegistryEntry>,
}

struct RegistryEntry {
    name: String,
    path: std::path::PathBuf,
    plugin: LoadedPlugin,
}

pub fn load_registry(config: &Config) -> Result<PluginRegistry, PluginError> {
    let directory = config.root.join("plugins");
    if !directory.is_dir() {
        return Ok(PluginRegistry {
            entries: Vec::new(),
        });
    }
    let mut entries = Vec::new();
    for item in fs::read_dir(directory)? {
        let path = item?.path();
        if !path.is_file() || path.to_string_lossy().ends_with(MANIFEST_SUFFIX) {
            continue;
        }
        let manifest_file = manifest_path(&path);
        if !manifest_file.is_file() {
            return Err(PluginError::ManifestMissing(path.display().to_string()));
        }
        let manifest: PluginManifest = toml::from_str(&fs::read_to_string(&manifest_file)?)?;
        let bytes = fs::read(&path)?;
        let mut digest = Sha256::new();
        digest.update(&bytes);
        if format_digest(digest.finalize()) != manifest.artifact_sha256
            || manifest.file_bytes != bytes.len() as u64
        {
            return Err(PluginError::HashMismatch(path.display().to_string()));
        }
        if manifest.abi_version != ABI_VERSION {
            return Err(PluginError::Abi(manifest.abi_version));
        }
        let plugin = load(&path)?;
        if plugin.kernel.name != manifest.name {
            return Err(PluginError::HashMismatch(path.display().to_string()));
        }
        entries.push(RegistryEntry {
            name: manifest.name,
            path,
            plugin,
        });
    }
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(PluginRegistry { entries })
}

impl PluginRegistry {
    fn kernel(&self, name: &str) -> Result<&LoadedKernel, PluginError> {
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| &entry.plugin.kernel)
            .ok_or_else(|| PluginError::NotFound(name.to_owned()))
    }

    pub fn run_f32(
        &self,
        name: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), PluginError> {
        self.kernel(name)?.run_f32(input, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_matmul(
        &self,
        name: &str,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), PluginError> {
        self.kernel(name)?.run_matmul(a, b, output, m, k, n)
    }

    pub fn run_u64(
        &self,
        name: &str,
        input: &[u64],
        output: &mut [u64],
        shift: u32,
        add: u64,
    ) -> Result<(), PluginError> {
        self.kernel(name)?.run_u64(input, output, shift, add)
    }

    pub fn run_index(
        &self,
        index: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), PluginError> {
        self.entries
            .get(index)
            .ok_or_else(|| PluginError::NotFound(index.to_string()))?
            .plugin
            .kernel
            .run_f32(input, output)
    }
}

impl crate::ops::PluginDispatch for PluginRegistry {
    fn run_index(&self, index: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        PluginRegistry::run_index(self, index, input, output).map_err(|error| error.to_string())
    }
}

impl LoadedKernel {
    fn check_contract(
        &self,
        operation: u16,
        input_type: u16,
        output_type: u16,
    ) -> Result<(), PluginError> {
        if self.operation != operation {
            return Err(PluginError::OperationMismatch);
        }
        if self.input_type != input_type || self.output_type != output_type {
            return Err(PluginError::BufferType);
        }
        Ok(())
    }

    fn check_alignment(&self, pointer: *const u8) -> Result<(), PluginError> {
        if !(pointer as usize).is_multiple_of(self.required_alignment as usize) {
            return Err(PluginError::BufferAlignment);
        }
        Ok(())
    }

    fn invoke(
        &self,
        input: (*const u8, usize),
        output: (*mut u8, usize),
        params: &[u8],
    ) -> Result<(), PluginError> {
        self.check_alignment(input.0)?;
        self.check_alignment(output.0.cast_const())?;
        if !params.is_empty() {
            self.check_alignment(params.as_ptr())?;
        }
        let mut context = KernelContext {
            input: input.0,
            input_bytes: u64::try_from(input.1).map_err(|_| PluginError::LengthOverflow)?,
            output: output.0,
            output_bytes: u64::try_from(output.1).map_err(|_| PluginError::LengthOverflow)?,
            params: if params.is_empty() {
                std::ptr::null()
            } else {
                params.as_ptr()
            },
            params_bytes: u64::try_from(params.len()).map_err(|_| PluginError::LengthOverflow)?,
        };
        // SAFETY: the caller-owned buffers are valid for the duration of the
        // call, their byte lengths and alignment were checked against the
        // descriptor, and the library remains loaded for the kernel lifetime.
        let status = unsafe { (self.run)(&mut context) };
        if status != 0 {
            return Err(PluginError::KernelStatus(status));
        }
        Ok(())
    }

    pub fn run_f32(&self, input: &[f32], output: &mut [f32]) -> Result<(), PluginError> {
        self.check_contract(OPERATION_ELEMENTWISE, TYPE_F32, TYPE_F32)?;
        if input.len() != output.len() {
            return Err(PluginError::BufferLength);
        }
        self.invoke(
            (input.as_ptr().cast(), std::mem::size_of_val(input)),
            (output.as_mut_ptr().cast(), std::mem::size_of_val(output)),
            &[],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_matmul(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), PluginError> {
        self.check_contract(OPERATION_MATMUL, TYPE_F32, TYPE_F32)?;
        if m == 0 || k == 0 || n == 0 {
            return Err(PluginError::InvalidDimensions);
        }
        let expected_a = matmul_element_count(m, k)?;
        let expected_b = matmul_element_count(k, n)?;
        let expected_c = matmul_element_count(m, n)?;
        if a.len() != expected_a || b.len() != expected_b {
            return Err(PluginError::MatmulInputLength);
        }
        if output.len() != expected_c {
            return Err(PluginError::OutputTooSmall);
        }
        if b.as_ptr() != a.as_ptr().wrapping_add(a.len()) {
            return Err(PluginError::NonContiguousInputs);
        }
        let mut params = [0_u8; MATMUL_PARAMS_BYTES];
        params[0..4].copy_from_slice(&m.to_le_bytes());
        params[4..8].copy_from_slice(&k.to_le_bytes());
        params[8..12].copy_from_slice(&n.to_le_bytes());
        let input_elements = a.len() + b.len();
        self.invoke(
            (
                a.as_ptr().cast(),
                input_elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(PluginError::LengthOverflow)?,
            ),
            (output.as_mut_ptr().cast(), std::mem::size_of_val(output)),
            &params,
        )
    }

    pub fn run_u64(
        &self,
        input: &[u64],
        output: &mut [u64],
        shift: u32,
        add: u64,
    ) -> Result<(), PluginError> {
        self.check_contract(OPERATION_XOR_SHIFT_ADD, TYPE_U64, TYPE_U64)?;
        if shift > MAX_SHIFT {
            return Err(PluginError::InvalidShift(shift));
        }
        if input.len() != output.len() {
            return Err(PluginError::BufferLength);
        }
        let mut params = [0_u8; XOR_SHIFT_ADD_PARAMS_BYTES];
        params[0..4].copy_from_slice(&shift.to_le_bytes());
        params[8..16].copy_from_slice(&add.to_le_bytes());
        self.invoke(
            (input.as_ptr().cast(), std::mem::size_of_val(input)),
            (output.as_mut_ptr().cast(), std::mem::size_of_val(output)),
            &params,
        )
    }
}

fn matmul_element_count(rows: u32, columns: u32) -> Result<usize, PluginError> {
    usize::try_from(u64::from(rows) * u64::from(columns)).map_err(|_| PluginError::LengthOverflow)
}

fn check_descriptor(descriptor: &KernelPlugin) -> Result<(), PluginError> {
    if descriptor.abi_version != ABI_VERSION {
        return Err(PluginError::Abi(descriptor.abi_version));
    }
    if !matches!(descriptor.input_type, TYPE_F32 | TYPE_U64)
        || !matches!(descriptor.output_type, TYPE_F32 | TYPE_U64)
    {
        return Err(PluginError::BufferType);
    }
    match descriptor.operation {
        OPERATION_ELEMENTWISE => {
            if descriptor.input_type != descriptor.output_type {
                return Err(PluginError::BufferType);
            }
        }
        OPERATION_MATMUL => {
            if descriptor.input_type != TYPE_F32 || descriptor.output_type != TYPE_F32 {
                return Err(PluginError::BufferType);
            }
        }
        OPERATION_XOR_SHIFT_ADD => {
            if descriptor.input_type != TYPE_U64 || descriptor.output_type != TYPE_U64 {
                return Err(PluginError::BufferType);
            }
        }
        other => return Err(PluginError::Operation(other)),
    }
    if descriptor.flags != 0 {
        return Err(PluginError::Flags(descriptor.flags));
    }
    if descriptor.required_alignment == 0
        || descriptor.required_alignment > MAX_ALIGNMENT
        || !descriptor.required_alignment.is_power_of_two()
    {
        return Err(PluginError::InvalidAlignment(descriptor.required_alignment));
    }
    if descriptor.scratch_bytes > MAX_SCRATCH_BYTES {
        return Err(PluginError::ScratchBytes(descriptor.scratch_bytes));
    }
    let features = crate::cpu::features();
    let available = u64::from(features.avx2) * CPU_AVX2 + u64::from(features.neon) * CPU_NEON;
    if descriptor.cpu_features & !available != 0 {
        return Err(PluginError::CpuFeatures(descriptor.cpu_features));
    }
    if descriptor.name.is_null() {
        return Err(PluginError::NullName);
    }
    // SAFETY: the ABI requires a static NUL-terminated name.
    let name = unsafe { CStr::from_ptr(descriptor.name) };
    if name.to_bytes().is_empty() || name.to_bytes().len() > MAX_NAME_BYTES {
        return Err(PluginError::NameLength);
    }
    Ok(())
}

pub fn inspect(path: &Path) -> Result<(), PluginError> {
    let metadata =
        fs::metadata(path).map_err(|_| PluginError::Missing(path.display().to_string()))?;
    let manifest_path = manifest_path(path);
    let manifest = if manifest_path.exists() {
        fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|text| toml::from_str::<PluginManifest>(&text).ok())
    } else {
        None
    };
    println!(
        "plugin: {}\nbytes: {}\nmanifest: {}\nstatus: requires explicit admission",
        path.display(),
        metadata.len(),
        manifest.map_or_else(
            || "missing".to_string(),
            |value| format!("{} ({})", value.name, manifest_path.display())
        )
    );
    Ok(())
}

pub fn test(path: &Path) -> Result<(), PluginError> {
    let plugin = load(path)?;
    probe_determinism(&plugin.kernel, 1000)?;
    println!("plugin self-test: passed (1000 persistent dispatches)");
    Ok(())
}

pub fn demo(config: &Config) -> Result<(), PluginError> {
    let registry = load_registry(config)?;
    if registry.entries.is_empty() {
        return Err(PluginError::Demo(
            "no admitted kernels; run `just kernel-demo` to build and install the reference kernels",
        ));
    }
    println!("verified {} kernel manifest(s):", registry.entries.len());
    for entry in &registry.entries {
        println!("  {} ({})", entry.name, entry.path.display());
    }

    let relu_input = [-1.5_f32, 0.25, -0.0, 3.5];
    let mut relu_output = [0.0_f32; 4];
    registry.run_f32("reference-relu", &relu_input, &mut relu_output)?;
    println!("relu({relu_input:?}) = {relu_output:?}");

    let ab = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (a, b) = ab.split_at(4);
    let mut c = [0.0_f32; 4];
    registry.run_matmul("reference-matmul", a, b, &mut c, 2, 2, 2)?;
    println!(
        "matmul([1 2; 3 4], [5 6; 7 8]) = [{} {}; {} {}]",
        c[0], c[1], c[2], c[3]
    );

    let xor_input = [1_u64, 42, u64::MAX];
    let mut xor_output = [0_u64; 3];
    registry.run_u64("reference-xor-shift-add", &xor_input, &mut xor_output, 7, 9)?;
    println!("xor-shift-add({xor_input:?}, shift=7, add=9) = {xor_output:?}");

    let mut rejected = [0.0_f32; 4];
    match registry.run_matmul("reference-matmul", a, b, &mut rejected, 3, 2, 2) {
        Err(error) => println!("rejected invalid matmul dimensions: {error}"),
        Ok(()) => return Err(PluginError::Demo("matmul accepted mismatched dimensions")),
    }
    let mut rejected_u64 = [0_u64; 3];
    match registry.run_u64(
        "reference-xor-shift-add",
        &xor_input,
        &mut rejected_u64,
        64,
        9,
    ) {
        Err(error) => println!("rejected invalid shift: {error}"),
        Ok(()) => return Err(PluginError::Demo("xor-shift-add accepted shift 64")),
    }
    let mut short_output = [0.0_f32; 3];
    match registry.run_f32("reference-relu", &relu_input, &mut short_output) {
        Err(error) => println!("rejected undersized output: {error}"),
        Ok(()) => return Err(PluginError::Demo("relu accepted an undersized output")),
    }
    println!("kernel demo: passed");
    Ok(())
}

pub fn install(config: &Config, path: &Path) -> Result<(), PluginError> {
    let metadata =
        fs::metadata(path).map_err(|_| PluginError::Missing(path.display().to_string()))?;
    if metadata.len() > config.max_plugin_bytes {
        return Err(PluginError::TooLarge);
    }
    let descriptor_name = validate_descriptor(path)?;
    let destination = config.root.join("plugins").join(
        path.file_name()
            .ok_or_else(|| PluginError::Missing(path.display().to_string()))?,
    );
    fs::copy(path, &destination)?;
    let bytes = fs::read(&destination)?;
    let mut digest = Sha256::new();
    digest.update(&bytes);
    let manifest = PluginManifest {
        format: 1,
        name: descriptor_name,
        abi_version: ABI_VERSION,
        artifact_sha256: format_digest(digest.finalize()),
        file_bytes: metadata.len(),
        cpu_features: Vec::new(),
    };
    let manifest_file = manifest_path(&destination);
    let temporary = manifest_file.with_extension("tmp");
    let encoded = toml::to_string_pretty(&manifest)?;
    let mut file = File::create(&temporary)?;
    file.write_all(encoded.as_bytes())?;
    file.sync_all()?;
    fs::rename(&temporary, &manifest_file)?;
    println!(
        "installed and tested plugin {}\nmanifest {}",
        destination.display(),
        manifest_file.display()
    );
    Ok(())
}

fn manifest_path(path: &Path) -> std::path::PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(MANIFEST_SUFFIX);
    value.into()
}

fn format_digest(digest: impl AsRef<[u8]>) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest.as_ref() {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn validate_descriptor(path: &Path) -> Result<String, PluginError> {
    let plugin = load(path)?;
    probe_determinism(&plugin.kernel, 2)?;
    println!("plugin self-test: passed ({})", plugin.kernel.name);
    Ok(plugin.kernel.name)
}

fn probe_determinism(kernel: &LoadedKernel, rounds: usize) -> Result<(), PluginError> {
    match (kernel.operation, kernel.input_type) {
        (OPERATION_ELEMENTWISE, TYPE_F32) => {
            let input = [-1.0_f32, 0.0, 1.0, 2.0];
            let mut reference = [f32::NAN; 4];
            kernel.run_f32(&input, &mut reference)?;
            let mut probe = [f32::INFINITY; 4];
            for _ in 1..rounds {
                kernel.run_f32(&input, &mut probe)?;
                if reference
                    .iter()
                    .zip(probe)
                    .any(|(left, right)| left.to_bits() != right.to_bits())
                {
                    return Err(PluginError::AdmissionOutput);
                }
            }
            Ok(())
        }
        (OPERATION_ELEMENTWISE, TYPE_U64) => {
            let input = [0_u64, 1, u64::MAX, 42];
            let mut reference = [0_u64; 4];
            kernel.run_u64_elementwise(&input, &mut reference)?;
            let mut probe = [u64::MAX; 4];
            for _ in 1..rounds {
                kernel.run_u64_elementwise(&input, &mut probe)?;
                if reference != probe {
                    return Err(PluginError::AdmissionOutput);
                }
            }
            Ok(())
        }
        (OPERATION_MATMUL, _) => {
            let ab = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let (a, b) = ab.split_at(4);
            let mut reference = [f32::NAN; 4];
            kernel.run_matmul(a, b, &mut reference, 2, 2, 2)?;
            let mut probe = [f32::INFINITY; 4];
            for _ in 1..rounds {
                kernel.run_matmul(a, b, &mut probe, 2, 2, 2)?;
                if reference
                    .iter()
                    .zip(probe)
                    .any(|(left, right)| left.to_bits() != right.to_bits())
                {
                    return Err(PluginError::AdmissionOutput);
                }
            }
            Ok(())
        }
        (OPERATION_XOR_SHIFT_ADD, _) => {
            let input = [0_u64, 1, u64::MAX, 42];
            let mut reference = [0_u64; 4];
            kernel.run_u64(&input, &mut reference, 13, 7)?;
            let mut probe = [u64::MAX; 4];
            for _ in 1..rounds {
                kernel.run_u64(&input, &mut probe, 13, 7)?;
                if reference != probe {
                    return Err(PluginError::AdmissionOutput);
                }
            }
            Ok(())
        }
        _ => Err(PluginError::BufferType),
    }
}

pub fn load(path: &Path) -> Result<LoadedPlugin, PluginError> {
    // SAFETY: the library is retained inside LoadedPlugin for the lifetime of
    // the copied function pointer and descriptor-derived name.
    let library = unsafe { Library::new(path)? };
    // SAFETY: symbol lookup only borrows the library.
    let entry: Symbol<PluginEntry> = match unsafe { library.get(ENTRY_SYMBOL) } {
        Ok(entry) => entry,
        Err(load_error) => {
            // SAFETY: symbol lookup only borrows the library.
            let legacy: Result<Symbol<LegacyPluginEntry>, libloading::Error> =
                unsafe { library.get(LEGACY_ENTRY_SYMBOL) };
            if let Ok(legacy) = legacy {
                // SAFETY: the legacy ABI v2 entry returns a descriptor whose
                // first field is the fixed-width ABI version.
                let pointer = unsafe { legacy() };
                if !pointer.is_null() {
                    // SAFETY: null was checked; the header is descriptor-owned.
                    let version = unsafe { (*pointer).abi_version };
                    return Err(PluginError::Abi(version));
                }
            }
            return Err(PluginError::Load(load_error));
        }
    };
    // SAFETY: the plugin ABI defines the entry return contract.
    let pointer = unsafe { entry() };
    if pointer.is_null() {
        return Err(PluginError::NullDescriptor);
    }
    // SAFETY: null was checked and the descriptor remains owned by the library.
    let descriptor = unsafe { &*pointer };
    check_descriptor(descriptor)?;
    // SAFETY: the ABI requires a static NUL-terminated name, validated above.
    let name = unsafe { CStr::from_ptr(descriptor.name) }
        .to_str()
        .map_err(|_| PluginError::NameEncoding)?
        .to_owned();
    Ok(LoadedPlugin {
        _library: Some(library),
        kernel: LoadedKernel {
            run: descriptor.run,
            name,
            input_type: descriptor.input_type,
            output_type: descriptor.output_type,
            operation: descriptor.operation,
            required_alignment: descriptor.required_alignment,
        },
    })
}

impl LoadedKernel {
    fn run_u64_elementwise(&self, input: &[u64], output: &mut [u64]) -> Result<(), PluginError> {
        self.check_contract(OPERATION_ELEMENTWISE, TYPE_U64, TYPE_U64)?;
        if input.len() != output.len() {
            return Err(PluginError::BufferLength);
        }
        self.invoke(
            (input.as_ptr().cast(), std::mem::size_of_val(input)),
            (output.as_mut_ptr().cast(), std::mem::size_of_val(output)),
            &[],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn test_elementwise_run(context: *mut KernelContext) -> i32 {
        if context.is_null() {
            return 1;
        }
        // SAFETY: test kernel with the same contract as a validated plugin.
        let context = unsafe { &mut *context };
        if context.input.is_null() || context.output.is_null() {
            return 2;
        }
        if context.input_bytes != context.output_bytes || context.input_bytes % 4 != 0 {
            return 4;
        }
        let len = match usize::try_from(context.input_bytes / 4) {
            Ok(value) => value,
            Err(_) => return 3,
        };
        // SAFETY: lengths validated above against the host contract.
        let input = unsafe { std::slice::from_raw_parts(context.input.cast::<f32>(), len) };
        // SAFETY: lengths validated above against the host contract.
        let output = unsafe { std::slice::from_raw_parts_mut(context.output.cast::<f32>(), len) };
        for (source, destination) in input.iter().zip(output.iter_mut()) {
            *destination = *source + 1.0;
        }
        0
    }

    unsafe extern "C" fn test_matmul_run(context: *mut KernelContext) -> i32 {
        if context.is_null() {
            return 1;
        }
        // SAFETY: test kernel with the same contract as a validated plugin.
        let context = unsafe { &mut *context };
        if context.input.is_null() || context.output.is_null() || context.params.is_null() {
            return 2;
        }
        if context.params_bytes != MATMUL_PARAMS_BYTES as u64 {
            return 6;
        }
        // SAFETY: params length validated above.
        let params = unsafe { std::slice::from_raw_parts(context.params, MATMUL_PARAMS_BYTES) };
        let m = u32::from_le_bytes([params[0], params[1], params[2], params[3]]) as usize;
        let k = u32::from_le_bytes([params[4], params[5], params[6], params[7]]) as usize;
        let n = u32::from_le_bytes([params[8], params[9], params[10], params[11]]) as usize;
        if m == 0 || k == 0 || n == 0 {
            return 7;
        }
        let input_values = match usize::try_from(context.input_bytes / 4) {
            Ok(value) => value,
            Err(_) => return 3,
        };
        let output_values = match usize::try_from(context.output_bytes / 4) {
            Ok(value) => value,
            Err(_) => return 3,
        };
        if input_values != m * k + k * n || output_values != m * n {
            return 4;
        }
        // SAFETY: lengths validated above against the host contract.
        let input =
            unsafe { std::slice::from_raw_parts(context.input.cast::<f32>(), input_values) };
        // SAFETY: lengths validated above against the host contract.
        let output =
            unsafe { std::slice::from_raw_parts_mut(context.output.cast::<f32>(), output_values) };
        let (a, b) = input.split_at(m * k);
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0_f32;
                for inner in 0..k {
                    sum += a[row * k + inner] * b[inner * n + column];
                }
                output[row * n + column] = sum;
            }
        }
        0
    }

    unsafe extern "C" fn test_xor_shift_add_run(context: *mut KernelContext) -> i32 {
        if context.is_null() {
            return 1;
        }
        // SAFETY: test kernel with the same contract as a validated plugin.
        let context = unsafe { &mut *context };
        if context.input.is_null() || context.output.is_null() || context.params.is_null() {
            return 2;
        }
        if context.params_bytes != XOR_SHIFT_ADD_PARAMS_BYTES as u64 {
            return 6;
        }
        // SAFETY: params length validated above.
        let params =
            unsafe { std::slice::from_raw_parts(context.params, XOR_SHIFT_ADD_PARAMS_BYTES) };
        let shift = u32::from_le_bytes([params[0], params[1], params[2], params[3]]);
        let add = u64::from_le_bytes([
            params[8], params[9], params[10], params[11], params[12], params[13], params[14],
            params[15],
        ]);
        if shift > MAX_SHIFT {
            return 6;
        }
        if context.input_bytes != context.output_bytes || context.input_bytes % 8 != 0 {
            return 4;
        }
        let len = match usize::try_from(context.input_bytes / 8) {
            Ok(value) => value,
            Err(_) => return 3,
        };
        // SAFETY: lengths validated above against the host contract.
        let input = unsafe { std::slice::from_raw_parts(context.input.cast::<u64>(), len) };
        // SAFETY: lengths validated above against the host contract.
        let output = unsafe { std::slice::from_raw_parts_mut(context.output.cast::<u64>(), len) };
        for (source, destination) in input.iter().zip(output.iter_mut()) {
            *destination = (source ^ (source << shift)).wrapping_add(add);
        }
        0
    }

    fn test_kernel(
        name: &str,
        run: KernelRun,
        input_type: u16,
        output_type: u16,
        operation: u16,
    ) -> LoadedKernel {
        LoadedKernel {
            run,
            name: name.to_owned(),
            input_type,
            output_type,
            operation,
            required_alignment: 4,
        }
    }

    fn test_registry() -> PluginRegistry {
        PluginRegistry {
            entries: vec![
                RegistryEntry {
                    name: "test-elementwise".to_owned(),
                    path: std::path::PathBuf::from("test"),
                    plugin: LoadedPlugin {
                        _library: None,
                        kernel: test_kernel(
                            "test-elementwise",
                            test_elementwise_run,
                            TYPE_F32,
                            TYPE_F32,
                            OPERATION_ELEMENTWISE,
                        ),
                    },
                },
                RegistryEntry {
                    name: "test-matmul".to_owned(),
                    path: std::path::PathBuf::from("test"),
                    plugin: LoadedPlugin {
                        _library: None,
                        kernel: test_kernel(
                            "test-matmul",
                            test_matmul_run,
                            TYPE_F32,
                            TYPE_F32,
                            OPERATION_MATMUL,
                        ),
                    },
                },
                RegistryEntry {
                    name: "test-xor-shift-add".to_owned(),
                    path: std::path::PathBuf::from("test"),
                    plugin: LoadedPlugin {
                        _library: None,
                        kernel: test_kernel(
                            "test-xor-shift-add",
                            test_xor_shift_add_run,
                            TYPE_U64,
                            TYPE_U64,
                            OPERATION_XOR_SHIFT_ADD,
                        ),
                    },
                },
            ],
        }
    }

    static DESCRIPTOR_NAME: &[u8] = b"test-descriptor\0";

    fn valid_descriptor() -> KernelPlugin {
        KernelPlugin {
            abi_version: ABI_VERSION,
            name: DESCRIPTOR_NAME.as_ptr().cast(),
            run: test_elementwise_run,
            input_type: TYPE_F32,
            output_type: TYPE_F32,
            operation: OPERATION_ELEMENTWISE,
            flags: 0,
            required_alignment: 4,
            scratch_bytes: 0,
            cpu_features: 0,
        }
    }

    #[test]
    fn descriptor_v3_is_accepted() {
        assert!(check_descriptor(&valid_descriptor()).is_ok());
    }

    #[test]
    fn descriptor_v2_is_rejected() {
        let mut descriptor = valid_descriptor();
        descriptor.abi_version = 2;
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::Abi(2))
        ));
    }

    #[test]
    fn descriptor_rejects_unknown_type() {
        let mut descriptor = valid_descriptor();
        descriptor.input_type = 7;
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::BufferType)
        ));
    }

    #[test]
    fn descriptor_rejects_unknown_operation() {
        let mut descriptor = valid_descriptor();
        descriptor.operation = 9;
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::Operation(9))
        ));
    }

    #[test]
    fn descriptor_rejects_mismatched_elementwise_types() {
        let mut descriptor = valid_descriptor();
        descriptor.output_type = TYPE_U64;
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::BufferType)
        ));
    }

    #[test]
    fn descriptor_rejects_matmul_with_u64_types() {
        let mut descriptor = valid_descriptor();
        descriptor.operation = OPERATION_MATMUL;
        descriptor.input_type = TYPE_U64;
        descriptor.output_type = TYPE_U64;
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::BufferType)
        ));
    }

    #[test]
    fn descriptor_rejects_reserved_flags() {
        let mut descriptor = valid_descriptor();
        descriptor.flags = 1;
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::Flags(1))
        ));
    }

    #[test]
    fn descriptor_rejects_bad_alignment() {
        for alignment in [0, 3, 128] {
            let mut descriptor = valid_descriptor();
            descriptor.required_alignment = alignment;
            assert!(matches!(
                check_descriptor(&descriptor),
                Err(PluginError::InvalidAlignment(_))
            ));
        }
    }

    #[test]
    fn descriptor_rejects_oversized_scratch() {
        let mut descriptor = valid_descriptor();
        descriptor.scratch_bytes = MAX_SCRATCH_BYTES + 1;
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::ScratchBytes(_))
        ));
    }

    #[test]
    fn descriptor_rejects_unsupported_cpu_features() {
        let mut descriptor = valid_descriptor();
        descriptor.cpu_features = 1 << 40;
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::CpuFeatures(_))
        ));
    }

    #[test]
    fn descriptor_rejects_null_name() {
        let mut descriptor = valid_descriptor();
        descriptor.name = std::ptr::null();
        assert!(matches!(
            check_descriptor(&descriptor),
            Err(PluginError::NullName)
        ));
    }

    #[test]
    fn empty_registry_rejects_unknown_kernel() {
        let registry = PluginRegistry {
            entries: Vec::new(),
        };
        let mut output = [0.0_f32; 1];
        let error = registry
            .run_f32("missing", &[1.0], &mut output)
            .expect_err("missing kernel must fail");
        assert!(matches!(error, PluginError::NotFound(name) if name == "missing"));
    }

    #[test]
    fn run_f32_matches_scalar_oracle() {
        let registry = test_registry();
        let input = [-1.5_f32, 0.0, 2.5, 100.0];
        let mut output = [0.0_f32; 4];
        registry
            .run_f32("test-elementwise", &input, &mut output)
            .expect("kernel executes");
        for (source, destination) in input.iter().zip(output) {
            assert_eq!(*source + 1.0, destination);
        }
    }

    #[test]
    fn run_f32_rejects_length_mismatch() {
        let registry = test_registry();
        let mut output = [0.0_f32; 3];
        assert!(matches!(
            registry.run_f32("test-elementwise", &[1.0, 2.0], &mut output),
            Err(PluginError::BufferLength)
        ));
    }

    #[test]
    fn run_f32_rejects_wrong_operation_kind() {
        let registry = test_registry();
        let mut output = [0.0_f32; 2];
        assert!(matches!(
            registry.run_f32("test-matmul", &[1.0, 2.0], &mut output),
            Err(PluginError::OperationMismatch)
        ));
    }

    #[test]
    fn run_f32_rejects_misaligned_buffer() {
        let mut kernel = test_kernel(
            "aligned",
            test_elementwise_run,
            TYPE_F32,
            TYPE_F32,
            OPERATION_ELEMENTWISE,
        );
        kernel.required_alignment = MAX_ALIGNMENT;
        let input = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 2];
        assert!(matches!(
            kernel.run_f32(&input, &mut output),
            Err(PluginError::BufferAlignment)
        ));
    }

    #[test]
    fn run_matmul_matches_scalar_oracle() {
        let registry = test_registry();
        let ab = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let (a, b) = ab.split_at(6);
        let mut output = [0.0_f32; 4];
        registry
            .run_matmul("test-matmul", a, b, &mut output, 2, 3, 2)
            .expect("kernel executes");
        for row in 0..2 {
            for column in 0..2 {
                let mut sum = 0.0_f32;
                for inner in 0..3 {
                    sum += a[row * 3 + inner] * b[inner * 2 + column];
                }
                assert_eq!(sum, output[row * 2 + column]);
            }
        }
    }

    #[test]
    fn run_matmul_rejects_zero_dimensions() {
        let registry = test_registry();
        let ab = [1.0_f32; 8];
        let (a, b) = ab.split_at(4);
        let mut output = [0.0_f32; 4];
        assert!(matches!(
            registry.run_matmul("test-matmul", a, b, &mut output, 0, 2, 2),
            Err(PluginError::InvalidDimensions)
        ));
    }

    #[test]
    fn run_matmul_rejects_mismatched_input_length() {
        let registry = test_registry();
        let ab = [1.0_f32; 8];
        let (a, b) = ab.split_at(4);
        let mut output = [0.0_f32; 4];
        assert!(matches!(
            registry.run_matmul("test-matmul", a, b, &mut output, 3, 2, 2),
            Err(PluginError::MatmulInputLength)
        ));
    }

    #[test]
    fn run_matmul_rejects_undersized_output() {
        let registry = test_registry();
        let ab = [1.0_f32; 8];
        let (a, b) = ab.split_at(4);
        let mut output = [0.0_f32; 3];
        assert!(matches!(
            registry.run_matmul("test-matmul", a, b, &mut output, 2, 2, 2),
            Err(PluginError::OutputTooSmall)
        ));
    }

    #[test]
    fn run_matmul_rejects_non_contiguous_inputs() {
        let registry = test_registry();
        let buffer = [1.0_f32; 9];
        let (a, rest) = buffer.split_at(4);
        let b = &rest[1..5];
        let mut output = [0.0_f32; 4];
        assert!(matches!(
            registry.run_matmul("test-matmul", a, b, &mut output, 2, 2, 2),
            Err(PluginError::NonContiguousInputs)
        ));
    }

    #[test]
    fn run_u64_matches_scalar_oracle() {
        let registry = test_registry();
        let input = [0_u64, 1, 42, u64::MAX];
        let mut output = [0_u64; 4];
        registry
            .run_u64("test-xor-shift-add", &input, &mut output, 13, 7)
            .expect("kernel executes");
        for (source, destination) in input.iter().zip(output) {
            assert_eq!((source ^ (source << 13)).wrapping_add(7), destination);
        }
    }

    #[test]
    fn run_u64_wraps_on_overflow() {
        let registry = test_registry();
        let input = [u64::MAX];
        let mut output = [0_u64; 1];
        registry
            .run_u64("test-xor-shift-add", &input, &mut output, 63, u64::MAX)
            .expect("kernel executes");
        assert_eq!(
            output[0],
            (u64::MAX ^ (u64::MAX << 63)).wrapping_add(u64::MAX)
        );
    }

    #[test]
    fn run_u64_rejects_invalid_shift() {
        let registry = test_registry();
        let mut output = [0_u64; 1];
        assert!(matches!(
            registry.run_u64("test-xor-shift-add", &[1], &mut output, 64, 0),
            Err(PluginError::InvalidShift(64))
        ));
    }

    #[test]
    fn run_index_dispatches_elementwise_kernel() {
        let registry = test_registry();
        let mut output = [0.0_f32; 2];
        registry
            .run_index(0, &[1.0, 2.0], &mut output)
            .expect("kernel executes");
        assert_eq!(output, [2.0, 3.0]);
        assert!(matches!(
            registry.run_index(1, &[1.0, 2.0], &mut output),
            Err(PluginError::OperationMismatch)
        ));
        assert!(matches!(
            registry.run_index(3, &[1.0, 2.0], &mut output),
            Err(PluginError::NotFound(_))
        ));
    }

    #[test]
    fn typed_dispatch_does_not_allocate_on_success() {
        let registry = test_registry();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        let ab = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (a, b) = ab.split_at(4);
        let mut c = [0.0_f32; 4];
        let words = [1_u64, 2, 3, 4];
        let mut words_output = [0_u64; 4];
        let tracking = crate::allocation_test_support::track();
        registry
            .run_f32("test-elementwise", &input, &mut output)
            .expect("kernel executes");
        registry
            .run_matmul("test-matmul", a, b, &mut c, 2, 2, 2)
            .expect("kernel executes");
        registry
            .run_u64("test-xor-shift-add", &words, &mut words_output, 5, 3)
            .expect("kernel executes");
        let allocations = tracking.count();
        drop(tracking);
        assert_eq!(allocations, 0);
    }

    struct TempPlugins {
        root: std::path::PathBuf,
    }

    impl TempPlugins {
        fn new() -> Self {
            let unique = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "native-machine-plugin-test-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(root.join("plugins")).expect("temp plugins directory");
            Self { root }
        }

        fn config(&self) -> Config {
            Config {
                root: self.root.clone(),
                max_artifact_bytes: 1024,
                max_plugin_bytes: 1024,
            }
        }
    }

    impl Drop for TempPlugins {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn registry_rejects_missing_manifest() {
        let temp = TempPlugins::new();
        std::fs::write(temp.root.join("plugins/libfake.dylib"), b"not a plugin")
            .expect("write fake plugin");
        let error = load_registry(&temp.config())
            .err()
            .expect("missing manifest must fail");
        assert!(matches!(error, PluginError::ManifestMissing(_)));
    }

    #[test]
    fn registry_rejects_hash_mismatch() {
        let temp = TempPlugins::new();
        let plugin_path = temp.root.join("plugins/libfake.dylib");
        std::fs::write(&plugin_path, b"not a plugin").expect("write fake plugin");
        let manifest = PluginManifest {
            format: 1,
            name: "fake".to_owned(),
            abi_version: ABI_VERSION,
            artifact_sha256: "00".repeat(32),
            file_bytes: 12,
            cpu_features: Vec::new(),
        };
        let encoded = toml::to_string_pretty(&manifest).expect("manifest serializes");
        std::fs::write(manifest_path(&plugin_path), encoded).expect("write manifest");
        let error = load_registry(&temp.config())
            .err()
            .expect("hash mismatch must fail");
        assert!(matches!(error, PluginError::HashMismatch(_)));
    }

    #[test]
    fn registry_rejects_legacy_abi_manifest() {
        let temp = TempPlugins::new();
        let plugin_path = temp.root.join("plugins/libfake.dylib");
        let bytes = b"not a plugin";
        std::fs::write(&plugin_path, bytes).expect("write fake plugin");
        let mut digest = Sha256::new();
        digest.update(bytes);
        let manifest = PluginManifest {
            format: 1,
            name: "fake".to_owned(),
            abi_version: 2,
            artifact_sha256: format_digest(digest.finalize()),
            file_bytes: bytes.len() as u64,
            cpu_features: Vec::new(),
        };
        let encoded = toml::to_string_pretty(&manifest).expect("manifest serializes");
        std::fs::write(manifest_path(&plugin_path), encoded).expect("write manifest");
        let error = load_registry(&temp.config())
            .err()
            .expect("legacy ABI must fail");
        assert!(matches!(error, PluginError::Abi(2)));
    }
}
