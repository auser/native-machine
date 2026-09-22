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

const ABI_VERSION: u32 = 2;
const TYPE_F32: u16 = 1;
const CPU_AVX2: u64 = 1;
const ENTRY_SYMBOL: &[u8] = b"hologram_kernel_plugin_v2\0";
const MANIFEST_SUFFIX: &str = ".manifest.toml";

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

type PluginEntry = unsafe extern "C" fn() -> *const KernelPlugin;

pub struct LoadedPlugin {
    _library: Library,
    run: KernelRun,
    pub name: String,
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
    #[error("unsupported plugin ABI {0}")]
    Abi(u32),
    #[error("plugin name is null")]
    NullName,
    #[error("plugin name is not valid UTF-8")]
    NameEncoding,
    #[error("plugin descriptor declares unsupported buffer types")]
    BufferType,
    #[error("plugin requires unsupported CPU features: {0:#x}")]
    CpuFeatures(u64),
    #[error("plugin declares unsupported scratch memory: {0} bytes")]
    ScratchBytes(u32),
    #[error("plugin kernel failed with status {0}")]
    KernelStatus(i32),
    #[error("plugin length does not fit the ABI")]
    LengthOverflow,
    #[error("plugin input and output lengths differ")]
    BufferLength,
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
            || manifest.abi_version != ABI_VERSION
        {
            return Err(PluginError::HashMismatch(path.display().to_string()));
        }
        let plugin = load(&path)?;
        if plugin.name != manifest.name {
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
    #[cfg(test)]
    pub fn get(&self, name: &str) -> Option<&LoadedPlugin> {
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| &entry.plugin)
    }

    #[cfg(test)]
    pub fn run_into(
        &self,
        name: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), PluginError> {
        self.get(name)
            .ok_or_else(|| PluginError::NotFound(name.to_owned()))?
            .run_into(input, output)
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
            .run_into(input, output)
    }
}

impl crate::ops::PluginDispatch for PluginRegistry {
    fn run_index(&self, index: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        PluginRegistry::run_index(self, index, input, output).map_err(|error| error.to_string())
    }
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
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    for _ in 0..1000 {
        plugin.run_into(&input, &mut output)?;
        if output != [2.0, 3.0, 4.0, 5.0] {
            return Err(PluginError::KernelStatus(-1));
        }
    }
    println!("plugin self-test: passed (1000 persistent dispatches)");
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
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    plugin.run_into(&input, &mut output)?;
    if output != [2.0, 3.0, 4.0, 5.0] {
        return Err(PluginError::KernelStatus(-1));
    }
    println!("plugin self-test: passed ({})", plugin.name);
    Ok(plugin.name)
}

pub fn load(path: &Path) -> Result<LoadedPlugin, PluginError> {
    // SAFETY: the library is retained inside LoadedPlugin for the lifetime of
    // the copied function pointer and descriptor-derived name.
    let library = unsafe { Library::new(path)? };
    // SAFETY: symbol is a fixed, NUL-terminated ABI entry point.
    let entry: Symbol<PluginEntry> = unsafe { library.get(ENTRY_SYMBOL)? };
    // SAFETY: the plugin ABI defines the entry return contract.
    let pointer = unsafe { entry() };
    if pointer.is_null() {
        return Err(PluginError::NullDescriptor);
    }
    // SAFETY: null was checked and the descriptor remains owned by the library.
    let descriptor = unsafe { &*pointer };
    if descriptor.abi_version != ABI_VERSION {
        return Err(PluginError::Abi(descriptor.abi_version));
    }
    if descriptor.input_type != TYPE_F32 || descriptor.output_type != TYPE_F32 {
        return Err(PluginError::BufferType);
    }
    if descriptor.scratch_bytes != 0 {
        return Err(PluginError::ScratchBytes(descriptor.scratch_bytes));
    }
    let available = if crate::cpu::features().avx2 {
        CPU_AVX2
    } else {
        0
    };
    if descriptor.cpu_features & !available != 0 {
        return Err(PluginError::CpuFeatures(descriptor.cpu_features));
    }
    if descriptor.name.is_null() {
        return Err(PluginError::NullName);
    }
    // SAFETY: the ABI requires a static NUL-terminated name.
    let name = unsafe { CStr::from_ptr(descriptor.name) }
        .to_str()
        .map_err(|_| PluginError::NameEncoding)?
        .to_owned();
    Ok(LoadedPlugin {
        _library: library,
        run: descriptor.run,
        name,
    })
}

impl LoadedPlugin {
    pub fn run_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PluginError> {
        if output.len() != input.len() {
            return Err(PluginError::BufferLength);
        }
        let len = input
            .len()
            .try_into()
            .map_err(|_| PluginError::LengthOverflow)?;
        let mut context = KernelContext {
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            len,
        };
        // SAFETY: the caller-owned slices are valid for the duration of the
        // call and have identical lengths; the library remains loaded.
        let status = unsafe { (self.run)(&mut context) };
        if status != 0 {
            return Err(PluginError::KernelStatus(status));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_rejects_unknown_kernel() {
        let registry = PluginRegistry {
            entries: Vec::new(),
        };
        let mut output = [0.0_f32; 1];
        let error = registry
            .run_into("missing", &[1.0], &mut output)
            .expect_err("missing kernel must fail");
        assert!(matches!(error, PluginError::NotFound(name) if name == "missing"));
    }
}
