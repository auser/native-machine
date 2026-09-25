//! Explicit configuration and first-run workspace initialization.

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("home directory is unavailable")]
    HomeUnavailable,
    #[error("could not read configuration {path}: {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("could not write {path}: {source}")]
    Write { path: PathBuf, source: io::Error },
    #[error("invalid configuration: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("configuration serialization failed: {0}")]
    Serialize(#[from] toml::ser::Error),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FileConfig {
    pub root: Option<PathBuf>,
    pub max_artifact_bytes: u64,
    pub max_plugin_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub root: PathBuf,
    pub max_artifact_bytes: u64,
    pub max_plugin_bytes: u64,
}

impl Config {
    pub fn load(path: Option<&Path>, root_override: Option<&Path>) -> Result<Self, ConfigError> {
        let default_root = root_override
            .map(Path::to_path_buf)
            .or_else(|| env::var_os("NATIVE_MACHINE_ROOT").map(PathBuf::from))
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".native-machine")))
            .ok_or(ConfigError::HomeUnavailable)?;
        let config_path = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| default_root.join("config.toml"));
        let file = if config_path.exists() {
            let bytes = fs::read(&config_path).map_err(|source| ConfigError::Read {
                path: config_path.clone(),
                source,
            })?;
            toml::from_str::<FileConfig>(&String::from_utf8_lossy(&bytes))?
        } else {
            FileConfig {
                root: None,
                max_artifact_bytes: 4 * 1024 * 1024 * 1024,
                max_plugin_bytes: 128 * 1024 * 1024,
            }
        };
        let root = root_override
            .map(Path::to_path_buf)
            .or(file.root)
            .unwrap_or(default_root);
        Ok(Self {
            root,
            max_artifact_bytes: file.max_artifact_bytes,
            max_plugin_bytes: file.max_plugin_bytes,
        })
    }

    pub fn init(&self, force: bool) -> Result<(), ConfigError> {
        for directory in ["cache", "artifacts", "plugins", "logs", "state"] {
            let path = self.root.join(directory);
            fs::create_dir_all(&path).map_err(|source| ConfigError::Write { path, source })?;
        }
        let path = self.root.join("config.toml");
        if force || !path.exists() {
            let file = FileConfig {
                root: Some(self.root.clone()),
                max_artifact_bytes: self.max_artifact_bytes,
                max_plugin_bytes: self.max_plugin_bytes,
            };
            fs::write(&path, toml::to_string_pretty(&file)?).map_err(|source| {
                ConfigError::Write {
                    path: path.clone(),
                    source,
                }
            })?;
        }
        println!("initialized {}", self.root.display());
        Ok(())
    }

    pub fn doctor(&self) -> Result<(), ConfigError> {
        self.init(false)?;
        for directory in ["cache", "artifacts", "plugins", "logs", "state"] {
            let path = self.root.join(directory);
            if !path.is_dir() {
                return Err(ConfigError::Write {
                    path,
                    source: io::Error::other("workspace directory is not a directory"),
                });
            }
        }
        println!("doctor: passed");
        Ok(())
    }

    pub fn print_effective(&self) {
        println!(
            "root = {}\nmax_artifact_bytes = {}\nmax_plugin_bytes = {}",
            self.root.display(),
            self.max_artifact_bytes,
            self.max_plugin_bytes
        );
    }
}
