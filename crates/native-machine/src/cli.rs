//! Command-line surface.

use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command(
    name = "native-machine",
    version,
    about = "CPU-native operation runtime"
)]
pub struct Cli {
    #[arg(long, global = true, env = "NATIVE_MACHINE_CONFIG")]
    pub config: Option<PathBuf>,
    #[arg(long, global = true, env = "NATIVE_MACHINE_ROOT")]
    pub root: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    Init {
        #[arg(long)]
        force: bool,
    },
    InspectHost,
    Doctor,
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    Kernel {
        #[command(subcommand)]
        command: KernelCommand,
    },
    Artifact {
        #[command(subcommand)]
        command: ArtifactCommand,
    },
    Run(RunArgs),
}

#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    Show,
}

#[derive(Subcommand, Debug)]
pub enum KernelCommand {
    List,
    Inspect { path: PathBuf },
    Test { path: PathBuf },
    Install { path: PathBuf },
}

#[derive(Subcommand, Debug)]
pub enum ArtifactCommand {
    Inspect { path: PathBuf },
    Validate { path: PathBuf },
    CreateFixture { path: PathBuf },
}

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(long)]
    pub artifact: PathBuf,
    #[arg(long)]
    pub input: PathBuf,
}
