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
    Demo,
    Bench,
}

#[derive(Subcommand, Debug)]
pub enum ArtifactCommand {
    Inspect {
        path: PathBuf,
    },
    Validate {
        path: PathBuf,
    },
    /// Read encoded tensor bytes once through a bounded caller-owned buffer.
    Stream {
        path: PathBuf,
        #[arg(long, default_value_t = 4096, value_name = "BYTES")]
        buffer_bytes: usize,
    },
    CreateFixture {
        path: PathBuf,
    },
    /// Compile a plan source file into a publishable artifact.
    CreatePlan {
        source: PathBuf,
        output: PathBuf,
    },
}

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(long)]
    pub artifact: PathBuf,
    #[arg(long)]
    pub input: PathBuf,
    /// Sustained bandwidth of the tier supplying non-resident bytes.
    #[arg(long, value_name = "GB/S")]
    pub transport_bandwidth_gbps: Option<f64>,
    /// Tier represented by --transport-bandwidth-gbps.
    #[arg(long, value_enum, default_value_t = TransportTierArg::Memory)]
    pub transport_tier: TransportTierArg,
    /// Tokens produced by one execution pass.
    #[arg(long, default_value_t = 1)]
    pub tokens_per_pass: u32,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum TransportTierArg {
    #[default]
    Memory,
    Storage,
}
