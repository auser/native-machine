//! Native Machine command-line entry point.

mod arena;
mod artifact;
mod cli;
mod config;
mod cpu;
mod host;
mod ops;
mod plugin;

use clap::{CommandFactory, Parser};
use cli::{Cli, Command};
use config::Config;
use std::error::Error;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let config = Config::load(cli.config.as_deref(), cli.root.as_deref())?;
    match cli.command {
        Some(Command::Init { force }) => config.init(force)?,
        Some(Command::InspectHost) => host::print_report(),
        Some(Command::Doctor) => config.doctor()?,
        Some(Command::Config { command }) => match command {
            cli::ConfigCommand::Show => config.print_effective(),
        },
        Some(Command::Kernel { command }) => match command {
            cli::KernelCommand::List => plugin::print_kernel_list(&config)?,
            cli::KernelCommand::Inspect { path } => plugin::inspect(&path)?,
            cli::KernelCommand::Test { path } => plugin::test(&path)?,
            cli::KernelCommand::Install { path } => plugin::install(&config, &path)?,
        },
        Some(Command::Artifact { command }) => match command {
            cli::ArtifactCommand::Inspect { path } => artifact::inspect(Path::new(&path))?,
            cli::ArtifactCommand::Validate { path } => artifact::validate(Path::new(&path))?,
            cli::ArtifactCommand::CreateFixture { path } => {
                artifact::create_fixture(Path::new(&path))?
            }
        },
        Some(Command::Run(cli::RunArgs { artifact, input })) => {
            let mapped =
                artifact::MappedArtifact::open(Path::new(&artifact), config.max_artifact_bytes)?;
            let view = mapped.view()?;
            let operation_section = view
                .section(ops::OPERATION_SECTION)?
                .ok_or("artifact has no operation section")?;
            let mut arena = arena::SessionArena::new();
            let input_bytes = fs::read(&input)?;
            if input_bytes.len() % std::mem::size_of::<f32>() != 0 {
                return Err("input file length must be a multiple of four bytes".into());
            }
            let mut values = [0.0_f32; arena::MAX_VALUES];
            let value_count = input_bytes.len() / std::mem::size_of::<f32>();
            if value_count > values.len() {
                return Err("input exceeds the fixed session arena".into());
            }
            for (index, value) in values.iter_mut().take(value_count).enumerate() {
                let offset = index * std::mem::size_of::<f32>();
                *value = f32::from_le_bytes(input_bytes[offset..offset + 4].try_into()?);
            }
            arena.load_input(&values[..value_count])?;
            let mut scratch = [0.0_f32; arena::MAX_VALUES];
            let plugins = plugin::load_registry(&config)?;
            ops::execute_records_with_plugins(
                &mut arena,
                operation_section.bytes,
                &mut scratch,
                &plugins,
            )?;
            println!(
                "executed artifact {} with input {}: {:?}",
                artifact.display(),
                input.display(),
                arena.output()
            );
        }
        None => Cli::command().print_help()?,
    }
    Ok(())
}

#[cfg(test)]
mod allocation_test_support {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

    pub struct CountingAllocator;

    // SAFETY: each operation delegates to the platform allocator and only adds
    // an atomic counter update.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            System.alloc(layout)
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            System.dealloc(pointer, layout);
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAllocator = CountingAllocator;

    pub fn reset() {
        ALLOCATIONS.store(0, Ordering::Relaxed);
    }
    pub fn count() -> usize {
        ALLOCATIONS.load(Ordering::Relaxed)
    }
}
