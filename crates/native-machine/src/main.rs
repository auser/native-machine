//! Native Machine command-line entry point.

mod allocation;
mod arena;
mod artifact;
mod bench;
mod cli;
mod config;
mod cpu;
mod host;
mod identity;
mod ir;
mod ops;
mod plugin;
mod stream;
mod transport;

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
            cli::KernelCommand::Demo => plugin::demo(&config)?,
            cli::KernelCommand::Bench => bench::run(&config)?,
        },
        Some(Command::Artifact { command }) => match command {
            cli::ArtifactCommand::Inspect { path } => artifact::inspect(Path::new(&path))?,
            cli::ArtifactCommand::Validate { path } => artifact::validate(Path::new(&path))?,
            cli::ArtifactCommand::Stream { path, buffer_bytes } => {
                stream_artifact(Path::new(&path), buffer_bytes)?
            }
            cli::ArtifactCommand::CreateFixture { path } => {
                artifact::create_fixture(Path::new(&path))?
            }
            cli::ArtifactCommand::CreatePlan { source, output } => {
                let text = fs::read_to_string(&source)?;
                let ir = ir::IrPlan::parse(&text)?;
                let input_length = ir
                    .input_length()
                    .ok_or("plan source must declare its input length with `input N`")?;
                let plugins = plugin::load_registry(&config)?;
                let records = ir.lower(&plugins, input_length)?;
                // Compile once here so malformed plans never reach an artifact.
                let plan = ops::compile_plan(&records, input_length)?;
                let provenance = format!(
                    "{{\"compiler\":\"native-machine\",\"source\":\"{}\",\"plan\":\"{}\"}}",
                    source.display(),
                    plan.identity()?
                );
                artifact::create_artifact(Path::new(&output), &records, provenance.as_bytes())?;
                println!("compiled plan {} -> {}", source.display(), output.display());
            }
        },
        Some(Command::Run(cli::RunArgs {
            artifact,
            input,
            transport_bandwidth_gbps,
            transport_tier,
            tokens_per_pass,
        })) => {
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
            // Compile the artifact's operation section once, then dispatch
            // the pre-digested plan (O(1) per operation, no per-record
            // parsing at execution) with deterministic byte accounting.
            let plan = ops::compile_plan(operation_section.bytes, value_count)?;
            let mut trace = ops::ExecutionTrace::new();
            ops::execute_compiled_plan_traced(
                &mut arena,
                &plan,
                &mut scratch,
                &plugins,
                &mut trace,
            )?;
            println!(
                "executed artifact {} with input {}: {:?}\ntrace: {} segment(s), {} bytes read, {} bytes written",
                artifact.display(),
                input.display(),
                arena.output(),
                trace.entries().len(),
                trace.bytes_read(),
                trace.bytes_written()
            );
            if let Some(bandwidth_gbps) = transport_bandwidth_gbps {
                let tier = match transport_tier {
                    cli::TransportTierArg::Memory => transport::TransportTier::Memory,
                    cli::TransportTierArg::Storage => transport::TransportTier::Storage,
                };
                let bound = transport::TransportBound::new(
                    tier,
                    bandwidth_gbps * 1_000_000_000.0,
                    trace.bytes_read(),
                    tokens_per_pass,
                )?;
                println!(
                    "transport ceiling: {:.3} tokens/s at {} GB/s {} bandwidth, {} logical bytes/pass, {} token(s)/pass",
                    bound.tokens_per_second_ceiling(),
                    bandwidth_gbps,
                    bound.tier,
                    bound.bytes_per_pass,
                    bound.tokens_per_pass
                );
            }
        }
        None => Cli::command().print_help()?,
    }
    Ok(())
}

fn stream_artifact(path: &Path, buffer_bytes: usize) -> Result<(), Box<dyn Error>> {
    let mapped = artifact::MappedArtifact::open(path, u64::MAX)?;
    let artifact = mapped.view()?;
    let tensors = artifact
        .tensors()?
        .ok_or("artifact has no tensor sections")?;
    let mut buffer = vec![0_u8; buffer_bytes];
    let mut stream = stream::TensorStream::new(tensors);
    let mut chunk_count = 0_u64;
    loop {
        match stream.read_next(&mut buffer) {
            Ok(_) => chunk_count += 1,
            Err(stream::StreamError::Exhausted) => break,
            Err(error) => return Err(error.into()),
        }
    }
    println!(
        "streamed {} encoded bytes from {} in {} chunk(s) with a {}-byte buffer",
        stream.bytes_read(),
        path.display(),
        chunk_count,
        buffer_bytes
    );
    Ok(())
}
