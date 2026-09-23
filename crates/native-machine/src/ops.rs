//! Bounded CPU-native operation executor.

use crate::arena::{ArenaError, SessionArena};
use thiserror::Error;

pub const OPERATION_SECTION: u32 = 1;
pub const OPERATION_BYTES: usize = 16;
pub const PLUGIN_OPCODE: u16 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operation {
    AddScalar { value: f32 },
    Copy,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ExecuteError {
    #[error("arena error: {0}")]
    Arena(#[from] ArenaError),
    #[error("operation record section is not aligned to {0} bytes")]
    RecordAlignment(usize),
    #[error("operation record {0} is truncated")]
    RecordTruncated(usize),
    #[error("operation record {0} has unknown opcode {1}")]
    UnknownOpcode(usize, u16),
    #[error("operation record {0} has incompatible lengths")]
    RecordLength(usize),
    #[error("operation record {0} breaks the input/output length chain")]
    RecordChain(usize),
    #[error("plugin dispatch failed: {0}")]
    Plugin(String),
}

pub trait PluginDispatch {
    fn run_index(&self, index: usize, input: &[f32], output: &mut [f32]) -> Result<(), String>;
}

#[cfg(test)]
pub fn execute(arena: &mut SessionArena, operation: Operation) -> Result<(), ExecuteError> {
    let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
    execute_with_scratch(arena, operation, &mut scratch)
}

pub fn execute_with_scratch(
    arena: &mut SessionArena,
    operation: Operation,
    scratch: &mut [f32],
) -> Result<(), ExecuteError> {
    let input = arena.input();
    if scratch.len() < input.len() {
        return Err(ExecuteError::Arena(ArenaError::InputTooLarge {
            required: input.len(),
            capacity: scratch.len(),
        }));
    }
    scratch[..input.len()].copy_from_slice(input);
    let output = arena.output_mut(input.len())?;
    match operation {
        Operation::AddScalar { value } => {
            add_scalar(&scratch[..output.len()], output, value);
        }
        Operation::Copy => output.copy_from_slice(&scratch[..output.len()]),
    }
    Ok(())
}

fn add_scalar(input: &[f32], output: &mut [f32], value: f32) {
    if crate::cpu::features().avx2 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: the cached feature probe only reports AVX2 when the host
            // supports it; the function processes valid slices and handles the
            // scalar tail explicitly.
            unsafe {
                add_scalar_avx2(input, output, value);
            }
            return;
        }
    }
    add_scalar_scalar(input, output, value);
}

fn add_scalar_scalar(input: &[f32], output: &mut [f32], value: f32) {
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = *source + value;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scalar_avx2(input: &[f32], output: &mut [f32], value: f32) {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };
    let value_vector: __m256 = _mm256_set1_ps(value);
    let mut index = 0;
    while index + 8 <= input.len() {
        let source = _mm256_loadu_ps(input.as_ptr().add(index));
        let result = _mm256_add_ps(source, value_vector);
        _mm256_storeu_ps(output.as_mut_ptr().add(index), result);
        index += 8;
    }
    add_scalar_scalar(&input[index..], &mut output[index..], value);
}

#[cfg(test)]
pub fn execute_records(arena: &mut SessionArena, records: &[u8]) -> Result<(), ExecuteError> {
    let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
    execute_records_with_scratch(arena, records, &mut scratch)
}

pub fn execute_records_with_scratch(
    arena: &mut SessionArena,
    records: &[u8],
    scratch: &mut [f32],
) -> Result<(), ExecuteError> {
    let (records, remainder) = records.as_chunks::<OPERATION_BYTES>();
    if !remainder.is_empty() {
        return Err(ExecuteError::RecordAlignment(OPERATION_BYTES));
    }
    for (index, record) in records.iter().enumerate() {
        let opcode = u16::from_le_bytes([record[0], record[1]]);
        let value = f32::from_le_bytes(
            record[4..8]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        let input_length = u32::from_le_bytes(
            record[8..12]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        let output_length = u32::from_le_bytes(
            record[12..16]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        if input_length != output_length
            || usize::try_from(input_length).map_err(|_| ExecuteError::RecordLength(index))?
                != arena.input().len()
        {
            return Err(ExecuteError::RecordLength(index));
        }
        let operation = match opcode {
            0 => Operation::Copy,
            1 => Operation::AddScalar { value },
            other => return Err(ExecuteError::UnknownOpcode(index, other)),
        };
        execute_with_scratch(arena, operation, scratch)?;
    }
    Ok(())
}

/// Where the current value of a chained plan resides between segments.
enum ChainSource {
    ArenaInput,
    ArenaOutput,
    Scratch,
}

/// Executes records as a chain: each record consumes the previous record's
/// output (the first consumes the session input). Consecutive built-in
/// elementwise records are fused into a single pass: each element is
/// transformed by every operation in the run before moving to the next, so
/// results are bitwise identical to executing the passes separately while
/// the intermediate values never round-trip through memory. Plugin records
/// break fusion runs and are dispatched whole-buffer. No allocation occurs
/// on the success path.
pub fn execute_chain_with_plugins<D: PluginDispatch>(
    arena: &mut SessionArena,
    records: &[u8],
    scratch: &mut [f32],
    plugins: &D,
) -> Result<(), ExecuteError> {
    let (records, remainder) = records.as_chunks::<OPERATION_BYTES>();
    if !remainder.is_empty() {
        return Err(ExecuteError::RecordAlignment(OPERATION_BYTES));
    }
    if records.is_empty() {
        return Ok(());
    }
    let length = arena.input().len();
    if length > scratch.len() {
        return Err(ExecuteError::Arena(ArenaError::InputTooLarge {
            required: length,
            capacity: scratch.len(),
        }));
    }
    // Validate the whole chain before touching any buffer: every record is
    // elementwise (input length equals output length) and matches the chain.
    for (index, record) in records.iter().enumerate() {
        let opcode = u16::from_le_bytes([record[0], record[1]]);
        if !matches!(opcode, 0 | 1 | PLUGIN_OPCODE) {
            return Err(ExecuteError::UnknownOpcode(index, opcode));
        }
        let input_length = u32::from_le_bytes(
            record[8..12]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        let output_length = u32::from_le_bytes(
            record[12..16]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        if input_length != output_length {
            return Err(ExecuteError::RecordLength(index));
        }
        if usize::try_from(input_length).map_err(|_| ExecuteError::RecordLength(index))? != length {
            return Err(ExecuteError::RecordChain(index));
        }
    }
    let mut index = 0;
    let mut source = ChainSource::ArenaInput;
    while index < records.len() {
        // A segment is one plugin record or a maximal run of built-in
        // records (which the executor fuses into one pass).
        let opcode = u16::from_le_bytes([records[index][0], records[index][1]]);
        let mut segment_end = index + 1;
        if opcode != PLUGIN_OPCODE {
            while segment_end < records.len()
                && u16::from_le_bytes([records[segment_end][0], records[segment_end][1]])
                    != PLUGIN_OPCODE
            {
                segment_end += 1;
            }
        }
        let segment = &records[index..segment_end];
        match source {
            ChainSource::ArenaInput => {
                let (input, output) = arena.input_and_output_mut(length)?;
                execute_segment(segment, index, input, output, plugins)?;
                source = ChainSource::ArenaOutput;
            }
            ChainSource::ArenaOutput => {
                execute_segment(
                    segment,
                    index,
                    arena.output(),
                    &mut scratch[..length],
                    plugins,
                )?;
                source = ChainSource::Scratch;
            }
            ChainSource::Scratch => {
                let output = arena.output_mut(length)?;
                execute_segment(segment, index, &scratch[..length], output, plugins)?;
                source = ChainSource::ArenaOutput;
            }
        }
        index = segment_end;
    }
    if matches!(source, ChainSource::Scratch) {
        let output = arena.output_mut(length)?;
        output.copy_from_slice(&scratch[..length]);
    }
    Ok(())
}

/// Maximum built-in operations fused into one pass; longer runs execute as
/// several in-place passes, still without touching intermediate buffers
/// beyond the strip-mined chunks.
const MAX_FUSED_OPERATIONS: usize = 64;

/// Elements processed per strip; the chunk stays L1-resident while every
/// fused operation is applied to it.
const FUSED_STRIP: usize = 256;

fn execute_segment<D: PluginDispatch>(
    segment: &[[u8; OPERATION_BYTES]],
    segment_index: usize,
    input: &[f32],
    output: &mut [f32],
    plugins: &D,
) -> Result<(), ExecuteError> {
    let opcode = u16::from_le_bytes([segment[0][0], segment[0][1]]);
    if opcode == PLUGIN_OPCODE {
        let plugin_index = u16::from_le_bytes([segment[0][2], segment[0][3]]) as usize;
        return plugins
            .run_index(plugin_index, input, output)
            .map_err(ExecuteError::Plugin);
    }
    // Validation guarantees this run contains only Copy and AddScalar
    // records. Operands are parsed once per group, then each strip of
    // elements is transformed by every operation in the group before moving
    // on, so results are bitwise identical to separate passes while the
    // intermediates stay register/L1-resident instead of round-tripping
    // through memory.
    let mut group_start = 0;
    while group_start < segment.len() {
        let group_end = (group_start + MAX_FUSED_OPERATIONS).min(segment.len());
        let group = &segment[group_start..group_end];
        let mut operands = [0.0_f32; MAX_FUSED_OPERATIONS];
        let mut count = 0;
        for (offset, record) in group.iter().enumerate() {
            if u16::from_le_bytes([record[0], record[1]]) == 1 {
                operands[count] = f32::from_le_bytes(
                    record[4..8]
                        .try_into()
                        .map_err(|_| ExecuteError::RecordTruncated(segment_index + offset))?,
                );
                count += 1;
            }
        }
        let operands = &operands[..count];
        if group_start == 0 {
            // The first AddScalar fuses into the input copy, so a fused run
            // costs exactly one pass per operation — the same as unfused
            // passes at cache-resident sizes, and one pass total when the
            // buffers exceed cache.
            let (first, rest) = match operands.split_first() {
                Some((first, rest)) => (*first, rest),
                None => (0.0, &[][..]),
            };
            for (input_chunk, output_chunk) in input
                .chunks(FUSED_STRIP)
                .zip(output.chunks_mut(FUSED_STRIP))
            {
                if count > 0 {
                    for (source, destination) in input_chunk.iter().zip(output_chunk.iter_mut()) {
                        *destination = *source + first;
                    }
                } else {
                    output_chunk.copy_from_slice(input_chunk);
                }
                for operand in rest {
                    for value in output_chunk.iter_mut() {
                        *value += *operand;
                    }
                }
            }
        } else {
            // Later groups refine the output in place; elementwise AddScalar
            // is safe to apply read-modify-write per strip.
            for chunk in output.chunks_mut(FUSED_STRIP) {
                for operand in operands {
                    for value in chunk.iter_mut() {
                        *value += *operand;
                    }
                }
            }
        }
        group_start = group_end;
    }
    Ok(())
}

pub fn execute_records_with_plugins<D: PluginDispatch>(
    arena: &mut SessionArena,
    records: &[u8],
    scratch: &mut [f32],
    plugins: &D,
) -> Result<(), ExecuteError> {
    let (records, remainder) = records.as_chunks::<OPERATION_BYTES>();
    if !remainder.is_empty() {
        return Err(ExecuteError::RecordAlignment(OPERATION_BYTES));
    }
    for (index, record) in records.iter().enumerate() {
        let opcode = u16::from_le_bytes([record[0], record[1]]);
        let input_length = u32::from_le_bytes(
            record[8..12]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        let output_length = u32::from_le_bytes(
            record[12..16]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        if input_length != output_length
            || usize::try_from(input_length).map_err(|_| ExecuteError::RecordLength(index))?
                != arena.input().len()
        {
            return Err(ExecuteError::RecordLength(index));
        }
        if opcode != PLUGIN_OPCODE {
            execute_records_with_scratch(arena, record, scratch)?;
            continue;
        }
        let plugin_index = u16::from_le_bytes([record[2], record[3]]) as usize;
        // Zero-copy: the plugin reads the session input directly; input and
        // output are disjoint arena buffers.
        let input_length = arena.input().len();
        let (input, output) = arena.input_and_output_mut(input_length)?;
        plugins
            .run_index(plugin_index, input, output)
            .map_err(ExecuteError::Plugin)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_operation_is_deterministic() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0, 3.0]).expect("input fits");
        execute(&mut arena, Operation::AddScalar { value: 1.0 }).expect("operation succeeds");
        assert_eq!(arena.output(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn copy_operation_preserves_values() {
        let mut arena = SessionArena::new();
        arena.load_input(&[4.0, 5.0]).expect("input fits");
        execute(&mut arena, Operation::Copy).expect("operation succeeds");
        assert_eq!(arena.output(), &[4.0, 5.0]);
    }

    #[test]
    fn executes_fixed_width_record() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let mut record = [0_u8; OPERATION_BYTES];
        record[0..2].copy_from_slice(&1_u16.to_le_bytes());
        record[4..8].copy_from_slice(&1.0_f32.to_le_bytes());
        record[8..12].copy_from_slice(&2_u32.to_le_bytes());
        record[12..16].copy_from_slice(&2_u32.to_le_bytes());
        execute_records(&mut arena, &record).expect("record executes");
        assert_eq!(arena.output(), &[2.0, 3.0]);
    }

    #[test]
    fn rejects_unknown_opcode() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0]).expect("input fits");
        let mut record = [0_u8; OPERATION_BYTES];
        record[0..2].copy_from_slice(&99_u16.to_le_bytes());
        record[8..12].copy_from_slice(&1_u32.to_le_bytes());
        record[12..16].copy_from_slice(&1_u32.to_le_bytes());
        assert!(matches!(
            execute_records(&mut arena, &record),
            Err(ExecuteError::UnknownOpcode(0, 99))
        ));
    }

    #[test]
    fn serialized_execution_does_not_allocate_with_caller_scratch() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let mut record = [0_u8; OPERATION_BYTES];
        record[0..2].copy_from_slice(&1_u16.to_le_bytes());
        record[4..8].copy_from_slice(&1.0_f32.to_le_bytes());
        record[8..12].copy_from_slice(&2_u32.to_le_bytes());
        record[12..16].copy_from_slice(&2_u32.to_le_bytes());
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        let tracking = crate::allocation::track();
        execute_records_with_scratch(&mut arena, &record, &mut scratch).expect("record executes");
        let allocations = tracking.count();
        drop(tracking);
        assert_eq!(allocations, 0);
    }

    struct TestPlugin;

    impl PluginDispatch for TestPlugin {
        fn run_index(&self, index: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            if index != 0 {
                return Err("unexpected plugin".to_string());
            }
            for (source, destination) in input.iter().zip(output.iter_mut()) {
                *destination = *source + 2.0;
            }
            Ok(())
        }
    }

    fn chain_records(values: &[(u16, f32)], length: u32) -> Vec<u8> {
        let mut records = Vec::new();
        for (opcode, value) in values {
            let mut record = [0_u8; OPERATION_BYTES];
            record[0..2].copy_from_slice(&opcode.to_le_bytes());
            record[4..8].copy_from_slice(&value.to_le_bytes());
            record[8..12].copy_from_slice(&length.to_le_bytes());
            record[12..16].copy_from_slice(&length.to_le_bytes());
            records.extend_from_slice(&record);
        }
        records
    }

    #[test]
    fn chained_builtin_records_match_sequential_passes() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0, 3.0]).expect("input fits");
        let records = chain_records(&[(1, 1.0), (1, 2.0), (1, 3.0)], 3);
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        execute_chain_with_plugins(&mut arena, &records, &mut scratch, &TestPlugin)
            .expect("chain executes");
        assert_eq!(arena.output(), &[7.0, 8.0, 9.0]);
        for (index, value) in [1.0_f32, 2.0, 3.0].iter().enumerate() {
            let sequential = ((value + 1.0) + 2.0) + 3.0;
            assert_eq!(arena.output()[index].to_bits(), sequential.to_bits());
        }
    }

    #[test]
    fn chain_dispatches_plugin_between_builtin_runs() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let records = chain_records(&[(1, 1.0), (PLUGIN_OPCODE, 0.0), (1, 1.0)], 2);
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        execute_chain_with_plugins(&mut arena, &records, &mut scratch, &TestPlugin)
            .expect("chain executes");
        assert_eq!(arena.output(), &[5.0, 6.0]);
    }

    #[test]
    fn chain_result_lands_in_output_for_even_segment_counts() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let records = chain_records(&[(PLUGIN_OPCODE, 0.0), (1, 1.0)], 2);
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        execute_chain_with_plugins(&mut arena, &records, &mut scratch, &TestPlugin)
            .expect("chain executes");
        assert_eq!(arena.output(), &[4.0, 5.0]);
    }

    #[test]
    fn chain_rejects_length_break() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let mut records = chain_records(&[(1, 1.0)], 2);
        let mut record = [0_u8; OPERATION_BYTES];
        record[0..2].copy_from_slice(&1_u16.to_le_bytes());
        record[8..12].copy_from_slice(&4_u32.to_le_bytes());
        record[12..16].copy_from_slice(&4_u32.to_le_bytes());
        records.extend_from_slice(&record);
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        assert!(matches!(
            execute_chain_with_plugins(&mut arena, &records, &mut scratch, &TestPlugin),
            Err(ExecuteError::RecordChain(1))
        ));
        assert_eq!(arena.output().len(), 0);
    }

    #[test]
    fn chain_rejects_unknown_opcode() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let records = chain_records(&[(99, 0.0)], 2);
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        assert!(matches!(
            execute_chain_with_plugins(&mut arena, &records, &mut scratch, &TestPlugin),
            Err(ExecuteError::UnknownOpcode(0, 99))
        ));
    }

    #[test]
    fn chained_execution_does_not_allocate_with_caller_scratch() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let records = chain_records(&[(1, 1.0), (PLUGIN_OPCODE, 0.0), (1, 1.0)], 2);
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        let tracking = crate::allocation::track();
        execute_chain_with_plugins(&mut arena, &records, &mut scratch, &TestPlugin)
            .expect("chain executes");
        let allocations = tracking.count();
        drop(tracking);
        assert_eq!(allocations, 0);
    }

    #[test]
    fn plugin_record_uses_registry_dispatch() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let mut record = [0_u8; OPERATION_BYTES];
        record[0..2].copy_from_slice(&PLUGIN_OPCODE.to_le_bytes());
        record[8..12].copy_from_slice(&2_u32.to_le_bytes());
        record[12..16].copy_from_slice(&2_u32.to_le_bytes());
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        execute_records_with_plugins(&mut arena, &record, &mut scratch, &TestPlugin)
            .expect("plugin executes");
        assert_eq!(arena.output(), &[3.0, 4.0]);
    }
}
