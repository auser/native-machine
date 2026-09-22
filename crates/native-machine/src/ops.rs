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
        scratch[..arena.input().len()].copy_from_slice(arena.input());
        let output = arena.output_mut(arena.input().len())?;
        plugins
            .run_index(plugin_index, &scratch[..output.len()], output)
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
        crate::allocation_test_support::reset();
        execute_records_with_scratch(&mut arena, &record, &mut scratch).expect("record executes");
        assert_eq!(crate::allocation_test_support::count(), 0);
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
