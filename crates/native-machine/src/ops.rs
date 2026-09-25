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
    #[error("plan has more than 16 segments")]
    PlanSegments,
    #[error("compiled plan expects {expected} values, session input has {actual}")]
    PlanLength { expected: usize, actual: usize },
    #[error("could not compute plan identity: {0}")]
    Identity(String),
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

/// Maximum built-in operations fused into one pass; longer runs become
/// multiple consecutive segments.
const MAX_FUSED_OPERATIONS: usize = 64;

/// Maximum segments in a compiled plan (a plugin record or a fused run).
pub const MAX_PLAN_SEGMENTS: usize = 16;

/// Elements processed per strip; the chunk stays L1-resident while every
/// fused operation is applied to it.
const FUSED_STRIP: usize = 256;

/// Where the current value of a chained plan resides between segments.
enum ChainSource {
    ArenaInput,
    ArenaOutput,
    Scratch,
}

#[derive(Clone, Copy, Debug)]
// The size gap is deliberate: Builtin carries a fixed-capacity operand array
// so the compiled plan stays allocation-free and caller-owned.
#[allow(clippy::large_enum_variant)]
enum PlanSegment {
    Builtin {
        operands: [f32; MAX_FUSED_OPERATIONS],
        count: usize,
    },
    Plugin {
        index: u16,
    },
}

impl PlanSegment {
    const EMPTY: Self = PlanSegment::Builtin {
        operands: [0.0; MAX_FUSED_OPERATIONS],
        count: 0,
    };
}

/// A content-addressed, pre-digested operation plan. Compilation validates
/// the record stream once, groups records into segments, and extracts
/// operands; execution then dispatches pre-digested segments in O(1) per
/// operation with no record parsing and no heap allocation.
pub struct CompiledPlan {
    segments: [PlanSegment; MAX_PLAN_SEGMENTS],
    segment_count: usize,
    length: usize,
}

impl CompiledPlan {
    fn segments(&self) -> &[PlanSegment] {
        &self.segments[..self.segment_count]
    }

    fn push(&mut self, segment: PlanSegment) -> Result<(), ExecuteError> {
        if self.segment_count == MAX_PLAN_SEGMENTS {
            return Err(ExecuteError::PlanSegments);
        }
        self.segments[self.segment_count] = segment;
        self.segment_count += 1;
        Ok(())
    }

    /// Canonical UOR address of the plan, derived from the segment kinds and
    /// operand bit patterns. Computed on demand; never on the hot path.
    pub fn identity(&self) -> Result<String, ExecuteError> {
        let mut json =
            String::from("{\"format\":\"native-machine-plan\",\"version\":1,\"segments\":[");
        for (index, segment) in self.segments().iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            match segment {
                PlanSegment::Builtin { operands, count } => {
                    json.push_str("{\"kind\":\"builtin\",\"operands\":[");
                    for (offset, operand) in operands[..*count].iter().enumerate() {
                        if offset > 0 {
                            json.push(',');
                        }
                        json.push_str(&operand.to_bits().to_string());
                    }
                    json.push_str("]}");
                }
                PlanSegment::Plugin { index } => {
                    json.push_str(&format!("{{\"kind\":\"plugin\",\"index\":{index}}}"));
                }
            }
        }
        json.push_str("]}");
        let outcome = uor_addr::json::address(json.as_bytes())
            .map_err(|error| ExecuteError::Identity(format!("{error:?}")))?;
        Ok(outcome.address.to_string())
    }
}

/// Validates and compiles a record stream into a [`CompiledPlan`]. All
/// parsing, validation, and operand extraction happens here, once; execution
/// is then free of both. No heap allocation on the success path.
pub fn compile_plan(records: &[u8], input_length: usize) -> Result<CompiledPlan, ExecuteError> {
    let (records, remainder) = records.as_chunks::<OPERATION_BYTES>();
    if !remainder.is_empty() {
        return Err(ExecuteError::RecordAlignment(OPERATION_BYTES));
    }
    let mut plan = CompiledPlan {
        segments: [PlanSegment::EMPTY; MAX_PLAN_SEGMENTS],
        segment_count: 0,
        length: input_length,
    };
    for (index, record) in records.iter().enumerate() {
        let opcode = u16::from_le_bytes([record[0], record[1]]);
        if !matches!(opcode, 0 | 1 | PLUGIN_OPCODE) {
            return Err(ExecuteError::UnknownOpcode(index, opcode));
        }
        let record_length = u32::from_le_bytes(
            record[8..12]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        let output_length = u32::from_le_bytes(
            record[12..16]
                .try_into()
                .map_err(|_| ExecuteError::RecordTruncated(index))?,
        );
        if record_length != output_length {
            return Err(ExecuteError::RecordLength(index));
        }
        if usize::try_from(record_length).map_err(|_| ExecuteError::RecordLength(index))?
            != input_length
        {
            return Err(ExecuteError::RecordChain(index));
        }
        if opcode == PLUGIN_OPCODE {
            let plugin_index = u16::from_le_bytes([record[2], record[3]]);
            plan.push(PlanSegment::Plugin {
                index: plugin_index,
            })?;
            continue;
        }
        let needs_segment = !matches!(
            plan.segments().last(),
            Some(PlanSegment::Builtin { count, .. }) if *count < MAX_FUSED_OPERATIONS
        );
        if needs_segment {
            plan.push(PlanSegment::EMPTY)?;
        }
        if opcode == 1 {
            let value = f32::from_le_bytes(
                record[4..8]
                    .try_into()
                    .map_err(|_| ExecuteError::RecordTruncated(index))?,
            );
            if let Some(PlanSegment::Builtin { operands, count }) = plan
                .segments
                .get_mut(plan.segment_count - 1)
                .map(|segment| {
                    debug_assert!(matches!(segment, PlanSegment::Builtin { .. }));
                    segment
                })
            {
                // count is only incremented here, immediately after a push or
                // below the cap checked above, so indexing cannot overflow.
                if *count < MAX_FUSED_OPERATIONS {
                    operands[*count] = value;
                    *count += 1;
                }
            }
        }
    }
    Ok(plan)
}

/// Executes a compiled plan with O(1) per-operation dispatch: no parsing, no
/// validation, no heap allocation. Buffers ping-pong between the arena output
/// and caller scratch; fused built-in segments run as strip-mined SIMD passes
/// that are bitwise identical to unfused passes.
pub fn execute_compiled_plan<D: PluginDispatch>(
    arena: &mut SessionArena,
    plan: &CompiledPlan,
    scratch: &mut [f32],
    plugins: &D,
) -> Result<(), ExecuteError> {
    let length = arena.input().len();
    if length != plan.length {
        return Err(ExecuteError::PlanLength {
            expected: plan.length,
            actual: length,
        });
    }
    if length > scratch.len() {
        return Err(ExecuteError::Arena(ArenaError::InputTooLarge {
            required: length,
            capacity: scratch.len(),
        }));
    }
    let mut source = ChainSource::ArenaInput;
    for segment in plan.segments() {
        match segment {
            PlanSegment::Builtin { operands, count } => {
                let operands = &operands[..*count];
                match source {
                    ChainSource::ArenaInput => {
                        let (input, output) = arena.input_and_output_mut(length)?;
                        run_builtin_segment(operands, input, output);
                        source = ChainSource::ArenaOutput;
                    }
                    ChainSource::ArenaOutput => {
                        run_builtin_segment(operands, arena.output(), &mut scratch[..length]);
                        source = ChainSource::Scratch;
                    }
                    ChainSource::Scratch => {
                        let output = arena.output_mut(length)?;
                        run_builtin_segment(operands, &scratch[..length], output);
                        source = ChainSource::ArenaOutput;
                    }
                }
            }
            PlanSegment::Plugin { index } => {
                let plugin_index = usize::from(*index);
                match source {
                    ChainSource::ArenaInput => {
                        let (input, output) = arena.input_and_output_mut(length)?;
                        plugins
                            .run_index(plugin_index, input, output)
                            .map_err(ExecuteError::Plugin)?;
                        source = ChainSource::ArenaOutput;
                    }
                    ChainSource::ArenaOutput => {
                        plugins
                            .run_index(plugin_index, arena.output(), &mut scratch[..length])
                            .map_err(ExecuteError::Plugin)?;
                        source = ChainSource::Scratch;
                    }
                    ChainSource::Scratch => {
                        let output = arena.output_mut(length)?;
                        plugins
                            .run_index(plugin_index, &scratch[..length], output)
                            .map_err(ExecuteError::Plugin)?;
                        source = ChainSource::ArenaOutput;
                    }
                }
            }
        }
    }
    if matches!(source, ChainSource::Scratch) {
        let output = arena.output_mut(length)?;
        output.copy_from_slice(&scratch[..length]);
    }
    Ok(())
}

/// Strip-mined fused pass over one compiled segment: the first AddScalar
/// fuses into the input copy, the rest apply in place per strip, all through
/// SIMD helpers. Bitwise identical to one pass per operation.
fn run_builtin_segment(operands: &[f32], input: &[f32], output: &mut [f32]) {
    if operands.is_empty() {
        output.copy_from_slice(input);
        return;
    }
    let (first, rest) = operands.split_at(1);
    for (input_chunk, output_chunk) in input
        .chunks(FUSED_STRIP)
        .zip(output.chunks_mut(FUSED_STRIP))
    {
        copy_add(input_chunk, output_chunk, first[0]);
        for operand in rest {
            add_scalar_in_place(output_chunk, *operand);
        }
    }
}

fn copy_add(input: &[f32], output: &mut [f32], operand: f32) {
    if crate::cpu::features().avx2 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: the cached feature probe only reports AVX2 when the host
            // supports it; the function processes valid slices and handles the
            // scalar tail explicitly.
            unsafe {
                copy_add_avx2(input, output, operand);
            }
            return;
        }
    }
    for (source, destination) in input.iter().zip(output.iter_mut()) {
        *destination = *source + operand;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_add_avx2(input: &[f32], output: &mut [f32], operand: f32) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    // SAFETY: AVX2 was detected by the caller; all loads and stores stay
    // within the slices, which have equal lengths.
    unsafe {
        let broadcast = _mm256_set1_ps(operand);
        let mut index = 0;
        while index + 8 <= input.len() {
            let values = _mm256_loadu_ps(input.as_ptr().add(index));
            _mm256_storeu_ps(
                output.as_mut_ptr().add(index),
                _mm256_add_ps(values, broadcast),
            );
            index += 8;
        }
        for (source, destination) in input[index..].iter().zip(output[index..].iter_mut()) {
            *destination = *source + operand;
        }
    }
}

fn add_scalar_in_place(values: &mut [f32], operand: f32) {
    if crate::cpu::features().avx2 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: the cached feature probe only reports AVX2 when the host
            // supports it; the function processes a valid slice and handles
            // the scalar tail explicitly.
            unsafe {
                add_scalar_in_place_avx2(values, operand);
            }
            return;
        }
    }
    for value in values {
        *value += operand;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scalar_in_place_avx2(values: &mut [f32], operand: f32) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    // SAFETY: AVX2 was detected by the caller; all loads and stores stay
    // within the slice.
    unsafe {
        let broadcast = _mm256_set1_ps(operand);
        let mut index = 0;
        while index + 8 <= values.len() {
            let current = _mm256_loadu_ps(values.as_ptr().add(index));
            _mm256_storeu_ps(
                values.as_mut_ptr().add(index),
                _mm256_add_ps(current, broadcast),
            );
            index += 8;
        }
        for value in &mut values[index..] {
            *value += operand;
        }
    }
}

/// Executes records as a chain: each record consumes the previous record's
/// output (the first consumes the session input). Compiles the records into a
/// [`CompiledPlan`] and executes it; callers dispatching the same plan more
/// than once should compile once with [`compile_plan`] and execute with
/// [`execute_compiled_plan`].
#[cfg(test)]
pub fn execute_chain_with_plugins<D: PluginDispatch>(
    arena: &mut SessionArena,
    records: &[u8],
    scratch: &mut [f32],
    plugins: &D,
) -> Result<(), ExecuteError> {
    let plan = compile_plan(records, arena.input().len())?;
    execute_compiled_plan(arena, &plan, scratch, plugins)
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
    fn compiled_plan_matches_chained_executor_bitwise() {
        let input = [1.0_f32, -2.0, 3.5];
        let records = chain_records(&[(1, 1.0), (PLUGIN_OPCODE, 0.0), (1, 0.5), (1, 0.25)], 3);
        let mut chained_arena = SessionArena::new();
        chained_arena.load_input(&input).expect("input fits");
        let mut chained_scratch = [0.0_f32; crate::arena::MAX_VALUES];
        execute_chain_with_plugins(
            &mut chained_arena,
            &records,
            &mut chained_scratch,
            &TestPlugin,
        )
        .expect("chain executes");
        let mut compiled_arena = SessionArena::new();
        compiled_arena.load_input(&input).expect("input fits");
        let plan = compile_plan(&records, input.len()).expect("plan compiles");
        let mut compiled_scratch = [0.0_f32; crate::arena::MAX_VALUES];
        execute_compiled_plan(
            &mut compiled_arena,
            &plan,
            &mut compiled_scratch,
            &TestPlugin,
        )
        .expect("plan executes");
        assert_eq!(chained_arena.output().len(), compiled_arena.output().len());
        for (chained, compiled) in chained_arena.output().iter().zip(compiled_arena.output()) {
            assert_eq!(chained.to_bits(), compiled.to_bits());
        }
    }

    #[test]
    fn long_fused_run_splits_segments_and_stays_correct() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let records = chain_records(&vec![(1, 0.5); 70], 2);
        let plan = compile_plan(&records, 2).expect("plan compiles");
        assert!(plan.segments().len() > 1);
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        execute_compiled_plan(&mut arena, &plan, &mut scratch, &TestPlugin).expect("plan executes");
        assert_eq!(arena.output(), &[36.0, 37.0]);
    }

    #[test]
    fn compiled_plan_rejects_length_mismatch() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let records = chain_records(&[(1, 1.0)], 3);
        let plan = compile_plan(&records, 3).expect("plan compiles");
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        assert!(matches!(
            execute_compiled_plan(&mut arena, &plan, &mut scratch, &TestPlugin),
            Err(ExecuteError::PlanLength {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn plan_identity_is_stable_and_content_addressed() {
        let records = chain_records(&[(1, 1.0), (PLUGIN_OPCODE, 0.0)], 2);
        let first = compile_plan(&records, 2)
            .expect("plan compiles")
            .identity()
            .expect("identity computes");
        let second = compile_plan(&records, 2)
            .expect("plan compiles")
            .identity()
            .expect("identity computes");
        assert_eq!(first, second);
        let different = compile_plan(&chain_records(&[(1, 2.0), (PLUGIN_OPCODE, 0.0)], 2), 2)
            .expect("plan compiles")
            .identity()
            .expect("identity computes");
        assert_ne!(first, different);
    }

    #[test]
    fn compiled_execution_does_not_allocate_with_caller_scratch() {
        let mut arena = SessionArena::new();
        arena.load_input(&[1.0, 2.0]).expect("input fits");
        let records = chain_records(&[(1, 1.0), (PLUGIN_OPCODE, 0.0), (1, 1.0)], 2);
        let plan = compile_plan(&records, 2).expect("plan compiles");
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        let tracking = crate::allocation::track();
        execute_compiled_plan(&mut arena, &plan, &mut scratch, &TestPlugin).expect("plan executes");
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
