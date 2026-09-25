//! Source-independent intermediate representation for operation plans.
//!
//! The IR is the compiler's boundary: a front end (any source language)
//! produces an [`IrPlan`], the lowering adapter emits fixed-width operation
//! records, and the runtime compiles and executes them. The IR knows nothing
//! about parsing; the lowering knows nothing about sources. Certification
//! executes a lowered plan and compares it bitwise against a direct
//! reference evaluation of the same IR.

use crate::ops::{self, ExecuteError};
use crate::plugin::{self, PluginError, PluginRegistry};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum IrOp {
    Copy,
    AddScalar {
        value: f32,
    },
    PluginElementwise {
        kernel: String,
    },
    Matmul {
        kernel: String,
        m: u32,
        k: u32,
        n: u32,
    },
    MatmulAct {
        kernel: String,
        m: u16,
        k: u16,
        n: u16,
        activation: u16,
    },
}

#[derive(Debug, Default)]
pub struct IrPlan {
    ops: Vec<IrOp>,
    input_length: Option<usize>,
}

#[derive(Debug, Error)]
pub enum IrError {
    #[error("kernel is not registered: {0}")]
    UnknownKernel(String),
    #[error("plan source line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("IR operation {index} does not match the chain length")]
    Chain { index: usize },
    #[error("kernel {kernel} declares operation kind {actual}, the IR needs {expected}")]
    KernelKind {
        kernel: String,
        expected: u16,
        actual: u16,
    },
    #[error("plan compilation failed: {0}")]
    Compile(#[from] ExecuteError),
    #[error("plan execution failed: {0}")]
    Execute(ExecuteError),
    #[error("kernel dispatch failed: {0}")]
    Plugin(#[from] PluginError),
    #[error("plan input requires {required} values, capacity is {capacity}")]
    InputTooLarge { required: usize, capacity: usize },
    #[error("certification mismatch at element {index}: plan {plan}, reference {reference}")]
    Certification {
        index: usize,
        plan: f32,
        reference: f32,
    },
}

impl IrPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, op: IrOp) {
        self.ops.push(op);
    }

    pub fn ops(&self) -> &[IrOp] {
        &self.ops
    }

    /// Declared input length from the plan source's `input N` directive.
    pub fn input_length(&self) -> Option<usize> {
        self.input_length
    }

    /// Parses a plan source into an IR plan. The format is one operation per
    /// line, with `#` comments and blank lines ignored. An optional
    /// `input N` directive declares the plan's input length, required when
    /// lowering to records (which carry lengths inline):
    ///
    /// ```text
    /// input VALUES
    /// copy
    /// add_scalar VALUE
    /// elementwise KERNEL
    /// matmul KERNEL M K N
    /// matmul_act KERNEL M K N ACTIVATION
    /// ```
    pub fn parse(source: &str) -> Result<Self, IrError> {
        let mut plan = IrPlan::new();
        for (index, raw_line) in source.lines().enumerate() {
            let line = index + 1;
            let text = raw_line.split('#').next().unwrap_or("").trim();
            if text.is_empty() {
                continue;
            }
            let tokens: Vec<&str> = text.split_whitespace().collect();
            match tokens.as_slice() {
                ["input", values] => {
                    if plan.input_length.is_some() {
                        return Err(IrError::Parse {
                            line,
                            message: "duplicate input directive".to_string(),
                        });
                    }
                    plan.input_length = Some(parse_u32(values, line)? as usize);
                }
                ["copy"] => plan.push(IrOp::Copy),
                ["add_scalar", value] => plan.push(IrOp::AddScalar {
                    value: parse_f32(value, line)?,
                }),
                ["elementwise", kernel] => plan.push(IrOp::PluginElementwise {
                    kernel: (*kernel).to_owned(),
                }),
                ["matmul", kernel, m, k, n] => plan.push(IrOp::Matmul {
                    kernel: (*kernel).to_owned(),
                    m: parse_u32(m, line)?,
                    k: parse_u32(k, line)?,
                    n: parse_u32(n, line)?,
                }),
                ["matmul_act", kernel, m, k, n, activation] => plan.push(IrOp::MatmulAct {
                    kernel: (*kernel).to_owned(),
                    m: parse_u16(m, line)?,
                    k: parse_u16(k, line)?,
                    n: parse_u16(n, line)?,
                    activation: parse_u16(activation, line)?,
                }),
                _ => {
                    return Err(IrError::Parse {
                        line,
                        message: format!("unrecognized operation: {text}"),
                    })
                }
            }
        }
        Ok(plan)
    }

    /// Lowers the IR to fixed-width operation records, resolving kernel names
    /// to registry indices, validating each kernel's declared operation kind,
    /// and filling in chain lengths starting from `input_length`.
    pub fn lower(
        &self,
        registry: &PluginRegistry,
        input_length: usize,
    ) -> Result<Vec<u8>, IrError> {
        let mut records = Vec::with_capacity(self.ops.len() * ops::OPERATION_BYTES);
        let mut expected = input_length;
        for (op_index, op) in self.ops.iter().enumerate() {
            let mut record = [0_u8; ops::OPERATION_BYTES];
            match op {
                IrOp::Copy => {
                    write_elementwise_lengths(&mut record, expected, op_index)?;
                }
                IrOp::AddScalar { value } => {
                    record[0..2].copy_from_slice(&1_u16.to_le_bytes());
                    record[4..8].copy_from_slice(&value.to_le_bytes());
                    write_elementwise_lengths(&mut record, expected, op_index)?;
                }
                IrOp::PluginElementwise { kernel } => {
                    let index = resolve(registry, kernel, plugin::OPERATION_ELEMENTWISE)?;
                    record[0..2].copy_from_slice(&ops::PLUGIN_OPCODE.to_le_bytes());
                    record[2..4].copy_from_slice(&index.to_le_bytes());
                    write_elementwise_lengths(&mut record, expected, op_index)?;
                }
                IrOp::Matmul { kernel, m, k, n } => {
                    if (*m * *k + *k * *n) as usize != expected {
                        return Err(IrError::Chain { index: op_index });
                    }
                    let index = resolve(registry, kernel, plugin::OPERATION_MATMUL)?;
                    record[0..2].copy_from_slice(&ops::PLUGIN_MATMUL_OPCODE.to_le_bytes());
                    record[2..4].copy_from_slice(&index.to_le_bytes());
                    record[4..8].copy_from_slice(&m.to_le_bytes());
                    record[8..12].copy_from_slice(&k.to_le_bytes());
                    record[12..16].copy_from_slice(&n.to_le_bytes());
                    expected = (*m * *n) as usize;
                }
                IrOp::MatmulAct {
                    kernel,
                    m,
                    k,
                    n,
                    activation,
                } => {
                    if usize::from(*m) * usize::from(*k) + usize::from(*k) * usize::from(*n)
                        != expected
                    {
                        return Err(IrError::Chain { index: op_index });
                    }
                    let index = resolve(registry, kernel, plugin::OPERATION_MATMUL_ACT)?;
                    record[0..2].copy_from_slice(&ops::PLUGIN_MATMUL_ACT_OPCODE.to_le_bytes());
                    record[2..4].copy_from_slice(&index.to_le_bytes());
                    record[4..6].copy_from_slice(&m.to_le_bytes());
                    record[6..8].copy_from_slice(&k.to_le_bytes());
                    record[8..10].copy_from_slice(&n.to_le_bytes());
                    record[10..12].copy_from_slice(&activation.to_le_bytes());
                    expected = usize::from(*m) * usize::from(*n);
                }
            }
            records.extend_from_slice(&record);
        }
        Ok(records)
    }

    /// Executes the IR directly with scalar elementwise semantics and typed
    /// registry dispatch, independent of the plan machinery. This is the
    /// certification oracle.
    fn evaluate(&self, input: &[f32], registry: &PluginRegistry) -> Result<Vec<f32>, IrError> {
        let mut current = input.to_vec();
        for op in &self.ops {
            match op {
                IrOp::Copy => {}
                IrOp::AddScalar { value } => {
                    for element in &mut current {
                        *element += *value;
                    }
                }
                IrOp::PluginElementwise { kernel } => {
                    let mut output = vec![0.0_f32; current.len()];
                    registry.run_f32(kernel, &current, &mut output)?;
                    current = output;
                }
                IrOp::Matmul { kernel, m, k, n } => {
                    let a_len = (*m * *k) as usize;
                    let (a, b) = current[..a_len + (*k * *n) as usize].split_at(a_len);
                    let mut output = vec![0.0_f32; (*m * *n) as usize];
                    registry.run_matmul(kernel, a, b, &mut output, *m, *k, *n)?;
                    current = output;
                }
                IrOp::MatmulAct {
                    kernel,
                    m,
                    k,
                    n,
                    activation,
                } => {
                    let a_len = usize::from(*m) * usize::from(*k);
                    let (a, b) =
                        current[..a_len + usize::from(*k) * usize::from(*n)].split_at(a_len);
                    let mut output = vec![0.0_f32; usize::from(*m) * usize::from(*n)];
                    registry.run_matmul_act(
                        kernel,
                        a,
                        b,
                        &mut output,
                        u32::from(*m),
                        u32::from(*k),
                        u32::from(*n),
                        u32::from(*activation),
                    )?;
                    current = output;
                }
            }
        }
        Ok(current)
    }

    /// Behavioral certification: lowers the IR, compiles and executes the
    /// plan through the runtime, and compares every output element bitwise
    /// against direct evaluation. A mismatch means the plan machinery
    /// (lowering, compilation, buffer chaining, fusion) changed semantics.
    pub fn certify(
        &self,
        input: &[f32],
        registry: &PluginRegistry,
        scratch: &mut [f32],
    ) -> Result<Vec<f32>, IrError> {
        if input.len() > crate::arena::MAX_VALUES {
            return Err(IrError::InputTooLarge {
                required: input.len(),
                capacity: crate::arena::MAX_VALUES,
            });
        }
        let records = self.lower(registry, input.len())?;
        let plan = ops::compile_plan(&records, input.len())?;
        let mut arena = crate::arena::SessionArena::new();
        arena
            .load_input(input)
            .map_err(|_| IrError::InputTooLarge {
                required: input.len(),
                capacity: crate::arena::MAX_VALUES,
            })?;
        ops::execute_compiled_plan(&mut arena, &plan, scratch, registry)
            .map_err(IrError::Execute)?;
        let reference = self.evaluate(input, registry)?;
        let output = arena.output();
        if output.len() != reference.len() {
            return Err(IrError::Certification {
                index: output.len().min(reference.len()),
                plan: f32::NAN,
                reference: f32::NAN,
            });
        }
        for (index, (plan, reference)) in output.iter().zip(&reference).enumerate() {
            if plan.to_bits() != reference.to_bits() {
                return Err(IrError::Certification {
                    index,
                    plan: *plan,
                    reference: *reference,
                });
            }
        }
        Ok(output.to_vec())
    }
}

fn parse_f32(token: &str, line: usize) -> Result<f32, IrError> {
    token.parse::<f32>().map_err(|_| IrError::Parse {
        line,
        message: format!("invalid number: {token}"),
    })
}

fn parse_u32(token: &str, line: usize) -> Result<u32, IrError> {
    token.parse::<u32>().map_err(|_| IrError::Parse {
        line,
        message: format!("invalid dimension: {token}"),
    })
}

fn parse_u16(token: &str, line: usize) -> Result<u16, IrError> {
    token.parse::<u16>().map_err(|_| IrError::Parse {
        line,
        message: format!("invalid dimension: {token}"),
    })
}

fn write_elementwise_lengths(
    record: &mut [u8; ops::OPERATION_BYTES],
    length: usize,
    op_index: usize,
) -> Result<(), IrError> {
    let length = u32::try_from(length).map_err(|_| IrError::Chain { index: op_index })?;
    record[8..12].copy_from_slice(&length.to_le_bytes());
    record[12..16].copy_from_slice(&length.to_le_bytes());
    Ok(())
}

fn resolve(registry: &PluginRegistry, kernel: &str, expected_kind: u16) -> Result<u16, IrError> {
    let index = registry
        .kernel_index(kernel)
        .ok_or_else(|| IrError::UnknownKernel(kernel.to_owned()))?;
    let actual = registry
        .kernel_operation(kernel)
        .ok_or_else(|| IrError::UnknownKernel(kernel.to_owned()))?;
    if actual != expected_kind {
        return Err(IrError::KernelKind {
            kernel: kernel.to_owned(),
            expected: expected_kind,
            actual,
        });
    }
    u16::try_from(index).map_err(|_| IrError::UnknownKernel(kernel.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_add_scalar_and_validates_kernel_kinds() {
        let registry = crate::plugin::tests::test_registry();
        let mut plan = IrPlan::new();
        plan.push(IrOp::AddScalar { value: 1.0 });
        plan.push(IrOp::PluginElementwise {
            kernel: "test-elementwise".to_owned(),
        });
        let records = plan.lower(&registry, 0).expect("lowering succeeds");
        assert_eq!(records.len(), 2 * ops::OPERATION_BYTES);
        let mut wrong_kind = IrPlan::new();
        wrong_kind.push(IrOp::PluginElementwise {
            kernel: "test-matmul".to_owned(),
        });
        assert!(matches!(
            wrong_kind.lower(&registry, 0),
            Err(IrError::KernelKind { .. })
        ));
        let mut missing = IrPlan::new();
        missing.push(IrOp::PluginElementwise {
            kernel: "missing".to_owned(),
        });
        assert!(matches!(
            missing.lower(&registry, 0),
            Err(IrError::UnknownKernel(_))
        ));
    }

    #[test]
    fn certifies_copy_and_exposes_ops() {
        let registry = crate::plugin::tests::test_registry();
        let mut plan = IrPlan::new();
        plan.push(IrOp::Copy);
        plan.push(IrOp::AddScalar { value: 1.0 });
        assert_eq!(plan.ops().len(), 2);
        assert_eq!(plan.ops()[0], IrOp::Copy);
        let input = [4.0_f32, -1.0, 0.25];
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        plan.certify(&input, &registry, &mut scratch)
            .expect("plan certifies");
    }

    #[test]
    fn parses_plan_source() {
        let source =
            "# a fused plan\ninput 8\n\nmatmul_act avx2-fma-matmul-act 2 2 2 1\nadd_scalar 0.5\n";
        let plan = IrPlan::parse(source).expect("source parses");
        assert_eq!(plan.input_length(), Some(8));
        assert_eq!(plan.ops().len(), 2);
        assert_eq!(
            plan.ops()[0],
            IrOp::MatmulAct {
                kernel: "avx2-fma-matmul-act".to_owned(),
                m: 2,
                k: 2,
                n: 2,
                activation: 1,
            }
        );
    }

    #[test]
    fn parse_reports_line_numbers() {
        assert!(matches!(
            IrPlan::parse("input 4\nbogus op\n"),
            Err(IrError::Parse { line: 2, .. })
        ));
        assert!(matches!(
            IrPlan::parse("input 4\ninput 8\n"),
            Err(IrError::Parse { line: 2, .. })
        ));
        assert!(matches!(
            IrPlan::parse("add_scalar elephant\n"),
            Err(IrError::Parse { line: 1, .. })
        ));
    }

    #[test]
    fn certifies_parsed_source_end_to_end() {
        let registry = crate::plugin::tests::test_registry();
        let plan = IrPlan::parse(
            "input 8\nmatmul_act test-matmul-act 2 2 2 1\nelementwise test-elementwise\n",
        )
        .expect("source parses");
        let input = [1.0_f32, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0];
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        let output = plan
            .certify(&input, &registry, &mut scratch)
            .expect("plan certifies");
        assert_eq!(output, vec![10.0, 1.0, 14.0, 1.0]);
    }

    struct TestRng(u64);

    impl TestRng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn parse_never_panics_on_random_source() {
        let mut rng = TestRng(0x2545_F491_4F6C_DD1D);
        let alphabet: Vec<char> = "abcdefghilmnprstuw_0123456789.- \t\n#".chars().collect();
        for _ in 0..5_000 {
            let length = (rng.next() % 80) as usize;
            let source: String = (0..length)
                .map(|_| alphabet[(rng.next() as usize) % alphabet.len()])
                .collect();
            let first = IrPlan::parse(&source);
            let second = IrPlan::parse(&source);
            match (&first, &second) {
                (Ok(first), Ok(second)) => {
                    assert_eq!(first.ops(), second.ops());
                    assert_eq!(first.input_length(), second.input_length());
                }
                (Err(first), Err(second)) => {
                    assert_eq!(format!("{first:?}"), format!("{second:?}"));
                }
                _ => panic!("parse is nondeterministic on {source:?}"),
            }
        }
    }

    #[test]
    fn certifies_fused_matmul_chain() {
        let registry = crate::plugin::tests::test_registry();
        let mut plan = IrPlan::new();
        plan.push(IrOp::MatmulAct {
            kernel: "test-matmul-act".to_owned(),
            m: 2,
            k: 2,
            n: 2,
            activation: 1,
        });
        plan.push(IrOp::AddScalar { value: 0.5 });
        let input = [1.0_f32, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0];
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        let output = plan
            .certify(&input, &registry, &mut scratch)
            .expect("plan certifies");
        assert_eq!(output, vec![9.5, 0.5, 13.5, 0.5]);
    }

    #[test]
    fn certifies_elementwise_chain_with_plugin() {
        let registry = crate::plugin::tests::test_registry();
        let mut plan = IrPlan::new();
        plan.push(IrOp::AddScalar { value: 1.0 });
        plan.push(IrOp::PluginElementwise {
            kernel: "test-elementwise".to_owned(),
        });
        plan.push(IrOp::AddScalar { value: -2.0 });
        let input = [1.0_f32, -2.0, 3.5, 0.0];
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        plan.certify(&input, &registry, &mut scratch)
            .expect("plan certifies");
    }

    #[test]
    fn certifies_matmul_then_relu_chain() {
        let registry = crate::plugin::tests::test_registry();
        let mut plan = IrPlan::new();
        plan.push(IrOp::Matmul {
            kernel: "test-matmul".to_owned(),
            m: 2,
            k: 3,
            n: 2,
        });
        plan.push(IrOp::PluginElementwise {
            kernel: "test-elementwise".to_owned(),
        });
        let input: Vec<f32> = (1..=12).map(|value| value as f32).collect();
        let mut scratch = [0.0_f32; crate::arena::MAX_VALUES];
        plan.certify(&input, &registry, &mut scratch)
            .expect("plan certifies");
    }
}
