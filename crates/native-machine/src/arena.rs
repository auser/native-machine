//! Fixed-capacity session storage.

use thiserror::Error;

pub const MAX_VALUES: usize = 4096;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ArenaError {
    #[error("session input requires {required} values, capacity is {capacity}")]
    InputTooLarge { required: usize, capacity: usize },
    #[error("session output requires {required} values, capacity is {capacity}")]
    OutputTooLarge { required: usize, capacity: usize },
}

pub struct SessionArena {
    input: [f32; MAX_VALUES],
    output: [f32; MAX_VALUES],
    input_len: usize,
    output_len: usize,
}

impl SessionArena {
    pub const fn new() -> Self {
        Self {
            input: [0.0; MAX_VALUES],
            output: [0.0; MAX_VALUES],
            input_len: 0,
            output_len: 0,
        }
    }

    pub fn load_input(&mut self, values: &[f32]) -> Result<(), ArenaError> {
        if values.len() > MAX_VALUES {
            return Err(ArenaError::InputTooLarge {
                required: values.len(),
                capacity: MAX_VALUES,
            });
        }
        self.input[..values.len()].copy_from_slice(values);
        self.input_len = values.len();
        Ok(())
    }

    pub fn input(&self) -> &[f32] {
        &self.input[..self.input_len]
    }

    pub fn output(&self) -> &[f32] {
        &self.output[..self.output_len]
    }

    pub fn output_mut(&mut self, length: usize) -> Result<&mut [f32], ArenaError> {
        if length > MAX_VALUES {
            return Err(ArenaError::OutputTooLarge {
                required: length,
                capacity: MAX_VALUES,
            });
        }
        self.output_len = length;
        Ok(&mut self.output[..length])
    }
}

impl Default for SessionArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_input_that_does_not_fit() {
        let mut arena = SessionArena::new();
        let values = [0.0; MAX_VALUES + 1];
        assert!(matches!(
            arena.load_input(&values),
            Err(ArenaError::InputTooLarge { .. })
        ));
    }
}
