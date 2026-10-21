//! Bounded, forward-only traversal of admitted tensor bytes.

use crate::artifact::{ArtifactError, TensorTable};
use thiserror::Error;

const ELEMENT_ALIGNMENT: usize = std::mem::size_of::<f32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamChunk {
    pub tensor_id: u32,
    pub tensor_offset: u64,
    pub bytes: usize,
    pub final_chunk: bool,
}

#[derive(Debug, Error)]
pub enum StreamError {
    #[error("stream buffer is {actual} bytes, need at least {minimum}")]
    BufferTooSmall { actual: usize, minimum: usize },
    #[error("stream buffer length {actual} is not aligned to {alignment} bytes")]
    BufferAlignment { actual: usize, alignment: usize },
    #[error("tensor stream is exhausted")]
    Exhausted,
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
}

/// Copies encoded tensor bytes into a caller-owned fixed-capacity buffer.
///
/// Exact shared tile ranges are emitted only for their first tensor, so every
/// physical byte in the admitted tile section crosses the stream once.
pub struct TensorStream<'a> {
    tensors: TensorTable<'a>,
    tensor_index: usize,
    tensor_offset: usize,
    consumed_through: u64,
}

impl<'a> TensorStream<'a> {
    pub fn new(tensors: TensorTable<'a>) -> Self {
        Self {
            tensors,
            tensor_index: 0,
            tensor_offset: 0,
            consumed_through: 0,
        }
    }

    pub fn bytes_read(&self) -> u64 {
        self.consumed_through
    }

    pub fn read_next(&mut self, buffer: &mut [u8]) -> Result<StreamChunk, StreamError> {
        if buffer.len() < ELEMENT_ALIGNMENT {
            return Err(StreamError::BufferTooSmall {
                actual: buffer.len(),
                minimum: ELEMENT_ALIGNMENT,
            });
        }
        if !buffer.len().is_multiple_of(ELEMENT_ALIGNMENT) {
            return Err(StreamError::BufferAlignment {
                actual: buffer.len(),
                alignment: ELEMENT_ALIGNMENT,
            });
        }

        loop {
            let Some(tensor) = self.tensors.get(self.tensor_index)? else {
                return Err(StreamError::Exhausted);
            };

            // Admission permits exact aliases for tied weights. The first
            // record transports the shared bytes; later aliases cost no I/O.
            if self.tensor_offset == 0 && tensor.tile_offset < self.consumed_through {
                self.tensor_index += 1;
                continue;
            }

            let remaining = &tensor.bytes[self.tensor_offset..];
            let byte_count = remaining.len().min(buffer.len());
            buffer[..byte_count].copy_from_slice(&remaining[..byte_count]);
            let chunk = StreamChunk {
                tensor_id: tensor.id,
                tensor_offset: self.tensor_offset as u64,
                bytes: byte_count,
                final_chunk: byte_count == remaining.len(),
            };
            self.tensor_offset += byte_count;
            self.consumed_through += byte_count as u64;
            if chunk.final_chunk {
                self.tensor_index += 1;
                self.tensor_offset = 0;
            }
            return Ok(chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation;
    use crate::artifact::{create_artifact_with_raw_tensors, Artifact, RawTensor};
    use crate::ops::OPERATION_BYTES;
    use std::fs;

    fn tensor_artifact(name: &str) -> Vec<u8> {
        let path = std::env::temp_dir().join(format!(
            "native-machine-stream-fixture-{}-{name}.nm",
            std::process::id(),
        ));
        let first = [1.0_f32, 2.0, 3.0];
        let second = [4.0_f32, 5.0];
        let first_bytes: Vec<u8> = first.iter().flat_map(|value| value.to_le_bytes()).collect();
        let second_bytes: Vec<u8> = second
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        create_artifact_with_raw_tensors(
            &path,
            &[0_u8; OPERATION_BYTES],
            br#"{"compiler":"stream-test"}"#,
            &[
                RawTensor {
                    id: 10,
                    shape: &[3],
                    bytes: &first_bytes,
                },
                RawTensor {
                    id: 20,
                    shape: &[2],
                    bytes: &second_bytes,
                },
            ],
        )
        .expect("stream fixture publishes");
        let bytes = fs::read(&path).expect("stream fixture reads");
        fs::remove_file(path).expect("stream fixture is removed");
        bytes
    }

    #[test]
    fn streams_every_encoded_byte_once_with_deterministic_chunks() {
        let bytes = tensor_artifact("chunks");
        let artifact = Artifact::parse(&bytes).expect("artifact parses");
        let tensors = artifact
            .tensors()
            .expect("tensor table parses")
            .expect("tensor table exists");
        let mut stream = TensorStream::new(tensors);
        let mut buffer = [0_u8; 8];
        let mut output = Vec::new();
        let mut chunks = Vec::new();
        loop {
            match stream.read_next(&mut buffer) {
                Ok(chunk) => {
                    output.extend_from_slice(&buffer[..chunk.bytes]);
                    chunks.push(chunk);
                }
                Err(StreamError::Exhausted) => break,
                Err(error) => panic!("stream failed: {error}"),
            }
        }

        let expected: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(output, expected);
        assert_eq!(stream.bytes_read(), expected.len() as u64);
        assert_eq!(
            chunks,
            [
                StreamChunk {
                    tensor_id: 10,
                    tensor_offset: 0,
                    bytes: 8,
                    final_chunk: false,
                },
                StreamChunk {
                    tensor_id: 10,
                    tensor_offset: 8,
                    bytes: 4,
                    final_chunk: true,
                },
                StreamChunk {
                    tensor_id: 20,
                    tensor_offset: 0,
                    bytes: 8,
                    final_chunk: true,
                },
            ]
        );
    }

    #[test]
    fn rejects_invalid_caller_buffers_with_typed_errors() {
        let bytes = tensor_artifact("buffers");
        let artifact = Artifact::parse(&bytes).expect("artifact parses");
        let tensors = artifact
            .tensors()
            .expect("tensor table parses")
            .expect("tensor table exists");
        let mut stream = TensorStream::new(tensors);

        assert!(matches!(
            stream.read_next(&mut []),
            Err(StreamError::BufferTooSmall {
                actual: 0,
                minimum: 4
            })
        ));
        assert!(matches!(
            stream.read_next(&mut [0_u8; 6]),
            Err(StreamError::BufferAlignment {
                actual: 6,
                alignment: 4
            })
        ));
    }

    #[test]
    fn steady_state_streaming_does_not_allocate() {
        let bytes = tensor_artifact("allocations");
        let artifact = Artifact::parse(&bytes).expect("artifact parses");
        let tensors = artifact
            .tensors()
            .expect("tensor table parses")
            .expect("tensor table exists");
        let mut stream = TensorStream::new(tensors);
        let mut buffer = [0_u8; 4];

        let allocations = {
            let guard = allocation::track();
            loop {
                match stream.read_next(&mut buffer) {
                    Ok(_) => {}
                    Err(StreamError::Exhausted) => break,
                    Err(_) => panic!("stream failed"),
                }
            }
            guard.count()
        };
        assert_eq!(allocations, 0);
        assert_eq!(stream.bytes_read(), 20);
    }
}
