//! Borrowed, fixed-width artifact parser and validator.
//!
//! The parser never owns or copies the artifact payload. It validates all
//! ranges before exposing section views.

use std::fs;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

pub const MAGIC: [u8; 8] = *b"NATMACH\0";
pub const FORMAT_VERSION: u16 = 1;
pub const HEADER_BYTES: usize = 32;
pub const SECTION_BYTES: usize = 32;
pub const MAX_SECTIONS: u32 = 65_536;
pub const PROVENANCE_SECTION: u32 = 2;
pub const TENSOR_SECTION: u32 = 3;
pub const TILE_DATA_SECTION: u32 = 4;
pub const TENSOR_RECORD_BYTES: usize = 48;
pub const ELEMENT_F32: u16 = 1;
pub const CODEC_RAW: u16 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactHeader {
    pub version: u16,
    pub flags: u16,
    pub section_count: u32,
    pub section_table_offset: u64,
    pub artifact_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Section<'a> {
    pub kind: u32,
    pub flags: u32,
    pub offset: u64,
    pub length: u64,
    pub alignment: u64,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorRecord<'a> {
    pub id: u32,
    pub element_type: u16,
    pub codec: u16,
    pub rank: u16,
    pub shape: [u32; 4],
    pub tile_offset: u64,
    pub tile_length: u64,
    pub bytes: &'a [u8],
}

pub struct TensorTable<'a> {
    records: &'a [u8],
    tiles: &'a [u8],
}

impl<'a> TensorTable<'a> {
    pub fn len(&self) -> usize {
        self.records.len() / TENSOR_RECORD_BYTES
    }

    pub fn get(&self, index: usize) -> Result<Option<TensorRecord<'a>>, ArtifactError> {
        if index >= self.len() {
            return Ok(None);
        }
        parse_tensor_record(self.records, self.tiles, index).map(Some)
    }

    pub fn logical_bytes(&self) -> Result<u64, ArtifactError> {
        let mut total = 0_u64;
        for index in 0..self.len() {
            if let Some(tensor) = self.get(index)? {
                total = total
                    .checked_add(tensor.tile_length)
                    .ok_or(ArtifactError::Overflow)?;
            }
        }
        Ok(total)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RawTensor<'a> {
    pub id: u32,
    pub shape: &'a [u32],
    pub bytes: &'a [u8],
}

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("artifact is too small: {actual} bytes, need at least {required}")]
    TooSmall { actual: usize, required: usize },
    #[error("invalid artifact magic")]
    Magic,
    #[error("unsupported artifact version {0}")]
    Version(u16),
    #[error("artifact section count {0} exceeds the maximum")]
    TooManySections(u32),
    #[error("artifact declares {declared} bytes but contains {actual}")]
    LengthMismatch { declared: u64, actual: usize },
    #[error("artifact is {actual} bytes, exceeding the configured limit of {limit} bytes")]
    TooLarge { actual: u64, limit: u64 },
    #[error("section table range is invalid")]
    SectionTableRange,
    #[error("section {index} is truncated")]
    SectionTruncated { index: u32 },
    #[error("section {index} range is invalid")]
    SectionRange { index: u32 },
    #[error("section {0} has invalid alignment {1}")]
    Alignment(u32, u64),
    #[error("section {index} is duplicated")]
    DuplicateSection { index: u32 },
    #[error("section {index} overlaps another section")]
    SectionOverlap { index: u32 },
    #[error("section {index} overlaps the section table")]
    SectionTableOverlap { index: u32 },
    #[error("operation section {index} is not aligned to {bytes} bytes")]
    OperationAlignment { index: u32, bytes: usize },
    #[error("tensor metadata section {index} is not aligned to {bytes} bytes")]
    TensorAlignment { index: u32, bytes: usize },
    #[error("tensor record {index} is invalid: {reason}")]
    Tensor { index: usize, reason: &'static str },
    #[error("tensor metadata and tile data sections must appear together")]
    TensorSections,
    #[error("section {0} has unsupported kind {1}")]
    UnknownSection(u32, u32),
    #[error("integer conversion overflow")]
    Overflow,
    #[error("could not read artifact: {0}")]
    Read(#[from] std::io::Error),
    #[error("could not compute artifact identity: {0}")]
    Identity(String),
}

pub struct Artifact<'a> {
    bytes: &'a [u8],
    pub header: ArtifactHeader,
}

pub struct MappedArtifact {
    _file: File,
    mapping: memmap2::Mmap,
}

impl MappedArtifact {
    pub fn open(path: &Path, max_bytes: u64) -> Result<Self, ArtifactError> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if metadata.len() > max_bytes {
            return Err(ArtifactError::TooLarge {
                actual: metadata.len(),
                limit: max_bytes,
            });
        }
        // SAFETY: the file is opened read-only and the mapping is retained
        // together with its File handle. Callers must not mutate or truncate a
        // mapped artifact; artifact publication uses atomic replacement.
        let mapping = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(ArtifactError::Read)?
        };
        Artifact::parse(&mapping)?;
        Ok(Self {
            _file: file,
            mapping,
        })
    }

    pub fn view(&self) -> Result<Artifact<'_>, ArtifactError> {
        Artifact::parse(&self.mapping)
    }
}

impl<'a> Artifact<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, ArtifactError> {
        if bytes.len() < HEADER_BYTES {
            return Err(ArtifactError::TooSmall {
                actual: bytes.len(),
                required: HEADER_BYTES,
            });
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(ArtifactError::Magic);
        }
        let version = read_u16(bytes, 8)?;
        if version != FORMAT_VERSION {
            return Err(ArtifactError::Version(version));
        }
        let flags = read_u16(bytes, 10)?;
        let section_count = read_u32(bytes, 12)?;
        if section_count > MAX_SECTIONS {
            return Err(ArtifactError::TooManySections(section_count));
        }
        let section_table_offset = read_u64(bytes, 16)?;
        let artifact_bytes = read_u64(bytes, 24)?;
        if artifact_bytes != bytes.len() as u64 {
            return Err(ArtifactError::LengthMismatch {
                declared: artifact_bytes,
                actual: bytes.len(),
            });
        }
        let table_bytes = u64::from(section_count)
            .checked_mul(SECTION_BYTES as u64)
            .ok_or(ArtifactError::Overflow)?;
        let table_end = section_table_offset
            .checked_add(table_bytes)
            .ok_or(ArtifactError::Overflow)?;
        if section_table_offset < HEADER_BYTES as u64 || table_end > artifact_bytes {
            return Err(ArtifactError::SectionTableRange);
        }
        let artifact = Self {
            bytes,
            header: ArtifactHeader {
                version,
                flags,
                section_count,
                section_table_offset,
                artifact_bytes,
            },
        };
        artifact.validate_records()?;
        Ok(artifact)
    }

    fn validate_records(&self) -> Result<(), ArtifactError> {
        let table_start = self.header.section_table_offset;
        let table_end = table_start
            .checked_add(
                u64::from(self.header.section_count)
                    .checked_mul(SECTION_BYTES as u64)
                    .ok_or(ArtifactError::Overflow)?,
            )
            .ok_or(ArtifactError::Overflow)?;
        let mut previous_end = 0_u64;
        let mut operation_seen = false;
        let mut provenance_seen = false;
        let mut tensor_range = None;
        let mut tile_range = None;
        for index in 0..self.header.section_count {
            let offset = table_start
                .checked_add(
                    u64::from(index)
                        .checked_mul(SECTION_BYTES as u64)
                        .ok_or(ArtifactError::Overflow)?,
                )
                .ok_or(ArtifactError::Overflow)? as usize;
            let record = self
                .bytes
                .get(offset..offset + SECTION_BYTES)
                .ok_or(ArtifactError::SectionTruncated { index })?;
            let kind = read_u32(record, 0)?;
            match kind {
                crate::ops::OPERATION_SECTION => {
                    if operation_seen {
                        return Err(ArtifactError::DuplicateSection { index });
                    }
                    operation_seen = true;
                }
                PROVENANCE_SECTION => {
                    if provenance_seen {
                        return Err(ArtifactError::DuplicateSection { index });
                    }
                    provenance_seen = true;
                }
                TENSOR_SECTION => {
                    if tensor_range.is_some() {
                        return Err(ArtifactError::DuplicateSection { index });
                    }
                }
                TILE_DATA_SECTION => {
                    if tile_range.is_some() {
                        return Err(ArtifactError::DuplicateSection { index });
                    }
                }
                _ => return Err(ArtifactError::UnknownSection(index, kind)),
            }
            let data_offset = read_u64(record, 8)?;
            let length = read_u64(record, 16)?;
            let alignment = read_u64(record, 24)?;
            let data_end = data_offset
                .checked_add(length)
                .ok_or(ArtifactError::Overflow)?;
            if alignment == 0 || !alignment.is_power_of_two() || data_offset % alignment != 0 {
                return Err(ArtifactError::Alignment(index, alignment));
            }
            if data_end > self.header.artifact_bytes || data_offset < HEADER_BYTES as u64 {
                return Err(ArtifactError::SectionRange { index });
            }
            if data_offset < table_end && data_end > table_start {
                return Err(ArtifactError::SectionTableOverlap { index });
            }
            if index > 0 && data_offset < previous_end {
                return Err(ArtifactError::SectionOverlap { index });
            }
            if kind == crate::ops::OPERATION_SECTION
                && length % crate::ops::OPERATION_BYTES as u64 != 0
            {
                return Err(ArtifactError::OperationAlignment {
                    index,
                    bytes: crate::ops::OPERATION_BYTES,
                });
            }
            if kind == TENSOR_SECTION && length % TENSOR_RECORD_BYTES as u64 != 0 {
                return Err(ArtifactError::TensorAlignment {
                    index,
                    bytes: TENSOR_RECORD_BYTES,
                });
            }
            if kind == TENSOR_SECTION {
                tensor_range = Some((data_offset, data_end));
            } else if kind == TILE_DATA_SECTION {
                tile_range = Some((data_offset, data_end));
            }
            previous_end = data_end;
        }
        match (tensor_range, tile_range) {
            (None, None) => {}
            (Some((tensor_start, tensor_end)), Some((tile_start, tile_end))) => {
                let tensors = self.range(tensor_start, tensor_end)?;
                let tiles = self.range(tile_start, tile_end)?;
                validate_tensor_records(tensors, tiles)?;
            }
            _ => return Err(ArtifactError::TensorSections),
        }
        Ok(())
    }

    pub fn section(&self, wanted_kind: u32) -> Result<Option<Section<'a>>, ArtifactError> {
        let mut found = None;
        let table_start = self.header.section_table_offset;
        let table_end = table_start
            .checked_add(
                u64::from(self.header.section_count)
                    .checked_mul(SECTION_BYTES as u64)
                    .ok_or(ArtifactError::Overflow)?,
            )
            .ok_or(ArtifactError::Overflow)?;
        let mut previous_end = 0_u64;
        for index in 0..self.header.section_count {
            let offset = self
                .header
                .section_table_offset
                .checked_add(
                    u64::from(index)
                        .checked_mul(SECTION_BYTES as u64)
                        .ok_or(ArtifactError::Overflow)?,
                )
                .ok_or(ArtifactError::Overflow)? as usize;
            let record = self
                .bytes
                .get(offset..offset + SECTION_BYTES)
                .ok_or(ArtifactError::SectionTruncated { index })?;
            let kind = read_u32(record, 0)?;
            let flags = read_u32(record, 4)?;
            let data_offset = read_u64(record, 8)?;
            let length = read_u64(record, 16)?;
            let alignment = read_u64(record, 24)?;
            if !matches!(
                kind,
                crate::ops::OPERATION_SECTION
                    | PROVENANCE_SECTION
                    | TENSOR_SECTION
                    | TILE_DATA_SECTION
            ) {
                return Err(ArtifactError::UnknownSection(index, kind));
            }
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(ArtifactError::Alignment(index, alignment));
            }
            if data_offset % alignment != 0 {
                return Err(ArtifactError::Alignment(index, alignment));
            }
            let data_end = data_offset
                .checked_add(length)
                .ok_or(ArtifactError::Overflow)?;
            if data_end > self.header.artifact_bytes || data_offset < HEADER_BYTES as u64 {
                return Err(ArtifactError::SectionRange { index });
            }
            if data_offset < table_end && data_end > table_start {
                return Err(ArtifactError::SectionTableOverlap { index });
            }
            if index > 0 && data_offset < previous_end {
                return Err(ArtifactError::SectionOverlap { index });
            }
            previous_end = data_end;
            if kind == wanted_kind {
                if found.is_some() {
                    return Err(ArtifactError::DuplicateSection { index });
                }
                let start = usize::try_from(data_offset).map_err(|_| ArtifactError::Overflow)?;
                let end = usize::try_from(data_end).map_err(|_| ArtifactError::Overflow)?;
                found = Some(Section {
                    kind,
                    flags,
                    offset: data_offset,
                    length,
                    alignment,
                    bytes: &self.bytes[start..end],
                });
            }
        }
        Ok(found)
    }

    pub fn tensors(&self) -> Result<Option<TensorTable<'a>>, ArtifactError> {
        let Some(records) = self.section(TENSOR_SECTION)? else {
            return Ok(None);
        };
        let tiles = self
            .section(TILE_DATA_SECTION)?
            .ok_or(ArtifactError::TensorSections)?;
        Ok(Some(TensorTable {
            records: records.bytes,
            tiles: tiles.bytes,
        }))
    }

    fn range(&self, start: u64, end: u64) -> Result<&'a [u8], ArtifactError> {
        let start = usize::try_from(start).map_err(|_| ArtifactError::Overflow)?;
        let end = usize::try_from(end).map_err(|_| ArtifactError::Overflow)?;
        self.bytes
            .get(start..end)
            .ok_or(ArtifactError::SectionRange { index: 0 })
    }

    /// Returns a deterministic identity for every byte in the validated artifact.
    /// The identity is provenance, not authorization.
    pub fn identity(&self) -> Result<String, ArtifactError> {
        use sha2::Digest;
        let mut digest = sha2::Sha256::new();
        digest.update(self.bytes);
        let digest = digest.finalize();
        let mut output = String::from("sha256:");
        for byte in digest {
            output.push_str(&format!("{byte:02x}"));
        }
        Ok(output)
    }

    /// Returns the canonical UOR address of the embedded provenance JSON.
    pub fn provenance_identity(&self) -> Result<Option<String>, ArtifactError> {
        let Some(section) = self.section(PROVENANCE_SECTION)? else {
            return Ok(None);
        };
        let outcome = uor_addr::json::address(section.bytes)
            .map_err(|error| ArtifactError::Identity(format!("{error:?}")))?;
        Ok(Some(outcome.address.to_string()))
    }
}

fn validate_tensor_records(records: &[u8], tiles: &[u8]) -> Result<(), ArtifactError> {
    if records.is_empty() {
        return Err(ArtifactError::Tensor {
            index: 0,
            reason: "metadata section is empty",
        });
    }
    let mut previous_id = None;
    let mut previous_range = None;
    for index in 0..records.len() / TENSOR_RECORD_BYTES {
        let tensor = parse_tensor_record(records, tiles, index)?;
        if previous_id.is_some_and(|id| tensor.id <= id) {
            return Err(ArtifactError::Tensor {
                index,
                reason: "ids must be strictly increasing",
            });
        }
        if let Some((offset, length, end)) = previous_range {
            if tensor.tile_offset < end
                && (tensor.tile_offset != offset || tensor.tile_length != length)
            {
                return Err(ArtifactError::Tensor {
                    index,
                    reason: "tile ranges partially overlap or are out of order",
                });
            }
            if tensor.tile_offset > end {
                return Err(ArtifactError::Tensor {
                    index,
                    reason: "tile ranges must not leave gaps",
                });
            }
        } else if tensor.tile_offset != 0 {
            return Err(ArtifactError::Tensor {
                index,
                reason: "first tile must start at offset zero",
            });
        }
        previous_id = Some(tensor.id);
        previous_range = Some((
            tensor.tile_offset,
            tensor.tile_length,
            tensor
                .tile_offset
                .checked_add(tensor.tile_length)
                .ok_or(ArtifactError::Overflow)?,
        ));
    }
    if previous_range.is_none_or(|(_, _, end)| end != tiles.len() as u64) {
        return Err(ArtifactError::Tensor {
            index: records.len() / TENSOR_RECORD_BYTES,
            reason: "tile data contains unreferenced bytes",
        });
    }
    Ok(())
}

fn parse_tensor_record<'a>(
    records: &[u8],
    tiles: &'a [u8],
    index: usize,
) -> Result<TensorRecord<'a>, ArtifactError> {
    let start = index
        .checked_mul(TENSOR_RECORD_BYTES)
        .ok_or(ArtifactError::Overflow)?;
    let record = records
        .get(start..start + TENSOR_RECORD_BYTES)
        .ok_or(ArtifactError::Tensor {
            index,
            reason: "record is truncated",
        })?;
    let id = read_u32(record, 0)?;
    let element_type = read_u16(record, 4)?;
    let codec = read_u16(record, 6)?;
    let rank = read_u16(record, 8)?;
    let reserved = read_u16(record, 10)?;
    let flags = read_u32(record, 12)?;
    if element_type != ELEMENT_F32 {
        return Err(ArtifactError::Tensor {
            index,
            reason: "unsupported element type",
        });
    }
    if codec != CODEC_RAW {
        return Err(ArtifactError::Tensor {
            index,
            reason: "unsupported codec",
        });
    }
    if !(1..=4).contains(&rank) {
        return Err(ArtifactError::Tensor {
            index,
            reason: "rank must be between 1 and 4",
        });
    }
    if reserved != 0 || flags != 0 {
        return Err(ArtifactError::Tensor {
            index,
            reason: "reserved fields must be zero",
        });
    }
    let shape = [
        read_u32(record, 16)?,
        read_u32(record, 20)?,
        read_u32(record, 24)?,
        read_u32(record, 28)?,
    ];
    let mut elements = 1_u64;
    for (dimension, size) in shape.iter().enumerate() {
        if dimension < usize::from(rank) {
            if *size == 0 {
                return Err(ArtifactError::Tensor {
                    index,
                    reason: "active dimensions must be nonzero",
                });
            }
            elements = elements
                .checked_mul(u64::from(*size))
                .ok_or(ArtifactError::Overflow)?;
        } else if *size != 0 {
            return Err(ArtifactError::Tensor {
                index,
                reason: "unused dimensions must be zero",
            });
        }
    }
    let tile_offset = read_u64(record, 32)?;
    let tile_length = read_u64(record, 40)?;
    if tile_offset % std::mem::size_of::<f32>() as u64 != 0 {
        return Err(ArtifactError::Tensor {
            index,
            reason: "raw f32 tile offset must be four-byte aligned",
        });
    }
    let expected_length = elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or(ArtifactError::Overflow)?;
    if tile_length != expected_length {
        return Err(ArtifactError::Tensor {
            index,
            reason: "raw f32 byte length does not match shape",
        });
    }
    let tile_end = tile_offset
        .checked_add(tile_length)
        .ok_or(ArtifactError::Overflow)?;
    let tile_start = usize::try_from(tile_offset).map_err(|_| ArtifactError::Overflow)?;
    let tile_end = usize::try_from(tile_end).map_err(|_| ArtifactError::Overflow)?;
    let bytes = tiles
        .get(tile_start..tile_end)
        .ok_or(ArtifactError::Tensor {
            index,
            reason: "tile range exceeds tile data section",
        })?;
    Ok(TensorRecord {
        id,
        element_type,
        codec,
        rank,
        shape,
        tile_offset,
        tile_length,
        bytes,
    })
}

pub fn validate(path: &Path) -> Result<(), ArtifactError> {
    let _artifact = MappedArtifact::open(path, u64::MAX)?;
    println!("artifact: valid ({})", path.display());
    Ok(())
}

pub fn inspect(path: &Path) -> Result<(), ArtifactError> {
    let mapped = MappedArtifact::open(path, u64::MAX)?;
    let artifact = mapped.view()?;
    let (tensor_count, tensor_bytes) = match artifact.tensors()? {
        Some(tensors) => (tensors.len(), tensors.logical_bytes()?),
        None => (0, 0),
    };
    println!(
        "artifact: {}\nidentity: {}\nprovenance: {}\nversion: {}\nsections: {}\ntensors: {}\ntensor bytes: {}\nbytes: {}",
        path.display(),
        artifact.identity()?,
        artifact
            .provenance_identity()?
            .unwrap_or_else(|| "none".to_string()),
        artifact.header.version,
        artifact.header.section_count,
        tensor_count,
        tensor_bytes,
        artifact.header.artifact_bytes
    );
    Ok(())
}

/// Builds an artifact from operation records and a provenance payload, and
/// publishes it atomically (temporary file plus rename). The artifact is
/// validated before publication; partial writes are never visible.
pub fn create_artifact(
    path: &Path,
    operations: &[u8],
    provenance: &[u8],
) -> Result<(), ArtifactError> {
    create_artifact_with_raw_tensors(path, operations, provenance, &[])
}

/// Builds an artifact with sorted raw `f32` tensors. Tensor bytes remain in
/// their own immutable section and are borrowed directly after admission.
pub fn create_artifact_with_raw_tensors(
    path: &Path,
    operations: &[u8],
    provenance: &[u8],
    tensors: &[RawTensor<'_>],
) -> Result<(), ArtifactError> {
    if tensors.is_empty() {
        return publish_artifact(path, operations, provenance, &[], &[]);
    }
    let mut records = Vec::with_capacity(tensors.len() * TENSOR_RECORD_BYTES);
    let tile_bytes = tensors.iter().try_fold(0_usize, |total, tensor| {
        total
            .checked_add(tensor.bytes.len())
            .ok_or(ArtifactError::Overflow)
    })?;
    let mut tiles = Vec::with_capacity(tile_bytes);
    for (index, tensor) in tensors.iter().enumerate() {
        if tensor.shape.is_empty() || tensor.shape.len() > 4 {
            return Err(ArtifactError::Tensor {
                index,
                reason: "rank must be between 1 and 4",
            });
        }
        let tile_offset = u64::try_from(tiles.len()).map_err(|_| ArtifactError::Overflow)?;
        let mut record = [0_u8; TENSOR_RECORD_BYTES];
        record[0..4].copy_from_slice(&tensor.id.to_le_bytes());
        record[4..6].copy_from_slice(&ELEMENT_F32.to_le_bytes());
        record[6..8].copy_from_slice(&CODEC_RAW.to_le_bytes());
        record[8..10].copy_from_slice(&(tensor.shape.len() as u16).to_le_bytes());
        for (dimension, size) in tensor.shape.iter().enumerate() {
            let start = 16 + dimension * 4;
            record[start..start + 4].copy_from_slice(&size.to_le_bytes());
        }
        record[32..40].copy_from_slice(&tile_offset.to_le_bytes());
        record[40..48].copy_from_slice(&(tensor.bytes.len() as u64).to_le_bytes());
        records.extend_from_slice(&record);
        tiles.extend_from_slice(tensor.bytes);
    }
    publish_artifact(path, operations, provenance, &records, &tiles)
}

fn publish_artifact(
    path: &Path,
    operations: &[u8],
    provenance: &[u8],
    tensors: &[u8],
    tiles: &[u8],
) -> Result<(), ArtifactError> {
    let bytes = build_artifact(operations, provenance, tensors, tiles)?;
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn build_artifact(
    operations: &[u8],
    provenance: &[u8],
    tensors: &[u8],
    tiles: &[u8],
) -> Result<Vec<u8>, ArtifactError> {
    if tensors.is_empty() != tiles.is_empty() {
        return Err(ArtifactError::TensorSections);
    }
    let has_tensors = !tensors.is_empty();
    let section_count = if has_tensors { 4_usize } else { 2 };
    let section_table_offset = HEADER_BYTES;
    let operation_offset = HEADER_BYTES + SECTION_BYTES * section_count;
    let provenance_offset = operation_offset
        .checked_add(operations.len())
        .ok_or(ArtifactError::Overflow)?;
    let provenance_end = provenance_offset
        .checked_add(provenance.len())
        .ok_or(ArtifactError::Overflow)?;
    let tensor_offset = align_up(provenance_end, 8)?;
    let tensor_end = tensor_offset
        .checked_add(tensors.len())
        .ok_or(ArtifactError::Overflow)?;
    let tile_offset = align_up(tensor_end, 4)?;
    let artifact_bytes = if has_tensors {
        tile_offset
            .checked_add(tiles.len())
            .ok_or(ArtifactError::Overflow)?
    } else {
        provenance_end
    };
    let mut bytes = vec![0_u8; artifact_bytes];
    bytes[..8].copy_from_slice(&MAGIC);
    bytes[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes[12..16].copy_from_slice(&(section_count as u32).to_le_bytes());
    bytes[16..24].copy_from_slice(&(section_table_offset as u64).to_le_bytes());
    bytes[24..32].copy_from_slice(&(artifact_bytes as u64).to_le_bytes());
    let table = section_table_offset;
    write_section(
        &mut bytes,
        table,
        0,
        crate::ops::OPERATION_SECTION,
        operation_offset,
        operations.len(),
        4,
    );
    write_section(
        &mut bytes,
        table,
        1,
        PROVENANCE_SECTION,
        provenance_offset,
        provenance.len(),
        4,
    );
    if has_tensors {
        write_section(
            &mut bytes,
            table,
            2,
            TENSOR_SECTION,
            tensor_offset,
            tensors.len(),
            8,
        );
        write_section(
            &mut bytes,
            table,
            3,
            TILE_DATA_SECTION,
            tile_offset,
            tiles.len(),
            4,
        );
    }
    bytes[operation_offset..operation_offset + operations.len()].copy_from_slice(operations);
    bytes[provenance_offset..provenance_end].copy_from_slice(provenance);
    if has_tensors {
        bytes[tensor_offset..tensor_end].copy_from_slice(tensors);
        bytes[tile_offset..].copy_from_slice(tiles);
    }
    Artifact::parse(&bytes)?;
    Ok(bytes)
}

fn write_section(
    bytes: &mut [u8],
    table: usize,
    index: usize,
    kind: u32,
    offset: usize,
    length: usize,
    alignment: u64,
) {
    let record = table + index * SECTION_BYTES;
    bytes[record..record + 4].copy_from_slice(&kind.to_le_bytes());
    bytes[record + 8..record + 16].copy_from_slice(&(offset as u64).to_le_bytes());
    bytes[record + 16..record + 24].copy_from_slice(&(length as u64).to_le_bytes());
    bytes[record + 24..record + 32].copy_from_slice(&alignment.to_le_bytes());
}

fn align_up(value: usize, alignment: usize) -> Result<usize, ArtifactError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(ArtifactError::Overflow)
}

pub fn create_fixture(path: &Path) -> Result<(), ArtifactError> {
    let mut operations = [0_u8; crate::ops::OPERATION_BYTES];
    operations[4..8].copy_from_slice(&1.0_f32.to_le_bytes());
    operations[8..12].copy_from_slice(&4_u32.to_le_bytes());
    operations[12..16].copy_from_slice(&4_u32.to_le_bytes());
    create_artifact(
        path,
        &operations,
        br#"{"compiler":"native-machine","layout":"bootstrap","profile":"scalar"}"#,
    )?;
    println!("created fixture {}", path.display());
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ArtifactError> {
    let end = offset.checked_add(2).ok_or(ArtifactError::Overflow)?;
    let value = bytes.get(offset..end).ok_or(ArtifactError::TooSmall {
        actual: bytes.len(),
        required: end,
    })?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ArtifactError> {
    let end = offset.checked_add(4).ok_or(ArtifactError::Overflow)?;
    let value = bytes.get(offset..end).ok_or(ArtifactError::TooSmall {
        actual: bytes.len(),
        required: end,
    })?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, ArtifactError> {
    let end = offset.checked_add(8).ok_or(ArtifactError::Overflow)?;
    let value = bytes.get(offset..end).ok_or(ArtifactError::TooSmall {
        actual: bytes.len(),
        required: end,
    })?;
    Ok(u64::from_le_bytes(
        value.try_into().map_err(|_| ArtifactError::Overflow)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let mut bytes = vec![0_u8; HEADER_BYTES + SECTION_BYTES + crate::ops::OPERATION_BYTES];
        let artifact_bytes = bytes.len() as u64;
        bytes[..8].copy_from_slice(&MAGIC);
        bytes[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&1_u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&(HEADER_BYTES as u64).to_le_bytes());
        bytes[24..32].copy_from_slice(&artifact_bytes.to_le_bytes());
        let table = HEADER_BYTES;
        bytes[table..table + 4].copy_from_slice(&crate::ops::OPERATION_SECTION.to_le_bytes());
        bytes[table + 8..table + 16]
            .copy_from_slice(&((HEADER_BYTES + SECTION_BYTES) as u64).to_le_bytes());
        bytes[table + 16..table + 24]
            .copy_from_slice(&(crate::ops::OPERATION_BYTES as u64).to_le_bytes());
        bytes[table + 24..table + 32].copy_from_slice(&4_u64.to_le_bytes());
        bytes
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
    fn parser_survives_random_mutations() {
        let mut rng = TestRng(0x62BF_6A22_5813_AA09);
        for _ in 0..1_000 {
            let mut bytes = fixture();
            for _ in 0..(rng.next() % 4 + 1) {
                let position = (rng.next() as usize) % bytes.len();
                bytes[position] = (rng.next() & 0xFF) as u8;
            }
            let first = Artifact::parse(&bytes);
            let second = Artifact::parse(&bytes);
            assert_eq!(first.is_err(), second.is_err());
            if let (Ok(first), Ok(second)) = (first, second) {
                assert_eq!(first.header, second.header);
            }
        }
    }

    #[test]
    fn parses_borrowed_section() {
        let bytes = fixture();
        let artifact = Artifact::parse(&bytes).expect("fixture is valid");
        let section = artifact
            .section(crate::ops::OPERATION_SECTION)
            .expect("section lookup succeeds")
            .expect("section exists");
        assert_eq!(section.bytes.len(), crate::ops::OPERATION_BYTES);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = fixture();
        bytes[0] = b'X';
        assert!(matches!(Artifact::parse(&bytes), Err(ArtifactError::Magic)));
    }

    #[test]
    fn rejects_declared_length_mismatch() {
        let mut bytes = fixture();
        bytes[24..32].copy_from_slice(&0_u64.to_le_bytes());
        assert!(matches!(
            Artifact::parse(&bytes),
            Err(ArtifactError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_section_table_overlap() {
        let mut bytes = fixture();
        let table = HEADER_BYTES;
        bytes[table + 8..table + 16].copy_from_slice(&(HEADER_BYTES as u64).to_le_bytes());
        assert!(matches!(
            Artifact::parse(&bytes),
            Err(ArtifactError::SectionTableOverlap { .. })
        ));
    }

    #[test]
    fn rejects_partial_operation_record() {
        let mut bytes = fixture();
        let table = HEADER_BYTES;
        bytes[table + 16..table + 24].copy_from_slice(&15_u64.to_le_bytes());
        assert!(matches!(
            Artifact::parse(&bytes),
            Err(ArtifactError::OperationAlignment { .. })
        ));
    }

    #[test]
    fn accepts_operation_and_provenance_sections() {
        let operation_offset = HEADER_BYTES + (SECTION_BYTES * 2);
        let provenance = br#"{"compiler":"test"}"#;
        let provenance_offset = operation_offset + crate::ops::OPERATION_BYTES;
        let mut bytes = vec![0_u8; provenance_offset + provenance.len()];
        let artifact_bytes = bytes.len() as u64;
        bytes[..8].copy_from_slice(&MAGIC);
        bytes[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&2_u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&(HEADER_BYTES as u64).to_le_bytes());
        bytes[24..32].copy_from_slice(&artifact_bytes.to_le_bytes());
        let operation = HEADER_BYTES;
        bytes[operation..operation + 4]
            .copy_from_slice(&crate::ops::OPERATION_SECTION.to_le_bytes());
        bytes[operation + 8..operation + 16]
            .copy_from_slice(&(operation_offset as u64).to_le_bytes());
        bytes[operation + 16..operation + 24]
            .copy_from_slice(&(crate::ops::OPERATION_BYTES as u64).to_le_bytes());
        bytes[operation + 24..operation + 32].copy_from_slice(&4_u64.to_le_bytes());
        let provenance_record = HEADER_BYTES + SECTION_BYTES;
        bytes[provenance_record..provenance_record + 4]
            .copy_from_slice(&PROVENANCE_SECTION.to_le_bytes());
        bytes[provenance_record + 8..provenance_record + 16]
            .copy_from_slice(&(provenance_offset as u64).to_le_bytes());
        bytes[provenance_record + 16..provenance_record + 24]
            .copy_from_slice(&(provenance.len() as u64).to_le_bytes());
        bytes[provenance_record + 24..provenance_record + 32].copy_from_slice(&4_u64.to_le_bytes());
        bytes[provenance_offset..].copy_from_slice(provenance);
        let artifact = Artifact::parse(&bytes).expect("multi-section artifact is valid");
        assert!(artifact
            .provenance_identity()
            .expect("provenance parses")
            .is_some());
    }

    #[test]
    fn publishes_and_borrows_raw_tensor_tiles() {
        let path = std::env::temp_dir().join(format!(
            "native-machine-tensor-fixture-{}.nm",
            std::process::id()
        ));
        let shape = [2, 2];
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        create_artifact_with_raw_tensors(
            &path,
            &[0_u8; crate::ops::OPERATION_BYTES],
            br#"{"compiler":"test"}"#,
            &[RawTensor {
                id: 7,
                shape: &shape,
                bytes: &bytes,
            }],
        )
        .expect("tensor artifact publishes");
        let mapped = MappedArtifact::open(&path, u64::MAX).expect("artifact maps");
        let artifact = mapped.view().expect("artifact parses");
        let tensors = artifact
            .tensors()
            .expect("tensor table parses")
            .expect("tensor table exists");
        let tensor = tensors
            .get(0)
            .expect("lookup succeeds")
            .expect("tensor exists");
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensor.shape, [2, 2, 0, 0]);
        assert_eq!(tensor.bytes, bytes);
        drop(mapped);
        fs::remove_file(path).expect("fixture is removed");
    }

    #[test]
    fn rejects_unsorted_tensor_ids() {
        let shape = [1];
        let bytes = 1.0_f32.to_le_bytes();
        let path = std::env::temp_dir().join(format!(
            "native-machine-unsorted-tensors-{}.nm",
            std::process::id()
        ));
        let error = create_artifact_with_raw_tensors(
            &path,
            &[0_u8; crate::ops::OPERATION_BYTES],
            br#"{"compiler":"test"}"#,
            &[
                RawTensor {
                    id: 2,
                    shape: &shape,
                    bytes: &bytes,
                },
                RawTensor {
                    id: 1,
                    shape: &shape,
                    bytes: &bytes,
                },
            ],
        )
        .expect_err("unsorted ids are rejected");
        assert!(matches!(error, ArtifactError::Tensor { index: 1, .. }));
        assert!(!path.exists());
    }

    #[test]
    fn rejects_raw_tensor_length_that_does_not_match_shape() {
        let shape = [2];
        let bytes = 1.0_f32.to_le_bytes();
        let path = std::env::temp_dir().join(format!(
            "native-machine-short-tensor-{}.nm",
            std::process::id()
        ));
        let error = create_artifact_with_raw_tensors(
            &path,
            &[0_u8; crate::ops::OPERATION_BYTES],
            br#"{"compiler":"test"}"#,
            &[RawTensor {
                id: 1,
                shape: &shape,
                bytes: &bytes,
            }],
        )
        .expect_err("short tensor is rejected");
        assert!(matches!(error, ArtifactError::Tensor { index: 0, .. }));
        assert!(!path.exists());
    }

    #[test]
    fn rejects_unreferenced_tile_bytes() {
        let mut record = [0_u8; TENSOR_RECORD_BYTES];
        record[0..4].copy_from_slice(&1_u32.to_le_bytes());
        record[4..6].copy_from_slice(&ELEMENT_F32.to_le_bytes());
        record[6..8].copy_from_slice(&CODEC_RAW.to_le_bytes());
        record[8..10].copy_from_slice(&1_u16.to_le_bytes());
        record[16..20].copy_from_slice(&1_u32.to_le_bytes());
        record[40..48].copy_from_slice(&4_u64.to_le_bytes());
        let error = build_artifact(
            &[0_u8; crate::ops::OPERATION_BYTES],
            br#"{"compiler":"test"}"#,
            &record,
            &[0_u8; 8],
        )
        .expect_err("unreferenced tile bytes are rejected");
        assert!(matches!(error, ArtifactError::Tensor { index: 1, .. }));
    }
}
