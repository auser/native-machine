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
            previous_end = data_end;
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
            if !matches!(kind, crate::ops::OPERATION_SECTION | PROVENANCE_SECTION) {
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

pub fn validate(path: &Path) -> Result<(), ArtifactError> {
    let _artifact = MappedArtifact::open(path, u64::MAX)?;
    println!("artifact: valid ({})", path.display());
    Ok(())
}

pub fn inspect(path: &Path) -> Result<(), ArtifactError> {
    let mapped = MappedArtifact::open(path, u64::MAX)?;
    let artifact = mapped.view()?;
    println!(
        "artifact: {}\nidentity: {}\nprovenance: {}\nversion: {}\nsections: {}\nbytes: {}",
        path.display(),
        artifact.identity()?,
        artifact
            .provenance_identity()?
            .unwrap_or_else(|| "none".to_string()),
        artifact.header.version,
        artifact.header.section_count,
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
    let section_table_offset = HEADER_BYTES;
    let operation_offset = HEADER_BYTES + (SECTION_BYTES * 2);
    let provenance_offset = operation_offset + operations.len();
    let artifact_bytes = provenance_offset + provenance.len();
    let mut bytes = vec![0_u8; artifact_bytes];
    bytes[..8].copy_from_slice(&MAGIC);
    bytes[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes[12..16].copy_from_slice(&2_u32.to_le_bytes());
    bytes[16..24].copy_from_slice(&(section_table_offset as u64).to_le_bytes());
    bytes[24..32].copy_from_slice(&(artifact_bytes as u64).to_le_bytes());
    let table = section_table_offset;
    bytes[table..table + 4].copy_from_slice(&crate::ops::OPERATION_SECTION.to_le_bytes());
    bytes[table + 8..table + 16].copy_from_slice(&(operation_offset as u64).to_le_bytes());
    bytes[table + 16..table + 24].copy_from_slice(&(operations.len() as u64).to_le_bytes());
    bytes[table + 24..table + 32].copy_from_slice(&4_u64.to_le_bytes());
    let provenance_record = table + SECTION_BYTES;
    bytes[provenance_record..provenance_record + 4]
        .copy_from_slice(&PROVENANCE_SECTION.to_le_bytes());
    bytes[provenance_record + 8..provenance_record + 16]
        .copy_from_slice(&(provenance_offset as u64).to_le_bytes());
    bytes[provenance_record + 16..provenance_record + 24]
        .copy_from_slice(&(provenance.len() as u64).to_le_bytes());
    bytes[provenance_record + 24..provenance_record + 32].copy_from_slice(&4_u64.to_le_bytes());
    bytes[operation_offset..operation_offset + operations.len()].copy_from_slice(operations);
    bytes[provenance_offset..].copy_from_slice(provenance);
    Artifact::parse(&bytes)?;
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
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
}
