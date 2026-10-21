//! UOR identities for the admitted model content graph.

use crate::artifact::{Artifact, ArtifactError, TensorRecord};
use crate::ops::{OPERATION_BYTES, OPERATION_SECTION};
use sha2::{Digest, Sha256};
use std::fmt::Write;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerIdentity {
    pub index: usize,
    pub address: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TileIdentity {
    pub offset: u64,
    pub length: u64,
    pub address: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorIdentity {
    pub id: u32,
    pub address: String,
    pub tile_address: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelIdentities {
    pub model: String,
    pub layers: Vec<LayerIdentity>,
    pub tensors: Vec<TensorIdentity>,
    pub tiles: Vec<TileIdentity>,
}

#[derive(Debug, Error)]
pub enum IdentityError {
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error("could not compute {kind} UOR identity: {reason}")]
    Address { kind: &'static str, reason: String },
}

pub fn derive(artifact: &Artifact<'_>) -> Result<ModelIdentities, IdentityError> {
    let operation_section = artifact
        .section(OPERATION_SECTION)?
        .ok_or(ArtifactError::Identity(
            "artifact has no operation section".to_string(),
        ))?;
    let (records, remainder) = operation_section.bytes.as_chunks::<OPERATION_BYTES>();
    debug_assert!(remainder.is_empty());
    let layers = records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let json = format!(
                "{{\"format\":\"native-machine-layer\",\"record_sha256\":\"{}\",\"version\":1}}",
                digest(record)
            );
            Ok(LayerIdentity {
                index,
                address: address("layer", &json)?,
            })
        })
        .collect::<Result<Vec<_>, IdentityError>>()?;

    let mut tensors = Vec::new();
    let mut tiles = Vec::new();
    if let Some(table) = artifact.tensors()? {
        for index in 0..table.len() {
            let tensor = table.get(index)?.ok_or(ArtifactError::Tensor {
                index,
                reason: "validated tensor disappeared",
            })?;
            let tile_address = tile_address(&tensor)?;
            if tiles
                .last()
                .is_none_or(|tile: &TileIdentity| tile.offset != tensor.tile_offset)
            {
                tiles.push(TileIdentity {
                    offset: tensor.tile_offset,
                    length: tensor.tile_length,
                    address: tile_address.clone(),
                });
            }
            tensors.push(TensorIdentity {
                id: tensor.id,
                address: tensor_address(&tensor, &tile_address)?,
                tile_address,
            });
        }
    }

    let mut json = String::from("{\"format\":\"native-machine-model\",\"layers\":[");
    for (index, layer) in layers.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        write!(
            json,
            "{{\"index\":{},\"address\":\"{}\"}}",
            layer.index, layer.address
        )
        .expect("writing to a String cannot fail");
    }
    json.push_str("],\"tensors\":[");
    for (index, tensor) in tensors.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        write!(
            json,
            "{{\"id\":{},\"address\":\"{}\"}}",
            tensor.id, tensor.address
        )
        .expect("writing to a String cannot fail");
    }
    json.push_str("],\"version\":1}");

    Ok(ModelIdentities {
        model: address("model", &json)?,
        layers,
        tensors,
        tiles,
    })
}

fn tile_address(tensor: &TensorRecord<'_>) -> Result<String, IdentityError> {
    let json = format!(
        "{{\"codec\":{},\"element_type\":{},\"encoded_sha256\":\"{}\",\"format\":\"native-machine-tile\",\"length\":{},\"version\":1}}",
        tensor.codec,
        tensor.element_type,
        digest(tensor.bytes),
        tensor.tile_length
    );
    address("tile", &json)
}

fn tensor_address(tensor: &TensorRecord<'_>, tile: &str) -> Result<String, IdentityError> {
    let mut json = format!(
        "{{\"codec\":{},\"element_type\":{},\"format\":\"native-machine-tensor\",\"rank\":{},\"shape\":[",
        tensor.codec, tensor.element_type, tensor.rank
    );
    for (index, dimension) in tensor.shape[..usize::from(tensor.rank)].iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        write!(json, "{dimension}").expect("writing to a String cannot fail");
    }
    write!(json, "],\"tile\":\"{tile}\",\"version\":1}}").expect("writing to a String cannot fail");
    address("tensor", &json)
}

fn digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn address(kind: &'static str, json: &str) -> Result<String, IdentityError> {
    uor_addr::json::address(json.as_bytes())
        .map(|outcome| outcome.address.to_string())
        .map_err(|error| IdentityError::Address {
            kind,
            reason: format!("{error:?}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{create_artifact_with_raw_tensors, RawTensor};
    use std::fs;

    fn artifact_bytes(name: &str, operation: [u8; OPERATION_BYTES], value: f32) -> Vec<u8> {
        artifact_bytes_with_provenance(name, operation, value, br#"{"compiler":"identity-test"}"#)
    }

    fn artifact_bytes_with_provenance(
        name: &str,
        operation: [u8; OPERATION_BYTES],
        value: f32,
        provenance: &[u8],
    ) -> Vec<u8> {
        let path = std::env::temp_dir().join(format!(
            "native-machine-identities-{}-{name}.nm",
            std::process::id()
        ));
        let tensor_bytes = value.to_le_bytes();
        create_artifact_with_raw_tensors(
            &path,
            &operation,
            provenance,
            &[RawTensor {
                id: 7,
                shape: &[1],
                bytes: &tensor_bytes,
            }],
        )
        .expect("identity fixture publishes");
        let bytes = fs::read(&path).expect("identity fixture reads");
        fs::remove_file(path).expect("identity fixture is removed");
        bytes
    }

    #[test]
    fn identities_are_stable_and_cover_every_content_level() {
        let bytes = artifact_bytes("stable", [0_u8; OPERATION_BYTES], 1.0);
        let artifact = Artifact::parse(&bytes).expect("artifact parses");
        let first = derive(&artifact).expect("identities derive");
        let second = derive(&artifact).expect("identities derive again");

        assert_eq!(first, second);
        assert_eq!(first.layers.len(), 1);
        assert_eq!(first.tensors.len(), 1);
        assert_eq!(first.tiles.len(), 1);
        assert!(first.model.starts_with("sha256:"));
        assert!(first.layers[0].address.starts_with("sha256:"));
        assert!(first.tensors[0].address.starts_with("sha256:"));
        assert!(first.tiles[0].address.starts_with("sha256:"));
    }

    #[test]
    fn tensor_change_propagates_to_tensor_and_model_only() {
        let first_bytes = artifact_bytes("tensor-first", [0_u8; OPERATION_BYTES], 1.0);
        let second_bytes = artifact_bytes("tensor-second", [0_u8; OPERATION_BYTES], 2.0);
        let first = derive(&Artifact::parse(&first_bytes).expect("first parses"))
            .expect("first identities derive");
        let second = derive(&Artifact::parse(&second_bytes).expect("second parses"))
            .expect("second identities derive");

        assert_eq!(first.layers, second.layers);
        assert_ne!(first.tiles, second.tiles);
        assert_ne!(first.tensors, second.tensors);
        assert_ne!(first.model, second.model);
    }

    #[test]
    fn layer_change_propagates_to_layer_and_model_only() {
        let mut changed_operation = [0_u8; OPERATION_BYTES];
        changed_operation[4..8].copy_from_slice(&1.0_f32.to_le_bytes());
        let first_bytes = artifact_bytes("layer-first", [0_u8; OPERATION_BYTES], 1.0);
        let second_bytes = artifact_bytes("layer-second", changed_operation, 1.0);
        let first = derive(&Artifact::parse(&first_bytes).expect("first parses"))
            .expect("first identities derive");
        let second = derive(&Artifact::parse(&second_bytes).expect("second parses"))
            .expect("second identities derive");

        assert_ne!(first.layers, second.layers);
        assert_eq!(first.tiles, second.tiles);
        assert_eq!(first.tensors, second.tensors);
        assert_ne!(first.model, second.model);
    }

    #[test]
    fn provenance_change_does_not_change_model_identity() {
        let first_bytes = artifact_bytes_with_provenance(
            "provenance-first",
            [0_u8; OPERATION_BYTES],
            1.0,
            br#"{"compiler":"first"}"#,
        );
        let second_bytes = artifact_bytes_with_provenance(
            "provenance-second",
            [0_u8; OPERATION_BYTES],
            1.0,
            br#"{"compiler":"second"}"#,
        );
        let first_artifact = Artifact::parse(&first_bytes).expect("first parses");
        let second_artifact = Artifact::parse(&second_bytes).expect("second parses");

        assert_eq!(
            derive(&first_artifact).expect("first identities derive"),
            derive(&second_artifact).expect("second identities derive")
        );
        assert_ne!(
            first_artifact.identity().expect("first artifact hashes"),
            second_artifact.identity().expect("second artifact hashes")
        );
    }
}
