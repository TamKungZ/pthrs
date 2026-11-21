use std::{
    collections::HashMap,
    io::{Read, Seek},
};

use crate::{zip, ByteOrder, Error, Result};

use super::{legacy, metadata::Checkpoint};

const MAX_DATA_PICKLE: u64 = 64 * 1024 * 1024;
const MAX_AUXILIARY_ENTRY: u64 = 1024 * 1024;

/// Where a checkpoint keeps its raw tensor storages.
#[derive(Debug)]
pub(super) enum Storage {
    Zip {
        archive: zip::Archive,
        prefix: String,
    },
    Legacy(HashMap<String, legacy::Region>),
}

pub(super) struct Opened {
    pub(super) storage: Storage,
    pub(super) checkpoint: Checkpoint,
    pub(super) byte_order: ByteOrder,
    pub(super) serialization_version: Option<String>,
}

impl Storage {
    /// Size in bytes of the storage, failing if it is missing or cannot be
    /// decoded.
    pub(super) fn size(&self, key: &str) -> Result<u64> {
        match self {
            Self::Zip { archive, prefix } => {
                let entry = archive
                    .find(&format!("{prefix}data/{key}"))
                    .ok_or_else(|| Error::InvalidArchive(format!("storage {key} is missing")))?;
                if !zip::Archive::is_supported(entry) {
                    return Err(Error::UnsupportedCompression {
                        method: entry.method,
                        entry: entry.name.clone(),
                    });
                }
                Ok(entry.uncompressed_size)
            }
            Self::Legacy(regions) => regions
                .get(key)
                .map(|region| region.len)
                .ok_or_else(|| Error::InvalidArchive(format!("storage {key} is missing"))),
        }
    }

    pub(super) fn read_range<R: Read + Seek>(
        &self,
        reader: &mut R,
        key: &str,
        offset: u64,
        output: &mut [u8],
    ) -> Result<()> {
        match self {
            Self::Zip { archive, prefix } => {
                let entry = archive
                    .find(&format!("{prefix}data/{key}"))
                    .ok_or_else(|| Error::InvalidArchive(format!("storage {key} is missing")))?;
                archive.read_range_into(reader, entry, offset, output)
            }
            Self::Legacy(regions) => {
                let region = regions
                    .get(key)
                    .ok_or_else(|| Error::InvalidArchive(format!("storage {key} is missing")))?;
                region.read_range(reader, offset, output)
            }
        }
    }

    pub(super) fn read_all<R: Read + Seek>(&self, reader: &mut R, key: &str) -> Result<Vec<u8>> {
        match self {
            Self::Zip { archive, prefix } => {
                let entry = archive
                    .find(&format!("{prefix}data/{key}"))
                    .ok_or_else(|| Error::InvalidArchive(format!("storage {key} is missing")))?;
                archive.read_all(reader, entry, usize::MAX as u64)
            }
            Self::Legacy(regions) => {
                let region = regions
                    .get(key)
                    .ok_or_else(|| Error::InvalidArchive(format!("storage {key} is missing")))?;
                let mut bytes = vec![0; region.len as usize];
                region.read_range(reader, 0, &mut bytes)?;
                Ok(bytes)
            }
        }
    }
}

pub(super) fn open_zip<R: Read + Seek>(reader: &mut R) -> Result<Opened> {
    let archive = zip::Archive::open(reader)?;
    let candidates: Vec<_> = archive
        .entries()
        .iter()
        .filter(|entry| entry.name == "data.pkl" || entry.name.ends_with("/data.pkl"))
        .collect();
    let data_entry = match candidates.as_slice() {
        [entry] => *entry,
        [] => return Err(Error::InvalidArchive("data.pkl is missing".into())),
        _ => {
            return Err(Error::InvalidArchive(
                "multiple data.pkl entries found".into(),
            ))
        }
    };
    let prefix = data_entry
        .name
        .strip_suffix("data.pkl")
        .expect("matched suffix")
        .to_owned();
    let pickle = archive.read_all(reader, data_entry, MAX_DATA_PICKLE)?;
    let checkpoint = Checkpoint::from_pickle(&pickle)?;

    let byte_order = match archive.find(&format!("{prefix}byteorder")) {
        Some(entry) => match trim_ascii(&archive.read_all(reader, entry, MAX_AUXILIARY_ENTRY)?) {
            b"little" => ByteOrder::Little,
            b"big" => ByteOrder::Big,
            other => {
                return Err(Error::InvalidArchive(format!(
                    "unknown byte order {:?}",
                    String::from_utf8_lossy(other)
                )))
            }
        },
        None => ByteOrder::Little,
    };
    let serialization_version = archive
        .find(&format!("{prefix}version"))
        .map(|entry| archive.read_all(reader, entry, MAX_AUXILIARY_ENTRY))
        .transpose()?
        .map(|bytes| String::from_utf8_lossy(trim_ascii(&bytes)).into_owned());

    Ok(Opened {
        storage: Storage::Zip { archive, prefix },
        checkpoint,
        byte_order,
        serialization_version,
    })
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}
