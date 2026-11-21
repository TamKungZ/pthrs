//! Reader for the pre-ZIP `torch.save` layout: a magic number, protocol
//! version, system info, the object pickle, the sorted storage keys, and then
//! each storage as an element count followed by its raw data.

use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom},
};

use crate::{pickle::parse_pickle_prefix, ByteOrder, Error, Result, StorageRef, Value};

use super::{
    metadata::Checkpoint,
    storage::{Opened, Storage},
};

const MAGIC_PICKLE: [u8; 15] = [
    0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, b'.',
];
const PROTOCOL_VERSION: i64 = 1001;
const INITIAL_HEAD: u64 = 1024 * 1024;
const MAX_HEAD: u64 = 128 * 1024 * 1024;
const MAX_DEPTH: usize = 128;

/// Location of one storage's raw data in the file.
#[derive(Clone, Copy, Debug)]
pub(super) struct Region {
    offset: u64,
    pub(super) len: u64,
}

impl Region {
    pub(super) fn read_range<R: Read + Seek>(
        &self,
        reader: &mut R,
        offset: u64,
        output: &mut [u8],
    ) -> Result<()> {
        offset
            .checked_add(output.len() as u64)
            .filter(|end| *end <= self.len)
            .ok_or_else(|| Error::InvalidArchive("tensor range lies outside its storage".into()))?;
        reader.seek(SeekFrom::Start(self.offset + offset))?;
        reader.read_exact(output)?;
        Ok(())
    }
}

pub(super) fn is_legacy(head: &[u8]) -> bool {
    head.starts_with(&MAGIC_PICKLE)
}

struct Head {
    checkpoint: Checkpoint,
    byte_order: ByteOrder,
    keys: Vec<String>,
    data_start: u64,
}

pub(super) fn open<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<Opened> {
    let mut want = INITIAL_HEAD.min(file_len);
    let head = loop {
        let mut bytes = vec![0; want as usize];
        reader.seek(SeekFrom::Start(0))?;
        reader.read_exact(&mut bytes)?;
        match parse_head(&bytes) {
            Ok(head) => break head,
            Err(_) if want < file_len && want < MAX_HEAD => {
                want = (want * 2).min(file_len).min(MAX_HEAD);
            }
            Err(error) => return Err(error),
        }
    };

    let mut references = HashMap::new();
    collect_storages(head.checkpoint.root(), 0, &mut references);
    let mut regions = HashMap::new();
    let mut cursor = head.data_start;
    for key in &head.keys {
        let Some(storage) = references.get(key) else {
            break;
        };
        let len = storage.byte_len()?;
        let end = cursor
            .checked_add(8)
            .and_then(|start| start.checked_add(len))
            .filter(|end| *end <= file_len)
            .ok_or_else(|| Error::InvalidArchive(format!("storage {key} is truncated")))?;
        let mut size = [0u8; 8];
        reader.seek(SeekFrom::Start(cursor))?;
        reader.read_exact(&mut size)?;
        let elements = match head.byte_order {
            ByteOrder::Little => i64::from_le_bytes(size),
            ByteOrder::Big => i64::from_be_bytes(size),
        };
        if u64::try_from(elements).ok() != Some(storage.elements) {
            return Err(Error::InvalidArchive(format!(
                "storage {key} declares {elements} elements, expected {}",
                storage.elements
            )));
        }
        regions.insert(
            key.clone(),
            Region {
                offset: cursor + 8,
                len,
            },
        );
        cursor = end;
    }

    Ok(Opened {
        storage: Storage::Legacy(regions),
        checkpoint: head.checkpoint,
        byte_order: head.byte_order,
        serialization_version: Some(PROTOCOL_VERSION.to_string()),
    })
}

fn parse_head(bytes: &[u8]) -> Result<Head> {
    let mut at = MAGIC_PICKLE.len();
    let (_, version, used) = parse_pickle_prefix(&bytes[at..])?;
    at += used;
    if version.as_i64() != Some(PROTOCOL_VERSION) {
        return Err(Error::InvalidArchive(
            "unsupported legacy serialization protocol".into(),
        ));
    }
    let (_, system, used) = parse_pickle_prefix(&bytes[at..])?;
    at += used;
    let byte_order = match system.get("little_endian") {
        Some(Value::Bool(false)) => ByteOrder::Big,
        _ => ByteOrder::Little,
    };
    let (checkpoint, used) = Checkpoint::from_pickle_prefix(&bytes[at..])?;
    at += used;
    let (_, keys, used) = parse_pickle_prefix(&bytes[at..])?;
    at += used;
    let keys = keys
        .as_sequence()
        .ok_or_else(|| Error::InvalidArchive("storage key list is not a list".into()))?
        .iter()
        .map(|key| {
            key.as_str()
                .map(str::to_owned)
                .ok_or_else(|| Error::InvalidArchive("storage key is not a string".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Head {
        checkpoint,
        byte_order,
        keys,
        data_start: at as u64,
    })
}

fn collect_storages(value: &Value, depth: usize, found: &mut HashMap<String, StorageRef>) {
    if depth > MAX_DEPTH {
        return;
    }
    match value {
        Value::Tensor(tensor) => {
            found
                .entry(tensor.storage.key.clone())
                .or_insert_with(|| tensor.storage.clone());
        }
        Value::List(values) | Value::Tuple(values) => {
            for value in values {
                collect_storages(value, depth + 1, found);
            }
        }
        Value::Map(entries) => {
            for (_, value) in entries {
                collect_storages(value, depth + 1, found);
            }
        }
        Value::Reduce { args, .. } => collect_storages(args, depth + 1, found),
        _ => {}
    }
}
