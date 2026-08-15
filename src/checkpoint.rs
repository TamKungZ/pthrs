use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek},
    path::Path,
    slice,
};

use crate::{
    pickle::parse_pickle_with_protocol, zip, ByteOrder, Error, Result, TensorData, TensorMeta,
    Value,
};

const MAX_DATA_PICKLE: u64 = 64 * 1024 * 1024;
const MAX_AUXILIARY_ENTRY: u64 = 1024 * 1024;

/// Parsed, inert metadata from a PyTorch checkpoint.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    protocol: u8,
    root: Value,
}

impl Checkpoint {
    pub fn pickle_protocol(&self) -> u8 {
        self.protocol
    }
    pub fn root(&self) -> &Value {
        &self.root
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.root.get(key)
    }

    /// Iterates top-level checkpoint metadata, excluding the potentially large
    /// `weight` dictionary.
    pub fn metadata(&self) -> MetadataIter<'_> {
        MetadataIter {
            inner: self.root.as_map().unwrap_or(&[]).iter(),
        }
    }

    /// Iterates tensors in the conventional `weight` state dictionary.
    pub fn tensors(&self) -> TensorIter<'_> {
        let entries = self
            .root
            .get("weight")
            .and_then(Value::as_map)
            .unwrap_or(&[]);
        TensorIter {
            inner: entries.iter(),
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors()
            .find_map(|(candidate, tensor)| (candidate == name).then_some(tensor))
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors().count()
    }
}

pub struct MetadataIter<'a> {
    inner: slice::Iter<'a, (Value, Value)>,
}
impl<'a> Iterator for MetadataIter<'a> {
    type Item = (&'a str, &'a Value);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.find_map(|(key, value)| {
            let key = key.as_str()?;
            (key != "weight").then_some((key, value))
        })
    }
}

pub struct TensorIter<'a> {
    inner: slice::Iter<'a, (Value, Value)>,
}
impl<'a> Iterator for TensorIter<'a> {
    type Item = (&'a str, &'a TensorMeta);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .find_map(|(key, value)| match (key.as_str(), value) {
                (Some(key), Value::Tensor(tensor)) => Some((key, tensor)),
                _ => None,
            })
    }
}

/// A lazy reader over a ZIP-based PyTorch `.pth` checkpoint.
///
/// Opening parses only ZIP metadata plus `data.pkl`; tensor storage is read on
/// demand by [`PthArchive::read_tensor`].
#[derive(Debug)]
pub struct PthArchive<R> {
    reader: R,
    zip: zip::Archive,
    prefix: String,
    byte_order: ByteOrder,
    serialization_version: Option<String>,
    checkpoint: Checkpoint,
}

impl PthArchive<File> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_reader(File::open(path)?)
    }
}

impl<R: Read + Seek> PthArchive<R> {
    pub fn from_reader(mut reader: R) -> Result<Self> {
        let zip = zip::Archive::open(&mut reader)?;
        let candidates: Vec<_> = zip
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
        let pickle = zip.read_all(&mut reader, data_entry, MAX_DATA_PICKLE)?;
        let (protocol, root) = parse_pickle_with_protocol(&pickle)?;
        let checkpoint = Checkpoint { protocol, root };

        let byte_order = match zip.find(&format!("{prefix}byteorder")) {
            Some(entry) => {
                match trim_ascii(&zip.read_all(&mut reader, entry, MAX_AUXILIARY_ENTRY)?) {
                    b"little" => ByteOrder::Little,
                    b"big" => ByteOrder::Big,
                    other => {
                        return Err(Error::InvalidArchive(format!(
                            "unknown byte order {:?}",
                            String::from_utf8_lossy(other)
                        )))
                    }
                }
            }
            None => ByteOrder::Little,
        };
        let serialization_version = zip
            .find(&format!("{prefix}version"))
            .map(|entry| zip.read_all(&mut reader, entry, MAX_AUXILIARY_ENTRY))
            .transpose()?
            .map(|bytes| String::from_utf8_lossy(trim_ascii(&bytes)).into_owned());

        let archive = Self {
            reader,
            zip,
            prefix,
            byte_order,
            serialization_version,
            checkpoint,
        };
        archive.validate_storages()?;
        Ok(archive)
    }

    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    pub fn byte_order(&self) -> ByteOrder {
        self.byte_order
    }
    pub fn serialization_version(&self) -> Option<&str> {
        self.serialization_version.as_deref()
    }

    pub fn read_tensor(&mut self, name: &str) -> Result<TensorData> {
        let meta = self
            .checkpoint
            .tensor(name)
            .cloned()
            .ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        let element_size = meta.dtype.element_size() as u64;
        let element_count = meta.element_count()?;
        let output_bytes = element_count
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidTensor("tensor byte length overflow".into()))?;
        if output_bytes > usize::MAX as u64 {
            return Err(Error::LimitExceeded {
                what: "tensor byte length",
                value: output_bytes,
                limit: usize::MAX as u64,
            });
        }
        if element_count == 0 {
            return Ok(TensorData {
                meta,
                bytes: Vec::new(),
                byte_order: self.byte_order,
            });
        }
        let entry_name = self.storage_entry_name(&meta.storage.key);
        let entry = self.zip.find(&entry_name).cloned().ok_or_else(|| {
            Error::InvalidArchive(format!("storage {} is missing", meta.storage.key))
        })?;
        let byte_offset = meta
            .storage_offset
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidTensor("storage byte offset overflow".into()))?;
        let bytes = if meta.is_contiguous() {
            self.zip
                .read_range(&mut self.reader, &entry, byte_offset, output_bytes)?
        } else {
            let span_elements = meta.storage_span_elements()?;
            let span_bytes = span_elements
                .checked_mul(element_size)
                .ok_or_else(|| Error::InvalidTensor("tensor storage span overflow".into()))?;
            let source = self
                .zip
                .read_range(&mut self.reader, &entry, byte_offset, span_bytes)?;
            gather_strided(&meta, &source)?
        };
        Ok(TensorData {
            meta,
            bytes,
            byte_order: self.byte_order,
        })
    }

    /// Read an entire underlying PyTorch storage. Most callers should prefer
    /// `read_tensor`, which respects offsets and strides.
    pub fn read_storage(&mut self, storage_key: &str) -> Result<Vec<u8>> {
        let name = self.storage_entry_name(storage_key);
        let entry =
            self.zip.find(&name).cloned().ok_or_else(|| {
                Error::InvalidArchive(format!("storage {storage_key} is missing"))
            })?;
        self.zip
            .read_all(&mut self.reader, &entry, usize::MAX as u64)
    }

    pub fn into_inner(self) -> R {
        self.reader
    }

    fn storage_entry_name(&self, key: &str) -> String {
        format!("{}data/{key}", self.prefix)
    }

    fn validate_storages(&self) -> Result<()> {
        let mut storages = BTreeMap::new();
        for (_, tensor) in self.checkpoint.tensors() {
            storages
                .entry(&tensor.storage.key)
                .or_insert(&tensor.storage);
        }
        for (key, storage) in storages {
            let name = self.storage_entry_name(key);
            let entry = self
                .zip
                .find(&name)
                .ok_or_else(|| Error::InvalidArchive(format!("storage {key} is missing")))?;
            if entry.method != 0 {
                return Err(Error::UnsupportedCompression {
                    method: entry.method,
                    entry: name,
                });
            }
            let expected = storage
                .elements
                .checked_mul(storage.dtype.element_size() as u64)
                .ok_or_else(|| Error::InvalidTensor(format!("storage {key} size overflow")))?;
            if entry.uncompressed_size < expected {
                return Err(Error::InvalidArchive(format!(
                    "storage {key} is truncated: expected at least {expected} bytes, found {}",
                    entry.uncompressed_size
                )));
            }
        }
        Ok(())
    }
}

fn gather_strided(meta: &TensorMeta, source: &[u8]) -> Result<Vec<u8>> {
    let width = meta.dtype.element_size();
    let count = usize::try_from(meta.element_count()?)
        .map_err(|_| Error::InvalidTensor("element count does not fit this platform".into()))?;
    let output_len = count
        .checked_mul(width)
        .ok_or_else(|| Error::InvalidTensor("tensor byte length overflow".into()))?;
    let mut output = Vec::with_capacity(output_len);
    for linear in 0..count as u64 {
        let mut remainder = linear;
        let mut source_element = 0u64;
        for (&dimension, &stride) in meta.shape.iter().zip(&meta.stride).rev() {
            let coordinate = remainder % dimension;
            remainder /= dimension;
            source_element = source_element
                .checked_add(
                    coordinate
                        .checked_mul(stride)
                        .ok_or_else(|| Error::InvalidTensor("strided offset overflow".into()))?,
                )
                .ok_or_else(|| Error::InvalidTensor("strided offset overflow".into()))?;
        }
        let start = usize::try_from(source_element)
            .ok()
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| Error::InvalidTensor("strided byte offset overflow".into()))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| Error::InvalidTensor("strided byte offset overflow".into()))?;
        output.extend_from_slice(source.get(start..end).ok_or_else(|| {
            Error::InvalidTensor("strided view exceeds loaded storage span".into())
        })?);
    }
    Ok(output)
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().map_or(false, u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().map_or(false, u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}
