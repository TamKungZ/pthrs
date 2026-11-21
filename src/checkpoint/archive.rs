use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek},
    path::Path,
};

use crate::{
    zip, ByteOrder, Error, F32Tensor, Result, TensorData, TensorMeta, TensorReadBuffer, TensorView,
};

use super::{
    metadata::Checkpoint,
    strided::{gather_strided, gather_strided_into},
};

const MAX_DATA_PICKLE: u64 = 64 * 1024 * 1024;
const MAX_AUXILIARY_ENTRY: u64 = 1024 * 1024;

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
        let checkpoint = Checkpoint::from_pickle(&pickle)?;

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

    /// Reads a tensor into reusable caller-owned buffers.
    pub fn read_tensor_into<'a>(
        &mut self,
        name: &str,
        buffer: &'a mut TensorReadBuffer,
    ) -> Result<TensorView<'a>> {
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
        let output_len = usize::try_from(output_bytes).map_err(|_| Error::LimitExceeded {
            what: "tensor byte length",
            value: output_bytes,
            limit: usize::MAX as u64,
        })?;
        buffer.bytes.clear();
        buffer.bytes.resize(output_len, 0);
        if element_count != 0 {
            let entry_name = self.storage_entry_name(&meta.storage.key);
            let entry = self.zip.find(&entry_name).cloned().ok_or_else(|| {
                Error::InvalidArchive(format!("storage {} is missing", meta.storage.key))
            })?;
            let byte_offset = meta
                .storage_offset
                .checked_mul(element_size)
                .ok_or_else(|| Error::InvalidTensor("storage byte offset overflow".into()))?;
            if meta.is_contiguous() {
                self.zip.read_range_into(
                    &mut self.reader,
                    &entry,
                    byte_offset,
                    &mut buffer.bytes,
                )?;
            } else {
                let span_elements = meta.storage_span_elements()?;
                let span_bytes = span_elements
                    .checked_mul(element_size)
                    .ok_or_else(|| Error::InvalidTensor("tensor storage span overflow".into()))?;
                let span_len = usize::try_from(span_bytes).map_err(|_| Error::LimitExceeded {
                    what: "tensor storage span",
                    value: span_bytes,
                    limit: usize::MAX as u64,
                })?;
                buffer.scratch.clear();
                buffer.scratch.resize(span_len, 0);
                self.zip.read_range_into(
                    &mut self.reader,
                    &entry,
                    byte_offset,
                    &mut buffer.scratch,
                )?;
                gather_strided_into(&meta, &buffer.scratch, &mut buffer.bytes)?;
            }
        }
        Ok(TensorView {
            meta,
            bytes: &buffer.bytes,
            byte_order: self.byte_order,
        })
    }

    pub fn read_tensor_f32(&mut self, name: &str) -> Result<F32Tensor> {
        let tensor = self.read_tensor(name)?;
        Ok(F32Tensor {
            meta: tensor.meta.clone(),
            values: tensor.to_f32_vec()?,
        })
    }

    pub fn read_tensor_f32_into(
        &mut self,
        name: &str,
        bytes: &mut TensorReadBuffer,
        output: &mut Vec<f32>,
    ) -> Result<TensorMeta> {
        let tensor = self.read_tensor_into(name, bytes)?;
        tensor.decode_f32_into(output)?;
        Ok(tensor.meta)
    }

    /// Loads and converts every state-dictionary tensor for model startup.
    pub fn load_all_f32(&mut self) -> Result<BTreeMap<String, F32Tensor>> {
        let names: Vec<String> = self.checkpoint.tensor_names().map(str::to_owned).collect();
        let mut tensors = BTreeMap::new();
        for name in names {
            tensors.insert(name.clone(), self.read_tensor_f32(&name)?);
        }
        Ok(tensors)
    }

    pub fn load_all_tensors(&mut self) -> Result<BTreeMap<String, TensorData>> {
        let names: Vec<String> = self.checkpoint.tensor_names().map(str::to_owned).collect();
        let mut tensors = BTreeMap::new();
        for name in names {
            tensors.insert(name.clone(), self.read_tensor(&name)?);
        }
        Ok(tensors)
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

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}
