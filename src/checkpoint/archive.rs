use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{
    buffer::resize_zeroed, ByteOrder, Error, F32Tensor, Result, TensorData, TensorMeta,
    TensorReadBuffer, TensorView,
};

use super::{
    legacy,
    metadata::Checkpoint,
    storage::{open_zip, Opened, Storage},
    strided::gather_strided_into,
};

/// A lazy reader over a `torch.save` checkpoint, either the ZIP layout or the
/// older single-stream layout.
///
/// Opening parses only the container index plus the object pickle; tensor
/// storage is read on demand by [`PthArchive::read_tensor`].
#[derive(Debug)]
pub struct PthArchive<R> {
    reader: R,
    storage: Storage,
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
        let file_len = reader.seek(SeekFrom::End(0))?;
        let mut head = vec![0; file_len.min(16) as usize];
        reader.seek(SeekFrom::Start(0))?;
        reader.read_exact(&mut head)?;
        let Opened {
            storage,
            checkpoint,
            byte_order,
            serialization_version,
        } = if legacy::is_legacy(&head) {
            legacy::open(&mut reader, file_len)?
        } else {
            open_zip(&mut reader)?
        };

        let archive = Self {
            reader,
            storage,
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
        let meta = self.tensor_meta(name)?;
        let mut bytes = Vec::new();
        self.read_tensor_bytes(&meta, &mut bytes, &mut Vec::new())?;
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
        let meta = self.tensor_meta(name)?;
        self.read_tensor_bytes(&meta, &mut buffer.bytes, &mut buffer.scratch)?;
        Ok(TensorView {
            meta,
            bytes: &buffer.bytes,
            byte_order: self.byte_order,
        })
    }

    fn tensor_meta(&self, name: &str) -> Result<TensorMeta> {
        self.checkpoint
            .tensor(name)
            .cloned()
            .ok_or_else(|| Error::TensorNotFound(name.to_owned()))
    }

    /// Fills `bytes` with the tensor in contiguous row-major order. `scratch`
    /// holds the enclosing storage span of a strided view.
    fn read_tensor_bytes(
        &mut self,
        meta: &TensorMeta,
        bytes: &mut Vec<u8>,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        let element_size = meta.dtype.element_size() as u64;
        resize_zeroed(bytes, to_usize(meta.byte_len()?, "tensor byte length")?)?;
        if bytes.is_empty() {
            return Ok(());
        }
        let key = &meta.storage.key;
        let byte_offset = meta
            .storage_offset
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidTensor("storage byte offset overflow".into()))?;
        if meta.is_contiguous() {
            return self
                .storage
                .read_range(&mut self.reader, key, byte_offset, bytes);
        }
        let span_bytes = meta
            .storage_span_elements()?
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidTensor("tensor storage span overflow".into()))?;
        resize_zeroed(scratch, to_usize(span_bytes, "tensor storage span")?)?;
        self.storage
            .read_range(&mut self.reader, key, byte_offset, scratch)?;
        gather_strided_into(meta, scratch, bytes)
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

    /// Loads and converts the state-dictionary tensors accepted by `keep`.
    pub fn load_f32_where(
        &mut self,
        mut keep: impl FnMut(&str, &TensorMeta) -> bool,
    ) -> Result<BTreeMap<String, F32Tensor>> {
        let names: Vec<String> = self
            .checkpoint
            .tensors()
            .filter(|(name, meta)| keep(name, meta))
            .map(|(name, _)| name.to_owned())
            .collect();
        let mut tensors = BTreeMap::new();
        for name in names {
            tensors.insert(name.clone(), self.read_tensor_f32(&name)?);
        }
        Ok(tensors)
    }

    /// Loads every floating-point tensor as `f32`, skipping integer entries
    /// such as batch-norm `num_batches_tracked` counters.
    pub fn load_floating_f32(&mut self) -> Result<BTreeMap<String, F32Tensor>> {
        self.load_f32_where(|_, meta| meta.dtype.is_floating_point())
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
        self.storage.read_all(&mut self.reader, storage_key)
    }

    pub fn into_inner(self) -> R {
        self.reader
    }

    fn validate_storages(&self) -> Result<()> {
        let mut storages = BTreeMap::new();
        for (_, tensor) in self.checkpoint.tensors() {
            storages
                .entry(&tensor.storage.key)
                .or_insert(&tensor.storage);
        }
        for (key, storage) in storages {
            let size = self.storage.size(key)?;
            let expected = storage.byte_len()?;
            if size < expected {
                return Err(Error::InvalidArchive(format!(
                    "storage {key} is truncated: expected at least {expected} bytes, found {size}"
                )));
            }
        }
        Ok(())
    }
}

fn to_usize(value: u64, what: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::LimitExceeded {
        what,
        value,
        limit: usize::MAX as u64,
    })
}
