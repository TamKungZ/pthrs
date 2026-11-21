use std::{collections::BTreeMap, slice};

use crate::{
    pickle::{parse_pickle_prefix, parse_pickle_with_protocol},
    DType, Error, Result, TensorMeta, Value,
};

/// Parsed, inert metadata from a PyTorch checkpoint.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    protocol: u8,
    root: Value,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CheckpointSummary {
    pub tensor_count: usize,
    pub storage_count: usize,
    pub tensor_elements: u64,
    pub logical_tensor_bytes: u64,
    pub storage_bytes: u64,
}

impl Checkpoint {
    pub fn from_pickle(bytes: &[u8]) -> Result<Self> {
        let (protocol, root) = parse_pickle_with_protocol(bytes)?;
        Ok(Self { protocol, root })
    }

    /// Parses the first pickle in `bytes` and reports how many bytes it used.
    pub(super) fn from_pickle_prefix(bytes: &[u8]) -> Result<(Self, usize)> {
        let (protocol, root, used) = parse_pickle_prefix(bytes)?;
        Ok((Self { protocol, root }, used))
    }

    pub fn pickle_protocol(&self) -> u8 {
        self.protocol
    }
    pub fn root(&self) -> &Value {
        &self.root
    }

    pub fn into_root(self) -> Value {
        self.root
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.root.get(key)
    }

    /// Iterates top-level checkpoint metadata, excluding the potentially large
    /// `weight` dictionary.
    pub fn metadata(&self) -> MetadataIter<'_> {
        MetadataIter {
            inner: self.root.as_map().unwrap_or(&[]).iter(),
            state_dict_key: self.state_dict_key(),
        }
    }

    /// Returns a conventional `weight`, `state_dict`, or `model` dictionary.
    /// A root dictionary containing tensors directly is also accepted.
    pub fn state_dict(&self) -> Option<&Value> {
        if let Some(key) = self.state_dict_key() {
            return self.root.get(key);
        }
        map_contains_tensor(&self.root).then_some(&self.root)
    }

    /// Iterates tensors in the detected state dictionary.
    pub fn tensors(&self) -> TensorIter<'_> {
        let entries = self.state_dict().and_then(Value::as_map).unwrap_or(&[]);
        TensorIter {
            inner: entries.iter(),
        }
    }

    /// Iterates state-dictionary tensors stored with the given element type.
    pub fn tensors_with_dtype(&self, dtype: DType) -> impl Iterator<Item = (&str, &TensorMeta)> {
        self.tensors()
            .filter(move |(_, tensor)| tensor.dtype == dtype)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors()
            .find_map(|(candidate, tensor)| (candidate == name).then_some(tensor))
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors().count()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors().map(|(name, _)| name)
    }

    pub fn summary(&self) -> Result<CheckpointSummary> {
        let mut summary = CheckpointSummary::default();
        let mut storages = BTreeMap::new();
        for (_, tensor) in self.tensors() {
            summary.tensor_count += 1;
            let elements = tensor.element_count()?;
            summary.tensor_elements = summary
                .tensor_elements
                .checked_add(elements)
                .ok_or_else(|| Error::InvalidTensor("tensor element total overflow".into()))?;
            let bytes = elements
                .checked_mul(tensor.dtype.element_size() as u64)
                .ok_or_else(|| Error::InvalidTensor("tensor byte total overflow".into()))?;
            summary.logical_tensor_bytes = summary
                .logical_tensor_bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::InvalidTensor("tensor byte total overflow".into()))?;
            storages
                .entry(&tensor.storage.key)
                .or_insert(&tensor.storage);
        }
        summary.storage_count = storages.len();
        for storage in storages.values() {
            let bytes = storage
                .elements
                .checked_mul(storage.dtype.element_size() as u64)
                .ok_or_else(|| Error::InvalidTensor("storage byte total overflow".into()))?;
            summary.storage_bytes = summary
                .storage_bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::InvalidTensor("storage byte total overflow".into()))?;
        }
        Ok(summary)
    }

    fn state_dict_key(&self) -> Option<&str> {
        ["weight", "state_dict", "model"]
            .into_iter()
            .find(|key| self.root.get(key).map(map_contains_tensor).unwrap_or(false))
    }
}

pub struct MetadataIter<'a> {
    inner: slice::Iter<'a, (Value, Value)>,
    state_dict_key: Option<&'a str>,
}
impl<'a> Iterator for MetadataIter<'a> {
    type Item = (&'a str, &'a Value);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.find_map(|(key, value)| {
            let key = key.as_str()?;
            (Some(key) != self.state_dict_key).then_some((key, value))
        })
    }
}

fn map_contains_tensor(value: &Value) -> bool {
    value
        .as_map()
        .map(|entries| {
            entries
                .iter()
                .any(|(_, value)| matches!(value, Value::Tensor(_)))
        })
        .unwrap_or(false)
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
