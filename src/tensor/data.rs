use crate::Result;

use super::{decode::decode_f32_into, ByteOrder, TensorMeta};

#[derive(Clone, Debug)]
pub struct TensorData {
    pub meta: TensorMeta,
    /// Tensor bytes in contiguous row-major order.
    pub bytes: Vec<u8>,
    pub byte_order: ByteOrder,
}

#[derive(Clone, Debug)]
pub struct F32Tensor {
    pub meta: TensorMeta,
    pub values: Vec<f32>,
}

impl F32Tensor {
    /// Splits the tensor into a `usize` shape and its values, the form most
    /// tensor libraries take to build a tensor from a vector.
    pub fn into_shape_and_values(self) -> Result<(Vec<usize>, Vec<f32>)> {
        Ok((self.meta.shape_usize()?, self.values))
    }
}

/// Reusable byte buffers for allocation-aware tensor reads.
#[derive(Debug, Default)]
pub struct TensorReadBuffer {
    pub(crate) bytes: Vec<u8>,
    pub(crate) scratch: Vec<u8>,
}

impl TensorReadBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            scratch: Vec::new(),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    pub fn clear(&mut self) {
        self.bytes.clear();
        self.scratch.clear();
    }
}

#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub meta: TensorMeta,
    pub bytes: &'a [u8],
    pub byte_order: ByteOrder,
}

impl TensorData {
    pub fn len(&self) -> usize {
        self.bytes.len() / self.meta.dtype.element_size()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        self.decode_f32_into(&mut output)?;
        Ok(output)
    }

    pub fn decode_f32_into(&self, output: &mut Vec<f32>) -> Result<()> {
        decode_f32_into(self.meta.dtype, self.byte_order, &self.bytes, output)
    }
}

impl TensorView<'_> {
    pub fn len(&self) -> usize {
        self.bytes.len() / self.meta.dtype.element_size()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        self.decode_f32_into(&mut output)?;
        Ok(output)
    }

    pub fn decode_f32_into(&self, output: &mut Vec<f32>) -> Result<()> {
        decode_f32_into(self.meta.dtype, self.byte_order, self.bytes, output)
    }
}
