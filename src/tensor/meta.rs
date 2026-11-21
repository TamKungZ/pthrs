use crate::{DType, Error, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageRef {
    pub key: String,
    pub dtype: DType,
    pub location: String,
    pub elements: u64,
}

impl StorageRef {
    pub fn byte_len(&self) -> Result<u64> {
        self.elements
            .checked_mul(self.dtype.element_size() as u64)
            .ok_or_else(|| Error::InvalidTensor("storage byte length overflow".into()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorMeta {
    pub storage: StorageRef,
    pub dtype: DType,
    pub storage_offset: u64,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
    pub requires_grad: bool,
}

impl TensorMeta {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Shape as `usize` dimensions, failing if one does not fit this platform.
    pub fn shape_usize(&self) -> Result<Vec<usize>> {
        self.shape
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension).map_err(|_| {
                    Error::InvalidTensor("dimension does not fit this platform".into())
                })
            })
            .collect()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
    }

    pub fn element_count(&self) -> Result<u64> {
        self.shape.iter().try_fold(1u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or_else(|| Error::InvalidTensor("element count overflow".into()))
        })
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.len() != self.stride.len() {
            return false;
        }
        let mut expected = 1u64;
        for (&dimension, &stride) in self.shape.iter().zip(&self.stride).rev() {
            if dimension > 1 && stride != expected {
                return false;
            }
            match expected.checked_mul(dimension) {
                Some(next) => expected = next,
                None => return false,
            }
        }
        true
    }

    pub fn byte_len(&self) -> Result<u64> {
        self.element_count()?
            .checked_mul(self.dtype.element_size() as u64)
            .ok_or_else(|| Error::InvalidTensor("tensor byte length overflow".into()))
    }

    pub(crate) fn storage_span_elements(&self) -> Result<u64> {
        if self.shape.len() != self.stride.len() {
            return Err(Error::InvalidTensor("shape and stride ranks differ".into()));
        }
        if self.shape.contains(&0) {
            return Ok(0);
        }
        self.shape
            .iter()
            .zip(&self.stride)
            .try_fold(1u64, |span, (&dim, &stride)| {
                let contribution = (dim - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| Error::InvalidTensor("storage span overflow".into()))?;
                span.checked_add(contribution)
                    .ok_or_else(|| Error::InvalidTensor("storage span overflow".into()))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_shape_to_usize() {
        let meta = TensorMeta {
            storage: StorageRef {
                key: "0".into(),
                dtype: DType::F32,
                location: "cpu".into(),
                elements: 6,
            },
            dtype: DType::F32,
            storage_offset: 0,
            shape: vec![2, 3],
            stride: vec![3, 1],
            requires_grad: false,
        };
        assert_eq!(meta.shape_usize().unwrap(), vec![2, 3]);
    }
}
