use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteOrder {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
    Complex32,
    Complex64,
    Complex128,
}

impl DType {
    pub fn element_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 | Self::Complex32 => 4,
            Self::I64 | Self::F64 | Self::Complex64 => 8,
            Self::Complex128 => 16,
        }
    }

    pub(crate) fn from_storage_global(module: &str, name: &str) -> Option<Self> {
        if module != "torch" {
            return None;
        }
        Some(match name {
            "BoolStorage" => Self::Bool,
            "ByteStorage" | "UntypedStorage" => Self::U8,
            "CharStorage" => Self::I8,
            "ShortStorage" => Self::I16,
            "IntStorage" => Self::I32,
            "LongStorage" => Self::I64,
            "HalfStorage" => Self::F16,
            "BFloat16Storage" => Self::BF16,
            "FloatStorage" => Self::F32,
            "DoubleStorage" => Self::F64,
            "ComplexHalfStorage" => Self::Complex32,
            "ComplexFloatStorage" => Self::Complex64,
            "ComplexDoubleStorage" => Self::Complex128,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageRef {
    pub key: String,
    pub dtype: DType,
    pub location: String,
    pub elements: u64,
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

#[derive(Clone, Debug)]
pub struct TensorData {
    pub meta: TensorMeta,
    /// Tensor bytes in contiguous row-major order.
    pub bytes: Vec<u8>,
    pub byte_order: ByteOrder,
}

impl TensorData {
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let width = self.meta.dtype.element_size();
        if self.bytes.len() % width != 0 {
            return Err(Error::InvalidTensor(
                "tensor byte length is not element-aligned".into(),
            ));
        }
        self.bytes
            .chunks_exact(width)
            .map(|bytes| self.element_to_f32(bytes))
            .collect()
    }

    fn element_to_f32(&self, bytes: &[u8]) -> Result<f32> {
        let u16v = || match self.byte_order {
            ByteOrder::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
            ByteOrder::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
        };
        let u32v = || match self.byte_order {
            ByteOrder::Little => u32::from_le_bytes(bytes.try_into().expect("four bytes")),
            ByteOrder::Big => u32::from_be_bytes(bytes.try_into().expect("four bytes")),
        };
        let u64v = || match self.byte_order {
            ByteOrder::Little => u64::from_le_bytes(bytes.try_into().expect("eight bytes")),
            ByteOrder::Big => u64::from_be_bytes(bytes.try_into().expect("eight bytes")),
        };
        Ok(match self.meta.dtype {
            DType::Bool => {
                if bytes[0] == 0 {
                    0.0
                } else {
                    1.0
                }
            }
            DType::U8 => bytes[0] as f32,
            DType::I8 => (bytes[0] as i8) as f32,
            DType::I16 => (u16v() as i16) as f32,
            DType::I32 => (u32v() as i32) as f32,
            DType::I64 => (u64v() as i64) as f32,
            DType::F16 => half_to_f32(u16v()),
            DType::BF16 => f32::from_bits((u16v() as u32) << 16),
            DType::F32 => f32::from_bits(u32v()),
            DType::F64 => f64::from_bits(u64v()) as f32,
            DType::Complex32 | DType::Complex64 | DType::Complex128 => {
                return Err(Error::TypeMismatch {
                    expected: "real-valued tensor",
                    found: "complex tensor",
                });
            }
        })
    }
}

fn half_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = (value >> 10) & 0x1f;
    let fraction = value & 0x03ff;
    let bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction as u32;
            let mut exponent = 113u32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((fraction & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | ((fraction as u32) << 13),
        _ => sign | (((exponent as u32) + 112) << 23) | ((fraction as u32) << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::half_to_f32;

    #[test]
    fn decodes_half_precision() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert!(half_to_f32(0x7e00).is_nan());
    }
}
