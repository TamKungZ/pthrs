use crate::{Error, Result};

use super::{ByteOrder, DType};

pub(super) fn decode_f32_into(
    dtype: DType,
    byte_order: ByteOrder,
    bytes: &[u8],
    output: &mut Vec<f32>,
) -> Result<()> {
    let width = dtype.element_size();
    if bytes.len() % width != 0 {
        return Err(Error::InvalidTensor(
            "tensor byte length is not element-aligned".into(),
        ));
    }
    output.clear();
    output.reserve(bytes.len() / width);
    for element in bytes.chunks_exact(width) {
        output.push(element_to_f32(dtype, byte_order, element)?);
    }
    Ok(())
}

fn element_to_f32(dtype: DType, byte_order: ByteOrder, bytes: &[u8]) -> Result<f32> {
    let u16v = || match byte_order {
        ByteOrder::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
        ByteOrder::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
    };
    let u32v = || match byte_order {
        ByteOrder::Little => u32::from_le_bytes(bytes.try_into().expect("four bytes")),
        ByteOrder::Big => u32::from_be_bytes(bytes.try_into().expect("four bytes")),
    };
    let u64v = || match byte_order {
        ByteOrder::Little => u64::from_le_bytes(bytes.try_into().expect("eight bytes")),
        ByteOrder::Big => u64::from_be_bytes(bytes.try_into().expect("eight bytes")),
    };
    Ok(match dtype {
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
        DType::U16 => u16v() as f32,
        DType::I32 => (u32v() as i32) as f32,
        DType::U32 => u32v() as f32,
        DType::I64 => (u64v() as i64) as f32,
        DType::U64 => u64v() as f32,
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
    use crate::{ByteOrder, DType};

    use super::{decode_f32_into, half_to_f32};

    fn decode(dtype: DType, order: ByteOrder, bytes: &[u8]) -> Vec<f32> {
        let mut output = Vec::new();
        decode_f32_into(dtype, order, bytes, &mut output).unwrap();
        output
    }

    #[test]
    fn decodes_half_precision() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn decodes_integer_and_wide_types() {
        let little = ByteOrder::Little;
        assert_eq!(
            decode(DType::U16, little, &65_535u16.to_le_bytes()),
            [65_535.0]
        );
        assert_eq!(decode(DType::I16, little, &(-2i16).to_le_bytes()), [-2.0]);
        assert_eq!(
            decode(DType::U32, little, &4_000_000_000u32.to_le_bytes()),
            [4e9]
        );
        assert_eq!(decode(DType::I64, little, &(-7i64).to_le_bytes()), [-7.0]);
        assert_eq!(
            decode(DType::U64, little, &(1u64 << 40).to_le_bytes()),
            [1.0995116e12]
        );
        assert_eq!(decode(DType::F64, little, &1.5f64.to_le_bytes()), [1.5]);
        assert_eq!(decode(DType::BF16, little, &0x3f80u16.to_le_bytes()), [1.0]);
        assert_eq!(decode(DType::Bool, little, &[0, 1]), [0.0, 1.0]);
    }

    #[test]
    fn decodes_big_endian_data() {
        let big = ByteOrder::Big;
        assert_eq!(decode(DType::F32, big, &1.25f32.to_be_bytes()), [1.25]);
        assert_eq!(decode(DType::F16, big, &0x3c00u16.to_be_bytes()), [1.0]);
        assert_eq!(decode(DType::I32, big, &(-9i32).to_be_bytes()), [-9.0]);
        assert_eq!(decode(DType::F64, big, &(-0.5f64).to_be_bytes()), [-0.5]);
    }

    #[test]
    fn rejects_complex_and_misaligned_data() {
        let mut output = Vec::new();
        assert!(
            decode_f32_into(DType::Complex64, ByteOrder::Little, &[0; 8], &mut output).is_err()
        );
        assert!(decode_f32_into(DType::F32, ByteOrder::Little, &[0; 3], &mut output).is_err());
    }
}
