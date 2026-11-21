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
