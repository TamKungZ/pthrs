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

    pub fn is_floating_point(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Complex32 => "complex32",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
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
