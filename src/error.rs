use std::{fmt, io};

/// Error returned by PTHrs.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    InvalidZip(&'static str),
    InvalidArchive(String),
    InvalidIndex(String),
    UnsupportedCompression {
        method: u16,
        entry: String,
    },
    Pickle {
        offset: usize,
        message: String,
    },
    UnsupportedPickle {
        offset: usize,
        opcode: u8,
    },
    InvalidTensor(String),
    TensorNotFound(String),
    DimensionMismatch {
        expected: usize,
        found: usize,
    },
    TypeMismatch {
        expected: &'static str,
        found: &'static str,
    },
    LimitExceeded {
        what: &'static str,
        value: u64,
        limit: u64,
    },
}

impl Error {
    pub(crate) fn pickle(offset: usize, message: impl Into<String>) -> Self {
        Self::Pickle {
            offset,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::InvalidZip(message) => write!(f, "invalid ZIP archive: {message}"),
            Self::InvalidArchive(message) => write!(f, "invalid PyTorch checkpoint: {message}"),
            Self::InvalidIndex(message) => write!(f, "invalid FAISS index: {message}"),
            Self::UnsupportedCompression { method, entry } => {
                write!(
                    f,
                    "ZIP compression method {method} is not supported for {entry}"
                )
            }
            Self::Pickle { offset, message } => {
                write!(f, "pickle error at byte {offset}: {message}")
            }
            Self::UnsupportedPickle { offset, opcode } => {
                write!(
                    f,
                    "unsupported pickle opcode 0x{opcode:02x} at byte {offset}"
                )
            }
            Self::InvalidTensor(message) => write!(f, "invalid tensor: {message}"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Self::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            Self::TypeMismatch { expected, found } => {
                write!(f, "expected {expected}, found {found}")
            }
            Self::LimitExceeded { what, value, limit } => {
                write!(
                    f,
                    "{what} is too large ({value}; configured limit is {limit})"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
