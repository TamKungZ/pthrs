//! Pure-Rust access to ZIP-based PyTorch `.pth` checkpoints.
//!
//! `pthrs` reads the checkpoint format produced by `torch.save`, interprets
//! its safe data subset, and exposes tensors lazily.
//! It does not execute pickle globals or require Python, PyTorch, libtorch, or
//! a C/C++ runtime.

#![forbid(unsafe_code)]

mod checkpoint;
mod error;
mod pickle;
mod tensor;
mod value;
mod zip;

pub use checkpoint::{Checkpoint, MetadataIter, PthArchive, TensorIter};
pub use error::{Error, Result};
pub use pickle::parse_pickle;
pub use tensor::{ByteOrder, DType, StorageRef, TensorData, TensorMeta};
pub use value::Value;
