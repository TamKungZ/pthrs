//! Pure-Rust access to ZIP-based PyTorch `.pth` checkpoints and FAISS
//! `IndexIVFFlat` retrieval indexes.
//!
//! `pthrs` reads the checkpoint format produced by `torch.save`, interprets
//! its safe data subset, exposes tensors lazily, and provides file-backed or
//! in-memory nearest-neighbor search. It does not execute pickle globals or
//! require Python, PyTorch, libtorch, FAISS, or a C/C++ runtime.

#![forbid(unsafe_code)]

mod checkpoint;
mod error;
mod faiss;
mod pickle;
mod tensor;
mod value;
mod voice;
mod zip;

pub use checkpoint::{Checkpoint, CheckpointSummary, MetadataIter, PthArchive, TensorIter};
pub use error::{Error, Result};
pub use faiss::{
    FaissIvfFlatIndex, LoadedIvfFlatIndex, Metric, Neighbor, SearchOptions, SearchWorkspace,
};
pub use pickle::parse_pickle;
pub use tensor::{
    ByteOrder, DType, F32Tensor, StorageRef, TensorData, TensorMeta, TensorReadBuffer, TensorView,
};
pub use value::Value;
pub use voice::{ValidationReport, VoiceModelConfig, VoiceModelInfo};
