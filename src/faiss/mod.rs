mod format;
mod io;
mod lazy;
mod loaded;
mod search;

pub use lazy::FaissIvfFlatIndex;
pub use loaded::LoadedIvfFlatIndex;
pub use search::{Metric, Neighbor, SearchOptions, SearchWorkspace};

#[cfg(test)]
mod tests;
