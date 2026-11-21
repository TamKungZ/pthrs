use crate::{Error, Result};

/// Replaces the contents of `buffer` with `len` zero bytes, reporting an
/// allocation failure instead of aborting.
pub(crate) fn resize_zeroed(buffer: &mut Vec<u8>, len: usize) -> Result<()> {
    buffer.clear();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| Error::Allocation { bytes: len as u64 })?;
    buffer.resize(len, 0);
    Ok(())
}
