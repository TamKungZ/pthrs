use crate::{Error, Result, TensorMeta};

pub(super) fn gather_strided(meta: &TensorMeta, source: &[u8]) -> Result<Vec<u8>> {
    let mut output = vec![0; output_len(meta)?];
    gather_strided_into(meta, source, &mut output)?;
    Ok(output)
}

pub(super) fn gather_strided_into(
    meta: &TensorMeta,
    source: &[u8],
    output: &mut [u8],
) -> Result<()> {
    let width = meta.dtype.element_size();
    if output.len() != output_len(meta)? {
        return Err(Error::InvalidTensor(
            "output buffer has the wrong length".into(),
        ));
    }
    for (linear, destination) in output.chunks_exact_mut(width).enumerate() {
        let start = source_byte_offset(meta, linear as u64, width)?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| Error::InvalidTensor("strided byte offset overflow".into()))?;
        destination.copy_from_slice(source.get(start..end).ok_or_else(|| {
            Error::InvalidTensor("strided view exceeds loaded storage span".into())
        })?);
    }
    Ok(())
}

fn output_len(meta: &TensorMeta) -> Result<usize> {
    let count = usize::try_from(meta.element_count()?)
        .map_err(|_| Error::InvalidTensor("element count does not fit this platform".into()))?;
    count
        .checked_mul(meta.dtype.element_size())
        .ok_or_else(|| Error::InvalidTensor("tensor byte length overflow".into()))
}

fn source_byte_offset(meta: &TensorMeta, linear: u64, width: usize) -> Result<usize> {
    let mut remainder = linear;
    let mut source_element = 0u64;
    for (&dimension, &stride) in meta.shape.iter().zip(&meta.stride).rev() {
        let coordinate = remainder % dimension;
        remainder /= dimension;
        source_element = source_element
            .checked_add(
                coordinate
                    .checked_mul(stride)
                    .ok_or_else(|| Error::InvalidTensor("strided offset overflow".into()))?,
            )
            .ok_or_else(|| Error::InvalidTensor("strided offset overflow".into()))?;
    }
    usize::try_from(source_element)
        .ok()
        .and_then(|value| value.checked_mul(width))
        .ok_or_else(|| Error::InvalidTensor("strided byte offset overflow".into()))
}

#[cfg(test)]
mod tests {
    use crate::{DType, StorageRef};

    use super::*;

    fn transposed_2x3() -> TensorMeta {
        TensorMeta {
            storage: StorageRef {
                key: "0".into(),
                dtype: DType::U8,
                location: "cpu".into(),
                elements: 6,
            },
            dtype: DType::U8,
            storage_offset: 0,
            shape: vec![3, 2],
            stride: vec![1, 3],
            requires_grad: false,
        }
    }

    #[test]
    fn gathers_transposed_view() {
        let meta = transposed_2x3();
        let gathered = gather_strided(&meta, &[0, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(gathered, [0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn rejects_short_source_and_wrong_output() {
        let meta = transposed_2x3();
        assert!(gather_strided(&meta, &[0, 1, 2]).is_err());
        assert!(gather_strided_into(&meta, &[0; 6], &mut [0; 5]).is_err());
    }
}
