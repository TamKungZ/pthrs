use std::io::{Read, Seek};

use crate::{Error, Result};

use super::{
    io::{
        checked_skip, magic_string, read_f32, read_i32, read_i64, read_magic, read_u8, read_usize64,
    },
    search::Metric,
};

pub(super) const INDEX_IVF_FLAT: [u8; 4] = *b"IwFl";
pub(super) const INDEX_FLAT_L2: [u8; 4] = *b"IxF2";
pub(super) const INDEX_FLAT_IP: [u8; 4] = *b"IxFI";
pub(super) const INDEX_FLAT_GENERIC: [u8; 4] = *b"IxFl";
pub(super) const ARRAY_INVERTED_LISTS: [u8; 4] = *b"ilar";
pub(super) const LISTS_FULL: [u8; 4] = *b"full";
pub(super) const LISTS_SPARSE: [u8; 4] = *b"sprs";

pub(super) const MAX_DIMENSION: usize = 65_536;
pub(super) const MAX_LISTS: usize = 1_000_000;
pub(super) const MAX_VECTORS: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug)]
pub(super) struct IndexHeader {
    pub(super) dimension: usize,
    pub(super) total: u64,
    pub(super) trained: bool,
    pub(super) metric: Metric,
}

pub(super) fn read_index_header<R: Read>(reader: &mut R) -> Result<IndexHeader> {
    let dimension_i32 = read_i32(reader)?;
    let dimension = usize::try_from(dimension_i32)
        .map_err(|_| Error::InvalidIndex("negative dimension".into()))?;
    let total_i64 = read_i64(reader)?;
    let total = u64::try_from(total_i64)
        .map_err(|_| Error::InvalidIndex("negative vector count".into()))?;
    let _legacy_dummy_a = read_i64(reader)?;
    let _legacy_dummy_b = read_i64(reader)?;
    let trained = match read_u8(reader)? {
        0 => false,
        1 => true,
        value => return Err(Error::InvalidIndex(format!("invalid bool value {value}"))),
    };
    let metric_code = read_i32(reader)?;
    let metric = match metric_code {
        0 => Metric::InnerProduct,
        1 => Metric::L2,
        other => {
            if other > 1 {
                let _metric_arg = read_f32(reader)?;
            }
            return Err(Error::InvalidIndex(format!(
                "unsupported metric type {other}"
            )));
        }
    };
    Ok(IndexHeader {
        dimension,
        total,
        trained,
        metric,
    })
}

pub(super) fn validate_header(header: &IndexHeader) -> Result<()> {
    if header.dimension == 0 || header.dimension > MAX_DIMENSION {
        return Err(Error::InvalidIndex(format!(
            "dimension {} is out of range",
            header.dimension
        )));
    }
    if header.total > MAX_VECTORS {
        return Err(Error::InvalidIndex(format!(
            "vector count {} is out of range",
            header.total
        )));
    }
    Ok(())
}

pub(super) fn skip_direct_map<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<()> {
    let kind = read_u8(reader)?;
    if kind > 2 {
        return Err(Error::InvalidIndex(format!(
            "unknown direct-map type {kind}"
        )));
    }
    let array_len = read_usize64(reader, "direct-map array length")?;
    checked_skip(
        reader,
        (array_len as u64)
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidIndex("direct-map size overflow".into()))?,
        file_len,
    )?;
    if kind == 2 {
        let pairs = read_usize64(reader, "direct-map hash length")?;
        checked_skip(
            reader,
            (pairs as u64)
                .checked_mul(16)
                .ok_or_else(|| Error::InvalidIndex("direct-map hash size overflow".into()))?,
            file_len,
        )?;
    }
    Ok(())
}

pub(super) fn read_list_sizes<R: Read>(reader: &mut R, nlist: usize) -> Result<Vec<usize>> {
    let kind = read_magic(reader)?;
    let mut sizes = vec![0; nlist];
    if kind == LISTS_FULL {
        let len = read_usize64(reader, "list-size vector length")?;
        if len != nlist {
            return Err(Error::InvalidIndex(
                "list-size vector length differs from nlist".into(),
            ));
        }
        for size in &mut sizes {
            *size = read_usize64(reader, "inverted-list size")?;
        }
    } else if kind == LISTS_SPARSE {
        let len = read_usize64(reader, "sparse list-size vector length")?;
        if len % 2 != 0 {
            return Err(Error::InvalidIndex(
                "sparse list-size vector has odd length".into(),
            ));
        }
        for _ in 0..len / 2 {
            let list = read_usize64(reader, "sparse list index")?;
            let size = read_usize64(reader, "sparse list size")?;
            let slot = sizes
                .get_mut(list)
                .ok_or_else(|| Error::InvalidIndex("sparse list index is out of range".into()))?;
            *slot = size;
        }
    } else {
        return Err(Error::InvalidIndex(format!(
            "unsupported list layout {:?}",
            magic_string(kind)
        )));
    }
    Ok(sizes)
}
