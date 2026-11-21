use std::io::{Read, Seek, SeekFrom};

use crate::{Error, Result};

use super::{
    io::{
        checked_skip, ensure_available, expect_magic, magic_string, read_f32, read_f32_vec,
        read_i32, read_i64, read_magic, read_u8, read_usize64,
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

#[derive(Clone, Debug)]
pub(super) struct ListMeta {
    pub(super) len: usize,
    pub(super) codes_offset: u64,
    pub(super) ids_offset: u64,
}

pub(super) struct IvfLayout {
    pub(super) dimension: usize,
    pub(super) total: u64,
    pub(super) nlist: usize,
    pub(super) default_nprobe: usize,
    pub(super) metric: Metric,
    pub(super) trained: bool,
    pub(super) centroids: Vec<f32>,
    pub(super) lists: Vec<ListMeta>,
}

/// Parses and validates an `IndexIVFFlat` file, recording where each list's
/// vectors and IDs live without reading them.
pub(super) fn read_ivf_flat_layout<R: Read + Seek>(reader: &mut R) -> Result<IvfLayout> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;

    expect_magic(reader, INDEX_IVF_FLAT, "IndexIVFFlat")?;
    let header = read_index_header(reader)?;
    validate_header(&header)?;
    let dimension = header.dimension;

    let nlist = read_usize64(reader, "nlist")?;
    let default_nprobe = read_usize64(reader, "nprobe")?;
    if nlist == 0 || nlist > MAX_LISTS {
        return Err(Error::InvalidIndex(format!(
            "nlist {nlist} is out of range"
        )));
    }
    if default_nprobe == 0 {
        return Err(Error::InvalidIndex("nprobe is zero".into()));
    }

    let quantizer_magic = read_magic(reader)?;
    let quantizer_metric = match quantizer_magic {
        INDEX_FLAT_L2 => Metric::L2,
        INDEX_FLAT_IP => Metric::InnerProduct,
        INDEX_FLAT_GENERIC => header.metric,
        other => {
            return Err(Error::InvalidIndex(format!(
                "unsupported IVF quantizer {:?}",
                magic_string(other)
            )))
        }
    };
    let quantizer = read_index_header(reader)?;
    validate_header(&quantizer)?;
    if quantizer.dimension != dimension {
        return Err(Error::InvalidIndex(
            "quantizer dimension differs from index".into(),
        ));
    }
    if quantizer.total != nlist as u64 {
        return Err(Error::InvalidIndex(format!(
            "quantizer contains {} centroids, expected {nlist}",
            quantizer.total
        )));
    }
    if quantizer.metric != quantizer_metric || header.metric != quantizer_metric {
        return Err(Error::InvalidIndex(
            "quantizer and index metrics differ".into(),
        ));
    }
    let centroid_count = read_usize64(reader, "quantizer vector length")?;
    let expected_centroids = nlist
        .checked_mul(dimension)
        .ok_or_else(|| Error::InvalidIndex("centroid length overflow".into()))?;
    if centroid_count != expected_centroids {
        return Err(Error::InvalidIndex(format!(
            "quantizer contains {centroid_count} values, expected {expected_centroids}"
        )));
    }
    ensure_available(reader, centroid_count as u64 * 4, file_len)?;
    let centroids = read_f32_vec(reader, centroid_count)?;

    skip_direct_map(reader, file_len)?;
    expect_magic(reader, ARRAY_INVERTED_LISTS, "ArrayInvertedLists")?;
    let stored_nlist = read_usize64(reader, "inverted-list count")?;
    let code_size = read_usize64(reader, "inverted-list code size")?;
    if stored_nlist != nlist {
        return Err(Error::InvalidIndex(
            "inverted-list count differs from nlist".into(),
        ));
    }
    let expected_code_size = dimension
        .checked_mul(4)
        .ok_or_else(|| Error::InvalidIndex("code size overflow".into()))?;
    if code_size != expected_code_size {
        return Err(Error::InvalidIndex(format!(
            "code size is {code_size}, expected {expected_code_size}"
        )));
    }
    let sizes = read_list_sizes(reader, nlist)?;
    let listed_total = sizes.iter().try_fold(0u64, |total, &size| {
        total
            .checked_add(size as u64)
            .ok_or_else(|| Error::InvalidIndex("inverted-list total overflow".into()))
    })?;
    if listed_total != header.total {
        return Err(Error::InvalidIndex(format!(
            "inverted lists contain {listed_total} vectors, header declares {}",
            header.total
        )));
    }

    let mut lists = Vec::with_capacity(nlist);
    for len in sizes {
        let codes_offset = reader.stream_position()?;
        let code_bytes = len
            .checked_mul(code_size)
            .ok_or_else(|| Error::InvalidIndex("list code length overflow".into()))?;
        checked_skip(reader, code_bytes as u64, file_len)?;
        let ids_offset = reader.stream_position()?;
        let id_bytes = len
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidIndex("list ID length overflow".into()))?;
        checked_skip(reader, id_bytes as u64, file_len)?;
        lists.push(ListMeta {
            len,
            codes_offset,
            ids_offset,
        });
    }
    if reader.stream_position()? != file_len {
        return Err(Error::InvalidIndex("unexpected trailing bytes".into()));
    }

    Ok(IvfLayout {
        dimension,
        total: header.total,
        nlist,
        default_nprobe,
        metric: header.metric,
        trained: header.trained,
        centroids,
        lists,
    })
}
