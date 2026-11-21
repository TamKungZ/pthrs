use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{Error, Result};

use super::{
    format::{
        read_index_header, read_list_sizes, skip_direct_map, validate_header, ARRAY_INVERTED_LISTS,
        INDEX_FLAT_GENERIC, INDEX_FLAT_IP, INDEX_FLAT_L2, INDEX_IVF_FLAT, MAX_LISTS,
    },
    io::{
        checked_skip, expect_magic, magic_string, read_f32_vec, read_i64, read_magic, read_usize64,
    },
    loaded::{LoadedIvfFlatIndex, LoadedList},
    search::{
        insert_neighbor, score_bytes, select_lists, sort_neighbors, validate_search, Metric,
        Neighbor, SearchOptions, SearchWorkspace,
    },
};

#[derive(Clone, Debug)]
struct ListMeta {
    len: usize,
    codes_offset: u64,
    ids_offset: u64,
}

/// Lazy reader for the FAISS `IndexIVFFlat` layout commonly paired with voice
/// checkpoints. List data stays on the underlying `Read + Seek` source.
#[derive(Debug)]
pub struct FaissIvfFlatIndex<R> {
    reader: R,
    dimension: usize,
    total: u64,
    nlist: usize,
    default_nprobe: usize,
    metric: Metric,
    trained: bool,
    centroids: Vec<f32>,
    lists: Vec<ListMeta>,
}

impl FaissIvfFlatIndex<File> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_reader(File::open(path)?)
    }
}

impl<R: Read + Seek> FaissIvfFlatIndex<R> {
    pub fn from_reader(mut reader: R) -> Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;

        expect_magic(&mut reader, INDEX_IVF_FLAT, "IndexIVFFlat")?;
        let header = read_index_header(&mut reader)?;
        validate_header(&header)?;
        let dimension = header.dimension;

        let nlist = read_usize64(&mut reader, "nlist")?;
        let default_nprobe = read_usize64(&mut reader, "nprobe")?;
        if nlist == 0 || nlist > MAX_LISTS {
            return Err(Error::InvalidIndex(format!(
                "nlist {nlist} is out of range"
            )));
        }
        if default_nprobe == 0 {
            return Err(Error::InvalidIndex("nprobe is zero".into()));
        }

        let quantizer_magic = read_magic(&mut reader)?;
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
        let quantizer = read_index_header(&mut reader)?;
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
        let centroid_count = read_usize64(&mut reader, "quantizer vector length")?;
        let expected_centroids = nlist
            .checked_mul(dimension)
            .ok_or_else(|| Error::InvalidIndex("centroid length overflow".into()))?;
        if centroid_count != expected_centroids {
            return Err(Error::InvalidIndex(format!(
                "quantizer contains {centroid_count} values, expected {expected_centroids}"
            )));
        }
        let centroids = read_f32_vec(&mut reader, centroid_count)?;

        skip_direct_map(&mut reader, file_len)?;
        expect_magic(&mut reader, ARRAY_INVERTED_LISTS, "ArrayInvertedLists")?;
        let stored_nlist = read_usize64(&mut reader, "inverted-list count")?;
        let code_size = read_usize64(&mut reader, "inverted-list code size")?;
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
        let sizes = read_list_sizes(&mut reader, nlist)?;
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
            checked_skip(&mut reader, code_bytes as u64, file_len)?;
            let ids_offset = reader.stream_position()?;
            let id_bytes = len
                .checked_mul(8)
                .ok_or_else(|| Error::InvalidIndex("list ID length overflow".into()))?;
            checked_skip(&mut reader, id_bytes as u64, file_len)?;
            lists.push(ListMeta {
                len,
                codes_offset,
                ids_offset,
            });
        }
        if reader.stream_position()? != file_len {
            return Err(Error::InvalidIndex("unexpected trailing bytes".into()));
        }

        Ok(Self {
            reader,
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

    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn len(&self) -> u64 {
        self.total
    }
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }
    pub fn nlist(&self) -> usize {
        self.nlist
    }
    pub fn default_nprobe(&self) -> usize {
        self.default_nprobe
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn is_trained(&self) -> bool {
        self.trained
    }
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        let mut workspace = SearchWorkspace::new();
        self.search_into(
            query,
            SearchOptions {
                k,
                nprobe: self.default_nprobe,
            },
            &mut workspace,
        )?;
        Ok(workspace.neighbors)
    }

    pub fn search_into<'a>(
        &mut self,
        query: &[f32],
        options: SearchOptions,
        workspace: &'a mut SearchWorkspace,
    ) -> Result<&'a [Neighbor]> {
        validate_search(query, self.dimension, options)?;
        select_lists(
            query,
            &self.centroids,
            self.dimension,
            self.metric,
            options.nprobe.min(self.nlist),
            &mut workspace.centroid_scores,
        );
        workspace.neighbors.clear();

        for &(_, list_index) in workspace
            .centroid_scores
            .iter()
            .take(options.nprobe.min(self.nlist))
        {
            let list = self.lists[list_index].clone();
            let code_len = list
                .len
                .checked_mul(self.dimension)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| Error::InvalidIndex("list code length overflow".into()))?;
            workspace.code_bytes.resize(code_len, 0);
            self.reader.seek(SeekFrom::Start(list.codes_offset))?;
            self.reader.read_exact(&mut workspace.code_bytes)?;
            workspace.id_bytes.resize(
                list.len
                    .checked_mul(8)
                    .ok_or_else(|| Error::InvalidIndex("list ID length overflow".into()))?,
                0,
            );
            self.reader.seek(SeekFrom::Start(list.ids_offset))?;
            self.reader.read_exact(&mut workspace.id_bytes)?;

            for vector_index in 0..list.len {
                let code_start = vector_index * self.dimension * 4;
                let score = score_bytes(
                    query,
                    &workspace.code_bytes[code_start..code_start + self.dimension * 4],
                    self.metric,
                );
                let id_start = vector_index * 8;
                let id = i64::from_le_bytes(
                    workspace.id_bytes[id_start..id_start + 8]
                        .try_into()
                        .expect("eight bytes"),
                );
                insert_neighbor(
                    &mut workspace.neighbors,
                    Neighbor {
                        id,
                        distance: score,
                    },
                    options.k,
                    self.metric,
                );
            }
        }
        sort_neighbors(&mut workspace.neighbors, self.metric);
        Ok(&workspace.neighbors)
    }

    /// Loads every inverted list into memory for low-latency repeated search.
    pub fn load(mut self) -> Result<LoadedIvfFlatIndex> {
        let mut lists = Vec::with_capacity(self.lists.len());
        let mut locations = HashMap::with_capacity(self.total as usize);
        for (list_index, meta) in self.lists.iter().cloned().enumerate() {
            let value_count = meta
                .len
                .checked_mul(self.dimension)
                .ok_or_else(|| Error::InvalidIndex("list value count overflow".into()))?;
            self.reader.seek(SeekFrom::Start(meta.codes_offset))?;
            let vectors = read_f32_vec(&mut self.reader, value_count)?;
            self.reader.seek(SeekFrom::Start(meta.ids_offset))?;
            let mut ids = Vec::with_capacity(meta.len);
            for vector_index in 0..meta.len {
                let id = read_i64(&mut self.reader)?;
                locations.entry(id).or_insert((list_index, vector_index));
                ids.push(id);
            }
            lists.push(LoadedList { vectors, ids });
        }
        Ok(LoadedIvfFlatIndex {
            dimension: self.dimension,
            total: self.total,
            nlist: self.nlist,
            default_nprobe: self.default_nprobe,
            metric: self.metric,
            trained: self.trained,
            centroids: self.centroids,
            lists,
            locations,
        })
    }

    pub fn into_inner(self) -> R {
        self.reader
    }
}
