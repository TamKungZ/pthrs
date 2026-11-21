use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{Error, Result};

use super::{
    format::{read_ivf_flat_layout, ListMeta},
    io::{read_f32_vec, read_i64_vec},
    loaded::{LoadedIvfFlatIndex, LoadedList},
    search::{
        insert_hit, score_bytes, select_lists, validate_search, Hit, Metric, Neighbor,
        SearchOptions, SearchWorkspace,
    },
};

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
        let layout = read_ivf_flat_layout(&mut reader)?;
        Ok(Self {
            reader,
            dimension: layout.dimension,
            total: layout.total,
            nlist: layout.nlist,
            default_nprobe: layout.default_nprobe,
            metric: layout.metric,
            trained: layout.trained,
            centroids: layout.centroids,
            lists: layout.lists,
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
        workspace.hits.clear();

        for &(_, list_index) in &workspace.centroid_scores {
            let list = &self.lists[list_index];
            let code_len = list
                .len
                .checked_mul(self.dimension)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| Error::InvalidIndex("list code length overflow".into()))?;
            workspace.code_bytes.resize(code_len, 0);
            self.reader.seek(SeekFrom::Start(list.codes_offset))?;
            self.reader.read_exact(&mut workspace.code_bytes)?;
            workspace.id_bytes.resize(list.len * 8, 0);
            self.reader.seek(SeekFrom::Start(list.ids_offset))?;
            self.reader.read_exact(&mut workspace.id_bytes)?;

            for slot in 0..list.len {
                let code_start = slot * self.dimension * 4;
                let distance = score_bytes(
                    query,
                    &workspace.code_bytes[code_start..code_start + self.dimension * 4],
                    self.metric,
                );
                let id_start = slot * 8;
                let id = i64::from_le_bytes(
                    workspace.id_bytes[id_start..id_start + 8]
                        .try_into()
                        .expect("eight bytes"),
                );
                insert_hit(
                    &mut workspace.hits,
                    Hit {
                        neighbor: Neighbor { id, distance },
                        list: list_index,
                        slot,
                    },
                    options.k,
                    self.metric,
                );
            }
        }
        workspace.finish(self.metric);
        Ok(&workspace.neighbors)
    }

    /// Loads every inverted list into memory for low-latency repeated search.
    pub fn load(mut self) -> Result<LoadedIvfFlatIndex> {
        let mut lists = Vec::with_capacity(self.lists.len());
        let mut locations = HashMap::with_capacity(self.total as usize);
        for (list_index, meta) in self.lists.iter().enumerate() {
            let value_count = meta
                .len
                .checked_mul(self.dimension)
                .ok_or_else(|| Error::InvalidIndex("list value count overflow".into()))?;
            self.reader.seek(SeekFrom::Start(meta.codes_offset))?;
            let vectors = read_f32_vec(&mut self.reader, value_count)?;
            self.reader.seek(SeekFrom::Start(meta.ids_offset))?;
            let ids = read_i64_vec(&mut self.reader, meta.len)?;
            for (slot, &id) in ids.iter().enumerate() {
                locations.entry(id).or_insert((list_index, slot));
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
