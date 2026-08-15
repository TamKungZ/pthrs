use std::{
    cmp::Ordering,
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{Error, Result};

const INDEX_IVF_FLAT: [u8; 4] = *b"IwFl";
const INDEX_FLAT_L2: [u8; 4] = *b"IxF2";
const INDEX_FLAT_IP: [u8; 4] = *b"IxFI";
const INDEX_FLAT_GENERIC: [u8; 4] = *b"IxFl";
const ARRAY_INVERTED_LISTS: [u8; 4] = *b"ilar";
const LISTS_FULL: [u8; 4] = *b"full";
const LISTS_SPARSE: [u8; 4] = *b"sprs";

const MAX_DIMENSION: usize = 65_536;
const MAX_LISTS: usize = 1_000_000;
const MAX_VECTORS: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Metric {
    InnerProduct,
    L2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    pub id: i64,
    /// Squared distance for L2 or similarity for inner product.
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchOptions {
    pub k: usize,
    pub nprobe: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self { k: 8, nprobe: 1 }
    }
}

/// Reusable allocations for repeated nearest-neighbor searches.
#[derive(Debug, Default)]
pub struct SearchWorkspace {
    centroid_scores: Vec<(f32, usize)>,
    code_bytes: Vec<u8>,
    id_bytes: Vec<u8>,
    neighbors: Vec<Neighbor>,
}

impl SearchWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(lists: usize, neighbors: usize) -> Self {
        Self {
            centroid_scores: Vec::with_capacity(lists),
            code_bytes: Vec::new(),
            id_bytes: Vec::new(),
            neighbors: Vec::with_capacity(neighbors),
        }
    }

    pub fn reserve(&mut self, lists: usize, neighbors: usize) {
        self.centroid_scores
            .reserve(lists.saturating_sub(self.centroid_scores.len()));
        self.neighbors
            .reserve(neighbors.saturating_sub(self.neighbors.len()));
    }
    pub fn neighbors(&self) -> &[Neighbor] {
        &self.neighbors
    }
    pub fn clear(&mut self) {
        self.neighbors.clear();
    }
}

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

#[derive(Debug)]
struct LoadedList {
    vectors: Vec<f32>,
    ids: Vec<i64>,
}

/// In-memory IVF-Flat index intended for repeated low-latency queries.
#[derive(Debug)]
pub struct LoadedIvfFlatIndex {
    dimension: usize,
    total: u64,
    nlist: usize,
    default_nprobe: usize,
    metric: Metric,
    trained: bool,
    centroids: Vec<f32>,
    lists: Vec<LoadedList>,
    locations: HashMap<i64, (usize, usize)>,
}

impl LoadedIvfFlatIndex {
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

    /// Creates a workspace sized so loaded-index search performs no heap
    /// allocation while `k` does not exceed `max_neighbors`.
    pub fn workspace(&self, max_neighbors: usize) -> SearchWorkspace {
        SearchWorkspace::with_capacity(self.nlist, max_neighbors)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
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
        &self,
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
            let list = &self.lists[list_index];
            for (vector_index, &id) in list.ids.iter().enumerate() {
                let start = vector_index * self.dimension;
                let score = score_f32(
                    query,
                    &list.vectors[start..start + self.dimension],
                    self.metric,
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

    pub fn reconstruct(&self, id: i64) -> Option<&[f32]> {
        let &(list, vector) = self.locations.get(&id)?;
        let start = vector.checked_mul(self.dimension)?;
        self.lists[list].vectors.get(start..start + self.dimension)
    }

    pub fn reconstruct_into(&self, id: i64, output: &mut [f32]) -> Result<()> {
        if output.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                found: output.len(),
            });
        }
        let vector = self
            .reconstruct(id)
            .ok_or_else(|| Error::InvalidIndex(format!("vector ID {id} does not exist")))?;
        output.copy_from_slice(vector);
        Ok(())
    }

    /// Searches and applies inverse-squared-distance retrieval blending.
    /// This matches the common voice-index update:
    /// `output = retrieved * rate + query * (1 - rate)`.
    pub fn search_and_blend(
        &self,
        query: &[f32],
        output: &mut [f32],
        options: SearchOptions,
        rate: f32,
        workspace: &mut SearchWorkspace,
    ) -> Result<usize> {
        if output.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                found: output.len(),
            });
        }
        if self.metric != Metric::L2 {
            return Err(Error::InvalidIndex(
                "retrieval blending requires an L2 index".into(),
            ));
        }
        if !(0.0..=1.0).contains(&rate) || !rate.is_finite() {
            return Err(Error::InvalidIndex(
                "blend rate must be between 0 and 1".into(),
            ));
        }
        self.search_into(query, options, workspace)?;
        output.copy_from_slice(query);
        if workspace.neighbors.is_empty() || rate == 0.0 {
            return Ok(0);
        }

        let weight_sum: f64 = workspace
            .neighbors
            .iter()
            .map(|neighbor| {
                let distance = f64::from(neighbor.distance).max(1e-12);
                1.0 / (distance * distance)
            })
            .sum();
        for value in output.iter_mut() {
            *value *= 1.0 - rate;
        }
        for neighbor in &workspace.neighbors {
            let distance = f64::from(neighbor.distance).max(1e-12);
            let weight = ((1.0 / (distance * distance)) / weight_sum) as f32 * rate;
            if let Some(vector) = self.reconstruct(neighbor.id) {
                for (output, value) in output.iter_mut().zip(vector) {
                    *output += value * weight;
                }
            }
        }
        Ok(workspace.neighbors.len())
    }
}

#[derive(Clone, Copy, Debug)]
struct IndexHeader {
    dimension: usize,
    total: u64,
    trained: bool,
    metric: Metric,
}

fn read_index_header<R: Read>(reader: &mut R) -> Result<IndexHeader> {
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

fn validate_header(header: &IndexHeader) -> Result<()> {
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

fn skip_direct_map<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<()> {
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

fn read_list_sizes<R: Read>(reader: &mut R, nlist: usize) -> Result<Vec<usize>> {
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

fn validate_search(query: &[f32], dimension: usize, options: SearchOptions) -> Result<()> {
    if query.len() != dimension {
        return Err(Error::DimensionMismatch {
            expected: dimension,
            found: query.len(),
        });
    }
    if options.k == 0 {
        return Err(Error::InvalidIndex("search k is zero".into()));
    }
    if options.nprobe == 0 {
        return Err(Error::InvalidIndex("search nprobe is zero".into()));
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidIndex(
            "query contains a non-finite value".into(),
        ));
    }
    Ok(())
}

fn select_lists(
    query: &[f32],
    centroids: &[f32],
    dimension: usize,
    metric: Metric,
    nprobe: usize,
    scores: &mut Vec<(f32, usize)>,
) {
    scores.clear();
    scores.extend(
        centroids
            .chunks_exact(dimension)
            .enumerate()
            .map(|(index, centroid)| (score_f32(query, centroid, metric), index)),
    );
    scores.sort_unstable_by(|a, b| compare_score(a.0, b.0, metric));
    scores.truncate(nprobe);
}

fn score_f32(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::L2 => a
            .iter()
            .zip(b)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum(),
        Metric::InnerProduct => a.iter().zip(b).map(|(a, b)| a * b).sum(),
    }
}

fn score_bytes(query: &[f32], bytes: &[u8], metric: Metric) -> f32 {
    match metric {
        Metric::L2 => query
            .iter()
            .zip(bytes.chunks_exact(4))
            .map(|(query, bytes)| {
                let value = f32::from_le_bytes(bytes.try_into().expect("four bytes"));
                let difference = query - value;
                difference * difference
            })
            .sum(),
        Metric::InnerProduct => query
            .iter()
            .zip(bytes.chunks_exact(4))
            .map(|(query, bytes)| query * f32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .sum(),
    }
}

fn insert_neighbor(neighbors: &mut Vec<Neighbor>, candidate: Neighbor, k: usize, metric: Metric) {
    if neighbors.len() < k {
        neighbors.push(candidate);
        return;
    }
    let (worst_index, worst) = neighbors
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| compare_score(a.distance, b.distance, metric))
        .expect("non-empty neighbor list");
    if compare_score(candidate.distance, worst.distance, metric) == Ordering::Less {
        neighbors[worst_index] = candidate;
    }
}

fn sort_neighbors(neighbors: &mut [Neighbor], metric: Metric) {
    neighbors.sort_unstable_by(|a, b| compare_score(a.distance, b.distance, metric));
}

fn compare_score(a: f32, b: f32, metric: Metric) -> Ordering {
    match metric {
        Metric::L2 => a.total_cmp(&b),
        Metric::InnerProduct => b.total_cmp(&a),
    }
}

fn read_f32_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![
        0;
        len.checked_mul(4).ok_or_else(|| Error::InvalidIndex(
            "float vector byte length overflow".into()
        ))?
    ];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect())
}

fn checked_skip<R: Seek>(reader: &mut R, bytes: u64, file_len: u64) -> Result<()> {
    let current = reader.stream_position()?;
    let target = current
        .checked_add(bytes)
        .ok_or_else(|| Error::InvalidIndex("file offset overflow".into()))?;
    if target > file_len {
        return Err(Error::InvalidIndex("truncated index data".into()));
    }
    reader.seek(SeekFrom::Start(target))?;
    Ok(())
}

fn read_usize64<R: Read>(reader: &mut R, field: &str) -> Result<usize> {
    let value = read_u64(reader)?;
    usize::try_from(value)
        .map_err(|_| Error::InvalidIndex(format!("{field} does not fit this platform")))
}

fn expect_magic<R: Read>(reader: &mut R, expected: [u8; 4], name: &str) -> Result<()> {
    let found = read_magic(reader)?;
    if found != expected {
        return Err(Error::InvalidIndex(format!(
            "expected {name} magic {:?}, found {:?}",
            magic_string(expected),
            magic_string(found)
        )));
    }
    Ok(())
}

fn magic_string(magic: [u8; 4]) -> String {
    String::from_utf8_lossy(&magic).into_owned()
}
fn read_magic<R: Read>(reader: &mut R) -> Result<[u8; 4]> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut bytes = [0; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}
fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}
fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}
fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn reads_searches_and_reconstructs_ivf_flat() {
        let bytes = fixture_index();
        let mut lazy = FaissIvfFlatIndex::from_reader(Cursor::new(bytes.clone())).unwrap();
        assert_eq!(lazy.dimension(), 2);
        assert_eq!(lazy.len(), 2);
        assert_eq!(lazy.nlist(), 1);
        assert_eq!(lazy.metric(), Metric::L2);

        let neighbors = lazy.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(
            neighbors[0],
            Neighbor {
                id: 7,
                distance: 0.0
            }
        );
        assert_eq!(
            neighbors[1],
            Neighbor {
                id: 9,
                distance: 5.0
            }
        );

        let loaded = FaissIvfFlatIndex::from_reader(Cursor::new(bytes))
            .unwrap()
            .load()
            .unwrap();
        assert_eq!(loaded.reconstruct(9), Some(&[0.0, 2.0][..]));
        let mut output = [0.0; 2];
        loaded.reconstruct_into(7, &mut output).unwrap();
        assert_eq!(output, [1.0, 0.0]);

        let mut workspace = SearchWorkspace::new();
        loaded
            .search_and_blend(
                &[1.0, 0.0],
                &mut output,
                SearchOptions { k: 1, nprobe: 1 },
                1.0,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(output, [1.0, 0.0]);
    }

    #[test]
    fn rejects_wrong_query_dimension() {
        let mut index = FaissIvfFlatIndex::from_reader(Cursor::new(fixture_index())).unwrap();
        assert!(matches!(
            index.search(&[1.0], 1),
            Err(Error::DimensionMismatch {
                expected: 2,
                found: 1
            })
        ));
    }

    #[test]
    fn loaded_index_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LoadedIvfFlatIndex>();
    }

    fn fixture_index() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"IwFl");
        header(&mut out, 2, 2);
        u64v(&mut out, 1); // nlist
        u64v(&mut out, 1); // nprobe
        out.extend_from_slice(b"IxF2");
        header(&mut out, 2, 1);
        u64v(&mut out, 2); // centroid float count
        f32v(&mut out, 0.0);
        f32v(&mut out, 0.0);
        out.push(0); // DirectMap::NoMap
        u64v(&mut out, 0); // empty direct-map array
        out.extend_from_slice(b"ilar");
        u64v(&mut out, 1); // nlist
        u64v(&mut out, 8); // code size: 2 * f32
        out.extend_from_slice(b"full");
        u64v(&mut out, 1); // size vector length
        u64v(&mut out, 2); // list size
        f32v(&mut out, 1.0);
        f32v(&mut out, 0.0);
        f32v(&mut out, 0.0);
        f32v(&mut out, 2.0);
        i64v(&mut out, 7);
        i64v(&mut out, 9);
        out
    }

    fn header(out: &mut Vec<u8>, dimension: i32, total: i64) {
        out.extend_from_slice(&dimension.to_le_bytes());
        i64v(out, total);
        i64v(out, 1 << 20);
        i64v(out, 1 << 20);
        out.push(1);
        out.extend_from_slice(&1i32.to_le_bytes()); // L2
    }

    fn u64v(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn i64v(out: &mut Vec<u8>, value: i64) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn f32v(out: &mut Vec<u8>, value: f32) {
        out.extend_from_slice(&value.to_le_bytes());
    }
}
