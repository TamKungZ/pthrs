use std::cmp::Ordering;

use crate::{Error, Result};

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
    pub(super) centroid_scores: Vec<(f32, usize)>,
    pub(super) code_bytes: Vec<u8>,
    pub(super) id_bytes: Vec<u8>,
    pub(super) neighbors: Vec<Neighbor>,
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

pub(super) fn validate_search(
    query: &[f32],
    dimension: usize,
    options: SearchOptions,
) -> Result<()> {
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

pub(super) fn select_lists(
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

pub(super) fn score_f32(a: &[f32], b: &[f32], metric: Metric) -> f32 {
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

pub(super) fn score_bytes(query: &[f32], bytes: &[u8], metric: Metric) -> f32 {
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

pub(super) fn insert_neighbor(
    neighbors: &mut Vec<Neighbor>,
    candidate: Neighbor,
    k: usize,
    metric: Metric,
) {
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

pub(super) fn sort_neighbors(neighbors: &mut [Neighbor], metric: Metric) {
    neighbors.sort_unstable_by(|a, b| compare_score(a.distance, b.distance, metric));
}

pub(super) fn compare_score(a: f32, b: f32, metric: Metric) -> Ordering {
    match metric {
        Metric::L2 => a.total_cmp(&b),
        Metric::InnerProduct => b.total_cmp(&a),
    }
}
