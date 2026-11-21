use std::cmp::Ordering;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
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

#[derive(Clone, Copy, Debug)]
pub(super) struct Hit {
    pub(super) neighbor: Neighbor,
    pub(super) list: usize,
    pub(super) slot: usize,
}

/// Reusable allocations for repeated nearest-neighbor searches.
#[derive(Debug, Default)]
pub struct SearchWorkspace {
    pub(super) centroid_scores: Vec<(f32, usize)>,
    pub(super) code_bytes: Vec<u8>,
    pub(super) id_bytes: Vec<u8>,
    pub(super) hits: Vec<Hit>,
    pub(super) neighbors: Vec<Neighbor>,
    pub(super) blend_output: Vec<f32>,
}

impl SearchWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(lists: usize, neighbors: usize) -> Self {
        Self {
            centroid_scores: Vec::with_capacity(lists),
            hits: Vec::with_capacity(neighbors),
            neighbors: Vec::with_capacity(neighbors),
            ..Self::default()
        }
    }

    pub fn reserve(&mut self, lists: usize, neighbors: usize) {
        self.centroid_scores
            .reserve(lists.saturating_sub(self.centroid_scores.len()));
        self.hits.reserve(neighbors.saturating_sub(self.hits.len()));
        self.neighbors
            .reserve(neighbors.saturating_sub(self.neighbors.len()));
    }

    /// Results of the most recent search, best first.
    pub fn neighbors(&self) -> &[Neighbor] {
        &self.neighbors
    }

    pub fn clear(&mut self) {
        self.hits.clear();
        self.neighbors.clear();
    }

    pub(super) fn finish(&mut self, metric: Metric) {
        self.hits
            .sort_unstable_by(|a, b| compare_hits(a, b, metric));
        self.neighbors.clear();
        self.neighbors
            .extend(self.hits.iter().map(|hit| hit.neighbor));
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
    let order =
        |a: &(f32, usize), b: &(f32, usize)| compare_score(a.0, b.0, metric).then(a.1.cmp(&b.1));
    scores.clear();
    scores.extend(
        centroids
            .chunks_exact(dimension)
            .enumerate()
            .map(|(index, centroid)| (score_f32(query, centroid, metric), index)),
    );
    if nprobe < scores.len() {
        scores.select_nth_unstable_by(nprobe - 1, order);
        scores.truncate(nprobe);
    }
    scores.sort_unstable_by(order);
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

/// Keeps the best `k` hits under a total order, so results do not depend on
/// scan order when distances tie.
pub(super) fn insert_hit(hits: &mut Vec<Hit>, candidate: Hit, k: usize, metric: Metric) {
    if hits.len() < k {
        hits.push(candidate);
        return;
    }
    let (worst_index, worst) = hits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| compare_hits(a, b, metric))
        .expect("non-empty hit list");
    if compare_hits(&candidate, worst, metric) == Ordering::Less {
        hits[worst_index] = candidate;
    }
}

fn compare_hits(a: &Hit, b: &Hit, metric: Metric) -> Ordering {
    compare_score(a.neighbor.distance, b.neighbor.distance, metric)
        .then(a.neighbor.id.cmp(&b.neighbor.id))
        .then(a.list.cmp(&b.list))
        .then(a.slot.cmp(&b.slot))
}

fn compare_score(a: f32, b: f32, metric: Metric) -> Ordering {
    match metric {
        Metric::L2 => a.total_cmp(&b),
        Metric::InnerProduct => b.total_cmp(&a),
    }
}
