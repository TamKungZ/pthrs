use std::collections::HashMap;

use crate::{Error, Result};

use super::search::{
    insert_neighbor, score_f32, select_lists, sort_neighbors, validate_search, Metric, Neighbor,
    SearchOptions, SearchWorkspace,
};

#[derive(Debug)]
pub(super) struct LoadedList {
    pub(super) vectors: Vec<f32>,
    pub(super) ids: Vec<i64>,
}

/// In-memory IVF-Flat index intended for repeated low-latency queries.
#[derive(Debug)]
pub struct LoadedIvfFlatIndex {
    pub(super) dimension: usize,
    pub(super) total: u64,
    pub(super) nlist: usize,
    pub(super) default_nprobe: usize,
    pub(super) metric: Metric,
    pub(super) trained: bool,
    pub(super) centroids: Vec<f32>,
    pub(super) lists: Vec<LoadedList>,
    pub(super) locations: HashMap<i64, (usize, usize)>,
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
        let mut workspace = SearchWorkspace::with_capacity(self.nlist, max_neighbors);
        workspace.blend_output.reserve(self.dimension);
        workspace
    }

    /// Replaces every row-major feature frame in `features` with its
    /// retrieval blend and returns the total neighbors used. Allocation-free
    /// when `workspace` came from [`Self::workspace`].
    pub fn blend_frames(
        &self,
        features: &mut [f32],
        options: SearchOptions,
        rate: f32,
        workspace: &mut SearchWorkspace,
    ) -> Result<usize> {
        if features.is_empty() || features.len() % self.dimension != 0 {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                found: features.len(),
            });
        }
        let mut scratch = std::mem::take(&mut workspace.blend_output);
        scratch.resize(self.dimension, 0.0);
        let mut used = 0;
        let mut result = Ok(());
        for frame in features.chunks_exact_mut(self.dimension) {
            match self.search_and_blend(frame, &mut scratch, options, rate, workspace) {
                Ok(count) => {
                    used += count;
                    frame.copy_from_slice(&scratch);
                }
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        workspace.blend_output = scratch;
        result.map(|()| used)
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
