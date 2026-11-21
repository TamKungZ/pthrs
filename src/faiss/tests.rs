use std::io::Cursor;

use crate::Error;

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
    fixture_index_with_ids([7, 9])
}

fn fixture_index_with_ids(ids: [i64; 2]) -> Vec<u8> {
    IndexSpec {
        ids,
        ..IndexSpec::default()
    }
    .build()
}

struct IndexSpec {
    ids: [i64; 2],
    metric: i32,
    quantizer: [u8; 4],
    sparse_lists: bool,
    direct_map: u8,
}

impl Default for IndexSpec {
    fn default() -> Self {
        Self {
            ids: [7, 9],
            metric: 1,
            quantizer: *b"IxF2",
            sparse_lists: false,
            direct_map: 0,
        }
    }
}

impl IndexSpec {
    fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"IwFl");
        header_with_metric(&mut out, 2, 2, self.metric);
        u64v(&mut out, 1); // nlist
        u64v(&mut out, 1); // nprobe
        out.extend_from_slice(&self.quantizer);
        header_with_metric(&mut out, 2, 1, self.metric);
        u64v(&mut out, 2); // centroid float count
        f32v(&mut out, 0.0);
        f32v(&mut out, 0.0);
        out.push(self.direct_map);
        match self.direct_map {
            1 => {
                u64v(&mut out, 2);
                i64v(&mut out, 0);
                i64v(&mut out, 1);
            }
            2 => {
                u64v(&mut out, 0);
                u64v(&mut out, 2);
                for pair in [(7, 0), (9, 1)] {
                    i64v(&mut out, pair.0);
                    i64v(&mut out, pair.1);
                }
            }
            _ => u64v(&mut out, 0),
        }
        out.extend_from_slice(b"ilar");
        u64v(&mut out, 1); // nlist
        u64v(&mut out, 8); // code size: 2 * f32
        if self.sparse_lists {
            out.extend_from_slice(b"sprs");
            u64v(&mut out, 2); // one (list, size) pair
            u64v(&mut out, 0);
            u64v(&mut out, 2);
        } else {
            out.extend_from_slice(b"full");
            u64v(&mut out, 1); // size vector length
            u64v(&mut out, 2); // list size
        }
        f32v(&mut out, 1.0);
        f32v(&mut out, 0.0);
        f32v(&mut out, 0.0);
        f32v(&mut out, 2.0);
        i64v(&mut out, self.ids[0]);
        i64v(&mut out, self.ids[1]);
        out
    }
}

fn header(out: &mut Vec<u8>, dimension: i32, total: i64) {
    header_with_metric(out, dimension, total, 1);
}

fn header_with_metric(out: &mut Vec<u8>, dimension: i32, total: i64, metric: i32) {
    out.extend_from_slice(&dimension.to_le_bytes());
    i64v(out, total);
    i64v(out, 1 << 20);
    i64v(out, 1 << 20);
    out.push(1);
    out.extend_from_slice(&metric.to_le_bytes());
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

#[test]
fn blends_each_frame_like_single_search() {
    let index = FaissIvfFlatIndex::from_reader(Cursor::new(fixture_index()))
        .unwrap()
        .load()
        .unwrap();
    let options = SearchOptions { k: 2, nprobe: 1 };
    let mut workspace = index.workspace(2);
    let mut features = [1.0, 0.5, 0.0, 1.5];
    let mut expected = features;
    for frame in expected.chunks_exact_mut(2) {
        let query = frame.to_vec();
        index
            .search_and_blend(&query, frame, options, 0.5, &mut workspace)
            .unwrap();
    }

    let used = index
        .blend_frames(&mut features, options, 0.5, &mut workspace)
        .unwrap();

    assert_eq!(used, 4);
    assert_eq!(features, expected);
    assert!(matches!(
        index.blend_frames(&mut [1.0, 2.0, 3.0], options, 0.5, &mut workspace),
        Err(Error::DimensionMismatch { .. })
    ));
}

#[test]
fn blends_the_hit_vector_when_ids_repeat() {
    let index = FaissIvfFlatIndex::from_reader(Cursor::new(fixture_index_with_ids([7, 7])))
        .unwrap()
        .load()
        .unwrap();
    let mut workspace = index.workspace(1);
    let mut output = [0.0; 2];

    index
        .search_and_blend(
            &[0.0, 2.0],
            &mut output,
            SearchOptions { k: 1, nprobe: 1 },
            1.0,
            &mut workspace,
        )
        .unwrap();

    assert_eq!(output, [0.0, 2.0]);
}

#[test]
fn orders_equal_distances_by_id() {
    let mut index =
        FaissIvfFlatIndex::from_reader(Cursor::new(fixture_index_with_ids([9, 7]))).unwrap();
    let found = index.search(&[0.5, 1.0], 2).unwrap();
    assert_eq!(found[0].distance, found[1].distance);
    assert_eq!([found[0].id, found[1].id], [7, 9]);
}

#[test]
fn rejects_centroid_block_larger_than_file() {
    let mut out = Vec::new();
    out.extend_from_slice(b"IwFl");
    header(&mut out, 60_000, 0);
    u64v(&mut out, 1_000);
    u64v(&mut out, 1);
    out.extend_from_slice(b"IxF2");
    header(&mut out, 60_000, 1_000);
    u64v(&mut out, 60_000_000);
    assert!(matches!(
        FaissIvfFlatIndex::from_reader(Cursor::new(out)),
        Err(Error::InvalidIndex(_))
    ));
}

#[test]
fn reads_sparse_list_layout() {
    let spec = IndexSpec {
        sparse_lists: true,
        ..IndexSpec::default()
    };
    let mut sparse = FaissIvfFlatIndex::from_reader(Cursor::new(spec.build())).unwrap();
    let mut full = FaissIvfFlatIndex::from_reader(Cursor::new(fixture_index())).unwrap();
    assert_eq!(
        sparse.search(&[1.0, 0.0], 2).unwrap(),
        full.search(&[1.0, 0.0], 2).unwrap()
    );
}

#[test]
fn reads_direct_maps_and_generic_quantizer() {
    for spec in [
        IndexSpec {
            direct_map: 1,
            ..IndexSpec::default()
        },
        IndexSpec {
            direct_map: 2,
            ..IndexSpec::default()
        },
        IndexSpec {
            quantizer: *b"IxFl",
            ..IndexSpec::default()
        },
    ] {
        let mut index = FaissIvfFlatIndex::from_reader(Cursor::new(spec.build())).unwrap();
        assert_eq!(index.search(&[1.0, 0.0], 1).unwrap()[0].id, 7);
    }
}

#[test]
fn ranks_inner_product_by_similarity() {
    let spec = IndexSpec {
        metric: 0,
        quantizer: *b"IxFI",
        ..IndexSpec::default()
    };
    let index = FaissIvfFlatIndex::from_reader(Cursor::new(spec.build()))
        .unwrap()
        .load()
        .unwrap();
    assert_eq!(index.metric(), Metric::InnerProduct);

    let found = index.search(&[0.0, 1.0], 2).unwrap();
    assert_eq!(
        found[0],
        Neighbor {
            id: 9,
            distance: 2.0
        }
    );
    assert_eq!(
        found[1],
        Neighbor {
            id: 7,
            distance: 0.0
        }
    );

    let mut workspace = index.workspace(2);
    assert!(index
        .search_and_blend(
            &[0.0, 1.0],
            &mut [0.0; 2],
            SearchOptions::default(),
            0.5,
            &mut workspace
        )
        .is_err());
}
