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
