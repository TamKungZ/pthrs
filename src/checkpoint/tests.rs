use std::io::Cursor;

use crate::{DType, Error, PthArchive};

fn le16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn le32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn stored_zip(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = out.len() as u32;
        let size = data.len() as u32;
        le32(&mut out, 0x0403_4b50);
        le16(&mut out, 20);
        out.extend_from_slice(&[0; 8]);
        le32(&mut out, 0);
        le32(&mut out, size);
        le32(&mut out, size);
        le16(&mut out, name.len() as u16);
        le16(&mut out, 0);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        le32(&mut central, 0x0201_4b50);
        le16(&mut central, 20);
        le16(&mut central, 20);
        central.extend_from_slice(&[0; 8]);
        le32(&mut central, 0);
        le32(&mut central, size);
        le32(&mut central, size);
        le16(&mut central, name.len() as u16);
        central.extend_from_slice(&[0; 8]);
        le32(&mut central, 0);
        le32(&mut central, offset);
        central.extend_from_slice(name.as_bytes());
    }
    let central_offset = out.len() as u32;
    out.extend_from_slice(&central);
    le32(&mut out, 0x0605_4b50);
    out.extend_from_slice(&[0; 4]);
    le16(&mut out, entries.len() as u16);
    le16(&mut out, entries.len() as u16);
    le32(&mut out, central.len() as u32);
    le32(&mut out, central_offset);
    le16(&mut out, 0);
    out
}

fn int(out: &mut Vec<u8>, value: i32) {
    out.push(b'J');
    out.extend_from_slice(&value.to_le_bytes());
}

fn text(out: &mut Vec<u8>, value: &str) {
    out.push(b'X');
    le32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn tuple(out: &mut Vec<u8>, values: &[i32]) {
    out.push(b'(');
    for &value in values {
        int(out, value);
    }
    out.push(b't');
}

struct TensorSpec<'a> {
    name: &'a str,
    storage_type: &'a str,
    key: &'a str,
    elements: i32,
    shape: &'a [i32],
    stride: &'a [i32],
}

fn checkpoint_pickle(tensors: &[TensorSpec]) -> Vec<u8> {
    let mut out = vec![0x80, 0x02, b'}', b'('];
    for tensor in tensors {
        text(&mut out, tensor.name);
        out.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((");
        text(&mut out, "storage");
        out.extend_from_slice(format!("ctorch\n{}\n", tensor.storage_type).as_bytes());
        text(&mut out, tensor.key);
        text(&mut out, "cpu");
        int(&mut out, tensor.elements);
        out.extend_from_slice(b"tQ");
        int(&mut out, 0);
        tuple(&mut out, tensor.shape);
        tuple(&mut out, tensor.stride);
        out.extend_from_slice(b"\x89ccollections\nOrderedDict\n)RtR");
    }
    out.extend_from_slice(b"u.");
    out
}

fn floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn archive(tensors: &[TensorSpec], storages: &[(&str, Vec<u8>)]) -> PthArchive<Cursor<Vec<u8>>> {
    let mut entries = vec![("m/data.pkl", checkpoint_pickle(tensors))];
    let names: Vec<String> = storages
        .iter()
        .map(|(key, _)| format!("m/data/{key}"))
        .collect();
    for (name, (_, data)) in names.iter().zip(storages) {
        entries.push((name.as_str(), data.clone()));
    }
    PthArchive::from_reader(Cursor::new(stored_zip(&entries))).unwrap()
}

#[test]
fn reads_contiguous_and_strided_tensors() {
    let storage = floats(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut archive = archive(
        &[
            TensorSpec {
                name: "w",
                storage_type: "FloatStorage",
                key: "0",
                elements: 6,
                shape: &[2, 3],
                stride: &[3, 1],
            },
            TensorSpec {
                name: "t",
                storage_type: "FloatStorage",
                key: "0",
                elements: 6,
                shape: &[3, 2],
                stride: &[1, 3],
            },
        ],
        &[("0", storage)],
    );

    let plain = archive.read_tensor_f32("w").unwrap();
    assert_eq!(plain.values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let transposed = archive.read_tensor_f32("t").unwrap();
    assert_eq!(transposed.values, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

    let mut buffer = crate::TensorReadBuffer::new();
    let view = archive.read_tensor_into("t", &mut buffer).unwrap();
    assert_eq!(view.to_f32_vec().unwrap(), transposed.values);
}

#[test]
fn filters_tensors_by_dtype() {
    let mut archive = archive(
        &[
            TensorSpec {
                name: "weight",
                storage_type: "FloatStorage",
                key: "0",
                elements: 2,
                shape: &[2],
                stride: &[1],
            },
            TensorSpec {
                name: "num_batches_tracked",
                storage_type: "LongStorage",
                key: "1",
                elements: 1,
                shape: &[],
                stride: &[],
            },
        ],
        &[
            ("0", floats(&[1.0, 2.0])),
            ("1", 7i64.to_le_bytes().to_vec()),
        ],
    );

    assert_eq!(archive.load_all_f32().unwrap().len(), 2);
    let floating = archive.load_floating_f32().unwrap();
    assert_eq!(floating.keys().collect::<Vec<_>>(), ["weight"]);
    let integers: Vec<_> = archive
        .checkpoint()
        .tensors_with_dtype(DType::I64)
        .map(|(name, _)| name)
        .collect();
    assert_eq!(integers, ["num_batches_tracked"]);
}

#[test]
fn reports_failure_instead_of_allocating_an_expanded_tensor() {
    let mut archive = archive(
        &[TensorSpec {
            name: "expanded",
            storage_type: "FloatStorage",
            key: "0",
            elements: 1,
            shape: &[1 << 30, 1 << 30],
            stride: &[0, 0],
        }],
        &[("0", floats(&[1.0]))],
    );

    assert!(matches!(
        archive.read_tensor("expanded"),
        Err(Error::Allocation { .. })
    ));
}
