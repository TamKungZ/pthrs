use std::io::Cursor;

use crate::{ByteOrder, DType, Error, PthArchive, TensorReadBuffer};

fn le16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn le32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn le64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[derive(Clone, Copy, Default)]
struct ZipOptions {
    deflate: bool,
    zip64: bool,
    method: Option<u16>,
}

fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![1];
    le16(&mut out, data.len() as u16);
    le16(&mut out, !(data.len() as u16));
    out.extend_from_slice(data);
    out
}

fn build_zip(entries: &[(&str, Vec<u8>)], options: ZipOptions) -> Vec<u8> {
    let method = options
        .method
        .unwrap_or(if options.deflate { 8 } else { 0 });
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let payload = if method == 8 {
            deflate_stored(data)
        } else {
            data.clone()
        };
        let offset = out.len() as u64;
        le32(&mut out, 0x0403_4b50);
        le16(&mut out, 20);
        le16(&mut out, 0);
        le16(&mut out, method);
        out.extend_from_slice(&[0; 8]);
        le32(&mut out, payload.len() as u32);
        le32(&mut out, data.len() as u32);
        le16(&mut out, name.len() as u16);
        le16(&mut out, 0);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&payload);

        le32(&mut central, 0x0201_4b50);
        le16(&mut central, 45);
        le16(&mut central, 45);
        le16(&mut central, 0);
        le16(&mut central, method);
        central.extend_from_slice(&[0; 8]);
        let extra = if options.zip64 {
            let mut extra = Vec::new();
            le16(&mut extra, 1);
            le16(&mut extra, 24);
            le64(&mut extra, data.len() as u64);
            le64(&mut extra, payload.len() as u64);
            le64(&mut extra, offset);
            extra
        } else {
            Vec::new()
        };
        let (compressed, uncompressed, local) = if options.zip64 {
            (u32::MAX, u32::MAX, u32::MAX)
        } else {
            (payload.len() as u32, data.len() as u32, offset as u32)
        };
        le32(&mut central, compressed);
        le32(&mut central, uncompressed);
        le16(&mut central, name.len() as u16);
        le16(&mut central, extra.len() as u16);
        central.extend_from_slice(&[0; 10]);
        le32(&mut central, local);
        central.extend_from_slice(name.as_bytes());
        central.extend_from_slice(&extra);
    }
    let central_offset = out.len() as u64;
    out.extend_from_slice(&central);
    if options.zip64 {
        let record_offset = out.len() as u64;
        le32(&mut out, 0x0606_4b50);
        le64(&mut out, 44);
        le16(&mut out, 45);
        le16(&mut out, 45);
        le32(&mut out, 0);
        le32(&mut out, 0);
        le64(&mut out, entries.len() as u64);
        le64(&mut out, entries.len() as u64);
        le64(&mut out, central.len() as u64);
        le64(&mut out, central_offset);
        le32(&mut out, 0x0706_4b50);
        le32(&mut out, 0);
        le64(&mut out, record_offset);
        le32(&mut out, 1);
        le32(&mut out, 0x0605_4b50);
        out.extend_from_slice(&[0; 4]);
        le16(&mut out, u16::MAX);
        le16(&mut out, u16::MAX);
        le32(&mut out, u32::MAX);
        le32(&mut out, u32::MAX);
    } else {
        le32(&mut out, 0x0605_4b50);
        out.extend_from_slice(&[0; 4]);
        le16(&mut out, entries.len() as u16);
        le16(&mut out, entries.len() as u16);
        le32(&mut out, central.len() as u32);
        le32(&mut out, central_offset as u32);
    }
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

fn archive_with(
    tensors: &[TensorSpec],
    storages: &[(&str, Vec<u8>)],
    extra: &[(&str, Vec<u8>)],
    options: ZipOptions,
) -> PthArchive<Cursor<Vec<u8>>> {
    let mut entries = vec![("m/data.pkl", checkpoint_pickle(tensors))];
    entries.extend(extra.iter().cloned());
    let names: Vec<String> = storages
        .iter()
        .map(|(key, _)| format!("m/data/{key}"))
        .collect();
    for (name, (_, data)) in names.iter().zip(storages) {
        entries.push((name.as_str(), data.clone()));
    }
    PthArchive::from_reader(Cursor::new(build_zip(&entries, options))).unwrap()
}

fn archive(tensors: &[TensorSpec], storages: &[(&str, Vec<u8>)]) -> PthArchive<Cursor<Vec<u8>>> {
    archive_with(tensors, storages, &[], ZipOptions::default())
}

fn transposed_specs() -> [TensorSpec<'static>; 2] {
    [
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
    ]
}

#[test]
fn reads_contiguous_and_strided_tensors() {
    let storage = floats(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut archive = archive(&transposed_specs(), &[("0", storage)]);

    let plain = archive.read_tensor_f32("w").unwrap();
    assert_eq!(plain.values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let transposed = archive.read_tensor_f32("t").unwrap();
    assert_eq!(transposed.values, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

    let mut buffer = TensorReadBuffer::new();
    let view = archive.read_tensor_into("t", &mut buffer).unwrap();
    assert_eq!(view.to_f32_vec().unwrap(), transposed.values);
}

#[test]
fn reads_deflated_and_zip64_archives() {
    for options in [
        ZipOptions {
            deflate: true,
            ..ZipOptions::default()
        },
        ZipOptions {
            zip64: true,
            ..ZipOptions::default()
        },
        ZipOptions {
            deflate: true,
            zip64: true,
            ..ZipOptions::default()
        },
    ] {
        let storage = floats(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut archive = archive_with(&transposed_specs(), &[("0", storage)], &[], options);
        let transposed = archive.read_tensor_f32("t").unwrap();
        assert_eq!(transposed.values, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}

#[test]
fn rejects_unsupported_compression_methods() {
    let entries = [
        ("m/data.pkl", checkpoint_pickle(&transposed_specs())),
        ("m/data/0", floats(&[0.0; 6])),
    ];
    let bytes = build_zip(
        &entries,
        ZipOptions {
            method: Some(93),
            ..ZipOptions::default()
        },
    );
    assert!(matches!(
        PthArchive::from_reader(Cursor::new(bytes)),
        Err(Error::UnsupportedCompression { method: 93, .. })
    ));
}

#[test]
fn reads_big_endian_archives() {
    let storage: Vec<u8> = [1.5f32, -2.0]
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let spec = [TensorSpec {
        name: "w",
        storage_type: "FloatStorage",
        key: "0",
        elements: 2,
        shape: &[2],
        stride: &[1],
    }];
    let mut archive = archive_with(
        &spec,
        &[("0", storage)],
        &[("m/byteorder", b"big".to_vec())],
        ZipOptions::default(),
    );

    assert_eq!(archive.byte_order(), ByteOrder::Big);
    assert_eq!(archive.read_tensor_f32("w").unwrap().values, [1.5, -2.0]);
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

fn legacy_file(
    tensors: &[TensorSpec],
    storages: &[(&str, i64, Vec<u8>)],
    little_endian: bool,
) -> Vec<u8> {
    let mut out = vec![
        0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, b'.',
    ];
    out.extend_from_slice(b"\x80\x02M\xe9\x03.");
    out.extend_from_slice(b"\x80\x02}(X\x0d\x00\x00\x00little_endian");
    out.push(if little_endian { 0x88 } else { 0x89 });
    out.extend_from_slice(b"u.");
    out.extend_from_slice(&checkpoint_pickle(tensors));
    out.extend_from_slice(b"\x80\x02](");
    for (key, _, _) in storages {
        text(&mut out, key);
    }
    out.extend_from_slice(b"e.");
    for (_, elements, data) in storages {
        if little_endian {
            out.extend_from_slice(&elements.to_le_bytes());
        } else {
            out.extend_from_slice(&elements.to_be_bytes());
        }
        out.extend_from_slice(data);
    }
    out
}

#[test]
fn reads_legacy_single_stream_checkpoints() {
    let storage = floats(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let file = legacy_file(&transposed_specs(), &[("0", 6, storage)], true);
    let mut archive = PthArchive::from_reader(Cursor::new(file)).unwrap();

    assert_eq!(archive.serialization_version(), Some("1001"));
    assert_eq!(archive.byte_order(), ByteOrder::Little);
    let transposed = archive.read_tensor_f32("t").unwrap();
    assert_eq!(transposed.values, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    assert_eq!(archive.read_storage("0").unwrap().len(), 24);
}

#[test]
fn reads_big_endian_legacy_checkpoints() {
    let storage: Vec<u8> = [2.5f32, -1.0]
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let spec = [TensorSpec {
        name: "w",
        storage_type: "FloatStorage",
        key: "0",
        elements: 2,
        shape: &[2],
        stride: &[1],
    }];
    let file = legacy_file(&spec, &[("0", 2, storage)], false);
    let mut archive = PthArchive::from_reader(Cursor::new(file)).unwrap();

    assert_eq!(archive.byte_order(), ByteOrder::Big);
    assert_eq!(archive.read_tensor_f32("w").unwrap().values, [2.5, -1.0]);
}

#[test]
fn reads_legacy_checkpoints_with_a_large_object_pickle() {
    let name = "x".repeat(3 << 20);
    let spec = [TensorSpec {
        name: &name,
        storage_type: "FloatStorage",
        key: "0",
        elements: 1,
        shape: &[1],
        stride: &[1],
    }];
    let file = legacy_file(&spec, &[("0", 1, floats(&[9.0]))], true);
    let mut archive = PthArchive::from_reader(Cursor::new(file)).unwrap();
    assert_eq!(archive.read_tensor_f32(&name).unwrap().values, [9.0]);
}

#[test]
fn rejects_legacy_checkpoints_with_inconsistent_storage() {
    let spec = [TensorSpec {
        name: "w",
        storage_type: "FloatStorage",
        key: "0",
        elements: 2,
        shape: &[2],
        stride: &[1],
    }];
    let wrong_count = legacy_file(&spec, &[("0", 3, floats(&[1.0, 2.0]))], true);
    assert!(PthArchive::from_reader(Cursor::new(wrong_count)).is_err());
    let mut truncated = legacy_file(&spec, &[("0", 2, floats(&[1.0, 2.0]))], true);
    truncated.truncate(truncated.len() - 4);
    assert!(PthArchive::from_reader(Cursor::new(truncated)).is_err());
}
