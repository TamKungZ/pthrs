use crate::DType;

use super::*;

#[test]
fn parses_checkpoint_metadata_shape() {
    let pickle = b"\x80\x02}(X\x06\x00\x00\x00config](K\x03K\x07eX\x07\x00\x00\x00versionX\x02\x00\x00\x00v2u.";
    let value = parse_pickle(pickle).unwrap();
    assert_eq!(value.get("version").and_then(Value::as_str), Some("v2"));
    assert_eq!(
        value.get("config").and_then(Value::as_list).unwrap().len(),
        2
    );
}

#[test]
fn parses_pytorch_tensor_reduction() {
    let pickle = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n((X\x07\x00\x00\x00storagectorch\nHalfStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x02tQK\x00K\x02\x85K\x01\x85\x89ccollections\nOrderedDict\n)RtR.";
    let value = parse_pickle(pickle).unwrap();
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor")
    };
    assert_eq!(tensor.dtype, DType::F16);
    assert_eq!(tensor.shape, vec![2]);
    assert!(tensor.is_contiguous());
}

#[test]
fn keeps_unknown_classes_inert() {
    let pickle = b"\x80\x02}(X\x03\x00\x00\x00cfgcargparse\nNamespace\n)\x81}bX\x03\x00\x00\x00old(K\x01iargparse\nNamespace\nu.";
    let value = parse_pickle(pickle).unwrap();
    let Some(Value::Reduce { callable, .. }) = value.get("cfg") else {
        panic!("expected inert reduction")
    };
    assert_eq!(
        **callable,
        Value::Global {
            module: "argparse".into(),
            name: "Namespace".into()
        }
    );
    assert!(matches!(value.get("old"), Some(Value::Reduce { .. })));
}

#[test]
fn fills_ordered_dict_from_item_list() {
    let pickle = b"\x80\x02ccollections\nOrderedDict\n]X\x01\x00\x00\x00aK\x01\x86a\x85R.";
    let value = parse_pickle(pickle).unwrap();
    assert_eq!(value.get("a").and_then(Value::as_i64), Some(1));
    assert_eq!(value.as_map().unwrap().len(), 1);
}

#[test]
fn decodes_hex_escapes_in_protocol_zero_strings() {
    let value = parse_pickle(
        br"S'a\x41\n'
.",
    )
    .unwrap();
    assert_eq!(value.as_str(), Some("aA\n"));
}

#[test]
fn rejects_deeply_nested_values() {
    let mut pickle = vec![0x80, 0x02, b')'];
    pickle.extend(std::iter::repeat(0x85).take(1000));
    pickle.push(b'.');
    assert!(matches!(
        parse_pickle(&pickle),
        Err(crate::Error::LimitExceeded { .. })
    ));
}

#[test]
fn rejects_exponential_duplication() {
    let mut pickle = vec![0x80, 0x02, b'}', b'q', 0];
    for _ in 0..40 {
        pickle.extend_from_slice(b"h\x00\x86");
        pickle.extend_from_slice(b"q\x00");
    }
    pickle.push(b'.');
    assert!(parse_pickle(&pickle).is_err());
}
