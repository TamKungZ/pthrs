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
