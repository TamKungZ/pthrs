# PyTorch `.pth` format notes

## Container

Modern checkpoints are ZIP archives sharing one generated prefix:

```text
<prefix>/data.pkl
<prefix>/byteorder
<prefix>/version
<prefix>/data/0
<prefix>/data/1
...
```

Each `data/<id>` entry is a raw typed storage array, normally stored without
compression.

## Pickle

Tensors use `torch._utils._rebuild_tensor_v2` with arguments
`(storage, storage_offset, shape, stride, requires_grad, backward_hooks)`.
The storage comes from a `BINPERSID` record:

```text
("storage", torch.<DType>Storage, "<id>", "<device>", element_count)
```

`<id>` maps to `<prefix>/data/<id>`. The device string is kept but not used.

Globals other than tensor and storage reconstructors, such as
`argparse.Namespace` in fairseq checkpoints, are never imported. `GLOBAL`,
`REDUCE`, `NEWOBJ`, `INST`, and `OBJ` produce inert `Value::Reduce` nodes and
`BUILD` state is discarded.

## Tensor views

Offset and stride are in elements. The storage element for coordinate `i` is:

```text
storage_offset + sum(i[axis] * stride[axis])
```

`read_tensor` returns contiguous row-major bytes. A contiguous view is one
range read; otherwise the smallest enclosing span is read and gathered.

## Voice-model fields

Inference checkpoints usually contain `weight` (state dictionary), `config`
(positional network configuration), `sr`, `f0`, `version`, and `info`. They
stay generic `Value` data.
