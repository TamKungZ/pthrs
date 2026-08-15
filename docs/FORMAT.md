# PyTorch `.pth` format notes

These notes record the subset reverse-engineered for PTHrs 0.1.0.

## Container

Modern PyTorch checkpoints begin with a ZIP local-file header (`PK\x03\x04`).
All relevant paths normally share one generated prefix:

```text
<prefix>/data.pkl
<prefix>/byteorder
<prefix>/version
<prefix>/data/0
<prefix>/data/1
...
```

`data.pkl` is small relative to the model. Each `data/<id>` entry is a raw
typed storage byte array. PyTorch normally stores these entries without ZIP
compression and adds alignment padding to the local-header extra field.

## Pickle connection

Tensor records use `torch._utils._rebuild_tensor_v2` with arguments equivalent
to:

```text
(storage, storage_offset, shape, stride, requires_grad, backward_hooks)
```

The `storage` value is supplied by the pickle `BINPERSID` opcode. Its persistent
ID is normally:

```text
("storage", torch.<DType>Storage, "<id>", "<device>", element_count)
```

PTHrs maps `<id>` to `<prefix>/data/<id>`. The device string is retained but
does not affect reading.

## Tensor views

The storage offset and stride are measured in elements, not bytes. For a tensor
coordinate `i`, the storage element is:

```text
storage_offset + sum(i[axis] * stride[axis])
```

`read_tensor` returns contiguous row-major bytes. If the saved view is already
contiguous, PTHrs performs one bounded range read. Otherwise it reads the
smallest enclosing storage span and gathers elements in logical order.

## Observed voice-conversion checkpoint fields

Common inference checkpoints contain:

- `weight`: state dictionary from parameter names to tensors
- `config`: positional network configuration
- `info`: training checkpoint label or epoch
- `sr`: sampling-rate label such as `40k`
- `f0`: whether pitch guidance is enabled
- `version`: application architecture version such as `v1` or `v2`

These remain generic `Value` data so callers can handle future variations.
