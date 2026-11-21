# FAISS IVF-Flat index format

FAISS serializes class fields in order and identifies concrete types with
four-byte codes.

The format is native C++ serialization. Supported files are little-endian
64-bit builds.

## `IndexIVFFlat`

```text
u8[4]  "IwFl"
IndexHeader
u64    nlist
u64    nprobe
Index  quantizer
DirectMap
ArrayInvertedLists
```

### `IndexHeader`

```text
i32    dimension
i64    ntotal
i64    legacy_dummy_1
i64    legacy_dummy_2
u8     is_trained
i32    metric_type       # 0 = inner product, 1 = squared L2
f32    metric_argument   # present only for metric types greater than 1
```

PTHrs currently accepts L2 and inner-product indexes.

## Flat quantizer

The IVF quantizer is normally `IndexFlatL2`:

```text
u8[4]  "IxF2"           # "IxFI" for inner product
IndexHeader
u64    float_count
f32[]  centroids
```

`float_count` must equal `nlist * dimension`.

## Direct map

```text
u8     type              # 0 = none, 1 = array, 2 = hash table
u64    array_length
i64[]  array

# Only when type == 2
u64    pair_count
(i64, i64)[] pairs
```

## Array inverted lists

```text
u8[4]  "ilar"
u64    nlist
u64    code_size         # dimension * sizeof(f32)
u8[4]  layout            # "full" or "sprs"
```

For `full`, the list-size table is:

```text
u64    size_count        # must equal nlist
u64[]  list_sizes
```

For `sprs`, it contains `(list_index, list_size)` pairs:

```text
u64    value_count       # even
u64[]  index_size_pairs
```

List payloads follow in list order. Empty lists have no payload.

```text
for each list:
    f32[list_size * dimension] vectors
    i64[list_size]             ids
```

## Search

The query is scored against all centroids, the best `nprobe` lists are scanned,
and the best `k` vectors are kept. L2 distances are squared, matching FAISS;
inner-product results are similarities where larger is better.

`LoadedIvfFlatIndex::search_and_blend` weights neighbors by inverse squared
distance:

```text
weight[i] = (1 / max(distance[i], epsilon))^2
weight    = weight / sum(weight)
output    = retrieved * rate + query * (1 - rate)
```

## Upstream

- [`faiss/impl/index_write.cpp`](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_write.cpp)
- [`faiss/impl/index_read.cpp`](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp)
