# FAISS IVF-Flat index format

PTHrs supports the FAISS `IndexIVFFlat` layout used by the supplied voice
indexes. FAISS serializes class fields in order and identifies concrete types
with four-byte codes.

The format is native C++ serialization rather than a stable interchange
standard. The supported files are little-endian, 64-bit builds where `size_t`
and `idx_t` are eight bytes.

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

The supplied indexes use type 0.

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

The vector and ID regions are recorded as offsets by the lazy reader. The
loaded reader decodes them into memory and builds an ID-to-vector lookup.

## Search

1. Score the query against all centroids.
2. Select the best `nprobe` lists.
3. Score vectors inside those lists.
4. Keep the best `k` results.

L2 results contain squared Euclidean distance, matching FAISS. Inner-product
results contain similarity, where larger values are better.

`LoadedIvfFlatIndex::search_and_blend` applies inverse-squared L2 weighting:

```text
weight[i] = (1 / max(distance[i], epsilon))²
weight    = weight / sum(weight)
output    = retrieved * rate + query * (1 - rate)
```

## Verified samples

See [COMPATIBILITY.md](COMPATIBILITY.md) for the tested index matrix, fixture
sources, hashes, and search checks.

## Upstream implementation

- [`faiss/impl/index_write.cpp`](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_write.cpp)
- [`faiss/impl/index_read.cpp`](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp)
- [`faiss/impl/io_macros.h`](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/io_macros.h)
