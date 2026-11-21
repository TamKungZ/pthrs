# PTHrs

[![crates.io](https://img.shields.io/crates/v/pthrs)](https://crates.io/crates/pthrs)
![Zero dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)

Pure-Rust readers for PyTorch `.pth` checkpoints and FAISS
`IndexIVFFlat` files. No dependencies, and no Python, PyTorch, or FAISS needed.

## Checkpoints

Reads `torch.save` archives (ZIP and ZIP64 with stored or deflated entries,
plus the older single-stream layout) without executing any Python code.
Tensors load lazily or in bulk, as raw bytes or `f32`, with contiguous and
strided views. Voice-model configs can be read and validated.

```rust
use pthrs::{PthArchive, Result, TensorReadBuffer};

fn main() -> Result<()> {
    let mut archive = PthArchive::open("model.pth")?;
    let model = archive.checkpoint().voice_model_info()?;
    let report = model.validate(archive.checkpoint());
    assert!(report.is_valid(), "{:?}", report.errors);

    let mut buffer = TensorReadBuffer::new();
    let tensor = archive.read_tensor_into("enc_p.emb_phone.weight", &mut buffer)?;
    println!("{:?} {:?}", tensor.meta.dtype, tensor.meta.shape);
    Ok(())
}
```

`PthArchive::open_voice_model` opens and validates in one call, and
`ArchitectureVersion` gives the v1 or v2 feature width. Use `load_all_tensors`
for raw bytes, `load_all_f32` for `f32` weights, or `load_floating_f32` to
skip integer entries such as `num_batches_tracked`. `F32Tensor::into_shape_and_values`
yields a `usize` shape and values for building tensors. Reuse a
`TensorReadBuffer` with `read_tensor_into` or `read_tensor_f32_into` to avoid
repeated allocation.

## Retrieval indexes

Reads FAISS `IndexIVFFlat` with L2 or inner-product metrics. An index can be
searched lazily from disk or loaded into memory, and supports vector
reconstruction and inverse-squared-distance blending.

```rust
use pthrs::{FaissIvfFlatIndex, Result, SearchOptions};

fn main() -> Result<()> {
    let index = FaissIvfFlatIndex::open("model.index")?.load()?;
    let mut workspace = index.workspace(8);
    let query = vec![0.0; index.dimension()];
    let mut output = vec![0.0; index.dimension()];

    index.search_and_blend(
        &query,
        &mut output,
        SearchOptions { k: 8, nprobe: 1 },
        0.75,
        &mut workspace,
    )?;
    Ok(())
}
```

`blend_frames` applies the blend to every row-major frame in a buffer. Create
the index and workspace once at startup. Search does not allocate while
`k` fits the workspace. `LoadedIvfFlatIndex` is `Send + Sync`, so share one
index and give each thread its own workspace.

## CLI

```bash
pthrs-inspect model.pth --validate
pthrs-inspect model.pth --tensor enc_p.emb_phone.weight --values 8
pthrs-index-inspect model.index --query-id 0 --k 8 --nprobe 1
```

## Limits

- ZIP entries compressed with anything other than deflate are not supported.
- Legacy single-stream files with storage views are not supported.
- Sparse and quantized tensors are not supported.
- Only little-endian 64-bit FAISS `IndexIVFFlat` files are read.

Format notes are in [docs/FORMAT.md](docs/FORMAT.md) and
[docs/FAISS_INDEX.md](docs/FAISS_INDEX.md). Tested files are listed in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## License

Apache-2.0