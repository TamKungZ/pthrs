# PTHrs

Pure-Rust readers for ZIP-based PyTorch `.pth` checkpoints and FAISS
`IndexIVFFlat` files.

The crate has no external dependencies and does not require Python, PyTorch,
libtorch, FAISS, or CUDA.

## Checkpoints

- ZIP and ZIP64 `torch.save` archives
- Safe pickle parser with no Python code execution
- Lazy and bulk tensor loading
- Caller-owned reusable read buffers
- Tensor dtype, shape, stride, offset, and storage metadata
- Contiguous and strided tensor views
- Raw, `f16`, `bf16`, `f32`, `f64`, integer, boolean, and complex data
- Named voice-model config and validation

```rust
use pthrs::{PthArchive, Result, TensorReadBuffer};

fn main() -> Result<()> {
    let mut checkpoint = PthArchive::open("model.pth")?;
    let model = checkpoint.checkpoint().voice_model_info()?;
    let report = model.validate(checkpoint.checkpoint());

    assert!(report.is_valid(), "{:?}", report.errors);

    let mut buffer = TensorReadBuffer::new();
    let tensor = checkpoint.read_tensor_into(
        "enc_p.emb_phone.weight",
        &mut buffer,
    )?;

    println!("{:?} {:?}", tensor.meta.dtype, tensor.meta.shape);
    Ok(())
}
```

Use `load_all_tensors` for original tensor bytes or `load_all_f32` for
backend-ready `f32` values.

## Retrieval indexes

- FAISS `IndexIVFFlat`
- L2 and inner-product metrics
- `IndexFlat` quantizers
- Full and sparse `ArrayInvertedLists`
- Lazy file-backed search
- Fully loaded low-latency search
- ID reconstruction
- Reusable search workspaces
- Inverse-squared-distance retrieval blending

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

Create the loaded index and workspace once during startup. Loaded search does
not allocate while the requested `k` fits the workspace capacity.

## CLI

```bash
pthrs-inspect model.pth --validate
pthrs-inspect model.pth --list
pthrs-inspect model.pth --tensor enc_p.emb_phone.weight --values 8

pthrs-index-inspect model.index
pthrs-index-inspect model.index --query-id 0 --k 8 --nprobe 1
```

## Current limits

- PyTorch legacy pre-ZIP serialization is not supported.
- Compressed checkpoint entries are not supported.
- Sparse and quantized PyTorch tensors are not supported.
- FAISS support currently targets little-endian 64-bit `IndexIVFFlat` files.
- Other FAISS index classes are rejected.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

See [docs/FORMAT.md](docs/FORMAT.md),
[docs/FAISS_INDEX.md](docs/FAISS_INDEX.md),
[docs/RUNTIME.md](docs/RUNTIME.md), and
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## License

Apache-2.0

Copyright 2026 TamKungZ_ <dev@tamkungz.me>
