# Runtime integration

## Startup

```rust
use pthrs::{FaissIvfFlatIndex, PthArchive, Result};

fn load(pth: &str, index: &str) -> Result<()> {
    let checkpoint = PthArchive::open(pth)?;
    let model = checkpoint.checkpoint().voice_model_info()?;
    let validation = model.validate(checkpoint.checkpoint());
    assert!(validation.is_valid());

    let index = FaissIvfFlatIndex::open(index)?.load()?;
    model.validate_index_dimension(index.dimension())?;
    Ok(())
}
```

Use `PthArchive::load_all_tensors` when a backend accepts original tensor
bytes, or `PthArchive::load_all_f32` when it requires `f32` weights.

For incremental backend loading, reuse a `TensorReadBuffer` with
`read_tensor_into` or `read_tensor_f32_into`.

## Retrieval hot path

```rust
use pthrs::{LoadedIvfFlatIndex, Result, SearchOptions, SearchWorkspace};

fn process(
    index: &LoadedIvfFlatIndex,
    features: &[f32],
    output: &mut [f32],
    workspace: &mut SearchWorkspace,
) -> Result<()> {
    index.search_and_blend(
        features,
        output,
        SearchOptions { k: 8, nprobe: 1 },
        0.75,
        workspace,
    )?;
    Ok(())
}
```

Create the workspace once with `index.workspace(max_k)`. Loaded-index search
does not perform heap allocation while `k <= max_k`.

`LoadedIvfFlatIndex` is `Send + Sync`; use one shared index and one workspace
per processing thread.

