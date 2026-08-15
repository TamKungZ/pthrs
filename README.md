# PTHrs

A zero-dependency Rust library for reading ZIP-based PyTorch `.pth`
checkpoint files.

PTHrs parses checkpoint metadata and exposes tensor storage without requiring
Python, PyTorch, libtorch, or CUDA. Tensor data is loaded only when requested.

## Features

- ZIP and ZIP64 PyTorch checkpoints
- Safe pickle parser with no Python code execution
- Lazy tensor and storage reads
- Tensor dtype, shape, stride, storage offset, and device metadata
- Contiguous and strided tensor views
- `f16`, `bf16`, `f32`, `f64`, integer, boolean, and complex storage types
- Conversion of real-valued tensors to `Vec<f32>`
- Safe Rust with no external dependencies

## Usage

```rust
use pthrs::{PthArchive, Result};

fn main() -> Result<()> {
    let mut checkpoint = PthArchive::open("model.pth")?;

    println!(
        "tensors: {}",
        checkpoint.checkpoint().tensor_count()
    );

    if let Some(meta) = checkpoint
        .checkpoint()
        .tensor("enc_p.emb_phone.weight")
    {
        println!("{:?} {:?}", meta.dtype, meta.shape);
    }

    let tensor = checkpoint.read_tensor("enc_p.emb_phone.weight")?;
    let values = tensor.to_f32_vec()?;

    println!("{:?}", &values[..values.len().min(8)]);
    Ok(())
}
```

`PthArchive::from_reader` accepts any `Read + Seek` source:

```rust
use std::io::Cursor;
use pthrs::PthArchive;

let bytes = std::fs::read("model.pth")?;
let checkpoint = PthArchive::from_reader(Cursor::new(bytes))?;

println!("{}", checkpoint.checkpoint().tensor_count());
# Ok::<(), pthrs::Error>(())
```

## CLI

```bash
cargo run --release --bin pthrs-inspect -- model.pth
cargo run --release --bin pthrs-inspect -- model.pth --list
cargo run --release --bin pthrs-inspect -- model.pth \
  --tensor enc_p.emb_phone.weight \
  --values 8
```

## Supported checkpoint data

- `data.pkl`
- `byteorder`
- `version`
- `data/<storage-id>`
- `torch._utils._rebuild_tensor`
- `torch._utils._rebuild_tensor_v2`
- `torch._utils._rebuild_parameter`

Compressed ZIP entries, legacy pre-ZIP checkpoints, sparse tensors, quantized
tensors, and FAISS `.index` files are not currently supported.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

## License

Apache-2.0

Copyright 2026 TamKungZ_ <dev@tamkungz.me>