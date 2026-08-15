# Changelog

All notable changes to PTHrs will be documented in this file.

## 0.2.0 - 2026-08-15

- Add a pure-Rust FAISS `IndexIVFFlat` reader.
- Add lazy and loaded nearest-neighbor search with reusable workspaces.
- Add vector reconstruction and inverse-squared retrieval blending.
- Add named voice-model configuration and checkpoint/index validation.
- Add reusable tensor read buffers and bulk raw/F32 loading APIs.
- Detect `weight`, `state_dict`, `model`, and root state dictionaries.
- Add checkpoint summaries and index inspection tooling.
- Verify PTH and index handling against Miku and Solar V3 model bundles.

## 0.1.0 - 2026-08-15

- Add a zero-dependency ZIP/ZIP64 PyTorch checkpoint reader.
- Add a safe pickle virtual machine for checkpoint metadata.
- Recognize PyTorch storage and tensor reconstruction records.
- Add lazy reads for contiguous and strided tensor data.
- Add real-valued tensor conversion to `f32`.
- Add the `pthrs-inspect` command-line tool.
