# Changelog

All notable changes to PTHrs will be documented in this file.

## 1.0.0 - 2026-11-21

- Split the FAISS, checkpoint, pickle, and tensor modules into focused files.
- Add `TensorMeta::shape_usize`, `DType::is_floating_point`, and
  `Checkpoint::tensors_with_dtype`.
- Add `PthArchive::open_voice_model`, `VoiceModelInfo::architecture`,
  `ArchitectureVersion`, and `ValidationReport::into_result`.
- Add `PthArchive::load_f32_where`, `PthArchive::load_floating_f32`, and
  `F32Tensor::into_shape_and_values`.
- Add `LoadedIvfFlatIndex::blend_frames` for whole-buffer retrieval blending.
- Make `Error`, `Value`, `DType`, and `Metric` non-exhaustive and add
  `Error::Allocation`. This is a breaking change for exhaustive matches.
- Report an allocation failure instead of aborting when a tensor's logical size
  cannot be allocated, such as a hostile stride-0 view.
- Limit pickle nesting depth, duplicated data, and map size, so a crafted
  checkpoint can no longer exhaust memory or overflow the stack.
- Reject an index whose centroid block is larger than the file before
  allocating it.
- Blend the vector that was actually found, so repeated IDs no longer pick the
  first stored vector.
- Order equal-distance neighbors by ID, making search results deterministic.
- Load index lists in blocks and select probed lists without a full sort.
- Accept tuples in voice-model config values and integer `sr` labels.
- Fill `OrderedDict` from an item list and decode `\x` escapes in protocol 0
  strings.
- Add `Value::as_sequence`.
- Drop the redundant `[lib]`, `[[bin]]`, and `[features]` tables.
- Read ZIP entries compressed with deflate through a built-in decoder.
- Read the older single-stream `.pth` layout.
- Add `DType::U16`, `DType::U32`, and `DType::U64`.
- Read pickles that contain `INST` and `OBJ` class instances as inert values.

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