# Changelog

All notable changes to PTHrs will be documented in this file.

## 0.1.0 - 2026-08-15

- Add a zero-dependency ZIP/ZIP64 PyTorch checkpoint reader.
- Add a safe pickle virtual machine for checkpoint metadata.
- Recognize PyTorch storage and tensor reconstruction records.
- Add lazy reads for contiguous and strided tensor data.
- Add real-valued tensor conversion to `f32`.
- Add the `pthrs-inspect` command-line tool.
