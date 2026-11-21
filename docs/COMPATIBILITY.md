# Compatibility

Real files tested during development. Model files are fixtures only and are
not distributed with the crate.

Each inference checkpoint was parsed, validated, and fully decoded to `f32`
with no NaN or infinity, and its dimension was checked against its index.
Each index was parsed, loaded, and searched lazily and in memory, with
reconstruction and blending. A stored vector used as a query returned its own
ID at distance 0, and `nprobe = nlist` was also run on the larger indexes.

## Inference checkpoints and indexes

| Fixture | Source | Version | Rate | Tensors | Index vectors | Index lists |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Miku | Local, source unknown | v2 | 40 kHz | 457 | 10,170 | 260 |
| Solar V3 | local | v2 | 40 kHz | 457 | 48,266 | 1,237 |
| Athena Asamiya | Local, source unknown | v2 | 40 kHz | 457 | 43,752 | 1,121 |
| TITAN Medium 32k | [Hugging Face](https://huggingface.co/blaise-tk/TITAN/tree/15040355c572262b405484e720f6bc613784be3e/models/medium/32k/model) | v2 | 32 kHz | 457 | 10,000 | 256 |
| Ayaka JP | [Hugging Face](https://huggingface.co/Harikomaster/rvc-genshin-impact/blob/3ba669bbb058a1a5cbeeea8bb320dcfdc5f3001b/prezipped/v1/ayaka-jp%20100%20epochs%2040k.zip) | v1 | 40 kHz | 457 | 85,895 | 2,202 |
| Wanderer JP | [Hugging Face](https://huggingface.co/Harikomaster/rvc-genshin-impact/blob/3ba669bbb058a1a5cbeeea8bb320dcfdc5f3001b/prezipped/v1/wanderer-jp%20100%20epochs%2048k.zip) | v1 | 48 kHz | 516 | 71,988 | 1,845 |
| Diona JP | [Hugging Face](https://huggingface.co/ArkanDash/rvc-genshin-impact/blob/f120a90b21dc31a3f14ed64732b96aea755a96d6/prezipped/v2/diona-jp%20105%20epochs%2048k%20v2.zip) | v2 | 48 kHz | 457 | 167,439 | 4,293 |
| Hatsune Miku default | [Hugging Face](https://huggingface.co/aple/HatsuneMikuRVC/blob/781194746498a5f997a7135541531a915358fdcd/miku_default_rvc.zip) | v2 | 40 kHz | 457 | 173,824 | 4,457 |

All use F0 and L2 `ilar/full` indexes. v2 uses 768-channel phone embeddings and
v1 uses 256. The v1 files have no `version` field, so the RVC v1 default
applies.

## Training checkpoint

`pretrained/v1/48k/G48k.pth` from
[`Politrees/RVC_resources`](https://huggingface.co/Politrees/RVC_resources/tree/0a6a139743218159776d021418d96dd57c128c8a/pretrained/v1/48k)
loads all 606 tensors. It has no exported `config` field, so
`voice_model_info()` rejects it as expected.

## Other checkpoints

Loaded and decoded to `f32` with no parse errors:

| File | Tensors | Elements | Type |
| --- | ---: | ---: | --- |
| fairseq `hubert_base.pt` | 214 | 94,697,600 | F16 |
| `rmvpe_20231006.pt` | 741 | 90,472,917 | F16 |
| `fcpe_c_v001.pt` | 73 | 10,832,953 | F32 |

The `cfg` object in `hubert_base.pt` holds an `argparse.Namespace`, which is
kept as an inert value. In `rmvpe_20231006.pt` the `num_batches_tracked`
counters are stored as F16 and decode to infinity, so consumers should still
skip them by name.

## Coverage

Real files above cover stored ZIP checkpoints, v1 and v2 RVC models, and
full-layout L2 indexes. A real model recompressed with deflate decoded to the
same tensors as the original.

These paths are tested only with generated fixtures: ZIP64, big-endian data,
the legacy single-stream layout, sparse list layout, inner-product indexes,
direct maps, strided tensors, and integer and wide dtypes. No real file of
those kinds has been available.