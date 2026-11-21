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
| Miku | local | v2 | 40 kHz | 457 | 10,170 | 260 |
| Solar V3 | local | v2 | 40 kHz | 457 | 48,266 | 1,237 |
| Athena Asamiya | local | v2 | 40 kHz | 457 | 43,752 | 1,121 |
| TITAN Medium 32k | [Hugging Face](https://huggingface.co/blaise-tk/TITAN/tree/15040355c572262b405484e720f6bc613784be3e/models/medium/32k/model) | v2 | 32 kHz | 457 | 10,000 | 256 |
| Ayaka JP | [Hugging Face](https://huggingface.co/Harikomaster/rvc-genshin-impact/blob/3ba669bbb058a1a5cbeeea8bb320dcfdc5f3001b/prezipped/v1/ayaka-jp%20100%20epochs%2040k.zip) | v1 | 40 kHz | 457 | 85,895 | 2,202 |
| Wanderer JP | [Hugging Face](https://huggingface.co/Harikomaster/rvc-genshin-impact/blob/3ba669bbb058a1a5cbeeea8bb320dcfdc5f3001b/prezipped/v1/wanderer-jp%20100%20epochs%2048k.zip) | v1 | 48 kHz | 516 | 71,988 | 1,845 |

All use F0 and L2 `ilar/full` indexes. v2 uses 768-channel phone embeddings and
v1 uses 256. The v1 files have no `version` field, so the RVC v1 default
applies.

## Training checkpoint

`pretrained/v1/48k/G48k.pth` from
[`Politrees/RVC_resources`](https://huggingface.co/Politrees/RVC_resources/tree/0a6a139743218159776d021418d96dd57c128c8a/pretrained/v1/48k)
loads all 606 tensors. It has no exported `config` field, so
`voice_model_info()` rejects it as expected.
