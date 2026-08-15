# Compatibility tests

This page records real files tested with PTHrs 0.2.0. Model files are test
fixtures only and are not distributed with the crate.

Tests were run on 2026-08-15.

## Test coverage

Each inference checkpoint was tested for:

- archive, pickle, metadata, storage, and tensor parsing
- named voice configuration and model validation
- decoding every tensor to `f32`
- non-finite value detection across all decoded tensors
- checkpoint and retrieval-index dimension agreement

Retrieval indexes were tested for parsing, loading, reconstruction, lazy and
in-memory search, and retrieval blending. A stored vector was used as a query
and had to return its own ID with distance 0. Exhaustive `nprobe = nlist`
search was also run on Miku, Athena, TITAN, Ayaka, and Wanderer.

## Inference checkpoints

| Fixture | Source | Version | Rate | F0 | Tensors | Phone channels | Result |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| Miku | Local fixture supplied by the project author; original source unknown | v2 | 40 kHz | yes | 457 | 768 | pass |
| Solar V3 | Local fixture supplied by the project author; original source unknown | v2 | 40 kHz | yes | 457 | 768 | pass |
| Athena Asamiya | Local fixture supplied by the project author; original source unknown | v2 | 40 kHz | yes | 457 | 768 | pass |
| TITAN Medium 32k | [Hugging Face](https://huggingface.co/blaise-tk/TITAN/tree/15040355c572262b405484e720f6bc613784be3e/models/medium/32k/model) | v2 | 32 kHz | yes | 457 | 768 | pass |
| Ayaka JP | [Hugging Face](https://huggingface.co/Harikomaster/rvc-genshin-impact/blob/3ba669bbb058a1a5cbeeea8bb320dcfdc5f3001b/prezipped/v1/ayaka-jp%20100%20epochs%2040k.zip) | v1 | 40 kHz | yes | 457 | 256 | pass |
| Wanderer JP | [Hugging Face](https://huggingface.co/Harikomaster/rvc-genshin-impact/blob/3ba669bbb058a1a5cbeeea8bb320dcfdc5f3001b/prezipped/v1/wanderer-jp%20100%20epochs%2048k.zip) | v1 | 48 kHz | yes | 516 | 256 | pass |

The v1 files do not contain a `version` field. PTHrs correctly applies the RVC
v1 default and validates their 256-channel phone embeddings.

No decoded tensor in this matrix contained NaN or infinity.

## Retrieval indexes

| Fixture | Dimension | Vectors | Lists | Metric | Layout | Result |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Miku | 768 | 10,170 | 260 | L2 | `ilar/full` | pass |
| Solar V3 | 768 | 48,266 | 1,237 | L2 | `ilar/full` | pass |
| Athena Asamiya | 768 | 43,752 | 1,121 | L2 | `ilar/full` | pass |
| TITAN Medium 32k | 768 | 10,000 | 256 | L2 | `ilar/full` | pass |
| Ayaka JP | 256 | 85,895 | 2,202 | L2 | `ilar/full` | pass |
| Wanderer JP | 256 | 71,988 | 1,845 | L2 | `ilar/full` | pass |

## Training checkpoint

`pretrained/v1/48k/G48k.pth` from
[`Politrees/RVC_resources`](https://huggingface.co/Politrees/RVC_resources/tree/0a6a139743218159776d021418d96dd57c128c8a/pretrained/v1/48k)
was tested as a non-F0 training checkpoint. PTHrs parsed and decoded all 606
tensors and 36,323,904 elements without non-finite values.

This file has training metadata and no exported voice-model `config` field.
`voice_model_info()` rejecting it is expected; it is not an inference model.

## Fixture hashes

```text
fb95ad927c03766119ad6a74a5ba59651ef7a479f6ebf5ea0db78094a04e23a0  Miku/model.pth
c831e82a8d3151de96aa04746ce1d1dfea29f545a0f5b59452839bb71b5b9e92  Miku/model.index
4bd859043c930f0f470b3a7e045c0656219a14f770de31b2518594ecd03a30d7  Solar_V3.pth
d52ca9d57d910b45826d755b4742234d8bc19aa45b3217f3dc27d200db965424  Solar_V3.index
5b077da5ad3bb0081b307e0e70755718625725050f3aab796c34de9f3f0ccf07  Athena/model.pth
69bfa66cd29c957fffc18eeb420dde07aaadaa03a7ef4971712d6adc30e40242  Athena/model.index
dbef42caee65d3bb2b290d0faedb2dcb093e9a02b0abe00633973ab88c6d80a4  f032k-Titan-Medium.pth
b45170a4ee93d09870be238b8fb04878a9d69e8ef54bb458a63b3e3f92caf559  TITAN-32k.index
bd3926782610dedcc627bc0ac071fc8e53b9bbb23df47240b06e3189eeb5e1df  ayaka-v1-40k.zip
668d29f9ee8a6bd001e7015cffc5064ad8a266f3622a90c50773cd6044c03425  wanderer-v1-48k.zip
3862a67ea6313e8ffefc05cee6bee656ef3e089442e9ecf4a6618d60721f3e95  G48k.pth
```

## Rejected fixture

The pinned Fischl v1/48k archive from the Genshin repository was not used. Its
downloaded SHA-256 matched the repository object
(`3ddf310c1bab847864ee643a57ab068c9d04d5ca811f806da69fa5055e4593ad`),
but the ZIP had no end-of-central-directory record and its `.pth` payload ended
inside a deflate stream. The index entry was recoverable; the checkpoint was
not. This is a damaged upstream fixture, not a PTHrs compatibility failure.
