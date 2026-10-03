# Third-party notices

The application code in this repository is MIT licensed; see [LICENSE](LICENSE).
The items below keep their own licences.

## Bundled fonts

`src/app/fonts` ships one font binary with the app:

| File | Font | Copyright | Licence |
| --- | --- | --- | --- |
| `InterVariable.woff2` | Inter | Copyright (c) 2016 The Inter Project Authors | SIL Open Font License 1.1 |

It is redistributed unmodified, so its Reserved Font Name is intact. The full
OFL 1.1 text is in [`src/app/fonts/OFL.txt`](src/app/fonts/OFL.txt) and ships
alongside the binary as the licence requires.

## Detection models

Model weights are **not** vendored in this repository or in the container
image. They are downloaded from Hugging Face on first use and cached in the
data volume, so each model is licensed by its own repository:

| Tier | Hugging Face repo | Licence |
| --- | --- | --- |
| Lite | [`onnx-community/chatgpt-detector-roberta-ONNX`](https://huggingface.co/onnx-community/chatgpt-detector-roberta-ONNX) | Apache-2.0 |
| Balanced | [`rasbt/ai-text-detector-qwen3-0.6b-variable`](https://huggingface.co/rasbt/ai-text-detector-qwen3-0.6b-variable) | Apache-2.0 |
| Deep | [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01) | MIT |

Apache-2.0 requires redistribution to carry the licence and any NOTICE file
the upstream repo ships. If you redistribute a Docker image that bundles
weights rather than downloading them at runtime, copy those files from the
upstream repos into the image alongside the weights.

The authoritative list is `src/lib/catalog.ts`, and each card in the model
picker names the repo a weight came from.

## npm dependencies

Runtime and build dependencies keep their own licences; see
`package-lock.json` and each package's `LICENSE` file in `node_modules`.