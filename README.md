<div align="center">
    <img src="public/logo-with-text.png" alt="oh Logo" width="348" height="230" />
    <h1 align="center">oh</h1>
    <p align="center">Self-hosted AI detector for text and pdf without GPU</p>
    <br />
</div>

https://github.com/user-attachments/assets/e64ad987-3139-4928-bcc2-1b5b2f1f2e6e

_(This was obviously a perfectly chosen example)_

# oh ...

is the sound you make once you find out your paper was flagged.

## Features

- Automated install using Docker
- Choose between three AI detection models
- Analyze text and PDF files: paste, drop a file on the box, or add a PDF
- Your text stays in the browser between visits, and never leaves the machine
- Runs entirely on your CPU, no GPU and no accounts


## 🚀 Deploy oh for yourself

### Single run command

```bash
docker run -d --restart=unless-stopped -p 3000:3000 -v oh-data:/data --name oh timwitzdam/oh:latest
```

### Docker compose

```yaml
services:
  oh:
    image: timwitzdam/oh:latest
    container_name: oh
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - oh-data:/data
    environment:
      # Lower this on small hosts
      OMP_NUM_THREADS: '4'

volumes:
  oh-data:
    driver: local
```

## Hardware Requirements

Yes, this might even be an important section to read because of the usage of our highly scarce RAM.

### Per tier

Every number below is measured, not estimated. All figures come from one
machine: an **AMD Ryzen 7 7840U (8 cores / 16 threads)**, 32 GB RAM, Linux, with the container's default `OMP_NUM_THREADS=4`. 

|  | Lite | Balanced | Deep |
| --- | --- | --- | --- |
| Runs on | onnxruntime, in the app process | torch, in the Python service | torch, in the Python service |
| Model | [`onnx-community/chatgpt-detector-roberta-ONNX`](https://huggingface.co/onnx-community/chatgpt-detector-roberta-ONNX) | [`rasbt/ai-text-detector-qwen3-0.6b-variable`](https://huggingface.co/rasbt/ai-text-detector-qwen3-0.6b-variable) | [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01) |
| Licence | Apache-2.0 | Apache-2.0 | MIT |
| Download | 125 MB | 1.1 GB | 1.7 GB |
| **RAM** | **0.8 GB** | **2.7 GB** | **2.2 GB** |
| Speed | ~3,900 words/s | ~195 words/s | ~172 words/s |
| A 12,000-word document | ~3 s | ~62 s | ~70 s |
| Cores used | all of them | `OMP_NUM_THREADS` | `OMP_NUM_THREADS` |


### Tuning cores

`OMP_NUM_THREADS` defines the maximum number of threads that should be used.

| `OMP_NUM_THREADS` | Lite | Balanced | Deep |
| --- | --- | --- | --- |
| 1 | 0.8 s | 33.0 s | 34.7 s |
| 2 | 0.6 s | 19.7 s | 20.7 s |
| 4 (default) | 0.8 s | 13.0 s | 13.6 s |
| 8 | 0.8 s | 7.6 s | 8.1 s |

## Models

Three tiers, all CPU-only, all downloaded from Hugging Face on first use and
cached in the data volume. Nothing is vendored in the image.

| Tier | Model | Size | Best at |
| --- | --- | --- | --- |
| Lite | RoBERTa-large, int8 quantised (2023) | 125 MB | Being fast and cheap. The least accurate on writing it has not seen. |
| Balanced | Qwen3-0.6B fine-tuned, reads out at the last token | 1.1 GB | A different kind of checker rather than a sharper one. Holds the most RAM of the three. |
| Deep | DeBERTa-v3-large trained on RAID | 1.7 GB | Accuracy. Clearly the best of the three on essays, academic prose and reviews, and it stays quiet on human text. |



## Development

Requires Node 22 and Python 3.11 or newer.

```bash
npm install
npm run dev            # http://localhost:3000
```

The Balanced and Deep tiers need the Python inference service as well; the Lite
tier does not:

```bash
python3 -m venv service/.venv
# torch first, from the CPU wheel index. The default index resolves 16 CUDA
# packages alongside torch - 2.6 GB of wheels for libraries this image never
# calls - because nothing tells pip you are on a CPU.
service/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
service/.venv/bin/pip install -r service/requirements.txt
PYTHONPATH=service OH_DATA_DIR=./data service/.venv/bin/python -m uvicorn \
  oh_service.main:app --host 127.0.0.1 --port 8001
```

Useful scripts:

```bash
npm run typecheck
npm run lint
npm run eval:detectors                       # ONNX candidates through transformers.js
node scripts/eval-markdown.mjs               # how much markdown formatting moves a score
node scripts/sync-catalog-sizes.mjs          # refresh published file sizes from the hub
python3 scripts/fetch-eval-set.py            # build the labelled sets under eval/

# need a torch install, so run these with the venv's interpreter:
service/.venv/bin/python scripts/eval_torch.py <name>    # torch detector candidates
service/.venv/bin/python scripts/compare-deep.py         # the two torch families head to head
service/.venv/bin/python scripts/eval_textsight.py       # scoring modes for one checkpoint
```

The eval scripts download their own candidate models from Hugging Face, so they are not limited to the three shipped tiers. `eval/` is gitignored: the passages
belong to other people's datasets. So feel free to check out different or new models, and if you find one that works well, please open an issue or PR so it can be added to the tool.

`OH_DEBUG_WINDOWS=1` logs the windows the pipeline builds, which is the first
thing to look at when a score looks wrong.


## Any questions, suggestions or problems?

You're welcome to contribute to oh or open an issue if you have any suggestions or find any problems.

I'm also available via mail: [contact@witzdam.com](mailto:contact@witzdam.com)


## Licences

Application code: MIT, see [LICENSE](LICENSE). Bundled font: Inter, SIL Open Font License 1.1 (`src/app/fonts`, licence text in `src/app/fonts/OFL.txt`). Models keep their own licences and are downloaded from Hugging Face rather than vendored: Apache-2.0 for the Lite and Balanced repos, MIT for the Deep one. Each card in the picker names the repository it came from, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) has the full list.
