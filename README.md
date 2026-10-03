<div align="center">
    <img src="https://i.imgur.com/LIcWf9r.png" alt="oh Logo" />
    <h1 align="center">oh</a></h1>
    <p align="center">Self-hosted AI detector for text and pdf without GPU</p>
    <br />
</div>

![GitSave 3 pages animation](https://i.imgur.com/i0SNNiL.gif)

https://github.com/user-attachments/assets/301b28ca-6b72-490a-8efb-217e39fb73d3

# oh ...

is the sound you make once you find out your paper was flagged.

## Features

- Automated install using Docker
- Choose between three AI detection models
- Analyze text and PDF files

## Hardware Requirements

Yes, this might even be an important section to read because of the usage of our highly scarce RAM.

## 🚀 Deploy oh for yourself

### Single run command

```bash
docker run -d --restart=unless-stopped -p 3000:3000 -v oh-data:/data --name oh timwitzdam/oh:latest
```

### Docker compose

1. Create `docker-compose.yml` file

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

volumes:
  oh-data:
    driver: local
```

## Any questions, suggestions or problems?

You're welcome to contribute to oh or open an issue if you have any suggestions or find any problems.

I'm also available via mail: [contact@witzdam.com](mailto:contact@witzdam.com)


## Development

```bash
npm install
npm run dev            # http://localhost:3000
```

The Deep tier needs the Python service as well:

```bash
python -m venv service/.venv
service/.venv/bin/pip install -r service/requirements.txt
PYTHONPATH=service OH_DATA_DIR=./data service/.venv/bin/python -m uvicorn \
  oh_service.main:app --host 127.0.0.1 --port 8001
```

`OH_DATA_DIR` defaults to `./data` locally and `/data` in the container.

Useful scripts:

```bash
npm run typecheck
npm run lint
npm run eval:detectors                       # ONNX encoder candidates through the app's path
node scripts/eval_torch.py <name>            # torch candidates, needs a venv with torch
node scripts/sync-catalog-sizes.mjs          # refresh published file sizes from the hub
```

`OH_DEBUG_WINDOWS=1` logs the windows the pipeline builds, which is the first thing to look at when a score looks wrong.


## Licences

Application code: MIT, see [LICENSE](LICENSE). Bundled fonts: Inter and JetBrains Mono, both SIL Open Font License 1.1 (`src/app/fonts`, licence text in `src/app/fonts/OFL.txt`). Models keep their own licences and are downloaded from Hugging Face rather than vendored: Apache-2.0 for the Lite and Mid repos, MIT for the Deep one. Each card in the picker names the repository it came from, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) has the full list.