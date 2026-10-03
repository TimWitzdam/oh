"""Loopback-only inference service for the torch detector tiers.

The Balanced and Deep tiers are torch classifiers, so they run here while the
Lite tier runs through onnxruntime in the Next.js process. Both share one
container and one volume.

Two heads are supported. `variable-eos` models append an EOS token and read a
two-logit classifier off it. `mean-pool-logit` checkpoints ship no modelling
code at all - just a bare encoder under a `model.` prefix plus a
`classifier.weight [1, H]` tensor - so those weights are loaded by hand and the
score is a sigmoid over the single pooled logit.
"""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = Path(os.environ.get("OH_DATA_DIR", "/data")).resolve()
MODELS_DIR = DATA_DIR / "models"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class ScoreRequest(BaseModel):
    model: str
    # Must stay >= SCORE_BATCH in src/lib/detect/torch.ts, which chunks a long
    # document into requests of exactly that size. This is a guard against a
    # runaway body, not the limit a caller is expected to hit.
    texts: list[str] = Field(min_length=1, max_length=64)


@dataclass
class LoadedModel:
    model_id: str
    manifest: dict[str, Any]
    tokenizer: Any
    model: Any
    load_ms: int
    threads: int = field(default=0)
    head: tuple[Any, Any] | None = None

    def pooled_logit(self, inputs: dict[str, Any]) -> float:
        assert self.head is not None
        weight, bias = self.head
        hidden = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return float((pooled @ weight.T + bias)[0])

    def score(self, text: str) -> float:
        manifest = self.manifest
        max_tokens = int(manifest.get("maxTokens", 512))
        readout = manifest.get("readout")

        if readout == "variable-eos":
            # The variable-position models read their head off the EOS token that
            # follows the text, so the readout has to be appended by hand.
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens - 1,
            )["input_ids"]
            inputs = {
                "input_ids": torch.tensor([ids + [self.tokenizer.eos_token_id]]),
                "attention_mask": torch.ones(1, len(ids) + 1, dtype=torch.long),
            }
        elif readout == "mean-pool-logit":
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )
            with torch.inference_mode():
                return 1.0 / (1.0 + math.exp(-self.pooled_logit(inputs)))
        else:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()

        index = min(int(manifest.get("aiIndex", 1)), logits.shape[-1] - 1)
        temperature = manifest.get("temperature")
        if temperature:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=-1)
        return float(probs[index])


_loaded: dict[str, LoadedModel] = {}
_load_lock = asyncio.Lock()
_score_lock = asyncio.Lock()


def manifest_for(model_id: str) -> dict[str, Any]:
    safe = "".join(ch for ch in model_id if ch.isalnum() or ch in "-_.")
    path = MODELS_DIR / safe / "oh.json"
    if not path.is_file():
        raise FileNotFoundError(f"model {model_id} is not installed")
    return json.loads(path.read_text())


def load_sync(model_id: str) -> LoadedModel:
    existing = _loaded.get(model_id)
    if existing is not None:
        return existing

    manifest = manifest_for(model_id)
    directory = MODELS_DIR / model_id
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    head = None

    if manifest.get("readout") == "mean-pool-logit":
        # The checkpoint is a bare deberta-v3-large encoder plus one linear
        # layer. from_pretrained would report every encoder tensor as missing and
        # hand back a randomly initialised model, so the weights are mapped by
        # hand and a gap is an error rather than a warning.
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        config.architectures = ["DebertaV2Model"]
        # Built from the config rather than from_pretrained: the checkpoint keys
        # do not match this class, so the loader would read 1.7 GB, report every
        # tensor as missing and hand back random weights that the load_state_dict
        # below then overwrites.
        model = AutoModel.from_config(config)
        blob = load_file(directory / "model.safetensors")
        head = (blob["classifier.weight"].float(), blob["classifier.bias"].float())
        encoder_state = {
            key[len("model."):]: value for key, value in blob.items() if key.startswith("model.")
        }
        missing, _ = model.load_state_dict(encoder_state, strict=False)
        if missing:
            raise RuntimeError(f"{model_id}: encoder is missing {len(missing)} weights")
        del blob, encoder_state
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            directory, dtype=torch.float32, local_files_only=True
        )

    model.eval()
    if manifest.get("quantize"):
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    loaded = LoadedModel(
        model_id=model_id,
        manifest=manifest,
        tokenizer=tokenizer,
        model=model,
        load_ms=int((time.perf_counter() - started) * 1000),
        head=head,
    )
    _loaded[model_id] = loaded
    return loaded


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "loaded": sorted(_loaded),
    }


@app.post("/load")
async def load(request: Request) -> JSONResponse:
    body = await request.json()
    model_id = str(body.get("model", ""))
    async with _load_lock:
        try:
            loaded = await asyncio.to_thread(load_sync, model_id)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a message
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    return {
        "model": model_id,
        "loadMs": loaded.load_ms,
        "cached": True,
    }


@app.post("/unload")
async def unload(request: Request) -> JSONResponse:
    body = await request.json()
    model_id = str(body.get("model", ""))
    # Scoring streams run under the lock for their whole duration; dropping the
    # weights midway would fault the batch that is still reading them.
    if _score_lock.locked():
        return JSONResponse({"error": "a scoring run is in progress"}, status_code=409)
    async with _load_lock:
        released = _loaded.pop(model_id, None)
        if released is None:
            return JSONResponse({"error": f"model {model_id} is not loaded"}, status_code=404)
        del released
        gc.collect()
    return {
        "model": model_id,
        "loaded": sorted(_loaded),
    }


@app.post("/score")
async def score(request: Request, body: ScoreRequest) -> StreamingResponse:
    try:
        loaded = await asyncio.to_thread(load_sync, body.model)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    async def stream():
        async with _score_lock:
            for index, text in enumerate(body.texts):
                if await request.is_disconnected():
                    return
                try:
                    ai = await asyncio.to_thread(loaded.score, text)
                except Exception as exc:  # noqa: BLE001
                    yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"
                    return
                yield json.dumps({"index": index, "ai": round(ai, 6)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")