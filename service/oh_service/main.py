"""Loopback-only inference service for the deep detector tier.

The deep model is a 0.6B language-model classifier, which needs torch. It is the
only model that does, so it runs here while the two encoder tiers run through
onnxruntime in the Next.js process. Both share one container and one volume.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = Path(os.environ.get("OH_DATA_DIR", "/data")).resolve()
MODELS_DIR = DATA_DIR / "models"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class ScoreRequest(BaseModel):
    model: str
    texts: list[str] = Field(min_length=1, max_length=512)


@dataclass
class LoadedModel:
    model_id: str
    manifest: dict[str, Any]
    tokenizer: Any
    model: Any
    load_ms: int
    threads: int = field(default=0)

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
        else:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()
        temperature = manifest.get("temperature")
        if temperature:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=-1)
        index = int(manifest.get("aiIndex", 1))
        return float(probs[min(index, probs.shape[-1] - 1)])


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