"""Loopback-only inference service for the torch detector tiers.

The Balanced and Deep tiers are torch classifiers, so they run here while the
Lite tier runs through onnxruntime in the Next.js process. Both share one
container and one volume.

Two heads are supported. `variable-eos` models append an EOS token and read a
two-logit classifier off it. `mean-pool-logit` checkpoints ship no modelling
code at all - just a bare encoder under a `model.` prefix plus a
`classifier.weight [1, H]` tensor - so those weights are loaded by hand and the
score is a sigmoid over the single pooled logit.

Loaded weights stay resident so a second document is instant. Left alone that
means every tier a user has ever picked is still allocated hours later, so an
idle sweep drops whatever has not been asked for in a while. See IDLE_SECONDS.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import math
import os
import time
from contextlib import asynccontextmanager, suppress
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

DEFAULT_IDLE_MINUTES = 60.0
SWEEP_SECONDS = 60.0


def idle_seconds_from_env() -> float:
    """How long a model may go unused before its weights are dropped.

    Anything unparseable falls back to the default rather than to "never", so a
    typo in the compose file cannot quietly pin 2.7 GB resident for the life of
    the container. `off` (or any non-positive number) is the deliberate opt-out.
    """
    raw = os.environ.get("OH_MODEL_IDLE_MINUTES", "").strip().lower()
    if raw in ("off", "never", "false", "no"):
        return 0.0
    if not raw:
        return DEFAULT_IDLE_MINUTES * 60
    try:
        minutes = float(raw)
    except ValueError:
        return DEFAULT_IDLE_MINUTES * 60
    return minutes * 60 if minutes > 0 else 0.0


IDLE_SECONDS = idle_seconds_from_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Runs the idle sweep for as long as the service is up.

    One task, started once at startup rather than per request, so an install
    that stops being used still gives its RAM back.
    """
    task = asyncio.create_task(reaper_loop()) if IDLE_SECONDS > 0 else None
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


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
    # Monotonic, because this only ever feeds a comparison against another
    # monotonic reading. A wall clock that steps backwards mid-eviction would
    # otherwise make a just-used model look like it had been idle for hours.
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()

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
# Models being loaded or streamed right now. Held in module scope rather than on
# LoadedModel because a model can be busy before it finishes loading, and the
# window where it is absent from _loaded is exactly when the sweep must not
# decide anything about it.
_busy: set[str] = set()
_load_lock = asyncio.Lock()
_score_lock = asyncio.Lock()


class InUse:
    """Marks a model as in use for as long as a request is touching it."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def __enter__(self) -> "InUse":
        _busy.add(self.model_id)
        return self

    def __exit__(self, *_exc: object) -> None:
        _busy.discard(self.model_id)


def trim_memory() -> None:
    """Hands freed weights back to the OS instead of to the allocator.

    Dropping the reference is only half of it. Torch's CPU tensors are large
    posix_memalign blocks, which glibc mmaps and does return on free, but enough
    of a 1.7 GB model ends up back in the arena free lists that RSS stays pinned
    at the high-water mark. malloc_trim is what turns "the sweep ran" into "the
    RAM came back"; without it the sweep is close to invisible.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # not glibc - gc.collect above is all that is on offer


async def reap_once() -> None:
    """Drops every model that has been idle past the timeout."""
    now = time.monotonic()
    async with _load_lock:
        # Busy models are skipped rather than refused: the running request keeps
        # its own reference, so dropping one would not break it, it would just
        # make the next request read the weights back off disk for nothing.
        victims = [
            model_id
            for model_id, loaded in _loaded.items()
            if model_id not in _busy and now - loaded.last_used > IDLE_SECONDS
        ]
        for model_id in victims:
            del _loaded[model_id]

    if not victims:
        return
    print(f"oh idle sweep dropped: {', '.join(sorted(victims))}", flush=True)
    await asyncio.to_thread(trim_memory)


async def reaper_loop() -> None:
    while True:
        await asyncio.sleep(SWEEP_SECONDS)
        try:
            await reap_once()
        except Exception as exc:  # noqa: BLE001 - a bad sweep must not end the service
            print(f"oh idle sweep failed: {type(exc).__name__}: {exc}", flush=True)


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
        # 0 means the sweep is off, which is worth being able to tell apart from
        # "nothing has been swept yet".
        "idleMinutes": IDLE_SECONDS / 60,
    }


@app.post("/load")
async def load(request: Request) -> JSONResponse:
    body = await request.json()
    model_id = str(body.get("model", ""))
    async with _load_lock:
        try:
            with InUse(model_id):
                loaded = await asyncio.to_thread(load_sync, model_id)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a message
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
        # Stamped inside the lock, so the sweep cannot read a timestamp from
        # before this load and decide the fresh weights are already idle.
        loaded.touch()
    return {
        "model": model_id,
        "loadMs": loaded.load_ms,
        "cached": True,
    }


@app.post("/score")
async def score(request: Request, body: ScoreRequest) -> StreamingResponse:
    try:
        # Under the same lock as /load. Without it, a score landing while the UI
        # warms the model loads a second copy of the same weights, and two
        # resident copies of a 1.7 GB model is exactly what this sweep exists
        # to stop happening.
        async with _load_lock:
            with InUse(body.model):
                loaded = await asyncio.to_thread(load_sync, body.model)
            loaded.touch()
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    async def stream():
        async with _score_lock:
            # Busy for the length of the stream rather than just the load: a
            # long document runs for minutes, and an eviction landing mid-stream
            # would send the next batch back to disk.
            with InUse(body.model):
                for index, text in enumerate(body.texts):
                    if await request.is_disconnected():
                        return
                    try:
                        ai = await asyncio.to_thread(loaded.score, text)
                    except Exception as exc:  # noqa: BLE001
                        yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"
                        return
                    loaded.touch()
                    yield json.dumps({"index": index, "ai": round(ai, 6)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")