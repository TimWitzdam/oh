# syntax=docker/dockerfile:1

# ------------------------------------------------------------------- builder ---
FROM node:22-bookworm-slim AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts
COPY . .
# public/ only holds the logo, so guard the runtime stage's `COPY ./public`
# against it ever being dropped again.
RUN mkdir -p public \
 && npm run build \
 # next traces these two whole (see outputFileTracingIncludes in next.config.ts),
 # which drags in the win32 and darwin onnxruntime binaries and the wasm build of
 # sharp. None of them can be loaded on linux-x64, so drop them before the
 # standalone tree is copied into the runtime stage.
 && find .next/standalone -type d \( -path '*/onnxruntime-node/bin/napi-v6/win32' \
      -o -path '*/onnxruntime-node/bin/napi-v6/darwin' -o -name 'sharp-wasm32' \) \
      -prune -exec rm -rf {} +

# ------------------------------------------------------------------- python ---
# Built on the same Debian base as the runtime stage so the virtualenv links
# against an interpreter that is present in the final image.
FROM node:22-bookworm-slim AS python-deps
WORKDIR /srv
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY service/requirements.txt ./requirements.txt
RUN python3 -m venv /srv/venv \
 && /srv/venv/bin/pip install --upgrade pip \
  && /srv/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
  && /srv/venv/bin/pip install -r requirements.txt \
 # The torch wheel is built for every use case, so it carries C++ headers and a
 # test suite. Neither is reachable at runtime. torch/bin stays: it holds
 # torch_shm_manager, which torch/__init__.py looks for while importing. The
 # paths are spelled out one by one because /bin/sh is dash, which has no brace
 # expansion, and the glob has to stay unquoted to expand.
 && for dead in test include; do rm -rf /srv/venv/lib/python*/site-packages/torch/$dead; done \
 # pip and setuptools only exist to build the venv, which is already done.
 && for dead in pip setuptools wheel; do rm -rf /srv/venv/lib/python*/site-packages/$dead; done \
 && find /srv/venv -type d -name __pycache__ -prune -exec rm -rf {} +

# ------------------------------------------------------------------- runtime ---
FROM node:22-bookworm-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    OH_DATA_DIR=/data \
    OH_INFERENCE_URL=http://127.0.0.1:8001 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Created before the copies below so they can hand over ownership as they go.
# A chown afterwards would restate every copied byte in a fresh layer.
RUN useradd --system --create-home --uid 10001 oh

COPY --from=python-deps --chown=oh:oh /srv/venv /srv/venv
COPY --chown=oh:oh service/oh_service /srv/oh_service
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The standalone output already carries the traced copy of everything the server
# requires (next, react, @huggingface/transformers, onnxruntime-node), so no
# separate node_modules copy is needed here.
COPY --from=builder --chown=oh:oh /app/.next/standalone ./
COPY --from=builder --chown=oh:oh /app/.next/static ./.next/static
COPY --from=builder --chown=oh:oh /app/public ./public

RUN mkdir -p /data/models \
 && chown -R oh:oh /data
USER oh
VOLUME ["/data"]
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]