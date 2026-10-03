# syntax=docker/dockerfile:1

# ------------------------------------------------------------------- builder ---
FROM node:22-bookworm-slim AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts
COPY . .
# public/ is empty and therefore absent from a fresh git checkout, so make sure
# the runtime stage always has something to copy.
RUN mkdir -p public \
 && npm run build

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
 && /srv/venv/bin/pip install -r requirements.txt

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

COPY --from=python-deps /srv/venv /srv/venv
COPY service/oh_service /srv/oh_service
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The standalone output already carries the traced copy of everything the server
# requires (next, react, @huggingface/transformers, onnxruntime-node), so no
# separate node_modules copy is needed here.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

RUN mkdir -p /data/models \
 && useradd --system --create-home --uid 10001 oh \
 && chown -R oh:oh /data /app
USER oh
VOLUME ["/data"]
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]