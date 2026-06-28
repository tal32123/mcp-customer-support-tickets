# syntax=docker/dockerfile:1.7
#
# Single image for all targets. Base ships CUDA torch + cuDNN; on a host
# without GPU (Railway, CI, CPU laptops) torch initialises lazily and runs
# CPU-only with no errors. On an NVIDIA host with the NVIDIA Container
# Toolkit, run with `--gpus all` to actually use the GPU.
#
# Data lives in Postgres (with pgvector). First boot ingests the HF dataset
# into the configured schema — see docker-compose.yml / railway.json.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# The base image ships torch 2.5.1+cu124 — same major version as the
# pyproject's CUDA pin — so uv pip install picks it up without
# re-downloading the wheel.
RUN uv pip install --system --no-cache .

# ---- runtime ----------------------------------------------------------------
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/opt/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /home/app --create-home app \
    && mkdir -p /opt/hf-cache \
    && chown -R app:app /opt/hf-cache

# Pytorch base uses conda; site-packages and console scripts live under /opt/conda.
COPY --from=builder /opt/conda/lib/python3.11/site-packages /opt/conda/lib/python3.11/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

# Pre-bake the embedding model into the image so cold starts don't hit HF.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
    && chown -R app:app /opt/hf-cache

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER app
WORKDIR /home/app

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["mcp-customer-support-tickets"]
