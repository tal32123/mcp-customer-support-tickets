# syntax=docker/dockerfile:1.7
#
# Single image for all targets. Base ships CUDA torch + cuDNN; on a host
# without GPU (Railway, CI, CPU laptops) torch initialises lazily and runs
# CPU-only with no errors. On an NVIDIA host with the NVIDIA Container
# Toolkit, run with `--gpus all` to actually use the GPU.
#
# Trade vs a slim CPU base: image is ~5-6 GB instead of ~3.5 GB. Railway's
# Hobby tier allows 100 GB images so this fits comfortably; the upside is
# one Dockerfile to maintain and GPU acceleration on dev machines that have
# it.

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

# Torch is already in the base image and satisfies torch>=2.5; uv pip
# install picks it up without re-downloading. No CPU-torch override here.
RUN uv pip install --system --no-cache .

# ---- runtime ----------------------------------------------------------------
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    PORT=8000 \
    MCP_CST_CACHE_DIR=/data \
    HF_HOME=/opt/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /home/app --create-home app \
    && mkdir -p /data /opt/hf-cache /opt/store-seed \
    && chown -R app:app /data /opt/hf-cache /opt/store-seed

# Pytorch base uses conda; site-packages and console scripts live under /opt/conda.
COPY --from=builder /opt/conda/lib/python3.11/site-packages /opt/conda/lib/python3.11/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

# Pre-bake the embedding model into the image so cold starts don't hit HF.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
    && chown -R app:app /opt/hf-cache

# Pre-bake the LanceDB store at /opt/store-seed. On a CPU builder (Railway,
# CI, Windows Docker Desktop without GPU passthrough) this is the slow step:
# ~3-5 min on Linux, ~30 min on Docker Desktop on Windows. On a GPU builder
# it drops to ~3 min. BuildKit caches this layer keyed on ingest code +
# dataset revision, so repeat builds reuse it.
RUN MCP_CST_CACHE_DIR=/opt/store-seed python -c "from mcp_cst.server import _init; _init()" \
    && chown -R app:app /opt/store-seed

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER app
WORKDIR /home/app

EXPOSE 8000
VOLUME ["/data"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["mcp-customer-support-tickets"]
