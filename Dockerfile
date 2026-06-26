# syntax=docker/dockerfile:1.7

# ---- builder ----------------------------------------------------------------
# Resolves deps and installs CPU-only torch up front, then the rest of the
# project on top. Doing torch first avoids pulling the ~2 GB CUDA wheels the
# default index would otherwise resolve to on Linux.
FROM python:3.11-slim AS builder

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

RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        torch
RUN uv pip install --system --no-cache .

# ---- runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    PORT=8000 \
    MCP_CST_CACHE_DIR=/data \
    HF_HOME=/opt/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /home/app --create-home app \
    && mkdir -p /data /opt/hf-cache \
    && chown -R app:app /data /opt/hf-cache

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pre-bake the embedding model into the image so cold starts don't hit HF.
# The model id mirrors config.EMBEDDING_MODEL; bump both together.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
    && chown -R app:app /opt/hf-cache

USER app
WORKDIR /home/app

EXPOSE 8000
VOLUME ["/data"]

CMD ["mcp-customer-support-tickets"]
